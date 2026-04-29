"""Offline reproducer for "MG plans a big-detour trajectory" on Vega1pSharpa.

Standalone — no IsaacSim, no MagicSim. Just cuRobo + the magicsim_vega1p_sharpa
yamls. Reproduces the exact scenario the dex-grasp pipeline sees and dumps
**every** intermediate so you can see WHY trajopt picks a long path:

  1. parked Vega base at world (-1, 0, 0.1)
  2. table (1.118 × 2.413 × 1.0) at world origin (matches mobile_dex_grasp_env)
  3. R_ee target = a sample bottle-grasp pose just above the table edge
     at (-0.30, -0.05, 1.18); L_ee left disabled
  4. plan_pose → MotionPlanner → IK + TrajOpt
  5. Dump:
     - retract / current state used as IK seed
     - IK solver's top-k solutions, position_error per seed
     - which seed trajopt picked
     - trajectory waypoint deltas (joint-by-joint cumulative motion)
     - first / last 5 waypoints' joint values
     - "reach distance" of R_ee (FK at start vs FK at goal)

Run::

    python -m curobo.examples.getting_started.repro_vega_mg_detour
    python -m curobo.examples.getting_started.repro_vega_mg_detour --mobile
    python -m curobo.examples.getting_started.repro_vega_mg_detour --no-anchor

The ``--no-anchor`` flag temporarily disables the MagicSim seed-anchor
patch (set ``_MAGICSIM_IK_SEED_NOISE_STD = 1e6`` so the noise dominates,
making seeds effectively random). Useful for A/B against the patched
behaviour to confirm whether the detour is coming from the IK seed
distribution or somewhere else (trajopt / null space / etc.).
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import torch

from curobo._src.geom.types import Cuboid
from curobo._src.motion import motion_planner as mp_module
from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose


# ----------------------------------------------------------------------
# Scenario constants — keep in sync with MagicSim's
# ``mobile_dex_grasp_env.yaml`` so the offline result is comparable.
# ----------------------------------------------------------------------
TABLE_SIZE = (1.118, 2.413, 1.0)   # x, y, z extents (m)
TABLE_CENTER = (0.0, 0.0, 0.5)     # cube center → top at z=1.0

# Sample R_ee target — borrowed from a typical bottle-edge grasp pose
# the dex-grasp scene generates. xyz roughly above the +X edge of the
# table top, quaternion roughly "palm down, fingers toward -y".
R_EE_TARGET_POS = (-0.30, -0.05, 1.18)
R_EE_TARGET_QUAT = (0.0, 1.0, 0.0, 0.0)  # wxyz, palm-down

# Vega base parked at world (-1, 0, 0.1) per single_vega1pSharpa.yaml.
BASE_POS_WORLD = (-1.0, 0.0, 0.1)


def _print_joint_deltas(traj: torch.Tensor, joint_names: List[str]) -> None:
    """Print per-joint cumulative |Δq| over the trajectory + max single-step."""
    # traj: (T, dof)
    if traj.shape[0] < 2:
        print(f"  ⚠ trajectory has only {traj.shape[0]} waypoint(s); cannot diff.")
        if traj.shape[0] == 1:
            print(f"    waypoint[0] = {[f'{v:+.3f}' for v in traj[0].tolist()]}")
        return
    diffs = (traj[1:] - traj[:-1]).abs()
    cum = diffs.sum(dim=0)
    mx = diffs.max(dim=0).values
    rng = traj.max(dim=0).values - traj.min(dim=0).values
    print(f"  {'joint':<32} {'cum |Δq|':>10} {'max |Δq| step':>14} {'range':>10}")
    for n, c, m, r in zip(joint_names, cum.tolist(), mx.tolist(), rng.tolist()):
        flag = "  ←DETOUR" if c > 1.5 or r > 1.0 else ""
        print(f"  {n:<32} {c:>10.4f} {m:>14.4f} {r:>10.4f}{flag}")
    print(f"  total cumulative joint motion: {cum.sum().item():.4f} rad")


def _print_pose_summary(label: str, p: Pose) -> None:
    pos = p.position.flatten().tolist()
    quat = p.quaternion.flatten().tolist()
    print(
        f"  {label}: pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"quat=({quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f})"
    )


def _build_planner(yaml_name: str, with_table: bool) -> MotionPlanner:
    raw = load_yaml(join_path(get_robot_configs_path(), yaml_name))
    scene = [{}]
    if with_table:
        # Table as a single cuboid obstacle. Match
        # ``mobile_dex_grasp_env.yaml``'s simple_desk cube.
        scene = [
            {
                "table": Cuboid(
                    name="table",
                    pose=[
                        TABLE_CENTER[0],
                        TABLE_CENTER[1],
                        TABLE_CENTER[2],
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dims=list(TABLE_SIZE),
                )
            }
        ]
    cfg = MotionPlannerCfg.create(
        robot=raw["robot_cfg"],
        scene_model=scene,
        device_cfg=DeviceCfg(device="cuda:0", dtype=torch.float32),
        self_collision_check=True,
        max_batch_size=1,
        multi_env=True,
        max_goalset=1,
        collision_cache={"cuboid": 10, "mesh": 500},
        num_trajopt_seeds=4,
        num_ik_seeds=32,
        use_cuda_graph=False,
        optimizer_collision_activation_distance=0.025,
    )
    p = MotionPlanner(cfg)
    p.warmup(enable_graph=False, num_warmup_iterations=1)
    return p


def _build_current_state(planner: MotionPlanner, mobile: bool) -> JointState:
    """Live-ish joint state: USD retract for torso / head / arms; for
    the mobile variant, also stick the dummy_base joints at the
    parked-base position so cuRobo's free-base solver sees the actual
    starting world pose."""
    q = planner.default_joint_state.position.clone()
    if mobile:
        names = list(planner.joint_names)
        for k, v in (
            ("dummy_base_prismatic_x_joint", BASE_POS_WORLD[0]),
            ("dummy_base_prismatic_y_joint", BASE_POS_WORLD[1]),
            ("dummy_base_revolute_z_joint", 0.0),
        ):
            if k in names:
                q[names.index(k)] = float(v)
    return JointState.from_position(q.unsqueeze(0), joint_names=planner.joint_names)


def _build_goal(planner: MotionPlanner) -> GoalToolPose:
    """R_ee tracks the bottle-grasp target; L_ee filled with current FK
    so it's not a NaN target (cuRobo would otherwise want a real
    track / disable criteria pair, which the MotionPlanner standalone
    API doesn't expose)."""
    pose_dict = {}
    for f in planner.tool_frames:
        if f == "R_ee":
            pose_dict[f] = Pose(
                position=torch.tensor(
                    [list(R_EE_TARGET_POS)], device="cuda:0", dtype=torch.float32,
                ),
                quaternion=torch.tensor(
                    [list(R_EE_TARGET_QUAT)], device="cuda:0", dtype=torch.float32,
                ),
            )
        else:
            # Compute current FK at the planner's default state for the
            # other tool frames (L_ee + extra info-only frames). Keeps
            # them tracked at a feasible pose so they don't fail the
            # solver.
            q0 = planner.default_joint_state.position.unsqueeze(0)
            kin = planner.compute_kinematics(
                JointState.from_position(q0, joint_names=planner.joint_names),
            )
            p = kin.tool_poses.get_link_pose(f, make_contiguous=True)
            pose_dict[f] = Pose(
                position=p.position.view(-1, 3).contiguous(),
                quaternion=p.quaternion.view(-1, 4).contiguous(),
            )
    return GoalToolPose.from_poses(
        pose_dict,
        ordered_tool_frames=list(planner.tool_frames),
        num_goalset=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mobile",
        action="store_true",
        help="Use ``magicsim_vega1p_sharpa_mobile.yml`` (free base, 23 DoF).",
    )
    ap.add_argument(
        "--no-anchor",
        action="store_true",
        help="Disable the MagicSim seed-anchor patch (sets noise std huge).",
    )
    ap.add_argument(
        "--no-table",
        action="store_true",
        help="Plan in empty scene to isolate IK/trajopt from collision avoidance.",
    )
    args = ap.parse_args()

    if args.no_anchor:
        # Bump noise so seeds are effectively random again — A/B test.
        mp_module._MAGICSIM_IK_SEED_NOISE_STD = 5.0

    yaml_name = (
        "magicsim_vega1p_sharpa_mobile.yml"
        if args.mobile
        else "magicsim_vega1p_sharpa.yml"
    )
    print(f"[setup] yaml={yaml_name}")
    print(
        f"[setup] anchor noise std = {mp_module._MAGICSIM_IK_SEED_NOISE_STD} rad "
        f"({'PATCHED' if not args.no_anchor else 'BYPASSED'})"
    )
    print(f"[setup] scene = {'empty' if args.no_table else 'desk cube'}")

    planner = _build_planner(yaml_name, with_table=not args.no_table)
    print(f"[setup] dof={len(planner.joint_names)}  joint_names={planner.joint_names}")

    cs = _build_current_state(planner, mobile=args.mobile)
    print(
        "[setup] current_state position:",
        [f"{v:+.4f}" for v in cs.position.flatten().tolist()],
    )

    # FK at start
    kin = planner.compute_kinematics(cs)
    print("[FK at current_state]:")
    for f in planner.tool_frames:
        _print_pose_summary(f, kin.tool_poses.get_link_pose(f, make_contiguous=True))

    # Goal
    goal = _build_goal(planner)
    print("[goal]:")
    for f in planner.tool_frames:
        _print_pose_summary(
            f,
            Pose(
                position=goal.position[:, 0, planner.tool_frames.index(f), :],
                quaternion=goal.quaternion[:, 0, planner.tool_frames.index(f), :],
            ),
        )

    # IK alone — see what configs the solver finds.
    print("\n========== IK (top-k seeds) ==========")
    ik_seed_n = max(
        planner.trajopt_solver.config.num_seeds,
        int(planner.ik_solver.config.num_seeds),
    )
    ik_seed_cfg = mp_module._anchor_ik_seeds_to_current_state(cs, ik_seed_n)
    ik_result = planner.ik_solver.solve_pose(
        goal,
        return_seeds=planner.trajopt_solver.config.num_seeds,
        current_state=cs,
        seed_config=ik_seed_cfg,
    )
    print(f"  IK seed_config shape: {tuple(ik_seed_cfg.shape)}")
    print(f"  IK return_seeds:      {planner.trajopt_solver.config.num_seeds}")
    print(f"  IK success per seed:  {ik_result.success.tolist()}")
    print(
        f"  IK pos_err (m) per seed: "
        f"{[f'{v:.4f}' for v in ik_result.position_error.flatten().tolist()]}"
    )
    # Joint distance from current_state for each IK solution
    sol = ik_result.solution.view(-1, ik_result.solution.shape[-1])
    cs_pos = cs.position.expand_as(sol)
    deltas = (sol - cs_pos).abs().sum(dim=-1)
    print(f"  IK |Δq|_1 from current per seed: "
          f"{[f'{v:.3f}' for v in deltas.tolist()]}")

    # Plan
    print("\n========== MG (plan_pose) ==========")
    result = planner.plan_pose(goal, cs)
    if result is None or not result.success.any():
        print(
            "  plan_pose FAILED  pos_err=",
            float(result.position_error.max()) if result is not None else "N/A",
        )
        return
    print(f"  ✓ plan_pose succeeded")
    print(f"  trajopt pos_err   = {float(result.position_error.max()):.6f} m")
    print(f"  trajopt rot_err   = {float(result.rotation_error.max()):.6f} rad")
    # Best-effort dump of all available result attrs.
    for attr in ("optimized_dt", "interpolation_dt", "solve_time", "total_time"):
        if hasattr(result, attr):
            print(f"  {attr:<17} = {getattr(result, attr)}")

    # Raw trajopt solution (knot points), pre-interpolation.
    raw = result.solution
    if hasattr(raw, "position"):
        raw = raw.position
    while raw.dim() > 2:
        raw = raw[0]
    print(f"  raw trajopt knots = {raw.shape[0]}")

    interp = result.get_interpolated_plan()
    traj = interp.position.squeeze(0)  # (T, dof) typically
    if traj.dim() == 3:
        traj = traj[0]
    print(f"  interp waypoints  = {traj.shape[0]}")
    print(
        f"  duration ≈ "
        f"{traj.shape[0] * planner.trajopt_solver.config.interpolation_dt:.2f}s"
    )

    # If interpolated traj is empty/single-step, fall back to raw knots.
    if traj.shape[0] < 2:
        print(
            "  ⚠ interpolated trajectory has < 2 waypoints; using raw knots instead."
        )
        traj = raw

    # Per-joint motion
    print("\n  ── per-joint trajectory motion ──")
    _print_joint_deltas(traj, list(planner.joint_names))

    # Sample first / last 5 waypoints
    print("\n  ── first 5 waypoints (joint pos) ──")
    for i in range(min(5, traj.shape[0])):
        print(f"    [{i:3d}] {[f'{v:+.3f}' for v in traj[i].tolist()]}")
    print("  ── last 5 waypoints ──")
    for i in range(max(0, traj.shape[0] - 5), traj.shape[0]):
        print(f"    [{i:3d}] {[f'{v:+.3f}' for v in traj[i].tolist()]}")

    # FK along trajectory — find max R_ee deviation from straight-line path.
    js_traj = JointState.from_position(traj, joint_names=planner.joint_names)
    kin_traj = planner.compute_kinematics(js_traj)
    r_ee_pos = kin_traj.tool_poses.get_link_pose("R_ee", make_contiguous=True).position
    # straight line from start to goal
    start_p = r_ee_pos[0]
    end_p = r_ee_pos[-1]
    print(f"\n  R_ee start FK: {start_p.tolist()}")
    print(f"  R_ee end FK:   {end_p.tolist()}")
    seg = end_p - start_p
    seg_len = seg.norm().clamp_min(1e-9)
    seg_dir = seg / seg_len
    devs: List[Tuple[int, float]] = []
    for i in range(r_ee_pos.shape[0]):
        rel = r_ee_pos[i] - start_p
        proj = (rel * seg_dir).sum() * seg_dir
        perp = rel - proj
        devs.append((i, float(perp.norm().item())))
    max_dev_i, max_dev = max(devs, key=lambda x: x[1])
    print(
        f"  R_ee max-perpendicular deviation from straight line: "
        f"{max_dev*100:.2f} cm at waypoint {max_dev_i}"
    )
    if max_dev > 0.05:
        print(
            "  ⚠ DETOUR: R_ee strays > 5 cm from a straight-line path. "
            "Likely a long IK→trajopt seed mismatch."
        )


if __name__ == "__main__":
    main()
