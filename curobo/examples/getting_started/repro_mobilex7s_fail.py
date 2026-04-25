"""Repro for MagicSim mobile_x7s fixed-base MotionGen failure.

Reproduces the failing solve without Isaac Sim / PlannerManager / Service
layer — direct cuRobo ``BatchMotionPlanner.plan_pose`` with the same
(robot YAML, targets, current state, empty scene) that MagicSim logs
showed failing:

    Submit IK / MG target:
        slot 0 = [0.5, -0.25, 0.9, 1, 0, 0, 0]
        slot 1 = [0.5, +0.25, 0.9, 1, 0, 0, 0]
    current_state:
        all zeros, 18 DOF (fixed base = no virtual base joints)
    scene:
        empty
    MG result:
        pos_err_max=0.0009 m, rot_err_max=0.0020 rad — WELL inside both
        thresholds (0.005, 0.05) — yet ``result.success == False``.

If success is False despite errors being below threshold, trajopt is
failing for a different reason (self-collision along the trajectory,
joint-limit violation, or arms genuinely can't move to those poses from
retract because the tool-frame order in the YAML inverts left/right).

This repro prints enough detail to see which of those is the root cause.

Run::

    uv run python -m curobo.examples.getting_started.repro_mobilex7s_fail
"""

from __future__ import annotations

import torch

from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose


ROBOT_YAML = "magicsim_mobile_x7s.yml"  # fixed-base variant
B = 1                                     # one problem


def main() -> None:
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

    # --- Build the planner exactly like DualMotionGenServer's LOCKED solver.
    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YAML,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],    # empty per-env scenes
        max_batch_size=B,
        multi_env=True,
        max_goalset=1,
        collision_cache={"cuboid": 0, "mesh": 500},
        num_trajopt_seeds=4,
        num_ik_seeds=32,
        use_cuda_graph=False,
        self_collision_check=True,
        optimizer_collision_activation_distance=0.025,
    )
    planner = BatchMotionPlanner(cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=3)

    tool_frames = list(planner.tool_frames)
    print(f"\n[repro] tool_frames (YAML order) = {tool_frames}")
    print(f"[repro] robot joint_names = {list(planner.kinematics.joint_names)}")
    print(f"[repro] dof = {len(planner.kinematics.joint_names)}\n")

    # --- Targets exactly as MagicSim submitted them (slot 0 / slot 1 order).
    targets_slot0_slot1 = torch.tensor([
        [0.5, -0.25, 0.9,  1.0, 0.0, 0.0, 0.0],   # slot 0 — intended "right" by caller
        [0.5,  0.25, 0.9,  1.0, 0.0, 0.0, 0.0],   # slot 1 — intended "left" by caller
    ], device=device_cfg.device, dtype=device_cfg.dtype)

    # Test both orderings so we can see if tool-frame order in the YAML
    # inverts the intended assignment.
    for label, tgt in (
        ("AS_SUBMITTED (slot 0 = -y, slot 1 = +y)", targets_slot0_slot1),
        ("SWAPPED     (slot 0 = +y, slot 1 = -y)",
         torch.stack([targets_slot0_slot1[1], targets_slot0_slot1[0]], dim=0)),
    ):
        print(f"\n========== {label} ==========")
        print(f"  tool_frames[0]={tool_frames[0]} ← target {tgt[0].tolist()}")
        print(f"  tool_frames[1]={tool_frames[1]} ← target {tgt[1].tolist()}")

        # (B=1, L=2, 7) → per-frame Pose on the solver device.
        pose_dict = {}
        for li, frame in enumerate(tool_frames):
            pose_dict[frame] = Pose(
                position=tgt[li, :3].view(1, 3).contiguous(),
                quaternion=tgt[li, 3:].view(1, 4).contiguous(),
            )
        # info-only frames (extra_fk_link) get current-FK filler per §8.1 of
        # the MagicSim Services README; reproducing that here too.
        dof = len(planner.kinematics.joint_names)
        zero_js = JointState.from_position(
            torch.zeros(B, dof, device=device_cfg.device, dtype=device_cfg.dtype),
            joint_names=list(planner.kinematics.joint_names),
        )
        fk_kin = planner.compute_kinematics(zero_js)
        for frame in planner.tool_frames:
            if frame not in pose_dict:
                pose_dict[frame] = fk_kin.tool_poses.get_link_pose(
                    frame, make_contiguous=True,
                )

        goal = GoalToolPose.from_poses(
            pose_dict,
            ordered_tool_frames=list(planner.tool_frames),
            num_goalset=1,
        )

        # Empty scene on every slot.
        for slot in range(B):
            planner.scene_collision_checker.load_collision_model(
                Scene(), env_idx=slot,
            )

        result = planner.plan_pose(
            goal, zero_js,
            max_attempts=2,
            enable_graph_attempt=0,
        )

        if result is None:
            print("  RESULT: None (no IK seed survived)")
            continue

        success = result.success
        pos_err = (
            float(result.position_error.max().item())
            if result.position_error is not None else float("nan")
        )
        rot_err = (
            float(result.rotation_error.max().item())
            if result.rotation_error is not None else float("nan")
        )
        print(f"  success={success.tolist()}")
        print(f"  pos_err_max={pos_err:.4f}m  rot_err_max={rot_err:.4f}rad")
        print(f"  thresholds (for reference): pos<=0.005m rot<=0.05rad")

        traj = getattr(result, "interpolated_trajectory", None)
        if traj is not None and getattr(traj, "position", None) is not None:
            t_pos = traj.position
            print(f"  interpolated_trajectory.position shape = {tuple(t_pos.shape)}")
            n_nan = int(torch.isnan(t_pos).sum().item())
            n_inf = int(torch.isinf(t_pos).sum().item())
            print(f"  interp traj: nan_count={n_nan} inf_count={n_inf}")
        else:
            print("  interpolated_trajectory = None")

        # Status codes exposed by curobo v2, when available:
        for attr in ("status", "optimized_dt", "collision_free",
                     "valid_query", "ik_success"):
            val = getattr(result, attr, "<not set>")
            if val != "<not set>":
                print(f"  result.{attr} = {val}")

    # --- Now also try IK-only to confirm the IK-success-but-MG-fail pattern.
    print("\n========== IK-ONLY (AS_SUBMITTED) ==========")
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    ik_cfg = InverseKinematicsCfg.create(
        robot=ROBOT_YAML,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
        num_seeds=32,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=True,
        use_cuda_graph=False,
        collision_cache={"cuboid": 0, "mesh": 500},
        max_batch_size=B,
        multi_env=True,
        max_goalset=1,
    )
    ik = InverseKinematics(ik_cfg)
    pose_dict = {}
    for li, frame in enumerate(tool_frames):
        pose_dict[frame] = Pose(
            position=targets_slot0_slot1[li, :3].view(1, 3).contiguous(),
            quaternion=targets_slot0_slot1[li, 3:].view(1, 4).contiguous(),
        )
    for frame in ik.tool_frames:
        if frame not in pose_dict:
            pose_dict[frame] = fk_kin.tool_poses.get_link_pose(
                frame, make_contiguous=True,
            )
    ik_goal = GoalToolPose.from_poses(
        pose_dict, ordered_tool_frames=list(ik.tool_frames), num_goalset=1,
    )
    for slot in range(B):
        ik.scene_collision_checker.load_collision_model(Scene(), env_idx=slot)
    ik_result = ik.solve_pose(ik_goal, current_state=zero_js)
    print(f"  success={ik_result.success.tolist()}")
    print(f"  pos_err={float(ik_result.position_error.max().item()):.4f}m "
          f"rot_err={float(ik_result.rotation_error.max().item()):.4f}rad")
    if ik_result.js_solution is not None:
        sol = ik_result.js_solution.position
        if sol.dim() == 3:
            sol = sol[:, 0, :]
        print(f"  IK solution joint positions: {sol[0].tolist()}")


if __name__ == "__main__":
    main()
