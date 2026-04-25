"""Repro for MagicSim Vega1pSharpa fixed-base MotionGen failure.

Same scaffold as ``repro_mobilex7s_fail.py`` but for Vega1pSharpa. The
mobile_x7s case turned out to be a tool_frames-order bug; vega has the
right order already (``[R_ee, L_ee]``) yet MG still fails:

    target slot 0 (R) = [0.5, -0.25, 0.9, 1, 0, 0, 0]
    target slot 1 (L) = [0.5, +0.25, 0.9, 1, 0, 0, 0]
    current_state    = all zeros, 20 DOF
    scene            = empty
    IK : success=True, pos_err=0.0078m, rot_err=0.0028rad
    MG : success=False, pos_err=0.0125m, rot_err=0.0035rad

IK pos_err is 1.6× the 0.005m threshold. MG pos_err is 2.5× the
threshold. So unlike mobile_x7s (where MG terminal error was tiny but
trajopt rejected the trajectory), here the solver actually can't drive
the EEFs to the target — the pose may be near or past the workspace
boundary, or it may be self-collision-bound.

Run::

    uv run python -m curobo.examples.getting_started.repro_vega_fail
"""

from __future__ import annotations

import torch

from copy import deepcopy

from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose


ROBOT_YAML = "magicsim_vega1p_sharpa.yml"  # fixed-base variant
B = 1


def _load_yaml_with_planner_manager_merge():
    """Load magicsim_vega1p_sharpa.yml exactly the way PlannerManager does:
    merge top-level ``extra_fk_link`` into ``robot_cfg.kinematics.tool_frames``
    (dedup, tracked first, then extras). Returns (merged_cfg, extras, info_links)."""
    raw = load_yaml(join_path(get_robot_configs_path(), ROBOT_YAML))
    cfg = deepcopy(raw["robot_cfg"])
    extras = list(raw.get("extra_fk_link") or [])
    info_links = raw.get("info_links")
    kin = cfg.setdefault("kinematics", {})
    tracked = list(kin.get("tool_frames") or [])
    merged = list(tracked)
    for f in extras:
        if f not in merged:
            merged.append(f)
    kin["tool_frames"] = merged
    return cfg, extras, info_links, tracked


def _build_targets(device, dtype):
    return torch.tensor([
        [0.5, -0.25, 0.9,  1.0, 0.0, 0.0, 0.0],   # slot 0 (R_ee)
        [0.5,  0.25, 0.9,  1.0, 0.0, 0.0, 0.0],   # slot 1 (L_ee)
    ], device=device, dtype=dtype)


def _build_pose_dict(planner_or_ik, targets, current_state):
    pose_dict = {}
    tool_frames = list(planner_or_ik.tool_frames)
    fk_kin = planner_or_ik.compute_kinematics(current_state)
    # tracked frames first
    n_tracked = min(targets.shape[0], len(tool_frames))
    for li in range(n_tracked):
        pose_dict[tool_frames[li]] = Pose(
            position=targets[li, :3].view(1, 3).contiguous(),
            quaternion=targets[li, 3:].view(1, 4).contiguous(),
        )
    # FK-filler for any remaining frames (extra_fk_link / extras).
    for frame in tool_frames:
        if frame not in pose_dict:
            pose_dict[frame] = fk_kin.tool_poses.get_link_pose(
                frame, make_contiguous=True,
            )
    return pose_dict, fk_kin


def main() -> None:
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

    # ----- IK first ----------------------------------------------------------
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
    print(f"\n[repro] tool_frames = {list(ik.tool_frames)}")
    print(f"[repro] joint_names = {list(ik.joint_names)}")
    print(f"[repro] dof = {len(ik.joint_names)}\n")

    targets = _build_targets(device_cfg.device, device_cfg.dtype)
    dof = len(ik.joint_names)
    zero_js = JointState.from_position(
        torch.zeros(B, dof, device=device_cfg.device, dtype=device_cfg.dtype),
        joint_names=list(ik.joint_names),
    )
    ik_pose_dict, fk_kin = _build_pose_dict(ik, targets, zero_js)

    # Print the FK pose of each tracked frame at the zero config — gives
    # us the "reachable from retract" baseline.
    print("===== Reachable poses at current_state = ZERO =====")
    for frame in ik.tool_frames:
        p = fk_kin.tool_poses.get_link_pose(frame, make_contiguous=True)
        pos = p.position.view(3).tolist()
        quat = p.quaternion.view(4).tolist()
        print(f"  FK[{frame}]  pos={pos}  quat={quat}")

    print("\n===== Targets =====")
    for li, frame in enumerate(ik.tool_frames):
        if li < targets.shape[0]:
            print(f"  target[{li}] -> {frame}: pos={targets[li, :3].tolist()} "
                  f"quat={targets[li, 3:].tolist()}")

    ik_goal = GoalToolPose.from_poses(
        ik_pose_dict, ordered_tool_frames=list(ik.tool_frames), num_goalset=1,
    )
    for slot in range(B):
        ik.scene_collision_checker.load_collision_model(Scene(), env_idx=slot)
    print("\n========== IK ==========")
    ik_result = ik.solve_pose(ik_goal, current_state=zero_js)
    pos_err = float(ik_result.position_error.max().item()) if ik_result.position_error is not None else float("nan")
    rot_err = float(ik_result.rotation_error.max().item()) if ik_result.rotation_error is not None else float("nan")
    print(f"  success={ik_result.success.tolist()}")
    print(f"  pos_err={pos_err:.4f}m rot_err={rot_err:.4f}rad")
    if ik_result.js_solution is not None:
        sol = ik_result.js_solution.position
        if sol.dim() == 3:
            sol = sol[:, 0, :]
        sol_list = sol[0].tolist()
        print(f"  IK solution joint positions ({len(sol_list)}-d): "
              f"[{', '.join(f'{v:+.3f}' for v in sol_list)}]")

        # Compute FK at the IK solution → see actual achieved EEF poses.
        # This tells us if 0.0078m error is residual gradient or
        # genuinely-unreachable target.
        sol_js = JointState.from_position(
            sol[:1, :dof], joint_names=list(ik.joint_names),
        )
        achieved_kin = ik.compute_kinematics(sol_js)
        print("\n  Achieved FK at IK solution (vs target):")
        for li in range(min(targets.shape[0], len(ik.tool_frames))):
            frame = ik.tool_frames[li]
            ach = achieved_kin.tool_poses.get_link_pose(
                frame, make_contiguous=True,
            )
            ach_pos = ach.position.view(3).tolist()
            tgt_pos = targets[li, :3].tolist()
            d = sum((a - b) ** 2 for a, b in zip(ach_pos, tgt_pos)) ** 0.5
            print(f"    {frame}:  achieved={[f'{v:+.4f}' for v in ach_pos]}  "
                  f"target={[f'{v:+.4f}' for v in tgt_pos]}  delta={d:.4f}m")

    # ----- MG ----------------------------------------------------------------
    print("\n========== MG (BatchMotionPlanner) ==========")
    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YAML,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
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

    mg_pose_dict, _ = _build_pose_dict(planner, targets, zero_js)
    mg_goal = GoalToolPose.from_poses(
        mg_pose_dict, ordered_tool_frames=list(planner.tool_frames), num_goalset=1,
    )
    for slot in range(B):
        planner.scene_collision_checker.load_collision_model(
            Scene(), env_idx=slot,
        )
    mg_result = planner.plan_pose(
        mg_goal, zero_js, max_attempts=2, enable_graph_attempt=0,
    )
    if mg_result is None:
        print("  RESULT: None")
        return
    pos_err = float(mg_result.position_error.max().item()) if mg_result.position_error is not None else float("nan")
    rot_err = float(mg_result.rotation_error.max().item()) if mg_result.rotation_error is not None else float("nan")
    print(f"  success={mg_result.success.tolist()}")
    print(f"  pos_err={pos_err:.4f}m rot_err={rot_err:.4f}rad")

    # ----- Sanity: try a target that's clearly inside the reachable set ------
    # Move the target back along x by 0.1m → should easily reach.
    print("\n========== MG SANITY: targets x=0.4 (closer to base) ==========")
    targets_close = targets.clone()
    targets_close[:, 0] = 0.40
    mg_pose_dict_c, _ = _build_pose_dict(planner, targets_close, zero_js)
    mg_goal_c = GoalToolPose.from_poses(
        mg_pose_dict_c, ordered_tool_frames=list(planner.tool_frames), num_goalset=1,
    )
    mg_result_c = planner.plan_pose(
        mg_goal_c, zero_js, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_c = float(mg_result_c.position_error.max().item()) if mg_result_c.position_error is not None else float("nan")
    rot_err_c = float(mg_result_c.rotation_error.max().item()) if mg_result_c.rotation_error is not None else float("nan")
    print(f"  success={mg_result_c.success.tolist()}")
    print(f"  pos_err={pos_err_c:.4f}m rot_err={rot_err_c:.4f}rad")

    # ----- Match what MagicSim ACTUALLY submitted to the solver --------------
    # User's [FAIL] log shows the target arrived at MG with z=0.79999...
    # not z=0.9 — that's the world→robot-frame transform subtracting the
    # robot's base_pos[z] (≈0.1m). Re-run with that as-solver-saw-it value.
    print("\n========== MG (target as solver saw it: z=0.7999...) ==========")
    targets_solver = targets.clone()
    targets_solver[:, 0] = 0.49999961   # match log precision
    targets_solver[:, 2] = 0.79999995
    mg_pose_dict_s, _ = _build_pose_dict(planner, targets_solver, zero_js)
    mg_goal_s = GoalToolPose.from_poses(
        mg_pose_dict_s, ordered_tool_frames=list(planner.tool_frames), num_goalset=1,
    )
    mg_result_s = planner.plan_pose(
        mg_goal_s, zero_js, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_s = float(mg_result_s.position_error.max().item()) if mg_result_s.position_error is not None else float("nan")
    rot_err_s = float(mg_result_s.rotation_error.max().item()) if mg_result_s.rotation_error is not None else float("nan")
    print(f"  success={mg_result_s.success.tolist()}")
    print(f"  pos_err={pos_err_s:.4f}m rot_err={rot_err_s:.4f}rad")

    # ----- KEY TEST: Server-side configuration (L=4 with extra_fk_link merged)
    # PlannerManager._merge_extra_fk_link mutates kinematics.tool_frames to
    # include FK-only extras. The Server builds the MotionPlanner with L=4
    # ([R_ee, L_ee, vega_1p_base, arm_center]) instead of L=2. cuRobo's IK
    # stage auto-retunes seed counts / iterations / tile sizes for L>2
    # (see solver_ik.py:132-138). That altered tuning may be the real
    # difference between this succeeding repro and the failing Server.
    print("\n========== MG with MERGED tool_frames (mimics Server exactly) ==========")
    merged_cfg, extras, info_links, tracked_orig = _load_yaml_with_planner_manager_merge()
    print(f"  tool_frames (pre-merge tracked) : {tracked_orig}")
    print(f"  tool_frames (post-merge L={len(merged_cfg['kinematics']['tool_frames'])}) : "
          f"{merged_cfg['kinematics']['tool_frames']}")

    cfg_merged = MotionPlannerCfg.create(
        robot=merged_cfg,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
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
    planner_m = BatchMotionPlanner(cfg_merged)
    planner_m.warmup(enable_graph=False, num_warmup_iterations=3)
    print(f"  planner_m.tool_frames = {list(planner_m.tool_frames)}")

    # Apply disabled() criteria to extras — mimics Server init.
    crit: dict = {}
    for frame in planner_m.tool_frames:
        if frame in extras:
            crit[frame] = ToolPoseCriteria.disabled()
        else:
            crit[frame] = ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.1, 0.1, 0.1],
            )
    planner_m.update_tool_pose_criteria(crit)

    # Use SAME targets as before (only tracked frames get real targets;
    # extras get FK filler just like the Server).
    zero_js_m = JointState.from_position(
        torch.zeros(B, dof, device=device_cfg.device, dtype=device_cfg.dtype),
        joint_names=list(planner_m.kinematics.joint_names),
    )
    fk_kin_m = planner_m.compute_kinematics(zero_js_m)
    pose_dict_m = {}
    for li, frame in enumerate(tracked_orig):
        pose_dict_m[frame] = Pose(
            position=targets[li, :3].view(1, 3).contiguous(),
            quaternion=targets[li, 3:].view(1, 4).contiguous(),
        )
    for frame in planner_m.tool_frames:
        if frame not in pose_dict_m:
            pose_dict_m[frame] = fk_kin_m.tool_poses.get_link_pose(
                frame, make_contiguous=True,
            )
    mg_goal_m = GoalToolPose.from_poses(
        pose_dict_m,
        ordered_tool_frames=list(planner_m.tool_frames),
        num_goalset=1,
    )
    for slot in range(B):
        planner_m.scene_collision_checker.load_collision_model(
            Scene(), env_idx=slot,
        )
    mg_result_m = planner_m.plan_pose(
        mg_goal_m, zero_js_m, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_m = float(mg_result_m.position_error.max().item()) if mg_result_m.position_error is not None else float("nan")
    rot_err_m = float(mg_result_m.rotation_error.max().item()) if mg_result_m.rotation_error is not None else float("nan")
    print(f"  success={mg_result_m.success.tolist()}")
    print(f"  pos_err={pos_err_m:.4f}m rot_err={rot_err_m:.4f}rad")

    # ----- Test: use per-env runtime disable instead of init-time broadcast
    # Same merged L=4 cfg, but call update_tool_pose_criteria_per_env (the
    # runtime per-env path) instead of update_tool_pose_criteria (the
    # init-time broadcast path). If init-time broadcast leaves something
    # uncovered, per-env should fix it.
    print("\n========== MG merged + per-env disable (runtime path) ==========")
    cfg_perenv = MotionPlannerCfg.create(
        robot=merged_cfg,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
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
    planner_pe = BatchMotionPlanner(cfg_perenv)
    planner_pe.warmup(enable_graph=False, num_warmup_iterations=3)
    # FIRST init-time broadcast (default tracked) — same as Server does.
    planner_pe.update_tool_pose_criteria(crit)
    # THEN per-env runtime override at slot 0 — Server does this every solve.
    planner_pe.update_tool_pose_criteria_per_env(0, crit)
    for slot in range(B):
        planner_pe.scene_collision_checker.load_collision_model(
            Scene(), env_idx=slot,
        )
    mg_goal_pe = GoalToolPose.from_poses(
        pose_dict_m,
        ordered_tool_frames=list(planner_pe.tool_frames),
        num_goalset=1,
    )
    mg_result_pe = planner_pe.plan_pose(
        mg_goal_pe, zero_js_m, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_pe = float(mg_result_pe.position_error.max().item()) if mg_result_pe.position_error is not None else float("nan")
    rot_err_pe = float(mg_result_pe.rotation_error.max().item()) if mg_result_pe.rotation_error is not None else float("nan")
    print(f"  success={mg_result_pe.success.tolist()}")
    print(f"  pos_err={pos_err_pe:.4f}m rot_err={rot_err_pe:.4f}rad")

    # ----- Test: replace disabled() with TINY-but-nonzero weight on extras
    # If disabled() (axes_weight = [0]*6) is the buggy case but small
    # nonzero weight succeeds, the bug is specifically in the zero-weight
    # code path (likely the convergence_tolerance=0 / divide-by-zero
    # safety branch). If both fail, the bug is in having extras AT ALL.
    print("\n========== MG merged + extras with TINY weight (1e-6) ==========")
    cfg_tiny = MotionPlannerCfg.create(
        robot=merged_cfg,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
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
    planner_tiny = BatchMotionPlanner(cfg_tiny)
    planner_tiny.warmup(enable_graph=False, num_warmup_iterations=3)
    crit_tiny: dict = {}
    for frame in planner_tiny.tool_frames:
        if frame in extras:
            crit_tiny[frame] = ToolPoseCriteria.track_position_and_orientation(
                xyz=[1e-6, 1e-6, 1e-6], rpy=[1e-6, 1e-6, 1e-6],
            )
        else:
            crit_tiny[frame] = ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.1, 0.1, 0.1],
            )
    planner_tiny.update_tool_pose_criteria(crit_tiny)
    for slot in range(B):
        planner_tiny.scene_collision_checker.load_collision_model(
            Scene(), env_idx=slot,
        )
    mg_goal_tiny = GoalToolPose.from_poses(
        pose_dict_m,
        ordered_tool_frames=list(planner_tiny.tool_frames),
        num_goalset=1,
    )
    mg_result_tiny = planner_tiny.plan_pose(
        mg_goal_tiny, zero_js_m, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_tiny = float(mg_result_tiny.position_error.max().item()) if mg_result_tiny.position_error is not None else float("nan")
    rot_err_tiny = float(mg_result_tiny.rotation_error.max().item()) if mg_result_tiny.rotation_error is not None else float("nan")
    print(f"  success={mg_result_tiny.success.tolist()}")
    print(f"  pos_err={pos_err_tiny:.4f}m rot_err={rot_err_tiny:.4f}rad")

    # ----- Test: extras with weight=0 BUT convergence_tolerance LARGE
    # If convergence_tolerance=0 (the disabled() default via __post_init__)
    # is what makes the kernel never trigger its "skip" branch and
    # propagates non-zero distances even with zero weight, setting
    # tolerance to a large value like 1e6 should make the kernel
    # short-circuit cleanly to zero distance for extras.
    print("\n========== MG merged + disabled() but LARGE convergence_tolerance ==========")
    cfg_loose = MotionPlannerCfg.create(
        robot=merged_cfg,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
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
    planner_loose = BatchMotionPlanner(cfg_loose)
    planner_loose.warmup(enable_graph=False, num_warmup_iterations=3)
    crit_loose: dict = {}
    for frame in planner_loose.tool_frames:
        if frame in extras:
            # Hand-craft: zero weights AND huge tolerances so the kernel's
            # `if distance < tolerance` branch always fires and pins the
            # geometric distance to 0.
            from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria as _TPC
            crit_loose[frame] = _TPC(
                terminal_pose_axes_weight_factor=[0.0]*6,
                non_terminal_pose_axes_weight_factor=[0.0]*6,
                terminal_pose_convergence_tolerance=[1e6, 1e6],
                non_terminal_pose_convergence_tolerance=[1e6, 1e6],
            )
        else:
            crit_loose[frame] = ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.1, 0.1, 0.1],
            )
    planner_loose.update_tool_pose_criteria(crit_loose)
    for slot in range(B):
        planner_loose.scene_collision_checker.load_collision_model(
            Scene(), env_idx=slot,
        )
    mg_goal_loose = GoalToolPose.from_poses(
        pose_dict_m,
        ordered_tool_frames=list(planner_loose.tool_frames),
        num_goalset=1,
    )
    mg_result_loose = planner_loose.plan_pose(
        mg_goal_loose, zero_js_m, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_loose = float(mg_result_loose.position_error.max().item()) if mg_result_loose.position_error is not None else float("nan")
    rot_err_loose = float(mg_result_loose.rotation_error.max().item()) if mg_result_loose.rotation_error is not None else float("nan")
    print(f"  success={mg_result_loose.success.tolist()}")
    print(f"  pos_err={pos_err_loose:.4f}m rot_err={rot_err_loose:.4f}rad")

    # ----- Stress test: num_trajopt_seeds=8 (2× current default)
    print("\n========== MG merged + num_trajopt_seeds=8 ==========")
    cfg_big = MotionPlannerCfg.create(
        robot=merged_cfg,
        device_cfg=device_cfg,
        scene_model=[{} for _ in range(B)],
        max_batch_size=B,
        multi_env=True,
        max_goalset=1,
        collision_cache={"cuboid": 0, "mesh": 500},
        num_trajopt_seeds=8,
        num_ik_seeds=32,
        use_cuda_graph=False,
        self_collision_check=True,
        optimizer_collision_activation_distance=0.025,
    )
    planner_big = BatchMotionPlanner(cfg_big)
    planner_big.warmup(enable_graph=False, num_warmup_iterations=3)
    planner_big.update_tool_pose_criteria(crit)
    for slot in range(B):
        planner_big.scene_collision_checker.load_collision_model(
            Scene(), env_idx=slot,
        )
    mg_goal_big = GoalToolPose.from_poses(
        pose_dict_m,
        ordered_tool_frames=list(planner_big.tool_frames),
        num_goalset=1,
    )
    mg_result_big = planner_big.plan_pose(
        mg_goal_big, zero_js_m, max_attempts=2, enable_graph_attempt=0,
    )
    pos_err_big = float(mg_result_big.position_error.max().item()) if mg_result_big.position_error is not None else float("nan")
    rot_err_big = float(mg_result_big.rotation_error.max().item()) if mg_result_big.rotation_error is not None else float("nan")
    print(f"  success={mg_result_big.success.tolist()}")
    print(f"  pos_err={pos_err_big:.4f}m rot_err={rot_err_big:.4f}rad")


if __name__ == "__main__":
    main()
