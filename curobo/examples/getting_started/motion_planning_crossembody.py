# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-embodiment motion-planning smoke test.

For every migrated ``magicsim_*.yml`` config under
``curobo/curobo/content/configs/robot/``, build a
:class:`~curobo.motion_planner.MotionPlanner`, run a real ``plan_pose`` to
an FK-plus-offset goal (guaranteed reachable), and assert the result has
``success.any() == True``. This is not a yaml-syntax check — the planner
actually runs.

Run all robots:

    python -m curobo.examples.getting_started.motion_planning_crossembody

Run one robot (for debug):

    python -m curobo.examples.getting_started.motion_planning_crossembody \\
        --robot magicsim_arx_x5.yml
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


ROBOTS: List[str] = [
    "magicsim_arx_x5.yml",
    "magicsim_dual_arx_x5.yml",
    "magicsim_dual_piper.yml",
    "magicsim_dual_so101.yml",
    "magicsim_franka.yml",
    "magicsim_franka_umi.yml",
    "magicsim_frankarobotiq.yml",
    "magicsim_genie1.yml",
    "magicsim_genie1_mobile.yml",
    "magicsim_lift2_mobile.yml",
    "magicsim_mobile_x7s.yml",
    "magicsim_mobile_x7s_mobile.yml",
    "magicsim_piper_x.yml",
    "magicsim_ridgebacksawyer.yml",
    "magicsim_ridgebacksawyer_mobile.yml",
    "magicsim_so101.yml",
    "magicsim_split_aloha.yml",
    "magicsim_split_aloha_mobile.yml",
    "magicsim_vega1p.yml",
    "magicsim_vega1p_mobile.yml",
    "magicsim_vega1p_sharpa.yml",
    "magicsim_vega1p_sharpa_mobile.yml",
    "magicsim_xtrainer.yml",
]

# Robots that are known to fail the smoke test and intentionally skipped.
# Populate this list as migration progresses.
XFAIL: set = set()


def motion_gen_smoke(
    robot_yaml: str,
    joint_delta: float = 0.1,
    max_attempts: int = 5,
) -> bool:
    """Build planner, call plan_pose to a reachable goal.

    Goal = FK(default_joints + joint_delta). This is reachable by
    construction (there is at least one IK solution: the perturbed
    joint state itself), so a ``success.any() == False`` means the
    robot config is bad (self-collision at rest, bad spheres, missing
    joint limits) rather than a difficult planning problem.
    """
    cfg = MotionPlannerCfg.create(robot=robot_yaml, use_cuda_graph=False)
    planner = MotionPlanner(cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=2)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    # Pick a nearby joint state (bounded by joint_delta), stay inside limits.
    q_goal_pos = q_start.position.clone()
    q_goal_pos[..., 0] += joint_delta  # just perturb first joint
    q_goal = JointState.from_position(q_goal_pos, joint_names=planner.joint_names)

    fk = planner.compute_kinematics(q_goal)
    tool_frames = list(planner.tool_frames)
    pose_dict = fk.tool_poses.to_dict()

    goal_poses: Dict[str, Pose] = {}
    for tf in tool_frames:
        p = pose_dict[tf]
        pos = p.position.detach().reshape(-1, 3)[:1]
        quat = p.quaternion.detach().reshape(-1, 4)[:1]
        goal_poses[tf] = Pose(position=pos, quaternion=quat)

    goal_tp = GoalToolPose.from_poses(
        goal_poses, ordered_tool_frames=tool_frames,
    )

    result = planner.plan_pose(
        current_state=q_start,
        goal_tool_poses=goal_tp,
        max_attempts=max_attempts,
    )
    if result is None:
        return False
    return bool(result.success.any().item())


def _one(robot: str) -> Dict[str, object]:
    try:
        ok = motion_gen_smoke(robot)
        return {"robot": robot, "ok": ok, "error": None}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return {"robot": robot, "ok": False, "error": str(e), "traceback": tb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--log-dir", default="/tmp/curobo_migrate")
    args = ap.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    targets = [args.robot] if args.robot else ROBOTS

    summary = []
    n_ok = 0
    for r in targets:
        log_path = Path(args.log_dir) / f"{Path(r).stem}.smoke.log"
        try:
            res = _one(r)
        except Exception as e:  # noqa: BLE001
            res = {"robot": r, "ok": False, "error": str(e)}
        summary.append(res)

        with log_path.open("w") as fh:
            fh.write(json.dumps(res, default=str, indent=2))

        if r in XFAIL:
            tag = "XFAIL"
        elif res["ok"]:
            tag = "OK"
            n_ok += 1
        else:
            tag = "FAIL"

        if args.json:
            print(json.dumps(res, default=str))
        else:
            msg = res.get("error") or ""
            if msg:
                msg = f"  [{msg.splitlines()[0][:120]}]"
            print(f"{tag:5s} {r}{msg}")

    if not args.json:
        print(f"\n{n_ok}/{len(targets)} ok (xfail: {len([r for r in targets if r in XFAIL])})")
    return 0 if n_ok == len(targets) - sum(1 for r in targets if r in XFAIL) else 1


if __name__ == "__main__":
    raise SystemExit(main())
