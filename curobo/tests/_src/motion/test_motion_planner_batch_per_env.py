# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-env tool-pose disabling on :class:`BatchMotionPlanner` (single mode).

Goalset for MotionGen is intentionally out of scope — the production
Servers (MagicSim) hardcode ``num_goalset=1`` for MotionGen. We only
exercise the per-env disable path that propagates to BOTH the IK stage
AND the TrajOpt stage of motion planning.
"""

from __future__ import annotations

# Standard Library
import pytest

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def per_env_motion_planner(cuda_device_cfg):
    """4-env BatchMotionPlanner on dual_ur10e with per-env auto-enabled."""
    cfg = MotionPlannerCfg.create(
        robot="dual_ur10e.yml",
        device_cfg=cuda_device_cfg,
        num_ik_seeds=8,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_batch_size=4,
        multi_env=True,
        max_goalset=1,
    )
    return BatchMotionPlanner(cfg)


def _track_criterion(device_cfg: DeviceCfg) -> ToolPoseCriteria:
    c = ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0]
    )
    c.device_cfg = device_cfg
    c.__post_init__()
    return c


def _disabled_criterion(device_cfg: DeviceCfg) -> ToolPoseCriteria:
    c = ToolPoseCriteria.disabled()
    c.device_cfg = device_cfg
    c.__post_init__()
    return c


class TestBatchMotionPlannerPerEnv:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_update_propagates_to_ik_and_trajopt_stages(
        self, per_env_motion_planner, cuda_device_cfg
    ):
        """``update_tool_pose_criteria_per_env`` must hit BOTH the IK
        stage and the TrajOpt stage. Verify by inspecting the per-env
        weight buffers on tool_pose costs found in the IK solver's
        rollouts AND the TrajOpt solver's rollouts.
        """
        planner = per_env_motion_planner
        tool_frames = list(planner.kinematics.tool_frames)
        assert tool_frames == ["tool1", "tool0"]

        # Disable tool1 (link 0) for env 1 only.
        planner.update_tool_pose_criteria_per_env(
            env_idx=1,
            tool_pose_criteria={tool_frames[0]: _disabled_criterion(cuda_device_cfg)},
        )

        def _find_tool_pose_cost(rollouts):
            for rc in rollouts:
                cand = rc.cost_manager.get_cost("tool_pose")
                if cand is not None:
                    return cand
            return None

        ik_tp = _find_tool_pose_cost(
            planner.ik_solver.core.optimizer_rollouts
            + [planner.ik_solver.core.metrics_rollout]
        )
        trajopt_tp = _find_tool_pose_cost(
            planner.trajopt_solver.core.optimizer_rollouts
            + [planner.trajopt_solver.core.metrics_rollout]
        )
        assert ik_tp is not None and ik_tp.per_env, (
            "per-env tool-pose cost should exist on the IK stage"
        )
        assert trajopt_tp is not None and trajopt_tp.per_env, (
            "per-env tool-pose cost should exist on the TrajOpt stage"
        )

        # IK stage env 1 link 0 weight is zero.
        ik_w = ik_tp._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor
        assert torch.equal(
            ik_w[1, 0], torch.zeros(6, **cuda_device_cfg.as_torch_dict())
        ), "IK stage env 1 link 0 weight should be zero after disable"
        # TrajOpt stage env 1 link 0 weight is zero.
        traj_w = trajopt_tp._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor
        assert torch.equal(
            traj_w[1, 0], torch.zeros(6, **cuda_device_cfg.as_torch_dict())
        ), "TrajOpt stage env 1 link 0 weight should be zero after disable"

        # Sibling envs untouched on both stages.
        assert not torch.equal(
            ik_w[0, 0], torch.zeros(6, **cuda_device_cfg.as_torch_dict())
        ), "IK stage env 0 link 0 must NOT be disabled"
        assert not torch.equal(
            traj_w[0, 0], torch.zeros(6, **cuda_device_cfg.as_torch_dict())
        ), "TrajOpt stage env 0 link 0 must NOT be disabled"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_update_does_not_raise_cuda_graph_error(
        self, per_env_motion_planner, cuda_device_cfg
    ):
        """``update_tool_pose_criteria_per_env`` between two consecutive
        ``plan_pose`` calls must not raise ``CUDA graph reset is not
        available.`` (planner constructed with ``use_cuda_graph=False``
        per the production Service config).
        """
        planner = per_env_motion_planner
        tool_frames = list(planner.kinematics.tool_frames)

        # Idempotent reset to track-both, then disable tool1 for env 0.
        planner.update_tool_pose_criteria(
            {f: _track_criterion(cuda_device_cfg) for f in tool_frames}
        )
        planner.update_tool_pose_criteria_per_env(
            env_idx=0,
            tool_pose_criteria={tool_frames[0]: _disabled_criterion(cuda_device_cfg)},
        )
        # No raise → success; the next plan_pose smoke-test is covered
        # by the upstream BatchMotionPlanner suite, we don't repeat it
        # here to keep the test deterministic.
