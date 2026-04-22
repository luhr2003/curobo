# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-env :class:`StackedToolPoseCriteria` storage and
the :class:`ToolPoseDistancePerEnv` warp kernel.

These tests exercise the kernel + criteria buffer directly, without
building any solver. They verify:

1. ``StackedToolPoseCriteria.from_tool_pose_criteria(criteria, num_envs=N)``
   allocates ``[N, num_links, K]`` weight buffers.
2. ``update_tool_pose_criteria_per_env(env_idx, criteria)`` writes only
   the targeted env row, leaving other env rows untouched.
3. The per-env warp kernel returns per-(env, link) costs that respect
   the per-env weight rows: envs with link L disabled report cost 0
   for that link, while sibling envs (with link L enabled) report
   non-zero cost for the same goal.
"""

# Standard Library
import pytest

# Third Party
import torch
import warp as wp

# Initialize the Warp runtime up front so kernel launches in the per-env
# tool-pose cost find a valid CUDA device. (Other curobo tests build a
# full IKSolver / MotionPlanner which triggers this transitively; we
# call ToolPoseCost.forward directly so we have to do it ourselves.)
wp.init()

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import (
    StackedToolPoseCriteria,
    ToolPoseCriteria,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


def _build_criteria_dict(device_cfg: DeviceCfg, frames):
    """Two-frame dict with identical track-position-and-orientation criteria."""
    return {
        f: ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0]
        )
        for f in frames
    }


# ---------------------------------------------------------------------------
# StackedToolPoseCriteria — per-env shape + per-env update isolation
# ---------------------------------------------------------------------------


class TestStackedToolPoseCriteriaPerEnv:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_buffer_shapes(self, cuda_device_cfg):
        criteria = _build_criteria_dict(cuda_device_cfg, ["tool1", "tool0"])
        # Move into device buffers (mimics from_tool_pose_criteria input).
        for c in criteria.values():
            c.device_cfg = cuda_device_cfg
            c.__post_init__()

        stacked = StackedToolPoseCriteria.from_tool_pose_criteria(
            criteria, num_envs=3
        )

        assert stacked.per_env is True
        assert stacked.num_envs == 3
        assert stacked.terminal_pose_axes_weight_factor.shape == (3, 2, 6)
        assert stacked.non_terminal_pose_axes_weight_factor.shape == (3, 2, 6)
        assert stacked.terminal_pose_convergence_tolerance.shape == (3, 2, 2)
        assert stacked.non_terminal_pose_convergence_tolerance.shape == (3, 2, 2)
        assert stacked.project_distance_to_goal.shape == (3, 2, 1)

        # Every env row must start with the same broadcast values.
        ref = stacked.terminal_pose_axes_weight_factor[0]
        for e in range(1, 3):
            assert torch.equal(stacked.terminal_pose_axes_weight_factor[e], ref)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_update_isolation(self, cuda_device_cfg):
        criteria = _build_criteria_dict(cuda_device_cfg, ["tool1", "tool0"])
        for c in criteria.values():
            c.device_cfg = cuda_device_cfg
            c.__post_init__()

        stacked = StackedToolPoseCriteria.from_tool_pose_criteria(
            criteria, num_envs=3
        )

        # Snapshot env 0 + env 2 weight rows so we can prove they don't move.
        before_env0 = stacked.terminal_pose_axes_weight_factor[0].clone()
        before_env2 = stacked.terminal_pose_axes_weight_factor[2].clone()

        # Disable tool1 (link 0) for env 1 only.
        disabled = ToolPoseCriteria.disabled()
        disabled.device_cfg = cuda_device_cfg
        disabled.__post_init__()
        stacked.update_tool_pose_criteria_per_env(
            env_idx=1,
            tool_pose_criteria={"tool1": disabled},
        )

        # Env 1, link 0 (tool1) terminal weight factor must be all zeros.
        assert torch.equal(
            stacked.terminal_pose_axes_weight_factor[1, 0],
            torch.zeros(6, **cuda_device_cfg.as_torch_dict()),
        )
        # Env 0 + env 2 rows untouched.
        assert torch.equal(stacked.terminal_pose_axes_weight_factor[0], before_env0)
        assert torch.equal(stacked.terminal_pose_axes_weight_factor[2], before_env2)
        # Env 1, link 1 (tool0) untouched.
        assert torch.equal(
            stacked.terminal_pose_axes_weight_factor[1, 1],
            before_env0[1],
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_update_rejects_when_per_env_off(self, cuda_device_cfg):
        criteria = _build_criteria_dict(cuda_device_cfg, ["tool1", "tool0"])
        for c in criteria.values():
            c.device_cfg = cuda_device_cfg
            c.__post_init__()
        # num_envs=0 -> per_env stays off
        stacked = StackedToolPoseCriteria.from_tool_pose_criteria(criteria)
        assert stacked.per_env is False
        with pytest.raises(ValueError, match="per_env"):
            stacked.update_tool_pose_criteria_per_env(0, {"tool1": criteria["tool1"]})


# ---------------------------------------------------------------------------
# ToolPoseCost — per-env kernel pathway
# ---------------------------------------------------------------------------


def _build_per_env_tool_pose_cost(device_cfg: DeviceCfg, num_envs: int, frames):
    # The tool-pose ``weight`` is consumed inside the warp kernel as
    # ``position_orientation_weight`` of shape (2,) — first element is
    # position weight, second is orientation weight. NOT a per-term scalar.
    cfg = ToolPoseCostCfg(
        device_cfg=device_cfg,
        weight=[1.0, 1.0],
        tool_frames=frames,
        per_env=True,
        num_envs=num_envs,
    )
    return ToolPoseCost(cfg)


class TestToolPoseCostPerEnvKernel:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_disable_pattern_yields_zero_cost_on_disabled_links(
        self, cuda_device_cfg
    ):
        """env 0 disables tool1, env 1 disables tool0, env 2 keeps both.

        With identical non-zero position errors on all (env, link) pairs,
        the per-env kernel must return cost==0 for the disabled (env, link)
        slots and cost>0 for the others.
        """
        frames = ["tool1", "tool0"]
        N, L = 3, len(frames)
        cost = _build_per_env_tool_pose_cost(cuda_device_cfg, N, frames)

        # Disable link 0 (tool1) for env 0; disable link 1 (tool0) for env 1;
        # env 2 keeps the broadcast track criteria.
        disabled = ToolPoseCriteria.disabled()
        disabled.device_cfg = cuda_device_cfg
        disabled.__post_init__()
        cost.update_tool_pose_criteria_per_env(env_idx=0, tool_pose_criteria={frames[0]: disabled})
        cost.update_tool_pose_criteria_per_env(env_idx=1, tool_pose_criteria={frames[1]: disabled})

        # Build current_tool_poses + goal_tool_poses with a known offset so
        # the un-disabled cost is provably positive.
        # current_position shape: (B=N, H=1, L, 3); goal_position shape:
        # (batch_goals=N, L, num_goalset=1, 3)
        current_position = torch.zeros((N, 1, L, 3), **cuda_device_cfg.as_torch_dict())
        current_quat = torch.zeros((N, 1, L, 4), **cuda_device_cfg.as_torch_dict())
        current_quat[..., 0] = 1.0
        goal_position = torch.zeros((N, L, 1, 3), **cuda_device_cfg.as_torch_dict())
        goal_position[..., 0] = 0.5     # 0.5m offset in x for every (env, link)
        goal_quat = torch.zeros((N, L, 1, 4), **cuda_device_cfg.as_torch_dict())
        goal_quat[..., 0] = 1.0

        current_tool = ToolPose(
            tool_frames=frames,
            position=current_position,
            quaternion=current_quat,
        )
        goal_tool = GoalToolPose(
            tool_frames=frames,
            position=goal_position.unsqueeze(1),     # (N, 1, L, 1, 3)
            quaternion=goal_quat.unsqueeze(1),
        )
        # idxs_goal: identity (env e -> goal row e). Shape (B, 1) per
        # ToolPoseDistance contract.
        idxs_goal = torch.arange(N, device=cuda_device_cfg.device, dtype=torch.int32).view(N, 1)

        cost.setup_batch_tensors(N, 1)
        cost_value, pos_dist, _, _ = cost.forward(
            current_tool, goal_tool, idxs_goal=idxs_goal
        )

        # cost_value is shape (B, H, L*2) — interleaved position/orientation.
        # Position cost per (env, link) lives at column 2 * link_idx.
        pos_cost = cost_value[:, 0, ::2]   # (N, L)

        # Disabled (env, link): exactly 0.
        assert torch.allclose(
            pos_cost[0, 0],
            torch.tensor(0.0, **cuda_device_cfg.as_torch_dict()),
        ), f"env 0 link 0 cost should be 0, got {pos_cost[0, 0]}"
        assert torch.allclose(
            pos_cost[1, 1],
            torch.tensor(0.0, **cuda_device_cfg.as_torch_dict()),
        ), f"env 1 link 1 cost should be 0, got {pos_cost[1, 1]}"

        # Tracked (env, link): > 0.
        assert pos_cost[0, 1] > 0
        assert pos_cost[1, 0] > 0
        assert pos_cost[2, 0] > 0
        assert pos_cost[2, 1] > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_update_is_isolated_across_envs(self, cuda_device_cfg):
        """Update env_idx=1 only — env 0 + env 2 cost results unchanged."""
        frames = ["tool1", "tool0"]
        N, L = 3, len(frames)
        cost = _build_per_env_tool_pose_cost(cuda_device_cfg, N, frames)

        current_position = torch.zeros((N, 1, L, 3), **cuda_device_cfg.as_torch_dict())
        current_quat = torch.zeros((N, 1, L, 4), **cuda_device_cfg.as_torch_dict())
        current_quat[..., 0] = 1.0
        goal_position = torch.zeros((N, L, 1, 3), **cuda_device_cfg.as_torch_dict())
        goal_position[..., 0] = 0.5
        goal_quat = torch.zeros((N, L, 1, 4), **cuda_device_cfg.as_torch_dict())
        goal_quat[..., 0] = 1.0

        current_tool = ToolPose(
            tool_frames=frames, position=current_position, quaternion=current_quat
        )
        goal_tool = GoalToolPose(
            tool_frames=frames,
            position=goal_position.unsqueeze(1),
            quaternion=goal_quat.unsqueeze(1),
        )
        idxs_goal = torch.arange(N, device=cuda_device_cfg.device, dtype=torch.int32).view(N, 1)
        cost.setup_batch_tensors(N, 1)

        # Baseline: all envs track everything.
        before, _, _, _ = cost.forward(current_tool, goal_tool, idxs_goal=idxs_goal)
        before = before.clone()

        # Update env 1 only.
        disabled = ToolPoseCriteria.disabled()
        disabled.device_cfg = cuda_device_cfg
        disabled.__post_init__()
        cost.update_tool_pose_criteria_per_env(
            env_idx=1, tool_pose_criteria={frames[0]: disabled}
        )

        after, _, _, _ = cost.forward(current_tool, goal_tool, idxs_goal=idxs_goal)

        # env 0 + env 2 rows must be identical to baseline.
        assert torch.equal(after[0], before[0]), "env 0 cost should not change"
        assert torch.equal(after[2], before[2]), "env 2 cost should not change"
        # env 1 link 0 (tool1) cost must drop to 0 (was non-zero).
        assert after[1, 0, 0] == 0.0, f"env 1 link 0 should be 0, got {after[1, 0, 0]}"
        # env 1 link 1 (tool0) cost unchanged from baseline.
        assert torch.equal(after[1, 0, 2:], before[1, 0, 2:])
