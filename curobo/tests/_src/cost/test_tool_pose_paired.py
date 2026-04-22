# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level unit tests for the paired per-env :class:`ToolPoseCost`
backend (``ToolPoseDistancePerEnvPaired`` + the matching warp kernel).

These tests exercise the cost forward pass directly without building
any solver. They verify:

1. **Paired vs unpaired argmin**: constructed goals where independent
   per-link argmins point to different ``g_idx`` but the paired
   sum-argmin points to a third ``g_idx``. Paired kernel returns the
   same ``g`` for every link in the group; unpaired returns independent
   per-link picks.
2. **G=1 degeneracy**: with only one candidate per link, paired and
   unpaired produce identical outputs.
3. **Paired + per-env disable**: a link whose ``ToolPoseCriteria`` is
   ``disabled()`` contributes zero to the paired sum, so the chosen ``g``
   is driven only by the remaining active links (and gradient on the
   disabled link is zero).
"""

# Standard Library
import pytest

# Third Party
import torch
import warp as wp

# Initialize Warp before any kernel JIT / launch (the paired kernel
# JITs per (num_goalset, rotation_method); without this call the test
# process hits ``'NoneType' has no attribute 'cuda_devices'`` because
# other curobo tests transitively init Warp inside solver builds — we
# do the raw forward here and must init ourselves.
wp.init()

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


def _build_cost(
    device_cfg: DeviceCfg,
    num_envs: int,
    frames,
    paired: bool,
) -> ToolPoseCost:
    """Build a per-env ToolPoseCost with the requested paired flag.

    ``weight=[1.0, 1.0]`` is the kernel's ``position_orientation_weight``
    (NOT a per-term scalar) — first element is position weight, second
    is orientation weight.
    """
    cfg = ToolPoseCostCfg(
        device_cfg=device_cfg,
        weight=[1.0, 1.0],
        tool_frames=list(frames),
        per_env=True,
        num_envs=num_envs,
        paired=paired,
    )
    return ToolPoseCost(cfg)


def _build_inputs(
    device_cfg: DeviceCfg,
    num_envs: int,
    frames,
    link0_goal_x,
    link1_goal_x,
):
    """Build (current_tool_poses, goal_tool_poses, idxs_goal) where:

    - Current: both links at origin, identity quat.
    - Goal: G candidates per link, varying along x only (position-only test).
    - idxs_goal: identity (env e → goal row e).

    ``link0_goal_x`` / ``link1_goal_x`` are length-G lists of x-offsets
    — same G across both so the kernel constant matches.
    """
    L = len(frames)
    G = len(link0_goal_x)
    assert len(link1_goal_x) == G, "this helper assumes matched G"
    cur_pos = torch.zeros((num_envs, 1, L, 3), **device_cfg.as_torch_dict())
    cur_quat = torch.zeros((num_envs, 1, L, 4), **device_cfg.as_torch_dict())
    cur_quat[..., 0] = 1.0

    goal_pos = torch.zeros((num_envs, L, G, 3), **device_cfg.as_torch_dict())
    goal_quat = torch.zeros((num_envs, L, G, 4), **device_cfg.as_torch_dict())
    goal_quat[..., 0] = 1.0
    for e in range(num_envs):
        for g in range(G):
            goal_pos[e, 0, g, 0] = link0_goal_x[g]
            goal_pos[e, 1, g, 0] = link1_goal_x[g]

    current_tool = ToolPose(
        tool_frames=list(frames), position=cur_pos, quaternion=cur_quat,
    )
    goal_tool = GoalToolPose(
        tool_frames=list(frames),
        position=goal_pos.unsqueeze(1),
        quaternion=goal_quat.unsqueeze(1),
    )
    idxs_goal = torch.arange(
        num_envs, device=device_cfg.device, dtype=torch.int32,
    ).view(num_envs, 1)
    return current_tool, goal_tool, idxs_goal


class TestPairedVsUnpairedArgmin:
    """Constructed goalset where paired and unpaired give different ``g``."""

    # Per-link distances along x for 4 candidates, crafted so:
    #   link0 independent argmin = 0 (x=0.10)
    #   link1 independent argmin = 3 (x=0.15)
    #   sum:   [0.90, 0.95, 0.55, 0.85]  → paired argmin = 2
    LINK0 = [0.10, 0.40, 0.25, 0.70]
    LINK1 = [0.80, 0.55, 0.30, 0.15]
    EXPECTED_PAIRED_G = 2
    EXPECTED_UNPAIRED = [0, 3]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_paired_kernel_picks_shared_sum_argmin(self, cuda_device_cfg):
        """Paired: every link in the group reports the same g (the sum-argmin)."""
        N, frames = 2, ["tool0", "tool1"]
        cost = _build_cost(cuda_device_cfg, N, frames, paired=True)
        current, goal, idxs = _build_inputs(
            cuda_device_cfg, N, frames, self.LINK0, self.LINK1,
        )
        cost.setup_batch_tensors(N, 1)
        _, _, _, g_idx = cost.forward(current, goal, idxs_goal=idxs)

        # g_idx shape (N, 1, L). For paired, all links share one g per env.
        for e in range(N):
            gs = g_idx[e, 0].cpu().tolist()
            assert gs == [self.EXPECTED_PAIRED_G, self.EXPECTED_PAIRED_G], (
                f"env {e}: paired should pick shared-sum argmin "
                f"{self.EXPECTED_PAIRED_G}, got {gs}"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_unpaired_kernel_picks_independent_argmins(self, cuda_device_cfg):
        """Unpaired control: same inputs → per-link argmin, as before."""
        N, frames = 2, ["tool0", "tool1"]
        cost = _build_cost(cuda_device_cfg, N, frames, paired=False)
        current, goal, idxs = _build_inputs(
            cuda_device_cfg, N, frames, self.LINK0, self.LINK1,
        )
        cost.setup_batch_tensors(N, 1)
        _, _, _, g_idx = cost.forward(current, goal, idxs_goal=idxs)

        for e in range(N):
            gs = g_idx[e, 0].cpu().tolist()
            assert gs == self.EXPECTED_UNPAIRED, (
                f"env {e}: unpaired should pick independent argmins "
                f"{self.EXPECTED_UNPAIRED}, got {gs}"
            )


class TestPairedG1Degeneracy:
    """With G=1 there's only one choice — paired and unpaired must agree."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_g1_paired_equals_unpaired(self, cuda_device_cfg):
        N, frames = 2, ["tool0", "tool1"]
        # G=1: one candidate per link at x=0.25.
        inputs = _build_inputs(cuda_device_cfg, N, frames, [0.25], [0.40])

        cost_paired = _build_cost(cuda_device_cfg, N, frames, paired=True)
        cost_paired.setup_batch_tensors(N, 1)
        d_p, p_p, r_p, g_p = cost_paired.forward(*inputs[:2], idxs_goal=inputs[2])

        cost_unpaired = _build_cost(cuda_device_cfg, N, frames, paired=False)
        cost_unpaired.setup_batch_tensors(N, 1)
        d_u, p_u, r_u, g_u = cost_unpaired.forward(*inputs[:2], idxs_goal=inputs[2])

        # All four outputs must match numerically.
        torch.testing.assert_close(d_p, d_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(p_p, p_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(r_p, r_u, rtol=1e-5, atol=1e-6)
        assert torch.equal(g_p, g_u), f"g_idx mismatch: paired={g_p} vs unpaired={g_u}"


class TestPairedPerEnvDisable:
    """Paired + per-env disable: disabled link drops from the paired sum.

    Scenario (2 envs, G=4, same goal data as ``TestPairedVsUnpairedArgmin``):
    - env 0: both links tracked → paired argmin = 2 (as before).
    - env 1: tool0 disabled → only tool1 contributes → argmin = 3 (tool1's own).

    Verifies:
    (a) env 0's chosen g is unaffected by env 1's disable.
    (b) env 1's chosen g equals tool1's independent argmin.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_paired_with_one_link_disabled_per_env(self, cuda_device_cfg):
        N, frames = 2, ["tool0", "tool1"]
        cost = _build_cost(cuda_device_cfg, N, frames, paired=True)

        # Disable tool0 for env 1 only.
        disabled = ToolPoseCriteria.disabled()
        disabled.device_cfg = cuda_device_cfg
        disabled.__post_init__()
        cost.update_tool_pose_criteria_per_env(
            env_idx=1, tool_pose_criteria={"tool0": disabled},
        )

        current, goal, idxs = _build_inputs(
            cuda_device_cfg, N, frames,
            TestPairedVsUnpairedArgmin.LINK0,
            TestPairedVsUnpairedArgmin.LINK1,
        )
        cost.setup_batch_tensors(N, 1)
        _, _, _, g_idx = cost.forward(current, goal, idxs_goal=idxs)

        # env 0 — both tracked, paired sum-argmin = 2
        assert g_idx[0, 0].cpu().tolist() == [2, 2], (
            f"env 0 expected paired [2, 2], got {g_idx[0, 0].cpu().tolist()}"
        )
        # env 1 — tool0 disabled (weight row zero). Paired sum over
        # link reduces to just tool1's distance → argmin = 3.
        expected_env1 = [3, 3]
        assert g_idx[1, 0].cpu().tolist() == expected_env1, (
            f"env 1 expected paired [3, 3] (tool1's argmin since tool0 "
            f"disabled), got {g_idx[1, 0].cpu().tolist()}"
        )
