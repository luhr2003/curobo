# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Solver-level integration tests for the paired ToolPoseCost backend.

Uses ``tri_ur10e.yml`` (three arms: tool0, tool1, tool2) so the scenarios
match the user-requested cases:

- **Test A — paired G=8 on tool0+tool1, tool2 disabled** (scenario 1):
  ``[G=8, G=8, G=0]``. Paired kernel must return the same ``g`` for
  tool0 and tool1 per env; tool2's column is unused.

- **Test B — padded ragged G** (scenario 2): caller wants
  ``[G=8 on tool0, G=16 on tool1, G=0 on tool2]`` but the kernel takes
  one ``num_goalset``. Caller pads tool0's 8 real candidates to 16 by
  duplicating; runs unpaired with ``num_goalset=16``; tool2 disabled.
  Assertions: solver succeeds, tool0's picked index mod-8 maps back to
  a valid real candidate, tool1's picked index is in [0, 16).
"""
from __future__ import annotations

# Standard Library
import pytest

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.solver_core_cfg import enable_paired_tool_pose
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


def _build_ik(cuda_device_cfg: DeviceCfg, G: int, paired: bool) -> IKSolver:
    """Build an ``InverseKinematics`` on tri_ur10e with ``multi_env=True``,
    ``max_batch_size=2``, ``max_goalset=G``, and the paired flag set on
    every tool-pose cost when ``paired=True``.
    """
    cfg = IKSolverCfg.create(
        robot="tri_ur10e.yml",
        device_cfg=cuda_device_cfg,
        num_seeds=16,
        use_cuda_graph=False,
        max_batch_size=2,
        multi_env=True,
        max_goalset=G,
    )
    cfg.use_lm_seed = False
    if paired:
        # Must be called AFTER the factory auto-enabled per_env (which
        # happens for ``multi_env=True``). Paired dispatch requires
        # per_env. See ``ToolPoseCost.forward``.
        enable_paired_tool_pose(cfg.core_cfg)
    return IKSolver(cfg)


def _track(dc: DeviceCfg) -> ToolPoseCriteria:
    c = ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0]
    )
    c.device_cfg = dc
    c.__post_init__()
    return c


def _disabled(dc: DeviceCfg) -> ToolPoseCriteria:
    c = ToolPoseCriteria.disabled()
    c.device_cfg = dc
    c.__post_init__()
    return c


# ---------------------------------------------------------------------------
# Test A — paired G=8 on tool0+tool1, tool2 disabled
# ---------------------------------------------------------------------------


class TestPairedPairedToolFrames:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_paired_g8_tool0_tool1_with_tool2_disabled(self, cuda_device_cfg):
        """Scenario 1: paired 2-arm goalset + one disabled arm.

        - tool_frames in YAML: ["tool0", "tool1", "tool2"]
        - G=8 candidates per arm, constructed so each g is simultaneously
          reachable for BOTH tool0 and tool1 from the retract config
          (small perturbations around retract FK).
        - tool2 is disabled globally via ``ToolPoseCriteria.disabled()``.
        """
        N, G = 2, 8
        ik = _build_ik(cuda_device_cfg, G=G, paired=True)
        tool_frames = list(ik.kinematics.tool_frames)
        assert tool_frames == ["tool0", "tool1", "tool2"]

        # Disable tool2 for every env (broadcast).
        ik.update_tool_pose_criteria(
            {"tool0": _track(cuda_device_cfg),
             "tool1": _track(cuda_device_cfg),
             "tool2": _disabled(cuda_device_cfg)}
        )

        # Build G=8 candidates near the retract FK so all are reachable.
        default_js_1 = ik.default_joint_state.clone().unsqueeze(0)
        kin = ik.compute_kinematics(default_js_1)
        r_tool0 = kin.tool_poses.get_link_pose("tool0", make_contiguous=True)
        r_tool1 = kin.tool_poses.get_link_pose("tool1", make_contiguous=True)
        r_tool2 = kin.tool_poses.get_link_pose("tool2", make_contiguous=True)

        # position: (N, 1, 3, G, 3); quaternion: (N, 1, 3, G, 4)
        position = torch.zeros((N, 1, 3, G, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((N, 1, 3, G, 4), **cuda_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        # Each candidate pair (tool0_g, tool1_g) perturbs the retract pose
        # by a small shared offset; solver can recover by small joint
        # adjustments. "Paired-compatible" means g=k yields retract+ε_k
        # on both arms simultaneously.
        eps = torch.linspace(-0.015, 0.015, G, **cuda_device_cfg.as_torch_dict())
        for e in range(N):
            for g in range(G):
                # tool0: perturb in +x
                position[e, 0, 0, g, :] = r_tool0.position.view(3) + torch.tensor(
                    [eps[g].item(), 0.0, 0.0], **cuda_device_cfg.as_torch_dict()
                )
                quaternion[e, 0, 0, g, :] = r_tool0.quaternion.view(4)
                # tool1: perturb in +y by the same g-index
                position[e, 0, 1, g, :] = r_tool1.position.view(3) + torch.tensor(
                    [0.0, eps[g].item(), 0.0], **cuda_device_cfg.as_torch_dict()
                )
                quaternion[e, 0, 1, g, :] = r_tool1.quaternion.view(4)
                # tool2 (disabled): filler = retract pose, G duplicates
                position[e, 0, 2, g, :] = r_tool2.position.view(3)
                quaternion[e, 0, 2, g, :] = r_tool2.quaternion.view(4)

        goal = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )
        result = ik.solve_pose(goal_tool_poses=goal)

        # Every env must succeed (retract is already near-reachable).
        assert bool(result.success.any(dim=-1).all().item()), (
            f"all envs should succeed; success={result.success.flatten().tolist()}"
        )

        # Paired: tool0's column and tool1's column must agree per env.
        # Disabled tool2's column is meaningless — don't read it.
        gi = result.goalset_index   # shape (N, 1, num_links)
        tool0_col = tool_frames.index("tool0")
        tool1_col = tool_frames.index("tool1")
        for e in range(N):
            g0 = int(gi[e, 0, tool0_col].item())
            g1 = int(gi[e, 0, tool1_col].item())
            assert g0 == g1, (
                f"env {e}: paired kernel must report same g for tool0 "
                f"({g0}) and tool1 ({g1})"
            )
            assert 0 <= g0 < G, f"env {e}: g={g0} out of [0, {G})"


# ---------------------------------------------------------------------------
# Test B — padded ragged G: tool0 G=8 padded to 16, tool1 G=16, tool2 disabled
# ---------------------------------------------------------------------------


class TestPaddedRaggedGoalset:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_padded_ragged_tool0_vs_tool1_tool2_disabled(self, cuda_device_cfg):
        """Scenario 2: ragged G via caller-side padding.

        The kernel carries one ``num_goalset``. Caller padding strategy:
        duplicate tool0's 8 real candidates across 2 slots each → 16
        padded slots; tool1 uses 16 real candidates; tool2 disabled.
        After solve, ``goalset_index[tool0] % 8`` recovers the real
        candidate; ``goalset_index[tool1]`` is already real.
        """
        N, G_PADDED = 2, 16
        G_REAL_TOOL0 = 8
        ik = _build_ik(cuda_device_cfg, G=G_PADDED, paired=False)
        tool_frames = list(ik.kinematics.tool_frames)

        ik.update_tool_pose_criteria(
            {"tool0": _track(cuda_device_cfg),
             "tool1": _track(cuda_device_cfg),
             "tool2": _disabled(cuda_device_cfg)}
        )

        default_js_1 = ik.default_joint_state.clone().unsqueeze(0)
        kin = ik.compute_kinematics(default_js_1)
        r_tool0 = kin.tool_poses.get_link_pose("tool0", make_contiguous=True)
        r_tool1 = kin.tool_poses.get_link_pose("tool1", make_contiguous=True)
        r_tool2 = kin.tool_poses.get_link_pose("tool2", make_contiguous=True)

        position = torch.zeros((N, 1, 3, G_PADDED, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((N, 1, 3, G_PADDED, 4), **cuda_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        # tool0 real 8 candidates; pad to 16 by duplicating [c0..c7, c0..c7].
        eps_tool0_real = torch.linspace(
            -0.015, 0.015, G_REAL_TOOL0, **cuda_device_cfg.as_torch_dict()
        )
        # tool1 real 16 candidates — a larger spread along y.
        eps_tool1 = torch.linspace(
            -0.03, 0.03, G_PADDED, **cuda_device_cfg.as_torch_dict()
        )

        for e in range(N):
            for g in range(G_PADDED):
                real_g_tool0 = g % G_REAL_TOOL0     # duplicate every 8 slots
                position[e, 0, 0, g, :] = r_tool0.position.view(3) + torch.tensor(
                    [eps_tool0_real[real_g_tool0].item(), 0.0, 0.0],
                    **cuda_device_cfg.as_torch_dict(),
                )
                quaternion[e, 0, 0, g, :] = r_tool0.quaternion.view(4)
                position[e, 0, 1, g, :] = r_tool1.position.view(3) + torch.tensor(
                    [0.0, eps_tool1[g].item(), 0.0],
                    **cuda_device_cfg.as_torch_dict(),
                )
                quaternion[e, 0, 1, g, :] = r_tool1.quaternion.view(4)
                # tool2 disabled filler.
                position[e, 0, 2, g, :] = r_tool2.position.view(3)
                quaternion[e, 0, 2, g, :] = r_tool2.quaternion.view(4)

        goal = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )
        result = ik.solve_pose(goal_tool_poses=goal)
        assert bool(result.success.any(dim=-1).all().item()), (
            f"all envs should succeed; success={result.success.flatten().tolist()}"
        )

        # Unpaired: tool0 and tool1 pick independently. Validate both
        # picks are in [0, G_PADDED) and that tool0's pick maps to a
        # valid real candidate via ``% G_REAL_TOOL0``.
        gi = result.goalset_index
        tool0_col = tool_frames.index("tool0")
        tool1_col = tool_frames.index("tool1")
        for e in range(N):
            g0 = int(gi[e, 0, tool0_col].item())
            g1 = int(gi[e, 0, tool1_col].item())
            assert 0 <= g0 < G_PADDED, f"env {e}: tool0 g={g0} out of padded range"
            assert 0 <= g1 < G_PADDED, f"env {e}: tool1 g={g1} out of range"
            # The caller's translation: picked real index for tool0.
            real_g = g0 % G_REAL_TOOL0
            assert 0 <= real_g < G_REAL_TOOL0, (
                f"env {e}: tool0 padded g={g0} → real g={real_g}, out of [0, {G_REAL_TOOL0})"
            )
