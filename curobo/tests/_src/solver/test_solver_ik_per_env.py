# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for per-env tool-pose disabling on
:class:`InverseKinematics` (single + goalset modes).

All tests run against the official ``dual_ur10e.yml`` (tool_frames =
``["tool1", "tool0"]``) so we have two independent tool frames to disable
per env.

Coverage:
- IK single-goal — per-env disable lets envs in one batch drive
  different arms, with the disabled arm contributing zero gradient.
- IK goalset (sub-option A) — each env picks one arm, gets G candidates
  on it, the other arm is disabled. ``result.goalset_index`` is
  meaningful for the chosen arm; ignored for the disabled one.
- ``update_tool_pose_criteria_per_env`` writes only the targeted env
  row; sibling envs see no behaviour change.
"""

from __future__ import annotations

# Standard Library
import pytest

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def per_env_ik_solver(cuda_device_cfg):
    """4-env multi_env IK solver on dual_ur10e (per_env auto-enabled).

    ``max_goalset=8`` so the same solver instance is reused for both the
    single-goal tests (call with ``num_goalset=1``) and the goalset tests
    (call with ``num_goalset=8``) without re-allocation.
    """
    cfg = IKSolverCfg.create(
        robot="dual_ur10e.yml",
        device_cfg=cuda_device_cfg,
        num_seeds=8,
        use_cuda_graph=False,
        max_batch_size=4,
        multi_env=True,
        max_goalset=8,
    )
    cfg.use_lm_seed = False
    return IKSolver(cfg)


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


def _apply_per_env_disable(solver: IKSolver, device_cfg: DeviceCfg, disable_pattern):
    """``disable_pattern[env_idx]`` -> set of tool_frame names to disable for that env.

    Other frames in the env are set to track_position_and_orientation.
    """
    tool_frames = list(solver.kinematics.tool_frames)
    for env_idx, disable_set in enumerate(disable_pattern):
        criteria = {}
        for f in tool_frames:
            if f in disable_set:
                criteria[f] = _disabled_criterion(device_cfg)
            else:
                criteria[f] = _track_criterion(device_cfg)
        solver.update_tool_pose_criteria_per_env(env_idx, criteria)


# ---------------------------------------------------------------------------
# IK single-goal — per-env disable
# ---------------------------------------------------------------------------


class TestIKPerEnvSingle:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_single_mode_disabled_arm_does_not_drive_solution(
        self, per_env_ik_solver, cuda_device_cfg
    ):
        """env 0 disables tool1 (link 0), env 1 disables tool0 (link 1),
        envs 2/3 keep both. All four envs solved in one ``solve_pose``.

        Indifference check: with the same seed, varying the disabled
        link's target row must NOT change env 0's / env 1's solution
        (weight=0 → no gradient → no dependence on filler value).
        """
        solver = per_env_ik_solver
        N = 4
        tool_frames = list(solver.kinematics.tool_frames)
        assert tool_frames == ["tool1", "tool0"], (
            f"dual_ur10e fixture expectation broke; got tool_frames={tool_frames}"
        )

        _apply_per_env_disable(
            solver,
            cuda_device_cfg,
            disable_pattern=[
                {"tool1"},  # env 0: drive tool0 only
                {"tool0"},  # env 1: drive tool1 only
                set(),      # env 2: drive both
                set(),      # env 3: drive both
            ],
        )

        # Generate provably-reachable goals by computing FK at the default
        # joint state and using those tool poses as targets. Both arms
        # already satisfy them at the seed, so env 2/3 (which track both)
        # are guaranteed to converge.
        default_js_1 = solver.default_joint_state.clone().unsqueeze(0)
        kin = solver.compute_kinematics(default_js_1)
        retract_link0 = kin.tool_poses.get_link_pose(tool_frames[0], make_contiguous=True)
        retract_link1 = kin.tool_poses.get_link_pose(tool_frames[1], make_contiguous=True)

        # Expand to N envs (same target across envs).
        pose_link0 = Pose(
            position=retract_link0.position.expand(N, 3).contiguous(),
            quaternion=retract_link0.quaternion.expand(N, 4).contiguous(),
        )
        pose_link1 = Pose(
            position=retract_link1.position.expand(N, 3).contiguous(),
            quaternion=retract_link1.quaternion.expand(N, 4).contiguous(),
        )
        goal = GoalToolPose.from_poses(
            {tool_frames[0]: pose_link0.unsqueeze(1),
             tool_frames[1]: pose_link1.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )

        result_a = solver.solve_pose(goal_tool_poses=goal)
        success_a = result_a.success.flatten()
        assert success_a.numel() == N
        # Every env must succeed on its tracked frames (the seed itself
        # satisfies them).
        assert bool(success_a.all()), f"all envs should succeed; got {success_a}"

        # Indifference check: vary env 0's tool1 target (the disabled
        # link). env 0's solution should NOT be affected by the disabled
        # link's filler value (weight=0 → no gradient). We validate this
        # at the env-0 success level: env 0 must still succeed even
        # though the disabled link's target is far away.
        position_a_alt_link0 = pose_link0.position.clone()
        position_a_alt_link0[0] = torch.tensor(
            [-2.0, -2.0, -2.0], **cuda_device_cfg.as_torch_dict()
        )
        pose_link0_alt = Pose(
            position=position_a_alt_link0,
            quaternion=pose_link0.quaternion.clone(),
        )
        goal_alt = GoalToolPose.from_poses(
            {tool_frames[0]: pose_link0_alt.unsqueeze(1),
             tool_frames[1]: pose_link1.unsqueeze(1)},
            ordered_tool_frames=tool_frames,
        )
        result_b = solver.solve_pose(goal_tool_poses=goal_alt)
        success_b = result_b.success.flatten()
        # env 0 (which has tool1 disabled) must still succeed despite the
        # absurd tool1 target.
        assert bool(success_b[0].item()), (
            "env 0 must succeed when its disabled tool1 target is unreachable"
        )


# ---------------------------------------------------------------------------
# IK goalset — sub-option A: per-env arm selection with G candidates
# ---------------------------------------------------------------------------


class TestIKPerEnvGoalset:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_goalset_mode_per_env_arm_selection(
        self, per_env_ik_solver, cuda_device_cfg
    ):
        """Sub-option A end-to-end:

        env 0: drive tool0 with 8 candidates, disable tool1.
        env 1: drive tool1 with 8 candidates, disable tool0.
        env 2: drive tool0 with 8 candidates, disable tool1.
        env 3: drive tool1 with 8 candidates, disable tool0.

        All 4 envs in one ``solve_pose``.
        """
        solver = per_env_ik_solver
        tool_frames = list(solver.kinematics.tool_frames)
        N = 4
        G = 8

        # arm_per_env[e] picks the chosen tool_frames index.
        # tool_frames = ["tool1", "tool0"], so 0 -> tool1, 1 -> tool0.
        arm_per_env = [1, 0, 1, 0]
        disable_pattern = [
            {tool_frames[1 - a]} for a in arm_per_env
        ]
        _apply_per_env_disable(solver, cuda_device_cfg, disable_pattern)

        # Build [N, 1, num_links=2, G, 3/4] goal tensor via direct fill.
        L = len(tool_frames)
        position = torch.zeros((N, 1, L, G, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((N, 1, L, G, 4), **cuda_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0  # identity

        # Per-env chosen-arm candidates: G poses scattered along x.
        for e in range(N):
            chosen = arm_per_env[e]
            for g in range(G):
                position[e, 0, chosen, g, 0] = 0.45 + 0.02 * g
                position[e, 0, chosen, g, 1] = 0.30 if chosen == 1 else -0.30
                position[e, 0, chosen, g, 2] = 0.50
            # Other arm: same identity quat + a placeholder pose (weight=0).
            other = 1 - chosen
            position[e, 0, other, :, 0] = 0.0  # arbitrary; weight is 0

        goal = GoalToolPose(
            tool_frames=tool_frames,
            position=position,
            quaternion=quaternion,
        )

        result = solver.solve_pose(goal_tool_poses=goal)
        success = result.success.flatten()
        assert bool(success.all()), f"all envs should succeed; got {success}"

        # goalset_index for the chosen arm must lie in [0, G).
        # result.goalset_index has shape (N, 1, L) per the kernel contract.
        gi = result.goalset_index
        assert gi.shape[0] == N
        for e in range(N):
            chosen = arm_per_env[e]
            picked = int(gi[e, 0, chosen].item())
            assert 0 <= picked < G, (
                f"env {e}, chosen arm {chosen}: goalset_index {picked} out of [0, {G})"
            )


# ---------------------------------------------------------------------------
# Per-env update isolation across an IKSolver round-trip
# ---------------------------------------------------------------------------


class TestIKPerEnvUpdateIsolation:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_per_env_update_only_touches_targeted_env(
        self, per_env_ik_solver, cuda_device_cfg
    ):
        """Reset all envs to track-both, run a baseline solve, then
        disable tool1 for env 1 only, and re-solve.

        Asserts: env 0/2/3 still see the SAME tool-pose cost weights
        as before (the per-env update only touched env 1's row).
        We validate this at the criteria-buffer level rather than
        comparing solutions, because seed sampling makes solver outputs
        non-bit-exact even when cost is unchanged.
        """
        solver = per_env_ik_solver
        tool_frames = list(solver.kinematics.tool_frames)

        # Reset all envs to track both (broadcast).
        track = _track_criterion(cuda_device_cfg)
        solver.update_tool_pose_criteria(
            {f: track for f in tool_frames}
        )

        # Find any rollout whose cost_manager carries the tool_pose cost
        # (not every optimizer stage builds one — e.g., the MPPI seed
        # stage may have only collision + cspace costs).
        tp_cost = None
        for rc in solver.core.optimizer_rollouts + [solver.core.metrics_rollout]:
            cand = rc.cost_manager.get_cost("tool_pose")
            if cand is not None:
                tp_cost = cand
                break
        assert tp_cost is not None and tp_cost.per_env, (
            "expected per-env tool-pose cost on at least one rollout"
        )
        weight_buf = tp_cost._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor
        before_env0 = weight_buf[0].clone()
        before_env2 = weight_buf[2].clone()
        before_env3 = weight_buf[3].clone()

        # Disable tool1 (link 0) for env 1 only.
        solver.update_tool_pose_criteria_per_env(
            env_idx=1,
            tool_pose_criteria={tool_frames[0]: _disabled_criterion(cuda_device_cfg)},
        )

        # Sibling envs untouched.
        assert torch.equal(weight_buf[0], before_env0), "env 0 weight row should not change"
        assert torch.equal(weight_buf[2], before_env2), "env 2 weight row should not change"
        assert torch.equal(weight_buf[3], before_env3), "env 3 weight row should not change"

        # env 1, link 0 (tool1) terminal weight is now zero.
        assert torch.equal(
            weight_buf[1, 0], torch.zeros(6, **cuda_device_cfg.as_torch_dict())
        ), "env 1, tool1 weight should be zero after disable"
        # env 1, link 1 (tool0) untouched (still tracking).
        assert torch.equal(weight_buf[1, 1], before_env0[1])
