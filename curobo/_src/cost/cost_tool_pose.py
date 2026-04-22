# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Dict, Optional, Tuple

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.tool_pose_criteria import StackedToolPoseCriteria, ToolPoseCriteria
from curobo._src.cost.wp_tool_pose import (
    ToolPoseDistance,
    ToolPoseDistancePerEnv,
    ToolPoseDistancePerEnvPaired,
    create_goalset_pose_distance_kernel_per_env_paired_with_constants,
    create_goalset_pose_distance_kernel_per_env_with_constants,
    create_goalset_pose_distance_kernel_with_constants,
)
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg


class ToolPoseCost(BaseCost):
    """Cost that computes pose goalset cost for multiple links.

    Two backends, picked at construction time by ``config.per_env``:

    * ``per_env=False`` — stock single weight buffer ``(num_links, K)``,
      shared across all envs in the batch. Stock warp kernel.
    * ``per_env=True`` — per-env weight buffer ``(num_envs, num_links, K)``,
      indexed by ``goal_idx = idxs_goal[b_idx]`` (the env index for that
      batch row). Lets each env disable a different subset of tool
      frames inside a single ``solve_pose`` / ``plan_pose`` call. See
      :class:`~curobo._src.cost.wp_tool_pose.ToolPoseDistancePerEnv`.
    """

    def __init__(self, config: ToolPoseCostCfg):
        """Initialize the ToolPoseCost.

        Args:
            config: Configuration for the cost.
        """
        self.config: ToolPoseCostCfg = config
        self._stacked_tool_pose_criteria = StackedToolPoseCriteria.from_tool_pose_criteria(
            config.tool_pose_criteria,
            num_envs=config.num_envs if config.per_env else 0,
        )
        super().__init__(config)

        # Store link names for which this cost applies
        self.tool_frames = config.tool_frames
        self.num_links = len(self.tool_frames)

        # Cache for warp kernels - one per link for efficiency
        self._warp_kernel: Optional[wp.kernel] = None
        self._constants = (
            -1,
            self.config.rotation_method,
        )

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        if batch_size != self._batch_size or horizon != self._horizon:

            num_links = len(self.tool_frames)
            self._out_distance = torch.zeros(
                (batch_size, horizon, 2 * num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_position_distance = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_distance = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_goalset_idx = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.int32,
                device=self.device_cfg.device,
            )

            self._out_position_gradient = torch.zeros(
                (batch_size, horizon, num_links, 3),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_gradient = torch.zeros(
                (batch_size, horizon, num_links, 4),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )

        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: GoalToolPose,
        idxs_goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized implementation for ToolPose."""
        # Input validation
        if current_tool_poses is None or goal_tool_poses is None:
            log_and_raise("current_tool_poses and goal_tool_poses must be provided")
        if current_tool_poses.tool_frames != goal_tool_poses.tool_frames:
            log_and_raise("current_tool_poses and goal_tool_poses must have the same link names")

        num_goalset = goal_tool_poses.num_goalset
        constants = (
            num_goalset,
            self.config.rotation_method,
        )
        if self._constants != constants:
            self._constants = constants
            # Kernel pick is purely a build-time decision based on
            # ``per_env`` / ``paired`` cfg flags — these are FIXED at
            # ToolPoseCost construction. We re-JIT only when num_goalset
            # / rotation_method change (which already happens today).
            # Per-env paired is the only paired path we ship; non-per-env
            # paired isn't wired (every multi-link solve_pose we use is
            # in multi_env mode, which auto-enables per_env).
            if self._stacked_tool_pose_criteria.per_env and self.config.paired:
                self._warp_kernel = (
                    create_goalset_pose_distance_kernel_per_env_paired_with_constants(
                        num_goalset,
                        self.config.rotation_method,
                    )
                )
            elif self._stacked_tool_pose_criteria.per_env:
                self._warp_kernel = (
                    create_goalset_pose_distance_kernel_per_env_with_constants(
                        num_goalset,
                        self.config.rotation_method,
                    )
                )
            elif self.config.paired:
                log_and_raise(
                    "ToolPoseCostCfg.paired=True requires per_env=True "
                    "(non-per-env paired kernel is intentionally not "
                    "implemented — every multi-link solve_pose runs in "
                    "multi_env mode in production)."
                )
            else:
                self._warp_kernel = create_goalset_pose_distance_kernel_with_constants(
                    num_goalset,
                    self.config.rotation_method,
                )

        warp_kernel = self._warp_kernel
        goal_pos = goal_tool_poses.position.squeeze(1)
        goal_quat = goal_tool_poses.quaternion.squeeze(1)
        if self._stacked_tool_pose_criteria.per_env and self.config.paired:
            apply_fn = ToolPoseDistancePerEnvPaired.apply
        elif self._stacked_tool_pose_criteria.per_env:
            apply_fn = ToolPoseDistancePerEnv.apply
        else:
            apply_fn = ToolPoseDistance.apply
        cost, linear_distance, angular_distance, goalset_idx = apply_fn(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            goal_pos,
            goal_quat,
            idxs_goal,
            self._weight,
            self._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor,
            self._stacked_tool_pose_criteria.non_terminal_pose_axes_weight_factor,
            self._stacked_tool_pose_criteria.terminal_pose_convergence_tolerance,
            self._stacked_tool_pose_criteria.non_terminal_pose_convergence_tolerance,
            self._stacked_tool_pose_criteria.project_distance_to_goal,
            self._out_distance,
            self._out_position_distance,
            self._out_rotation_distance,
            self._out_position_gradient,
            self._out_rotation_gradient,
            self._out_goalset_idx,
            self.config.use_grad_input,
            warp_kernel,
        )
        return cost, linear_distance, angular_distance, goalset_idx


    def update_tool_pose_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ):

        self._stacked_tool_pose_criteria.update_tool_pose_criteria(
            tool_pose_criteria=tool_pose_criteria,
        )

    def update_tool_pose_criteria_per_env(
        self,
        env_idx: int,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ):
        """Per-env row update — only valid when ``self.per_env=True``.

        Forwards to :meth:`StackedToolPoseCriteria.update_tool_pose_criteria_per_env`.
        Use to specialise one env's tracked / disabled tool frames without
        touching the others, e.g. ``criteria = {f: ToolPoseCriteria.disabled()}``.
        """
        self._stacked_tool_pose_criteria.update_tool_pose_criteria_per_env(
            env_idx=env_idx,
            tool_pose_criteria=tool_pose_criteria,
        )

    @property
    def per_env(self) -> bool:
        return self._stacked_tool_pose_criteria.per_env

    @property
    def num_envs(self) -> int:
        return self._stacked_tool_pose_criteria.num_envs

    @property
    def paired(self) -> bool:
        """Whether this cost was built with paired-goalset semantics.

        Fixed at construction from ``config.paired``; does not change
        at runtime.
        """
        return bool(self.config.paired)
