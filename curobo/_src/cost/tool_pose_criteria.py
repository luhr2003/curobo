# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolPoseCriteria:
    """Criteria for a link pose.

    This class is used to define the nature of the cost between the current pose and the goal pose.
    This used as part of the goalset cost term.
    """

    #: Factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the terminal position
    #: and orientation. This is multiplied with the weight.
    terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None

    #: Factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the non-terminal position
    #: and orientation. This is multiplied with the weight.
    non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None

    #: Convergence tolerance for the terminal position and orientation. This should be of shape
    #: (2,). Position unit is meter and orientation unit is radian.
    terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None

    #: Convergence tolerance for the non-terminal position and orientation. This should be of shape
    #: (2,). Position unit is meter and orientation unit is radian.
    non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None

    #: If true, the distance is computed after projecting the current pose to the goal frame.
    project_distance_to_goal: Union[torch.Tensor, bool] = False

    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self):
        if self.terminal_pose_axes_weight_factor is None:
            self.terminal_pose_axes_weight_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        elif len(self.terminal_pose_axes_weight_factor) != 6:
            log_and_raise(
                "terminal_pose_axes_weight_factor must be a list of 6 floats, "
                + f"got {self.terminal_pose_axes_weight_factor}"
            )

        if self.non_terminal_pose_axes_weight_factor is None:
            self.non_terminal_pose_axes_weight_factor = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif len(self.non_terminal_pose_axes_weight_factor) != 6:
            log_and_raise(
                "non_terminal_pose_axes_weight_factor must be a list of 6 floats, "
                + f"got {self.non_terminal_pose_axes_weight_factor}"
            )

        if self.terminal_pose_convergence_tolerance is None:
            self.terminal_pose_convergence_tolerance = [0.0, 0.0]
        elif len(self.terminal_pose_convergence_tolerance) != 2:
            log_and_raise(
                "terminal_pose_convergence_tolerance must be a list of 2 floats, "
                + f"got {self.terminal_pose_convergence_tolerance}"
            )

        if self.non_terminal_pose_convergence_tolerance is None:
            self.non_terminal_pose_convergence_tolerance = [0.0, 0.0]
        elif len(self.non_terminal_pose_convergence_tolerance) != 2:
            log_and_raise(
                "non_terminal_pose_convergence_tolerance must be a list of 2 floats, "
                + f"got {self.non_terminal_pose_convergence_tolerance}"
            )

        if not isinstance(self.project_distance_to_goal, torch.Tensor):
            if isinstance(self.project_distance_to_goal, bool):
                self.project_distance_to_goal = torch.tensor(
                    [self.project_distance_to_goal],
                    device=self.device_cfg.device,
                    dtype=torch.uint8,
                )
            else:
                log_and_raise(
                    "project_distance_to_goal must be a bool or a torch.Tensor, "
                    + f"got {self.project_distance_to_goal}"
                )

        # copy to device:
        self.terminal_pose_axes_weight_factor = self.device_cfg.to_device(self.terminal_pose_axes_weight_factor)
        self.non_terminal_pose_axes_weight_factor = self.device_cfg.to_device(
            self.non_terminal_pose_axes_weight_factor
        )
        self.terminal_pose_convergence_tolerance = self.device_cfg.to_device(
            self.terminal_pose_convergence_tolerance
        )
        self.non_terminal_pose_convergence_tolerance = self.device_cfg.to_device(
            self.non_terminal_pose_convergence_tolerance
        )

    def clone(self):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=self.terminal_pose_axes_weight_factor.clone(),
            non_terminal_pose_axes_weight_factor=self.non_terminal_pose_axes_weight_factor.clone(),
            terminal_pose_convergence_tolerance=self.terminal_pose_convergence_tolerance.clone(),
            non_terminal_pose_convergence_tolerance=self.non_terminal_pose_convergence_tolerance.clone(),
            project_distance_to_goal=self.project_distance_to_goal.clone(),
            device_cfg=self.device_cfg,
        )

    def copy_(self, other: ToolPoseCriteria):
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")

        if other.terminal_pose_axes_weight_factor is not None:
            self.terminal_pose_axes_weight_factor.copy_(other.terminal_pose_axes_weight_factor)
        if other.non_terminal_pose_axes_weight_factor is not None:
            self.non_terminal_pose_axes_weight_factor.copy_(other.non_terminal_pose_axes_weight_factor)
        if other.terminal_pose_convergence_tolerance is not None:
            self.terminal_pose_convergence_tolerance.copy_(
                other.terminal_pose_convergence_tolerance
            )
        if other.non_terminal_pose_convergence_tolerance is not None:
            self.non_terminal_pose_convergence_tolerance.copy_(
                other.non_terminal_pose_convergence_tolerance
            )
        if other.project_distance_to_goal is not None:
            self.project_distance_to_goal[:] = other.project_distance_to_goal

    @staticmethod
    def track_position(xyz: List[float] = [1.0, 1.0, 1.0]):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0],
            non_terminal_pose_axes_weight_factor=[xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0],
        )

    @staticmethod
    def track_orientation(
        rpy: List[float] = [0.001, 0.001, 0.001], non_terminal_scale: float = 1.0
    ):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, rpy[0], rpy[1], rpy[2]],
            non_terminal_pose_axes_weight_factor=[
                0.0,
                0.0,
                0.0,
                non_terminal_scale * rpy[0],
                non_terminal_scale * rpy[1],
                non_terminal_scale * rpy[2],
            ],
        )

    @staticmethod
    def track_position_and_orientation(
        xyz: List[float] = [1.0, 1.0, 1.0],
        rpy: List[float] = [1.0, 1.0, 1.0],
        non_terminal_scale: float = 0.1,
    ):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]],
            non_terminal_pose_axes_weight_factor=[
                non_terminal_scale * xyz[0],
                non_terminal_scale * xyz[1],
                non_terminal_scale * xyz[2],
                non_terminal_scale * rpy[0],
                non_terminal_scale * rpy[1],
                non_terminal_scale * rpy[2],
            ],
        )
    @staticmethod
    def linear_motion(
        axis: str = "z",
        non_terminal_scale: float = 1.0,
        project_distance_to_goal: bool = True,
    ):
        axis_vector = [0.0, 0.0, 0.0]
        if axis == "x":
            axis_vector[0] = 1.0
        elif axis == "y":
            axis_vector[1] = 1.0
        elif axis == "z":
            axis_vector[2] = 1.0
        else:
            log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            non_terminal_pose_axes_weight_factor=[
                non_terminal_scale * (1 - axis_vector[0]),
                non_terminal_scale * (1 - axis_vector[1]),
                non_terminal_scale * (1 - axis_vector[2]),
                non_terminal_scale * 1.0    ,
                non_terminal_scale * 1.0,
                non_terminal_scale * 1.0,
            ],
            project_distance_to_goal=project_distance_to_goal,
        )

    @staticmethod
    def disabled():
        """Create criteria that disables pose tracking for this tool frame.

        Use this when you want to include a tool frame in the solver but not
        apply any pose cost to it.

        Returns:
            ToolPoseCriteria with all weight factors set to zero.
        """
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            non_terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


@dataclass
class StackedToolPoseCriteria:
    """Stacked link pose criteria.

    Two storage modes:

    * ``per_env=False`` (default): the five per-link tensors are sized
      ``(num_links, K)`` and broadcast across every env in the batch.
      Backwards-compatible with all existing callers.
    * ``per_env=True``: the five per-link tensors are sized
      ``(num_envs, num_links, K)``. Used together with
      :class:`~curobo._src.cost.wp_tool_pose.ToolPoseDistancePerEnv`
      whose kernel reads ``buf[goal_idx * num_links * K + link_idx * K + k]``
      where ``goal_idx = idxs_goal[b_idx]`` (the env index for that batch
      row, see ``GoalRegistry.create_idx`` + ``tensor_repeat_seeds``).
      Per-env mode lets each env disable a different subset of tool
      frames inside a single ``solve_pose`` / ``plan_pose`` call.
    """

    tool_frames: List[str]

    #: Weight factor for the terminal position/orientation. Shape is
    #: ``(num_links, 6)`` when ``per_env=False`` or
    #: ``(num_envs, num_links, 6)`` when ``per_env=True``.
    terminal_pose_axes_weight_factor: torch.Tensor

    #: Weight factor for the non-terminal position/orientation. Same
    #: shape contract as ``terminal_pose_axes_weight_factor``.
    non_terminal_pose_axes_weight_factor: torch.Tensor

    #: Convergence tolerance for the terminal position and orientation.
    #: Shape is ``(num_links, 2)`` or ``(num_envs, num_links, 2)``.
    terminal_pose_convergence_tolerance: torch.Tensor

    #: Convergence tolerance for the non-terminal position and orientation.
    #: Shape is ``(num_links, 2)`` or ``(num_envs, num_links, 2)``.
    non_terminal_pose_convergence_tolerance: torch.Tensor

    #: Project distance to goal flag. Shape is ``(num_links, 1)`` or
    #: ``(num_envs, num_links, 1)``.
    project_distance_to_goal: torch.Tensor

    device_cfg: DeviceCfg = DeviceCfg()

    #: When True, all five per-link tensors above carry an extra leading
    #: ``num_envs`` axis. Auto-set by ``from_tool_pose_criteria`` when
    #: ``num_envs > 0``.
    per_env: bool = False

    #: Number of envs in the per-env layout. Only meaningful when
    #: ``per_env=True``. Equal to ``shape[0]`` of the per-link tensors.
    num_envs: int = 1

    _tool_pose_criteria: Optional[Dict[str, ToolPoseCriteria]] = None

    @staticmethod
    def from_tool_pose_criteria(
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
        num_envs: int = 0,
    ):
        """Build a stacked criteria buffer.

        Args:
            tool_pose_criteria: per-frame criteria dict.
            num_envs: when 0 (default) → per-env mode is OFF, tensors are
                ``(num_links, K)``. When ``> 0`` → per-env mode is ON,
                tensors are ``(num_envs, num_links, K)`` with every env
                row initialized to a copy of the broadcast values
                (callers then call ``update_tool_pose_criteria_per_env``
                to specialise individual env rows).
        """
        tool_frames = list(tool_pose_criteria.keys())
        terminal_pose_axes_weight_factor = torch.stack(
            [tool_pose_criteria[link_name].terminal_pose_axes_weight_factor for link_name in tool_frames]
        )
        non_terminal_pose_axes_weight_factor = torch.stack(
            [
                tool_pose_criteria[link_name].non_terminal_pose_axes_weight_factor
                for link_name in tool_frames
            ]
        )
        terminal_pose_convergence_tolerance = torch.stack(
            [
                tool_pose_criteria[link_name].terminal_pose_convergence_tolerance
                for link_name in tool_frames
            ]
        )
        non_terminal_pose_convergence_tolerance = torch.stack(
            [
                tool_pose_criteria[link_name].non_terminal_pose_convergence_tolerance
                for link_name in tool_frames
            ]
        )
        project_distance_to_goal = torch.stack(
            [tool_pose_criteria[link_name].project_distance_to_goal for link_name in tool_frames]
        )

        per_env = num_envs > 0
        if per_env:
            # Promote each (num_links, K) tensor → (num_envs, num_links, K)
            # by repeating the broadcast row. Use repeat (not expand) so
            # the underlying storage is independent per env — required for
            # in-place per-env writes in update_tool_pose_criteria_per_env.
            def _broadcast(t: torch.Tensor) -> torch.Tensor:
                return t.unsqueeze(0).repeat(
                    num_envs, *[1] * t.ndim
                ).contiguous()

            terminal_pose_axes_weight_factor = _broadcast(terminal_pose_axes_weight_factor)
            non_terminal_pose_axes_weight_factor = _broadcast(
                non_terminal_pose_axes_weight_factor
            )
            terminal_pose_convergence_tolerance = _broadcast(
                terminal_pose_convergence_tolerance
            )
            non_terminal_pose_convergence_tolerance = _broadcast(
                non_terminal_pose_convergence_tolerance
            )
            project_distance_to_goal = _broadcast(project_distance_to_goal)

        return StackedToolPoseCriteria(
            tool_frames=tool_frames,
            terminal_pose_axes_weight_factor=terminal_pose_axes_weight_factor,
            non_terminal_pose_axes_weight_factor=non_terminal_pose_axes_weight_factor,
            terminal_pose_convergence_tolerance=terminal_pose_convergence_tolerance,
            non_terminal_pose_convergence_tolerance=non_terminal_pose_convergence_tolerance,
            project_distance_to_goal=project_distance_to_goal,
            device_cfg=tool_pose_criteria[list(tool_pose_criteria.keys())[0]].device_cfg,
            per_env=per_env,
            num_envs=max(1, num_envs),
            _tool_pose_criteria=tool_pose_criteria,
        )

    def __post_init__(self):
        num_links = len(self.tool_frames)
        if self.per_env:
            E = self.num_envs
            if self.terminal_pose_axes_weight_factor.shape != (E, num_links, 6):
                log_and_raise(
                    f"terminal_pose_axes_weight_factor must be of shape "
                    f"(num_envs={E}, num_links={num_links}, 6), got "
                    f"{self.terminal_pose_axes_weight_factor.shape}"
                )
            if self.non_terminal_pose_axes_weight_factor.shape != (E, num_links, 6):
                log_and_raise(
                    f"non_terminal_pose_axes_weight_factor must be of shape "
                    f"(num_envs={E}, num_links={num_links}, 6), got "
                    f"{self.non_terminal_pose_axes_weight_factor.shape}"
                )
            if self.terminal_pose_convergence_tolerance.shape != (E, num_links, 2):
                log_and_raise(
                    f"terminal_pose_convergence_tolerance must be of shape "
                    f"(num_envs={E}, num_links={num_links}, 2), got "
                    f"{self.terminal_pose_convergence_tolerance.shape}"
                )
            if self.non_terminal_pose_convergence_tolerance.shape != (E, num_links, 2):
                log_and_raise(
                    f"non_terminal_pose_convergence_tolerance must be of shape "
                    f"(num_envs={E}, num_links={num_links}, 2), got "
                    f"{self.non_terminal_pose_convergence_tolerance.shape}"
                )
            if self.project_distance_to_goal.shape != (E, num_links, 1):
                log_and_raise(
                    f"project_distance_to_goal must be of shape "
                    f"(num_envs={E}, num_links={num_links}, 1), got "
                    f"{self.project_distance_to_goal.shape}"
                )
            return

        # per_env=False: stock shape contract.
        if self.terminal_pose_axes_weight_factor.shape != (num_links, 6):
            log_and_raise(
                f"terminal_pose_axes_weight_factor must be of shape (num_links, 6), got {self.terminal_pose_axes_weight_factor.shape}"
            )
        if self.non_terminal_pose_axes_weight_factor.shape != (num_links, 6):
            log_and_raise(
                f"non_terminal_pose_axes_weight_factor must be of shape (num_links, 6), got {self.non_terminal_pose_axes_weight_factor.shape}"
            )
        if self.terminal_pose_convergence_tolerance.shape != (num_links, 2):
            log_and_raise(
                f"terminal_pose_convergence_tolerance must be of shape (num_links, 2), got {self.terminal_pose_convergence_tolerance.shape}"
            )
        if self.non_terminal_pose_convergence_tolerance.shape != (num_links, 2):
            log_and_raise(
                f"non_terminal_pose_convergence_tolerance must be of shape (num_links, 2), got {self.non_terminal_pose_convergence_tolerance.shape}"
            )
        if self.project_distance_to_goal.shape != (num_links, 1):
            log_and_raise(
                f"project_distance_to_goal must be of shape (num_links,1), got {self.project_distance_to_goal.shape}"
            )

    def clone(self) -> StackedToolPoseCriteria:
        return StackedToolPoseCriteria(
            tool_frames=self.tool_frames,
            terminal_pose_axes_weight_factor=self.terminal_pose_axes_weight_factor.clone(),
            non_terminal_pose_axes_weight_factor=self.non_terminal_pose_axes_weight_factor.clone(),
            terminal_pose_convergence_tolerance=self.terminal_pose_convergence_tolerance.clone(),
            non_terminal_pose_convergence_tolerance=self.non_terminal_pose_convergence_tolerance.clone(),
            project_distance_to_goal=self.project_distance_to_goal.clone(),
            device_cfg=self.device_cfg,
            per_env=self.per_env,
            num_envs=self.num_envs,
            _tool_pose_criteria=self._tool_pose_criteria,
        )

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        """Broadcast update: write the same criteria values into every env row.

        In ``per_env=False`` mode this writes the single ``(num_links, K)``
        slice. In ``per_env=True`` mode this writes the same value into all
        env rows (i.e. resets the per-env specialisation back to a common
        baseline).
        """
        for link_name in tool_pose_criteria.keys():
            if link_name not in self.tool_frames:
                log_and_raise(f"link_name {link_name} not found in tool_frames")
            self._tool_pose_criteria[link_name].copy_(tool_pose_criteria[link_name])
            self._update_criteria_in_stack(link_name, tool_pose_criteria[link_name])

    def update_tool_pose_criteria_per_env(
        self,
        env_idx: int,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ):
        """Per-env row update — only valid when ``per_env=True``.

        Writes ``tool_pose_criteria`` into row ``env_idx`` of the per-env
        weight tensors. Other env rows are not touched. Caller-side use:
        ``update_tool_pose_criteria_per_env(env_idx, {frame: ToolPoseCriteria.disabled()})``.
        """
        if not self.per_env:
            log_and_raise(
                "update_tool_pose_criteria_per_env requires per_env=True; "
                "build with from_tool_pose_criteria(criteria, num_envs=N)."
            )
        if env_idx < 0 or env_idx >= self.num_envs:
            log_and_raise(
                f"env_idx={env_idx} out of range [0, {self.num_envs})"
            )
        for link_name, criterion in tool_pose_criteria.items():
            if link_name not in self.tool_frames:
                log_and_raise(f"link_name {link_name} not found in tool_frames")
            self._update_criteria_in_stack(
                link_name, criterion, env_idx=env_idx,
            )

    def _update_criteria_in_stack(
        self,
        link_name: str,
        tool_pose_criteria: ToolPoseCriteria,
        env_idx: Optional[int] = None,
    ):
        """Write a single frame's criterion into the stacked tensors.

        ``env_idx=None`` updates every env row (broadcast) when in per_env
        mode, or the single per-frame slice when not. ``env_idx=k`` writes
        only row ``k`` (per_env mode only).
        """
        link_idx = self.tool_frames.index(link_name)

        def _write_terminal_weight(buf: torch.Tensor, val: torch.Tensor):
            if not self.per_env:
                buf[link_idx, :] = val
                return
            if env_idx is None:
                buf[:, link_idx, :] = val
            else:
                buf[env_idx, link_idx, :] = val

        def _write_proj(buf: torch.Tensor, val: torch.Tensor):
            if not self.per_env:
                buf[link_idx, :] = val
                return
            if env_idx is None:
                buf[:, link_idx, :] = val
            else:
                buf[env_idx, link_idx, :] = val

        if tool_pose_criteria.terminal_pose_axes_weight_factor is not None:
            _write_terminal_weight(
                self.terminal_pose_axes_weight_factor,
                tool_pose_criteria.terminal_pose_axes_weight_factor,
            )
        if tool_pose_criteria.non_terminal_pose_axes_weight_factor is not None:
            _write_terminal_weight(
                self.non_terminal_pose_axes_weight_factor,
                tool_pose_criteria.non_terminal_pose_axes_weight_factor,
            )
        if tool_pose_criteria.terminal_pose_convergence_tolerance is not None:
            _write_terminal_weight(
                self.terminal_pose_convergence_tolerance,
                tool_pose_criteria.terminal_pose_convergence_tolerance,
            )
        if tool_pose_criteria.non_terminal_pose_convergence_tolerance is not None:
            _write_terminal_weight(
                self.non_terminal_pose_convergence_tolerance,
                tool_pose_criteria.non_terminal_pose_convergence_tolerance,
            )
        if tool_pose_criteria.project_distance_to_goal is not None:
            _write_proj(
                self.project_distance_to_goal,
                tool_pose_criteria.project_distance_to_goal,
            )
