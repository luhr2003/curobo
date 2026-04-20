# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Viser viewer for retargeted humanoid motion CSV files.

Reads a SOMA CSVAnimationBuffer-format CSV (as written by
``humanoid_retargeting.py``) and plays it back in a Viser web viewer using
the same Unitree G1 cuRobo robot config.

Usage:
    python -m curobo.examples.getting_started.viser_csv_viewer \
        --input /tmp/Neutral_walk_forward_002.csv

    python -m curobo.examples.getting_started.viser_csv_viewer \
        --input /tmp/csv_folder/ --fps 120
"""

from __future__ import annotations

import argparse
import csv as csv_lib
import pathlib
import time
from typing import List, Tuple

import numpy as np
import torch

from curobo.types import ContentPath, JointState
from curobo.viewer import ViserVisualizer


VIRTUAL_BASE_JOINTS = [
    "base_j_x", "base_j_y", "base_j_z",
    "base_j_xtheta", "base_j_ytheta", "base_j_ztheta",
]


def load_csv(path: pathlib.Path) -> Tuple[np.ndarray, List[str]]:
    """Load a SOMA-format retargeted CSV.

    Returns:
        positions: (num_frames, 35) array in cuRobo conventions — translation in
            meters, rotations in radians, body joints in SOMA CSV order.
        joint_names: length-35 list of joint names matching the columns.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader], dtype=np.float64)

    body_headers = header[7:]
    body_joint_names = [h.replace("_dof", "") for h in body_headers]
    joint_names = VIRTUAL_BASE_JOINTS + body_joint_names

    num_frames = rows.shape[0]
    positions = np.zeros((num_frames, 6 + len(body_joint_names)), dtype=np.float32)
    positions[:, 0:3] = rows[:, 1:4] * 0.01                 # cm -> m
    positions[:, 3:6] = np.deg2rad(rows[:, 4:7])            # deg -> rad
    positions[:, 6:] = np.deg2rad(rows[:, 7:])              # deg -> rad
    return positions, joint_names


def visualize(
    csv_paths: List[pathlib.Path],
    robot_config: str,
    fps: float,
    port: int,
    default_frame_skip: int = 1,
):
    motions = [load_csv(p) for p in csv_paths]
    motion_names = [p.stem for p in csv_paths]

    joint_names = motions[0][1]
    for _, names in motions[1:]:
        if names != joint_names:
            raise ValueError("CSV files have inconsistent joint columns.")

    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_config),
        add_robot_to_scene=True,
        connect_port=port,
        add_control_frames=False,
    )
    server = viz._server

    n_motions = len(motions)
    current_motion = [0]
    positions = [motions[0][0]]
    num_frames = [positions[0].shape[0]]

    playing = [True]
    looping = [True]
    speed = [1.0]
    current_frame = [0]
    skip = [max(1, default_frame_skip)]
    frame_dt = 1.0 / fps

    with server.gui.add_folder("Playback"):
        if n_motions > 1:
            gui_motion = server.gui.add_dropdown(
                "Motion", options=motion_names, initial_value=motion_names[0]
            )
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=num_frames[0] - 1, step=skip[0], initial_value=0
        )
        gui_play = server.gui.add_button("Pause")
        gui_loop = server.gui.add_checkbox("Loop", initial_value=True)
        gui_speed = server.gui.add_slider(
            "Speed", min=0.1, max=20.0, step=0.1, initial_value=1.0
        )
        gui_skip = server.gui.add_slider(
            "Frame Skip", min=1, max=100, step=1, initial_value=skip[0]
        )

    def _set_frame(idx: int):
        q = torch.as_tensor(
            positions[0][idx], dtype=torch.float32, device="cuda:0"
        ).unsqueeze(0)
        js = JointState.from_position(q, joint_names=joint_names)
        viz.set_joint_state(js)

    def _switch_motion(idx: int):
        current_motion[0] = idx
        positions[0] = motions[idx][0]
        num_frames[0] = positions[0].shape[0]
        current_frame[0] = 0
        gui_frame.max = num_frames[0] - 1
        gui_frame.value = 0
        _set_frame(0)

    if n_motions > 1:
        @gui_motion.on_update
        def _on_motion(_):
            _switch_motion(motion_names.index(gui_motion.value))

    @gui_play.on_click
    def _on_play(_):
        playing[0] = not playing[0]
        gui_play.name = "Pause" if playing[0] else "Play"

    @gui_loop.on_update
    def _on_loop(_):
        looping[0] = gui_loop.value

    @gui_speed.on_update
    def _on_speed(_):
        speed[0] = gui_speed.value

    @gui_skip.on_update
    def _on_skip(_):
        skip[0] = max(1, int(gui_skip.value))
        gui_frame.step = skip[0]

    @gui_frame.on_update
    def _on_frame(_):
        if not playing[0]:
            current_frame[0] = int(gui_frame.value)
            _set_frame(current_frame[0])

    print(f"\n[INFO] Viewer at http://localhost:{port}")
    print(f"[INFO] {n_motions} motion(s) loaded, playing at {fps} FPS")
    print("[INFO] Press Ctrl+C to exit\n")

    _set_frame(0)
    try:
        while True:
            if playing[0]:
                _set_frame(current_frame[0])
                gui_frame.value = current_frame[0]
                current_frame[0] += skip[0]
                if current_frame[0] >= num_frames[0]:
                    if looping[0]:
                        current_frame[0] = 0
                    else:
                        current_frame[0] = num_frames[0] - 1
                        playing[0] = False
                        gui_play.name = "Play"
                time.sleep(frame_dt * skip[0] / speed[0])
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Viewer stopped.")


def _collect_csv_paths(input_path: pathlib.Path) -> List[pathlib.Path]:
    if input_path.is_dir():
        return sorted(input_path.rglob("*.csv"))
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=pathlib.Path, required=True,
        help="Path to a retargeted CSV file or a directory of CSV files."
    )
    parser.add_argument(
        "--robot-config", default="unitree_g1_29dof_retarget.yml",
        help="cuRobo robot YAML config name or path."
    )
    parser.add_argument("--fps", type=float, default=120.0, help="Playback FPS.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--frame-skip", type=int, default=1,
        help="Initial frame skip. Raise to play high-FPS MPC trajectories "
             "at the original input rate."
    )
    args = parser.parse_args()

    csv_paths = _collect_csv_paths(args.input)
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {args.input}")

    visualize(
        csv_paths=csv_paths,
        robot_config=args.robot_config,
        fps=args.fps,
        port=args.port,
        default_frame_skip=args.frame_skip,
    )


if __name__ == "__main__":
    main()
