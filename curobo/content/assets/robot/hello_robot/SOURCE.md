# Hello Robot Stretch 3 model

Source: https://github.com/hello-robot/stretch_urdf
Commit: `c8c6e27e1f87db126118049f22611399253cbe2b`
URDF: `stretch_urdf/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf`
License: [Clear BSD](LICENSE.md).

Rebuild using `python scripts/hello_robot/prepare_model.py --source PATH_TO_PINNED_REPO`.

Adaptations for MagicSim:
- Preserve upstream geometry, inertia, link transforms and arm joint limits.
- Add a fixed-root planar x/y/yaw virtual base, executed with differential drive controls.
- Fix wheel joints; wheel meshes are cosmetic in this virtual-base simulation.
- Simulation keeps four arm sliders; the action couples them equally. Planning uses
  `joint_arm_l3` plus three 1:1 mimic sliders, retaining the 0.13 m per-segment limit.
- Preserve `link_grasp_center` as the TCP.
- Upstream SG3 URDF has zero finger limits. Use the conservative [0, 0.25] rad
  animation range provided by upstream `tools/stretch_urdf_viz.py:197-198`;
  effort=5 and velocity=1 are simulation settings, not claimed hardware ratings.
- Planning locks head and gripper at zero; Reach keeps the gripper closed.
- Collision spheres are deterministic collision-mesh AABB subdivisions, not calibrated
  fits. Cosmetic markers/IMU/wheel spheres are omitted and base spheres clipped above
  z=0.005 m. Rigid, neighboring and nested telescoping components are ignored for
  self collision. This initial approximation requires simulation validation.

`model_metadata.json` records legal source-joint configurations and CPU-FK target poses
(quaternion wxyz) for the bounded Reach checks.
