# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim helpers shared by the ``examples/isaacsim`` samples.

Intentionally tiny: loads a robot URDF/USD into the open stage, enables the
livestream extensions when ``--headless_mode`` is passed, and exposes a small
stage-sync helper that pulls obstacles off ``/World`` as a
:class:`curobo.scene.Scene` (equivalent to v1's
``UsdHelper.get_obstacles_from_stage``).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from omni.isaac.core import World

try:  # Isaac Sim 4.5+
    from isaacsim.core.api.robots import Robot
except ImportError:
    from omni.isaac.core.robots import Robot

try:  # Isaac Sim 4.5+
    from isaacsim.asset.importer.urdf import _urdf
    _ISAAC_SIM_45 = True
except ImportError:
    try:
        from omni.importer.urdf import _urdf  # Isaac Sim 2023.1+
    except ImportError:
        from omni.isaac.urdf import _urdf      # Isaac Sim 2022.2
    _ISAAC_SIM_45 = False

from omni.isaac.core.utils.extensions import enable_extension

from curobo.config_io import get_filename, get_path_of_dir, join_path
from curobo.content import get_assets_path
from curobo.logging import log_warn
from curobo.scene import Scene
from curobo.viewer import UsdWriter

# Only USD helper not re-exported publicly in v2: transform-op setter used to
# position the just-imported URDF root. Pull it from the shared _src utility
# module. UsdWriter itself comes from the public ``curobo.viewer`` namespace.
from curobo._src.util.usd_util import set_prim_transform


def add_extensions(simulation_app, headless_mode: Optional[str] = None) -> bool:
    """Enable the asset-importer extensions (plus livestream if headless).

    Extension-enable failures are non-fatal: Isaac Sim 5.1 has reshuffled some
    extension names (``omni.kit.livestream.native`` was removed in favour of
    streaming bundles), so we log a warning and carry on rather than abort.
    """
    core_ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    optional_ext_list: list[str] = []
    if headless_mode is not None:
        log_warn(f"Running in headless mode: {headless_mode}")
        optional_ext_list.append(f"omni.kit.livestream.{headless_mode}")

    for ext in core_ext_list:
        try:
            enable_extension(ext)
        except Exception as e:  # noqa: BLE001  (Kit raises bare Exception for missing ext)
            log_warn(f"Optional extension '{ext}' could not be enabled: {e}")

    for ext in optional_ext_list:
        try:
            enable_extension(ext)
        except Exception as e:  # noqa: BLE001
            log_warn(
                f"Livestream extension '{ext}' not available in this Isaac Sim install "
                f"(ok for a headless smoke test): {e}"
            )

    simulation_app.update()
    return True


def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.ndarray = np.array([0, 0, 0]),
    initialize_world: bool = True,
) -> Tuple[Robot, str]:
    """Import the robot from its URDF and attach it to *my_world*.

    The ``robot_config`` dict is the ``robot_cfg`` block loaded from the v2
    YAML (same layout as v1).  Returns ``(robot, robot_prim_path)``.
    """
    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = str(get_assets_path())
    ext_asset = robot_config.get("kinematics", {}).get("external_asset_path")
    if ext_asset is not None:
        asset_path = ext_asset

    urdf_rel = robot_config["kinematics"]["urdf_path"]
    full_path = join_path(asset_path, urdf_rel)
    robot_root = get_path_of_dir(full_path)
    filename = get_filename(full_path)

    if _ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name  # noqa: F401
        import omni.kit.commands
        import omni.usd

        dest_path = join_path(
            robot_root, get_filename(filename, remove_extension=True) + "_temp.usd"
        )
        _result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=f"{robot_root}/{filename}",
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:
        imported_robot = urdf_interface.parse_urdf(robot_root, filename, import_config)
        robot_path = urdf_interface.import_robot(
            robot_root, filename, imported_robot, import_config, subroot,
        )

    base_link_name = robot_config["kinematics"]["base_link"]
    robot_p = Robot(prim_path=f"{robot_path}/{base_link_name}", name=robot_name)

    stage = robot_p.prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world and _ISAAC_SIM_45:
        my_world.initialize_physics()
        robot.initialize()
    return robot, robot_path


def stage_obstacles_as_scene(
    stage,
    reference_prim_path: str,
    only_paths=("/World",),
    ignore_substring=None,
) -> Scene:
    """Parse obstacles from the open USD stage into a v2 :class:`Scene`.

    v2 equivalent of v1's
    ``UsdHelper.get_obstacles_from_stage(...).get_collision_check_world()``;
    uses the public :class:`curobo.viewer.UsdWriter` API (its
    ``get_obstacles_from_stage`` method is the same parser used internally).
    Returns a :class:`curobo.scene.Scene` suitable for
    ``ik.update_world(scene)``.
    """
    reader = UsdWriter()
    reader.load_stage(stage)
    return reader.get_obstacles_from_stage(
        only_paths=list(only_paths),
        ignore_substring=list(ignore_substring) if ignore_substring else None,
        reference_prim_path=reference_prim_path,
    )
