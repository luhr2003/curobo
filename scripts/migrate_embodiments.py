#!/usr/bin/env python3
"""Parallel orchestrator: build v2 yamls for every MagicSim embodiment.

CPU-parallel (thread pool, ~8 workers):
    - Run ``curobo.examples.getting_started.build_robot_model`` on each
      fixed-base URDF to produce an initial v2 yaml in
      ``/tmp/curobo_migrate/<name>/fixed.yml``.
    - Port v1 metadata (``lock_joints``, ``cspace.default_joint_position``,
      etc.) from ``MagicCurobo/.../magicsim_<name>.yml`` on top.
    - Derive the ``_mobile`` variant (if v1 ships one) by copying the
      fixed-base spheres + ignore matrix and adjusting ``urdf_path`` /
      ``base_link`` / cspace joint list.

GPU-serial (single worker):
    - Run the ``motion_planning_crossembody`` smoke test per robot and
      record pass/fail.

All writes stay under ``/home/magics/magicsim/curobo/`` and ``/tmp/``. No
MagicCurobo paths are modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CUROBO_ROOT = Path("/home/magics/magicsim/curobo")
MAGICCUROBO_CFG = Path(
    "/home/magics/magicsim/MagicCurobo/src/curobo/content/configs/robot"
)
ASSETS_ROBOT = CUROBO_ROOT / "curobo" / "content" / "assets" / "robot"
CONFIGS_ROBOT = CUROBO_ROOT / "curobo" / "content" / "configs" / "robot"
SCRATCH = Path("/tmp/curobo_migrate")
PY = "/home/magics/magicsim/curobo/.venv/bin/python"

SCRATCH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embodiment registry
# ---------------------------------------------------------------------------


@dataclass
class Embodiment:
    name: str                          # yaml stem (no magicsim_ prefix)
    asset_dir: str                     # dir under content/assets/robot/
    fixed_urdf: Optional[str]          # relative to asset_dir
    tool_frames: List[str]             # EE links for --tool-frames
    v1_fixed_yaml: Optional[str] = None   # under MAGICCUROBO_CFG
    v1_left_yaml: Optional[str] = None    # dual-arm left/right merge
    v1_right_yaml: Optional[str] = None
    mobile_urdf: Optional[str] = None     # triggers mobile variant creation
    v1_mobile_yaml: Optional[str] = None  # metadata source for mobile yaml
    mobile_base_link: Optional[str] = None
    base_link_override: Optional[str] = None   # pass to builder if urdf root needs override
    clip_base_z: bool = True           # emit --clip-link <base> z 0.0
    sphere_density: float = 1.0
    protrusion_weight: Optional[float] = None
    extra_args: List[str] = field(default_factory=list)


EMBODIMENTS: List[Embodiment] = [
    Embodiment(
        name="arx_x5",
        asset_dir="arx_x5_description",
        fixed_urdf="urdf/X5A.urdf",
        tool_frames=["link6"],
        v1_fixed_yaml="magicsim_arx_x5.yml",
    ),
    Embodiment(
        name="dual_arx_x5",
        asset_dir="dual_arx_x5_description",
        fixed_urdf="urdf/dual_arx_x5.urdf",
        tool_frames=["L_link6", "R_link6"],
        v1_left_yaml="magicsim_dual_arx_x5_left.yml",
        v1_right_yaml="magicsim_dual_arx_x5_right.yml",
    ),
    Embodiment(
        name="dual_piper",
        asset_dir="dual_piper_description",
        fixed_urdf="urdf/dual_piper.urdf",
        # tool frames filled in after URDF inspection (may be L_link6/R_link6 pattern)
        tool_frames=["L_link6", "R_link6"],
        v1_left_yaml="magicsim_dual_piper_left.yml",
        v1_right_yaml="magicsim_dual_piper_right.yml",
    ),
    Embodiment(
        name="dual_so101",
        asset_dir="dual_so101",
        fixed_urdf="dual_so101.urdf",
        tool_frames=["L_SO101_gripper", "R_SO101_gripper"],
        v1_left_yaml="magicsim_dual_so101_left.yml",
        v1_right_yaml="magicsim_dual_so101_right.yml",
    ),
    Embodiment(
        name="franka_umi",
        asset_dir="franka_description",
        fixed_urdf="franka_umi.urdf",
        tool_frames=["panda_hand"],
        v1_fixed_yaml="magicsim_franka_umi.yml",
    ),
    Embodiment(
        # frankarobotiq.urdf sits at robot/ top level and references
        # 2f_85_urdf/ + 2f_85_mount/ + franka_description/, so asset_path
        # needs to be robot/. Use a synthetic asset_dir="" and let
        # _asset_path_for fall through.
        name="frankarobotiq",
        asset_dir="",
        fixed_urdf="frankarobotiq.urdf",
        tool_frames=["panda_link8"],
        v1_fixed_yaml="magicsim_frankarobotiq.yml",
    ),
    Embodiment(
        name="genie1",
        asset_dir="genie1",
        fixed_urdf="genie1.urdf",
        tool_frames=["arm_l_end_link", "arm_r_end_link"],
        v1_fixed_yaml="magicsim_genie1.yml",
        v1_left_yaml="magicsim_genie1_left.yml",
        v1_mobile_yaml="magicsim_genie1_mobile.yml",
        # genie1 mobile uses extra_links instead of a separate URDF
        mobile_urdf=None,  # handled via extra_links port
    ),
    Embodiment(
        name="lift2_mobile",
        asset_dir="lift2",
        fixed_urdf="lift2_mobile.urdf",  # mobile-only per v1
        tool_frames=["L_ee", "R_ee"],
        v1_fixed_yaml="magicsim_lift2_mobile.yml",
        clip_base_z=False,
    ),
    Embodiment(
        name="mobile_x7s",
        asset_dir="x7s_mobile",
        fixed_urdf="x7s_mobile_fix.urdf",
        tool_frames=["link11_tip", "link20_tip"],
        v1_fixed_yaml="magicsim_mobile_x7s.yml",
        v1_left_yaml="magicsim_mobile_x7s_left.yml",
        mobile_urdf="x7s_mobile.urdf",
        v1_mobile_yaml="magicsim_mobile_x7s_mobile.yml",
    ),
    Embodiment(
        name="piper_x",
        asset_dir="piper_x_description",
        fixed_urdf="urdf/piper_x_description.urdf",
        tool_frames=["link6"],
        v1_fixed_yaml="magicsim_piper_x.yml",
    ),
    Embodiment(
        name="ridgebacksawyer",
        asset_dir="sawyer",
        fixed_urdf="ridgeback_sawyer_fix.urdf",
        tool_frames=["right_l6"],
        v1_fixed_yaml="magicsim_ridgebacksawyer.yml",
        mobile_urdf="ridgeback_sawyer.urdf",
        v1_mobile_yaml="magicsim_ridgebacksawyer_mobile.yml",
    ),
    Embodiment(
        name="so101",
        asset_dir="so101",
        fixed_urdf="SO101.urdf",
        tool_frames=["SO101_gripper"],
        v1_fixed_yaml="magicsim_so101.yml",
    ),
    Embodiment(
        name="split_aloha",
        asset_dir="split_aloha",
        fixed_urdf="split_aloha.urdf",
        tool_frames=["L_ee", "R_ee"],
        v1_fixed_yaml="magicsim_split_aloha.yml",
        v1_left_yaml="magicsim_split_aloha_left.yml",
        v1_mobile_yaml="magicsim_split_aloha_mobile.yml",
        mobile_urdf=None,  # extra_links-based mobile
    ),
    Embodiment(
        name="vega1p",
        asset_dir="vega_1p",
        fixed_urdf="vega_1p_fixed.urdf",
        tool_frames=["L_ee", "R_ee"],
        v1_fixed_yaml="magicsim_vega1p.yml",
        mobile_urdf="vega_1p.urdf",
        v1_mobile_yaml="magicsim_vega1p_mobile.yml",
    ),
    Embodiment(
        name="vega1p_sharpa",
        asset_dir="vega_1p_sharpa",
        fixed_urdf="vega_1p_sharpa_fix.urdf",
        tool_frames=["L_ee", "R_ee"],
        v1_fixed_yaml="magicsim_vega1p_sharpa.yml",
        v1_left_yaml="magicsim_vega1p_sharpa_left.yml",
        mobile_urdf="vega_1p_sharpa.urdf",
        v1_mobile_yaml="magicsim_vega1p_sharpa_mobile.yml",
    ),
    Embodiment(
        name="xtrainer",
        asset_dir="xtrainer",
        fixed_urdf="urdf/xtrainer.urdf",
        tool_frames=["J1_6", "J2_6"],
        v1_fixed_yaml="magicsim_xtrainer.yml",
    ),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _read_v1_yaml(stem: str) -> Optional[dict]:
    path = MAGICCUROBO_CFG / stem
    if not path.is_file():
        return None
    try:
        with path.open() as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as e:
        print(f"[warn] malformed v1 yaml {stem}: {e}", file=sys.stderr)
        return None


def _resolve_fixed_urdf(emb: Embodiment) -> Optional[str]:
    """Resolve the fixed URDF path under the v2 asset tree."""
    if emb.fixed_urdf:
        if emb.asset_dir:
            return str(ASSETS_ROBOT / emb.asset_dir / emb.fixed_urdf)
        return str(ASSETS_ROBOT / emb.fixed_urdf)
    # fallback: pick the first *.urdf in urdf/ or asset_dir root
    asset_dir = ASSETS_ROBOT / emb.asset_dir
    for sub in ["urdf", ""]:
        for p in sorted((asset_dir / sub).glob("*.urdf")):
            return str(p)
    return None


def _asset_path_for(emb: Embodiment, urdf: str) -> str:
    """Pick the right asset-path for build_robot_model.

    - URDF uses ``package://<pkg>/…`` → strip prefix → needs parent of <pkg>.
    - URDF uses ``../meshes/…`` (relative) → needs the URDF's own dir.
    - ``asset_dir`` empty → URDF at the robot/ root → use ASSETS_ROBOT.
    - Otherwise → asset_dir.
    """
    try:
        text = open(urdf).read(65536)
    except Exception:  # noqa: BLE001
        text = ""
    import re
    if not emb.asset_dir:
        return str(ASSETS_ROBOT)
    m = re.search(r"package://([^/]+)/", text)
    if m and m.group(1) == emb.asset_dir:
        return str(ASSETS_ROBOT)
    if re.search(r'filename="\.\./', text):
        return str(Path(urdf).parent)
    return str(ASSETS_ROBOT / emb.asset_dir)


def _run_builder(emb: Embodiment, out_yaml: Path, log_path: Path) -> bool:
    urdf = _resolve_fixed_urdf(emb)
    if urdf is None:
        log_path.write_text(f"No URDF resolved for {emb.name}\n")
        return False
    asset_path = _asset_path_for(emb, urdf)

    cmd = [
        PY, "-m", "curobo.examples.getting_started.build_robot_model",
        "--urdf", urdf,
        "--asset-path", asset_path,
        "--output", str(out_yaml),
        "--seed", "42",
        "--sphere-density", str(emb.sphere_density),
        "--num-collision-samples", "200",
    ]
    if emb.tool_frames:
        cmd += ["--tool-frames", *emb.tool_frames]
    if emb.protrusion_weight is not None:
        cmd += ["--protrusion-weight", str(emb.protrusion_weight)]
    cmd += emb.extra_args

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_env())
    return rc == 0 and out_yaml.is_file()


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    # Make sure PYTHONPATH picks up the curobo src first
    env.setdefault("PYTHONPATH", str(CUROBO_ROOT))
    env["CUROBO_ASSETS_PATH"] = str(ASSETS_ROBOT.parent / "assets")
    env["CUROBO_ROBOT_CONFIGS_PATH"] = str(CONFIGS_ROBOT)
    return env


def _v2_kin(cfg: dict) -> dict:
    """Return the kinematics block of a v2 cfg.

    Handles both top-level ``kinematics:`` (builder output) and wrapped
    ``robot_cfg: kinematics:`` (canonical v2 style). Does NOT create a
    stray ``robot_cfg`` wrapper if one isn't present already."""
    if "kinematics" in cfg:
        return cfg["kinematics"]
    if "robot_cfg" in cfg:
        return cfg["robot_cfg"].setdefault("kinematics", {})
    # Only called with a config that already has one of those keys; create
    # top-level kinematics for a fresh builder output.
    return cfg.setdefault("kinematics", {})


def _merge_v1_metadata(v2_cfg: dict, emb: Embodiment) -> dict:
    """Overlay lock_joints / default_joint_position / extra_links from v1.

    Also auto-locks every URDF joint that is NOT in the union of v1's active
    cspace (so passive joints like head_pan, wheels, or fingers without an
    explicit lock are held at their v1 retract value). This keeps v2 planning
    scoped to the same DoFs the v1 fork used.
    """
    v1_stems = [
        s for s in (emb.v1_fixed_yaml, emb.v1_left_yaml, emb.v1_right_yaml)
        if s
    ]
    v2_kin = _v2_kin(v2_cfg)
    v2_cspace = v2_kin.setdefault("cspace", {})
    v2_joint_names: List[str] = list(v2_cspace.get("joint_names") or [])

    v1_active_joints: set = set()
    v1_retract_map: Dict[str, float] = {}
    v1_lock_joints: Dict[str, float] = {}

    for stem in v1_stems:
        v1 = _read_v1_yaml(stem)
        if v1 is None:
            continue
        v1_kin = ((v1.get("robot_cfg") or {}).get("kinematics")
                  or v1.get("kinematics") or {})
        v1_cspace = v1_kin.get("cspace", {}) or {}
        jn = list(v1_cspace.get("joint_names") or [])
        v1_active_joints.update(jn)
        # v1 uses either retract_config (old) or default_joint_position.
        retract = v1_cspace.get("retract_config") or v1_cspace.get(
            "default_joint_position"
        )
        if isinstance(retract, list) and len(retract) == len(jn):
            for name, val in zip(jn, retract):
                v1_retract_map.setdefault(name, float(val))
        # For L/R collapsed embodiments, v1_left_yaml locks the R arm and
        # vice-versa at their "parked" pose. We don't want to lock those in
        # the collapsed yaml (both arms should be free), BUT the park
        # values are the right starting pose for the opposite arm to avoid
        # self-collision. Fold them into the retract map so the v2 default
        # matches a known good arm-away-from-arm configuration.
        side_yaml = stem in (emb.v1_left_yaml, emb.v1_right_yaml)
        if v1_kin.get("lock_joints"):
            if not side_yaml:
                v1_lock_joints.update(v1_kin["lock_joints"])
            else:
                for name, val in v1_kin["lock_joints"].items():
                    v1_retract_map.setdefault(name, float(val))

    # Port lock_joints from v1 directly
    if v1_lock_joints:
        existing = v2_kin.get("lock_joints") or {}
        existing = dict(existing) if isinstance(existing, dict) else {}
        existing.update(v1_lock_joints)
        v2_kin["lock_joints"] = existing

    # Auto-lock passive URDF joints (in v2 cspace.joint_names but not in any
    # v1 active cspace). Value = v1 retract value if known, else 0.0.
    if v1_active_joints:
        auto_locks: Dict[str, float] = dict(v2_kin.get("lock_joints") or {})
        for jn in v2_joint_names:
            if jn in v1_active_joints:
                continue
            if jn in auto_locks:
                continue
            auto_locks[jn] = v1_retract_map.get(jn, 0.0)
        if auto_locks:
            v2_kin["lock_joints"] = auto_locks

    # Port default_joint_position only after locks are settled so length
    # matching is done against v2's remaining active joint list.
    if emb.v1_fixed_yaml:
        v1 = _read_v1_yaml(emb.v1_fixed_yaml)
        if v1:
            v1_kin = ((v1.get("robot_cfg") or {}).get("kinematics")
                      or v1.get("kinematics") or {})
            v1_cspace = v1_kin.get("cspace", {})
            dj = v1_cspace.get("default_joint_position") or \
                 v1_cspace.get("retract_config")
            jn = v1_cspace.get("joint_names")
            if dj and jn and len(dj) == len(jn):
                # Rebuild v2 default from per-joint map (align to v2 order).
                joint_to_val = {n: v for n, v in zip(jn, dj)}
                v2_dj = []
                for j in v2_joint_names:
                    if j in joint_to_val:
                        v2_dj.append(float(joint_to_val[j]))
                    else:
                        # keep existing default if present, else 0
                        existing = v2_cspace.get("default_joint_position") or []
                        try:
                            v2_dj.append(float(existing[v2_joint_names.index(j)]))
                        except Exception:
                            v2_dj.append(0.0)
                v2_cspace["default_joint_position"] = v2_dj
    return v2_cfg


_ASSETS_ROBOT_ABS = str(ASSETS_ROBOT) + "/"


def _relativize_paths(kin: dict) -> None:
    """Rewrite urdf_path / asset_root_path to content-relative (``robot/…``)
    so yamls match the style of the bundled v2 configs and don't hard-code
    absolute paths."""
    u = kin.get("urdf_path")
    if isinstance(u, str) and u.startswith(_ASSETS_ROBOT_ABS):
        kin["urdf_path"] = "robot/" + u[len(_ASSETS_ROBOT_ABS):]
    arp = kin.get("asset_root_path")
    if isinstance(arp, str) and arp.startswith(_ASSETS_ROBOT_ABS):
        kin["asset_root_path"] = "robot/" + arp[len(_ASSETS_ROBOT_ABS):]
    # Re-derive asset_root_path from urdf_path when it's wrong (e.g. still
    # pointing at the builder's asset-path hack).
    u = kin.get("urdf_path", "")
    if isinstance(u, str) and u.startswith("robot/"):
        parts = u.split("/")
        if len(parts) >= 3:
            kin["asset_root_path"] = f"{parts[0]}/{parts[1]}"
        elif len(parts) == 2:
            kin["asset_root_path"] = "robot"


def _wrap_robot_cfg(cfg: dict) -> dict:
    """Wrap a top-level-`kinematics` cfg under `robot_cfg:` (v2 canonical
    style, matches bundled ridgebackfranka_mobile.yml)."""
    if "robot_cfg" in cfg:
        return cfg
    if "kinematics" not in cfg:
        return cfg
    kin = cfg.pop("kinematics")
    return {"robot_cfg": {"kinematics": kin, "load_dynamics": False}}


def do_build_fixed(emb: Embodiment) -> Dict[str, object]:
    work = SCRATCH / emb.name
    work.mkdir(parents=True, exist_ok=True)
    fixed_out = work / "fixed.yml"
    log = work / "build_fixed.log"

    ok = _run_builder(emb, fixed_out, log)
    if not ok:
        return {"step": "build_fixed", "ok": False, "log": str(log)}

    # Merge v1 metadata
    with fixed_out.open() as fh:
        cfg = yaml.safe_load(fh)
    # Builder emits runtime-only fields (load_collision_spheres, num_envs)
    # into the yaml. They are passed as kwargs during create(), so leaving
    # them in the yaml produces "multiple values for keyword" at load time.
    kin = _v2_kin(cfg)
    for runtime_only in ("load_collision_spheres", "num_envs"):
        kin.pop(runtime_only, None)
    cfg = _merge_v1_metadata(cfg, emb)
    _relativize_paths(_v2_kin(cfg))
    cfg = _wrap_robot_cfg(cfg)

    final_yaml = CONFIGS_ROBOT / f"magicsim_{emb.name}.yml"
    with final_yaml.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return {"step": "build_fixed", "ok": True, "yaml": str(final_yaml)}


def do_build_mobile(emb: Embodiment) -> Dict[str, object]:
    """Derive <name>_mobile.yml from the fixed yaml + v1 mobile cspace."""
    if not (emb.mobile_urdf or emb.v1_mobile_yaml):
        return {"step": "build_mobile", "ok": True, "skipped": "no mobile"}

    fixed_yaml = CONFIGS_ROBOT / f"magicsim_{emb.name}.yml"
    if not fixed_yaml.is_file():
        return {"step": "build_mobile", "ok": False, "error": "fixed yaml missing"}
    with fixed_yaml.open() as fh:
        base = yaml.safe_load(fh)

    mobile = copy.deepcopy(base)
    # Strip any stray robot_cfg wrapper; write everything under top-level kinematics
    mobile.pop("robot_cfg", None)
    kin = _v2_kin(mobile)
    v1_mobile = _read_v1_yaml(emb.v1_mobile_yaml) if emb.v1_mobile_yaml else None
    if v1_mobile is None:
        return {"step": "build_mobile", "ok": False, "error": "v1 mobile yaml not found"}

    v1m_kin = (v1_mobile.get("robot_cfg") or {}).get("kinematics") or \
              v1_mobile.get("kinematics") or {}

    # urdf_path + base_link. Use absolute path (matches the fixed yaml style
    # produced by build_robot_model).
    if emb.mobile_urdf:
        kin["urdf_path"] = str(ASSETS_ROBOT / emb.asset_dir / emb.mobile_urdf)
    if "base_link" in v1m_kin:
        kin["base_link"] = v1m_kin["base_link"]
    # extra_links: prefer v1 mobile's (it may inject the virtual base)
    if v1m_kin.get("extra_links"):
        kin["extra_links"] = v1m_kin["extra_links"]

    # Port the v1 mobile cspace, converting to v2 schema on the way:
    #   retract_config → default_joint_position
    #   scalar max_acceleration/max_jerk → per-joint list
    #   fill missing acceleration_scale/velocity_scale/jerk_scale/
    #   null_space_maximum_distance with 1.0 per joint.
    v1m_cspace = v1m_kin.get("cspace")
    if v1m_cspace:
        v1m_cspace = dict(v1m_cspace)
        jn = list(v1m_cspace.get("joint_names") or [])
        n = len(jn)
        if "retract_config" in v1m_cspace and "default_joint_position" not in v1m_cspace:
            v1m_cspace["default_joint_position"] = v1m_cspace.pop("retract_config")
        else:
            v1m_cspace.pop("retract_config", None)
        for key in ("max_acceleration", "max_jerk"):
            val = v1m_cspace.get(key)
            if isinstance(val, (int, float)):
                v1m_cspace[key] = [float(val)] * n
        # Fill per-joint scale vectors if absent
        for key, default in (
            ("acceleration_scale", 1.0),
            ("velocity_scale", 1.0),
            ("jerk_scale", 1.0),
            ("null_space_maximum_distance", 1.0),
        ):
            if key not in v1m_cspace:
                v1m_cspace[key] = [default] * n
        # position_limit_clip as scalar is fine (v2 accepts float)
        kin["cspace"] = v1m_cspace

    # lock_joints. Filter out joints that v1 referenced but that the v2
    # parser can't see (fixed joints in the URDF — wheels, steering, grippers
    # beyond the arm). Anything that isn't in v2 cspace.joint_names won't
    # appear in joint_data and triggers a KeyError during lock resolution.
    v2_joint_set = set(kin.get("cspace", {}).get("joint_names") or [])
    if v1m_kin.get("lock_joints"):
        filtered = {k: v for k, v in v1m_kin["lock_joints"].items()
                    if k in v2_joint_set or k in kin.get("cspace", {}).get("joint_names", [])}
        # If filter dropped everything, keep an empty dict (not None).
        kin["lock_joints"] = filtered

    _relativize_paths(kin)
    # Normalize to robot_cfg wrapper (matching bundled v2 yamls).
    if "kinematics" in mobile and "robot_cfg" not in mobile:
        mobile = _wrap_robot_cfg(mobile)
    elif "kinematics" in mobile and "robot_cfg" in mobile:
        mobile.pop("kinematics", None)

    out = CONFIGS_ROBOT / f"magicsim_{emb.name}_mobile.yml"
    with out.open("w") as fh:
        yaml.safe_dump(mobile, fh, sort_keys=False)
    return {"step": "build_mobile", "ok": True, "yaml": str(out)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Run only this embodiment (by name, e.g. arx_x5)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stage", choices=["build", "mobile", "both"],
                    default="both")
    args = ap.parse_args()

    targets = [e for e in EMBODIMENTS if not args.only or e.name == args.only]

    results: List[Dict[str, object]] = []
    started = time.time()

    # CPU-parallel build_fixed
    if args.stage in ("build", "both"):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(do_build_fixed, e): e for e in targets}
            for fut in as_completed(futs):
                emb = futs[fut]
                res = fut.result()
                res["embodiment"] = emb.name
                print(f"[build_fixed] {emb.name:24s} {'OK' if res.get('ok') else 'FAIL'}"
                      f"  {res.get('log','') or res.get('error','')}")
                results.append(res)

    # Mobile derivation (cheap, serial is fine)
    if args.stage in ("mobile", "both"):
        for emb in targets:
            if not (emb.mobile_urdf or emb.v1_mobile_yaml):
                continue
            res = do_build_mobile(emb)
            res["embodiment"] = emb.name
            print(f"[build_mobile] {emb.name:24s} {'OK' if res.get('ok') else 'FAIL'}"
                  f"  {res.get('error','')}")
            results.append(res)

    elapsed = time.time() - started
    summary = {"elapsed_s": elapsed, "results": results}
    (SCRATCH / "orchestrator_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nDone in {elapsed:.1f}s — summary at "
          f"{SCRATCH / 'orchestrator_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
