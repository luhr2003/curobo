# MagicSim fork of cuRobo v2

This is a MagicSim-internal fork of NVIDIA's cuRobo v2 (upstream:
[`4ea7736 cuRoboV2 research release`](https://github.com/NVlabs/curobo)).
See `README.md` for upstream docs. This file lists **everything added on
top of that baseline**.

All deltas are **additive** — no upstream kernel / solver / config is
modified in-place. A solver built with stock cfg (`multi_env=False`,
`paired=False`) takes the original v2 code paths byte-for-byte.

---

## 1. Per-env tool-pose disable

Added by: `a9b168d add dual arm motiongen` (2026-04-21)

**Problem it solves.** Upstream's `ToolPoseCost` has a single weight
buffer shared across all envs in a batched solve. You can disable a
tool frame globally or leave it tracked globally — but can't say "env 5
disables arm A, env 7 disables arm B" in one `solve_pose` call. Every
disable-pattern change had to rebuild the whole solver.

**What's new.**

| Layer | Addition |
|---|---|
| `curobo/_src/cost/tool_pose_criteria.py` | `ToolPoseCriteria.track_position_and_orientation(xyz, rpy)` + `ToolPoseCriteria.disabled()` factories. `StackedToolPoseCriteria` now supports `per_env=True` mode with per-env weight buffer shape `(num_envs, num_links, K)` instead of `(num_links, K)`. |
| `curobo/_src/cost/cost_tool_pose_cfg.py` | `ToolPoseCostCfg.per_env: bool`, `num_envs: int`. |
| `curobo/_src/cost/wp_tool_pose.py` | New `ToolPoseDistancePerEnv` autograd + `create_goalset_pose_distance_kernel_per_env_with_constants` warp kernel. Weight indexed by `goal_idx = idxs_goal[b_idx]` at runtime. |
| `curobo/_src/cost/cost_tool_pose.py` | `ToolPoseCost.update_tool_pose_criteria_per_env(env_idx, {frame: criteria})` runtime hook. |
| `curobo/_src/solver/*`, `curobo/_src/motion/*`, `curobo/_src/rollout/cost_manager/*` | `update_tool_pose_criteria_per_env` plumbed through `InverseKinematics`, `MotionPlanner`, `BatchMotionPlanner`, `MPC`, the cost-manager, and the seed-IK solver. |
| `curobo/_src/solver/solver_core_cfg.py` | `enable_per_env_tool_pose(core_cfg, num_envs)` helper — walks every rollout cfg and flips `tp_cfg.per_env = True, num_envs = N`. Auto-called by `InverseKinematicsCfg.create(multi_env=True, max_batch_size=N)`. |

**Call pattern (unchanged for stock usage):**

```python
# Stock upstream path:
cfg = InverseKinematicsCfg.create(robot=..., multi_env=False)     # per_env stays False
ik = InverseKinematics(cfg)
# ← ToolPoseDistance + create_goalset_pose_distance_kernel_with_constants

# MagicSim per-env path:
cfg = InverseKinematicsCfg.create(robot=..., multi_env=True, max_batch_size=N)
ik = InverseKinematics(cfg)
# ← ToolPoseDistancePerEnv + create_goalset_pose_distance_kernel_per_env_with_constants
for env_idx in range(N):
    ik.update_tool_pose_criteria_per_env(
        env_idx,
        {"arm_R": ToolPoseCriteria.track_position_and_orientation(xyz, rpy),
         "arm_L": ToolPoseCriteria.disabled()},
    )
ik.solve_pose(goal, current_state)    # env 0..N-1 each get their own tool-pose weight row
```

**Examples / tests:**
- `curobo/examples/isaacsim/per_env_disable_ik_dual_ur10e.py`
- `curobo/examples/isaacsim/per_env_disable_motiongen_dual_ur10e.py`
- `curobo/tests/_src/cost/test_tool_pose_per_env.py`

---

## 2. Paired goalset

Added by: `a24d4a6 paired goalset` (2026-04-21)

**Problem it solves.** Upstream's goalset argmin is **per-link**: each
tracked tool_frame independently picks
``g_l = argmin_g d(current_l, goal_l[g])``. For bimanual rigid grasps
(both arms grabbing one object) this can pick different `g`s for right
vs left — which is nonsense when the two poses must match a paired
candidate. Needed a kernel variant that minimizes the SUM and returns
one shared `g*`.

**What's new.**

| Layer | Addition |
|---|---|
| `curobo/_src/cost/cost_tool_pose_cfg.py` | `ToolPoseCostCfg.paired: bool = False`. |
| `curobo/_src/cost/wp_tool_pose.py` | New `ToolPoseDistancePerEnvPaired` autograd + `create_goalset_pose_distance_kernel_per_env_paired_with_constants` warp kernel. Thread layout `(B, H)` instead of `(B, H, L)`; two-pass body — outer `g` loop over summed per-link cost picks `best_g`, then per-link output write-out at `best_g`. |
| `curobo/_src/cost/cost_tool_pose.py` | New branch in `forward`: `if per_env and paired → ToolPoseDistancePerEnvPaired.apply`. `per_env=False, paired=True` is intentionally not wired (all production paths run `multi_env=True`). |
| `curobo/_src/solver/solver_core_cfg.py` | `enable_paired_tool_pose(core_cfg)` helper — flips `tp_cfg.paired = True` on every rollout. Must be called AFTER `enable_per_env_tool_pose`. |

**Call pattern:**

```python
cfg = InverseKinematicsCfg.create(robot=..., multi_env=True, max_batch_size=N, max_goalset=G)
enable_paired_tool_pose(cfg.core_cfg)           # BEFORE InverseKinematics(cfg)
ik = InverseKinematics(cfg)
# ← ToolPoseDistancePerEnvPaired + paired kernel
```

**Semantics (L=2, G=8 example).** Per (env, horizon-step):

- **Unpaired** (stock): `g_right = argmin_g d(cur_R, goal_R[g])`, independent of `g_left`. Two independent argmins.
- **Paired** (new): `g* = argmin_g [d(cur_R, goal_R[g]) + d(cur_L, goal_L[g])]`. One shared `g*`. The kernel writes `out_goalset_idx[..., li] = g*` for EVERY link in the group.

**Degeneracies (handled silently):**
- `L = 1` → paired sum over one link reduces to plain argmin. Kernel runs but produces the same result as unpaired. Safe to always enable paired.
- `G = 1` → `g* = 0` trivially. Paired ≡ unpaired.

**Per-env disable composes.** A link whose `terminal_pose_axes_weight_factor[env, link, :] == 0` contributes zero to the paired sum, so that link is excluded from the argmin for that env without affecting other envs. Paired + disable is how MagicSim handles "bimanual grasp, but env 3 only has right arm active this solve".

**Examples / tests:**
- `curobo/examples/isaacsim/goalset_groups_tri_ur10e.py` — dual grasp-ring sweep with `--mode paired` / `--mode padded`. Uses the new 3-arm `tri_ur10e.yml` robot.
- `curobo/tests/_src/cost/test_tool_pose_paired.py` — kernel-level argmin verification.
- `curobo/tests/_src/solver/test_solver_ik_paired.py` — end-to-end solver test with paired goalset.

---

## 3. MagicSim robot YAMLs

Added by: `6fa4294 cross embodiement` + `1652e52 fix bug` + subsequent.

Ten new dual-arm / mobile-manip robot configs under
`curobo/content/configs/robot/`:

```
magicsim_dual_arx_x5.yml         [+ _mobile.yml]    (dual ARX X5)
magicsim_dual_piper.yml          [+ _mobile.yml]    (dual Agilex Piper)
magicsim_dual_so101.yml          [+ _mobile.yml]    (dual SO101)
magicsim_xtrainer.yml            [+ _mobile.yml]    (Xtrainer)
magicsim_g1_simple.yml           [+ _mobile.yml]    (Unitree G1 simplified bimanual)
magicsim_genie1.yml              [+ _mobile.yml]    (Agibot Genie)
magicsim_lift2_mobile.yml                           (Lift2 mobile)
magicsim_mobile_x7s.yml          [+ _mobile.yml]    (X7s mobile)
magicsim_ridgebackfranka.yml     [+ _mobile.yml]    (Ridgeback + Franka)
magicsim_vega1p_sharpa.yml       [+ _mobile.yml]    (Vega 1P + Sharpa)
magicsim_split_aloha.yml                            (split ALOHA)
```

Pairs where `_mobile.yml` exists are the locked-base / free-base pair
consumed by MagicSim's `DualIKServer` / `DualMotionGenServer` — they
declare identical `tool_frames`, `extra_fk_link`, `info_links` but
different `base_link` (fixed arm root vs mobile platform root).

Each YAML adds the MagicSim contract fields on top of stock cuRobo:

- **`extra_fk_link`** (top-level, optional): FK-only frames. Merged
  into `kinematics.tool_frames` by PlannerManager, then marked with
  `ToolPoseCriteria.disabled()` so they contribute zero cost at IK seed,
  main IK, and TrajOpt stages but still get FK buffers allocated.
- **`info_links`** (top-level, optional): ordering for position-mode
  FK readout. Defaults to tracked `tool_frames` if omitted.
- **`ignore_joints`** (top-level, optional): dict of `joint_name →
  constant fill value` for joints that exist in the sim articulation
  but shouldn't influence IK.
- **`add_joints`** (top-level, optional): dict of `joint_name → sim
  state index` for virtual base joints (`base_x`, `base_y`, `base_h`,
  `base_z`) on the free-base YAML of a mobile pair.

Stock cuRobo YAMLs do not read these fields; they're consumed by
MagicSim's Service layer (see `src/magicsim/Env/Planner/Services/README.md`).

Also added under `curobo/content/configs/robot/spheres/`: sphere-set
definitions for the new robots.

---

## 4. `tri_ur10e` test robot

Added by: `a24d4a6 paired goalset`.

Three-arm UR10e fixture used exclusively by the paired-goalset test
suite and Isaac Sim example:

```
curobo/content/configs/robot/tri_ur10e.yml
curobo/content/assets/robot/ur_description/tri_ur10e.urdf
curobo/content/configs/robot/spheres/quad_ur10e.yml
```

18 DOF, `tool_frames = [tool0, tool1, tool2]`. Lets us exercise
`L >= 3` in the paired kernel (the real robot needs only L=2 for
bimanual grasp; the third frame covers "disabled" scenarios).

---

## 5. MPC fixes + humanoid retargeting docs

Added by: `3b80b95 fix mpc`.

- `curobo/examples/isaacsim/batch_mpc_example.py` — batched-MPC demo.
- `curobo/examples/isaacsim/mpc_example.py` — patched for MagicSim's
  multi_env world.
- `QUICKSTART.md` — supplemental quickstart doc ported from internal wiki.
  Humanoid retargeting guide is now part of `CUROBO_V2_GUIDE.md` §5.
- `curobo/examples/getting_started/viser_csv_viewer.py` — viewer
  utility for pre-recorded motion traces.

---

## 6. Compatibility guarantees

Every stock cuRobo call site keeps working unchanged:

- **Stock `ToolPoseCost` dispatch**: `per_env=False, paired=False`
  (both default) → same `ToolPoseDistance.apply` +
  `create_goalset_pose_distance_kernel_with_constants` as upstream.
  Verified: `cost_tool_pose.py` diff is purely additive branches; the
  final `else` fallthrough is byte-identical to upstream.
- **Stock cfg factories**: `InverseKinematicsCfg.create(multi_env=False)`
  does not auto-enable `per_env`. You have to pass `multi_env=True`
  (or call `enable_per_env_tool_pose` manually) to opt in.
- **Paired opt-in**: `ToolPoseCostCfg.paired` defaults to `False`.
  Even with `multi_env=True`, stock + per-env paths are unchanged —
  you must explicitly call `enable_paired_tool_pose(core_cfg)` to
  switch to the paired kernel.
- **New YAML fields**: `extra_fk_link`, `info_links`, `ignore_joints`,
  `add_joints` are top-level and ignored by stock `InverseKinematicsCfg.create`.
  They only take effect when consumed by MagicSim's PlannerManager.

If you build a solver from a stock cuRobo YAML + stock cfg options, you
get the upstream behavior with no MagicSim code on the hot path.

---

## 7. Where to look next

Usage-side docs live on the MagicSim side:

- `src/magicsim/Env/Planner/Services/README.md` — Service-layer
  architecture, request/result pipeline, slot mapping,
  NaN-as-disable, left/right flatten.
- `MERGE_LEFT_RIGHT.md §9` — NaN-disable + per-frame goalset wiring
  spec.
- `PER_ENV_TOOL_POSE_COST_PLAN.md` — per-env weight buffer design
  notes.
- `GOALSET_PER_FRAME_ANALYSIS.md` — `(B, L, G, 7)` per-frame goalset
  design notes.

Upstream / local docs:

- `README.md` — stock cuRobo README (upstream, untouched).
- `MIGRATION_V1_TO_V2.md` — cuRobo v1→v2 API migration notes.
- `QUICKSTART.md` — environment setup + one-command demos.
- `CUROBO_V2_GUIDE.md` — merged usage guide: robot-model builder,
  batched solver interfaces, multi-EEF targets, MotionRetargeter,
  humanoid retargeting end-to-end.
- `curobo/examples/isaacsim/README.md` — Isaac Sim example index.
