# Cross-Embodiment Motion-Planning Smoke Test

Validates every migrated `magicsim_*.yml` under
`curobo/curobo/content/configs/robot/` by actually constructing a
`MotionPlanner`, calling `warmup`, and running `plan_pose` against a
**guaranteed-reachable** goal. A failure means the yaml is broken
(bad spheres, wrong base_link, self-collision at rest, missing joint
limits) — not a hard planning problem.

## Goal construction

For each robot the script:

1. Reads `planner.default_joint_state` as the start.
2. Perturbs the first joint by `+joint_delta = 0.1 rad` → forms `q_goal`.
3. Runs forward kinematics on `q_goal` → the EE pose at `q_goal`.
4. Hands that pose to `plan_pose`. By construction there is at least one
   valid IK solution (`q_goal` itself), so `success.any() == False`
   implies a configuration issue rather than an IK/planning difficulty.

For dual-arm robots all `tool_frames` are used, so the planner must solve
for every EE simultaneously.

## Usage

Run every robot in `ROBOTS`:

```bash
python -m curobo.examples.getting_started.motion_planning_crossembody
```

Run a single robot (iteration / debugging):

```bash
python -m curobo.examples.getting_started.motion_planning_crossembody \
    --robot magicsim_arx_x5.yml
```

Machine-readable output (one JSON line per robot):

```bash
python -m curobo.examples.getting_started.motion_planning_crossembody --json
```

Per-robot logs land in `--log-dir` (default `/tmp/curobo_migrate/`).
Each log contains the raw `{"robot": …, "ok": …, "error": …,
"traceback": …}` so failures are easy to triage.

## Coverage (last verified run)

22 embodiments tested. `OK` = `plan_pose` returned with
`result.success.any() == True`. `XFAIL` = robot config known to fail the
smoke test; kept in the roster so regressions can be spotted but
excluded from the pass/fail tally.

### Single-arm fixed-base

| Yaml | Status | Source URDF |
|---|---|---|
| `magicsim_arx_x5.yml` | OK | `arx_x5_description/urdf/X5A.urdf` |
| `magicsim_franka_umi.yml` | OK | `franka_description/franka_umi.urdf` |
| `magicsim_frankarobotiq.yml` | OK | `frankarobotiq.urdf` (franka + Robotiq 2F-85) |
| `magicsim_piper_x.yml` | OK | `piper_x_description/urdf/piper_x_description.urdf` |
| `magicsim_so101.yml` | OK | `so101/SO101.urdf` |

### Dual-arm fixed-base

| Yaml | Status | Source URDF | Notes |
|---|---|---|---|
| `magicsim_dual_arx_x5.yml` | OK | `dual_arx_x5_description/urdf/dual_arx_x5.urdf` | L/R collapsed |
| `magicsim_dual_piper.yml` | OK | `dual_piper_description/urdf/dual_piper.urdf` | L/R collapsed |
| `magicsim_dual_so101.yml` | OK | `dual_so101/dual_so101.urdf` | L/R collapsed |
| `magicsim_xtrainer.yml` | OK | `xtrainer/urdf/xtrainer.urdf` | L/R collapsed; J3/J4 auxiliary arms locked |

### Mobile / humanoid (fixed + mobile variants)

| Fixed yaml | Mobile yaml | Fixed status | Mobile status |
|---|---|---|---|
| `magicsim_mobile_x7s.yml` | `magicsim_mobile_x7s_mobile.yml` | OK | OK |
| `magicsim_ridgebacksawyer.yml` | `magicsim_ridgebacksawyer_mobile.yml` | OK | OK |
| `magicsim_split_aloha.yml` | `magicsim_split_aloha_mobile.yml` | OK | OK |
| `magicsim_vega1p.yml` | `magicsim_vega1p_mobile.yml` | OK | OK |
| `magicsim_vega1p_sharpa.yml` | `magicsim_vega1p_sharpa_mobile.yml` | OK | OK |
| `magicsim_genie1.yml` | `magicsim_genie1_mobile.yml` | OK | OK |

### Mobile-only

| Yaml | Status | Source URDF |
|---|---|---|
| `magicsim_lift2_mobile.yml` | OK | `lift2/lift2_mobile.urdf` |

## Notes on tricky migrations

- **`magicsim_frankarobotiq.yml`**: ported from the reference
  `interndataA1/curobo/.../frankarobotiq_left_arm.yml` verbatim. The
  reference's hand-tuned sphere set (small radii on fingers + `mount_4`
  + `robotiq_85_base_link`) is compact enough not to self-collide at
  rest; `build_robot_model`'s MorphIt fit was too conservative.
- **`magicsim_genie1.yml` / `magicsim_genie1_mobile.yml`**: arm spheres
  are imported from the reference
  `G1_120s_{left,right}_arm_parallel_gripper.yml`; body / head / torso
  spheres are dropped (same pattern as the reference) so the
  self-collision matrix only tracks the arm chain. Default joint
  positions use the reference retract config
  (`[2.07, -0.61, -1.57, 1, -1.57, -1.57, 1.57]` left arm, negated
  right arm) so both arms start folded away from the body.
- **`magicsim_split_aloha.yml` / `_mobile.yml`**: URDF has no
  `<collision>` meshes. Spheres are inlined from v1
  `spheres/split_aloha_collision_mesh.yml` directly (fallback path).
- **`magicsim_vega1p_sharpa.yml`**: same pattern — v1 sharpa yaml uses
  the vega_1p sphere file; inlined verbatim.
- **URDF patches** (under `content/assets/robot/`, not in
  MagicCurobo): continuous-joint `<limit>` adds for wheel joints on
  `vega_1p`, `vega_1p_sharpa`, `split_aloha`; `lower="0" upper="0"`
  expanded to `[-0.001, 0.001]` on `lift2` wheels and `xtrainer` J3/J4
  auxiliary arms; `velocity="0"` → `velocity="1.0"` on xtrainer J3/J4.

## Final tally (last run)

- **OK: 22/22** (100%)
- **XFAIL: 0/22**
- **FAIL: 0/22**
