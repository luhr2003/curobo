# Building a cuRobo Robot Config from a URDF

A URDF doesn't carry the two things cuRobo needs at runtime:

1. **Collision spheres** — one sphere cloud per link, fitted to the mesh.
   Used by every cuRobo cost/collision pipeline (IK, TrajOpt, MPC).
2. **Self-collision ignore matrix** — link pairs the planner should skip
   (adjacent joints that always touch; pairs that can never reach each
   other in any configuration).

`curobo.examples.getting_started.build_robot_model` generates both and
serializes them (plus the kinematic tree) to a `.yml` (cuRobo native) or
`.xrdf` (Isaac Sim / Isaac Lab) file that the v2 solvers consume directly.

Script: `curobo/examples/getting_started/build_robot_model.py`
Full doc: `docs/getting-started/build_robot_model.rst`

---

## Quickstart — standard URDF → cuRobo YAML

```bash
uv run python -m curobo.examples.getting_started.build_robot_model \
    --urdf /path/to/robot.urdf \
    --asset-path /path/to/meshes_root \
    --output my_robot.yml
```

- `--urdf` — path to the URDF.
- `--asset-path` — directory the URDF's relative mesh paths resolve against.
  `package://my_robot/meshes/link.stl` has its `package://` prefix stripped
  automatically, so `--asset-path` should point at the **parent of**
  `my_robot/`.
- `--output` — output file. Extension decides format: `.yml` cuRobo native,
  `.xrdf` Isaac Sim/Lab. `--export-xrdf` emits both.

The generated YAML plugs straight into the v2 entry points:

```python
MotionPlannerCfg.create(robot="my_robot.yml", ...)
InverseKinematicsCfg.create(robot="my_robot.yml", ...)
ModelPredictiveControlCfg.create(robot="my_robot.yml", ...)
```

---

## Useful flags

| Flag | What it does |
|---|---|
| `--tool-frames link_a link_b ...` | Declare the end-effector links. Multi-EEF robots pass multiple names; this becomes the `tool_frames` list in the YAML. |
| `--export-xrdf` | Also emit an XRDF alongside the YAML. |
| `--compute-metrics` | Print per-link sphere-fit quality (cover%, protrusion%, surface gap, volume ratio). Useful sanity check when tuning sphere parameters. |
| `--sphere-density 2.0` | Multiply the default sphere budget per link (default 1.0). Higher = tighter fit, more runtime cost. |
| `--coverage-weight 1000` | MorphIt interior-coverage weight (default 1000). Higher = spheres fill the mesh volume more aggressively. |
| `--protrusion-weight 10` | MorphIt surface-overshoot weight (default 10). Higher = spheres stay inside the mesh more strictly. |
| `--clip-link LINK AXIS OFFSET` | Stop a given link's spheres from crossing a plane. Example: `--clip-link panda_link0 z 0.0` prevents the base's spheres from overlapping the mounting surface. Repeatable. |
| `--num-collision-samples 100` | Random joint samples used to decide which link pairs are always/never in collision. |
| `--no-prune` | Skip the self-collision pruning stage (faster but bigger ignore matrix). |
| `--seed 42` | Pin NumPy + torch RNGs for reproducible sphere fits. |
| `--visualize` / `--viz-port 8080` | Open a Viser viewer at `http://localhost:8080` showing mesh vs. fitted spheres. |

---

## Editing an existing config (no full rebuild)

```bash
# Re-fit spheres on one link only
uv run python -m curobo.examples.getting_started.build_robot_model \
    --edit-config my_robot.yml \
    --refit-link panda_hand \
    --sphere-density 1.5 \
    --output my_robot_refit.yml

# Add self-collision ignores (link_a will not be collision-checked against
# link_b or link_c), then recompute the collision matrix.
uv run python -m curobo.examples.getting_started.build_robot_model \
    --edit-config my_robot.yml \
    --add-collision-ignore link_a link_b,link_c \
    --recompute-collisions \
    --output my_robot_ignored.yml
```

---

## Humanoid / floating base

cuRobo supports floating bases without modifying the URDF: declare
`extra_links` + `child_link_name` in the YAML to insert six virtual joints
(three prismatic + three revolute) between the `base_link` and the robot's
root body.

See `curobo/content/configs/robot/unitree_g1_29dof_retarget.yml` for a
complete example, and the humanoid retargeting guide for the background.

---

## Smoke test

Verify the builder pipeline end-to-end with the bundled Franka URDF:

```bash
uv run python -m curobo.examples.getting_started.build_robot_model --test
```

Runs three stages inside a tempdir: new build from URDF → single-link refit
→ collision-ignore + matrix recompute. Passing all three confirms the
builder toolchain is healthy.

---

## Inside the pipeline

The builder is three stages:

1. **Sphere fitting** — MorphIt (Adam-based iterative optimizer) fits
   spheres per link. Two objectives traded off by `coverage_weight` and
   `protrusion_weight`: spheres should fill the mesh interior (coverage)
   without poking through the surface (protrusion). Per-link sphere budget
   is `sphere_density × (link-specific default)`.

2. **Self-collision matrix** — random joint-space samples drive forward
   kinematics; pairs that always collide (adjacent joints) or never can
   (out-of-reach pairs) are added to the ignore set.

3. **Export** — spheres + ignore matrix + kinematic tree + default cspace
   parameters go into the YAML / XRDF.

---

## Tips

- For custom robots, pass `--tool-frames` explicitly — otherwise the builder
  infers an empty list and downstream IK/motion-gen has nothing to track.
- When swapping meshes, prefer `--refit-link` over a full rebuild; it's an
  order of magnitude faster.
- If `--visualize` shows spheres protruding, bump `--protrusion-weight`
  (from 10 to 30–50) or clamp the link with `--clip-link`. If spheres leave
  hollow pockets, bump `--coverage-weight` (from 1000 to 2000–3000).
- `--seed` is worth setting for anything you'll commit — without it every
  build gives slightly different sphere positions.
