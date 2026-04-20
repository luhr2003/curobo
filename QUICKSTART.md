# Quickstart — local setup for cuRobo 2.0 + Isaac Sim

This file is the short-form onboarding for **this working copy** of cuRobo.
It assumes you just cloned and want a working end-to-end loop (environment
built, Isaac Sim pulled in, examples runnable). For the upstream project
overview see [`README.md`](./README.md); for the v2 API map and the Isaac
Sim example catalogue see the links at the bottom.

---

## 1. System requirements

- Ubuntu 22.04 (other modern Linuxes should work; Isaac Sim's only
  validated on Ubuntu 22.04 / 24.04).
- NVIDIA GPU, Turing or newer, ≥ 4 GB VRAM.
- NVIDIA driver ≥ 580.65.06 (`nvidia-smi | grep CUDA` should report CUDA 12
  or higher).
- **Python 3.11 exactly.** Isaac Sim 5.x wheels are only built for cp311;
  3.10 / 3.12 will fail to resolve.
- [`uv`](https://docs.astral.sh/uv/) ≥ 0.9.

---

## 2. Install (one-time)

```bash
git clone https://github.com/NVlabs/curobo
cd curobo

# Fresh venv pinned to Python 3.11 (uv fetches a managed interpreter).
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install curobo (editable) + Isaac Lab + Isaac Sim in one shot.
# pyproject.toml's `[isaaclab]` extra pins torch==2.7.0 (cu128), pulls
# isaacsim-5.1.0 and all its subpackages from https://pypi.nvidia.com,
# and installs Isaac Lab 2.3.2.post1.
uv pip install -e '.[isaaclab]'
```

First run of `isaacsim` will fetch Omniverse extensions (~10 min) and
prompt for the Nvidia EULA — reply `Yes`.

Smoke-test:

```bash
python -c "import curobo, isaacsim, isaaclab, torch; print(torch.__version__)"
# expect: 2.7.0+cu128
```

Alternative extras if you're **not** using Isaac Sim:

```bash
uv pip install -e '.[cu12-torch]'   # curobo + cuda 12 + torch >= 2.5
uv pip install -e '.[cu13-torch]'   # curobo + cuda 13 + torch >= 2.9
```

See [`pyproject.toml`](./pyproject.toml) `[project.optional-dependencies]`
for the full list. `.` alone installs curobo's runtime deps without torch
or a CUDA runtime.

---

## 3. Auto-activate the venv (recommended)

The repo ships a `.envrc` that wires up [`direnv`](https://direnv.net/) to
auto-export `VIRTUAL_ENV` and prepend `.venv/bin/` to `PATH` on every shell
entering the directory:

```bash
direnv allow           # one-time; direnv will auto-activate from now on
```

After that, `which isaacsim` and `which isaacsim-motion-gen-reacher` both
resolve to `.venv/bin/...` without manually sourcing activate scripts.
`.envrc.local` is gitignored — drop per-host overrides (proxies,
`CUDA_HOME`, …) in there.

---

## 4. Run an example

Every Isaac Sim example we've ported is registered as a short `uv run`
entrypoint in `pyproject.toml`:

```bash
# With the venv active (or via direnv):
isaacsim-motion-gen-reacher                            # single-arm reacher
isaacsim-batch-motion-gen-reacher                      # 2 envs, batched
isaacsim-batched-multi-arm-reacher                     # N envs × L arms
isaacsim-dynamic-batch                                 # variable-B batched
isaacsim-ik-reachability                               # IK grid sweep
isaacsim-ik-batched-env                                # batched IK
isaacsim-ik-humanoid-batched-env                       # batched multi-EEF IK
isaacsim-mpc                                           # reactive MPC
curobo-build-robot-model --help                        # URDF → cuRobo YAML
```

The equivalent explicit form (works in any venv that has curobo installed):

```bash
python -m curobo.examples.isaacsim.motion_gen_reacher
```

Headless smoke test that exits on its own (no GUI, no dragging):

```bash
isaacsim-motion-gen-reacher --headless_mode native --max_steps 150
```

See the [Isaac Sim examples README](./curobo/examples/isaacsim/README.md)
for the full catalogue, per-example flags, and known gotchas.

> **Tip.** Plain `uv run <script>` re-resolves dependencies as a pre-flight
> and may trip on a stale VCS dep. Pass `--no-sync` to skip that step:
> ```bash
> uv run --no-sync isaacsim-motion-gen-reacher
> ```
> Or activate the venv once (either via `direnv` or `source
> .venv/bin/activate`) and call the script by name directly.

---

## 5. What's where

| File | For |
|---|---|
| [`README.md`](./README.md) | Upstream NVIDIA overview + citation. |
| [`QUICKSTART.md`](./QUICKSTART.md) | (this file) local setup + run recipes. |
| [`BATCH_INTERFACES.md`](./BATCH_INTERFACES.md) | v2 API map for IK / MotionPlanner / MPC: `max_batch_size × multi_env × max_goalset` → `solve_single / solve_batch / solve_batch_env / +_goalset`. Also multi-tool-frame and retargeting. |
| [`MIGRATION_V1_TO_V2.md`](./MIGRATION_V1_TO_V2.md) | Call-by-call translation for porting v1 `MotionGen / IKSolver / MpcSolver` scripts onto the v2 API. |
| [`BUILD_ROBOT_MODEL.md`](./BUILD_ROBOT_MODEL.md) | Generating a cuRobo YAML/XRDF from a URDF (collision spheres, self-collision ignore matrix). |
| [`curobo/examples/isaacsim/README.md`](./curobo/examples/isaacsim/README.md) | Catalogue of the ported Isaac Sim examples with `uv run` entry points and shared gotchas. |
| [`curobo/examples/getting_started/`](./curobo/examples/getting_started/) | Upstream headless examples (forward_kinematics, inverse_kinematics, motion_planning, …). |
| [`pyproject.toml`](./pyproject.toml) | Install extras (`cu12-torch`, `cu13-torch`, `isaaclab`, …), uv indexes, and script entrypoints. |
| [`.envrc`](./.envrc) | direnv auto-activation. |

---

## 6. Common troubleshooting

| Symptom | Cause → fix |
|---|---|
| `uv pip install -e '.[isaaclab]'` fails with "no matching distribution" for isaacsim | Python isn't 3.11. Recreate venv: `rm -rf .venv && uv venv --python 3.11 .venv`. |
| `TypeError: func() got an unexpected keyword argument 'module'` at cuRobo import time | Isaac Sim's extscache Warp 1.8.2 shadowed pip Warp 1.12. Make sure `from curobo.examples.isaacsim import bootstrap` runs before `SimulationApp()`. |
| `AttributeError: 'NoneType' object has no attribute 'cuda_devices'` | Same root cause as above; bootstrap also pre-imports `warp.torch` / `warp.context` and calls `warp.init()` so this can't happen once it's in place. |
| `isaacsim` command not found | venv not active. `source .venv/bin/activate` or `direnv allow`. |
| `omni.kit.livestream.native` missing error | Harmless — Isaac Sim 5.1 dropped that extension. Our helper catches it and logs a warning. |
| Env 1's robot doesn't move in a multi-env demo | Per-env sub-root Xform wasn't placed correctly — `UsdWriter.add_subroot("/World", "/World/world_i", pose)` misplaces to `/World/World/world_i`. Use `UsdGeom.Xform.Define(stage, "/World/world_i")` + `AddTranslateOp()` instead. |
| Plain `uv run isaacsim-…` fails resolving `robometrics` (or other VCS dep) | `uv run` re-syncs before invoking; pass `--no-sync`, or just use the activated venv. |

---

## 7. Upgrade paths

- **New example port**: copy `curobo/examples/isaacsim/motion_gen_reacher.py`
  as a template. First line after `__future__` imports must be
  `from curobo.examples.isaacsim import bootstrap  # noqa: F401`.
  Add the `main()` entrypoint to `[project.scripts]` in pyproject.toml and
  rerun `uv pip install -e . --no-deps` to register the wrapper script.
- **Custom robot**: generate a YAML via
  `curobo-build-robot-model` ([`BUILD_ROBOT_MODEL.md`](./BUILD_ROBOT_MODEL.md))
  and reference it via `--robot my_robot.yml` in any reacher.
- **Different CUDA stack**: swap `[isaaclab]` for `[cu12-torch]` or
  `[cu13-torch]`. Conflicts between `isaaclab` and `cu13-*` extras are
  declared in `[tool.uv].conflicts` so uv will refuse a bad combination.
