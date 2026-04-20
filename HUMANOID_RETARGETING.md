# Humanoid Motion Retargeting (SOMA → Unitree G1)

GPU-accelerated retargeting of SOMA human motion capture (BVH) to the Unitree
G1 29-DOF humanoid using cuRobo IK / MPC. Outputs a SOMA `CSVAnimationBuffer`
CSV that can be played back in a Viser web viewer.

Source example: `curobo/examples/getting_started/humanoid_retargeting.py`
Standalone viewer: `curobo/examples/getting_started/viser_csv_viewer.py`
Demo input: `assets/motions/Neutral_walk_forward_002__A057.bvh`

## 1. Environment

Python 3.12, CUDA 12 (driver 545+), an NVIDIA GPU.

```bash
cd /path/to/magicsim/curobo
uv venv --python 3.12 .venv
uv pip install -e ".[cu12-torch,usd]"
uv pip install "soma-retargeter @ git+https://github.com/NVIDIA/soma-retargeter.git"
```

### Fix: replace Git-LFS pointer files in `soma-retargeter`

When `soma-retargeter` is installed via `pip install git+...`, two asset
files come through as 130-byte text **LFS pointers** instead of the real
binary files, and the BVH parser crashes with
`ValueError: could not convert string to float: 'size'`.

> **What is Git LFS?** "Large File Storage" — an extension that keeps big
> binaries (BVH, USD, models, images, …) on a separate server. The Git repo
> only tracks a tiny text pointer like:
> ```
> version https://git-lfs.github.com/spec/v1
> oid sha256:103b...
> size 29038
> ```
> `git clone` alone downloads only pointers. To pull the real files you
> need `git lfs install` + `git lfs pull`, *or* a clone done in an
> LFS-aware environment. `pip`/`uv install git+...` fetches only pointers.

Real copies of both files are bundled here at
`assets/soma_lfs/`. Drop them into the installed package:

```bash
SP=$(.venv/bin/python -c "import soma_retargeter, os; print(os.path.dirname(soma_retargeter.__file__))")
cp assets/soma_lfs/soma_zero_frame0.bvh       "$SP/configs/soma/"
cp assets/soma_lfs/soma_base_skel_minimal.usd "$SP/configs/soma/"
```

## 2. Robot config

A prebuilt config ships at
`curobo/content/configs/robot/unitree_g1_29dof_retarget.yml` — 35 DOFs total
(6 virtual base joints + 29 body joints).

To rebuild it (e.g. after URDF changes), see `BUILD_ROBOT_MODEL.md` or run
`python -m curobo.examples.getting_started.build_robot_model ...`.

## 3. Retarget a BVH clip

Three fidelity levels, same script:

| Level | Mode                  | Flag                   | Notes                              |
|------:|-----------------------|------------------------|------------------------------------|
| 1     | IK, no self-collision | `--no-self-collision`  | Fastest; may self-interpenetrate.  |
| 2     | IK + self-collision   | *(default)*            | Recommended default.               |
| 3     | MPC                   | `--mpc`                | 2–4× slower; smoothest trajectory. |

Demo (MPC + live viewer):

```bash
.venv/bin/python -m curobo.examples.getting_started.humanoid_retargeting \
    --input assets/motions/Neutral_walk_forward_002__A057.bvh \
    --output /tmp/walk_forward.csv \
    --mpc --steps-per-target 4 \
    --visualize
```

Batch (directory of BVH files):

```bash
.venv/bin/python -m curobo.examples.getting_started.humanoid_retargeting \
    --input assets/motions/ --output /tmp/g1_csv/ --mpc --steps-per-target 4
```

Useful flags: `--max-frames N`, `--max-batch N`, `--robot-config <path>`,
`--viz-port 8080`.

## 4. Play back a saved CSV (standalone viewer)

The retargeting step already opens a viewer when `--visualize` is set, but the
solve is expensive. To replay an existing CSV without re-solving:

```bash
.venv/bin/python -m curobo.examples.getting_started.viser_csv_viewer \
    --input /tmp/walk_forward.csv
```

Folder of CSVs (dropdown lets you switch clips):

```bash
.venv/bin/python -m curobo.examples.getting_started.viser_csv_viewer \
    --input /tmp/g1_csv/ --fps 120
```

Viewer opens at `http://localhost:8080`. Controls: play/pause, loop, speed,
frame slider, frame-skip (raise for MPC high-FPS output), motion dropdown.

Flags: `--robot-config` (default `unitree_g1_29dof_retarget.yml`),
`--fps` (default 120), `--port` (default 8080), `--frame-skip`.

## 5. Output format

Each row of the CSV:
`Frame, root_translateX/Y/Z (cm), root_rotateX/Y/Z (deg), 29 × *_joint_dof (deg)`

Body joint order follows SOMA's
`UnitreeG129DOF_CSVConfig.csv_header` (legs → waist → arms). The cuRobo DOF
order differs (waist → arms → legs); the viewer reindexes by joint name, so
values can be fed back in either order.

## 6. Troubleshooting

- **`could not convert string to float: 'size'`** — `soma_retargeter`
  shipped LFS pointer files. See §1 "Fix".
- **`Expected vector of shape (3,), but got (1, 3)`** in `viser`'s
  `add_icosphere` — patched in `humanoid_retargeting.py` (the
  `target_positions` tensor has a trailing goalset dimension that needs to be
  flattened before passing to Viser).
- **CUDA / driver mismatch** — driver must be 545+ for CUDA 12. Check with
  `nvidia-smi`.
