# cuRobo 2.0 — IK / MPC / Motion Planner Interface Guide

This document is organized into three chapters:

1. **[Batched Interfaces](#1-batched-interfaces)** — how to run
   `solve_single / solve_goalset / solve_batch / solve_batch_goalset /
   solve_batch_env / solve_batch_env_goalset` for each of the three solvers
   (IK, MPC, Motion Planning).
2. **[Multi-Tool-Frame (Multi-EEF) Targets](#2-multi-tool-frame-multi-eef-targets)**
   — how to drive a solver with more than one end-effector at the same time.
3. **[MotionRetargeter](#3-motionretargeter)** — the high-level wrapper that
   uses the multi-EEF IK / MPC stack to retarget human motion capture to a
   humanoid robot.

Symbols used below:

| symbol | meaning                                                          |
|--------|------------------------------------------------------------------|
| `B`    | batch size (independent problems solved together)                |
| `H`    | horizon on the goal side (`H=1` for static goals)                |
| `L`    | number of tool frames / end-effectors (`len(tool_frames)`)       |
| `G`    | `num_goalset` — candidate goals per problem (`G=1` = no goalset) |
| `E`    | `num_envs` — number of distinct collision worlds                 |

The canonical goal tensor is
`GoalToolPose.position : (B, H, L, G, 3)` and
`GoalToolPose.quaternion : (B, H, L, G, 4)` (wxyz).
See `curobo/_src/types/tool_pose.py`.

---

## 1. Batched Interfaces

### 1.1 Mode selection happens in the config, not in the call

cuRobo 2.0 does **not** expose separate Python methods called
`solve_batch_env` or `solve_goalset`. Every solver has a single entry point
(`solve_pose`, `plan_pose`, `update_goal_tool_poses`) that routes internally to
one of three solve modes defined in `curobo/_src/solver/solve_mode.py`:

```python
class SolveMode(Enum):
    SINGLE    = "single"     # batch_size == 1
    BATCH     = "batch"      # batch_size > 1, shared collision world
    MULTI_ENV = "multi_env"  # batch_size > 1, per-problem collision world
```

The dispatch (from `solver_ik.py:688-702`, mirrored in `solver_mpc.py:281-296`):

```python
if batch_size == 1:                solve_mode = SolveMode.SINGLE
elif self.config.multi_env:        solve_mode = SolveMode.MULTI_ENV
else:                              solve_mode = SolveMode.BATCH
```

So every `*Cfg` (`IKSolverCfg`, `MotionPlannerCfg`, `MPCSolverCfg`) carries the
same three axes:

| config flag        | what it controls                                                       |
|--------------------|------------------------------------------------------------------------|
| `max_batch_size=N` | allocate buffers / CUDA graphs for up to `N` problems (`B ≤ N`)        |
| `multi_env=False`  | one collision world shared by every problem (`E=1`)                    |
| `multi_env=True`   | per-problem collision world (`E=N=max_batch_size`), disables PRM graph |
| `max_goalset=G`    | allocate goalset dim of size `G` (`num_goalset ≤ G`)                   |

Mapping of the six variant names you asked about to the config triple:

| variant                   | `max_batch_size` | `multi_env` | `max_goalset` |
|---------------------------|:----------------:|:-----------:|:-------------:|
| `solve_single`            | `1`              | `False`     | `1`           |
| `solve_goalset`           | `1`              | `False`     | `G > 1`       |
| `solve_batch`             | `N > 1`          | `False`     | `1`           |
| `solve_batch_goalset`     | `N > 1`          | `False`     | `G > 1`       |
| `solve_batch_env`         | `N > 1`          | `True`      | `1`           |
| `solve_batch_env_goalset` | `N > 1`          | `True`      | `G > 1`       |

Padding: you can pass fewer than `max_batch_size` problems at runtime. The
solver pads up to `max_batch_size` internally and slices the result back
(`_pad_batch_inputs` at `solver_ik.py:43`). So one solver built with
`max_batch_size=4` handles `B ∈ {1,2,3,4}` without rebuilding.

### 1.2 Inverse kinematics (`InverseKinematics` / `IKSolver`)

Entry point (`curobo/_src/solver/solver_ik.py:634`):

```python
def solve_pose(
    self,
    goal_tool_poses: GoalToolPose,          # (B, 1, L, G, 3/4)
    current_state: JointState | None = None,
    seed_config: torch.Tensor | None = None,
    return_seeds: int = 1,
    run_optimizer: bool = True,
) -> IKSolverResult
```

`solve_single`:

```python
cfg = InverseKinematicsCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    num_seeds=32,                 # max_batch_size=1, max_goalset=1 (defaults)
)
ik = InverseKinematics(cfg)
goal = GoalToolPose.from_poses({ik.tool_frames[0]: pose},  # pose.position: (1,3)
                               num_goalset=1)
result = ik.solve_pose(goal)
```

`solve_goalset` (same env, `G` candidate targets per problem):

```python
cfg = InverseKinematicsCfg.create(robot="franka.yml",
                                  scene_model="collision_table.yml",
                                  max_goalset=G)
ik = InverseKinematics(cfg)
goal = GoalToolPose(                           # raw form
    tool_frames=ik.tool_frames,
    position=pos,                              # (1, 1, L, G, 3)
    quaternion=quat,                           # (1, 1, L, G, 4)
)
result = ik.solve_pose(goal)                   # result.goalset_index: which G was picked
```

`solve_batch` (shared env, `B` independent targets):

```python
cfg = InverseKinematicsCfg.create(robot="franka.yml",
                                  scene_model="collision_table.yml",
                                  max_batch_size=B)
ik = InverseKinematics(cfg)
goal = GoalToolPose.from_poses({ik.tool_frames[0]: pose_B},  # pose_B.position: (B,3)
                               num_goalset=1)
result = ik.solve_pose(goal)                    # result.success / .solution leading dim = B
```
This is the pattern used by the 100-pose reachability example in
`curobo/examples/getting_started/inverse_kinematics.py::batched_ik_example`.

`solve_batch_goalset`: same as above with `max_goalset=G` and `(B, 1, L, G, *)`.

`solve_batch_env` (per-problem collision world):

```python
cfg = InverseKinematicsCfg.create(robot="franka.yml",
                                  scene_model="collision_table.yml",
                                  max_batch_size=N,
                                  multi_env=True)
ik = InverseKinematics(cfg)
# Per-env obstacles (and link_spheres) are allocated with num_envs=N; see
# solver_ik_cfg.py:221 and test_motion_planner_num_envs.py for confirmation.
# Update obstacle poses per env via ik.scene_collision_checker.update_obstacle_pose.
goal = GoalToolPose.from_poses({ik.tool_frames[0]: pose_B}, num_goalset=1)
result = ik.solve_pose(goal)
```
Confirmed by `curobo/tests/_src/solver/test_solver_ik.py::TestIKSolverSolvePoseBatchEnv`.

`solve_batch_env_goalset`: `max_batch_size=N` + `multi_env=True` + `max_goalset=G`.

Runtime scene updates:

```python
ik.update_world(Scene(cuboid=[...]))                          # replace full scene
ik.scene_collision_checker.update_obstacle_pose(name, pose)   # per-obstacle
```

### 1.3 Motion planning (`MotionPlanner` vs `BatchMotionPlanner`)

Two distinct classes split by the solve shape, both taking the same
`MotionPlannerCfg` (which bundles an `IKSolverCfg` + `TrajOptSolverCfg`, so the
same `max_batch_size` / `multi_env` / `max_goalset` triple flows through):

| class                                                          | `batch_size` | retries                              | graph seeding                                       |
|----------------------------------------------------------------|:------------:|--------------------------------------|-----------------------------------------------------|
| `MotionPlanner`  (`_src/motion/motion_planner.py`)             | `1`          | yes (`max_attempts` IK+TrajOpt loop) | yes, PRM                                            |
| `BatchMotionPlanner` (`_src/motion/motion_planner_batch.py`)   | `≥ 1`        | optional, first-success-wins         | yes when `multi_env=False`; disabled when `True`    |

`MotionPlanner.plan_pose` — single or goalset:

```python
def plan_pose(
    goal_tool_poses: GoalToolPose,          # num_goalset auto-detected
    current_state: JointState,              # (1, dof)
    use_implicit_goal: bool = True,
    max_attempts: int = 5,
    enable_graph_attempt: int = 1,
) -> TrajOptSolverResult | None
```
When `goal_tool_poses.num_goalset > 1` the planner takes the goalset code
path (no graph seeding, `motion_planner.py:224`).

Grasp planning (goalset under the hood) — `motion_planner.py:412`:

```python
planner.plan_grasp(
    grasp_poses,                    # GoalToolPose (1, 1, L, G, *)
    current_state,                  # (1, dof)
    grasp_approach_axis="z", grasp_approach_offset=-0.15,
    grasp_lift_axis="z",     grasp_lift_offset=-0.15,
    plan_approach_to_grasp=True, plan_grasp_to_lift=True,
)
```
This is the flow used by `motion_planning.py::grasp_planning_example` with
`max_goalset=10`.

`BatchMotionPlanner.plan_pose` — covers batch / batch_env (± goalset):

```python
def plan_pose(
    goal_tool_poses: GoalToolPose,     # (B, 1, L, G, *)
    current_state: JointState,         # (B, dof)
    use_implicit_goal: bool = True,
    max_attempts: int = 1,
    success_ratio: float = 1.0,
    enable_graph_attempt: int = 0,
) -> TrajOptSolverResult | None
```

| variant                   | config                                                 |
|---------------------------|--------------------------------------------------------|
| `solve_batch`             | `max_batch_size=N, multi_env=False, max_goalset=1`     |
| `solve_batch_goalset`     | `max_batch_size=N, multi_env=False, max_goalset=G`     |
| `solve_batch_env`         | `max_batch_size=N, multi_env=True,  max_goalset=1`     |
| `solve_batch_env_goalset` | `max_batch_size=N, multi_env=True,  max_goalset=G`     |

`BatchMotionPlanner.plan_grasp` is the batched version and follows the same
per-problem `goalset_index` logic (`motion_planner_batch.py:475`).

Joint-space variants: `plan_cspace(goal_state, current_state, …)` on both
classes; same batch/env matrix.

### 1.4 MPC (`ModelPredictiveControl` / `MPCSolver`)

One class, same three config axes.

```python
cfg = ModelPredictiveControlCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    max_batch_size=B,        # number of robots / problems
    multi_env=<bool>,        # per-problem collision world
    max_goalset=G,
    optimization_dt=0.025,
    interpolation_steps=4,
    use_cuda_graph=True,
)
mpc = ModelPredictiveControl(cfg)

mpc.setup(current_state)                                        # current_state: (B, dof)
mpc.update_goal_tool_poses(
    GoalToolPose.from_poses(target_poses,
                            ordered_tool_frames=mpc.tool_frames,
                            num_goalset=G),                     # (B, 1, L, G, *)
    run_ik=True,
)

while True:
    result = mpc.optimize_action_sequence(current_state)        # or optimize_next_action
    current_state = step_robot(result.action_sequence)
```

Key signatures (`curobo/_src/solver/solver_mpc.py`):

```python
mpc.setup(current_state, tool_frames=None, dt=None)
mpc.update_goal_tool_poses(goal_tool_poses, robot_ids=None,
                           run_ik=True, use_ik_goal=True,
                           use_best_effort_ik=False)
mpc.update_goal_state(goal_state, robot_ids=None)
mpc.update_current_state(current_state)
mpc.optimize_next_action(current_state)       # single-step command
mpc.optimize_action_sequence(current_state)   # full horizon
```

Variant mapping is identical to IK:

| variant                 | `max_batch_size` | `multi_env` | `max_goalset` |
|-------------------------|:----------------:|:-----------:|:-------------:|
| single MPC              | `1`              | F           | `1`           |
| single MPC goalset      | `1`              | F           | `G`           |
| batch MPC (shared env)  | `B`              | F           | `1`           |
| batch MPC goalset       | `B`              | F           | `G`           |
| batch-env MPC           | `B`              | T           | `1`           |
| batch-env MPC goalset   | `B`              | T           | `G`           |

`update_goal_tool_poses` accepts `robot_ids` for updating a subset of the `B`
robots in place (`solver_mpc.py:395-407`).

---

## 2. Multi-Tool-Frame (Multi-EEF) Targets

Multi-EEF is a **robot-config** property, not a solver flag. It combines
freely with every batch/env/goalset variant from Chapter 1.

### 2.1 Declaring multiple tool frames

In the robot YAML:

```yaml
robot_cfg:
  kinematics:
    tool_frames: ["tool1", "tool0"]        # curobo/content/configs/robot/dual_ur10e.yml
    # or 14 links for humanoid:            # curobo/content/configs/robot/unitree_g1_29dof_retarget.yml
```

`Kinematics.tool_frames` becomes a list of length `L`; every `GoalToolPose`
the solver consumes has its `num_links` dim equal to `L`.

### 2.2 Automatic solver tuning when `L > 1`

From `solver_ik.py:132-138`, the IK solver widens its seed search when there
is more than one target link:

- `L > 1` → 128 seed-LM seeds, 20 max iterations.
- `L > 2` → 64 seeds, 30 iterations, 256 tile threads.
- `override_iters_for_multi_link_ik` raises the LBFGS iteration count.

This tuning is transparent — you only change `tool_frames` in the robot YAML.

### 2.3 Building multi-EEF goals

Dict form, one `Pose` per link:

```python
goal = GoalToolPose.from_poses(
    {"tool0": pose_tool0, "tool1": pose_tool1},     # each Pose: (B*G, 3)
    ordered_tool_frames=ik.tool_frames,
    num_goalset=G,
)                                                   # position: (B, 1, L, G, 3)
```

Raw form (used by the goalset IK test):

```python
goal = GoalToolPose(
    tool_frames=ik.tool_frames,
    position=pos_tensor,                            # (B, 1, L, G, 3)
    quaternion=quat_tensor,                         # (B, 1, L, G, 4)
)
```

Both forms work across IK, MotionPlanner / BatchMotionPlanner, and MPC, and
are independent of `multi_env` and `max_goalset`.

### 2.4 Per-link weighting

Each EEF can be weighted independently via
`curobo._src.cost.tool_pose_criteria.ToolPoseCriteria`. Pass a dict to
the solver:

```python
solver.update_tool_pose_criteria({
    "pelvis":              ToolPoseCriteria.track_position_and_orientation(
                               xyz=[1.0,1.0,1.0], rpy=[0.067,0.067,0.067]),
    "left_ankle_roll_link": ToolPoseCriteria.track_position_and_orientation(
                               xyz=[1.0,1.0,1.0], rpy=[0.067,0.067,0.067]),
    # ...
})
```

Typical humanoid pattern: feet and pelvis heavy (balance), mid-chain links
light (already constrained by wrist / elbow targets).

---

## 3. MotionRetargeter

`curobo.motion_retargeter.MotionRetargeter` is the high-level wrapper that
drives multi-EEF IK (or MPC) frame-by-frame with temporally-coherent
warm-starting. It is the canonical way to retarget motion-capture data to a
humanoid robot, but it is also useful any time you need to track a sequence
of multi-link pose targets.

Code path: `curobo/_src/motion/motion_retargeter.py`.
End-to-end example: `curobo/examples/getting_started/humanoid_retargeting.py`.

### 3.1 What it actually does

Internally the retargeter runs a two-phase loop:

1. **Frame 0 — global IK.** Broad search with many random seeds (default 64),
   no velocity limit. This finds a good initial pose with no prior knowledge.
2. **Frames 1..N — local solver** warm-started from frame `k-1`:
   - `use_mpc=False` → single-seed velocity-limited IK (fast).
   - `use_mpc=True`  → MPC over a short horizon (smoother, uses acceleration
     and jerk costs).

The velocity limit and warm-start together prevent the solver from jumping
between disconnected IK branches as the target moves.

### 3.2 Config

```python
from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg, ToolPoseCriteria

cfg = MotionRetargeterCfg.create(
    robot="unitree_g1_29dof_retarget.yml",       # robot YAML with tool_frames=[...]
    tool_pose_criteria={                         # one entry per tracked link
        link_name: ToolPoseCriteria.track_position_and_orientation(
            xyz=[pw, pw, pw], rpy=[rw, rw, rw]),
        ...
    },
    num_envs=N,                                  # N clips retargeted in parallel
    self_collision_check=True,                   # Level 2 (default); False = Level 1
    use_mpc=False,                               # True = Level 3 (MPC local solver)
    steps_per_target=4,                          # only used when use_mpc=True
    velocity_regularization_weight=None,
    acceleration_regularization_weight=None,
)
retargeter = MotionRetargeter(cfg)
```

`num_envs` maps directly onto the underlying IK/MPC solver's `max_batch_size`
with `multi_env=True`, so retargeting `N` clips in parallel is literally
"batch-env IK with `L > 1` tool frames" from Chapter 1.

`tool_pose_criteria` keys define the tool frames that will be tracked, so
they must be a subset of `tool_frames` in the robot YAML.

### 3.3 Input tensor layout — `SequenceGoalToolPose`

`curobo._src.types.sequence_tool_pose.SequenceGoalToolPose` is the
time-first counterpart of `GoalToolPose`:

```
tool_frames : List[str], length L
position    : (num_frames, num_envs, num_links, num_goalset, 3)
quaternion  : (num_frames, num_envs, num_links, num_goalset, 4)   # wxyz
```

Note the difference from `GoalToolPose`: here `num_envs` is dim 1 (not 0) and
`num_frames` is the leading axis. Per-frame extraction
`seq.position[t]` already has the right shape for `GoalToolPose`
(`seq.at(t)` is provided as a convenience).

### 3.4 Entry points

```python
# Offline, whole clip(s):
result = retargeter.solve_sequence(seq)
# result.joint_state.position : (num_envs, num_frames, num_dof)
# result.trajectory           : smoother intermediate states (only with use_mpc=True)

# Streaming, one frame at a time (online / teleoperation):
frame_goal = GoalToolPose(...)      # (num_envs, 1, num_links, num_goalset, 3/4)
result = retargeter.solve_frame(frame_goal)
```

Both methods validate `num_envs` against the config value
(`motion_retargeter.py:138-173`); passing a mismatched batch dim raises.

### 3.5 Three fidelity levels

These are the same levels documented in
`humanoid_retargeting.py`, restated for reference:

| level | config                                                 | trade-off                                                       |
|:-----:|--------------------------------------------------------|-----------------------------------------------------------------|
| 1     | `self_collision_check=False`                           | fastest; robot links may interpenetrate on constrained poses.   |
| 2     | `self_collision_check=True` (default)                  | ~10-20% slower; prevents self-collision.                        |
| 3     | `self_collision_check=True, use_mpc=True`              | 2-4× slower; smoother trajectories (accel / jerk costs).        |

All three use the **same** multi-EEF IK stack underneath (level 3 additionally
uses the MPC stack for frames `1..N`). Nothing in Chapters 1 and 2 needs to
change to enable retargeting — the retargeter just drives those solvers with
the right warm-start policy and the right per-link weights.

### 3.6 Outputs

`RetargetResult` (`curobo/_src/motion/motion_retargeter_result.py`) exposes:

- `joint_state.position` — `(num_envs, num_frames, num_dof)`, one joint vector
  per input target frame.
- `trajectory`           — populated only when `use_mpc=True`; contains the
  smooth intermediate frames produced by MPC (shape
  `(num_envs, num_frames * steps_per_target * interpolation_steps, num_dof)`
  in the retargeting pipeline).

For humanoid configs that use virtual base joints (e.g.
`unitree_g1_29dof_retarget.yml`), the first six DOFs are the floating-base
`[x, y, z, roll, pitch, yaw]`; see the docstring of
`humanoid_retargeting.py` for the full CSV conversion example.

---

## Appendix: where to look in code

| question                              | file                                                                     |
|---------------------------------------|--------------------------------------------------------------------------|
| IK API signature                      | `curobo/_src/solver/solver_ik.py:634`                                    |
| IK config (flags → batch/env modes)   | `curobo/_src/solver/solver_ik_cfg.py:30`                                 |
| IK batch padding + slicing            | `curobo/_src/solver/solver_ik.py:43,82`                                  |
| Solve-mode dispatch                   | `curobo/_src/solver/solve_mode.py`, `solver_ik.py:688-702`               |
| MotionPlanner (single)                | `curobo/_src/motion/motion_planner.py:43`                                |
| BatchMotionPlanner                    | `curobo/_src/motion/motion_planner_batch.py:38`                          |
| Grasp planning (goalset)              | `motion_planner.py:412`, `motion_planner_batch.py:291`                   |
| MPC API                               | `curobo/_src/solver/solver_mpc.py:255,359,575`                           |
| `GoalToolPose` / layout               | `curobo/_src/types/tool_pose.py:183-356`                                 |
| `SequenceGoalToolPose`                | `curobo/_src/types/sequence_tool_pose.py`                                |
| `MotionRetargeter`                    | `curobo/_src/motion/motion_retargeter.py`                                |
| Multi-env kinematics test             | `curobo/tests/_src/motion/test_motion_planner_num_envs.py`               |
| Multi-EEF example (dual arm)          | `curobo/content/configs/robot/dual_ur10e.yml`                            |
| Multi-EEF example (humanoid, 14 links)| `curobo/content/configs/robot/unitree_g1_29dof_retarget.yml`             |
