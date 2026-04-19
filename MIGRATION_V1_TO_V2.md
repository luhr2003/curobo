# v1 → v2 Migration Guide (MagicCurobo → cuRobo 2.0)

Companion document to [`BATCH_INTERFACES.md`](./BATCH_INTERFACES.md). That file
describes what cuRobo 2.0 exposes today; this file maps every solver entry
point used in `~/magicsim/MagicCurobo/examples/isaac_sim/**` onto the
equivalent v2 call so we can port the Isaac Sim examples one by one without
changing anything under `MagicCurobo/`.

Scope of the migration:

- `curobo.wrap.reacher.ik_solver.IKSolver`      → `curobo.inverse_kinematics.InverseKinematics`
- `curobo.wrap.reacher.motion_gen.MotionGen`    → `curobo.motion_planner.MotionPlanner` / `curobo.batch_motion_planner.BatchMotionPlanner`
- `curobo.wrap.reacher.mpc.MpcSolver`           → `curobo.model_predictive_control.ModelPredictiveControl`

Out of scope (for now): the Isaac Sim glue itself (`helper.add_robot_to_scene`,
`UsdHelper`, debug-draw, nvblox, realsense). Those are untouched by the
solver API change; they keep working as-is.

Symbol convention is the same as `BATCH_INTERFACES.md`:
`B` = batch size, `L` = number of tool frames, `G` = `num_goalset`,
`E` = `num_envs`, `H = 1` on the goal side.

---

## Chapter 1 — Four structural changes that affect every migration

Before the per-method table, four shape-level changes must be understood.
They show up in every single port.

### 1.1 Named methods → one entry + config flags

v1 exposes six named methods per solver (e.g. `solve_single`, `solve_batch`,
`solve_batch_env`, `solve_goalset`, `solve_batch_goalset`,
`solve_batch_env_goalset`). v2 exposes **one** entry point
(`solve_pose` / `plan_pose` / `update_goal_tool_poses`) and picks the mode
from `(max_batch_size, multi_env, max_goalset)` at construction time. The
mapping is:

| v1 method name suffix    | `max_batch_size` | `multi_env` | `max_goalset` |
|--------------------------|:----------------:|:-----------:|:-------------:|
| `_single`                | `1`              | `False`     | `1`           |
| `_goalset`               | `1`              | `False`     | `G`           |
| `_batch`                 | `N`              | `False`     | `1`           |
| `_batch_goalset`         | `N`              | `False`     | `G`           |
| `_batch_env`             | `N`              | `True`      | `1`           |
| `_batch_env_goalset`     | `N`              | `True`      | `G`           |

Practical consequence: **one v2 solver instance covers exactly one mode**.
Migrating a v1 file that used two named methods on the same solver object
either requires (a) two v2 solver instances, or (b) the larger mode that
supersedes both.

### 1.2 `goal_pose + link_poses` → `GoalToolPose`

v1 splits the goal across two arguments:

```python
# v1
ik_solver.solve_single(
    goal_pose=Pose(position=(1,3), quaternion=(1,4)),   # primary EE only
    link_poses={link: Pose(...) for link in other_links},
    retract_config=..., seed_config=...,
)
```

v2 packs all tracked frames (primary EE + extra links) into a single
5-D tensor:

```python
# v2 — GoalToolPose.position : (B, 1, L, G, 3), quaternion : (B, 1, L, G, 4)
goal = GoalToolPose.from_poses(
    {link_name: Pose(...) for link_name in ik.tool_frames},   # ALL links
    ordered_tool_frames=ik.tool_frames,
    num_goalset=G,
)
ik.solve_pose(goal, current_state=..., seed_config=...)
```

The v2 robot YAML declares `tool_frames: [...]`, so the set of tracked links
is fixed at config time, not per call. v2 auto-boosts seed count / LBFGS
iters for `L>1`, `L>2` (`solver_ik.py:132-138`).

### 1.3 `retract_config` + `seed_config` → `current_state` + `seed_config`

v1 takes two separate tensor arguments:

- `retract_config: (B, dof)` — regularization target.
- `seed_config:   (B, 1, dof)` — initial seed for optimization.

v2 takes a single `JointState` (`current_state`) of shape `(B, dof)`; the
internal LM seed solver uses its position as the seed. You can still pass an
explicit `seed_config` tensor, but the default is to use `current_state`:

```python
# v2
ik.solve_pose(goal_tool_poses, current_state=current_state)
# or, with an explicit seed:
ik.solve_pose(goal_tool_poses, current_state=current_state,
              seed_config=seed_tensor)       # (B, num_seeds, dof)
```

`current_state` also carries `dt`, which enables velocity-limited (servoing)
IK when `optimization_dt` is set in the config.

### 1.4 `WorldConfig` (or `List[WorldConfig]`) → `SceneCfg` + `multi_env`

v1:

```python
# Single env
ik_config = IKSolverConfig.load_from_robot_config(robot_cfg, world_cfg, ...)
# Multi env — pass a LIST of WorldConfig
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg, [world_cfg0, world_cfg1], ...)
```

v2:

```python
# Single env
cfg = InverseKinematicsCfg.create(robot="franka.yml", scene_model="collision_table.yml", ...)
# Multi env
cfg = InverseKinematicsCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",    # template; allocated with num_envs=N
    max_batch_size=N, multi_env=True,
)
# Then update per-env obstacles at runtime:
ik.scene_collision_checker.update_obstacle_pose(name, pose_env_i)
```

The v2 `create(..., scene_model=…)` call ultimately builds a
`SceneCfg`. To replace it wholesale at runtime, use `ik.update_world(Scene(...))`
(see `solver_ik.py:759`). For Isaac Sim stage sync, the v1 idiom
`ik_solver.update_world(obstacles)` maps to the same `update_world` call, but
the argument type changes from `WorldConfig` to `Scene`.

### 1.5 Result object renames

| v1 type                | v2 type                                                  |
|------------------------|----------------------------------------------------------|
| `IKResult`             | `IKSolverResult`                                         |
| `MotionGenResult`      | `TrajOptSolverResult`                                    |
| `MotionGenResult.get_interpolated_plan()` | `TrajOptSolverResult.get_interpolated_plan()` (same name, same idea) |
| `MotionGenResult.get_paths()` (batch-env) | iterate `trajopt_result.success[i]` + `js_solution[i]` yourself; there is no list-of-paths accessor today |
| `MpcSolverResult.js_action` | `MPCSolverResult.next_action` / `.action_sequence`   |

`success`, `solution`, `js_solution`, `position_error`, `rotation_error`,
`goalset_index` keep their names.

---

## Chapter 2 — IK mapping

v1 class: `curobo.wrap.reacher.ik_solver.IKSolver` (6 named methods).
v2 class: `curobo.inverse_kinematics.InverseKinematics` (one method).

### 2.1 Config

```python
# v1
ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg, world_cfg,
    rotation_threshold=0.05, position_threshold=0.005,
    num_seeds=20, self_collision_check=True, self_collision_opt=True,
    tensor_args=tensor_args,
    use_cuda_graph=True,
    collision_checker_type=CollisionCheckerType.MESH,
    collision_cache={"obb": 30, "mesh": 100},
    grad_iters=500,                     # override LBFGS iters
)
ik_solver = IKSolver(ik_config)

# v2
cfg = InverseKinematicsCfg.create(
    robot="franka.yml",                 # or pass a robot_cfg dict / RobotCfg
    scene_model="collision_table.yml",  # replaces world_cfg
    num_seeds=20,
    position_tolerance=0.005,           # renamed from position_threshold
    orientation_tolerance=0.05,         # renamed from rotation_threshold
    self_collision_check=True,
    use_cuda_graph=True,
    collision_cache={"cuboid": 30, "mesh": 100},   # key name: "obb" → "cuboid"
    override_iters_for_multi_link_ik=500,          # replaces grad_iters for multi-EE
    max_batch_size=1, multi_env=False, max_goalset=1,   # pick the mode here
)
ik = InverseKinematics(cfg)
```

### 2.2 Method-by-method signature map

All v1 methods share the same keyword set:
`(goal_pose, retract_config, seed_config, return_seeds, num_seeds, use_nn_seed, newton_iters, link_poses)`.

Every one maps to the same v2 call
`ik.solve_pose(goal_tool_poses, current_state, seed_config, return_seeds, run_optimizer)`
with a different `GoalToolPose` shape. The six rows below are the only
differences:

| v1 method                       | v1 `goal_pose` shape                    | v1 `link_poses`                    | v2 config                                               | v2 `GoalToolPose.position` shape |
|---------------------------------|------------------------------------------|------------------------------------|---------------------------------------------------------|-----------------------------------|
| `solve_single`                  | `Pose(pos=(1,3), quat=(1,4))`            | `Dict[str, Pose]` (each `(1,3/4)`) | `max_batch_size=1, multi_env=F, max_goalset=1`          | `(1, 1, L, 1, 3)`                 |
| `solve_goalset`                 | `Pose(pos=(1,G,3), quat=(1,G,4))`        | `Dict[str, Pose]` (each `(1,3/4)`) | `max_batch_size=1, multi_env=F, max_goalset=G`          | `(1, 1, L, G, 3)`                 |
| `solve_batch`                   | `Pose(pos=(B,3), quat=(B,4))`            | `Dict[str, List[Pose]]` len `B`    | `max_batch_size=B, multi_env=F, max_goalset=1`          | `(B, 1, L, 1, 3)`                 |
| `solve_batch_goalset`           | `Pose(pos=(B,G,3), quat=(B,G,4))`        | `Dict[str, List[Pose]]`            | `max_batch_size=B, multi_env=F, max_goalset=G`          | `(B, 1, L, G, 3)`                 |
| `solve_batch_env`               | `Pose(pos=(B,3), quat=(B,4))`            | `Dict[str, List[Pose]]`            | `max_batch_size=B, multi_env=T, max_goalset=1`          | `(B, 1, L, 1, 3)`                 |
| `solve_batch_env_goalset`       | `Pose(pos=(B,G,3), quat=(B,G,4))`        | `Dict[str, List[Pose]]`            | `max_batch_size=B, multi_env=T, max_goalset=G`          | `(B, 1, L, G, 3)`                 |

Helper for the goal tensor (it is the same code path for all six):

```python
def v1_to_v2_goal(primary_ee_name: str,
                  goal_pose,                       # v1 Pose
                  link_poses=None,                 # v1 Dict[str, Pose] or Dict[str, List[Pose]]
                  tool_frames=None,                # v2 ik.tool_frames
                  num_goalset=1) -> GoalToolPose:
    pose_dict = {primary_ee_name: goal_pose}
    if link_poses:
        pose_dict.update(link_poses)
    return GoalToolPose.from_poses(
        pose_dict, ordered_tool_frames=tool_frames, num_goalset=num_goalset,
    )
```

### 2.3 Worked example — `multi_arm_humanoid_parallel_ik.py` (solve_batch_env)

```python
# v1 (MagicCurobo/examples/isaac_sim/multi_arm_humanoid_parallel_ik.py:303)
result = ik_solver.solve_batch_env(
    goal_pose=ik_goal,                      # Pose (B=num_envs, 3/4)
    retract_config=full_js.position,        # (num_envs, dof)
    seed_config=full_js.position.unsqueeze(1),
    link_poses=link_poses_dict,             # {link: Pose(B, 3/4)}
    num_seeds=64,
)

# v2 equivalent
cfg = InverseKinematicsCfg.create(
    robot="magicsim_g1.yml",
    max_batch_size=num_envs, multi_env=True,        # batch_env
    num_seeds=64, self_collision_check=True,
)
ik = InverseKinematics(cfg)

goal = GoalToolPose.from_poses(
    {ik.tool_frames[0]: ik_goal, **link_poses_dict},
    ordered_tool_frames=ik.tool_frames,
    num_goalset=1,
)                                            # (num_envs, 1, L, 1, 3/4)

result = ik.solve_pose(goal, current_state=full_js)    # full_js: (num_envs, dof)
```

### 2.4 Scene updates

| v1                                    | v2                                                    |
|---------------------------------------|-------------------------------------------------------|
| `ik_solver.update_world(obstacles)`   | `ik.update_world(Scene(cuboid=[...], mesh=[...]))`    |
| (update one obstacle pose)            | `ik.scene_collision_checker.update_obstacle_pose(name, pose)` |
| `ik_solver.fk(q)` → `CudaRobotModelState` | `ik.compute_kinematics(js)` → `KinematicsState`   |
| `ik_solver.get_retract_config()`      | `ik.default_joint_state.position`                     |
| `ik_solver.kinematics.joint_names`    | `ik.joint_names`                                      |

---

## Chapter 3 — Motion planning mapping

v1 class: `curobo.wrap.reacher.motion_gen.MotionGen` (6 named methods +
`plan_grasp` + `plan_single_js`).
v2 classes: `MotionPlanner` (single) **and** `BatchMotionPlanner` (batch).

### 3.1 Config

```python
# v1
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg, tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=12, num_graph_seeds=12,
    interpolation_dt=0.05, collision_cache={"obb": 30, "mesh": 100},
    optimize_dt=True, trajopt_dt=None, trajopt_tsteps=32,
    trim_steps=None,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

# v2 — single-problem
cfg = MotionPlannerCfg.create(
    robot="franka.yml", scene_model="collision_table.yml",
    # max_batch_size defaults to 1, multi_env=False
)
planner = MotionPlanner(cfg)
planner.warmup(enable_graph=True, num_warmup_iterations=5)

# v2 — batch
cfg_batch = MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model=["collision_test.yml", "collision_thin_walls.yml"],
    max_batch_size=N, multi_env=True,            # or multi_env=False for shared world
)
batch_planner = BatchMotionPlanner(cfg_batch)
batch_planner.warmup(enable_graph=True, num_warmup_iterations=5)
```

### 3.2 Method-by-method map

`plan_grasp` exists on both v1 `MotionGen` and v2 `MotionPlanner`/`BatchMotionPlanner`;
the v2 signature is in [`BATCH_INTERFACES.md`](./BATCH_INTERFACES.md) §1.3.

| v1 method                  | v2 target class        | v2 call                                                                                 |
|----------------------------|------------------------|-----------------------------------------------------------------------------------------|
| `plan_single`              | `MotionPlanner`        | `plan_pose(goal, current_state, max_attempts, enable_graph_attempt)`, `G=1`             |
| `plan_goalset`             | `MotionPlanner`        | `plan_pose(goal, current_state, ...)`, `G>1` (auto-detected from `num_goalset`)         |
| `plan_batch`               | `BatchMotionPlanner`   | `plan_pose(goal, current_state, ...)`, cfg: `multi_env=F, max_goalset=1`                |
| `plan_batch_goalset`       | `BatchMotionPlanner`   | `plan_pose(goal, current_state, ...)`, cfg: `multi_env=F, max_goalset=G`                |
| `plan_batch_env`           | `BatchMotionPlanner`   | `plan_pose(goal, current_state, ...)`, cfg: `multi_env=T, max_goalset=1`                |
| `plan_batch_env_goalset`   | `BatchMotionPlanner`   | `plan_pose(goal, current_state, ...)`, cfg: `multi_env=T, max_goalset=G`                |
| `plan_grasp(...)`          | `MotionPlanner.plan_grasp` / `BatchMotionPlanner.plan_grasp` | same signature shape (approach/grasp/lift), see `BATCH_INTERFACES.md` |
| `plan_single_js`           | `MotionPlanner.plan_cspace` / `BatchMotionPlanner.plan_cspace` | joint-space goals                                                |

Every v1 call takes `start_state: JointState, goal_pose: Pose,
plan_config: MotionGenPlanConfig, link_poses`. In v2:

- `start_state` → `current_state: JointState` (same thing; note `(1, dof)` for
  `MotionPlanner`, `(B, dof)` for `BatchMotionPlanner`).
- `goal_pose + link_poses` → single `GoalToolPose` per §1.2.
- `MotionGenPlanConfig.max_attempts / enable_graph / enable_graph_attempt /
  enable_finetune_trajopt / time_dilation_factor` → mostly keyword args on
  `plan_pose` itself (e.g. `max_attempts`, `enable_graph_attempt`). Finetune /
  time dilation live on `MotionPlannerCfg` (see
  `curobo/_src/motion/motion_planner_cfg.py`).
- `PoseCostMetric` (constrained planning, `reach_partial_pose`, etc.) →
  `ToolPoseCriteria` in v2, set via `planner.update_tool_pose_criteria({link: criteria})`.

### 3.3 Result accessors

```python
# v1
result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
if result.success.item():
    traj = result.get_interpolated_plan()
    traj = motion_gen.get_full_js(traj)            # pad locked joints back in

# v2
result = planner.plan_pose(goal, current_state)    # Optional[TrajOptSolverResult]
if result is not None and result.success.any():
    traj = result.get_interpolated_plan()
    traj = planner.kinematics.get_full_js(traj)    # same idea
```

Batch-env result: v1's `result.get_paths()` returns a `List[JointState]`, one
per successful env (variable lengths). v2 has no equivalent helper; iterate
the success mask yourself:

```python
# v1
trajs = result.get_paths()
for s in range(len(result.success)):
    if result.success[s]:
        apply(trajs[s])

# v2
interp = result.get_interpolated_plan()            # (B, H, dof)
for s in range(result.success.shape[0]):
    if result.success[s].any():
        apply(interp[s])                           # interp[s]: (H, dof)
```

---

## Chapter 4 — MPC mapping

v1 class: `curobo.wrap.reacher.mpc.MpcSolver` (6 `setup_solve_*` + `step` +
`update_goal`).
v2 class: `curobo.model_predictive_control.ModelPredictiveControl` (`setup` +
`update_goal_tool_poses` + `optimize_action_sequence` /
`optimize_next_action`).

### 4.1 Config

```python
# v1
mpc_config = MpcSolverConfig.load_from_robot_config(
    robot_cfg, world_cfg,
    use_cuda_graph=True,
    use_cuda_graph_metrics=True,
    use_cuda_graph_full_step=False,
    self_collision_check=True,
    collision_checker_type=CollisionCheckerType.MESH,
    collision_cache={"obb": 30, "mesh": 10},
    use_mppi=True, use_lbfgs=False, use_es=False,
    store_rollouts=True,
    step_dt=0.02,
)
mpc = MpcSolver(mpc_config)

# v2
cfg = ModelPredictiveControlCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    use_cuda_graph=True, self_collision_check=True,
    optimization_dt=0.02,                 # renamed from step_dt
    interpolation_steps=4,
    # mode:
    max_batch_size=1, multi_env=False, max_goalset=1,
)
mpc = ModelPredictiveControl(cfg)
```

Solver-internal flags (`use_mppi / use_lbfgs / use_es`) are now selected via
`optimizer_configs` in the v2 config (defaults to an MPPI+LBFGS combo).

### 4.2 Lifecycle map

```python
# v1
retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
joint_names = mpc.rollout_fn.joint_names
state = mpc.rollout_fn.compute_kinematics(
    JointState.from_position(retract_cfg, joint_names=joint_names))
retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
goal = Goal(current_state=current_state,
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
            goal_pose=retract_pose)
goal_buffer = mpc.setup_solve_single(goal, 1)
mpc.update_goal(goal_buffer)
# ...
goal_buffer.goal_pose.copy_(ik_goal)      # update target between steps
mpc.update_goal(goal_buffer)
mpc_result = mpc.step(current_state, max_attempts=2)
action = mpc_result.js_action

# v2
current_state = JointState.from_position(mpc.default_joint_position.unsqueeze(0),
                                         joint_names=mpc.joint_names)
current_state.velocity     = torch.zeros_like(current_state.position)
current_state.acceleration = torch.zeros_like(current_state.position)

mpc.setup(current_state)                  # replaces Goal() + setup_solve_*

kin = mpc.compute_kinematics(current_state)
mpc.update_goal_tool_poses(
    GoalToolPose.from_poses(kin.tool_poses.to_dict(),
                            ordered_tool_frames=mpc.tool_frames,
                            num_goalset=1),
    run_ik=True,                          # optional seed-IK
)
# ...
# update target between steps:
mpc.update_goal_tool_poses(
    GoalToolPose.from_poses({mpc.tool_frames[0]: ik_goal},
                            ordered_tool_frames=mpc.tool_frames,
                            num_goalset=1),
    run_ik=False,
)
result = mpc.optimize_action_sequence(current_state)   # full horizon
# or: mpc.optimize_next_action(current_state) for a single command
action = result.action_sequence            # JointState over the horizon
```

### 4.3 Method-by-method map

| v1 `setup_solve_*`                     | v2 config                                                           |
|----------------------------------------|---------------------------------------------------------------------|
| `setup_solve_single`                   | `max_batch_size=1, multi_env=F, max_goalset=1`                      |
| `setup_solve_goalset`                  | `max_batch_size=1, multi_env=F, max_goalset=G`                      |
| `setup_solve_batch`                    | `max_batch_size=B, multi_env=F, max_goalset=1`                      |
| `setup_solve_batch_goalset`            | `max_batch_size=B, multi_env=F, max_goalset=G`                      |
| `setup_solve_batch_env`                | `max_batch_size=B, multi_env=T, max_goalset=1`                      |
| `setup_solve_batch_env_goalset`        | `max_batch_size=B, multi_env=T, max_goalset=G`                      |

Per-step:

| v1                                                     | v2                                                                           |
|--------------------------------------------------------|------------------------------------------------------------------------------|
| `mpc.step(current_state, max_attempts=2)` → `.js_action` | `mpc.optimize_next_action(current_state)` → `.next_action`                  |
| `mpc.step(...)` with buffer access → `.action_buffer`  | `mpc.optimize_action_sequence(current_state)` → `.action_sequence`           |
| `goal_buffer.goal_pose.copy_(new_pose)` + `mpc.update_goal(goal_buffer)` | `mpc.update_goal_tool_poses(GoalToolPose…, run_ik=…)`    |
| `mpc.get_visual_rollouts()`                            | `mpc.core.get_visual_rollouts()` (still available via `SolverCore`)          |
| `mpc.world_coll_checker.load_collision_model(obstacles)` | `mpc.scene_collision_checker.update_obstacle_pose(...)` or `mpc.update_world(scene)` |

---

## Chapter 5 — Per-example migration checklist

Files under `~/magicsim/MagicCurobo/examples/isaac_sim/**` that matter for
this migration, grouped by v2 target solver.

### IK examples

| file                                        | v1 method(s)                       | v2 target                                   | notes                                                                                                    |
|---------------------------------------------|------------------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `ik_reacher.py`                             | `IKSolver.solve_single`            | `InverseKinematics` (`max_batch_size=1`)    | baseline single-EE IK; direct port                                                                       |
| `ik_reachability.py`                        | `IKSolver.solve_batch`             | `InverseKinematics` (`max_batch_size=B`)    | 100×pose grid reachability; maps to v2 batched solve_pose (`batched_ik_example` in v2 examples)          |
| `multi_arm_humanoid_ik.py`                  | `IKSolver.solve_single` + `link_poses` | `InverseKinematics` with `L>1`          | merge `goal_pose` + `link_poses` via `GoalToolPose.from_poses`; set `tool_frames` in robot YAML          |
| `multi_arm_humanoid_parallel_ik.py`         | `IKSolver.solve_batch_env`          | `InverseKinematics` (`multi_env=T, L>1`)    | classic batch-env multi-EEF IK; `world_cfg_list` → `multi_env=True`                                      |
| `multi_arm_humanoid_bottle_grasp_goalset.py`| `IKSolver.solve_goalset`            | `InverseKinematics` (`max_goalset=G`)       | `goal_pose` has shape `(1,G,3/4)` → v2 `(1,1,L,G,3/4)`; `result.goalset_index` semantics unchanged       |

### MotionGen examples

| file                                  | v1 method(s)                     | v2 target                                                              | notes                                                                                     |
|---------------------------------------|----------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `motion_gen_reacher.py`               | `MotionGen.plan_single`          | `MotionPlanner.plan_pose`                                              | reactive / constrained flags map to `ToolPoseCriteria` + planner kwargs                   |
| `single_plan_motion_gen_reacher.py`   | `MotionGen.plan_single`          | `MotionPlanner.plan_pose`                                              | same                                                                                      |
| `constrained_reacher.py`              | `MotionGen.plan_single` + `PoseCostMetric` | `MotionPlanner.plan_pose` + `update_tool_pose_criteria`       | `reach_partial_pose / hold_partial_pose` → `ToolPoseCriteria.linear_motion` / partial rpy |
| `motion_gen_reacher_nvblox.py`        | `MotionGen.plan_single` + nvblox | `MotionPlanner` + v2 nvblox collision checker                          | collision side; solver migration same as `motion_gen_reacher.py`                          |
| `motion_gen_reacher_autotest.py`      | `MotionGen.plan_single`          | `MotionPlanner.plan_pose`                                              | add tests under `curobo/tests/` once ported                                               |
| `motion_gen_humanoid.py`              | `MotionGen.plan_single` (humanoid) | `MotionPlanner` (robot YAML with many tool_frames)                   | multi-EEF humanoid plan                                                                   |
| `dual_motion_gen_humanoid.py`         | `MotionGen.plan_single` ×2       | two `MotionPlanner` instances OR one with dual-arm robot YAML          | file constructs a “dual” wrapper → prefer dual-arm robot YAML in v2                       |
| `multi_arm_humanoid.py`               | `MotionGen.plan_single` (multi EE) | `MotionPlanner` with `L>1`                                           | merge goal + link poses                                                                   |
| `multi_arm_reacher.py`                | `MotionGen.plan_single` (multi EE) | `MotionPlanner` with `L>1`                                           | same                                                                                      |
| `batch_motion_gen_reacher.py`         | `MotionGen.plan_batch_env`       | `BatchMotionPlanner.plan_pose` (`multi_env=T`)                         | list-of-worlds → `multi_env=True`; `get_paths()` → iterate success mask                   |
| `dynamic_batch_motion_gen_reacher.py` | `MotionGen.plan_batch_env`       | `BatchMotionPlanner.plan_pose` (`multi_env=T`)                         | same; dynamic obstacle updates via `scene_collision_checker.update_obstacle_pose`         |
| `simple_stacking.py`                  | `MotionGen.plan_single` + grasp loop | `MotionPlanner.plan_grasp`                                          | v2 has first-class `plan_grasp` — prefer that                                             |
| `realsense_reacher.py`                | `MotionGen.plan_single` + realsense | `MotionPlanner` + v2 depth/esdf pipeline                            | solver migration same as `motion_gen_reacher.py`                                          |

### MPC examples

| file                                  | v1 method(s)                                    | v2 target                                   | notes                                                                                             |
|---------------------------------------|-------------------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------|
| `mpc_example.py`                      | `MpcSolver.setup_solve_single` + `step`         | `ModelPredictiveControl` (`max_batch_size=1`) | direct port of the reactive tracking loop                                                       |
| `mpc_nvblox_example.py`               | `MpcSolver.setup_solve_single` + `step` + nvblox | `ModelPredictiveControl` + nvblox          | same solver logic as above                                                                        |
| `realsense_mpc.py`                    | `MpcSolver.setup_solve_single` + `step` + realsense | `ModelPredictiveControl` + realsense    | same                                                                                              |

### Non-solver examples (no API change needed for this migration)

`batch_collision_checker.py`, `collision_checker.py`, `load_all_robots.py`,
`realsense_collision.py`, `realsense_viewer.py`, `helper.py`, `util/…` — these
touch collision checking / USD / I/O only. Port only if their collision or
robot-loading helpers need v2-specific replacements.

---

## Chapter 6 — Migration order we’ll follow

1. **IK — `solve_single`** (`ik_reacher.py`). Smallest surface, covers config
   translation, current_state plumbing, result handling.
2. **IK — `solve_batch`** (`ik_reachability.py`). Verifies `max_batch_size>1`
   padding and the batched `GoalToolPose`.
3. **IK — `solve_goalset`** (`multi_arm_humanoid_bottle_grasp_goalset.py`).
   Verifies `max_goalset` + `goalset_index` semantics.
4. **IK — multi-EEF single** (`multi_arm_humanoid_ik.py`). Verifies
   `goal_pose + link_poses` → `GoalToolPose` and `tool_frames` in the robot
   YAML.
5. **IK — `solve_batch_env`** (`multi_arm_humanoid_parallel_ik.py`). Verifies
   `multi_env=True` plus per-env obstacle updates. This is the riskiest step
   because v2's per-env obstacle plumbing diverges most from v1.
6. **MotionGen — `plan_single`** (`motion_gen_reacher.py`). Covers warmup,
   graph seeding, `get_interpolated_plan`, constrained planning via
   `ToolPoseCriteria`.
7. **MotionGen — `plan_batch_env`** (`batch_motion_gen_reacher.py`). Covers
   `BatchMotionPlanner` + result iteration without `get_paths()`.
8. **MotionGen — `plan_grasp`** (`simple_stacking.py`). Covers v2's first-class
   grasp planner (`approach → grasp → lift`), which was a hand-written loop in v1.
9. **MPC — single-env** (`mpc_example.py`). Last, because it depends on the
   IK warmup wrapper inside `MPCSolver`.

Each step produces:

- one ported example under `curobo/examples/` (or a new `curobo/examples/isaac_sim/` subdir),
- one test under `curobo/tests/` that exercises the new wrapper without Isaac Sim,
- a notes section appended to this file if we hit a gotcha that doesn't match
  the table above.
