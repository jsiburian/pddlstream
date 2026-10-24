# RoboLab integration

The RoboLab integration lives in its own repository, `pddlstream-robolab`, so that the core keeps no simulator
dependencies. It runs pddlstream as a task and motion planning policy for
[RoboLab](https://github.com/NVlabs/RoboLab):

- A planner server plans each episode with a `PlannerSession` in a PyBullet mirror of the scene (a typed tabletop
  domain for a Franka arm), returns joint-position action chunks, and replans when objects move.
- RoboLab's `InferenceClient` sends ground-truth observations to the server; the planner runs in its own process.
- A kinematic stand-in simulator with eight tasks runs without a GPU, for tests and development.
- Scripts probe RoboLab's tasks, export scene geometry and run evaluations on a GPU machine (see its README).

Success conditionals map to PDDL goals (`object_in_container`, `object_on_top`, `stacked`, `object_picked_up`);
tasks with other conditionals are reported as skipped. Where the geometric samplers run (the PyBullet mirror or
cuRobo) is decided by measurements on a GPU machine (its ADR-0001).
