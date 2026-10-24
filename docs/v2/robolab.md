# Robot benchmarks

The robot benchmark integrations live in [`benchmarks/`](../../benchmarks/README.md), outside the `pddlstream`
package, which never imports them. RoboLab is the first; LIBERO and RoboDojo can be added next to it. The
integration runs pddlstream as a task and motion planning policy:

- A planner server plans each episode with a `PlannerSession` in a PyBullet mirror of the scene (a typed tabletop
  domain for a Franka arm). It returns joint-position action chunks and replans when objects move.
- Each benchmark's client sends ground-truth observations to the server. The planner runs in its own process
  because every benchmark pins its own Python environment: RoboLab needs Python 3.11 with Isaac Sim, and LIBERO needs
  Python 3.8. The client modules need only the standard library and numpy.
- [RoboLab](https://github.com/NVlabs/RoboLab) is an opt-in submodule (`benchmarks/robolab/RoboLab`), pinned to an
  upstream commit. Its license, Apache-2.0, is compatible with pddlstream's GPL-3.0.
- A kinematic stand-in simulator with nine tasks runs without a GPU, for tests and development.

Success conditionals map to PDDL goals (`object_in_container`, `object_on_top`, `stacked`, `object_picked_up`).
These cover 97 of RoboLab's 120 tasks. Tasks with other conditionals are reported as unsupported. Measurements on a
GPU machine will decide where the geometric samplers run: the PyBullet mirror or cuRobo
([ADR-004](adr-004-robolab-samplers.md)).

The integration also uses pddlstream in two other roles:

- **Demonstrations.** Episodes on jittered and perturbed tasks that succeed with plans whose certificates verify are
  saved for training learned policies, with each plan's certificate digest. They are replayed open loop to check
  them.
- **Hybrid plans.** A learned skill (`SkillSpec`) is a planning action whose effects the planner assumes. Its cost is
  `-log p_success`. The policy runs the skill's policy until the simulator confirms the skill's success conditional,
  then adds the confirmed facts to the session's world and plans the rest.

Goals can also come from language. An agent proposes a goal for an instruction through the `propose_goal` tool
([agents](../agents.md)), which checks it against the problem before any search.
