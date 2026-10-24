# Benchmarks

pddlstream as a task and motion planning (TAMP) policy on robot manipulation benchmarks. RoboLab is supported now,
and other benchmarks (LIBERO, RoboDojo, ...) can be added next to it. A stand-in simulator runs the same policy on
a CPU, for tests and development.

The policy reads the simulator's ground-truth object poses. It plans pick-and-place sequences with a
`PlannerSession` (adaptive algorithm) and returns joint-position action chunks for a Franka arm. When an object is not
where the plan expects it, the policy patches the session's world and replans, reusing the stream results that are
still valid.

This directory is not part of the `pddlstream` package. Nothing in `pddlstream/` imports it.

## Layout

| Directory | Contents |
|---|---|
| `common/` | What every benchmark shares: the planning domain, the PyBullet mirror of the scene, the policy, the planner server and its client, learned skills, goals from conditionals and from language |
| `standin/` | The stand-in simulator (kinematic PyBullet, no GPU), its nine tasks, episode recording and demonstrations |
| `robolab/` | [RoboLab](https://github.com/NVlabs/RoboLab): its client, task probe, scene export and evaluation script. `RoboLab/` is an opt-in submodule; see [robolab/README.md](robolab/README.md) |

## The planner runs as a server

```
Benchmark process (its own Python, GPU)          Planner process (pddlstream's environment, CPU)
  client (e.g. robolab/client.py)  -- JSON over HTTP -->  common/server.py: TampPolicy
                                                            PlannerSession + PyBullet mirror of the scene
```

Each benchmark pins its own environment. RoboLab needs Python 3.11 with Isaac Sim 5.0, and LIBERO needs Python 3.8
with torch 1.11, so neither can share pddlstream's interpreter. The benchmark side imports only the client modules
(`common/protocol.py`, `common/remote.py` and its own client), which need the standard library and numpy, not
pddlstream. `tests/test_benchmarks.py` checks this.

| Module | Contents |
|---|---|
| `common/protocol.py` | Task specifications (`TaskSpec`, `Conditional`) and observations (plain data) |
| `common/goals.py` | PDDL goals from success conditionals (RoboLab's names); tasks with other conditionals raise `UnsupportedTask` |
| `common/domain.pddl`, `streams.pddl` | A typed tabletop domain: pick, place, free and holding motions, collision tests |
| `common/world.py`, `primitives.py` | The PyBullet scene and the stream procedures (grasps, placements, IK, motions) |
| `common/policy.py` | `TampPolicy`: plan, compile the plan into action chunks, monitor, run skills and replan |
| `common/server.py`, `remote.py` | The planner server, and its client with `start_server` |
| `common/skills.py` | Learned skills as planning actions (`SkillSpec`), and a scripted button press standing in for a VLA |
| `common/language.py` | Goals from instructions: a scripted agent, checked with pddlstream's `propose_goal` tool, and scoring |

Supported success conditionals: `object_in_container`, `object_on_top`, `stacked`, `object_picked_up` and
`object_grabbed`, plus those of the skills in use (`button_pressed` with the `press` skill).

## The stand-in (no GPU)

In pddlstream's environment, from the checkout:

```bash
python -m benchmarks.standin.run                    # Nine tasks; one JSON line per episode
python -m benchmarks.standin.run --perturb --plans  # Move an object early in each episode, and print the plans
pixi run test-benchmarks                            # The episodes, demonstrations and language tests
```

`pixi run test` runs the quick tests in `tests/test_benchmarks.py` (goals, protocol, skills). The episode tests take
about two minutes and are marked `benchmarks`.

## Learned skills (hybrid plans)

A `SkillSpec` turns a learned policy into a planning action. It has a PDDL precondition and effect, a prior success
probability (the action costs `-log p`), a timeout, and the success conditional that confirms it. The planner treats
the skill's effects as assumptions: each plan report lists them, and the plan's certificate verifies up to them.

When execution reaches the skill, the policy hands control to the skill's policy until the simulator reports the
success conditional (`Observation.conditions`). It then adds the confirmed facts to the session's world and plans the
rest. In `PressThenPlace` the domain cannot press buttons: `ScriptedPress` (a stand-in for a VLA) does, and the
planner places the block afterwards (`python -m benchmarks.standin.run --task PressThenPlace --plans`). On a real
benchmark, a skill's `make_policy` would query a VLA server, and the client must report the skill's conditional.

## Demonstrations

`python -m benchmarks.standin.generate_demos --output demos --seeds 0 1 2 --perturb` runs each task on jittered
initial positions, with and without a perturbation. It keeps the episodes that succeeded with plans whose
certificates verify, and saves them as `.npz` files with an `index.jsonl`. Each file holds the actions, the observed
arm and object states, and JSON metadata: the instruction, plan reports with certificate digests, and the
perturbation. It then replays each one open loop.

## Goals from language

`python -m benchmarks.standin.score_language` turns each task's instruction into a goal with `ScriptedAgent`. It
checks the goal with pddlstream's `propose_goal` tool (predicates, arities, objects, and whether each predicate can
become true) and compares it with the goal derived from the task's success conditionals. An LLM agent can use the
same tool through pddlstream's MCP server, with a `Workspace` holding `language.symbolic_problem(scene)`.

## Adding a benchmark

1. Add the benchmark under `benchmarks/<name>/`, with its repository as an opt-in submodule (`update = none` in
   `.gitmodules`).
2. Write its client in the benchmark's own Python: observations into `Observation`, success conditions into
   `TaskSpec`. Use `common/remote.py` to talk to the planner.
3. Map its success conditions to goals: extend `common/goals.py`, or translate them to the shared conditional names.
4. If its robot or objects differ, extend `common/world.py` (URDF, tool frame) and the scene export.

## Status

- **Stand-in:** 9/9 tasks succeed, including the hybrid task, and every plan's certificate verifies. With an object
  the robot has not handled yet moved early in each episode, 8/9 succeed. Replanning StackThree after its base moves
  takes longer than the policy's 60 s budget on a heavily loaded machine.
- **Demonstrations:** they replay open loop in the stand-in.
- **Language:** the scripted agent gets 9/9 stand-in instructions right, and `propose_goal` accepts each goal.
- **RoboLab:** see [robolab/README.md](robolab/README.md).
