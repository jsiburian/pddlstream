# RoboLab

[RoboLab](https://github.com/NVlabs/RoboLab) (Apache-2.0) is NVIDIA's simulation benchmark for manipulation
policies, built on Isaac Lab. It has 120 tasks with automated success detection. Here, pddlstream acts as one of its
policies:
- RoboLab's evaluation loop calls the client in `client.py`;
- the client sends ground-truth observations to the planner server (`benchmarks/common/server.py`);
- the server returns joint-position action chunks, which the client executes.

## Setup

RoboLab is the opt-in submodule `RoboLab/`, pinned to an upstream commit. `git clone --recursive` and
`pixi run setup` skip it. To fetch it with its assets (Git LFS, about 6 GB) and install its environment (Python 3.11,
Isaac Sim 5.0, Isaac Lab 2.2; about 18 GB):

```bash
git submodule update --init --checkout benchmarks/robolab/RoboLab
cd benchmarks/robolab/RoboLab
git lfs pull
uv venv --python 3.11 && uv sync --extra isaac50
```

The installation needs no GPU. Running it does: an RTX GPU with 48 GB (Isaac Sim needs RT cores, so not an A100 or
H100). NVIDIA's Omniverse license must be accepted with `export OMNI_KIT_ACCEPT_EULA=YES`.

## Evaluate

The planner server runs in pddlstream's environment. The evaluation runs in RoboLab's environment, with the
pddlstream checkout on `PYTHONPATH`. It uses RoboLab's own evaluation loop and options (`--task`, `--tag`,
`--num-envs`, `--num-runs`, `--video-mode`, ...), and writes results to RoboLab's output folder.

```bash
pixi run python -m benchmarks.common.server --port 8765 &
cd benchmarks/robolab/RoboLab
PYTHONPATH=../../.. .venv/bin/python -m benchmarks.robolab.run --task BananaInBowlTask --headless --num-envs 1
```

On a SLURM cluster, `evaluate.sbatch` does both on one GPU node:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
sbatch --account=<account> --constraint=<rtx nodes> --output=robolab-%j.out \
    benchmarks/robolab/evaluate.sbatch --task BananaInBowlTask --num-envs 1
```

The planner's scene comes from the running environment, as boxes of the objects' bounding boxes. Objects that the
success conditionals put things in or on (bowls, bins, plates) are containers, which the planner keeps fixed. The
table's height comes from where the objects rest.

## Tasks

`python -m benchmarks.robolab.probe --output tests/fixtures/robolab_tasks.json` scans RoboLab's task definitions,
with no Isaac Sim needed. It saves each task's instruction, success conditionals and objects, and lists the tasks
the planner does not support. 97 of the 120 tasks are supported:
- `object_in_container` and `object_on_top`, with RoboLab's `all`, `any` and `choose` modes;
- `stacked`, in a given order or any order;
- `object_picked_up`.

The others need conditionals the domain does not model yet: `object_outside_of_and_on_surface` (7 tasks),
`object_upright` (4), relative placements (`object_left_of` and the like, 4), `object_outside_of` (3), and ordered
or grouped placement (3).

`python -m benchmarks.robolab.probe --live --task BananaInBowlTask --headless --output live.json` starts Isaac Sim
and saves each task's first observation layout and ground-truth state, with the robot's joint positions and body
poses.

## The robot

RoboLab's robot is DROID's Franka arm with a Robotiq 2F-85 gripper. The planner's PyBullet mirror models the
Franka arm, whose kinematics match, but with the Panda gripper. The Robotiq fingertips are 162.8 mm from its mounting
flange, so the mirror's tool frame needs calibrating from a live probe (the gripper's pose against the arm's joint
positions). Until then, grasps are expected to miss.

## Files

| File | Contents |
|---|---|
| `client.py` | `TampInferenceClient`, RoboLab's `InferenceClient` for the planner server |
| `run.py` | Evaluates the planner with RoboLab's evaluation loop |
| `evaluate.sbatch` | The same on one SLURM GPU node, with the planner server |
| `probe.py` | Task fixtures and coverage (static), or observation layouts (live) |
| `export_assets.py` | Box approximations of a scene's objects from its USD file, offline (needs usd-core) |
