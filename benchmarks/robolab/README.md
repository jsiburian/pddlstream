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
PYTHONHASHSEED=0 pixi run python -m benchmarks.common.server --port 8765 &  # Plans repeat under a fixed hash seed
cd benchmarks/robolab/RoboLab
PYTHONPATH=../../.. .venv/bin/python -m benchmarks.robolab.run --task BananaInBowlTask --headless --num-envs 1
```

A planner error fails only its episode (the arm holds still); restarting on the same `--output-folder-name` skips
the runs RoboLab already finished.

On a SLURM cluster, `evaluate.sbatch` does both on one GPU node:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
sbatch --account=<account> --constraint=<rtx nodes> --output=robolab-%j.out \
    benchmarks/robolab/evaluate.sbatch --task BananaInBowlTask --num-envs 1
```

The planner's scene comes from the running environment: each object's bounding box (RoboLab's `WorldState`) and its
surface points, which `meshes.py` reads from the task's USD scene in the rigid body's frame. Objects that the success
conditionals put things in or on (bowls, bins, plates) are containers, which the planner keeps fixed:
- movable objects are their points' convex hulls, and grasps close across a slice of the object narrower than the
  gripper opens, so curved objects such as a banana fit;
- a container keeps only its base as a body (its hull would be solid), and objects are placed in it at its rim;
- an object counts as in a container when its center is inside the container's footprint, as RoboLab's
  `object_in_container` checks, so an object placed in a bowl may settle without a replan.

The table's height comes from where the objects rest: level with the robot's base. Two more things make RoboLab's
scenes plannable:
- some meshes lie on their side in their own frame; the planner plans such objects in an upright frame at the same
  origin, since grasps approach from above in the object's frame;
- most scenes are cluttered with objects the goal does not mention; the planner first plans with them fixed (as
  obstacles, for up to 20 s), and with them movable only if that finds no plan.

`replay.py` rebuilds the planner's problem of a recorded episode (RoboLab's `run_N.hdf5` and the task's scene) and
solves it on a CPU, with an explanation when it fails:

```bash
cd benchmarks/robolab/RoboLab  # In RoboLab's environment, with pxr (Isaac Sim's USD libraries or usd-core)
PYTHONPATH=../../.. .venv/bin/python -m benchmarks.robolab.replay export output/<folder>/<Task> --output problem.json
cd ../../.. && pixi run python -m benchmarks.robolab.replay solve benchmarks/robolab/RoboLab/problem.json
```

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

RoboLab's robot is DROID's Franka arm with a Robotiq 2F-85 gripper. A live probe
(`tests/fixtures/robolab_live_probe.json`) calibrated the planner's PyBullet mirror:
- its Franka arm matches RoboLab's to 0.001 mm at the home configuration;
- the Robotiq sits 18.1 mm past the flange, without the Panda hand's 45 degree turn, and its fingertips are 162.8 mm
  further; the client asks for `gripper='robotiq'`, whose tool frame is 1 cm above the fingertips and whose
  grasps fit its 85 mm opening;
- the arm's joint controller lags 1-2 cm behind a descent and the gripper takes about a second to close, so the arm
  holds still before and after the fingers move;
- when a replan starts with the gripper touching an object (after a slipped grasp), the tool first rises 10 cm.

## pi0.5 as a skill

When a planned grasp slips (a held object is not where the gripper should carry it), the planner leaves out its own
pick of that object and uses the `vla_pick` skill instead: the server answers with the skill's instruction ("pick up
the banana") rather than actions, the client runs a VLA, and once the object is off the table next to the gripper,
the planner plans the rest from the grasp it observes. When the planner gives up on an episode (it finds no plan,
or has no replans left), the VLA takes over the whole task from its instruction. On RoboLab the VLA is pi0.5 (DROID, joint positions), served
by [openpi](https://github.com/Physical-Intelligence/openpi) in its own environment and queried through RoboLab's
pi0-family client (`policies/pi0_family`), as RoboLab documents:

```bash
git clone https://github.com/xuningy/openpi.git ~/openpi && cd ~/openpi  # The fork RoboLab's docs use
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd <pddlstream>/benchmarks/robolab/RoboLab && uv pip install -e ~/openpi/packages/openpi-client
# On the GPU node (the checkpoint, 12 GB, downloads on first use):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 ~/openpi/.venv/bin/python ~/openpi/scripts/serve_policy.py --port 8000 \
    policy:checkpoint --policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
PYTHONPATH=../../.. .venv/bin/python -m benchmarks.robolab.run --task BananaInBowlTask --headless --vla-port 8000
```

The stand-in's `SlipperyBlock` task tests the same loop on a CPU, with a scripted picker as the VLA.

## Baseline

```bash
python -m benchmarks.robolab.baseline list --shard 1/2   # The supported tasks, in shards for parallel jobs
python -m benchmarks.robolab.baseline summarize benchmarks/robolab/RoboLab/output/<folders>
```

## Files

| File | Contents |
|---|---|
| `client.py` | `TampInferenceClient`, RoboLab's `InferenceClient` for the planner server; `OpenpiVLA` |
| `run.py` | Evaluates the planner with RoboLab's evaluation loop |
| `evaluate.sbatch` | The same on one SLURM GPU node, with the planner server |
| `probe.py` | Task fixtures and coverage (static), or observation layouts (live) |
| `meshes.py` | The objects' surface points from a scene's USD file (needs pxr: Isaac Sim, or usd-core on a CPU) |
| `baseline.py` | The supported tasks in shards, and a summary of RoboLab's episode results |
| `replay.py` | A recorded episode's planning problem, exported (RoboLab's environment) and solved (pddlstream's) |
