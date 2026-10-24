# RoboLab

[RoboLab](https://github.com/NVlabs/RoboLab) (Apache-2.0) is NVIDIA's simulation benchmark for manipulation
policies, built on Isaac Lab. It has 120 tasks with automated success detection. Here, pddlstream acts as one of its
policies: the client in `client.py` sends observations to the planner server (`benchmarks/common/server.py`) and
executes the returned joint-position action chunks.

## Getting RoboLab

RoboLab is the opt-in submodule `RoboLab/`, pinned to an upstream commit. `git clone --recursive` and
`pixi run setup` skip it. To fetch it with its assets (Git LFS, about 7 GB):

```bash
git submodule update --init --checkout benchmarks/robolab/RoboLab
cd benchmarks/robolab/RoboLab && git lfs pull
```

RoboLab keeps its own environment (Python 3.11, Isaac Sim 5.0, Isaac Lab 2.2) and needs an RTX GPU with 48 GB:

```bash
cd benchmarks/robolab/RoboLab
uv venv --python 3.11 && uv sync --extra isaac50
```

The client side imports only `benchmarks/common/protocol.py`, `benchmarks/common/remote.py` and `client.py`. Run
it with the pddlstream checkout on `PYTHONPATH`. The planner server runs in pddlstream's environment.

## Files

| File | Contents |
|---|---|
| `client.py` | `TampInferenceClient`, RoboLab's `InferenceClient` for the planner server |
| `probe.py` | Saves each task's instruction, success conditions and objects as fixtures, and reports which tasks the planner supports |
| `export_assets.py` | Box approximations of a scene's objects from its USD file, for the planner's PyBullet mirror |
| `run.py` | Evaluates the planner on RoboLab tasks |

## Status

97 of RoboLab's 120 tasks have success conditions that the goal mapping covers. The main uncovered ones are
`object_outside_of_and_on_surface` (7 tasks) and `object_upright` (4).
