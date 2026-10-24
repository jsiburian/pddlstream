# ADR-004: Where RoboLab's geometric samplers run

**Status:** proposed. The default is decided; switching needs the GPU measurements below.

## Context

The planner needs grasps, placements, inverse kinematics, motions and collision checks for the scene in RoboLab. Two
ways to provide them:

1. **A PyBullet mirror** (implemented): the planner process rebuilds the scene from ground-truth poses and box
   approximations of the objects (`benchmarks/robolab/export_assets.py`) and samples with pybullet-planning on a CPU.
2. **cuRobo** in the RoboLab process or a GPU worker: batched IK and collision-free motion generation against the
   real meshes.

## Decision

Use the PyBullet mirror by default: it runs anywhere (including CI and this machine), keeps the planner out of the
simulator's GPU memory, and the stream names stay the same whichever backend runs them. Switch the motion and IK
streams to cuRobo only if the measurements show it is clearly better.

## Criteria (fixed before measuring)

Measured on the eight stand-in tasks and twenty supported RoboLab tasks, with fixed seeds and the same time limits:

| Criterion | Measure |
|---|---|
| Success | Episodes whose success conditionals hold |
| Planning latency | Median and 95th percentile time to the first plan, and per replan |
| Geometry mismatch | Execution failures caused by the box approximations (collisions or missed grasps in Isaac Sim) |
| Resources | GPU memory next to Isaac Sim; CPU and GPU seconds per episode |
| Install friction | Steps to install in RoboLab's environment |
| Determinism | Whether seeded episodes repeat |

## Consequences

- With the mirror, geometric fidelity is limited to boxes; tasks with thin or concave objects may need better shapes
  (convex hulls from the USD meshes) before cuRobo is worth its cost.
- If cuRobo wins, its batched IK fits in a single stream call (many seeds at once), so it does not need pddlstream's
  parallel executor; generating demonstrations at scale would (V2 Phase 19).
