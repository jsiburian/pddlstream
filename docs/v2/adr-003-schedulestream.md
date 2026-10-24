# ADR-003: ScheduleStream

Status: proposed (research only; no code). Date: 2026-10-24.

## Context

[ScheduleStream](https://github.com/NVlabs/ScheduleStream) (Garrett and Ramos, [arXiv:2511.04758](https://arxiv.org/abs/2511.04758),
ICRA 2026) is a framework for planning and scheduling with sampling operations. It targets robots that move several
arms at once:

- **Actions:** durative actions that can start asynchronously, with durations that depend on their parameters.
- **Algorithms:** its own domain-independent stream, finite, scheduling and search algorithms.
- **Language:** its own planning language (constants, predicates, actions, durative actions, streams, problems).
- **Applications:** cuRobo v2 on GPUs is the recommended backend; others are Isaac Lab demo generation, cuMotion,
  Trimesh 2D and blocksworld.
- **License:** the NVIDIA License, for non-commercial use.

From its README and project page, it does not build on this package's code or on FastDownward. Its streams play the
same role as PDDLStream's: generators of continuous parameters (grasps, IK, motions).

The V2 vision asked whether to bridge to it: export pddlstream problems to it, or call it as a backend.

## Options

1. **Vendor or import it in the core.** Ruled out. Its license (non-commercial) is incompatible with distributing it
   inside GPL-3.0 pddlstream, and the core has no GPU dependencies.
2. **Run it as a backend from pddlstream.** pddlstream has no durative actions (temporal FastDownward was removed in
   the refactor), so a backend could only receive sequential problems. That throws away the reason to use
   ScheduleStream. It would also need a translation of every stream to ScheduleStream's stream form, with procedures
   in the same process.
3. **Export the problem model.** `describe()` already gives a stable JSON `ProblemModel`: predicates, actions, streams
   with their domains and certified facts, and capabilities. A converter outside this repository could write
   ScheduleStream's language from it. The user would add durations and the robot-specific procedures. Nothing in the
   core changes.
4. **Do nothing now.** Revisit when a task needs parallel arms.

## Decision

Options 3 and 4: no bridge in the core for 2.0.

- If a RoboLab task needs two arms moving at once, write the converter in a separate repository (like
  pddlstream-robolab) from `describe()` output.
- Keep `ProblemModel` stable (its JSON is covered by tests) so that such a converter has a fixed input.

## Consequences

- pddlstream stays sequential. Multi-arm scheduling is out of scope for 2.0.
- Before writing a converter, check ScheduleStream's planning-language documentation (not read for this ADR). Check
  how its streams declare inputs, outputs and certified facts, and whether its non-commercial license fits the
  intended use.
