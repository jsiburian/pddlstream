# Changelog

## 2.0.0a1 (2026-10-24)

The first V2 pre-release. Programs written for earlier versions keep working: see
[migrating from earlier versions](docs/migration.md) for what changed and the old names that still work. The design
notes are in [docs/v2](docs/v2/charter.md).

### Added

- **Honest outcomes.** Every stream call is recorded with its outcome (yielded, no sample yet, exhausted, error). A
  solve reports `INFEASIBLE` only when nothing was cut short. Otherwise it reports `FAILED`, with
  `stats['termination']` and `stats['gaps']`. `strict=True` also requires exhausted streams to be declared complete
  ([ADR-001](docs/v2/adr-001-stream-outcomes.md)).
- **Certificates.** A `PlanCertificate` lists the stream calls behind a plan's preimage facts, and
  `verify_certificate` checks it independently of the solver. This covers fluent streams, the incremental algorithm,
  and plans whose costs assume unevaluated cost functions (`unevaluated`).
- **Planner sessions.** `PlannerSession` keeps stream results between solves and takes `WorldPatch`es.
  `StreamInfo(reads=...)` says which results a change invalidates, and a report says what was reused
  ([sessions](docs/sessions.md)).
- **Traces and explanations.** `trace=True` and `on_event` expose the solver's events. `explain(result)` says why a
  solve ended as it did.
- **Problem model and lint.** `describe(problem)` gives a stable JSON model with capabilities and lint issues (with
  source lines). PDDL `:types` are supported.
- **Agent tools.** JSON tools over registered problems (describe, check streams, search, hints, goals, validation,
  explanations, traces), plus an optional MCP server (`pip install 'pddlstream[agent]'`). No tool accepts facts
  ([agents](docs/agents.md)).
- **Strategy SDK.** The algorithms are compositions of search and refinement strategies on one loop.
  `pddlstream.strategy` and `solve_with` run your own, and `BindingPolicy` orders the adaptive queue. Trace datasets
  (`rows_from_trace`) support learned policies ([strategies](docs/strategies.md)).
- **Executors.** Stream procedures can run on threads (`ThreadExecutor`), in worker processes (`ProcessExecutor`) or
  in batches (`from_batch_fn`). Calls are committed in the solver's order ([executors](docs/executors.md)).
- **Per-session state.** Object registries live in a context variable, so concurrent solves on different threads are
  isolated.
- **Examples.** Replanning (`continuous_tamp.replan`), a scripted agent (`agent.scripted_agent`), executors
  (`executors.run`) and learning policies (`learning.run`).
- **Robot benchmarks.** `benchmarks/` runs pddlstream as a task and motion planning policy on robot benchmarks, with
  RoboLab as an opt-in submodule and a CPU stand-in simulator. It supports learned skills, demonstrations and goals
  from language ([benchmarks](benchmarks/README.md), [overview](docs/v2/robolab.md)).

### Changed

- `pixi run bench` is now `pixi run timing` (`tests/timing.py`), so that it is not confused with `benchmarks/`.
- Unknown or misspelled solver options raise a `TypeError` before solving. Unsupported combinations raise
  `CapabilityError`.
- `solve()` passes `stream_info` to the incremental algorithm too.
- Example plans must have certificates that verify, unless they defer streams.

### Fixed

- Plans with fluent streams had incomplete certificates.
- `max_failures` was not honored by the focused and binding algorithms.
- `solve_serialized` used constants from an earlier solve.
- Solving modified the `StreamInfo` objects passed in.
- Replaying an instance's history made an extra call.
- A binding policy that never computes complexities crashed the skeleton queue.

### Decided

- FastDownward stays at the fork for 2.0 ([ADR-002](docs/v2/adr-002-planner-backend.md)).
- ScheduleStream gets no bridge in the core ([ADR-003](docs/v2/adr-003-schedulestream.md)).

## 0.1.0

The refactored package: layered modules, `SolverConfig`, typed results, deterministic solves, Python 3.11+, logging,
and the test harness with recorded outputs.
