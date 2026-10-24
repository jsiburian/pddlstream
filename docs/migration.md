# Migrating from earlier versions

Programs written for earlier versions of pddlstream keep working: the old import paths and keyword arguments are
still supported. This page lists what changed and what to use instead.

## Installation

- pddlstream needs Python 3.11 or newer. `pixi run setup` installs the dependencies and builds FastDownward (see the
  README).
- FastDownward is loaded the first time a solver needs it, so `import pddlstream` works without a build and no
  longer changes `sys.path` or `sys.argv`. Set `PDDLSTREAM_FD_PATH` to use a FastDownward checkout other than the
  `downward` submodule, for example after a non-editable install.

## Imports

The package is now layered (see [architecture.md](architecture.md)). The old paths still work, as modules that
re-export the new ones:

| Old path | New path |
|---|---|
| `pddlstream.language.constants` | `pddlstream.core.syntax` (expressions) and `pddlstream.core.problem` (problems, plans, results) |
| `pddlstream.language.object` | `pddlstream.core.objects` |
| `pddlstream.language.conversion` | `pddlstream.core.conversion` |
| `pddlstream.utils` | `pddlstream.core.utils` |
| `pddlstream.algorithms.downward` | `pddlstream.fastdownward.tasks` (plus `translator` and `planner`) |
| `pddlstream.algorithms.instantiate_task` | `pddlstream.fastdownward.grounding` |
| `pddlstream.algorithms.search` | `pddlstream.fastdownward.search` |

The common names can also be imported from the package itself: `from pddlstream import solve, PDDLProblem, And`.

## Options

- `solve()` takes the same keyword arguments, documented as the fields of `SolverConfig`
  (`pddlstream/core/config.py`). A misspelled or unknown option now raises a `TypeError` before solving starts,
  instead of failing inside FastDownward. A `SolverConfig` can be built once and passed as `solve(problem,
  config=config)`.
- Module flags that used to be assigned are now options. Assigning the old flags still works, with a warning:

| Old | New `solve()` option |
|---|---|
| `statistics.LOAD_STATISTICS = False` | `load_statistics=False` |
| `statistics.SAVE_STATISTICS = False` | `save_statistics=False` |
| `instantiate_task.FD_INSTANTIATE = False` | `fd_instantiate=False` (assigning the flag used to have no effect) |
| `exogenous.EXOGENOUS_AXIOMS = False` | `exogenous_axioms=False` |
| `exogenous.REPLACE_STREAM = False` | `exogenous_future_values=False` |
| `skeleton.USE_PRIORITIES`, `GREEDY_VISITS`, `GREEDY_BEST`, `REQUIRE_DOWNSTREAM` | `skeleton_priorities`, `greedy_visits`, `greedy_best`, `require_downstream` (the module flags are gone) |

## Results

The solvers return a `SolveResult`, which unpacks and compares like the old `(plan, cost, certificate)` tuple. It
adds `status` (`Status.SUCCEEDED`, `FAILED` or `INFEASIBLE`) and `stats`, the summary the solver prints. When a
problem is infeasible, the plan is still `None`, and `status` says so.

`INFEASIBLE` means that the solver ran out of things to try without cutting anything short. A solve that hit a skeleton
limit, ran out of time, had a FastDownward search time out or pruned streams by `max_effort` used to be reported
infeasible in some cases; it now ends `FAILED`, with `stats['termination']` (why it stopped) and `stats['gaps']` (what
was cut short). See [ADR-001](v2/adr-001-stream-outcomes.md).

The certificate is a `PlanCertificate`, which still unpacks as `(all_facts, preimage_facts)`. It adds `supports`,
the stream calls behind the preimage facts (each after those that support its domain), and
`pddlstream.algorithms.validate.verify_certificate` checks it. The incremental algorithm now records preimage facts
too, where it used to leave `preimage_facts` as `None`.

An exception raised by a stream now carries a note naming the call. `stream_errors='fail'` treats such a call as
exhausted instead of stopping the solve.

## Output and files

- Progress is reported through `logging` (loggers under `pddlstream`). Without a logging configuration it is still
  printed to stdout; `logging.getLogger('pddlstream').setLevel(logging.WARNING)` silences it.
- FastDownward's files go to a private temporary directory per search, so several solves can run from the same
  directory. `debug=True` keeps them in `./temp/`, as before.
- Stream statistics are saved in `~/.cache/pddlstream/statistics/` instead of `./statistics/`, which is still read.

## Reproducibility

- `solve(..., deterministic=True)` with a fixed `PYTHONHASHSEED` makes seeded runs repeat exactly.
- Solver state is reset after each solve, so a second `solve()` in one process behaves like the first.
- Iteration order no longer depends on memory addresses, so a solver may find a different (equally valid) plan
  than earlier versions did.

## Removed

- Temporal FastDownward and durative actions. The retiming helpers in `pddlstream.language.temporal` remain.
- pyplanners attachments: the incremental algorithm now rejects fluent streams with a clear error; use a focused
  algorithm such as adaptive.
- The hard-coded hooks for the Cerberus, Forbid, tflap, OPTIC and TPSHE planners.
- `pddlstream/retired`, the retired examples, and examples that no longer ran (NAMO, multi-robot turtlebots, the
  unfactored and RMMP continuous TAMP variants).
- The dummy, random and greedy stream reordering algorithms.
