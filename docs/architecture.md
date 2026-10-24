# Architecture

PDDLStream plans in domains whose objects come from samplers. A problem is a PDDL domain plus *streams*:
Python procedures (samplers, tests, cost functions) declared in a stream file with the facts they need and the
facts they certify. The solvers interleave symbolic planning with FastDownward and calls to the streams.

## Layers

The package is split into five layers. Each layer imports only from the layers listed before it (`tools` is the last), and
`tests/test_layers.py` checks this.

| Package | Contents |
|---|---|
| `core` | The foundation: expression syntax (`syntax`: `And`, `Exists`, ...), problem and result types (`problem`: `PDDLProblem`, `SolveResult`, plan actions), objects and placeholders (`objects`), conversions between values, objects and facts (`conversion`), evaluations (`evaluations`), the solver's options (`config`), the per-solve context (`context`), console output (`log`) and general helpers (`utils`, `relation`). It depends only on the standard library and numpy. |
| `fastdownward` | The planner. `translator` loads FastDownward's Python translator on first use; `tasks` converts between pddlstream's facts and the translator's PDDL objects; `grounding` grounds tasks and translates them to SAS+; `planner` runs the compiled search in a private directory; `search` solves PDDL and SAS+ tasks, optionally with goal serialization or an ABSTRIPS hierarchy. |
| `language` | What users declare: streams (`stream`), functions and predicates (`function`), optimizers (`optimizer`), rules (`rule`), their shared machinery (`external`), the generator helpers (`generator`), stream statistics (`statistics`), stream-plan helpers (`plans`), and compilations of fluent and exogenous streams into the domain (`fluent`, `exogenous`). |
| `algorithms` | The solvers: `meta.solve` dispatches to `incremental` or to `focused` (focused, binding and adaptive), which build on parsing (`algorithm`), the solution store (`common`), optimistic planning (`refinement`, `scheduling/`), the skeleton queue (`skeleton`), plan constraints (`constraints`), and the alternatives `serialized` and `satisfaction`. |
| `tools` | Agent tools over registered problems (`toolbox`, `workspace`) and an optional MCP server (`mcp_server`); see [agents](agents.md). |

Some older module paths (for example `pddlstream.language.constants`, `pddlstream.utils` and
`pddlstream.algorithms.downward`) are short compatibility modules that re-export the moved code.
`pddlstream/__init__.py` exposes the commonly used names, which it imports on first use.

## A solve

1. `solve(problem, ...)` builds a `SolverConfig` from its arguments, rejects unknown options, and opens a solve
   scope. The scope holds a `SolveContext` (the configuration and the clock) that deep code reads with
   `get_config()` and `get_clock()`. When the outermost solve ends, the object registries are cleared, so every
   solve starts from the same state.
2. `parse_problem` parses the domain with the translator, turns the initial facts into *evaluations* (facts with
   the result that certified them and its *complexity*), and parses the stream file into externals: streams,
   functions, predicates and optimizers.
3. The algorithm alternates between planning and sampling:
   - **Incremental** calls every stream instance up to a complexity limit, then searches with the facts found so
     far, and raises the limit until a plan exists.
   - **Focused** plans with *optimistic* stream outputs (placeholder objects such as `#q0`), recovers the stream
     plan that the action plan relies on (`scheduling/plan_streams.py`), calls those streams, and repeats.
     **Binding** also binds the outputs it obtains into the rest of the stream plan, and **adaptive** keeps a queue
     of plan skeletons (`skeleton.py`) and divides its time between sampling them and searching for new ones.
4. Each search writes a SAS+ task to a private temporary directory and runs FastDownward on it
   (`fastdownward/planner.py`).
5. The solver returns a `SolveResult`: the plan, its cost and a certificate, plus a `status` (succeeded, failed
   or infeasible) and `stats` (iterations, search, sampling and total time, the outcomes of the stream calls, why
   the solver stopped and what it cut short). The certificate lists the stream calls behind the plan's preimage
   facts (`algorithms/certificate.py`), which `validate.verify_certificate` checks independently.

## Conventions

- Facts and formulas are tuples, such as `('On', block, region)` and `And(fact1, fact2)`.
- Progress goes through `logging` under the `pddlstream` logger. Without a logging configuration, solves print it
  to stdout.
- With `deterministic=True` (and a fixed `PYTHONHASHSEED`), decisions that depend on elapsed time use work units
  instead, and saved stream statistics are not loaded, so seeded runs repeat exactly.
- Stream statistics are saved in `~/.cache/pddlstream/statistics/`.

## Tests and tools

`pixi run test` runs every headless example through `tests/harness.py`. The harness checks each plan with the
validator in `pddlstream/algorithms/validate.py` and compares it with `tests/recorded_outputs.json`, which records
each deterministic case's plan, cost and stream calls. `pixi run test-gui`, `pixi run -e full test-gurobi` and
`pytest -m slow` cover the remaining examples. `pixi run bench` compares timings with a saved baseline, and
`pixi run lint` runs ruff and mypy.
