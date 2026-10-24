# PDDLStream V2 charter

V2 turns pddlstream into a planner that can be embedded in robot systems and extended by researchers and agents.
Streams report what happened. Plans carry certificates and failures are explained. A planning session survives
changes to the world, and new search strategies, executors and integrations plug in without editing the solver.

## Decisions

- **Grow, don't rewrite.** Each V2 concept is added inside the current engine, phase by phase. There is no second
  kernel. The package stays `pddlstream`, and existing programs keep working (see [migration](../migration.md)).
- **Recorded outputs are the contract.** `tests/recorded_outputs.json` pins the plans, costs and stream-call
  sequences of 35 deterministic runs. A change to them is deliberate: it gets its own commit and is reviewed.
- **The core stays light.** The core imports only the standard library and numpy. Nothing in the core imports an
  LLM, ROS, CUDA, a simulator or an agent protocol; those live in integrations or optional extras.
- **Evidence, not assertions.** A plan rests on initial facts and on stream results. Agents and learned components
  can propose goals, hints, samplers and skeletons, and rank alternatives. They never assert facts.
- **Honest failures.** Only a search that skipped nothing concludes that no plan exists. A timeout, a missing sample
  or a pruned stream ends with FAILED, never INFEASIBLE (see [ADR-001](adr-001-stream-outcomes.md)).
- **Integrations live outside the core.** RoboLab and other simulators use pddlstream through its public API, from
  their own packages. pddlstream is GPL-3.0, so an integration runs the planner as its own process when its host
  has a different license.
- **Abstractions arrive with their first user.** The problem model, the executor and the backend interface are each
  added in the phase that needs them.

## Vocabulary

| V2 term | In the code |
|---|---|
| Fact | An `Evaluation` (`core/syntax.py`), held in `evaluations` with the result that supports it |
| Stream call | One call of an `Instance` (`language/external.py`): a stream, function or predicate applied to input objects |
| Outcome | What one call did: yielded outputs, produced no sample yet, was exhausted, or raised an error |
| Witness | A real `Result`: outputs a stream produced, with the facts it certifies |
| Hypothesis | An `OptimisticObject` (`#o1`) and the facts assumed about it while planning |
| Support | The result that certifies a fact, back to the initial facts |
| Certificate | A plan's preimage facts and their supports |
| Session | A planner that keeps its stream results across solves as the world changes |

## Non-goals

- A new modeling language. PDDL plus stream declarations stays the input, and V2 may extend it (e.g. with types).
- Formal safety guarantees. A certificate says which facts a plan relies on and where they came from, relative to
  the models the streams used.
- A built-in learning system or LLM. Learning and agents are users of the API.
- Temporal planning inside the core. Temporal problems would go through a bridge (see [ADR-003](adr-003-schedulestream.md)).

## Phases

| # | Phase |
|---|---|
| 9 | Charter, outcome semantics, conformance tests |
| 10 | Stream outcomes and honest failure semantics |
| 11 | Provenance and certificates |
| 12 | Per-session state |
| 13 | Planner sessions and replanning |
| 14 | Traces and explanations |
| 15 | Problem model, lint and PDDL types |
| 16 | RoboLab I: a TAMP policy baseline |
| 17 | Agent tools |
| 18 | Strategy SDK |
| 19 | Parallel and batched stream execution |
| 20 | RoboLab II: demonstrations and learned skills |
| 21 | Learning hooks: trace datasets, policies that learn |
| 22 | Planner backend and bridges: [ADR-002](adr-002-planner-backend.md) (FastDownward 26.6), [ADR-003](adr-003-schedulestream.md) (ScheduleStream) |
| 23 | Release readiness |

The [feature matrix](features.md) lists which examples exercise which features under which algorithms.
