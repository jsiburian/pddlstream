# ADR-001: Stream outcomes and when a problem is infeasible

**Status:** accepted (Phase 9). Implemented in Phase 10.

## Context

A stream call can produce outputs, produce nothing this time, run out, or fail. Today the engine tracks only
counters on each instance: `num_calls`, `successful`, and `enumerated` (no further outputs). The solvers report
`Status.INFEASIBLE` when the queues they search are empty. Several paths reach that verdict after cutting the search
short. Examples are a skeleton limit, a timeout while sampling, a FastDownward search that timed out, and streams
pruned by `max_effort`. In those cases the problem may still be solvable (see `tests/test_semantics.py`).

## Decision

### Outcome of each call

Each call of a stream, function or predicate has exactly one outcome:

| Outcome | Meaning |
|---|---|
| `YIELDED` | The call produced at least one new output tuple or fact |
| `NO_SAMPLE_YET` | The call produced nothing new (no output, or only duplicates); later calls may |
| `EXHAUSTED` | The call produced nothing new and the procedure has no further outputs for these inputs |
| `ERROR` | The procedure raised an exception |

`TIMEOUT` and `CANCELLED` are reserved for an executor that can interrupt a call (Phase 19).

### What the outcome is evidence of

- **A yielded output** is a witness for the facts the stream certifies. This is the legacy assumption: a stream
  checks its own outputs. Stochastic or approximate procedures are trusted as declared.
- **Exhaustion** proves no more than the procedure's own enumeration. It is a sound negative only for a stream
  declared complete (`StreamInfo(complete=True)`), i.e. one whose outputs are exactly the values that satisfy its
  certified facts.
- **No sample, an error, a timeout or a skipped call** proves nothing.

### When a solve may report INFEASIBLE

- **Legacy mode (the default)** keeps today's rule for exhaustion: every exhausted instance counts as fully
  enumerated. A solve still reports `INFEASIBLE` only if nothing was cut short. If any of the following happened, it
  reports `FAILED` and names them in `stats['gaps']`:
  - the skeleton limit was reached;
  - time or memory ran out;
  - a FastDownward search timed out;
  - streams were pruned by `max_effort`;
  - facts above the complexity limit were never used;
  - a stream raised an error.
- **Strict mode** (`strict=True`) additionally requires every exhausted instance to belong to a complete stream.

### How the legacy wrappers map

| Procedure | Outcomes |
|---|---|
| `from_fn`, `from_list_fn`, `from_test`, `from_constant` | One call, which yields or not, then `EXHAUSTED` |
| `from_gen_fn`, `from_list_gen_fn` | `YIELDED` per output tuple; `NO_SAMPLE_YET` for a yielded `None` or `[]`; `EXHAUSTED` when the generator returns (`StopIteration`) |
| `from_sampler(f, max_attempts)` | As `from_gen_fn`, `EXHAUSTED` after `max_attempts` calls; this is not complete |
| Functions (`:function`) and predicates (`:predicate`) | One call, then `EXHAUSTED`; complete by definition |
| `WildOutput(enumerated=True)` | `EXHAUSTED` (the flag is ignored today; Phase 10 honors it) |

**Negated streams** (`negate=True`) treat a failed test as proof that the negated fact holds for those inputs. That
is sound only for complete tests. Strict mode warns about negated streams that are not marked complete.

## Alternatives considered

- **Strict by default.** This is sound, but most existing domains could then never be reported infeasible, because
  their wrappers are not declared complete. That would contradict the decision to keep existing programs working.
- **Recording evidence strength on every fact.** This would duplicate what the support chain already says once
  certificates exist (Phase 11). The outcome of each call plus the solve's gaps is enough to decide infeasibility.
- **Treating `from_fn` as complete by default.** `from_fn` is often a sampler with a single attempt, such as inverse
  kinematics from one seed, so this would be unsound. Completeness has to be declared.

## Consequences

- `Status.INFEASIBLE` becomes trustworthy in legacy mode: it means the solver's search space was exhausted without
  cutting anything short.
- `stats['termination']` and `stats['gaps']` say why a solve stopped. `explain()` (Phase 14) builds on them.
- Programs that relied on `INFEASIBLE` after a timeout now see `FAILED`. This is documented in the migration notes.
