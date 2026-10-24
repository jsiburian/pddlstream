# Executors

A stream call has two parts:

1. The **procedure run** calls your procedure for the call's outputs. It touches only that instance's generator.
2. The **commit** turns the outputs into objects and results and updates the statistics. Commits always happen on the
   solver's thread, in the order the algorithm makes its calls.

An *executor* decides where procedure runs happen. Pass one as `solve(..., executor=...)` (a `SolverConfig` field).
The default runs each procedure on the solver's thread when its call is made.

A parallel executor also starts the calls that the algorithm is likely to make next (*prefetching*), so procedures
run while the solver is busy:

- the incremental algorithm prefetches the next instances in its queue;
- the adaptive algorithm prefetches the instances of the next skeleton bindings it will revisit.

Commits keep their order. A solve whose procedures are *pure* therefore makes the same calls and finds the same plans
with any executor. A procedure is pure if its outputs depend only on its inputs and on that instance's earlier calls,
for example if it is seeded per instance rather than sharing a global random generator.

```python
from pddlstream import ThreadExecutor, StreamInfo, solve

with ThreadExecutor(workers=8) as executor:
    result = solve(problem, executor=executor)
print(result.stats['resources'])  # {'default': {'calls': 33, 'seconds': 0.31}}
```

`examples/executors/run.py` solves one problem three ways (serially, with threads, with processes) and prints the
time and procedure use of each.

## Thread executor

`ThreadExecutor(workers=4, resources=None, lookahead=None)` runs procedures on threads. Use it when the procedures
spend their time outside the Python interpreter, as numpy, simulators, network calls and GPU models do.

- A stream can name a pool with `StreamInfo(resource='gpu')`. `resources={'gpu': 1}` sets that pool's size.
- `StreamInfo(resource='main')` keeps a procedure on the solver's thread, for procedures that are not thread-safe.
- An instance's generator is never entered by two threads at once.
- Exceptions are raised on the solver's thread, with a note naming the call, as without an executor.

## Process executor

`ProcessExecutor(make_procedures, resource='process', workers=2, seed=0)` runs the procedures of streams with
`StreamInfo(resource='process')` in worker processes. Use it for procedures that must not share a process, such as
simulators with global state.

- Each worker calls `make_procedures()` once to build its own procedures (a dict by stream or function name) and
  seeds `random` and numpy with `seed` plus its index.
- Each instance always runs on the same worker, which keeps that instance's generator.
- Procedures of other resources run on the solver's thread.
- `make_procedures` must be importable in a new process (a module-level function in a regular package).
- Inputs and outputs must be picklable.
- Workers start on first use, which takes a second or two.

## Batch procedures

`from_batch_fn(batch_fn)` wraps a function from a list of input tuples to a list of output lists, one per input, for
example a vectorized GPU sampler. Each instance is called once. Without a parallel executor, it gets one input tuple
per call. With `ThreadExecutor(lookahead=k)`, the prefetched calls of a batch procedure (up to k) are passed to it
together.

## Determinism and accounting

`deterministic=True` needs the default executor: concurrent procedures that share state (such as a global random
generator) could produce different outputs on each run. With an executor, `stats['resources']` reports each
resource's procedure calls and running time. Time spent on prefetched procedures overlaps the search, so it is not
counted in `sample_time`.

Not covered yet: the focused and binding algorithms call their stream plans one call at a time, without
prefetching. The adaptive algorithm does not prefetch a binding's first call, since binding its stream result early
would create objects out of order.
