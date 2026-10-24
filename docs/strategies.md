# Strategies

All four algorithms run the same loop. Each one is a pair of strategies:

- A **search strategy** proposes plans.
- A **refinement strategy** decides which streams to call, and when the solve is over.

`pddlstream.strategy` exposes the loop, the built-in strategies and `solve_with()`. With these you can try a new
algorithm without forking the solver. A plugin should import only `pddlstream.strategy`.

| Algorithm | Search | Refinement |
|---|---|---|
| incremental | `FiniteSearch`: plans with the facts found so far | `ExhaustiveRefinement`: calls every stream instance up to the complexity limit |
| focused | `OptimisticSearch`: plans with optimistic stream outputs | `GreedyRefinement(bind=False)`: calls the plan's streams in order |
| binding | `OptimisticSearch` | `GreedyRefinement(bind=True)`: also binds their outputs into later streams |
| adaptive | `OptimisticSearch` | `SkeletonRefinement`: keeps a queue of plan skeletons and divides its time between sampling them and searching |

## Composing a solver

`solve_with(problem, compose, ...)` parses and compiles the problem, builds a `SolveState` and calls
`compose(state)`. `compose` returns a `(search, refinement)` pair, and `solve_with` runs the pair on the loop. The
keyword arguments are `SolverConfig` fields, as for `solve()`. The adaptive algorithm is this composition:

```python
from pddlstream.strategy import OptimisticSearch, SkeletonRefinement, solve_with


def compose(state):
    return OptimisticSearch.from_state(state), SkeletonRefinement.from_state(state)


result = solve_with(problem, compose, max_time=30, unit_costs=True)
```

`from_state` builds a built-in strategy from the parsed problem: its streams, negated streams, optimizers and eager
streams. Keyword arguments are passed to the strategy's constructor, for example
`OptimisticSearch.from_state(state, reorder=False)` or
`SkeletonRefinement.from_state(state, search_sample_ratio=2)`.

## The loop

`drive(state, search, refinement)`:

1. Calls `refinement.initialize(state)` once.
2. Repeats until the store is terminated (solved, out of time or memory), or `max_iterations` or `max_complexity`
   is reached:
   1. Increments `state.iteration`.
   2. Calls `refinement.before_search(state)`.
   3. If `refinement.may_search(state)`, calls `search.propose(state, refinement)`. The result is an `OptSolution`
      `(stream_plan, opt_plan, cost)`. A failed search returns `FAILED` (`None`) or `INFEASIBLE` (`False`) in place
      of the plans. If `may_search` returns False, the search counts as `INFEASIBLE`.
   4. Calls `search.postprocess(state, solution)`, which can reorder and report the solution.
   5. Calls `refinement.after_search(state, solution)`, which calls streams and adds the plans it completes to
      `state.store`. If it returns True, the solve ends. Before ending, it sets `state.infeasible` if it proved that
      no plan exists.

`finish(state, refinement)` returns the `SolveResult` and adds `refinement.stats(state)` to its `stats`. The result
reports `INFEASIBLE` only if nothing was cut short (see `stats['gaps']`).

`SolveState` holds what the strategies share:

- the solution store (`store`, with `best_cost`, `evaluations` and the timers);
- the compiled `domain` and `goal_exp`;
- the `externals` (the parsed streams);
- the `constraints` and the search options (`search_kwargs`);
- the loop's counters: `iteration` and `complexity_limit` (raised by `complexity_step`).

## Writing a strategy

A strategy is any object with the protocol's methods (`SearchStrategy`, `RefinementStrategy`). It can wrap the
built-in strategies. `tests/plugins/round_robin.py` takes turns between search configurations:

```python
class RoundRobinSearch:
    def __init__(self, searches):
        self.searches = searches

    def current(self, state):
        return (state.iteration - 1) % len(self.searches)

    def propose(self, state, refinement):
        return self.searches[self.current(state)].propose(state, refinement)

    def postprocess(self, state, solution):
        return self.searches[self.current(state)].postprocess(state, solution)


def compose(state):
    searches = [
        OptimisticSearch.from_state(state, search_kwargs=dict(state.search_kwargs, planner=planner))
        for planner in ['ff-astar', 'dijkstra']
    ]
    return RoundRobinSearch(searches), SkeletonRefinement.from_state(state)
```

`OptimisticSearch.propose` uses `refinement.skeleton_queue`. A refinement used with it must therefore have one, as
the `FocusedRefinement` subclasses do.

**Capabilities.** A strategy that cannot handle some streams declares this in an `unsupported` attribute. The
attribute maps a capability (`'fluent'`, `'optimizer'`, `'negated'`, `'exogenous'`, as `describe()` names them) to
a message, which is formatted with the names of the streams that have that capability. `drive` raises a
`CapabilityError` with the message before solving:

```python
class ExhaustiveRefinement:
    unsupported = {'fluent': 'The incremental algorithm does not support fluent streams ({}); ...'}
```

**Rules.** Strategies can rank and choose what to try, but cannot add facts. Facts come only from stream calls
made through the store (`store.sample`, or the built-in refinements), so every plan keeps a certificate that
`verify_certificate` can check.

## Binding policies

The adaptive queue samples *bindings*: partial assignments of a skeleton's stream outputs. A `BindingPolicy` orders
them, and is passed as `solve(..., binding_policy=policy)` (or to `solve_with`). The policy has three methods:

- `priority(binding)` returns the sort key (lowest first). Any comparable value works.
- `is_greedy(binding)` says whether to keep sampling the binding right away.
- `is_standby(binding)` says whether to set the binding aside until the queue's next pass.

For example, `RoundRobinPolicy` in the plugin takes turns between skeletons among bindings of the same complexity:

```python
class RoundRobinPolicy(BindingPolicy):
    def __init__(self):
        super().__init__()
        self.turns = Counter()

    def priority(self, binding):
        default = super().priority(binding)
        self.turns[binding.skeleton.index] += 1
        return (default.not_greedy, default.complexity, self.turns[binding.skeleton.index], *default[2:])
```

## Learning

`rows_from_trace(result.trace, **labels)` turns the trace of a solve (`trace=True`) into dataset rows. There is a row
per stream call (stream, inputs, call index, outcome, results, whether the procedure ran) and a row per skeleton
(streams, actions, cost, how far binding it got). `write_jsonl` and `read_jsonl` store the rows.

`examples/learning/policies.py` has two binding policies that learn, written against the facade:

- `LearnedPolicy` learns each stream's success rate from the rows (`StreamStatistics.from_rows`). It treats a binding
  as more complex by the expected failures of its remaining streams.
- `ConflictPolicy` learns within a solve (conflict-directed sampling). Values introduced by bindings that could not
  be extended count as conflicts, and bindings that use them are tried later.

`python -m examples.learning.run` trains on two continuous TAMP problems and solves three held-out ones:

| Held-out problem | Default | Learned | Conflicts |
|---|---|---|---|
| tight, 2 blocks | 152 calls | 152 | 152 |
| blocked, 3 blocks | 100 | 100 | 100 |
| blocked, 4 blocks | 137 | 107 | 137 |
