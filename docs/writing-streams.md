# Writing streams

A PDDLStream problem is a `PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)`:

- `domain_pddl`: an ordinary PDDL domain (as a string).
- `constant_map`: the value of each constant in the domain, e.g. `{'@table': table}`.
- `stream_pddl`: the stream declarations (as a string), described below.
- `stream_map`: the Python procedure for each stream and function, by name.
- `init`: the initial facts as tuples, e.g. `[('Block', 'b1'), ('AtPose', 'b1', pose)]`. Arguments may be any
  Python values (numpy arrays included); pddlstream wraps them in objects.
- `goal`: a formula, e.g. `And(('Holding', 'b1'))` or `Exists(['?p'], ('AtPose', 'b1', '?p'))`.

```python
from pddlstream import solve, PDDLProblem, And, from_fn, from_gen_fn, from_test, print_solution

problem = PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)
result = solve(problem, algorithm='adaptive', unit_costs=True)
print_solution(result)
plan, cost, certificate = result
```

## Streams

A stream declares the facts its inputs must satisfy (`:domain`), the outputs it produces, and the facts that hold
for each output tuple (`:certified`):

```lisp
(define (stream tamp)
  (:stream sample-pose
    :inputs (?b ?r)
    :domain (Placeable ?b ?r)
    :outputs (?p)
    :certified (and (Contain ?b ?p ?r) (Pose ?b ?p)))
  (:stream test-cfree
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CFree ?b1 ?p1 ?b2 ?p2))
  (:function (Distance ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))))
```

A stream without outputs is a test. Its procedure receives the input values and returns a generator of lists of
output tuples, one list per call. The helpers in `pddlstream.language.generator` build that from simpler
procedures:

| Helper | Procedure it wraps |
|---|---|
| `from_gen_fn(f)` | a generator function that yields one output tuple per call (or None for "nothing this time") |
| `from_fn(f)` | a function that returns one output tuple, or None; it is called once |
| `from_list_fn(f)` | a function that returns a list of output tuples; it is called once |
| `from_sampler(f, max_attempts)` | a function that samples one output tuple per call |
| `from_test(f)` | a function that returns whether the certified facts hold (for tests) |
| `from_list_gen_fn(f)` | a generator function that yields lists of output tuples |

```python
def sample_poses(b, r):
    while True:  # Each call yields one more pose
        yield (sample_pose(b, r),)


stream_map = {
    'sample-pose': from_gen_fn(sample_poses),
    'test-cfree': from_test(lambda b1, p1, b2, p2: not collision(b1, p1, b2, p2)),
    'Distance': lambda q1, q2: np.linalg.norm(q2 - q1),
}
```

## Functions and costs

A `:function` declares a numeric function of its arguments, whose procedure returns a nonnegative number. Actions
use it as a cost, e.g. `(increase (total-cost) (Distance ?q1 ?q2))`. During planning, the solver uses an
optimistic value: 0 by default, or `FunctionInfo(opt_fn=...)`. A `:predicate` declares an external predicate,
which is evaluated as needed instead of certified by a stream.

## Stream options

`solve(..., stream_info={'name': StreamInfo(...)})` changes how a stream is used. The most useful options:

- `p_success` and `overhead`: prior estimates of how often a call succeeds and how expensive it is. The solvers
  use them to order streams and to prefer cheap plans, and update them from saved statistics.
- `eager=True`: call the stream on every new input before searching (good for cheap tests).
- `negate=True`: also allow the test to appear negated in the domain (for example, a collision test used as
  "not in collision").
- `opt_gen_fn=PartialInputs(...)`: which inputs the optimistic placeholder outputs depend on.
- `defer_fn`: whether a call can wait until the plan reaches the step that needs it (`never_defer`,
  `defer_unique`, `defer_shared`). Deferred calls appear in the plan as `StreamAction` steps.

`FunctionInfo` and `PredicateInfo` do the same for functions and predicates.

## Fluent and wild streams

A stream with `:fluents (AtPose AtGrasp)` is also called with the current values of those fluents, as the keyword
argument `fluents` (a list of facts), for example to plan motions around the objects' current poses. Only the
focused algorithms support fluent streams.

A *wild* stream returns `WildOutput(values, facts)` to certify facts beyond its declared ones.

## Debugging

- `stream_map=DEBUG` (or `SHARED_DEBUG`) replaces every procedure with placeholder outputs, which tests the domain
  and stream declarations before the samplers exist.
- `verbose=True` (the default) prints every stream call, and `debug=True` prints FastDownward's output and keeps
  its files in `./temp/`.
- `result.status` says whether a plan was found, a limit was reached, or the solver concluded that no plan exists.
- The `examples/` directory covers each feature, from `examples/blocksworld` (no streams) to the PyBullet robots.
