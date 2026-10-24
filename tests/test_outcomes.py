"""Each solver run records the outcome of every stream call and why its search may have been incomplete
(docs/v2/adr-001-stream-outcomes.md)."""

from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import Status
from pddlstream.language.generator import from_gen_fn
from test_semantics import make_problem


def test_outcome_counts():
    sample = from_gen_fn(lambda: iter([None, None, (1,)]))  # Two calls without a sample, then a value
    result = solve(make_problem(sample), algorithm='incremental', verbose=False)
    assert result.status is Status.SUCCEEDED
    # sample yields on its third call, and the test of its value (a single call) yields too
    assert result.stats['outcomes'] == {'no_sample_yet': 2, 'yielded': 2}
