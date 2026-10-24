"""Each solver run records the outcome of every stream call and why its search may have been incomplete
(docs/v2/adr-001-stream-outcomes.md)."""

import pytest

from pddlstream.algorithms.meta import solve
from pddlstream.core.config import SolverConfig
from pddlstream.core.problem import PDDLProblem, Status
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from test_semantics import DOMAIN_PDDL, STREAM_PDDL, make_problem


def test_outcome_counts():
    sample = from_gen_fn(lambda: iter([None, None, (1,)]))  # Two calls without a sample, then a value
    result = solve(make_problem(sample), algorithm='incremental', verbose=False)
    assert result.status is Status.SUCCEEDED
    # sample yields on its third call, and the test of its value (a single call) yields too
    assert result.stats['outcomes'] == {'no_sample_yet': 2, 'yielded': 2}


def raise_error(*args):
    raise ZeroDivisionError()


def test_stream_error_fails_the_call():
    """With stream_errors='fail', a procedure that raises is treated as exhausted and the solve goes on."""
    problem = make_problem(from_fn(raise_error), extra=from_fn(lambda: (1,)))
    result = solve(problem, algorithm='adaptive', stream_errors='fail', verbose=False)
    assert result.plan is None
    assert result.stats['outcomes'] == {'error': 1}


def test_predicate_error_fails_the_call():
    stream_pddl = STREAM_PDDL[:-1] + '\n  (:predicate (Fine ?x) (Value ?x)))'
    domain_pddl = DOMAIN_PDDL.replace('(Good ?x))', '(Good ?x) (Fine ?x))').replace('(Done))', '(Done) (Fine ?x))', 1)
    stream_map = {'sample': from_fn(lambda: (1,)), 'test': from_test(lambda x: True), 'Fine': raise_error}
    problem = PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, [('Value', 0)], ('Done',))
    result = solve(problem, algorithm='adaptive', stream_errors='fail', verbose=False)  # Predicates are eager
    assert result.plan is None
    assert result.stats['outcomes']['error'] >= 1


def test_cost_function_error_stops_the_solve():
    """The planner needs a cost function's value, so its errors always stop the solve."""
    stream_pddl = STREAM_PDDL[:-1] + '\n  (:function (Cost ?x) (Value ?x)))'
    domain_pddl = """(define (domain costs) (:requirements :strips :action-costs)
  (:predicates (Value ?x) (Good ?x) (Done))
  (:functions (total-cost) (Cost ?x))
  (:action finish :parameters (?x) :precondition (and (Value ?x) (Good ?x))
           :effect (and (Done) (increase (total-cost) (Cost ?x)))))"""
    problem = make_problem(from_fn(lambda: (1,)), domain_pddl=domain_pddl, stream_pddl=stream_pddl, Cost=raise_error)
    with pytest.raises(ZeroDivisionError):
        solve(problem, algorithm='adaptive', stream_errors='fail', verbose=False)


def test_unknown_policy():
    with pytest.raises(ValueError, match='stream_errors'):
        SolverConfig(stream_errors='ignore')
