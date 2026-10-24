"""Each solver run records the outcome of every stream call and why its search may have been incomplete; a solve
is reported infeasible only if nothing was cut short (docs/v2/adr-001-stream-outcomes.md)."""

import subprocess
import time

import pytest

from pddlstream.algorithms import focused
from pddlstream.algorithms.meta import solve
from pddlstream.core.config import SolverConfig
from pddlstream.core.problem import PDDLProblem, Status
from pddlstream.fastdownward import planner
from pddlstream.language.generator import from_fn, from_gen_fn, from_list_gen_fn, from_test
from pddlstream.language.stream import StreamInfo, WildOutput
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


##################################################

# Gaps: each makes a solve that ran out of things to try FAILED rather than INFEASIBLE


def assert_gap(result, gap):
    assert (result.plan is None) and (result.status is Status.FAILED)
    assert gap in result.stats['gaps'], result.stats['gaps']


TWO_WAYS_PDDL = """(define (domain two-ways) (:requirements :strips :action-costs)
  (:predicates (Value ?x) (Good ?x) (Other ?y) (Done))
  (:functions (total-cost))
  (:action finish :parameters (?x) :precondition (and (Value ?x) (Good ?x))
           :effect (and (Done) (increase (total-cost) 1)))
  (:action finish-other :parameters (?y) :precondition (Other ?y)
           :effect (and (Done) (increase (total-cost) 5))))"""

TWO_WAYS_STREAMS = """(define (stream two-ways)
  (:stream sample :outputs (?x) :certified (Value ?x))
  (:stream test :inputs (?x) :domain (Value ?x) :certified (Good ?x))
  (:stream sample-other :outputs (?y) :certified (Other ?y)))"""

CHAIN_STREAMS = """(define (stream chain)
  (:stream sample :outputs (?y) :certified (Other ?y))
  (:stream extend :inputs (?y) :domain (Other ?y) :outputs (?x) :certified (and (Value ?x) (Good ?x))))"""


def test_max_skeletons_gap():
    # The cheaper skeleton's sample fails its test; the other skeleton would succeed but exceeds max_skeletons
    problem = make_problem(
        from_fn(lambda: (1,)),
        test=from_test(lambda x: False),
        domain_pddl=TWO_WAYS_PDDL,
        stream_pddl=TWO_WAYS_STREAMS,
        **{'sample-other': from_fn(lambda: (2,))},
    )
    result = solve(problem, algorithm='adaptive', max_skeletons=1, verbose=False)
    assert_gap(result, 'max_skeletons')


def test_timeout_gap():
    def sample():  # Never produces a value
        while True:
            time.sleep(0.01)
            yield None

    problem = make_problem(from_gen_fn(sample))
    result = solve(problem, algorithm='adaptive', max_skeletons=1, max_time=0.5, verbose=False)
    assert_gap(result, 'max_time')
    assert result.stats['termination'] == 'max_time'


def test_planner_timeout_gap(monkeypatch):
    def timed_out(command, **kwargs):
        return subprocess.CompletedProcess(command, 23, stdout=b'')  # FastDownward's search timeout code

    monkeypatch.setattr(planner.subprocess, 'run', timed_out)
    problem = PDDLProblem(DOMAIN_PDDL, {}, None, {}, [('Value', 1), ('Good', 1)], ('Done',))
    result = solve(problem, algorithm='incremental', verbose=False)
    assert_gap(result, 'planner_timeout')


def test_max_effort_gap():
    problem = make_problem(from_fn(lambda: (1,)))
    stream_info = {'sample': StreamInfo(effort=10)}
    result = solve(problem, algorithm='adaptive', stream_info=stream_info, max_effort=5, verbose=False)
    assert_gap(result, 'max_effort')


def test_unreached_facts_gap():
    # Focused gives sampled facts infinite complexity, so it never binds the chain (FOCUSED_TAMP_BUG)
    stream_map = {'sample': from_fn(lambda: (1,)), 'extend': from_fn(lambda y: (y + 1,))}
    problem = PDDLProblem(DOMAIN_PDDL, {}, CHAIN_STREAMS, stream_map, [], ('Done',))
    result = solve(problem, algorithm='focused', verbose=False)
    assert_gap(result, 'unreached_facts')


def test_infeasible_without_gaps():
    problem = make_problem(from_fn(lambda: (1,)), test=from_test(lambda x: False))
    result = solve(problem, algorithm='adaptive', verbose=False)
    assert result.status is Status.INFEASIBLE
    assert (result.stats['termination'] == 'exhausted') and (result.stats['gaps'] == [])


def test_strict_needs_complete_streams():
    problem = make_problem(from_fn(lambda: (1,)), test=from_test(lambda x: False))
    result = solve(problem, algorithm='adaptive', strict=True, verbose=False)
    assert_gap(result, 'incomplete_streams')
    complete = {name: StreamInfo(complete=True) for name in ['sample', 'test']}
    result = solve(problem, algorithm='adaptive', stream_info=complete, strict=True, verbose=False)
    assert result.status is Status.INFEASIBLE


##################################################

# Options


def test_max_failures_is_honored(monkeypatch):
    max_failures = []
    process_stream_plan = focused.process_stream_plan

    def record(*args, **kwargs):
        max_failures.append(kwargs['max_failures'])
        return process_stream_plan(*args, **kwargs)

    monkeypatch.setattr(focused, 'process_stream_plan', record)
    solve(make_problem(from_fn(lambda: (1,))), algorithm='binding', max_failures=3, verbose=False)
    assert max_failures and set(max_failures) == {3}


def test_wild_output_enumerated():
    """A wild stream can declare that it has no further outputs."""
    problem = make_problem(from_list_gen_fn(lambda: iter([WildOutput(enumerated=True), [(1,)]])))
    result = solve(problem, algorithm='adaptive', verbose=False)
    assert result.status is Status.INFEASIBLE
