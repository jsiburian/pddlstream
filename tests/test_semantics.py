"""The current semantics of stream calls and solver verdicts, pinned before V2 changes them (docs/v2/adr-001).

Characterization tests describe today's behavior, including its quirks. The strict xfail tests are known bugs: each
expects the behavior ADR-001 requires and fails until the phase that fixes it (an xfail that starts passing fails
the suite, so the fix also removes the mark).
"""

import subprocess
import time

import pytest

from pddlstream.algorithms import focused
from pddlstream.algorithms.algorithm import parse_problem, solve_scope
from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import PDDLProblem, Status
from pddlstream.fastdownward import planner
from pddlstream.language.function import Function
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.language.stream import StreamInfo

DOMAIN_PDDL = """(define (domain semantics) (:requirements :strips)
  (:predicates (Value ?x) (Good ?x) (Other ?y) (Done))
  (:action finish :parameters (?x) :precondition (and (Value ?x) (Good ?x)) :effect (Done)))"""

STREAM_PDDL = """(define (stream semantics)
  (:stream sample :outputs (?x) :certified (Value ?x))
  (:stream test :inputs (?x) :domain (Value ?x) :certified (Good ?x)))"""


def make_problem(sample, test=from_test(lambda x: True), domain_pddl=DOMAIN_PDDL, stream_pddl=STREAM_PDDL, **streams):
    stream_map = {'sample': sample, 'test': test, **streams}
    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, [], ('Done',))


def get_stream(externals, name):
    [external] = [external for external in externals if external.name == name]
    return external


##################################################

# Characterization


@pytest.mark.parametrize('algorithm', ['incremental', 'adaptive'])
def test_no_sample_then_value(algorithm):
    """A sampler may produce nothing for several calls (NO_SAMPLE_YET) and then a value."""
    result = solve(make_problem(from_gen_fn(lambda: iter([None, None, (1,)]))), algorithm=algorithm, verbose=False)
    assert result.status is Status.SUCCEEDED
    assert [action.args for action in result.plan] == [(1,)]


def test_duplicate_outputs_are_not_new():
    """A repeated output tuple is not a new result."""
    problem = make_problem(from_gen_fn(lambda: iter([(1,), (1,), (2,)])))
    with solve_scope():
        _, _, _, externals = parse_problem(problem)
        instance = get_stream(externals, 'sample').get_instance(())
        num_results = [len(instance.next_results()[0]) for _ in range(3)]
        assert num_results == [1, 0, 1]


def test_stream_exception_propagates():
    """By default (stream_errors='raise'), an exception raised by a stream procedure stops the solve, naming the
    call."""

    def sample():
        raise ZeroDivisionError()
        yield

    with pytest.raises(ZeroDivisionError) as info:
        solve(make_problem(from_gen_fn(sample)), algorithm='adaptive', verbose=False)
    assert info.value.__notes__ == ['Raised by sample()']


def test_function_exception_leaves_its_instance_uncalled():
    """A function that raises leaves its instance as it was."""
    stream_pddl = STREAM_PDDL[:-1] + '\n  (:function (Cost ?x) (Value ?x)))'
    problem = make_problem(from_fn(lambda: (1,)), stream_pddl=stream_pddl, Cost=lambda x: 1 / 0)
    with solve_scope():
        _, _, _, externals = parse_problem(problem)
        [function] = [external for external in externals if isinstance(external, Function)]
        [result], _ = get_stream(externals, 'sample').get_instance(()).next_results()
        instance = function.get_instance(result.output_objects)
        with pytest.raises(ZeroDivisionError):
            instance.next_results()
        assert not instance.enumerated and (instance.num_calls == 0) and not instance.history


def test_replay_makes_a_spurious_call():
    """After reset(), an instance replays its history, then calls its exhausted generator once more."""
    problem = make_problem(from_gen_fn(lambda: iter([(1,)])))
    with solve_scope():
        _, _, _, externals = parse_problem(problem)
        instance = get_stream(externals, 'sample').get_instance(())
        while not instance.enumerated:
            instance.next_results()
        assert len(instance.history) == 2  # [(1,)], then [] when the generator stopped
        instance.reset()
        while not instance.enumerated:
            instance.next_results()
        assert len(instance.history) == 3  # The replayed [] did not mark it enumerated


def test_enable_keeps_streams_disabled():
    """StreamInstance.enable removes the blocked fact but leaves the instance disabled. The original code disabled
    re-enabling on purpose ('strange infinite loop bug'); adaptive relies on it."""
    problem = make_problem(from_fn(lambda: (1,)))
    with solve_scope():
        evaluations, _, domain, externals = parse_problem(problem)
        instance = get_stream(externals, 'sample').get_instance(())
        instance.disable(evaluations, domain)
        instance.enable(evaluations, domain)
        assert instance.disabled


##################################################

# Known bugs: verdicts that ADR-001 forbids or options that are ignored

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


def bug(reason):
    return pytest.mark.xfail(reason=reason, strict=True)


@bug('Phase 10: reaching max_skeletons is reported as infeasible (focused.py:139)')
def test_max_skeletons_is_not_infeasible():
    # The cheaper skeleton's sample fails its test; the other skeleton would succeed but exceeds max_skeletons
    problem = make_problem(
        from_fn(lambda: (1,)),
        test=from_test(lambda x: False),
        domain_pddl=TWO_WAYS_PDDL,
        stream_pddl=TWO_WAYS_STREAMS,
        **{'sample-other': from_fn(lambda: (2,))},
    )
    result = solve(problem, algorithm='adaptive', max_skeletons=1, verbose=False)
    assert result.status is not Status.INFEASIBLE


@bug('Phase 10: running out of time while sampling a skeleton is reported as infeasible (skeleton.py:390-413)')
def test_timeout_is_not_infeasible():
    def sample():  # Never produces a value
        while True:
            time.sleep(0.01)
            yield None

    problem = make_problem(from_gen_fn(sample))
    result = solve(problem, algorithm='adaptive', max_skeletons=1, max_time=0.5, verbose=False)
    assert result.stats['timeout']
    assert result.status is not Status.INFEASIBLE


@bug('Phase 10: a FastDownward search that times out is treated as "no plan"')
def test_planner_timeout_is_not_infeasible(monkeypatch):
    def timed_out(command, **kwargs):
        return subprocess.CompletedProcess(command, 23, stdout=b'')  # FastDownward's search timeout code

    monkeypatch.setattr(planner.subprocess, 'run', timed_out)
    problem = PDDLProblem(DOMAIN_PDDL, {}, None, {}, [('Value', 1), ('Good', 1)], ('Done',))
    result = solve(problem, algorithm='incremental', verbose=False)
    assert result.status is not Status.INFEASIBLE


@bug('Phase 10: streams pruned by max_effort are ignored when concluding infeasibility (refinement.py:40-51)')
def test_max_effort_is_not_infeasible():
    problem = make_problem(from_fn(lambda: (1,)))
    stream_info = {'sample': StreamInfo(effort=10)}
    result = solve(problem, algorithm='adaptive', stream_info=stream_info, max_effort=5, verbose=False)
    assert result.status is not Status.INFEASIBLE


@bug('Phase 10: focused gives sampled facts infinite complexity, so chained from_fn streams are never bound')
def test_focused_chain_is_not_infeasible():
    stream_map = {'sample': from_fn(lambda: (1,)), 'extend': from_fn(lambda y: (y + 1,))}
    problem = PDDLProblem(DOMAIN_PDDL, {}, CHAIN_STREAMS, stream_map, [], ('Done',))
    result = solve(problem, algorithm='focused', verbose=False)
    assert result.status is not Status.INFEASIBLE


@bug('Phase 10: max_failures is passed to focused and binding as 0 or infinity (meta.py:136)')
def test_max_failures_is_honored(monkeypatch):
    max_failures = []
    process_stream_plan = focused.process_stream_plan

    def record(*args, **kwargs):
        max_failures.append(kwargs['max_failures'])
        return process_stream_plan(*args, **kwargs)

    monkeypatch.setattr(focused, 'process_stream_plan', record)
    solve(make_problem(from_fn(lambda: (1,))), algorithm='binding', max_failures=3, verbose=False)
    assert max_failures and set(max_failures) == {3}
