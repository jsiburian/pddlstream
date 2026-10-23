"""Solvers return a SolveResult that still unpacks as (plan, cost, certificate)."""

import pytest

from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import PDDLProblem, Solution, SolveResult, Status

DOMAIN_PDDL = """(define (domain line) (:requirements :strips)
  (:predicates (At ?l) (Adj ?a ?b))
  (:action move :parameters (?a ?b) :precondition (and (At ?a) (Adj ?a ?b))
                :effect (and (At ?b) (not (At ?a)) (increase (total-cost) 1))))"""


def line_problem(goal):
    init = [('At', 'l0'), ('Adj', 'l0', 'l1'), ('Adj', 'l1', 'l2')]
    return PDDLProblem(DOMAIN_PDDL, {}, None, {}, init, ('At', goal))


@pytest.mark.parametrize('algorithm', ['incremental', 'adaptive'])
def test_solved(algorithm):
    result = solve(line_problem('l2'), algorithm=algorithm, verbose=False)
    assert isinstance(result, SolveResult) and isinstance(result, Solution)
    plan, cost, certificate = result
    assert [action.name for action in plan] == ['move', 'move'] and cost == 2
    assert result == (plan, cost, certificate)
    assert result.status is Status.SUCCEEDED
    assert result.stats['solved'] and (result.stats['iterations'] >= 1) and ('run_time' in result.stats)


@pytest.mark.parametrize('algorithm', ['incremental', 'adaptive'])
def test_infeasible(algorithm):
    result = solve(line_problem('l3'), algorithm=algorithm, verbose=False)
    assert result.plan is None
    assert result.status is Status.INFEASIBLE


def test_failed():
    result = solve(line_problem('l2'), algorithm='adaptive', max_iterations=0, verbose=False)
    assert (result.plan is None) and (result.status is Status.FAILED)
