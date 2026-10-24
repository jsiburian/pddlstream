"""Typed domains: PDDL types compile into unary predicates, with facts for the closure of the type hierarchy."""

import pytest

from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import PDDLProblem, Status
from pddlstream.core.syntax import And, Exists
from pddlstream.language.generator import from_fn

TYPED_DOMAIN = """(define (domain typed) (:requirements :strips :typing)
  (:types item - object block - item)
  (:constants table - item)
  (:predicates (On ?b - block ?i - item) (Clear ?i - item) (Holding ?b - block) (HandEmpty))
  (:action pick :parameters (?b - block ?i - item)
    :precondition (and (On ?b ?i) (Clear ?b) (HandEmpty))
    :effect (and (Holding ?b) (Clear ?i) (not (On ?b ?i)) (not (HandEmpty))))
  (:action place :parameters (?b - block ?i - item)
    :precondition (and (Holding ?b) (Clear ?i) (exists (?o - block) (On ?o table)))
    :effect (and (On ?b ?i) (HandEmpty) (not (Holding ?b)) (not (Clear ?i)))))"""


def typed_problem(goal, extra=()):
    # b is only declared a block; that it is an item follows from the type hierarchy
    init = [('Block', 'a'), ('Block', 'b'), ('HandEmpty',), *extra]
    init += [('On', 'a', 'table'), ('On', 'b', 'table'), ('Clear', 'a'), ('Clear', 'b'), ('Clear', 'table')]
    return PDDLProblem(TYPED_DOMAIN, {'table': 'table'}, None, {}, init, goal)


@pytest.mark.parametrize('algorithm', ['incremental', 'adaptive'])
def test_typed_domain(algorithm):
    result = solve(typed_problem(('On', 'a', 'b')), algorithm=algorithm, unit_costs=True, verbose=False)
    assert [(action.name, action.args) for action in result.plan] == [('pick', ('a', 'table')), ('place', ('a', 'b'))]


def test_untyped_object_cannot_act():
    extra = [('On', 'c', 'table'), ('Clear', 'c')]  # c has no type, so it is not a block
    result = solve(typed_problem(('Holding', 'c'), extra), algorithm='incremental', unit_costs=True, verbose=False)
    assert result.status is Status.INFEASIBLE


def test_typed_streams_certify_supertypes():
    streams = """(define (stream spawn)
  (:stream make-block :outputs (?b) :certified (and (Block ?b) (Clear ?b) (On ?b table))))"""
    init = [('HandEmpty',), ('Clear', 'table'), ('Block', 'a'), ('On', 'a', 'table'), ('Clear', 'a')]
    goal = Exists(['?b'], And(('Block', '?b'), ('On', 'a', '?b')))  # Placing on ?b needs it to be an item
    problem = PDDLProblem(
        TYPED_DOMAIN, {'table': 'table'}, streams, {'make-block': from_fn(lambda: ('b',))}, init, goal
    )
    result = solve(problem, algorithm='adaptive', unit_costs=True, verbose=False)
    assert [(action.name, action.args) for action in result.plan] == [('pick', ('a', 'table')), ('place', ('a', 'b'))]
