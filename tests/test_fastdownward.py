"""Tests of PDDLStream's calls into the FastDownward fork's translator API."""
from pddlstream.algorithms.common import evaluations_from_init
from pddlstream.algorithms.downward import parse_sequential_domain, get_problem, task_from_domain_problem
from pddlstream.algorithms.instantiate_task import instantiate_task
from pddlstream.algorithms.scheduling.reinstantiate import instantiate_unsatisfiable
from pddlstream.language.conversion import obj_from_value_expression
from pddlstream.language.optimizer import UNSATISFIABLE

import pddl


def test_instantiate_domain_costs(tmp_path, monkeypatch):
    # Exercises Action.instantiate(var_mapping, init_facts, init_assignments, fluent_facts, ...)
    monkeypatch.chdir(tmp_path)
    domain = parse_sequential_domain("""(define (domain t) (:requirements :strips)
      (:predicates (Adj ?a ?b) (At ?l))
      (:action move :parameters (?a ?b) :precondition (and (Adj ?a ?b) (At ?a))
         :effect (and (At ?b) (not (At ?a)) (increase (total-cost) 1))))""")
    evaluations = evaluations_from_init([('Adj', 'l1', 'l2'), ('Adj', 'l2', 'l3'), ('At', 'l1')])
    goal = obj_from_value_expression(('At', 'l3'))
    task = task_from_domain_problem(domain, get_problem(evaluations, goal, domain))
    instantiated = instantiate_task(task, use_fd=False)
    assert len(instantiated.actions) == 2
    assert all(action.cost == 1000 for action in instantiated.actions) # Unit cost times the 1e3 cost scale


def test_instantiate_unsatisfiable():
    # Exercises Effect.instantiate(var_mapping, init_facts, fluent_facts, objects_by_type, result, predicate_to_atoms)
    blocked = '~blocked-negative'
    effect = pddl.Effect([pddl.TypedObject('?y', 'object')],
                         pddl.Conjunction([pddl.Atom('p', ('?y',)), pddl.Atom(blocked, ('?x', '?y'))]),
                         pddl.Atom(UNSATISFIABLE, ()))
    action = pddl.Action('act', [pddl.TypedObject('?x', 'object')], 1, pddl.Truth(), [effect], None)
    state = {pddl.Atom('p', ('b',))}
    precondition = instantiate_unsatisfiable(state, action, {'?x': 'a'}, negative_from_name={blocked: None})
    assert precondition == [pddl.NegatedAtom(blocked, ('a', 'b'))]
