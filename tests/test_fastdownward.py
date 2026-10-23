"""Tests of PDDLStream's calls into the FastDownward fork's translator API."""

import os
import subprocess
import sys

from pddlstream.algorithms.scheduling.reinstantiate import instantiate_unsatisfiable
from pddlstream.core.conversion import obj_from_value_expression
from pddlstream.core.evaluations import evaluations_from_init
from pddlstream.fastdownward.grounding import instantiate_task
from pddlstream.fastdownward.tasks import get_problem, parse_sequential_domain, task_from_domain_problem
from pddlstream.fastdownward.translator import pddl
from pddlstream.language.optimizer import UNSATISFIABLE


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
    assert all(action.cost == 1000 for action in instantiated.actions)  # Unit cost times the 1e3 cost scale


def test_instantiate_unsatisfiable():
    # Exercises Effect.instantiate(var_mapping, init_facts, fluent_facts, objects_by_type, result, predicate_to_atoms)
    blocked = '~blocked-negative'
    effect = pddl.Effect(
        [pddl.TypedObject('?y', 'object')],
        pddl.Conjunction([pddl.Atom('p', ('?y',)), pddl.Atom(blocked, ('?x', '?y'))]),
        pddl.Atom(UNSATISFIABLE, ()),
    )
    action = pddl.Action('act', [pddl.TypedObject('?x', 'object')], 1, pddl.Truth(), [effect], None)
    state = {pddl.Atom('p', ('b',))}
    precondition = instantiate_unsatisfiable(state, action, {'?x': 'a'}, negative_from_name={blocked: None})
    assert precondition == [pddl.NegatedAtom(blocked, ('a', 'b'))]


def run_python(code, *args, **env):
    return subprocess.run(
        [sys.executable, '-c', code, *args], capture_output=True, text=True, env=dict(os.environ, **env)
    )


def test_import_needs_no_build():
    # Importing pddlstream neither loads the translator nor changes sys.path or sys.argv
    code = (
        'import sys; path, argv = list(sys.path), list(sys.argv)\n'
        'import pddlstream.algorithms.meta, pddlstream.algorithms.focused, pddlstream.language.stream\n'
        "assert 'pddl' not in sys.modules and sys.path == path and sys.argv == argv"
    )
    result = run_python(code, PDDLSTREAM_FD_PATH=os.devnull)
    assert result.returncode == 0, result.stderr


def test_missing_build(tmp_path):
    result = run_python(
        "from pddlstream.algorithms.downward import parse_lisp; parse_lisp('(a)')", PDDLSTREAM_FD_PATH=str(tmp_path)
    )
    assert 'FastDownward is not compiled' in result.stderr and 'PDDLSTREAM_FD_PATH' in result.stderr


def test_translator_ignores_command_line():
    # The translator's options module parses sys.argv, which holds the calling script's own flags
    result = run_python("from pddlstream.algorithms.downward import parse_lisp; print(parse_lisp('(a b)'))", '-p', 'x')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['a', 'b']"
