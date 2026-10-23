"""FastDownward runs in a private directory per search, so solves never share files."""

import os
import pathlib
import subprocess
import sys

from pddlstream.algorithms.search import solve_from_pddl

DOMAIN_PDDL = """(define (domain line) (:requirements :strips)
  (:predicates (At ?l) (Adj ?a ?b))
  (:action move :parameters (?a ?b) :precondition (and (At ?a) (Adj ?a ?b)) :effect (and (At ?b) (not (At ?a)))))"""


def line_problem(length):
    """A problem whose only plan moves along a line of locations in `length` steps."""
    objects = ' '.join(f'l{i}' for i in range(length + 1))
    adjacent = ' '.join(f'(Adj l{i} l{i + 1})' for i in range(length))
    return f'(define (problem p) (:domain line) (:objects {objects}) (:init (At l0) {adjacent}) (:goal (At l{length})))'


def test_private_directory():
    plan, _ = solve_from_pddl(DOMAIN_PDDL, line_problem(3))
    assert len(plan) == 3
    assert os.listdir() == []  # Nothing written to the working directory


def test_debug_directory():
    plan, _ = solve_from_pddl(DOMAIN_PDDL, line_problem(3), debug=True)
    assert len(plan) == 3
    assert {'domain.pddl', 'problem.pddl', 'output.sas', 'sas_plan'} <= set(os.listdir('temp'))


SOLVE_LOOP = """
import sys
from tests.test_planner import DOMAIN_PDDL, line_problem
from pddlstream.algorithms.search import solve_from_pddl
length = int(sys.argv[1])
for _ in range(10):
    plan, _ = solve_from_pddl(DOMAIN_PDDL, line_problem(length))
    assert len(plan) == length, plan
"""


def test_concurrent_solves(tmp_path):
    # Two processes in one working directory would read each other's plans if they shared files
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    processes = [
        subprocess.Popen([sys.executable, '-c', SOLVE_LOOP, str(length)], cwd=tmp_path, env=env, stderr=subprocess.PIPE)
        for length in [3, 5]
    ]
    for process in processes:
        _, error = process.communicate(timeout=120)
        assert process.returncode == 0, error.decode()
