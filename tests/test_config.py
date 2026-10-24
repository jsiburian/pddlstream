"""solve() options: SolverConfig, keyword overrides and up-front checks."""

import dataclasses

import pytest

from pddlstream.algorithms import meta
from pddlstream.algorithms.algorithm import solve_scope
from pddlstream.algorithms.focused import solve_adaptive
from pddlstream.core.config import SolverConfig
from pddlstream.core.context import get_config
from pddlstream.core.conversion import obj_from_value_expression
from pddlstream.core.evaluations import evaluations_from_init
from pddlstream.fastdownward.tasks import get_problem, parse_sequential_domain, task_from_domain_problem


@pytest.fixture
def dispatched(monkeypatch):
    """Records the solver and keyword arguments that solve() dispatches to, instead of solving."""
    calls = []
    for name in ['solve_incremental', 'solve_adaptive', 'solve_focused_original', 'solve_binding']:
        monkeypatch.setattr(meta, name, lambda problem, _name=name, **kwargs: calls.append((_name, kwargs)))
    return calls


def test_unknown_options_fail_before_solving():
    # problem=None would fail as soon as solving started
    with pytest.raises(TypeError, match="did you mean 'planner'"):
        meta.solve(None, plannr='ff-astar')
    with pytest.raises(TypeError, match="did you mean 'max_planner_time'"):
        solve_adaptive(None, max_planer_time=10)
    with pytest.raises(ValueError, match='Unknown algorithm'):
        meta.solve(None, algorithm='bogus')


def test_keyword_options(dispatched):
    meta.solve(None, algorithm='incremental', max_time=5, planner='ff-astar', debug=True)
    [(solver, kwargs)] = dispatched
    assert solver == 'solve_incremental'
    assert (kwargs['max_time'], kwargs['planner'], kwargs['debug']) == (5, 'ff-astar', True)
    assert 'max_planner_time' not in kwargs  # Search options keep the planner's defaults unless set


def test_config_with_overrides(dispatched):
    config = SolverConfig(algorithm='focused', max_time=5, max_failures=1, planner='ff-astar')
    meta.solve(None, config=config, max_time=10)
    [(solver, kwargs)] = dispatched
    assert solver == 'solve_focused_original'
    assert (kwargs['max_time'], kwargs['planner'], kwargs['max_failures']) == (10, 'ff-astar', 1)


def test_config_is_frozen():
    config = SolverConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_time = 1
    assert dataclasses.replace(config, max_time=1).max_time == 1


def test_legacy_module_flags(monkeypatch):
    from pddlstream.language import statistics

    monkeypatch.setattr(statistics, 'LOAD_STATISTICS', False)
    with pytest.warns(FutureWarning, match='pass load_statistics=False'):
        assert get_config().load_statistics is False
    with pytest.warns(FutureWarning), solve_scope(SolverConfig(save_statistics=False)):
        assert (get_config().load_statistics, get_config().save_statistics) == (False, False)


def test_fd_instantiate(monkeypatch):
    # Assigning instantiate_task.FD_INSTANTIATE used to have no effect, since it was bound as a default argument
    from pddlstream.algorithms import instantiate_task as legacy_module
    from pddlstream.fastdownward import grounding

    calls = []
    instantiate_domain = grounding.instantiate_domain
    monkeypatch.setattr(
        grounding, 'instantiate_domain', lambda task, **kwargs: calls.append(task) or instantiate_domain(task)
    )

    def make_task():  # Instantiation normalizes the task in place
        domain = parse_sequential_domain("""(define (domain t) (:requirements :strips) (:predicates (Adj ?a ?b) (At ?l))
          (:action move :parameters (?a ?b) :precondition (and (Adj ?a ?b) (At ?a))
                        :effect (and (At ?b) (not (At ?a)))))""")
        evaluations = evaluations_from_init([('Adj', 'l1', 'l2'), ('At', 'l1')])
        goal = obj_from_value_expression(('At', 'l2'))
        return task_from_domain_problem(domain, get_problem(evaluations, goal, domain))

    with solve_scope(SolverConfig(fd_instantiate=False)):
        grounding.instantiate_task(make_task())
    monkeypatch.setattr(legacy_module, 'FD_INSTANTIATE', False)
    with pytest.warns(FutureWarning, match='pass fd_instantiate=False'):
        grounding.instantiate_task(make_task())
    assert len(calls) == 2
