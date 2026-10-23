"""solve() options: SolverConfig, keyword overrides and up-front checks."""

import dataclasses

import pytest

from pddlstream.algorithms import meta
from pddlstream.algorithms.focused import solve_adaptive
from pddlstream.core.config import SolverConfig


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
    assert (kwargs['max_time'], kwargs['planner'], kwargs['fail_fast']) == (10, 'ff-astar', True)


def test_config_is_frozen():
    config = SolverConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_time = 1
    assert dataclasses.replace(config, max_time=1).max_time == 1
