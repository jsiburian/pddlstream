"""The built-in algorithms are compositions of strategies, and an outside plugin can compose its own through
pddlstream.strategy without importing anything else from pddlstream."""

import ast
import itertools
import pathlib

import pytest

from examples.blocksworld.run import get_problem as get_blocksworld
from examples.continuous_tamp.primitives import blocked
from examples.continuous_tamp.run import pddlstream_from_tamp, set_deterministic
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.validate import verify_certificate
from pddlstream.core.problem import Status
from pddlstream.strategy import BindingPolicy, CapabilityError, OptimisticSearch, SkeletonRefinement, solve_with
from tests.plugins.round_robin import RoundRobinPolicy, solve_round_robin

PLUGIN = pathlib.Path(__file__).parent / 'plugins' / 'round_robin.py'
OPTIONS = dict(unit_costs=True, deterministic=True, verbose=False, max_time=60)


def get_continuous():
    set_deterministic()
    return pddlstream_from_tamp(blocked(n_blocks=2))


def get_calls(result):
    return [event for event in result.trace if event['event'] == 'Call']


def test_plugin_imports_only_the_facade():
    modules = set()
    for node in ast.walk(ast.parse(PLUGIN.read_text())):
        if isinstance(node, ast.ImportFrom):
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    assert {module for module in modules if module.startswith('pddlstream')} == {'pddlstream.strategy'}


@pytest.mark.parametrize('get_problem', [get_blocksworld, get_continuous])
def test_round_robin(get_problem):
    problem = get_problem()
    strategies = []
    result = solve_round_robin(problem, strategies=strategies, **OPTIONS)
    assert result.status is Status.SUCCEEDED
    assert verify_certificate(problem, result, unit_costs=True) == (True, [])
    [(search, _)] = strategies
    assert sum(search.proposals.values()) == result.stats['iterations']


def test_round_robin_policy():
    problem = get_continuous()
    result = solve_round_robin(problem, binding_policy=RoundRobinPolicy(), **OPTIONS)
    assert result.status is Status.SUCCEEDED
    assert verify_certificate(problem, result, unit_costs=True) == (True, [])


def test_adaptive_is_a_composition():
    def compose(state):
        return OptimisticSearch.from_state(state), SkeletonRefinement.from_state(state)

    expected = solve(get_continuous(), algorithm='adaptive', trace=True, **OPTIONS)
    result = solve_with(get_continuous(), compose, trace=True, **OPTIONS)
    assert (str(result.plan), result.cost) == (str(expected.plan), expected.cost)
    assert get_calls(result) == get_calls(expected)


def test_declared_capabilities():
    class NoStreams(SkeletonRefinement):
        unsupported = {'streams': 'This refinement cannot call streams: {}'}

    def compose(state):
        return OptimisticSearch.from_state(state), NoStreams.from_state(state)

    with pytest.raises(CapabilityError, match='cannot call streams: s-region, s-grasp'):
        solve_with(get_continuous(), compose, **OPTIONS)
    assert solve_with(get_blocksworld(), compose, **OPTIONS).status is Status.SUCCEEDED


def test_policy_need_not_compute_complexity():
    class Fifo(BindingPolicy):
        def __init__(self):
            super().__init__()
            self.counter = itertools.count()

        def priority(self, binding):
            return next(self.counter)

    # The queue computes a binding's complexity, and its ancestors', when it needs them
    result = solve(get_continuous(), algorithm='adaptive', binding_policy=Fifo(), **dict(OPTIONS, max_time=3))
    assert result.status in [Status.SUCCEEDED, Status.FAILED]
