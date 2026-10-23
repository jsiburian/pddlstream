"""Seeded solves with deterministic=True must repeat exactly, including a second solve in the same process."""

from pddlstream.algorithms.algorithm import solve_scope
from pddlstream.algorithms.meta import solve
from pddlstream.core.config import INF, SolverConfig
from pddlstream.core.context import WallClock, WorkClock, get_clock, get_config
from pddlstream.core.objects import Object
from pddlstream.core.utils import CreationHash
from pddlstream.fastdownward.translator import pddl


def test_creation_hash():
    first, second = CreationHash(), CreationHash()
    assert hash(second) == hash(first) + 1
    assert (first != second) and (first == first)


def test_solve_scope():
    assert isinstance(get_clock(), WallClock)
    with solve_scope(SolverConfig(deterministic=True)):
        clock = get_clock()
        assert isinstance(clock, WorkClock)
        with solve_scope(SolverConfig(deterministic=False, max_time=5)):  # Nested solves keep the outer clock
            assert (get_clock() is clock) and (get_config().max_time == 5)
        assert get_config().max_time == INF
        Object.from_value('value')
        assert Object.has_value('value')
    assert isinstance(get_clock(), WallClock)
    assert not Object.has_value('value')  # Registries are cleared after the outermost solve


def test_fastdownward_hashes():

    atom = pddl.Atom('on', ['a', 'b'])
    assert hash(atom) == hash(('Atom', 'on', ('a', 'b')))  # Not the class's memory address
    assert hash(pddl.Conjunction([atom])) == hash(('Conjunction', (atom,)))


def solve_continuous_tamp():
    from examples.continuous_tamp.primitives import tight
    from examples.continuous_tamp.run import get_stream_info, pddlstream_from_tamp, set_deterministic

    set_deterministic()
    problem = pddlstream_from_tamp(tight(n_blocks=2))
    plan, cost, _ = solve(
        problem, stream_info=get_stream_info(), planner='max-astar', unit_costs=True, max_time=120, deterministic=True
    )
    return [(action.name, str(action.args)) for action in plan], cost


def test_repeated_solves():
    assert solve_continuous_tamp() == solve_continuous_tamp()
