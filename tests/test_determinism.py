"""Seeded solves with deterministic=True must repeat exactly, including a second solve in the same process."""

from pddlstream.algorithms.algorithm import solve_scope
from pddlstream.algorithms.meta import solve
from pddlstream.language.object import Object
from pddlstream.utils import CreationHash, WallClock, WorkClock, get_clock


def test_creation_hash():
    first, second = CreationHash(), CreationHash()
    assert hash(second) == hash(first) + 1
    assert (first != second) and (first == first)


def test_solve_scope():
    assert isinstance(get_clock(), WallClock)
    with solve_scope(deterministic=True):
        clock = get_clock()
        assert isinstance(clock, WorkClock)
        with solve_scope(deterministic=False):  # Nested solves share the outer scope
            assert get_clock() is clock
        Object.from_value('value')
        assert Object.has_value('value')
    assert isinstance(get_clock(), WallClock)
    assert not Object.has_value('value')  # Registries are cleared after the outermost solve


def test_fastdownward_hashes():
    import pddl

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
