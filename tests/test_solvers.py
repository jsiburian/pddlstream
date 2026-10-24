"""In-process tests of solver entry points and helpers."""

import pathlib

import pytest

from examples.blocksworld.run import get_problem as get_blocksworld
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.focused import solve_hierarchical
from pddlstream.algorithms.incremental import solve_exhaustive, solve_immediate
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.serialized import solve_serialized
from pddlstream.core.problem import PDDLProblem, is_plan, print_solution
from pddlstream.core.syntax import And
from pddlstream.language.generator import compose_gen_fns, from_fn, gen_fn_from_wild_gen_fn

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('solver', [solve_serialized, solve_immediate, solve_exhaustive, solve_hierarchical])
def test_blocksworld_entry_points(solver):
    plan, cost, certificate = solver(get_blocksworld(), unit_costs=True)
    assert [action.name for action in plan] == ['unstack', 'putdown', 'pickup', 'stack']


def test_print_solution_counts_facts(capsys):
    solution = solve(get_blocksworld(), unit_costs=True)
    print_solution(solution)
    assert f'Evaluations: {len(solution.certificate.all_facts)}' in capsys.readouterr().out


def test_compose_gen_fns():
    composed = compose_gen_fns(from_fn(lambda x: (x + 1,)), from_fn(lambda x, y: (10 * y,)))
    assert [outputs for outputs in composed(1) if outputs is not None] == [(2, 20)]


def test_gen_fn_from_wild_gen_fn():
    wild_gen_fn = lambda x: iter([([(x,)], [('Fact', x)])])
    assert list(gen_fn_from_wild_gen_fn(wild_gen_fn)(5)) == [[(5,)]]


@pytest.mark.parametrize('use_skeleton', [False, True])
def test_continuous_tamp(use_skeleton):
    from examples.continuous_tamp.primitives import tight
    from examples.continuous_tamp.run import TIGHT_SKELETON, pddlstream_from_tamp, set_deterministic

    set_deterministic()
    constraints = PlanConstraints(skeletons=[TIGHT_SKELETON] if use_skeleton else None, exact=True)
    plan, cost, _ = solve(
        pddlstream_from_tamp(tight(n_blocks=2)),
        constraints=constraints,
        planner='max-astar',
        unit_costs=True,
        max_time=120,
    )
    assert is_plan(plan)
    if use_skeleton:
        assert [action.name for action in plan] == [name for name, _ in TIGHT_SKELETON]


@pytest.mark.skipif(
    not (ROOT / 'examples/pybullet/utils/pybullet_tools/utils.py').exists(),
    reason='pybullet-planning submodule not checked out',
)
def test_numpy_math_shim():
    from examples.pybullet.utils.pybullet_tools.utils import get_yaw

    assert get_yaw((1, 1)) == pytest.approx(0.785398, abs=1e-6)


CONSTANT_DOMAIN = """(define (domain constant) (:requirements :strips) (:constants @t)
  (:predicates (Block ?b) (Table ?t) (Placement ?b ?p) (Placed ?b))
  (:action place :parameters (?b ?p) :precondition (Placement ?b ?p) :effect (Placed ?b)))"""

CONSTANT_STREAMS = """(define (stream constant)
  (:stream sample-placement :inputs (?b) :domain (and (Block ?b) (Table @t)) :outputs (?p)
    :certified (Placement ?b ?p)))"""


def test_serialized_with_constants():
    """solve_serialized reuses its parsed streams, whose domain facts refer to constants, across subgoals."""
    init = [('Block', 'a'), ('Block', 'b'), ('Table', 'table')]
    stream_map = {'sample-placement': from_fn(lambda b: (f'{b}-pose',))}
    goal = And(('Placed', 'a'), ('Placed', 'b'))
    problem = PDDLProblem(CONSTANT_DOMAIN, {'@t': 'table'}, CONSTANT_STREAMS, stream_map, init, goal)
    plan, _, _ = solve_serialized(problem, unit_costs=True, verbose=False)
    assert [action.args for action in plan] == [('a', 'a-pose'), ('b', 'b-pose')]
