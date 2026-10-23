"""The public API that examples and downstream projects (e.g. pyrobosim) rely on must keep working."""

import importlib
import inspect
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PUBLIC_NAMES = {
    'pddlstream.algorithms.meta': ['solve', 'create_parser', 'solve_restart'],
    'pddlstream.algorithms.focused': ['solve_adaptive', 'solve_focused', 'solve_binding'],
    'pddlstream.algorithms.incremental': ['solve_incremental'],
    'pddlstream.algorithms.algorithm': ['reset_globals'],
    'pddlstream.algorithms.downward': [
        'set_cost_scale',
        'get_cost_scale',
        'MAX_FD_COST',
        'SEARCH_OPTIONS',
        'DEFAULT_PLANNER',
        'DEFAULT_MAX_TIME',
    ],
    'pddlstream.algorithms.instantiate_task': ['FD_INSTANTIATE'],
    'pddlstream.algorithms.constraints': ['PlanConstraints', 'WILD', 'OrderedSkeleton'],
    'pddlstream.core.config': ['SolverConfig'],
    'pddlstream.language.constants': [
        'PDDLProblem',
        'Solution',
        'SolveResult',
        'Status',
        'Certificate',
        'Action',
        'And',
        'Or',
        'Not',
        'Equal',
        'Exists',
        'ForAll',
        'Imply',
        'Minimize',
        'Output',
        'print_solution',
        'is_plan',
        'FAILED',
        'INFEASIBLE',
    ],
    'pddlstream.language.stream': ['StreamInfo', 'PartialInputs', 'WildOutput', 'DEBUG', 'SHARED_DEBUG'],
    'pddlstream.language.external': [
        'DEBUG',
        'SHARED_DEBUG',
        'defer_shared',
        'defer_unique',
        'never_defer',
        'get_defer_any_unbound',
        'get_defer_all_unbound',
    ],
    'pddlstream.language.function': ['FunctionInfo', 'PredicateInfo'],
    'pddlstream.language.generator': [
        'from_gen_fn',
        'from_list_fn',
        'from_fn',
        'from_test',
        'from_gen',
        'from_list_gen_fn',
        'from_sampler',
        'negate_test',
        'empty_gen',
        'fn_from_constant',
        'outputs_from_boolean',
        'universe_test',
    ],
    'pddlstream.language.optimizer': ['OptimizerOutput'],
    'pddlstream.language.statistics': ['LOAD_STATISTICS', 'SAVE_STATISTICS'],
    'pddlstream.utils': ['read', 'get_file_path', 'INF', 'read_pddl', 'Profiler'],
}

# Keyword arguments that examples and downstream projects pass to solve(); the last three reach the planner
SOLVE_KWARGS = [
    'algorithm',
    'constraints',
    'stream_info',
    'replan_actions',
    'unit_costs',
    'success_cost',
    'max_time',
    'max_iterations',
    'max_memory',
    'initial_complexity',
    'complexity_step',
    'max_complexity',
    'max_skeletons',
    'search_sample_ratio',
    'max_failures',
    'unit_efforts',
    'max_effort',
    'effort_weight',
    'reorder',
    'visualize',
    'verbose',
]

DOMAIN_PDDL = """(define (domain navigation)
  (:requirements :strips)
  (:predicates (Location ?l) (Door ?l ?d) (Path ?l1 ?l2 ?p) (Open ?d) (At ?l) (Visited ?l))
  (:action move
    :parameters (?l1 ?l2 ?d ?p)
    :precondition (and (At ?l1) (Door ?l2 ?d) (Open ?d) (Path ?l1 ?l2 ?p))
    :effect (and (At ?l2) (Visited ?l2) (not (At ?l1)) (increase (total-cost) 1))))
"""

STREAM_PDDL = """(define (stream navigation)
  (:stream s-door
    :inputs (?l)
    :domain (Location ?l)
    :outputs (?d)
    :certified (Door ?l ?d))
  (:stream t-open
    :inputs (?l ?d)
    :domain (Door ?l ?d)
    :certified (Open ?d))
  (:stream s-path
    :inputs (?l1 ?l2)
    :domain (and (Location ?l1) (Location ?l2))
    :outputs (?p)
    :certified (Path ?l1 ?l2 ?p)))
"""


@pytest.mark.parametrize('module_name', PUBLIC_NAMES)
def test_public_names(module_name):
    module = importlib.import_module(module_name)
    assert [name for name in PUBLIC_NAMES[module_name] if not hasattr(module, name)] == []


def test_debug_streams_are_shared():
    from pddlstream.language import external, stream

    assert (stream.DEBUG is external.DEBUG) and (stream.SHARED_DEBUG is external.SHARED_DEBUG)


def test_positional_optimistic_functions():
    from pddlstream.language.function import FunctionInfo, PredicateInfo
    from pddlstream.language.stream import StreamInfo

    assert next(iter(inspect.signature(StreamInfo).parameters)) == 'opt_gen_fn'
    assert next(iter(inspect.signature(FunctionInfo).parameters)) == 'opt_fn'
    assert next(iter(inspect.signature(PredicateInfo).parameters)) == 'opt_fn'


def test_solve_keyword_arguments():
    from pddlstream.algorithms.meta import solve

    parameters = inspect.signature(solve).parameters
    assert [name for name in SOLVE_KWARGS if name not in parameters] == []
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def test_fastdownward_beside_package():
    # pyrobosim finds the planner through the source layout pddlstream/../downward/
    import pddlstream

    downward = pathlib.Path(pddlstream.__file__).resolve().parents[1] / 'downward'
    assert (downward / 'builds').is_dir()


def test_downstream_usage(tmp_path):
    # Mirrors how pyrobosim builds and solves a problem
    from pddlstream.algorithms.focused import solve_adaptive
    from pddlstream.language.constants import And, PDDLProblem, Solution
    from pddlstream.language.generator import from_gen_fn, from_list_fn, from_test
    from pddlstream.language.stream import StreamInfo
    from pddlstream.utils import read

    (tmp_path / 'domain.pddl').write_text(DOMAIN_PDDL)
    (tmp_path / 'stream.pddl').write_text(STREAM_PDDL)
    stream_map = {
        's-door': from_list_fn(lambda l: [(l + '-door',)]),
        't-open': from_test(lambda l, d: d.startswith(l)),
        's-path': from_gen_fn(lambda l1, l2: iter([((l1, l2),)])),
    }
    init = [('Location', 'hall'), ('Location', 'kitchen'), ('At', 'hall')]
    problem = PDDLProblem(
        read(tmp_path / 'domain.pddl'),
        {},
        read(tmp_path / 'stream.pddl'),
        stream_map,
        init,
        And(('Visited', 'kitchen')),
    )
    solution = solve_adaptive(
        problem,
        stream_info={'t-open': StreamInfo(eager=True)},
        planner='ff-astar',
        max_planner_time=10,
        max_time=60,
        debug=False,
        verbose=False,
    )
    assert isinstance(solution, Solution)
    plan, cost, certificate = solution
    assert [(action.name, action.args) for action in plan] == [
        ('move', ('hall', 'kitchen', 'kitchen-door', ('hall', 'kitchen')))
    ]
    assert cost == 1
    assert ('path', 'hall', 'kitchen', ('hall', 'kitchen')) in [tuple(fact) for fact in certificate.all_facts]
