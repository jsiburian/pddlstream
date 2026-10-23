import argparse
import inspect
import logging
import time
from collections import defaultdict

from pddlstream.algorithms.algorithm import parse_problem, solve_scope
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.focused import get_negative_externals, solve_adaptive, solve_binding, solve_focused_original
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.refinement import optimistic_process_streams
from pddlstream.algorithms.scheduling.recover_streams import evaluations_from_stream_plan
from pddlstream.algorithms.scheduling.reinstantiate import reinstantiate_axiom
from pddlstream.core.config import ALGORITHMS, DEFAULT_ALGORITHM, FIELD_NAMES, SolverConfig
from pddlstream.core.config import FOCUSED_ALGORITHMS as FOCUSED_ALGORITHMS  # Re-exported (name as name)
from pddlstream.core.conversion import value_from_obj_expression
from pddlstream.core.evaluations import evaluations_from_init
from pddlstream.core.log import console_output
from pddlstream.core.problem import Certificate, PDDLProblem, SolveResult, Status, is_plan
from pddlstream.core.syntax import EQ, get_prefix
from pddlstream.core.utils import INF, SEPARATOR, Verbose, elapsed_time, irange
from pddlstream.fastdownward.grounding import convert_instantiated, instantiate_task
from pddlstream.fastdownward.tasks import (
    INTERNAL_AXIOM,
    fact_from_fd,
    fd_from_evaluations,
    fd_from_fact,
    get_problem,
    task_from_domain_problem,
)
from pddlstream.language.external import DEBUG, SHARED_DEBUG

logger = logging.getLogger(__name__)

# The SolverConfig fields that each algorithm takes
INCREMENTAL_OPTIONS = [
    'unit_costs',
    'success_cost',
    'max_time',
    'max_iterations',
    'max_memory',
    'initial_complexity',
    'complexity_step',
    'max_complexity',
    'verbose',
    'deterministic',
]
FOCUSED_OPTIONS = INCREMENTAL_OPTIONS + ['unit_efforts', 'max_effort', 'effort_weight', 'reorder', 'visualize']
ALGORITHM_OPTIONS = {
    'incremental': INCREMENTAL_OPTIONS,
    'focused': FOCUSED_OPTIONS,
    'binding': FOCUSED_OPTIONS,
    'adaptive': FOCUSED_OPTIONS + ['max_skeletons', 'search_sample_ratio'],
}

##################################################


def create_parser(default_algorithm=DEFAULT_ALGORITHM):
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser()  # Automatically includes help
    parser.add_argument(
        '-a',
        '--algorithm',
        type=str,
        default=default_algorithm,
        choices=ALGORITHMS,
        required=False,
        help='Specifies the PDDLStream algorithm to use',
    )
    parser.add_argument('-u', '--unit', action='store_true', help='Uses unit costs')  # --unit_costs
    # TODO: search planner, debug
    # TODO: method that calls solve with args
    return parser


##################################################


def solve(
    problem,
    algorithm=DEFAULT_ALGORITHM,
    constraints=PlanConstraints(),
    stream_info={},
    replan_actions=set(),
    unit_costs=False,
    success_cost=INF,
    max_time=INF,
    max_iterations=INF,
    max_memory=INF,
    initial_complexity=0,
    complexity_step=1,
    max_complexity=INF,
    max_skeletons=INF,
    search_sample_ratio=1,
    max_failures=0,
    unit_efforts=False,
    max_effort=INF,
    effort_weight=None,
    reorder=True,
    visualize=False,
    verbose=True,
    deterministic=False,
    config=None,
    **kwargs,
):
    """
    Solves a PDDLStream problem generically using one of the available algorithms
    :param problem: a PDDLStream problem
    :param algorithm: a PDDLStream algorithm name
    :param constraints: PlanConstraints on the set of legal solutions
    :param stream_info: a dictionary from stream name to StreamInfo altering how individual streams are handled
    :param replan_actions: the actions declared to induce replanning for the purpose of deferred stream evaluation

    :param unit_costs: use unit action costs rather than numeric costs
    :param success_cost: the exclusive (strict) upper bound on plan cost to successfully terminate
    :param max_time: the maximum runtime
    :param max_iterations: the maximum number of search iterations
    :param max_memory: the maximum amount of memory

    :param initial_complexity: the initial stream complexity limit
    :param complexity_step: the increase in the stream complexity limit per iteration
    :param max_complexity: the maximum stream complexity limit

    :param max_skeletons: the maximum number of plan skeletons (max_skeletons=None indicates not adaptive)
    :param search_sample_ratio: the desired ratio of sample time / search time when max_skeletons!=None
    :param max_failures: the maximum number of stream failures before switching phases when max_skeletons=None

    :param unit_efforts: use unit stream efforts rather than estimated numeric efforts
    :param max_effort: the maximum amount of stream effort
    :param effort_weight: a multiplier for stream effort compared to action costs
    :param reorder: if True, reorder stream plans to minimize the expected sampling overhead

    :param visualize: if True, draw the constraint network and stream plan as a graphviz file
    :param verbose: if True, print the result of each stream application
    :param deterministic: if True, make seeded runs repeatable: time-based decisions (e.g. the sampling budget) count
        work units instead of seconds, and saved stream statistics are not loaded. max_time is still wall-clock time
    :param config: a SolverConfig holding these options; keyword arguments that differ from their defaults
        override its fields
    :param kwargs: the other SolverConfig fields, such as the search options planner (a key of SEARCH_OPTIONS),
        max_planner_time and debug. Each search runs in a private temporary directory, except that debug=True
        prints FastDownward's output and keeps its files in ./temp/ (temp_dir=... chooses another directory).
        Unknown names raise a TypeError before solving starts

    :return: a SolveResult, which unpacks as (plan, cost, certificate): plan is a sequence of actions (or None),
        cost is its cost (INF without a plan), and certificate holds the facts that support it. Its status and stats
        attributes report how the solve ended
    """
    arguments = {name: value for name, value in locals().items() if name in FIELD_NAMES}
    defaults = {name: p.default for name, p in inspect.signature(solve).parameters.items() if name in FIELD_NAMES}
    overrides = {name: value for name, value in arguments.items() if value != defaults[name]}
    config = SolverConfig.from_kwargs(config, **overrides, **kwargs)
    # TODO: portfolios of PDDLStream algorithms

    options = {name: getattr(config, name) for name in ALGORITHM_OPTIONS[config.algorithm]}
    options.update(config.search_kwargs())
    with solve_scope(config):  # The solver's options also reach the context, but not, e.g., load_statistics
        if config.algorithm == 'incremental':
            return solve_incremental(problem, constraints=constraints, **options)
        options.update(stream_info=stream_info, replan_actions=replan_actions)
        if config.algorithm == 'adaptive':
            return solve_adaptive(problem, constraints=constraints, **options)
        solver = solve_focused_original if config.algorithm == 'focused' else solve_binding
        return solver(problem, constraints=constraints, fail_fast=config.max_failures < INF, **options)


##################################################


@console_output()
def solve_restart(problem, max_time=INF, max_restarts=0, iteration_time=INF, abort=True, **kwargs):
    # TODO: iteratively lower the cost bound
    # TODO: a sequence of different planner configurations
    # TODO: reset objects and/or streams
    if (max_restarts >= 1) and (iteration_time == INF):
        iteration_time = min(2 * 60, iteration_time)
    assert (max_restarts == 0) or (iteration_time != INF)

    assert max_restarts >= 0
    start_time = time.time()
    for attempt in irange(1 + max_restarts):
        iteration_start_time = time.time()
        if elapsed_time(start_time) > max_time:
            break
        if attempt >= 1:
            logger.info(SEPARATOR)
        remaining_time = min(iteration_time, max_time - elapsed_time(start_time))
        solution = solve(problem, max_time=remaining_time, **kwargs)
        plan, cost, certificate = solution
        if is_plan(plan):  # TODO: INFEASIBLE
            return solution
        if abort and (elapsed_time(iteration_start_time) < remaining_time):
            break  # TODO: return the cause of failure

    certificate = Certificate(all_facts=[], preimage_facts=[])  # TODO: aggregate
    return SolveResult(None, INF, certificate, status=Status.FAILED)


##################################################


@console_output()
def examine_instantiated(problem, unique=False, normalize=True, unit_costs=False, verbose=False, debug=False, **kwargs):
    # TODO: refactor to an analysis file
    domain_pddl, constant_map, stream_pddl, _, init, goal = problem
    stream_map = DEBUG if unique else SHARED_DEBUG  # DEBUG_MODES
    problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    evaluations, goal_exp, domain, externals = parse_problem(problem, **kwargs)
    negative = get_negative_externals(externals)
    externals = list(filter(lambda s: s not in negative, externals))

    # TODO: drop all fluents and instantiate
    # TODO: relaxed planning version of this
    results, exhausted = optimistic_process_streams(evaluations, externals, complexity_limit=INF, max_effort=None)

    evaluations = evaluations_from_stream_plan(evaluations, results, max_effort=None)
    problem = get_problem(evaluations, goal_exp, domain, unit_costs)
    task = task_from_domain_problem(domain, problem)
    with Verbose(debug):
        instantiated = instantiate_task(task, check_infeasible=False)
        if instantiated is None:
            return results, None
        # TODO: reinstantiate actions?
        instantiated.axioms[:] = [reinstantiate_axiom(axiom) for axiom in instantiated.axioms]
        if normalize:
            instantiated = convert_instantiated(instantiated)
    return results, instantiated


##################################################


def recurse_subgoals(goals, condition_from_effect):
    possible = set()

    def recurse(goal):
        if goal in possible:
            return
        possible.add(goal)
        for condition in condition_from_effect[goal]:
            recurse(condition)

    for goal in goals:
        recurse(goal)
    return possible


@console_output()
def analyze_goal(problem, use_actions=False, use_axioms=True, use_streams=True, blocked_predicates=[], **kwargs):
    # TODO: instantiate all goal partial states
    # TODO: remove actions/axioms that never could achieve a subgoal
    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = problem
    evaluations = evaluations_from_init(init)
    init = set(fd_from_evaluations(evaluations))

    results, instantiated = examine_instantiated(problem, **kwargs)  # TODO: only do if the goals are derived
    if instantiated is None:
        return None

    # This is like backchaining in a relaxed space
    condition_from_effect = defaultdict(set)
    if use_actions:
        # TODO: selectively ignore some conditions (e.g. HandEmpty)
        # TODO: refactor into separate method
        for action in instantiated.actions:
            for conditional, effect in action.add_effects:
                for condition in action.precondition + conditional:
                    if condition.predicate not in blocked_predicates:
                        condition_from_effect[effect].add(condition)
            for conditional, effect in action.del_effects:
                for condition in action.precondition + conditional:
                    if condition.predicate not in blocked_predicates:
                        condition_from_effect[effect.negate()].add(condition)
    if use_axioms:
        # TODO: axiom_rules.handle_axioms(...)
        for axiom in instantiated.axioms:
            for condition in axiom.condition:
                condition_from_effect[axiom.effect].add(condition)
    if use_streams:
        for result in results:
            for effect in result.certified:
                if get_prefix(effect) == EQ:
                    continue
                for condition in result.domain:
                    condition_from_effect[fd_from_fact(effect)].add(fd_from_fact(condition))

    logger.info(f'Goals: {list(map(fact_from_fd, instantiated.goal_list))}')
    all_subgoals = recurse_subgoals(instantiated.goal_list, condition_from_effect)
    filtered_subgoals = [subgoal for subgoal in all_subgoals if subgoal in init]  # TODO: return the goals as well?
    external_subgoals = [
        value_from_obj_expression(fact_from_fd(subgoal))
        for subgoal in sorted(filtered_subgoals, key=lambda g: g.predicate)
        if not subgoal.predicate.startswith(INTERNAL_AXIOM)
    ]
    logger.info(f'Initial: {external_subgoals}')

    return external_subgoals  # TODO: decompose into simplified components
