import logging

from pddlstream.algorithms.advanced import automatically_negate_externals, enforce_simultaneous
from pddlstream.algorithms.algorithm import parse_problem, scoped_solve
from pddlstream.algorithms.common import SolutionStore
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.disable_skeleton import create_disabled_axioms
from pddlstream.algorithms.disabled import process_stream_plan, push_disabled, reenable_disabled
from pddlstream.algorithms.incremental import process_stream_queue
from pddlstream.algorithms.instantiation import Instantiator
from pddlstream.algorithms.recover_optimizers import combine_optimizers
from pddlstream.algorithms.refinement import get_optimistic_solve_fn, iterative_plan_streams
from pddlstream.algorithms.reorder import reorder_stream_plan
from pddlstream.algorithms.scheduling.plan_streams import OptSolution
from pddlstream.algorithms.skeleton import SkeletonQueue
from pddlstream.algorithms.visualization import create_visualizations, has_pygraphviz, log_plans, reset_visualizations
from pddlstream.core.context import get_config, note_gap
from pddlstream.core.problem import INFEASIBLE, get_length, is_plan, str_from_plan
from pddlstream.core.utils import INF, implies
from pddlstream.language.fluent import compile_fluent_streams
from pddlstream.language.function import Function, Predicate
from pddlstream.language.optimizer import ComponentStream
from pddlstream.language.statistics import compute_plan_effort, load_stream_statistics
from pddlstream.language.stream import Stream

logger = logging.getLogger(__name__)


def get_negative_externals(externals):
    negative_predicates = list(filter(lambda s: type(s) is Predicate, externals))  # and s.is_negative()
    negated_streams = list(filter(lambda s: isinstance(s, Stream) and s.is_negated, externals))
    return negative_predicates + negated_streams


def partition_externals(externals, verbose=False):
    functions = list(filter(lambda s: type(s) is Function, externals))
    negative = get_negative_externals(externals)
    optimizers = list(filter(lambda s: isinstance(s, ComponentStream) and (s not in negative), externals))
    streams = list(filter(lambda s: s not in (functions + negative + optimizers), externals))
    if verbose:
        logger.info(f'Streams: {streams}\nFunctions: {functions}\nNegated: {negative}\nOptimizers: {optimizers}')
    return streams, functions, negative, optimizers


##################################################


@scoped_solve
def solve_abstract(
    problem,
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
    search_sample_ratio=0,
    bind=True,
    max_failures=0,
    unit_efforts=False,
    max_effort=INF,
    effort_weight=None,
    reorder=True,
    visualize=False,
    verbose=True,
    deterministic=False,
    **search_kwargs,
):
    """The algorithm behind focused, binding and adaptive: plans with optimistic stream outputs, then samples the
    streams that the plans use.

    :param problem: a PDDLProblem
    :param constraints: PlanConstraints that plans must satisfy, such as plan skeletons
    :param stream_info: a StreamInfo, FunctionInfo or PredicateInfo by stream or function name
    :param replan_actions: actions after which deferred streams are evaluated and the solver replans
    :param bind: propagate the outputs of sampled streams to the rest of the stream plan (without skeletons)
    :param max_failures: stream failures before switching back to planning (without skeletons)
    The other parameters are SolverConfig fields, documented there; search_kwargs are its search options
    (planner, max_planner_time, debug, ...). Note that search_sample_ratio defaults to 0 here, unlike in solve().
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    # TODO: select whether to search or sample based on expected success rates
    # TODO: no optimizers during search with relaxed_stream_plan
    # TODO: locally optimize only after a solution is identified
    # TODO: replan with a better search algorithm after feasible
    # TODO: change the search algorithm and unit costs based on the best cost
    use_skeletons = max_skeletons is not None
    eager_disabled = effort_weight is None  # No point if no stream effort biasing
    num_iterations = eager_calls = 0
    complexity_limit = initial_complexity

    evaluations, goal_exp, domain, externals = compile_problem(
        problem, stream_info, constraints, unit_costs, unit_efforts
    )
    # TODO: make effort_weight be a function of the current cost
    if not deterministic:
        load_stream_statistics(externals)
    visualize = prepare_visualizations(visualize)
    streams, functions, negative, optimizers = partition_externals(externals, verbose=verbose)
    eager_externals = list(filter(lambda e: e.info.eager, externals))
    positive_externals = streams + functions + optimizers
    has_optimizers = bool(optimizers)  # TODO: deprecate
    assert implies(has_optimizers, use_skeletons)

    store = SolutionStore(evaluations, max_time, success_cost, verbose, max_memory=max_memory)
    skeleton_queue = SkeletonQueue(store, domain, disable=not has_optimizers)
    disabled = set()  # Max skeletons after a solution
    infeasible = False
    while (not store.is_terminated()) and (num_iterations < max_iterations) and (complexity_limit <= max_complexity):
        num_iterations += 1
        eager_instantiator, num_calls = process_eager_streams(
            eager_externals, store, disabled, complexity_limit, eager_disabled, verbose
        )
        eager_calls += num_calls
        log_iteration(store, skeleton_queue, disabled, num_iterations, complexity_limit, eager_calls)

        # Plan with optimistic stream outputs
        optimistic_solve_fn = get_optimistic_solve_fn(
            goal_exp,
            domain,
            negative,
            replan_actions=replan_actions,
            reachieve=use_skeletons,
            max_cost=min(store.best_cost, constraints.max_cost),
            max_effort=max_effort,
            effort_weight=effort_weight,
            **search_kwargs,
        )
        # TODO: just set unit effort for each stream beforehand
        if (max_skeletons is None) or (len(skeleton_queue.skeletons) < max_skeletons):
            stream_plan, opt_plan, cost = plan_optimistically(
                skeleton_queue, positive_externals, optimistic_solve_fn, complexity_limit, max_effort, has_optimizers
            )
        else:
            note_gap('max_skeletons')
            stream_plan, opt_plan, cost = OptSolution(INFEASIBLE, INFEASIBLE, INF)  # TODO: apply elsewhere
        stream_plan = combine_optimizers(evaluations, stream_plan)
        if reorder:
            # TODO: this blows up memory wise for long stream plans
            stream_plan = reorder_stream_plan(store, stream_plan)
        report_plans(evaluations, stream_plan, opt_plan, cost, num_iterations, visualize)

        if (stream_plan is INFEASIBLE) and (not eager_instantiator) and (not skeleton_queue) and (not disabled):
            note_unreached_facts(evaluations, complexity_limit)
            infeasible = True
            break
        if not is_plan(stream_plan):
            logger.info(
                f'No plan: increasing complexity from {complexity_limit} to {complexity_limit + complexity_step}'
            )
            complexity_limit += complexity_step
            if not eager_disabled:
                reenable_disabled(evaluations, domain, disabled)

        # Sample the plan's streams: greedily for focused and binding, through the skeleton queue for adaptive
        if not use_skeletons:
            process_stream_plan(
                store, domain, disabled, stream_plan, opt_plan, cost, bind=bind, max_failures=max_failures
            )
            continue
        allocated_sample_time = (
            (search_sample_ratio * store.search_time) - store.sample_time
            if len(skeleton_queue.skeletons) <= max_skeletons
            else INF
        )
        if skeleton_queue.process(stream_plan, opt_plan, cost, complexity_limit, allocated_sample_time) is INFEASIBLE:
            note_unreached_facts(evaluations, complexity_limit)
            infeasible = True
            break

    return store.finish(
        externals,
        infeasible,
        iterations=num_iterations,
        complexity=complexity_limit,
        skeletons=len(skeleton_queue.skeletons),
    )


def note_unreached_facts(evaluations, complexity_limit):
    """Optimistic planning only uses facts within the complexity limit, so facts above it make a conclusion that no
    plan exists unsound."""
    if any(node.complexity > complexity_limit for node in evaluations.values()):
        note_gap('unreached_facts')


def compile_problem(problem, stream_info, constraints, unit_costs, unit_efforts):
    """Parses the problem and compiles its negated, simultaneous and fluent streams into the domain."""
    evaluations, goal_exp, domain, externals = parse_problem(
        problem, stream_info=stream_info, constraints=constraints, unit_costs=unit_costs, unit_efforts=unit_efforts
    )
    automatically_negate_externals(domain, externals)
    enforce_simultaneous(domain, externals)
    compile_fluent_streams(domain, externals)
    if get_config().strict:
        for external in externals:
            if isinstance(external, Stream) and external.is_negated and not external.is_complete:
                logger.warning(
                    f'{external.name} is used negated but not declared complete, so its failures are trusted'
                )
    return evaluations, goal_exp, domain, externals


def prepare_visualizations(visualize):
    """Whether to visualize, which needs pygraphviz; clears earlier visualizations if so."""
    if visualize and not has_pygraphviz():
        logger.warning('visualize=True requires pygraphviz. Setting visualize=False')
        return False
    if visualize:
        reset_visualizations()
    return visualize


def process_eager_streams(eager_externals, store, disabled, complexity_limit, eager_disabled, verbose):
    """Calls the eager streams up to the complexity limit, returning their instantiator and the number of calls."""
    eager_instantiator = Instantiator(eager_externals, store.evaluations)  # Only update after an increase?
    if eager_disabled:
        push_disabled(eager_instantiator, disabled)
    if not eager_externals:
        return eager_instantiator, 0
    num_calls = process_stream_queue(eager_instantiator, store, complexity_limit=complexity_limit, verbose=verbose)
    return eager_instantiator, num_calls


def plan_optimistically(skeleton_queue, externals, optimistic_solve_fn, complexity_limit, max_effort, has_optimizers):
    """Searches for a stream plan and action plan; with optimizers, the skeletons found so far are excluded."""
    domain = skeleton_queue.domain
    disabled_axioms = create_disabled_axioms(skeleton_queue) if has_optimizers else []
    domain.axioms.extend(disabled_axioms)
    stream_plan, opt_plan, cost = iterative_plan_streams(
        skeleton_queue.evaluations, externals, optimistic_solve_fn, complexity_limit, max_effort=max_effort
    )
    for axiom in disabled_axioms:
        domain.axioms.remove(axiom)
    return stream_plan, opt_plan, cost


def log_iteration(store, skeleton_queue, disabled, num_iterations, complexity_limit, eager_calls):
    logger.info(
        f'\nIteration: {num_iterations} | Complexity: {complexity_limit} | '
        f'Skeletons: {len(skeleton_queue.skeletons)} | Skeleton Queue: {len(skeleton_queue)} | '
        f'Disabled: {len(disabled)} | Evaluations: {len(store.evaluations)} | Eager Calls: {eager_calls} | '
        f'Cost: {store.best_cost:.3f} | Search Time: {store.search_time:.3f} | '
        f'Sample Time: {store.sample_time:.3f} | Total Time: {store.elapsed_time():.3f}'
    )


def report_plans(evaluations, stream_plan, opt_plan, cost, num_iterations, visualize):
    """Logs the stream plan and action plan, and draws them when visualizing."""
    num_optimistic = sum(r.optimistic for r in stream_plan) if stream_plan else 0
    action_plan = opt_plan.action_plan if is_plan(opt_plan) else opt_plan
    logger.info(
        f'Stream plan ({get_length(stream_plan)}, {num_optimistic}, '
        f'{compute_plan_effort(stream_plan):.3f}): {stream_plan}\n'
        f'Action plan ({get_length(action_plan)}, {cost:.3f}): {str_from_plan(action_plan)}'
    )
    if is_plan(stream_plan) and visualize:
        log_plans(stream_plan, action_plan, num_iterations)
        create_visualizations(evaluations, stream_plan, num_iterations)


solve_focused = solve_abstract  # TODO: deprecate solve_focused

##################################################


def solve_focused_original(problem, fail_fast=False, **kwargs):
    """The focused algorithm: plans with optimistic stream outputs, calls the plan's streams, and repeats (raising
    the complexity limit when no plan is found) until a plan's streams all succeed. fail_fast switches back to
    planning after the first stream failure. The other keyword arguments are those of solve_abstract.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    max_failures = 0 if fail_fast else INF
    return solve_abstract(
        problem, max_skeletons=None, search_sample_ratio=None, bind=False, max_failures=max_failures, **kwargs
    )


def solve_binding(problem, fail_fast=False, **kwargs):
    """The binding algorithm: like focused, but the outputs that the plan's streams produce are bound into the rest of
    the stream plan before it is called. fail_fast switches back to planning after the first stream failure. The
    other keyword arguments are those of solve_abstract.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    max_failures = 0 if fail_fast else INF
    return solve_abstract(
        problem, max_skeletons=None, search_sample_ratio=None, bind=True, max_failures=max_failures, **kwargs
    )


def solve_adaptive(problem, max_skeletons=INF, search_sample_ratio=1, **kwargs):
    """The adaptive algorithm, usually the best choice: like binding, but it keeps a queue of plan skeletons and
    divides its time between sampling them and searching for new ones.

    :param max_skeletons: the most skeletons to keep (None plans without skeletons, like binding)
    :param search_sample_ratio: the desired ratio of sampling time to search time
    The other keyword arguments are those of solve_abstract.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    max_skeletons = INF if max_skeletons is None else max_skeletons
    return solve_abstract(
        problem,
        max_skeletons=max_skeletons,
        search_sample_ratio=search_sample_ratio,
        bind=None,
        max_failures=None,
        **kwargs,
    )


def solve_hierarchical(problem, **kwargs):
    """The adaptive algorithm with a single plan skeleton that it samples until it succeeds.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    return solve_adaptive(
        problem, max_skeletons=1, search_sample_ratio=INF, **kwargs
    )  # TODO: rename to sample_search_ratio
