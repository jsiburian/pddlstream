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
from pddlstream.algorithms.strategies import SolveState, drive, finish
from pddlstream.algorithms.visualization import create_visualizations, has_pygraphviz, log_plans, reset_visualizations
from pddlstream.core.context import get_config, note_gap
from pddlstream.core.problem import INFEASIBLE, get_length, is_plan, str_from_plan
from pddlstream.core.trace import Iteration, SkeletonProgress, emit
from pddlstream.core.utils import INF
from pddlstream.language.fluent import compile_fluent_streams
from pddlstream.language.function import Function, Predicate
from pddlstream.language.lint import CapabilityError
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
    evaluations, goal_exp, domain, externals = compile_problem(
        problem, stream_info, constraints, unit_costs, unit_efforts
    )
    # TODO: make effort_weight be a function of the current cost
    if not deterministic:
        load_stream_statistics(externals)
    visualize = prepare_visualizations(visualize)
    streams, functions, negative, optimizers = partition_externals(externals, verbose=verbose)
    has_optimizers = bool(optimizers)  # TODO: deprecate
    if has_optimizers and not use_skeletons:
        raise CapabilityError('Optimizers need the adaptive algorithm (with plan skeletons)')

    store = SolutionStore(evaluations, max_time, success_cost, verbose, max_memory=max_memory)
    state = SolveState(
        store, domain, goal_exp, externals, constraints, initial_complexity, complexity_step, max_iterations,
        max_complexity,
    )  # fmt: skip
    search = OptimisticSearch(
        negative,
        streams + functions + optimizers,
        replan_actions=replan_actions,
        reachieve=use_skeletons,
        max_effort=max_effort,
        effort_weight=effort_weight,
        has_optimizers=has_optimizers,
        reorder=reorder,
        visualize=visualize,
        search_kwargs=search_kwargs,
    )
    eager_externals = list(filter(lambda e: e.info.eager, externals))
    refinement_args = dict(eager_externals=eager_externals, effort_weight=effort_weight, verbose=verbose)
    if use_skeletons:
        refinement = SkeletonRefinement(max_skeletons, search_sample_ratio, has_optimizers, **refinement_args)
    else:
        refinement = GreedyRefinement(bind, max_failures, has_optimizers, **refinement_args)
    drive(state, search, refinement)
    return finish(state, refinement)


class OptimisticSearch:
    """Plans with optimistic stream outputs (focused, binding and adaptive)."""

    def __init__(
        self,
        negative,
        externals,
        replan_actions=set(),
        reachieve=True,
        max_effort=INF,
        effort_weight=None,
        has_optimizers=False,
        reorder=True,
        visualize=False,
        search_kwargs={},
    ):
        self.negative, self.externals = negative, externals
        self.replan_actions, self.reachieve = replan_actions, reachieve
        self.max_effort, self.effort_weight, self.has_optimizers = max_effort, effort_weight, has_optimizers
        self.reorder, self.visualize, self.search_kwargs = reorder, visualize, search_kwargs

    def propose(self, state, refinement):
        optimistic_solve_fn = get_optimistic_solve_fn(
            state.goal_exp,
            state.domain,
            self.negative,
            replan_actions=self.replan_actions,
            reachieve=self.reachieve,
            max_cost=state.max_cost(),
            max_effort=self.max_effort,
            effort_weight=self.effort_weight,
            **self.search_kwargs,
        )
        # TODO: just set unit effort for each stream beforehand
        return plan_optimistically(
            refinement.skeleton_queue,
            self.externals,
            optimistic_solve_fn,
            state.complexity_limit,
            self.max_effort,
            self.has_optimizers,
        )

    def postprocess(self, state, solution):
        stream_plan, opt_plan, cost = solution
        stream_plan = combine_optimizers(state.evaluations, stream_plan)
        if self.reorder:
            # TODO: this blows up memory wise for long stream plans
            stream_plan = reorder_stream_plan(state.store, stream_plan)
        report_plans(state.evaluations, stream_plan, opt_plan, cost, state.iteration, self.visualize)
        emit(get_iteration(state.iteration, state.complexity_limit, stream_plan, opt_plan))
        return OptSolution(stream_plan, opt_plan, cost)


class FocusedRefinement:
    """What focused, binding and adaptive share: eager streams before each search, the conclusion that no plan exists
    when nothing is left to try, and raising the complexity limit when the search finds no plan."""

    def __init__(self, has_optimizers=False, eager_externals=(), effort_weight=None, verbose=True):
        self.has_optimizers, self.eager_externals, self.verbose = has_optimizers, eager_externals, verbose
        self.eager_disabled = effort_weight is None  # No point if no stream effort biasing
        self.disabled = set()  # Max skeletons after a solution
        self.eager_instantiator, self.eager_calls = None, 0
        self.skeleton_queue = None

    def initialize(self, state):
        self.skeleton_queue = SkeletonQueue(state.store, state.domain, disable=not self.has_optimizers)

    def before_search(self, state):
        self.eager_instantiator, num_calls = process_eager_streams(
            self.eager_externals, state.store, self.disabled, state.complexity_limit, self.eager_disabled, self.verbose
        )
        self.eager_calls += num_calls
        log_iteration(
            state.store, self.skeleton_queue, self.disabled, state.iteration, state.complexity_limit, self.eager_calls
        )

    def may_search(self, state):
        return True

    def after_search(self, state, solution):
        stream_plan, _, _ = solution
        nothing_left = (not self.eager_instantiator) and (not self.skeleton_queue) and (not self.disabled)
        if (stream_plan is INFEASIBLE) and nothing_left:
            note_unreached_facts(state.evaluations, state.complexity_limit)
            state.infeasible = True
            return True
        if not is_plan(stream_plan):
            logger.info(
                f'No plan: increasing complexity from {state.complexity_limit} to '
                f'{state.complexity_limit + state.complexity_step}'
            )
            state.complexity_limit += state.complexity_step
            if not self.eager_disabled:
                reenable_disabled(state.evaluations, state.domain, self.disabled)
        return self.sample(state, solution)

    def sample(self, state, solution):
        raise NotImplementedError()

    def stats(self, state):
        for skeleton in self.skeleton_queue.skeletons:
            emit(get_progress(skeleton))
        return {'skeletons': len(self.skeleton_queue.skeletons)}


class GreedyRefinement(FocusedRefinement):
    """Calls the plan's streams in order (focused), binding their outputs into later streams (binding)."""

    def __init__(self, bind=True, max_failures=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind, self.max_failures = bind, max_failures

    def sample(self, state, solution):
        stream_plan, opt_plan, cost = solution
        process_stream_plan(
            state.store, state.domain, self.disabled, stream_plan, opt_plan, cost, bind=self.bind,
            max_failures=self.max_failures,
        )  # fmt: skip
        return False


class SkeletonRefinement(FocusedRefinement):
    """Keeps a queue of plan skeletons and divides its time between sampling them and searching for new ones."""

    def __init__(self, max_skeletons=INF, search_sample_ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_skeletons, self.search_sample_ratio = max_skeletons, search_sample_ratio

    def may_search(self, state):
        if len(self.skeleton_queue.skeletons) < self.max_skeletons:
            return True
        note_gap('max_skeletons')
        return False

    def sample(self, state, solution):
        stream_plan, opt_plan, cost = solution
        store = state.store
        allocated_sample_time = (
            (self.search_sample_ratio * store.search_time) - store.sample_time
            if len(self.skeleton_queue.skeletons) <= self.max_skeletons
            else INF
        )
        if (
            self.skeleton_queue.process(stream_plan, opt_plan, cost, state.complexity_limit, allocated_sample_time)
            is INFEASIBLE
        ):
            note_unreached_facts(state.evaluations, state.complexity_limit)
            state.infeasible = True
            return True
        return False


def get_iteration(index, complexity, stream_plan, opt_plan):
    """The Iteration event of a search that returned stream_plan and opt_plan."""
    status = 'plan' if is_plan(opt_plan) else 'infeasible' if stream_plan is INFEASIBLE else 'failed'
    num_streams = len(stream_plan) if is_plan(stream_plan) else 0
    num_actions = len(opt_plan.action_plan) if is_plan(opt_plan) else 0
    return Iteration(index, complexity, status, num_streams, num_actions)


def get_progress(skeleton):
    bound = skeleton.best_binding.index
    stream_plan = skeleton.stream_plan
    next_stream = stream_plan[bound].external.name if bound < len(stream_plan) else None
    return SkeletonProgress(skeleton.index, bound, len(stream_plan), next_stream)


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


def solve_focused_original(problem, fail_fast=False, max_failures=None, **kwargs):
    """The focused algorithm: plans with optimistic stream outputs, calls the plan's streams, and repeats (raising
    the complexity limit when no plan is found) until a plan's streams all succeed. fail_fast switches back to
    planning after the first stream failure, and max_failures after that many (by default 0 with fail_fast, else
    INF). The other keyword arguments are those of solve_abstract.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    if max_failures is None:
        max_failures = 0 if fail_fast else INF
    return solve_abstract(
        problem, max_skeletons=None, search_sample_ratio=None, bind=False, max_failures=max_failures, **kwargs
    )


def solve_binding(problem, fail_fast=False, max_failures=None, **kwargs):
    """The binding algorithm: like focused, but the outputs that the plan's streams produce are bound into the rest of
    the stream plan before it is called. fail_fast and max_failures are as for solve_focused_original. The other
    keyword arguments are those of solve_abstract.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    if max_failures is None:
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
