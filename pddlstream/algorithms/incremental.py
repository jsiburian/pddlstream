import logging
from collections import Counter

from pddlstream.algorithms.algorithm import parse_problem, scoped_solve
from pddlstream.algorithms.common import SolutionStore, get_action_plan
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.instantiation import Instantiator
from pddlstream.algorithms.scheduling.plan_streams import OptSolution, preimage_from_plan
from pddlstream.algorithms.strategies import SolveState, drive, finish
from pddlstream.core.conversion import obj_from_pddl_plan
from pddlstream.core.evaluations import add_certified, add_facts
from pddlstream.core.problem import OptPlan, is_plan
from pddlstream.core.trace import Iteration, emit
from pddlstream.core.utils import INF, Verbose, str_from_object
from pddlstream.fastdownward.grounding import instantiate_task, sas_from_instantiated
from pddlstream.fastdownward.search import abstrips_solve_from_task
from pddlstream.fastdownward.tasks import get_problem, task_from_domain_problem
from pddlstream.language.statistics import load_stream_statistics

logger = logging.getLogger(__name__)

UPDATE_STATISTICS = False


def solve_finite(evaluations, goal_exp, domain, unit_costs=False, debug=False, **search_args):
    """A plan for the problem with the facts in evaluations (or None), its cost, and the grounded task."""
    problem = get_problem(evaluations, goal_exp, domain, unit_costs)
    task = task_from_domain_problem(domain, problem)
    with Verbose(debug):
        instantiated = instantiate_task(task)
        sas_task = sas_from_instantiated(instantiated)
        sas_task.metric = task.use_min_cost_metric
    pddl_plan, cost = abstrips_solve_from_task(sas_task, debug=debug, **search_args)
    plan = obj_from_pddl_plan(pddl_plan)
    return plan, cost, instantiated


##################################################


def process_instance(instantiator, store, instance, verbose=False):
    if instance.enumerated:
        return []
    new_results, new_facts = store.sample(instance, verbose=verbose)

    evaluations = store.evaluations
    for result in new_results:
        complexity = result.compute_complexity(evaluations)
        for evaluation in add_certified(evaluations, result):
            instantiator.add_atom(evaluation, complexity)
    fact_complexity = 0  # Wild facts are supported by the instance that produced them
    for evaluation in add_facts(evaluations, new_facts, result=instance, complexity=fact_complexity):
        instantiator.add_atom(evaluation, fact_complexity)
    if not instance.enumerated:
        instantiator.push_instance(instance)
    return new_results


def process_stream_queue(instantiator, store, complexity_limit=INF, verbose=False):
    instances = []
    results = []
    num_successes = 0
    while not store.is_terminated() and instantiator and (instantiator.min_complexity() <= complexity_limit):
        instance = instantiator.pop_stream()
        if instance.enumerated:
            continue
        instances.append(instance)
        new_results = process_instance(instantiator, store, instance, verbose=verbose)
        results.extend(new_results)
        num_successes += bool(new_results)  # TODO: max_results?
    if verbose:
        counts = Counter(instance.external.name for instance in instances)
        logger.info(
            f'Eager Calls: {len(instances)} | Successes: {num_successes} | Results: {len(results)} | '
            f'Counts: {str_from_object(counts)}'
        )
    return len(instances)


##################################################


@scoped_solve
def solve_incremental(
    problem,
    constraints=PlanConstraints(),
    stream_info={},
    unit_costs=False,
    success_cost=INF,
    max_iterations=INF,
    max_time=INF,
    max_memory=INF,
    initial_complexity=0,
    complexity_step=1,
    max_complexity=INF,
    verbose=False,
    deterministic=False,
    **search_kwargs,
):
    """The incremental algorithm: alternates between calling every stream instance up to the complexity limit and
    searching for a plan with the facts found so far. It needs no optimistic stream outputs, but calls many
    streams that the plan does not need.

    :param problem: a PDDLProblem
    :param constraints: PlanConstraints that plans must satisfy, such as plan skeletons
    :param stream_info: a StreamInfo, FunctionInfo or PredicateInfo by stream or function name (incremental uses
        their complete and reads options)
    The other parameters are SolverConfig fields, documented there (max_complexity=0 uses only the initial facts,
    complexity_step=INF calls everything at once); search_kwargs are its search options (planner, debug, ...).
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    # max_complexity = 0 => current
    # complexity_step = INF => exhaustive
    # TODO: warning if optimizers are present
    evaluations, goal_expression, domain, externals = parse_problem(
        problem, stream_info=stream_info, constraints=constraints, unit_costs=unit_costs
    )
    store = SolutionStore(
        evaluations, max_time, success_cost, verbose, max_memory=max_memory
    )  # TODO: include other info here?
    if UPDATE_STATISTICS and not deterministic:
        load_stream_statistics(externals)
    state = SolveState(
        store, domain, goal_expression, externals, constraints, initial_complexity, complexity_step, max_iterations,
        max_complexity,
    )  # fmt: skip
    refinement = ExhaustiveRefinement(externals, verbose)
    drive(state, FiniteSearch(search_kwargs), refinement)
    return finish(state, refinement, statistics_externals=externals if UPDATE_STATISTICS else [])


class FiniteSearch:
    """Plans with the facts found so far (incremental), recording what each better plan relies on."""

    def __init__(self, search_kwargs={}):
        self.search_kwargs = search_kwargs

    def propose(self, state, refinement):
        plan, cost, instantiated = solve_finite(
            state.evaluations, state.goal_exp, state.domain, max_cost=state.max_cost(), **self.search_kwargs
        )
        if is_plan(plan) and (cost < state.store.best_cost):
            # Records what the plan relies on, for its certificate, reusing the search's grounded task
            preimage_facts = preimage_from_plan(state.evaluations, state.goal_exp, state.domain, plan, instantiated)
            if preimage_facts is not None:
                plan = OptPlan(plan, preimage_facts)
        return OptSolution([], plan, cost)

    def postprocess(self, state, solution):
        plan = solution.opt_plan
        num_actions = len(get_action_plan(plan)) if is_plan(plan) else 0
        emit(Iteration(state.iteration, state.complexity_limit, 'plan' if is_plan(plan) else 'failed', 0, num_actions))
        return solution


class ExhaustiveRefinement:
    """Calls every stream instance up to the complexity limit after each search (incremental)."""

    unsupported = {
        'fluent': 'The incremental algorithm does not support fluent streams ({}); use a focused algorithm such as '
        'adaptive'
    }

    def __init__(self, externals, verbose=False):
        self.externals, self.verbose = externals, verbose
        self.instantiator, self.num_calls = None, 0

    def initialize(self, state):
        self.instantiator = Instantiator(self.externals, state.evaluations)
        self.num_calls += process_stream_queue(
            self.instantiator, state.store, state.complexity_limit, verbose=self.verbose
        )

    def before_search(self, state):
        store = state.store
        logger.info(
            f'Iteration: {state.iteration} | Complexity: {state.complexity_limit} | Calls: {self.num_calls} | '
            f'Evaluations: {len(state.evaluations)} | Solved: {store.has_solution()} | Cost: {store.best_cost:.3f} | '
            f'Search Time: {store.search_time:.3f} | Sample Time: {store.sample_time:.3f} | '
            f'Time: {store.elapsed_time():.3f}'
        )

    def may_search(self, state):
        return True

    def after_search(self, state, solution):
        _, plan, cost = solution
        if is_plan(plan) and (cost < state.store.best_cost):
            state.store.add_plan(plan, cost)
        if not self.instantiator:
            state.infeasible = not state.store.has_solution()
            return True
        if state.complexity_step is None:
            # TODO: option to select the next k-smallest complexities
            state.complexity_limit = self.instantiator.min_complexity()
        else:
            state.complexity_limit += state.complexity_step
        self.num_calls += process_stream_queue(
            self.instantiator, state.store, state.complexity_limit, verbose=self.verbose
        )
        return False

    def stats(self, state):
        return {}


##################################################


def solve_immediate(problem, **kwargs):
    """
    Solves a PDDLStream problem by searching only
    INCOMPLETENESS WARNING: only use if no stream evaluations are necessarily (otherwise terminates early)
    :param problem: a PDDLStream problem
    :param kwargs: keyword args for solve_incremental
    :return: a SolveResult, which unpacks as (plan, cost, certificate): plan is a sequence of actions (or None),
        cost is its cost (INF without a plan), and certificate holds the facts that support it. Its status and stats
        attributes report how the solve ended
    """
    return solve_incremental(problem, initial_complexity=0, complexity_step=0, max_complexity=0, **kwargs)


def solve_exhaustive(problem, **kwargs):
    """
    Solves a PDDLStream problem by applying all possible streams and searching once
    INCOMPLETENESS WARNING: only use if a finite set of instantiable stream instances (otherwise infinite loop)
    :param problem: a PDDLStream problem
    :param kwargs: keyword args for solve_incremental
    :return: a SolveResult, which unpacks as (plan, cost, certificate): plan is a sequence of actions (or None),
        cost is its cost (INF without a plan), and certificate holds the facts that support it. Its status and stats
        attributes report how the solve ended
    """
    return solve_incremental(problem, initial_complexity=INF, complexity_step=INF, max_complexity=INF, **kwargs)
