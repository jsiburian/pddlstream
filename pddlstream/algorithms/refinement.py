import logging
import time
from itertools import product

from pddlstream.algorithms.instantiation import Instantiator
from pddlstream.algorithms.scheduling.plan_streams import OptSolution, plan_streams
from pddlstream.core.context import note_gap
from pddlstream.core.conversion import evaluation_from_fact, substitute_expression
from pddlstream.core.problem import FAILED, INFEASIBLE, is_plan
from pddlstream.core.utils import INF, elapsed_time, get_mapping, safe_zip
from pddlstream.language.function import Function, FunctionResult
from pddlstream.language.statistics import check_effort
from pddlstream.language.stream import StreamResult

logger = logging.getLogger(__name__)


def is_refined(stream_plan):
    # TODO: lazily expand the shared objects in some cases to prevent increase in size
    if not is_plan(stream_plan):
        return True
    # TODO: some of these opt_index equal None
    return all((result.opt_index is None) or result.is_refined() for result in stream_plan)


##################################################


def optimistic_process_instance(instantiator, instance, verbose=False):
    for result in instance.next_optimistic():
        if verbose:
            logger.info(result)  # TODO: make a debug tools that reports the optimistic streams
        new_facts = False
        complexity = instantiator.compute_complexity(instance)
        for fact in result.get_certified():
            new_facts |= instantiator.add_atom(evaluation_from_fact(fact), complexity)
        if isinstance(result, FunctionResult) or new_facts:
            yield result


def prune_high_effort_streams(streams, max_effort=INF, **effort_args):
    # TODO: convert streams to test streams with extremely high effort
    low_effort_streams = []
    for stream in streams:
        effort = stream.get_effort(**effort_args)
        if isinstance(stream, Function) or check_effort(effort, max_effort):
            low_effort_streams.append(stream)
        else:
            note_gap('max_effort')
    return low_effort_streams


def optimistic_process_streams(evaluations, streams, complexity_limit=INF, **effort_args):
    optimistic_streams = prune_high_effort_streams(streams, **effort_args)
    instantiator = Instantiator(optimistic_streams)
    for evaluation, node in evaluations.items():
        if node.complexity <= complexity_limit:
            instantiator.add_atom(evaluation, node.complexity)
    results = []
    while instantiator and (instantiator.min_complexity() <= complexity_limit):
        results.extend(optimistic_process_instance(instantiator, instantiator.pop_stream()))
        # TODO: instantiate and solve to avoid repeated work
    exhausted = not instantiator
    return results, exhausted


##################################################


def optimistic_stream_instantiation(instance, bindings, opt_evaluations, only_immediate=False):
    # TODO: combination for domain predicates
    new_instances = []
    input_candidates = [bindings.get(i, [i]) for i in instance.input_objects]
    if only_immediate and not all(len(candidates) == 1 for candidates in input_candidates):
        return new_instances
    for input_combo in product(*input_candidates):
        mapping = get_mapping(instance.input_objects, input_combo)
        domain_evaluations = set(
            map(evaluation_from_fact, substitute_expression(instance.get_domain(), mapping))
        )  # TODO: could just instantiate first
        if domain_evaluations <= opt_evaluations:
            new_instance = instance.external.get_instance(input_combo)
            # TODO: method for eagerly evaluating some of these?
            if not new_instance.is_refined():
                new_instance.refine()
            new_instances.append(new_instance)
    return new_instances


def optimistic_stream_evaluation(evaluations, stream_plan, use_bindings=True):
    # TODO: can also use the instantiator and operate directly on the outputs
    # TODO: could bind by just using new_evaluations
    evaluations = set(evaluations)  # Converts to a set for subset testing
    opt_evaluations = set(evaluations)
    new_results = []
    bindings = {}  # TODO: report the depth considered
    for opt_result in stream_plan:  # TODO: just refine the first step of the plan
        if not opt_result.optimistic:  # Real outputs stay as they are
            continue
        for new_instance in optimistic_stream_instantiation(
            opt_result.instance, (bindings if use_bindings else {}), opt_evaluations
        ):
            for new_result in new_instance.next_optimistic():
                opt_evaluations.update(map(evaluation_from_fact, new_result.get_certified()))
                new_results.append(new_result)
                if isinstance(new_result, StreamResult):  # Could not add if same value
                    for opt, obj in safe_zip(opt_result.output_objects, new_result.output_objects):
                        bindings.setdefault(opt, []).append(obj)
    return new_results, bindings


##################################################


def get_optimistic_solve_fn(goal_exp, domain, negative, max_cost=INF, **kwargs):
    # TODO: apply to hierarchical actions representations (will need to instantiate more actions)
    def fn(evaluations, results):
        return plan_streams(evaluations, goal_exp, domain, results, negative, max_cost=max_cost, **kwargs)

    return fn


##################################################


def hierarchical_plan_streams(evaluations, results, optimistic_solve_fn):
    """Plans with the optimistic results. If the stream plan still uses unrefined optimistic outputs, refines those
    instances and returns FAILED with depth 1, so the caller plans again with the refined outputs."""
    stream_plan, opt_plan, cost = optimistic_solve_fn(evaluations, results)
    if not is_plan(opt_plan) or is_refined(stream_plan):
        return OptSolution(stream_plan, opt_plan, cost), 0
    # TODO: identify control parameters that can be separated across actions
    optimistic_stream_evaluation(evaluations, stream_plan)
    return OptSolution(FAILED, FAILED, INF), 1


def iterative_plan_streams(all_evaluations, externals, optimistic_solve_fn, complexity_limit, **effort_args):
    # Previously didn't have unique optimistic objects that could be constructed at arbitrary depths
    start_time = time.time()
    complexity_evals = {e: n for e, n in all_evaluations.items() if n.complexity <= complexity_limit}
    num_iterations = 0
    while True:
        num_iterations += 1
        results, exhausted = optimistic_process_streams(complexity_evals, externals, complexity_limit, **effort_args)
        opt_solution, final_depth = hierarchical_plan_streams(complexity_evals, results, optimistic_solve_fn)
        stream_plan, action_plan, cost = opt_solution
        logger.info(
            f'Attempt: {num_iterations} | Results: {len(results)} | Depth: {final_depth} | '
            f'Success: {is_plan(action_plan)} | Time: {elapsed_time(start_time):.3f}'
        )
        if is_plan(action_plan):
            return OptSolution(stream_plan, action_plan, cost)
        if final_depth == 0:
            status = INFEASIBLE if exhausted else FAILED
            return OptSolution(status, status, cost)
    # TODO: should streams along the sampled path automatically have no optimistic value
