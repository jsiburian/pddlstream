import copy
from collections import defaultdict, namedtuple

from pddlstream.algorithms.advanced import UNIVERSAL_TO_CONDITIONAL
from pddlstream.algorithms.scheduling.add_optimizers import (
    add_optimizer_effects,
    recover_simultaneous,
    using_optimizers,
)
from pddlstream.algorithms.scheduling.apply_fluents import convert_fluent_streams
from pddlstream.algorithms.scheduling.negative import convert_negative, recover_negative_axioms
from pddlstream.algorithms.scheduling.postprocess import postprocess_stream_plan
from pddlstream.algorithms.scheduling.recover_axioms import recover_axioms_plans
from pddlstream.algorithms.scheduling.recover_functions import compute_function_plan
from pddlstream.algorithms.scheduling.recover_streams import (
    evaluations_from_stream_plan,
    extract_stream_plan,
    get_achieving_streams,
)
from pddlstream.algorithms.scheduling.stream_action import add_stream_actions
from pddlstream.algorithms.scheduling.utils import add_unsatisfiable_to_goal, get_instance_facts, partition_results
from pddlstream.core.context import get_config
from pddlstream.core.conversion import (
    evaluation_from_fact,
    fact_from_evaluation,
    obj_from_pddl,
    transform_action_args,
    transform_plan_args,
)
from pddlstream.core.problem import FAILED, OptPlan
from pddlstream.core.syntax import EQ, Not, get_prefix
from pddlstream.core.utils import INF, Verbose, get_ancestors
from pddlstream.fastdownward.grounding import instantiate_task, sas_from_instantiated
from pddlstream.fastdownward.search import solve_from_task
from pddlstream.fastdownward.tasks import (
    apply_action,
    conditions_hold,
    evaluation_from_fd,
    fact_from_fd,
    fd_from_fact,
    get_problem,
    make_domain,
    make_predicate,
    pddl_from_instance,
    plan_preimage,
    task_from_domain_problem,
)
from pddlstream.fastdownward.translator import get_cost_scale, scale_cost
from pddlstream.language.exogenous import get_fluent_domain
from pddlstream.language.external import Result
from pddlstream.language.function import Function
from pddlstream.language.optimizer import UNSATISFIABLE
from pddlstream.language.statistics import compute_plan_effort
from pddlstream.language.stream import StreamResult

RENAME_ACTIONS = True

OptSolution = namedtuple('OptSolution', ['stream_plan', 'opt_plan', 'cost'])  # TODO: move to the below

##################################################


def add_stream_efforts(node_from_atom, instantiated, effort_weight, **kwargs):
    if effort_weight is None:
        return
    # TODO: make effort just a multiplier (or relative) to avoid worrying about the scale
    # TODO: regularize & normalize across the problem?
    for instance in instantiated.actions:
        # TODO: prune stream actions here?
        # TODO: round each effort individually to penalize multiple streams
        facts = get_instance_facts(instance, node_from_atom)
        stream_plan = []
        extract_stream_plan(node_from_atom, facts, stream_plan)
        effort = compute_plan_effort(stream_plan, **kwargs)
        instance.cost += scale_cost(effort_weight * effort)
        # TODO: store whether it uses shared/unique outputs and prune too expensive streams


##################################################


def rename_instantiated_actions(instantiated, rename):
    # TODO: rename SAS instead?
    actions = instantiated.actions[:]
    renamed_actions = []
    action_from_name = {}
    for i, action in enumerate(actions):
        renamed_actions.append(copy.copy(action))
        renamed_name = f'a{i}' if rename else action.name
        renamed_actions[-1].name = f'({renamed_name})'
        action_from_name[renamed_name] = action  # Change reachable_action_params?
    instantiated.actions[:] = renamed_actions
    return action_from_name


##################################################


def get_plan_cost(action_plan, cost_from_action):
    if action_plan is None:
        return INF
    # TODO: return cost per action instance
    scaled_cost = sum([0.0] + [cost_from_action[instance] for instance in action_plan])
    return scaled_cost / get_cost_scale()


def instantiate_optimizer_axioms(instantiated, domain, results):
    # Needed for instantiating axioms before adding stream action effects
    # Otherwise, FastDownward will prune these unreachable axioms
    # TODO: compute this first and then apply the eager actions
    stream_init = {fd_from_fact(result.stream_fact) for result in results if isinstance(result, StreamResult)}
    evaluations = list(map(evaluation_from_fd, stream_init | instantiated.atoms))
    temp_domain = make_domain(
        predicates=[make_predicate(UNSATISFIABLE, [])], axioms=[ax for ax in domain.axioms if ax.name == UNSATISFIABLE]
    )
    temp_problem = get_problem(evaluations, Not((UNSATISFIABLE,)), temp_domain)
    # TODO: UNSATISFIABLE might be in atoms making the goal always infeasible
    with Verbose():
        # TODO: the FastDownward instantiation prunes static preconditions
        use_fd = False if using_optimizers(results) else get_config().fd_instantiate
        new_instantiated = instantiate_task(
            task_from_domain_problem(temp_domain, temp_problem),
            use_fd=use_fd,
            check_infeasible=False,
            prune_static=False,
        )
        assert new_instantiated is not None
    instantiated.axioms.extend(new_instantiated.axioms)
    instantiated.atoms.update(new_instantiated.atoms)


##################################################


def recover_partial_orders(stream_plan, node_from_atom):
    # Useful to recover the correct DAG
    partial_orders = set()
    for child in stream_plan:
        # TODO: account for fluent objects
        for fact in child.get_domain():
            parent = node_from_atom[fact].result
            if parent is not None:
                partial_orders.add((parent, child))
    return partial_orders


def recover_stream_plan(
    evaluations,
    current_plan,
    opt_evaluations,
    goal_expression,
    domain,
    node_from_atom,
    action_plan,
    axiom_plans,
    negative,
    replan_step,
):
    """Recovers the stream plan that supports an optimistic action plan, and schedules each of its results either
    now (the returned eager plan) or deferred to the step of the action plan that needs it (in the OptPlan)."""
    # Universally quantified conditions are converted into negative axioms
    # Existentially quantified conditions are made additional preconditions
    # Universally quantified effects are instantiated by doing the cartesian produce of types (slow)
    # Added effects cancel out removed effects
    # TODO: node_from_atom is a subset of opt_evaluations (only missing functions)
    real_task = task_from_domain_problem(domain, get_problem(evaluations, goal_expression, domain))
    opt_task = task_from_domain_problem(domain, get_problem(opt_evaluations, goal_expression, domain))
    negative_from_name = {external.blocked_predicate: external for external in negative if external.is_negated}
    real_states, full_plan = recover_negative_axioms(real_task, opt_task, axiom_plans, action_plan, negative_from_name)
    function_plan = compute_function_plan(opt_evaluations, action_plan)

    full_preimage = plan_preimage(full_plan, [])  # Does not contain the stream preimage!
    negative_preimage = set(filter(lambda a: a.predicate in negative_from_name, full_preimage))
    negative_plan = convert_negative(negative_preimage, negative_from_name, full_preimage, real_states)
    function_plan.update(negative_plan)
    # TODO: OrderedDict for these plans

    steps_from_fact, last_from_fact = get_fact_steps(full_preimage, real_states[0], negative_preimage, function_plan)
    stream_plan, last_from_stream = partition_current_plan(
        current_plan, function_plan, negative, action_plan, replan_step
    )
    stream_plan, node_from_atom = extract_supporting_streams(
        evaluations, domain, node_from_atom, stream_plan, function_plan, last_from_fact, last_from_stream
    )
    evaluate_bound_ancestors(stream_plan, node_from_atom, last_from_stream)

    first_from_stream = get_first_steps(real_task.init, stream_plan, action_plan)
    if first_from_stream:
        replan_step = min(replan_step, *first_from_stream.values())
    eager_plan, results_from_step = schedule_results(stream_plan, first_from_stream, last_from_stream, replan_step)

    # TODO: some sort of obj side-effect bug that requires obj_from_pddl to be applied last
    # (likely due to fluent streams)
    eager_plan = convert_fluent_streams(eager_plan, real_states, action_plan, steps_from_fact, node_from_atom)
    combined_plan = []
    for step, action in enumerate(action_plan):
        combined_plan.extend(result.get_action() for result in results_from_step[step])
        combined_plan.append(transform_action_args(pddl_from_instance(action), obj_from_pddl))
    preimage_facts = get_preimage_facts(full_preimage, negative_plan, eager_plan)
    return eager_plan, OptPlan(combined_plan, preimage_facts)


def get_fact_steps(full_preimage, initial_state, negative_preimage, function_plan):
    """The steps at which the plan uses each positive preimage fact, and the earliest of them (last_from_fact)."""
    # TODO: this assumes that actions do not negate preimage goals
    positive_preimage = {l for l in (set(full_preimage) - initial_state - negative_preimage) if not l.negated}
    steps_from_fact = {fact_from_fd(l): full_preimage[l] for l in positive_preimage}
    last_from_fact = {fact: min(steps) for fact, steps in steps_from_fact.items() if get_prefix(fact) != EQ}
    for result, step in function_plan.items():
        for fact in result.get_domain():
            last_from_fact[fact] = min(step, last_from_fact.get(fact, INF))
    return steps_from_fact, last_from_fact


def partition_current_plan(current_plan, function_plan, negative, action_plan, replan_step):
    """Splits the results already in the plan into streams to schedule and functions and negated streams, which are
    added to function_plan so that they are not pruned."""
    # TODO: get_steps_from_stream
    stream_plan = []
    last_from_stream = dict(function_plan)
    for result in current_plan:  # + negative_plan?
        # TODO: actually compute when these are needed + dependencies
        last_from_stream[result] = 0
        if isinstance(result.external, Function) or (result.external in negative):
            if len(action_plan) > replan_step:
                raise NotImplementedError()  # TODO: deferring negated optimizers
            function_plan[result] = replan_step
        else:
            stream_plan.append(result)
    return stream_plan, last_from_stream


def extract_supporting_streams(
    evaluations, domain, node_from_atom, stream_plan, function_plan, last_from_fact, last_from_stream
):
    """Adds the streams that achieve the facts the plan still needs, and computes the latest step at which each
    result can be evaluated (updating last_from_fact and last_from_stream)."""
    curr_evaluations = evaluations_from_stream_plan(evaluations, stream_plan, max_effort=None)
    extraction_facts = set(last_from_fact) - set(map(fact_from_evaluation, curr_evaluations))
    extract_stream_plan(node_from_atom, extraction_facts, stream_plan)

    # Recomputing due to postprocess_stream_plan
    stream_plan = postprocess_stream_plan(evaluations, domain, stream_plan, last_from_fact)
    node_from_atom = get_achieving_streams(evaluations, stream_plan, max_effort=None)
    fact_sequence = [set(result.get_domain()) for result in stream_plan] + [extraction_facts]
    for facts in reversed(fact_sequence):  # Bellman ford
        for fact in facts:  # could flatten instead
            result = node_from_atom[fact].result
            if result is None:
                continue
            step = last_from_fact[fact] if result.is_deferrable() else 0
            last_from_stream[result] = min(step, last_from_stream.get(result, INF))
            for domain_fact in result.instance.get_domain():
                last_from_fact[domain_fact] = min(last_from_stream[result], last_from_fact.get(domain_fact, INF))
    stream_plan.extend(function_plan)
    return stream_plan, node_from_atom


def evaluate_bound_ancestors(stream_plan, node_from_atom, last_from_stream):
    """Results that cannot be deferred are evaluated now (step 0), and so are their ancestors."""
    partial_orders = recover_partial_orders(stream_plan, node_from_atom)
    bound_objects = set()
    for result in stream_plan:
        if (last_from_stream[result] == 0) or not result.is_deferrable(bound_objects=bound_objects):
            for ancestor in get_ancestors(result, partial_orders) | {result}:
                # TODO: this might change descendants of ancestor. Perform in a while loop.
                last_from_stream[ancestor] = 0
                if isinstance(ancestor, StreamResult):
                    bound_objects.update(out for out in ancestor.output_objects if out.is_unique())


def get_first_steps(initial_state, stream_plan, action_plan):
    """The first step of the action plan at which each result's domain holds, for results not ready initially."""
    # Each stream has an earliest evaluation time
    # When computing the latest, use 0 if something isn't deferred
    # Evaluate each stream as soon as possible
    # Option to defer streams after a point in time?
    # TODO: action costs for streams that encode uncertainty
    state = set(initial_state)
    remaining_results = list(stream_plan)
    first_from_stream = {}
    for step, instance in enumerate(action_plan):
        for result in list(remaining_results):
            # TODO: could do this more efficiently if need be
            domain = result.get_domain() + get_fluent_domain(result)
            if conditions_hold(state, map(fd_from_fact, domain)):
                remaining_results.remove(result)
                certified = {fact for fact in result.get_certified() if get_prefix(fact) != EQ}
                state.update(map(fd_from_fact, certified))
                if step != 0:
                    first_from_stream[result] = step
        # TODO: assumes no fluent axiom domain conditions
        apply_action(state, instance)
    return first_from_stream


def schedule_results(stream_plan, first_from_stream, last_from_stream, replan_step):
    """The results to evaluate now, and the deferred ones by the step before which they run."""
    eager_plan = []
    results_from_step = defaultdict(list)
    for result in stream_plan:
        earliest_step = first_from_stream.get(result, 0)  # exogenous
        latest_step = last_from_stream.get(result, 0)  # defer
        defer = replan_step <= latest_step
        if not defer:
            eager_plan.append(result)
        # We only perform a deferred evaluation if it has all deferred dependencies
        # TODO: make a flag that also allows dependencies to be deferred
        future = (earliest_step != 0) or defer
        if future:
            future_step = latest_step if defer else earliest_step
            results_from_step[future_step].append(result)
    return eager_plan, results_from_step


def get_preimage_facts(full_preimage, negative_plan, eager_plan):
    """The facts that support the plan: its preimage, plus what the negated and eager results use and certify."""
    # TODO: the returned facts have the same side-effect bug as above
    # TODO: annotate when each preimage fact is used
    preimage_facts = {fact_from_fd(l) for l in full_preimage if (l.predicate != EQ) and not l.negated}
    for negative_result in negative_plan:  # TODO: function_plan
        preimage_facts.update(negative_result.get_certified())
    for result in eager_plan:
        preimage_facts.update(result.get_domain())
        # Might not be able to regenerate facts involving the outputs of streams
        preimage_facts.update(result.get_certified())  # Some facts might not be in the preimage
    # TODO: record streams and axioms
    return preimage_facts


##################################################


def solve_optimistic_sequential(
    domain,
    stream_domain,
    applied_results,
    all_results,
    opt_evaluations,
    node_from_atom,
    goal_expression,
    effort_weight,
    debug=False,
    **kwargs,
):
    problem = get_problem(opt_evaluations, goal_expression, stream_domain)  # begin_metric
    with Verbose(verbose=debug):
        task = task_from_domain_problem(stream_domain, problem)
        instantiated = instantiate_task(task)
    if instantiated is None:
        return instantiated, None, INF

    cost_from_action = {action: action.cost for action in instantiated.actions}
    add_stream_efforts(node_from_atom, instantiated, effort_weight)
    if using_optimizers(applied_results):
        add_optimizer_effects(instantiated, node_from_atom)
        # TODO: reachieve=False when using optimizers or should add applied facts
        instantiate_optimizer_axioms(instantiated, domain, all_results)
    action_from_name = rename_instantiated_actions(instantiated, RENAME_ACTIONS)
    # TODO: the action unsatisfiable conditions are pruned
    with Verbose(debug):
        sas_task = sas_from_instantiated(instantiated)
        sas_task.metric = True

    # TODO: apply renaming to hierarchy as well
    # solve_from_task | serialized_solve_from_task | abstrips_solve_from_task | abstrips_solve_from_task_sequential
    renamed_plan, _ = solve_from_task(sas_task, debug=debug, **kwargs)
    if renamed_plan is None:
        return instantiated, None, INF

    action_instances = [
        action_from_name[name if RENAME_ACTIONS else '({} {})'.format(name, ' '.join(args))]
        for name, args in renamed_plan
    ]
    cost = get_plan_cost(action_instances, cost_from_action)
    return instantiated, action_instances, cost


##################################################


def plan_streams(
    evaluations,
    goal_expression,
    domain,
    all_results,
    negative,
    effort_weight,
    max_effort,
    simultaneous=False,
    reachieve=True,
    replan_actions=set(),
    **kwargs,
):
    # TODO: alternatively could translate with stream actions on real opt_state and just discard them
    # TODO: only consider axioms that have stream conditions?
    applied_results, deferred_results = partition_results(
        evaluations, all_results, apply_now=lambda r: not (simultaneous or r.external.info.simultaneous)
    )
    stream_domain, deferred_from_name = add_stream_actions(domain, deferred_results)

    if reachieve and not using_optimizers(all_results):
        achieved_results = {n.result for n in evaluations.values() if isinstance(n.result, Result)}
        init_evaluations = {e for e, n in evaluations.items() if n.result not in achieved_results}
        applied_results = achieved_results | set(applied_results)
        evaluations = init_evaluations  # For clarity

    # TODO: could iteratively increase max_effort
    node_from_atom = get_achieving_streams(
        evaluations,
        applied_results,  # TODO: apply to all_results?
        max_effort=max_effort,
    )
    opt_evaluations = {evaluation_from_fact(f): n.result for f, n in node_from_atom.items()}
    if UNIVERSAL_TO_CONDITIONAL or using_optimizers(all_results):
        goal_expression = add_unsatisfiable_to_goal(stream_domain, goal_expression)

    instantiated, action_instances, cost = solve_optimistic_sequential(
        domain,
        stream_domain,
        applied_results,
        all_results,
        opt_evaluations,
        node_from_atom,
        goal_expression,
        effort_weight,
        **kwargs,
    )
    if action_instances is None:
        return OptSolution(FAILED, FAILED, cost)

    action_instances, axiom_plans = recover_axioms_plans(instantiated, action_instances)
    # TODO: extract out the minimum set of conditional effects that are actually required
    stream_plan, action_instances = recover_simultaneous(
        applied_results, negative, deferred_from_name, action_instances
    )

    action_plan = transform_plan_args(map(pddl_from_instance, action_instances), obj_from_pddl)
    replan_step = min(
        [step + 1 for step, action in enumerate(action_plan) if action.name in replan_actions] or [len(action_plan) + 1]
    )  # step after action application

    stream_plan, opt_plan = recover_stream_plan(
        evaluations,
        stream_plan,
        opt_evaluations,
        goal_expression,
        stream_domain,
        node_from_atom,
        action_instances,
        axiom_plans,
        negative,
        replan_step,
    )
    return OptSolution(stream_plan, opt_plan, cost)
