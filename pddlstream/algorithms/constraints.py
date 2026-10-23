from collections import namedtuple
from copy import deepcopy
from typing import NamedTuple

from pddlstream.core.evaluations import INTERNAL_EVALUATION, add_fact
from pddlstream.core.objects import Object, OptimisticObject
from pddlstream.core.problem import str_from_plan
from pddlstream.core.syntax import EQ, And, Not, Or, is_parameter
from pddlstream.core.utils import (
    INF,
    find_unique,
    get_ancestors,
    get_descendants,
    is_hashable,
    neighbors_from_orders,
    safe_zip,
    str_from_object,
)
from pddlstream.fastdownward.tasks import add_predicate, fd_from_fact, make_effects, make_preconditions, make_predicate
from pddlstream.fastdownward.translator import pddl


class OrderedSkeleton(NamedTuple):  # TODO: AND/OR tree
    """A plan skeleton whose steps are partially ordered: actions holds (name, args) pairs, where args may contain
    free parameters ('?x') and WILD, and orders holds (before, after) pairs of step indices."""

    actions: list
    orders: set


INTERNAL_PREFIX = '_'  # TODO: possibly apply elsewhere

WILD = '*'  # In a skeleton step, stands for any number of arbitrary arguments
ASSIGNED_PREDICATE = '{}assigned'
BOUND_PREDICATE = '{}bound'  # TODO: switch with assigned
GROUP_PREDICATE = '{}group'
ORDER_PREDICATE = '{}order'

GOAL_INDEX = -1


def linear_order(actions):
    if not actions:
        return set()
    return {(i, i + 1) for i in range(len(actions) - 1)} | {(len(actions) - 1, GOAL_INDEX)}


class PlanConstraints:
    """Constraints on the plans that a solver may return.

    :param skeletons: plan skeletons, each a list of (name, args) steps or an OrderedSkeleton; plans must follow one
    :param groups: sets of values by group name, so that a skeleton argument may name a group instead of a value
    :param exact: plans consist of exactly a skeleton's steps (otherwise other actions may be interleaved)
    :param hint: not used
    :param max_cost: plans must cost less than this
    """

    def __init__(self, skeletons=None, groups={}, exact=True, hint=False, max_cost=INF):
        # TODO: constraint that the skeleton is the tail of the plan
        if skeletons is not None:
            skeletons = [
                skeleton if isinstance(skeleton, OrderedSkeleton) else OrderedSkeleton(skeleton, linear_order(skeleton))
                for skeleton in skeletons
            ]
        self.skeletons = skeletons
        self.groups = groups  # Could make this a list of lists
        self.exact = exact
        self.max_cost = max_cost

    def dump(self):
        print(f'{self.__class__.__name__}(exact={self.exact}, max_cost={self.max_cost})')
        if self.skeletons is None:
            return
        for i, skeleton in enumerate(self.skeletons):
            print(i, str_from_plan(skeleton))

    def __repr__(self):
        return f'{self.__class__.__name__}{str_from_object(self.__dict__)}'


# TODO: rename other costs to be terminate_cost (or decision cost)


def to_constant(parameter):
    name = parameter[1:]
    return to_obj(f'@{name}')


def to_obj(value):
    # Allows both raw values as well as objects to be specified
    if any(isinstance(value, Class) for Class in [Object, OptimisticObject]):
        return value
    return Object.from_value(value)


def get_internal_prefix(domain, internal):
    # Lengthens the prefix until the constraint predicates don't collide with the domain's (or an outer layer's)
    names = {predicate.name for predicate in domain.predicates}
    prefix = INTERNAL_PREFIX if internal else ''
    while any(
        template.format(prefix) in names
        for template in [ASSIGNED_PREDICATE, BOUND_PREDICATE, GROUP_PREDICATE, ORDER_PREDICATE]
    ):
        prefix += INTERNAL_PREFIX
    return prefix


def is_constant(arg):
    # Only strings can be parameters or the wildcard; other values (e.g. numpy arrays) are always constants
    return not (isinstance(arg, str) and (is_parameter(arg) or arg == WILD))


##################################################


def add_plan_constraints(constraints, domain, evaluations, goal_exp, internal=False):
    """Compiles the plan skeletons in constraints into the domain: each skeleton step becomes a copy of its action
    that binds the step's free parameters and follows the skeleton's order. Returns the new goal."""
    if (constraints is None) or (constraints.skeletons is None):
        return goal_exp

    # TODO: unify this with the constraint ordering
    # TODO: can constrain to use a plan prefix
    prefix = get_internal_prefix(domain, internal)
    predicates = SkeletonPredicates(
        assigned=ASSIGNED_PREDICATE.format(prefix),
        bound=BOUND_PREDICATE.format(prefix),
        group=GROUP_PREDICATE.format(prefix),
        order=ORDER_PREDICATE.format(prefix),
    )
    order_predicate = predicates.order
    new_facts = []
    for group in constraints.groups:
        for value in constraints.groups[group]:
            # TODO: could make all constants groups (like an equality group)
            fact = (predicates.group, to_obj(group), to_obj(value))
            new_facts.append(fact)
    new_actions = []
    new_goals = []
    for num, skeleton in enumerate(constraints.skeletons):
        actions, orders = skeleton
        incoming_orders, _ = neighbors_from_orders(orders)
        order_facts = [(order_predicate, to_obj(f'n{num}'), to_obj(f't{step}')) for step in range(len(actions))]
        for step in range(len(actions)):
            new_actions.append(
                constrain_step(
                    domain, constraints.groups, predicates, actions, orders, incoming_orders, order_facts, step
                )
            )
        new_goals.append(And(*[order_facts[idx] for idx in incoming_orders[GOAL_INDEX]]))

    add_predicate(domain, make_predicate(order_predicate, ['?num', '?step']))
    if constraints.exact:
        domain.actions[:] = []
    domain.actions.extend(new_actions)
    new_goal_exp = And(goal_exp, Or(*new_goals))
    for fact in new_facts:
        add_fact(evaluations, fact, result=INTERNAL_EVALUATION)
    return new_goal_exp


SkeletonPredicates = namedtuple('SkeletonPredicates', ['assigned', 'bound', 'group', 'order'])


def constrain_step(domain, groups, predicates, actions, orders, incoming_orders, order_facts, step):
    """A copy of the action at this skeleton step whose preconditions require the step's constants, the bindings of
    parameters that earlier steps bound, and the skeleton's order, and whose effects record its own bindings."""
    name, args = actions[step]
    # TODO: could also just remove the free parameter from the action
    new_action = deepcopy(find_unique(lambda a: a.name == name, domain.actions))
    local_from_global = {a: p.name for a, p in safe_zip(args, new_action.parameters) if is_parameter(a)}

    ancestors, descendants = get_ancestors(step, orders), get_descendants(step, orders)
    parallel = set(range(len(actions))) - ancestors - descendants - {step}

    parameters = set(filter(is_parameter, args))
    ancestor_parameters = parameters & set(filter(is_parameter, (p for idx in ancestors for p in actions[idx][1])))
    parallel_parameters = parameters & set(filter(is_parameter, (p for idx in parallel for p in actions[idx][1])))

    bound_condition = pddl.Conjunction(
        [
            pddl.Disjunction(
                map(
                    fd_from_fact,
                    [
                        Not((predicates.bound, to_constant(p))),
                        (predicates.assigned, to_constant(p), local_from_global[p]),
                    ],
                )
            )
            for p in parallel_parameters
        ]
    )
    existing_preconditions = [(predicates.assigned, to_constant(p), local_from_global[p]) for p in ancestor_parameters]

    constant_pairs = [(a, p.name) for a, p in safe_zip(args, new_action.parameters) if is_constant(a)]
    group_preconditions = [
        (predicates.group if is_hashable(a) and (a in groups) else EQ, to_obj(a), p) for a, p in constant_pairs
    ]
    order_preconditions = [order_facts[idx] for idx in incoming_orders[step]]
    new_preconditions = existing_preconditions + group_preconditions + order_preconditions + [Not(order_facts[step])]
    new_action.precondition = pddl.Conjunction(
        [new_action.precondition, bound_condition, make_preconditions(new_preconditions)]
    ).simplified()

    # TODO: ancestors holds step indices, so this removes nothing (parameters - ancestor_parameters was probably
    #  intended). It is harmless: re-asserting an earlier step's bindings changes nothing.
    new_parameters = parameters - ancestors
    bound_facts = [(predicates.bound, to_constant(p)) for p in new_parameters]
    assigned_facts = [(predicates.assigned, to_constant(p), local_from_global[p]) for p in new_parameters]
    new_effects = bound_facts + assigned_facts + [order_facts[step]]
    new_action.effects.extend(make_effects(new_effects))
    # TODO: should also negate the effects of all other sequences here
    return new_action
