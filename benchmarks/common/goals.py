"""PDDL goals from success conditionals (RoboLab's names and parameters).

Only conditionals that the planning domain models are supported. A task with any other conditional raises
UnsupportedTask, so it is reported as unsupported instead of being attempted with a weaker goal.

A conditional's `object` may be one name or a list, with `logical` saying how many must satisfy it: `all`, `any`, or
`choose` (at least `K`). `any` and `choose` become a disjunction over the possible selections. When there are more
than MAX_ALTERNATIVES selections, the first one is used: any selection satisfies the conditional, but fewer choices
leave the planner less freedom.
"""

import itertools

from pddlstream.core.syntax import AND, OR, And, Or

MAX_ALTERNATIVES = 10


class UnsupportedTask(ValueError):
    """A task whose success conditionals the domain cannot express."""


def names(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def selections(objects, params):
    """The sets of objects whose satisfying the conditional satisfies it, by its logical mode."""
    logical = params.get('logical', 'all')
    if logical == 'all':
        return [objects]
    if logical in ('any', 'choose'):
        k = 1 if logical == 'any' else int(params['K'])
        return [list(selection) for selection in itertools.combinations(objects, k)]
    raise UnsupportedTask(f'logical mode {logical!r}')


def disjunction(alternatives):
    """A formula for any one of the alternatives (lists of literals), or for the first if there are too many."""
    if len(alternatives) > MAX_ALTERNATIVES:
        alternatives = alternatives[:1]
    return Or(*(And(*literals) for literals in alternatives))


def placed(support_key):
    def goal(params):
        support = params[support_key]
        return disjunction(
            [[('On', obj, support) for obj in selection] for selection in selections(names(params['object']), params)]
        )

    return goal


def stacked_goal(params):
    objects = names(params['objects'])
    order = params.get('order')
    if order in ('bottom_to_top', None, 'None'):  # Without an order, any stacking satisfies it
        orders = [objects] if order == 'bottom_to_top' else [list(p) for p in itertools.permutations(objects)]
    elif order == 'top_to_bottom':
        orders = [objects[::-1]]
    else:
        raise UnsupportedTask(f'stacking order {order!r}')
    return disjunction([[('On', top, bottom) for bottom, top in zip(stack, stack[1:])] for stack in orders])


GOAL_FROM_CONDITIONAL = {
    'object_in_container': placed('container'),
    'object_on_top': placed('reference_object'),
    'stacked': stacked_goal,
    'object_picked_up': lambda params: And(*(('Holding', obj) for obj in names(params['object']))),
    'object_grabbed': lambda params: And(*(('Holding', obj) for obj in names(params['object']))),
}


def goal_from_task(task, skills=()):
    """The goal formula that achieves a TaskSpec's success conditionals, with the domain and the given skills."""
    goal_from_conditional = dict(GOAL_FROM_CONDITIONAL)
    for skill in skills:
        goal_from_conditional.update(skill.goals)
    parts = []
    for conditional in task.success:
        if conditional.func not in goal_from_conditional:
            raise UnsupportedTask(f'{task.name}: conditional {conditional.func} is not supported')
        try:
            part = goal_from_conditional[conditional.func](conditional.params)
        except KeyError as error:
            raise UnsupportedTask(f'{task.name}: {conditional.func} is missing parameter {error}') from error
        except UnsupportedTask as error:
            raise UnsupportedTask(f'{task.name}: {conditional.func} with {error}') from error
        parts.extend(part if isinstance(part, list) else [part])  # A skill may give a list of literals
    return And(*parts)


def goal_literals(goal):
    """Every literal in a goal formula."""
    if goal[0] in (AND, OR):
        return [literal for part in goal[1:] for literal in goal_literals(part)]
    return [goal]


def coverage(tasks, skills=()):
    """Which tasks are supported: (supported task names, {task name: reason} for the rest)."""
    supported, skipped = [], {}
    for task in tasks:
        try:
            goal_from_task(task, skills)
            supported.append(task.name)
        except UnsupportedTask as error:
            skipped[task.name] = str(error)
    return supported, skipped
