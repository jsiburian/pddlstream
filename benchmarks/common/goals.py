"""PDDL goals for RoboLab tasks, from their success conditionals.

Only conditionals that the planning domain models are supported. A task with any other conditional raises
UnsupportedTask, so it is reported as skipped instead of being attempted with a weaker goal.
"""

from pddlstream.core.syntax import And


class UnsupportedTask(ValueError):
    """A task whose success conditionals the domain cannot express."""


def stacked_goal(params):
    objects = list(params['objects'])  # From the top of the stack to its base
    return [('On', top, bottom) for top, bottom in zip(objects, objects[1:])]


GOAL_FROM_CONDITIONAL = {
    'object_in_container': lambda params: [('On', params['object'], params['container'])],
    'object_on_top': lambda params: [('On', params['object'], params['reference_object'])],
    'stacked': stacked_goal,
    'object_picked_up': lambda params: [('Holding', params['object'])],
    'object_grabbed': lambda params: [('Holding', params['object'])],
}


def goal_from_task(task, skills=()):
    """The goal formula that achieves a TaskSpec's success conditionals, with the domain and the given skills."""
    goal_from_conditional = dict(GOAL_FROM_CONDITIONAL)
    for skill in skills:
        goal_from_conditional.update(skill.goals)
    literals = []
    for conditional in task.success:
        if conditional.func not in goal_from_conditional:
            raise UnsupportedTask(f'{task.name}: conditional {conditional.func} is not supported')
        try:
            literals.extend(goal_from_conditional[conditional.func](conditional.params))
        except KeyError as error:
            raise UnsupportedTask(f'{task.name}: {conditional.func} is missing parameter {error}') from error
    return And(*literals)


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
