"""Helpers that give the actions of a sequential plan start times and durations (e.g. to animate the plan)."""

from pddlstream.language.constants import DurativeAction


def get_end(action):
    return action.start + action.duration


def compute_start(plan):
    if not plan:
        return 0.0
    return min(action.start for action in plan)


def compute_end(plan):
    if not plan:
        return 0.0
    return max(map(get_end, plan))


def compute_duration(plan):
    return compute_end(plan) - compute_start(plan)


def retime_plan(plan, duration=1):
    if plan is None:
        return plan
    # TODO: duration per action
    return [DurativeAction(name, args, i * duration, duration) for i, (name, args) in enumerate(plan)]
