"""The success conditionals of the stand-in tasks, reimplemented after RoboLab's (robolab.core.task.conditionals)
for box objects: they read the objects' poses and sizes, which object the gripper holds, and which buttons were
pressed (button_pressed is the stand-in's own, for the hybrid task)."""

import itertools

import numpy as np

CONTACT = 0.005  # Meters between resting surfaces


def bottom(spec, position):
    return position[2] - spec.size[2] / 2


def top(spec, position):
    return position[2] + spec.size[2] / 2


def within_footprint(position, spec, support_position):
    half = np.array(spec.size[:2]) / 2
    return np.all(np.abs(np.array(position[:2]) - np.array(support_position[:2])) <= half)


def on_top(state, obj, support):
    """Whether obj rests on support (within its footprint) and is not held."""
    if state.held == obj:
        return False
    position, support_position = state.positions[obj], state.positions[support]
    resting = abs(bottom(state.specs[obj], position) - top(state.specs[support], support_position)) <= CONTACT
    return resting and within_footprint(position, state.specs[support], support_position)


def satisfied(params, holds_for):
    """Whether enough of a conditional's objects satisfy it, by its logical mode (all, any, or choose K)."""
    objects = params['object'] if isinstance(params['object'], (list, tuple)) else [params['object']]
    count = sum(map(holds_for, objects))
    needed = {'all': len(objects), 'any': 1}.get(params.get('logical', 'all'), params.get('K', len(objects)))
    return count >= needed


def stacked(state, objects, order=None, **kwargs):
    """Whether the objects form a stack, bottom first with order bottom_to_top, or in any order."""
    orders = [objects] if order == 'bottom_to_top' else itertools.permutations(objects)
    return any(all(on_top(state, top, bottom) for bottom, top in zip(stack, stack[1:])) for stack in orders)


CONDITIONALS = {
    'object_in_container': lambda state, **params: satisfied(params, lambda o: on_top(state, o, params['container'])),
    'object_on_top': lambda state, **params: satisfied(params, lambda o: on_top(state, o, params['reference_object'])),
    'stacked': stacked,
    'object_picked_up': lambda state, object, **kwargs: (
        (state.held == object) and (state.positions[object][2] > state.initial[object][2] + 0.05)
    ),
    'object_grabbed': lambda state, object, **kwargs: state.held == object,
    'button_pressed': lambda state, button, **kwargs: button in state.pressed,
}


def holds(conditional, state):
    return CONDITIONALS[conditional.func](state, **conditional.params)
