"""The success conditionals of the stand-in tasks, reimplemented after RoboLab's (robolab.core.task.conditionals)
for box objects: they read the objects' poses and sizes, which object the gripper holds, and which buttons were
pressed (button_pressed is the stand-in's own, for the hybrid task)."""

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


CONDITIONALS = {
    'object_in_container': lambda state, object, container, **kwargs: on_top(state, object, container),
    'object_on_top': lambda state, object, reference_object, **kwargs: on_top(state, object, reference_object),
    'stacked': lambda state, objects, **kwargs: all(on_top(state, a, b) for a, b in zip(objects, objects[1:])),
    'object_picked_up': lambda state, object, **kwargs: (
        (state.held == object) and (state.positions[object][2] > state.initial[object][2] + 0.05)
    ),
    'object_grabbed': lambda state, object, **kwargs: state.held == object,
    'button_pressed': lambda state, button, **kwargs: button in state.pressed,
}


def holds(conditional, state):
    return CONDITIONALS[conditional.func](state, **conditional.params)
