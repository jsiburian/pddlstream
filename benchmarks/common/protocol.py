"""The data that passes between RoboLab (or the stand-in) and the planner: task specifications and observations.

Everything here is plain data (numpy arrays at most), so it can be sent between processes as JSON or msgpack.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Conditional:
    """A success condition of a task: a RoboLab conditional function name and its parameters, e.g.
    Conditional('object_in_container', {'object': 'apple', 'container': 'bowl'})."""

    func: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """A task: its name, instruction(s), the conditions under which it succeeds, and the objects it mentions."""

    name: str
    instruction: dict  # e.g. {'default': 'Put the apple in the bowl'}
    success: tuple  # Conditionals that must all hold
    objects: tuple = ()  # Names of the task's objects (RoboLab's contact_object_list)


@dataclass(frozen=True)
class ObjectSpec:
    """An object of the scene: a box of size (x, y, z) in meters; kind is 'movable', 'container' (a flat tray objects
    can be put in), 'surface' (fixed, objects can be put on it), 'button' (fixed, pressed by a skill) or 'obstacle'
    (fixed, and nothing is put on it). points, when
    given, are its surface in its own frame ((x, y, z) in meters), and the planner uses their shape instead of the
    box."""

    name: str
    size: tuple
    kind: str = 'movable'
    points: tuple = ()


def object_spec_from_json(data):
    points = tuple(tuple(point) for point in data.get('points', ()))
    return ObjectSpec(data['name'], tuple(data['size']), data.get('kind', 'movable'), points)


@dataclass(frozen=True)
class ObjectState:
    """An object's pose: position (meters, env-local frame) and quaternion (w, x, y, z), as in RoboLab's gt_state."""

    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class Observation:
    """What the planner sees: the arm's joint positions, the gripper's opening and the objects' poses, plus which of
    the task's success conditionals hold (by condition_key), when the simulator reports them (RoboLab's subtask
    status). A skill's success is confirmed by its conditional."""

    arm: np.ndarray  # 7 joint positions
    gripper: float  # 0 open, 1 closed
    objects: dict  # ObjectState by name
    conditions: dict = field(default_factory=dict)  # bool by condition_key


def condition_key(conditional):
    """A conditional as a string, e.g. 'button_pressed(button=red)'."""
    params = ','.join(f'{key}={value}' for key, value in sorted(conditional.params.items()))
    return f'{conditional.func}({params})'


def wxyz_to_xyzw(quaternion):
    w, x, y, z = quaternion
    return np.array([x, y, z, w])


def pose_from_state(state):
    """A PyBullet pose (point, xyzw quaternion) from an ObjectState."""
    return tuple(state.position), tuple(wxyz_to_xyzw(state.quaternion))


def to_json(value):
    """value with numpy arrays as lists and dataclasses as dicts, for sending between processes."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if hasattr(value, '__dataclass_fields__'):
        return {name: to_json(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    return value


def observation_from_json(data):
    objects = {
        name: ObjectState(np.array(state['position']), np.array(state['quaternion']))
        for name, state in data['objects'].items()
    }
    return Observation(np.array(data['arm']), float(data['gripper']), objects, dict(data.get('conditions', {})))


def task_from_json(data):
    success = tuple(Conditional(c['func'], c['params']) for c in data['success'])
    return TaskSpec(data['name'], data['instruction'], success, tuple(data.get('objects', ())))
