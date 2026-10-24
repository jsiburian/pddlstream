"""A stand-in for RoboLab that runs without a GPU: a kinematic PyBullet tabletop with a Franka arm that takes the same
joint-position actions and gives the same kind of observations (with ground-truth object poses).

When the gripper closes near an object, the object follows the gripper until it opens; nothing falls. A button is
pressed when the tool point reaches its top. Eight tasks mirror RoboLab task families (pick and place, containers,
stacking, rearrangement); a ninth needs a step the planning domain does not model (pressing a button), which a
learned skill performs.
"""

from dataclasses import dataclass, field

import numpy as np

from benchmarks.common.protocol import Conditional, ObjectState, Observation, TaskSpec, condition_key
from benchmarks.common.world import HOME_CONF, ObjectSpec, World, pbu, quaternion_wxyz
from benchmarks.standin.conditionals import holds

GRASP_DISTANCE = 0.05  # Meters from the gripper's tool point to an object's center for the gripper to take it
PRESS_DEPTH = 0.01  # Meters above a button's top that the tool point must reach to press it
BLOCK = (0.05, 0.05, 0.05)


@dataclass
class State:
    """What the conditionals read: object positions, sizes, initial positions and the held object."""

    positions: dict
    specs: dict
    initial: dict
    held: str | None = None
    pressed: frozenset = frozenset()


@dataclass(frozen=True)
class Scenario:
    """A stand-in task: the task, its objects and their initial positions: (x, y) on the table, or (x, y, support) on
    an object listed before."""

    task: TaskSpec
    objects: tuple
    positions: dict = field(default_factory=dict)
    skills: tuple = ()  # Names of the skills (benchmarks.common.skills) the planner may use
    slippery: tuple = ()  # Objects that slip out of the gripper's first grasp of them


class StandinEnv:
    def __init__(self, scenario, use_gui=False):
        self.scenario = scenario
        self.world = World(scenario.objects, use_gui=use_gui)
        for name, (x, y, *support) in scenario.positions.items():  # Optionally resting on another object
            pose = self.world.resting_pose(name, (x, y))
            if support:
                height = self.world.get_pose(support[0])[0][2] + self.world.specs[support[0]].size[2] / 2
                pose = ((x, y, height + self.world.specs[name].size[2] / 2 + 1e-3), pose[1])
            self.world.set_pose(name, pose)
        self.initial = {name: self.world.get_pose(name)[0] for name in self.world.bodies}
        self.held, self.grasp, self.gripper = None, None, 0.0
        self.pressed = set()
        self.slips = set(self.scenario.slippery)
        self.world.set_arm(HOME_CONF)

    def observe(self):
        objects = {
            name: ObjectState(np.array(self.world.get_pose(name)[0]), quaternion_wxyz(self.world.get_pose(name)))
            for name in self.world.bodies
        }
        state = self.state()
        conditions = {condition_key(c): holds(c, state) for c in self.scenario.task.success}
        return Observation(np.array(self.world.get_arm()), self.gripper, objects, conditions)

    def step(self, action):
        self.world.set_arm(action[:7])
        gripper = float(action[7])
        tool_pose = self.world.tool_pose()
        if (gripper > 0.5) and (self.gripper <= 0.5) and (self.held is None):
            self.take(tool_pose)
        elif (gripper <= 0.5) and (self.held is not None):
            self.held, self.grasp = None, None
        self.gripper = gripper
        if self.held is not None:
            self.world.set_pose(self.held, pbu.multiply(tool_pose, self.grasp))
        self.press(tool_pose)
        return self.observe()

    def press(self, tool_pose):
        """Presses the buttons whose top the tool point reaches."""
        for name, spec in self.world.specs.items():
            if spec.kind == 'button':
                position = self.world.get_pose(name)[0]
                above = tool_pose[0][2] - (position[2] + spec.size[2] / 2)
                inside = np.all(
                    np.abs(np.array(tool_pose[0][:2]) - np.array(position[:2])) <= np.array(spec.size[:2]) / 2
                )
                if inside and (above <= PRESS_DEPTH):
                    self.pressed.add(name)

    def take(self, tool_pose):
        distances = {
            name: np.linalg.norm(np.array(self.world.get_pose(name)[0]) - np.array(tool_pose[0]))
            for name in self.world.movables
        }
        if distances:
            name = min(distances, key=distances.get)
            if name in self.slips:  # It slips out
                self.slips.discard(name)
            elif distances[name] <= GRASP_DISTANCE:
                self.held = name
                self.grasp = pbu.multiply(pbu.invert(tool_pose), self.world.get_pose(name))

    def move(self, name, point):
        """Moves an object, as a person would, to test replanning."""
        self.world.set_pose(name, self.world.resting_pose(name, point))

    def state(self):
        positions = {name: self.world.get_pose(name)[0] for name in self.world.bodies}
        return State(positions, self.world.specs, self.initial, self.held, frozenset(self.pressed))

    def success(self):
        return all(holds(conditional, self.state()) for conditional in self.scenario.task.success)

    def close(self):
        self.world.close()


def task(name, instruction, *success):
    return TaskSpec(name, {'default': instruction}, tuple(success))


def block(name):
    return ObjectSpec(name, BLOCK)


def tray(name, size=0.2):
    return ObjectSpec(name, (size, size, 0.01), kind='container')


def button(name):
    return ObjectSpec(name, (0.06, 0.06, 0.02), kind='button')


SCENARIOS = [
    Scenario(
        task(
            'BlockInTray',
            'Put the block in the tray',
            Conditional('object_in_container', {'object': 'a', 'container': 'tray'}),
        ),
        (block('a'), tray('tray')),
        {'a': (0.5, -0.15), 'tray': (0.45, 0.2)},
    ),
    Scenario(
        task(
            'TwoBlocksInTray',
            'Put both blocks in the tray',
            Conditional('object_in_container', {'object': 'a', 'container': 'tray'}),
            Conditional('object_in_container', {'object': 'b', 'container': 'tray'}),
        ),
        (block('a'), block('b'), tray('tray', 0.25)),
        {'a': (0.5, -0.2), 'b': (0.4, -0.1), 'tray': (0.45, 0.2)},
    ),
    Scenario(
        task(
            'BlockOnBlock',
            'Put block a on block b',
            Conditional('object_on_top', {'object': 'a', 'reference_object': 'b'}),
        ),
        (block('a'), block('b')),
        {'a': (0.5, -0.15), 'b': (0.45, 0.15)},
    ),
    Scenario(
        task(
            'StackThree',
            'Stack block a on block b on block c',
            Conditional('stacked', {'objects': ['c', 'b', 'a'], 'order': 'bottom_to_top'}),
        ),
        (block('a'), block('b'), block('c')),
        {'a': (0.5, -0.2), 'b': (0.4, -0.05), 'c': (0.45, 0.2)},
    ),
    Scenario(
        task('PickUp', 'Pick up the block', Conditional('object_picked_up', {'object': 'a'})),
        (block('a'),),
        {'a': (0.5, 0.0)},
    ),
    Scenario(
        task(
            'ClearTheTray',
            'Put block a in the small tray',
            Conditional('object_in_container', {'object': 'a', 'container': 'tray'}),
        ),
        (block('a'), block('b'), tray('tray', 0.08)),
        {'a': (0.5, -0.15), 'tray': (0.45, 0.2), 'b': (0.45, 0.2, 'tray')},
    ),
    Scenario(
        task(
            'TwoTrays',
            'Put block a in the left tray and block b in the right tray',
            Conditional('object_in_container', {'object': 'a', 'container': 'left'}),
            Conditional('object_in_container', {'object': 'b', 'container': 'right'}),
        ),
        (block('a'), block('b'), tray('left', 0.15), tray('right', 0.15)),
        {'a': (0.55, -0.05), 'b': (0.55, 0.05), 'left': (0.4, 0.3), 'right': (0.4, -0.3)},
    ),
    Scenario(
        task(
            'FarTray',
            'Put the block in the far tray',
            Conditional('object_in_container', {'object': 'a', 'container': 'tray'}),
        ),
        (block('a'), tray('tray', 0.15)),
        {'a': (0.35, -0.4), 'tray': (0.35, 0.45)},
    ),
    Scenario(  # The domain cannot press buttons: a learned skill does (the hybrid task)
        task(
            'PressThenPlace',
            'Press the button, then put the block in the tray',
            Conditional('button_pressed', {'button': 'button'}),
            Conditional('object_in_container', {'object': 'a', 'container': 'tray'}),
        ),
        (block('a'), tray('tray'), button('button')),
        {'a': (0.5, -0.15), 'tray': (0.45, 0.2), 'button': (0.4, -0.35)},
        skills=('press',),
    ),
    Scenario(  # The planned grasp slips: a VLA that the client runs picks the block up instead (the second hybrid task)
        task(
            'SlipperyBlock',
            'Put the slippery block in the tray',
            Conditional('object_in_container', {'object': 'a', 'container': 'tray'}),
        ),
        (block('a'), tray('tray')),
        {'a': (0.5, -0.15), 'tray': (0.45, 0.2)},
        skills=('vla_pick',),
        slippery=('a',),
    ),
]
