"""A PyBullet model of a tabletop scene: a Franka arm at the origin, a table just below it, flat trays
(containers) and box-shaped objects. Both the planner's mirror of the scene and the stand-in simulator use it.
"""

from dataclasses import dataclass

import numpy as np

from examples.pybullet.utils.pybullet_tools import utils as pbu

ARM_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]
FINGER_JOINTS = ['panda_finger_joint1', 'panda_finger_joint2']
TOOL_LINK = 'tool_link'
HOME_CONF = (0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8)  # Arm raised in front of the robot
TABLE_SIZE = (1.6, 1.6, 0.02)
TABLE_TOP = -0.01  # Just below the robot's base, which would otherwise touch the table
COLORS = {'movable': pbu.RED, 'container': pbu.GREY, 'surface': pbu.TAN, 'button': pbu.BLUE}


@dataclass(frozen=True)
class ObjectSpec:
    """An object of the scene: a box of size (x, y, z) in meters; kind is 'movable', 'container' (a flat tray objects
    can be put in), 'surface' (fixed, objects can be put on it) or 'button' (fixed, pressed by a skill)."""

    name: str
    size: tuple
    kind: str = 'movable'


def object_spec_from_json(data):
    return ObjectSpec(data['name'], tuple(data['size']), data.get('kind', 'movable'))


class World:
    """The PyBullet bodies of a scene (in this process's PyBullet client)."""

    def __init__(self, specs, use_gui=False):
        self.client = pbu.connect(use_gui=use_gui)
        with pbu.HideOutput():
            self.robot = pbu.load_pybullet(pbu.PANDA_ARM_URDF, fixed_base=True)
        self.arm = pbu.joints_from_names(self.robot, ARM_JOINTS)
        self.fingers = pbu.joints_from_names(self.robot, FINGER_JOINTS)
        self.tool = pbu.link_from_name(self.robot, TOOL_LINK)
        self.table = pbu.create_box(*TABLE_SIZE, color=pbu.TAN)
        pbu.set_point(self.table, (0.0, 0.0, TABLE_TOP - TABLE_SIZE[2] / 2))
        self.specs = {spec.name: spec for spec in specs}
        self.bodies = {spec.name: pbu.create_box(*spec.size, color=COLORS[spec.kind]) for spec in specs}
        self.names = {body: name for name, body in self.bodies.items()}
        self.set_arm(HOME_CONF)
        self.open_gripper()

    @property
    def movables(self):
        return [name for name, spec in self.specs.items() if spec.kind == 'movable']

    @property
    def fixed(self):
        """Bodies that never move: the table and the fixed objects."""
        return [self.table] + [self.bodies[name] for name, spec in self.specs.items() if spec.kind != 'movable']

    def set_arm(self, conf):
        pbu.set_joint_positions(self.robot, self.arm, conf)

    def get_arm(self):
        return tuple(pbu.get_joint_positions(self.robot, self.arm))

    def open_gripper(self):
        pbu.set_joint_positions(self.robot, self.fingers, [0.04, 0.04])

    def tool_pose(self):
        return pbu.get_link_pose(self.robot, self.tool)

    def set_pose(self, name, pose):
        pbu.set_pose(self.bodies[name], pose)

    def get_pose(self, name):
        return pbu.get_pose(self.bodies[name])

    def resting_pose(self, name, point):
        """The pose of an object resting on the table at (x, y)."""
        return (point[0], point[1], TABLE_TOP + self.specs[name].size[2] / 2), pbu.unit_quat()

    def close(self):
        pbu.disconnect()


def quaternion_wxyz(pose):
    x, y, z, w = pose[1]
    return np.array([w, x, y, z])
