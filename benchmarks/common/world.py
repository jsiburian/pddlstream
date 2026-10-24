"""A PyBullet model of a tabletop scene: a Franka arm at the origin, a table just below it, flat trays
(containers) and box-shaped objects. Both the planner's mirror of the scene and the stand-in simulator use it.
"""

from dataclasses import dataclass

import numpy as np

from benchmarks.common.protocol import ObjectSpec, object_spec_from_json  # noqa: F401 (part of the world's interface)
from examples.pybullet.utils.pybullet_tools import utils as pbu

ARM_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]
FINGER_JOINTS = ['panda_finger_joint1', 'panda_finger_joint2']
HOME_CONF = (0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8)  # Arm raised in front of the robot
TABLE_SIZE = (1.6, 1.6, 0.02)
TABLE_TOP = -0.01  # Just below the robot's base, which would otherwise touch the table
COLORS = {'movable': pbu.RED, 'container': pbu.GREY, 'surface': pbu.TAN, 'button': pbu.BLUE}


@dataclass(frozen=True)
class Gripper:
    """Where the gripper's tool frame is: a pose in one of the arm's links. The tool approaches along its z axis and
    the fingers close along its y axis (pybullet_tools' top grasps); max_width is the widest object they close on.
    At a grasp or placement, the arm holds still for arrive_steps actions before the fingers move (so that a lagging
    controller reaches the pose first), then for settle_steps actions while they open or close."""

    link: str
    offset: tuple = pbu.unit_pose()
    max_width: float = pbu.INF
    settle_steps: int = 5
    arrive_steps: int = 0


GRIPPERS = {
    'panda': Gripper('tool_link'),  # The Panda hand of the URDF
    # RoboLab's DROID Franka: a Robotiq 2F-85 mounted 18.1 mm past the flange (panda_link8) without the Panda hand's
    # 45 degree turn (live probe), its fingertips 162.8 mm further; the tool frame is 1 cm above the fingertips. In
    # Isaac Sim the arm's joint controller lags 1-2 cm behind a descent, and the gripper takes about a second (15
    # actions) to close on an object
    'robotiq': Gripper(
        'panda_link8', pbu.Pose(point=pbu.Point(z=0.0181 + 0.1628 - 0.01)), 0.085, settle_steps=15, arrive_steps=10
    ),
}


def create_body(spec):
    """A static PyBullet body for an object: its box, or the convex hull of its points."""
    if not spec.points:
        return pbu.create_box(*spec.size, color=COLORS[spec.kind])
    collision = pbu.p.createCollisionShape(pbu.p.GEOM_MESH, vertices=list(spec.points), physicsClientId=pbu.CLIENT)
    return pbu.create_body(collision)  # Without a visual shape: the mirror has no viewer


class World:
    """The PyBullet bodies of a scene (in this process's PyBullet client)."""

    def __init__(self, specs, use_gui=False, gripper='panda', containment='full'):
        """containment: how much of an object must be inside a container's footprint to count as in it, 'full' or
        'center' (its center, as RoboLab's object_in_container checks)."""
        self.client = pbu.connect(use_gui=use_gui)
        with pbu.HideOutput():
            self.robot = pbu.load_pybullet(pbu.PANDA_ARM_URDF, fixed_base=True)
        self.arm = pbu.joints_from_names(self.robot, ARM_JOINTS)
        self.fingers = pbu.joints_from_names(self.robot, FINGER_JOINTS)
        self.gripper = GRIPPERS[gripper]
        self.containment = containment
        self.tool = pbu.link_from_name(self.robot, self.gripper.link)  # The link the tool frame is fixed to
        self.table = pbu.create_box(*TABLE_SIZE, color=pbu.TAN)
        self.set_table_top(TABLE_TOP)
        self.specs = {spec.name: spec for spec in specs}
        self.bodies = {spec.name: create_body(spec) for spec in specs}
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
        return pbu.multiply(pbu.get_link_pose(self.robot, self.tool), self.gripper.offset)

    def set_pose(self, name, pose):
        pbu.set_pose(self.bodies[name], pose)

    def get_pose(self, name):
        return pbu.get_pose(self.bodies[name])

    def set_table_top(self, height):
        """Moves the table so that its top is at height (meters, in the robot's base frame)."""
        self.table_top = height
        pbu.set_point(self.table, (0.0, 0.0, height - TABLE_SIZE[2] / 2))

    def resting_pose(self, name, point):
        """The pose of an object resting on the table at (x, y)."""
        return (point[0], point[1], self.table_top + self.specs[name].size[2] / 2), pbu.unit_quat()

    def close(self):
        pbu.disconnect()


def quaternion_wxyz(pose):
    x, y, z, w = pose[1]
    return np.array([w, x, y, z])
