"""Stream procedures for the tabletop domain, computed in the planner's PyBullet mirror of the scene (world.py)."""

import random

import numpy as np

from examples.pybullet.utils.pybullet_tools import utils as pbu
from examples.pybullet.utils.pybullet_tools.pr2_utils import get_top_grasps
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.language.stream import StreamInfo

APPROACH = pbu.Pose(point=pbu.Point(z=-0.1))  # The gripper approaches 10 cm along its axis
CENTER_MARGIN = 0.02  # With 'center' containment, object centers stay this far inside a container's footprint
PAD_HALF_WIDTH = 0.011  # Half a finger pad's width, across the direction the fingers close in
GRASP_DEPTH = 0.02  # How far below an object's top the tool frame goes
FINGER_CLEARANCE = 0.01  # Room left between the object and each open finger, and above the table
SELF_COLLISIONS = False  # PyBullet's Panda model reports contacts between the hand and its fingers
IK_ATTEMPTS = 10


class Pose:
    """An object's pose (a value of the planning problem)."""

    def __init__(self, world, name, value):
        self.world, self.name, self.value = world, name, value

    def assign(self):
        if self.name in self.world.bodies:
            self.world.set_pose(self.name, self.value)

    def __repr__(self):
        return f'p{id(self) % 1000}'


class Grasp:
    """The pose of an object in the gripper's tool frame, and the approach to it."""

    def __init__(self, name, value, approach=APPROACH):
        self.name, self.value, self.approach = name, value, approach

    def __repr__(self):
        return f'g{id(self) % 1000}'


class Conf:
    def __init__(self, values):
        self.values = tuple(values)

    def __repr__(self):
        return f'q{id(self) % 1000}'


class Trajectory:
    """A path of arm configurations, possibly carrying an object in a grasp."""

    def __init__(self, path, holding=None):
        self.path, self.holding = [tuple(conf) for conf in path], holding  # holding: (name, Grasp)

    def __repr__(self):
        return f't{id(self) % 1000}'


def assign_fluents(fluents):
    """Assigns the poses in AtPose fluents; returns the names of the objects that the fluents place."""
    names = []
    for fluent in fluents:
        predicate, *args = fluent
        if predicate.lower() == 'atpose':
            name, pose = args
            pose.assign()
            names.append(name)
    return names


def get_procedures(world, teleport=False):
    """The stream map of the tabletop domain for a World."""
    robot, arm = world.robot, world.arm

    def sample_pose(name, surface, surface_pose):
        surface_pose.assign()
        body, support = world.bodies[name], world.bodies.get(surface, world.table)
        container = world.specs[surface].kind == 'container' if surface in world.specs else False
        while True:
            if container and world.containment == 'center':
                value = pbu.sample_placement_on_aabb(body, center_region(world.container_aabb(surface)), percent=0.0)
            else:
                value = pbu.sample_placement(body, support)
            obstacles = [b for b in world.fixed if b != support]
            if (value is None) or any(pbu.pairwise_collision(body, obstacle) for obstacle in obstacles):
                yield None
                continue
            yield (Pose(world, name, value),)

    def sample_grasp(name):
        body, points = world.bodies[name], world.specs[name].points
        if points:
            grasps = point_grasps(points, world.gripper.max_width)
        else:
            grasps = get_top_grasps(
                body, under=True, tool_pose=pbu.Pose(), max_width=world.gripper.max_width, grasp_length=0.02
            )
        for value in grasps:
            yield (Grasp(name, value),)

    def inverse_kinematics(name, pose, grasp):
        pose.assign()
        gripper_pose = pbu.end_effector_from_body(pose.value, grasp.value)
        approach_pose = pbu.multiply(gripper_pose, grasp.approach)
        obstacles = world.fixed + [world.bodies[name]]
        sample_fn = pbu.get_sample_fn(robot, arm)
        for _ in range(IK_ATTEMPTS):
            world.set_arm(sample_fn())
            q_approach = solve_ik(world, approach_pose)
            if (q_approach is None) or collides(world, obstacles):
                continue
            q_grasp = solve_ik(world, gripper_pose)
            if (q_grasp is None) or collides(world, obstacles):
                continue
            world.set_arm(q_approach)
            path = (
                [q_approach, q_grasp]
                if teleport
                else pbu.plan_direct_joint_motion(robot, arm, q_grasp, self_collisions=SELF_COLLISIONS)
            )
            if path is None:
                continue
            return Conf(q_approach), Trajectory(path)
        return None

    def plan_motion(q1, q2, holding=None, fluents=[]):
        placed = [
            name for name in assign_fluents(fluents) if name in world.movables
        ]  # Fixed objects are already obstacles
        obstacles = world.fixed + [world.bodies[name] for name in placed]
        attachments = []
        if holding is not None:
            name, grasp = holding
            world.set_arm(q1.values)
            attachments = [attach(world, name, grasp)]
            obstacles = [b for b in obstacles if b != world.bodies[name]]
        world.set_arm(q1.values)
        if teleport:
            path = [q1.values, q2.values]
        else:
            kwargs = dict(obstacles=obstacles, attachments=attachments, self_collisions=SELF_COLLISIONS)
            path = pbu.plan_direct_joint_motion(robot, arm, q2.values, **kwargs)
            if path is None:
                path = pbu.plan_joint_motion(robot, arm, q2.values, **kwargs)
        if path is None:
            return None
        return (Trajectory(path, holding),)

    def free_motion(q1, q2, fluents=[]):
        return plan_motion(q1, q2, fluents=fluents)

    def holding_motion(q1, q2, name, grasp, fluents=[]):
        return plan_motion(q1, q2, holding=(name, grasp), fluents=fluents)

    def cfree_pose_pose(name1, pose1, name2, pose2):
        if name1 == name2:
            return True
        pose1.assign()
        pose2.assign()
        return not pbu.pairwise_collision(world.bodies[name1], world.bodies[name2])

    def cfree_approach_pose(name1, pose1, grasp, name2, pose2):
        if name1 == name2:
            return True
        pose2.assign()
        body1, body2 = world.bodies[name1], world.bodies[name2]
        gripper_pose = pbu.end_effector_from_body(pose1.value, grasp.value)
        approach_pose = pbu.multiply(gripper_pose, grasp.approach)
        for gripper in pbu.interpolate_poses(gripper_pose, approach_pose):
            pbu.set_pose(body1, pbu.multiply(gripper, grasp.value))
            if pbu.pairwise_collision(body1, body2):
                return False
        return True

    def cfree_traj_pose(trajectory, name, pose):
        if (trajectory.holding is not None) and (trajectory.holding[0] == name):
            return True
        pose.assign()
        body = world.bodies[name]
        for conf in trajectory.path:
            world.set_arm(conf)
            if pbu.pairwise_collision(robot, body):
                return False
            if trajectory.holding is not None:
                attach(world, *trajectory.holding).assign()
                if pbu.pairwise_collision(world.bodies[trajectory.holding[0]], body):
                    return False
        return True

    return {
        'sample-pose': from_gen_fn(sample_pose),
        'sample-grasp': from_gen_fn(sample_grasp),
        'inverse-kinematics': from_fn(inverse_kinematics),
        'plan-free-motion': from_fn(free_motion),
        'plan-holding-motion': from_fn(holding_motion),
        'test-cfree-pose-pose': from_test(cfree_pose_pose),
        'test-cfree-approach-pose': from_test(cfree_approach_pose),
        'test-cfree-traj-pose': from_test(cfree_traj_pose),
    }


def get_stream_info():
    """Stream options: the tests read only their inputs, and so do samplers, given the fixed scene."""
    pure = ['sample-pose', 'sample-grasp', 'inverse-kinematics', 'test-cfree-pose-pose', 'test-cfree-approach-pose']
    info = {name: StreamInfo(reads=()) for name in pure}
    info['test-cfree-traj-pose'] = StreamInfo(reads=(), negate=True)
    return info


def solve_ik(world, target_pose):
    """Arm joint positions that put the tool at target_pose, or None."""
    link_pose = pbu.multiply(target_pose, pbu.invert(world.gripper.offset))
    conf = pbu.inverse_kinematics(world.robot, world.tool, link_pose)
    if conf is None:
        return None
    conf = tuple(conf[: len(world.arm)])
    world.set_arm(conf)
    lower, upper = zip(*(pbu.get_joint_limits(world.robot, joint) for joint in world.arm))
    return conf if pbu.all_between(lower, conf, upper) else None


def balance_point(xy, bins=10):
    """Where an object seen from above balances, about: the mean of its centerline (the centers of slices across its
    main axis), as for a uniform rod. Unlike the mean of mesh vertices, which their density biases, it stays on the
    middle of a curved object such as a banana."""
    xy = np.asarray(xy, dtype=float)
    axis = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)[2][0]  # The direction of most spread
    along = (xy - xy.mean(axis=0)) @ axis
    edges = np.linspace(along.min(), along.max(), bins + 1)
    slices = [xy[(along >= low) & (along <= high)] for low, high in zip(edges, edges[1:])]
    return np.mean([part.mean(axis=0) for part in slices if len(part)], axis=0)


def point_grasps(points, max_width, yaws=16, num_centers=24, max_grasps=16):
    """Top grasps (tool from object poses) of an object given by surface points in its frame, assumed upright. The
    fingers close across a slice of the object (the points under the finger pads) narrower than max_width, centered
    on it; grasps nearest where the object balances (balance_point) come first, so it doesn't swing down. Unlike box grasps, they fit curved objects such as a
    banana, whose bounding box is wider than the gripper opens."""
    points = np.asarray(points, dtype=float)
    xy = points[:, :2]
    middle = balance_point(xy)
    # The middle, then points spread over the object, nearest the middle first
    order = np.argsort(np.linalg.norm(xy - middle, axis=1), kind='stable')
    centers = [middle] + [xy[i] for i in order[:: max(1, len(order) // num_centers)]]
    bottom, candidates = points[:, 2].min(), []
    for center in centers:
        for k in range(yaws):  # Over a full turn: a grasp and its mirror image need different wrist angles
            theta = 2 * np.pi * k / yaws
            close, across = np.array([np.cos(theta), np.sin(theta)]), np.array([-np.sin(theta), np.cos(theta)])
            offsets = xy - center
            under_pads = np.abs(offsets @ across) <= PAD_HALF_WIDTH
            if not under_pads.any():
                continue
            spans = offsets[under_pads] @ close
            low, high = spans.min(), spans.max()
            if (high - low) > (max_width - 2 * FINGER_CLEARANCE) or not (low <= 0 <= high):
                continue
            point = center + close * (low + high) / 2  # The middle of the slice
            top = points[under_pads, 2].max()
            z = max(top - GRASP_DEPTH, bottom + FINGER_CLEARANCE + 0.01)  # The fingertips are 1 cm below the tool
            # The tool approaches down (-z) and its fingers close along its y axis
            rotation = np.array([[across[0], close[0], 0.0], [across[1], close[1], 0.0], [0.0, 0.0, -1.0]])
            object_from_tool = ((point[0], point[1], z), pbu.quat_from_matrix(rotation))
            distance = round(float(np.linalg.norm(point - middle)), 3)
            candidates.append((distance, round(float(high - low), 3), k, pbu.invert(object_from_tool)))
    candidates.sort(key=lambda candidate: candidate[:3])
    return [grasp for *_, grasp in candidates[:max_grasps]]


def center_region(aabb, margin=CENTER_MARGIN):
    """The part of an AABB at least margin inside its sides (its center line when it is narrower), same top."""
    lower, upper = np.array(aabb[0]), np.array(aabb[1])
    shrink = np.minimum(margin, (upper - lower) / 2)
    shrink[2] = 0.0
    return pbu.AABB(lower + shrink, upper - shrink)


def collides(world, obstacles):
    """Whether the arm's moving links touch any of the obstacles. Its base is left out: on RoboLab's tables it stands
    level with the table top, as the motion planner's collision checks already allow."""
    moving = (world.robot, frozenset(pbu.get_moving_links(world.robot, world.arm)))
    return any(pbu.pairwise_collision(moving, obstacle) for obstacle in obstacles)


def attach(world, name, grasp):
    """The attachment of an object to the gripper in a grasp."""
    return pbu.Attachment(world.robot, world.tool, pbu.multiply(world.gripper.offset, grasp.value), world.bodies[name])


def seed(value=0):
    random.seed(value)
    np.random.seed(value)
