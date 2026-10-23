import numpy as np

from examples.discrete_belief.dist import DDist
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    SELF_COLLISIONS,
    Attach,
    Command,
    Conf,
    Detach,
    Pose,
    Trajectory,
    create_trajectory,
    get_target_path,
)
from examples.pybullet.utils.pybullet_tools.pr2_problems import get_fixed_bodies
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    HEAD_LINK_NAME,
    MAX_KINECT_DISTANCE,
    get_detection_cone,
    get_group_conf,
    get_group_joints,
    get_kinect_registrations,
    get_viewcone,
    get_visual_detections,
    inverse_visibility,
    plan_scan_path,
    set_group_conf,
    visible_base_generator,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    GREEN,
    RED,
    BodySaver,
    LockRenderer,
    apply_alpha,
    child_link_from_joint,
    create_mesh,
    get_body_name,
    get_length,
    get_link_pose,
    get_link_subtree,
    get_name,
    is_center_stable,
    link_from_name,
    pairwise_collision,
    plan_direct_joint_motion,
    plan_waypoints_joint_motion,
    point_from_pose,
    remove_body,
    set_pose,
    unit_pose,
    wait_for_duration,
    wait_for_user,
)

VIS_RANGE = (0.5, 1.5)
REG_RANGE = (0.5, 1.5)
ROOM_SCAN_TILT = np.pi / 6
P_LOOK_FP = 0
P_LOOK_FN = 0  # 1e-1


def get_in_range_test(task, range):
    # TODO: could just test different visibility w/o ranges
    def test(o, p, bq):
        if o in task.rooms:
            return True
        target_xy = point_from_pose(p.value)[:2]
        base_xy = bq.values[:2]
        return range[0] <= get_length(np.array(target_xy) - base_xy) <= range[1]

    return test


def get_vis_base_gen(task, base_range, collisions=False):
    robot = task.robot
    base_joints = get_group_joints(robot, 'base')
    obstacles = get_fixed_bodies(task) if collisions else []

    def gen(o, p):
        if o in task.rooms:  # TODO: predicate instead
            return
        # TODO: check collisions with fixed links
        target_point = point_from_pose(p.value)
        base_generator = visible_base_generator(robot, target_point, base_range)
        while True:
            set_pose(o, p.value)  # p.assign()
            bq = Conf(robot, base_joints, next(base_generator))
            bq.assign()
            if any(pairwise_collision(robot, b) for b in obstacles):
                yield None
            else:
                yield (bq,)

    # TODO: return list_fn & accelerated
    return gen


def get_inverse_visibility_fn(task, collisions=True):
    robot = task.robot
    head_joints = get_group_joints(robot, 'head')
    obstacles = get_fixed_bodies(task) if collisions else []

    def fn(o, p, bq):
        set_pose(o, p.value)  # p.assign()
        bq.assign()
        if o in task.rooms:
            waypoints = plan_scan_path(task.robot, tilt=ROOM_SCAN_TILT)
            set_group_conf(robot, 'head', waypoints[0])
            path = plan_waypoints_joint_motion(
                robot, head_joints, waypoints[1:], obstacles=obstacles, self_collisions=SELF_COLLISIONS
            )
            if path is None:
                return None
            ht = create_trajectory(robot, head_joints, path)
            hq = ht.path[0]
        else:
            target_point = point_from_pose(p.value)
            head_conf = inverse_visibility(robot, target_point)
            if head_conf is None:  # TODO: test if visible
                return None
            hq = Conf(robot, head_joints, head_conf)
            ht = Trajectory([hq])
        return (hq, ht)

    return fn


#######################################################


def plan_head_traj(task, head_conf):
    robot = task.robot
    obstacles = get_fixed_bodies(task)  # TODO: movable objects
    head_joints = get_group_joints(robot, 'head')
    head_path = plan_direct_joint_motion(
        robot, head_joints, head_conf, obstacles=obstacles, self_collisions=SELF_COLLISIONS
    )
    assert head_path is not None
    return create_trajectory(robot, head_joints, head_path)


def move_look_trajectory(task, trajectory, max_tilt=np.pi / 6):  # max_tilt=INF):
    # TODO: implement a minimum distance instead of max_tilt
    # TODO: pr2 movement restrictions
    base_path = trajectory.path
    if not base_path:
        return trajectory
    obstacles = get_fixed_bodies(task)  # TODO: movable objects
    robot = base_path[0].body
    target_path = get_target_path(trajectory)
    waypoints = []
    index = 0
    with BodySaver(robot):
        for i, conf in enumerate(base_path):  # TODO: just do two loops?
            conf.assign()
            while index < len(target_path):
                if i < index:
                    # Don't look at past or current conf
                    target_point = target_path[index]
                    head_conf = inverse_visibility(robot, target_point)  # TODO: this is slightly slow
                    if (head_conf is not None) and (head_conf[1] < max_tilt):
                        break
                index += 1
            else:
                head_conf = get_group_conf(robot, 'head')
            set_group_conf(robot, 'head', head_conf)
            waypoints.append(np.concatenate([conf.values, head_conf]))
    joints = tuple(base_path[0].joints) + tuple(get_group_joints(robot, 'head'))
    path = plan_waypoints_joint_motion(robot, joints, waypoints, obstacles=obstacles, self_collisions=SELF_COLLISIONS)
    return create_trajectory(robot, joints, path)


#######################################################


class AttachCone(Command):  # TODO: make extend Attach?
    def __init__(self, robot):
        self.robot = robot
        self.group = 'head'
        self.cone = None

    def apply(self, state, **kwargs):
        with LockRenderer():
            self.cone = get_viewcone(color=apply_alpha(RED, 0.5))
            state.poses[self.cone] = None
            cone_pose = Pose(self.cone, unit_pose())
            attach = Attach(self.robot, self.group, cone_pose, self.cone)
            attach.assign()
            wait_for_duration(1e-2)
        for _ in attach.apply(state, **kwargs):
            yield

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class DetachCone(Command):  # TODO: make extend Detach?
    def __init__(self, attach):
        self.attach = attach

    def apply(self, state, **kwargs):
        cone = self.attach.cone
        detach = Detach(self.attach.robot, self.attach.group, cone)
        for _ in detach.apply(state, **kwargs):
            yield
        del state.poses[cone]
        remove_body(cone)
        wait_for_duration(1e-2)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_cone_commands(robot):
    attach = AttachCone(robot)
    detach = DetachCone(attach)
    return attach, detach


#######################################################


def get_observation_fn(surface, p_look_fp=P_LOOK_FP, p_look_fn=P_LOOK_FN):
    # TODO: clip probabilities so doesn't become zero
    def fn(s):
        # P(obs | s1=loc1, a=control_loc)
        if s == surface:
            return DDist({True: 1 - p_look_fn, False: p_look_fn})
        return DDist({True: p_look_fp, False: 1 - p_look_fp})

    return fn


# TODO: update whether localized on scene


class Scan(Command):
    _duration = 0.5

    def __init__(self, robot, surface, detect=True, camera_frame=HEAD_LINK_NAME):
        self.robot = robot
        self.surface = surface
        self.camera_frame = camera_frame
        self.link = link_from_name(robot, self.camera_frame)
        self.detect = detect

    def apply(self, state, **kwargs):
        # TODO: identify surface automatically
        with LockRenderer():
            cone = get_viewcone(color=apply_alpha(RED, 0.5))
            set_pose(cone, get_link_pose(self.robot, self.link))
            wait_for_duration(1e-2)
        wait_for_duration(self._duration)  # TODO: don't sleep if no viewer?
        remove_body(cone)
        wait_for_duration(1e-2)

        if self.detect:
            # TODO: the collision geometries are being visualized
            # TODO: free the renderer
            head_joints = get_group_joints(self.robot, 'head')
            exclude_links = set(get_link_subtree(self.robot, child_link_from_joint(head_joints[-1])))
            detections = get_visual_detections(
                self.robot,
                camera_link=self.camera_frame,
                exclude_links=exclude_links,
            )
            print('Detections:', detections)
            for body, dist in state.b_on.items():
                obs = (body in detections) and (is_center_stable(body, self.surface))
                if obs or (self.surface not in state.task.rooms):
                    # TODO: make a command for scanning a room instead?
                    dist.obsUpdate(get_observation_fn(self.surface), obs)
        # TODO: pose for each object that can be real or fake
        yield

    def __repr__(self):
        return f'{self.__class__.__name__}({get_body_name(self.surface)})'


class ScanRoom(Command):
    _tilt = np.pi / 6

    def __init__(self, robot, surface):
        self.robot = robot
        self.surface = surface

    def apply(self, state, **kwargs):
        assert self.surface in state.task.rooms
        obs_fn = get_observation_fn(self.surface)
        for body, dist in state.b_on.items():
            if 0 < dist.prob(self.surface):
                dist.obsUpdate(obs_fn, True)
        yield

    def __repr__(self):
        return f'{self.__class__.__name__}({get_body_name(self.surface)})'


#######################################################


class Detect(Command):
    def __init__(self, robot, surface, body):
        self.robot = robot
        self.surface = surface
        self.body = body

    def apply(self, state, **kwargs):
        yield

    def __repr__(self):
        return f'{self.__class__.__name__}({get_name(self.surface)},{get_name(self.body)})'


class Register(Command):
    _duration = 1.0

    def __init__(self, robot, body, camera_frame=HEAD_LINK_NAME, max_depth=MAX_KINECT_DISTANCE):
        self.robot = robot
        self.body = body
        self.camera_frame = camera_frame
        self.max_depth = max_depth
        self.link = link_from_name(robot, camera_frame)

    def control(self, **kwargs):
        # TODO: filter for target object and location?
        return get_kinect_registrations(self.robot)

    def apply(self, state, **kwargs):
        # TODO: check if actually can register
        mesh, _ = get_detection_cone(self.robot, self.body, camera_link=self.camera_frame, depth=self.max_depth)
        if mesh is None:
            wait_for_user()
        assert mesh is not None  # TODO: is_visible_aabb(body_aabb, **kwargs)=False sometimes without collisions
        with LockRenderer():
            cone = create_mesh(mesh, color=apply_alpha(GREEN, 0.5))
            set_pose(cone, get_link_pose(self.robot, self.link))
            wait_for_duration(1e-2)
        wait_for_duration(self._duration)
        remove_body(cone)
        wait_for_duration(1e-2)
        state.registered.add(self.body)
        yield

    def __repr__(self):
        return f'{self.__class__.__name__}({get_body_name(self.robot)},{get_name(self.body)})'
