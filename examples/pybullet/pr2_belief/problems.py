from examples.discrete_belief.dist import DeltaDist, MixtureDD, UniformDist
from examples.pybullet.utils.pybullet_tools.pr2_primitives import State
from examples.pybullet.utils.pybullet_tools.pr2_problems import create_kitchen, create_pr2
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    REST_LEFT_ARM,
    arm_conf,
    close_arm,
    create_gripper,
    get_carry_conf,
    get_other_arm,
    open_arm,
    set_arm_conf,
)
from examples.pybullet.utils.pybullet_tools.utils import HideOutput, get_bodies, get_name, is_center_stable

USE_DRAKE_PR2 = True
OTHER = 'other'
LOCALIZED_PROB = 0.99


class BeliefTask:
    def __init__(
        self,
        robot,
        arms=tuple(),
        grasp_types=tuple(),
        class_from_body={},
        movable=tuple(),
        surfaces=tuple(),
        rooms=tuple(),
        goal_localized=tuple(),
        goal_registered=tuple(),
        goal_holding=tuple(),
        goal_on=tuple(),
    ):
        self.robot = robot
        self.arms = arms
        self.grasp_types = grasp_types
        self.class_from_body = class_from_body
        self.movable = movable
        self.surfaces = surfaces
        self.rooms = rooms
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_localized = goal_localized
        self.goal_registered = goal_registered
        self.gripper = None

    def get_bodies(self):
        return self.movable + self.surfaces + self.rooms

    @property
    def fixed(self):
        movable = [self.robot] + list(self.movable)
        if self.gripper is not None:
            movable.append(self.gripper)
        return list(filter(lambda b: b not in movable, get_bodies()))

    def get_supports(self, body):
        if body in self.movable:
            return self.surfaces
        if body in self.surfaces:
            return self.rooms
        if body in self.rooms:
            return None
        raise ValueError(body)

    def get_gripper(self, arm='left'):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm)
        return self.gripper


# TODO: operate on histories to do open-world
class BeliefState(State):
    def __init__(self, task, b_on={}, registered=tuple(), **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.b_on = b_on
        self.registered = set(registered)
        # TODO: store configurations
        """
        for body in task.get_bodies():
            if not self.is_localized(body):
                #self.poses[body] = None
                self.poses[body] = object()
                #del self.poses[body]
            #elif body not in registered:
            #    point, quat = self.poses[body].value
            #    self.poses[body] = Pose(body, (point, None))
        """

    def is_localized(self, body):
        for surface in self.b_on[body].support():
            if (surface != OTHER) and (LOCALIZED_PROB <= self.b_on[body].prob(surface)):
                return True
        return False

    def __repr__(self):
        items = []
        for b in sorted(self.b_on.keys()):
            d = self.b_on[b]
            support_items = [f'{s}: {d.prob(s):.2f}' for s in sorted(d.support(), key=str)]
            items.append('{}: {{{}}}'.format(b, ', '.join(support_items)))
        return '{}({},{})'.format(
            self.__class__.__name__, '{{{}}}'.format(', '.join(items)), list(map(get_name, self.registered))
        )


#######################################################


def set_uniform_belief(task, b_on, body, p_other=0.0):
    # p_other is the probability that it doesn't actually exist
    # TODO: option to bias towards particular bottom
    other = DeltaDist(OTHER)
    uniform = UniformDist(task.get_supports(body))
    b_on[body] = MixtureDD(other, uniform, p_other)


def set_delta_belief(task, b_on, body):
    supports = task.get_supports(body)
    if supports is None:
        b_on[body] = DeltaDist(supports)
        return
    for bottom in task.get_supports(body):
        if is_center_stable(body, bottom):
            b_on[body] = DeltaDist(bottom)
            return
    raise RuntimeError(f'No support for body {body}')


#######################################################


def get_localized_rooms(task, **kwargs):
    # TODO: I support that in a closed world, it would automatically know where they are
    # TODO: difference between knowing position confidently and where it is
    b_on = {}
    for body in task.surfaces + task.movable:
        set_uniform_belief(task, b_on, body, **kwargs)
    for body in task.rooms:
        set_delta_belief(task, b_on, body)
    return BeliefState(task, b_on=b_on)


def get_localized_surfaces(task, **kwargs):
    b_on = {}
    for body in task.movable:
        set_uniform_belief(task, b_on, body, **kwargs)
    for body in task.rooms + task.surfaces:
        set_delta_belief(task, b_on, body)
    return BeliefState(task, b_on=b_on)


def get_localized_movable(task):
    b_on = {}
    for body in task.rooms + task.surfaces + task.movable:
        set_delta_belief(task, b_on, body)
    return BeliefState(task, b_on=b_on)


#######################################################


def get_kitchen_task(arm='left', grasp_type='top'):
    with HideOutput():
        pr2 = create_pr2(use_drake=USE_DRAKE_PR2)
    set_arm_conf(pr2, arm, get_carry_conf(arm, grasp_type))
    open_arm(pr2, arm)
    other_arm = get_other_arm(arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)

    table, cabbage, sink, stove = create_kitchen()
    floor = get_bodies()[1]
    class_from_body = {
        table: 'table',
        cabbage: 'cabbage',
        sink: 'sink',
        stove: 'stove',
    }  # TODO: use for debug
    movable = [cabbage]
    surfaces = [table, sink, stove]
    rooms = [floor]

    return BeliefTask(
        robot=pr2,
        arms=[arm],
        grasp_types=[grasp_type],
        class_from_body=class_from_body,
        movable=movable,
        surfaces=surfaces,
        rooms=rooms,
        goal_on=[(cabbage, sink)],
    )


def get_problem1(localized='rooms', **kwargs):
    task = get_kitchen_task()
    if localized == 'rooms':
        initial = get_localized_rooms(task, **kwargs)
    elif localized == 'surfaces':
        initial = get_localized_surfaces(task, **kwargs)
    elif localized == 'movable':
        initial = get_localized_movable(task)
    else:
        raise ValueError(localized)
    return task, initial
