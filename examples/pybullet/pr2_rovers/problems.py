import numpy as np

from examples.pybullet.tamp.problems import sample_placements
from examples.pybullet.turtlebot_rovers.problems import RoversProblem
from examples.pybullet.utils.pybullet_tools.pr2_problems import create_pr2
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    ARM_NAMES,
    REST_LEFT_ARM,
    arm_conf,
    close_arm,
    set_arm_conf,
    set_group_conf,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    BLACK,
    BLUE,
    BROWN,
    GREY,
    HUSKY_URDF,
    RED,
    TAN,
    Point,
    add_data_path,
    create_box,
    dump_body,
    load_pybullet,
    set_point,
    stable_z,
)


def problem1(n_rovers=1, n_objectives=1, n_rocks=2, n_soil=2, n_stores=1):
    base_extent = 5.0
    base_limits = (-base_extent / 2.0 * np.ones(2), base_extent / 2.0 * np.ones(2))

    floor = create_box(base_extent, base_extent, 0.001, color=TAN)  # TODO: two rooms
    set_point(floor, Point(z=-0.001 / 2.0))

    add_data_path()
    lander = load_pybullet(HUSKY_URDF, scale=1)
    lander_z = stable_z(lander, floor)
    set_point(lander, Point(-2, -2, lander_z))

    mount_width = 0.5
    mound_height = mount_width
    mound1 = create_box(mount_width, mount_width, mound_height, color=GREY)
    set_point(mound1, [+2, 1.5, mound_height / 2.0])
    mound2 = create_box(mount_width, mount_width, mound_height, color=GREY)
    set_point(mound2, [-2, 1.5, mound_height / 2.0])

    initial_surfaces = {}

    rover_confs = [(+1, -1.75, np.pi), (-1, -1.75, 0)]
    assert n_rovers <= len(rover_confs)

    landers = [lander]
    stores = [f'store{i}' for i in range(n_stores)]

    rovers = []
    for i in range(n_rovers):
        robot = create_pr2()
        dump_body(robot)  # camera_rgb_optical_frame
        rovers.append(robot)
        set_group_conf(robot, 'base', rover_confs[i])
        for arm in ARM_NAMES:
            set_arm_conf(robot, arm, arm_conf(arm, REST_LEFT_ARM))
            close_arm(robot, arm)

    obj_width = 0.07
    obj_height = 0.2

    objectives = []
    for _ in range(n_objectives):
        body = create_box(obj_width, obj_width, obj_height, color=BLUE)
        objectives.append(body)
        initial_surfaces[body] = mound1
    for _ in range(n_objectives):
        body = create_box(obj_width, obj_width, obj_height, color=RED)
        initial_surfaces[body] = mound2

    # TODO: it is moving to intermediate locations attempting to reach the rocks
    rocks = []
    for _ in range(n_rocks):
        body = create_box(0.075, 0.075, 0.05, color=BLACK)
        rocks.append(body)
        initial_surfaces[body] = floor
    soils = []
    for _ in range(n_soil):
        body = create_box(0.1, 0.1, 0.025, color=BROWN)
        soils.append(body)
        initial_surfaces[body] = floor
    sample_placements(initial_surfaces)

    return RoversProblem(rovers, landers, objectives, rocks, soils, stores, base_limits)


PROBLEMS = [
    problem1,
]
