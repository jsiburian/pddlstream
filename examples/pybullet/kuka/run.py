#!/usr/bin/env python

from examples.pybullet.tamp.streams import get_cfree_obj_approach_pose_test, get_cfree_pose_pose_test
from examples.pybullet.utils.pybullet_tools.kuka_primitives import (
    BodyConf,
    BodyPose,
    Command,
    get_free_motion_gen,
    get_grasp_gen,
    get_holding_motion_gen,
    get_ik_fn,
    get_movable_collision_test,
    get_stable_gen,
    get_tool_link,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    BLOCK_URDF,
    DRAKE_IIWA_URDF,
    SINK_URDF,
    SMALL_BLOCK_URDF,
    STOVE_URDF,
    HideOutput,
    LockRenderer,
    Point,
    Pose,
    WorldSaver,
    connect,
    disconnect,
    draw_global_system,
    draw_pose,
    get_bodies,
    get_body_name,
    get_configuration,
    get_pose,
    has_gui,
    is_placement,
    load_model,
    set_default_camera,
    set_pose,
    stable_z,
    wait_for_user,
)
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.utils import INF, Profiler, get_file_path, negate_test, read, str_from_object


def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed


def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed


#######################################################


def pddlstream_from_problem(robot, movable=[], teleport=False, grasp_name='top'):

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    print('Robot:', robot)
    conf = BodyConf(robot, get_configuration(robot))
    init = [('CanMove',), ('Conf', conf), ('AtConf', conf), ('HandEmpty',)]

    fixed = get_fixed(robot, movable)
    print('Movable:', movable)
    print('Fixed:', fixed)
    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [('Graspable', body), ('Pose', body, pose), ('AtPose', body, pose)]
        for surface in fixed:
            init += [('Stackable', body, surface)]
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    for body in fixed:
        name = get_body_name(body)
        if 'sink' in name:
            init += [('Sink', body)]
        if 'stove' in name:
            init += [('Stove', body)]

    body = movable[0]
    goal = (
        'and',
        ('AtConf', conf),
        ('Cooked', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())),
        'TrajCollision': get_movable_collision_test(),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################


def load_world():
    # TODO: store internal world info here to be reloaded
    set_default_camera()
    draw_global_system()
    with HideOutput():
        robot = load_model(DRAKE_IIWA_URDF, fixed_base=True)  # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = load_model('models/short_floor.urdf')
        sink = load_model(SINK_URDF, pose=Pose(Point(x=-0.5)))
        stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5)))
        celery = load_model(BLOCK_URDF, fixed_base=False)
        radish = load_model(SMALL_BLOCK_URDF, fixed_base=False)

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot))  # TODO: not working

    body_names = {
        sink: 'sink',
        stove: 'stove',
        celery: 'celery',
        radish: 'radish',
    }
    movable_bodies = [celery, radish]

    set_pose(celery, Pose(Point(y=0.5, z=stable_z(celery, floor))))
    set_pose(radish, Pose(Point(y=-0.5, z=stable_z(radish, floor))))

    return robot, body_names, movable_bodies


def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths
    return Command(paths)


#######################################################


def main():
    parser = create_parser()
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=True)
    robot, names, movable = load_world()
    print('Objects:', names)
    saver = WorldSaver()

    problem = pddlstream_from_problem(robot, movable=movable, teleport=args.teleport)
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    with Profiler():
        with LockRenderer(lock=not args.enable):
            solution = solve(problem, algorithm=args.algorithm, unit_costs=args.unit, success_cost=INF)
            saver.restore()
    print_solution(solution)
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    command = postprocess_plan(plan)
    if args.simulate:
        wait_for_user('Simulate?')
        command.control()
    else:
        wait_for_user('Execute?')
        command.refine(num_steps=10).execute(time_step=0.001)
    wait_for_user('Finish?')
    disconnect()


if __name__ == '__main__':
    main()
