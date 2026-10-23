import os

import numpy as np

from examples.motion.viewer import create_box, draw_roadmap, draw_solution, get_distance, is_collision_free, sample_box
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.meta import create_parser
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import Profiler, read, str_from_object, user_input

ARRAY = np.array  # No hashing


def create_problem(goal, obstacles=(), regions={}, max_distance=0.5):
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'stream.pddl'))
    constant_map = {}

    q0 = ARRAY([0, 0])
    init = [
        ('Conf', q0),
        ('AtConf', q0),
    ] + [('Region', r) for r in regions]

    if isinstance(goal, str):
        goal = ('In', goal)
    else:
        init += [('Conf', goal)]
        goal = ('AtConf', goal)

    np.set_printoptions(precision=3)
    samples = []

    def region_gen(region):
        lower, upper = regions[region]
        # TODO: sample proportional to area
        while True:
            q = ARRAY(sample_box(regions[region]))
            samples.append(q)
            yield (q,)

    roadmap = []

    def connected_test(q1, q2):
        threshold = max_distance
        are_connected = (get_distance(q1, q2) <= threshold) and is_collision_free((q1, q2), obstacles)
        if are_connected:
            roadmap.append((q1, q2))
        return are_connected

    stream_map = {
        'sample-region': from_gen_fn(region_gen),
        'connect': from_test(connected_test),
        'distance': get_distance,
    }

    problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    return problem, samples, roadmap


##################################################

# TODO: algorithms that take advantage of metric space (RRT)


def main(max_time=20):
    """
    Creates and solves the 2D motion planning problem.
    """
    parser = create_parser()
    args = parser.parse_args()
    print('Arguments:', args)

    obstacles = [create_box((0.5, 0.5), (0.2, 0.2))]
    regions = {
        'env': create_box((0.5, 0.5), (1, 1)),
        'green': create_box((0.8, 0.8), (0.4, 0.4)),
    }

    goal = 'green'
    if goal not in regions:
        goal = ARRAY([1, 1])

    max_distance = 0.25  # 0.2 | 0.25 | 0.5 | 1.0
    problem, samples, roadmap = create_problem(goal, obstacles, regions, max_distance=max_distance)
    print('Initial:', str_from_object(problem.init))
    print('Goal:', str_from_object(problem.goal))
    constraints = PlanConstraints(max_cost=1.25)  # max_cost=INF)

    with Profiler(field='tottime', num=10):
        solution = solve_incremental(
            problem, constraints=constraints, unit_costs=args.unit, success_cost=0, max_time=max_time, verbose=False
        )

    print_solution(solution)
    plan, cost, evaluations = solution

    # TODO: use the same viewer here
    draw_roadmap(roadmap, obstacles, regions)  # TODO: do this in realtime
    user_input('Continue?')

    if plan is None:
        return
    segments = [args for name, args in plan]
    draw_solution(segments, obstacles, regions)
    user_input('Finish?')


if __name__ == '__main__':
    main()
