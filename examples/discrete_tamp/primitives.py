from collections import namedtuple

import numpy as np

GRASP = np.array([0, 0])


def get_length(vec, ord=1):
    return np.linalg.norm(vec, ord=ord)


def get_difference(p1, p2):
    return np.array(p2) - p1


def collision_test(p1, p2):
    return get_length(get_difference(p1, p2)) < 1e-3


def distance_fn(q1, q2):
    return get_length(get_difference(q1, q2))


##################################################

DiscreteTAMPState = namedtuple('DiscreteTAMPState', ['conf', 'holding', 'block_poses'])
DiscreteTAMPProblem = namedtuple('DiscreteTAMPProblem', ['initial', 'poses', 'goal_poses'])
BLOCK_TEMPLATE = 'b{}'
INITIAL_CONF = np.array([0, -1])


def get_shift_one_problem(n_blocks=2, n_poses=9):
    assert 1 <= n_blocks <= n_poses
    blocks = [BLOCK_TEMPLATE.format(i) for i in range(n_blocks)]
    poses = [np.array([x, 0]) for x in range(n_poses)]

    block_poses = dict(zip(blocks, poses))
    initial = DiscreteTAMPState(INITIAL_CONF, None, block_poses)
    goal_poses = {blocks[0]: poses[1]}

    return DiscreteTAMPProblem(initial, poses[n_blocks:], goal_poses)


def get_shift_all_problem(n_blocks=2, n_poses=9):
    assert n_blocks + 1 <= n_poses
    blocks = [BLOCK_TEMPLATE.format(i) for i in range(n_blocks)]
    poses = [np.array([x, 0]) for x in range(n_poses)]

    block_poses = dict(zip(blocks, poses))
    initial = DiscreteTAMPState(INITIAL_CONF, None, block_poses)
    goal_poses = dict(zip(blocks, poses[1:]))

    return DiscreteTAMPProblem(initial, poses[n_blocks + 1 :], goal_poses)
