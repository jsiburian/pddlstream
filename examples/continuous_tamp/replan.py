"""Replanning with a PlannerSession: solve a 2D pick-and-place problem, move a block out of the way, then add a wall,
and show which stream results each new plan reused. Runs headless: python -m examples.continuous_tamp.replan
"""

import logging

import numpy as np

from examples.continuous_tamp.primitives import CARRY_Y, blocked, plan_motion
from examples.continuous_tamp.run import pddlstream_from_tamp, set_deterministic
from pddlstream.algorithms.session import PlannerSession, WorldPatch
from pddlstream.core.problem import print_solution
from pddlstream.language.generator import from_fn
from pddlstream.language.stream import StreamInfo

WORLD = {'walls': []}  # x coordinates of walls that the robot must lift blocks over; not part of the facts
WALL_HEIGHT = 3


def plan_motion_over_walls(q1, q2):
    """A trajectory between two configurations that rises over any wall between them."""
    (trajectory,) = plan_motion(q1, q2)
    low, high = sorted([q1[0], q2[0]])
    if any(low < wall < high for wall in WORLD['walls']):
        height = CARRY_Y + WALL_HEIGHT
        trajectory = [q1, np.array([q1[0], height]), np.array([q2[0], height]), q2]
    return (trajectory,)


def get_stream_info():
    # Each procedure depends only on its inputs, except motions, which also read the walls
    info = {name: StreamInfo(reads=()) for name in ['s-grasp', 's-region', 's-ik', 't-region', 't-cfree', 'dist']}
    info['s-motion'] = StreamInfo(reads=('walls',))
    return info


def solve_and_report(session, title):
    print(f'\n{title}')
    result = session.solve()
    print_solution(result)
    print(session.report)
    return result


def main():
    logging.getLogger('pddlstream').setLevel(logging.WARNING)  # Only the plans and reports
    set_deterministic()
    tamp_problem = blocked(n_blocks=3)
    problem = pddlstream_from_tamp(tamp_problem)
    problem.stream_map['s-motion'] = from_fn(plan_motion_over_walls)
    session = PlannerSession(
        problem, stream_info=get_stream_info(), planner='max-astar', unit_costs=True, deterministic=True, verbose=False
    )
    solve_and_report(session, 'Block B is in the goal region, so the robot moves it before placing A')

    # Someone moves block B out of the goal region
    old_pose, new_pose = tamp_problem.initial.block_poses['B'], np.array([-3.0, 0.0])
    session.apply(
        WorldPatch(add=[('Pose', 'B', new_pose), ('AtPose', 'B', new_pose)], remove=[('AtPose', 'B', old_pose)])
    )
    solve_and_report(session, 'B moved away: no result depended on where B was, so all of them are reused')

    # A wall appears between the blocks and the goal region; only motions read the walls
    WORLD['walls'].append(4.0)
    session.apply(WorldPatch(changed=['walls']))
    solve_and_report(session, 'A wall appeared: only motions are planned again')
    session.close()


if __name__ == '__main__':
    main()
