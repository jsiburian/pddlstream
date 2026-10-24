"""Evaluates the planner as a RoboLab policy with RoboLab's own evaluation loop (robolab/eval/runner.py), which writes
results, videos and summaries to RoboLab's output folder. Start the planner server in pddlstream's environment, then
run this in RoboLab's environment with the pddlstream checkout on PYTHONPATH:

  python -m benchmarks.common.server --port 8765                                  # pddlstream's environment
  python -m benchmarks.robolab.run --task BananaInBowlTask --headless --num-envs 1  # RoboLab's environment

The planner's scene comes from the running environment: a box with each object's bounding box (RoboLab's
WorldState). Tasks whose success conditionals the planner does not support run with the arm held still, so they
fail; benchmarks/robolab/probe.py lists them.
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 (RoboLab's runners import it before isaaclab)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Evaluate pddlstream as a RoboLab policy.')
parser.add_argument('--remote-host', default='localhost', help='The planner server (benchmarks/common/server.py)')
parser.add_argument('--remote-port', type=int, default=8765)
parser.add_argument('--skills', nargs='*', default=[], help='Learned skills the planner may use')
parser.add_argument('--vla-host', default='localhost', help="openpi's policy server, for the VLA skills")
parser.add_argument('--vla-port', type=int, default=None, help='With a port, the VLA takes picks and failed tasks')
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
args.enable_gt_state = True  # The planner reads the objects' poses
simulation_app = AppLauncher(args).app

from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.eval.episode import run_episode  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from benchmarks.common.protocol import ObjectSpec  # noqa: E402
from benchmarks.robolab.client import OpenpiVLA, TampInferenceClient, task_spec_from_robolab  # noqa: E402
from benchmarks.robolab.meshes import object_points  # noqa: E402

FIXTURES = {'table', 'robot'}  # The planner models the robot and the table itself

auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task)


def scene_points(env_cfg, names):
    """The objects' surface points from the task's scene file, or {} when it can't be read (the planner then uses
    their bounding boxes)."""
    try:
        return object_points(env_cfg.scene.scene.spawn.usd_path, names)
    except Exception:  # Any failure falls back to boxes
        traceback.print_exc()
        return {}


def scene_from_env(env, env_cfg, task):
    """The task's objects as ObjectSpecs: their bounding boxes, and their surface points from the scene file. Objects
    that the success conditionals put things in or on are containers, which the planner keeps fixed; the others are
    movable."""
    world = get_world(env)
    supports = {c.params.get('container') or c.params.get('reference_object') for c in task.success} - {None}
    names = [name for name in env_cfg.contact_object_list or [] if name not in FIXTURES]
    points = scene_points(env_cfg, names)
    specs = []
    for name in names:
        size = tuple(float(x) for x in world.get_dimensions(name))
        shape = tuple(map(tuple, points[name].round(5).tolist())) if name in points else ()
        specs.append(ObjectSpec(name, size, 'container' if name in supports else 'movable', shape))
    print(f'Scene: {[(spec.name, spec.kind, len(spec.points)) for spec in specs]}', flush=True)
    return specs


def make_client(args):
    if args.vla_port is None:
        return TampInferenceClient(args.remote_host, args.remote_port, args.skills)
    vla = OpenpiVLA(args.vla_host, args.vla_port)  # Waits until the server answers
    skills = list(dict.fromkeys([*args.skills, 'vla_pick']))
    return TampInferenceClient(args.remote_host, args.remote_port, skills, vla=vla)


def episode_runner(*, env, env_cfg, client, **kwargs):
    """RoboLab's episode, after giving the client the task and its scene."""
    task = task_spec_from_robolab(env_cfg)
    client.set_task(task, scene_from_env(env, env_cfg, task))
    return run_episode(env=env, env_cfg=env_cfg, client=client, **kwargs)


def main():
    run_evaluation(args, policy='pddlstream', client_factory=make_client, episode_runner=episode_runner)
    simulation_app.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
