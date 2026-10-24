"""Run on the GPU machine: evaluates the TAMP policy on RoboLab tasks, following RoboLab's run_eval.py template
(docs/environment_run.md), with ground-truth state enabled and the planner server running separately.

  # Terminal 1, in this repository's planning environment (pddlstream installed):
  python -m benchmarks.common.server --port 8765
  # Terminal 2, in RoboLab's environment:
  python -m benchmarks.robolab.run --task BananaInBowlTask --scenes scenes/ --headless --remote-port 8765

--scenes is a directory of ObjectSpec files named <task>.json (benchmarks/robolab/export_assets.py). Results are
appended to --results as JSON lines. Tasks whose conditionals the planner does not support are recorded as skipped.
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', nargs='+', default=None)
parser.add_argument('--tag', nargs='+', default=None)
parser.add_argument('--num-runs', type=int, default=1)
parser.add_argument('--scenes', required=True, help='Directory of <task>.json ObjectSpec files')
parser.add_argument('--remote-host', default='localhost')
parser.add_argument('--remote-port', type=int, default=8765)
parser.add_argument('--results', default='robolab_results.jsonl')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import torch  # noqa: E402 (Isaac Sim must start before these imports)
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from benchmarks.common.goals import UnsupportedTask, goal_from_task  # noqa: E402
from benchmarks.common.world import object_spec_from_json  # noqa: E402
from benchmarks.robolab.client import TampInferenceClient, task_spec_from_robolab  # noqa: E402


def run_episode(env, client, instruction):
    obs, _ = env.reset()
    success, steps = None, 0
    while steps < env.max_episode_length:
        steps += 1
        output = client.infer(obs, instruction)
        action = torch.as_tensor(output['action'], device=env.device).unsqueeze(0)
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated.any()):
            success = True
            break
        if bool(truncated.any()):
            success = False
            break
    client.reset()
    return success, steps


def main():
    auto_register_droid_envs(task=args.task)
    task_envs = get_envs(task=args.task) if args.task else get_envs(tag=args.tag) if args.tag else get_envs()
    for task_env in task_envs:
        env, env_cfg = create_env(task_env, device=args.device, num_envs=1, use_fabric=True)
        task = task_spec_from_robolab(env_cfg.task)  # The Task the environment was registered from
        result = {'task': task.name}
        try:
            goal_from_task(task)
        except UnsupportedTask as error:
            result.update(status='skipped', reason=str(error))
        else:
            with open(os.path.join(args.scenes, f'{task.name}.json')) as file:
                scene = [object_spec_from_json(spec) for spec in json.load(file)]
            client = TampInferenceClient(task, scene, remote_host=args.remote_host, remote_port=args.remote_port)
            for episode in range(args.num_runs):
                start = time.time()
                success, steps = run_episode(env, client, task.instruction['default'])
                end_episode(env)
                result.update(episode=episode, success=success, steps=steps, time=round(time.time() - start, 1))
        with open(args.results, 'a') as file:
            file.write(json.dumps(result) + '\n')
        env.close()
    simulation_app.close()


if __name__ == '__main__':
    main()
