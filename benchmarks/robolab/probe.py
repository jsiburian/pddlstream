"""RoboLab's tasks as the planner sees them.

By default, scans the task definitions in the RoboLab submodule (no Isaac Sim needed): each task's instruction,
success conditionals and objects. It saves them as JSON fixtures and reports which tasks the goal mapping supports.

With --live (on a GPU machine, in RoboLab's environment), it also starts Isaac Sim, resets each given task's
environment and saves the raw observation layout. That includes the ground-truth state (object poses, the gripper's
pose) and the arm's joint positions, to calibrate the planner's model of the robot.

Usage:
  python -m benchmarks.robolab.probe --output tests/fixtures/robolab_tasks.json
  python -m benchmarks.robolab.probe --live --task BananaInBowlTask --headless --output live.json
"""

import argparse
import ast
import json
import os
import pathlib
import sys

from benchmarks.common.goals import coverage
from benchmarks.common.protocol import Conditional, TaskSpec, to_json

TASKS_DIR = pathlib.Path(__file__).resolve().parent / 'RoboLab' / 'robolab' / 'tasks' / 'benchmark'


def literal(node, default=None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return default


def scan_task_file(path):
    """The TaskSpecs that a RoboLab task file defines: success terminations (DoneTerm with a func and no time_out),
    instructions and contact objects, read from its literal values."""
    tree = ast.parse(path.read_text())
    conditionals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'DoneTerm':
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if ('func' not in keywords) or literal(keywords.get('time_out'), False):
                continue
            func = getattr(keywords['func'], 'id', None) or getattr(keywords['func'], 'attr', None)
            conditionals.append(Conditional(func, literal(keywords.get('params'), {}) or {}))
    specs = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not any(getattr(base, 'id', None) == 'Task' for base in node.bases):
            continue
        fields = {}
        for statement in node.body:
            target = getattr(statement, 'target', None) or next(iter(getattr(statement, 'targets', [])), None)
            if isinstance(target, ast.Name) and getattr(statement, 'value', None) is not None:
                fields[target.id] = literal(statement.value)
        instruction = fields.get('instruction') or ''
        instruction = instruction if isinstance(instruction, dict) else {'default': instruction}
        objects = tuple(fields.get('contact_object_list') or ())
        specs.append(TaskSpec(fields.get('task_name') or node.name, instruction, tuple(conditionals), objects))
    return specs


def scan_tasks(tasks_dir=TASKS_DIR):
    return [spec for path in sorted(pathlib.Path(tasks_dir).glob('*.py')) for spec in scan_task_file(path)]


def probe_live(args):
    """Starts Isaac Sim and saves each task's first raw observation (with ground-truth state) as JSON."""
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    import numpy as np
    from robolab.core.environments.runtime import create_env
    from robolab.eval.gt_state import GroundTruthStateExporter
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    auto_register_droid_envs(task=args.task)
    dumps = {}
    for task in args.task:
        env, env_cfg = create_env(task, device=args.device, num_envs=1)
        obs, _ = env.reset()
        exporter = GroundTruthStateExporter(env, env_cfg)
        dumps[task] = {
            'observation_groups': {group: sorted(terms) for group, terms in obs.items() if hasattr(terms, 'keys')},
            'gt_state': exporter.export_all([0])[0],
            'action_dim': int(env.action_space.shape[-1]),
            'joint_names': list(env.scene['robot'].joint_names),
            'joint_pos': env.scene['robot'].data.joint_pos[0].cpu().numpy(),
            'body_names': list(env.scene['robot'].body_names),
            'body_pose': env.scene['robot'].data.body_state_w[0, :, :7].cpu().numpy(),
        }
        env.close()
    with open(args.output, 'w') as file:
        json.dump(dumps, file, indent=1, default=lambda value: np.asarray(value).tolist())
    simulation_app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-dir', default=str(TASKS_DIR), help='RoboLab task definitions to scan')
    parser.add_argument('--output', default='robolab_tasks.json')
    parser.add_argument('--live', action='store_true', help='Start Isaac Sim and dump observations (GPU)')
    parser.add_argument('--task', nargs='+', default=['BananaInBowlTask'], help='Tasks to probe with --live')
    if '--live' in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.live:
        probe_live(args)
        return
    if not os.path.isdir(args.tasks_dir):
        sys.exit(f'No task definitions in {args.tasks_dir}: git submodule update --init benchmarks/robolab/RoboLab')
    specs = scan_tasks(args.tasks_dir)
    with open(args.output, 'w') as file:
        json.dump([to_json(spec) for spec in specs], file, indent=1)
    supported, skipped = coverage(specs)
    print(f'{len(specs)} tasks; supported: {len(supported)}')
    for _, reason in sorted(skipped.items()):
        print(f'  skipped {reason}')


if __name__ == '__main__':
    main()
