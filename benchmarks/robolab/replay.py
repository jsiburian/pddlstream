"""Rebuilds the planner's problem of a recorded RoboLab episode on a CPU and solves it, to debug the planner without
Isaac Sim. Two steps, in two environments:

  # RoboLab's (needs h5py, and pxr: Isaac Sim's USD libraries or usd-core): the run's first state and the meshes
  python -m benchmarks.robolab.replay export output/<folder>/<Task> --run 0 --output problem.json
  # pddlstream's: plans it as the server would, and explains a failure
  python -m benchmarks.robolab.replay solve problem.json
"""

import argparse
import json
import pathlib

import numpy as np

from benchmarks.robolab.meshes import object_points


def export(folder, run=0):
    """The task, the scene (its objects' sizes and surface points) and the first observation of a recorded run."""
    import h5py

    folder = pathlib.Path(folder)
    config = json.loads((folder / 'env_cfg.json').read_text())
    data = h5py.File(folder / f'run_{run}.hdf5', 'r')['data/demo_0']
    initial = data['initial_state']
    poses = {name: initial[f'rigid_object/{name}/root_pose'][0] for name in initial['rigid_object']}
    names = [name for name in poses if name != 'table']
    points = object_points(config['scene']['scene']['spawn']['usd_path'], names)
    sizes = {}  # Objects whose mesh can't be read here (e.g. a payload on NVIDIA's servers): their recorded box
    for name in set(names) - set(points):
        corners = data[f'bbox/bbox_mm/{name}'][0] / 1000.0
        sizes[name] = (corners.max(axis=0) - corners.min(axis=0)).tolist()
    return {
        'task': folder.name,
        'arm': initial['articulation/robot/joint_position'][0][:7].tolist(),
        'objects': {
            name: {'position': pose[:3].tolist(), 'quaternion': pose[3:].tolist()}
            for name, pose in poses.items()
            if name in names
        },
        'points': {name: np.round(array, 5).tolist() for name, array in points.items()},
        'sizes': sizes,
    }


def solve(problem, max_time=60):
    """Plans the problem with the policy the RoboLab client asks the server for; returns the policy and the solve."""
    from benchmarks.common.policy import TampPolicy
    from benchmarks.common.protocol import ObjectSpec, observation_from_json, task_from_json
    from benchmarks.robolab.baseline import TASKS
    from pddlstream.algorithms.explain import explain

    task = next(task_from_json(data) for data in json.loads(TASKS.read_text()) if data['name'] == problem['task'])
    supports = {c.params.get('container') or c.params.get('reference_object') for c in task.success} - {None}
    scene = []
    for name, points in problem['points'].items():
        array = np.array(points)
        size = tuple(array.max(axis=0) - array.min(axis=0))
        scene.append(ObjectSpec(name, size, 'container' if name in supports else 'movable', tuple(map(tuple, points))))
    for name, size in problem.get('sizes', {}).items():  # Boxes, as the runner makes objects without points
        scene.append(ObjectSpec(name, tuple(size), 'container' if name in supports else 'movable'))
    observation = observation_from_json({'arm': problem['arm'], 'gripper': 0.0, 'objects': problem['objects']})
    policy = TampPolicy(scene, gripper='robotiq', containment='center', focus=True, max_time=max_time)
    policy.reset(task, observation)
    return policy, (explain(policy.result) if policy.result is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    exporting = commands.add_parser('export', help="A recorded run's planning problem (RoboLab's environment)")
    exporting.add_argument('folder', help='The task folder of a RoboLab output folder')
    exporting.add_argument('--run', type=int, default=0)
    exporting.add_argument('--output', default='problem.json')
    solving = commands.add_parser('solve', help="Plans an exported problem (pddlstream's environment)")
    solving.add_argument('problem')
    solving.add_argument('--max-time', type=float, default=60)
    args = parser.parse_args()
    if args.command == 'export':
        pathlib.Path(args.output).write_text(json.dumps(export(args.folder, args.run)))
    else:
        policy, explanation = solve(json.loads(pathlib.Path(args.problem).read_text()), args.max_time)
        print(f'status: {policy.status}')
        for plan in policy.plans:
            print(plan)
        if explanation is not None:
            print(explanation)
        policy.close()


if __name__ == '__main__':
    main()
