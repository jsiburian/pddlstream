"""Generates demonstrations on the stand-in tasks: runs the TAMP policy on jittered (and optionally perturbed) variants
of each task, keeps the episodes that succeeded with verified plans, and replays them open loop.

Usage: python -m benchmarks.standin.generate_demos --output demos [--seeds 0 1 2] [--perturb] [--task NAME]
"""

import argparse
import json
import pathlib

from benchmarks.common.remote import start_server
from benchmarks.standin.env import SCENARIOS
from benchmarks.standin.episodes import generate, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='demos', help='Directory for the episodes and index.jsonl')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--perturb', action='store_true', help='Also record episodes with an object moved')
    parser.add_argument('--task', default=None, help='Only this task')
    args = parser.parse_args()
    scenarios = [s for s in SCENARIOS if (args.task is None) or (s.task.name == args.task)]
    process, client = start_server()
    try:
        perturb = (False, True) if args.perturb else (False,)
        results = generate(client, args.output, scenarios, args.seeds, perturb)
    finally:
        process.kill()
    for result in results:
        print(json.dumps({key: result[key] for key in ['task', 'seed', 'perturbation', 'success', 'demonstration']}))
    demos = sorted(pathlib.Path(args.output).glob('*.npz'))
    replayed = sum(replay(path) for path in demos)
    print(
        f'Demonstrations: {sum(r["demonstration"] for r in results)}/{len(results)} episodes; '
        f'replayed successfully: {replayed}/{len(demos)}'
    )


if __name__ == '__main__':
    main()
