"""The planner's RoboLab baseline: which tasks to run (the ones whose success conditions map to goals, split into
shards for parallel GPU jobs) and a summary of RoboLab's episode results.

  python -m benchmarks.robolab.baseline list --shard 1/2            # the supported tasks of shard 1 of 2
  python -m benchmarks.robolab.baseline summarize output/pddlstream-*  # success rates from RoboLab's output folders
"""

import argparse
import collections
import json
import pathlib
import re

from benchmarks.common.goals import coverage
from benchmarks.common.protocol import task_from_json

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASKS = ROOT / 'tests' / 'fixtures' / 'robolab_tasks.json'  # From benchmarks/robolab/probe.py


def supported_tasks(path=TASKS):
    """The names of the tasks whose success conditions the planner supports, sorted."""
    tasks = [task_from_json(data) for data in json.loads(pathlib.Path(path).read_text())]
    supported, _ = coverage(tasks)
    return sorted(supported)


def shard(names, index, count):
    """The index-th (1-based) of count interleaved shards, so that each holds a mix of task families."""
    return names[index - 1 :: count]


def read_results(folders):
    """RoboLab's episode results (episode_results.jsonl) from output folders, one dict per episode."""
    results = []
    for folder in folders:
        path = pathlib.Path(folder) / 'episode_results.jsonl'
        if path.exists():
            results.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return results


def summarize(results):
    """Episodes, successes, the success rate, and the failures grouped by the first condition that RoboLab's reason
    names as not holding."""
    successes = [result for result in results if result.get('success')]
    reasons = collections.Counter()
    for result in results:
        if not result.get('success'):
            condition = re.search(r'(\w+)\(', result.get('reason') or '')  # RoboLab words its reasons in several ways
            reasons[condition.group(1) if condition else 'unknown'] += 1
    return {
        'episodes': len(results),
        'successes': len(successes),
        'success_rate': round(len(successes) / len(results), 3) if results else None,
        'tasks': len({result['task_name'] for result in results}),
        'failures': dict(reasons.most_common()),
        'succeeded': sorted({result['task_name'] for result in successes}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='Print the supported tasks')
    listing.add_argument('--shard', default='1/1', help='index/count, e.g. 2/3')
    listing.add_argument('--tasks', default=str(TASKS), help='Task fixtures from benchmarks/robolab/probe.py')
    summary = commands.add_parser('summarize', help="Summarize RoboLab's episode results")
    summary.add_argument('folders', nargs='+', help='RoboLab output folders (with episode_results.jsonl)')
    args = parser.parse_args()
    if args.command == 'list':
        index, count = map(int, args.shard.split('/'))
        print(' '.join(shard(supported_tasks(args.tasks), index, count)))
    else:
        print(json.dumps(summarize(read_results(args.folders)), indent=1))


if __name__ == '__main__':
    main()
