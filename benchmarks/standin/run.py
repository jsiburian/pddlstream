"""Runs the TAMP policy on the stand-in tasks, with the planner in its own server process, and reports each episode.

Usage: python -m benchmarks.standin.run [--task NAME] [--perturb] [--max-steps N] [--plans]
--perturb moves an object partway through each episode, to exercise replanning; --plans also prints each plan, its
certificate digest, whether it verifies, and the facts it assumes skills will achieve.
"""

import argparse
import json

from benchmarks.common.remote import start_server
from benchmarks.standin.env import SCENARIOS
from benchmarks.standin.episodes import run_episode

__all__ = ['run_episode']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=None, help='Only this task')
    parser.add_argument('--perturb', action='store_true', help='Move an object during each episode')
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--plans', action='store_true', help='Print the plans')
    args = parser.parse_args()
    scenarios = [s for s in SCENARIOS if (args.task is None) or (s.task.name == args.task)]
    process, client = start_server()
    try:
        results = []
        for scenario in scenarios:
            result = run_episode(client, scenario, args.max_steps, args.perturb)
            plans = result.pop('plans')
            print(json.dumps(result), flush=True)
            for report in plans if args.plans else []:
                print('  ', json.dumps(report), flush=True)
            results.append(result)
        num_success = sum(result['success'] for result in results)
        print(f'Succeeded: {num_success}/{len(results)}')
    finally:
        process.kill()


if __name__ == '__main__':
    main()
