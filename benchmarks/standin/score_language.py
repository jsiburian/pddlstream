"""Scores the scripted language agent on the stand-in tasks: each instruction becomes a goal, which pddlstream's
propose_goal tool checks and which is compared with the goal from the task's success conditionals.

Usage: python -m benchmarks.standin.score_language
"""

import json

from benchmarks.common.language import ScriptedAgent, score
from benchmarks.standin.env import SCENARIOS, StandinEnv


def observe(scenario):
    env = StandinEnv(scenario)
    try:
        return env.observe()
    finally:
        env.close()


def main():
    results = score(ScriptedAgent(), SCENARIOS, [observe(scenario) for scenario in SCENARIOS])
    for result in results:
        print(json.dumps(result))
    correct = sum(result['correct'] and result['accepted'] for result in results)
    print(f'Correct and accepted: {correct}/{len(results)}')


if __name__ == '__main__':
    main()
