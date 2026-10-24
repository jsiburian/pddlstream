"""Learning binding policies: collects a dataset from the traces of training solves, then solves held-out problems
with the default policy, a policy learned from the dataset, and conflict-directed sampling, and reports their calls.
Run: python -m examples.learning.run
"""

import logging

from examples.continuous_tamp.primitives import blocked, tight
from examples.continuous_tamp.run import pddlstream_from_tamp, set_deterministic
from examples.learning.policies import ConflictPolicy, LearnedPolicy, StreamStatistics
from pddlstream.algorithms.meta import solve
from pddlstream.strategy import rows_from_trace

OPTIONS = dict(algorithm='adaptive', unit_costs=True, deterministic=True, verbose=False, max_time=60)
TRAINING = {'tight-1': lambda: tight(n_blocks=1, n_goals=1), 'blocked-2': lambda: blocked(n_blocks=2)}
HELD_OUT = {
    'tight-2': lambda: tight(n_blocks=2, n_goals=2),
    'blocked-3': lambda: blocked(n_blocks=3),
    'blocked-4': lambda: blocked(n_blocks=4),
}


def make_problem(make_tamp):
    set_deterministic()
    return pddlstream_from_tamp(make_tamp())


def collect(problems):
    rows = []
    for name, make_tamp in problems.items():
        result = solve(make_problem(make_tamp), trace=True, **OPTIONS)
        rows.extend(rows_from_trace(result.trace, problem=name))
    return rows


def evaluate(problems, policies):
    """Calls and status of each held-out problem under each policy."""
    table = {}
    for name, make_tamp in problems.items():
        for label, make_policy in policies.items():
            result = solve(make_problem(make_tamp), binding_policy=make_policy(), **OPTIONS)
            table[name, label] = (result.status.value, sum(result.stats['outcomes'].values()))
    return table


def main():
    logging.getLogger('pddlstream').setLevel(logging.WARNING)
    rows = collect(TRAINING)
    statistics = StreamStatistics.from_rows(rows)
    print(
        f'Dataset: {len(rows)} rows; success rates:',
        {stream: round(statistics.p_success(stream), 2) for stream in sorted(statistics.calls)},
    )
    policies = {'default': lambda: None, 'learned': lambda: LearnedPolicy(statistics), 'conflicts': ConflictPolicy}
    for (name, label), (status, calls) in evaluate(HELD_OUT, policies).items():
        print(f'{name:10} {label:9} {status:9} {calls} calls')


if __name__ == '__main__':
    main()
