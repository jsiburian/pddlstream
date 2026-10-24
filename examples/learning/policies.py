"""Binding policies that learn, written against pddlstream.strategy only.

LearnedPolicy orders bindings by the success rates of their remaining streams, learned from the traces of earlier
solves. ConflictPolicy learns within a solve: values that keep leading to dead ends are tried later (conflict-directed
sampling). Both add to the default's complexity, which still orders bindings first.
"""

import math
from collections import Counter

from pddlstream.strategy import BindingPolicy


class StreamStatistics:
    """Each stream's success rate (calls that yielded results) from dataset rows, with a uniform prior."""

    def __init__(self, successes=None, calls=None):
        self.successes, self.calls = Counter(successes), Counter(calls)

    @classmethod
    def from_rows(cls, rows):
        statistics = cls()
        for row in rows:
            if (row['kind'] == 'call') and row['fresh']:
                statistics.calls[row['stream']] += 1
                statistics.successes[row['stream']] += row['outcome'] == 'yielded'
        return statistics

    def p_success(self, stream):
        return (self.successes[stream] + 1) / (self.calls[stream] + 2)


class LearnedPolicy(BindingPolicy):
    """Tries a binding as if it were more complex by weight times the expected failures of its remaining streams
    (-log of the probability that they all succeed), so likely bindings come first."""

    def __init__(self, statistics, weight=1.0):
        super().__init__()
        self.statistics, self.weight = statistics, weight

    def priority(self, binding):
        default = super().priority(binding)
        remaining = binding.skeleton.stream_plan[binding.index :]
        failures = -sum(math.log(self.statistics.p_success(result.external.name)) for result in remaining)
        return (default.not_greedy, default.complexity + self.weight * failures, *default[2:])


class ConflictPolicy(BindingPolicy):
    """Conflict-directed sampling. When a binding was visited but could not be extended, the values its last stream
    call introduced count as a conflict; a binding that uses values with conflicts is tried as if it were that much
    more complex."""

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.conflicts = Counter()  # By object

    def priority(self, binding):
        default = super().priority(binding)
        if binding.visits and not binding.children and (binding.parent_result is not None):
            self.conflicts.update(getattr(binding.parent_result, 'output_objects', ()))
        penalty = sum(self.conflicts[obj] for obj in binding.mapping.values())
        return (default.not_greedy, default.complexity + self.weight * penalty, *default[2:])
