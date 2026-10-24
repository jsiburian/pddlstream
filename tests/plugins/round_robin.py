"""An outside strategy plugin, written against pddlstream.strategy only (tests/test_strategies.py checks its imports).

RoundRobinSearch takes turns between search configurations, one per iteration: e.g. a fast greedy planner and an
optimal one. RoundRobinPolicy gives each plan skeleton a turn at sampling among bindings of the same complexity. Both
run with adaptive's skeleton refinement.
"""

from collections import Counter

from pddlstream.strategy import BindingPolicy, OptimisticSearch, SkeletonRefinement, solve_with


class RoundRobinSearch:
    """A search strategy that delegates to its searches in turn."""

    def __init__(self, searches):
        self.searches = searches
        self.proposals = Counter()  # Searches by index

    def current(self, state):
        return (state.iteration - 1) % len(self.searches)

    def propose(self, state, refinement):
        index = self.current(state)
        self.proposals[index] += 1
        return self.searches[index].propose(state, refinement)

    def postprocess(self, state, solution):
        return self.searches[self.current(state)].postprocess(state, solution)


class RoundRobinPolicy(BindingPolicy):
    """Among bindings of the same complexity, takes turns between skeletons, rather than preferring the binding with
    the fewest visits."""

    def __init__(self):
        super().__init__()
        self.turns = Counter()  # Skeletons by index

    def priority(self, binding):
        default = super().priority(binding)
        skeleton = binding.skeleton.index
        self.turns[skeleton] += 1
        return (default.not_greedy, default.complexity, self.turns[skeleton], *default[2:])


def compose_round_robin(planners, strategies=None):
    """A compose function for solve_with; strategies, if given, collects the strategies of each solve."""

    def compose(state):
        searches = [
            OptimisticSearch.from_state(state, search_kwargs=dict(state.search_kwargs, planner=planner))
            for planner in planners
        ]
        search, refinement = RoundRobinSearch(searches), SkeletonRefinement.from_state(state)
        if strategies is not None:
            strategies.append((search, refinement))
        return search, refinement

    return compose


def solve_round_robin(problem, planners=('ff-astar', 'dijkstra'), strategies=None, **kwargs):
    """Solves problem with RoundRobinSearch over planners. Pass binding_policy=RoundRobinPolicy() to also take turns
    between skeletons."""
    return solve_with(problem, compose_round_robin(planners, strategies), **kwargs)
