"""The strategy SDK: build a solver from a search strategy and a refinement strategy, on the loop that the built-in
algorithms share. A plugin imports only this module, so it keeps working as the algorithms package changes.

    from pddlstream.strategy import OptimisticSearch, SkeletonRefinement, solve_with

    def compose(state):
        return OptimisticSearch.from_state(state, reorder=False), SkeletonRefinement.from_state(state)

    result = solve_with(problem, compose, max_time=30)

A search strategy proposes plans (propose, postprocess); a refinement strategy calls streams and decides when the
solve is over (initialize, before_search, may_search, after_search, stats). Either can be your own class, or wrap the
built-in ones. A BindingPolicy orders the bindings that the skeleton queue samples (solve(..., binding_policy=...)).
See docs/strategies.md.
"""

from pddlstream.algorithms.focused import (
    FocusedRefinement,
    GreedyRefinement,
    OptimisticSearch,
    SkeletonRefinement,
    solve_abstract,
)
from pddlstream.algorithms.incremental import ExhaustiveRefinement, FiniteSearch
from pddlstream.algorithms.scheduling.plan_streams import OptSolution
from pddlstream.algorithms.skeleton import BindingPolicy, Priority
from pddlstream.algorithms.strategies import (
    RefinementStrategy,
    SearchStrategy,
    SolveState,
    check_capabilities,
    drive,
    finish,
    solve_with,
)
from pddlstream.core.problem import FAILED, INFEASIBLE, is_plan
from pddlstream.language.lint import CapabilityError

__all__ = [
    'BindingPolicy',
    'CapabilityError',
    'ExhaustiveRefinement',
    'FAILED',
    'FiniteSearch',
    'FocusedRefinement',
    'GreedyRefinement',
    'INFEASIBLE',
    'OptSolution',
    'OptimisticSearch',
    'Priority',
    'RefinementStrategy',
    'SearchStrategy',
    'SkeletonRefinement',
    'SolveState',
    'check_capabilities',
    'drive',
    'finish',
    'is_plan',
    'solve_abstract',
    'solve_with',
]
