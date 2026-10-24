"""Solvers as compositions of strategies. A search strategy proposes plans, possibly with optimistic stream outputs;
a refinement strategy decides which streams to call and when the solve is over. drive() runs the loop that the
incremental, focused, binding and adaptive algorithms share (their strategies are in incremental.py and focused.py).

A new algorithm can reuse the loop with its own strategies (see pddlstream.strategy and docs/strategies.md).
"""

from dataclasses import dataclass
from typing import Protocol

from pddlstream.algorithms.common import SolutionStore
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.scheduling.plan_streams import OptSolution
from pddlstream.core.problem import INFEASIBLE
from pddlstream.core.utils import INF


@dataclass
class SolveState:
    """What the strategies of a solve share: the solution store, the compiled problem and the loop's counters."""

    store: SolutionStore
    domain: object
    goal_exp: object
    externals: list
    constraints: PlanConstraints
    complexity_limit: float = 0
    complexity_step: float | None = 1
    max_iterations: float = INF
    max_complexity: float = INF
    iteration: int = 0
    infeasible: bool = False  # Whether the refinement concluded that no plan exists

    @property
    def evaluations(self):
        return self.store.evaluations

    def keep_iterating(self):
        return (
            (not self.store.is_terminated())
            and (self.iteration < self.max_iterations)
            and (self.complexity_limit <= self.max_complexity)
        )

    def max_cost(self):
        """The cost a new plan must beat."""
        return min(self.store.best_cost, self.constraints.max_cost)


class SearchStrategy(Protocol):
    def propose(self, state: SolveState, refinement: 'RefinementStrategy') -> OptSolution:
        """A plan: an OptSolution (stream_plan, action plan, cost), or FAILED/INFEASIBLE in place of the plans."""

    def postprocess(self, state: SolveState, solution: OptSolution) -> OptSolution:
        """The proposed solution after reordering and reporting it."""


class RefinementStrategy(Protocol):
    def initialize(self, state: SolveState) -> None:
        """Called once before the loop."""

    def before_search(self, state: SolveState) -> None:
        """Called at the start of each iteration."""

    def may_search(self, state: SolveState) -> bool:
        """Whether to search this iteration (otherwise the search counts as infeasible)."""

    def after_search(self, state: SolveState, solution: OptSolution) -> bool:
        """Samples streams for the solution (and records plans); returns True to end the solve."""

    def stats(self, state: SolveState) -> dict:
        """Statistics to add to the solve's summary."""


def drive(state, search, refinement):
    """Alternates searching and refining until the refinement ends the solve or a limit is reached."""
    refinement.initialize(state)
    while state.keep_iterating():
        state.iteration += 1
        refinement.before_search(state)
        if refinement.may_search(state):
            solution = search.propose(state, refinement)
        else:
            solution = OptSolution(INFEASIBLE, INFEASIBLE, INF)
        solution = search.postprocess(state, solution)
        if refinement.after_search(state, solution):
            break
    return state


def finish(state, refinement, statistics_externals=None):
    """The solve's SolveResult, with the loop's counters and the refinement's statistics in its summary."""
    externals = state.externals if statistics_externals is None else statistics_externals
    stats = dict(iterations=state.iteration, complexity=state.complexity_limit, **refinement.stats(state))
    return state.store.finish(externals, state.infeasible, **stats)
