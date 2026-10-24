"""Solvers as compositions of strategies. A search strategy proposes plans, possibly with optimistic stream outputs;
a refinement strategy decides which streams to call and when the solve is over. drive() runs the loop that the
incremental, focused, binding and adaptive algorithms share (their strategies are in incremental.py and focused.py).

A new algorithm can reuse the loop with its own strategies through solve_with() (see pddlstream.strategy and
docs/strategies.md).
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from pddlstream.algorithms.advanced import automatically_negate_externals, enforce_simultaneous
from pddlstream.algorithms.algorithm import parse_problem, scoped_solve
from pddlstream.algorithms.common import SolutionStore
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.model import get_external_capabilities
from pddlstream.algorithms.scheduling.plan_streams import OptSolution
from pddlstream.core.context import get_config
from pddlstream.core.problem import INFEASIBLE
from pddlstream.core.utils import INF
from pddlstream.language.fluent import compile_fluent_streams
from pddlstream.language.lint import CapabilityError
from pddlstream.language.statistics import load_stream_statistics
from pddlstream.language.stream import Stream

logger = logging.getLogger(__name__)


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
    search_kwargs: dict = field(default_factory=dict)  # Search options (planner, max_planner_time, ...)

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
    """Proposes plans. A strategy can declare the stream capabilities it does not support (see check_capabilities)
    in an unsupported attribute."""

    def propose(self, state: SolveState, refinement: 'RefinementStrategy') -> OptSolution:
        """A plan: an OptSolution (stream_plan, action plan, cost), or FAILED/INFEASIBLE in place of the plans."""

    def postprocess(self, state: SolveState, solution: OptSolution) -> OptSolution:
        """The proposed solution after reordering and reporting it."""


class RefinementStrategy(Protocol):
    """Decides which streams to call, and when the solve is over."""

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


def check_capabilities(state, *strategies):
    """Raises a CapabilityError if a strategy does not support a capability of the problem's streams. A strategy
    declares these in its unsupported attribute: a message by capability ('fluent', 'optimizer', 'negated', ... as in
    describe()), formatted with the names of the streams that have it."""
    for strategy in strategies:
        for capability, message in getattr(strategy, 'unsupported', {}).items():
            names = [external.name for external in state.externals if capability in get_external_capabilities(external)]
            if names:
                raise CapabilityError(message.format(', '.join(names)))


def drive(state, search, refinement):
    """Alternates searching and refining until the refinement ends the solve or a limit is reached."""
    check_capabilities(state, search, refinement)
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


def compile_problem(problem, stream_info, constraints, unit_costs, unit_efforts):
    """Parses the problem and compiles its negated, simultaneous and fluent streams into the domain."""
    evaluations, goal_exp, domain, externals = parse_problem(
        problem, stream_info=stream_info, constraints=constraints, unit_costs=unit_costs, unit_efforts=unit_efforts
    )
    automatically_negate_externals(domain, externals)
    enforce_simultaneous(domain, externals)
    compile_fluent_streams(domain, externals)
    if get_config().strict:
        for external in externals:
            if isinstance(external, Stream) and external.is_negated and not external.is_complete:
                logger.warning(
                    f'{external.name} is used negated but not declared complete, so its failures are trusted'
                )
    return evaluations, goal_exp, domain, externals


@scoped_solve
def solve_with(
    problem,
    compose,
    constraints=PlanConstraints(),
    stream_info={},
    unit_costs=False,
    unit_efforts=False,
    success_cost=INF,
    max_time=INF,
    max_iterations=INF,
    max_memory=INF,
    initial_complexity=0,
    complexity_step=1,
    max_complexity=INF,
    verbose=True,
    deterministic=False,
    **search_kwargs,
):
    """Solves problem with the search and refinement strategies that compose(state) returns, on the loop that the
    built-in algorithms use (solve_adaptive is solve_with and the adaptive strategies). The problem is compiled as for
    the focused algorithms, so negated, simultaneous and fluent streams are supported.

    :param problem: a PDDLProblem
    :param compose: a function from the SolveState to a (search strategy, refinement strategy) pair
    The other parameters are those of solve_abstract; search_kwargs (planner, max_planner_time, ...) are kept in the
    state's search_kwargs for the search strategy.
    Returns a SolveResult, which unpacks as (plan, cost, certificate) and reports its status and stats.
    """
    evaluations, goal_exp, domain, externals = compile_problem(
        problem, stream_info, constraints, unit_costs, unit_efforts
    )
    if not deterministic:
        load_stream_statistics(externals)
    store = SolutionStore(evaluations, max_time, success_cost, verbose, max_memory=max_memory)
    state = SolveState(
        store, domain, goal_exp, externals, constraints, initial_complexity, complexity_step, max_iterations,
        max_complexity, search_kwargs=search_kwargs,
    )  # fmt: skip
    search, refinement = compose(state)
    drive(state, search, refinement)
    return finish(state, refinement)
