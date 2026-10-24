"""Problems, plans and results: PDDLProblem, Solution, SolveResult, plan actions, and printing and reading them."""

import enum
import os
from typing import Any, NamedTuple

from pddlstream.core.syntax import Formula
from pddlstream.core.utils import INF, read, str_from_object

# TODO: OPTIMAL
SUCCEEDED = True
FAILED = None
INFEASIBLE = False
NOT_PLAN = [FAILED, INFEASIBLE]


# TODO: rename PDDLProblem
class PDDLProblem(NamedTuple):
    """A PDDLStream problem: what to plan in, which samplers to use, and where to start and end."""

    domain_pddl: Any  # The PDDL domain, as a string (or already parsed)
    constant_map: dict  # The value of each of the domain's constants
    stream_pddl: Any  # The stream definitions, as a string or a list of strings; None without streams
    stream_map: Any  # The procedure for each stream and function by name, or DEBUG or SHARED_DEBUG
    init: list[Formula]  # The facts that hold initially, e.g. ('On', block, region)
    goal: Formula  # The goal formula, e.g. And(('Holding', block))


class Certificate(NamedTuple):
    """The facts behind a solution: all those known at the end, and the ones the plan relies on."""

    all_facts: list
    preimage_facts: list | None


class Solution(NamedTuple):
    """A plan (a list of actions, or None), its cost (INF without a plan) and the certificate that supports it."""

    plan: list | None
    cost: float
    certificate: Certificate | None


class Status(enum.Enum):
    """How a solve ended. stats['termination'] says why the solver stopped, and stats['gaps'] what it cut short
    (docs/v2/adr-001-stream-outcomes.md)."""

    SUCCEEDED = 'succeeded'  # Found a plan
    FAILED = 'failed'  # Found no plan, either before reaching a limit or after cutting part of the search short
    INFEASIBLE = 'infeasible'  # Ran out of things to try without cutting anything short: no plan exists


class Incumbent(NamedTuple):
    """A plan that a solve found along the way; anytime solves find several, each cheaper than the last."""

    plan: list
    cost: float
    time: float  # Seconds since the solve started


class SolveResult(Solution):
    """What the solvers return: a Solution (plan, cost, certificate), which it unpacks and compares like, with the
    solver's status, statistics (the summary that it prints, including search_time, sample_time, run_time, the
    outcomes of the stream calls, termination and gaps) and solutions (every Incumbent, the best last)."""

    status: Status | None = None
    stats: dict | None = None
    solutions: list | None = None

    def __new__(cls, plan, cost, certificate, status=None, stats=None, solutions=None):
        self = super().__new__(cls, plan, cost, certificate)
        self.status = status
        self.stats = {} if stats is None else stats
        self.solutions = [] if solutions is None else solutions
        return self

    def __repr__(self):
        return f'{super().__repr__()[:-1]}, status={self.status})'


class OptPlan(NamedTuple):
    """An optimistic plan: its actions (with deferred stream and function steps) and the facts it relies on."""

    action_plan: list
    preimage_facts: Any


# TODO: stream and axiom plans
# TODO: annotate which step each fact is first used via layer


class Assignment(NamedTuple):
    """The step of a constraint-satisfaction plan that assigns values to free parameters."""

    args: Any


class Action(NamedTuple):
    """A plan step: an action's name and its arguments."""

    name: str
    args: tuple


class DurativeAction(NamedTuple):
    """A plan step with a start time and duration."""

    name: str
    args: tuple
    start: float
    duration: float


class StreamAction(NamedTuple):
    """A deferred step that calls a stream once the plan reaches it."""

    name: str
    inputs: tuple
    outputs: tuple


class FunctionAction(NamedTuple):
    """A deferred step that evaluates a function once the plan reaches it."""

    name: str
    inputs: tuple


##################################################


##################################################


##################################################


def is_plan(plan):
    """Whether plan is a plan rather than FAILED or INFEASIBLE."""
    return not any(plan is status for status in NOT_PLAN)


def get_length(plan):
    return len(plan) if is_plan(plan) else INF


def str_from_action(action):
    name, args = action[:2]
    return f'{name}{str_from_object(tuple(args))}'


def str_from_plan(plan):
    if not is_plan(plan):
        return str(plan)
    return str_from_object(list(map(str_from_action, plan)))


def print_plan(plan):
    if not is_plan(plan):
        return
    step = 1
    for action in plan:
        if isinstance(action, DurativeAction):
            name, args, start, duration = action
            print('{:.2f} - {:.2f}) {} {}'.format(start, start + duration, name, ' '.join(map(str_from_object, args))))
        elif isinstance(action, Action):
            name, args = action
            print('{:2}) {} {}'.format(step, name, ' '.join(map(str_from_object, args))))
            step += 1
        elif isinstance(action, StreamAction):
            name, inputs, outputs = action
            print(
                '    {}({})->({})'.format(
                    name, ', '.join(map(str_from_object, inputs)), ', '.join(map(str_from_object, outputs))
                )
            )
        elif isinstance(action, FunctionAction):
            name, inputs = action
            print('    {}({})'.format(name, ', '.join(map(str_from_object, inputs))))
        else:
            raise NotImplementedError(action)


def print_solution(solution):
    """Prints a solution's plan, cost and number of facts."""
    plan, cost, evaluations = solution
    if isinstance(evaluations, Certificate):  # Solvers return a Certificate of (all_facts, preimage_facts)
        evaluations = evaluations.all_facts
    solved = is_plan(plan)
    if plan is None:
        num_deferred = 0
    else:
        num_deferred = len(
            [action for action in plan if isinstance(action, StreamAction) or isinstance(action, FunctionAction)]
        )
    print()
    print(f'Solved: {solved}')
    print(f'Cost: {cost:.3f}')
    print(f'Length: {get_length(plan) - num_deferred}')
    print(f'Deferred: {num_deferred}')
    print(f'Evaluations: {len(evaluations)}')
    print_plan(plan)


##################################################

DOMAIN_FILE = 'domain.pddl'
PROBLEM_FILE = 'problem.pddl'
STREAM_FILE = 'stream.pddl'
PDDL_FILES = [DOMAIN_FILE, PROBLEM_FILE]
PDDLSTREAM_FILES = [DOMAIN_FILE, STREAM_FILE]


def read_relative(file, relative_path):  # file=__file__
    directory = os.path.dirname(file)
    path = os.path.abspath(os.path.join(directory, relative_path))
    return read(os.path.join(directory, path))


def read_relative_dir(file, relative_dir='./', filenames=[]):
    return [read_relative(file, os.path.join(relative_dir, filename)) for filename in filenames]


def read_pddl_pair(file, **kwargs):
    return read_relative_dir(file, filenames=PDDL_FILES, **kwargs)


def read_pddlstream_pair(file, **kwargs):
    return read_relative_dir(file, filenames=PDDLSTREAM_FILES, **kwargs)
