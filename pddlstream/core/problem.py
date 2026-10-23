"""Problems, plans and results: PDDLProblem, Solution, SolveResult, plan actions, and printing and reading them."""

import enum
import os
from collections import namedtuple

from pddlstream.core.utils import INF, read, str_from_object

# TODO: OPTIMAL
SUCCEEDED = True
FAILED = None
INFEASIBLE = False
NOT_PLAN = [FAILED, INFEASIBLE]

# TODO: rename PDDLProblem
PDDLProblem = namedtuple('PDDLProblem', ['domain_pddl', 'constant_map', 'stream_pddl', 'stream_map', 'init', 'goal'])
Solution = namedtuple('Solution', ['plan', 'cost', 'certificate'])
Certificate = namedtuple('Certificate', ['all_facts', 'preimage_facts'])


class Status(enum.Enum):
    SUCCEEDED = 'succeeded'  # Found a plan
    FAILED = 'failed'  # Found no plan before a limit was reached (stats['timeout'] says whether time or memory ran out)
    INFEASIBLE = 'infeasible'  # Concluded that no plan exists, counting searches that hit max_planner_time as failed


class SolveResult(Solution):
    """What the solvers return: a Solution (plan, cost, certificate), which it unpacks and compares like, with the
    solver's status and statistics (the summary that it prints, including search_time, sample_time and run_time)."""

    status = None
    stats = None

    def __new__(cls, plan, cost, certificate, status=None, stats=None):
        self = super().__new__(cls, plan, cost, certificate)
        self.status = status
        self.stats = {} if stats is None else stats
        return self

    def __repr__(self):
        return f'{super().__repr__()[:-1]}, status={self.status})'


OptPlan = namedtuple('OptPlan', ['action_plan', 'preimage_facts'])
# TODO: stream and axiom plans
# TODO: annotate which step each fact is first used via layer

Assignment = namedtuple('Assignment', ['args'])
Action = namedtuple('Action', ['name', 'args'])
DurativeAction = namedtuple('DurativeAction', ['name', 'args', 'start', 'duration'])
StreamAction = namedtuple('StreamAction', ['name', 'inputs', 'outputs'])
FunctionAction = namedtuple('FunctionAction', ['name', 'inputs'])


##################################################


##################################################


##################################################


def is_plan(plan):
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
