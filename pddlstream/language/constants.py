import enum
import os
from collections import namedtuple

from pddlstream.utils import INF, read, str_from_object

EQ = '='  # xnor
AND = 'and'
OR = 'or'
NOT = 'not'
EXISTS = 'exists'
FORALL = 'forall'
WHEN = 'when'
IMPLY = 'imply'
MINIMIZE = 'minimize'
MAXIMIZE = 'maximize'
INCREASE = 'increase'
PARAMETER = '?'
TYPE = '-'
OBJECT = 'object'
TOTAL_COST = 'total-cost'  # TotalCost
CONNECTIVES = (AND, OR, NOT, IMPLY)
QUANTIFIERS = (FORALL, EXISTS)
OBJECTIVES = (MINIMIZE, MAXIMIZE, INCREASE)
OPERATORS = CONNECTIVES + QUANTIFIERS + (WHEN,)  # + OBJECTIVES

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

Head = namedtuple('Head', ['function', 'args'])
Evaluation = namedtuple('Evaluation', ['head', 'value'])
Atom = lambda head: Evaluation(head, True)
NegatedAtom = lambda head: Evaluation(head, False)

##################################################


def Output(*args):
    return tuple(args)


def And(*expressions):
    if len(expressions) == 1:
        return expressions[0]
    return (AND,) + tuple(expressions)


def Or(*expressions):
    if len(expressions) == 1:
        return expressions[0]
    return (OR,) + tuple(expressions)


def Not(expression):
    return (NOT, expression)


def Imply(expression1, expression2):
    return (IMPLY, expression1, expression2)


def Equal(expression1, expression2):
    return (EQ, expression1, expression2)


def Minimize(expression):
    return (MINIMIZE, expression)


def Type(param, ty):
    return (param, TYPE, ty)


def Exists(args, expression):
    return (EXISTS, args, expression)


def ForAll(args, expression):
    return (FORALL, args, expression)


##################################################


def get_prefix(expression):
    return expression[0]


def get_args(head):
    return head[1:]


def concatenate(*args):
    output = []
    for arg in args:
        output.extend(arg)
    return tuple(output)


def Fact(predicate, args):
    return (predicate,) + tuple(args)


def is_parameter(expression):
    return isinstance(expression, str) and expression.startswith(PARAMETER)


def get_parameter_name(expression):
    if is_parameter(expression):
        return expression[len(PARAMETER) :]
    return expression


def is_head(expression):
    return get_prefix(expression) not in OPERATORS


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


def get_function(term):
    if get_prefix(term) in (EQ, MINIMIZE, NOT):
        return term[1]
    return term


def partition_facts(facts):
    functions = []
    negated = []
    positive = []
    for fact in facts:
        prefix = get_prefix(fact)
        func = get_function(fact)
        if prefix in (EQ, MINIMIZE):
            functions.append(func)
        elif prefix == NOT:
            negated.append(func)
        else:
            positive.append(func)
    return positive, negated, functions


def is_cost(o):
    return get_prefix(o) == MINIMIZE


def get_costs(objectives):
    return [o for o in objectives if is_cost(o)]


def get_constraints(objectives):
    return [o for o in objectives if not is_cost(o)]


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
