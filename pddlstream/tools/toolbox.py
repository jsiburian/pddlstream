"""Tools that an agent can call to inspect problems, test procedures, search for plans and read why searches failed.

Every tool takes and returns JSON-compatible data. None accepts facts, initial states or certificates: an agent can
propose (a plan to validate, a skeleton to try first) but never assert what holds. Toolbox.call() wraps each call as
{'ok': True, 'result': ...} or {'ok': False, 'error': {'code', 'message'}}.
"""

import dataclasses
import time
from collections import Counter

from pddlstream.algorithms.algorithm import solve_scope
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.explain import explain
from pddlstream.algorithms.model import describe
from pddlstream.algorithms.validate import validate_solution, verify_certificate
from pddlstream.core.config import check_keywords
from pddlstream.core.conversion import fact_from_evaluation, transform_plan_args
from pddlstream.core.evaluations import evaluations_from_init
from pddlstream.core.objects import Object
from pddlstream.core.problem import Action, Certificate, Solution
from pddlstream.core.registry import Registry
from pddlstream.fastdownward.tasks import (
    apply_action,
    evaluation_from_fd,
    get_action_instances,
    get_derived_predicates,
    get_problem,
    literal_holds,
    parse_sequential_domain,
    task_from_domain_problem,
)
from pddlstream.language.generator import BoundedGenerator
from pddlstream.language.stream import Stream, WildOutput

BUDGET_KEYS = ['max_time', 'max_iterations', 'max_complexity', 'success_cost', 'algorithm', 'deterministic']


@dataclasses.dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict  # JSON Schema of the arguments
    fn: object


def schema(properties, required=()):
    return {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}


STRING, INTEGER, NUMBER = {'type': 'string'}, {'type': 'integer'}, {'type': 'number'}
BUDGET = {'type': 'object', 'description': f'Any of {BUDGET_KEYS}'}
PLAN = {'type': 'array', 'description': 'Actions as [name, [arguments...]]', 'items': {'type': 'array'}}


def to_value(value):
    """A JSON-compatible stand-in for a value (its string if it is not a number, string or boolean)."""
    return value if isinstance(value, (bool, int, float, str, type(None))) else str(value)


def plan_to_json(plan):
    if plan is None:
        return None
    return [[step.name, [to_value(arg) for arg in getattr(step, 'args', getattr(step, 'inputs', ()))]] for step in plan]


##################################################


def describe_problem(workspace, problem_id):
    """What the problem declares: predicates, actions, streams (with their source lines), capabilities, lint issues."""
    problem, stream_info, config = workspace.get_problem(problem_id)
    return describe(problem, stream_info, unit_costs=config.get('unit_costs', False)).to_json()


def list_stream_contracts(workspace, problem_id, name=None):
    """The streams' contracts (inputs, outputs, domain, certified facts, options), with their statistics so far."""
    streams = describe_problem(workspace, problem_id)['streams']
    session = workspace.sessions.get(problem_id)
    for stream in streams:
        if session is not None:
            external = session.get_external(stream['name'])
            stream['statistics'] = {'calls': external.total_calls, 'p_success': round(external.get_p_success(), 3)}
    return [stream for stream in streams if (name is None) or (stream['name'] == name)]


def check_stream(workspace, problem_id, stream, inputs, max_calls=3):
    """Calls a stream's procedure on input values (outside any search), and reports each call's outputs and the
    contract violations found: wrong-input-arity, exception, not-a-sequence, wrong-output-arity, duplicate-outputs."""
    session = workspace.session(problem_id)
    external = session.get_external(stream)
    issues, calls = [], []
    if len(inputs) != len(external.inputs):
        return {'calls': [], 'issues': [f'wrong-input-arity: expected {len(external.inputs)}, got {len(inputs)}']}
    num_outputs = len(external.outputs) if isinstance(external, Stream) else None
    seen = set()
    try:
        generator = external.gen_fn(*inputs) if isinstance(external, Stream) else None
    except Exception as error:
        return {'calls': [], 'issues': [f'exception: {error!r}']}
    for _ in range(max_calls):
        start = time.time()
        try:
            if generator is None:  # A function or predicate
                output_list = [(external.fn(*inputs),)]
            else:
                output_list = next(generator)
        except StopIteration:
            calls.append({'outputs': [], 'exhausted': True, 'time': round(time.time() - start, 4)})
            break
        except Exception as error:
            issues.append(f'exception: {error!r}')
            break
        if isinstance(output_list, WildOutput):
            output_list = output_list.values
        if not isinstance(output_list, (list, tuple)):
            issues.append(f'not-a-sequence: {output_list!r}')
            break
        for outputs in output_list:
            if (num_outputs is not None) and (
                (not isinstance(outputs, (list, tuple))) or (len(outputs) != num_outputs)
            ):
                issues.append(f'wrong-output-arity: expected {num_outputs} values, got {outputs!r}')
            key = repr(outputs)
            if key in seen:
                issues.append(f'duplicate-outputs: {key}')
            seen.add(key)
        calls.append(
            {
                'outputs': [[to_value(v) for v in o] for o in output_list if isinstance(o, (list, tuple))],
                'time': round(time.time() - start, 4),
            }
        )
        if isinstance(generator, BoundedGenerator) and generator.enumerated:
            break
    return {'calls': calls, 'issues': sorted(set(issues))}


def run_search(workspace, problem_id, budget=None):
    """Searches for a plan within a budget; a proposed hint is tried first for part of the budget."""
    budget = dict(budget or {})
    check_keywords(budget, BUDGET_KEYS, 'run_search budget')
    session = workspace.session(problem_id)
    hint = workspace.hints.pop(problem_id, None)
    result = None
    if hint is not None:
        skeleton, fraction = hint
        hint_budget = dict(budget, max_time=fraction * budget.get('max_time', 60))
        constraints = PlanConstraints(skeletons=[skeleton], exact=False)
        result = session.solve(constraints=constraints, **hint_budget)
    if (result is None) or (result.plan is None):  # The hint only ranks: without it, the search is complete again
        result = session.solve(trace=True, **budget)
    run_id = workspace.add_result(problem_id, result)
    stats = {key: result.stats.get(key) for key in ['termination', 'gaps', 'outcomes', 'iterations', 'run_time']}
    return {
        'run_id': run_id,
        'status': result.status.value,
        'plan': plan_to_json(result.plan),
        'cost': result.cost,
        'stats': stats,
    }


def propose_hint(workspace, problem_id, skeleton, budget_fraction=0.5):
    """Asks the next run_search to first look for a plan that follows skeleton ([[action name, [arguments, with
    '?x' for free ones]], ...]). It only orders the search: if no such plan is found, the search continues without
    it."""
    problem, _, _ = workspace.get_problem(problem_id)
    actions = {action.name for action in parse_sequential_domain(problem.domain_pddl).actions}
    unknown = [name for name, _ in skeleton if name.lower() not in actions]
    if unknown:
        return {'accepted_for_ranking': False, 'reason': f'Unknown actions {unknown}'}
    workspace.hints[problem_id] = ([(name, list(args)) for name, args in skeleton], budget_fraction)
    return {'accepted_for_ranking': True, 'reason': f'Tried first for {budget_fraction:.0%} of the next budget'}


def validate_plan(workspace, problem_id, plan=None, run_id=None):
    """Checks a plan: a run's plan (by run_id) through its certificate, or a plan an agent wrote ([[name, [args]],
    ...]) by replaying it from the initial state plus the facts that earlier searches certified. For a failing plan,
    reports the first step whose (non-derived) preconditions do not hold and the missing facts."""
    problem, _, config = workspace.get_problem(problem_id)
    unit_costs = config.get('unit_costs', False)
    if run_id is not None:
        _, result = workspace.get_result(run_id)
        verified, issues = verify_certificate(problem, result, unit_costs=unit_costs)
        return {'valid': verified, 'issues': issues}
    last = workspace.last_result(problem_id)
    facts = list(problem.init) + (list(last.certificate.all_facts) if last is not None else [])
    actions = [Action(name, tuple(args)) for name, args in plan]
    valid, cost = validate_solution(problem, Solution(actions, 0, Certificate(facts, None)), unit_costs=unit_costs)
    if valid:
        return {'valid': True, 'cost': cost}
    return {'valid': False, **first_failure(problem, facts, actions)}


def first_failure(problem, facts, plan):
    """The first step of plan whose non-derived preconditions do not hold, and the missing facts."""
    with solve_scope(registry=Registry()):
        domain = parse_sequential_domain(problem.domain_pddl)
        task = task_from_domain_problem(domain, get_problem(evaluations_from_init(facts), None, domain))
        derived = set(get_derived_predicates(task.axioms))
        state = set(task.init)
        try:
            instances = get_action_instances(task, transform_plan_args(plan, Object.from_value))
        except Exception as error:
            return {'first_failed_step': None, 'missing': [], 'reason': f'Could not instantiate the plan: {error!r}'}
        for step, instance in enumerate(instances):
            missing = [
                literal
                for literal in instance.precondition
                if (literal.predicate not in derived) and not literal_holds(state, literal)
            ]
            if missing:
                facts = [plan_fact(literal) for literal in missing]
                return {'first_failed_step': step, 'missing': facts}
            apply_action(state, instance)
    return {'first_failed_step': None, 'missing': [], 'reason': 'Derived preconditions or the goal do not hold'}


def plan_fact(literal):
    fact = fact_from_evaluation(evaluation_from_fd(literal))
    return [to_value(part) if not isinstance(part, Object) else to_value(part.value) for part in fact]


def explain_failure(workspace, run_id):
    """Why a run ended as it did (see pddlstream.algorithms.explain)."""
    _, result = workspace.get_result(run_id)
    return explain(result).to_dict()


def inspect_trace(workspace, run_id, event=None, offset=0, limit=100):
    """A run's events (optionally only one kind, e.g. 'Call' or 'Iteration'), paginated."""
    _, result = workspace.get_result(run_id)
    events = [e for e in result.trace or [] if (event is None) or (e['event'] == event)]
    return {'total': len(events), 'events': events[offset : offset + limit]}


def profile(workspace, run_id):
    """Where a run's time went, and its stream calls by stream."""
    _, result = workspace.get_result(run_id)
    calls = Counter(e['stream'] for e in result.trace or [] if e['event'] == 'Call')
    keys = ['search_time', 'sample_time', 'run_time', 'iterations', 'skeletons']
    return {'times': {key: result.stats.get(key) for key in keys}, 'calls': dict(calls)}


def run_regressions(workspace, problem_ids, budget=None):
    """Runs a search on each problem and reports their statuses."""
    return {pid: run_search(workspace, pid, budget)['status'] for pid in problem_ids}


TOOLS = [
    Tool(
        'describe_problem', describe_problem.__doc__, schema({'problem_id': STRING}, ['problem_id']), describe_problem
    ),
    Tool(
        'list_stream_contracts',
        list_stream_contracts.__doc__,
        schema({'problem_id': STRING, 'name': STRING}, ['problem_id']),
        list_stream_contracts,
    ),
    Tool(
        'check_stream',
        check_stream.__doc__,
        schema(
            {'problem_id': STRING, 'stream': STRING, 'inputs': {'type': 'array'}, 'max_calls': INTEGER},
            ['problem_id', 'stream', 'inputs'],
        ),
        check_stream,
    ),
    Tool(
        'run_search', run_search.__doc__, schema({'problem_id': STRING, 'budget': BUDGET}, ['problem_id']), run_search
    ),
    Tool(
        'propose_hint',
        propose_hint.__doc__,
        schema({'problem_id': STRING, 'skeleton': PLAN, 'budget_fraction': NUMBER}, ['problem_id', 'skeleton']),
        propose_hint,
    ),
    Tool(
        'validate_plan',
        validate_plan.__doc__,
        schema({'problem_id': STRING, 'plan': PLAN, 'run_id': STRING}, ['problem_id']),
        validate_plan,
    ),
    Tool('explain_failure', explain_failure.__doc__, schema({'run_id': STRING}, ['run_id']), explain_failure),
    Tool(
        'inspect_trace',
        inspect_trace.__doc__,
        schema({'run_id': STRING, 'event': STRING, 'offset': INTEGER, 'limit': INTEGER}, ['run_id']),
        inspect_trace,
    ),
    Tool('profile', profile.__doc__, schema({'run_id': STRING}, ['run_id']), profile),
    Tool(
        'run_regressions',
        run_regressions.__doc__,
        schema({'problem_ids': {'type': 'array', 'items': STRING}, 'budget': BUDGET}, ['problem_ids']),
        run_regressions,
    ),
]


class Toolbox:
    """The tools, bound to a Workspace."""

    def __init__(self, workspace, tools=TOOLS):
        self.workspace = workspace
        self.tools = {tool.name: tool for tool in tools}

    def call(self, name, arguments=None):
        """Calls a tool; returns {'ok': True, 'result'} or {'ok': False, 'error': {'code', 'message'}}."""
        arguments = arguments or {}
        if name not in self.tools:
            return {'ok': False, 'error': {'code': 'unknown-tool', 'message': f'Unknown tool {name!r}'}}
        tool = self.tools[name]
        unknown = set(arguments) - set(tool.input_schema['properties'])
        missing = set(tool.input_schema['required']) - set(arguments)
        if unknown or missing:
            message = f'Unknown arguments {sorted(unknown)}, missing arguments {sorted(missing)}'
            return {'ok': False, 'error': {'code': 'bad-arguments', 'message': message}}
        try:
            return {'ok': True, 'result': tool.fn(self.workspace, **arguments)}
        except Exception as error:
            return {'ok': False, 'error': {'code': type(error).__name__, 'message': str(error)}}
