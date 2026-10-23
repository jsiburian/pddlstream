import dataclasses
import functools
import inspect
import logging
from collections import Counter
from contextlib import contextmanager

from pddlstream.algorithms.common import SOLUTIONS
from pddlstream.algorithms.constraints import add_plan_constraints
from pddlstream.core.config import FIELD_NAMES, SEARCH_FIELDS, check_keywords
from pddlstream.core.context import (
    SolveContext,
    WallClock,
    WorkClock,
    activate,
    apply_legacy_flags,
    get_config,
    get_context,
)
from pddlstream.core.conversion import obj_from_value_expression
from pddlstream.core.evaluations import evaluations_from_init
from pddlstream.core.log import console_output
from pddlstream.core.objects import DebugValue, Object, OptimisticObject
from pddlstream.core.syntax import get_args, get_prefix
from pddlstream.core.utils import INF, reset_creation_hashes
from pddlstream.fastdownward.tasks import (
    has_costs,
    normalize_domain_goal,
    parse_goal,
    parse_lisp,
    parse_sequential_domain,
    set_unit_costs,
)
from pddlstream.language.exogenous import FutureValue, compile_to_exogenous
from pddlstream.language.external import External
from pddlstream.language.function import parse_function, parse_predicate
from pddlstream.language.optimizer import parse_optimizer
from pddlstream.language.rule import RULES, apply_rules_to_streams, parse_rule
from pddlstream.language.stream import Stream, StreamInstance, parse_stream

logger = logging.getLogger(__name__)

# TODO: rename file to parsing


def parse_constants(domain, constant_map):
    obj_from_constant = {}
    for constant in domain.constants:
        if constant.name.startswith(Object._prefix):  # TODO: check other prefixes
            raise NotImplementedError(f'Constants are not currently allowed to begin with {Object._prefix}')
        if constant.name not in constant_map:
            raise ValueError(f'Undefined constant {constant.name}')
        value = constant_map.get(constant.name, constant.name)
        obj_from_constant[constant.name] = Object(value, name=constant.name)  # TODO: remap names
        # TODO: add object predicate

    for name in constant_map:
        for constant in domain.constants:
            if constant.name == name:
                break
        else:
            raise ValueError(f'Constant map value {name} not mentioned in domain :constants')
    del domain.constants[:]  # So not set twice
    return obj_from_constant


def check_problem(domain, streams, obj_from_constant):
    for action in domain.actions + domain.axioms:
        for p, c in Counter(action.parameters).items():
            if c != 1:
                raise ValueError(f'Parameter [{p.name}] for action [{action.name}] is not unique')
        # TODO: check that no undeclared parameters & constants

    undeclared_predicates = set()
    for stream in streams:
        # TODO: domain.functions
        facts = list(stream.domain)
        if isinstance(stream, Stream):
            facts.extend(stream.certified)
        for fact in facts:
            name = get_prefix(fact)
            if name not in domain.predicate_dict:
                undeclared_predicates.add(name)
            elif len(get_args(fact)) != domain.predicate_dict[name].get_arity():  # predicate used with wrong arity: {}
                logger.warning(f'predicate used with wrong arity in stream [{stream.name}]: {fact}')
    if undeclared_predicates:
        logger.warning(f'Undeclared predicates: {sorted(undeclared_predicates)}')  # Undeclared predicate: {}


def reset_registries():
    # TODO: maintain these dictionaries in an object
    Object.reset()
    OptimisticObject.reset()
    DebugValue._output_counts.clear()
    FutureValue._output_counts.clear()
    RULES[:] = []
    reset_creation_hashes()


def reset_globals():
    reset_registries()
    SOLUTIONS[:] = []


@contextmanager
def solve_scope(config=None):
    """Makes config (by default the current one) the current solve's configuration while solving. With
    config.deterministic, decisions based on elapsed time use work units instead of seconds (max_time is still
    wall-clock time). Nested solves keep the outer solve's clock. After the outermost solve, the object registries
    are cleared, so every solve starts from the same state."""
    config = apply_legacy_flags(config or get_config())
    outer = get_context()
    if outer is not None:
        with activate(SolveContext(config, outer.clock)):
            yield
        return
    try:
        with activate(SolveContext(config, WorkClock() if config.deterministic else WallClock())), console_output():
            yield
    finally:
        # Cleared afterwards rather than beforehand because callers may create objects before solving (satisfaction)
        reset_registries()


def scoped_solve(solve_fn):
    """Runs a solver inside solve_scope, with the current configuration updated by the solver's arguments.
    Keyword arguments that are neither the solver's parameters nor search options raise a TypeError up front, rather
    than deep inside the search."""
    signature = inspect.signature(solve_fn)
    var_keyword = [name for name, p in signature.parameters.items() if p.kind is inspect.Parameter.VAR_KEYWORD]
    valid = [name for name in signature.parameters if name not in var_keyword] + SEARCH_FIELDS

    @functools.wraps(solve_fn)
    def wrapper(problem, *args, **kwargs):
        check_keywords(kwargs, valid, f'{solve_fn.__name__}()')
        bound = signature.bind(problem, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        for name in var_keyword:
            arguments.update(arguments.pop(name))
        config = dataclasses.replace(
            get_config(), **{name: arguments[name] for name in FIELD_NAMES if name in arguments}
        )
        with solve_scope(config):
            return solve_fn(problem, *args, **kwargs)

    return wrapper


def parse_problem(problem, stream_info={}, constraints=None, unit_costs=False, unit_efforts=False):
    # TODO: just return the problem if already written programmatically
    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = problem

    domain = parse_sequential_domain(domain_pddl)  # TODO: normalize here
    if len(domain.types) != 1:
        raise NotImplementedError('Types are not currently supported')
    if unit_costs:
        set_unit_costs(domain)
    if not has_costs(domain):
        # TODO: set effort_weight to 1 if no costs
        logger.warning('All actions have no cost. Recommend setting unit_costs=True')
    obj_from_constant = parse_constants(domain, constant_map)  # Keep before parse_stream_pddl

    streams = parse_stream_pddl(
        stream_pddl, stream_map, stream_info=stream_info, unit_costs=unit_costs, unit_efforts=unit_efforts
    )
    check_problem(domain, streams, obj_from_constant)

    evaluations = evaluations_from_init(init)
    goal_exp = obj_from_value_expression(goal)

    goal_exp = add_plan_constraints(constraints, domain, evaluations, goal_exp)
    parse_goal(goal_exp, domain)  # Just to check that it parses
    normalize_domain_goal(domain, goal_exp)  # TODO: does not normalize goal_exp

    compile_to_exogenous(evaluations, domain, streams)
    return evaluations, goal_exp, domain, streams


##################################################


def parse_streams(streams, rules, stream_pddl, procedure_map, procedure_info, use_functions=True):
    stream_iter = iter(parse_lisp(stream_pddl))
    assert 'define' == next(stream_iter)
    pddl_type, pddl_name = next(stream_iter)
    assert 'stream' == pddl_type
    for lisp_list in stream_iter:
        name = lisp_list[0]  # TODO: refactor at this point
        if name == ':stream':
            externals = [parse_stream(lisp_list, procedure_map, procedure_info)]
        elif name == ':rule':
            externals = [parse_rule(lisp_list, procedure_map, procedure_info)]
        elif name == ':function':
            if not use_functions:
                continue
            externals = [parse_function(lisp_list, procedure_map, procedure_info)]
        elif name == ':predicate':  # Cannot just use args if want a bound
            externals = [parse_predicate(lisp_list, procedure_map, procedure_info)]
        elif name == ':optimizer':
            externals = parse_optimizer(lisp_list, procedure_map, procedure_info)
        else:
            raise ValueError(name)
        for external in externals:
            if any(e.name == external.name for e in streams):
                raise ValueError(f'Stream [{external.name}] is not unique')
            if name == ':rule':
                rules.append(external)
            external.pddl_name = pddl_name  # TODO: move within constructors
            streams.append(external)


def set_unit_efforts(externals):
    for external in externals:
        if external.get_effort() < INF:
            external.info.effort = 1


def parse_stream_pddl(stream_pddl, stream_map, stream_info={}, unit_costs=False, unit_efforts=False):
    if stream_info is None:  # NO_INFO
        stream_info = {}
    externals = []
    if stream_pddl is None:
        return externals  # No streams
    if isinstance(stream_pddl, str):
        stream_pddl = [stream_pddl]
    if all(isinstance(e, External) for e in stream_pddl):
        return stream_pddl
    if isinstance(stream_map, dict):  # DEBUG_MODES
        stream_map = {k.lower(): v for k, v in stream_map.items()}
    stream_info = {k.lower(): v for k, v in stream_info.items()}
    rules = []
    for pddl in stream_pddl:
        # TODO: check which functions are actually used and prune the rest
        parse_streams(externals, rules, pddl, stream_map, stream_info, use_functions=not unit_costs)
    apply_rules_to_streams(rules, externals)
    if unit_efforts:
        set_unit_efforts(externals)
    return externals


##################################################


def remove_blocked(evaluations, domain, instance, new_results):
    # TODO: finish refactoring this
    if new_results and isinstance(instance, StreamInstance):
        instance.enable(evaluations, domain)
