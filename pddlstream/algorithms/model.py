"""A read-only description of a problem (its predicates, actions, streams and capabilities) and lint issues, for
documentation, agents and tools. The solvers do not use it.
"""

import copy
import dataclasses
from dataclasses import dataclass, field

from pddlstream.algorithms.advanced import automatically_negate_externals
from pddlstream.algorithms.algorithm import parse_problem, solve_scope
from pddlstream.core.config import SolverConfig
from pddlstream.core.registry import Registry
from pddlstream.core.syntax import EXISTS, FORALL, IMPLY, OR, get_args, get_prefix
from pddlstream.fastdownward.tasks import has_costs, parse_sequential_domain
from pddlstream.fastdownward.translator import pddl
from pddlstream.language.exogenous import FutureStream
from pddlstream.language.function import Function, Predicate
from pddlstream.language.lint import Source, find_sources, lint
from pddlstream.language.optimizer import ComponentStream, OptimizerStream
from pddlstream.language.stream import Stream


@dataclass(frozen=True)
class StreamSpec:
    """A stream, function, predicate or optimizer component, as declared."""

    name: str
    kind: str  # 'stream', 'test', 'function', 'predicate' or 'optimizer'
    inputs: tuple
    outputs: tuple
    domain: tuple  # Facts the inputs must satisfy, as strings
    certified: tuple  # Facts each output tuple satisfies
    fluents: tuple
    info: dict  # The options that matter to users (see StreamInfo)
    source: Source | None = None


@dataclass(frozen=True)
class ActionSpec:
    name: str
    parameters: tuple
    source: Source | None = None


@dataclass(frozen=True)
class ProblemModel:
    """What a problem declares: types, constants, predicates (with arities), actions, derived predicates, functions,
    streams, the goal, its capabilities (language features it uses) and lint issues."""

    types: tuple
    constants: tuple
    predicates: dict
    actions: tuple
    derived: tuple
    functions: tuple
    streams: tuple
    goal: str
    capabilities: tuple
    issues: tuple = field(default=())

    def to_json(self):
        return dataclasses.asdict(self)


def str_from_fact(fact):
    return f'({" ".join(map(str, [get_prefix(fact), *get_args(fact)]))})'


def get_kind(external):
    if isinstance(external, (ComponentStream, OptimizerStream)):
        return 'optimizer'
    if isinstance(external, Predicate):
        return 'predicate'
    if isinstance(external, Function):
        return 'function'
    return 'test' if external.is_test else 'stream'


def get_info(external):
    info = external.info
    options = {'eager': info.eager, 'complete': info.complete, 'reads': info.reads}
    if isinstance(external, Stream):
        options.update(negate=info.negate, defer=getattr(info.defer_fn, '__name__', str(info.defer_fn)))
    return options


def get_stream_spec(external, sources):
    certified = getattr(external, 'certified', ())
    outputs = getattr(external, 'outputs', ())
    return StreamSpec(
        name=external.name,
        kind=get_kind(external),
        inputs=tuple(external.inputs),
        outputs=tuple(outputs),
        domain=tuple(map(str_from_fact, external.domain)),
        certified=tuple(map(str_from_fact, certified)),
        fluents=tuple(getattr(external, 'fluents', ())),
        info=get_info(external),
        source=sources.get(external.name),
    )


CAPABILITY_FROM_KIND = {
    'stream': 'streams',
    'test': 'streams',
    'function': 'functions',
    'predicate': 'predicates',
    'optimizer': 'optimizer',
}


def has_condition(condition, condition_types):
    if isinstance(condition, condition_types):
        return True
    return any(has_condition(part, condition_types) for part in getattr(condition, 'parts', ()))


def has_operator(expression, operators):
    if not isinstance(expression, tuple) or not expression:
        return False
    return (get_prefix(expression) in operators) or any(has_operator(part, operators) for part in expression[1:])


def get_capabilities(domain, externals, goal):
    """The language features a problem uses, named like the feature tags of tests/cases.py. domain is the domain as
    written (before normalization, which compiles quantifiers into axioms)."""
    capabilities = set()
    if domain.axioms:
        capabilities.add('derived')
    if has_costs(domain):
        capabilities.add('costs')
    complex_conditions = (pddl.ExistentialCondition, pddl.UniversalCondition, pddl.Disjunction)
    if (
        any(effect.condition != pddl.Truth() for action in domain.actions for effect in action.effects)
        or any(has_condition(action.precondition, complex_conditions) for action in domain.actions)
        or has_operator(goal, {OR, EXISTS, FORALL, IMPLY})
    ):
        capabilities.add('adl')
    if len(domain.types) > 1:
        capabilities.add('types')
    for external in externals:
        capabilities.update(get_external_capabilities(external))
    return tuple(sorted(capabilities))


def get_external_capabilities(external):
    """The capabilities that a stream, function or optimizer brings to a problem."""
    capabilities = {CAPABILITY_FROM_KIND[get_kind(external)]}
    if isinstance(external, Stream) and external.is_negated:
        capabilities.add('negated')
    if isinstance(external, Stream) and external.is_fluent:
        capabilities.add('fluent')
    if isinstance(external, FutureStream):
        capabilities.add('exogenous')
    if external.name.startswith('rule'):
        capabilities.add('rules')
    return capabilities


def describe(problem, stream_info=None, unit_costs=False):
    """The ProblemModel of a PDDLProblem, parsed with its own objects (so it does not affect solves)."""
    domain_pddl, constant_map, stream_pddl, _, _, goal = problem
    with solve_scope(SolverConfig(), registry=Registry(), reuse=()):
        written = parse_sequential_domain(domain_pddl)
        if not isinstance(domain_pddl, str):  # parse_problem modifies an already parsed domain
            written = copy.deepcopy(written)
        _, goal_exp, domain, externals = parse_problem(problem, stream_info or {}, unit_costs=unit_costs)
        automatically_negate_externals(domain, externals)
        stream_texts = stream_pddl if isinstance(stream_pddl, (list, tuple)) else [stream_pddl]
        sources = find_sources({'domain': domain_pddl, **{f'stream{i}': t for i, t in enumerate(stream_texts)}})
        return ProblemModel(
            types=tuple(sorted(t.name for t in written.types)),
            constants=tuple(constant_map),
            predicates={p.name: p.get_arity() for p in written.predicates if not p.name.startswith(('=', '_'))},
            actions=tuple(
                ActionSpec(a.name, tuple(p.name for p in a.parameters), sources.get(a.name)) for a in written.actions
            ),
            derived=tuple(sorted({axiom.name for axiom in written.axioms})),
            functions=tuple(sorted(f.name for f in getattr(written, 'functions', []))),
            streams=tuple(get_stream_spec(external, sources) for external in externals),
            goal=str(goal_exp),
            capabilities=get_capabilities(written, externals, goal),
            issues=tuple(lint(domain, externals, sources)),
        )
