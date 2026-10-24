"""describe() gives a problem's model, capabilities and lint issues."""

import json

import pytest

from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.model import describe
from pddlstream.core.problem import PDDLProblem
from pddlstream.language.external import DEBUG
from pddlstream.language.generator import from_gen_fn
from pddlstream.language.lint import CapabilityError, Source
from test_certificates import chain_problem
from test_semantics import DOMAIN_PDDL
from test_types import typed_problem


def test_describe():
    model = describe(chain_problem())
    assert [(stream.name, stream.kind, stream.inputs, stream.outputs) for stream in model.streams] == [
        ('sample', 'stream', (), ('?y',)),
        ('extend', 'stream', ('?y',), ('?x',)),
    ]
    assert model.streams[1].certified == ('(value ?x)', '(good ?x)')
    assert model.streams[1].source == Source('stream0', 3)
    assert model.capabilities == ('streams',)
    assert [(action.name, action.parameters) for action in model.actions] == [('finish', ('?x',))]
    json.dumps(model.to_json())


def test_describe_typed():
    model = describe(typed_problem(('On', 'a', 'b')))
    assert (model.types, model.capabilities) == (('block', 'item', 'object'), ('adl', 'types'))


def test_lint():
    streams = """(define (stream lint)
  (:stream sample :outputs (?x) :certified (Value ?x ?x))
  (:stream extend :inputs (?y) :domain (Unknown ?y) :outputs (?x) :certified (Good ?x)))"""
    problem = PDDLProblem(DOMAIN_PDDL, {}, streams, DEBUG, [], ('Done',))
    issues = describe(problem).issues
    assert [(issue.code, issue.subject, issue.source) for issue in issues] == [
        ('wrong-arity', 'sample', Source('stream0', 2)),
        ('undeclared-predicate', 'unknown', None),
    ]


def test_capability_error():
    streams = """(define (stream fluent)
  (:stream sample :fluents (Done) :outputs (?x) :certified (Value ?x)))"""
    problem = PDDLProblem(DOMAIN_PDDL, {}, streams, {'sample': from_gen_fn(lambda fluents=[]: iter([]))}, [], ('Done',))
    with pytest.raises(CapabilityError, match='fluent streams'):
        solve(problem, algorithm='incremental', verbose=False)
    with pytest.raises(NotImplementedError):  # CapabilityError is still a NotImplementedError
        solve(problem, algorithm='incremental', verbose=False)
