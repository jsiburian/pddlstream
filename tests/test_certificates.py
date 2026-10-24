"""A plan's certificate lists the stream calls behind its preimage facts, back to the initial facts, and
verify_certificate checks it independently of the solver."""

import pytest

from pddlstream.algorithms.certificate import PlanCertificate
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.validate import verify_certificate
from pddlstream.core.problem import Certificate, PDDLProblem
from pddlstream.language.generator import from_fn
from test_semantics import DOMAIN_PDDL, make_problem

CHAIN_STREAMS = """(define (stream chain)
  (:stream sample :outputs (?y) :certified (Other ?y))
  (:stream extend :inputs (?y) :domain (Other ?y) :outputs (?x) :certified (and (Value ?x) (Good ?x))))"""


def chain_problem():
    stream_map = {'sample': from_fn(lambda: (1,)), 'extend': from_fn(lambda y: (y + 1,))}
    return PDDLProblem(DOMAIN_PDDL, {}, CHAIN_STREAMS, stream_map, [], ('Done',))


@pytest.mark.parametrize('algorithm', ['adaptive', 'binding', 'incremental'])
def test_supports(algorithm):
    problem = chain_problem()
    result = solve(problem, algorithm=algorithm, verbose=False)
    certificate = result.certificate
    assert isinstance(certificate, PlanCertificate) and isinstance(certificate, Certificate)
    all_facts, preimage_facts = certificate  # Still unpacks as before
    assert set(preimage_facts) >= {('value', 2), ('good', 2)}  # Focused solvers also include ('other', 1)
    assert certificate.complete and not certificate.internal
    calls = [(support.stream, support.inputs, support.outputs) for support in certificate.supports]
    assert calls == [('sample', (), (1,)), ('extend', (1,), (2,))]  # Each after the supports of its domain
    assert certificate.supports[1].domain == [('other', 1)]
    assert verify_certificate(problem, result) == (True, [])


def test_tampered_certificate():
    problem = make_problem(from_fn(lambda: (1,)))
    result = solve(problem, algorithm='adaptive', verbose=False)
    certificate = result.certificate
    certificate.supports = [support for support in certificate.supports if support.stream != 'test']
    assert verify_certificate(problem, result) == (
        False,
        [f'Preimage fact {("good", 1)} is neither initial nor certified by a support'],
    )
