"""explain() says why a solve ended as it did."""

import json

from pddlstream.algorithms.explain import GAPS, explain
from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import PDDLProblem
from pddlstream.language.generator import from_fn, from_test
from pddlstream.language.stream import StreamInfo
from test_certificates import chain_problem
from test_outcomes import TWO_WAYS_PDDL, TWO_WAYS_STREAMS
from test_semantics import DOMAIN_PDDL, make_problem


def test_success():
    explanation = explain(solve(chain_problem(), algorithm='adaptive', trace=True, verbose=False))
    assert (explanation.status, explanation.termination) == ('succeeded', 'solved')
    assert explanation.supports == {'sample': 1, 'extend': 1}
    assert explanation.skeletons == [{'skeleton': 0, 'bound': 2, 'length': 2, 'next_stream': None}]
    json.dumps(explanation.to_dict())


def test_failing_stream():
    problem = make_problem(from_fn(lambda: (1,)), test=from_test(lambda x: False))
    explanation = explain(solve(problem, algorithm='adaptive', trace=True, verbose=False))
    assert explanation.status == 'infeasible'
    assert explanation.streams['test'] == {'exhausted': 1}
    text = str(explanation)
    assert 'Streams that never produced outputs:\n  - test' in text


def test_gaps():
    problem = make_problem(
        from_fn(lambda: (1,)),
        test=from_test(lambda x: False),
        domain_pddl=TWO_WAYS_PDDL,
        stream_pddl=TWO_WAYS_STREAMS,
        **{'sample-other': from_fn(lambda: (2,))},
    )
    explanation = explain(solve(problem, algorithm='adaptive', max_skeletons=1, trace=True, verbose=False))
    assert (explanation.status, explanation.gaps) == ('failed', ['max_skeletons'])
    assert GAPS['max_skeletons'] in str(explanation)
    [skeleton] = explanation.skeletons
    assert (skeleton['bound'], skeleton['next_stream']) == (1, 'test')  # sample succeeded; its test failed


def test_without_trace():
    problem = make_problem(from_fn(lambda: (1,)))
    result = solve(problem, algorithm='adaptive', stream_info={'sample': StreamInfo(effort=10)}, max_effort=5)
    text = str(explain(result))
    assert GAPS['max_effort'] in text and 'trace=True' in text


def test_plan_without_streams():
    stream_map = {'sample': from_fn(lambda: (1,)), 'extend': from_fn(lambda y: (y + 1,))}
    problem = PDDLProblem(DOMAIN_PDDL, {}, None, stream_map, [('Value', 1), ('Good', 1)], ('Done',))
    explanation = explain(solve(problem, algorithm='adaptive', verbose=False))
    assert (explanation.supports, explanation.assumptions) == ({}, [])


def test_traced_without_calls():
    problem = make_problem(from_fn(lambda: (1,)), test=from_test(lambda x: False))
    result = solve(problem, algorithm='adaptive', max_iterations=1, trace=True, verbose=False)
    assert 'trace=True' not in str(explain(result))
