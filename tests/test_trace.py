"""With trace=True a solve returns its events (core/trace.py), and on_event receives them as they happen."""

import json

from pddlstream.algorithms.meta import solve
from test_certificates import chain_problem


def solve_traced(**kwargs):
    return solve(chain_problem(), algorithm='adaptive', deterministic=True, trace=True, verbose=False, **kwargs)


def test_no_trace_by_default():
    assert solve(chain_problem(), verbose=False).trace is None


def test_trace():
    trace = solve_traced().trace
    json.dumps(trace)  # Plain data
    events = [event['event'] for event in trace]
    assert set(events) == {'Iteration', 'Call', 'SkeletonAdded', 'Incumbent', 'SkeletonProgress'}
    calls = [(event['stream'], event['inputs'], event['outcome']) for event in trace if event['event'] == 'Call']
    assert calls == [('sample', [], 'yielded'), ('extend', ['1'], 'yielded')]
    [progress] = [event for event in trace if event['event'] == 'SkeletonProgress']
    assert (progress['bound'], progress['length'], progress['next_stream']) == (2, 2, None)


def test_trace_is_deterministic():
    without_times = [[{k: v for k, v in event.items() if k != 'time'} for event in solve_traced().trace] for _ in '12']
    assert without_times[0] == without_times[1]


def test_on_event():
    events = []
    solve(chain_problem(), algorithm='incremental', on_event=events.append, verbose=False)
    assert {type(event).__name__ for event in events} == {'Iteration', 'Call', 'Incumbent'}
