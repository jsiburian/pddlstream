"""Executors run stream procedures elsewhere, but commit their calls in the solver's order, so a solve with pure
procedures makes the same calls and finds the same plan with any executor."""

import threading
import time

import pytest

from examples.executors import procedures
from examples.executors.run import main, make_problem
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.validate import verify_certificate
from pddlstream.core.problem import Status
from pddlstream.language.executor import ProcessExecutor, ThreadExecutor
from pddlstream.language.generator import from_batch_fn, from_gen_fn, from_list_fn
from pddlstream.language.stream import StreamInfo

OPTIONS = dict(unit_costs=True, verbose=False, max_time=60, trace=True)


def step(x):
    return procedures.step(x, delay=0)


def get_calls(result):
    return [
        (event['stream'], event['inputs'], event['call_index'], event['outcome'])
        for event in result.trace
        if event['event'] == 'Call'
    ]


class Monitor:
    """A generator procedure that records how many instances run at once, and whether one is entered twice."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active, self.reentered, self.max_active = set(), [], 0

    def gen_fn(self, x):
        for y in [x + 1, x + 3]:
            with self.lock:
                if x in self.active:
                    self.reentered.append(x)
                self.active.add(x)
                self.max_active = max(self.max_active, len(self.active))
            time.sleep(0.01)
            with self.lock:
                self.active.discard(x)
            yield (y,)


@pytest.mark.parametrize('algorithm', ['incremental', 'adaptive'])
def test_threads_commit_in_order(algorithm):
    serial_monitor, threaded_monitor = Monitor(), Monitor()
    serial = solve(make_problem(from_gen_fn(serial_monitor.gen_fn)), algorithm=algorithm, **OPTIONS)
    with ThreadExecutor(workers=4) as executor:
        problem = make_problem(from_gen_fn(threaded_monitor.gen_fn))
        threaded = solve(problem, algorithm=algorithm, executor=executor, **OPTIONS)
    assert threaded.status is Status.SUCCEEDED
    assert verify_certificate(problem, threaded, unit_costs=True) == (True, [])
    assert (serial_monitor.max_active, threaded_monitor.reentered) == (1, [])
    assert threaded.stats['resources']['default']['calls'] == sum(threaded.stats['outcomes'].values())
    if algorithm == 'incremental':  # Adaptive divides its time by the clock, so its calls depend on timing
        assert (str(threaded.plan), get_calls(threaded)) == (str(serial.plan), get_calls(serial))
        assert threaded_monitor.max_active > 1


def test_batch_procedure():
    sizes = []

    def batch_step(input_tuples):
        sizes.append(len(input_tuples))
        return [step(x) for (x,) in input_tuples]

    serial = solve(make_problem(from_batch_fn(batch_step)), algorithm='incremental', **OPTIONS)
    assert set(sizes) == {1}
    sizes.clear()
    with ThreadExecutor(workers=1, lookahead=8) as executor:
        problem = make_problem(from_batch_fn(batch_step))
        batched = solve(problem, algorithm='incremental', executor=executor, **OPTIONS)
    assert max(sizes) > 1
    assert (str(batched.plan), get_calls(batched)) == (str(serial.plan), get_calls(serial))


def test_main_resource():
    threads = set()

    def main_step(x):
        threads.add(threading.current_thread())
        return step(x)

    stream_info = {'step': StreamInfo(resource='main')}
    with ThreadExecutor(workers=4) as executor:
        problem = make_problem(from_list_fn(main_step))
        result = solve(problem, algorithm='incremental', stream_info=stream_info, executor=executor, **OPTIONS)
    assert result.status is Status.SUCCEEDED
    assert threads == {threading.main_thread()}


def test_errors_reach_the_solver():
    def broken_step(x):
        raise ValueError('broken')

    with ThreadExecutor(workers=2) as executor, pytest.raises(ValueError, match='broken') as info:
        solve(make_problem(from_list_fn(broken_step)), algorithm='incremental', executor=executor, **OPTIONS)
    assert any('Raised by step' in note for note in info.value.__notes__)


def test_process_executor():
    def solver_step(x):
        raise AssertionError('step must run in a worker process')

    stream_info = {'step': StreamInfo(resource='process')}
    with ProcessExecutor(procedures.make_procedures, workers=2) as executor:
        problem = make_problem(from_list_fn(solver_step))
        result = solve(problem, algorithm='incremental', stream_info=stream_info, executor=executor, **OPTIONS)
    assert result.status is Status.SUCCEEDED
    assert result.stats['resources']['process']['calls'] > 1
    serial = solve(make_problem(from_list_fn(step)), algorithm='incremental', **OPTIONS)
    assert (str(result.plan), get_calls(result)) == (str(serial.plan), get_calls(serial))


def test_deterministic_needs_serial():
    with ThreadExecutor() as executor, pytest.raises(ValueError, match='serially'):
        solve(make_problem(from_list_fn(step)), deterministic=True, executor=executor)


def test_example(capsys):
    main()
    assert 'Processes' in capsys.readouterr().out
