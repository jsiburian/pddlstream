"""Runs stream procedures in parallel: the same problem solved serially, with a thread pool, and with worker
processes. The calls are committed in the same order each time, so the plans and calls are identical; only the
time changes. Run: python -m examples.executors.run
"""

import logging
import time

from examples.executors.procedures import make_procedures, step
from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import PDDLProblem
from pddlstream.language.executor import ProcessExecutor, ThreadExecutor
from pddlstream.language.generator import from_list_fn, from_test
from pddlstream.language.stream import StreamInfo

DOMAIN_PDDL = """(define (domain chain) (:requirements :strips)
  (:predicates (Num ?x) (Step ?x ?y) (At ?x) (Big ?x) (Done))
  (:action move :parameters (?x ?y) :precondition (and (Step ?x ?y) (At ?x)) :effect (and (At ?y) (not (At ?x))))
  (:action finish :parameters (?x) :precondition (and (At ?x) (Big ?x)) :effect (Done)))"""

STREAM_PDDL = """(define (stream chain)
  (:stream step :inputs (?x) :domain (Num ?x) :outputs (?y) :certified (and (Num ?y) (Step ?x ?y)))
  (:stream big :inputs (?x) :domain (Num ?x) :certified (Big ?x)))"""


def make_problem(step_procedure=from_list_fn(step), threshold=16):
    """Reach a number of at least threshold from 0 in steps of 1 or 3."""
    stream_map = {'step': step_procedure, 'big': from_test(lambda x: x >= threshold)}
    return PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, stream_map, [('Num', 0), ('At', 0)], ('Done',))


def timed_solve(title, **kwargs):
    start = time.perf_counter()
    result = solve(make_problem(), algorithm='incremental', unit_costs=True, verbose=False, **kwargs)
    elapsed, procedures = time.perf_counter() - start, result.stats.get('resources')
    print(f'{title}: {elapsed:.2f}s, {len(result.plan)} steps, procedures {procedures}')
    return result


def main():
    logging.getLogger('pddlstream').setLevel(logging.WARNING)
    serial = timed_solve('Serial')
    with ThreadExecutor(workers=8) as executor:
        threaded = timed_solve('Threads', executor=executor)
    stream_info = {'step': StreamInfo(resource='process')}  # Only the steps run in the workers
    with ProcessExecutor(make_procedures, workers=4, lookahead=8) as executor:
        processes = timed_solve('Processes', executor=executor, stream_info=stream_info)
    assert str(serial.plan) == str(threaded.plan) == str(processes.plan)


if __name__ == '__main__':
    main()
