"""Runs one example's main() and prints a JSON summary of its solve call.

Usage: python tests/harness.py <example module> [example args...]
e.g.   python tests/harness.py continuous_tamp.run -p blocked -n 3

The harness wraps the solver entry points before the example is imported and records each outermost solve call
(problem, keyword arguments, solution, run time and the solver's summary). PDDLStream plans are then checked with
pddlstream.algorithms.validate; plain PDDL plans and constraint-satisfaction assignments are only recorded.
The last stdout line starting with RESULT_PREFIX holds the JSON result.
"""
import functools
import json
import random
import runpy
import sys
import time
import traceback

import numpy as np

from pddlstream.algorithms import focused, incremental, meta, satisfaction, search, serialized
from pddlstream.algorithms.common import SolutionStore
from pddlstream.algorithms.validate import validate_solution
from pddlstream.language.constants import Action, is_plan
from pddlstream.utils import INF

RESULT_PREFIX = 'HARNESS_RESULT '
PDDLSTREAM, PDDL, SATISFACTION = 'pddlstream', 'pddl', 'satisfaction'
SOLVERS = {  # (module, name): what the solver returns
    **{(meta, name): PDDLSTREAM for name in ['solve']},
    **{(incremental, name): PDDLSTREAM for name in ['solve_incremental', 'solve_immediate', 'solve_exhaustive']},
    **{(focused, name): PDDLSTREAM for name in ['solve_focused', 'solve_focused_original', 'solve_binding',
                                                'solve_adaptive', 'solve_hierarchical']},
    **{(serialized, name): PDDLSTREAM for name in ['solve_serialized']},
    (search, 'solve_from_pddl'): PDDL,  # (plan, cost)
    (satisfaction, 'constraint_satisfaction'): SATISFACTION,  # (bindings, cost, facts)
}


class Recorder:
    def __init__(self):
        self.calls = []
        self.summaries = []
        self.depth = 0

    def wrap(self, solver, kind):
        @functools.wraps(solver)
        def wrapper(problem, *args, **kwargs):
            outermost = (self.depth == 0)
            self.depth += 1
            start_time = time.perf_counter()
            try:
                solution = solver(problem, *args, **kwargs)
            finally:
                self.depth -= 1
            if outermost:
                self.calls.append({'solver': solver.__name__, 'kind': kind, 'problem': problem, 'kwargs': kwargs,
                                   'solution': solution, 'elapsed': time.perf_counter() - start_time,
                                   'summary': self.summaries[-1] if self.summaries else None})
            return solution
        return wrapper

    def install(self):
        for (module, name), kind in SOLVERS.items():
            setattr(module, name, self.wrap(getattr(module, name), kind))
        export_summary = SolutionStore.export_summary
        def record_summary(store):
            summary = export_summary(store)
            self.summaries.append(summary)
            return summary
        SolutionStore.export_summary = record_summary


def str_from_plan(plan, kind):
    if not is_plan(plan):
        return None
    if kind == SATISFACTION:  # An assignment of values to parameters
        return sorted([parameter, str(value)] for parameter, value in plan.items())
    # Plain PDDL plans hold (name, args) pairs; deferred StreamActions and FunctionActions have no args
    return [[action[0], [str(arg) for arg in action[1]]] if isinstance(action, (Action, tuple)) and len(action) == 2
            else str(action) for action in plan]


def to_json(value):
    if isinstance(value, float) and (value == INF):
        return 'inf'
    if isinstance(value, dict):
        return {key: to_json(v) for key, v in value.items()}
    if isinstance(value, (bool, int, float, str)) or (value is None):
        return value
    return str(value)


def summarize_call(call):
    plan, cost = call['solution'][:2]
    valid, replay_cost, validation_error = None, INF, None
    if call['kind'] == PDDLSTREAM:
        try:
            valid, replay_cost = validate_solution(call['problem'], call['solution'],
                                                   unit_costs=call['kwargs'].get('unit_costs', False))
        except Exception:
            validation_error = traceback.format_exc()
    return {
        'solver': call['solver'],
        'kind': call['kind'],
        'solved': is_plan(plan),
        'plan': str_from_plan(plan, call['kind']),
        'length': len(plan) if is_plan(plan) else None,
        'cost': to_json(cost),
        'valid': valid,
        'replay_cost': to_json(replay_cost),
        'validation_error': validation_error,
        'elapsed': call['elapsed'],
        'summary': to_json(call['summary']) if call['summary'] else None,
    }


def main():
    module, *args = sys.argv[1:]
    random.seed(0)
    np.random.seed(0)
    recorder = Recorder()
    recorder.install()
    sys.argv = [module, *args]
    error = None
    try:
        runpy.run_module('examples.' + module, run_name='__main__', alter_sys=True)
    except SystemExit as exception:
        if exception.code not in (None, 0):
            error = 'SystemExit({})'.format(exception.code)
    except Exception:
        error = traceback.format_exc()
    result = {
        'example': ' '.join([module, *args]),
        'error': error,
        'calls': [summarize_call(call) for call in recorder.calls],
    }
    print('\n' + RESULT_PREFIX + json.dumps(result))  # Examples may leave a prompt without a newline


if __name__ == '__main__':
    main()
