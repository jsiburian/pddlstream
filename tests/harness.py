"""Runs one example's main() and prints a JSON summary of its solve calls.

Usage: python tests/harness.py [--wall-clock] <example module> [example args...]
e.g.   python tests/harness.py continuous_tamp.run -p blocked -n 3

The harness wraps the solver entry points before the example is imported. Each outermost PDDLStream solve runs with
deterministic=True (unless --wall-clock) inside its own solve scope. Within that scope, before the object registries
are cleared, its plan is checked with pddlstream.algorithms.validate, its certificate is verified, and the solution is
checked against invariants (check_solution). Plain PDDL plans and constraint-satisfaction assignments are only
recorded. Every stream and function call is logged by stream name and input object names, so the digest of the log
identifies a run. The last stdout line starting with RESULT_PREFIX holds the JSON result.
"""

import argparse
import functools
import hashlib
import json
import random
import runpy
import sys
import time
import traceback
from collections import Counter

import numpy as np

from pddlstream.algorithms import focused, incremental, meta, satisfaction, search, serialized
from pddlstream.algorithms.algorithm import solve_scope
from pddlstream.algorithms.common import SolutionStore
from pddlstream.algorithms.validate import validate_solution, verify_certificate
from pddlstream.core.config import SolverConfig
from pddlstream.core.objects import OPT_PREFIX
from pddlstream.core.problem import Status, StreamAction
from pddlstream.language.constants import Action, is_plan
from pddlstream.language.external import Instance
from pddlstream.utils import INF

RESULT_PREFIX = 'HARNESS_RESULT '
PDDLSTREAM, PDDL, SATISFACTION = 'pddlstream', 'pddl', 'satisfaction'
SOLVERS = {  # (module, name): what the solver returns
    **{(meta, name): PDDLSTREAM for name in ['solve']},
    **{(incremental, name): PDDLSTREAM for name in ['solve_incremental', 'solve_immediate', 'solve_exhaustive']},
    **{
        (focused, name): PDDLSTREAM
        for name in ['solve_focused', 'solve_focused_original', 'solve_binding', 'solve_adaptive', 'solve_hierarchical']
    },
    **{(serialized, name): PDDLSTREAM for name in ['solve_serialized']},
    (search, 'solve_from_pddl'): PDDL,  # (plan, cost)
    (satisfaction, 'constraint_satisfaction'): SATISFACTION,  # (bindings, cost, facts)
}


def str_from_plan(plan, kind):
    if not is_plan(plan):
        return None
    if kind == SATISFACTION:  # An assignment of values to parameters
        return sorted([parameter, str(value)] for parameter, value in plan.items())
    # Plain PDDL plans hold (name, args) pairs; deferred StreamActions and FunctionActions have no args
    return [
        [action[0], [str(arg) for arg in action[1]]]
        if isinstance(action, (Action, tuple)) and len(action) == 2
        else str(action)
        for action in plan
    ]


def to_json(value):
    if isinstance(value, float) and (value == INF):
        return 'inf'
    if isinstance(value, dict):
        return {key: to_json(v) for key, v in value.items()}
    if isinstance(value, (bool, int, float, str)) or (value is None):
        return value
    return str(value)


def check_solution(solution):
    """Violations of what every returned solution must satisfy: its status matches whether it has a plan, and a
    placeholder (optimistic object) in an action is an output of an earlier deferred stream step."""
    plan, violations, bound = solution[0], [], set()
    if (solution.status is Status.SUCCEEDED) != is_plan(plan):
        violations.append(f'Status {solution.status} with plan {plan}')
    for step in plan if is_plan(plan) else []:
        args = step.inputs if isinstance(step, StreamAction) else getattr(step, 'args', ())
        unbound = [arg for arg in args if isinstance(arg, str) and arg.startswith(OPT_PREFIX) and arg not in bound]
        if unbound:
            violations.append(f'{step.name} uses placeholders {unbound} before a stream binds them')
        if isinstance(step, StreamAction):
            bound.update(step.outputs)
    return violations


class Recorder:
    def __init__(self, deterministic):
        self.deterministic = deterministic
        self.calls = []
        self.summaries = []
        self.stream_calls = []
        self.depth = 0

    def wrap(self, solver, kind):
        @functools.wraps(solver)
        def wrapper(problem, *args, **kwargs):
            if self.depth:  # Only the outermost call is recorded
                return solver(problem, *args, **kwargs)
            self.depth += 1
            try:
                if kind != PDDLSTREAM:
                    return self.record(solver, kind, problem, args, kwargs)
                kwargs.setdefault('deterministic', self.deterministic)
                # Keeps the objects alive until the plan is validated
                with solve_scope(SolverConfig(deterministic=self.deterministic)):
                    return self.record(solver, kind, problem, args, kwargs)
            finally:
                self.depth -= 1

        return wrapper

    def record(self, solver, kind, problem, args, kwargs):
        num_summaries, first_stream_call = len(self.summaries), len(self.stream_calls)
        start_time = time.perf_counter()
        solution = solver(problem, *args, **kwargs)
        elapsed = time.perf_counter() - start_time
        summary = self.summaries[-1] if len(self.summaries) > num_summaries else None
        stream_calls = self.stream_calls[first_stream_call:]
        plan, cost = solution[:2]
        valid, replay_cost, validation_error, violations = None, INF, None, []
        certified, certificate_issues = None, []
        if kind == PDDLSTREAM:
            violations = check_solution(solution)
            unit_costs = kwargs.get('unit_costs', getattr(kwargs.get('config'), 'unit_costs', False))
            try:
                valid, replay_cost = validate_solution(problem, solution, unit_costs=unit_costs)
                if valid:
                    certified, certificate_issues = verify_certificate(problem, solution, unit_costs=unit_costs)
            except Exception:
                validation_error = traceback.format_exc()
        self.calls.append(
            {
                'solver': solver.__name__,
                'kind': kind,
                'solved': is_plan(plan),
                'plan': str_from_plan(plan, kind),
                'length': len(plan) if is_plan(plan) else None,
                'cost': to_json(cost),
                'valid': valid,
                'replay_cost': to_json(replay_cost),
                'validation_error': validation_error,
                'violations': violations,
                'certified': certified,
                'certificate_issues': certificate_issues,
                'elapsed': elapsed,
                'summary': to_json(summary) if summary else None,
                'stream_calls': dict(sorted(Counter(name for name, _ in stream_calls).items())),
                'digest': hashlib.sha256(repr(stream_calls).encode()).hexdigest()[:16],
            }
        )
        return solution

    def install(self):
        for (module, name), kind in SOLVERS.items():
            setattr(module, name, self.wrap(getattr(module, name), kind))

        export_summary = SolutionStore.export_summary

        def record_summary(store):
            summary = export_summary(store)
            self.summaries.append(summary)
            return summary

        SolutionStore.export_summary = record_summary

        # Every stream and function call that runs the user's procedure updates the statistics once
        update_statistics = Instance.update_statistics

        def record_stream_call(instance, *args, **kwargs):
            self.stream_calls.append((instance.external.name, tuple(obj.pddl for obj in instance.input_objects)))
            return update_statistics(instance, *args, **kwargs)

        Instance.update_statistics = record_stream_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wall-clock', action='store_true', help='Solves with deterministic=False')
    parser.add_argument('example', nargs=argparse.REMAINDER, help='The example module and its arguments')
    args = parser.parse_args()
    module, *example_args = args.example
    random.seed(0)
    np.random.seed(0)
    recorder = Recorder(deterministic=not args.wall_clock)
    recorder.install()
    sys.argv = [module, *example_args]
    error = None
    try:
        runpy.run_module('examples.' + module, run_name='__main__', alter_sys=True)
    except SystemExit as exception:
        if exception.code not in (None, 0):
            error = f'SystemExit({exception.code})'
    except Exception:
        error = traceback.format_exc()
    result = {'example': ' '.join(args.example), 'error': error, 'calls': recorder.calls}
    print('\n' + RESULT_PREFIX + json.dumps(result))  # Examples may leave a prompt without a newline


if __name__ == '__main__':
    main()
