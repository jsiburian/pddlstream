"""Planner sessions: plan repeatedly in a changing world, reusing the stream results that a change does not affect."""

import dataclasses
from collections import Counter
from dataclasses import dataclass

from pddlstream.algorithms.algorithm import parse_problem, preload_results, solve_scope
from pddlstream.algorithms.meta import solve
from pddlstream.core.config import SolverConfig
from pddlstream.core.conversion import values_from_objects
from pddlstream.core.evaluations import evaluations_from_init
from pddlstream.core.problem import PDDLProblem
from pddlstream.core.registry import Registry
from pddlstream.core.syntax import get_args, get_prefix
from pddlstream.core.utils import value_or_id
from pddlstream.language.exogenous import FutureStream
from pddlstream.language.optimizer import ComponentStream, OptimizerStream
from pddlstream.language.statistics import load_stream_statistics, write_stream_statistics


@dataclass(frozen=True)
class WorldPatch:
    """A change to the world: facts that now hold, facts that no longer hold, and the names of other things that
    changed (e.g. obstacle geometry that streams read; see StreamInfo(reads=...)). The predicates of the added and
    removed facts count as changed too."""

    add: tuple = ()
    remove: tuple = ()
    changed: tuple = ()


@dataclass
class StreamReport:
    """What a session reused of one stream for its last solve."""

    reused: int = 0  # Results from earlier solves whose domain facts still held, so the solve started from them
    lost: int = 0  # Results from earlier solves whose domain facts no longer held
    invalidated: int = 0  # Called instances (with their results and histories) dropped by changes since the last solve
    replayed: int = 0  # Calls answered from an instance's history
    fresh: int = 0  # Calls that ran the procedure


class SessionReport(dict):
    """A StreamReport by stream name."""

    def __str__(self):
        fields = [field.name for field in dataclasses.fields(StreamReport)]
        rows = [['stream', *fields]] + [[name, *map(str, dataclasses.astuple(r))] for name, r in sorted(self.items())]
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        return '\n'.join('  '.join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows)


def fact_key(fact):
    """A hashable key for a fact with values: predicate names ignore case, and unhashable values compare by identity."""
    return (get_prefix(fact).lower(), *map(value_or_id, get_args(fact)))


class PlannerSession:
    """Plans repeatedly as the world changes, reusing the stream results and histories that the changes leave valid.

    Each solve starts from the current initial facts plus the certified facts of earlier results whose domain facts
    still hold, and instances replay their histories before calling their procedures again. apply() and invalidate()
    drop the instances whose results a change may affect, which a stream declares with StreamInfo(reads=...).
    Stream statistics are loaded when the session starts and saved when it closes.

    Values are matched by equality (unhashable ones, such as numpy arrays, by identity), so a fact that a patch
    removes must use the same value objects as the fact that was added.
    """

    def __init__(self, problem, stream_info=None, config=None, **kwargs):
        self.problem = problem
        self.config = SolverConfig.from_kwargs(config, 'PlannerSession()', **kwargs)
        self.registry = Registry()  # The session's objects, which its stream results refer to
        self.init, self.goal = list(problem.init), problem.goal
        self.report = SessionReport()
        self._invalidated = Counter()
        with self._scope(self.config):
            _, _, _, self.externals = parse_problem(
                problem, stream_info, unit_costs=self.config.unit_costs, unit_efforts=self.config.unit_efforts
            )
        unsupported = [
            e.name for e in self.externals if isinstance(e, (FutureStream, ComponentStream, OptimizerStream))
        ]
        if unsupported:
            raise NotImplementedError(f'Planner sessions do not support exogenous streams or optimizers: {unsupported}')
        if self.config.load_statistics and not self.config.deterministic:
            load_stream_statistics(self.externals)

    def _scope(self, config, reuse=()):
        return solve_scope(config, registry=self.registry, reuse=reuse)

    def get_external(self, name):
        [external] = [external for external in self.externals if external.name == name.lower()]
        return external

    def results(self):
        """The real results of every instance so far."""
        return [
            result
            for external in self.externals
            for instance in external.instances.values()
            for results in instance.results_history
            for result in results
        ]

    def solve(self, goal=None, **kwargs):
        """Solves for goal (by default the last one) from the current initial facts; kwargs override the session's
        configuration. Returns a SolveResult, and sets report to what was reused."""
        if goal is not None:
            self.goal = goal
        calls = []
        config = dataclasses.replace(
            SolverConfig.from_kwargs(self.config, 'PlannerSession.solve()', **kwargs),
            load_statistics=False,
            save_statistics=False,
            on_event=calls.append,
        )
        for external in self.externals:  # Instances whose procedure raised get another chance
            for key in [key for key, instance in external.instances.items() if instance.error is not None]:
                del external.instances[key]
        with self._scope(config):
            previous = self.results()
            lost = preload_results(evaluations_from_init(self.init), previous)
            lost_results = set(lost)
            reused = [result for result in previous if result not in lost_results]
            for external in self.externals:
                external.reset()
            domain_pddl, constant_map, _, _, _, _ = self.problem
            problem = PDDLProblem(domain_pddl, constant_map, list(self.externals), None, self.init, self.goal)
            with self._scope(config, reuse=tuple(reused)):
                result = solve(problem, config=config)
        self.report = self._get_report(reused, lost, calls)
        self._invalidated.clear()
        return result

    def _get_report(self, reused, lost, calls):
        report = SessionReport({external.name: StreamReport() for external in self.externals})
        for result in reused:
            report[result.external.name].reused += 1
        for result in lost:
            report[result.external.name].lost += 1
        for name, num in self._invalidated.items():
            report[name].invalidated += num
        for call in calls:
            stream_report = report[call.instance.external.name]
            if call.fresh:
                stream_report.fresh += 1
            else:
                stream_report.replayed += 1
        return report

    def apply(self, patch):
        """Changes the world by patch and drops the instances that the change may affect. Returns the number of
        called instances dropped by stream name."""
        removed = set(map(fact_key, patch.remove))
        self.init = [fact for fact in self.init if fact_key(fact) not in removed] + list(patch.add)
        changed = {name.lower() for name in patch.changed}
        changed.update(get_prefix(fact).lower() for fact in list(patch.add) + list(patch.remove))
        dropped = Counter()
        if changed:
            for external in self.externals:
                reads = external.info.reads
                if (reads is None) or (changed & {name.lower() for name in reads}):
                    dropped[external.name] += sum(1 for instance in external.instances.values() if instance.history)
                    external.instances.clear()
        self._invalidated.update(dropped)
        return {name: num for name, num in dropped.items() if num}

    def invalidate(self, stream, inputs=None):
        """Drops a stream's instances (only those with these input values, if given), for example after its procedure
        changed. Returns the number of called instances dropped."""
        external = self.get_external(stream)
        keys = [
            key
            for key, instance in external.instances.items()
            if (inputs is None)
            or (tuple(map(value_or_id, values_from_objects(instance.input_objects))) == tuple(map(value_or_id, inputs)))
        ]
        num_called = sum(1 for key in keys if external.instances[key].history)
        for key in keys:
            del external.instances[key]
        self._invalidated[external.name] += num_called
        return num_called

    def close(self):
        """Saves the stream statistics."""
        if self.config.save_statistics:
            write_stream_statistics(self.externals, self.config.verbose)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
