"""The state of the solve in progress: its configuration and clock.

pddlstream.algorithms.algorithm.solve_scope makes a SolveContext current for the duration of a solve, and code deep
inside the solver reads it with get_config() and get_clock(). Outside a solve, they return the defaults.
The object registries are not part of the context: they are process-wide and cleared after each top-level solve.
"""

import contextlib
import contextvars
import dataclasses
import sys
import time
import warnings
from dataclasses import dataclass

from pddlstream.core.config import DEFAULTS, SolverConfig


class WallClock:
    """Measures time in seconds."""

    unit = 'seconds'

    def now(self):
        return time.time()

    def charge(self, work):
        pass


class WorkClock:
    """Measures time in work units that the solver charges for each stream call and search, so that decisions based
    on elapsed time (sampling budgets, stream overhead estimates) are reproducible."""

    unit = 'work units'

    def __init__(self):
        self.work = 0.0

    def now(self):
        return self.work

    def charge(self, work):
        self.work += work


CALL_WORK = 1.0  # Work units charged per stream or function call
SEARCH_WORK = 10.0  # Work units charged per FastDownward search
BINDING_WORK = 0.1  # Work units charged per processed skeleton binding, so sampling loops always advance

# Module flags that set these fields before SolverConfig existed. Assigning them still works, with a warning.
LEGACY_FLAGS = {
    'load_statistics': ('pddlstream.language.statistics', 'LOAD_STATISTICS'),
    'save_statistics': ('pddlstream.language.statistics', 'SAVE_STATISTICS'),
    'fd_instantiate': ('pddlstream.algorithms.instantiate_task', 'FD_INSTANTIATE'),
    'exogenous_axioms': ('pddlstream.language.exogenous', 'EXOGENOUS_AXIOMS'),
    'exogenous_future_values': ('pddlstream.language.exogenous', 'REPLACE_STREAM'),
}


def apply_legacy_flags(config):
    """config with the fields that are still at their defaults set from any reassigned legacy module flag."""
    changes = {}
    for name, (module_name, flag) in LEGACY_FLAGS.items():
        value = getattr(sys.modules.get(module_name), flag, DEFAULTS[name])
        if (value != DEFAULTS[name]) and (getattr(config, name) == DEFAULTS[name]):
            message = f'Assigning {module_name}.{flag} is deprecated; pass {name}={value!r} to solve() instead'
            warnings.warn(message, FutureWarning, stacklevel=2)
            changes[name] = value
    return dataclasses.replace(config, **changes) if changes else config


@dataclass(frozen=True)
class SolveContext:
    config: SolverConfig
    clock: WallClock | WorkClock


_current = contextvars.ContextVar('pddlstream_solve_context', default=None)
_wall_clock = WallClock()


def get_context():
    """The current solve's context, or None outside a solve."""
    return _current.get()


def get_config():
    context = get_context()
    return apply_legacy_flags(SolverConfig()) if context is None else context.config


def get_clock():
    """The clock that the current solve's time-based decisions use."""
    context = get_context()
    return _wall_clock if context is None else context.clock


@contextlib.contextmanager
def activate(context):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
