"""The objects of one planning session: the tables that map values and names to objects, and the counters that name
objects and order their hashes.

Each thread (more precisely, each context) has its own current Registry, so concurrent solves in different threads
do not share objects. solve_scope clears the current registry after the outermost solve, unless the caller (e.g. a
planner session) supplies its own registry to keep.
"""

import contextlib
import contextvars
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count


@dataclass
class Registry:
    obj_from_id: dict = field(default_factory=dict)  # Object by id(value), for unhashable values
    obj_from_value: dict = field(default_factory=dict)  # Object by value
    obj_from_name: dict = field(default_factory=dict)  # Object by PDDL name
    opt_from_inputs: dict = field(default_factory=dict)  # OptimisticObject by (value, param)
    opt_from_name: dict = field(default_factory=dict)  # OptimisticObject by PDDL name
    opt_counts: dict = field(default_factory=dict)  # Counter by name prefix, for naming optimistic objects
    output_counts: defaultdict = field(default_factory=lambda: defaultdict(lambda: defaultdict(count)))  # By class
    hashes: count = field(default_factory=count)  # Creation order, which CreationHash uses as the hash

    def clear(self):
        for name, value in Registry().__dict__.items():
            setattr(self, name, value)


_current: contextvars.ContextVar[Registry] = contextvars.ContextVar('pddlstream_registry')


def get_registry():
    """The current context's registry (created on first use)."""
    try:
        return _current.get()
    except LookupError:
        registry = Registry()
        _current.set(registry)
        return registry


@contextlib.contextmanager
def use_registry(registry):
    token = _current.set(registry)
    try:
        yield registry
    finally:
        _current.reset(token)
