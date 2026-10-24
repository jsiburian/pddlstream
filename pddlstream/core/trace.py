"""Solver events: what happens during a solve, passed to SolverConfig.on_event as it happens and, with trace=True,
returned as SolveResult.trace (a list of dicts with values as strings).

The events are stream calls (language.external.Call), Iteration, SkeletonAdded, Incumbent (core.problem) and, at the
end of a solve with skeletons, SkeletonProgress.
"""

from typing import NamedTuple

from pddlstream.core.context import get_context


class Iteration(NamedTuple):
    """One iteration of a solver's main loop: its search with the facts within the complexity limit."""

    number: int
    complexity: float
    plan: str  # What the search found: 'plan', 'failed' or 'infeasible'
    stream_plan: int  # Number of stream calls the plan relies on
    action_plan: int  # Number of actions


class SkeletonAdded(NamedTuple):
    """A plan skeleton the adaptive algorithm will try to bind: an optimistic plan and the streams it relies on."""

    skeleton: int  # Its number
    stream_plan: list
    action_plan: list
    cost: float


class SkeletonProgress(NamedTuple):
    """How far the solve got in binding a skeleton: its best binding binds bound of length streams."""

    skeleton: int
    bound: int
    length: int
    next_stream: str | None  # The first stream its best binding has not bound, if any


def emit(event):
    """Records event in the current solver run's trace (with trace=True) and passes it to on_event."""
    context = get_context()
    if context is None:
        return
    if context.config.trace:
        context.record.events.append(event)
    if context.config.on_event is not None:
        context.config.on_event(event)


def to_string(value):
    return value if isinstance(value, str) else str(value)


def event_to_dict(event):
    """An event as JSON-compatible data."""
    name = type(event).__name__
    if name == 'Call':
        external = event.instance.external
        return {
            'event': name,
            'stream': external.name,
            'inputs': [to_string(obj.value) for obj in event.instance.input_objects],
            'call_index': event.call_index,
            'outcome': event.outcome.value,
            'results': len(event.results),
            'facts': len(event.facts),
            'fresh': event.fresh,
        }
    data = {'event': name}
    for field, value in event._asdict().items():
        if isinstance(value, (list, tuple)):
            value = [to_string(item) for item in value]
        elif not isinstance(value, (bool, int, float, str, type(None))):
            value = to_string(value)
        data[field] = value
    return data
