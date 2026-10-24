"""Stream procedures for the executor example. ProcessExecutor workers import this module in fresh processes, so it
lives in a regular package and imports little."""

import time

DELAY = 0.02  # Seconds per call, standing in for an expensive sampler (motion planning, simulation, a GPU model)


def step(x, delay=DELAY):
    """The numbers one and three past x."""
    time.sleep(delay)
    return [(x + 1,), (x + 3,)]


def make_procedures():
    """A ProcessExecutor worker's procedures, by stream name."""
    from pddlstream.language.generator import from_list_fn

    return {'step': from_list_fn(step)}
