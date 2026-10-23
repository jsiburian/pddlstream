"""Moved to pddlstream.fastdownward.grounding; this module re-exports it for existing code."""

from pddlstream.fastdownward.grounding import *  # noqa: F403

# Deprecated: pass fd_instantiate=False to solve() instead of assigning this (it was previously ignored)
FD_INSTANTIATE = True
