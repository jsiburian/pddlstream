"""The interface to the FastDownward planner (the pinned fork in the `downward` submodule); the rest of pddlstream
reaches the planner and its translator only through this package.

translator loads the translator on first use, so importing pddlstream needs no build and changes nothing global.
tasks converts between pddlstream's facts and the translator's PDDL objects, grounding grounds tasks and translates
them to SAS+, planner runs the compiled search, and search solves PDDL and SAS+ tasks with it.
"""
