"""The interface to the FastDownward planner (the pinned fork in the `downward` submodule).

`translator` finds the FastDownward build and loads its Python translator on first use, so importing pddlstream
needs no build and changes nothing global. `planner` runs the compiled search on a translated task.
"""
