"""The foundation that the rest of pddlstream builds on: expression syntax, problem and result types, objects,
evaluations, configuration and the per-solve context. It depends on nothing else in pddlstream.

The packages form layers, each using only those before it: core, fastdownward (the planner), language (streams,
functions and optimizers), and algorithms (the solvers). tests/test_layers.py checks this.
"""
