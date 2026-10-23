#!/usr/bin/env python

from examples.ipc.rovers.run import dump_plan
from pddlstream.algorithms.search import solve_from_pddl
from pddlstream.language.constants import read_pddl_pair


def main():
    domain_pddl, problem_pddl = read_pddl_pair(__file__)
    plan, cost = solve_from_pddl(domain_pddl, problem_pddl)
    dump_plan(plan, cost)


if __name__ == '__main__':
    main()
