"""Solves in different threads do not share objects or solver state, so they give the same results as serial solves.
(Samplers that draw from the global random state are the caller's to keep apart.)"""

from concurrent.futures import ThreadPoolExecutor

from examples.blocksworld.run import get_problem as get_blocksworld
from pddlstream.algorithms.meta import solve
from test_certificates import chain_problem
from test_results import line_problem

PROBLEMS = {
    'blocksworld': lambda: solve(get_blocksworld(), unit_costs=True, verbose=False),
    'chain': lambda: solve(chain_problem(), algorithm='adaptive', deterministic=True, verbose=False),
    'line': lambda: solve(line_problem('l2'), algorithm='incremental', verbose=False),
}


def summary(result):
    return [(action.name, action.args) for action in result.plan], result.cost, result.status


def test_threads_match_serial():
    serial = {name: summary(solve_fn()) for name, solve_fn in PROBLEMS.items()}
    names = 4 * list(PROBLEMS)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda name: summary(PROBLEMS[name]()), names))
    assert results == [serial[name] for name in names]
