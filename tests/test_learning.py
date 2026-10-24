"""Learning from solves: traces become dataset rows, and binding policies written against pddlstream.strategy learn
from them (or within a solve)."""

import ast
import pathlib

from examples.continuous_tamp.primitives import blocked
from examples.learning.policies import ConflictPolicy, LearnedPolicy, StreamStatistics
from examples.learning.run import OPTIONS, TRAINING, collect, make_problem
from helpers import output_tail, run_python
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.validate import verify_certificate
from pddlstream.core.problem import Status
from pddlstream.strategy import read_jsonl, rows_from_trace, write_jsonl

POLICIES = pathlib.Path(__file__).resolve().parents[1] / 'examples' / 'learning' / 'policies.py'


def test_rows(tmp_path):
    result = solve(make_problem(lambda: blocked(n_blocks=2)), trace=True, **OPTIONS)
    rows = rows_from_trace(result.trace, problem='blocked-2')
    calls = [row for row in rows if row['kind'] == 'call']
    assert len(calls) == sum(result.stats['outcomes'].values())
    assert all(row['problem'] == 'blocked-2' for row in rows)
    skeletons = [row for row in rows if row['kind'] == 'skeleton']
    assert any(row['bound'] == row['length'] == len(row['streams']) for row in skeletons)  # The one that was solved
    write_jsonl(rows, tmp_path / 'rows.jsonl')
    assert read_jsonl(tmp_path / 'rows.jsonl') == rows


def test_policies_import_only_the_facade():
    modules = {node.module for node in ast.walk(ast.parse(POLICIES.read_text())) if isinstance(node, ast.ImportFrom)}
    assert {module for module in modules if module.startswith('pddlstream')} == {'pddlstream.strategy'}


# Prints the calls of the default and the learned policy on the held-out problem
COMPARE = """
from examples.continuous_tamp.primitives import blocked
from examples.learning.policies import LearnedPolicy, StreamStatistics
from examples.learning.run import OPTIONS, TRAINING, collect, make_problem
from pddlstream.algorithms.meta import solve

statistics = StreamStatistics.from_rows(collect(TRAINING))
for policy in [None, LearnedPolicy(statistics)]:
    result = solve(make_problem(lambda: blocked(n_blocks=4)), binding_policy=policy, **OPTIONS)
    print('calls', sum(result.stats['outcomes'].values()))
"""


def test_learned_policy_on_a_held_out_problem(tmp_path):
    statistics = StreamStatistics.from_rows(collect(TRAINING))
    assert statistics.p_success('s-region') > statistics.p_success('t-cfree')
    problem = make_problem(lambda: blocked(n_blocks=4))
    learned = solve(problem, binding_policy=LearnedPolicy(statistics), **OPTIONS)
    assert learned.status is Status.SUCCEEDED
    assert verify_certificate(problem, learned, unit_costs=True) == (True, [])
    # Call counts repeat only under a fixed PYTHONHASHSEED (the translator iterates string-keyed sets), which this
    # process may not have, so they are compared in a subprocess
    result = run_python(['-c', COMPARE], cwd=tmp_path)
    assert result.returncode == 0, output_tail(result)
    default_calls, learned_calls = [
        int(line.split()[1]) for line in result.stdout.splitlines() if line.startswith('calls ')
    ]
    assert learned_calls < default_calls


def test_conflict_policy():
    problem = make_problem(lambda: blocked(n_blocks=3))
    result = solve(problem, binding_policy=ConflictPolicy(), **OPTIONS)
    assert result.status is Status.SUCCEEDED
    assert verify_certificate(problem, result, unit_costs=True) == (True, [])
