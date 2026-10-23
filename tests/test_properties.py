"""Each example in tests/cases.py must solve within its time budget and return a plan the validator accepts.

GUI cases run under a virtual X server (`pixi run test-gui`); slow and Gurobi cases are opt-in (`-m slow`,
`pixi run -e full test-gurobi`).
"""
import importlib.util

import pytest

from cases import CASES
from helpers import has_xvfb, run_case

TOLERANCE = dict(rel=1e-3, abs=1e-2)  # FastDownward rounds each action cost to 1/get_cost_scale()


def to_param(case):
    marks = [getattr(pytest.mark, mark) for mark in case.marks]
    if case.xfail:
        marks.append(pytest.mark.xfail(reason=case.xfail, strict=True))
    return pytest.param(case, id=case.example, marks=marks)


@pytest.mark.parametrize('case', [to_param(case) for case in CASES])
def test_example(case, tmp_path):
    display = 'gui' in case.marks
    if display and not has_xvfb():
        pytest.skip('xvfb-run is not installed')
    if ('gurobi' in case.marks) and (importlib.util.find_spec('gurobipy') is None):
        pytest.skip('gurobipy is not installed (use the full environment)')
    result = run_case(case.example, cwd=tmp_path, timeout=case.timeout, display=display)
    assert result['error'] is None, result['error']
    assert result['calls'], 'The example never called a solver'
    for call in result['calls']:
        assert call['solved'], call['summary']
        if call['kind'] == 'pddlstream':
            assert call['valid'] == case.valid, call['validation_error']
        if call['valid']:
            assert float(call['replay_cost']) == pytest.approx(float(call['cost']), **TOLERANCE)

    last_call = result['calls'][-1]
    cost = float(last_call['cost'])
    if case.cost is not None:
        assert cost == pytest.approx(case.cost, **TOLERANCE)
    assert cost <= case.max_cost
    if case.length is not None:
        assert last_call['length'] == case.length
