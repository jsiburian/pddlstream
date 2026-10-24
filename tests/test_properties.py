"""Each example in tests/cases.py must solve within its time budget and return a plan the validator accepts.

Deterministic cases solve with deterministic=True and must also reproduce their recorded output
(tests/recorded_outputs.json): the actions, cost, stream calls and search counters of every solve. After an intended
behavior change, rerun the affected cases with `--record` and review the diff of that file.

GUI cases run under a virtual X server (`pixi run test-gui`); slow and Gurobi cases are opt-in (`-m slow`,
`pixi run -e full test-gurobi`).
"""

import importlib.util
import json
import pathlib

import pytest

from cases import CASES
from helpers import has_xvfb, run_case

TOLERANCE = dict(rel=1e-3, abs=1e-2)  # FastDownward rounds each action cost to 1/get_cost_scale()
RECORDED_PATH = pathlib.Path(__file__).with_name('recorded_outputs.json')
COUNTERS = ['iterations', 'evaluations', 'solutions', 'complexity', 'skeletons']


def to_param(case):
    marks = [getattr(pytest.mark, mark) for mark in case.marks]
    if case.xfail:
        marks.append(pytest.mark.xfail(reason=case.xfail, strict=True))
    return pytest.param(case, id=case.example, marks=marks)


def recorded_output(result):
    """The parts of a deterministic run that refactoring must not change."""
    output = []
    for call in result['calls']:
        cost = call['cost']
        summary = call['summary'] or {}
        output.append(
            {
                'actions': [step[0] if isinstance(step, list) else step for step in call['plan'] or []],
                'cost': round(cost, 6) if isinstance(cost, float) else cost,
                'stream_calls': call['stream_calls'],
                'digest': call['digest'],
                **{name: summary[name] for name in COUNTERS if name in summary},
            }
        )
    return output


def check_recorded_output(case, result, record):
    recorded = json.loads(RECORDED_PATH.read_text()) if RECORDED_PATH.exists() else {}
    output = recorded_output(result)
    if record:
        recorded[case.example] = output
        RECORDED_PATH.write_text(json.dumps(recorded, indent=1, sort_keys=True) + '\n')
    elif case.example in recorded:
        assert output == recorded[case.example], 'The run differs from its recorded output'


# Plans that may rely on facts that no stream has certified yet, so their certificates need not be complete
UNCERTIFIED = {'deferred', 'optimizer', 'exogenous'}


@pytest.mark.parametrize('case', [to_param(case) for case in CASES])
def test_example(case, tmp_path, record):
    display = 'gui' in case.marks
    if display and not has_xvfb():
        pytest.skip('xvfb-run is not installed')
    if ('gurobi' in case.marks) and (importlib.util.find_spec('gurobipy') is None):
        pytest.skip('gurobipy is not installed (use the full environment)')
    result = run_case(
        case.example, cwd=tmp_path, timeout=case.timeout, display=display, wall_clock=not case.deterministic
    )
    assert result['error'] is None, result['error']
    assert result['calls'], 'The example never called a solver'
    for call in result['calls']:
        assert call['solved'], call['summary']
        if call['kind'] == 'pddlstream':
            assert not call['violations'], call['violations']
            missing = set(call['capabilities'] or []) - set(case.features)
            assert not missing, f'Feature tags missing from tests/cases.py: {sorted(missing)}'
            assert call['valid'] == case.valid, call['validation_error']
        if call['valid']:
            assert float(call['replay_cost']) == pytest.approx(float(call['cost']), **TOLERANCE)
            assert call['certified'] is not False, call['certificate_issues']
            if not (UNCERTIFIED & set(case.features)) and (case.uncertified is None):
                assert call['certified'], call['certificate_issues']

    last_call = result['calls'][-1]
    cost = float(last_call['cost'])
    if case.cost is not None:
        assert cost == pytest.approx(case.cost, **TOLERANCE)
    assert cost <= case.max_cost
    if case.length is not None:
        assert last_call['length'] == case.length
    if case.deterministic:
        check_recorded_output(case, result, record)
