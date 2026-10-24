"""docs/v2/features.md lists, for each language feature, the example runs that exercise it (tests/cases.py).
After changing the cases, rerun with `--record` to regenerate it."""

import pathlib

from cases import CASES, FEATURES

FEATURES_PATH = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'v2' / 'features.md'
HEADER = """# Feature matrix

Generated from `tests/cases.py` by `pytest tests/test_features.py --record`. Each example run is tested by
`tests/test_properties.py`; marks say which suite runs it (gui, slow, gurobi), and known failures are struck out.
"""


def render_case(case):
    marks = ', '.join(case.marks)
    name = f'`{case.example}`' + (f' ({marks})' if marks else '')
    return f'~~{name}~~' if case.xfail else name


def render_features():
    lines = [HEADER, '| Feature | Example runs |', '|---|---|']
    for feature, description in FEATURES.items():
        cases = [render_case(case) for case in CASES if feature in case.features]
        lines.append(f'| **{feature}**: {description} | {"<br>".join(cases) or "none"} |')
    return '\n'.join(lines) + '\n'


def test_known_features():
    for case in CASES:
        assert case.features, f'{case.example} has no features'
        assert set(case.features) <= set(FEATURES), case.example


def test_matrix_is_current(record):
    text = render_features()
    if record:
        FEATURES_PATH.write_text(text)
    assert FEATURES_PATH.read_text() == text, 'docs/v2/features.md is stale; rerun with --record'
