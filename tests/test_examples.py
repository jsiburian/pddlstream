"""Smoke tests: each headless example must run to completion and report a solution."""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

HEADLESS_EXAMPLES = [
    'blocksworld.run -a incremental',
    'blocksworld.run -a adaptive',
    'blocksworld.run -a focused',
    'blocksworld.run -a binding',
    'blocksworld.run_derived',
    'kitchen.run',
    'advanced.adl.run',
    'advanced.counting.run',
    'advanced.defer.run',
    'advanced.exogenous.run',
    'advanced.fluent.run',
    'advanced.multi_cost',
    'advanced.satisfy.run',
    'advanced.storage',
    'advanced.wild.run',
    'temporal.run',
    'ipc.rovers.run',
    'ipc.satellites.run',
]


def run_example(args, cwd, timeout=600):
    module, *flags = args.split()
    python_path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))
    return subprocess.run([sys.executable, '-m', 'examples.' + module, *flags],
                          cwd=cwd, env=dict(os.environ, PYTHONPATH=python_path),
                          input='\n' * 100, capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize('args', HEADLESS_EXAMPLES)
def test_example_solves(args, tmp_path):
    # Solvers write temp/ and statistics/ into the working directory, so each example gets its own
    result = run_example(args, cwd=tmp_path)
    output = result.stdout[-3000:] + result.stderr[-3000:]
    assert result.returncode == 0, output
    assert 'Solved: True' in result.stdout, output
