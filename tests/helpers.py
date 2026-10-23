"""Helpers for running examples in isolated subprocesses."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'tests' / 'harness.py'
RESULT_PREFIX = 'HARNESS_RESULT '
XVFB_COMMAND = ['xvfb-run', '-a', '-s', '-screen 0 1280x1024x24 +extension GLX']


def has_xvfb():
    return shutil.which('xvfb-run') is not None


def subprocess_env():
    python_path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))
    # A fixed hash seed makes FastDownward's translator (which iterates string-keyed sets) repeatable
    return dict(os.environ, PYTHONPATH=python_path, PYTHONHASHSEED='0')


def run_python(args, cwd, timeout=600, display=False):
    """Runs `python <args>` in cwd, answering any prompts. Solvers write temp/ and statistics/ into the working
    directory, so callers pass a fresh directory. display=True runs under a virtual X server."""
    command = [sys.executable, *args]
    if display:
        command = XVFB_COMMAND + command
    return subprocess.run(command, cwd=cwd, env=subprocess_env(), input='\n' * 100,
                          capture_output=True, text=True, timeout=timeout)


def output_tail(result, num_chars=3000):
    return result.stdout[-num_chars:] + result.stderr[-num_chars:]


def run_case(example, cwd, timeout=600, display=False):
    """Runs an example (e.g. 'blocksworld.run -a focused') through tests/harness.py and returns its JSON result."""
    result = run_python([str(HARNESS), *example.split()], cwd=cwd, timeout=timeout, display=display)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise RuntimeError('{} produced no result (exit code {}):\n{}'.format(
        example, result.returncode, output_tail(result)))
