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


def subprocess_env(cwd):
    python_path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))
    # A fixed hash seed makes FastDownward's translator (which iterates string-keyed sets) repeatable.
    # Stream statistics go to cwd, so that earlier runs don't influence this one.
    cache_dir = os.path.join(cwd, 'cache')
    return dict(os.environ, PYTHONPATH=python_path, PYTHONHASHSEED='0', PDDLSTREAM_CACHE_DIR=cache_dir)


def run_python(args, cwd, timeout=600, display=False):
    """Runs `python <args>` in cwd, answering any prompts. Examples can write files into the working directory and
    save stream statistics, so callers pass a fresh directory. display=True runs under a virtual X server."""
    command = [sys.executable, *args]
    if display:
        command = XVFB_COMMAND + command
    return subprocess.run(
        command, cwd=cwd, env=subprocess_env(cwd), input='\n' * 100, capture_output=True, text=True, timeout=timeout
    )


def output_tail(result, num_chars=3000):
    return result.stdout[-num_chars:] + result.stderr[-num_chars:]


def run_case(example, cwd, timeout=600, display=False, wall_clock=False):
    """Runs an example (e.g. 'blocksworld.run -a focused') through tests/harness.py and returns its JSON result."""
    flags = ['--wall-clock'] if wall_clock else []
    result = run_python([str(HARNESS), *flags, *example.split()], cwd=cwd, timeout=timeout, display=display)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f'{example} produced no result (exit code {result.returncode}):\n{output_tail(result)}')
