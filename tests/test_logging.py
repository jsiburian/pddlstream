"""Solvers log through 'pddlstream' loggers, and print to stdout only when logging is not configured."""

import logging
import os
import pathlib
import subprocess
import sys

from pddlstream.algorithms.meta import solve
from pddlstream.core.problem import PDDLProblem

# Without action costs, parsing warns and recommends unit_costs=True
DOMAIN_PDDL = """(define (domain switch) (:requirements :strips) (:predicates (On))
  (:action turn-on :parameters () :precondition (and) :effect (On)))"""
PROBLEM = PDDLProblem(DOMAIN_PDDL, {}, None, {}, [], ('On',))


def test_console_without_logging_configuration():
    code = (
        'from tests.test_logging import PROBLEM; from pddlstream.algorithms.meta import solve; '
        "solve(PROBLEM, algorithm='incremental')"
    )
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'Warning! All actions have no cost' in result.stdout
    assert 'Summary: {' in result.stdout


def test_configured_logging(caplog, capsys):
    # pytest's log capturing counts as configured logging, so nothing reaches stdout
    with caplog.at_level(logging.INFO, logger='pddlstream'):
        solve(PROBLEM, algorithm='incremental')
    assert 'Summary' not in capsys.readouterr().out
    messages = [(record.name, record.levelno, record.getMessage()) for record in caplog.records]
    assert any(level == logging.WARNING and 'All actions have no cost' in message for _, level, message in messages)
    assert any(
        name.startswith('pddlstream.algorithms') and message.startswith('Summary') for name, _, message in messages
    )
