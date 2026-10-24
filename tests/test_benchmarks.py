"""The benchmark integrations (benchmarks/) without a GPU: the goal mapping, the protocol, the conditionals, and the
policy on stand-in tasks through the planner server (marked benchmarks: pixi run test-benchmarks)."""

import ast
import json
import pathlib

import numpy as np
import pytest

from benchmarks.common.goals import UnsupportedTask, coverage, goal_from_task, goal_literals
from benchmarks.common.language import ScriptedAgent, score, symbolic_problem
from benchmarks.common.protocol import Conditional, TaskSpec, observation_from_json, task_from_json, to_json
from benchmarks.common.remote import start_server
from benchmarks.common.skills import PRESS, compile_skills
from benchmarks.standin.env import SCENARIOS, StandinEnv
from benchmarks.standin.episodes import generate, load, replay, run_episode
from pddlstream.tools import Toolbox, Workspace

FIXTURES = pathlib.Path(__file__).with_name('fixtures') / 'robolab_tasks.json'


def test_goals():
    task = TaskSpec('Stack', {}, (Conditional('stacked', {'objects': ['c', 'b', 'a'], 'order': 'bottom_to_top'}),))
    assert goal_from_task(task) == ('and', ('On', 'b', 'c'), ('On', 'a', 'b'))
    either = Conditional('object_in_container', {'object': ['a', 'b'], 'container': 'bin', 'logical': 'any'})
    assert goal_from_task(TaskSpec('Either', {}, (either,))) == ('or', ('On', 'a', 'bin'), ('On', 'b', 'bin'))
    two = Conditional(
        'object_on_top', {'object': ['a', 'b', 'c'], 'reference_object': 'plate', 'logical': 'choose', 'K': 2}
    )
    assert len(goal_from_task(TaskSpec('Two', {}, (two,)))) == 1 + 3  # 'or' and the three pairs
    any_order = Conditional('stacked', {'objects': ['a', 'b', 'c'], 'order': 'None'})
    assert len(goal_from_task(TaskSpec('Stack', {}, (any_order,)))) == 1 + 6
    with pytest.raises(UnsupportedTask, match='object_upright'):
        goal_from_task(TaskSpec('Upright', {}, (Conditional('object_upright', {'object': 'a'}),)))


def test_robolab_coverage():
    # The fixtures come from benchmarks/robolab/probe.py, which scans RoboLab's task definitions
    tasks = [task_from_json(data) for data in json.loads(FIXTURES.read_text())]
    supported, skipped = coverage(tasks)
    assert (len(tasks), len(supported)) == (120, 97), skipped
    for task in tasks:
        if task.name in supported:
            for literal in goal_literals(goal_from_task(task)):
                assert literal[0] in ('On', 'Holding') and all(isinstance(arg, str) for arg in literal[1:]), literal


def test_protocol_round_trip():
    env = StandinEnv(SCENARIOS[0])
    try:
        observation = env.observe()
        restored = observation_from_json(json.loads(json.dumps(to_json(observation))))
        assert np.allclose(restored.arm, observation.arm)
        assert set(restored.objects) == set(observation.objects)
        task = task_from_json(json.loads(json.dumps(to_json(SCENARIOS[0].task))))
        assert task == SCENARIOS[0].task
    finally:
        env.close()


def test_success_needs_the_goal():
    env = StandinEnv(SCENARIOS[0])
    try:
        assert not env.success()
        env.move('a', env.world.get_pose('tray')[0][:2])
        assert not env.success()  # On the table under the tray, not in it
    finally:
        env.close()


@pytest.fixture(scope='module')
def client():
    process, client = start_server()
    yield client
    process.kill()


@pytest.mark.benchmarks
@pytest.mark.parametrize('scenario', SCENARIOS, ids=[scenario.task.name for scenario in SCENARIOS])
def test_standin_task(client, scenario):
    result = run_episode(client, scenario)
    assert result['success'], result
    plans = [report for report in result['plans'] if report['plan'] is not None]
    assert plans and all(report['verified'] for report in plans), plans


@pytest.mark.benchmarks
def test_replanning(client):
    result = run_episode(client, SCENARIOS[0], perturb=True)  # The block moves before the robot reaches it
    assert result['success'] and (result['perturbation'] is not None), result
    assert len(result['plans']) == 2  # The policy replanned once


def test_skills():
    scenario = next(s for s in SCENARIOS if s.task.name == 'PressThenPlace')
    with pytest.raises(UnsupportedTask, match='button_pressed'):
        goal_from_task(scenario.task)
    assert goal_from_task(scenario.task, [PRESS]) == ('and', ('Pressed', 'button'), ('On', 'a', 'tray'))
    domain = compile_skills('(define (domain d) (:predicates (P))\n  (:action a :effect (and (P))))', [PRESS])
    assert '(Pressed ?b)' in domain and f'(increase (total-cost) {PRESS.cost})' in domain
    assert '(and (increase (total-cost) 1) (P))' in domain  # The domain's own actions cost 1


@pytest.mark.benchmarks
def test_hybrid_task(client):
    # The plan assumes the skill presses the button; once the stand-in confirms it, the policy plans the rest
    scenario = next(s for s in SCENARIOS if s.task.name == 'PressThenPlace')
    result = run_episode(client, scenario)
    assert result['success'], result
    first, second = result['plans']
    assert first['plan'][0].startswith('press(') and first['assumptions'] == ['(Pressed button)']
    assert not any(step.startswith('press(') for step in second['plan'])


@pytest.mark.benchmarks
def test_demonstrations(client, tmp_path):
    scenarios = [s for s in SCENARIOS if s.task.name == 'BlockInTray']
    results = generate(client, tmp_path, scenarios, seeds=(1,), perturb=(False, True))
    assert all(result['demonstration'] for result in results)
    paths = sorted(tmp_path.glob('*.npz'))
    assert len(paths) == 2 == len((tmp_path / 'index.jsonl').read_text().splitlines())
    arrays, metadata = load(paths[0])
    num_actions = len(arrays['actions'])
    assert arrays['arm'].shape == (num_actions + 1, 7)
    assert arrays['positions'].shape == (num_actions + 1, len(arrays['object_names']), 3)
    assert metadata['instruction'] == 'Put the block in the tray'
    assert all(replay(path) for path in paths)


def observe(scenario):
    env = StandinEnv(scenario)
    try:
        return env.observe()
    finally:
        env.close()


@pytest.mark.benchmarks
def test_language_agent():
    results = score(ScriptedAgent(), SCENARIOS, [observe(scenario) for scenario in SCENARIOS])
    assert all(result['correct'] and result['accepted'] for result in results), results


def test_propose_goal_checks():
    workspace = Workspace()
    workspace.add_problem('task', symbolic_problem(SCENARIOS[0].objects), unit_costs=True)
    toolbox = Toolbox(workspace)
    proposal = toolbox.call('propose_goal', {'problem_id': 'task', 'goal': ['and', ['On', 'a'], ['Pressed', 'a']]})
    codes = [issue['code'] for issue in proposal['result']['issues']]
    assert codes == ['wrong-arity', 'unknown-predicate']  # Pressing needs the press skill
    workspace.close()


CLIENT_MODULES = ['benchmarks/common/protocol.py', 'benchmarks/common/remote.py', 'benchmarks/robolab/client.py']


def test_client_modules_do_not_import_pddlstream():
    # Benchmark clients run in the benchmark's own Python environment, without pddlstream
    root = pathlib.Path(__file__).resolve().parents[1]
    for module in CLIENT_MODULES:
        for node in ast.walk(ast.parse((root / module).read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in getattr(node, 'names', [])]
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert not any(name and name.startswith('pddlstream') for name in names), (module, names)


def test_package_does_not_import_benchmarks():
    root = pathlib.Path(__file__).resolve().parents[1]
    for path in (root / 'pddlstream').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith('benchmarks'), path
