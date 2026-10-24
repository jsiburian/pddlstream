"""The agent tools take and return JSON, never accept facts, and only rank the search when given hints."""

import json

import pytest

from examples.agent.scripted_agent import fixed_sampler, make_problem
from pddlstream.core.problem import PDDLProblem
from pddlstream.tools import TOOLS, Toolbox, Workspace


@pytest.fixture
def toolbox():
    workspace = Workspace()
    workspace.add_problem('squares', make_problem(fixed_sampler), unit_costs=True, deterministic=True)
    yield Toolbox(workspace)
    workspace.close()


def call(toolbox, tool, **arguments):
    response = toolbox.call(tool, arguments)
    json.dumps(response)  # Plain data
    assert response['ok'], response
    return response['result']


def test_no_tool_accepts_facts():
    for tool in TOOLS:
        for name in tool.input_schema['properties']:
            assert not any(word in name for word in ['fact', 'init', 'certif', 'evaluation']), (tool.name, name)


def test_search_and_inspect(toolbox):
    run = call(toolbox, 'run_search', problem_id='squares', budget={'max_time': 30})
    assert (run['status'], run['plan']) == ('succeeded', [['report', [7]]])
    run_id = run['run_id']
    assert call(toolbox, 'explain_failure', run_id=run_id)['status'] == 'succeeded'
    calls = call(toolbox, 'inspect_trace', run_id=run_id, event='Call', limit=2)
    assert (calls['total'] > 2) and (len(calls['events']) == 2)
    assert call(toolbox, 'profile', run_id=run_id)['calls']['test-valid'] >= 1
    assert call(toolbox, 'list_stream_contracts', problem_id='squares', name='test-valid')[0]['statistics']['calls']
    assert call(toolbox, 'run_regressions', problem_ids=['squares']) == {'squares': 'succeeded'}


def test_wrong_hint_only_ranks(toolbox):
    assert call(toolbox, 'propose_hint', problem_id='squares', skeleton=[['report', [3]]])['accepted_for_ranking']
    run = call(toolbox, 'run_search', problem_id='squares', budget={'max_time': 30})
    assert run['plan'] == [['report', [7]]]
    assert not call(toolbox, 'propose_hint', problem_id='squares', skeleton=[['fly', []]])['accepted_for_ranking']


def test_check_stream_leaves_the_solver_alone(toolbox):
    session = toolbox.workspace.session('squares')
    instances = {external.name: dict(external.instances) for external in session.externals}
    result = call(toolbox, 'check_stream', problem_id='squares', stream='test-valid', inputs=[7])
    assert (result['issues'], result['calls'][0]['outputs']) == ([], [[]])
    assert {external.name: dict(external.instances) for external in session.externals} == instances
    assert call(toolbox, 'check_stream', problem_id='squares', stream='test-valid', inputs=[])['issues']


def test_validate_plan(toolbox):
    wrong = call(toolbox, 'validate_plan', problem_id='squares', plan=[['report', [3]]])
    assert (wrong['valid'], wrong['first_failed_step'], wrong['missing']) == (
        False,
        0,
        [['candidate', 3], ['valid', 3]],
    )


def test_errors(toolbox):
    assert toolbox.call('fly')['error']['code'] == 'unknown-tool'
    assert toolbox.call('run_search', {'problem': 'squares'})['error']['code'] == 'bad-arguments'
    assert toolbox.call('run_search', {'problem_id': 'cubes'})['error']['code'] == 'KeyError'
    response = toolbox.call('run_search', {'problem_id': 'squares', 'budget': {'max_tme': 1}})
    assert "did you mean 'max_time'" in response['error']['message']


def test_mcp_server_needs_mcp(toolbox):
    try:
        import mcp  # noqa: F401
    except ImportError:
        from pddlstream.tools.mcp_server import serve

        with pytest.raises(ImportError, match=r'pddlstream\[agent\]'):
            serve(toolbox.workspace)
    else:
        pytest.skip('mcp is installed')


def test_propose_goal(toolbox):
    proposal = call(toolbox, 'propose_goal', problem_id='squares', goal=['and', ['Done']])
    assert (proposal['accepted'], proposal['issues']) == (True, [])
    assert call(toolbox, 'run_search', problem_id='squares', budget={'max_time': 30})['status'] == 'succeeded'
    # 7 is not an initial object, but a stream can produce it
    [warning] = call(toolbox, 'propose_goal', problem_id='squares', goal=['Valid', 7])['issues']
    assert (warning['severity'], warning['code']) == ('warning', 'unknown-object')
    rejected = call(toolbox, 'propose_goal', problem_id='squares', goal=['and', ['Fly'], ['Valid'], ['not', 'Done']])
    assert not rejected['accepted']
    assert [issue['code'] for issue in rejected['issues']] == ['unknown-predicate', 'wrong-arity', 'malformed']


def test_unreachable_goal():
    domain = """(define (domain door) (:requirements :strips)
  (:predicates (Closed ?d) (Open ?d) (Locked ?d))
  (:action open :parameters (?d) :precondition (Closed ?d) :effect (and (Open ?d) (not (Closed ?d)))))"""
    workspace = Workspace()
    workspace.add_problem('door', PDDLProblem(domain, {}, None, {}, [('Closed', 'front')], ('Open', 'front')))
    toolbox = Toolbox(workspace)
    assert call(toolbox, 'propose_goal', problem_id='door', goal=['Open', 'front'])['accepted']
    issues = call(toolbox, 'propose_goal', problem_id='door', goal=['and', ['Locked', 'front'], ['Open', 'back']])
    assert [issue['code'] for issue in issues['issues']] == ['unreachable', 'unknown-object']
    workspace.close()
