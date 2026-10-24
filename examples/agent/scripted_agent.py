"""A scripted agent that uses the agent tools (pddlstream.tools) to debug a sampler, then plan and check the plan.

The problem: find a candidate whose square is 49. Its sampler is broken (it returns two values per candidate instead
of one). The agent finds this with check_stream, the host program installs a fixed sampler, and the agent plans and
validates. An agent never asserts facts: it proposes a plan, which validate_plan checks.

Run: python -m examples.agent.scripted_agent
"""

import json
import logging

from pddlstream.core.problem import PDDLProblem
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.tools import Toolbox, Workspace

DOMAIN_PDDL = """(define (domain search) (:requirements :strips)
  (:predicates (Candidate ?x) (Valid ?x) (Done))
  (:action report :parameters (?x) :precondition (and (Candidate ?x) (Valid ?x)) :effect (Done)))"""

STREAM_PDDL = """(define (stream search)
  (:stream sample-candidate :outputs (?x) :certified (Candidate ?x))
  (:stream test-valid :inputs (?x) :domain (Candidate ?x) :certified (Valid ?x)))"""


def broken_sampler():
    for x in range(10):
        yield (x, x)  # Two values for one output


def fixed_sampler():
    for x in range(10):
        yield (x,)


def make_problem(sampler):
    stream_map = {'sample-candidate': from_gen_fn(sampler), 'test-valid': from_test(lambda x: x * x == 49)}
    return PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, stream_map, [], ('Done',))


def call(toolbox, name, **arguments):
    response = toolbox.call(name, arguments)
    print(f'> {name}({json.dumps(arguments)})\n  {json.dumps(response)[:300]}')
    assert response['ok'], response
    return response['result']


def main():
    logging.getLogger('pddlstream').setLevel(logging.WARNING)
    workspace = Workspace()
    workspace.add_problem('squares', make_problem(broken_sampler), unit_costs=True, deterministic=True)
    toolbox = Toolbox(workspace)

    model = call(toolbox, 'describe_problem', problem_id='squares')
    [sampler] = [stream for stream in model['streams'] if stream['name'] == 'sample-candidate']
    issues = call(toolbox, 'check_stream', problem_id='squares', stream=sampler['name'], inputs=[])['issues']
    assert any(issue.startswith('wrong-output-arity') for issue in issues)

    # The host program installs the agent's fix; the old sampler's results go with the old session
    workspace.add_problem('squares', make_problem(fixed_sampler), unit_costs=True, deterministic=True)
    assert not call(toolbox, 'check_stream', problem_id='squares', stream='sample-candidate', inputs=[])['issues']

    wrong = call(toolbox, 'validate_plan', problem_id='squares', plan=[['report', [3]]])
    print(f'A plan the agent guessed is rejected: missing {wrong.get("missing")}')
    run = call(toolbox, 'run_search', problem_id='squares', budget={'max_time': 30})
    assert run['status'] == 'succeeded'
    assert call(toolbox, 'validate_plan', problem_id='squares', run_id=run['run_id'])['valid']
    print(f'Plan: {run["plan"]}')
    workspace.close()


if __name__ == '__main__':
    main()
