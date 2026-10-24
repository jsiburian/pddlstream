"""Goals from language. An agent reads a task's instruction and a description of the scene, proposes a goal through
pddlstream's propose_goal tool (which checks it against the planning problem), and is scored against the goal derived
from the task's success conditionals (goals.py).

ScriptedAgent understands the stand-in's instructions (put or place X in or on Y, stack A on B on C, pick up X,
press X) and grounds their noun phrases in the scene. It runs in CI; an LLM agent can call the same tool through
pddlstream's MCP server (pddlstream.tools.mcp_server) with a Workspace holding the task's problem.
"""

import os
import re

import numpy as np

from benchmarks.common.goals import UnsupportedTask, goal_from_task
from benchmarks.common.skills import SKILLS, compile_skills
from pddlstream.core.problem import PDDLProblem
from pddlstream.core.utils import read
from pddlstream.language.external import DEBUG
from pddlstream.tools import Toolbox, Workspace

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ALL = re.compile(r'^(?:both|all|each|the|the two|the three|the four) blocks?$|^(?:all|both) the blocks$')


def describe_scene(scene, observation):
    """What an agent sees: each object's name, kind, size and position."""
    return [
        {'name': spec.name, 'kind': spec.kind, 'size': list(spec.size),
         'position': list(map(float, observation.objects[spec.name].position))}
        for spec in scene
    ]  # fmt: skip


def symbolic_problem(scene, skills=()):
    """The planning problem of a scene with placeholder poses and procedures, for checking goals without a World."""
    domain_pddl = compile_skills(read(os.path.join(DIRECTORY, 'domain.pddl')), skills)
    init = [('Conf', 'q0'), ('AtConf', 'q0'), ('HandEmpty',), ('CanMove',)]
    for spec in scene:
        kind = {'movable': 'Movable', 'button': 'Button'}.get(spec.kind, 'Surface')
        init += [(kind, spec.name), ('Pose', spec.name, f'p_{spec.name}'), ('AtPose', spec.name, f'p_{spec.name}')]
    for spec in scene:
        if spec.kind == 'movable':
            init += [
                ('Stackable', spec.name, other.name) for other in scene if other != spec and other.kind != 'button'
            ]
    return PDDLProblem(domain_pddl, {}, read(os.path.join(DIRECTORY, 'streams.pddl')), DEBUG, init, ('HandEmpty',))


class ScriptedAgent:
    """Turns an instruction into a goal by templates; raises ValueError when it cannot."""

    def propose(self, instruction, scene):
        literals = []
        for clause in re.split(r',?\s+then\s+|\.\s*', instruction.lower().strip().rstrip('.')):
            if clause:
                literals.extend(self.parse(clause, scene))
        return ['and', *map(list, literals)]

    def parse(self, clause, scene):
        if clause.startswith(('put ', 'place ')):
            rest = clause.split(' ', 1)[1]
            literals = []
            for part in re.split(r'\s+and\s+', rest):
                objects, target = re.split(r'\s+(?:in|on|into|onto)\s+', part, maxsplit=1)
                [support] = self.ground(target, scene)
                literals.extend(('On', name, support) for name in self.ground(objects, scene))
            return literals
        if clause.startswith('stack '):
            names = [self.ground(phrase, scene)[0] for phrase in re.split(r'\s+on\s+', clause[len('stack ') :])]
            return [('On', top, bottom) for top, bottom in zip(names, names[1:])]
        if clause.startswith('pick up '):
            return [('Holding', name) for name in self.ground(clause[len('pick up ') :], scene)]
        if clause.startswith('press '):
            return [('Pressed', name) for name in self.ground(clause[len('press ') :], scene)]
        raise ValueError(f'Cannot parse {clause!r}')

    def ground(self, phrase, scene):
        """The names of the objects a noun phrase refers to."""
        phrase = phrase.strip()
        names = {obj['name'] for obj in scene}
        blocks = sorted(obj['name'] for obj in scene if obj['kind'] == 'movable')
        if ALL.match(phrase) and (phrase.endswith('blocks') or phrase.startswith('each')):
            return blocks
        words = phrase.removeprefix('the ').split()
        if (len(words) == 2) and (words[1] in names):  # e.g. "block a"
            return [words[1]]
        if (len(words) == 1) and (words[0] in names):
            return words
        noun, adjectives = words[-1], words[:-1]
        kind = {'block': 'movable', 'tray': 'container', 'button': 'button'}.get(noun)
        candidates = [obj for obj in scene if obj['kind'] == kind]
        for adjective in adjectives:
            if adjective in names:  # A tray named after its side
                return [adjective]
            if adjective == 'small':
                candidates = [min(candidates, key=lambda obj: np.prod(obj['size'][:2]))]
            elif adjective == 'far':
                candidates = [max(candidates, key=lambda obj: np.linalg.norm(obj['position'][:2]))]
            elif adjective in ('left', 'right'):  # The robot's left is +y
                candidates = [(max if adjective == 'left' else min)(candidates, key=lambda obj: obj['position'][1])]
        if len(candidates) != 1:
            raise ValueError(f'{phrase!r} does not identify one object')
        return [candidates[0]['name']]


def literal_set(goal):
    """A goal's literals as a set of tuples of strings."""
    parts = goal[1:] if goal[0] in ('and', 'AND') else [goal]
    return {tuple(map(str, literal)) for literal in parts}


def score(agent, scenarios, observations):
    """Each scenario's instruction through the agent and propose_goal: whether the tool accepted the goal without
    issues, and whether it matches the goal derived from the success conditionals."""
    results = []
    for scenario, observation in zip(scenarios, observations):
        skills = [SKILLS[name] for name in scenario.skills]
        workspace = Workspace()
        workspace.add_problem('task', symbolic_problem(scenario.objects, skills), unit_costs=not skills)
        toolbox = Toolbox(workspace)
        instruction = scenario.task.instruction['default']
        result = {'task': scenario.task.name, 'instruction': instruction}
        try:
            goal = agent.propose(instruction, describe_scene(scenario.objects, observation))
            response = toolbox.call('propose_goal', {'problem_id': 'task', 'goal': goal})
            proposal = response['result'] if response['ok'] else {'accepted': False, 'issues': [response['error']]}
            expected = literal_set(goal_from_task(scenario.task, skills))
            result.update(
                goal=goal,
                accepted=proposal['accepted'] and not proposal['issues'],
                issues=proposal['issues'],
                correct=literal_set(goal) == expected,
            )
        except (ValueError, UnsupportedTask) as error:
            result.update(goal=None, accepted=False, issues=[str(error)], correct=False)
        finally:
            workspace.close()
        results.append(result)
    return results
