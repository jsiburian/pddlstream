"""Learned skills as planning actions.

A SkillSpec describes a skill that a learned policy (e.g. a VLA) performs: its PDDL action (parameters, precondition,
effect), the condition that confirms it succeeded, and its prior success probability, which sets the action's cost
(-log p). The planner treats a skill's effects as assumptions: a plan that uses a skill is certified only up to them.
When the policy reaches the skill, it runs the skill's policy until the success condition is observed (or a timeout),
then adds the confirmed facts to the session's world and replans from what it observes.

ScriptedPress stands in for a learned button-pressing policy on the stand-in simulator.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from benchmarks.common.primitives import APPROACH, solve_ik
from benchmarks.common.protocol import Conditional, pose_from_state
from examples.pybullet.utils.pybullet_tools import utils as pbu
from examples.pybullet.utils.pybullet_tools.pr2_utils import get_top_grasps

COST_SCALE = 10  # Action cost units per nat of -log(p_success); other actions cost 1


@dataclass(frozen=True)
class SkillSpec:
    """A learned skill, as a planning action.

    :param name: the action's name
    :param parameters: its PDDL parameters, e.g. '?b ?q'
    :param precondition: a PDDL formula
    :param effect: a PDDL formula: what the planner assumes the skill achieves
    :param predicates: PDDL declarations of the predicates the skill introduces
    :param goals: goal literals by RoboLab conditional name, for tasks whose conditionals only this skill achieves
    :param success: the Conditional that confirms the skill succeeded, from the action's arguments
    :param confirms: the facts that hold once it succeeded, from the action's arguments
    :param instruction: the language instruction for the skill's policy, from the action's arguments
    :param p_success: the prior probability that the skill succeeds
    :param timeout: actions before the skill counts as failed
    :param make_policy: the skill's policy, from the planner's World: act(observation, args, instruction) returns an
        (n, 8) array of actions
    """

    name: str
    parameters: str
    precondition: str
    effect: str
    predicates: tuple
    success: object
    confirms: object
    instruction: object
    make_policy: object
    goals: dict = field(default_factory=dict)
    p_success: float = 0.9
    timeout: int = 200

    @property
    def cost(self):
        return 1 + round(-math.log(self.p_success) * COST_SCALE)

    def to_pddl(self):
        return (
            f'  (:action {self.name}\n    :parameters ({self.parameters})\n    :precondition {self.precondition}\n'
            f'    :effect (and {self.effect} (increase (total-cost) {self.cost})))\n'
        )


def compile_skills(domain_pddl, skills):
    """The domain with the skills' predicates and actions. With skills, every action has a cost (1 for the domain's
    own), so the planner weighs a skill's success probability."""
    skills = list(skills)
    if not skills:
        return domain_pddl
    predicates = ' '.join(declaration for skill in skills for declaration in skill.predicates)
    domain_pddl = domain_pddl.replace('(:predicates', f'(:predicates {predicates}', 1)
    domain_pddl = domain_pddl.replace(':effect (and ', ':effect (and (increase (total-cost) 1) ')
    end = domain_pddl.rstrip().rindex(')')
    return domain_pddl[:end] + ''.join(skill.to_pddl() for skill in skills) + domain_pddl[end:]


class SkillStep:
    """A plan step that a skill performs, in the policy's list of actions."""

    def __init__(self, skill, args):
        self.skill, self.args = skill, tuple(args)

    def __repr__(self):
        return f'{self.skill.name}{self.args}'


##################################################


MAX_STEP = 0.05  # Radians between consecutive actions
IK_ATTEMPTS = 10


class ScriptedPress:
    """Presses a button: moves the tool above it, down onto its top and back up. It reads only the observation (the
    button's pose and the arm's joints), like a learned policy, and uses the planner's PyBullet mirror for IK."""

    def __init__(self, world):
        self.world, self.queue = world, []

    def act(self, observation, args, instruction):
        name = args[0]
        if not self.queue:
            self.queue = self.plan(observation, name)
        chunk, self.queue = self.queue[:8], self.queue[8:]
        return np.array(chunk if chunk else [[*observation.arm, 0.0]])

    def plan(self, observation, name):
        body = self.world.bodies[name]
        self.world.set_pose(name, pose_from_state(observation.objects[name]))
        # The tool touches the top of the button the way a top grasp of it would hold it
        [grasp] = get_top_grasps(body, under=True, tool_pose=pbu.Pose(), max_width=pbu.INF, grasp_length=0.0)[:1]
        press = pbu.end_effector_from_body(pbu.get_pose(body), grasp)
        approach = pbu.multiply(press, APPROACH)
        confs = [tuple(observation.arm)]
        for target in [approach, press, approach]:
            conf = self.solve(target, confs[-1])
            if conf is None:
                return [[*confs[-1], 0.0]]
            confs.append(conf)
        actions = []
        for start, end in zip(confs, confs[1:]):
            start, end = np.array(start), np.array(end)
            steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / MAX_STEP)))
            actions.extend([*(start + (end - start) * (i / steps)), 0.0] for i in range(1, steps + 1))
        return actions

    def solve(self, target, seed_conf):
        """Joint positions that put the tool at target, starting from seed_conf, then from random configurations."""
        sample_fn = pbu.get_sample_fn(self.world.robot, self.world.arm)
        for attempt in range(IK_ATTEMPTS):
            self.world.set_arm(seed_conf if attempt == 0 else sample_fn())
            conf = solve_ik(self.world, target)
            if conf is not None:
                return conf
        return None


PRESS = SkillSpec(
    name='press',
    parameters='?b ?q',
    precondition='(and (Button ?b) (AtConf ?q) (HandEmpty))',
    effect='(Pressed ?b) (CanMove)',  # The planner assumes the arm returns to ?q
    predicates=('(Button ?b)', '(Pressed ?b)'),
    goals={'button_pressed': lambda params: [('Pressed', params['button'])]},
    success=lambda args: Conditional('button_pressed', {'button': args[0]}),
    confirms=lambda args: [('Pressed', args[0])],
    instruction=lambda args: f'Press the {args[0]}',
    make_policy=ScriptedPress,
    p_success=0.8,
    timeout=150,
)

SKILLS = {skill.name: skill for skill in [PRESS]}
