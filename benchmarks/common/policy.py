"""The TAMP policy: plans a task from an observation with a PlannerSession, returns the plan as chunks of joint-position
actions, and replans when objects are not where the plan expects them. Plan steps that learned skills perform
(skills.py) are run by the skill's policy until the environment confirms them; then the policy replans.
"""

import hashlib
import logging
import os

import numpy as np

from benchmarks.common.goals import goal_from_task, goal_literals
from benchmarks.common.primitives import Conf, Pose, collides, get_procedures, get_stream_info, seed, solve_ik
from benchmarks.common.protocol import condition_key, pose_from_state
from benchmarks.common.skills import SKILLS, SkillStep, compile_skills
from benchmarks.common.world import World, pbu
from pddlstream.algorithms.session import PlannerSession, WorldPatch
from pddlstream.algorithms.validate import verify_certificate
from pddlstream.core.problem import PDDLProblem
from pddlstream.core.utils import read

logger = logging.getLogger(__name__)
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TABLE = 'table'
OPEN, CLOSED = 0.0, 1.0
MAX_JOINT_STEP = 0.05  # Radians between consecutive actions
TOLERANCE = 0.03  # Meters an object may be from where the plan expects it before replanning
RETREAT = 0.1  # Meters the tool rises before replanning when the arm touches an object (e.g. after a slipped grasp)


def lowest_point(spec, state):
    """The height of an object's lowest point: its box's bottom, or its lowest surface point."""
    if not spec.points:
        return state.position[2] - spec.size[2] / 2
    rotation = pbu.matrix_from_quat(pose_from_state(state)[1])
    return state.position[2] + (np.asarray(spec.points) @ rotation.T)[:, 2].min()


class TampPolicy:
    """A policy for one environment: reset() with a task, the scene's objects and the first observation, then act()
    with each observation."""

    def __init__(
        self,
        scene,
        horizon=8,
        max_time=60,
        max_replans=3,
        teleport=False,
        skills=(),
        gripper='panda',
        containment='full',
    ):
        self.world = World(scene, gripper=gripper, containment=containment)
        self.horizon, self.max_time, self.max_replans = horizon, max_time, max_replans
        self.teleport = teleport
        self.skills = {skill.name: skill for skill in (SKILLS[s] if isinstance(s, str) else s for s in skills)}
        self.session = None
        self.actions, self.expected = (
            [],
            [],
        )  # Remaining actions (or SkillSteps), with the object poses expected after each
        self.status, self.replans = 'idle', 0
        self.skill_run = None  # The SkillStep being performed, its policy and the actions it took
        self.plans = []  # A report of each plan (see report_plan)
        self.prefix = []  # Actions that precede the next plan's (see retreat)

    def reset(self, task, observation):
        seed(0)
        self.goal = goal_from_task(task, skills=self.skills.values())
        self.poses = {
            name: Pose(self.world, name, pose_from_state(state)) for name, state in observation.objects.items()
        }
        self.poses[TABLE] = Pose(self.world, TABLE, ((0, 0, 0), (0, 0, 0, 1)))
        bottoms = [
            lowest_point(self.world.specs[name], state)
            for name, state in observation.objects.items()
            if name in self.world.specs
        ]
        if bottoms:  # The table is where the lowest object rests
            self.world.set_table_top(min(bottoms))
        for pose in self.poses.values():  # The mirror starts where the objects are
            pose.assign()
        self.conf = Conf(observation.arm)
        self.domain_pddl = compile_skills(read(os.path.join(DIRECTORY, 'domain.pddl')), self.skills.values())
        problem = PDDLProblem(
            self.domain_pddl,
            {},
            read(os.path.join(DIRECTORY, 'streams.pddl')),
            get_procedures(self.world, teleport=self.teleport),
            self.get_init(),
            self.goal,
        )
        self.session = PlannerSession(
            problem, stream_info=get_stream_info(), unit_costs=not self.skills, deterministic=True, verbose=False
        )
        self.replans, self.skill_run, self.plans, self.prefix = 0, None, [], []
        self.plan()

    def get_init(self):
        movables = set(self.world.movables) & set(self.poses)
        literals = goal_literals(self.goal)
        supports = {support for name, _, support in filter(lambda literal: literal[0] == 'On', literals)}
        buttons = {name for name, spec in self.world.specs.items() if spec.kind == 'button'}
        init = [('Conf', self.conf), ('AtConf', self.conf), ('HandEmpty',), ('CanMove',)]
        for name, pose in self.poses.items():
            kind = 'Movable' if name in movables else 'Button' if name in buttons else 'Surface'
            init += [(kind, name), ('Pose', name, pose), ('AtPose', name, pose)]
        for name in movables:
            surfaces = [
                other
                for other in self.poses
                if ((other not in movables) and (other not in buttons)) or (other in supports - {name})
            ]
            init += [('Stackable', name, surface) for surface in surfaces]
        return init

    def plan(self):
        result = self.session.solve(max_time=self.max_time)
        self.plans.append(self.report_plan(result))
        if result.plan is None:
            self.status, self.actions, self.expected = 'failed', [], []
            logger.warning(f'No plan ({result.stats.get("termination")}, gaps {result.stats.get("gaps")})')
            return result
        self.status = 'executing'
        self.actions, self.expected = self.compile(result.plan)
        if self.prefix:  # The retreat that the plan starts after
            positions = {name: pose.value[0] for name, pose in self.poses.items()}
            self.actions = self.prefix + self.actions
            self.expected = [positions] * len(self.prefix) + self.expected
            self.prefix = []
        return result

    def retreat(self, observation):
        """When the arm touches an object or the table, actions that raise the tool RETREAT meters with the gripper
        open, so that motion planning starts from a collision-free configuration; otherwise []."""
        world = self.world
        for name, state in observation.objects.items():
            if name in world.bodies:
                world.set_pose(name, pose_from_state(state))
        world.set_arm(observation.arm)
        if not collides(world, world.fixed + [world.bodies[name] for name in world.movables]):
            return []
        target = pbu.multiply(pbu.Pose(point=pbu.Point(z=RETREAT)), world.tool_pose())
        conf = solve_ik(world, target)  # Seeded with the current configuration
        if conf is None:
            return []
        logger.info('The arm touches an object: retreating before replanning')
        return [np.array([*q, OPEN]) for q in interpolate([tuple(observation.arm), conf])[1:]]

    def report_plan(self, result):
        """What a plan is and what it rests on: its steps, a digest of its certificate, whether the certificate
        verifies independently, and the facts it assumes that skills will achieve."""
        if result.plan is None:
            return {'plan': None, 'termination': result.stats.get('termination'), 'gaps': result.stats.get('gaps')}
        problem = PDDLProblem(self.domain_pddl, {}, None, {}, self.session.init, self.goal)
        verified, issues = verify_certificate(problem, result, unit_costs=not self.skills)
        supports = [
            (s.stream, tuple(map(str, s.inputs)), tuple(map(str, s.outputs))) for s in result.certificate.supports
        ]
        assumptions = [
            fact for name, args in result.plan if name in self.skills for fact in self.skills[name].confirms(args)
        ]
        return {
            'plan': [f'{name}({", ".join(map(str, args))})' for name, args in result.plan],
            'certificate': hashlib.sha256(repr(supports).encode()).hexdigest()[:16],
            'verified': verified,
            'issues': issues,
            'assumptions': [f'({" ".join(map(str, fact))})' for fact in assumptions],
        }

    def compile(self, plan):
        """The plan as actions [7 joint positions, gripper], with the object poses expected after each."""
        actions, expected = [], []
        positions = {name: pose.value[0] for name, pose in self.poses.items()}
        gripper = OPEN

        def add(path):
            for conf in interpolate(path):
                actions.append(np.array([*conf, gripper]))
                expected.append(dict(positions))

        for action in plan:
            name, args = action
            if name in self.skills:  # Plans after a skill are made once the skill has run
                actions.append(SkillStep(self.skills[name], args))
                expected.append(dict(positions))
                break
            if name in ('move_free', 'move_holding'):
                add(args[-1].path)
            elif name in ('pick', 'place'):
                obj, pose, _, _, trajectory = args
                add(trajectory.path)
                add([trajectory.path[-1]] * self.world.gripper.arrive_steps)
                gripper = CLOSED if name == 'pick' else OPEN
                if name == 'pick':
                    positions.pop(obj, None)  # Held objects are not monitored
                else:
                    positions[obj] = pose.value[0]
                add([trajectory.path[-1]] * self.world.gripper.settle_steps)
                add(trajectory.path[::-1])
        return actions, expected

    def deviation(self, observation, expected):
        for name, position in expected.items():
            if name in observation.objects:
                if np.linalg.norm(np.array(observation.objects[name].position) - np.array(position)) > TOLERANCE:
                    return name
        return None

    def replan(self, observation, confirmed=()):
        """Updates the session's world from the observation (and the facts that a skill confirmed) and plans again."""
        patch_add, patch_remove = list(confirmed), []
        for name, state in observation.objects.items():
            old = self.poses[name]
            if np.linalg.norm(np.array(state.position) - np.array(old.value[0])) > TOLERANCE:
                new = Pose(self.world, name, pose_from_state(state))
                patch_remove += [('AtPose', name, old)]
                patch_add += [('Pose', name, new), ('AtPose', name, new)]
                self.poses[name] = new
        self.prefix = self.retreat(observation)
        conf = Conf(self.prefix[-1][:7] if self.prefix else observation.arm)
        patch_remove += [('AtConf', self.conf)]
        patch_add += [('Conf', conf), ('AtConf', conf)]
        self.conf = conf
        self.session.apply(WorldPatch(add=patch_add, remove=patch_remove))
        self.replans += 1
        return self.plan()

    def act(self, observation):
        """The next chunk of actions, of shape (horizon, 8)."""
        if self.skill_run is not None:
            return self.act_skill(observation)
        if self.actions and self.expected and (self.deviation(observation, self.expected[0]) is not None):
            if self.replans >= self.max_replans:
                self.status, self.actions = 'failed', []
            else:
                logger.info('An object moved: replanning')
                self.replan(observation)
        if self.actions and isinstance(self.actions[0], SkillStep):
            step = self.actions.pop(0)
            self.expected.pop(0)
            self.skill_run = [step, step.skill.make_policy(self.world), 0]
            return self.act_skill(observation)
        hold = np.array([*observation.arm, observation.gripper])
        if not self.actions:
            if self.status == 'executing':
                self.status = 'done'
            return np.array([hold] * self.horizon)
        chunk = []
        while self.actions and (len(chunk) < self.horizon) and not isinstance(self.actions[0], SkillStep):
            chunk.append(self.actions.pop(0))
            self.expected.pop(0)
        chunk += [chunk[-1]] * (self.horizon - len(chunk))
        return np.array(chunk)

    def act_skill(self, observation):
        """Runs the current skill until the environment confirms its success condition, or it times out; then adds
        what it confirmed to the session's world and replans."""
        step, policy, num_actions = self.skill_run
        skill = step.skill
        succeeded = observation.conditions.get(condition_key(skill.success(step.args)), False)
        if succeeded or (num_actions >= skill.timeout):
            self.skill_run = None
            if not succeeded:
                logger.warning(f'{step} did not succeed within {skill.timeout} actions')
            if self.replans >= self.max_replans:
                self.status, self.actions = 'failed', []
            else:
                self.replan(observation, confirmed=skill.confirms(step.args) if succeeded else ())
            return self.act(observation)
        chunk = list(policy.act(observation, step.args, skill.instruction(step.args)))
        chunk = chunk[: self.horizon] + [chunk[-1]] * max(0, self.horizon - len(chunk))
        self.skill_run[2] += len(chunk)
        return np.array(chunk)

    def close(self):
        if self.session is not None:
            self.session.close()
        self.world.close()


def interpolate(path):
    """path with intermediate configurations so that no joint moves more than MAX_JOINT_STEP between them."""
    if len(path) < 2:
        return [tuple(conf) for conf in path]
    confs = [tuple(path[0])]
    for start, end in zip(path, path[1:]):
        start, end = np.array(start), np.array(end)
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / MAX_JOINT_STEP)))
        confs.extend(tuple(start + (end - start) * (i / steps)) for i in range(1, steps + 1))
    return confs
