"""The TAMP policy: plans a task from an observation with a PlannerSession, returns the plan as chunks of joint-position
actions, and replans when objects are not where the plan expects them. Plan steps that learned skills perform
(skills.py) are run by the skill's policy until the environment confirms them; then the policy replans.
"""

import hashlib
import logging
import os

import numpy as np

from benchmarks.common.goals import goal_from_task, goal_literals
from benchmarks.common.primitives import Conf, Grasp, Pose, collides, get_procedures, get_stream_info, seed, solve_ik
from benchmarks.common.protocol import condition_key, pose_from_state
from benchmarks.common.skills import SKILLS, SkillStep, compile_skills
from benchmarks.common.world import World, pbu
from pddlstream.algorithms.session import PlannerSession, WorldPatch
from pddlstream.algorithms.validate import verify_certificate
from pddlstream.core.conversion import values_from_objects
from pddlstream.core.problem import PDDLProblem
from pddlstream.core.utils import read

logger = logging.getLogger(__name__)
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TABLE = 'table'
OPEN, CLOSED = 0.0, 1.0
MAX_JOINT_STEP = 0.05  # Radians between consecutive actions
TOLERANCE = 0.03  # Meters an object may be from where the plan expects it before replanning
RETREAT = 0.1  # Meters the tool rises before replanning when the arm touches an object (e.g. after a slipped grasp)
HELD_TOLERANCE = 0.08  # Meters a held object may be from where the gripper should carry it before it counts as dropped
LIFT = 0.03  # Meters above the table that a picked-up object's lowest point rises
NEAR_GRIPPER = 0.15  # Meters from the tool point to the center of an object the gripper holds
UPRIGHT = 0.99  # The least cosine between an object's z axis and the vertical for its frame to count as upright
FOCUS_TIME = 60  # Seconds for the solve with the objects the goal does not mention fixed (see focus)


class Reobserve:
    """A step of the policy's actions: plan the rest from the grasp of the object it holds, as observed."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'Reobserve({self.name})'


def lowest_point(spec, pose):
    """The height of an object's lowest point at pose: its box's bottom, or its lowest surface point."""
    if not spec.points:
        return pose[0][2] - spec.size[2] / 2
    rotation = pbu.matrix_from_quat(pose[1])
    return pose[0][2] + (np.asarray(spec.points) @ rotation.T)[:, 2].min()


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
        reobserve=None,
        focus=False,
        focus_time=FOCUS_TIME,
    ):
        """reobserve: whether to replan from the observed grasp after lifting an object (by default, the gripper's).
        focus: whether to plan first with the movable objects that the goal does not mention fixed (as obstacles),
        which cluttered scenes need, and with all of them movable only if that finds no plan within focus_time."""
        self.focus, self.focus_time = focus, focus_time
        self.world = World(scene, gripper=gripper, containment=containment)
        self.reobserve = self.world.gripper.reobserve if reobserve is None else reobserve
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
        self.held = None  # The object and grasp that the session's world holds
        self.result = None  # The last solve's SolveResult
        self.frames = {}  # The orientations (of their own frames) of objects planned in an upright frame (see align)
        self.own_points = {}  # The surface points of objects given by points, in their own frames
        self.delegate = None  # The instruction of the skill step that the client performs now

    def reset(self, task, observation):
        seed(0)
        self.goal = goal_from_task(task, skills=self.skills.values())
        self.align(observation)
        self.poses = {
            name: Pose(self.world, name, self.object_pose(name, state)) for name, state in observation.objects.items()
        }
        self.poses[TABLE] = Pose(self.world, TABLE, ((0, 0, 0), (0, 0, 0, 1)))
        bottoms = [
            lowest_point(self.world.specs[name], self.poses[name].value)
            for name in observation.objects
            if name in self.world.specs
        ]
        if bottoms:  # The table is where the lowest object rests
            self.world.set_table_top(min(bottoms))
        for pose in self.poses.values():  # The mirror starts where the objects are
            pose.assign()
        self.conf = Conf(observation.arm)
        self.domain_pddl = compile_skills(read(os.path.join(DIRECTORY, 'domain.pddl')), self.skills.values())
        self.replans, self.skill_run, self.plans, self.prefix, self.held = 0, None, [], [], None
        obstacles = self.obstacles() if self.focus else []
        for name in obstacles:  # Fixed for a first, smaller solve
            self.world.set_kind(name, 'obstacle')
        self.start(max_time=min(self.max_time, self.focus_time) if obstacles else self.max_time)
        if obstacles and (self.status == 'failed'):  # Maybe one is in the way: solve again with them all movable
            logger.info(f'No plan with {obstacles} fixed: planning with them movable')
            for name in obstacles:
                self.world.set_kind(name, 'movable')
            self.session.close()
            self.start()

    def obstacles(self):
        """The movable objects that the goal does not mention."""
        mentioned = {argument for literal in goal_literals(self.goal) for argument in literal[1:]}
        return [name for name in self.world.movables if (name in self.poses) and (name not in mentioned)]

    def start(self, max_time=None):
        """Starts a planner session for the current world and goal, and plans."""
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
        self.plan(max_time)

    def get_init(self):
        movables = set(self.world.movables) & set(self.poses)
        literals = goal_literals(self.goal)
        supports = {support for name, _, support in filter(lambda literal: literal[0] == 'On', literals)}
        buttons = {name for name, spec in self.world.specs.items() if spec.kind == 'button'}
        obstacles = {name for name, spec in self.world.specs.items() if spec.kind == 'obstacle'}
        init = [('Conf', self.conf), ('AtConf', self.conf), ('HandEmpty',), ('CanMove',)]
        for name, pose in self.poses.items():
            kind = 'Movable' if name in movables else 'Button' if name in buttons else 'Surface'
            init += [(kind, name), ('Pose', name, pose), ('AtPose', name, pose)]
        for name in movables:
            surfaces = [
                other
                for other in self.poses
                if ((other not in movables) and (other not in buttons | obstacles)) or (other in supports - {name})
            ]
            init += [('Stackable', name, surface) for surface in surfaces]
        return init

    def plan(self, max_time=None):
        result = self.result = self.session.solve(max_time=self.max_time if max_time is None else max_time)
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
                world.set_pose(name, self.object_pose(name, state))
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
        assumptions = []
        for name, args in result.plan:
            if name in self.skills:
                skill = self.skills[name]
                assumptions += skill.confirms(args) + ([('Holding', skill.holds(args))] if skill.holds else [])
        return {
            'plan': [f'{name}({", ".join(map(str, args))})' for name, args in result.plan],
            'certificate': hashlib.sha256(repr(supports).encode()).hexdigest()[:16],
            'verified': verified,
            'issues': issues,
            'assumptions': [f'({" ".join(map(str, fact))})' for fact in assumptions],
        }

    def compile(self, plan):
        """The plan as actions [7 joint positions, gripper], with where each object is expected after each: {name:
        (position, kind)}, where kind is 'rest', 'held' (expected where the gripper carries it) or 'placed' (released
        by the plan, and free to settle)."""
        actions, expected = [], []
        held = self.held  # (name, Grasp) that the plan starts holding
        positions = {
            name: (pose.value[0], 'rest') for name, pose in self.poses.items() if not (held and name == held[0])
        }
        gripper = CLOSED if held else OPEN

        def add(path):
            for conf in interpolate(path):
                actions.append(np.array([*conf, gripper]))
                entry = dict(positions)
                if held is not None:
                    entry[held[0]] = (self.carried(conf, held[1]), 'held')
                expected.append(entry)

        for action in plan:
            name, args = action
            if name in self.skills:  # Plans after a skill are made once the skill has run
                actions.append(SkillStep(self.skills[name], args))
                expected.append(dict(positions))
                break
            if name in ('move_free', 'move_holding'):
                add(args[-1].path)
            elif name in ('pick', 'place'):
                obj, pose, grasp, _, trajectory = args
                add(trajectory.path)
                add([trajectory.path[-1]] * self.world.gripper.arrive_steps)
                gripper = CLOSED if name == 'pick' else OPEN
                if name == 'pick':
                    positions.pop(obj, None)
                    held = (obj, grasp)
                else:
                    positions[obj] = (pose.value[0], 'placed')
                    held = None
                add([trajectory.path[-1]] * self.world.gripper.settle_steps)
                add(trajectory.path[::-1])
                if (name == 'pick') and self.reobserve:  # Plan the rest from the observed grasp
                    actions.append(Reobserve(obj))
                    expected.append(dict(expected[-1]))
                    break
        return actions, expected

    def align(self, observation):
        """Gives each object given by points whose frame is not upright (e.g. a figurine whose mesh lies on its side
        in its own frame) an upright planning frame at the same origin, with its points turned into it: grasps
        approach from above in the object's frame. self.frames holds the orientation of each such object's own frame
        when it got its planning frame; objects that tumble later get a new one (reframe)."""
        self.frames = {}
        self.own_points = {name: np.asarray(spec.points) for name, spec in self.world.specs.items() if spec.points}
        for name in self.own_points:
            if name in observation.objects:
                self.reframe(name, pose_from_state(observation.objects[name])[1])

    def reframe(self, name, orientation):
        """Gives an object an upright planning frame, from its own frame's orientation in the world, if its current
        planning frame is not upright. Returns whether it did."""
        current = self.frames.get(name, (0.0, 0.0, 0.0, 1.0))
        relative = pbu.multiply(((0, 0, 0), orientation), pbu.invert(((0, 0, 0), current)))[1]
        if pbu.matrix_from_quat(relative)[2, 2] >= UPRIGHT:
            return False
        self.frames[name] = orientation
        self.world.set_points(name, self.own_points[name] @ pbu.matrix_from_quat(orientation).T)
        return True

    def forget(self, name):
        """Drops the session's stream results that involve an object, e.g. after its shape changed (reframe)."""
        for external in self.session.externals:
            for instance in list(external.instances.values()):
                inputs = values_from_objects(instance.input_objects)
                if name in inputs:
                    self.session.invalidate(external.name, inputs)

    def object_pose(self, name, state):
        """An observed object's pose (an ObjectState), in its planning frame (see align)."""
        pose = pose_from_state(state)
        if name in self.frames:
            pose = pbu.multiply(pose, pbu.invert(((0.0, 0.0, 0.0), self.frames[name])))
        return pose

    def containing(self, name, pose):
        """With 'center' containment, the containers that an object at pose is in, by RoboLab's object_in_container:
        its center inside the container's footprint, at most a container's height above it."""
        if (self.world.containment != 'center') or (self.world.specs[name].kind != 'movable'):
            return []
        position, containers = np.array(pose.value[0]), []
        for other, spec in self.world.specs.items():
            if (spec.kind == 'container') and (other != name):
                self.poses[other].assign()
                lower, upper = map(np.array, self.world.container_aabb(other))
                height = upper[2] - lower[2]
                if np.all(lower[:2] <= position[:2]) and np.all(position[:2] <= upper[:2]):
                    if lower[2] <= position[2] <= upper[2] + height:
                        containers.append(other)
        return containers

    def carried(self, conf, grasp):
        """Where a held object is when the arm is at conf."""
        self.world.set_arm(conf)
        return pbu.multiply(self.world.tool_pose(), grasp.value)[0]

    def deviation(self, observation, expected):
        """The first object that is not where the plan expects it, and whether the plan holds it then; or None. A held
        object may shift in the gripper and a placed one settle (e.g. drop into a bowl), so only a drop counts."""
        for name, (position, kind) in expected.items():
            if name in observation.objects:
                distance = np.linalg.norm(np.array(observation.objects[name].position) - np.array(position))
                if distance > (TOLERANCE if kind == 'rest' else HELD_TOLERANCE):
                    return name, kind == 'held'
        return None

    def replan(self, observation, confirmed=(), holding=None, count=True):
        """Updates the session's world from the observation (and the facts that a skill confirmed) and plans again.
        holding is the object the robot holds now, if any: its grasp is the one observed. Replans that count are
        limited to max_replans."""
        patch_add, patch_remove = list(confirmed), []
        if self.held and (holding != self.held[0]):  # The session's world holds an object the robot no longer holds
            name, grasp = self.held
            patch_remove += [('AtGrasp', name, grasp)]
            patch_add += [('HandEmpty',)]
            self.poses[name] = Pose(self.world, name, self.object_pose(name, observation.objects[name]))
            patch_add += [('Pose', name, self.poses[name]), ('AtPose', name, self.poses[name])]
            self.held = None
        for name, state in observation.objects.items():
            if name == holding:
                continue
            old = self.poses[name]
            tumbled = (name in self.own_points) and self.reframe(name, pose_from_state(state)[1])
            if tumbled:  # E.g. after a drop: its grasps and placements were for its old frame
                logger.info(f'{name} tumbled: planning in a new upright frame of it')
                self.forget(name)
            if tumbled or (np.linalg.norm(np.array(state.position) - np.array(old.value[0])) > TOLERANCE):
                new = Pose(self.world, name, self.object_pose(name, state))
                patch_remove += [('AtPose', name, old)]
                patch_add += [('Pose', name, new), ('AtPose', name, new)]
                patch_add += [
                    ('Supported', name, new, other, self.poses[other]) for other in self.containing(name, new)
                ]
                self.poses[name] = new
        if holding is not None and not self.held:
            self.world.set_arm(observation.arm)
            pose = self.object_pose(holding, observation.objects[holding])
            value = pbu.multiply(pbu.invert(self.world.tool_pose()), pose)
            grasp = Grasp(holding, value)
            patch_remove += [('HandEmpty',), ('AtPose', holding, self.poses[holding])]
            patch_add += [('Grasp', holding, grasp), ('AtGrasp', holding, grasp)]
            self.held = (holding, grasp)
        self.prefix = [] if self.held else self.retreat(observation)
        conf = Conf(self.prefix[-1][:7] if self.prefix else observation.arm)
        patch_remove += [('AtConf', self.conf)]
        patch_add += [('Conf', conf), ('AtConf', conf)]
        self.conf = conf
        self.session.apply(WorldPatch(add=patch_add, remove=patch_remove))
        self.replans += int(count)
        return self.plan()

    def act(self, observation):
        """The next chunk of actions, of shape (horizon, 8), or None when the client performs a skill's step (its
        instruction is in self.delegate)."""
        self.delegate = None
        if self.skill_run is not None:
            return self.act_skill(observation)
        moved = self.deviation(observation, self.expected[0]) if (self.actions and self.expected) else None
        if moved is not None:
            if self.replans >= self.max_replans:
                self.status, self.actions = 'failed', []
            else:
                name, held = moved
                carrying = [other for other, (_, kind) in self.expected[0].items() if kind == 'held']
                if held:  # The planned grasp slipped: skills that replace it may pick the object up instead
                    logger.info(f'{name} slipped out of the gripper: replanning')
                    fumbled = [('Fumbled', name)] if any('pick' in s.restricts for s in self.skills.values()) else []
                    self.replan(observation, confirmed=fumbled)
                else:
                    logger.info(f'{name} moved: replanning')
                    self.replan(observation, holding=carrying[0] if carrying else None)
        if self.actions and isinstance(self.actions[0], Reobserve):
            step = self.actions.pop(0)
            self.expected.pop(0)
            self.replan(observation, holding=step.name, count=False)
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
        while self.actions and (len(chunk) < self.horizon) and not isinstance(self.actions[0], (SkillStep, Reobserve)):
            chunk.append(self.actions.pop(0))
            self.expected.pop(0)
        chunk += [chunk[-1]] * (self.horizon - len(chunk))
        return np.array(chunk)

    def act_skill(self, observation):
        """Runs the current skill until its success condition holds (as the environment reports it, or as the
        planner observes it), or it times out; then adds what it confirmed to the session's world and replans."""
        step, policy, num_actions = self.skill_run
        skill = step.skill
        conditional = skill.success(step.args)
        succeeded = observation.conditions.get(condition_key(conditional), False) or self.observed(
            conditional, observation
        )
        if succeeded or (num_actions >= skill.timeout):
            self.skill_run = None
            if not succeeded:
                logger.warning(f'{step} did not succeed within {skill.timeout} actions')
            if self.replans >= self.max_replans:
                self.status, self.actions = 'failed', []
            elif succeeded:
                holding = skill.holds(step.args) if skill.holds else None
                self.replan(observation, confirmed=skill.confirms(step.args), holding=holding)
            else:
                self.replan(observation)
            return self.act(observation)
        actions = policy.act(observation, step.args, skill.instruction(step.args))
        if actions is None:  # The client performs it
            self.delegate = skill.instruction(step.args)
            self.skill_run[2] += self.horizon
            return None
        chunk = list(actions)
        chunk = chunk[: self.horizon] + [chunk[-1]] * max(0, self.horizon - len(chunk))
        self.skill_run[2] += len(chunk)
        return np.array(chunk)

    def observed(self, conditional, observation):
        """Whether a skill's success conditional holds, as far as the planner can tell from the objects' poses:
        object_picked_up holds when the object is off the table, next to the gripper."""
        if conditional.func != 'object_picked_up':
            return False
        name = conditional.params['object']
        if (name not in observation.objects) or (name not in self.world.specs):
            return False
        state = observation.objects[name]
        lifted = lowest_point(self.world.specs[name], self.object_pose(name, state)) > self.world.table_top + LIFT
        self.world.set_arm(observation.arm)
        near = np.linalg.norm(np.array(state.position) - np.array(self.world.tool_pose()[0])) < NEAR_GRIPPER
        return bool(lifted and near)

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
