"""RoboLab's side: an InferenceClient (robolab/eval/base_client.py) that sends observations to the planner server and
returns its joint-position action chunks.

It runs in RoboLab's environment (Isaac Sim 5.0, Isaac Lab 2.2, an RTX GPU) and imports robolab, not pddlstream: the
planner runs in its own process (benchmarks/common/server.py). It reads the objects' poses from the ground-truth
state, so evaluations run with --enable-gt-state (benchmarks/robolab/run.py sets it); the state's layout is RoboLab's
robolab/eval/gt_state.py.
"""

import json
import os

import numpy as np
from robolab.eval import InferenceClient

from benchmarks.common.protocol import Conditional, ObjectState, Observation, TaskSpec
from benchmarks.common.remote import PlannerClient

DUMP = os.environ.get('PDDLSTREAM_ROBOLAB_DUMP')  # A file to save the first raw ground-truth state in


def objects_from_gt_state(state):
    """ObjectStates by name from one environment's gt_state: positions in meters (env-local) and (w, x, y, z)
    quaternions, from state['objects'][name] = {'pos', 'quat', 'vel'}."""
    objects = {}
    for name, data in state['objects'].items():
        position = data.get('pos', data.get('position'))
        quaternion = data.get('quat', data.get('orientation'))
        objects[name] = ObjectState(np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float))
    return objects


def conditions_from_gt_state(state, task):
    """Which of the task's success conditionals hold (bool by condition_key), for confirming learned skills. RoboLab
    reports subtask completion (state['subtask']), but not by conditional; until skills are used on RoboLab, nothing is
    reported, so a skill's step counts as failed after its timeout."""
    return {}


def task_spec_from_robolab(task):
    """A TaskSpec from a RoboLab Task or environment configuration: its instruction, the conditionals of its success
    terminations and its objects."""
    instruction = task.instruction if isinstance(task.instruction, dict) else {'default': task.instruction}
    success = []
    for name in dir(task.terminations):
        term = getattr(task.terminations, name)
        func = getattr(term, 'func', None)
        if (func is None) or getattr(term, 'time_out', False):
            continue
        success.append(Conditional(func.__name__, dict(getattr(term, 'params', {}) or {})))
    name = getattr(task, '_task_name', None) or getattr(task, 'task_name', None) or type(task).__name__
    return TaskSpec(name, instruction, tuple(success), tuple(task.contact_object_list or ()))


class TampInferenceClient(InferenceClient):
    """Plans each episode with the planner server and plays back its joint-position action chunks. The runner sets
    the task and its scene (set_task) before a task's episodes; RoboLab's runner creates one client per task."""

    open_loop_horizon = 8

    def __init__(self, remote_host='localhost', remote_port=8765, skills=()):
        super().__init__()
        self.planner = PlannerClient(remote_host, remote_port)
        self.skills = list(skills)  # Names of learned skills (benchmarks.common.skills) the planner may use
        self.task, self.scene = None, None
        self.started = set()  # Environments whose episode has been planned

    def set_task(self, task, scene):
        """The task (a TaskSpec, or a RoboLab Task or environment configuration) and its ObjectSpecs."""
        self.task = task if isinstance(task, TaskSpec) else task_spec_from_robolab(task)
        self.scene = scene
        self.started.clear()

    def _extract_observation(self, raw_obs, *, env_id=0):
        state = self._get_env_gt_state(raw_obs, env_id)
        if state is None:
            raise RuntimeError('The planner needs the ground-truth state: run with --enable-gt-state')
        if DUMP and not os.path.exists(DUMP):
            with open(DUMP, 'w') as file:
                json.dump(state, file, default=lambda value: np.asarray(value).tolist())
        arm = self._to_numpy(self._find_obs_term(raw_obs, 'arm_joint_pos'), env_id)
        gripper = float(np.ravel(self._to_numpy(self._find_obs_term(raw_obs, 'gripper_pos'), env_id))[0])
        objects = objects_from_gt_state(state)
        return {
            'env_id': env_id,
            'observation': Observation(arm, gripper, objects, conditions_from_gt_state(state, self.task)),
        }

    def _pack_request(self, extracted_obs, instruction):
        return extracted_obs

    def _query_server(self, request):
        env_id, observation = request['env_id'], request['observation']
        if env_id not in self.started:
            options = {'gripper': 'robotiq', 'containment': 'center', 'skills': self.skills}
            self.planner.reset(self.task, self.scene, observation, env_id=env_id, **options)
            self.started.add(env_id)
        actions, _ = self.planner.act(observation, env_id=env_id)  # An unsupported task holds the arm still
        return actions

    def _unpack_response(self, response):
        return np.asarray(response, dtype=np.float32)

    def reset(self, *, env_id=None):
        super().reset(env_id=env_id)
        if env_id is None:
            self.started.clear()
        else:
            self.started.discard(env_id)
