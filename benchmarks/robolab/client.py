"""RoboLab's side: an InferenceClient that sends observations to the planner server and returns its action chunks.

This module imports robolab, so it only runs where RoboLab is installed (Isaac Sim 5.0, Isaac Lab 2.2, a GPU). It does
not import pddlstream: the planner runs in its own process (server.py), which keeps it out of the simulator's
process and GPU memory.

Assumptions to check with the probe (benchmarks/robolab/probe.py) on the first run, where RoboLab's documentation does
not say: the layout of gt_state (see objects_from_gt_state), and the gripper's encoding (0 open, 1 closed here).
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
    quaternions. Assumed layout: state['objects'][name] = {'pos': [...], 'quat': [...]}."""
    objects = {}
    for name, data in state['objects'].items():
        position = data.get('pos', data.get('position'))
        quaternion = data.get('quat', data.get('orientation'))
        objects[name] = ObjectState(np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float))
    return objects


def conditions_from_obs(raw_obs, task, env_id):
    """Which of the task's success conditionals hold (bool by condition_key), for confirming learned skills. RoboLab
    tracks subtask completion inside its environment; until it is wired in here (on the GPU machine), nothing is
    reported, so a skill's step counts as failed after its timeout."""
    return {}


def task_spec_from_robolab(task):
    """A TaskSpec from a RoboLab Task: its instruction, the conditionals of its success terminations and its objects."""
    instruction = task.instruction if isinstance(task.instruction, dict) else {'default': task.instruction}
    success = []
    for name in dir(task.terminations):
        term = getattr(task.terminations, name)
        func = getattr(term, 'func', None)
        if (func is None) or getattr(term, 'time_out', False):
            continue
        success.append(Conditional(func.__name__, dict(getattr(term, 'params', {}) or {})))
    name = task.task_name or type(task).__name__
    return TaskSpec(name, instruction, tuple(success), tuple(task.contact_object_list or ()))


class TampInferenceClient(InferenceClient):
    """Plans each episode with the planner server and plays back its joint-position action chunks."""

    open_loop_horizon = 8

    def __init__(self, task, scene, remote_host='localhost', remote_port=8765, skills=(), **kwargs):
        super().__init__(**kwargs)
        self.task = task if isinstance(task, TaskSpec) else task_spec_from_robolab(task)
        self.scene = scene  # ObjectSpecs (from benchmarks/robolab/export_assets.py)
        self.skills = list(skills)  # Names of learned skills (benchmarks.common.skills) the planner may use
        self.planner = PlannerClient(remote_host, remote_port)
        self.started = set()  # Environments whose episode has been planned

    def _extract_observation(self, raw_obs, *, env_id=0):
        state = self._get_env_gt_state(raw_obs, env_id)
        if DUMP and not os.path.exists(DUMP):
            with open(DUMP, 'w') as file:
                json.dump(state, file, default=lambda value: np.asarray(value).tolist())
        arm = self._to_numpy(raw_obs['proprio_obs']['arm_joint_pos'], env_id)
        gripper = float(np.ravel(self._to_numpy(raw_obs['proprio_obs']['gripper_pos'], env_id))[0])
        conditions = conditions_from_obs(raw_obs, self.task, env_id)
        return {'env_id': env_id, 'observation': Observation(arm, gripper, objects_from_gt_state(state), conditions)}

    def _pack_request(self, extracted_obs, instruction):
        return extracted_obs

    def _query_server(self, request):
        env_id, observation = request['env_id'], request['observation']
        if env_id not in self.started:
            options = {'skills': self.skills} if self.skills else {}
            self.planner.reset(self.task, self.scene, observation, env_id=env_id, **options)
            self.started.add(env_id)
        actions, _ = self.planner.act(observation, env_id=env_id)
        return actions

    def _unpack_response(self, response):
        return np.asarray(response, dtype=np.float32)

    def reset(self, env_id=0):
        super().reset(env_id=env_id)
        self.started.discard(env_id)
