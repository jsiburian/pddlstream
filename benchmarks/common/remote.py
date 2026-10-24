"""A client of the planner server (server.py), with only the standard library and numpy, so that simulators (the
stand-in or RoboLab's inference client) can use it without importing pddlstream."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

from benchmarks.common.protocol import to_json


class PlannerClient:
    def __init__(self, host='localhost', port=8765, timeout=600):
        self.url, self.timeout = f'http://{host}:{port}', timeout
        self.last_response = None  # The planner's last response, with its plan reports
        self.delegate = None  # The instruction of the skill step the client performs now (see act)

    def post(self, path, request):
        data = json.dumps(to_json(request)).encode()
        http_request = urllib.request.Request(self.url + path, data, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                self.last_response = json.loads(response.read())
                return self.last_response
        except urllib.error.HTTPError as error:
            raise RuntimeError(json.loads(error.read()).get('error')) from error

    def reset(self, task, scene, observation, env_id=0, **options):
        """Plans for a task; returns the planner's status ('executing', 'failed' or 'unsupported')."""
        request = {'env_id': env_id, 'task': task, 'scene': scene, 'observation': observation, 'options': options}
        return self.post('/reset', request)

    def act(self, observation, env_id=0):
        """The next chunk of actions (horizon, 8) and the planner's status. The actions are None when the client is
        to perform a skill's step itself (e.g. with a VLA); its instruction is then in self.delegate."""
        response = self.post('/act', {'env_id': env_id, 'observation': observation})
        self.delegate = response.get('delegate')
        actions = response['actions']
        return (None if actions is None else np.array(actions)), response['status']

    def close(self, env_id=0):
        return self.post('/close', {'env_id': env_id})


def free_port():
    with socket.socket() as sock:
        sock.bind(('localhost', 0))
        return sock.getsockname()[1]


def start_server(port=None, timeout=120):
    """Starts a planner server in a new process and waits until it answers. Returns (process, PlannerClient)."""
    port = free_port() if port is None else port
    command = [sys.executable, '-m', 'benchmarks.common.server', '--port', str(port)]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # The pddlstream checkout
    path = os.pathsep.join(filter(None, [root, os.environ.get('PYTHONPATH')]))
    # Plans repeat only under a fixed hash seed (FastDownward's translator iterates string-keyed sets)
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED', '0'))
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True)
    process.stdout.readline()  # 'Listening on ...'
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(('localhost', port), timeout=1):
                return process, PlannerClient(port=port)
        except OSError:
            time.sleep(0.1)
    process.kill()
    raise RuntimeError('The planner server did not start')
