"""The planner as a server, the way RoboLab runs policies: python -m benchmarks.common.server --port 8765

It runs a TampPolicy for one environment (start a server per environment) and answers JSON requests (standard library
only):
  POST /reset {"env_id", "task", "scene", "observation", "options"} -> {"status"}
  POST /act   {"env_id", "observation"}                             -> {"actions", "status"}
  POST /close {"env_id"}                                            -> {"status"}
Running the planner in its own process keeps its PyBullet mirror separate from the simulator, and pddlstream's GPL
code out of the simulator's process.
"""

import argparse
import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from benchmarks.common.goals import UnsupportedTask
from benchmarks.common.policy import TampPolicy
from benchmarks.common.protocol import object_spec_from_json, observation_from_json, task_from_json, to_json

POLICIES = {}  # TampPolicy by env_id; at most one, since pybullet_tools uses one PyBullet client per process


def reset(request):
    env_id = request.get('env_id', 0)
    for policy in POLICIES.values():
        policy.close()
    POLICIES.clear()
    scene = [object_spec_from_json(spec) for spec in request['scene']]
    policy = TampPolicy(scene, **request.get('options', {}))
    POLICIES[env_id] = policy
    task = task_from_json(request['task'])
    print(f'Reset {task.name}: scene {[(spec.name, spec.kind, len(spec.points)) for spec in scene]}', flush=True)
    try:
        policy.reset(task, observation_from_json(request['observation']))
    except UnsupportedTask as error:
        policy.status = 'unsupported'
        print(f'  unsupported: {error}', flush=True)
        return {'status': policy.status, 'reason': str(error)}
    print(f'  {policy.status}: {policy.plans[-1] if policy.plans else None}', flush=True)
    return {'status': policy.status, 'plans': policy.plans}


def act(request):
    policy = POLICIES[request.get('env_id', 0)]
    replans, delegated = len(policy.plans), policy.delegate
    actions = policy.act(observation_from_json(request['observation']))
    if len(policy.plans) > replans:
        print(f'  replanned ({policy.status}): {policy.plans[-1]}', flush=True)
    if policy.delegate and (policy.delegate != delegated):
        print(f'  client performs: {policy.delegate}', flush=True)
    return {
        'actions': None if actions is None else to_json(actions),
        'delegate': policy.delegate,  # The instruction of a skill step the client performs, instead of actions
        'status': policy.status,
        'replans': policy.replans,
        'plans': policy.plans,
    }


def close(request):
    policy = POLICIES.pop(request.get('env_id', 0), None)
    if policy is not None:
        policy.close()
    return {'status': 'closed'}


ROUTES = {'/reset': reset, '/act': act, '/close': close}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            response, code = ROUTES[self.path](request), 200
        except Exception:
            response, code = {'error': traceback.format_exc()}, 500
        body = json.dumps(response).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # Requests are frequent; errors are returned to the client
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    server = HTTPServer((args.host, args.port), Handler)
    print(f'Listening on {args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
