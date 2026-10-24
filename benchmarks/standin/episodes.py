"""Episodes of the TAMP policy on the stand-in simulator, optionally recorded step by step, and demonstrations: the
recorded episodes that succeeded with verified plans, saved for training learned policies and replayed to check them.

An episode file (.npz) holds, for T actions:
  actions (T, 8): joint positions and gripper command; arm (T+1, 7), gripper (T+1,): the observed robot state;
  positions (T+1, N, 3), quaternions (T+1, N, 4, wxyz): the observed object poses, in the order of object_names;
  metadata: JSON with the task, instruction, seed, initial positions, perturbation, plan reports and outcome.
An index.jsonl next to the episodes lists each one's metadata.
"""

import dataclasses
import json
import pathlib
import time

import numpy as np

from benchmarks.standin.env import SCENARIOS, StandinEnv

PERTURB_STEP = 16  # Actions before the perturbation, which moves the last object the robot has not moved 5 cm in -x


def run_episode(client, scenario, max_steps=3000, perturb=False, record=False):
    """Runs scenario with the planner behind client. Returns the episode's result, with its recording (arrays) under
    'recording' if record."""
    env = StandinEnv(scenario)
    start = time.time()
    options = {'skills': list(scenario.skills)} if scenario.skills else {}
    try:
        observation = env.observe()
        observations, actions, perturbation = [observation], [], None
        response = client.reset(scenario.task, scenario.objects, observation, **options)
        status, steps = response['status'], 0
        while status == 'executing' and steps < max_steps:
            untouched = [name for name in sorted(env.world.movables) if not moved(env, name)]
            if perturb and (perturbation is None) and (steps >= PERTURB_STEP) and (env.held is None) and untouched:
                name = untouched[-1]  # Someone moves an object the robot has not handled yet
                x, y, _ = env.world.get_pose(name)[0]
                perturbation = {'step': steps, 'object': name, 'point': [x - 0.05, y]}
                env.move(name, perturbation['point'])
                observation = observations[-1] = env.observe()
            chunk, status = client.act(observation)
            for action in chunk:
                observation = env.step(action)
                observations.append(observation)
                actions.append(np.array(action))
                steps += 1
        _, status = client.act(observation)  # Reports done once the last chunk was executed
        result = {
            'task': scenario.task.name,
            'status': status,
            'success': env.success(),
            'steps': steps,
            'time': round(time.time() - start, 2),
            'reason': response.get('reason'),
            'plans': (client.last_response or {}).get('plans', []),
            'perturbation': perturbation,
        }
        if record:
            result['recording'] = recording(env, observations, actions)
        return result
    finally:
        env.close()


def moved(env, name, tolerance=1e-3):
    """Whether an object has left its initial position."""
    return np.linalg.norm(np.array(env.world.get_pose(name)[0]) - np.array(env.initial[name])) > tolerance


def recording(env, observations, actions):
    names = sorted(env.world.bodies)
    return {
        'object_names': np.array(names),
        'actions': np.array(actions).reshape(-1, 8),
        'arm': np.array([o.arm for o in observations]),
        'gripper': np.array([o.gripper for o in observations]),
        'positions': np.array([[o.objects[n].position for n in names] for o in observations]),
        'quaternions': np.array([[o.objects[n].quaternion for n in names] for o in observations]),
    }


##################################################


def jittered(scenario, rng, scale=0.03):
    """scenario with its objects' initial (x, y) shifted by up to scale meters; an object resting on another moves with
    it."""
    offsets, positions = {}, {}
    for name, (x, y, *support) in scenario.positions.items():  # Supports are listed before what they support
        offsets[name] = offsets[support[0]] if support else rng.uniform(-scale, scale, size=2)
        positions[name] = (x + offsets[name][0], y + offsets[name][1], *support)
    return dataclasses.replace(scenario, positions=positions)


def is_demonstration(result):
    """Whether an episode can be a demonstration: it succeeded, and every plan it executed has a certificate that
    verifies independently (up to the assumptions of its skills)."""
    plans = [report for report in result['plans'] if report.get('plan') is not None]
    return result['success'] and bool(plans) and all(report['verified'] for report in plans)


def generate(client, directory, scenarios=SCENARIOS, seeds=(0,), perturb=(False,), scale=0.03, max_steps=3000):
    """Runs each scenario for each seed (jittering its initial positions) and perturbation setting, and saves the
    episodes that are demonstrations. Returns the results of all episodes (without recordings)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    with open(directory / 'index.jsonl', 'a') as index:
        for scenario in scenarios:
            for seed in seeds:
                for perturbed in perturb:
                    variant = jittered(scenario, np.random.default_rng(seed), scale)
                    result = run_episode(client, variant, max_steps, perturbed, record=True)
                    arrays = result.pop('recording')
                    result.update(
                        seed=seed,
                        instruction=scenario.task.instruction.get('default'),
                        positions={name: list(map(_float, p)) for name, p in variant.positions.items()},
                        demonstration=is_demonstration(result),
                    )
                    if result['demonstration']:
                        name = f'{scenario.task.name}_seed{seed}{"_perturbed" if perturbed else ""}.npz'
                        np.savez_compressed(directory / name, metadata=json.dumps(result), **arrays)
                        index.write(json.dumps({'file': name, **result}) + '\n')
                    results.append(result)
    return results


def _float(value):
    return value if isinstance(value, str) else float(value)


def load(path):
    """An episode file's arrays and metadata."""
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != 'metadata'}
        return arrays, json.loads(str(data['metadata']))


def replay(path):
    """Executes an episode's actions open loop in the stand-in, from its initial positions and with its perturbation.
    Returns whether the task succeeded."""
    arrays, metadata = load(path)
    scenario = next(s for s in SCENARIOS if s.task.name == metadata['task'])
    positions = {name: tuple(position) for name, position in metadata['positions'].items()}
    env = StandinEnv(dataclasses.replace(scenario, positions=positions))
    perturbation = metadata.get('perturbation')
    try:
        for step, action in enumerate(arrays['actions']):
            if (perturbation is not None) and (step == perturbation['step']):
                env.move(perturbation['object'], perturbation['point'])
            env.step(action)
        return env.success()
    finally:
        env.close()
