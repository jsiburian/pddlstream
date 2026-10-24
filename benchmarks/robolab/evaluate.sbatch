#!/bin/bash
#SBATCH --job-name=pddlstream-robolab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Evaluates pddlstream as a RoboLab policy on one GPU node: the planner server runs in pddlstream's pixi environment
# and RoboLab's evaluation in RoboLab's uv environment (benchmarks/robolab/README.md sets both up). Submit from the
# pddlstream checkout, with your account and RTX GPU nodes (Isaac Sim needs RTX GPUs); arguments go to run.py:
#   export OMNI_KIT_ACCEPT_EULA=YES  # Accepts NVIDIA's Omniverse license
#   sbatch --account=<account> --constraint=<nodes> --output=robolab-%j.out \
#       benchmarks/robolab/evaluate.sbatch --task BananaInBowlTask --num-envs 1
set -euo pipefail
: "${OMNI_KIT_ACCEPT_EULA:?Isaac Sim needs NVIDIA's Omniverse license accepted: export OMNI_KIT_ACCEPT_EULA=YES}"
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
ROBOLAB=$ROOT/benchmarks/robolab/RoboLab
PORT=$((20000 + RANDOM % 20000))

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd "$ROOT"
pixi run python -m benchmarks.common.server --port "$PORT" &
SERVER=$!
trap 'kill $SERVER' EXIT

cd "$ROBOLAB"
PYTHONPATH=$ROOT "$ROBOLAB/.venv/bin/python" -m benchmarks.robolab.run --headless --remote-port "$PORT" "$@"
