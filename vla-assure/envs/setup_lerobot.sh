#!/usr/bin/env bash
# Environment for pi0 / pi0.5 / SmolVLA on LIBERO via LeRobot (Linux + NVIDIA GPU).
set -euo pipefail
conda create -y -n assure-lerobot python=3.12
eval "$(conda shell.bash hook)"
conda activate assure-lerobot
pip install "lerobot[pi,smolvla,libero,peft]>=0.6,<0.7" "mujoco==3.3.2"
pip install -r "$(dirname "$0")/../requirements.txt"
export MUJOCO_GL=egl
echo "pi0/pi0.5 need the gated PaliGemma tokenizer: request access to google/paligemma-3b-pt-224, then: hf auth login"
echo "Check: lerobot-eval --policy.path=lerobot/pi05_libero_finetuned --env.type=libero --env.task=libero_spatial --eval.n_episodes=2 --eval.batch_size=1"
