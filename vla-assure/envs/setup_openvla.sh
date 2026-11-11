#!/usr/bin/env bash
# Environments for OpenVLA and OpenVLA-OFT on LIBERO. Each repo pins its own torch/transformers,
# so keep them in separate conda envs and point --repo_root at the matching checkout.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK=${WORK:-$HOME/vla-repos}
mkdir -p "$WORK" && cd "$WORK"
eval "$(conda shell.bash hook)"

# --- OpenVLA (tokenized actions)
conda create -y -n assure-openvla python=3.10
conda activate assure-openvla
git clone https://github.com/openvla/openvla.git
pip install -e openvla
pip install packaging ninja && pip install "flash-attn==2.5.5" --no-build-isolation || echo "flash-attn optional"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r openvla/experiments/robot/libero/libero_requirements.txt
pip install -r "$REPO_DIR/requirements.txt"
conda deactivate

# --- OpenVLA-OFT (continuous action chunks)
conda create -y -n assure-oft python=3.10
conda activate assure-oft
git clone https://github.com/moojink/openvla-oft.git
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e openvla-oft
pip install -e LIBERO
pip install -r openvla-oft/experiments/robot/libero/libero_requirements.txt
pip install -r "$REPO_DIR/requirements.txt"
echo "Use: conda activate assure-openvla  && python scripts/00_smoke_test.py --model openvla --repo_root $WORK/openvla"
echo "Use: conda activate assure-oft      && python scripts/00_smoke_test.py --model openvla_oft --repo_root $WORK/openvla-oft"
