#!/usr/bin/env bash
# Controlled "LoRA update" variant for LeRobot VLAs (spec: configs/experiment.yaml -> variants.lora_update).
#   data       LIBERO demonstrations of the evaluated suite only (HuggingFaceVLA/libero, filtered by suite)
#   objective  the policy's native training loss (flow-matching velocity MSE on action chunks)
#   adapter    LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o and MLP projections of the VLM backbone and
#              action expert; vision encoder, embeddings and normalisation statistics frozen
#   schedule   6000 steps, batch 8, lr 1e-4, seed 7
# Monitors trained on the ORIGINAL policy are then evaluated against it. Needs one 24-80 GB GPU.
#
#   bash scripts/05_lora_update.sh pi05 lerobot/pi05_libero_finetuned libero_10
set -euo pipefail
MODEL=${1:-pi05}
BASE=${2:-lerobot/pi05_libero_finetuned}
SUITE=${3:-libero_10}
STEPS=${STEPS:-6000}
OUT=outputs/lora_${MODEL}_${SUITE}
TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

lerobot-train \
  --policy.path="${BASE}" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.lora_dropout=0.05 \
  --peft.target_modules="${TARGETS}" \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero --env.task="${SUITE}" \
  --optimizer.lr=1e-4 --seed=7 \
  --steps="${STEPS}" --batch_size=8 \
  --eval_freq=0 --save_freq="${STEPS}" \
  --output_dir="${OUT}"

CKPT=$(ls -d "${OUT}"/checkpoints/*/pretrained_model | tail -1)
cat > "${CKPT}/lora_spec.json" <<JSON
{"base": "${BASE}", "suite": "${SUITE}", "data": "HuggingFaceVLA/libero (${SUITE} episodes only)",
 "objective": "native policy loss", "r": 16, "lora_alpha": 32, "dropout": 0.05,
 "target_modules": "${TARGETS}", "steps": ${STEPS}, "batch_size": 8, "lr": 1e-4, "seed": 7}
JSON
echo "LoRA checkpoint: ${CKPT}  (spec in lora_spec.json)"
echo "Set experiment.yaml -> variants.lora_update.path: ${CKPT}  then run 01_collect.py --variant lora_update"
# VERIFY on your LeRobot version: `lerobot-train --help` for the PEFT flag names, and that the dataset is
# restricted to the suite's episodes (some releases need --dataset.episodes or a suite-specific repo id).
# Record the training loss curve and the final suite success rate with the paper.
