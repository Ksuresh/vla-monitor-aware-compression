"""Inspect SmolVLA action-expert blocks before block-wise quantization experiments.

This script loads the real checkpoint but does not run inference or quantization.
"""
import json
from pathlib import Path
import torch
import yaml
from torch import nn
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.envs.configs import LiberoEnv

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]["smolvla"]
checkpoint = cfg["checkpoint"]
pcfg = PreTrainedConfig.from_pretrained(checkpoint)
pcfg.pretrained_path = checkpoint
pcfg.device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA T4 required")
policy = make_policy(cfg=pcfg, env_cfg=LiberoEnv(task="libero_spatial"))
expert_path = "model.vlm_with_expert.lm_expert"
modules = dict(policy.named_modules())
expert = modules[expert_path]
print("Action expert:", type(expert).__name__, flush=True)

# Detect immediate block collection from actual loaded module tree.
candidates = []
for name, module in expert.named_modules():
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        children = list(module.named_children())
        if len(children) >= 2 and all(k.isdigit() for k, _ in children):
            candidates.append((name, module, len(children)))
if not candidates:
    print("No repeated block collection found. Direct children:", flush=True)
    for name, module in expert.named_children():
        print(name, type(module).__name__, flush=True)
    raise RuntimeError("Review expert structure before proceeding")
# Choose largest repeated collection; print alternatives for transparency.
candidates.sort(key=lambda x: (-x[2], len(x[0])))
collection_name, collection, count = candidates[0]
print("Candidate collections:", [(n, c) for n, _, c in candidates], flush=True)
blocks = []
for i, block in enumerate(collection):
    full_name = f"{expert_path}.{collection_name}.{i}"
    linears = [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]
    blocks.append({
        "index": i,
        "path": full_name,
        "class": type(block).__name__,
        "linear_layers": len(linears),
        "linear_weight_parameters": sum(m.weight.numel() for _, m in linears),
        "linear_examples": [n for n, _ in linears[:3]]
    })
result = {
    "checkpoint": checkpoint,
    "expert_path": expert_path,
    "block_collection": collection_name,
    "block_count": len(blocks),
    "blocks": blocks,
    "limitations": "Architecture inspection only; no quantization sensitivity or robot evaluation."
}
out = ROOT / "results/pilot/smolvla_expert_blocks.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
print("PASS: action-expert block structure inspected", flush=True)
