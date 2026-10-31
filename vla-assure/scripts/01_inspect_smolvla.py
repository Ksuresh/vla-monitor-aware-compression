"""Read-only SmolVLA checkpoint and hook inspection; no LIBERO simulator required.

Run after installing a compatible LeRobot version:
  python scripts/01_inspect_smolvla.py --device cuda

This script downloads model weights; GPU/cloud charges may apply.
It does NOT run inference or claim that hooks fire.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/pilot/smolvla_inspection.json")
    args = parser.parse_args()

    import torch
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]["smolvla"]
    ckpt = args.checkpoint or cfg["checkpoint"]
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA unavailable; no GPU charged by this script itself.")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy
    from lerobot.envs.configs import LiberoEnv

    print("Loading checkpoint:", ckpt, flush=True)
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = args.device
    policy = make_policy(cfg=pcfg, env_cfg=LiberoEnv(task="libero_spatial", task_ids=[0]))
    policy.eval()
    names = dict(policy.named_modules())
    taps = {key: {"configured_path": path, "exists": path in names}
            for key, path in cfg["tap"].items()}
    candidates = [name for name in names if any(
        word in name.lower() for word in ("norm", "expert", "vlm")
    )][-120:]
    result = {
        "checkpoint": ckpt,
        "device": args.device,
        "torch_version": torch.__version__,
        "parameter_count": sum(p.numel() for p in policy.parameters()),
        "linear_module_count": sum(isinstance(m, torch.nn.Linear) for m in policy.modules()),
        "taps": taps,
        "candidate_modules": candidates,
        "limitations": "Module existence only; no hook firing, action prediction, or LIBERO rollout tested.",
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidate_modules"}, indent=2))
    if not all(t["exists"] for t in taps.values()):
        print("WARNING: configured feature taps need correction before experiment.")
        return 2
    print("PASS: model loads and configured module paths exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
