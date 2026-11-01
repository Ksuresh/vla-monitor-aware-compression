"""Synthetic SmolVLA inference smoke test: verify two feature hooks fire.

Not a LIBERO rollout or a failure-monitor calibration experiment.
Run: python scripts/01_verify_feature_hooks.py --device cuda
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    import torch
    import yaml
    from transformers import AutoProcessor
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.envs.configs import LiberoEnv

    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]["smolvla"]
    ckpt = config["checkpoint"]
    print("Loading:", ckpt, flush=True)
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = args.device
    policy = make_policy(cfg=pcfg, env_cfg=LiberoEnv(task="libero_spatial"))
    policy.eval()
    device = next(policy.parameters()).device
    print("Model device:", device, flush=True)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    tok = processor.tokenizer("Pick up the red cube and place it on the plate.", return_tensors="pt")
    batch = {
        "observation.images.image": torch.full((1, 3, 256, 256), 0.5, device=device),
        "observation.images.image2": torch.full((1, 3, 256, 256), 0.5, device=device),
        "observation.state": torch.zeros((1, 8), device=device),
        "observation.language.tokens": tok["input_ids"].to(device),
        "observation.language.attention_mask": tok["attention_mask"].to(device=device, dtype=torch.bool),
    }

    observations = {name: [] for name in config["tap"]}
    handles = []
    modules = dict(policy.named_modules())
    for name, path in config["tap"].items():
        if path not in modules:
            raise RuntimeError(f"Missing hook module: {name}: {path}")
        def record(module, inputs, output, label=name):
            tensors = []
            def collect(value):
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
                elif isinstance(value, (tuple, list)):
                    for v in value:
                        collect(v)
            collect(output)
            observations[label].append([
                {"shape": list(t.shape), "dtype": str(t.dtype), "finite": bool(torch.isfinite(t).all().item())}
                for t in tensors
            ])
        handles.append(modules[path].register_forward_hook(record))

    try:
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
        result = {
            "checkpoint": ckpt,
            "test_type": "synthetic constant-image, zero-state inference smoke test",
            "action_shape": list(actions.shape),
            "actions_finite": bool(torch.isfinite(actions).all().item()),
            "hooks": {
                name: {"call_count": len(calls), "first_call_tensors": calls[0] if calls else []}
                for name, calls in observations.items()
            },
            "limitations": "No physical observations, failure labels, quantization, calibration, or robot rollout."
        }
        output = ROOT / "results/pilot/smolvla_feature_hooks.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["actions_finite"] or any(
            x["call_count"] == 0 or not x["first_call_tensors"] or
            not all(t["finite"] for t in x["first_call_tensors"])
            for x in result["hooks"].values()
        ):
            raise RuntimeError("Hook verification incomplete")
        print("PASS: action output and both monitoring hooks are finite", flush=True)
    finally:
        for handle in handles:
            handle.remove()

if __name__ == "__main__":
    main()
