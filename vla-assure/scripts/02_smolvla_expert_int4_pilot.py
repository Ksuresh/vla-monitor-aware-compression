"""Matched BF16 vs fake-INT4 SmolVLA synthetic inference pilot.

This is NOT real INT4 deployment, robot success, or failure-monitor calibration.
"""
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from transformers import AutoProcessor
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.envs.configs import LiberoEnv
from assure.quant import QuantSpec, apply_fake_quant

def main():
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]["smolvla"]
    checkpoint = config["checkpoint"]
    pcfg = PreTrainedConfig.from_pretrained(checkpoint)
    pcfg.pretrained_path = checkpoint
    pcfg.device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print("Loading:", checkpoint, flush=True)
    policy = make_policy(cfg=pcfg, env_cfg=LiberoEnv(task="libero_spatial"))
    policy.eval()
    device = next(policy.parameters()).device
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    tokens = processor.tokenizer("Pick up the red cube and place it on the plate.", return_tensors="pt")
    batch = {
        "observation.images.image": torch.full((1,3,256,256), .5, device=device),
        "observation.images.image2": torch.full((1,3,256,256), .5, device=device),
        "observation.state": torch.zeros((1,8), device=device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].to(device=device, dtype=torch.bool),
    }
    # Use identical noise for both denoising runs.
    torch.manual_seed(123)
    noise = torch.randn((1, pcfg.chunk_size, pcfg.max_action_dim), device=device)
    modules = dict(policy.named_modules())
    taps = config["tap"]
    captured = {name: [] for name in taps}
    handles = []
    for label, path in taps.items():
        def hook(_module, _inputs, output, name=label):
            if isinstance(output, torch.Tensor):
                captured[name].append(output.detach().float().cpu().clone())
            else:
                raise RuntimeError(f"Unexpected non-tensor hook output for {name}")
        handles.append(modules[path].register_forward_hook(hook))

    def run():
        for key in captured:
            captured[key].clear()
        # Clone because LeRobot may mutate batch/queues.
        policy.reset()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(
                {k: v.clone() for k, v in batch.items()}, noise=noise.clone()
            ).detach().float().cpu()
        result = {k: [x.clone() for x in v] for k, v in captured.items()}
        return actions, result

    try:
        baseline_actions, baseline_taps = run()
        print("Baseline actions:", tuple(baseline_actions.shape), flush=True)
        # Quantize only action expert linear weights in this first low-risk pilot.
        # The VLM remains unchanged; VLM feature drift is expected near zero.
        spec = QuantSpec(bits=4, group_size=128,
                         include=["model.vlm_with_expert.lm_expert.*"])
        report = apply_fake_quant(policy, spec)
        if report["linear_quantized"] == 0:
            raise RuntimeError("No expert linear weights were quantized; check include pattern")
        print("Quantized linear modules:", report["linear_quantized"], flush=True)
        quant_actions, quant_taps = run()

        def metrics(a, b):
            if a.shape != b.shape:
                raise RuntimeError(f"Tensor shapes differ: {a.shape} vs {b.shape}")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError("Non-finite values detected")
            delta = a - b
            return {
                "shape": list(a.shape),
                "mae": float(delta.abs().mean()),
                "rmse": float(delta.square().mean().sqrt()),
                "max_abs": float(delta.abs().max()),
                "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a).clamp_min(1e-8)),
            }
        tap_metrics = {}
        for name in taps:
            x, y = baseline_taps[name], quant_taps[name]
            if not x or len(x) != len(y):
                raise RuntimeError(f"Hook count mismatch for {name}: {len(x)} vs {len(y)}")
            tap_metrics[name] = {
                "call_count": len(x),
                "per_call": [metrics(a,b) for a,b in zip(x,y)]
            }
        result = {
            "checkpoint": checkpoint,
            "baseline_precision": "model-native mixed/bfloat16",
            "quantization": "fake weight-only symmetric INT4, action-expert Linear layers only",
            "noise_seed": 123,
            "input": "synthetic constant images, zero state, fixed instruction",
            "quantization_report": report,
            "action_drift": metrics(baseline_actions, quant_actions),
            "feature_drift": tap_metrics,
            "limitations": "No LIBERO rollouts, no ground-truth failure labels, no calibration, no real INT4 kernels or speed/memory measurements."
        }
        out = ROOT / "results/pilot/smolvla_expert_fake_int4.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        print("PASS: matched baseline vs fake INT4 pilot completed", flush=True)
    finally:
        for handle in handles:
            handle.remove()

if __name__ == "__main__":
    main()
