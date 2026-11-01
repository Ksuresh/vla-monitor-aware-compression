"""Single-block fake INT4 sensitivity screening for SmolVLA.

Synthetic observations only. Each trial reloads original block weights after
quantization. Identical noise is used within every baseline/quantized pair.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import yaml
from torch import nn
from transformers import AutoProcessor
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.envs.configs import LiberoEnv
from assure.quant import QuantSpec, apply_fake_quant

def relative_l2(a,b):
    return float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 7, 15, 23, 31])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 202])
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks):
        raise ValueError("Duplicate block IDs")
    if any(i < 0 or i >= 32 for i in args.blocks):
        raise ValueError("Block indices must be between 0 and 31")
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]["smolvla"]
    pcfg = PreTrainedConfig.from_pretrained(cfg["checkpoint"])
    pcfg.pretrained_path = cfg["checkpoint"]
    pcfg.device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    policy = make_policy(cfg=pcfg, env_cfg=LiberoEnv(task="libero_spatial"))
    policy.eval()
    device = next(policy.parameters()).device
    tok = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct").tokenizer(
        "Pick up the red cube and place it on the plate.", return_tensors="pt")
    batch = {
        "observation.images.image": torch.full((1,3,256,256), .5, device=device),
        "observation.images.image2": torch.full((1,3,256,256), .5, device=device),
        "observation.state": torch.zeros((1,8), device=device),
        "observation.language.tokens": tok["input_ids"].to(device),
        "observation.language.attention_mask": tok["attention_mask"].to(device=device, dtype=torch.bool),
    }
    expert_path = "model.vlm_with_expert.lm_expert"
    modules = dict(policy.named_modules())
    expert = modules[expert_path]
    if len(expert.layers) != 32:
        raise RuntimeError(f"Expected 32 expert blocks, found {len(expert.layers)}")
    features = []
    handle = modules[cfg["tap"]["expert_last"]].register_forward_hook(
        lambda _m, _i, out: features.append(out.detach().float().cpu().clone()))
    noises = {}
    for seed in args.seeds:
        g = torch.Generator(device=device).manual_seed(seed)
        noises[seed] = torch.randn((1,pcfg.chunk_size,pcfg.max_action_dim),generator=g,device=device)
    def run(seed):
        features.clear()
        policy.reset()
        with torch.inference_mode():
            action = policy.predict_action_chunk(
                {k:v.clone() for k,v in batch.items()}, noise=noises[seed].clone()
            ).detach().float().cpu()
        if len(features) != pcfg.num_steps:
            raise RuntimeError(f"Expected {pcfg.num_steps} expert features, got {len(features)}")
        if not torch.isfinite(action).all() or not all(torch.isfinite(f).all() for f in features):
            raise RuntimeError("Nonfinite model output")
        return action, [f.clone() for f in features]
    try:
        baselines = {seed:run(seed) for seed in args.seeds}
        results = []
        for index in args.blocks:
            prefix = f"{expert_path}.layers.{index}"
            block = expert.layers[index]
            linears = [(name,mod) for name,mod in block.named_modules() if isinstance(mod,nn.Linear)]
            originals = [(mod,mod.weight.detach().clone()) for _,mod in linears]
            try:
                report = apply_fake_quant(policy, QuantSpec(bits=4,group_size=128,include=[prefix+".*"]))
                if report["linear_quantized"] != len(linears):
                    raise RuntimeError(f"Block {index}: expected {len(linears)} quantized layers, got {report['linear_quantized']}")
                pairs = []
                for seed in args.seeds:
                    base_action, base_features = baselines[seed]
                    q_action, q_features = run(seed)
                    pairs.append({
                        "seed":seed,
                        "action_relative_l2":relative_l2(base_action,q_action),
                        "final_expert_relative_l2":relative_l2(base_features[-1],q_features[-1]),
                    })
                row = {
                    "block":index,
                    "linear_weight_parameters":sum(w.numel() for _,w in originals),
                    "quantized_linear_layers":len(linears),
                    "mean_action_relative_l2":sum(p["action_relative_l2"] for p in pairs)/len(pairs),
                    "mean_final_expert_relative_l2":sum(p["final_expert_relative_l2"] for p in pairs)/len(pairs),
                    "per_seed":pairs,
                }
                results.append(row)
                print(f"block={index} action={row['mean_action_relative_l2']:.4f} final_expert={row['mean_final_expert_relative_l2']:.4f}",flush=True)
            finally:
                with torch.no_grad():
                    for mod, weight in originals:
                        mod.weight.copy_(weight)
        result = {
            "checkpoint":cfg["checkpoint"],
            "input":"single synthetic observation; matched denoising noise",
            "seeds":args.seeds,
            "method":"one expert block fake-INT4 at a time, restoring original weights between blocks",
            "results":results,
            "limitations":"Pilot screening only; no LIBERO tasks, monitor calibration, or real INT4 efficiency."
        }
        out = ROOT/"results/pilot/smolvla_block_screen.json"
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+"\n")
        print("PASS: block screening completed",flush=True)
    finally:
        handle.remove()

if __name__ == "__main__":
    main()
