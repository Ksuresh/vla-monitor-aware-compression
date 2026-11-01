"""Matched-budget selective-precision exploratory test on one synthetic observation.

Compare all-expert fake INT4 against protecting sensitive vs control blocks.
Protected block sets were selected on this SAME observation: exploratory only.
"""
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

SENSITIVE = [24, 18, 30, 22, 0]
# Lower-drift blocks among those already measured, chosen as a preliminary control.
# Not a random sample or a parameter-exact matched set.
CONTROL = [7, 15, 23, 31, 20]
SEEDS = [101, 123, 202, 303, 404]
EXPERT = "model.vlm_with_expert.lm_expert"

def relative_l2(a, b):
    return float(torch.linalg.vector_norm(a-b) /
                 torch.linalg.vector_norm(a).clamp_min(1e-8))

def main():
    cfg = yaml.safe_load((ROOT/"configs/models.yaml").read_text())["models"]["smolvla"]
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
        "observation.language.attention_mask": tok["attention_mask"].to(device=device,dtype=torch.bool),
    }
    modules = dict(policy.named_modules())
    expert = modules[EXPERT]
    if len(expert.layers) != 32:
        raise RuntimeError("Unexpected action expert block count")
    # Keep a CPU snapshot of expert Linear weights to restore exact baseline between conditions.
    linears = [(name, mod) for name, mod in expert.named_modules() if isinstance(mod,nn.Linear)]
    originals = [(mod, mod.weight.detach().cpu().clone()) for _,mod in linears]
    features = []
    handle = modules[cfg["tap"]["expert_last"]].register_forward_hook(
        lambda _m, _i, output: features.append(output.detach().float().cpu().clone()))
    noises = {}
    for seed in SEEDS:
        g = torch.Generator(device=device).manual_seed(seed)
        noises[seed] = torch.randn((1,pcfg.chunk_size,pcfg.max_action_dim),
                                  generator=g,device=device)
    def run(seed):
        features.clear()
        policy.reset()
        with torch.inference_mode():
            action = policy.predict_action_chunk(
                {k:v.clone() for k,v in batch.items()},noise=noises[seed].clone()
            ).detach().float().cpu()
        if len(features) != pcfg.num_steps:
            raise RuntimeError(f"Expected {pcfg.num_steps} feature hooks, got {len(features)}")
        if not torch.isfinite(action).all() or not all(torch.isfinite(f).all() for f in features):
            raise RuntimeError("Nonfinite model output")
        return action, features[-1].clone()
    def restore():
        with torch.no_grad():
            for mod, weight in originals:
                mod.weight.copy_(weight.to(mod.weight.device))
    try:
        baseline = {seed:run(seed) for seed in SEEDS}
        print("Five baseline passes complete",flush=True)
        configs = {
            "full_fake_int4": [],
            "protect_sensitive": SENSITIVE,
            "protect_control": CONTROL,
        }
        results = {}
        for name, protected in configs.items():
            restore()
            spec = QuantSpec(bits=4,group_size=128,include=[EXPERT+".*"],
                             protect=[f"{EXPERT}.layers.{i}" for i in protected])
            report = apply_fake_quant(policy,spec)
            expected = 224-7*len(protected)
            if report["linear_quantized"] != expected:
                raise RuntimeError(f"{name}: expected {expected} layers, got {report['linear_quantized']}")
            rows = []
            for seed in SEEDS:
                a0,f0 = baseline[seed]
                a1,f1 = run(seed)
                rows.append({"seed":seed,
                             "action_relative_l2":relative_l2(a0,a1),
                             "final_expert_relative_l2":relative_l2(f0,f1)})
            result = {
                "protected_blocks":protected,
                "quantized_linear_layers":report["linear_quantized"],
                "quantized_linear_weight_parameters":sum(
                    mod.weight.numel() for n,mod in linears
                    if not any(n == f"layers.{i}" or n.startswith(f"layers.{i}.") for i in protected)
                ),
                "mean_action_relative_l2":sum(x["action_relative_l2"] for x in rows)/len(rows),
                "mean_final_expert_relative_l2":sum(x["final_expert_relative_l2"] for x in rows)/len(rows),
                "per_seed":rows,
            }
            results[name] = result
            print(f"{name}: quantized_layers={result['quantized_linear_layers']} "
                  f"action={result['mean_action_relative_l2']:.4f} "
                  f"final_expert={result['mean_final_expert_relative_l2']:.4f}",flush=True)
        result = {
            "checkpoint":cfg["checkpoint"],
            "input":"single synthetic constant-image, zero-state observation",
            "seeds":SEEDS,
            "method":"fake INT4 symmetric RTN, original expert weights restored between configurations",
            "conditions":results,
            "limitations":"Protected blocks selected using same synthetic observation (selection bias). Control not parameter-exact matched. No real LIBERO observations, failure monitor, calibration, task success, or INT4 efficiency."
        }
        out = ROOT/"results/pilot/smolvla_selective_precision.json"
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+"\n")
        print("PASS: selective-precision exploratory comparison completed",flush=True)
    finally:
        restore()
        handle.remove()

if __name__ == "__main__":
    main()
