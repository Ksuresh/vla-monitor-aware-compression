"""Multi-seed paired-noise sensitivity on one synthetic observation; no robot rollouts."""
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
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]["smolvla"]
    pcfg = PreTrainedConfig.from_pretrained(cfg["checkpoint"])
    pcfg.pretrained_path = cfg["checkpoint"]
    pcfg.device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
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
    features = []
    path = cfg["tap"]["expert_last"]
    handle = dict(policy.named_modules())[path].register_forward_hook(
        lambda _m, _i, out: features.append(out.detach().float().cpu().clone()))
    seeds = [101, 123, 202, 303, 404]
    noises = {}
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        noises[seed] = torch.randn((1, pcfg.chunk_size, pcfg.max_action_dim),
                                   generator=generator, device=device)
    def run(seed):
        features.clear()
        policy.reset()
        with torch.inference_mode():
            action = policy.predict_action_chunk(
                {k: v.clone() for k,v in batch.items()}, noise=noises[seed].clone()
            ).detach().float().cpu()
        if len(features) != pcfg.num_steps:
            raise RuntimeError(f"Unexpected expert hook count {len(features)}")
        if not torch.isfinite(action).all() or not all(torch.isfinite(f).all() for f in features):
            raise RuntimeError("Non-finite output")
        return action, [f.clone() for f in features]
    try:
        baselines = {}
        for seed in seeds:
            baselines[seed] = run(seed)
            print("Baseline completed:", seed, flush=True)
        report = apply_fake_quant(policy, QuantSpec(bits=4, group_size=128,
                                                    include=["model.vlm_with_expert.lm_expert.*"]))
        assert report["linear_quantized"] > 0
        rows = []
        for seed in seeds:
            base_action, base_feats = baselines.pop(seed)
            q_action, q_feats = run(seed)
            def relative_l2(a,b):
                return float(torch.linalg.vector_norm(a-b) /
                             torch.linalg.vector_norm(a).clamp_min(1e-8))
            row = {
                "seed": seed,
                "action_relative_l2": relative_l2(base_action, q_action),
                "expert_per_step_relative_l2": [
                    relative_l2(a,b) for a,b in zip(base_feats,q_feats)
                ]
            }
            rows.append(row)
            print(f"seed={seed} action_drift={row['action_relative_l2']:.4f} "
                  f"final_expert_drift={row['expert_per_step_relative_l2'][-1]:.4f}", flush=True)
        actions = torch.tensor([r["action_relative_l2"] for r in rows])
        final = torch.tensor([r["expert_per_step_relative_l2"][-1] for r in rows])
        result = {
            "checkpoint": cfg["checkpoint"],
            "baseline": "model-native mixed/bfloat16",
            "quantization": "action expert fake INT4 RTN, no real 4-bit kernels",
            "input": "single synthetic constant-image, zero-state, fixed-language observation",
            "seeds": seeds,
            "quantization_report": report,
            "per_seed": rows,
            "summary": {
                "action_drift_mean": float(actions.mean()),
                "action_drift_std_sample": float(actions.std(unbiased=True)),
                "final_expert_drift_mean": float(final.mean()),
                "final_expert_drift_std_sample": float(final.std(unbiased=True))
            },
            "limitations": "Multiple noise seeds on ONE synthetic input; no task success, failure labels, monitor calibration, or deployed INT4 efficiency."
        }
        out = ROOT / "results/pilot/smolvla_expert_int4_multiseed.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2))
        print("PASS: multi-seed paired-noise pilot completed")
    finally:
        handle.remove()

if __name__ == "__main__":
    main()
