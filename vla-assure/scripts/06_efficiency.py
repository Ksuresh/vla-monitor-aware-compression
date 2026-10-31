"""End-to-end efficiency of REAL quantized policies (no fake quantization, no shadows, no taps).

For each variant: loads the policy with real kernels (torchao int8/int4 for LeRobot policies,
bitsandbytes INT8/NF4 for OpenVLA), detaches all monitoring hooks, and measures
  weight_mem_gib       GPU memory allocated right after loading (weights + buffers)
  peak_mem_gib         peak GPU memory during the timed episodes
  query_ms_median/p95  observation batch -> action chunk (model forward incl. sampling), CUDA-synced,
                       after warm-up queries
  episode_s_median     wall-clock per LIBERO episode, including simulation and pre/post-processing
  control_hz           executed environment steps per second over the timed episodes
  task_success         success over the timed episodes (sanity check only; not the evaluation)
Fake-quant variants (w8, w4, w4p-*) are rejected: they do not change memory or speed.

  python scripts/06_efficiency.py --model pi05 --suite libero_10 --variants fp16,tao8,tao4
  python scripts/06_efficiency.py --model pi05 --variants tao4,tao4p-margin_drift-f10,tao4p-action_drift-f10
      # the proposed mixed-precision configurations with real kernels (same protected sets as w4p-*)
  python scripts/06_efficiency.py --model openvla --suite libero_spatial --variants fp16,bnb8,bnb4 --repo_root ~/openvla
Writes results/efficiency/<model>.json (read by 04_report.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assure.adapters import make_session  # noqa: E402
from assure.common import ROOT, load_configs, setup_logging, write_json  # noqa: E402
from assure.variants import resolve  # noqa: E402

REAL_KINDS = ("reference", "torchao", "bnb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--variants", default="fp16,tao8,tao4")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--warmup_queries", type=int, default=10)
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    setup_logging()
    models, exp = load_configs()
    results = {}
    for vname in a.variants.split(","):
        var = resolve(vname, exp)
        if var.get("kind", "reference") not in REAL_KINDS:
            sys.exit(f"{vname}: kind {var.get('kind')} is fake quantization; use tao8/tao4/bnb8/bnb4")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        s = make_session(a.model, models[a.model], vname, var, a.suite, device=a.device,
                         repo_root=a.repo_root, base_seed=exp["seed"])
        torch.cuda.synchronize()
        weight_mem = (torch.cuda.memory_allocated() - base) / 2**30
        s.tap.detach()  # no monitoring overhead in efficiency numbers
        # warm-up (kernel autotuning, cudnn) on a trial id that is never evaluated
        s.rollout_episode(task_id=0, episode_idx=800_000, seed=10**7)
        s.query_seconds.clear()
        torch.cuda.reset_peak_memory_stats()
        ep_s, steps, succ = [], 0, []
        for e in range(a.episodes):
            t0 = time.perf_counter()
            ep = s.rollout_episode(task_id=e % 10, episode_idx=800_001 + e, seed=10**7 + e + 1)
            torch.cuda.synchronize()
            ep_s.append(time.perf_counter() - t0)
            steps += ep.length
            succ.append(ep.success)
        q = np.array(s.query_seconds[a.warmup_queries:] or s.query_seconds) * 1000
        results[vname] = {
            "kind": var.get("kind", "reference"), "weight_mem_gib": round(weight_mem, 3),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "query_ms_median": float(np.median(q)), "query_ms_p95": float(np.percentile(q, 95)),
            "episode_s_median": float(np.median(ep_s)), "control_hz": steps / sum(ep_s),
            "task_success": float(np.mean(succ)), "episodes": a.episodes,
            "gpu": torch.cuda.get_device_name(0),
        }
        print(vname, results[vname])
        s.close()
        del s
    write_json(ROOT / "results/efficiency" / f"{a.model}.json", {"model": a.model, "suite": a.suite, "variants": results})


if __name__ == "__main__":
    main()
