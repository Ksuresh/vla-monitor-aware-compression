"""Summaries of the CPU re-scoring run (experiments/smolvla/cpu_all/), rebuilt from the JSON files in this repo.

1. Reproducibility: the CPU run re-scores the same saved SmolVLA episodes as the four GPU stage-2 runs. Mahalanobis is
   deterministic and must match exactly; RND and the SAFE-style probe are trained networks whose weights depend on
   hardware and library versions, so their fixed-split rates can move.
2. Recalibration on all pool successes (recal_all) against the 19 / 44-success minimum (recal), Mahalanobis,
   separate repair pool, judged on the 140 test slots.

Run from the repository root:  python experiments/cpu_all_summary.py
"""
import glob
import json
import pathlib
import re

import numpy as np
from scipy.stats import binom

ROOT = pathlib.Path(__file__).resolve().parent / "smolvla"
PLACES = ("vlm_only", "both_layers", "expert_only")
CHANGES = ("w4", "w4_lm", "w8", "cfg_denoise5", "cfg_steps5")

cpu = {p: json.load(open(ROOT / "cpu_all" / f"resplit_{p}.json")) for p in PLACES}
gpu = {}
for f in glob.glob(str(ROOT / "stage2_part*" / "resplit_part*_*.json")):
    p = re.search(r"resplit_part\d_(.+)\.json", f).group(1)
    for v, vv in json.load(open(f))["variants"].items():
        gpu[(p, v)] = vv

print("1. CPU re-run vs GPU stage-2 runs (changed-policy false-alarm rate on all successes, fixed split)")
for mon in ("mahalanobis", "rnd", "safe_mlp"):
    diffs, med = [], []
    for p in PLACES:
        for v in CHANGES:
            a = gpu[(p, v)]["monitors"][mon]
            b = cpu[p]["variants"][v]["monitors"][mon]
            diffs.append(abs(a["preregistered"]["fpr_var_all"] - b["preregistered"]["fpr_var_all"]))
            med.append(abs(a["over_splits"]["fpr_var_all"][0] - b["over_splits"]["fpr_var_all"][0]))
    print(f"   {mon:12s} max |difference| fixed split {max(diffs):.3f}, re-split median {max(med):.3f} "
          f"({sum(d == 0 for d in diffs)} of {len(diffs)} cells identical)")

print("\n2. Recalibration on all pool successes vs the minimum sample (Mahalanobis, separate pool, 12 feasible cells)")
for mode in ("separate_pool_marginal", "separate_pool_pac"):
    rows = []
    for v in ("w4", "w8", "cfg_denoise5", "cfg_steps5"):          # w4_lm: 24 pool successes, PAC infeasible
        for p in PLACES:
            o = cpu[p]["variants"][v]["monitors"]["mahalanobis"]["fixes_crossfit"][mode]["options"]
            rows.append([o["inherited"]["fpr"][0], o["recal"]["fpr"][0], o["recal_all"]["fpr"][0],
                         o["inherited"]["tpr"][0], o["recal"]["tpr"][0], o["recal_all"]["tpr"][0],
                         o["recal"]["p_violate"], o["recal_all"]["p_violate"]])
    r = np.array(rows)
    valid = r[:, 0] <= 0.15
    print(f"   {mode}: held-out FPR {r[:, 1].mean():.3f} -> {r[:, 2].mean():.3f}; "
          f"share of orderings above 0.15 {r[:, 6].mean():.2f} -> {r[:, 7].mean():.2f}; "
          f"detection lost vs inherited on the {valid.sum()} valid cells {(r[valid, 3] - r[valid, 4]).mean():.3f} "
          f"(max {(r[valid, 3] - r[valid, 4]).max():.2f}) -> {(r[valid, 3] - r[valid, 5]).mean():.3f} "
          f"(max {(r[valid, 3] - r[valid, 5]).max():.2f})")

# Why PAC mode gains nothing here: the pools hold 46-53 successes (23-27 threshold episodes), and the PAC rank only
# drops below the maximum once P(Bin(n, 0.9) >= n - 1) <= 0.10.
n = next(n for n in range(5, 400) if binom.sf(n - 2, n, 0.9) <= 0.10)
print(f"\n   PAC rank leaves the maximum at n = {n} threshold episodes, i.e. N = {2 * n} successes")
