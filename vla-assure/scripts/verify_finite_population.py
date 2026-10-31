"""Independent Monte-Carlo check of the finite-population conformal statements (conformal.finite_population_fpr).

This simulation does NOT use the combinatorial formula. It re-creates the actual design from scratch:
  - 500 slots (R = 50 x 10 tasks) with FIXED potential outcomes: each slot has a success indicator and a
    continuous score for the changed policy (no interference, condition C2), drawn once;
  - each replicate draws a fresh uniformly random partition into train / calib / repair / test with the
    pre-registered fixed sizes (condition C1, same procedure as common.trial_roles);
  - recalibration: a seeded random subset of `budget` repair-pool successes, split at random into shaping
    and threshold halves (conformal.split_sizes); threshold = the r-th smallest threshold score;
  - FPR_test = fraction of ALL test-slot successes whose score exceeds the threshold.
It reports E[FPR_test] and P(FPR_test > alpha) over partitions, next to the exact formula averaged over the
realised test-success counts, for the 19 (marginal) and 44 (PAC under exchangeability) budgets. Repeated for several fixed outcome populations (success rates) because the claim is conditional
on the population.

  python scripts/verify_finite_population.py --reps 200000
Writes results/power/finite_population_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assure.common import load_configs  # noqa: E402
from assure.conformal import conformal_rank, finite_population_fpr, split_sizes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--success", type=float, nargs="*", default=[0.6, 0.75, 0.9])
    a = ap.parse_args()
    _, exp = load_configs()
    split = exp["split"]
    N = 500
    sizes = {k: int(round(v * N)) for k, v in split.items()}
    rows = []
    for p_s in a.success:
        pop = np.random.default_rng(int(1000 * p_s))
        succ = pop.random(N) < p_s                       # fixed potential outcomes of the changed policy
        score = pop.standard_normal(N)                   # fixed continuous scores (ties have probability 0)
        for budget, g in ((19, "marginal"), (44, "pac")):
            n_shape, n_thr = split_sizes(budget, exp["calibration"]["shape_frac"], exp["calibration"].get("min_shape", 10))
            r = conformal_rank(n_thr, a.alpha, g, 0.1)
            rng = np.random.default_rng(budget + int(100 * p_s))
            fprs, ms, skipped = [], [], 0
            for _ in range(a.reps):
                perm = rng.permutation(N)
                cut = np.cumsum([sizes["train"], sizes["calib"], sizes["repair"]])
                repair, test = perm[cut[1]:cut[2]], perm[cut[2]:]
                pool = repair[succ[repair]]
                if len(pool) < budget:
                    skipped += 1
                    continue
                sub = rng.choice(pool, budget, replace=False)          # seeded random recalibration subset
                thr_idx = rng.permutation(sub)[n_shape:]                # random shaping / threshold split
                q = np.sort(score[thr_idx])[r - 1]
                ts = test[succ[test]]
                fprs.append(float((score[ts] > q).mean()))
                ms.append(len(ts))
            fprs, ms = np.array(fprs), np.array(ms)
            viol_mc = float((fprs > a.alpha + 1e-12).mean())
            se = float(np.sqrt(viol_mc * (1 - viol_mc) / len(fprs)))
            mvals, mcnt = np.unique(ms, return_counts=True)
            exact = {m: finite_population_fpr(n_thr, int(m), r, a.alpha) for m in mvals}
            w = mcnt / mcnt.sum()
            viol_ex = float(sum(wi * exact[m]["p_test_fpr_above_alpha"] for m, wi in zip(mvals, w)))
            e_ex = float(sum(wi * exact[m]["expected_test_fpr"] for m, wi in zip(mvals, w)))
            row = dict(success_rate=p_s, budget=budget, guarantee=g, n_threshold=n_thr, rank=r, reps_used=len(fprs),
                       reps_skipped_budget_unmet=skipped, mean_test_successes=float(ms.mean()),
                       E_fpr_mc=float(fprs.mean()), E_fpr_exact=e_ex,
                       P_viol_mc=viol_mc, P_viol_mc_se=se, P_viol_exact=viol_ex,
                       agree=bool(abs(viol_mc - viol_ex) < 4 * se + 1e-4))
            rows.append(row)
            print(f"success {p_s:.2f} budget {budget} ({g}, n_thr {n_thr}, r {r}): "
                  f"E[FPR_test] MC {row['E_fpr_mc']:.4f} exact {e_ex:.4f} | "
                  f"P(FPR_test>{a.alpha}) MC {viol_mc:.4f}+/-{1.96 * se:.4f} exact {viol_ex:.4f}  agree={row['agree']}",
                  flush=True)
    out = ROOT / "results/power/finite_population_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
