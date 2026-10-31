"""Stage 0 reproduction check: is our fp16 success rate consistent with the published LIBERO number?

Stage 0 verifies that the pipeline runs correctly (model loading, action semantics, feature hooks, task
execution); this script adds an UNCERTAINTY-AWARE COMPATIBILITY assessment against the published value
p_pub, not a strict pass bar. Requiring the lower confidence bound to clear p_pub - 5 points would fail
correct implementations, so the gate only stops on a clear shortfall:

  COMPATIBLE    p_pub - 0.05 lies at or below our one-sided 95% UPPER bound (no evidence of a >5-point
                shortfall); report the estimate and interval and proceed
  INCOMPATIBLE  our one-sided 95% upper bound < p_pub - 0.05 (clearly short) -> investigate before Stage A
The lower bound and the observed difference are reported alongside, for the paper.

Bounds: Clopper-Pearson (exact) and a task-clustered percentile bootstrap; the decision uses the more
conservative of the two. Sample size: the standard LIBERO protocol, 50 trials per task (500 per suite),
makes a correct implementation essentially never INCOMPATIBLE and flags most implementations that are
>= 10 points short. These 500 episodes are the fp16 nominal reference cell that Stage B needs anyway.

  python scripts/stage0_check.py --model pi05 --key libero_10 --published 0.92
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assure.common import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True, help="rollout folder, e.g. libero_10 (nominal)")
    ap.add_argument("--published", type=float, required=True)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--rollouts", default=str(ROOT / "rollouts"))
    a = ap.parse_args()
    d = Path(a.rollouts) / a.model / "fp16" / a.key
    files = sorted(d.glob("*.npz"))
    if not files:
        sys.exit(f"no rollouts in {d}")
    succ = np.array([bool(np.load(f)["success"]) for f in files])
    task = np.array([int(f.name[1:3]) for f in files])
    n, x = len(succ), int(succ.sum())
    lo_cp = 0.0 if x == 0 else float(beta.ppf(0.05, x, n - x + 1))
    hi_cp = 1.0 if x == n else float(beta.ppf(0.95, x + 1, n - x))
    rng = np.random.default_rng(0)
    uniq = np.unique(task)
    boots = []
    for _ in range(4000):
        ts = rng.choice(uniq, len(uniq))
        idx = np.concatenate([rng.choice(np.where(task == t)[0], (task == t).sum()) for t in ts])
        boots.append(succ[idx].mean())
    lo_bs, hi_bs = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))
    lo, hi = min(lo_cp, lo_bs), max(hi_cp, hi_bs)
    thr = a.published - a.margin
    verdict = "INCOMPATIBLE" if hi < thr else "COMPATIBLE"
    print(f"{a.model} {a.key}: {x}/{n} = {x / n:.3f} over {len(uniq)} tasks; published {a.published:.3f}")
    print(f"  one-sided 95% bounds  Clopper-Pearson [{lo_cp:.3f}, {hi_cp:.3f}]  task bootstrap [{lo_bs:.3f}, {hi_bs:.3f}]")
    print(f"  threshold p_pub - {a.margin} = {thr:.3f}  ->  {verdict}")
    if n < 400:
        print(f"  WARNING: {n} episodes; the check is sized for 500 (50 per task)")


if __name__ == "__main__":
    main()
