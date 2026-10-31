"""Independent check of the calibration sample sizes used in the paper.

  python scripts/verify_budgets.py            # table + exact tail values + Monte-Carlo check

For each (guarantee, alpha) it prints the split (shaping / threshold episodes), the rank r of the
order statistic used as q, the exact false-alarm bound, and a Monte-Carlo estimate obtained by
fitting the real FunctionalCP to i.i.d. synthetic success trajectories. The Monte-Carlo check uses
the same code path as the experiments, so it also verifies the implementation, not just the algebra.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assure.conformal import FunctionalCP, conformal_rank, pac_violation_prob, required_successes  # noqa: E402


def traj(rng, K=25):
    return np.cumsum(np.abs(rng.normal(0, 1, K))) / np.arange(1, K + 1)


def main(reps: int = 4000, n_test: int = 400, delta: float = 0.1):
    rng = np.random.default_rng(0)
    print(f"{'guarantee':9s} {'alpha':>5s} {'N':>4s} {'shape':>5s} {'thresh':>6s} {'r':>3s} "
          f"{'exact bound':>22s} {'Monte-Carlo':>22s}")
    for g in ("marginal", "pac"):
        for a in (0.20, 0.10, 0.05):
            req = required_successes(a, guarantee=g, delta=delta)
            n, r = req["n_conf"], req["rank"]
            assert conformal_rank(n, a, g, delta) == r and conformal_rank(n - 1, a, g, delta) is None
            fprs = []
            for k in range(reps if g == "marginal" else reps // 4):
                cal = [traj(rng) for _ in range(req["total"])]
                cp = FunctionalCP(a, guarantee=g, delta=delta, seed=k).fit(cal)
                assert cp.n_cal == n and not cp.underpowered
                fprs.append(np.mean([cp.first_alarm(traj(rng)) is not None for _ in range(n_test // 4)]))
            fprs = np.array(fprs)
            if g == "marginal":
                exact = f"E[FPR] <= {1 - r / (n + 1):.4f}"
                mc = f"E[FPR] = {fprs.mean():.4f}"
            else:
                exact = f"P(FPR>a) = {pac_violation_prob(n, r, a):.4f}"
                mc = f"P(FPR>a) ~ {np.mean(fprs > a):.4f}"
            print(f"{g:9s} {a:5.2f} {req['total']:4d} {req['n_shape']:5d} {n:6d} {r:3d} {exact:>22s} {mc:>22s}")
    print("\nNote: the Monte-Carlo P(FPR>a) uses a finite test sample per draw, so it is slightly noisy.")
    from assure.conformal import finite_population_fpr, split_sizes
    print("\nFinite-population randomisation model (exact; the evaluated TEST SUBSET of m successes):")
    for N, r_ in ((19, None), (44, None)):
        _, n = split_sizes(N)
        r = n if N == 44 else 9
        vals = [(m, finite_population_fpr(n, m, r, 0.10)) for m in range(40, 176)]
        e = max(v["expected_test_fpr"] for _, v in vals)
        pv = [v["p_test_fpr_above_alpha"] for _, v in vals]
        print(f"  {N} successes: max over m=40..175 of E[FPR_test] = {e:.4f}; P(FPR_test>0.10) ranges "
              f"{min(pv):.4f}..{max(pv):.4f} (lattice k/m; -> {0.0985 if N == 44 else 0.387:.4f} as m grows)")


if __name__ == "__main__":
    main()
