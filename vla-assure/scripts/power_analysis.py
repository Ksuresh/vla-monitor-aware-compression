"""Power / precision analysis for the primary endpoint (paired false-alarm difference, W4 vs fp16).

Simulates the exact analysis pipeline (task-clustered paired bootstrap, one-sided tests, Holm) on
synthetic paired alarm indicators with realistic structure:
  - 10 tasks per suite; per-task baseline false-alarm rate varies (task clustering, ICC)
  - variant alarms are correlated with reference alarms on the same trial (paired design): a
    reference alarm disappears with probability `flip` and new alarms appear at the rate that keeps
    the marginal false-alarm rate unchanged; `delta` adds extra variant false alarms (true D = delta)
  - only trials BOTH policies solved enter the endpoint: n_pairs = n_test * pi11, where pi11 is the
    JOINT success rate. Paired trials share initial state and seed, so successes are positively
    correlated and pi11 = p_r p_v + rho sqrt(p_r(1-p_r) p_v(1-p_v)), bounded by
    max(0, p_r + p_v - 1) <= pi11 <= min(p_r, p_v). Independence (rho = 0) is a planning scenario, not a
    bound: rho can be negative if compression changes which situations are hard;
    --joint sets pi11 directly, and --from_eval estimates it (and p_r, p_v) from Stage A.

Reports, per design point:
  power_ni     P(non-inferiority shown | true delta = 0)            -- can a robustness claim succeed?
  power_deg    P(degradation shown    | true delta = delta_alt)     -- can a real problem be found?
  half_width   median 95% CI half-width of the paired difference      -- precision
both at the per-comparison level and after Holm across `m` comparisons (worst-case step: alpha/m).

  python scripts/power_analysis.py                      # default grid, ~2-4 min on CPU
  python scripts/power_analysis.py --reps 400 --B 800   # more precise
  python scripts/power_analysis.py --from_pairs results/eval_pairs/pi05__libero_10+shift_mild.csv \
      --from_eval results/eval/pi05__libero_10+shift_mild.csv
      # after Stage A: replace the assumed false-alarm rate and discordance with the observed ones
Writes results/power/power.csv.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assure.common import ROOT  # noqa: E402
from assure.metrics import confirmatory  # noqa: E402


def simulate(rng, n_pairs, base=0.09, icc_sd=0.04, flip=0.3, delta=0.0, n_tasks=20):
    """The SAME exact generator as scripts/calibrate_level.py (true D = delta per task, so the null
    boundary is hit exactly). Returns (ref, var, tasks)."""
    from calibrate_level import sim
    d, g = sim(rng, n_tasks, n_pairs, delta, base, flip, icc_sd)
    return np.zeros_like(d), d, g   # only the difference var - ref enters the test


def joint_success(p_ref, p_var, rho=0.0):
    """P(both succeed) for paired trials with success correlation rho, clipped to the Frechet bounds."""
    import math
    pi = p_ref * p_var + rho * math.sqrt(p_ref * (1 - p_ref) * p_var * (1 - p_var))
    return min(max(pi, max(0.0, p_ref + p_var - 1)), min(p_ref, p_var))


def run(n_test, success, margin, m, delta_alt, reps, B, flip, base, seed=0, joint=None, n_tasks=20,
        level_deg=0.0375, level_ni=0.03):
    """Power and size of the CONFIRMATORY test (WCR bootstrap-t, Webb weights) as used in 04_report.py.
    n_tasks = clusters: 20 for the pooled Stage B endpoint (2 suites x 10 tasks), 10 for one cell."""
    rng = np.random.default_rng(seed)
    pi11 = joint if joint is not None else success * success  # independence planning scenario
    n_pairs = max(4, int(round(n_test * pi11)))
    ni1 = nih = dg1 = dgh = t1_ni = t1_deg = 0
    hw = []
    for r in range(reps):
        ref, var, t = simulate(rng, n_pairs, base=base, flip=flip, delta=0.0, n_tasks=n_tasks)
        c = confirmatory(var - ref, t, margin, B=B, seed=r)
        ni1 += c["p_noninf"] < level_ni
        nih += c["p_noninf"] < level_ni / m
        hw.append((c["delta_hi"] - c["delta_lo"]) / 2)
        ref, var, t = simulate(rng, n_pairs, base=base, flip=flip, delta=delta_alt, n_tasks=n_tasks)
        c = confirmatory(var - ref, t, margin, B=B, seed=10_000 + r)
        dg1 += c["p_worse"] < level_deg
        dgh += c["p_worse"] < level_deg / m
        # size AT the null boundaries
        ref, var, t = simulate(rng, n_pairs, base=base, flip=flip, delta=margin, n_tasks=n_tasks)   # true D = margin
        t1_ni += confirmatory(var - ref, t, margin, B=B, seed=20_000 + r)["p_noninf"] < level_ni
        ref, var, t = simulate(rng, n_pairs, base=base, flip=flip, delta=0.0, n_tasks=n_tasks)      # true D = 0
        t1_deg += confirmatory(var - ref, t, margin, B=B, seed=30_000 + r)["p_worse"] < level_deg
    mc = lambda k: 1.96 * np.sqrt(max(k / reps * (1 - k / reps), 1e-12) / reps)  # noqa: E731  95% MC half-width
    return dict(n_test=n_test, success=success, joint_success=pi11, n_pairs=n_pairs, n_clusters=n_tasks, margin=margin,
                m=m, flip=flip, base=base, delta_alt=delta_alt, level_ni=level_ni, level_deg=level_deg, reps=reps,
                power_ni=ni1 / reps, power_ni_mc=mc(ni1), power_ni_holm=nih / reps,
                power_deg=dg1 / reps, power_deg_mc=mc(dg1), power_deg_holm=dgh / reps,
                type1_noninf_at_margin=t1_ni / reps, type1_noninf_mc=mc(t1_ni),
                type1_degraded_at_zero=t1_deg / reps, type1_degraded_mc=mc(t1_deg),
                ci_half_width=float(np.median(hw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--B", type=int, default=1999)
    ap.add_argument("--success", type=float, default=0.7, help="success rate of each policy (if --joint unset)")
    ap.add_argument("--joint", type=float, default=None, help="joint success rate pi11 of paired trials")
    ap.add_argument("--rho", type=float, default=None, help="success correlation; pi11 from --success and rho")
    ap.add_argument("--from_eval", default=None, help="Stage A eval CSV: estimate p_ref, p_var, pi11 (paired rows)")
    ap.add_argument("--flip", type=float, default=0.3,
                    help="P(a reference alarm disappears under the variant); new alarms balance it")
    ap.add_argument("--base", type=float, default=0.09, help="reference false-alarm rate on successes")
    ap.add_argument("--from_pairs", default=None, help="eval_pairs CSV from Stage A to estimate base / flip")
    ap.add_argument("--alpha", type=float, default=0.10)
    a = ap.parse_args()
    if a.from_pairs:
        import pandas as pd
        g = pd.read_csv(a.from_pairs)
        g = g[(abs(g.alpha - a.alpha) < 1e-9) & ~g.monitor.str.startswith("ext_")]
        a.base = float(g.ref_alarm.mean())
        on = g[g.ref_alarm == 1]
        a.flip = float((on.var_alarm == 0).mean()) if len(on) else a.flip
        print(f"from Stage A: reference false-alarm rate {a.base:.3f}, alarm discordance {a.flip:.3f}")
    if a.from_eval:
        import pandas as pd
        e = pd.read_csv(a.from_eval)
        e = e[(e.analysis == "paired") & (e.fix == "none") & (e.variant == "w4")].drop_duplicates("monitor")
        if len(e):
            r0 = e.iloc[0]
            a.joint = float(r0.n_both_success / r0.n_pairs)
            pr, pv = float(r0.task_success_ref_same_trials), float(r0.task_success)
            rho = ((a.joint - pr * pv) / ((pr * (1 - pr) * pv * (1 - pv)) ** 0.5)) if 0 < pr < 1 and 0 < pv < 1 else float("nan")
            print(f"from Stage A: p_ref {pr:.3f}, p_var {pv:.3f}, joint success {a.joint:.3f} "
                  f"(independence would give {pr * pv:.3f}; success correlation {rho:.2f})")
    if a.joint is None and a.rho is not None:
        a.joint = joint_success(a.success, a.success, a.rho)
    rows = []
    # design points: one cell (175 test trials, 10 task clusters) and the pooled Stage B endpoint (700, 20)
    grid = itertools.product([(140, 10), (175, 10), (700, 20)], [0.05, 0.10], [4])
    for (n_test, n_tasks), margin, m in grid:
        r = run(n_test, a.success, margin, m, delta_alt=0.10, reps=a.reps, B=a.B, flip=a.flip, base=a.base,
                joint=a.joint, n_tasks=n_tasks)
        rows.append(r)
        print(f"n_test={n_test:4d} pairs={r['n_pairs']:4d} margin={margin:.2f} m={m:2d} | "
              f"NI {r['power_ni']:.2f} (Holm {r['power_ni_holm']:.2f}) | "
              f"detect +0.10 {r['power_deg']:.2f} (Holm {r['power_deg_holm']:.2f}) | CI +/-{r['ci_half_width']:.3f} | "
              f"size NI {r['type1_noninf_at_margin']:.3f}+/-{r['type1_noninf_mc']:.3f} (nominal {r['level_ni']}), "
              f"deg {r['type1_degraded_at_zero']:.3f}+/-{r['type1_degraded_mc']:.3f} (nominal {r['level_deg']})")
    import pandas as pd
    out = ROOT / "results/power"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "power.csv", index=False)
    import json
    (out / "power_params.json").write_text(json.dumps({
        "reps": a.reps, "bootstrap_B": a.B, "seed": 0, "success": a.success, "joint_success": a.joint if a.joint is not None else a.success ** 2,
        "joint_assumption": "independence planning scenario (rho = 0)" if a.joint is None else "given/estimated", "flip": a.flip, "base_fpr": a.base,
        "icc_sd": 0.04, "n_tasks_per_cell": 10, "delta_alt": 0.10, "test_level": 0.05,
        "decision_noninferior": "WCR p for H0 D >= margin below 0.03 (Bonferroni 0.03/m column)",
        "decision_degraded": "WCR p for H0 D <= 0 below 0.0375",
        "test": "WCR bootstrap-t, Webb weights; levels degradation 0.0375, non-inferiority 0.03 (simulation-calibrated)",
        "grid": {"(n_test, clusters)": [[140, 10], [175, 10], [700, 20]], "margin": [0.05, 0.10], "m": [4]}}, indent=2))
    print("wrote", out / "power.csv")


if __name__ == "__main__":
    main()
