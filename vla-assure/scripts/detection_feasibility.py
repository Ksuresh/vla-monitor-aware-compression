"""Feasibility of the DETECTION component of the confirmatory method families F2-F4.

In F2-F4 the proposed rule must (i) lower false alarms on joint successes, (ii) be non-inferior in
detection on JOINT FAILURES (both rules' policies fail the same trial; margin 0.10), and (iii) be
non-inferior in task success. Component (ii) rests on joint failures only, which can be few. This script
answers, before Stage C deployment is sized:

  1. how many joint failures to expect:  n_jf = n_trials * pi00,
     pi00 = 1 - p_a - p_b + pi11 (pi11 from success correlation rho, clipped to the Frechet bounds);
     P(n_jf >= N) under that rate.
  2. what n_jf the detection non-inferiority test needs: power at true D_TPR = 0 and size at D = margin,
     for the SAME wild cluster restricted bootstrap-t used in 04_report.py, with failures concentrated
     in few tasks (Dirichlet task weights), task-varying TPR, and alarm discordance between the two
     policies (`disc` = P(their alarm decisions differ on a joint failure); equal marginal TPR under H0).

Pre-registered rule (configs/experiment.yaml: families.detection): N_min(d) = smallest n_jf with power >= 0.80
at alarm discordance d, with the upper 95% Monte-Carlo bound of size <= 0.05 (tabulated). Before any deployed
Stage C configuration executes, K extension trials per task are fixed as the smallest K whose PREDICTIVE
probability of n_jf >= N_min(d) is >= 0.90, integrating jointly over the posterior of the paired joint-outcome
table (pilot) and of the discordance (shadow pass on reference failures); see plan(). Extension trials are a
separate pre-registered population (fresh seeds, LIBERO initial states redrawn uniformly with replacement);
they enter only the detection component. If the realised n_jf is below N_min, or fewer than
`min_clusters` tasks contain a joint failure, the family's verdict is "detection not confirmable" and the
method claim is withheld, whatever the false-alarm improvement.

  python scripts/detection_feasibility.py                    # planning grid (~5-10 min CPU)
  python scripts/detection_feasibility.py --plan --joint_counts 95 10 12 23 --disc_counts 3 40   # Stage C sizing
Writes results/power/detection_feasibility.csv and .json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assure.metrics import wild_cluster_test  # noqa: E402


def joint_failure_rate(f_a, f_b, rho):
    """P(both fail) for failure rates f_a, f_b with correlation rho (Frechet-clipped)."""
    pi = f_a * f_b + rho * math.sqrt(f_a * (1 - f_a) * f_b * (1 - f_b))
    return min(max(pi, max(0.0, f_a + f_b - 1)), min(f_a, f_b))


def sim(rng, n, G, disc, shift, sd, conc):
    """Per-joint-failure d = alarm_base - alarm_prop in {-1, 0, 1} (D > 0 = proposed detects less).
    disc = P(the two policies' alarm decisions differ) on a joint failure, varying by task (sd);
    P(d = +1) = (disc + shift) / 2, P(d = -1) = (disc - shift) / 2, so the true mean is exactly `shift`.
    Failures are spread over tasks with Dirichlet(conc) weights (LIBERO failures concentrate)."""
    w = rng.dirichlet(np.full(G, conc))
    g = rng.choice(G, n, p=w)
    q = np.clip(disc + rng.normal(0, sd, G), max(shift, 0.0) + 1e-9, 1.0)[g]
    u = rng.random(n)
    d = np.where(u < (q + shift) / 2, 1.0, np.where(u < q, -1.0, 0.0))
    return d, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jf", type=int, nargs="*", default=[30, 60, 90, 120, 160, 200, 240])
    ap.add_argument("--disc", type=float, nargs="*", default=[0.05, 0.10, 0.20, 0.30],
                    help="alarm-decision discordance between the two rules' policies on joint failures")
    ap.add_argument("--conc", type=float, default=1.0, help="Dirichlet concentration of failures over tasks")
    ap.add_argument("--sd", type=float, default=0.03)
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--level", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--n_trials", type=int, default=350)
    ap.add_argument("--p_fail", type=float, nargs=2, default=None, help="failure rates of the two rules' policies")
    ap.add_argument("--rho", type=float, nargs="*", default=[0.0, 0.3, 0.6])
    ap.add_argument("--plan", action="store_true",
                    help="Stage C sizing only (no simulation): uses the pre-registered N_min table in experiment.yaml")
    ap.add_argument("--disc_hat", type=float, default=None, help="--plan point scenario: alarm discordance")
    ap.add_argument("--joint_counts", type=int, nargs=4, default=None,
                    help="--plan: paired pilot counts n11 n10 n01 n00 (both succeed, only A fails, only B fails, both fail)")
    ap.add_argument("--disc_counts", type=int, nargs=2, default=None,
                    help="--plan: alarm-decision disagreements x among n reference failures (shadow pass)")
    ap.add_argument("--plan_draws", type=int, default=20000)
    ap.add_argument("--clusters_per_suite", type=int, default=10)
    ap.add_argument("--suites", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    out = ROOT / "results/power"
    out.mkdir(parents=True, exist_ok=True)
    if a.plan:
        return plan(a, out)

    # 1. expected joint failures
    print("# expected joint failures among", a.n_trials, "paired trials")
    fails = [tuple(a.p_fail)] if a.p_fail else [(0.15, 0.15), (0.20, 0.20), (0.25, 0.25), (0.30, 0.30)]
    exp_rows = []
    for (fa, fb), rho in itertools.product(fails, a.rho):
        pi00 = joint_failure_rate(fa, fb, rho)
        exp_rows.append(dict(f_a=fa, f_b=fb, rho=rho, pi00=pi00, expected_n_jf=a.n_trials * pi00))
        print(f"  fail {fa:.2f}/{fb:.2f} rho {rho:.1f}: pi00 {pi00:.3f} -> E[n_jf] {a.n_trials * pi00:5.1f}")

    # 2. power and size of the detection non-inferiority test
    mc = lambda k, n: 1.96 * math.sqrt(max(k / n * (1 - k / n), 1e-12) / n)  # noqa: E731
    rows = []
    for n, disc in itertools.product(a.n_jf, a.disc):
        pw = sz = 0
        geff = []
        for r in range(a.reps):
            d, g = sim(rng, n, a.clusters, disc, 0.0, a.sd, a.conc)
            geff.append(len(np.unique(g)))
            pw += wild_cluster_test(d, g, a.margin, "less", a.B, seed=r)["p"] < a.level
            d, g = sim(rng, n, a.clusters, max(disc, a.margin), a.margin, a.sd, a.conc)
            sz += wild_cluster_test(d, g, a.margin, "less", a.B, seed=50_000 + r)["p"] < a.level
        row = dict(n_jf=n, disc=disc, conc=a.conc, level=a.level, margin=a.margin, reps=a.reps,
                   power=pw / a.reps, power_mc=mc(pw, a.reps), size=sz / a.reps, size_mc=mc(sz, a.reps),
                   size_upper=sz / a.reps + mc(sz, a.reps), median_clusters_with_failures=float(np.median(geff)))
        rows.append(row)
        print(f"  n_jf {n:4d} discordance {disc:.2f}: power {row['power']:.2f}+/-{row['power_mc']:.2f} "
              f"size {row['size']:.3f}+/-{row['size_mc']:.3f}  clusters {row['median_clusters_with_failures']:.0f}")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "detection_feasibility.csv", index=False)
    size_ok = bool(df.size_upper.max() <= 0.05)
    print(f"max size {df['size'].max():.3f} (upper MC bound {df.size_upper.max():.3f}) -> size ok: {size_ok}")
    n_min_by_disc = {}
    for disc, x in df.groupby("disc"):
        hit = x[x.power >= 0.80].n_jf
        n_min_by_disc[float(disc)] = int(hit.min()) if len(hit) else None
        print(f"  N_min at discordance {disc:.2f}: {n_min_by_disc[float(disc)]}")
    n_min = n_min_by_disc.get(0.10)

    # 3. slots needed so that P(n_jf >= N_min) >= 0.90
    need = []
    if n_min:
        for r_ in exp_rows:
            n_tr = next((m for m in range(50, 5001, 10) if binom.sf(n_min - 1, m, r_["pi00"]) >= 0.90), None)
            need.append({**r_, "n_min": n_min, "paired_trials_needed": n_tr})
            print(f"  fail {r_['f_a']:.2f} rho {r_['rho']:.1f}: {n_tr} paired trials for P(n_jf >= {n_min}) >= 0.90")
    (out / "detection_feasibility.json").write_text(json.dumps(
        {"n_min_at_disc_0.10": n_min, "n_min_by_discordance": n_min_by_disc, "size_ok": size_ok, "expected": exp_rows, "trials_needed": need, "args": vars(a)}, indent=2))


def _n_min(det, d):
    """Pre-registered N_min for discordance d: the tabulated value at the smallest tabulated discordance >= d
    (conservative step-up), floored at n_min_floor; None above the table (infeasible)."""
    table = {float(k): v for k, v in det["n_min_by_discordance"].items()}
    ok = [k for k in sorted(table) if k >= d - 1e-9]
    return max(table[ok[0]], det["n_min_floor"]) if ok else None


def plan(a, out):
    """Fix the number K of extension trials per task BEFORE any deployed Stage C configuration executes.

    Pre-registered rule: K is the smallest value whose PREDICTIVE probability of meeting the detection
    minimum is >= sizing_probability, where the prediction integrates over the uncertainty of all planning
    estimates jointly (not separate one-sided bounds, which do not by themselves give that probability):
      joint outcome table (both succeed, only A fails, only B fails, both fail) ~ Dirichlet(counts + 1/2)
        from the paired pilot (--joint_counts n11 n10 n01 n00); the joint-failure rate pi00 is read
        directly, so no success-vs-failure correlation is needed (for binary indicators the correlation of
        failures equals the correlation of successes, so the two parametrisations agree);
      alarm discordance on failures d ~ Beta(x + 1/2, n - x + 1/2) from the shadow pass (--disc_counts x n);
      N_min(d) from the pre-registered table (step-up; d above the table counts as a miss);
      n_jf ~ Binomial(n_trials + G K, pi00).
    Point mode (--p_fail, --rho, --disc_hat) is kept as a planning scenario only."""
    import yaml
    det = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())["families"]["detection"]
    G = a.clusters_per_suite * a.suites
    cap, target = det["max_extension_per_task"], det["sizing_probability"]
    rng = np.random.default_rng(17)
    if a.joint_counts and a.disc_counts:
        n_draw = a.plan_draws
        pi00 = rng.dirichlet(np.asarray(a.joint_counts, float) + 0.5, n_draw)[:, 3]
        x, n = a.disc_counts
        d = rng.beta(x + 0.5, n - x + 0.5, n_draw)
        mode = "joint posterior simulation"
    elif a.p_fail is not None and a.disc_hat is not None and len(a.rho) == 1:
        pi00 = np.array([joint_failure_rate(*a.p_fail, a.rho[0])])
        d = np.array([a.disc_hat])
        mode = "point scenario (not the pre-registered rule)"
    else:
        sys.exit("--plan needs --joint_counts n11 n10 n01 n00 and --disc_counts x n "
                 "(or, as a scenario only, --p_fail F_A F_B --rho R --disc_hat D)")

    infl = det.get("discordance_inflation", 1.0)   # pre-registered allowance for proxy underestimation
    nmin = np.array([_n_min(det, min(1.0, v * infl)) or 10**9 for v in d], float)

    def p_meet(k, nm=None):
        nm = nmin if nm is None else nm
        return float(np.mean(binom.sf(nm - 1, a.n_trials + G * k, pi00)))
    k_need = next((k for k in range(cap + 1) if p_meet(k) >= target), None)
    # The shadow-pass discordance (reference failures, observation-controlled) is only a PLANNING PROXY for
    # the executed-trajectory discordance on joint failures, which is what determines actual power. Sensitivity:
    # predictive probability at the chosen K if the executed discordance is the proxy inflated by a factor.
    k_eval = k_need if k_need is not None else cap
    sens = {}
    for f_ in (1.0, 1.25, 1.5, 2.0, 3.0):
        nm_f = np.array([_n_min(det, min(1.0, v * infl * f_)) or 10**9 for v in d], float)
        sens[str(f_)] = p_meet(k_eval, nm_f)
    res = {"mode": mode, "discordance_inflation_used_for_sizing": infl, "n_trials_test_slots": a.n_trials, "clusters": G, "cap_per_task": cap,
           "target_probability": target, "extension_per_task": k_need,
           "predicted_probability_at_K": p_meet(k_need) if k_need is not None else p_meet(cap),
           "predicted_probability_without_extension": p_meet(0),
           "sensitivity_executed_discordance_inflation": sens,
           "note": "shadow-pass discordance is a planning proxy; executed discordance on joint failures is "
                   "reported by 04_report.py (method_families.csv: executed_discordance) and decides nothing",
           "pi00_median": float(np.median(pi00)), "pi00_90pct_interval": [float(np.quantile(pi00, .05)), float(np.quantile(pi00, .95))],
           "n_min_median": float(np.median(nmin)) if np.median(nmin) < 10**9 else None,
           "p_discordance_above_table": float(np.mean(nmin >= 10**9)),
           "n_min": int(np.median(nmin)) if np.median(nmin) < 10**9 else None,
           "verdict": "feasible" if k_need is not None else
                      "INFEASIBLE within the cap: detection non-inferiority cannot be confirmed; the method claim is "
                      "withheld a priori (reported as such)",
           "inputs": {"joint_counts": a.joint_counts, "disc_counts": a.disc_counts, "p_fail": a.p_fail,
                      "rho": a.rho if a.p_fail else None, "disc_hat": a.disc_hat}}
    (out / "stage_c_sizing.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if k_need is not None:
        print(f"-> run each deployed 10% configuration with: 01_collect.py ... --extension {k_need}")


if __name__ == "__main__":
    main()
