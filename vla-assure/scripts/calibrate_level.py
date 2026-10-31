"""Size calibration of the confirmatory test (WCR bootstrap-t, Webb weights) before any data is seen.

The non-inferiority test of a mean paired difference of rare binary alarms is slightly liberal at the
margin with ~20 clusters (the t statistic is skewed). This script estimates the actual size at the
null boundaries over a grid of nuisance settings and reports the nominal level whose MEAN size over the
grid is <= 0.05. Pre-registered result (seed 11, 600 studies per point, B = 1999, 20 clusters, 340
pairs): nominal 0.05 -> mean size 0.059; 0.04 -> 0.048; 0.03 -> 0.037. Degradation test at 0.05 -> 0.049.
Hence levels 0.03 (non-inferiority) and 0.05 (degradation) in experiment.yaml:stats.
After Stage A, rerun with the observed base rate / discordance / heterogeneity:

  python scripts/calibrate_level.py                         # pre-registered grid
  python scripts/calibrate_level.py --base 0.11 --flip 0.25 --sd 0.03 --reps 2000   # Stage A values
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assure.common import ROOT  # noqa: E402
from assure.metrics import wild_cluster_test  # noqa: E402


def sim(rng, G, n, shift, base, flip, sd):
    g = rng.integers(0, G, n)
    p_ref = np.clip(base + rng.normal(0, sd, G), 0.01, 0.6)[g]
    ref = (rng.random(n) < p_ref).astype(float)
    var = ref.copy()
    var[(ref == 1) & (rng.random(n) < flip)] = 0
    var[(ref == 0) & (rng.random(n) < (flip * p_ref / (1 - p_ref) + shift / (1 - p_ref)))] = 1
    return var - ref, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=340)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--base", type=float, nargs="*", default=[0.05, 0.09, 0.15])
    ap.add_argument("--flip", type=float, nargs="*", default=[0.2, 0.3, 0.5])
    ap.add_argument("--sd", type=float, nargs="*", default=[0.02, 0.04, 0.08])
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--refine", type=int, default=3, help="re-simulate the K worst settings at the chosen levels")
    ap.add_argument("--reps_refine", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    levels = np.round(np.arange(0.005, 0.0501, 0.0025), 4)
    rows, P_ni, P_dg = [], [], []
    for base, flip, sd in itertools.product(a.base, a.flip, a.sd):
        ni, dg = [], []
        for r in range(a.reps):
            d, g = sim(rng, a.clusters, a.pairs, a.margin, base, flip, sd)
            ni.append(wild_cluster_test(d, g, a.margin, "less", a.B, seed=r)["p"])
            d, g = sim(rng, a.clusters, a.pairs, 0.0, base, flip, sd)
            dg.append(wild_cluster_test(d, g, 0.0, "greater", a.B, seed=r + 7)["p"])
        P_ni.append(np.array(ni)); P_dg.append(np.array(dg))
        rows.append({"base": base, "flip": flip, "sd": sd})
        print(rows[-1], flush=True)
    P_ni, P_dg = np.array(P_ni), np.array(P_dg)
    mc = lambda x: 1.96 * np.sqrt(x * (1 - x) / a.reps)  # noqa: E731

    def summary(P, lv):
        size = (P < lv).mean(1)
        return {"nominal": float(lv), "mean": float(size.mean()), "max": float(size.max()),
                "max_mc95": float(mc(size.max())), "argmax": rows[int(size.argmax())],
                "n_above_target": int((size > a.target).sum()),
                "n_significantly_above": int((size - mc(size) > a.target).sum()),
                "per_setting": [float(x) for x in size]}
    out = {"settings": vars(a), "grid": rows, "noninferiority": {}, "degradation": {}}
    for lv in levels:
        out["noninferiority"][str(lv)] = summary(P_ni, lv)
        out["degradation"][str(lv)] = summary(P_dg, lv)

    def worst_case_level(key):
        ok = [float(lv) for lv in levels if out[key][str(lv)]["max"] <= a.target]
        return max(ok) if ok else float(levels[0])
    out["chosen"] = {"noninferiority": worst_case_level("noninferiority"),
                     "degradation": worst_case_level("degradation"),
                     "rule": f"largest nominal level whose MAXIMUM size over the grid is <= {a.target}"}
    # Refinement: the grid maximum carries Monte-Carlo error (+/- ~0.01 at 2000 studies), so the K worst
    # settings at the chosen levels are re-simulated with reps_refine studies and judged by the one-sided
    # 95% Clopper-Pearson UPPER bound. Stage A rule: an operating level is kept only if every refined upper
    # bound is <= target; otherwise it is lowered by 0.0025 and the refinement repeated. These are
    # simulation-calibrated operating choices for this grid, not a proof of uniform size control.
    from scipy.stats import beta as _beta
    refined = {}
    for key, P, alt, d0 in (("noninferiority", P_ni, "less", a.margin), ("degradation", P_dg, "greater", 0.0)):
        lv = out["chosen"][key]
        while a.refine:
            size = (P < lv).mean(1)
            worst = np.argsort(-size)[:a.refine]
            res_ = []
            for w in worst:
                st_ = rows[int(w)]
                rr = np.random.default_rng(1000 + int(w))
                k = sum(wild_cluster_test(*sim(rr, a.clusters, a.pairs, d0, st_["base"], st_["flip"], st_["sd"]),
                                          d0, alt, a.B, seed=10**6 + r)["p"] < lv for r in range(a.reps_refine))
                ub = float(_beta.ppf(0.95, k + 1, a.reps_refine - k)) if k < a.reps_refine else 1.0
                res_.append({**st_, "size": k / a.reps_refine, "upper95": ub, "reps": a.reps_refine})
                print(f"refine {key} @{lv}: {st_} size {k / a.reps_refine:.4f} upper95 {ub:.4f}", flush=True)
            if all(r_["upper95"] <= a.target for r_ in res_) or lv <= levels[0]:
                refined[key] = {"level": lv, "worst_settings": res_}
                break
            lv = round(lv - 0.0025, 4)
    out["refined"] = refined
    if refined:
        out["operating_levels"] = {k: v["level"] for k, v in refined.items()}
    path = ROOT / "results/power/level_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    for key in ("noninferiority", "degradation"):
        for lv in (0.05, 0.04, 0.03, 0.025, 0.02):
            r = out[key][str(lv)] if str(lv) in out[key] else None
            if r:
                print(f"{key:15s} nominal {lv:.3f}: mean {r['mean']:.4f} max {r['max']:.4f} +/- {r['max_mc95']:.4f} "
                      f"above {a.target}: {r['n_above_target']} (significantly: {r['n_significantly_above']})")
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
