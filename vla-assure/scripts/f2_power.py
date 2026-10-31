"""Joint power and error rates of the COMPLETE F2 decision rule (the same rule applies to F3 and F4).

The method claim against baseline b needs all three component tests, combined into one composite p-value,
then Holm across the pre-specified baselines, AND the pre-registered detection feasibility gate:

  p_FPR,b  : FPR superiority on joint successes                 (WCR, H0: D >= 0)
  p_succ,b : task-success non-inferiority on all paired trials   (WCR, margin 0.05)
  p_TPR,b  : detection non-inferiority on joint failures         (WCR, margin 0.10)
  all three on the pre-registered method-comparison population: test slots + K extension trials per task
  per suite (the conformal guarantee is not involved in F2-F4, so no finite-population argument is needed)
  p_b      = max(p_FPR, (0.0375/0.03) p_succ, (0.0375/0.02) p_TPR)  (capped at 1; missing -> 1)
  claim_b  = Holm-adjusted p_b < 0.0375  and  n_jf >= N_min  and  >= 12 tasks with a joint failure

Component power estimates (power_analysis.py, detection_feasibility.py) ignore that all three must pass
together, with Holm across baselines; this script simulates the whole rule on one generator:
  - 20 task clusters (2 suites x 10 tasks), 350 paired test trials, 20 K extension trials (K per task per suite);
  - success of the proposed configuration Bernoulli(p_g), task-heterogeneous; each baseline's success is
    correlated with it through rho (joint table from p and rho), so joint successes / failures are realistic;
  - FPR differences on joint successes: P(d = -1) = (c_s + Delta)/2, P(d = +1) = (c_s - Delta)/2 (true D = -Delta);
  - success differences arise from the joint table (equal success rates unless --succ_gap);
  - detection differences on joint failures: P(d = +1) = (c_f + tpr_gap)/2, P(d = -1) = (c_f - tpr_gap)/2.
Reports, per design point: P(claim against ALL baselines), P(against at least one), each component's and the
gate's pass rate, and the expected n_jf. Null configurations check the family-wise error of the full rule.

  python scripts/f2_power.py                      # default grid, ~5 min CPU
Writes results/power/f2_power.csv.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assure.common import load_configs  # noqa: E402
from assure.metrics import holm, wild_cluster_test  # noqa: E402


def ternary(rng, n, p_plus, p_minus):
    u = rng.random(n)
    return np.where(u < p_plus, 1.0, np.where(u < p_plus + p_minus, -1.0, 0.0))


def one_study(rng, a, delta, K, succ_gap, tpr_gap, n_min, levels, B, seed):
    G = 20
    n_test, n_ext = a.n_test, G * K
    g_test = np.sort(rng.integers(0, G, n_test))
    g_ext = np.repeat(np.arange(G), K)
    p_g = np.clip(a.success + rng.normal(0, a.sd_success, G), 0.05, 0.98)
    pv, pt = [], {}
    gates = []
    comp = {"fpr": [], "succ": [], "tpr": []}
    for b in range(a.n_baselines):
        out = {}
        for part, g in (("test", g_test), ("ext", g_ext)):
            pp = p_g[g]
            pb = np.clip(pp - succ_gap, 0.01, 0.99)
            sp = rng.random(len(g)) < pp
            pi11 = pp * pb + a.rho * np.sqrt(pp * (1 - pp) * pb * (1 - pb))
            pi11 = np.clip(pi11, np.maximum(0, pp + pb - 1), np.minimum(pp, pb))
            p_b1 = np.where(sp, pi11 / pp, (pb - pi11) / (1 - pp))
            sb = rng.random(len(g)) < np.clip(p_b1, 0, 1)
            out[part] = (g, sp, sb)
        # F2-F4 population (pre-registered): test slots + extension trials for ALL three components
        g = np.concatenate([out["test"][0], out["ext"][0]])
        sp = np.concatenate([out["test"][1], out["ext"][1]])
        sb = np.concatenate([out["test"][2], out["ext"][2]])
        js = sp & sb
        cs = np.clip(a.c_s + rng.normal(0, 0.02, G), delta + 1e-6, 1)[g[js]]
        d_fpr = ternary(rng, js.sum(), (cs - delta) / 2, (cs + delta) / 2)   # alarm_p - alarm_b
        p_fpr = wild_cluster_test(d_fpr, g[js], 0.0, "less", B, seed=seed + 10 * b)["p"] if js.sum() > 5 else 1.0
        d_s = sb.astype(float) - sp.astype(float)                           # s_b - s_p (D > 0 = proposed worse)
        p_succ = wild_cluster_test(d_s, g, 0.05, "less", B, seed=seed + 10 * b + 1)["p"]
        gf = g[~sp & ~sb]
        n_jf, g_f = len(gf), len(np.unique(gf))
        cf = np.clip(a.c_f + rng.normal(0, 0.03, G), tpr_gap + 1e-6, 1)[gf]
        d_t = ternary(rng, n_jf, (cf + tpr_gap) / 2, (cf - tpr_gap) / 2)    # alarm_b - alarm_p
        p_tpr = wild_cluster_test(d_t, gf, 0.10, "less", B, seed=seed + 10 * b + 2)["p"] if n_jf >= 10 else np.nan
        gate = bool(n_jf >= n_min and g_f >= 12)
        p_iut = max(p_fpr, levels[0] / levels[1] * p_succ, levels[0] / levels[2] * (1.0 if p_tpr != p_tpr else p_tpr))
        pv.append(min(1.0, p_iut))
        gates.append(gate)
        comp["fpr"].append(p_fpr < levels[0])
        comp["succ"].append(p_succ < levels[1])
        comp["tpr"].append((p_tpr == p_tpr) and p_tpr < levels[2])
        pt.setdefault("n_jf", []).append(n_jf)
    adj = holm(np.array(pv))
    claims = (adj < levels[0]) & np.array(gates)
    return claims, comp, gates, float(np.mean(pt["n_jf"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--n_test", type=int, default=350)
    ap.add_argument("--n_baselines", type=int, default=3)
    ap.add_argument("--success", type=float, default=0.75)
    ap.add_argument("--sd_success", type=float, default=0.08)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--c_s", type=float, default=0.10, help="alarm discordance on joint successes")
    ap.add_argument("--c_f", type=float, default=0.10, help="alarm discordance on joint failures")
    ap.add_argument("--n_min", type=int, default=200, help="pre-registered N_min (0.10 x inflation 1.5 -> 0.15 -> 200)")
    ap.add_argument("--K", type=int, nargs="*", default=[0, 45, 83])
    ap.add_argument("--delta", type=float, nargs="*", default=[0.02, 0.03, 0.04])
    ap.add_argument("--size", action="store_true",
                    help="Stage C sizing: smallest K (step 5) with P(claim vs ALL baselines) >= --target at --delta_star; "
                         "merged into results/power/stage_c_sizing.json as K = max(K_detection, K_joint)")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--delta_star", type=float, default=0.03)
    a = ap.parse_args()
    _, exp = load_configs()
    st, det = exp["stats"], exp["families"]["detection"]
    levels = (st["level_degraded"], st["level_noninf"], det["level_tpr_noninf"])
    rng = np.random.default_rng(23)
    if a.size:
        import json
        cap = det["max_extension_per_task"]
        k_joint, pw = None, None
        for k in range(0, cap + 1, 5):
            hits = sum(one_study(rng, a, a.delta_star, k, 0.0, 0.0, a.n_min, levels, a.B, seed=100 * r)[0].all()
                       for r in range(a.reps))
            pw = hits / a.reps
            print(f"K {k:3d}: P(claim vs all) {pw:.3f}", flush=True)
            if pw >= a.target:
                k_joint = k
                break
        f = ROOT / "results/power/stage_c_sizing.json"
        plan = json.loads(f.read_text()) if f.exists() else {}
        k_det = plan.get("extension_per_task")
        plan.update({"joint_power_target": a.target, "delta_star": a.delta_star, "extension_joint_power": k_joint,
                     "joint_power_at_K_joint": pw if k_joint is not None else None,
                     "extension_final": (max(k_det, k_joint) if (k_det is not None and k_joint is not None) else None),
                     "joint_inputs": {"success": a.success, "rho": a.rho, "c_s": a.c_s, "c_f": a.c_f, "n_min": a.n_min}})
        if plan["extension_final"] is None:
            plan["verdict"] = ("INFEASIBLE within the cap: the full F2 rule cannot reach the target power; the method "
                               "claim is withheld a priori (reported as such)")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(plan, indent=2))
        print(json.dumps({k: plan[k] for k in ("extension_per_task", "extension_joint_power", "extension_final")
                          if k in plan}))
        return
    designs = [dict(kind="alternative", delta=d, K=k, succ_gap=0.0, tpr_gap=0.0) for d, k in itertools.product(a.delta, a.K)]
    for k in a.K[1:]:  # null configurations of the full rule (each violates one component's alternative)
        designs += [dict(kind="null: FPR equal", delta=0.0, K=k, succ_gap=0.0, tpr_gap=0.0),
                    dict(kind="null: success worse by margin", delta=max(a.delta), K=k, succ_gap=-0.05, tpr_gap=0.0),
                    dict(kind="null: detection worse by margin", delta=max(a.delta), K=k, succ_gap=0.0, tpr_gap=0.10)]
    rows = []
    for dsg in designs:
        all_, any_, cf, cs, ct, gt, njf = 0, 0, 0, 0, 0, 0, []
        for r in range(a.reps):
            claims, comp, gates, nj = one_study(rng, a, dsg["delta"], dsg["K"], dsg["succ_gap"], dsg["tpr_gap"],
                                                a.n_min, levels, a.B, seed=100 * r)
            all_ += claims.all()
            any_ += claims.any()
            cf += np.mean(comp["fpr"])
            cs += np.mean(comp["succ"])
            ct += np.mean(comp["tpr"])
            gt += np.mean(gates)
            njf.append(nj)
        mc = lambda k: 1.96 * np.sqrt(max(k / a.reps * (1 - k / a.reps), 1e-12) / a.reps)  # noqa: E731
        row = dict(**dsg, reps=a.reps, p_claim_all=all_ / a.reps, p_claim_any=any_ / a.reps, mc95_any=mc(any_),
                   fpr_component=cf / a.reps, success_component=cs / a.reps, detection_component=ct / a.reps,
                   detection_gate=gt / a.reps, mean_n_jf=float(np.mean(njf)), success=a.success, rho=a.rho,
                   c_s=a.c_s, c_f=a.c_f, n_min=a.n_min, n_test=a.n_test)
        rows.append(row)
        print(f"{dsg['kind']:32s} Delta {dsg['delta']:.2f} K {dsg['K']:3d}: claim vs all {row['p_claim_all']:.3f}, "
              f"vs >=1 {row['p_claim_any']:.3f}+/-{row['mc95_any']:.3f} | components FPR {row['fpr_component']:.2f} "
              f"succ {row['success_component']:.2f} TPR {row['detection_component']:.2f} gate {row['detection_gate']:.2f} "
              f"| n_jf {row['mean_n_jf']:.0f}", flush=True)
    import pandas as pd
    out = ROOT / "results/power/f2_power.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
