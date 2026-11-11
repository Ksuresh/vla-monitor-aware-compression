"""Robustness of the closed-loop false-alarm result to the calibration split (CPU only, no robot runs).

The paired TEST slots stay fixed (variants were only run there). The remaining reference slots
(pre-registered train / calib / repair) are re-partitioned at random S times into train and calib sets of
the original sizes; for each split the monitor is refitted, the conformal band recalibrated and the paired
test trials re-scored. Reports, per monitor x variant: the paired outcome transition table, false alarms on
joint successes (fp16 vs variant) with exact Clopper-Pearson 95% intervals for the pre-registered split,
the median and 5-95% range over splits, the per-task alarm rates, and detection of the NEW failures.

  python scripts/resplit.py --model smolvla --key libero_spatial --taps expert_last --splits 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assure import monitors as M  # noqa: E402
from assure.common import EXT_OFFSET, ROOT, TOPUP_OFFSET, load_configs, role_of, trial_roles  # noqa: E402
from assure.conformal import FunctionalCP  # noqa: E402
from assure.metrics import auroc  # noqa: E402
from assure.store import load_cell  # noqa: E402


def cp95(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 3), round(float(hi), 3)]


def eprocess_flag(alarms, alpha_tol=0.15, p1=0.5, delta=0.05):
    """Likelihood-ratio e-process on a success-only alarm sequence; True once E >= 1/delta (Ville)."""
    up, dn = np.log(p1 / alpha_tol), np.log((1 - p1) / (1 - alpha_tol))
    L = np.cumsum(np.where(np.asarray(alarms, bool), up, dn))
    return bool((L >= np.log(1 / delta)).any())


def certify(alarms, alpha_tol=0.15, conf=0.95):
    """True if the one-sided Clopper-Pearson upper bound on the inherited false-alarm rate is <= alpha_tol."""
    n, k = len(alarms), int(np.sum(alarms))
    return n > 0 and (beta.ppf(conf, k + 1, n - k) if k < n else 1.0) <= alpha_tol


def certify_rule(n, alpha_tol=0.15, conf=0.95):
    """Exact acceptance rule: largest alarm count k with one-sided Clopper-Pearson upper bound <= alpha_tol, and the
    test's exact size P(Bin(n, alpha_tol) <= k) (the worst-case probability of keeping a threshold at FPR = alpha_tol)."""
    from scipy.stats import binom
    ks = [k for k in range(n) if beta.ppf(conf, k + 1, n - k) <= alpha_tol]
    if not ks:
        return None, 0.0
    return ks[-1], float(binom.cdf(ks[-1], n, alpha_tol))


def _release(mon, cp, train, fit, test, v_ok, a, cpkw, guarantee, n_req, rng, alpha_tol, cache=None):
    """Fit every release option on `fit` episodes (indices into V via closures) and judge it on `test`."""
    from assure.fixes import DiagCoral
    order = [fit[i] for i in rng.permutation(len(fit)) if fit[i].success]
    fit_succ = order[:n_req]          # first n_req successes, random order (the minimum budget)
    all_succ = order                  # every success in the fitting pool (budget fixed in advance by the pool size)
    sc = (lambda e: cache[id(e)]) if cache is not None else mon.score  # noqa: E731
    alarm = lambda c, e, f=None: c.first_alarm((f or sc)(e)) is not None  # noqa: E731
    ev = lambda c, f=None: (np.array([alarm(c, e, f) for e in test if e.success]),  # noqa: E731
                            np.array([alarm(c, e, f) for e in test if not e.success]))
    feasible = len(fit_succ) >= n_req
    outs = {"inherited": ev(cp)}
    if mon.taps:
        mon.feature_transform = DiagCoral(mon.taps).fit(train, fit)
        outs["coral"] = ev(cp, mon.score)      # realigned features: fresh scores
        mon.feature_transform = None
    inh = [alarm(cp, e) for e in fit_succ]
    flagged, certified = eprocess_flag(inh, alpha_tol), certify(inh, alpha_tol)
    if feasible:
        kw = dict(cpkw, guarantee=guarantee)
        recal = ev(FunctionalCP(a.alpha, **kw).fit([sc(e) for e in fit_succ]))
        outs["recal"] = recal
        outs["flag_then_recal"] = recal if flagged else outs["inherited"]
        outs["certify_or_recal"] = outs["inherited"] if certified else recal
        # Detection-preserving variant: recalibrate on ALL successes in the pool (n >= n_req). With more threshold
        # episodes the conformal rank sits below the maximum, so the threshold is less conservative and detection
        # recovers, while the same marginal / PAC guarantee holds at the larger n.
        if len(all_succ) > n_req:
            recal_all = ev(FunctionalCP(a.alpha, **kw).fit([sc(e) for e in all_succ]))
            outs["recal_all"] = recal_all
            outs["certify_or_recal_all"] = outs["inherited"] if certify([alarm(cp, e) for e in all_succ], alpha_tol) else recal_all
    return outs, dict(n_all_success=len(all_succ), n_fit_success=len(fit_succ), feasible=feasible, flagged=flagged, certified=certified)


def compare_fixes(mon, cp, train, V, a, cpkw, seed, pool=None, folds=10, alpha_tol=0.15):
    """Release options for the changed policy, judged on held-out deployment episodes.
      inherited         fp16 band unchanged
      recal             band refitted on the first n_req successes (19 marginal / 44 PAC): certified
      coral             label-free realignment of features onto the reference statistics, fp16 band kept
      flag_then_recal   recalibrate only if the anytime-valid test flags: RISK-BASED RETENTION, a retained
                        inherited band carries no guarantee
      certify_or_recal  keep the inherited band only if a one-sided 95% Clopper-Pearson bound on its false-alarm
                        rate over the same successes is <= alpha_tol, else recalibrate (PAC): every released band
                        satisfies P(FPR > alpha_tol) <= 0.05 + delta by a union bound
    Modes: 'separate_pool' fits on the changed policy's repair/top-up episodes (disjoint from test) and judges on
    all test episodes; 'crossfit_*' split the test slots at random (pooled, so fit and judged episodes are
    exchangeable): marginal with half/half, PAC with 70/30 so that ~44 successes are available."""
    modes = []
    if pool:
        modes += [("separate_pool_marginal", "marginal", 19, None), ("separate_pool_pac", "pac", 44, None)]
    modes += [("crossfit_marginal", "marginal", 19, 0.5), ("crossfit_pac", "pac", 44, 0.7)]
    summ = lambda x: [float(np.nanmedian(x)), float(np.nanpercentile(x, 5)), float(np.nanpercentile(x, 95))]  # noqa: E731
    report = {"alpha_tol": alpha_tol}
    cache = {id(e): mon.score(e) for e in list(V) + list(pool or [])}   # scores of the unmodified monitor, once
    for name, g, n_req, frac in modes:
        res, info = {}, []
        reps = [(pool, V)] if frac is None else []
        for f in range(folds if frac is not None else 5):
            rng = np.random.default_rng([seed, 900 + f])
            if frac is not None:
                perm = rng.permutation(len(V)); cut = int(round(frac * len(V)))
                reps = [([V[i] for i in perm[:cut]], [V[i] for i in perm[cut:]])]
                if frac == 0.5:
                    reps.append(reps[0][::-1])
            for fit, test in reps:
                outs, inf = _release(mon, cp, train, fit, test, None, a, cpkw, g, n_req, rng, alpha_tol, cache)
                info.append(inf)
                for k, (fs, ff) in outs.items():
                    d = res.setdefault(k, {"fpr": [], "tpr": [], "viol": []})
                    d["fpr"].append(fs.mean() if len(fs) else np.nan)
                    d["tpr"].append(ff.mean() if len(ff) else np.nan)
                    d["viol"].append(float(fs.mean() > alpha_tol) if len(fs) else np.nan)
        k_max, size = certify_rule(n_req, alpha_tol)
        from assure.conformal import pac_violation_prob
        n_conf = n_req - max(cpkw.get("min_shape", 10), int(np.ceil(cpkw.get("shape_frac", 0.5) * n_req)))
        pac = pac_violation_prob(n_conf, n_conf, a.alpha) if g == "pac" else float("nan")
        report[name] = {"guarantee": g, "n_required": n_req,
                        "certify_max_alarms": k_max, "certify_exact_size": size,
                        "combined_failure_bound": size + pac if g == "pac" else float("nan"),
                        "n_fit_success": [int(min(x["n_fit_success"] for x in info)),
                                          float(np.median([x["n_fit_success"] for x in info]))],
                        "feasible_share": float(np.mean([x["feasible"] for x in info])),
                        "flag_rate": float(np.mean([x["flagged"] for x in info])),
                        "certify_rate": float(np.mean([x["certified"] for x in info])),
                        "options": {k: {"fpr": summ(d["fpr"]), "tpr": summ(d["tpr"]), "p_violate": float(np.nanmean(d["viol"]))}
                                    for k, d in res.items()}}
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--rollouts", default=str(ROOT / "rollouts"))
    ap.add_argument("--monitors", default="mahalanobis,rnd,safe_mlp")
    ap.add_argument("--taps", default=None)
    ap.add_argument("--variants", default=None, help="comma list (default: all variant folders)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results/resplit.json"))
    a = ap.parse_args()
    models, exp = load_configs()
    taps = a.taps.split(",") if a.taps else list(models[a.model]["tap"])
    rj = Path(a.rollouts) / a.model / "fp16" / a.key / "run.json"
    root = Path(a.rollouts)
    ref = load_cell(root, a.model, "fp16", a.key)
    run = json.loads(rj.read_text()) if rj.exists() else {}
    R = int(run.get("n_reference") or exp["reference_episodes_per_task"])
    roles = trial_roles(exp["seed"], sorted({e.meta["task_id"] for e in ref}), R, run.get("split") or exp["split"])
    role = lambda e: role_of(roles, e.meta["task_id"], e.meta["episode_idx"])  # noqa: E731
    key = lambda e: (e.meta["task_id"], e.meta["episode_idx"])  # noqa: E731
    ref_test = {key(e): e for e in ref if role(e) == "test"}
    pool = [e for e in ref if role(e) in ("train", "calib", "repair")]
    n_train = sum(role(e) == "train" for e in ref)
    n_cal = sum(role(e) == "calib" for e in ref)
    names = a.variants.split(",") if a.variants else sorted(
        p.name for p in (root / a.model).iterdir() if p.name != "fp16" and (p / a.key).exists())
    var_all = {v: load_cell(root, a.model, v, a.key) for v in names}
    var_test = {v: {key(e): e for e in var_all[v] if role(e) == "test"} for v in names}
    # the changed policy's own calibration pool: repair slots + top-ups (never test), if they were run
    var_pool = {v: [e for e in var_all[v] if role(e) == "repair" or TOPUP_OFFSET <= e.meta["episode_idx"] < EXT_OFFSET]
                for v in names}
    cal = exp.get("calibration", {})
    cpkw = dict(guarantee=cal.get("guarantee", "marginal"), delta=cal.get("delta", 0.1),
                shape_frac=cal.get("shape_frac", 0.5), min_shape=cal.get("min_shape", 10),
                strict=cal.get("strict", False), seed=exp["seed"])
    print(f"taps {taps} | test slots {len(ref_test)} | re-split pool {len(pool)} -> train {n_train} / calib {n_cal}")

    out = {"taps": taps, "alpha": a.alpha, "splits": a.splits, "variants": {}}
    for v in names:
        ks = sorted(set(ref_test) & set(var_test[v]))
        r_ok = np.array([ref_test[k].success for k in ks], bool)
        v_ok = np.array([var_test[v][k].success for k in ks], bool)
        task = np.array([k[0] for k in ks])
        trans = {"both_succeed": int((r_ok & v_ok).sum()), "new_failures": int((r_ok & ~v_ok).sum()),
                 "rescued": int((~r_ok & v_ok).sum()), "both_fail": int((~r_ok & ~v_ok).sum())}
        b, c = trans["new_failures"], trans["rescued"]
        from scipy.stats import binomtest
        mcnemar_p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else float("nan")
        print(f"\n=== {v}: paired transitions {trans} | flipped {(b + c) / len(ks):.0%} of slots | "
              f"McNemar p = {mcnemar_p:.3f}")
        js, nf = r_ok & v_ok, r_ok & ~v_ok
        vs, vf, rs = v_ok, ~v_ok, r_ok      # PRIMARY: all successes of each policy (Section III definition)
        vres = {"n_pairs": len(ks), "transitions": trans, "mcnemar_p": mcnemar_p, "monitors": {}}
        for mname in a.monitors.split(","):
            base, half = M.parse_name(mname)
            per, fixes = [], None
            for s in range(a.splits + 1):   # s = 0: the pre-registered split
                if s == 0:
                    tr = [e for e in pool if role(e) == "train"]
                    ca = [e for e in pool if role(e) == "calib"]
                else:
                    p = np.random.default_rng([exp["seed"], 500 + s]).permutation(len(pool))
                    tr = [pool[i] for i in p[:n_train]]
                    ca = [pool[i] for i in p[n_train:n_train + n_cal]]
                try:
                    mon = M.build(base, taps).fit(M.train_subset(tr, half, exp["seed"]))
                    cp = FunctionalCP(a.alpha, **cpkw).fit([mon.score(e) for e in ca if e.success])
                except Exception as ex:  # e.g. too few failures for a supervised probe in this split
                    print(f"  [{mname}] split {s} skipped: {ex}")
                    continue
                R_ = [ref_test[k] for k in ks]
                V_ = [var_test[v][k] for k in ks]
                a0 = np.array([cp.first_alarm(mon.score(e)) is not None for e in R_])
                a1 = np.array([cp.first_alarm(mon.score(e)) is not None for e in V_])
                per.append(dict(split=s,
                                fpr_fp16_all=float(a0[rs].mean()), fpr_var_all=float(a1[vs].mean()),
                                k_fp16_all=int(a0[rs].sum()), k_var_all=int(a1[vs].sum()),
                                fpr_var_rescued=float(a1[~r_ok & v_ok].mean()) if (~r_ok & v_ok).any() else float("nan"),
                                fpr_fp16=float(a0[js].mean()), fpr_var=float(a1[js].mean()),
                                k_fp16=int(a0[js].sum()), k_var=int(a1[js].sum()),
                                detect_new=float(a1[nf].mean()) if nf.any() else float("nan"),
                                tpr_var=float(a1[vf].mean()) if vf.any() else float("nan"),
                                task_fpr_var={int(t): float(a1[vs & (task == t)].mean()) for t in np.unique(task[vs])}))
                if s == 0:
                    mx = lambda e: float(np.max(mon.score(e)))  # noqa: E731  threshold-free ranking
                    per[-1]["auroc_fp16"] = auroc(np.array([mx(e) for e in R_]), (~r_ok).astype(int))
                    per[-1]["auroc_var"] = auroc(np.array([mx(e) for e in V_]), (~v_ok).astype(int))
                    fixes = compare_fixes(mon, cp, tr, V_, a, cpkw, exp["seed"], pool=var_pool[v])
            if not per:
                continue
            p0 = per[0] if per[0]["split"] == 0 else None
            arr = lambda f: np.array([q[f] for q in per])  # noqa: E731
            n, nv, nr = int(js.sum()), int(vs.sum()), int(rs.sum())
            r = {"n_joint_success": n, "n_var_success": nv, "n_fp16_success": nr, "n_new_failures": int(nf.sum()),
                 "preregistered": p0, "fixes_crossfit": fixes,
                 "fpr_fp16_all_ci95": cp95(p0["k_fp16_all"], nr) if p0 else None,
                 "fpr_var_all_ci95": cp95(p0["k_var_all"], nv) if p0 else None,
                 "fpr_fp16_ci95": cp95(p0["k_fp16"], n) if p0 else None,
                 "fpr_var_ci95": cp95(p0["k_var"], n) if p0 else None,
                 "over_splits": {f: [float(np.nanmedian(arr(f))), float(np.nanpercentile(arr(f), 5)),
                                     float(np.nanpercentile(arr(f), 95))]
                                 for f in ("fpr_fp16_all", "fpr_var_all", "fpr_fp16", "fpr_var", "detect_new", "tpr_var")},
                 "n_splits_ok": len(per)}
            vres["monitors"][mname] = r
            o = r["over_splits"]
            if p0:
                print(f"[{mname}] PRIMARY all successes: FPR fp16 {p0['k_fp16_all']}/{nr} = {p0['fpr_fp16_all']:.2f} "
                      f"{r['fpr_fp16_all_ci95']} -> {v} {p0['k_var_all']}/{nv} = {p0['fpr_var_all']:.2f} "
                      f"{r['fpr_var_all_ci95']} (rescued successes {p0['fpr_var_rescued']:.2f}) | over {len(per)} splits: "
                      f"{o['fpr_fp16_all'][0]:.2f} [{o['fpr_fp16_all'][1]:.2f}, {o['fpr_fp16_all'][2]:.2f}] -> "
                      f"{o['fpr_var_all'][0]:.2f} [{o['fpr_var_all'][1]:.2f}, {o['fpr_var_all'][2]:.2f}]")
                print(f"   paired (joint successes n={n}): {p0['fpr_fp16']:.2f} {r['fpr_fp16_ci95']} -> {p0['fpr_var']:.2f} "
                      f"{r['fpr_var_ci95']} | detection: all {v} failures {p0['tpr_var']:.2f}, new {p0['detect_new']:.2f} "
                      f"| AUROC (threshold-free) fp16 {p0['auroc_fp16']:.2f} -> {v} {p0['auroc_var']:.2f}")
                print("   per-task FPR on " + v + " successes: " + ", ".join(f"t{t} {x:.2f}" for t, x in p0["task_fpr_var"].items()))
            if fixes:
                for mode, m_ in fixes.items():
                    if mode == "alpha_tol":
                        continue
                    print(f"   release [{mode}] successes available {m_['n_fit_success'][1]:.0f} (min {m_['n_fit_success'][0]}) "
                          f"vs {m_['n_required']} required -> feasible {m_['feasible_share']:.0%} | flagged {m_['flag_rate']:.0%}, "
                          f"certified {m_['certify_rate']:.0%} | keep if <= {m_['certify_max_alarms']} alarms "
                          f"(exact size {m_['certify_exact_size']:.3f})"
                          + (f" | P(released FPR > tol) <= {m_['combined_failure_bound']:.3f}" if m_['guarantee'] == 'pac' else ""))
                    for fx, o_ in m_["options"].items():
                        print(f"      {fx:17s} held-out FPR {o_['fpr'][0]:.2f} [{o_['fpr'][1]:.2f}, {o_['fpr'][2]:.2f}] | "
                              f"P(FPR > {fixes['alpha_tol']}) {o_['p_violate']:.2f} | detection {o_['tpr'][0]:.2f}")
        out["variants"][v] = vres
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
