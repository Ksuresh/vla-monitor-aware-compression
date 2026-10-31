"""Train monitors on the reference model, then test them on every compressed/updated variant.

Calibration unit and population
  One conformal band per (model, suite, condition, monitor, alpha), fitted on successful reference
  episodes POOLED over the suite's tasks. Its guarantee covers "a successful episode of the reference
  policy drawn from this cell's trial design" (uniform task mix, LIBERO initial states, this camera
  condition). No per-task guarantee is claimed; per-task rates are descriptive only.

Reference split (exchangeable by design; common.trial_roles)
  Every reference trial slot (task, idx) gets a pre-registered role -- train / calib / test -- from one
  seeded random partition of all slots pooled over tasks, fixed before any rollout. Variants run exactly
  the test slots. Calib and test are therefore a random partition of the same trials: exchangeable
  without further assumptions. Reference top-ups are train-only and never enter calibration.
Variant trials
  Variants run the 'test' slots (all paired, all used for the endpoint) and the 'repair' slots (the
  repair pool). Both roles come from the same partition, so the repair pool is exchangeable with the
  variant's test trials. The monitor is fitted on train slots only; reference runs of repair slots are
  never used for fitting (sp["repair_ref"]), so calibration, repair and test stay outside monitor fitting.

Three analyses per (variant, monitor, alpha):
  deployment        each policy's own test rollouts. What a deployed system sees; mixes feature shift
                    with a change in which episodes succeed.
  paired            the same trials under both policies: false alarms on trials both solved,
                    detection of failures the variant newly introduced, paired task-success change.
  obs_controlled    reference observations fed through a non-controlling SHADOW copy of the variant
                    (collect with --shadow). An observation-controlled MONITOR-SCORE comparison: it
                    isolates representation drift, but the outcome labels are the reference policy's,
                    so it says nothing about what the variant would have done in control. Task success
                    and detection claims come only from executed trajectories (deployment, paired).

Statistics: CIs are task-clustered bootstrap (tasks, then episodes within task). Variant-vs-reference
differences ('cmp_*' columns) are paired where trials match; 04_report.py applies Holm per analysis
family and a prespecified non-inferiority margin.

  python scripts/03_evaluate.py --model pi05 --key libero_10+shift_mild
Output: results/eval/<model>__<key>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assure import monitors as M  # noqa: E402
from assure.common import EXT_OFFSET, ROOT, TOPUP_OFFSET, load_configs, role_of, setup_logging, trial_roles  # noqa: E402
from assure.conformal import FunctionalCP, finite_population_fpr, required_successes  # noqa: E402
from assure.fixes import DiagCoral, feature_drift  # noqa: E402
from assure.metrics import (auroc, auroc_stat, cluster_bootstrap, compare, compare_joint,  # noqa: E402
                            evaluate, masked_mean)
from assure.store import load_cell, split_pooled  # noqa: E402

log = logging.getLogger("evaluate")
MIN_FAIL_TRAIN = 5   # default feasibility gate; supervised probes use experiment.yaml: min_train_failures


def tid(e):
    return e.meta["task_id"]


def idx(e):
    return e.meta["episode_idx"]


def scored(mon, eps):
    return [mon.score(e) for e in eps]


def alarms(cp, mon, eps):
    return np.array([cp.first_alarm(mon.score(e)) is not None for e in eps], float)


def load_onsets(path, model, key):
    """Optional failure-onset annotations: CSV with model,variant,key,task,idx,onset_step (env steps).
    Without them only the normalised alarm time is reported, never an actionable lead time."""
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    df = df[(df.model == model) & (df.key == key)]
    return {(r.variant, int(r.task), int(r.idx)): int(r.onset_step) for r in df.itertuples()}


def metrics_row(cp, mon, eps, boot=500, onsets=None):
    if not eps:
        return {"n": 0}
    seqs = scored(mon, eps)
    failed = [not e.success for e in eps]
    on = [(onsets or {}).get((e.meta.get("variant"), tid(e), idx(e))) for e in eps]
    res = evaluate(cp, seqs, failed, [e.query_step for e in eps], [e.length for e in eps], onsets=on)
    margins = np.array([cp.margin(s) for s in seqs])
    labels = np.array(failed, int)
    alarmed = np.array([cp.first_alarm(s) is not None for s in seqs])
    tasks = np.array([tid(e) for e in eps])
    res["auroc_lo"], res["auroc_hi"] = cluster_bootstrap(lambda i: auroc(margins[i], labels[i]), tasks, boot)
    succ = np.where(labels == 0)[0]
    if len(succ):
        res["fpr_lo"], res["fpr_hi"] = cluster_bootstrap(lambda i: float(alarmed[succ][i].mean()), tasks[succ], boot)
    res["n_tasks"] = int(len(np.unique(tasks)))
    return res


def _alarm_time(cp, mon, e):
    k = cp.first_alarm(mon.score(e))
    return 1.0 if k is None else float(e.query_step[k]) / max(e.length, 1)


def _time_shift(cp, mon, pairs):
    if not pairs:
        return float("nan")
    return float(np.mean([_alarm_time(cp, mon, v) - _alarm_time(cp, mon, r) for r, v in pairs]))


def trial_table(cp, mon, ref_test, var_test):
    """Trial-aligned arrays over the test slots both policies ran: success, alarm, alarm statistic, task."""
    ref = {e.pair_key: e for e in ref_test}
    pairs = [(ref[e.pair_key], e) for e in var_test if e.pair_key in ref]
    if not pairs:
        return None
    rs = np.array([r.success for r, _ in pairs])
    vs = np.array([v.success for _, v in pairs])
    sr = [mon.score(r) for r, _ in pairs]
    sv = [mon.score(v) for _, v in pairs]
    ra = np.array([cp.first_alarm(x) is not None for x in sr], float)
    va = np.array([cp.first_alarm(x) is not None for x in sv], float)
    rm = np.array([cp.margin(x) for x in sr])
    vm = np.array([cp.margin(x) for x in sv])
    return rs, vs, ra, va, rm, vm, np.array([tid(r) for r, _ in pairs])


def cmp_cols(prefix, d):
    return {f"cmp_{prefix}_{k}": v for k, v in d.items()}


def paired_row(cp, mon, ref_eps, var_eps, margin=0.05, success_margin=0.05, dump=None):
    """Trial-matched comparison between reference and variant (executed trajectories only)."""
    ref = {e.pair_key: e for e in ref_eps}
    pairs = [(ref[e.pair_key], e) for e in var_eps if e.pair_key in ref]
    if not pairs:
        return None
    both = [(r, v) for r, v in pairs if r.success and v.success]
    induced = [v for r, v in pairs if r.success and not v.success]
    rescued = [v for r, v in pairs if not r.success and v.success]
    both_fail = [v for r, v in pairs if not r.success and not v.success]
    a_ref = alarms(cp, mon, [r for r, _ in both])
    a_var = alarms(cp, mon, [v for _, v in both])
    if dump is not None:  # per-trial rows for the pooled primary endpoint (04_report.py)
        for (r, v), x, y in zip(both, a_ref, a_var):
            dump.append(dict(task=tid(r), idx=idx(r), ref_alarm=int(x), var_alarm=int(y)))
    out = {
        "n_pairs": len(pairs), "n_both_success": len(both), "n_induced_fail": len(induced),
        "n_rescued": len(rescued), "n_both_fail": len(both_fail),
        "fpr": float(a_var.mean()) if len(both) else float("nan"),
        "fpr_ref_same_trials": float(a_ref.mean()) if len(both) else float("nan"),
        "tpr_induced": float(alarms(cp, mon, induced).mean()) if induced else float("nan"),
        "tpr": float(alarms(cp, mon, induced + both_fail).mean()) if induced + both_fail else float("nan"),
        # alarm timing on trials both policies FAILED (paired): shift of the normalised first-alarm time;
        # positive = the variant alarms later. Missing alarms count as the episode end (1.0).
        "alarm_time_shift_both_fail": _time_shift(cp, mon, [(r, v) for r, v in pairs if not r.success and not v.success]),
        "task_success_ref_same_trials": float(np.mean([r.success for r, _ in pairs])),
        "task_success": float(np.mean([v.success for _, v in pairs])),
    }
    if both:
        out.update(cmp_cols("fpr", compare(a_ref, a_var, [tid(r) for r, _ in both], margin=margin, worse="higher")))
    out.update(cmp_cols("success", compare([float(r.success) for r, _ in pairs], [float(v.success) for _, v in pairs],
                                           [tid(r) for r, _ in pairs], margin=success_margin, worse="lower")))
    return out


def obs_controlled_row(cp, mon, ref_test, variant, margin=0.05, auroc_margin=0.05, boot=500):
    """Reference observations through the shadow model. Labels = REFERENCE outcomes."""
    eps = [e for e in ref_test if variant in e.shadows()]
    views = [e.as_shadow(variant) for e in eps]
    if not views or not all(mon.available(x) for x in views[:3]):
        return None
    res = metrics_row(cp, mon, views, boot)
    res["labels_from"] = "reference_policy"
    m_ref = np.array([cp.margin(mon.score(e)) for e in eps])
    m_sh = np.array([cp.margin(mon.score(v)) for v in views])
    lab = np.array([not e.success for e in eps], int)
    tasks = np.array([tid(e) for e in eps])
    ok = lab == 0
    sd = float(np.std(m_ref[ok])) + 1e-9 if ok.any() else float("nan")
    res["score_shift_sd"] = float(np.mean(m_sh[ok] - m_ref[ok]) / sd) if ok.any() else float("nan")
    # how differently the shadow would have acted: if large, observation-controlled results are a
    # weaker proxy for the variant's own behaviour
    div = []
    for e in eps:
        a, b = e.chunk, e.shadow_chunks.get(variant)
        if a is not None and b is not None and a.shape == b.shape:
            div.append(float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-8)))
    res["action_div"] = float(np.mean(div)) if div else float("nan")
    if ok.any():
        a_ref = np.array([cp.first_alarm(mon.score(e)) is not None for e in np.array(eps)[ok]], float)
        a_sh = np.array([cp.first_alarm(mon.score(v)) is not None for v in np.array(views)[ok]], float)
        res.update(cmp_cols("fpr", compare(a_ref, a_sh, tasks[ok], margin=margin, worse="higher")))
    # D_AUROC = AUROC_ref - AUROC_shadow, both ranked against REFERENCE outcomes (drift diagnostic only)
    res.update(cmp_cols("auroc", compare(np.c_[m_ref, lab], np.c_[m_sh, lab], tasks, margin=auroc_margin,
                                         worse="lower", stat=auroc_stat, B=boot)))
    return res


SAMPLER_KEYS = ("num_steps", "n_action_steps", "chunk_size", "alt_samples", "noise_seeding")


def check_sampler(root, model, key, variants):
    """Flow-matching sampling must be identical across variants (denoising steps, query schedule, seeding),
    otherwise sampling variability is attributed to quantization. Compares run.json 'sampler' blocks."""
    def load(v):
        p = root / model / v / key / "run.json"
        return json.loads(p.read_text()).get("sampler") if p.exists() else None
    ref = load("fp16")
    bad = [v for v in variants if ref is not None and load(v) is not None and load(v) != ref]
    if bad:
        log.error("sampler settings differ from fp16 for %s -- comparisons confounded", bad)
    return not bad


def emit_learning_curve(emit, base0, mname, taps, device, sp, exp):
    """Reference test AUROC of a supervised probe trained with k failures (all training successes kept)."""
    fails = [e for e in sp["train"] if not e.success]
    succ = [e for e in sp["train"] if e.success]
    rng = np.random.default_rng(exp["seed"])
    order = rng.permutation(len(fails))
    lab = np.array([not e.success for e in sp["test"]], int)
    for k in [k for k in (5, 10, 20, 40, 80) if k < len(fails)] + [len(fails)]:
        m = M.build(mname, taps, device=device).fit(succ + [fails[i] for i in order[:k]])
        sc = np.array([float(np.max(m.score(e))) for e in sp["test"]])
        emit(dict(base0, status="ok", analysis="learning_curve", variant="fp16", fix="none",
                  n_train_failures=k, auroc=auroc(sc, lab)))


def reference_split(root, model, key, ref, exp):
    """Split reference episodes by their pre-registered trial roles (common.trial_roles)."""
    run = root / model / "fp16" / key / "run.json"
    meta = json.loads(run.read_text()) if run.exists() else {}
    R = int(meta.get("n_reference") or exp["reference_episodes_per_task"])
    split = meta.get("split") or exp["split"]
    roles = trial_roles(exp["seed"], sorted({tid(e) for e in ref}), R, split)
    # Monitor fitting uses ONLY the train slots. Reference runs of repair slots are kept apart
    # ("repair_ref", descriptive use only) so that calibration, repair and test slots never touch the
    # monitor: conditional on the fitted monitor, repair and test slots are then exchangeable by design.
    out = {"train": [], "calib": [], "test": [], "repair_ref": []}
    for e in ref:
        r = role_of(roles, tid(e), idx(e))
        out["repair_ref" if r == "repair" else r].append(e)
    return out, R, roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True, help="rollout folder, e.g. libero_10+shift_mild")
    ap.add_argument("--monitors", default=None)
    ap.add_argument("--taps", default=None, help="comma list of taps (default: all taps in models.yaml)")
    ap.add_argument("--rollouts", default=str(ROOT / "rollouts"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--learning_curve", action="store_true",
                    help="supervised probes: test AUROC vs number of training failures (Stage A)")
    a = ap.parse_args()
    setup_logging()
    models, exp = load_configs()
    mcfg = models.get(a.model, {})
    root = Path(a.rollouts)
    taps = a.taps.split(",") if a.taps else list(mcfg.get("tap", {}))
    mon_names = a.monitors.split(",") if a.monitors else exp["monitors"]
    cal = exp.get("calibration", {})
    cpkw = dict(guarantee=cal.get("guarantee", "marginal"), delta=cal.get("delta", 0.1),
                shape_frac=cal.get("shape_frac", 0.5), min_shape=cal.get("min_shape", 10),
                strict=cal.get("strict", False), seed=exp["seed"])
    st = exp.get("stats", {})
    margin, auroc_margin = st.get("fpr_margin", 0.05), st.get("auroc_margin", 0.05)
    success_margin, tpr_margin = st.get("success_margin", 0.05), st.get("tpr_margin", 0.10)
    onsets = load_onsets(ROOT / "results/onsets.csv", a.model, a.key)
    pair_rows: list = []
    trial_rows: list = []

    ref = load_cell(root, a.model, "fp16", a.key)
    if not ref:
        sys.exit(f"no reference rollouts in {root / a.model / 'fp16' / a.key}")
    variants = sorted(p.name for p in (root / a.model).iterdir()
                      if p.name != "fp16" and (p / a.key).exists()) if (root / a.model).exists() else []
    sp, R, roles = reference_split(root, a.model, a.key, ref, exp)
    sampler = check_sampler(root, a.model, a.key, variants)
    log.info("reference: %d eps (%d failures) -> train %d / calib %d / test %d (pre-registered roles)", len(ref),
             sum(not e.success for e in ref), len(sp["train"]), len(sp["calib"]), len(sp["test"]))
    shadow_vars = sorted(set().union(*[set(e.shadows()) for e in sp["test"]])) if sp["test"] else []

    var_eps = {}
    for v in variants:
        eps = load_cell(root, a.model, v, a.key)
        # test slots -> paired endpoint; repair slots (+ top-ups) -> repair pool. Both roles come from the
        # same pre-registered partition, so the repair pool is exchangeable with the variant's test trials.
        test = [e for e in eps if role_of(roles, tid(e), idx(e)) == "test"]
        rep_ = [e for e in eps if role_of(roles, tid(e), idx(e)) == "repair"]
        top = [e for e in eps if TOPUP_OFFSET <= idx(e) < EXT_OFFSET]
        ext = [e for e in eps if idx(e) >= EXT_OFFSET]   # Stage C detection extension: paired across rules only
        var_eps[v] = {"adapt": rep_ + top, "test": test, "ext": ext}
    rows = []
    emit = rows.append
    design = dict(sampler_consistent=sampler, calib_unit="pooled_over_tasks", population="successes of the calibrated policy; this suite's "
                  "task mix, LIBERO initial states and camera condition")

    for mname in mon_names:
        base0 = dict(model=a.model, key=a.key, monitor=mname)
        mbase, half = M.parse_name(mname)
        ok, why = M.applicable(mbase, mcfg) if mcfg else (True, "")
        if not ok:
            emit(dict(base0, status="not_applicable", reason=why))
            continue
        mon = M.build(mbase, taps, device=a.device)
        train_m = M.train_subset(sp["train"], half, exp["seed"])
        if not all(mon.available(e) for e in ref[:5]):
            emit(dict(base0, status="missing_data", reason="features/chunks/samples not recorded"))
            continue
        min_fail = exp.get("min_train_failures", {}).get(mbase, exp.get("min_train_failures", {}).get("default",
                                                                                                      MIN_FAIL_TRAIN))
        n_fail_train = sum(not e.success for e in train_m)
        if mon.needs_failures and n_fail_train < min_fail:
            emit(dict(base0, status="too_few_failures", reason=f"{n_fail_train} < {min_fail} independent training failures"))
            continue
        if mon.needs_failures and a.learning_curve:
            emit_learning_curve(emit, base0, mbase, taps, a.device, sp, exp)
        mon.fit(train_m)
        cal_succ = [e for e in sp["calib"] if e.success]
        cal_scores = scored(mon, cal_succ)

        for alpha in exp["alpha"]:
            need = required_successes(alpha, cpkw["shape_frac"], cpkw["guarantee"], cpkw["delta"], cpkw["min_shape"])
            if len(cal_succ) < 4:
                emit(dict(base0, alpha=alpha, status="too_few_successes", reason=f"{len(cal_succ)} calibration successes"))
                continue
            cp = FunctionalCP(alpha, **cpkw).fit(cal_scores)
            base = dict(base0, **design, alpha=alpha, status="ok", guarantee=cpkw["guarantee"],
                        n_cal_success=len(cal_succ), n_cal_success_required=need["total"],
                        n_shape=cp.n_shape, rank=None if cp.underpowered else need["rank"])
            emit(dict(base, variant="fp16", analysis="deployment", fix="none", n_adapt=0,
                      task_success=float(np.mean([e.success for e in sp["test"]])), **metrics_row(cp, mon, sp["test"])))

            for v in shadow_vars:
                r = obs_controlled_row(cp, mon, sp["test"], v, margin, auroc_margin)
                if r:
                    emit(dict(base, variant=v, analysis="obs_controlled", fix="none", n_adapt=0, **r))

            ref_succ = [e for e in sp["test"] if e.success]
            for v, parts in var_eps.items():
                adapt, test = parts["adapt"], parts["test"]
                if not test:
                    continue
                drift = feature_drift(sp["train"], adapt or test, mon.taps) if mon.taps else {}
                common = dict(base, variant=v, task_success=float(np.mean([e.success for e in test])),
                              n_adapt_pool=len(adapt), n_adapt_success=sum(e.success for e in adapt),
                              **{f"drift_{k}": x for k, x in drift.items()})
                row = dict(common, analysis="deployment", fix="none", n_adapt=0, **metrics_row(cp, mon, test, onsets=onsets))
                # Deployment comparisons on the SAME trial slots, resampled jointly by task (never two
                # independent bootstraps): each policy's FPR on its own successes, TPR on its own failures and
                # AUROC on all its trials, all computed on the same resampled set of initial conditions.
                T = trial_table(cp, mon, sp["test"], test)
                if T is not None:
                    rs, vs_, ra, va, rm, vm, cl = T
                    keys_ = [e.pair_key for e in test if e.pair_key in {r.pair_key for r in sp["test"]}]
                    trial_rows.extend(dict(model=a.model, key=a.key, monitor=mname, variant=v, alpha=alpha,
                                           task=int(k[0]), idx=int(k[1]), ref_success=int(x1), var_success=int(x2),
                                           ref_alarm=int(x3), var_alarm=int(x4))
                                      for k, x1, x2, x3, x4 in zip(keys_, rs, vs_, ra, va))
                for e in parts.get("ext", []):  # no reference run: ref_* = -1; used only for rule-vs-rule families
                    trial_rows.append(dict(model=a.model, key=a.key, monitor=mname, variant=v, alpha=alpha,
                                           task=tid(e), idx=idx(e), ref_success=-1, var_success=int(e.success),
                                           ref_alarm=-1, var_alarm=int(cp.first_alarm(mon.score(e)) is not None)))
                    row.update(cmp_cols("fpr", compare_joint(lambda i: masked_mean(ra, rs, i),
                                                             lambda i: masked_mean(va, vs_, i), cl,
                                                             margin=margin, worse="higher")))
                    if (~rs).any() and (~vs_).any():
                        row.update(cmp_cols("tpr", compare_joint(lambda i: masked_mean(ra, ~rs, i),
                                                                 lambda i: masked_mean(va, ~vs_, i), cl,
                                                                 margin=tpr_margin, worse="lower")))
                        row.update(cmp_cols("auroc", compare_joint(lambda i: auroc(rm[i], (~rs[i]).astype(int)),
                                                                   lambda i: auroc(vm[i], (~vs_[i]).astype(int)), cl,
                                                                   margin=auroc_margin, worse="lower", B=500)))
                emit(row)
                dump: list = []
                pr = paired_row(cp, mon, sp["test"], test, margin, success_margin, dump)
                if pr:
                    emit(dict(common, analysis="paired", fix="none", n_adapt=0, **pr))
                    pair_rows.extend(dict(model=a.model, key=a.key, monitor=mname, variant=v, alpha=alpha, **d)
                                     for d in dump)

                # ---- repairs (deployment analysis, variant test trials) ----
                # recalibration set = a SEEDED RANDOM subset of the realized repair-pool successes (never "the
                # first n"), drawn independently of scores, so it stays exchangeable with the test successes
                succ_pool = [e for e in adapt if e.success and idx(e) < TOPUP_OFFSET]
                succ_pool = [succ_pool[i] for i in np.random.default_rng([exp["seed"], 99]).permutation(len(succ_pool))]
                for n in exp["fixes"]["recal_success"]["n_rollouts"]:
                    if len(succ_pool) < n:
                        emit(dict(common, analysis="deployment", fix="recal_success", n_adapt=n, status="budget_not_met",
                                  reason=f"{len(succ_pool)} successes in repair pool < {n}"))
                        continue
                    used = succ_pool[:n]
                    cp2 = FunctionalCP(alpha, **cpkw).fit(scored(mon, used))
                    m_ts = sum(e.success for e in test)
                    fp = (finite_population_fpr(cp2.n_cal, m_ts, cp2.rank, alpha)
                          if cp2.rank is not None and m_ts > 0 else {})
                    emit(dict(common, analysis="deployment", fix="recal_success", n_adapt=n,
                              uses_topups=any(idx(e) >= TOPUP_OFFSET for e in used),
                              # exact finite-population model for THIS cell's test subset (m_ts successes):
                              # P(FPR_test > alpha) depends on m through the lattice k/m; reported, not claimed
                              finite_pop_m_test=m_ts, finite_pop_p_viol=fp.get("p_test_fpr_above_alpha"),
                              finite_pop_expected_fpr=fp.get("expected_test_fpr"),
                              **metrics_row(cp2, mon, test)))

                if mon.taps:
                    for n in exp["fixes"]["coral"]["n_rollouts"]:
                        if len(adapt) < n:
                            emit(dict(common, analysis="deployment", fix="coral", n_adapt=n, status="budget_not_met",
                                      reason=f"{len(adapt)} episodes in repair pool < {n}"))
                            continue
                        mon.feature_transform = DiagCoral(mon.taps).fit(sp["train"], adapt[:n])
                        emit(dict(common, analysis="deployment", fix="coral", n_adapt=n, **metrics_row(cp, mon, test)))
                        mon.feature_transform = None

                # fully supervised reference baseline (not an upper bound: retraining can underperform)
                pool = adapt[: exp["fixes"]["retrain_supervised"]["n_rollouts"]]
                halves = split_pooled(pool, {"train": 0.6, "calib": 0.4}, exp["seed"]) if pool else None
                if halves and (not mon.needs_failures or sum(not e.success for e in halves["train"]) >= MIN_FAIL_TRAIN):
                    m2 = M.build(mbase, taps, device=a.device).fit(halves["train"])
                    cs = [e for e in halves["calib"] if e.success]
                    if len(cs) >= 4:
                        cp3 = FunctionalCP(alpha, **cpkw).fit(scored(m2, cs))
                        emit(dict(common, analysis="deployment", fix="retrain_supervised", n_adapt=len(pool),
                                  **metrics_row(cp3, m2, test)))
        log.info("monitor %s done", mname)

    out = ROOT / "results/eval" / f"{a.model}__{a.key}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows -> %s", len(rows), out)
    if pair_rows:
        pp = ROOT / "results/eval_pairs" / f"{a.model}__{a.key}.csv"
        pp.parent.mkdir(parents=True, exist_ok=True)
        with pp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
            w.writeheader()
            w.writerows(pair_rows)
        log.info("wrote %d paired-trial rows -> %s", len(pair_rows), pp)
    if trial_rows:  # all test trials, both outcomes: used for the confirmatory rule-vs-rule contrasts
        tp = ROOT / "results/eval_trials" / f"{a.model}__{a.key}.csv"
        tp.parent.mkdir(parents=True, exist_ok=True)
        with tp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trial_rows[0]))
            w.writeheader()
            w.writerows(trial_rows)


if __name__ == "__main__":
    main()
