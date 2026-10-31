"""Aggregate run.json + results/eval/*.csv into paper-ready tables and figures.

  python scripts/04_report.py --alpha 0.1

Outputs results/report.md, results/tables/*.csv, results/figs/*.png
Tables: task success, applicability, false alarms by analysis (deployment / paired / obs_controlled),
variant-vs-reference comparisons with Holm-adjusted verdicts, AUROC, repairs vs budget, real-kernel
efficiency, mixed-precision Pareto.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assure.common import ROOT, load_configs  # noqa: E402
from assure.metrics import confirmatory, holm, verdict  # noqa: E402


def run_table(rollouts: Path) -> pd.DataFrame:
    rows = []
    for p in rollouts.glob("*/*/*/run.json"):
        r = json.loads(p.read_text())
        rows.append(dict(model=r.get("model"), variant=r.get("variant"), key=p.parent.name,
                         episodes=r.get("episodes"), success_rate=r.get("success_rate"),
                         mem_ratio=r.get("nominal_linear_weight_mem_ratio_vs_fp16", 1.0),
                         protect_rule=r.get("protect_rule"), protect_frac=r.get("protect_frac"),
                         protected_frac_actual=r.get("protected_frac_actual"), filler_share=r.get("filler_share"),
                         peak_gib=r.get("peak_gpu_mem_gib"), latency_s=r.get("query_latency_median_s"),
                         shadows=",".join(r.get("shadow_names", []) or []), shadow_only=False,
                         seconds=r.get("seconds")))
        for name, srep in (r.get("shadows") or {}).items():  # shadow copies: memory + rule, no task success
            rows.append(dict(model=r.get("model"), variant=name, key=p.parent.name, episodes=None, success_rate=None,
                             mem_ratio=srep.get("nominal_linear_weight_mem_ratio_vs_fp16", 1.0),
                             protect_rule=srep.get("protect_rule"), protect_frac=srep.get("protect_frac"),
                             protected_frac_actual=srep.get("protected_frac_actual"),
                             filler_share=srep.get("filler_share"),
                             peak_gib=None, latency_s=None, shadows="", shadow_only=True))
    return pd.DataFrame(rows)


def md(df) -> str:
    return df.round(3).to_markdown() if len(df) else "_no data_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rollouts", default=str(ROOT / "rollouts"))
    a = ap.parse_args()
    res = ROOT / "results"
    (res / "tables").mkdir(parents=True, exist_ok=True)
    (res / "figs").mkdir(parents=True, exist_ok=True)
    out = ["# Results (auto-generated)", "",
           "False-alarm guarantees are MARGINAL (or PAC if configured) bounds on successful episodes only; "
           "they are not detection or safety guarantees.", ""]

    rt = run_table(Path(a.rollouts))
    if not rt.empty:
        ts = rt[~rt.shadow_only].pivot_table(index=["model", "key"], columns="variant", values="success_rate")
        ts.to_csv(res / "tables/task_success.csv")
        out += ["## Task success by variant (all executed trials of the cell)", "", md(ts), ""]
        th = rt[~rt.shadow_only & rt.seconds.notna() & (rt.episodes.fillna(0) > 0)].copy()
        if not th.empty:
            th["episodes_per_gpu_hour"] = th.episodes / (th.seconds / 3600)
            th = th.groupby(["model", "variant", "shadows"])[["episodes", "seconds", "episodes_per_gpu_hour"]].sum(
                min_count=1)
            th["episodes_per_gpu_hour"] = th.episodes / (th.seconds / 3600)
            th.to_csv(res / "tables/throughput.csv")
            out += ["## Measured throughput (replaces the planning estimates for later stages)", "", md(th), ""]
    eff_rows = []
    for p in sorted((res / "efficiency").glob("*.json")):
        e = json.loads(p.read_text())
        eff_rows += [dict(model=e["model"], variant=v, **m) for v, m in e["variants"].items()]
    if eff_rows:
        eff = pd.DataFrame(eff_rows).set_index(["model", "variant"])
        eff.to_csv(res / "tables/efficiency.csv")
        par_files = sorted((res / "parity").glob("*.json"))
        if par_files:
            pj = {}
            for pf in par_files:
                js = json.loads(pf.read_text())
                pj.update({f"tao4p-{r}-f{pf.stem.split('_f')[-1]}": v["pass"] for r, v in js["results"].items()})
            eff["parity_pass"] = [pj.get(v, np.nan) if str(v).startswith("tao4p") else np.nan
                                  for v in eff.index.get_level_values("variant")]
        out += ["## End-to-end efficiency, real kernels only (06_efficiency.py; no taps, no shadows)",
                "Mixed-precision rows count only if parity_pass is True (07_parity.py, pre-registered criteria).",
                "", md(eff), ""]

    files = sorted((res / "eval").glob("*.csv"))
    if not files:
        (res / "report.md").write_text("\n".join(out))
        print("no eval CSVs yet")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ("status", "reason", "analysis"):
        if col not in df:
            df[col] = "ok" if col == "status" else ""

    na = df[df.status != "ok"][["model", "monitor", "status", "reason"]].drop_duplicates()
    na.to_csv(res / "tables/applicability.csv", index=False)
    out += ["## Monitors not evaluated (and why)", "", md(na.set_index(["model", "monitor"])), ""]

    d = df[(df.status == "ok") & (df.alpha == a.alpha)]
    if "cp_underpowered" in d:
        n_up = int(d.cp_underpowered.fillna(False).astype(bool).sum())
        out += [f"Rows with an underpowered band at alpha={a.alpha}: {n_up} (excluded from claims).", ""]
        d = d[~d.cp_underpowered.fillna(False).astype(bool)]

    nf = d[d.fix == "none"]
    for analysis in ["deployment", "paired", "obs_controlled"]:
        g = nf[nf.analysis == analysis]
        if g.empty:
            continue
        t = g.pivot_table(index=["model", "key", "monitor"], columns="variant", values="fpr")
        t.to_csv(res / f"tables/fpr_{analysis}.csv")
        title = ("alarm rate on REFERENCE-success observations through the shadow model (labels = reference "
                 "outcomes; a representation-drift diagnostic, not the variant's executed false-alarm rate)"
                 if analysis == "obs_controlled" else f"false-alarm rate on successes, {analysis}")
        out += [f"## {title}, no repair (target {a.alpha})", "", md(t), ""]
    g = nf[nf.analysis == "paired"]
    if not g.empty:
        cols = ["fpr_ref_same_trials", "fpr", "tpr_induced", "alarm_time_shift_both_fail", "n_both_success",
                "n_induced_fail", "n_rescued"]
        t = g.groupby(["model", "key", "monitor", "variant"])[[c for c in cols if c in g]].mean()
        t.to_csv(res / "tables/paired_detail.csv")
        out += ["## Paired trials: same trials, reference vs variant", "", md(t), ""]
    t = nf[nf.analysis == "deployment"].pivot_table(index=["model", "key", "monitor"], columns="variant", values="auroc")
    t.to_csv(res / "tables/auroc.csv")
    out += ["## Episode AUROC on executed trajectories (deployment)",
            "Computed on ALL executed test trials of each policy (successes and failures). Never computed on "
            "jointly successful pairs, which contain no failures by construction.", "", md(t), ""]
    g = nf[nf.analysis == "obs_controlled"]
    if not g.empty:
        cols = [c for c in ["auroc", "cmp_auroc_delta", "cmp_auroc_delta_lo", "cmp_auroc_delta_hi", "score_shift_sd",
                            "action_div"] if c in g]
        t = g.groupby(["model", "key", "monitor", "variant"])[cols].mean()
        t.to_csv(res / "tables/obs_controlled.csv")
        out += ["## Observation-controlled diagnostics (labels = reference outcomes)",
                "auroc: shadow scores ranked against the REFERENCE policy's outcomes; score_shift_sd: mean shift of "
                "the alarm statistic on reference successes in reference-SD units; action_div: relative difference "
                "between shadow and executed action chunks (large = weaker proxy for the variant's own behaviour).",
                "", md(t), ""]

    # ---- secondary comparisons per cell: DESCRIPTIVE only (estimates + percentile intervals, no tests) ----
    _, exp = load_configs()
    st = exp.get("stats", {})
    lvl, lvl_ni = st.get("level_degraded", 0.0375), st.get("level_noninf", 0.03)
    min_g = st.get("min_clusters_confirmatory", 20)
    cmp_rows = []
    fams = [("paired", "fpr"), ("paired", "success"), ("deployment", "fpr"), ("deployment", "tpr"),
            ("deployment", "auroc"), ("obs_controlled", "fpr"), ("obs_controlled", "auroc")]
    for analysis, metric in fams:
        g = nf[(nf.analysis == analysis) & (nf.variant != "fp16")]
        col = f"cmp_{metric}_delta"
        if g.empty or col not in g:
            continue
        g = g[g[col].notna()].copy()
        g["analysis_metric"] = f"{analysis}:{metric}"
        keep = ["analysis_metric", "model", "key", "monitor", "variant", col, f"cmp_{metric}_delta_lo",
                f"cmp_{metric}_delta_hi", f"cmp_{metric}_margin"]
        cmp_rows.append(g[keep].rename(columns=lambda c: c.replace(f"cmp_{metric}_", "")))
    if cmp_rows:
        cm = pd.concat(cmp_rows, ignore_index=True)
        cm.to_csv(res / "tables/comparisons.csv", index=False)
        out += ["## Secondary comparisons per cell (DESCRIPTIVE): degradation D (D > 0 = worse)",
                "D_FPR = FPR_var - FPR_ref; D_TPR, D_AUROC, D_success = ref - var. Estimates with two-sided 95% "
                "percentile cluster-bootstrap intervals; no p-values or verdicts (10 clusters per cell).", "",
                md(cm.set_index(["analysis_metric", "model", "monitor", "variant"])), ""]

    # ---- PRIMARY endpoint: paired D_FPR pooled over each model's cells, clustered by (suite, task) ----
    pfiles = sorted((res / "eval_pairs").glob("*.csv"))
    if pfiles:
        st = exp.get("stats", {})
        pv, margin = st.get("primary_variant", "w4"), st.get("fpr_margin", 0.05)
        pdf = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True)
        pdf = pdf[np.isclose(pdf.alpha, a.alpha)]
        excl = st.get("primary_exclude_conditions", [])
        cond = pdf.key.str.split("+").str[1].fillna("nominal")
        pdf_secondary = pdf[cond.isin(excl)]
        pdf = pdf[~cond.isin(excl)]  # secondary conditions are reported per cell only
        prim = []
        for (model, mon, var), g in pdf.groupby(["model", "monitor", "variant"]):
            cl = (g.key.str.split("+").str[0] + ":" + g.task.astype(str)).values
            # CONFIRMATORY: wild cluster restricted bootstrap-t on the per-trial paired differences
            c = confirmatory((g.var_alarm - g.ref_alarm).values, cl, margin, B=st.get("wcr_B", 9999))
            prim.append(dict(model=model, monitor=mon, variant=var, cells=g.key.nunique(), n_pairs=len(g),
                             fpr_ref=g.ref_alarm.mean(), fpr_var=g.var_alarm.mean(), **c))
        pr = pd.DataFrame(prim)
        # heterogeneity: a pooled result can hide a degradation concentrated in one suite/condition
        cell_rows = []
        for (model, mon, var, key), g in pd.concat([pdf, pdf_secondary]).groupby(["model", "monitor", "variant", "key"]):
            c = confirmatory((g.var_alarm - g.ref_alarm).values, g.task.values, margin, B=1999)  # interval only
            cell_rows.append(dict(model=model, monitor=mon, variant=var, key=key, n_pairs=len(g), **c))
        pc = pd.DataFrame(cell_rows)
        if not pc.empty and "delta_lo" in pc:
            pc["cell_degraded"] = (pc.delta_lo > 0) & (pc.delta >= margin)
            pc["secondary_condition"] = pc.key.str.split("+").str[1].fillna("nominal").isin(excl)
            pc.to_csv(res / "tables/primary_by_cell.csv", index=False)
            bad = pc[pc.cell_degraded & ~pc.secondary_condition].groupby(["model", "monitor", "variant"]).key.apply(lambda x: ",".join(x))
            pr = pr.merge(bad.rename("cells_degraded").reset_index(), how="left", on=["model", "monitor", "variant"])
        if not pr.empty and "p_worse" in pr:
            pr["family"] = np.where((pr.variant == pv) & ~pr.monitor.str.startswith("ext_"), "primary", "secondary")
            for (fam, model), idx in pr.groupby(["family", "model"]).groups.items():
                pr.loc[idx, "p_worse_holm"] = holm(pr.loc[idx, "p_worse"].values)
                pr.loc[idx, "p_noninf_holm"] = holm(pr.loc[idx, "p_noninf"].values)
            pr["verdict"] = [verdict(x, y, lvl, lvl_ni) if gc >= min_g else f"descriptive only (<{min_g} clusters)"
                             for x, y, gc in zip(pr.p_worse_holm, pr.p_noninf_holm, pr.n_clusters)]
            if "cells_degraded" in pr:  # a pooled non-inferiority claim is withheld if any cell is clearly degraded
                het = pr.cells_degraded.notna() & (pr.verdict == "non-inferior")
                pr.loc[het, "verdict"] = "non-inferior pooled, degraded in a cell"
            pr.to_csv(res / "tables/primary_endpoint.csv", index=False)
            out = out[:3] + [f"## PRIMARY endpoint: paired D_FPR ({pv} vs fp16), pooled over cells, Holm per model",
                             f"Margin {margin}. WCR bootstrap-t (Webb), levels: degradation {lvl}, non-inferiority "
                             f"{lvl_ni} (size-calibrated). Pairs = trials both policies solved; clustered by (suite, task).",
                             "",
                             md(pr.set_index(["family", "model", "monitor", "variant"])[
                                 [c for c in ["cells", "n_pairs", "fpr_ref", "fpr_var", "delta", "delta_lo", "delta_hi",
                                              "p_worse_holm", "p_noninf_holm", "verdict", "cells_degraded"] if c in pr]]),
                             "", "Per-cell estimates: tables/primary_by_cell.csv", ""] + out[3:]

    fx = d[(d.analysis == "deployment") & (d.variant != "fp16")]
    fx = fx.groupby(["model", "monitor", "variant", "fix", "n_adapt"])[
        [c for c in ["fpr", "fpr_lo", "fpr_hi", "tpr", "auroc", "guaranteed_alpha"] if c in fx]].mean()
    fx.to_csv(res / "tables/repairs.csv")
    out += ["## Repairs vs budget (deployment)", "", md(fx), ""]

    # Figure: executed (deployment, paired) vs observation-controlled false alarms per variant
    for model, g in nf[nf.analysis.isin(["deployment", "paired", "obs_controlled"])].groupby("model"):
        p = g.groupby(["variant", "analysis"]).fpr.mean().unstack()
        ax = p.plot(kind="bar", figsize=(8, 3.5), rot=0)
        ax.axhline(a.alpha, ls="--", lw=1, color="k")
        ax.set_ylabel("false-alarm rate on successes")
        ax.set_title(f"{model}: executed (deployment, paired) vs observation-controlled")
        plt.tight_layout()
        plt.savefig(res / f"figs/fpr_{model}.png", dpi=200)
        plt.close()

    # Pareto: memory vs task success vs false alarms for mixed-precision variants
    if not rt.empty:
        mp = rt[rt.protect_rule.notna()]
        cf = nf[(nf.analysis == "obs_controlled")].groupby(["model", "variant"]).fpr.mean().rename("oc_fpr")
        dep = nf[(nf.analysis == "deployment")].groupby(["model", "variant"]).fpr.mean().rename("dep_fpr")
        det = nf[(nf.analysis == "deployment")].groupby(["model", "variant"])[["tpr", "auroc"]].mean().add_prefix("dep_")
        par = (rt.sort_values("shadow_only").drop_duplicates(["model", "variant"])
               .set_index(["model", "variant"])[["mem_ratio", "success_rate", "protect_rule", "protect_frac",
                                                 "protected_frac_actual", "filler_share"]])
        par = par.join(cf, how="outer").join(dep, how="outer").join(det, how="outer").reset_index()
        from assure.variants import BUDGET_TOLERANCE
        par["budget_ok"] = (par.protect_frac.isna() |
                            ((par.protected_frac_actual >= par.protect_frac - BUDGET_TOLERANCE) &
                             (par.protected_frac_actual <= par.protect_frac + 1e-9)))
        # a rule that lowers false alarms by suppressing scores also loses detection: show both
        par.to_csv(res / "tables/pareto.csv", index=False)
        if not mp.empty:
            for model, g in par[par.protect_rule.notna()].groupby("model"):
                fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
                for rule, h in g.groupby("protect_rule"):
                    axs[0].plot(h.mem_ratio, h.success_rate, "o-", label=rule)
                    axs[1].plot(h.mem_ratio, h.dep_fpr.fillna(h.oc_fpr), "o-", label=rule)
                axs[0].set(xlabel="Linear-weight memory vs fp16", ylabel="task success")
                axs[1].set(xlabel="Linear-weight memory vs fp16",
                           ylabel="false-alarm rate (deployed; obs-controlled if not deployed)")
                axs[1].axhline(a.alpha, ls="--", lw=1, color="k")
                axs[1].legend(fontsize=7)
                fig.suptitle(f"{model}: mixed precision at matched memory")
                plt.tight_layout()
                plt.savefig(res / f"figs/pareto_{model}.png", dpi=200)
                plt.close()

    # ---- RQ3 confirmatory families (pre-registered multiplicity map; see experiment.yaml: families) ----
    # F2 method (selection monitor #A), F3 transfer (independent replicate #B), F4 real-kernel validation:
    # proposed rule (margin_drift) vs each baseline at the PRIMARY budget only, pooled over both suites
    # (20 task clusters). "Better than baseline b" is an intersection-union test of three WCR tests, each at
    # its own level (no multiplicity inflation for requiring all three): (i) FPR superiority on trials both
    # variants solved, (ii) TPR non-inferiority (margin 0.10) on trials both variants failed, (iii) task
    # success non-inferiority (margin 0.05) on all paired test trials. Holm over baselines within a family,
    # applied to the IUT p-value (the max of the three). Every other budget, rule, suite split and monitor
    # family is DESCRIPTIVE.
    tfiles = sorted((res / "eval_trials").glob("*.csv"))
    fam_cfg = exp.get("families", {})
    prop = fam_cfg.get("proposed", "margin_drift")
    pct0 = str(fam_cfg.get("primary_budget_pct", 10))
    sel_mon = exp.get("stages", {}).get("C", {}).get("sensitivity_monitor", "mahalanobis#A")
    fams_def = {"F2 method": (sel_mon, "w4p"), "F3 transfer": (sel_mon.split("#")[0] + "#B", "w4p"),
                "F4 real-kernel": (sel_mon, "tao4p")}
    if tfiles:
        tr = pd.concat([pd.read_csv(f) for f in tfiles], ignore_index=True)
        tr = tr[np.isclose(tr.alpha, a.alpha)]
        m_ = tr.variant.str.extract(r"^(?P<kind>w4p|tao4p)-(?P<rule>[a-z_]+)-f(?P<pct>\d+)$")
        tr = pd.concat([tr, m_], axis=1).dropna(subset=["rule"])
        rows_ = []
        for fam, (mon_, kind) in fams_def.items():
            g = tr[(tr.monitor == mon_) & (tr.kind == kind) & (tr.pct == pct0)]
            if g.empty or prop not in set(g.rule):
                continue
            det = fam_cfg.get("detection", {})
            lvl_tpr = det.get("level_tpr_noninf", 0.02)
            sizing_f = res / "power/stage_c_sizing.json"
            n_min = (json.loads(sizing_f.read_text()).get("n_min") if sizing_f.exists() else None) or det.get("n_min_floor", 90)
            g_min = det.get("min_clusters_with_failures", 12)
            conf_base = set(fam_cfg.get("confirmatory_baselines", []))
            for model, gm in g.groupby("model"):
                gp = gm[gm.rule == prop]
                bases = sorted((set(gm.rule) - {prop}) & conf_base if conf_base else set(gm.rule) - {prop})
                if fam.startswith("F4"):
                    pj = res / "parity" / f"{model}_f{pct0}.json"
                    pj = json.loads(pj.read_text()) if pj.exists() else {}
                    f4b = pj.get("f4_baseline")
                    if not f4b:
                        rows_.append(dict(family=fam, model=model, monitor=mon_, budget_pct=int(pct0), proposed=prop,
                                          baseline=None, n_clusters=np.nan, detection_feasible=False, p_iut=np.nan,
                                          note="F4 not confirmable: parity file missing, proposed fails parity, "
                                               "or no pre-specified baseline passes"))
                        continue
                    bases = [b for b in bases if b == f4b] or [f4b]
                for base_rule in bases:
                    gb = gm[gm.rule == base_rule]
                    mm = gp.merge(gb, on=["key", "task", "idx"], suffixes=("_p", "_b"))
                    if len(mm) < 10:
                        continue
                    cl = (mm.key.str.split("+").str[0] + ":" + mm.task.astype(str)).values
                    # Pre-registered method-comparison population for ALL three components of F2-F4: the
                    # paired test slots plus the extension trials (fresh seeds, LIBERO initial states redrawn
                    # uniformly). Rule-vs-rule comparisons do not use the conformal guarantee, so they need no
                    # finite-population argument; clustering by (suite, task) absorbs reused initial states.
                    # `orig` only splits the reported counts.
                    orig = (mm.idx < 20_000).values
                    both_s = (mm.var_success_p == 1) & (mm.var_success_b == 1)
                    both_f = (mm.var_success_p == 0) & (mm.var_success_b == 0)
                    B_ = st.get("wcr_B", 9999)
                    # (i) FPR: d = alarm_p - alarm_b on joint successes; H0 d >= 0 (proposed not better)
                    t1 = confirmatory((mm.var_alarm_p - mm.var_alarm_b)[both_s].values, cl[both_s], 0.0, B=B_)
                    # (ii) TPR: D = alarm_b - alarm_p on joint failures; non-inferiority at 0.10
                    g_f = len(set(cl[both_f]))
                    det_ok = bool(both_f.sum() >= n_min and g_f >= g_min)   # pre-registered detection feasibility
                    t2 = confirmatory((mm.var_alarm_b - mm.var_alarm_p)[both_f].values, cl[both_f],
                                      st.get("tpr_margin", 0.10), B=B_) if both_f.sum() >= 10 else {"p_noninf": np.nan}
                    # (iii) success: D = s_b - s_p on all paired trials; non-inferiority at 0.05
                    t3 = confirmatory((mm.var_success_b - mm.var_success_p).values, cl,
                                      st.get("success_margin", 0.05), B=B_)
                    p_fpr, p_tpr, p_succ = t1.get("p_noninf"), t2.get("p_noninf"), t3.get("p_noninf")
                    # IUT with component levels (FPR sup lvl_sup, TPR NI lvl_tpr, success NI lvl_ni): each p is
                    # rescaled to the FPR-side level so one Holm step over baselines applies to all three
                    lvl_sup_ = st.get("level_degraded", 0.0375)
                    scaled = [p_fpr, p_tpr * lvl_sup_ / lvl_tpr if p_tpr == p_tpr else np.nan, p_succ * lvl_sup_ / lvl_ni]
                    rows_.append(dict(family=fam, model=model, monitor=mon_, budget_pct=int(pct0), proposed=prop,
                                      baseline=base_rule, n_trials=len(mm), n_test_slot_trials=int(orig.sum()),
                                      n_extension_trials=int((~orig).sum()),
                                      n_joint_failure_test_slots=int((both_f & orig).sum()), n_joint_success=int(both_s.sum()),
                                      n_joint_failure=int(both_f.sum()), n_min_joint_failure=n_min,
                                      clusters_with_joint_failure=g_f, detection_feasible=det_ok,
                                      tpr_diff=t2.get("delta"), tpr_diff_hi=t2.get("delta_hi"), n_clusters=t1.get("n_clusters"),
                                      executed_discordance=float((mm.var_alarm_b != mm.var_alarm_p)[both_f].mean()) if both_f.any() else np.nan,
                                      fpr_diff=t1.get("delta"), fpr_diff_lo=t1.get("delta_lo"), fpr_diff_hi=t1.get("delta_hi"),
                                      p_fpr_better=p_fpr, p_tpr_noninf=p_tpr, p_success_noninf=p_succ,
                                      p_iut=float(min(1.0, np.nanmax(scaled))) if not np.isnan(p_tpr) else np.nan))
        if rows_:
            cd = pd.DataFrame(rows_)
            for (fam, model), ix in cd.groupby(["family", "model"]).groups.items():
                # every pre-specified baseline counts in Holm's m: a missing component p (e.g. too few joint
                # failures) enters as p = 1, so dropping a comparison can never weaken the correction for the others
                cd.loc[ix, "p_iut_holm"] = holm(np.nan_to_num(cd.loc[ix, "p_iut"].values.astype(float), nan=1.0))
            lvl_sup = st.get("level_degraded", 0.0375)  # superiority components use the degradation-side level
            cd["claim"] = np.where(cd.n_clusters < min_g, f"descriptive only (<{min_g} clusters)",
                          np.where(~cd.detection_feasible.astype(bool), "detection not confirmable (claim withheld)",
                                   np.where(cd.p_iut_holm < lvl_sup, "proposed better (all three hold)",
                                            "no confirmatory evidence")))
            if "note" in cd:
                cd.loc[cd.note.notna(), "claim"] = cd.loc[cd.note.notna(), "note"]
            cd.to_csv(res / "tables/method_families.csv", index=False)
            out += ["## RQ3 confirmatory families: proposed rule vs baselines at the primary budget (IUT, Holm)",
                    "Claim requires FPR superiority AND TPR non-inferiority AND success non-inferiority. F3 = "
                    "independent monitor replicate (never used in selection); F4 = real int4 kernels.", "",
                    md(cd.set_index(["family", "model", "baseline"])), ""]
    # all other rule x budget x monitor contrasts: descriptive FPR differences on joint successes
    if tfiles:
        desc = []
        for (model, mon_, kind, pct), g in tr.groupby(["model", "monitor", "kind", "pct"]):
            gp = g[g.rule == prop]
            for base_rule in sorted(set(g.rule) - {prop}):
                mm = gp.merge(g[g.rule == base_rule], on=["key", "task", "idx"], suffixes=("_p", "_b"))
                js = (mm.var_success_p == 1) & (mm.var_success_b == 1)
                if js.sum() < 10:
                    continue
                c = confirmatory((mm.var_alarm_p - mm.var_alarm_b)[js].values,
                                 (mm.key.str.split("+").str[0] + ":" + mm.task.astype(str))[js].values, 0.0, B=1999)
                desc.append(dict(model=model, monitor=mon_, kind=kind, budget_pct=int(pct), baseline=base_rule,
                                 n_joint_success=int(js.sum()), fpr_diff=c["delta"], lo=c.get("delta_lo"), hi=c.get("delta_hi")))
        if desc:
            dd = pd.DataFrame(desc)
            dd.to_csv(res / "tables/method_contrasts_descriptive.csv", index=False)
            out += ["## All rule-vs-baseline FPR differences (DESCRIPTIVE; no tests)", "",
                    md(dd.set_index(["model", "monitor", "kind", "budget_pct", "baseline"])), ""]

    # ---- block-selection checks: restricted vs broad greedy, interactions, ranking stability ----
    for p in sorted((res / "sensitivity").glob("*.json")):
        sj = json.loads(p.read_text())
        if sj.get("validation"):
            v = pd.DataFrame(sj["validation"])
            v.to_csv(res / f"tables/greedy_check_{p.stem}.csv", index=False)
            out += [f"## Greedy interaction check ({p.stem}; search: {sj.get('greedy_search')})",
                    "Selection is an observation-controlled surrogate on reference rollouts (reference labels).", "",
                    md(v.set_index("budget_pct")), ""]
        if sj.get("stability"):
            v = pd.DataFrame(sj["stability"])
            v.to_csv(res / f"tables/ranking_stability_{p.stem}.csv", index=False)
            out += [f"## Ranking stability vs number of captured episodes ({p.stem})", "",
                    md(v.set_index(["episodes", "budget_pct"])), ""]

    (res / "report.md").write_text("\n".join(out))
    print("wrote", res / "report.md")


if __name__ == "__main__":
    main()
