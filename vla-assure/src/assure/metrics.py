"""Metrics for a calibrated monitor on a set of test episodes."""
from __future__ import annotations

import numpy as np

from .conformal import FunctionalCP


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """labels: 1 = failure. Mann-Whitney U with tie handling. NaN if one class is missing."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(cp: FunctionalCP, score_seqs: list[np.ndarray], failed: list[bool],
             query_steps: list[np.ndarray], lengths: list[int], onsets: list | None = None) -> dict:
    labels = np.array(failed, int)
    margins = np.array([cp.margin(s) for s in score_seqs])
    alarms = [cp.first_alarm(s) for s in score_seqs]
    alarmed = np.array([a is not None for a in alarms])
    succ, fail = labels == 0, labels == 1
    fpr = float(alarmed[succ].mean()) if succ.any() else float("nan")
    tpr = float(alarmed[fail].mean()) if fail.any() else float("nan")
    # NORMALISED ALARM TIME: first alarm as a fraction of the failed episode's length. Not an actionable
    # lead time -- the episode end is a timeout, not the moment the failure became unrecoverable.
    t_det = [query_steps[i][a] / max(lengths[i], 1) for i, a in enumerate(alarms) if fail[i] and a is not None]
    # LEAD TIME to annotated failure onset (env steps; positive = alarm before onset), where labelled.
    lead, before = [], []
    for i, a in enumerate(alarms):
        o = onsets[i] if onsets else None
        if fail[i] and o is not None:
            step = query_steps[i][a] if a is not None else None
            before.append(step is not None and step <= o)
            if step is not None:
                lead.append(o - step)
    return {
        "n": int(len(labels)),
        "n_fail": int(fail.sum()),
        "auroc": auroc(margins, labels),
        "fpr": fpr,                                   # should be <= alpha if calibration still holds
        "fpr_excess": fpr - cp.alpha if succ.any() else float("nan"),
        "tpr": tpr,
        "balanced_acc": float(np.nanmean([1 - fpr, tpr])),
        "alarm_time_norm": float(np.mean(t_det)) if t_det else float("nan"),
        "n_onset_labelled": len(before),
        "frac_alarm_before_onset": float(np.mean(before)) if before else float("nan"),
        "lead_steps_median": float(np.median(lead)) if lead else float("nan"),
        "n_cal": cp.n_cal,
        "guaranteed_alpha": cp.guaranteed_alpha,
        "cp_underpowered": cp.underpowered,
    }


def _cluster_resample(rng, clusters: np.ndarray) -> np.ndarray:
    """Two-level resample: tasks with replacement, then episodes within each drawn task."""
    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    out = []
    for c in rng.choice(uniq, len(uniq), replace=True):
        m = members[c]
        out.append(rng.choice(m, len(m), replace=True))
    return np.concatenate(out) if out else np.array([], int)


def cluster_bootstrap(fn, clusters, B: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of fn(indices) under the task-clustered bootstrap. Episodes of one LIBERO task
    share a scene and goal, so they are correlated; resampling tasks propagates that."""
    clusters = np.asarray(clusters)
    if clusters.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = [v for v in (fn(_cluster_resample(rng, clusters)) for _ in range(B)) if v == v]
    if not vals:
        return float("nan"), float("nan")
    lo = (1 - level) / 2 * 100
    return float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo))


def compare(ref: np.ndarray, var: np.ndarray, clusters_ref, clusters_var=None, paired: bool = True,
            margin: float = 0.05, B: int = 2000, seed: int = 0, worse: str = "higher", stat=None) -> dict:
    """DESCRIPTIVE estimate and percentile cluster-bootstrap intervals for a degradation D (D > 0 = WORSE).

    NOTE: the bootstrap tail proportions returned as p_worse / p_noninf are NOT used as hypothesis-test
    p-values in any confirmatory claim. Confirmatory tests use confirmatory() (wild cluster bootstrap-t
    with the null imposed), on linear paired endpoints with >= 20 task clusters.

      worse="higher" (false-alarm rate):   D = stat(var) - stat(ref)
      worse="lower"  (AUROC, TPR, success): D = stat(ref) - stat(var)

    `stat` maps (values, index array) -> float; default is the mean of per-episode values. For AUROC
    pass values as (score, label) rows and an AUROC stat. paired=True: ref[i], var[i] are the same
    trial (clusters_ref = its task) and are resampled jointly; paired=False: two task-clustered samples.
    TESTS = INVERSION OF ONE-SIDED PERCENTILE CLUSTER-BOOTSTRAP CONFIDENCE BOUNDS (no recentring):
      degraded      reject H0: D <= 0      at level a  iff  the lower one-sided (1-a) bound  > 0
                                                         iff  p_worse  = P*(D* <= 0)      < a
      non-inferior  reject H0: D >= margin at level a  iff  the upper one-sided (1-a) bound < margin
                                                         iff  p_noninf = P*(D* >= margin) < a
    The null boundary is the margin itself (or 0); the bound is compared with it directly. These p-values
    are the smallest level at which the bound excludes the boundary, and are what Holm adjusts.
    Reported: delta_lo/delta_hi = TWO-SIDED 95% percentile interval (descriptive only);
    delta_lower95_1s / delta_upper95_1s = ONE-SIDED 95% bounds (the 5th / 95th percentiles) used by
    the tests. They are not interchangeable. The same procedure is used in scripts/power_analysis.py,
    which also estimates its type-I error at the boundary.
    """
    ref, var = np.asarray(ref, float), np.asarray(var, float)
    sign = 1.0 if worse == "higher" else -1.0
    stat = stat or (lambda x, i: float(x[i].mean()))
    rng = np.random.default_rng(seed)
    if paired:
        cl = np.asarray(clusters_ref)
        all_i = np.arange(len(ref))
        est = sign * (stat(var, all_i) - stat(ref, all_i)) if len(ref) else float("nan")
        boots = []
        for _ in range(B):
            i = _cluster_resample(rng, cl)
            if len(i):
                boots.append(sign * (stat(var, i) - stat(ref, i)))
    else:
        cr, cv = np.asarray(clusters_ref), np.asarray(clusters_var)
        est = (sign * (stat(var, np.arange(len(var))) - stat(ref, np.arange(len(ref))))
               if len(ref) and len(var) else float("nan"))
        boots = []
        for _ in range(B):
            i, j = _cluster_resample(rng, cr), _cluster_resample(rng, cv)
            if len(i) and len(j):
                boots.append(sign * (stat(var, j) - stat(ref, i)))
    boots = np.asarray([b for b in boots if b == b])
    if est != est or not len(boots):
        return {"delta": float("nan")}
    return {
        "delta": float(est),
        "delta_lo": float(np.percentile(boots, 2.5)), "delta_hi": float(np.percentile(boots, 97.5)),
        "delta_upper95_1s": float(np.percentile(boots, 95)),
        "delta_lower95_1s": float(np.percentile(boots, 5)),
        "p_worse": float((np.sum(boots <= 0) + 1) / (len(boots) + 1)),
        "p_noninf": float((np.sum(boots >= margin) + 1) / (len(boots) + 1)),
        "margin": margin, "worse": worse, "n_units": int(len(ref) if paired else len(ref) + len(var)),
        "paired": paired,
    }


def compare_joint(fn_ref, fn_var, clusters, margin: float = 0.05, B: int = 2000, seed: int = 0,
                  worse: str = "higher") -> dict:
    """Like compare(), for statistics of TWO policies computed on the SAME resampled trials.

    `clusters` gives the task (or suite:task) of each trial slot; fn_ref(idx) / fn_var(idx) compute
    each policy's statistic on the resampled trial indices (e.g. FPR on that policy's successes among
    those trials). Task clusters are resampled once and shared by both policies, so the dependence
    between reference and variant on the same initial states is preserved -- never bootstrap them
    independently. Tests as in compare().
    """
    sign = 1.0 if worse == "higher" else -1.0
    cl = np.asarray(clusters)
    allv = np.arange(len(cl))
    est = sign * (fn_var(allv) - fn_ref(allv))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        i = _cluster_resample(rng, cl)
        v = sign * (fn_var(i) - fn_ref(i))
        if v == v:
            boots.append(v)
    boots = np.asarray(boots)
    if est != est or not len(boots):
        return {"delta": float("nan")}
    return {
        "delta": float(est),
        "delta_lo": float(np.percentile(boots, 2.5)), "delta_hi": float(np.percentile(boots, 97.5)),
        "delta_upper95_1s": float(np.percentile(boots, 95)), "delta_lower95_1s": float(np.percentile(boots, 5)),
        "p_worse": float((np.sum(boots <= 0) + 1) / (len(boots) + 1)),
        "p_noninf": float((np.sum(boots >= margin) + 1) / (len(boots) + 1)),
        "margin": margin, "worse": worse, "n_units": int(len(cl)), "paired": True,
    }


def masked_mean(x: np.ndarray, mask: np.ndarray, i: np.ndarray) -> float:
    m = mask[i]
    return float(x[i][m].mean()) if m.any() else float("nan")


def auroc_stat(x: np.ndarray, i: np.ndarray) -> float:
    """For compare(stat=...): x has rows (score, label)."""
    return auroc(x[i, 0], x[i, 1])


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (strong family-wise error control); NaNs pass through."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def verdict(p_worse_adj: float, p_noninf_adj: float, level: float = 0.05, level_noninf: float | None = None) -> str:
    """level for the degradation test; level_noninf (default = level) for the non-inferiority test.
    The confirmatory analysis uses 0.05 / 0.03: the non-inferiority level is size-calibrated by simulation
    (scripts/calibrate_level.py) because the WCR test is slightly liberal at the margin with 20 clusters."""
    ln = level if level_noninf is None else level_noninf
    if p_worse_adj == p_worse_adj and p_worse_adj < level:
        return "degraded"
    if p_noninf_adj == p_noninf_adj and p_noninf_adj < ln:
        return "non-inferior"
    return "inconclusive"


def bootstrap(fn, n_items: int, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of fn(indices) over i.i.d. episode resamples (kept for unclustered use)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        v = fn(rng.integers(0, n_items, n_items))
        if v == v:  # drop NaN
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- confirmatory test: wild cluster bootstrap-t
_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _cr_t(d: np.ndarray, g: np.ndarray, uniq: np.ndarray, mu0: float) -> tuple[float, float]:
    """Mean, cluster-robust (CR1) standard error and t statistic for H0: mean = mu0."""
    n, G = len(d), len(uniq)
    m = d.mean()
    s = np.array([(d[g == c] - m).sum() for c in uniq])
    se = np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n
    return m, (m - mu0) / max(se, 1e-12)


def wild_cluster_test(d, clusters, null: float, alternative: str, B: int = 9999, seed: int = 0) -> dict:
    """One-sided test of the MEAN of per-trial differences d with few clusters.

    Wild cluster restricted (WCR) bootstrap-t with Webb six-point weights (Cameron, Gelbach & Miller
    2008; MacKinnon & Webb 2017), the standard choice when there are only 10-20 clusters:
      1. t = (mean(d) - null) / SE_CR1(d)
      2. impose the null: residuals e_i = d_i - null
      3. for b = 1..B: draw one weight w_c per CLUSTER, d*_i = null + w_c e_i, recompute t*
      4. alternative="less"    (H0: mean >= null; e.g. non-inferiority at the margin): p = P(t* <= t)
         alternative="greater" (H0: mean <= null; e.g. degradation above 0):          p = P(t* >= t)
    Clusters carry all trials of a task together. Valid for linear statistics (means of paired
    differences) -- the confirmatory endpoints; ratio statistics and AUROC are reported descriptively.
    """
    d = np.asarray(d, float)
    g = np.asarray(clusters)
    uniq = np.unique(g)
    if len(d) < 3 or len(uniq) < 3:
        return {"p": float("nan"), "t": float("nan"), "n_clusters": int(len(uniq))}
    m, t = _cr_t(d, g, uniq, null)
    n, G = len(d), len(uniq)
    idx = np.searchsorted(uniq, g)
    E = np.bincount(idx, weights=d - null, minlength=G)      # cluster sums of null-imposed residuals
    nc = np.bincount(idx, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    W = _WEBB[rng.integers(0, 6, size=(B, G))]
    ms = null + (W * E).sum(1) / n                           # bootstrap means
    S = W * E - nc * (ms - null)[:, None]                    # bootstrap cluster sums of (d* - m*)
    se = np.sqrt(G / (G - 1) * (S ** 2).sum(1)) / n
    ts = (ms - null) / np.maximum(se, 1e-12)
    p = float((np.sum(ts <= t) + 1) / (B + 1)) if alternative == "less" else float((np.sum(ts >= t) + 1) / (B + 1))
    return {"p": p, "t": float(t), "mean": float(m), "n_clusters": int(G)}


def confirmatory(d, clusters, margin: float, B: int = 9999, seed: int = 0) -> dict:
    """Both one-sided tests for a paired degradation D (per-trial differences d, D > 0 = worse)."""
    deg = wild_cluster_test(d, clusters, 0.0, "greater", B, seed)
    ni = wild_cluster_test(d, clusters, margin, "less", B, seed + 1)
    g = np.asarray(clusters)
    uniq = np.unique(g)
    out = {"delta": float(np.mean(d)) if len(d) else float("nan"), "p_worse": deg["p"], "p_noninf": ni["p"],
           "margin": margin, "n_units": int(len(d)), "n_clusters": int(len(uniq)), "test": "WCR bootstrap-t (Webb)"}
    if len(uniq) >= 3:  # descriptive cluster-robust interval, t with G-1 df
        from scipy.stats import t as tdist
        m, _ = _cr_t(np.asarray(d, float), g, uniq, 0.0)
        s = np.array([(np.asarray(d, float)[g == c] - m).sum() for c in uniq])
        se = np.sqrt(len(uniq) / (len(uniq) - 1) * np.sum(s ** 2)) / len(d)
        q = tdist.ppf(0.975, len(uniq) - 1)
        out.update(delta_lo=float(m - q * se), delta_hi=float(m + q * se))
    return out
