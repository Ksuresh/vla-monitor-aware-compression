"""Per-block quantization sensitivity: quantize ONE block, replay captured episodes, measure drift.

Records, per transformer block:
  feature_drift  1 - cos(features) on the monitored taps                 -> feature-aware rule
  action_drift   relative L2 change of the predicted chunk                -> task-aware rule
  score_drift    mean |change in the monitor's score s_k| in units of the reference score spread
  margin_drift   J_b = E|M_b - M_ref| / sd(M_ref), with M = max_k (s_k - mu_k)/sd_k the exact statistic the
                 conformal threshold q is compared with. ABSOLUTE change, so upward and downward shifts
                 cannot cancel; averaged with equal weight over captured successes and failures
                 (J = (J_succ + J_fail)/2), so a block that disturbs only failure episodes still counts
                                                                          -> monitor-decision rule
  suppression    signed mean change of M on captured FAILURES (negative = alarms pushed down): a rule
                 that lowers false alarms by suppressing scores indiscriminately shows up here
  flip_rate      fraction of captured episodes whose alarm decision (M > q) flips
  timing_drift   mean |first-alarm query index (block quantized) - reference| / episode queries, over episodes
                 where either configuration alarms: same max score can still mean a different alarm time

SELECTION SET. Captured episodes are dedicated trials (ids 900000+, fresh seeds and perturbation draws)
outside the train / calib / repair / test partition; they share LIBERO's pool of 50 initial states per
task, which no split can avoid, so they count as model-selection data, never as validation evidence.

SELECTION vs EVALUATION. Captured episodes are rollouts of the REFERENCE policy, so success/failure labels
are reference outcomes and the quantized candidates only see reference observations: the ranking is an
observation-controlled SURROGATE whose target is "keep the reference monitor's decisions". It does not
claim better detection than the reference monitor. Whether a selected configuration keeps calibration,
detection and task success is decided only on its own executed trajectories (03_evaluate.py).
--stability 12,24,48 recomputes the ranking on nested subsets of the captured episodes and reports how
stable the selected sets are (Jaccard vs the largest) and the rank correlation of J_b.
Feature drift can be large along directions the monitor ignores and small along its decision direction;
score_drift / margin_drift measure the change the monitor actually sees. They need a monitor trained on
reference rollouts (--ref_key), fitted exactly as in 03_evaluate (same roles, same band).

(weight_error, edge and random rules need no data; see src/assure/variants.py.)
Writes results/sensitivity/<model>.json, read by the w4p-<rule>-f<pct> variants.

  python scripts/02_sensitivity.py --model pi05 --suite libero_10 --condition shift_mild \
      --ref_key libero_10+shift_mild --monitor mahalanobis --bits 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assure.adapters import make_session  # noqa: E402
from assure.common import ROOT, load_configs, setup_logging, write_json  # noqa: E402
from assure.quant import QuantSpec, apply_fake_quant, blocks_of  # noqa: E402

PER_QUERY_MONITORS = ("mahalanobis", "rnd", "safe_mlp", "safe_lstm")  # score from features only
SD_FLOOR = 0.05  # M and s are already in band-normalised / standardised units, so 0.05 is a small floor


class _Ep:
    """Minimal stand-in for LoadedEpisode: monitors only read .feats for these monitor types."""

    def __init__(self, feats_seq: list[dict]):
        keys = feats_seq[0].keys()
        self.feats = {k: np.stack([f[k] for f in feats_seq]) for k in keys}


def cos_drift(a: dict, b: dict) -> float:
    vals = []
    for k in a:
        x, y = a[k].ravel(), b[k].ravel()
        vals.append(1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    return float(np.mean(vals))


def load_monitor(model, key, monitor, taps, alpha, exp):
    """Train the monitor and band on reference rollouts with the same roles as 03_evaluate."""
    import importlib
    sys.path.insert(0, str(ROOT / "scripts"))
    ev = importlib.import_module("03_evaluate")
    from assure import monitors as M
    from assure.conformal import FunctionalCP
    from assure.store import load_cell
    ref = load_cell(ROOT / "rollouts", model, "fp16", key)
    if not ref:
        sys.exit(f"--ref_key {key}: no reference rollouts for {model}")
    sp, _, _ = ev.reference_split(ROOT / "rollouts", model, key, ref, exp)
    base, half = M.parse_name(monitor)
    if half and half.upper() == "B":
        sys.exit("monitor #B is the frozen independent replicate: block selection must never use it")
    if base not in PER_QUERY_MONITORS:
        sys.exit(f"--monitor {monitor}: block selection needs a per-query monitor {PER_QUERY_MONITORS}")
    mon = M.build(base, taps).fit(M.train_subset(sp["train"], half, exp["seed"]))
    cal = exp["calibration"]
    cp = FunctionalCP(alpha, guarantee=cal["guarantee"], delta=cal["delta"], shape_frac=cal["shape_frac"],
                      min_shape=cal.get("min_shape", 10), seed=exp["seed"]).fit(
        [mon.score(e) for e in sp["calib"] if e.success])
    return mon, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--condition", default="nominal")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=12,
                    help="captured episodes (disjoint from evaluation trials); run under the shifted condition so "
                         "some fail")
    ap.add_argument("--max_queries", type=int, default=60, help="captured queries per episode (cap)")
    ap.add_argument("--block_regex", default=r"layers\.(\d+)$")
    ap.add_argument("--ref_key", default=None, help="reference rollout folder to train the monitor on")
    ap.add_argument("--monitor", default="mahalanobis#A",
                    help="per-query monitor; '#A' = trained on half A of the training slots (half B is kept for "
                         "the independent-replicate test in 03_evaluate)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--greedy", action="store_true",
                    help="interaction check: greedy protection with J recomputed after each pick (needs --ref_key)")
    ap.add_argument("--greedy_budgets", default="5,10,20", help="protected %% of Linear params")
    ap.add_argument("--greedy_candidates", type=int, default=8,
                    help="RESTRICTED greedy: top-C blocks by independent J considered per step (0 = all blocks)")
    ap.add_argument("--greedy_random", type=int, default=2, help="extra random candidates per step")
    ap.add_argument("--select_episodes", type=int, default=24,
                    help="captured episodes used for ALL selection (block scores *_sel and the greedy search)")
    ap.add_argument("--holdout_episodes", type=int, default=24,
                    help="further captured episodes used ONLY to compare selection strategies (never for selection)")
    ap.add_argument("--stability", default="", help="nested captured-episode counts, e.g. 12,24,48 (exploratory)")
    ap.add_argument("--min_failures", type=int, default=8, help="per selection / held-out set (Stage C entry)")
    ap.add_argument("--min_successes", type=int, default=8)
    ap.add_argument("--max_captured", type=int, default=192)
    ap.add_argument("--rank_boot", type=int, default=500, help="bootstrap replicates for ranking uncertainty")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. _broad)")
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    stab = [int(x) for x in a.stability.split(",") if x]
    if stab:
        a.episodes = max(a.episodes, max(stab))
    if a.greedy:
        a.episodes = max(a.episodes, a.select_episodes + a.holdout_episodes)
    setup_logging()
    models, exp = load_configs()
    mcfg = models[a.model]
    s = make_session(a.model, mcfg, "fp16", exp["variants"]["fp16"], a.suite, device=a.device,
                     repo_root=a.repo_root, base_seed=exp["seed"], condition=a.condition,
                     condition_spec=exp["conditions"][a.condition])

    # Capture. Episode e goes to the SELECTION set if e is even and to the HELD-OUT set if odd (fixed by
    # index before any outcome is seen). Capture continues until each set has at least --select_episodes
    # episodes AND >= --min_failures failures AND >= --min_successes successes, up to --max_captured.
    # Stopping depends on outcome COUNTS only (never on scores); these episodes are model-selection data.
    episodes: list[list] = []
    outcome: list[bool] = []
    t_cap = time.time()

    def enough():
        o = np.array(outcome, bool)
        sets = [o[0::2], o[1::2]] if a.greedy else [o]
        return all(len(x) >= a.select_episodes and (~x).sum() >= a.min_failures and x.sum() >= a.min_successes
                   for x in sets)
    e = 0
    while e < a.max_captured and (e < a.episodes or not enough()):
        cap: list = []
        ep = s.rollout_episode(task_id=e % 10, episode_idx=900_000 + e, seed=10**6 + e, capture=cap,
                               cap_max=a.max_queries)
        episodes.append(cap)
        outcome.append(bool(ep.success))
        e += 1
    if not enough():
        print(f"WARNING: failure/success minimum not met after {e} captured episodes; Stage C entry criterion FAILS")
    if a.greedy:  # reorder: selection set first (even indices), then held-out (odd)
        order = list(range(0, len(episodes), 2)) + list(range(1, len(episodes), 2))
        a.select_episodes = len(range(0, len(episodes), 2))
        a.holdout_episodes = len(episodes) - a.select_episodes
        episodes = [episodes[i] for i in order]
        outcome = [outcome[i] for i in order]
    ok = np.array(outcome)
    capture_seconds = time.time() - t_cap
    print(f"captured {ok.sum()} successes / {(~ok).sum()} failures")
    ref = [[s.replay(c) for c in ep] for ep in episodes]  # per episode: list of (features, chunk)

    mon = cp = None
    if a.ref_key:
        mon, cp = load_monitor(a.model, a.ref_key, a.monitor, list(mcfg["tap"]), a.alpha, exp)

    def mon_stats(run):
        """Per-episode score sequences and alarm statistics for one replay pass."""
        eps = [_Ep([f for f, _ in ep]) for ep in run]
        S = [mon.score(x) for x in eps]
        Mx = np.array([cp.margin(x) for x in S])
        return S, Mx

    if mon is not None:
        S_ref, M_ref = mon_stats(ref)
        # ONE reference scale, fixed before any candidate is evaluated: the SD of M over ALL captured
        # reference episodes (successes and failures pooled), with a floor so a near-constant M cannot
        # blow up J. The same frozen fp16 monitor and band are used for every candidate; no recalibration.
        s_sd = max(float(np.std(np.concatenate(S_ref))), SD_FLOOR)
        m_sd = max(float(np.std(M_ref)) if len(M_ref) > 1 else 1.0, SD_FLOOR)
        alarm_ref = M_ref > cp.q
        K_ref = [len(x) for x in S_ref]
        k_ref = np.array([cp.first_alarm(x) if cp.first_alarm(x) is not None else len(x) for x in S_ref])

    blocks = blocks_of(s.module, a.block_regex)
    n_q = sum(len(ep) for ep in episodes)
    print(f"{len(blocks)} blocks, {len(episodes)} episodes, {n_q} captured queries")
    out, t0 = [], time.time()
    dsel: dict = {}
    for name in blocks:
        t_blk = time.time()
        blk = dict(s.module.named_modules())[name]
        backup = {k: v.detach().clone() for k, v in blk.state_dict().items()}
        apply_fake_quant(blk, QuantSpec(bits=a.bits))
        new = [[s.replay(c) for c in ep] for ep in episodes]
        blk.load_state_dict(backup)
        pairs = [(r, n) for er, en in zip(ref, new) for r, n in zip(er, en)]
        nsel = min(a.select_episodes, len(episodes))
        pairs_sel = [(r, n) for er, en in zip(ref[:nsel], new[:nsel]) for r, n in zip(er, en)]
        fd = lambda P: float(np.mean([cos_drift(r[0], n[0]) for r, n in P]))  # noqa: E731
        ad = lambda P: float(np.mean([np.linalg.norm(n[1] - r[1]) / (np.linalg.norm(r[1]) + 1e-8) for r, n in P]))  # noqa: E731
        row = {
            "name": name, "params": int(sum(p.numel() for p in blk.parameters())),
            "feature_drift": fd(pairs), "action_drift": ad(pairs),
            # *_sel: computed on the SELECTION episodes only; these are what every data rule ranks by
            "feature_drift_sel": fd(pairs_sel), "action_drift_sel": ad(pairs_sel),
        }
        if mon is not None:
            S_new, M_new = mon_stats(new)
            row["score_drift"] = float(np.mean([np.mean(np.abs(x - y)) for x, y in zip(S_new, S_ref)]) / s_sd)
            row["score_drift_sel"] = float(np.mean([np.mean(np.abs(x - y)) for x, y in
                                                    zip(S_new[:nsel], S_ref[:nsel])]) / s_sd)
            d = (M_new - M_ref) / m_sd
            parts = [np.mean(np.abs(d[g])) for g in (ok, ~ok) if g.any()]
            row["margin_drift"] = float(np.mean(parts))
            ok_s, d_s = ok[:nsel], d[:nsel]
            dsel[name] = np.abs(d_s)
            row["margin_drift_sel"] = float(np.mean([np.mean(np.abs(d_s[g])) for g in (ok_s, ~ok_s) if g.any()]))
            row["margin_drift_succ"] = float(np.mean(np.abs(d[ok]))) if ok.any() else float("nan")
            row["margin_drift_fail"] = float(np.mean(np.abs(d[~ok]))) if (~ok).any() else float("nan")
            row["suppression"] = float(np.mean(d[~ok])) if (~ok).any() else float("nan")
            row["flip_rate"] = float(np.mean((M_new > cp.q) != alarm_ref))
            k_new = np.array([cp.first_alarm(x) if cp.first_alarm(x) is not None else len(x) for x in S_new])
            either = (k_new < np.array(K_ref)) | (k_ref < np.array(K_ref))
            row["timing_drift"] = (float(np.mean(np.abs(k_new - k_ref)[either] / np.array(K_ref)[either]))
                                   if either.any() else 0.0)
            for n_sub in stab:  # ranking-stability check on nested subsets of captured episodes
                g_ok = ok[:n_sub]
                dd = d[:n_sub]
                row[f"margin_drift@{n_sub}"] = float(np.mean([np.mean(np.abs(dd[g])) for g in (g_ok, ~g_ok) if g.any()]))
        row["seconds"] = round(time.time() - t_blk, 2)
        out.append(row)
        print(f"{name:70s} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    greedy = validation = stability = greedy_trace = ranking_uncertainty = None
    if mon is not None and dsel:
        ranking_uncertainty = rank_bootstrap(s.module, blocks, dsel, ok[:min(a.select_episodes, len(ok))], a.rank_boot)
    # 1) PRIMARY selection (frozen here): *_sel block scores on the selection episodes + greedy search.
    if a.greedy and mon is not None:
        greedy, validation, greedy_trace = greedy_check(a, s, episodes, ok, ref, mon_stats, M_ref, m_sd, blocks, out)
    # 2) EXPLORATORY, after selection is frozen: ranking stability on nested 12/24/48 subsets. The 48-episode
    #    ranking uses the held-out episodes, so it never feeds selection, tuning or any confirmatory claim.
    if stab and mon is not None:
        stability = stability_check(s.module, blocks, out, stab)
    s.close()
    write_json(ROOT / "results/sensitivity" / f"{a.model}{a.tag}.json",
               {"model": a.model, "bits": a.bits, "n_episodes": len(episodes), "n_queries": n_q,
                "n_success": int(ok.sum()), "n_failure": int((~ok).sum()),
                "monitor": a.monitor if mon is not None else None, "alpha": a.alpha, "ref_key": a.ref_key,
                "seconds": round(time.time() - t0, 1), "blocks": out, "greedy": greedy, "greedy_trace": greedy_trace, "validation": validation,
                "sd_floor": SD_FLOOR, "scale": "sd of M over all captured reference episodes, floored",
                "greedy_search": {"restricted": a.greedy_candidates > 0, "candidates": a.greedy_candidates,
                                  "random_per_step": a.greedy_random, "select_episodes": a.select_episodes,
                                  "holdout_episodes": a.holdout_episodes} if a.greedy else None,
                "stability": stability,
                "stability_role": "exploratory; computed after primary selection was frozen; not used for selection",
                "ranking_uncertainty": ranking_uncertainty,
                "selection_counts": {"selection": int(min(a.select_episodes, len(ok))),
                                     "selection_failures": int((~ok[:a.select_episodes]).sum()),
                                     "holdout": int(len(ok) - min(a.select_episodes, len(ok))),
                                     "holdout_failures": int((~ok[a.select_episodes:]).sum())},
                "cost": {"capture_seconds": round(capture_seconds, 1),
                         "seconds_per_block": float(np.mean([r["seconds"] for r in out])) if out else None,
                         "greedy_seconds_per_J_eval": (greedy_trace or [{}])[0].get("sec_per_J") if greedy_trace else None,
                         "greedy_seconds_per_step": float(np.mean([x["seconds"] for x in greedy_trace]))
                         if greedy_trace else None},
                "note": "selection is an observation-controlled surrogate on reference rollouts (reference labels)"})


def rank_bootstrap(module, blocks, dsel, ok_sel, B=500, budgets=(5, 10, 20), seed=0):
    """Uncertainty of the margin_drift ranking from the finite selection set: resample selection episodes
    within outcome group, recompute J_b and the selected set (same fill rule) each time. Reports per budget
    the mean Jaccard overlap of bootstrap sets with the actual set, and per block the probability of
    being selected. Stage C entry criterion: mean Jaccard at the 10% budget >= 0.5, else capture more."""
    from torch import nn

    from assure.variants import _block_params, fill_budget
    params = _block_params(module, blocks)
    names = [b for b in blocks if b in dsel]
    D = np.stack([dsel[b] for b in names])         # (blocks, selection episodes)
    rng = np.random.default_rng(seed)
    grp = [np.where(ok_sel)[0], np.where(~ok_sel)[0]]

    def J(cols):
        o = ok_sel[cols]
        return np.mean([D[:, cols][:, o == k].mean(1) for k in (True, False) if (o == k).any()], axis=0)

    def selected(j, pct):
        order = [names[i] for i in np.argsort(-j / np.array([max(params[n], 1) for n in names]))]
        return set(fill_budget(module, order, pct / 100))
    base_j = J(np.arange(len(ok_sel)))
    actual = {p: selected(base_j, p) for p in budgets}
    jac = {p: [] for p in budgets}
    freq = {p: dict.fromkeys(names, 0) for p in budgets}
    for _ in range(B):
        cols = np.concatenate([rng.choice(g_, len(g_)) for g_ in grp if len(g_)])
        jb = J(cols)
        for p in budgets:
            S = selected(jb, p)
            jac[p].append(len(S & actual[p]) / max(len(S | actual[p]), 1))
            for n in names:
                freq[p][n] += any(x == n or x.startswith(n + ".") for x in S)
    out = {str(p): {"mean_jaccard": float(np.mean(jac[p])), "jaccard_q05": float(np.percentile(jac[p], 5)),
                    "selection_prob": {n: freq[p][n] / B for n in names if freq[p][n]}} for p in budgets}
    out["entry_criterion_10pct"] = bool(out["10"]["mean_jaccard"] >= 0.5) if "10" in out else None
    print("ranking uncertainty:", {p: round(out[str(p)]["mean_jaccard"], 3) for p in budgets})
    _ = nn
    return out


def stability_check(module, blocks, out, sizes, budgets=(5, 10, 20)):
    """Selected sets and rank correlation of J_b when the ranking uses fewer captured episodes."""
    from scipy.stats import spearmanr
    from torch import nn

    from assure.variants import _block_params
    params = _block_params(module, blocks)
    total = sum(m.weight.numel() for m in module.modules() if isinstance(m, nn.Linear))
    big = max(sizes)

    def chosen(key, pct):
        order = sorted((r for r in out if key in r), key=lambda r: r[key] / max(params[r["name"]], 1), reverse=True)
        sel, used = set(), 0
        for r in order:
            if used + params[r["name"]] <= pct / 100 * total:
                sel.add(r["name"])
                used += params[r["name"]]
        return sel
    rows = []
    for n in sizes:
        ka, kb = f"margin_drift@{n}", f"margin_drift@{big}"
        rho = spearmanr([r[ka] for r in out], [r[kb] for r in out]).correlation
        for pct in budgets:
            A, B = chosen(ka, pct), chosen(kb, pct)
            rows.append({"episodes": n, "budget_pct": pct, "spearman_vs_max": float(rho),
                         "jaccard_vs_max": len(A & B) / max(len(A | B), 1)})
    for r in rows:
        print("stability", r)
    return rows


def greedy_check(a, s, episodes, ok, ref, mon_stats, M_ref, m_sd, blocks, out):
    """RESTRICTED greedy (unless --greedy_candidates 0) and a held-out comparison of selection strategies.

    J(P) = outcome-balanced mean |M - M_ref| / sd(M_ref) for the model quantized to `bits` EXCEPT the
    protected set P (the deployed configuration), with the frozen fp16 monitor and band.
    SELECTION uses only the first --select_episodes captured episodes: the independent ranking
    (margin_drift_sel) and the greedy search both see exactly these. COMPARISON uses only the next
    --holdout_episodes episodes, never seen by any selection step, so a strategy cannot win by fitting
    the episodes it was chosen on. Every strategy's set is completed to the same budget by
    variants.fill_budget (sub-block), so they are compared at matched memory. Reports, per budget, held-out
    J for independent / greedy / random (3 seeds), held-out differences with episode-bootstrap 95%
    intervals, the additivity gap, overlap, actual protected fractions and non-improving greedy steps.
    Not a global optimum: the search budget is written to the output and must be reported.
    """
    import random as _random

    from torch import nn

    from assure.variants import _block_params, fill_budget, protected_fraction

    n_sel = min(a.select_episodes, len(episodes))
    sel = list(range(n_sel))
    hold = list(range(n_sel, min(n_sel + a.holdout_episodes, len(episodes))))
    full = {k: v.detach().to("cpu", copy=True) for k, v in s.module.state_dict().items()}
    params = _block_params(s.module, blocks)
    total = sum(m.weight.numel() for m in s.module.modules() if isinstance(m, nn.Linear))
    cache: dict = {}

    def per_episode(P, idxs):
        key = (tuple(sorted(P)), tuple(idxs))
        if key in cache:
            return cache[key]
        s.module.load_state_dict(full)
        apply_fake_quant(s.module, QuantSpec(bits=a.bits, protect=list(P)))
        run = [[s.replay(c) for c in episodes[i]] for i in idxs]
        _, M_new = mon_stats(run)
        s.module.load_state_dict(full)
        cache[key] = np.abs(M_new - M_ref[idxs]) / m_sd
        return cache[key]

    def J_of(v, idxs):
        o = ok[idxs]
        return float(np.mean([np.mean(v[g]) for g in (o, ~o) if g.any()]))

    def J(P, idxs):
        return J_of(per_episode(P, idxs), idxs)

    key = "margin_drift_sel" if "margin_drift_sel" in out[0] else "margin_drift"
    jb = {r["name"]: r[key] for r in out}
    by_ratio = sorted(jb, key=lambda b: jb[b] / max(params[b], 1), reverse=True)
    budgets = [int(x) for x in a.greedy_budgets.split(",")]

    # greedy search on SELECTION episodes, to the largest budget; smaller budgets are prefixes
    rng = _random.Random(0)
    P, used, seq, j_now = [], 0, [], J([], sel)
    cap = max(budgets) / 100 * total
    top = by_ratio if a.greedy_candidates <= 0 else by_ratio[: a.greedy_candidates]
    while True:
        cand = [b for b in top if b not in P and used + params[b] <= cap]
        rest = [b for b in blocks if b not in P and b not in cand and used + params[b] <= cap]
        cand += rng.sample(rest, min(a.greedy_random, len(rest)))
        if not cand:
            break
        t_step = time.time()
        gains = {b: (j_now - J(P + [b], sel)) / max(params[b], 1) for b in cand}
        best = max(gains, key=gains.get)
        j_before = j_now
        P.append(best)  # the budget is always completed, even by a non-improving block (matched memory)
        used += params[best]
        j_now = J(P, sel)
        dt = time.time() - t_step
        seq.append({"block": best, "used_frac": used / total, "J_sel": j_now, "improved": bool(j_now < j_before),
                    "seconds": round(dt, 2), "sec_per_J": round(dt / (len(cand) + 1), 2)})
        print(f"greedy +{best} -> J_sel={j_now:.4f} ({used / total:.1%} protected)")
    greedy = {str(pct): [x["block"] for x in seq if x["used_frac"] <= pct / 100] for pct in budgets}

    def boot_diff(va, vb, B=2000):
        """95% interval of J(a) - J(b) on held-out episodes, resampling episodes within outcome groups."""
        o = ok[hold]
        r_ = np.random.default_rng(0)
        grp = [np.where(o)[0], np.where(~o)[0]]
        vals = []
        for _ in range(B):
            ii = np.concatenate([r_.choice(g_, len(g_)) for g_ in grp if len(g_)])
            oo = o[ii]
            ja = np.mean([np.mean(va[ii][oo == k]) for k in (True, False) if (oo == k).any()])
            jb_ = np.mean([np.mean(vb[ii][oo == k]) for k in (True, False) if (oo == k).any()])
            vals.append(ja - jb_)
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    rows = []
    j0_hold = J([], hold) if hold else float("nan")
    for pct in budgets:
        f = pct / 100
        ind = fill_budget(s.module, by_ratio, f)
        gre = fill_budget(s.module, greedy[str(pct)] + [b for b in by_ratio if b not in greedy[str(pct)]], f)
        rnd = [fill_budget(s.module, _random.Random(k).sample(blocks, len(blocks)), f) for k in range(3)]
        row = {"budget_pct": pct,
               "protected_frac_independent": protected_fraction(s.module, ind),
               "protected_frac_greedy": protected_fraction(s.module, gre),
               "protected_frac_random_mean": float(np.mean([protected_fraction(s.module, r) for r in rnd])),
               "J_sel_independent": J(ind, sel), "J_sel_greedy": J(gre, sel),
               "overlap_independent_greedy": len(set(ind) & set(gre)) / max(len(set(ind) | set(gre)), 1),
               "greedy_nonimproving_steps": sum(not x["improved"] for x in seq if x["used_frac"] <= f),
               "independent_predicted_drop_sel": float(sum(jb[b] for b in ind if b in jb)),
               "independent_actual_drop_sel": J([], sel) - J(ind, sel)}
        if hold:
            vi, vg = per_episode(ind, hold), per_episode(gre, hold)
            vr = [per_episode(r, hold) for r in rnd]
            row.update(J_hold_unprotected=j0_hold, J_hold_independent=J_of(vi, hold), J_hold_greedy=J_of(vg, hold),
                       J_hold_random_mean=float(np.mean([J_of(v, hold) for v in vr])))
            lo, hi = boot_diff(vg, vi)
            row.update(J_hold_greedy_minus_independent=row["J_hold_greedy"] - row["J_hold_independent"],
                       ci_greedy_minus_independent=[lo, hi])
            lo, hi = boot_diff(vi, vr[0])
            row.update(J_hold_independent_minus_random0=row["J_hold_independent"] - J_of(vr[0], hold),
                       ci_independent_minus_random0=[lo, hi])
        rows.append(row)
        print(row)
    return greedy, rows, seq


if __name__ == "__main__":
    main()
