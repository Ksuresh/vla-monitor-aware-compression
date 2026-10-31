"""Real-kernel parity: does a mixed-precision set behave the same with real int4 kernels (tao4p-*) as with
the fake quantization (w4p-*) it was selected and evaluated with?

For each rule, the SAME protected set is built both ways and replayed on the same captured reference
inputs, scored with the frozen fp16 monitor and band. Pre-registered acceptance criteria (all must hold;
configs/experiment.yaml: parity):

  identical_set        the two variants protect exactly the same modules
  action_ratio         median_q ||a_real - a_fake|| / median_q ||a_fake - a_fp16||   <= 0.25
                       (kernel discrepancy at most a quarter of the quantization perturbation itself)
  dJ                   |J_real - J_fake| <= 0.10, J = outcome-balanced mean |M - M_fp16| / sd(M_fp16)
  score_spearman       Spearman correlation of per-query monitor scores, real vs fake      >= 0.95
  fa_disagreement      on reference SUCCESSES: fraction whose alarm decision differs (false-alarm side)  <= 0.10
  md_disagreement      on reference FAILURES: fraction whose alarm decision differs (missed-detection side) <= 0.10
                       (reported separately so disagreements concentrated on failures cannot hide)
Across rules (reported, and required for the efficiency claim of the proposed rule):
  advantage_preserved  for the proposed rule vs every baseline, sign(J_real(prop) - J_real(base)) equals
                       sign(J_fake(prop) - J_fake(base)): real kernels keep the direction of the advantage

Efficiency for a configuration is reported in the paper only if it passes; otherwise it is reported as
failing parity, with its numbers, and its assurance results are attributed to the fake-quantized model.

  python scripts/07_parity.py --model pi05 --suite libero_10 --condition shift_mild \
      --ref_key libero_10+shift_mild --rules margin_drift,margin_greedy,action_drift,random --pct 10
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from assure.adapters import make_session  # noqa: E402
from assure.common import load_configs, setup_logging, write_json  # noqa: E402
from assure.variants import resolve  # noqa: E402

sens = importlib.import_module("02_sensitivity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--condition", default="nominal")
    ap.add_argument("--ref_key", required=True)
    ap.add_argument("--monitor", default="mahalanobis#A")
    ap.add_argument("--proposed", default="margin_drift")
    ap.add_argument("--rules", default="margin_drift,action_drift,weight_error,random")
    ap.add_argument("--pct", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    setup_logging()
    models, exp = load_configs()
    crit = exp.get("parity", {})
    n_ep = int(crit.get("episodes", 24))
    mcfg = models[a.model]
    kw = dict(device=a.device, repo_root=a.repo_root, base_seed=exp["seed"], condition=a.condition,
              condition_spec=exp["conditions"][a.condition])
    ref_s = make_session(a.model, mcfg, "fp16", exp["variants"]["fp16"], a.suite, **kw)
    episodes, ok = [], []
    for e in range(n_ep):  # dedicated trial ids, outside every role of the partition
        cap: list = []
        ep = ref_s.rollout_episode(task_id=e % 10, episode_idx=950_000 + e, seed=2 * 10**6 + e, capture=cap, cap_max=60)
        episodes.append(cap)
        ok.append(bool(ep.success))
    ok = np.array(ok)
    base = [[ref_s.replay(c) for c in ep] for ep in episodes]
    ref_s.close()
    mon, cp = sens.load_monitor(a.model, a.ref_key, a.monitor, list(mcfg["tap"]), a.alpha, exp)

    def stats(run):
        S = [mon.score(sens._Ep([f for f, _ in ep])) for ep in run]
        return S, np.array([cp.margin(x) for x in S])
    S0, M0 = stats(base)
    m_sd = max(float(np.std(M0)), sens.SD_FLOOR)

    def J(M):
        d = np.abs(M - M0) / m_sd
        return float(np.mean([d[g].mean() for g in (ok, ~ok) if g.any()]))

    results = {}
    for rule in a.rules.split(","):
        out = {}
        runs = {}
        for kind in ("w4p", "tao4p"):
            name = f"{kind}-{rule}-f{a.pct}"
            s = make_session(a.model, mcfg, name, resolve(name, exp), a.suite, **kw)
            runs[kind] = ([[s.replay(c) for c in ep] for ep in episodes], s.report.get("protected_blocks"))
            s.close()
            del s
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        (fake, set_f), (real, set_r) = runs["w4p"], runs["tao4p"]
        num = [np.linalg.norm(r[1] - f[1]) for er, ef in zip(real, fake) for r, f in zip(er, ef)]
        den = [np.linalg.norm(f[1] - b[1]) for ef, eb in zip(fake, base) for f, b in zip(ef, eb)]
        Sf, Mf = stats(fake)
        Sr, Mr = stats(real)
        out["identical_set"] = sorted(set_f or []) == sorted(set_r or [])
        out["action_ratio"] = float(np.median(num) / max(np.median(den), 1e-12))
        out["dJ"] = abs(J(Mr) - J(Mf))
        out["score_spearman"] = float(spearmanr(np.concatenate(Sr), np.concatenate(Sf)).correlation)
        dis = (Mr > cp.q) != (Mf > cp.q)
        out["fa_disagreement"] = float(dis[ok].mean()) if ok.any() else float("nan")
        out["md_disagreement"] = float(dis[~ok].mean()) if (~ok).any() else float("nan")
        out["n_success"], out["n_failure"] = int(ok.sum()), int((~ok).sum())
        out["J_fake"], out["J_real"] = J(Mf), J(Mr)
        out["pass"] = bool(out["identical_set"] and out["action_ratio"] <= crit.get("action_ratio_max", 0.25)
                           and out["dJ"] <= crit.get("dJ_max", 0.10)
                           and out["score_spearman"] >= crit.get("score_spearman_min", 0.95)
                           and not (out["fa_disagreement"] > crit.get("fa_disagreement_max", 0.10))
                           and not (out["md_disagreement"] > crit.get("md_disagreement_max", 0.10)))
        results[rule] = out
        print(rule, out)
    adv = {}
    if a.proposed in results:
        p_ = results[a.proposed]
        for b, rb in results.items():
            if b != a.proposed:
                adv[b] = bool(np.sign(p_["J_real"] - rb["J_real"]) == np.sign(p_["J_fake"] - rb["J_fake"]))
        print("advantage preserved under real kernels:", adv)
    # Pre-registered F4 baseline: first rule in families.f4_baseline_order whose real-kernel build passes parity.
    # Parity uses captured reference episodes only, so this choice never sees confirmatory outcomes.
    order = exp.get("families", {}).get("f4_baseline_order", [])
    proposed_pass = bool(results.get(a.proposed, {}).get("pass"))
    f4_base = next((r for r in order if results.get(r, {}).get("pass")), None) if proposed_pass else None
    print("F4:", "proposed fails parity -> F4 not confirmable" if not proposed_pass else
          (f"baseline = {f4_base}" if f4_base else "no baseline in the order passes parity -> F4 not confirmable"))
    write_json(ROOT / "results/parity" / f"{a.model}_f{a.pct}.json",
               {"criteria": crit, "episodes": n_ep, "n_failures": int((~ok).sum()), "results": results,
                "proposed": a.proposed, "advantage_preserved": adv, "f4_baseline_order": order,
                "proposed_passes_parity": proposed_pass, "f4_baseline": f4_base})


if __name__ == "__main__":
    main()
