"""Observation-controlled check: does quantization alone move the failure monitor, on IDENTICAL inputs?

The fp16 policy drives the robot on a few held-out TEST slots while a non-controlling shadow copy of the
compressed variant (e.g. w4) sees exactly the same camera frames and robot state (collect with
--variant fp16 --shadow w4 --ref_roles test --max_per_task K --out <shadow_root>). Every difference in the
monitor between the two views is therefore caused by the model change alone, not by the robot taking a
different path. Labels are the fp16 outcomes (this says nothing about what w4 would do in control).

The monitor and its conformal band are fitted exactly as in 03_evaluate.py: monitor on the main reference
run's train slots (half A for 'mahalanobis#A'), band on its calibration successes. Shadow episodes are test or
repair slots, so they never touched fitting. The decision rule (READING line) is fixed in this file and was
set before any stage-2 run: CLEAR if score shift >= 0.3 SD or the FPR interval lies above 0.

Reports per variant:
  feature drift per tap   relative L2 change of the tapped features (fp16 -> shadow); a tap of an
                          UNQUANTIZED module should read ~0 (sanity check of the quantization scope)
  score shift             mean change of the episode alarm score on fp16 successes, in units of the
                          fp16 score SD (0 = no effect; 0.5 = half an SD)
  FPR fp16 vs shadow      alarm rate on fp16 successes under each view, paired, with a task-clustered
                          bootstrap 90% interval for the difference
  action divergence       relative change of the predicted action chunk (how differently it would act)

  python scripts/shadow_check.py --model smolvla --key libero_spatial --shadow_root rollouts_shadow
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assure import monitors as M  # noqa: E402
from assure.common import ROOT, load_configs, role_of, trial_roles  # noqa: E402
from assure.conformal import FunctionalCP  # noqa: E402
from assure.store import load_cell  # noqa: E402


def boot_diff(d, g, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    tasks = np.unique(g)
    by = {t: d[g == t] for t in tasks}
    vals = []
    for _ in range(B):
        s = np.concatenate([by[t] for t in rng.choice(tasks, len(tasks))])
        vals.append(s.mean())
    return float(d.mean()), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--shadow_root", required=True)
    ap.add_argument("--rollouts", default=str(ROOT / "rollouts"))
    ap.add_argument("--monitors", default="mahalanobis#A,mahalanobis")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--taps", default=None, help="comma list of taps the monitor reads (default: all in models.yaml)")
    ap.add_argument("--out", default=str(ROOT / "results/shadow_check.json"))
    ap.add_argument("--roles", default="test,repair",
                    help="slot roles accepted as shadow episodes (never train/calib). Stage 2 shadows run on "
                         "REPAIR slots so the predictor never sees the closed-loop test slots it is judged on")
    a = ap.parse_args()
    models, exp = load_configs()
    taps = a.taps.split(",") if a.taps else list(models[a.model]["tap"])
    all_taps = list(models[a.model]["tap"])
    root = Path(a.rollouts)
    ref = load_cell(root, a.model, "fp16", a.key)
    run = json.loads((root / a.model / "fp16" / a.key / "run.json").read_text())
    R = int(run.get("n_reference") or exp["reference_episodes_per_task"])
    roles = trial_roles(exp["seed"], sorted({e.meta["task_id"] for e in ref}), R, run.get("split") or exp["split"])
    role = lambda e: role_of(roles, e.meta["task_id"], e.meta["episode_idx"])  # noqa: E731
    train = [e for e in ref if role(e) == "train"]
    cal_succ = [e for e in ref if role(e) == "calib" and e.success]
    sh = [e for e in load_cell(Path(a.shadow_root), a.model, "fp16", a.key) if role(e) in set(a.roles.split(","))]
    variants = sorted(set().union(*[set(e.shadows()) for e in sh])) if sh else []
    if not sh or not variants:
        sys.exit(f"no shadow episodes under {a.shadow_root}")
    cal = exp.get("calibration", {})
    cpkw = dict(guarantee=cal.get("guarantee", "marginal"), delta=cal.get("delta", 0.1),
                shape_frac=cal.get("shape_frac", 0.5), min_shape=cal.get("min_shape", 10),
                strict=cal.get("strict", False), seed=exp["seed"])
    print(f"monitor reads taps: {taps}")
    print(f"reference: train {len(train)} / calib successes {len(cal_succ)} | shadow episodes {len(sh)} "
          f"({sum(e.success for e in sh)} fp16 successes) | variants {variants}")
    out = {"n_shadow": len(sh), "n_success": int(sum(e.success for e in sh)), "variants": {}}
    for v in variants:
        views = [e.as_shadow(v) for e in sh]
        drift = {}
        for t in all_taps:
            num = [np.linalg.norm(w.feats[t] - e.feats[t]) for e, w in zip(sh, views) if t in w.feats]
            den = [np.linalg.norm(e.feats[t]) + 1e-8 for e, w in zip(sh, views) if t in w.feats]
            drift[t] = float(np.mean(np.array(num) / np.array(den))) if num else float("nan")
        div = [float(np.linalg.norm(e.shadow_chunks[v] - e.chunk) / (np.linalg.norm(e.chunk) + 1e-8))
               for e in sh if e.chunk is not None and v in e.shadow_chunks and e.shadow_chunks[v].shape == e.chunk.shape]
        res = {"feature_drift": drift, "action_div": float(np.mean(div)) if div else float("nan"), "monitors": {}}
        print(f"\n=== {v} (shadow on identical observations) ===")
        print("feature drift (relative L2): " + ", ".join(f"{t} {x:.3f}" for t, x in drift.items()))
        print(f"action-chunk divergence: {res['action_div']:.3f}")
        ok = np.array([e.success for e in sh])
        g = np.array([e.meta["task_id"] for e in sh])
        for mname in a.monitors.split(","):
            base, half = M.parse_name(mname)
            mon = M.build(base, taps).fit(M.train_subset(train, half, exp["seed"]))
            cp = FunctionalCP(a.alpha, **cpkw).fit([mon.score(e) for e in cal_succ])
            m0 = np.array([cp.margin(mon.score(e)) for e in sh])
            m1 = np.array([cp.margin(mon.score(w)) for w in views])
            a0 = np.array([cp.first_alarm(mon.score(e)) is not None for e in sh], float)
            a1 = np.array([cp.first_alarm(mon.score(w)) is not None for w in views], float)
            sd = float(np.std(m0[ok])) + 1e-9
            shift = float(np.mean(m1[ok] - m0[ok]) / sd) if ok.any() else float("nan")
            d, lo, hi = boot_diff(a1[ok] - a0[ok], g[ok]) if ok.sum() >= 5 else (float("nan"),) * 3
            fail = ~ok
            r = {"score_shift_sd": shift, "fpr_fp16": float(a0[ok].mean()) if ok.any() else float("nan"),
                 "fpr_shadow": float(a1[ok].mean()) if ok.any() else float("nan"), "d_fpr": d, "d_fpr_90": [lo, hi],
                 "tpr_fp16": float(a0[fail].mean()) if fail.any() else float("nan"),
                 "tpr_shadow": float(a1[fail].mean()) if fail.any() else float("nan"),
                 "score_up_share": float(np.mean(m1[ok] > m0[ok])) if ok.any() else float("nan")}
            res["monitors"][mname] = r
            print(f"[{mname}] on fp16 successes (n={int(ok.sum())}): FPR fp16 {r['fpr_fp16']:.2f} -> shadow "
                  f"{r['fpr_shadow']:.2f} (D {d:+.2f}, 90% [{lo:+.2f}, {hi:+.2f}]) | score shift {shift:+.2f} SD, "
                  f"higher in {r['score_up_share']:.0%} of episodes | detection on fp16 failures "
                  f"{r['tpr_fp16']:.2f} -> {r['tpr_shadow']:.2f}")
        out["variants"][v] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    m = out["variants"][variants[0]]["monitors"].get(a.monitors.split(",")[0], {})
    s = m.get("score_shift_sd", float("nan"))
    verdict = ("CLEAR SHIFT: quantization alone moves the monitor on identical inputs" if s >= 0.3 or m.get("d_fpr_90", [0])[0] > 0
               else "SMALL SHIFT: the monitor barely reacts on identical inputs" if s >= 0.1
               else "NO SHIFT: the monitor is robust to this quantization on identical inputs")
    print("\nREADING:", verdict, "(score shift >= 0.3 SD or FPR interval above 0 = clear)")


if __name__ == "__main__":
    main()
