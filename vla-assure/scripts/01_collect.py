"""Collect LIBERO rollouts (actions, success, monitor features) for one model x variant x suite.

  python scripts/01_collect.py --model pi05 --variant fp16 --suite libero_spatial
  python scripts/01_collect.py --model pi05 --variant w4 --suite libero_spatial --pilot
  python scripts/01_collect.py --model pi05 --variant w4 --suite libero_10 --condition shift_mild
  python scripts/01_collect.py --model openvla --variant fp16 --suite libero_spatial --repo_root ~/openvla
  # reference rollouts with shadow copies (observation-controlled monitor scores):
  python scripts/01_collect.py --model pi05 --variant fp16 --suite libero_10 --condition shift_mild \
      --shadow w8,w4,update

Trial design (src/assure/common.py), identical for every variant of a comparison:
  trial (t, i) -> seed = seed + 1000 t + i and a fixed initial state trial_init_state(seed, t, i).
  reference (fp16): trials 0..R-1 per task (R = reference episodes per task). Every slot has a
                    pre-registered role from trial_roles() (train / calib / test, pooled random).
                    Optional reference top-ups R, R+1, ... are monitor-training only.
  variants:         the reference 'test' slots (paired endpoint) and 'repair' slots (repair pool), plus
                    optional top-ups at TOPUP_OFFSET + k that join the repair pool.
Do not change R, the split, the seed or --tasks between variants of one comparison.

Writes rollouts/<model>/<variant>/<suite>[+<condition>]/*.npz and run.json (success rate, timing, tap check).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assure.adapters import make_session  # noqa: E402
from assure.common import (EXT_OFFSET, ROOT, TOPUP_OFFSET, load_configs, setup_logging, trial_roles,  # noqa: E402
                           trial_seed, write_json)
from assure.variants import resolve  # noqa: E402

log = logging.getLogger("collect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--condition", default="nominal", help="camera perturbation from experiment.yaml:conditions")
    ap.add_argument("--tasks", default="all", help="'all' or comma list of task ids")
    ap.add_argument("--episodes", type=int, default=None,
                    help="reference trials per task R (default experiment.yaml). Variants always run the test slots")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--alt_samples", type=int, default=0, help="extra noise samples per query (flow VLAs; enables 'fiper')")
    ap.add_argument("--repo_root", default=None, help="openvla / openvla-oft repo root (openvla backend)")
    ap.add_argument("--out", default=str(ROOT / "rollouts"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--shadow", default="", help="comma list of variants run as non-controlling shadow copies")
    ap.add_argument("--target_successes", type=int, default=0,
                    help="reference: total successes; VARIANTS: successes in the repair pool (repair slots + "
                         "top-ups). Precommitted: rounds of one trial per task, at most --max_extra_rounds")
    ap.add_argument("--max_extra_rounds", type=int, default=40)
    ap.add_argument("--extension", type=int, default=0,
                    help="VARIANTS (Stage C families F2-F4): also run K pre-registered extension slots per task "
                         "(EXT_OFFSET + k), paired across rules; K from results/power/stage_c_sizing.json")
    a = ap.parse_args()
    setup_logging()

    models, exp = load_configs()
    mcfg, var = models[a.model], resolve(a.variant, exp)
    eps = a.episodes or (exp["pilot_episodes_per_task"] if a.pilot else exp["reference_episodes_per_task"])
    shadow = {n: resolve(n, exp) for n in a.shadow.split(",") if n}
    tasks = list(range(10)) if a.tasks == "all" else [int(t) for t in a.tasks.split(",")]
    out = Path(a.out)

    is_ref = a.variant == "fp16"
    R = eps
    roles = trial_roles(exp["seed"], tasks, R, exp["split"])
    if is_ref:
        trial_ids = {t: list(range(R)) for t in tasks}
        topup_base = R
    else:
        trial_ids = {t: sorted(i for (tt, i), r in roles.items() if tt == t and r in ("test", "repair"))
                     + [EXT_OFFSET + k for k in range(a.extension)] for t in tasks}
        topup_base = TOPUP_OFFSET
    s = make_session(a.model, mcfg, a.variant, var, a.suite, device=a.device, base_seed=exp["seed"],
                     n_alt_samples=a.alt_samples, repo_root=a.repo_root,
                     condition=a.condition, condition_spec=exp["conditions"][a.condition], shadow=shadow)
    log.info("variant report: %s", s.report)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_ok = n = n_pool = 0
    in_pool = (lambda t, e: True) if is_ref else (lambda t, e: TOPUP_OFFSET <= e < EXT_OFFSET or roles.get((t, e)) == "repair")

    def run(t, e):
        nonlocal n_ok, n, n_pool
        target = out / a.model / a.variant / s.key / f"t{t:02d}_e{e:03d}.npz"
        if a.skip_existing and target.exists():
            import numpy as np
            ok_ = bool(np.load(target)["success"])  # count it toward --target_successes
            n_ok += ok_
            n_pool += ok_ and in_pool(t, e)
            n += 1
            return
        ep = s.rollout_episode(t, e, seed=trial_seed(exp["seed"], t, e))
        ep.save(out)
        n_ok += ep.success
        n_pool += bool(ep.success) and in_pool(t, e)
        n += 1
        log.info("task %d ep %d success=%s len=%d queries=%d", t, e, ep.success, ep.length, len(ep.query_step))

    extra = 0
    try:
        for t in tasks:
            for e in trial_ids[t]:
                run(t, e)
        # Top up so the conformal band has enough successes (see experiment.yaml:calibration).
        while a.target_successes and n_pool < a.target_successes and extra < a.max_extra_rounds:
            for t in tasks:
                run(t, topup_base + extra)
            extra += 1
        if a.target_successes and n_pool < a.target_successes:
            log.warning("only %d/%d successes after %d extra rounds (target not met; reported, no further top-up)",
                        n_pool, a.target_successes, extra)
    finally:
        fired = dict(s.tap.fired)
        qs = sorted(s.query_seconds)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        s.close()
    rep = dict(s.report, sampler=getattr(s, "sampler", None), pool_successes=n_pool,
               topup_rule={"target": a.target_successes, "max_rounds": a.max_extra_rounds, "per_round": len(tasks)},
               episodes=n, success_rate=n_ok / max(n, 1), seconds=round(time.time() - t0, 1),
               tap_fired=fired, episodes_per_task=eps, tasks=tasks, n_reference=R, split=exp["split"], topup_rounds=extra if a.target_successes else 0,
               trial_ids={str(t): v for t, v in trial_ids.items()}, extension_per_task=a.extension,
               role_counts_per_task={str(t): {k: sum(1 for (tt, _), r in roles.items() if tt == t and r == k)
                                              for k in exp["split"]} for t in tasks}, condition=a.condition, shadow_names=list(shadow),
               note_peak_mem="includes shadow copies" if shadow else "",
               peak_gpu_mem_gib=peak, query_latency_median_s=qs[len(qs) // 2] if qs else None,
               query_latency_p95_s=qs[int(0.95 * (len(qs) - 1))] if qs else None)
    write_json(out / a.model / a.variant / s.key / "run.json", rep)
    if any(v == 0 for v in fired.values()):
        log.error("Some taps never fired: %s -- fix module paths in configs/models.yaml", fired)
    log.info("done: success_rate=%.3f over %d episodes", rep["success_rate"], n)


if __name__ == "__main__":
    main()
