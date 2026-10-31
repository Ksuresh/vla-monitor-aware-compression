"""Gate 0: confirm a model loads, a LIBERO episode runs, and every feature tap fires.

  python scripts/00_smoke_test.py --model pi05
  python scripts/00_smoke_test.py --model smolvla --list_modules norm     # find module paths
  python scripts/00_smoke_test.py --model openvla --repo_root ~/openvla

Run this for every model before any long job. It takes a few minutes per model.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assure.adapters import make_session  # noqa: E402
from assure.common import load_configs, setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--list_modules", default=None, help="regex; print matching module names and exit")
    a = ap.parse_args()
    setup_logging()
    models, exp = load_configs()
    mcfg = models[a.model]

    if a.list_modules:
        # Load with taps disabled so wrong paths don't crash the listing.
        mcfg = dict(mcfg, tap={})
    s = make_session(a.model, mcfg, "fp16", exp["variants"]["fp16"], a.suite, device=a.device, repo_root=a.repo_root)
    if a.list_modules:
        rx = re.compile(a.list_modules)
        for n, m in s.module.named_modules():
            if rx.search(n):
                print(f"{n:90s} {type(m).__name__}")
        return

    ep = s.rollout_episode(task_id=0, episode_idx=0, seed=exp["seed"])
    with tempfile.TemporaryDirectory() as d:
        ep.save(Path(d))
    print("\n=== smoke test", a.model, "===")
    print("success:", ep.success, "| steps:", ep.length, "| policy queries:", len(ep.query_step))
    for k, v in ep.feats.items():
        print(f"tap {k}: {len(v)} features of dim {v[0].shape[0]}")
    if ep.chunks:
        print("action chunk shape:", ep.chunks[0].shape)
    print("tap fire counts:", s.tap.fired)
    bad = [k for k, v in s.tap.fired.items() if v == 0]
    s.close()
    if bad:
        sys.exit(f"FAIL: taps never fired: {bad}. Use --list_modules norm to find the right paths.")
    print("PASS")


if __name__ == "__main__":
    main()
