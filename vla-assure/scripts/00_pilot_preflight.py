"""Free CPU-only preflight. No checkpoint download, GPU or simulator required.

Run from vla-assure/: python scripts/00_pilot_preflight.py
This checks repository plumbing, not VLA checkpoint compatibility.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import yaml

    problems = []
    for rel in (
        "requirements.txt",
        "configs/models.yaml",
        "configs/experiment.yaml",
        "src/assure/quant.py",
        "scripts/02_sensitivity.py",
        "scripts/03_evaluate.py",
        "tests/test_pipeline.py",
    ):
        exists = (ROOT / rel).is_file()
        print(f"{'OK' if exists else 'MISSING'} {rel}")
        if not exists:
            problems.append(rel)

    cfg_path = ROOT / "configs/models.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        smol = cfg.get("models", {}).get("smolvla", {})
        if not smol:
            problems.append("smolvla config")
        else:
            print("SmolVLA checkpoint configured:", smol.get("checkpoint", "<missing>"))
            print("SmolVLA taps:", ", ".join(smol.get("tap", {})))
            if not smol.get("checkpoint"):
                problems.append("smolvla checkpoint")
            if not smol.get("tap"):
                problems.append("smolvla taps")

    for module in ("torch", "numpy", "pytest", "yaml"):
        available = importlib.util.find_spec(module) is not None
        print(f"{'OK' if available else 'MISSING'} import {module}")
        if not available:
            problems.append(f"dependency {module}")

    print("\nNOT VERIFIED: checkpoint availability, LeRobot version, hook firing,")
    print("action inference, real INT4 deployment, or LIBERO compatibility.")
    if problems:
        print("PREFLIGHT FAILED:", ", ".join(problems))
        return 1
    print("PREFLIGHT PASSED: repository structure and declared dependencies only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
