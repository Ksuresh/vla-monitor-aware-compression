from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_configs():
    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]
    exp = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    return models, exp


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


# ---------------------------------------------------------------- trial design (pairing)
N_INIT_STATES = 50  # LIBERO ships 50 fixed initial states per task


def trial_seed(base_seed: int, task: int, idx: int) -> int:
    return base_seed + 1000 * task + idx


def trial_init_state(base_seed: int, task: int, idx: int, n_init: int = N_INIT_STATES) -> int:
    """Initial state of trial (task, idx). The first n_init trials of a task visit every LIBERO
    initial state once in a seeded random order; later (top-up) trials draw one uniformly at random.
    The assignment depends only on (seed, task, idx), never on outcomes, and is identical for every
    variant, so trial idx is the same situation for every policy."""
    import numpy as np
    if idx < n_init:
        return int(np.random.default_rng([base_seed, task, 0]).permutation(n_init)[idx])
    return int(np.random.default_rng([base_seed, task, idx]).integers(n_init))


def trial_roles(base_seed: int, tasks: list[int], n_reference: int, split: dict[str, float]) -> dict[tuple[int, int], str]:
    """Pre-registered role of every reference trial slot (task, idx < n_reference).

    One seeded uniformly random partition of ALL slots (pooled over tasks) into train / calib / test,
    fixed before any rollout and independent of outcomes. Variants run exactly the 'test' slots, so
    every variant trial has a reference counterpart. Because calib and test are a random partition
    of the same slots, calibration and test episodes are exchangeable by design (finite-population
    randomisation), with no assumption about how LIBERO's initial states were generated.
    Reference top-up trials (idx >= n_reference) are always 'train': they help the monitor but
    never enter calibration, so data-dependent stopping cannot touch the guarantee.
    """
    import numpy as np
    slots = [(t, i) for t in sorted(tasks) for i in range(n_reference)]
    perm = np.random.default_rng([base_seed, 7]).permutation(len(slots))
    names = list(split)
    cuts = np.cumsum([int(round(split[n] * len(slots))) for n in names[:-1]])
    roles = {}
    for name, part in zip(names, np.split(perm, cuts)):
        for k in part:
            roles[slots[k]] = name
    return roles


def role_of(roles: dict, task: int, idx: int) -> str:
    if idx >= EXT_OFFSET:
        return "ext_test"
    return roles.get((task, idx), "train" if idx < TOPUP_OFFSET else "adapt")


TOPUP_OFFSET = 10_000  # trial indices >= this are variant-only top-ups (repair pool, never paired)
# Stage C detection extension (families F2-F4): variant-only paired slots EXT_OFFSET + k, k < K per task,
# with K fixed from results/power/stage_c_sizing.json BEFORE any deployed variant executes. Every rule's
# variant runs the same slots, so they are paired across rules (not with the reference).
EXT_OFFSET = 20_000
