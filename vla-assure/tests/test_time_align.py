"""time_align keeps the reference's query times; pairing ignores the null-control seed shift."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from assure.store import LoadedEpisode  # noqa: E402
from resplit import time_align  # noqa: E402


def _ep(steps, seed=7):
    q = np.arange(0, 100, steps)
    return LoadedEpisode(Path("x"), {"task_id": 1, "episode_idx": 2, "seed": seed}, True, 100, q,
                         {"vlm_last": np.arange(len(q))[:, None].astype(float)}, np.zeros((len(q), 5, 7)), None,
                         np.zeros((100, 7)))


def test_time_align_matches_reference_steps():
    e = time_align(_ep(5), 2)
    assert list(e.query_step) == list(np.arange(0, 100, 10))
    assert e.feats["vlm_last"].shape[0] == len(e.query_step) == e.chunk.shape[0]


def test_pair_key_ignores_seed_shift():
    assert _ep(10, seed=7).pair_key == _ep(10, seed=100007).pair_key
