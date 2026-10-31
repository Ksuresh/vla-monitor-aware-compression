"""On-disk format for rollouts: one .npz per episode.

Each episode stores, at every policy *query* k (a new action chunk was predicted):
  feat__<tap>   (K, D)   pooled hidden state from that tap
  query_step    (K,)     environment step at which query k happened
  chunk         (K, H, A) predicted action chunk (H=1 for single-step policies)
  chunk_alt     (K, S, H, A) optional extra noise samples (for sampling-variance scores)
plus actions (T, A), success (bool), length T and metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    model: str
    variant: str
    suite: str
    task_id: int
    episode_idx: int
    seed: int
    success: bool = False
    length: int = 0
    actions: list = field(default_factory=list)
    query_step: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    chunk_alt: list = field(default_factory=list)
    feats: dict = field(default_factory=dict)  # tap -> list of (D,)
    shadow_chunks: dict = field(default_factory=dict)  # shadow variant -> list of (H, A)
    init_state_id: int | None = None

    def add_shadow(self, variant: str, feats: dict[str, np.ndarray], chunk: np.ndarray | None) -> None:
        """Features/actions a SHADOW model (e.g. the W4 copy) produced on this same observation.
        Stored as feat__shadow:<variant>:<tap>; the shadow never controls the robot."""
        for k, v in feats.items():
            self.feats.setdefault(f"shadow:{variant}:{k}", []).append(np.asarray(v, dtype=np.float32))
        if chunk is not None:
            self.shadow_chunks.setdefault(variant, []).append(np.asarray(chunk, dtype=np.float32))

    def add_query(self, step: int, feats: dict[str, np.ndarray], chunk: np.ndarray | None,
                  alt: np.ndarray | None = None) -> None:
        self.query_step.append(step)
        for k, v in feats.items():
            self.feats.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        if chunk is not None:
            self.chunks.append(np.asarray(chunk, dtype=np.float32))
        if alt is not None:
            self.chunk_alt.append(np.asarray(alt, dtype=np.float32))

    def save(self, root: Path) -> Path:
        d = root / self.model / self.variant / self.suite
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"t{self.task_id:02d}_e{self.episode_idx:03d}.npz"
        arrays = {
            "actions": np.asarray(self.actions, dtype=np.float32),
            "query_step": np.asarray(self.query_step, dtype=np.int32),
            "success": np.asarray(self.success),
            "length": np.asarray(self.length),
        }
        if self.chunks:
            arrays["chunk"] = np.stack(self.chunks)
        if self.chunk_alt:
            arrays["chunk_alt"] = np.stack(self.chunk_alt)
        for k, v in self.feats.items():
            arrays[f"feat__{k}"] = np.stack(v)
        for v, chunks in self.shadow_chunks.items():
            arrays[f"shadowchunk__{v}"] = np.stack(chunks)
        meta = dict(model=self.model, variant=self.variant, suite=self.suite, task_id=self.task_id,
                    episode_idx=self.episode_idx, seed=self.seed, init_state_id=self.init_state_id)
        np.savez_compressed(p, meta=json.dumps(meta), **arrays)
        return p


@dataclass
class LoadedEpisode:
    path: Path
    meta: dict
    success: bool
    length: int
    query_step: np.ndarray
    feats: dict[str, np.ndarray]
    chunk: np.ndarray | None
    chunk_alt: np.ndarray | None
    actions: np.ndarray
    shadow_chunks: dict = field(default_factory=dict)

    @property
    def pair_key(self) -> tuple:
        """Identifies the paired trial across variants: same task, initial state and seed."""
        return (self.meta["task_id"], self.meta["episode_idx"], self.meta["seed"])

    def shadows(self) -> list[str]:
        return sorted({k.split(":")[1] for k in self.feats if k.startswith("shadow:")})

    def as_shadow(self, variant: str) -> "LoadedEpisode":
        """Counterfactual view: the REFERENCE trajectory and outcome, but the features and action
        chunks the shadow model `variant` produced on the same observations. Any change in a monitor
        on this view is caused by the model change alone, not by a different set of episodes."""
        pre = f"shadow:{variant}:"
        feats = {k[len(pre):]: v for k, v in self.feats.items() if k.startswith(pre)}
        if "ext_pixels" in self.feats:
            feats["ext_pixels"] = self.feats["ext_pixels"]  # camera input is identical by construction
        return LoadedEpisode(self.path, dict(self.meta, variant=f"{variant}@shadow"), self.success, self.length,
                             self.query_step, feats, self.shadow_chunks.get(variant), None, self.actions)


def load_episode(path: Path) -> LoadedEpisode:
    z = np.load(path, allow_pickle=False)
    feats = {k[len("feat__"):]: z[k] for k in z.files if k.startswith("feat__")}
    shadow_chunks = {k[len("shadowchunk__"):]: z[k] for k in z.files if k.startswith("shadowchunk__")}
    return LoadedEpisode(
        path=path,
        meta=json.loads(str(z["meta"])),
        success=bool(z["success"]),
        length=int(z["length"]),
        query_step=z["query_step"],
        feats=feats,
        chunk=z["chunk"] if "chunk" in z.files else None,
        chunk_alt=z["chunk_alt"] if "chunk_alt" in z.files else None,
        actions=z["actions"],
        shadow_chunks=shadow_chunks,
    )


def load_cell(root: Path, model: str, variant: str, suite: str) -> list[LoadedEpisode]:
    d = root / model / variant / suite
    return [load_episode(p) for p in sorted(d.glob("*.npz"))]


def split_by_task(eps: list[LoadedEpisode], fractions: dict[str, float], seed: int) -> dict[str, list]:
    """Split episodes within each task so every task appears in every split."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in fractions}
    by_task: dict[int, list] = {}
    for e in eps:
        by_task.setdefault(e.meta["task_id"], []).append(e)
    names = list(fractions)
    for _, group in sorted(by_task.items()):
        idx = rng.permutation(len(group))
        cuts = np.cumsum([int(round(fractions[n] * len(group))) for n in names[:-1]])
        parts = np.split(idx, cuts)
        for n, part in zip(names, parts):
            out[n].extend(group[i] for i in part)
    return out


def split_pooled(eps: list[LoadedEpisode], fractions: dict[str, float], seed: int) -> dict[str, list]:
    """Simple random split of the pooled cell (all tasks together).

    Used wherever a conformal guarantee is claimed: a uniformly random partition makes calibration
    and test episodes exactly exchangeable, so the guarantee refers to "a successful episode drawn
    from this cell's design" (the suite's task mix x initial states x condition), pooled over tasks.
    A task-stratified split fixes per-task counts in calibration, which is not exchangeable with a
    test episode from a different task; it is fine for monitor TRAINING, not for calibration.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    names = list(fractions)
    cuts = np.cumsum([int(round(fractions[n] * len(eps))) for n in names[:-1]])
    return {n: [eps[i] for i in part] for n, part in zip(names, np.split(idx, cuts))}


def split_reference(eps: list[LoadedEpisode], fractions: dict[str, float], seed: int) -> dict[str, list]:
    """Reference split: 'train' is taken per task (every task represented for the monitor); the
    remaining episodes are split into 'calib' / 'test' by a pooled random split (exchangeable)."""
    tr = fractions["train"]
    first = split_by_task(eps, {"train": tr, "rest": 1 - tr}, seed)
    rest = {k: v / (1 - tr) for k, v in fractions.items() if k != "train"}
    out = split_pooled(first["rest"], rest, seed + 1)
    out["train"] = first["train"]
    return out
