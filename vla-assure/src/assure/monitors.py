"""Runtime failure monitors, re-implemented compactly so every model shares one code path.

Each monitor maps an episode to a score sequence s_k over policy queries k (higher = more
likely failing). Calibration into alarms is done separately (conformal.py), so all monitors are
compared under the same alarm rule.

  safe_mlp / safe_lstm  SAFE-style hidden-state probes (Gu et al., NeurIPS 2025). Trained on
                        success AND failure rollouts.
  mahalanobis           Last-layer Mahalanobis distance, as in VLA-FAIL's LLMD. Success-only.
  acc                   Action-chunk consistency between overlapping chunks (VLA-FAIL's ACC).
                        Needs chunked policies that replan before a chunk is used up.
  rnd                   Random-network-distillation novelty on embeddings (FIPER's observation
                        score). Success-only.
  fiper                 FIPER-style conjunction of rnd and action-sampling spread; needs chunk_alt.

These are faithful-in-spirit re-implementations, not the authors' code. For headline numbers,
also run the official SAFE / FIPER repos on the same rollouts (scripts/03 writes compatible files).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .store import LoadedEpisode


# ---------------------------------------------------------------- feature helpers
def episode_features(ep: LoadedEpisode, taps: list[str]) -> np.ndarray:
    return np.concatenate([ep.feats[t] for t in taps], axis=-1)  # (K, D)


@dataclass
class Standardizer:
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


class Monitor:
    name = "base"
    needs_failures = False

    def __init__(self, taps: list[str]):
        self.taps = taps
        self.std = Standardizer()
        # Optional per-feature transform applied BEFORE standardization (used by the CORAL fix).
        self.feature_transform = None

    def feats(self, ep: LoadedEpisode) -> np.ndarray:
        X = episode_features(ep, self.taps)
        if self.feature_transform is not None:
            X = self.feature_transform(X)
        return X

    def fit(self, eps: list[LoadedEpisode]) -> "Monitor":
        raise NotImplementedError

    def score(self, ep: LoadedEpisode) -> np.ndarray:
        raise NotImplementedError

    def available(self, ep: LoadedEpisode) -> bool:
        return all(t in ep.feats for t in self.taps)


# ---------------------------------------------------------------- SAFE-style probes
class _MLP(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _LSTM(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.lstm = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, 1)

    def forward(self, x):  # (B, K, D)
        y, _ = self.lstm(torch.relu(self.inp(x)))
        return self.out(y).squeeze(-1)


class SafeProbe(Monitor):
    needs_failures = True

    def __init__(self, taps, kind="mlp", epochs=200, lr=1e-3, wd=1e-2, seed=0, device="cpu"):
        super().__init__(taps)
        self.kind, self.epochs, self.lr, self.wd, self.seed, self.device = kind, epochs, lr, wd, seed, device
        self.name = f"safe_{kind}"
        self.net = None

    def fit(self, eps):
        torch.manual_seed(self.seed)
        Xs = [self.feats(e) for e in eps]
        self.std.fit(np.concatenate(Xs))
        Xs = [self.std(X).astype(np.float32) for X in Xs]
        ys = [float(not e.success) for e in eps]  # 1 = failure
        d = Xs[0].shape[1]
        self.net = (_MLP(d) if self.kind == "mlp" else _LSTM(d)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        pos = max(sum(ys), 1.0)
        pw = torch.tensor((len(ys) - pos) / pos, device=self.device)  # balance success/failure
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.net.train()
        for _ in range(self.epochs):
            perm = np.random.default_rng(self.seed + _).permutation(len(Xs))
            for i in perm:
                x = torch.from_numpy(Xs[i]).to(self.device)
                y = torch.full((x.shape[0],), ys[i], device=self.device)
                logit = self.net(x) if self.kind == "mlp" else self.net(x[None])[0]
                loss = lossf(logit, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, ep):
        x = torch.from_numpy(self.std(self.feats(ep)).astype(np.float32)).to(self.device)
        if self.kind == "mlp":
            p = torch.sigmoid(self.net(x)).cpu().numpy()
            return np.cumsum(p) / np.arange(1, len(p) + 1)  # running mean, bounded in (0,1)
        return torch.sigmoid(self.net(x[None])[0]).cpu().numpy()


# ---------------------------------------------------------------- Mahalanobis (LLMD-style)
class Mahalanobis(Monitor):
    name = "mahalanobis"

    def __init__(self, taps, shrinkage=0.1, rank=256):
        super().__init__(taps)
        self.shrinkage, self.rank = shrinkage, rank

    def fit(self, eps):
        X = np.concatenate([self.feats(e) for e in eps if e.success])
        self.std.fit(X)
        Z = self.std(X)
        # PCA to `rank` dims keeps the covariance well conditioned when D >> #samples.
        self.mean = Z.mean(0)
        U, S, Vt = np.linalg.svd(Z - self.mean, full_matrices=False)
        r = min(self.rank, Vt.shape[0])
        self.P = Vt[:r].T
        Y = (Z - self.mean) @ self.P
        C = np.cov(Y, rowvar=False)
        C = (1 - self.shrinkage) * C + self.shrinkage * np.trace(C) / r * np.eye(r)
        self.Ci = np.linalg.inv(C)
        return self

    def score(self, ep):
        Y = (self.std(self.feats(ep)) - self.mean) @ self.P
        d = np.sqrt(np.einsum("kd,de,ke->k", Y, self.Ci, Y))
        return np.cumsum(d) / np.arange(1, len(d) + 1)


# ---------------------------------------------------------------- action-chunk consistency
class ActionChunkConsistency(Monitor):
    name = "acc"

    def __init__(self, taps=None):
        super().__init__(taps or [])

    def available(self, ep):
        return ep.chunk is not None and ep.chunk.shape[0] > 1 and ep.chunk.shape[1] > 1 and self._overlaps(ep)

    @staticmethod
    def _overlaps(ep):
        H = ep.chunk.shape[1]
        return bool(np.any(np.diff(ep.query_step) < H))

    def fit(self, eps):
        A = np.concatenate([e.actions for e in eps if e.success])
        self.scale = np.abs(np.diff(A, axis=0)).mean(0) + 1e-6  # typical per-dim step change
        return self

    def score(self, ep):
        C, q = ep.chunk, ep.query_step
        H = C.shape[1]
        s = [0.0]
        for k in range(1, len(q)):
            off = int(q[k] - q[k - 1])
            if off >= H:
                s.append(s[-1])
                continue
            prev = C[k - 1, off:]
            cur = C[k, : H - off]
            s.append(float(np.mean(np.abs(prev - cur) / self.scale)))
        s = np.asarray(s)
        return np.cumsum(s) / np.arange(1, len(s) + 1)


# ---------------------------------------------------------------- RND novelty (FIPER obs score)
class RND(Monitor):
    name = "rnd"

    def __init__(self, taps, out=128, epochs=300, lr=1e-3, seed=0):
        super().__init__(taps)
        self.out, self.epochs, self.lr, self.seed = out, epochs, lr, seed

    def fit(self, eps):
        torch.manual_seed(self.seed)
        X = np.concatenate([self.feats(e) for e in eps if e.success])
        self.std.fit(X)
        Z = torch.from_numpy(self.std(X).astype(np.float32))
        d = Z.shape[1]
        self.target = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, self.out))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, self.out))
        opt = torch.optim.Adam(self.pred.parameters(), lr=self.lr)
        with torch.no_grad():
            T = self.target(Z)
        for _ in range(self.epochs):
            loss = ((self.pred(Z) - T) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def score(self, ep):
        Z = torch.from_numpy(self.std(self.feats(ep)).astype(np.float32))
        e = ((self.pred(Z) - self.target(Z)) ** 2).mean(-1).numpy()
        return np.cumsum(e) / np.arange(1, len(e) + 1)


class FiperStyle(Monitor):
    """Conjunction of RND novelty and action-sample spread, each z-scored on successful train
    rollouts; score = min of the two, so the alarm needs BOTH signals high (FIPER's rule)."""

    name = "fiper"

    def __init__(self, taps):
        super().__init__(taps)
        self.rnd = RND(taps)

    def available(self, ep):
        return super().available(ep) and ep.chunk_alt is not None

    @staticmethod
    def _spread(ep):
        # chunk_alt: (K, S, H, A) -> per-query mean std across samples
        return ep.chunk_alt.std(axis=1).mean(axis=(1, 2))

    def fit(self, eps):
        self.rnd.feature_transform = self.feature_transform
        self.rnd.fit(eps)
        ok = [e for e in eps if e.success]
        r = np.concatenate([self.rnd.score(e) for e in ok])
        v = np.concatenate([self._spread(e) for e in ok])
        self.r_mu, self.r_sd = r.mean(), r.std() + 1e-6
        self.v_mu, self.v_sd = v.mean(), v.std() + 1e-6
        return self

    def score(self, ep):
        self.rnd.feature_transform = self.feature_transform
        r = (self.rnd.score(ep) - self.r_mu) / self.r_sd
        v = self._spread(ep)
        v = (np.cumsum(v) / np.arange(1, len(v) + 1) - self.v_mu) / self.v_sd
        return np.minimum(r, v)


EXTERNAL_TAPS = ["ext_pixels"]  # evidence that does not pass through the policy's weights

# Which monitors are defined for which policy architectures (declared per model in models.yaml).
# A monitor that needs something the architecture does not produce is reported as not applicable,
# never silently skipped or forced through a different definition.
REQUIREMENTS = {
    "acc": ("chunk_overlap", "needs overlapping action chunks (replanning before a chunk is used up); "
                             "single-step or execute-whole-chunk policies have nothing to compare"),
    "fiper": ("stochastic_actions", "needs several action samples per observation; deterministic heads "
                                    "(greedy tokens, L1 regression) have no sampling spread"),
}


def applicable(name: str, model_cfg: dict) -> tuple[bool, str]:
    req = REQUIREMENTS.get(name)
    if req is None:
        return True, ""
    flag, why = req
    arch = model_cfg.get("arch", {})
    return (bool(arch.get(flag, False)), "" if arch.get(flag) else why)


def build(name: str, taps: list[str], device: str = "cpu") -> Monitor:
    """Monitors prefixed 'ext_' read only policy-independent features; they are the
    independent-lineage control: compressing the policy should not move them."""
    return {
        "ext_mahalanobis": lambda: Mahalanobis(EXTERNAL_TAPS, rank=64),
        "ext_rnd": lambda: RND(EXTERNAL_TAPS),
        "safe_mlp": lambda: SafeProbe(taps, "mlp", device=device),
        "safe_lstm": lambda: SafeProbe(taps, "lstm", device=device),
        "mahalanobis": lambda: Mahalanobis(taps),
        "acc": lambda: ActionChunkConsistency(),
        "rnd": lambda: RND(taps),
        "fiper": lambda: FiperStyle(taps),
    }[name]()


# ---------------------------------------------------------------- independent monitor replicates
def parse_name(name: str) -> tuple[str, str | None]:
    """'mahalanobis#A' -> ('mahalanobis', 'A'). A suffix trains the monitor on one half of the reference
    training slots (seeded pooled split): '#A' is the half the block SELECTION uses; '#B' is an independent
    replicate on disjoint data that selection never sees, used to test whether a precision allocation
    chosen with monitor A also helps an independently trained monitor."""
    base, _, half = name.partition("#")
    return base, (half or None)


def train_subset(train: list, half: str | None, seed: int) -> list:
    if half is None:
        return train
    rng = np.random.default_rng([seed, 4242])
    idx = rng.permutation(len(train))
    cut = len(train) // 2
    keep = idx[:cut] if half.upper() == "A" else idx[cut:]
    return [train[i] for i in sorted(keep)]
