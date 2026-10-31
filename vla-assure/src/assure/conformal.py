"""Time-varying conformal alarm band (functional split-conformal prediction).

WHAT IS GUARANTEED, AND WHAT IS NOT
The band controls exactly one quantity: the probability that a SUCCESSFUL episode raises at least
one alarm (the episode-level false-alarm rate). It says nothing about how many failures are caught,
how early, or whether the robot is safe. It holds only if the calibration episodes and the new
episode are exchangeable -- same policy weights, same task mix, same conditions -- which is exactly
what a model change breaks.

Two guarantee types:
  marginal  P(alarm on a new success) <= alpha, averaged over the draw of the calibration set.
            Needs n_conf >= ceil(1/alpha) - 1 conformal episodes (9 at alpha=0.1, 19 at 0.05).
  pac       P_calibration( false-alarm rate > alpha ) <= delta  ("training-conditional",
            Vovk 2012). Needs more episodes, but holds for the one calibration set you actually
            deployed with -- the honest statement for a release gate.

Procedure (exact, see split_sizes / conformal_rank):
  1. Take the N successful calibration episodes; permute them with a seeded RNG.
  2. Shape set: the first n_shape = max(min_shape, ceil(shape_frac * N)) episodes (capped so n_conf >= 1)
     fix a per-query centre mu_k and spread sd_k. Threshold set: the remaining n_conf = N - n_shape.
  3. Each threshold episode gives M_i = max_k (s_ik - mu_k)/sd_k; sort ascending, M_(1) <= ... <= M_(n).
  4. q = M_(r) with r from conformal_rank(n_conf, alpha, guarantee, delta). Alarm iff M_new > q,
     i.e. at the first k with s_k > mu_k + q sd_k.
Validity: conditional on the shape set, the threshold episodes and a new success are exchangeable,
so the band (mu, sd) is a fixed function and M is an exchangeable score.
  marginal: P(M_new > M_(r)) <= 1 - r/(n+1) <= alpha for r = ceil((n+1)(1-alpha)).
  pac:      F(M_(r)) >= U_(r) stochastically, U_(r) ~ Beta(r, n+1-r), so the false-alarm rate
            1 - F(M_(r)) is stochastically <= Beta(n+1-r, r); r is the smallest rank with
            P(Beta(n+1-r, r) <= alpha) = P(Bin(n, alpha) >= n+1-r) >= 1 - delta.
Ties: the alarm uses a strict '>' against an observed order statistic, so ties can only lower the
false-alarm rate; both bounds stay valid (conservative) without randomised tie-breaking.
Population: successful episodes of the calibrated policy under the calibration task mix, initial-state
distribution and camera condition. Nothing is claimed for failures, other tasks or other weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _pad(seqs: list[np.ndarray]) -> np.ndarray:
    K = max(len(s) for s in seqs)
    out = np.full((len(seqs), K), np.nan)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def _binom_sf(n: int, p: float, k: int) -> float:
    """P(Binomial(n, p) >= k)."""
    return float(sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(max(k, 0), n + 1)))


def conformal_rank(n: int, alpha: float, guarantee: str = "marginal", delta: float = 0.1) -> int | None:
    """Rank r (1-based) of the order statistic to use as q, or None if n is too small.

    marginal: r = ceil((n+1)(1-alpha)).
    pac:      smallest r with P(FPR <= alpha) >= 1-delta, where for continuous scores
              FPR ~ Beta(n+1-r, r), and P(Beta(a,b) <= x) = P(Bin(n, x) >= a) with a+b-1 = n.
    """
    if guarantee == "marginal":
        r = math.ceil((n + 1) * (1 - alpha))
        return r if r <= n else None
    for r in range(1, n + 1):
        if _binom_sf(n, alpha, n + 1 - r) >= 1 - delta:
            return r
    return None


def split_sizes(n_success: int, shape_frac: float = 0.5, min_shape: int = 10) -> tuple[int, int]:
    """(n_shape, n_conf) used by FunctionalCP.fit for N successful episodes. Single source of truth."""
    n_shape = max(min_shape, math.ceil(shape_frac * n_success))
    n_shape = min(n_shape, n_success - 1)
    return n_shape, n_success - n_shape


def required_successes(alpha: float, shape_frac: float = 0.5, guarantee: str = "marginal",
                       delta: float = 0.1, min_shape: int = 10) -> dict:
    """Smallest N (successful calibration episodes) for which the split of `split_sizes` leaves a
    threshold set large enough for a valid band at `alpha`."""
    N = 2
    while True:
        n_shape, n_conf = split_sizes(N, shape_frac, min_shape)
        r = conformal_rank(n_conf, alpha, guarantee, delta) if n_conf >= 1 else None
        if r is not None:
            return {"n_conf": n_conf, "n_shape": n_shape, "total": N, "rank": r}
        N += 1


def pac_violation_prob(n_conf: int, r: int, alpha: float) -> float:
    """Exact P(false-alarm rate > alpha) for the rank-r band with continuous scores: P(Bin(n, alpha) < n+1-r)."""
    return 1.0 - _binom_sf(n_conf, alpha, n_conf + 1 - r)


def finite_population_fpr(n_conf: int, m_test: int, r: int, alpha: float) -> dict:
    """EXACT finite-population false-alarm behaviour of the rank-r threshold on a whole TEST SUBSET.

    Setting (randomisation model, conditional on the training slots): the n_conf threshold successes and
    the m_test test successes are a uniformly random split of n_conf + m_test fixed, distinct scores.
    The threshold is the r-th smallest threshold score; the test FPR is the fraction of the m_test test
    scores above it. If the threshold sits at global rank p, then P(p) = C(p-1, r-1) C(M-p, n-r) / C(M, n)
    (M = n + m) and FPR_test = (m - (p - r)) / m. Returns E[FPR_test] (marginal) and
    P(FPR_test > alpha) (the training-conditional violation probability), both exact for this model."""
    from math import comb, ceil  # noqa: F401
    n, m = n_conf, m_test
    M = n + m
    tot = comb(M, n)
    e = viol = 0.0
    for p in range(r, r + m + 1):
        w = comb(p - 1, r - 1) * comb(M - p, n - r) / tot
        fpr = (m - (p - r)) / m
        e += w * fpr
        viol += w * (fpr > alpha + 1e-12)
    return {"n_conf": n, "m_test": m, "rank": r, "alpha": alpha, "expected_test_fpr": e, "p_test_fpr_above_alpha": viol}


def finite_population_budget(alpha: float, delta: float, m_test: int, shape_frac: float = 0.5,
                             min_shape: int = 10, n_max: int = 400) -> dict:
    """Smallest number of successes N whose threshold split (split_sizes) admits a rank r with
    P(FPR_test > alpha) <= delta EXACTLY under the finite-population randomisation model for a test subset
    of m_test successes (finite_population_fpr). Compare required_successes(..., 'pac'), which is exact
    under i.i.d. exchangeability (an infinite target population)."""
    for N in range(11, n_max + 1):
        _, n = split_sizes(N, shape_frac, min_shape)
        for r in range(1, n + 1):
            v = finite_population_fpr(n, m_test, r, alpha)
            if v["p_test_fpr_above_alpha"] <= delta:
                return {"total": N, "n_threshold": n, "rank": r, **v}
    return {"total": None}


@dataclass
class FunctionalCP:
    alpha: float
    guarantee: str = "marginal"     # "marginal" | "pac"
    delta: float = 0.1              # only for "pac"
    shape_frac: float = 0.5
    min_shape: int = 10
    seed: int = 0
    strict: bool = False            # if too few episodes: True -> never alarm (q=inf); False -> use max
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    q: float | None = None
    n_cal: int = 0
    n_shape: int = 0
    guaranteed_alpha: float | None = None
    underpowered: bool = False

    def fit(self, success_scores: list[np.ndarray]) -> "FunctionalCP":
        if len(success_scores) < 4:
            raise ValueError("need >=4 successful calibration rollouts")
        idx = np.random.default_rng(self.seed).permutation(len(success_scores))
        # Same split as required_successes(); small pools keep >=2 shaping episodes so a band exists.
        # Exactly the stated rule for every N (no small-N fallback): n_shape = max(min_shape, ceil(N/2)),
        # capped at N - 1. E.g. N = 10 -> 9 shaping / 1 threshold -> best marginal level 1/2.
        n_shape, _ = split_sizes(len(idx), self.shape_frac, self.min_shape)
        shape_set = [success_scores[i] for i in idx[:n_shape]]
        conf_set = [success_scores[i] for i in idx[n_shape:]]
        S = _pad(shape_set)
        mu, sd = np.nanmean(S, 0), np.nanstd(S, 0)
        # Late query indices have few episodes: carry the last well-estimated value forward.
        counts = np.sum(~np.isnan(S), 0)
        good = counts >= max(2, int(0.2 * len(shape_set)))
        last = np.where(good)[0].max() if good.any() else 0
        mu[last + 1:] = mu[last]
        sd[last + 1:] = sd[last]
        floor = np.nanmedian(sd[: last + 1]) * 0.1 + 1e-8
        self.mu, self.sd = mu, np.maximum(sd, floor)

        M = np.sort([float(np.max((s - self._musd(len(s))[0]) / self._musd(len(s))[1])) for s in conf_set])
        n = len(M)
        r = conformal_rank(n, self.alpha, self.guarantee, self.delta)
        self.n_cal, self.n_shape, self.rank = n, n_shape, r
        if r is not None:
            self.q = float(M[r - 1])
            self.guaranteed_alpha, self.underpowered = self.alpha, False
        else:
            self.underpowered = True
            if self.strict:
                self.q = float("inf")             # no alarms: guarantee holds trivially, monitor is mute
                self.guaranteed_alpha = self.alpha
            else:
                self.q = float(M[-1])             # max: valid only at the coarser level 1/(n+1) (marginal)
                self.guaranteed_alpha = 1.0 / (n + 1) if self.guarantee == "marginal" else float("nan")
        return self

    def _musd(self, K: int) -> tuple[np.ndarray, np.ndarray]:
        pad = max(0, K - len(self.mu))
        mu = np.concatenate([self.mu, np.full(pad, self.mu[-1])])[:K]
        sd = np.concatenate([self.sd, np.full(pad, self.sd[-1])])[:K]
        return mu, sd

    def band(self, K: int) -> np.ndarray:
        mu, sd = self._musd(K)
        return mu + self.q * sd

    def first_alarm(self, s: np.ndarray) -> int | None:
        hit = np.where(s > self.band(len(s)))[0]
        return int(hit[0]) if len(hit) else None

    def margin(self, s: np.ndarray) -> float:
        """Max normalised excess over the band centre: a threshold-free episode score for AUROC."""
        mu, sd = self._musd(len(s))
        return float(np.max((s - mu) / sd))
