"""Cheap ways to restore a monitor after the policy changes.

recal_success   Keep the probe; re-run conformal calibration on n SUCCESSFUL rollouts of the
                new model. Needs no failure data -- the realistic option at release time.
coral           Keep probe and band; map the new model's features onto the reference feature
                statistics (per-dimension mean/std, label-free, from n rollouts of any outcome).
retrain_supervised  Retrain the probe and recalibrate on new-model rollouts including failures. A fully
                supervised reference baseline -- not an upper bound: with few failures, retraining can
                do worse than the other repairs. Collecting failures is costly and unsafe on robots.
"""
from __future__ import annotations

import numpy as np

from .monitors import episode_features


class DiagCoral:
    def __init__(self, taps):
        self.taps = taps

    def fit(self, ref_eps, new_eps):
        R = np.concatenate([episode_features(e, self.taps) for e in ref_eps])
        N = np.concatenate([episode_features(e, self.taps) for e in new_eps])
        self.mr, self.sr = R.mean(0), R.std(0) + 1e-6
        self.mn, self.sn = N.mean(0), N.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mn) / self.sn * self.sr + self.mr


def feature_drift(ref_eps, new_eps, taps) -> dict:
    """How far the new model's features moved: cosine of means and relative mean shift."""
    R = np.concatenate([episode_features(e, taps) for e in ref_eps])
    N = np.concatenate([episode_features(e, taps) for e in new_eps])
    mr, mn = R.mean(0), N.mean(0)
    cos = float(mr @ mn / (np.linalg.norm(mr) * np.linalg.norm(mn) + 1e-12))
    shift = float(np.linalg.norm(mn - mr) / (np.linalg.norm(R.std(0)) + 1e-12))
    return {"mean_cosine": cos, "mean_shift_in_sd_units": shift}
