"""Seeded perturbations used to create a realistic range of failure rates.

Camera conditions (noise_std, brightness) corrupt observations; the action_noise condition
(action_noise_std) adds seeded Gaussian noise to EXECUTED arm actions (not the gripper, not the
logged policy output), a non-visual failure source used as a secondary robustness check.

Applied identically to every variant of a model, so differences between variants are not caused
by the perturbation itself. Each (episode, query) gets its own deterministic noise draw.
"""
from __future__ import annotations

import numpy as np
import torch


def condition_key(suite: str, condition: str) -> str:
    return suite if condition in (None, "", "nominal") else f"{suite}+{condition}"


class Perturber:
    def __init__(self, spec: dict | None):
        self.spec = spec or {}
        self.active = bool(self.spec)
        self.base_seed = int(self.spec.get("seed", 0))
        self.counter = 0

    def reset(self, episode_seed: int) -> None:
        self.counter = 0
        self.act_counter = 0
        self.episode_seed = episode_seed

    @property
    def acts(self) -> bool:
        return float(self.spec.get("action_noise_std", 0.0)) > 0

    def action(self, a):
        """Seeded per (episode, executed step) noise on all dims but the last (gripper). numpy or torch."""
        if not self.acts:
            return a
        self.act_counter += 1
        rng = np.random.default_rng([self.base_seed, getattr(self, "episode_seed", 0), 10**6 + self.act_counter])
        n = rng.normal(0, float(self.spec["action_noise_std"]), size=np.shape(a)[-1])
        n[-1] = 0.0
        if torch.is_tensor(a):
            return a + torch.as_tensor(n, dtype=a.dtype, device=a.device)
        return np.asarray(a) + n

    def _rng(self):
        self.counter += 1
        return np.random.default_rng([self.base_seed, getattr(self, "episode_seed", 0), self.counter])

    @property
    def visual(self) -> bool:
        return bool(self.spec.get("noise_std", 0.0) or self.spec.get("brightness", 0.0))

    def numpy_image(self, img: np.ndarray) -> np.ndarray:
        """uint8 HxWxC image (OpenVLA backend)."""
        if not self.visual:
            return img
        rng = self._rng()
        x = img.astype(np.float32) / 255.0
        x = x * (1 + rng.uniform(-1, 1) * self.spec.get("brightness", 0.0))
        x = x + rng.normal(0, self.spec.get("noise_std", 0.0), size=x.shape)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def torch_batch(self, batch: dict) -> dict:
        """LeRobot batch: perturbs every float image tensor in [0,1] whose key contains 'image'."""
        if not self.visual:
            return batch
        rng = self._rng()
        g = torch.Generator().manual_seed(int(rng.integers(2**31)))
        b = float(rng.uniform(-1, 1) * self.spec.get("brightness", 0.0))
        out = dict(batch)
        for k, v in batch.items():
            if "image" in k and torch.is_tensor(v) and v.is_floating_point():
                noise = torch.randn(v.shape, generator=g).to(v.device, v.dtype) * self.spec.get("noise_std", 0.0)
                y = v * (1 + b) + noise
                in_unit = bool(v.min() >= 0) and bool(v.max() <= 1)
                out[k] = y.clamp(0, 1) if in_unit else y
        return out
