"""Feature taps: forward hooks that pool hidden states the monitors read.

A tap is attached to a named submodule (e.g. the VLM's final norm). Every time that module
runs, its output is mean-pooled over tokens and appended to a buffer. `pop()` returns the
pooled feature for the most recent policy call and clears the buffer.

Hooks only fire when a module is invoked as `module(...)`. Code that calls `module.forward(...)`
directly bypasses hooks, which is why we tap final *norm* layers (called as modules inside the
models we use) rather than the language-model wrapper. `scripts/00_inspect_modules.py` checks
every configured tap fires before you spend GPU hours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path like 'model.vlm.layers.3.norm' (integers index ModuleLists)."""
    mod = root
    for part in path.split("."):
        if part.isdigit():
            mod = mod[int(part)]  # type: ignore[index]
        else:
            if not hasattr(mod, part):
                raise AttributeError(f"'{type(mod).__name__}' has no submodule '{part}' (full path: {path})")
            mod = getattr(mod, part)
    return mod


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return None


def _pool(h: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B, D) mean over tokens; (B, D) unchanged."""
    h = h.detach().float()
    if h.dim() == 3:
        return h.mean(dim=1)
    if h.dim() == 2:
        return h
    return h.reshape(h.shape[0], -1)


@dataclass
class FeatureTap:
    """Collects pooled outputs from several named modules.

    mode="last" keeps the last call since the previous pop (e.g. final denoising step of an
    action expert); mode="mean" averages all calls (e.g. every decoded action token of OpenVLA).
    """

    taps: dict[str, str]
    mode: dict[str, str] = field(default_factory=dict)
    _buf: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    _handles: list = field(default_factory=list)
    fired: dict[str, int] = field(default_factory=dict)

    def attach(self, root: nn.Module) -> "FeatureTap":
        for name, path in self.taps.items():
            mod = resolve_module(root, path)
            self._buf[name] = []
            self.fired[name] = 0
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name: str):
        def hook(_mod, _inp, out):
            t = _first_tensor(out)
            if t is None:
                return
            self._buf[name].append(_pool(t).cpu())
            self.fired[name] += 1

        return hook

    def has_new(self) -> bool:
        return any(len(v) > 0 for v in self._buf.values())

    def pop(self) -> dict[str, torch.Tensor] | None:
        """Return {tap_name: (B, D)} for calls since last pop, or None if nothing fired."""
        if not self.has_new():
            return None
        out = {}
        for name, items in self._buf.items():
            if not items:
                continue
            if self.mode.get(name, "last") == "mean":
                out[name] = torch.stack(items, 0).mean(0)
            else:
                out[name] = items[-1]
            self._buf[name] = []
        return out

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
