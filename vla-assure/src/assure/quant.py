"""Weight-only fake quantization (quantize -> dequantize in place) for any PyTorch VLA.

Why fake quantization: it reproduces the numerical effect of INT8/INT4 weights on the model's
hidden states and actions -- which is what can break a monitor -- without needing an integer
kernel for every architecture. It does NOT give memory or latency savings; report those from a
real backend (bitsandbytes / torchao / QuantVLA) in a separate efficiency table and say so.

Scheme: symmetric round-to-nearest, per output channel, in groups of `group_size` input columns
(the standard GPTQ/AWQ grouping). Only nn.Linear weights are touched by default.
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


@dataclass
class QuantSpec:
    bits: int = 4
    group_size: int = 128
    include: list[str] = field(default_factory=lambda: ["*"])   # glob on module names
    exclude: list[str] = field(default_factory=list)            # substring or glob
    protect: list[str] = field(default_factory=list)            # exact module-name prefixes kept full precision


@torch.no_grad()
def fake_quant_weight(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Symmetric per-group RTN. w: (out, in). Returns dequantized tensor of w's dtype."""
    orig_dtype, orig_shape = w.dtype, w.shape
    x = w.float()
    out_f, in_f = x.shape
    g = group_size if group_size > 0 and in_f % group_size == 0 else in_f
    x = x.reshape(out_f, in_f // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return (q * scale).reshape(orig_shape).to(orig_dtype)


def _matches(name: str, patterns: list[str]) -> bool:
    for p in patterns:
        if any(c in p for c in "*?[") and fnmatch.fnmatch(name, p):
            return True
        if p in name:
            return True
    return False


@torch.no_grad()
def apply_fake_quant(model: nn.Module, spec: QuantSpec) -> dict:
    """Quantize matching Linear layers in place. Returns a report for the run log."""
    n_q, n_skip, params_q, params_total = 0, 0, 0, 0
    skipped: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        params_total += mod.weight.numel()
        keep = (
            not any(fnmatch.fnmatch(name, p) for p in spec.include)
            or _matches(name, spec.exclude)
            or any(name == p or name.startswith(p + ".") for p in spec.protect)
        )
        if keep:
            n_skip += 1
            skipped.append(name)
            continue
        mod.weight.copy_(fake_quant_weight(mod.weight, spec.bits, spec.group_size))
        n_q += 1
        params_q += mod.weight.numel()
    # Nominal memory of Linear weights if they were really stored at `bits` (scales ignored).
    frac = params_q / max(params_total, 1)
    return {
        "bits": spec.bits,
        "group_size": spec.group_size,
        "linear_quantized": n_q,
        "linear_kept_fp": n_skip,
        "param_fraction_quantized": round(frac, 4),
        "nominal_linear_weight_mem_ratio_vs_fp16": round(frac * spec.bits / 16 + (1 - frac), 4),
        "kept_examples": skipped[:20],
    }


def blocks_of(model: nn.Module, pattern: str = r"layers\.(\d+)$") -> list[str]:
    """Names of repeated transformer blocks (used by the sensitivity sweep)."""
    rx = re.compile(pattern)
    return [n for n, _ in model.named_modules() if rx.search(n)]


def load_protect_list(path: str | Path, top_k: int) -> list[str]:
    """Read results/sensitivity/<model>.json (written by 05_sensitivity.py) and return top-k blocks."""
    data = json.loads(Path(path).read_text())
    ranked = sorted(data["blocks"], key=lambda b: b["feature_drift"], reverse=True)
    return [b["name"] for b in ranked[:top_k]]
