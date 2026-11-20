"""Real low-bit path (bitsandbytes): replaces only in-scope Linear layers and stores 4-bit weights."""
import copy

import pytest
import torch
from torch import nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
bnb = pytest.importorskip("bitsandbytes")
from assure.variants import apply_bnb  # noqa: E402


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Sequential(nn.Linear(256, 256))
        self.lm_expert = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, x):
        return self.lm_expert(self.vlm(x))


def test_apply_bnb_scope_and_storage():
    torch.manual_seed(0)
    m = _Toy().eval()
    x = torch.randn(4, 256)
    ref = m(x)
    q = copy.deepcopy(m)
    rep = apply_bnb(q, 4, ["lm_expert.*"])
    assert rep["linear_replaced"] == 2 and isinstance(q.vlm[0], nn.Linear)
    assert isinstance(q.lm_expert[0], bnb.nn.Linear4bit)
    assert q.lm_expert[0].weight.dtype == torch.float32      # storage dtype matches the layer (LeRobot casts to it)
    assert rep["scoped_linear_weight_bytes_after"] * 4 < rep["scoped_linear_weight_bytes_before"]
    with torch.no_grad():
        out = q(x)
    assert 0 < float((out - ref).norm() / ref.norm()) < 0.5


def test_apply_bnb_empty_scope_raises():
    with pytest.raises(RuntimeError):
        apply_bnb(_Toy(), 4, ["nothing.*"])
