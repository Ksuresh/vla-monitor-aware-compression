"""Turn a variant name/spec into a modified policy.

Named variants live in experiment.yaml. Mixed-precision variants can also be requested by name:

    w4p-<rule>-f<pct>     4-bit everywhere except blocks chosen by <rule>, kept in full precision
                          until <pct>% of Linear-weight parameters are protected (matched memory).
                          Fake quantization: the numerical effect only.
    tao4p-<rule>-f<pct>   the SAME protected set with real torchao int4 kernels (protected blocks stay
                          bf16), so the proposed configurations are measured end to end, not only
                          uniform INT8/INT4.

Rules (all compared at the SAME parameter budget, so memory is matched by construction):
    margin_greedy   interaction-aware variant: the greedy set from 02_sensitivity.py --greedy, where J of the
                    whole quantized model is recomputed after each protected block (budgets 5/10/20%)
    margin_drift    alarm-statistic-aware (a continuous surrogate; decision flips are logged, not optimised): blocks whose solo quantization moves the episode alarm
                    statistic M (what the conformal threshold sees) the most
    score_drift     monitor-score-aware: blocks whose solo quantization moves the monitor score most
    feature_drift   feature-aware: blocks whose solo quantization moves the monitored features most
    action_drift    task-aware: blocks whose solo quantization moves the predicted actions most
    weight_error    sensitivity-based (no data): largest relative weight quantization error
    edge            common heuristic: first and last blocks of each transformer stack
    random          control: random blocks
Data-driven rules read results/sensitivity/<model>.json (scripts/02_sensitivity.py; score_drift and
margin_drift need --ref_key so a reference-trained monitor and band exist).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from torch import nn

from .quant import QuantSpec, apply_fake_quant, blocks_of, fake_quant_weight

BUDGET_TOLERANCE = 0.0025  # all rules must protect between target - 0.25 points and target (sub-block fill)

RULES = ("margin_greedy", "margin_drift", "score_drift", "feature_drift", "action_drift", "weight_error", "edge", "random")
DATA_RULES = ("margin_greedy", "margin_drift", "score_drift", "feature_drift", "action_drift")
_DYN = re.compile(r"^(?P<kind>w|tao)(?P<bits>\d+)p-(?P<rule>[a-z_]+)-f(?P<pct>\d+)$")


def resolve(name: str, exp: dict) -> dict:
    """Variant spec for a configured or dynamic (w4p-rule-fNN) variant name."""
    scope = exp.get("quant_scope") or {}   # e.g. go/no-go: action expert only, blocks of the expert

    def scoped(spec):
        # a variant with its own "include" (e.g. whole-model or VLM+expert) ignores the go/no-go scope
        if spec.get("kind") in ("quant", "torchao") and scope and not spec.get("own_scope"):
            spec = {**spec, **{k: scope[k] for k in ("include", "block_regex") if k in scope}}
        return spec
    if name in exp["variants"]:
        return scoped(dict(exp["variants"][name]))
    m = _DYN.match(name)
    if not m or m["rule"] not in RULES:
        raise KeyError(f"unknown variant {name}; configured: {list(exp['variants'])}; dynamic: w4p-<rule>-f<pct>")
    return scoped({"kind": "quant" if m["kind"] == "w" else "torchao", "bits": int(m["bits"]), "group_size": 128,
                   "protect_rule": m["rule"], "protect_frac": int(m["pct"]) / 100})


def checkpoint_for(model_cfg: dict, variant: dict, suite: str) -> str:
    """Pick the checkpoint for this variant (the 'update' variant swaps to a sibling checkpoint)."""
    ck = model_cfg["update_checkpoint"] if variant.get("kind") == "checkpoint_swap" else model_cfg["checkpoint"]
    if variant.get("kind") == "lora":
        ck = variant.get("path") or ck  # set from 05_lora_update.sh output
    if ck is None:
        raise ValueError("variant needs update_checkpoint in models.yaml (or a LoRA run path)")
    return ck.replace("{suite}", suite.replace("libero_", ""))


def _block_params(policy: nn.Module, blocks: list[str]) -> dict[str, int]:
    mods = dict(policy.named_modules())
    return {b: sum(m.weight.numel() for m in mods[b].modules() if isinstance(m, nn.Linear)) for b in blocks}


def _weight_error(policy: nn.Module, blocks: list[str], bits: int, group: int) -> dict[str, float]:
    mods, out = dict(policy.named_modules()), {}
    for b in blocks:
        num = den = 0.0
        for m in mods[b].modules():
            if isinstance(m, nn.Linear):
                w = m.weight.detach()
                num += float((fake_quant_weight(w, bits, group) - w).float().pow(2).sum())
                den += float(w.float().pow(2).sum())
        out[b] = num / max(den, 1e-12)
    return out


def _edge_score(blocks: list[str]) -> dict[str, float]:
    """Score 1.0 for the first/last block of each stack, decaying inward."""
    stacks: dict[str, list[tuple[int, str]]] = {}
    for b in blocks:
        prefix, idx = b.rsplit(".", 1)
        stacks.setdefault(prefix, []).append((int(idx), b))
    out = {}
    for items in stacks.values():
        n = max(i for i, _ in items)
        for i, b in items:
            out[b] = 1.0 / (1 + min(i, n - i))
    return out


def _in_scope(name: str, include: list[str] | None) -> bool:
    import fnmatch
    return include is None or any(fnmatch.fnmatch(name, p) for p in include)


def scope_total(policy: nn.Module, include: list[str] | None = None) -> int:
    """Linear weight parameters inside the quantization scope (whole model when include is None).
    Budgets (5/10/20%) are fractions of THIS total, so an expert-only scope protects a share of the expert."""
    return sum(m.weight.numel() for n, m in policy.named_modules()
               if isinstance(m, nn.Linear) and _in_scope(n, include))


def fill_budget(policy: nn.Module, order: list[str], frac: float, include: list[str] | None = None) -> list[str]:
    """Protect whole blocks in `order` while they fit, then complete the budget with single Linear layers
    of the next blocks in the same order (largest first), never exceeding it. Blocks alone are too coarse
    (one block can be 2-3% of the parameters), so this sub-block completion is what lets every rule reach
    the same budget within BUDGET_TOLERANCE (0.25 percentage points)."""
    mods = dict(policy.named_modules())
    params = _block_params(policy, order)
    total = scope_total(policy, include)
    cap = frac * total
    chosen, used = [], 0
    for b in order:
        if used + params[b] <= cap:
            chosen.append(b)
            used += params[b]
    for b in order:
        if b in chosen:
            continue
        lins = sorted(((f"{b}.{n}" if n else b, m.weight.numel()) for n, m in mods[b].named_modules()
                       if isinstance(m, nn.Linear)), key=lambda x: -x[1])
        for name, k in lins:
            if used + k <= cap:
                chosen.append(name)
                used += k
        if cap - used < min((k for _, k in lins), default=0):
            continue
    return chosen


def protected_breakdown(policy: nn.Module, protect: list[str], include: list[str] | None = None,
                        block_regex: str | None = None) -> dict:
    """How much of the protected budget is whole ranked blocks vs sub-block filler layers. The filler is a
    size-based heuristic, not part of the ranking, so its share is reported for every method."""
    total = scope_total(policy, include)
    blocks = set(blocks_of(policy, block_regex) if block_regex else blocks_of(policy))
    whole = [p for p in protect if p in blocks]
    filler = [p for p in protect if p not in blocks]
    pw = sum(_block_params(policy, whole).values()) if whole else 0
    pf = sum(_block_params(policy, filler).values()) if filler else 0
    return {"protected_frac_blocks": pw / max(total, 1), "protected_frac_filler": pf / max(total, 1),
            "filler_share": pf / max(pw + pf, 1), "n_whole_blocks": len(whole), "n_filler_layers": len(filler)}


def protected_fraction(policy: nn.Module, protect: list[str], include: list[str] | None = None) -> float:
    total = scope_total(policy, include)
    return (sum(_block_params(policy, protect).values()) if protect else 0) / max(total, 1)


def choose_protected(policy: nn.Module, model_name: str, rule: str, frac: float, bits: int = 4,
                     group: int = 128, seed: int = 0, sensitivity_dir: Path | None = None,
                     block_regex: str | None = None, include: list[str] | None = None) -> list[str]:
    """Rank blocks by `rule` (score per parameter), then fill_budget(). Every rule uses the same filling
    procedure, so all methods protect the target fraction to within BUDGET_TOLERANCE; the actual fraction
    is reported (protected_frac_actual) and checked by 04_report.py.
    Data rules use the SELECTION-subset score (e.g. margin_drift_sel) when 02_sensitivity.py wrote one, so
    the held-out captured episodes stay untouched by selection."""
    blocks = blocks_of(policy, block_regex) if block_regex else blocks_of(policy)
    params = _block_params(policy, blocks)

    def sens():
        from .common import ROOT
        path = (sensitivity_dir or ROOT / "results/sensitivity") / f"{model_name}.json"
        return path, json.loads(path.read_text())

    if rule == "margin_greedy":
        path, js = sens()
        g = (js.get("greedy") or {}).get(str(int(round(frac * 100))))
        if g is None:
            raise KeyError(f"{path}: no greedy set for {frac:.0%}; run 02_sensitivity.py --greedy")
        data = {b["name"]: b for b in js["blocks"]}
        key = "margin_drift_sel" if any("margin_drift_sel" in v for v in data.values()) else "margin_drift"
        rest = sorted((b for b in blocks if b in data and b not in g),
                      key=lambda b: data[b][key] / max(params[b], 1), reverse=True)
        return fill_budget(policy, list(g) + rest, frac, include)
    if rule == "random":
        order = blocks[:]
        random.Random(seed).shuffle(order)
    else:
        if rule in DATA_RULES:
            path, js = sens()
            data = {b["name"]: b for b in js["blocks"]}
            key = f"{rule}_sel" if any(f"{rule}_sel" in v for v in data.values()) else rule
            missing = [b for b in blocks if b in data and key not in data[b]]
            if missing:
                raise KeyError(f"{path} has no '{key}' scores; rerun 02_sensitivity.py with --ref_key")
            score = {b: data[b][key] for b in blocks if b in data}
        elif rule == "weight_error":
            score = _weight_error(policy, blocks, bits, group)
        else:
            score = _edge_score(blocks)
        order = sorted(score, key=lambda b: score[b] / max(params[b], 1), reverse=True)
    return fill_budget(policy, order, frac, include)


def apply_torchao(policy: nn.Module, bits: int, group_size: int = 128, protect: list[str] | None = None,
                  include: list[str] | None = None) -> dict:
    """Real weight-only quantization with torchao kernels (int8 / int4), for end-to-end efficiency
    numbers on LeRobot policies. API names differ across torchao releases; both are tried.
    int4 kernels need CUDA and bf16 weights. Verify on first run with scripts/06_efficiency.py."""
    from torchao.quantization import quantize_
    try:  # torchao >= 0.10 config API
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
        cfg = Int8WeightOnlyConfig() if bits == 8 else Int4WeightOnlyConfig(group_size=group_size)
    except ImportError:  # older function API
        from torchao.quantization import int4_weight_only, int8_weight_only
        cfg = int8_weight_only() if bits == 8 else int4_weight_only(group_size=group_size)
    if bits == 4:
        policy.to(torch.bfloat16)
    protect = list(protect or [])

    def keep_low_bit(mod, fqn):  # real mixed precision: protected blocks stay in bf16/fp16
        return (isinstance(mod, nn.Linear) and _in_scope(fqn, include)
                and not any(fqn == p or fqn.startswith(p + ".") for p in protect))
    quantize_(policy, cfg, filter_fn=keep_low_bit)
    return {"kind": "torchao", "bits": bits, "group_size": group_size, "real_kernels": True,
            "protected_blocks": protect}


def apply_variant(policy: nn.Module, model_name: str, variant: dict) -> dict:
    """Mutates `policy` in place for quant variants; returns a report dict for the run log."""
    kind = variant.get("kind", "reference")
    if kind == "torchao":
        protect = []
        if variant.get("protect_rule"):
            protect = choose_protected(policy, model_name, variant["protect_rule"], float(variant["protect_frac"]),
                                       int(variant["bits"]), int(variant.get("group_size", 128)),
                                       int(variant.get("protect_seed", 0)), block_regex=variant.get("block_regex"),
                                       include=variant.get("include"))
        rep = apply_torchao(policy, int(variant["bits"]), int(variant.get("group_size", 128)), protect,
                            include=variant.get("include"))
        rep.update(protect_rule=variant.get("protect_rule"), protect_frac=variant.get("protect_frac"),
                   protected_frac_actual=(sum(_block_params(policy, protect).values()) if protect else 0) /
                   max(sum(m.weight.numel() for m in policy.modules() if isinstance(m, nn.Linear)), 1))
        return rep
    if kind != "quant":
        return {"kind": kind}
    protect: list[str] = []
    if variant.get("protect_rule"):
        protect = choose_protected(policy, model_name, variant["protect_rule"], float(variant["protect_frac"]),
                                   int(variant["bits"]), int(variant.get("group_size", 128)),
                                   int(variant.get("protect_seed", 0)), block_regex=variant.get("block_regex"),
                                   include=variant.get("include"))
    spec = QuantSpec(bits=int(variant["bits"]), group_size=int(variant.get("group_size", 128)),
                     include=list(variant.get("include") or ["*"]),
                     exclude=list(variant.get("exclude", [])), protect=protect)
    rep = apply_fake_quant(policy, spec)
    total = scope_total(policy, variant.get("include"))
    used = sum(_block_params(policy, protect).values()) if protect else 0
    rep.update(kind="quant", protect_rule=variant.get("protect_rule"), protect_frac=variant.get("protect_frac"),
               quant_scope=variant.get("include") or ["*"], protected_blocks=protect, protected_frac_actual=used / max(total, 1),
               **(protected_breakdown(policy, protect, variant.get("include"), variant.get("block_regex"))
                  if protect else {}))
    return rep
