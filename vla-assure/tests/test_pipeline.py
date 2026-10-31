"""CPU-only checks of the analysis pipeline on synthetic rollouts (no simulator, no GPU).

These validate the plumbing and the statistics -- NOT any claim about real VLAs.
    pytest -q tests/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assure import monitors as M  # noqa: E402
from assure.conformal import FunctionalCP  # noqa: E402
from assure.fixes import DiagCoral  # noqa: E402
from assure.hooks import FeatureTap  # noqa: E402
from assure.metrics import auroc, evaluate  # noqa: E402
from assure.quant import QuantSpec, apply_fake_quant, fake_quant_weight  # noqa: E402
from assure.store import Episode, load_cell, split_by_task  # noqa: E402

D, H, A = 32, 10, 7


def synth_episode(rng, model, variant, key, task, idx, fail, shift=0.0, scale=1.0, shadow=None):
    """Successes: stationary features. Failures: features drift and chunks disagree over time."""
    K = 20
    ep = Episode(model, variant, key, task, idx, seed=idx)
    ep.success = not fail
    ep.length = K * 5
    base = rng.normal(0, 1, D)
    for k in range(K):
        drift = (k / K) * 2.5 if fail else 0.0
        f = (base * 0.3 + rng.normal(0, 1, D) + drift) * scale + shift
        chunk = rng.normal(0, 0.05, (H, A)) + (rng.normal(0, 0.4 * k / K, (H, A)) if fail else 0)
        # ext_pixels: camera evidence that does not depend on the policy's weights (no shift/scale)
        px = base * 0.3 + rng.normal(0, 1, D) + ((k / K) * 2.0 if fail else 0.0)
        ep.add_query(k * 5, {"vlm": f.astype(np.float32), "ext_pixels": px.astype(np.float32)},
                     chunk.astype(np.float32))
        if shadow:  # shadow copy: same observation, features through a shifted "model"
            sname, sshift, sscale = shadow
            ep.add_shadow(sname, {"vlm": (f * sscale + sshift).astype(np.float32)}, chunk.astype(np.float32))
    ep.actions = rng.normal(0, 0.05, (ep.length, A)).tolist()
    return ep


def make_cell(tmp, rng, variant, n_per_task=12, fail_rate=0.3, outcomes=None, **kw):
    """`outcomes` (dict pair_index -> fail) forces paired outcomes across variants."""
    for t in range(4):
        for i in range(n_per_task):
            fail = outcomes[(t, i)] if outcomes is not None else rng.random() < fail_rate
            synth_episode(rng, "toy", variant, "suite", t, i, fail, **kw).save(tmp)


def test_conformal_controls_false_alarms_on_exchangeable_successes():
    rng = np.random.default_rng(0)
    seqs = [np.cumsum(rng.normal(0, 1, 30)) / np.arange(1, 31) for _ in range(2000)]
    cal, test = seqs[:400], seqs[400:]
    for alpha in (0.05, 0.1, 0.2):
        cp = FunctionalCP(alpha).fit(cal)
        fpr = np.mean([cp.first_alarm(s) is not None for s in test])
        assert fpr <= alpha + 0.03, (alpha, fpr)


def test_underpowered_calibration_is_flagged():
    rng = np.random.default_rng(1)
    seqs = [np.cumsum(rng.normal(0, 1, 10)) for _ in range(10)]
    cp = FunctionalCP(0.1).fit(seqs)          # N=10 -> 9 shaping / 1 threshold -> best bound 1/2
    assert cp.underpowered and cp.n_shape == 9 and abs(cp.guaranteed_alpha - 1 / 2) < 1e-9
    levels = [FunctionalCP(0.1).fit(seqs * 2 if n > 10 else seqs[:n]).guaranteed_alpha
              for n in (4, 6, 8, 10)]
    assert all(x >= y - 1e-12 for x, y in zip(levels, levels[1:]))  # never better with fewer episodes
    cp = FunctionalCP(0.1).fit(seqs * 4)      # 20 conformal rollouts -> alpha reachable
    assert not cp.underpowered


def test_auroc_basic():
    assert auroc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])) == 1.0
    assert abs(auroc(np.array([1, 1, 1, 1.0]), np.array([0, 1, 0, 1])) - 0.5) < 1e-9


def test_fake_quant_error_ordering_and_protect():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    e8 = (fake_quant_weight(w, 8, 128) - w).abs().mean()
    e4 = (fake_quant_weight(w, 4, 128) - w).abs().mean()
    assert 0 < e8 < e4
    net = nn.Sequential()
    net.add_module("layers", nn.ModuleList([nn.Linear(256, 256) for _ in range(12)]))
    ref = {n: p.clone() for n, p in net.named_parameters()}
    rep = apply_fake_quant(net, QuantSpec(bits=4, protect=["layers.1"]))
    # 'layers.1' must be kept but 'layers.10'/'layers.11' must NOT be (prefix bug guard)
    assert torch.equal(net.layers[1].weight, ref["layers.1.weight"])
    assert not torch.equal(net.layers[10].weight, ref["layers.10.weight"])
    assert rep["linear_kept_fp"] == 1


def test_feature_tap_fires_and_pools():
    net = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16))
    tap = FeatureTap({"norm": "1"}).attach(net)
    net(torch.randn(2, 5, 8))
    out = tap.pop()
    assert out["norm"].shape == (2, 16) and tap.fired["norm"] == 1
    assert tap.pop() is None


def test_monitors_detect_synthetic_failures(tmp_path):
    rng = np.random.default_rng(1)
    make_cell(tmp_path, rng, "fp16", n_per_task=30)
    eps = load_cell(tmp_path, "toy", "fp16", "suite")
    sp = split_by_task(eps, {"train": 0.5, "calib": 0.25, "test": 0.25}, 0)
    for name in ["safe_mlp", "safe_lstm", "mahalanobis", "acc", "rnd"]:
        m = M.build(name, ["vlm"])
        if isinstance(m, M.SafeProbe):
            m.epochs = 20
        if isinstance(m, M.RND):
            m.epochs = 100
        m.fit(sp["train"])
        cp = FunctionalCP(0.1).fit([m.score(e) for e in sp["calib"] if e.success])
        r = evaluate(cp, [m.score(e) for e in sp["test"]], [not e.success for e in sp["test"]],
                     [e.query_step for e in sp["test"]], [e.length for e in sp["test"]])
        assert r["auroc"] > 0.75, (name, r)


def test_shift_breaks_calibration_and_fixes_restore_it(tmp_path):
    rng = np.random.default_rng(2)
    make_cell(tmp_path, rng, "fp16", n_per_task=40)
    make_cell(tmp_path, rng, "w4", n_per_task=40, shift=0.8, scale=1.3)  # synthetic "compression" shift
    ref = load_cell(tmp_path, "toy", "fp16", "suite")
    new = load_cell(tmp_path, "toy", "w4", "suite")
    sp = split_by_task(ref, {"train": 0.5, "calib": 0.25, "test": 0.25}, 0)
    vs = split_by_task(new, {"adapt": 0.3, "test": 0.7}, 0)
    m = M.build("mahalanobis", ["vlm"]).fit(sp["train"])
    cp = FunctionalCP(0.1).fit([m.score(e) for e in sp["calib"] if e.success])

    def fpr(cp_, eps):
        s = [e for e in eps if e.success]
        return np.mean([cp_.first_alarm(m.score(e)) is not None for e in s])

    before = fpr(cp, vs["test"])
    # marginal guarantee = average over calibration draws, so average over split seeds
    cps_recal = [FunctionalCP(0.1, seed=k).fit([m.score(e) for e in vs["adapt"] if e.success]) for k in range(10)]
    m.feature_transform = DiagCoral(["vlm"]).fit(sp["train"], vs["adapt"])
    after_coral = fpr(cp, vs["test"])
    m.feature_transform = None
    after_recal = float(np.mean([fpr(c, vs["test"]) for c in cps_recal]))
    assert before > 0.3, before            # shift inflates false alarms
    assert after_recal <= 0.15, after_recal  # recalibration restores the target on average (alpha=0.1)
    assert after_coral < before, (after_coral, before)


def test_external_monitor_is_immune_to_policy_shift(tmp_path):
    rng = np.random.default_rng(4)
    make_cell(tmp_path, rng, "fp16", n_per_task=40)
    make_cell(tmp_path, rng, "w4", n_per_task=40, shift=0.8, scale=1.3)
    ref = load_cell(tmp_path, "toy", "fp16", "suite")
    new = load_cell(tmp_path, "toy", "w4", "suite")
    sp = split_by_task(ref, {"train": 0.5, "calib": 0.25, "test": 0.25}, 0)
    out = {}
    for name, taps in [("mahalanobis", ["vlm"]), ("ext_mahalanobis", None)]:
        m = M.build(name, taps or [])
        m.fit(sp["train"])
        cp = FunctionalCP(0.1).fit([m.score(e) for e in sp["calib"] if e.success])
        out[name] = np.mean([cp.first_alarm(m.score(e)) is not None for e in new if e.success])
    assert out["mahalanobis"] > 0.3 and out["ext_mahalanobis"] < 0.2, out


def test_end_to_end_scripts(tmp_path):
    rng = np.random.default_rng(3)
    roll = tmp_path / "rollouts"
    make_cell(roll, rng, "fp16", n_per_task=30)
    make_cell(roll, rng, "w4", n_per_task=30, shift=0.5)
    # 03_evaluate expects model names from models.yaml for taps; pass --taps explicitly instead.
    r = subprocess.run([sys.executable, str(ROOT / "scripts/03_evaluate.py"), "--model", "toy", "--key", "suite",
                        "--rollouts", str(roll), "--taps", "vlm", "--monitors", "mahalanobis,acc"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = ROOT / "results/eval/toy__suite.csv"
    assert out.exists()
    import pandas as pd
    df = pd.read_csv(out)
    assert {"none", "recal_success"} <= set(df.fix)
    assert {"deployment", "paired"} <= set(df.analysis.dropna())
    pr = df[(df.analysis == "paired") & (df.monitor == "mahalanobis")]
    assert pr.cmp_fpr_delta.notna().any() and pr.n_pairs.min() > 0
    assert (df.calib_unit.dropna() == "pooled_over_tasks").all()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/04_report.py"), "--rollouts", str(roll)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cm = pd.read_csv(ROOT / "results/tables/comparisons.csv")
    assert "verdict" not in cm and "p_worse_holm" not in cm          # per-cell tables are descriptive only
    pe = pd.read_csv(ROOT / "results/tables/primary_endpoint.csv")
    assert pe.verdict.str.startswith("descriptive only").all()        # 4 synthetic tasks < 20 clusters
    out.unlink()


def test_calibration_budgets_and_pac():
    from assure.conformal import required_successes
    assert required_successes(0.10)["n_conf"] == 9 and required_successes(0.05)["n_conf"] == 19
    assert required_successes(0.10, guarantee="pac", delta=0.1)["n_conf"] > 9
    rng = np.random.default_rng(0)
    pool = [np.cumsum(np.abs(rng.normal(0, 1, 15))) / np.arange(1, 16) for _ in range(20000)]
    viol = []
    for d in range(150):
        idx = rng.choice(20000, 80, replace=False)
        cp = FunctionalCP(0.1, guarantee="pac", delta=0.1, seed=d).fit([pool[i] for i in idx])
        te = [pool[i] for i in rng.choice(20000, 300)]
        viol.append(np.mean([cp.first_alarm(s) is not None for s in te]) > 0.1)
    assert np.mean(viol) <= 0.17  # delta = 0.1 plus Monte-Carlo slack


def test_shadow_counterfactual_and_paired(tmp_path):
    rng = np.random.default_rng(6)
    outcomes = {(t, i): rng.random() < 0.3 for t in range(4) for i in range(40)}
    make_cell(tmp_path, rng, "fp16", n_per_task=40, outcomes=outcomes, shadow=("w4", 0.8, 1.3))
    make_cell(tmp_path, rng, "w4", n_per_task=40, outcomes=outcomes, shift=0.8, scale=1.3)
    ref = load_cell(tmp_path, "toy", "fp16", "suite")
    assert ref[0].shadows() == ["w4"]
    sp = split_by_task(ref, {"train": 0.5, "calib": 0.25, "test": 0.25}, 0)
    m = M.build("mahalanobis", ["vlm"]).fit(sp["train"])
    cp = FunctionalCP(0.1).fit([m.score(e) for e in sp["calib"] if e.success])
    views = [e.as_shadow("w4") for e in sp["test"]]
    assert [v.success for v in views] == [e.success for e in sp["test"]]  # same outcomes by construction
    cf_fpr = np.mean([cp.first_alarm(m.score(v)) is not None for v in views if v.success])
    ref_fpr = np.mean([cp.first_alarm(m.score(e)) is not None for e in sp["test"] if e.success])
    assert cf_fpr > ref_fpr + 0.3, (cf_fpr, ref_fpr)
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    ev = importlib.import_module("03_evaluate")
    new = load_cell(tmp_path, "toy", "w4", "suite")
    pr = ev.paired_row(cp, m, sp["test"], new)
    assert pr["n_pairs"] == len(sp["test"]) and pr["n_induced_fail"] == 0 and pr["fpr"] > pr["fpr_ref_same_trials"]


def test_variant_resolution_and_matched_memory():
    from assure.variants import choose_protected, resolve
    exp = {"variants": {"w4": {"kind": "quant", "bits": 4}}}
    v = resolve("w4p-weight_error-f10", exp)
    assert v["protect_rule"] == "weight_error" and abs(v["protect_frac"] - 0.10) < 1e-9
    torch.manual_seed(0)
    net = nn.Module()
    net.vision = nn.Module(); net.vision.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(6)])
    net.llm = nn.Module(); net.llm.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(8)])
    total = sum(m.weight.numel() for m in net.modules() if isinstance(m, nn.Linear))
    for rule in ("weight_error", "edge", "random"):
        chosen = choose_protected(net, "toy", rule, 0.2)
        used = sum(dict(net.named_modules())[b].weight.numel() for b in chosen)
        assert 0 < used <= 0.2 * total, (rule, used / total)


def test_applicability_matrix():
    import yaml
    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]
    assert M.applicable("acc", models["pi05"])[0]
    assert not M.applicable("acc", models["openvla"])[0]
    assert not M.applicable("acc", models["openvla_oft"])[0]
    assert not M.applicable("fiper", models["openvla"])[0]
    assert M.applicable("safe_mlp", models["openvla"])[0]


def test_exact_budgets_match_derivation():
    from assure.conformal import pac_violation_prob, required_successes
    m = required_successes(0.10)
    assert (m["total"], m["n_shape"], m["n_conf"], m["rank"]) == (19, 10, 9, 9)  # q = max of 9; 1 - 9/10 = 0.1
    p = required_successes(0.10, guarantee="pac", delta=0.1)
    assert (p["total"], p["n_shape"], p["n_conf"], p["rank"]) == (44, 22, 22, 22)
    assert abs(pac_violation_prob(22, 22, 0.10) - 0.9 ** 22) < 1e-12 and 0.9 ** 22 <= 0.1 < 0.9 ** 21
    rng = np.random.default_rng(0)
    cp = FunctionalCP(0.10).fit([np.abs(rng.normal(0, 1, 12)) for _ in range(19)])
    assert cp.n_shape == 10 and cp.n_cal == 9 and not cp.underpowered
    cp = FunctionalCP(0.10, guarantee="pac").fit([np.abs(rng.normal(0, 1, 12)) for _ in range(44)])
    assert cp.n_shape == 22 and cp.n_cal == 22 and not cp.underpowered


def test_ties_keep_the_bound():
    # heavily tied (discretised) scores: strict '>' against an order statistic stays conservative
    rng = np.random.default_rng(5)
    f = []
    for k in range(400):
        cal = [np.round(np.abs(rng.normal(0, 1, 8)), 0) for _ in range(19)]
        cp = FunctionalCP(0.10, seed=k).fit(cal)
        f.append(np.mean([cp.first_alarm(np.round(np.abs(rng.normal(0, 1, 8)), 0)) is not None for _ in range(50)]))
    assert np.mean(f) <= 0.10 + 0.01, np.mean(f)


def test_trial_roles_are_a_preregistered_pooled_partition():
    from assure.common import TOPUP_OFFSET, role_of, trial_roles
    split = {"train": 0.3, "calib": 0.3, "test": 0.4}
    r1 = trial_roles(7, list(range(10)), 50, split)
    assert r1 == trial_roles(7, list(range(10)), 50, split)            # deterministic, outcome-free
    counts = {k: sum(v == k for v in r1.values()) for k in split}
    assert counts == {"train": 150, "calib": 150, "test": 200}
    per_task = [sum(r1[(t, i)] == "test" for i in range(50)) for t in range(10)]
    assert len(set(per_task)) > 1                                       # pooled, not stratified
    assert role_of(r1, 0, 55) == "train" and role_of(r1, 0, TOPUP_OFFSET + 3) == "adapt"


def test_holm_compare_and_verdicts():
    from assure.metrics import compare, holm, verdict
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    tasks = np.repeat(np.arange(10), 30)
    ref = (rng.random(300) < 0.08).astype(float)
    same = ref.copy()
    worse = np.maximum(ref, (rng.random(300) < 0.25).astype(float))
    c0 = compare(ref, same, tasks, margin=0.05)
    c1 = compare(ref, worse, tasks, margin=0.05)
    assert verdict(c0["p_worse"], c0["p_noninf"]) == "non-inferior"
    assert verdict(c1["p_worse"], c1["p_noninf"]) == "degraded"
    small = compare(ref[:20], same[:20], tasks[:20], margin=0.05)      # tiny sample: no robustness claim
    assert verdict(small["p_worse"], small["p_noninf"]) != "degraded"


def test_cluster_bootstrap_is_wider_for_correlated_tasks():
    from assure.metrics import bootstrap, cluster_bootstrap
    rng = np.random.default_rng(1)
    tasks = np.repeat(np.arange(10), 30)
    task_rate = rng.uniform(0, 0.4, 10)
    x = (rng.random(300) < task_rate[tasks]).astype(float)
    lo_c, hi_c = cluster_bootstrap(lambda i: x[i].mean(), tasks, 1000)
    lo_i, hi_i = bootstrap(lambda i: x[i].mean(), 300, 1000)
    assert hi_c - lo_c > hi_i - lo_i


def test_margin_drift_rule_reads_sensitivity(tmp_path):
    import json
    from assure.variants import choose_protected
    net = nn.Module()
    net.llm = nn.Module(); net.llm.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(10)])
    blocks = [{"name": f"llm.layers.{i}", "margin_drift": float(i == 3), "score_drift": float(i == 5)} for i in range(10)]
    (tmp_path / "toy.json").write_text(json.dumps({"blocks": blocks}))
    assert choose_protected(net, "toy", "margin_drift", 0.1, sensitivity_dir=tmp_path) == ["llm.layers.3"]
    assert choose_protected(net, "toy", "score_drift", 0.1, sensitivity_dir=tmp_path) == ["llm.layers.5"]


def test_obs_controlled_row_labels_and_shift(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    ev = importlib.import_module("03_evaluate")
    rng = np.random.default_rng(8)
    make_cell(tmp_path, rng, "fp16", n_per_task=40, shadow=("w4", 0.8, 1.3))
    ref = load_cell(tmp_path, "toy", "fp16", "suite")
    sp = split_by_task(ref, {"train": 0.5, "calib": 0.25, "test": 0.25}, 0)
    m = M.build("mahalanobis", ["vlm"]).fit(sp["train"])
    cp = FunctionalCP(0.1).fit([m.score(e) for e in sp["calib"] if e.success])
    r = ev.obs_controlled_row(cp, m, sp["test"], "w4")
    assert r["labels_from"] == "reference_policy" and r["score_shift_sd"] > 0.5
    assert r["action_div"] == 0.0  # synthetic shadow reuses the executed chunk


def test_degradation_orientation_per_metric():
    from assure.metrics import auroc_stat, compare
    rng = np.random.default_rng(3)
    t = np.repeat(np.arange(10), 30)
    ref_s = (rng.random(300) < 0.8).astype(float)
    var_s = ref_s * (rng.random(300) > 0.25)          # variant succeeds less -> worse
    c = compare(ref_s, var_s, t, worse="lower")
    assert c["delta"] > 0 and c["p_worse"] < 0.05     # D_success = ref - var > 0
    c = compare(var_s, ref_s, t, worse="lower")       # variant succeeds MORE -> not degraded
    assert c["delta"] < 0 and c["p_worse"] > 0.5
    lab = (rng.random(300) < 0.3).astype(float)
    good = lab + rng.normal(0, 0.3, 300)
    bad = lab * 0.2 + rng.normal(0, 0.3, 300)
    c = compare(np.c_[good, lab], np.c_[bad, lab], t, worse="lower", stat=auroc_stat, B=300)
    assert c["delta"] > 0.2 and c["p_worse"] < 0.05   # D_AUROC = AUROC_ref - AUROC_var


def test_power_analysis_runs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    pa = importlib.import_module("power_analysis")
    r = pa.run(n_test=200, success=0.7, margin=0.10, m=5, delta_alt=0.15, reps=20, B=100, flip=0.3, base=0.09)
    assert 0 <= r["power_ni"] <= 1 and r["n_pairs"] == 98


def test_margin_greedy_rule_reads_prefix(tmp_path):
    import json
    from assure.variants import choose_protected
    net = nn.Module()
    net.llm = nn.Module(); net.llm.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(10)])
    (tmp_path / "toy.json").write_text(json.dumps({"blocks": [], "greedy": {"10": ["llm.layers.7"], "20": ["llm.layers.7", "llm.layers.2"]}}))
    assert choose_protected(net, "toy", "margin_greedy", 0.2, sensitivity_dir=tmp_path) == ["llm.layers.7", "llm.layers.2"]


def test_greedy_check_runs_on_a_mock_policy():
    import importlib
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "scripts"))
    sens = importlib.import_module("02_sensitivity")
    torch.manual_seed(0)
    net = nn.Module()
    net.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(6)])

    def fwd(x):
        for l in net.layers:
            x = torch.tanh(l(x))
        return x
    sess = SimpleNamespace(module=net,
                           replay=lambda c: ({"vlm": fwd(c).detach().numpy().ravel()}, np.zeros((2, 2))))
    episodes = [[torch.randn(1, 16) for _ in range(5)] for _ in range(6)]
    ok = np.array([True, True, True, False, False, True])
    ref = [[sess.replay(c) for c in ep] for ep in episodes]

    def mon_stats(run):
        S = [np.array([f["vlm"].sum() for f, _ in ep]) for ep in run]
        return S, np.array([x.max() for x in S])
    _, M_ref = mon_stats(ref)
    blocks = [f"layers.{i}" for i in range(6)]
    out = [{"name": b, "margin_drift": float(i + 1), "margin_drift_sel": float(i + 1)} for i, b in enumerate(blocks)]
    a = SimpleNamespace(bits=4, greedy_budgets="20,40", greedy_candidates=4, greedy_random=1,
                        select_episodes=3, holdout_episodes=3)
    greedy, rows, trace = sens.greedy_check(a, sess, episodes, ok, ref, mon_stats, M_ref, float(np.std(M_ref)) + 1e-9,
                                     blocks, out)
    assert set(greedy) == {"20", "40"} and len(greedy["40"]) >= len(greedy["20"]) >= 1
    assert all("improved" in x for x in trace)
    for r in rows:  # matched memory, held-out comparison present
        b = r["budget_pct"] / 100
        fr = [r[k] for k in ("protected_frac_independent", "protected_frac_greedy", "protected_frac_random_mean")]
        assert max(fr) <= b + 1e-9 and max(fr) - min(fr) < 1e-9, fr   # same memory for every strategy
        assert "J_hold_greedy" in r and len(r["ci_greedy_minus_independent"]) == 2
    w = {k: v.clone() for k, v in net.state_dict().items()}
    assert all(torch.equal(w[k], v) for k, v in net.state_dict().items())  # weights restored


def test_small_budget_levels_follow_stated_split():
    rng = np.random.default_rng(0)
    for N, shape, conf, level in [(10, 9, 1, 0.5), (11, 10, 1, 0.5), (14, 10, 4, 0.2), (19, 10, 9, 0.1)]:
        cp = FunctionalCP(0.1).fit([np.abs(rng.normal(0, 1, 8)) for _ in range(N)])
        assert (cp.n_shape, cp.n_cal) == (shape, conf) and abs(cp.guaranteed_alpha - level) < 1e-9


def test_torchao_mixed_variant_resolves():
    from assure.variants import resolve
    v = resolve("tao4p-margin_drift-f10", {"variants": {}})
    assert v["kind"] == "torchao" and v["protect_rule"] == "margin_drift" and v["bits"] == 4


def test_joint_success_bounds():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    pa = importlib.import_module("power_analysis")
    assert abs(pa.joint_success(0.7, 0.7, 0.0) - 0.49) < 1e-12
    assert pa.joint_success(0.7, 0.6, 1.0) <= 0.6 + 1e-12          # Frechet upper bound
    assert pa.joint_success(0.7, 0.6, -1.0) >= 0.3 - 1e-12         # Frechet lower bound
    assert pa.joint_success(0.7, 0.7, 0.3) > 0.49


def test_compare_joint_and_action_noise():
    from assure.metrics import compare_joint, masked_mean
    from assure.perturb import Perturber
    rng = np.random.default_rng(2)
    t = np.repeat(np.arange(10), 20)
    rs, vs = rng.random(200) < 0.7, rng.random(200) < 0.7
    ra = (rng.random(200) < 0.1).astype(float)
    va = np.maximum(ra, (rng.random(200) < 0.2).astype(float))
    c = compare_joint(lambda i: masked_mean(ra, rs, i), lambda i: masked_mean(va, vs, i), t, worse="higher")
    assert c["delta"] > 0 and c["delta_lower95_1s"] <= c["delta"] <= c["delta_upper95_1s"]
    p1, p2 = Perturber({"action_noise_std": 0.1, "seed": 0}), Perturber({"action_noise_std": 0.1, "seed": 0})
    p1.reset(5); p2.reset(5)
    a1, a2 = p1.action(np.zeros(7)), p2.action(np.zeros(7))
    assert np.allclose(a1, a2) and a1[-1] == 0 and np.abs(a1[:-1]).sum() > 0   # seeded, gripper untouched
    assert Perturber({"action_noise_std": 0.1}).visual is False


def test_sub_block_budget_fill_and_wcr_test():
    from assure.metrics import confirmatory, wild_cluster_test
    from assure.variants import BUDGET_TOLERANCE, choose_protected, protected_fraction

    class Blk(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.q, self.k = nn.Linear(d, d), nn.Linear(d, d // 4)
            self.up, self.down = nn.Linear(d, h), nn.Linear(h, d)
    net = nn.Module()
    net.layers = nn.ModuleList([Blk(128, 512) for _ in range(18)])
    for f in (0.05, 0.10, 0.20):
        for rule in ("random", "edge", "weight_error"):
            got = protected_fraction(net, choose_protected(net, "x", rule, f))
            assert f - BUDGET_TOLERANCE <= got <= f + 1e-12, (rule, f, got)
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 15)
    d = rng.normal(0.0, 0.3, 300)
    assert wild_cluster_test(d + 0.2, g, 0.0, "greater", B=999)["p"] < 0.01    # clear degradation
    assert wild_cluster_test(d, g, 0.2, "less", B=999)["p"] < 0.01             # clearly below the margin
    c = confirmatory(d, g, 0.05, B=999)
    assert c["n_clusters"] == 20 and c["delta_lo"] < c["delta"] < c["delta_hi"]


def test_monitor_halves_are_disjoint_and_named():
    assert M.parse_name("mahalanobis#A") == ("mahalanobis", "A") and M.parse_name("rnd") == ("rnd", None)
    items = list(range(101))
    A, B = M.train_subset(items, "A", 7), M.train_subset(items, "B", 7)
    assert not set(A) & set(B) and len(A) + len(B) == 101 and M.train_subset(items, None, 7) == items


def test_power_generator_hits_the_boundary_exactly():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    cl = importlib.import_module("calibrate_level")
    d, g = cl.sim(np.random.default_rng(0), 20, 400_000, 0.05, 0.09, 0.3, 0.04)
    assert abs(d.mean() - 0.05) < 0.003


def test_rank_bootstrap_and_breakdown():
    import importlib
    sys.path.insert(0, str(ROOT / "scripts"))
    sens = importlib.import_module("02_sensitivity")
    from assure.variants import fill_budget, protected_breakdown

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(64, 64), nn.Linear(64, 256)
    net = nn.Module()
    net.layers = nn.ModuleList([Blk() for _ in range(12)])
    blocks = [f"layers.{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    ok = np.array([True] * 16 + [False] * 8)
    dsel = {b: np.abs(rng.normal(i * 0.3, 0.1, 24)) for i, b in enumerate(blocks)}   # strong, stable signal
    out = sens.rank_bootstrap(net, blocks, dsel, ok, B=100)
    assert out["10"]["mean_jaccard"] > 0.8 and out["entry_criterion_10pct"] is True
    P = fill_budget(net, blocks[::-1], 0.10)
    br = protected_breakdown(net, P)
    assert abs(br["protected_frac_blocks"] + br["protected_frac_filler"] - 0.10) < 0.0025 + 1e-9


def test_round10_monitor_fit_excludes_repair_and_extension_roles():
    import importlib
    from assure.common import EXT_OFFSET, load_configs, role_of, trial_roles
    sys.path.insert(0, str(ROOT / "scripts"))
    ev = importlib.import_module("03_evaluate")
    _, exp = load_configs()
    roles = trial_roles(exp["seed"], list(range(10)), 50, exp["split"])
    eps = [type("E", (), {"meta": {"task_id": t, "episode_idx": i}})() for t in range(10) for i in range(50)]
    sp, R, _ = ev.reference_split(ROOT / "nonexistent_rollouts", "m", "k", eps, exp)
    tr = {(e.meta["task_id"], e.meta["episode_idx"]) for e in sp["train"]}
    assert all(roles[k] == "train" for k in tr)                       # monitor sees train slots only
    assert len(sp["repair_ref"]) == sum(r == "repair" for r in roles.values())
    assert role_of(roles, 0, EXT_OFFSET + 3) == "ext_test"


def test_round10_detection_sizing_plan(tmp_path):
    import json as _json
    import subprocess
    f = ROOT / "scripts/detection_feasibility.py"
    r = subprocess.run([sys.executable, str(f), "--plan", "--p_fail", "0.25", "0.25", "--rho", "0.6", "--disc_hat", "0.10"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    plan = _json.loads((ROOT / "results/power/stage_c_sizing.json").read_text())
    assert plan["n_min"] == 200 and plan["extension_per_task"] == 45   # 0.10 x pre-registered inflation 1.5
    # joint posterior simulation: more uncertainty -> more extension trials than the point scenario
    r = subprocess.run([sys.executable, str(f), "--plan", "--joint_counts", "95", "10", "12", "23", "--disc_counts", "3", "40",
                        "--plan_draws", "4000"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    plan = _json.loads((ROOT / "results/power/stage_c_sizing.json").read_text())
    assert plan["mode"] == "joint posterior simulation"
    assert plan["extension_per_task"] is None or plan["predicted_probability_at_K"] >= 0.90


def test_round11_finite_population_conformal():
    from assure.conformal import finite_population_budget, finite_population_fpr
    assert abs(finite_population_fpr(9, 122, 9, 0.10)["expected_test_fpr"] - 0.10) < 1e-9   # 19-success marginal
    v = finite_population_fpr(22, 175, 22, 0.10)["p_test_fpr_above_alpha"]
    assert 0.10 < v < 0.11                                    # 44 is the exchangeability budget, slightly short here
    assert finite_population_budget(0.10, 0.10, 175)["total"] == 46
    # m-dependence (lattice k/m): at small test subsets the violation probability is larger
    assert finite_population_fpr(22, 49, 22, 0.10)["p_test_fpr_above_alpha"] > 0.14


def test_round13_f2_power_rule_and_f4_order():
    import importlib
    import argparse
    sys.path.insert(0, str(ROOT / "scripts"))
    f2 = importlib.import_module("f2_power")
    a = argparse.Namespace(n_test=350, n_baselines=3, success=0.75, sd_success=0.08, rho=0.6, c_s=0.10, c_f=0.10)
    rng = np.random.default_rng(0)
    claims, comp, gates, njf = f2.one_study(rng, a, 0.04, 83, 0.0, 0.0, 200, (0.0375, 0.03, 0.02), 199, 0)
    assert claims.shape == (3,) and len(gates) == 3 and njf > 200
    from assure.common import load_configs
    _, exp = load_configs()
    assert exp["families"]["f4_baseline_order"][0] == "action_drift"
    assert exp["families"]["detection"]["delta_star"] == 0.03
