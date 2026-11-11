"""Consolidated execution budget: every policy execution the staged plan requires, before starting.

One row per (stage, model, configuration, suite, condition): trials, shadow copies, estimated GPU-hours
and storage. Episode counts follow configs/experiment.yaml exactly (roles, stages, families.detection);
time and storage per episode are PLANNING ASSUMPTIONS (configs/experiment.yaml: budget) until Stage A,
after which `--measured` replaces them with the seconds and file sizes recorded in rollouts/*/run.json.

  python scripts/budget.py                     # K = 0 / planning K / cap for the detection extension
  python scripts/budget.py --K 40 --measured   # after Stage A / the Stage C sizing
Writes results/budget/budget.csv and budget.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assure.common import load_configs  # noqa: E402


def slots(R, split):
    n = R * 10
    return {k: int(round(v * n)) for k, v in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None, help="detection extension trials per task (default: planning K)")
    ap.add_argument("--measured", action="store_true", help="use run.json timings / sizes where available")
    a = ap.parse_args()
    models, exp = load_configs()
    bud = exp["budget"]
    st = exp["stages"]
    det = exp["families"]["detection"]
    K = bud["planning_extension_per_task"] if a.K is None else a.K
    rows = []

    def add(stage, model, config, suite, cond, trials, shadows=0, note=""):
        rows.append(dict(stage=stage, model=model, configuration=config, suite=suite, condition=cond,
                         trials=int(trials), shadows=shadows, note=note))

    # Stage 0 / A
    sA = slots(st["A"]["reference_episodes_per_task"], exp["split"])
    for m in st["A"]["models"]:
        for s in st["A"]["suites"]:
            for c in st["A"]["conditions"]:
                add("A", m, "fp16", s, c, sum(sA.values()), len(st["A"]["shadows"]))
                for v in st["A"]["deploy"]:
                    if v != "update" or models[m].get("update_checkpoint"):
                        add("A", m, v, s, c, sA["test"] + sA["repair"])
    rp = st["A"]["replication"]
    add("A", rp["model"], "fp16", rp["suite"], rp["condition"], sum(sA.values()), len(rp["shadows"]), "replication of [25]")
    for v in rp["deploy"]:
        add("A", rp["model"], v, rp["suite"], rp["condition"], sA["test"] + sA["repair"], 0, "replication of [25]")
    # Stage B (the nominal fp16 cells double as the Stage 0 compatibility check)
    sB = slots(exp["reference_episodes_per_task"], exp["split"])
    for m in st["B"]["models"]:
        for s in st["B"]["suites"]:
            for c in st["B"]["conditions"]:
                add("B", m, "fp16", s, c, sum(sB.values()), len(st["B"]["shadows"]),
                    "also Stage 0 check" if c == "nominal" else "")
                for v in st["B"]["deploy"]:
                    if v != "update" or models[m].get("update_checkpoint"):
                        add("B", m, v, s, c, sB["test"] + sB["repair"])
    sec = st["B"]["secondary"]
    for m in sec["models"]:
        for s in sec["suites"]:
            for c in sec["conditions"]:
                add("B", m, "fp16", s, c, sum(sB.values()), 0, "secondary (actuator noise)")
                for v in sec["deploy"]:
                    add("B", m, v, s, c, sB["test"] + sB["repair"], 0, "secondary (actuator noise)")
    # Stage C
    C = st["C"]
    m, s0, c0 = C["pareto_model"], C["pareto_suite"], C["pareto_condition"]
    add("C", m, "fp16 capture (selection + held-out)", s0, c0, bud["max_captured"], 0, "02_sensitivity, upper bound")
    n_shadow_cfg = len(exp["pareto"]["rules"]) * len(exp["pareto"]["protect_pct"])
    for i in range(0, n_shadow_cfg, 5):
        add("C", m, "fp16 + Pareto shadows", s0, c0, sum(sB.values()), min(5, n_shadow_cfg - i), "shadow pass")
    add("C", m, "parity replays", s0, c0, exp["parity"]["episodes"], 0, "07_parity capture")
    ext = 10 * K
    conf = [exp["families"]["proposed"]] + exp["families"]["confirmatory_baselines"]
    for ds in C["pareto_deploy_suites"]:
        for r in conf:
            add("C", m, f"w4p-{r}-f10", ds, c0, sB["test"] + sB["repair"] + ext, 0, f"F2/F3; {ext} extension")
        for v in C["real_kernel_deploy"]:
            add("C", m, v, ds, c0, sB["test"] + sB["repair"] + ext, 0, f"F4; {ext} extension")
        for pct, rules in exp["pareto"]["deploy"]["descriptive"].items():
            for r in rules:
                add("C", m, f"w4p-{r}-f{pct}", ds, c0, sB["test"] + sB["repair"], 0, "descriptive frontier")
    gb = C["greedy_broad"]
    add("C", gb["model"], "fp16 capture x2 (restricted, broad)", s0, c0, 2 * bud["max_captured"], 0, "greedy adequacy")
    for mr in C["models_reduced"]:
        add("C", mr, "fp16", "libero_10", "shift_mild", sum(sB.values()), 1)
        add("C", mr, "w4", "libero_10", "shift_mild", sB["test"] + sB["repair"])
        if mr == "openvla":
            add("C", mr, "bnb4", "libero_10", "shift_mild", sB["test"] + sB["repair"])
    for mm, vs in C["efficiency"].items():
        for v in vs:
            add("C", mm, v, "libero_10" if mm != "openvla" else "libero_spatial", "nominal", bud["efficiency_episodes"], 0,
                "latency / memory")

    df = pd.DataFrame(rows)
    sec_ep = dict(bud["seconds_per_episode"])
    mb_ep = dict(bud["mb_per_episode"])
    src = {k: "assumed" for k in sec_ep}
    if a.measured:
        for f in (ROOT / "rollouts").glob("*/*/*/run.json"):
            j = json.loads(f.read_text())
            if j.get("episodes") and not j.get("shadow_names"):
                mdl = f.parts[-4]
                sec_ep[mdl] = j["seconds"] / j["episodes"]
                src[mdl] = "measured"
                sizes = [p.stat().st_size for p in f.parent.glob("*.npz")]
                if sizes:
                    mb_ep[mdl] = sum(sizes) / len(sizes) / 2**20
    so = bud["shadow_overhead"]
    df["sec_per_trial"] = [sec_ep[mm] * (1 + so * sh) for mm, sh in zip(df.model, df.shadows)]
    df["gpu_hours"] = df.trials * df.sec_per_trial / 3600
    df["storage_gb"] = df.trials * [mb_ep[mm] * (1 + sh) for mm, sh in zip(df.model, df.shadows)] / 1024
    df["timing_source"] = [src[mm] for mm in df.model]
    out = ROOT / "results/budget"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "budget.csv", index=False)
    tot = df.groupby("stage")[["trials", "gpu_hours", "storage_gb"]].sum().round(1)
    print(tot.to_string())
    print(f"TOTAL: {int(df.trials.sum())} policy executions, {df.gpu_hours.sum():.0f} GPU-hours, "
          f"{df.storage_gb.sum():.0f} GB (detection extension K = {K} per task; cap {det['max_extension_per_task']})")
    ext_rows = df[df.note.str.contains("extension")]
    per_k = len(ext_rows) * 10 * sec_ep[m] / 3600   # 10 tasks per suite row, one extra trial per task
    print(f"each extra extension trial per task costs ~{per_k:.2f} GPU-hours across the {len(ext_rows)} confirmatory configurations")
    (out / "budget.md").write_text(df.round(2).to_markdown(index=False) + "\n\n" + tot.to_markdown() + "\n")


if __name__ == "__main__":
    main()
