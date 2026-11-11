"""Run one stage of the staged plan defined in configs/experiment.yaml:stages.

  python scripts/run_stage.py A --dry_run          # print the commands only
  python scripts/run_stage.py A                    # feasibility (pi0.5, one suite, one shift)
  python scripts/run_stage.py B --models smolvla   # core results, one model at a time
  python scripts/run_stage.py C                    # mixed-precision Pareto, reduced extra models, efficiency
  REPO=~/vla-repos/openvla-oft python scripts/run_stage.py B --models openvla_oft

Before collecting, prints the trial design (slots per role) and whether the calibration slots can meet
the alpha = 0.10 sample sizes at plausible success rates. Each stage ends with 03_evaluate + 04_report;
check the stage's go/no-go criteria (README) before starting the next one.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assure.common import load_configs  # noqa: E402
from assure.conformal import required_successes  # noqa: E402


def key(suite, cond):
    return suite if cond == "nominal" else f"{suite}+{cond}"


def _sf(n, p, k):
    """P(Binomial(n, p) >= k)."""
    from math import comb
    return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


def design_report(R, exp):
    n = R * 10
    sl = {k: int(round(v * n)) for k, v in exp["split"].items()}
    cal = exp["calibration"]
    need = {g: required_successes(0.10, cal["shape_frac"], g, cal["delta"], cal.get("min_shape", 10))["total"]
            for g in ("marginal", "pac")}
    print(f"# trial design: R={R}/task -> {n} slots: " + ", ".join(f"{k} {v}" for k, v in sl.items()))
    print(f"# calibration at alpha=0.10 needs {need['marginal']} (marginal) / {need['pac']} (PAC) REALIZED successes "
          f"among {sl['calib']} calib slots (expected count reaches them at success {need['marginal'] / sl['calib']:.1%} / "
          f"{need['pac'] / sl['calib']:.1%}). Qualification uses the realized count, never the rate.")
    print(f"# each variant runs {sl['test']} test slots (all paired, all in the endpoint) + {sl['repair']} repair slots")
    print("# feasibility by reference success rate p (variant assumed similar):")
    print("#    p   P(calib>=marg) P(calib>=PAC) P(train fail>=5) E[test fails] both-success pairs (indep. .. max)")
    for p in (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        print(f"#  {p:.2f}  {_sf(sl['calib'], p, need['marginal']):12.3f} {_sf(sl['calib'], p, need['pac']):13.3f} "
              f"{_sf(sl['train'], 1 - p, 5):16.3f} {sl['test'] * (1 - p):13.0f} "
              f"{sl['test'] * p * p:16.0f} .. {sl['test'] * p:.0f}")
    print("# pairs: 'indep.' = uncorrelated successes (a planning scenario, not a bound); positive correlation")
    print("#        (likely for shared initial states) raises it up to 'max', negative lowers it. Measure in Stage A")
    print("#        (power_analysis.py --from_eval).")
    print("# target window: reference success 0.60-0.85 (enough pairs for the endpoint AND enough failures to detect)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["A", "B", "C"])
    ap.add_argument("--models", default=None, help="override model list")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--alt", type=int, default=0, help="extra noise samples (flow VLAs) for the fiper monitor")
    a = ap.parse_args()
    models, exp = load_configs()
    st = exp["stages"][a.stage]
    repo = os.environ.get("REPO")
    py = [sys.executable]
    cmds: list[list[str]] = []

    def collect(model, variant, suite, cond, R, shadows=(), target=0, max_rounds=None):
        c = py + [str(ROOT / "scripts/01_collect.py"), "--model", model, "--variant", variant, "--suite", suite,
                  "--condition", cond, "--skip_existing", "--alt_samples", str(a.alt), "--episodes", str(R)]
        if shadows:
            c += ["--shadow", ",".join(shadows)]
        if target:
            c += ["--target_successes", str(target)]
            if max_rounds is not None:
                c += ["--max_extra_rounds", str(max_rounds)]
        if models[model]["backend"] == "openvla":
            if not repo:
                sys.exit(f"{model} needs REPO=<path to openvla or openvla-oft checkout>")
            c += ["--repo_root", repo]
        cmds.append(c)

    def evaluate(model, k, monitors=None):
        c = py + [str(ROOT / "scripts/03_evaluate.py"), "--model", model, "--key", k]
        if a.stage == "A":
            c += ["--learning_curve"]
        if monitors:
            c += ["--monitors", ",".join(monitors)]
        cmds.append(c)

    if a.stage in ("A", "B"):
        R = st.get("reference_episodes_per_task", exp["reference_episodes_per_task"])
        design_report(R, exp)
        vt = st.get("repair_target_successes", 0)       # precommitted repair-pool top-up rule
        vr = st.get("max_topup_rounds", 0)
        for model in (a.models.split(",") if a.models else st["models"]):
            for suite in st["suites"]:
                for cond in st["conditions"]:
                    collect(model, "fp16", suite, cond, R, st["shadows"])
                    for v in st["deploy"]:
                        if v == "update" and not models[model].get("update_checkpoint"):
                            continue
                        collect(model, v, suite, cond, R, target=vt, max_rounds=vr)
                    evaluate(model, key(suite, cond), st.get("monitors"))
        sec = st.get("secondary")  # non-visual failure source; reported separately, not in the primary pool
        if sec and not a.models:
            for model in sec["models"]:
                for suite in sec["suites"]:
                    for cond in sec["conditions"]:
                        collect(model, "fp16", suite, cond, R)
                        for v in sec["deploy"]:
                            collect(model, v, suite, cond, R)
                        evaluate(model, key(suite, cond))
        rp = st.get("replication")
        if rp and not a.models:
            collect(rp["model"], "fp16", rp["suite"], rp["condition"], R, rp.get("shadows", ()))
            for v in rp.get("deploy", []):
                collect(rp["model"], v, rp["suite"], rp["condition"], R)
            evaluate(rp["model"], key(rp["suite"], rp["condition"]), st.get("monitors"))
    else:
        R = exp["reference_episodes_per_task"]
        design_report(R, exp)
        m, s, c = st["pareto_model"], st["pareto_suite"], st["pareto_condition"]
        k = key(s, c)
        # reference rollouts first (needed to train the monitor behind score_drift / margin_drift)
        collect(m, "fp16", s, c, R)
        cmds.append(py + [str(ROOT / "scripts/02_sensitivity.py"), "--model", m, "--suite", s, "--condition", c,
                          "--ref_key", k, "--monitor", st.get("sensitivity_monitor", "mahalanobis"), "--greedy",
                          "--stability", ",".join(str(x) for x in st.get("sensitivity_stability", [12, 24, 48]))])
        gb = st.get("greedy_broad")
        if gb:  # unrestricted candidate set on a smaller model / budget: is the restricted search adequate?
            collect(gb["model"], "fp16", s, c, R)
            for tag, cands in (("_restricted", "16"), ("_broad", "0")):
                cmds.append(py + [str(ROOT / "scripts/02_sensitivity.py"), "--model", gb["model"], "--suite", s,
                                  "--condition", c, "--ref_key", k, "--monitor", st.get("sensitivity_monitor", "mahalanobis"),
                                  "--greedy", "--greedy_budgets", str(gb["budget_pct"]), "--greedy_candidates", cands,
                                  "--tag", tag])
        pv = [f"w4p-{r}-f{p}" for r in exp["pareto"]["rules"] for p in exp["pareto"]["protect_pct"]]
        for i in range(0, len(pv), 5):  # 5 shadows per pass keeps GPU memory bounded
            collect(m, "fp16", s, c, R, pv[i:i + 5])
        print("Before deploying: size the detection component of families F2-F4 (pre-registered, configs: families.detection):")
        print("  python scripts/detection_feasibility.py --plan --joint_counts <n11 n10 n01 n00 from the Stage A pairs> "
              "--disc_counts <x n: alarm disagreements among reference failures of TRAIN/CALIB/REPAIR slots only>")
        print("  python scripts/f2_power.py --size   (K_joint; final K = max(K_det, K_joint))")
        print("  NEVER use test-slot shadow results for planning or for choosing configurations.")
        print("  -> results/power/stage_c_sizing.json gives K; add --extension K to EVERY deployed 10% variant below")
        print("     (INFEASIBLE -> detection not confirmable, the method claim is withheld; run the frontier anyway)")
        dep = exp["pareto"]["deploy"]
        print("Deploy the PRE-SPECIFIED configurations (experiment.yaml: pareto.deploy; fixed before Stage C, never")
        print("chosen from pareto.csv) on BOTH suites (20 task clusters):")
        for ds in st.get("pareto_deploy_suites", [s]):
            print(f"  python scripts/01_collect.py --model {m} --variant fp16 --suite {ds} --condition {c} --episodes {R}")
            for grp, ext_ in (("confirmatory", " --extension K"), ("descriptive", "")):
                for pct, rules in dep[grp].items():
                    for r in rules:
                        print(f"  python scripts/01_collect.py --model {m} --variant w4p-{r}-f{pct} --suite {ds} "
                              f"--condition {c} --episodes {R}{ext_}   # {grp}")
        print("Evaluate each deployed suite with the selection monitor, its independent replicate and other families:")
        for ds in st.get("pareto_deploy_suites", [s]):
            print(f"  python scripts/03_evaluate.py --model {m} --key {key(ds, c)} --monitors "
                  f"'{','.join(st.get('pareto_monitors', []))}'")
        order = exp.get("families", {}).get("f4_baseline_order", [])
        print("Real-kernel parity screen FIRST (captured reference episodes only; fixes the F4 baseline by the")
        print(f"pre-registered order {order}: the first that passes; never chosen from confirmatory outcomes):")
        print(f"  python scripts/07_parity.py --model {m} --suite {s} --condition {c} --ref_key {k} "
              f"--rules {exp['families']['proposed']},{','.join(order)} --pct 10")
        print("Then family F4 on BOTH suites with B = results/parity/<model>_f10.json:f4_baseline "
              "(skip F4 if it is null: not confirmable):")
        for ds in st.get("pareto_deploy_suites", [s]):
            for v in (f"tao4p-{exp['families']['proposed']}-f10", "tao4p-<B>-f10"):
                print(f"  python scripts/01_collect.py --model {m} --variant {v} --suite {ds} --condition {c} "
                      f"--episodes {R} --extension K   # family F4")
        evaluate(m, k)
        for model in st["models_reduced"]:
            collect(model, "fp16", "libero_10", "shift_mild", R, ["w4"])
            collect(model, "w4", "libero_10", "shift_mild", R)
            if model == "openvla":
                collect(model, "bnb4", "libero_10", "shift_mild", R)
            evaluate(model, key("libero_10", "shift_mild"))
        for model, vs in st.get("efficiency", {}).items():
            c = py + [str(ROOT / "scripts/06_efficiency.py"), "--model", model, "--variants", ",".join(vs)]
            if models[model]["backend"] == "openvla":
                c += ["--suite", "libero_spatial", "--repo_root", repo or "<REPO>"]
            cmds.append(c)
    cmds.append(py + [str(ROOT / "scripts/04_report.py")])

    for c in cmds:
        print("$", " ".join(c))
        if not a.dry_run:
            r = subprocess.run(c)
            if r.returncode != 0:
                print(f"WARN: command failed ({r.returncode}); continuing")


if __name__ == "__main__":
    main()
