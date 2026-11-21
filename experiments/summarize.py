"""Builds experiments/summary_smolvla.csv: one row per (change, placement, monitor) from the stage-2 re-split files.

Primary metric: inherited monitor's false-alarm rate on ALL successes of each policy (alpha = 0.10, pre-registered
split), with exact 95 % Clopper-Pearson intervals; secondary: joint-success (paired) rate, re-split median and range,
threshold-free AUROC, paired outcome transitions.

  python experiments/summarize.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {  # change -> stage-2 part that holds its final scoring
    "w4": "stage2_part4", "w4_lm": "stage2_part1", "w8": "stage2_part2",
    "cfg_denoise5": "stage2_part3", "cfg_steps5": "stage2_part4",
    "real_nf4": "stage2_part5",   # real 4-bit inference (bitsandbytes NF4 kernels) of the action expert
    "fp16_reseed": "stage2_part8",   # null-change control: the unchanged policy with fresh random draws
}
rows = []
for change, part in SOURCES.items():
    for placement in ("vlm_only", "both_layers", "expert_only"):
        f = HERE / "smolvla" / part / f"resplit_{part.split('_')[1]}_{placement}.json"
        v = json.loads(f.read_text())["variants"][change]
        for mon, r in v["monitors"].items():
            p, o = r["preregistered"], r["over_splits"]
            rows.append(dict(
                change=change, placement=placement, monitor=mon,
                n_fp16_success=r["n_fp16_success"], n_var_success=r["n_var_success"],
                fpr_fp16_all=round(p["fpr_fp16_all"], 3), fpr_fp16_all_ci95=r["fpr_fp16_all_ci95"],
                fpr_var_all=round(p["fpr_var_all"], 3), fpr_var_all_ci95=r["fpr_var_all_ci95"],
                fpr_var_rescued=round(p["fpr_var_rescued"], 3),
                n_joint_success=r["n_joint_success"], fpr_fp16_joint=round(p["fpr_fp16"], 3),
                fpr_var_joint=round(p["fpr_var"], 3),
                resplit_fpr_var_median=round(o["fpr_var_all"][0], 3),
                resplit_fpr_var_p05=round(o["fpr_var_all"][1], 3), resplit_fpr_var_p95=round(o["fpr_var_all"][2], 3),
                auroc_fp16=round(p["auroc_fp16"], 3), auroc_var=round(p["auroc_var"], 3),
                detect_new_failures=round(p["detect_new"], 3), n_new_failures=r["n_new_failures"],
                both_succeed=v["transitions"]["both_succeed"], new_failures=v["transitions"]["new_failures"],
                rescued=v["transitions"]["rescued"], both_fail=v["transitions"]["both_fail"],
                mcnemar_p=round(v["mcnemar_p"], 3), source=str(f.relative_to(HERE)),
            ))
out = HERE / "summary_smolvla.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")

# pi0.5: same columns from experiments/pi05/partN, plus cross-fit release options (no repair pools for pi0.5) and the
# two controls (action-chunk consistency, pixel Mahalanobis) read from the per-trial files.
import csv as _csv  # noqa: E402
PI05 = {"w4": "part2", "w4_lm": "part3", "w8": "part4"}
rows = []
for change, part in PI05.items():
    for placement in ("vlm_only", "both_layers", "expert_only"):
        f = HERE / "pi05" / part / f"pi05_resplit_{placement}.json"
        v = json.loads(f.read_text())["variants"][change]
        for mon, r in v["monitors"].items():
            p, o = r["preregistered"], r["over_splits"]
            cf = r["fixes_crossfit"]["crossfit_pac"]
            opt = {k: cf["options"][k] for k in ("inherited", "coral", "recal", "certify_or_recal", "recal_all")}
            rows.append(dict(
                change=change, placement=placement, monitor=mon,
                n_fp16_success=r["n_fp16_success"], n_var_success=r["n_var_success"],
                fpr_fp16_all=round(p["fpr_fp16_all"], 3), fpr_fp16_all_ci95=r["fpr_fp16_all_ci95"],
                fpr_var_all=round(p["fpr_var_all"], 3), fpr_var_all_ci95=r["fpr_var_all_ci95"],
                n_joint_success=r["n_joint_success"], fpr_fp16_joint=round(p["fpr_fp16"], 3),
                fpr_var_joint=round(p["fpr_var"], 3),
                resplit_fpr_var_median=round(o["fpr_var_all"][0], 3),
                resplit_fpr_var_p05=round(o["fpr_var_all"][1], 3), resplit_fpr_var_p95=round(o["fpr_var_all"][2], 3),
                auroc_fp16=round(p["auroc_fp16"], 3), auroc_var=round(p["auroc_var"], 3),
                detect_all_var_failures=round(p["tpr_var"], 3), n_new_failures=r["n_new_failures"],
                xfit44_early_warning_share=round(cf["flag_rate"], 3), xfit44_certified_share=round(cf["certify_rate"], 3),
                **{f"xfit44_{k}_fpr": round(x["fpr"][0], 3) for k, x in opt.items()},
                **{f"xfit44_{k}_detect": round(x["tpr"][0], 3) for k, x in opt.items()},
                both_succeed=v["transitions"]["both_succeed"], new_failures=v["transitions"]["new_failures"],
                rescued=v["transitions"]["rescued"], both_fail=v["transitions"]["both_fail"],
                mcnemar_p=round(v["mcnemar_p"], 3), source=str(f.relative_to(HERE)),
            ))
    t = list(_csv.DictReader(open(HERE / "pi05" / part / "pi05_trials_vlm_only.csv")))
    for mon in ("acc", "ext_mahalanobis"):
        g = [x for x in t if x["monitor"] == mon and x["variant"] == change and abs(float(x["alpha"]) - 0.10) < 1e-9]
        rs = [x for x in g if x["ref_success"] == "1"]; vs = [x for x in g if x["var_success"] == "1"]
        rows.append(dict(change=change, placement="control", monitor=mon, n_fp16_success=len(rs), n_var_success=len(vs),
                         fpr_fp16_all=round(sum(x["ref_alarm"] == "1" for x in rs) / len(rs), 3),
                         fpr_var_all=round(sum(x["var_alarm"] == "1" for x in vs) / len(vs), 3)))
out = HERE / "summary_pi05.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")
