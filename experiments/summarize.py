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
