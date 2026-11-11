"""Builds VLA_expert_eval.ipynb: CPU-only re-evaluation of the go/no-go rollouts with two monitor feature sets
(both layers vs action-expert only), including detection of the NEW failures 4-bit introduced. No robot runs."""
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})  # noqa: E731
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],  # noqa: E731
                                "source": s.strip("\n")})

md("""
# Closed-loop re-evaluation: does the monitor catch the failures 4-bit creates?
CPU only (Accelerator = **None**), about 10-20 min. Inputs: the `vla-assure` dataset and the output of the go/no-go
notebook (Version 8). No robot is run: this re-scores the saved episodes with two monitors.
""")
code(r"""
# 1. Locate code and saved episodes
import glob, os, pathlib, shutil, subprocess, sys
CODE = pathlib.Path("/root/vla-assure")
def newest(root):
    f = pathlib.Path(root) / "scripts/shadow_check.py"
    return f.exists() and "all_taps" in f.read_text()
trees = sorted({str(pathlib.Path(p).parents[3]) for p in
                glob.glob("/kaggle/input/**/src/assure/adapters/lerobot_adapter.py", recursive=True)})
trees = [t for t in trees if newest(t)]
assert trees, "attach the vla-assure dataset (latest version)"
shutil.rmtree(CODE, ignore_errors=True); shutil.copytree(trees[0], CODE, symlinks=False)
runs = [pathlib.Path(p) for p in glob.glob("/kaggle/input/**/rollouts/smolvla/fp16/libero_spatial/run.json", recursive=True)]
runs = [p for p in runs if p.parents[3].name == "rollouts"]
assert runs, "attach the go/no-go notebook's output (Version 8)"
ROLL = runs[0].parents[3]
print("code:", trees[0]); print("rollouts:", ROLL)
print({p.name: len(list((p / "libero_spatial").glob("*.npz"))) for p in sorted((ROLL / "smolvla").iterdir())})
RES = pathlib.Path("/kaggle/working/results"); RES.mkdir(exist_ok=True)
if (CODE / "results").exists() or (CODE / "results").is_symlink():
    (CODE / "results").unlink() if (CODE / "results").is_symlink() else shutil.rmtree(CODE / "results")
(CODE / "results").symlink_to(RES)
os.chdir(CODE)
""")
code(r"""
# 2. Re-score with each monitor feature set (both layers, then action expert only)
OUT = {}
for name, taps in (("both_layers", "vlm_last,expert_last"), ("expert_only", "expert_last")):
    r = subprocess.run([sys.executable, "scripts/03_evaluate.py", "--model", "smolvla", "--key", "libero_spatial",
                        "--monitors", "mahalanobis,mahalanobis#A", "--taps", taps, "--rollouts", str(ROLL)],
                       capture_output=True, text=True)
    print(name, "->", (r.stdout + r.stderr)[-600:])
    assert r.returncode == 0, name
    src = RES / "eval_trials/smolvla__libero_spatial.csv"
    OUT[name] = RES / f"trials_{name}.csv"; shutil.copy(src, OUT[name])
""")
code(r"""
# 3. The four groups of paired test trials, per monitor and compressed variant (alpha = 0.10)
import pandas as pd, numpy as np
rows = []
for name, path in OUT.items():
    t = pd.read_csv(path); t = t[(t.alpha.round(3) == 0.10) & (t.ref_success >= 0)]
    for (mon, v), d in t.groupby(["monitor", "variant"]):
        js = d[(d.ref_success == 1) & (d.var_success == 1)]          # both succeed
        new_f = d[(d.ref_success == 1) & (d.var_success == 0)]       # NEW failure caused by compression
        old_f = d[(d.ref_success == 0) & (d.var_success == 0)]       # both fail
        ref_f = d[d.ref_success == 0]
        rows.append(dict(features=name, monitor=mon, variant=v, n=len(d),
                         success_fp16=d.ref_success.mean(), success_var=d.var_success.mean(),
                         joint_succ=len(js), FPR_fp16=js.ref_alarm.mean(), FPR_var=js.var_alarm.mean(),
                         detect_fp16_own_failures=ref_f.ref_alarm.mean(),
                         n_new_failures=len(new_f), detect_NEW_failures=new_f.var_alarm.mean() if len(new_f) else np.nan,
                         n_shared_failures=len(old_f), detect_shared_failures=old_f.var_alarm.mean() if len(old_f) else np.nan))
res = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
res.to_csv("/kaggle/working/closed_loop_groups.csv", index=False)
print("\nSaved /kaggle/working/closed_loop_groups.csv -- send me this file (Output tab) or paste the table.")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("/home/claude/VLA_expert_eval.ipynb", "w"), indent=1)
print("cells", len(cells))
