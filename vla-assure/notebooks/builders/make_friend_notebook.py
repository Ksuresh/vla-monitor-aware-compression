"""Builds VLA_stage2_variants.ipynb: the next SmolVLA variants (whole-LM compression, W8, runtime-config shift),
runnable on a fresh Kaggle account. Reuses the setup cells of VLA_gonogo_kaggle.ipynb; needs the reference runs
(dataset gonogo-v5) and the latest vla-assure code as inputs. Two commits: PART = 1, then PART = 2."""
import json

base = json.load(open("/home/claude/VLA_gonogo_kaggle.ipynb"))["cells"]
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}  # noqa: E731
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}  # noqa: E731

C = []
C.append(md(r"""
# Stage 2: more model changes on SmolVLA (no new reference runs)
Reuses the 400 reference episodes from the go/no-go (dataset **gonogo-v5**) and runs new variants on the same 140
paired test slots, then re-scores every saved run with 5 monitor families x 3 placements, plus 20 random re-splits of calibration.

| PART | Runs | Approx. 2xT4 time |
|---|---|---|
| 1 | shadow replay on repair slots (W4-LM, W8, 5 denoising steps) + **W4 on the VLM language model + expert** (test + repair slots) | ~9 h |
| 2 | **W8 on the expert**, test + repair slots | ~5 h |
| 3 | **5 denoising steps** (runtime-config shift, same weights), test + repair slots | ~4 h |
| 4 | **5 actions per call** (runtime-config shift) + W4 calibration pool (repair slots) | ~7 h |

**Setup (once):** Settings -> Accelerator **GPU T4 x2**, Internet **On**. Add Input: dataset **vla-assure** (latest),
dataset **gonogo-v5**. Then set `PART = 1` in cell 2a, **Save Version -> Save & Run All (Commit)**.
**Then:** `PART = 2`, `3`, `4`, one commit each (no need to attach earlier outputs).
**Send back:** `stage2_partN.zip` from each version's Output tab.
"""))
C += [base[1], base[2]]
C.append(code(r'''
# 1b. Code: the newest vla-assure copy among the inputs (the one with the stage-2 variants)
import zipfile
CODE = pathlib.Path("/root/vla-assure")

def code_ok(root):
    f = root / "src/assure/adapters/lerobot_adapter.py"
    return f.exists() and "_chunk_method" in f.read_text() and "_require_lerobot_06" in f.read_text()

def newest(root):
    root = pathlib.Path(root)
    f, y = root / "scripts/shadow_check.py", root / "configs/experiment.yaml"
    return (f.exists() and "all_taps" in f.read_text() and y.exists() and "w4_lm" in y.read_text()
            and (root / "scripts/resplit.py").exists())

if CODE.exists() and not (code_ok(CODE) and newest(CODE)):
    shutil.rmtree(CODE)
if not CODE.exists():
    trees = sorted({str(pathlib.Path(p).parents[3]) for p in
                    glob.glob("/kaggle/input/**/src/assure/adapters/lerobot_adapter.py", recursive=True)})
    trees = [t for t in trees if newest(t)]
    if not trees:
        raise FileNotFoundError("No vla-assure copy with the stage-2 variants (w4_lm + resplit.py) under /kaggle/input: "
                                "upload the newest vla-assure.zip as a dataset version and attach it")
    print("Using unpacked dataset", trees[0])
    shutil.copytree(trees[0], CODE, symlinks=False)
assert code_ok(CODE), "This code is outdated: upload the latest vla-assure.zip"
for d in ("rollouts", "results"):
    (WORK / d).mkdir(exist_ok=True)
    tgt = CODE / d
    if tgt.is_symlink():
        tgt.unlink()
    elif tgt.exists():
        shutil.rmtree(tgt)
    tgt.symlink_to(WORK / d)
os.chdir(CODE); print("Code OK at", os.getcwd())
'''))
C += [base[4], base[5], base[6]]
cfg = base[7]["source"]
cfg = cfg.replace("R_REF = 50 ", "R_REF = 40 ")
i, j = cfg.index("CONDITION = (WORK"), cfg.index('print("taps:"')
cfg = cfg[:i] + ('CONDITION, KEY = "nominal", SUITE   # fixed by the go/no-go\n'
                 'VARIANTS = []                         # set per PART in cell 2a\n') + cfg[j:]
C.append(code(cfg))
C.append(base[8])
C.append(md("## 2. Stage-2 runs"))
C.append(code(r'''
# 2a. Which part this commit runs (1, 2, 3, then 4)
PART = 1
VARIANTS = {1: ["w4_lm"], 2: ["w8"], 3: ["cfg_denoise5"], 4: ["cfg_steps5", "w4"]}[PART]
# test slots = paired endpoint; repair slots + top-ups (until 44 successes) = the changed policy's OWN calibration
# pool, disjoint from test, so recalibration (19 marginal / 44 PAC) is judged on independent episodes.
ROLES = {"w4": "repair"}   # w4 test slots already exist from the go/no-go; only its calibration pool is new
ref = CODE / f"rollouts/{MODEL}/fp16/{KEY}"
n_ref = len(list(ref.glob("*.npz")))
print("PART", PART, "| variants:", VARIANTS, "| reference episodes on disk:", n_ref)
assert n_ref >= 10 * R_REF, "reference runs missing: attach the gonogo-v5 dataset as input"
if "w4" in VARIANTS:
    n_w4 = len(list((CODE / f"rollouts/{MODEL}/w4/{KEY}").glob("*.npz")))
    print("w4 episodes on disk (go/no-go test slots):", n_w4)
    assert n_w4 >= 140, "w4 test episodes missing: attach the gonogo-v5 dataset (it holds the go/no-go W4 run)"
'''))
C.append(code(r'''
# 2b. Shadow replay on identical frames (PART 1 only; ~1.5-2 h): W4-LM, W8 and 5 denoising steps see the fp16 robot's frames.
#     Runs on REPAIR slots, disjoint from the closed-loop test slots, so the prediction is judged out of sample.
if PART == 1:
    collect(["--model", MODEL, "--variant", "fp16", "--shadow", "w4_lm,w8,cfg_denoise5", "--suite", SUITE,
             "--condition", CONDITION, "--episodes", str(R_REF), "--ref_roles", "repair", "--max_per_task", "3",
             "--out", str(WORK / "rollouts_shadow2")])
    for taps in ("vlm_last,expert_last", "expert_last", "vlm_last"):
        run(["scripts/shadow_check.py", "--model", MODEL, "--key", KEY, "--shadow_root", str(WORK / "rollouts_shadow2"),
             "--taps", taps, "--out", str(WORK / f"shadow2_{taps.replace(',', '+')}.json")], tail=1800)
'''))
C.append(code(r'''
# 2c. Closed-loop variants on the 140 paired test slots (resumable)
for v in VARIANTS:
    print("=" * 30, v)
    collect(["--model", MODEL, "--variant", v, "--suite", SUITE, "--condition", CONDITION,
             "--episodes", str(R_REF), "--roles", ROLES.get(v, "test,repair"),
             "--target_successes", "44", "--max_extra_rounds", "4"], tail=600)
    rj = json.loads((CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json").read_text())
    print(f"{v}: success {rj['success_rate']:.3f} | sampler {rj.get('sampler')}")
'''))
C.append(code(r'''
# 2d. Re-score every saved run with 5 monitor families x 3 placements (CPU part, ~30-60 min)
import pandas as pd, numpy as np
OUT = {}
for name, taps in (("both_layers", "vlm_last,expert_last"), ("expert_only", "expert_last"), ("vlm_only", "vlm_last")):
    print("scoring", name, flush=True)
    run(["scripts/03_evaluate.py", "--model", MODEL, "--key", KEY,
         "--monitors", "mahalanobis,rnd,safe_mlp,acc,ext_mahalanobis", "--taps", taps], tail=400)
    OUT[name] = WORK / f"trials_part{PART}_{name}.csv"
    shutil.copy(CODE / f"results/eval_trials/{MODEL}__{KEY}.csv", OUT[name])
for name, taps in (("both_layers", "vlm_last,expert_last"), ("expert_only", "expert_last"), ("vlm_only", "vlm_last")):
    run(["scripts/resplit.py", "--model", MODEL, "--key", KEY, "--taps", taps, "--variants", ",".join(VARIANTS),
         "--splits", "20", "--out", str(WORK / f"resplit_part{PART}_{name}.json")], tail=3000)
rows = []
for name, path in OUT.items():
    t = pd.read_csv(path); t = t[(t.alpha.round(3) == 0.10) & (t.ref_success >= 0) & t.variant.isin(VARIANTS)]
    for (mon, v), d in t.groupby(["monitor", "variant"]):
        js = d[(d.ref_success == 1) & (d.var_success == 1)]
        nf = d[(d.ref_success == 1) & (d.var_success == 0)]
        rows.append(dict(features=name, monitor=mon, variant=v, success_fp16=d.ref_success.mean(),
                         success_var=d.var_success.mean(), joint_succ=len(js), FPR_fp16=js.ref_alarm.mean(),
                         FPR_var=js.var_alarm.mean(), n_new_failures=len(nf),
                         detect_NEW_failures=nf.var_alarm.mean() if len(nf) else np.nan))
res = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250); print(res.to_string(index=False))
res.to_csv(WORK / f"groups_part{PART}.csv", index=False)
with zipfile.ZipFile(f"/kaggle/working/stage2_part{PART}.zip", "w") as z:
    for p in list(OUT.values()) + [WORK / f"groups_part{PART}.csv"] + list(WORK.glob("shadow2_*.json")) + list(WORK.glob(f"resplit_part{PART}_*.json")):
        z.write(p, p.name)
    for v in VARIANTS:
        z.write(CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json", f"run_{v}.json")
print(f"Send me /kaggle/working/stage2_part{PART}.zip (Output tab)")
'''))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                               "language_info": {"name": "python"},
                               "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("/home/claude/VLA_stage2_variants.ipynb", "w"), indent=1)
print("cells", len(C))
