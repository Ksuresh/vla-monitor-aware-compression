"""Builds VLA_pi05_full.ipynb: the second-model experiment (pi0.5) on Kaggle 2xT4, in commits of < 11 h.
PART 1: pick a condition with 55-88 % fp16 success, then the 400 pre-registered reference episodes.
PART 2: W4 on the action expert.   PART 3: W4 on language model + expert.   PART 4: W8 on the action expert.
Variants run on the 140 paired test slots only; recalibration is evaluated by cross-fitting (no repair pools).
Each part after 1 needs the previous part's output attached (it resumes from it). Both GPUs run in parallel."""
import pathlib
NB = pathlib.Path(__file__).resolve().parents[1]   # vla-assure/notebooks: builders read and write notebooks here
import json

base = json.load(open(NB / "VLA_gonogo_kaggle.ipynb"))["cells"]
pil = json.load(open(NB / "VLA_pi05_pilot.ipynb"))["cells"]
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}  # noqa: E731
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}  # noqa: E731

C = [md(r"""
# pi0.5: does the placement rule hold on a second model?
| PART | Runs | Approx. time | Inputs besides vla-assure |
|---|---|---|---|
| 1 | condition check (4 x 50 episodes at most) + 400 reference episodes | ~6-8 h | none |
| 2 | W4 on the action expert (140 paired test slots) | ~9.5 h | output of PART 1 |
| 3 | W4 on language model + expert (140 paired test slots) | ~9.5 h | output of PART 2 |
| 4 | W8 on the action expert (140 paired test slots) | ~9.5 h | output of PART 3 |

**Setup (every part):** GPU T4 x2, Internet On, secret `HF_TOKEN` switched on, Add Input: dataset **vla-assure** (latest)
and, from PART 2 on, **Notebook Output** of the previous version of THIS notebook. Set `PART` in cell 2a, then
**Save Version -> Save & Run All (Commit)**. **Send back:** `pi05_partN.zip` from the Output tab.
If a part stops at ~11 h with "OutOfTime", commit the SAME part again with that version's output attached.
""")]
C += [pil[1], pil[2], pil[3], pil[4], pil[5], pil[6], pil[7]]      # setup: GPUs/resume, code, token, venv, tokenizer, rendering
cfg = pil[8]["source"].replace('DTYPE = "float32"       # set by the dtype test in 2a ("float32" or "bfloat16")',
                               'DTYPE = "bfloat16"      # float32 does not fit a T4 (pilot)')
cfg = cfg.replace('CONDITION, KEY = "nominal", "libero_spatial"\n', '')
cfg = cfg.replace('R_REF = 50 ', 'R_REF = 40 ')
C.append(code(cfg))
C.append(code(r'''
# 1f. Two GPUs in parallel: shard g runs on cuda:g (rendering stays on GPU 0); shard 1 starts a few minutes later so
#     the two 3B-parameter loads do not overlap in host RAM. Top-ups (--target_successes) run in the merge pass only.
R_REF = 40
exp["reference_episodes_per_task"] = R_REF; write_cfg(DTYPE)
STAGGER_S = 300
def collect(args, tail=1500):
    if hours_left() < 0.5:
        raise OutOfTime("time budget reached: Save Version now; attach this output and run the same PART again")
    args = list(args) + ([] if "--skip_existing" in args else ["--skip_existing"])
    shard_args, skip = [], 0
    for k, x in enumerate(args):                 # top-ups only in the single merge pass
        if skip: skip -= 1; continue
        if x in ("--target_successes", "--max_extra_rounds"): skip = 1; continue
        shard_args.append(x)
    env = dict(os.environ); env.update(MUJOCO_GL=GL, PYOPENGL_PLATFORM=GL, TOKENIZERS_PARALLELISM="false",
                                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if NGPU > 1 and "--tasks" not in args:
        t0 = time.time(); procs, logs = [], []
        for g in range(NGPU):
            if g:
                time.sleep(STAGGER_S)
            logs.append(open(f"/root/shard{g}.log", "w"))
            e = dict(env); e["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NGPU))
            p = subprocess.Popen([PY, "scripts/01_collect.py"] + shard_args + ["--device", f"cuda:{g}", "--tasks",
                                 ",".join(str(t) for t in range(10) if t % NGPU == g)],
                                 env=e, stdout=logs[g], stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
            p.stdin.write("n\n"); p.stdin.close(); procs.append(p)
        bad = []
        while any(p.poll() is None for p in procs):
            bad = [g for g, p in enumerate(procs) if p.poll() not in (None, 0)]
            late = hours_left() * 3600 < 900
            if bad or late:
                for p in procs:
                    if p.poll() is None: p.kill()
                for p in procs: p.wait()
                _clean_partial(t0)
                if late and not bad:
                    raise OutOfTime("time budget reached during collection: Save Version; attach this output and rerun")
                break
            time.sleep(15)
        for f in logs: f.close()
        for g, p in enumerate(procs):
            if p.returncode != 0:
                print(open(f"/root/shard{g}.log").read()[-3000:])
                raise RuntimeError(f"shard {g} failed")
    e = dict(env); e["CUDA_VISIBLE_DEVICES"] = "0"
    p = subprocess.run([PY, "scripts/01_collect.py"] + args, capture_output=True, text=True, env=e, input="n\n")
    out = "\n".join(l for l in (p.stdout + p.stderr).replace("\r", "\n").splitlines() if "it/s" not in l)
    print(out[-tail:])
    if p.returncode != 0:
        raise RuntimeError("merge pass failed")
print("parallel collect ready on", NGPU, "GPU(s)")
'''))
C.append(md("## 2. Runs"))
C.append(code(r'''
# 2a. Which part this commit runs. Runs behind the paper (notebook49b3154248 on Kaggle):
#     PART 1 = versions 3-4 (reference), PART 2 = version 6 (w4), PART 3 = version 8 (w4_lm), PART 4 = version 9 (w8)
PART = 1
VARIANTS = {1: [], 2: ["w4"], 3: ["w4_lm"], 4: ["w8"]}[PART]
print("PART", PART, "| variants:", VARIANTS)
'''))
C.append(code(r'''
# 2b. Condition: the monitor needs failures and the paired analysis needs successes -> target 55-88 % fp16 success.
#     pi0.5 succeeds on 100 % of nominal LIBERO-Spatial and 98 % with the strong camera shift (checked), so the
#     harder settings are checked first (5 episodes per task, outside the slot partition); the first in range is
#     used, otherwise the one closest to 72 %. The choice is saved, so later parts reuse it.
cf = WORK / "pi05_condition.json"
if cf.exists():
    SUITE, CONDITION = json.loads(cf.read_text())["choice"]
else:
    assert PART == 1, "condition not found: attach the output of the previous part"
    tried = {("libero_spatial", "shift_strong"): 0.98}          # measured in the cancelled run
    for suite, cond in (("libero_10", "nominal"), ("libero_spatial", "action_noise"), ("libero_10", "shift_strong")):
        out = WORK / f"cond_pilot/{suite}_{cond}"
        collect(["--model", MODEL, "--variant", "fp16", "--suite", suite, "--condition", cond, "--episodes", "5",
                 "--out", str(out)], tail=400)
        rj = json.loads(next(out.rglob("run.json")).read_text())
        tried[(suite, cond)] = rj["success_rate"]
        print(f"  {suite} + {cond}: fp16 success {rj['success_rate']:.2f}", flush=True)
        if 0.55 <= rj["success_rate"] <= 0.88:
            break
    SUITE, CONDITION = min(tried, key=lambda k: abs(tried[k] - 0.72))
    cf.write_text(json.dumps({"choice": [SUITE, CONDITION], "tried": {f"{s}+{c}": v for (s, c), v in tried.items()}}))
KEY = SUITE if CONDITION == "nominal" else f"{SUITE}+{CONDITION}"
print("Using", SUITE, "+", CONDITION, "| key", KEY)
'''))
C.append(code(r'''
# 2c. Reference run (PART 1): 400 pre-registered slots (100 train / 80 calib / 80 repair / 140 test), resumable
ref = CODE / f"rollouts/{MODEL}/fp16/{KEY}"
if PART == 1 or len(list(ref.glob("*.npz"))) < 10 * R_REF:
    collect(["--model", MODEL, "--variant", "fp16", "--suite", SUITE, "--condition", CONDITION, "--episodes", str(R_REF)])
n_ref = len(list(ref.glob("*.npz")))
rj = json.loads((ref / "run.json").read_text())
print(f"reference episodes on disk: {n_ref} | fp16 success {rj['success_rate']:.3f}")
assert n_ref >= 10 * R_REF, "reference incomplete: attach the previous output and run PART 1 again"
'''))
C.append(code(r'''
# 2c'. Check that the reference episodes stored the monitor features (stops early if not)
import numpy as np
eps = sorted((CODE / f"rollouts/{MODEL}/fp16/{KEY}").glob("t*.npz"))
z = np.load(eps[0])
feats = {k: z[k].shape for k in z.files if k.startswith("feat__")}
print("episodes:", len(eps), "| features in first episode:", feats)
assert "feat__vlm_last" in feats and "feat__expert_last" in feats, "features missing: send me this output"
'''))
C.append(code(r'''
# 2d. Variants on the 140 paired test slots only (pi0.5 is slow on LIBERO-10; recalibration uses cross-fitting)
for v in VARIANTS:
    print("=" * 30, v, flush=True)
    collect(["--model", MODEL, "--variant", v, "--suite", SUITE, "--condition", CONDITION, "--episodes", str(R_REF),
             "--roles", "test"], tail=600)
    rj = json.loads((CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json").read_text())
    print(f"{v}: success {rj['success_rate']:.3f} | sampler {rj.get('sampler')}")
'''))
C.append(code(r'''
# 2e. Score this part's variants (CPU part): 5 monitor families x 3 placements + 20 re-splits + release options.
#     Time-guarded: a placement is scored only if it can finish before Kaggle's 12 h limit; the zip is always written.
import pandas as pd, numpy as np, traceback
done = sorted(p.name for p in (CODE / f"rollouts/{MODEL}").iterdir() if p.name != "fp16" and (p / KEY / "run.json").exists())
print("variants on disk:", done)
score = [v for v in done if v in VARIANTS] or done
print("scoring now:", score, "| hours left in budget:", round(hours_left(), 2))
files = [cf, CODE / f"rollouts/{MODEL}/fp16/{KEY}/run.json"]
for name, taps in (("vlm_only", "vlm_last"), ("expert_only", "expert_last"), ("both_layers", "vlm_last,expert_last")):
    if not score:
        break
    if hours_left() < 0.9:
        print(f"SKIPPED {name}: not enough time left (score it in the next run)")
        continue
    try:
        run(["scripts/03_evaluate.py", "--model", MODEL, "--key", KEY,
             "--monitors", "mahalanobis,rnd,safe_mlp,acc,ext_mahalanobis", "--taps", taps], tail=400)
        dst = WORK / f"pi05_trials_{name}.csv"; shutil.copy(CODE / f"results/eval_trials/{MODEL}__{KEY}.csv", dst); files.append(dst)
    except Exception:
        print("evaluate failed for", name); traceback.print_exc()
    if hours_left() < 0.6:
        print(f"SKIPPED resplit for {name}: not enough time left")
        continue
    try:
        run(["scripts/resplit.py", "--model", MODEL, "--key", KEY, "--taps", taps, "--variants", ",".join(score),
             "--splits", "20", "--out", str(WORK / f"pi05_resplit_{name}.json")], tail=6000)
        files.append(WORK / f"pi05_resplit_{name}.json")
    except Exception:
        print("resplit failed for", name); traceback.print_exc()
for v in done:
    files.append(CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json")
with zipfile.ZipFile(f"/kaggle/working/pi05_part{PART}.zip", "w") as z:
    for p in files:
        p = pathlib.Path(p)
        if p.exists():
            z.write(p, p.name if "rollouts" not in str(p) else f"run_{p.parent.parent.name}.json")
print(f"Send me /kaggle/working/pi05_part{PART}.zip (Output tab)")
'''))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                               "language_info": {"name": "python"},
                               "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(NB / "VLA_pi05_full.ipynb", "w"), indent=1)
print("cells", len(C))
