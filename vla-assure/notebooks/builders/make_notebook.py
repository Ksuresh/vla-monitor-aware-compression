"""Builds VLA_gonogo.ipynb: the go/no-go experiment for monitor-aware compression (SmolVLA, LIBERO-Spatial, T4)."""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Go / no-go: does alarm-aware compression preserve a VLA's failure monitor better than action-aware compression?

**Model:** SmolVLA (`HuggingFaceVLA/smolvla_libero`) · **Benchmark:** LIBERO-Spatial · **GPU:** one T4 · **Code:** `vla-assure` pipeline

**Question.** If we quantize SmolVLA's action expert to 4 bits:
1. Does the inherited Mahalanobis failure monitor's false-alarm rate move? (If not, there is nothing to fix.)
2. When we keep ~15% of the expert in full precision, does choosing those blocks by their effect on the **alarm statistic** keep the monitor closer to fp16 than choosing them by **action** change or at random?

**Scope (deliberately small).** Only the action expert is quantized (fake INT4, group 128); the VLM stays bf16. One suite, one camera condition, one monitor. This is a decision gate, not the paper's experiment: no verdicts, no p-values; estimates with task-clustered intervals and a pre-stated decision table.

**How to run.** Top to bottom. Everything is written to Google Drive, and every collection step resumes where it stopped (`--skip_existing`), so a Colab disconnect costs nothing: reconnect, run the *Setup* cells again, and continue. If pip asks you to restart the session, restart and continue from cell 2.

| Step | What | Episodes | Rough T4 time* |
|---|---|---|---|
| 1 | Setup, install, smoke test | 2 | 20 min |
| 2 | Stage 0-lite: fp16 nominal sanity | 100 | 1–2 h |
| 3 | Pick the camera condition (60–85 % success) | 100 | 1–2 h |
| 4 | Reference fp16 run, 50 slots/task | 500 | 4–8 h |
| 5 | Block sensitivity (alarm-aware and action-aware rankings) | ≤ 96 captured | 1–2 h |
| 6 | Four compressed variants on the 175 test slots | 700 | 6–10 h |
| 7 | Evaluation and decision table | — | 5 min |

\*Measured in step 1 and re-estimated there; run steps 4 and 6 across several sessions if needed.
""")

md("## 1. Setup")
code(r"""
# 1a. Runtime checks: GPU and Python 3.12 (LeRobot 0.6.1 needs Python >= 3.12)
import sys, subprocess
print("Python", sys.version.split()[0])
assert sys.version_info >= (3, 12), "LeRobot 0.6.1 needs Python 3.12+: Runtime > Change runtime type > use the latest runtime"
if sys.version_info >= (3, 13):
    print("Note: Python 3.13 runtime. LeRobot 0.6.1 supports it, but if 1d fails on a dependency, switch to a "
          "3.12 runtime: Runtime > Change runtime type > Runtime version")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
""")
code(r"""
# 1b. Persistent storage on Google Drive (all rollouts and results live here).
#     If mounting fails with "credential propagation was unsuccessful": make sure Colab and Drive use the SAME
#     Google account (top-right avatar), allow third-party cookies for colab.research.google.com, and do not
#     use an incognito window; then rerun this cell. If it still fails, set USE_DRIVE = False: everything is
#     kept on the Colab disk instead and LOST on disconnect, so download the results at the end of each session.
USE_DRIVE = True
import os, pathlib
WORK = None
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        WORK = pathlib.Path("/content/drive/MyDrive/vla-gonogo")
    except Exception as e:
        print("Drive mount failed:", e, "\n-> falling back to local disk (not persistent)")
if WORK is None:
    WORK = pathlib.Path("/content/vla-gonogo")
WORK.mkdir(parents=True, exist_ok=True)
PERSISTENT = str(WORK).startswith("/content/drive")
print("Work folder:", WORK, "| persistent:", PERSISTENT)
""")
code(r"""
# 1c. Code: the LATEST vla-assure.zip is required (the GitHub copy predates the SmolVLA / LeRobot 0.6 fixes).
#     Looks for it in Drive (MyDrive/vla-gonogo/ or MyDrive/) and /content; otherwise opens an upload box.
import shutil, zipfile, glob
CODE = pathlib.Path("/content/vla-assure")

def code_ok(root):
    f = root / "src/assure/adapters/lerobot_adapter.py"
    return f.exists() and "_chunk_method" in f.read_text() and "_require_lerobot_06" in f.read_text()

if CODE.exists() and not code_ok(CODE):
    print("Removing an outdated copy at", CODE)
    shutil.rmtree(CODE)
if not CODE.exists():
    cands = [WORK / "vla-assure.zip", pathlib.Path("/content/drive/MyDrive/vla-assure.zip"),
             pathlib.Path("/content/vla-assure.zip")] + [pathlib.Path(p) for p in
             glob.glob("/content/drive/MyDrive/**/vla-assure*.zip", recursive=True)[:5]]
    z = next((c for c in cands if c.exists()), None)
    if z is None:
        from google.colab import files
        print("vla-assure.zip not found in Drive. Upload it now (the file I sent, ~150 KB):")
        up = files.upload()
        name = next((n for n in up if n.endswith(".zip")), None)
        assert name, "no zip uploaded"
        z = WORK / "vla-assure.zip"; z.write_bytes(up[name])
    print("Using", z)
    tmp = pathlib.Path("/content/_unzip"); shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    root = next((p.parents[3] for p in tmp.rglob("src/assure/adapters/lerobot_adapter.py")), None)
    assert root is not None, f"{z} does not contain the vla-assure code"
    shutil.move(str(root), CODE)
assert code_ok(CODE), "This zip is outdated: use the vla-assure.zip from the latest message"
# rollouts/ and results/ live in WORK (Drive when mounted) so they survive disconnects
for d in ("rollouts", "results"):
    (WORK / d).mkdir(exist_ok=True)
    tgt = CODE / d
    if tgt.is_symlink():
        tgt.unlink()
    elif tgt.exists():
        shutil.rmtree(tgt)
    tgt.symlink_to(WORK / d)
os.chdir(CODE); print("Code OK at", os.getcwd())
""")
code(r"""
# 1d. Install LeRobot 0.6.1 with SmolVLA and LIBERO (one environment; no separate simulator env needed).
#     egl_probe / hf-egl-probe (pulled in via hf-libero -> robomimic) are C++ packages that fail to build on
#     Colab. Only robomimic's own env wrappers import them; LeRobot's LIBERO env never does. So we install
#     tiny pure-Python stand-ins first, and pip then skips the build. (Tested with CMake 4.1 / Python 3.13.)
import os, pathlib, subprocess, sys
os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
stub = pathlib.Path("/content/_stubs")
(stub / "egl_probe/egl_probe").mkdir(parents=True, exist_ok=True); (stub / "hf_egl_probe").mkdir(parents=True, exist_ok=True)
(stub / "egl_probe/pyproject.toml").write_text('[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n'
    '[project]\nname = "egl_probe"\nversion = "1.0.2"\n[tool.setuptools]\npackages = ["egl_probe"]\n')
(stub / "egl_probe/egl_probe/__init__.py").write_text("def get_available_devices():\n    return [0]\n")
(stub / "hf_egl_probe/pyproject.toml").write_text('[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n'
    '[project]\nname = "hf-egl-probe"\nversion = "1.0.2"\n[tool.setuptools]\npy-modules = []\n')
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(stub / "egl_probe"), str(stub / "hf_egl_probe")],
                   capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-1500:]); assert r.returncode == 0, "stub install failed"
%pip install -q "lerobot[smolvla,libero]==0.6.1" pandas scipy tabulate pyyaml
""")
code(r"""
# 1e. LIBERO's first import asks interactive questions and downloads its assets from the Hugging Face Hub.
#     Answer them non-interactively once; set headless GPU rendering.
import os, subprocess, sys
os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
r = subprocess.run([sys.executable, "-c", "import libero.libero as L; print('assets:', L.get_libero_path('assets'))"],
                   input="n\n", capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
import importlib.metadata as md
print("lerobot", md.version("lerobot"), "| torch", md.version("torch"), "| transformers", md.version("transformers"))
""")
code(r"""
# 1f. Experiment configuration for this gate (written into the repo copy on this machine only)
import yaml, json
CFG = CODE / "configs"
exp = yaml.safe_load((CFG / "experiment.yaml").read_text())
models = yaml.safe_load((CFG / "models.yaml").read_text())

MODEL, SUITE = "smolvla", "libero_spatial"
R_REF = 50              # reference slots per task (500 slots: 125 train / 100 calib / 100 repair / 175 test)
BUDGET_PCT = 15         # % of the ACTION EXPERT's linear weights kept in full precision (~5 of 32 blocks)
N_ACTION_STEPS = None   # None = the checkpoint's own value (as lerobot-eval); set 10 if step 1g shows it is too slow

exp["quant_scope"] = {"include": ["model.vlm_with_expert.lm_expert.*"],
                      "block_regex": r"lm_expert\.layers\.(\d+)$"}
exp["monitors"] = ["mahalanobis#A", "mahalanobis"]
exp["reference_episodes_per_task"] = R_REF
models["models"][MODEL]["checkpoint"] = "HuggingFaceVLA/smolvla_libero"
models["models"][MODEL]["n_action_steps"] = N_ACTION_STEPS
(CFG / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
(CFG / "models.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
CONDITION = (WORK / "condition.txt").read_text().strip() if (WORK / "condition.txt").exists() else None
KEY = f"{SUITE}+{CONDITION}" if CONDITION else None
print("condition from an earlier session:", CONDITION)
VARIANTS = ["w4", f"w4p-margin_drift-f{BUDGET_PCT}", f"w4p-action_drift-f{BUDGET_PCT}", f"w4p-random-f{BUDGET_PCT}"]
print("variants:", VARIANTS)
print("taps:", models["models"][MODEL]["tap"])

def run(cmd, tail=3000):
    # Run a pipeline script, stream nothing, print the end of the log, raise on failure.
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", TOKENIZERS_PARALLELISM="false")
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=env, input="n\n")
    out = (p.stdout + p.stderr).replace("\r", "\n")
    out = "\n".join(l for l in out.splitlines() if "Running rollout" not in l and "it/s" not in l)
    print(out[-tail:])
    if p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(cmd)}")
""")
code(r"""
# 1g. Smoke test: model loads, hooks fire, camera shift applied, 2 episodes run. Measures seconds per episode.
run(["scripts/01_collect.py", "--model", MODEL, "--variant", "fp16", "--suite", SUITE, "--condition", "nominal",
     "--tasks", "0", "--episodes", "2", "--out", str(WORK / "rollouts_smoke")])
rj = json.loads((WORK / f"rollouts_smoke/{MODEL}/fp16/{SUITE}/run.json").read_text())
sec = rj["seconds"] / max(rj["episodes"], 1)
print(f"\nsuccess {rj['success_rate']:.2f} | taps fired {rj['tap_fired']} | {sec:.0f} s/episode")
assert all(v > 0 for v in rj["tap_fired"].values()), "feature hooks did not fire: check tap paths in models.yaml"
for step, n in (("Stage 0-lite", 100), ("condition pilot", 100), ("reference", 10 * R_REF), ("variants", 4 * 175)):
    print(f"  {step:16s} {n:5d} episodes ~ {n * sec / 3600:5.1f} h")
print("If 'reference' + 'variants' exceeds ~20 h, set N_ACTION_STEPS = 10 in 1f, rerun 1f and 1g.")
""")

md("""
## 2. Stage 0-lite: does fp16 SmolVLA behave as published?
Ten trials per task on nominal LIBERO-Spatial. This only catches a broken setup (wrong preprocessing, wrong checkpoint). The full compatibility check (500 episodes) is part of the paper's Stage 0, not this gate. Fill in the success rate you compare against (model card or the SmolVLA paper).
""")
code(r"""
PUBLISHED_SPATIAL = None   # e.g. 0.90 -- fill in from the source you compare against
run(["scripts/01_collect.py", "--model", MODEL, "--variant", "fp16", "--suite", SUITE, "--condition", "nominal",
     "--episodes", "10", "--skip_existing", "--out", str(WORK / "rollouts_stage0")])
if PUBLISHED_SPATIAL:
    run(["scripts/stage0_check.py", "--model", MODEL, "--key", SUITE, "--published", str(PUBLISHED_SPATIAL),
         "--rollouts", str(WORK / "rollouts_stage0")])
else:
    rj = json.loads((WORK / f"rollouts_stage0/{MODEL}/fp16/{SUITE}/run.json").read_text())
    print("fp16 nominal success:", round(rj["success_rate"], 3), "over", rj["episodes"], "episodes")
""")

md("""
## 3. Pick the camera condition
The monitor needs failures and the paired analysis needs successes: target 60–85 % fp16 success. Five trials per task under each seeded camera shift; the condition closest to 72 % inside the window is chosen automatically (override `CONDITION` if you prefer).
""")
code(r"""
pilot = {}
for cond in ("shift_mild", "shift_strong"):
    run(["scripts/01_collect.py", "--model", MODEL, "--variant", "fp16", "--suite", SUITE, "--condition", cond,
         "--episodes", "5", "--skip_existing", "--out", str(WORK / "rollouts_pilot")], tail=400)
    pilot[cond] = json.loads((WORK / f"rollouts_pilot/{MODEL}/fp16/{SUITE}+{cond}/run.json").read_text())["success_rate"]
print("pilot success:", pilot)
inside = {c: p for c, p in pilot.items() if 0.60 <= p <= 0.85}
CONDITION = min(inside or pilot, key=lambda c: abs(pilot[c] - 0.72))
KEY = f"{SUITE}+{CONDITION}"
(WORK / "condition.txt").write_text(CONDITION)
print("CONDITION =", CONDITION, "| KEY =", KEY,
      "" if inside else "\nWARNING: no condition in 60-85 %: edit experiment.yaml conditions (noise/brightness) and rerun")
""")

md("""
## 4. Reference run (fp16): 500 pre-registered slots
Every slot has a fixed role (train / calibration / repair / test), seed, initial state and camera draw. The monitor is trained on the training slots only; calibration slots fit the conformal band; test slots are what every variant is compared on. **Resumable**: rerun this cell after a disconnect.
""")
code(r"""
run(["scripts/01_collect.py", "--model", MODEL, "--variant", "fp16", "--suite", SUITE, "--condition", CONDITION,
     "--episodes", str(R_REF), "--skip_existing"])
rj = json.loads((CODE / f"rollouts/{MODEL}/fp16/{KEY}/run.json").read_text())
n_files = len(list((CODE / f"rollouts/{MODEL}/fp16/{KEY}").glob("*.npz")))
print(f"reference: {n_files}/{10 * R_REF} episodes on disk, success {rj['success_rate']:.3f}")
assert n_files >= 10 * R_REF, "not finished yet: rerun this cell"
""")

md("""
## 5. Block sensitivity: alarm-aware and action-aware rankings
Captured reference episodes (outside the slot partition) are replayed with one expert block quantized at a time. The **alarm-aware** score is the change of the monitor's alarm statistic, averaged over successes and failures; the **action-aware** score is the change of the predicted action chunk. Rankings use the selection half of the captured episodes only.
""")
code(r"""
run(["scripts/02_sensitivity.py", "--model", MODEL, "--suite", SUITE, "--condition", CONDITION, "--ref_key", KEY,
     "--monitor", "mahalanobis#A", "--select_episodes", "16", "--holdout_episodes", "16",
     "--min_failures", "6", "--min_successes", "6", "--max_captured", "96", "--max_queries", "12", "--rank_boot", "200"],
    tail=2500)
sens = json.loads((CODE / f"results/sensitivity/{MODEL}.json").read_text())
import pandas as pd
blk = pd.DataFrame(sens["blocks"])
blk["block"] = blk["name"].str.extract(r"layers\.(\d+)$").astype(int)
blk["type"] = (blk["block"] % 2).map({0: "self-attn", 1: "cross-attn"})
cols = [c for c in ("block", "type", "margin_drift_sel", "action_drift_sel", "margin_drift", "action_drift") if c in blk]
print(blk.sort_values(cols[2], ascending=False)[cols].head(10).to_string(index=False))
print("\nranking uncertainty (mean bootstrap Jaccard):",
      {k: round(v.get("mean_jaccard", float("nan")), 2) for k, v in (sens.get("ranking_uncertainty") or {}).items()})
""")

md("""
## 6. Four compressed variants on the 175 test slots
W4 on the whole expert, and W4 with 15 % of the expert protected by three rules at matched memory: alarm-aware (`margin_drift`), action-aware (`action_drift`), random. All run the same paired test slots as the reference. **Resumable.**
""")
code(r"""
for v in VARIANTS:
    print("=" * 30, v)
    run(["scripts/01_collect.py", "--model", MODEL, "--variant", v, "--suite", SUITE, "--condition", CONDITION,
         "--episodes", str(R_REF), "--roles", "test", "--skip_existing"], tail=600)
    rj = json.loads((CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json").read_text())
    print(f"{v}: success {rj['success_rate']:.3f}, protected {rj.get('protected_frac_actual', 0):.3f} of the expert,"
          f" blocks {[b.split('.')[-1] for b in rj.get('protected_blocks', [])]}")
""")

md("## 7. Evaluation and decision")
code(r"""
run(["scripts/03_evaluate.py", "--model", MODEL, "--key", KEY, "--monitors", "mahalanobis#A,mahalanobis"], tail=1500)
tr = pd.read_csv(CODE / f"results/eval_trials/{MODEL}__{KEY}.csv")
tr = tr[(tr.alpha.round(3) == 0.10) & (tr.monitor == "mahalanobis#A")]
print(len(tr), "trial rows;", tr.variant.unique())
""")
code(r"""
import numpy as np
rng = np.random.default_rng(0)

def boot(df, stat, B=2000):
    # task-clustered bootstrap: resample tasks, keep all their trials together
    tasks = df.task.unique(); vals = []
    groups = {t: df[df.task == t] for t in tasks}
    for _ in range(B):
        s = pd.concat([groups[t] for t in rng.choice(tasks, len(tasks))])
        vals.append(stat(s))
    vals = np.array([v for v in vals if v == v])
    return stat(df), np.percentile(vals, 5), np.percentile(vals, 95)

rows = []
for v in VARIANTS:
    d = tr[tr.variant == v]
    js = d[(d.ref_success == 1) & (d.var_success == 1)]
    f = d[d.var_success == 0]
    est, lo, hi = boot(js, lambda s: (s.var_alarm - s.ref_alarm).mean())
    rows.append(dict(variant=v, success=d.var_success.mean(), fp16_success=d.ref_success.mean(),
                     joint_successes=len(js), FPR_fp16=js.ref_alarm.mean(), FPR_variant=js.var_alarm.mean(),
                     D_FPR=est, D_FPR_90lo=lo, D_FPR_90hi=hi,
                     TPR_variant=f.var_alarm.mean() if len(f) else np.nan, failures=len(f)))
summary = pd.DataFrame(rows).round(3)
print("Paired false-alarm change vs fp16 on jointly successful test trials (90% task-clustered interval)")
print(summary.to_string(index=False))

def contrast(a, b):
    m = tr[tr.variant == a].merge(tr[tr.variant == b], on=["task", "idx"], suffixes=("_a", "_b"))
    js = m[(m.var_success_a == 1) & (m.var_success_b == 1)]
    jf = m[(m.var_success_a == 0) & (m.var_success_b == 0)]
    fpr = boot(js, lambda s: (s.var_alarm_a - s.var_alarm_b).mean())
    succ = boot(m, lambda s: (s.var_success_a - s.var_success_b).mean())
    tpr = (jf.var_alarm_a - jf.var_alarm_b).mean() if len(jf) else np.nan
    return dict(pair=f"{a} - {b}", joint_successes=len(js), FPR_diff=fpr[0], FPR_90lo=fpr[1], FPR_90hi=fpr[2],
                success_diff=succ[0], joint_failures=len(jf), TPR_diff=tpr)

A, T, Rn = VARIANTS[1], VARIANTS[2], VARIANTS[3]
con = pd.DataFrame([contrast(A, T), contrast(A, Rn), contrast(T, Rn)]).round(3)
print("\nRule-vs-rule (negative FPR_diff = first rule has fewer false alarms)")
print(con.to_string(index=False))
""")
code(r"""
# Pre-stated decision table (planning thresholds for this gate, not significance tests)
w4 = summary.set_index("variant").loc["w4"]
c_at = con.iloc[0]
w4_moves = (w4.D_FPR >= 0.03) and (w4.D_FPR_90lo > 0)
method_better = (c_at.FPR_diff <= -0.02) and (c_at.FPR_90hi < 0.01) and (c_at.success_diff >= -0.05) \
                and not (c_at.TPR_diff == c_at.TPR_diff and c_at.TPR_diff < -0.10)
action_fixes = summary.set_index("variant").loc[T].D_FPR <= 0.01
print(f"W4 moves false alarms (D_FPR >= 0.03, 90% lo > 0): {w4_moves}  [D_FPR = {w4.D_FPR:+.3f}]")
print(f"alarm-aware beats action-aware (FPR -0.02 or better, success within 0.05, detection within 0.10): {method_better}")
if not w4_moves:
    verdict = "NO-GO for the method: the monitor is robust to expert W4 here. Write up as a robustness audit; consider W4 of the VLM too before giving up."
elif method_better:
    verdict = "GO: W4 degrades the monitor and the alarm-aware rule preserves it better than the action-aware rule. Proceed to Stage A."
elif action_fixes:
    verdict = "NO-GO for the method: action-aware protection already preserves the monitor. Report the degradation finding."
else:
    verdict = "INCONCLUSIVE: W4 degrades the monitor but the rules do not separate. Check the ranking uncertainty and layer types before deciding; consider more test slots."
print("\nDECISION:", verdict)

# Layer-type composition of each protected set (SmolVLA expert: even = self-attention, odd = cross-attention)
for v in VARIANTS[1:]:
    rj = json.loads((CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json").read_text())
    ids = [int(b.split(".")[-1]) for b in rj.get("protected_blocks", []) if b.split(".")[-1].isdigit()]
    print(f"{v:28s} whole blocks {sorted(ids)}  self-attn {sum(i % 2 == 0 for i in ids)} / cross-attn {sum(i % 2 for i in ids)}"
          f"  filler share {rj.get('filler_share', 0):.2f}")
""")
code(r"""
# Save a compact bundle to send back for analysis
out = WORK / "gonogo_summary"; out.mkdir(exist_ok=True)
summary.to_csv(out / "variants.csv", index=False); con.to_csv(out / "contrasts.csv", index=False)
(out / "decision.json").write_text(json.dumps({"condition": CONDITION, "budget_pct_of_expert": BUDGET_PCT,
    "n_action_steps": N_ACTION_STEPS, "verdict": verdict, "w4_moves": bool(w4_moves),
    "method_better": bool(method_better)}, indent=2))
shutil.copy(CODE / f"results/sensitivity/{MODEL}.json", out / "sensitivity.json")
shutil.copy(CODE / f"results/eval_trials/{MODEL}__{KEY}.csv", out / "eval_trials.csv")
for v in ["fp16"] + VARIANTS:
    shutil.copy(CODE / f"rollouts/{MODEL}/{v}/{KEY}/run.json", out / f"run_{v}.json")
shutil.make_archive(str(WORK / "gonogo_summary"), "zip", out)
print("Send this file:", WORK / "gonogo_summary.zip")
if not PERSISTENT:
    from google.colab import files
    files.download(str(WORK / "gonogo_summary.zip"))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("/home/claude/VLA_gonogo.ipynb", "w"), indent=1)
print("cells:", len(cells))
