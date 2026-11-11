"""Builds VLA_gonogo_kaggle.ipynb from the Colab notebook: same experiment, Kaggle runtime (2x T4, 12 h runs).

Differences from the Colab version:
  - code from an uploaded Kaggle dataset; work folder /kaggle/working/vla-gonogo (saved as the notebook's output);
  - LeRobot 0.6.1 in its own Python 3.12 venv (Kaggle's kernel Python / packages are left alone);
  - collection is sharded over both T4s by task, then a merge pass writes the full run.json;
  - a wall-clock guard stops cleanly before Kaggle's 12 h limit, so the output is saved and the next run resumes
    from it (attach the previous version's output as input).
"""
import json
import runpy

ns = runpy.run_path("/home/claude/make_notebook.py")
C = ns["cells"]
assert len(C) == 24, len(C)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def sub(cell, old, new, count=1):
    s = cell["source"]
    assert s.count(old) >= 1, old
    cell["source"] = s.replace(old, new) if count is None else s.replace(old, new, count)


C[0] = md(r"""
# Go / no-go: does alarm-aware compression preserve a VLA's failure monitor better than action-aware compression?

**Model:** SmolVLA (`HuggingFaceVLA/smolvla_libero`) · **Benchmark:** LIBERO-Spatial · **GPU:** Kaggle 2× T4 · **Code:** `vla-assure`

Same experiment and decision table as the Colab notebook; only the runtime plumbing differs.

**Question.** If we quantize SmolVLA's action expert to 4 bits: (1) does the inherited Mahalanobis monitor's false-alarm rate move, and (2) with ~15 % of the expert kept in full precision, does choosing those blocks by their effect on the **alarm statistic** keep the monitor closer to fp16 than choosing them by **action** change or at random?

### One-time setup (5 min)
1. Kaggle account with **phone verification** (needed for GPU and internet).
2. **Datasets → New dataset**, upload `vla-assure.zip` (the latest one I sent), name it `vla-assure`.
3. New notebook → **File → Import notebook** → this file. Right panel: **Accelerator = GPU T4 x2**, **Internet = On**, **Add input** → your `vla-assure` dataset.

### Running it
Run it as a background job: **Save Version → Save & Run All (Commit)**. It runs up to 12 h without the browser open, and everything in `/kaggle/working` is saved as the version's output. The notebook stops itself cleanly at `TIME_BUDGET_H` (11 h) if it is not done.

**To continue after a stopped run:** *Add input → Notebook output →* this notebook's latest version, then Save & Run All again. Setup reinstalls (~10 min), finished episodes are skipped, and it carries on. Expect **2 runs** in total (~25–35 GPU-h of the 30 h weekly quota with both T4s counted; check the measured time in step 1g).

| Step | What | Episodes | Rough 2×T4 time* |
|---|---|---|---|
| 1 | Setup, install, smoke test | 2 | 15 min |
| 2 | Stage 0-lite: fp16 nominal sanity | 100 | 0.5–1 h |
| 3 | Pick the camera condition (60–85 % success) | 100 | 0.5–1 h |
| 4 | Reference fp16 run, 50 slots/task | 500 | 2–4 h |
| 5 | Block sensitivity | ≤ 96 captured | 1–2 h |
| 6 | Four compressed variants on the 175 test slots | 700 | 3–5 h |
| 7 | Evaluation and decision table | — | 5 min |

\*Re-estimated from the measured seconds/episode in step 1g. When it finishes, download `vla-gonogo/gonogo_summary.zip` from the version's **Output** tab and send it to me.
""")

C[2] = code(r"""
# 1a. GPUs, time guard, work folder; resume from a previous version's output if one is attached as input.
import os, sys, time, json, glob, shutil, pathlib, subprocess
T_START = time.time()
TIME_BUDGET_H = 11.0          # stop cleanly before Kaggle's 12 h limit so the output is saved
def hours_left():
    return TIME_BUDGET_H - (time.time() - T_START) / 3600

print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
assert NGPU >= 1, "No GPU: right panel > Accelerator > GPU T4 x2"
print("GPUs:", NGPU, "| kernel Python", sys.version.split()[0])

WORK = pathlib.Path("/kaggle/working/vla-gonogo"); WORK.mkdir(parents=True, exist_ok=True)
# Kaggle mounts inputs at different depths (e.g. /kaggle/input/notebooks/<user>/<slug>/vla-gonogo)
prev = sorted({p for p in glob.glob("/kaggle/input/**/vla-gonogo", recursive=True) if pathlib.Path(p).is_dir()})
for p in prev:
    if pathlib.Path(p).resolve() != WORK.resolve():
        print("Resuming from previous output:", p)
        shutil.copytree(p, WORK, dirs_exist_ok=True, symlinks=False)
n_done = len(list(WORK.rglob("*.npz")))
print("Work folder:", WORK, "|", n_done, "episodes already on disk")
PERSISTENT = True   # /kaggle/working is saved as the version's output
""")

C[3] = code(r"""
# 1b. Code: the LATEST vla-assure.zip, uploaded as a Kaggle dataset and attached as input.
import zipfile
CODE = pathlib.Path("/root/vla-assure")

def code_ok(root):
    f = root / "src/assure/adapters/lerobot_adapter.py"
    return f.exists() and "_chunk_method" in f.read_text() and "_require_lerobot_06" in f.read_text()

if CODE.exists() and not code_ok(CODE):
    shutil.rmtree(CODE)
if not CODE.exists():
    zips = sorted(glob.glob("/kaggle/input/**/vla-assure*.zip", recursive=True))
    # Kaggle sometimes unpacks an uploaded zip into a folder instead
    trees = sorted({str(pathlib.Path(p).parents[3]) for p in
                    glob.glob("/kaggle/input/**/src/assure/adapters/lerobot_adapter.py", recursive=True)})
    if zips:
        print("Using", zips[0])
        tmp = pathlib.Path("/root/_unzip"); shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zips[0]) as f:
            f.extractall(tmp)
        root = next((p.parents[3] for p in tmp.rglob("src/assure/adapters/lerobot_adapter.py")), None)
        assert root is not None, f"{zips[0]} does not contain the vla-assure code"
        shutil.move(str(root), CODE)
    elif trees:
        print("Using unpacked dataset", trees[0])
        shutil.copytree(trees[0], CODE, symlinks=False)
    else:
        raise FileNotFoundError("vla-assure not found under /kaggle/input: right panel > Add input > your vla-assure dataset")
assert code_ok(CODE), "This zip is outdated: upload the vla-assure.zip from the latest message as a new dataset version"
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

C[4] = code(r"""
# 1c. Python 3.12 venv for LeRobot 0.6.1 (needs Python >= 3.12; Kaggle's kernel env is left untouched).
#     egl_probe / hf-egl-probe (via hf-libero -> robomimic) are C++ packages that fail to build with CMake 4;
#     LeRobot's LIBERO env never imports them, so tiny pure-Python stand-ins are installed first.
VENV = pathlib.Path("/root/py312"); PY = str(VENV / "bin/python")
def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       env=dict(os.environ, CMAKE_POLICY_VERSION_MINIMUM="3.5"))
    if r.returncode != 0 or not check:
        print(r.stdout[-1500:], r.stderr[-3000:])
    if check:
        assert r.returncode == 0, f"failed: {cmd}"
    return r
if not pathlib.Path(PY).exists():
    sh(f"{sys.executable} -m pip install -q uv")
    sh(f"{sys.executable} -m uv venv --seed --python 3.12 {VENV}")
stub = pathlib.Path("/root/_stubs")
(stub / "egl_probe/egl_probe").mkdir(parents=True, exist_ok=True); (stub / "hf_egl_probe").mkdir(parents=True, exist_ok=True)
(stub / "egl_probe/pyproject.toml").write_text('[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n'
    '[project]\nname = "egl_probe"\nversion = "1.0.2"\n[tool.setuptools]\npackages = ["egl_probe"]\n')
(stub / "egl_probe/egl_probe/__init__.py").write_text("def get_available_devices():\n    return [0]\n")
(stub / "hf_egl_probe/pyproject.toml").write_text('[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n'
    '[project]\nname = "hf-egl-probe"\nversion = "1.0.2"\n[tool.setuptools]\npy-modules = []\n')
sh(f"{PY} -m pip install -q {stub / 'egl_probe'} {stub / 'hf_egl_probe'}")
t = time.time()
sh(f'{PY} -m pip install -q "lerobot[smolvla,libero]==0.6.1" pandas scipy tabulate pyyaml')
print(f"installed in {time.time() - t:.0f} s:",
      sh(f"{PY} -c \"import importlib.metadata as m, torch; print('lerobot', m.version('lerobot'), '| torch', "
         f"torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count())\"").stdout)
""")

C[5] = code(r"""
# 1d. Headless rendering. Try NVIDIA EGL first; if MuJoCo cannot render, install Mesa EGL (software) and retry;
#     last resort OSMesa. The chosen backend is used by every later step.
probe = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/>"
         "<geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); "
         "d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); print('render ok', r.render().shape)")
def try_gl(gl):
    r = subprocess.run([PY, "-c", probe], capture_output=True, text=True,
                       env=dict(os.environ, MUJOCO_GL=gl, PYOPENGL_PLATFORM=gl))
    return r.returncode == 0, (r.stdout + r.stderr)[-400:]
GL = None
for attempt in ("egl", "egl+mesa", "osmesa"):
    if attempt == "egl+mesa":
        sh("apt-get -qq update && apt-get -qq install -y libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri >/dev/null", check=False)
    if attempt == "osmesa":
        sh("apt-get -qq install -y libosmesa6 >/dev/null", check=False)
    ok, log = try_gl(attempt.split("+")[0])
    print(attempt, "->", "OK" if ok else log)
    if ok:
        GL = attempt.split("+")[0]; break
assert GL, "no rendering backend works"
if attempt == "egl+mesa":
    print("NOTE: software EGL (Mesa). Rendering is CPU-bound; expect slower episodes than on Colab.")
os.environ["MUJOCO_GL"] = GL; os.environ["PYOPENGL_PLATFORM"] = GL
os.environ["MPLBACKEND"] = "Agg"   # the kernel's inline backend is not valid inside the venv's matplotlib
# LIBERO's first import asks interactive questions and downloads assets: answer once, non-interactively.
r = subprocess.run([PY, "-c", "import libero.libero as L; print('assets:', L.get_libero_path('assets'))"],
                   input="n\n", capture_output=True, text=True, env=dict(os.environ))
print(r.stdout[-600:], r.stderr[-600:])
""")

C[6] = md("Configuration and helpers (same settings as the Colab notebook).")

# 1f: run() uses the venv Python; collect() shards 01_collect over the GPUs and stops before the time limit.
sub(C[7], "# 1f. Experiment configuration", "# 1e. Experiment configuration")
sub(C[7], "N_ACTION_STEPS = None   # None = the checkpoint's own value (as lerobot-eval); set 10 if step 1g shows it is too slow",
    "N_ACTION_STEPS = 10     # 10 actions per model call (~5x faster than the checkpoint's 1); same for every variant")
sub(C[7], 'env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", TOKENIZERS_PARALLELISM="false")',
    'env = dict(os.environ, MUJOCO_GL=GL, PYOPENGL_PLATFORM=GL, TOKENIZERS_PARALLELISM="false")')
sub(C[7], "p = subprocess.run([sys.executable] + cmd,", "p = subprocess.run([PY] + cmd,")
C[7]["source"] += r"""


class OutOfTime(Exception):
    pass

def _clean_partial(since):
    # remove episode files that a killed shard may have left half-written
    import numpy as np
    for f in (CODE / "rollouts").rglob("*.npz"):
        if f.stat().st_mtime >= since - 120:
            try:
                np.load(f)["success"]
            except Exception:
                print("removing partial file", f); f.unlink()

def collect(args, tail=1500):
    # 01_collect.py sharded by task over the GPUs, then one merge pass (skips finished episodes) that writes the
    # complete run.json. Stops cleanly when TIME_BUDGET_H is reached; rerun (next version) to resume.
    if hours_left() < 0.25:
        raise OutOfTime("time budget reached: Save Version now; attach this output and run again to resume")
    args = list(args) + ([] if "--skip_existing" in args else ["--skip_existing"])
    env = dict(os.environ, MUJOCO_GL=GL, PYOPENGL_PLATFORM=GL, TOKENIZERS_PARALLELISM="false")
    if NGPU > 1 and "--tasks" not in args:
        logs = [open(f"/root/shard{g}.log", "w") for g in range(NGPU)]
        t0 = time.time()
        procs = [subprocess.Popen([PY, "scripts/01_collect.py"] + args +
                                  ["--tasks", ",".join(str(t) for t in range(10) if t % NGPU == g)],
                                  # all shards on GPU 0 (the setup that works with robosuite's EGL checks); SmolVLA is small and the
                                  # simulator is CPU-bound, so two processes on one T4 still give ~2x
                                  env=dict(env, CUDA_VISIBLE_DEVICES="0"), stdout=logs[g], stderr=subprocess.STDOUT,
                                  stdin=subprocess.PIPE, text=True) for g in range(NGPU)]
        for p in procs:
            p.stdin.write("n\n"); p.stdin.close()
        # poll: stop all shards at once if one fails or the time budget runs out
        bad = []
        while any(p.poll() is None for p in procs):
            bad = [g for g, p in enumerate(procs) if p.poll() not in (None, 0)]
            late = hours_left() * 3600 < 600
            if bad or late:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                for p in procs:
                    p.wait()
                _clean_partial(t0)
                if late and not bad:
                    raise OutOfTime("time budget reached during collection: Save Version; attach this output and run again")
                break
            time.sleep(10)
        for f in logs:
            f.close()
        for g, p in sorted(enumerate(procs), key=lambda x: x[0] not in bad):   # the shard that failed first
            if p.returncode != 0:
                print(open(f"/root/shard{g}.log").read()[-3000:])
                raise RuntimeError(f"shard {g} failed")
    run(["scripts/01_collect.py"] + args, tail=tail)   # merge pass (or the single-GPU run)
"""

# 1g smoke test, then collection cells use collect()
sub(C[8], "# 1g. Smoke test", "# 1f. Smoke test")
sub(C[8], 'print("If \'reference\' + \'variants\' exceeds ~20 h, set N_ACTION_STEPS = 10 in 1f, rerun 1f and 1g.")',
    'print(f"With {NGPU} GPUs in parallel divide by {NGPU}. If reference + variants exceeds ~16 h, set "\n'
    '      "N_ACTION_STEPS = 10 in 1e and rerun.")')
sub(C[10], 'run(["scripts/01_collect.py", ', "collect([")
sub(C[12], 'run(["scripts/01_collect.py", ', "collect([")
sub(C[14], 'run(["scripts/01_collect.py", ', "collect([")
sub(C[18], 'run(["scripts/01_collect.py", ', "collect([")
for i in (10, 12, 14, 18):
    sub(C[i], ', "--skip_existing"', "", count=None)
sub(C[13], "**Resumable**: rerun this cell after a disconnect.", "**Resumable** across Kaggle runs.")
# sensitivity uses one GPU: guard the time
sub(C[16], 'run(["scripts/02_sensitivity.py"',
    'if hours_left() < 2:\n    raise OutOfTime("less than 2 h left: Save Version; attach this output and run again")\n'
    'run(["scripts/02_sensitivity.py"')
# sensitivity: skip if a previous run already produced it
sub(C[16], 'if hours_left() < 2:', 'SENS = CODE / f"results/sensitivity/{MODEL}.json"\nif not SENS.exists() and hours_left() < 2:')
sub(C[16], 'run(["scripts/02_sensitivity.py"', 'if not SENS.exists():\n  run(["scripts/02_sensitivity.py"')
# sensitivity display: both rankings + their rank correlation; skip non-dict entries of ranking_uncertainty
src = C[16]["source"]
i = src.index("sens = json.loads(")
C[16]["source"] = src[:i] + """sens = json.loads(SENS.read_text())
import pandas as pd
blk = pd.DataFrame(sens["blocks"])
blk["block"] = blk["name"].str.extract(r"layers\\.(\\d+)$").astype(int)
blk["type"] = (blk["block"] % 2).map({0: "self-attn", 1: "cross-attn"})
for col in ("margin_drift", "action_drift"):
    print(f"\\nTop 8 by {col}:")
    print(blk.sort_values(col, ascending=False)[["block", "type", "margin_drift", "action_drift"]].head(8).to_string(index=False))
print("\\nrank correlation (Spearman) alarm-aware vs action-aware:", round(blk["margin_drift"].rank().corr(blk["action_drift"].rank()), 2))
print("ranking uncertainty (mean bootstrap Jaccard):",
      {k: round(v["mean_jaccard"], 2) for k, v in (sens.get("ranking_uncertainty") or {}).items() if isinstance(v, dict)})
"""
# 5b: observation-controlled shadow check, before the variants
C.insert(17, md("""
## 5b. Shadow check: does 4-bit alone move the monitor on identical inputs?
The fp16 robot runs 3 held-out test slots per task while a 4-bit shadow copy sees exactly the same camera frames. Any change in the monitor between the two is caused by quantization alone, not by a different trajectory. About 40 min; read the REALING line before the long variant runs.
""".replace("REALING", "READING")))
C.insert(18, code(r"""
collect(["--model", MODEL, "--variant", "fp16", "--shadow", "w4", "--suite", SUITE, "--condition", CONDITION,
         "--episodes", str(R_REF), "--ref_roles", "test", "--max_per_task", "3", "--out", str(WORK / "rollouts_shadow")])
run(["scripts/shadow_check.py", "--model", MODEL, "--key", KEY, "--shadow_root", str(WORK / "rollouts_shadow"),
     "--out", str(WORK / "shadow_check.json")], tail=2500)
"""))
# final bundle: no Colab download
sub(C[25], 'if not PERSISTENT:\n    from google.colab import files\n    files.download(str(WORK / "gonogo_summary.zip"))',
    'print("Download it from this version\'s Output tab: vla-gonogo/gonogo_summary.zip")')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                               "language_info": {"name": "python"},
                               "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("/home/claude/VLA_gonogo_kaggle.ipynb", "w"), indent=1)
print("kaggle cells:", len(C))
