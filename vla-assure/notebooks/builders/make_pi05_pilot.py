"""Builds VLA_pi05_pilot.ipynb: a 1-2 h feasibility pilot for the second model (pi0.5, LeRobot) on Kaggle 2xT4.
Answers: does it load and run on a T4 (float32 vs bfloat16), seconds per episode, success, do the feature taps fire,
does 4-/8-bit quantization of its action expert apply, does the second checkpoint (update) load, can a second
process use GPU 1, and does a shadow copy fit in memory. No experiment data is produced."""
import json

base = json.load(open("/home/claude/VLA_gonogo_kaggle.ipynb"))["cells"]
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}  # noqa: E731
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}  # noqa: E731

C = [md(r"""
# Second model pilot: pi0.5 on LIBERO-Spatial (about 1-2 h, GPU T4 x2)
Checks that pi0.5 runs on Kaggle before we spend quota on the full experiment. **Run interactively** (Shift+Enter
cell by cell), not as a commit, so we can stop early if something fails.

**Setup:** Settings -> Accelerator **GPU T4 x2**, Internet **On**. Add-ons -> Secrets -> `HF_TOKEN` (a Hugging Face
read token whose account has accepted google/paligemma-3b-pt-224), switched **on** for this notebook.
Add Input: dataset **vla-assure** (latest version). **Send back:** `pi05_pilot.zip` (Output / file browser) or
a screenshot of the last cell.
""")]
C.append(md("## 1. Setup"))
C.append(code(base[2]["source"].replace("TIME_BUDGET_H = 11.0", "TIME_BUDGET_H = 11.0")))
C.append(code(r'''
# 1b. Code: the newest vla-assure copy among the inputs (needs the dtype switch added for pi0.5)
import zipfile
CODE = pathlib.Path("/root/vla-assure")
def newest(root):
    root = pathlib.Path(root)
    a = root / "src/assure/adapters/lerobot_adapter.py"
    return a.exists() and "pcfg.dtype" in a.read_text() and (root / "scripts/resplit.py").exists()
if CODE.exists() and not newest(CODE):
    shutil.rmtree(CODE)
if not CODE.exists():
    trees = sorted({str(pathlib.Path(p).parents[3]) for p in
                    glob.glob("/kaggle/input/**/src/assure/adapters/lerobot_adapter.py", recursive=True)})
    trees = [t for t in trees if newest(t)]
    if not trees:
        raise FileNotFoundError("No vla-assure copy with the pi0.5 dtype switch: upload the newest vla-assure.zip "
                                "as a new dataset version and attach it")
    shutil.copytree(trees[0], CODE, symlinks=False)
for d in ("rollouts", "results"):
    (WORK / d).mkdir(exist_ok=True)
    tgt = CODE / d
    if tgt.is_symlink(): tgt.unlink()
    elif tgt.exists(): shutil.rmtree(tgt)
    tgt.symlink_to(WORK / d)
os.chdir(CODE); print("Code OK at", os.getcwd())
'''))
C.append(code(r'''
# 1b'. Hugging Face token (pi0.5 uses Google's gated PaliGemma tokenizer)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print("HF_TOKEN loaded (length", len(os.environ["HF_TOKEN"]), ")")
except Exception as e:
    raise RuntimeError("Add-ons -> Secrets: add HF_TOKEN and switch it on for this notebook") from e
'''))
C.append(code(base[4]["source"].replace('"lerobot[smolvla,libero]==0.6.1"', '"lerobot[pi,smolvla,libero]==0.6.1"')))
C.append(code(r'''
# 1c'. Can this token read the gated tokenizer? (fails fast with a clear message instead of deep inside LeRobot)
r = subprocess.run([PY, "-c", "from transformers import AutoTokenizer; "
                    "AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print('tokenizer OK')"],
                   capture_output=True, text=True, env=dict(os.environ))
print((r.stdout + r.stderr)[-800:])
assert r.returncode == 0, ("token cannot read google/paligemma-3b-pt-224: open that page on huggingface.co with the "
                           "same account, click 'Agree and access', then rerun this cell")
'''))
C.append(code(base[5]["source"]))
cfg = base[7]["source"]
i, j = cfg.index("MODEL, SUITE = "), cfg.index("def run(cmd")
cfg = cfg[:i] + r'''MODEL, SUITE = "pi05", "libero_spatial"
CONDITION, KEY = "nominal", "libero_spatial"
N_ACTION_STEPS = 10     # same throughput setting as SmolVLA
DTYPE = "float32"       # set by the dtype test in 2a ("float32" or "bfloat16")
X = "model.paligemma_with_expert."
exp["quant_scope"] = {"include": [X + "gemma_expert.*"], "block_regex": r"gemma_expert\.model\.layers\.(\d+)$"}
exp["variants"]["w4_lm"] = {"kind": "quant", "bits": 4, "group_size": 128, "own_scope": True,
                            "include": [X + "paligemma.model.language_model.*", X + "gemma_expert.*"]}
exp["variants"]["w8_lm"] = dict(exp["variants"]["w4_lm"], bits=8)
models["models"][MODEL]["n_action_steps"] = N_ACTION_STEPS
def write_cfg(dtype):
    models["models"][MODEL]["dtype"] = dtype
    (CFG / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    (CFG / "models.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
write_cfg(DTYPE)
print("checkpoint:", models["models"][MODEL]["checkpoint"], "| update:", models["models"][MODEL].get("update_checkpoint"))
print("taps:", models["models"][MODEL]["tap"])

''' + cfg[j:]
C.append(code(cfg))
C.append(md("## 2. Pilot"))
C.append(code(r'''
# 2a. Precision test: one episode of task 0 in float32 and in bfloat16 (T4 has no native bfloat16, so it may be slow)
import numpy as np
P = {}
def trial(tag, extra, env_extra=None, tail=1200):
    out = WORK / f"pilot/{tag}"
    cmd = [PY, "scripts/01_collect.py", "--model", MODEL, "--suite", SUITE, "--condition", "nominal", "--out", str(out)] + extra
    env = dict(os.environ, MUJOCO_GL=GL, PYOPENGL_PLATFORM=GL, TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="0",
               **(env_extra or {}))
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, input="n\n")
    log = "\n".join(l for l in (p.stdout + p.stderr).replace("\r", "\n").splitlines() if "it/s" not in l)
    rj = next(iter(out.rglob("run.json")), None)
    rj = json.loads(rj.read_text()) if rj else {}
    res = dict(ok=p.returncode == 0, wall_s=round(time.time() - t), s_per_ep=round(rj.get("seconds", 0) / max(rj.get("episodes", 1), 1)),
               success=rj.get("success_rate"), episodes=rj.get("episodes"), peak_gib=rj.get("peak_gpu_mem_gib"),
               taps=rj.get("tap_fired"), query_s=rj.get("query_latency_median_s"))
    P[tag] = res
    print(f"[{tag}] {res}")
    if not res["ok"]:
        print(log[-tail:])
    return res

for dt in ("float32", "bfloat16"):
    write_cfg(dt)
    trial(f"dtype_{dt}", ["--variant", "fp16", "--tasks", "0", "--episodes", "1"])
ok = {dt: P[f"dtype_{dt}"] for dt in ("float32", "bfloat16") if P[f"dtype_{dt}"]["ok"]}
assert ok, "pi0.5 did not run in either precision: send me the log above"
DTYPE = min(ok, key=lambda d: ok[d]["s_per_ep"])
write_cfg(DTYPE)
print("\nUsing", DTYPE, "| taps fired:", ok[DTYPE]["taps"])
assert all(v > 0 for v in (ok[DTYPE]["taps"] or {"none": 0}).values()), "feature taps did not fire: send me this output"
'''))
C.append(code(r'''
# 2b. Speed and success: one episode per task, all 10 tasks (~10-30 min)
r = trial("fp16_10tasks", ["--variant", "fp16", "--episodes", "1"])
print(f"\nfp16 success {r['success']} over {r['episodes']} episodes | {r['s_per_ep']} s/episode | peak {r['peak_gib']} GiB")
'''))
C.append(code(r'''
# 2c. Model changes load and run: W4 / W8 on the action expert, W4 on language model + expert, the second checkpoint
for v in ("w4", "w8", "w4_lm", "update"):
    trial(f"{v}", ["--variant", v, "--tasks", "0,1,2", "--episodes", "40", "--roles", "test", "--max_per_task", "1"])
'''))
C.append(code(r'''
# 2d. Can a second process use GPU 1? (doubles throughput if yes; rendering stays on GPU 0)
trial("gpu1", ["--variant", "fp16", "--tasks", "1", "--episodes", "1", "--device", "cuda:1"],
      env_extra={"CUDA_VISIBLE_DEVICES": "0,1"})
'''))
C.append(code(r'''
# 2e. Shadow copy memory: fp16 robot + a W4 shadow on the same GPU (needed for zero-rollout shadow replay)
trial("shadow_w4", ["--variant", "fp16", "--shadow", "w4", "--tasks", "0", "--episodes", "1"])
'''))
C.append(code(r'''
# 2f. Summary and size of the full run
import pandas as pd
df = pd.DataFrame(P).T[["ok", "s_per_ep", "success", "episodes", "peak_gib", "query_s"]]
print(df.to_string())
s = P["fp16_10tasks"]["s_per_ep"] or P[f"dtype_{DTYPE}"]["s_per_ep"]
par = 2 if P.get("gpu1", {}).get("ok") else 1
for name, n in (("reference 400", 400), ("one variant (140 test + 80 repair)", 220)):
    print(f"{name:36s} ~ {n * s / 3600 / par:5.1f} GPU-session hours ({par} process(es) in parallel)")
json.dump({"dtype": DTYPE, "pilot": P}, open(WORK / "pi05_pilot.json", "w"), indent=1, default=str)
with zipfile.ZipFile("/kaggle/working/pi05_pilot.zip", "w") as z:
    z.write(WORK / "pi05_pilot.json", "pi05_pilot.json")
    for f in (WORK / "pilot").rglob("run.json"):
        z.write(f, str(f.relative_to(WORK)))
print("Send me /kaggle/working/pi05_pilot.zip or a screenshot of this table")
'''))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                               "language_info": {"name": "python"},
                               "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("/home/claude/VLA_pi05_pilot.ipynb", "w"), indent=1)
print("cells", len(C))
