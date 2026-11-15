"""Builds VLA_resplit_cpu.ipynb: CPU-only robustness check (20 calibration re-splits, exact CIs, paired transitions) re-evaluation of the go/no-go rollouts with two monitor feature sets
(both layers vs action-expert only), including detection of the NEW failures 4-bit introduced. No robot runs."""
import pathlib
NB = pathlib.Path(__file__).resolve().parents[1]   # vla-assure/notebooks: builders read and write notebooks here
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})  # noqa: E731
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],  # noqa: E731
                                "source": s.strip("\n")})

md("""
# All SmolVLA results in one pass (CPU only): primary table, re-splits, release options incl. recalibration on all successes
CPU only (Accelerator = **None**, Internet **On**), about 1-2 h. Inputs: the latest `vla-assure` dataset, dataset `gonogo-v5`, and the outputs of the four stage-2 versions (Add Input -> Notebook Output); not the go/no-go
notebook (Version 8). No robot is run: this re-scores the saved episodes with 3 monitors x 3 placements.
""")
code(r"""
# 0. Python packages the re-scoring needs (Kaggle's image normally has them; installs only what is missing)
import importlib, subprocess, sys
for mod, pkg in (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("yaml", "pyyaml"), ("torch", "torch")):
    try:
        importlib.import_module(mod)
    except ImportError:
        extra = ["--index-url", "https://download.pytorch.org/whl/cpu"] if pkg == "torch" else []
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg] + extra, check=True)
import numpy, scipy, torch
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| torch", torch.__version__)
""")
code(r"""
# 1. Locate code and saved episodes
import glob, os, pathlib, shutil, subprocess, sys
CODE = pathlib.Path("/root/vla-assure")
def newest(root):
    f, r = pathlib.Path(root) / "scripts/shadow_check.py", pathlib.Path(root) / "scripts/resplit.py"
    return f.exists() and "all_taps" in f.read_text() and r.exists() and "recal_all" in r.read_text()
trees = sorted({str(pathlib.Path(p).parents[3]) for p in
                glob.glob("/kaggle/input/**/src/assure/adapters/lerobot_adapter.py", recursive=True)})
trees = [t for t in trees if newest(t)]
assert trees, "attach the vla-assure dataset uploaded from the latest vla-assure.zip (needs recal_all in resplit.py)"
print("code:", trees[0])
shutil.rmtree(CODE, ignore_errors=True); shutil.copytree(trees[0], CODE, symlinks=False)
# merge every attached output: gonogo-v5 (fp16, w4) + the four stage-2 versions (w4_lm, w8, cfg_denoise5, cfg_steps5, w4 repair)
ROLL = pathlib.Path("/kaggle/working/roll")
for d in glob.glob("/kaggle/input/**/rollouts/smolvla/*/libero_spatial", recursive=True):
    d = pathlib.Path(d)
    if "rollouts_" in str(d):          # skip shadow / smoke folders
        continue
    tgt = ROLL / "smolvla" / d.parent.name / "libero_spatial"; tgt.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.npz"):
        if not (tgt / f.name).exists():
            (tgt / f.name).symlink_to(f)
    if (d / "run.json").exists() and (d.parent.name != "w4" or not (tgt / "run.json").exists()):
        shutil.copy(d / "run.json", tgt / "run.json")
assert (ROLL / "smolvla/fp16/libero_spatial").exists(), "attach gonogo-v5 and the four stage-2 notebook outputs"
for v in sorted(p.name for p in (ROLL / "smolvla").iterdir()):
    print(f"  {v:14s} {len(list((ROLL / 'smolvla' / v / 'libero_spatial').glob('*.npz'))):4d} episodes")
need = {"fp16": 400, "w4": 140, "w4_lm": 140, "w8": 140, "cfg_denoise5": 140, "cfg_steps5": 140}
miss = [v for v, n in need.items() if len(list((ROLL / "smolvla" / v / "libero_spatial").glob("*.npz"))) < n]
assert not miss, f"episodes missing for {miss}: attach gonogo-v5 and all four stage-2 notebook outputs"
RES = pathlib.Path("/kaggle/working/results"); RES.mkdir(exist_ok=True)
if (CODE / "results").exists() or (CODE / "results").is_symlink():
    (CODE / "results").unlink() if (CODE / "results").is_symlink() else shutil.rmtree(CODE / "results")
(CODE / "results").symlink_to(RES)
os.chdir(CODE)
""")
code(r"""
# 2. 20 random calibration re-splits x 3 placements x 3 monitors (test slots fixed). ~30-60 min on CPU
for name, taps in (("expert_only", "expert_last"), ("both_layers", "vlm_last,expert_last"), ("vlm_only", "vlm_last")):
    r = subprocess.run([sys.executable, "scripts/resplit.py", "--model", "smolvla", "--key", "libero_spatial",
                        "--rollouts", str(ROLL), "--taps", taps, "--monitors", "mahalanobis,rnd,safe_mlp",
                        "--variants", "w4,w4_lm,w8,cfg_denoise5,cfg_steps5", "--splits", "20", "--out", f"/kaggle/working/resplit_{name}.json"], capture_output=True, text=True)
    print("#" * 20, name); print((r.stdout + r.stderr)[-20000:])
    assert r.returncode == 0, name
import zipfile, glob
with zipfile.ZipFile("/kaggle/working/smolvla_all.zip", "w") as z:
    for p in glob.glob("/kaggle/working/resplit_*.json"):
        z.write(p, os.path.basename(p))
print("Send me /kaggle/working/smolvla_all.zip (Output tab)")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(NB / "VLA_resplit_cpu.ipynb", "w"), indent=1)
print("cells", len(cells))
