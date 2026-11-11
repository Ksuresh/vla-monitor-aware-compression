# Calibrated Until Changed

Research repository for **Calibrated Until Changed: Keeping Robot Failure Monitors Reliable After Model Compression**
(target: IEEE RA-L). Question: when a vision-language-action (VLA) policy is quantized, fine-tuned or run with
different inference settings, does its learned failure monitor keep its stated false-alarm rate, and how can a
deployer tell and fix it before release?

| Folder | What |
|---|---|
| `vla-assure/` | Experiment code: LeRobot/OpenVLA adapters, feature taps, fake quantization, monitors, functional conformal calibration, shadow replay, evaluation, release-option comparison (`scripts/resplit.py`), 38 CPU tests |
| `vla-assure/notebooks/` | The Kaggle notebooks that produced every result (and `builders/`, the scripts that generate them) |
| `experiments/` | All results: per-trial CSVs, re-split/release JSONs, run metadata, and `summary_smolvla.csv` (main table). See `experiments/README.md` |
| `paper/` | Current manuscript draft (Word) |

## Status (November 2026)

- **SmolVLA, LIBERO-Spatial: complete.** Five changes (W4 expert, W4 language model + expert, W8 expert, 5 denoising
  steps, 5 actions per call). A monitor breaks when, and only when, the change touches what it reads; shadow replay
  predicted the closed-loop outcome in all four changes it was run on; recalibration restores false-alarm control,
  label-free realignment does not.
- **pi0.5, LIBERO-10 + strong camera shift: reference run done** (400 episodes, success 0.945); variants pending.

## Tests

```
cd vla-assure && PYTHONPATH=src python -m pytest -q tests
```

They check the analysis pipeline on synthetic rollouts; they do not run a VLA or LIBERO.

**Caution:** fake INT4 quantization does not establish real GPU memory or latency savings.
