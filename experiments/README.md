# Experiment results: Calibrated Until Changed

Results behind the paper *Calibrated Until Changed: Keeping Robot Failure Monitors Reliable After Model Compression*
(draft in `../paper/`). Code in `../vla-assure/`; the Kaggle notebooks that produced every folder below are in
`../vla-assure/notebooks/` (built by the scripts in `notebooks/builders/`).

`summary_smolvla.csv` is the main table: one row per model change x monitor placement x monitor, rebuilt from the
files here with `python experiments/summarize.py`.

## Common setup

| Item | Value |
|---|---|
| Benchmark | LIBERO, Kaggle 2x T4, MuJoCo EGL rendering, LeRobot 0.6.1 |
| Design | Pre-registered slot partition per task (seed fixed before any rollout): 25 % train, 20 % calib, 20 % repair, 35 % test; variants run the same test slots (paired) |
| Alarm | Functional conformal band (SAFE-style), alpha = 0.10, shape/threshold split max(10, ceil(N/2)) |
| Monitors | Mahalanobis, RND, SAFE-style MLP probe, at three placements (VL-only `vlm_last`, expert-only `expert_last`, both); controls: action-chunk consistency (`acc`), pixel Mahalanobis (`ext_mahalanobis`) |
| Primary metric | Inherited monitor's false-alarm rate on all successes of each policy, exact 95 % Clopper-Pearson interval |
| Robustness | 20 random re-partitions of the non-test reference slots (`over_splits`) |
| Release options | inherited, recalibrated (19 / 44 successes; and on all successes), label-free CORAL realignment, flag-then-recalibrate, certify-or-recalibrate; separate pool (repair slots) and cross-fit |

## SmolVLA (`HuggingFaceVLA/smolvla_libero`), LIBERO-Spatial, nominal camera, 10 actions per call

R = 40 slots per task (400 reference episodes, 140 paired test slots); fp16 test success 0.529.

| Folder | Contents |
|---|---|
| `smolvla/gonogo/` | Go/no-go run: fp16 reference, W4 on the action expert, two 15 % mixed-precision protection rules; block sensitivity (`sensitivity.json`), per-trial alarms for both-layer Mahalanobis (`eval_trials.csv`) |
| `smolvla/stage2_part1/` | W4 on language model + expert (`w4_lm`); shadow replay of `w4_lm`, `w8`, `cfg_denoise5` on repair slots (`shadow2_*.json`) |
| `smolvla/stage2_part2/` | W8 on the action expert, test + repair slots |
| `smolvla/stage2_part3/` | 5 denoising steps (`cfg_denoise5`, same weights) |
| `smolvla/stage2_part4/` | 5 actions per call (`cfg_steps5`); W4's own calibration pool on repair slots; W4 re-scored with all monitors |
| `exploratory/` | Earlier SmolVLA checks (module inspection, feature hooks, fake-INT4 pilots, selective-precision screen) |

Each `stage2_partN/` holds `trials_partN_<placement>.csv` (per-trial alarms, all monitors, alpha 0.05/0.10/0.20),
`resplit_partN_<placement>.json` (primary rates, re-splits, AUROC, transitions, release options), `groups_partN.csv`
(quick table) and `run_<variant>.json` (success rate, sampler, timing).

Headline (Mahalanobis, false alarms on all successes, fp16 -> changed policy):

| Change | Success fp16 -> changed | VL-only | Both layers | Expert-only |
|---|---|---|---|---|
| W4, action expert | 0.53 -> 0.51 | 0.03 -> 0.01 | 0.04 -> 0.15 | 0.01 -> 1.00 |
| W4, language model + expert | 0.53 -> 0.21 | 0.03 -> 1.00 | 0.04 -> 1.00 | 0.01 -> 1.00 |
| W8, action expert | 0.53 -> 0.52 | 0.03 -> 0.03 | 0.04 -> 0.07 | 0.01 -> 0.07 |
| 5 denoising steps | 0.53 -> 0.58 | 0.03 -> 0.02 | 0.04 -> 0.30 | 0.01 -> 1.00 |
| 5 actions per call | 0.53 -> 0.60 | 0.03 -> 0.05 | 0.04 -> 0.02 | 0.01 -> 0.06 |

Under 5 actions per call only the output-level monitor (action-chunk consistency) moves: 0.00 -> 0.19 on joint successes.

## pi0.5 (`lerobot/pi05_libero_finetuned`, bfloat16), in progress

| Folder | Contents |
|---|---|
| `pi05/part1/` | Condition check (`pi05_condition.json`: LIBERO-Spatial is too easy, 0.94-1.00; chosen LIBERO-10 + strong camera shift, 0.90 in the check) and the 400-episode reference run (success 0.945) |

Pending: W4 / W8 / W4-LM on the 140 paired test slots (`VLA_pi05_full.ipynb`, PART 2 onward).

## Where the raw episodes are

Episode files (`*.npz`: actions, success, monitor features per query) are too large for git. They live in Kaggle:
dataset `sureshkandasamy/gonogo-v5` (SmolVLA fp16 + W4 + protection variants), the four output versions of the
stage-2 notebook run on a collaborator's Kaggle account (SmolVLA `w4_lm`, `w8`, `cfg_denoise5`, `cfg_steps5`, W4 repair pool), and the output versions of
`notebook49b3154248` (pi0.5). Every CSV/JSON here can be rebuilt from them with `vla-assure/scripts/03_evaluate.py`
and `vla-assure/scripts/resplit.py`.
