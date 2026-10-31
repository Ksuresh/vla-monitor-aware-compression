# vla-assure — Monitor-Aware Compression for VLA failure monitors

Measures whether VLA runtime failure monitors (SAFE-style probes, Mahalanobis/LLMD, action-chunk
consistency, RND/FIPER-style novelty, plus pixel-only controls) keep their finite-sample false-alarm
control when the policy they watch is **quantized** or **updated**, and tests release-time repairs.
Models: π0.5, π0, SmolVLA (LeRobot), OpenVLA, OpenVLA-OFT (official repos), on LIBERO.

> Status: complete code; analysis path tested on synthetic data (`pytest -q tests/`, 35 tests).
> **No VLA has been run yet.** Every reported number must come from your own runs.

## 1. What the band controls, exactly

One conformal band per (model, suite, condition, monitor, α), fitted on successful reference episodes
**pooled over the suite's tasks**. It controls one quantity: the probability that a **successful**
episode raises at least one alarm, for successes of the calibrated policy drawn from this cell's trial
design. It does not bound missed failures, detection delay or physical safety, and no per-task
guarantee is claimed.

Procedure (`src/assure/conformal.py`), with N successful calibration episodes:

1. Seeded random permutation. The first `n_shape = max(10, ⌈N/2⌉)` episodes (capped so at least one
   remains) fix the centre μ_k and spread σ_k per query; the remaining `n = N − n_shape` set the threshold.
2. Each threshold episode gives `M_i = max_k (s_ik − μ_k)/σ_k`; sort ascending.
3. `q = M_(r)`. Alarm iff `M_new > q` (strict).
   - marginal: `r = ⌈(n+1)(1−α)⌉` (must be ≤ n). Then `P(alarm on a new success) ≤ 1 − r/(n+1) ≤ α`.
   - PAC: smallest r with `P(Bin(n, α) ≥ n+1−r) ≥ 1−δ`. For continuous scores the false-alarm rate of
     the band is `Beta(n+1−r, r)`, so `P(FPR > α) ≤ δ`.
4. Ties: comparing with `>` against an observed order statistic can only lower the alarm rate, so
   both bounds stay valid without randomised tie-breaking (tested on discretised scores).
5. Validity: conditional on the shaping set, the band is a fixed function, and threshold episodes and a
   new success are exchangeable because of the trial design below.

| guarantee | α | N | shaping | threshold n | rank r | exact bound | Monte Carlo (`scripts/verify_budgets.py`) |
|---|---|---|---|---|---|---|---|
| marginal | 0.20 | 14 | 10 | 4 | 4 | E[FPR] ≤ 0.200 | 0.203 |
| marginal | 0.10 | **19** | 10 | 9 | 9 (max) | E[FPR] ≤ 0.100 | 0.099 |
| marginal | 0.05 | 38 | 19 | 19 | 19 (max) | E[FPR] ≤ 0.050 | 0.050 |
| PAC δ=.1 | 0.20 | 22 | 11 | 11 | 11 (max) | P(FPR>α) = 0.086 | 0.084 |
| PAC δ=.1 | 0.10 | **44** | 22 | 22 | 22 (max) | P(FPR>α) = 0.9²² = 0.0985 | 0.099 |
| PAC δ=.1 | 0.05 | 90 | 45 | 45 | 45 (max) | P(FPR>α) = 0.099 | 0.099 |

Below N the row is flagged `cp_underpowered` (only 1/(n+1) is guaranteed) and excluded from claims.

## 2. Trial design: pre-registered roles

`common.trial_roles`: before any rollout, one seeded random partition of **all** R×10 reference trial
slots (pooled over tasks) into train / calib / repair / test in the proportions 25% / 20% / 20% / 35% of the POOLED R×10 slots
(R = 50 → 125 / 100 / 100 / 175; not stratified, so per-task counts vary: test 13–24, calib 5–15;
written to run.json as `role_counts_per_task`). Trial (t, i) always has the
same seed and initial state for every policy.

- **Reference:** runs all slots. Monitors are fitted on the **train slots only**; reference runs of repair
  slots are recorded but never used for fitting, so calib / repair / test never touch the monitor.
- **Variants:** run the **test** slots (every one paired, all used for the endpoint) and the **repair**
  slots (their repair pool).

Two exchangeability statements are kept apart:

- **(i) design-based:** over the random partition of this benchmark's fixed slots. This is what the
  experiments verify.
- **(ii) deployment:** additionally assumes new episodes come from the process that generated
  LIBERO's initial states. This is stated, not tested.

There are no top-ups: the repair pool is a fixed, pre-registered set of slots, so recalibration rests on
(i) only. Recalibration with 19 or 44 successes uses a seeded random subset of the pool's realized
successes (never "the first k"), so the subset is exchangeable with the pool.

Slot usage (formal statement):

| Role | Reference | Each variant | Used for |
|---|---|---|---|
| train | runs | — | the only data any monitor is fitted on (halves A/B for the replicate) |
| calib | runs | — | fp16 shaping band + threshold (shared by #A and #B) |
| repair | runs (not used for fitting) | runs | variant recalibration pool (Section 4.2) |
| test | runs | runs | every paired, deployment and confirmatory analysis |
| selection (900000+) | captured | replayed | block ranking only; never calibration or validation |
| extension (20000+, Stage C) | — | 10% configs | K per task per suite; fresh seeds, initial state redrawn uniformly; with the test slots it forms the method-comparison population used by ALL three F2–F4 components; never used by F1 or any calibration |

| R per task | train / calib / repair / test | calib success needed at α=0.1 (marginal / PAC) |
|---|---|---|
| 50 (Stages B–C) | 125 / 100 / 100 / 175 | 19% / 44% |
| 40 (Stage A) | 100 / 80 / 80 / 140 | 24% / 55% |

**Guarantee.** Claimed under (A1): exchangeability of successful threshold and future test scores given
the monitor. 19 / 44 successes are exact under (A1). Finite-population special case needs (C1) uniform
fixed-count partition and (C2) no interference; the proof conditions on the TRAINING SET, not the
monitor. Exact whole-test-subset distribution: `conformal.finite_population_fpr`. Marginal 19: exact
(E[FPR_test] = 0.10). PAC 44: P(FPR_test > 0.1) = 0.101–0.107 for 60–175 test successes → the
In general (ties, strict alarms) E[FPR_test] ≤ 1 − r/(n+1); equality only for tie-free scores. Finite-population PAC is m-dependent (lattice k/m): 0.100–0.147 over m = 40–175, so NO finite-population PAC
budget is claimed; each recalibration row reports `finite_pop_p_viol` for its realised m (supplementary).
Standalone proof: paper S1. Independent check from scratch (no formula): `scripts/verify_finite_population.py`
(200 000 partitions × 3 populations; MC = exact within error). Not an i.i.d. deployment guarantee, and it
does not cover the extension trials.

## 3. Three analyses

| Analysis | Episodes | Supports claims about |
|---|---|---|
| deployment | each policy's own executed test trials | what a deployed system sees (features **and** which episodes succeed both change) |
| paired | the same executed trials under both policies | false alarms on trials both solved, detection of newly induced failures, paired task-success change |
| obs_controlled | reference observations through a non-controlling **shadow** copy of the variant | representation / monitor-score drift only |

The observation-controlled analysis is a monitor-score comparison on fixed observations. It does
**not** tell you what the variant would have done in control: its outcome labels (and hence its
AUROC) are the reference policy's. Task-success and failure-detection conclusions come only from
executed trajectories. `action_div` reports how differently the shadow would have acted (large = weaker
proxy). The pixel-only `ext_*` monitors read no policy weights; in the deployment analysis their
change estimates the population shift itself.

## 4. Statistics (prespecified, `experiment.yaml: stats`)

- **Degradation D, oriented per metric (D > 0 = worse):**
  - `D_FPR = FPR_var − FPR_ref`
  - `D_AUROC = AUROC_ref − AUROC_var`
  - `D_TPR = TPR_ref − TPR_var`
  - `D_succ = succ_ref − succ_var`
- **Two one-sided tests on D:**
  - **degraded:** H0 D ≤ 0 rejected.
  - **non-inferior:** H0 D ≥ ε rejected (a one-sided non-inferiority test, not equivalence).
  - **inconclusive:** otherwise. This is never read as preserved calibration.
  - Margins ε: 0.05 for FPR, AUROC and success; 0.10 for TPR.
- **Intervals:** task-clustered bootstrap (tasks, then episodes). D is paired where trials match.
- **Primary endpoint — monitor calibration degradation on jointly successful, trial-paired
  executions:** paired D_FPR, W4 vs fp16, per (model, internal monitor), pooled over the model's
  Stage B cells and clustered by (suite, task). Holm within each model's family
  (`results/tables/primary_endpoint.csv`). Conditioning on joint success is a selection, so the
  deployment FPR (each policy's own successes) is always reported beside it as the operational measure.
- **Qualification:** a cell qualifies by its *realized* count of calibration successes (≥19 marginal,
  ≥44 PAC at α=0.1), never by a success rate. `run_stage.py --dry_run` prints P(count met) and the
  expected pairs / test failures per success rate. Target window: 60–85% reference success.
- **Power** (`scripts/power_analysis.py`, assumed FPR 0.09, alarm discordance 0.3, success 0.7):

  | test trials | both-success pairs | CI half-width | P(non-inferior \| D=0, ε=0.05) | P(detect D=+0.10) |
  |---|---|---|---|---|
  | 140 (Stage A cell) | 69 | ±0.072 | 0.44 | 0.49 |
  | 175 (one Stage B cell) | 86 | ±0.063 | 0.50 | 0.62 |
  | 700 (four cells pooled) | 343 | ±0.033 | 0.93 (≥0.77 Bonferroni, m=4) | 1.00 |

  Pairs = test trials × joint success π₁₁. The table uses independence (π₁₁ = p²) as a planning
  scenario, not a bound. Shared initial states make positive correlation likely (π₁₁ up to
  min(p_ref, p_var)), but compression can change which situations are hard (ρ < 0 possible), and it
  lowers p_var. Pooled examples: ρ=0.3 gives 387 pairs and NI power 0.98; p_var=0.6 under independence
  gives 294 pairs and 0.89.

  Single cells are therefore estimation only. After Stage A, re-size with
  `python scripts/power_analysis.py --from_pairs results/eval_pairs/<model>__<key>.csv --from_eval results/eval/<model>__<key>.csv` (estimates FPR, discordance, p_ref, p_var, π₁₁ and ρ).
- **Early warning:** `alarm_time_norm` (fraction of a failed episode elapsed) is *not* an actionable
  lead time. If `results/onsets.csv` (model,variant,key,task,idx,onset_step) exists, lead time before
  annotated onset and the fraction alarmed before onset are reported.

## 4b. Protocol safeguards

- **Confirmatory tests** (primary endpoint, rule-vs-rule contrasts only):
  - Linear paired endpoints with ≥ 20 task clusters.
  - Wild cluster restricted bootstrap-t, Webb weights, CR1 SE, null imposed (`metrics.confirmatory`).
  - Operating levels: degradation **0.0375**; non-inferiority **0.03**; detection NI in F2–F4 **0.02**.
    Simulation-calibrated operating choices for the stated grid, not proof of uniform size control.
  - Chosen by WORST-CASE size, not the average. `scripts/calibrate_level.py`: one exact generator (true
    D at the boundary, verified 0.0498), 20 clusters, ~340 pairs, 27 settings × 2000 studies:

    | | NI @0.05 | NI @0.03 | DEG @0.05 | DEG @0.04 |
    |---|---|---|---|---|
    | mean over grid | 0.055 | 0.034 | 0.049 | 0.040 |
    | max over grid | 0.071 ± 0.011 | 0.046 ± 0.009 | 0.059 | 0.049 ± 0.009 |
    | settings > 0.05 | 15 (6 significant) | 0 | 10 (0 significant) | 0 |
  - Refinement (`--refine 3 --reps_refine 10000`): the 3 worst settings per level re-run with 10 000
    studies, judged by the one-sided 95% Clopper–Pearson UPPER bound. NI @0.0325: worst 0.044 (upper
    0.047) → 0.03 kept. DEG @0.04: worst 0.046 (upper 0.050) → fails; @0.0375: upper 0.049 → adopted.
  - Prespecified response: re-run grid + refinement with Stage A estimates; any refined upper bound
    above 0.05 lowers that level by 0.0025 (repeat), before Stage B.
  - Everything else (per-cell, deployment ratios, AUROC, TPR) is descriptive: intervals, no p-values.
- **Power** (`power_analysis.py`, same test and generator, 2000 studies, MC ±0.01): pooled 343 pairs /
  20 clusters gives NI power 0.96 (Bonferroni/4: 0.86) and detection of +0.10 at 1.00. Size there is
  0.037 ± 0.008 (NI, nominal 0.03); DEG is run at 0.0375. A single cell (10 clusters) has DEG size
  0.082, so it is descriptive only.
- **Resampling:** tasks are resampled once and carry the reference, every variant and all conditions of
  that task together (`metrics.compare_joint`). Reference and variant are never bootstrapped
  independently.
- **Multiplicity map** (`experiment.yaml: families`; everything outside these is descriptive):
  - F1 — W4 vs fp16 primary endpoint, Holm over monitors within each model.
  - F2 — proposed vs the pre-specified baselines {action_drift, weight_error, random} at 10%, monitor
    #A, fake-quant: intersection-union of FPR superiority (joint successes, 0.0375), TPR
    non-inferiority (joint failures, ε 0.10, level 0.02) and success non-inferiority (ε 0.05, 0.03).
    Composite p: `p_IUT = max(p_FPR, (0.0375/0.03)·p_succ, (0.0375/0.02)·p_TPR)`, then Holm over the 3
    baselines at 0.0375 (operational convention: component levels are simulation-calibrated only at
    their own level). `tables/method_families.csv`.
  - **Detection feasibility** (`scripts/detection_feasibility.py`): with 350 paired trials and 20%
    failure each, independence gives ~14 joint failures. Simulated N_min for power ≥ 0.80 at level 0.02
    (size upper bound ≤ 0.047 for n_jf ≥ 90): 90 / 120 / 300 joint failures at alarm discordance
    0.05 / 0.10 / 0.20; > 0.20 infeasible. Before deployment,
    `--plan --joint_counts n11 n10 n01 n00 --disc_counts x n` fixes K extension trials per task
    (`01_collect.py --extension K`) by JOINT posterior simulation (Dirichlet joint-outcome table, Beta
    discordance, N_min table, binomial n_jf): smallest K with predictive P(n_jf ≥ N_min) ≥ 0.90, cap 100;
    the predicted probability is reported. Shadow-pass discordance is a PLANNING PROXY: draws are inflated
    ×1.5 (pre-registered) and a sensitivity table gives P at ×1–3; executed discordance on joint failures
    is reported (`executed_discordance`) and decides nothing. N_min table: 0.05/0.075 → 90, 0.10 → 120,
    0.125 → 160, 0.15 → 200, 0.20 → 300. Examples: 140-pair pilot (95,10,12,23), 3/40 → infeasible
    (0.85 at cap); 280-pair pilot, same proportions → K = 83.
    Realised n_jf < N_min or < 12 tasks with a joint failure → "detection not confirmable (claim
    withheld)".
  - Headline method claim: better than ALL three baselines (per-baseline results reported).
  - F3 — same as F2 with the independent monitor #B (transfer).
  - F4 — same as F2 with real int4 kernels (tao4p) against ONE baseline: the first of
    `families.f4_baseline_order` = [action_drift, weight_error, random] whose build passes the parity
    screen, which runs BEFORE F4 deployment on captured reference episodes (07_parity.py writes
    `f4_baseline`). Proposed fails parity → F4 not confirmable. The method claim requires F2 and F4 to agree.
  - **Joint power of the complete rule** (`scripts/f2_power.py`; 3 components + composite p + Holm +
    detection gate, 1000 studies/point): P(claim vs all) at Δ_FPR = 0.03 is 0 / 0.35 / 0.81 / 0.92 at
    K = 0 / 45 / 83 / 100; FWER under 3 null configurations 0.014–0.040 (±0.012). Sizing: K = max(K_det,
    K_joint), K_joint from `f2_power.py --size` (power 0.80 vs all baselines at Δ* = 0.03).
  - Other budgets, greedy, other monitors: `method_contrasts_descriptive.csv`.
- **Repair pool:** fixed 100 slots (R=50; P(≥44 successes) 0.90 at 50% success, ≥0.999 from 60%), no top-ups. Realized pool successes are logged; a cell that
  misses the 44-success PAC budget reports it as unmet, never extends the pool.
- **Supervised probes:** need ≥20 training failures (`min_train_failures`). Stage A runs
  `--learning_curve`.
- **Sampler consistency:** denoising steps, query schedule and seeding are recorded in run.json
  (`sampler`). A mismatch with fp16 sets `sampler_consistent=False`.
- **Block selection:**
  - The fp16 monitor and band are frozen (no recalibration during selection).
  - One scale: sd(M) over all captured reference episodes, floored at 0.05.
  - Selection trials (ids 900000+) are outside the partition and count as model-selection data.
  - Greedy always fills the budget and logs non-improving steps.
  - Budgets are filled at sub-block granularity (whole blocks, then single Linear layers in rank order):
    every method protects within 0.25 points of the target (`fill_budget`, `BUDGET_TOLERANCE`).
    `protected_frac_actual` is reported, and `budget_ok` is flagged in pareto.csv.
  - Adaptive capture: episodes alternate selection (even) / held-out (odd) until each set has ≥24
    episodes and ≥8 failures and ≥8 successes (`--min_failures`, `--min_successes`), cap 192.
    Stopping depends only on outcome counts.
  - Ranking uncertainty: 500-replicate bootstrap within outcome group; mean Jaccard, its 5th
    percentile and per-block selection probability. **Stage C entry criterion:** mean Jaccard ≥ 0.5 at
    10%; else capture 24 more, up to the cap; if still unmet, the method claim is withheld.
  - Strategy comparison (independent vs restricted greedy vs random) on the *held-out* set, with
    bootstrap intervals. Held out from selection, NOT independent of the test distribution (same 50
    LIBERO initial states); executed test trajectories are the evidence. Greedy uses 8 + 2 candidates per step.
  - Every rule is a hybrid (ranked whole blocks + size-based filler); pareto.csv reports the
    block vs filler share of protected parameters.
  - Cost outputs (`sensitivity/*cost*`): seconds per candidate block, per greedy J evaluation and per
    greedy step, plus capture time.
  - Ranking stability (12/24/48) is EXPLORATORY. It runs after selection is frozen, and its 48-episode
    ranking includes the held-out set.
  - **Independent monitor test:** the selection monitor `mahalanobis#A` is trained on half A of the
    training slots; `mahalanobis#B`, on the disjoint half B, is frozen (family, features, calibration,
    protocol) before Stage C, refused by 02_sensitivity.py, shares the calib slots with #A, and is
    evaluated only on deployed test trials (family F3). Other monitor families are descriptive.
- **Real-kernel parity** (`scripts/07_parity.py`, criteria in `experiment.yaml: parity`):
  - identical protected set
  - action ratio ≤ 0.25
  - |ΔJ| ≤ 0.10
  - score Spearman ≥ 0.95
  - false-alarm disagreement (successes) ≤ 0.10, reported separately
  - missed-detection disagreement (failures) ≤ 0.10, reported separately
  - advantage direction preserved under real kernels
  - The proposed rule and the best baseline are then executed with real kernels on both suites:
    family F4, part of the main claim.

  Efficiency counts only for configurations that pass.
- **Commitment:** Stage B/C budgets are fixed only after Stage A's measured throughput. Pre-stated drop
  order: π0 reduced → OpenVLA reduced → actuator-noise cell → OFT Stage B → SmolVLA Spatial. The π0.5
  primary endpoint and the π0.5 two-suite frontier are never dropped.
- **Stage 0 compatibility** (`scripts/stage0_check.py`): functional checks plus an uncertainty-aware
  comparison with the published rate. INCOMPATIBLE (stop and investigate) only if the one-sided 95%
  UPPER bound < published − 5 points; otherwise COMPATIBLE, with estimate and interval reported. 500
  episodes (the Stage B nominal reference, reused): a correct implementation is essentially never
  stopped; a 10-point shortfall is flagged 84–96% of the time.
- **Guarantee vs preservation:** monitor-aware compression can only *empirically* preserve
  calibration (less need to recalibrate). A formal guarantee for a compressed policy still needs
  recalibration on it.
- **Failure scope:** camera shifts mostly cause perception failures. The `action_noise` condition
  (seeded noise on executed arm actions) is a secondary cell, kept out of the pooled primary endpoint.

## 4c. Execution budget

Authoritative budget (stage figures elsewhere are its totals). K = extension trials per task per
suite; 6 configurations (4 fake-quant + 2 real-kernel) × 2 suites × 10 tasks → +120K episodes →
120K × 30 s / 3600 = 1.0·K GPU-hours. `python scripts/budget.py [--K k] [--measured]` → `results/budget/budget.csv` (model × configuration × suite ×
condition: trials, shadows, GPU-hours, storage). Planning assumptions in `experiment.yaml: budget`,
replaced by run.json timings after Stage A. At K = 40: 34 720 executions, ~336 GPU-hours, ~123 GB
(A 13, B 135, C 188 GPU-hours); each extension trial per task ≈ 1 GPU-hour.

## 5. Monitor applicability

| Monitor | π0.5 / π0 / SmolVLA | OpenVLA-OFT | OpenVLA |
|---|---|---|---|
| SAFE-MLP / SAFE-LSTM, Mahalanobis, RND | yes | yes | yes |
| Action-chunk consistency | yes (replans every 10 of 50) | **n/a** (executes whole 8-step chunk) | **n/a** (one step) |
| FIPER-style (needs action samples) | yes (`--alt_samples 4`) | **n/a** (deterministic L1 head) | **n/a** (greedy tokens) |
| ext_* (pixels only) | yes | yes | yes |

## 6. Mixed precision at matched memory

`w4p-<rule>-f<pct>`: 4-bit everywhere except blocks chosen by `<rule>`, protected up to `<pct>`% of
Linear parameters. Rules:

- Selection runs on reference rollouts with reference labels: an observation-controlled *surrogate*
  (keep the reference monitor's decisions). Evaluation is always on the variant's own executed trials.
  Ranking stability is checked with `--stability 12,24,48` (Spearman of J_b, Jaccard of selected sets).
- `margin_greedy`: the RESTRICTED greedy set from `02_sensitivity.py --greedy`. J of the *whole* quantized model
  is recomputed after each protected block (16 top candidates + 4 random per step, 6 episodes).
  The output also compares independent vs greedy vs random J and predicted vs observed drops: this is
  the block-interaction check.
- `margin_drift` (alarm-statistic-aware, the proposed method): `J_b = ½ E_succ|ΔM|/sd + ½ E_fail|ΔM|/sd`.
  The absolute value means upward and downward shifts can't cancel, and failures count equally. Also
  logs `suppression`: the signed ΔM on failures, negative = alarms pushed down.
- `score_drift` (monitor-score-aware): change in the monitor score.
- `feature_drift`: 1 − cos on the monitored features.
- `action_drift` (task-aware): change in the predicted action chunk.
- `weight_error` (data-free sensitivity).
- `edge`: first and last blocks.
- `random`.

`margin_drift`/`score_drift` need a monitor and band trained on reference rollouts
(`02_sensitivity.py --ref_key`, 12 captured episodes under the shift so both outcomes occur). Stage C
runs all rule × budget points as shadows, then deploys the frontier.

A rule wins only if it lowers D_FPR with D_TPR / D_AUROC non-inferior. Lower false alarms with lost
detection counts as suppression, not a win. Closest prior methods are Mix-QVLA (task evidence) and
ActQuant (action-guided); neither looks at the runtime monitor.

## 7. Efficiency (real kernels only)

`tao4p-<rule>-f<pct>` builds the same protected set as `w4p-<rule>-f<pct>` with real torchao int4
kernels (protected blocks in bf16), so the proposed mixed-precision configurations are measured, not
only uniform INT8/INT4.

Parity check before reporting efficiency:

```bash
python scripts/01_collect.py --variant tao4p-<rule>-f10 --pilot ...
python scripts/03_evaluate.py ...   # drift_* and action drift should match w4p-<rule>-f10
```


`w8/w4/w4p-*` are fake (quantize–dequantize) and give no speed-up; they isolate the numerical effect.
End-to-end numbers come from `scripts/06_efficiency.py` with real kernels: torchao int8/int4 (`tao8`,
`tao4`) for LeRobot policies, bitsandbytes INT8/NF4 (`bnb8`, `bnb4`) for OpenVLA. It reports weight
memory, peak memory, query latency (median/p95), episode wall-clock, control rate, with taps and
shadows detached.

## 8. Controlled update (LoRA)

`variants.lora_update` + `scripts/05_lora_update.sh`:

- **Data:** the same checkpoint, fine-tuned on the evaluated suite's LIBERO demonstrations.
- **Objective:** the policy's native loss.
- **Adapter:** LoRA r 16, α 32, dropout 0.05 on the attention and MLP projections; the vision
  encoder is frozen.
- **Schedule:** 6000 steps, batch 8, lr 1e-4, seed 7.

The script writes `lora_spec.json` next to the checkpoint.

## Layout

```
configs/models.yaml        models, checkpoints, feature taps, architecture flags
configs/experiment.yaml    trial design, calibration, stats plan, variants, monitors, repairs, stages A/B/C
src/assure/                common (trial design), conformal, metrics (clustered bootstrap, Holm), monitors, ...
scripts/00_smoke_test.py   model loads, episode runs, every tap fires
scripts/01_collect.py      rollouts on pre-registered trials (+ --shadow, --target_successes)
scripts/02_sensitivity.py  per-block feature / action / score / margin drift
scripts/03_evaluate.py     deployment / paired / obs_controlled + repairs -> CSV
scripts/04_report.py       tables, Holm verdicts, figures -> results/report.md
scripts/05_lora_update.sh  controlled LoRA update
scripts/06_efficiency.py   end-to-end efficiency with real kernels
scripts/verify_budgets.py  exact + Monte-Carlo check of the calibration sample sizes
scripts/power_analysis.py  power / precision of the primary endpoint (simulation of the real analysis)
scripts/run_stage.py       staged plan A / B / C (use --dry_run)
```

## Setup

```bash
bash envs/setup_lerobot.sh          # pi0 / pi0.5 / SmolVLA   (PaliGemma gated access for pi0/pi0.5)
bash envs/setup_openvla.sh          # OpenVLA + OpenVLA-OFT (two envs, pinned torch)
pip install torchao                 # real int8/int4 kernels for 06_efficiency.py
export MUJOCO_GL=egl
pytest -q tests/ && python scripts/verify_budgets.py
```

## Staged plan with go/no-go gates

| Stage | What | Cost (rough) | Go to next stage when |
|---|---|---|---|
| 0 | `00_smoke_test.py`; fp16 nominal reproduction at 50 trials/task (reused as Stage B reference); 50-episode pilot per candidate condition | ~3–6 GPU-hours/model | taps fire; `stage0_check.py` PASS; a condition with 60–85% success found |
| A | π0.5, LIBERO-10, mild shift; R=40 with W8/W4/update shadows; W4 and update on 140 test + 60 repair slots; plus a replication of Bysani (NeurIPS 2026): SmolVLA, W4, nominal LIBERO-Spatial | ~10–12 GPU-hours | reference success 60–85% (pairs for the endpoint AND failures for detection); realized calib successes ≥ 19; ≥5 training failures; paired trials verified to start from identical states |
| B | π0.5, SmolVLA, OpenVLA-OFT; 2 suites; nominal + mild; repair pools topped up | ~40–60 GPU-hours | Stage A decided the framing (below) |
| C | Block sensitivity + greedy interaction check, Pareto with all eight rules on π0.5; π0 and OpenVLA reduced; real-kernel efficiency | ~45–70 GPU-hours | — |

All costs are planning estimates; `results/tables/throughput.csv` (episodes per GPU-hour, measured) replaces them after Stage A.

**Stage A estimates and sizes; it never classifies.**

- It reports D_FPR estimates with intervals, alarm discordance, realized success/failure counts and
  throughput.
- Then: `python scripts/power_analysis.py --from_pairs results/eval_pairs/<model>__<key>.csv` to fix R
  for Stage B.
- Verdicts (margin 0.05, levels 0.0375 / 0.03) come only from the Stage B pooled primary endpoint. A pooled
  "non-inferior" is withheld if any single cell is clearly degraded (`primary_by_cell.csv`).
- The replication cell must reproduce the direction of Bysani's result first.

```bash
python scripts/run_stage.py A --dry_run
python scripts/run_stage.py A
python scripts/run_stage.py B --models pi05 --alt 4
REPO=~/vla-repos/openvla-oft python scripts/run_stage.py B --models openvla_oft
REPO=~/vla-repos/openvla python scripts/run_stage.py C
```

## Verify on first run

- checkpoint ids marked `verify`
- tap paths for π0 and SmolVLA (`--list_modules norm`)
- `lerobot-train` PEFT flags and suite filtering
- that `set_attr("init_state_id")` succeeds (a warning is logged if pairing cannot be enforced)
- the torchao API version

Monitors are compact re-implementations; cross-check with the official SAFE (github.com/vla-safe/SAFE)
and FIPER (github.com/utiasdsl/fiper) code on Stage A rollouts.
