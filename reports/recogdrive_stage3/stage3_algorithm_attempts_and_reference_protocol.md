# ReCogDrive Stage3 Algorithm Attempts And Reference Protocol

Date: 2026-06-14 to 2026-06-15 UTC

This is the required ledger for ReCogDrive Stage3 policy-improvement work. Before any new algorithm change or large run, read this file, update the planned attempt, and check the cited paper/open-source implementations for the exact mechanism being borrowed.

The goal is not to prove a simplified toy version. The goal is to implement mature enough versions of each method that a negative result actually means something.

## Mandatory Protocol

Every new Stage3 algorithm change must satisfy this checklist before launch:

1. Past evidence:
   - Identify which previous run, checkpoint, or buffer statistic motivates the change.
   - State the targeted failure mode: weak policy absorption, poor exploration, DDC/TTC regression, invalid candidate leakage, low valid-candidate coverage, learning-rate confound, or inefficient reward evaluation.

2. Reference audit:
   - Search current papers and official/open-source implementations for the method family.
   - Record the sources checked and the exact mechanism copied.
   - Do not use a hand-rolled lightweight approximation as evidence that the method works or fails unless it is explicitly labeled as a smoke test.

3. Mature implementation checklist:
   - Data path: train-only rewards/cache for training; no navtest reward, label, metric cache, or navtest-derived artifact.
   - Policy objective: include the core pieces required by the reference method, such as old policy logprobs, ratio clipping, advantage normalization, multi-epoch minibatch replay, reference model terms, or shared-noise preference comparisons.
   - Diffusion details: use correct denoising timestep treatment, shared noise/timestep where needed, and no target leakage through conditioning tokens.
   - Safety semantics: NC/DAC/TTC/DDC guards must be explicit and logged.
   - Diagnostics: log the quantities that prove the objective is active, not just final PDMS.
   - Controls: separate algorithm effect from LR, effective batch size, optimizer steps, checkpoint epoch, and evaluation seed/sampling differences.

4. Launch requirement:
   - Add a planned-attempt entry to this document before starting the run.
   - Include config, expected diagnostic movement, failure criteria, and references.

5. Result requirement:
   - After each evaluated checkpoint, append the result row.
   - Include PDMS plus NC, DAC, TTC, EP, comfort, DDC, and TLC when available.

## Fair Comparison Rule

Multi-GPU and few-GPU Stage3 runs must not be compared by checkpoint label alone.

- If `devices * per_gpu_batch_size * accumulate_grad_batches` is matched, `global_step` is a reasonable early comparison axis because one optimizer step consumes nearly the same number of train scenes.
- If effective batch differs, compare by estimated train-scene exposure:

```text
effective_batch_size = devices * batch_size * accumulate_grad_batches
seen_train_scenes ~= global_step * effective_batch_size
epoch_fraction ~= seen_train_scenes / 85109
```

- `step-step_300` is an early health check, not a full algorithm verdict. For the current 8GPU `b2acc4` Stage3 setting it is about `19,200` train-scene exposures; for a 2GPU `b2acc4` run it would be only about `4,800` exposures.
- Epoch-bearing checkpoints such as `epoch_0-step_1330`, or step-only checkpoints with about one epoch of exposure, can be compared to epoch0 controls.
- For GRPO-family methods that are not catastrophically unsafe, the first real
  algorithm verdict should use the relaxed window `epoch3-4` or roughly
  `step5000`/equivalent train-scene exposure. This is because Safe DiffGRPO was
  weak at epoch0 but recovered by `epoch_3-step_5320`.
- Full success/failure against the user-reported original Stage3 `0.9055` reference requires comparable long-horizon exposure: that result was obtained at `epoch9-step13300`, not at an early checkpoint. Safe DiffGRPO `0.906184` was obtained at `epoch_12-step_17290`. Treat both as long-horizon targets, not early-step gates.
- Patched on 2026-06-15: `scripts/training/summarize_recogdrive_stage3_runs.py` now reports `devices`, `batch_size`, `accumulate_grad_batches`, `effective_batch_size`, `latest_seen_scenes`, and best-checkpoint exposure fields. `scripts/training/decide_recogdrive_stage3_next_action.py` now treats step-only checkpoints below about `80,000` seen train scenes as early health checks rather than comparing them directly with the long-run `0.906` baseline.

## Retrospective Fairness Audit On Prior Decisions

Updated on 2026-06-15 after adding effective-batch and train-scene exposure
tracking, and after user clarification that the historical `90.55` PDMS was
obtained at `epoch9-step13300`.

Some earlier text used the long-horizon `0.9055` / Safe DiffGRPO `0.906184`
baselines too aggressively when discussing `step300/600/900` checkpoints. The
correct interpretation is:

- `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`
  should not be treated as a fully failed algorithm direction. It is an early
  positive signal with unstable safety/progress tradeoff. With the same
  effective batch size `64`, it beat the current-repo GRPO control at matched
  early steps `300` and `900`:
  - step300: cap05 `0.885432` vs control `0.881910`, delta `+0.003522`
  - step600: cap05 `0.885497` vs control `0.886777`, delta `-0.001280`
  - step900: cap05 `0.887107` vs control `0.883839`, delta `+0.003268`
  The reason not to continue the exact cap05 recipe as default is the paired
  submetric diagnosis: it can improve EP/high-score bins while also creating
  NC/TTC/DDC regressions. Future methods should preserve the useful absorption
  idea but add better safety-aware preference/advantage handling.
- `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`
  remains a valid negative result under the new comparison rule. At matched
  early steps it was below the current-repo GRPO control:
  - step300: v3 `0.879395` vs control `0.881910`, delta `-0.002515`
  - step600: v3 `0.882934` vs control `0.886777`, delta `-0.003843`
  The failure mode remains over-hard TTC/DDC gating that protects safety by
  suppressing EP/DAC too much.
- `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z`
  remains a valid negative result. Even allowing for the multi/few-GPU caveat,
  its step300/600 PDMS values were far below the matched current-repo GRPO early
  control (`0.744740` and `0.801195` vs `0.881910` and `0.886777`), and its
  epoch0-step800 PDMS `0.831015` stayed below the original early Stage3 gate.
- AWAC/IQL pure offline-regression variants remain negative as implemented. Their
  evaluated epoch checkpoints were far below the current GRPO controls
  (`0.78-0.86` range, best corrected warmup epoch0 about `0.862915`), so the
  conclusion is not mainly a step-exposure artifact. The useful artifact from
  that direction is the strict train-only elite buffer, not the pure AWAC/IQL
  training objective.

Decision consequence: do not discard all self-imitation/preference absorption
ideas. The corrected lesson is narrower: broad auxiliary imitation has early
signal, but it must be made safety-aware and compared at matched exposure before
being promoted or rejected. Hard safety gates that zero too much reward are
likely harmful unless they preserve progress pressure.

## Relaxed Horizon Audit: Epoch4 / Step5000

Updated on 2026-06-15 after the user correctly noted that `step300` and even
epoch0 can be too early for Stage3 RL conclusions. The reproducible audit script
is:

`scripts/evaluation/audit_recogdrive_stage3_relaxed_horizon.py`

Current full-scan outputs are:

- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5000_eval_audit_fullscan.tsv`
- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5000_eval_audit_fullscan_best_by_run.tsv`
- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5000_eval_audit_run_summary.tsv`
- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_fullscan.tsv`
- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_fullscan_best_by_run.tsv`
- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_run_summary.tsv`

The `step5000` and `step5500` scans currently have the same run-level top
ordering because the filter is an OR condition with `epoch<=4`. The slightly
wider `5500` scan is the default table because it keeps step-only checkpoints
near 5k while also retaining epoch-bearing evidence such as Safe DiffGRPO
`epoch_3-step_5320` and `epoch_4-step_6650`.

As of the 2026-06-15 21:20 UTC refresh, the audit TSV schema also includes
`devices`, `batch_size`, `accumulate_grad_batches`, `effective_batch_size`, and
`seen_train_scenes`. Use these columns before comparing step-only checkpoints:
for example, current-repo clean GRPO `epoch_0-step_1330` has effective batch
`64` and about `85120` seen scenes, while zt2 clean GRPO `step300` has only
about `19200` seen scenes.

The older hand-built audit file is kept only for traceability:

- `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_with_safe.tsv`

Important results under this wider horizon:

| Run | Checkpoint | PDMS | EP | TTC | DDC | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| zt3 original-LR GRPO control | `epoch_0-step_1290` | `0.897167` | `0.826724` | `0.963050` | `0.973801` | Strongest retained current-code early/mid control; full-scan mean over two independent exact eval records. |
| Safe DiffGRPO historical stream | `epoch_3-step_5320` | `0.895788` | `0.852496` | `0.940353` | `0.957777` | Clear slow improvement: epoch0 was only `0.878473`, so judging this family at epoch0 would be wrong. |
| Safe DiffGRPO historical stream | `epoch_4-step_6650` | `0.894501` | `0.838791` | `0.949662` | `0.953782` | Still strong at epoch4, but below its epoch3 point and below zt3 GRPO epoch0. |
| current-repo original-LR GRPO control | `epoch_0-step_1330` | `0.893686` | `0.827006` | `0.958148` | `0.965398` | Best local control among retained current-repo runs. |
| auxiliary-decay soft-safety buffer-distill | `step-step_900` | `0.892335` | `0.832688` | `0.950816` | `0.974172` | Best matched step900 variant so far; strong EP/DDC recovery, with NC/TTC still weaker than clean-GRPO control. |
| non-decay soft-safety buffer-distill | `step-step_600` | `0.891373` | `0.815760` | `0.960043` | `0.970135` | Strong early signal but later unstable. |
| non-decay soft-safety buffer-distill | `step-step_2100` | `0.890360` | `0.822739` | `0.953534` | `0.969394` | Recovered after epoch0, so stopping at epoch0 alone was too harsh; however step2400 later dropped to `0.882233`. |
| zt2 clean GRPO b2acc8 e4-gate | `step-step_300` | `0.887157` | `0.826053` | `0.952628` | `0.972360` | Healthy clean-GRPO launch result and above the current-repo clean step300, but still too early for a method verdict; continue to the relaxed horizon. |
| cap05 self-imitation | `step-step_900` | `0.887107` | `0.823058` | `0.955841` | `0.971742` | Not enough horizon to reject under the new rule; old evidence only proves early oscillation. |
| Buffer-DPO ref-control | `step-step_300` | `0.859864` | `0.809982` | `0.938375` | `0.967087` | Still a valid negative; too far below control even at the first checkpoint. |

Revised decision rule:

- Do not use `step300` as an algorithm rejection gate for GRPO-family methods.
  It is only a launch-health check.
- For GRPO / GSPO / self-imitation / buffer-guided variants, require at least
  `epoch3-4` or about `step5000` before a final negative decision, unless the
  run is catastrophically low or has clear safety collapse.
- Earlier decisions that are now softened:
  - cap05 self-imitation should be considered inconclusive beyond step900, not
    fully disproven.
  - non-decay soft-safety buffer-distill should be considered promising but
    unstable, not simply failed at epoch0; it needs a longer controlled rerun or
    a decay/regularization variant evaluated to epoch3-4.
  - auxiliary-decay soft-safety buffer-distill should now be considered a live
    promising GRPO-family route after its step900 recovery, not a weak step600
    result. Its remaining risk is NC/TTC erosion versus clean GRPO.
- Decisions that remain negative:
  - pure AWAC/IQL regression and current lightweight Buffer-DPO remain poor
    because their evaluated PDMS was far below GRPO controls, not merely because
    of early-step comparison.
  - over-hard main TTC/DDC gates remain risky because they suppress EP/DAC.
- The audit script also writes a run-level summary with each run's first, best,
  and latest relaxed-window checkpoint plus best/latest seen-scene exposure. Use
  that summary for promotion/rejection discussions so late recovery, late
  instability, and unfair low-exposure checkpoints are all visible.

Re-audit on 2026-06-15 21:12 UTC with the same script,
`--max-epoch 4 --max-step 5500`, now finds `94` raw eval rows, `42` grouped
checkpoint rows, and `33` rows in the relaxed window after adding the zt2 clean
GRPO step300 eval and auxiliary-decay epoch0 eval. The revised classification
is:

- Keep as strongest evidence:
  - Current-code original-LR GRPO remains the best local reproducible control:
    `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z`
    reached `0.893686` at `epoch_0-step_1330`.
  - The stopped zt3 original-LR GRPO control remains the strongest early/mid
    result in the scan: `0.897167` at `epoch_0-step_1290`. It is kept only as
    evidence because zt3 is no longer used.
  - Safe DiffGRPO is the clearest slow-recovery example: `0.878473` at
    `epoch_0-step_1330`, then `0.895788` at `epoch_3-step_5320`. Any GRPO-family
    method with non-catastrophic early results should therefore be judged at
    about `epoch3-4` or `step5000`, not only at step300.
  - The new zt2 clean GRPO b2acc8 e4-gate run reached `0.887157` at
    `step300`, with NC `0.984017`, DAC `0.967046`, TTC `0.952628`, EP
    `0.826053`, and DDC `0.972360`. This is a healthy launch check and above
    the current-repo clean step300 `0.881910`, but it is still below the best
    relaxed-window controls and is not yet a verdict.
- Reclassify as inconclusive/promising-but-unstable, not fully disproven:
  - auxiliary-decay soft-safety buffer-distill: local8 step900 reached
    `0.892335`, beating the matched current-repo clean GRPO step900 by
    `+0.008496`, the non-decay buffer-distill step900 by `+0.004130`, and cap05
    step900 by `+0.005227`. Its later epoch0-step1330 eval is `0.888593`, so
    the teacher signal is useful but not retained strongly enough yet. Continue
    analysis rather than promote or reject based on step300 alone.
  - cap05 self-imitation: best observed relaxed-window result is `0.887107` at
    step900. It shows early absorption signal, but no epoch3-4 evidence.
  - non-decay soft-safety buffer-distill: `0.890785` step300, `0.891373`
    step600, `0.890360` step2100, but `0.884028` at epoch0 and `0.882233` at
    step2400. The signal is real but unstable and below the matched epoch0 GRPO
    control.
- Keep as negative:
  - Buffer-DPO ref-control: `0.859864` at step300, too far below the matched
    GRPO control to explain by horizon alone.
  - Pure AWAC/IQL regression variants: evaluated epoch checkpoints remained in
    the `0.78-0.86` band, so the issue is objective/absorption mismatch, not
    only early stopping.
  - Over-hard TTC/DDC gating: still suppresses EP/DAC enough to underperform
    matched controls.

Operational consequence: do not retire a GRPO-family idea before `epoch3-4` /
`step5000` unless it is catastrophically low or safety collapses. At the same
time, do not spend full resources on offline-regression or lightweight DPO paths
without a mature implementation rewrite. The next resource allocation should
favor clean GRPO-primary/Safe-DiffGRPO-like runs as the comparison spine, with
buffer knowledge used only after it is made tightly safety-aware and shown not
to damage EP/DAC by the relaxed `epoch3-4` / `step5000` window. Earlier
`step300/600/900/epoch0` points are diagnostics for whether to watch safety
more closely, not final promotion/rejection gates.

Repeat scans on 2026-06-15 19:38, 19:54, 20:14, 20:50, 21:12, 21:28, 21:44,
22:06, 22:20, and 22:42 UTC after explicitly relaxing the review condition to
`epoch<=4 OR step<=5000/5500`. The OR condition is intentional:
`epoch_4-step_6650` should still count as epoch4 evidence even though its raw
step is above 5500, while step-only checkpoints are retained up to the
roughly-step5000 window.

- The 22:42 UTC refresh again finds `94` raw eval rows, `42` grouped checkpoint
  rows, and `33` rows inside the relaxed window. The `step5000` and `step5500`
  scans still have the same run-level best ordering. There were no newer
  complete exact PDMS rows after the 21:12 UTC zt2 clean-GRPO and
  auxiliary-decay eval files; zt2 clean-GRPO `step600` is still under exact
  evaluation. The leading methods are unchanged.
- The top run-level ordering under this relaxed window remains:
  - zt3 original-LR GRPO evidence-only control: `0.897167` at
    `epoch_0-step_1290`.
  - Safe DiffGRPO historical stream: `0.895788` at `epoch_3-step_5320`.
  - current-repo original-LR clean GRPO: `0.893686` at `epoch_0-step_1330`.
  - auxiliary-decay soft-safety buffer-distill: `0.892335` at `step900`.
  - non-decay soft-safety buffer-distill: `0.891373` at `step600`, with later
    instability.
  - zt2 clean GRPO b2acc8 e4-gate: `0.887157` at `step300`, launch-healthy but
    not yet comparable to epoch3-4 or step5000 evidence.
- This relaxed criterion changes the interpretation of earlier decisions:
  - Safe DiffGRPO would have been wrongly rejected at epoch0; it moved from
    `0.878473` at `epoch_0-step_1330` to `0.895788` at
    `epoch_3-step_5320`.
  - Current-repo clean GRPO should be treated as the local comparison spine:
    its relaxed-window best is `0.893686` at `epoch_0-step_1330`, but the later
    step-only points show instability rather than a complete algorithm verdict.
  - Auxiliary-decay soft-safety buffer-distill is not a step600 failure, but its
    new `step1200` result regressed to `0.886896` after the auxiliary teacher
    had nearly faded. Its best available relaxed-window point remains
    `0.892335` at `step900`, so the idea is now classified as early-useful but
    unstable rather than a resource-leading route.
  - Cap05 self-imitation and non-decay buffer-distill were over-penalized if
    judged only by final early rows. They contain useful absorption signals, but
    the observed safety/progress oscillation means they need redesigned
    safety-aware objectives before another full run.
- This relaxed criterion does not rescue all failed paths:
  - Buffer-DPO ref-control at `0.859864` step300 is too far below the matched
    GRPO control to justify continuing that exact lightweight implementation.
  - Pure AWAC/IQL regression remains negative because evaluated checkpoints were
    in the `0.78-0.86` band, not because only one early checkpoint was checked.
  - Main hard TTC/DDC gates remain negative because the failure mode is clear
    EP/DAC suppression, not insufficient horizon alone.
- Promotion rule from now on:
  - `step300`: launch-health only.
  - `step600/900/epoch0`: matched-exposure diagnostics and safety triage.
  - `epoch3-4` or about `step5000`: first real GRPO-family algorithm verdict.
  - `epoch9-step13300` or equivalent: fair comparison against the historical
    original Stage3 `0.9055`; `epoch_12-step_17290` or equivalent for the
    historical Safe DiffGRPO `0.906184`.

Direct answer to the relaxed-audit question:

- Yes, some earlier GRPO-family variants were over-penalized by too-short
  horizons. Cap05 self-imitation and both soft-safety buffer-distill variants
  should be treated as containing useful early absorption signals, not as
  fully disproven ideas.
- No, this does not rescue the currently implemented AWAC/IQL regression,
  lightweight Buffer-DPO, or over-hard TTC/DDC gate paths. Their failures are
  large enough, or mechanistically clear enough, that horizon alone does not
  explain them.
- The resource spine should therefore remain clean/original-LR GRPO and
  Safe-DiffGRPO-like long-horizon GRPO, with buffer knowledge added only as a
  small safety-aware shaping/proposal signal and judged at the relaxed
  `epoch3-4` / `step5000` window.
- Practical resource rule after this re-audit: do not stop the currently active
  clean GRPO or buffer-bonus GRPO just because `step300` is below the best
  epoch0/epoch3 controls. Their first real decision point is the relaxed
  horizon. Conversely, do not restart AWAC/IQL pure regression or the current
  lightweight Buffer-DPO implementation just to give them more steps; they need
  an implementation-level redesign before another expensive run.

## Buffer Absorption Diagnosis

Updated on 2026-06-15 after comparing AWAC/IQL, Buffer-DPO, cap05
self-imitation, soft-safety buffer-distill, and the current auxiliary-decay
GRPO design.

The strict train-only elite buffer is a useful data asset, but high-PDMS
buffer candidates should not be treated as new GT labels for long-horizon
Stage3 training. The reason is an objective mismatch:

- The buffer records an oracle-selected trajectory for a scene, while the DiT
  policy must learn a conditional denoising distribution that remains supported
  by its own rollout samples.
- PDMS improvements often come from thresholded or discontinuous safety terms
  such as NC, DAC, TTC, and DDC, but plain denoising MSE only teaches waypoint
  proximity and does not expose the causal safety credit assignment.
- A high-scoring buffer trajectory may be outside the current policy's likely
  action manifold. Strong regression can then cause distribution drift; strong
  trust-region weighting can prevent any meaningful movement.
- Positive-only best-trajectory supervision is weaker than mature preference
  learning. A robust preference path needs paired positives/negatives, shared
  noise and timestep treatment for diffusion targets, reference-policy or KL
  control, reward-gap weighting, and hard negatives that preserve safety
  semantics.
- Static offline targets can become stale as the policy changes. Buffer data is
  therefore better used as an early teacher, proposal source, reference
  constraint, or hard-negative/preference source, while the main improvement
  signal should remain on-policy or near-on-policy GRPO/DPPO/RIPT-style
  rollout learning.

Current design implication: do not spend more full runs on pure AWAC/IQL-style
offline diffusion regression or the current lightweight Buffer-DPO objective.
The active auxiliary-decay run keeps GRPO as the main objective, uses the
train-only buffer only as a small early teacher, and decays buffer/self-imitation
from step600 to epoch0-step1330 so the policy can recover progress/DAC learning.
If this still fails by the relaxed `epoch3-4` / `step5000` window, the next
buffer-absorption design must be a mature diffusion preference or on-policy
proposal method rather than another direct regression variant.

### 2026-06-15 Candidate: GRPO buffer-bonus-only absorption

Motivation:
- The train-only strict-v2 elite buffer is high quality: cycle1 keep-best
  summaries show mean best-valid reward around `0.9754`, has-valid ratio `1.0`,
  and best-valid-above-GT around `59%`. The data asset is not the main failure.
- The failed AWAC/IQL and lightweight Buffer-DPO paths suggest static
  denoising-target supervision is not reliably absorbed by the DiT planner.
- DDPO and DPPO both support treating diffusion fine-tuning as policy-gradient
  optimization over sampled denoising trajectories rather than only
  reward-weighted likelihood regression:
  - DDPO paper/code: https://arxiv.org/abs/2305.13301 and
    https://github.com/jannerm/ddpo
  - DPPO paper/code: https://arxiv.org/abs/2409.00588 and
    https://github.com/irom-princeton/dppo
- Diffusion-DPO is a valid preference-learning reference
  (https://arxiv.org/abs/2311.12908), but our current lightweight Buffer-DPO
  result was very weak at step300. The next buffer test should therefore first
  isolate a safer mechanism before adding another preference loss.

Mechanism:
- Keep clean GRPO/reference-KL as the main optimizer.
- Load the train-only elite buffer only to identify valid high-margin targets.
- Add a small reward-neighborhood bonus when sampled policy trajectories are
  close to those high-margin train-only buffer targets.
- Disable buffer distillation, Buffer-DPO, and self-imitation. This prevents a
  static offline regression loss from overriding the sampled PDMS reward.
- Use strict buffer validity and margin filtering. Invalid or low-margin buffer
  records should not shape the reward.

Implementation:
- Added launcher:
  `scripts/training/launch_recogdrive_stage3_grpo_buffer_bonus_only_2b_local.sh`
- Defaults:
  - LR `1e-4`, `MAX_EPOCHS=20`, `sample_time=16`, `batch=2`,
    `accumulate_grad_batches=4`, effective batch `64`.
  - BC anneal `0.10 -> 0.05`, `reference_kl_coeff=0.02`.
  - `GRPO_USE_GSPO_RATIO=false` to match the clean-GRPO comparison spine unless
    a matched GSPO control is explicitly selected.
  - `GRPO_BUFFER_GUIDANCE_ENABLED=true`.
  - `GRPO_BUFFER_REWARD_BONUS_WEIGHT=0.01`,
    `GRPO_BUFFER_REWARD_BONUS_SCALE_M=3.0`,
    `GRPO_BUFFER_REWARD_BONUS_USE_MARGIN=true`.
  - `GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0`.
  - `GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT=0.0`.
  - `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.0`.
  - Buffer target eligibility uses
    `GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN=0.02` and top-1 valid buffer target.
  - Watcher defaults evaluate every step checkpoint starting at the launch
    health check `step300`, plus epoch checkpoints:
    `EVAL_MIN_CHECKPOINT_STEP=300`,
    `EVAL_CHECKPOINT_STEP_INTERVAL=300`,
    `EVAL_ALWAYS_EPOCH_CHECKPOINTS=1`.

Verification:
- `bash -n scripts/training/launch_recogdrive_stage3_grpo_buffer_bonus_only_2b_local.sh` passed.
- Dry-run wrote `strict_gspo_launch_config.txt` with:
  - `grpo_use_gspo_ratio=false`
  - `grpo_buffer_guidance_enabled=true`
  - `grpo_buffer_reward_bonus_weight=0.01`
  - `grpo_buffer_distill_loss_weight=0.0`
  - `grpo_buffer_preference_dpo_loss_weight=0.0`
  - `grpo_self_imitation_loss_weight=0.0`
  - `eval_min_checkpoint_step=300`
  - `eval_checkpoint_step_interval=300`

Launch decision:
- Do not launch while local DPPO-step, zt2 clean GRPO, and zt2 keep-best
  buffer exploration are all productively using resources.
- If DPPO `step2400` is weak and clean GRPO remains the stronger spine, this
  buffer-bonus-only run is the next buffer-absorption candidate because it tests
  the specific hypothesis that high-score buffer trajectories help as a small
  proposal prior while PDM-scored GRPO remains the main learning signal.

## Current Experiment Decisions

| Run / Process | Decision | Reason |
|---|---|---|
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped after epoch0 ckpt | High-LR `2e-4` GRPO is not a new mature algorithm mechanism and train reward regressed after epoch0. Keep `epoch_0-step_1330` only as an LR/eval data point. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | stopped/keep evidence | Current-code original-LR `1e-4` GRPO control with similar effective batch scale. It is not AWAC/IQL, Buffer-DPO, buffer guidance, self-imitation, hard-gate, or a bit-exact historical original-code rerun. Exact navtest at `epoch_0-step_1290` reached PDMS `0.896794`, clearly above the recent cap05 and v3 early checkpoints. Stopped on 2026-06-15 because zt3 is no longer to be used; keep artifacts only as evidence. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | Good limited diagnostic. Ratio/KL/clip/advantage logs are active, but this still does not validate a full PPO replay implementation. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z/local_eval_epoch0_exact_pool_8x1gpu_shards_20260614T031521Z` | completed eval | Local 8-way independent 1GPU exact navtest shards for the stopped `2e-4` epoch0 checkpoint. PDMS `0.872059`, clearly below the historical strong Stage3 baselines, so do not continue `2e-4` GRPO as a default route. |
| GRPO stable launcher defaults | changed | User clarified that original Stage3 used LR `1e-4`, and original `epoch0-1` PDMS is already around `0.88+`. Stable GRPO launchers now default to `1e-4`; `2e-4` should be treated as an explicit ablation only. |
| zt2 stale checkpoint watchers for old AWAC / old `2e-4` GRPO runs | stopped on 2026-06-14 | These watchers pointed to runs that no longer satisfy the protocol and could auto-consume GPUs when resources become free. |
| zt3 stale checkpoint watchers for stopped buffer-guided `2e-4` and stopped `2e-4` GRPO | stopped on 2026-06-14 | Exact eval of the stopped `2e-4` epoch0 checkpoint is handled by the local shard run; the stale remote watchers were redundant or invalid. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped | It tested buffer absorption, but Lightning did not log the returned buffer/self-imitation diagnostics. Continuing would not prove whether the auxiliary path was active. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped | It was launched before the PPO/advantage diagnostics patch. It could not report `gspo_approx_kl` or the full advantage-transform diagnostics, so it was stopped before becoming a full run. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | Hydra struct rejected the new override before the agent YAML was updated. No training occurred. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stop and replace | It entered training and wrote only `lr-AdamW`; with default `log_every_n_steps=50`, it is too low-information for a limited diagnostic. Replace with a shorter `log_every_n_steps=1` diagnostic and a long behavior-policy sync interval. |
| Local queued `stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait2_20260614T003614Z` launcher | stopped | It was a stale queued run from before this protocol and duplicates the zt2 buffer-guided experiment. |
| `supervise_recogdrive_stage3_experiment.py` auto-supervisor | stopped | It can auto-launch stale next actions without the new reference-audit protocol. Future launches should be manual after updating this file. |
| `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z` | completed smoke | New PPO replay path completed 40 train batches and wrote `epoch=0-step=200.ckpt`. This validates plumbing and diagnostics only; it is not a PDMS result and not a full algorithm verdict. |
| `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z` | completed diagnostic | Near-full sampling diagnostic passed the replay gates at `sample_time=16`: valid ratio stayed `1.0`, PPO clip fraction became active, KL stayed finite, and safety submetrics were logged. Promote to controlled 1-epoch training, not full 20-epoch training yet. |
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running/watch | Controlled full-navtrain 1-epoch PPO replay run. Replay validity is `1.0` and KL is small; one weak safety/reward batch appeared, so keep but do not promote until epoch eval. |
| `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z` | running | Controlled zt2 ablation. Same as i2 but `ppo_replay_inner_epochs=1`; currently cleaner ratio/clip/KL behavior and no hard stop signal. |
| `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z` | failed promotion gate | DPPO-style transition replay improved from step300 to epoch0 but stayed far below the original Stage3 `epoch0-1 ~= 0.88+` early gate: step300 `0.744740`, step600 `0.801195`, epoch0-step800 `0.831015`. Do not continue this transition replay implementation without a method-level redesign. |
| `stage3_grpo_rloo_selfimit_s16_lr1e4_b2acc4_8gpu_20260614T101257Z` | stopped on 2026-06-14 | This no-cap run was launched before the self-imitation scene-cap patch and had `checkpoint.every_n_train_steps=0`; given the original Stage3 `epoch0-1 ~= 0.88+` early gate, continuing it would delay a meaningful PDMS decision. |
| `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z` | stopped after `step-step_900` checkpoint | Original LR `1e-4`, sample_time `16`, scene cap `0.5`, and step checkpoints every `300` train steps. `step-step_300` exact navtest PDMS is `0.885432`, passing the original Stage3 early `0.88+` gate but not yet a final success. `step-step_600` is `0.885497`: EP improved, but NC/TTC/DDC declined. `step-step_900` later reached `0.887107` by recovering NC/TTC/DDC while EP fell back. Do not call this a final failure just because it is below the `epoch9-step13300` `0.9055` target; the reason not to keep it as the default is the matched-step safety/progress oscillation. |
| `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z` | stopped/deleted | This auxiliary-target-only safetygate run was stopped before checkpoint because the main GRPO update still allowed TTC/DDC-regressing positive-advantage samples. Do not repeat this exact configuration. |
| `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` | stopped/delete artifacts | Main TTC `0.95` and DDC `0.99` hard gates improved safety but suppressed EP/DAC. Step300 PDMS `0.879395`; step600 PDMS `0.882934`, still below cap05 step900 `0.887107` and far below the zt3 original-LR control step1290 `0.896794`. Do not repeat this main-hardgate + strict self-imitation configuration as a default route. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z` | running/evaluated through step1800 | This is not a new algorithm. It is a matched original-LR GRPO control on the current repository after AWAC/GRPO code changes: LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, no GSPO ratio, no hard gates, no buffer, no self-imitation. Epoch0-step1330 exact navtest PDMS is `0.893686`; step1500 is `0.891700`; step1800 dropped to `0.884834` with EP `0.837906` but weaker TTC `0.937222` and DDC `0.952958`. The best local point remains epoch0-step1330 and still trails the stronger stopped zt3 current-code original-LR control epoch0-step1290 `0.896794`. Keep running as a control and compare future epochs at matched horizon. |
| `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z` | stopped on 2026-06-15 | Control-aligned Buffer-DPO was a meaningful test of train-only elite-buffer preference absorption, but step300 exact navtest PDMS was only `0.859864` with NC `0.975078`, DAC `0.942000`, TTC `0.938375`, EP `0.809982`, comfort `0.999423`, and DDC `0.967087`. At matched effective batch and step it is far below the current-repo GRPO control step300 PDMS `0.881910`, so do not continue this exact Buffer-DPO weighting/path. Keep the train-only buffer as evidence; the failed part is the current DPO absorption objective. |
| `stage3_grpo_softsafety_selfimit_s16_lr1e4_b2acc4_local8_longsched_20260615T044311Z` | stopped before checkpoint | Launched before the schedule and buffer-distill audit. It used `MAX_EPOCHS=30`, `GRPO_SCHEDULER_EPOCHS=30`, `GRPO_BUFFER_GUIDANCE_ENABLED=false`, and `GRPO_SELF_IMITATION_BASELINE_MODE=group_leave_one_out`. Code inspection showed this self-imitation path only distills selected on-policy samples; it does not directly train on elite-buffer trajectories. Stopped at about step99 with no checkpoint so local 8GPU could run a corrected configuration. |
| `stage3_grpo_gspo_refkl_s16_lr1e4_b2acc4_chunk16_20260615T051804Z` | stopped before checkpoint | This was a corrected soft-safety + buffer-distill GRPO run but used `MAX_EPOCHS=40`, `GRPO_SCHEDULER_EPOCHS=40`, and `offline_rl_cache_elite_records_in_memory=false`. After comparing to the original Stage3 convention (`10` epochs with LR decaying to zero), a 40/60-epoch default was judged too large for the main algorithm comparison. It was stopped before any checkpoint/eval row, preserving only first TensorBoard diagnostics. |
| `stage3_grpo_softsafety_bufferdistill_s16_lr1e4_b2acc4_20e_20260615T054137Z` | stopped after `step-step_2400` | Prior soft-safety + low-weight buffer/self-imitation run. It was the strongest matched early-step variant at step300/600/900, but epoch0 underperformed the matched current-repo GRPO control through EP/DAC loss and step2400 regressed to PDMS `0.882233` with DDC `0.955141`. The failure mode is keeping non-decayed auxiliary imitation active too long; local 8GPU resources were moved to the auxiliary-decay run. |
  - First TensorBoard diagnostics at `step=49` confirm the intended 20-epoch algorithm path is active: `grpo_buffer_guidance_enabled=1.0`, `grpo_buffer_distill_loss=0.036794`, `grpo_buffer_distill_weight=0.0025` from the 4-epoch warmup, `grpo_buffer_distill_weight_sum=1.25`, `grpo_buffer_distill_zero_weight_batch=0.25`, `grpo_buffer_selected_valid_ratio=0.993421`, `grpo_buffer_best_valid_minus_gt=0.061794`, `grpo_buffer_best_valid_minus_il=0.201322`, `grpo_self_imitation_baseline_from_buffer=1.0`, `grpo_self_imitation_target_ratio=0.375`, and `soft_safety_mode_enabled=1.0`. Early train reward `0.745240`, base reward `0.780949`, shaped reward `0.745240`, loss `-0.079806`, reference KL loss `0.012285`, mean TTC `0.875000`, mean DDC `0.953125`, mean DAC `0.957031`, mean EP `0.769410`. This is a diagnostic batch only, not a PDMS result. No checkpoint/eval row yet as of the step49 check.
  - 2026-06-15 06:01 UTC liveness check: no `step300` checkpoint or eval row yet. TensorBoard remains at `step=49`, but two-point sampling over 20 seconds showed all 8 local rank CPU times advancing by about 20 seconds and GPU utilization fluctuating across all 8 A800s. This indicates active training rather than idle GPU memory. Continue waiting for the first step checkpoint before making a PDMS decision.
  - 2026-06-15 06:05 UTC TensorBoard advanced to `step=99`, still epoch `0`. Objective diagnostics remain active: `grpo_buffer_guidance_enabled=1.0`, `grpo_buffer_distill_loss=0.012516`, `grpo_buffer_distill_weight=0.0025`, `grpo_buffer_selected_valid_ratio=0.986111`, `grpo_buffer_best_valid_minus_gt=0.054013`, `grpo_buffer_best_valid_minus_il=0.350648`, `grpo_self_imitation_baseline_from_buffer=1.0`, `grpo_self_imitation_target_ratio=0.1875`, and `soft_safety_mode_enabled=1.0`. Train-batch reward was weaker than step49 (`reward=0.621427`, `base_reward=0.693929`, `shaped_reward=0.621427`) with `TTC=0.843750`, `DAC=0.886719`, `EP=0.689526`, and `DDC=0.992188`. Treat this as noisy batch-level telemetry, not a PDMS verdict; continue to the first `step300` checkpoint.
  - 2026-06-15 06:15 UTC TensorBoard advanced to `step=149`, still epoch `0`, with no checkpoint/eval row yet. The weak step99 batch did not persist: `reward=0.748632`, `base_reward=0.779800`, `shaped_reward=0.748632`, `TTC=0.898438`, `DAC=0.929688`, `EP=0.781910`, and `DDC=0.957031`. Auxiliary paths remain active with `grpo_buffer_distill_loss=0.019594`, `grpo_buffer_distill_weight=0.0025`, `grpo_buffer_selected_valid_ratio=0.986806`, `grpo_self_imitation_target_ratio=0.3125`, and `soft_safety_penalty_mean=0.064354`. Continue to `step300`; expected timing from the observed cadence is roughly 06:40-06:45 UTC.
  - 2026-06-15 06:26 UTC TensorBoard advanced to `step=199`, still epoch `0`, with no checkpoint/eval row yet. This train batch again had weaker raw reward and TTC: `reward=0.630186`, `base_reward=0.708118`, `shaped_reward=0.630186`, `TTC=0.781250`, `DAC=0.949219`, `EP=0.720952`, and `DDC=0.970703`; the soft-safety penalty rose to `0.122968`. Buffer/self-imitation paths remained active: `grpo_buffer_distill_loss=0.011043`, `grpo_buffer_selected_valid_ratio=0.993421`, `grpo_buffer_best_valid_minus_gt=0.041519`, `grpo_self_imitation_target_ratio=0.375`, and `grpo_self_imitation_baseline_from_buffer=1.0`. Keep watching but do not stop from this batch-level signal alone; the first hard decision remains the `step300` navtest PDMS and submetrics.
  - 2026-06-15 06:40 UTC TensorBoard advanced to `step=249`, still epoch `0`, with no checkpoint/eval row yet. The safety/progress batch metrics recovered again: `reward=0.750700`, `base_reward=0.776963`, `shaped_reward=0.750700`, `TTC=0.890625`, `DAC=0.933594`, `EP=0.763805`, and `DDC=0.998047`; `soft_safety_penalty_mean=0.054688`. Buffer/self-imitation remain active: `grpo_buffer_distill_loss=0.100647` with low active weight `0.0025`, `grpo_buffer_distill_zero_weight_batch=0.0`, `grpo_buffer_selected_valid_ratio=0.986111`, `grpo_buffer_best_valid_minus_gt=0.052328`, `grpo_self_imitation_target_ratio=0.4375`, and `grpo_self_imitation_baseline_from_buffer=1.0`. Continue to the first checkpoint/eval.
  - 2026-06-15 06:47 UTC `step-step=300.ckpt` was written and archived by the zt2 watcher as `checkpoint_archive/step-step_300.ckpt`; exact navtest PDM evaluation started under `unique_lock_watch_on_vla_zt2_4gpu/eval_step-step_300`. The eval command uses navtest logs/cache for evaluation only, `agent.grpo=False`, the archived step300 checkpoint, `nproc_per_node=4`, and exact async PDM scoring. At 06:52 UTC the eval was running normally: all four ranks loaded the checkpoint with `missing keys: 0`, rank0 had processed `200 / 3035` scenarios, and no PDMS result was available yet.
  - Training continued past the step300 checkpoint while zt2 evaluated it. Step299 train diagnostics were healthy: `reward=0.784173`, `base_reward=0.807923`, `TTC=0.886719`, `DAC=0.972656`, `EP=0.802520`, and `DDC=0.941406`; buffer valid ratio was `0.987171`. Step349 improved further on the sampled train batch: `reward=0.799551`, `base_reward=0.815598`, `TTC=0.914063`, `DAC=0.949219`, `EP=0.812711`, and `DDC=0.968750`; buffer valid ratio was `0.993421`. These are train-batch diagnostics only, but they show the run did not stall or immediately degrade around the first checkpoint.
  - 2026-06-15 07:12 UTC update: no final step300 PDMS yet. The zt2 exact navtest job was still healthy, with rank0 at `1500 / 3035` scenarios and no result files beyond logs/commands. Training had advanced to TensorBoard `step=399` with `reward=0.791209`, `base_reward=0.801324`, `TTC=0.933594`, `DAC=0.949219`, `EP=0.791827`, `DDC=0.982422`, and `soft_safety_penalty_mean=0.046776`. The current evidence is therefore forward progress but not a new navtest conclusion; the decision gate remains the exact step300 PDMS and submetrics.
  - 2026-06-15 07:37 UTC exact navtest result for `step-step_300`: PDMS `0.8907848533`, NC `0.983152`, DAC `0.971247`, TTC `0.954770`, EP `0.828101`, comfort `1.000000`, DDC `0.972318`; TLC was not emitted by this evaluator CSV. This clears the early health gate and is better than the matched current-repo GRPO control step300 PDMS `0.881910` by `+0.008875`, and better than cap05 self-imitation step300 `0.885432` by `+0.005353`. Against the current-repo GRPO control step300, the gain is mostly EP `+0.022870` and DDC `+0.007002`, with small NC `-0.001648` and TTC `-0.002801` regressions. Decision: continue this run to `step600` and epoch0, but keep NC/TTC under watch; it is promising at matched step, not yet a final improvement over the epoch0 controls (`0.893686` local, `0.896794` zt3 stopped).
  - 2026-06-15 07:52 UTC `step-step=600.ckpt` was archived as `checkpoint_archive/step-step_600.ckpt` and exact navtest evaluation started under `eval_step-step_600`. The four eval ranks loaded the checkpoint with `missing keys: 0`. The immediately preceding train diagnostic at `step=599` was strong: `reward=0.860550`, `base_reward=0.853096`, NC `0.978516`, DAC `0.976562`, TTC `0.921875`, EP `0.852216`, DDC `0.990234`, safe ratio `0.957031`, and `soft_safety_penalty_mean=0.042969`. Wait for exact step600 PDMS before changing the algorithm; this batch-level signal suggests the soft-safety/buffer-distill path has not collapsed after the first checkpoint.
  - 2026-06-15 08:39 UTC update: `step600` exact navtest evaluation is still running, not failed; rank0 reached `2700 / 3035` scenarios and no result CSV has been written yet. Local training is also still active on 8 GPUs and TensorBoard reached `step=799`, still before the next `step900` checkpoint. Latest train-batch diagnostics remain in the healthy early band: `reward=0.842200`, `base_reward=0.844967`, NC `0.980469`, DAC `0.980469`, TTC `0.933594`, EP `0.840740`, DDC `0.964844`, safe ratio `0.960938`, `soft_safety_penalty_mean=0.062263`. Buffer absorption is active but still low weight: `grpo_buffer_distill_weight=0.0025`, `grpo_buffer_guidance_target_ratio=0.4375`, `grpo_buffer_selected_valid_ratio=1.0`, `grpo_buffer_best_valid_minus_gt=0.053337`, and self-imitation target ratio `0.375`. Current interpretation: this is a real new candidate path beyond the earlier AWAC/Buffer-DPO failures, but the next hard conclusion must wait for exact `step600` PDMS and then epoch0.
  - 2026-06-15 08:44 UTC exact navtest result for `step-step_600`: PDMS `0.891373488`, NC `0.986077`, DAC `0.976520`, TTC `0.960043`, EP `0.815760`, comfort `1.000000`, DDC `0.970135`; TLC was not emitted by the evaluator CSV. This is slightly above this run's own `step300` by `+0.000589`: NC `+0.002925`, DAC `+0.005273`, TTC `+0.005273`, EP `-0.012341`, DDC `-0.002183`. Against the matched current-repo GRPO control `step600` PDMS `0.886777`, it is higher by `+0.004597`, with NC `+0.002801`, DAC `+0.008156`, TTC `+0.011781`, DDC `+0.006550`, but EP `-0.013320`. It also beats cap05 `step600` (`0.885497`, delta `+0.005877`) and main-hardgate v3 `step600` (`0.882934`, delta `+0.008439`). Interpretation: the soft-safety + low-weight buffer-distill route is now the strongest matched `step300/600` variant in this repo, with the gain coming from safety/DAC/DDC rather than progress. It is still not a final Stage3 success because it remains below the matched local epoch0 control `0.893686` and the stopped zt3 current-code epoch0 control `0.896794`; the historical original `0.9055` and Safe DiffGRPO `0.906184` are long-horizon targets and should not be used as early-step rejection thresholds. Continue to `step900` and epoch0 before changing algorithm again.
  - 2026-06-15 08:55 UTC `step-step=900.ckpt` was written at `08:52:50`, archived by the zt2 watcher as `checkpoint_archive/step-step_900.ckpt`, and exact navtest evaluation started under `eval_step-step_900`. The checkpoint loaded with `missing keys: 0`, and by `08:58 UTC` rank0 had processed `100 / 3035` scenarios, so the eval is healthy. The immediately preceding train diagnostic at `step=899` was weaker on safety: `reward=0.737128`, `base_reward=0.782527`, NC `0.945312`, DAC `0.960938`, TTC `0.792969`, EP `0.810374`, DDC `0.984375`, safe ratio `0.906250`, and `soft_safety_penalty_mean=0.113242`. Buffer/self-imitation remain active: `grpo_buffer_selected_valid_ratio=0.993056`, `grpo_buffer_best_valid_minus_gt=0.032702`, and self-imitation target ratio `0.375`. Treat this as a cautionary batch-level signal, not a stop signal; the decision gate is the exact `step900` navtest result.
  - 2026-06-15 09:28 UTC exact navtest result for `step-step_900`: PDMS `0.888204624`, NC `0.981752`, DAC `0.971659`, TTC `0.948014`, EP `0.827619`, comfort `0.999918`, DDC `0.972936`; TLC was not emitted by the evaluator CSV. This is below this run's step600 by `-0.003169`, mostly from NC `-0.004325`, DAC `-0.004861`, and TTC `-0.012028`, while EP recovered `+0.011860` and DDC improved `+0.002801`. Against matched current-repo GRPO control step900 `0.883839`, it is still higher by `+0.004366`, with DAC `+0.001565`, EP `+0.020786`, and DDC `+0.004655`, but NC `-0.005561` and TTC `-0.011781`. Against cap05 step900 `0.887107`, it is higher by `+0.001098`, with better DAC/EP/DDC but weaker NC/TTC. Interpretation: the method remains the strongest matched step900 variant among our tested variants, but the improvement mechanism shifted from step600's safety/DAC advantage to step900's progress/DDC advantage. Continue to step1200/epoch0, but if NC/TTC remain depressed the next design should reduce positive-advantage acceptance for TTC-regressing samples or make soft-safety adaptive, rather than increasing buffer distill blindly.
  - 2026-06-15 exact navtest result for `step-step_1200`: PDMS `0.885556525`, NC `0.979445`, DAC `0.970835`, TTC `0.948426`, EP `0.824450`, comfort `1.000000`, DDC `0.965851`. Against matched current-repo GRPO control step1200 PDMS `0.891510`, this is lower by `-0.005953`; the only clear safety gain is TTC `+0.002224`, while EP is lower by `-0.012790`, DAC by `-0.002307`, and DDC is higher by `+0.008939`. This weakens the early positive signal.
  - 2026-06-15 exact navtest result for `epoch_0-step_1330`: PDMS `0.884027892`, NC `0.985500`, DAC `0.967293`, TTC `0.964739`, EP `0.807146`, comfort `0.999835`, DDC `0.969146`. Against the matched local current-repo GRPO epoch0-step1330 control `0.893686`, this is lower by `-0.009659`, mainly because EP is lower by `-0.019860` and DAC lower by `-0.006261`, despite TTC `+0.006591` and DDC `+0.003749`. The correct conclusion is not "failed versus 90.55"; it is "underperforming the matched epoch0 control due to progress/DAC loss."
  - 2026-06-15 exact navtest result for `step-step_1500`: PDMS `0.887740968`, NC `0.983070`, DAC `0.970753`, TTC `0.950568`, EP `0.824720`, comfort `1.000000`, DDC `0.975655`. Against matched current-repo GRPO step1500 `0.891700`, it is lower by `-0.003959`; EP is slightly higher `+0.002472` and DDC higher `+0.005479`, but NC/DAC/TTC are lower. This is mixed, not a clear improvement.
  - 2026-06-15 exact navtest result for `step-step_1800`: PDMS `0.887295030`, NC `0.981710`, DAC `0.968281`, TTC `0.951392`, EP `0.827262`, comfort `0.999918`, DDC `0.972154`. Against matched current-repo GRPO step1800 `0.884834`, it is higher by `+0.002461`, mostly from TTC `+0.014170` and DDC `+0.019196`, while EP is lower by `-0.010643`. This shows safety-side recovery, but still not a robust progress/safety Pareto improvement.
  - 2026-06-15 exact navtest result for `step-step_2100`: PDMS `0.890360492`, NC `0.982781`, DAC `0.975943`, TTC `0.953534`, EP `0.822739`, comfort `1.000000`, DDC `0.969394`. This is a recovery from step1200/epoch0 and close to the run's best step600 `0.891373`, but there is no matched current-repo control step2100 in the current summary. Treat it as continuation evidence, not as a matched win.
  - 2026-06-15 exact navtest result for `step-step_2400`: PDMS `0.882233210`, NC `0.977385`, DAC `0.969929`, TTC `0.948838`, EP `0.818034`, comfort `0.999835`, DDC `0.955141`. This is the sharpest drop in this run after the step2100 recovery: PDMS `-0.008127` vs step2100, with DDC `-0.014253`, NC `-0.005396`, TTC `-0.004696`, and EP `-0.004704`. The regenerated `navtest_pdms_analysis.md` shows first-to-latest deltas of PDMS `-0.008552`, EP `-0.010067`, and DDC `-0.017177`.
  - Final corrected interpretation for this run: early step300/600/900 were positive against matched-step variants, but epoch0 underperformed the current-repo GRPO control through EP/DAC loss, and step2400 later regressed further through DDC/safety. The useful lesson is that low-weight buffer/self-imitation can help early, but keeping the auxiliary terms active into later epoch0/epoch1 training is not robust. This run was stopped on 2026-06-15 after the step2400 eval, and local 8GPU resources were moved to the auxiliary-decay run. The `90.55` historical original Stage3 result is `epoch9-step13300`; it remains the long-horizon target, not the reason this run was stopped.
| GRPO buffer cache default | patched on 2026-06-15 | Train-only elite buffer records are small compressed per-token files. The full strict-v2 buffer is about `342MB` for `85109` records, so per-rank lazy in-memory caching is safe on the current machines and avoids repeated `.pkl.xz` decompression on every batch. `launch_recogdrive_stage3_grpo_softsafety_selfimit_2b_local.sh` and `launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh` now default `OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY=true`. |
| `stage3_awac_keepbest_buffer_zt2_gpu0123_softsafety_source_20260615T045541Z` | background buffer exploration | zt2 GPU 0-3 are running train-only keep-best elite-buffer exploration, not an AWAC/IQL training run. The process writes into the strict-v2 train buffer `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z` with `MERGE_EXISTING_RECORDS=1`, so records are updated only by keep-best merge semantics. It uses navtrain logs and `metric_cache_train_full`; navtest remains evaluation-only. Important correction from the 2026-06-15 06:02 UTC audit: the running process has `AUTO_POLICY_CHECKPOINT_DIR=/mnt/project/VLA-AD/outputs/stage3_grpo_softsafety_selfimit_s16_lr1e4_b2acc4_local8_longsched_20260615T044311Z/train`, and cycle 1 resolved `policy_checkpoint=` because that stopped run has no checkpoint. Therefore this cycle should be treated as IL/reference plus structured exploration, not as a buffer generated from the current 20-epoch GRPO policy. Do not count it as current-policy buffer evidence. Launched 4 shards on zt2 GPU `0,1,2,3` with `SHARD_COUNT=4`, `ONLINE_POLICY_SAMPLES=32`, and `ELITE_TOP_M=8`. Cycle 1 completed all `85109` scenes with shard mean best-valid reward about `0.97534-0.97546`, has-valid ratio `1.0`, selected-valid ratio about `0.979`, and best-valid-above-GT about `58.8-59.1%`. Cycle 2 is active as of 2026-06-15 21:19 UTC at about `6144 / 21277` scenes per shard. |
| GRPO schedule defaults | patched on 2026-06-15 | User requested a longer Stage3 schedule while keeping the original initial LR and avoiding decay to zero, then correctly flagged that `60` epochs is too large relative to the original Stage3 `10` epochs. Main GRPO launchers now default to `LR=1e-4`, `MAX_EPOCHS=20`, `GRPO_SCHEDULER_EPOCHS=${MAX_EPOCHS}`, and `GRPO_SCHEDULER_MIN_LR=1e-5`. This doubles the original epoch budget without making schedule length the dominant experimental variable. `ReCogDriveAgent` rejects `grpo_scheduler_min_lr <= 0`, and the launcher rejects `GRPO_SCHEDULER_EPOCHS != MAX_EPOCHS`. Dry-run verification confirmed Hydra receives `trainer.params.max_epochs=20`, `agent.grpo_scheduler_epochs=20`, and `agent.grpo_scheduler_min_lr=1e-5`. |
| Stage3 next-action gate | patched | `scripts/training/decide_recogdrive_stage3_next_action.py` now tracks the current-repo original-LR control run instead of the stopped `2e-4` run, and its gated next experiment is the control-aligned Buffer-DPO launcher in this repository. After the step300 eval, dry-run result on 2026-06-14 UTC is `launch_buffer_dpo_after_current_checkpoint`, but default policy waits while the current control is still running. |
| Stage3 exact eval infrastructure | patched | Distributed eval timeout is now configurable and defaults to `3600s` for async/exact PDM runners and watcher launchers. Watchers also write global per-checkpoint done markers under `GLOBAL_EVAL_LOCK_DIR` after successful eval so relaxed/strict watchers do not repeat the same checkpoint. These changes only affect orchestration; they do not change trajectory inference, PDM scoring, or submetric calculation. |
| Train-only elite buffer exploration | paused on zt3 | The zt3 keep-best buffer builder was stopped on 2026-06-15 because zt3 is no longer to be used. The existing strict-v2 train-only buffer remains valid evidence and may be used by Buffer-DPO/preference absorption; future buffer generation must run on local or zt2 resources only. Use navtest only for checkpoint diagnosis, never for buffer generation or training reward. |
| Train-only elite buffer validation | passed on 2026-06-14 | Full strict-v2 validation of `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z` passed with `85109` records, valid-candidate ratio `0.982056`, has-valid ratio `1.0`, mean best-valid reward `0.973614`, `pct_best_valid_above_gt=0.582477`, and `pct_best_valid_above_il=0.792349`. Summary files: `stage3_awac_keepbest_buffer_zt3_wait_20260614T142658Z/validate_live_summary_20260614T2015Z.json` and `.csv`. This supports using the buffer for future train-only Buffer-DPO/preference absorption; it does not by itself prove sampler improvement. |

## Artifact Retention And Cleanup

Policy:
- Keep summary evidence in this ledger before deleting large artifacts.
- Keep current training runs, current evaluation watchers, train-only elite buffers, and strong reproducible baselines unless explicitly superseded.
- Remove large checkpoints, Lightning logs, and duplicate evaluation shard outputs for failed, invalid, or smoke-only attempts after their result and failure mode are recorded here.
- Do not use deleted navtest artifacts as training data; navtest remains diagnostic only.

Runtime hygiene on 2026-06-15 UTC:
- Stopped three stale local `watch_recogdrive_stage3_early_gate.sh` processes that still pointed at already stopped/superseded local runs: `stage3_grpo_rpp_constrained_s16_lr1e4_b2acc4_local8_20260615T035445Z`, `stage3_grpo_softsafety_selfimit_s16_lr1e4_b2acc4_local8_longsched_20260615T044311Z`, and `stage3_grpo_gspo_refkl_s16_lr1e4_b2acc4_chunk16_20260615T051804Z`. These were local log/gate watchers only, not zt2 training/eval jobs.
- Stopped the local `stage3_grpo_softsafety_bufferdistill_s16_lr1e4_b2acc4_20e_20260615T054137Z` process tree on 2026-06-15 after exact navtest evaluation through `step2400`. The stable launcher had already exited, so the remaining orphaned `torchrun`/worker tree was terminated and local GPU memory returned to `0 MB` on all 8 GPUs. Its zt2 checkpoint watcher had no pending useful checkpoint after step2400 and is no longer a priority.

### 2026-06-15 Planned Attempt: Soft-Safety GRPO With Auxiliary Decay

Motivation:
- Corrected comparison rule:
  - The historical original Stage3 `0.9055` result was obtained at `epoch9-step13300`.
  - It is a long-horizon target, not a rejection threshold for `step300`, `step600`, `epoch0`, or other early checkpoints.
  - Early algorithm decisions must compare matched effective batch, matched optimizer step, matched epoch, and similar train-scene exposure.
- Current evidence from `stage3_grpo_softsafety_bufferdistill_s16_lr1e4_b2acc4_20e_20260615T054137Z`:
  - `step300`, `step600`, and `step900` beat matched early variants.
  - `epoch0-step1330` underperformed the matched current-repo GRPO control by `-0.009659`, mainly through lower EP and DAC despite better TTC/DDC.
  - `step2100` recovered to `0.890360`, but there is no matched current-repo control at that step yet.
  - `step2400` then dropped to `0.882233`, mostly through DDC/safety regression. This confirms that the old non-decay auxiliary schedule is not the version to spend more local 8GPU time on.
- The likely failure mode is not "below 90.55"; it is "early auxiliary buffer/self-imitation helps at step600 but remains too active by the epoch0 comparison and suppresses GRPO progress/DAC learning."

Reference audit:
- DPPO (`arXiv:2409.00588`, official `irom-princeton/dppo`) supports policy-gradient fine-tuning of diffusion policies and reports that the diffusion parameterization can give structured on-manifold exploration when the policy-optimization path is respected.
- RIPT-VLA (`arXiv:2505.17016`, official `Ariostgx/ript-vla`) supports VLA post-training with dynamic rollout sampling and leave-one-out advantage estimation, reinforcing that rollout/on-policy improvement should remain the main learning signal.
- FDPP (`arXiv:2501.08259`) supports using preference/reward guidance for diffusion-policy fine-tuning, but emphasizes KL/reference regularization to preserve the pretrained policy. This matches using train-buffer knowledge as a constrained auxiliary, not as a dominant regression objective.

Implementation change:
- Add step-aware auxiliary loss schedules:
  - `linear_warmup_linear_decay`
  - `linear_warmup_cosine_decay`
- Propagate `trainer.global_step` into `ReCogDriveAgent` and `ReCogDriveDiffusionPlanner` before every train forward pass, not only at epoch start.
- Keep the main objective as original-LR GRPO/GSPO with reference KL.
- Let buffer distill and self-imitation provide a small early teacher, then decay both from `step600` to `step1330`.

Config for the next controlled run:
- Launcher:
  - `scripts/training/launch_recogdrive_stage3_grpo_softsafety_auxdecay_2b_local.sh`
- Defaults:
  - `LR=1e-4`
  - `MAX_EPOCHS=20`
  - `GRPO_SCHEDULER_EPOCHS=20`
  - `GRPO_SCHEDULER_MIN_LR=1e-5`
  - `GRPO_SAMPLE_TIME=16`
  - effective batch `64` on local 8GPU when `BATCH_SIZE=2`, `ACCUMULATE_GRAD_BATCHES=4`
  - `GRPO_SAFETY_ADVANTAGE_MODE=soft_penalty`
  - TTC/DDC thresholds `0.95/0.99`
  - buffer distill weight target `0.01`, cosine decay `step600 -> step1330`, end `0.0`
  - self-imitation weight target `0.01`, cosine decay `step600 -> step1330`, end `0.0`
  - Buffer-DPO disabled.

Expected diagnostics:
- Before `step600`:
  - Keep the earlier positive matched-step behavior.
  - `grpo_buffer_distill_weight` and `grpo_self_imitation_weight` should be nonzero but small.
- By `epoch0-step1330`:
  - Both auxiliary weights should reach `0.0`.
  - EP and DAC should be closer to the matched current-repo GRPO control than the current soft-safety-bufferdistill run.
  - TTC/DDC should not collapse relative to the current run.
- Promotion gate:
  - Treat `step300/600/900/1330` only as matched-exposure diagnostics.
  - Do not finally reject a non-catastrophic GRPO-family method before
    `epoch3-4` or roughly `step5000`/equivalent exposure.
  - Compare to original Stage3 `0.9055` only near `epoch9-step13300` or equivalent train-scene exposure.

Verification:
- Dry run wrote the expected Hydra config with `MAX_EPOCHS=20`, `GRPO_SCHEDULER_EPOCHS=20`, `GRPO_SCHEDULER_MIN_LR=1e-5`, and both auxiliary schedules set to `linear_warmup_cosine_decay`.
- Direct schedule test produced weight `0.0025` at steps `0/300/600`, `0.00125` near the midpoint `965`, and `0.0` at `1330+`, confirming step-aware decay will take effect.

Launch status:
- Launched on `training-vla-zt2` at `2026-06-15T14:15Z`:
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_softsafety_auxdecay_s16_lr1e4_b2acc8_zt2_4gpu_20260615T141527Z`
- Training GPUs:
  - zt2 GPU `4,5,6,7`
  - `GPUS_PER_NODE=4`
  - `BATCH_SIZE=2`
  - `ACCUMULATE_GRAD_BATCHES=8`
  - effective optimizer batch remains `64`, matching the local 8GPU `batch=2, accumulate=4` runs.
- Evaluation watcher:
  - zt2 GPU `0,1,2,3`
  - exact navtest PDMS eval per step checkpoint, using navtest only for evaluation.
- Initial process check:
  - stable launcher, torchrun, early-gate watcher, and zt2 checkpoint watcher are alive.
  - Hydra command contains `agent.offline_rl_grpo_buffer_distill_loss_schedule=linear_warmup_cosine_decay`, `agent.offline_rl_grpo_buffer_distill_decay_start_step=600`, `agent.offline_rl_grpo_buffer_distill_decay_end_step=1330`, and the matching self-imitation decay fields.
- Startup correction:
  - The first zt2 launch failed before any training checkpoint because `AgentLightningDiT` called `_propagate_training_progress()` but that helper had only been added to the sibling `AgentLightningModule` class.
  - This is a plumbing failure, not an algorithm result.
  - Fixed by adding `AgentLightningDiT._propagate_training_progress()` as a direct delegation to the shared implementation and recompiling `navsim/planning/training/agent_lightning_module.py`.
- Relaunch status:
  - Relaunched fixed run:
    - `/mnt/project/VLA-AD/outputs/stage3_grpo_softsafety_auxdecay_s16_lr1e4_b2acc8_zt2_4gpu_fix_20260615T142305Z`
  - It waited for zt2 GPU `4,5,6,7` while the old run's `step2400` exact navtest eval was active, then started training at `2026-06-15T14:41Z`.
  - First TensorBoard scalar at `step=49` confirms the fixed schedule is active: `lr=1e-4`, `grpo_buffer_distill_weight=0.0025`, `grpo_self_imitation_weight=0.005`, `grpo_buffer_selected_valid_ratio=0.986842`, `reward=0.658414`, `base_reward=0.739177`, `mean_ep=0.726131`, `mean_ttc=0.820313`, and `mean_ddc=0.898438`. This is only a launch-health diagnostic, not a PDMS result.
  - A matching local 8GPU run was then started after stopping the old non-decay run:
    - `/mnt/project/VLA-AD/outputs/stage3_grpo_softsafety_auxdecay_s16_lr1e4_b2acc4_20e_local8_20260615T152043Z`
  - The local run uses `BATCH_SIZE=2`, `ACCUMULATE_GRAD_BATCHES=4`, `devices=8`, effective batch `64`, and the same `step600 -> step1330` cosine auxiliary-decay schedule. Its zt2 exact-eval watcher is active on GPU `0,1,2,3` and is configured to wait/share resources without killing existing zt2 jobs.
  - First local8 TensorBoard scalar at `step=49` confirms the same schedule is active: `lr=1e-4`, `grpo_buffer_distill_weight=0.0025`, `grpo_self_imitation_weight=0.005`, `grpo_buffer_selected_valid_ratio=1.0`, `reward=0.725306`, `base_reward=0.761965`, `mean_ep=0.751199`, `mean_ttc=0.871094`, `mean_ddc=0.970703`, and `soft_safety_penalty_mean=0.080039`. This is a launch-health diagnostic only; wait for exact navtest PDMS at `step300` before judging the algorithm.
  - 2026-06-15 16:54 UTC exact navtest result for local8 `step-step_300`: PDMS `0.883328294`, NC `0.985459`, DAC `0.964162`, TTC `0.963009`, EP `0.809764`, comfort `0.999918`, DDC `0.972977`; TLC was not emitted by the evaluator CSV. This is only `+0.001418` above the matched current-repo GRPO control step300 `0.881910`, but below the prior non-decay soft-safety buffer-distill step300 `0.890785` by `-0.007457` and below cap05 step300 `0.885432` by `-0.002104`. The early evidence therefore does not preserve the previous early buffer-distill gain. The reason to continue this run is narrower: test whether auxiliary decay prevents the later EP/DAC/DDC degradation seen in the non-decay run over the relaxed `epoch3-4` / `step5000` window, not because step300 is already a strong improvement.
  - 2026-06-15 17:02 UTC exact navtest result for zt2fix `step-step_300`: PDMS `0.880672021`, NC `0.978786`, DAC `0.961773`, TTC `0.943978`, EP `0.827483`, comfort `1.000000`, DDC `0.969682`; TLC was not emitted by the evaluator CSV. This is below the matched current-repo GRPO control step300 by `-0.001238` and below the non-decay soft-safety buffer-distill step300 by `-0.010113`.
  - 2026-06-15 17:47 UTC exact navtest result for local8 `step-step_600`: PDMS `0.881064728`, NC `0.975737`, DAC `0.971494`, TTC `0.938046`, EP `0.824112`, comfort `1.000000`, DDC `0.965192`; TLC was not emitted by the evaluator CSV. This is below the run's own step300 by `-0.002264` and below the matched current-repo GRPO control step600 `0.886777` by `-0.005712`. The failure mode is safety-side regression versus step300: NC `-0.009722`, TTC `-0.024963`, and DDC `-0.007785`, partly offset by EP `+0.014348` and DAC `+0.007332`.
  - Revised after the epoch4/step5000 audit and the local8 step600 result: do not use step300 alone as a rejection gate, and do not use epoch0 as a final gate for non-catastrophic GRPO-family methods. Current aux-decay is no longer the highest-priority route, but it should continue toward the relaxed `epoch3-4` / `step5000` window unless NC/TTC/DDC collapse or resource contention forces a choice. If it remains below matched clean-GRPO controls by that window, stop and keep the result as evidence that this decay schedule removed the early buffer-distill gain without solving late stability. Allocate future main resources to GRPO-primary designs closer to original/Safe DiffGRPO behavior, with buffer knowledge used as candidate/proposal or tightly safety-aware preference signal rather than broad auxiliary regression.
  - 2026-06-15 18:00 UTC liveness check: local8 is still running and has not
    emitted a `step-step=900.ckpt` yet. Latest TensorBoard scalar is at
    `step=749`: `reward=0.687542`, `base_reward=0.756203`, safe ratio
    `0.910156`, NC `0.957031`, DAC `0.953125`, TTC `0.804688`, EP `0.791794`,
    DDC `0.902344`, buffer-distill weight `0.002252`, and self-imitation weight
    `0.004503`. This train batch is safety-weak, but the run should still reach
    at least the next exact-eval checkpoints and should not receive a final
    rejection before the relaxed `epoch3-4` / `step5000` window unless safety
    collapses.
  - 2026-06-15 18:03 UTC follow-up: local8 TensorBoard advanced to `step=799`
    with a healthier train batch: `reward=0.804166`, `base_reward=0.823386`,
    EP `0.823975`, TTC `0.925781`, DDC `0.970703`, buffer-distill weight
    `0.002069`, and self-imitation weight `0.004138`. No `step-step=900.ckpt`
    exists yet, so this remains a liveness/diagnostic update rather than a PDMS
    conclusion.
  - 2026-06-15 18:19 UTC follow-up: local8 is still active and still has only
    `step-step=300/600.ckpt`; watcher reports no pending evals and
    `training_state=running`. Latest TensorBoard scalar is `step=849`:
    `reward=0.761932`, `base_reward=0.793480`, NC `0.964844`, DAC
    `0.953125`, TTC `0.816406`, EP `0.812978`, DDC `0.962891`, safe ratio
    `0.933594`, buffer-distill weight `0.001848`, and self-imitation weight
    `0.003697`. The low TTC batch means safety remains under watch, but this
    is not a checkpoint-level PDMS result and does not justify interruption
    under the relaxed horizon rule.
  - 2026-06-15 18:24 UTC `step-step=900.ckpt` was archived by the zt2 watcher
    as `checkpoint_archive/step-step_900.ckpt` and exact navtest PDM evaluation
    started under `eval_step-step_900`. All four eval ranks loaded the checkpoint
    with `missing keys: 0`. By 18:42 UTC, rank0 had processed `1000 / 3035`
    scenarios with no traceback or error.
  - 2026-06-15 19:13 UTC exact navtest result for local8 `step-step_900`: PDMS
    `0.892334660`, NC `0.981669`, DAC `0.973719`, TTC `0.950816`, EP
    `0.832688`, comfort `1.000000`, DDC `0.974172`; TLC was not emitted by the
    evaluator CSV. This is a meaningful recovery from step600 and the best
    matched step900 variant so far:
    - vs current-repo clean GRPO step900: PDMS `+0.008496`, EP `+0.025854`,
      DDC `+0.005891`, DAC `+0.003625`, but NC `-0.005643` and TTC
      `-0.008980`.
    - vs non-decay buffer-distill step900: PDMS `+0.004130`, EP `+0.005068`,
      TTC `+0.002801`, DDC `+0.001236`, DAC `+0.002060`, essentially tied NC
      (`-0.000082`).
    - vs cap05 self-imitation step900: PDMS `+0.005227`, EP `+0.009629`,
      DDC `+0.002430`, DAC `+0.006508`, but NC `-0.002595` and TTC
      `-0.005026`.
    - vs current-repo clean GRPO epoch0-step1330: PDMS is still lower by
      `-0.001352`, with better EP `+0.005682` and DDC `+0.008774` but weaker
      NC `-0.004820` and TTC `-0.007332`.
    Interpretation: the auxiliary-decay idea should no longer be treated as
    weak after step600. It has a real step900 recovery signal, driven by EP/DAC
    and DDC, but TTC/NC remain the tradeoff to watch. Continue to step1200 and
    epoch0/relaxed horizon before a final decision.
  - 2026-06-15 19:19 UTC status under the relaxed-horizon rule: local8 has no
    new exact PDMS beyond step900 yet, but TensorBoard has advanced to
    `step=1149`. At that point the auxiliary losses are already mostly decayed:
    `grpo_buffer_distill_weight=0.000360` and
    `grpo_self_imitation_weight=0.000721`. Train-batch metrics are
    `reward=0.792780`, `base_reward=0.815566`, safe ratio `0.929688`, NC
    `0.984375`, DAC `0.945312`, TTC `0.937500`, EP `0.811852`, and DDC
    `0.943359`. The next exact checkpoint is important because it tests whether
    the GRPO path can retain the step900 EP/DDC gain after buffer/self-imitation
    guidance has nearly faded.
  - 2026-06-15 19:23 UTC: `step-step=1200.ckpt` was written, archived by the
    zt2 watcher as `checkpoint_archive/step-step_1200.ckpt`, and exact navtest
    evaluation started under `eval_step-step_1200`. The watcher handled this
    through its normal age check and lock path, so no duplicate manual eval was
    launched.
  - 2026-06-15 19:54 UTC status: no new exact PDMS beyond step900 yet. The
    `step1200` exact eval is healthy: rank0 progressed to `1900 / 3035`
    scenarios with no traceback. The local 8GPU training process is also still
    active; TensorBoard's latest scalar is `step=1299`, with auxiliary weights
    nearly zero (`grpo_buffer_distill_weight=0.000011`,
    `grpo_self_imitation_weight=0.000022`). This means the next result tests
    mostly the GRPO path after the buffer/self-imitation teacher has faded.
  - 2026-06-15 20:14 UTC exact navtest result for local8 `step-step_1200`:
    PDMS `0.886895855`, NC `0.979321`, DAC `0.972318`, TTC `0.947438`, EP
    `0.825674`, comfort `1.000000`, DDC `0.963627`. This is a clear regression
    from the run's `step900` best (`-0.005439` PDMS, `-0.007013` EP,
    `-0.010545` DDC, `-0.003378` TTC) and is also below the current-repo clean
    GRPO step1200 point (`0.891510`) by `-0.004614`. Interpretation: the buffer
    distill / self-imitation teacher produced a useful early bump, but once it
    faded the policy did not retain the gain. Do not allocate more primary
    resources to this exact auxiliary-decay recipe; keep the result as evidence
    that buffer knowledge must be absorbed through a stronger policy-gradient or
    preference mechanism rather than a short-lived low-weight regression teacher.

### 2026-06-15 Planned Attempt: Clean GRPO-Primary Wide-Horizon Control On zt2

Motivation:
- The relaxed audit shows that the strongest reliable path is still original-LR
  GRPO / Safe-DiffGRPO-style training, and that some methods recover only after
  epoch3.
- The local aux-decay run recovered strongly at step900. A clean GRPO-primary
  run is still needed as the matched long-window comparison spine, so we can
  separate auxiliary-decay gains from the base GRPO/Safe-DiffGRPO recovery
  pattern.
- zt2 GPU `4,5,6,7` are free while GPU `0,1,2,3` continue the train-only buffer
  generator. Use the free cards for a clean GRPO-primary wide-horizon control
  rather than launching another AWAC/Buffer-DPO variant.

Config:
- Host / GPUs: `training-vla-zt2`, training GPU `4,5,6,7`.
- Eval watcher: `training-vla-zt2`, GPU `0,1,2,3`, exact navtest PDMS per
  checkpoint. Navtest remains evaluation-only.
- Effective batch: `4 GPUs * batch 2 * accumulate 8 = 64`, matching the local
  8GPU `batch=2, accumulate=4` comparison runs.
- LR / schedule: `1e-4`, `20` epochs, scheduler min LR `1e-5`, scheduler epochs
  exactly equal to max epochs.
- GRPO settings: `sample_time=16`, BC anneal `0.10 -> 0.05` over `5` epochs,
  reference KL `0.02`, no buffer guidance, no self-imitation, no Buffer-DPO, no
  hard TTC/DDC gate.
- `GRPO_USE_GSPO_RATIO=false` to keep this as the clean current-repo GRPO
  control path. A strict-GSPO ablation can be run separately after this control
  reaches the relaxed horizon.
- 2026-06-15 19:19 UTC status: no checkpoint or exact PDMS result yet. Latest
  TensorBoard scalar is `step=149` with `lr=1e-4`, `reward=0.663988`,
  `base_reward=0.711589`, safe ratio `0.835938`, NC `0.964844`, DAC
  `0.867188`, TTC `0.921875`, EP `0.688607`, DDC `1.000000`, BC coefficient
  `0.10`, and reference KL loss `0.008346`. This is only a train-batch
  diagnostic; the first usable comparison point remains the step300 exact
  navtest PDMS.
- Resource status at the same time: local 8 GPUs are occupied by the local8
  aux-decay run, zt2 GPU `4,5,6,7` are occupied by this clean GRPO run, and zt2
  GPU `0,1,2,3` are already running Python GPU tasks. Do not launch a third
  large experiment until at least one stream frees resources or the next
  checkpoint/eval changes the priority.
- 2026-06-15 19:39 UTC status: still no checkpoint or exact PDMS result. This
  is not a stall. A 20-second process sample showed all four clean-GRPO worker
  CPU times advancing by about 20 seconds, and zt2 GPU `4,5,6,7` utilization
  resumed to `79/42/34/72%` after an earlier low-util snapshot. TensorBoard's
  latest scalar remains `step=199`; continue waiting for the first `step300`
  checkpoint/eval rather than launching another competing run.
- 2026-06-15 19:45 UTC status: still no checkpoint or exact PDMS result.
  Watcher reports `training_state=running`, TensorBoard still shows latest
  scalar at `step=199`, and zt2 GPU utilization is high on the active training
  cards. Continue waiting rather than treating the lack of a checkpoint as a
  failure.
- 2026-06-15 19:46 UTC liveness check: still no checkpoint, but worker CPU
  times are close to elapsed time (`~01:47` CPU over `~01:49` elapsed for each
  rank) and zt2 GPU `4,5,6,7` remain allocated to the run with nonzero
  utilization on the active cards. Treat this as slow but alive until the
  `step300` checkpoint or a real error appears.
- 2026-06-15 19:48 UTC status: still no checkpoint. TensorBoard remains at
  `step=199`, but all four worker CPU times continue to track elapsed time
  (`~01:49` CPU over `~01:51` elapsed), so this is still treated as active
  training rather than a failed run.
- 2026-06-15 19:49 UTC update: TensorBoard advanced to `step=249`, confirming
  training progress beyond the earlier `step199` plateau. The latest train
  batch recovered substantially: `reward=0.753582`, `base_reward=0.759062`,
  safe ratio `0.914063`, NC `1.000000`, DAC `0.914063`, TTC `0.898438`, EP
  `0.749780`, DDC `0.996094`, and reference KL loss `0.008892`. There is still
  no `step300` checkpoint, so the first usable navtest comparison remains
  pending.
- 2026-06-15 19:54 UTC status: still no checkpoint or exact PDMS result. The
  watcher reports `training_state=running`, and zt2 GPU `4,5,6,7` remain
  allocated with sampled utilization `51/71/79/41%`, so the run remains slow but
  alive rather than stalled.
- 2026-06-15 20:14 UTC status: `step-step=300.ckpt` was written at 20:10 UTC,
  archived by the zt2 watcher at 20:12 UTC, and exact navtest evaluation started
  under `unique_lock_watch_on_vla_zt2_4gpu/eval_step-step_300`.
- 2026-06-15 21:12 UTC exact navtest result for `step-step_300`: PDMS
  `0.887157480`, NC `0.984017`, DAC `0.967046`, TTC `0.952628`, EP `0.826053`,
  comfort `0.999918`, DDC `0.972360`; TLC was not emitted by the evaluator CSV.
  This is `+0.005247` above the current-repo clean GRPO step300 `0.881910`, but
  below current-repo clean epoch0-step1330 `0.893686`, auxiliary-decay step900
  `0.892335`, Safe DiffGRPO epoch3 `0.895788`, and zt3 evidence-only GRPO
  epoch0-step1290 `0.897167`. Treat it as a healthy launch check, not a final
  verdict.
- 2026-06-15 21:19 UTC health check:
  - Training is still alive on zt2 GPU `4,5,6,7`; TensorBoard latest scalar is
    `step=449`.
  - Recent train diagnostics remain healthy for a clean-GRPO control: reward
    `0.837572`, base reward `0.824627`, safe ratio `0.945312`, EP `0.805772`,
    TTC `0.921875`, group reward std `0.242570`, mixed-group ratio `0.375`,
    and all-unsafe-group ratio `0.0`.
  - No `step600` checkpoint exists yet. Based on the recent TensorBoard cadence,
    expected `step600` checkpoint timing is around 2026-06-15 22:24 UTC.

Expected diagnostic movement:
- This run should reproduce or improve the current-repo clean GRPO control band:
  `0.881910` step300, `0.886777` step600, `0.893686` epoch0.
- The key decision point is not step300; promote or reject after `epoch3-4` /
  about `step5000`, with submetric focus on EP/DAC versus TTC/DDC tradeoff.

Launch status:
- Dry-run config check passed before launch:
  - `max_epochs=20`
  - `grpo_scheduler_epochs=20`
  - `grpo_scheduler_min_lr=1e-5`
  - `grpo_use_gspo_ratio=false`
  - `offline_rl_enabled=false`
  - `grpo_buffer_guidance_enabled=false`
  - `grpo_buffer_distill_loss_weight=0`
  - `grpo_self_imitation_loss_weight=0`
- Launched on zt2 at `2026-06-15T17:57:15Z`:
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_clean_s16_lr1e4_b2acc8_zt2_4gpu_e4gate_20260615T175659Z`
- Checkpoint eval watcher started on zt2:
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_clean_s16_lr1e4_b2acc8_zt2_4gpu_e4gate_20260615T175659Z/unique_lock_watch_on_vla_zt2_4gpu`
- Initial log check at `2026-06-15T17:58Z`:
  - torchrun command contains the intended clean-GRPO overrides.
  - watcher reports `training_state=running` and no pending checkpoint yet.
  - no existing zt2 task was killed; buffer generation remains on GPU `0,1,2,3`.
- 2026-06-15 18:19 UTC health check:
  - no checkpoint has been written yet, and the watcher reports no pending
    checkpoint evals.
  - TensorBoard still has only the initial `lr-AdamW` scalar. This is expected
    before the first logging interval because the zt2 run uses `4` GPUs with
    `accumulate_grad_batches=8`.
  - zt2 GPU `4,5,6,7` are active for training, with sampled utilization
    `100/100/47/100%` and about `39.6-39.7GB` memory used per GPU. GPU `0,1,2,3`
    remain occupied by the train-only buffer/eval side work, and were not
    preempted.
- 2026-06-15 18:46 UTC health check:
  - The apparent `0%` GPU-utilization samples on GPU `4,5,6,7` were not a stall.
    A 30-second CPU-time diff showed all four train workers advancing by about
    `29` seconds each, and the event file updated at 18:43 UTC.
  - TensorBoard has now reached `step=99`: reward `0.800026`, base reward
    `0.791387`, safe ratio `0.945312`, NC `0.988281`, DAC `0.960938`, TTC
    `0.945312`, EP `0.755205`, DDC `1.000000`, comfort `0.718750`, BC coeff
    `0.10`, and reference KL loss `0.006810`. This confirms the clean-GRPO run
    has entered normal training. No checkpoint has been written yet, so there is
    still no navtest result for this run.
- Health check at `2026-06-15T18:01Z`:
  - zt2 GPU `4,5,6,7` are now actively training with about `39 GB` allocated per
    GPU and `98-99%` utilization.
  - training logs show `LOCAL_RANK: 0..3 - CUDA_VISIBLE_DEVICES: [4,5,6,7]`,
    confirming the clean-GRPO run is bound to the intended zt2 cards.
  - The eval watcher remains alive and reports no pending checkpoint yet.
- Follow-up at `2026-06-15T18:03Z`:
  - The run remains active on GPU `4,5,6,7`; the TensorBoard event file has only
    the initial `lr-AdamW` scalar so far, which is expected before the first
    logging interval.
  - No checkpoint has been written yet, and the eval watcher still reports no
    pending checkpoint.

### 2026-06-15 Planned Attempt: Controlled DPPO-Step Replay Diagnostic

Motivation:
- The local auxiliary-decay buffer teacher produced a useful `step900` bump but
  regressed at `step1200` once the teacher weight had nearly faded. This supports
  the current diagnosis: high-PDMS buffer trajectories are useful as evidence,
  but short-lived low-weight regression does not reliably make the diffusion
  policy retain the behavior.
- The strongest surviving family remains GRPO/Safe-DiffGRPO-style policy
  gradient training. The next experiment should therefore improve the policy
  gradient machinery rather than add more offline regression.
- Updated reference check on 2026-06-15:
  - DPPO paper / official implementation: `arXiv:2409.00588`,
    `irom-princeton/dppo`. Mechanism: diffusion policies can be fine-tuned with
    policy-gradient/PPO-style updates and can benefit from structured
    on-manifold exploration.
  - DDPO paper / implementations: `arXiv:2305.13301`, `kvablack/ddpo-pytorch`.
    Mechanism: keep old log probabilities and use PPO-style clipped importance
    sampling for diffusion model rewards.
  - RIPT-VLA / VLA-RL references remain relevant as VLA post-training examples:
    RL fine-tuning should optimize sampled actions with explicit reward feedback,
    not only imitate static labels.

Implementation status:
- `scripts/training/launch_recogdrive_stage3_grpo_dppo_step_2b_local.sh`
  exists and passes `bash -n`.
- Initial dry-run config check passed on 2026-06-15 with:
  - `stage3_objective=grpo_replay`
  - `max_epochs=1`, `grpo_scheduler_epochs=1`, `grpo_scheduler_min_lr=1e-5`
  - local 8GPU target, `batch_size=2`, `accumulate_grad_batches=1`
  - `sample_time=16`
  - `grpo_use_gspo_ratio=true`
  - `grpo_normalize_advantage_batch=true`
  - `grpo_advantage_clip_abs=5.0`
  - `grpo_hard_gate_ttc=true`, `grpo_hard_gate_ddc=true`
  - `grpo_ppo_replay_logprob_mode=step`
  - `grpo_ppo_replay_step_minibatch_mode=transition`
  - `grpo_ppo_replay_step_clip_schedule=dppo_exp`
  - `grpo_ppo_replay_inner_epochs=1`
  - `grpo_ppo_replay_minibatch_size=128`
  - buffer guidance, buffer distill, Buffer-DPO, and self-imitation disabled.
- After the 20:24 UTC OOM, the launcher defaults were patched to the
  memory-safe settings that are actually running:
  - `batch_size=1`
  - `reference_kl_chunk_size=4`
  - `grpo_ppo_replay_minibatch_size=64`
  The old `batch_size=2`, unchunked reference-KL default should not be reused on
  80GB A800.

Launch rule:
- This is a controlled 1-epoch diagnostic, not a full 20-epoch run.
- Stop the local aux-decay run before launch, because its step1200 result shows
  the exact auxiliary-decay recipe should no longer consume the primary local
  8GPU budget.
- Keep zt2 clean GRPO running and keep its `step300` evaluation. Do not kill any
  unrelated remote task.
- Start local DPPO-step only after local target GPUs are genuinely free.

Expected diagnostics:
- `ppo_replay_loss_active > 0`
- finite `gspo_ratio_mean`, `gspo_ratio_clip_frac`, and `gspo_approx_kl`
- rollout/replay NC, DAC, TTC, DDC, EP logged
- `ppo_replay_valid_ratio` should stay above `0.8`
- first exact navtest step checkpoint should clear the `0.88` early-health gate
  unless diagnostics show a principled slow-start reason.

Promotion criteria:
- Do not promote to 20 epochs until the 1-epoch or relaxed-window diagnostic is
  competitive with matched clean-GRPO controls and does not regress DDC/TTC.
- If step-level replay is again far below clean GRPO, inspect whether the
  transition-level logprob/clip mapping is too aggressive before launching
  another replay variant.

Launch status:
- 2026-06-15 20:18 UTC: local aux-decay run was stopped after recording its
  `step1200` exact result. Only the run-specific local process tree was
  terminated; zt2 clean GRPO and its exact eval were not touched.
- 2026-06-15 20:19 UTC: launched
  `stage3_grpo_dppo_step_s16_i1_lr1e4_b2acc1_local8_1e_20260615T201945Z`.
- Run root:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b2acc1_local8_1e_20260615T201945Z`.
- Launcher log:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b2acc1_local8_1e_20260615T201945Z/launch.log`.
- The launcher confirmed target local GPUs were free and started training.
- zt2 evaluation watcher started at:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b2acc1_local8_1e_20260615T201945Z/unique_lock_watch_on_vla_zt2_4gpu`.
- The DPPO-step watcher is configured to wait for free zt2 GPU `0,1,2,3`; zt2
  clean GRPO continues training on GPU `4,5,6,7` and its `step300` exact eval is
  already running on GPU `0,1,2,3`. No remote task was killed.
- 2026-06-15 20:21 UTC first health check: local training processes are alive
  and all 8 local GPUs show initial model-loading allocation around `2.8GB`.
  Wait for the first TensorBoard scalars before judging training behavior.
- 2026-06-15 20:23 UTC health check: event file exists under
  `train/hydra/training_recogdrive_agent/2026.06.15.20.20.25/lightning_logs`;
  only `lr-AdamW=1e-4` is logged so far. All 8 local GPUs are allocated at about
  `9.5GB`, with low utilization, consistent with model/data loading before the
  first logged training batch. No traceback has appeared in the launch log.
- 2026-06-15 20:24 UTC result for that first launch: failed with CUDA OOM in
  `_chain_transition_reference_kl` / `_chain_transition_distribution` while
  computing the transition reference-KL path. The failed config had
  `batch_size=2`, `sample_time=16`, and `reference_kl_chunk_size=0`, so this is
  a memory-sizing/config failure, not an algorithm verdict.
- Relaunch rule learned here:
  - Do not wrap the launcher itself in an extra shell background job and then let
    the parent command exit immediately. The launcher must run in the foreground
    until it writes its own `launcher.pid`; after that the stable launcher has
    already `setsid`-detached the durable training process.
  - A zero-byte outer `launch.log` with only partial config files usually means
    the external wrapper was killed before the stable launcher detached. The
    correct evidence of a live run is `launcher.pid`, `status/*.json`, and the
    stable-launcher `logs/stage3_rl_2b.log`.
- 2026-06-15 20:32 UTC memory-safe relaunch:
  - Run:
    `stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_trace_20260615T203224Z`
  - Root:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_trace_20260615T203224Z`
  - Config changes from the OOM run: `batch_size=1`,
    `reference_kl_chunk_size=4`, and `grpo_ppo_replay_minibatch_size=64`.
    Kept `sample_time=16`, `stage3_objective=grpo_replay`, step-level
    transition replay, GSPO ratio, batch advantage normalization, and hard
    TTC/DDC safety gates.
  - The actual resolved command confirms `agent.stage3_objective=grpo_replay`,
    `agent.reference_kl_chunk_size=4`, `agent.grpo_sample_time=16`,
    `trainer.params.devices=8`, and `dataloader.params.batch_size=1`.
  - zt2 watcher started at
    `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_trace_20260615T203224Z/unique_lock_watch_on_vla_zt2_4gpu`,
    pid `3725772`, with `WAIT_FOR_FREE_GPUS=1` on GPU `0,1,2,3`. It will wait
    behind existing zt2 exact-eval work and does not kill or preempt any remote
    task.
- 2026-06-15 20:36 UTC health check for the memory-safe relaunch:
  - Training is alive and has entered the Lightning training loop. All local 8
    GPUs are allocated at about `66GB` and show nonzero utilization.
  - Logs show `Num training samples: 85109`, `Num validation samples: 18179`,
    `LOCAL_RANK: 0..7`, and model summary `Trainable params: 34.3M`.
  - No checkpoint has been written yet; zt2 watcher reports no pending
    checkpoint evals and `training_state=running`.
- 2026-06-15 20:40 UTC training-diagnostic check:
  - TensorBoard reached `step=49`, so the run is past initialization.
  - The DPPO/step-replay path is active:
    `ppo_replay_loss_active=1`, `ppo_replay_transition_mode=1`,
    `ppo_replay_optimizer_steps=3`, `ppo_replay_valid_ratio=1`,
    `ppo_replay_selected_transition_count=16`,
    `ppo_replay_step_clip_mean=0.012621`, and
    `ppo_replay_step_clip_max=0.05`.
  - GSPO/old-policy ratio diagnostics are finite:
    `gspo_ratio_mean=0.998787`, `gspo_ratio_clip_frac=0.343750`,
    `gspo_approx_kl=0.000009`, and `sampled_from_behavior_policy=1`.
  - Early rollout quality is weak but interpretable under the hard gate:
    shaped reward `0.418766`, base reward `0.667595`, safe ratio `0.664062`,
    NC `0.941406`, DAC `0.898438`, TTC `0.726562`, EP `0.731006`, comfort
    `0.492188`, DDC `0.984375`, TLC `1.0`.
  - Decision: continue to at least the first checkpoint/eval. This is the first
    run where the intended step-level replay machinery is demonstrably active,
    so do not stop based only on low step49 train reward.
  - Removed the two empty wrapper residue directories that never reached
    stable-launcher detachment:
    `stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_20260615T202836Z`
    and
    `stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_direct_20260615T203101Z`.
- 2026-06-15 20:46 UTC update:
  - TensorBoard has advanced to `step=99`.
  - Train diagnostics improved from step49 to step99:
    shaped reward `0.418766 -> 0.687514`, base reward `0.667595 -> 0.827579`,
    safe ratio `0.664062 -> 0.820312`, EP `0.731006 -> 0.849853`, TTC
    `0.726562 -> 0.851562`, and DDC `0.984375 -> 0.964844`.
  - Step-level replay remains active at step99:
    `ppo_replay_loss_active=1`, `gspo_ratio_mean=0.998790`, and
    `gspo_approx_kl=0.000013`.
  - The first checkpoint was written:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_trace_20260615T203224Z/train/hydra/training_recogdrive_agent/2026.06.15.20.33.04/step_checkpoints/step-step=300.ckpt`
  - zt2 watcher archived it as
    `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_trace_20260615T203224Z/unique_lock_watch_on_vla_zt2_4gpu/checkpoint_archive/step-step_300.ckpt`
    but is waiting for GPU `0,1,2,3`, which are still occupied by the clean-GRPO
    step300 exact eval. This is the intended non-preemptive behavior.
- 2026-06-15 20:52 UTC update:
  - Local DPPO-step training is not stalled. TensorBoard advanced to `step=199`
    and the run wrote a second checkpoint:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_step_s16_i1_lr1e4_b1acc1_local8_1e_chunk4_trace_20260615T203224Z/train/hydra/training_recogdrive_agent/2026.06.15.20.33.04/step_checkpoints/step-step=600.ckpt`
  - Recent train diagnostics at step199 are in a usable diagnostic band:
    shaped reward `0.733392`, base reward `0.819329`, safe ratio `0.843750`,
    EP `0.814735`, TTC `0.867188`, DDC `1.000000`, `ppo_replay_loss_active=1`,
    `ppo_replay_optimizer_steps=3`, `gspo_ratio_mean=1.000771`, and
    `gspo_approx_kl=0.000004`.
  - The zt2 watcher has archived only `step-step_300` so far and is still
    waiting on GPU `0,1,2,3` because the clean-GRPO `step300` exact eval is
    still running there. No DPPO navtest PDMS exists yet.
  - Fairness note: this memory-safe diagnostic uses effective batch
    `8 * 1 * 1 = 8`, not the `64` used by the b2/acc4/8GPU controls. Therefore
    DPPO `step300/600` are much earlier in train-scene exposure than clean-GRPO
    `step300/600`; they should only be treated as catastrophe checks. The first
    roughly matched exposure point is about `step2400`.
  - Launcher/watch infrastructure was patched after this observation:
    `watch_stage3_checkpoints_eval_8gpu.sh` now supports
    `EVAL_MIN_CHECKPOINT_STEP`, `EVAL_CHECKPOINT_STEP_INTERVAL`, and
    `EVAL_ALWAYS_EPOCH_CHECKPOINTS`; the DPPO-step launcher defaults to
    `CHECKPOINT_EVERY_N_TRAIN_STEPS=2400`, `EVAL_MIN_CHECKPOINT_STEP=2400`,
    and `EVAL_CHECKPOINT_STEP_INTERVAL=2400`. This only changes checkpoint
    emission / checkpoint selection for evaluation. It does not change
    inference, PDM scoring, or any submetric calculation. The already running
    watcher keeps its original queue to avoid disturbing active zt2 work.
  - A local smoke check with a fake `step-step=300.ckpt` and
    `EVAL_MIN_CHECKPOINT_STEP=2400`, `EVAL_CHECKPOINT_STEP_INTERVAL=2400`
    produced `step-step_300.filter_skipped` and exited with an empty eval queue,
    confirming the filter works without invoking the evaluator.
  - The already-running DPPO watcher was launched before this filter existed
    and had already archived `step-step_300` and `step-step_600`. Because both
    are low-exposure checkpoints under effective batch `8`, they were manually
    marked as skipped in the watcher state (`*.filter_skipped` and
    `*.external_skipped`) before any DPPO eval had started. This keeps zt2 exact
    eval capacity for the clean-GRPO active eval and later exposure-aligned DPPO
    checkpoints instead of spending it on unfair early points.
  - After confirming the watcher reported `no pending checkpoint evals`, skip
    markers were also pre-created for future low-exposure step checkpoints up
    to step12000. Multiples of `2400` are intentionally kept (`2400`, `4800`,
    `7200`, `9600`, `12000`) because they are the approximate exposure-aligned
    comparison points for this effective-batch-8 diagnostic.
- 2026-06-15 21:15 UTC update:
  - The memory-safe local DPPO-step run has written `step-step=300/600/900/1200`
    checkpoints and remains alive. The active zt2 watcher archived through
    `step-step_1200`, but all four checkpoints are below the exposure-aligned
    `step2400` point for effective batch `8`, so they are intentionally skipped
    rather than evaluated. No DPPO navtest PDMS exists yet.
- 2026-06-15 21:19 UTC update:
  - The local DPPO-step run has advanced to TensorBoard `step=499` and written
    `step-step=1500.ckpt`. This is still below the exposure-aligned first eval
    point: effective batch is `8`, so `step1500` is only about `12000` seen
    scenes versus about `19200` for b2/acc4/8GPU `step300` and about `85120` for
    epoch0-step1330.
  - Latest train diagnostics: reward `0.734198`, base reward `0.823844`, safe
    ratio `0.867188`, EP `0.827878`, TTC `0.875000`, group reward std
    `0.367868`, mixed-group ratio `0.625`, all-unsafe-group ratio `0.0`,
    GSPO ratio mean `0.999755`, and clip fraction `0.171875`. There were weak
    batches at steps 349 and 399, so this remains a high-variance diagnostic.
  - Keep waiting for `step2400` before spending zt2 exact-eval capacity on this
    run. Estimated ETA for step2400 from the monitor was around
    2026-06-16 00:05 UTC.
- 2026-06-15 21:23 UTC update:
  - The run advanced to TensorBoard `step=549`; watcher archived
    `step-step_1500` and still has no pending eval because the low-exposure
    checkpoints are intentionally skipped.
  - Latest batch is weak again: reward `0.416382`, base reward `0.662061`,
    safe ratio `0.625000`, EP `0.689565`, TTC `0.765625`, mixed-group ratio
    `1.0`, group reward std `0.568919`, GSPO ratio mean `1.000096`, and clip
    fraction `0.156250`. This confirms the DPPO-step diagnostic is high
    variance, not yet a promoted method.
  - Do not stop or promote from this batch-level signal alone. The intended
    test remains the first exposure-aligned exact eval at `step2400`.
- 2026-06-15 21:30 UTC update:
  - Local 8GPU DPPO-step training remains active and all local GPUs are fully
    utilized. TensorBoard reached `step=599`, and the training directory has
    checkpoints through `step-step=1800.ckpt`.
  - The zt2 watcher archived `step-step_1800` and intentionally marks
    non-aligned low-exposure checkpoints as `filter_skipped` /
    `external_skipped`. There is no `step-step_2400.filter_skipped` marker, so
    the first exposure-aligned DPPO eval point is still preserved.
  - Latest batch recovered from the weak step549 sample: reward `0.842514`,
    base reward `0.885694`, safe ratio `0.914062`, EP `0.898649`, TTC
    `0.914062`, group reward std `0.274990`, mixed-group ratio `0.375`,
    all-safe-group ratio `0.625`, GSPO ratio mean `0.999296`, and clip fraction
    `0.078125`.
  - The train-batch variance remains high, so the decision is unchanged: wait
    for the exact navtest PDMS at `step2400` before judging this DPPO-step
    implementation.

### 2026-06-15 21:30 UTC Active Resource Snapshot

- Local machine: all 8 GPUs are occupied by the local DPPO-step diagnostic with
  high utilization. Do not launch another local training/eval job until this
  reaches the planned `step2400` evaluation or is explicitly stopped.
- zt2 GPUs `0-3`: occupied by the keep-best train-only buffer generator and an
  existing stage1 job. Do not kill either; the buffer cycle continues to update
  shard builder logs.
- zt2 GPUs `4-7`: occupied by the clean GRPO b2acc8 e4-gate run. This run has
  one exact navtest result so far, `step-step_300` PDMS `0.887157`, and remains
  the active clean comparison stream.
- No new navtest PDMS row was available at this snapshot beyond the previously
  recorded zt2 clean-GRPO `step300` result. The next expected hard evidence is
  clean GRPO `step600` and DPPO-step `step2400`.
- 2026-06-15 21:32 UTC follow-up:
  - No new `checkpoint_eval_submetrics.tsv` row appeared after the zt2 clean
    GRPO `step300` result, so the relaxed audit was not rerun.
  - Local DPPO-step TensorBoard advanced to `step=649`; this latest batch was
    weak again with reward `0.269551`, base reward `0.633693`, safe ratio
    `0.523438`, EP `0.716048`, TTC `0.585938`, all-unsafe-group ratio
    `0.125`, and clip fraction `0.328125`. This reinforces that batch-level
    DPPO telemetry is high variance; do not judge before the planned
    `step2400` exact eval.
  - zt2 keep-best train-only buffer cycle2 is still active. All four shards
    reported `6400` processed scenes at about `21:28 UTC`, so GPUs `0-3` remain
    productively occupied and should not be preempted.
  - zt2 clean GRPO on GPUs `4-7` remains active by `nvidia-smi pmon`, but its
    latest TensorBoard scalar is still `step=449`; given the observed cadence
    and 50-step scalar logging, this is not yet evidence of a stall.
- 2026-06-15 21:36 UTC follow-up:
  - Local DPPO-step wrote `step-step=2100.ckpt`. The watcher already had
    `step-step_2100.filter_skipped` and `step-step_2100.external_skipped`, so
    no exact eval was launched for this still-low-exposure checkpoint.
  - This confirms the checkpoint filter is protecting zt2 eval capacity for the
    intended first exposure-aligned DPPO point, `step2400`.
- 2026-06-15 21:37 UTC follow-up:
  - zt2 clean GRPO TensorBoard advanced from `step=449` to `step=499`, proving
    the earlier long scalar gap was not a training stall. The step499 batch was
    healthy: reward `0.872473`, base reward `0.840593`, safe ratio `0.976562`,
    EP `0.871976`, TTC `0.867188`, all-unsafe-group ratio `0.0`, and group
    reward std `0.200877`.
  - Local DPPO-step TensorBoard advanced to `step=699`; after the weak step649
    batch it recovered to reward `0.866566`, base reward `0.882299`, safe ratio
    `0.937500`, EP `0.920960`, TTC `0.937500`, and all-unsafe-group ratio
    `0.0`. This remains batch telemetry only; the first meaningful DPPO verdict
    is still `step2400`.
  - No new exact navtest PDMS row exists at this checkpoint. The current
    expected ETAs from the monitors are clean GRPO `step600` around
    `2026-06-15T22:21Z` and DPPO-step `step2400` around
    `2026-06-16T00:06Z`.
- 2026-06-15 21:38 UTC follow-up:
  - DPPO-step `step-step=2100.ckpt` has now been archived by the zt2 watcher,
    but it remains skipped for exact eval via the pre-existing
    `filter_skipped` / `external_skipped` markers. No DPPO navtest row exists.
  - zt2 keep-best train-only buffer cycle2 remains active; all four shards
    reached `6656` processed scenes at about `21:38 UTC`.
  - No new clean-GRPO or DPPO exact PDMS row exists, so the relaxed audit is
    still the `21:28 UTC` refresh.
- 2026-06-15 21:40 UTC follow-up:
  - zt2 clean GRPO remains at latest logged scalar `step=499` and has not yet
    written `step-step=600.ckpt`. The monitor still estimates `step600` around
    `2026-06-15T22:24Z`.
  - Local DPPO-step advanced to TensorBoard `step=749`. The latest batch is
    weak again, with reward `0.425810`, base reward `0.635091`, safe ratio
    `0.664062`, EP `0.658561`, TTC `0.796875`, mixed-group ratio `1.0`, and
    group reward std `0.571489`. This does not change the decision rule: wait
    for `step2400` exact eval rather than judging from noisy train batches.
  - No new exact PDMS row exists; no relaxed audit rerun was needed.
- 2026-06-15 21:46 UTC follow-up:
  - Local DPPO-step wrote and the zt2 watcher archived
    `step-step=2400.ckpt`. This is the first planned exposure-aligned DPPO
    exact-eval checkpoint under effective batch `8`; lower-exposure
    checkpoints through `step2100` were intentionally skipped.
  - The DPPO watcher is now correctly pending one eval and waiting for free zt2
    GPUs. zt2 GPUs `0-3` are busy with the keep-best buffer generator and an
    existing stage1 job; zt2 GPUs `4-7` are busy with clean GRPO. Do not kill
    these remote jobs just to force the eval.
  - Local DPPO TensorBoard advanced to `step=799`; the latest batch remains
    high-variance and weak (`reward=0.494979`, `safe_ratio=0.687500`,
    `mean_ep=0.758512`, `mean_ttc=0.757812`, `mixed_group_ratio=1.0`,
    `gspo_ratio_clip_frac=0.281250`). This still should not be used as the
    verdict; wait for the exact `step2400` navtest PDMS.
  - zt2 clean GRPO is still alive with latest scalar `step=499`; no
    `step-step=600.ckpt` exists yet. Its last healthy batch remains
    `reward=0.872473`, `safe_ratio=0.976562`, `mean_ep=0.871976`, and
    `mean_ttc=0.867188`.
  - zt2 keep-best train-only buffer cycle2 continues productively; all four
    shards reported `6656` processed scenes at about `21:38 UTC`.
  - No new exact PDMS row appeared after the 21:12 UTC eval files, so the
    relaxed audit remains the 21:44 UTC refresh.
- 2026-06-15 21:55 UTC resource action:
  - Stopped the local DPPO-step training process after preserving
    `step-step=2400.ckpt` and the zt2 watcher archive. This does not cancel the
    pending exact navtest eval; the zt2 watcher remains queued on
    `step-step_2400` and is waiting for free zt2 GPUs.
  - Rationale: `step2400` is the first planned exposure-aligned diagnostic
    checkpoint for this effective-batch-8 run. Continuing local DPPO training
    before the exact `step2400` result would consume all local GPUs and create
    more queued checkpoints without changing the immediate decision gate. If
    `step2400` is strong, the run can be resumed from the saved checkpoint.
  - Launched the next local 8GPU candidate:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z`.
    It uses clean GRPO/reference KL as the main objective, enables only the
    small train-only buffer reward-neighborhood bonus, and keeps buffer
    distillation, Buffer-DPO, and self-imitation disabled. The zt2 watcher is
    active but configured to wait for free GPUs.
  - Verified launch config:
    `grpo_buffer_guidance_enabled=true`,
    `grpo_buffer_reward_bonus_weight=0.01`,
    `grpo_buffer_distill_loss_weight=0.0`,
    `grpo_buffer_preference_dpo_loss_weight=0.0`,
    `grpo_self_imitation_loss_weight=0.0`,
    `offline_rl_cache_elite_records_in_memory=true`,
    `EVAL_MIN_CHECKPOINT_STEP=600`, and
    `EVAL_CHECKPOINT_STEP_INTERVAL=600`.
    This was the active-run launch config. After the relaxed-horizon audit, the
    launcher default was changed to `300/300` so future buffer-bonus runs do not
    skip the step300 health eval.
  - At launch verification, the run had spawned the 8 local DDP ranks and was
    still in model/dataloader initialization. No TensorBoard event or
    checkpoint existed yet.
  - 2026-06-15 21:59 UTC follow-up: local buffer-bonus-only training is now
    genuinely using the local 8 GPUs, with about `39.6GB` allocated per GPU and
    SM utilization roughly `45-97%` across ranks. TensorBoard event creation is
    confirmed under
    `train/hydra/training_recogdrive_agent/2026.06.15.21.55.46/lightning_logs/version_0/`;
    only the initial `lr-AdamW=1e-4` scalar is present so far, so wait for the
    first GRPO metrics at the normal logging cadence before interpreting train
    diagnostics.
  - DPPO-step also wrote `step-step=2700.ckpt` just before the local stop. The
    zt2 watcher archived it but also wrote `step-step_2700.filter_skipped` and
    `step-step_2700.external_skipped`, so it is not part of the eval queue. The
    planned first diagnostic remains the single pending `step-step_2400` exact
    eval.
  - 2026-06-15 22:01 UTC follow-up:
    - No new exact PDMS TSV exists after the 21:12 UTC eval files, so the
      relaxed audit does not need another refresh yet.
    - zt2 clean GRPO advanced to TensorBoard `step=549`; latest batch remains
      healthy enough to continue: reward `0.821177`, base reward `0.814160`,
      safe ratio `0.929688`, EP `0.843448`, TTC `0.835938`, group reward std
      `0.221174`, all-safe-group ratio `0.75`, and all-unsafe-group ratio
      `0.0`. It has not written `step600` yet; monitor ETA is around
      `2026-06-15T22:24Z`.
    - zt2 keep-best buffer cycle2 remains active and productive; all four
      shards reached `7168` processed scenes at about `21:58 UTC`.
    - Local buffer-bonus-only run completed dataset/trainer initialization and
      started training with all 8 ranks. The run is using about `39.6GB/GPU`
      with active SM utilization. First GRPO metrics are not logged yet; only
      the initial LR scalar exists, so wait for the normal logging cadence
      before interpreting train diagnostics.
    - DPPO `step2400` exact eval remains queued on zt2 and is waiting for free
      GPUs. No remote training/eval job was killed.
  - 2026-06-15 22:08 UTC follow-up:
    - Still no new exact PDMS TSV after the 21:12 UTC eval files. The relaxed
      audit was rerun at 22:06 UTC with `--max-epoch 4 --max-step 5000` and
      `--max-epoch 4 --max-step 5500`; both scans still find `94` raw eval
      rows, `42` grouped checkpoint rows, and `33` rows inside the relaxed
      window, with unchanged top ordering.
    - Local buffer-bonus-only GRPO is alive on all 8 local GPUs. First GRPO
      TensorBoard metrics at `step=49` confirm the buffer path is active:
      `grpo_buffer_guidance_enabled=1`,
      `grpo_buffer_reward_bonus=0.128486`,
      `grpo_buffer_reward_bonus_weight=0.01`,
      `grpo_buffer_guidance_target_ratio=0.5625`,
      `grpo_buffer_selected_valid_ratio=1.0`, and
      `grpo_buffer_best_valid_minus_gt=0.063061`.
    - The same `step=49` batch has `reward=0.768401`,
      `base_reward=0.778053`, safe ratio `0.902344`, EP `0.759799`, TTC
      `0.898438`, DAC `0.917969`, and DDC `0.949219`. The positive buffer
      bonus contributes only about `+0.001285` to reward at the current weight,
      while the hard-safety reward composition can still reduce reward below
      raw PDMS on unsafe samples. Therefore this is not evidence that the
      buffer failed to activate; it is only an early batch health signal. Wait
      for exact `step600` PDMS before judging this run.
    - zt2 clean GRPO remains the clean comparison spine. Its latest scalar is
      still `step=549` with reward `0.821177`, base reward `0.814160`, safe
      ratio `0.929688`, EP `0.843448`, TTC `0.835938`, DAC `0.960938`, DDC
      `1.0`, and no buffer guidance active. It has not emitted `step600` yet.
    - DPPO-step training was already manually stopped after preserving
      `step2400`; its zt2 watcher remains queued on the exact `step2400` eval
      and is waiting for free zt2 GPUs. Do not kill zt2 jobs to force it.
    - zt2 keep-best buffer cycle2 is still productive: all four shards reached
      `7424` processed scenes at about `22:08 UTC`.
  - 2026-06-15 22:15 UTC follow-up:
    - No new exact PDMS TSV and no new `step-step=300/600.ckpt` appeared after
      the 22:08 check. Local buffer-bonus-only GRPO, zt2 clean GRPO, and zt2
      keep-best buffer generation are still using their assigned GPUs; no
      remote task was killed or preempted.
    - The active buffer-bonus-only run was launched with watcher filters
      `EVAL_MIN_CHECKPOINT_STEP=600` and `EVAL_CHECKPOINT_STEP_INTERVAL=600`.
      This would skip the `step300` launch-health eval. The launcher default
      has now been corrected for future runs to `300/300`, matching the revised
      relaxed-horizon protocol where `step300` is a health check rather than an
      algorithm verdict.
    - Verification:
      `RUN_TRAIN=0 OUT_ROOT=/tmp/recogdrive_buffer_bonus_dryrun_20260615T221522Z bash scripts/training/launch_recogdrive_stage3_grpo_buffer_bonus_only_2b_local.sh`
      produced `strict_gspo_launch_config.txt` with
      `grpo_buffer_reward_bonus_weight=0.01`,
      `eval_min_checkpoint_step=300`, and
      `eval_checkpoint_step_interval=300`.
  - Operational consequence: when the active buffer-bonus run writes
      `step-step=300.ckpt`, check whether the already-running watcher filters
      it out. If it does, run a one-off exact navtest eval for that checkpoint
      without changing training or preempting zt2 tasks.
  - 2026-06-15 22:21 UTC follow-up:
    - A supplemental zt2 watcher was launched for the active buffer-bonus run
      under
      `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/step300_watch_on_vla_zt2_4gpu`.
    - It uses the same
      `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/global_checkpoint_eval_locks`
      as the main watcher, so it should not duplicate later evaluations.
    - Verified watcher config: `eval_min_checkpoint_step=300`,
      `eval_checkpoint_step_interval=300`, `wait_for_free_gpus=1`,
      `gpu_max_mem_used_mb=2000`, and `gpu_max_util=5`. zt2 was fully busy at
      launch, so this watcher waits for resources instead of preempting other
      tasks.
  - 2026-06-15 22:23 UTC follow-up:
    - zt2 clean GRPO wrote
      `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_clean_s16_lr1e4_b2acc8_zt2_4gpu_e4gate_20260615T175659Z/train/hydra/training_recogdrive_agent/2026.06.15.17.57.56/step_checkpoints/step-step=600.ckpt`.
    - Its watcher archived the checkpoint and started exact navtest eval under
      `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_clean_s16_lr1e4_b2acc8_zt2_4gpu_e4gate_20260615T175659Z/unique_lock_watch_on_vla_zt2_4gpu/eval_step-step_600`.
    - The eval loaded the checkpoint with `missing keys: 0` and four rank
      processes are active. No step600 PDMS row is available yet.
    - Local buffer-bonus-only still has no checkpoint; TensorBoard remains at
      step99, but all 8 local GPUs are active and the run log shows no error.
  - 2026-06-15 22:25 UTC follow-up:
    - No new complete `checkpoint_eval_submetrics.tsv` has appeared after the
      21:12 UTC eval files.
    - zt2 clean GRPO `step-step_600` exact eval is actively processing scenes;
      rank0 reported `100 / 3035` scenarios at `22:25:37 UTC`.
    - Local buffer-bonus-only advanced to TensorBoard step149. Buffer guidance
      remains active: `grpo_buffer_guidance_enabled=1`,
      `grpo_buffer_reward_bonus=0.118365`,
      `grpo_buffer_guidance_target_ratio=0.5`,
      `grpo_buffer_selected_valid_ratio=0.99375`, and
      `grpo_buffer_best_valid_minus_gt=0.050954`.
    - The same step149 batch has reward `0.736047`, base reward `0.751907`,
      safe ratio `0.890625`, EP `0.754413`, TTC `0.847656`, and no all-unsafe
      groups. This is still only train-batch telemetry; the first hard
      checkpoint remains `step300`.
  - 2026-06-15 22:31 UTC follow-up:
    - zt2 clean GRPO `step-step_600` exact eval continues to make progress;
      rank0 reached `400 / 3035` scenarios. No `done` row or step600 submetric
      row has been written yet.
    - Local buffer-bonus-only still has no `step300` checkpoint. The
      supplemental step300 watcher remains alive and reports no pending
      checkpoint evals.
  - 2026-06-15 22:36 UTC follow-up:
    - zt2 clean GRPO `step-step_600` exact eval reached rank0 `700 / 3035`.
      The summary still has only `step-step_600 started`; no step600 PDMS row
      has been written yet.
    - Local buffer-bonus-only advanced to TensorBoard step199, so the run is
      not merely holding GPU memory. Buffer guidance remains active:
      `grpo_buffer_reward_bonus=0.094319`,
      `grpo_buffer_guidance_target_ratio=0.5`,
      `grpo_buffer_selected_valid_ratio=1.0`, and
      `grpo_buffer_best_valid_minus_gt=0.041603`.
    - The step199 train batch is weaker on safety/progress than step149:
      reward `0.657806`, base reward `0.701031`, safe ratio `0.847656`, EP
      `0.736331`, TTC `0.757812`, and all-unsafe-group ratio `0.0`. Treat this
      as noisy batch telemetry, not a stop signal; the first hard decision
      remains exact `step300` PDMS.
  - 2026-06-15 22:40 UTC follow-up:
    - zt2 clean GRPO `step-step_600` exact eval reached rank0 `1000 / 3035`.
      The eval is still running; there is no step600 `done` row or submetric
      row yet.
    - Local buffer-bonus-only still has no `step300` checkpoint and TensorBoard
      remains at step199. The step300 watcher is alive and reports no pending
      checkpoint evals.
  - 2026-06-15 22:55 UTC follow-up:
    - Local buffer-bonus-only wrote
      `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/train/hydra/training_recogdrive_agent/2026.06.15.21.55.46/step_checkpoints/step-step=300.ckpt`
      at `22:52:35 UTC`, size `4637966958` bytes.
    - The supplemental zt2 step300 watcher archived it by hardlink as
      `step300_watch_on_vla_zt2_4gpu/checkpoint_archive/step-step_300.ckpt`.
      It has not started exact navtest eval yet because zt2 GPUs `4-7` are
      still occupied by the clean GRPO `step600` exact eval, so the watcher is
      correctly waiting for free GPUs rather than launching a duplicate or
      preempting another task.
    - TensorBoard step299 confirms the intended buffer-bonus-only objective:
      `grpo_buffer_guidance_enabled=1`,
      `grpo_buffer_reward_bonus_weight=0.01`,
      `grpo_buffer_distill_loss_weight=0`,
      `grpo_buffer_preference_dpo_loss_weight=0`, and
      self-imitation disabled. The same batch had reward `0.830390`, base
      reward `0.820392`, safe ratio `0.937500`, EP `0.800526`, TTC
      `0.914062`, and no all-unsafe groups. This remains train-batch telemetry
      only; exact navtest PDMS is still pending.
    - zt2 clean GRPO `step-step_600` exact eval reached rank0 `2000 / 3035` at
      `22:55:13 UTC`. No step600 `done` row or submetric row exists yet, so the
      relaxed audit should not be rerun until a new exact PDMS row is written.

Cleanup on 2026-06-14 UTC:
- Freed about `300+ GB` under `/mnt/project/VLA-AD/outputs`, increasing available `/mnt/project` space from about `93 GB` to over `400 GB`.
- Removed obsolete AWAC/IQL outputs whose results are already listed in the completed AWAC ledger: `stage3_awac_iql_dualhost_20260612T182947Z`, `stage3_awac_iql_bc015_20260613T0648Z`, `stage3_awac_iql_pref_rank_ttc_20260613T112709Z`, `stage3_awac_warmup_absddc_blend025_lownoise_online_20260613T183113Z`, `stage3_awac_lownoise_dpo3gpu_20260613T163215Z`, `stage3_awac_blend035_lownoise_dpo8gpu_20260613T164458Z`, `stage3_awac_gapdpo_hybrid4gpu_v2_20260613T155350Z`, and `stage3_awac_gapdpo_lowvalid_2gpu_20260613T155703Z`.
- Removed obsolete DPPO/replay/high-LR diagnostic outputs whose conclusions are already recorded: `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z`, `stage3_dppo_transition_smoke2_20260614T082252Z`, `stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z`, `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`, `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`, `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`, `stage3_grpo_replay_stepmode_smoke_1gpu_20260614T0530`, `stage3_grpo_replay_smoke_1gpu_nullclip_20260614T041722Z`, `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z`, `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z`, `stage3_grpo_rloo_selfimit_smoke_20260614T_check`, and `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z`.
- Preserved the then-current safetygate/main-GRPO successor chain, the zt3 original-LR GRPO control, the zt3 keep-best train-only buffer builder, and the cap05 summaries/checkpoints referenced by the keep-best buffer process.

Second cleanup on 2026-06-14 UTC:
- User approved deleting obsolete results/logs once the failed attempt is recorded here.
- Deleted remaining local dry-run/profile/benchmark residues and low-information launch leftovers, including `exact_pdm_profile_awac_epoch0_32`, `exact_pdm_profile_awac_epoch0_32_retry`, `pdm_exact_benchmarks`, `pdm_bench_exact`, `stage3_rl_dry_run_check`, `stage3_rl_safe_diffgrpo_dry_run_check`, `stage3_safe_diffgrpo_eval_dry_run_check`, and `stage3_rl_2b_safe_diffgrpo_relaunch_dry_run`.
- Deleted failed or superseded Stage3 diagnostic residues whose conclusions are already captured above: `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z`, `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step300_20260614T0852Z`, `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step300_20260614T0857Z`, `stage3_grpo_dppo_transition_step600_zt3_eval_root.txt`, `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042322Z`, `stage3_grpo_refkl_s16_lr2e4_8gpu_20260613T204941Z`, `stage3_grpo_refkl_s16_lr2e4_b4acc2_8gpu_20260613T205949Z`, `stage3_grpo_refkl_s16_lr2e4_b8_chunk32_fast_8gpu_20260613T212548Z`, `stage3_rl_2b_safe_diffgrpo_online_20260613T221859Z`, and `stage3_rl_2b_safe_diffgrpo_online_20260613T224055Z`.
- Deleted `/mnt/project/VLA-AD/outputs/hidden_cache_equiv_navtest2`, which was previously inventoried as old smoke/equivalence output and is not a dependency of current Stage3 training, current Stage3 evaluation, the train-only elite buffer, or the running two-expert job.
- Deleted stale local Stage3 supervisor test logs and status files after confirming the recorded supervisor pid was no longer alive. The auto-supervisor remains disabled by policy because it can relaunch stale actions without the reference-audit protocol.
- Preserved active or decision-critical artifacts: the current main hard-gate v3 GRPO run, the zt3 original-LR GRPO control, the zt3 keep-best train-only elite-buffer generator, the Safe DiffGRPO best-result directory, and the cap05 run summaries/checkpoints used for diagnosis.
- Do not repeat the deleted attempts as default routes. Re-opening any of them requires a fresh planned-attempt entry with a reference audit, a mature implementation checklist, matched-step controls, and explicit NC/DAC/TTC/DDC diagnostics.

Third cleanup on 2026-06-14 UTC:
- User approved deleting obsolete failed-attempt results/logs after recording the conclusion in this ledger.
- Deleted redundant cap05 navtest watcher artifacts, including `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/manual_retry_step300_zt3_2gpu_20260614T130727Z`, `secondary_watch_on_rl_zt3_memfit_4gpu`, and `unique_lock_watch_on_vla_zt2_4gpu`. Their step300/600/900 PDMS and submetric conclusions are preserved in this document and in the compact top-level cap05 analysis files.
- Preserved the cap05 training `step_checkpoints` directory because the zt3 keep-best train-only buffer generator currently references it as an automatic policy-checkpoint source.
- Do not repeat the cap05 broad self-imitation configuration as a default route. Its failure mode is already diagnosed: it can increase EP/high-score bins, but it does not reliably improve mean PDMS because NC/TTC/DDC regressions offset the progress gain. Any future self-imitation run must include main-objective safety gating or a stronger preference objective, and must be judged at matched step/epoch against original Stage3 early PDMS.

Fourth cleanup on 2026-06-14 UTC:
- Stopped the local orphaned v3 `torchrun`/worker process tree after the stable launcher exited but the workers kept 8 GPUs allocated. Confirmed local GPU memory returned to `0 MB` on all 8 cards.
- Deleted the failed v3 run directory `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` after verifying no corresponding watcher was still active on `training-vla-zt2` or `training-rl-zt3`.
- The retained evidence for this failed direction is the row/result summary in this ledger: step300 PDMS `0.879395`, step600 PDMS `0.882934`, and the failure mode that main TTC/DDC hard gates improved safety but suppressed EP/DAC enough to underperform cap05 and original-LR GRPO.
- Do not repeat the v3 main TTC/DDC hard-gate plus strict self-imitation setup as a default route. Any future safety/progress method must preserve progress pressure and be compared against the zt3 original-LR control at matched step/epoch.

Fifth cleanup on 2026-06-14 UTC:
- Removed the final failed-v3 residue after it was recreated by a stale local `watch_recogdrive_stage3_early_gate.sh` process. The stale watcher PID `3706056` had `RUN_ROOT` set to `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` and only wrote `early_gate_latest.json` / `early_gate_watch.log`; it was stopped and the regenerated 12 KB directory was deleted.
- No remote `training-vla-zt2` or `training-rl-zt3` tasks were killed during this cleanup. The only remaining local early-gate watcher is attached to the active current-repo original-LR GRPO control run.
- Keep only this ledger entry as evidence for the failed v3 attempt. Do not restore or re-evaluate deleted v3 logs unless a new planned-attempt entry justifies a different safety/progress design.

Sixth cleanup on 2026-06-14 UTC:
- User approved deleting obsolete results/logs after recording failed or superseded attempts in this ledger.
- Deleted about `46 GB` of old Last-VLA / Stage2 result and eval artifacts under `/mnt/project/VLA-AD/outputs/last_vla_v2`, including obsolete `highcap_no_risk`, `decoupled_highcap_no_risk`, `direct_text_navtest`, `no_residual_stage2_ab`, `vlm_text_residual_anchor`, and `residual_anchor_*` directories. These were superseded by the current two-expert random-HMEF work and are not part of the ReCogDrive Stage3 GRPO / Buffer-DPO plan.
- Preserved `/mnt/project/VLA-AD/outputs/last_vla_v2/navtrain_pdms_val6000_probe_20260611T0315Z`, because the active two-expert watcher still references its token file. Also preserved small audit/VQA residues that are not meaningful space consumers.
- Deleted obsolete Last-VLA hidden-cache subtrees under `/mnt/project/VLA-AD/cache/last_vla_v2`: `highcap_no_risk`, `experiments`, and `decoupled_highcap_no_risk`. No active process referenced this cache. Preserved only `teacher_traj` and `manifests`.
- Result: `/mnt/project` available space increased from about `2.4 TB` to about `4.4 TB`; `/mnt/project/VLA-AD/cache/last_vla_v2` dropped to `24 KB`.
- No active Stage3 training/evaluation, remote zt2/zt3 task, train-only elite buffer, metric cache, or active two-expert run was killed or deleted.
- Do not restore or relaunch the deleted Last-VLA / Stage2 intermediate routes as default experiments. Reusing any of those directions requires a new planned-attempt entry with a current reference audit, matched-step controls, and explicit navtrain/navtest split handling.

Seventh cleanup on 2026-06-14 UTC:
- User approved deleting obsolete results/logs after recording failed attempts here.
- Current-repo original-LR GRPO control step300 exact navtest was successfully evaluated by the local foreground exact-pool path, not by the failed background wrapper attempts:
  - PDMS `0.881910`, NC `0.984800`, DAC `0.969682`, TTC `0.957571`, EP `0.805231`, comfort `0.999918`, DDC `0.965316`.
  - Result root kept: `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z/local_foreground_eval_step300_4gpu`.
- The following local eval-launch residues are obsolete and should not be repeated as default evaluation routes: `local_relaxed_watch_shared_4gpu`, `local_direct_eval_step300_shared_4gpu`, `local_direct_eval_step300_flock_4gpu`, `local_direct_eval_step300_execflock_4gpu`, and `local_direct_eval_step300_nolock_4gpu`.
- Failure modes: SIGHUP-prone background wrapper, launcher-wrapper mismatch with eval scripts that end in `exec`, and duplicate hardlinked checkpoint archives. The correct path for this run is either the existing checkpoint watcher with global done markers, or a directly supervised foreground/nohup exact eval whose result directory is retained.
- Updated the global step300 done marker so its archive path points to the real training checkpoint instead of the soon-deleted failed wrapper archive.
- Removed stale cap05 runtime-status residue (`pids`, `status`, and old pid files) after verifying no cap05 training process is alive. Preserved its `train/step_checkpoints` directory because the zt3 keep-best train-only buffer generator still references it, and preserved compact cap05 navtest analysis files.
- Deleted the failed old Safe DiffGRPO relaunch residue `stage3_rl_2b_safe_diffgrpo_online_relaunch_20260609T1944Z` and stale local supervisor latest log/status files. Preserved the actual Safe DiffGRPO historical-best directory `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z`.
- Preserved active training, zt2/zt3 watcher directories, successful foreground eval outputs, the train-only elite buffer, and all strong baseline summaries. No remote task was killed.

Eighth cleanup on 2026-06-14 UTC:
- User approved deleting outdated results/logs and keeping only the summary evidence for failed attempts.
- Deleted small, no-dependency residues:
  - `/mnt/project/VLA-AD/outputs/last_vla_v2/vqa_smoke_20260608T142011Z`
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z_navtest_exact_eval`
- Did not delete `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`, even though broad cap05 self-imitation is not a default route, because the active zt3 keep-best train-only buffer generator still references its `step_checkpoints` directory as `AUTO_POLICY_CHECKPOINT_DIR`.
- Did not delete current-repo GRPO control archives. The apparent duplicate `step-step_300` and `step-step_600` checkpoint files under the train dir, zt2 watcher archive, and zt3 watcher archive share the same inodes, so removing watcher hardlinks would not materially free space and could confuse active checkpoint watchers.
- Did not delete Safe DiffGRPO historical-best artifacts. The archive contains only the retained `epoch_12-step_17290.ckpt` best checkpoint plus compact eval summaries.
- Current cleanup conclusion: there is no large Stage3 artifact that is both obsolete and safely deletable without breaking an active dependency or a decision-critical baseline. Future cleanup should target completed Buffer-DPO/control runs after their PDMS/submetric evidence is recorded here.
- Do not repeat the deleted VQA smoke / empty zt3 wrapper paths as evaluation routes. They contain no Stage3 algorithm evidence beyond what is already summarized in this document.

Ninth cleanup / resource rule on 2026-06-15 UTC:
- User requested stopping zt3 usage. All known Stage3-related processes on `training-rl-zt3` were stopped: the original-LR GRPO control, the RPP-constrained GRPO attempt, the zt3 secondary checkpoint watchers, and the keep-best train-only buffer builder.
- Do not launch new Stage3 training, checkpoint evaluation, or buffer generation on `training-rl-zt3` unless the user explicitly reverses this rule.
- Updated launch scripts so `SECONDARY_EVAL_HOST` defaults to empty instead of `training-rl-zt3`; secondary evaluation now starts only when a host is explicitly supplied.
- `scripts/training/monitor_recogdrive_stage3_grpo_run.py` no longer treats `secondary_watch_on_rl_zt3_memfit_4gpu` as a default watcher directory.
- Existing zt3 artifacts remain evidence only. Stale status JSON files may still say `"alive": true`, but process verification showed no matching Stage3 zt3 process after cleanup.

## Navtest Diagnostic: Cap05 Self-Imitation Step300 To Step600

Run:
`/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`.

This analysis uses navtest evaluation outputs only. It is diagnostic evidence for
algorithm selection and must not be used as a training reward/cache or buffer
source.

Grouped checkpoint summary:

| Checkpoint | Evals | PDMS | NC | DAC | TTC | EP | Comfort | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step-step_300` | 2 | `0.885432` | `0.981237` | `0.969435` | `0.946326` | `0.826787` | `1.000000` | `0.969023` |
| `step-step_600` | 3 | `0.885497` | `0.976039` | `0.968831` | `0.939941` | `0.835866` | `1.000000` | `0.959068` |
| `step-step_900` | 1 | `0.887107` | `0.984264` | `0.967210` | `0.955841` | `0.823058` | `1.000000` | `0.971742` |

Paired token-level analysis on the zt3 step300 and step600 CSVs:

- Paired valid scenarios: `12139`.
- Mean PDMS delta: `+0.000349`; median delta `0.0`.
- Improved by more than `0.01` PDMS: `3936` scenes (`32.42%`).
- Degraded by more than `0.01` PDMS: `1322` scenes (`10.89%`).
- PDMS-zero scenes increased from `583` to `649`.
- High-score bins improved: `(.95,.99]` increased by `141`, and `(.99,1]` increased by `441`.
- Mid-score bins shrank: `(.8,.88]` decreased by `278`, `(.88,.9]` by `131`, and `(.9,.95]` by `224`.

Safety failure movement under practical gates NC `1.0`, DAC `1.0`,
TTC `0.95`, DDC `0.99`:

| Metric | Fails at step300 | Fails at step600 | New fails | Recovered |
|---|---:|---:|---:|---:|
| NC | `237` | `298` | `125` | `64` |
| DAC | `371` | `380` | `140` | `131` |
| TTC | `664` | `733` | `253` | `184` |
| DDC | `509` | `650` | `226` | `85` |

Interpretation:

- The policy is learning some higher-progress/high-PDMS trajectories: many scenes
  move into the `0.95+` and `0.99+` PDMS bins.
- The mean does not improve because safety regressions also increase. EP rises
  by about `+0.009`, while NC falls `-0.0053`, TTC `-0.0064`, and DDC `-0.0099`.
- This matches the training-path flaw: the self-imitation target selector used
  reward/margin plus broad hard safety, but did not require TTC/DDC-safe targets.
- Next runs must treat NC/DAC/TTC/DDC as target-validity gates for auxiliary
  diffusion regression. Otherwise the auxiliary path can absorb high-progress
  trajectories that hurt the same safety submetrics which dominate navtest PDMS.
- `step-step_900` partially recovered NC/TTC/DDC and improved PDMS to `0.887107`,
  but this came with EP dropping from about `0.8359` to `0.8231`. This indicates
  the run is oscillating between progress and safety rather than producing a
  stable Pareto improvement. It remains only `+0.007107` over the early gate.
  Its gap to the original Stage3 `epoch9-step13300` `0.9055` result is a
  long-horizon target gap, not a fair early-step failure criterion.

Artifacts:

- Summary: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/navtest_pdms_analysis.md`
- Pairwise analysis: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/navtest_pairwise_step300_step600_analysis.md`
- Worst paired deltas: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/navtest_pairwise_step300_step600_worst.tsv`

## Current Baselines And Controls

| Run / Reference | Status | Key Config | Best Known PDMS | Notes |
|---|---:|---|---:|---|
| User-reported original local Stage3 RL | historical | original LR `1e-4`, `epoch9-step13300` | `0.9055` | Treat this as a long-horizon/final-training reference, not a 1-epoch or early-step diagnostic reference. |
| User-reported original Stage3 early training | historical | original Stage3, `epoch0-1` | `0.88+` | Treat this as the short-run sanity gate. A new Stage3 method that is already clearly below this range at matched early epoch/step should not be promoted without a method-level fix. |
| `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z` | completed | Safe DiffGRPO | `0.906184` at `epoch_12-step_17290` | Strongest confirmed Stage3 result so far. Compare against it only at comparable training length, or label the comparison as short-run diagnostic only. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped/evaluated | LR `2e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `64` | `0.872059` at `epoch_0-step_1330` | Exact 8-shard navtest eval: NC `0.9750`, DAC `0.9619`, TTC `0.9371`, EP `0.8159`, comfort `1.0000`, DDC `0.9459`. This is weak versus original early Stage3 `0.88+` and current matched epoch0 controls; the gap to `0.9055/0.906184` should be read only as a long-horizon target gap. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | evaluated/running | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `66`; no AWAC/IQL, no Buffer-DPO, no buffer guidance, no self-imitation, no hard gate | `0.896794` at `epoch_0-step_1290` | zt3 current-code original-LR GRPO control, not a bit-exact historical original-code run. Submetrics: NC `0.987189`, DAC `0.975861`, TTC `0.962597`, EP `0.826516`, comfort `1.000000`, DDC `0.974378`. This is the strongest recent early checkpoint and argues against aggressive safety/self-imitation changes that suppress progress. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z` | running/evaluated epoch0+step1500 | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch `64`, `GRPO_USE_GSPO_RATIO=false`, no buffer/self-imitation/hard gates | `0.893686` at `epoch_0-step_1330` | Purpose: reproduce the strong original-LR GRPO path on the current repository before judging algorithmic variants. Step1500 exact navtest is `0.891700`, below epoch0-step1330 due mainly to EP decline, and both local points remain below the stronger zt3 current-code original-LR control epoch0-step1290 `0.896794`. |
| `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z` | stopped | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, GSPO ratio, RLOO self-imitation, max target scene ratio `0.5`, step ckpt every `300` | `0.887107` at `step-step_900` | Step300 passed the early gate (`0.885432`). Step600 was flat and traded EP for safety loss. Step900 recovered NC/TTC/DDC and reached `0.887107`, but EP fell to `0.823058`; judge this run by matched early-step controls and the progress/safety oscillation, not by direct comparison to the `epoch9-step13300` `0.9055` target. |
| `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z` | stopped/deleted | Same as cap05, with explicit self-imitation target gates NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`, but no main TTC/DDC hard gate | none | Stopped before checkpoint. This was lower-information than the main hard-gate test because it only filtered auxiliary targets. |
| `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` | stopped/delete artifacts | Same LR/sample/effective batch as cap05, with main TTC `0.95` and DDC `0.99` hard gates plus strict self-imitation targets | `0.882934` at `step-step_600` | Failed matched early comparison. Step600 improved safety over cap05 but EP fell to `0.792348`; PDMS stayed below cap05 step900 and below the zt3 original-LR control. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because key diagnostics were not logged, so the run could not validate buffer absorption. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because it was launched before the full PPO/advantage diagnostics patch. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer | none | Failed at Hydra config parsing because `grpo_normalize_advantage_batch` was missing from `recogdrive_agent.yaml`. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stopped/replaced | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=80`, `limit_val_batches=0`, default logging | none | Wrote only `lr-AdamW`; insufficient diagnostic density. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`, behavior sync interval `100000` | no eval | Ratio/KL/clip diagnostics moved after optimizer updates; this validates instrumentation and short-run GSPO activity only. |
| `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z` | completed smoke | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=4`, inner PPO epochs `2`, replay minibatch `4`, 4 GPUs, 40 train batches, no buffer | no eval | End-to-end PPO replay path is active: fixed old logprob replay, manual optimization, PPO ratio diagnostics, reference KL, and BC update ran without NaNs. Too small/conservative to evaluate as a main result. |
| `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z` | completed diagnostic | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, 40 train batches, no buffer | no eval | Near-full sample-time diagnostic passed. Checkpoint `epoch=0-step=360.ckpt`; keep as implementation evidence only, not as a final PDMS result. |
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, full navtrain, no buffer | pending | Latest logged step `339`: `valid_ratio=1.0`, ratio mean `0.9694`, clip fraction `0.2031`, KL `7.87e-4`; latest rollout safety was weak (`safe_ratio=0.6133`), so this remains a watched diagnostic. |
| `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `1`, replay minibatch `8`, 8 zt2 GPUs, full navtrain, no buffer | pending | Motivation: update-strength ablation from i2. Latest logged step `179`: `valid_ratio=1.0`, ratio mean `0.9992`, clip fraction `0`, KL `2.30e-6`, safe ratio `0.8281`. |

Current-repo original-LR control status on 2026-06-14 19:23 UTC:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z`.
- Training is running on local GPUs `0-7`; monitor reported about `40.9 GB` per GPU and high SM utilization after SceneLoader finished.
- The resolved command confirms no buffer, no DPO, no self-imitation, no hard TTC/DDC gate, and `GRPO_USE_GSPO_RATIO=false`.
- TensorBoard had only `lr-AdamW` at step `0` at this timestamp; no reward/loss scalar and no checkpoint yet. This is expected before the first GRPO/PDM training log window completes.
- zt2 and zt3 checkpoint watchers are attached but only waiting for checkpoint files; they are configured to wait for free GPUs and not preempt existing remote tasks.

Update on 2026-06-14 19:34 UTC:
- The same current-repo control has entered normal training and logged step `49`:
  - `train/reward_step=0.785141`
  - `train/base_reward_step=0.789642`
  - `train/safe_ratio_step=0.914062`
  - `train/mean_ep_step=0.769345`
  - `train/mean_ttc_step=0.898438`
  - `train/mean_ddc_step=0.939453`
  - `train/reference_kl_loss_step=0.007989`
  - `train/bc_coeff_step=0.1`
- No checkpoint or navtest eval row exists yet for this run. Keep it running as the current-code control before launching the buffer-DPO diagnostic.

Update on 2026-06-14 19:51 UTC:
- The run is still alive on local GPUs `0-7`; GPU memory is about `41 GB/card` and utilization is active.
- Latest TensorBoard train scalar is step `99`:
  - `train/reward_step=0.694921`
  - `train/base_reward_step=0.722842`
  - `train/safe_ratio_step=0.875000`
  - `train/mean_ep_step=0.745482`
  - `train/mean_nc_step=0.945312`
  - `train/mean_dac_step=0.937500`
  - `train/mean_ttc_step=0.820312`
  - `train/mean_ddc_step=0.994141`
- Still no `step300` checkpoint and no navtest `checkpoint_eval_submetrics.tsv` row. The zt2/zt3 checkpoint watchers are healthy and repeatedly report `no pending checkpoint evals; training_state=running`.
- Decision unchanged: do not start buffer-DPO until this current-code control has an early exact navtest row.

Update on 2026-06-14 21:20 UTC:
- `step-step_300` exact navtest eval completed through the local foreground exact-pool path:
  - PDMS `0.881910`
  - NC `0.984800`, DAC `0.969682`, TTC `0.957571`, EP `0.805231`, comfort `0.999918`, DDC `0.965316`
  - CSV: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z/local_foreground_eval_step300_4gpu/eval_step-step_300/hydra/stage3_safe_diffgrpo_eval_exact_pool_pdm/2026.06.14.20.44.29/2026.06.14.21.14.37.csv`
- zt2/zt3 watchers saw the global `step-step_300.done` marker and skipped duplicate evaluation. Keep them attached for future checkpoints, but do not repeat the failed local background wrapper attempts listed in the cleanup section.
- Decision gate now recommends the control-aligned Buffer-DPO attempt as the next algorithmic run. The reason is not that the buffer is bad: strict train-only buffer validation is strong. The reason is that the current GRPO-control step300 is below the stronger zt3 original-LR control, so any buffer absorption experiment must be isolated and compared against this control shape.

Update on 2026-06-14 21:33 UTC:
- Current-repo GRPO control is still running locally on GPUs `0-7`; latest summary step is `549` with train reward `0.840181` and safe ratio `0.929688`. No new checkpoint beyond `step-step_300` exists yet.
- zt2 currently has all 8 GPUs occupied by an existing two-expert training job. A Buffer-DPO training launcher was therefore queued with strict wait thresholds rather than starting on top of that job:
  - Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`.
  - Remote waiting process on `training-vla-zt2`: shell PID `2647910`, launcher PID `2647912`.
  - Training wait gate: target GPUs `0,1,2,3,4,5,6,7`, `TRAIN_GPU_MAX_MEM_USED_MB=2000`, `TRAIN_GPU_MAX_UTIL=5`.
  - Eval watcher wait gate is also strict: `EVAL_GPU_MAX_MEM_USED_MB=2000`, `EVAL_GPU_MAX_UTIL=5`, so navtest eval will not start on busy remote GPUs.
  - Config file: `strict_gspo_launch_config.txt`.
- Immediate strict-v2 buffer validation before launch still passed:
  - `num_records=85109`, `valid_candidate_ratio=0.982654`, `has_valid_candidate_ratio=1.0`.
  - `mean_best_valid_reward=0.974021`, `pct_best_valid_above_gt=0.582594`, `pct_best_valid_above_il=0.793641`.
  - Saved under the run root as `buffer_validate_before_launch.json` and `.csv`.
- Monitoring/gate scripts were patched after the queue launch:
  - `summarize_recogdrive_stage3_runs.py` now reads `queued_status.txt` and reports queued runs as `queued_*` with recommendation `wait_for_free_gpus`.
  - `decide_recogdrive_stage3_next_action.py` now detects queued/running Buffer-DPO runs and returns `wait_for_buffer_dpo_queue` instead of generating another duplicate launch command.
  - Current gate output: active Buffer-DPO run is `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`.

Update on 2026-06-14 21:44 UTC:
- Current-repo GRPO control wrote `step-step=600.ckpt` at `2026-06-14T21:40:46Z`.
- Both attached exact-navtest watchers saw it after the 120 second age guard and archived it by hardlink:
  - zt2 archive: `unique_lock_watch_on_vla_zt2_4gpu/checkpoint_archive/step-step_600.ckpt`
  - zt3 archive: `secondary_watch_on_rl_zt3_memfit_4gpu/checkpoint_archive/step-step_600.ckpt`
- No step600 PDMS result yet. Both watchers are in `pending_evals=1` and waiting for free GPUs; no remote task was killed or preempted.

Update on 2026-06-14 22:29 UTC:
- `step-step_600` exact navtest eval completed through the local foreground exact-pool path while holding the global `step-step_600.lock`; a `step-step_600.done` marker was written so remote watchers skip duplicates.
- Result:
  - PDMS `0.886777`
  - NC `0.983276`, DAC `0.968364`, TTC `0.948262`, EP `0.829079`, comfort `1.000000`, DDC `0.963585`
  - CSV: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z/local_foreground_eval_step600_4gpu/eval_step-step_600/hydra/stage3_safe_diffgrpo_eval_exact_pool_pdm/2026.06.14.21.57.57/2026.06.14.22.29.02.csv`
- Step300 to step600 delta:
  - PDMS `+0.004866`
  - EP `+0.023848`
  - NC `-0.001524`, DAC `-0.001318`, TTC `-0.009310`, DDC `-0.001730`
- Interpretation: current-repo original-LR GRPO is improving early PDMS, but the improvement is mostly progress-driven and costs TTC/DDC/NC/DAC. This matches the earlier cap05 self-imitation failure mode and supports the next controlled Buffer-DPO/preference test: absorb train-only high-PDMS buffer trajectories while explicitly preventing safety-submetric regression.

Update on 2026-06-15 01:00 UTC:
- `step-step_900` and `step-step_1200` exact navtest evals completed through the local foreground exact-pool path, each with a global done marker:
  - `step-step_900`: PDMS `0.883839`, NC `0.987313`, DAC `0.970094`, TTC `0.959796`, EP `0.806834`, comfort `0.999918`, DDC `0.968281`.
  - `step-step_1200`: PDMS `0.891510`, NC `0.979486`, DAC `0.973142`, TTC `0.946202`, EP `0.837240`, comfort `0.999753`, DDC `0.956912`.
- Step300 to step1200 delta:
  - PDMS `+0.009600`
  - EP `+0.032008`, DAC `+0.003460`
  - NC `-0.005314`, TTC `-0.011369`, DDC `-0.008403`
- Interpretation: the current-repo original-LR GRPO control is now clearly above the `0.88` early gate and above the stopped cap05 step900 result, but it still trails the zt3 original-LR control at a comparable early horizon (`0.896794` at `epoch_0-step_1290`). The gain is again dominated by progress, while TTC/DDC/NC regress. This reinforces the same algorithmic lesson: plain GRPO can find higher-progress samples, but the next useful method must preserve that progress while preventing safety-submetric erosion. Continue the control for matched-epoch evidence, but do not treat step1200 as a solved improvement.

Update on 2026-06-15 01:37 UTC:
- `epoch_0-step_1330` exact navtest eval completed through the local foreground exact-pool path and wrote a global done marker:
  - PDMS `0.893686`, NC `0.986489`, DAC `0.973554`, TTC `0.958148`, EP `0.827006`, comfort `0.999918`, DDC `0.965398`.
- Step300 to epoch0 delta:
  - PDMS `+0.011776`
  - EP `+0.021775`, DAC `+0.003872`, NC `+0.001689`, TTC `+0.000577`, DDC `+0.000082`
- Step1200 to epoch0 delta:
  - PDMS `+0.002177`
  - NC `+0.007003`, TTC `+0.011946`, DDC `+0.008486`
  - EP `-0.010234`, DAC `+0.000412`
- Compared with zt3 current-code original-LR GRPO control `epoch_0-step_1290`, local current-repo control is lower by `0.003107` PDMS, with similar EP but lower NC/DAC/TTC/DDC. zt3 is not a buffer/AWAC/DPO improvement; it is a current-code GRPO control using LR `1e-4`, sample_time `16`, BC anneal, and reference KL.
- Interpretation: local current-repo GRPO recovered the safety regression seen at step1200 while retaining a meaningful progress gain over step300. This is a valid early GRPO baseline, but it is still not a new improvement over the strongest recent early control. The historical original Stage3 `epoch9-step13300` and Safe DiffGRPO `epoch_12-step_17290` references remain long-horizon targets. Continue to the next epoch/checkpoint and keep the queued Buffer-DPO run as the next isolated buffer-absorption test once zt2 resources become available.

Update on 2026-06-15 02:24 UTC:
- `step-step_1500` exact navtest eval completed through the local foreground exact-pool path with `12138` valid rows and no failed scenarios:
  - PDMS `0.891700`, NC `0.986200`, DAC `0.973801`, TTC `0.958066`, EP `0.822248`, comfort `0.999918`, DDC `0.970176`.
- Step1500 is below local epoch0-step1330 by `0.001987` PDMS:
  - EP `-0.004757`
  - DDC `+0.004778`
  - NC `-0.000288`, DAC `+0.000247`, TTC `-0.000082`
- Compared with zt3 current-code original-LR GRPO control `epoch_0-step_1290`, step1500 is lower by `0.005094` PDMS. This reinforces that the zt3 run is currently the stronger short-run GRPO control, not an AWAC/IQL, Buffer-DPO, self-imitation, hard-gate, or buffer-guidance improvement.
- Interpretation: after epoch0 the local control did not continue monotonic PDMS improvement. The small step1500 drop is mainly progress loss while DDC improves. Treat this as normal GRPO progress/safety oscillation evidence, not as proof of a successful algorithm change.

Train-only keep-best elite buffer status on 2026-06-14 19:23 UTC:
- Buffer path: `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`.
- Full train-token coverage exists: `85109` `*.pkl.xz` records, about `342 MB`.
- The keep-best zt3 builder is still running and updating records; the latest file mtimes were current at 19:21 UTC.
- A read-only random sample of `5000` records showed:
  - schema version: all v2; missing required fields: `0`; load/shape errors: `0`.
  - mean selected candidates per record: `9.2222`.
  - mean selected valid ratio: `0.980509`; has-valid-candidate ratio: `1.0`.
  - mean GT reward: `0.931496`; mean IL reward: `0.700525`.
  - mean raw best reward: `0.973363`; mean valid best reward: `0.973272`.
  - mean valid best minus GT: `+0.041776`; mean valid best minus IL: `+0.272747`.
  - `pct_best_valid_above_gt=0.587`; `pct_best_valid_above_il=0.802`.
  - valid sources are dominated by structured perturbations: `progress_endpoint`, `progress_speed`, `progress_gamma`, plus policy/lateral/timing candidates.
- Interpretation: the oracle/buffer side is not the main current bottleneck. The buffer contains many valid trajectories that beat GT, so the next algorithmic question remains policy absorption into the diffusion sampler. Any buffer-DPO/preference run must be isolated on top of the original-LR GRPO control shape, not on top of the failed v3 hard-gate configuration.

Update on 2026-06-14 19:51 UTC:
- The keep-best builder is still active on `training-rl-zt3`; shard logs updated at 19:38 and 19:41 UTC.
- Buffer coverage remains `85109` records, about `342 MB`, with latest record mtimes around 19:44 UTC. This means the full train-only buffer is available and still being refreshed; no intervention is needed.

## Planned Attempt: Original-LR GRPO With RLOO Self-Imitation

Motivation:
- User correction: original Stage3 `epoch0-1` is already around `0.88+` PDMS, so the transition replay `epoch0=0.831015` is a failed early gate, not a training-length issue.
- The AWAC elite buffer is good as an oracle (`mean_best_valid_reward=0.971664`, `pct_best_valid_above_gt=0.581537`), but pure weighted denoising did not transfer to sampled trajectories.
- The strongest confirmed family remains on-policy GRPO/Safe DiffGRPO, so the next change must preserve the original GRPO objective and `1e-4` LR.

Reference audit:
- DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo. Mechanism checked: diffusion-policy PG should optimize sampled actions with old-policy logprobs/ratio clipping; reward-weighted regression is not enough by itself.
- DDPO paper: https://arxiv.org/abs/2305.13301. Mechanism checked: denoising can be treated as a multi-step decision process, which supports direct policy-gradient optimization over diffusion samples.
- RIPT-VLA code: https://github.com/Ariostgx/ript-vla. Mechanism checked: K-rollout, leave-one-out advantage, dynamic sampling, and no value net are used for VLA post-training.

Implementation:
- Add `group_leave_one_out` to `offline_rl_grpo_self_imitation_baseline_mode`.
- Keep `stage3_objective=grpo`, `LR=1e-4`, `sample_time=16`, BC anneal `0.10 -> 0.05`, reference KL `0.02`, GSPO ratio on.
- Add only a small auxiliary self-imitation loss:
  - `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.01`
  - `GRPO_SELF_IMITATION_LOSS_SCHEDULE=linear_warmup`
  - `GRPO_SELF_IMITATION_WARMUP_EPOCHS=2`
  - `GRPO_SELF_IMITATION_MIN_REWARD=0.88`
  - `GRPO_SELF_IMITATION_MIN_REWARD_MARGIN=0.01`
  - `GRPO_SELF_IMITATION_BASELINE_MODE=group_leave_one_out`
  - `GRPO_SELF_IMITATION_TIMESTEP_SAMPLING=low_noise`
- Do not enable buffer reward bonus or buffer distill in this first run. The offline buffer can remain available for diagnostics, but it must not dominate the update.

Smoke status:
- `stage3_grpo_rloo_selfimit_smoke_20260614T_check` completed 1 train batch on 1 GPU with `sample_time=2`.
- Hydra accepted `offline_rl_grpo_self_imitation_baseline_mode=group_leave_one_out`.
- Logged `grpo_self_imitation_enabled=1.0`, `baseline_from_buffer=0.0`, `use_gspo_ratio=1.0`.
- This smoke batch had `base_reward=0.625077`, below the `0.88` imitation threshold, so `target_ratio=0.0` and `zero_weight_batch=1.0`; this is expected for a low-reward smoke batch and confirms that low-quality samples are not imitated.

Live full-run status:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_s16_lr1e4_b2acc4_8gpu_20260614T101257Z`.
- Training host/GPU: local `training-vla-zt`, GPUs `0-7`.
- Eval watchers: two `training-vla-zt2` watcher processes are active and waiting for epoch checkpoints.
- Additional eval watcher: `training-rl-zt3` pid `918395`, GPUs `6,7`, strict free-GPU wait, same `global_checkpoint_eval_locks`, added because zt2 GPUs were occupied by an unrelated existing run.
- Config keeps original-LR GRPO/Safe DiffGRPO defaults: `LR=1e-4`, `sample_time=16`, effective batch about `64`, BC `0.10 -> 0.05`, reference KL `0.02`, GSPO ratio on.
- First logged train scalar at `step=49`:
  - `reward=0.797566`, `base_reward=0.792112`, `safe_ratio=0.933594`, `mean_ep=0.773152`, `mean_ttc=0.902344`.
  - `grpo_self_imitation_enabled=1.0`, `weight=0.005` during warmup.
  - `grpo_self_imitation_candidate_ratio=0.429688`, `target_ratio=0.875000`, `target_reward_mean=0.976924`.
- Second logged train scalar at `step=99`:
  - `reward=0.731460`, `base_reward=0.750897`, `safe_ratio=0.890625`, `mean_ep=0.768155`, `mean_ttc=0.820312`.
  - `grpo_self_imitation_candidate_ratio=0.378906`, `target_ratio=0.875000`, `target_reward_mean=0.969151`.
- Third logged train scalar at `step=149`:
  - `reward=0.762590`, `base_reward=0.770046`, `safe_ratio=0.906250`.
  - `grpo_self_imitation_target_ratio=0.937500`, `target_reward_mean=0.977498`.
- Interpretation: the self-imitation path is active and is selecting very high train-reward sampled trajectories rather than imitating low-quality rollouts. However `target_ratio` is `0.875`, `0.875`, then `0.9375` on the first three logged batches, so the auxiliary selection is not sparse. If this run fails navtest despite high target rewards, the first algorithmic fix should be a stricter self-imitation gate or rank/quantile cap, not another LR-only change. This is still only an internal train diagnostic; promotion requires exact navtest PDMS at `epoch0` near or above the original Stage3 early `0.88+` band.

Expected diagnostics:
- `grpo_self_imitation_target_ratio` should be non-zero but sparse; a zero ratio for many steps means the gate is too strict.
- `grpo_self_imitation_target_reward_mean` should be above `0.88`.
- `grpo_self_imitation_zero_weight_batch` should not stay at `1.0` after warmup if the policy finds good samples.
- Main GRPO diagnostics must remain healthy: `gspo_ratio_mean` near `1`, finite small `gspo_approx_kl`, no explosion in `safe_ratio` or hard safety metrics.

Follow-up implementation after live diagnostics:
- Evidence from the first logged full-run batches showed `grpo_self_imitation_target_ratio` was `0.875`, `0.875`, then `0.9375`, while `target_reward_mean` stayed high around `0.969-0.977`.
- This means the on-policy target quality is good, but the self-imitation gate is too broad to behave like sparse elite imitation. It risks turning the auxiliary loss into a broad reward-weighted BC signal, which is exactly the failure mode observed in earlier offline AWAC-style attempts.
- Added configurable batch-level scene cap:
  - `offline_rl_grpo_self_imitation_max_target_scene_ratio` default `1.0` for backward compatibility.
  - `offline_rl_grpo_self_imitation_batch_cap_score` in `{reward, margin}`, default `reward`.
  - Logs `grpo_self_imitation_pre_cap_target_ratio`, `grpo_self_imitation_target_scene_cap_ratio`, and `grpo_self_imitation_target_scene_cap_active`.
- The RLOO launcher default for the next run is `GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO=0.5`; the base GSPO launcher remains `1.0`.
- Synthetic CPU check confirmed a cap of `0.5` reduces `pre_cap_target_ratio=1.0` to `target_ratio=0.5` and keeps the highest-reward scenes.
- Live cap05 run first logged batch on 2026-06-14:
  - `step=49`, reward `0.785556`, base reward `0.784031`, safe ratio `0.925781`, EP `0.772696`, TTC `0.882812`.
  - `grpo_self_imitation_candidate_ratio=0.429688`, `target_ratio=0.500000`, `target_reward_mean=0.983409`, `target_margin_mean=0.162914`.
  - This confirms the scene cap is active and avoids the earlier no-cap target ratios around `0.75-0.94`; promotion still requires step/epoch PDMS.
- Exact navtest gate result on 2026-06-14:
  - `step-step_300` reached PDMS `0.885432`, passing the original Stage3 early `0.88+` sanity gate.
  - Submetrics: NC `0.981237`, DAC `0.969435`, TTC `0.946326`, EP `0.826787`, comfort `1.000000`, DDC `0.969023`.
  - Interpretation: this is the first recent non-original variant to clear the early gate. Continue the run and evaluate `step-step_600`/epoch0 before judging whether the self-imitation cap improves the trend beyond the original Stage3 baseline.
- Exact navtest step600 result on 2026-06-14:
  - `step-step_600` reached PDMS `0.885497`, essentially flat from step300 and still only in the original early-training band.
  - Submetrics: NC `0.976039`, DAC `0.968831`, TTC `0.939941`, EP `0.835866`, comfort `1.000000`, DDC `0.959068`.
  - Delta from step300: EP improved by about `+0.0091`, but NC fell by about `-0.0052`, TTC by about `-0.0064`, and DDC by about `-0.0100`.
  - Interpretation: the auxiliary self-imitation path is likely absorbing higher-progress on-policy samples, but target selection does not explicitly protect TTC/DDC. A further run without fixing target validity would be low-information.

Main hard-gate v3 early result on 2026-06-14:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`.
- `step-step_300` exact navtest PDMS is `0.879395`, below the `0.88` original early gate but within the configured `0.005` watch margin.
- Submetrics: NC `0.985335`, DAC `0.962679`, TTC `0.961855`, EP `0.803776`, comfort `0.999835`, DDC `0.982699`.
- `step-step_600` exact navtest PDMS is `0.882934`.
- Step600 submetrics: NC `0.990649`, DAC `0.968776`, TTC `0.972730`, EP `0.792348`, comfort `0.999835`, DDC `0.979609`.
- Delta from step300 to step600: NC/TTC/DAC improved, but EP fell by about `-0.0114`, and DDC stayed below the strict target threshold.
- Interpretation: the hard safety gate did what it was asked to do locally, but the objective over-corrected toward safety and did not learn a better progress/safety Pareto point. It underperforms both cap05 step900 (`0.887107`) and the original-LR control at a similar early horizon (`0.896794`), so the run should be stopped and its artifacts deleted after this record. Do not repeat main TTC/DDC hard gating plus strict self-imitation as the default GRPO route.

## Planned Attempt: GRPO With Train-Buffer Diffusion-DPO Absorption

Motivation:
- Past AWAC/IQL attempts show that the train-only elite buffer can discover high-PDMS trajectories, but weighted denoising regression alone did not make the sampler reliably output better trajectories.
- The main hard-gate v3 GRPO run showed that strict hard safety gates can raise NC/TTC while lowering EP enough to reduce PDMS. Therefore buffer preference should not inherit the v3 hard gates by default.
- The next buffer use should therefore not replace GRPO. It should add a small preference signal that tells the diffusion planner: for the same scene/context/noise level, assign lower denoising loss to valid train-buffer winners than to GT/IL behavior losers.

Reference audit:
- Diffusion-DPO paper and official code: https://arxiv.org/abs/2311.12908 and https://github.com/SalesforceAIResearch/DiffusionDPO. Mechanism copied: chosen/rejected samples share the same diffusion timestep and noise; the objective compares current-model loss gap against reference-model loss gap with a sigmoid/DPO classification loss.
- DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo. Mechanism retained from the GRPO backbone: keep policy-gradient style optimization over sampled diffusion trajectories rather than relying only on offline regression.
- DGPO code: https://github.com/Luo-Yihong/DGPO. Mechanism borrowed conceptually: group/reward ordering information can be used as direct preference supervision for diffusion models, not only as scalar reward-weighted regression.
- SIPO / SDPO: https://arxiv.org/abs/2505.21893. Caution borrowed: diffusion preference optimization is timestep-sensitive and off-policy biased. The implementation keeps timestep sampling configurable and starts with a small warmup-controlled weight.

Implementation:
- Added GRPO-specific buffer preference-DPO fields under `OfflineRLConfig`, all defaulting to disabled:
  - `grpo_buffer_preference_dpo_loss_weight=0.0`
  - `grpo_buffer_preference_dpo_loss_schedule=linear_warmup`
  - `grpo_buffer_preference_dpo_timestep_sampling=uniform`
  - `grpo_buffer_preference_dpo_beta=8.0`
  - `grpo_buffer_preference_dpo_pair_mode=best_vs_gt_il`
  - `grpo_buffer_preference_dpo_min_reward_gap=0.02`
  - `grpo_buffer_preference_dpo_max_pairs_per_scene=2`
- GRPO buffer guidance now keeps the loaded selected candidates/source codes in memory for the current batch.
- The DPO batch is constructed as:
  - valid buffer target(s) selected from train-only elite buffer are winners;
  - GT trajectory from `action_input.action` is a behavior loser;
  - IL/reference trajectory is loaded from the same buffer when present;
  - GT/IL rows are real loser rows but are not marked valid winners.
- Implementation hardening on 2026-06-14 UTC: old-policy fallback for missing IL losers was removed from the GRPO buffer-DPO path. A sampled fallback IL trajectory would not have a reward recomputed for the exact sampled trajectory and would instead inherit `guidance["il_reward"]` from the buffer record, so it could create mismatched preference pairs. If a future buffer record lacks IL support, that IL loser row is simply not marked real; GT remains available as the behavior loser.
- Smoke coverage added: `scripts/smoke_test_recogdrive_awac_iql.py` now checks a two-row GRPO buffer-DPO target batch where one scene has stored IL support and one scene does not. The missing-IL row must keep the IL column `real_mask=False`, proving the code does not synthesize an unmatched IL loser.
- The existing diffusion-DPO loss is reused, including shared noise/timestep and reference-policy loss gap. Candidate trajectories are denoising targets only; `_prepare_dit_context(..., allow_target_tokens=False)` remains enforced through `_diffusion_per_target_loss_on_targets`.
- Added training logs:
  - `grpo_buffer_preference_dpo_loss`
  - `grpo_buffer_preference_dpo_weight`
  - `grpo_buffer_preference_dpo_pair_count`
  - `grpo_buffer_preference_dpo_active_row_ratio`
  - `grpo_buffer_preference_dpo_reward_gap_mean`
  - `grpo_buffer_preference_dpo_logit_mean`
  - `grpo_buffer_preference_dpo_implicit_accuracy`
  - `grpo_buffer_preference_dpo_timestep_mean/min/max`
- Added dedicated launcher:
  `scripts/training/launch_recogdrive_stage3_grpo_buffer_dpo_2b_local_stable.sh`.
  Its defaults isolate buffer-DPO by setting buffer reward bonus/distill to `0.0`
  and self-imitation to `0.0`; set `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.01`
  only for an explicit v3-plus-buffer-DPO comparison.
- Launcher correction on 2026-06-14 UTC:
  - The buffer-DPO launcher previously still defaulted to a `mainhardgate` run name plus `GRPO_HARD_GATE_TTC=true` and `GRPO_HARD_GATE_DDC=true`, contradicting the post-v3 rule above.
  - It was corrected to default to `stage3_grpo_buffer_dpo_refctrl...`, `GRPO_USE_GSPO_RATIO=false`, `GRPO_HARD_GATE_TTC=false`, `GRPO_HARD_GATE_DDC=false`, and TTC/DDC thresholds `1.0`, matching the active current-repo original-LR GRPO control shape.
  - A `RUN_TRAIN=0` preflight confirmed: `offline_rl_enabled=true`, train-only elite buffer path set, `grpo_buffer_guidance_enabled=true`, `grpo_buffer_preference_dpo_loss_weight=0.02`, and buffer reward bonus/distill/self-imitation weights all `0.0`. The temporary preflight directory was deleted after inspection.
- Buffer-DPO data-path audit on 2026-06-14 UTC:
  - A deterministic random sample of `5000 / 85109` v2 train-only elite-buffer records showed `gt_present_ratio=1.0` and `il_present_ratio=1.0`.
  - `gt_valid_ratio=0.9992`; `il_valid_ratio=0.8314`. The DPO builder can therefore use stored GT/IL support for the intended behavior-loser pairs in normal cases, rather than relying on old-policy IL fallback with potentially mismatched stored `il_reward`.
  - In the same sample, `best_valid_source` was dominated by `gt`, `progress_endpoint`, and `policy`; this keeps the first buffer-DPO run focused on absorbing valid progress/policy improvements rather than the failed AWAC weighted-regression path.

First experiment rule after v3 result:
- Do not launch a buffer-DPO run on top of the failed v3 main-hardgate configuration by default. That would confound buffer absorption with a known EP-suppressing safety gate.
- If buffer-DPO is tested next, match the stronger original-LR control shape instead: LR `1e-4`, `sample_time=16`, BC `0.10->0.05`, reference KL `0.02`, no main TTC/DDC hard gate, no broad self-imitation. Enable only a small train-buffer DPO auxiliary:
  - `GRPO_BUFFER_GUIDANCE_ENABLED=true`
  - `GRPO_BUFFER_REWARD_BONUS_WEIGHT=0.0`
  - `GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0`
  - `GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT=0.02`
  - `GRPO_BUFFER_PREFERENCE_DPO_LOSS_SCHEDULE=linear_warmup`
  - `GRPO_BUFFER_PREFERENCE_DPO_WARMUP_EPOCHS=2`
  - keep `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.0` for the first isolated buffer-DPO diagnostic.
- Success at the diagnostic level requires active nonzero DPO pairs and no deterioration in matched step300/600 exact navtest NC/TTC/DDC/EP compared with the zt3 original-LR control. Final success still requires comparable-length PDMS above the original Stage3 `epoch9-step13300` `0.9055` and Safe DiffGRPO `0.906184` references.

Launch status on 2026-06-14 21:33 UTC:
- The first isolated Buffer-DPO diagnostic is queued on `training-vla-zt2` as
  `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`.
- It is intentionally waiting for all 8 zt2 GPUs to become free; no existing remote task was killed.
- The launch config confirms the intended isolation:
  - `grpo_buffer_preference_dpo_loss_weight=0.02`
  - `grpo_buffer_reward_bonus_weight=0.0`
  - `grpo_buffer_distill_loss_weight=0.0`
  - `grpo_self_imitation_loss_weight=0.0`
  - `grpo_hard_gate_ttc=false`, `grpo_hard_gate_ddc=false`
  - `grpo_use_gspo_ratio=false`, matching the current-repo control.
- The run should be judged first by whether DPO diagnostics become active after training starts:
  `grpo_buffer_preference_dpo_pair_count`, `active_row_ratio`, `reward_gap_mean`,
  `logit_mean`, and `implicit_accuracy`. Do not judge it only from final PDMS.
- Follow-up guard on 2026-06-14 21:38 UTC:
  - Summary now reports this run as `queued_waiting_for_free_zt2_gpus` with train pid `2647912`.
  - Next-action gate now returns `wait_for_buffer_dpo_queue`, so operators should not launch another Buffer-DPO copy while this queued run is alive.

## Planned Attempt: Safety-Filtered GRPO Self-Imitation Targets

Motivation:
- The cap05 RLOO self-imitation run passed the original Stage3 early `0.88+` gate, but step600 showed EP gains coupled with NC/TTC/DDC degradation.
- Code inspection confirmed that self-imitation target selection used `hard_safe_mask + reward + margin`, while `hard_gate_ttc=False` and `hard_gate_ddc=False` by default. Therefore high-PDMS/high-progress samples with weaker TTC/DDC could become diffusion regression targets.

Reference audit:
- RIPT-VLA / Interactive Post-Training for VLA models: https://arxiv.org/abs/2505.17016 and https://github.com/Ariostgx/ript-vla. Relevant mechanism: K-rollout policy improvement with leave-one-out advantage and PPO-style policy updates, not pure offline weighted regression.
- DPPO: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo. Relevant mechanism: keep direct policy-gradient optimization of sampled diffusion actions as the main objective; auxiliary regression should not replace on-policy improvement.
- The resulting design keeps GRPO as the main update and only filters the auxiliary self-imitation targets more strictly.

Implementation:
- Add explicit GRPO self-imitation target gates:
  - `offline_rl_grpo_self_imitation_require_nc=true`
  - `offline_rl_grpo_self_imitation_require_dac=true`
  - `offline_rl_grpo_self_imitation_require_ttc=true`, threshold `0.95`
  - `offline_rl_grpo_self_imitation_require_ddc=true`, threshold `0.99`
- Add logs:
  - `grpo_self_imitation_safety_candidate_ratio`
  - `grpo_self_imitation_nc/dac/ttc/ddc_pass_ratio`
  - `grpo_self_imitation_target_nc/dac/ttc/ep/comfort/ddc/tlc_mean`
- Keep the main GRPO reward and PDMS scorer unchanged.
- Keep original LR `1e-4`, sample_time `16`, BC `0.10->0.05`, reference KL `0.02`, GSPO ratio, RLOO baseline, and scene cap `0.5` so the algorithmic change is isolated.

Launch record:
- Run root:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z`.
- Local training PID file:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z/pids/stage3_rl_2b.pid`.
- Confirmed resolved command contains:
  - `agent.lr=1e-4`
  - `agent.grpo_sample_time=16`
  - `agent.reference_kl_coeff=0.02`
  - `checkpoint.every_n_train_steps=300`
  - `agent.offline_rl_grpo_self_imitation_require_nc=true`
  - `agent.offline_rl_grpo_self_imitation_require_dac=true`
  - `agent.offline_rl_grpo_self_imitation_require_ttc=true`
  - `agent.offline_rl_grpo_self_imitation_require_ddc=true`
  - thresholds NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`.
- Eval watchers:
  - zt2: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z/unique_lock_watch_on_vla_zt2_4gpu`
  - zt3: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z/secondary_watch_on_rl_zt3_memfit_4gpu`
  - Both watchers use strict free-GPU gates (`EVAL_GPU_MAX_MEM_USED_MB=2000`, `EVAL_GPU_MAX_UTIL=5`) because zt2/zt3 already had existing evaluation/buffer tasks.
- The old cap05 local training was stopped after step900 checkpoint creation.
  One remote step900 eval later completed with PDMS `0.887107`; another duplicate
  eval, if still running, is left to finish naturally.

Expected diagnostics:
- `grpo_self_imitation_safety_candidate_ratio` should be lower than the current broad candidate ratio, but not zero for long stretches.
- Target TTC/DDC means should stay near the configured thresholds or above.
- If early PDMS stays around `0.885` but DDC/TTC no longer decline, the safety filter is doing its intended job; a later run can then test stronger buffer absorption.

Failure criteria:
- If `step-step_300` or epoch0 drops materially below the original early `0.88+` band, do not promote.
- If target ratio collapses to zero for most logged batches, loosen the target safety gate or switch the auxiliary off rather than continuing a no-op run.
- If EP improves but DDC/TTC still decline, the issue is not just target filtering and the next fix should be reward/advantage shaping in the GRPO path or preference ranking, not another self-imitation gate.

## Background Task: Keep-Best Train-Only Buffer Exploration

Motivation:
- The full v2 navtrain buffer already has `mean_best_valid_reward=0.971664` and can still provide a useful oracle for discovering better-than-GT trajectories.
- The pure AWAC training path did not transfer well, but the buffer remains useful for diagnostics, future safety-constrained self-imitation, and preference pairs.

Implementation:
- Added `MERGE_EXISTING_RECORDS=1` support to `build_recogdrive_stage3_awac_elite_buffer.py`.
- New records are merged with existing v2 records per token, deduplicated, and retained by valid top-k selection plus GT/IL support candidates.
- Existing high-quality records are not blindly overwritten by a lower-quality exploration pass.
- Added `scripts/training/launch_recogdrive_stage3_awac_buffer_keepbest_background.sh`:
  - waits for target GPUs instead of killing or preempting existing tasks;
  - loops over navtrain shards;
  - supports `AUTO_POLICY_CHECKPOINT_DIR` so the latest Stage3 policy checkpoint can provide additional sampled candidates;
  - uses only `metric_cache_train_full` for scoring.

Navtest analysis rule:
- Added `scripts/evaluation/analyze_recogdrive_stage3_navtest_pdms.py`.
- It reads existing `checkpoint_eval_submetrics.tsv` files and reports best/latest checkpoint, step deltas, submetric deltas, and deltas to the early `0.88` gate plus the long-horizon `0.9055` / `0.906184` targets.
- `scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh` now refreshes `navtest_pdms_analysis.md` and `navtest_pdms_analysis.tsv` under the run root after each successful checkpoint evaluation.
- `AgentLightningModule` now logs the GRPO self-imitation safety-candidate pass ratios and selected-target NC/DAC/TTC/EP/comfort/DDC/TLC means when the planner returns them. This is required to connect navtest regressions back to train-time target selection.
- This is diagnostic only. Navtest summaries must not be used to choose buffer records or train rewards.

Promotion / failure criteria:
- At matched early epoch, exact navtest PDMS must be near or above the original Stage3 early `0.88+` band. If epoch0 is materially below `0.88`, do not run 20 epochs unless diagnostics show the self-imitation path was inactive and the run is effectively a control.
- Compare final training only against the original Stage3 `epoch9-step13300` `0.9055` and Safe DiffGRPO `0.906184` when training length/steps are comparable.

## Completed Attempt Ledger

### 1. AWAC/IQL Offline Elite-Buffer Training

Motivation:
- Treat GT as behavior support, not as optimal.
- Use train-only PDMS as an oracle to discover better trajectories from GT, IL/reference policy, current policy samples, and structured perturbations.
- Train the diffusion planner with advantage-weighted denoising targets.

Implementation work completed:
- Added Stage3 objective switch and AWAC/IQL path.
- Added strict v2 elite buffer with valid masks and submetric guards.
- Added structured perturbation explorer and validator.
- Added missing-submetric safety handling.
- Added DPO/ranking/repulsion options inside the AWAC path.

Oracle/buffer evidence:
- Full v2 train buffer validation over `85109` records:
  - `valid_candidate_ratio`: `0.979694`
  - `has_valid_candidate_ratio`: `1.0`
  - `mean_gt_reward`: `0.932272`
  - `mean_il_reward`: `0.709482`
  - `mean_best_valid_reward`: `0.971664`
  - `pct_best_valid_above_gt`: `0.581537`
  - `pct_best_valid_above_il`: `0.791338`
- The buffer is useful: many train scenes have valid candidates above GT and IL.

Evaluation results:

| Run | Best Ckpt | PDMS | NC | DAC | TTC | EP | Comfort | DDC | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `stage3_awac_iql_dualhost_20260612T182947Z` | `epoch_0-step_665` | `0.860226` | `0.9766` | `0.9439` | `0.9330` | `0.8129` | `0.9998` | `0.9738` | Good buffer, weak policy transfer. |
| `stage3_awac_iql_bc015_20260613T0648Z` | `epoch_0-step_665` | `0.860100` | `0.9768` | `0.9435` | `0.9333` | `0.8127` | `0.9996` | `0.9738` | Similar to dualhost. |
| `stage3_awac_iql_pref_rank_ttc_20260613T112709Z` | `epoch_0-step_665` | `0.850661` | `0.9756` | `0.9341` | `0.9332` | `0.8033` | `0.9997` | `0.9719` | Preference/rank additions did not fix transfer. |
| `stage3_awac_blend035_lownoise_dpo8gpu_20260613T164458Z` | `epoch_0-step_665` | `0.833381` | `0.9682` | `0.9200` | `0.9136` | `0.8032` | `0.9988` | `0.9752` | Lower-noise/blend DPO still below target. |
| `stage3_awac_gapdpo_hybrid4gpu_v2_20260613T155350Z` | `epoch_0-step_665` | `0.798205` | `0.9339` | `0.9219` | `0.8667` | `0.7749` | `0.9977` | `0.9741` | Hybrid was not mature enough to validate DPO. |
| `stage3_awac_gapdpo_lowvalid_2gpu_20260613T155703Z` | `epoch_0-step_665` | `0.784474` | `0.9213` | `0.9208` | `0.8375` | `0.7748` | `0.9997` | `0.9722` | Poor safety transfer. |
| `stage3_awac_lownoise_dpo3gpu_20260613T163215Z` | `epoch_1-step_1184` | `0.780348` | `0.9315` | `0.9072` | `0.8492` | `0.7730` | `0.9908` | `0.9670` | More steps helped this bad config but remained far below GRPO. |
| `stage3_awac_warmup_absddc_blend025_lownoise_online_20260613T183113Z` | `epoch_0-step_665` | `0.862299` | `0.9825` | `0.9449` | `0.9444` | `0.8044` | `0.9996` | `0.9786` | Strongest AWAC online warmup so far; still far below GRPO. |

Lessons:
- The buffer discovery problem is not the main blocker.
- The blocker is policy absorption: high-PDMS candidates did not become high-probability diffusion samples.
- AWAC runs are confounded by LR and effective steps, so they do not prove AWAC/IQL is impossible. They do prove our implementation was not yet a mature, reliable offline-RL post-training implementation.
- Pure weighted denoising toward elite targets can overfit target likelihood without improving final sampling distribution.

Do not resume pure AWAC/IQL until:
- LR/effective-step controls are matched against GRPO.
- DPO/preference diagnostics are complete.
- The implementation is aligned with a mature offline diffusion policy method rather than a simple weighted-regression variant.

### 2. Diffusion-DPO / Preference Attempts Inside AWAC

Reference implementation checked:
- Diffusion-DPO official repo: https://github.com/SalesforceAIResearch/DiffusionDPO

Mechanism to preserve:
- Winner and loser are paired in the same batch order.
- Current and reference model evaluate the same noisy latents/timesteps.
- Loss uses the difference of current MSE gap and reference MSE gap.
- Logs include implicit accuracy and current/reference MSE.

Local status:
- Local `_compute_diffusion_dpo_preference_loss` uses shared noise/timestep and a frozen reference model when configured.
- The loss sign is broadly consistent with Diffusion-DPO's MSE-difference formulation.
- Diagnostics were completed on 2026-06-14: implicit accuracy, winner/loser current MSE, winner/loser reference MSE, current/reference margins, and logit scale are returned and logged.
- DPO currently lives in the AWAC path, not the main GRPO path.

Lesson:
- Existing poor DPO results should not be interpreted as "preference learning does not work." They indicate our DPO use was not yet a fully instrumented, mature preference-training experiment.

Before next preference run:
- Validate active pair ratio and reward-gap distribution.
- Calibrate beta against observed current/reference logratio magnitude.
- Consider preference as a small auxiliary to GRPO only after diagnostics show it is active and correctly ordered.

### 3. Safe DiffGRPO / GRPO Reference-KL Path

Motivation:
- Directly optimize sampled trajectories with train PDMS reward.
- Avoid treating GT as optimal.
- Preserve behavior through BC and reference KL.

Known evidence:
- Historical Safe DiffGRPO reached `0.906184`.
- User-reported original local Stage3 RL reached `0.9055` at `epoch9-step13300` with original LR `1e-4`.
- Current `2e-4` run is not a pure reproduction; compare it against the zt3 `1e-4` control.

Implementation notes:
- Current GRPO samples `G=16` trajectories per scene and computes group advantages.
- It uses hard safety masks and asymmetric safe advantage.
- The main `2e-4` run uses reference KL `0.02` and BC anneal `0.10 -> 0.05`.
- The non-GSPO path is effectively single-step trajectory-level policy gradient with current-policy sampling.

Reference gap:
- DDPO, DPPO, and RIPT-VLA all emphasize storing old logprobs/rollouts and applying PPO-style clipped updates over collected samples.
- Our current non-GSPO path does not replay a sampled batch for multiple PPO epochs/minibatches.
- The GSPO variant is closer because it can sample from a behavior policy and use a ratio, but it still needs diagnostics to confirm ratio/clip behavior and policy absorption.

### 3.1 GSPO Advantage-Control Diagnostic

Run:
- `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z`

Config:
- LR `1e-4`
- `sample_time=16`
- `batch_size=2`, `accumulate_grad_batches=8`, 4 GPUs
- `limit_train_batches=40`, `limit_val_batches=0`
- `log_every_n_steps=1`
- `grpo_use_gspo_ratio=true`
- `grpo_behavior_policy_sample=true`
- `grpo_behavior_policy_sync_interval=100000`
- `grpo_normalize_advantage_batch=true`
- `grpo_advantage_clip_abs=3.0`
- no buffer guidance, no AWAC, no DPO

Observed step diagnostics over 5 optimizer steps:
- `gspo_ratio_mean`: `1.0000 -> 0.9844`
- `gspo_ratio_min/max` at last step: `0.9649 / 1.0017`
- `gspo_ratio_clip_frac`: max observed `0.2266`, last `0.0156`
- `gspo_log_ratio_std`: last `0.00875`
- `gspo_abs_log_ratio_mean`: last `0.01625`
- `gspo_approx_kl`: last `0.000203`
- `grpo_advantage_batch_normalized`: `1.0`
- `grpo_advantage_std_after_transform`: near `0.96-1.00`
- `grpo_advantage_clip_frac`: `0.0-0.0234`
- `sampled_from_behavior_policy`: `1.0`
- `behavior_policy_synced`: `0.0` after launch, as intended for the long sync interval

Decision:
- Keep these code/logging changes. They prove the GSPO ratio path and advantage transform are active.
- Do not claim this is SOTA PPO yet. It is still a single-pass Lightning training loop, not DDPO/DPPO/RIPT-VLA-style rollout replay with fixed old logprobs over multiple inner epochs.
- The next full run may use these controls as a clean GSPO baseline only if we explicitly label it as "GSPO single-pass", not as full DPPO.

### 3.2 GRPO PPO Replay Smoke Diagnostic

Run:
- `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`

Reference alignment:
- DDPO, DPPO, and RIPT-VLA all use collected rollouts or old log-probabilities with PPO-style clipped updates instead of one immediate reward-weighted regression step.
- This run is the first local path that performs fixed-rollout replay in the Lightning training loop with manual optimizer steps.

Config:
- LR `1e-4`
- `stage3_objective=grpo_replay`
- `sample_time=4`
- `batch_size=2`, `accumulate_grad_batches=1`, 4 GPUs
- `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`
- `grpo_use_gspo_ratio=true`
- `grpo_normalize_advantage_batch=true`
- `grpo_advantage_clip_abs=3.0`
- `ppo_replay_inner_epochs=2`
- `ppo_replay_minibatch_size=4`
- `ppo_replay_sync_behavior_each_batch=true`
- `ppo_replay_bc_update=true`

Observed diagnostics:
- Checkpoint written: `epoch=0-step=200.ckpt`.
- `ppo_replay_optimizer_steps`: `5` per batch, matching `2` inner epochs over `8` rollout rows with minibatch `4`, plus one BC trust-region update.
- `ppo_replay_valid_ratio`: epoch mean `0.9852`, last step `0.8750`.
- `ppo_replay_loss_active`: `1.0` for all 40 logged batches.
- `gspo_ratio_mean`: epoch mean `0.9908`, last step `0.9886`.
- `gspo_ratio_min/max`: epoch min/max around `0.9823 / 0.9981`; last step `0.9731 / 0.9985`.
- `gspo_ratio_clip_frac`: `0.0` throughout.
- `gspo_approx_kl`: epoch mean `8.66e-5`, last step `1.13e-4`.
- `grpo_advantage_std_after_transform`: `1.0`, confirming batch advantage normalization.
- `grpo_advantage_zero_ratio`: epoch mean `0.00625`.
- Rollout component means were finite; last-step `mean_nc=0.8438`, `mean_dac=0.8125`, `mean_ttc=0.7188`, `mean_ddc=0.9844`, `safe_ratio=0.6875`.

Decision:
- Keep the implementation. It passes the minimal maturity smoke for PPO replay plumbing.
- Do not evaluate this checkpoint as an algorithm result; `sample_time=4` and 40 batches are intentionally too small.
- Do not launch full training directly. The next meaningful experiment is a controlled 1-epoch PPO replay diagnostic with `sample_time=16`, original LR `1e-4`, and the same safety/BC/reference-KL settings as the GRPO control.

Failure criteria for the next replay run:
- If `ppo_replay_valid_ratio` falls below `0.8` for sustained windows, stop and inspect group sampling.
- If `gspo_ratio_clip_frac` remains exactly `0` while reward does not move, the replay update may be too weak; increase inner epochs or replay minibatch exposure before full training.
- If `gspo_approx_kl` spikes or DDC/TTC means regress, reduce inner epochs or tighten reference/BC before continuing.

### 4. GRPO Buffer-Guided Self-Imitation

Motivation:
- The elite buffer contains useful trajectories, but AWAC did not make the diffusion sampler output them.
- GRPO can only reinforce trajectories it samples. It cannot directly learn unseen buffer trajectories unless buffer knowledge enters through reward shaping, distillation, or preference losses.

Current implementation:
- Buffer reward-neighborhood bonus, only for hard-safe samples.
- Low-noise denoising distillation on valid elite-buffer targets.
- Self-imitation denoising loss on current GRPO samples that are hard-safe and above a baseline.

Invalidated run:
- `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` was stopped on 2026-06-14 because buffer/self-imitation diagnostics were not logged by Lightning.

Next planned run:
- `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z`.
- Treat the first 50-100 steps as a go/no-go diagnostic check before letting it reach a full checkpoint.
- Evaluation watchers are disabled during the diagnostic window to avoid using GPUs occupied by unrelated tasks.

Required diagnostics:
- `grpo_buffer_guidance_target_ratio`
- `grpo_buffer_reward_bonus_mean/max`
- `grpo_buffer_target_distance_mean`
- `grpo_buffer_distill_weight_sum`
- `grpo_self_imitation_candidate_ratio`
- `grpo_self_imitation_target_ratio`
- `grpo_self_imitation_target_reward_mean/max`
- `gspo_ratio_mean`
- `gspo_ratio_clip_frac`

Failure criteria:
- If buffer target ratio is high but target distance does not fall, the reward bonus is not shaping exploration.
- If distill/self-imitation weight sum is zero or target ratio is near zero, the absorption channel is inactive.
- If DDC/TTC regress, stop the variant even if raw PDMS or EP improves.

## External Reference Findings

### 2026-06-14 Reference Audit Update

Checked sources:
- DDPO paper: https://arxiv.org/abs/2305.13301
- DPPO official implementation: https://github.com/irom-princeton/dppo
- RIPT-VLA paper and code: https://arxiv.org/abs/2505.17016 and https://github.com/Ariostgx/ript-vla

Concrete requirements borrowed from these references:
- DDPO frames denoising as a multi-step decision-making process, so a mature diffusion RL run should reason about denoising-step likelihoods or stored denoising-chain probabilities, not only final-trajectory weighted regression.
- DPPO's official implementation exposes PPO-specific diffusion code and configs such as clipping coefficient, denoising discount, likelihood standard deviation guards, and a CleanRL-style PPO training loop. A Stage3 run should not be called DPPO-equivalent unless it has fixed old logprobs and replay-style PPO updates.
- RIPT-VLA uses VLA log-prob interfaces, rollout generation, leave-one-out advantages, PPO updates, and dynamic sampling/filtering of zero-advantage groups. This directly matches our observed failure mode: many sampled groups become low-information or unsafe, so group composition and non-zero advantage coverage must be first-class diagnostics.

Decision:
- Do not launch another full single-pass GSPO run as the next main algorithm experiment. The short GSPO diagnostic remains useful, but it is not a mature SOTA implementation.
- The next main implementation target is a full PPO replay path aligned with DDPO/DPPO/RIPT-VLA: rollout collection, fixed old logprobs, advantage normalization/clipping, minibatch replay for multiple inner epochs, ratio clipping, approximate KL, and non-zero-advantage group filtering.

### Diffusion-DPO

Sources:
- Official implementation: https://github.com/SalesforceAIResearch/DiffusionDPO

Core mechanism:
- Shared noise/timestep current-vs-reference MSE gap on winner/loser pairs.
- Implicit accuracy and MSE diagnostics are first-class training signals.

Local gap:
- Add missing diagnostics before using DPO results to make algorithm decisions.

### DDPO

Sources:
- Paper: https://arxiv.org/abs/2305.13301
- Official implementation: https://github.com/kvablack/ddpo-pytorch

Core mechanism:
- Collect sampled trajectories and old denoising logprobs.
- Compute rewards, normalize advantages, and train with PPO ratio clipping over diffusion timesteps.
- Use inner epochs/minibatches over collected samples.
- Log approximate KL and clip fraction.

Local gap:
- Current GRPO mostly updates immediately per batch, without a stored rollout buffer and multi-epoch PPO replay in the non-GSPO path.

### DPPO

Sources:
- Paper: https://arxiv.org/abs/2409.00588
- Official implementation: https://github.com/irom-princeton/dppo

Core mechanism:
- Treat denoising as a diffusion MDP.
- Store chains and old logprobs.
- Optimize PPO clipped loss over denoising steps.
- Normalize and clip advantages.
- Use multi-epoch minibatch updates.

Local gap:
- We use trajectory-level reduced logprob and no critic. That may be acceptable only if justified, but it is not a complete DPPO reproduction.

### RIPT-VLA

Sources:
- Official implementation: https://github.com/Ariostgx/ript-vla

Core mechanism:
- K-rollout sampling per context.
- Leave-one-out advantage estimation.
- Dynamic sampling to avoid uninformative all-success/all-failure groups.
- Multiple PPO epochs over rollout data.
- Explicit valid masks and rollout statistics.

Local gap:
- Our group advantage resembles RLOO, but we do not yet have dynamic sampling or full PPO replay in the main Stage3 loop.

## Next Mature Directions

### Direction A: PPO/GSPO-Complete Stage3 GRPO

Hypothesis:
- A mature PPO-style update over collected diffusion chains may absorb reward gradients more reliably than one immediate policy-gradient step.

Must include:
- Collect chains, trajectories, rewards, components, old logprobs, and group advantages.
- Rebatch and replay for `num_inner_epochs > 1`.
- Ratio clipping, approximate KL, clip fraction, advantage normalization/clipping.
- Denoising-step-aware weighting.
- Reference KL and BC trust region.

Do not launch if:
- It only toggles `use_gspo_ratio` without rollout replay and diagnostics.

Implementation hardening completed on 2026-06-14:
- Added GRPO/GSPO diagnostics required by mature PPO-style methods:
  - `gspo_ratio_min/max`
  - `gspo_abs_log_ratio_mean`
  - `gspo_approx_kl`
  - `gspo_reverse_approx_kl`
  - group-weight and safe-count statistics
  - advantage mean/std/min/max before and after optional transforms
  - advantage positive/zero/clip ratios
- Added optional `grpo_normalize_advantage_batch` and `grpo_advantage_clip_abs`.
- Defaults preserve existing behavior; these controls are for the next controlled run after the current diagnostics are read.

Remaining gap before claiming a SOTA-level PPO implementation:
- The current Lightning path still does not store rollout batches and replay them for multiple PPO epochs/minibatches.
- A full DPPO/DDPO/RIPT-VLA-aligned implementation needs a rollout buffer containing trajectories, denoising chains, old logprobs, rewards, components, masks, and advantages, followed by replay updates with fixed old logprobs.

## Revised Experiment Plan 2026-06-14

### Phase 0: Stop Low-Value Runs And Preserve Evidence

Keep only experiments that answer a specific question:
- Local `2e-4` GRPO: stopped after `epoch_0-step_1330`. Evaluate the checkpoint, but do not continue training.
- zt3 `1e-4` GRPO: keep as the original-LR control.
- zt2 `1e-4` GSPO/advantage-control diagnostic: completed. Use it only as evidence that instrumentation and single-pass GSPO ratio are active.

Stop or avoid:
- Pure AWAC/IQL full training.
- Any preference/DPO run without pair count, active row ratio, logit scale, implicit accuracy, and current/reference winner-loser losses.
- Any GRPO variant that lacks ratio/KL/clip/advantage diagnostics.
- Any launch where LR, effective batch, sample_time, or checkpoint cadence differs from the control without being the explicit variable under test.

### Phase 1: Finish Evidence Collection

Required outputs:
- Full or shard-aggregated navtest eval for local `2e-4` `epoch_0-step_1330`: completed, PDMS `0.872059`.
- First checkpoint/eval for zt3 `1e-4` GRPO.
- Store zt2 diagnostic scalars in this ledger and compare with any future GSPO run.
- Store local PPO replay smoke scalars in this ledger: completed.

Go/no-go:
- Local `2e-4` is below Safe DiffGRPO by about `0.0341` PDMS at epoch0; do not revisit `2e-4` as a default LR.
- If the zt3 `1e-4` control is competitive with Safe DiffGRPO, use `1e-4` as the default for algorithm tests.
- If zt3 `1e-4` is also weak, inspect data/eval parity before modifying algorithms.

Diagnostic launch rule:
- Limited diagnostics must set `LOG_EVERY_N_STEPS=1`.
- If testing GSPO ratio behavior, use a behavior-policy sync interval larger than the diagnostic window, e.g. `GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL=100000`, because the first forward syncs the behavior policy and ratio is expected to be near `1.0` before the first optimizer update.
- Use at least `2 * accumulate_grad_batches` train batches so there are logged batches after one optimizer step.

### Phase 2: Do Not Promote Single-Pass GSPO To A Main Full Run

Allowed use:
- GSPO/GRPO with original LR `1e-4`, `sample_time=16`, same effective batch as the control.
- Enable batch advantage normalization and conservative fixed clipping:
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
- Keep BC anneal `0.10 -> 0.05`, reference KL `0.02`, scheduler `20` epochs to `1e-5`.
- Do not add buffer guidance in this run. This isolates the PPO/advantage-control effect.
- Limit this to a short diagnostic or ablation unless Phase 1 evidence shows the current implementation is already competitive and stable.

Rationale:
- DDPO/DPPO/RIPT-VLA-style methods all rely on stable old-policy ratio updates and controlled advantages.
- The completed short GSPO diagnostic proves the ratio path and advantage transforms are active, but it does not prove that the training loop has enough policy-improvement structure to absorb high-reward trajectories.

Interpretation limit:
- This run can validate whether single-pass GSPO improves over current GRPO.
- It cannot validate full DPPO/DDPO/RIPT-VLA until rollout replay exists.

### Phase 3: Prove PPO Replay At Near-Full GRPO Sampling Before Heavy Buffer Absorption

Status:
- A trajectory-level PPO replay path now exists and passed a `sample_time=4` smoke test.
- It stores sampled trajectories, denoising chains, old reduced logprobs, rewards, components, safety masks, group ids, group advantages, and optional old-policy BC chains.
- It replays fixed rollouts for multiple inner epochs/minibatches with PPO clipping, reference KL, and manual optimizer steps.

Required before the next main full-training launch:
- Run a near-full-sampling diagnostic with `sample_time=16`, LR `1e-4`, short duration, and `LOG_EVERY_N_STEPS=1`.
- Compare replay diagnostics against zt3 original-LR GRPO control:
  - reward/base_reward trend,
  - valid rollout ratio,
  - mixed/all-safe/all-unsafe group ratios,
  - ratio mean/min/max,
  - clip fraction,
  - approximate KL,
  - NC/DAC/TTC/DDC means.
- Add dynamic group filtering or resampling only if the diagnostic shows too many all-safe/all-unsafe or near-zero-advantage groups.
- Add per-denoising-step PPO only if trajectory-level replay shows a useful signal but insufficient policy absorption; do not add it preemptively because it increases memory and implementation risk.

This is the first implementation level that can reasonably be compared to DDPO/DPPO/RIPT-VLA.

Implementation shape in this codebase:
- Keep the VLM backbone and DiT architecture unchanged.
- Add a new Stage3 objective/path rather than mutating the current control path in place, e.g. `stage3_objective=grpo_replay` or an explicit `grpo_ppo_replay_enabled=true`.
- Reuse existing `sample_chain`, `get_logprobs`, `_reduce_chain_logprobs`, `_compose_stage3_reward`, and `_compute_stage3_advantages` so reward and logprob semantics stay identical to the current code.
- Move the extra PPO update logic into the Lightning training layer only if multiple optimizer steps over the same fixed rollout are required. Summing several replay losses inside a single forward pass is not equivalent to PPO replay because parameters do not change between inner epochs.
- Start with trajectory-level reduced logprob replay, because current code already supports chain logprob reduction. Only add per-step denoising PPO if the trajectory-level replay diagnostic is stable and if memory allows storing enough denoising-chain state.

PPO replay smoke-test acceptance before a full run:
- Can collect a rollout mini-buffer with shapes `[B, G, K, H, D]` or an explicitly documented equivalent: passed for trajectory-level replay.
- Old logprob stays fixed across replay epochs; new logprob changes after optimizer steps: passed by finite ratio/KL movement.
- Ratio mean begins near `1.0`, clip fraction becomes non-zero only after updates, and approximate KL remains finite: partially passed. KL is finite, but clip fraction stayed `0.0`, so the first full-sampling diagnostic should test whether the update is too conservative.
- Advantage rows with no useful group signal are filtered or down-weighted and logged: partially passed. Zero-advantage ratio is logged and low in the smoke, but dynamic resampling is not implemented.
- NC/DAC/TTC/DDC component means are logged for rollout and replay batches: passed for rollout components.
- A diagnostic run with `limit_train_batches<=40` logs all replay diagnostics at `LOG_EVERY_N_STEPS=1`: passed.

Only after this smoke test passes:
- Launch a limited 1-epoch replay run with checkpoint eval.
- Launch full 20-epoch replay only if the 1-epoch eval is not below the `1e-4` control by more than normal eval noise and does not degrade DDC/TTC.

Planned near-full diagnostic:
- Run name pattern: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_*`.
- Purpose: test the PPO replay implementation under the same group sampling scale as the historical GRPO runs before committing to full training.
- Config:
  - LR `1e-4`
  - `stage3_objective=grpo_replay`
  - `sample_time=16`
  - `batch_size=2`
  - 8 local GPUs
  - `max_epochs=1`
  - `limit_train_batches=40`
  - `limit_val_batches=0`
  - `log_every_n_steps=1`
  - BC anneal `0.10 -> 0.05`
  - reference KL `0.02`
  - scheduler horizon `20` epochs, min LR `1e-5`
  - PPO replay inner epochs `2`
  - PPO replay minibatch size `8`
  - max grad norm `1.0`
  - no elite buffer, no AWAC, no DPO
- Expected signs:
  - `ppo_replay_valid_ratio >= 0.8`.
  - `gspo_approx_kl` finite and small.
  - `gspo_ratio_mean` near `1.0` but not exactly constant.
  - `gspo_ratio_clip_frac` may become non-zero after several replay updates; if it stays zero, the update is conservative and should be tuned before full training.
  - NC/DAC/TTC/DDC component means must remain comparable to the GRPO control windows.
- Stop criteria:
  - NaN/inf in loss, ratio, KL, or reward.
  - Sustained `ppo_replay_loss_active=0`.
  - Severe DDC/TTC regression in rollout components.
  - Out-of-memory at `sample_time=16`; retry with the same algorithm and smaller per-GPU batch rather than changing the objective.

Near-full diagnostic result:
- Run: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`.
- Status: completed 40/40 train batches in about 3 minutes and wrote `epoch=0-step=360.ckpt`.
- `ppo_replay_optimizer_steps`: `9` per batch, matching `2` inner epochs over 32 rollout rows with minibatch `8`, plus one BC update.
- `ppo_replay_valid_ratio`: `1.0` for all logged batches.
- `ppo_replay_loss_active`: `1.0` for all logged batches.
- `gspo_ratio_mean`: epoch `0.9746`; step range `0.9436 - 0.9890`.
- `gspo_ratio_clip_frac`: epoch `0.1020`; step range `0.0 - 0.5`.
- `gspo_approx_kl`: epoch `5.38e-4`; step max `0.00223`.
- Advantage normalization/clipping was active: epoch `grpo_advantage_std_after_transform=0.9494`, `grpo_advantage_clip_frac=0.0135`, `zero_ratio=0.0`.
- Group composition was informative: epoch `mixed_group_ratio=0.6469`, `all_safe_group_ratio=0.3438`, `all_unsafe_group_ratio=0.0094`.
- Safety/submetric rollout means: `NC=0.9312`, `DAC=0.8471`, `TTC=0.8032`, `DDC=0.9555`, `safe_ratio=0.7863`.
- Decision: launch a controlled 1-epoch PPO replay training run with the same algorithm settings and epoch-end PDMS evaluation. Do not launch full 20-epoch replay until the 1-epoch checkpoint evaluation is competitive and does not regress DDC/TTC.

Buffer absorption should then be added as a controlled auxiliary:
1. reward-neighborhood bonus only
2. buffer low-noise distillation only
3. self-imitation from high-reward sampled trajectories only
4. preference/DPO auxiliary only after active-pair diagnostics pass
5. combined, only if individual terms are active and non-regressive

Buffer absorption success criteria:
- `grpo_buffer_target_distance_mean` decreases or remains low while reward improves.
- self-imitation target reward is above group/buffer baseline.
- NC/DAC/TTC/DDC do not regress.
- Evaluation PDMS improves over both `1e-4` GRPO control and Safe DiffGRPO `0.906184`.

### Phase 4: Preference/DPO Only As A Mature Auxiliary

Do not run preference/DPO as a standalone replacement again until:
- winner/loser active pair ratio is high enough,
- reward-gap distribution is logged,
- current/reference winner-loser MSE margins are logged,
- implicit accuracy is not saturated or random,
- beta/logit scale is calibrated from observed logratio magnitude,
- shared timestep/noise and reference model parity are verified.

Preferred use:
- Add small preference auxiliary to a stable GRPO/GSPO run, not to pure AWAC.
- Pair buffer valid-best against GT/IL/current low-reward samples only when DDC/TTC/NC/DAC guards pass.

### Direction B: GRPO + Buffer Absorption With Proof Of Activity

Hypothesis:
- Buffer trajectories help only when sampled trajectories are close enough for reward bonus or distillation to be active.

Must include:
- Target coverage and target distance diagnostics.
- Hard-safe gating.
- Low-noise denoising target schedule.
- Ablation against same LR/effective steps without buffer.

Do not increase weights until:
- `grpo_buffer_target_distance_mean`, target ratios, and self-imitation ratios show the auxiliary loss is active.

### Direction C: Proper Diffusion-DPO Auxiliary

Hypothesis:
- Preference learning can help rank valid high-PDMS buffer trajectories above GT/IL/current low-PDMS samples, but only if the implementation matches Diffusion-DPO mechanics.

Must include:
- Shared noise/timestep current/reference loss.
- Implicit accuracy.
- Winner/loser current and reference MSE.
- Pair-gap distribution and active row ratio.
- Beta calibration from observed logratio scale.
- Evaluation as an auxiliary to GRPO first, not as a replacement.

Do not use as evidence if:
- Pair count is low, implicit accuracy is unlogged, or beta/logits saturate.

### Direction D: Revisit AWAC/IQL Only As A Full Offline Diffusion RL Method

Hypothesis:
- AWAC/IQL may still work, but our previous version was closer to weighted target regression than a mature offline diffusion RL implementation.

Must include:
- Matched LR/effective optimizer steps.
- Explicit behavior support/trust-region checks.
- Diagnostics proving selected targets become more likely under the diffusion sampler.
- No invalid candidate training by default.

Do not prioritize this over GRPO until:
- Current GRPO LR control and buffer-guided run are evaluated.

## Active Questions

- Does `2e-4` improve GRPO over original `1e-4`, or only change early training dynamics?
- Does GSPO ratio/clipping improve stability compared with current-policy single-step GRPO?
- Do buffer-guided auxiliary terms actually activate, or are they mostly zero/too weak?
- Does the model's denoising likelihood on elite targets improve without navtest PDMS improving? If yes, the issue is sampler distribution or target mismatch.
- Are DDC/TTC protected during progress improvements, or are gains coming from reward-hacking side effects?

## Current Run Decisions

As of 2026-06-14 05:18 UTC:

- Keep running: `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`.
  - Reason: this is the needed original-LR GRPO control. It isolates algorithm effects from the failed/weak `2e-4` route.
  - Latest status: still running, no checkpoint yet.
  - Evaluation watchers already exist; do not start another duplicate watcher:
    - primary zt3 2GPU watcher: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z/primary_watch_on_rl_zt3_2gpu`
    - secondary zt2 4GPU watcher: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z/secondary_watch_on_vla_zt2_4gpu`
  - Latest logged scalar window: step `49 -> 349`, reward `0.50699 -> 0.88152`, base reward `0.61353 -> 0.84410`, TTC `0.80208 -> 0.88542`, safe ratio `0.72917 -> 0.98958`, GSPO ratio stays `1.0` because this is the non-replay control path.
  - Decision: keep it. It is a meaningful control, not a stale or weakly configured experiment.
- Completed and keep as smoke evidence: `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`.
  - Reason: it validates the new PPO replay implementation path, but is intentionally too small for PDMS conclusions.
- Completed and promote to controlled 1-epoch training: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`.
  - Reason: the near-full sample-time diagnostic passed replay, ratio/KL, and safety logging gates.
- Running: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
  - Reason: this is the first controlled full-navtrain 1-epoch PPO replay run. It keeps LR, `sample_time`, BC, reference KL, and scheduler semantics aligned with the GRPO control, and changes only the replay update path.
  - Latest status: still running, no checkpoint yet.
  - Latest logged scalar window: step `19 -> 339`, `ppo_replay_valid_ratio=1.0` throughout, ratio mean `0.9282 - 0.9951`, latest ratio mean `0.96936`, latest clip fraction `0.2031`, max clip fraction `0.7344`, latest approx KL `7.87e-4`, max approx KL `0.00336`.
  - Latest rollout reward/submetrics include one weak batch: reward `0.34987`, base reward `0.52296`, NC `0.73438`, TTC `0.625`, DDC `0.77148`, safe ratio `0.61328`.
  - Decision: keep for now, but watch closely. The PPO ratio/KL are still within a small range, so this is not an implementation failure. If weak safety batches persist rather than fluctuate, stop before promoting this setting to a 20-epoch run.
  - Evaluation watcher: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z_navtest_exact_eval`.
- Running on zt2: `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`.
  - Reason: same algorithm and data path as the i2 run, but half the replay inner epochs. This isolates whether replay update strength is too aggressive.
  - zt2 GPUs were free; existing zt2 watchers/supervisors were not stopped.
  - Latest status: still running, no checkpoint yet.
  - Latest logged scalar window: step `19 -> 179`, `ppo_replay_valid_ratio=1.0` throughout, latest ratio mean `0.99916`, clip fraction `0.0`, approx KL `2.30e-6`, reward `0.63984`, base reward `0.69427`, safe ratio `0.82813`, DDC `0.91992`.
  - Decision: keep. This is the cleaner conservative replay-strength ablation and may be preferable if i2 continues to show weak safety windows.
  - Evaluation watcher: `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z_navtest_exact_eval`.
- Do not resume: pure AWAC/IQL and AWAC+DPO variants listed above.
  - Reason: buffer quality is good, but policy absorption failed; current variants are not mature enough to justify more full-scale compute.
- Do not resume: `2e-4` GRPO/buffer-guided variants as main evidence.
  - Reason: the exact epoch-0 evaluation from the `2e-4` path was low (`PDMS 0.872059`), and the LR confound is already known.
- Treat any new GRPO replay run as diagnostic until it has passed smoke, short-run training diagnostics, and at least one controlled checkpoint evaluation.

## 2026-06-14 Attempt: Stage3 GRPO PPO Replay Diagnostic

Motivation:
- Previous evidence: original/Safe DiffGRPO is the strongest confirmed family (`0.9055` historical, `0.906184` confirmed), while AWAC/IQL discovered better train candidates but did not make the diffusion sampler emit better trajectories.
- Failure mode targeted: weak policy absorption and single-update GRPO instability.
- Reference sources checked:
  - DDPO paper/code: https://arxiv.org/abs/2305.13301 and https://github.com/kvablack/ddpo-pytorch
  - DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo
  - RIPT-VLA code: https://github.com/Ariostgx/ript-vla
  - Diffusion-DPO code for preference mechanics, not used as the main path here: https://github.com/SalesforceAIResearch/DiffusionDPO
- Mechanism copied: collect a fixed rollout, store old trajectory logprob, compute group/leave-one-out style advantages, then replay the fixed rollout with clipped PPO ratios over multiple inner minibatch updates.

Implementation completeness:
- Required pieces present:
  - `stage3_objective=grpo_replay` objective switch.
  - Fixed rollout collection with sampled chains, trajectories, rewards, PDM submetrics, old trajectory logprob, and transformed advantages.
  - PPO clipped ratio loss against old logprob.
  - Multi-inner-epoch/minibatch manual optimization in Lightning.
  - Manual AMP-aware gradient clipping through `self.clip_gradients`; launcher sets `trainer.params.gradient_clip_val=null` for replay.
  - BC trust-region update after replay when `grpo_ppo_replay_bc_update=true`.
  - Logs for `ppo_replay_*`, `gspo_ratio_*`, `gspo_approx_kl`, NC/DAC/TTC/DDC/TLC, and advantage statistics.
- Known simplifications:
  - Uses trajectory-level reduced diffusion logprob, not per-denoising-step PPO yet.
  - No critic/value function.
  - No dynamic re-sampling for all-success/all-failure groups yet; current implementation filters near-zero advantages.
- Why acceptable now:
  - This is the minimum diagnostic needed before a full DPPO-style implementation. A bad result here cannot rule out DPPO/RIPT-VLA, but it can tell whether replay and old-logprob ratio mechanics are active in this codebase.

Smoke result:
- Command family: `STAGE3_OBJECTIVE=grpo_replay`, `LR=1e-4`, `GRPO_SAMPLE_TIME=2`, `GRPO_USE_GSPO_RATIO=true`, `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`, `GRPO_ADVANTAGE_CLIP_ABS=3.0`, `GRPO_PPO_REPLAY_INNER_EPOCHS=2`, `GRPO_PPO_REPLAY_MINIBATCH_SIZE=1`, `MAX_SCENES=8`, `LIMIT_TRAIN_BATCHES=1`.
- Output root: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_smoke_1gpu_nullclip_20260614T041722Z`.
- Result: passed one real training batch and saved `epoch=0-step=5.ckpt`.
- Key diagnostics:
  - `train/ppo_replay_optimizer_steps_step = 5.0` (2 samples x 2 inner epochs + 1 BC update)
  - `train/ppo_replay_valid_ratio_step = 1.0`
  - `train/ppo_replay_valid_count_step = 2.0`
  - `train/ppo_replay_minibatch_valid_ratio_step = 1.0`
  - `train/ppo_replay_loss_active_step = 1.0`
  - `train/gspo_ratio_mean_step = 0.979439`
  - `train/gspo_ratio_clip_frac_step = 0.0`
  - `train/gspo_approx_kl_step = 0.000214`
  - `train/grpo_advantage_std_after_transform_step = 1.0`
  - `train/mean_nc_step = 1.0`, `train/mean_dac_step = 1.0`, `train/mean_ttc_step = 1.0`
- Decision:
  - Implementation smoke is now meaningful enough for a short diagnostic run.
  - Do not launch full training yet. First run a short 1-epoch/limited-batch diagnostic and inspect replay ratio, KL, advantage activity, safe submetrics, and optimizer-step count.

## Revised Experiment Plan

This plan replaces "try another variant" with gated experiments. A run is meaningful only if it isolates one variable, has mature implementation pieces for the method family, and can be stopped on objective diagnostics before spending a full training cycle.

### Comparison Protocol

Do not compare checkpoints across unmatched training lengths as final evidence.

Rules:
- Short-run replay checkpoints, such as a 1-epoch run, may only be compared against same-epoch or same-step controls.
- The historical `0.9055` original Stage3 result was obtained at `epoch9-step13300`. It is a final-training target, not a fair comparator for a 1-epoch replay diagnostic.
- The confirmed `0.906184` Safe DiffGRPO result is `epoch_12-step_17290`; it is also a long-run target, not a short-run comparator.
- The user-reported original Stage3 early-training reference is already around `0.88+` PDMS at `epoch0-1`; use that as the short-run sanity gate until a locally aligned checkpoint/control is available.
- If an early replay/preference/offline-RL checkpoint is materially below `0.88` PDMS at a comparable epoch or step, treat it as a failed promotion gate unless another diagnostic proves the method needs more burn-in for a principled reason.
- A 1-epoch replay run can pass the promotion gate if it is competitive with the zt3 original-LR GRPO control at the same completed epoch and does not regress DDC/TTC.
- A method can only claim "better than original/Safe DiffGRPO" after running for comparable epochs/steps with the same navtest evaluation protocol.
- Because `run_training_recogdrive_rl.py` saves checkpoints every epoch via `ModelCheckpoint(save_top_k=-1, every_n_epochs=1, save_on_train_epoch_end=True)`, current active runs can be aligned by completed epoch once their first checkpoints appear.
- For replay objectives, same epoch is not always the same optimizer-step budget because inner PPO replay and BC updates increase Lightning `global_step`. Future replay/control launches should enable optional train-step checkpoints and compare both epoch-aligned and step-aligned PDMS.

### Checkpointing For Step-Aligned Comparisons

Patch added on 2026-06-14:
- `navsim/planning/script/run_training_recogdrive_rl.py` now supports optional extra step checkpoints through `checkpoint.every_n_train_steps`.
- `scripts/training/run_recogdrive_stage3_rl_2b_local.sh` exposes this as `CHECKPOINT_EVERY_N_TRAIN_STEPS`, while preserving the default epoch checkpoint path with `CHECKPOINT_EVERY_N_EPOCHS=1`.
- Step checkpoints are written under `step_checkpoints/` so existing epoch checkpoint watchers remain compatible.
- Default behavior is unchanged for the base launcher: no step checkpoints unless `CHECKPOINT_EVERY_N_TRAIN_STEPS > 0`.
- RLOO self-imitation launches now default to `CHECKPOINT_EVERY_N_TRAIN_STEPS=300` so new algorithm attempts can be screened before epoch0 when the original Stage3 early reference is already `0.88+` PDMS.
- `scripts/training/gate_recogdrive_stage3_early_pdms.py` reads watcher `checkpoint_eval_submetrics.tsv` files and returns a read-only `wait/continue/watch/stop` decision. Use it after the first step checkpoint eval; it does not kill processes automatically.

### Phase 0: Keep Only Clean Controls And Mature Diagnostics

Current keep list:
- zt3 original-LR GRPO control: `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`.
- local PPO replay i2 one-epoch run: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
- zt2 PPO replay i1 one-epoch run: `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`.
- zt3 PPO replay i1 step-checkpoint diagnostic: `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`.
- Existing zt3 control eval watchers: keep the already-running primary/secondary watchers; do not launch a third watcher for the same checkpoint stream.

Current stop/avoid list:
- Pure AWAC/IQL, AWAC+DPO, and simplified preference variants until implementation is upgraded to a mature offline/preference diffusion policy method.
- `2e-4` GRPO and buffer-guided variants as a default route, because the exact epoch-0 eval was `PDMS 0.872059`, below the original Stage3 early `0.88+` gate and weaker than matched epoch0 controls.
- Any run without ratio/KL/clip/advantage/safety diagnostics.

No new full training should start until at least one of the current one-epoch replay runs has an exact navtest checkpoint evaluation.

### Phase 1: Finish The Three Active Evidence Runs

Expected outputs:
- zt3 original-LR GRPO control checkpoint and exact PDMS/submetrics.
- local replay i2 epoch checkpoint and exact PDMS/submetrics.
- zt2 replay i1 epoch checkpoint and exact PDMS/submetrics.

Interpretation:
- If a comparable long-horizon `1e-4` control approaches the historical `0.9055` / Safe DiffGRPO `0.906184`, use it as the primary control for future final-horizon algorithm comparisons.
- If replay i1/i2 beats or matches zt3 while preserving DDC/TTC, promote replay to the next multi-epoch candidate.
- If replay i2 is worse but i1 is stable, reduce replay update strength rather than abandoning PPO replay.
- If both replay variants are clearly below the original-LR GRPO control, do not launch 20-epoch replay. Inspect whether trajectory-level reduced logprob is too coarse and move to denoising-step-level PPO before spending more full-run compute.

Hard stop criteria before epoch end:
- NaN/inf in loss, reward, logprob ratio, or KL.
- Sustained `ppo_replay_valid_ratio < 0.8`.
- Sustained high clip saturation, not just isolated windows.
- Repeated rollout safety collapse: NC/DAC/TTC/DDC or safe ratio stays far below control windows across several logs.

Current read:
- None of the three active runs meets a hard stop criterion.
- i2 is higher-update-strength and should be treated as riskier.
- i1 is the cleaner conservative ablation and should remain running even if i2 later looks weak.

### Phase 2: Decide Whether PPO Replay Merits Multi-Epoch Training

Only launch a 20-epoch PPO replay run if a one-epoch replay checkpoint is competitive enough to be informative:
- Not more than normal eval noise below zt3 original-LR GRPO.
- Does not regress DDC/TTC relative to the GRPO control.
- Replay diagnostics show active but controlled ratio movement: finite small KL, non-saturated clip fraction, and nonzero useful advantage coverage.

Default candidate if Phase 1 passes:
- LR `1e-4`, scheduler horizon `20` epochs down to `1e-5`.
- `sample_time=16`.
- BC anneal `0.10 -> 0.05`.
- reference KL `0.02`.
- `stage3_objective=grpo_replay`.
- Start from the better of i1 vs i2:
  - i1 if i2 has unstable safety/clip windows;
  - i2 only if it improves PDMS without DDC/TTC regression.

Do not change LR, sample_time, buffer loss, and replay strength simultaneously. One new variable per experiment.

### Phase 3: Upgrade Replay Toward Full DPPO/RIPT-VLA Before Calling It SOTA-Level

Trajectory-level replay is a necessary intermediate, not the final mature implementation. If Phase 1 shows promise but not enough PDMS:
- Add denoising-step-level old logprobs and PPO ratios instead of only reduced trajectory logprob.
- Add dynamic group filtering/resampling for all-success/all-failure or near-zero-advantage groups, matching the RIPT-VLA style failure mode.
- Consider a value/critic baseline only after advantage-normalized PPO replay still shows high variance.

This phase is implementation work first, not a blind full run. It needs smoke tests proving:
- old per-step logprobs stay fixed across replay epochs;
- current per-step logprobs change after optimizer steps;
- KL/clip are computed over the same denoising timesteps;
- memory fits at `sample_time=16` or a documented equivalent.

### Phase 4: Add Buffer Knowledge Only Through Proven Absorption Channels

The elite buffer is useful, but previous AWAC showed that good targets alone do not guarantee good sampled trajectories. Buffer work resumes only after the base replay path is stable.

Allowed buffer auxiliaries, one at a time:
- Reward-neighborhood bonus for sampled trajectories close to high-PDMS valid buffer targets.
- Low-noise denoising distillation toward valid elite targets.
- Self-imitation on current sampled trajectories that are hard-safe and above baseline.
- Preference/DPO auxiliary using shared noise/timestep and a frozen reference, only when active-pair diagnostics pass.

Required diagnostics:
- target coverage and target distance;
- distill/self-imitation weight sums;
- winner/loser active-pair ratio for DPO;
- current/reference winner-loser loss margins and implicit accuracy;
- DDC/TTC/NC/DAC protection.

Stop buffer variants if:
- target ratios are near zero;
- target distance does not improve;
- DDC/TTC regress;
- PDMS gains come only from EP while safety declines.

### Phase 5: Revisit AWAC/IQL Only As A Mature Offline Diffusion RL Method

Do not continue the existing AWAC/IQL line as-is. A future AWAC/IQL revisit must include:
- matched LR/effective optimizer steps against GRPO;
- diagnostics proving elite-target likelihood increases under the sampler;
- behavior-support/trust-region checks;
- strict valid-mask persistence and no invalid candidate training by default;
- preference or Q/value structure aligned with a reference implementation, not only weighted denoising regression.

Until then, AWAC results are evidence of a policy-absorption failure in our implementation, not evidence that offline RL or preference learning is inherently unsuitable.

## 2026-06-14 06:09 UTC Active Run Snapshot

No active run has produced a checkpoint yet, so there is no new PDMS result and no basis to claim a better method.

Current diagnostics:
- Local replay i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`:
  - Still running, no checkpoint.
  - Latest logged step `799`: reward `0.59714`, base reward `0.67684`, safe ratio `0.78516`, NC `0.84180`, DAC `0.95703`, TTC `0.67578`, DDC `0.95898`.
  - Replay mechanics remain active: `ppo_replay_valid_ratio=1.0`, `optimizer_steps=9`, ratio mean `0.96882`, clip fraction `0.04688`, approx KL `6.05e-4`.
  - Decision: keep running but do not promote i2 until epoch eval. Recent low reward/safety windows make it a watched diagnostic.
- zt2 replay i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`:
  - Still running, no checkpoint.
  - Latest logged step `759`: reward `0.87667`, base reward `0.85840`, safe ratio `0.94922`, NC `0.98828`, DAC `0.96094`, TTC `0.92188`, DDC `0.95508`.
  - Conservative replay remains very stable: `ppo_replay_valid_ratio=1.0`, `optimizer_steps=5`, ratio mean `0.99749`, clip fraction `0.0`, approx KL `7.51e-6`.
  - Decision: keep running. This remains the cleaner replay-strength ablation.
- zt3 original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`:
  - Still running, no checkpoint.
  - Latest logged step `399`: reward `0.83706`, base reward `0.83057`, TTC `0.84375`, safe ratio `0.92708`.
  - Decision: keep running. It is slow but progressing and is still the necessary LR/control baseline.

Decision:
- Do not stop any of these three runs.
- Do not launch a new full training run yet.
- The step-level PPO replay implementation is ready for a short diagnostic, but it should wait until at least one current one-epoch replay checkpoint has exact PDMS unless an active run fails.

## 2026-06-14 06:14 UTC Step-Checkpoint Diagnostic Launch

Reason:
- Existing replay i1/i2 jobs were launched before step checkpoint support. They are healthy but will only produce checkpoints at epoch end.
- Replay objectives increment Lightning `global_step` by inner PPO/BC optimizer updates, so waiting for epoch-only checkpoints delays same-step diagnostics.
- The original Stage3 early reference is already around `0.88+` PDMS at `epoch0-1`; replay should be screened earlier before committing to long multi-epoch runs.

Launched on zt3:
- Run: `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`.
- GPUs: `6,7`; existing zt3 tasks were not stopped.
- Config: LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO replay epochs `1`, replay minibatch `8`, batch size `1`, 2 GPUs, `LIMIT_TRAIN_BATCHES=200`, `LIMIT_VAL_BATCHES=0`.
- Checkpointing: `CHECKPOINT_EVERY_N_TRAIN_STEPS=300`, `CHECKPOINT_EVERY_N_EPOCHS=1`.
- Eval watcher: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z_navtest_step_eval`, configured for recursive checkpoint discovery under the training Hydra root.

Expected use:
- Use the first step checkpoint as a quick PDMS sanity check only.
- Do not compare this limited-batch diagnostic as final evidence against original Stage3 `epoch9-step13300` or Safe DiffGRPO `epoch_12-step_17290`.
- If the step checkpoint is materially below the original early `0.88+` band, do not promote replay i1/i2 without fixing the method.
- If it is plausible, wait for the full one-epoch replay/control checkpoints for the real gate.

## 2026-06-14 06:24 UTC Active Run Snapshot

No PDMS result is available yet.

Current status:
- Local replay i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`:
  - Still running, no checkpoint.
  - Latest logged step `1079`: reward `0.70439`, base reward `0.72044`, safe ratio `0.89844`, NC `0.91406`, DAC `0.98438`, TTC `0.82422`, DDC `0.85938`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `9`, ratio clip fraction `0.15625`, approx KL `7.33e-4`.
- zt2 replay i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`:
  - Still running, no checkpoint.
  - Latest logged step `1119`: reward `0.80203`, base reward `0.80814`, safe ratio `0.90625`, NC `0.96680`, DAC `0.94141`, TTC `0.89062`, DDC `1.0`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `5`, ratio clip fraction `0.0`, approx KL `8.13e-6`.
- zt3 original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`:
  - Still running, no checkpoint.
  - Latest logged step `449`: reward `0.89311`, base reward `0.86898`, safe ratio `0.95833`, TTC `0.93750`.
- zt3 step-checkpoint replay diagnostic `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`:
  - `step-step=300.ckpt` was written at `2026-06-14T06:20:12Z`.
  - Watcher archived it as `step-step_300.ckpt` at `2026-06-14T06:22:59Z`.
  - Training completed successfully with return code `0` at `2026-06-14T06:24:39Z`.
  - Additional checkpoints were produced: `step-step=600.ckpt` and `epoch=0-step=600.ckpt`.
  - Eval of `step-step_300` started at `2026-06-14T06:24:59Z`; latest eval log shows progress through roughly `100 / 6069` navtest scenarios.
  - Latest logged step `79`: reward `0.22160`, base reward `0.44747`, safe ratio `0.5`, NC `0.95312`, DAC `0.5`, TTC `0.875`, DDC `0.76562`, replay valid ratio `0.5`.

Decision:
- Do not stop the full replay i1/i2 jobs before their first checkpoint.
- Do not judge the step diagnostic from its noisy training scalar windows; wait for exact navtest PDMS on `step-step_300.ckpt`.
- Let the watcher evaluate queued checkpoints in order; do not start duplicate evals for the same ckpts.
- If `step-step_300` is far below the original early `0.88+` PDMS band, replay needs method changes before promotion.

## 2026-06-14 06:46 UTC Active Run Snapshot

No new PDMS result is available yet.

Current status:
- Local replay i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`:
  - Still running, no checkpoint.
  - Latest TensorBoard scalar step `1379`: reward `0.69689`, base reward `0.74012`, safe ratio `0.83984`, NC `0.97070`, DAC `0.86719`, TTC `0.91016`, DDC `0.98047`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `9`, ratio clip fraction `0.0`, approx KL `1.54e-4`, reference KL loss `0.03040`, BC coeff `0.10`.
- zt2 replay i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`:
  - Still running, no checkpoint.
  - Latest TensorBoard scalar step `1519`: reward `0.68934`, base reward `0.73539`, safe ratio `0.87109`, NC `0.94141`, DAC `0.95312`, TTC `0.90234`, DDC `0.90234`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `5`, ratio clip fraction `0.0`, approx KL `1.20e-5`, reference KL loss `0.04611`, BC coeff `0.10`.
- zt3 original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`:
  - Still running, no checkpoint.
  - Latest TensorBoard scalar step `499`: reward `0.85184`, base reward `0.83066`, safe ratio `0.95833`, TTC `0.84375`, reference KL loss `0.00900`, BC coeff `0.10`.
- zt3 step-checkpoint replay diagnostic `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`:
  - `step-step_300` exact navtest eval is still running.
  - Latest eval log around `2026-06-14T06:43Z` shows progress through roughly `2067 / 6069` navtest scenarios.
  - `step-step_600` remains queued by the watcher.

Resource state:
- Local 8 A800 GPUs are occupied by replay i2 training.
- zt2 8 A800 GPUs are occupied by replay i1 training.
- zt3 is running the original-LR control plus the step-checkpoint eval; do not start duplicate evals or kill existing remote tasks.

Decision:
- Continue waiting for exact navtest PDMS. The replay training diagnostics show the objective is active and numerically stable, but recent safety/reward batches are not strong enough to promote without exact evaluation.
- Apply the `0.88+` original Stage3 `epoch0-1` sanity gate to the first comparable replay checkpoints.

## 2026-06-14 07:18 UTC Trajectory-Level Replay Gate Result

Exact navtest result:
- Run: `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`.
- Checkpoint: `step-step_300`.
- Eval completed at `2026-06-14T07:16:11Z` with return code `0`.
- Valid rows: `12138`.
- PDMS: `0.816440`.
- Submetrics: NC `0.966263`, DAC `0.914483`, TTC `0.908304`, EP `0.775452`, comfort `0.998517`, DDC `0.951598`, TLC not reported by this summary.

Interpretation:
- This is materially below the user-reported original Stage3 early `epoch0-1` sanity band of `0.88+`.
- It is also below the stopped high-LR `2e-4` epoch0 result `0.872059`.
- The limited-batch step checkpoint is not a final same-epoch comparison, but the gap is too large to justify promoting the trajectory-level replay implementation.

Action taken:
- Stopped the trajectory-level replay long jobs:
  - local i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
  - zt2 i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`.
- Stopped the failed replay checkpoint watcher/eval streams:
  - local i2 watcher.
  - zt2 i1 watcher.
  - zt3 replay step-checkpoint watcher and the queued `step-step_600` eval.
- Kept the zt3 original-LR GRPO control and its zt2/zt3 watchers running.

Decision:
- Do not run or promote more trajectory-level PPO replay as-is.
- Treat the failure as a method/objective issue, not a simple LR or epoch-count issue.
- Next diagnostic should use the already implemented denoising-step-level PPO replay mode, because DDPO/DPPO operate on denoising transitions rather than one reduced trajectory logprob.
- The next run must be an isolated same-scale diagnostic against the failed trajectory-level setup: same LR `1e-4`, `sample_time=16`, `LIMIT_TRAIN_BATCHES=200`, 2 GPUs, but `GRPO_PPO_REPLAY_LOGPROB_MODE=step`.

## 2026-06-14 Attempt: Step-Level PPO Replay Implementation

Motivation:
- Previous evidence: AWAC/IQL found high-PDMS train candidates but did not make the diffusion sampler output better trajectories. Trajectory-level PPO replay is now running, but it still reduces the full denoising chain into one scalar logprob.
- Targeted failure mode: weak policy absorption caused by a coarse trajectory-level objective. DDPO/DPPO treat denoising transitions as the RL action sequence, so the PPO ratio should be able to operate at the denoising-step level.
- Reference sources:
  - DDPO paper/code: https://arxiv.org/abs/2305.13301 and https://github.com/kvablack/ddpo-pytorch
  - DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo
  - RIPT-VLA code: https://github.com/Ariostgx/ript-vla

Implementation completed:
- Added optional `grpo_ppo_replay_logprob_mode` / `ppo_replay_logprob_mode`.
- Default is `trajectory`, preserving existing replay experiments.
- New `step` mode stores old per-denoising-step logprobs in the rollout and computes PPO ratios per denoising transition.
- Step-level surrogate is weighted by the existing normalized `gamma_denoising` discount.
- Logs `ppo_replay_step_logprob_mode` so runs can prove which path was active.
- No VLM, DiT architecture, trainable module, reward, or navtest-training path change.

Smoke test:
- Run: `stage3_grpo_replay_stepmode_smoke_1gpu_20260614T0530`
- Host/GPU: `training-rl-zt3`, `CUDA_VISIBLE_DEVICES=6`.
- Config: `MAX_SCENES=8`, `LIMIT_TRAIN_BATCHES=1`, `sample_time=2`, `BATCH_SIZE=1`, `GRPO_PPO_REPLAY_LOGPROB_MODE=step`, `GRPO_PPO_REPLAY_INNER_EPOCHS=1`, `GRPO_PPO_REPLAY_MINIBATCH_SIZE=2`.
- Result: completed one training batch without NaN/OOM.
- Key scalars:
  - `train/ppo_replay_step_logprob_mode_step = 1.0`
  - `train/ppo_replay_valid_ratio_step = 1.0`
  - `train/ppo_replay_optimizer_steps_step = 2.0`
  - `train/gspo_ratio_mean_step = 1.0`
  - `train/gspo_approx_kl_step = 0.0`
  - `train/reward_step = 0.91329`
  - `train/safe_ratio_step = 1.0`

Decision:
- This is an implementation maturity improvement, not a better-PDMS result yet.
- Do not launch a full step-level run immediately.
- First wait for the current trajectory-level replay i1/i2 one-epoch PDMS results and zt3 original-GRPO control.
- If trajectory-level replay is promising but not clearly better, run a short step-level diagnostic before any 20-epoch full run.
- If trajectory-level replay is already worse than the control, step-level PPO becomes the next implementation-focused diagnostic rather than another hyperparameter tweak.

### 2026-06-14 07:27 UTC Step-Level PPO Diagnostic Launch Plan

Reason:
- The trajectory-level replay gate failed with PDMS `0.816440` at `step-step_300`, which is far below the original Stage3 early `0.88+` band.
- This points to a coarse-objective absorption issue: reducing a diffusion rollout to one trajectory logprob is not aligned with DDPO/DPPO, which optimize denoising transitions.

Implementation fix before launch:
- `scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh` now forwards:
  - `GRPO_PPO_REPLAY_LOGPROB_MODE`
  - `GPUS`, `GPUS_PER_NODE`
  - `CHECKPOINT_EVERY_N_EPOCHS`, `CHECKPOINT_EVERY_N_TRAIN_STEPS`
  - `TRAINER_GRADIENT_CLIP_VAL`
- Dry-run verified that `agent.grpo_ppo_replay_logprob_mode=step`, `trainer.params.devices=2`, and `checkpoint.every_n_train_steps=300` reach the final Hydra command.

Planned isolated diagnostic:
- Run name: `stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z`.
- Training host/GPU: zt3 GPUs `6,7`.
- Eval watcher: local 2 GPUs watching the shared output directory.
- Config held equal to the failed trajectory-level diagnostic where possible:
  - LR `1e-4`
  - `stage3_objective=grpo_replay`
  - `sample_time=16`
  - `GRPO_PPO_REPLAY_INNER_EPOCHS=1`
  - `GRPO_PPO_REPLAY_MINIBATCH_SIZE=8`
  - `GRPO_USE_GSPO_RATIO=true`
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
  - `REFERENCE_KL_COEFF=0.02`
  - `REFERENCE_KL_CHUNK_SIZE=16`
  - `BATCH_SIZE=1`, `GPUS_PER_NODE=2`
  - `LIMIT_TRAIN_BATCHES=200`, `LIMIT_VAL_BATCHES=0`
  - step checkpoints every `300` train steps.
- Single intended variable: `GRPO_PPO_REPLAY_LOGPROB_MODE=step` instead of `trajectory`.

Promotion gate:
- User clarification on `2026-06-14`: original Stage3 is already around `0.88+` PDMS at `epoch0-1`; use that as the short-run comparator, not the long-horizon `0.9055` result. User clarification on `2026-06-15`: the `0.9055` result was obtained at `epoch9-step13300`.
- If `step-step_300` remains far below `0.88`, step-level replay alone is not enough and we should inspect reward/action-distribution mismatch or return to original GRPO with better reward shaping.
- If `step-step_300` materially improves over `0.816440`, continue evaluating queued `step-step_600` / epoch checkpoint and consider a larger run.

Launch status:
- Training launched on zt3 at `2026-06-14T07:29Z`.
- Output root: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z`.
- Verified resolved command contains `agent.grpo_ppo_replay_logprob_mode=step`, `trainer.params.devices=2`, and `checkpoint.every_n_train_steps=300`.
- Local watcher launched with output root `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z_navtest_step_eval`.
- Initial watcher state: no pending checkpoint, training state running.
- Training finished cleanly at `2026-06-14T07:40:13Z` with return code `0`.
- Produced checkpoints:
  - `step-step=300.ckpt` at `2026-06-14T07:35:47Z`
  - `step-step=600.ckpt` at `2026-06-14T07:39:29Z`
  - `epoch=0-step=600.ckpt` at `2026-06-14T07:39:42Z`
- The local watcher archived `step-step=300` and started exact navtest evaluation at `2026-06-14T07:39:45Z`; PDMS is pending.
- A second watcher was launched on `training-vla-zt2` at `2026-06-14T07:45:23Z` using GPUs `0-7`.
  - Eval root: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z_zt2_8gpu_step600_eval`.
  - It uses the local watcher summary as `EXTERNAL_SUMMARY_TSV`, so `step-step_300` is skipped there and `step-step_600` is evaluated first.
  - This keeps the early gate aligned: local `step300` checks the first update point; zt2 `step600` checks whether the method recovers by the end of the limited diagnostic.
- zt2 `step-step_600` exact navtest completed at `2026-06-14T07:59:19Z`, return code `0`.
  - PDMS `0.824322`, NC `0.975943`, DAC `0.913330`, TTC `0.927748`, EP `0.770388`, comfort `0.999753`, DDC `0.979980`, valid rows `12138`.
  - This is still far below the original Stage3 `epoch0-1` sanity gate of `0.88+`, so the current simplified step-level replay implementation fails the promotion gate.
  - zt2 then started the duplicate `epoch_0-step_600` eval automatically; it was stopped intentionally because the same update point was already evaluated.
- Local `step-step_300` exact navtest completed at `2026-06-14T08:27:05Z`, return code `0`.
  - PDMS `0.787823`, NC `0.958436`, DAC `0.900066`, TTC `0.893557`, EP `0.742667`, comfort `1.000000`, DDC `0.947067`, valid rows `12138`.
  - This is worse than both the failed trajectory-level replay and the later all-step `step600` checkpoint, so the simplified step-level replay is rejected.
  - The local watcher then started a duplicate `step-step_600` eval; it was stopped because zt2 had already completed `step-step_600`.

Implementation audit while evals are running:
- DPPO official PPO diffusion code clamps both old/new logprobs, normalizes/clips advantages, discounts by denoising step, and samples PPO minibatches across `(environment step, denoising step)` rather than only across trajectory rows.
- DPPO also uses step-dependent PPO clipping and an optional value/critic loss. We currently use a fixed clip range and no critic.
- Our `step` mode is therefore a meaningful improvement over trajectory-level replay, but it is still not a complete DPPO/RIPT-VLA-equivalent implementation.
- Since `step600` remains below the `0.88+` early gate, the next implementation target should be mature DPPO-style transition sampling and step-dependent clipping before any new LR or buffer-tuning run.

### 2026-06-14 Attempt: DPPO-Style Transition Replay Implementation

Motivation:
- The failed trajectory-level replay (`step300` PDMS `0.816440`) and simplified all-step replay (`step600` PDMS `0.824322`) both underperform the original Stage3 early `0.88+` PDMS band.
- The failure pattern suggests the issue is not just LR or epoch count: diffusion policy improvement needs to optimize denoising transitions in a way closer to DDPO/DPPO/RIPT-VLA instead of treating the whole sampled trajectory as one coarse action.

Reference sources:
- DDPO optimizes diffusion sampling trajectories using PPO-style likelihood ratios over denoising steps.
- DPPO samples replay minibatches over diffusion transitions, clips old/new logprobs, applies denoising-step discounting, and uses step-dependent PPO clipping.
- RIPT-VLA applies RL post-training to VLA/action-generation policies with PPO-style rollout replay rather than plain behavior cloning onto discovered trajectories.

Implementation completeness:
- Added `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=transition`, which changes the manual PPO replay minibatch unit from sampled trajectory rows to `(sample, denoising_step)` transitions when `GRPO_PPO_REPLAY_LOGPROB_MODE=step`.
- Added configurable step logprob clamp:
  - `GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN=-5.0`
  - `GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX=2.0`
- Added DPPO-style denoising-step clip schedule:
  - `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=dppo_exp`
  - `GRPO_PPO_REPLAY_STEP_CLIP_BASE=0.001`
  - `GRPO_PPO_REPLAY_STEP_CLIP_RATE=3.0`
- Kept backward compatibility:
  - default `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=trajectory_all_steps`
  - default `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=constant`
  - old trajectory and all-step replay behavior remain available.

Known simplifications:
- No critic/value head has been added.
- Reference KL still computes on selected rollout rows rather than only selected denoising transitions.
- This implementation is closer to DPPO than the previous replay, but still not a full DPPO reproduction.

Smoke test:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_dppo_transition_smoke2_20260614T082252Z`.
- Config: `MAX_SCENES=8`, `LIMIT_TRAIN_BATCHES=1`, `sample_time=2`, `BATCH_SIZE=1`, `LR=1e-4`, `REFERENCE_KL_COEFF=0`, transition minibatch size `4`.
- Result: completed one training batch without NaN/OOM/autograd error.
- Key TensorBoard scalars:
  - `train/ppo_replay_step_logprob_mode_step = 1.0`
  - `train/ppo_replay_step_minibatch_mode_step = 1.0`
  - `train/ppo_replay_transition_mode_step = 1.0`
  - `train/ppo_replay_optimizer_steps_step = 4.0`
  - `train/ppo_replay_step_clip_min_step = 0.003868`
  - `train/ppo_replay_step_clip_mean_step = 0.013330`
  - `train/ppo_replay_step_clip_max_step = 0.022791`
  - `train/ppo_replay_logprob_clamped_frac_step = 0.0`

Next diagnostic:
- Run a short 2-GPU transition replay diagnostic only after the smoke-tested code is committed.
- Use the original Stage3 `epoch0-1 ~= 0.88+` as the short-run gate.
- If the new transition replay still scores far below `0.88`, do not proceed to full training. Next step should be either:
  - complete DPPO with critic/value or stronger KL/ratio controls, or
  - revert to original GRPO baseline and add buffer-derived preference/self-imitation only after verifying the policy can absorb the signal.

Launch status:
- Code committed and pushed as `06c79ad Add DPPO-style transition replay for Stage3 GRPO`.
- Run name: `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z`.
- Training host/GPU: `training-vla-zt2`, GPUs `0,1`.
- Eval watcher: local 8-GPU watcher at `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_eval`.
- Config:
  - `MAX_EPOCHS=1`, `LIMIT_TRAIN_BATCHES=200`
  - `LR=1e-4`, `sample_time=16`
  - `GRPO_PPO_REPLAY_LOGPROB_MODE=step`
  - `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=transition`
  - `GRPO_PPO_REPLAY_MINIBATCH_SIZE=32`
  - `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=dppo_exp`
  - `REFERENCE_KL_COEFF=0.02`, `REFERENCE_KL_CHUNK_SIZE=16`
  - `GRPO_USE_GSPO_RATIO=true`
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
  - step checkpoints every `300` train steps.
- Fairness note:
  - `num_denoising_steps` is `4` in the smoke run. With `sample_time=16` and transition minibatch `32`, PPO replay should perform about two transition minibatches plus one BC update per train batch, matching the previous simplified step replay's optimizer-step cadence more closely than a smaller transition minibatch would.

Evaluation result:
- The original torchrun watcher launched for `step-step_300` lost its parent and left orphan ranks before scoring. The final reported numbers below use independent exact-PDMS token shards with no scoring formula changes:
  - `step-step_300`: local 8 independent 1GPU shards.
  - `step-step_600`: zt3 2 independent 1GPU shards on idle GPUs 6 and 7.
  - `epoch=0-step=800`: local 8 independent 1GPU shards.
- Full navtest exact PDMS results:

| Checkpoint | Eval root | PDMS | NC | DAC | TTC | EP | Comfort | DDC | TLC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step-step_300` | `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step300_20260614T0900Z` | `0.744740` | `0.954894` | `0.848163` | `0.878234` | `0.719376` | `0.996375` | `0.942289` |  |
| `step-step_600` | `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step600_zt3_2gpu_20260614T0858Z` | `0.801195` | `0.972524` | `0.891251` | `0.923875` | `0.752611` | `0.999423` | `0.947767` |  |
| `epoch=0-step=800` | `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_epoch0_step800_20260614T0912Z` | `0.831015` | `0.978003` | `0.912836` | `0.927583` | `0.786319` | `0.999423` | `0.961691` |  |

Conclusion:
- This run fails the user-reported original Stage3 early gate of `0.88+` at the matched epoch0 point.
- The trajectory quality recovers over the limited 1-epoch run, but the recovery is not close enough; the gap at epoch0 is about `-0.049` PDMS versus the early gate.
- Main regressions versus the expected early Stage3 band are DAC and EP, with TTC/DDC also below a healthy original Stage3 trajectory distribution.
- Do not promote this DPPO-style transition replay variant to full training. Any next GRPO redesign should revisit the objective itself, not only tune LR or minibatch count.

### 2026-06-14 Protocol Update: early Stage3 gate

Baseline:
- The original ReCogDrive Stage3 run is reported to reach roughly `0.88+` navtest PDMS by epoch0-1.
- Therefore new Stage3 variants must be compared at matched early checkpoints, not only at the long-horizon `epoch9-step13300` reference or later.

Operational rule:
- Use `scripts/training/gate_recogdrive_stage3_early_pdms.py` with `--threshold 0.88 --margin 0.005`.
- Treat `PDMS < 0.875` at the selected early checkpoint as a diagnostic warning,
  not an automatic stop signal, unless there is also implementation evidence
  that the line is not worth continuing.
- Treat `0.875 <= PDMS < 0.88` as a watch zone; continue collecting matched
  checkpoints for non-catastrophic GRPO-family variants.
- Treat `PDMS >= 0.88` as the minimum healthy early band; it is not success by
  itself because the final target remains beating the original Stage3/Safe
  DiffGRPO best band.

Automation:
- `scripts/training/watch_recogdrive_stage3_early_gate.sh` polls the gate output for a run.
- With `STOP_ON_FAIL=0`, it reports the decision and leaves training untouched.
- With `STOP_ON_FAIL=1`, it only terminates the local training process group recorded by the gate JSON. It does not terminate remote evaluation watchers or unrelated remote tasks.
- For new stable launches, set `START_EARLY_GATE_WATCHER=1 EARLY_GATE_THRESHOLD=0.88 EARLY_GATE_MARGIN=0.005`.
- Use `EARLY_GATE_STOP_ON_FAIL=1` only for deliberately short diagnostics where
  we already have strong evidence that low early PDMS is unrecoverable. Do not
  enable it for main GRPO-family training by default.
- The GRPO/GSPO launcher now defaults to `CHECKPOINT_EVERY_N_TRAIN_STEPS=300` and starts the early gate watcher.
- The buffer-guided GRPO wrapper defaults to `CHECKPOINT_EVERY_N_TRAIN_STEPS=300`,
  `START_EARLY_GATE_WATCHER=1`, and `EARLY_GATE_STOP_ON_FAIL=0`; early gate
  output is now a report, not an automatic kill path.

### 2026-06-14 Attempt: GRPO self-imitation safetygate stopped before checkpoint

Motivation:
- The previous cap05 run reached `0.887107` at step 900 but improved EP while regressing NC/TTC/DDC.
- The next test added stricter self-imitation target gates so only high-reward, NC/DAC/TTC/DDC-safe generated samples could become distillation targets.

Config:
- Run name: `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z`
- LR: `1e-4`.
- `sample_time=16`, batch size `2`, accumulation `4`, 8 local GPUs.
- Self-imitation target gates: NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`.
- Main GRPO group filtering still only hard-gated NC/DAC, not TTC/DDC.

Observed training diagnostics before stopping:
- Last logged step: `249`.
- `train/reward_step`: `0.766856`.
- `train/base_reward_step`: `0.778479`.
- `train/safe_ratio_step` / `hard_safe_ratio`: `0.898438`.
- `train/mean_ep_step`: `0.783179`.
- `train/mean_ttc_step`: `0.886719`.
- Self-imitation was active: target scene ratio `0.5`, target reward mean `0.987467`.

Decision:
- Stop before the first checkpoint and navtest evaluation.
- Reason: this run only constrained auxiliary self-imitation targets, while the primary GRPO update could still accept TTC/DDC-regressing positive-advantage samples. That makes it a lower-information variant than a main-objective hard-gate run.
- Do not repeat this exact configuration.

### 2026-06-14 Attempt: main GRPO TTC/DDC hard gate first launch

Motivation:
- Step300 to step900 diagnostics from the best cap05 run indicate PDMS gains were dominated by EP, while TTC/DDC and NC regressed.
- The targeted next change is to hard-mask the main GRPO advantage for samples that fail TTC/DDC, so progress-improving but direction/timing-unsafe samples do not contribute positive policy gradient.

Implementation:
- Added agent-level controls:
  - `grpo_hard_gate_ttc`
  - `grpo_hard_gate_ddc`
  - `grpo_ttc_safe_threshold`
  - `grpo_ddc_safe_threshold`
- Wired launcher env/Hydra passthrough and `GRPOConfig` assignment.

Launch issue:
- First launch name: `stage3_grpo_rloo_selfimit_mainhardgate_step300_s16_lr1e4_b2acc4_8gpu_20260614T160328Z`.
- Hydra failed before training:
  - `Could not override 'agent.grpo_hard_gate_ttc'`
  - Root cause: the Python constructor and scripts were patched, but the structured agent YAML lacked default keys.

Decision:
- This is a configuration-wiring failure, not an algorithm result.
- Fix by adding the four default fields to `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`, then relaunch under a new run name.

### 2026-06-14 Attempt: main GRPO hard gate v2 launch wiring

Launch issue:
- Run name: `stage3_grpo_rloo_selfimit_mainhardgate_v2_step300_s16_lr1e4_b2acc4_8gpu_20260614T161141Z`.
- Hydra parsing succeeded after adding the YAML defaults.
- The shell preflight failed before training:
  - `ELITE_BUFFER_DIR must exist when OFFLINE_RL_ENABLED/GRPO_BUFFER_GUIDANCE_ENABLED is true`.

Root cause:
- Self-imitation in `forward_grpo` is gated by `offline_cfg.enabled`.
- Therefore `OFFLINE_RL_ENABLED=true` is needed for this specific self-imitation variant, even when buffer guidance and buffer distillation are disabled.
- The launcher safety check still requires a real train-only `ELITE_BUFFER_DIR` whenever offline RL is enabled.

Decision:
- This is a launch-contract error, not an algorithm result.
- Do not repeat this exact startup command.
- Correct startup for self-imitation GRPO hard-gate runs:
  - `OFFLINE_RL_ENABLED=true`
  - pass a real train-only elite buffer directory
  - keep `GRPO_BUFFER_GUIDANCE_ENABLED=false`, `GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0`, and `GRPO_BUFFER_REWARD_BONUS_WEIGHT=0.0` if the experiment should isolate online GRPO + self-imitation.

### 2026-06-14 Attempt: main GRPO TTC/DDC hard gate v3 running

Motivation:
- Preserve the cap05 run's EP upside while preventing positive policy-gradient updates from TTC/DDC-regressing samples.
- Keep the online self-imitation auxiliary loss, but gate targets by NC/DAC/TTC/DDC and cap target scenes at `0.5`.

Config:
- Run name: `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`.
- LR: `1e-4`, scheduler epochs `20`, min LR `1e-5`.
- `sample_time=16`, batch size `2`, accumulation `4`, 8 local GPUs.
- Checkpoint every `300` train steps and every epoch.
- Main GRPO hard gates:
  - NC/DAC existing hard safe mask.
  - TTC threshold `0.95`.
  - DDC threshold `0.99`.
- Self-imitation:
  - weight `0.01`, linear warmup over 2 epochs.
  - top-k `1`, min reward `0.88`, min reward margin `0.01`.
  - target scene cap `0.5`.
  - baseline mode `group_leave_one_out`.
  - low-noise diffusion timestep regression.
  - target gates NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`.
- Offline buffer path passed only to satisfy `offline_cfg.enabled` safety contract:
  - `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`
  - buffer guidance disabled.
  - buffer distillation disabled.
  - buffer reward bonus disabled.
- Eval watchers:
  - zt2: `unique_lock_watch_on_vla_zt2_4gpu`, pid `2446350`, GPU list `4,5,6,7`, strict wait `GPU_MAX_MEM_USED_MB=2000`, `GPU_MAX_UTIL=5`.
  - zt3: `secondary_watch_on_rl_zt3_memfit_4gpu`, pid `1015107`, GPU list `0,1,2,4`, strict wait `GPU_MAX_MEM_USED_MB=2000`, `GPU_MAX_UTIL=5`.

Startup verification:
- Training entered Lightning train loop at `2026-06-14 16:17:48`.
- Local GPU utilization after entering training: roughly `56-95%`.
- Dataset sizes: train `85109`, validation `18179`.
- First checkpoint and navtest PDMS are pending.
- First available training scalar at step `49`:
  - `reward_step=0.645440`
  - `base_reward_step=0.790706`
  - `hard_safe_ratio_step=0.816406`
  - `mean_ttc_step=0.902344`
  - self-imitation target ratio `0.4375`
  - self-imitation target reward mean `0.867652`
- Diagnostic gap found: main GRPO logging had `mean_ttc` but not main `mean_ddc`/NC/DAC/TLC pass-ratio fields. A follow-up logging-only patch adds those fields for future runs; it does not affect this already-running v3 process.

Expected early gate:
- Stop if the first evaluated early checkpoint is below `0.875` PDMS.
- Watch for one more early checkpoint if within `[0.875, 0.88)`.
- Continue only if the run clears `0.88` early PDMS, then compare against matched early controls such as cap05 `0.887107`; compare against Safe DiffGRPO `0.906184` and original Stage3 `epoch9-step13300` `0.9055` only at comparable long-horizon exposure.

Artifact cleanup:
- Deleted no-checkpoint failure directories after recording the causes:
  - `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z`
  - `stage3_grpo_rloo_selfimit_mainhardgate_step300_s16_lr1e4_b2acc4_8gpu_20260614T160328Z`
  - `stage3_grpo_rloo_selfimit_mainhardgate_v2_step300_s16_lr1e4_b2acc4_8gpu_20260614T161141Z`

### 2026-06-14 Reference audit: diffusion RL and preference training

Purpose:
- Do not keep iterating only by local hyperparameter tweaks.
- Before another algorithmic change, compare our implementation against mature diffusion-policy and diffusion-preference implementations.

References inspected:
- DPPO: Diffusion Policy Policy Optimization.
  - Paper/project/code: `https://diffusion-ppo.github.io/`, `https://github.com/irom-princeton/dppo`.
  - Local audit clone: `/tmp/dppo_ref`.
  - Key files inspected:
    - `/tmp/dppo_ref/model/diffusion/diffusion_ppo.py`
    - `/tmp/dppo_ref/agent/finetune/train_ppo_diffusion_agent.py`
    - `/tmp/dppo_ref/README.md`
- Diffusion-DPO.
  - Paper/code: `https://arxiv.org/abs/2311.12908`, `https://github.com/SalesforceAIResearch/DiffusionDPO`.
  - Local audit clone: `/tmp/DiffusionDPO_ref`.
  - Key files inspected:
    - `/tmp/DiffusionDPO_ref/train.py`
    - `/tmp/DiffusionDPO_ref/README.md`
- SimpleVLA-RL repository existence confirmed:
  - `https://github.com/PRIME-RL/SimpleVLA-RL`
  - A shallow clone was attempted but cancelled because the transfer was slow. Do a focused follow-up audit before implementing a VLA-specific PPO/DPO variant.

DPPO implementation takeaways:
- It treats each denoising transition as a PPO sample:
  - stores `chains_prev`, `chains_next`, old logprobs, values, returns, and advantages.
  - samples minibatches over `(environment step, denoising step)`.
- It uses true PPO clipping against old logprobs:
  - `ratio = exp(newlogprob - oldlogprob)`.
  - `max(-A * ratio, -A * clipped_ratio)`.
- It schedules clipping by denoising step:
  - small base clip at early denoising steps, larger clip later.
- It has advantage normalization and quantile clipping.
- It supports KL diagnostics and early stop when approximate KL exceeds `target_kl`.
- It uses reward scaling / GAE / critic in online environments.
- It keeps a BC regularizer to the base policy as a trust region.

Diffusion-DPO implementation takeaways:
- Preference learning is not plain regression to the preferred sample.
- The chosen/rejected pair shares the same diffusion timestep and noise.
- Loss is based on the difference between current-model diffusion MSE gap and reference-model diffusion MSE gap:
  - `model_diff = loss_w - loss_l`
  - `ref_diff = ref_loss_w - ref_loss_l`
  - `loss = -logsigmoid(-0.5 * beta * (model_diff - ref_diff))`
- This is important for our buffer setting: if we use high-PDMS buffer trajectories for preference learning, the preferred and rejected trajectories should be compared under matched diffusion noise/timestep and against the IL/reference policy, not just distilled with MSE.

Gap versus current ReCogDrive implementation:
- Current v3 run is a trajectory-level GRPO/GSPO variant with online PDM rewards and hard safety masks.
- We have GSPO ratio, reference KL, BC anneal, safety-shaped reward, and self-imitation.
- We do not yet have a full DPPO-style transition replay path where every denoising transition from sampled chains becomes a PPO sample with old logprob, denoising-step clip schedule, KL early stop, and multiple minibatch epochs.
- The earlier DPPO-style smoke in this repo was intentionally simplified and did not clear the early Stage3 gate; it should not be treated as a decisive test of mature DPPO.
- The earlier AWAC/IQL path was also not a decisive test of preference-style learning, because it mainly used weighted diffusion regression. Diffusion-DPO suggests a more appropriate pairwise diffusion objective for buffer knowledge absorption.

Actionable next algorithm directions after v3 early gate:
- If v3 clears early PDMS and improves safety submetrics:
  - Continue to step600/900 and compare against cap05.
  - Consider adding a mature DPPO transition replay branch only after confirming the hard safety gate helps.
- If v3 fails early PDMS because reward drops too much:
  - Try a less restrictive main mask or soft penalty schedule for TTC/DDC, but keep self-imitation target gates strict.
  - Do not remove DDC/TTC diagnostics.
- If buffer knowledge still needs absorption:
  - Implement a diffusion-DPO style pairwise loss over high-PDMS buffer candidate vs GT/IL/rejected candidate.
  - Pair samples must share the same diffusion timestep and noise.
  - Use IL/reference model losses as the DPO reference term.
  - Keep BC/reference KL small but present.
- If adopting DPPO:
  - Implement old-logprob storage for denoising transitions.
  - Minibatch over `(scene, sample, denoising step)`, not only final trajectory.
  - Add denoising-step clip schedule, target KL, advantage normalization, and diagnostics.
  - Validate on a small run with matched update steps before full training.

### 2026-06-14 Cleanup: Remove Obsolete Stage3 Artifacts

Reason:
- `/mnt/project/VLA-AD/outputs` had grown to roughly `252G`.
- The user explicitly allowed outdated failed results/logs to be deleted after recording the failed attempt in this summary.
- The cleanup must not affect active training/evaluation, train-only elite buffers, or the strongest retained checkpoints.

Kept:
- Current active v3 run:
  - `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`
- Active zt3 original-LR control:
  - `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`
- Active keep-best buffer generation:
  - `stage3_awac_keepbest_buffer_zt3_wait_20260614T142658Z`
- Train-only AWAC elite buffer cache:
  - `recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`
- Safe DiffGRPO best checkpoint only:
  - `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z/checkpoint_archive/epoch_12-step_17290.ckpt`
- Best cap05 checkpoint only:
  - `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/train/hydra/training_recogdrive_agent/2026.06.14.11.15.51/step_checkpoints/step-step=900.ckpt`

Deleted or pruned:
- Dry-run/smoke/debug/preflight/probe directories, including AWAC preflights, GRPO launcher dry-runs, replay smoke tests, DPPO smoke tests, and evaluation launcher dry-runs.
- Failed or rejected runs whose conclusions are already recorded above:
  - trajectory-level replay one-epoch runs and their empty navtest watcher directories;
  - stopped `2e-4` buffer-guided GRPO variants;
  - no-cap self-imitation run;
  - stale queued self-imitation wait directories.
- Repeated historical checkpoints:
  - Safe DiffGRPO archive checkpoints except `epoch_12-step_17290.ckpt`;
  - cap05 duplicate watcher archive checkpoints and train checkpoints except `step-step=900.ckpt`;
  - old Safe DiffGRPO relaunch checkpoint directory;
  - old lrfix/bcanneal train checkpoint directory;
  - old lrfix/bcanneal eval checkpoint archive.

Result:
- Deleted `69` dry-run/smoke/debug/preflight/probe directories.
- Reduced `/mnt/project/VLA-AD/outputs` from roughly `252G` to roughly `170G`.
- Did not kill active local or remote tasks.

Do not repeat:
- Do not resume the deleted dry-run/preflight directories.
- Do not rerun the rejected trajectory-level replay or simplified step-level replay path as a full experiment without a mature DPPO-style implementation.
- Do not treat AWAC weighted regression failures as proof that buffer/preference learning is useless; use mature Diffusion-DPO or DPPO-style mechanics if revisiting buffer absorption.

### 2026-06-14 Cleanup: Remove Remaining Obsolete Logs And Probe Caches

Reason:
- The user approved deleting outdated results/logs after the failed attempts are recorded in this ledger.
- The cleanup target is stale artifacts that can confuse future experiment selection or waste space, not active training/evaluation state.

Deleted on 2026-06-14 UTC:
- Stale Stage3 metric-cache launcher/watcher wrappers:
  - `stage3_rl_after_metric_cache_watch_20260608T145806Z`
  - `metric_cache_navtrain_full_20260608T145628Z`
  - `metric_cache_navtrain_full_safe_diffgrpo_20260609T175818Z`
- Obsolete non-best Safe DiffGRPO / lrfix outputs:
  - `stage3_rl_2b_safe_diffgrpo_online_20260609T175818Z`
  - `stage3_safe_diffgrpo_navtest_eval_after_relaunch_20260609T1944Z`
  - `stage3_rl_2b_safe_diffgrpo_lrfix_g16_20260612T082041Z`
  - `stage3_safe_diffgrpo_lrfix_g16_ckpt_stream_eval_20260612T082041Z`
  - `stage3_rl_2b_safe_diffgrpo_lrfix_g16_bcanneal_20260612T082634Z`
  - `stage3_safe_diffgrpo_lrfix_g16_bcanneal_ckpt_stream_eval_20260612T082634Z`
  - `stage3_rl_2b_safe_diffgrpo_lrfix_g16_bcanneal_b4a2_20260612T084307Z`
  - `stage3_safe_diffgrpo_lrfix_g16_bcanneal_b4a2_ckpt_stream_eval_20260612T084307Z`
- Stale metric-cache smoke/probe artifacts:
  - `fast_metric_cache_navtest_mirror_20260613T011857Z`
  - `metric_cache_navtest_first1024`
  - `metric_cache_navtest_smoke16`
  - `metric_cache_train_smoke_stage3_rl`
- Early AWAC elite-buffer probe caches, superseded by the full train-only v2 elite buffer:
  - `stage3_awac_iql_builder_b32_probe_20260612T191838Z_elite_buffer`
  - `stage3_awac_iql_builder_b4_probe_20260612T184659Z_elite_buffer`
  - `stage3_awac_iql_builder_b4_probe_fg_20260612T184819Z_elite_buffer`
  - `stage3_awac_iql_builder_b8_probe_fg_20260612T185232Z_elite_buffer`
- Stale two-expert Stage2 intermediate output dirs whose conclusions are already in `reports/two_expert_slot/` and which have no live pid:
  - `two_expert_slot_stage2_full_dit_sft_A0init_clonefix_20260613T133206Z`
  - `two_expert_slot_stage2_full_dit_sft_A0init_indexfix_20260613T195319Z`
  - `two_expert_slot_stage2_full_dit_sft_A0init_indexfix_4gpu_eqbs128_20260613T205724Z`
  - `two_expert_slot_stage2_full_dit_sft_A0init_collatefix_4gpu_eqbs128_20260613T210233Z`

Explicitly kept:
- `a0_stage2_repro_20260531_003029`, because highcap/two-expert scripts still use its A0 checkpoint path as a default.
- `recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot`, because residual-anchor scripts still reference it by default; delete only after updating or retiring those scripts.
- `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`, because the zt3 keep-best train-only buffer generator still points at its `step_checkpoints` directory.
- Current active v3 run, zt3 original-LR control, active keep-best buffer generator, full train-only elite buffer, and Safe DiffGRPO best checkpoint.

Do not repeat as default routes:
- Do not relaunch old lrfix/bcanneal Safe DiffGRPO outputs. The retained baseline is the already-evaluated Safe DiffGRPO best checkpoint, not every intermediate rerun.
- Do not use the tiny AWAC probe buffers as training inputs. Use the full train-only v2 elite buffer or rebuild a new validated full buffer.
- Do not restart stale two-expert Stage2 intermediate jobs; the current tracked two-expert work is the random-HMEF val6000/navtest run.
- Do not rerun auxiliary-target-only safetygate self-imitation. If GRPO safety gating is tested, TTC/DDC must be in the main GRPO hard-safe mask and reported at matched checkpoints.

### 2026-06-14 Eval Scheduling Update: relaxed checkpoint watchers

Reason:
- The v3 main hard-gate run had already produced `step-step_300.ckpt`, but both strict remote watchers were blocked by a conservative `GPU_MAX_MEM_USED_MB=2000` threshold.
- zt2 GPUs had enough remaining 80GB-card memory for exact navtest eval while existing remote tasks continued. The user explicitly allowed using remote resources, with the constraint that existing remote tasks must not be killed.
- The zt3 original-LR control also had an `epoch_0-step_1290.ckpt` waiting for eval. Its old watcher config pointed to `/mnt/project/VLA-AD_stage3_algo_clean_4f3eb73`, so a current-repo relaxed watcher was launched to avoid stale script risk.

Actions:
- Launched v3 relaxed zt2 watcher without killing any remote process:
  - Host: `training-vla-zt2`
  - Dir: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z/relaxed_watch_on_vla_zt2_4gpu`
  - PID: `2494061`
  - GPU list: `4,5,6,7`
  - `GPUS_PER_NODE=4`, `GPU_MAX_MEM_USED_MB=10000`, `GPU_MAX_UTIL=100`
  - Eval script remains `run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh`.
  - PDM runner remains `exact_pool`; navtest full metric cache remains unchanged.
  - Started evaluating `step-step_300` at `2026-06-14T17:37:17Z`.
- Launched original-LR control relaxed zt3 watcher without killing any remote process:
  - Host: `training-rl-zt3`
  - Dir: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z/relaxed_watch_on_rl_zt3_2gpu`
  - PID: `1043991`
  - GPU list: `6,7`
  - `GPUS_PER_NODE=2`, `GPU_MAX_MEM_USED_MB=20000`, `GPU_MAX_UTIL=100`
  - Eval script uses the current repo path under `/mnt/project/VLA-AD_last_vla_dev`.
  - Started evaluating `epoch_0-step_1290` at `2026-06-14T17:42:54Z`.

Interpretation rules:
- These relaxed watchers only change scheduling thresholds. They do not change inference, PDM scorer, metric cache, or submetric calculation.
- Use their `checkpoint_eval_submetrics.tsv` rows as authoritative navtest results once completed.
- If the strict original watchers later wake up, avoid duplicate conclusions by preferring the first completed exact full-navtest row and checking checkpoint ids.

### 2026-06-15 Live Status: GRPO control, zt3 control, and Buffer-DPO queue

Reason:
- The zt3 run name looks like a Stage3 GRPO baseline, but it was launched from the current code tree after several Stage3 implementation changes. It must not be mislabeled as a bit-exact historical original ReCogDrive Stage3 reproduction.
- The next absorption experiment is Buffer-DPO over the train-only v2 elite buffer, but it is queued behind existing zt2 work. Existing remote jobs must not be killed.

Current-code original-LR GRPO control on zt3:
- Run: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`
- Code dir: `/mnt/project/VLA-AD_stage3_algo_clean_4f3eb73`
- Config identity: current-code original-LR GRPO control, not AWAC/IQL, not Buffer-DPO, not self-imitation, not GSPO-ratio.
- Key config: `agent.grpo=True`, `agent.lr=1e-4`, `agent.grpo_sample_time=16`, `agent.bc_anneal=true`, `agent.bc_coeff_start=0.10`, `agent.bc_coeff_end=0.05`, `agent.reference_kl_coeff=0.02`, `trainer.params.max_epochs=20`, `trainer.params.devices=3`, `trainer.params.accumulate_grad_batches=11`.
- Disabled controls: `offline_rl_enabled=false`, `offline_rl_grpo_buffer_guidance_enabled=false`, buffer reward bonus/distill/self-imitation weights all `0.0`, `agent.grpo_use_gspo_ratio=false`.
- Status at `2026-06-15T02:44Z`: training alive, TensorBoard reached step `2349` in epoch `1`; only saved/evaluated checkpoint remains `epoch_0-step_1290`.
- Navtest result at epoch0-step1290: PDMS `0.8967935081`, NC `0.9871889932`, DAC `0.9758609326`, TTC `0.9625968034`, EP `0.8265160702`, comfort `1.0`, DDC `0.9743779865`.

Local current-repo original-LR GRPO control:
- Run: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z`
- Status at `2026-06-15T02:41Z`: training alive, TensorBoard reached step `1649` in epoch `1`; local watcher `local_auto_watch_shared_4gpu_v3` is alive and waiting for the next checkpoint.
- Latest evaluated checkpoint: `step-step_1500`, PDMS `0.8916996087`; current best in this run remains `epoch_0-step_1330`, PDMS `0.8936864074`.

Buffer-DPO absorption experiment:
- Run queued on zt2: `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`
- Status at `2026-06-15T02:42Z`: launcher alive but still waiting for GPUs `0-7` under the configured free-GPU gate; no training checkpoint yet.
- Config identity: GRPO plus diffusion-DPO preference absorption from the train-only v2 elite buffer; not AWAC/IQL weighted regression.
- Key settings: LR `1e-4`, `sample_time=16`, BC `0.10 -> 0.05`, ref KL `0.02`, `offline_rl_enabled=true`, `grpo_buffer_preference_dpo_loss_weight=0.02`, reward bonus/distill/self-imitation weights `0.0`, `grpo_use_gspo_ratio=false`.

Current decision:
- Keep local and zt3 GRPO control runs alive for matched-epoch evidence.
- Keep the zt2 Buffer-DPO queue alive; do not start a duplicate absorption run unless a cross-host start lock is added or the zt2 queue is deliberately retired.
- Do not claim buffer/preference improvement until Buffer-DPO actually starts and reaches matched checkpoint evaluation.

### 2026-06-15 Attempt: RPP-constrained GRPO with TTC/DDC hard mask

Motivation:
- Previous evidence:
  - `/mnt/project/tsinghua_code/recogdrive_new` 91.04 analysis shows the strongest local Stage3 result is not AWAC/IQL; it is a Reinforce++/GRPO-style Stage3 initialized from a stronger cross-attention SFT policy.
  - The stable 91.04 recipe uses `sample_time=16`, BC anneal `0.10 -> 0.05`, `reference_kl_coeff=0.02`, denoising discount, and batch-level advantage normalization.
  - Its navtest gain is mostly EP/DAC/TTC, while DDC regresses from Stage2-level `0.9814` to about `0.9663`. Therefore the next GRPO run should keep the successful policy-gradient recipe but constrain unsafe TTC/DDC-positive updates.
  - Our current-repo original-LR GRPO controls are below historical 91.04 at matched early checkpoints: zt3 epoch0-step1290 PDMS `0.8967935081`; local epoch0-step1330 PDMS `0.8936864074`; local step1500 PDMS `0.8916996087`.
- Reference sources:
  - Local code/result analysis: `reports/recogdrive_stage3/recogdrive_new_pdms9104_analysis.md`.
  - REINFORCE++ (`arXiv:2501.03262`) supports critic-free policy-gradient post-training with global/batch advantage normalization and KL/clip-style stabilizers.
  - DDPO (`arXiv:2305.13301`) supports treating the denoising trajectory as a multi-step decision process and directly optimizing diffusion models against black-box rewards; it reports policy-gradient training outperforming reward-weighted likelihood for downstream objectives.
  - DPPO (`arXiv:2409.00588`) supports diffusion-policy fine-tuning with policy gradients in continuous control/robotics, emphasizing on-manifold exploration and stable diffusion-policy RL.
  - Diffusion-DPO (`arXiv:2311.12908`) supports preference/pairwise objectives for diffusion models, but our previous lightweight buffer-DPO attempt has not yet produced matched-checkpoint evidence; it remains secondary until its implementation and training signal are proven.
- Mechanism copied:
  - Keep the 91.04-style RPP/GRPO configuration intact where possible.
  - Add TTC/DDC hard-safe gating to the main GRPO advantage path, not as an auxiliary self-imitation-only filter.
  - Keep AWAC/buffer/DPO/self-imitation disabled in this run to isolate the effect of mature RPP recipe plus safety gating.

Implementation completeness:
- Required pieces present:
  - Existing planner supports batch advantage normalization, reference KL, BC anneal, `sample_time=16`, behavior/reference policy logprob replay, TTC/DDC hard gates, and submetric logging.
  - New launcher: `scripts/training/launch_recogdrive_stage3_grpo_rpp_constrained_2b_local.sh`.
  - It delegates to the stable Stage3 GRPO launcher and passes all required Hydra overrides.
- Known simplifications:
  - This run does not yet absorb the generated high-PDMS train-only buffer.
  - This run does not use DPPO step-level transition minibatching by default; `ppo_replay_logprob_mode=trajectory` and one replay inner epoch are kept to match the known ReCogDrive 91-style recipe first.
  - This run is on zt3 with 2 GPUs because zt2/local resources already have active jobs; effective optimizer batch is matched with larger accumulation.
- Why simplifications are acceptable:
  - The first question is whether our current code can reproduce the historical early Stage3 lift when configured like the 91.04 recipe and with DDC/TTC protection.
  - If this run cannot reach the original early range (`>=88` PDMS at epoch0 and ideally closer to `89-90` at matched steps), adding buffer absorption on top would be hard to interpret.

Config:
- Run name:
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_rpp_constrained_s16_lr1e4_b2acc16_zt3_2gpu_20260615T030832Z`
- Host / GPUs:
  - Training: `training-rl-zt3`, GPUs `6,7`.
  - Watchers: primary zt2 `unique_lock_watch_on_vla_zt2_4gpu`; secondary zt3 `secondary_watch_on_rl_zt3_memfit_4gpu`.
- LR / schedule:
  - `agent.lr=1e-4`, `max_epochs=20`, scheduler min LR `1e-5`, no warmup.
- Effective batch / sample_time / steps:
  - `devices=2`, `batch_size=2`, `accumulate_grad_batches=16`; effective optimizer batch is `64` scenes/update, matching prior 8GPU `b2acc4` scale.
  - `grpo_sample_time=16`.
  - `checkpoint.every_n_train_steps=300`, `checkpoint.every_n_epochs=1`.
- Reward/cache split:
  - Training uses `train_test_split=navtrain`, `metric_cache_train_full`, and online hidden/cache dataset path only.
  - No navtest reward/cache artifact is used for training.
- Safety guards:
  - `grpo_hard_gate_ttc=true`, `grpo_ttc_safe_threshold=0.95`.
  - `grpo_hard_gate_ddc=true`, `grpo_ddc_safe_threshold=0.99`.
  - `offline_rl_enabled=false`, buffer reward bonus/distill/DPO/self-imitation weights all `0`.
  - `grpo_normalize_advantage_batch=true`, `grpo_advantage_clip_abs=5.0`.

Expected diagnostics:
- Should increase:
  - Early checkpoint PDMS should be competitive with original current-code controls and move toward the historical 91.04 trajectory.
  - EP/DAC/TTC should not regress relative to original-LR control.
- Should stay stable:
  - DDC should be higher than the unconstrained 91-style result and ideally not below the current controls.
  - TTC pass ratio and DDC pass ratio should remain high after the first checkpoint.
- Stop if:
  - First evaluated full navtest checkpoint is materially below the original-LR controls at similar step/epoch.
  - Hard gates collapse the effective positive-advantage ratio and the run cannot produce useful updates.
  - Training makes no checkpoint progress after a reasonable first-step window and logs/event files remain frozen.

Results:
- Launch verification:
  - `status/stage3_rl_2b.json` reports the launcher alive with the intended command.
  - `train/resolved_command.txt` confirms `sample_time=16`, `lr=1e-4`, BC `0.10 -> 0.05`, `reference_kl_coeff=0.02`, `grpo_normalize_advantage_batch=true`, `grpo_hard_gate_ttc=true`, `grpo_hard_gate_ddc=true`, `devices=2`, `batch_size=2`, `accumulate_grad_batches=16`, `offline_rl_enabled=false`.
  - Hydra overrides confirm the same values.
- Runtime status at `2026-06-15T03:16Z`:
  - Training process is alive on zt3.
  - GPU processes for the run occupy GPUs `6,7`; those GPUs are active, but an older keep-best/buffer process also shares memory on the same cards.
  - TensorBoard event file is still initialization-only and no `*.ckpt` has been produced yet, so no navtest evaluation result is available.
- Checkpoint:
  - None yet.
- PDMS / NC / DAC / TTC / EP / comfort / DDC / TLC:
  - Pending first checkpoint evaluation.
- Decision:
  - Keep this run alive for the first step300/epoch0 evidence.
  - Do not start another overlapping constrained-GRPO duplicate before this first checkpoint result unless this run is proven stalled or invalid.

### 2026-06-15 Candidate: DPPO-style step-level GRPO replay

Motivation:
- Previous evidence:
  - Plain/current-code trajectory-level GRPO controls improve over Stage2 but have not matched the historical 91.04 trajectory.
  - The AWAC/IQL weighted-regression path is not yet convincing: it can discover high-PDMS candidates, but absorbing them with plain diffusion regression appears weak and may not match how diffusion policies are best post-trained.
  - DPPO/DDPO style work argues that a diffusion denoising chain should be optimized as a multi-step policy rather than only as one trajectory-level scalar likelihood target.
  - Current code already contains a fixed-rollout replay path (`stage3_objective=grpo_replay`) with old/new log-prob ratios, step-level logprobs, transition minibatches, and DPPO-style step clip scheduling. The missing piece was a clear launcher and complete upper-level parameter logging.
- Reference sources:
  - DPPO (`arXiv:2409.00588`): diffusion policy optimization for continuous control/robotics with policy-gradient fine-tuning.
  - DDPO (`arXiv:2305.13301`): denoising diffusion policy optimization with black-box rewards over the denoising MDP.
  - REINFORCE++ (`arXiv:2501.03262`): batch/global advantage normalization and critic-free policy-gradient stabilizers.
- Mechanism copied:
  - Collect frozen rollout from the behavior policy.
  - Score rollout samples with train PDM cache only.
  - Store old trajectory/step log-probs.
  - Replay clipped old/new ratio updates on step-level denoising transitions.
  - Use `dppo_exp` per-step clip width so denoising steps are not all clipped with one blunt scalar.

Implementation completeness:
- Required pieces present:
  - `ReCogDriveAgent.forward` routes `stage3_objective=grpo_replay` to `collect_grpo_replay_rollout`.
  - `AgentLightningModule` switches to manual optimization when `stage3_objective=grpo_replay`.
  - `compute_grpo_replay_loss` supports `ppo_replay_logprob_mode=step`, `ppo_replay_step_minibatch_mode=transition`, and `ppo_replay_step_clip_schedule=dppo_exp`.
  - New launcher: `scripts/training/launch_recogdrive_stage3_grpo_dppo_step_2b_local.sh`.
  - Upper-level stable launcher now exports and logs all DPPO-step replay knobs, so future runs are auditable.
- Known simplifications:
  - No train-only elite buffer absorption is enabled in this candidate launcher.
  - One inner replay epoch by default. This tests whether the step-level policy-gradient formulation itself helps before adding more replay epochs.
  - `GRPO_PPO_REPLAY_MINIBATCH_SIZE=128` is a conservative transition minibatch default; it can be adjusted after memory/progress diagnostics.
- Why simplifications are acceptable:
  - The aim is to separate algorithmic formulation from buffer quality. If step-level clipped policy-gradient cannot beat trajectory-level GRPO under the same reward/cache split, adding buffer supervision is unlikely to be the first fix.

Config:
- Launcher:
  - `scripts/training/launch_recogdrive_stage3_grpo_dppo_step_2b_local.sh`
- Intended default:
  - `STAGE3_OBJECTIVE=grpo_replay`
  - `LR=1e-4`, `MAX_EPOCHS=20`
  - `GRPO_SAMPLE_TIME=16`
  - BC anneal `0.10 -> 0.05`, `REFERENCE_KL_COEFF=0.02`
  - `GRPO_USE_GSPO_RATIO=true`, behavior-policy sample/sync enabled
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`, `GRPO_ADVANTAGE_CLIP_ABS=5.0`
  - TTC/DDC hard gates enabled with thresholds `0.95` and `0.99`
  - `GRPO_PPO_REPLAY_LOGPROB_MODE=step`
  - `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=transition`
  - `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=dppo_exp`
  - `GRPO_PPO_REPLAY_STEP_CLIP_BASE=0.001`, `GRPO_PPO_REPLAY_STEP_CLIP_RATE=3.0`
  - Buffer/DPO/self-imitation disabled.
  - Resource guard defaults patched on 2026-06-15: `TRAIN_WAIT_FOR_FREE_GPUS=1`
    with `TRAIN_GPU_MAX_MEM_USED_MB=2000` and `TRAIN_GPU_MAX_UTIL=5`; eval
    watcher defaults mirror this with `EVAL_WAIT_FOR_FREE_GPUS=1`,
    `EVAL_GPU_MAX_MEM_USED_MB=2000`, and `EVAL_GPU_MAX_UTIL=5`. This launcher
    should wait rather than stack on top of active long-horizon runs.

Expected diagnostics:
- Should increase:
  - Better early PDMS than current trajectory-level GRPO controls at matched checkpoint ids.
  - More stable log-ratio/clip diagnostics than plain trajectory logprob.
- Should stay stable:
  - `ppo_replay_loss_active=1`, nonzero `ppo_replay_selected_transition_count`.
  - DDC/TTC pass ratio should not collapse due to the hard mask.
- Stop if:
  - Replay optimizer steps explode runtime without producing checkpoints.
  - `gspo_ratio_clip_frac` is saturated for most minibatches.
  - `ppo_replay_logprob_clamped_frac` is high enough that the logprob signal is mostly clamp artifacts.

Results:
- Status:
  - Not launched yet. Keep as the next mature GRPO experiment if the active
    aux-decay and clean-GRPO streams free resources or if matched evidence shows
    trajectory-level updates are the bottleneck.
- Decision:
  - Do not start this while the aux-decay and clean-GRPO streams are both active
    unless one stream is proven stalled/invalid or resources are deliberately
    freed for a higher-priority DPPO test.

### 2026-06-15 Operational update: stop local control and launch local8 RPP-constrained GRPO

Actions:
- Stopped the local current-repo original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z` after it had already produced enough baseline evidence:
  - step300 PDMS `0.881910`
  - step600 PDMS `0.886777`
  - step900 PDMS `0.883839`
  - step1200 PDMS `0.891510`
  - epoch0-step1330 PDMS `0.893686`
  - step1500 PDMS `0.891700`
  - step1800 PDMS `0.884834`, with EP `0.837906` but weaker TTC `0.937222` and DDC `0.952958`
- Reason: continuing this control was lower value than using local 8GPU for a targeted GRPO design. The best local point remains epoch0-step1330 and still trails the stopped zt3 control epoch0-step1290 `0.896794`.
- Launched the replacement local run:
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_rpp_constrained_s16_lr1e4_b2acc4_local8_20260615T035445Z`
  - local 8GPU, `batch_size=2`, `accumulate_grad_batches=4`, effective update batch `64`
  - `lr=1e-4`, `max_epochs=20`, `sample_time=16`
  - BC anneal `0.10 -> 0.05`, `reference_kl_coeff=0.02`
  - `grpo_normalize_advantage_batch=true`, `grpo_advantage_clip_abs=5.0`
  - main GRPO TTC/DDC hard masks enabled: TTC threshold `0.95`, DDC threshold `0.99`
  - offline buffer, Buffer-DPO, distill, and self-imitation disabled to isolate the RPP/TTC-DDC design
- Evaluation watcher:
  - `training-vla-zt2` only, with strict wait thresholds `EVAL_GPU_MAX_MEM_USED_MB=2000`, `EVAL_GPU_MAX_UTIL=5`
  - `SECONDARY_EVAL_HOST` empty; zt3 is not used
- Initial verification:
  - `strict_gspo_launch_config.txt` and `train/resolved_command.txt` confirm the intended parameters.
  - Local process tree is alive and loading/building the train job.

Buffer-DPO status:
- `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z` produced `step-step=300.ckpt`.
- zt2 watcher started exact navtest eval for step300 at `2026-06-15T03:55:22Z`.
- Training diagnostics at step299:
  - reward `0.815543`
  - base reward `0.810751`
  - safe ratio `0.925781`
  - mean EP `0.838042`
  - mean TTC `0.843750`
  - buffer selected valid ratio `0.987171`
  - buffer best-valid minus GT `0.069830`
- Caveat: the active Buffer-DPO process predates the logging whitelist fix below, so TensorBoard does not expose `grpo_buffer_preference_dpo_*` diagnostics even though the planner adds the Buffer-DPO loss into `total_loss`.

Code change:
- Patched `navsim/planning/training/agent_lightning_module.py` to log GRPO Buffer-DPO diagnostics:
  - `grpo_buffer_preference_dpo_loss`
  - `grpo_buffer_preference_dpo_weight`
  - `grpo_buffer_preference_dpo_target_ratio`
  - `grpo_buffer_preference_dpo_pair_count`
  - `grpo_buffer_preference_dpo_active_row_ratio`
  - reward gap / logit / margin / timestep diagnostics
- This patch changes logging only. It does not change the training objective, trajectory inference, PDM scoring, or any navtest/navtrain split behavior.
- Verification: `python -m compileall navsim/planning/training/agent_lightning_module.py` passed.

### 2026-06-15 Retrospective: early-stopped lines under the epoch4/step5000 comparison rule

Reason:
- The comparison rule was tightened after several rounds: do not reject a Stage3
  RL method only from step300 unless it is catastrophically below the original
  Stage3 early band. Prefer judging promising methods around epoch4 or roughly
  step5000, and compare effective batch / seen train scenes.
- Re-audit source files:
  - `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5000_eval_audit_run_summary.tsv`
  - `/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_run_summary.tsv`
  - per-run `navtest_pdms_analysis.tsv` files under `/mnt/project/VLA-AD/outputs/stage3_grpo_*`

Lines that may have been stopped too early:
- `stage3_grpo_softsafety_auxdecay_s16_lr1e4_b2acc4_20e_local8_20260615T152043Z`
  - Evidence:
    - step300 PDMS `0.883328`
    - step600 PDMS `0.881065`
    - step900 PDMS `0.892335`
    - step1200 PDMS `0.886896`
    - epoch0-step1330 PDMS `0.888593`
  - Interpretation:
    - This run has the clearest "do not judge from step300/600" pattern. It
      dipped at step600, then became the best non-clean improvement at step900.
    - It was not evaluated near epoch4/step5000, so it was too early to make a
      final negative call.
  - Next action:
    - Worth a controlled rerun or resume-style run to at least step3000/epoch2
      and preferably epoch4/step5000.
    - Keep the aux/distill weight decaying so the auxiliary target does not
      dominate later GRPO progress.
    - Evaluate every 300 steps and keep submetric gates focused on TTC/DDC
      monitoring rather than main hard gating.

- `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`
  - Evidence:
    - step300 PDMS `0.885432`
    - step600 PDMS `0.885497`
    - step900 PDMS `0.887107`
  - Interpretation:
    - The absolute score was below clean GRPO and below aux-decay/buffer-distill,
      but the curve was still slowly improving when stopped.
    - Under the new rule, cap05 should not be treated as fully falsified. It is
      evidence that capped on-policy self-imitation can be absorbed by the
      diffusion planner, though the target policy still traded EP and safety.
  - Next action:
    - Worth revisiting as an upgraded component, not as an exact old rerun:
      preserve scene cap `0.5`, strict target safety filters, low-noise target
      timesteps, and small warmup/decay weights.
    - Do not combine it with main TTC/DDC hard gates by default; the v3 hard-gate
      line showed EP suppression.

- `stage3_grpo_softsafety_bufferdistill_s16_lr1e4_b2acc4_20e_20260615T054137Z`
  - Evidence:
    - step300 PDMS `0.890785`
    - step600 PDMS `0.891373`
    - step900 PDMS `0.888205`
    - step1200 PDMS `0.885557`
    - epoch0-step1330 PDMS `0.884028`
    - step2100 PDMS `0.890360`
    - step2400 PDMS `0.882233`
  - Interpretation:
    - This was not stopped at the first dip; it had evidence through step2400.
      However, it still did not reach epoch4/step5000, and the later recovery at
      step2100 means the line has real signal rather than pure early noise.
    - The exact run also shows instability: high PDMS points alternate with sharp
      safety/EP shifts. Continuing the exact same static distillation is less
      valuable than running a decayed/isolation variant.
  - Next action:
    - Continue the current buffer-bonus-only isolation run until step300/600
      evals are known.
    - If buffer-bonus is weak, return to a decayed buffer-distill + capped
      self-imitation recipe and judge it at step3000-5000.

Lines that were probably not stopped too early:
- `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`
  - step300 PDMS `0.859864`, with weak DAC/TTC/EP. This is below the original
    Stage3 early `0.88+` band by a large margin. More epochs are not justified
    unless the DPO implementation is revised toward the reference Diffusion-DPO
    matched-noise/matched-timestep objective and diagnostics are complete.
- Simplified trajectory-level / step-level replay attempts:
  - trajectory replay step300 PDMS `0.816440`
  - simplified all-step replay step600 PDMS `0.824322`
  - These are far below the early gate and reflect implementation/formulation
    gaps versus DDPO/DPPO/RIPT-VLA, not just insufficient epochs.
- Main hard TTC/DDC gate plus strict self-imitation v3:
  - step600 PDMS `0.882934`, with EP around `0.792348`.
  - The failure mode is clear: safety gating suppressed progress. Do not
    continue this exact line.
- High-LR `2e-4` GRPO/buffer-guided variants:
  - epoch0-step1330 PDMS `0.872059` for the evaluated high-LR GRPO control.
  - This supports the user's correction that the later failure was LR-related.
    Do not revisit high LR before the original-LR lines are exhausted.

Prioritized rerun candidates after the active buffer-bonus eval:
1. Aux-decay / soft-safety auxiliary line to epoch4/step5000, because it already
   reached `0.892335` at step900 after a weak step600.
2. Cap05-style capped self-imitation, upgraded with strict target filters and
   decay, because it was still rising at step900 and represents a different
   absorption path from static buffer distillation.
3. Decayed buffer-distill plus capped self-imitation, if buffer-bonus-only proves
   too weak. The old buffer-distill line has high early points but unstable
   later behavior; the next version must reduce auxiliary dominance over time.

Current operational implication:
- Do not start a new run until the active buffer-bonus-only step300 exact eval
  returns, because that run is the isolated test of whether the train-only
  high-PDMS buffer helps as a small reward-neighborhood prior.
- Low-step checkpoints are health diagnostics, not default stop criteria. Do
  not stop a run at step300/600/900 solely because it is below cap05 or
  aux-decay, since the new comparison rule requires judging promising methods
  closer to epoch4 or step5000.
- If buffer-bonus step300 is below the cap05/aux-decay band (`<0.887`), mark it
  as lower priority and inspect submetrics/training diagnostics, but continue
  unless there is a clear implementation failure, reward collapse, NaN/loss
  instability, or a catastrophic navtest band comparable to the rejected
  Buffer-DPO/replay failures.
- Default continuation target for any non-catastrophic GRPO-family variant is
  at least step3000 and preferably epoch4/step5000 with every-checkpoint
  evaluation.
- Only free resources early if there is positive evidence that the method is
  not worth continuing, not merely because an early checkpoint is behind another
  early checkpoint.

### 2026-06-15 Operational update: disable automatic low-step early stopping

User correction:
- Do not casually stop a run at low training step or low epoch. This wastes
  training/evaluation time unless there is strong evidence that the method is
  not worth continuing.

Code / config changes:
- `scripts/training/launch_recogdrive_stage3_grpo_buffer_guided_2b_local_stable.sh`
  now defaults `EARLY_GATE_STOP_ON_FAIL=0` instead of `1`.
- `scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh` now writes
  the early gate rule as a report-only rule unless `early_gate_stop_on_fail=1`
  is explicitly set.
- The active buffer-bonus run's generated `early_gate_config.txt` was corrected
  to the same report-only wording.

Active run verification at `2026-06-15T23:22:25Z`:
- Run:
  - `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z`
- Training:
  - alive on local 8GPU
  - latest TensorBoard summary step `449`
  - latest reward `0.801857`
  - safe ratio `0.933594`
  - latest checkpoint still `step-step=300.ckpt`; `step600` not yet written
- Evaluation:
  - zt2 memfit watcher is evaluating `step-step_300` exact navtest on GPUs 4-7.
  - It had processed rank0 progress through `400 / 3035` scenarios.
- Early gate:
  - current buffer-bonus watcher has `STOP_ON_FAIL=0`.
  - stale DPPO diagnostic watcher also has `STOP_ON_FAIL=0`.
  - No active early-gate process has automatic termination enabled.

Decision:
- Continue buffer-bonus-only training. Do not stop it based on step300/600 alone.
- Continue exact PDMS evaluation for every available checkpoint.
- Revisit resource allocation only after longer-horizon evidence, preferably
  step3000-5000 / epoch4, unless an implementation failure, NaN/loss collapse,
  or clearly unrecoverable reward/submetric collapse appears.

Follow-up verification at `2026-06-15T23:27:29Z`:
- The active buffer-bonus run still has only `step-step=300.ckpt`; `step600`
  has not been written yet.
- TensorBoard scalar summary is still at step `449`, but all 8 child training
  workers are alive and using about one CPU core each, and local GPUs show high
  utilization. Treat this as continued execution, not a stop condition.
- zt2 exact step300 eval is still running and rank0 has progressed to
  `800 / 3035` scenarios. No `checkpoint_eval_submetrics.tsv` row is available
  yet.
- The active early-gate watcher continues to log `stop_on_fail=0`; low-step
  PDMS results are report-only unless a future run explicitly opts into
  `EARLY_GATE_STOP_ON_FAIL=1`.

Standing rule:
- Do not stop any non-catastrophic GRPO-family run at low step/epoch merely
  because its early PDMS is behind another early checkpoint. Require either
  longer-horizon evidence around step3000-5000 / epoch4, or a concrete failure
  mode such as crash, NaN/loss collapse, invalid reward implementation, or
  catastrophic submetric collapse comparable to the previously rejected replay
  attempts.

Tooling hardening at `2026-06-15T23:30Z`:
- Added `--min-stop-step` to
  `scripts/training/gate_recogdrive_stage3_early_pdms.py`.
  - A checkpoint below `min_stop_step` can still report low PDMS, but the gate
    returns `watch` instead of `stop`.
  - The JSON now includes `selected_checkpoint_step`, `best_checkpoint_step`,
    `latest_checkpoint_step`, and `min_stop_step`.
- `scripts/training/watch_recogdrive_stage3_early_gate.sh` now defaults
  `MIN_STOP_STEP=3000` and passes it to the gate.
- Stage3 stable/wrapper launchers now default
  `EARLY_GATE_MIN_STOP_STEP=3000` and pass it through to the watcher.
- Smoke checks:
  - `python -m compileall scripts/training/gate_recogdrive_stage3_early_pdms.py`
  - `bash -n` for the Stage3 early-gate watcher and GRPO launcher wrappers.
  - Synthetic step300 low-PDMS row returns `action=watch`.
  - Synthetic step3000 low-PDMS row returns `action=stop`.

Runtime check at `2026-06-15T23:32Z`:
- The active buffer-bonus run advanced from TensorBoard step `449` to `499`,
  confirming the previous event-file pause was not a hang.
- The latest train reward is `0.800910`, base reward `0.803502`, safe ratio
  `0.910156`, and LR remains `1e-4`.
- No new checkpoint has been written yet beyond `step-step=300.ckpt`.
- zt2 exact eval for step300 advanced to rank0 `1200 / 3035`; no PDMS summary
  row is available yet.
- Decision unchanged: keep training and eval running; do not start competing
  jobs while local GPUs and zt2 GPUs are already saturated.

Runtime check at `2026-06-15T23:35Z`:
- No `step600` checkpoint yet; latest checkpoint remains `step-step=300.ckpt`.
- No step300 PDMS summary row yet.
- zt2 exact step300 eval advanced to rank0 `1400 / 3035`.
- The active run's generated `early_gate_config.txt` was updated to document
  `early_gate_min_stop_step=3000`. This is a config-record sync only; no
  training or remote evaluation process was stopped or restarted.

Runtime check at `2026-06-15T23:39Z`:
- Training advanced to TensorBoard step `549`, so the run is not hung.
- Latest train reward `0.849070`, base reward `0.838997`, safe ratio
  `0.933594`, mean EP `0.854028`, TTC `0.890625`, comfort `0.738281`.
- Latest checkpoint is still `step-step=300.ckpt`; `step600` has not been
  emitted yet.
- zt2 exact step300 eval advanced to rank0 `1700 / 3035`; no PDMS summary row
  is available yet.

Runtime check at `2026-06-15T23:50Z`:
- `step-step=600.ckpt` was written at `2026-06-15T23:47:45Z`.
- Training summary advanced to TensorBoard step `599`.
- Latest train reward `0.856061`, base reward `0.838416`, safe ratio
  `0.949219`, mean EP `0.851869`, TTC `0.925781`, comfort `0.632812`.
- zt2 exact step300 eval advanced to rank0 `2600 / 3035`; no PDMS summary row
  is available yet.
- The active zt2 watcher is configured with `EVAL_CHECKPOINT_STEP_INTERVAL=300`,
  so `step600` should be picked up after the in-flight step300 eval completes.

Runtime check at `2026-06-15T23:58Z`:
- Exact navtest PDMS for buffer-bonus-only `step-step=300` completed:
  - PDMS `0.876675`
  - NC `0.982575`
  - DAC `0.964574`
  - TTC `0.955429`
  - EP `0.802436`
  - comfort `0.999918`
  - DDC `0.962020`
  - TLC field was blank in the summary row.
  - `num_valid_rows=12138`, `num_rows=12139`.
- `step300` is below the stronger early historical candidates such as
  `softsafety_auxdecay` step900 `0.892335` and cap05-style step900 `0.887107`.
  This is a prioritization signal, not a stop signal: under the current
  comparison rule, do not stop this run at low step unless an implementation
  failure or catastrophic submetric collapse appears.
- The watcher immediately archived and started exact eval for
  `step-step=600` at `2026-06-15T23:56:04Z`; rank0 reached `100 / 3035` at
  `2026-06-15T23:58:28Z`.
- Training continued to TensorBoard step `649`; latest train reward `0.736870`,
  base reward `0.744943`, safe ratio `0.910156`, mean EP `0.751109`, TTC
  `0.878906`, comfort `0.589844`. Treat this as training-batch noise unless
  sustained across later summaries/checkpoints.

Runtime / reference check at `2026-06-16T00:02Z`:
- Active buffer-bonus-only run is still alive on local 8 GPUs:
  - Run:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z`
  - Latest checkpoint remains `step-step=600.ckpt`.
  - Latest TensorBoard summary is step `649`, seen scenes `41536`, train reward
    `0.736870`, base reward `0.744943`, safe ratio `0.910156`, mean EP
    `0.751109`, TTC `0.878906`, comfort `0.589844`, reference KL loss
    `0.010163`, LR `1e-4`.
- zt2 exact PDMS eval for `step-step_600` is running normally:
  - watcher root:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/step300_memfit_watch_on_vla_zt2_4gpu`
  - Started `step-step_600` at `2026-06-15T23:56:04Z`.
  - Rank0 progressed to `300 / 3035` by `2026-06-16T00:01:05Z`.
  - No step600 summary row is available yet.
- Resource status:
  - Local GPUs are occupied by the active 8GPU training.
  - zt2 GPUs are occupied by the exact PDM evaluation and existing unrelated
    watchers/jobs. No remote task was killed or preempted.
- Decision:
  - Continue this run. The step300 PDMS `0.876675` is weak versus the best
    early historical variants, but it is not catastrophic and does not satisfy
    the new low-step stop rule.
  - Wait for step600, then keep evaluating each checkpoint. Do not make a final
    negative decision before step3000-5000 / epoch4 unless there is a concrete
    implementation failure or sustained submetric collapse.

Reference-backed implementation notes for the next GRPO-family design:
- DPPO paper/repo:
  - Paper: https://arxiv.org/abs/2409.00588
  - Official code: https://github.com/irom-princeton/dppo
  - Relevant lesson: do not replace diffusion policy fine-tuning with pure
    supervised regression to high-score trajectories. Mature diffusion-policy RL
    keeps policy-gradient optimization of sampled denoising trajectories as the
    main update, with PPO-style control, rollout batches, and careful denoising
    step handling.
- RIPT-VLA:
  - Paper: https://arxiv.org/abs/2505.17016
  - Relevant lesson: VLA post-training uses dynamic rollout sampling and
    leave-one-out advantages as the stable core. This supports keeping
    GRPO/RLOO as the main ReCogDrive Stage3 path rather than switching back to
    pure offline AWAC.
- Diffusion-DPO:
  - Paper: https://arxiv.org/abs/2311.12908
  - Official code: https://github.com/SalesforceAIResearch/DiffusionDPO
  - Relevant lesson: preference optimization for diffusion models needs a
    diffusion-likelihood / ELBO-consistent pair objective, shared/noisy
    timestep treatment, reference-policy control, and large effective batches.
    The failed Buffer-DPO line should therefore not be interpreted as "DPO is
    impossible"; it shows that our light auxiliary was not yet a mature
    diffusion preference implementation.
- DSPO:
  - OpenReview: https://openreview.net/forum?id=xyfb9HHvMe
  - Relevant lesson: score-matching-aligned preference objectives may be better
    matched to diffusion pretraining than direct LLM-style DPO adaptations.
    If we revisit preference learning, the next implementation should be closer
    to score/noise prediction preference alignment, not another thin pairwise
    wrapper around the current auxiliary loss.

Current algorithm implication:
- The most defensible near-term path is still GRPO-centered:
  1. finish the active buffer-bonus-only isolation run to a fair horizon;
  2. if it remains weak, rerun the best early line, aux-decay soft-safety /
     buffer-distill, to step3000-5000 with decayed auxiliary weight;
  3. treat preference/buffer absorption as a carefully implemented auxiliary
     only after it matches diffusion-DPO/DSPO mechanics well enough to be a real
     test.
- Buffer trajectories with train PDMS around `0.97` are useful, but they should
  not be forced into the policy by broad AWAC-style regression. The policy must
  learn to assign higher probability to those trajectories while preserving the
  on-policy GRPO exploration distribution and NC/DAC/TTC/DDC guards.

Code-path audit at `2026-06-16T00:05Z`:
- Existing mature pieces already present:
  - `stage3_objective=grpo_replay` enters manual optimization in
    `AgentLightningModule`.
  - `collect_grpo_replay_rollout` stores fixed sampled chains, old trajectory
    logprob, old per-denoising-step logprob, advantages, replay masks, and
    optional BC chains.
  - `compute_grpo_replay_loss` supports trajectory-level PPO replay and
    step-level replay with transition minibatches.
  - `ppo_replay_step_clip_schedule=dppo_exp` exists and logs per-step clip
    widths / selected denoising step indices.
  - Reference-KL is computed from reverse-process transition distributions, with
    chunking available through `reference_kl_chunk_size`.
- Gaps versus a full DPPO-style experiment:
  - The only memory-safe DPPO diagnostic used effective batch `8`, not the
    effective batch `64` of the current b2/acc4/8GPU GRPO controls. Its low-step
    checkpoints are therefore unfair for direct comparison.
  - The transition-replay path was launched as a short diagnostic and did not
    yet produce an exposure-aligned exact navtest PDMS row in the main summary.
  - The main active run is `stage3_objective=none` / standard GRPO with buffer
    reward bonus only; it is not exercising the DPPO replay machinery.
  - The Buffer-DPO auxiliary is more complete than the earliest lightweight
    version because it reuses shared noise/timestep and a frozen reference loss,
    but it remains an auxiliary MSE-loss-ratio preference term. It is not yet a
    DSPO-style score-preference main objective.
- Practical conclusion:
  - Do not claim the DPPO/replay idea has been falsified. The code path exists,
    but the fair experiment is still pending.
  - The next DPPO/replay run should be configured for comparable train-scene
    exposure: either raise effective batch if memory permits, or evaluate only
    at step multiples that match the b2/acc4 controls.
  - If local resources free and the active buffer-bonus run remains weak after
    the fair horizon, the next high-value experiment is either:
    1. aux-decay soft-safety/buffer-distill rerun to step3000-5000; or
    2. a memory-fit `grpo_replay` run with exposure-aligned checkpoints and no
       low-step navtest spending.

Resource check at `2026-06-16T00:06Z`:
- `/mnt/project` is at `284T / 310T` used, about `27T` free.
- Active buffer-bonus run is about `8.7G`.
- `/mnt/project/VLA-AD/outputs` is about `295G`.
- No cleanup is required before the current training/evaluation can continue.
- zt2 step600 exact eval advanced to rank0 `500 / 3035` by
  `2026-06-16T00:03:42Z`.

Launcher readiness check at `2026-06-16T00:07Z`:
- `launch_recogdrive_stage3_grpo_dppo_step_2b_local.sh`
  - Default path is `stage3_objective=grpo_replay`.
  - It uses memory-fit `batch_size=1`, `accumulate_grad_batches=1`, 8 GPUs, and
    `sample_time=16`.
  - Because effective batch is only `8`, it explicitly sets
    `CHECKPOINT_EVERY_N_TRAIN_STEPS=2400`,
    `EVAL_MIN_CHECKPOINT_STEP=2400`, and
    `EVAL_CHECKPOINT_STEP_INTERVAL=2400`. This avoids unfair low-exposure
    navtest comparisons against b2/acc4/effective64 controls.
  - Buffer distillation, Buffer-DPO, reward bonus, and self-imitation are off
    by default, so this is a clean DPPO/replay policy-gradient diagnostic.
- `launch_recogdrive_stage3_grpo_softsafety_auxdecay_2b_local.sh`
  - Default path is b2/acc4/8GPU effective batch `64`, matching the current GRPO
    controls.
  - It keeps `MAX_EPOCHS=20`, `LR=1e-4`, scheduler min LR `1e-5`, sample_time
    `16`, BC `0.10 -> 0.05`, reference KL `0.02`.
  - It keeps the train-only buffer/self-imitation teacher as a decayed auxiliary
    rather than a permanent regression objective.
  - Checkpoints remain every 300 steps and are suitable for exact navtest eval
    at each checkpoint under the current matched-step protocol.
- Stable launcher / watcher wiring:
  - `launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh` propagates
    `EVAL_MIN_CHECKPOINT_STEP`, `EVAL_CHECKPOINT_STEP_INTERVAL`, and
    `EVAL_ALWAYS_EPOCH_CHECKPOINTS` to `watch_stage3_checkpoints_eval_8gpu.sh`.
  - Early gate is report-only by default and has `EARLY_GATE_MIN_STOP_STEP=3000`.
  - This prevents repeating the previous low-step stopping mistake.
- Syntax / whitespace checks passed:
  - `bash -n` for the DPPO-step, aux-decay, selfimit, stable launcher, and eval
    watcher scripts.
  - `git diff --check` for the same scripts plus this development record.
- Runtime:
  - Active buffer-bonus `step600` exact eval reached rank0 `700 / 3035` by
    `2026-06-16T00:06:17Z`, still without traceback or result row.

Tooling update at `2026-06-16T00:10Z`: exposure-normalized run summary
- Updated `scripts/training/summarize_recogdrive_stage3_runs.py` so run
  summaries no longer rely only on raw checkpoint step.
- New output columns:
  - `baseline_effective_batch`
  - `latest_equivalent_step`
  - `latest_exposure_band`
  - `best_equivalent_step`
  - `best_exposure_band`
  - `best_low_exposure`
- Defaults:
  - baseline effective batch `64`
  - fair minimum equivalent step `3000`
  - preferred equivalent step `5000`
  - long-horizon minimum seen scenes `800000`
- Exposure bands:
  - `<600`: `launch_health`
  - `600-2999`: `early_diagnostic`
  - `3000-4999`: `fair_window`
  - `5000` until long-horizon: `preferred_fair_window`
  - long-horizon threshold: `long_horizon`
- Active buffer-bonus status from the updated tool:
  - latest TensorBoard step `699`
  - latest seen scenes `44736`
  - latest equivalent step `699.0`, exposure band `early_diagnostic`
  - best evaluated checkpoint is still `step-step_300`
  - best seen scenes `19200`
  - best equivalent step `300.0`, exposure band `launch_health`
  - `best_low_exposure=True`
  - recommendation is now `continue_until_fair_exposure_then_compare`
- Validation:
  - `python -m compileall scripts/training/summarize_recogdrive_stage3_runs.py`
  - Active run summary writes TSV/JSON successfully.
  - `git diff --check` passed for the summary script and this development
    record.
- Runtime:
  - Active buffer-bonus `step600` exact eval reached rank0 `1000 / 3035` by
    `2026-06-16T00:10:12Z`, still without traceback or result row.

Tooling update at `2026-06-16T00:13Z`: checkpoint-step fallback for summaries
- Issue:
  - In lightweight shells without the `tensorboard` package, the run summarizer
    could find checkpoint/eval artifacts but could not read TensorBoard scalar
    tags, so active runs were reported with blank `latest_step` and
    `latest_exposure_band=unknown`.
- Fix:
  - `scripts/training/summarize_recogdrive_stage3_runs.py` now falls back to
    the newest checkpoint filename when scalar tags are unavailable.
  - This is conservative because it uses the last persisted checkpoint step,
    not the possibly-later in-memory training step.
- Validation:
  - `python -m compileall scripts/training/summarize_recogdrive_stage3_runs.py`
  - Active buffer-bonus summary now reports:
    - `latest_step=600`
    - `latest_equivalent_step=600.0`
    - `latest_exposure_band=early_diagnostic`
    - best evaluated checkpoint remains `step-step_300`
    - recommendation remains `continue_until_fair_exposure_then_compare`
- Runtime:
  - Active buffer-bonus `step600` exact eval reached rank0 `1200 / 3035` by
    `2026-06-16T00:12:49Z`, still without traceback or result row.
- Decision:
  - Do not stop the current run based on step300. Continue until at least
    fair exposure (`equivalent_step >= 3000`) unless a concrete implementation
    or runtime failure appears.

Runtime/resource update at `2026-06-16T00:16Z`: zt2 is fully occupied
- zt2 status:
  - GPU 0-3: keep-best elite buffer generation, cycle `0002`, still updating
    builder logs.
  - GPU 4-7: exact navtest PDMS evaluation for active buffer-bonus GRPO
    checkpoint `step-step_600`.
  - No zt2 GPUs are currently free, and no unrelated remote task was killed.
- Keep-best buffer source:
  - Run root:
    `/mnt/project/VLA-AD/outputs/stage3_awac_keepbest_buffer_zt2_gpu0123_softsafety_source_20260615T045541Z`
  - Record cache:
    `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`
- Cycle `0001` oracle summaries:
  - Per-shard `mean_best_valid_reward` is about `0.9753-0.9755`.
  - Per-shard `mean_gt_reward` is about `0.9320-0.9324`.
  - Per-shard `pct_best_valid_above_gt` is about `58.8-59.1%`.
  - `has_valid_candidate_ratio=1.0`.
  - `selected_valid_ratio_mean` is about `0.979`.
  - Major best-valid sources are `gt`, `policy`, and
    `progress_endpoint`, with smaller contributions from speed/lateral/timing.
- Full cache validator:
  - Command:
    `python scripts/training/validate_recogdrive_stage3_awac_elite_buffer.py --buffer-dir <cache> --strict-v2 --allow-low-valid-ratio`
  - `num_records=85109`
  - `mean_candidates_per_record=9.3017`
  - `valid_candidate_ratio=0.9936`
  - `has_valid_candidate_ratio=1.0`
  - `mean_gt_reward=0.93227`
  - `mean_il_reward=0.70765`
  - `mean_best_valid_reward=0.98004`
  - `pct_best_valid_above_gt=0.58715`
  - `pct_best_valid_above_il=0.79630`
  - Best-valid source distribution:
    - `gt`: 35076
    - `policy`: 21881
    - `progress_endpoint`: 21442
    - `progress_speed`: 3011
    - `progress_gamma`: 1045
    - `timing_slow_first`: 831
    - `timing_delay`: 821
    - `il`: 617
    - `lateral_offset`: 260
    - `endpoint_lateral`: 125
- Interpretation:
  - The oracle/buffer generation path is not the bottleneck: it has broad
    train coverage, strict valid masks, and high best-valid rewards.
  - The main unresolved problem is policy absorption: how to make the diffusion
    planner reliably output those high-quality trajectories on navtest.
  - Therefore the next training effort should remain GRPO-centered with buffer
    guidance or a more mature preference/policy-gradient objective, not a
    return to AWAC as the primary Stage3 objective.

Active-run status at `2026-06-16T00:18Z`
- Active run remains:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z`
- Using navsim Python with TensorBoard support, the run summary reports:
  - latest TensorBoard step `749`
  - latest seen scenes `47936`
  - latest equivalent step `749.0`
  - exposure band `early_diagnostic`
  - train_reward/shaped_reward `0.70016`
  - base_reward `0.73374`
  - safe_ratio `0.85938`
  - mean_ep `0.71885`
  - mean_ttc `0.88281`
  - mean_comfort `0.70313`
  - group_reward_std `0.33275`
  - reference_kl_loss `0.00999`
  - bc_coeff `0.10`
  - lr `1e-4`
- The same summary with a lightweight Python lacking TensorBoard now falls
  back to latest checkpoint step `600`, so exposure is conservative rather
  than unknown.
- Active exact navtest eval:
  - `step-step_600` reached rank0 `1600 / 3035` by
    `2026-06-16T00:18:01Z`.
  - No error or result row yet.
- Resource decision:
  - Do not launch an additional run right now: local 8GPU is training, zt2
    GPU 0-3 is keep-best buffer generation, and zt2 GPU 4-7 is exact eval.
  - Keep the current run alive until at least fair exposure
    (`equivalent_step >= 3000`), unless a concrete runtime or implementation
    failure appears.
  - If this run remains weak at fair exposure, next GRPO-family priority is:
    1. `softsafety_auxdecay` / `softsafety_selfimit` retrained to the same
       fair exposure because it has the strongest historical buffer-absorption
       early point (`step900` PDMS `0.892335`).
    2. exposure-aligned `dppo_step` only after the policy-gradient replay path
       reaches its planned `step2400`-style equivalent checkpoint; do not judge
       it from raw low-step numbers.
  - Do not restart AWAC/IQL or current Buffer-DPO as a primary path; the buffer
    is strong, but those absorption objectives have not transferred well in
    our implementation.

Active buffer-bonus result at `2026-06-16T00:38Z`: `step-step_600`
- Exact navtest PDMS eval completed successfully:
  - checkpoint: `step-step_600`
  - CSV:
    `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/step300_memfit_watch_on_vla_zt2_4gpu/eval_step-step_600/hydra/stage3_safe_diffgrpo_eval_exact_pool_pdm/2026.06.15.23.56.17/2026.06.16.00.37.22.csv`
  - rows: `12139`, valid rows: `12138`
  - PDMS: `0.8887223712736164`
  - NC: `0.9829461196243203`
  - DAC: `0.9681166584280771`
  - TTC: `0.9493326742461691`
  - EP: `0.8323732325822251`
  - comfort: `1.0`
  - DDC: `0.9689817103311913`
- Change from `step-step_300`:
  - PDMS: `+0.012047195851`
  - NC: `+0.000370736530`
  - DAC: `+0.003542593508`
  - TTC: `-0.006096556270`
  - EP: `+0.029936974818`
  - comfort: `+0.000082385896`
  - DDC: `+0.006961608173`
- Current training summary after this eval:
  - latest TensorBoard step: `849`
  - latest equivalent step: `849.0`
  - train_reward: `0.80020`
  - base_reward: `0.79361`
  - safe_ratio: `0.92578`
  - group_reward_std: `0.27581`
  - reference_kl_loss: `0.022996`
  - lr: `1e-4`
  - best evaluated checkpoint is now `step-step_600`.
- Interpretation:
  - This is a clear recovery from the weak step300 launch-health point and is
    above the matched current-repo clean-GRPO step600 reference recorded earlier
    (`0.886777`), but still below the strongest buffer-absorption early point
    from auxiliary-decay (`step900` PDMS `0.892335`).
  - The improvement is driven mainly by EP, DDC, and DAC; TTC regressed slightly.
  - Continue the run to fair exposure (`equivalent_step >= 3000`) and monitor
    whether TTC recovers without losing EP/DDC.
  - Do not stop or promote based only on `step600`; this remains an
    early-diagnostic result.

Active buffer-bonus status at `2026-06-16T00:46Z`: `step-step_900` eval running
- Training produced:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/train/hydra/training_recogdrive_agent/2026.06.15.21.55.46/step_checkpoints/step-step=900.ckpt`
  at `2026-06-16T00:42:42Z`.
- The zt2 exact-PDMS watcher archived it as:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z/step300_memfit_watch_on_vla_zt2_4gpu/checkpoint_archive/step-step_900.ckpt`
  and started exact navtest evaluation at `2026-06-16T00:44:27Z`.
- Current run summary at this check:
  - checkpoint count: `3`
  - latest TensorBoard step: `899`
  - latest equivalent step: `899.0`
  - exposure band: `early_diagnostic`
  - latest train_reward: `0.68945`
  - latest safe_ratio: `0.85156`
  - best completed exact PDMS remains `step-step_600` with
    `0.8887223712736164`.
- zt2 resource note:
  - GPU 0-3 remain occupied by keep-best buffer generation.
  - GPU 4-7 are running the `step-step_900` exact-PDMS eval while also sharing
    memory with an existing two-expert Stage1 process; do not kill or replace
    those jobs.
- Decision:
  - Continue this run. `step-step_900` is pending and the run is still below the
    fair-exposure gate (`equivalent_step >= 3000`, preferably also inspect
    around `5000`).
  - Do not launch a competing Stage3 run until the active exact eval finishes or
    a genuinely free GPU slot appears; starting another training now would only
    interfere with the synchronized checkpoint-eval loop.

Active buffer-bonus result at `2026-06-16T01:27Z`: `step-step_900`
- Current active Stage3 run remains:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z`
- Config identity:
  - standard GRPO spine, `agent.stage3_objective=none` with `agent.grpo=True`
  - LR `1e-4`, max epochs `20`, scheduler min LR `1e-5`
  - local 8GPU, per-GPU batch `2`, grad accumulation `4`, effective batch `64`
  - `sample_time=16`
  - BC anneal `0.10 -> 0.05` over `5` epochs
  - reference KL coeff `0.02`, reference-KL chunk size `16`
  - GSPO ratio disabled
  - offline buffer enabled only for tiny reward-neighborhood bonus:
    `grpo_buffer_reward_bonus_weight=0.01`,
    distill/DPO/self-imitation all `0.0`
- Current progress:
  - training is still running
  - checkpoint count: `3`
  - latest TensorBoard step: `1099`
  - latest equivalent step: `1099.0`
  - exposure band: `early_diagnostic`
  - latest train_reward: `0.81192`
  - latest safe_ratio: `0.921875`
- Exact navtest PDMS results:
  - `step-step_300`: PDMS `0.8766751754`, NC `0.982575`, DAC `0.964574`,
    TTC `0.955429`, EP `0.802436`, comfort `0.999918`, DDC `0.962020`
  - `step-step_600`: PDMS `0.8887223713`, NC `0.982946`, DAC `0.968117`,
    TTC `0.949333`, EP `0.832373`, comfort `1.000000`, DDC `0.968982`
  - `step-step_900`: PDMS `0.8840528240`, NC `0.986653`, DAC `0.970918`,
    TTC `0.957901`, EP `0.808524`, comfort `0.999835`, DDC `0.964615`
- Interpretation:
  - Best completed checkpoint remains `step-step_600`.
  - `step900` improved NC/DAC/TTC over step600, but EP dropped sharply
    (`0.832373 -> 0.808524`), causing PDMS to fall by about `-0.00467`.
  - This supports the Core-Pareto motivation: the current buffer-bonus GRPO
    does not maintain the EP/TTC Pareto balance; safety recovery without EP
    preservation is not enough for the PDMS formula.
  - Still continue the existing run to the fair-exposure gate unless resources
    are needed for the new Core-Pareto E1. Do not make a final algorithm verdict
    at step900.

### 2026-06-16 Attempt: Core-Pareto GRPO v2 E1 Prototype

Motivation:
- The exact PDMS formula is `NC * DAC * (5*EP + 5*TTC + 2*comfort) / 12`.
- Prior hard TTC/DDC gates over-corrected toward safety and suppressed EP.
- Prior AWAC/IQL and buffer regression produced strong train buffers but weak
  sampled-policy transfer.
- Current buffer-bonus step900 again shows the same tradeoff: safety recovered,
  but EP fell and PDMS dropped.

Implementation status:
- Added explicit `grpo_reward_mode=core_pareto`.
- Core reward is `(5*EP + 5*TTC + 2*comfort) / 12`.
- NC and DAC are hard feasibility constraints.
- DDC is a guard/penalty only, not a positive reward component.
- EP floor is enforced through an adjusted-core penalty and positive-advantage
  down-scaling for low-EP samples.
- Pareto-front bonus is applied only inside the feasible set.
- Added Core-Pareto diagnostics to Lightning logging.
- Added launcher:
  `scripts/training/launch_recogdrive_stage3_grpo_core_pareto_2b_local.sh`
- Added run wrapper:
  `scripts/training/run_recogdrive_stage3_core_pareto_2b_local.sh`
- E1 defaults keep buffer disabled:
  `OFFLINE_RL_ENABLED=false`,
  `GRPO_BUFFER_GUIDANCE_ENABLED=false`,
  `GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0`,
  `GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT=0.0`,
  `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.0`.

Verification:
- `python -m compileall` passed for the modified planner, agent, Lightning
  module, and Core-Pareto smoke test.
- `bash -n` passed for the Core-Pareto launch/run scripts and the shared Stage3
  launch scripts.
- `scripts/smoke_test_recogdrive_core_pareto_grpo.py` passed. It verifies:
  Core formula, NC/DAC/DDC guard masks, negative invalid advantages, finite
  advantages, and that DDC does not add positive reward for otherwise equal
  Core candidates.

Next run plan:
- Launch Core-Pareto E1 as the next clean GRPO-primary experiment once GPU
  placement is available without interrupting existing jobs.
- Use the same main training spine as the current run:
  LR `1e-4`, epochs `20`, min LR `1e-5`, `sample_time=16`,
  per-GPU batch `2`, grad accumulation `4`, BC `0.10 -> 0.05`, ref-KL `0.02`.
- Evaluate every `300` steps with exact navtest PDMS.
- Treat step300 as launch health only. First real verdict should be around
  step3000-5000 / epoch3-4 unless there is an OOM or clear implementation
  failure.

### 2026-06-16 Planned Attempt: Core-Pareto GRPO v2 Full E1

Plan file:
- `reports/recogdrive_stage3_core_pareto_grpo_v2_plan_20260616.md`

Motivation:
- The earlier Core-Pareto prototype above was a useful smoke path, but it does
  not yet implement the full reference-relative Core-Pareto design.
- The active buffer-bonus GRPO run improved to PDMS `0.888722` at step600, then
  dropped to `0.884053` at step900 because EP fell while NC/DAC/TTC recovered.
  This confirms the targeted failure mode: safety-heavy shaping can trade away
  EP even though EP has equal weight with TTC in PDMS.

Reference audit:
- GRPO reference: keep critic-free same-scene group-relative policy gradients
  with group-normalized advantages rather than adding a value critic.
- Diffusion-DPO/offline diffusion-policy references: preference and offline
  regression methods need full diffusion-likelihood/noise/timestep treatment.
  E1 therefore keeps DPO/AWAC/self-imitation disabled and uses them only as
  future ablation candidates after the GRPO-primary spine is correct.

Mechanism:
- NC/DAC remain hard feasibility constraints.
- DDC is used as a relative/absolute guard only and has no positive reward term.
- EP floor is measured against max(GT, IL/reference) progress.
- Core is `(5*EP + 5*TTC + 2*comfort) / 12`.
- TTC is optimized through Core plus a soft EP/TTC tradeoff penalty, not a hard
  default gate.
- Pareto front over EP/TTC/comfort gets a small bonus; dominated valid samples
  cannot keep positive advantage by default.
- Group-state advantages explicitly handle all-valid, mixed, all-invalid, and
  all-slow groups.

Config:
- Clean GRPO primary, no buffer/AWAC/DPO/self-imitation.
- LR `1e-4`, 20 epochs, min LR `1e-5`, sample_time `16`, local 8GPU
  `b2acc4`, BC `0.10 -> 0.05`, reference KL `0.02`.
- E1 keeps phenotype buckets, adaptive dual, and buffer-neighborhood bonus
  disabled.

Expected diagnostics:
- `core_pareto_enabled=1`.
- `positive_advantage_slow_fail_ratio` near `0`.
- `ep_floor_pass_ratio`, `pareto_front_ratio`, and `core_pareto_valid_ratio`
  finite and non-collapsed.
- NC/DAC should not collapse; DDC should be guarded but not rewarded.

Failure criteria:
- NaN/Inf loss or advantage.
- reward_fn/submetric failure.
- catastrophic NC/DAC collapse or sustained all-invalid/all-slow domination.
- Do not stop solely from step300/600 underperformance unless catastrophic.

Results:
- Pending implementation, smoke test, dry-run, and launch.

## Update Template

Append a new section for every algorithm run:

```text
### YYYY-MM-DD Attempt: <short name>

Motivation:
- Previous evidence:
- Reference sources:
- Mechanism copied:

Implementation completeness:
- Required pieces present:
- Known simplifications:
- Why simplifications are acceptable, or why this is only a smoke test:

Config:
- Run name:
- LR / schedule:
- Effective batch / sample_time / steps per epoch:
- Reward/cache split:
- Safety guards:

Expected diagnostics:
- Should increase:
- Should stay stable:
- Stop if:

Results:
- Checkpoint:
- PDMS / NC / DAC / TTC / EP / comfort / DDC / TLC:
- Training diagnostics:
- Decision:
```

### 2026-06-16 Attempt: Core-Pareto GRPO v2 E1 launched

Motivation:
- Replace safety-heavy reward shaping with a PDMS-formula-aligned GRPO objective.
- Address the repeated EP-collapse pattern observed in hard-gate and buffer-bonus runs.

Implementation completeness:
- Core-Pareto advantage, diagnostics, Hydra wiring, launch scripts, and smoke tests are implemented.
- E1 intentionally disables AWAC/IQL, Buffer-DPO, buffer distillation, self-imitation, phenotype buckets, adaptive dual, and buffer-neighborhood bonus.

Config:
- Run name: `stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`
- LR/schedule: `1e-4`, 20 epochs, min LR `1e-5`.
- Effective batch/sample_time: local 8GPU, per-GPU batch `2`, grad accumulation `4`, effective batch `64`, `sample_time=16`.
- Reward/cache split: navtrain training metric cache only; navtest is evaluation-only.
- Safety guards: NC/DAC constraints, DDC guard only, EP floor, TTC soft tradeoff, Pareto front over EP/TTC/comfort.

Results:
- Local old buffer-bonus run was stopped to free 8 GPUs because step900 regressed after step600.
- Training is running under `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`.
- zt2 checkpoint watcher is running for step300/600/900/... evaluation.
- No checkpoint result yet at launch record time.

### 2026-06-21 Update: Core-Pareto GRPO v2 E1 late-run results

Current evaluation status:
- Completed navtest eval is available through `step-step_24300`.
- Training is still running toward the configured 20 epochs.
- Navtest remains evaluation-only and is not used for training reward or cache.

Best completed checkpoints so far:

| Rank | Checkpoint | PDMS | Core | NC | DAC | TTC | EP | Comfort | DDC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `step-step_21600` | `0.910274` | `0.923721` | `0.984388` | `0.980475` | `0.958148` | `0.858781` | `1.000000` | `0.977344` |
| 2 | `step-step_24300` | `0.908567` | `0.923048` | `0.981710` | `0.978827` | `0.953946` | `0.861370` | `1.000000` | `0.975284` |
| 3 | `step-step_20400` | `0.907306` | `0.922248` | `0.982287` | `0.978085` | `0.952546` | `0.860850` | `1.000000` | `0.974749` |
| 4 | `step-step_24000` | `0.907301` | `0.922169` | `0.979733` | `0.979156` | `0.946861` | `0.866346` | `1.000000` | `0.971412` |
| 5 | `step-step_23100` | `0.907023` | `0.922344` | `0.980475` | `0.978168` | `0.951475` | `0.862151` | `1.000000` | `0.974749` |

Comparison:
- Original Stage3 local historical reference: `0.9055` at `epoch9-step13300`.
- Safe DiffGRPO local reference: `0.906184`.
- Current best `step-step_21600` is `+0.004774` over original and `+0.004090` over Safe DiffGRPO.
- Multiple checkpoints now exceed both references, including `17400`, `18600`,
  `18900`, `20400`, `21600`, `23100`, `24000`, and `24300`.

Interpretation:
- Core-Pareto GRPO v2 is currently the strongest local Stage3 direction.
- The best checkpoint is not just a high-Core artifact: `step-step_21600`
  combines high Core with strong NC/DAC/TTC/DDC.
- `step-step_24300` becoming second-best shows the late-run improvement is not a
  single isolated spike.
- Volatility remains: several checkpoints between `21900` and `23700` regressed
  toward the original baseline before recovering at `24000/24300`.

Detailed summary:
- `reports/recogdrive_stage3/core_pareto_grpo_v2_results_summary_20260621.md`
- `reports/recogdrive_stage3/core_pareto_grpo_v2_experiment_log.md`
