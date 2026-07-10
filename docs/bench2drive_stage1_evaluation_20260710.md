# Bench2Drive Stage1 Evaluation — 2026-07-10

## Decision

**Pass the Stage1 representation gate.** Use the completed Stage1 checkpoint to
rebuild the disjoint Bench2Drive train/validation hidden-state caches and train
Stage2 DiT from random initialization.

This is not a final-planner pass. The direct VLM decoder still has substantial
absolute open-loop error, particularly on turns, and Stage1 alone is not the
policy served in the 220-route evaluation. A full closed-loop run should wait
for the scratch Stage2 policy and its held-out-cache evaluation.

## Artifacts

- Stage1 checkpoint:
  `outputs/bench2drive_recogdrive_stage1_sft_20260710T065201Z`
- Held-out annotation:
  `outputs/bench2drive_recogdrive_stage1_sft_data_v1/val.jsonl`
- Machine-readable evaluation reports:
  `outputs/bench2drive_stage1_eval_20260710T090000Z`
- Evaluation entry point:
  `scripts/bench2drive/evaluate_recogdrive_b2d_stage1.py`

The comparison baseline is the released `ReCogDrive-VLM-2B` checkpoint from
which this Bench2Drive domain-adaptation run started. It is not a pristine
InternVL foundation checkpoint.

## Training completion

The run completed all 3 epochs and 1,728 optimizer steps over 36,851 samples
from 950 training clips. It trained 2,088,982,016 parameters with an effective
batch size of 64. Runtime was 7,138.5 seconds (1:58:58), and aggregate training
loss was 0.75462. The logged single-step loss moved from 1.5723 at step 1 to
0.6906 at step 1,728 with finite gradient norms throughout the run.

## Evaluation protocol

The validation split contains 2,083 examples from 50 clips that do not occur in
training. It is highly command-imbalanced: 1,965 straight, 49 left-turn, and 69
right-turn examples.

Three complementary checks were run with the exact Stage1 image preprocessing,
online prompt, history, command, and eight-point trajectory answer contract:

1. Teacher-forced token NLL on all 2,083 held-out examples.
2. Greedy autoregressive generation on a deterministic 256-example sample that
   covers all 50 validation clips.
3. Greedy generation and counterfactual NLL on all 118 held-out turn examples.

Generation is scored after parsing exactly eight `(x, y, heading)` points.
Reported confidence intervals are 95% percentile intervals from 10,000 paired
bootstrap resamples at the clip level with seed `20260710`.

## Full held-out NLL

| Metric | Released VLM | Stage1 | Relative improvement |
|---|---:|---:|---:|
| Token NLL | 1.54483 | 0.72040 | 53.37% (95% CI 51.96–54.49%) |
| Perplexity | 4.68719 | 2.05525 | 56.15% |

Stage1 had lower per-example NLL on all 2,083 validation examples. The gain is
present in every command group:

| Command | Examples | Released VLM NLL | Stage1 NLL | Reduction |
|---|---:|---:|---:|---:|
| Straight | 1,965 | 1.54338 | 0.71283 | 53.81% |
| Left | 49 | 1.53997 | 0.86764 | 43.66% |
| Right | 69 | 1.58983 | 0.82756 | 47.95% |

## Greedy generation on 256 examples

| Metric | Released VLM | Stage1 | Relative improvement |
|---|---:|---:|---:|
| Parse success | 100% | 100% | no regression |
| XY ADE | 13.5930 m | 3.6443 m | 73.19% (95% CI 68.59–77.30%) |
| XY FDE | 24.7505 m | 7.1366 m | 71.17% (95% CI 65.67–76.11%) |
| Heading MAE | 0.19873 rad | 0.13334 rad | 32.90% (95% CI 22.21–43.06%) |
| Raw trajectory L1 | 5.62139 | 1.49315 | 73.44% (95% CI 68.91–77.51%) |

Stage1 reduced ADE on 94.14% and FDE on 92.97% of the paired examples. The
sample contains only 18 turns, so the turn conclusion below uses every turn in
the held-out split rather than extrapolating from this sample.

## Greedy generation on all 118 turns

| Metric | Released VLM | Stage1 | Relative improvement |
|---|---:|---:|---:|
| Parse success | 100% | 100% | no regression |
| XY ADE | 14.5207 m | 5.4953 m | 62.16% (95% CI 51.48–70.51%) |
| XY FDE | 26.8010 m | 10.8198 m | 59.63% (95% CI 47.10–69.25%) |
| Heading MAE | 0.67150 rad | 0.36439 rad | 45.74% (95% CI 36.62–54.63%) |
| Raw trajectory L1 | 6.48563 | 2.43536 | 62.45% (95% CI 51.38–71.09%) |

The turn subset spans 16 clips. Stage1 reduced ADE on 86.44% and FDE on 83.90%
of these examples. Breakdown by requested turn direction is:

| Command | Examples | ADE, released → Stage1 | FDE, released → Stage1 | Heading MAE, released → Stage1 |
|---|---:|---:|---:|---:|
| Left | 49 | 21.1587 → 5.8180 m | 39.2442 → 10.9842 m | 0.79058 → 0.35982 rad |
| Right | 69 | 9.8068 → 5.2662 m | 17.9644 → 10.7030 m | 0.58693 → 0.36763 rad |

## Counterfactual checks

Each value below is the increase in token NLL after replacing one input or
target with a deterministic shuffled counterpart. A positive value means the
real pairing is preferred.

| Subset / perturbation | Released VLM | Stage1 |
|---|---:|---:|
| 256: shuffled image | +0.00309 | +0.05583 |
| 256: shuffled command | +0.00679 | +0.02170 |
| 256: shuffled answer | -0.00602 | +0.13554 |
| All turns: shuffled image | +0.00733 | +0.03032 |
| All turns: shuffled command | -0.01592 | +0.02536 |
| All turns: shuffled answer | -0.00127 | +0.09641 |

The Stage1 model is materially more sensitive to the correct image and
sample-specific trajectory than the released VLM. Its command sensitivity is
also positive, including when evaluated only on left/right turns, but it is
smaller than its image and answer sensitivity. This should remain a diagnostic
in Stage2 because the training and validation distributions are 94.3% straight.

## Interpretation and next gate

The improvement generalizes to held-out clips, survives clip-level paired
bootstrap testing, appears on both turn directions, and is not explained only
by learning the answer format. That is enough evidence to stop Stage1 and use
this checkpoint as the sole source of the new hidden-state caches.

The remaining absolute errors are too large to claim that direct VLM generation
is a usable closed-loop planner: Stage1 FDE is 7.14 m on the clip-covering sample
and 10.82 m over all turns. This diagnostic also does not measure whether the
hidden states expose the information in a form that a scratch DiT can learn.

The next acceptance sequence is therefore:

1. Rebuild train and validation caches from this exact Stage1 checkpoint.
2. Train Stage2 DiT/action head from random initialization.
3. Evaluate Stage2 on the held-out cache and inspect straight/left/right metrics.
4. Run the complete 220-route suite only after the Stage2 open-loop gate passes.
5. Begin Bench2Drive GRPO only from the accepted Stage2 checkpoint.
