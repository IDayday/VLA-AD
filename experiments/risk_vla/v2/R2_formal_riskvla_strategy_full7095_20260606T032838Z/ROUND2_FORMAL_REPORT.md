# RISK-VLA v2 Round2 Formal Report

Date: 2026-06-06

## Training

- Method: RISK-VLA v2 strategy-conditioned planner fine-tune.
- Train source: train-side rare-risk D5 mining labels only.
- Training samples: 7095.
- Global steps: 7096.
- Final checkpoint: `train/latest.ckpt` (not committed; binary artifact).
- Best checkpoint: `train/best.ckpt` (not committed; binary artifact).
- Leakage guard: navtest/test labels were not used for training, label construction, router supervision, or hard-negative mining.

## Candidate Assets

- Source: train-side counterfactual candidates.
- Tokens: 2000.
- Utility label rows: 10000.
- Safe alignment pairs: 1353.
- Candidate tensor shape: `[2000, 5, 8, 3]`.

Candidate-level summary:

| Candidate | Mean PDMS | Zero | DAC0 | NC0 | TTC0 | Tail Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_det | 0.916047 | 60 | 54 | 6 | 35 | 94 |
| bit_det | 0.916370 | 58 | 54 | 4 | 29 | 88 |
| bit_stochastic_seed0 | 0.918172 | 53 | 51 | 2 | 28 | 84 |
| bit_stochastic_seed1 | 0.917755 | 54 | 52 | 2 | 28 | 85 |
| bit_stochastic_seed2 | 0.918813 | 53 | 50 | 3 | 26 | 81 |

## Trajectory Risk Critic

- Train/val split: 1600 / 400 tokens.
- Best epoch: 19.
- Best validation loss: 0.728272.
- Pairwise accuracy: 0.625000.
- Selection accuracy vs constrained anchor: 0.582500.

Held-out val candidate selection:

| Selection | PDMS | Zero | DAC0 | NC0 | TTC0 | Progress | Comfort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.918833 | 11 | 9 | 2 | 6 | 0.853700 | 1.000000 |
| selected | 0.918460 | 11 | 9 | 2 | 7 | 0.855305 | 1.000000 |
| oracle | 0.924798 | 9 | 8 | 1 | 5 | 0.861014 | 1.000000 |

Risk AUROC:

- low_score: 0.723100.
- path_dac: 0.764808.
- interaction_nc: 0.658893.
- ttc: 0.623479.
- progress: 0.754170.

Interpretation: the critic learns useful risk and pairwise preference signals, but the current direct scorer selection is not yet safe enough to replace the baseline router because selected PDMS and TTC0 regress vs base on held-out candidate validation.

## Navtest10k PDM

Step2000 checkpoint, 32-shard matched navtest10k PDM:

| Run | PDMS | P1 | P5 | P10 | Zero | DAC0 | NC0 | TTC0 | Progress | Comfort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RISK-VLA v2 step2000 | 0.861617 | 0.000000 | 0.000000 | 0.583333 | 728 | 589 | 150 | 527 | 0.803414 | 0.999700 |

Reference navtest10k baselines from prior Round1 reports:

| Run | PDMS | Zero | DAC0 | NC0 | TTC0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4 baseline | 0.866242 | 658 | 519 | 146 | 540 |
| R11 | 0.865280 | 664 | 527 | 145 | 547 |
| R13 | 0.865499 | 659 | 522 | 144 | 545 |
| R14 | 0.865575 | 659 | 522 | 144 | 543 |

Interpretation: RISK-VLA v2 step2000 reduces TTC0 vs R4 (527 vs 540), but materially regresses full PDMS, zero-score, DAC0, and NC0. This checkpoint should not be treated as a winning full benchmark result.

## Current Conclusion

Risk-aware VLA improves the tail-risk direction but the fixed strategy-token / simple learned router path still over-intervenes on safe or non-critical cases. The next version should gate intervention more conservatively and use constrained candidate scoring: activate only on high-confidence tail-risk cases, suppress candidates with predicted NC/TTC/DAC regression, and use the critic as a safety filter rather than a direct selector until scorer validation clears safety gates.
