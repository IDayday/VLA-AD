# BiT-v4 Risk-Aware Intent Report

Updated: `2026-06-02 03:00:32 UTC`

Branch: `research/bit-drive-left-tail`
Commit: `593f05d`

No Target-Constrained GRPO/RL was run.

## Motivation

BiT direct conditioning is useful but unsafe as a standalone IL method:

- B5/C1 improves Mean/P10/Zero/DAC0/TTC0.
- B5/C1 increases NC0.
- Selector/reranker experiments show Base and BiT are complementary, but a learned gate is a post-hoc engineering fix rather than improving BiT itself.
- A taxonomy pass shows BiT's strict DAC/zero-fix domain is small on navtest-1024, so a tiny-domain gate is not a general method.

The v4 direction keeps BiT as a single planner and adds risk-aware intent modeling for IL-stage initialization before later RL.

## Method

BiT-v4 D1 adds:

- `BitRiskHead`: predicts `zero`, `DAC0`, `NC0`, and `TTC0` risk from context + terminal/path intent.
- `RiskTokenEncoder`: encodes predicted risks as a diffusion context token.
- Risk BCE loss trained only from non-test counterfactual/PDM labels.
- Low-strength terminal-only context conditioning inherited from C1.

This is not a selector. Inference uses one BiT planner and its own predicted terminal/path/risk tokens.

## Implemented

- `navsim/agents/recogdrive/bit_drive.py`: `BitRiskHead`, `RiskTokenEncoder`.
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`: risk config, token injection, risk loss/logging.
- `scripts/train_bit_drive_chunked.py`: `--risk-label-jsonl`, `--require-risk-labels`.
- `configs/bit_drive/v4/bit_v4_D1_risk_aware_terminal.yaml`.
- `scripts/run_bit_v4_risk_aware_plan.py`.

## Validation

Passed:

```bash
/root/miniconda3/envs/navsim/bin/python -m py_compile \
  navsim/agents/recogdrive/bit_drive.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  scripts/train_bit_drive_chunked.py \
  scripts/run_bit_v4_risk_aware_plan.py \
  scripts/smoke_test_bit_drive_dummy_flow.py

/root/miniconda3/envs/navsim/bin/python scripts/smoke_test_bit_drive_dummy_flow.py --device cpu
```

Non-test risk-label source:

`/mnt/project/VLA-AD/experiments/bit_drive/select/P11_enriched_features/counterfactual_train_merged_3336_enriched/counterfactual_samples.jsonl`

Valid risk-label counts:

- rows: `3328`
- BiT zero: `132`
- BiT DAC0: `116`
- BiT NC0: `16`
- BiT TTC0: `72`

## Navtest-1024 Results

Matched reference:

| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 Base | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 | baseline |
| B5/C1 BiT | 0.8738 | 0.8158 | 58 | 49 | 9 | 40 | 0.8008 | useful but unsafe NC |

BiT-v4:

| Method | Key Change | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D1 best | risk head + risk token strength 0.05 | 0.8732 | 0.8030 | 59 | 52 | 7 | 46 | 0.8075 | controls NC, fails DAC/TTC |
| D1 latest | latest checkpoint | 0.8659 | 0.7520 | 66 | 59 | 7 | 45 | 0.7988 | worse, not used |
| D2 best | risk head auxiliary, no risk token | 0.8714 | 0.8029 | 60 | 53 | 7 | 44 | 0.8037 | safer, weak DAC/TTC |
| D3 best | low risk loss 0.005, no token | 0.8702 | 0.8024 | 62 | 55 | 7 | 44 | 0.8025 | too weak |
| D4 best | weak risk token 0.01, TTC weight 6 | 0.8732 | 0.8031 | 59 | 52 | 7 | 46 | 0.8076 | same tradeoff as D1 |

Acceptance target was not met. D1/D4 improve Mean/Zero and control NC to `A0+1`, but DAC0 remains one above the desired `<=51`, and TTC0 regresses materially from `43` to `46`. D2/D3 reduce TTC regression but lose too much DAC/Zero improvement.

## Interpretation

Risk-aware auxiliary learning is useful for NC containment, but current fixed counterfactual labels are not enough to optimize final driving score:

- The risk head learns scene/candidate risk from VLM hidden and intent, but BCE supervision only predicts risk; it does not directly teach the planner how to avoid NC/TTC while preserving DAC gains.
- Risk token injection affects the planner, but D1/D4 show it can trade NC improvement for TTC regression.
- Removing risk token improves TTC but weakens DAC/Zero gains.

This supports using risk-aware BiT as an IL initialization component before later RL, but not claiming standalone success.

## Commands Run

Dry-run:

```bash
/root/miniconda3/envs/navsim/bin/python scripts/run_bit_v4_risk_aware_plan.py \
  --dry-run \
  --project-root /mnt/project/VLA-AD \
  --exp-root /mnt/project/VLA-AD/experiments/bit_drive/v4 \
  --max-train-samples 1024 \
  --max-eval-samples 1024 \
  --num-steps 1000 \
  --batch-size 8 \
  --precision bf16
```

D1 real run:

```bash
/root/miniconda3/envs/navsim/bin/python scripts/run_bit_v4_risk_aware_plan.py \
  --project-root /mnt/project/VLA-AD \
  --exp-root /mnt/project/VLA-AD/experiments/bit_drive/v4 \
  --max-train-samples 1024 \
  --max-eval-samples 1024 \
  --num-steps 1000 \
  --batch-size 8 \
  --precision bf16
```

D2-D4 were run manually with isolated output directories under:

`/mnt/project/VLA-AD/experiments/bit_drive/v4`

All evaluations used:

`/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`

## Next Step: D5 Conservative Risk-Regularized BiT

Do not keep tuning risk token strength. The next method-centric improvement should use risk labels to constrain the trajectory, not just predict risk:

- Load non-test counterfactual `base_pred_traj` alongside risk labels.
- For samples where BiT/B5 counterfactual has `NC0` or `TTC0`, add a conservative early-longitudinal distillation loss toward Base trajectory.
- Keep terminal/path conditioning for DAC/path correction.
- Apply conservative loss only to non-test safety-risk samples, not navtest.
- Evaluate whether this preserves D1's NC control while recovering B5's DAC/TTC gains.

This remains supervised IL. No GRPO/RL should run until this branch has a safe IL-stage initialization candidate.

Acceptance remains left-tail focused for D5:

- Mean should not collapse.
- P10 should improve or stay stable.
- Zero and DAC0 should decrease.
- NC0 and TTC0 should not materially regress.
