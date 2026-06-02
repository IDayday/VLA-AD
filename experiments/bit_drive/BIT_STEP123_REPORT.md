# BiT-Drive Step 1-3 Report

Status: implementation, smoke test, checkpoint-loading check, dry-run plan, and real 256-sample NAVSIM Step 1-3 run completed.

## Run Identity

- Branch: `research/bit-drive-left-tail`
- Commit: `593f05d0c09de40f00e3a6449c58fd984ad30582`
- Method: BiT-Drive Step 1-3 only, no RL/GRPO

## Completed Validation

- `python scripts/smoke_test_bit_drive_dummy_flow.py --device cpu`: passed under conda env `navsim`.
- `python scripts/check_bit_checkpoint_loading.py --base-il-checkpoint /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL --config configs/bit_drive/bit_step1_terminal_path.yaml --output /mnt/project/VLA-AD/experiments/bit_drive/checkpoint_loading_report.md`: passed.
- `python scripts/run_bit_step123_plan.py --dry-run ...`: passed after correcting the training chunk pattern to `train_full_chunk_*` and auto-detecting `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`.

Checkpoint-loading result:

- Loaded matching Base-IL keys: 347.
- Loaded original action-head-related keys: 312.
- Expected missing BiT keys: 25.
- Missing non-BiT keys: 0.
- Shape mismatches: 0.

## Metrics

| Run | Samples | Mean PDMS | Median PDMS | P5 PDMS | P10 PDMS | Zero | DAC0 | NC0 | TTC0 | Ego Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 Base-IL | 256 | 0.8557079415 | 0.9508860729 | 0.0 | 0.5833333135 | 21 | 17 | 4 | 15 | 0.8076053096 |
| Step 1 | 256 | 0.8607919631 | 0.9450768921 | 0.0 | 0.5833333135 | 19 | 10 | 9 | 19 | 0.8049632115 |
| Step 2 | 256 | 0.8590740393 | 0.9488582423 | 0.0 | 0.5833333135 | 20 | 15 | 5 | 16 | 0.8102151942 |
| Step 3 | 256 | 0.8488056531 | 0.9428750241 | 0.0 | 0.5751451254 | 22 | 18 | 4 | 15 | 0.7965085675 |

## Dataset And Training

- Dataset/eval subset: NAVSIM `navtest`, first 256 samples from `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1/navtest_full_chunk_*`.
- Metric cache: `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`.
- Training cache: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1/train_full_chunk_*`.
- Training samples: max 1024 samples per chunk scan, 500 optimizer steps per stage, batch size 2.
- Base checkpoint: `/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL`.
- No RL/GRPO was run.

## Left-Tail Comparison

| Run | Zero Delta | DAC0 Delta | NC0 Delta | TTC0 Delta | P10 Delta | Mean Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 vs A0 | -2 | -7 | +5 | +4 | 0.0 | +0.0050840216 |
| Step 2 vs A0 | -1 | -2 | +1 | +1 | 0.0 | +0.0033660977 |
| Step 3 vs A0 | +1 | +1 | 0 | 0 | -0.0081881881 | -0.0069022884 |

## Judgment

- Step 1 reduced zero-score count from 21 to 19 and DAC0 from 17 to 10, with a small mean PDMS gain. It also worsened NC0 from 4 to 9 and TTC0 from 15 to 19, so the improvement is not clean.
- Step 2 partially retained the zero/DAC improvement but still worsened NC0/TTC0 relative to A0.
- Step 3 failure-focused fine-tuning did not improve the left tail on this run. Zero-score increased to 22, DAC0 increased to 18, P10 fell, mean PDMS fell, and ego progress fell.
- P5 PDMS remained 0.0 for all runs, so the extreme left tail is not fixed.
- Median PDMS did not collapse, but Step3 regressed median and mean enough that it should not be treated as successful.

## Remaining Failures

- Zero-score samples persist after all stages.
- DAC failures persist and worsen in Step3.
- NC/TTC regressions appear in Step1/Step2, consistent with unsafe proximity/collision-risk failure modes remaining unresolved.
- Ego progress is slightly sacrificed in Step1 and more in Step3.

## Recommendation

Do not proceed to Target-Constrained GRPO from these Step 1-3 results. Revise BiT losses and training schedule first:

- Keep Step1 terminal/path conditioning as the only tentatively useful component.
- Rework reverse consistency weighting before using Step2 broadly.
- Rework failure-focused sampling for Step3; current oversampling does not reduce left-tail failures and may overfit or destabilize the policy.
- Re-run on a larger fixed subset after loss/sampler changes before making any claim.

## Required Judgment

BiT success is not established by this run. The current evidence supports revising BiT losses/sampling rather than moving to GRPO.
