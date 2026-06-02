# BiT-Drive Real Run Plan

## Prerequisites

- Branch: `research/bit-drive-left-tail`.
- NAVSIM root: `/mnt/navsim`.
- Base-IL checkpoint: `/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL`.
- VLM hidden chunk cache under `/mnt/project/VLA-AD/cache`, preferably VLM-only chunks. BiT Step 1-3 does not require JEPA/VGGT weights.
- Experiment root: `/mnt/project/VLA-AD/experiments/bit_drive`.

## A0 Baseline

Run same-pipeline Base-IL evaluation with `configs/bit_drive/bit_ablation_base_no_bit.yaml`, export per-sample metrics, and build `failure_index/baseline_failure_index.jsonl`.

## A1 Step 1

Train `configs/bit_drive/bit_step1_terminal_path.yaml` from Base-IL. Evaluate on the same navtest subset as A0.

## A2 Step 2

Train `configs/bit_drive/bit_step2_reverse_consistency.yaml`, preferably initialized from A1. Evaluate on the same subset.

## A3 Step 3

Train `configs/bit_drive/bit_step3_failure_focused.yaml`, initialized from A2, using the A0 failure index. Evaluate on the same subset.

## Required Conclusion

The final report must state whether zero-score, DAC zero, NC zero, TTC zero, and P5/P10 PDMS improved without median PDMS or ego progress collapse. If they do not improve on real NAVSIM metrics, do not claim success and do not proceed to GRPO without revising the losses or sampling.
