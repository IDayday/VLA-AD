# BiT-Drive

BiT-Drive on this branch means Bidirectional Target-Constrained Diffusion Planning for left-tail NAVSIM failures. This branch only implements Step 1-3:

- Step 1: terminal/path intent prediction and terminal/path conditioning.
- Step 2: reverse trajectory consistency and terminal end-consistency losses.
- Step 3: failure-focused fine-tuning with left-tail oversampling.

Target-Constrained GRPO is intentionally not implemented here.

## Model Additions

`navsim/agents/recogdrive/bit_drive.py` adds:

- `TerminalPathHead`: predicts terminal state `[B, 3]` and path anchors `[B, 4, 3]` from the 384-d planner context plus optional status/history.
- `TargetPathTokenEncoder`: converts selected terminal/path conditions to 384-d context tokens and an action-feature summary.
- `ReverseTrajectoryDecoder`: training-only decoder predicting the first seven trajectory points from terminal/path intent.

BiT conditioning never uses the future target in evaluation. During training, `bit_use_gt_condition_prob` controls scheduled mixing between ground-truth conditions and predicted conditions. During evaluation, only predicted terminal/path conditions are used.

## Main Commands

```bash
cd /mnt/project/VLA-AD
python scripts/smoke_test_bit_drive_dummy_flow.py --device cpu
```

```bash
python scripts/check_bit_checkpoint_loading.py \
  --base-il-checkpoint /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL \
  --config configs/bit_drive/bit_step1_terminal_path.yaml \
  --output /mnt/project/VLA-AD/experiments/bit_drive/checkpoint_loading_report.md
```

```bash
python scripts/run_bit_step123_plan.py \
  --dry-run \
  --project-root /mnt/project/VLA-AD \
  --navsim-root /mnt/navsim \
  --exp-root /mnt/project/VLA-AD/experiments/bit_drive \
  --max-train-samples 1024 \
  --max-eval-samples 256
```

## Evaluation Judgment

Do not judge this branch by mean PDMS only. The primary metrics are zero-score count, DAC zero count, NC zero count, TTC zero count, and P5/P10 PDMS. Mean/median PDMS and ego progress are guardrails against collapse.
