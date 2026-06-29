# OneVL AR Answer + ReCogDrive DiT Stage2 Attempt

This note records the current attempt to use the OneVL AR Answer model as a
Qwen-based stage1 policy and attach the ReCogDrive DiT stage2 planner behind it.
The goal is to keep the stage2 training and evaluation path close to the
validated ReCogDrive stage2 setup while replacing the stage1 hidden-state source.

## Scope

The implementation in this branch is focused on the DiT/stage2 side inside
`VLA-AD`. OneVL stage1 data preparation, AR Answer SFT, hidden-cache generation,
and stage2 evaluation orchestration are now tracked under `scripts/onevl/`.
Runtime checkpoints, caches, predictions, and logs remain outside git.

The current design treats:

- OneVL AR Answer prompt4hist checkpoint as stage1.
- OneVL/Qwen hidden states as the VLM context consumed by ReCogDrive DiT.
- ReCogDrive stage2 small DiT as the planner architecture.
- The existing improved multi-target stage2 targets as the training target pool.

## Core Code Changes

### Variable VLM Hidden Dimension

Original ReCogDrive small-stage2 code assumed InternVL-2B hidden dimension 1536
and large-stage2 hidden dimension 3584. OneVL AR Answer uses Qwen hidden states
with hidden dimension 2560, so the planner and cache validators now accept an
explicit `vlm_feature_dim`.

Touched files:

- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/expert_cache.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `scripts/train_recogdrive_expert_chunked.py`
- `scripts/eval_recogdrive_expert_pdm.py`

The active OneVL config sets:

```yaml
vlm_size: small
vlm_feature_dim: 2560
planner_dim: 384
```

This means the DiT size remains the ReCogDrive small stage2 planner, but the
input projection is changed from `[N, 1536] -> 384` to `[N, 2560] -> 384`.

### Command Conditioning

The stage2 cache and training loop now require and pass
`high_command_one_hot`. This keeps command handling aligned with the ReCogDrive
stage2 interface instead of blocking on prompt text differences.

Training and evaluation both build planner inputs with:

```text
his_traj
high_command_one_hot
status_feature
```

### Fixed-Length Hidden Handling

The dataset collator pads variable-length `last_hidden_state` to a batch tensor.
Individual cache samples keep their original token length; padding happens at
collation time. This follows the ReCogDrive pattern and avoids truncating OneVL
hidden states before the DiT sees them.

### Stage2 Target Diversity

The training loop supports the improved stage2 target pool through optional
sample fields:

```text
support_trajectories: [B, 3, 8, 3]
support_mask:         [B, 3]
support_weights:      [B, 3]
support_scores:       [B, 3]
support_missing_mask: [B]
```

When these fields exist, each training sample selects a valid support trajectory
according to the normalized support weights and uses it as the diffusion action
target for that step. If a sample has no support target, the code falls back to
the original `trajectory` target. The train log records:

```text
stage2_pareto_enabled
stage2_pareto_used_ratio
stage2_pareto_missing_ratio
stage2_pareto_support_count_mean
stage2_pareto_selected_index_mean
stage2_pareto_selected_weight_mean
stage2_pareto_selected_score_mean
```

In the current full run, `stage2_pareto_used_ratio` has been 1.0 and
`stage2_pareto_missing_ratio` has been 0.0, confirming the improved targets are
being consumed.

### Checkpoint Saving

The chunked trainer now supports:

- `--save-every-epoch`
- `--save-every 10000`

The current run saves every epoch as `epoch_XXX.ckpt`, every 10000 steps as
`step_XXXXXXXX.ckpt`, and continuously updates `latest.ckpt`.

### Sharded Evaluation

`scripts/eval_recogdrive_expert_pdm.py` now supports:

- `--num-shards`
- `--shard-index`
- `--trajectory-output-key`

This lets the remote evaluation watcher split val6000/navtest evaluation across
8 GPUs and aggregate top-k checkpoints.

## Active Config

The OneVL stage2 config is:

```text
configs/onevl_ar_answer_stage2_small.yaml
```

Important fields:

```yaml
use_expert_features: false
use_jepa: false
use_vggt: false
vlm_size: small
vlm_feature_dim: 2560
planner_dim: 384
action_horizon: 8
action_dim: 3
sampling_method: ddim
num_inference_steps: 5
diffusion_loss_weight: 1.0
```

This is intentionally a VLM-hidden-only stage2 setting. JEPA/VGGT expert tokens
are disabled for this route.

## Current Experiment Paths

Stage1 AR Answer prompt4hist checkpoint:

```text
/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133/checkpoint-3228
```

Stage2 cache/train root:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z
```

Stage2 train directory:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z/train_full200
```

Stage2 eval root:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_eval_20260628T120059Z
```

The cache/eval orchestration now lives in `scripts/onevl/`. Key scripts used
during this attempt:

```text
scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py
scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh
scripts/onevl/watch_prompt4hist_stage2_cache_then_train.sh
scripts/onevl/build_prompt4hist_stage2_eval_caches.sh
scripts/onevl/watch_prompt4hist_stage2_eval_cache_then_eval.sh
scripts/onevl/launch_prompt4hist_stage2_eval_remote.sh
scripts/onevl/watch_onevl_stage2_eval_top5_remote.py
```

## Training Command Shape

The active training process uses:

```bash
torchrun --nproc_per_node=8 --master_port 29551 \
  scripts/train_recogdrive_expert_chunked.py \
  --config configs/onevl_ar_answer_stage2_small.yaml \
  --chunk-cache-root /mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z/cache_full103k \
  --chunk-name-pattern 'shard_*' \
  --global-epochs 200 \
  --flat-global-dataset \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --lr-scheduler official-cosine \
  --lr-scheduler-epochs 200 \
  --lr-warmup-epochs 3 \
  --min-lr 0.000001 \
  --lr-action-head 0.0001 \
  --lr-expert 0.0 \
  --jepa-align-weight 0.0 \
  --vggt-align-weight 0.0 \
  --precision bf16 \
  --final-check-precision fp32 \
  --save-every 10000 \
  --save-every-epoch \
  --seed 20260530
```

## Current Results Snapshot

Stage1 AR Answer prompt4hist navtest result after abnormal-output repair:

```text
PDMS = 0.8641601403126207
```

The earlier navtest cache was found to be misaligned: row prompt commands were
all `MOVE FORWARD` while cached planner command/status tensors contained
left/right commands. A corrected scene-prompt navtest cache was generated and
used for the latest stage2 checkpoint sweep.

Corrected stage2 checkpoint results:

```text
corrected val6000 selected best: epoch_200, PDMS = 0.8966278090198142
corrected navtest epoch120-200 best: epoch_147, PDMS = 0.8577635056565133
corrected navtest epoch120-200 mean: 0.8385811286603977
```

Interpretation:

- The corrected navtest result is much better than the broken-cache navtest
  numbers, but still does not exceed the repaired AR Answer stage1 baseline.
- `epoch_147` is the current strongest stage2 navtest checkpoint.
- Late checkpoints around `epoch_178` through `epoch_200` are more stable near
  0.85, but do not exceed `epoch_147`.
- See `docs/onevl_stage2_current_results_20260629.md` for paths, cache
  alignment notes, and the top checkpoint table.

## Known Open Questions

1. Prompt/source alignment is still not fully solved. Stage1 was retrained with
   the prompt4hist format, but OneVL prompt wording is still closer to OneVL AR
   Answer than to the original ReCogDrive stage1 prompt.
2. Hidden-state token length is padded at batch time, not truncated. This should
   match the ReCogDrive training pattern, but the effect of OneVL/Qwen prompt
   length on hidden token distribution should continue to be monitored.
3. The current stage2 result has not yet exceeded the AR Answer stage1 navtest
   result. The likely next checkpoint decision should be based on completed
   navtest top-k, not training loss.
4. OneVL SFT and cache/evaluation orchestration scripts are now tracked in this
   branch under `scripts/onevl/`; see
   `docs/onevl_ar_answer_stage1_cache_stage2_pipeline.md` for the full process.
