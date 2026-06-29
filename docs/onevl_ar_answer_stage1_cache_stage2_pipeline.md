# OneVL AR Answer Stage1, Hidden Cache, and DiT Stage2 Pipeline

This document records the repository-tracked workflow used for the OneVL AR
Answer -> ReCogDrive DiT stage2 experiment. Runtime artifacts, caches,
checkpoints, and predictions stay outside git; the orchestration and analysis
code now lives under `scripts/onevl/`.

## Scope

The current route treats OneVL AR Answer as the stage1 VLM policy and feeds its
Qwen hidden states into the ReCogDrive diffusion planner. The objective is to
make the experiment process inspectable:

```text
AR Answer JSON/JSONL
  -> prompt4hist stage1 SFT
  -> AR Answer checkpoint
  -> Qwen final hidden-state cache + NAVSIM planner tensors
  -> ReCogDrive DiT stage2 training
  -> val6000/navtest sharded evaluation and top-k checkpoint backup
```

## Repository Script Map

Stage1 data preparation:

- `scripts/onevl/prepare_ar_answer_prompt4hist_dataset.py`
  converts the original AR Answer data to the prompt4hist format used by the
  latest stage1 retrain.
- `scripts/onevl/prepare_ar_answer_official_dataset.py` and
  `scripts/onevl/normalize_navsim_answer_paths.py` are helpers for the original
  OneVL AR Answer data shape and local path normalization.

Stage1 AR Answer training:

- `scripts/onevl/run_answer_training_full.sh` launches the upstream OneVL SFT
  script at `/mnt/project/OneVL_training/run_script/train/navsim/sft_distributed_qwen3vl_answer_bs64.sh`.
- `scripts/onevl/launch_answer_training_full.sh` starts the training runner in
  the background and records the command under the run root.
- `scripts/onevl/watch_answer_checkpoints_validate.sh` validates new HF
  checkpoints as they appear.
- `scripts/onevl/watch_answer_training_then_eval_latest.sh` waits for training
  to finish and then launches latest-checkpoint navtest inference/evaluation.

Stage1 inference and PDMS evaluation:

- `scripts/onevl/run_ar_answer_latest_infer_eval_full.sh` selects the newest
  valid checkpoint from a Swift output directory.
- `scripts/onevl/run_ar_answer_infer_eval_full.sh` shards the navtest JSON,
  calls the OneVL inference entrypoint, merges predictions, converts them to a
  NAVSIM submission, and optionally runs PDM scoring.
- `scripts/onevl/split_infer_dataset.py`,
  `scripts/onevl/merge_sharded_predictions.py`,
  `scripts/onevl/navsim_predictions_to_submission.py`, and
  `scripts/onevl/pdm_score_from_submission_parallel.py` implement the local
  conversion and scoring stages.
- `scripts/onevl/retry_remote_ar_answer_latest_eval.sh` dispatches the same
  latest-checkpoint navtest evaluation to a remote shared-filesystem host.

Stage2 hidden cache generation:

- `scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py` is the cache bridge.
  It loads the AR Answer Qwen checkpoint, applies the row prompt through the
  processor chat template, extracts the final hidden state, and writes the
  ReCogDrive planner tensors.
- `scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh` builds
  the full sharded train cache and can launch DiT training after cache
  validation.
- `scripts/onevl/watch_prompt4hist_stage2_cache_then_train.sh` waits for a
  cache run to finish, validates `aggregate_summary.json`, then launches DiT
  training once.

Stage2 evaluation preparation and remote top-k evaluation:

- `scripts/onevl/build_prompt4hist_stage2_eval_caches.sh` builds val6000 and
  navtest hidden caches for evaluation.
- `scripts/onevl/watch_prompt4hist_stage2_eval_cache_then_eval.sh` waits for
  eval caches and starts the top-k evaluation watcher.
- `scripts/onevl/launch_prompt4hist_stage2_eval_on_rl_zt3.sh` dispatches eval
  cache preparation and checkpoint evaluation to `training-rl-zt3`.
- `scripts/onevl/watch_onevl_stage2_eval_top5_remote.py` watches stage2
  checkpoints, evaluates eligible checkpoints on val6000/navtest, maintains
  split-specific top-k rankings, and backs up top-k checkpoint objects.

## Stage1 Prompt4Hist Data Contract

The prompt4hist conversion keeps the original OneVL AR Answer supervision while
changing the prompt to better match the stage2 planner context:

- Command terms remain OneVL-style: `MOVE FORWARD`, `TURN LEFT`, `TURN RIGHT`.
- Image paths and assistant `<answer>...</answer>` trajectory targets are not
  changed.
- Historical trajectory is changed from three points to four points by adding
  the current ego-local state `[0.00, 0.00, 0.00]`.
- The prompt no longer asks for `<think></think>` because the dataset does not
  contain CoT targets.
- The output instruction asks for only eight `[x, y, heading]` future waypoints
  inside `<answer></answer>`.

Current stage1 data paths:

```text
train source: /mnt/project/onevl_navsim_data/navsim_answer_official_paths.jsonl
train prompt4hist: /mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl
navtest prompt4hist: /mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.json
```

The stage1 wrapper defaults to the prompt4hist train JSONL. Override
`DATASET_PATH` only when intentionally reproducing an older AR Answer run.

## Stage1 Training Contract

The wrapper preserves the effective bs64 setting on 8 GPUs:

```text
NPROC_PER_NODE=8
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
effective batch size = 64
NUM_TRAIN_EPOCHS=2
LR and scheduler are inherited from the upstream OneVL bs64 script
```

The active prompt4hist checkpoint used for stage2 cache generation was:

```text
/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133/checkpoint-3228
```

The repaired stage1 AR Answer navtest PDMS recorded for that checkpoint was:

```text
0.8641601403126207
```

## Hidden Cache Bridge Contract

`bridge_ar_answer_to_recogdrive_dit.py` writes one `.pt` sample per NAVSIM row
plus `index.jsonl`, `metadata.json`, and `summary.json` per shard.

The cache sample contains:

```text
last_hidden_state:      [T, 2560] or [2800, 2560] when fixed padding is enabled
history_trajectory:    [4, 3]
status_feature:        [8]
high_command_one_hot:  [3]
trajectory:            [8, 3]
support_trajectories:  [3, 8, 3] when preserve_support is enabled
support_mask:          [3]
support_weights:       [3]
support_scores:        [3]
support_missing_mask:  scalar bool
meta:                  provenance and NAVSIM alignment metadata
```

The current full-cache settings are:

```text
--hidden-padding max_length
--hidden-max-length 2800
--hidden-padding-side left
--no-hidden-truncation
--stage2-support-mode preserve_support
--target-selection max_weight
--dit-type small
--sampling-method ddim
--no-dit-forward
```

Important implementation details:

- Hidden extraction uses `processor.apply_chat_template(..., add_generation_prompt=True)`.
- The extracted tensor is the final Qwen hidden state (`outputs.hidden_states[-1]`).
- `assistant_prefix` is appended before tokenization if set; the current run uses
  the default empty prefix.
- `valid_hidden_length` is taken from `attention_mask.sum()` and recorded in
  sample `meta`.
- Each sample records `hidden_source`, `hidden_layer`,
  `hidden_extraction_mode`, `model_checkpoint`, `processor_checkpoint`,
  `prompt_template_hash`, `input_token_count`, and `valid_hidden_length`.

The planner tensors are built from NAVSIM scene frames, not from the AR Answer
text output, unless `--use-answer-target` is explicitly set. In the current
stage2 route, `trajectory` remains the raw NAVSIM ego-local future trajectory
and support targets come from the improved stage2 support index.

## Stage2 Target Contract

The current support index path is:

```text
/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt
```

With `--stage2-support-mode preserve_support`, the cache keeps:

- `trajectory`: the raw NAVSIM future trajectory.
- `support_trajectories`: the improved three-candidate target pool.
- `support_mask`, `support_weights`, `support_scores`, and
  `support_missing_mask`: training-time target selection metadata.

The current stage2 training script consumes these fields and samples a support
trajectory during training. This design preserves target diversity instead of
baking a single selected target into the cache.

## Stage2 Training Contract

The active config is:

```text
configs/onevl_ar_answer_stage2_small.yaml
```

Key config values:

```text
vlm_size: small
vlm_feature_dim: 2560
planner_dim: 384
action_horizon: 8
action_dim: 3
sampling_method: ddim
num_inference_steps: 5
use_expert_features: false
use_jepa: false
use_vggt: false
```

The training command shape recorded by the launcher is:

```bash
torchrun --nproc_per_node=8 --master_port "${MASTER_PORT}" \
  scripts/train_recogdrive_expert_chunked.py \
  --config configs/onevl_ar_answer_stage2_small.yaml \
  --chunk-cache-root "${CACHE_ROOT}" \
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
  --save-every-epoch
```

## Stage2 Evaluation Contract

Evaluation uses the same cache bridge to build split-specific hidden caches:

```text
val6000 cache: prompt4hist val6000 JSONL + trainval NAVSIM logs
navtest cache: prompt4hist navtest JSON + test NAVSIM logs
```

`watch_onevl_stage2_eval_top5_remote.py` evaluates eligible checkpoints after
`MIN_EPOCH` and `MIN_STEP`, aggregates shard metrics, writes
`rankings/{split}/current_top5.{json,tsv}`, and backs up top-k checkpoint files
under `checkpoint_backups/`.

Current recorded stage2 run paths:

```text
train root: /mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z/train_full200
eval root: /mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_eval_20260628T120059Z
```

Current recorded stage2 results:

```text
val6000 best: epoch_087, PDMS = 0.8768169550203945
val6000 latest completed: epoch_096, PDMS = 0.8716178867422207
navtest best: epoch_080, PDMS = 0.8210427529653845
navtest latest completed: epoch_095, PDMS = 0.7595136183975849
```

These results should be interpreted as split mismatch or overfitting risk, not
as proof that the stage2 bridge is correct. The engineering contract still
needs explicit review of hidden padding masks, trajectory normalization, support
target selection, and cache provenance.

## Minimal Reproduction Commands

Prepare prompt4hist train data:

```bash
python scripts/onevl/prepare_ar_answer_prompt4hist_dataset.py \
  --input /mnt/project/onevl_navsim_data/navsim_answer_official_paths.jsonl \
  --output /mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl
```

Launch stage1 AR Answer SFT:

```bash
OUT_ROOT=/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_$(date +%Y%m%d_%H%M%S) \
bash scripts/onevl/launch_answer_training_full.sh
```

Generate full stage2 train cache only:

```bash
RUN_TRAIN=0 \
STAGE1_CKPT=/path/to/ar_answer_checkpoint \
bash scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh
```

Generate full stage2 train cache and start DiT training after validation:

```bash
RUN_TRAIN=1 \
STAGE1_CKPT=/path/to/ar_answer_checkpoint \
bash scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh
```

Dispatch eval cache generation and top-k evaluation watcher to the remote host:

```bash
REMOTE_HOST=training-rl-zt3 \
TRAIN_DIR=/path/to/stage2_train_full200 \
bash scripts/onevl/launch_prompt4hist_stage2_eval_on_rl_zt3.sh
```

## Files That Must Not Be Committed

Do not commit generated artifacts:

```text
*.ckpt
*.pt cache samples
prediction JSON files
NAVSIM submission PKL files
metric-cache directories
large logs under /mnt/project/onevl_navsim_exp
```

The repository should only track scripts, configs, and documentation needed to
reconstruct and audit the workflow.
