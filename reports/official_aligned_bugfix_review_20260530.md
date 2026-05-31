# Official-Aligned Bugfix Review 2026-05-30

## Scope

Only bug fixes were applied. The official-aligned training semantics remain:

- `ReCogDriveAgent` through `AgentLightningModule`
- local chunk cache adapter with official train/val log split
- official A0 batch keys
- AdamW + `WarmupCosLR`
- Lightning `precision=16-mixed`
- 8 GPU `torchrun --nproc_per_node=8`

## Failed Run

Failed output:

`outputs/a0_stage2_repro_20260530_135148/a0_official_aligned_strict_20260530_151516`

Observed failure:

- `step_00050000.ckpt` was saved.
- full eval for 50k completed with `PDMS=0.8474548020439976`.
- training then failed with NCCL watchdog timeout:
  `WorkNCCL(... OpType=BROADCAST/ALLREDUCE ... Timeout(ms)=600000)`.

## Bugs Found

### 1. DDP checkpoint call was rank-divergent

File: `navsim/planning/script/run_training_recogdrive.py`

`StepCheckpointCallback._save()` called `trainer.save_checkpoint()` only on global zero.
PyTorch Lightning 2.6 documents and implements this method as a distributed call that must be invoked by all ranks. The DDP strategy writes the file only on global zero and synchronizes internally.

Fix:

- Removed the early return on non-global-zero ranks.
- All ranks now call `trainer.save_checkpoint(str(path))`.
- Actual file writing remains rank0-only through Lightning's DDP strategy.

### 2. Eval watcher was sharing GPUs with 8-GPU training

The previous watcher started full eval as soon as `step_00050000.ckpt` appeared, assigning it to GPU0 while the 8-GPU training job was still using all GPUs. This violated the one-experiment-uses-one-8GPU-server rule and could amplify rank skew/timeouts.

Fix:

- Stopped the preemptive checkpoint watcher.
- Launched a post-training watcher that waits for the torchrun parent process to exit, then runs the 8 checkpoint evals in parallel across GPUs 0-7.

## Verification

Commands passed:

- `python -m py_compile navsim/planning/script/run_training_recogdrive.py scripts/smoke_check_official_aligned_local_loader.py`
- `bash -n scripts/eval_a0_stage2_checkpoints.sh scripts/run_a0_official_aligned_8gpu.sh`
- `python scripts/smoke_check_official_aligned_local_loader.py --cache-path cache/recogdrive_expert_chunks/full_v1 --official-log-split --sample-count 16 --batch-size 4 --output reports/official_aligned_local_loader_smoke.json`

Smoke result:

- dataset_len: `84918`
- feature_keys: `high_command_one_hot`, `history_trajectory`, `last_hidden_state`, `status_feature`
- target_keys: `trajectory`
- missing_keys: `{}`
- train-only target keys in collated features: `[]`

## Restarted Fixed Run

Current output:

`outputs/a0_stage2_repro_20260530_135148/a0_official_aligned_fixed_20260530_222618`

Runtime reports:

- loader_mode: `official-aligned-local-loader-log-split`
- train records: `84918`
- val records: `18118`
- train/val overlap: `0`
- requested_precision: `16-mixed`
- true_bf16_weights: `false`
- model_param_dtype_counts: `{"fp32": 327}`
- optimizer_group_lrs: `[0.0001]`
- optimizer_group_weight_decay: `[0.0001]`

Post-training eval watcher:

`outputs/a0_stage2_repro_20260530_135148/a0_official_aligned_fixed_20260530_222618/logs/a0_official_aligned.post_train_parallel_eval.log`

It waits for train pid `224232` and then runs the 8 checkpoint evals in parallel, one GPU per checkpoint.

## Remaining Watch Item

The next decisive check is whether the fixed run passes `step_00050000` and continues to `step_00060000` without NCCL timeout.
