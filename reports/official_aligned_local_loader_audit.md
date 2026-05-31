# A0 official-aligned local-loader audit

Date: 2026-05-31

## Executive summary

Data coverage is now sufficient for A0 official Stage2 reproduction when using:

`cache/recogdrive_expert_chunks/full_v1_a0_complete`

The current non-data training path is **credible for A0-official-aligned full training** with the local chunk cache. The important official Stage2 pieces are aligned:

- 8-GPU `torchrun` entrypoint through `navsim/planning/script/run_training_recogdrive.py`.
- `ReCogDriveAgent` wrapped by `AgentLightningModule`.
- Official-style batch structure: `features, targets, tokens_list`.
- A0 feature whitelist at collate output: `history_trajectory`, `high_command_one_hot`, `last_hidden_state`, `status_feature`; target `trajectory`.
- `ReCogDriveAgent.forward` uses cached `last_hidden_state`, flattens history trajectory to `[B, 12]`, and sends `targets["trajectory"]` into the action head.
- Optimizer and scheduler use `ReCogDriveAgent.get_optimizers()`: AdamW, lr `1e-4`, weight decay `1e-4`, betas `(0.9, 0.95)`, `WarmupCosLR(min_lr=1e-6, epochs=200, warmup_epochs=3)`.
- Lightning Trainer config matches official defaults plus the requested 200-epoch override: DDP, 8 devices, batch 16, grad accumulation 1, gradient clipping norm 1.0, validation every epoch.
- Precision is official Lightning mixed precision (`16-mixed`) with fp32 model parameters before Trainer control; there is no `model.to(torch.bfloat16)` in the official-aligned path.
- Checkpoint selection includes `ModelCheckpoint(monitor="val/loss_epoch", mode="min", save_top_k=5)`.

Current remaining mismatches are not blockers for the A0 chunk-cache run, but they should be kept explicit:

1. `ModelCheckpoint` does not set `save_last=True`; `latest.ckpt` is instead saved by the extra `StepCheckpointCallback` at train end. This is additive, but crash-resume behavior is not identical.
2. The on-the-fly `Dataset` branch still returns `(features, targets)` while the ReCogDrive collate expects triples. This does not affect the intended `use_cache_without_dataset=True` local chunk-cache run.
3. The current official-aligned local dataset casts tensors to fp32 in `__getitem__`/collate. This matches the desired fp32 input-to-Lightning behavior, but is not byte-for-byte identical to the original pickle loader if a source tensor is not fp32.

Recommendation: A0-official-aligned full training can be restarted with the A0-complete chunk root. The safest pre-run cleanup is small and non-behavioral for optimization: add `save_last=True` to `ModelCheckpoint`, and keep the report/commands log paths as already implemented.

## Smoke evidence

Smoke command executed without training:

```bash
python scripts/smoke_check_official_aligned_local_loader.py \
  --cache-path cache/recogdrive_expert_chunks/full_v1_a0_complete \
  --official-log-split \
  --batch-size 16 \
  --sample-count 16 \
  --output reports/official_aligned_local_loader_smoke_a0_complete.json
```

Smoke output:

- train dataset length: `85109`
- validation dataset length: `18179`
- train/val overlap: `0`
- duplicate sampled tokens: `0`
- collated feature keys: `high_command_one_hot`, `history_trajectory`, `last_hidden_state`, `status_feature`
- collated target keys: `trajectory`
- train-only target keys in collated features: none
- sampled `last_hidden_state` lengths: min `2800`, mean `2800.0`, max `2800`
- sampled tensor dtypes: all required A0 tensors are `float32`

Full JSON: `reports/official_aligned_local_loader_smoke_a0_complete.json`.

## Official Stage2 baseline behavior

The official 2B Stage2 shell script is `scripts/training/run_recogdrive_train_multi_node_2b.sh`. It launches `navsim/planning/script/run_training_recogdrive.py` with:

- `agent=recogdrive_agent`
- `agent.lr=1e-4`
- `agent.grpo=False`
- `agent.cache_hidden_state=True`
- `agent.vlm_type=internvl`
- `agent.dit_type=small`
- `agent.vlm_size=small`
- `agent.sampling_method=ddim`
- `trainer.params.max_epochs=200`
- `trainer.params.devices=8`
- `train_test_split=navtrain`
- `use_cache_without_dataset=True`
- `force_cache_computation=False`

Evidence:

- `scripts/training/run_recogdrive_train_multi_node_2b.sh:24-48`
- `navsim/planning/script/config/training/default_training.yaml:40-68`

## Current local aligned entrypoint

`scripts/run_a0_official_aligned_8gpu.sh` matches the official command and adds reproducibility/logging:

- same script: `navsim/planning/script/run_training_recogdrive.py`
- same agent and A0 settings
- `torchrun --nproc_per_node=8`
- `trainer.params.max_epochs=200`
- `trainer.params.devices=8`
- `trainer.params.accumulate_grad_batches=1`
- `dataloader.params.batch_size=16`
- `dataloader.params.num_workers=8`
- `dataloader.params.prefetch_factor=2`
- command logged to `commands.log`
- stdout/stderr logged under `${OUTPUT_DIR}/logs`

Evidence: `scripts/run_a0_official_aligned_8gpu.sh:32-71`.

## Alignment table

| Check item | Official behavior | Local aligned behavior | Status | Risk | Evidence |
|---|---|---|---|---|---|
| Training entrypoint | `run_training_recogdrive.py` via `torchrun` | Same | aligned | low | `scripts/run_a0_official_aligned_8gpu.sh:32-35` |
| 8 GPU training | `--nproc_per_node=8`, `devices=8` | Same | aligned | low | `scripts/run_a0_official_aligned_8gpu.sh:32-50` |
| A0 expert disabled | no expert features | `agent.use_expert_features=False`, `agent.use_jepa=False`, `agent.use_vggt=False` | aligned | low | `scripts/run_a0_official_aligned_8gpu.sh:45-47` |
| Cache hidden state | `agent.cache_hidden_state=True` | Same | aligned | low | `scripts/run_a0_official_aligned_8gpu.sh:40` |
| Dataset source | official cache-only Stage2 | local chunk cache adapter, filtered by official train/val log split | aligned for A0 local-cache run | low | `navsim/planning/script/run_training_recogdrive.py:416-448` |
| Train samples | official `navtrain` train logs | `85109` train records, `85109` unique tokens | aligned | low | smoke JSON |
| Val samples | official `navtrain` val logs | `18179` val records, `18179` unique tokens | aligned | low | smoke JSON |
| Train/val overlap | none | `0` overlap | aligned | low | `run_training_recogdrive.py:434-438`, smoke JSON |
| Batch return | `features, targets, tokens_list` | Same | aligned | low | `run_training_recogdrive.py:281-310` |
| Feature keys | `history_trajectory`, `high_command_one_hot`, `last_hidden_state`, `status_feature` | Same collated keys | aligned | low | `run_training_recogdrive.py:298-304`, smoke JSON |
| Target keys | `trajectory` | Same | aligned | low | `run_training_recogdrive.py:306-310`, smoke JSON |
| Hidden-state padding | `pad_sequence(..., batch_first=True, padding_value=0.0)` | Same | aligned | low | `run_training_recogdrive.py:290-294` |
| Forward path | `AgentLightningModule -> ReCogDriveAgent.forward` | Same | aligned | low | `run_training_recogdrive.py:403-406`, `agent_lightning_module.py:63-66` |
| Cached hidden state in forward | use `features["last_hidden_state"]` | Same | aligned | low | `recogdrive_agent.py:294-295` |
| History shape | view history trajectory into `[B, 12]` | Same | aligned | low | `recogdrive_agent.py:351-356` |
| Status feature | `[B, 8]` used in state | Same | aligned | low | `recogdrive_agent.py:344-356` |
| Action target | `targets["trajectory"]` | Same | aligned | low | `recogdrive_agent.py:362-366` |
| Optimizer API | `agent.get_optimizers()` | Same via Lightning module | aligned | low | `agent_lightning_module.py:102-104` |
| Optimizer | AdamW lr `1e-4`, wd `1e-4`, betas `(0.9,0.95)` | Same | aligned | low | `recogdrive_agent.py:603-610` |
| Scheduler | `WarmupCosLR`, min lr `1e-6`, epochs `200`, warmup `3` | Same | aligned | low | `recogdrive_agent.py:612-617` |
| Trainer | Lightning DDP | Same | aligned | low | `run_training_recogdrive.py:483-489` |
| Batch and workers | batch 16, workers 8, pin memory true, prefetch 2 | Same | aligned | low | `default_training.yaml:40-45`, `run_a0_official_aligned_8gpu.sh:52-54` |
| Grad accumulation | 1 | 1 | aligned | low | `default_training.yaml:65`, `run_a0_official_aligned_8gpu.sh:50` |
| Grad clipping | norm 1.0 | Same from default config | aligned | low | `default_training.yaml:67-68` |
| Mixed precision | Lightning `16-mixed` | Same default, not overridden | aligned | low | `default_training.yaml:59` |
| True bf16 weights | not used in official Lightning path | not used | aligned | low | no bf16 conversion in `run_training_recogdrive.py`; report writes false at `run_training_recogdrive.py:257-264` |
| Val checkpoint selection | monitor `val/loss_epoch`, mode min, top 5 | Same monitor/mode/top-k | aligned | low | `run_training_recogdrive.py:478-480` |
| `save_last` | official EMA helper uses `save_last=True`; plain official script does not in this file | current run saves `latest.ckpt` only via `StepCheckpointCallback` at train end | partially_aligned | medium | `run_training_recogdrive.py:78-99`, `run_training_recogdrive.py:478-480` |
| Key step checkpoints | not part of official selection | extra exact 50k/60k/80k/100k/120k/140k/160k saves | additive | low | `run_training_recogdrive.py:35`, `run_training_recogdrive.py:78-99` |
| Eval command | fp32 PDM eval, no target teacher tokens | eval helper uses `--precision fp32`, no JEPA/VGGT target args | aligned | low | `scripts/eval_a0_stage2_checkpoints.sh:93-102` |

## Difference from `scripts/train_recogdrive_expert_chunked.py`

| Item | official-aligned local-loader | existing chunked trainer |
|---|---|---|
| Training framework | PyTorch Lightning `Trainer` | manual PyTorch DDP loop |
| Batch object reaching model | `features, targets, tokens_list`; `ReCogDriveAgent.forward` builds `BatchFeature` internally | local loop builds planner inputs directly |
| Optimizer source | `ReCogDriveAgent.get_optimizers()` | `optimizer_for(planner, args)` |
| Scheduler source | official `WarmupCosLR` | manual `official-cosine` implementation |
| Precision | Lightning `16-mixed`; model params stay fp32 before Trainer | `--precision bf16` now keeps fp32 weights unless `--true-bf16-weights` is passed |
| Checkpoint selection | `val/loss_epoch` top-k plus extra key-step saves | step/latest/best-by-train-loss behavior |
| Train/val loop | explicit train and validation DataLoaders from official log split | primarily train-only chunk loop; eval is separate script |
| Use for A0 official reproduction | preferred | diagnostic/local-fixed comparison only |

Chunked precision fix evidence:

- `--true-bf16-weights` exists: `scripts/train_recogdrive_expert_chunked.py:102-110`.
- full resume intentionally unsupported: `scripts/train_recogdrive_expert_chunked.py:750-751`.
- bf16 weight conversion is only behind the explicit flag: `scripts/train_recogdrive_expert_chunked.py:752-800`.
- precision report records dtype and optimizer metadata: `scripts/train_recogdrive_expert_chunked.py:811-832`.

## Remaining risks and minimal fixes

### Medium: `save_last=True` is not on `ModelCheckpoint`

`StepCheckpointCallback` writes `latest.ckpt` at train end, but if a run is interrupted before `on_train_end`, the official top-k checkpoints exist while `latest.ckpt` may not. This does not change optimization or validation selection.

Minimal fix:

```python
pl.callbacks.ModelCheckpoint(
    monitor="val/loss_epoch",
    mode="min",
    save_top_k=5,
    every_n_epochs=1,
    save_last=True,
)
```

### Medium: on-the-fly `Dataset` branch still emits pairs

`CacheOnlyDataset` now returns `(features, targets, token)`, but `Dataset.__getitem__` still returns `(features, targets)`. The current A0 official-aligned run uses `use_cache_without_dataset=True`, so this is not on the path. It should be fixed only if we intend to use `use_cache_without_dataset=False` with this ReCogDrive collate.

Evidence:

- `CacheOnlyDataset` triple return: `navsim/planning/training/dataset.py:116-139`
- `Dataset` pair return: `navsim/planning/training/dataset.py:283-308`
- collate expects triples: `navsim/planning/script/run_training_recogdrive.py:281-310`

### Low: type annotations are stale

Several signatures still say a batch/dataset returns two items even though ReCogDrive now uses triples. This is not a runtime bug but should be cleaned later.

Examples:

- `run_training_recogdrive.py:281-283`
- `dataset.py:78-84`
- `agent_lightning_module.py:56`

### Low: local loader casts to fp32 early

The local chunk adapter calls `.float()` for required A0 tensors. That is consistent with keeping input tensors fp32 before Lightning mixed precision, and the smoke check confirms fp32. It is still a small implementation difference from a pure pickle cache loader that simply stacks whatever dtype is stored.

## Restart recommendation

For the intended A0-official-aligned local chunk-cache reproduction, I recommend proceeding after either:

1. accepting the current `latest.ckpt` train-end behavior, or
2. applying the minimal `save_last=True` checkpoint cleanup.

No data-layer blocker remains for A0 official Stage2. No optimizer/scheduler/forward/precision blocker remains for the local chunk-cache run.
