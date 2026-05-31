# official-aligned-local-loader restart check

Date: 2026-05-31

Status: ready for A0-official-aligned local chunk-cache full training, subject to the small checkpoint cleanup noted below.

Canonical audit: `reports/official_aligned_local_loader_audit.md`

## Current A0-complete cache root

Use:

`cache/recogdrive_expert_chunks/full_v1_a0_complete`

This root is A0-complete only. It should not be used as a complete A4 expert-token cache because the 252 backfilled samples contain VLM hidden state and A0 fields, but not JEPA/VGGT target features.

## Smoke result

Smoke output:

`reports/official_aligned_local_loader_smoke_a0_complete.json`

| item | value |
|---|---:|
| official_log_split | true |
| train records | 85109 |
| train unique sample tokens | 85109 |
| train duplicate tokens | 0 |
| val records | 18179 |
| val unique sample tokens | 18179 |
| val duplicate tokens | 0 |
| train/val overlap | 0 |
| batch size checked | 16 |
| sampled examples | 16 |

Collated feature keys:

- `high_command_one_hot`
- `history_trajectory`
- `last_hidden_state`
- `status_feature`

Collated target keys:

- `trajectory`

Required tensor shapes from the smoke sample:

- `features.high_command_one_hot`: `[16, 4]`
- `features.history_trajectory`: `[16, 4, 3]`
- `features.last_hidden_state`: `[16, 2800, 1536]`
- `features.status_feature`: `[16, 8]`
- `targets.trajectory`: `[16, 8, 3]`

Required tensor dtypes: all `float32`.

Expert-token check:

- `train_only_target_keys_in_collated_features`: `[]`
- `expert_context_keys_in_collated_features`: `[]`

## Training alignment status

Aligned for the intended A0 run:

- official `run_training_recogdrive.py` entrypoint
- `AgentLightningModule(agent=ReCogDriveAgent(...))`
- official-style collate output
- cached hidden-state forward path
- AdamW + `WarmupCosLR`
- Lightning DDP, 8 devices, batch 16, grad accumulation 1
- Lightning `16-mixed`, no true bf16 weight conversion
- `ModelCheckpoint(monitor="val/loss_epoch", mode="min", save_top_k=5)`
- full fp32 eval helper with parallel checkpoint eval support

Remaining non-blocking cleanup:

- add `save_last=True` to `ModelCheckpoint` if we want crash-time `last.ckpt` behavior in addition to the existing train-end `latest.ckpt`.

## Recommendation

Proceed with the A0-official-aligned 8GPU full run after the optional `save_last=True` cleanup. No data, optimizer, scheduler, forward, precision, or validation-split blocker remains for the A0 local chunk-cache run.
