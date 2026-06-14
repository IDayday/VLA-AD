# Two-Expert Slot Stage2 Cache/Training Status

Updated: 2026-06-13 19:55 UTC

## Update 2026-06-13 21:12 UTC

- The 8GPU delayed waiter saw only 4 local GPUs become available, while the other 4 remained occupied by existing Stage3 work.
- To avoid killing or preempting other runs, Stage2 was restarted on GPUs `4,5,6,7` with equivalent global batch:
  - per-GPU batch: `16`
  - devices: `4`
  - `trainer.params.accumulate_grad_batches=2`
  - effective batch: `16 * 4 * 2 = 128`, matching the original 8GPU `16 * 8` plan.
- The launcher remains default-8GPU unless explicitly overridden through:
  - `NPROC_PER_NODE`
  - `TRAINER_DEVICES`
  - `ACCUMULATE_GRAD_BATCHES`
- A second data-path bug was found and fixed:
  - dataset samples contained `two_expert_h_dyn/two_expert_h_geo`
  - `custom_collate_fn()` dropped them before `ReCogDriveAgent.forward()`
  - fix: collate now stacks `two_expert_h_dyn` and `two_expert_h_geo`.
- Current active Stage2 run:
  `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_collatefix_4gpu_eqbs128_20260613T210233Z`
- Current run evidence:
  - command: `torchrun --nproc_per_node=4 ... trainer.params.devices=4 ... trainer.params.accumulate_grad_batches=2`
  - `precision_report.json` written
  - `data_report.json` written
  - train records: `84918`
  - val records: `18118`
  - train/val overlap: `0`
  - required two-expert hidden keys: `["two_expert_h_dyn", "two_expert_h_geo"]`
  - `use_two_expert_slots=true`
  - `use_last_vla=false`
  - `use_last_rd=false`
  - `use_expert_features=false`
  - `last_vla_use_residual_diffusion=false`
  - precision: `bf16-mixed`
  - trainable params: `37.3M`
- 5-minute startup stability check:
  - launcher and torchrun child still alive
  - GPUs `4,5,6,7` active
  - no tail evidence of `Traceback`, `RuntimeError`, OOM, `KeyError`, `ValueError`, or NaN
  - training reached Lightning DDP forward/backward startup warnings, beyond the previous first-step `two_expert_h_dyn/h_geo` failure.
- Hourly monitor for the active run:
  `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_collatefix_4gpu_eqbs128_20260613T210233Z/logs/lowfreq_monitor/hourly_status.log`

## Update 2026-06-13 22:15 UTC

- Low-frequency check used the active run hourly monitor and selected top-level files only.
- Active Stage2 run remains:
  `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_collatefix_4gpu_eqbs128_20260613T210233Z`
- Process state from the 22:05 monitor block:
  - launcher PID `890700` alive
  - torchrun child PID `890708` alive
  - elapsed time about `01:02:33`
- GPU state:
  - GPUs `4,5,6,7` active for this run
  - GPUs `0,1,2,3` remain occupied by other work
- Files present:
  - `commands.log`
  - `precision_report.json`
  - `trainable_parameter_counts.json`
  - `data_report.json`
  - `train_args.json`
  - `logs/two_expert_stage2_dit_sft.train.log`
- Confirmed config summary:
  - loader mode: `official-aligned-local-loader-log-split`
  - precision: `bf16-mixed`
  - optimizer groups: `expert` at `1e-4`, `action_head` at `3e-5`
  - train records: `84918`
  - val records: `18118`
  - overlap: `0`
  - required hidden keys: `two_expert_h_dyn`, `two_expert_h_geo`
  - `use_two_expert_slots=true`
  - `use_last_vla=false`
  - `use_last_rd=false`
  - `use_expert_features=false`
- Tail error scan found no `Traceback`, OOM, `KeyError`, `ValueError`, `ChildFailedError`, or NaN.
- Disk free on `/mnt/project`: about `646G`.

## Update 2026-06-13 23:15 UTC

- Low-frequency check used the active run hourly monitor, selected top-level files, and a shallow enumeration of the active run's own `lightning_logs`.
- Active Stage2 run remains alive:
  - launcher PID `890700`
  - torchrun child PID `890708`
  - elapsed time about `02:02:33` at the 23:05 monitor block
- GPU state:
  - current Stage2 on GPUs `4,5,6,7`
  - all four GPUs show active memory/use
- Lightning output confirms real training progress beyond startup:
  - `lightning_logs/version_0/events.out.tfevents...` exists
  - checkpoints present:
    - `lightning_logs/version_0/checkpoints/epoch=5-step=3984.ckpt`
    - `lightning_logs/version_0/checkpoints/epoch=6-step=4648.ckpt`
    - `lightning_logs/version_0/checkpoints/epoch=11-step=7968.ckpt`
    - `lightning_logs/version_0/checkpoints/epoch=13-step=9296.ckpt`
    - `lightning_logs/version_0/checkpoints/epoch=16-step=11288.ckpt`
    - `lightning_logs/version_0/checkpoints/last.ckpt`
  - checkpoint size is about `495MB` each.
- Tail error scan again found no `Traceback`, OOM, `KeyError`, `ValueError`, `ChildFailedError`, or NaN.
- Disk free on `/mnt/project`: about `568G`.
- No additional cache cleanup was performed. The largest reclaim candidate remains the old base highcap training chunk cache, but it is shared historical data and should only be removed if disk pressure becomes acute or deletion is explicitly approved.

## Update 2026-06-13 19:55 UTC

- Hidden cache generation completed and passed supervisor gates:
  - shard total: `103036`
  - merged metadata: `merged=true`, `num_samples=103036`, `num_shards=8`, `output_dtypes=["bf16"]`
  - audit: `ok=true`, `failed_records=0`, `total_records=2048`
- The original supervisor launched Stage2 at:
  `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_clonefix_20260613T133206Z`
- That first Stage2 launch failed before training because the loader misidentified the new hidden cache as the legacy directory cache layout:
  - error: `ValueError: num_samples should be a positive integer value, but got num_samples=0`
  - root cause: `ChunkCacheDataset.looks_like()` required a top-level `samples/` directory, while this hidden cache stores samples under `shards/shard_*/samples/*.pt` and points to them through the root `index.jsonl`.
- Code fix:
  - `navsim/planning/script/run_training_recogdrive.py`
  - `looks_like()` now validates indexed sample paths instead of requiring top-level `samples/`.
  - `ChunkCacheDataset` now passes `two_expert_h_dyn` and `two_expert_h_geo` into features when `agent.use_two_expert_slots=true`.
- Verification after the fix:
  - `ChunkCacheDataset.looks_like(...) == True`
  - train records: `84918`
  - val records: `18118`
  - one train sample loads with:
    - `last_hidden_state: (2800, 1536)`
    - `two_expert_h_dyn: (3, 12, 1536)`
    - `two_expert_h_geo: (12, 1536)`
    - `trajectory: (8, 3)`
  - `py_compile` passed for `navsim/planning/script/run_training_recogdrive.py`.
- Local GPU state at restart time:
  - all 8 GPUs were occupied by other Stage3 training/eval processes.
  - remote SSH alias `training-vla-zt-peer` was not usable: `Permission denied (publickey,password)`.
- A delayed formal 8GPU Stage2 restart is now waiting for all 8 local GPUs to become free:
  - output root: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_indexfix_20260613T195319Z`
  - waiter PID: `851365`
  - log: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_indexfix_20260613T195319Z/logs/gpu_waiter.outer.log`
  - polling cadence: once per `600s`
  - first snapshot: `free_gpus=0/8`
- A new low-frequency monitor is attached to the delayed Stage2 output:
  - monitor PID: `852313`
  - log: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_indexfix_20260613T195319Z/logs/lowfreq_monitor/hourly_status.log`
  - cadence: once per hour
  - first snapshot: waiter alive, Stage2 files not created yet
- Disk free on `/mnt/project` at 19:42 UTC was about `770G`.

## Current State

- Stage1 VLM SFT is complete.
- Stage2 hidden cache is complete and audited.
- Stage2 code/config has been patched for the new hidden-cache layout and two-expert hidden features.
- Formal 8GPU Stage2 training is queued behind local GPU availability; it has not re-entered training yet.

## Active Hidden Cache

- Cache root: `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16`
- Log root: `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/logs/hidden_navtrain_stage1_lora_bf16_clonefix_batch4_20260613T133054Z`
- Config:
  - `num_shards=8`
  - `batch_size=4`
  - `precision=bf16`
  - `output_dtype=bf16`
  - `train_vlm_mode=lora`
  - `stage1_train_mode=lora`
  - `max_image_patches=12`

Progress at 2026-06-13 17:36 UTC:

- Logged total written: `57,000 / 103,036`
  - This is conservative because shard logs emit every 1000 samples.
  - Slow shards are currently at about `5,000` written; faster shards are at `10,000-11,000`.
- The shard processes are still active and doing CPU/I/O work; they are not stuck in dead state.
- Approximate ETA is governed by the slow shards and is still several hours.
- Cache size: not recalculated to avoid extra I/O; projected full cache remains about `840GB`
- Disk free on `/mnt/project`: about `1.3T`
- Stage2 has not started yet; supervisor still reports `cache_shards_running=8`.
- Shard logs and supervisor log have no `Traceback`, `RuntimeError`, OOM, or killed-process markers.

## Cache Space Fix

The first batch-4 cache run produced oversized sample files:

- Incorrect per-sample file size: about `33.4MB`
- Root cause: per-sample tensors were views into batch storage, so each `.pt` retained the full batch backing storage.
- Fix: clone each sample tensor before saving:
  - `raw[idx].clone().contiguous()`
  - `h_dyn[idx].clone().contiguous()`
  - `h_geo[idx].clone().contiguous()`

After the fix:

- Correct per-sample file size: about `8.35MB`
- Projected full navtrain hidden cache: about `840GB`

Additional cleanup:

- Removed old two-expert lora smoke hidden-cache directories:
  - `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/smoke/hidden_cache_latest_stage1_8`
  - `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/smoke/hidden_cache_latest_stage1_8_bf16`
  - `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/smoke/hidden_cache_train_smoke`
- Removed the remaining old smoke/restart artifacts:
  - `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/smoke`
  - `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/stage1/full_vlm_sft_lora/restarts/batch8_oom_20260613T065809Z`
  - `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/stage1/full_vlm_sft_lora/restarts/pre_batch8_restart_20260613T065447Z`
- Removed old LoRA/cache smoke paths to avoid confusion with the current cache:
  - `/mnt/project/VLA-AD/outputs/last_vla_v2/tmp_b_lora_cache_batch2_smoke`
  - `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_cot_lora_eval_20260607T142428Z/logs/B_epoch003_lora_navtest_full_cache`
- Kept current active full cache, teacher caches, Stage1 checkpoint, and logs.

Old-cache inventory:

- Report: `/mnt/project/VLA-AD_last_vla_dev/reports/two_expert_slot/old_cache_size_inventory_report.md`
- Biggest old cache candidate after the active hidden-cache job finishes:
  `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks`
  at about `1.79 TiB`.
- Do not delete that directory while the current hidden-cache builder is reading it.

## Stage2 Supervisor

- Supervisor log root: `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/logs/stage2_after_hidden_cache_supervisor_clonefix_20260613T133206Z`
- Planned Stage2 output root: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_clonefix_20260613T133206Z`
- Low-frequency monitor:
  `/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/logs/lowfreq_monitor/hourly_status.log`
  - Runs once per hour.
  - Checks only supervisor tail, Stage2 process presence, a small list of Stage2 output files, and `df -h /mnt/project`.
  - Does not run recursive `du/find` and does not inspect cache sample files.

The supervisor will only start Stage2 after:

1. All 8 hidden-cache shards exit.
2. Shard index and metadata files exist.
3. Total hidden-cache samples equal `103036`.
4. Cache merge succeeds.
5. Hidden-cache audit passes on a 2048-record sample.

## Formal Stage2 Config

Formal Stage2 is configured to test the full two-expert method, not only a diagnostic branch:

- A0 checkpoint initialization:
  `/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt`
- `train_expert_only=false`
- raw VLM context preserved
- full DiT/action head SFT plus zero-init two-expert horizon branch
- base/action LR: `3e-5`
- two-expert branch LR: `1e-4`
- batch size: `16/GPU`
- max epochs: `200`
- precision: `bf16-mixed`
- residual diffusion: disabled
- diffusion target: GT normalized trajectory

Validation:

- Python compile passed for:
  - `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
  - `navsim/agents/recogdrive/recogdrive_agent.py`
  - `scripts/last_vla_v2/two_expert_slot/build_two_expert_hidden_cache.py`
- Shell syntax passed for:
  - `scripts/last_vla_v2/two_expert_slot/wait_hidden_cache_then_run_stage2.sh`
  - `scripts/last_vla_v2/two_expert_slot/run_stage2_two_expert_dit_sft_8gpu.sh`
- Hydra compose check for `two_expert_slot_stage2_dit_sft` confirmed:
  - `use_two_expert_slots=true`
  - `use_last_vla=false`
  - `use_last_rd=false`
  - `use_expert_features=false`
  - `last_vla_use_residual_diffusion=false`
  - `last_vla_teacher_traj_mode=none`
  - `precision=bf16-mixed`

Baseline:

- A0 official-aligned Stage2 full navtest PDMS: `0.864891`
