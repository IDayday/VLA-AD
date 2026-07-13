# Last-VLA v2 High-cap Final Execution Report

Date: 2026-06-04 UTC

## 1. Code Status

- Branch: `feature/recogdrive-last-vla-v2`
- Commit: `97415d098097fd3ce51d8ba0a239f4a25f6e08c0`
- Remote sync: `git fetch origin` and `git pull --ff-only origin feature/recogdrive-last-vla-v2` completed; local branch was up to date.
- Git status before report generation: clean.
- Git status after this task: dirty only because reports were generated/updated.
- Verified code fixes already present:
  - LoRA scope enforcement
  - actual trainable LoRA audit
  - Server B LoRA metadata consistency
  - `hidden_anchor_every_n_steps`
  - old Last-VLA v2 minimal / smoke / geometry-lite production configs archived

No Last-VLA v2 model structure or algorithm code was changed in this execution step.

## 2. Data Status

- Cleanup status: not executed. No cache or raw data was deleted.
- Base chunk root discovered: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
- A0 checkpoint discovered: `/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt`
- VGGT model discovered: `/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B`
- VJEPA model discovered: `/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256`
- Existing old cache inspected: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/full_geometry_chunks_512_20260603_035727`
- Existing old cache sample shapes:
  - `jepa_context_tokens`: `[12,1024]`
  - `jepa_target_tokens`: `[12,1024]`
  - `vggt_geometry_tokens`: `[12,512]`
- Required high-cap train cache: `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks`
- High-cap train cache status: missing
- JEPA128 coverage: not available
- Geometry192 coverage: not available
- Strict manifest status: not generated because the high-cap cache does not exist
- Cache generation dry-run: completed; command log written to `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/commands.log`

The old 12-token JEPA / 12-token geometry cache is not acceptable for strict high-cap training.

## 3. Readiness Status

- Readiness report: `/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_prelaunch_20260604_utc/readiness/final_readiness_report.md`
- Readiness status: `NOT READY`
- Readiness tests: pass
- Server A/B launcher dry-run: pass
- Blocker from readiness gate: high-cap train cache does not exist

Additional operational blockers:

- Local GPUs are occupied by an existing Last-VLA process using the old cache:
  - PID: `729208`
  - Command includes `cache_path=/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/full_geometry_chunks_512_20260603_035727`
- Remote default host `training-vla-zt-peer` is reachable, but `/mnt/project/VLA-AD` is on branch `research/bit-drive-left-tail` with local modifications, so automatic checkout/pull is blocked.

## 4. Training Launch Status

- Local Server A launched: no
- Remote Server B launched: no
- Local pid/log path: none
- Remote pid/log path: none
- Reason: readiness is `NOT READY`; high-cap cache is missing; local GPUs are occupied; remote worktree is dirty and on the wrong branch.

Training was not launched in this task.

## 5. Next Eval Status

- Eval command files prepared: no
- Frozen high-cap navtest cache ready: no
- Line B LoRA navtest hidden cache ready: no
- Reason: training was not launched and high-cap train/navtest cache is not ready.

Full navtest eval was not launched in this task.

## 6. Baseline

- A0-official-aligned best checkpoint: `step_00100000`
- A0-official-aligned full navtest PDMS: `0.864891`

Future Last-VLA v2 results must be compared against `0.864891`.

## 7. Next Commands

Use `reports/last_vla_v2_highcap_manual_launch_commands.md` for exact data-prep, readiness, launch, and eval-prep commands with discovered local paths.
