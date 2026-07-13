# Last-VLA v2 High-cap LoRA Final Train Readiness Report

Date: 2026-06-04

Commit at start of task: `f395a091179e7e78ad11ccdd20fb4d6bc8f161e9`

Baseline: A0-official-aligned `step_00100000`, full navtest PDMS = `0.864891`.

## Status

- Readiness: `NOT READY`
- Training launched: `no`
- Full eval launched: `no`
- Local Server A launch: `not launched`
- Remote Server B launch: `not launched`

Blockers:

- `FULL_HIGHCAP_TRAIN_CHUNK_ROOT` was not provided in this Codex run.
- `A0_INIT_CHECKPOINT` was not provided in this Codex run.
- No strict production cache manifest was available to prove JEPA `[128,1024]`, geometry `[192,512]`, full geometry mode, and no patch fallback.

## Formal Config

- VLM summary tokens: `64`
- CoT tokens: `192`
- JEPA context/target tokens: `128`
- JEPA dim: `1024`
- geometry tokens: `192`
- geometry dim: `512`
- geometry grid: `12 x 16`
- risk tokens: `0`
- risk head: disabled
- raw VLM context to DiT: `false`
- CoT bottleneck: `true`
- residual diffusion: `true`
- policy KD: `0`

Canonical configs now point to high-cap no-risk:

- `configs/last_vla_v2/last_vla_cot_alignment.yaml`
- `configs/last_vla_v2/last_vla_progressive_bottleneck.yaml`
- `configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml`
- `configs/last_vla_v2/last_vla_vlm_lora_cot_alignment.yaml`

## Changed Files

Core code:

- `navsim/agents/recogdrive/vlm_lora_utils.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/planning/training/agent_lightning_module.py`
- `scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py`
- `scripts/merge_last_vla_geometry_cache_into_chunks.py`

Configs and docs:

- `configs/last_vla_v2/*.yaml`
- `configs/last_vla_v2/lora/*.yaml`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `navsim/planning/script/config/experiment/last_vla_*.yaml`
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`
- `docs/Last_VLA_v2_Round2_Training_Runbook.md`
- `docs/Last_VLA_v2_HighCap_NoRisk_Runbook.md`
- `docs/Last_VLA_v2_VLM_LoRA_Design.md`

New/updated scripts:

- `scripts/last_vla_v2/highcap_no_risk/cleanup_old_last_vla_caches.sh`
- `scripts/last_vla_v2/highcap_no_risk/prepare_highcap_no_risk_data.sh`
- `scripts/last_vla_v2/highcap_no_risk/run_final_readiness_gate.sh`
- `scripts/last_vla_v2/highcap_no_risk/launch_local_remote_full_training.sh`
- `scripts/last_vla_v2/highcap_no_risk/prepare_eval_after_training.sh`
- `scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh`
- `scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh`

Tests:

- `tests/test_last_vla_lora_scope_filter.py`
- `tests/test_last_vla_lora_target_resolution.py`
- `tests/test_last_vla_lora_hidden_anchor_loss.py`
- `tests/test_last_vla_lora_adapter_extract.py`
- `tests/test_last_vla_lora_sweep_dryrun.py`
- `tests/test_last_vla_highcap_launchers_dryrun.py`
- `tests/test_last_vla_config_compose.py`
- `tests/test_last_vla_round2_launchers_dryrun.py`
- `tests/test_last_vla_vlm_lora_trainable_scope.py`

## Archived Old Configs And Scripts

Archived configs:

- `configs/last_vla_v2/archive/last_vla_cot_alignment_geometry_lite.yaml`
- `configs/last_vla_v2/archive/last_vla_teacher_traj_sft.yaml`
- `configs/last_vla_v2/archive/last_vla_teacher_traj_sft_eval.yaml`
- `navsim/planning/script/config/experiment/archive/last_vla_teacher_traj_sft.yaml`

Archived old launchers:

- `scripts/last_vla_v2/archive/run_cot_alignment_8gpu.sh`
- `scripts/last_vla_v2/archive/run_progressive_bottleneck_8gpu.sh`
- `scripts/last_vla_v2/archive/run_vlm_lora_cot_alignment_8gpu.sh`
- `scripts/last_vla_v2/archive/run_teacher_traj_sft_8gpu.sh`
- `scripts/last_vla_v2/archive/run_best_of_k_oracle.sh`
- `scripts/last_vla_v2/archive/round2/*`

A0/A4/LastRD baseline code and generic eval utilities were not deleted.

## Fix Summaries

LoRA scope:

- Scoped presets now return full module names instead of suffixes.
- `all_linear` no longer returns global `all-linear` unless scope is `all` and `allow_all_linear_global=true`.
- Added actual trainable LoRA audit over `model.named_parameters()` after PEFT injection.
- Scope violations in actual trainable LoRA params fail fast unless mixed scope is explicitly allowed.

Server B metadata consistency:

- High-cap and archived round2 Server B launchers parse a single LoRA env block.
- Training and extraction commands use the same preset/scope/r/alpha/dropout/bias/rsLoRA/DoRA/target modules.
- Extraction reads `lora_training_config.json` first and rejects CLI mismatch by default.

Hidden anchor:

- Implemented `last_vla_hidden_anchor_every_n_steps`.
- Default is `4`.
- Skipped steps log computed status and do not add hidden-anchor loss.

## Cache Status

- Cache cleanup: dry-run only; no cache deleted.
- Cache generation: dry-run only; no production cache generated.
- Strict manifest: not available because no `FULL_HIGHCAP_TRAIN_CHUNK_ROOT` was provided.
- Readiness gate dry-run report: `NOT READY`, with tests and launcher dry-runs passing.

## Commands

Local command:

```bash
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=/path/to/highcap_no_risk/train_full_highcap_chunks \
A0_INIT_CHECKPOINT=/path/to/A0-official-aligned/step_00100000.ckpt \
OUT_ROOT=/path/to/last_vla_v2/highcap_no_risk/train_local \
MASTER_PORT=29531 \
scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh
```

Remote command:

See `reports/last_vla_v2_highcap_manual_launch_commands.md`.

## Validation

Passed:

```bash
python -m py_compile \
  navsim/agents/recogdrive/vlm_lora_utils.py \
  navsim/agents/recogdrive/recogdrive_agent.py \
  navsim/agents/recogdrive/dynamic_tokenizer.py \
  navsim/agents/recogdrive/geometry_tokenizer.py \
  scripts/build_recogdrive_hidden_cache_with_lora.py \
  scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py \
  scripts/last_vla_v2/lora/summarize_lora_alignment_sweep.py \
  scripts/audit_last_vla_cache_manifest.py
```

Passed:

```bash
bash -n \
  scripts/last_vla_v2/highcap_no_risk/cleanup_old_last_vla_caches.sh \
  scripts/last_vla_v2/highcap_no_risk/prepare_highcap_no_risk_data.sh \
  scripts/last_vla_v2/highcap_no_risk/run_final_readiness_gate.sh \
  scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh \
  scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh \
  scripts/last_vla_v2/highcap_no_risk/launch_local_remote_full_training.sh \
  scripts/last_vla_v2/highcap_no_risk/prepare_eval_after_training.sh \
  scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh
```

Passed:

```bash
pytest -q \
  tests/test_last_vla_lora_*.py \
  tests/test_last_vla_highcap_*.py \
  tests/test_last_vla_*.py \
  tests/test_last_rd_*.py \
  tests/test_expert_*.py \
  tests/test_no_future_leakage.py
```

Result: `119 passed, 4 skipped, 39 warnings`.

Passed: `git diff --check`.

## Remaining Blockers

- Provide or generate strict high-cap train cache with JEPA `[128,1024]` and full geometry `[192,512]`.
- Provide `A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt`.
- Run `run_final_readiness_gate.sh` with real paths and require `READY`.
- Provide remote host/path/VLM/log/blob env vars before remote Server B launch.

No training was launched. No full eval was launched.
