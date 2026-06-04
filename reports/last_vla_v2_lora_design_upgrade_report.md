# Last-VLA v2 VLM-LoRA Design Upgrade Report

Date: 2026-06-04

Baseline: A0-official-aligned full navtest PDMS = 0.864891.

No training launched. No full eval launched.

## Scope

This change upgrades the Last-VLA v2 Line B VLM-LoRA latent adaptation path only. It does not change the latent CoT or residual diffusion core structure.

## Changed Files

- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `navsim/planning/script/config/experiment/last_vla_vlm_lora_cot_alignment.yaml`
- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/planning/training/agent_lightning_module.py`
- `configs/last_vla_v2/last_vla_vlm_lora_cot_alignment.yaml`
- `scripts/build_recogdrive_hidden_cache_with_lora.py`
- `scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py`
- `scripts/last_vla_v2/run_vlm_lora_cot_alignment_8gpu.sh`
- `scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh`
- `scripts/last_vla_v2/round2/build_lora_navtest_cache.sh`
- `scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh`
- `scripts/last_vla_v2/highcap_no_risk/build_lora_navtest_cache_highcap.sh`
- `tests/test_last_vla_round2_launchers_dryrun.py`
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`
- `docs/Last_VLA_v2_Round2_Training_Runbook.md`

## New Files

- `navsim/agents/recogdrive/vlm_lora_utils.py`
- `configs/last_vla_v2/lora/last_vla_vlm_lora_attention_only_r16.yaml`
- `configs/last_vla_v2/lora/last_vla_vlm_lora_attention_mlp_r32.yaml`
- `configs/last_vla_v2/lora/last_vla_vlm_lora_all_linear_r64.yaml`
- `configs/last_vla_v2/lora/last_vla_highcap_vlm_lora_attention_mlp_r32.yaml`
- `configs/last_vla_v2/lora/last_vla_highcap_vlm_lora_all_linear_r64.yaml`
- `navsim/planning/script/config/experiment/last_vla_vlm_lora_attention_only_r16.yaml`
- `navsim/planning/script/config/experiment/last_vla_vlm_lora_attention_mlp_r32.yaml`
- `navsim/planning/script/config/experiment/last_vla_vlm_lora_all_linear_r64.yaml`
- `scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh`
- `scripts/last_vla_v2/lora/summarize_lora_alignment_sweep.py`
- `tests/test_last_vla_lora_target_resolution.py`
- `tests/test_last_vla_lora_scope_filter.py`
- `tests/test_last_vla_lora_config_serialization.py`
- `tests/test_last_vla_lora_optimizer_groups.py`
- `tests/test_last_vla_lora_hidden_anchor_loss.py`
- `tests/test_last_vla_lora_adapter_extract.py`
- `tests/test_last_vla_lora_sweep_dryrun.py`
- `docs/Last_VLA_v2_VLM_LoRA_Design.md`

## Old LoRA Design

- Target modules: `q_proj,k_proj,v_proj,o_proj`
- Rank/alpha: `r=16`, `alpha=32`
- No dropout
- No rsLoRA or DoRA support
- No LoRA-specific optimizer group
- Adapter extraction saved state only, making hidden-cache regeneration depend on manual r/alpha/target arguments.

## New LoRA Design

Recommended formal setting:

- `last_vla_vlm_lora_preset=attention_mlp`
- `last_vla_vlm_lora_scope=llm`
- `last_vla_vlm_lora_r=32`
- `last_vla_vlm_lora_alpha=64`
- `last_vla_vlm_lora_dropout=0.05`
- `last_vla_vlm_lora_use_rslora=true`
- `last_vla_vlm_lora_use_dora=false`
- `lr_vlm_lora=1e-5`
- `lr_last_vla_cot=1e-4`
- `last_vla_hidden_anchor_weight=0.01`
- `last_vla_hidden_anchor_mode=summary_cosine`

Supported presets:

- `attention_only`: q/k/v/o conservative baseline.
- `attention_mlp`: attention plus MLP projections, recommended default.
- `all_linear`: all eligible linear modules, excluding action head and lm head.
- `vision_last_n`: last N vision blocks when module names expose block indices.
- `custom`: exact comma-separated target module suffixes.

Supported scopes:

- `llm`
- `vision`
- `llm_vision`
- `projector`
- `all`

## Target Module Audit

LoRA target resolution now audits matched module names and categories:

- `llm_attention`
- `llm_mlp`
- `vision_attention`
- `vision_mlp`
- `projector`
- `other`

Startup fails if no modules match. `scope=llm` fails if vision modules are matched unless mixed scope is explicitly allowed. The run writes `lora_target_report.json`, and the same report is included in `precision_report.json` when available.

## Optimizer Groups

During `last_vla_train_vlm_lora=true`, optimizer construction is restricted to exactly:

- `last_vla_cot`: `action_head.last_vla_cot.*`, lr `lr_last_vla_cot`, wd `weight_decay_last_vla_cot`
- `vlm_lora`: parameters containing `lora_`, lr `lr_vlm_lora`, wd `weight_decay_vlm_lora`

The code raises if action base or non-LoRA backbone parameters are trainable. It writes `trainable_parameter_counts.json` and includes optimizer groups in LoRA runtime reporting.

## Hidden Anchor

Online VLM-LoRA CoT alignment can compute a frozen-base hidden reference with PEFT adapter disabled. The anchor is active only for training-time VLM-LoRA `cot_alignment` with uncached hidden states.

Modes:

- `summary_cosine`
- `token_mean_cosine`
- `mse_mean`
- `none`

Logged outputs:

- `hidden_anchor_loss`
- `hidden_anchor_loss_weighted`
- `hidden_drift_cosine`
- `hidden_drift_l2`

## Adapter Save/Load Format

Extraction now writes:

- `adapters/last_vla_cot_adapter.pt`
- `adapters/vlm_lora/adapter_model.bin`
- `adapters/vlm_lora/adapter_config.json`
- `adapters/vlm_lora/lora_metadata.json`
- `adapters/vlm_lora/lora_target_report.json` when available
- `adapters/vlm_lora_adapter_state.pt` as a legacy compatibility copy

Metadata includes base VLM path/type, preset, scope, target modules, resolved modules, r, alpha, dropout, bias, rsLoRA/DoRA flags, matched modules, PEFT version, source checkpoint, and an adapter config hash.

## Hidden Regeneration

`scripts/build_recogdrive_hidden_cache_with_lora.py` now prefers `--vlm-lora-adapter-dir`. It reads `adapter_config.json` and `lora_metadata.json`, reconstructs the exact LoRA config, checks explicit CLI override mismatches, and records the adapter config hash in regenerated cache metadata.

Legacy state-only arguments remain available as fallback, but adapter-dir regeneration is the supported path for Line B.

## Dry-Run Commands

LoRA sweep dry-run used:

```bash
LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS=1 \
RUN_TRAIN=0 \
NAVSIM_LOG_PATH=/tmp/navsim_logs \
SENSOR_BLOBS_PATH=/tmp/sensor_blobs \
EXPERT_TEACHER_CACHE_ROOT=/tmp/expert_cache \
VLM_PATH=/tmp/vlm \
OUT_ROOT=/tmp/last_vla_lora_sweep \
MASTER_PORT_BASE=29600 \
scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh
```

Future real sweep command:

```bash
RUN_TRAIN=1 \
NAVSIM_LOG_PATH=/path/to/navsim_logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
EXPERT_TEACHER_CACHE_ROOT=/path/to/full_geometry_teacher_cache \
VLM_PATH=/path/to/vlm \
OUT_ROOT=/path/to/last_vla_lora_sweep \
MASTER_PORT_BASE=29600 \
scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh
```

Future Line B hidden regeneration should use:

```bash
python scripts/build_recogdrive_hidden_cache_with_lora.py \
  --base-cache /path/to/base_hidden_cache \
  --output-cache /path/to/lora_regenerated_hidden_cache \
  --vlm-path /path/to/vlm \
  --vlm-type internvl \
  --vlm-lora-adapter-dir /path/to/run/adapters/vlm_lora
```

## Validation

Passed:

```bash
python -m py_compile \
  navsim/agents/recogdrive/recogdrive_agent.py \
  navsim/agents/recogdrive/vlm_lora_utils.py \
  scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py \
  scripts/build_recogdrive_hidden_cache_with_lora.py \
  scripts/last_vla_v2/lora/summarize_lora_alignment_sweep.py
```

Passed:

```bash
bash -n scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh
```

Passed:

```bash
python -m pytest -q \
  tests/test_last_vla_lora_*.py \
  tests/test_last_vla_*.py \
  tests/test_last_rd_*.py \
  tests/test_expert_*.py \
  tests/test_no_future_leakage.py
```

Result: 96 passed, 3 skipped, 39 warnings.

Passed:

```bash
git diff --check
```

## Remaining Risks

- No performance claims can be made before real Line B training and navtest evaluation.
- rsLoRA and DoRA are gated by installed PEFT support; requested unsupported features fail early.
- `vision_last_n` depends on recognizable vision block naming and fails with a diagnostic if block indices cannot be inferred.
- Line B evaluation must use LoRA-regenerated navtest hidden cache; evaluating Line B on the original hidden cache is invalid.
