# Last-VLA v2 High-cap No-risk Readiness Report

Baseline: A0-official-aligned PDMS `0.864891`.

## Changed Files

- `navsim/agents/recogdrive/last_vla_cot_planning.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/recogdrive_features.py`
- `navsim/agents/recogdrive/expert_cache.py`
- `navsim/agents/recogdrive/geometry_tokenizer.py`
- `navsim/agents/recogdrive/expert_extractors/vjepa2_extractor.py`
- `navsim/agents/recogdrive/expert_extractors/vggt_extractor.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `scripts/build_recogdrive_chunk_cache.py`
- `scripts/build_last_vla_full_geometry_cache.py`
- `scripts/merge_last_vla_geometry_cache_into_chunks.py`
- `scripts/audit_last_vla_cache_manifest.py`
- `scripts/build_recogdrive_hidden_cache_with_lora.py`
- `scripts/eval_recogdrive_expert_pdm.py`
- `scripts/last_vla_v2/generate_full_geometry_overlay_cache.sh`
- `scripts/last_vla_v2/merge_full_geometry_overlay_cache.sh`
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`
- `docs/Last_VLA_v2_Round2_Training_Runbook.md`

## New Files

- `navsim/agents/recogdrive/dynamic_tokenizer.py`
- `configs/last_vla_v2/last_vla_highcap_no_risk_base.yaml`
- `configs/last_vla_v2/last_vla_cot_alignment_highcap_no_risk.yaml`
- `configs/last_vla_v2/last_vla_progressive_bottleneck_highcap_no_risk.yaml`
- `configs/last_vla_v2/last_vla_progressive_bottleneck_highcap_no_risk_eval.yaml`
- `configs/last_vla_v2/last_vla_vlm_lora_cot_alignment_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/last_vla_cot_alignment_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/last_vla_progressive_bottleneck_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/last_vla_vlm_lora_cot_alignment_highcap_no_risk.yaml`
- `scripts/last_vla_v2/highcap_no_risk/generate_full_geometry_overlay_cache_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/merge_full_geometry_overlay_cache_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/run_strict_preflight_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh`
- `scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh`
- `scripts/last_vla_v2/highcap_no_risk/build_lora_navtest_cache_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/eval_checkpoint_sweep_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/eval_cot_corruption_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/summarize_highcap_no_risk.py`
- `tests/test_last_vla_highcap_no_risk_shapes.py`
- `tests/test_last_vla_highcap_no_risk_cache_schema.py`
- `tests/test_last_vla_geometry_token_packer_highcap.py`
- `tests/test_last_vla_dynamic_tokenizer_highcap.py`
- `tests/test_last_vla_highcap_config_compose.py`
- `tests/test_last_vla_highcap_launchers_dryrun.py`
- `docs/Last_VLA_v2_HighCap_NoRisk_Runbook.md`
- `reports/last_vla_v2_highcap_no_risk_readiness_report.md`

## Exact High-cap Token Config

- `last_vla_vlm_summary_tokens=64`
- `last_vla_cot_num_tokens=192`
- `num_jepa_tokens=128`
- `num_dynamic_tokens=128`
- `jepa_dim=1024`
- `num_vggt_tokens=128` for optional legacy VGGT context only
- `num_geometry_tokens=192`
- `last_vla_geometry_teacher_dim=512`
- `last_vla_geometry_grid_rows=12`
- `last_vla_geometry_grid_cols=16`
- `last_vla_use_risk_head=false`
- `num_risk_tokens=0`
- `last_vla_risk_loss_weight=0.0`
- `last_vla_require_full_geometry=true`
- `last_vla_allow_patch_geometry_fallback=false`
- `last_vla_raw_vlm_context_to_dit=false`
- `last_vla_cot_bottleneck_mode=true`

Expected eval DiT context length: `192 CoT + 64 VLM summary = 256` tokens. Full raw VLM tokens do not enter DiT in bottleneck mode.

## Risk Status

Risk head construction is skipped when `last_vla_use_risk_head=false` or `num_risk_tokens=0`. Risk tokens are not appended to `cot_final`; `last_vla_risk_loss` stays zero; `LastVLAOutput.risk_logits` is `None`; strict high-cap dataset and audit paths do not require risk labels.

## Cache Regeneration Requirements

High-cap no-risk requires regenerated caches:

- JEPA context and target tokens must be `[128, 1024]`.
- Strict high-cap JEPA cannot repeat or interpolate the old 12 already-pooled tokens.
- VGGT full geometry tokens must be `[192, 512]`.
- Geometry mode must be `full_geometry` with `vggt_geometry_mode_code=2`.
- Patch fallback count must be `0`.
- Line B VLM-LoRA requires regenerated train hidden cache and regenerated navtest hidden cache before frozen-cache SFT/eval.

## JEPA High-cap Status

Implemented `DynamicTokenPacker` with temporal-spatial pooling to `8 x 16 = 128` tokens and dense adaptive pooling fallback for sufficiently dense sequences. Strict high-cap cache generation raises:

`High-cap JEPA requires dense or sufficiently many JEPA tokens; old 12-token cache is insufficient.`

Real high-cap JEPA production cache generation was not run in this task.

## VGGT Full Geometry High-cap Status

`GeometryTokenPacker` supports grid `12 x 16`, `192` output tokens, and `512` teacher dimension. Full geometry overlay generation and merge scripts pass high-cap geometry arguments and preserve geometry metadata. Strict audit verifies shape `[192, 512]` and mode code `2`.

Real full geometry overlay production generation was not run in this task.

## Dry-run Commands

- `scripts/last_vla_v2/highcap_no_risk/generate_full_geometry_overlay_cache_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/merge_full_geometry_overlay_cache_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/run_strict_preflight_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh`
- `scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh`
- `scripts/last_vla_v2/highcap_no_risk/build_lora_navtest_cache_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/eval_checkpoint_sweep_highcap.sh`
- `scripts/last_vla_v2/highcap_no_risk/eval_cot_corruption_highcap.sh`

All expensive scripts are dry-run by default and require explicit `RUN_CACHE=1`, `RUN_TRAIN=1`, `RUN_EVAL=1`, or `RUN_PREFLIGHT=1`.

## Tests Run

Passed:

```bash
python -m py_compile \
  navsim/agents/recogdrive/last_vla_cot_planning.py \
  navsim/agents/recogdrive/geometry_tokenizer.py \
  navsim/agents/recogdrive/dynamic_tokenizer.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  navsim/agents/recogdrive/recogdrive_agent.py \
  navsim/planning/script/run_training_recogdrive.py \
  scripts/build_last_vla_full_geometry_cache.py \
  scripts/audit_last_vla_cache_manifest.py \
  scripts/build_recogdrive_hidden_cache_with_lora.py \
  scripts/eval_recogdrive_expert_pdm.py \
  scripts/eval_last_vla_cot_corruption_pdm.py
```

Additional changed-file compile check passed:

```bash
python -m py_compile \
  navsim/agents/recogdrive/expert_cache.py \
  navsim/agents/recogdrive/recogdrive_features.py \
  navsim/agents/recogdrive/expert_extractors/vjepa2_extractor.py \
  navsim/agents/recogdrive/expert_extractors/vggt_extractor.py \
  scripts/build_recogdrive_chunk_cache.py \
  scripts/merge_last_vla_geometry_cache_into_chunks.py \
  scripts/last_vla_v2/highcap_no_risk/summarize_highcap_no_risk.py
```

Passed:

```bash
bash -n \
  scripts/last_vla_v2/highcap_no_risk/generate_full_geometry_overlay_cache_highcap.sh \
  scripts/last_vla_v2/highcap_no_risk/merge_full_geometry_overlay_cache_highcap.sh \
  scripts/last_vla_v2/highcap_no_risk/run_strict_preflight_highcap.sh \
  scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh \
  scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh \
  scripts/last_vla_v2/highcap_no_risk/build_lora_navtest_cache_highcap.sh \
  scripts/last_vla_v2/highcap_no_risk/eval_checkpoint_sweep_highcap.sh \
  scripts/last_vla_v2/highcap_no_risk/eval_cot_corruption_highcap.sh
```

Passed:

```bash
pytest -q \
  tests/test_last_vla_highcap_*.py \
  tests/test_last_vla_*.py \
  tests/test_last_rd_*.py \
  tests/test_expert_*.py \
  tests/test_no_future_leakage.py
```

Result: `71 passed, 4 skipped, 37 warnings`.

Passed:

```bash
python -m pytest -q tests/test_last_vla_highcap_no_risk_shapes.py
```

Result: `1 passed`.

Passed:

```bash
git diff --check
```

## Remaining Blockers

- Production high-cap JEPA cache has not been generated or audited.
- Production full VGGT geometry overlay has not been generated or audited.
- Line B LoRA-regenerated train/navtest hidden caches have not been generated.
- No training has been run, so no performance can be claimed.
- No full PDM eval has been run.

Training launched: no.

Full eval launched: no.
