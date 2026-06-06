# Last-VLA v2 Decoupled HighCap Readiness Report

Date: 2026-06-06

Formal path: `ReCogDrive-LaST-v2 Decoupled HighCap NoRisk`.

Baseline: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`.

## Summary

The formal Last-VLA v2 path now preserves full raw ReCogDrive VLM hidden tokens as the DiT base context and injects latent CoT through an independent zero-init residual condition branch. Hard bottleneck and VLM summary replacement are removed from the model path.

No training launched. No full eval launched. No production cache generated.

## Changed Implementation Files

- `navsim/agents/recogdrive/last_vla_cot_planning.py`
- `navsim/agents/recogdrive/recogdrive_dit.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `scripts/eval_last_vla_cot_corruption_pdm.py`

## New Formal Configs

- `configs/last_vla_v2/decoupled_highcap_no_risk/base.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/cot_alignment.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/vlm_lora_cot_alignment.yaml`

Hydra experiments:

- `last_vla_decoupled_cot_alignment_highcap_no_risk`
- `last_vla_decoupled_progressive_highcap_no_risk`
- `last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk`

## Formal Config Values

- `use_last_vla=true`
- `use_last_rd=false`
- `use_expert_features=false`
- `last_vla_condition_mode=decoupled_cot_residual`
- `last_vla_cot_bottleneck_mode=false`
- `last_vla_raw_vlm_context_to_dit=true`
- `last_vla_cot_num_tokens=192`
- `last_vla_cot_num_steps=5`
- `num_jepa_tokens=128`
- `num_dynamic_tokens=128`
- `num_geometry_tokens=192`
- `last_vla_geometry_teacher_dim=512`
- `last_vla_geometry_grid_rows=12`
- `last_vla_geometry_grid_cols=16`
- `num_risk_tokens=0`
- `last_vla_use_risk_head=false`
- `last_vla_teacher_traj_mode=none`
- `policy_kd_mode=none`

## Summary Removal

The code no longer contains `vlm_summary`, `VLMTokenCompressor`, `drop_vlm_summary`, `include_summary`, or `use_summary_for_cot` in the formal model/config/script path. Last-VLA construction rejects hard-bottleneck or raw-VLM-disabled configs.

## Raw VLM Preservation

`LastVLAOutput.planner_context_tokens` and planner `context_tokens` are raw VLM base context in decoupled mode. CoT condition tokens are returned separately as `cot_condition_tokens`.

## Decoupled CoT Branch

`LightningDiTBlock` now has a separate `cot_cross_attn` and zero-init `cot_out_proj`. Initial output with CoT condition equals output without CoT condition. Gradients reach both `cot_out_proj` and `cot_cross_attn` on the first backward pass.

Norm diagnostics are logged for:

- `last_vla_cot_condition_norm`
- `last_vla_cot_condition_delta_norm`
- `last_vla_raw_vlm_base_context_norm`
- `last_vla_context_mean_cot_residual_norm`
- `last_vla_horizon_cot_residual_norm`

## Parallel CoT

CoT structure:

1. scene grounding from raw VLM
2. geometry and dynamic CoT parallel from scene tokens
3. fusion CoT
4. ego/coarse planning
5. action-refinement CoT

Dynamic CoT can attend projected geometry memory but does not depend solely on geometry CoT.

## A/B Definition

A: frozen ReCogDrive VLM hidden cache.

B: online VLM-LoRA CoT alignment, adapter extraction, regenerated train hidden cache, then the same progressive SFT as A. B navtest eval must use LoRA-regenerated navtest hidden cache.

## New Scripts

- `scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/build_lora_navtest_cache_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/eval_cot_corruption_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/summarize_decoupled_highcap_no_risk.py`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_eval_after_training_decoupled.sh`

All launchers are dry-run unless `RUN_TRAIN=1`, `RUN_CACHE=1`, or `RUN_EVAL=1`.

## Deprecated Path Archive

Old hard-bottleneck / summary replacement configs and launch wrappers are archived under:

- `configs/last_vla_v2/archive/hard_bottleneck_legacy/`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/`

Formal configs live only under `configs/last_vla_v2/decoupled_highcap_no_risk/`.

## Tests

Passed:

```bash
python -m py_compile \
  navsim/agents/recogdrive/last_vla_cot_planning.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  navsim/agents/recogdrive/recogdrive_agent.py \
  navsim/agents/recogdrive/dynamic_tokenizer.py \
  navsim/agents/recogdrive/geometry_tokenizer.py \
  scripts/audit_last_vla_cache_manifest.py \
  scripts/eval_last_vla_cot_corruption_pdm.py \
  scripts/last_vla_v2/decoupled_highcap_no_risk/summarize_decoupled_highcap_no_risk.py
```

```bash
bash -n scripts/last_vla_v2/decoupled_highcap_no_risk/*.sh
```

```bash
pytest -q \
  tests/test_last_vla_decoupled_*.py \
  tests/test_dit_decoupled_cot_branch_zero_init.py \
  tests/test_last_vla_*.py \
  tests/test_last_rd_*.py \
  tests/test_expert_*.py \
  tests/test_no_future_leakage.py
```

Result: `107 passed, 3 skipped`.

Also passed: `git diff --check`.

## Cache Generation Command

See `reports/last_vla_v2_decoupled_cache_generation_commands.md`.

## Archived Legacy Files

The old hard-bottleneck / summary replacement configs and wrappers are archived, not deleted. Use only the decoupled highcap no-risk directory for formal cache generation and training.

## Remaining Blockers

None for code readiness. Production cache generation still needs real paths for base chunks, VGGT, and V-JEPA.
