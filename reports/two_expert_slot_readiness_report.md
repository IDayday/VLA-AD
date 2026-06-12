# Two-Expert Slot Readiness Report

Baseline reference: A0 official-aligned ReCogDrive Stage2 `step_00100000`, full navtest PDMS = `0.864891`.

## Changed Files

Core implementation:

- `navsim/agents/recogdrive/two_expert_slots.py`
- `navsim/agents/recogdrive/two_expert_adapters.py`
- `navsim/agents/recogdrive/two_expert_vlm_sft.py`
- `navsim/agents/recogdrive/recogdrive_backbone.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `scripts/eval_recogdrive_expert_pdm.py`

Cache, audit, config, launchers, docs, tests:

- `scripts/last_vla_v2/two_expert_slot/`
- `scripts/audit_two_expert_teacher_cache.py`
- `scripts/audit_two_expert_hidden_cache.py`
- `configs/last_vla_v2/two_expert_slot/`
- `navsim/planning/script/config/experiment/two_expert_slot_stage1_vlm_sft.yaml`
- `navsim/planning/script/config/experiment/two_expert_slot_stage2_dit_sft.yaml`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `docs/two_expert_slot/Two_Expert_LaST_CoWorld_ReCogDrive_Design.md`
- `tests/test_two_expert_*.py`

Unrelated pre-existing modified files are still present in the worktree and were not used by this route: `navsim/agents/recogdrive/utils/lr_scheduler.py`, `navsim/planning/script/run_training_recogdrive_rl.py`, `navsim/planning/training/agent_lightning_module.py`, plus untracked Stage3/cache launchers.

## Architecture Summary

The new route is enabled by `use_two_expert_slots=true`, `two_expert_slot_mode=vlm_soft_slots`, and `two_expert_condition_mode=horizon_hmef_lite`.

VLM side:

- `H_dyn`: 3 groups x 12 soft slots, inserted into VLM `inputs_embeds`.
- `H_geo`: 12 soft slots, inserted into the same VLM sequence.
- No tokenizer special tokens and no tokenizer embedding resize.
- If the active InternVL wrapper cannot accept `inputs_embeds`, `forward_with_two_expert_slots` raises `NotImplementedError`.

Stage1:

- `JEPADynamicAdapter`: random-masked image hidden + `H_dyn` predicts `[3,12,1024]`.
- `VGGTFeature23Adapter`: random-masked image hidden + `H_geo` predicts `[12,D_vggt23]`.
- `TwoExpertTrajectoryProbe`: weak planning grounding only, not the final planner.

Stage2:

- Reads cached raw VLM hidden, `two_expert_h_dyn`, `two_expert_h_geo`.
- Projects expert slots to planner dim and cross-attends 8 horizon queries to produce `F_dyn` and `F_geo`.
- Adds zero-initialized per-step expert deltas to the base ReCogDrive fused input.
- Raw VLM hidden remains DiT encoder context.
- Diffusion target is GT normalized trajectory. No residual target or coarse-plus-residual output.

## LaST-VLA Adapter Mapping

- VLM internal latent slots: implemented as soft embeddings passed through the VLM transformer.
- External teachers are train-time only.
- Masked adapters combine image/VLM hidden tokens with slot hidden features.
- No A4-V2 direct expert path, summary replacement, hard bottleneck, or action-side `LastVLACoTTransformer` in the two-expert mode.

## CoWorld-VLA Planner Mapping

- Expert tokens condition the planner, not the VLM output replacement.
- Horizon-aligned `F_dyn/F_geo` are `[B,8,384]`.
- The current formal path is per-step additive zero-init conditioning. Optional multi-head expert noise heads are not implemented in this pass.

## Teacher Status

JEPA dynamic teacher:

- Strict schema supported: `jepa_dynamic_teacher_tokens [3,12,1024]`.
- Dev fallback supported only when strict mode is off: legacy `jepa_target_tokens [N,1024]` deterministically downsampled to `[3,12,1024]`.
- Actual multi-horizon JEPA model extraction is not implemented in this pass; production strict cache must provide or generate the strict key.

VGGT Feature(23):

- Strict schema supported: `vggt_feature23_tokens [12,D]`.
- Dev fallback supported only when strict mode is off: old `vggt_geometry_tokens` packed to 12 tokens and marked non-strict.
- Aggregator layer discovery helper exists, but end-to-end VGGT model loading/hook extraction is still a production integration blocker.

## VLM Soft-Slot Injection Status

Implemented in `ReCogDriveBackbone.forward_with_two_expert_slots` for InternVL-style wrappers that expose `inputs_embeds`. Tested with a mock VLM to confirm slots pass through the model forward and are extracted from returned hidden states. Actual production InternVL compatibility must be verified against the deployed checkpoint wrapper.

Train modes:

- `frozen`: supported; base VLM params frozen.
- `lora`: supported as a scope placeholder; requires existing LoRA modules to already be applied.
- `top_layers`: supported by parameter-name filtering; should be audited on the exact model before long training.

## Cache Schema

Teacher cache:

- `jepa_dynamic_teacher_tokens [3,12,1024]`
- `jepa_dynamic_teacher_metadata`
- `vggt_feature23_tokens [12,D]`
- `vggt_feature23_metadata`

Hidden cache:

- `last_hidden_state [N,1536]`
- `two_expert_h_dyn [3,12,1536]`
- `two_expert_h_geo [12,1536]`
- `two_expert_metadata`
- Base fields: history, command, status, trajectory, sample token.
- Train cache may include teacher targets. Eval cache must not require them, and eval ignores train-only targets.

## Validation

Commands run:

- `python -m py_compile` on two-expert modules, modified planner/backbone/agent, cache builders, audits, and eval script.
- `bash -n` on all new two-expert shell launchers.
- `pytest -q tests/test_two_expert_*.py`
- `pytest -q tests/test_two_expert_*.py tests/test_last_vla_*.py tests/test_last_rd_*.py tests/test_expert_*.py tests/test_no_future_leakage.py`

Results:

- Two-expert tests: `24 passed`.
- Full requested set: `140 passed, 4 skipped, 46 warnings`.

## Remaining Blockers

- Production multi-horizon JEPA extraction must be connected if strict teacher cache is not already available.
- Production VGGT Feature(23) forward-hook extraction must be connected and validated against the actual VGGT model.
- Stage1 trainer entrypoint still needs production Lightning/Hydra binding around `TwoExpertVLMSFTModule`; the module and launcher/config skeleton are present.
- Actual InternVL checkpoint must be smoke-tested for `inputs_embeds` support.
- No training was launched.
- No production cache was generated.
- No full navtest eval was launched.
