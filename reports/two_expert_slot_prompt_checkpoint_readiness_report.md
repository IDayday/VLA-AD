# Two-Expert Slot Prompt and Checkpoint Readiness Report

- Base commit: `8523adb`
- Branch: `feature/recogdrive-last-vla-v2`
- Baseline reference: A0 official-aligned ReCogDrive Stage2 `step_00100000`, full navtest PDMS `0.864891`

## Fixes

- Prompt consistency fixed:
  - Added `two_expert_prompt_utils.py` with `build_two_expert_prompt` and prompt version `two_expert_slot_prompt_v1`.
  - Stage1 VLM SFT and hidden cache builder now use the same prompt builder.
  - Missing `history_trajectory` / `high_command_one_hot` fails fast unless explicit minimal-prompt mode is enabled.
- Compact Stage1 checkpoint implemented:
  - `stage1.ckpt` now stores slots, adapters, probe, metadata, and schema only.
  - Full module state is saved only with `--save-full-stage1-state`.
  - LoRA adapters are saved under `adapters/vlm_lora` and referenced from metadata.
  - Top-layer/full VLM trainable weights are saved under `adapters/vlm_trainable_state.pt` and referenced from metadata.
- Hidden cache compact loading implemented:
  - Loads compact slot state directly from `two_expert_slots`.
  - Resolves relative LoRA adapter dir from checkpoint metadata.
  - Resolves relative top-layer trainable state path from checkpoint metadata.
  - Records checkpoint schema and loaded state diagnostics in hidden-cache metadata.
- Final readiness gate implemented:
  - Added `final_two_expert_readiness_gate.py`.
  - Stage1 and Stage2 launchers require `READINESS_GATE_JSON` with `status=READY` for full training unless explicitly skipped.
  - Gate requires strict teacher preflight, InternVL smoke sensitivity, and VGGT Feature(23) smoke output shape.
- ReCogDrive DiT target remains GT normalized trajectory; residual diffusion was not enabled.

## Validation

- `python -m py_compile` on prompt/checkpoint/gate scripts: passed.
- `pytest -q tests/test_two_expert_prompt_consistency.py tests/test_two_expert_compact_checkpoint.py tests/test_two_expert_*.py tests/test_no_future_leakage.py`: `49 passed`.
- `python scripts/last_vla_v2/two_expert_slot/smoke_stage1_two_expert_sft.py`: passed.
- `python scripts/last_vla_v2/two_expert_slot/smoke_stage2_two_expert_batch.py`: passed.
- `python scripts/last_vla_v2/two_expert_slot/final_two_expert_readiness_gate.py` with synthetic passing JSONs: passed.
- `bash -n` on Stage1, hidden-cache, Stage2, and real InternVL smoke launchers: passed.

## Remaining Blockers

- Real InternVL soft-slot smoke must still be run with `RUN_SMOKE=1`, `VLM_PATH`, and a real image/base-cache sample.
- Real VGGT Feature(23) smoke must still be run with `RUN_SMOKE=1`, `VGGT_MODEL_PATH`, and a real image/base-cache sample.
- Strict JEPA dynamic teacher cache availability must be confirmed by preflight.
- No full training launched.
- No full navtest eval launched.
