# OneVL Stage2 ReCogDrive Contract Review

## Executive Summary

- ReCogDrive DiT is not architecturally bound to InternVL. The plain Stage2 path consumes tensors: `last_hidden_state`, `his_traj`, `status_feature`, and raw `action`; it does not call InternVL-specific APIs inside the DiT.
- Pretrained ReCogDrive/InternVL weights are distribution-bound. The learned `feature_encoder`, `fusion_projector`, DiT cross-attention, and `context_mean` conditioning were trained on InternVL hidden states. Random-init training with Qwen hidden is valid only if the cache contract is clean.
- Highest-probability contract risks for the current OneVL/Qwen migration are:
  1. P0 current image/token alignment: the bridge historically used `row["images"][0]`, while Stage1 submission conversion uses `images[-1]`; the launcher defaults now use `single_or_last`, but older caches should be treated as suspect until audited.
  2. P1 prompt/planner tensor source split: older cache generation used hidden prompt from JSON row text while history/status/trajectory came from SceneLoader; the current default is JSON single-source planner tensors with SceneLoader used only for repair/audit.
  3. P3 hidden distribution shift: Qwen hidden is not InternVL hidden; a direct Linear adapter is the minimum ReCogDrive copy, but Qwen-specific pre-LN/MLP adapters are reasonable controlled ablations.
  4. P4 trajectory/support scale: DiT expects raw NAVSIM ego-local targets and normalizes internally with ReCogDrive fixed `norm_odo`.
- This review intentionally does not implement residual DiT and does not use AR Answer text trajectory as DiT input.

## ReCogDrive Contract Reconstruction

Plain ReCogDrive Stage2 dataflow:

`cache last_hidden_state -> train collate/eval make_batch -> vl_features -> feature_encoder -> context_tokens/context_mean -> DiT cross-attention + fusion_projector -> pred_traj`

The DiT path is tensor-contract based:

- `feature_encoder`: Linear projection from VLM hidden dim to planner dim.
- `context_tokens`: projected VLM tokens used as DiT cross-attention encoder states.
- `context_mean`: `vl_embeds.mean(1)` in the plain path, concatenated with history/action features before `fusion_projector`.
- `his_traj_encoder`: consumes flattened 4x3 history.
- `ego_status_encoder`: consumes 8-d status feature.
- `forward`: calls `norm_odo(action_input.action)` before diffusion loss.
- `get_action`: calls `denorm_odo(current_actions)` before returning `pred_traj`.

`context_mean` is not a separately trained module. It is a deterministic mean pooling operation over projected hidden tokens. Its training signal is the diffusion loss and any explicitly enabled auxiliary losses.

Minimum Qwen-equivalent implementation:

`Qwen hidden -> feature_encoder/adapter -> context_tokens -> mean(dim=1) -> fusion_projector/DiT`

The default implementation remains exactly this ReCogDrive-style linear adapter.

## Current OneVL/Qwen Trace

- Stage1 AR Answer SFT uses OneVL/Qwen prompts and `<answer>` trajectory text.
- Stage2 bridge extracts Qwen final-layer hidden with `processor.apply_chat_template(..., add_generation_prompt=True)`.
- The bridge does not include assistant answer text in hidden extraction unless `--assistant-prefix` is explicitly set; default is empty.
- Planner tensors are now built from prepared JSON text by default:
  - `history_trajectory`: 4x3 ego-local history.
  - `high_command_one_hot`: 3-way command.
  - `status_feature`: 8-d `[command4, velocity2, acceleration2]` feature.
  - `trajectory`: raw future ego-local target parsed from JSON answer/target text.
  - optional `support_trajectories`: raw ego-local support targets for training-only sampling.
- `--planner-source scene` remains available for reproducing older scene-derived cache runs.
- `--planner-source json_strict_scene_check` builds tensors from JSON and checks them against SceneLoader within prompt-rounding tolerance.
- Train script pads variable hidden with `pad_sequence`, then passes `vl_features` to `ReCogDriveDiffusionPlanner.forward`.
- Eval script loads one sample at a time and calls `get_action`; support target sampling is not used in eval.

## Core Findings

### InternVL Binding

- Architecture: not bound to InternVL. The DiT sees tensors and dimensions only.
- Weight distribution: bound if loading pretrained ReCogDrive/InternVL policy weights.
- Random-init Qwen training contract: hidden dim, token/current-frame alignment, prompt/status/history/target alignment, and raw trajectory scale must be correct.

### Context Mean

- Original plain path computes `context_mean = _encode_vlm(hidden).mean(1)`.
- It is trained only through diffusion loss unless optional auxiliary losses are enabled.
- The current default Qwen path still uses this same contract.
- Added optional QwenContextAdapter variants:
  - `linear`: default, previous behavior.
  - `pre_ln_linear`.
  - `pre_ln_linear_post_ln`.
  - `mlp_adapter`.

### Command Semantics

- Bridge generates `high_command_one_hot`.
- Plain DiT does not directly feed `high_command_one_hot` as a separate branch.
- Command affects DiT through:
  - prompt text hidden;
  - `status_feature` consumed by `ego_status_encoder`.
- The audit now records `prompt_command`, `scene_command`, `high_command_values`, `raw_command_values`, `status_feature_values`, and policy labels so train/val/nav command distributions can be compared.

### Current Image Alignment

- Previous bridge behavior selected `row["images"][0]` as current image.
- Stage1 submission conversion uses `images[-1]`.
- ReCogDrive feature builder uses the latest/current frame (`cameras[-1]`, `ego_statuses[-1]`).
- This is a P0 risk when rows contain multiple images.
- Added `--current-image-policy` with `first`, `last`, `single_or_last`, `strict_single`.
- Current launcher default is `single_or_last`, matching the stage1 submission conversion policy while preserving strict behavior for single-image rows.

### Prompt/Tensor Source

- Older bridge/cache runs had dual source:
  - prompt text from row JSON;
  - planner tensors from SceneLoader.
- Added:
  - `--prompt-source row|scene|row_strict_scene_check`;
  - `--planner-source json|scene|json_strict_scene_check`;
  - `build_qwen_prompt_from_scene_tensors(...)`;
  - JSON prompt parser for `Command`, `Velocity`, `Acceleration`, `Historical trajectory`;
  - JSON target parser for assistant/GT/trajectory fields;
  - row-vs-scene command/history/velocity/acceleration alignment metadata.
- Current default is row prompt hidden plus JSON-derived planner tensors. SceneLoader is only required to repair/check JSON splits and to recover token/log metadata where the JSON does not carry token fields.

### Trajectory/Support Contract

- DiT forward still normalizes `action_input.action` internally via `norm_odo`.
- Inference still denormalizes via `denorm_odo`.
- Cache `trajectory` and `support_trajectories` should be raw NAVSIM ego-local trajectories.
- Added audit statistics for raw scale, `norm_odo` scale, out-of-range ratio, support counts/weights/scores, and round-trip error.
- Support pool is not treated as the default problem. It remains train-only and auditable.

## Code Changes

- `scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py`
  - Added current-image policy selection and metadata.
  - Added scene-derived prompt builder and row strict check mode.
  - Added `--planner-source json|scene|json_strict_scene_check`.
  - Added JSON-derived planner tensor construction from prompt command/history/velocity/acceleration and JSON target text.
  - Added prompt/status/command/current image provenance fields.
  - Added optional control-convention prompt block for scene prompts only.
- `scripts/onevl/repair_prompt4hist_json_from_scene.py`
  - Added one-time JSON repair/check utility to create a canonical row-prompt dataset before cache generation.
- `scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh`
  - Added environment passthrough for current-image/prompt controls.
- `scripts/onevl/build_prompt4hist_stage2_eval_caches.sh`
  - Added the same cache-generation controls for val/nav cache builds.
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
  - Added backward-compatible QwenContextAdapter modes.
  - Added optional action-aware auxiliary coarse head, default off.
  - Added context/token diagnostics.
- `scripts/train_recogdrive_expert_chunked.py`
  - Added CLI overrides for adapter and action-aware aux.
  - Added training log fields for new diagnostics.
- `configs/onevl_ar_answer_stage2_small.yaml`
  - Explicitly records default adapter/aux settings.
- `scripts/onevl/audit_onevl_stage2_contract.py`
  - New unified contract audit script.
- Tests:
  - `tests/test_bridge_current_image_policy.py`
  - `tests/test_qwen_context_adapter.py`
  - `tests/test_action_aware_aux_head.py`
  - `tests/test_onevl_stage2_contract_audit.py`

Default behavior changed: yes for cache generation. The current-image policy is now `single_or_last`, and planner tensors default to `--planner-source json`. Prompt source remains `row`, adapter remains `linear`, and action-aware aux remains disabled.

## Validation

Run commands:

```bash
python -m py_compile \
  scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py \
  scripts/onevl/repair_prompt4hist_json_from_scene.py \
  scripts/onevl/audit_onevl_stage2_contract.py \
  scripts/train_recogdrive_expert_chunked.py \
  scripts/eval_recogdrive_expert_pdm.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  navsim/agents/recogdrive/recogdrive_dit.py \
  navsim/agents/recogdrive/blocks/attention.py
```

Result: passed.

```bash
pytest -q \
  tests/test_onevl_stage2_contract_audit.py \
  tests/test_qwen_context_adapter.py \
  tests/test_bridge_current_image_policy.py \
  tests/test_action_aware_aux_head.py
```

Result: `12 passed`.

No full training, navtest, or val6000 evaluation was run.

Additional JSON single-source validation run after the cache bridge update:

```bash
python scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py \
  --data-jsonl /mnt/project/onevl/test_data/navsim_test_prompt4hist_scene_aligned.json \
  --navsim-log-path /mnt/navsim/test_navsim_logs \
  --output-dir /tmp/onevl_fixed_navtest_json_strict_full \
  --max-samples 0 \
  --skip-vlm \
  --no-save-cache \
  --no-dit-forward \
  --current-image-policy single_or_last \
  --prompt-source row \
  --planner-source json_strict_scene_check
```

Result: passed for all 12146 navtest rows. `planner_state_source=json_prompt_fields`;
prompt command distribution was `MOVE FORWARD=8070`, `TURN LEFT=2501`,
`TURN RIGHT=1575`; maximum JSON-vs-SceneLoader tensor differences were about
0.005 from two-decimal prompt rounding.

```bash
python scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py \
  --data-jsonl /mnt/project/onevl/test_data/navsim_test_prompt4hist_scene_aligned.json \
  --navsim-log-path /mnt/navsim/test_navsim_logs \
  --output-dir /tmp/onevl_json_single_source_cache_smoke \
  --max-samples 8 \
  --skip-vlm \
  --save-cache \
  --cache-format flat \
  --no-skip-existing \
  --no-dit-forward \
  --current-image-policy single_or_last \
  --prompt-source row \
  --planner-source json_strict_scene_check
```

Result: saved-cache smoke passed. The sample payload contains
`last_hidden_state`, `history_trajectory=[4,3]`, `status_feature=[8]`,
`high_command_one_hot=[3]`, `trajectory=[8,3]`, and meta
`status_policy=json_command4_velocity2_acceleration2`. The last two
`status_feature` values match the JSON prompt `Acceleration` field.

## How To Run Audit

```bash
python scripts/onevl/audit_onevl_stage2_contract.py \
  --config configs/onevl_ar_answer_stage2_small.yaml \
  --train-cache-root /path/to/cache_full103k \
  --val-cache-root /path/to/cache_val6000 \
  --nav-cache-root /path/to/cache_navtest \
  --chunk-name-pattern 'shard_*' \
  --val-chunk-name-pattern 'val6000_chunk_*' \
  --max-samples-per-split 1024 \
  --support-index-path /mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt \
  --output-json /tmp/onevl_stage2_contract_audit.json \
  --output-md /tmp/onevl_stage2_contract_audit.md
```

## Suggested Next Experiments

Do not include residual DiT in the next contract experiments.

1. Current image policy audit:

```bash
CURRENT_IMAGE_POLICY=first PROMPT_SOURCE=row RUN_TRAIN=0 \
scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh
```

Then repeat with:

```bash
CURRENT_IMAGE_POLICY=last PROMPT_SOURCE=row RUN_TRAIN=0 \
scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh
```

2. Row prompt strict check:

```bash
CURRENT_IMAGE_POLICY=last PROMPT_SOURCE=row_strict_scene_check RUN_TRAIN=0 \
scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh
```

3. Adapter ablation:

```bash
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/onevl_ar_answer_stage2_small.yaml \
  --vlm-adapter-type pre_ln_linear_post_ln \
  --chunk-cache-root /path/to/cache_full103k \
  --chunk-name-pattern 'shard_*' \
  --global-epochs 200 \
  --flat-global-dataset \
  --output-dir /path/to/train_pre_ln_post_ln
```

4. Action-aware aux warmup then diffusion:

```bash
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/onevl_ar_answer_stage2_small.yaml \
  --diffusion-loss-weight 0 \
  --use-action-aware-aux \
  --action-aware-aux-weight 1.0 \
  --vlm-adapter-type pre_ln_linear_post_ln \
  --chunk-cache-root /path/to/cache_full103k \
  --chunk-name-pattern 'shard_*' \
  --global-epochs 5 \
  --flat-global-dataset \
  --output-dir /path/to/adapter_aux_warmup
```

Then resume with diffusion loss enabled and aux disabled or down-weighted.

5. ReCogDrive hidden parity, if original ReCogDrive cache is available:

Compare InternVL hidden length/norm/context token diagnostics against Qwen cache with the audit script.

## Remaining Risks

- Existing full cache must be regenerated or audited with the new metadata to prove image/token alignment.
- Older caches generated with the historical `first` image policy should be treated as compatibility artifacts; current launchers default to `single_or_last`.
- Scene prompt builder is available but not default; changing prompt source may affect Stage1 hidden distribution and should be tested separately.
- Qwen hidden distribution may still require adapter warmup even after current-frame alignment is fixed.
- Support target generalization to navtest is unproven; audit can show scale/token consistency but not guarantee split generalization.
- Fixed ReCogDrive normalization ranges are unchanged; audit must decide whether OneVL target/support ranges are under-covered.
- Full training/eval has not been run in this task.

## Follow-Up Cache Alignment Check

Date: 2026-06-29.

Checked current symlinks:

- Train cache: `/mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_full103k_prompt4hist_latest`
- Val6000 cache: `/mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_val6000_prompt4hist_latest`
- Navtest cache: `/mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_navtest_prompt4hist_latest`

Findings:

- The source JSON rows for train, val6000, and navtest all contain exactly one image per row, so `images[0] == images[-1]`. The feared first/last image policy mismatch does not apply to these specific caches.
- A 256-sample token-level check per split found `0` mismatches for `row image -> SceneLoader token -> cache sample_token/scene_token`.
- OneVL train/val/nav cache history tensors are all `[4, 3]`; in 256 checked samples per split, `history_trajectory[-1] == [0, 0, 0]`.
- ReCogDrive official navtest expert cache also uses `[4, 3]` history; in 256 checked samples, `history_trajectory[-1] == [0, 0, 0]`.
- Train and val6000 prompt command/history/velocity/acceleration align with cache/SceneLoader tensors within rounding tolerance.
- Navtest prompt history/velocity/acceleration align, but navtest prompt command does not: `/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.json` has `Command: MOVE FORWARD` for all 12146 rows, while cache `high_command_one_hot` and `status_feature` come from SceneLoader and include left/right commands. This is a real train/eval hidden prompt inconsistency.
- The navtest JSON was repaired into `/mnt/project/onevl/test_data/navsim_test_prompt4hist_scene_aligned.json`.
- A full 12146-row `--planner-source json_strict_scene_check` pass on the repaired navtest JSON succeeded. The repaired prompt command distribution is `MOVE FORWARD=8070`, `TURN LEFT=2501`, `TURN RIGHT=1575`, and all checked samples use `status_policy=json_command4_velocity2_acceleration2`.
- A saved-cache smoke test confirmed `status_feature=[command4, velocity2, acceleration2]`; the final two values exactly match the JSON prompt `Acceleration` field.

Interpretation:

- The current major confirmed alignment problem was not 3-frame vs 4-frame history and not first-vs-last image selection.
- The confirmed issue was navtest hidden extraction prompt command. The repaired JSON fixes this at the dataset level while keeping row prompt hidden extraction.
- The next cache regeneration should use repaired/prepared JSON with `PROMPT_SOURCE=row`, `PLANNER_SOURCE=json`, and `CURRENT_IMAGE_POLICY=single_or_last`. Use `PLANNER_SOURCE=json_strict_scene_check` as a preflight audit when preparing a new split.
