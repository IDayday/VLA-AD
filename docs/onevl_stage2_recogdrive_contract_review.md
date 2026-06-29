# OneVL Stage2 ReCogDrive Contract Review

## Executive Summary

- ReCogDrive DiT is not architecturally bound to InternVL. The plain Stage2 path consumes tensors: `last_hidden_state`, `his_traj`, `status_feature`, and raw `action`; it does not call InternVL-specific APIs inside the DiT.
- Pretrained ReCogDrive/InternVL weights are distribution-bound. The learned `feature_encoder`, `fusion_projector`, DiT cross-attention, and `context_mean` conditioning were trained on InternVL hidden states. Random-init training with Qwen hidden is valid only if the cache contract is clean.
- Highest-probability contract risks for the current OneVL/Qwen migration are:
  1. P0 current image/token alignment: the bridge historically used `row["images"][0]`, while Stage1 submission conversion uses `images[-1]`; ReCogDrive feature building uses the current/latest frame.
  2. P1 prompt/planner tensor source split: hidden prompt came from JSON row text, while history/status/trajectory came from SceneLoader.
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
- Planner tensors are built from NAVSIM SceneLoader frames:
  - `history_trajectory`: 4x3 ego-local history.
  - `high_command_one_hot`: 3-way command.
  - `status_feature`: 8-d command/velocity/acceleration feature.
  - `trajectory`: raw future ego-local target.
  - optional `support_trajectories`: raw ego-local support targets for training-only sampling.
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
- Default remains `first` to avoid silently changing existing runs. The audit script should be run before changing launcher defaults.

### Prompt/Tensor Source

- Previous bridge had dual source:
  - prompt text from row JSON;
  - planner tensors from SceneLoader.
- Added:
  - `--prompt-source row|scene|row_strict_scene_check`;
  - `build_qwen_prompt_from_scene_tensors(...)`;
  - row-vs-scene command/history/velocity/acceleration alignment metadata.
- Default remains `row`.

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
  - Added prompt/status/command/current image provenance fields.
  - Added optional control-convention prompt block for scene prompts only.
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

Default behavior changed: no, except cache samples now record richer metadata when regenerated. Existing default bridge policy remains `first`, prompt source remains `row`, adapter remains `linear`, action-aware aux remains disabled.

## Validation

Run commands:

```bash
python -m py_compile \
  scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py \
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

Result: `10 passed`.

No full training, navtest, or val6000 evaluation was run.

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
- The historical `first` image default is preserved for compatibility, but may be wrong for multi-image rows.
- Scene prompt builder is available but not default; changing prompt source may affect Stage1 hidden distribution and should be tested separately.
- Qwen hidden distribution may still require adapter warmup even after current-frame alignment is fixed.
- Support target generalization to navtest is unproven; audit can show scale/token consistency but not guarantee split generalization.
- Fixed ReCogDrive normalization ranges are unchanged; audit must decide whether OneVL target/support ranges are under-covered.
- Full training/eval has not been run in this task.
