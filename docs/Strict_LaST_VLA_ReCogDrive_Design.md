# Strict LaST-VLA ReCogDrive Design

This document describes the strict LaST-VLA external-knowledge injection adaptation for ReCogDrive.

## Scope

The implementation targets a clean comparison against the A0 official-aligned ReCogDrive Stage2 baseline:

- A0 official-aligned `step_00100000` full NAVTEST PDMS: `0.864891`.
- Stage2 diffusion target remains the GT normalized trajectory.
- No A4-V2 direct expert context path.
- No VLM LoRA.
- No hard bottleneck.
- No VLM summary replacement.
- No residual diffusion target.
- No coarse-plus-residual final output.
- External world model and VGGT features are train-time teachers only.
- Inference does not run external teacher models.

## VLM-Side Latent Slots

Strict LaST-VLA latent CoT is represented as VLM hidden-sequence soft slots:

- `H_dyn`: dynamics latent slots, default `64`.
- `H_geo`: geometry latent slots, default `64`.
- `H_plan`: planning latent slots, default `32`.

`navsim/agents/recogdrive/last_vla_latent_slots.py` implements:

- `LastVLALatentSlotConfig`
- `LastVLASoftLatentSlots`
- `LastVLASlotBatch`
- `StructuredCausalMaskBuilder`

The slots are inserted through `inputs_embeds`; tokenizer special tokens are not added by default. If the underlying VLM wrapper cannot accept `inputs_embeds` or strict 4D masks, the code raises `NotImplementedError` instead of silently skipping the strict path.

## Teacher Adapters

`navsim/agents/recogdrive/last_vla_teacher_adapters.py` implements:

- `RandomTokenMask`
- `DynamicsAdapter`
- `GeometryAdapter`
- `PlanLatentHead`

The Dynamics and Geometry adapters consume random-masked VLM/image hidden tokens plus the corresponding latent hidden slots. They predict teacher features:

- Dynamics: `[B, 128, 1024]`, using JEPA dev fallback now, Cosmos path reserved.
- Geometry: `[B, 192, 512]`, using VGGT geometry tokens.

Cosmos teacher is not formally implemented yet. The current strict implementation uses `last_vla_wm_teacher_source=jepa_dev`.

## Strict Reasoner

`navsim/agents/recogdrive/last_vla_strict_reasoner.py` implements `StrictLastVLAReasoner`.

It returns:

- raw VLM hidden state
- `h_dyn`, `h_geo`, `h_plan`
- `latent_condition_tokens`
- `wm_loss`, `geometry_loss`, `plan_loss`, `heading_loss`, `progress_loss`
- diagnostics for mask ratios, slot norms, teacher missing flags, and slot counts

The latent condition projection is:

`LayerNorm -> Linear -> GELU -> Linear -> LayerNorm`

The old action-side `LastVLACoTTransformer` is not used in strict mode.

## Stage1

Config: `configs/last_vla_v2/strict_last_vla/stage1_latent_alignment.yaml`

Stage1 runs frozen-VLM online latent alignment:

- `use_strict_last_vla=true`
- `last_vla_stage=stage1_latent_alignment`
- `cache_hidden_state=false`
- `freeze_vlm_backbone=true`
- `train_latent_slots=true`
- `train_adapters=true`
- `train_dit=false`
- `diffusion_loss_weight=0.0`

The VLM base is frozen; soft latent slots and strict adapters are trainable.

## Cache

Script: `scripts/last_vla_v2/strict_last_vla/build_strict_last_vla_latent_cache.py`

Strict Stage2 cache schema:

- `last_hidden_state`
- `last_vla_h_dyn`
- `last_vla_h_geo`
- `last_vla_h_plan`
- `last_vla_slot_metadata`
- `trajectory`
- `history_trajectory`
- `high_command_one_hot`
- `status_feature`

Train cache may retain teacher targets for auxiliary floors. Eval/navtest cache must not include future teacher targets unless explicitly flagged, and the planner ignores teacher targets during inference.

Audit script: `scripts/audit_strict_last_vla_latent_cache.py`

## Stage2

Config: `configs/last_vla_v2/strict_last_vla/stage2_diffusion_sft.yaml`

Stage2 follows the A0 official-aligned training surface:

- `cache_hidden_state=true`
- `diffusion_loss_weight=1.0`
- `last_vla_use_residual_diffusion=false`
- `last_vla_teacher_traj_mode=none`
- `lr=1e-4`
- `batch_size=16/GPU`
- `epochs=200`
- `warmup=3`
- `min_lr=1e-6`

Planner behavior:

- `last_hidden_state` is projected through the existing `feature_encoder` into raw VLM context tokens.
- `last_vla_h_dyn/h_geo/h_plan` are projected to latent condition tokens.
- DiT receives raw VLM tokens as `encoder_hidden_states`.
- DiT receives latent condition tokens through the existing decoupled condition branch.
- Diffusion target is GT normalized trajectory.

## Differences From Previous Paths

Compared with old action-side CoT:

- latent CoT lives inside VLM hidden slots, not an action-side transformer.
- adapters align VLM hidden slots to teachers.
- Stage2 consumes cached VLM-side latent slots.

Compared with A4-V2:

- no direct JEPA/VGGT expert context path is passed to DiT.
- external teachers supervise adapters during training only.
- inference uses only cached VLM hidden and latent slots.

Compared with full LaST-VLA:

- Cosmos is not integrated yet.
- JEPA dev features are the current world-model teacher fallback.
