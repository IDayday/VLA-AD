# Last-VLA v2 Decoupled CoT Design

Formal name: `ReCogDrive-LaST-v2 Decoupled HighCap NoRisk`.

Baseline for comparison: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`.

## Deprecated Paths

Hard bottleneck and VLM summary replacement are deprecated. The formal code path does not instantiate a VLM summary compressor and does not generate or consume `vlm_summary` tokens.

Deprecated hard-bottleneck / summary replacement entrypoints are archived in:

- `configs/last_vla_v2/archive/hard_bottleneck_legacy/`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/`

Formal configs live only under `configs/last_vla_v2/decoupled_highcap_no_risk/`.

Invalid formal settings:

- `last_vla_cot_bottleneck_mode=true`
- `last_vla_raw_vlm_context_to_dit=false`
- summary replacement
- patch fallback geometry
- risk head
- teacher trajectory SFT
- GRPO

## Condition Design

Raw ReCogDrive VLM hidden tokens are preserved as the DiT base cross-attention context.

Latent CoT tokens are an extra teacher-supervised reasoning/control state. They enter DiT through an independent residual cross-attention branch:

- raw VLM cross-attention remains the base path
- CoT cross-attention is a separate residual path
- CoT output projection is zero-init, so initial output matches the base path
- no scalar/global gate
- no softmax competition between VLM and CoT
- no raw/CoT concat as a substitute for the residual branch

## CoT Structure

The CoT graph is:

1. raw VLM scene grounding
2. geometry CoT and dynamic CoT in parallel
3. fusion CoT
4. ego/coarse planning
5. action-refinement CoT at the diffusion timestep

Geometry CoT is aligned to full VGGT geometry `[192, 512]` with grid `12 x 16`.

Dynamic CoT is aligned to JEPA future tokens `[128, 1024]` and can attend projected geometry memory, but it starts from scene-grounded CoT rather than depending on geometry CoT only.

## A/B Definition

A and B use the same planner, CoT, teacher losses, DiT condition branch, and progressive SFT strategy.

Only the raw VLM hidden source differs:

- A: frozen ReCogDrive VLM hidden cache
- B: VLM-LoRA alignment, then regenerated hidden cache

Line B eval must use the LoRA-regenerated navtest hidden cache.

## Formal Token Contract

- raw VLM tokens: full `last_hidden_state -> feature_encoder`
- latent CoT: `192`
- JEPA context/target: `128 x 1024`
- dynamic teacher: `128 x 1024`
- VGGT geometry: `192 x 512`
- geometry grid: `12 x 16`
- risk tokens: `0`
- summary tokens: none

## Training Objective

Stage 1 `cot_alignment` trains CoT modules and teacher/coarse heads jointly. Diffusion loss is off.

Stage 2 `progressive_sft_decoupled` trains diffusion/action head, Last-VLA CoT modules, and the CoT residual condition branch jointly. Diffusion loss dominates and geometry/dynamic/coarse/progress auxiliary losses keep floors.

## Evaluation

Required evals after training:

- full checkpoint sweep
- corruption eval
- `normal` vs `raw_vlm_only`
- `zero_scene_cot`
- `zero_geometry_cot`
- `zero_dynamic_cot`
- `zero_fusion_cot`
- `zero_action_refine_cot`
- `zero_coarse_prior`
- `zero_cot_condition_branch`

Meaningful `normal > raw_vlm_only` PDMS margin indicates the CoT branch helps beyond raw VLM context.
