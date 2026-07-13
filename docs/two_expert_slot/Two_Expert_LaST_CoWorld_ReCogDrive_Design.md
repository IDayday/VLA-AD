# Two-Expert LaST/CoWorld Slot ReCogDrive

This route keeps ReCogDrive Stage2 DiT and replaces the old action-side CoT path with VLM-internal soft expert slots. It combines LaST-VLA style train-time knowledge injection with a CoWorld-VLA style expert-conditioned planner, but only for two experts: JEPA dynamics and VGGT geometry.

Baseline reference: A0 official-aligned ReCogDrive Stage2 `step_00100000`, full navtest PDMS = `0.864891`.

## Why The Old Action-Side CoT Failed

The earlier no-residual/no-LoRA CoT path generated or aligned tokens after the frozen VLM hidden state. Those tokens could condition the action head, but they were not part of the VLM transformer sequence and therefore did not become VLM hidden latent knowledge. This made the path closer to an external action adapter than to LaST-VLA's VLM-side latent injection.

## VLM Soft Slots

`TwoExpertSoftSlots` owns learned soft embeddings:

- `H_dyn`: 3 groups x 12 tokens, for short/mid/long JEPA dynamics.
- `H_geo`: 12 tokens, for VGGT Feature(23) geometry.

The slots are inserted as `inputs_embeds` into the InternVL forward path. No tokenizer special tokens are added, and tokenizer embeddings are not resized. If the VLM wrapper cannot accept `inputs_embeds`, the code raises `NotImplementedError` instead of silently falling back to post-hoc hidden tokens.

## LaST-VLA Mapping

LaST-VLA external knowledge injection is adapted as:

- VLM internal latent slots: `H_dyn`, `H_geo`.
- Train-time teachers only: JEPA dynamic tokens and VGGT Feature(23) tokens.
- Masked adapters: random-masked image/VLM hidden tokens plus the corresponding expert slot hidden predict teacher features.
- No summary replacement, hard bottleneck, residual diffusion, or A4-V2 direct expert context path.

JEPA is used as the current dynamic teacher. Cosmos/Wan-style world model features are not implemented in this two-expert route.

## CoWorld-VLA Mapping

Stage2 reads cached raw VLM hidden plus cached expert slot hidden. The planner projects `H_dyn` and `H_geo` to planner dimension, cross-attends 8 horizon queries to each expert, and obtains per-step expert conditions:

- `F_dyn`: `[B,8,384]`
- `F_geo`: `[B,8,384]`

These are added to the base ReCogDrive fused input through zero-initialized delta projections. Raw VLM context remains the DiT encoder context. The diffusion target remains GT normalized trajectory.

## Training Pipeline

1. Build JEPA dynamic teacher cache: `jepa_dynamic_teacher_tokens [3,12,1024]`.
2. Build VGGT Feature(23) teacher cache: `vggt_feature23_tokens [12,D]`.
3. Stage1 VLM SFT aligns soft slots through masked adapters and a weak trajectory probe.
4. Build hidden cache after Stage1: raw VLM hidden, `two_expert_h_dyn`, `two_expert_h_geo`.
5. Stage2 DiT SFT trains ReCogDrive diffusion with horizon-aligned expert conditions.
6. Eval and corruption check modes: normal, zero dynamic, zero geometry, zero all experts, raw VLM only, dyn only, geo only.

## Inference Rules

External teacher models are not used during Stage2 eval/inference. Eval caches do not require teacher targets, and the eval path ignores train-only target keys. Prediction is direct denormalized diffusion output, not coarse plus residual.

## Implementation Entry Points

- Soft slots: `navsim/agents/recogdrive/two_expert_slots.py`
- Adapters: `navsim/agents/recogdrive/two_expert_adapters.py`
- Stage1 module: `navsim/agents/recogdrive/two_expert_vlm_sft.py`
- VLM injection: `ReCogDriveBackbone.forward_with_two_expert_slots`
- Stage2 conditioning: `ReCogDriveDiffusionPlanner` with `use_two_expert_slots=true`
- Cache tools: `scripts/last_vla_v2/two_expert_slot/`
- Audits: `scripts/audit_two_expert_teacher_cache.py`, `scripts/audit_two_expert_hidden_cache.py`
