# Stage2 Prefuse Cross-Attention Design

## Goal

Use the Stage1-v2 `H_dyn` and `H_geo` slot hidden states as a single expert memory bank, fuse them with raw VLM hidden states before DiT, and keep the ReCogDrive DiT interface unchanged.

This is intentionally simpler than the previous HMEF-style denoise conditioning:

- no expert branch inside DiT
- no timestep-aware expert gate
- no noisy-action expert interaction
- no residual diffusion
- no Stage3
- no action-side CoT

## Data Flow

`H_dyn` and `H_geo` are already VLM hidden-state slot tokens. The cache stores them separately from `last_hidden_state` so Stage2 can ablate and corrupt them cleanly.

The prefuse path projects them to planner dimension and treats them as one memory:

```text
V = project(raw VLM hidden)                  [B, N, 384]
E = concat(project(H_dyn), project(H_geo))   [B, 48, 384]

delta = CrossAttention(query=V, key=E, value=E)
V_fused = V + scale * out_proj(delta)

DiT context_tokens = V_fused                 [B, N, 384]
```

The DiT context length stays identical to the original raw VLM path. This differs from `flat_context`, which appends the 48 expert tokens and lets DiT learn fusion internally.

## adaLN-Zero Compatibility

The local `LightningDiT` already uses adaLN-style modulation and zero-initializes the modulation layers. Prefuse does not conflict with that design:

- adaLN-Zero stabilizes the DiT denoiser blocks.
- Prefuse cross-attention only changes the context tokens before DiT.
- The prefuse residual output projection is zero-initialized, so the initial behavior is equivalent to raw VLM context.

## Dropout

Prefuse adds two training-only regularizers:

- `two_expert_prefusion_token_dropout`: drops individual expert memory tokens.
- `two_expert_prefusion_condition_dropout`: drops the whole expert memory for a sample.

Both are disabled in eval. The whole-condition dropout is important because it forces the planner to preserve a raw-VLM fallback and makes the `raw_vlm_only` ablation meaningful.

## Recommended Config

Use:

```yaml
two_expert_condition_mode: prefuse_cross_attention
two_expert_dit_condition_mode: prefuse_cross_attention
two_expert_memory_tokens_to_dit: false
two_expert_prefusion_heads: 8
two_expert_prefusion_dropout: 0.10
two_expert_prefusion_token_dropout: 0.05
two_expert_prefusion_condition_dropout: 0.10
two_expert_prefusion_zero_init: true
```

Hydra experiment:

```text
+experiment=two_expert_slot_stage2_dit_sft_prefuse_cross_attention
```

## Required Ablations

Evaluate at minimum:

- `normal`
- `raw_vlm_only`
- `zero_h_dyn`
- `zero_h_geo`
- `zero_all_experts`
- `dyn_only`
- `geo_only`
- `random_slots`

The expected signal is `normal > raw_vlm_only`, and corruption of the learned slots should hurt relative to normal.
