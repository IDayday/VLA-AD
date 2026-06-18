# Stage2 Flat Context Design

Goal: test a deliberately simple Stage2 where two-expert Stage1 hidden states are fed directly to the original ReCogDrive DiT context, without adding the denoise-time HMEF branch.

## What Changes

- Add `two_expert_dit_condition_mode=flat_context`.
- Keep raw VLM context tokens.
- Flatten `H_dyn [B,3,12,D]` to `36` tokens.
- Keep `H_geo [B,12,D]` as `12` tokens.
- Project both to planner dim with the existing two-expert projection MLPs.
- Add a two-token type embedding for dynamic vs geometry.
- Concatenate `[raw_vlm_tokens, dyn_tokens, geo_tokens]` and pass this as `encoder_hidden_states` to the existing `LightningDiT`.
- Use the mean of the concatenated context as the normal context mean feature.

## What Does Not Change

- No DiT block modification.
- No denoise gate.
- No expert branch fusion weight.
- No explicit expert-timestep interaction.
- No explicit expert-noisy-action interaction.
- No residual diffusion.
- No Stage3.
- No token count change in Stage1.
- Stage2 target remains GT normalized trajectory.

## Expected Diagnostics

- `two_expert_flat_context_enabled=1`
- `two_expert_denoise_v2_enabled=0`
- `two_expert_memory_token_count=48` in normal mode.
- `two_expert_context_token_count=raw_vlm_count+48` in normal mode.
- `raw_vlm_only` removes the appended expert tokens.

## Training Config

- Direct config: `configs/last_vla_v2/two_expert_slot/stage2_dit_sft_flat_context.yaml`
- Hydra experiment: `+experiment=two_expert_slot_stage2_dit_sft_flat_context`

The launcher can use the existing script:

```bash
HYDRA_EXPERIMENT=two_expert_slot_stage2_dit_sft_flat_context \
RUN_TRAIN=1 \
TRAIN_CHUNK_CACHE_ROOT=/path/to/stage1_v2_hidden_cache \
OUTPUT_DIR=/path/to/output \
MASTER_PORT=29631 \
bash scripts/last_vla_v2/two_expert_slot/run_stage2_two_expert_dit_sft_8gpu.sh
```
