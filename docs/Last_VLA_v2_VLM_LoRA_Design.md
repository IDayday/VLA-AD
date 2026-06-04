# Last-VLA v2 VLM-LoRA Design

Baseline: A0-official-aligned full navtest PDMS `0.864891`.

## Why LoRA

VLM-LoRA adapts the VLM latent space for Last-VLA without full Stage1 VLM retraining. Line B uses online VLM-LoRA CoT alignment, extracts LoRA and CoT adapters, regenerates train/navtest hidden caches with that exact LoRA adapter, then runs progressive bottleneck SFT on regenerated hidden cache.

Hidden caches produced before LoRA are not valid for Line B after LoRA alignment. Train and navtest hidden caches must be regenerated.

## Baseline And Recommended Settings

The old conservative baseline is q/k/v/o attention-only LoRA with rank `16`, alpha `32`, no dropout, and no rsLoRA.

Recommended setting:

- preset `attention_mlp`
- scope `llm`
- rank `32`
- alpha `64`
- dropout `0.05`
- rsLoRA `true`
- DoRA `false`

High-capacity setting:

- preset `all_linear`
- scope `llm`
- rank `64`
- alpha `128`
- dropout `0.05`
- rsLoRA `true`
- hidden anchor weight `0.02`
- LoRA LR `5e-6`

## Target Module Categories

Target audit groups modules into:

- LLM attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- LLM MLP: `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`
- vision attention
- vision MLP
- projector / multimodal bridge
- other

Scopes:

- `llm`: excludes vision tower and projector modules
- `vision`: vision tower only
- `llm_vision`: LLM and vision tower
- `projector`: multimodal bridge only
- `all`: all allowed modules except `lm_head` and `action_head`

If a preset matches zero modules, startup fails. If `attention_mlp` matches no MLP modules, the target report records a high-risk warning.

Scoped presets resolve to full module names rather than bare suffixes. This prevents PEFT from globally matching `q_proj` or `gate_proj` inside the vision tower when `scope=llm`. After PEFT injection, the code audits actual trainable `lora_` parameters and fails if the trainable scope crosses into vision/projector modules unless `last_vla_lora_allow_mixed_scope=true` is explicitly set.

## Trainable Scope

During online VLM-LoRA CoT alignment, only two groups are trainable:

- `action_head.last_vla_cot.*`
- VLM LoRA parameters containing `lora_`

Non-LoRA backbone parameters and action base parameters must be frozen. `cache_hidden_state=true` with `last_vla_train_vlm_lora=true` raises `ValueError`.

## Optimizer

VLM-LoRA alignment uses explicit optimizer groups:

- `last_vla_cot`: LR `1e-4`, weight decay `1e-4`
- `vlm_lora`: LR `1e-5`, weight decay `0`

The high-cap all-linear preset uses LoRA LR `5e-6`.

## Hidden Anchor

Hidden-anchor regularization keeps LoRA from destroying the ReCogDrive semantic hidden space. During VLM-LoRA `cot_alignment`, the agent computes LoRA hidden states and frozen-base hidden states with the PEFT adapter disabled. The default mode is `summary_cosine`:

`loss = 1 - cosine(mean(lora_hidden), mean(frozen_hidden))`

Default weight is `0.01`. The loss is training-only and disabled for cached hidden progressive SFT.

The frozen-base anchor forward is throttled by `last_vla_hidden_anchor_every_n_steps`, default `4`, to avoid doubling VLM forward cost every step. Skipped steps log `hidden_anchor_computed=0` and do not add a zero-valued anchor loss.

## Adapter Saving

Extraction writes:

- `last_vla_cot_adapter.pt`
- `vlm_lora/adapter_model.bin`
- `vlm_lora/adapter_config.json`
- `vlm_lora/lora_metadata.json`
- `vlm_lora/lora_target_report.json`
- `vlm_lora_adapter_state.pt` for backward compatibility

Metadata includes base VLM path, VLM type, preset, scope, target modules, resolved target modules, rank, alpha, dropout, bias, rsLoRA, DoRA, matched modules, trainable LoRA parameter count, PEFT version, source checkpoint, and adapter config hash.

## Hidden Cache Regeneration

Use the adapter directory:

```bash
python scripts/build_recogdrive_hidden_cache_with_lora.py \
  --base-chunk-root /path/to/full_geometry_chunks \
  --output-chunk-root /path/to/lora_hidden_cache \
  --vlm-path /path/to/base_vlm \
  --vlm-lora-adapter-dir /path/to/adapters/vlm_lora
```

Manual `--lora-r`, `--lora-alpha`, or `--lora-target-modules` is legacy fallback only. If both adapter dir metadata and explicit CLI config are provided and mismatch, regeneration fails unless `--allow-lora-config-override` is set.

## Sweep Plan

Dry-run:

```bash
RUN_TRAIN=0 \
LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS=1 \
NAVSIM_LOG_PATH=/tmp/navsim_logs \
SENSOR_BLOBS_PATH=/tmp/sensor_blobs \
EXPERT_TEACHER_CACHE_ROOT=/tmp/expert_cache \
VLM_PATH=/tmp/vlm \
OUT_ROOT=/tmp/last_vla_lora_sweep \
MASTER_PORT_BASE=29600 \
scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh
```

Default sweep:

- `last_vla_vlm_lora_attention_only_r16`
- `last_vla_vlm_lora_attention_mlp_r32`
- `last_vla_vlm_lora_all_linear_r64`

Summarize:

```bash
python scripts/last_vla_v2/lora/summarize_lora_alignment_sweep.py \
  --out-root /tmp/last_vla_lora_sweep
```

## Evaluation

Line B eval must use LoRA-regenerated navtest hidden cache. Compare all regenerated-cache Line B results against A0-official-aligned PDMS `0.864891`. No performance should be claimed before full training and full PDM eval artifacts exist.
