# ReCogDrive-LaST-v2 Design

This implementation adds a new `last_vla` path while preserving A0, A4, and LaST-RD v1 behavior.

## Difference From LaST-RD v1

LaST-RD v1 is retained under `use_last_rd`. It appends latent reasoning/adaptation tokens to the existing planner context. ReCogDrive-LaST-v2 is enabled only with `use_last_vla=True` and routes the planner through `LastVLACoTTransformer` in `navsim/agents/recogdrive/last_vla_cot_planning.py`.

`use_last_vla` and `use_last_rd` are mutually exclusive. A0/A4/LastRD configs still instantiate and forward through their old branches when `use_last_vla=False`.

## CoT Bottleneck

Last-VLA compresses raw VLM tokens with learned summary queries and cross-attention, then performs latent geometry, dynamics, ego-intent, and action-refinement CoT steps. In bottleneck mode, DiT context is only:

- final CoT tokens
- optional compressed VLM summary tokens

Full raw VLM tokens are not passed to DiT when `last_vla_cot_bottleneck_mode=true` and `last_vla_raw_vlm_context_to_dit=false`.

## Dual Teacher Alignment

Geometry CoT tokens are supervised by VGGT geometry/depth/pointmap/camera tokens when available. Patch-context fallback is separate and reported as `patch_fallback`; it should be disabled for SOTA configs once full geometry cache exists.

Dynamic CoT tokens predict future JEPA/world latent tokens. Future target tokens are train-only and are ignored in eval/get_action.

## Action-Conditioned Dynamics

Dynamic prediction attends to action condition tokens built from:

- coarse trajectory
- noisy residual/current action
- coarse + noisy residual candidate full action
- diffusion timestep embedding

This makes future JEPA/world latent prediction depend on the action being denoised.

## Residual Diffusion

The CoT path predicts `coarse_traj_norm` as the main trajectory prior. Diffusion then predicts a residual:

`residual_target = selected_target_norm - coarse_traj_norm`

During inference, the sampled residual is added back to the coarse prior before denormalization.

## Teacher Trajectory SFT

`_select_last_vla_training_target` supports:

- `none` / `gt`: use ground truth normalized action
- `teacher_if_better`: use teacher trajectory when `teacher_score >= gt_score + margin`
- `mix`: epoch-scheduled interpolation between GT and teacher

Teacher trajectory caches are generated out-of-place and merged into chunk caches only through the explicit merge script.

## Geometry Teacher Requirements

Smoke and geometry-lite experiments may set `last_vla_allow_patch_geometry_fallback=true`. SOTA-oriented configs should set:

- `last_vla_require_full_geometry=true`
- `last_vla_allow_patch_geometry_fallback=false`

when full VGGT teacher cache exists.

## Optional VLM LoRA

Agent-level LoRA hooks are present but disabled by default. If `last_vla_train_vlm_lora=True`, cached hidden-state training is rejected because VLM LoRA requires online VLM forward or regenerated hidden caches.

The recommended Line B latent adaptation setting is `attention_mlp`, scope `llm`, rank `32`, alpha `64`, dropout `0.05`, and rsLoRA enabled. The old q/k/v/o rank-16 configuration remains available as a conservative baseline. See `docs/Last_VLA_v2_VLM_LoRA_Design.md` for target module auditing, optimizer grouping, hidden-anchor regularization, and adapter save/load format.

## Training Stages

1. `cot_alignment`: train only Last-VLA CoT/teacher/coarse heads, diffusion loss off.
2. `progressive_sft_bottleneck`: enable residual diffusion through the CoT bottleneck with auxiliary loss floors.
3. `teacher_traj_sft`: train with best-of-K/PDM-reranked teacher trajectory targets.

Strict Last-VLA v2 SFT uses `cot_alignment -> progressive_sft_bottleneck` as the core path. PDM-reranked teacher trajectory SFT is an optional extension for later diagnostics and performance exploration; it is not required for LaST-VLA SFT alignment. GRPO is deferred and not part of this stage.

## Full Geometry Cache

Strict reproduction requires full VGGT geometry teacher tokens with `vggt_geometry_mode=full_geometry` and `vggt_geometry_mode_code=2`. The geometry tokenizer packs VGGT depth/point-map/camera outputs into deterministic non-trainable geometry teacher tokens. Legacy smoke configs may use `[12, 512]`; the formal high-cap config uses `[192, 512]`.

Patch fallback is geometry-lite only. It may be used for smoke tests or ablations, but it must be reported as `patch_fallback` and cannot be used for strict SOTA claims.

## High-capacity No-risk Official Configuration

Minimal Last-VLA configs remain smoke/debug only. The formal Last-VLA v2 SFT line is the high-capacity no-risk configuration:

- VLM summary tokens: `64`
- latent CoT tokens: `192`
- JEPA context and target tokens: `128 x 1024`
- dynamic teacher tokens: `128`
- VGGT full geometry tokens: `192 x 512`
- geometry grid: `12 x 16`
- risk branch: disabled, `num_risk_tokens=0`, `last_vla_risk_loss_weight=0.0`
- patch fallback: disabled

The expected DiT context length in eval, when the summary is kept, is:

`192 CoT + 64 VLM summary = 256 tokens`

Full raw VLM tokens still do not enter DiT in bottleneck mode. Training may schedule the summary keep/drop probability, but the raw hidden sequence is not used as DiT context when `last_vla_cot_bottleneck_mode=true` and `last_vla_raw_vlm_context_to_dit=false`.

High-cap no-risk deliberately excludes the risk branch, PDM-reranked SFT as the main line, and GRPO. The strict SFT path is:

1. frozen or VLM-LoRA CoT alignment with full VGGT geometry and JEPA future dynamics teachers;
2. progressive residual-diffusion bottleneck SFT;
3. optional VLM-LoRA hidden-cache regeneration for the Line B frozen-cache phase and navtest eval.

The high-cap cache is not compatible with old 12-token JEPA caches. JEPA cache generation must expose dense or sufficiently many JEPA tokens and deterministically pool to `128`; re-pooling or repeating the old 12 already-pooled tokens is invalid. Full VGGT geometry cache must also be regenerated to `[192, 512]` with mode code `2`.

## Progressive Residual Schedule

Progressive SFT uses a residual alpha schedule:

`diffusion_target = selected_target_norm - alpha * coarse_traj_norm`

At alpha `0`, the diffusion target matches A0 full-action training. At alpha `1`, the model trains on the residual to the CoT coarse prior. Inference always uses alpha `1` and returns `coarse + sampled_residual`.

The VLM summary context is separately scheduled inside the CoT bottleneck. Full raw VLM tokens remain disabled in strict bottleneck mode.

## Two-Server Round2 Plan

Server A runs the frozen-VLM line using the original ReCogDrive Stage1 hidden cache plus strict full-geometry teacher cache.

Server B runs the VLM-LoRA line:

1. online VLM-LoRA CoT alignment with full geometry and future JEPA teachers;
2. extract VLM LoRA and Last-VLA CoT adapters;
3. regenerate train/val hidden-state cache with the LoRA adapter;
4. run progressive bottleneck SFT on the regenerated hidden cache.

VLM-LoRA cannot train on static cached hidden states. It requires online VLM forward during alignment, then regenerated hidden cache for frozen-cache SFT/eval.

## Success Criteria

The hard diagnostic before teacher trajectory SFT is best-of-K oracle gain:

- oracle best-of-8 should exceed deterministic by at least 1 PDMS point, otherwise teacher trajectory SFT is unlikely to provide a 3-point gain.

Corruption eval should show normal minus zero-all-CoT dependency:

- at least 0.01 on full eval, or 0.008 on a 1k subset.

## Readiness Fixes Before SOTA Training

Production Last-VLA v2 training must not use proxy diagnostics as proof of progress.

1. Best-of-K oracle and teacher trajectory cache generation use real NAVSIM metric-cache PDM when `--score-mode=pdm`. Missing metric cache is a hard failure.
2. Proxy scoring is only for smoke/debug via `--score-mode=proxy`; proxy outputs keep PDMS and submetrics null and cannot support SOTA claims.
3. CoT corruption eval reports PDMS and NAVSIM submetrics in PDM mode, not only trajectory L1. Proxy corruption mode is opt-in and clearly labeled.
4. Geometry mode is explicit: `full_geometry`, `patch_fallback`, `no_geometry`, or `missing`. Patch fallback is never reported as full geometry.
5. Teacher trajectory SFT should not run in production until best-of-K oracle PDM gain is positive and meaningful, teacher trajectory coverage is at least 99%, and geometry mode coverage has been audited.

## Baseline

All training/eval reports must use A0-official-aligned as the primary comparison:

`A0-official-aligned full navtest best PDMS = 0.864891`
