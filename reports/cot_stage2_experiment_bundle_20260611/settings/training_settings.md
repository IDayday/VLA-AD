# Training Settings Summary

Generated: 2026-06-11 UTC

## Baselines

- A0 official-aligned ReCogDrive stage2: official Lightning-style training on local chunk cache; max epochs 200; batch size 16; AdamW lr 1e-4; validation enabled; checkpoint selection by full NAVTEST PDMS identified `step_00100000` as best (`PDMS=0.864891`).
- Base 2B IL NAVTEST harness baseline: `ReCogDrive-2B-IL`, `pred_traj`, full NAVTEST `PDMS=0.858575`.
- Direct VLM text baseline: `ReCogDrive-VLM-2B`, direct `[PT,...]` text trajectory, full NAVTEST `PDMS=0.845804`; this is not a diffusion stage2 result.

## Decoupled HighCap NoRisk CoT

- Raw ReCogDrive VLM hidden tokens remain the base DiT context.
- Latent CoT enters through `last_vla_condition_mode=decoupled_cot_residual`.
- CoT target length: 192 latent tokens; CoT steps: 5.
- JEPA dynamic/context tokens: 128 x 1024.
- VGGT geometry teacher tokens: 192 x 512.
- Risk supervision disabled: `num_risk_tokens=0`, `last_vla_use_risk_head=false`, risk weights 0.
- Policy-KD disabled in these runs: `policy_kd_loss_weight=0.0`, `policy_kd_mode=none`.
- A line: frozen VLM hidden cache plus CoT/progressive stage2.
- B line: VLM LoRA CoT alignment, adapter extraction, optional online/regen hidden state, then progressive stage2.

## No-Residual Stage2

- Restores original diffusion target semantics: target is `selected_target_norm` / GT.
- `pred_traj` is the denormalized diffusion output, not `coarse_traj + residual`.
- `pred_coarse_traj` is diagnostic only.
- Validation stays enabled so top-5 val/loss checkpoints can be swept on NAVTEST.

## VLM-Text Residual Anchor

- Generates fixed 2B-base direct text trajectories first.
- Stores anchors in chunk cache as `vlm_text_trajectory_norm` / `vlm_text_trajectory`.
- Training target is residual from the fixed anchor; inference reconstructs anchor plus predicted residual.
- Pure ReCogDrive control sets `use_last_vla=false`; Last-VLA A keeps CoT/expert condition machinery. These are not a perfectly isolated one-factor ablation.

## Evaluation Protocol

- Full NAVTEST rows use `12146` samples and usually `12138` valid PDM rows with `8` known missing metric-cache rows.
- Evaluation precision is fp32 for stage2 PDM sweeps unless noted.
- Expert target tokens are disabled in eval.
- Sharded eval output is aggregated into a parent `metrics.json`; shard-level `metrics.json` files are excluded from `data/full_navtest_pdms.csv`.
