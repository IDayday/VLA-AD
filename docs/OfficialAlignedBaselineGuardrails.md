# Official-Aligned Stage2 Baseline Guardrails

This document records the verified ReCogDrive Stage2 training contract used as the baseline for A0 and for the A4-V2 migration. Future A4 algorithm work should avoid changing these settings unless the change is explicitly reviewed against this checklist.

## Verified A0 Baseline

The local official-aligned A0 Stage2 reproduction matched the public ReCogDrive Stage2 performance range in our environment. The best local official-aligned checkpoint reached PDMS around `0.8649` on full navtest.

The protected training entrypoint is:

```bash
navsim/planning/script/run_training_recogdrive.py
```

It must train through:

- `ReCogDriveAgent`
- `AgentLightningModule`
- the official `features, targets, tokens_list` batch contract
- PyTorch Lightning `Trainer`

The local chunk-cache adapter is allowed only as a data source adapter. It must not change model forward semantics, optimizer semantics, scheduler semantics, checkpoint selection, or evaluation semantics.

## Protected Training Settings

Keep these settings fixed for official-aligned Stage2 comparisons:

| Area | Required setting |
|---|---|
| Hardware | 1 server, 8 GPUs, `torchrun --nproc_per_node=8` |
| Per-GPU batch | `16` |
| Grad accumulation | `1` |
| Effective batch | `128` |
| Epochs | `200` for full Stage2 |
| Precision | Lightning `16-mixed`; model weights stay fp32 |
| Optimizer | AdamW |
| LR | `1e-4` base LR |
| Weight decay | `1e-4` |
| Betas | `(0.9, 0.95)` |
| Scheduler | `WarmupCosLR` |
| Warmup | `3` epochs |
| Min LR | `1e-6` |
| Validation | official train/val log split |
| Checkpoint monitor | `val/loss_epoch`, mode `min`, top-k checkpoints |
| Exact checkpoints | `50k, 60k, 80k, 100k, 120k, 140k, 160k, latest` |
| Eval precision | `fp32` |
| Eval split | full navtest unless explicitly marked as subset |

LR schedule: warm up from about `3.333e-5` to `1e-4` over the first 3 epochs, then cosine decay to `1e-6` by epoch 200.

## Data Contract

The official-aligned local loader reads local chunk cache records but returns the official collate structure:

```python
return features, targets, tokens_list
```

Base A0 feature keys:

- `history_trajectory`: `[4, 3]`, float32
- `high_command_one_hot`: `[4]`, float32
- `last_hidden_state`: `[N, 1536]`, float32, padded by `pad_sequence(..., batch_first=True, padding_value=0.0)`
- `status_feature`: `[8]`, float32

Target keys:

- `trajectory`: `[8, 3]`, float32

A4-V2 expert keys, when enabled:

- `jepa_context_tokens`: `[12, 1024]`
- `vggt_context_tokens`: `[12, 2048]`
- `jepa_target_tokens`: `[12, 1024]`, train-only alignment supervision
- `vggt_target_tokens`: `[12, 2048]`, train-only alignment supervision

The loader must fail fast if required A4 expert keys are missing or have mismatched shapes. It must not silently fill missing expert tokens with zeros for real training.

## Evaluation Contract

Evaluation remains direct planner evaluation through:

```bash
scripts/eval_recogdrive_expert_pdm.py
```

Rules:

- Use `--precision fp32`.
- Use the same navtest chunk cache and metric cache for paired comparisons.
- Do not pass `jepa_target_tokens` or `vggt_target_tokens` into eval.
- Keep `allow_future_targets_in_inference=false`.
- Mark any `--max-samples` run as subset smoke, not final evidence.

## A4-V2 Migration Points

A4-V2 now has official-aligned Hydra agent configs:

- `navsim/planning/script/config/common/agent/recogdrive_agent_a4_v2.yaml`
- `navsim/planning/script/config/common/agent/recogdrive_agent_a4_v2_align_stage1.yaml`
- `navsim/planning/script/config/common/agent/recogdrive_agent_a4_v2_stage2_noalign.yaml`

Launchers:

- `scripts/run_a4_v2_official_aligned_8gpu.sh`
- `scripts/run_a4_v2_official_align_first_8gpu.sh`
- `scripts/eval_a4_v2_official_aligned_checkpoints.sh`

Allowed A4 algorithm work should mainly touch:

- A4-V2 config values under the A4 agent configs
- expert branch modules in `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- expert cache generation and validation scripts
- A4-only launch wrappers

Avoid modifying the protected baseline trainer, scheduler, precision mode, train/val split, checkpoint monitor, or eval semantics unless the change is explicitly documented and revalidated against A0.

## Review Checklist

Before merging future A4 changes, check:

- A0 official-aligned command still uses `run_training_recogdrive.py`.
- Batch format is still `features, targets, tokens_list`.
- `last_hidden_state` is still padded with `pad_sequence`.
- Model parameters are not manually cast to bf16/fp16.
- Lightning precision is still the only mixed-precision mechanism for official-aligned runs.
- AdamW, LR, betas, weight decay, warmup, cosine decay, and min LR match the table above unless the experiment is explicitly a scheduler ablation.
- Train and val are split by official log lists with zero sample-token overlap.
- ModelCheckpoint still monitors `val/loss_epoch`.
- Eval remains fp32 and does not use future expert target tokens.
- Any subset eval is labeled as smoke or pilot.
