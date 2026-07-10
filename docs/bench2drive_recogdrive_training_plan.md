# Bench2Drive ReCogDrive Training Plan

This repo now has two separate Bench2Drive paths:

1. closed-loop evaluation through the official Bench2Drive leaderboard runner and CARLA 0.9.15;
2. Bench2Drive domain adaptation for the ReCogDrive 2B diffusion planner from cached VLM hidden states.

## Current Local Assets

- Official ReCogDrive VLM: `checkpoints/recogdrive/ReCogDrive-VLM-2B`.
- Official ReCogDrive 2B IL planner checkpoint:
  `checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt`.
- Bench2Drive VLM hidden cache:
  `outputs/bench2drive_recogdrive_vlm_cache_full_v1`.
- Cache shape: 8 shards, 38,934 records, 4 history frames, 8 future frames, frame step 5, sample stride 5.

## Official Training Evidence

The released ReCogDrive repository describes three stages:

- Stage 1: VLM driving pretraining on trajectory and QA data.
- Stage 2: diffusion planner imitation learning from cached VLM hidden states.
- Stage 3: DiffGRPO/RL using a metric cache and reference policy.

The public training scripts use InternVL, single camera, cached hidden states, a small DiT planner, DDIM sampling, and `agent.lr=1e-4`. The paper reports Stage II with a large global batch and long training schedule. The released code has NAVSIM metric-cache/RL scripts, but the Bench2Drive evaluation framework is still marked unreleased upstream, so a faithful B2D Stage 3 needs a local metric-cache analogue before it should be treated as official.

## Recommended Local Path

Run Stage 2 B2D adaptation first:

```bash
GPU_LIST=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/bench2drive/run_recogdrive_b2d_stage2_il_official.sh
```

Default effective batch with 8 GPUs is `32 * 8 * 2 = 512`. The launcher saves periodic step checkpoints every 1,000 micro-steps and the final `latest.ckpt`; use `SAVE_EVERY_EPOCH=1` only if epoch checkpoints are needed.

Useful smoke run:

```bash
GPU_LIST=0 NPROC_PER_NODE=1 MAX_SAMPLES=64 NUM_OPTIMIZER_STEPS=2 \
OUTPUT_DIR=outputs/bench2drive_recogdrive_stage2_il_official_smoke \
bash scripts/bench2drive/run_recogdrive_b2d_stage2_il_official.sh
```

The script validates that shards are real, indexed, and contain VLM hidden states before training.

## Optional Stage 1 Data

If we want to reproduce the likely paper setup more closely, download the Bench2Drive trajectory and QA pretraining JSONL from the ReCogDrive dataset:

```bash
LOCAL_DIR=/mnt/project/recogdrive_pretraining \
bash scripts/bench2drive/download_recogdrive_b2d_pretraining_jsonl.sh
```

The script unsets proxy environment variables so Hugging Face traffic goes directly through the container network.

## Evaluation After Training

Use the accelerated closed-loop runner for throughput comparison:

```bash
START_INFERENCE_SERVERS=1 TASK_NUM=8 \
TEAM_CONFIG=configs/bench2drive_recogdrive_closed_loop.remote.yaml \
CHECKPOINT=outputs/bench2drive_recogdrive_stage2_il_official_2b_<run>/best.ckpt \
bash scripts/bench2drive/run_recogdrive_closed_loop_multi.sh
```

Use the strict config for final metric runs if the accelerated visual cache/action-repeat settings show score drift:

```bash
TEAM_CONFIG=configs/bench2drive_recogdrive_closed_loop.strict.yaml
```
