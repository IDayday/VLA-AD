# ReCogDrive Bench2Drive Closed-Loop Evaluation

This wiring follows the public Bench2Drive/Bench2DriveZoo pattern used by UniAD/VAD, UniDriveVLA, DiffAD, and DriveMoE-style projects:

- run CARLA 0.9.15 closed-loop through the official Bench2Drive `leaderboard_evaluator.py`
- implement a `leaderboard.autoagents.autonomous_agent.AutonomousAgent`
- return `carla.VehicleControl` every 20 Hz tick
- save per-tick `metric_info.json` for efficiency and smoothness metrics
- run 220 routes, merge route JSON, then compute ability and efficiency/smoothness metrics

References checked while implementing:

- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- Bench2DriveZoo: https://github.com/Thinklab-SJTU/Bench2DriveZoo
- Bench2Drive-VL: https://github.com/Thinklab-SJTU/Bench2Drive-VL
- UniDriveVLA Bench2Drive docs: https://github.com/xiaomi-research/unidrivevla/blob/main/docs/train_eval_bench2drive.md
- DiffAD Bench2Drive docs: https://github.com/wantsu/DiffAD
- DriveMoE evaluation docs: https://github.com/Thinklab-SJTU/DriveMoE/blob/main/docs/evaluation.md

## Environment Split

Use two environments:

- `navsim`: existing ReCogDrive/VLM inference environment.
- `b2d_eval`: Python 3.8 CARLA/Bench2Drive client environment.

The split matches UniDriveVLA and Bench2Drive-VL practice: CARLA 0.9.15 is tied to Python 3.7/3.8, while modern VLM dependencies are easier to keep in Python 3.9+.

Create the eval environment:

```bash
conda env create -f configs/envs/b2d_eval.yaml
```

Install CARLA 0.9.15 and AdditionalMaps, then add the CARLA egg:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" \
  >> /root/miniconda3/envs/b2d_eval/lib/python3.8/site-packages/carla.pth
```

## Start ReCogDrive Inference Server

Run this in the existing `navsim` environment:

```bash
PLANNER_CHECKPOINT=/mnt/project/VLA-AD/outputs/bench2drive_recogdrive_vlm_il_train_full_epoch1/best.ckpt \
VLM_PATH=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
bash scripts/bench2drive/run_recogdrive_inference_server.sh
```

The server exposes:

- `GET /health`
- `POST /predict`

The CARLA agent sends front-camera JPEG path, 4-step history, command one-hot, and status vector. The server returns an 8x3 ReCogDrive trajectory.

The default remote config is the accelerated closed-loop path:

- `sensor_profile: front_only`, because the current ReCogDrive server consumes only `CAM_FRONT`
- `visual_refresh_interval_steps: 10`, so InternVL hidden states refresh at 2 Hz while planner/control still run at 20 Hz
- `image_dir: /dev/shm/recogdrive_b2d_images`, avoiding slow persistent image writes
- `save_debug_images: false` and `save_debug_meta: false`, while `metric_info.json` is still written for scoring

For a slow apples-to-apples baseline with the old full Bench2DriveZoo-style sensor suite and VLM refresh every tick:

```bash
export TEAM_CONFIG=/mnt/project/VLA-AD/configs/bench2drive_recogdrive_closed_loop.strict.yaml
```

For a speed ablation that also lowers planner frequency to 2 Hz and repeats the last control between requests:

```bash
export TEAM_CONFIG=/mnt/project/VLA-AD/configs/bench2drive_recogdrive_closed_loop.turbo.yaml
```

Treat `turbo.yaml` as an iteration tool until Dev10 confirms the score impact.

## Preflight

Run preflight inside `b2d_eval` after setting paths:

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export CARLA_ROOT=/path/to/CARLA_0.9.15
/root/miniconda3/bin/conda run -n b2d_eval python scripts/bench2drive/preflight_recogdrive_closed_loop.py \
  --bench2drive-root "$BENCH2DRIVE_ROOT" \
  --carla-root "$CARLA_ROOT" \
  --require-server
```

This checks Python version, CARLA files, Bench2Drive files, imports, route XML, agent config, and inference server health.

## Debug Route

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export CARLA_ROOT=/path/to/CARLA_0.9.15
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/run_recogdrive_closed_loop_debug.sh
```

## Full 220-Route Evaluation

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export CARLA_ROOT=/path/to/CARLA_0.9.15
export GPU_RANK_LIST="0 1 2 3 4 5 6 7"
export TASK_LIST="0 1 2 3 4 5 6 7"
export TASK_NUM=8
export START_INFERENCE_SERVERS=1
export SERVER_GPU_RANK_LIST="0 1 2 3 4 5 6 7"
export SERVER_STARTUP_SECONDS=90
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/run_recogdrive_closed_loop_multi.sh
```

`START_INFERENCE_SERVERS=1` starts one ReCogDrive HTTP server per worker port, waits once for model loading, health-checks each port, writes generated per-worker agent configs, and cleans those servers when the evaluation script exits. This avoids serializing all CARLA workers through a single VLM/planner server.

For final score reporting, run one short Dev10 comparison before launching all 220 routes:

```bash
TEAM_CONFIG=/mnt/project/VLA-AD/configs/bench2drive_recogdrive_closed_loop.strict.yaml \
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/run_recogdrive_closed_loop_debug.sh

TEAM_CONFIG=/mnt/project/VLA-AD/configs/bench2drive_recogdrive_closed_loop.remote.yaml \
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/run_recogdrive_closed_loop_debug.sh
```

If the accelerated config changes DS/SR materially on Dev10, use `strict.yaml` for publication-style reproduction and keep the accelerated config for iteration.

Then collect metrics:

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export ROUTE_JSON_DIR="$BENCH2DRIVE_ROOT/recogdrive_b2d_only_traj"
export METRIC_DIR=/mnt/project/VLA-AD/outputs/bench2drive_recogdrive_closed_loop/bench2drive220
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/collect_recogdrive_closed_loop_metrics.sh
```

## Weight Strategy

Do not treat the current one-epoch IL checkpoint as final. Public VLA projects report Bench2Drive closed-loop using Bench2Drive-specific training:

- UniDriveVLA reports staged B2D training and Stage 3 B2D weights.
- DiffAD reports B2D stage1/stage2 training before closed-loop.
- ReCogDrive README states additional Bench2Drive-Traj/QA adaptation after mixed-data training.

For credible reproduction, use this loop:

1. Run closed-loop debug on a small route set to validate wrapper correctness.
2. Evaluate official ReCogDrive IL/RL or our trained checkpoints with identical wrapper settings.
3. Retrain/fine-tune on the full Bench2Drive cache and, if available, Bench2Drive-Traj/QA data.
4. Re-run full 220-route closed-loop and compare DS/SR/Efficiency/Smoothness, not just open-loop L1.

Current B2D IL retraining wrapper:

```bash
GLOBAL_EPOCHS=4 \
BATCH_SIZE=16 \
OUTPUT_DIR=/mnt/project/VLA-AD/outputs/bench2drive_recogdrive_vlm_il_train_full_stage \
bash scripts/bench2drive/run_recogdrive_b2d_il_train.sh
```

Then point the inference server to the resulting checkpoint:

```bash
PLANNER_CHECKPOINT=/mnt/project/VLA-AD/outputs/bench2drive_recogdrive_vlm_il_train_full_stage/best.ckpt \
bash scripts/bench2drive/run_recogdrive_inference_server.sh
```

## Coordinate Notes

The wrapper keeps the same feature convention as `scripts/build_bench2drive_recogdrive_chunk_cache.py`:

- 4 history poses are current-frame relative `[x, y, compass]`
- command one-hot is `[left, straight, right]`
- status is `[cmd3, speed, 0, ax_local, ay_local, 0]`

The ReCogDrive trajectory is converted to Bench2DriveZoo PID waypoints by flipping the local y axis before control. This keeps the trained cache convention and the public Bench2DriveZoo PID convention aligned.

## Speed Notes

Do not speed up Bench2Drive by changing CARLA synchronous mode, fixed delta, route count, or metric collection semantics. CARLA documents synchronous fixed-step mode as the deterministic setup, and Bench2DriveZoo explicitly notes that efficiency/smoothness need per-step `metric_info` at 20 Hz. The safe optimization layer is inside the agent/server:

- remove sensors the model never reads
- cache or gate expensive VLM visual features
- keep planner/control updates at 20 Hz where possible
- avoid debug image/meta writes during scored runs
- run one inference server per active model GPU, or add request batching before increasing CARLA workers

The server prints average timing every `--profile-every` requests. Use this to decide whether the next bottleneck is VLM feature extraction, planner inference, CARLA rendering, or HTTP/image I/O.
