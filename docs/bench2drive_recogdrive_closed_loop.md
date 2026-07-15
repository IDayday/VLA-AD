# ReCogDrive Bench2Drive Closed-Loop Evaluation

> Updated 2026-07-11: the formal baseline is the six-view, six-pose
> `closest-public` contract. The older front-only/8-pose profiles are retained
> only for historical diagnostics and are not admissible reproduction results.

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
PLANNER_CHECKPOINT=/path/to/stage2/epoch_200.ckpt \
VLM_PATH=/path/to/completed/stage1 \
bash scripts/bench2drive/run_recogdrive_inference_server.sh
```

The server exposes:

- `GET /health`
- `POST /predict`

The CARLA agent sends six ordered camera JPEG paths, four history poses, exact
command id, speed, acceleration and planner status. The server rebuilds the
same released Stage1 prompt used for cache generation and returns a 6x3
trajectory. Temporary JPEGs are removed synchronously after each response.

The formal config is
`configs/bench2drive_recogdrive_closed_loop.closest_public.yaml`: official
six-camera zoo calibration, 20 Hz control with 10 Hz VLM/planner refresh, four
history poses at 0.5-second spacing and six planner poses. The front-only
`remote.yaml` and `turbo.yaml` profiles are speed diagnostics only.

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

For the local 8x80 GB host, use the checked launcher.  Its measured stable
default is 16 workers (two independent CARLA/server stacks per GPU), with
round-robin GPU launch order, four CPU threads per inference server, workload-
balanced route shards, health-driven model startup, and a stalled-worker
watchdog:

```bash
PLANNER_CHECKPOINT=/path/to/stage2/latest.ckpt \
VLM_PATH=/path/to/completed/stage1 \
bash scripts/bench2drive/launch_local_b2d220_eval.sh
```

When the evaluator is run by root, the local launcher automatically selects
`/mnt/project/CARLA_0.9.15_nonroot`.  That root contains a `CarlaUE4.sh`
wrapper which drops only the Unreal process to a non-root UID; the Python
evaluator remains in `b2d_eval` as root.

The lower-level equivalent remains available for other host layouts:

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export CARLA_ROOT=/path/to/CARLA_0.9.15
export GPU_RANK_LIST="0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7"
export TASK_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
export TASK_NUM=16
export START_INFERENCE_SERVERS=1
export SERVER_GPU_RANK_LIST="${GPU_RANK_LIST}"
export SERVER_STARTUP_SECONDS=300  # maximum; health checks normally finish earlier
export WORKER_START_DELAY=3
export WORKER_STALL_TIMEOUT=240
export SPLIT_STRATEGY=balanced
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/run_recogdrive_closed_loop_multi.sh
```

`START_INFERENCE_SERVERS=1` starts one ReCogDrive HTTP server per worker port, waits once for model loading, health-checks each port, writes generated per-worker agent configs, and cleans those servers when the evaluation script exits. This avoids serializing all CARLA workers through a single VLM/planner server.

Before all 220 routes, run one short debug route with the fixed formal config to
verify sensor delivery, coordinate signs and PID stability:

```bash
TEAM_CONFIG=/mnt/project/VLA-AD/configs/bench2drive_recogdrive_closed_loop.closest_public.yaml \
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/run_recogdrive_closed_loop_debug.sh
```

Then collect metrics:

```bash
export BENCH2DRIVE_ROOT=/path/to/Bench2Drive
export ROUTE_JSON_DIR="$BENCH2DRIVE_ROOT/recogdrive_b2d_only_traj"
export METRIC_DIR=/mnt/project/VLA-AD/outputs/bench2drive_recogdrive_closed_loop/bench2drive220
/root/miniconda3/bin/conda run --no-capture-output -n b2d_eval \
  bash scripts/bench2drive/collect_recogdrive_closed_loop_metrics.sh
```

## Weight Strategy

Only the epoch-200 checkpoint from
`run_recogdrive_b2d_stage2_closest_public.sh`, paired with the exact Stage1 VLM
that generated its cache, is admitted to the formal baseline. Do not substitute
the released NAVSIM IL/RL planner, the retired front-only B2D checkpoint, a
loss-selected checkpoint, or any research branch.

## Coordinate Notes

The wrapper keeps the same feature convention as
`scripts/build_bench2drive_recogdrive_chunk_cache.py`:

- raw cache history/targets use exact `world2lidar` matrices;
- live history uses the equivalent planar Bench2Drive compass convention;
- 4 history poses are current-frame relative `[forward, lateral, yaw]`;
- command one-hot is `[left, straight, right]`
- status is `[cmd3, speed, 0, acceleration_x, acceleration_y, 0]`;
- output is 6 poses in `[forward, lateral, relative_yaw]`.

The ReCogDrive trajectory is converted to Bench2DriveZoo PID waypoints by flipping the local y axis before control. This keeps the trained cache convention and the public Bench2DriveZoo PID convention aligned.

## Speed Notes

Do not speed up Bench2Drive by changing CARLA synchronous mode, fixed delta, route count, or metric collection semantics. CARLA documents synchronous fixed-step mode as the deterministic setup, and Bench2DriveZoo explicitly notes that efficiency/smoothness need per-step `metric_info` at 20 Hz. The safe optimization layer is inside the agent/server:

- retain all six formal camera sensors
- do not cache a prompt across changed speed/acceleration/command state
- keep VLM/planner/control updates at the fixed formal frequency
- avoid debug image/meta writes during scored runs
- use one inference server per active worker; sharing the current threaded predictor between workers is unsafe without explicit request batching and cache isolation

The server prints average timing every `--profile-every` requests. Use this to decide whether the next bottleneck is VLM feature extraction, planner inference, CARLA rendering, or HTTP/image I/O.

### Measured local optimum (2026-07-15)

All rows below kept the formal six-camera `closest-public` contract unchanged:
20 Hz synchronous CARLA, 10 Hz inference, visual refresh every two inference
steps, four history poses, horizon 6, DDIM 5, BF16, all 220 routes, and normal
metric writes.

| layout | CPU threads/server | aggregate sim/wall ratio | server mean request | stability | decision |
|---|---:|---:|---:|---|---|
| 8 workers, one/GPU, contiguous split | unconstrained | 0.356 | 1.450 s | stable but slow | reject |
| 16 workers, two/GPU | 4 | 1.154-1.174 | 0.726 s | no probe crashes | **formal default** |
| 24 workers, three/GPU | 4 | 1.21-1.43 projected | not retained | repeated CARLA RenderThread timeouts in both grouped and round-robin probes | reject |

The 16-worker layout delivered about 10.75 successful inference requests/s and
3.2-3.3x the aggregate simulation throughput of the original 8-worker run.
The main gain was not merely adding workers: unconstrained PyTorch/OpenMP
created 70-134 threads per server and severe preprocessing contention.  Capping
each server at four threads reduced the cumulative `visual_features` timing
from 1.341 s/request to 0.582 s/request.  At 16 workers the host used roughly
30-32 GB of each 80 GB GPU and about 31% aggregate CPU, leaving safety margin
for route-dependent peaks.

Probe artifacts:

- 8-worker baseline: `outputs/bench2drive_recogdrive_closed_loop/local_final_220route_20260715T005000Z`
- stable 16-worker probe: `outputs/bench2drive_recogdrive_closed_loop/local_final_220route_16shard_probe_20260715T012440Z`
- rejected 24-worker probes: `local_final_220route_24shard_probe_20260715T014108Z` and `local_final_220route_24shard_rr_probe_20260715T014937Z`

The official contiguous splitter balances only route count.  On these 220
routes, a 16-way contiguous split had 776-2466 historical simulation seconds
per shard.  `split_xml_balanced.py` optionally reads a prior complete route
JSON and uses only `meta.duration_game` as a scheduling cost, reducing that
range to 1546-1568 seconds.  The prior JSON is copied into each new output
directory for provenance.  It never contributes records, scores, reward,
checkpoint selection, or final metric aggregation; if it is unavailable, the
splitter falls back to geometric route length.

Using the historical total of about 24,987 simulated seconds and the stable
16-worker ratio gives a planning estimate of roughly 6 hours of active rollout,
or approximately 6-7 hours including CARLA setup, route transitions, and any
automatic retry.  Treat this as an ETA, not an evaluation result.
