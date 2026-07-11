# ReCogDrive Bench2Drive Reproduction Gate

## Decision

The previous front-only Stage1/Stage2 run is retained only as a diagnostic
baseline. It is not an initialization for the new reproduction and its cache,
planner checkpoint, online prompt, and controller configuration are not
admissible artifacts for the main reproduction.

The reproduction now uses three explicit labels:

1. `paper-exact`: allowed only if the missing author Bench2Drive artifacts are
   obtained.
2. `closest-public`: an author-checkpoint-anchored reproduction using published
   data, paper settings, and clearly recorded proxy decisions.
3. `custom-aligned-extension`: the only valid label for Stage3 when its reward
   is designed locally rather than obtained from the authors.

Passing a gate means the label and evidence are valid. It does not silently
upgrade a public proxy or custom extension into a paper-exact reproduction.

## Pinned provenance

The machine-readable source of truth is
`configs/bench2drive_recogdrive_reproduction_gate.json`.

| Artifact | Pinned value | Current status |
|---|---|---|
| ReCogDrive paper | arXiv `2506.08052v2` | recorded |
| Public ReCogDrive source | `d54404796de7a44ca418b96057e3f8c3de3e8c0d` | recorded; B2D code absent |
| ReCogDrive pretraining dataset | HF revision `f55bb18e0aca846bbedfb516760a1b9b7cfe3ebf` | local files verified |
| ReCogDrive-VLM-2B | HF revision `16873acca08e3c04ab229b3d973f39aeba9db68d` | local weight verified |
| Bench2Drive checkout | `2645714eb1f3a100217928dd113093cae0779f36` | clean proxy checkout under `/mnt/data` |
| CARLA | `0.9.15` | verified |
| Route suite | `bench2drive220.xml`, 220 routes | verified |

The clean Bench2Drive checkout is `/mnt/data/Bench2Drive`. The checkout under
`/mnt/project/Bench2Drive` contains local CARLA launch changes and generated
evaluation files, so the reproduction gate intentionally rejects it as the
formal scoring source.

The official local data audit establishes:

- `Bench2drive_Traj.jsonl`: 196,761 records, 950 clips, six images per record;
- `Bench2drive_QA.jsonl`: 49,942 multi-turn records, 1,000 clips, six images per
  record and 2,878,608 human/GPT turn pairs;
- the trajectory clip set is a strict subset of the QA/raw-data clip set, with
  exactly 50 clips in the complement;
- the six camera streams are front, front-left, front-right, back-left,
  back-right, and back.

## Gate status by stage

### Stage1: ready as closest-public

Stage1 may be rebuilt after the full artifact audit passes. It must:

- initialize from the pinned official `ReCogDrive-VLM-2B` checkpoint;
- jointly train the unmodified official Bench2Drive Traj and QA JSONL records;
- preserve all six camera views and their original prompt/answer format;
- use the author-described `1x2 patches + thumbnail` preprocessing per view;
- use the paper-wide full-parameter three-epoch SFT settings as the declared
  public proxy: batch 1,024, LR `4e-5`, weight decay 0.05, 10% warmup, cosine
  schedule, 448 resolution, sequence length 12,288, and drop path 0.1.

The exact B2D-only data mixing ratio and whether its adaptation changed any
paper-wide Stage1 hyperparameter remain unknown. These are disclosed proxy
choices, not silently filled official values.

The approved public proxy uses the natural record frequency of the two
unmodified JSONLs (`repeat_time=1` for each and no weighted resampling). The
loader receives the public SFT launcher's `max_dynamic_patch=16`; for six images
it assigns at most two dynamic patches plus one thumbnail to each view,
producing 18 image tiles per sample as described by the authors.

The training software is also locked. The vendored InternVL pins Transformers
4.37.2, Tokenizers 0.15.1, DeepSpeed 0.13.5, and several preprocessing
dependencies. A dedicated Python 3.9 environment uses Torch 2.2.2/CUDA 12.1,
torchvision 0.17.2, Accelerate 0.28.0, and FlashAttention 2.5.8 as explicit
compatibility choices. Those four versions are not author-confirmed. Before
loading data, the launcher records every version and executes a real BF16
FlashAttention CUDA forward pass on the visible devices.
The Python 3.9/Torch 2.2 FlashAttention wheel is additionally pinned by SHA-256
in the manifest and verified before installation.

The dedicated environment can be rebuilt without changing or accepting terms
for an existing Conda installation:

```bash
REBUILD_STAGE1_ENV=1 \
bash scripts/bench2drive/create_recogdrive_b2d_stage1_env.sh
```

The public shell script launches 8 nodes x 8 GPUs and accumulates 16 microsteps.
This machine has one 8-GPU node, so the local launcher accumulates 128
microsteps to preserve the effective global batch of 1,024. This hardware
topology adaptation is recorded explicitly; it is not presented as identical
floating-point execution.

Run the real multi-view two-step smoke before a formal launch:

```bash
RUN_MODE=smoke \
TRAIN_PYTHON=/mnt/project/recogdrive_stage1_env/bin/python \
bash scripts/bench2drive/run_recogdrive_b2d_stage1_official_multiview_sft.sh
```

The formal launcher enforces the declared proxy hyperparameters and refuses
unlabeled changes:

```bash
RUN_MODE=formal \
TRAIN_PYTHON=/mnt/project/recogdrive_stage1_env/bin/python \
bash scripts/bench2drive/run_recogdrive_b2d_stage1_official_multiview_sft.sh
```

Both modes execute the reproduction gate and load real trajectory, ordinary
QA, and maximum-length QA samples through the InternVL tokenizer/image pipeline
before allocating the full model. The old front-only launcher is not called.

### Stage2: ready as closest-public

The released trajectory JSONL and matched raw annotations close the previously
ambiguous data contract sufficiently for a declared public proxy:

- all 1,000 raw clips are fitting data, as explicitly selected for this run;
- training anchors retain every consecutive raw frame (10 Hz), matching the
  released Traj row cadence; history and future waypoints use five-frame
  spacing (0.5 seconds);
- history is four poses at 0.5-second intervals;
- the target is six poses at 0.5-second intervals (3 seconds total), matching
  the released trajectory rows; same-team UniDriveVLA independently uses a
  six-pose B2D action horizon;
- all six views and the released Stage1 prompt are reused byte-for-byte;
- hidden caching keeps every non-padding final-layer token and stores BF16;
- planner actions are `[LIDAR-forward, LIDAR-lateral, relative-yaw]`, computed
  from the raw 4x4 `world2lidar` matrices. The textual Stage1 answer swaps the
  first two axes and adds a pi/2 heading offset, so it is not used directly as
  the DiT action target;
- the DiT starts randomly, trains 200 epochs with effective batch 512, and the
  epoch-200 checkpoint is the formal candidate.

The 1,000 clips yield exactly 202,656 windows. Eight windows rejected by the old
pipeline had NaN scalar `theta`, but their official `world2lidar` matrices are
finite and the released Stage1 JSONL includes those frames. The corrected
matrix-based cache therefore retains them. The launcher refuses any other
count or contract identifier.

Build the cache and start Stage2 manually:

```bash
VLM_PATH=/path/to/completed/stage1 \
OUTPUT_ROOT=outputs/bench2drive_recogdrive_stage2_cache_<run> \
bash scripts/bench2drive/build_recogdrive_b2d_stage2_cache.sh

CHUNK_CACHE_ROOT=outputs/bench2drive_recogdrive_stage2_cache_<run>/train \
EXPECTED_VLM_PATH=/path/to/completed/stage1 \
bash scripts/bench2drive/run_recogdrive_b2d_stage2_closest_public.sh
```

Or use `watch_recogdrive_b2d_stage1_then_stage2.sh` to wait for the final
Stage1 checkpoint, build/validate the full cache, and launch scratch Stage2
exactly once.

### Stage3: blocked from the current baseline

The paper and public code define DiffGRPO rollouts with the NAVSIM non-reactive
PDM simulator and scalar NAVSIM PDMS. They do not disclose a Bench2Drive reward,
state whether Table 2 used Stage3, or explain how public 8-pose NAVSIM GRPO
connects to the released six-pose B2D contract.

Therefore this baseline does not invent a B2D scalar reward and does not add
Pareto optimization. The checked-in public NAVSIM DiffGRPO path remains
unchanged. The corrected Stage2 baseline is trained and evaluated first. A
later custom reward or migration of our research method must be a separate,
explicitly labeled experiment initialized from the reproduced baseline.

### Evaluation: ready as closest-public

The public proxy wrapper now uses the official Bench2Drive six-camera zoo
calibration, the same multi-view prompt as caching, four 0.5-second history
poses, a six-pose planner output, and the repository's trajectory PID. Control
runs at 20 Hz while VLM/planner inference runs every second tick (10 Hz),
matching the released consecutive-anchor cadence without reusing a prompt
across more than one intermediate control tick.

The exact private ReCogDrive controller, inference frequency, evaluator commit,
seed policy and raw route JSON remain unknown. Those limitations prevent a
paper-exact label but no longer justify using the retired front-only wrapper.
The complete 220 routes are final evaluation only and are scored by the
official Bench2Drive evaluator.

## Running the gate

Full provenance and Stage1 readiness audit:

```bash
python scripts/bench2drive/check_recogdrive_b2d_reproduction_gate.py \
  --target stage1 \
  --full-hash \
  --report outputs/bench2drive_recogdrive_reproduction_gate/stage1_gate.json
```

The optional image-completeness audit stats every referenced image and is much
slower:

```bash
python scripts/bench2drive/check_recogdrive_b2d_reproduction_gate.py \
  --target artifacts \
  --full-hash \
  --verify-all-images
```

Stage1, Stage2 and evaluation return zero when their closest-public contracts
and artifacts pass. Stage3 intentionally returns exit code 2. Artifact or
provenance failure returns exit code 1.
