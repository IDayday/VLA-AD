# Bench2Drive ReCogDrive Training Plan

## Decision and baseline

The previous Bench2Drive run is a retired diagnostic baseline, not a result to
continue training from. Its Stage2 policy was initialized from the released
ReCogDrive IL checkpoint and its hidden states came from the VLM before the new
Bench2Drive-specific Stage1. The interrupted 220-route run reached 129 routes;
at 111 completed routes the aggregate was DS 44.96, RC 80.0%, and strict
success 14.4%, well below the reported ReCogDrive result.

The old 625 GB hidden-state cache
`outputs/bench2drive_recogdrive_vlm_cache_full_v1` was deleted on 2026-07-10.
Stage2 launchers now reject caches sourced from the pre-Stage1 base VLM unless
an explicit diagnostic override is supplied.

The new pipeline is:

1. Stage1: full VLM SFT on Bench2Drive with the exact online prompt/trajectory contract.
2. Rebuild disjoint train and validation hidden-state caches from that Stage1 VLM.
3. Stage2: train the DiT policy from random initialization.
4. Evaluate Stage2 on the held-out cache and the complete 220-route closed-loop suite.
5. Stage3: train with a Bench2Drive GRPO reward/cache implementation, initialized from the accepted Stage2 model.
6. Evaluate the final model on the same complete 220-route suite.

## Stage1 data contract

The released Bench2Drive trajectory JSONL uses six cameras, no pose history,
six output points, and a different textual coordinate convention. Closed-loop
serving in this repository uses one front image, four relative ego poses, a
three-way navigation command, and eight future relative poses. Mixing those
contracts would train one prompt and cache another.

`prepare_recogdrive_b2d_stage1_sft.py` derives the SFT rows from the raw B2D
annotations with the online contract. The deterministic split is by whole
clip, so no clip appears in both training and validation.

Current generated manifest:

- 1,000 clips: 950 train / 50 validation;
- 36,851 train samples / 2,083 validation samples;
- 8 invalid windows excluded because their pose history contained non-finite values;
- train JSONL SHA256 `1845e18b610a3da1473e022f1dc6a42899752f051693216e6c89b3df25f1c6d1`;
- validation JSONL SHA256 `9cfed73d9f2e590c6aabce77a37701a4e4a0dd1ff71e2a7467eb8912d845e600`.

The local machine does not have a pristine InternVL3-2B base checkpoint. The
Stage1 default therefore starts from the released `ReCogDrive-VLM-2B` and does
a Bench2Drive-specific, online-contract-aligned full fine-tune. This is a
domain adaptation experiment, not a claim that the complete upstream
multi-dataset Stage1 was reproduced from the original foundation model.

## Stage1 launch

The full-model smoke tests passed with 2,088,982,016 trainable parameters,
BF16, and nine dynamic image patches. A two-step eight-GPU DDP run reduced loss
from 1.5496 to 1.3587 with finite gradient norms (19.75 and 15.94) and no OOM.
The default effective batch is 64 on eight GPUs, learning rate is 1e-5, and the
schedule is three epochs.

```bash
GPU_LIST=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/bench2drive/run_recogdrive_b2d_stage1_sft.sh
```

Every run writes the source checkpoint, git commit, data path, effective batch,
and optimizer settings to `launch_env.txt`. The eight-GPU smoke measured about
1.1 seconds per micro-step after startup, giving an initial 4-6 hour estimate;
use the first 100 full-run steps to replace that estimate with measured throughput.

## Stage1 result and gate

The formal run `bench2drive_recogdrive_stage1_sft_20260710T065201Z` completed
all three epochs (1,728 optimizer steps) in 1:58:58 with aggregate train loss
0.75462. On all 2,083 examples from the 50 held-out clips, token NLL decreased
from 1.54483 for the released VLM to 0.72040 for Stage1, a 53.37% reduction.

Greedy generation on a deterministic 256-example, all-clip sample reduced ADE
from 13.5930 m to 3.6443 m and FDE from 24.7505 m to 7.1366 m. On all 118 turn
examples, ADE decreased from 14.5207 m to 5.4953 m and FDE from 26.8010 m to
10.8198 m. Clip-level paired bootstrap intervals exclude zero improvement, and
shuffled-image, shuffled-command, and shuffled-answer controls show that the
fine-tuned model uses the paired inputs rather than only memorizing the output
syntax.

Stage1 therefore passes the gate for rebuilding hidden states and starting the
scratch Stage2 run. It does not pass a final-planner gate, and a 220-route run is
not meaningful until Stage2 is trained. See
`docs/bench2drive_stage1_evaluation_20260710.md` for the full protocol, metrics,
confidence intervals, limitations, and artifact paths.

## Rebuild Stage2 caches

Only a completed Stage1 output may be supplied as `VLM_PATH`:

```bash
VLM_PATH=outputs/bench2drive_recogdrive_stage1_sft_<run> \
GPU_LIST=0,1,2,3,4,5,6,7 \
OUTPUT_ROOT=outputs/bench2drive_recogdrive_stage2_cache_<run> \
bash scripts/bench2drive/build_recogdrive_b2d_stage2_cache.sh
```

This creates `train/shard_*` and `val/shard_*` from the exact Stage1 clip
lists. Cache metadata records the Stage1 checkpoint and the `bench2drive`
system-prompt profile. The old pre-Stage1 cache must not be restored or reused.

If the experimental decision is to use all 1,000 clips for Stage2 fitting,
materialize the union of the Stage1 train/validation clip lists and pass it as
`TRAIN_CLIP_LIST`, with `SPLITS=train` and `EXPECTED_TRAIN_CLIPS=1000`. This is
an explicit full-data fit: the former 50-clip validation split is then
in-sample and must not be reported as held-out Stage2 performance. Use the
complete 220-route suite, or a separately defined clip split, for model
selection and final reporting.

## Stage2 scratch training

Stage2 defaults to a random DiT/action-head initialization. The cache source is
required rather than silently defaulting to an old artifact.

```bash
CHUNK_CACHE_ROOT=outputs/bench2drive_recogdrive_stage2_cache_<run>/train \
EXPECTED_VLM_PATH=outputs/bench2drive_recogdrive_stage1_sft_<run> \
GPU_LIST=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/bench2drive/run_recogdrive_b2d_stage2_il_official.sh
```

The default Stage2 schedule remains 200 global epochs, effective batch 512,
action-head learning rate 1e-4, three warmup epochs, and cosine decay. The
launcher refuses dummy features, non-VLM features, NAVSIM prompt caches,
mixed-source shards, and caches generated from the pre-Stage1 base VLM.

## Stage2 evaluation gate

First run the cached open-loop evaluator on the held-out `val/shard_*` cache
and report L1, ADE, FDE, and heading error. Then run all 220 closed-loop routes
with the Stage1 VLM and Stage2 checkpoint. Do not start Stage3 from a partial
route subset. Record route completion count, DS, RC, strict success, infractions,
and scenario-family breakdown.

Stage3 is accepted only if its Stage2 initialization and evaluation artifacts
can be traced to the same Stage1 checkpoint and clip manifest.

## Stage3 GRPO

The checked-in GRPO path currently computes NAVSIM/PDM rewards. It is not a
faithful Bench2Drive reward implementation and must not be relabeled as B2D
GRPO. After the Stage2 gate, implement a B2D metric cache/reward adapter using
the official route criteria (progress, infraction penalties, success, and
comfort), freeze a Stage2 reference policy, and validate reward parity on a
small route set before the full GRPO run. The final checkpoint is then evaluated
on the complete 220-route suite with the same strict serving contract.
