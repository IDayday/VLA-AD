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

### Stage2: blocked pending contract closure

The main Stage2 reproduction must not start until the following are resolved or
explicitly approved as public proxies:

- 950/50 versus all-1,000 raw clips;
- raw frame cadence and downsampling;
- history duration and interval;
- six versus eight future waypoints and the prediction horizon;
- planner coordinate/heading convention;
- multi-view hidden-state prompt and token selection;
- checkpoint selection rule.

The architecture and headline optimizer settings are known and recorded in the
manifest, but they do not resolve the B2D data contract.

### Stage3: custom-aligned extension

The official B2D reward is unavailable. A local reward is authorized, but the
stage must be called `custom-aligned-extension`, not an official Stage3
reproduction. Training/evaluation consistency means sharing semantics, units,
thresholds, coordinate transforms, horizon, and component direction. It does
not mean replacing final evaluation with the training reward.

The reward must be versioned as `bench2drive_diffgrpo_reward_v1` and must expose
separate components for at least:

- official-style route progress/completion;
- collisions and drivable-area/outside-lane compliance;
- traffic lights, stop signs, and emergency-vehicle yielding;
- blocked/minimum-speed behavior and bounded useful progress;
- command/route adherence;
- comfort based on the same acceleration/jerk/yaw semantics and 20 Hz ego-state
  sampling used by the official evaluation.

Safety-critical penalties should retain the official multiplicative/hard-gate
semantics, while dense shaping may be added to prevent zero-advantage groups.
Every dense term must be logged separately so reward gaming can be identified.

Before Stage3 training, the reward gate requires:

1. unit tests showing monotonic ordering for controlled safe/unsafe candidate
   trajectories;
2. component-level golden cases for collision, off-road, traffic-law, blocked,
   progress, and comfort events;
3. shadow scoring of frozen validation rollouts using both the custom reward and
   the official Bench2Drive evaluator;
4. a calibration report produced without tuning on the complete 220-route
   result;
5. a frozen reward configuration and source commit recorded in every Stage3
   checkpoint.

Final model selection may use only training/validation evidence. The complete
220-route suite remains a final evaluation and is always scored with the
official Bench2Drive tools, not the custom training reward.

### Evaluation: infrastructure available, ReCogDrive wrapper unresolved

The 220 routes, CARLA version, and official metric tools are available. The
reported ReCogDrive sensor/controller contract is not. A formal final run still
requires closure of multi-view sensor calibration, VLM/planner/control
frequency, trajectory-to-control conversion, PID parameters, evaluator commit,
and seed/repetition policy.

The current front-only 2 Hz visual-refresh HTTP wrapper cannot be used for the
formal reproduction.

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

Requests for Stage2, Stage3, or evaluation return exit code 2 while their
contract is blocked. Artifact corruption or provenance mismatch returns exit
code 1. A ready target returns exit code 0.
