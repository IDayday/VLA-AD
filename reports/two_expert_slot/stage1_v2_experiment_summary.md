# Stage1-v2 Experiment Summary

## Scope

This round rebuilt two-expert slot Stage1-v2 only. It did not start Stage2,
did not start Stage3, did not enable residual diffusion, did not restore
action-side CoT, and did not add an A4 direct expert path.

Token counts stayed unchanged:

- `H_dyn = 3 groups x 12 tokens`
- `H_geo = 12 tokens`
- JEPA dynamic teacher: `[3, 12, 1024]`
- VGGT Feature(23) teacher: `[12, D]`

## Code State

- branch: `feature/recogdrive-last-vla-v2`
- base HEAD before this report: `160ffb42f3e03800f20455414462e2624dafec9d`
- user-visible run root: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154`

The working tree also contained unrelated Stage3 changes. The Stage1-v2 commit
is intentionally scoped to Stage1 replay, slot losses, Stage1 evaluation/gate
helpers, configs, tests, and this report.

## Replay Data

The replay source is NAVSIM trajectory JSON data, not a fabricated fallback:

- source label: `official_navsim_traj`
- replay cache: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/data/stage1_replay_cache`
- image root resolved through replay samples: `/mnt/navsim`
- written replay samples: `84,918`
- unmatched JSON rows: `191`
- parse failures: `0`
- JEPA/VGGT/replay three-way intersection: `84,918`
- readiness gate: `READY`

Some replay payloads do not carry `history_trajectory`, `high_command_one_hot`,
or `status_feature` tensors. The dataset now recovers these from the replay
prompt when needed:

- `history_trajectory`: parsed from the last four `(x, y, heading)` triples
- `high_command_one_hot`: parsed from `[TURN LEFT]`, `[GO STRAIGHT]`, or `[TURN RIGHT]`
- `status_feature`: true tensor if present; otherwise derived as
  `[command_one_hot(3), velocity_xy(2), acceleration_xy_heading(3)]`

## Stage1-v2 Algorithm Changes

- Replay CE is trained through normal online VLM forward, with prompt tokens
  masked out of the labels.
- Replay CE does not insert soft slots and is intended to update VLM LoRA.
- Slot-only reconstruction is strict by default: no image memory is passed to
  the dynamic or geometry adapter.
- Adapter image memory still exists for the normal reconstruction path.
- Image masking and image-hidden dropout are scheduled rather than applied at
  full strength from step 0.
- Contrastive losses use smaller final weights and warm up over the early run.
- Branch trajectory probes include fused, dyn-only, and geo-only heads.
- Hidden anchor remains enabled on a sparse schedule.
- DDP contrastive gathers target features across ranks, so per-rank batch size
  `1` still gives a global batch for retrieval pressure.

## Clean Training Run

Output directory:

`/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/stage1_clean/clean_replay_8gpu_local_warmup_status_20260616T013758Z`

Launch summary:

- launcher PID: `4070625`
- start: `2026-06-16T01:37:58Z`
- `stage1_start`: `2026-06-16T01:38:53Z`
- complete: `2026-06-16T10:03:52Z`
- wall time from `stage1_start`: about `8h25m`
- world size: `8`
- batch size per GPU: `1`
- grad accumulation: `8`
- effective batch size: `64`
- max epochs: `3`
- total forward steps per rank: `31,845`
- optimizer steps: `3,981`
- precision: `bf16-mixed`

Key hyperparameters:

- `lr_vlm = 5e-6`
- `lr_slots_adapters = 1e-4`
- `stage1_image_mask_ratio_start = 0.4`
- `stage1_image_mask_ratio = 0.7`
- `stage1_image_dropout_prob_start = 0.05`
- `stage1_image_dropout_prob = 0.25`
- `contrastive_dyn_loss_weight = 0.02`
- `contrastive_geo_loss_weight = 0.02`
- `recogdrive_replay_ce_loss_weight = 0.1`
- `recogdrive_replay_every_n_steps = 2`
- `hidden_anchor_weight = 0.03`
- `hidden_anchor_every_n_steps = 4`
- `slot_only_use_image_memory = false`
- `allow_replay_only_base = true`

Artifacts:

- checkpoint: `stage1.ckpt` (`54M`)
- LoRA/adapter directory: `adapters/`
- metadata: `stage1_metadata.json`
- metrics: `metrics.json`
- compact progress log: `train_progress.jsonl`
- command log: `commands.log`

Completion metrics:

| metric | final logged value |
| --- | ---: |
| `loss` | `0.0483398` |
| `dyn_loss` | `0.000839233` |
| `geo_loss` | `0.000404358` |
| `slot_only_dyn_loss` | `0.000724792` |
| `slot_only_geo_loss` | `0.0000858307` |
| `probe_loss` | `0.000036478` |
| `probe_dyn_loss` | `0.0000498295` |
| `probe_geo_loss` | `0.0000371933` |
| `recogdrive_replay_ce_loss` | `0.215820` |
| `recogdrive_replay_ce_loss_weight_effective` | `0.100098` |
| `contrastive_dyn_loss` | `0.406250` |
| `contrastive_geo_loss` | `0.855469` |
| `contrastive_*_loss_weight_effective` | `0.0200195` |
| `stage1_effective_image_mask_ratio` | `0.699219` |
| `stage1_effective_image_dropout_prob` | `0.25` |

Last-100 logged-step averages:

| metric | last-100 mean |
| --- | ---: |
| `loss` | `0.0493921` |
| `dyn_loss` | `0.000878563` |
| `geo_loss` | `0.000385761` |
| `slot_only_dyn_loss` | `0.000749588` |
| `slot_only_geo_loss` | `0.0000620604` |
| `probe_loss` | `0.000555811` |
| `probe_dyn_loss` | `0.000565990` |
| `probe_geo_loss` | `0.000552462` |
| `recogdrive_replay_ce_loss` | `0.132646` |
| `contrastive_dyn_loss` | `0.513861` |
| `contrastive_geo_loss` | `1.20969` |
| `hidden_anchor_loss` | `0.00124786` |

## Interpretation

The training run completed normally and produced a compact Stage1 checkpoint.
The teacher reconstruction losses stayed small through the run. Geometry
slot-only reconstruction improved sharply relative to the first logged steps,
which is the desired direction because previous Stage1 was too dependent on
image-hidden shortcut features.

This training summary alone is not a Stage1 representation gate pass. The next
decision point is to run `evaluate_stage1_two_expert_ckpt.py` on a held-out
subset and then `check_stage1_v2_representation_gate.py`. Hidden-cache
generation and Stage2 should remain blocked until the Stage1-v2 gate is
explicitly `READY`.

## Validation

Commit-time checks:

- `python -m py_compile` on Stage1 replay, adapter, SFT, data discovery,
  replay preparation, audit, evaluation, and gate scripts: passed
- `bash -n` on clean, continue, and wait launchers: passed
- `pytest -q tests/test_recogdrive_stage1_replay_text.py tests/test_stage1_replay_dataset.py tests/test_two_expert_*.py tests/test_no_future_leakage.py`: `60 passed`

## Current Recommendation

Keep this checkpoint as the clean Stage1-v2 candidate:

`/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/stage1_clean/clean_replay_8gpu_local_warmup_status_20260616T013758Z/stage1.ckpt`

Do not build full hidden cache or start Stage2 until Stage1 representation
evaluation confirms that slot-only, image-only, retrieval, replay-direct, and
hidden-drift checks are acceptable.
