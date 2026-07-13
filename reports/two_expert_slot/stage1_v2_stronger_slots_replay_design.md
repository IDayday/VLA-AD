# Stage1-v2 Stronger Slots + ReCogDrive Replay Design

Goal: improve two-expert slot Stage1 without changing token counts:

- `H_dyn = 3 x 12` soft slots
- `H_geo = 12` soft slots
- JEPA dynamic teacher shape `[3,12,1024]`
- VGGT Feature(23) teacher shape `[12,D]`

This round does not start Stage2/Stage3, does not enable residual diffusion,
does not restore action-side CoT, and does not use an A4 direct expert path.

Implemented changes:

- Official NAVSIM-Traj replay CE path, with prompt tokens masked to `-100`.
- Replay CE uses normal online VLM forward without inserting soft slots.
- Replay CE requires trainable VLM mode and is intended to update VLM LoRA.
- Stronger image bottleneck via configurable image mask ratio and image-hidden dropout.
- Slot-only dynamic and geometry reconstruction losses.
- Batch contrastive retrieval losses for dynamic and geometry teacher features.
- In DDP, contrastive uses all-gathered global target features, so the loss remains active even when per-rank batch size is `1`.
- Branch trajectory probes: fused, dyn-only, and geo-only.
- Geo lateral/heading profile losses.
- Hidden-anchor scheduling and online LoRA-disabled raw hidden comparison when PEFT supports it.
- Replay-only base mode for the case where old base chunks were deleted; official replay cache supplies image/prompt/trajectory while JEPA/VGGT caches supply teacher tokens.
- Replay-only context recovery: when `history_trajectory`, `high_command_one_hot`,
  or `status_feature` are missing from NAVSIM replay payloads, they are recovered
  from the replay prompt and recent motion rather than filled with all zeros.
- Scheduled regularization: image mask ramps from `0.4` to `0.7`, image-hidden
  dropout ramps from `0.05` to `0.25`, contrastive losses ramp to `0.02`, and
  replay CE ramps to `0.1` every 2 steps.
- Slot-only reconstruction defaults to strict slot-only memory, without image
  memory, so this diagnostic no longer measures an image shortcut path.

Validation added:

- trajectory answer formatter/parser and label masking tests
- replay dataset test
- replay CE test
- slot-only loss test
- contrastive loss test
- image dropout test
- branch probe test
- hidden anchor test
- Stage1-v2 gate test

Local verification:

- `python -m py_compile` on modified Stage1/replay/eval/gate scripts passed.
- `bash -n` on Stage1-v2 clean, continue, and wait launchers passed.
- `/root/miniconda3/envs/navsim/bin/python -m pytest -q tests/test_recogdrive_stage1_replay_text.py tests/test_stage1_replay_dataset.py tests/test_two_expert_*.py tests/test_no_future_leakage.py` passed with `60 passed`.
