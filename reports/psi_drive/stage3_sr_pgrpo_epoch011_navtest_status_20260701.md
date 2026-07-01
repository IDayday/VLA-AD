# Stage3 SR-PGRPO Epoch011 Navtest Status

Update time: `2026-07-01T01:20:48Z`

## Run

- Run root: `/mnt/project/VLA-AD/outputs/psi_drive_stage3_sr_pgrpo_epoch011_b2acc4_20e_20260627T135941Z`
- Stage2 init checkpoint: `/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/checkpoint_store/objects/051ac3312a848b36eeeaccc3935e03582f9acc89737a99672349a5887faa0aff.ckpt`
- Support index: `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt`
- Hidden cache: `/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b`
- Metric cache: `/mnt/project/VLA-AD/cache/metric_cache_train_full`

## Training And Evaluation Settings

- Algorithm: SR-PGRPO on top of Core-Pareto GRPO v2
- `grpo_sample_time=16`
- `lr=1e-4`
- `batch_size=2`
- `accumulate_grad_batches=4`
- `effective_batch=64`
- `max_epochs=20`
- `reference_kl_coeff=0.02`
- `bc_coeff=0.10 -> 0.05` over 5 epochs
- Checkpoint cadence: every 300 train steps
- Evaluation split: full `navtest`
- Evaluation device policy after remote shutdown: local only

## Current Navtest Completion

- Raw step checkpoints: `84`
- Completed navtest evaluations: `84`
- Running evaluations: `0`
- Failed evaluations: `0`
- Pending evaluations: `0`
- Latest evaluated checkpoint: `step_00025200.ckpt`

## Navtest Top 10

| rank | checkpoint | PDMS | NC | DAC | TTC | EP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `step_00021000` | `0.908939815` | `0.978620860` | `0.975366617` | `0.946531554` | `0.876468897` |
| 2 | `step_00021600` | `0.908874631` | `0.981751524` | `0.975613775` | `0.950486077` | `0.870372905` |
| 3 | `step_00018900` | `0.907124711` | `0.981875103` | `0.973471742` | `0.952710496` | `0.866854630` |
| 4 | `step_00018600` | `0.906701621` | `0.980721700` | `0.971659252` | `0.949085517` | `0.871723410` |
| 5 | `step_00023700` | `0.906184594` | `0.979073983` | `0.972483111` | `0.946696326` | `0.873921931` |
| 6 | `step_00022500` | `0.906093456` | `0.975902126` | `0.973142198` | `0.938869666` | `0.880544717` |
| 7 | `step_00025200` | `0.906064817` | `0.977096721` | `0.971082551` | `0.940682155` | `0.881088801` |
| 8 | `step_00020700` | `0.905970487` | `0.981133630` | `0.972730269` | `0.949497446` | `0.868733659` |
| 9 | `step_00024300` | `0.905874011` | `0.976849563` | `0.972071181` | `0.941506014` | `0.877439522` |
| 10 | `step_00023100` | `0.905655917` | `0.975819740` | `0.971082551` | `0.940270226` | `0.880632952` |

## SOTA Backup

- Current navtest Top1: `step_00021000`
- Top1 sha256: `f865bdc6a70500b49106fc39919afecfe83f04d9b7cac7e9f204e19bddca090d`
- Backup object: `/mnt/project/VLA-AD/outputs/psi_drive_stage3_sr_pgrpo_epoch011_b2acc4_20e_20260627T135941Z/checkpoint_backups/objects/f865bdc6a70500b49106fc39919afecfe83f04d9b7cac7e9f204e19bddca090d.ckpt`

## Comparison To Core-Pareto GRPO v2 SOTA

Reference v2 SOTA:

- Run: `stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`
- Best checkpoint: `step_00021600`
- Navtest PDMS: `0.910274164`

Current SR-PGRPO best:

- Best checkpoint: `step_00021000`
- Navtest PDMS: `0.908939815`
- Gap to v2 SOTA: `-0.001334349`

The current run improves over its early SR-PGRPO checkpoints, but it has not exceeded the previous Core-Pareto GRPO v2 SOTA. The main tradeoff is that SR-PGRPO obtains higher EP on its best checkpoint, while NC, DAC, and TTC remain below the v2 SOTA checkpoint. This suggests the support-relative advantage is shifting probability mass toward stronger progress but is not yet preserving the full Core-Pareto safety/TTC profile.

## Code-Level Difference From V2

Compared with the matched Core-Pareto GRPO v2 setting, this run enables:

- support-relative advantage via `grpo_use_support_relative=true`
- support bank lookup and trajectory-to-support bucket assignment
- intra-support z-score ranking for valid+EP candidates
- positive-only inter-support bucket comparison
- free-bucket novelty cap
- sign-preserving RMS advantage scaling
- final cap reapplication for invalid, slow-fail, and dominated candidates

Observed training diagnostics confirm the SR path was active:

- `support_relative_enabled=1`
- `support_missing_ratio=0`
- `grpo_advantage_rms_scaled=1`
- `grpo_reapplied_final_caps=1`
- `final_positive_invalid_ratio=0`
- `final_positive_slow_fail_ratio=0`
- `final_positive_dominated_ratio=0`

The most important remaining attribution experiment is `epoch011 + Core-Pareto GRPO v2 matched`, with `grpo_use_support_relative=false` and all other settings kept aligned. That separates the effect of the new SR-PGRPO objective from the effect of changing the Stage2 initialization checkpoint.
