# SG-FPS Implementation Notes

## Finalized Scope

This implementation adds the SG-FPS support archive stack for ReCogDrive/VLA diffusion planning:

- candidate dataclasses and archive IO
- evaluator metric/feasibility utilities
- GT, ReCogDrive Stage3 cache/precomputed, DDV2 precomputed, and DriveOR precomputed seed generators
- path-speed factorization and recombination
- failure-conditioned operators
- cheap filtering and scorer/fallback acquisition
- evaluator-verified Pareto support selection
- full archive plus v3 elite-buffer dual write
- FS-Norm stats builder
- vector scorer dataset/model/training entrypoint
- B-PDAS scene weight builder

Existing ReCogDrive Stage2/Stage3 paths remain opt-in through config/env flags. The root output directory from `mine_pareto_support.py` is compatible with the existing planner `offline_rl_buffer` v3 loader. The complete SG-FPS archive is written under `full_archive/`.

The finalized experiment path for the current run uses the evaluator-verified clean support archive:

```text
/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3
```

This path is not stored in git; it is recorded here so future runs can reproduce the exact code path and configuration choices.

## Verified Label Contract

Positive support must be evaluator-verified. By default, `mine_pareto_support.py` requires candidates to contain true evaluator components, including:

- `pdms`
- `no_at_fault_collisions`
- `drivable_area_compliance`
- `time_to_collision_within_bound`
- `ego_progress`
- `history_comfort`
- `driving_direction_compliance`

Unverified generated candidates are not written as support unless a real NAVSIM evaluator context is provided by the caller. `--allow_unverified_smoke` exists only for synthetic tests and must not be used for real experiments.

## Final Support Construction Contract

The finalized construction path is:

1. Build seed candidates from GT, optional ReCogDrive Stage3/IL/current-policy trajectories, DDV2 precomputed trajectories, DriveOR precomputed trajectories, and structured generation.
2. Apply command/semantic alignment gates before expansion so turn/straight intent mismatches do not propagate.
3. Apply start alignment and local geometry gates so selected trajectories share the same ego-local origin and avoid visible early kinks or tail reverse artifacts.
4. Run evaluator-backed metric extraction where available; support records must carry true PDMS components, DDC, and feasibility statistics.
5. Select support by trajectory quality and Pareto diversity rather than source/category quota. GT is retained as a hard anchor unless GT is poor and a clearly better replacement exists.
6. Use reference-relative DDC/feasibility checks where appropriate. This avoids incorrectly discarding useful trajectories in scenes whose GT/reference DDC is intrinsically hard.
7. Re-audit and gap-check the selected support archive before training. If a small number of scenes remain low-count after repeated supplementation, the pipeline proceeds to training rather than overfitting candidate generation to those cases.

The selected support is intended to diversify policy supervision. Source diversity is useful only insofar as it creates trajectory-level diversity; final selection is based on evaluator quality, semantic consistency, geometry, and Pareto coverage.

## Dual Output Layout

```text
output_archive_dir/
  <sha1(scene_token)>.pkl.xz        # planner-compatible v3 elite-buffer record
  global_summary_rank0.json
  full_archive/
    <sha1(scene_token)>.pkl.xz      # full ParetoSupportArchive record
```

Use `output_archive_dir` as `SUPPORT_ARCHIVE_PATH` for DPSI/GRPO training. Use `output_archive_dir/full_archive` or simply `output_archive_dir` for inspection/FS-Norm scripts; the scripts auto-detect `full_archive/`.

## Final Archive Audit

The last full audit was run from:

```text
/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/audit_full_20260705T190919Z
```

Key results:

- `record_count`: 103288
- `version_counts`: `{"3": 103288}`
- `has_valid_candidate_ratio`: 1.0
- `selected_valid_ratio.mean`: 1.0
- `selected_count.mean`: 10.279
- `selected_count.p50`: 12
- `selected_count.min`: 1
- `selected_has_improver_over_gt_ratio`: 0.58486
- `external_candidate_scene_ratio`: 0.96323
- `external_selected_scene_ratio`: 0.91125
- `selected_source_diversity_ge2_ratio`: 0.96078
- `selected_low_reward_count`: 0
- `selected_first_point_mismatch_count`: 0
- `selected_local_kink_count`: 0
- `selected_semantic_mismatch_count`: 0
- `fallback_tag_ratio`: 0.00779
- `warnings`: `[]`

Remaining gaps are low-count scenes rather than invalid-quality scenes:

- `low_support_count`: 12702
- `poor_gt_few_candidates`: 3
- `record_count`: 103288

This is acceptable for the current training run because the archive has full scene coverage, no selected low-reward records, no selected semantic mismatches, no selected start mismatch, and no selected local-kink records under the current audit thresholds.

## Example Commands

Build seed archive:

```bash
python scripts/pareto_support/build_seed_archive.py \
  --cache_path /mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b \
  --output_archive_dir /path/to/sg_fps_archive \
  --recogdrive_stage3_checkpoint cache \
  --ddv2_precomputed_dir /path/to/ddv2_outputs \
  --driveor_precomputed_dir /path/to/driveor_outputs \
  --support_top_m 12
```

Mine Pareto support:

```bash
python scripts/pareto_support/mine_pareto_support.py \
  --round all \
  --cache_path /mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b \
  --metric_cache_path /mnt/project/VLA-AD/cache/metric_cache_train_full \
  --output_archive_dir /path/to/sg_fps_archive \
  --recogdrive_stage3_checkpoint cache \
  --ddv2_precomputed_dir /path/to/ddv2_outputs \
  --driveor_precomputed_dir /path/to/driveor_outputs \
  --support_top_m 12 \
  --true_eval_budget 16
```

Inspect archive:

```bash
python scripts/pareto_support/inspect_pareto_archive.py \
  --archive_dir /path/to/sg_fps_archive \
  --output_json /path/to/summary.json \
  --output_csv /path/to/per_scene.csv
```

Build FS-Norm stats:

```bash
python scripts/pareto_support/build_fs_norm_stats.py \
  --archive_dir /path/to/sg_fps_archive \
  --output_stats /path/to/fs_norm_stats.npz \
  --robust true
```

Train vector scorer:

```bash
python scripts/pareto_support/train_vector_scorer.py \
  --archive_dir /path/to/sg_fps_archive \
  --output_ckpt /path/to/pareto_vector_scorer.pt \
  --epochs 5 \
  --batch_size 256
```

Export DPSI targets:

```bash
python scripts/pareto_support/export_dpsi_dataset.py \
  --archive_dir /path/to/sg_fps_archive \
  --output_dir /path/to/dpsi_targets \
  --max_support_per_scene 12
```

Build B-PDAS weights:

```bash
python scripts/pareto_support/build_bpdas_scene_weights.py \
  --archive_dir /path/to/sg_fps_archive \
  --output_weights_jsonl /path/to/bpdas_weights.jsonl \
  --sample_count 8
```

Train DPSI:

```bash
SUPPORT_ARCHIVE_PATH=/path/to/sg_fps_archive \
SG_FPS_USE_DPSI=true \
scripts/training/sg_fps/run_train_dpsi.sh
```

The current 200-epoch DPSI/FS-Norm run was launched with:

```text
run_root=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_stepckpt_20260705T193744Z
train_out=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_stepckpt_20260705T193744Z/dpsi_fs_norm_200ep
support_archive_path=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3
max_epochs=200
checkpoint_every_n_epochs=1
checkpoint_every_n_train_steps=10000
use_fs_norm=true
x0_aux_weight=0.1
geo_aux_weight=0.05
```

Train Feasible Pareto-GRPO:

```bash
SUPPORT_ARCHIVE_PATH=/path/to/sg_fps_archive \
GRPO_REWARD_MODE=feasible_pareto \
GRPO_USE_FEASIBLE_PARETO=true \
GRPO_FP_USE_PDAS=true \
scripts/training/sg_fps/run_train_sg_fps_grpo.sh
```

The current evaluation watcher is prepared to evaluate epoch >= 101 checkpoints:

```text
watch_root=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_stepckpt_20260705T193744Z/eval_epoch100_val6000_top10_navtest/watch_epoch101_val6000_top10_navtest
val_cache_root=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_eval_resources/val6000_recogdrive_raw_index
val_metric_cache=/mnt/project/VLA-AD/cache/metric_cache_train_full
navtest_cache_root=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
navtest_metric_cache=/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1
top_k=10
poll_seconds=1800
stagger_seconds=1
```

## Validation Run

The synthetic 10-scene smoke verified:

- `scene_count=10`
- `avg_seed_count=7.0`
- `avg_support_count=6.0`
- `avg_true_evaluated_count=7.0`
- root v3 buffer records load through `offline_rl_buffer.load_elite_record_path`

Unit tests:

```bash
pytest -q tests/pareto_support
```

Smoke:

```bash
python scripts/testing/smoke_sg_fps_core.py
```

Additional syntax checks used for the finalized code path:

```bash
python -m py_compile \
  scripts/eval_recogdrive_expert_pdm.py \
  scripts/tools/build_recogdrive_raw_eval_index.py \
  scripts/last_vla_v2/decoupled_highcap_no_risk/watch_epoch_val6000_top5_navtest.py

bash -n \
  scripts/training/run_recogdrive_stage3_awac_iql_2b_local.sh \
  scripts/training/sg_fps/run_train_dpsi.sh
```

The val6000 raw ReCogDrive index for evaluator reuse was built at:

```text
/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_eval_resources/val6000_recogdrive_raw_index
```

## Remaining Risks

- Direct DDV2/DriveOR model adapters are still intentionally fail-fast; current supported experimental path is precomputed trajectory import.
- `mine_pareto_support.py` can create a NAVSIM evaluator context when `--metric_cache_path` is provided. Without it, candidates must already contain true evaluator components.
- Scorer-guided refinement entrypoint refuses to write scorer labels without true evaluator verification.
- A small tail of scenes still has only 1-5 selected supports after quality gates. These are tracked as low-count gaps, not invalid-quality failures.
- Early smoke evaluation with an incorrect val6000 metric cache path failed; the active watcher uses `/mnt/project/VLA-AD/cache/metric_cache_train_full`, which matches the val6000 raw-index tokens.
