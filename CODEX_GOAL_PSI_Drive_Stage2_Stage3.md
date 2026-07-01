# Codex Goal：实现并完整运行 PSI-Drive Stage2–Stage3

## Goal

在仓库 `IDayday/VLA-AD` 的 `feature/recogdrive-last-vla-v2` 分支基础上，实现并验证 **PSI-Drive**：

1. **Stage1 完全不改**；
2. **Stage2 基于原版 ReCogDrive Stage2**，实现 Adaptive Pareto Support Distillation（APSD）；
3. **Stage3 基于当前 SOTA Core-Pareto GRPO v2**，实现 Support-Relative Pareto GRPO（SR-PGRPO）；
4. 完成代码、配置、测试、support bank 构建、Stage2/Stage3 训练、固定 val6000 评估和 full navtest 评估；
5. 对 Stage2 和 Stage3 分别安全保存 **val6000 Top-5** 和 **navtest Top-5** 权重；
6. 不删除、不覆盖、不遗漏任何原始、候选、Top-5 或历史 checkpoint；
7. 所有新增功能默认关闭，旧配置和原版行为保持兼容。

先阅读并遵循设计文档：

```text
PSI_Drive_Stage2_Stage3_方案与执行规范.md
```

若该文档尚未在仓库中，先将其复制到：

```text
docs/research/PSI_Drive_Stage2_Stage3_方案与执行规范.md
```

---

## Context

- Repository: `IDayday/VLA-AD`
- Base branch: `feature/recogdrive-last-vla-v2`
- Stage2 base: original ReCogDrive official-aligned Stage2 path
- Stage3 base: current Core-Pareto GRPO v2 implementation and SOTA run semantics
- Current SOTA reference: `step21600`, navtest PDMS approximately `0.910274`
- Do not modify Stage1 or introduce new inference-time dependencies.

Relevant files to inspect before editing:

```text
navsim/agents/recogdrive/recogdrive_diffusion_planner.py
navsim/agents/recogdrive/recogdrive_agent.py
navsim/agents/recogdrive/offline_action_explorer.py
navsim/agents/recogdrive/offline_rl_buffer.py
navsim/planning/script/run_training_recogdrive.py
navsim/planning/script/run_training_recogdrive_rl.py
navsim/planning/training/agent_lightning_module.py
scripts/last_vla_v2/two_expert_slot/build_stage2_elite_target_index.py
scripts/eval_recogdrive_expert_pdm.py
scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh
scripts/evaluation/analyze_recogdrive_stage3_navtest_pdms.py
docs/OfficialAlignedBaselineGuardrails.md
reports/recogdrive_stage3/core_pareto_grpo_v2_results_summary_20260621.md
reports/recogdrive_stage3/core_pareto_grpo_v2_experiment_log.md
```

---

## Non-negotiable safety constraints

### Git and code safety

- Start by recording:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

- Work in a new branch/worktree such as `feature/psi-drive-stage2-stage3` unless already in a dedicated clean worktree.
- Never use:

```text
git reset --hard
git clean -fdx
rm -rf on repository/output/checkpoint roots
force push
history rewriting
```

- Do not modify or delete the existing SOTA checkpoint or its backup.
- Do not silently alter Stage1, train/val splits, fp32 evaluation semantics, or official Stage2 optimizer/scheduler contract.

### Data safety

- Build all support data from navtrain only.
- Never use navtest samples, navtest metrics, navtest hidden-neighbor labels, or navtest outcomes to build support targets or make training decisions.
- Fail fast on train/val/navtest token overlap.
- Dummy/smoke data may only validate computation; never report dummy metrics as benchmark results.

### Checkpoint safety

- Never delete or overwrite a `.ckpt` file.
- Raw checkpoints are append-only.
- Ranking changes update manifests only; archived checkpoint objects remain forever.
- If disk space is insufficient, stop with a clear error and inventory. Do not clean old checkpoints automatically.
- Do not use a symlink as the only saved copy of a Top-5 checkpoint.

---

## Required algorithm implementation

# Phase 0 — Baseline audit

1. Verify the disabled path reproduces the original Stage2 forward and current Core-Pareto v2 behavior.
2. Record baseline configs, checkpoint paths, commit SHA, package versions, CUDA/PyTorch versions, dataset/cache roots and current checkpoint inventory.
3. Add a preflight report under:

```text
reports/psi_drive/preflight_<timestamp>.md
```

4. Do not proceed to full training until unit tests and smoke tests pass.

---

# Phase 1 — Shared Pareto support utilities

Create:

```text
navsim/agents/recogdrive/pareto_support.py
```

Implement typed, tested utilities:

```python
trajectory_descriptor(traj, interval_length=0.5)
normalize_descriptors(descriptor, mean, std)
select_adaptive_pareto_supports(...)
load_pareto_support_index(...)
lookup_support_batch(tokens, ...)
assign_support_buckets(...)
```

Descriptor fields:

```text
x_final
y_final
heading_final
mean_speed
terminal_speed
early_progress_ratio
```

Requirements:

- physical coordinates in the index;
- finite-value checks;
- deterministic tie-breaking;
- global navtrain descriptor stats stored in the index;
- support count 1–3;
- quality band `best_score - 0.02` configurable;
- minimum normalized descriptor distance `0.75` configurable;
- no invalid positive support;
- explicit fallback metadata.

---

# Phase 2 — Build the Stage2 Pareto support index

Create:

```text
scripts/build_stage2_pareto_support_index.py
scripts/audit_stage2_pareto_support_index.py
```

The builder must reuse existing elite buffer/candidate records and existing exact PDM components whenever available. It must support two source modes:

```text
clean:
  GT + original Stage2 deterministic/stochastic + structured perturbations

bootstrap:
  clean + current step21600 SOTA proposals
```

For every token:

1. load candidates and components;
2. compute Core-Pareto validity and EP floor using the current v2 semantics;
3. compute the current Core-Pareto score;
4. keep candidates in the configurable quality band;
5. select up to 3 supports using best-first + farthest descriptor selection;
6. save trajectories, masks, weights, scores, components, descriptors, sources and selection metadata;
7. if no valid+EP candidate exists, fall back to GT, then deterministic IL if GT is unavailable;
8. never select an invalid candidate as positive support.

Output a compact `.pt` index plus JSON/Markdown audit reports.

Required audit checks:

```text
unique token count
support count 1/2/3 histogram
source histogram
quality gap distribution
descriptor-distance distribution
fallback counts
finite checks
weights sum to 1
train/val/navtest overlap = 0
config hash
git commit
source checkpoint SHA256
```

Add tests:

```text
tests/test_pareto_support_selection.py
tests/test_pareto_support_index_io.py
```

---

# Phase 3 — Stage2 APSD integration

Modify the official-aligned Stage2 path without changing the default path.

## Dataset/config

Extend `stage2_target_source`:

```text
gt
awac_elite_best_valid_above_gt_or_gt
pareto_support
```

Add config fields:

```yaml
stage2_pareto_support_index_path: null
stage2_pareto_best_weight: 0.50
stage2_pareto_gt_weight: 0.20
stage2_pareto_other_weight: 0.30
stage2_pareto_require_index: true
stage2_pareto_log_diagnostics: true
```

In `ChunkCacheDataset`, return optional fields:

```python
support_trajectories: [3, 8, 3]
support_mask: [3]
support_weights: [3]
support_scores: [3]
```

Update `custom_collate_fn` to stack optional fields. Do not change the output for legacy GT mode.

## Planner

Add:

```python
_select_stage2_pareto_support_target(...)
```

Requirements:

- active only in training;
- validation and inference always use the original trajectory path;
- use normalized valid weights and `torch.multinomial`;
- GT probability is reallocated to best support if GT is unavailable/non-finite;
- output diagnostics for selected source/index/support count;
- disabled mode is numerically equivalent to the original path under a fixed seed.

Add tests:

```text
tests/test_stage2_pareto_support_loader.py
tests/test_stage2_pareto_support_forward.py
tests/test_stage2_original_path_regression.py
```

---

# Phase 4 — Stage3 SR-PGRPO integration

Keep `forward_grpo()` rollout, reward computation, chain log-prob, BC and reference KL unchanged except for advantage construction and diagnostics.

Add config fields with backward-compatible defaults:

```yaml
grpo_use_support_relative: false
grpo_support_index_path: null
grpo_support_rank_margin: 0.01
grpo_support_std_floor: 0.005
grpo_support_free_distance: 1.5
grpo_support_novel_margin: 0.01
grpo_support_novel_positive_cap: 0.20
grpo_support_intra_weight: 1.0
grpo_support_inter_weight: 0.15
grpo_support_inter_clip: 0.30
grpo_support_positive_only_inter: true
grpo_support_low_rank_group_weight: 0.25
grpo_use_rms_advantage_scale: true
grpo_reapply_final_caps: true
```

Implement:

```python
_compute_support_relative_pareto_advantages(...)
_sign_preserving_rms_scale(...)
_reapply_final_advantage_caps(...)
```

Algorithm requirements:

1. Map each on-policy trajectory to the nearest scene support using the shared normalized descriptor.
2. Assign distant trajectories to a free bucket.
3. Compute `valid+EP` masks first.
4. Compute intra-bucket z-score only over `valid+EP` candidates.
5. Require bucket count >=2 and score range >=0.01; otherwise intra advantage is zero.
6. Compute bucket representative by Top-2 mean.
7. Inter-bucket advantage is positive-only and clipped to `[0, 0.30]`, weighted by `0.15`.
8. A singleton free candidate can receive positive advantage only when Pareto and above reference by the configured novel margin; cap it.
9. Invalid/slow/dominated semantics remain absolute and must be re-applied after all scaling.
10. Replace full-batch mean-centered advantage normalization with optional sign-preserving RMS scaling when the new method is active.
11. Missing support tokens fall back to original Core-Pareto v2 for that scene.
12. No support data is required for inference or navtest evaluation.

Add diagnostics:

```text
support_missing_ratio
support_count_mean
occupied_support_bucket_count
rankable_support_bucket_count
singleton_support_bucket_ratio
free_bucket_ratio
free_bucket_valid_ratio
within_support_score_std
between_support_rep_std
best_valid_minus_median_valid
best_valid_minus_reference
support occupancy 0/1/2
support concentration
frontier gain
final positive invalid ratio
final positive slow-fail ratio
final positive dominated ratio
valid_count histogram
valid+EP_count histogram
```

Add tests:

```text
tests/test_support_bucket_assignment.py
tests/test_support_relative_advantage.py
tests/test_final_advantage_caps.py
tests/test_core_pareto_v2_regression_disabled.py
```

---

# Phase 5 — Checkpoint lifecycle and Top-5 preservation

This phase is mandatory before any long training.

## Required storage model

For every Stage2 and Stage3 run create:

```text
RUN_ROOT/
  checkpoints/raw/
  checkpoints/val_loss_top5/
  checkpoint_store/objects/
  checkpoint_store/inventory.tsv
  rankings/val6000/current_top5.json
  rankings/val6000/current_top5.tsv
  rankings/val6000/history/
  rankings/navtest/current_top5.json
  rankings/navtest/current_top5.tsv
  rankings/navtest/history/
  eval/val6000/
  eval/navtest/
  state/
  logs/
```

## Implement reusable tools

Create:

```text
scripts/checkpoints/archive_checkpoint_immutable.py
scripts/checkpoints/build_checkpoint_inventory.py
scripts/checkpoints/rank_eval_checkpoints.py
scripts/checkpoints/verify_checkpoint_store.py
scripts/evaluation/watch_psi_stage2_checkpoints.sh
scripts/evaluation/watch_psi_stage3_checkpoints.sh
```

### Immutable archive behavior

For each checkpoint:

1. wait until file size and mtime are stable;
2. calculate SHA256;
3. hardlink into a temporary object when possible, otherwise `shutil.copy2`;
4. verify destination SHA256;
5. atomically rename to `checkpoint_store/objects/<sha256>.ckpt`;
6. append/update inventory under file lock;
7. never move or delete the source checkpoint.

### Ranking behavior

- Deduplicate by SHA256, not path/name.
- val6000 ranking metric: PDMS descending.
- tie-break: Core, NC×DAC, TTC, EP, then checkpoint step.
- navtest ranking metric: PDMS descending with the same tie-break.
- `current_top5` is a manifest referencing immutable object paths.
- Every change writes a timestamped history manifest.
- Never delete an old object or old ranking history.
- If fewer than five completed unique evals exist, write `complete=false` and preserve all available entries.
- Verify that every manifest object exists and matches SHA256.

### Raw checkpoint behavior

- Use an append-only periodic callback with `save_top_k=-1` or equivalent.
- Keep `last.ckpt`, epoch checkpoints and explicit step checkpoints.
- Keep Lightning val-loss Top-5 in a separate directory; it must not be the only checkpoint source.
- Do not run cleanup scripts.
- On low disk space, fail before writing a partial checkpoint and emit an inventory/report.

Add tests simulating:

```text
same checkpoint at two paths
two different checkpoints with same filename
watcher restart
partial copy
hash mismatch
fewer than 5 evals
Top-5 rotation
cross-filesystem fallback
```

---

# Phase 6 — Fixed val6000 protocol

Create a fixed token file and hash, for example:

```text
artifacts/splits/navtrain_val6000_seed260306049.txt
artifacts/splits/navtrain_val6000_seed260306049.sha256
```

Requirements:

- deterministic token selection;
- exactly 6000 unique tokens;
- no overlap with training tokens used for optimizer updates if using a heldout split;
- no overlap with navtest;
- store generation script, seed, source split and SHA256;
- all experiments reuse this exact file;
- use `--sample-token-file` rather than relying only on `--max-samples`;
- fp32 exact PDM evaluation.

Create an audit report and fail on overlap.

---

# Phase 7 — Stage2 training and evaluation

## Training configs

Create clean, isolated configs and launchers; do not overwrite existing configs:

```text
configs/psi_drive/stage2_apsd_clean.yaml
configs/psi_drive/stage2_apsd_bootstrap.yaml
scripts/psi_drive/run_stage2_apsd_8gpu.sh
```

Development run:

```text
initialize from original ReCogDrive Stage2 checkpoint
60 epochs
LR 5e-5
AdamW
weight decay 1e-4
2 warmup epochs
cosine to 1e-6
effective batch 128
16-mixed, fp32 model weights
```

Preserve official train/val log split and optimizer semantics.

## Checkpoint schedule

- save append-only periodic checkpoints at a configurable interval;
- save epoch checkpoints at a configurable interval;
- save `last.ckpt`;
- maintain separate val-loss Top-5;
- archive every periodic candidate before evaluation.

## Evaluation

1. Evaluate every archived Stage2 candidate on fixed val6000.
2. Maintain val6000 Top-5 immutable manifests and weights.
3. Every checkpoint that newly enters the val6000 Top-5 is evaluated on full navtest.
4. Maintain navtest Top-5 among all completed full-navtest evaluations.
5. Stage3 initialization is selected by val6000 only, not navtest.
6. Wait for all queued evaluations before declaring the Stage2 run complete.

Required reports:

```text
stage2_support_audit.md
stage2_val6000_ranking.tsv/md
stage2_navtest_ranking.tsv/md
stage2_checkpoint_inventory.tsv
stage2_top5_verification.md
stage2_metric_summary.md
```

---

# Phase 8 — Stage3 training and evaluation

## Configs and launchers

Create:

```text
configs/psi_drive/stage3_sr_pgrpo.yaml
configs/psi_drive/stage3_core_pareto_v2_matched.yaml
scripts/psi_drive/run_stage3_sr_pgrpo_8gpu.sh
```

Use the val6000-selected APSD Stage2 checkpoint.

Matched SR-PGRPO configuration:

```text
G=16
LR=1e-4
effective scene batch=64
20 epochs
BC 0.10 -> 0.05 over 5 epochs
reference KL=0.02
EP floor on
slow positive cap on
TTC hard gate off
DDC positive reward off; guard only
```

Only support-relative advantage, RMS scaling/re-cap and diagnostics differ from the SOTA base.

## Checkpoint schedule

Follow the existing SOTA cadence, initially every 300 optimizer steps, configurable through environment/config.

## Evaluation

1. Archive every stable Stage3 checkpoint without deleting source.
2. Evaluate each candidate on fixed val6000.
3. Enqueue every new val6000 Top-5 entrant for full navtest.
4. Maintain immutable val6000 Top-5 and navtest Top-5.
5. Do not use navtest to stop training or change hyperparameters.
6. After training ends, wait until the eval queue is empty.
7. Produce complete checkpoint trend, best, last and Top-5 mean/std reports.

Required reports:

```text
stage3_val6000_ranking.tsv/md
stage3_navtest_ranking.tsv/md
stage3_checkpoint_inventory.tsv
stage3_top5_verification.md
stage3_group_diagnostics.md
stage3_support_diagnostics.md
stage3_checkpoint_trend.md
```

---

# Phase 9 — End-to-end experiment matrix

At minimum run or prepare fully reproducible launchers for:

```text
A: original GT Stage2 + Core-Pareto v2
B: Elite-1 Stage2 + Core-Pareto v2
C: APSD Clean + Core-Pareto v2
D: APSD Clean + SR-PGRPO
E: APSD Bootstrap + SR-PGRPO
F: APSD Clean + fixed 3x3 phenotype GRPO
G: D without free bucket
H: D with signed inter-support comparison
```

The central comparisons are C vs A/B and D vs C.

---

## Required validation metrics

### Stage2

```text
MeanPDMS@1
P10/P50/P90 PDMS@1
BestValidPDMS@8/16
NearOptimalHit@16 at delta 0.01/0.02
Valid@16
Valid+EP@16
SupportRecall@16
SupportOccupancyEntropy
MaxSupportOccupancyRatio
mean-pADE/pFDE
minADE/minFDE
```

### Stage3

```text
valid and valid+EP histograms
rankable group ratio
rankable support bucket count
free bucket valid ratio
within-support score std
between-support score std
best-valid minus median-valid
frontier gain
final invalid/slow/dominated positive ratios
reference KL
trajectory log-prob
checkpoint volatility
```

### Checkpoint reporting

For Stage2 and Stage3 separately report:

```text
val6000 Top-5 checkpoint IDs, metrics, SHA256 and immutable paths
navtest Top-5 checkpoint IDs, metrics, SHA256 and immutable paths
last checkpoint
val-loss-best checkpoint
val6000-best checkpoint
navtest-best checkpoint
Top-5 mean/std
number of raw checkpoints
number of archived unique objects
number of completed/failed/pending evaluations
```

---

## Acceptance criteria

### Code and compatibility

- Existing Stage1 files are unchanged except documentation references, if any.
- Original Stage2 and Core-Pareto v2 configs run with new features disabled.
- Disabled-path regression tests pass.
- New support features are controlled only by explicit config flags.
- No inference-time dependency on the support index.

### Support bank

- All support index values are finite.
- Each token has 1–3 real supports.
- Positive supports are valid+EP or an explicit GT/IL fallback.
- Weights sum to 1 over valid entries.
- Train/val/navtest overlap audit passes.
- Clean and Bootstrap indexes are versioned separately.

### Stage3 advantage

- Invalid and slow candidates do not enter valid-only normalization.
- Low-gap/singleton buckets do not receive amplified intra z-scores.
- Positive-only inter-support logic is verified.
- Final positive invalid/slow/dominated ratios are zero within tolerance.
- No NaN/Inf in rollout, reward, advantage, log-prob or loss.

### Checkpoints

- No checkpoint source file is deleted or overwritten.
- Every evaluated checkpoint has a verified immutable SHA256 object.
- val6000 and navtest current Top-5 manifests are valid and have history.
- If five completed evaluations exist, exactly five unique entries are present.
- All Top-5 object files exist and pass SHA256 verification.
- Watcher restart is idempotent.
- Evaluation failure never removes a checkpoint.

### Training/evaluation

- Stage2 and Stage3 launchers, resolved configs and run manifests are saved.
- Fixed val6000 token file and hash are saved.
- All formal evaluation is fp32 exact PDM.
- navtest is evaluation-only.
- No benchmark claim is made from smoke/dummy/subset runs.

---

## Execution behavior

- Work phase by phase and write a progress report after every phase.
- Show the first discovered blocking bug immediately in the report, then continue with all non-blocked work.
- When a path or dependency is missing, fail clearly and document the missing input; do not fabricate data, checkpoints or metrics.
- Do not claim SOTA unless a full navtest evaluation has completed and the exact CSV/TSV plus checkpoint SHA256 are present.
- Do not stop after code changes: complete tests, support-index audit, training launchers, evaluation watchers, ranking/archival tools and runbooks.
- If full training cannot be executed in the current environment, still complete all code, tests, dry-runs, preflight reports and exact launch commands, and state precisely which external resource is missing.

---

## Final deliverables

```text
1. Code diff with APSD and SR-PGRPO.
2. Backward-compatible configs.
3. Support index builder/auditor and generated audit report.
4. Unit, smoke and regression tests.
5. Stage2 and Stage3 launchers.
6. Fixed val6000 split file and audit.
7. Immutable checkpoint manager and Top-5 ranking tools.
8. Stage2 val6000 Top-5 and navtest Top-5 manifests/weights.
9. Stage3 val6000 Top-5 and navtest Top-5 manifests/weights.
10. Full metric and checkpoint trend reports.
11. Reproduction runbook with exact commands and environment variables.
12. Final implementation report listing completed, failed and pending items without overstating results.
```
