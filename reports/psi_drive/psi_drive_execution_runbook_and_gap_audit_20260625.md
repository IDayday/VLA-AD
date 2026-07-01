# PSI-Drive Execution Runbook and Gap Audit, 2026-06-25

本文档对应目标文件：

- `CODEX_GOAL_PSI_Drive_Stage2_Stage3.md`
- `PSI_Drive_Stage2_Stage3_方案与执行规范.md`

当前目标尚未完成。本文只记录当前可验证状态、复现实验命令和剩余缺口，避免把运行中实验误报为完成。

## 当前主实验

- Stage2 run root: `/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z`
- Stage2 mode: random init, APSD Clean, full train cache `103288`
- Stage2 support index: `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt`
- Stage2 support index SHA256: `817ada27e18a92062e2fb4bd5fddacaa3f75cbd30a0f8d903119bc04ab4bbf27`
- Stage2 training PID at 2026-06-26T01:11:38Z: `1521627`
- Latest raw checkpoint at 2026-06-26T01:14:15Z: `epoch_082`, archived
- Latest audited checkpoint store: 84 archived objects, audit passed with 4 temp-file warnings
- Latest archived checkpoint SHA256: `a2883f5bdae7d43faa8d81ee575e31942737fafd12b1c31e1cf50c3b4f571624`
- Stage3 orchestrator PID at 2026-06-26T01:11:38Z: `1548272`
- Stage3 launch condition: wait for Stage2 val6000 eval `done=200/200`, then select val6000 Top-1 object checkpoint
- Current preflight supplement: `reports/psi_drive/preflight_random_init_full103k_gt_supp_20260625T185920Z.md`

## Data and Cache Evidence

- Official Stage1 hidden cache: `/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b`
- Stage2 command disables external feature injection:
  - `agent.use_jepa=false`
  - `agent.use_vggt=false`
  - `agent.use_last_rd=false`
  - `agent.use_last_vla=false`
  - `agent.use_two_expert_slots=false`
- Stage2 data report:
  - train records: `103288`
  - validation/log-val records: `18179`
  - loader mode: `official-cache-loader-all-cache-train-log-val`
  - support index records: `103288`
- Fixed val6000:
  - token file: `artifacts/splits/navtrain_val6000_seed260306049.txt`
  - token count: `6000`
  - SHA256: `1b6355bfd1f1fbf9438897d34c64320d3f46d07c62437fe8da7df5e42c5cbc54`
  - train-complement overlap: `0`
  - navtest overlap: `0`

## Support Index Evidence

- Supplemented support index audit: `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z_audit.json`
- Audit result: `passed=true`
- Explicit overlap audit: `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z_explicit_overlap_audit.json`
  - result: `passed=true`
  - overlap with train-complement: `97288`
  - overlap with fixed val6000: `6000`
  - overlap with navtest: `0`
  - interpretation: the fixed val6000 overlap is expected for the current user-requested full `103k` train-cache run; navtest remains excluded.
- Unique tokens: `103288`
- Support count histogram:
  - `1`: `58287`
  - `2`: `34054`
  - `3`: `10947`
- Fallbacks: `gt_fallback=69`
- Weights sum over valid support entries: min `1.0`, max `1.0`
- Finite checks passed for trajectories, weights, and scores
- GT-only supplement:
  - missing-candidate token count: `18179`
  - final supplemental files: `18179`
  - generation errors: `0`
  - latest shard summaries: `14663` written, `3516` skipped as already done
  - valid candidate better than GT in latest shard summaries: `11024`
  - weighted mean valid-best-minus-GT: `+0.01641`
- Support target generation conclusion:
  - The active run should not be restarted for lateral/timing expansion because GT-only supplement wins are dominated by `progress_endpoint` and `progress_speed` candidates.
  - A later ablation, if needed, should first densify endpoint-progress and speed-scale grids; lateral/timing-only candidates had very low valid-best frequency in the supplement audit.

## Exact Active Stage2 Command

Resolved command is stored at:

```text
/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z/resolved_command.sh
```

Key command fields:

```text
agent.checkpoint_path=
agent.allow_random_init=true
agent.lr=1e-4
trainer.params.max_epochs=200
trainer.params.devices=8
dataloader.params.batch_size=16
cache_train_all_records=true
stage2_target_source=pareto_support
stage2_pareto_require_index=true
```

This is intentionally different from the original 60-epoch official-initialized development run: the user requested random-initialized Stage2 over the full `103k` train cache.

## Watchers and Gates

Stage2 eval watchers:

```text
remote host: training-rl-zt3
remote val6000 watcher PID: 3015882
remote navtest Top-5 watcher PID: 3016634
```

Expected behavior:

- Each watcher first archives stable raw checkpoints into `checkpoint_store/objects`.
- val6000 exact PDM runs on `training-rl-zt3` GPUs `1,2,3,4,5,6`.
- navtest watcher reads `rankings/val6000/current_top5.tsv` and evaluates only the current Top-5 checkpoint objects.
- Completed checkpoint SHA values are skipped by marker/status and are not re-evaluated.
- `backup_ranked_checkpoints.py` copies val6000 Top-5 and navtest Top-3 checkpoint objects into `checkpoint_backups`.
- navtest is evaluation-only and must not select Stage3 initialization.

Stage3 orchestrator:

```text
PID: 1548272
script: scripts/psi_drive/run_stage3_after_val6000_top1.sh
```

Expected behavior:

- Wait until all expected Stage2 checkpoints have completed val6000 eval.
- Fail if any val6000 eval fails.
- Select Top-1 from `rankings/val6000/current_top5.tsv`.
- Launch `scripts/psi_drive/run_stage3_sr_pgrpo_8gpu.sh`.

## Verification Completed

Recent checks:

```text
pytest -q tests/test_psi_checkpoint_tools.py tests/test_checkpoint_store_audit.py
```

Result: `8 passed`

```text
pytest -q tests/test_pareto_support_selection.py tests/test_pareto_support_index_io.py tests/test_stage2_pareto_support_loader.py tests/test_stage2_pareto_support_forward.py tests/test_stage2_original_path_regression.py
```

Result: `13 passed`

```text
pytest -q tests/test_support_bucket_assignment.py tests/test_support_relative_advantage.py tests/test_final_advantage_caps.py tests/test_core_pareto_v2_regression_disabled.py
```

Result: `5 passed`

Shell and compile checks:

- checkpoint lifecycle tools: `py_compile` passed
- checkpoint store audit warning extension: `pytest -q tests/test_checkpoint_store_audit.py` passed, now reports hidden `.tmp` files as warnings without failing verified immutable objects.
- checkpoint archive temp cleanup: `pytest -q tests/test_psi_checkpoint_tools.py tests/test_checkpoint_store_audit.py` passed with `8 passed`; the archive tool now removes only its own temporary hardlink after a same-inode concurrent archive race.
- PSI Stage2/Stage3 watcher and launcher scripts: `bash -n` passed
- PSI run summary utility:
  - script: `scripts/psi_drive/summarize_psi_run.py`
  - test: `pytest -q tests/test_psi_run_summary.py`
  - result: `1 passed`
  - current Stage2 summary: `/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z/reports/run_summary_latest.md`

## Completion Gap Audit

### Complete or Currently Proven

- Stage1 code path is not modified for this experiment.
- APSD support utilities exist and have unit coverage.
- Stage2 APSD target injection exists and disabled-path tests pass.
- SR-PGRPO support-relative advantage exists and disabled-path tests pass.
- Fixed val6000 token file and audit exist.
- Content-addressed checkpoint tools exist and unit tests pass.
- Current random-init full103k Stage2 run has been launched.
- Stable Stage2 raw checkpoints are being archived append-only.
- Latest refreshed Stage2 run summary at `2026-06-26T01:14:15Z` confirms 84 raw checkpoints, 84 immutable archived objects, and checkpoint-store audit `passed=true` with 4 temp-file warnings.
- Temp-file warning check after the `2026-06-26T01:14:06Z` audit confirmed the same 4 hidden `.tmp` entries remain warning-only lifecycle residue; 84 inventory rows and 84 object files validated successfully.

### Running, Not Yet Complete

- Stage2 random-init full103k training: running, expected `200` epochs.
- Process-tree check at `2026-06-26T01:11:38Z` confirmed active `torchrun` PID `1521634`; current GPU memory/utilization samples remain consistent with active 8-GPU training.
- Stage2 val6000 exact PDM evaluation: running remotely on `training-rl-zt3`; `epoch_001` marked `started` at `2026-06-26T01:49:19Z`.
- Stage2 navtest exact PDM evaluation: remote watcher is running and waits for val6000 Top-5 ranking rows; it will evaluate only those Top-5 checkpoint objects.
- Stage3 SR-PGRPO training: waiting for Stage2 val6000 Top-1; current gate status `done=0/200` at `2026-06-26T01:10:33Z`.

### Missing Until Training/Eval Finishes

- Stage2 val6000 Top-5 manifest and backup weights.
- Stage2 navtest Top-3 manifest and backup weights.
- Stage2 full metric summary and checkpoint trend.
- Stage2 SupportRecall@16 / occupancy / BestValid@K diagnostics on the final candidate set.
- Stage3 checkpoint archive/eval results.
- Stage3 val6000 Top-5 manifest and immutable Top-5 weights.
- Stage3 navtest Top-5 manifest and immutable Top-5 weights.
- Stage3 support diagnostics, group diagnostics, checkpoint volatility, Top-5 mean/std.
- Final implementation report with completed/failed/pending items.

## Resume Commands

Refresh checkpoint store audit:

```bash
RUN=/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z
python scripts/checkpoints/audit_checkpoint_store.py \
  --run-root "$RUN" \
  --output-json "$RUN/checkpoint_store/audit_latest.json" \
  --output-md "$RUN/checkpoint_store/audit_latest.md"
```

Generate or refresh the PSI run summary:

```bash
RUN=/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z
python scripts/psi_drive/summarize_psi_run.py \
  --run-root "$RUN" \
  --stage stage2 \
  --output-json "$RUN/reports/run_summary_latest.json" \
  --output-md "$RUN/reports/run_summary_latest.md"
```

Check Stage2 watcher status:

```bash
RUN=/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z
tail -80 "$RUN/logs/watch_stage2_eval_val6000.log"
tail -80 "$RUN/logs/watch_stage2_eval_val6000.training-rl-zt3.log"
tail -80 "$RUN/logs/watch_stage2_eval_navtest_top5.training-rl-zt3.log"
wc -l "$RUN/eval/val6000/checkpoint_eval_status.tsv" "$RUN/eval/navtest/checkpoint_eval_status.tsv"
cat "$RUN/state/watch_stage2_eval_val6000.training-rl-zt3.pid"
cat "$RUN/state/watch_stage2_eval_navtest_top5.training-rl-zt3.pid"
```

Check Stage3 orchestrator status:

```bash
STAGE3=/mnt/project/VLA-AD/outputs/psi_drive_stage3_sr_pgrpo_from_stage2_random_full103k_gt_supp_wait200_20260625T171030Z
tail -80 "$STAGE3/logs/stage3_after_val6000_orchestrator.log"
cat "$STAGE3/state/stage3_after_val6000.pid"
```

Do not delete checkpoints or output roots. If disk pressure appears, stop and write an inventory report before making any cleanup decision.
