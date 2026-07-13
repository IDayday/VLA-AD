# CoT Stage2 Experiment Bundle

Generated: 2026-06-11 UTC. This folder collects the CoT / Last-VLA stage2 experiment evidence, training settings, and code snapshots behind the current conclusion that these CoT designs have not produced a reliable PDMS gain over original ReCogDrive.

## Bottom Line

- Primary original ReCogDrive stage2 baseline: A0 official-aligned `step_00100000`, full NAVTEST `PDMS=0.864891`.
- Best current CoT-related stage2 row in this bundle: no-residual A corrected `latest`, `PDMS=0.863557`, delta `-0.001334` vs A0. This is close, but not an improvement.
- Best VLM-text residual-anchor Last-VLA rows are around `0.846-0.848`, below A0 by roughly `0.017-0.019`.
- Earlier decoupled high-cap progressive A/B rows are much weaker (`A latest 0.538152`, `B step_00120000 0.641284`).
- LoRA did not improve direct text trajectory output: direct-text base `0.845804`, best LoRA row `0.822256`.

The safe conclusion is: do not claim a stage2 CoT improvement unless a same-protocol full NAVTEST result beats `0.864891` with a clear margin and records the exact trajectory output key.

## Files

- `data/full_navtest_pdms.csv`: normalized full NAVTEST rows, including baselines and all aggregate CoT/no-residual/residual-anchor results found locally.
- `data/best_by_family.csv`: best full NAVTEST row per experiment family.
- `data/probe_and_smoke_pdms.csv`: small-sample diagnostics kept separate from full NAVTEST claims.
- `data/metrics_snapshot.jsonl`: raw aggregate metric JSON/CSV rows with source paths.
- `settings/training_settings.md`: concise training/eval settings for each design family.
- `settings/config_snapshots/`: current config snapshots used by these experiment families.
- `code/code_manifest.md`: code and script inventory.
- `code/current_worktree_diff.patch`: tracked source diff at bundle generation time.
- `code/untracked_files_at_generation.txt`: non-bundle untracked files present when this artifact was generated.
- `code/source_snapshots/` and `code/script_snapshots/`: current source/script snapshots for reproduction.
- `reports/`: copied prior reports and runbooks that contain context and older conclusions.

## Reading Order

1. Start with `data/best_by_family.csv` to see why the conclusion is negative.
2. Use `data/full_navtest_pdms.csv` for exact rows, metrics, deltas, and source paths.
3. Check `settings/training_settings.md` and `settings/config_snapshots/` for the paired training settings.
4. Check `code/code_manifest.md` for the source files and launch/eval scripts behind the runs.

Large checkpoints, caches, predictions, and full per-scene CSVs are intentionally not copied into git. The bundle records their absolute source paths under `/mnt/project/VLA-AD/outputs` and `/mnt/project/VLA-AD/cache`.
