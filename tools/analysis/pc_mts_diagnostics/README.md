# PC-MTS mechanism diagnostics

These scripts are analysis-only. They do not update weights, train a model, or rewrite historical candidate archives.

The requested experiment is preserved in `configs/pc_mts_diagnostics/REQUEST_zh.txt`. The frozen scene list and checkpoint paths/hashes are under `outputs/pc_mts_diagnostics/manifests`.

## Running

Use `/root/miniconda3/envs/navsim/bin/python` from the repository root. All CPU stages should set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
python tools/analysis/pc_mts_diagnostics/discover_assets.py
python tools/analysis/pc_mts_diagnostics/build_scene_manifest.py --num-scenes 1000
python tools/analysis/pc_mts_diagnostics/run_all.py --config configs/pc_mts_diagnostics/smoke.yaml --manifest outputs/pc_mts_diagnostics/manifests/scenes_smoke.json --phase gpu
python tools/analysis/pc_mts_diagnostics/run_all.py --config configs/pc_mts_diagnostics/smoke.yaml --manifest outputs/pc_mts_diagnostics/manifests/scenes_smoke.json --phase score
python tools/analysis/pc_mts_diagnostics/run_all.py --config configs/pc_mts_diagnostics/analysis.yaml --manifest outputs/pc_mts_diagnostics/manifests/scenes_1000.json --phase gpu
python tools/analysis/pc_mts_diagnostics/run_all.py --config configs/pc_mts_diagnostics/analysis.yaml --manifest outputs/pc_mts_diagnostics/manifests/scenes_1000.json --phase score
```

`--phase candidates` constructs and scores common candidates, applies selection, validates pool differences, and performs held-out robustness analysis. `--phase analysis` generates policy/readiness statistics and all available figures. `--phase all` runs these phases sequentially. The actual run initially launched a driver version with GPU and rollout-scoring phases only; later candidate/analysis commands are recorded separately in execution logs.

PC-MTS can have zero eligible candidates even after the prescribed relaxations. The default `--empty-policy error` preserves the original protocol. The user subsequently explicitly required 16 candidates for every one of the 1000 scenes. The authorized `--empty-policy coverage` extension uses the frozen `coverage_fallback.yaml`: for zero-parent scenes only, try original quality without q, then feasible + selection-local, then feasible, then NC=DAC=1. It selects real trajectories from the same reservoir and records every added tier. No held-out scores enter selection. Original nonempty pools remain unchanged.

The final full-cohort stage is:

```bash
/root/miniconda3/envs/navsim/bin/python tools/analysis/pc_mts_diagnostics/run_all.py --config configs/pc_mts_diagnostics/analysis.yaml --manifest outputs/pc_mts_diagnostics/manifests/scenes_1000.json --phase pc-coverage
```

For a fresh end-to-end run, pass `--phase all --empty-policy coverage`; after it finishes, run `--phase pc-coverage` for the coverage-specific validation, subgroup statistics, and report. Repeat with the smoke config and scene manifest before the main coverage stage. The final result is called **PC-MTS + coverage**, retaining the original 80.8% zero-parent rate. The extension is exploratory and was defined after partial results were known, before its new candidate/held-out evaluation.

The historical `--phase pc-partial` stage completed only the 192 originally eligible-parent scenes, while the zero-parent definition was unresolved. This snapshot is preserved under `metrics/conditional_pc`, `figures/conditional_pc`, and `report/PC_MTS_CONDITIONAL_DIAGNOSTICS.md`; it is not the current full-cohort result. Run it only before filling the zero-parent scenes, into an output directory preserving the original protocol. Its original command was:

```bash
/root/miniconda3/envs/navsim/bin/python tools/analysis/pc_mts_diagnostics/run_all.py --config configs/pc_mts_diagnostics/analysis.yaml --manifest outputs/pc_mts_diagnostics/manifests/scenes_1000.json --phase pc-partial
```

The partial stage was also exercised on the 20-scene smoke cohort (3 originally with accepted parents). The subsequent coverage stage completed the other 17 and passed the full four-method smoke gate before main coverage evaluation. `audit_zero_parents.py` records the original frozen eligibility funnel without searching for more favorable thresholds.

Do not change a frozen config and resume into the same output directory. Cache identities include a scene-manifest hash, config hash, checkpoint hash/path where applicable, and parent artifact hashes for derived arrays. Shared observation caches are tied to their separate VLM provenance manifest. Identical cached expensive outputs can be reused. The 2000-scene manifest option is supported, but a new config/output directory is required for its computations.

## Metrics and caveats

Distance is mean XY displacement over eight matched future time points, measured in metres. Policy entropy is not inferred from PDMS. The empirical kNN support percentile is not a calibrated probability density.

Standard PDMS uses native evaluator weights 5/5/2; archived original GRPO used 10/5/2 for progress/TTC/comfort. Advantage diagnostics execute the archived advantage code on the archived reward definition, while p_plus and Hit@8 use common standard PDMS. Feasible means NC=DAC=TTC=DDC=1, and hard failure means NC<1 or DAC<1. These are explicitly declared diagnostics, not a claim of real-world safety.

The four reconstructed pools do not identify the training recipe of either MTS checkpoint. Their source bank excludes official IL and does not retrain any checkpoint. Preserve native learned FS-Norm in MTS models; compare common physical outputs rather than changing a trained model's representation.

Report both ALL-16 and UNIQUE-NON-FILL. Missing subsets remain missing, not zero width. Bootstrap and paired comparisons use scenes; positive-advantage denominators and valid subset counts are retained.
