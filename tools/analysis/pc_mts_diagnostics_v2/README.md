# PC-MTS diagnostics V2

The V1 commit `2dd4b55fc658e6182aaf857bf3267b652c873cc7` and every existing V1 result are read-only inputs. V2 output root is **`outputs/pc_mts_diagnostics_v2/`**. The diagnostic reservoir is not historical training data.

Primary policy-distribution comparison: **five checkpoints × 1000 scenes × 64 cached V1 rollouts**. Do not regenerate these. The 16 trajectories selected by each of four rules are supervision candidates, not a 16-sample estimate of each model's stochastic policy. New official IL support is R128/Q128 plus a separate 32-trajectory native candidate bank per scene.

Use `/root/miniconda3/envs/navsim/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. The isolated sklearn 1.5.2, joblib 1.4.2, threadpoolctl 3.5.0 installation is under the V2 `deps/` directory; it is a dependency, not a result. `run_phase.py NAME -- COMMAND ...` records execution, logs and CPU/GPU telemetry. No automatic retry changes any scientific threshold.

Execution dependencies:

1. `prepare.py`: seal V1 files and freeze scene/seed manifests. `prepare.py --verify`: compare the seal and inventory after all work.
2. `resources.py stop`: temporarily stop only the existing identified GPU pressure jobs. Preserve the snapshot; restore with `resources.py restore` after GPU work.
3. `torchrun --standalone --nproc_per_node=8 gpu_banks.py`: independent R/Q/native samples, 4 DataLoader workers per rank. `--smoke` uses the fixed 20-scene smoke manifest. Every sample cache has hashes and partitioned seeds.
4. `build_reservoir.py`: GT32 + native32 + external64 + bridge64; 96 CPU workers. `score.py raw_candidates --validate`, then `score.py selection_local`: actual NAVSIM batch evaluation and scalar parity checks.
5. `raw_diagnostics.py`: inspect all 192 raw candidates and freeze the contrastive subset before selection. Do not alter alpha or subset criteria from its output.
6. `select_pools.py`: frozen four-pool rules and auditable PC fallbacks. Neither denoising nor held-out robustness is read. The never-triggered zero-compatible fallback uses minimum continuous KNN distance among native candidates.
7. `seeded_robustness.py`, then `score.py heldout_seeded`: 24 independently seeded smooth shape probes per selected slot, 96 CPU workers. The initial deterministic-shape probes remain in `heldout/` and `evaluator/heldout/`; primary corrected probes use `heldout_seeded/` and `evaluator/heldout_seeded/`. The correction manifest records why it was necessary and fixes the shape rule before reevaluation. This is the only robustness implementation correction; selection and model rollouts do not depend on it.
8. `torchrun --standalone --nproc_per_node=8 denoising.py`: exactly 500 frozen scenes, native official-IL normalized forward diffusion and deterministic native reverse chain. Q128 calibrates each scene's epsilon. Full-chain parity is checked against native `get_action`.
9. `historical_chains.py --discover`: verify three actual initializations and choose four temporal snapshots without reading new outcomes. For each non-reused snapshot, run `torchrun --standalone --nproc_per_node=8 historical_chains.py --snapshot NAME`; then `historical_chains.py --score` and `analyze.py chains`. The three requested Score/Pareto/GT-distance recipe chains remain unverified; available A5/V6 histories are explicitly labeled LFP-GRPO.
10. `sample_stability.py`: 100 common-index 16/32/64 subsamples of V1's existing 64 rollouts, plus separate paired-scene bootstrap intervals. `analyze.py stratification`: IL-spread quartiles, driving commands, GT-heading strata.
11. `analyze.py pools`: full1000, contrastive, same-source matched-scene, non-fill, common qualified-scene, and matched-cardinality comparisons. Primary inputs are corrected `heldout_seeded` scores. No candidate is discarded from the primary cohort.
12. `audit.py --calibration`, `audit.py`: independent arithmetic and completeness checks. `make_figures.py` exports PNG/PDF; `make_report.py` renders the Chinese A–L report. The report also documents the initial deterministic-shape results and corrected seeded results.

The experiment used four-dimensional protocol distinctions that should not be conflated: scene identity, model sampling stream, reference/calibration partition, and downstream validation probe. “Held-out” here refers to the latter streams, not an unseen Navtest population. All 1000 scenes are Navtrain scenes; PDMS in this diagnostic cohort is not the historical Navtest checkpoint label.

The remote commit includes analysis code, frozen configuration/manifests, full and per-candidate result tables, figures, and the report. Per-scene inference/evaluator arrays remain cached on this server and are indexed by SHA256 for audit; installed dependencies and runtime GPU pressure logs are not versioned as scientific results. V1's source caches are also server-local, as in the original observational commit.
