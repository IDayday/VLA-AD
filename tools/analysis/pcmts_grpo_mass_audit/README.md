# Shared candidate / native GRPO mass audit

This analysis performs **zero optimizer updates**. V1/V2/V3 are read-only.
All new outputs belong to `outputs/pcmts_grpo_mass_audit`.

1. `preflight.py` freezes the prescribed YAML and fixed 1000 scenes.
2. `sample_native.py --smoke` compares eight actual scenes against native GRPO,
   its reward/advantage computation and the scalar NAVSIM evaluator.
3. `batch_native.py` validates independent group RNG routing at the original
   `forward_grpo` B=8, G=8 batch shape. No sampler mathematics are replaced.
4. `external_candidates.py` captures real DDV2 pre-scorer trajectories and
   DrivoR proposals. It loads all active tensors exactly from released weights;
   DDV2's unused training vocabulary is explicitly audited.
5. `launch.py` starts one genuine inference process per available GPU, with
   detached logs. At most eight GPUs total; no pressure/idle tasks.
6. `score_watch.py --watch` uses 96 persistent CPU workers and the verified
   NAVSIM batch evaluator. Atomic incomplete writes are never input banks.
7. `orchestrate.py` overlaps **A-only** selections with inference. It freezes
   every primary, sensitivity, no-native selection and quality/source match
   before it invokes `analyze_B.py`.
8. `summarize.py`, `figures.py`, `audit.py`, `write_report.py` produce scene-equal
   bootstrap statistics, all six figures, source identities and the report.
   `coordinate_audit.py` and `ddv2_lineage.py` validate actual frame identities
   and native augmentation parents. `final_review.py` independently checks
   real B hit counts and creates every prespecified sensitivity paired table
   before report generation. `examples.py` records unavailable example types.

The completed primary result is negative: **0 / 1000 nonempty strict PC pools**
at ADE 0.5 m and mass 3%. Operational PC equals Score through method-constraint
fallback. The 1 m sensitivity is reported separately; it does not replace the
primary and has no training evidence. Eight original logs have irregular future
timestamps; their native frame-indexed GT contract and scene identities are
retained and explicitly flagged.

Use `/root/miniconda3/envs/navsim/bin/python` and set
`PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.
The maps are read from `/mnt/navsim/maps`.

The raw bank builder only reads external proposals, the real GT and independent
C. A/B are never candidate material. Four selectors share raw IDs. The current
policy sees only cached observed vision/history/status; raw candidates and
scores are strictly offline inputs to selection and analysis.

Exact duplicates retain every source tag and parent. Natural repeated draws in
A/B remain separate samples. Strict pools and operational fallback slots are
reported separately; no repeated parents or fake tiny perturbation fillers.

Trajectory addresses are `trajectory_path` + `raw_index`, checked against
`content_hash` and the candidate metadata JSON. Big rollouts, model weights,
features, dependency directories and per-candidate analysis caches stay local.

Run tests: `python -m pytest -q tools/analysis/pcmts_grpo_mass_audit/tests`.
Scientific negative results never trigger threshold or scene changes.
