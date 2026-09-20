# IL → GRPO / PSI-SFT: 5,000-scene distribution replication

This is a read-only checkpoint diagnostic, with **zero optimizer updates**.
The cohort is frozen before new inference: original random 1,000 scenes plus
4,000 new token-hash-selected scenes, stratified to the canonical Navtrain
command distribution. Both FULL5000 and NEW4000 must be reported. No scene is
selected by checkpoint outcomes. These are internal Navtrain diagnostics, not
untouched test-set evaluation.

Models: official GT-IL, its original GRPO epoch8/step11970 checkpoint (historical
name 90.41), and the recovered PSI candidate-SFT epoch11/step7315 checkpoint.
PSI downstream GRPO checkpoints remain unavailable; no unrelated APR checkpoint
is substituted. The recoverable PSI archived implementation has the provenance
limitation documented in the historical report.

Each model/scene uses four independent groups of 16 under both `get_action`
evaluation and the real `forward_grpo` sampling preamble, with common random
numbers across model/sampler comparisons. G16 is a standardized diagnostic:
the original formal GRPO training used G8. Native sampling is captured before
reward, advantage assignment, or any update. Parameter and buffer hashes must
remain unchanged. The sampler implementation is imported, never reimplemented.

The original 1,000-scene banks remain in their original namespace and retain
their original protocol metadata. Their file hashes are verified and exact
fresh-feature/scoring parity plus numerical sampling parity are required.
Only 4,000 new feature files and 12,000 new model-scene trajectory banks are
generated. Big local caches are ignored by Git.

Run with `/root/miniconda3/envs/navsim/bin/python`:

1. `prepare.py` freezes config, scenes, model/runtime identities.
2. `features.py --smoke`; then torchrun 8 ranks on `features.py`.
3. `audit_5000.py` verifies imported caches and real scalar/batch NAVSIM parity.
4. `orchestrate.py` waits for features, then runs each model across 8 GPUs.
5. `score_5000.py --workers 80 --watch` runs alongside GPU work.
6. `analyze_5000.py`, `figures_5000.py`, `audit_5000.py --final`.
7. `test_5000.py`, `report_5000.py`; commit small outputs and report.

Metrics: XY ADE in meters, centroid displacement relative to **same-sampler**
official IL, conservative feasibility NC/DAC/TTC/DDC all one, NAVSIM-v1 PDMS
converted to 0–100 points. Main width uses all 64 samples; group-of-16 width is
also saved for exact comparison to historical diagnostics. Group minima/maxima
are calculated within G16, averaged over four groups and then equally over scenes.
Paired bootstrap: 3,000 draws over scenes and a separate log-cluster sensitivity.
No sample-level pseudo-replication. Best-safe PDMS is NA for groups without a
safe member, with their frequency reported separately.

Fresh feature extraction processes only the four observation frames. Future
GT and metric caches enter offline evaluation only. New config/outputs are
isolated; old V1/V2/V3/historical artifacts are never rewritten.
