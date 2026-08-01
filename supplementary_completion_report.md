# AMPT supplementary completion report

## Completed

- Read and audited all nine pages of the submitted paper; normalized 352
  manuscript claims into `supplementary/audit/paper_claims.csv`.
- Built Appendix A--H with detailed PC-MTS, FF-PGRPO, and APR descriptions,
  three implementation-linked algorithms, implementation/evaluation protocol,
  statistical analyses, qualitative cases, related work, and limitations.
- Produced a standalone, anonymous supplementary PDF and a guarded entry point
  that can be included from the AAAI manuscript.
- Built a deterministic command-line pipeline with schema validation, paired
  bootstrap/Wilson intervals, generated CSV/LaTeX tables, vector figures,
  300-dpi PNG mirrors, input/output hashes, software versions, and dry-run
  launch controls.
- Final acceptance checks pass on the 18-page PDF, 19 CSV/LaTeX table pairs,
  six PDF/300-dpi PNG figure pairs, 37,738 canonical rows, and 145 packaged-file
  hashes; no undefined citation/reference, author metadata, or internal absolute
  path enters the manuscript.

## Existing results reused

- NAVSIM v1 final export: 12,138 scored scenes; PDMS 91.4505%, reproducing the
  paper's rounded 91.4 row and its component metrics.
- NAVSIM v2 final export: 12,146 scenes; EPDMS 89.1168%, reproducing the
  paper's rounded 89.1 row and its component metrics.
- Paper-reported supervision comparison: GT 86.4→90.3, Score 87.2→85.8,
  Pareto 87.5→86.7, PC-MTS 86.9→91.1.
- Paper-reported stage ablation and APR round progression.
- Audited APR teacher construction: 39,261 source candidates and 28,910
  accepted post-interpolation/re-evaluation targets.
- Historical fixed 658-scene pair required by the paper: Scalar GRPO 367
  positive outcomes versus AMPT recovery checkpoint 440.

## New analyses generated

- Canonical long table with 37,738 rows and NAVSIM v1/v2 formula checks.
- Fixed-hard-set transition: 198 persistent failures, 93 repairs, 20 new
  failures, and 347 retained successes; net repair 73.
- Recovery-rate Wilson intervals: Scalar GRPO 55.8% [52.0, 59.5] and AMPT
  recovery checkpoint 66.9% [63.2, 70.4].
- Paired binary recovery gain: 11.1 percentage points, scene-bootstrap 95% CI
  [8.1, 14.3], 10,000 resamples, analysis seed 2027.
- Failure-type decomposition: DAC 59/14 repairs/regressions, collision 32/6,
  both 2/0; net changes close exactly to 73.
- Five traceable hard-set cases and a separately labeled final-policy
  camera/BEV contact sheet.

## Experiments still lacking measured data

- Multi-seed Score/Pareto/PC-MTS, Scalar-GRPO/FF-PGRPO, and no-APR/full-APR
  controls.
- Candidate-count/acceptance-matched supervision controls and PC-MTS rollout
  diagnostics.
- Per-rollout positive-credit traces for the full FF-PGRPO mechanism ablation.
- Step-matched APR extra-training, fixed-teacher, interpolation, retention, and
  teacher-source controls.
- End-to-end wall time/storage and the exact scorer worktree SHA for every
  reported table.

Concrete proposed values, configurations, and gated dry-run commands are
provided for each missing control.  They are labeled proposed-unrun and are not
presented as observed outcomes; no expensive training was started.

## Appendix-ready tables and figures

- Generated table pairs are under `supplementary/tables/generated/`.
- Vector and 300-dpi figure pairs are under
  `supplementary/figures/generated/`.
- The compiled appendix is `supplementary/supplementary.pdf`.
- Machine-readable statistics and canonical data are under
  `supplementary/derived/`.

## User confirmation requested before main-paper submission

- Confirm that the historical 367/440 recovery checkpoint should remain the
  named AMPT recovery checkpoint in the main text.  This report deliberately
  does not replace it with the later 91.45 hard-set result.
- Decide whether to keep task-vector merging as an optional APR consolidation
  detail or weaken it in the main text until an isolated ablation is available.
- Decide whether the main paper should expose the explicit reported/
  reproduced/proposed evidence labels used in the supplement.

## Most important main-text changes

1. Replace IPR with APR on PDF page 2.
2. Replace “11.1%” with “11.1 percentage points.”
3. Identify the historical recovery checkpoint and report the 93/20 net
   transition, not only the 440 gross count.
4. Qualify APR's small single-lineage increments as suggestive rather than a
   multi-seed causal isolation.
5. Explain the Table-2/Table-4 GT-only 86.4/86.5 discrepancy and disclose the
   `navtest` checkpoint-selection limitation.
