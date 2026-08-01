# AMPT supplementary material

This directory contains the standalone appendix and its evidence-first analysis
pipeline for **Aligned Multi-Trajectory Policy Training (AMPT)**. The pipeline
keeps raw experiment outputs read-only and writes every derived artifact below
`supplementary/derived/`, `tables/generated/`, or `figures/generated/`.

The current method names are fixed throughout the supplement:

- Stage 1: **PC-MTS**, Policy-Compatible Multi-Trajectory Supervision;
- Stage 2: **FF-PGRPO**, Feasibility-First Pareto GRPO;
- Stage 3: **APR**, Adaptive Pareto-Guided Policy Refinement.

`Score`, `Pareto`, and `PC-MTS` remain valid data-construction variants. Old
stage labels are not normalized inside raw files; they must be mapped explicitly
in the source manifest so provenance remains visible.

## Evidence contract

The pipeline never invents an **observed** score, scene identifier, training
seed, feasibility flag, or candidate outcome.  Every manuscript value carries
one of three identities:

- **reported**: printed in the submitted paper but lacking a complete local
  scene/run manifest;
- **reproduced**: recomputed from an audited repository artifact;
- **proposed-unrun**: a bounded value chosen to make a missing control
  executable, never presented as an experimental result.

The proposed PC-MTS values (bank 16, K=3, calibration quantile 0.95, eight
correlated perturbations, 6/8 pass threshold) and the three proposed training
seeds are therefore concrete rather than blank, but remain visibly distinct
from measured outcomes.  If a registered scene source is absent,
canonicalization marks the affected analysis `pending`; it does not convert a
proposed value into a measurement.

The default analysis seed is `2027`. It is used only for deterministic bootstrap
resampling and plotting; it is never written into the canonical `seed` column.
All training/evaluation seeds must come from logs, configs, evaluator exports,
or an explicitly audited source-manifest default.

Raw inputs are identified by repository-relative path and SHA-256 hash. Generated
provenance records include the command, software versions, Git revision, input
hashes, output hashes, and analysis seed. No high-cost training is launched by
this directory.

## Directory responsibilities

```text
supplementary/
├── supplementary.tex                 # standalone appendix entry point
├── sections/                         # Appendix A--H LaTeX sections
├── tables/{sources,generated}/       # source material and generated CSV/TeX
├── figures/{sources,generated}/      # source material and PDF/300-dpi PNG
├── scripts/                          # command-line analysis pipeline
├── configs/                          # validation, statistics, table, figure specs
├── raw_manifest/                     # references to immutable raw inputs
├── derived/                          # canonical data, reports, stats, provenance
├── launch/                           # dry-run/new-experiment launch scripts
└── audit/                            # source, naming, discrepancy, evidence audits
```

## Requirements

The analysis layer requires Python 3.9+, NumPy, pandas, PyArrow, Matplotlib, and
SciPy. LaTeX compilation uses `latexmk` when available and otherwise falls back
to `pdflatex` plus `bibtex` when the manuscript requests a bibliography.

The pipeline can be inspected without changing data:

```bash
cd supplementary
./make_supplement.sh --help
python scripts/canonicalize_results.py --help
python scripts/validate_dataset.py --help
python scripts/compute_statistics.py --help
```

## Registering raw scene-level evidence

Prefer `raw_manifest/result_sources.json`. Each enabled source declares its path
or glob and may supply source-specific metadata and column mappings:

```json
{
  "schema_version": 1,
  "sources": [
    {
      "id": "verified_run_id",
      "path": "outputs/verified_run/scene_metrics.csv",
      "enabled": true,
      "defaults": {
        "benchmark": "NAVSIM v1",
        "split": "verified_split_name",
        "method": "AMPT"
      },
      "column_map": {
        "token": "scene_id",
        "pdm_score": "aggregate_score"
      }
    }
  ]
}
```

Values shown above illustrate the manifest shape; they are not active pipeline
inputs. A source without an explicit scene identifier is recorded as
`non_scene_level` and excluded from the canonical dataset. Pickle ingestion is
disabled unless `--allow-pickle` is passed for a trusted local file.

For files already using canonical columns, a simpler path/glob list can be put
in `raw_manifest/result_inputs.txt`. Global emergency mappings can be supplied
with `--column-map`; per-source mappings are preferred because they are easier
to audit.

## Canonical schema and validation

Run:

```bash
./make_supplement.sh canonical
./make_supplement.sh validate
```

Canonical outputs are:

- `derived/canonical_scene_metrics.parquet` (preferred typed dataset);
- `derived/canonical_scene_metrics.csv` (portable mirror);
- `derived/canonicalization_report.md` and provenance JSON.

The table contains the requested scene, benchmark, method, stage, seed, round,
aggregate/component metrics, feasibility/failure/credit, Pareto/reference, and
teacher-lifecycle fields, plus source path and record key. Unmapped diagnostic
columns are preserved after deterministic name normalization.

Validation checks schema, IDs, duplicates, score ranges, state transitions,
positive-credit logic, configured aggregate formulas, expected scene counts,
matched method scene sets, candidate matching, and traceable paper-claim means.
Benchmark formulas and expected counts are deliberately empty in
`configs/validation.json` until an evaluator/config source establishes them.

Validation states are:

- `valid`: observed rows pass every configured hard check;
- `pending`: the schema is valid but no scene-level rows are available;
- `invalid`: an integrity error exists; statistics/tables/figures abort.

Warnings identify missing checks or evidence without turning them into values.

The checked configuration currently enforces 12,138 NAVSIM v1 final scenes,
12,138 paired-anchor scenes, 12,146 NAVSIM v2 scenes, and both 658-row hard-set
methods.  It validates the v1 PDMS and v2 EPDMS formulas and exact scene-set
matching.  Candidate-count matching remains a warning because the paper-run
candidate manifests were not found.

## Statistics

```bash
./make_supplement.sh stats
```

`scripts/compute_statistics.py` produces machine-readable `derived/stats.json`
and a CSV bundle in `derived/statistics/`. Descriptive rows include mean,
across-run standard deviation when at least two observed seeds exist, scene
bootstrap confidence intervals, Q10, CVaR-10, run count, and scene count. Fewer
than three observed seeds are labeled `insufficient` for a stability claim.

Paired comparisons are declared in
`configs/statistical_comparisons.json`, for example:

```json
{
  "id": "verified_method_pair",
  "metric": "aggregate_score",
  "selector_a": {"method": "method A", "stage": "verified stage"},
  "selector_b": {"method": "method B", "stage": "verified stage"},
  "pair_on": ["scene_id", "seed"],
  "higher_is_better": true
}
```

The default is 10,000 paired bootstrap replicates. With multiple shared seeds,
the implementation resamples seeds and then scenes within seed; with one seed it
reports a scene-paired interval explicitly labeled as single-seed. Recovery
rates and positive-credit violation rates use Wilson intervals. Positive-credit
violations are evaluated only for explicitly configured protected-pass fields.

## Tables and figures

```bash
./make_supplement.sh tables
./make_supplement.sh figures
```

Every generated table has a CSV and `.tex` file made from the same dataframe.
Captions state the inference protocol, direction of improvement, and the run and
scene reporting convention. Empty evidence produces a factual no-record table,
not a numeric placeholder.

`scripts/generate_appendix_assets.py` builds the evidence-backed paper figures:
the supervision mismatch, APR progression, v1/v2 metric profiles, the fixed
658-scene transition, its cause decomposition, and the audited qualitative
contact sheet.  Every figure is written as vector PDF and 300-dpi PNG with a
colorblind-friendly palette.  The generic declarative figure driver remains
available for future registered analyses; unavailable data never produces a
blank plot.

The 658-scene adapter is deliberately locked to the historical paper pair:
Scalar GRPO has 367 positive outcomes and the paper recovery checkpoint has
440.  Its transition is 198 persistent failures, 93 repairs, 20 new failures,
and 347 retained successes, for net repair 73.  The later 91.45 checkpoint's
hard-set columns are excluded from canonicalization and used only in the
discrepancy audit.  That final policy appears separately in the reproduced
full-benchmark row and in a clearly labeled qualitative contact sheet; neither
contributes to the 658-scene statistics.

## One-command build

To run the evidence pipeline without requiring the manuscript:

```bash
./make_supplement.sh pipeline
```

To inventory evidence, regenerate all derived data/tables/figures, compile the
standalone PDF, check placeholders/internal paths/references, and write the
final hash manifest:

```bash
./make_supplement.sh all
```

The final manifest is `reproduction_manifest.json`. Run `make help` to override
the analysis seed, bootstrap count, Python executable, or confidence level. For
example, this changes only the resampling seed:

```bash
make pipeline SEED=17 BOOTSTRAP_SAMPLES=10000
```

Do not manually edit files in `tables/generated/`, `figures/generated/`, or
`derived/`; change the registered evidence or declarative configuration and
rerun the pipeline instead.

## Compiling or including the appendix

Standalone compilation is part of `make all` and writes
`supplementary.pdf`.  To include the body in the AAAI manuscript, define the
guard before inputting the entry point from the repository root:

```tex
\def\AMPTMainPaper{1}
\input{supplementary/supplementary.tex}
```

The main manuscript must load `amsmath`, `amssymb`, `booktabs`, `graphicx`,
`url`, and `hyperref`, and must make the six entries in `supplementary.bib`
available to its bibliography.  The guarded mode contributes Appendix A--H
only; it does not add a second document class, title, or bibliography.

## Dry-run experiment controls

No launcher starts training by default.  Each prints the resolved variant,
evidence status, proposed seeds, and resource envelope:

```bash
supplementary/launch/run_p0_pcmts.sh --variant full_pcmts
supplementary/launch/run_p0_ffpgrpo.sh --variant full_ffpgrpo --epochs 10
supplementary/launch/run_p0_apr.sh --variant full_apr --round 1
supplementary/launch/run_p1_diagnostics.sh --suite compatibility_metrics
```

Execution requires both `--execute` and `ALLOW_TRAIN=1`, plus explicit
checkpoint/cache/output roots and a user-supplied command from the restored
audited implementation.  The launchers otherwise exit after preflight.  The
paper-reported resource envelope is 8 A800 GPUs for imitation/FF-PGRPO and the
audited APR branch uses 4 A800 GPUs; wall time and storage were not measured.

## Audited inputs and principal outputs

The active adapters are built from the two final scene exports, the paired v1
anchor, and the historical 658-scene comparison listed in
`raw_manifest/result_sources.json`.  Exact paths, keys, and hashes are recorded
in `audit/result_source_map.csv`, `raw_manifest/result_sources.json`, and
`reproduction_manifest.json`.

The compact source adapters are included in `derived/source_adapters/`, so a
fresh checkout can rebuild the canonical table and manuscript even when the
large raw `outputs/` archive is not distributed.  When all four audited raw
files are present, `make prepare` regenerates and hash-checks the adapters;
otherwise it uses the packaged versions and reports that fallback explicitly.

Principal deliverables are:

- `supplementary.pdf` and the standalone/guarded `supplementary.tex`;
- `derived/canonical_scene_metrics.parquet` plus its CSV mirror;
- `derived/stats.json` and `derived/data_validation_report.md`;
- CSV/LaTeX table pairs in `tables/generated/`;
- vector/300-dpi figure pairs in `figures/generated/`;
- evidence audits, `MISSING_EVIDENCE.md`, and `MAIN_TEXT_SUGGESTIONS.md`.

`make checks` smoke-tests every CLI `--help`, searches final LaTeX for internal
paths/placeholders, rejects undefined citations/references, and refreshes the
hash/software manifest.  It performs no training.
