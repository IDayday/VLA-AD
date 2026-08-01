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

The scripts never create an experimental score, scene identifier, training or
evaluation seed, feasibility flag, candidate count, or hyperparameter. If no
verified scene-level source is registered, canonicalization emits a typed,
zero-row CSV and Parquet dataset and marks the analysis `pending`. This is a
valid pipeline state, not a result.

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

Figures are declarative (`configs/figure_generation.json`) and support scatter,
line, bar, histogram, stacked-fraction, and failure-transition-matrix plots.
Every successful specification writes a vector PDF and a 300-dpi PNG using a
colorblind-friendly palette. If source rows are unavailable, no blank image is
created; `derived/figure_generation_report.md` records the pending figure.

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
