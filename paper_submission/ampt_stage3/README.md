# AMPT evaluation code (anonymous submission)

This archive contains only the test-time code used to evaluate the final AMPT
policy on NAVSIM v1 and NAVSIM v2. It does not contain training code, training
configurations, teacher construction, candidate selection, checkpoints,
predictions, benchmark data, or cached metric outputs.

The release is prepared for double-blind review. Source files and generated
metadata contain no author names, affiliations, email addresses, machine
names, private checkpoint names, private hashes, or local absolute paths.

## Included functionality

- read-only validation of a user-supplied ReCogDrive policy checkpoint;
- shard-independent deterministic single-trajectory inference;
- NAVSIM v1 evaluation through the ReCogDrive/NAVSIM entry point;
- conversion of a trusted local submission to the NAVSIM v2 JSON format;
- official NAVSIM v2 EPDMS scoring, including two-frame extended comfort;
- fail-closed verification of scene counts, scorer revision, metrics, and
  anonymous artifact hashes;
- deterministic packaging and double-blind release auditing.

`configs/release_scope.json` defines this boundary in machine-readable form.
Every launcher is a dry run unless `--execute` is passed.

## Reference protocol

Both benchmarks use `navtest`, one emitted trajectory per scene, and no
metric-based candidate selection or reranking in the packaged inference
adapter. The per-scene seed is derived from base seed `20260726` and the scene
token, so changing the GPU or shard assignment does not change a prediction.

| benchmark | valid scenes | aggregate metric | archived raw value |
|---|---:|---:|---:|
| NAVSIM v1 | 12,138 | PDMS | 0.9145046521395536 |
| NAVSIM v2 | 12,146 | EPDMS | 0.8911677435925488 |

The raw values are the verifier targets. They correspond to the rounded paper
results (91.45 PDMS and 89.1 EPDMS). Component targets and exact scene-count
contracts are stored in `configs/eval_navsim_v1.yaml` and
`configs/eval_navsim_v2.yaml`; they are reference assertions, not fabricated
evaluation rows.

The exact scores require the matching evaluation artifacts, benchmark data,
metric cache, model dependencies, and official scorer revision. Code cannot
make a non-matching weight reproduce a target score. The supplied verifier
therefore exits nonzero on any mismatch instead of silently reporting success.

## Layout

```text
configs/                 sealed v1/v2 evaluation protocols
scripts/                 inference, scoring, and verification launchers
src/ampt_stage3/         deterministic adapter and evaluation utilities
tests/                   unit, scope, and anonymity checks
RELEASE_MANIFEST.json    generated file hashes for the submitted archive
```

## Requirements

Install this overlay inside the public ReCogDrive/NAVSIM environment:

```bash
python -m pip install -e paper_submission/ampt_stage3
```

The following external artifacts are required but intentionally not bundled:

- the evaluation policy checkpoint supplied by the authors;
- the matching InternVL3-2B model or feature cache;
- NAVSIM v1 `navtest` data and metric cache;
- the official NAVSIM v2 checkout and `navtest` metric cache.

By default, scripts assume this directory is located two levels below the
ReCogDrive repository root. For a standalone extracted archive, set
`RECOGDRIVE_ROOT` to the public ReCogDrive checkout.

Validated package versions are listed in `VALIDATED_ENVIRONMENT.md`. Dataset
and model licenses remain governed by their respective projects.

## 1. Validate a supplied checkpoint

This is a read-only structural and SHA-256 check. The receipt contains an
anonymous identifier and never records the input path.

```bash
python -m ampt_stage3.validate_checkpoint \
  --checkpoint /data/evaluation_policy.ckpt \
  --output /data/results/checkpoint_receipt.json
```

If the authors distribute an expected hash separately, seal it with
`--expected-sha256`. Checkpoint names and hashes are deliberately absent from
this code archive.

## 2. NAVSIM v1 evaluation

First inspect the dry-run command:

```bash
CHECKPOINT=/data/evaluation_policy_v1.ckpt \
VLM_CHECKPOINT=/data/InternVL3-2B \
METRIC_CACHE=/data/navsim_v1_metric_cache \
OUTPUT_DIR=/data/results/navsim_v1 \
paper_submission/ampt_stage3/scripts/evaluate_navsim_v1.sh
```

Set the standard NAVSIM environment variables, then add `--execute`:

```bash
OPENSCENE_DATA_ROOT=/data/navsim \
NAVSIM_EXP_ROOT=/data/navsim_runtime \
NUPLAN_MAPS_ROOT=/data/navsim/maps \
CHECKPOINT=/data/evaluation_policy_v1.ckpt \
VLM_CHECKPOINT=/data/InternVL3-2B \
METRIC_CACHE=/data/navsim_v1_metric_cache \
OUTPUT_DIR=/data/results/navsim_v1 \
paper_submission/ampt_stage3/scripts/evaluate_navsim_v1.sh --execute
```

The launcher fixes the ReCogDrive architecture contract used by the submitted
policy and rejects random initialization. `GPUS` defaults to 8.

## 3. Export one trajectory per scene

Use this step when a supplied evaluation checkpoint exposes the standard
ReCogDrive action head:

```bash
OPENSCENE_DATA_ROOT=/data/navsim \
NAVSIM_EXP_ROOT=/data/navsim_runtime \
NUPLAN_MAPS_ROOT=/data/navsim/maps \
CHECKPOINT=/data/evaluation_policy_v2.ckpt \
VLM_CHECKPOINT=/data/InternVL3-2B \
OUTPUT_DIR=/data/results/predictions \
paper_submission/ampt_stage3/scripts/export_single_trajectory_predictions.sh --execute
```

This writes `submission.pkl` and `predictions.json`. The converter accepts only
a trusted locally generated pickle, requires exactly one prediction mapping,
sorts scene tokens, and validates every trajectory as a finite `[8, 3]` array.

If the released v2 evaluation artifact has its own final inference entry point,
run that entry point and provide its one-trajectory `predictions.json` directly
to the scorer below. This scoring archive never substitutes cached paper
predictions for model output.

## 4. Official NAVSIM v2 scoring

```bash
PREDICTIONS_DIR=/data/results/predictions \
NAVSIM_V2_ROOT=/data/official_navsim_v2 \
V2_METRIC_CACHE=/data/navsim_v2_metric_cache \
OUTPUT_DIR=/data/results/navsim_v2 \
paper_submission/ampt_stage3/scripts/evaluate_navsim_v2.sh --execute
```

The scorer writes scene-level CSV output and `summary.json`. Reproduction is
accepted only for the official source revision recorded in the sealed v2
protocol. The summary records the revision and aggregate values, but not local
data or installation paths.

## 5. Verify the two benchmark results

The final verifier treats the v1 and v2 artifacts separately because benchmark
releases may use different serialized containers. `V2_ARTIFACT` may be a file
or a directory; a deterministic tree digest is used for directories.

```bash
V1_CHECKPOINT=/data/evaluation_policy_v1.ckpt \
V2_ARTIFACT=/data/evaluation_artifact_v2 \
V1_RESULTS_CSV=/data/results/navsim_v1/full_navtest_pdms.csv \
V2_SUMMARY_JSON=/data/results/navsim_v2/summary.json \
OUTPUT_FILE=/data/results/reproduction_receipt.json \
paper_submission/ampt_stage3/scripts/verify_reproduction.sh --execute
```

Optional `EXPECTED_V1_SHA256` and `EXPECTED_V2_SHA256` values can be supplied
out of band. The generated receipt contains only anonymous artifact IDs,
digests, input-result digests, protocol values, and pass/fail status.

## Validation and packaging

```bash
make -C paper_submission/ampt_stage3 check
make -C paper_submission/ampt_stage3 package
```

`check` runs unit tests, syntax checks, CLI dry runs, a strict evaluation-only
scope audit, and a double-blind leak scan. `package` creates
`paper_submission/ampt_evaluation_release.zip` with normalized timestamps and
an embedded SHA-256 manifest.

## Scope and reproducibility notes

- No AMPT or ReCogDrive training stage is included.
- No benchmark data, model weight, prediction, or result file is included.
- The v2 scorer uses only the official benchmark implementation and supplied
  trajectories; it does not alter model predictions.
- Eight v1 predictions without corresponding metric-cache entries are not
  included in the 12,138-scene v1 mean. NAVSIM v2 scores all 12,146 entries.
- Pickle input must be locally generated and trusted.
- The benchmark is a non-reactive offline evaluation; this release does not
  remove the limitations of that protocol.
