# Full-Test PDMS checkpoint selection

The latest user instruction replaces Validation-based checkpoint selection with
full-Test PDMS selection. Test therefore participates in model selection. The
selected best Test score must not be described as independently held out.

This module is external to the frozen training source. Training checkpoints
retain their original source, asset, optimizer, RNG and exposure identities.
Neither legacy Validation scores nor a legacy selection lock are overwritten.

- Retained runs: H_full, H_gain1, H_mu_teacher, H_mu_student; seed 20261027.
- Selection endpoints: epochs 8, 16, 24, 32, 40.
- Every endpoint: all 12,146 Test scenes, explicit reference noise S0–S4.
- Criterion: maximum scene-equal mean of five-noise PDMS; exact ties choose the
  earlier epoch. Base is reported as a comparison and is not a checkpoint.
- Ordinary endpoint scoring can start as soon as that epoch checkpoint exists.
  Final selection still requires all 40 epochs, 53,640 optimizer updates and
  3,432,520 real scene exposures, plus all five completed full-Test evaluations.

`endpoint_checkpoint(run_dir, epoch)` verifies the immutable checkpoint,
completion SHA, expected training assets, all-rank identity and exact exposure.
`endpoint_candidate(run_dir, epoch)` additionally verifies actual complete
paired trajectories and real PDM components for every scene and noise, the
evaluation cohort, teacher, features, calibration and explicit noise identity.

`select(run_dir)` writes `checkpoints/test_pdms_best.json`.
`lock_suite(results)` writes `TEST_PDMS_SELECTION_LOCK.json` only after all four
runs satisfy the complete budget and endpoint evidence. `verify_selection_lock`
verifies this new evidence; it does not read a legacy Validation lock.

The module reads the main controller's `REQUEST.json` in this directory and
binds its SHA into every final selection and lock. Final files are immutable;
changed evidence requires a distinct protocol rather than an overwrite.

The controller owns evaluator admission and resource scheduling. This module
does not launch, pause, signal or modify any training or evaluation process.

```bash
ZT_AUTO_PROXY=0 OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python -B \
  -m unittest -v test_test_pdms_selector
```

The CPU tests include one actual 12,146-scene native-format paired evidence
file, a missing-noise error with correctly re-sealed artifact hashes,
checkpoint/asset/teacher identity failures, full-budget and four-run gates,
Test ranking and earlier ties, ignored opposite Validation ranking, preserved
legacy evidence, cancelled-seed rejection, NaN rejection and changed lock or
summary evidence. They use synthetic small checkpoint payloads and no GPU.
