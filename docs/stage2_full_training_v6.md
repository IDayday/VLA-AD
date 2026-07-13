# Stage2 full-training v6 data contract

## Decision

The v5 evidence-frontier archive is valid for A5 continuation but fails the
data gate for a new randomly initialized Stage2 model. It retains non-GT
supervision in only 11,855 of 103,288 scenes (11.48%). Independent evidence is
therefore a useful confidence signal, but it must not be a hard admission gate
for the full Stage2 teacher distribution.

v6 is built from the raw v5 candidate records with a stricter anti-circularity
contract:

- GT is always the anchor.
- Previous-Stage2 `policy` candidates are excluded.
- A5 policy reachability, neighbor counts, and evidence fields are not used for
  admission or ranking.
- Non-GT candidates must pass the persisted evaluator-valid mask, trajectory
  quality mask, GT-relative ADE/FDE trust region, reward-drop trust region, and
  GT-relative SNSAD mode-distance gate.
- Derived external expansions are excluded.
- The remaining candidates are ranked on a four-objective Pareto front:
  safety, efficiency, GT-relative diversity, and source confidence.
- Family-aware SNSAD farthest-point selection keeps GT plus at most three
  non-GT modes. This is a support cap, not a fixed per-scene quota.

This makes a new Stage2 optimization run independent of an old Stage2
checkpoint while retaining direct DDV2/DriveOR proposals and evaluator-checked
progress, lateral, and timing perturbations.

## Full archive result

Archive:

```text
outputs/sg_fps_v6_full_training_20260713T1648Z/support_v6
```

Validation report:

```text
outputs/sg_fps_v6_full_training_20260713T1648Z/validation.json
```

Static result:

| Item | Result |
|---|---:|
| Unique records | 103,288 / 103,288 |
| Duplicate tokens | 0 |
| Contract errors | 0 |
| Scenes with non-GT support | 96,693 (93.6149%) |
| Scenes with at least two non-GT modes | 45,774 (44.3169%) |
| Eligible non-GT candidates before Pareto/FPS | 439,978 |
| Selected non-GT candidates | 161,525 |
| Selected previous-policy candidates | 0 |
| DDV2 | 12,848 |
| DriveOR | 15,710 |
| Progress | 78,776 |
| Lateral | 17,486 |
| Timing | 36,705 |
| Minimum pairwise SNSAD, median | 0.57535 |
| Minimum pairwise SNSAD, global minimum | 0.400001 |
| Mean selected reward delta vs GT | +0.01403 |

The static full-training gate passes. This is not yet a driving-performance
claim; it means the archive is complete, diverse, policy-independent, and
internally consistent enough to start the real random-initialized run.

## Build and validate

```bash
python scripts/tools/build_sg_fps_v6_full_training_archive.py \
  --input-archive outputs/sg_fps_v5_full_evidence_20260713T1329Z/support_v5 \
  --output-archive outputs/sg_fps_v6_full_training_20260713T1648Z/support_v6

python scripts/tools/validate_sg_fps_v6_archive.py \
  --archive outputs/sg_fps_v6_full_training_20260713T1648Z/support_v6 \
  --output-json outputs/sg_fps_v6_full_training_20260713T1648Z/validation.json \
  --expected-record-count 103288 \
  --workers 32
```

Build FS-Norm from the same selected support distribution:

```bash
python scripts/tools/build_fs_norm_stats.py \
  --support_archive_path outputs/sg_fps_v6_full_training_20260713T1648Z/support_v6 \
  --output_path outputs/sg_fps_v6_full_training_20260713T1648Z/fs_norm/fs_norm_stats_v2_standard.npz \
  --use_robust false
```

## Training contract

Launcher:

```text
scripts/training/sg_fps/run_train_pta_fs_dit_full_training_v6.sh
```

Defaults:

- random initialization; `CHECKPOINT_PATH` is rejected;
- 200 epochs, learning rate `1e-4`, three scheduler warmup epochs;
- non-GT target mass ramps to `0.50` over 40 epochs;
- target difficulty expands from `0.30` to `1.0` over 40 epochs;
- paired timestep/noise draws;
- non-GT residual mass capped at `0.35`;
- two paired targets throughout (GT plus one rotating non-GT mode);
- uniform scene sampling, because v6 already covers 93.61% of scenes;
- trajectory and feasibility auxiliary weights `0.05` and `0.01`.

There is no separate fine-tuning probe. Epochs 1--5 of the actual 200-epoch
job are the dynamic gate and continue in place when they pass. The gate checks:

1. finite loss/gradients and all eight ranks making progress;
2. no archive contract or missing-token failure;
3. a declining diffusion-loss moving average;
4. finite GT and non-GT target losses;
5. post-budget non-GT residual mass no greater than `0.35`;
6. checkpoint production and resumability.

Passing this dynamic gate supports continuing the same job. Final data/model
adequacy is decided only by full training and the fixed evaluation protocol.
