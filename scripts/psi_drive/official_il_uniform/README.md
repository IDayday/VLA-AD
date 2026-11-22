# ReCogDrive IL with uniform PSI target sampling

The [experiment report](../../../reports/psi_drive/official_il_uniform_20261121/README.md) records the training stages, pinned inputs, and evaluation results. Run these commands from the repository root in the NAVSIM environment with `PYTHONPATH` set to that checkout. Supply the Stage 1 weights, PSI support archive, NAVSIM data and metric caches separately. The report's provenance identifies the exact evaluation inputs and checkpoints.

## Training

Prepare the original PSI archive for uniform sampling. The native planner draws one valid slot with `torch.multinomial` and uses the selected trajectory in the existing IL loss.

```bash
export PYTHONPATH="$PWD"
recipe=scripts/psi_drive/official_il_uniform
python "$recipe/prepare_support.py" "$PSI_SOURCE_INDEX" "$UNIFORM_INDEX"
python "$recipe/train.py" single --stage1 "$STAGE1" --cache "$TRAIN_CACHE" \
    --support "$UNIFORM_INDEX" --output "$SINGLE_RUN"
```

The cache uses the native Stage 1 feature builder and the native GT trajectory target builder. If constructing it, run the following once per disjoint TSV shard. Each line contains a log name and scene token, separated by a tab. For the experiment, the shards covered the full PSI archive and the resulting cache was split by the repository's NAVTRAIN log split.

```bash
CUDA_VISIBLE_DEVICES=0 python "$recipe/cache_worker.py" "$LOG_TOKEN_TSV" "$TRAIN_CACHE" \
    --stage1 "$STAGE1" --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS"
```

For the recorded handoff, use the single-node step 1330 checkpoint at the end of epoch 2. The `dual` stage is the batch-128 two-node throughput trial. Both the trial and the batch-256 stage resume from step 1330. Scale the original resume before launching batch 256. Run the final command on both nodes with their respective `NODE_RANK`.

```bash
python "$recipe/scale_resume.py" "$STEP_1330" "$BATCH256_RESUME"
python "$recipe/train.py" batch256 --stage1 "$STAGE1" --cache "$TRAIN_CACHE" \
    --support "$UNIFORM_INDEX" --output "$DUAL_RUN" --resume "$BATCH256_RESUME" \
    --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
```

Use `--render` to inspect the torchrun command. For socket-based NCCL, set `NCCL_SOCKET_IFNAME` and `NCCL_IB_DISABLE` for the cluster before launching. Use the standard NAVSIM sensor and map environment variables required by the base trainer.

## Evaluation

Copy [eval_config.example.json](eval_config.example.json) outside the repository and set its paths. The Navtest manifest contains current front camera paths, the four ego-history poses, velocity, acceleration and navigation command. The metric index supplies the official metric-cache file for each token. The diagnostic manifest and PSI model metadata come from the historical PSI SFT analysis. The PSI checkpoint is the archived epoch 11 checkpoint. `train_args` is the original Hydra run's resolved `train_args.json`.

Run the identity initialization once per checkpoint. Start one inference worker per shard with a unique `--rank` and a common `--world`. If the shared Stage 1 feature bank already exists, use `--read-features` to reuse its observations. The CPU scorer can start as soon as the inference workers have launched. Use distinct output directories for each checkpoint.

```bash
export RECOGDRIVE_UNIFORM_EVAL_CONFIG=/path/to/config.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$recipe/infer.py" init
CUDA_VISIBLE_DEVICES=0 python "$recipe/infer.py" nav --rank 0 --world 1
python "$recipe/score.py" nav --workers 32
CUDA_VISIBLE_DEVICES=0 python "$recipe/infer.py" diag --rank 0 --world 1
python "$recipe/score.py" diag --workers 32
```

Navtest can reuse a completed PSI prediction bank via the optional `psi_nav_reference` config path and `infer.py nav --historical-psi`. For diagnostics, `infer.py diag --historical-psi` evaluates one model and `score.py diag --historical-psi ARCHIVE_JSON` joins the paired PSI group scores. The optional archive JSON carries the source `path` and `sha256`, `reference_checkpoint_sha256`, and `records` keyed by token. Each token provides status `ok` and four group values for both ADE measures and all three GRPO PDMS measures. Use fractional PDMS in the normalized archive JSON. Summaries use the 0–100 scale. Omit these flags to evaluate both heads directly.

With three complete outputs, generate the step trend report using the same paired populations and PSI baseline.

```bash
python "$recipe/compare_steps.py" --step-37960 "$EVAL_37960" \
    --step-50000 "$EVAL_50000" --step-60000 "$EVAL_60000" --output step_trends.json
```
