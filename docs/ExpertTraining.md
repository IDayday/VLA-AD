# ReCogDrive-2B Expert-Token Training

This project trains ReCogDrive-2B-IL incrementally on NAVSIM planner-stage data with JEPA/VGGT expert-token conditioning. It does not require the ReCogDrive QA/pretraining datasets and does not require a full 1-2 TB global hidden-state cache.

## Environment

```bash
cd /mnt/project/VLA-AD

export PROJECT_ROOT=/mnt/project/VLA-AD
export NAVSIM_DATA_ROOT=/mnt/navsim
export OPENSCENE_DATA_ROOT=/mnt/navsim/openscene
export NUPLAN_MAPS_ROOT=/mnt/navsim/maps

export CHECKPOINT_ROOT=/mnt/project/VLA-AD/checkpoints
export EXPERIMENT_ROOT=/mnt/project/VLA-AD/experiments/recogdrive_expert
export CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks
export HF_HOME=/mnt/project/VLA-AD/hf_home
export HF_HUB_CACHE=/mnt/project/VLA-AD/hf_home/hub
export HF_XET_CACHE=/mnt/project/VLA-AD/hf_home/xet
export MODELSCOPE_CACHE=/mnt/project/VLA-AD/modelscope_cache
export HF_TOKEN=<optional-token>
```

Proxy variables are optional and must be provided through env or CLI flags, never hardcoded:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
```

## Download Weights

The downloader is resumable by default. It reuses partial local files and the Hugging Face cache (`--resume` is the default), and it does not remove incomplete fragments unless `--clean-partials` is explicitly passed for the selected target. Use `--no-resume` only when debugging a downloader issue. Proxy is disabled by default; use CLI proxy flags only when needed.

List configured targets:

```bash
python scripts/download_required_weights.py \
  --list-targets
```

Inspect local progress:

```bash
python scripts/download_required_weights.py \
  --status \
  --output-root /mnt/project/VLA-AD/checkpoints
```

This writes:

- `/mnt/project/VLA-AD/checkpoints/download_status.json`
- `/mnt/project/VLA-AD/checkpoints/download_status.md`

Mirror-first, no proxy by default. `--source auto` tries ModelScope only when `configs/weights.yaml` contains a non-null `modelscope_repo`; otherwise it goes directly to Hugging Face:

```bash
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source auto \
  --repo-workers 4 \
  --file-workers 8
```

Download one target at a time through an explicit proxy:

```bash
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source huggingface \
  --proxy http://127.0.0.1:7890 \
  --only recogdrive_2b_il \
  --repo-workers 1 \
  --file-workers 8 \
  --timeout 300
```

Try the Hugging Face mirror without proxy. This sets `HF_ENDPOINT=https://hf-mirror.com` unless `--hf-endpoint` is provided:

```bash
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source hf-mirror \
  --only recogdrive_vlm_2b \
  --repo-workers 1 \
  --file-workers 8 \
  --timeout 300
```

Supported target filters:

```bash
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source huggingface \
  --proxy http://127.0.0.1:7890 \
  --only recogdrive_2b_il,recogdrive_vlm_2b \
  --skip vggt_1b \
  --repo-workers 1 \
  --file-workers 8 \
  --timeout 300
```

To clean only incomplete fragments for one selected repo. This scans that target's local checkpoint directory and its matching `HF_HUB_CACHE/models--org--repo` cache directory, not the whole checkpoint tree:

```bash
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source huggingface \
  --proxy http://127.0.0.1:7890 \
  --only recogdrive_2b_il \
  --clean-partials \
  --repo-workers 1 \
  --file-workers 8 \
  --timeout 300
```

Do not use `--clean-partials` as a routine option; it is only for broken fragments and never wipes all checkpoints.

Run the priority plan in gate order:

```bash
python scripts/run_weight_download_priority_plan.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source huggingface \
  --proxy http://127.0.0.1:7890 \
  --timeout 300
```

The priority plan downloads in this order: `recogdrive_2b_il`, `recogdrive_vlm_2b`, `vjepa2`, `vggt_1b`. It writes:

- `/mnt/project/VLA-AD/checkpoints/priority_download_report.md`
- `/mnt/project/VLA-AD/checkpoints/priority_download_report.json`

Dry run:

```bash
python scripts/run_weight_download_priority_plan.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source huggingface \
  --proxy http://127.0.0.1:7890 \
  --timeout 300 \
  --dry-run
```

Required repos are `owl10/ReCogDrive-2B-IL`, `owl10/ReCogDrive-VLM-2B`, `facebook/vjepa2-vitl-fpc64-256`, and `facebook/VGGT-1B`. Add `--include-rl` to the direct downloader only when explicitly preparing RL weights.

Resolve actual checkpoint files after each download attempt:

```bash
python scripts/resolve_weight_paths.py \
  --weights-config configs/weights.yaml \
  --output /mnt/project/VLA-AD/checkpoints/resolved_weight_paths.json

python scripts/check_real_experiment_gate.py
```

The gate script writes:

- `/mnt/project/VLA-AD/experiments/recogdrive_expert/current_gate.md`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/current_gate.json`

Gate meanings:

- `W0`: no complete required weights; only environment/data inspection and download/status work are allowed.
- `W1`: Base-IL complete; checkpoint loading test is allowed.
- `W2`: Base-IL and VLM complete; A0 same-pipeline baseline eval and VLM hidden chunks are allowed.
- `W3`: Base-IL, VLM, and V-JEPA2 complete; JEPA-only real chunk/training/eval is allowed.
- `W4`: all four complete; full JEPA+VGGT pilot and ablations are allowed.

## Check Weights

```bash
python scripts/check_required_weights.py \
  --root /mnt/project/VLA-AD/checkpoints

python scripts/print_required_weight_paths.py \
  --root /mnt/project/VLA-AD/checkpoints \
  --format yaml
```

## Smoke Gate

This is only a computation-flow gate. It does not measure driving performance.

```bash
python scripts/smoke_test_recogdrive_expert_dummy_flow.py \
  --device cuda \
  --use-expert-features \
  --expert-adapter-dim 768

python scripts/smoke_test_recogdrive_agent_dummy_forward.py
```

## Build Debug Chunk

Use 128 samples only for debugging. The first real debug chunk should usually be 1024 samples.

```bash
python scripts/build_recogdrive_chunk_cache.py \
  --data-root /mnt/navsim \
  --project-root /mnt/project/VLA-AD \
  --split navtrain \
  --chunk-index 0 \
  --chunk-size 1024 \
  --output-dir /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/chunk_000000 \
  --build-vlm-hidden \
  --build-jepa \
  --build-vggt \
  --recogdrive-vlm-path /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
  --jepa-model-path /mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256 \
  --vggt-model-path /mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B \
  --precision bf16 \
  --num-gpus 8
```

Validate the chunk:

```bash
python scripts/check_recogdrive_chunk_cache.py \
  --chunk-dir /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/chunk_000000 \
  --require-vlm-hidden \
  --require-targets
```

## Tiny Overfit

```bash
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/recogdrive2b_expert768_chunk.yaml \
  --base-il-checkpoint /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL \
  --chunk-cache-dir /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/chunk_000000 \
  --max-samples 1024 \
  --batch-size 2 \
  --num-steps 500 \
  --train-expert-only \
  --freeze-base-action-head \
  --precision bf16 \
  --debug-overfit \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/tiny_overfit
```


## Full Cache Generation

For paper-scale IL experiments, generate the reusable full expert-token cache before training. The cache runner builds full VLM hidden states plus JEPA and VGGT context/target tokens, skips completed sample files, retries failed chunks, and writes resumable reports after every chunk.

Recommended full-cache root:

```bash
/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
```

Dry-run/status:

```bash
python scripts/run_full_cache_generation.py \
  --output-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --report-root /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation \
  --assume-navtrain-samples 103288 \
  --assume-navtest-samples 12146 \
  --train-chunk-size 4096 \
  --eval-chunk-size 4096 \
  --window-mode valid-fill \
  --num-gpus 8 \
  --workers-per-gpu 2 \
  --dry-run
```

Long-running detached generation:

```bash
setsid bash -lc 'cd /mnt/project/VLA-AD && python scripts/run_full_cache_generation.py \
  --output-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --report-root /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation \
  --assume-navtrain-samples 103288 \
  --assume-navtest-samples 12146 \
  --train-chunk-size 4096 \
  --eval-chunk-size 4096 \
  --window-mode valid-fill \
  --num-gpus 8 \
  --workers-per-gpu 2 \
  --precision bf16 \
  --retries 3 \
  >> /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation/full_cache_generation.nohup.log 2>&1' < /dev/null &
```

Monitor:

```bash
tail -f /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation/full_cache_generation.nohup.log
cat /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation/full_cache_generation_report.md
```

Resume after interruption by running the same command again. Completed chunks are detected from `metadata.json`, `index.jsonl`, and sample file counts, and incomplete chunks are retried without deleting existing sample files.

Full-cache generation now uses persistent workers by default. Each worker loads VLM, JEPA, and VGGT once, then receives one chunk shard after another from the runner, so model weights are not reloaded between chunks. If a persistent worker attempt fails, the runner restarts the worker pool before retrying that chunk. Use `--no-persistent-workers` only when debugging a long-lived CUDA/process state issue and you want the older one-builder-subprocess-per-chunk isolation.

Use `--workers-per-gpu 2` on 80GB GPUs when utilization is low; this launches 16 persistent workers across 8 GPUs. Each worker still owns one VLM, JEPA, and VGGT model copy in memory, so `--workers-per-gpu 3` is close to the memory limit and should only be used after watching `nvidia-smi` during a full chunk. `--workers N` can force an exact total worker count.

To split the full cache across two servers sharing the same disk, run partition 0 on the first server and partition 1 on the second server. Partitioning is by raw NAVSIM token range for each selected split, and each builder receives an exclusive `--chunk-stop`, so valid-fill scanning cannot cross into the other server's range. Reports are automatically written under `partition_00_of_02` and `partition_01_of_02` subdirectories.

First server:

```bash
python scripts/run_full_cache_generation.py \
  --output-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --report-root /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation \
  --assume-navtrain-samples 103288 \
  --assume-navtest-samples 12146 \
  --train-chunk-size 4096 \
  --eval-chunk-size 4096 \
  --window-mode valid-fill \
  --partition-count 2 \
  --partition-index 0 \
  --num-gpus 8 \
  --workers-per-gpu 2 \
  --precision bf16 \
  --retries 3
```

Second server:

```bash
python scripts/run_full_cache_generation.py \
  --output-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --report-root /mnt/project/VLA-AD/experiments/recogdrive_expert/cache_generation \
  --assume-navtrain-samples 103288 \
  --assume-navtest-samples 12146 \
  --train-chunk-size 4096 \
  --eval-chunk-size 4096 \
  --window-mode valid-fill \
  --partition-count 2 \
  --partition-index 1 \
  --num-gpus 8 \
  --workers-per-gpu 2 \
  --precision bf16 \
  --retries 3
```

With `--window-mode valid-fill`, each chunk starts at the previous chunk `chunk_end` and scans forward until it has up to `4096` valid samples. This avoids overlap while keeping chunks as full as possible despite missing sensor frames. Expected full VLM+JEPA+VGGT cache size is about 1.0-2.0 TiB depending on how many raw samples are skipped for missing images.

Training from the full cache should point at the full-cache root and use the train chunk pattern:

```bash
python scripts/train_recogdrive_expert_chunked.py \
  --chunk-cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --chunk-name-pattern 'train_full_chunk_*' \
  ...
```

## Warmup

```bash
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/recogdrive2b_expert768_warmup.yaml \
  --base-il-checkpoint /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL \
  --chunk-cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks \
  --auto-build-next-chunk \
  --delete-old-chunk \
  --num-chunks 8 \
  --epochs-per-chunk 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --train-expert-only \
  --lr-expert 1e-4 \
  --lr-action-head 0 \
  --jepa-align-weight 0.10 \
  --vggt-align-weight 0.10 \
  --precision bf16 \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/warmup
```

## Main IL

```bash
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/recogdrive2b_expert768_il.yaml \
  --resume-from /mnt/project/VLA-AD/experiments/recogdrive_expert/warmup/latest.ckpt \
  --chunk-cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks \
  --auto-build-next-chunk \
  --delete-old-chunk \
  --num-chunks 32 \
  --epochs-per-chunk 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --lr-expert 1e-4 \
  --lr-action-head 2e-5 \
  --jepa-align-weight 0.03 \
  --vggt-align-weight 0.05 \
  --precision bf16 \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/main_il
```

## Evaluation

```bash
python scripts/eval_recogdrive_expert_pdm.py \
  --config configs/recogdrive2b_expert768_eval.yaml \
  --checkpoint /mnt/project/VLA-AD/experiments/recogdrive_expert/main_il/best.ckpt \
  --data-root /mnt/navsim \
  --split navtest \
  --feature-source chunk \
  --chunk-cache-dir /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/chunk_000000 \
  --max-samples 256 \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/eval_navtest_256
```

Aggregate ablations:

```bash
python scripts/aggregate_recogdrive_expert_results.py \
  --input-root /mnt/project/VLA-AD/experiments/recogdrive_expert \
  --output-csv /mnt/project/VLA-AD/experiments/recogdrive_expert/results.csv \
  --output-md /mnt/project/VLA-AD/experiments/recogdrive_expert/results.md
```

## Future Leakage

`jepa_target_tokens`, `vggt_target_tokens`, future `cam_f0` frames, and future teacher features are train-only. `get_action()` uses only context tokens. If target tokens are present during eval, they are ignored and a warning is emitted.

## Dimensional Contract

`expert_adapter_dim=768` is internal to the expert modules. The planner remains `planner_dim=384`, and every token passed into the original ReCogDrive DiT is 384-dim. The small DiT, `ReCogDriveDiffusionPlanner` input/output dimensions, `action_horizon=8`, and `action_dim=3` remain compatible with Base-IL.

## Training Strategy

This is incremental training from the official ReCogDrive-2B-IL action head, not training from scratch. Shape-compatible Base-IL keys load into the modified model; missing expert keys are expected and randomly initialized.
