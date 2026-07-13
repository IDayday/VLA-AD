#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
REPO_ROOT="${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
FULL_GEOMETRY_CHUNK_ROOT="${FULL_GEOMETRY_CHUNK_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft_remote/train_remote_retry_20260605T050837Z_peftgeneric_vggt12}"
VLM_PATH="${VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
VLM_TYPE="${VLM_TYPE:-internvl}"
MASTER_PORT="${MASTER_PORT:-29841}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
PRECISION="${PRECISION:-bf16}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
LORA_CACHE_BATCH_SIZE="${LORA_CACHE_BATCH_SIZE:-1}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-103036}"
PROCS_PER_GPU="${PROCS_PER_GPU:-8}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_SHARDS="${NUM_SHARDS:-$((NUM_GPUS * PROCS_PER_GPU))}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
validation_overrides=()
if [[ "${SKIP_VALIDATION}" == "1" || "${SKIP_VALIDATION}" == "true" ]]; then
  validation_overrides+=(
    trainer.params.limit_val_batches=0
    trainer.params.check_val_every_n_epoch=999999
  )
fi

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ROOT="${OUT_ROOT}/serverB_lora_highcap_no_risk"
B1="${ROOT}/vlm_lora_cot_alignment"
B2="${ROOT}/lora_hidden_cache_highcap_no_risk"
B3="${ROOT}/progressive_bottleneck"
LOG_DIR="${ROOT}/logs"
SHARD_ROOT="${ROOT}/lora_hidden_cache_highcap_no_risk_shards_${NUM_SHARDS}/$(date -u +%Y%m%dT%H%M%SZ)"
SHARD_LOG_DIR="${LOG_DIR}/cache32_shards_$(basename "${SHARD_ROOT}")"
SUMMARY_LOG="${LOG_DIR}/cache32_supervisor.log"

mkdir -p "${B2}/samples" "${B3}" "${SHARD_ROOT}" "${SHARD_LOG_DIR}" "${LOG_DIR}"
cd "${REPO_ROOT}"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "Starting ${NUM_SHARDS}-shard LoRA hidden cache regeneration"
  echo "repo=${REPO_ROOT}"
  echo "root=${ROOT}"
  echo "base=${FULL_GEOMETRY_CHUNK_ROOT}"
  echo "final_cache=${B2}"
  echo "shard_root=${SHARD_ROOT}"
  echo "num_gpus=${NUM_GPUS} procs_per_gpu=${PROCS_PER_GPU} num_shards=${NUM_SHARDS}"
  echo "skip_validation=${SKIP_VALIDATION}"
} >>"${SUMMARY_LOG}"

declare -a pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard / PROCS_PER_GPU))
  shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "${shard}")"
  shard_log="${SHARD_LOG_DIR}/shard_$(printf '%02d' "${shard}").log"
  mkdir -p "${shard_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py \
      --base-chunk-root "${FULL_GEOMETRY_CHUNK_ROOT}" \
      --output-chunk-root "${shard_dir}" \
      --vlm-path "${VLM_PATH}" \
      --vlm-type "${VLM_TYPE}" \
      --vlm-lora-adapter-dir "${B1}/adapters/vlm_lora" \
      --precision "${PRECISION}" \
      --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}" \
      --batch-size "${LORA_CACHE_BATCH_SIZE}" \
      --shard-index "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --cache-variant highcap_no_risk
  ) >"${shard_log}" 2>&1 &
  pids+=("$!")
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launched shard=${shard} gpu=${gpu} pid=${pids[-1]} log=${shard_log}" >>"${SUMMARY_LOG}"
  sleep "${LAUNCH_STAGGER_SECONDS}"
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "${pid}"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) shard=${i} completed" >>"${SUMMARY_LOG}"
  else
    rc=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) shard=${i} failed rc=${rc}" >>"${SUMMARY_LOG}"
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one cache shard failed; not merging or launching progressive." >>"${SUMMARY_LOG}"
  exit 1
fi

"${PYTHON_BIN}" - "${SHARD_ROOT}" "${B2}" "${EXPECTED_SAMPLES}" "${NUM_SHARDS}" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

shard_root = Path(sys.argv[1])
final_root = Path(sys.argv[2])
expected = int(sys.argv[3])
num_shards = int(sys.argv[4])
final_samples = final_root / "samples"
final_samples.mkdir(parents=True, exist_ok=True)

rows = []
seen = set()
metadata = None
for shard_dir in sorted(p for p in shard_root.iterdir() if p.is_dir()):
    index_path = shard_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing shard index: {index_path}")
    meta_path = shard_dir / "metadata.json"
    if metadata is None and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            token = str(row.get("sample_token") or Path(row["path"]).stem)
            if token in seen:
                raise ValueError(f"Duplicate sample_token during merge: {token}")
            src = shard_dir / row["path"]
            if not src.is_file():
                raise FileNotFoundError(f"Missing shard sample: {src}")
            dst = final_samples / f"{token}.pt"
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            merged = dict(row)
            merged["sample_token"] = token
            merged["path"] = str(Path("samples") / dst.name)
            rows.append(merged)
            seen.add(token)

if len(rows) != expected:
    raise RuntimeError(f"Merged {len(rows)} rows, expected {expected}.")

tmp_index = final_root / ".index.jsonl.cache32.tmp"
with tmp_index.open("w", encoding="utf-8") as f:
    for row in sorted(rows, key=lambda item: item["sample_token"]):
        f.write(json.dumps(row, sort_keys=True) + "\n")
tmp_index.replace(final_root / "index.jsonl")

if metadata is None:
    metadata = {}
metadata.update(
    {
        "version": "recogdrive_hidden_cache_with_lora_v1",
        "cache_variant": "highcap_no_risk",
        "cache_hidden_state_regenerated": True,
        "hidden_cache_source": "vlm_lora_regenerated",
        "num_samples": len(rows),
        "num_skipped_by_shard": 0,
        "num_shards": num_shards,
        "shard_index": None,
        "merge_source": str(shard_root),
        "merged_cache": True,
    }
)
tmp_meta = final_root / ".metadata.json.cache32.tmp"
tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp_meta.replace(final_root / "metadata.json")
print(json.dumps({"merged_rows": len(rows), "final_root": str(final_root), "shard_root": str(shard_root)}, indent=2))
PY

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "Merged cache shards into ${B2}; launching B progressive"
} >>"${SUMMARY_LOG}"

"${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "$((MASTER_PORT + 1))" \
  navsim/planning/script/run_training_recogdrive.py \
  +experiment=last_vla_progressive_bottleneck_highcap_no_risk \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "cache_path=${B2}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  "output_dir=${B3}" \
  "agent.checkpoint_path=${A0_INIT_CHECKPOINT}" \
  "agent.last_vla_adapter_checkpoint=${B1}/adapters/last_vla_cot_adapter.pt" \
  agent.num_jepa_tokens=128 \
  agent.num_dynamic_tokens=128 \
  agent.num_vggt_tokens=12 \
  agent.num_geometry_tokens=192 \
  agent.num_risk_tokens=0 \
  agent.last_vla_cot_num_tokens=192 \
  agent.last_vla_vlm_summary_tokens=64 \
  agent.last_vla_use_risk_head=false \
  agent.last_vla_risk_loss_weight=0.0 \
  agent.last_vla_require_full_geometry=true \
  agent.last_vla_allow_patch_geometry_fallback=false \
  agent.last_vla_geometry_teacher_dim=512 \
  agent.last_vla_geometry_grid_rows=12 \
  agent.last_vla_geometry_grid_cols=16 \
  agent.last_vla_raw_vlm_context_to_dit=false \
  "${validation_overrides[@]}" \
  trainer.params.devices=8 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  >"${LOG_DIR}/progressive_bottleneck.log" 2>&1
