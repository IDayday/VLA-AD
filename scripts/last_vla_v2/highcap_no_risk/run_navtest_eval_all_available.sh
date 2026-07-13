#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_navtest_eval_${RUN_ID}}"

BASE_NAVTEST_ROOT="${BASE_NAVTEST_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
VJEPA_MODEL_PATH="${VJEPA_MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B}"
VLM_PATH="${VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
VLM_TYPE="${VLM_TYPE:-internvl}"
CONFIG="${CONFIG:-configs/last_vla_v2/last_vla_progressive_bottleneck_highcap_no_risk_eval.yaml}"

A_ROOT="${A_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk}"
B_ROOT="${B_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft_remote/train_remote_retry_20260605T050837Z_peftgeneric_vggt12/serverB_lora_highcap_no_risk}"
B_LORA_ADAPTER_DIR="${B_LORA_ADAPTER_DIR:-${B_ROOT}/vlm_lora_cot_alignment/adapters/vlm_lora}"

NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
NUM_GPUS="${NUM_GPUS:-8}"
CACHE_PROCS_PER_GPU="${CACHE_PROCS_PER_GPU:-8}"
JEPA_SHARDS="${JEPA_SHARDS:-$((NUM_GPUS * CACHE_PROCS_PER_GPU))}"
GEOMETRY_SHARDS="${GEOMETRY_SHARDS:-$((NUM_GPUS * CACHE_PROCS_PER_GPU))}"
LORA_PROCS_PER_GPU="${LORA_PROCS_PER_GPU:-${CACHE_PROCS_PER_GPU}}"
LORA_NUM_SHARDS="${LORA_NUM_SHARDS:-$((NUM_GPUS * LORA_PROCS_PER_GPU))}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-8}"
MIN_CKPT_AGE_SECONDS="${MIN_CKPT_AGE_SECONDS:-30}"
EVAL_PRECISION="${EVAL_PRECISION:-fp32}"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LOG_DIR="${OUT_ROOT}/logs"
CACHE_ROOT="${OUT_ROOT}/cache"
JEPA_ROOT="${CACHE_ROOT}/navtest_jepa128_overlay"
GEOMETRY_ROOT="${CACHE_ROOT}/navtest_geometry192_overlay"
A_NAVTEST_CACHE="${CACHE_ROOT}/serverA_navtest_highcap"
B_NAVTEST_CACHE="${CACHE_ROOT}/serverB_lora_navtest_highcap"
B_LORA_SHARD_ROOT="${CACHE_ROOT}/serverB_lora_navtest_highcap_shards"
CHECKPOINT_SNAPSHOT_ROOT="${OUT_ROOT}/checkpoint_snapshot"
MANIFEST_JSONL="${OUT_ROOT}/eval_manifest.jsonl"
EVAL_ROOT="${OUT_ROOT}/eval"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
PROGRESS_LOG="${OUT_ROOT}/progress.log"

mkdir -p "${LOG_DIR}" "${CACHE_ROOT}" "${EVAL_ROOT}"
cd "${REPO_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_LOG}"
}

on_err() {
  local rc=$?
  log "aborted rc=${rc} line=${BASH_LINENO[0]}"
  exit "${rc}"
}
trap on_err ERR

quote_cmd() {
  printf '%q ' "$@" >>"${COMMANDS_LOG}"
  printf '\n' >>"${COMMANDS_LOG}"
}

wait_stage() {
  local stage="$1"
  shift
  local failed=0
  local name pid rc
  for item in "$@"; do
    name="${item%%:*}"
    pid="${item##*:}"
    if wait "${pid}"; then
      log "${stage} ${name} completed"
    else
      rc=$?
      log "${stage} ${name} failed rc=${rc}"
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    log "${stage} failed; see ${LOG_DIR}"
    exit 1
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_path "${BASE_NAVTEST_ROOT}" "base navtest cache"
require_path "${METRIC_CACHE_DIR}" "metric cache"
require_path "${VJEPA_MODEL_PATH}" "V-JEPA2 model"
require_path "${VGGT_MODEL_PATH}" "VGGT model"
require_path "${VLM_PATH}" "VLM checkpoint"
require_path "${B_LORA_ADAPTER_DIR}" "B LoRA adapter dir"
require_path "${CONFIG}" "eval config"

log "out_root=${OUT_ROOT}"
log "collecting current highcap/no-risk progressive checkpoints"
"${PYTHON_BIN}" - "${A_ROOT}" "${B_ROOT}" "${CHECKPOINT_SNAPSHOT_ROOT}" "${MANIFEST_JSONL}" "${MIN_CKPT_AGE_SECONDS}" <<'PY'
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

a_root = Path(sys.argv[1])
b_root = Path(sys.argv[2])
snapshot_root = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
min_age = int(sys.argv[5])
now = time.time()

lines = [
    ("serverA_frozen_highcap_no_risk", a_root),
    ("serverB_lora_highcap_no_risk", b_root),
]

def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)

def candidate_paths(root: Path):
    progressive = root / "progressive_bottleneck"
    patterns = [
        "step_*.ckpt",
        "latest.ckpt",
        "last.ckpt",
        "lightning_logs/version_0/checkpoints/*.ckpt",
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(progressive.glob(pattern)):
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield path

rows = []
skipped = []
for line_name, root in lines:
    for src in candidate_paths(root):
        age = now - src.stat().st_mtime
        if age < min_age:
            skipped.append({"line": line_name, "checkpoint": str(src), "reason": f"age_seconds={age:.1f}"})
            continue
        if "lightning_logs" in src.parts:
            label = "lightning_" + src.name
        else:
            label = "top_" + src.name
        dst_dir = snapshot_root / line_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / safe(label)
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        rows.append(
            {
                "line": line_name,
                "checkpoint_name": dst.stem,
                "checkpoint": str(dst),
                "source_checkpoint": str(src),
                "source_size": src.stat().st_size,
                "source_mtime": src.stat().st_mtime,
            }
        )

if not rows:
    raise SystemExit("No stable progressive checkpoints found for current highcap/no-risk roots.")

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
(manifest_path.parent / "eval_manifest_skipped.json").write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"manifest": str(manifest_path), "num_checkpoints": len(rows), "skipped_recent": len(skipped)}, indent=2))
PY

EXPECTED_NAVTEST_SAMPLES="$("${PYTHON_BIN}" - "${BASE_NAVTEST_ROOT}" "${NAVTEST_CHUNK_NAME_PATTERN}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pattern = sys.argv[2]
count = 0
seen = set()
for item in pattern.split(","):
    for chunk in sorted(root.glob(item.strip())):
        index = chunk / "index.jsonl"
        if not index.is_file():
            continue
        with index.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                token = str(row.get("sample_token") or Path(row["path"]).stem)
                if token in seen:
                    raise SystemExit(f"duplicate token in base navtest cache: {token}")
                seen.add(token)
                count += 1
print(count)
PY
)"
log "base navtest samples=${EXPECTED_NAVTEST_SAMPLES}"

log "building JEPA128 navtest overlay with ${JEPA_SHARDS} shards"
mkdir -p "${JEPA_ROOT}" "${LOG_DIR}/jepa128"
declare -a stage_pids=()
for shard in $(seq 0 $((JEPA_SHARDS - 1))); do
  gpu=$((shard % NUM_GPUS))
  shard_name="navtest_jepa128_shard_$(printf '%02d' "${shard}")_of_${JEPA_SHARDS}"
  shard_dir="${JEPA_ROOT}/${shard_name}"
  shard_log="${LOG_DIR}/jepa128/${shard_name}.log"
  mkdir -p "${shard_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/build_recogdrive_jepa_overlay_from_chunks.py
    --base-chunk-root "${BASE_NAVTEST_ROOT}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
    --output-dir "${shard_dir}"
    --split navtest
    --jepa-model-path "${VJEPA_MODEL_PATH}"
    --num-jepa-tokens 128
    --strict-highcap-jepa
    --precision bf16
    --device cuda
    --shard-index "${shard}"
    --num-shards "${JEPA_SHARDS}"
    --strict-coverage
    --log-every 100
  )
  if [[ -n "${CACHE_MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${CACHE_MAX_SAMPLES}"); fi
  quote_cmd env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${cmd[@]}"
  ) >"${shard_log}" 2>&1 &
  child_pid=$!
  stage_pids+=("${shard_name}:${child_pid}")
  log "jepa128 launched ${shard_name} gpu=${gpu} pid=${child_pid}"
done
wait_stage "jepa128" "${stage_pids[@]}"
if [[ "${STOP_AFTER_STAGE:-}" == "jepa128" ]]; then
  log "STOP_AFTER_STAGE=jepa128 requested"
  exit 0
fi

log "building VGGT geometry192 navtest overlay with ${GEOMETRY_SHARDS} shards"
mkdir -p "${GEOMETRY_ROOT}" "${LOG_DIR}/geometry192"
stage_pids=()
for shard in $(seq 0 $((GEOMETRY_SHARDS - 1))); do
  gpu=$((shard % NUM_GPUS))
  shard_name="full_geometry_overlay_shard_$(printf '%02d' "${shard}")_of_${GEOMETRY_SHARDS}"
  shard_dir="${GEOMETRY_ROOT}/${shard_name}"
  shard_log="${LOG_DIR}/geometry192/${shard_name}.log"
  mkdir -p "${shard_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/build_last_vla_full_geometry_cache.py
    --chunk-cache-root "${BASE_NAVTEST_ROOT}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
    --output-cache-root "${shard_dir}"
    --vggt-model-path "${VGGT_MODEL_PATH}"
    --precision fp32
    --device cuda
    --num-geometry-tokens 192
    --geometry-grid-rows 12
    --geometry-grid-cols 16
    --geometry-teacher-dim 512
    --require-full-geometry
    --shard-index "${shard}"
    --num-shards "${GEOMETRY_SHARDS}"
  )
  if [[ -n "${CACHE_MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${CACHE_MAX_SAMPLES}"); fi
  quote_cmd env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${cmd[@]}"
  ) >"${shard_log}" 2>&1 &
  child_pid=$!
  stage_pids+=("${shard_name}:${child_pid}")
  log "geometry192 launched ${shard_name} gpu=${gpu} pid=${child_pid}"
done
wait_stage "geometry192" "${stage_pids[@]}"
if [[ "${STOP_AFTER_STAGE:-}" == "geometry192" ]]; then
  log "STOP_AFTER_STAGE=geometry192 requested"
  exit 0
fi

log "merging A navtest highcap cache"
mkdir -p "${A_NAVTEST_CACHE}"
cmd=(
  "${PYTHON_BIN}" scripts/merge_last_vla_geometry_cache_into_chunks.py
  --base-chunk-root "${BASE_NAVTEST_ROOT}"
  --geometry-cache-root "${GEOMETRY_ROOT}"
  --geometry-chunk-name-pattern "full_geometry_overlay_shard_*_of_*"
  --jepa-cache-root "${JEPA_ROOT}"
  --jepa-chunk-name-pattern "navtest_jepa128_shard_*_of_*"
  --output-chunk-root "${A_NAVTEST_CACHE}"
  --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
  --copy-mode hardlink
  --strict-coverage
  --min-coverage 0.99
  --strict-jepa-coverage
  --min-jepa-coverage 0.99
  --expected-jepa-tokens 128
  --jepa-dim 1024
  --num-geometry-tokens 192
  --geometry-grid-rows 12
  --geometry-grid-cols 16
  --geometry-teacher-dim 512
)
quote_cmd "${cmd[@]}"
"${cmd[@]}" >"${LOG_DIR}/merge_A_navtest_highcap.log" 2>&1

log "auditing A navtest highcap cache"
cmd=(
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py
  --cache-root "${A_NAVTEST_CACHE}"
  --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
  --strict-full-geometry
  --min-full-geometry-coverage 0.99
  --expected-jepa-tokens 128
  --expected-geometry-tokens 192
  --geometry-teacher-dim 512
  --strict-no-risk
  --output "${OUT_ROOT}/A_navtest_highcap_manifest.json"
)
quote_cmd "${cmd[@]}"
"${cmd[@]}" >"${LOG_DIR}/audit_A_navtest_highcap.log" 2>&1

log "building B LoRA navtest hidden cache with ${LORA_NUM_SHARDS} shards (${LORA_PROCS_PER_GPU}/gpu)"
mkdir -p "${B_LORA_SHARD_ROOT}" "${B_NAVTEST_CACHE}/samples" "${LOG_DIR}/lora_navtest"
stage_pids=()
for shard in $(seq 0 $((LORA_NUM_SHARDS - 1))); do
  gpu=$((shard / LORA_PROCS_PER_GPU))
  shard_name="lora_navtest_shard_$(printf '%02d' "${shard}")_of_${LORA_NUM_SHARDS}"
  shard_dir="${B_LORA_SHARD_ROOT}/${shard_name}"
  shard_log="${LOG_DIR}/lora_navtest/${shard_name}.log"
  mkdir -p "${shard_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py
    --base-chunk-root "${A_NAVTEST_CACHE}"
    --output-chunk-root "${shard_dir}"
    --vlm-path "${VLM_PATH}"
    --vlm-type "${VLM_TYPE}"
    --vlm-lora-adapter-dir "${B_LORA_ADAPTER_DIR}"
    --precision bf16
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
    --batch-size 1
    --shard-index "${shard}"
    --num-shards "${LORA_NUM_SHARDS}"
    --cache-variant highcap_no_risk_navtest
  )
  if [[ -n "${CACHE_MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${CACHE_MAX_SAMPLES}"); fi
  quote_cmd env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${cmd[@]}"
  ) >"${shard_log}" 2>&1 &
  child_pid=$!
  stage_pids+=("${shard_name}:${child_pid}")
  log "lora_navtest launched ${shard_name} gpu=${gpu} pid=${child_pid}"
done
wait_stage "lora_navtest" "${stage_pids[@]}"
if [[ "${STOP_AFTER_STAGE:-}" == "lora_navtest" ]]; then
  log "STOP_AFTER_STAGE=lora_navtest requested"
  exit 0
fi

log "merging B LoRA navtest cache"
"${PYTHON_BIN}" - "${B_LORA_SHARD_ROOT}" "${B_NAVTEST_CACHE}" "${EXPECTED_NAVTEST_SAMPLES}" "${LORA_NUM_SHARDS}" <<'PY' >"${LOG_DIR}/merge_B_lora_navtest.log" 2>&1
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

with (final_root / "index.jsonl").open("w", encoding="utf-8") as f:
    for row in sorted(rows, key=lambda item: item["sample_token"]):
        f.write(json.dumps(row, sort_keys=True) + "\n")

if metadata is None:
    metadata = {}
metadata.update(
    {
        "version": "recogdrive_hidden_cache_with_lora_v1",
        "cache_variant": "highcap_no_risk_navtest",
        "cache_hidden_state_regenerated": True,
        "hidden_cache_source": "vlm_lora_regenerated",
        "num_samples": len(rows),
        "num_shards": num_shards,
        "shard_index": None,
        "merge_source": str(shard_root),
        "merged_cache": True,
    }
)
(final_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"merged_rows": len(rows), "final_root": str(final_root)}, indent=2))
PY

log "auditing B LoRA navtest cache"
cmd=(
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py
  --cache-root "${B_NAVTEST_CACHE}"
  --strict-full-geometry
  --min-full-geometry-coverage 0.99
  --expected-jepa-tokens 128
  --expected-geometry-tokens 192
  --geometry-teacher-dim 512
  --strict-no-risk
  --output "${OUT_ROOT}/B_lora_navtest_highcap_manifest.json"
)
quote_cmd "${cmd[@]}"
"${cmd[@]}" >"${LOG_DIR}/audit_B_lora_navtest_highcap.log" 2>&1

log "launching all checkpoint eval jobs simultaneously"
mkdir -p "${LOG_DIR}/eval"
stage_pids=()
idx=0
while IFS= read -r row; do
  [[ -z "${row}" ]] && continue
  line="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.stdin.read())["line"])' <<<"${row}")"
  ckpt="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.stdin.read())["checkpoint"])' <<<"${row}")"
  name="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.stdin.read())["checkpoint_name"])' <<<"${row}")"
  source_ckpt="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.stdin.read())["source_checkpoint"])' <<<"${row}")"
  gpu=$((idx % EVAL_NUM_GPUS))
  out_dir="${EVAL_ROOT}/${line}/${name}"
  mkdir -p "${out_dir}"
  printf '%s\n' "${row}" >"${out_dir}/checkpoint_manifest.json"
  if [[ "${line}" == "serverB_lora_highcap_no_risk" ]]; then
    cache_args=(--chunk-cache-dir "${B_NAVTEST_CACHE}")
  else
    cache_args=(--chunk-cache-root "${A_NAVTEST_CACHE}" --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}")
  fi
  cmd=(
    "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONFIG}"
    --checkpoint "${ckpt}"
    "${cache_args[@]}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision "${EVAL_PRECISION}"
    --output-dir "${out_dir}"
  )
  quote_cmd env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  log "eval launch line=${line} gpu=${gpu} checkpoint=${name} source=${source_ckpt}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${cmd[@]}"
  ) >"${out_dir}/eval.log" 2>&1 &
  child_pid=$!
  stage_pids+=("${line}_${name}:${child_pid}")
  log "eval launched pid=${child_pid} line=${line} checkpoint=${name}"
  idx=$((idx + 1))
done <"${MANIFEST_JSONL}"
wait_stage "eval" "${stage_pids[@]}"

log "aggregating navtest eval metrics"
"${PYTHON_BIN}" - "${EVAL_ROOT}" "${OUT_ROOT}" <<'PY' >"${LOG_DIR}/aggregate_eval.log" 2>&1
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
out_root = Path(sys.argv[2])
rows = []
for metrics_path in sorted(eval_root.glob("*/*/metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = metrics_path.parent / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    row = {
        "line": manifest.get("line", metrics_path.parents[1].name),
        "checkpoint_name": manifest.get("checkpoint_name", metrics_path.parent.name),
        "source_checkpoint": manifest.get("source_checkpoint", metrics.get("checkpoint")),
        "num_samples": metrics.get("num_samples"),
        "num_pdm_valid": metrics.get("num_pdm_valid"),
        "num_pdm_missing_metric_cache": metrics.get("num_pdm_missing_metric_cache"),
        "num_pdm_failed": metrics.get("num_pdm_failed"),
        "trajectory_l1": metrics.get("trajectory_l1"),
        "PDMS": metrics.get("PDMS"),
        "NC": metrics.get("NC"),
        "DAC": metrics.get("DAC"),
        "TTC": metrics.get("TTC"),
        "comfort": metrics.get("comfort"),
        "EP": metrics.get("EP"),
        "DDC": metrics.get("DDC"),
        "metrics_path": str(metrics_path),
    }
    rows.append(row)

if not rows:
    raise SystemExit("No metrics.json files found.")

fields = [
    "line",
    "checkpoint_name",
    "PDMS",
    "trajectory_l1",
    "num_samples",
    "num_pdm_valid",
    "num_pdm_missing_metric_cache",
    "num_pdm_failed",
    "NC",
    "DAC",
    "TTC",
    "comfort",
    "EP",
    "DDC",
    "source_checkpoint",
    "metrics_path",
]
summary_csv = out_root / "navtest_eval_summary.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (item["line"], -(item["PDMS"] or -1), item["checkpoint_name"])):
        writer.writerow({key: row.get(key) for key in fields})

best = {}
for row in rows:
    key = row["line"]
    if row.get("PDMS") is None:
        continue
    if key not in best or row["PDMS"] > best[key]["PDMS"]:
        best[key] = row

summary_json = out_root / "navtest_eval_summary.json"
summary_json.write_text(json.dumps({"rows": rows, "best_by_line": best}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Highcap No-Risk Navtest Evaluation",
    "",
    f"Total evaluated checkpoints: {len(rows)}",
    "",
    "## Best By Line",
    "",
    "| line | checkpoint | PDMS | L1 | valid PDM | missing metric |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for line, row in sorted(best.items()):
    lines.append(
        f"| {line} | {row['checkpoint_name']} | {row.get('PDMS')} | {row.get('trajectory_l1')} | "
        f"{row.get('num_pdm_valid')} | {row.get('num_pdm_missing_metric_cache')} |"
    )
lines.extend(["", "## All Checkpoints", "", "| line | checkpoint | PDMS | L1 | valid PDM |", "| --- | --- | ---: | ---: | ---: |"])
for row in sorted(rows, key=lambda item: (item["line"], -(item["PDMS"] or -1), item["checkpoint_name"])):
    lines.append(
        f"| {row['line']} | {row['checkpoint_name']} | {row.get('PDMS')} | "
        f"{row.get('trajectory_l1')} | {row.get('num_pdm_valid')} |"
    )
(out_root / "navtest_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"summary_csv": str(summary_csv), "best_by_line": best}, indent=2, sort_keys=True))
PY

log "navtest eval complete; summary=${OUT_ROOT}/navtest_eval_summary.md"
