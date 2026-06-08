#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_HIGHCAP_TRAIN_CHUNK_ROOT OUT_ROOT MASTER_PORT VLM_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
export NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
PRECISION="${PRECISION:-bf16}"
LORA_CACHE_BATCH_SIZE="${LORA_CACHE_BATCH_SIZE:-1}"
LORA_CACHE_SKIP_EXISTING="${LORA_CACHE_SKIP_EXISTING:-0}"
LORA_CACHE_REUSE_EXISTING_ROOTS="${LORA_CACHE_REUSE_EXISTING_ROOTS:-}"
LORA_CACHE_PROGRESS_EVERY="${LORA_CACHE_PROGRESS_EVERY:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-8}"
NUM_SHARDS="${NUM_SHARDS:-$((NUM_GPUS * PROCS_PER_GPU))}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WAIT_FOR_ADAPTERS="${WAIT_FOR_ADAPTERS:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
MAX_WAIT_ADAPTER_SECONDS="${MAX_WAIT_ADAPTER_SECONDS:-864000}"
SKIP_VALIDATION="${SKIP_VALIDATION:-1}"
SKIP_HIDDEN_CACHE_AUDIT="${SKIP_HIDDEN_CACHE_AUDIT:-0}"
COPY_MODE="${COPY_MODE:-hardlink}"

ROOT="${OUT_ROOT}/serverB_vlm_lora_decoupled_highcap_no_risk"
B1="${ROOT}/vlm_lora_cot_alignment"
B2="${ROOT}/lora_regenerated_train_hidden_cache"
B4="${ROOT}/progressive_sft_decoupled"
LOG_DIR="${ROOT}/logs"
SHARD_ROOT="${ROOT}/lora_regenerated_train_hidden_cache_shards_${NUM_SHARDS}/${RUN_ID}"
SHARD_LOG_DIR="${LOG_DIR}/lora_cache_shards_${RUN_ID}"
SUMMARY_LOG="${LOG_DIR}/lora_cache_and_progressive_${RUN_ID}.log"
AUDIT_JSON="${LOG_DIR}/lora_hidden_cache_audit_${RUN_ID}.json"
mkdir -p "${LOG_DIR}" "${SHARD_ROOT}" "${SHARD_LOG_DIR}" "${B4}"

validation_overrides=()
if [[ "${SKIP_VALIDATION}" == "1" || "${SKIP_VALIDATION}" == "true" ]]; then
  validation_overrides+=(trainer.params.limit_val_batches=0 trainer.params.check_val_every_n_epoch=999999)
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${SUMMARY_LOG}"
}

adapter_ready() {
  [[ -f "${B1}/adapters/last_vla_cot_adapter.pt" ]] && [[ -d "${B1}/adapters/vlm_lora" ]]
}

wait_for_adapters() {
  if adapter_ready; then
    return 0
  fi
  if [[ "${WAIT_FOR_ADAPTERS}" != "1" && "${WAIT_FOR_ADAPTERS}" != "true" ]]; then
    echo "Missing B1 adapters under ${B1}/adapters." >&2
    exit 3
  fi
  local waited=0
  while ! adapter_ready; do
    if (( waited >= MAX_WAIT_ADAPTER_SECONDS )); then
      echo "Timed out waiting for B1 adapters under ${B1}/adapters." >&2
      exit 3
    fi
    log "waiting for B1 adapters: waited=${waited}s path=${B1}/adapters"
    sleep "${WAIT_SECONDS}"
    waited=$((waited + WAIT_SECONDS))
  done
}

b2_has_content() {
  if [[ ! -e "${B2}" ]]; then
    return 1
  fi
  if [[ -f "${B2}/index.jsonl" || -f "${B2}/metadata.json" ]]; then
    return 0
  fi
  local first_sample first_other
  first_sample="$(find "${B2}/samples" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)"
  if [[ -n "${first_sample}" ]]; then
    return 0
  fi
  first_other="$(find "${B2}" -mindepth 1 -maxdepth 1 ! -name samples -print -quit 2>/dev/null || true)"
  [[ -n "${first_other}" ]]
}

expected_samples() {
  "${PYTHON_BIN}" - "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" "${TRAIN_CHUNK_NAME_PATTERN}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
pattern = sys.argv[2]
paths = []
if (root / "index.jsonl").is_file():
    paths = [root / "index.jsonl"]
else:
    seen = set()
    for part in pattern.split(","):
        part = part.strip()
        if not part:
            continue
        for path in sorted(root.glob(part)):
            index = path / "index.jsonl"
            if index.is_file() and index not in seen:
                paths.append(index)
                seen.add(index)
if not paths:
    raise SystemExit(f"No index.jsonl found under {root} with pattern {pattern!r}.")
total = 0
for index in paths:
    with index.open("r", encoding="utf-8") as f:
        total += sum(1 for line in f if line.strip())
print(total)
PY
}

launch_cache_shards() {
  # B1's launcher pre-creates B2. That empty directory is safe to reuse; real
  # cache content is only index/metadata or files under samples/.
  if b2_has_content; then
    echo "Refusing to overwrite existing non-empty LoRA hidden cache: ${B2}" >&2
    exit 4
  fi
  local expected
  expected="$(expected_samples)"
  log "starting ${NUM_SHARDS}-shard LoRA hidden cache regeneration; expected_samples=${expected}"
  log "base=${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
  log "final_cache=${B2}"
  log "shard_root=${SHARD_ROOT}"

  declare -a pids=()
  local skip_existing_args=()
  if [[ "${LORA_CACHE_SKIP_EXISTING}" == "1" || "${LORA_CACHE_SKIP_EXISTING}" == "true" ]]; then
    skip_existing_args+=(--skip-existing)
  fi
  if [[ -n "${LORA_CACHE_REUSE_EXISTING_ROOTS}" ]]; then
    IFS=',' read -r -a reuse_roots <<< "${LORA_CACHE_REUSE_EXISTING_ROOTS}"
    for reuse_root in "${reuse_roots[@]}"; do
      if [[ -n "${reuse_root}" ]]; then
        skip_existing_args+=(--reuse-existing-root "${reuse_root}")
      fi
    done
  fi
  if [[ "${LORA_CACHE_PROGRESS_EVERY}" != "0" ]]; then
    skip_existing_args+=(--progress-every "${LORA_CACHE_PROGRESS_EVERY}")
  fi
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu shard_name shard_dir shard_log
    gpu=$((shard / PROCS_PER_GPU))
    shard_name="$(printf 'shard_%05d' "${shard}")"
    shard_dir="${SHARD_ROOT}/${shard_name}"
    shard_log="${SHARD_LOG_DIR}/${shard_name}.log"
    mkdir -p "${shard_dir}"
    (
      cd "${REPO_ROOT}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
      exec "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py \
        --base-chunk-root "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" \
        --output-chunk-root "${shard_dir}" \
        --vlm-path "${VLM_PATH}" \
        --vlm-type "${VLM_TYPE:-internvl}" \
        --vlm-lora-adapter-dir "${B1}/adapters/vlm_lora" \
        --precision "${PRECISION}" \
        --device cuda \
        --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}" \
        --batch-size "${LORA_CACHE_BATCH_SIZE}" \
        --shard-index "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --cache-variant decoupled_highcap_no_risk \
        "${skip_existing_args[@]}"
    ) >"${shard_log}" 2>&1 &
    pids+=("$!")
    log "launched cache shard=${shard} gpu=${gpu} pid=${pids[-1]} log=${shard_log}"
    sleep "${LAUNCH_STAGGER_SECONDS}"
  done

  local failed=0
  for i in "${!pids[@]}"; do
    local pid rc
    pid="${pids[$i]}"
    if wait "${pid}"; then
      log "cache shard=${i} completed"
    else
      rc=$?
      log "cache shard=${i} failed rc=${rc}"
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one LoRA hidden cache shard failed; not merging or launching B4." >&2
    exit 5
  fi
  merge_cache_shards "${expected}"
}

merge_cache_shards() {
  local expected="$1"
  mkdir -p "${B2}/samples"
  "${PYTHON_BIN}" - "${SHARD_ROOT}" "${B2}" "${expected}" "${NUM_SHARDS}" "${COPY_MODE}" <<'PY'
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
copy_mode = sys.argv[5]
final_samples = final_root / "samples"
final_samples.mkdir(parents=True, exist_ok=True)

rows = []
seen = set()
metadata = None
missing = []
for shard_idx in range(num_shards):
    shard_dir = shard_root / f"shard_{shard_idx:05d}"
    index_path = shard_dir / "index.jsonl"
    if not index_path.is_file():
        missing.append(str(index_path))
        continue
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
                raise ValueError(f"Duplicate sample_token during LoRA hidden cache merge: {token}")
            src = shard_dir / row["path"]
            if not src.is_file():
                raise FileNotFoundError(f"Missing shard sample: {src}")
            dst = final_samples / f"{token}.pt"
            if dst.exists():
                dst.unlink()
            if copy_mode == "hardlink":
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
            merged = dict(row)
            merged["sample_token"] = token
            merged["path"] = str(Path("samples") / dst.name)
            rows.append(merged)
            seen.add(token)
if missing:
    raise FileNotFoundError("Missing shard indexes:\n" + "\n".join(missing[:20]))
if len(rows) != expected:
    raise RuntimeError(f"Merged {len(rows)} rows, expected {expected}.")

tmp_index = final_root / ".index.jsonl.tmp"
with tmp_index.open("w", encoding="utf-8") as f:
    for row in sorted(rows, key=lambda item: item["sample_token"]):
        f.write(json.dumps(row, sort_keys=True) + "\n")
tmp_index.replace(final_root / "index.jsonl")

if metadata is None:
    metadata = {}
metadata.update(
    {
        "version": "recogdrive_hidden_cache_with_lora_v1",
        "cache_variant": "decoupled_highcap_no_risk",
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
tmp_meta = final_root / ".metadata.json.tmp"
tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp_meta.replace(final_root / "metadata.json")
print(json.dumps({"merged_rows": len(rows), "final_root": str(final_root), "shard_root": str(shard_root)}, indent=2))
PY
  log "merged LoRA hidden cache shards into ${B2}"
}

audit_cache() {
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py \
      --cache-root "${B2}" \
      --chunk-name-pattern "*" \
      --strict-full-geometry \
      --min-full-geometry-coverage 0.99 \
      --expected-jepa-tokens 128 \
      --expected-geometry-tokens 192 \
      --geometry-teacher-dim 512 \
      --strict-no-risk \
      --output "${AUDIT_JSON}"
  ) >"${LOG_DIR}/lora_hidden_cache_audit_${RUN_ID}.log" 2>&1
  log "LoRA hidden cache audit complete: ${AUDIT_JSON}"
}

launch_b4() {
  log "launching B4 progressive SFT: ${B4}"
  (
    cd "${REPO_ROOT}"
    "${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS}" --master_port "${MASTER_PORT}" \
      navsim/planning/script/run_training_recogdrive.py \
      +experiment=last_vla_decoupled_progressive_highcap_no_risk \
      "train_test_split=${TRAIN_TEST_SPLIT}" \
      "cache_path=${B2}" \
      use_cache_without_dataset=true \
      force_cache_computation=false \
      "output_dir=${B4}" \
      "agent.last_vla_adapter_checkpoint=${B1}/adapters/last_vla_cot_adapter.pt" \
      agent.last_vla_condition_mode=decoupled_cot_residual \
      agent.last_vla_cot_bottleneck_mode=false \
      agent.last_vla_raw_vlm_context_to_dit=true \
      agent.last_vla_use_risk_head=false \
      agent.num_jepa_tokens=128 \
      agent.use_vggt=false \
      agent.num_dynamic_tokens=128 \
      agent.num_vggt_tokens=128 \
      agent.num_geometry_tokens=192 \
      agent.num_risk_tokens=0 \
      agent.last_vla_cot_num_tokens=192 \
      agent.last_vla_cot_num_steps=5 \
      agent.last_vla_require_full_geometry=true \
      agent.last_vla_allow_patch_geometry_fallback=false \
      agent.last_vla_geometry_teacher_dim=512 \
      agent.last_vla_geometry_grid_rows=12 \
      agent.last_vla_geometry_grid_cols=16 \
      "${validation_overrides[@]}" \
      trainer.params.devices="${NUM_GPUS}" \
      trainer.params.strategy=ddp_find_unused_parameters_true
  ) >"${LOG_DIR}/progressive_sft_decoupled.log" 2>&1
}

main() {
  cd "${REPO_ROOT}"
  log "B3/B4 decoupled follow-up starting"
  log "root=${ROOT}"
  wait_for_adapters
  log "B1 adapters are ready"
  launch_cache_shards
  if [[ "${SKIP_HIDDEN_CACHE_AUDIT}" == "1" || "${SKIP_HIDDEN_CACHE_AUDIT}" == "true" ]]; then
    log "skipping LoRA hidden cache audit by SKIP_HIDDEN_CACHE_AUDIT=${SKIP_HIDDEN_CACHE_AUDIT}"
  else
    audit_cache
  fi
  launch_b4
}

main "$@"
