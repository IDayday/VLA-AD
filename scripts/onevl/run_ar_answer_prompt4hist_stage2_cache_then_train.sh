#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
VLA_AD_ROOT="${VLA_AD_ROOT:-/mnt/project/VLA-AD}"
ONEVL_ROOT="${ONEVL_ROOT:-/mnt/project/OneVL_training}"

STAGE1_CKPT="${STAGE1_CKPT:-/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133/checkpoint-3228}"
DATA_JSONL="${DATA_JSONL:-/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl}"
TARGET_INDEX="${TARGET_INDEX:-/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-/mnt/navsim/trainval_navsim_logs}"
IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-/mnt/project/OneVL_training}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_dit_${RUN_ID}}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_ROOT}/cache_full103k}"
TRAIN_ROOT="${TRAIN_ROOT:-${OUT_ROOT}/train_full200}"
RUN_TRAIN="${RUN_TRAIN:-0}"

CACHE_LATEST_LINK="${CACHE_LATEST_LINK:-/mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_full103k_prompt4hist_latest}"
TRAIN_LATEST_LINK="${TRAIN_LATEST_LINK:-/mnt/project/onevl_navsim_exp/ar_answer_stage2_train_full200_prompt4hist_latest}"

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DTYPE="${DTYPE:-bfloat16}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1792}"
HIDDEN_MAX_LENGTH="${HIDDEN_MAX_LENGTH:-2800}"
CURRENT_IMAGE_POLICY="${CURRENT_IMAGE_POLICY:-single_or_last}"
PROMPT_SOURCE="${PROMPT_SOURCE:-row}"
PLANNER_SOURCE="${PLANNER_SOURCE:-json}"
INCLUDE_CONTROL_CONVENTION="${INCLUDE_CONTROL_CONVENTION:-0}"
MASTER_PORT="${MASTER_PORT:-29551}"

CONFIG_PATH="${CONFIG_PATH:-/mnt/project/VLA-AD/configs/onevl_ar_answer_stage2_small.yaml}"
GLOBAL_EPOCHS="${GLOBAL_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
LR_ACTION_HEAD="${LR_ACTION_HEAD:-0.0001}"
MIN_LR="${MIN_LR:-0.000001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"
SEED="${SEED:-20260530}"

if [[ ! -d "${STAGE1_CKPT}" ]]; then
  echo "Missing STAGE1_CKPT: ${STAGE1_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${DATA_JSONL}" ]]; then
  echo "Missing DATA_JSONL: ${DATA_JSONL}" >&2
  exit 2
fi
if [[ ! -f "${TARGET_INDEX}" ]]; then
  echo "Missing TARGET_INDEX: ${TARGET_INDEX}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing CONFIG_PATH: ${CONFIG_PATH}" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "Need at least NUM_SHARDS GPUs in GPU_LIST; NUM_SHARDS=${NUM_SHARDS} GPU_LIST=${GPU_LIST}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs" "${CACHE_ROOT}" "${TRAIN_ROOT}"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

record_cmd() {
  {
    printf '[%s] ' "$(date -Is)"
    printf '%q ' "$@"
    printf '\n'
  } >> "${COMMANDS_LOG}"
}

echo "out_root=${OUT_ROOT}"
echo "cache_root=${CACHE_ROOT}"
echo "train_root=${TRAIN_ROOT}"
env | sort | grep -E '^(PYTHON_BIN|STAGE1_CKPT|DATA_JSONL|TARGET_INDEX|NAVSIM_LOG_PATH|IMAGE_BASE_PATH|OUT_ROOT|CACHE_ROOT|TRAIN_ROOT|RUN_TRAIN|NUM_SHARDS|GPU_LIST|DTYPE|MAX_IMAGE_SIZE|HIDDEN_MAX_LENGTH|CURRENT_IMAGE_POLICY|PROMPT_SOURCE|PLANNER_SOURCE|INCLUDE_CONTROL_CONVENTION|MASTER_PORT|CONFIG_PATH|GLOBAL_EPOCHS|BATCH_SIZE|GRAD_ACCUM|NUM_WORKERS|PREFETCH_FACTOR|LR_ACTION_HEAD|MIN_LR|WARMUP_EPOCHS|SAVE_EVERY|LOG_EVERY|SEED)=' \
  > "${OUT_ROOT}/env.snapshot"

echo "== cache generation start $(date -Is) =="
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  shard_name="shard_$(printf '%02d' "${shard}")"
  shard_dir="${CACHE_ROOT}/${shard_name}"
  mkdir -p "${shard_dir}"
  cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/bridge_ar_answer_to_recogdrive_dit.py"
    --model-path "${STAGE1_CKPT}"
    --data-jsonl "${DATA_JSONL}"
    --navsim-log-path "${NAVSIM_LOG_PATH}"
    --image-base-path "${IMAGE_BASE_PATH}"
    --output-dir "${shard_dir}"
    --max-samples 0
    --num-shards "${NUM_SHARDS}"
    --shard-id "${shard}"
    --device cuda:0
    --dtype "${DTYPE}"
    --max-image-size "${MAX_IMAGE_SIZE}"
    --cache-format flat
    --hidden-padding max_length
    --hidden-max-length "${HIDDEN_MAX_LENGTH}"
    --hidden-padding-side left
    --no-hidden-truncation
    --current-image-policy "${CURRENT_IMAGE_POLICY}"
    --prompt-source "${PROMPT_SOURCE}"
    --planner-source "${PLANNER_SOURCE}"
    --target-index-path "${TARGET_INDEX}"
    --target-selection max_weight
    --stage2-support-mode preserve_support
    --dit-type small
    --sampling-method ddim
    --no-dit-forward
  )
  if [[ "${INCLUDE_CONTROL_CONVENTION}" == "1" ]]; then
    cmd+=(--include-control-convention)
  fi
  record_cmd env "CUDA_VISIBLE_DEVICES=${gpu}" "${cmd[@]}"
  (
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${OUT_ROOT}/logs/cache_${shard_name}.log" 2>&1
    status=$?
    echo "${status}" > "${OUT_ROOT}/logs/cache_${shard_name}.status"
    exit "${status}"
  ) &
  pids+=("$!")
  echo "$!" > "${OUT_ROOT}/logs/cache_${shard_name}.pid"
done

cache_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    cache_status=1
  fi
done
echo "${cache_status}" > "${OUT_ROOT}/cache.status"
if [[ "${cache_status}" -ne 0 ]]; then
  echo "cache generation failed; see ${OUT_ROOT}/logs/cache_shard_*.log" >&2
  exit "${cache_status}"
fi

"${PYTHON_BIN}" - "${CACHE_ROOT}" "${DATA_JSONL}" "${STAGE1_CKPT}" "${TARGET_INDEX}" "${HIDDEN_MAX_LENGTH}" "${PLANNER_SOURCE}" > "${OUT_ROOT}/logs/validate_cache.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

import torch

cache_root = Path(sys.argv[1])
data_jsonl = Path(sys.argv[2])
stage1_ckpt = sys.argv[3]
target_index = sys.argv[4]
hidden_max_length = int(sys.argv[5])
planner_source = sys.argv[6]
expected_planner_state_source = "json_prompt_fields" if planner_source != "scene" else "navsim_scene_frames"

expected = sum(1 for line in data_jsonl.open("r", encoding="utf-8") if line.strip())
rows = []
sample_checks = []
total = 0
bad = []
for shard_dir in sorted(cache_root.glob("shard_*")):
    index_path = shard_dir / "index.jsonl"
    metadata_path = shard_dir / "metadata.json"
    if not index_path.is_file():
        bad.append(f"missing index: {index_path}")
        count = 0
    else:
        records = [json.loads(line) for line in index_path.open("r", encoding="utf-8") if line.strip()]
        count = len(records)
        if records:
            sample_path = shard_dir / records[0]["path"]
            sample = torch.load(sample_path, map_location="cpu")
            checks = {
                "chunk": shard_dir.name,
                "path": str(sample_path),
                "last_hidden_state": list(sample["last_hidden_state"].shape),
                "history_trajectory": list(sample["history_trajectory"].shape),
                "status_feature": list(sample["status_feature"].shape),
                "high_command_one_hot": list(sample["high_command_one_hot"].shape),
                "trajectory": list(sample["trajectory"].shape),
                "support_trajectories": list(sample["support_trajectories"].shape),
                "support_mask": list(sample["support_mask"].shape),
                "support_weights": list(sample["support_weights"].shape),
                "support_scores": list(sample["support_scores"].shape),
                "support_missing_mask": list(sample["support_missing_mask"].shape),
                "target_source": sample.get("meta", {}).get("target_source"),
                "stage2_target_mode": sample.get("meta", {}).get("stage2_target_mode"),
                "planner_state_source": sample.get("meta", {}).get("planner_state_source"),
                "planner_source_mode": sample.get("meta", {}).get("planner_source_mode"),
                "hidden_padding": sample.get("meta", {}).get("hidden_padding"),
                "hidden_truncation": sample.get("meta", {}).get("hidden_truncation"),
            }
            sample_checks.append(checks)
            if checks["last_hidden_state"] != [hidden_max_length, 2560]:
                bad.append(f"{sample_path} hidden shape {checks['last_hidden_state']}")
            if checks["history_trajectory"] != [4, 3]:
                bad.append(f"{sample_path} history shape {checks['history_trajectory']}")
            if checks["trajectory"] != [8, 3]:
                bad.append(f"{sample_path} trajectory shape {checks['trajectory']}")
            if checks["support_trajectories"] != [3, 8, 3]:
                bad.append(f"{sample_path} support_trajectories shape {checks['support_trajectories']}")
            if checks["support_mask"] != [3] or checks["support_weights"] != [3] or checks["support_scores"] != [3]:
                bad.append(f"{sample_path} support field shape mismatch")
            if checks["support_missing_mask"] != []:
                bad.append(f"{sample_path} support_missing_mask shape {checks['support_missing_mask']}")
            if checks["target_source"] != "stage2_pareto_support":
                bad.append(f"{sample_path} target_source {checks['target_source']}")
            if checks["stage2_target_mode"] != "preserve_support":
                bad.append(f"{sample_path} stage2_target_mode {checks['stage2_target_mode']}")
            if checks["planner_state_source"] != expected_planner_state_source:
                bad.append(f"{sample_path} planner_state_source {checks['planner_state_source']}")
            if checks["planner_source_mode"] != planner_source:
                bad.append(f"{sample_path} planner_source_mode {checks['planner_source_mode']}")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    for key, value in {
        "model_path": stage1_ckpt,
        "data_jsonl": str(data_jsonl),
        "target_index_path": target_index,
        "hidden_padding": "max_length",
        "hidden_padding_side": "left",
        "stage2_support_mode": "preserve_support",
        "planner_source_mode": planner_source,
    }.items():
        if metadata.get(key) != value:
            bad.append(f"{metadata_path} {key}: {metadata.get(key)!r} != {value!r}")
    if int(metadata.get("hidden_max_length", -1)) != hidden_max_length:
        bad.append(f"{metadata_path} hidden_max_length mismatch")
    rows.append({"chunk": shard_dir.name, "count": count, "metadata": str(metadata_path)})
    total += count

summary = {
    "cache_root": str(cache_root),
    "data_jsonl": str(data_jsonl),
    "expected_records": expected,
    "total_records": total,
    "chunks": rows,
    "sample_checks": sample_checks,
    "errors": bad,
}
(cache_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if total != expected:
    raise SystemExit(f"cache count mismatch: total={total} expected={expected}")
if bad:
    raise SystemExit("cache validation failed")
PY

ln -sfn "${CACHE_ROOT}" "${CACHE_LATEST_LINK}"
echo "== cache generation done $(date -Is) =="

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "RUN_TRAIN=${RUN_TRAIN}; stopping after cache generation."
  exit 0
fi

echo "== stage2 train start $(date -Is) =="
cd "${VLA_AD_ROOT}"
train_cmd=(
  torchrun
  --nproc_per_node=8
  --master_port "${MASTER_PORT}"
  "${VLA_AD_ROOT}/scripts/train_recogdrive_expert_chunked.py"
  --config "${CONFIG_PATH}"
  --chunk-cache-root "${CACHE_ROOT}"
  --chunk-name-pattern "shard_*"
  --global-epochs "${GLOBAL_EPOCHS}"
  --flat-global-dataset
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --lr-scheduler official-cosine
  --lr-scheduler-epochs "${GLOBAL_EPOCHS}"
  --lr-warmup-epochs "${WARMUP_EPOCHS}"
  --min-lr "${MIN_LR}"
  --lr-action-head "${LR_ACTION_HEAD}"
  --lr-expert 0.0
  --jepa-align-weight 0.0
  --vggt-align-weight 0.0
  --precision bf16
  --final-check-precision fp32
  --save-every "${SAVE_EVERY}"
  --save-every-epoch
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
  --output-dir "${TRAIN_ROOT}"
)
record_cmd env "PYTHONPATH=${VLA_AD_ROOT}:${PYTHONPATH:-}" "${train_cmd[@]}"
set +e
PYTHONPATH="${VLA_AD_ROOT}:${PYTHONPATH:-}" "${train_cmd[@]}" > "${OUT_ROOT}/logs/train.log" 2>&1
train_status=$?
set -e
echo "${train_status}" > "${OUT_ROOT}/train.status"
ln -sfn "${TRAIN_ROOT}" "${TRAIN_LATEST_LINK}"
echo "stage2 train exited with status ${train_status}" >&2
exit "${train_status}"
