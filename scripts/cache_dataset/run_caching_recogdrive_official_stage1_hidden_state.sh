#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD/cache}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
VLM_PATH="${RECOGDRIVE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
CACHE_PATH="${RECOGDRIVE_OFFICIAL_HIDDEN_CACHE_DIR:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
ELITE_TARGET_INDEX_PATH="${ELITE_TARGET_INDEX_PATH:-/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T222308Z/artifacts/stage2_elite_best_valid_above_gt_targets.pt}"
SUPPLEMENT_ELITE_TARGETS="${SUPPLEMENT_ELITE_TARGETS:-1}"
SUPPLEMENTED_CACHE_PATH="${RECOGDRIVE_OFFICIAL_HIDDEN_ELITE_CACHE_DIR:-${CACHE_PATH}_elite_targets}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-recogdrive_official_stage1_hidden_cache}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/recogdrive_official_stage1_hidden_cache.log}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/trainval_navsim_logs/${SPLIT}}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/trainval_sensor_blobs/${SPLIT}}"
FORCE_CACHE_COMPUTATION="${FORCE_CACHE_COMPUTATION:-false}"
SKIP_IF_VALID="${SKIP_IF_VALID:-1}"
AUDIT_MAX_SAMPLES="${AUDIT_MAX_SAMPLES:-16}"
CACHE_ALREADY_VALID=0
SKIP_MARKER="${CACHE_PATH}.skip_generation"

GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS}}"
NODES="${NODES:-$((GPUS / GPUS_PER_NODE))}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}"

if [[ ! -d "${VLM_PATH}" ]]; then
  echo "ReCogDrive official VLM path does not exist: ${VLM_PATH}" >&2
  exit 2
fi
if [[ ! -d "${NAVSIM_LOG_PATH}" ]]; then
  echo "NAVSIM_LOG_PATH does not exist: ${NAVSIM_LOG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${SENSOR_BLOBS_PATH}" ]]; then
  echo "SENSOR_BLOBS_PATH does not exist: ${SENSOR_BLOBS_PATH}" >&2
  exit 2
fi
if [[ -f "${SKIP_MARKER}" ]]; then
  echo "Cache generation skipped by marker: ${SKIP_MARKER}"
  cat "${SKIP_MARKER}"
  exit 0
fi

if [[ "${SKIP_IF_VALID}" == "1" && -d "${CACHE_PATH}" ]]; then
  if python "${REPO_ROOT}/scripts/cache_dataset/audit_recogdrive_official_hidden_cache.py" \
      --cache-path "${CACHE_PATH}" \
      --expected-vlm-path "${VLM_PATH}" \
      --max-samples "${AUDIT_MAX_SAMPLES}" \
      --require-official > "${LOG_FILE%.log}.pre_audit.json"; then
    CACHE_ALREADY_VALID=1
    echo "Existing official ReCogDrive stage1 hidden cache is valid; skip generation."
    cat "${LOG_FILE%.log}.pre_audit.json"
  fi
fi

EXTRA_OVERRIDES=()
if [[ -n "${SCENE_TOKENS:-}" ]]; then
  EXTRA_OVERRIDES+=("train_test_split.scene_filter.tokens=[${SCENE_TOKENS}]")
fi
if [[ -n "${SCENE_LOG_NAMES:-}" ]]; then
  EXTRA_OVERRIDES+=("train_test_split.scene_filter.log_names=[${SCENE_LOG_NAMES}]")
fi
if [[ -n "${MAX_SCENES:-}" ]]; then
  EXTRA_OVERRIDES+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

CMD=(
  torchrun
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py"
  "agent=recogdrive_agent"
  "experiment_name=${EXPERIMENT_NAME}"
  "agent.cam_type=single"
  "agent.cache_hidden_state=True"
  "agent.cache_mode=True"
  "agent.vlm_path=${VLM_PATH}"
  "agent.vlm_type=internvl"
  "agent.dit_type=small"
  "agent.vlm_size=small"
  "agent.train_backbone=false"
  "agent.use_expert_features=false"
  "agent.expert_feature_source=none"
  "agent.allow_expert_target_features=false"
  "agent.use_jepa=false"
  "agent.use_vggt=false"
  "agent.use_last_rd=false"
  "agent.last_rd_stage=disabled"
  "agent.use_last_vla=false"
  "agent.last_vla_stage=disabled"
  "agent.use_two_expert_slots=false"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "cache_path=${CACHE_PATH}"
  "use_cache_without_dataset=false"
  "force_cache_computation=${FORCE_CACHE_COMPUTATION}"
  "${EXTRA_OVERRIDES[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "${LOG_FILE}")"
if [[ "${CACHE_ALREADY_VALID}" != "1" ]]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

  python - "${CACHE_PATH}" "${VLM_PATH}" "${TRAIN_TEST_SPLIT}" "${NAVSIM_LOG_PATH}" "${SENSOR_BLOBS_PATH}" "${SCENE_TOKENS:-}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

cache_path = Path(sys.argv[1])
cache_path.mkdir(parents=True, exist_ok=True)
manifest = {
    "official_recogdrive_stage1_hidden_cache": True,
    "status": "completed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "vlm_path": sys.argv[2],
    "stage1_train_mode": "official",
    "external_knowledge_injection": False,
    "train_test_split": sys.argv[3],
    "navsim_log_path": sys.argv[4],
    "sensor_blobs_path": sys.argv[5],
    "scene_tokens_override": sys.argv[6],
    "feature_file": "internvl_feature.gz",
    "target_file": "trajectory_target.gz",
    "required_feature_keys": [
        "history_trajectory",
        "high_command_one_hot",
        "last_hidden_state",
        "status_feature",
    ],
}
(cache_path / ".recogdrive_official_stage1_hidden_cache.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

  python "${REPO_ROOT}/scripts/cache_dataset/audit_recogdrive_official_hidden_cache.py" \
    --cache-path "${CACHE_PATH}" \
    --expected-vlm-path "${VLM_PATH}" \
    --max-samples "${AUDIT_MAX_SAMPLES}" \
    --require-official
fi

if [[ "${SUPPLEMENT_ELITE_TARGETS}" == "1" ]]; then
  python "${REPO_ROOT}/scripts/cache_dataset/supplement_recogdrive_hidden_cache_with_elite_targets.py" \
    --base-cache-path "${CACHE_PATH}" \
    --output-cache-path "${SUPPLEMENTED_CACHE_PATH}" \
    --elite-target-index-path "${ELITE_TARGET_INDEX_PATH}" \
    --expected-vlm-path "${VLM_PATH}" \
    --overwrite
  python "${REPO_ROOT}/scripts/cache_dataset/audit_recogdrive_official_hidden_cache.py" \
    --cache-path "${SUPPLEMENTED_CACHE_PATH}" \
    --expected-vlm-path "${VLM_PATH}" \
    --max-samples "${AUDIT_MAX_SAMPLES}" \
    --require-official
fi
