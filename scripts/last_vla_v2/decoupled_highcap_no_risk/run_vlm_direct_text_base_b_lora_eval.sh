#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON=${PYTHON:-/root/miniconda3/envs/navsim/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}
ADAPTER_ROOT=${ADAPTER_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_vlm_pred_traj_eval_20260608T033900Z/adapters}
OUT_ROOT=${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/direct_text_navtest_eval_$(date -u +%Y%m%dT%H%M%SZ)}
PRECISION=${PRECISION:-bf16}
SAVE_EVERY=${SAVE_EVERY:-250}
MAX_SAMPLES=${MAX_SAMPLES:-}
STAGGER_SECONDS=${STAGGER_SECONDS:-8}

mkdir -p "${OUT_ROOT}/logs"
cd "${REPO_ROOT}"

declare -a RUNS=(
  "base|"
  "B_epoch002|${ADAPTER_ROOT}/B_epoch_002/vlm_lora"
  "B_epoch003|${ADAPTER_ROOT}/B_epoch_003/vlm_lora"
  "B_epoch005|${ADAPTER_ROOT}/B_epoch_005/vlm_lora"
)

declare -a GPUS=(0 1 2 3 4 5 6 7)
declare -a PIDS=()
declare -a NAMES=()
gpu_idx=0

echo "OUT_ROOT=${OUT_ROOT}" | tee "${OUT_ROOT}/commands.log"
for item in "${RUNS[@]}"; do
  IFS='|' read -r run adapter <<< "${item}"
  for shard in 0 1; do
    gpu=${GPUS[$gpu_idx]}
    gpu_idx=$((gpu_idx + 1))
    shard_dir="${OUT_ROOT}/${run}/shard_$(printf '%02d' "${shard}")"
    mkdir -p "${shard_dir}"
    cmd=(
      env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1
      "${PYTHON}" scripts/eval_vlm_direct_text_pdm.py
      --model-path "${MODEL_PATH}"
      --output-dir "${shard_dir}"
      --num-shards 2
      --shard-index "${shard}"
      --precision "${PRECISION}"
      --save-every "${SAVE_EVERY}"
    )
    if [[ -n "${adapter}" ]]; then
      [[ -f "${adapter}/adapter_model.bin" ]] || { echo "missing adapter: ${adapter}/adapter_model.bin" >&2; exit 2; }
      cmd+=(--lora-adapter-dir "${adapter}")
    fi
    if [[ -n "${MAX_SAMPLES}" ]]; then
      cmd+=(--max-samples "${MAX_SAMPLES}")
    fi
    printf '[%s] run=%s shard=%s gpu=%s\n' "$(date -Is)" "${run}" "${shard}" "${gpu}" | tee -a "${OUT_ROOT}/commands.log"
    printf '%q ' "${cmd[@]}" | tee -a "${OUT_ROOT}/commands.log"
    printf '\n' | tee -a "${OUT_ROOT}/commands.log"
    "${cmd[@]}" > "${OUT_ROOT}/logs/${run}_shard_${shard}.log" 2>&1 &
    pid=$!
    PIDS+=("${pid}")
    NAMES+=("${run}_shard_${shard}")
    echo "${pid}" > "${OUT_ROOT}/logs/${run}_shard_${shard}.pid"
    echo "[${run}_shard_${shard}] pid=${pid}" | tee -a "${OUT_ROOT}/commands.log"
    sleep "${STAGGER_SECONDS}"
  done
done

status=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  name="${NAMES[$idx]}"
  if wait "${pid}"; then
    echo "[${name}] exit=0" | tee -a "${OUT_ROOT}/commands.log"
  else
    rc=$?
    echo "[${name}] exit=${rc}" | tee -a "${OUT_ROOT}/commands.log"
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "one or more direct-text eval shards failed; skipping aggregation" >&2
  exit "${status}"
fi

"${PYTHON}" scripts/aggregate_vlm_direct_text_eval.py --out-root "${OUT_ROOT}" \
  --run base --run B_epoch002 --run B_epoch003 --run B_epoch005 \
  > "${OUT_ROOT}/aggregate.log" 2>&1

echo "completed ${OUT_ROOT}"
