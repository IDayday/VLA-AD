#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON=${PYTHON:-/root/miniconda3/envs/navsim/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}
ADAPTER_ROOT=${ADAPTER_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_vlm_pred_traj_eval_20260608T033900Z/adapters}
OUT_ROOT=${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/direct_text_navtest_eval_8shard_parallel_$(date -u +%Y%m%dT%H%M%SZ)}
PRECISION=${PRECISION:-bf16}
SAVE_EVERY=${SAVE_EVERY:-250}
MAX_SAMPLES=${MAX_SAMPLES:-}
STAGGER_SECONDS=${STAGGER_SECONDS:-8}
POLL_SECONDS=${POLL_SECONDS:-20}
LAUNCHER=${LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/jobs"
cd "${REPO_ROOT}"
echo "$$" > "${OUT_ROOT}/driver.pid"

declare -a RUNS=(
  "base|"
  "B_epoch002|${ADAPTER_ROOT}/B_epoch_002/vlm_lora"
  "B_epoch003|${ADAPTER_ROOT}/B_epoch_003/vlm_lora"
  "B_epoch005|${ADAPTER_ROOT}/B_epoch_005/vlm_lora"
)

jobs_tsv="${OUT_ROOT}/jobs/all_8shard_parallel.tsv"
: > "${jobs_tsv}"

for item in "${RUNS[@]}"; do
  IFS='|' read -r run adapter <<< "${item}"
  for shard in 0 1 2 3 4 5 6 7; do
    shard_name=$(printf "%02d" "${shard}")
    shard_dir="${OUT_ROOT}/${run}/shard_${shard_name}"
    mkdir -p "${shard_dir}"
    cmd="${PYTHON} scripts/eval_vlm_direct_text_pdm.py --model-path ${MODEL_PATH} --output-dir ${shard_dir} --num-shards 8 --shard-index ${shard} --precision ${PRECISION} --save-every ${SAVE_EVERY}"
    if [[ -n "${adapter}" ]]; then
      [[ -f "${adapter}/adapter_model.bin" ]] || { echo "missing adapter: ${adapter}/adapter_model.bin" >&2; exit 2; }
      cmd="${cmd} --lora-adapter-dir ${adapter}"
    fi
    if [[ -n "${MAX_SAMPLES}" ]]; then
      cmd="${cmd} --max-samples ${MAX_SAMPLES}"
    fi
    printf '%s\t%s\t%s\n' "${run}_shard_${shard_name}" "${shard}" "${cmd}" >> "${jobs_tsv}"
  done
done

{
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "driver_pid=$$"
  echo "mode=parallel_all_ckpts_8_shards"
  echo "jobs_tsv=${jobs_tsv}"
  echo "precision=${PRECISION}"
  echo "save_every=${SAVE_EVERY}"
  echo "stagger_seconds=${STAGGER_SECONDS}"
  echo "poll_seconds=${POLL_SECONDS}"
} | tee "${OUT_ROOT}/commands.log"

"${PYTHON}" "${LAUNCHER}" \
  --jobs-tsv "${jobs_tsv}" \
  --out-root "${OUT_ROOT}" \
  --stagger-seconds "${STAGGER_SECONDS}" \
  --poll-seconds "${POLL_SECONDS}"

"${PYTHON}" scripts/aggregate_vlm_direct_text_eval.py --out-root "${OUT_ROOT}" \
  --run base --run B_epoch002 --run B_epoch003 --run B_epoch005 \
  > "${OUT_ROOT}/aggregate.log" 2>&1

echo "completed ${OUT_ROOT}"
