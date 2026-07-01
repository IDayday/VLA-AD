#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:?SUPPLEMENT_ROOT must point to stage2_gt_only_structured_elite_buffer_*}"
ORIGINAL_BUFFER_ROOT="${ORIGINAL_BUFFER_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
TOKEN_TSV="${TOKEN_TSV:-/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_support_gt_only_missing_candidate_tokens_18179.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/psi_drive/support_index}"
NUM_SHARDS="${NUM_SHARDS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUALITY_BAND="${QUALITY_BAND:-0.02}"
MIN_DESCRIPTOR_DISTANCE="${MIN_DESCRIPTOR_DISTANCE:-0.75}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

SUPPLEMENT_BUFFER_ROOT="${SUPPLEMENT_ROOT}/buffer"
UNION_ROOT="${OUTPUT_DIR}/stage2_elite_buffer_union_clean_gt_supplemented_${STAMP}"
OUTPUT_PT="${OUTPUT_DIR}/stage2_pareto_support_clean_full_gt_supplemented_${STAMP}.pt"
OUTPUT_SUMMARY_JSON="${OUTPUT_DIR}/stage2_pareto_support_clean_full_gt_supplemented_${STAMP}_summary.json"
OUTPUT_REPORT_MD="${OUTPUT_DIR}/stage2_pareto_support_clean_full_gt_supplemented_${STAMP}.md"
OUTPUT_AUDIT_JSON="${OUTPUT_DIR}/stage2_pareto_support_clean_full_gt_supplemented_${STAMP}_audit.json"
OUTPUT_AUDIT_MD="${OUTPUT_DIR}/stage2_pareto_support_clean_full_gt_supplemented_${STAMP}_audit.md"
STATE_DIR="${SUPPLEMENT_ROOT}/state"
LOG_DIR="${SUPPLEMENT_ROOT}/logs"

mkdir -p "${OUTPUT_DIR}" "${STATE_DIR}" "${LOG_DIR}"

expected_records="$(( $(wc -l < "${TOKEN_TSV}") - 1 ))"
if [[ "${expected_records}" -le 0 ]]; then
  echo "Invalid expected record count from ${TOKEN_TSV}: ${expected_records}" >&2
  exit 2
fi

echo "waiting_for_supplement root=${SUPPLEMENT_ROOT} expected_records=${expected_records} num_shards=${NUM_SHARDS}"
while true; do
  summary_count="$(find "${SUPPLEMENT_ROOT}/summaries" -maxdepth 1 -type f -name 'shard_*.json' 2>/dev/null | wc -l)"
  record_count="$(find "${SUPPLEMENT_BUFFER_ROOT}" -maxdepth 1 -type f -name '*.pkl.xz' 2>/dev/null | wc -l)"
  alive_count=0
  for pid_file in "${STATE_DIR}"/shard_*.pid; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      state="$(ps -p "${pid}" -o stat= | tr -d ' ')"
      if [[ "${state}" != Z* ]]; then
        alive_count=$((alive_count + 1))
      fi
    fi
  done
  echo "supplement_status summaries=${summary_count}/${NUM_SHARDS} records=${record_count}/${expected_records} alive=${alive_count}"
  if [[ "${summary_count}" -ge "${NUM_SHARDS}" && "${record_count}" -ge "${expected_records}" ]]; then
    break
  fi
  if [[ "${alive_count}" -eq 0 && "${summary_count}" -lt "${NUM_SHARDS}" ]]; then
    echo "Supplement generation has no live shard processes but summaries are incomplete." >&2
    exit 2
  fi
  sleep "${POLL_SECONDS}"
done

if [[ -s "${SUPPLEMENT_ROOT}/errors.jsonl" ]]; then
  echo "Supplement generation produced errors: ${SUPPLEMENT_ROOT}/errors.jsonl" >&2
  tail -n 20 "${SUPPLEMENT_ROOT}/errors.jsonl" >&2
  exit 2
fi

echo "materializing_union output=${UNION_ROOT}"
python "${REPO_ROOT}/scripts/materialize_elite_buffer_union.py" \
  --source-root "${ORIGINAL_BUFFER_ROOT}" \
  --source-root "${SUPPLEMENT_BUFFER_ROOT}" \
  --output-root "${UNION_ROOT}" \
  --overwrite

echo "building_support_index output=${OUTPUT_PT}"
python "${REPO_ROOT}/scripts/build_stage2_pareto_support_index.py" \
  --cache-root "${CACHE_ROOT}" \
  --elite-buffer-root "${UNION_ROOT}" \
  --output-pt "${OUTPUT_PT}" \
  --output-summary-json "${OUTPUT_SUMMARY_JSON}" \
  --output-report-md "${OUTPUT_REPORT_MD}" \
  --source-mode clean \
  --quality-band "${QUALITY_BAND}" \
  --min-descriptor-distance "${MIN_DESCRIPTOR_DISTANCE}" \
  --progress-interval 5000

echo "auditing_support_index index=${OUTPUT_PT}"
python "${REPO_ROOT}/scripts/audit_stage2_pareto_support_index.py" \
  --index "${OUTPUT_PT}" \
  --output-json "${OUTPUT_AUDIT_JSON}" \
  --output-md "${OUTPUT_AUDIT_MD}"

cat > "${STATE_DIR}/supplemented_support_index.env" <<EOF
SUPPORT_INDEX=${OUTPUT_PT}
SUPPORT_SUMMARY_JSON=${OUTPUT_SUMMARY_JSON}
SUPPORT_REPORT_MD=${OUTPUT_REPORT_MD}
SUPPORT_AUDIT_JSON=${OUTPUT_AUDIT_JSON}
SUPPORT_AUDIT_MD=${OUTPUT_AUDIT_MD}
UNION_BUFFER_ROOT=${UNION_ROOT}
EOF

echo "supplemented_support_index_ready ${OUTPUT_PT}"
