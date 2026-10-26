#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set an authorized idle GPU UUID before launching this diagnostic}"
if [[ $# -ne 1 ]]; then
  echo 'Usage: CUDA_VISIBLE_DEVICES=GPU-... bash rerun_controlled_visual.sh /absolute/new/output/directory' >&2
  exit 2
fi
root=/mnt/project/recogdrive-ppu-controlled-visual-20261026
export ZT_AUTO_PROXY=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/common_packages:/mnt/project/recogdrive-teacher2-a800-20261026/diagnostics/torch240_packages:/root/miniconda3/envs/ddp/lib/python3.10/site-packages"
export HF_HOME="$root/hf_home" HF_MODULES_CACHE="$root/hf_modules"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$root/tmp" TRITON_CACHE_DIR="$root/triton" MPLCONFIGDIR="$root/mpl"
exec "$root/py31013/bin/python" -B \
  "$root/audit_repo/reports/ppu-a800-alignment-followup-20261027/run_a800_controlled_visual.py" \
  --bundle "$root/bundle/ppu-a800-embedding-audit-20261027-v2" \
  --frozen-repo /mnt/project/-recogdrive-representation-teacher-2 \
  --vlm /mnt/project/recogdrive-representation-teacher-1/local/models/ReCogDrive-VLM-2B \
  --output-dir "$1"
