#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:-}"
MAX_MANIFEST_SAMPLES="${MAX_MANIFEST_SAMPLES:-128}"
FULL_MANIFEST="${FULL_MANIFEST:-0}"

OUT_DIR="${OUT_ROOT}/preflight"
mkdir -p "${OUT_DIR}"

{
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_CHUNK_NAME_PATTERN=%s\n' "${TRAIN_CHUNK_NAME_PATTERN}"
  printf 'MAX_MANIFEST_SAMPLES=%s\n' "${MAX_MANIFEST_SAMPLES}"
  printf 'FULL_MANIFEST=%s\n' "${FULL_MANIFEST}"
} > "${OUT_DIR}/commands.log"

python "${REPO_ROOT}/scripts/check_last_rd_hydra_config.py" \
  --experiment last_rd_stage1_5 \
  --output "${OUT_DIR}/last_rd_hydra_config_check_stage1_5.json"

python "${REPO_ROOT}/scripts/check_last_rd_hydra_config.py" \
  --experiment last_rd_progressive_sft \
  --output "${OUT_DIR}/last_rd_hydra_config_check_progressive_sft.json"

python "${REPO_ROOT}/scripts/check_last_rd_eval_configs.py" \
  --output "${OUT_DIR}/last_rd_eval_config_check.json"

MANIFEST_ARGS=(
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}"
  --max-samples "${MAX_MANIFEST_SAMPLES}"
  --future-jepa-loss-weight 0.30
  --risk-loss-weight 0.0
  --strict
  --output "${OUT_DIR}/last_rd_cache_manifest_smoke.json"
)
if [[ -n "${TRAIN_CHUNK_NAME_PATTERN}" ]]; then
  MANIFEST_ARGS+=(--chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}")
fi
python "${REPO_ROOT}/scripts/audit_last_rd_cache_manifest.py" "${MANIFEST_ARGS[@]}"

python "${REPO_ROOT}/scripts/run_last_rd_real_batch_smoke.py" \
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}" \
  --max-samples 2 \
  --stage both \
  --output "${OUT_DIR}/last_rd_real_batch_smoke.json"

if [[ "${FULL_MANIFEST}" == "1" ]]; then
  FULL_ARGS=(
    --cache-root "${TRAIN_CHUNK_CACHE_ROOT}"
    --future-jepa-loss-weight 0.30
    --risk-loss-weight 0.0
    --strict
    --output "${OUT_DIR}/last_rd_cache_manifest_full.json"
  )
  if [[ -n "${TRAIN_CHUNK_NAME_PATTERN}" ]]; then
    FULL_ARGS+=(--chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}")
  fi
  python "${REPO_ROOT}/scripts/audit_last_rd_cache_manifest.py" "${FULL_ARGS[@]}"
fi

python - "${OUT_DIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])

def load(name: str):
    with (out_dir / name).open("r", encoding="utf-8") as f:
        return json.load(f)

hydra_stage1 = load("last_rd_hydra_config_check_stage1_5.json")
hydra_prog = load("last_rd_hydra_config_check_progressive_sft.json")
eval_cfg = load("last_rd_eval_config_check.json")
manifest = load("last_rd_cache_manifest_smoke.json")
smoke = load("last_rd_real_batch_smoke.json")
full_manifest_path = out_dir / "last_rd_cache_manifest_full.json"
full_manifest = load("last_rd_cache_manifest_full.json") if full_manifest_path.is_file() else None

ready = all([
    hydra_stage1.get("pass"),
    hydra_prog.get("pass"),
    eval_cfg.get("pass"),
    manifest.get("readiness", {}).get("pass"),
    smoke.get("pass"),
    True if full_manifest is None else full_manifest.get("readiness", {}).get("pass"),
])

teacher = manifest.get("teacher_token_coverage", {})
base = manifest.get("required_base_key_coverage", {})
geom = manifest.get("vggt_geometry_mode_distribution", {})
legacy = manifest.get("high_command_one_hot_legacy_repair", {})
dupes = manifest.get("sample_token_duplicates", {})
patch_count = int(geom.get("patch_fallback", 0))
full_count = int(geom.get("full_geometry", 0))
mostly_patch = patch_count > full_count

lines = [
    "# LaST-RD Round1 Preflight Summary",
    "",
    f"Status: {'READY' if ready else 'NOT READY'}",
    "",
    "## Base Key Coverage",
]
for key, value in sorted(base.items()):
    lines.append(f"- {key}: {value.get('coverage')}")
lines.extend([
    "",
    "## Teacher Token Coverage",
    f"- jepa_context_tokens: {teacher.get('jepa_context_tokens', {}).get('coverage')}",
    f"- jepa_target_tokens: {teacher.get('jepa_target_tokens', {}).get('coverage')}",
    f"- vggt_context_tokens: {teacher.get('vggt_context_tokens', {}).get('coverage')}",
    f"- vggt_target_tokens: {teacher.get('vggt_target_tokens', {}).get('coverage')}",
    "",
    "## Command Shape",
    f"- raw distribution: {manifest.get('high_command_one_hot_shape_distribution')}",
    f"- normalized distribution: {legacy.get('normalized_shape_distribution')}",
    f"- legacy repairable: {legacy.get('num_legacy_4d_repairable')}",
    f"- legacy unrepairable: {legacy.get('num_legacy_4d_unrepairable')}",
    "",
    "## Geometry",
    f"- mode distribution: {geom}",
    f"- patch fallback majority: {mostly_patch}",
    "",
    "## Duplicates",
    f"- duplicate sample_token count: {dupes.get('num_duplicate_tokens')}",
    "",
    "## Real-Batch Smoke",
    f"- pass: {smoke.get('pass')}",
    f"- stage1_5_trainable_scope_pass: {smoke.get('stage1_5_trainable_scope_pass')}",
    f"- get_action_no_future_targets_pass: {smoke.get('get_action_no_future_targets_pass')}",
    "",
    "## Warnings",
])
for warning in manifest.get("readiness", {}).get("high_risk_warnings", []):
    lines.append(f"- {warning}")
if mostly_patch:
    lines.append("- VGGT geometry is mostly patch_fallback; do not claim full geometry distillation.")
if full_manifest is None:
    lines.append("- Full manifest was not run. Set FULL_MANIFEST=1 for the full preflight audit.")

(out_dir / "preflight_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if ready else 1)
PY
