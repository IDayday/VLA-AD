#!/usr/bin/env bash
set -Eeuo pipefail

OUT_ROOT="${OUT_ROOT:-/tmp/last_vla_cache_cleanup}"
mkdir -p "${OUT_ROOT}"
PLAN_JSON="${OUT_ROOT}/cleanup_plan.json"
REPORT_JSON="${OUT_ROOT}/cleanup_report.json"

if [[ -z "${OLD_LAST_VLA_CACHE_ROOTS:-}" ]]; then
  cat >"${PLAN_JSON}" <<'EOF'
{
  "dry_run": true,
  "message": "OLD_LAST_VLA_CACHE_ROOTS is empty; no cache roots were inspected.",
  "candidates": []
}
EOF
  echo "No OLD_LAST_VLA_CACHE_ROOTS set. Wrote ${PLAN_JSON}"
  exit 0
fi

python - "$OLD_LAST_VLA_CACHE_ROOTS" "${PLAN_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

allowed = ("last_vla", "last-vla", "last_vla_v2", "geometry_lite", "patch_fallback", "old_jepa", "minimal", "highcap_old")
forbidden = ("navsim_logs", "sensor_blobs", "metric_cache", "maps", "A0", "a0_official", "checkpoints/a0", "raw")
roots = [Path(item).expanduser().resolve() for item in sys.argv[1].split(":") if item]
candidates = []
for root in roots:
    text = str(root)
    allowed_match = any(token in text for token in allowed)
    forbidden_match = any(token in text for token in forbidden)
    exists = root.exists()
    stat = root.stat() if exists else None
    size = ""
    if exists:
        try:
            import subprocess
            size = subprocess.check_output(["du", "-sh", str(root)], text=True).split()[0]
        except Exception:
            size = "unknown"
    candidates.append(
        {
            "path": text,
            "exists": exists,
            "allowed_name": allowed_match,
            "forbidden_name": forbidden_match,
            "delete_allowed": exists and allowed_match and not forbidden_match,
            "size": size,
            "mtime": None if stat is None else int(stat.st_mtime),
        }
    )
Path(sys.argv[2]).write_text(json.dumps({"dry_run": True, "candidates": candidates}, indent=2, sort_keys=True) + "\n")
PY

cat "${PLAN_JSON}"

if [[ "${CLEAR_OLD_CACHE:-0}" != "1" ]]; then
  echo "CLEAR_OLD_CACHE is not 1; dry-run only."
  exit 0
fi
if [[ "${CONFIRM_DELETE_LAST_VLA_CACHE:-}" != "delete-last-vla-old-caches" ]]; then
  echo "Refusing deletion: set CONFIRM_DELETE_LAST_VLA_CACHE=delete-last-vla-old-caches." >&2
  exit 2
fi

python - "${PLAN_JSON}" "${REPORT_JSON}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
deleted = []
skipped = []
for item in plan.get("candidates", []):
    path = Path(item["path"])
    if not item.get("delete_allowed"):
        skipped.append({**item, "reason": "not allowed by safety filters"})
        continue
    if path.exists():
        shutil.rmtree(path)
        deleted.append(item)
    else:
        skipped.append({**item, "reason": "missing"})
Path(sys.argv[2]).write_text(json.dumps({"dry_run": False, "deleted": deleted, "skipped": skipped}, indent=2, sort_keys=True) + "\n")
PY

cat "${REPORT_JSON}"
