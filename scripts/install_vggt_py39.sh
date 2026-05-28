#!/usr/bin/env bash
set -euo pipefail

VGGT_SOURCE="${1:-${VGGT_SOURCE:-}}"

if [[ -z "${VGGT_SOURCE}" ]]; then
  echo "Usage: bash scripts/install_vggt_py39.sh /path/to/vggt" >&2
  echo "Set VGGT_SOURCE=/path/to/vggt as an alternative." >&2
  exit 2
fi

if [[ ! -d "${VGGT_SOURCE}" ]]; then
  echo "VGGT source directory does not exist: ${VGGT_SOURCE}" >&2
  exit 2
fi

python - <<'PY'
import sys

version = f"{sys.version_info.major}.{sys.version_info.minor}"
if version != "3.9":
    raise SystemExit(
        f"Refusing to install VGGT with Python {version}. "
        "Activate the project Python 3.9 environment first."
    )
print(f"Using Python {version}: {sys.executable}")
PY

python -m pip install --no-deps --ignore-requires-python -e "${VGGT_SOURCE}"

python - <<'PY'
import importlib
import sys

vggt = importlib.import_module("vggt")
print(f"VGGT import OK from: {getattr(vggt, '__file__', vggt)}")
print(f"Python preserved: {sys.version.split()[0]}")
PY
