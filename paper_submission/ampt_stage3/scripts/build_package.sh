#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="${RECOGDRIVE_ROOT:-$(cd -- "$release_root/../.." && pwd)}"
export PYTHONPATH="$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m ampt_stage3.build_release \
  --root "$release_root" \
  --archive "$release_root/../ampt_evaluation_release.zip"
