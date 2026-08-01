#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  make -C "$script_dir" help
  exit 0
fi

target="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
make -C "$script_dir" "$target" "$@"

