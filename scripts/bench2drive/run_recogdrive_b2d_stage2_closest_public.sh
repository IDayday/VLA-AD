#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# This is the only Stage2 entry point admitted by the closest-public gate.  The
# implementation is shared with the retired diagnostic launcher, but formal
# mode adds exact contract/config/record checks and never loads the old policy.
python "${VLA_AD_ROOT}/scripts/bench2drive/check_recogdrive_b2d_reproduction_gate.py" \
  --target stage2 \
  --report "${STAGE2_GATE_REPORT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_reproduction_gate/stage2_gate.json}"

exec env \
  FORMAL_CLOSEST_PUBLIC=1 \
  RANDOM_INIT_POLICY=1 \
  CONFIG="${CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_stage2_closest_public_2b.yaml}" \
  bash "${VLA_AD_ROOT}/scripts/bench2drive/run_recogdrive_b2d_stage2_il_official.sh"
