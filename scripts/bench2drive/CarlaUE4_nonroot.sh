#!/usr/bin/env bash
set -euo pipefail

# Unreal's Linux shipping binary refuses to start as UID 0.  The evaluator
# itself may run as root in the service container, so drop only the CARLA
# server to the non-root owner used by the shared CARLA installation.
CARLA_ROOT=${CARLA_ROOT:-/mnt/project/CARLA_0.9.15}
SHIPPING_BINARY=${CARLA_ROOT}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
CARLA_RUN_UID=${CARLA_RUN_UID:-$(stat -c %u "${SHIPPING_BINARY}")}
CARLA_RUN_GID=${CARLA_RUN_GID:-$(stat -c %g "${SHIPPING_BINARY}")}
if [[ "${CARLA_RUN_UID}" == "0" ]]; then
  echo "CARLA_RUN_UID resolved to root; set it to a non-root UID" >&2
  exit 2
fi
exec setpriv --reuid="${CARLA_RUN_UID}" --regid="${CARLA_RUN_GID}" --clear-groups \
  "${CARLA_ROOT}/CarlaUE4.sh" "$@"
