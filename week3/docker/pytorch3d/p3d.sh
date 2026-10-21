#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -f "$TASK_DIR/../../experiments/d6_check.py" ]; then
  printf 'Run the installed entry at /home/bogger/10weekplan/week3/docker/pytorch3d/p3d.sh\n' >&2
  exit 1
fi
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
TASK_COMPOSE=(docker compose -f "$TASK_DIR/compose.yaml")
TASK_ACTION=${1:-shell}
if [ "$#" -gt 0 ]; then shift; fi
case "$TASK_ACTION" in
  build)
    exec "${TASK_COMPOSE[@]}" build "$@"
    ;;
  verify|verify-cpu)
    TASK_DEVICE=cuda
    if [ "$TASK_ACTION" = verify-cpu ]; then
      TASK_DEVICE=cpu
      TASK_COMPOSE+=(-f "$TASK_DIR/compose.cpu.yaml")
    fi
    exec "${TASK_COMPOSE[@]}" run --rm --no-deps --interactive=false -T research python /opt/pytorch3d/verify.py \
      --device "$TASK_DEVICE" \
      --smpl-model /workspace/week3/models/smpl_clean/SMPL_NEUTRAL.pkl "$@"
    ;;
  python)
    exec "${TASK_COMPOSE[@]}" run --rm --no-deps -T research python "$@"
    ;;
  shell)
    exec "${TASK_COMPOSE[@]}" run --rm --no-deps research bash "$@"
    ;;
  *)
    printf 'Usage: %s {build|verify|verify-cpu|python <args>|shell}\n' "$0" >&2
    exit 2
    ;;
esac
