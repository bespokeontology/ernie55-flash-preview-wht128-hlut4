#!/usr/bin/env bash
set -euo pipefail

ROOT="."
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,2,3,4,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-0}"

exec "$ROOT/bin/ernie55_fe" \
  --host 127.0.0.1 \
  --port 8052 \
  --model-id ERNIE-5.5-Flash-Preview-WHT128-HLUT4 \
  --weight-root "$ROOT/checkpoints/active" \
  --topology "$ROOT/config/topology_192g.json" \
  --shm /ernie55_p150_ipc \
  --max-kv 1048576 \
  --draft-mode dspark \
  --dspark-k 5 \
  "$@"
