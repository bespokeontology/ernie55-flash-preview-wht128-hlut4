#!/usr/bin/env bash
set -euo pipefail

ROOT="${ERNIE55_ROOT:-.}"
TT_TOOL="${TT_TOOLCHAIN_ROOT:-./toolchains}"
HWLOC_LIB="${HWLOC_LIB:-${HOME}/.local/state/p150-recovery-20261107/runtime/usr/lib/x86_64-linux-gnu}"

export LD_LIBRARY_PATH="$TT_TOOL/llvm-20/lib:$TT_TOOL/llvm-deps/usr/lib/x86_64-linux-gnu:$TT_TOOL/ttnn-0.80.0rc6/build/lib:$TT_TOOL/ttnn.libs:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
export TT_METAL_RUNTIME_ROOT="$TT_TOOL/ttnn-0.80.0rc6"
export TT_METAL_HOME="$TT_TOOL/ttnn-0.80.0rc6"
export ARCH_NAME="blackhole"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-0}"

exec "$ROOT/bin/p150_ernie55_worker" \
  --weight-root "$ROOT/checkpoints/active" \
  --topology "$ROOT/config/topology_192g.json" \
  --config "$ROOT/checkpoints/active/config.json" \
  --kernels-dir "$ROOT/p150/kernels" \
  --shm /ernie55_p150_ipc \
  "$@"
