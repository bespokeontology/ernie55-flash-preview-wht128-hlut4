#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="."
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,2,4,1,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-0}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"
export ERNIE55_KV_DDR4_BLKS="${ERNIE55_KV_DDR4_BLKS:-512}"

# Best-effort non-fatal clock lock verification/application (leaves pp_dpm_pcie strictly untouched).
# Primary persistent root lock executes inside /usr/local/sbin/amdgpu-safe-bind.sh on every boot.
if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$gov" 2>/dev/null || true
  done
fi
for d in /sys/class/drm/card[0-9]*/device; do
  [[ "$(cat "$d/vendor" 2>/dev/null || true)" == "0x1002" ]] || continue
  if [[ -w "$d/power_dpm_force_performance_level" ]]; then
    echo "manual" > "$d/power_dpm_force_performance_level" 2>/dev/null || true
    mclk_max=$(awk -F: 'NF>=2{gsub(/^[ \t]+|[ \t]+$/,"",$1); id=$1} END{print id}' "$d/pp_dpm_mclk" 2>/dev/null || true)
    fclk_max=$(awk -F: 'NF>=2{gsub(/^[ \t]+|[ \t]+$/,"",$1); id=$1} END{print id}' "$d/pp_dpm_fclk" 2>/dev/null || true)
    soc_max=$(awk -F: 'NF>=2{gsub(/^[ \t]+|[ \t]+$/,"",$1); id=$1} END{print id}' "$d/pp_dpm_socclk" 2>/dev/null || true)
    [[ -n "$mclk_max" ]] && echo "$mclk_max" > "$d/pp_dpm_mclk" 2>/dev/null || true
    [[ -n "$fclk_max" ]] && echo "$fclk_max" > "$d/pp_dpm_fclk" 2>/dev/null || true
    [[ -n "$soc_max" ]]  && echo "$soc_max"  > "$d/pp_dpm_socclk" 2>/dev/null || true
    echo "7 8" > "$d/pp_dpm_sclk" 2>/dev/null || true
  fi
done

exec "$ROOT/bin/ernie55_fe" \
  --host 127.0.0.1 \
  --port 8052 \
  --model-id ERNIE-5.5-Flash-Preview-WHT128-HLUT4 \
  --weight-root "$ROOT/checkpoints/active" \
  --topology "$ROOT/config/topology_192g.json" \
  --shm /ernie55_p150_ipc \
  --max-kv 1048576 \
  --prefill-chunk 2048 \
  --kvblk-ddr4-slots "${ERNIE55_KV_DDR4_BLKS}" \
  --draft-mode dspark \
  --dspark-k 5 \
  "$@"
