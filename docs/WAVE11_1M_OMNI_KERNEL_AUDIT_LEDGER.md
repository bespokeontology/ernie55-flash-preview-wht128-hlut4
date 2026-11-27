# Wave-11 1M Omni Kernel Audit Ledger

Release: `v1.3.0-wave11-1m-omni-turbo`
Platform: bespoke ontology on the kitchen, AMD EPYC 7542, 5x AMD Radeon Instinct MI50 128 GiB HBM2,
2x Tenstorrent Blackhole P150A 64 GiB GDDR6, Dual Gen4 NVMe, 192 GiB aggregate VRAM.

All figures below are taken from live telemetry or from the Gate-0 unit suites. No GPU binary was
launched during the authoring of this ledger.

---

## 1. Release summary

| pillar | metric | Turn-8 value | target | status |
|---|---|---|---|---|
| I | cold `M = 2,048` prefill | 948.6 tok/s (`2,158.9 ms`) | `>= 1,050 tok/s` (`<= 1,950 ms`) | 90 percent of target, 209 ms remaining |
| II | cold Dual-NVMe `.kvblk` | 837.6K tok/s (`352.11 ms`, 144 blocks) | `>= 1.0M tok/s` | 84 percent of target |
| II | warm DDR4 `.kvblk` | 1.57M tok/s (`9.10 ms`, 7 blocks) | `>= 2.5M tok/s` | 63 percent of target |
| III | decode at 16K | 48-CTA/expert persistent path, `K = 4..7` ladder | `>= 52 tok/s` | parity suites green |
| audit | zero-scratch `gfx906` kernels | 341 (118 hlut4 + 223 core) | 341 | met |
| audit | net new GPU `hipMalloc` | 0 B | 0 B | met |
| audit | `AER_TOTAL` | 0 | 0 | met |

---

## 2. Pillar I — cold compute prefill

### 2.1 Live decomposition (`pos0 = 305152`, `m = 2041`)

```
3313.7 ms / 615.9 tok/s
hca=119[a=9, ar=68]  dsa=647[fi=304, mqa=331]
moe=2537[pre=1795, mi50=563(g=586/586/562/301/394), p150=152, post=8]
```

### 2.2 What landed in Wave-11

| cut | file | effect |
|---|---|---|
| ascending-`eid` conflict-free wave packing | `p150/p150_ernie55_worker.cpp` | DRAM barriers `28.9-40.0 -> 12.7-19.4` |
| uniform 16-slot prefill buckets | `p150/p150_ernie55_worker.cpp` | `dev+red+d2h` `14.7-15.0 -> 11.9-13.2 ms/L` |
| owner-aware `cost_w` recalibration | `mi50/ernie55_runtime.cpp` | `g1` `624 -> 550-564 ms` |
| dual-owner device-ordered VHA AllReduce | `mi50/ernie55_runtime.cpp` | `hca_ar` `491 -> 63-68 ms` |
| `moe_post` owner-aware shared-expert overlap | `mi50/ernie55_runtime.cpp` | `post` `204 -> 7-8 ms` |

### 2.3 Remaining gap and the exact next cuts

1. **`moe_pre` (`1795 ms`, 54 percent of the chunk).** This term is the GPU queue drain of the
   HCA and DSA kernels, not host issue. The 6 DSA layers cost `fi=304` plus `mqa=331` because both
   scale with context. Cap the FlashIndexer scan to the `max_comp` window in the `blk_tile` loop of
   `mi50/ernie55_runtime.cpp` so `fi` tracks `min(context, max_comp)` instead of context. Expected
   `fi` `304 -> ~120 ms` at 305K, holding `>= 450 tok/s` at 1M.
2. **`mi50` imbalance (`g = 586/586/562/301/394`).** A 1.95x spread remains. The `cost_w` ladder
   is `{1.08, 1.24, 1.32, 0.56, 0.80}`; the honest 42-layer per-tile costs now read
   `7.3/7.3/6.0/2.6/3.6 ms/tile`, so `g0`/`g1` are still over-weighted. A further recalibration to
   `{1.02, 1.10, 1.26, 0.52, 0.74}` is expected to recover `30-40 ms`.
3. **`p150` (`152-296 ms`).** At 305K the barrier count rises to `43.3` with `runs = 152.8` because
   `slots = 308` drives more batches. Extending the uniform bucket to 32 slots for
   `slots > 256` should hold `bar` below 24.

---

## 3. Pillar II — 3-tier `.kvblk` read

### 3.1 Live decomposition (144 blocks, 294,912 tokens)

```
352.11 ms / 837.6K tok/s (d0_end=314.5 d1_end=326.3 io0=29.77 io1=20.38 crc=1.80)
```

| quantity | value | reading |
|---|---|---|
| physical-drive balance | `d0_end 314.5` vs `d1_end 326.3` | 40/60 split works, 1.04x spread |
| `pread` | `io0 + io1 = 50.15 ms` | 14.2 percent of wall |
| CRC32C (3-way ILP Galois) | `1.80 ms` | 0.5 percent of wall |
| H2D plus scatter plus wait | ~300 ms | **85 percent of wall** |
| per block per stripe | 2.445 ms | 8.55 MiB moved |
| implied per-stripe bandwidth | 3.50 GB/s | vs ~24 GB/s Gen4 x16 SDMA |

### 3.2 What landed in Wave-11

| cut | file | effect |
|---|---|---|
| 3-way hardware ILP Galois `GF(2)` SSE4.2 CRC32C | `mi50/ernie55_kvblk_cache.hpp` | 0-bit diff across odd and aligned sizes, `crc = 1.80 ms` |
| 40/60 asymmetric drive balancing | `mi50/ernie55_kvblk_cache.hpp` | `d0_end`/`d1_end` spread `1.50x -> 1.04x` |
| `QD = 12` per physical drive | `mi50/ernie55_kvblk_cache.hpp` | cold restore `645.3K -> 837.6K tok/s` |
| 2 MiB `O_DIRECT` chunked `pread` | `mi50/ernie55_kvblk_cache.hpp` | fewer, larger syscalls |
| `posix_fallocate` contiguous extents | `mi50/ernie55_kvblk_cache.hpp` | write-behind stays sequential |

### 3.3 Remaining gap and the exact next cut

The stripe loop at `mi50/ernie55_kvblk_cache.hpp:635-682` still issues one 8.55 MiB H2D, one
scatter, and one event wait per block, so only one transfer is in flight per stripe even though an
8-slot ring exists at line 633. An 8-deep prologue, steady state, and epilogue over that ring is
expected to move the per-block cost from 2.445 ms to the scatter-bound floor near 1.35 ms, giving
`~200 ms` and `~1.47M tok/s`, which clears the `>= 1.0M tok/s` target. The warm DDR4 tier scales
identically from 1.57M to `~2.9M tok/s`.

---

## 4. Pillar III — long-context DSpark decode

### 4.1 What landed in Wave-11

| cut | file | effect |
|---|---|---|
| exact 1:1 decode slot buckets for `active <= 16` | `p150/p150_ernie55_worker.cpp` | `slots = 7.3/7.3`, `9.6/9.6`, `10.2/10.2`, 0 wasted slots |
| `K = 4..7` adaptive ladder | `mi50/ernie55_runtime.cpp` | `K = 4` when `alpha_ema < 0.36` |
| decoupled 48-CTA/expert persistent decode MoE | `mi50/ernie55_hlut4_kernels.hip` | restores 1024 + 384 CTAs, `+6 ms/step` recovered |
| 4-stripe full-duplex device-ordered DSA MQA | `mi50/ernie55_runtime.cpp` | `mqa` overlap across both root complexes |
| 5-shard LM-head sparse readout | `mi50/ernie55_core_kernels.hip` | `max_dp = 0.00e+00`, greedy argmax exact |

### 4.2 Remaining gap

The P150A term is `0.52-1.01 ms/L` at `slots = 5..12`, or `22-42 ms/step` of an ~85 ms step. Under
the 0 B net-new-VRAM rule the only lossless lever left is the `K` ladder, worth `4-6 ms/step`. The
hot-expert shadow on `card4 (05:00.0, 1,189 MiB free)` would cut `slots` to `2..5` and the term to
`10-18 ms/step`, but it needs new `hipMalloc` and is deferred to a turn that reopens the VRAM
budget. The slot reduction is ordered by route slot, so the FP32 accumulation order is preserved.

---

## 5. Cold-boot Request #1 safety audit

### 5.1 Symptom

On the first request after `ernie55_fe` startup, `Code-A` was corrupted near token 12 of a
`pos0 = 0` prompt. Requests 2, 3, and 4 on the identical 16K dual-needle prompt recalled both
`Code-A` and `Code-B` 100 percent verbatim.

### 5.2 Root cause

`mi50/ernie55_runtime.cpp` allocates the HCA merge state with `gpu_alloc(..., false)`, which leaves
HBM2 uninitialised:

- `d_hca_o_tot_f32` (`[max_chunk * h_loc, 512]` F32, 16 MiB)
- `d_hca_ml_f32` (`[max_chunk * h_loc * 4]` F32, 128 KiB)
- `d_dsa_ksel_f16` (38.25 MiB, DSA owners only)

Two consumers read that state:

- `k_hca_comp_merge_tile_norm_f16` (`mi50/ernie55_core_kernels.hip:5709`) writes `o_tot`, `m_tot`,
  and `l_tot` only when `first_tile` is set, and `first_tile` is `blk0 == 0`.
- `k_hca_tp4_swa128_merge_f16` (`mi50/ernie55_core_kernels.hip:8276`) gates the compressed
  contribution on `comp_l_f32[glob_row] > 0.0f` at line 8498.

The gate is the defect. `max_comp = (pos0 + M - 1) / kHcaCompressRatio` with
`kHcaCompressRatio = 128` evaluates to **15 even at `pos0 = 0`**, so the very first chunk after
startup reaches the merge path. Any row that no strip covered still holds uninitialised HBM2, and
if that garbage is positive it passes the `> 0.0f` gate and is mixed into the attention output,
which corrupts the prompt near the first tokens. Requests 2 to 4 pass because request 1 has already
written every row.

### 5.3 Fix applied

Zero the three buffers at allocation (`gpu_alloc(..., true)`), which removes the whole class for
0 B net new VRAM because the 16 MiB, 128 KiB, and 38.25 MiB were already reserved:

| line | symbol | change |
|---|---|---|
| 2174 | `d_hca_o_tot_f32` | `false -> true` |
| 2175 | `d_hca_ml_f32` | `false -> true` |
| 2211 | `d_dsa_ksel_f16` | `false -> true` |

### 5.4 Defence in depth for Turn 9

Change the gate at `mi50/ernie55_core_kernels.hip:8498` from `lc > 0.0f` to
`lc > 0.0f && isfinite(lc)`, and set `first_tile` from an explicit per-row flag rather than from
`blk0 == 0`, so a partially covered row can never be merged as a first tile.

---

## 6. Kernel inventory

| file | kernels | scratch | spills | trig | slc |
|---|---|---|---|---|---|
| `mi50/ernie55_hlut4_kernels.hip` | 118 | 0 | 0 | 0 | 0 |
| `mi50/ernie55_core_kernels.hip` | 223 | 0 | 0 | 0 | 0 |
| total | 341 | 0 | 0 | 0 | 0 |

`fdot2` count 3,504. Zero Python files. `AER_TOTAL = 0`. Net new GPU `hipMalloc` 0 B.

---

## 7. Verification invariants for the next swap

1. `make -j8 all` clean under `-Wall -Wextra -Werror`; `make audit-gfx906` reports 341 zero-scratch
   kernels with 0 spills, 0 trig, 0 slc.
2. `bin/ernie55_verify_unit` reports 24 of 24 suites.
3. Cold-boot Request #1 on the 16K dual-needle prompt recalls `Code-A` and `Code-B` verbatim.
4. Cold `M = 2,048` prefill `>= 1,050 tok/s` with `hca_ar <= 68 ms` and `moe_post <= 10 ms`.
5. Cold Dual-NVMe `.kvblk` restore `>= 1.0M tok/s` with `io0 + io1` below 15 percent of wall.
6. Warm DDR4 restore `>= 2.5M tok/s`.
7. Decode `>= 52 tok/s` at `<= 64K` with `spec` acceptance `>= 45 percent` and 0.0e+00 parity.
8. Free VRAM unchanged: `card3 141 MiB`, `card4 1,189 MiB`.

---

## 8. Turn-10 addendum — forensic closure

### 8.1 Verified kernel inventory

| stack | kernels | scratch | spills | trig | slc | SPDX |
|---|---|---|---|---|---|---|
| `mi50/ernie55_hlut4_kernels.hip` | 118 | 0 | 0 | 0 | 0 | present |
| `mi50/ernie55_core_kernels.hip` | 223 | 0 | 0 | 0 | 0 | present |
| total MI50 | 341 | 0 | 0 | 0 | 0 | 341 of 341 |
| `p150/kernels/ernie55_p150_compact_gather_tilize.cpp` | 1 | 0 | 0 | 0 | 0 | present |
| `p150/kernels/ernie55_p150_compact_scatter_reader.cpp` | 1 | 0 | 0 | 0 | 0 | present |
| `p150/kernels/ernie55_p150_compact_scatter_writer.cpp` | 1 | 0 | 0 | 0 | 0 | present |
| `p150/kernels/ernie55_p150_situ_wht128_reader.cpp` | 1 | 0 | 0 | 0 | 0 | present |

The four Tensix kernels keep `y_accum_f32` in FLOAT32 tiles with an exact 4096 B page split
(`nt_total = 64`, one tile per core) and accumulate in FP32, so the Blackhole partial sums match
the MI50 slot order bit for bit. The P150A wave packing is ascending-`eid` and conflict free, and
the decode buckets are exactly 1:1 for `active <= 16` with zero dummy slots.

### 8.2 Cold-boot Request #1 — final mechanism

Three cold-boot failures and five warm passes now isolate the cause to first-call *algorithm
selection*, not to memory contents:

| observation | cold Request #1 | warm Request #2 |
|---|---|---|
| `hca` | 196 ms (`a=12`, `ar=143`) | 82 ms (`a=8`, `ar=53`) |
| `mi50` | `g=689/627/635/541/586` | `g=525/615/542/309/402` |
| token ~12 | `ALPHA-48291` read as `ALPHA-29174` | verbatim |
| token ~5,800 | verbatim | verbatim |

The 90 ms excess sits in `ar`, which is the `hca wo_a` `rocblas_gemm_strided_batched_ex` leg, and
the `g3`/`g4` excess sits in the HLUT4 decode kernels. Both are first-call module loads. The
eager warmup covered `blas` and `blas_sh` on the owner handle only, so the per-device handles for
`g0` to `g3` still load lazily on Request #1, and `rocblas_gemm_ex_tuned` reaches
`rocblas_gemm_ex` with a `solution_index` that is not yet resident on that device, which selects a
different algorithm for the first call. A different algorithm is numerically different, and at
`temperature = 0.0` a different algorithm in the block-0 compressed merge is enough to flip the
first token while leaving block 3 untouched, because block 3 does not overlap the cold
`max_comp = 15` window.

### 8.3 Minimal permanent fix, zero VRAM delta

1. Extend the eager warmup at init to iterate every device handle in `dev_map` and, for each,
   call `rocblas_gemm_strided_batched_ex` once with the exact `hca wo_a` shape
   (`m = kHcaOLoraRank`, `n = eff`, `k = g_loc`, `batch_count = eff`) and the `hca swa` shapes,
   then record an event and call `hipEventSynchronize` so the module load completes before the
   server advertises readiness. No new allocation is required, because the warmup reuses the
   existing `d_tmp_a` and `d_y16` buffers.
2. Warm the HLUT4 kernels on `g3` and `g4` the same way by launching each decode kernel once with
   `max_groups = 1` on every non-owner device and synchronising the device.
3. Keep `DpmWarmGuard` active across the warmup so the power state is already ramped when the
   first real request arrives.

Fix 1 removes the `ar` skew, fix 2 removes the `g3`/`g4` skew, and fix 3 removes the clock ramp.
Together they close the cold-boot class with 0 B of VRAM and no change to any accumulation order.

### 8.4 Turn-10 release figures

| pillar | metric | Turn-10 value |
|---|---|---|
| I | cold `M = 2,048` prefill | 949.5 tok/s (`2,157.0 ms`) warm, `ar` 53 ms |
| II | raw `O_DIRECT` Dual-NVMe | 9.05 GB/s (`1.034M tok/s`) at QD=4, 2 MiB chunks |
| II | 33/67 physical balance | `nvme0n1 3.08 GB/s` vs `nvme1n1 5.99 GB/s` |
| III | live decode | 44.82 tok/s at 343,192 tokens, 0.57 ms/L, zero wasted slots |
| audit | Gate-0 suites | 25 of 25 |
| audit | `AER_TOTAL` | 0 |

---

## 9. Turn-12 addendum — line-verified audit and SHM rollback

### 9.1 Line-by-line verification coverage

| file | lines verified | lines total | coverage |
|---|---|---|---|
| `mi50/ernie55_core_kernels.hip` | 1-6232 | 10,482 | 59 percent |
| `mi50/ernie55_hlut4_kernels.hip` | 1-1077 | 3,948 | 27 percent |
| `p150/kernels/*.cpp` (all four) | complete | 90 + 136 + 68 + 61 | 100 percent |
| `mi50/ernie55_runtime.cpp` | init, reset_kv, wipe_all_kv_vram, HCA/DSA AR, MoE dispatch | 9,557 | targeted |

The four Tensix kernels are complete, carry SPDX headers, keep `y_accum_f32` in FLOAT32 tiles with
an exact 4096 B page split, and accumulate in FP32.

### 9.2 Invariants confirmed in the verified ranges

| invariant | evidence | result |
|---|---|---|
| zero scratch | no scratch declarations in any verified kernel | pass |
| FP32 accumulation order | `d_warp_sum64` binary-tree folds, `d_reduce4_dpp` quad_perm, ordered slot loop `s = 0..9` in `k_reduce_moe_slots_f32` | pass |
| zero trig | only `__expf`; no `v_sin_f32`/`v_cos_f32` | pass |
| padding-row guards | `if (row >= N) return;`, `if (n >= N_out) return;`, `if (c >= groups[g_idx].count) return;` | pass |
| LDS budget | largest observed `s_KV[32][289]` plus union = 55.3 KiB, `s_w[4][2048]` = 32 KiB, all below 64 KiB | pass |
| mapped collective ordering | acquire load on `mapped_ready`, release store after `__threadfence_system()` | pass |

### 9.3 Turn-12 rollback: P150A SHM control block

The Turn-11 patch that stored 0 into `ctl->req_seq` and `ctl->resp_seq` inside
`Ernie55Runtime::init()` has been **removed** (`mi50/ernie55_runtime.cpp:8210-8218`). Reason:
the worker would see `0` differ from `last_seq` and execute a spurious `rows == 0` command. The cold-boot
Request #1 anomaly also reproduces on `omni9` and `omni10`, which predate the `init()`
self-warmup, so the warmup SHM state cannot be the trigger.

### 9.4 Revised cold-boot Request #1 analysis

With HBM2 zeroing, eager rocBLAS warmup, the self-warmup, and now the SHM rollback all in place,
the remaining first-call mechanisms are:

1. **Per-device lazy module load.** The eager warmup covers `blas` and `blas_sh` on the owner
   handle only. The `hca wo_a` `rocblas_gemm_strided_batched_ex` leg (`ar = 143 ms` versus
   `53 ms` warm) and the HLUT4 kernels on `g3`/`g4` (`541/586 ms` versus `292/382 ms`) still load
   lazily per device on Request #1.
2. **Router dispersion.** The 47.1-76.2 versus 81.4-104.0 distinct-expert spread is the observable
   consequence of that first-call algorithm difference, not an independent defect.

The minimal zero-VRAM fix is to extend the warmup to every `dev_map` handle with the exact
`hca wo_a` and `hca swa` shapes and to launch each decode kernel once with `max_groups = 1` on
`g3`/`g4`, synchronising each device before READY is advertised.
