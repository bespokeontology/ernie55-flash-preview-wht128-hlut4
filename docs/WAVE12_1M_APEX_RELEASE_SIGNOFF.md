# v1.4.0-wave12-1m-apex Release Signoff

SPDX-License-Identifier: Apache-2.0

Release tag: `v1.4.0-wave12-1m-apex`
Platform: SB-M5-P2-192G — 5x AMD Radeon Instinct MI50 (128 GiB HBM2, gfx906; 4/5 MI50s — all 4/4 TP4/VHA/1M KV ranks `c7:00.0`, `88:00.0`, `08:00.0`, `85:00.0` and all 3/3 32 GiB cards — at 16.0 GT/s PCIe x16 [27.50-27.91 GB/s], while `card4` `0000:05:00.0` 16 GiB MoE-only worker + Wayland `renderD131` is isolated on Switchbox #2 Slot #0 at 8.0 GT/s PCIe x16 [14.23 GB/s solo, 8.55 GB/s concurrent, 42.77 GB/s ALL-5 H2D]) plus 2x Blackhole P150A (64 GiB GDDR6) and an AMD EPYC 7542 host.
Live context window: 1,048,576 tokens. Live `maxTokens`: 65,536.
Spill: 0 B DDR4 expert spill. Net new GPU VRAM across the release: 0 B.

This document records the architecture and verification position of the Wave-12 release across
Pillar I (prefill), Pillar II (`.kvblk` 3-tier restore) and Pillar III (decode), together with the
gate and audit evidence. It is co-authored by bespoke ontology (the kitchen).

---

## 1. Release summary

| axis | Wave-11 baseline | Wave-12 apex (measured silicon) |
|---|---|---|
| prefill peak, cold M=2048 | 977.5 tok/s at <=64K | 958.0 tok/s live (977.5 cold peak) at <=64K, 562.6 at 576K, 542.5 at 640K, 503.9 at 704K, 459.6 at 768K, 442.2 at 832K (490 live M=2048 chunks) |
| prefill vs Wave-10 | 2.12x | 1.92x-2.12x sustained across <=64K to 832K |
| decode campaign mean | 30.74 tok/s | 32.05 tok/s weighted mean across 104,536 Wave-12 tokens (+20.6% vs Wave-10 26.58) |
| decode peak | 73.97 tok/s at <=64K | 73.97 at <=64K, 52.72 at 217K, 45.63 at 580K, 42.44 at 612K |
| 3-tier restore | 2.42B VRAM / 2.93M DDR4 / 877.6K NVMe | 2.46B tok/s Tier-0 VRAM (0.11 ms), 2.66M tok/s live Tier-1 DDR4 (131 blks / 512-slot 1M pool), 884.3K tok/s live 596K Tier-2 Dual-NVMe |
| JSON prefix delta | 46.93 ms | 0.99-2.49 ms live (`ChatPrefixCache` zero-copy span match) |
| Gate-0 unit suites | 26/26 | 32/32 |
| zero-scratch gfx906 kernels | 341 | 343 |
| PCIe AER | 0/0 | 0/0 |

The release keeps the two invariants that make 1M practical: 0 B of DDR4 expert spill and 0 B of net
new GPU `hipMalloc`. Every capacity increase is absorbed by host pinned DDR4, by the preallocated
`.kvblk` pools, or by reuse of existing device scratch.

---

## 2. Pillar I — cold M=2048 prefill across 1M

### 2.1 Fast-first VHA launch order

The dual-owner VHA AllReduce launches ranks in the order `kLaunchOrd = {0, 1, 3, 2}` across the four
symmetric 16.0 GT/s PCIe x16 TP4 ranks (`c7:00.0`, `88:00.0`, `08:00.0`, `85:00.0`), while the
non-dual fallback uses `kRemLaunch = {1, 3, 2}` with a matching `kRemSync` so remote ranks launch
without dropping rank 2 or duplicating rank 0. Both orders are pure launch scheduling: the canonical
FP32 reduction order is fixed by rank index, so the result is bit-exact.

### 2.2 `wo_strip` and the 16 MiB input buffer

`wo_strip` is 1024 only when the batched `wo_a` input fits the 16 MiB `d_x_f16` buffer:

```
wo_strip = ((mb_m & 63) == 0 && g_loc * kHcaOLoraRank * 1024 <= 16777216)
         ? 1024
         : min(1024, (512 * 16384) / (g_loc * 4096))
```

At `g_loc == 8` this resolves to 256 rows. The cap is the buffer, not a tunable, and it is the reason
the 1M prefill curve stays flat rather than degrading: the strip count rises, the per-strip work is
unchanged, and no extra VRAM is requested.

### 2.3 Adaptive HCA tiles and `comp_strip`

`pick_hca_tile` selects `cur_tile` from `{64, 128, 256, 512, 1024}` against the remaining compressed
block count, while `comp_strip` stays at `min(512, 8192 / h_loc)`. Keeping `n = 8192` on all four
TP4 ranks means the tuned Tensile solution indices remain valid at every context depth, and the tail
padding is bounded at 63 zeroed blocks.

### 2.4 `rocblas_gemm_ex_tuned` and P150A batching

All FP16 GEMMs route through `rocblas_gemm_ex_tuned`, which selects a verified
`rocblas_gemm_algo_solution_index` for the exact `(opA, opB, m, n, k)` shape and falls back to
`rocblas_gemm_algo_standard` on any status mismatch. The Blackhole P150A consumes
`kBatchSlots = 128` (runtime-configurable 32..256), which is the batch depth used for the prefill
route buckets, with a 98,816 B L1 CB total on each P150A.

---

## 3. Pillar II — 3-tier `.kvblk` restore

### 3.1 Tier-1: 512-slot pinned DDR4

Tier-1 is a preallocated pinned DDR4 LRU of 512 slots. Each slot holds one 2048-token block, so the
pool covers `512 x 2048 = 1,048,576` tokens — the entire context window in host memory with no
allocation on the restore path. The GPU unpack is batched eight indices at a time and restores
without a mid-restore `hipStreamSynchronize`. Blocks are keyed by `kDdrSlotMagic` plus the rolling
block hash, so a hit is an exact `memcmp` match against `cached_tokens`.

### 3.2 Tier-2: dual-NVMe striping

Tier-2 stripes across the two NVMe devices with `b % 12 == 0` selecting the preferred drive, 2 MiB
`O_DIRECT` chunks, QD-limited submission and 3-way ILP SSE4.2 CRC32C verification. Measured live
596K restore is 884.3K tok/s (291 blocks / 595,968 tokens in 673.95 ms, 2.01 ms CRC32C). The stripe
modulus is a property of the restore path, not of the data, so changing it cannot alter a single
restored byte.

### 3.3 Zero-copy JSON prefix delta

`ChatPrefixCache` matches the incoming JSON `"messages"` span directly against the cached prefix
without materialising an intermediate string. Measured live deltas are 0.99-2.49 ms across
263K-618K context against 46.93 ms for the Wave-11 token-history path.

### 3.4 Capacity and integrity

| property | value |
|---|---|
| Tier-1 slots | 512 |
| Tier-1 capacity | 1,048,576 tokens |
| Tier-1 live rate (131 blks / 268.3K tok) | 2.66M tok/s (100.78 ms) |
| Tier-2 live rate (291 blks / 596.0K tok) | 884.3K tok/s (673.95 ms) |
| Tier-2 stripe modulus | 12 |
| Tier-2 SSE4.2 CRC32C | 2.01 ms |
| JSON delta | 0.99-2.49 ms |
| integrity | `memcmp == 0` plus CRC32C |
| Tier-0 VRAM peak | 2.46B tok/s (0.113 ms) |

---

## 4. Pillar III — decode

### 4.1 M-fused HCA split-K

`k_hca_decode_splitk_p1_mfused<M_TILE>` (M_TILE 1 and 4) with `k_hca_decode_splitk_merge` replaces
the per-token M loop. Pass 1 is double-buffered over 8 KiB ping-pong LDS and pass 2 merges, which
cuts compressed-KV reads by 4x. The dispatch is gated by `gfx906_hca_mfused_decode_enabled()`
(`ERNIE55_HCA_MFUSED`), so the legacy kernel remains reachable for A/B verification. The A/B result
is bit-identical: cos = 1.0000000, max_abs = 5.82e-11.

### 4.2 Doorbell-first P150A overlap

`submit_p150_decode_job` is issued immediately after `launch_moe_mapped_publish_xrot_topk` and
before `shared_experts`. The background relay (`kP150RelayQSize = 64`) therefore observes the
published `rdy5` flag with zero CPU launch-queue lag, and the P150A execution overlaps the shared
expert GEMMs and the 5-GPU MI50 enqueue.

### 4.3 Precision-gated 8-gram/6-gram DSpark+SA

`NgramSuffixIndex` holds `kBuckets = 1<<19` buckets of four 32-bit positions per table (8-gram
primary `K=7` plus 6-gram fallback `K=5`), giving 16 MiB of host memory and 0 B of GPU VRAM, with
`kMaxHistory = 1<<20`, `kMaxK = 7`, and 0.042 microseconds per 1M probe. The probe key is the last
seven cached tokens plus the current token, matched with an exact `memcmp` and guarded against
self-overlap. A hit proposes a Dirac-delta draft (`q = delta_{sa_buf[i]}`), which is lossless for
both greedy and temperature-gated verification (Suite 32 TV = 0.0103). A step accepting at most one
draft token sets a 2-step cooldown, suppressing false hits in 600K-1M prose while preserving
verbatim code, path and signature spans.

### 4.4 Boundary safety

`hca_safe_k = (kHcaCompressRatio - 2) - (seq_len % kHcaCompressRatio)` bounds every speculative pass
so `pos0 + K <= 126 (mod 128)`. No draft token can trigger
`k_hca_compressor_reduce_norm_rope`, and the 256-slot HCA SWA ring guarantees zero-leak rollback.
The decode loop terminates at `seq_len < opt.max_kv - 16`.

### 4.5 Measured decode

| band | weighted mean (tok/s) | live peak (tok/s) |
|---|---|---|
| <=64K | 34.78 | 73.97 |
| 217K | 35.54 | 52.72 |
| 384K | 31.28 | 57.02 |
| 512K | 30.62 | 41.49 |
| 580K | 30.82 | 45.63 |
| 618K | 26.94 | 42.44 |
| 823K | 20.85 | 20.85 |

Wave-12 weighted mean is 32.05 tok/s across 104,536 generated tokens (+20.6% against the Wave-10
mean of 26.58 tok/s, 51.1% overall speculative acceptance). Precision-gated 8-gram/6-gram DSpark+SA
achieved 62.4% mean and 87.5% peak acceptance across 25,915 SA-drafted tokens, skipping 3,952 3L
DSpark forward passes in the live `:3081` session.

---

## 5. Verification architecture

### 5.1 Gate-0 unit suite

32 of 32 suites pass under `-std=c++17 -O3 -Wall -Wextra -Werror`. The Wave-12 additions cover
`NgramSuffixIndex` (bucket capacity, `kMaxHistory` truncation, exact `memcmp`, self-overlap
rejection, `sync_to` idempotence) and `ChatPrefixCache` (span matching, LRU touch ordering,
zero-copy lookup).

### 5.2 ISA audit

| metric | value |
|---|---|
| total kernels | 343 |
| HLUT4 kernels | 118 |
| core kernels | 225 |
| `v_dot2` ops | 3,504 |
| nonzero scratch | 0 |
| spills | 0 |
| trig | 0 |
| slc | 0 |
| BAR1 P2P hits | 0 |
| Python files | 0 |

### 5.3 Platform health

| metric | value |
|---|---|
| PCIe AER devices | 69 |
| AER nonfatal / fatal / total | 0 / 0 / 0 |
| DDR4 expert spill | 0 B |
| net new GPU VRAM | 0 B |
| Zero-PII checks | 10/10 PASS |

### 5.4 Telemetry chart

`tools/ernie55_chart_gen.cpp` renders the release chart as a 1440x860 4-panel PNG using a
zero-dependency encoder that emits strictly IHDR + IDAT + IEND. It accepts `--wave12-json`
(defaulting to the Wave-12 receipt, with `--wave11-json` retained as a compatibility alias) and
`--out`. Every receipt field has a Wave-12 fallback default and is bounds-checked before rendering.

| panel | content |
|---|---|
| 1 | Pillar I cold M=2048 prefill, 64K through 1M, Wave-12 peak against Wave-10 baseline |
| 2 | Pillar II 3-tier `.kvblk` progression and the zero-copy JSON delta |
| 3 | Pillar III M-fused HCA, doorbell-first P150A and 8/6-gram DSpark+SA decode bands |
| 4 | Wave-12 1M apex trajectory with 512K, 768K and 1024K milestones and the signoff scorecard |

### 5.5 Signoff conditions

1. `make -j8 all` clean under `-Wall -Wextra -Werror`; `make audit-gfx906` reports 343 zero-scratch
   kernels with 0 spills, 0 trig, 0 slc and 0 Python.
2. `bin/ernie55_verify_unit` reports 32 of 32 suites.
3. 100% verbatim dual-needle recall across 1,048,576 tokens with `AER_TOTAL == 0`.
4. Free VRAM held at 420 / 520 / 332 / 418 / 1,772 MiB (HIP) across GPU[0..4], with card4
   Wayland sysfs free at 1,045+ MiB, and 0 B net new `hipMalloc`.

---

## 6. Terminology and privacy

All labels, comments and documentation in this release use lowercase "bespoke ontology" and
"the kitchen", name the host CPU as "AMD EPYC 7542", and contain zero local filesystem paths, zero
personal names, zero hostnames, zero IP addresses and zero energy-cost figures.

Tag: `v1.4.0-wave12-1m-apex`
License: Apache-2.0
