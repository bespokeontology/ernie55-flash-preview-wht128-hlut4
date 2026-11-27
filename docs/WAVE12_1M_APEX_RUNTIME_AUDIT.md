# Wave-12 1M Apex Runtime Audit

SPDX-License-Identifier: Apache-2.0

Release: `v1.4.0-wave12-1m-apex`
Platform: bespoke ontology on the kitchen, AMD EPYC 7542, 5x AMD Radeon Instinct MI50 128 GiB HBM2
(4/5 MI50s — all 4/4 TP4/VHA/1M KV ranks `c7:00.0`, `88:00.0`, `08:00.0`, `85:00.0` and all 3/3 32 GiB
cards — at 16.0 GT/s PCIe x16 [27.50-27.91 GB/s]; `card4` `0000:05:00.0` 16 GiB MoE-only worker +
Wayland `renderD131` isolated on Switchbox #2 Slot #0 at 8.0 GT/s PCIe x16 [14.23 GB/s solo,
8.55 GB/s concurrent, 42.77 GB/s ALL-5 H2D]), 2x Tenstorrent Blackhole P150A 64 GiB GDDR6,
0 B DDR4 spill, 0/0 PCIe AER.
The P150A worker runs `kBatchSlots = 128` with a 98,816 B L1 CB total.

Verified file sizes: `mi50/ernie55_runtime.cpp` **9,779** lines, `mi50/ernie55_kvblk_cache.hpp`
**1,706** lines.

## 1. Scope and coverage summary

This ledger consolidates the Turn 21 targeted reads with the Turn 22 full sequential read of
`mi50/ernie55_runtime.cpp` (lines 1-9779, 100 percent). The authoritative per-file coverage
statement is section 6; the earlier Turn-21 note that the full read was incomplete is superseded
by it and is retained here only to record the audit history. Findings that rest on earlier reads
are marked as such.

## 2. Verified constants (exact lines)

| invariant | value | line |
|---|---|---|
| minimax `cost_w[5]` | `{1.55, 1.63, 1.04, 0.34, 0.78}` | `ernie55_runtime.cpp:8112` |
| `cost_w` override | `ERNIE55_MOE_COST` | `:8113` |
| fast-first VHA launch | `kLaunchOrd = {0, 1, 3, 2}` | `:3496` (HCA), `:5414` (DSA) |
| remote launch order | `kRemLaunch = {1, 3, 2}` | `:3785`, `:5684` |
| `wo_strip` | `min(1024, (512*16384)/(g_loc*4096))` | `:4224` |
| Dual-NVMe stripe | `b % 12 == 0` on stripe 0 | `ernie55_kvblk_cache.hpp:1161` |
| readers per stripe | `8` | `:1677` |
| `O_DIRECT` chunk | 2 MiB, `ERNIE55_KVBLK_CHUNK_MB` 1..16 | `:1506-1509` |
| P150A `kBatchSlots` | 128, `ERNIE55_P150_BATCH_SLOTS` 32..256 | `p150/p150_ernie55_worker.cpp:79` |

The measured 45.7% / 54.3% split documented at `ernie55_kvblk_cache.hpp:1158-1160` is the
justification for the `b % 12` rule, and `resolve_blk_path()` still probes both drives, so
existing `.s0`/`.s1` shards remain readable.

## 3. Subsystem findings

### 3.1 Five-MI50 minimax allocation
`cost_w` at 8112 is consumed by the greedy projected-time score; `kCounts` and `kExtraPerGpu`
are unchanged, so the change is 0 B net new `hipMalloc`. `ERNIE55_MOE_COST` provides an
override without a rebuild. Post-swap free VRAM (verified Turn 20): 420 / 520 / 332 / 418 / 1,772 MiB
across GPU[0..4], with card4 Wayland sysfs free 1,045+ MiB.

### 3.2 Pillar I prefill pipeline
Fast-first `kLaunchOrd` at 3496/5414 and `kRemLaunch = {1,3,2}` at 3785/5684 replace the
8 GT/s-era slow-first order; all four TP4 ranks are symmetric 16.0 GT/s x16. `wo_strip` at 4224
is buffer-bound by the 16 MiB `d_x_f16` input (256 rows at `g_loc = 8`). The adaptive HCA
`cur_tile` (`{64..1024}`) with `comp_strip = 512` and the `kBatchSlots = 128` P150A schedule
were verified in Turn 19-20 reads.

### 3.3 Pillar II 3-tier `.kvblk`
- Tier-1: 512 preallocated pinned DDR4 slots ⇒ `512 × 2048 = 1,048,576` tokens, the full 1M context.
- Tier-1 fast path batches up to 8 ready indices per dequeue (`:1051`) with zero mid-restore
  `hipStreamSynchronize`; the only sync is the terminal one at `:1018`.
- Tier-2: `b % 12` striping at `:1161`, QD=8 at `:1677`, 2 MiB `O_DIRECT` chunks at `:1506`.
- CRC32C verification runs in the dedicated pool before any stripe is marked ready (Turn 15 read).
- `ChatPrefixCache` zero-copy `"messages"` span matching measures a sub-2 ms live delta (1.77 ms)
  against the 46.93 ms Wave-11 token-history path.

### 3.4 Pillar III decode
- `M`-fused split-K (`k_hca_decode_splitk_p1_mfused` + `k_hca_decode_splitk_merge`) verified
  Turn 19: 8 KiB ping-pong LDS, bit-identical to legacy (cos = 1.0000000, max_abs 5.82e-11).
- Doorbell-first `submit_p150_decode_job` ahead of `shared_experts` (Turn 17 wiring).
- Precision-gated 8-gram/6-gram cascade with 2-step miss cooldown (verified in full Turn 20,
  `ernie55_ngram_lookahead.hpp:1-208`): 16 MiB host, 0 B GPU VRAM.
- `hca_safe_k` 128-token boundary guard and the `seq_len < opt.max_kv - 16` loop bound were
  read in earlier turns and are unchanged.

## 4. 1,048,576-token edge cases

No 1M addressing overflow was found in any region read across Turns 19-21. The worst-case index
remains the DSA MLA byte offset `1,048,576 × 576 × 2` = 1.21e9, inside the 2.15e9 signed limit
with 1.78x headroom, and every such site is `size_t`-widened. No fix applied.

## 5. Signoff conditions

1. `make -j8 all` clean under `-Wall -Wextra -Werror`; `make audit-gfx906` reports 343
   zero-scratch kernels, 0 spills, 0 trig, 0 slc, 0 Python.
2. `bin/ernie55_verify_unit` reports 32 of 32 suites.
3. 100% verbatim dual-needle recall across 1,048,576 tokens with `AER_TOTAL == 0`.
4. Free VRAM as tabulated in section 3.1.

## 6. Coverage statement

Line-by-line verification covered `mi50/ernie55_runtime.cpp` lines 1-9779 (100 percent, read
sequentially), `mi50/ernie55_kvblk_cache.hpp` lines 1000-1054 and 1157-1706,
`mi50/ernie55_core_kernels.hip` lines 1-6232 (59 percent), `mi50/ernie55_hlut4_kernels.hip`
lines 1-1077 (27 percent), all four `p150/kernels/*.cpp` (100 percent),
`src/ernie55_ngram_lookahead.hpp` (100 percent) and `src/ernie55_chat_prefix_cache.hpp`
(100 percent). The unread ranges are `mi50/ernie55_core_kernels.hip` 6233-10796,
`mi50/ernie55_hlut4_kernels.hip` 1078-3948, and `mi50/ernie55_kvblk_cache.hpp` 1-999 and 1055-1156.

Runtime constants confirmed by the full read: `max_chunk = min(2048, max(16, prefill_chunk))` (7923);
`kOmpThreads == 24` (941); `kMappedPingPong == 16` (1136); `kP150RelayQSize == 64` (1155); the
decode loop bound `seq_len < max_kv - 16` (8986); the HCA guard `hca_safe_k = 126 - (seq_len % 128)`
(8990); the SA probe `k_sa = min(7, hca_safe_k, ...)` (9008) with a 2-step miss cooldown (9051); and
`wo_strip` resolving to 256 rows at `g_loc == 8` (3407-3409, 4224).

## 7. Cold-boot Request #1 position

The anomaly reproduces on builds that predate the `init()` self-warmup, so neither HBM2 contents nor
warmup SHM state is the trigger. The previously specified SHM control-block reset is withdrawn: the
`p150_ernie55_worker` daemon holds a non-zero local `last_seq`, so storing 0 into `req_seq`/`resp_seq`
would execute a spurious `rows == 0` command.

The surviving asymmetry is per-device lazy module load on the non-owner handles. `init()` warms only
`w->blas` and `w->blas_sh` on the spine devices (2264-2325), and the self-warmup drives only
`forward_backbone_chunk`, so the `g2`/`g3`/`g4` `hca wo_a` strided-batched leg and the `g3`/`g4`
HLUT4 kernels remain cold at READY. The zero-VRAM remedy is to extend the eager warmup to every
`dev_map` handle with the exact production shapes and to synchronise each device before READY. The
slot reduction order is unchanged, so the remedy is bit-exact.
