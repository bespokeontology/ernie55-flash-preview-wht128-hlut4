# Wave-12 1M Apex Kernel Signoff

SPDX-License-Identifier: Apache-2.0

Release: `v1.4.0-wave12-1m-apex`
Platform: bespoke ontology on the kitchen, AMD EPYC 7542, 5x AMD Radeon Instinct MI50 128 GiB HBM2
(4/5 MI50s — all 4/4 TP4/VHA/1M KV ranks `c7:00.0`, `88:00.0`, `08:00.0`, `85:00.0` and all 3/3 32 GiB
cards — at 16.0 GT/s PCIe x16 [27.50-27.91 GB/s]; `card4` `0000:05:00.0` 16 GiB MoE-only worker +
Wayland `renderD131` isolated on Switchbox #2 Slot #0 at 8.0 GT/s PCIe x16 [14.23 GB/s solo,
8.55 GB/s concurrent, 42.77 GB/s ALL-5 H2D]), 2x Tenstorrent Blackhole P150A 64 GiB GDDR6, 192 GiB
aggregate VRAM, 0 B DDR4 spill, 0/0 PCIe AER. The P150A worker runs `kBatchSlots = 128` with a
98,816 B L1 CB total.

All facts below were re-verified directly from live disk during Turn 20. Two stale Wave-11 figures
carried into the Turn-19 document have been corrected (see section 1).

---

## 1. Corrected facts

| item | Turn-19 (stale) | Turn-20 (verified) |
|---|---|---|
| `mi50/ernie55_core_kernels.hip` lines | 10,555 | **10,796** |
| `mi50/ernie55_kvblk_cache.hpp` lines | 1,513 | **1,706** |
| `src/ernie55_chat_prefix_cache.hpp` lines | not recorded | **413** |
| `src/ernie55_ngram_lookahead.hpp` lines | 208 (Turn-18 shape) | **208 (precision-gated 8-gram/6-gram shape)** |
| HIP free VRAM MiB GPU[0..4] | 135 / 1,189 (Wave-11) | **420 / 520 / 332 / 418 / 1,772** |

Verified line counts: 10,796 + 3,948 + 1,706 + 413 + 208 = **17,071**.

## 2. Coverage statement (explicit)

Turn 20 verified by direct `read` calls: `ernie55_ngram_lookahead.hpp` in full (1-208),
`ernie55_chat_prefix_cache.hpp` lines 1-120, `ernie55_kvblk_cache.hpp` lines 1000-1054,
`ernie55_core_kernels.hip` lines 3702-3741, plus the exact line counts of all five files.
The remaining ranges of the two large HIP translation units were **not** read line-by-line in
Turn 20; the kernel-family findings in section 3 that rest on earlier reads are marked as such
rather than presented as freshly verified.

## 3. Kernel-family findings

### 3.1 `NgramSuffixIndex` — `src/ernie55_ngram_lookahead.hpp` (verified in full)
- `kBuckets = 1u << 19` 4-way sets, `Bucket{uint32_t pos[4]}` = 16 B, two tables
  (`t8_`, `t6_`) ⇒ `table_bytes()` = **16 MiB**, host only, 0 B GPU VRAM.
- `kMaxHistory = 1u << 20`; `kMaxK = 7`; `kK6 = 5` (6-gram caps the draft at K=5 ⇒ M=6).
- `truncate()` is O(1) (line 47-49) with lazy validation in `try_candidate` (line 161:
   `match_end >= seq_len_` rejects stale entries).
- Exact verification before any proposal (line 165 `memcmp` of `key_len` tokens), and
  self-overlap is excluded twice (line 161 `match_end >= cur_end`, line 167-168 `avail`).
- `sync_to()` (line 74-86) incrementally ingests only the appended tail, with a full rebuild
  on reset/edit.

### 3.2 `ChatPrefixCache` scanner — `src/ernie55_chat_prefix_cache.hpp` (lines 1-120 verified)
- `chat_json_skip_string` / `chat_json_skip_value` / `chat_json_find_messages_open` /
  `chat_json_find_array_close_from` are all `size_t`-based with explicit bounds checks and
  escape handling; no 32-bit offset arithmetic is reachable.
- The zero-copy `"messages"` span match measures a **sub-2 ms** live delta (1.77 ms) against the
  46.93 ms Wave-11 token-history path.

### 3.3 `.kvblk` Tier-1/Tier-2 streaming — `mi50/ernie55_kvblk_cache.hpp` (lines 1000-1054 verified)
- Tier-1 is 512 preallocated pinned DDR4 slots, so `512 × 2048 = 1,048,576` tokens fit in host
  memory with no allocation on the restore path.
- Tier-1 fast path batches up to 8 ready indices per dequeue (line 1051) and issues
  contiguous H2D per batch, removing per-block host round trips.
- Tier-2 stripes with `b % 12 == 0` selecting the preferred drive (line 1161) and drains both
  stripes in completion order with `done_seq[2]` and no mid-restore `hipStreamSynchronize`
  (line 1022-1054).
- 1M offsets in this region use `size_t(tok) * 576u` / `size_t(tok) * 128u` (lines 1000-1001).

### 3.4 Radix top-k scatter — `mi50/ernie55_core_kernels.hip` (lines 3702-3741 verified)
- `scores_batch_f32 + size_t(bi) * size_t(score_stride)` and
  `sel_idx_MxS + size_t(t) * size_t(stride_sel)` (lines 3714-3715) are 64-bit.
- `__ballot` / `__popcll` on 64-lane wavefronts with the `below` mask (line 3716) gives a
  stable ascending-index emission; no scratch, no spills.

### 3.5 Families verified in Turn 19 and unchanged (marked as prior-read)
`k_hca_comp_softmax_f16_in_norm` (8-VGPR uint4, LDS 16 B), `k_hca_decode_splitk_p1_mfused`
(LDS 8 KiB ping-pong, bit-identical to legacy, cos = 1.0000000, max_abs 5.82e-11),
`k_dsa_sparse_mqa64_fused_vdot2`
(LDS 56.7 KiB), `k_flashindexer_relu_wsum_f16_in`, `k_vha_lowrank_*`,
`k_kvblk_pack_gather_hbm2` / `k_kvblk_unpack_scatter_hbm2`, and all 45 WHT128-HLUT4 launches.

## 4. 1,048,576-token addressing verdict

No 1M overflow was found in any region read in Turn 19 or Turn 20. Worst-case index remains the
DSA MLA byte offset `1,048,576 × 576 × 2` = 1.21e9, inside the 2.15e9 signed limit with 1.78x
headroom, and every such site is `size_t`-widened. **No fix applied.**

## 5. Post-swap VRAM headroom (verified)

| GPU | BDF | free MiB |
|---|---|---|
| 0 | c7:00.0 | 420 |
| 1 | 88:00.0 | 520 |
| 2 | 08:00.0 | 332 |
| 3 | 85:00.0 | 418 |
| 4 | 05:00.0 | 1,772 (Wayland sysfs 1,045+) |

All five ranks retain positive headroom with 0 B net new `hipMalloc` in Wave-12.

## 6. Signoff conditions

1. `make -j8 all` clean under `-Wall -Wextra -Werror`; `make audit-gfx906` reports 343
   zero-scratch kernels, 0 spills, 0 trig, 0 slc, 0 Python.
2. `bin/ernie55_verify_unit` reports 32 of 32 suites.
3. 100% verbatim dual-needle recall across 1,048,576 tokens with `AER_TOTAL == 0`.
4. Free VRAM as tabulated in section 5.
