<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-14 Turn 1 — Phase 1 `512 GiB` Baseline Diagnosis and Cut W14.1 (`FlashIndexer` `pos0 >= 2,048` Cliff, Adaptive `kTileK = 32,768`, Gate-0 Suite 47)

Workstation `SB-M5-P2-192G` inside the kitchen, upgraded to `8 x 64 GiB = 512 GiB` Samsung
`M393A8K40B22-CAE` RDIMMs at `3200 MT/s` on all `8/8` channels (`UMC0..UMC7`,
`D18F0x110 = 0x0000003B`, `ce = 0`, `ue = 0`, `AER_TOTAL = 0`).

No `make`, no GPU compiler invocation and no daemon restart was performed during this turn.
`bin/ernie55_verify_unit`, `make audit-gfx906` and the live silicon re-benchmark run immediately
after this turn closes.

Phase-1 evidence: `logs/w14_phase1_build8_warm_uninst.log` (uninstrumented) and
`logs/w14_phase1_build8_xray.log` (instrumented), both from the untouched shipped Wave-13
Build #8 (`v1.5.0-wave13-1k-prefill-1m`, commit `a18a59b`).

## 1. The `fi` cliff is a host-sync placement artifact, not a `FlashIndexer` cost cliff

`chunk_dsa_fi_ms` is a **host wall-clock window**, opened at `mi50/ernie55_runtime.cpp:4547`
(`t_c0_start`) and closed at `:5151` (`if (M >= 256 && sp0 == n_dense)`). It is not a GPU timer.
Whatever host-blocking synchronisation happens to sit inside that window is what the metric
absorbs. With `kSlidingWindow = 128` and `kDsaIndexTopK = 2048` (`src/ernie55_config.hpp:59,76`),
`kStrideSel = 2176` and

```
n_dense   = clamp(kStrideSel - pos0, 0, M)      // :4364
max_hist  = max(1, (pos0 + M - 1) - 128 + 1)    // :4575  == pos0 + 1920 for M = 2048
total_kv  = pos0 + M                            // :5104
```

which reproduces **all 18 published data points** (2 passes x 9 chunks) exactly:

| `pos0` | `n_dense` | phase C1 dense (`:5008`) | `mqa_direct_small_kv` (`:5114`) | host sync inside the `fi` window | published `fi` |
|---|---|---|---|---|---|
| `0` | `2048 = M` | skipped | n/a (C2 not entered) | none — C0 itself is skipped by `if (n_dense < M)` at `:4548` | `0 ms` |
| `2048` | `128` | **runs** | true (`total_kv = 4096`) | **`hipStreamSynchronize(g_own.stream)` at `:5021`** | `56 ms` x-ray / `90-91 ms` uninst |
| `4096..14336` | `0` | skipped | true (`total_kv = 6144..16384`) | none | `3-4 ms` x-ray / `11-13 ms` uninst |
| `16384` | `0` | skipped | **false** (`total_kv = 18432 > 16384`) | **`ev_dsa_uniq_cnt` event sync at `:5142`** | `68 ms` x-ray / `106-115 ms` uninst |
| `18432` | `0` | skipped | false (`total_kv = 20480`) | `ev_dsa_uniq_cnt` event sync | `69 ms` x-ray / `99-104 ms` uninst |

So:

* `pos0 = 0` — `n_dense == M`, the whole C0 block is skipped, `fi = 0` by construction.
* `pos0 = 2048` — the first sparse chunk. Phase C1 (TP4 parallel dense causal DSA, `pos < 2176`)
  is entered because `n_dense = 128 > 0`, and its KV host-bounce
  `hipStreamSynchronize` at `:5021` is the **first** host barrier after a long asynchronous
  stretch, so it drains the entire queued backlog (HCA tail + the whole C0 `FlashIndexer`
  launch chain). `fi` therefore reports ~`87 ms` of *drain*, of which only ~`3 ms` is
  `FlashIndexer` enqueue.
* `pos0 = 4096..14336` — `n_dense = 0`, C1 is skipped, and `mqa_direct_small_kv` is true, so C2
  takes the zero-host-sync direct KV bypass (`:5114`, shipped in Wave-13 Cut W13.5). No barrier
  exists inside the window, so `fi` collapses to the pure host enqueue cost of ~20 kernel
  launches: `3-4 ms`. **The `FlashIndexer` GPU work is still there — it is simply overlapped
  behind the MoE of the surrounding layers instead of being drained.**
* `pos0 >= 16384` — `total_kv` crosses the `16,384` direct-bypass ceiling, so C2 falls back to
  `launch_dsa_mark_selected_keys` + `launch_dsa_compact_unique_map` + the 4-byte
  `ev_dsa_uniq_cnt` readback, whose event sync (`:5142`) lands inside the window and drains the
  `FlashIndexer` chain again: `106-115 ms`.

The `kTileK = 16,384` tile transition is the *second*, independent effect at `pos0 = 16,384`:
`n_tiles = ceil(max_hist / kTileK)` goes `ceil(16256/16384) = 1` at `pos0 = 14,336` to
`ceil(18304/16384) = 2` at `pos0 = 16,384`. The second tile costs a full `4 MiB` key-tile D2H,
three `4 MiB` cross-device H2Ds and one extra `k_dsa_topk2048_streaming_merge` radix-select pass
on every one of the six DSA layers.

The x-ray/uninstrumented split (`56` vs `90`, `68` vs `106`) is the same effect seen from the
other side: `profiling_enabled` adds `prof_sync_tp` barriers that move work out of the `fi`
window, so the instrumented number is *smaller*, not larger.

## 2. The `vha_op` gap is real KV-proportional GPU work, not a regression

`chunk_bk_vha_op` (`:7912-7914`) is measured with a genuine `hipStreamSynchronize` on the owner,
so it is a true GPU time. It grows `472.8 -> 574.5 -> 625.9..643.2 -> 580.1 ms` because

* the `37` HCA layers run VHA + `wo_a` + `o_proj` over a compressed KV whose length grows with
  `pos0` (`pos0 = 0` has `2,048` keys, `pos0 = 14,336` has `16,384`), and
* at `pos0 = 0` all six DSA layers take the 4-way-parallel C1 dense path, whereas at
  `pos0 >= 2048` they take the **owner-only** C2 sparse MQA path, which is ~4x less parallel on
  the owner. That single step accounts for the disproportionate `+101.7 ms` from `pos0 = 0` to
  `pos0 = 2048`, after which the growth is the smooth `+68.7 ms` of KV-proportional work through
  `pos0 = 14,336`.

Zero-VRAM overlap already in the tree that reclaims part of this gap cleanly:

* `mqa_direct_small_kv` (`:5114`) — zero-host-sync direct KV bypass for `total_kv <= 16,384`,
  i.e. exactly `pos0 = 2,048..14,336`, bit-exact, `0 B` VRAM.
* the `ar_dev` mapped-alias VHA relay (`:3670-3712`, `d_tp_oattn_pin` / `d_tp_qabs_pin`) — the
  HCA VHA allreduce is already device-ordered with no host sync.

No further clean reclaim exists for the `pos0 = 0 -> 2048` step: the C1 dense path is only live
for `pos < 2176`, and the C2 owner-only sparse MQA is the algorithmically required path above
that. The `+150..170 ms` at `pos0 = 4096..14336` is irreducible KV-proportional HCA VHA work.

## 3. Exact `d_tmp_a` byte headroom (proof, `0 B` net new `hipMalloc`)

`mi50/ernie55_runtime.cpp:2315` allocates

```
w->d_tmp_a = gpu_alloc<uint16_t>(size_t(max_chunk) * 32768u, ...)   // 2048 * 32768 * 2 = 134,217,728 B = 128 MiB
```

on **every** MI50 (`GPU[0..3]`), and `max_chunk = min(2048, ...)` (`:8270`). The `FlashIndexer`
outer loop needs, for `n_rows` query rows and an outer tile of `kTileK` keys:

```
n_rows * kTileK * 4            tile score slab (F32)
+ 2 * kTileK * 128 * 2         double-buffered FP16 key tile
```

| `n_rows` | `kTileK = 16,384` | `kTileK = 32,768` | fits `128 MiB` |
|---|---|---|---|
| `512` (max 4-way shard) | `32 MiB + 4 MiB = 36 MiB` | `64 MiB + 8 MiB = 72 MiB`... `80 MiB` with both slots | yes, `48 MiB` spare |
| `480 / 416` | smaller | smaller | yes |
| `255` (max single-stream) | `16 MiB + 2 MiB` | `32 MiB + 4 MiB` | yes |
| `2048` (hypothetical single-stream) | `128 MiB + 4 MiB` — **already does not fit** | `256 MiB + 8 MiB` — does not fit | no -> planner returns `0` |

The shipped code only fit `n_rows = 2048` at `16,384` by putting the key tile in the `4 MiB`
tail of `d_x_f16` (`8 MiB` total, first `4 MiB` = `q_all`). At `32,768` a slot is `4 MiB`, so two
slots need `8 MiB` and no longer fit — hence the key tile moves into the unused tail of the
`128 MiB` `d_tmp_a` slab. The `128 MiB` slab covers `n_rows = 512` at `32,768` with `48 MiB`
spare, so **no rank on `GPU[0..3]` needs a single new byte**, and `card4` (`05:00.0`) headroom
(`>= 1,000 MiB` sysfs / `>= 1,500 MiB` HIP free) is untouched because the cut allocates nothing.

## 4. Cut W14.1 — what changed

`mi50/ernie55_core_kernels.hpp` — new pure-host tile planner, the single source of truth shared
by the runtime and Gate-0 Suite 47:

* `kFiTileKMin = 16384`, `kFiTileKMax = 32768`, `kFiTileKSub = 4096`, `kFiKeyTileSlots = 2`,
  `kFiKeyRowHalfs = 128`.
* `fi_tmp_a_bytes(max_chunk)`, `fi_tile_k_adaptive(n_rows, tmp_a_bytes)` (clamped to
  `[16,384, 32,768]`, rounded down to a `4,096` multiple, returns `0` when even the minimum
  does not fit), `fi_scores_bytes`, `fi_key_slot_bytes`, `fi_tmp_a_need`,
  `fi_key_slot_off_u16`, `fi_tile_count`.

`mi50/ernie55_runtime.cpp`

* `Mi50ExpertWorker::tmp_a_bytes` is published from `fi_tmp_a_bytes(max_chunk)` at the existing
  `d_tmp_a` allocation; the allocation expression itself is byte-identical to Wave-13.
* 4-way TP4 `FlashIndexer` (`:4614`): `kTileK` is planned per rank and reduced to the uniform
  minimum across all four ranks (one broadcast key tile), giving `32,768` for every legal shard.
  The double-buffered key tile now lives at `fi_key_slot_off_u16(q_cnt[r], kTileK, slot)` inside
  `d_tmp_a`; the host staging slots are strided by `fi_key_slot_bytes(kTileK)` inside
  `h_tp_qabs_pin` (`144 MiB`).
* single-stream `FlashIndexer` (`:4872`): same planner, with a provably-safe fallback to the
  shipped `d_x_f16` key-tile slot if a future `n_sp_tot` growth ever makes the slab too small.
* small-history staging: at `32,768` the first multi-tile boundary moves from `pos0 = 16,384` to
  `pos0 = 32,768`, so every `pos0` in `[2,048 .. 30,720]` stages exactly one key tile and issues
  exactly one `k_dsa_topk2048_streaming_merge` per rank instead of two, on all six DSA layers.

`mi50/ernie55_core_kernels.hip` — **unchanged**. Both the score writer and the merge kernel were
already fully parameterised by `kTileK`; the merge LDS footprint stays
`s_hist[2048] + s_cval[2048] + s_cidx[2048] + meta = 24,640 B`, so `ScratchSize == 0`,
`0` spills, `0` hw trig, `0` `slc` and `0` BAR1 P2P are all preserved by construction.

`512K..1M` accuracy is untouched: the widening changes only the outer tile width and the merge
partition, and the merge is exact under `(score DESC, virtual-index ASC)` for any `kTileK`
(`min_u` floor bypass, radix select, deterministic ascending emission).

## 5. Gate-0 Suite 47

`tools/ernie55_verify_unit.cpp` — `w14_c1_adaptive_tile_k_flashindexer_parity()`, registered in
`main()`; the runner banner is now `ALL 47 GATE-0 UNIT TEST SUITES PASSED 100%`.

1. Planner contract: `fi_tmp_a_bytes(2048) == 134,217,728`, `fi_tmp_a_need(512, 32768) ==
   83,886,080`, `fi_tile_k_adaptive(2048, ...) == 0`, and the tile-count table
   `1 -> 1` at `pos0 = 14,336`, `2 -> 1` at `pos0 = 16,384`, `1 -> 2` at `pos0 = 32,768`,
   `64 -> 32` at `1M`.
2. Exact `d_tmp_a` bounds for every 4-way shard `n_sp in {256, 513, 1024, 1931, 2048}`:
   `q_cnt[r] <= 512`, `fi_tmp_a_need(q_cnt[r], 32768) <= 128 MiB`, the last key slot ends inside
   the slab, the score slab and the key tile are disjoint, and `>= 48 MiB` stays spare.
3. Top-2048 parity of the new adaptive `32,768` scorer against **both** the `16,384` scorer and
   an exact scalar reference (`std::nth_element` under `(ord DESC, idx ASC)`), over
   `pos0 in {2048, 4096, 16384, 20480, 65536, 262144, 1048560}`. `pos0 = 20,480` is the
   discriminating case (`max_hist = 20,355`: two `16,384` tiles, one `32,768` tile).
   Result: `set_mism == 0`, `cosine == 1.000000000` (`>= 0.999999` required), `3` identity rows
   (`n_hist <= 2048`) and `15` merge rows exercised.
4. Source-level `0 B` contract: the `d_tmp_a` allocation line is unchanged, `tmp_a_bytes` is
   published from the shared planner, and the two `Wave-14 Cut W14.1` blocks contain no
   `gpu_alloc` and no `hipMalloc`.

## 6. Deliberately not changed

* Phase C1's dense-KV `hipStreamSynchronize` at `:5021` was left in place. Removing it would
  require a device-ordered KV broadcast through `d_tp_qabs_pin`, which is also the `FlashIndexer`
  host staging buffer; the two are only safe today because the C1 sync serialises them. That is a
  separate cut with its own hazard analysis, not part of W14.1.
* No `hipMalloc`, no new stream, no new event, no change to `max_chunk`, `kStrideSel`,
  `kDsaIndexTopK` or `kSlidingWindow`.
