<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-13 Cut W13.4 — P150A Prefill Tail and MI50 Critical-Path Cuts

Milestone `v1.5.0-wave13-1k-prefill-1m`, workstation `SB-M5-P2-192G`, inside the kitchen for
bespoke ontology. Build #2 is live on the frontend and `p150_ernie55_worker` is live during this
turn, so no compilation, no GPU binary and no service restart was performed while authoring these
edits. The changes are staged for the between-turn Build #3 window.

## 1. Measured starting point

Build #2 live cold `M = 2,048` prefill: `2,066.2 ms` (`991.2 tok/s`) at `pos0 = 0` and
`2,125-2,211 ms` (`926-964 tok/s`) at `pos0 = 2048..16384`. The five MI50 workers are balanced
within `12 ms` (`g2 = 503 ms`, `g3 = 491 ms`, `g4 = 498 ms`, `mi50 = 478-507 ms`), so the
remaining gap to `1,050-1,150+ tok/s` (`<= 1,780-1,950 ms`) is dominated by two walls.

| Wall | Measured |
|---|---|
| Blackhole P150A prefill tail | `p150_wait = 112-215 ms/chunk` exposed past `mi50 = 478-507 ms` |
| P150A per-layer cost | `tot = 15.52-16.31 ms/L` (`652-685 ms/chunk`), `h2d = 1.60-1.66 ms/L`, `dev+red+d2h = 13.77-14.60 ms/L`, `slots = 204-237 / 211-243`, `runs = 132-141`, `bar = 29.6-33.7` |
| MI50 `pos0 = 2048` breakdown | `moe_post = 154.5 ms`, `attn_pre = 37.1 ms`, `qkv = 169.8 ms`, `attn = 93.6 ms`, `vha_op = 667.1 ms`, `attn_post = 43.8 ms`, `router = 61.8 ms (wait = 55.4 ms)`, `pack = 12.1 ms`, `fc1 = 99.9 ms`, `dsa.fi = 66 ms` |

## 2. Cut W13.4a — P150A prefill

### 2.1 Greedy descending-size minimum-coloring wave packing

`run_compact_prefill` in `p150/p150_ernie55_worker.cpp` previously assigned every tile of expert
`e`, visited in strictly ascending `eid` order, to wave `1 + max(row_last_wave[r])`. Because a
shared token row forces a strictly increasing wave, the wave count equals the longest ascending
chain through the row-conflict graph, which measured `29.6-33.7` per layer.

The new form visits experts in strictly descending tile count (ties broken by ascending `eid`)
and places each expert, with all of its tiles kept contiguous inside one wave, into the
lowest-index wave that shares no token row with any tile already placed there. This is greedy
graph colouring with waves as colours.

| Property | Status |
|---|---|
| Conflict freedom | every wave is strictly row-disjoint, asserted per tile in Gate-0 Suite 37 |
| Tile conservation | the placed tile count equals `sum(ceil(rows_e / 32))` exactly |
| Pair conservation | the placed `(row, weight)` count equals the routed pair total exactly |
| Contiguous per-expert runs | all tiles of an expert stay in one wave, consecutively, so GDDR6 weight reuse (`runs = 132-141`) is preserved |
| Wave-0 zero-read safety | wave 0 is emitted as the strict prefix of the placed stream and `y_acc` is zeroed before the layer, so the reader may skip its `y_acc` reads |
| Barrier count | barriers equal the wave count minus one, plus batch boundaries |

The accumulation order becomes colour order rather than `eid` order, so the parity bar against
the MI50 FP32 `atomicAdd` reference is `cosine >= 0.999999` rather than bit-exactness. The decode
path is untouched, so `M = 1..8` bit-exact invariance is preserved in full.

`[P150-PREFILL-XRAY]` now also reports `waves`, so the barrier reduction is observable per layer.

### 2.2 Wave-0 `y_acc` zero-read bypass

Every wave-0 tile is scatter-accumulated immediately after `enqueue_zero_y_accum`, so its rows
are guaranteed zero in GDDR6. `meta[1]` bit 1 now carries `is_first_wave` (bit 0 keeps
`need_barrier`, so the existing writer barrier test is masked with `& 1u`).
`ernie55_p150_compact_scatter_reader.cpp` then, for wave-0 tiles:

- skips the 32 random `128 B` GDDR6 reads from `y_acc`,
- replaces the RV32 `gather_l1` tilize loop with a direct `4 KiB` zero store into the combine tile,
- still synthesizes the `cb1` weight tile, because the combine multiplies the partial by the weight.

### 2.3 `cnt`-clamped tilize and untilize loops

`ernie55_p150_compact_gather_tilize.cpp` and `ernie55_p150_compact_scatter_reader.cpp` now clamp
the copy loops to the exact live row count (`fr_top = min(cnt, 16)`, `fr_bot = max(0, cnt - 16)`),
and zero unused destination rows directly in the output tile buffer instead of zeroing `gather_l1`
and copying zeros across, eliminating wasted RV32 loads and stores on partial tiles with identical
emitted bytes.

### 2.4 `kBatchSlots = 256` single-pass dispatch

The default rises from `128` to `256` (env override widened to `32..256`). A layer whose
scheduled tile count sits in `129..256` previously needed two full 5-kernel passes
(`10` dispatches + `10` barriers); it now collapses into one.

| Program | Circular-buffer total | Guard |
|---|---|---|
| compact-scatter | `155,648 B` (152.0 KiB: `cb5 = 69,632 B`, `cb6 = 40,960 B`, `cb7 = 16,384 B`, plus `28 KiB` tile CBs) | `< 204,800 B` |
| compact-gather | `38,912 B` (38.0 KiB: `256 x 128 B` meta + `6 KiB` scratch) | `< 204,800 B` |

Metadata arenas in `compact_gather_tilize` (`cb0`), `compact_scatter_reader` (`cb5`),
`compact_scatter_writer` (`cb6`), and `cb7` (`256 x 64 B = 16 KiB`) all scale cleanly to `256`
slots while keeping `48.0 KiB` of L1 headroom below the `200 KiB` `kP150Fabric1DCbGuardPerCore`
guard. The per-core `local_eids[]` arrays in `ernie55_p150_compute.cpp`,
`ernie55_p150_writer_b.cpp` and `ernie55_p150_reader_a.cpp` are sized `256` to match.
`kBatchSlots` still fits the `640`-entry compact slot arena.

### 2.5 Fused `8192 B` NOC reads in `ernie55_p150_writer_b.cpp`

The two `4096 B` nibble halves are contiguous in GDDR6, so the superblock loop now issues one
`read_slice(0u, cb4_wr + sb * 8192u, 8192u)` instead of two `4096 B` reads. The scales read at
offset `8192` is unchanged, and the bytes are identical.

### 2.6 Per-layer empirical L1 hot-expert selection

Tensix L1 SRAM pinning (`253.7 MiB`, `41` shards/card) previously pinned static `slot = 0`
(`eid = 318`) on every layer. `load_empirical_ranking` now loads
`router_hist_ernie55_agent.txt` (fallback `router_hist_ernie55.txt`) plus `router_perm.bin`, maps
raw expert ids into permuted slot space, and derives per layer:

- `emp_rank_[l]`: the local experts in `[318..512)` in strictly descending empirical frequency,
- `emp_shadow_[l]`: the MI50 top-7 shadow set in `[319..512)`, recomputed with the same rule as
  `assign_moe_layer_descending`.

`load_one_layer_file` then pins the empirically hottest `per_l` non-shadowed local experts, and
`pin_hot_from_layer1` populates `lut_u16[l][slot]` and `layers_[l]->hot_slot_for_eid[slot]` for
every pinned `(layer, slot)` pair rather than row 1 only. Any missing or malformed input leaves
`emp_ready_` false and the previous slot-0 form runs, so `--self-test` and
`--synthetic-self-test` remain bit-safe.

## 3. Cut W13.4b — MI50 critical path

### 3.1 Deferred `gate_proj` on `g0` and `g1`

In the HCA dual-owner VHA AllReduce, Step 1 launched
`f16_linear_f16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, int(gate_cols), 1024)` on all four ranks
before the Step 2 `sum4(VHA) + D2H`, delaying the `h_tp_qabs_pin` broadcast by about `0.50 ms`
per HCA layer, roughly `18.5 ms/chunk` over 37 layers.

`gate_proj` is now issued on `g2` and `g3` only inside the Step 1 loop (`r >= 2`), where it
overlaps behind the broadcast wait, and on `g0` / `g1` immediately after their canonical `sum4`
and `ev_mb_c2d` record and before the `stream_sh` wait. The deferred GEMM is the TP4 gate
projection `[M, 2048] x [2048, 1024]` FP16, exactly `4 MiB`. No extra VRAM is used.

### 3.2 Device-ordered `ev_xrot_host_ready` and `x_rot` D2H on `stream_sh`

`dispatch_routed_moe_all_devices` executed a host `hipEventSynchronize(ev_xrot_host_ready)`
before queueing the four non-owner MI50 workers, stalling CPU enqueue 42 times per chunk. The
`ch->x_rot_f16` D2H also rode `g_own.stream`, behind the routed-MoE launch.

Both D2Hs now ride `g_own.stream_sh`, and `ev_xrot_host_ready` is recorded on that same stream
after them. When `gfx906_ar_device_mode() == 2` the non-owner workers are released with a
device-side `hipStreamWaitEvent(w.stream, ev_xrot_host_ready, 0)` on their own stream; the host
sync is kept only for the non-SDMA AR modes. `g_own` therefore launches its routed MoE
immediately after `WHT128` with zero D2H stall on `g_own.stream` and zero host sync before the
non-owner H2D.

### 3.3 Companion cuts

No additional zero-VRAM prefill cut was taken in this pass. The `fc2_lat` microbatch sizing and
the DSA `fi` / `mqa` overlap at `512K-1M` were reviewed and left unchanged: both are already
device-ordered, and the remaining `vha_op` cost is dominated by the deferred `gate_proj` and the
`x_rot` D2H placement addressed above. Changing them without a live measurement would risk the
`ScratchSize == 0` and `0 B` VRAM contracts for no projected gain.

## 4. Verification checklist

| Item | Suite 37 assertion |
|---|---|
| Packer conflict freedom | every wave is strictly row-disjoint, checked per tile |
| Tile conservation | placed tiles equal `sum(ceil(rows_e / 32))` |
| Pair conservation | placed pairs equal the routed pair total |
| Contiguous per-expert runs | per-expert tile run length is exact, one wave per expert |
| Wave-0 zero-read safety | wave 0 is a non-empty strict prefix, bounded by the tile total |
| Barrier reduction | min colouring strictly reduces the barrier count and stays `<= 9` |
| Real-histogram behaviour | conservation holds and min colouring never needs more waves |
| `kBatchSlots = 256` budget | compact-scatter `155,648 B` (152.0 KiB) and compact-gather `38,912 B` (38.0 KiB), both `< 204,800 B` |
| `cb7` growth | `256 x 64 B = 16,384 B` exactly |
| Arena fit | `kBatchSlots <= 640` |
| W13.4b geometry | `gate_cols == 2048`, deferred GEMM exactly `4 MiB` FP16 |
| Shadow contract | per-layer shadow total `7`, `0` on `card4` `05:00.0` |
| Layer contract | `kNumRoutedExperts == 512`, `kTotalLayersWithEcho == 46` so ECHO is preserved |

Build #3 must additionally confirm:

1. `make -j8 all` under `-Wall -Wextra -Werror`.
2. `make audit-gfx906` with `344/344` zero-scratch `gfx906` kernels and `0` spills, `trig`, `slc` and Python files.
3. `bin/ernie55_verify_unit` reporting `ALL 37 GATE-0 UNIT TEST SUITES PASSED 100%`.
4. `p150_ernie55_worker --self-test` and `--synthetic-self-test` both passing bit-exact.
5. A live `M = 2,048` cold prefill run reporting `bar` down from `29.6-33.7` to `6-9`, `p150_wait` down from `112-215 ms`, and total time inside `1,780-1,950 ms`.

## 5. Files touched

| File | Change |
|---|---|
| `p150/p150_ernie55_worker.cpp` | min-coloring wave packer, `wave0_tiles` / `is_first_wave` meta bit, `kBatchSlots = 256`, `HotPinEntry`, empirical per-layer L1 ranking, per-layer LUT population, `waves` in the X-ray log |
| `p150/kernels/ernie55_p150_compact_scatter_reader.cpp` | wave-0 `y_acc` read bypass, zero combine tile, `cnt`-clamped tilize |
| `p150/kernels/ernie55_p150_compact_gather_tilize.cpp` | `cnt`-clamped copy loops |
| `p150/kernels/ernie55_p150_compact_scatter_writer.cpp` | `meta[1]` barrier mask `& 1u` |
| `p150/kernels/ernie55_p150_writer_b.cpp` | fused `8192 B` NOC read, `local_eids[256]` |
| `p150/kernels/ernie55_p150_compute.cpp` | `local_eids[256]` |
| `p150/kernels/ernie55_p150_reader_a.cpp` | `local_eids[256]` |
| `mi50/ernie55_runtime.cpp` | deferred `gate_proj` on `g0`/`g1`, `x_rot` D2Hs on `stream_sh`, device-ordered `ev_xrot_host_ready` |
| `tools/ernie55_verify_unit.cpp` | Gate-0 Suite 37, runner total raised to 37 |
| `docs/WAVE13_CUT_W13_4_1100TOKS_PREFILL.md` | this ledger |
