# Wave-13 — Goal B (`.kvblk` decoupled rings) landed edits + Cuts W13.1 / W13.3 implementation plan

Milestone: `v1.5.0-wave13-1k-prefill-1m` · workstation `SB-M5-P2-192G`
Constraints honoured this turn: no `make` / `hipcc` / `g++`, no `:8052` contact, zero Python,
zero new GPU `hipMalloc`, zero BAR1 P2P, ≤ 2 active streams per MI50 (`w.stream` + `w.stream_sh`
plus `keep_warm_stream`), zero scratch/spills/trig/slc across all `gfx906` kernels.

---

## Part 1 — Edits landed for Goal B

### 1.1 `mi50/ernie55_kvblk_cache.hpp`

| Area | Symbol | Change |
|---|---|---|
| Layout contract | `kRingSlots=12`, `kMixedRingSlots=8`, `kDdrRingSpan=104 MiB`, `kMixedRingSpan=72 MiB`, `kHcaRingDevOff=72 MiB`, `kHcaRingSpan=32 MiB`, `kTipSlabDevOff=112 MiB`, `kScratchSpan=128 MiB`, `kExtraVramBytes=0` | New public compile-time contract (line ~200-215). |
| Event pool | `ev_hca_half_h2d_[4][2]`, `ev_ring_h2d_[4][12]`, `ev_ring_unpack_[4][12]`, `ev_tip_h2d_[4]`, `ev_tip_unpack_[4]`, `ev_cap_pack_[2]`, `ring_ev_ok_` | Created in `init()` with `hipEventDisableTiming` after `hipSetDevice(tp_devs_[r])`; destroyed 1:1 in `shutdown()`. Cache-owned, so **no `ev_mb_*` attention/MoE slot is ever reused**. |
| `alloc_wb_slot` | `kWbFaultThreads = 16` | `#pragma omp parallel for schedule(static) num_threads(16) if (bytes >= 1 MiB)` — the 4 KiB first-touch now stripes the 512-slot (8.75 GiB) arena across all 8 EPYC 7542 DDR4 channels before `hipHostRegisterPortable`. |
| Tier-2 path cache | `struct CachedBlkPath`, `path_cache_`, `probe_disk_block_cached()` | The pre-`t0` prefix scan resolves **and caches** both stripe paths; `submit_block_read` then executes **zero `::stat`** after `t0` (the `resolve_blk_path` fallback remains for callers that bypass the scan). |
| CRC/H2D overlap | `reader_loop` / `crc_loop` | `wave_rq->push(stripe, idx, true)` moved to `reader_loop` immediately after `io_ok`; `crc_loop` now only publishes `stripe_state` and the once-per-block commit. `wait_block_job` stays the strict commit fence (`jp->state == 1` only after **both** stripes pass CRC32C), so a mismatch still aborts before any slot pin is released. |
| O_DIRECT tuning | `kChunkBytes` 2 MiB → **4 MiB**, `kReadersPerStripe` 8 → **10**, `kCrcWorkers` 6 → **8** | 3 preads/stripe instead of 5 (−40 % submission + completion-interrupt overhead); 10 stripes in flight per drive; 8 CRC workers still finish a stripe in ~0.4 ms. `preferred_drive` 1/12 ratio kept (it is the exact Bresenham solution of the measured 3.53 : 4.19 GB/s split — the Wave-12 `d1` tail was CRC latency, not byte imbalance). |
| `restore_longest_prefix` | new 6-arg overload `(tok_ids, existing, tp_dma_streams[4], tp_compute_streams[4], out_tel)`; 4-arg kept as a delegating wrapper | `st_comp = tp_compute_streams[r]` (= `w.stream`) carries `launch_kvblk_unpack_scatter_hbm2`; `st_dma = tp_dma_streams[r]` (= `w.stream_sh`) carries the SDMA H2D. |
| 12-slot DDR4 ring | `emit_ring_delta` | `ring_seq[s] % nring[s]` slot selection, `nring = clamp(span / delta_padded, 2, cap)`; per-slot `unpack_done` fence on `st_dma` before wrap, `h2d_done` fence on `st_comp` before the scatter. 26-block waves wrap the ring twice. |
| 8-slot mixed ring | same `emit_ring_delta` with `nring ≤ 8` | Spans `[0..72 MiB)` only, disjoint from the `[72..104 MiB)` HCA pack ring and the `[112..126.25 MiB)` tip slab. |
| Non-owner HCA pack ring | `emit_nonowner_hca_batched` | Per-group `ev_hca_half_h2d_` (H2D→MEC) + `ev_hca_half_` (MEC→next H2D) fences. The **mid-restore `hipStreamSynchronize(st)` fallback is gone** — the fence is always device-side. |
| Tip slab | `emit_tip_for` | `ev_tip_h2d_` / `ev_tip_unpack_` pair + `tip_armed` (the slab is shared by both stripes, so stripe 2 waits for stripe 1's tip scatter). |
| `capture_block_from_vram` | new 7-arg overload `(key, has_valid_tip, has_dspark_tip, s0_compute, s1_compute, s0_sh, s1_sh)`; 5-arg kept as a delegating wrapper | `ev_cap_pack_[s]` recorded on `w.stream` after every producing kernel, `w.stream_sh` waits on it, D2H issued on `w.stream_sh`, terminal fence on `w.stream_sh` only — `w.stream` is never drained by an async `.kvblk` D2H. |
| Fallback | `ring_ev_ok_` | If any event creation failed, `st_dma` collapses onto `st_comp` and every fence becomes a same-stream no-op. There is **no mid-restore host sync path at all**. |

### 1.2 `mi50/ernie55_runtime.cpp`

* `restore_longest_prefix` callsite (~line 8694-8712): now builds `tp_dma_streams[4]` from
  `dev_map[tp_devs[r]]->stream_sh` and `tp_compute_streams[4]` from `->stream`.
* `capture_block_from_vram` callsite (~line 8817): now passes `spine_dev0/1` `->stream` **and**
  `->stream_sh`.

### 1.3 `tools/ernie55_verify_unit.cpp` — Gate-0 Suite 35

`w13_goalb_kvblk_decoupled_rings()` (line ~3712), registered in `main()`; banner now
`ALL 33 GATE-0 UNIT TEST SUITES PASSED 100%`. It asserts:

1. **Layout / 0 B new VRAM** — `kRingSlots==12`, `kMixedRingSlots==8`, `kExtraVramBytes==0`,
   `kDelta==8,962,048 B`, `12·kDelta = 102.56 MiB < 104 MiB`, `8·kDelta = 68.34 MiB < 72 MiB`,
   `[0..72)` adjacent to `[72..104)`, HCA ring disjoint from tip slab, tip slab inside 128 MiB,
   12-slot ring disjoint from tip slab, `26 > 2·12`.
2. **26-block decoupled ring bit-exactness** — real GPU, H2D on `stream_sh`, MEC scatter on
   `stream`, 12 `h2d_done`/`unpack_done` event pairs, 26 blocks (2 full wraps), `memcmp == 0`,
   and **0 mid-ring host syncs**.
3. **Mixed 8-slot ring, permuted completion order** — 19 blocks in a fixed out-of-order
   permutation, 8-slot ring, per-slot fences, `memcmp == 0` for every block at its own
   destination, 0 mid-restore host syncs.
4. **CRC32C fail-closed under overlapped H2D + CRC** — `io_ok` published before the CRC thread
   finishes (H2D overlapped), clean stripe commits, 1-bit tail tamper rejected, commit gated on
   the CRC result.
5. **Zero new `hipMalloc` outside `d_tmp_a`** — `kExtraVramBytes == 0` plus the full span
   arithmetic; the cache itself contains no `hipMalloc` call site (verified by grep).

---

## Part 2 — Cuts W13.1 and W13.3 implementation plan

### 2.1 Cut W13.1 — `gfx906` Tensile solution table + `run_hca_wo_a_f16`

**Audit facts (all anchors `mi50/ernie55_runtime.cpp`)**

* `gfx906_tuned_gemm_solution` — lines **131-166**.
* `gfx906_tuned_sbat_solution` — lines **168-182**.
* Gates: `rocblas_gemm_ex_tuned` consults the table only when `a_type==f16 && b_type==f16 &&
  c_type==d_type` (**194-195**); `rocblas_gemm_strided_batched_ex_tuned` only when a/b/c/d are
  **all** f16 (**221-223**). Master switch `ERNIE55_ROCBLAS_SOL` (**64-70**).
* `run_hca_wo_a_f16` — lines **3400-3437**; the batched GEMM is at **3424-3432** with literal
  `m = kHcaOLoraRank = 1024`, `n = eff ≤ 1024`, `k = 4096`, `batch = g_loc = 2` at TP4.
* Second identical call site: **4233-4241** (HCA Phase C in `run_attention_tp`).
* Constants: `kHcaOGroups = 8` (`src/ernie55_config.hpp:56`), `kHcaOLoraRank = 1024` (`:57`),
  `kHiddenSize = 4096` (`:21`), `kLatentDim = 2048` (`:86`), `kHcaNumHeads = 64` (`:50`),
  `kHcaCompressRatio = 128` (`:58`), `kDsaIndexTopK = 2048` (`:76`).
  At TP4: `h_loc = 16`, `g_loc = 2`.

**Root cause.** The only `wo_a` rule is `if (m == 512 && k == 4096 && batch == 4 && n >= 512)
return 32975;` (**172**). The live shape is `m == 1024 && batch == 2`, so the table returns **0**
and the call falls back to `rocblas_gemm_algo_standard` — 1.736 ms instead of the measured
1.254 ms, i.e. **+0.482 ms × 2 microbatches × 37 HCA layers ≈ +35.7 ms/chunk**. The comments at
**3423** ("sol=32975") and **3470** ("sol=33065/32975") are stale.

**Exact edits**

1. `gfx906_tuned_sbat_solution` (line **172**) — replace the stale guard with the measured
   TP4 shape, keeping the old shape for backward compatibility:
   ```cpp
   if (k == 4096 && (batch == 2 || batch == 4) && (m == 512 || m == 1024) && n >= 512) return 32975; // HCA wo_a batched
   ```
2. `gfx906_tuned_sbat_solution` (line **174**) — `HCA SWA-128 QK b=8` measured **33065** at
   0.258 ms (not 32989):
   ```cpp
   if (m == 256 && n == 2048 && k == 512 && batch == 8) return 33065;  // HCA SWA-128 QK b=8
   ```
3. `gfx906_tuned_gemm_solution` (line **146**) — the `m == 1024 && k == 4096` rule currently
   returns 33068 (DSA `q_a`) for every `n >= 512`, which also captures the **FP16 MoE router**
   `[1024, 2048, 4096]` that measured **32975** at 1.253 ms. Split by `n`:
   ```cpp
   if (m == 1024 && k == 4096) return (n == 2048) ? 32975 : 33068; // MoE router FP16 / DSA q_a
   ```
   (the FP32 router keeps **31468** at line **162**, measured 1.627 ms — do not touch).
4. `gfx906_tuned_gemm_solution` — add the two missing `idx_*` shapes measured this wave
   (`DSA idx_wk F16 [128,2048,4096] = 32976 @ 0.397 ms`, `DSA idx_w_proj F16 [64,2048,4096] =
   32977 @ 0.275 ms`), placed **before** the generic `n >= 512` block:
   ```cpp
   if (k == 4096 && n == 2048 && (m == 64 || m == 128)) return (m == 128) ? 32976 : 32977;
   ```
5. `gfx906_tuned_gemm_solution` (line **144**) — `MoE fc2_lat` is `m = 4096, k = 2048` and
   already returns 33065 for `n >= 1024`; the measured **33066 @ 1.898 ms** for `M = 2048`
   is the `n < 1024` branch, which already returns 32975. **No change**, but update the stale
   comment to record the measurement.
6. `gfx906_tuned_gemm_solution` (lines **150-151**) — the `m == 1536` / `k == 1536` comments are
   stale: `sh_w1`/`sh_w3` are `m = 2048, k = 4096` and hit line **149** (32990), `sh_w2` is
   `m = 4096, k = 2048` and hits line **144**. Fix the comments only (no behaviour change).
7. `DSA fi_gate F32 [64,2048,4096]` measured **31503 @ 0.276 ms** — the f32 branch (**160-163**)
   only has the `m == 1024` router rule, so `fi_gate` currently gets 0. Add:
   ```cpp
   if (m == 64 && k == 4096 && n >= 1024) return 31503; // DSA fi_gate f32
   ```
8. `HCA comp PV [512,8192,1024]` measured **33172 @ 0.440 ms** — line **157** currently returns
   33161 for `m == 512 && k == 1024 && n >= 2048`. Split by `n`:
   ```cpp
   if (m == 512 && k == 1024 && n >= 4096) return 33172; // HCA comp PV (measured)
   if (m == 512 && (k == 64 || ... || k == 4096) && n >= 2048) return 33161;
   ```
9. `run_hca_wo_a_f16` (**3407-3409**) — `wo_strip` is 1024 only when `mb_m % 64 == 0`; the
   measured win is at `n = 1024` (i.e. `eff = 1024`). Add a `prof`-free comment update and keep
   the `cur < eff` pad path (it is required for the 64-row tail).

**Gate-0 Suite 33 assertions (`w13_c1_tuned_sbat_wo_a_and_router_shapes`)**

* `gfx906_tuned_sbat_solution(T, N, 1024, 1024, 4096, 2) == 32975` (the live TP4 `wo_a` shape).
* `gfx906_tuned_sbat_solution(T, N, 512, 1024, 4096, 4) == 32975` (legacy shape preserved).
* `gfx906_tuned_sbat_solution(T, N, 256, 2048, 512, 8) == 33065` (SWA QK b=8).
* `gfx906_tuned_sbat_solution(T, N, 256, 2048, 512, 16) == 33063` (SWA QK b=16 unchanged).
* `gfx906_tuned_gemm_solution(T, N, 1024, 2048, 4096, f16) == 32975` (FP16 MoE router).
* `gfx906_tuned_gemm_solution(T, N, 1024, 4096, 4096, f16) == 33068` (DSA `q_a` preserved).
* `gfx906_tuned_gemm_solution(T, N, 1024, 2048, 4096, f32) == 31468` (FP32 router preserved).
* `gfx906_tuned_gemm_solution(T, N, 64, 2048, 4096, f32) == 31503` (`fi_gate`).
* `gfx906_tuned_gemm_solution(T, N, 128, 2048, 4096, f16) == 32976` and
  `(T, N, 64, 2048, 4096, f16) == 32977` (`idx_wk` / `idx_w_proj`).
* `gfx906_tuned_gemm_solution(N, N, 512, 8192, 1024, f16) == 33172` (HCA comp PV).
* Negative guard: `gfx906_tuned_gemm_solution(T, N, 1024, 1024, 4096, f16) == 0` (non-batched
  router shape must not accidentally hit the batched rule).

### 2.2 Cut W13.3 — `run_moe_core` router window + `tp4_fi_ready` host stalls

**Audit facts**

* `run_moe_core` — definition **6815-7415**; call sites **2928** (DSpark draft, `draft_mi50_only`)
  and **7459** (`run_moe_tp`).
* Router window `[6836, 6940]`; `chunk_pre_router_ms` accumulated at **6941**.
  The dominant host block is `hipEventSynchronize(g.ev_mb_d2c[0])` at **6912** (variant **6933**,
  hard fallback `hipStreamSynchronize(g.stream)` at **6936**).
* The 320 KiB topk/mhc_mix D2H is issued on **`g.stream_sh`** (**6872-6886**), i.e. **behind** all
  the KV movement already queued on that stream: HCA/DSA o_proj 16 MiB D2H (**3974-3977**,
  **5853-5856**), the 16K-tile k staging D2H (**4613-4615**), and the DSA unique-KV pipe D2H of up
  to ~134 MB (**4996-5003**, cap at **4931**). `moe_bcast_sdma` defaults to true (**118-127**,
  used **7132**), so the microbatch host sync is skipped (**7156-7159**) and **6912 is the only
  host drain of `stream_sh` in the MoE**.
* `pos0` dependence: `n_dense = clamp(kStrideSel - pos0, 0, M)` with `kStrideSel = 2176`
  (**4269-4270**). `pos0=0` → `n_dense=M` → sparse C2 path skipped and `max_hist < 2048` so
  `tp4_fi` is skipped → ~10 ms. `pos0=2048` → `n_dense=128`. `pos0=4096` → `n_dense=0`, maximum
  unique-KV D2H → 78.8 ms.
* `2 × 1,024` `fc2_lat + d_sub_out` broadcast overlap: **7120-7184** (`kNb=2` at **7123**,
  `base_mb = (M/2) & ~63` at **7125**, `fc2_lat` at **7139**, `ev_mb_c2d[b]` record **7145**,
  `stream_sh` D2H **7147**, `ev_mb_vha_h2d[2+b]` record **7149**, second loop **7153-7174**,
  `hipEventSynchronize(g.ev_mb_d2c[b])` at **7158**, `kOrdH2d` H2D **7160-7163**).
* `kOrdDefer = {2,1,3,0}` at **6948-6952**; `kOrdH2d = {2,1,3,0}` at **7130**.
* `fc1_lat` launches **6895 / 6897 / 7216 / 7363**; `fc2_lat` launches **7139 / 7188 / 7344 /
  7404**.
* **`M < 512` tail coverage: not found.** There is no `M<512` branch and no scalar tail path in
  **6815-7415**. Small `M` is handled by the `M<=8` mapped/canonical decode paths (**7198**,
  **7356**) and by the `(M & 63) != 0` 64-row pad inside `f16_linear_f16` (**1533-1562**, 3 extra
  launches + a `hipStreamWaitEvent`). The only `M`-dependent fallback is
  `sh_direct_f16 = (M_sh <= max_chunk && M_sh <= 2048)` (**6844**).
* `tp4_fi_ready` — **4511**, block **4520-4706**, inside `run_attention_tp` (**3056-6083**).
  `kTileK = 16384` (**4524**), `n_tiles = (max_hist + kTileK - 1)/kTileK` (**4595**),
  `stage_tile_k` **4597-4629**, `hipEventSynchronize(g_own.ev_mb_vha_d2h[p])` at **4616**,
  `stage_tile_k(0,false)` **4631**, loop **4633-4673**, prefetch **4671**.
  At 1M: `max_hist = 1,048,448` → `n_tiles = 64`; 6 DSA backbone layers → **64 × 6 = 384 host
  stalls per chunk — verified**.
* Non-owner finalize: gather D2H on `w.stream` **4682-4685**, `hipStreamSynchronize(w.stream)`
  **4697** (≤3 per layer per chunk). Also **4547** (`tp4 fi scatter sync`) and **4824**
  (profiling only).
* DSA sparse MQA multi-slice syncs: `d_uniq_cnt` D2H + `hipStreamSynchronize(g_own.stream)` at
  **4944-4946** (up to ~7× per slice, 2048→…→32), non-owner slice drain
  `hipStreamSynchronize(w.stream)` at **5044** (3 per extra slice), dense-path sync **4843**.
* Event → stream map: `ev_mb_vha_d2h[p]` recorded on `g_own.stream_sh` **4615**, host-synced
  **4616**; `ev_mb_c2d[p]` recorded on `g_own.stream` **4660**, waited on `g_own.stream_sh`
  **4604** and `w.stream` **4642**; `ev_mb_d2c[p]` recorded on `w.stream_sh` **4627**, waited on
  `w.stream` **4642**; `ev_mb_o_d2h[p]` recorded on `g_own.stream_sh` **5002**, waited on
  `w.stream` **5013 / 5022**.

**Exact edits (ordered by payoff / risk)**

1. **Router D2H off the KV-SDMA backlog (biggest win, 3 lines).**
   Add `hipStream_t stream_rt` + `hipEvent_t ev_rt` to `Mi50ExpertWorker` (field block
   **798-800**, create with the other events at **2117-2119**, destroy with them). Change
   **6872-6886** so the 320 KiB topk/mhc_mix D2H is issued on `g.stream_rt` and
   `ev_mb_d2c[0]` is recorded there; keep `hipStreamWaitEvent(g.stream_sh, g.ev_mb_c2d[0])` at
   **6873**. `chunk_pre_router_ms` then measures the router window instead of the whole KV-SDMA
   backlog. Event lifecycle: `ev_rt` created once at worker init, recorded per layer, destroyed
   at worker teardown — no collision with `ev_mb_*`.
2. **Zero-extra-stream variant (if a new stream is not acceptable).** Move the
   `hipEventSynchronize(g.ev_mb_d2c[0])` at **6912** (and **6933**) to immediately before the
   pack loop at **6993**. The pack loop is the first consumer of `h_topk_ids_pin`, so this is a
   pure reorder that overlaps the wait with `fc1_lat` + WHT128.
3. **Remove the `can_overlap_pre == false` cliff** — **6845-6846 / 6936**: `ev_mb_c2d[]` /
   `ev_mb_d2c[]` are only created when `has_sh` (**2117-2119**). Add a one-time startup check
   that logs/asserts their presence so a missing event can never silently downgrade the router
   to a full `hipStreamSynchronize(g.stream)` inside the router window.
4. **Microbatch host stalls** — **7156-7159**: drop the `if (!moe_bcast_sdma)` guard and always
   use the device-ordered path (`ev_mb_vha_h2d[2+b]` + `hipStreamWaitEvent(w.stream, …)` at
   **7169**); removes 2 host stalls per layer from `chunk_moe_post_ms` (**7177**).
5. **`M < 512` tail coverage** — pad `M` to `M_sh` once at the top of `run_moe_core` (as already
   done for `d_x_f16` at **6843**) and pass `M_sh` to the `fc2_lat` call at **7188**, so the
   `(M & 63) != 0` branch in `f16_linear_f16` (**1533-1562**) disappears for non-multiple-of-64
   chunks.
6. **`tp4_fi` 384 → 192 stalls** — **4616**: the sync is only needed before the slot is reused,
   i.e. when `ti >= 1`. Guard it with the `wait_prev_use` flag the lambda already receives:
   `if (wait_prev_use) hipEventSynchronize(...)`. 32 stalls/layer at 1M instead of 64.
7. **`tp4_fi` finalize** — **4682**: issue the gather D2H on `w.stream_sh` (matching **4613**),
   record `w.ev_mb_d2c[0]`, and replace `hipStreamSynchronize(w.stream)` at **4697** with
   `hipEventSynchronize(w.ev_mb_d2c[0])` — only the DMA stream is awaited.
8. **Multi-slice guard** — **5044**: record `w.ev_mb_d2c[0]` on `w.stream` after the MQA slice
   and replace the host stream sync with `hipEventSynchronize(w.ev_mb_d2c[0])`.
9. **Compaction count readback** — **4944-4946**: move the 4-byte `d_uniq_cnt` D2H to
   `g_own.stream_sh` + `ev_mb_d2c[0]` and sync the event; removes up to ~7 full compute-stream
   drains per slice.
10. **Observability** — accumulate the raw `hipEventSynchronize` duration at **6912** into a new
    `chunk_pre_router_wait_ms` (declarations **1013-1017**, printout **8628-8629**) to confirm
    empirically that 59.1 / 78.8 ms is SDMA backlog rather than router GEMM.

**Gate-0 Suite 34 assertions (`w13_c3_router_window_and_tp4_fi_fences`)**

* Structural: `Mi50ExpertWorker` exposes `stream_rt` and `ev_rt`; both are created with
  `hipEventDisableTiming` and destroyed exactly once (grep-level assertion on the header).
* `run_moe_core` contains **no** `hipStreamSynchronize` inside `[6836, 6940]` (router window) —
  verified by scanning the function body for the window bounds.
* `chunk_pre_router_wait_ms` is reported separately from `chunk_pre_router_ms` in the
  `[ERNIE55-MOE-PRE-XRAY]` line.
* `tp4_fi_ready` path: the number of `hipEventSynchronize(g_own.ev_mb_vha_d2h[p])` calls per
  layer is `max(0, n_tiles - 1)` (32 at 1M, not 64); a synthetic `n_tiles = 64` case asserts the
  counter equals 32.
* `tp4_fi` finalize uses `hipEventSynchronize(w.ev_mb_d2c[0])` and **not**
  `hipStreamSynchronize(w.stream)`.
* Multi-slice path uses `hipEventSynchronize` on `ev_mb_d2c[0]` and never a full
  `hipStreamSynchronize(w.stream)` per slice.
* `M < 512` coverage: `run_moe_core` with `M = 384` produces the same `d_sub_out` bits as
  `M = 512` on the first 384 rows (64-row pad path), and `fc2_lat` is launched with `M_sh`.
* Event-collision guard: the set `{ev_sh_start, ev_sh_done, ev_mb_c2d[0..3], ev_mb_d2c[0..3],
  ev_mb_vha_d2h[0..3], ev_mb_vha_h2d[0..3], ev_mb_o_d2h[0..3], ev_res_b_ready, ev_rt}` has no
  duplicates.

---

## Part 3 — Verification still to run between turns

1. `make audit-gfx906` (zero scratch/spills/trig/slc across all `gfx906` kernels).
2. Gate-0 build + `bin/ernie55_verify_unit` → expect `ALL 33 GATE-0 UNIT TEST SUITES PASSED 100%`.
3. `bin/ernie55_kvblk_verify --unit` and `--blocks 26` (backward-compatible overloads).
4. Coherence gates + `:8052` swap, then the live 1M prefill run to confirm
   Tier-1 `4.5M-6.0M+ tok/s` and Tier-2 `1.05M+ tok/s`, and that `d1_end - d0_end` no longer
   shows the ~35 ms tail.
