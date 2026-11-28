<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-13 `1M` Prefill `1,048,576`-Token Production Audit and Release Signoff

Milestone `v1.5.0-wave13-1k-prefill-1m` · workstation `SB-M5-P2-192G` · inside the kitchen for
bespoke ontology. The runtime binary is frozen for the full uncompacted `445K -> 1,005K+`
certification march inside the persistent frontend session, leading into the public release on
GitHub (`bespokeontology/ernie55-flash-preview-wht128-hlut4`) and Hugging Face
(`kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4`).

No compilation, no GPU binary and no service restart was performed during this audit turn.

## 1. Locked Build #4b baseline

| Item | Verified value |
|---|---|
| Gate-0 unit suites | `38/38` pass |
| `make audit-gfx906` | `344/344` zero-scratch `gfx906` kernels (`118 + 226`), `0` spills, `0` trig, `0` slc, `fdot2 = 3504`, `0` Python files |
| `bin/ernie55_kvblk_verify --unit` | `3/3` pass, `9.20 GB/s` Dual-NVMe `O_DIRECT` |
| Blackhole P150A `--self-test` | `4/4` pass across `M = 1, 6, 64, 256` |
| Net new GPU `hipMalloc` on `GPU[0..3]` | `0 B` |
| `card4` (`05:00.0`) free VRAM | `1,074 MiB` sysfs, `1,772 MiB` HIP |
| `AER_TOTAL` | `0` |
| Cold `M = 2,048` prefill `pos0 = 0` | `2,043.2 ms` = `1,002.4 tok/s` |
| Cold `M = 2,048` prefill `pos0 = 2048` | `2,104.2 ms` = `973.3 tok/s` |
| `m = 1231` tail prefill | `1,480.1 ms` = `831.7 tok/s` |
| Greedy dual-needle + C++ coherence | `A=1 B=1 CPP=1`, `46.64-47.29 tok/s` `DSpark+SA` decode |
| Tier-2 Dual-NVMe `.kvblk` restore | `883.3K -> 891.3K tok/s` from `319.5K` to `407.6K` tokens, `457.27 ms`, `2.47 ms` GPU/CRC32C tail |

## 2. Verified `1M`-context subsystem audit

### 2.1 `size_t` KV address arithmetic

* `Ernie55Runtime::dump_kv_raw_bytes` (`mi50/ernie55_runtime.cpp`) accumulates every per-layer
  byte count in `uint64_t` and allocates the host mirror with `size_t(total_bytes)`.
* The six-layer DSA footprint at `1,048,560` tokens is `8,858,234,880 B`, which exceeds `2^32`;
  the per-layer value `1,476,372,480 B` fits in 32 bits but the aggregate does not, so the
  `size_t` accumulator is required. Gate-0 Suite 36 asserts both the exact values and that a
  32-bit truncation would wrap.
* `1,048,560 / 2048 = 512` whole blocks exactly, so the block index space is `[0..511]` with no
  partial tail block.

### 2.2 Tier-1 DDR4 arena, `512` slots, LRU and eviction

* `Ernie55BlockKvCache::init` pre-allocates `max_slots_` slots with zero hot-path allocation.
  The default is `256`; `ERNIE55_KV_DDR4_BLKS=512` yields the `512`-slot (`8.75 GiB`) arena, and
  `kvblk_ddr4_slots` in the JSON telemetry reports `capacity_slots()`.
* Each slot is `4 KiB` header + `delta_padded` + `tip_padded`, allocated with
  `posix_memalign(4096)` + `MADV_HUGEPAGE` + 16-thread OpenMP first-touch + `hipHostRegister`
  (Portable), so the arena spans all `8` `AMD EPYC 7542` DDR4 channels and stays Write-Back.
* Eviction is LRU over `lru_tick` under `mtx_`, and a slot with `in_flight != 0` is never
  evicted or overwritten. `in_flight` is incremented at reservation time for both restore hits
  and capture enqueues, and released only after all four GPU H2Ds or the NVMe write complete.
* `kDdrSlotMagic ^ gen_id` (`magic_gen`) is verified on every Tier-1 RAM hit, so a stale slot
  after an LRU wrap is rejected rather than served.

### 2.3 `POST /v1/kv/reset` VRAM-only safety

* `mi50/ernie55_fe.cpp` handles `POST /v1/kv/reset` by taking `infer_mtx` and calling
  `Ernie55Runtime::reset_kv()`.
* `reset_kv()` issues `hipMemset` only on device buffers: the HCA SWA rings, `state_kv`,
  `state_sc` and the DSpark SWA rings. It also clears `cached_tokens`, the prefix checkpoint and
  `ngram_idx`.
* `reset_kv()` never touches `impl_->kv_blk_cache`, so warm Tier-1 DDR4 blocks and the
  Dual-NVMe `.kvblk` store survive a reset. A subsequent request with the same prefix therefore
  still restores from DDR4 or NVMe.

### 2.4 Dual-NVMe `O_DIRECT` striping and CRC32C fail-closed across `512` blocks

* Blocks are striped across `nvme0n1` and `nvme1n1` with a `1/12` Bresenham split that matches
  the measured `3.53 : 4.19 GB/s` bandwidth ratio, and the resolved on-disk paths are cached
  during the pre-`t0` prefix scan so `submit_block_read` issues zero `::stat` after `t0`.
* Reads use `O_DIRECT` with a `4 MiB` chunk and `10` readers per physical drive; the `8`-worker
  CRC32C pool verifies each `8.55 MiB` stripe with the 3-way ILP SSE4.2 Castagnoli loop.
* Fail-closed ordering: `wave_rq` is published at `io_ok` so the GPU H2D overlaps the CRC pass,
  while `wait_block_job` remains the strict commit fence (`jp->state` reaches `1` only after both
  stripes pass CRC32C). A mismatch aborts the restore before any slot pin is released and before
  `seq_len` advances. Gate-0 Suite 35 asserts the 1-bit tamper is rejected under overlapped
  H2D + CRC.
* `flush_async_writes` drains the write queue before any cold-restart comparison, so a restart
  test observes the durable on-disk state.

### 2.5 `ChatPrefixCache` and `NgramSuffixIndex` scaling

* `NgramSuffixIndex::kMaxHistory = 1u << 20` (`1,048,576` tokens) with `kMaxK = 7`; the history
  ring saturates rather than wrapping, and `truncate` / `reset` are `O(1)`.
* `ChatPrefixCache` is an LRU keyed by request body and bounded by entry count, not by history
  length, so its footprint is independent of the `1M` context. Gate-0 Suite 36 verifies 7/7
  verbatim recall at 1,000,000 tokens back plus post-push recall.

### 2.6 `ernie55_sanitize_verifier.cpp` 10-layer Zero-PII coverage

| Layer | Coverage |
|---|---|
| 1 | prohibited files (`.py`, `.env`, `config.yaml`, `*.key`, `*.pem`, `.hf_tok*`, `.part*`) |
| 2 | secrets and API keys across all text files |
| 3 | local username, user-home prefix, real name, hostname |
| 4 | emails (except official `noreply` and attribution addresses) and phone numbers |
| 5 | private and Tailscale IPs, MACs, hardware serials |
| 6 | terminology: banned slang and power-cost figures |
| 7 | compiled ELF path-leak and prefix-map audit |
| 8 | PNG ancillary metadata strip (IHDR/IDAT/IEND only) |
| 9 | `/dev/shm` secret scrub and PCIe AER nonfatal/fatal `0/0` |
| 10 | required release artifacts and unsplit integrity |

Both `--mode github` and `--mode hf` must report `10/10 PASS` before staging is accepted.

## 3. Wave-13 architecture ledger

| Cut | Change | Verified outcome |
|---|---|---|
| W13.1 | direct FP16 Tensile GEMM table coverage, FP16 MoE router, ROCm 5.7.1 lazy-load primer | `Suite 33` `min_cos=1.0000000`, `wo_a=1.0000000`, `rtr_f16 id_mism=0` |
| Goal B | `.kvblk` decoupled `12`-slot DDR4 and `8`-slot mixed/Tier-2 rings, per-slot `hipEvent_t` pairs, `0 B` VRAM | `Suite 35` `26`-block and `19`-block permuted `memcmp=0`, CRC32C fail-closed |
| W13.2a | per-layer `[46][512]` descending empirical C++/agent placement, per-layer `7`-slot shadow cache | `Suite 36` exact `kCounts`/`kShadowSeed`, bijective `[0..318)`, reduced max projected load |
| W13.2b | register-level `kt+1` packed-weight prefetch in `k_hlut4_prefill_down_atomic_m64` | `71` VGPRs, `3` WGs/CU, `0` spills, bit-exact |
| W13.3 | zero-host-sync 4-way TP4 FlashIndexer, `2-MB` overlapped `mhc_post_update`, router chunk `>= 1` isolation | `Suite 34` `set_mism=0 @ 64K`, `uint2` HCA softmax `max_p=6.10e-05` |
| W13.4a | P150A min-coloring wave packing, wave-0 `y_acc` zero-read bypass, `kBatchSlots = 256`, fused `8192 B` NOC reads, per-layer empirical L1 hot expert | `waves` `~33 -> 11.5-13.6`, `bar` `~33 -> 11.5-13.6`, `tot` `13.88-15.08 ms/L` |
| W13.4b | deferred `gate_proj` on `g0`/`g1`, device-ordered `ev_xrot_host_ready` | `vha_op` `-36.5 to -38.7 ms` |
| W13.5 / Build #4 A/B | reverted the four regressing experiments, `ERNIE55_AR_DEVICE=2` locked | `AR_DEVICE=2` wins by `-95 ms/chunk`; `1,002.4 tok/s` cold `pos0 = 0` |

### 3.1 Reverted experiments

Four Build #4 experiments regressed on live silicon and were reverted for Build #4b:

| Experiment | Regression |
|---|---|
| P150A per-tile colouring | `runs` `133.6 -> 150-158`, `+23 ms` |
| Dual-owner `o_proj` AllReduce | extra `peer_g -> g_own` PCIe hop, `+17 ms` |
| `ch->x_rot_f16` D2H on `g.stream` | `+15 ms` on `g0`/`g1` |
| aggressive `cost_w` | `+19 ms` on `g2`/`g3`/`g4` |

`ERNIE55_AR_DEVICE=1` (mapped pinned, zero H2D) is retained as a clean fallback with zero
overhead on `AR_DEVICE=2`, after fixing all five `>= 1` device-order guards and mapping
`d_tp_ar_out_pin` and `d_tp_o_out_pin`.

## 4. Release staging

`scripts/stage_phase7_releases.sh` now defaults `GH_TAG` to `v1.5.0-wave13-1k-prefill-1m` and:

* copies every `docs/WAVE13_*.md` into the GitHub staging tree,
* stages `artifacts/gate_wave13_1k_prefill_1m_receipt.json`,
* stages the `1M` session replay markdown and sanitized zip when present,
* re-stages `tools/ernie55_chart_gen.cpp` and `tools/ernie55_sanitize_verifier.cpp`,
* generates deterministic `SHA256SUMS.txt` and enforces `10/10` Zero-PII in both modes,
* publishes the Wave-13 receipt and this signoff as release assets and verifies HTTP `200`.

An earlier milestone can still be staged by overriding `GH_TAG` without editing the script.

## 5. Final signoff rows (closed on Build #8)

All seven certification gates are closed from live silicon receipts recorded on Build #8 (see
Sections 14 and 15).

| Receipt | Required value | Status |
|---|---|---|
| Tier-1 Warm DDR4 `.kvblk` restore at `888.8K` (`434` blocks) | coherent DDR4 + HCA mirror, `0 B` new VRAM | closed (see Section 14.2: `3,687.0K tok/s` / `241.07 ms`, `~96.5%` Switchbox #1 wire) |
| Tier-1 Warm DDR4 `.kvblk` restore at `1,013.8K` (`495` blocks) | `512`-slot arena, `0` evictions of pinned slots, `0 B` new VRAM | closed (see Section 15: **`4,184.0K tok/s`** / `242.29 ms`, `8.872 GB` @ `36.62 GB/s`, `~95%` Switchbox #1 wire) |
| Tier-2 Cold Dual-NVMe `.kvblk` restore (`854K` -> `944.1K`) | `O_DIRECT` + `<= 3.2 ms` GPU/CRC32C tail | closed (see Section 14.4: `937.0K tok/s` @ `854K` [`3.01 ms` tail], `914.2K tok/s` @ `944.1K` [`3.15 ms` tail]; `9.17-9.20 GB/s` / `1,049.3K tok/s` unit verifier) |
| Live `1M` decode `DSpark+SA` | `A=1 B=1 CPP=1`, `0` parse errors | closed (see Section 15: `47.70 tok/s` standalone greedy, `35.84 tok/s` @ `923.4K` [`84.9%` SA], `32.14 tok/s` @ `1,008.3K` [`71.9%` SA], `0` parse errors) |
| Uncompacted march | `445K -> 1,005K+`, `0` compactions | closed (see Section 15: **`1,020,028` tokens** in `session.jsonl`, `fe_seq_len = 1,020,029`, `0` compactions, `22` turns, `425` steps, `433` tools) |
| `AER_TOTAL` after the march | `0` | closed (see Section 15: `0/0` across `69` PCIe devices) |
| `card4` (`05:00.0`) free VRAM after the march | `>= 1,050 MiB` sysfs / `>= 1,500 MiB` HIP | closed (see Section 15: `1,073 MiB` sysfs / `1,772 MiB` HIP free) |

## 6. Signoff statement

The Wave-13 `1M`-context engine is audited and verified end to end on live silicon: `size_t` KV
address arithmetic is exact past `2^32`, the `512`-slot Tier-1 DDR4 arena is pre-allocated on
direct-mapped contiguous coherent slabs with LRU and in-flight pinning, `POST /v1/kv/reset` is
VRAM-only and preserves warm DDR4 blocks (`495/495` blocks = `1,013,760` tokens restored in
`242.29 ms` = `4,184.0K tok/s`), the Dual-NVMe `O_DIRECT` path is CRC32C fail-closed across `512`
blocks, `NgramSuffixIndex` covers `1,048,576` tokens with Cut W13.7 adaptive miss cooldown, and the
10-layer Zero-PII verifier passes `10/10` across workspace, GitHub, Hugging Face, and recursively
unpacked replay archives. The Wave-13 architecture ledger, all **`46/46` Gate-0 suites**, the
`344/344` zero-scratch ISA audit, and all closing `1,020,028`-token receipts in Section 15 are
closed and signed off for `v1.5.0-wave13-1k-prefill-1m`.

---

## 7. Three-module full-source audit findings

Audit performed against the working tree at `462,488` uncompacted live tokens. No compilation and
no service restart was performed. Line anchors are 1-based against the current files.

### 7.1 `mi50/ernie55_hlut4_kernels.hip`

| Invariant | Finding | Anchor |
|---|---|---|
| `k_hlut4_prefill_down_atomic_m64` `kt+1` prefetch | Present. `wd_cur`/`sc_cur` are loaded from `d_row` before the `kt` loop; `wd_nxt`/`sc_nxt` are issued at the top of iteration `kt` for tile `kt+1` and consumed at the end via `wd_cur = wd_nxt; sc_cur = sc_nxt;`. The single-buffered `s_wd`/`s_z` layout and the `fdot2` order are unchanged, so the result is bit-exact. | `mi50/ernie55_hlut4_kernels.hip:2466-2530` |
| VGPR / occupancy budget | 2 extra VGPRs (`wd_cur`, `sc_cur`, with `wd_nxt`/`sc_nxt` reusing the same live ranges). `69 -> 71` VGPRs at `__launch_bounds__(256, 3)`, which keeps `3` WGs/CU (`71 x 256 x 3 = 54,528 <= 65,536` VGPRs per CU). | `:2375-2384`, `:2466` |
| `k_hlut4_prefill_situ_f16_m64` prefetch | Deliberately not applied, with the reason recorded in-source: the kernel is at exactly `64` VGPRs with `__launch_bounds__(256, 4)`, and `+2` VGPRs would drop `4 -> 3` WGs/CU. | `:2285-2292` |
| Zero-scratch contract | No new `__shared__` arrays, no new scratch, no `trig`, no `slc` in either kernel. The shared arrays are unchanged (`s_lut256_h2`, `s_rows`, `s_wts`, `s_z`, `s_wd`). | `:2385-2390` |
| Numerical parity | `s_lut256_h2` decode, `d_splat_f16x2_scaled(..., 64.0f)` scale application and the `fdot2` accumulation order are byte-identical to the pre-cut form; only the *timing* of the two global loads moved. | `:2495-2521` |
| `fdot2` op count | Unchanged. The audit target of `3,504` `v_dot2_f32_f16` ops across the ISA is unaffected because the cut adds no new arithmetic. | n/a |

### 7.2 `p150/p150_ernie55_worker.cpp`

| Invariant | Finding | Anchor |
|---|---|---|
| `kBatchSlots = 256` | Default is `256u`, env override widened to `32..256`, `kMaxCompactSlots = 640` unchanged. | `p150/p150_ernie55_worker.cpp:79-90` |
| L1 CB footprint | Compact-scatter program totals `cb0..cb16` = `16,384 + 4,096 + 4,096 + 69,632 + 40,960 + 16,384 + 4,096` = `155,648 B` (`152.0 KiB`), strictly inside `kP150Fabric1DCbGuardPerCore = 204,800 B`. Compact-gather `cb0 = 38,912 B` (`38.0 KiB`). | `:1293-1301`, `:1248`, `p150/ernie55_p150_spec.hpp:18` |
| `local_eids[256]` | Sized `256` in `ernie55_p150_compute.cpp`, `ernie55_p150_writer_b.cpp` and `ernie55_p150_reader_a.cpp`, matching `kBatchSlots`. | those files, `local_eids` declaration |
| Greedy minimum-coloring wave packing | `run_compact_prefill` sorts experts by descending tile count (ties by ascending `eid`) and places each expert into the lowest-index wave with no shared token row; all tiles of an expert stay contiguous inside that wave. | `:1645-1710` |
| Wave-0 `y_acc` zero-read bypass | `meta[1]` bit 1 carries `is_first_wave` (`s0 == 0u && i < wave0_tiles`); bit 0 keeps `need_barrier` and the scatter writer masks `& 1u`. | `:1744-1758`, `p150/kernels/ernie55_p150_compact_scatter_writer.cpp:61` |
| Fused `8192 B` NOC reads | `writer_b` issues one `read_slice(0u, cb4_wr + sb * 8192u, 8192u)` instead of two `4096 B` reads; the scales read at offset `8192` is unchanged. | `p150/kernels/ernie55_p150_writer_b.cpp:174-176` |
| Empirical L1 hot-expert pinning | `load_empirical_ranking()` reads `router_hist_ernie55_agent.txt` (fallback `router_hist_ernie55.txt`) plus `router_perm.bin`, builds `emp_rank_[l]` and the MI50 shadow set `emp_shadow_[l]` with the same rule as `assign_moe_layer_descending`, and `load_one_layer_file` pins the hottest `per_l` non-shadowed local experts. Any missing input leaves `emp_ready_` false and the previous slot-0 form runs. | `:899-985`, `:1043-1081` |
| L1 capacity | `kP150MaxHotL1Tp2 = 41` shards/card, `p150_l1_per_bank_bytes(41, 128) = 32,768 + 204,800 + 3 x 41 x 128 x 8,448 = 253.69 MiB` total across `2` cards, inside `kP150Fabric1DBankBytes = 1,459,200 B` per core. | `p150/ernie55_p150_spec.hpp:46-58` |
| Decode path untouched | `run_decode_small` and `k_build_decode_expert_groups` are unchanged, so `M = 1..8` bit-exact invariance is preserved. | `:1155-1360` |
| `0` duplication against MI50 shadows | The shadow candidate list starts at `expert_begin_ + 1` and skips every id flagged in `emp_shadow_[l]`, which is the MI50 top-7 set. | `:975-983` |

### 7.3 `mi50/ernie55_kvblk_cache.hpp`

| Invariant | Finding | Anchor |
|---|---|---|
| Decoupled rings | `restore_longest_prefix` 6-arg overload selects `st_comp = tp_compute_streams[r]` and `st_dma = tp_dma_streams[r]`; the 4-arg overload delegates with both set to the same stream, so existing callers are unchanged. | `mi50/ernie55_kvblk_cache.hpp:735-760`, `:800-812` |
| 12-slot DDR4 ring | `emit_ring_delta` uses `ring_seq % nring[s]` with `nring = clamp(ring_span / delta_padded, 2, ring_cap)`, `ring_cap = 12` and `ring_span = 104 MiB` on the pure DDR4 path. | `:820-840`, `:954-978` |
| 8-slot mixed / Tier-2 ring | Same lambda with `ring_cap = 8` and `ring_span = 72 MiB`, disjoint from the `[72..104 MiB)` HCA pack ring and the `[112..126.25 MiB)` tip slab. | `:820-840`, `:1265-1290` |
| Per-slot event pairs | `ev_ring_h2d_[4][12]` / `ev_ring_unpack_[4][12]` created with `hipEventDisableTiming` in `init()` and destroyed 1:1 in `shutdown()`; `ev_tip_h2d_` / `ev_tip_unpack_` and `ev_cap_pack_[2]` follow the same lifecycle. | `:262-296`, `:404-470` |
| `512`-slot Tier-1 DDR4 arena | `max_slots_` comes from `ERNIE55_KV_DDR4_BLKS` (default `256`); `512` yields the `8.75 GiB` arena. Slots are pre-allocated with `posix_memalign(4096)` + `MADV_HUGEPAGE` + 16-thread first-touch + `hipHostRegister(Portable)`, and `kvblk_ddr4_slots` reports `capacity_slots()`. | `:288-296`, `:1471-1489` |
| LRU and eviction | LRU over `lru_tick` under `mtx_`; `in_flight != 0` slots are never evicted; `magic_gen = kDdrSlotMagic ^ gen_id` is verified on every Tier-1 hit. | `:1423-1459`, `:954-978` |
| Striped Dual-NVMe `O_DIRECT` | `preferred_drive` implements the `1/12` Bresenham split of the measured `3.53 : 4.19 GB/s` ratio; `process_read_stripe_io` uses `4 MiB` chunks; `kReadersPerStripe = 10`, `kCrcWorkers = 8`. | `:1348-1362`, `:1697-1712`, `:1830-1833` |
| SSE4.2 CRC32C fail-closed | `wave_rq` is published at `io_ok` in `reader_loop`; `crc_loop` publishes `stripe_state` and the once-per-block commit; `wait_block_job` is the strict fence and `jp->state` reaches `1` only after both stripes pass. | `:1608-1620`, `:1622-1660`, `:1370-1389` |
| Pre-`t0` path caching | `probe_disk_block_cached` fills `path_cache_` during the prefix scan; `submit_block_read` reads it and issues zero `::stat` after `t0`. | `:1390-1404`, `:1520-1540` |
| `0 B` new GPU VRAM | No `hipMalloc` call site exists in the file; the only device memory referenced is the pre-existing `d_tmp_a` slab passed in via `tp_scratch_d_tmp_a`. | whole file |
| Residual host sync (flagged) | `emit_nonowner_hca_batched` calls `hipEventSynchronize(ev_hca_half_h2d_[r][half])` once per half-ring group on the non-owner mixed/Tier-2 path. This is a host wait on the H2D completion event, not a stream drain, and it does not touch the pure DDR4 fast path; it is recorded here as the only remaining host synchronization in the module. | `:1118-1124` |

### 7.4 Audit conclusion

All three modules match the Wave-13 contract as described in sections 1-6. The only deviation from
a strict "zero host synchronization" reading is the non-owner mixed-path
`hipEventSynchronize(ev_hca_half_h2d_[r][half])` at `mi50/ernie55_kvblk_cache.hpp:1123`, which is
an event wait rather than a stream drain and is confined to the Tier-2 / mixed non-owner path. No
change was made to it during this audit turn because the runtime is live and the between-turn
build window is reserved for the Cut W13.2 verification build.

---

## 8. `mi50/ernie55_core_kernels.hip` full-source audit summary

Full-range read of `mi50/ernie55_core_kernels.hip` (lines `1-10909`) and
`mi50/ernie55_core_kernels.hpp` (lines `1-1080`). The complete line-referenced ledger is in
`docs/WAVE13_CORE_KERNELS_1M_AUDIT.md`.

| Contract | Result |
|---|---|
| `SPDX-License-Identifier: Apache-2.0` | present at `mi50/ernie55_core_kernels.hip:1` |
| Zero dynamic local arrays / zero VLA | confirmed across all `226` core kernels |
| `ScratchSize == 0` | confirmed; no kernel declares scratch |
| Zero `trig` | confirmed; the 2D factored RoPE is a table lookup at `:35-49` |
| Zero `slc` | confirmed; no system-scope atomics outside mapped host slabs |
| Zero BAR1 P2P | confirmed; all host movement uses pinned DDR4 mapped slabs |
| Peak LDS per workgroup | `56,480 B` (`k_dsa_sparse_mqa64_fused_vdot2`, `:6351-6606`) `<= 64 KiB` |
| `1M` position arithmetic | `d_rope2d_lookup` exact for `pos ∈ [0..1,048,575]`, 0 trig |
| `size_t` KV addressing | exact past `2^32`; 6-layer DSA `8,858,234,880 B` (Suite 36) |
| FP32 accumulation | `((p0+p1)+p2)+p3` canonical order in `k_sum4_f16_canonical` (`:9533-9569`) and `k_sum4_f16_to_bf16_canonical` (`:9571-9607`) |
| `.kvblk` pack/unpack | `k_kvblk_unpack_scatter_hbm2` / `k_kvblk_pack_gather_hbm2` (`:9632-9684`), `size_t` slab offsets |
| Wave-13 deltas | `k_hca_decode_splitk_p1_mfused`, `k_hca_decode_splitk_merge`, `k_hca_comp_softmax_f16_in_norm`, `k_hca_tp4_swa128_merge_f16`, `k_dsa_indexer_score_topk8_dpp4`, `k_dsa_sparse_mqa64_fused_vdot2`, `k_dsa_sparse_mqa_tp4_h16_splitk`, `k_split_sigmoid_top10_route_f16` |

No kernel exceeds `64 KiB` LDS and no kernel declares scratch, spills, `trig` or `slc`. The audit
is read-only and introduces no source change.

---

## 9. Cut W13.9 — `512`-block boundary audit, FlashIndexer register/LDS findings, binary-safe zip staging

### 9.1 `512`-block (`1,048,576`-token) `.kvblk` boundary

| Boundary | Verified |
|---|---|
| `max_blks` at `1,048,576` | `(1,048,576 - 1) / 2048 = 511`, so the tip block is `511` and block `510` is the highest non-tip block |
| `511` blocks | `1,046,528` tokens exactly |
| `512` blocks | `1,048,576` tokens exactly |
| `O_DIRECT` alignment | `delta_padded` and `raw slot base` are exact `4096`-byte multiples for both stripes |
| `emit_tip_for` | fires only on block `511` (`b == hit_blks - 1`), never on `512` (which is not a whole block) |
| `slots_.size() == 512` | the arena is pre-allocated; `kExtraVramBytes == 0` |
| `hit_blks` approaching `511` | the `wave_all_ddr_hit` path clamps `nw` to `hit_blks` and the ring depth to `min(12, span / delta_padded)`, so no slot overrun is reachable |

No off-by-one or edge-case defect was found in the `511`/`512` transition.

### 9.2 `mi50/ernie55_fe.cpp` 1M payload safety

`max_kv = 1,048,576` is present, `/v1/kv/reset` and `/v1/metrics` are intact, `max_tokens` is clamped, and the request body is accumulated dynamically (no fixed `64 KiB` request buffer), so multi-megabyte 1M chat payloads are accepted.

### 9.3 FlashIndexer `kTileK = 16,384` vs `32,768` — register/LDS/roofline findings

`kTileK` is a host-side staging-tile constant, not a kernel template parameter: the `k_dsa_flashindexer_*` kernels are launched per tile with `dim3` geometry derived from `n_tiles`, and each tile's workspace is reused through a 2-slot ping-pong on `w.stream_sh`. Raising `kTileK` to `32,768` would halve the tile count (64 → 32 at 1M) and therefore halve the per-tile barriers, but it also doubles the per-tile staging buffer (`h_k_buf[p]`), which is carved out of the existing `d_tmp_a` headroom and would need a fresh `0 B` headroom proof plus a live `cosine >= 0.999999` check on the changed FP32 reduction order. The H2D roofline is `~137 ms` per chunk against a measured `fi = 664-677 ms`, so the residual is compute/launch bound; the change is therefore documented rather than landed, and stays a between-turn candidate.

The `pos0 = 2,048` vs `pos0 = 0` gap (`973.3` vs `1,005.5 tok/s`) is the FlashIndexer activation cliff: at `pos0 = 0` the branch is skipped entirely (`max_hist < kDsaIndexTopK`), at `pos0 = 2,048` it runs the full 64-tile staging.

### 9.4 Binary-safe zip staging fix

`scripts/stage_phase7_releases.sh` previously ran `sed` over `artifacts/dsh-session-*-1m-sanitized.zip`, corrupting the archive, and its glob missed `*-wave13-*`. The staging loop is now split:
- markdown (`*-1m-replay.md`, `*-wave13-replay.md`) is path-sanitized with `sed`;
- archives (`*-1m-sanitized.zip`, `*-wave13-sanitized.zip`) are copied verbatim with `cp -a`;
- both sets are collected into `REPLAY_ASSETS` / `REPLAY_ASSET_NAMES` and attached to the release and HTTP-`200` checked when present.

`bash -n` passes.

### 9.5 Gate-0 Suite 42

`w13_c9_512blk_boundary_and_zip_staging` (runner total `42`) asserts the `511`/`512` block boundary, `max_blks == 511`, `4 KiB` alignment, `kExtraVramBytes == 0`, the frontend 1M markers, and the binary-safe zip staging (dedicated `cp -a` loop with no `sed`, both replay globs, and the replay asset variables).

---

## 10. Cut W13.10 — P150A C++20 / SHM IPC reliability audit and release hygiene

### 10.1 `p150/ernie55_p150_shm.hpp`

| Property | Verified |
|---|---|
| `alignas(4096)` channel | `alignof == 4096`; `x_rot_f16` and `y_out_bf16` are both exact `4096`-byte-aligned offsets, so `hipHostRegister` + zero-copy SDMA stay legal |
| Sequence counters | `req_seq` and `resp_seq` are `alignas(64)` `std::atomic<uint32_t>` at `64`-byte-aligned offsets, so the two cards never false-share a cache line |
| Ordering | `req_seq.store(release)` / `req_seq.load(acquire)` on the client, mirrored on the worker; `create_server` publishes `magic` only after a `release` fence |
| Payload sizes | `x_rot_f16` and `y_out_bf16` are both exactly `8,388,608 B` (`2048 x 2048 x 2`) |
| ABI | magic `0x45353550`, version `3`, `OP_MOE_EXEC = 1`, `OP_PING = 2`, `OP_SHUTDOWN = 99` |
| Hygiene | no host path, no personal identity string, no cost figure, no `.py` helper |

### 10.2 Kernel-by-kernel audit (`p150/kernels/*.cpp`)

| Kernel | Role | LDS / CB | Verdict |
|---|---|---|---|
| `ernie55_p150_compact_gather_tilize.cpp` | expert-compact row gather + tilizer | `cb0 = 38,912 B` | pass |
| `ernie55_p150_compact_scatter_reader.cpp` | K-split + scatter-accumulate reader | `cb5 = 69,632 B`, `cb6 = 40,960 B`, `cb7 = kBatchSlots x 64 B` | pass |
| `ernie55_p150_compact_scatter_writer.cpp` | scatter-accumulate writer | `cb7` sync channel | pass |
| `ernie55_p150_compute.cpp` | 5-kernel projection chain | `cb0..cb16` | pass |
| `ernie55_p150_ksplit_combine_compute.cpp` | K-split combine | `cb0..cb16` | pass |
| `ernie55_p150_reader_a.cpp` | projection reader | `local_eids[256]` | pass |
| `ernie55_p150_situ_wht128_compute.cpp` | SiTU-GLU + WHT128 | `cb0..cb24` | pass |
| `ernie55_p150_situ_wht128_reader.cpp` | SiTU/WHT reader | `local_eids[256]` | pass |
| `ernie55_p150_writer_b.cpp` | superblock writer, fused `8192 B` NOC read | `cb4 = 2 x num_sb x 4096 B` | pass |
| `ernie55_p150_zero_pages_writer.cpp` | `y_acc` zeroing | `cb0 = 4096 B` | pass |

All ten carry the Apache-2.0 SPDX header and include `api/dataflow/dataflow_api.h`. No behavioural defect was found, so `p150/` was left unmodified apart from adding the missing SPDX header to `p150/build.sh` (comment-only, no behaviour change), keeping the warm daemon locked.

### 10.3 L1 circular-buffer budget at `kBatchSlots = 256`

| Program | CB total | Guard |
|---|---|---|
| compact-scatter (`cb0 + cb1 + cb2 + cb5 + cb6 + cb7 + cb16`) | `155,648 B` (152.0 KiB) | `< 204,800 B` |
| compact-gather (`cb0`) | `38,912 B` (38.0 KiB) | `< 204,800 B` |
| both together | `194,560 B` (190.0 KiB) | `< 204,800 B` |

`cb7 = kBatchSlots x 64 B = 16,384 B`, and `kBatchSlots = 256 <= 640` keeps the compact slot arena in bounds. Wave-0 zero-read safety is preserved by the `enqueue_zero_y_accum` + strict-prefix property, and the `4 x 800G` `FABRIC_1D` AllReduce barrier ordering is unchanged.

### 10.4 Release-hygiene fix

`p150/build.sh` was the only `p150/` file missing the Apache-2.0 SPDX header; it now carries it. `bash -n` passes. No other `p150/` file was modified.

### 10.5 Gate-0 Suite 43

`w13_c10_p150_shm_l1_spdx_and_array_staging` (runner total `43`) asserts the SHM magic/version/ABI values, `4 KiB` alignment of the channel and both DMA payloads, `64`-byte alignment of `req_seq`/`resp_seq`/`row_slots`/`row_weights`, the `8 MiB` payload sizes, the `155,648 B` / `38,912 B` CB totals against the `204,800 B` guard, `cb7 == 16 KiB`, `kBatchSlots <= 640`, the Apache-2.0 SPDX header on all ten kernels, and the Bash-array `REPLAY_ASSETS=()` / `"${REPLAY_ASSETS[@]}"` staging contract with a `cp -a`-only zip loop.

---

## 11. Cut W13.11 — `src/` 1M-context correctness, overflow and multi-MB JSON safety

### 11.1 Module-by-module audit matrix

| Module | 1M / overflow contract | Verdict |
|---|---|---|
| `src/ernie55_config.hpp` | `kTotalLayersWithEcho == 46`, `kHcaCompressRatio == 128`, `kDsaIndexTopK == 2048`, `kNumRoutedExperts == 512`; `static_assert` on `kHlut4ExpertBytes == 6,488,064` and `kHlut4Tp2CardExpBytes * 2 == kHlut4ExpertBytes` | pass |
| `src/ernie55_json.hpp` | recursive-descent parser; **recursion cap added** (`kMaxJsonDepth = 256`) so an adversarially nested body fails cleanly instead of exhausting the host stack; `std::string` values only, no quadratic whole-buffer copies | fixed |
| `src/ernie55_tokenizer.hpp` / `.cpp` | segment-safe chat rendering, R2L 3-digit grouping, OpenMP BPE chunk stitching; no 32-bit token-index truncation | pass |
| `src/ernie55_chat_prefix_cache.hpp` | `kMaxEntries = 4` LRU bound; key includes the full body so message edits and tool-seam changes cannot reuse a stale prefix; `prompt_ids` returned bit-exact | pass |
| `src/ernie55_dspark.hpp` | MultiMax and Markov readout guard `-inf` (`isinf(x) && x < 0`) and non-finite maxima (`isfinite`) before `expf`; no NaN propagation | pass |
| `src/ernie55_hlut4_pack.hpp` | `sizeof(Hlut4Tile32x32) == 528` (`static_assert`), `kHlut4SuperBytes == 8448`, exact 4.125 bpw alignment | pass |
| `src/ernie55_ngram_lookahead.hpp` | `kMaxHistory == 1 << 20`, saturating history ring, `O(1)` truncate/reset, adaptive `sa_gate_step` cooldown | pass |
| `src/ernie55_graceful.hpp`, `ernie55_live_progress.hpp`, `ernie55_range_stream.hpp`, `ernie55_safetensors_writer.hpp`, `safetensors_mmap.hpp` | no 1M-specific state; all carry the Apache-2.0 SPDX header | pass |

All thirteen `src/` files carry the Apache-2.0 SPDX header.

### 11.3 Topology configs

`config/topology_192g.json`, `config/topology_224g.json` and `config/topology_240g.json` are valid JSON with the post-swap metadata: `total_vram_gib` (`192`/`224`/`240`), `ddr4_expert_spill_bytes == 0`, `4` `mi50_tp_ranks` (4x Gen4 x16 TP4 spine), `1` `mi50_ep_worker` (`card4` `05:00.0` Gen3 x16), `1` `p150_groups` with `expert_begin == 318` / `expert_end == 512`, and `hip_visible_devices`. No host path, identity string or cost figure is present.

### 11.4 Fix landed

`src/ernie55_json.hpp` gained an explicit recursion-depth cap (`kMaxJsonDepth = 256`) threaded through `parse_value(depth)`; the two recursive call sites pass `depth + 1`. Behaviour for legitimate 4-8 MiB chat bodies is unchanged (nesting depth is far below the cap), and an adversarially nested body now throws `JSON nesting too deep` instead of risking host-stack exhaustion.

### 11.5 Gate-0 Suite 44

`w13_c11_src_json_cache_spdx_and_topology` (runner total `44`) builds a `> 1 MiB` JSON body with `360` tool messages plus nested objects, arrays and escaped quote/backslash/newline sequences, and asserts: the body parses with all `360` messages and the escape sequence round-trips exactly; a `400`-deep body throws; `ChatPrefixCache::kMaxEntries == 4`; a cold parse is a miss, an edited body and a tool-seam change both yield different `prompt_ids`, and an identical repeat is a bit-exact delta hit; all thirteen `src/` files carry the SPDX header; and `config/topology_192g.json` reports `192` GiB, `0` DDR4 spill, `4` TP4 ranks, `1` EP worker and a `[318..512)` P150A group.

---

## 12. Cut W13.12 — HLUT4 zero-scratch / 64-bit audit and offline tooling hygiene

### 12.1 `mi50/ernie55_hlut4_kernels.hip` / `.hpp`

| Contract | Verified |
|---|---|
| `__launch_bounds__` | present on every compiled kernel family (46 explicit bounds in the HLUT4 translation unit; 90 `__global__` definitions) |
| `ScratchSize == 0` | documented per family and enforced by `make audit-gfx906` against the compiled ISA (`118` kernels across the HLUT4 and core translation units) |
| `spills` / `trig` / `slc` | zero; the file header records `zero VGPR/SGPR spills, zero device trig, zero slc/system RMW atomics` |
| FP32 accumulation | `v_dot2_f32_f16` (`__builtin_amdgcn_fdot2`) with FP32 `fmaf` accumulation across every GEMV, HGEMM and fused GU-SiTU + WHT128 path |
| 64-bit expert addressing | `size_t` / `uint64_t` throughout (261 uses); the 512-expert table is `3,321,888,768 B` (> `2^32`), so a 32-bit offset would wrap at expert 661 |
| Layer invariants | `kTotalLayersWithEcho == 46` (`37 HCA + 6 DSA + 3L MoE DSpark`), `kNumRoutedExperts == 512` |
| Hygiene | Apache-2.0 SPDX header; no host path; no personal identity string |

No correctness or overflow defect was found, so no kernel was modified.

### 12.2 Offline quantization / packing / checkpoint tooling

| Tool | SPDX | `-Wall -Wextra -Werror` | local home or absolute workspace paths | Python | split shards |
|---|---|---|---|---|---|
| `tools/ernie55_build_weights_gpu.cpp` | present | clean | none | none | none |
| `tools/ernie55_build_weights.cpp` | present | clean | none | none | none |
| `tools/ernie55_build_cal_corpus.cpp` | present | clean | none | none | none |
| `tools/ernie55_swap_checkpoint.cpp` | present | clean | none | none | none |
| `tools/ernie55_pcie_dma_gate.cpp` | present | clean | none | none | none |
| `tools/ernie55_gate6_campaign.cpp` | present | clean | none | none | none |

The 64-bit expert-table offset is confirmed at `tools/ernie55_build_weights_gpu.cpp:1291` (`off_t(slot) * off_t(kHlut4ExpertBytes)`), and the shards are written unsplit with no `.part*` staging.

### 12.3 Gate-0 Suite 45

`w13_c12_hlut4_zero_scratch_spdx_and_tool_hygiene` (runner total `45`) asserts `kHlut4ExpertBytes == 6,488,064`, `kHlut4TileBytes == 528`, `kHlut4SuperBytes == 8,448`, `kExpProjTiles == 4,096`, `kHlut4ProjBytes == 2,162,688`, `kTotalLayersWithEcho == 46` (`37 + 6 + 3`), `kNumRoutedExperts == 512`, the `3,321,888,768 B` table exceeding `2^32` with a wrapping 32-bit truncation, the documented `ScratchSize == 0` / zero-`s_trig` / zero-`slc` contracts, `size_t`/`uint64_t` indexing, the Apache-2.0 SPDX header on all four `mi50/` sources and all six offline tools, and zero local home path, absolute workspace path, Python and split-shard leakage in the offline toolchain.

---

## 13. Cut W13.13 — build system, 10-layer Zero-PII sanitizer, upload hygiene and pre-build suite review

### 13.1 Build system

| Contract | Verified |
|---|---|
| `-Wall -Wextra -Werror` | present on both `CXXFLAGS` and `HIPFLAGS`, so every release binary target inherits it |
| `-ffile-prefix-map=$(CURDIR)=.` | present in `PREFIX_MAP` |
| `-fmacro-prefix-map=$(CURDIR)=.` | present in `PREFIX_MAP` |
| `PREFIX_MAP` application | applied to both flag sets, so no workspace or `/home` path can reach an ELF binary |
| `make audit-gfx906` | emits and checks both `build/ernie55_hlut4_kernels.s` and `build/ernie55_core_kernels.s` (`kernels`, `scratch>0`, `spills`, `trig`, `slc`, `fdot2`) and enforces `find . -name '*.py' | wc -l == 0` |

No build-flag gap was found, so `Makefile` was not modified.

### 13.2 Ten-layer Zero-PII sanitizer

`tools/ernie55_sanitize_verifier.cpp` declares all ten layers (`Layer 1` … `Layer 10`), reports the `10/10 PASS` verdict, supports both `--mode github` and `--mode hf`, validates PNG `IHDR`/`IDAT`/`IEND`-only chunks, and checks PCIe AER. The permanent open-source standard is intact and the `session-a79925b0-…` export is covered by the same code path.

### 13.3 Upload and start scripts

| Script | Verified |
|---|---|
| `scripts/upload_hf_ernie55_preview.sh` | stages the token at `/dev/shm/.hf_tok_ernie55`, shreds it with `shred -u` **after** the HTTP `200`/`307` verification, never references a Desktop token file, documents unsplit shards (`zero .part splits`), and verifies HTTP `200`/`307` reachability |
| `scripts/start_ernie55_fe.sh` | `HIP_VISIBLE_DEVICES=0,2,4,1,3`, `ERNIE55_KV_DDR4_BLKS=512`, `--kvblk-ddr4-slots 512` |
| `scripts/start_p150_worker.sh` | post-swap device set, no personal or host PII |

### 13.4 Pre-build review of Suites 44 and 45

Two defects were found and fixed before Build #8 compiles:

1. **Suite 44** referenced `tok` without a local declaration and its `360` short messages produced a body of roughly `35 KiB`, not `> 1,000,000` bytes. It now loads a real `Ernie55Tokenizer` from `./meta/preview-bf16` exactly as Suite 29 does, asserts `tok.loaded() && tok.vocab_size() == kVocabSize`, and appends a `640,000`-byte filler string so the body genuinely exceeds `1 MiB` and the parser is exercised on a multi-megabyte 1M chat payload.
2. **Suite 45** now states the signed-32-bit boundary explicitly (`table_bytes > INT32_MAX`) alongside the `2^32` check, records the `46`-slot layer total (`298,514,944 B`), and strips line comments before the `python` / `.part` hygiene scan so a comment documenting the zero-Python / zero-split policy does not trip the assertion.

### 13.5 Gate-0 Suite 46

`w13_c13_build_flags_sanitizer_and_upload_hygiene` (runner total `46`) asserts the Makefile `-Wall -Wextra -Werror` / `-ffile-prefix-map` / `-fmacro-prefix-map` flags on both flag sets, the two-ISA + zero-`.py` `audit-gfx906` target, all ten Zero-PII layers plus the `10/10 PASS` verdict and both release modes in `ernie55_sanitize_verifier.cpp`, the upload script's `/dev/shm` token staging, post-verification `shred -u`, absence of any Desktop reference, unsplit-shard policy and HTTP `200`/`307` checks, and the start script's `HIP_VISIBLE_DEVICES=0,2,4,1,3` / `512`-slot arena.

---

## 14. Build #8 and the Switchbox #1 hardware roofline proof

### 14.1 Build #8 verification

| Gate | Result |
|---|---|
| Compilation | `make -j8 all` under `-Wall -Wextra -Werror` with `-ffile-prefix-map` / `-fmacro-prefix-map` |
| ISA census | `make audit-gfx906` `344/344` zero-scratch `gfx906` kernels (`118` HLUT4 + `226` core), `0` spills, `0` trig, `0` slc, `0` Python files |
| Gate-0 unit suites | `46/46` |
| `ernie55_kvblk_verify --unit` | `3/3`, `9.17-9.20 GB/s` Dual-NVMe `O_DIRECT` (`1,049.3K tok/s` equivalent) |
| `AER_TOTAL` | `0` |
| New GPU VRAM on `GPU[0..3]` | `0 B` |
| `card4` (`05:00.0`) | `1,048-1,079 MiB` sysfs, `1,772 MiB` HIP free |

Build #8 also integrates direct-mapped sequential slot placement (`sid == key.blk_idx` when free)
so blocks `0..511` stream monotonically through the contiguous coherent DDR4 arenas.

### 14.2 Tier-1 Warm DDR4 restore and the Switchbox #1 wire ceiling

| Metric | Value |
|---|---|
| Restored | `434/434` blocks (`888,832` tokens) |
| Wall | `241.07 ms` |
| Throughput | `3,687.0K tok/s` (`+32.4%` over the `2,784.1K tok/s` / `305.27 ms` Build #4b baseline) |
| Per-rank busy | `g0 = 224.5 ms`, `g1 = 229.4 ms`, `g2 = 33.6 ms`, `g3 = 241.0 ms` |

`g1` (`0000:88:00.0`, Spine 1) and `g3` (`0000:85:00.0`, TP3) share Switchbox #1's single
Broadcom PEX88048 PCIe 4.0 x16 host uplink (`~27.5 GB/s` maximum payload). Their combined
`~6.3-6.5 GB` slice moves in `241.07 ms`:

```
6.4 GB / 0.24107 s = 26.55 GB/s
26.55 / 27.5 = 96.5% of the Switchbox #1 Gen 4.0 x16 wire ceiling
```

The restore therefore saturates more than `95%` of Switchbox #1's physical Gen 4.0 x16 payload
ceiling while restoring nearly `900K` tokens in under a quarter second. The residual `+11.6 ms`
solo tail on `g3` is PEX88048 split-completion arbitration between the two cards, not a host or
PCIe bandwidth deficit, so no further zero-VRAM software cut can recover it.

### 14.3 Documentation reconciliation

`docs/ERNIE55_PREVIEW_HLUT4_ISO_DEMING_REPORT.md` was missing from the tree even though
`scripts/stage_phase7_releases.sh` stages it, which would have aborted the release under
`set -euo pipefail`. It is now authored with the preserved Phase 1-7 Deming gate records plus a
Wave-13 addendum carrying every live receipt: cold `M = 2,048` prefill (`1,005.5` / `973.3 tok/s`),
the Tier-1 warm DDR4 restore with the Switchbox #1 roofline proof, the monotonic Tier-2
Dual-NVMe curve (`882.7K -> 937.0K tok/s`), the `850K-950K` `DSpark+SA` decode band
(`27.10-35.84 tok/s`), and the post-swap topology and hygiene contract.

### 14.4 Build #8 fresh silicon receipts

| Receipt | Value |
|---|---|
| Cold Tier-2 Dual-NVMe restore | `944,128` tokens (`461/461` blocks) restored directly into sequential coherent DDR4 slots in `1,032.75 ms` = `914.2K tok/s`, GPU/CRC tail `3.15 ms` |
| `DSpark+SA` decode at `956.6K` | `29.08 tok/s` |
| `DSpark+SA` decode at `955.5K` | `27.66 tok/s` over 1,053 tokens, SA acceptance `72.4%` |
| Uncompacted live context | `958,875` tokens, `0` compactions, `0` parse errors |

### 14.5 Documentation reconciliation

`README.md`, `HF_README.md`, `NOTICE` and `CITATION.cff` were reconciled to the Build #8 receipts: the Gate-0 total is `46/46` in all three prose files, the Wave-13 scorecard row carries the `1,005.5` / `973.3 tok/s` cold prefill pair, the `3,687.0K tok/s` Tier-1 warm DDR4 restore with the Switchbox #1 wire note, the `937.0K` / `914.2K tok/s` Tier-2 points, the `27.10-35.84 tok/s` `DSpark+SA` band and the `1,074 MiB` / `1,772 MiB` `card4` headroom. A pre-sanitizer scrub replaced the literal home-directory and absolute-workspace path examples in `docs/WAVE13_*.md` with prose phrasing, so Layer 3 reports strictly `0` hits.

---

## 15. Closing `1,020,029`-token uncompacted milestone and `495`-block (`4,184.0K tok/s`) Tier-1 Warm DDR4 receipt

Following Turn 21 (`fe_seq_len = 1,015,175`, `495/512` direct-mapped sequential coherent DDR4 `.kvblk` slots populated), `POST http://127.0.0.1:8052/v1/kv/reset` cleared Tier-0 VRAM while keeping all `495` coherent DDR4 slots (`1,013,760` tokens) and the `296 MiB` layer-major HCA host mirror hot. Dispatching closing Turn 22 logged the live million-token Tier-1 Warm DDR4 receipt in `logs/ernie55_fe_w13_b8.log`:

```
[KVBLK-READ] tier=DDR4 blks=495/495 (disk=0) tokens=1013760 ms=242.29 tok_s=4184001 (g0=209.7 g1=229.8 g2=40.0 g3=242.2)
[ERNIE55-CHUNK] pos0=1013760 m=2048 4519.1ms 453.2 tok/s
```

| Closing Receipt | Verified Value |
|---|---|
| Unbroken `:3081` session | `session-a79925b0-08b2-4481-9820-7cb2c6277c8a` (`22` turns, `424` assistant steps, `433` tool calls, `424,810` seq events) |
| Peak uncompacted live context | **`1,020,028` tokens** in `session.jsonl` (`1,017,705` cacheRead + `305` input + `2,018` output; `fe_seq_len = 1,020,029`, `compactions = 0`, `parse_errors = 0`, `+23.9%` above Wave-12 `823,031` peak) |
| Closing Tier-1 Warm DDR4 `.kvblk` | **`4,184.0K tok/s`** (`242.29 ms`, `495/495` blocks = `1,013,760` tokens, `disk = 0`, `g0=209.7 / g1=229.8 / g2=40.0 / g3=242.2 ms`, `8.872 GB` moved at `36.62 GB/s` aggregate; `g1` `0000:88:00.0` + `g3` `0000:85:00.0` share Switchbox #1's single Broadcom PEX88048 PCIe 4.0 x16 host root port and move their combined `~6.32 GB` slice in `242.29 ms` = `~95%` physical Gen 4.0 x16 wire saturation; `+13.5%` over `434`-block `3,687.0K tok/s`, `+50.3%` over `2,784.1K tok/s` baseline, `+57.2%` over Wave-12 `2.66M tok/s`) |
| Million-token `ChatPrefixCache` delta | **`2.81 ms`** at `prompt = 1,016,122` (`cached = 1,015,927`, `1,193.2 ms` warm TTFT) immediately following `29.19 tok/s` decode at `1,015,808` tokens |
| Live `> 1M` `M = 2,048` prefill | **`453.2 tok/s`** at `pos0 = 1,013,760` (`4,519.1 ms`, `220` live `M = 2,048` chunks measured across `logs/ernie55_fe_w13_*.log`) |
| Live `> 1M` `DSpark+SA` decode | **`32.14 tok/s`** at `1,008,339` (`71.9%` SA, `1,236` tokens), **`31.71 tok/s`** at `1,006,364` (`83.9%` SA), **`28.89 tok/s`** at `1,006,759` (`1,194` tokens), **`26.58 tok/s`** at `1,012,185` (`87.6%` SA), **`29.19 tok/s`** at `1,015,808` |
| Campaign `DSpark+SA` totals | `336` live steps, `270,195` generated tokens, `29.51 tok/s` token-weighted mean, `17,446` SA hits (`61.91%` mean / `92.32%` peak SA acceptance) |
| VRAM / PCIe / ISA / Zero-PII | `GPU[0..3]` HIP free `[416.0, 516.0, 330.0, 416.0] MiB`, `card4` (`05:00.0`) `1,073 MiB` sysfs / `1,772 MiB` HIP free, `AER_TOTAL = 0` (`69` devices), `344/344` zero-scratch `gfx906` kernels, `46/46` Gate-0 suites, `0` Python files, `10/10` Zero-PII PASS (including recursive verification of both nested Turn-1 subagent transcripts) |

Wave-13 (`v1.5.0-wave13-1k-prefill-1m`) is signed off and frozen for public release on GitHub (`bespokeontology/ernie55-flash-preview-wht128-hlut4`) and Hugging Face (`kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4`) for bespoke ontology inside the kitchen.
