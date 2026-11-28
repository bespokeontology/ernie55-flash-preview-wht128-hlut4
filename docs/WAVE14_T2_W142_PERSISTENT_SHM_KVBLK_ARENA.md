<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-14 Turn 2 — Build W14-1 Silicon Receipt and Cut W14.2 (Persistent `/dev/shm` `1,024`-Slot `2M`-Token Tier-1 DDR4 `.kvblk` Arena + Gate-0 Suite 48)

Workstation `SB-M5-P2-192G` (`512 GiB` / `8ch` `3200 MT/s`), Build W14-1 (`Cut W14.1`) live on `:8052`.

No `make`, no GPU compiler invocation and no daemon restart was performed during this turn.

## 1. Build W14-1 receipt (given)

| Item | Value |
|---|---|
| Compile | clean first pass under `-Wall -Wextra -Werror` |
| `make audit-gfx906` | `344/344` zero-scratch `gfx906` kernels, `0` spills, `0` trig, `0` `slc`, `0` `.py` |
| Gate-0 | `47/47` suites |
| `[SUITE47]` | `tile_k=16384..32768 tmp_a=134217728 B need512=83886080 B tiles1m=32 ident=3 merge=15 set_mism=0 min_cos=1.0000000 vram=0 B` |
| Standalone silicon gate | `38,263` tokens, `AER_TOTAL = 0` |
| `pos0 = 0` / `pos0 = 2,048` | `1,963.1 ms` (`1,043.3 tok/s`) / `2,038.9 ms` (`1,004.5 tok/s`) |
| Decode at `38,263` tokens | `47.99 tok/s`, `ttft = 92.3 ms`, `100.0%` SA, `A=1 B=1 CPP=1` |
| Thermal (Turn-3 input) | `g0` `0000:c7:00.0` `87C` junction / `82C` HBM2 vs `44–57C` on the four switchbox cards |

## 2. Exact persistent layout (`1,024` slots)

`mi50/ernie55_kvblk_cache.hpp`, `Ernie55BlockKvCache::shm_geometry()`. Every region base is
4 KiB aligned; both slot strides are 4 KiB multiples, so each slot stays one contiguous
`O_DIRECT`-safe DMA span of `[4096 hdr + delta_padded + tip_padded]`.

| Region | Offset | Size |
|---|---|---|
| `Ernie55KvShmHdr` (sealed, CRC32C) | `0` | `4,096 B` |
| `Ernie55KvShmSlotMeta[1024]` (64 B each) | `4,096` | `65,536 B` |
| stripe-0 arena (`1024 x 23,908,352 B`) | `69,632` | `24,482,152,448 B` |
| stripe-1 arena (`1024 x 23,891,968 B`) | `24,482,222,080` | `24,465,375,232 B` |
| layer-major HCA host mirror (`37 x 1024 x 16 KiB`) | `48,947,597,312` | `620,756,992 B` |
| **total** | | **`49,568,354,304 B` = `46.16 GiB`** |

Per-stripe geometry (43-layer Preview, `kCompPerBlk = 16`, `kBlockTokens = 2048`):

* `delta_bytes_s0 = 19 x 16,384 + 3 x 2048 x 704 x 2 = 8,962,048 B`
* `delta_bytes_s1 = 18 x 16,384 + 3 x 2048 x 704 x 2 = 8,945,664 B`
* `tip_bytes_s0 = tip_bytes_s1 = 19 x 786,432 = 14,942,208 B`
  (`18 x 786,432 + 3 x 262,144` on stripe 1 — the DSpark rings make the two stripes equal)
* `slot_stride_s0 = 4096 + 8,962,048 + 14,942,208 = 23,908,352 B`
* `slot_stride_s1 = 4096 + 8,945,664 + 14,942,208 = 23,891,968 B`

Capacity: `1,024 x 2048 = 2,097,152` tokens (`2x` the `1,048,576` `max_kv` ceiling, i.e. genuine
multi-session prefix reuse). `46.16 GiB` of `252 GiB` `/dev/shm` leaves `> 200 GiB` spare, and
`325 GiB` of DDR4 is free even with all `159 GiB` of weights warm in page cache.

### Why full per-slot tips (and not the `~18.5 GiB` delta-only pool)

A `1,024`-block restore needs **every** block's HCA `swa_ring` + `state_kv`/`state_sc` resident
simultaneously, so a shared tip ring cannot serve a `2M`-token prefix. The `~18.5 GiB`
delta-only variant is only smaller because it drops the tips, which would force tip
recomputation on re-attach and break the bit-exact restore contract. The full layout is the
smallest one that satisfies exactness, 4 KiB alignment, mirror persistence and
`0 B` net new GPU `hipMalloc`, so it is the one implemented.

## 3. Startup pinning impact

* **One** `hipHostRegister(hipHostRegisterPortable)` for the whole `46.16 GiB` segment — one KFD
  mapping set and one pin pass instead of `2,048` per-slot registrations. Every per-slot
  `raw_s0`/`raw_s1` is DMA-capable with zero per-slot registration cost.
* **First run** pays a 16-thread 4 KiB first-touch fault loop (`kWbFaultThreads`) plus the pin.
* **Re-attach skips the fault loop entirely** — the pages are already resident from the previous
  process — so a `:8052` hot-swap pays only `mmap` + `hipHostRegister` + the parallel CRC32C
  sweep over the committed slots (OpenMP over slots, `~21 GB/s` per core, so `~0.2-0.3 s` for a
  full `1,024`-slot arena).
* If `hipHostRegister` fails (host that cannot pin `46 GiB`), the segment is left in place
  (never unlinked) and the cache falls back to the anonymous `hipHostMallocCoherent` arena, i.e.
  exactly Wave-13 behaviour. No `hipMalloc` is ever issued for the arena, so `kExtraVramBytes == 0`
  and `card4` headroom are untouched.

## 4. Recovery contract

* `init()` reads `ERNIE55_KV_DDR4_SHM` (e.g. `/ernie55_kvblk_ddr4`). Unset, or any of
  `shm_open` / `ftruncate` / `mmap` / `hipHostRegister` failing → anonymous path.
* `shm_hdr_ok()` requires magic `W14KVBLK`, version `1`, `header_bytes == 4096`, exact geometry
  (slot count, both deltas, both tips, mirror span, all four region offsets, `total_bytes`) and a
  valid `header_crc32c`. Any mismatch ⇒ **trust nothing**, wipe and re-seal.
* Valid segment ⇒ **never wiped**. `shm_recover_index()` rebuilds `slots_` + `index_` from every
  record whose `meta_crc32c` and all four payload CRC32Cs pass; failures are cleared in place
  (fail-closed) and counted. Logs
  `[KVBLK-SHM] Re-attached … recovered=N/1024 slots (R rejected) in X.XX ms | … B registered, disk=0`.
* The per-slot record is sealed by the **existing off-critical-path NVMe writer thread**, reusing
  the very CRC32Cs the disk header already carries, guarded by the slot generation so an
  evicted-and-reused slot can never be sealed under an old key.
* `shutdown()` calls `hipHostUnregister` + `munmap` and **never** `shm_unlink`, so warm blocks
  survive hot-swaps. `Ernie55BlockKvCache::shm_unlink(name)` exists for tests/ops only.

## 5. Gate-0 Suite 48

`tools/ernie55_verify_unit.cpp` — `w14_c2_persistent_shm_kvblk_arena()`, registered in `main()`
(banner now `ALL 48 …`). Host-only, no GPU required.

1. Exact layout math: `total_bytes == 49,568,354,304`, both strides, all four 4 KiB-aligned region
   bases, `2,097,152` token capacity, `kExtraVramBytes == 0`, `46.16 GiB < 252 GiB` tmpfs.
2. Real `shm_open(O_CREAT|O_EXCL)` → `ftruncate` → `mmap` → commit 24 slots spread across both
   stripes → `munmap` → re-`mmap` → `shm_recover_index()`: `recovered == 24`, `rejected == 0`,
   `index.size() == 24`, `memcmp == 0` on every stripe-0 payload, exact `key → slot id` mapping.
3. Fail-closed: a 1-bit payload flip ⇒ `recovered == 23`, `rejected == 1`, slot cleared, key
   absent; a slot-count geometry mismatch ⇒ `0` recovered before any record is inspected; a
   corrupted `header_crc32c` ⇒ `0` recovered.
4. `shm_unlink` + `::stat("/dev/shm/<seg>") != 0` proves `/dev/shm` is left spotless.

Verified against the real statics with a host-only scratch driver:
`recovered=24 rejected=0 / tamper: recovered=23 rejected=1 / geom mismatch: recovered=0 rejected=0`.

## 6. Files edited

* `mi50/ernie55_kvblk_cache.hpp` — `Ernie55KvShmHdr`, `Ernie55KvShmSlotMeta`, `ShmGeom`,
  `shm_geometry`, `shm_align4k`, `shm_fill_meta`, `shm_slot_crc_ok`, `shm_hdr_ok`,
  `shm_recover_index`, `shm_unlink`, `shm_setup`, `shm_teardown`, SHM members, `init()`/`shutdown()`
  integration, writer-thread meta sealing, `<cerrno>`.
* `scripts/start_ernie55_fe.sh` — `ERNIE55_KV_DDR4_BLKS` default `512 → 1024`,
  `ERNIE55_KV_DDR4_SHM` default `/ernie55_kvblk_ddr4`.
* `tools/ernie55_verify_unit.cpp` — Suite 48 + registration, Suite 46 updated to the new
  defaults/banner, `<cerrno>`/`<fcntl.h>`/`<sys/mman.h>`/`<sys/stat.h>`/`<unistd.h>`.
