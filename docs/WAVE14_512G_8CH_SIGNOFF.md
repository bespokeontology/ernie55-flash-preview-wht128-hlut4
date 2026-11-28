<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-14 `v1.6.0-wave14-512g-8ch` Engineering Signoff and Receipt

Release `v1.6.0-wave14-512g-8ch` · workstation `SB-M5-P2-192G` · inside `the kitchen` for bespoke
ontology · host `AMD EPYC 7542` · `5x AMD Radeon Instinct MI50` `128 GiB` HBM2 +
`2x Tenstorrent Blackhole P150A` `64 GiB` GDDR6.

This signoff covers the complete Wave-14 progression from the Wave-13 baseline
(`v1.5.0-wave13-1k-prefill-1m`, commit `a18a59b`) through the Phase 1 `512 GiB` 8-channel hardware
baseline and the three Phase 2 cuts W14.1, W14.2 and W14.3. No `make`, no GPU compiler invocation
and no daemon, PCIe sysfs or Python access was performed during this turn.

## 1. Platform baseline

| Item | Value |
|---|---|
| Host memory | `8 x 64 GiB = 512 GiB` Samsung `M393A8K40B22-CAE` RDIMM, `3200 MT/s` on all `8/8` channels |
| Memory controller | `UMC0..UMC7` populated, `D18F0x110 = 0x0000003B` 8-channel hashed interleave |
| ECC | `S4ECD4ED` `ce = 0`, `ue = 0`; `AER_TOTAL = 0` |
| Usable ECC DDR4 | `503 GiB` (`325 GiB` free with all `159 GiB` of weights warm in page cache) |
| `/dev/shm` tmpfs | `252 GiB` |
| MI50 | `5x` `128 GiB` HBM2, `gfx906` |
| P150A | `2x` `64 GiB` GDDR6, `253.7 MiB` L1 pinned, `zcopy_dma = 1/1` |
| `max_kv` | `1,048,576` tokens |
| Tier-1 DDR4 spill | `0 B` |

## 2. Phase 1 — `512 GiB` 8-channel silicon baseline

The untouched shipped Wave-13 Build #8 (`v1.5.0-wave13-1k-prefill-1m`, `a18a59b`) ran on the
upgraded host before any code was touched.

### 2.1 Host PCIe 4.0 x16 wire speed (`bin/ernie55_pcie_dma_gate`, `HIP_VISIBLE_DEVICES=0,2,4,1,3`)

| Metric | Before | After | Delta |
|---|---|---|---|
| `ALL-5` concurrent H2D | `42.77 GB/s` | **`68.07 GB/s`** | **`+59.2%`** |
| `ALL-5` concurrent D2H | `38.31 GB/s` | **`64.06 GB/s`** | **`+67.2%`** |
| `05:00.0` concurrent H2D | `8.55 GB/s` | **`13.62 GB/s`** | **`+59.3%`** |
| `REPLAY_DELTA_SUM` | `0` | `0` | unchanged |
| `AER_TOTAL` | `0` | `0` | unchanged |

All three host root ports now saturate concurrently: `c7:00.0` `27.47 GB/s`, Switchbox #1
(`88:00.0` + `85:00.0`) `13.90 + 13.90 = 27.80 GB/s`, Switchbox #2 (`08:00.0` + `05:00.0`)
`14.35 + 13.62 = 27.97 GB/s`. `02:00.0` stays locked at `8.0 GT/s x16` with `0` errors.

### 2.2 Tier-1 DDR4 restore and page-cache pre-warm

| Metric | Before | After |
|---|---|---|
| `159 GiB` checkpoint page-cache pre-warm | n/a | `34.94 s` (`4.55 GB/s`), `161 GiB` cached, `325 GiB` free |
| `495`-block (`1,013,760`-token) Tier-1 Warm DDR4 restore | `242.29 ms` / `4.184M tok/s` | **`184.16 ms` / `5.50M tok/s`** at `48.18 GB/s` |
| Per-GPU restore split | n/a | `g0 = 171.5`, `g1 = 184.1`, `g2 = 13.8`, `g3 = 24.2 ms` |

### 2.3 Untouched Build #8 reference (`logs/w14_phase1_build8_warm_uninst.log`)

`pos0 = 0` `1,955.6 ms` (`1,047.2 tok/s`), `pos0 = 2,048` `2,037.8 ms` (`1,005.0 tok/s`),
`pos0 = 4,096..14,336` `2,037.0-2,075.4 ms` (`1,005.4 -> 986.8 tok/s`), `pos0 = 16,384`
`2,114.7 ms` (`968.4 tok/s`); cold `20,395`-token prefill `20,593.0 ms` (`990.4 tok/s`);
greedy decode `45.97 tok/s` at `20,396` tokens with `A=1 B=1 CPP=1`.

## 3. Phase 2 — Cut W14.1 (`FlashIndexer` `pos0 >= 2,048` cliff, adaptive `kTileK = 32,768`)

Design note `docs/WAVE14_T1_W141_FLASHINDEXER_CLIFF_AND_TILE_K.md`; build `v1.6.0-wave14-512g-8ch`
Build W14-1.

* Diagnosis: `chunk_dsa_fi_ms` is a host wall-clock window, so the `90 ms` at `pos0 = 2,048` is the
  phase C1 dense-KV `hipStreamSynchronize` draining the backlog, the `3 ms` at `4,096..14,336` is
  the same GPU work fully overlapped with no barrier in the window, and the `106 ms` at
  `16,384` is the `ev_dsa_uniq_cnt` event sync plus the `kTileK` tile transition.
* Change: `FlashIndexer` outer tile widened from `16,384` to `32,768` through a shared host-side
  planner over the existing `128 MiB` `d_tmp_a` slab, moving the first multi-tile boundary from
  `pos0 = 16,384` to `pos0 = 32,768`.
* Gate: `47/47` suites, `344/344` zero-scratch kernels, `0 B` net new GPU `hipMalloc`.
  `[SUITE47] tile_k=16384..32768 tmp_a=134217728 B need512=83886080 B tiles1m=32 ident=3 merge=15
  set_mism=0 min_cos=1.0000000 vram=0 B`.
* Silicon: `38,263`-token standalone gate (`logs/w14_b1_36k_gate.log`), `pos0 = 0` `1,963.1 ms`
  (`1,043.3 tok/s`), `pos0 = 2,048` `2,038.9 ms` (`1,004.5 tok/s`), greedy decode `47.99 tok/s` at
  `38,263` tokens with `ttft = 92.3 ms` and `A=1 B=1 CPP=1`; `AER_TOTAL = 0`.

## 4. Phase 2 — Cut W14.2 (persistent `/dev/shm` `1,024`-slot `2M`-token Tier-1 DDR4 arena)

Design note `docs/WAVE14_T2_W142_PERSISTENT_SHM_KVBLK_ARENA.md`; Build W14-2.

| Region | Offset | Size |
|---|---|---|
| `Ernie55KvShmHdr` (CRC32C sealed) | `0` | `4,096 B` |
| `Ernie55KvShmSlotMeta[1024]` (64 B each) | `4,096` | `65,536 B` |
| stripe-0 arena (`1024 x 23,908,352 B`) | `69,632` | `24,482,152,448 B` |
| stripe-1 arena (`1024 x 23,891,968 B`) | `24,482,222,080` | `24,465,375,232 B` |
| layer-major HCA host mirror (`37 x 1024 x 16 KiB`) | `48,947,597,312` | `620,756,992 B` |
| total | | `49,568,354,304 B` = `46.16 GiB` |

* Capacity `1,024 x 2048 = 2,097,152` tokens, `2x` the `1,048,576` `max_kv` ceiling.
* One `hipHostRegister(hipHostRegisterPortable)` for the whole segment; re-attach skips the 4 KiB
  first-touch fault loop. `0 B` net new GPU `hipMalloc`; `kExtraVramBytes == 0`.
* Fail-closed recovery: magic, version, exact geometry and `header_crc32c`, then per-record
  `meta_crc32c` plus all four payload CRC32Cs. Mismatches are rejected before any record is
  inspected; tampered slots are cleared in place.
* `shutdown()` unregisters and `munmap`s without `shm_unlink`, so warm blocks survive hot-swaps.
* Gate: `48/48` suites, `3/3` `ernie55_kvblk_verify` (`9.13 GB/s`), `344/344` kernels,
  `AER_TOTAL = 0`. `[SUITE48] slots=1024 tokens=2097152 total=49568354304 B (46.16 GiB) align=4K
  recovered=24/24 tamper_rej=1 geom_rej=0 vram=0 B shm=spotless`.
* Silicon: segment created in `16,825.84 ms`, `READY in 117.6 s`, `free_mib = [410.0, 510.0, 322.0,
  408.0, 1764.0]`; cooled `pos0 = 0` `1,964.7 ms` (`1,042.4 tok/s`), `pos0 = 2,048` `2,030.3 ms`
  (`1,008.7 tok/s`), `pos0 = 16,384` `2,100.6 ms` (`975.0 tok/s`).

## 5. Phase 2 — Cut W14.3 (cooled `cost_w` minimax and `512K-1M` decode tightening)

Design note `docs/WAVE14_T3_W143_COSTW_AND_1M_DECODE.md`; Build W14-3.

* Cooled minimax `cost_w` re-derived from the measured per-layer ms per raw unit:
  `{1.15, 1.15, 0.90, 1.08, 1.12}` -> **`{1.05, 1.15, 0.90, 1.03, 1.10}`**. Chunk `mi50` proxy
  `491.0 -> 478.6 ms` (`-2.5%`), worst layer `13.29 -> 12.39 ms` (`-6.8%`), per-GPU spread
  `1.09 -> 0.19 ms`; `g2` `11.55 -> 11.05 ms`, `g1` `11.32 -> 11.02 ms`, `g3` `10.86 -> 11.08 ms`,
  `g4` `11.31 -> 11.27 ms`, `g0` `10.46 -> 11.08 ms`. Caps `{66, 68, 97, 38, 49}` and the
  `318 + 7` shadow layout unchanged.
* Occupancy-aware HCA decode split count, `M`-independent and capped at `base_splits`: with
  `h_loc = 16` (`x_blocks = 4`) the `M = 1` grid fills 60 CUs at `n_comp_splits = 16`, so the
  `512K..1M` decode uses `17` terms instead of `33` — `1.94x` lower `part_o_f32` HBM2 traffic
  (`6.19 -> 3.19 MiB` per layer per GPU at `M = 6`), bit-exact `M = 1 == M = 6`, zero regression
  below `256K`.
* Gate: `49/49` suites, `3/3` `ernie55_kvblk_verify` (`9.27 GB/s`), `344/344` kernels,
  `AER_TOTAL = 0`. `[SUITE49] cost_w=1.05/1.15/0.90/1.03/1.10 mi50proxy 491.0->478.6 ms
  worstlayer 13.29->12.39 ms spread=0.191 ms hw=[11.08 11.02 11.05 11.08 11.27] splits 33->17
  part_o 6.19->3.19 MiB/L/GPU min_cos=1.0000000 vram=0 B`.
* Silicon (`logs/w14_b3_cooled_uninst.log`, `logs/w14_b3_cooled_thermals.log`): cold `20,935`-token
  prefill `21,438.2 ms` (`976.5 tok/s`), `-78.2 ms` versus cooled Build W14-2 (`21,516.4 ms` /
  `973.0 tok/s`); `pos0 = 4,096` `1,008.4 tok/s`, `6,144` `1,006.4 tok/s`, `8,192` `999.6 tok/s`,
  `10,240` `998.9 tok/s`; `mi50` `445 -> 435 ms` at `pos0 = 16,384`; `g2..g4` spread `6 ms` at
  `pos0 = 6,144..8,192`; `g1` junction `97C -> 94C` with `1725 MHz` residency `50% -> 78%`;
  greedy decode `42.29 tok/s` at `20,936` tokens with `ttft = 74.8 ms` and `A=1 B=1 CPP=1`.

## 6. Live `:8052` hot-swap and Tier-1 restore proof

`logs/ernie55_fe_w14_b3.log`, recorded on Build W14-3 `:8052` onto the surviving Build W14-2
segment:

```
[KVBLK-SHM] Re-attached /ernie55_kvblk_ddr4 recovered=175/1024 slots (0 rejected) in 16185.80 ms | 49568354304 B registered, disk=0
[ERNIE55-RT] 3-Tier 2048-Block .kvblk cache ready (DDR4_prealloc=1024 slots, hbm2_slab=1, ...)
[KVBLK-READ] tier=DDR4 blks=174/174 (disk=0) tokens=356352 ms=67.74 tok_s=5260614 (g0=61.9 g1=62.3 g2=6.4 g3=67.7)
[ERNIE55-KVBLK] RESTORE tier=DDR4 reused=356352 (+356352 tok / 174 blk) 67.74ms 5260.6K tok/s
```

* `READY in 112.2 s`, `kvblk_ddr4_slots = 1024`, `kvblk_ddr4_used = 175` at boot with `0` requests
  served, `free_mib = [410.0, 510.0, 322.0, 408.0, 1764.0]`, `AER_TOTAL = 0`, `UE = 0`.
* Live Tier-1 DDR4 restore at `5,260.6K tok/s` with `disk = 0`, matching the Phase 1 `5.50M tok/s`
  hardware ceiling.
* The request served by this session executed on Build W14-3 `:8052` at `prompt = 363,123` with
  `cached = 356,352` and `dec = 32.63 tok/s` (`spec = 186/356`), exercising the live hot-swap
  restore and the Cut W14.3 `33 -> 17` split-K decode above `361K` context.

## 7. Gate and audit matrix

| Gate | Wave-13 baseline | W14-1 | W14-2 | W14-3 |
|---|---|---|---|---|
| Gate-0 unit suites | `46/46` | `47/47` | `48/48` | **`49/49`** |
| `make audit-gfx906` zero-scratch kernels | `344/344` | `344/344` | `344/344` | **`344/344`** |
| Spills / hw trig / `slc` | `0 / 0 / 0` | `0 / 0 / 0` | `0 / 0 / 0` | **`0 / 0 / 0`** |
| `ernie55_kvblk_verify` | `3/3` | `3/3` | `3/3` (`9.13 GB/s`) | **`3/3` (`9.27 GB/s`)** |
| Net new GPU `hipMalloc` | `0 B` | `0 B` | `0 B` | **`0 B`** |
| Python files | `0` | `0` | `0` | **`0`** |
| `AER_TOTAL` / `UE` | `0 / 0` | `0 / 0` | `0 / 0` | **`0 / 0`** |
| Greedy coherence | `A=1 B=1 CPP=1` | `A=1 B=1 CPP=1` | `A=1 B=1 CPP=1` | **`A=1 B=1 CPP=1`** |

## 8. Invariants held across all three cuts

* `ScratchSize == 0`, `0` spills, `0` hardware trig, `0` `slc`, `0` BAR1 P2P.
* `0 B` net new GPU `hipMalloc` on `GPU[0..3]`; `card4` (`0000:05:00.0`) headroom preserved
  (`free_mib` `1764.0` HIP).
* `kExtraVramBytes == 0`; `max_chunk`, `kStrideSel`, `kDsaIndexTopK` and `kSlidingWindow` unchanged.
* Fixed slot caps `{66, 68, 97, 38, 49}` and the `318 + 7` P150A shadow layout unchanged.
* Bit-exact greedy coherence `A=1 B=1 CPP=1` on every silicon gate.
* `0 B` DDR4 spill; `AER_TOTAL = 0` and `UE = 0` on every run.

## 9. Delivered artifacts

* `docs/WAVE14_T1_W141_FLASHINDEXER_CLIFF_AND_TILE_K.md`
* `docs/WAVE14_T2_W142_PERSISTENT_SHM_KVBLK_ARENA.md`
* `docs/WAVE14_T3_W143_COSTW_AND_1M_DECODE.md`
* `docs/WAVE14_512G_8CH_SIGNOFF.md`
* `artifacts/gate_wave14_512g_8ch_receipt.json`

## 10. Signoff

Wave-14 `v1.6.0-wave14-512g-8ch` is signed off. The `512 GiB` 8-channel `3200 MT/s` upgrade lifts
concurrent host DMA by `+59.2%` H2D and `+67.2%` D2H and the `1,013,760`-token Tier-1 DDR4 restore
by `+31.5%` to `5.50M tok/s`; Cut W14.1 removes the `FlashIndexer` `pos0 >= 2,048` cliff; Cut W14.2
makes the `1,024`-slot `2,097,152`-token Tier-1 arena survive `:8052` hot-swaps with a verified
`175/1024` re-attach and a live `disk = 0` restore at `5,260.6K tok/s`; Cut W14.3 rebalances the
five cards onto the cooled minimax and halves the long-context decode split traffic. All three cuts
hold `49/49` Gate-0 suites, `344/344` zero-scratch `gfx906` kernels, `0 B` net new GPU
`hipMalloc`, `0` spills, `0` trig, `0` `slc`, `0` BAR1 P2P, `0` Python files, `AER_TOTAL = 0`,
`UE = 0` and `A=1 B=1 CPP=1`.
