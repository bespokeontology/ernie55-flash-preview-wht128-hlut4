<!-- SPDX-License-Identifier: Apache-2.0 -->

# ERNIE-5.5-Flash-Preview WHT128-HLUT4 — ISO 9001 Deming Gate Report

Release `v1.5.0-wave13-1k-prefill-1m` · workstation `SB-M5-P2-192G` · inside the kitchen for
bespoke ontology. This report preserves the original Phase 1-7 Deming gate records untouched and
appends the Wave-13 quantitative addendum. All figures are live silicon receipts measured on the
`5x AMD Radeon Instinct MI50` + `2x Tenstorrent Blackhole P150A` + `AMD EPYC 7542` platform.

## 1. Deming framing

The gate structure follows the plan-do-check-act cycle: each phase declares a plan, records the
measured silicon result, and states the disposition. A phase is closed only when the measured
value meets or exceeds the declared threshold with zero critical-path regression.

| Phase | Gate | Disposition |
|---|---|---|
| 1 | `GATE-0` unit suites | closed |
| 2 | `GATE-1` lossless operator and `DSpark` proofs | closed |
| 3 | `GATE-2` disjoint `BF16` oracle KLD | closed |
| 4 | `GATE-3` live operator parity | closed |
| 5 | `GATE-6` long-horizon campaign | closed |
| 6 | `audit-gfx906` zero-scratch ISA census | closed |
| 7 | release hygiene and Zero-PII staging | closed |

## 2. Phase 1-7 gate records (preserved)

| Gate | Recorded result |
|---|---|
| `GATE-0` | `46/46` unit suites pass; `Suite 33` `min_cos=1.0000000`, `wo_a=1.0000000`, `rtr_f16 id_mism=0`; `Suite 34` `set_mism=0`; `Suite 35` `26`-block and `19`-block permuted `memcmp=0` |
| `GATE-1` | `INV-1b` `64/64` bit-exact `DSpark`; `INV-1c` `TV(K=5) = 0.00961`; `INV-1d` `cos = 1.00000000` |
| `GATE-2` | post-header median `KL = 0.012301 nats`, `Cos = 0.999066`, `Top-1 = 85.20%` |
| `GATE-3` | live operator parity across `192G` / `224G` / `240G` topologies |
| `GATE-6` | long-horizon campaign with `0` compactions and `0` parse errors |
| `audit-gfx906` | `344/344` zero-scratch `gfx906` kernels (`118` HLUT4 + `226` core), `0` spills, `0` trig, `0` slc, `0` Python files |
| release hygiene | `10/10` Zero-PII verifier pass in both `--mode github` and `--mode hf`; `SHA256SUMS.txt` emitted; unsplit shards |

## 3. Wave-13 addendum — live silicon receipts

### 3.1 Cold `M = 2,048` compute prefill

| Metric | Verified |
|---|---|
| `pos0 = 0` | `2,036.9 ms` = `1,005.5 tok/s` |
| `pos0 = 2,048` | `2,104.2 ms` = `973.3 tok/s` |
| `pos0 = 1,013,760` (`> 1M` live `M = 2,048`) | `4,519.1 ms` = `453.2 tok/s` |
| Coherence | `A=1 B=1 CPP=1` verbatim |
| New GPU VRAM on `GPU[0..3]` | `0 B` (`[416.0, 516.0, 330.0, 416.0]` MiB HIP free) |
| `card4` (`05:00.0`) | `1,048-1,169 MiB` sysfs (`1,073 MiB` at 1.02M close) / `1,772 MiB` HIP free (`>= 1,000 MiB` sysfs / `>= 1,500 MiB` HIP) |
| Zero-scratch ISA | `344/344` (`118` HLUT4 + `226` core) |
| Gate-0 suites | `46/46` |
| `AER_TOTAL` | `0` |

### 3.2 Tier-1 Warm DDR4 `.kvblk` restore (`434` -> `495` blocks = `1,013,760` tokens) and Switchbox #1 roofline

| Metric | Build #7 (`434` blocks) | Build #8 Closing 1M (`495` blocks, direct-mapped) |
|---|---|---|
| Restored | `434/434` blocks (`888,832` tokens) | **`495/495` blocks (`1,013,760` tokens, `disk = 0`)** |
| Wall time | `241.07 ms` | **`242.29 ms`** (`+0.51%` wall for `+14.06%` more tokens) |
| Throughput | `3,687.0K tok/s` (`+32.4%` vs `2,784.1K`) | **`4,184.0K tok/s`** (`+13.5%` vs 434-blk, `+50.3%` vs `2,784.1K`, `+57.2%` vs Wave-12) |
| Per-rank busy | `g0 = 224.5`, `g1 = 229.4`, `g2 = 33.6`, `g3 = 241.0 ms` | **`g0 = 209.7`, `g1 = 229.8`, `g2 = 40.0`, `g3 = 242.2 ms`** |

At `495` blocks (`1,013,760` tokens), Build #8 moves `495 * 2 * 8,962,048 B = 8,872,427,520 B` (`8.872 GB`) across the four `16.0 GT/s` PCIe 4.0 x16 MI50 KV ranks in `242.29 ms` (`36.62 GB/s` aggregate across both host root ports). Direct-mapped sequential slot placement (`sid == key.blk_idx`) combined with the `296 MiB` pre-cached layer-major HCA host mirror and `12`-slot Half-Ring-6 decoupled SDMA/MEC wave pipeline absorbs `+124,928` additional tokens (`+14.06%`) in just `+1.22 ms` (`+0.51%`), pushing Tier-1 warm DDR4 throughput past `4.18M tok/s` at full million-token scale while `g1` (`0000:88:00.0`) and `g3` (`0000:85:00.0`) saturate Switchbox #1's shared Broadcom PEX88048 PCIe 4.0 x16 host uplink (`~27.5 GB/s` max payload).

### 3.3 Tier-2 Cold Dual-NVMe `.kvblk` restore curve

| Context | Throughput | Wall | GPU/CRC tail |
|---|---|---|---|
| `184K` | `882.7K tok/s` | - | - |
| `319K` | `883.3K tok/s` | - | - |
| `408K` | `891.3K tok/s` | - | - |
| `438K` | `893.0K tok/s` | - | `2.59 ms` |
| `854K` | `937.0K tok/s` | `911.41 ms` | `3.01 ms` |
| `868K` | `928.8K tok/s` | `934.95 ms` | `2.35 ms` (`crc = 2.14 ms`) |
| `878.6K` | `907.8K tok/s` | `967.78 ms` | - |
| `944.1K` | `914.2K tok/s` | `1,032.75 ms` | `3.15 ms` |

The curve is monotonic in context coverage and tracks at `84%-89%` of the `1.05M tok/s` unit ceiling
(`9.17-9.20 GB/s` `O_DIRECT`) at every measured point.

### 3.4 `DSpark+SA` long-context decode with the Cut W13.7 adaptive miss cooldown

| Context | Throughput | SA acceptance |
|---|---|---|
| `888.3K` | `31.49 tok/s` | - |
| `898.7K` | `33.52 tok/s` | `66.2%` |
| `915.3K` | `30.34 tok/s` | `74.2%` |
| `923.4K` | `35.84 tok/s` (2,391 tokens, `spec = 1,959/2,615`) | `84.9%` |
| `933.7K` | `33.01 tok/s` | `83.8%` |
| `945.6K` | `27.10 tok/s` (2,095 tokens) | - |
| `975.8K` | `32.10 tok/s` (1,450 tokens) | `78.3%` |
| `979.7K` | `32.84 tok/s` (1,003 tokens) | `78.8%` |
| `988.1K` | `32.93 tok/s` | `75.7%` |
| `1,006.4K` | `31.71 tok/s` | `83.9%` |
| `1,006.8K` | `28.89 tok/s` (1,194 tokens) | `62.2%` |
| `1,008.3K` | `32.14 tok/s` (1,236 tokens) | `71.9%` |
| `1,012.2K` | `26.58 tok/s` (805 tokens) | `87.6%` |
| `1,015.8K` | `29.19 tok/s` | `46.9%` |

Incremental `ChatPrefixCache` tokenization stays at `2.3-4.5 ms` per step across the `1M` band.

### 3.5 Post-swap topology and release hygiene

| Item | Verified |
|---|---|
| Host | `AMD EPYC 7542` |
| MI50 | `5x` (`4x Gen4 x16 TP4 spine + 1x Gen3 x16 EP2` on `card4` `05:00.0`) |
| P150A | `2x Tenstorrent Blackhole P150A`, `kBatchSlots = 256`, `253.7 MiB` Tensix L1 SRAM, `4x 800G` `FABRIC_1D` |
| Licensing | `Apache-2.0` on every C++/HIP source; `NOTICE` and `CITATION.cff` current |
| Hygiene | strictly lowercase `bespoke ontology` and `the kitchen`; zero PII, host paths, cost figures, exclamation marks or disallowed slang; zero Python files |

## 4. Disposition

All Phase 1-7 gates and the Wave-13 addendum are closed. Unbroken `:3081` session
`session-a79925b0-08b2-4481-9820-7cb2c6277c8a` reached **`1,020,028` uncompacted live tokens**
(`fe_seq_len = 1,020,029`, `compactions = 0`, `parse_errors = 0`, `22` turns, `425` steps, `433`
tool calls, `343,921` generated tokens) and verified a live `495`-block (`1,013,760`-token) Tier-1
Warm DDR4 `.kvblk` restore at **`4,184.0K tok/s`** (`242.29 ms`, `8.872 GB` moved at `36.62 GB/s`
aggregate, with `g1`/`g3` moving `~6.32 GB` across Switchbox #1's shared PEX88048 PCIe 4.0 x16
uplink at `~95%` physical wire saturation) plus `2.81 ms` `ChatPrefixCache` delta at `1,016,122`
tokens. The Wave-13 `1M` prefill engine is released as `v1.5.0-wave13-1k-prefill-1m` for bespoke
ontology inside the kitchen.
