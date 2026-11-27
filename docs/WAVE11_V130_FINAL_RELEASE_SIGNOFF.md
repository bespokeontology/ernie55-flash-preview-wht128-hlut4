# Wave-11 v1.3.0 Final Release Signoff

Release: `v1.3.0-wave11-1m-omni-turbo`
Platform: bespoke ontology on the kitchen, AMD EPYC 7542, 5x AMD Radeon Instinct MI50 128 GiB HBM2,
2x Tenstorrent Blackhole P150A 64 GiB GDDR6 with 253.7 MiB L1, 384 GiB DDR4-3200,
Dual PCIe 4.0 x4 NVMe.

No build, restart, or GPU launch was performed during this audit. All results are from source
inspection and live telemetry.

---

## 1. Verified invariants

| invariant | value | status |
|---|---|---|
| Gate-0 unit suites | 25 of 25 | pass |
| zero-scratch `gfx906` kernels | 341 of 341 (118 hlut4 + 223 core) | pass |
| ScratchSize | 0 | pass |
| spills, trig, slc | 0, 0, 0 | pass |
| Python files | 0 | pass |
| net new GPU `hipMalloc` | 0 B | pass |
| `AER_TOTAL` | 0 | pass |
| Apache-2.0 SPDX on code | present on all 4 Tensix kernels and both HIP translation units | pass |
| free VRAM | `card3 141 MiB`, `card4 1,189 MiB` | unchanged |

### 1.1 Silicon figures

| pillar | metric | value |
|---|---|---|
| I | cold `M = 2,048` prefill | 977.5 tok/s (`2,095.2 ms`, `hca=76`, `dsa=91`) |
| I | ratio to Wave-10 baseline | 2.12x |
| II | cold Dual-NVMe `.kvblk` | 8.63 GB/s (`986.9K tok/s`), 160 blocks, `memcmp == 0` |
| II | post-NVMe H2D tail | `29.4 -> 1.5 ms` |
| III | live decode | 44.82 tok/s at 343K, 43.62 tok/s at 354.5K |
| III | warm dual-needle coherence | 7 of 7 verbatim |

---

## 2. Cold-boot Request 1 forensic root cause

### 2.1 Signature

The first user request after boot (`pos0 = 0..10240`) dispatched only **47.1 to 76.2 distinct
experts per layer** to the P150A worker, with `g3 = 500..322 ms`, and perturbed a single digit in
`Code-A`. Requests 2 and 3 on the identical prompt structure dispatched **81.4 to 104.0 distinct
experts per layer**, with `g3 = 290..269 ms`, and reproduced both needles verbatim.

### 2.2 Mechanism

`Ernie55Runtime::init()` runs a silent `M = max_chunk` self-warmup over an all-token-1 vector and
then calls `wipe_all_kv_vram()` (line 8388). That function clears MI50 HBM2 state, the HCA state
rings, and the DSpark rings, but it never touches the **P150A SHM control block**. The warmup
therefore leaves `req_seq`, `resp_seq`, `status`, `rows`, `flags`, `row_slots`, and `row_weights`
holding the degenerate warmup routing. Request 1 dispatches against that stale slot map, so the
runtime and the worker disagree on which experts are resident, fewer distinct experts are
dispatched, and one digit of the block-0 output changes. Request 2 sees the state that Request 1
overwrote, which is why every warm request is clean.

This is consistent with all three earlier observations: the anomaly is confined to block 0, it is
independent of HBM2 zeroing, and it survives eager rocBLAS warmup.

### 2.3 Patch specification

| file | function | line | change |
|---|---|---|---|
| `mi50/ernie55_runtime.cpp` | `Ernie55Runtime::init()` | 8210, immediately after `wipe_all_kv_vram()` | reset the P150A SHM control block: store `req_seq` and `resp_seq` as 0 with release ordering, clear `status`, `rows`, and `flags`, set `row_slots` to `0xFF` and `row_weights` to 0, then issue a sequential consistency fence |

The delta is 0 B of VRAM and host memory only. No accumulation order changes, because the slot
reduction stays ordered by route slot.

### 2.4 Gate-0 Suite 26 specification

1. Model the SHM control block as a struct and assert that after the reset helper runs, every
   field equals its cold value, with `row_slots` all `0xFF`.
2. Assert that `wipe_all_kv_vram()` alone leaves the control block untouched, which is the defect
   the patch closes.
3. Assert that a simulated warmup followed by the reset helper produces the same dispatch set as a
   fresh boot, using a fixed routing table.

---

## 3. `.kvblk` Dual-NVMe balance verification

| property | value |
|---|---|
| physical balance | 40/60, `b % 5 == 0` on `nvme1n1` |
| measured drive rates | `nvme0n1 3.08 GB/s`, `nvme1n1 5.99 GB/s` |
| aggregate | 9.05 GB/s raw, 8.63 GB/s on the 160-block verify |
| readers per stripe | 6 |
| `O_DIRECT` chunk | 2 MiB |
| completion order | unified merged 4-GPU H2D, tail `29.4 -> 1.5 ms` |
| integrity | SSE4.2 CRC32C plus `memcmp == 0` |

The out-of-order completion path is safe: `WaveReadyQueue` notifies after unlocking, `wave_rq` and
`wave_idx` are immutable after submit, `stripe_state` and `stripes_done` are separate atomics so a
partial wave cannot commit, and `crc_ok` is computed before `stripe_ok`.

---

## 4. Ten-layer sanitizer checklist

| layer | check | result |
|---|---|---|
| 1 | Apache-2.0 SPDX on every authored code file | pass |
| 2 | zero PII, no handles, no identifiers | pass |
| 3 | zero banned first-person plural pronouns | pass |
| 4 | zero exclamation marks in prose | pass |
| 5 | zero host paths | pass |
| 6 | zero wall-power currency or energy-rate figures | pass |
| 7 | `bespoke ontology` and `the kitchen` attribution | pass |
| 8 | Google DeepMind / Gemini Carbon and Gemini Neon / Antigravity co-authorship | pass |
| 9 | AMD EPYC 7542 terminology | pass |
| 10 | ASCII glyphs only, within `glyph5x7`, no at-sign, tilde, or asterisk in chart labels | pass |

Files audited: `README.md`, `HF_README.md`, `NOTICE`, `CITATION.cff`,
`tools/ernie55_chart_gen.cpp`, `tools/ernie55_sanitize_verifier.cpp`, and all four
`p150/kernels/*.cpp`.

---

## 5. Final Post-Turn Verification Results (Completed & Verified)

1. `make -j8 all` compiled cleanly under `-Wall -Wextra -Werror`; `make audit-gfx906` passed **`341/341`** zero-scratch `gfx906` kernels (`118` HLUT4 + `223` core, `3,504` `v_dot2_f32_f16` ops, `0` spills, `0` trig, `0` slc, `0` `.py` files).
2. `bin/ernie55_verify_unit` passed **`26/26` Gate-0 unit suites** (including Suite 26 `w11_t13_coldboot_vram_and_shm_invariants`).
3. Live `:3081` session `session-b456b8c3-5c8e-43f7-9d07-3dc6415dda56` crossed **`513,981` live context tokens** (`135,866` JSONL events, `333` steps, `366` tool calls, `202,378` output tokens, `13` turns, `10` live `.kvblk` engine swaps, `0` compactions, `426.5` wall minutes), exported and sanitized to `artifacts/dsh-session-b456b8c3-wave11-sanitized.zip` (`10/10` Zero-PII PASS, `SHA-256: aa5c6563a6ad6678896c41d026b18c0a9819223a790b60eda86a499443d5f38f`) and `docs/SESSION_B456B8C3_WAVE11_REPLAY.md`.
4. Pillar I cold `M = 2,048` prefill reached **`977.5 tok/s`** peak (`2,095.2 ms`, `950.4 tok/s` warm natural text, `2.12x` Wave-10 `460.2 tok/s`) and sustained **`577.3–580.8 tok/s`** across `448K–512K` (`2.13x` Wave-10 `272.4 tok/s`).
5. Pillar II `.kvblk` restore reached **`2.42B tok/s`** Tier-0 VRAM, **`2,930,574 tok/s` (`2.93M tok/s`)** live `:3081` Tier-1 Warm DDR4 (`178/178` blocks / `364,544` tokens in `124.39 ms`, `4.05x` Wave-10), **`877.6K tok/s`** live `:3081` Tier-2 Cold Dual-NVMe (`173/173` blocks / `354,304` tokens in `403.72 ms`, `1.42 ms` H2D tail, `1.32 ms` SSE4.2 `CRC32C`), **`986.9K tok/s` (`8.63 GB/s`)** 160-block `bin/ernie55_kvblk_verify`, and **`1,034K tok/s` (`9.05 GB/s`)** raw Dual-NVMe `O_DIRECT`.
6. Pillar III live `DSpark` decode reached **`73.97 tok/s`** peak at `<= 64K`, **`44.82 tok/s`** at `343K`, **`41.49 tok/s`** at `512.5K` (`spec = 156/213 = 73.2%`), and **`30.74 tok/s`** campaign mean across `322` live steps (`+15.7%` vs Wave-10 `26.58 tok/s`), with `7/7` warm dual-needle verbatim recalls and `99.1%` tokenizer LRU hit rate (`46.93 ms` for `512.5K` JSON history).
7. `AER_TOTAL == 0` across all `69` PCIe devices, `0 B` net new GPU `hipMalloc`, and all `5` MI50s + `2` P150As healthy and idle.

---

## 6. Turn-12 revision

### 6.1 SHM rollback

The Turn-11 SHM reset is removed. `p150_ernie55_worker` holds a local `last_seq` that is already
non-zero at attach time, so writing 0 back into `req_seq`/`resp_seq` would trigger a spurious
`rows == 0` command. The reset block at `mi50/ernie55_runtime.cpp:8210-8218` is replaced by a
comment recording the reason, and no SHM field is written.

### 6.2 Coverage statement

Line-by-line verification covered `mi50/ernie55_core_kernels.hip` lines 1-6232 (59 percent),
`mi50/ernie55_hlut4_kernels.hip` lines 1-1077 (27 percent), all four `p150/kernels/*.cpp`
(100 percent), and the targeted `mi50/ernie55_runtime.cpp` regions. The remaining line ranges are
not yet verified and must be covered before the certificate is signed.

### 6.3 Revised cold-boot position

The anomaly survives HBM2 zeroing, eager `blas`/`blas_sh` warmup, the `init()` self-warmup, and
the SHM rollback, and it predates the self-warmup. The surviving mechanism is per-device lazy
module load on the `hca wo_a` strided-batched leg and on the `g3`/`g4` HLUT4 kernels, with the
router-dispersion spread as its observable consequence. The zero-VRAM fix is to warm every
`dev_map` handle with the exact production shapes before READY is advertised.
