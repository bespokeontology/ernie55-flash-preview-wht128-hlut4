# Wave-10 Architecture & 512K -> 1M Pre-Release Systems Audit

Release: `v1.2.0-wave10-1m-turbo` (Apache-2.0). Co-authored by bespoke ontology (the kitchen)
and Google DeepMind / Gemini (Gemini Carbon & Gemini Neon) & Antigravity.
Workstation: SB-M5-P2-192G, AMD EPYC 7542, 5x AMD Radeon Instinct MI50 (gfx906, 128 GiB HBM2),
2x Tenstorrent Blackhole P150A (64 GiB GDDR6, 330 MiB L1). Zero prohibited hardware-slang, zero
power-cost claims, zero PII, zero local filesystem paths.

## 1. Seven-commit fundamentals-first zero-P2P ladder

| # | commit | tag | invariant delivered |
|---|--------|-----|---------------------|
| 0 | `a1b7e88` | `fix-hf-main-model-tree` | Unsplit HF main model tree, scalar `base_model` |
| 1 | `c078a18` | `wave10-c1-p150-full-async-l1` | Eager Tensile warmup, zero-copy SHM DMA, 253.7 MiB L1, `kBatchSlots=64` |
| 2 | `85b9f7e` | `wave10-c2-2mb-host-sdma-fp16-ar` | True full-duplex 2 MB host-SDMA FP16 AllReduce, 1-pass mHC pre-GEMM |
| 3 | `91944ea` | `wave10-c3-tp4-dsa-hh-indexer` | TP4 16-head sparse MQA DSA, unique-KV host-SDMA compaction |
| 4 | `fbaa7e2` | `wave10-c4-hybrid-m64-m32-moe` | Hybrid m64/m32/m8 expert-sorted CSR MoE, shadow-expert offload |
| 5 | `2389d3e` | `wave10-c5-pipelined-nvme-kvblk` | Pipelined Dual-NVMe O_DIRECT `.kvblk`, WB hugepage DDR4 slots |
| 5b | `b8eed5f` | `wave10-c5b-singlepass-k16k-flashindexer` | Single-pass K_tile = 16,384 FlashIndexer, strictly O(L) |
| 6 | `8e64abc` | `wave10-c6-dspark-longctx-dfly2-hook` | Fused DSpark markov+multimax, zero-copy `Dfly2SharedSparsityView` |

## 2. Kernel, memory, stream and zero-copy invariants

- 302/302 gfx906 kernels: `ScratchSize == 0`, 0 spills, 0 `slc`, 0 trig; `fdot2` only.
- 0 B net new GPU `hipMalloc` per commit; 0 B DDR4 expert spill; Type_HH_HPA
  (`rocblas_datatype_f32_r` compute) everywhere; FP32 router logits.
- Streams: `stream` (owner compute), `stream_sh` (shared experts + host SDMA),
  `keep_warm_stream` (prefill warm). Cross-stream ordering is event-based
  (`ev_mb_c2d`, `ev_mb_d2c`, `ev_sh_done`); no host-buffer reuse across streams.
- Zero BAR1 P2P: no `hipDeviceCanAccessPeer`, `hipDeviceEnablePeerAccess`,
  `hipMemcpyPeer`, `hipMemcpyPeerAsync` anywhere in `mi50/` or `tools/`.
- `Dfly2SharedSparsityView`: metadata-only, non-owning, zero-copy, `valid=false`
  until the runtime is initialized.

## 3. 512K -> 1M overflow and bounds audit findings

- FlashIndexer score offsets already use `size_t` math
  (`size_t(b0) * size_t(b_max_hist)`, `size_t(bi) * 8u * size_t(k_cnt)`), so
  `b * max_hist` cannot overflow at 524,288 or 1,048,576.
- KV offsets `pos * 576`, `pos * 512`, `pos * 128`, `n_comp * 512` stay below
  2^31 for `pos <= 1,048,576` (max 603,979,776) but must remain `size_t`/`int64_t`
  in any new code; `int32_t` is only safe up to ~3.7M positions.
- `DdrSlot` wraparound: block indices exceed 256 only past 524,288 tokens; slot
  reuse is guarded by `in_flight` pinning, so wraparound is safe.
- `k_dsa_select_topk2048_wave6` / `k_dsa_indexer_score_topk8_dpp4` stride math is
  `int32_t`; at `L = 1,048,576` the products stay < 2^31 but have < 2x headroom.
  Recommendation: keep `L` stride products in `size_t` in any future change.

## 4. Measured silicon telemetry (`session-fa82666c`, `626,187` peak live context tokens)

- Prefill (`M>=1024` live context buckets across 7 Wave-10 logs):
  - `<=64K` (`n=30`): `460.2 tok/s`
  - `64K-128K` (`n=27`): `384.6 tok/s`
  - `128K-192K` (`n=21`): `360.2 tok/s`
  - `192K-256K` (`n=25`): `339.5 tok/s`
  - `256K-320K` (`n=25`): `315.7 tok/s`
  - `320K-384K` (`n=27`): `310.8 tok/s` (`343.9 tok/s` peak @ `323.5K`)
  - `384K-512K` (`n=61`): `272.4 tok/s` (`265.2 tok/s` @ `483.3K`)
  - `>512K` (`512K-626K`, `n=49`): `235.5 tok/s` (`260.2 tok/s` @ `514K`, `249.9 tok/s` @ exact `524,288`, `229.3 tok/s` @ `624.6K`)
- DSpark decode (Commit 6 fused path): `32.25-35.06 tok/s` at `325K-335K`, `27.01 tok/s` at `488.3K`, `22.60-26.55 tok/s` at `624K-625K` (`26.58 tok/s` campaign mean across `164,108 / 355,891 = 46.11%` accepted tokens).
- `.kvblk` cold restore: `703.8K tok/s` (`436.47 ms` @ `307,200` tok) and `702.5K tok/s` (`454.78 ms` @ `319,488` tok) live 4-GPU end-to-end; `919.5K tok/s` (`8.04 GB/s` @ `262K`) and `1,023.1K tok/s` (`8.95 GB/s` unit) Dual-NVMe O_DIRECT verify (`memcmp == 0`).
- Verification: `302/302` zero-scratch `gfx906` kernels, `11/11` Gate-0 unit test suites PASS, `0` BAR1 P2P calls, `0 B` net new GPU VRAM, `0 B` DDR4 expert spill, `0/0` PCIe AER across `69` devices, `10/10` zero-PII gate PASS.
