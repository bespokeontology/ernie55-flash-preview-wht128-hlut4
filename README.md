# ERNIE-5.5-Flash-Preview (`300B` / `14B` Active) — Higher-Precision `BF16`-Master `WHT128-HLUT4` (`4.125 bpw`, `W4A16/FP32`) + Lossless `BF16` Spine, Indexer & `1M` KV on `192 GiB`

**Pure C++17 / `gfx906` HIP / Tenstorrent Blackhole `TT-Metalium` C++20 inference engine, `FP64` `LDLQ` Walsh-Hadamard `4`-bit lookup quantizer, and silicon verification suite (`v1.2.0-wave10-1m-turbo`, `302/302` zero-scratch `gfx906` kernels, `0` `.py` files, `0 B` CPU DDR4 expert spill) built directly from the unquantized [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) master checkpoint (`Apache-2.0`).**

Unlike Baidu's default [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) deployment checkpoint—which quantizes the non-expert spine to blockwise `FP8`, runs routed experts in `W4A8` `MXFP4` (`4.25 bpw`), stores the `1M` KV cache in `FP8`, and runs the `DSA` sparse indexer in `4`-bit `MXFP4` with distilled head pruning ([vLLM Day-0 Technical Specification](https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview))—this release preserves **`100%` unquantized `BF16`/`FP32` across the entire non-expert spine, `100%` lossless `BF16` across the `1,048,576`-token KV cache (`0.0000 nats` KV loss), `100%` lossless `BF16`/`FP32` across all `8/8` `DSA` indexer heads (`0` IEEE-754 bit-diffs, `5.19x` faster via `4`-lane VALU DPP), and `W4A16/FP32` `16`-centroid `WHT128-HLUT4` (`4.125 bpw`, `19.3858 dB` mean SQNR, `0.012301 nats` post-header median KLD)** across all `23,040` backbone and `DSpark` routed experts on commodity `HBM2` + spatial `SRAM` without `FP4` tensor-core hardware lock-in.

* **Engine & Kernels Repository:** `bespokeontology/ernie55-flash-preview-wht128-hlut4` (`v1.2.0-wave10-1m-turbo`, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) + `NOTICE`)
* **Unsplit Native Quantized Weights:** `https://huggingface.co/kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4` (`Apache-2.0`, `23,040` routed `WHT128-HLUT4` experts + `100%` unquantized `BF16`/`FP32` spine + `SHA256SUMS.txt`)
* **Upstream Ground Truth:** [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) (master `BF16` source), [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (official `W4A8` + `FP8` default), [`baidu/vLLM-ERNIE`](https://github.com/baidu/vLLM-ERNIE), [vLLM Day-0 ERNIE-5.5-Flash-Preview Post](https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview), and [NVIDIA NeMo AutoModel ERNIE-5.5 Reference](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/ernie5_5/model)
* **Companion `SB-M5-P2-192G` Releases:** [`bespokeontology/dsv41-wht128-hlut2` (`v2.1.0-wave10-192g`)](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v2.1.0-wave10-192g), [`bespokeontology/hy4-flash-wht128-hlut4` (`v1.0.0-hlut4-192g`)](https://github.com/bespokeontology/hy4-flash-wht128-hlut4/releases/tag/v1.0.0-hlut4-192g), and [`bespokeontology/hy4-770b-wht128-192g` (`v1.0.0-770b-192g`)](https://github.com/bespokeontology/hy4-770b-wht128-192g/releases/tag/v1.0.0-770b-192g)

## 1. Subsystem-by-Subsystem Precision Proof vs. Official Default Checkpoint (`baidu/ERNIE-5.5-Flash-Preview`)

Baidu ships three official repositories: [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) (`~600 GB` unquantized master), [`baidu/ERNIE-5.5-Flash-Preview-FP8`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-FP8) (`~300 GB` blockwise `FP8`), and the default [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (`MXFP4` `W4A8` experts + `FP8` spine + `FP8` KV + `MXFP4` `DSA` indexer). While Baidu's default checkpoint uses post-training quantization-aware training (QAT) to align `W4A8` fake-quant rules on datacenter `FP4`/`FP8` GPUs, starting directly from [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) lets `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` hold strictly higher numerical precision on every surrounding weight, activation, attention, sparse-indexer, and `1M` KV path—at a smaller expert bit-budget (`4.125 bpw` vs. `4.25 bpw`):

| Inference Path / Tensor Group | Official Default [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (`vLLM-ERNIE`) | This Release (`ERNIE-5.5-Flash-Preview-WHT128-HLUT4`) | Measured Fidelity & Architectural Advantage |
| :--- | :--- | :--- | :--- |
| **Target Non-Expert Spine (`43` `HCA`/`MLA-DSA` Layers, Shared Experts, Routers, `DSpark` Spine, Target LM Head)** | Blockwise `FP8` (`8`-bit `E4M3`, `128x128` blocks) + folded `FP8` VHA post-mix | **`100%` Unquantized `BF16` / `FP16` / `FP32`** on disk (`16/32`-bit target runtime; `FP8 E4M3FN` used strictly on the `DSpark` draft proposal head, `100%` verified against the `16`-bit target LM head) | Eliminates `FP8` weight & intermediate requantization across every attention projection, router, shared expert, and `201,216`-vocab `MultiMax` verification head |
| **Expert Activation & Accumulator Path** | **`W4A8`** (`8`-bit dynamic `FP8` input quantization before routed expert GEMMs) | **`W4A16` with `FP32` Accumulation** (`16`-bit `FP16` Hadamard-rotated activations into `v_dot2_f32_f16` & `SFPLUTFP32`) | Doubles activation bit-width (`16`-bit vs `8`-bit) entering all `23,040` routed `SiTU-GLU(4.0, 25.0)` experts |
| **Routed Expert Weight Codebook (`42` Backbone + `3` `DSpark` Layers = `23,040` Experts)** | **`4.25 bpw` `MXFP4` (`E2M1` + `E8M0`)**: `15` fixed Cartesian float values (duplicate `±0.0` wastes 1 of 16 codes) + coarse power-of-two `E8M0` scale per `32` weights (QAT-aligned; requires native `FP4` silicon) | **`4.125 bpw` `WHT128-HLUT4` (`FP64` `LDLQ`)**: `128`-dim Walsh-Hadamard incoherence rotation + **`16` distinct learned `FP16` Lloyd-Max centroids** (`0` wasted codes, continuous `FP16` levels) + Cholesky error feedback over `524,288` tokens | **`0.125 bpw` denser** (`139.22 GiB` total) with **`19.3858 dB` 45-layer mean SQNR**, **`0.012301 nats` post-header median KLD (`0.999066` cosine)** vs exact `BF16`, and zero `Blackwell` `FP4` lock-in |
| **`1,048,576`-Token (`1M`) KV Cache (`37` `128:1` `HCA` + `6` `MLA-DSA`)** | **`FP8` (`8`-bit, `--kv-cache-dtype fp8`)** + optional CPU offload | **`100%` Lossless `BF16` (`16`-bit)** resident in HBM2 (`0 B` DDR4 spill) + striped `O_DIRECT` `.kvblk` (`memcmp == 0`) | **`0.0000 nats` KV quantization loss** across the entire `1,048,576`-token window and across cold NVMe session restores (`793,925 tok/s` @ `262K`) |
| **`DSA` Sparse Indexer (`6` Layers Selecting `Top-2048` Keys out to `1M`)** | **`4`-bit `MXFP4` (`indexer_kv_dtype: mxfp4`)** + `1B`-token distilled head-pruning router (`1.4x` speedup @ `1M`) | **`16`-bit `BF16` `ik_cache` (`4x` bit-width) + `FP32` Exact `8/8`-Head Scoring** via `4`-lane VALU DPP `quad_perm` (`0` dropped heads) | **`0` IEEE-754 bit-diffs**, **`13,056 / 13,056` exact `Top-2048` indices**, and **`5.19x` speedup @ `262K`** (`30.27 -> 5.84 ms`) without approximate head pruning |
| **Runtime & Deployment Stack** | Python `vllm==0.32.0` + `vLLM-ERNIE` plugin (`--trust-remote-code`) on `FP4`-capable datacenter GPUs | **Standalone Pure C++17 / `gfx906` HIP / Blackhole C++20 (`0` `.py` files)** on `192 GiB` commodity `MI50` HBM2 + `P150A` spatial SRAM | Live `:3081` Web GUI parity (`x_flash_progress` v1 counters, `cached_tokens`, `2 s` SSE keepalives, identifier-first `<ernie:tool_call>` streaming) |

## 2. Executive Silicon Scorecard (`SB-M5-P2-192G` Single Workstation)

[`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) pairs a `43`-layer (`280B`) hybrid sparse backbone (`37` `128:1` `HCA` layers + `6` `MLA-DSA` layers + `4`-stream `FP32` `mHC` residual connections + `4`-piece [`MultiMax` / `SegLU`](https://zhouyuxuanyx.github.io/2026/11/05/one-knob-is-not-enough-softmax-temperature-vs-multimax.html) logit transform over `201,216` vocabulary tokens) with a native `3`-layer (`~20B`) `512`-expert MoE `DSpark` parallel speculative drafter (`model.mtp.{0,1,2}`). Compressing all `45` MoE layers (`23,040` routed `2048x2048` experts) into uniform `WHT128-HLUT4` shrinks `562.5 GiB` of `BF16` expert weights to **`139.22 GiB`**, fitting the entire `300B` system plus a **`1,048,576`-token lossless `BF16` KV cache** inside `5x AMD Radeon Instinct MI50` (`128 GiB` HBM2) + `2x Tenstorrent Blackhole P150A` (`64 GiB` GDDR6 + `330 MiB` Tensix L1 SRAM over [`4x 800G` `FABRIC_1D`](https://docs.tenstorrent.com/aibs/blackhole/multi-card-topologies.html)) with **`0 B` CPU DDR4 expert spill**:

| Subsystem / Verification Gate | Verified Silicon Result | Receipt / Artifact |
| :--- | :--- | :--- |
| **Pure C++ / Zero-Scratch `gfx906` ISA** | **`302/302`** kernels (`112` HLUT4 + `190` core; `ScratchSize == 0`, `0` spills, `0` hw trig, `0` `slc` atomics, `2,448` `fdot2` ops, `0` `.py` files, `0` BAR1 P2P calls, `-Wall -Wextra -Werror`, `11/11` Gate-0 unit suites) | `build/ernie55_hlut4_kernels.s`, `build/ernie55_core_kernels.s`, `artifacts/gate_wave10_512k_receipt.json` |
| **`45`-Layer `WHT128-HLUT4` Pack (`23,040` Experts)** | **`139.22 GiB`** (`3,321,892,864 B` x `45` unsplit shards), **`19.3858 dB`** mean SQNR (`cal_tokens = 524,288`, `ldlq_damp = 0.01`), **`0 B` DDR4 spill** | `artifacts/hlut4_preview_pack_manifest.json` |
| **Empirical Router Placement (`"5MRP"`)** | `318 + 13` Wave-8 shadow-cached hot experts/layer on `5x MI50` (`94.89%` base hit; **`99.33% / 97.85% / 97.29%` on `DSpark` `L43..L45`**) + `68` in `P150A` Tensix L1 (**`98.06%` combined**) + `113` in `P150A` GDDR6 | `router_perm.bin` (`47,112 B`) |
| **`GATE-1` Lossless Operator & `DSpark` Proofs** | `INV-1b` **`64/64` bit-exact** `DSpark == M=1` (`2,040` `HCA`-boundary rollbacks); `INV-1c` `TV(K=5) = 0.00961`; `INV-1d` **`cos = 1.00000000`** (`L = 129, 4097, 65539`; `L = 262,147` `cos(warm) = 1.00000000`, `cos(137) = 0.99973877`, exact Top-1); `MI50` vs `P150A` `cos = 0.999990-0.999994` | `artifacts/gate1_ernie55_preview_hlut4_1m.json` |
| **`GATE-2` Disjoint `BF16` Oracle KLD (`0` 8-gram overlap)** | Post-header (`2,304` pos): **`Median KL = 0.012301 nats`**, **`Cos = 0.999066`**, `Top-1 = 85.20%` (All `4,096` pos: `Median KL = 0.039143 nats`); Router `EAR@10 = 94.72%` | `artifacts/gate2_ernie55_preview_hlut4_1m.json` |
| **`GATE-4` Wave-9 (`v1.1.0`) Live Prefill & `2K -> 262K` Ladder** | Cold `M=2048` prefill: **`457.6 tok/s`** peak (`454.4 tok/s` across cold `4K`, **`+26.1%`** vs `v1.0.0` `362.8`); `DSpark` greedy sweep (`64/64` exact): **`34.39 tok/s`** (`K=4`), live `:3081` probabilistic peak **`41.23–42.35 tok/s`** (`spec=56/73`, `76.7%`); Context ladder (`2K/4K/64K/262K`): **`41.23 / 38.38 / 27.38 / 22.01 tok/s` decode**, **`457.6 / 454.4 / 303.8 / 228.5 tok/s` cold prefill** | `artifacts/gate4_ernie55_preview_hlut4_1m.json` |
| **`Wave-10` (`v1.2.0-wave10-1m-turbo`) Zero-P2P Live Scaling (`<=64K -> 626.2K`)** | Single-diff 7-commit zero-P2P ladder (`0 B` net new GPU VRAM, `0/0` PCIe AER across `69` devices): **`460.2 / 384.6 / 360.2 / 339.5 / 315.7 / 310.8 / 272.4 / 235.5 tok/s`** mean live `M>=1024` prefill across `<=64K / 64-128K / 128-192K / 192-256K / 256-320K / 320-384K / 384-512K / >512K` (**`260.2 tok/s` @ `514K`**, **`249.9 tok/s` @ exact `524,288`**, **`229.3 tok/s` @ `624.6K`** — beating Wave-8's `262K` `228.5 tok/s` at `2.38x` the context!); Commit 6 fused `DSpark` decode: **`32.25–35.06 tok/s` @ `325K–335K`**, **`27.01 tok/s` @ `488K`**, **`22.60–26.55 tok/s` @ `624K–625K`** (`26.58 tok/s` mean across `355,891` drafted tokens) + zero-copy `Dfly2SharedSparsityView` | `artifacts/gate_wave10_512k_receipt.json`, `docs/WAVE10_ARCHITECTURE_AND_AUDIT.md`, `docs/WAVE10_1M_KERNEL_AUDIT_LEDGER.md` |
| **3-Tier `2,048`-Block `.kvblk` Engine (`VRAM` + `DDR4` + Dual-NVMe `O_DIRECT`)** | Live `:3081` end-to-end 4-GPU restores (`memcmp == 0`, SSE4.2 `CRC32C`, `WB` hugepage `DdrSlots` + `8+8=16` stripe workers): **Tier 0 `VRAM` `109.9M–1,106M tok/s`**; **Tier 1 `64`-slot `DDR4` `723.1K tok/s`**; **Tier 2 cold Dual-NVMe `O_DIRECT` `703.8K tok/s` (`436.47 ms` @ `307,200` tok / `150` blks) & `702.5K tok/s` (`454.78 ms` @ `319,488` tok / `156` blks, `5.29x` vs Commit 2 `133.0K`)**; offline `--blocks 128` Dual-NVMe `O_DIRECT` **`919.5K tok/s` (`8.04 GB/s` @ `262K`, `1,023.1K tok/s` / `8.95 GB/s` unit)** | `mi50/ernie55_kvblk_cache.hpp`, `tools/ernie55_kvblk_verify.cpp`, `artifacts/gate_wave10_512k_receipt.json` |
| **Sanitized Long-Horizon Autonomous `:3081` Traces (`626.2K` + `198.0K` + `158.6K`)** | **`626.2K`-token `session-fa82666c` (`140,513` JSONL events, `206` steps, `222` tool calls, `211,461` output tokens, `0` compactions, `0/0` PCIe AER)**: single unbroken Wave-10 co-engineering campaign across Commits 1–6 (`307K`/`319K` cold `.kvblk` swaps, `11/11` unit tests, `10/10` Zero-PII PASS, replay in `docs/SESSION_FA82666C_626K_REPLAY.md`); plus Wave-9 **`198.0K` `session-6b2a64a4`** and **`158.6K` `session-b4cf2731`** | `artifacts/dsh-session-fa82666c-626k-sanitized.zip`, `docs/SESSION_FA82666C_626K_REPLAY.md`, `artifacts/dsh-session-6b2a64a4-198k-sanitized.zip` |

![ERNIE-5.5-Flash-Preview WHT128-HLUT4 1M Telemetry](./artifacts/ernie55_preview_hlut4_1m_telemetry.png)

## 3. Heterogeneous `192 GiB` Placement & `1M` Lossless KV Budget

All `5x Radeon Instinct MI50` cards (`3x 32 GiB` + `2x 16 GiB` = `128 GiB` HBM2 at `~5.1 TB/s` aggregate bandwidth, mapped with thermal ordering `HIP_VISIBLE_DEVICES=0,2,3,4,1` across the dual-`PEX88048` switched PCIe backplane) and `2x Tenstorrent Blackhole P150A` cards (`64 GiB` GDDR6 + `330 MiB` Tensix L1 SRAM linked via `4x 800G` = `3.2 Tbps` `FABRIC_1D`) operate with **`0 B` CPU DDR4 expert spill**:

| Accelerator / Pool | Physical Capacity | Assigned Tensors & `1,048,576`-Token Lossless `BF16` KV Allocation |
| :--- | :--- | :--- |
| **`TP=4` `MI50` Spine + Hot MoE (`3x 32 GiB` + `1x 16 GiB`)** | `112 GiB` HBM2 | `TP=4` unquantized `FP16`/`BF16`/`FP32` non-expert spine (`L00-L42` `HCA`/`MLA-DSA`, shared experts, `mHC`, `FlashIndexer`, `DSpark` spine, `MultiMax`) + **`264` hot `WHT128-HLUT4` experts/layer** (`{66, 68, 92, 38}` across `45` layers) + **`1,048,576`-token lossless `BF16` KV pool** (`37` `128:1` `HCA` block/SWA/sink caches + `6` `MLA-DSA` `576`-dim latent & `128`-dim `BF16` `ik_cache`) |
| **Dedicated `MI50` MoE Worker (`1x 16 GiB`)** | `16 GiB` HBM2 | **`54` hot `WHT128-HLUT4` experts/layer** across all `45` MoE layers (`2,430` experts = `14.68 GiB`), bringing total `MI50` HBM2 coverage to **`318 / 512` experts/layer (`94.89%` backbone hit; `97.29%-99.33%` `DSpark` hit)** |
| **`2x Tenstorrent Blackhole P150A` (`FABRIC_1D`)** | `64 GiB` GDDR6 + `330 MiB` L1 | **`194` remaining `WHT128-HLUT4` experts/layer** across `45` layers (`8,730` experts = `52.75 GiB` total), with the top `68` tail experts pinned in on-die Tensix L1 SRAM (**`98.06%` combined `MI50 + P150A L1` hit**) and `SFPMULI + SFPLUTFP32` native `SiTU-GLU(4.0, 25.0)` evaluation (`cos = 0.999990-0.999994` vs `MI50`) |
| **Dual Striped PCIe NVMe (`nvme0n1` + `nvme1n1`)** | `O_DIRECT` `.kvblk` | Lossless `BF16` session persistence (`CRC32C` hardware-verified per stripe): **`793,925 tok/s` (`7.03 GB/s`)** cold VRAM restore at `262K` tokens (`memcmp == 0`, `max_logit_diff = 0.000000`) |

## 4. Silicon Kernel Architecture & Wave-10 (`v1.2.0-wave10-1m-turbo`) Zero-P2P Stack

Every device kernel in `mi50/ernie55_hlut4_kernels.hip` and `mi50/ernie55_core_kernels.hip` is compiled for `gfx906` (`-Wall -Wextra -Werror`) and mechanically audited via `make audit-gfx906` for **`302/302`** `ScratchSize == 0`, `0` VGPR/SGPR spills, `0` hardware trig instructions, `0` `slc` PCIe RMW atomics, and `0` BAR1 P2P calls (`AMD EPYC 7542` host, `0/0` PCIe AER):

1. **On-Chip `gfx906` LDS SRAM `M=64` `WHT128-HLUT4` MoE & Direct Weighted `FP32` `atomicAdd`:** Pre-expands `16`-centroid `FP16` codebooks into `32`-bank `s_lut256` LDS and streams `M=64` active expert tiles through on-chip LDS SRAM with direct weighted `FP32` `atomicAdd` into `d_y_acc_f32[M, 2048]` (eliminating `d_y_slots_f32`, freeing `+120 MiB/GPU`, unlocking `--prefill-chunk 2048` on `1M` KV).
2. **Aligned `Type_HH_HPA` Dense/HCA/DSA Projections & Exact Rank-16 Low-Rank VHA (`43/43` Layers):** Replaces `HS_HPA` (`FP32` C/D) with `64`-row aligned `Type_HH_HPA` (`FP16` C/D, `FP32` accumulate, `max_abs_diff <= 1.22e-4`, freeing `+123 MiB/GPU`), and factors all `43/43` VHA mixing matrices in `FP64` to exact rank-16 (`< 1e-4` error, `4x` less `TP=4` PCIe traffic).
3. **`1`-WG/Token (`512`-Thread) Zero-Gather `4x4` QK + `8x8` PV `v_dot2_f32_f16` Sparse `DSA` & `select_topk2048_gpu`:** Eliminates the `d_dsa_ksel_f16` gather tensor (`-19.25 MiB`/DSA GPU) via branchless `__builtin_amdgcn_cvt_pkrtz` + bank-conflict-free `s_KV[32][289]` LDS (`43.46 ms` vs `61.78 ms` = `1.42x`, `max_abs_diff = 6.10e-5`, `ScratchSize == 0`, `0` spills), paired with GPU parallel `select_topk2048_gpu` (`12.2 ms` vs `56.2 ms` = `4.60x`, `0` mismatches) -> live `dsa` prefill dropped `1,861 ms -> 1,110 ms` (`1.56x`).
4. **Blackhole `P150A` Single-Pass `2,048`-Row Prefill on `PCIe Gen4 x16` (`{0,1}`), `HiFi2` & On-Device `BF16` `FABRIC_1D` AllReduce:** Routes SHM v3 native `FP16` inputs / `BF16` outputs directly to the `x16` Blackhole card (`h2d = 1.05 ms/L`, `traced = 42/42` on both prefill and decode, decode `0.77–1.13 ms/L`) with `10/10` pre-captured decode traces (`1.7 MiB`) and `5`-GPU uncached `1,000 MHz` HBM2 `mclk` lock -> live `M=2048` `moe` dropped `~2,800 ms -> 1,820 ms` (`mi50_moe` `~2,100 ms -> 781 ms`, `2.6x`), cold `M=2048` prefill reached **`457.6 tok/s`** (`454.4 tok/s` cold `4K`), and live `:3081` decode peaked **`41.23–42.35 tok/s`**.
5. **3-Tier `2,048`-Block `.kvblk` Engine (`mi50/ernie55_kvblk_cache.hpp`, `tools/ernie55_kvblk_verify.cpp`):** Combines Tier-0 VRAM (`109.9M–1,106M tok/s`), Tier-1 `64`-slot Write-Back hugepage `hipHostRegisterPortable` DDR4 LRU (`723.1K tok/s`, `0` hot-path `hipHostMalloc`), and Tier-2 `8+8=16`-worker Dual-NVMe `O_DIRECT` + SSE4.2 `CRC32C` persistence (`703.8K tok/s` live 4-GPU cold restore @ `307K`, `919.5K tok/s` / `8.04 GB/s` @ `262K` verify), alongside `4`-lane VALU DPP `FlashIndexer` (`5.19x` @ `262K`, `0` bit-diffs) and `36`-layer Split-K `HCA` (`6.80x` @ `512K`, `6.59x` @ `1M`).
6. **Wave-10 (`v1.2.0-wave10-1m-turbo`) Seven-Commit Zero-P2P Ladder (`docs/WAVE10_ARCHITECTURE_AND_AUDIT.md`):** Adds eager P150A Tensile warmup + zero-copy SHM DMA (`253.7 MiB` L1, `kBatchSlots = 64`), true full-duplex 2-MB host-SDMA FP16 AllReduce + 1-pass `mHC` pre-GEMM, `TP=4` 16-head sparse/dense `DSA` with unique-KV FP16/u16 host-SDMA compaction (`10x` faster sparse `mqa`), prefill-only symmetric offload of all 13 shadow experts to P150A slack + HCA 2-MB `ev_mb_d2c` D2H overlap, inverted outer `K_tile = 16,384` single-pass `O(L)` `FlashIndexer` (`260.2 tok/s` @ `514K`, `249.9 tok/s` @ `524K`, `229.3 tok/s` @ `624.6K`), and Commit 6 `DSpark` launch fusion (`k_dspark_markov_bias_multimax_f32` `0.000e+00` diff @ `644 GB/s`, single-DMA single-sync packed sparse softmax `10 -> 5` host syncs/block, async `commit_dspark_context_kv`, and zero-copy `Dfly2SharedSparsityView` header hook).

## 5. Reproducible Pure-C++ Build, Self-Quantization & Gate Verification

Zero Python scripts or external ML frameworks are required at build, quantization, or runtime.

### Build All Binaries & Audit `gfx906` ISA

```bash
make -j"$(nproc)" all
make audit-gfx906
make verify-unit
```

### Option A: Download Pre-Quantized Unsplit `WHT128-HLUT4` Shards (`139.22 GiB` MoE + Lossless `BF16` Spine)

```bash
hf download kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4 --local-dir ./weights_hlut4
sha256sum -c ./weights_hlut4/SHA256SUMS.txt
```

### Option B: Quantize Directly from Upstream `baidu/ERNIE-5.5-Flash-Preview-BF16`

```bash
./bin/ernie55_build_weights_gpu \
  --bf16-dir ./upstream_bf16 \
  --out-dir ./weights_hlut4 \
  --cal-tokens 524288 \
  --ldlq-damp 0.01 \
  --layers 0..45
```

### Launch Blackhole `P150A` Worker, `:8052` OpenAI Server & Official Gates (`GATE-1..GATE-4`)

```bash
./bin/p150_ernie55_worker --weights ./weights_hlut4 &
./bin/ernie55_verify_gates --weights ./weights_hlut4 --gate 1,2,4
./scripts/start_ernie55_fe.sh --weights ./weights_hlut4 --port 8052 --max-kv 1048576
./bin/ernie55_gate3_live --endpoint http://localhost:8052 --gui http://localhost:3081
```

### Replay or Audit the Sanitized `626.2K`, `198.0K` & `158.6K` Autonomous Co-Engineering Sessions

* **`artifacts/dsh-session-fa82666c-626k-sanitized.zip`** (`140,513` JSONL events, `206` steps, `222` tool calls, **`626,187` peak live context tokens**, `211,461` output tokens, `0` compactions, `0/0` PCIe AER, `SHA-256: 0bd41e7b72cdc4d046eaf03e8462419ebe350543f97d42595ad2f2e6beb41867`) + readable Markdown transcript **`docs/SESSION_FA82666C_626K_REPLAY.md`**: single unbroken `:3081` Wave-10 co-engineering campaign across Commits 1 through 6, surviving `180K -> 241K -> 280K -> 307K -> 319K` cold Dual-NVMe `.kvblk` swaps (`703.8K tok/s`), authoring C++17/`gfx906` HIP upgrades and `1M`-context audit tests (`11/11` PASS), and sustaining `229.3 tok/s` prefill and `22.60–26.55 tok/s` `DSpark` decode at `625K` live context (`10/10` Zero-PII PASS).

* **`artifacts/dsh-session-6b2a64a4-198k-sanitized.zip`** (`67,690` JSONL events, `116` steps, `129` tool calls, `198,049` peak context, `94,832` output tokens, `86.5 minutes`, `SHA-256: c1b7fa4a09444b154edc79927d5aa474d2b119bcc6bf4fe17ec4361ecf89b5c5`) + readable Markdown transcript **`docs/SESSION_6B2A64A4_198K_REPLAY.md`** (rendered via `tools/ernie55_session_to_md.cpp`): unscripted six-figure-context `:3081` co-engineering session where `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` read production `gfx906` HIP kernels, caught out-of-bounds stride and `56 MiB` radix-scratch traps from first principles, audited its own `VramBudget` draft before compiling, and authored `tools/bench_single_layer_gfx906.cpp` under `-Wall -Wextra -Werror`.
* **`artifacts/dsh-session-b4cf2731-158k-sanitized.zip`** (`9,172` JSONL events, `SHA-256: 3414a3d8f2a37af55eea39a531747061048262282024955e50e1872bc0438180`): unbroken `:3081` session where `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` wrote and passed a `49/49` cross-silicon shard/ISA auditor, survived a cold VRAM wipe + `113,307`-token `.kvblk` `O_DIRECT` restore at `621,656 tok/s`, and authored `7/7` zero-scratch inline `v_dot2_f32_f16` `gfx906` HIP kernels.

```bash
./bin/ernie55_kvblk_verify --unit --live http://127.0.0.1:8052
./bin/ernie55_sanitize_verifier --root . --mode github
sha256sum -c SHA256SUMS.txt
```

## 6. License, Upstream Attribution (`NOTICE`) & Co-Authorship

* **License:** Released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (`LICENSE` and `NOTICE` in repository root).
* **Upstream Attribution:** [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16), [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview), and [`baidu/vLLM-ERNIE`](https://github.com/baidu/vLLM-ERNIE) are Copyright Baidu, Inc. and licensed under `Apache-2.0`. All quantized weights in `kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4` are derived strictly from unmodified upstream `Preview-BF16` tensors (`0` refusal-direction edits, `0` fine-tuning).
* **Co-Authors:** **bespoke ontology** (`the kitchen`) and **Google DeepMind / Gemini** (`Gemini Carbon` & `Gemini Neon`) & **Antigravity**.

```bibtex
@software{bespokeontology2026ernie55hlut4,
  title   = {ERNIE-5.5-Flash-Preview WHT128-HLUT4: Higher-Precision BF16-Master W4A16/FP32 + Lossless 1M BF16 KV Inference Engine on 192 GiB Heterogeneous Silicon},
  author  = {{bespoke ontology} and {Gemini (Google DeepMind)}},
  year    = {2026},
  license = {Apache-2.0},
  url     = {https://github.com/bespokeontology/ernie55-flash-preview-wht128-hlut4}
}
```
