# ERNIE-5.5-Flash-Preview (`300B` / `14B` Active) — Higher-Precision `BF16`-Master `WHT128-HLUT4` (`4.125 bpw`, `W4A16/FP32`) + Lossless `BF16` Spine, Indexer & `1M` KV on `192 GiB`

**Pure C++17 / `gfx906` HIP / Tenstorrent Blackhole `TT-Metalium` C++20 inference engine, `FP64` `LDLQ` Walsh-Hadamard `4`-bit lookup quantizer, and silicon verification suite (`0` `.py` files, `0 B` CPU DDR4 expert spill) built directly from the unquantized [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) master checkpoint (`Apache-2.0`).**

Unlike Baidu's default [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) deployment checkpoint—which quantizes the non-expert spine to blockwise `FP8`, runs routed experts in `W4A8` `MXFP4` (`4.25 bpw`), stores the `1M` KV cache in `FP8`, and runs the `DSA` sparse indexer in `4`-bit `MXFP4` with distilled head pruning ([vLLM Day-0 Technical Specification](https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview))—this release preserves **`100%` unquantized `BF16`/`FP32` across the entire non-expert spine, `100%` lossless `BF16` across the `1,048,576`-token KV cache (`0.0000 nats` KV loss), `100%` lossless `BF16`/`FP32` across all `8/8` `DSA` indexer heads (`0` IEEE-754 bit-diffs, `5.19x` faster via `4`-lane VALU DPP), and `W4A16/FP32` `16`-centroid `WHT128-HLUT4` (`4.125 bpw`, `19.3858 dB` mean SQNR, `0.012301 nats` post-header median KLD)** across all `23,040` backbone and `DSpark` routed experts on commodity `HBM2` + spatial `SRAM` without `FP4` tensor-core hardware lock-in.

* **Engine & Kernels Repository:** `bespokeontology/ernie55-flash-preview-wht128-hlut4` (`v1.0.0-preview-hlut4-192g`, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) + `NOTICE`)
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
| **`1,048,576`-Token (`1M`) KV Cache (`37` `128:1` `HCA` + `6` `MLA-DSA`)** | **`FP8` (`8`-bit, `--kv-cache-dtype fp8`)** + optional CPU offload | **`100%` Lossless `BF16` (`16`-bit)** resident in HBM2 (`0 B` DDR4 spill) + striped `O_DIRECT` `.kvblk` (`memcmp == 0`) | **`0.0000 nats` KV quantization loss** across the entire `1,048,576`-token window and across cold NVMe session restores (`792,854 tok/s` @ `262K`) |
| **`DSA` Sparse Indexer (`6` Layers Selecting `Top-2048` Keys out to `1M`)** | **`4`-bit `MXFP4` (`indexer_kv_dtype: mxfp4`)** + `1B`-token distilled head-pruning router (`1.4x` speedup @ `1M`) | **`16`-bit `BF16` `ik_cache` (`4x` bit-width) + `FP32` Exact `8/8`-Head Scoring** via `4`-lane VALU DPP `quad_perm` (`0` dropped heads) | **`0` IEEE-754 bit-diffs**, **`13,056 / 13,056` exact `Top-2048` indices**, and **`5.19x` speedup @ `262K`** (`30.27 -> 5.84 ms`) without approximate head pruning |
| **Runtime & Deployment Stack** | Python `vllm==0.32.0` + `vLLM-ERNIE` plugin (`--trust-remote-code`) on `FP4`-capable datacenter GPUs | **Standalone Pure C++17 / `gfx906` HIP / Blackhole C++20 (`0` `.py` files)** on `192 GiB` commodity `MI50` HBM2 + `P150A` spatial SRAM | Live `:3081` Web GUI parity (`x_flash_progress` v1 counters, `cached_tokens`, `2 s` SSE keepalives, identifier-first `<ernie:tool_call>` streaming) |

## 2. Executive Silicon Scorecard (`SB-M5-P2-192G` Single Workstation)

[`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) pairs a `43`-layer (`280B`) hybrid sparse backbone (`37` `128:1` `HCA` layers + `6` `MLA-DSA` layers + `4`-stream `FP32` `mHC` residual connections + `4`-piece [`MultiMax` / `SegLU`](https://zhouyuxuanyx.github.io/2026/11/05/one-knob-is-not-enough-softmax-temperature-vs-multimax.html) logit transform over `201,216` vocabulary tokens) with a native `3`-layer (`~20B`) `512`-expert MoE `DSpark` parallel speculative drafter (`model.mtp.{0,1,2}`). Compressing all `45` MoE layers (`23,040` routed `2048x2048` experts) into uniform `WHT128-HLUT4` shrinks `562.5 GiB` of `BF16` expert weights to **`139.22 GiB`**, fitting the entire `300B` system plus a **`1,048,576`-token lossless `BF16` KV cache** inside `5x AMD Radeon Instinct MI50` (`128 GiB` HBM2) + `2x Tenstorrent Blackhole P150A` (`64 GiB` GDDR6 + `330 MiB` Tensix L1 SRAM over [`4x 800G` `FABRIC_1D`](https://docs.tenstorrent.com/aibs/blackhole/multi-card-topologies.html)) with **`0 B` CPU DDR4 expert spill**:

| Subsystem / Verification Gate | Verified Silicon Result | Receipt / Artifact |
| :--- | :--- | :--- |
| **Pure C++ / Zero-Scratch `gfx906` ISA** | **`241/241`** kernels (`100` HLUT4 + `141` core; `ScratchSize == 0`, `0` spills, `0` hw trig, `0` `slc` atomics, `1,912` `fdot2` ops, `0` `.py` files, `-Wall -Wextra -Werror`) | `build/ernie55_hlut4_kernels.s`, `build/ernie55_core_kernels.s` |
| **`45`-Layer `WHT128-HLUT4` Pack (`23,040` Experts)** | **`139.22 GiB`** (`3,321,892,864 B` x `45` unsplit shards), **`19.3858 dB`** mean SQNR (`cal_tokens = 524,288`, `ldlq_damp = 0.01`), **`0 B` DDR4 spill** | `artifacts/hlut4_preview_pack_manifest.json` |
| **Empirical Router Placement (`"5MRP"`)** | `318 + 13` Wave-8 shadow-cached hot experts/layer on `5x MI50` (`94.89%` base hit; **`99.33% / 97.85% / 97.29%` on `DSpark` `L43..L45`**) + `68` in `P150A` Tensix L1 (**`98.06%` combined**) + `113` in `P150A` GDDR6 | `router_perm.bin` (`47,112 B`) |
| **`GATE-1` Lossless Operator & `DSpark` Proofs** | `INV-1b` **`64/64` bit-exact** `DSpark == M=1` (`2,040` `HCA`-boundary rollbacks); `INV-1c` `TV(K=5) = 0.009615`; `INV-1d` **`cos = 1.00000000`** (`L = 129, 4097, 65539`); `MI50` vs `P150A` `cos = 0.999990-0.999994` | `artifacts/gate1_ernie55_preview_hlut4_1m.json` |
| **`GATE-2` Disjoint `BF16` Oracle KLD (`0` 8-gram overlap)** | Post-header (`2,304` pos): **`Median KL = 0.012301 nats`**, **`Cos = 0.999066`**, `Top-1 = 85.20%` (All `4,096` pos: `Median KL = 0.039143 nats`); Router `EAR@10 = 94.72%` | `artifacts/gate2_ernie55_preview_hlut4_1m.json` |
| **`GATE-4` Wave-8 Single-Stream Decode & Context Scaling** | Wave-8 mixed `DSpark` (`K=5`) decode: **`34.58 tok/s` greedy** (`3.52 tok/step`, `64/64` exact, **`+24.7%`** vs Wave-5b `27.74`) and **`32.20 tok/s` probabilistic** (`T=1.0, p=0.95`, `3.17 tok/step`, **`+31.8%`** vs Wave-5b `24.43`); Wave-5b `GATE-4` ladder (`2K/4K/64K`): `29.28 / 30.39 / 31.41 tok/s` decode, `376.1 / 359.0 / 311.4 tok/s` cold prefill (full Wave-8 `262K` receipt updating next) | `artifacts/gate4_ernie55_preview_hlut4_1m.json` |
| **Dual-NVMe `O_DIRECT` + `CRC32C` `.kvblk` Persistence** | **`792,854 tok/s` E2E (`330.7 ms`, `7.02 GB/s`)** @ `262,191` tokens (`2.16 GiB`, `memcmp == 0`); **`621,656 tok/s` (`182.3 ms`, `5.60 GB/s`)** @ `113,307` tokens (`1.02 GiB`) | `artifacts/gate4_ernie55_preview_hlut4_1m.json`, `artifacts/gate6_long_horizon_1m_campaign.json` |
| **Sanitized Long-Horizon Autonomous GUI Trace** | **`158.6K` tokens, `9,172` JSONL events, `95` steps, `112` tool results** (`49/49` Phase 1 auditor + Step 51 cold `113.3K` `.kvblk` restore + `7/7` zero-scratch `fdot2` HIP kernels, `10/10` Zero-PII PASS); standalone `:3081` `GATE-3` harness (`tools/ernie55_gate3_live.cpp`) included for post-launch receipt updates | `artifacts/dsh-session-b4cf2731-158k-sanitized.zip` (`SHA-256: 3414a3d8f2a37af55eea39a531747061048262282024955e50e1872bc0438180`) |

![ERNIE-5.5-Flash-Preview WHT128-HLUT4 1M Telemetry](./artifacts/ernie55_preview_hlut4_1m_telemetry.png)

## 3. Heterogeneous `192 GiB` Placement & `1M` Lossless KV Budget

All `5x Radeon Instinct MI50` cards (`3x 32 GiB` + `2x 16 GiB` = `128 GiB` HBM2 at `~5.1 TB/s` aggregate bandwidth, mapped with thermal ordering `HIP_VISIBLE_DEVICES=0,2,3,4,1` across the dual-`PEX88048` switched PCIe backplane) and `2x Tenstorrent Blackhole P150A` cards (`64 GiB` GDDR6 + `330 MiB` Tensix L1 SRAM linked via `4x 800G` = `3.2 Tbps` `FABRIC_1D`) operate with **`0 B` CPU DDR4 expert spill**:

| Accelerator / Pool | Physical Capacity | Assigned Tensors & `1,048,576`-Token Lossless `BF16` KV Allocation |
| :--- | :--- | :--- |
| **`TP=4` `MI50` Spine + Hot MoE (`3x 32 GiB` + `1x 16 GiB`)** | `112 GiB` HBM2 | `TP=4` unquantized `FP16`/`BF16`/`FP32` non-expert spine (`L00-L42` `HCA`/`MLA-DSA`, shared experts, `mHC`, `FlashIndexer`, `DSpark` spine, `MultiMax`) + **`264` hot `WHT128-HLUT4` experts/layer** (`{66, 68, 92, 38}` across `45` layers) + **`1,048,576`-token lossless `BF16` KV pool** (`37` `128:1` `HCA` block/SWA/sink caches + `6` `MLA-DSA` `576`-dim latent & `128`-dim `BF16` `ik_cache`) |
| **Dedicated `MI50` MoE Worker (`1x 16 GiB`)** | `16 GiB` HBM2 | **`54` hot `WHT128-HLUT4` experts/layer** across all `45` MoE layers (`2,430` experts = `14.68 GiB`), bringing total `MI50` HBM2 coverage to **`318 / 512` experts/layer (`94.89%` backbone hit; `97.29%-99.33%` `DSpark` hit)** |
| **`2x Tenstorrent Blackhole P150A` (`FABRIC_1D`)** | `64 GiB` GDDR6 + `330 MiB` L1 | **`194` remaining `WHT128-HLUT4` experts/layer** across `45` layers (`8,730` experts = `52.75 GiB` total), with the top `68` tail experts pinned in on-die Tensix L1 SRAM (**`98.06%` combined `MI50 + P150A L1` hit**) and `SFPMULI + SFPLUTFP32` native `SiTU-GLU(4.0, 25.0)` evaluation (`cos = 0.999990-0.999994` vs `MI50`) |
| **Dual Striped PCIe NVMe (`nvme0n1` + `nvme1n1`)** | `O_DIRECT` `.kvblk` | Lossless `BF16` session persistence (`CRC32C` hardware-verified per stripe): **`792,854 tok/s` (`7.02 GB/s`)** cold VRAM restore at `262K` tokens (`memcmp == 0`) |

## 4. Silicon Kernel Architecture & Live `:3081` Web GUI Parity (`Waves 5b -> 8`)

Every device kernel in `mi50/ernie55_hlut4_kernels.hip` and `mi50/ernie55_core_kernels.hip` is compiled for `gfx906` (`-Wall -Wextra -Werror`) and mechanically audited via `make audit-gfx906` for `ScratchSize == 0`, `0` VGPR/SGPR spills, `0` hardware trig instructions, and `0` `slc` PCIe RMW atomics:

1. **`32`-Bank `s_lut256` LDS + `v_dot2_f32_f16` `W4A16/FP32` `WHT128-HLUT4` MoE Kernels:** Pre-expands `16`-centroid `FP16` codebooks into `256`-entry byte-pair LUTs across `32` conflict-free LDS banks, fusing fast `128`-point Walsh-Hadamard `FP16` activation rotation with packed `fdot2` `FP32` accumulation and exact `SiTU-GLU(4.0, 25.0)` gating (`0` `FP8` activation quantization loss).
2. **`4`-Lane VALU DPP Lossless `BF16` `FlashIndexer` (`k_dsa_indexer_score_topk8_dpp4`, `0` `ds_bpermute`):** Rather than quantizing `ik_cache` to `MXFP4` and pruning heads via a distilled router (`1.4x` at `1M` in `vLLM-ERNIE`), maps `1` `128`-dim `BF16` history token per `4`-lane hardware DPP quad (`16` tokens/wavefront, `64` tokens/block). Folds `d_warp_sum64` steps `32, 16, 8, 4` inside lane registers and steps `2, 1` via two single-cycle VALU DPP `quad_perm` ops (`0x4E`, `0xB1`), cutting `75.5M` `ds_bpermute_b32` wave shuffles to `0` at `262K` with **`0` IEEE-754 bit-diffs** and **`13,056/13,056` exact TopK indices**:
   * `65K (M=6, 6 DSA layers)`: `7.65 ms -> 1.82 ms` (**`4.21x`**)
   * `262K (M=6, 6 DSA layers)`: `30.27 ms -> 5.84 ms` (**`5.19x`**)
3. **`32`-Block Parallel Zero-Skip `DSA` `Top-2048` Selector & `B=1` `MQA-64` Decode:** Exploits exact `ReLU` sparsity in `FlashIndexer` block scores across `32` cooperative thread blocks (`15.07 ms -> 3.35 ms` across `6` layers at `262K`, **`4.50x`**) and routes `M <= 8` decode through latent `MQA-64` (`64x` less HBM2 gather while preserving `INV-1b` `64/64` exactness).
4. **`36`-Layer Split-K `4`-Head `HCA` Decode (`launch_hca_attn_c128_swa_sink_splitk`):** Parallelizes lossless `BF16` `128:1` compressed block history + `128`-token local sliding window + attention sink across Split-K partitions (`cos >= 0.99999999`, `M=1 == M=6` **`0` bit-diffs**):
   * `75K (M=6, 36L)`: `17.1 ms -> 5.9 ms` (`2.90x`)
   * `262K (M=6, 36L)`: `64.2 ms -> 14.0 ms` (`4.57x`)
   * `512K (M=6, 36L)`: `126.5 ms -> 18.6 ms` (**`6.80x`**)
   * `1M (M=6, 36L)`: `244.8 ms -> 37.2 ms` (**`6.59x`**, **`20.96x` @ `M=1`**)
5. **Wave-8 Full-Stack Runtime & Full `:3081` Live-UX Parity:** Fuses `r8_ws` BF16-in/FP16-weight `fdot2` GEMVs (`k_bf16x_f16w_gemv_r8_ws`, `k_hca_grouped_wo_a_lrga_bf16_f16_ws`), `count == 2` zero-barrier direct-L1 MoE fast paths with zero-LDS `d_reduce4_dpp` (`0xB1`/`0x4E`) reductions, self-`D2D` SDMA `TP=4` AllGather, widened `sub=64` `FlashIndexer` `Top-2048` selection, and `13` MI50 HBM2 shadow-cached top `P150A` GDDR6 experts (`[352..365)` on `L01..L42`)—reaching **`34.58 tok/s` greedy (`+24.7%`)** and **`32.20 tok/s` probabilistic (`+31.8%`)** mixed decode—paired with `2 s` [WHATWG SSE](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation) `: keepalive` heartbeats, `MSG_PEEK` instant client-disconnect aborts, `ernie55_live_progress.hpp` (`x_flash_progress` v1 live prefill/decode counters + `usage.prompt_tokens_details.cached_tokens`), and schema-safe identifier-first (`file_path` / `description` first) `<ernie:tool_call>` argument streaming (`13/13` fuzzed parser tests PASS).

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

### Replay or Audit the Sanitized `158.6K`-Token Autonomous Engineering Session

`artifacts/dsh-session-b4cf2731-158k-sanitized.zip` (`9,172` JSONL events, `SHA-256: 3414a3d8f2a37af55eea39a531747061048262282024955e50e1872bc0438180`) records an unbroken `:3081` session where `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` autonomously wrote and passed a `49/49` cross-silicon shard/ISA auditor, survived a cold VRAM wipe + `113,307`-token `.kvblk` `O_DIRECT` restore at `621,656 tok/s`, and authored `7/7` zero-scratch inline `v_dot2_f32_f16` `gfx906` HIP kernels:

```bash
./bin/ernie55_sanitize_verifier --root . --mode github
sha256sum -c SHA256SUMS.txt
```

## 6. License, Upstream Attribution (`NOTICE`) & Co-Authorship

* **License:** Released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (`LICENSE` and `NOTICE` in repository root).
* **Upstream Attribution:** [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16), [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview), and [`baidu/vLLM-ERNIE`](https://github.com/baidu/vLLM-ERNIE) are Copyright Baidu, Inc. and licensed under `Apache-2.0`. All quantized weights in `kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4` are derived strictly from unmodified upstream `Preview-BF16` tensors (`0` refusal-direction edits, `0` fine-tuning).
* **Co-Authors:** **bespoke ontology** (`the kitchen`) and **Google DeepMind / Gemini** (Gemini Carbon & Gemini Neon).

```bibtex
@software{bespokeontology2026ernie55hlut4,
  title   = {ERNIE-5.5-Flash-Preview WHT128-HLUT4: Higher-Precision BF16-Master W4A16/FP32 + Lossless 1M BF16 KV Inference Engine on 192 GiB Heterogeneous Silicon},
  author  = {{bespoke ontology} and {Gemini (Google DeepMind)}},
  year    = {2026},
  license = {Apache-2.0},
  url     = {https://github.com/bespokeontology/ernie55-flash-preview-wht128-hlut4}
}
```
