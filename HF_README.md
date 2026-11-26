---
license: apache-2.0
pipeline_tag: text-generation
library_name: bespokeontology-hlut4
# release: v1.2.0-wave10-1m-turbo (Wave-10 zero-P2P ladder, 302/302 zero-scratch gfx906 kernels)
base_model: baidu/ERNIE-5.5-Flash-Preview
base_model_relation: quantized
language:
  - en
  - zh
tags:
  - ernie5_5_moe
  - ernie
  - ernie-5.5-flash
  - moe
  - dspark
  - quantized
  - 4-bit
  - wht128
  - hlut4
  - rarebal-ldlq
  - gfx906
  - rocm
  - tenstorrent
  - blackhole
  - bespokeontology
---

# ERNIE-5.5-Flash-Preview-WHT128-HLUT4 (`300B` Total / `14B` Active — Higher-Precision `BF16`-Master `W4A16/FP32` + Lossless `BF16` Spine, Indexer & `1M` KV)

**`100%` un-abliterated, SHA-256-verified `WHT128-HLUT4` (`4.125 bpw`, `FP64` `LDLQ`) quantization — quantized directly in one pass from the lossless `600 GB` [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) master checkpoint (`Apache-2.0`, zero `FP4`/`FP8` double-quantization) — covering all `42` backbone MoE layers AND all `3` native `DSpark` parallel draft MoE layers (`model.mtp.{0,1,2}` = `23,040` routed experts in `139.22 GiB`) while preserving `100%` unquantized `BF16`/`FP32` across the entire non-expert spine, `DSA` sparse indexer, and `1,048,576`-token KV cache.**

* **Pure-C++17 / `gfx906` HIP / Tenstorrent Blackhole Engine & Kernels:** [`bespokeontology/ernie55-flash-preview-wht128-hlut4`](https://github.com/bespokeontology/ernie55-flash-preview-wht128-hlut4) (`v1.2.0-wave10-1m-turbo`, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0))
* **Master Source Checkpoint:** [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) (`~600 GB` across `74` `BF16` shards, `201,216` vocab, `1,048,576` context length, `Apache-2.0`)
* **Upstream Official Quantized Reference:** [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (`MXFP4` `W4A8` + blockwise `FP8` spine + `FP8` KV + `MXFP4` `DSA` indexer) & [vLLM Day-0 ERNIE-5.5-Flash-Preview Technical Post](https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview)
* **Companion Unsplit `WHT128` Releases:** [`kapitoshkina-ai/DeepSeek-V4.1-Flash-1017-WHT128-HLUT3`](https://huggingface.co/kapitoshkina-ai/DeepSeek-V4.1-Flash-1017-WHT128-HLUT3) and [`kapitoshkina-ai/Hy4-flash-WHT128-HLUT4`](https://huggingface.co/kapitoshkina-ai/Hy4-flash-WHT128-HLUT4)

## 1. Why This Checkpoint Holds Higher Precision Than the Official Default Quant (`baidu/ERNIE-5.5-Flash-Preview`)

Baidu's default [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) repository targets `FP4`/`FP8` datacenter GPUs: its routed experts are stored in `MXFP4` (`4.25 bpw`) and executed as `W4A8` (QAT-aligned during post-training), all non-expert spine weights are quantized to blockwise `FP8`, the `1M` KV cache is quantized to `FP8`, and the `DSA` sparse indexer runs in `4`-bit `MXFP4` with a distilled head-pruning router ([vLLM Day-0 Specification](https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview)).

Because `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` is quantized directly from the unquantized [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) master weights, it maintains higher numerical precision across every surrounding subsystem while packing routed experts into `0.125 bpw` *fewer* bits (`4.125` vs. `4.25 bpw`) without requiring `Blackwell`-class `FP4` tensor cores:

| Tensor Group / Subsystem | Official Default [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) | `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` (This Checkpoint) | Concrete Numerical Difference |
| :--- | :--- | :--- | :--- |
| **Target Non-Expert Spine (`43` `HCA`/`MLA-DSA` Layers, Shared Experts, Routers, `DSpark` Spine, Target LM Head)** | Blockwise `FP8` (`8`-bit `E4M3`, `128x128` tiles) | **`100%` Unquantized `BF16` / `FP32`** on disk (`16/32`-bit target runtime; `FP8 E4M3FN` used strictly on the `DSpark` draft proposal head, `100%` verified against the `16`-bit target LM head) | Exact `BF16` attention, shared-expert, router, and `DSpark` spine weights + `FP32` `4`-stream `mHC` & `MultiMax` (`201,216` vocab) |
| **Expert Activation & Accumulator Precision** | **`W4A8`** (`8`-bit dynamic `FP8` input activations) | **`W4A16` + `FP32` Accumulator** (`16`-bit `FP16` Walsh-Hadamard rotated activations) | Zero `8`-bit activation quantization floor entering all `23,040` routed `SiTU-GLU(4.0, 25.0)` experts |
| **Routed Expert Weight Codebook (`23,040` Experts)** | **`4.25 bpw` `MXFP4` (`E2M1` + `E8M0`)**: `15` unique fixed levels (`±0.0` collision) + power-of-two `E8M0` scale per `32` weights (QAT-trained) | **`4.125 bpw` `WHT128-HLUT4` (`FP64` `LDLQ`)**: `128`-point Hadamard rotation + **`16` distinct learned `FP16` Lloyd-Max centroids** + Cholesky error feedback (`524,288` cal tokens) | Smaller footprint (`139.22 GiB`), **`19.3858 dB` 45-layer mean SQNR**, **`0.012301 nats` post-header median KLD (`0.999066` cosine)** vs `BF16`, runs on standard `HBM2` + spatial `SRAM` |
| **`1,048,576`-Token (`1M`) KV Cache (`37` `HCA` + `6` `MLA-DSA`)** | **`FP8` (`8`-bit, `--kv-cache-dtype fp8`)** | **`100%` Lossless `BF16` (`16`-bit)** (`0 B` DDR4 spill; `memcmp == 0` NVMe `.kvblk`) | **`0.0000 nats` KV quantization loss** across the full `1M` context window |
| **`DSA` Sparse Indexer (`6` Layers, `Top-2048` Selection)** | **`4`-bit `MXFP4` (`indexer_kv_dtype: mxfp4`)** + distilled head-pruning router (`1.4x` @ `1M`) | **`16`-bit `BF16` `ik_cache` + `FP32` `8/8`-Head Exact Scoring** (`4`-lane VALU DPP, `0` `ds_bpermute`) | **`0` IEEE-754 bit-diffs**, **`13,056 / 13,056` exact TopK**, and **`5.19x` speedup @ `262K`** (`0` dropped heads) |

## 2. Cryptographic Provenance & Unsplit Byte Geometry (`100%` Un-Abliterated)

Every file in this repository is derived strictly from unmodified [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) (`0` refusal-direction projections, `0` LoRA merges, `0` fine-tuning). All `45` MoE layer shards are published **unsplit** (`SHA256SUMS.txt` covers `100%` of files; zero `.part*` reassembly):

| Path | Count / Exact Size | Format & Contents |
| :--- | :--- | :--- |
| `experts_hlut4/hlut4_layer_01.bin` .. `hlut4_layer_45.bin` | `45` files x **`3,321,892,864 B`** (`139.22 GiB` total) | `4,096 B` aligned header + `512` routed experts x `3` projections (`gate`, `up`, `down`, `2048x2048` each) x `4,096` blocks of `1,024` weights x `528 B` (`4.125 bpw`: `512 B` packed `4`-bit LUT indices + `16 B` `FP16` codebook/scale metadata per `1,024`-weight subblock) |
| `packed_spine/spine_shared.safetensors`, `spine_layer_00..42.safetensors`, `spine_dspark.safetensors` | `45` lossless `BF16` / `FP32` files | Complete unquantized `L00-L42` `HCA` + `MLA-DSA` + `FlashIndexer` + shared experts + `mHC` + `spine_dspark.safetensors` (`91` `model.mtp.{0,1,2}` spine tensors) + `MultiMax` |
| `router_perm.bin` | **`47,112 B`** (`"5MRP"` magic) | Empirical `524,288`-token activation-frequency permutation (`46` layer slots x `512` `uint16`): places top `318` hot experts/layer on `5x MI50` HBM2 (`94.89%` mean hit; **`99.33% / 97.85% / 97.29%` on `DSpark` `L43/L44/L45`**) and `68` in `Blackhole P150A` Tensix L1 (**`98.06%` combined**) |
| `meta/hlut4_preview_pack_manifest.json` | Verified JSON receipt | Per-layer `SQNR` (`mean_sqnr_db = 19.3858 dB`), hit-rate tables, and shard digests |
| `config.json`, `tokenizer.json`, `tokenizer_config.json` | Upstream `Preview-BF16` | Unmodified Baidu tokenizer & model configuration (`201,216` vocab, `max_position_embeddings = 1048576`) |
| `SHA256SUMS.txt`, `LICENSE`, `NOTICE`, `CITATION.cff` | Integrity & Legal | Complete SHA-256 manifest, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), and Baidu + `bespoke ontology` attribution |

## 3. Verified Numerical Quality & Silicon Performance (`SB-M5-P2-192G`)

Evaluated on a single heterogeneous workstation (`5x AMD Radeon Instinct MI50` `128 GiB` HBM2 + `2x Tenstorrent Blackhole P150A` `64 GiB` GDDR6 over [`4x 800G` `FABRIC_1D`](https://docs.tenstorrent.com/aibs/blackhole/multi-card-topologies.html), `0 B` CPU DDR4 expert spill, full `1,048,576` lossless `BF16` KV cache):

| Verification Gate / Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **`45`-Layer Mean Expert `SQNR`** | **`19.3858 dB`** | `L01 = 18.3110 dB`, `L07 = 19.1872 dB`, `L43 (DSpark) = 19.3396 dB`; `MI50` vs `P150A` cosine `0.999990-0.999994` |
| **Disjoint `GATE-2` `BF16` Oracle KLD** | **`0.012301 nats`** (post-header median) | **`0.999066` mean logit cosine**, `85.20%` Top-1, `94.72%` Router `EAR@10` across `64` held-out probes (`0` 8-gram overlap vs `524,288` cal tokens; full-sequence median `0.039143 nats`) |
| **`INV-1b` `DSpark` Greedy Exactness** | **`64 / 64` bit-exact (`100%`)** | `DSpark` `K=2..7` == `M=1` target greedy token-for-token across `2,040` rollbacks straddling `128`-token `HCA` block boundaries |
| **`INV-1c` Probabilistic `DSpark` TV** | **`TV(K=5) = 0.00961`** (`< 0.01`) | `TV(K=1) = 0.00858`, `0` NaNs across `4`-piece [`MultiMax`](https://zhouyuxuanyx.github.io/2026/11/05/one-knob-is-not-enough-softmax-temperature-vs-multimax.html) residual resampling |
| **`INV-1d` Non-128 Chunked & Warm Prefix** | **`cos = 1.00000000`** | Exact parity across odd non-128 lengths (`L = 129, 4097, 65539`; `L = 262,147` `cos(warm) = 1.00000000`, `cos(137) = 0.99973877`, exact Top-1) and `FlashIndexer` (`cos = 0.998859`, exact argmax) |
| **Long-Context `DSA` + `HCA` Decode Kernels** | **`5.19x` (`DSA` `262K`) / `6.80x` (`HCA` `512K`)** | `4`-lane VALU DPP `FlashIndexer` (`0` `ds_bpermute`, `0` bit-diffs, `5.84 ms` across `6L` @ `262K`) + `36L` Split-K `HCA` (`18.6 ms` @ `512K`, `37.2 ms` @ `1M`, `6.59x`) |
| **Wave-10 (`v1.2.0-wave10-1m-turbo`) Zero-P2P Scaling & Dual-NVMe `.kvblk`** | **`460.2 -> 235.5 tok/s` prefill (`<=64K -> 626K`) / `35.06 -> 26.55 tok/s` live `DSpark` decode (`330K -> 625K`) / `703.8K tok/s` live cold `.kvblk` restore** | `302/302` zero-scratch `gfx906` kernels (`0` BAR1 P2P, `0 B` net new GPU VRAM, `0/0` PCIe AER across `69` devices, `11/11` Gate-0 unit suites): **`260.2 tok/s` @ `514K`**, **`249.9 tok/s` @ exact `524,288`**, **`229.3 tok/s` @ `624.6K`** (beating Wave-8's `262K` `228.5 tok/s` at `2.38x` context); Commit 6 fused `DSpark` (`32.25–35.06 tok/s` @ `330K`, `27.01 tok/s` @ `488K`, `22.60–26.55 tok/s` @ `624K–625K`) + zero-copy `Dfly2SharedSparsityView`; WB hugepage `8+8=16`-worker Dual-NVMe `O_DIRECT` `.kvblk` (`703.8K tok/s` live 4-GPU cold restore @ `307K`, `919.5K tok/s` / `8.04 GB/s` @ `262K` verify, `memcmp == 0`) |
| **Canonical Autonomous `:3081` Traces (`626.2K` + `198.0K` + `158.6K`)** | **`626.2K`-token `session-fa82666c`** (`140,513` events, `206` steps, `222` tools, `211,461` output tokens, `0` compactions) + **`198.0K` `session-6b2a64a4`** + **`158.6K` `session-b4cf2731`** | Single unbroken Wave-10 co-engineering campaign across Commits 1–6 (`SHA-256: 0bd41e7b72cdc4d046eaf03e8462419ebe350543f97d42595ad2f2e6beb41867`, `10/10` Zero-PII PASS) + Wave-9 `198K`/`158K` traces |

![ERNIE-5.5-Flash-Preview WHT128-HLUT4 1M Telemetry](./ernie55_preview_hlut4_1m_telemetry.png)

## 4. Quickstart & Roll-Your-Own `WHT128-HLUT4` Quantization (Pure C++17)

Clone [`bespokeontology/ernie55-flash-preview-wht128-hlut4`](https://github.com/bespokeontology/ernie55-flash-preview-wht128-hlut4) (`0` `.py` files required):

```bash
# 1. Download unsplit weights and verify SHA-256 integrity:
hf download kapitoshkina-ai/ERNIE-5.5-Flash-Preview-WHT128-HLUT4 --local-dir ./weights_hlut4
sha256sum -c ./weights_hlut4/SHA256SUMS.txt

# 2. Build engine & verify lossless gates + start :8052 OpenAI-compatible server:
git clone https://github.com/bespokeontology/ernie55-flash-preview-wht128-hlut4.git
cd ernie55-flash-preview-wht128-hlut4
make -j"$(nproc)" all && make audit-gfx906
./bin/p150_ernie55_worker --weights ../weights_hlut4 &
./scripts/start_ernie55_fe.sh --weights ../weights_hlut4 --port 8052 --max-kv 1048576
```

To quantize [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16) from scratch on your own hardware or adapt expert placement to a different GPU/ASIC topology:

```bash
./bin/ernie55_build_weights_gpu \
  --bf16-dir ./upstream_bf16 \
  --out-dir ./custom_hlut4 \
  --cal-tokens 524288 \
  --ldlq-damp 0.01 \
  --layers 0..45
./bin/ernie55_verify_gates --weights ./custom_hlut4 --gate 1,2
```

## 5. License, `NOTICE` & Co-Authorship

* **License:** `Apache-2.0` (matching upstream [`baidu/ERNIE-5.5-Flash-Preview-BF16`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16)).
* **Upstream Attribution:** Base model weights, configuration, and tokenizer are Copyright Baidu, Inc., licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* **Co-Authors:** **bespoke ontology** (`the kitchen`) and **Google DeepMind / Gemini** (`Gemini Carbon` & `Gemini Neon`) & **Antigravity**.
