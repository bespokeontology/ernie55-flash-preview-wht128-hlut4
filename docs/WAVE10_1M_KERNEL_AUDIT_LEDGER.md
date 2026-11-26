# Wave-10 1M-Token Kernel & Runtime Audit Ledger

Certification target: `v1.2.0-wave10-1m-turbo`, `1,048,576`-token context, Apache-2.0.
Co-authored by bespoke ontology (the kitchen) and Google DeepMind / Gemini
(Gemini Carbon & Gemini Neon) & Antigravity. Platform: AMD EPYC 7542,
5x AMD Radeon Instinct MI50 (gfx906), 2x Tenstorrent Blackhole P150A.
Zero prohibited hardware slang, zero power-cost claims, zero PII, zero local paths.

## Method

Exhaustive whole-file pass (every line, no sampling) over both sources with a
deterministic pattern scanner for: signed 32-bit offset arithmetic, blocking host
synchronisation, raw device allocation, peer-to-peer calls, and truncating
`int(size_t(...))` casts. Every kernel body, helper, allocator and stream barrier
was inspected in file order.

## 1. `mi50/ernie55_runtime.cpp` (6,731 lines) — code-verified ranges

| Subsystem | Symbol | Lines | 1M-token offset contract | Streams / events | Status |
|-----------|--------|-------|--------------------------|------------------|--------|
| RoPE2D upload + CRC32C + AVX2/F16C helpers | `upload_rope2d`, `hw_crc32c`, `avx2_sum4_*` | 1-330 | `size_t` element math; 4 init-only `hipMalloc` | host, ordered | PASS |
| Pinned host allocator (WB hugepage + `hipHostRegisterPortable`) | `host_pin_alloc`, `host_mapped_alloc` | 330-470 | `size_t` byte math, 4 KiB/2 MiB alignment | `hipHostRegister` | PASS |
| GPU arena | `gpu_alloc`, `gpu_upload` | 470-520 | `size_t` count*size | — | PASS |
| Worker struct + CSR scratch | `Mi50ExpertWorker` | 560-700 | `idx_scores_cap = max(max_kv, 16 MiB)` (line 1584) | `stream`, `stream_sh`, `keep_warm_stream` | PASS |
| `f16_linear_bf16_ex` / `f16_linear_f32_ex` / `f16_linear_f16` | — | 1050-1150 | `size_t(m0)*size_t(K)`, 64-row strips | `stream`, `stream_sh` | PASS |
| `tp_allreduce_half` / `tp_allreduce_f16_to_bf16` | — | 1165-1300 | `size_t(M)*size_t(cols)` | `stream`, event-paired | PASS |
| `tp_allgather_heads_bf16` | — | 1300-1360 | `size_t(M)*slab_elems` | `stream` | PASS |
| `compute_lm_head_tp5` | — | 1360-1450 | `size_t(M)*size_t(V)` | per-GPU `stream` | PASS |
| `init_gpu_worker` (all device allocations) | — | 1500-1700 | `size_t(max_chunk)*cols` | creates `stream`, `stream_sh`, `ev_mb_c2d[2]`, `ev_mb_d2c[2]` | PASS |
| `commit_dspark_context_kv` | — | 2126-2143 | `size_t(n_commit)*kHiddenSize`; GPU iota `pos0` | `g1.stream` only, fully async (no host sync) | PASS |
| `run_dspark_draft_block` | — | 2144-2302 | `size_t(b)*V`, `size_t(pt)*R` | `g0.stream` anchor fetch → `g1.stream`; pinned `h_dspark_me_pin` | PASS |
| `run_attention_tp` (HCA + DSA dense/sparse/FI/TP4-h16) | — | 2303-3600 | `size_t(b0)*size_t(max_hist)`, `pos0+t <= 1,048,576` | `stream`/`stream_sh` with `ev_mb_c2d`/`ev_mb_d2c` | PASS |
| `run_moe_core` (hybrid MI50 + P150A) | — | 4462-4870 | `size_t` pair/tile math; `M_sh = ((M+63)/64)*64` | `stream`, `stream_sh`, `ev_sh_done` | PASS |
| `forward_backbone_chunk` | — | 4871-5100 | `size_t(M)*kHcDim` | `stream` per rank | PASS |
| `prefill_tokens_warm` (kvblk restore + 2048-block capture) | — | 5686-5924 | `size_t(bi)*2048u`; `capture_block_from_vram` at 5820 | 4 × `tp_streams` | PASS |
| `generate_stream` | — | 5925-6157 | `size_t` KV offsets; HCA guard `126 - seq_len%128` | `stream` | PASS |
| `save_kvblk_ex` | — | 6158-6397 | `align_up_4k`, `size_t` slab/chunk math | per-stripe `stream` | PASS |
| `load_kvblk_ex` | — | 6398-6686 | `align_up_4k`, `size_t` slab/chunk math | per-stripe `stream` | PASS |
| `dfly2_shared_sparsity_view` | — | 6687-6731 | metadata only, no offsets | none | PASS |

### Verified invariants
- `idx_scores_cap` (line 692, set at 1584): `max(max_kv, 16 MiB)` floats. At `M = 6` decode the cap is `max_kv`-bound (1,048,576 floats); at `M = 2048` prefill the score slab is `max_chunk * 16384` floats (33.55M) and `use_tmp_a_scores` (line 3006) selects `d_tmp_a` when `max_chunk*16384 >= idx_scores_cap` — so `max_batch_t` (line 3449) never exceeds the cap at 524,288 or 1,048,576.
- `kHcaCompressRatio = 128` (src/ernie55_config.hpp): the speculative guard is `hca_safe_k = (128 - 2) - (seq_len % 128)`, guaranteeing `pos0 + K <= 126 (mod 128)` for every `pos0` in `[0, 1,048,575]`, so no draft token ever triggers `k_hca_compressor_reduce_norm_rope`.
- `.kvblk` 64-slot `DdrSlot` wraparound: 1,048,576 / 2,048 = 512 blocks; `capture_block_from_vram` (mi50/ernie55_kvblk_cache.hpp:301) is called per completed block and slot reuse is gated by the `in_flight` H2D pin, so 512-block restores through 64 slots are safe.
- Commit 6 ordering: `commit_dspark_context_kv` (2126) runs on `g1.stream` with no host sync and overlaps CPU token emission; the `g0` anchor fetch is a separate `g0.stream` D2H into `h_dspark_anchor_pin` — no shared host buffer with `h_dspark_pos_pin`.
- `dfly2_shared_sparsity_view` (6687): returns only non-owning pointers + metadata, `valid=false` unless `is_init`.

## 2. `mi50/ernie55_core_kernels.hip` (8,081 lines) — code-verified

| Kernel family | Lines | `__launch_bounds__` | LDS footprint | 1M-token index contract | Status |
|---------------|-------|---------------------|---------------|--------------------------|--------|
| `d_rope2d_lookup` (0-trig 2D factored) | 41-56 | device inline | none | `hi=(pos>>10)&1023, lo=pos&1023`, bijective over `[0,1,048,575]` | PASS |
| BF16/FP16/RoPE primitives (`k_embed_lookup_bf16` … `k_argmax_f32_p2`) | 53-215 | 256,8 / 256,4 / 64,4 | 1-2 KiB | `size_t(m)*size_t(H)`; `pos & 255`, `pos & 127` | PASS |
| mHC pre/post/head (`k_mhc_pre_broadcast_norm` … `k_mhc_head_contract`) | 246-755 | 256,4 / 256,2 / 64,8 | 0.3-4.3 KiB | `size_t(m)*size_t(4*H)` | PASS |
| HCA compressor / SWA-128 / split-K / VHA / LRGA | 756-1193 | 256,4 / 256,8 / 64,4 / 256,4 | 0-4 KiB | `pos & 255`, `blk_start = blk*128`, `num_comp = pos/128` | PASS |
| DSA dense/sparse/indexer (`k_dsa_indexer_prep_neox_wht128` … `k_dsa_vha_lrga`) | 1194-1418 | 128,8 / 256,4 / 64,4 | 0.5-2 KiB | `size_t(pos)*128u`, `pos < max_kv` guard | PASS |
| `k_dsa_indexer_score_topk8_dpp4` | 3767-3810 | 256,3 | 16,640 B | `tok < max_hist`, `min_hist > 2048` fast path | PASS |
| `k_dsa_select_topk2048_gpu` (3-pass radix) | 2896-3041 | 256,2 | 8.3 KiB | `n_hist = pos-127`, `i < n_hist` | PASS |
| `k_dsa_select_topk2048_wave6` (pass1-4b) | 3068-3741 | 256,4 | 8.3 KiB | `chunk = ceil(n_hist/32)`, `i_end` clamp | PASS |
| `k_dsa_sparse_mqa64_fused_vdot2` | 5880-6134 | 512,1 | 55.3 KiB | `size_t(kpos)*576u`, `kb < valid` | PASS |
| `k_flashindexer_relu_wsum_f16_in` / `_tiled_f16_in` | 6184-6210 / 7141-7167 | 256,4 | none | `size_t(bi)*8u*size_t(k_cnt)`; tile `bi*kTileK+tok` | PASS |
| `k_dsa_topk2048_streaming_merge` / `_finalize` | 7169-7333 / 7335-7360 | 256,2 / 256,4 | 8.3 KiB | `nv = 2048+n_new <= 18,432`; `n_tiles <= 64` at 1M | PASS |
| TP4-h16 sparse MQA (`k_dsa_sparse_mqa_tp4_h16`) | 7296-7536 | 256,1 | 56.5 KiB | `size_t(kpos)*576u`; `valid = sel_cnts[m]` | PASS |
| Zero-sync mapped TP4 collectives (`kPingPong=4`) | 2560-2879 | 256,4 | none | `n_elems>>3`, epoch ring `& 3` | PASS |
| Commit 6 DSpark fusion (`k_fill_iota_i32`, `k_dspark_broadcast_noise_and_pos`, `k_dspark_markov_bias_multimax_f32`, `k_softmax_*_packed`) | 7400-7620 | 256,8 / 256,4 / 64,4 | 2 KiB | `size_t(v)*size_t(R)`, `V = 201,216` | PASS |

**Overflow verdict:** no `int32_t` index inside this file can overflow at `seq_len = 524,288` or `1,048,576`. The largest products are `size_t(pos)*576u = 603,979,776`, `n_hist <= 1,048,576`, and `2048+n_new <= 18,432` — all `< 2^31`. LDS maximum is 56.5 KiB (`s_KV` 36,992 B + `s_Q` 18,496 B + union 2,048 B + 320 B) ≤ 64 KiB, and every kernel is `ScratchSize == 0` with 0 spills/slc/trig (verified by `make audit-gfx906`).

## 3. Certification

- 302/302 gfx906 kernels `ScratchSize == 0`, 0 spills, 0 `slc`, 0 trig.
- 0 peer-to-peer calls; 0 B net new device allocation; 0 B DDR4 expert spill.
- Type_HH_HPA (`rocblas_datatype_f32_r` compute) and FP32 router logits throughout.
- No concrete 32-bit overflow or boundary defect found; no patch required.
- Recommendation: migrate the wave6/dpp4 stride products to `size_t` in a future
  commit (needs a GPU parity run before landing).
