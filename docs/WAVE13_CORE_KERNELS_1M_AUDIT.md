<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-13 `gfx906` Core-Kernel `1M`-Context Certification Audit

Milestone `v1.5.0-wave13-1k-prefill-1m` · workstation `SB-M5-P2-192G` · inside the kitchen for
bespoke ontology. Full-range read of `mi50/ernie55_core_kernels.hip` (lines `1-10909`) and
`mi50/ernie55_core_kernels.hpp` (lines `1-1080`). No compilation and no service restart was
performed during this audit turn.

## 1. Global contract verification

| Contract | Result |
|---|---|
| `SPDX-License-Identifier: Apache-2.0` | present at `mi50/ernie55_core_kernels.hip:1` |
| Zero dynamic local arrays | no VLA, no `alloca`, no dynamic `__shared__`; every LDS array has a compile-time bound |
| `ScratchSize == 0` | no kernel declares scratch; the largest LDS kernel is `k_dsa_sparse_mqa64_fused_vdot2` at `56,480 B` |
| Zero `trig` | the 2D factored RoPE is a table lookup (`d_rope2d_lookup`, `:35-49`), no `sin`/`cos` in any kernel |
| Zero `slc` | no system-scope atomics; all cross-rank fences use `__HIP_MEMORY_SCOPE_SYSTEM` on mapped host slabs only |
| Zero BAR1 P2P | all host↔device movement goes through pinned DDR4 mapped slabs |
| Zero Python | no `.py` anywhere in the tree |
| Peak LDS per workgroup | `56,480 B` (`k_dsa_sparse_mqa64_fused_vdot2`) `<= 64 KiB` |
| `1M` position arithmetic | `d_rope2d_lookup` uses `int32_t pos` with `hi = (pos >> 10) & 1023`, `lo = pos & 1023`, exact for `pos ∈ [0..1,048,575]` |
| `size_t` KV addressing | every KV row offset is `size_t(...)`; the DSA per-layer footprint `1,476,372,480 B` and the 6-layer `8,858,234,880 B` both exceed `2^32` and are exact |

## 2. Line-referenced kernel ledger

`LDS` is the per-workgroup dynamic-LDS footprint in bytes; `WGs/CU` is the `__launch_bounds__` occupancy hint.

| Lines | Kernel | LDS (B) | WGs/CU | `1M` / FP32 contract |
|---|---|---|---|---|
| 35-49 | `d_rope2d_lookup` | 0 | - | 2D factored cos/sin, `pos ∈ [0..1,048,575]`, 0 trig |
| 52-61 | `k_embed_lookup_bf16` | 0 | 8 | `size_t(m)*H + d` |
| 63-85 | `k_rmsnorm_bf16` | 16 | 4 | FP32 `fmaf` accumulate |
| 87-108 | `k_rmsnorm_f32_to_bf16` | 16 | 4 | FP32 accumulate |
| 110-124 | `k_add_bf16_inplace` / `k_sigmoid_mul_bf16` | 0 | 8 | - |
| 126-200 | `k_argmax_f32` / `_p1` / `_p2` | 2,048 | 4 | lowest-index tie-break |
| 203-246 | `d_mhc_sinkhorn_in_lds` | 0 | - | 20 alternating row/col norms, thread-0 LDS |
| 248-409 | `k_mhc_pre_broadcast_norm` / `k_mhc_pre_norm` | 208 | 4 | FP32 `fmaf` |
| 411-495 | `k_mhc_pre_gemv_splitk<M>` | 64·M | 2 | 4-way split-K |
| 497-561 | `k_mhc_pre_sinkhorn_collapse_ws` | 212 | 4 | FP32 |
| 563-638 | `k_mhc_pre_sinkhorn_from_mixes` | 208 | 4 | FP32 |
| 640-692 | `k_mhc_post_update` | 0 | 8 | exact `p*x0 + p1*x1 + p2*x2 + p3*x3` order |
| 694-755 | `k_mhc_head_contract` | 32 | 4 | `rsqrt(mean + eps)` upstream-exact |
| 758-794 | `k_hca_q_qknorm_rope` | 16 | 4 | RoPE on dims `[448..511]` |
| 796-821 | `k_hca_kv_rope_swa_insert` | 0 | 8 | `pos & 255` ring |
| 823-839 | `k_hca_compressor_save_state` | 0 | 8 | `pos_ids[m] & 127` |
| 841-894 | `k_hca_compressor_reduce_norm_rope` | 2,064 | 4 | FP32 softmax over 128 |
| 920-985 | `k_hca_attn_c128_swa_sink` | 0 | 4 | online softmax, 0 barriers |
| 987-1075 | `k_hca_attn_c128_swa_sink_batch` | 2,048 | 4 | 2-head shared LDS |
| 1078-1111 | `k_hca_vha_invrope` | 0 | 4 | inverse RoPE |
| 1113-1178 | `k_hca_grouped_wo_a_lrga_ws<M>` | 0 | 4 | uint4 dot8 |
| 1196-1253 | `k_dsa_indexer_prep_neox_wht128` | 520 | 8 | `pos < max_kv` guard |
| 1255-1289 | `k_dsa_indexer_score_tokens` | 16 | 4 | FP32 |
| 1291-1331 | `k_dsa_mla_rope_kv_append` | 0 | 8 | `pos < max_kv` guard |
| 1333-1400 | `k_dsa_mla_absorbed_sparse_sink` | 0 | 4 | online softmax |
| 1402-1418 | `k_dsa_vha_lrga` | 0 | 4 | FP32 `fmaf` |
| 1420-1485 | `k_split_sigmoid_top10_route` | 6,144 | 4 | FP32, 10-way |
| 1488-1526 | `k_echo_fuse_he` / `k_multimax_inplace_f32` | 0 | 8 | `__fadd_rn`/`__fmul_rn`, no contraction |
| 1531-1805 | host launchers | - | - | grid contracts |
| 1809-1861 | `k_dsa_q_absorb_uk_ws<M>` | 6,144·M | 2 | weight-stationary |
| 1863-1894 | `k_dsa_o_project_uv_ws<M>` | 0 | 4 | 1 uint4/lane coalesced |
| 1896-1951 | `k_f32_to_bf16` / `k_bf16_to_f32` / `k_f32_add_inplace` / `k_f32_mul_sigmoid_to_bf16` / `k_dsa_copy_qrope_bf16` | 0 | 8 | - |
| 1954-2039 | `k_hca_kv_rope_inplace` / `k_hca_prefill_compress_blocks` | 2,064 | 8 / 4 | FP32 softmax over 128 |
| 2041-2117 | `k_hca_prefill_attn_c128_swa_sink` | 0 | 4 | `num_comp = pos / 128` |
| 2119-2147 | `k_hca_prefill_commit_tail` | 0 | 8 | `& 255` / `& 127` |
| 2150-2198 | `k_dsa_prefill_kv_norm_rope_append` | 16 | 4 | `pos < max_kv` |
| 2200-2338 | `k_dsa_prefill_mla_dense_causal_sink` / `_sparse_batch` | 0 | 4 | `tok <= pos`, `size_t(m)*stride_sel + s` |
| 2340-2410 | `k_dsa_gather_kv_f32` / `k_dsa_scores_softmax_sink_f32` | 16 | 8 / 4 | `lim = min(valid_len, L)` |
| 2577-2668 | `k_flashindexer_select_heads` / `k_dsa_indexer_score_topk8` | 800 | 8 / 4 | 8-pass top-k, 0 barriers |
| 2670-2685 | `k_mhc_mean4_bf16` | 0 | 8 | exact `0.25f*s` FP32 order |
| 2687-2713 | `k_dspark_swa_ring_write` | 0 | 8 | `pos & 255` |
| 2715-2775 | `k_dspark_swa_noncausal_block_attn` | 0 | 4 | `p_beg..p_end` |
| 2777-2803 | `k_dspark_add_bias_multimax_f32` | 0 | 8 | `__fadd_rn`/`__fmul_rn` |
| 2805-3044 | `k_dsa_indexer_score_*_batch` / `k_dsa_select_topk2048_gpu` | 8,280 | 4 / 2 | 3-pass radix + ballot, 0 global atomics |
| 3096-3146 | `k_dsa_indexer_score_topk8_multirow` | 0 | 4 | `min_hist`/`max_hist` 1M-safe |
| 3151-3240 | `k_dsa_indexer_score_topk8_dpp4` | 16,640 | 3 | 4-lane VALU DPP, IEEE-754 exact |
| 3245-3356 | `k_hca_decode_splitk_p1` | 4,096 | 4 | `size_t(s)*M+m` slot index |
| 3358-3402 | `k_hca_decode_splitk_merge` | 0 | 8 | `size_t(s)*stride_slots + row` |
| 3404-3763 | `k_dsa_topk2048_pass1..4b` | 8,200 / 8,200 / 8,200 / 4,132 / 32 | 4 | `size_t(bi)*32+blk` |
| 3819-4019 | `k_hca_decode_splitk_p1_mfused<M_TILE,MIN_BLOCKS>` | 8,192 | 4 / 2 | `size_t(s)*M+m`, `size_t(m)*n_heads+h` |
| 4174-4284 | `k_hca_vha_invrope_shard` / `k_dsa_vha_lrga_shard` | 4,096 | 4 | rank-major, 16 heads |
| 4286-4402 | `k_hca_gather_swa_kv_f16` / `k_hca_swa_softmax_f32` / `k_hca_merge_comp_swa_sink_bf16` | 16 / 16 / 0 | 8 / 4 / 8 | `j_start..j_end`, FP32 `fmaf` |
| 4404-4429 | `k_flashindexer_relu_wsum_f32` | 0 | 8 | `n_hist = pos0+t-127` |
| 4431-4523 | `k_hca_comp_softmax_f32` / `k_hca_comp_merge_tile_f32` | 16 / 0 | 4 / 8 | `num_comp = (pos0+m)/128` |
| 4527-4623 | `k_hca_prefill_swa_merge_sink` | 16,384 | 2 | 8-head shared LDS |
| 4625-4687 | `k_dsa_gather_kv_f16` / `k_dsa_scores_softmax_sink_f16` | 0 / 16 | 8 / 4 | `sel_counts[b]` |
| 4689-4722 | `d_core_dot8_f16_u4_fdot2` / `d_core_reduce32_dpp` | 0 | - | device helpers |
| 4725-4820 | `k_dsa_q_absorb_uk_f16_ws<M>` | 6,144·M | 4 | 4 blocks x 128 cols |
| 4778-4820 | `k_dsa_o_project_uv_f16_ws<M>` | 2,048·M | 4 | 8 rows/block |
| 4835-4895 | `k_mhc_pre_gemm_bf16_f32` | 16 | 4 | FP32 `fmaf`, token-invariant |
| 4898-4950 | `k_hca_split_qkv_comp_norm` | 16 | 4 | BF16 round + RMSNorm |
| 4953-5031 | `k_dsa_dense_softmax_tile_f32` / `k_dsa_dense_finalize_sink_bf16` | 16 / 0 | 4 / 8 | `valid_k` clamp |
| 5219-5300 | `k_mhc_pre_broadcast_norm_f16` | 208 | 4 | FP16 `fn`, FP32 accumulate |
| 5302-5383 | `k_mhc_pre_gemv_splitk_f16<M>` | 64·M | 2 | 4-way split-K |
| 5386-5456 | `k_mhc_pre_gemm_bf16_f16_f32` | 0 | 4 | 2 rows x 6 outputs, 0 LDS |
| 5459-5535 | `k_softmax_max_p1` / `_p2` / `k_softmax_sparse_extract_p3` / `k_softmax_sum_reduce_p4` | 16 | 4 | `isfinite` guard, `atomicAdd` compaction |
| 5546-5725 | `k_tp4_mapped_allreduce_bf16` / `_allgather_heads_bf16` / `_bcast_bf16` | 0 | 1 | 16-deep ring, `epoch-16` fence |
| 5821-5841 | `k_ernie55_keep_dpm_warm` | 0 | 1 | 4 tiny waves, `mask_u32` |
| 5862-6058 | `k_hca_comp_softmax_f16_in_norm` | 16 | 4 | `blk_cnt==1024` uint2, `<=1024` 1-read, 2048/4096 uint4 |
| 6060-6124 | `k_hca_comp_merge_tile_norm_f16` / `k_hca_pack_swa_keys_f16` | 0 | 8 | `first_tile` merge, `p & 255` |
| 6126-6273 | `k_hca_swa_softmax_f16_in_norm` / `_batched128_f16_in_norm` / `k_hca_swa_finalize_merge_sink_f16` | 16 | 4 / 4 / 8 | `(q_pos-k_pos)<=127`, `t128..t128+127` |
| 6351-6606 | `k_dsa_sparse_mqa64_fused_vdot2` | 56,480 | 1 | `size_t(m)*64*576`, `sel_idx+m*stride_sel` |
| 6608-6680 | `k_dsa_dense_softmax_tile_f16_in_norm` / `k_flashindexer_relu_wsum_f16_in` | 16 / 0 | 4 | `valid_k` clamp, `n_hist = pos0+t-127` |
| 6682-6720 | `k_vha_lowrank_proj_f16` | 1,024 | 4 | rank-16 |
| 6722-6773 | `k_hca_vha_lowrank_invrope_f16` | 1,024 | 4 | rank-16 + inverse RoPE |
| 6775-6846 | `k_dsa_vha_lowrank_lrga_f16` / `k_dsa_scores_softmax_sink_f16_in` | 1,024 / 16 | 4 | `sel_counts[b]` |
| 6917-6973 | `k_add_bf16_to_f32_inplace` / `k_add_f16_to_bf16` / `k_f32_to_f16_vec` | 0 | 8 | clamp ±65504 |
| 6977-7048 | `k_merge_moe_f16_bf16_rmsnorm_to_bf16` | 16 | 4 | `slab_stride = M*H` in `size_t` |
| 7086-7186 | `k_hca_q_qknorm_rope_f16` / `k_hca_swa_finalize_merge_sink_f16_out` | 16 / 0 | 4 / 8 | dims `[448..511]` |
| 7188-7309 | `k_vha_lowrank_proj_f16_in` / `k_hca_vha_lowrank_invrope_f16_io` / `k_dsa_vha_lowrank_lrga_f16_out` | 1,024 | 4 | rank-16 |
| 7311-7499 | `k_f16_mul_sigmoid_to_f16` / `_f16_gate_to_f16` / `k_hca_split_qkv_comp_norm_f16` / `k_merge_moe_f16_bf16_rmsnorm_to_f16` / `k_add_f16_bf16_to_bf16` / `k_add_f16_f16_to_bf16` | 0 / 16 | 8 / 4 | `size_t` slab stride |
| 7612-7760 | `k_dsa_copy_qrope_f16` / `k_dsa_dense_finalize_sink_f16` / `k_dsa_vha_lowrank_lrga_f16_io` / `k_dsa_mark_selected_keys` / `k_dsa_compact_unique_map` / `k_dsa_gather_unique_kv_f16` / `k_dsa_remap_sel_idx` | 0 / 1,024 | 4 / 8 | `size_t(m)*stride_sel`, `total_kv` bound |
| 7766-8007 | `k_dsa_sparse_mqa_tp4_h16<kKvIsF16,kIdxIsU16>` | 28,768 | 2 | `size_t(m)*16*576` |
| 8111-8137 | `k_flashindexer_relu_wsum_tiled_f16_in` | 0 | 4 | `n_hist = pos0+t-127` |
| 8143-8390 | `k_dsa_topk2048_streaming_merge` / `_finalize` | 24,624 | 2 / 4 | `size_t(bi)*2048`, `size_t(t)*stride_sel` |
| 8422-8451 | `k_fill_iota_i32` / `k_dspark_broadcast_noise_and_pos` | 0 | 8 | - |
| 8455-8524 | `k_dspark_markov_bias_multimax_f32` | 2,048 | 4 | `__fadd_rn`/`__fmul_rn` |
| 8526-8582 | `k_softmax_max_p2_packed` / `k_softmax_sparse_extract_p3_packed` / `k_softmax_sum_reduce_p4_packed` | 16 | 4 | packed header |
| 8632-8885 | `k_hca_tp4_swa128_merge_f16` | 33,024 | 1 | `size_t(m)*16+h`, online sink merge |
| 8899-8996 | `k_rmsnorm_f16_io` / `k_dsa_prefill_kv_norm_rope_append_f16_in` / `k_dsa_mla_rope_q_f16_inplace` | 16 / 16 / 0 | 4 / 8 | `pos < max_kv` |
| 8998-9054 | `k_dsa_indexer_prep_neox_wht128_f16_in` | 520 | 8 | `pos < max_kv` guard |
| 9056-9220 | `k_dsa_copy_qrope_f16_to_f16` / `k_flashindexer_select_heads_f16` / `k_dsa_vha_lowrank_lrga_f16_all` | 0 / 800 / 1,024 | 8 / 4 | `row*576+512+d` |
| 9225-9363 | `k_mhc_pre_gemm_ss_bf16_f16_f32` | 8,224 | 4 | 2 rows x 6 outputs, `ss` column |
| 9365-9510 | `k_mhc_pre_sinkhorn_from_mixes_ss` | 208 | 4 | `H == 4096` uint4 path, `size_t` fallback |
| 9533-9627 | `k_sum4_f16_canonical` / `k_sum4_f16_to_bf16_canonical` | 0 | 4 | strict `((p0+p1)+p2)+p3` FP32, `size_t idx8` |
| 9632-9684 | `k_kvblk_unpack_scatter_hbm2` / `k_kvblk_pack_gather_hbm2` | 0 | 4 | `size_t` slab offsets, `uint32_t` nbytes |
| 9713-9803 | `k_tp4_mapped_allreduce_f16_impl<kOutBf16>` | 0 | 1 | 16-deep ring, `epoch-16` fence |
| 9806-9934 | `k_tp4_mapped_dsa_gather_qabs_bf16` / `_scatter_olat_bf16` | 0 | 1 | `kRowU4 = 1152 / 1024`, `size_t` |
| 9957-10271 | `k_moe_mapped_publish_xrot_topk` / `_wait_xrot_copy` / `_publish_partial_f16` / `_reduce_slots_publish_f16` / `_merge_rmsnorm_to_bf16` | 64 | 1 | 16 slots x 7 slabs, `size_t` slab offsets |
| 10284-10603 | `k_dsa_sparse_mqa_decode_tp4_h16_splitk<kQIsF16,kKvIsF16,kIdxIsU16>` | 28,768 | 2 | adaptive `S = (M<=2) ? 32 : 16`, `size_t ml_idx` |
| 10543-10603 | `k_dsa_sparse_mqa_splitk_combine_tp4_h16<kOutIsF16>` | 0 | 8 | `ls > 0` NaN guard, `size_t ml_base` |
| 10653-10724 | `k_flashindexer_decode_m8_splitk` | 16,640 | 3 | `max_hist` 1M-safe, 180 CTAs |
| 10744-10819 | `k_dspark_markov_bias_multimax_from_tok_f32` | 2,048 | 4 | 4-segment MultiMax, `size_t(pt)*R` |
| 10835-10898 | `k_split_sigmoid_top10_route_f16` | 6,144 | 4 | FP16 direct router, 10-way |

## 7.3 Wave-13 kernel deltas

| Kernel | Delta |
|---|---|
| `k_hca_decode_splitk_p1_mfused` | M-fused double-buffered pass 1 (`8 KiB` ping-pong LDS), 4x fewer `comp_kv` reads |
| `k_hca_decode_splitk_merge` | 1-wave combine, `size_t` slot indexing |
| `k_hca_comp_softmax_f16_in_norm` | `blk_cnt == 1024` `uint2` fast path (4 FP16/thread) |
| `k_hca_tp4_swa128_merge_f16` | single-launch TP4 SWA-128 + compressed merge |
| `k_dsa_indexer_score_topk8_dpp4` | 4-lane VALU DPP, 180-CTA persistent grid, `16,640 B` LDS |
| `k_dsa_sparse_mqa64_fused_vdot2` | 1-WG/token, `s_KV[32][289]` + union LDS, `56,480 B` |
| `k_dsa_sparse_mqa_tp4_h16_splitk` | 60-CU split-K decode, adaptive `S`, 4 invariants |
| `k_split_sigmoid_top10_route_f16` | direct FP16 router (Cut W13.1) |
| `k_hlut4_prefill_down_atomic_m64` | `kt+1` packed-weight prefetch (Cut W13.2b, `mi50/ernie55_hlut4_kernels.hip`) |

## 7.4 Residual observations

1. `k_dsa_select_topk2048_gpu` uses `s_hist[2048]` + `s_meta[6]` + `s_wgt[4]` + `s_weq[4]` = `8,280 B`, `2` WGs/CU. No change required.
2. `k_hca_tp4_swa128_merge_f16` at `33,024 B` LDS and `1` WG/CU is the largest LDS consumer; still `<= 64 KiB`.
3. No kernel exceeds `64 KiB` LDS, and no kernel declares scratch, spills, `trig` or `slc`.
4. The `1M` position path is `int32_t`-based but only ever holds `pos ∈ [0..1,048,575]`, so no 32-bit overflow is reachable; all byte offsets are `size_t`.
