// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure gfx906 HIP Zero-Scratch (ScratchSize == 0, 0 trig, 0 spills, 0 slc) Core Architectural Kernels
// for baidu/ERNIE-5.5-Flash (ernie5_5_moe):
//   - mHC (4-stream Manifold-Constrained Hyper-Connections, 1/(sqrt(mean)+eps), 20-iter Sinkhorn, hc_head)
//   - HCA (37 layers: 128:1 block compressor + 128-SWA, 512-d heads, YaRN RoPE, QK-norm, VHA, inv-RoPE, LRGA)
//   - mqa_dsa (6 backbone + 1 ECHO: Lightning Indexer NeoX RoPE + WHT128 + top-2048 + 128-SWA + Absorbed MLA + VHA + LRGA)
//   - Multi-Router Latent MoE (split-feature dual-sigmoid top-10 + bias + learnable per-expert scale)
//   - ECHO (L43 multi-horizon drafter) + MultiMax (exact FP32 piecewise output head)

#include "../src/ernie55_config.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ernie55 {

// 2D Factored RoPE Table: pos = hi*1024 + lo -> 0 device trig across full 1,048,576 context (~512 KiB/table!)
struct Rope2DTableGpu {
    float* d_cos_hi = nullptr;  // [1024, 32]
    float* d_sin_hi = nullptr;  // [1024, 32]
    float* d_cos_lo = nullptr;  // [1024, 32]
    float* d_sin_lo = nullptr;  // [1024, 32]
};

// ---- Basic Primitives ----
void launch_embed_lookup_bf16(const uint16_t* d_embed_bf16, const int32_t* d_tok_ids,
                              uint16_t* d_out_bf16, int M, int H, hipStream_t stream);
void launch_rmsnorm_bf16(const uint16_t* d_in_bf16, const uint16_t* d_w_bf16,
                         uint16_t* d_out_bf16, int M, int D, float eps, hipStream_t stream);
void launch_rmsnorm_f32_to_bf16(const float* d_in_f32, const uint16_t* d_w_bf16,
                                uint16_t* d_out_bf16, int M, int D, float eps, hipStream_t stream);
void launch_add_bf16_inplace(uint16_t* d_dst_bf16, const uint16_t* d_src_bf16, int total, hipStream_t stream);
void launch_sigmoid_mul_bf16(uint16_t* d_x_bf16, const uint16_t* d_gate_bf16, int total, hipStream_t stream);
void launch_argmax_f32(const float* d_logits, int32_t* d_out_id, int V, hipStream_t stream);
void launch_argmax_f32_fast(const float* d_logits, float* d_part_vals64, int32_t* d_part_idx64,
                            int32_t* d_out_id, int V, hipStream_t stream, float* d_out_val = nullptr);

// ---- mHC (Manifold-Constrained Hyper-Connections, hc_mult=4, sinkhorn=20) ----
void launch_mhc_pre_broadcast_norm(
    const uint16_t* d_embed_bf16,   // [M, 4096]
    const float* d_fn_24x16384,     // [24, 16384] (transposed ckpt mapping_proj)
    const float* d_hc_scale3,       // [3]
    const float* d_hc_base24,       // [24]
    const uint16_t* d_norm_w_bf16,  // [4096]
    uint16_t* d_res_out_bf16,       // [M, 4, 4096]
    float* d_post_mix,              // [M, 4]
    float* d_comb_mix,              // [M, 16]
    uint16_t* d_layer_in_bf16,      // [M, 4096]
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

void launch_mhc_pre_norm(
    const uint16_t* d_res_bf16,     // [M, 4, 4096]
    const float* d_fn_24x16384,     // [24, 16384]
    const float* d_hc_scale3,       // [3]
    const float* d_hc_base24,       // [24]
    const uint16_t* d_norm_w_bf16,  // [4096] (may be nullptr)
    float* d_post_mix,              // [M, 4]
    float* d_comb_mix,              // [M, 16]
    uint16_t* d_layer_in_bf16,      // [M, 4096]
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

// Multi-CU 4-way split-K weight-stationary mHC pre-norm for decode/verify (M <= 8, 96 blocks!)
void launch_mhc_pre_norm_fast(
    const uint16_t* d_res_bf16,     // [M, 4, 4096]
    const float* d_fn_24x16384,     // [24, 16384]
    const float* d_hc_scale3,       // [3]
    const float* d_hc_base24,       // [24]
    const uint16_t* d_norm_w_bf16,  // [4096] (may be nullptr)
    float* d_mhc_ws,                // [M, 25, 4] F32 split-K workspace (<= 3.2 KiB for M<=8)
    float* d_post_mix,              // [M, 4]
    float* d_comb_mix,              // [M, 16]
    uint16_t* d_layer_in_bf16,      // [M, 4096]
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

// Prefill mHC Sinkhorn + collapse + RMSNorm consuming pre-computed rocblas_sgemm mixes [M, 24]
void launch_mhc_pre_sinkhorn_from_mixes(
    const uint16_t* d_res_bf16,     // [M, 4, 4096]
    const float* d_raw_mixes_Mx24,  // [M, 24] = res_f32 @ fn^T
    const float* d_hc_scale3,       // [3]
    const float* d_hc_base24,       // [24]
    const uint16_t* d_norm_w_bf16,  // [4096] (may be nullptr)
    float* d_post_mix,              // [M, 4]
    float* d_comb_mix,              // [M, 16]
    uint16_t* d_layer_in_bf16,      // [M, 4096]
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

void launch_mhc_post_update(
    const uint16_t* d_sublayer_bf16, // [M, 4096]
    const uint16_t* d_res_in_bf16,   // [M, 4, 4096]
    const float* d_post_mix,         // [M, 4]
    const float* d_comb_mix,         // [M, 16] (einsum ij,ih->jh!)
    uint16_t* d_res_out_bf16,        // [M, 4, 4096]
    int M, int H, hipStream_t stream);

void launch_mhc_head_contract(
    const uint16_t* d_res_bf16,      // [M, 4, 4096]
    const float* d_hc_head_fn,       // [4, 16384]
    const float* d_hc_head_scale,    // [1]
    const float* d_hc_head_base,     // [4]
    const uint16_t* d_final_norm_w,  // [4096]
    uint16_t* d_out_bf16,            // [M, 4096]
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

// ---- HCA (37 Layers: 128:1 Compressor + 128-SWA, 512-d, YaRN RoPE, QK-Norm, VHA, inv-RoPE, LRGA) ----
void launch_hca_q_qknorm_rope(
    uint16_t* d_q_bf16,              // [M, n_heads, 512] in-place weightless RMSNorm + interleaved YaRN RoPE [448..511]
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    int M, int n_heads, float eps, hipStream_t stream);

void launch_hca_kv_rope_swa_insert(
    const uint16_t* d_kv_normed_bf16, // [M, 512]
    const int32_t* d_pos_ids,         // [M]
    const Rope2DTableGpu& rope,
    uint16_t* d_swa_ring_bf16,        // [128, 512] ring buffer indexed by pos % 128
    int M, hipStream_t stream);

void launch_hca_compressor_save_state(
    const float* d_wkv_f32,          // [M, 512]
    const float* d_wgate_f32,        // [M, 512]
    const float* d_ape_128x512,      // [128, 512]
    const int32_t* d_pos_ids,        // [M]
    float* d_state_kv,               // [128, 512] current block buffer
    float* d_state_score,            // [128, 512] current block buffer
    int M, hipStream_t stream);

void launch_hca_compressor_reduce_norm_rope(
    const float* d_state_kv,         // [128, 512]
    const float* d_state_score,      // [128, 512]
    const uint16_t* d_comp_norm_w,   // [512]
    const Rope2DTableGpu& rope,
    int32_t compressed_pos,          // (pos / 128) * 128 = block_idx * 128!
    uint16_t* d_comp_kv_dst_bf16,    // [512] destination row in compressed KV cache!
    float eps, hipStream_t stream);

void launch_hca_attn_c128_swa_sink(
    const uint16_t* d_q_bf16,        // [n_heads, 512] single token decode/step query
    const uint16_t* d_comp_kv_bf16,  // [num_comp, 512] compressed KV cache
    const uint16_t* d_swa_ring_bf16, // [256, 512] SWA ring (indexed pos & 255)
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_out_bf16,            // [n_heads, 512]
    int n_heads, int32_t pos, int num_comp, hipStream_t stream);

void launch_hca_attn_c128_swa_sink_batch(
    const uint16_t* d_q_bf16,        // [M, n_heads, 512] M<=8 sub-chunk without crossing 128-boundary
    const uint16_t* d_comp_kv_bf16,  // [num_comp, 512] compressed KV cache
    const uint16_t* d_swa_ring_bf16, // [256, 512] SWA ring (already populated with pos0..pos0+M-1)
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_out_bf16,            // [M, n_heads, 512]
    int n_heads, int32_t pos0, int M, int num_comp, hipStream_t stream);

// ---- Batched HCA Prefill Kernels (M up to 4096 in 1 launch, 100% bit-exact with decode!) ----
void launch_hca_kv_rope_inplace(
    uint16_t* d_kv_chunk_bf16,       // [M, 512] in-place YaRN RoPE [448..511]
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    int M, hipStream_t stream);

void launch_hca_prefill_compress_blocks(
    const float* d_wkv_chunk_f32,    // [M, 512]
    const float* d_wgate_chunk_f32,  // [M, 512]
    const float* d_ape_128x512,      // [128, 512]
    const float* d_prev_state_kv,    // [128, 512] (for partial block straddling pos0)
    const float* d_prev_state_sc,    // [128, 512]
    const uint16_t* d_comp_norm_w,   // [512]
    const Rope2DTableGpu& rope,
    int32_t pos0, int M,
    uint16_t* d_comp_kv_cache_bf16,  // [max_kv/128, 512]
    float eps, hipStream_t stream);

void launch_hca_prefill_attn_c128_swa_sink(
    const uint16_t* d_q_bf16,        // [M, n_heads, 512]
    const uint16_t* d_comp_kv_bf16,  // [max_kv/128, 512]
    const uint16_t* d_swa_ring_bf16, // [128, 512] pre-chunk SWA state (< pos0)
    const uint16_t* d_kv_chunk_rot,  // [M, 512] rotated chunk KV (>= pos0)
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_out_bf16,            // [M, n_heads, 512]
    int n_heads, int32_t pos0, int M, hipStream_t stream);

void launch_hca_prefill_commit_tail(
    const uint16_t* d_kv_chunk_rot,  // [M, 512]
    const float* d_wkv_chunk_f32,    // [M, 512]
    const float* d_wgate_chunk_f32,  // [M, 512]
    const float* d_ape_128x512,      // [128, 512]
    int32_t pos0, int M,
    uint16_t* d_swa_ring_bf16,       // [128, 512] updated with tail <= 128
    float* d_state_kv,               // [128, 512] updated with unfinished block tail
    float* d_state_sc,               // [128, 512]
    hipStream_t stream);

// ---- Batched DSA Prefill Kernels (M tokens in 1 launch, zero per-token host sync!) ----
void launch_dsa_prefill_kv_norm_rope_append(
    uint16_t* d_q_bf16,              // [M, n_heads, 256] interleaved RoPE [192..255]
    const uint16_t* d_kv_a_raw_576,  // [M, 576] = [512 kv_c_raw | 64 k_pe_raw]
    const uint16_t* d_kv_a_norm_w,   // [512]
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    uint16_t* d_mla_kv_cache_bf16,   // [max_kv, 576]
    int M, int n_heads, int max_kv, float eps, hipStream_t stream);

void launch_dsa_prefill_mla_dense_causal_sink(
    const uint16_t* d_q_abs_bf16,    // [M, n_heads, 576]
    const uint16_t* d_mla_kv_bf16,   // [max_kv, 576]
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_o_lat_bf16,          // [M, n_heads, 512]
    int n_heads, int32_t pos0, int M, hipStream_t stream);

void launch_dsa_prefill_mla_sparse_batch(
    const uint16_t* d_q_abs_bf16,    // [M, n_heads, 576]
    const uint16_t* d_mla_kv_bf16,   // [max_kv, 576]
    const int32_t* d_sel_indices_MxS,// [M, stride_sel]
    const int32_t* d_sel_counts,     // [M]
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_o_lat_bf16,          // [M, n_heads, 512]
    int n_heads, int M, int stride_sel, hipStream_t stream);

void launch_hca_vha_invrope(
    const uint16_t* d_o_in_bf16,     // [M, 64, 512]
    const float* d_vha_m_64x64,      // [64, 64] = I + V @ U^T
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    uint16_t* d_o_out_bf16,          // [M, 64, 512]
    int M, hipStream_t stream);

void launch_hca_grouped_wo_a_lrga(
    const uint16_t* d_o_bf16,        // [M, 8, 4096] (64 heads grouped as 8 x 8x512)
    const uint16_t* d_wo_a_bf16,     // [8, 1024, 4096]
    const uint16_t* d_gate_bf16,     // [M, 8192] raw o_gate(qr) logits
    uint16_t* d_z_out_bf16,          // [M, 8192] = wo_a(o) * sigmoid(gate)
    int M, int n_groups, int o_lora_rank, hipStream_t stream);

// ---- mqa_dsa (6 Backbone + 1 ECHO: Lightning Indexer + Absorbed Sparse MLA + VHA + LRGA) ----
void launch_dsa_indexer_prep_neox_wht128(
    uint16_t* d_iq_bf16,             // [M, 64, 128] in-place NeoX RoPE [0..63] + WHT128
    const uint16_t* d_ik_raw_bf16,   // [M, 128]
    const uint16_t* d_knorm_w,       // [128] LayerNorm weight
    const uint16_t* d_knorm_b,       // [128] LayerNorm bias
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    uint16_t* d_ik_cache_bf16,       // [max_kv, 128] destination cache
    int M, int max_kv, hipStream_t stream);

void launch_dsa_indexer_score_tokens(
    const uint16_t* d_iq_bf16,       // [64, 128]
    const uint16_t* d_iw_bf16,       // [64] raw weights_proj(x)
    const uint16_t* d_ik_cache_bf16, // [n_hist, 128]
    float* d_scores_f32,             // [n_hist]
    int n_hist, hipStream_t stream);

void launch_dsa_mla_rope_kv_append(
    uint16_t* d_q_bf16,              // [M, n_heads, 256] interleaved RoPE on [192..255]
    const uint16_t* d_kv_c_norm_bf16,// [M, 512]
    const uint16_t* d_k_pe_bf16,     // [M, 64]
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    uint16_t* d_mla_kv_cache_bf16,   // [max_kv, 576] = [512 kv_c | 64 k_pe_rot]
    int M, int n_heads, int max_kv, hipStream_t stream);

void launch_dsa_mla_absorbed_sparse_sink(
    const uint16_t* d_q_abs_bf16,    // [n_heads, 576] = [512 q_nope@W_UK | 64 q_rope]
    const uint16_t* d_mla_kv_bf16,   // [max_kv, 576]
    const int32_t* d_sel_indices,    // [n_sel] (<= 2176)
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_o_lat_bf16,          // [n_heads, 512]
    int n_heads, int n_sel, hipStream_t stream);

void launch_dsa_q_absorb_uk(
    const uint16_t* d_q_bf16,        // [M, 64, 256] (192 nope + 64 rope)
    const uint16_t* d_kv_b_bf16,     // [64, 448, 512] (first 192 rows = W_UK, next 256 rows = W_UV)
    uint16_t* d_q_abs_bf16,          // [M, 64, 576] ([512 absorbed | 64 rope])
    int M, int n_heads, hipStream_t stream);

void launch_dsa_o_project_uv(
    const uint16_t* d_o_lat_bf16,    // [M, 64, 512]
    const uint16_t* d_kv_b_bf16,     // [64, 448, 512]
    uint16_t* d_o_v_bf16,            // [M, 64, 256]
    int M, int n_heads, hipStream_t stream);

void launch_f32_to_bf16(const float* d_in_f32, uint16_t* d_out_bf16, int total, hipStream_t stream);
void launch_f32_to_f16(const float* d_in_f32, uint16_t* d_out_f16, int total, hipStream_t stream);
void launch_bf16_to_f32(const uint16_t* d_in_bf16, float* d_out_f32, int total, hipStream_t stream);
void launch_f32_add_inplace(float* d_dst_f32, const float* d_src_f32, int total, hipStream_t stream);
void launch_add_bf16_to_f32_inplace(float* d_dst_f32, const uint16_t* d_src_bf16, int total, hipStream_t stream);
void launch_add_f16_to_bf16(uint16_t* d_dst_bf16, const uint16_t* d_src_f16, int total, hipStream_t stream);
void launch_merge_moe_f16_bf16_rmsnorm_to_bf16(
    const float* d_own_f32, const uint16_t* d_f16_slabs, int n_f16_slabs,
    const uint16_t* d_bf16_slab_or_null, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_bf16, int M, int H, float eps, hipStream_t stream);
void launch_f32_mul_sigmoid_to_bf16(const float* d_z_f32, const uint16_t* d_gate_bf16, uint16_t* d_out_bf16, int total, hipStream_t stream);
void launch_dsa_copy_qrope_bf16(const uint16_t* d_q_bf16, uint16_t* d_q_abs_bf16, int M, int n_heads, hipStream_t stream);

void launch_dsa_gather_kv_f32(
    const uint16_t* d_mla_kv_bf16, const int32_t* d_sel_indices_BxS,
    const int32_t* d_sel_counts, float* d_k_sel_f32,
    int B, int stride_sel, hipStream_t stream);

void launch_dsa_scores_softmax_sink_f32(
    float* d_scores_BxHxL, const float* d_attn_sink,
    const int32_t* d_sel_counts_or_null,
    int B, int n_heads, int L, int32_t pos0_causal, float scale, hipStream_t stream);

void launch_dsa_vha_lrga(
    const uint16_t* d_o_in_bf16,     // [M, 64, 256]
    const float* d_vha_m_64x64,      // [64, 64]
    const uint16_t* d_gate_bf16,     // [M, 16384]
    uint16_t* d_out_bf16,            // [M, 16384]
    int M, hipStream_t stream);

// ---- Multi-Router Split-Feature Sigmoid Top-10 Routing ----
void launch_split_sigmoid_top10_route(
    const float* d_logits_2E,        // [M, 1024] = [l0 (512) | l1 (512)]
    const float* d_corr_bias,        // [512]
    const float* d_scale_param,      // [512]
    int32_t* d_topk_ids,             // [M, 10]
    float* d_topk_weights,           // [M, 10]
    int M, hipStream_t stream,
    int max_expert = 512);

// ---- ECHO (L43) Fusion & MultiMax Output Head ----
void launch_echo_fuse_he(
    const uint16_t* d_hproj_4xH,     // [M, 4, 4096]
    const uint16_t* d_eproj_H,       // [M, 4096]
    uint16_t* d_out_4xH,             // [M, 4, 4096]
    int M, int H, hipStream_t stream);

void launch_multimax_inplace_f32(
    float* d_logits_f32,             // [M, V] updated in-place in exact IEEE order
    const float* d_ranges4,          // [4]
    const float* d_ts4,              // [4]
    int M, int V, hipStream_t stream);

// ---- ERNIE-5.5-Flash-Preview: FlashIndexer (Top-8 of 64 DSA Heads) & 3L MoE DSpark ----
void launch_flashindexer_select_heads(
    const float* d_gate_logits_f32,  // [M, 64] = x_in @ linear_gate.weight^T
    const float* d_bias_f32,         // [64] flashindexer_bias
    const uint16_t* d_iq_bf16,       // [M, 64, 128]
    const uint16_t* d_iw_bf16,       // [M, 64]
    uint16_t* d_q_sel_bf16,          // [M, 8, 128]
    float* d_w_sel_f32,              // [M, 8]
    int M, hipStream_t stream);

void launch_dsa_indexer_score_topk8(
    const uint16_t* d_q_sel_bf16,    // [8, 128]
    const float* d_w_sel_f32,        // [8]
    const uint16_t* d_ik_cache_bf16, // [n_hist, 128]
    float* d_scores_f32,             // [n_hist]
    int n_hist, hipStream_t stream);

void launch_dsa_indexer_score_topk8_batch(
    const uint16_t* d_q_sel_bf16,    // [M, 8, 128]
    const float* d_w_sel_f32,        // [M, 8]
    const uint16_t* d_ik_cache_bf16, // [max_kv, 128]
    float* d_scores_batch_f32,       // [B_sc, score_stride]
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream);

void launch_dsa_indexer_score_full_batch(
    const uint16_t* d_iq_bf16,       // [M, 64, 128]
    const uint16_t* d_iw_bf16,       // [M, 64]
    const uint16_t* d_ik_cache_bf16, // [max_kv, 128]
    float* d_scores_batch_f32,       // [B_sc, score_stride]
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream);

void launch_dsa_select_topk2048_gpu(
    const float* d_scores_batch_f32, // [B_sc, score_stride]
    int32_t* d_sel_idx_MxS,          // [M, stride_sel]
    int32_t* d_sel_cnts,             // [M]
    int32_t pos0, int t0, int B_sc, int score_stride, int stride_sel, hipStream_t stream);

// Wave-6 Zero-Scratch LDS Q-Stationary Multi-Row FlashIndexer & 32-Block VGPR Zero-Skip Top-2048 Selector
void launch_dsa_indexer_score_topk8_wave6(
    const uint16_t* d_q_sel_bf16,    // [M, 8, 128]
    const float* d_w_sel_f32,        // [M, 8]
    const uint16_t* d_ik_cache_bf16, // [max_kv, 128]
    float* d_scores_batch_f32,       // [B_sc, score_stride]
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream);

// Wave-7 4-Lane VALU DPP FlashIndexer (0 ds_bpermute, 0 bit-diffs vs d_warp_sum64, LDS Q-stationary)
void launch_dsa_indexer_score_topk8_dpp4(
    const uint16_t* d_q_sel_bf16,    // [M, 8, 128]
    const float* d_w_sel_f32,        // [M, 8]
    const uint16_t* d_ik_cache_bf16, // [max_kv, 128]
    float* d_scores_batch_f32,       // [B_sc, score_stride]
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream);

// Wave-12 Turn-17 Cut D12.1: M-fused double-buffered HCA split-K (pass 1 + pass 2 merge).
// One comp_kv_bf16 load feeds M_TILE = 4 verify tokens (4x fewer comp_kv HBM2 reads), ScratchSize=0.
void launch_hca_decode_splitk_mfused(
    const uint16_t* d_q_bf16, const uint16_t* d_comp_kv_bf16, const uint16_t* d_swa_ring_bf16,
    const float* d_attn_sink, float* d_part_o_f32, float* d_part_ml_f32, uint16_t* d_out_bf16,
    int n_heads, int32_t pos0, int M, int num_comp, hipStream_t stream);

// Wave-7 Split-K 4-Head HCA Compressed-KV + SWA Decode (M<=8, 60-CU saturation, 0 B new VRAM, M=1==M=6 bit-identical!)
void launch_hca_attn_c128_swa_sink_splitk(
    const uint16_t* d_q_bf16,        // [M, n_heads, 512]
    const uint16_t* d_comp_kv_bf16,  // [num_comp, 512]
    const uint16_t* d_swa_ring_bf16, // [256, 512]
    const float* d_attn_sink,        // [n_heads]
    float* d_part_o_f32,             // existing w.d_hca_o_tot_f32 (>= (S+1)*M*n_heads*512 floats)
    float* d_part_ml_f32,            // existing w.d_hca_ml_f32    (>= (S+1)*M*n_heads*2 floats)
    uint16_t* d_out_bf16,            // [M, n_heads, 512]
    int n_heads, int32_t pos0, int M, int num_comp, hipStream_t stream);

void launch_dsa_select_topk2048_wave6(
    const float* d_scores_batch_f32, // [B_sc, score_stride]
    uint32_t* d_scratch_u32,         // >= 5.01 MiB inside existing g_own.d_y_f32 (0 B new VRAM!)
    int32_t* d_sel_idx_MxS,          // [M, stride_sel]
    int32_t* d_sel_cnts,             // [M]
    int32_t pos0, int t0, int B_sc, int score_stride, int stride_sel, hipStream_t stream);

void launch_mhc_mean4_bf16(
    const uint16_t* d_res_4xH,       // [M, 4, 4096]
    uint16_t* d_out_H,               // [M, 4096] = mean across 4 mHC streams
    int M, int H, hipStream_t stream);

void launch_dspark_swa_ring_write(
    const uint16_t* d_kv_norm_bf16,  // [M, 512] already kv_norm'd
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,      // plain RoPE at rope_theta=8,000,000!
    uint16_t* d_swa_ring_256,        // [256, 512] indexed by pos & 255
    int M, hipStream_t stream);

void launch_dspark_swa_noncausal_block_attn(
    const uint16_t* d_q_bf16,        // [K, 64, 512] (already QK-norm + plain RoPE!)
    const uint16_t* d_swa_ring_256,  // [256, 512] (context [0..query_pos0-1] + K block queries)
    const float* d_attn_sink,        // [64]
    uint16_t* d_out_bf16,            // [K, 64, 512]
    int32_t query_pos0, int K, int sliding_window, hipStream_t stream);

void launch_dspark_add_bias_multimax_f32(
    const float* d_base_logits_f32,  // [V]
    const float* d_bias_f32,         // [V]
    const float* d_ranges4,          // [4] (nullable -> skip multimax)
    const float* d_ts4,              // [4]
    float* d_out_logits_f32,         // [V]
    int V, hipStream_t stream);

// ---- TP4 Sharded Rank-Major VHA + Batched GEMM FlashIndexer + Batched GEMM HCA Comp/SWA + FP16 DSA ----
void launch_hca_vha_invrope_shard(
    const uint16_t* d_o_in_rm_bf16,  // [tp_size, M, h_loc, 512] rank-major contiguous (0 memcpy2D!)
    const float* d_vha_m_64x64,      // [64, 64]
    const int32_t* d_pos_ids,        // [M]
    const Rope2DTableGpu& rope,
    uint16_t* d_o_out_bf16,          // [M, n_local_heads, 512]
    int M, int tp_size, int head_begin, int n_local_heads, hipStream_t stream);

void launch_dsa_vha_lrga_shard(
    const uint16_t* d_o_in_rm_bf16,  // [tp_size, M, h_loc, 256] rank-major contiguous (0 memcpy2D!)
    const float* d_vha_m_64x64,      // [64, 64]
    const uint16_t* d_gate_bf16,     // [M, n_local_heads * 256]
    uint16_t* d_out_bf16,            // [M, n_local_heads * 256]
    int M, int tp_size, int head_begin, int n_local_heads, hipStream_t stream);

void launch_hca_gather_swa_kv_f16(
    const uint16_t* d_swa_ring_bf16, // [256, 512]
    const uint16_t* d_kv_chunk_rot,  // [M, 512]
    uint16_t* d_k_swa_f16,           // [swa_pad, 512]
    int32_t pos0, int M, int32_t swa_lo, int swa_pad, hipStream_t stream);

void launch_hca_swa_softmax_f32(
    const float* d_scores_f32,       // [M * n_heads, swa_pad]
    float* d_m_swa_f32,              // [M * n_heads]
    float* d_l_swa_f32,              // [M * n_heads]
    uint16_t* d_probs_f16,           // [M * n_heads, swa_pad]
    int M, int n_heads, int32_t pos0, int32_t swa_lo, int swa_pad, hipStream_t stream);

void launch_hca_merge_comp_swa_sink_bf16(
    const float* d_comp_o_f32,       // [M * n_heads, 512] (ignored if !has_comp)
    const float* d_comp_m_f32,       // [M * n_heads]
    const float* d_comp_l_f32,       // [M * n_heads]
    const float* d_swa_o_f32,        // [M * n_heads, 512]
    const float* d_swa_m_f32,        // [M * n_heads]
    const float* d_swa_l_f32,        // [M * n_heads]
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_out_bf16,            // [M, n_heads, 512]
    int M, int n_heads, bool has_comp, hipStream_t stream);

void launch_flashindexer_relu_wsum_f32(
    const float* d_dots_f32,         // [B_sc * 8, k_cnt]
    const float* d_w_sel_f32,        // [M, 8]
    float* d_scores_batch_f32,       // [B_sc, score_stride]
    int32_t pos0, int t0, int B_sc, int k0, int k_cnt, int score_stride, hipStream_t stream);

void launch_hca_comp_softmax_f32(
    const float* d_scores_f32,       // [M * n_heads, blk_cnt]
    float* d_m_blk_f32,              // [M * n_heads]
    float* d_l_blk_f32,              // [M * n_heads]
    uint16_t* d_probs_f16,           // [M * n_heads, blk_cnt]
    int M, int n_heads, int32_t pos0, int blk0, int blk_cnt, hipStream_t stream);

void launch_hca_comp_merge_tile_f32(
    float* d_o_tot_f32, float* d_m_tot_f32, float* d_l_tot_f32,
    const float* d_o_blk_f32, const float* d_m_blk_f32, const float* d_l_blk_f32,
    int n_rows, bool first_tile, hipStream_t stream);

void launch_hca_prefill_swa_merge_sink(
    const uint16_t* d_q_bf16,        // [M, n_heads, 512]
    const uint16_t* d_swa_ring_bf16, // [256, 512]
    const uint16_t* d_kv_chunk_rot,  // [M, 512]
    const float* d_comp_o_f32,       // [M * n_heads, 512]
    const float* d_comp_m_f32,       // [M * n_heads]
    const float* d_comp_l_f32,       // [M * n_heads]
    const float* d_attn_sink,        // [n_heads]
    uint16_t* d_out_bf16,            // [M, n_heads, 512]
    int n_heads, int32_t pos0, int M, hipStream_t stream);

void launch_dsa_gather_kv_f16(
    const uint16_t* d_mla_kv_bf16, const int32_t* d_sel_indices_BxS,
    const int32_t* d_sel_counts, uint16_t* d_k_sel_f16,
    int B, int stride_sel, hipStream_t stream);

void launch_dsa_scores_softmax_sink_f16(
    const float* d_scores_BxHxL, const float* d_attn_sink,
    const int32_t* d_sel_counts, uint16_t* d_probs_f16,
    int B, int n_heads, int L, float scale, hipStream_t stream);

void launch_dsa_q_absorb_uk_f16(
    const uint16_t* d_q_bf16, const uint16_t* d_kv_b_f16,
    uint16_t* d_q_abs_bf16, int M, int n_heads, hipStream_t stream);

void launch_dsa_o_project_uv_f16(
    const uint16_t* d_o_lat_bf16, const uint16_t* d_kv_b_f16,
    uint16_t* d_o_v_bf16, int M, int n_heads, hipStream_t stream);

// 60-CU direct BF16->FP32 weight-stationary mHC pre-mix GEMM [M, 16384] @ [24, 16384]^T -> [M, 24] F32 (13x faster than SGEMM!)
void launch_mhc_pre_gemm_bf16_f32(
    const uint16_t* d_res_bf16, const float* d_fn_24x16384,
    float* d_out_Mx24, int M, int H, hipStream_t stream);

// Fused HCA [M, 2560] F32 split + BF16-rounded RMSNorm(q_norm, kv_norm) + wkv/wgate copy!
void launch_hca_split_qkv_comp_norm(
    const float* d_qkv_f32, const uint16_t* d_q_norm_w, const uint16_t* d_kv_norm_w,
    uint16_t* d_qr_bf16, uint16_t* d_kv_bf16, float* d_wkv_f32, float* d_wgate_f32,
    int M, float eps, hipStream_t stream);

// Hardware Batched FP16->FP32 GEMM DSA Dense Causal Attention helpers (pos < 2176, M > 8 -> 15x faster!)
void launch_dsa_dense_softmax_tile_f32(
    const float* d_scores_f32, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, int32_t pos0_strip, int k0, int k_tile, float scale, hipStream_t stream);

void launch_dsa_dense_finalize_sink_bf16(
    const float* d_o_tot_f32, const float* d_m_tot_f32, const float* d_l_tot_f32,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int B, int n_heads, hipStream_t stream);

// ---- Wave-8 FP16 fdot2 mHC Kernels (2x less HBM2 for fn_24x16384!) ----
void launch_mhc_pre_broadcast_norm_f16(
    const uint16_t* d_embed_bf16, const uint16_t* d_fn_24x16384_f16,
    const float* d_hc_scale3, const float* d_hc_base24, const uint16_t* d_norm_w_bf16,
    uint16_t* d_res_out_bf16, float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

void launch_mhc_pre_norm_fast_f16(
    const uint16_t* d_res_bf16, const uint16_t* d_fn_24x16384_f16,
    const float* d_hc_scale3, const float* d_hc_base24, const uint16_t* d_norm_w_bf16,
    float* d_mhc_ws, float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

void launch_mhc_pre_gemm_bf16_f16_f32(
    const uint16_t* d_res_bf16, const uint16_t* d_fn_24x16384_f16,
    float* d_out_Mx24, int M, int H, hipStream_t stream);

// ---- Wave-8 GPU Sparse Softmax + Top-Candidates Compaction (eliminates 805 KiB D2H + 201K CPU exp!) ----
// Extracts all vocab indices with (logit - max_l)/T > -14.0f (up to max_cands) + exact total sum_exp!
void launch_gpu_softmax_sparse_top_candidates(
    const float* d_logits_f32, int V, float temperature,
    float* d_part64_f32,             // [64] F32 scratch (reuses d_argmax_v64!)
    float* d_stats2_f32,             // [2] F32 out: [0]=max_l, [1]=total_sum_exp
    int32_t* d_cand_cnt,             // [1] I32 out: total matching count (if > max_cands, host falls back!)
    int32_t* d_cand_idx,             // [max_cands] I32 out
    float* d_cand_exp,               // [max_cands] F32 out: unnormalized exp((l - max_l)/T)
    int max_cands, hipStream_t stream);

// ---- Wave-8 Zero-Sync Mapped Pinned Host-Memory TP=4 Collectives (M<=8, 0 slc, 0 CPU sync!) ----
void launch_tp4_mapped_allreduce_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_tp4_mapped_allgather_heads_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int slab_elems, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_tp4_mapped_bcast_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int root_rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream);

// ---- User-space Vega 20 DPM Keep-Warm (4 tiny waves on uncached 64 KiB scratch -> 1725 MHz sclk & 1000 MHz mclk!) ----
void launch_ernie55_keep_dpm_warm(
    const uint32_t* d_active_flag, uint32_t* d_uc_scratch, uint32_t mask_u32, hipStream_t stream);

// ---- Commit 3: Type_HH_HPA HCA Compressed-KV & SWA-128 GEMM Helpers (ScratchSize == 0!) ----
void launch_hca_comp_softmax_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int M, int n_heads, int32_t pos0, int blk0, int blk_cnt, hipStream_t stream);

void launch_hca_comp_merge_tile_norm_f16(
    float* d_o_tot_f32, float* d_m_tot_f32, float* d_l_tot_f32,
    const uint16_t* d_o_blk_norm_f16, const float* d_m_blk_f32, const float* d_l_blk_f32,
    int n_rows, bool first_tile, hipStream_t stream);

void launch_hca_pack_swa_keys_f16(
    const uint16_t* d_swa_ring_bf16, const uint16_t* d_kv_chunk_rot, uint16_t* d_out_k_f16,
    int32_t pos0, int n_pre, int M, int n_pad, hipStream_t stream);

void launch_hca_swa_softmax_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, int32_t q_pos0, int32_t k_start_pos, hipStream_t stream);

void launch_hca_swa_softmax_batched128_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, hipStream_t stream);

void launch_hca_swa_finalize_merge_sink_f16(
    const uint16_t* d_o_swa_norm_f16, const float* d_m_swa_f32, const float* d_l_swa_f32,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int B, int n_heads, int row_off, hipStream_t stream);

// ---- Commit 4: Zero-Gather Sparse DSA MQA-64 + Type_HH_HPA FlashIndexer/Dense + Exact Rank-16 Low-Rank VHA ----
void launch_dsa_sparse_mqa64_fused_vdot2(
    const uint16_t* d_q_abs_bf16, const uint16_t* d_kv_mla_bf16, const float* d_attn_sink,
    const int32_t* d_sel_idx, const int32_t* d_sel_cnts, uint16_t* d_o_lat_bf16,
    int M, int stride_sel, hipStream_t stream);

void launch_dsa_scores_softmax_sink_f16_in(
    const uint16_t* d_scores_BxHxL_f16, const float* d_attn_sink,
    const int32_t* d_sel_counts, uint16_t* d_probs_f16,
    int B, int n_heads, int L, hipStream_t stream);

void launch_dsa_dense_softmax_tile_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, int32_t pos0_strip, int k0, int k_tile, hipStream_t stream);

void launch_flashindexer_relu_wsum_f16_in(
    const uint16_t* d_dots_f16, const float* d_w_sel_f32, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int k0, int k_cnt, int score_stride, hipStream_t stream);

void launch_vha_lowrank_proj_f16(
    const uint16_t* d_o_loc_bf16, const float* d_vha_r_16x16,
    uint16_t* d_p_loc_f16, int M, int D, hipStream_t stream);

void launch_hca_vha_lowrank_invrope_f16(
    const uint16_t* d_o_loc_bf16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, uint16_t* d_o_out_bf16,
    int M, hipStream_t stream);

void launch_dsa_vha_lowrank_lrga_f16(
    const uint16_t* d_o_loc_bf16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_bf16,
    uint16_t* d_o_out_bf16, int M, hipStream_t stream);

// ---- Wave-10 C2: Zero-Bounce FP16-Stay Attention/VHA/o_proj Kernels (0 scratch!) ----
void launch_hca_q_qknorm_rope_f16(
    uint16_t* d_q_f16, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    int M, int n_heads, float eps, hipStream_t stream);

void launch_hca_swa_finalize_merge_sink_f16_out(
    const uint16_t* d_o_swa_norm_f16, const float* d_m_swa_f32, const float* d_l_swa_f32,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_f16,
    int B, int n_heads, int row_off, hipStream_t stream);

void launch_vha_lowrank_proj_f16_in(
    const uint16_t* d_o_loc_f16, const float* d_vha_r_16x16,
    uint16_t* d_p_loc_f16, int M, int D, hipStream_t stream);

void launch_hca_vha_lowrank_invrope_f16_io(
    const uint16_t* d_o_loc_f16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, uint16_t* d_o_out_f16,
    int M, hipStream_t stream);

void launch_dsa_vha_lowrank_lrga_f16_out(
    const uint16_t* d_o_loc_bf16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_bf16,
    uint16_t* d_o_out_f16, int M, hipStream_t stream);

void launch_f16_mul_sigmoid_to_f16(
    const uint16_t* d_z_f16, const uint16_t* d_gate_bf16,
    uint16_t* d_out_f16, int total, hipStream_t stream);

void launch_f16_mul_sigmoid_f16_gate_to_f16(
    const uint16_t* d_z_f16, const uint16_t* d_gate_f16,
    uint16_t* d_out_f16, int total, hipStream_t stream);

void launch_hca_split_qkv_comp_norm_f16(
    const uint16_t* d_qkv_f16, const uint16_t* d_q_norm_w, const uint16_t* d_kv_norm_w,
    uint16_t* d_qr_f16, uint16_t* d_kv_bf16, float* d_wkv_f32, float* d_wgate_f32,
    int M, float eps, hipStream_t stream);

void launch_merge_moe_f16_bf16_rmsnorm_to_f16(
    const float* d_own_f32, const uint16_t* d_f16_slabs, int n_f16_slabs,
    const uint16_t* d_bf16_slab_or_null, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_f16, int M, int H, float eps, hipStream_t stream);

void launch_add_f16_bf16_to_bf16(
    uint16_t* d_dst_f16_in_bf16_out, const uint16_t* d_add_bf16,
    int total, hipStream_t stream);

void launch_add_f16_f16_to_bf16(
    uint16_t* d_dst_f16_in_bf16_out, const uint16_t* d_add_f16,
    int total, hipStream_t stream);

// ---- Wave-10 C3: TP4-Specialized 16-Head Sparse DSA MQA (56.5 KiB LDS Q-Resident, ScratchSize == 0, 0 P2P!) ----
void launch_dsa_copy_qrope_f16(
    const uint16_t* d_q_bf16, uint16_t* d_q_abs_f16,
    int M, int n_heads, hipStream_t stream);

void launch_dsa_dense_finalize_sink_f16(
    const float* d_o_tot_f32, const float* d_m_tot_f32, const float* d_l_tot_f32,
    const float* d_attn_sink, uint16_t* d_out_f16,
    int B, int n_heads, hipStream_t stream);

void launch_dsa_vha_lowrank_lrga_f16_io(
    const uint16_t* d_o_loc_f16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_bf16,
    uint16_t* d_o_out_f16, int M, hipStream_t stream);

void launch_dsa_mark_selected_keys(
    const int32_t* d_sel_idx, const int32_t* d_sel_cnts,
    int32_t* d_map, int n_sp, int stride_sel, int total_kv, hipStream_t stream);

void launch_dsa_compact_unique_map(
    int32_t* d_map, int32_t* d_uniq_kpos, int32_t* d_uniq_cnt,
    int total_kv, hipStream_t stream);

void launch_dsa_gather_unique_kv_f16(
    const uint16_t* d_kv_mla_bf16, const int32_t* d_uniq_kpos,
    uint16_t* d_kv_uniq_f16, int n_uniq, hipStream_t stream);

void launch_dsa_remap_sel_idx(
    const int32_t* d_sel_idx, const int32_t* d_sel_cnts, const int32_t* d_map,
    void* d_sel_remapped, int n_sp, int stride_sel, bool use_u16, hipStream_t stream);

void launch_dsa_sparse_mqa_tp4_h16(
    const uint16_t* d_q_abs_f16,      // [n_sp, 16, 576] FP16
    const uint16_t* d_kv_mla,         // [L or n_uniq, 576] FP16 (kv_is_f16) or BF16 (!kv_is_f16)
    const float* d_attn_sink_16,      // [16] F32 per-rank sink
    const void* d_sel_idx,            // [n_sp, stride_sel] uint16_t (idx_is_u16) or int32_t
    const int32_t* d_sel_cnts,        // [n_sp] I32
    uint16_t* d_o_lat_f16,            // [n_sp, 16, 512] FP16
    int n_sp, int stride_sel, bool kv_is_f16, bool idx_is_u16, hipStream_t stream);


// ---- Wave-10 C5b: single-pass K_tile = 16,384 FlashIndexer streaming top-2048 ----
void launch_flashindexer_relu_wsum_tiled_f16_in(
    const uint16_t* d_dots_f16, const float* d_w_sel_f32, float* d_tile_scores,
    int32_t pos0, int t0, int B_sc, int k_blk, int k_off_in_tile, int k_cnt, int kTileK,
    hipStream_t stream);

void launch_dsa_topk2048_streaming_merge(
    const float* d_tile_scores, const float* d_best_val_in, const int32_t* d_best_idx_in,
    float* d_best_val_out, int32_t* d_best_idx_out,
    int32_t pos0, int t0, int n_sp, int k_blk, int kTileK, hipStream_t stream);

void launch_dsa_topk2048_streaming_finalize(
    const int32_t* d_best_idx_final, int32_t* d_sel_idx_MxS, int32_t* d_sel_cnts,
    int32_t pos0, int t0, int n_sp, int stride_sel, hipStream_t stream);


// ---- Wave-10 C6: DSpark launch fusion (iota pos, noise+pos broadcast, fused fdot2 Markov+MultiMax, packed sparse softmax) ----
struct DsparkCandPair {
    int32_t idx;
    float   exp;
};
struct DsparkSparsePacketHeader {
    float   stats2[2]; // [0]=max_logit, [1]=sum_exp
    int32_t cand_cnt;
    int32_t pad;
};
static_assert(sizeof(DsparkCandPair) == 8, "DsparkCandPair must be 8 bytes");
static_assert(sizeof(DsparkSparsePacketHeader) == 16, "DsparkSparsePacketHeader must be 16 bytes");

void launch_fill_iota_i32(int32_t* d_out, int32_t start_val, int n, hipStream_t stream);

void launch_dspark_broadcast_noise_and_pos(
    const uint16_t* d_noise_emb, uint16_t* d_x_in, int32_t* d_pos_ids,
    int32_t pos0, int B, int H, hipStream_t stream);

void launch_dspark_markov_bias_multimax_f32(
    const uint16_t* d_me_bf16, const uint16_t* d_w2_VxR_f16,
    const float* d_base_logits_f32,
    const float* d_mm_ranges4, const float* d_mm_ts4,
    float* d_out_logits_f32,
    int V, int R, hipStream_t stream);

void launch_gpu_softmax_sparse_top_candidates_packed(
    const float* d_logits_f32, int V, float temperature,
    float* d_part64_f32, DsparkSparsePacketHeader* d_hdr,
    DsparkCandPair* d_cands, int max_cands, hipStream_t stream);


// ---- Wave-11 C2: single-launch TP4 HCA SWA-128 + compressed merge + FP16 DSA Phase A (0 scratch) ----
void launch_hca_tp4_swa128_merge_f16(
    const uint16_t* d_q_f16, const uint16_t* d_kv_f16,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_f16,
    int M, int32_t pos0, int32_t base_pos, hipStream_t stream);

void launch_rmsnorm_f16_io(
    const uint16_t* d_x_f16, const uint16_t* d_w_bf16, uint16_t* d_out_f16,
    int M, int D, float eps, hipStream_t stream);

void launch_dsa_prefill_kv_norm_rope_append_f16_in(
    uint16_t* d_q_f16, const uint16_t* d_kv_a_raw_f16, const uint16_t* d_kv_a_norm_w,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_mla_kv_cache_bf16, int M, int n_heads, int max_kv, float eps, hipStream_t stream);

void launch_dsa_mla_rope_q_f16_inplace(
    uint16_t* d_q_f16, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    int M, int n_heads, hipStream_t stream);

void launch_dsa_indexer_prep_neox_wht128_f16_in(
    uint16_t* d_iq_f16, const uint16_t* d_ik_raw_f16,
    const uint16_t* d_knorm_w, const uint16_t* d_knorm_b,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_ik_cache_bf16, int M, int max_kv, hipStream_t stream);

void launch_dsa_copy_qrope_f16_to_f16(
    const uint16_t* d_q_f16, uint16_t* d_q_abs_f16,
    int M, int n_heads, hipStream_t stream);

void launch_flashindexer_select_heads_f16(
    const float* d_gate_logits_f32, const float* d_bias_f32,
    const uint16_t* d_iq_f16, const uint16_t* d_iw_f16,
    uint16_t* d_q_sel_f16, float* d_w_sel_f32,
    int M, hipStream_t stream);

void launch_dsa_vha_lowrank_lrga_f16_all(
    const uint16_t* d_o_loc_f16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_f16,
    uint16_t* d_o_out_f16, int M, hipStream_t stream);

// ---- Wave-11 C3: Single-Read LDS-Staged mHC Pre-GEMM + Fused ss + Dual BF16/FP16 Output ----
void launch_mhc_pre_gemm_ss_bf16_f16_f32(
    const uint16_t* d_res_bf16, const uint16_t* d_fn_24x16384_f16,
    float* d_out_Mx25, int M, int H, hipStream_t stream);

void launch_mhc_pre_sinkhorn_from_mixes_ss(
    const uint16_t* d_res_bf16, const float* d_raw_mixes_Mx25,
    const float* d_hc_scale3, const float* d_hc_base24, const uint16_t* d_norm_w_bf16,
    float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16, uint16_t* d_layer_in_f16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream);

// ---- Wave-11 C5: Zero-Scratch HBM2 Canonical 4-Way FP16 AllReduce Sum Kernels (100% bit-exact FP32 RNE!) ----
void launch_sum4_f16_canonical(
    const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
    uint16_t* dst_f16, size_t n, hipStream_t stream);

void launch_sum4_f16_to_bf16_canonical(
    const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
    uint16_t* dst_bf16, size_t n, hipStream_t stream);

// ---- Wave-11 Omni Turbo Pillar II (K2.1/K2.2): Zero-Scratch 1 TB/s HBM2 .kvblk Slab Scatter/Gather (0 B new VRAM!) ----
struct KvBlkCopyTable {
    uint32_t count;
    uint32_t pad;
    uint32_t slab_off[64];
    uint32_t nbytes[64];
    uint64_t dev_ptr[64];
};
static_assert(sizeof(KvBlkCopyTable) <= 2048, "KvBlkCopyTable fits easily in gfx906 kernarg");

void launch_kvblk_unpack_scatter_hbm2(const uint8_t* d_slab, const KvBlkCopyTable& tbl, hipStream_t stream);
void launch_kvblk_pack_gather_hbm2(uint8_t* d_slab, const KvBlkCopyTable& tbl, hipStream_t stream);

// ---- Wave-11 Omni Turbo Pillar III (D3.1/D3.2): Deep kPingPong=16 Monotonic Mapped TP4 & 5-MI50 Decode MoE ----
void launch_tp4_mapped_allreduce_f16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_tp4_mapped_allreduce_f16_to_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_tp4_mapped_dsa_gather_qabs_bf16(
    const uint16_t* d_src_loc, uint16_t* d_dst_own_or_null,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int owner_rank, int M, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_tp4_mapped_dsa_scatter_olat_bf16(
    const uint16_t* d_src_own_or_null, uint16_t* d_dst_loc,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int owner_rank, int M, int slab_stride, uint32_t epoch, hipStream_t stream);

// Mapped 5-MI50 + Blackhole P150A Decode MoE (M <= 8, 16-slot x 7-slab monotonic ring, 0 host syncs!)
void launch_moe_mapped_publish_xrot_topk(
    const uint16_t* d_x_rot_f16, const int32_t* d_topk_ids, const float* d_topk_wts,
    uint16_t* d_moe_mapped_slabs, uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_moe_mapped_wait_xrot_copy(
    uint16_t* d_x_rot_dst, const uint16_t* d_moe_mapped_slabs, const uint32_t* d_moe_mapped_ready,
    int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_moe_mapped_publish_partial_f16(
    const float* d_y_acc_f32, uint16_t* d_moe_mapped_slabs,
    uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    int gi, int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream);

void launch_moe_mapped_reduce_slots_publish_f16(
    const float* d_y_slots_f32, uint16_t* d_moe_mapped_slabs,
    uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    int gi, int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream,
    const int32_t* d_slot_mask3 = nullptr);

void launch_moe_mapped_merge_rmsnorm_to_bf16(
    const float* d_own_f32, const uint16_t* d_moe_mapped_slabs,
    const uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    const uint16_t* d_p150_bf16_or_null, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_bf16, int own_gi, int n_gpus, int M, int H,
    int slab_stride, uint32_t epoch, float eps, hipStream_t stream,
    int wait_p150_mapped = 0);

// ---- Wave-11 Omni Turbo Pillar III (D3.3): 60-CU Split-K DSA Sparse MQA Decode (4 strict invariants, 0 scratch!) ----
void launch_dsa_sparse_mqa_decode_splitk(
    const uint16_t* d_q_in,          // [M, H_total, 576] BF16 or FP16 (H_total = n_hgroups * 16)
    const uint16_t* d_kv_mla,        // [max_kv, 576] BF16 or FP16
    const float*    d_attn_sink,     // [H_total] F32 (folded strictly ONCE in 1-wave combine!)
    const void*     d_sel_idx,       // [M, stride_sel] int32_t or uint16_t
    const int32_t*  d_sel_cnts,      // [M] int32_t
    void*           d_scratch,       // inside existing w.d_tmp_a (0 B new hipMalloc!)
    uint16_t*       d_o_out,         // [M, H_total, 512] BF16 or FP16
    int M, int n_hgroups, int stride_sel, int32_t pos0,
    bool q_is_f16, bool kv_is_f16, bool idx_is_u16, bool out_is_f16,
    hipStream_t stream);

// Direct-BF16 uint4 + v_dot2_f32_f16 persistent 60-CU FlashIndexer decode (M <= 8, >800 GB/s HBM2!)
void launch_flashindexer_decode_m8_splitk(
    const uint16_t* d_q_sel_bf16,    // [total_M, 8, 128] BF16
    const float*    d_w_sel_f32,     // [total_M, 8] F32
    const uint16_t* d_ik_cache_bf16, // [max_seq, 128] BF16
    float*          d_scores_f32,    // [B_sc, score_stride] F32
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream);

// ---- Wave-11 Omni Turbo Pillar III (D3.4): Zero-Sync On-GPU DSpark Markov+MultiMax from GPU prev_tok ----
void launch_dspark_markov_bias_multimax_from_tok_f32(
    const uint16_t* d_markov_w1_T_mapped, // [V, R] BF16 host-mapped (0 B GPU VRAM!)
    const int32_t*  d_prev_tok_ptr,       // GPU pointer to 1 int32_t prev_tok
    const uint16_t* d_w2_VxR_f16,         // [V, R] FP16
    const float*    d_base_logits_f32,    // [V]
    const float*    d_mm_ranges4,         // [4] (nullable)
    const float*    d_mm_ts4,             // [4]
    float*          d_out_logits_f32,     // [V]
    int V, int R, hipStream_t stream);

} // namespace ernie55
