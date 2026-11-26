// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure gfx906 HIP Zero-Scratch (ScratchSize == 0) Packed v_dot2_f32_f16 (fdot2) WHT128-HLUT4,
// WHT128 Transform, SiTU-GLU (beta=4.0, linear_beta=25.0), and Lossless BF16 GEMV Kernels for ERNIE-5.5.

#include "../src/ernie55_hlut4_pack.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ernie55 {

inline constexpr int kMaxGroupTokens = 8;

struct alignas(16) ExpertDispatchGroup {
    int32_t expert;                      // local expert slot [0 .. local_count) or -1
    int32_t count;                       // number of tokens routed to this expert (1 .. kMaxGroupTokens)
    int32_t rows[kMaxGroupTokens];       // input token row index [0 .. M)
    float   weights[kMaxGroupTokens];    // routed weight (renormed * per-expert learnable scale)
    int32_t slots[kMaxGroupTokens];      // canonical router top-k rank [0 .. 9] for 0-atomic deterministic reduction
};

// WHT128 forward transforms (K multiple of 128, e.g. K=2048)
void launch_wht128_bf16_to_f16(const uint16_t* d_in_bf16, const uint64_t* d_signs,
                               uint16_t* d_out_f16, int M, int K, hipStream_t stream);
void launch_wht128_f32_to_f16(const float* d_in_f32, const uint64_t* d_signs,
                              uint16_t* d_out_f16, int M, int K, hipStream_t stream);
void launch_wht128_grouped_f32_to_f16(const float* d_in_f32, const ExpertDispatchGroup* d_groups, int n_groups,
                                      const uint64_t* d_signs, uint16_t* d_out_f16, int K, hipStream_t stream,
                                      int max_c = kMaxGroupTokens);
void launch_wht128_f32_to_f32(const float* d_in_f32, const uint64_t* d_signs,
                              float* d_out_f32, int M, int K, hipStream_t stream);

// Single-expert WHT128-HLUT4 fdot2 GEMV verifier (for GATE-1 parity)
void launch_hlut4_fdot2_gemv(const uint16_t* d_x_rot_f16, const Hlut4Tile32x32* d_tiles,
                             const uint16_t* d_levels8_f16, float* d_y_f32,
                             int N, int K, hipStream_t stream);

// Grouped WHT128-HLUT4 fdot2 Gate(w1) + Up(w3) + SiTU-GLU(beta=4.0, linear_beta=25.0)
void launch_hlut4_fdot2_grouped_gu_situ(
    const uint16_t* d_x_rot_f16,
    const ExpertDispatchGroup* d_groups, int n_groups,
    const Hlut4Tile32x32* d_w1_tiles, const Hlut4Tile32x32* d_w3_tiles,
    const uint16_t* d_levels8_f16,
    float* d_z_f32, int N_inter, int K_in,
    float situ_beta, float situ_linear_beta, hipStream_t stream);

// Grouped WHT128-HLUT4 fdot2 Down(w2) + weighted scatter-accumulate into FP32 [M, N_out]
void launch_hlut4_fdot2_grouped_down_atomic(
    const uint16_t* d_z_rot_f16,
    const ExpertDispatchGroup* d_groups, int n_groups,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_accum_f32, int N_out, int K_inter, hipStream_t stream);

// Deterministic 0-atomic Grouped Down(w2) writing into per-(row, router_slot) buffer [M, 10, N_out]
void launch_hlut4_fdot2_grouped_down_slot(
    const uint16_t* d_z_rot_f16,
    const ExpertDispatchGroup* d_groups, int n_groups,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_slots_f32, int N_out, int K_inter, hipStream_t stream);

// Unpack one WHT128-HLUT4 matrix [N, K] into FP16 row-major for rocBLAS prefill crossover (M_e >= 16)
void launch_unpack_hlut4_to_f16(const Hlut4Tile32x32* d_tiles, const uint16_t* d_levels8_f16,
                                uint16_t* d_out_f16, int N, int K, hipStream_t stream);

// Prefill MoE M_e >= 16 rocblas_hgemm crossover helpers (zero scratch)
void launch_gather_rows_f16(const uint16_t* d_src_MxK, const int32_t* d_row_ids,
                            uint16_t* d_dst_CxK, int C, int K, hipStream_t stream);
void launch_situ_glu_f16_to_f32(const uint16_t* d_gate_f16, const uint16_t* d_up_f16,
                                float* d_out_f32, int C, int N,
                                float situ_beta, float situ_linear_beta, hipStream_t stream);
void launch_scatter_add_weighted_f32(const float* d_src_CxK, const int32_t* d_row_ids,
                                     const float* d_row_wts, float* d_acc_MxK,
                                     int C, int K, hipStream_t stream);
void launch_scatter_slot_weighted_f32(const float* d_src_CxK, const int32_t* d_row_ids,
                                      const int32_t* d_row_slots, const float* d_row_wts,
                                      float* d_y_slots_f32, int C, int K, hipStream_t stream);
void launch_reduce_moe_slots_f32(const float* d_y_slots_f32, float* d_y_out_f32,
                                 int M, int N_out, hipStream_t stream);

// GPU-accelerated WHT128-HLUT4 LDLQ quantizer kernel (for 5x MI50 Phase-2 sequential builder)
void launch_wht128_hlut4_ldlq_quant_gpu(
    const float* d_w_rot_NxK,       // [N, K] already WHT128-rotated FP32 weight matrix
    const float* d_rup_KxK,         // [K, K] FP32 reverse-Cholesky R_up (or nullptr for H=I)
    const float* d_levels8_f32,     // [8] Lloyd-Max positive centroids
    const float* d_thresh7_f32,     // [7] Lloyd-Max positive decision thresholds
    Hlut4Tile32x32* d_out_tiles,    // [N/32, K/32] packed 528 B tiles
    float* d_sq_sig_err2,           // [2] atomic sum of (signal^2, error^2) for exact SQNR
    int N, int K, hipStream_t stream);

// Dense / Shared-Expert SiTU-GLU on BF16 gate + up buffers -> BF16 out
void launch_situ_glu_bf16(const uint16_t* d_gate_bf16, const uint16_t* d_up_bf16,
                          uint16_t* d_out_bf16, int M, int N,
                          float situ_beta, float situ_linear_beta, hipStream_t stream);
void launch_situ_glu_f32(const float* d_gate_f32, const float* d_up_f32,
                         float* d_out_f32, int M, int N,
                         float situ_beta, float situ_linear_beta, hipStream_t stream);

// Fast zero-scratch weight-stationary uint4-vectorized BF16 / FP16(fdot2) row-major GEMV (for decode M = 1..8)
void launch_bf16_gemv_bf16(const uint16_t* d_x_bf16, const uint16_t* d_w_bf16,
                           uint16_t* d_y_bf16, int M, int N, int K, hipStream_t stream);
void launch_bf16_gemv_f32(const uint16_t* d_x_bf16, const uint16_t* d_w_bf16,
                          float* d_y_f32, int M, int N, int K, hipStream_t stream);
void launch_bf16_f32w_gemv_f32(const uint16_t* d_x_bf16, const float* d_w_f32,
                               float* d_y_f32, int M, int N, int K, hipStream_t stream);

// Fast clamped BF16 -> FP16 converter and hardware v_dot2_f32_f16 (fdot2) FP16 GEMV for M=1..8
void launch_bf16_to_f16(const uint16_t* d_in_bf16, uint16_t* d_out_f16, int total, hipStream_t stream);
void launch_f16_to_bf16(const uint16_t* d_in_f16, uint16_t* d_out_bf16, int total, hipStream_t stream);
void launch_f16_to_f32(const uint16_t* d_in_f16, float* d_out_f32, int total, hipStream_t stream);
void launch_f16_mul_sigmoid_to_bf16(const uint16_t* d_z_f16, const uint16_t* d_gate_bf16,
                                    uint16_t* d_out_bf16, int total, hipStream_t stream);
void launch_f16_gemv_bf16(const uint16_t* d_x_f16, const uint16_t* d_w_f16,
                          uint16_t* d_y_bf16, int M, int N, int K, hipStream_t stream);
void launch_f16_gemv_f32(const uint16_t* d_x_f16, const uint16_t* d_w_f16,
                         float* d_y_f32, int M, int N, int K, hipStream_t stream);
void launch_hca_grouped_wo_a_lrga_f16(const uint16_t* d_o_f16, const uint16_t* d_wo_a_f16,
                                      const uint16_t* d_gate_bf16, uint16_t* d_z_out_bf16,
                                      int M, int n_groups, int o_lora_rank, hipStream_t stream);

// Wave-8 Fused BF16-in FP16-weight LDS-tiled 2-row/warp v_dot2_f32_f16 (fdot2) GEMV (0 separate bf16->f16 launches!)
void launch_bf16x_f16w_gemv_bf16(const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                 uint16_t* d_y_bf16, int M, int N, int K, hipStream_t stream);
void launch_bf16x_f16w_gemv_f32(const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                float* d_y_f32, int M, int N, int K, hipStream_t stream);
void launch_hca_grouped_wo_a_lrga_bf16_f16(const uint16_t* d_o_bf16, const uint16_t* d_wo_a_f16,
                                           const uint16_t* d_gate_bf16, uint16_t* d_z_out_bf16,
                                           int M, int n_groups, int o_lora_rank, hipStream_t stream);
// Wave-8 Draft-Only FP8 E4M3FN Per-Row Scaled fdot2 GEMV (2x less HBM2 traffic for DSpark lm_head!)
void launch_bf16x_fp8w_gemv_f32(const uint16_t* d_x_bf16, const uint8_t* d_w_fp8,
                                const float* d_row_scale, float* d_y_f32,
                                int M, int N, int K, hipStream_t stream);

// ---- Commit 2: On-Chip gfx906 LDS SRAM M=64 Prefill MoE + Direct Weighted FP32 atomicAdd (0 HBM2 unpack, 0 d_y_slots!) ----
struct alignas(16) Ernie55MoeWorkTile {
    int32_t le;
    int32_t p_beg;
    int32_t cnt;
    int32_t pad;
};

struct alignas(16) Ernie55MoeRoutedPair {
    int32_t row;
    int32_t le;
    float   weight;
    int32_t pad;
};

void launch_ernie55_build_expert_csr_prefill(
    const int32_t* d_topk_ids, const float* d_topk_weights, const int16_t* d_global_to_local,
    int M, int max_local_slot, int tile_m, int max_tiles, int max_pairs,
    int32_t* d_csr_counts, Ernie55MoeWorkTile* d_tiles, Ernie55MoeRoutedPair* d_pairs,
    hipStream_t stream);

void launch_hlut4_prefill_situ_f16_m64(
    const uint16_t* d_x_rot_f16,
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles,
    const Ernie55MoeRoutedPair* d_pairs,
    const Hlut4Tile32x32* d_w1_tiles,
    const Hlut4Tile32x32* d_w3_tiles,
    const uint16_t* d_levels8_f16,
    uint16_t* d_z_f16,
    int N, int K, float situ_beta, float situ_linear_beta, int max_tiles_grid,
    hipStream_t stream);

void launch_wht128_f16_inplace_prefill_pairs(
    uint16_t* d_z_f16,
    const int32_t* d_csr_counts,
    const uint64_t* d_signs_i,
    int N, int max_pairs_grid,
    hipStream_t stream);

void launch_hlut4_prefill_down_atomic_m64(
    const uint16_t* d_z_f16,
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles,
    const Ernie55MoeRoutedPair* d_pairs,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_acc_f32,
    int N_down, int K_inter, int max_tiles_grid,
    hipStream_t stream);

void launch_hlut4_prefill_situ_f16_m32(
    const uint16_t* d_x_rot_f16,
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles,
    const Ernie55MoeRoutedPair* d_pairs,
    const Hlut4Tile32x32* d_w1_tiles,
    const Hlut4Tile32x32* d_w3_tiles,
    const uint16_t* d_levels8_f16,
    uint16_t* d_z_f16,
    int N, int K, float situ_beta, float situ_linear_beta, int max_tiles_grid,
    hipStream_t stream);

void launch_hlut4_prefill_down_atomic_m32(
    const uint16_t* d_z_f16,
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles,
    const Ernie55MoeRoutedPair* d_pairs,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_acc_f32,
    int N_down, int K_inter, int max_tiles_grid,
    hipStream_t stream);

} // namespace ernie55
