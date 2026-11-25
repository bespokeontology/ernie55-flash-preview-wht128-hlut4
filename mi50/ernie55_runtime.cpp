// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 / gfx906 HIP / Blackhole P150A Multi-Accelerator Runtime for baidu/ERNIE-5.5-Flash (ernie5_5_moe)
// 100% uniform WHT128-HLUT4 across all 21,504 backbone + 512 ECHO experts (0 B CPU DDR4 spill!).
// Architecture:
//   - 2-Stage Lossless BF16 Spine on MI50 dev0 (embed + L00..L21) & dev1 (L22..L43 + lm_head)
//     => ZERO intra-layer attention TP AllReduce, ZERO KV duplication, full 1M lossless BF16 KV cache!
//   - Concurrent All-Device WHT128-HLUT4 MoE across ALL MI50s (dev0..devN) + Blackhole P150A SHM IPC
//     overlapped directly with shared_experts on the owning spine GPU!
//   - Supports BOTH ECHO 2-step greedy MTP draft (Base) AND bundled DSpark probabilistic block draft (Nov 25).

#include "ernie55_runtime.hpp"
#include "ernie55_core_kernels.hpp"
#include "ernie55_hlut4_kernels.hpp"
#include "ernie55_p150_shm.hpp"
#include "safetensors_mmap.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ernie55 {

namespace {

inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in ") + what + ": " + hipGetErrorString(e));
    }
}

inline void rocblas_check(rocblas_status s, const char* what) {
    if (s != rocblas_status_success) {
        throw std::runtime_error(std::string("rocBLAS error in ") + what + ": status=" + std::to_string(int(s)));
    }
}

// Build 2D factored RoPE tables (cos_hi/sin_hi [1024, 32], cos_lo/sin_lo [1024, 32]) on host and upload to GPU
Rope2DTableGpu upload_rope2d(const std::vector<float>& inv_freq32, std::vector<void*>& allocs) {
    constexpr int kDim = 1024 * 32;
    std::vector<float> ch(kDim), sh(kDim), cl(kDim), sl(kDim);
    for (int hi = 0; hi < 1024; ++hi) {
        const double p = double(hi) * 1024.0;
        for (int i = 0; i < 32; ++i) {
            const double ang = p * double(inv_freq32[size_t(i)]);
            ch[size_t(hi * 32 + i)] = float(std::cos(ang));
            sh[size_t(hi * 32 + i)] = float(std::sin(ang));
        }
    }
    for (int lo = 0; lo < 1024; ++lo) {
        const double p = double(lo);
        for (int i = 0; i < 32; ++i) {
            const double ang = p * double(inv_freq32[size_t(i)]);
            cl[size_t(lo * 32 + i)] = float(std::cos(ang));
            sl[size_t(lo * 32 + i)] = float(std::sin(ang));
        }
    }
    Rope2DTableGpu t{};
    auto up = [&](const std::vector<float>& src) -> float* {
        float* d = nullptr;
        hip_check(hipMalloc(&d, src.size() * sizeof(float)), "hipMalloc rope2d");
        hip_check(hipMemcpy(d, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy rope2d");
        allocs.push_back(d);
        return d;
    };
    t.d_cos_hi = up(ch);
    t.d_sin_hi = up(sh);
    t.d_cos_lo = up(cl);
    t.d_sin_lo = up(sl);
    return t;
}

inline uint32_t hw_crc32c(const void* data, size_t len, uint32_t init = 0u) noexcept {
    uint64_t crc = ~uint64_t(init);
    const auto* p = static_cast<const uint8_t*>(data);
    while (len >= 8) {
        uint64_t v = 0;
        std::memcpy(&v, p, 8);
        crc = _mm_crc32_u64(crc, v);
        p += 8;
        len -= 8;
    }
    while (len > 0) {
        crc = _mm_crc32_u8(uint32_t(crc), *p++);
        --len;
    }
    return ~uint32_t(crc);
}

// Hardware F16C FP16 -> FP32 (8 elems/cycle on AMD EPYC 7542, exact IEEE-754)
inline void avx2_f16_to_f32(const uint16_t* src, float* dst, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n >= 65536)
    for (size_t i = 0; i < n8; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
    for (size_t i = n8; i < n; ++i) dst[i] = fp16_to_f32(src[i]);
}

// Hardware AVX2 FP32 vector accumulate (dst[i] += src[i])
inline void avx2_add_f32_inplace(float* dst, const float* src, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n >= 65536)
    for (size_t i = 0; i < n8; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }
    for (size_t i = n8; i < n; ++i) dst[i] += src[i];
}

// Hardware AVX2 canonical left-to-right (((p0+p1)+p2)+p3) BF16 RNE AllReduce sum
inline void avx2_sum4_bf16_canonical(const uint16_t* p0, const uint16_t* p1,
                                     const uint16_t* p2, const uint16_t* p3,
                                     uint16_t* dst, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    const __m256i v_bias = _mm256_set1_epi32(0x7FFF);
    const __m256i v_one  = _mm256_set1_epi32(1);
    const __m256i v_shuf = _mm256_setr_epi8(
        2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1,
        2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n >= 65536)
    for (size_t i = 0; i < n8; i += 8) {
        auto ld8 = [](const uint16_t* p) -> __m256 {
            __m128i u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u16), 16));
        };
        __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(ld8(p0 + i), ld8(p1 + i)), ld8(p2 + i)), ld8(p3 + i));
        __m256i u = _mm256_castps_si256(s);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), v_one);
        u = _mm256_add_epi32(u, _mm256_add_epi32(v_bias, lsb));
        __m256i p = _mm256_shuffle_epi8(u, v_shuf);
        __m128i out = _mm_unpacklo_epi64(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), out);
    }
    for (size_t i = n8; i < n; ++i) {
        float s = bf16_to_f32(p0[i]);
        s += bf16_to_f32(p1[i]);
        s += bf16_to_f32(p2[i]);
        s += bf16_to_f32(p3[i]);
        dst[i] = f32_to_bf16_rne(s);
    }
}

template <typename T>
T* host_pin_alloc(size_t count, std::vector<void*>& pin_allocs, bool zero = false) {
    if (count == 0) return nullptr;
    void* p = nullptr;
    hip_check(hipHostMalloc(&p, count * sizeof(T), hipHostMallocDefault), "hipHostMalloc");
    if (zero) std::memset(p, 0, count * sizeof(T));
    pin_allocs.push_back(p);
    return static_cast<T*>(p);
}

template <typename T>
T* host_mapped_alloc(size_t count, std::vector<void*>& pin_allocs, bool zero = false) {
    if (count == 0) return nullptr;
    void* p = nullptr;
    hip_check(hipHostMalloc(&p, count * sizeof(T), hipHostMallocMapped | hipHostMallocCoherent), "hipHostMallocMapped");
    if (zero) std::memset(p, 0, count * sizeof(T));
    pin_allocs.push_back(p);
    return static_cast<T*>(p);
}

// Exact IEEE FP8 E4M3FN encoder matching device d_fp8_e4m3fn_to_f16 (max=448.0, 0 NaNs!)
inline uint8_t f32_to_fp8_e4m3fn(float v) noexcept {
    const uint8_t sign = (v < 0.0f) ? 0x80u : 0x00u;
    float av = std::fabs(v);
    if (!(av > 0.0f)) return sign;
    if (av >= 448.0f) return uint8_t(sign | 0x7Eu); // max normal = 448.0 (exp=15, mant=6)
    // Subnormal range: < 2^-6 = 0.015625 (step 2^-9 = 1/512)
    if (av < (1.0f / 64.0f)) {
        int m = int(std::lrintf(av * 512.0f));
        if (m <= 0) return sign;
        if (m >= 8) return uint8_t(sign | 0x08u); // exp=1, mant=0
        return uint8_t(sign | uint8_t(m));
    }
    int exp2 = 0;
    float frac = std::frexp(av, &exp2); // av = frac * 2^exp2, frac in [0.5, 1.0)
    // Convert to 1.xxx * 2^(e_unbiased): mant_f = (2*frac - 1) * 8 in [0, 8)
    int e_biased = (exp2 - 1) + 7;
    int m = int(std::lrintf((2.0f * frac - 1.0f) * 8.0f));
    if (m >= 8) { m = 0; ++e_biased; }
    if (e_biased <= 0) return uint8_t(sign | 0x08u);
    if (e_biased > 15 || (e_biased == 15 && m >= 7)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (uint8_t(e_biased) << 3) | uint8_t(m));
}

template <typename T>
T* gpu_alloc(size_t count, std::vector<void*>& allocs, bool zero = false) {
    if (count == 0) return nullptr;
    T* d = nullptr;
    hip_check(hipMalloc(&d, count * sizeof(T)), "hipMalloc");
    if (zero) hip_check(hipMemset(d, 0, count * sizeof(T)), "hipMemset");
    allocs.push_back(d);
    return d;
}

template <typename T>
T* gpu_upload(const void* host_ptr, size_t count, std::vector<void*>& allocs) {
    T* d = gpu_alloc<T>(count, allocs, false);
    if (count > 0 && host_ptr) {
        hip_check(hipMemcpy(d, host_ptr, count * sizeof(T), hipMemcpyHostToDevice), "hipMemcpy H2D");
    }
    return d;
}

using SafetensorsFile = hy4::SafetensorsMmap;

const uint8_t* require_st_bytes(const SafetensorsFile& st, const std::string& name, size_t expected_bytes = 0) {
    if (!st.contains(name)) throw std::runtime_error("Missing tensor in spine shard: " + name);
    const auto& e = st.at(name);
    if (expected_bytes > 0 && e.byte_size() != expected_bytes) {
        throw std::runtime_error("Tensor size mismatch for " + name + ": got " + std::to_string(e.byte_size()) +
                                 ", expected " + std::to_string(expected_bytes));
    }
    return st.tensor_bytes(name);
}

// Convert BF16 tensor on host to F32 if stored as BF16, or copy directly if F32
std::vector<float> load_st_as_f32(const SafetensorsFile& st, const std::string& name) {
    if (!st.contains(name)) throw std::runtime_error("Missing tensor: " + name);
    const auto& e = st.at(name);
    const uint8_t* p = st.tensor_bytes(name);
    if (e.dtype == "F32") {
        const size_t n = size_t(e.byte_size()) / 4u;
        std::vector<float> out(n);
        std::memcpy(out.data(), p, size_t(e.byte_size()));
        return out;
    } else if (e.dtype == "BF16") {
        const size_t n = size_t(e.byte_size()) / 2u;
        std::vector<float> out(n);
        const auto* b = reinterpret_cast<const uint16_t*>(p);
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(b[i]);
        return out;
    }
    throw std::runtime_error("Unsupported dtype " + e.dtype + " for " + name);
}

// ---- GPU Data Structures ----
struct MhcGpuWeights {
    uint16_t* d_fn_24x16384_f16 = nullptr;  // [24, 16384] FP16 (2x less HBM2 + fdot2!)
    float*    d_scale3          = nullptr;  // [3] F32: [alpha_pre, alpha_post, alpha_res]
    float*    d_base24          = nullptr;  // [24] F32
};

struct HcaAttnGpuWeights {
    uint16_t* d_q_down      = nullptr;  // [1024, 4096] FP16 (DSpark)
    uint16_t* d_q_norm      = nullptr;  // [1024] BF16
    uint16_t* d_q_up        = nullptr;  // [32768/tp, 1024] FP16
    uint16_t* d_kv_proj     = nullptr;  // [512, 4096] FP16 (DSpark)
    uint16_t* d_kv_norm     = nullptr;  // [512] BF16
    uint16_t* d_comp_wkv    = nullptr;  // [512, 4096] FP16
    uint16_t* d_comp_wgate  = nullptr;  // [512, 4096] FP16
    uint16_t* d_qkv_comp    = nullptr;  // [2560, 4096] FP16 fused [q_down|kv_proj|comp_wkv|comp_wgate] (backbone HCA!)
    uint16_t* d_comp_norm   = nullptr;  // [512] BF16
    float*    d_comp_ape    = nullptr;  // [128, 512] F32
    float*    d_attn_sink   = nullptr;  // [64/tp] F32
    float*    d_vha_m       = nullptr;  // [64, 64] F32 (folded I + V @ U^T)
    uint16_t* d_wo_a        = nullptr;  // [8/tp, 1024, 4096] FP16
    uint16_t* d_gate_proj   = nullptr;  // [8192/tp, 1024] FP16
    uint16_t* d_o_proj      = nullptr;  // [4096, 8192/tp] FP16
};

struct DsaAttnGpuWeights {
    uint16_t* d_q_a         = nullptr;  // [1024, 4096] BF16
    uint16_t* d_q_a_norm    = nullptr;  // [1024] BF16
    uint16_t* d_q_b         = nullptr;  // [16384, 1024] BF16
    uint16_t* d_kv_a        = nullptr;  // [576, 4096] BF16
    uint16_t* d_kv_a_norm   = nullptr;  // [512] BF16
    uint16_t* d_kv_b        = nullptr;  // [28672, 512] BF16
    uint16_t* d_idx_wq_b    = nullptr;  // [8192, 1024] BF16
    uint16_t* d_idx_wk      = nullptr;  // [128, 4096] BF16
    uint16_t* d_idx_kn_w    = nullptr;  // [128] BF16
    uint16_t* d_idx_kn_b    = nullptr;  // [128] BF16
    uint16_t* d_idx_w_proj  = nullptr;  // [64, 4096] BF16
    uint16_t* d_fi_gate     = nullptr;  // [64, 4096] BF16 (FlashIndexer dynamic top-8 head router!)
    float*    d_fi_bias     = nullptr;  // [64] F32
    float*    d_attn_sink   = nullptr;  // [64] F32
    float*    d_vha_m       = nullptr;  // [64, 64] F32
    uint16_t* d_gate_proj   = nullptr;  // [16384, 1024] BF16
    uint16_t* d_o_proj      = nullptr;  // [4096, 16384] BF16
};

struct DenseMlpGpuWeights {
    uint16_t* d_gate        = nullptr;  // [14336, 4096] BF16
    uint16_t* d_up          = nullptr;  // [14336, 4096] BF16
    uint16_t* d_down        = nullptr;  // [4096, 14336] BF16
};

struct MoeSpineGpuWeights {
    uint16_t* d_fc1_lat     = nullptr;  // [2048, 4096] FP16
    uint16_t* d_fc2_lat     = nullptr;  // [4096, 2048] FP16
    uint16_t* d_lat_norm    = nullptr;  // [2048] BF16
    uint16_t* d_sh_w1       = nullptr;  // [2048, 4096] FP16
    uint16_t* d_sh_w3       = nullptr;  // [2048, 4096] FP16
    uint16_t* d_sh_w2       = nullptr;  // [4096, 2048] FP16
    uint16_t* d_router_w01_f16 = nullptr;  // [1024, 4096] FP16 (2x less HBM2 + fdot2!)
    float*    d_corr_bias   = nullptr;  // [512] F32
    float*    d_scale_param = nullptr;  // [512] F32
};

struct LayerSpineGpu {
    static constexpr int kMaxTp = 4;
    int  layer_idx     = 0;
    int  owner_dev     = 0;     // hip device owning DSA KV & primary HCA KV stripe (dev0 or dev1)
    int  owner_tp_rank = 0;     // rank in [0..tp_size) matching owner_dev
    int  tp_size       = 1;     // 4 on 5x MI50 (dev0..dev3), 1 fallback
    bool is_dsa        = false;
    bool is_dense      = false;
    bool is_echo       = false;

    MhcGpuWeights attn_mhc{};
    MhcGpuWeights mlp_mhc{};
    uint16_t* d_in_norm    = nullptr;   // [4096] BF16
    uint16_t* d_post_norm  = nullptr;   // [4096] BF16

    HcaAttnGpuWeights hca{};
    DsaAttnGpuWeights dsa{};
    DenseMlpGpuWeights dense{};
    MoeSpineGpuWeights moe{};

    // TP-sharded per-rank spine pointers (index r in [0..tp_size); linear weights stored in FP16!)
    MhcGpuWeights      attn_mhc_tp[kMaxTp]{};
    MhcGpuWeights      mlp_mhc_tp[kMaxTp]{};
    uint16_t*          d_in_norm_tp[kMaxTp]{};
    uint16_t*          d_post_norm_tp[kMaxTp]{};
    HcaAttnGpuWeights  hca_tp[kMaxTp]{};
    DsaAttnGpuWeights  dsa_tp[kMaxTp]{};
    DenseMlpGpuWeights dense_tp[kMaxTp]{};
    MoeSpineGpuWeights moe_tp[kMaxTp]{};

    // ECHO (L43) extra tensors
    uint16_t* d_echo_enorm     = nullptr;
    uint16_t* d_echo_hnorm     = nullptr;
    uint16_t* d_echo_eproj     = nullptr;
    uint16_t* d_echo_hproj     = nullptr;
    float*    d_echo_hc_fn     = nullptr;  // [4, 16384] F32
    float*    d_echo_hc_scale  = nullptr;  // [1] F32
    float*    d_echo_hc_base   = nullptr;  // [4] F32
    uint16_t* d_echo_head_norm = nullptr;  // [4096] BF16

    // KV Cache:
    // HCA: tiny (10 MiB/L @ 1M = 0.36 GiB total for 37L) replicated across TP ranks for zero-comm HCA attn!
    // Primary aliases (d_hca_comp_kv etc.) point to owner_tp_rank for save_kvblk_ex / dump_kv_raw_bytes!
    uint16_t* d_hca_comp_kv_tp[kMaxTp]{};
    uint16_t* d_hca_swa_ring_tp[kMaxTp]{};
    float*    d_hca_state_kv_tp[kMaxTp]{};
    float*    d_hca_state_sc_tp[kMaxTp]{};
    uint16_t* d_hca_snap_swa_tp[kMaxTp]{};
    float*    d_hca_snap_kv_tp[kMaxTp]{};
    float*    d_hca_snap_sc_tp[kMaxTp]{};

    uint16_t* d_hca_comp_kv    = nullptr;
    uint16_t* d_hca_swa_ring   = nullptr;
    float*    d_hca_state_kv   = nullptr;
    float*    d_hca_state_sc   = nullptr;
    uint16_t* d_hca_snap_swa   = nullptr;
    float*    d_hca_snap_kv    = nullptr;
    float*    d_hca_snap_sc    = nullptr;
    // DSA: mla_kv [max_kv, 576] BF16 + ik_cache [max_kv, 128] BF16 (1 copy ONLY on owner_dev — zero replication!)
    uint16_t* d_dsa_mla_kv     = nullptr;
    uint16_t* d_dsa_ik_cache   = nullptr;

    // Tiny host copies (12 KiB/layer) for zero-round-trip EPYC TP4 Router + WHT128 + RMSNorm!
    std::array<float, kNumRoutedExperts> h_corr_bias{};
    std::array<float, kNumRoutedExperts> h_scale_param{};
    std::array<float, kLatentDim>        h_lat_norm{};
    std::array<uint64_t, kLatentDim / 64> h_signs_in{};
};

// One per MI50 GPU in the topology (dev0, dev1, dev2, dev3, dev4[, dev5])
struct Mi50ExpertWorker {
    int hip_dev = 0;
    int bb_begin = 0;
    int bb_end   = 0;
    int echo_begin = 0;
    int echo_end   = 0;
    std::vector<int> bb_global_ids;  // strided hot-expert + shadow-cached GDDR6 assignment across all MI50s!
    int bb_base_count = 0;           // core [0..317] expert count (DSpark L43..L45 loads ONLY these!)
    int bb_count()   const noexcept { return !bb_global_ids.empty() ? int(bb_global_ids.size()) : (bb_end - bb_begin); }
    int echo_count() const noexcept { return echo_end - echo_begin; }

    hipStream_t stream    = nullptr;
    rocblas_handle blas   = nullptr;
    hipStream_t stream_sh = nullptr;     // secondary stream for concurrent shared_experts overlap!
    rocblas_handle blas_sh = nullptr;
    hipEvent_t ev_sh_start = nullptr;
    hipEvent_t ev_sh_done  = nullptr;
    std::vector<void*> allocs;

    uint16_t* d_levels8_f16 = nullptr;
    std::array<uint64_t*, kTotalLayersWithEcho> d_signs_in{};
    std::array<uint64_t*, kTotalLayersWithEcho> d_signs_inter{};
    std::array<Hlut4Tile32x32*, kTotalLayersWithEcho> d_w1{};
    std::array<Hlut4Tile32x32*, kTotalLayersWithEcho> d_w3{};
    std::array<Hlut4Tile32x32*, kTotalLayersWithEcho> d_w2{};

    // MoE dispatch scratch on this GPU (up to max_chunk rows)
    uint16_t* d_x_rot_f16 = nullptr;        // [max_chunk, 2048] FP16
    ExpertDispatchGroup* d_groups = nullptr;// [max_groups]
    ExpertDispatchGroup* h_groups_pin = nullptr; // [max_groups] pinned H2D!
    float*    d_z_f32     = nullptr;        // [max_groups * 16, 2048] F32
    uint16_t* d_z_rot_f16 = nullptr;        // [max_groups * 16, 2048] FP16
    float*    d_y_slots_f32 = nullptr;      // [max_chunk, 10, 2048] F32 (0-atomic canonical slot buffer!)
    float*    d_y_acc_f32 = nullptr;        // [max_chunk, 2048] F32
    float*    h_y_partial_pin = nullptr;    // [max_chunk, 2048] F32 pinned!

    // M_e >= 16 rocblas_hgemm prefill crossover scratch (22 MiB/GPU!)
    uint16_t* d_w_unpack_f16 = nullptr;     // [2048, 2048] FP16
    uint16_t* d_xg_f16       = nullptr;     // [max_chunk, 2048] FP16
    float*    d_g_f32        = nullptr;     // [max_chunk, 2048] F32 (lossless FP32 before SiTU-GLU!)
    float*    d_u_f32        = nullptr;     // [max_chunk, 2048] F32
    float*    d_yh_f32       = nullptr;     // [max_chunk, 2048] F32
    int32_t*  d_row_ids      = nullptr;     // [max_chunk * kTopKExperts] I32
    int32_t*  d_row_slots    = nullptr;     // [max_chunk * kTopKExperts] I32
    float*    d_row_wts      = nullptr;     // [max_chunk * kTopKExperts] F32
    int32_t*  h_row_ids_pin  = nullptr;     // [max_chunk * kTopKExperts] I32 pinned!
    int32_t*  h_row_slots_pin= nullptr;     // [max_chunk * kTopKExperts] I32 pinned!
    float*    h_row_wts_pin  = nullptr;     // [max_chunk * kTopKExperts] F32 pinned!

    // Pre-allocated zero-heap flat MoE dispatch buckets (eliminates unordered_map per layer!)
    std::array<int32_t, kNumRoutedExperts> b_head{};
    std::array<int32_t, kNumRoutedExperts> b_cnt{};
    std::vector<int32_t> b_next;
    std::vector<int32_t> b_rows;
    std::vector<int32_t> b_slots;
    std::vector<float>   b_wts;
    std::vector<int32_t> active_exps;
    std::vector<ExpertDispatchGroup> h_groups;
    struct HgemmPlan { int le; int off; int cnt; };
    std::vector<HgemmPlan> hgemm_plans;

    // TP=5 vocab-sharded lm_head across all MI50s (FP16 fdot2 lossless verify + FP8 E4M3FN draft!)
    int       lm_v_begin       = 0;
    int       lm_v_end         = 0;
    int       lm_v_count() const noexcept { return lm_v_end - lm_v_begin; }
    uint16_t* d_lm_head_shard  = nullptr; // [lm_v_count, 4096] FP16 (10b mantissa > BF16 7b!)
    uint8_t*  d_lm_head_fp8    = nullptr; // [lm_v_count, 4096] FP8 E4M3FN (draft-only, 2x less HBM2!)
    float*    d_lm_head_fp8_scale = nullptr; // [lm_v_count] F32 per-row FP8 scale
    uint16_t* d_lm_x_in        = nullptr; // [64, 4096] BF16
    uint16_t* d_lm_x_f16       = nullptr; // [64, 4096] FP16 (for M>8 rocblas_gemm_ex!)
    float*    d_lm_logits_f32  = nullptr; // [64, lm_v_count] F32
    float*    h_lm_logits_pin  = nullptr; // [64, lm_v_count] F32 pinned!
    float*    d_mm_ranges4     = nullptr; // [4] F32
    float*    d_mm_ts4         = nullptr; // [4] F32
    // Mapped host-memory TP=4 device pointers (zero-sync M<=8 collectives!)
    uint16_t* d_tp_mapped_slabs = nullptr;
    uint32_t* d_tp_mapped_ready = nullptr;
    uint32_t* d_tp_mapped_done  = nullptr;

    // Spine-device extra scratch (allocated on all TP spine ranks dev0..dev3!)
    Rope2DTableGpu yarn_rope{};
    Rope2DTableGpu dsa_rope{};
    uint16_t* d_x_f16         = nullptr;  // [max_chunk, 16384] FP16 (16 MiB)
    float*    d_y_f32         = nullptr;  // [max_chunk, 36864] F32 on DSA owners (72 MiB), [max_chunk, 16384] (32 MiB) on dev2/dev3
    uint16_t* d_hca_kcomp_f16 = nullptr;  // [8192, 512] FP16 (8 MiB, covers full 1M HCA compressed blocks!)
    float*    d_hca_o_tot_f32 = nullptr;  // [max_chunk * 16, 512] F32 (16 MiB)
    float*    d_hca_ml_f32    = nullptr;  // [max_chunk * 16 * 4] F32 (128 KiB: m_tot, l_tot, m_blk, l_blk)
    uint16_t* d_dsa_ksel_f16  = nullptr;  // [16, 2176, 576] FP16 (38.25 MiB, DSA owners dev0/dev1 only!)
    // Dedicated secondary-stream shared_experts workspace (zero race with routed MoE on w.stream!)
    uint16_t* d_sh_x_f16      = nullptr;  // [max_chunk, 4096] FP16 (4 MiB)
    float*    d_sh_y_f32      = nullptr;  // [max_chunk, 4096] F32 (8 MiB)
    uint16_t* d_sh_tmp_a      = nullptr;  // [max_chunk, 2048] BF16 (2 MiB)
    uint16_t* d_sh_tmp_b      = nullptr;  // [max_chunk, 2048] BF16 (2 MiB)
    uint16_t* d_sh_out        = nullptr;  // [max_chunk, 4096] BF16 (4 MiB)
    float*    d_mhc_ws        = nullptr;  // [max_chunk, 25, 4] F32 (split-K mHC & prefill mixes)
    float*    d_argmax_v64    = nullptr;  // [64] F32
    int32_t*  d_argmax_i64    = nullptr;  // [64] I32
    uint16_t* d_res_a         = nullptr;  // [max_chunk, 4, 4096] BF16
    uint16_t* d_res_b         = nullptr;  // [max_chunk, 4, 4096] BF16
    float*    d_post_mix      = nullptr;  // [max_chunk, 4] F32
    float*    d_comb_mix      = nullptr;  // [max_chunk, 16] F32
    uint16_t* d_x_in          = nullptr;  // [max_chunk, 4096] BF16
    uint16_t* d_sub_out       = nullptr;  // [max_chunk, 4096] BF16
    uint16_t* d_tmp_a         = nullptr;  // [max_chunk, 32768] BF16
    uint16_t* d_tmp_b         = nullptr;  // [max_chunk, 16384] BF16 (32768 on dev1 for DSpark 64-head!)
    uint16_t* d_tmp_c         = nullptr;  // [max_chunk, 36864] BF16 on DSA owners, [max_chunk, 9216] on dev2/dev3
    uint16_t* d_tmp_d         = nullptr;  // [max_chunk, 16384] BF16
    uint16_t* d_qr            = nullptr;  // [max_chunk, 1024] BF16
    uint16_t* d_kv_tmp        = nullptr;  // [max_chunk, 576] BF16
    uint16_t* d_h_lat         = nullptr;  // [max_chunk, 2048] BF16
    float*    d_f32_tmp0      = nullptr;  // [max_chunk, 2048] F32
    float*    d_f32_tmp1      = nullptr;  // [max_chunk, 1024] F32
    int32_t*  d_pos_ids       = nullptr;  // [max_chunk]
    int32_t*  d_tok_ids       = nullptr;  // [max_chunk]
    int32_t*  d_topk_ids      = nullptr;  // [max_chunk, 10]
    float*    d_topk_wts      = nullptr;  // [max_chunk, 10]
    float*    d_idx_scores    = nullptr;  // [max_kv] F32 (DSA owners only)
    float*    d_fi_gates      = nullptr;  // [max_chunk, 64] F32 (DSA owners only)
    uint16_t* d_fi_q8         = nullptr;  // [max_chunk, 8, 128] BF16 (DSA owners only)
    float*    d_fi_w8         = nullptr;  // [max_chunk, 8] F32 (DSA owners only)
    int32_t*  d_sel_idx       = nullptr;  // [max_chunk * 2176] I32 (DSA owners only)
    int32_t*  d_sel_cnts      = nullptr;  // [max_chunk] I32 (DSA owners only)
    size_t    idx_scores_cap  = 0;        // floats allocated in d_idx_scores (DSA owners only)
    float*    d_logits_f32    = nullptr;  // [8, vocab_size] F32 (only on dev1!)
};

// Bundled 3L MoE DSpark Draft Layer (Preview-BF16: model.mtp.{0,1,2} -> layers 43, 44, 45!)
struct DSparkLayerGpu {
    int layer_idx = 43;               // 43, 44, 45 (uniform WHT128-HLUT4 experts across 5x MI50 + 2x P150A!)
    MhcGpuWeights attn_mhc{};
    MhcGpuWeights mlp_mhc{};
    uint16_t* d_in_norm    = nullptr;
    uint16_t* d_post_norm  = nullptr;
    HcaAttnGpuWeights swa_attn{};     // SWA-only HCA attention (plain RoPE at rope_theta=8,000,000!)
    MoeSpineGpuWeights moe{};         // Full 512-expert WHT128-HLUT4 + shared expert MoE!
    uint16_t* d_swa_ring   = nullptr; // [256, 512] BF16 (256-slot ring prevents context/query collision!)
    uint16_t* d_swa_snap   = nullptr; // [256, 512] BF16 prompt-boundary snapshot for warm-prefix reuse
};

} // namespace

struct Ernie55Runtime::Impl {
    Ernie55RuntimeOptions opt;
    Ernie55ModelConfig mcfg;
    Ernie55TopologyPlan topo;
    Ernie55Tokenizer tok;

    bool is_init = false;
    int split_layer = 22;   // dev0 owns L00..L21 (3 DSA), dev1 owns L22..L43 (3 DSA + DSpark)
    int spine_dev0  = 0;
    int spine_dev1  = 0;
    int tp_size     = 1;    // 4 on 5x MI50 (dev0..dev3), 1 fallback
    int tp_devs[LayerSpineGpu::kMaxTp] = {0, 0, 0, 0};
    static constexpr int kOmpThreads = 24;  // Cap below 32 physical EPYC cores -> zero SMT/futex contention with ROCm & P150A!
    int max_chunk   = 1024;
    int max_groups  = 512;
    int seq_len     = 0;
    int echo_seq_len = 0;
    std::array<int8_t,  kNumRoutedExperts> bb_owner_gpu{};  // global expert -> GPU index [0..gpus.size()), -1 if P150A
    std::array<int16_t, kNumRoutedExperts> bb_local_slot{}; // global expert -> local slot on owner GPU

    // Exact warm-prefix KV tracking + prompt-boundary HCA/DSpark snapshot
    std::vector<int32_t> cached_tokens;        // Exact tokens currently committed in VRAM KV [0 .. seq_len)
    std::vector<int32_t> ckpt_prompt_tokens;   // Exact prompt tokens at last captured HCA+DSpark snapshot
    int ckpt_prompt_len = 0;                   // 0 = no snapshot valid

    // Optional router-frequency permutation (loaded from weight_root/router_perm.bin if present!)
    bool has_router_perm = false;
    std::array<std::array<uint16_t, kNumRoutedExperts>, kTotalLayersWithEcho> slot_to_raw{};

    // Profiling telemetry & live prefill progress callback
    bool profiling_enabled = false;
    Ernie55ProfileStats prof{};
    PrefillProgressCallback prefill_progress_cb{};
    std::vector<int16_t>* router_capture_dst = nullptr;
    int router_capture_stride = 0;

    void prof_sync_tp(int tp, bool is_echo, int owner_dev) {
        if (!profiling_enabled) return;
        int prev_dev = 0;
        hip_check(hipGetDevice(&prev_dev), "prof get dev");
        for (int r = 0; r < tp; ++r) {
            auto& w = *dev_map[is_echo ? owner_dev : tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "prof sync dev");
            hip_check(hipStreamSynchronize(w.stream), "prof sync");
        }
        hip_check(hipSetDevice(prev_dev), "prof restore dev");
    }

    // Pinned host allocations (freed cleanly via hipHostFree in shutdown!)
    std::vector<void*> pin_allocs;
    uint16_t* h_lat_rot_f16_pin   = nullptr;  // [max_chunk, 2048]
    int32_t*  h_topk_ids_pin      = nullptr;  // [max_chunk, 10]
    float*    h_topk_wts_pin      = nullptr;  // [max_chunk, 10]
    float*    h_y_sum_f32_pin     = nullptr;  // [max_chunk, 2048]
    uint16_t* h_hop_bf16_pin      = nullptr;  // [max_chunk, 16384]
    uint16_t* h_echo_res_pin      = nullptr;  // [max_chunk, 16384]
    int32_t*  h_sel_idx_batch     = nullptr;  // [max_chunk, 2176]
    int32_t*  h_sel_cnts_batch    = nullptr;  // [max_chunk]
    int32_t*  h_pos_pin           = nullptr;  // [max_chunk]
    int32_t*  h_tok_pin           = nullptr;  // [max_chunk]
    int32_t*  h_dspark_pos_pin    = nullptr;  // [max_chunk]
    uint16_t* h_dspark_anchor_pin = nullptr;  // [4096]
    int32_t*  h_echo_pos_pin      = nullptr;  // [1]
    uint16_t* h_echo_emb_pin      = nullptr;  // [4096]
    uint16_t* h_lm_x_pin          = nullptr;  // [64, 4096] BF16 pinned for TP=5 lm_head!
    float*    h_lm_full_pin       = nullptr;  // [8, vocab_size] F32 pinned for DSpark TP=5 lm_head H2D!
    static constexpr uint64_t kKvSlabBytes = 64ULL * 1024ULL * 1024ULL;
    uint8_t*  h_kv_pin[2][2]    = {{nullptr, nullptr}, {nullptr, nullptr}}; // 2 stripes x 2 ping-pong 64 MiB slabs
    // TP=4 host-staged pinned AllReduce / AllGather staging buffers (100% 1D contiguous DMA — ZERO hipMemcpy2DAsync, ZERO P2P!)
    uint16_t* h_tp_ar_in_pin[LayerSpineGpu::kMaxTp]  = {}; // [max_chunk, 4096] BF16 per rank
    uint16_t* h_tp_ar_out_pin   = nullptr;                 // [max_chunk, 4096] BF16 canonical sum
    uint16_t* h_tp_oattn_pin    = nullptr;                 // [tp_size, max_chunk, h_loc, 512] BF16 rank-major contiguous
    uint16_t* h_tp_qabs_pin     = nullptr;                 // [max_chunk, 64, 576] BF16 token-major DSA q_abs
    uint16_t* h_tp_qabs_rm_pin  = nullptr;                 // [tp_size, max_chunk, 16, 576] BF16 rank-major contiguous
    float*    h_tp_rtr_in_pin[LayerSpineGpu::kMaxTp] = {}; // [max_chunk, 256] F32 contiguous per rank
    uint16_t* h_tp_fc1_in_pin[LayerSpineGpu::kMaxTp] = {}; // [max_chunk, 512] BF16 contiguous per rank
    uint16_t* h_tp_fc2_in_pin[LayerSpineGpu::kMaxTp] = {}; // [max_chunk, 512] BF16 contiguous per rank
    // Wave-8 Zero-Sync Mapped Pinned Host-Memory TP=4 Ring Arena (4 slots x 4 ranks x 128 KiB = 2 MiB, 0 slc, 0 CPU sync!)
    static constexpr int kMappedSlabStride = 8 * 16 * 512; // 65,536 halfs (128 KiB/rank/slot)
    uint16_t* h_tp_mapped_slabs = nullptr;
    uint32_t* h_tp_mapped_ready = nullptr;
    uint32_t* h_tp_mapped_done  = nullptr;
    uint32_t  tp_mapped_epoch   = 0u;
    // Wave-8 GPU Sparse Softmax + Top-Candidates Compaction Scratch (on spine_dev1 + tiny pinned readout!)
    static constexpr int kGpuMaxCands = 4096;
    float*    d_dspark_sm_stats2     = nullptr;
    int32_t*  d_dspark_cand_cnt      = nullptr;
    int32_t*  d_dspark_cand_idx      = nullptr;
    float*    d_dspark_cand_exp      = nullptr;
    float*    h_dspark_sm_stats2_pin = nullptr;
    int32_t*  h_dspark_cand_cnt_pin  = nullptr;
    int32_t*  h_dspark_cand_idx_pin  = nullptr;
    float*    h_dspark_cand_exp_pin  = nullptr;

    std::vector<std::unique_ptr<Mi50ExpertWorker>> gpus;
    std::unordered_map<int, Mi50ExpertWorker*> dev_map;

    // Shared root weights
    uint16_t* d_embed_bf16    = nullptr;  // on spine_dev0
    uint16_t* d_lm_head_bf16  = nullptr;  // on spine_dev1
    uint16_t* d_final_norm    = nullptr;  // on spine_dev1
    float*    d_hc_head_fn    = nullptr;  // [4, 16384] F32 on spine_dev1
    float*    d_hc_head_scale = nullptr;  // [1] F32 on spine_dev1
    float*    d_hc_head_base  = nullptr;  // [4] F32 on spine_dev1
    float*    d_mm_ranges4    = nullptr;  // [4] F32 on spine_dev1
    float*    d_mm_ts4        = nullptr;  // [4] F32 on spine_dev1
    float     h_mm_ranges4[4] = {};
    float     h_mm_ts4[4]     = {};
    bool      has_multimax    = false;

    std::array<LayerSpineGpu, kTotalLayersWithEcho> layers{};
    bool has_echo_l43 = false;

    // Stashed pre-hc_head 4-stream residual [4, 4096] BF16 for ECHO L43
    std::vector<uint16_t> h_last_backbone_res_4xH;

    // Optional Bundled 3L MoE DSpark Probabilistic Draft state (Preview-BF16!)
    bool has_dspark       = false;
    bool use_flashindexer = true;
    int  dspark_block_size = 5;
    float dspark_conf_tau  = 0.0f;
    DSparkHeadsCpu dspark_heads;
    uint16_t* d_dspark_main_proj   = nullptr; // [4096, 4096*3] BF16 on spine_dev1
    uint16_t* d_dspark_main_norm   = nullptr; // [4096] BF16
    uint16_t* d_dspark_norm        = nullptr; // [4096] BF16
    float*    d_dspark_hc_fn       = nullptr;
    float*    d_dspark_hc_scale    = nullptr;
    float*    d_dspark_hc_base     = nullptr;
    uint16_t* d_dspark_aux_cat     = nullptr; // [max_chunk, 3, 4096] BF16 on spine_dev1
    uint16_t* d_dspark_noise_emb   = nullptr; // [4096] BF16 cached embed_tokens[noise_token_id]
    uint16_t* d_dspark_markov_w2   = nullptr; // [201216, 256] BF16 on spine_dev1 (<0.12ms GEMV!)
    uint16_t* d_dspark_me_bf16     = nullptr; // [256] BF16
    float*    d_dspark_bias_f32    = nullptr; // [201216] F32
    float*    d_dspark_step_logits = nullptr; // [201216] F32
    std::vector<uint16_t> h_dspark_markov_w1_T; // [201216, 256] BF16 on host RAM
    std::vector<DSparkLayerGpu> dspark_layers;

    DraftMethod draft_method = DraftMethod::kNone;

    // Blackhole P150A SHM IPC
    Ernie55P150ShmMapping p150_shm;
    int p150_exp_begin = 512;
    int p150_exp_end   = 512;
    bool p150_ok = false;

    // Host staging buffers
    std::vector<uint16_t> h_lat_rot_f16;
    std::vector<float>    h_lat_rot_f32;
    std::vector<int32_t>  h_topk_ids;
    std::vector<float>    h_topk_wts;
    std::vector<float>    h_y_sum_f32;
    std::vector<float>    h_idx_scores;
    std::vector<int32_t>  h_sel_idx;

    // Cumulative telemetry
    uint64_t total_prefill_tokens = 0;
    uint64_t total_decode_tokens  = 0;
    uint64_t total_spec_steps     = 0;
    uint64_t total_spec_drafted   = 0;
    uint64_t total_spec_accepted  = 0;

    // ---- GEMM / GEMV Dispatcher: Resident FP16 spine weights (ZERO runtime weight upcast!),
    //      Hardware v_dot2_f32_f16 (fdot2) GEMV for M<=8 (bit-exact DSpark K=0..7),
    //      Constant pad_M=512 strip-tiled hardware rocblas_gemm_ex for M>8 (100% 137==512==1024 bit-exact!) ----
    static void f16_linear_bf16_ex(rocblas_handle blas, hipStream_t stream,
                                   uint16_t* d_x_f16, float* d_y_f32,
                                   const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                   uint16_t* d_y_bf16, int M, int N, int K, int pad_M = 512) {
        if (M <= 8) {
            launch_bf16x_f16w_gemv_bf16(d_x_bf16, d_w_f16, d_y_bf16, M, N, K, stream);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            for (int m0 = 0; m0 < M; m0 += pad_M) {
                const int cur = std::min(pad_M, M - m0);
                launch_bf16_to_f16(d_x_bf16 + size_t(m0) * size_t(K), d_x_f16, cur * K, stream);
                if (cur < pad_M) {
                    hip_check(hipMemsetAsync(d_x_f16 + size_t(cur) * size_t(K), 0,
                                             size_t(pad_M - cur) * size_t(K) * sizeof(uint16_t), stream), "zero f16 pad");
                }
                rocblas_check(rocblas_gemm_ex(
                    blas, rocblas_operation_transpose, rocblas_operation_none,
                    N, pad_M, K, &alpha,
                    d_w_f16, rocblas_datatype_f16_r, K,
                    d_x_f16, rocblas_datatype_f16_r, K,
                    &beta,
                    d_y_f32, rocblas_datatype_f32_r, N,
                    d_y_f32, rocblas_datatype_f32_r, N,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "f16_linear_bf16 gemm_ex");
                launch_f32_to_bf16(d_y_f32, d_y_bf16 + size_t(m0) * size_t(N), cur * N, stream);
            }
        }
    }

    static void f16_linear_f32_ex(rocblas_handle blas, hipStream_t stream,
                                  uint16_t* d_x_f16,
                                  const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                  float* d_y_f32, int M, int N, int K, int pad_M = 512) {
        if (M <= 8) {
            launch_bf16x_f16w_gemv_f32(d_x_bf16, d_w_f16, d_y_f32, M, N, K, stream);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            for (int m0 = 0; m0 < M; m0 += pad_M) {
                const int cur = std::min(pad_M, M - m0);
                launch_bf16_to_f16(d_x_bf16 + size_t(m0) * size_t(K), d_x_f16, cur * K, stream);
                if (cur < pad_M) {
                    hip_check(hipMemsetAsync(d_x_f16 + size_t(cur) * size_t(K), 0,
                                             size_t(pad_M - cur) * size_t(K) * sizeof(uint16_t), stream), "zero f16 pad");
                }
                float* dst_strip = d_y_f32 + size_t(m0) * size_t(N);
                rocblas_check(rocblas_gemm_ex(
                    blas, rocblas_operation_transpose, rocblas_operation_none,
                    N, pad_M, K, &alpha,
                    d_w_f16,   rocblas_datatype_f16_r, K,
                    d_x_f16,   rocblas_datatype_f16_r, K,
                    &beta,
                    dst_strip, rocblas_datatype_f32_r, N,
                    dst_strip, rocblas_datatype_f32_r, N,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "f16_linear_f32 gemm_ex");
            }
        }
    }

    static void bf16_linear_bf16(Mi50ExpertWorker& g, const uint16_t* d_x, const uint16_t* d_w_f16,
                                 uint16_t* d_y, int M, int N, int K) {
        f16_linear_bf16_ex(g.blas, g.stream, g.d_x_f16, g.d_y_f32, d_x, d_w_f16, d_y, M, N, K, 512);
    }

    static void bf16_linear_f32(Mi50ExpertWorker& g, const uint16_t* d_x, const uint16_t* d_w_f16,
                                float* d_y, int M, int N, int K) {
        f16_linear_f32_ex(g.blas, g.stream, g.d_x_f16, d_x, d_w_f16, d_y, M, N, K, 512);
    }

    // ---- Fast Multi-CU FP16 fdot2 mHC Pre-Norm Helper (96-block split-K for M<=8, direct fdot2 GEMM for M>8!) ----
    void run_mhc_pre_norm_fast(Mi50ExpertWorker& g, const uint16_t* d_res,
                               const MhcGpuWeights& mhc, const uint16_t* d_norm_w, int M) {
        if (M <= 8) {
            launch_mhc_pre_norm_fast_f16(d_res, mhc.d_fn_24x16384_f16, mhc.d_scale3, mhc.d_base24, d_norm_w,
                                         g.d_mhc_ws, g.d_post_mix, g.d_comb_mix, g.d_x_in,
                                         M, kHiddenSize, mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
        } else {
            launch_mhc_pre_gemm_bf16_f16_f32(d_res, mhc.d_fn_24x16384_f16, g.d_mhc_ws, M, kHiddenSize, g.stream);
            launch_mhc_pre_sinkhorn_from_mixes(
                d_res, g.d_mhc_ws, mhc.d_scale3, mhc.d_base24, d_norm_w,
                g.d_post_mix, g.d_comb_mix, g.d_x_in,
                M, kHiddenSize, mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
        }
    }

    // ---- Canonical SDMA Host-Staged TP=4 AllReduce (11 GB/s SDMA + AVX2 canonical BF16 sum, 0 PCIe spin!) ----
    void tp_allreduce_bf16(int M, int H, uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m)) {
        if (tp_size <= 1) return;
        const auto t_c0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const size_t n = size_t(M) * size_t(H);
        const size_t nb = n * sizeof(uint16_t);
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar d2h dev");
            hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r], w.*src_m, nb, hipMemcpyDeviceToHost, w.stream), "ar d2h");
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar sync dev");
            hip_check(hipStreamSynchronize(w.stream), "ar sync");
        }
        const uint16_t* p0 = h_tp_ar_in_pin[0];
        const uint16_t* p1 = h_tp_ar_in_pin[1];
        const uint16_t* p2 = (tp_size > 2) ? h_tp_ar_in_pin[2] : nullptr;
        const uint16_t* p3 = (tp_size > 3) ? h_tp_ar_in_pin[3] : nullptr;
        uint16_t* dst = h_tp_ar_out_pin;
        if (tp_size == 4) {
            avx2_sum4_bf16_canonical(p0, p1, p2, p3, dst, n, kOmpThreads);
        } else {
            #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (n >= 65536)
            for (size_t i = 0; i < n; ++i) {
                float s = 0.0f;
                for (int r = 0; r < tp_size; ++r) s += bf16_to_f32(h_tp_ar_in_pin[r][i]);
                dst[i] = f32_to_bf16_rne(s);
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar h2d dev");
            hip_check(hipMemcpyAsync(w.*dst_m, dst, nb, hipMemcpyHostToDevice, w.stream), "ar h2d");
        }
        if (profiling_enabled) {
            prof_sync_tp(tp_size, false, spine_dev0);
            prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0).count();
        }
    }

    // ---- 100% 1D Contiguous Rank-Major Head AllGather (Self D2D + 3-peer SDMA H2D, 25% less PCIe!) ----
    void tp_allgather_heads_bf16(int M, int total_heads, int head_dim,
                                 uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m),
                                 uint16_t* host_pin_buf) {
        if (tp_size <= 1) return;
        const auto t_c0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int h_loc = total_heads / tp_size;
        const size_t slab_elems = size_t(M) * size_t(h_loc) * size_t(head_dim);
        const size_t slab_b = slab_elems * sizeof(uint16_t);
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ag d2h dev");
            hip_check(hipMemcpyAsync(host_pin_buf + size_t(r) * slab_elems, w.*src_m,
                                     slab_b, hipMemcpyDeviceToHost, w.stream), "ag 1d d2h");
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ag sync dev");
            hip_check(hipStreamSynchronize(w.stream), "ag sync");
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ag h2d dev");
            hip_check(hipMemcpyAsync(w.*dst_m + size_t(r) * slab_elems, w.*src_m,
                                     slab_b, hipMemcpyDeviceToDevice, w.stream), "ag self d2d");
            if (r > 0) {
                hip_check(hipMemcpyAsync(w.*dst_m, host_pin_buf,
                                         size_t(r) * slab_b, hipMemcpyHostToDevice, w.stream), "ag 1d h2d lo");
            }
            if (r + 1 < tp_size) {
                hip_check(hipMemcpyAsync(w.*dst_m + size_t(r + 1) * slab_elems,
                                         host_pin_buf + size_t(r + 1) * slab_elems,
                                         size_t(tp_size - r - 1) * slab_b, hipMemcpyHostToDevice, w.stream), "ag 1d h2d hi");
            }
        }
        if (profiling_enabled) {
            prof_sync_tp(tp_size, false, spine_dev0);
            prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0).count();
        }
    }

    // ---- Concurrent TP=5 Vocab-Sharded lm_head Across ALL MI50s (FP16 fdot2 lossless verify, FP8 E4M3FN draft!) ----
    void compute_lm_head_tp5(const uint16_t* d_x_on_dev1, int M, bool apply_multimax,
                             float* out_host_MxV, float* out_dev1_MxV, bool use_fp8_draft = false) {
        auto& g1 = *dev_map[spine_dev1];
        const int V = mcfg.vocab_size;
        const size_t x_bytes = size_t(M) * kHiddenSize * sizeof(uint16_t);

        hip_check(hipSetDevice(g1.hip_dev), "tp5 lm dev1");
        hip_check(hipMemcpyAsync(h_lm_x_pin, d_x_on_dev1, x_bytes, hipMemcpyDeviceToHost, g1.stream), "D2H tp5 lm_x");
        hip_check(hipStreamSynchronize(g1.stream), "sync tp5 lm_x");

        const float alpha = 1.0f, beta = 0.0f;
        for (auto& wptr : gpus) {
            auto& w = *wptr;
            const int vc = w.lm_v_count();
            if (vc <= 0) continue;
            hip_check(hipSetDevice(w.hip_dev), "tp5 lm worker");
            const uint16_t* x_ptr = (w.hip_dev == g1.hip_dev) ? d_x_on_dev1 : w.d_lm_x_in;
            if (w.hip_dev != g1.hip_dev) {
                hip_check(hipMemcpyAsync(w.d_lm_x_in, h_lm_x_pin, x_bytes, hipMemcpyHostToDevice, w.stream), "H2D tp5 lm_x");
            }
            if (M <= 8) {
                if (use_fp8_draft && w.d_lm_head_fp8 && w.d_lm_head_fp8_scale) {
                    launch_bf16x_fp8w_gemv_f32(x_ptr, w.d_lm_head_fp8, w.d_lm_head_fp8_scale,
                                               w.d_lm_logits_f32, M, vc, kHiddenSize, w.stream);
                } else {
                    launch_bf16x_f16w_gemv_f32(x_ptr, w.d_lm_head_shard, w.d_lm_logits_f32, M, vc, kHiddenSize, w.stream);
                }
            } else {
                launch_bf16_to_f16(x_ptr, w.d_lm_x_f16, M * kHiddenSize, w.stream);
                rocblas_check(rocblas_gemm_ex(
                    w.blas, rocblas_operation_transpose, rocblas_operation_none,
                    vc, M, kHiddenSize, &alpha,
                    w.d_lm_head_shard, rocblas_datatype_f16_r, kHiddenSize,
                    w.d_lm_x_f16,      rocblas_datatype_f16_r, kHiddenSize,
                    &beta,
                    w.d_lm_logits_f32, rocblas_datatype_f32_r, vc,
                    w.d_lm_logits_f32, rocblas_datatype_f32_r, vc,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "tp5 lm gemm_ex");
            }
            if (apply_multimax && has_multimax) {
                launch_multimax_inplace_f32(w.d_lm_logits_f32, w.d_mm_ranges4, w.d_mm_ts4, M, vc, w.stream);
            }
            hip_check(hipMemcpyAsync(w.h_lm_logits_pin, w.d_lm_logits_f32,
                                     size_t(M) * size_t(vc) * sizeof(float), hipMemcpyDeviceToHost, w.stream), "D2H tp5 logits");
        }

        float* dst_host = out_host_MxV ? out_host_MxV : h_lm_full_pin;
        for (auto& wptr : gpus) {
            auto& w = *wptr;
            const int vc = w.lm_v_count();
            if (vc <= 0) continue;
            hip_check(hipSetDevice(w.hip_dev), "tp5 lm sync dev");
            hip_check(hipStreamSynchronize(w.stream), "tp5 lm sync");
            for (int r = 0; r < M; ++r) {
                std::memcpy(dst_host + size_t(r) * size_t(V) + size_t(w.lm_v_begin),
                            w.h_lm_logits_pin + size_t(r) * size_t(vc),
                            size_t(vc) * sizeof(float));
            }
        }
        hip_check(hipSetDevice(g1.hip_dev), "tp5 lm restore dev1");
        if (out_dev1_MxV != nullptr) {
            hip_check(hipMemcpyAsync(out_dev1_MxV, dst_host, size_t(M) * size_t(V) * sizeof(float),
                                     hipMemcpyHostToDevice, g1.stream), "H2D tp5 full logits");
        }
    }

    // ---- Upload Helpers ----
    static std::vector<uint16_t> f32_buf_to_f16(const float* src, size_t elems) {
        std::vector<uint16_t> out(elems);
        #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (elems >= 65536)
        for (size_t i = 0; i < elems; ++i) {
            float v = src[i];
            if (v > 65504.0f) v = 65504.0f;
            else if (v < -65504.0f) v = -65504.0f;
            out[i] = f32_to_fp16_rne(v);
        }
        return out;
    }

    MhcGpuWeights upload_mhc(const SafetensorsFile& st, const std::string& pfx, Mi50ExpertWorker& g) {
        MhcGpuWeights m{};
        auto fn = load_st_as_f32(st, pfx + ".mapping_proj.weight");
        auto fn_f16 = f32_buf_to_f16(fn.data(), fn.size());
        auto sc = load_st_as_f32(st, pfx + ".hc_scale3");
        auto bs = load_st_as_f32(st, pfx + ".bias");
        m.d_fn_24x16384_f16 = gpu_upload<uint16_t>(fn_f16.data(), fn_f16.size(), g.allocs);
        m.d_scale3          = gpu_upload<float>(sc.data(), sc.size(), g.allocs);
        m.d_base24          = gpu_upload<float>(bs.data(), bs.size(), g.allocs);
        return m;
    }

    uint16_t* upload_bf16_tensor(const SafetensorsFile& st, const std::string& name, size_t elems, Mi50ExpertWorker& g) {
        const uint8_t* p = require_st_bytes(st, name, elems * 2u);
        return gpu_upload<uint16_t>(p, elems, g.allocs);
    }

    static std::vector<uint16_t> bf16_buf_to_f16(const uint16_t* src, size_t elems) {
        std::vector<uint16_t> out(elems);
        #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (elems >= 65536)
        for (size_t i = 0; i < elems; ++i) {
            float v = bf16_to_f32(src[i]);
            if (v > 65504.0f) v = 65504.0f;
            else if (v < -65504.0f) v = -65504.0f;
            out[i] = f32_to_fp16_rne(v);
        }
        return out;
    }

    uint16_t* upload_bf16_as_f16(const SafetensorsFile& st, const std::string& name, size_t elems, Mi50ExpertWorker& g) {
        const auto* raw = reinterpret_cast<const uint16_t*>(require_st_bytes(st, name, elems * 2u));
        auto f16 = bf16_buf_to_f16(raw, elems);
        return gpu_upload<uint16_t>(f16.data(), elems, g.allocs);
    }

    uint16_t* upload_bf16_rows_as_f16(const SafetensorsFile& st, const std::string& name,
                                      size_t full_rows, size_t row0, size_t nrows, size_t cols, Mi50ExpertWorker& g) {
        const auto* raw = reinterpret_cast<const uint16_t*>(require_st_bytes(st, name, full_rows * cols * 2u));
        auto f16 = bf16_buf_to_f16(raw + row0 * cols, nrows * cols);
        return gpu_upload<uint16_t>(f16.data(), nrows * cols, g.allocs);
    }

    uint16_t* upload_bf16_cols_as_f16(const SafetensorsFile& st, const std::string& name,
                                      size_t rows, size_t col0, size_t ncols, size_t full_cols, Mi50ExpertWorker& g) {
        const auto* raw = reinterpret_cast<const uint16_t*>(require_st_bytes(st, name, rows * full_cols * 2u));
        std::vector<uint16_t> f16(rows * ncols);
        #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (rows * ncols >= 65536)
        for (size_t r = 0; r < rows; ++r) {
            const uint16_t* sr = raw + r * full_cols + col0;
            uint16_t* dr = f16.data() + r * ncols;
            for (size_t c = 0; c < ncols; ++c) {
                float v = bf16_to_f32(sr[c]);
                if (v > 65504.0f) v = 65504.0f;
                else if (v < -65504.0f) v = -65504.0f;
                dr[c] = f32_to_fp16_rne(v);
            }
        }
        return gpu_upload<uint16_t>(f16.data(), rows * ncols, g.allocs);
    }

    float* upload_f32_tensor(const SafetensorsFile& st, const std::string& name, size_t elems, Mi50ExpertWorker& g) {
        auto v = load_st_as_f32(st, name);
        if (v.size() != elems) {
            throw std::runtime_error("F32 tensor size mismatch: " + name + " got " + std::to_string(v.size()));
        }
        return gpu_upload<float>(v.data(), elems, g.allocs);
    }

    void init_gpu_worker(int hip_dev, int bb_b, int bb_e, int echo_b, int echo_e, bool is_spine, bool is_dsa_owner) {
        hip_check(hipSetDevice(hip_dev), "hipSetDevice");
        auto w = std::make_unique<Mi50ExpertWorker>();
        w->hip_dev = hip_dev;
        w->bb_begin = bb_b; w->bb_end = bb_e;
        w->echo_begin = echo_b; w->echo_end = echo_e;
        hip_check(hipStreamCreate(&w->stream), "hipStreamCreate");
        rocblas_check(rocblas_create_handle(&w->blas), "rocblas_create_handle");
        rocblas_check(rocblas_set_stream(w->blas, w->stream), "rocblas_set_stream");
        rocblas_check(rocblas_set_atomics_mode(w->blas, rocblas_atomics_not_allowed), "rocblas_set_atomics_mode");
        if (is_dsa_owner) {
            hip_check(hipStreamCreate(&w->stream_sh), "hipStreamCreate sh");
            rocblas_check(rocblas_create_handle(&w->blas_sh), "rocblas_create_handle sh");
            rocblas_check(rocblas_set_stream(w->blas_sh, w->stream_sh), "rocblas_set_stream sh");
            rocblas_check(rocblas_set_atomics_mode(w->blas_sh, rocblas_atomics_not_allowed), "rocblas_set_atomics_mode sh");
            hip_check(hipEventCreateWithFlags(&w->ev_sh_start, hipEventDisableTiming), "ev_sh_start");
            hip_check(hipEventCreateWithFlags(&w->ev_sh_done,  hipEventDisableTiming), "ev_sh_done");
        }

        const auto lv32 = hlut4_levels_f32();
        uint16_t lv16[8];
        for (int i = 0; i < 8; ++i) lv16[i] = f32_to_fp16_rne(lv32[size_t(i)]);
        w->d_levels8_f16 = gpu_upload<uint16_t>(lv16, 8, w->allocs);

        w->d_x_rot_f16     = gpu_alloc<uint16_t>(size_t(max_chunk) * kLatentDim, w->allocs, true);
        w->d_groups        = gpu_alloc<ExpertDispatchGroup>(size_t(max_groups), w->allocs, true);
        w->h_groups_pin    = host_pin_alloc<ExpertDispatchGroup>(size_t(max_groups), pin_allocs, true);
        w->d_z_f32         = gpu_alloc<float>(size_t(max_groups) * kMaxGroupTokens * kMoeInterSize, w->allocs, true);
        w->d_z_rot_f16     = gpu_alloc<uint16_t>(size_t(max_groups) * kMaxGroupTokens * kMoeInterSize, w->allocs, true);
        w->d_y_slots_f32   = gpu_alloc<float>(size_t(max_chunk) * kTopKExperts * kLatentDim, w->allocs, true);
        w->d_y_acc_f32     = gpu_alloc<float>(size_t(max_chunk) * kLatentDim, w->allocs, true);
        w->h_y_partial_pin = host_pin_alloc<float>(size_t(max_chunk) * kLatentDim, pin_allocs, true);

        w->b_head.fill(-1);
        w->b_cnt.fill(0);
        w->b_next.resize(size_t(max_chunk) * kTopKExperts);
        w->b_rows.resize(size_t(max_chunk) * kTopKExperts);
        w->b_slots.resize(size_t(max_chunk) * kTopKExperts);
        w->b_wts.resize(size_t(max_chunk) * kTopKExperts);
        w->active_exps.reserve(kNumRoutedExperts);
        w->h_groups.reserve(size_t(max_groups) * 4u);

        // Wave-8 M_e >= 16 rocblas_gemm_ex MoE Prefill Crossover Scratch (~26 MiB/GPU!)
        w->d_w_unpack_f16  = gpu_alloc<uint16_t>(size_t(kMoeInterSize) * kLatentDim, w->allocs, false);
        w->d_xg_f16        = gpu_alloc<uint16_t>(size_t(max_chunk) * kLatentDim, w->allocs, false);
        w->d_g_f32         = gpu_alloc<float>(size_t(max_chunk) * kMoeInterSize, w->allocs, false);
        w->d_u_f32         = gpu_alloc<float>(size_t(max_chunk) * kMoeInterSize, w->allocs, false);
        w->d_yh_f32        = gpu_alloc<float>(size_t(max_chunk) * kLatentDim, w->allocs, false);
        w->d_row_ids       = gpu_alloc<int32_t>(size_t(max_chunk) * kTopKExperts, w->allocs, false);
        w->d_row_slots     = gpu_alloc<int32_t>(size_t(max_chunk) * kTopKExperts, w->allocs, false);
        w->d_row_wts       = gpu_alloc<float>(size_t(max_chunk) * kTopKExperts, w->allocs, false);
        w->h_row_ids_pin   = host_pin_alloc<int32_t>(size_t(max_chunk) * kTopKExperts, pin_allocs, false);
        w->h_row_slots_pin = host_pin_alloc<int32_t>(size_t(max_chunk) * kTopKExperts, pin_allocs, false);
        w->h_row_wts_pin   = host_pin_alloc<float>(size_t(max_chunk) * kTopKExperts, pin_allocs, false);
        w->hgemm_plans.reserve(kNumRoutedExperts);

        // Compact 512 KiB FP16 input buffer for M>8 rocblas_gemm_ex lm_head (0 MiB d_lm_w_f32!)
        w->d_lm_x_f16 = gpu_alloc<uint16_t>(64u * kHiddenSize, w->allocs, false);

        if (is_spine && h_tp_mapped_slabs) {
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_tp_mapped_slabs), h_tp_mapped_slabs, 0), "mapped slabs ptr");
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_tp_mapped_ready), h_tp_mapped_ready, 0), "mapped ready ptr");
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_tp_mapped_done),  h_tp_mapped_done,  0), "mapped done ptr");
        }

        if (is_spine) {
            const auto yarn_inv = compute_hca_yarn_inv_freq(
                kHcaRopeHeadDim, mcfg.compress_rope_theta, mcfg.yarn_factor,
                mcfg.yarn_orig_max_pos, mcfg.yarn_beta_fast, mcfg.yarn_beta_slow);
            const auto dsa_inv = compute_dsa_inv_freq(kDsaQkRopeHeadDim, mcfg.rope_theta);
            w->yarn_rope = upload_rope2d(yarn_inv, w->allocs);
            w->dsa_rope  = upload_rope2d(dsa_inv,  w->allocs);

            // GEMM strip buffers stay compact at 512-row strip size even when max_chunk == 1024!
            w->d_x_f16         = gpu_alloc<uint16_t>(512u * 16384u, w->allocs, false);
            w->d_y_f32         = gpu_alloc<float>(512u * (is_dsa_owner ? 36864u : 16384u), w->allocs, false);
            w->d_hca_kcomp_f16 = gpu_alloc<uint16_t>(size_t((opt.max_kv + 127) / 128 + 1024) * 512u, w->allocs, true);
            const int h_loc    = kHcaNumHeads / std::max(1, tp_size);
            w->d_hca_o_tot_f32 = gpu_alloc<float>(size_t(max_chunk) * size_t(h_loc) * 512u, w->allocs, false);
            w->d_hca_ml_f32    = gpu_alloc<float>(size_t(max_chunk) * size_t(h_loc) * 4u, w->allocs, false);
            if (is_dsa_owner) {
                w->d_sh_x_f16  = gpu_alloc<uint16_t>(512u * kHiddenSize, w->allocs, false);
                w->d_sh_y_f32  = gpu_alloc<float>(512u * kHiddenSize, w->allocs, false);
                w->d_sh_tmp_a  = gpu_alloc<uint16_t>(size_t(max_chunk) * kMoeInterSize, w->allocs, false);
                w->d_sh_tmp_b  = gpu_alloc<uint16_t>(size_t(max_chunk) * kMoeInterSize, w->allocs, false);
                w->d_sh_out    = gpu_alloc<uint16_t>(size_t(max_chunk) * kHiddenSize, w->allocs, false);
            }
            w->d_mhc_ws        = gpu_alloc<float>(size_t(max_chunk) * 100u, w->allocs, true);
            w->d_argmax_v64    = gpu_alloc<float>(64u, w->allocs, false);
            w->d_argmax_i64    = gpu_alloc<int32_t>(64u, w->allocs, false);

            w->d_res_a         = gpu_alloc<uint16_t>(size_t(max_chunk) * kHcDim, w->allocs, true);
            w->d_res_b         = gpu_alloc<uint16_t>(size_t(max_chunk) * kHcDim, w->allocs, true);
            w->d_post_mix      = gpu_alloc<float>(size_t(max_chunk) * kHcMult, w->allocs, true);
            w->d_comb_mix      = gpu_alloc<float>(size_t(max_chunk) * kHcMult2, w->allocs, true);
            w->d_x_in          = gpu_alloc<uint16_t>(size_t(max_chunk) * kHiddenSize, w->allocs, true);
            w->d_sub_out       = gpu_alloc<uint16_t>(size_t(max_chunk) * kHiddenSize, w->allocs, true);
            w->d_tmp_a         = gpu_alloc<uint16_t>(size_t(max_chunk) * 32768u, w->allocs, true);
            w->d_tmp_b         = gpu_alloc<uint16_t>(std::max<size_t>(size_t(max_chunk) * 16384u, hip_dev == spine_dev1 ? 64u * 32768u : 0u), w->allocs, true);
            w->d_tmp_c         = gpu_alloc<uint16_t>(size_t(max_chunk) * (is_dsa_owner ? 36864u : 9216u), w->allocs, true);
            w->d_tmp_d         = gpu_alloc<uint16_t>(size_t(max_chunk) * 16384u, w->allocs, true);
            w->d_qr            = gpu_alloc<uint16_t>(size_t(max_chunk) * 1024u, w->allocs, true);
            w->d_kv_tmp        = gpu_alloc<uint16_t>(size_t(max_chunk) * 576u, w->allocs, true);
            w->d_h_lat         = gpu_alloc<uint16_t>(size_t(max_chunk) * kLatentDim, w->allocs, true);
            w->d_f32_tmp0      = gpu_alloc<float>(size_t(max_chunk) * kLatentDim, w->allocs, true);
            w->d_f32_tmp1      = gpu_alloc<float>(size_t(max_chunk) * 1024u, w->allocs, true);
            w->d_pos_ids       = gpu_alloc<int32_t>(size_t(max_chunk), w->allocs, true);
            w->d_tok_ids       = gpu_alloc<int32_t>(size_t(max_chunk), w->allocs, true);
            w->d_topk_ids      = gpu_alloc<int32_t>(size_t(max_chunk) * kTopKExperts, w->allocs, true);
            w->d_topk_wts      = gpu_alloc<float>(size_t(max_chunk) * kTopKExperts, w->allocs, true);
            if (is_dsa_owner) {
                w->d_dsa_ksel_f16 = gpu_alloc<uint16_t>(16u * size_t(kDsaIndexTopK + kSlidingWindow) * 576u, w->allocs, false);
                w->idx_scores_cap = std::max<size_t>(size_t(opt.max_kv), 16ULL * 1024ULL * 1024ULL);
                w->d_idx_scores   = gpu_alloc<float>(w->idx_scores_cap, w->allocs, true);
                w->d_fi_gates     = gpu_alloc<float>(size_t(max_chunk) * 64u, w->allocs, true);
                w->d_fi_q8        = gpu_alloc<uint16_t>(size_t(max_chunk) * size_t(kFlashIndexerTopKHeads) * 128u, w->allocs, true);
                w->d_fi_w8        = gpu_alloc<float>(size_t(max_chunk) * size_t(kFlashIndexerTopKHeads), w->allocs, true);
                w->d_sel_idx      = gpu_alloc<int32_t>(size_t(max_chunk) * size_t(kDsaIndexTopK + kSlidingWindow), w->allocs, true);
                w->d_sel_cnts     = gpu_alloc<int32_t>(size_t(max_chunk), w->allocs, true);
            }
        }

        dev_map[hip_dev] = w.get();
        gpus.push_back(std::move(w));
    }

    void load_spine_shared() {
        const std::string p = opt.weight_root + "/packed_spine/spine_shared.safetensors";
        SafetensorsFile st;
        st.open(p);

        auto& g0 = *dev_map[spine_dev0];
        hip_check(hipSetDevice(g0.hip_dev), "set dev0 shared");
        d_embed_bf16 = upload_bf16_tensor(st, "model.embed_tokens.weight", size_t(mcfg.vocab_size) * kHiddenSize, g0);

        auto& g1 = *dev_map[spine_dev1];
        hip_check(hipSetDevice(g1.hip_dev), "set dev1 shared");
        d_lm_head_bf16  = nullptr; // TP=5 vocab-sharded across ALL MI50s (~314 MiB/GPU)!
        d_final_norm    = upload_bf16_tensor(st, "model.norm.weight", kHiddenSize, g1);
        d_hc_head_fn    = upload_f32_tensor(st, "model.mhc_contract.hc_head_fn", size_t(kHcMult) * kHcDim, g1);
        d_hc_head_scale = upload_f32_tensor(st, "model.mhc_contract.hc_head_scale", 1, g1);
        d_hc_head_base  = upload_f32_tensor(st, "model.mhc_contract.hc_head_base", kHcMult, g1);
        static constexpr int kMaxLogitsBatch = 64;
        g1.d_logits_f32 = gpu_alloc<float>(size_t(std::max(8, opt.echo_draft_steps + 2)) * size_t(mcfg.vocab_size), g1.allocs, false);
        h_lm_x_pin      = host_pin_alloc<uint16_t>(size_t(kMaxLogitsBatch) * kHiddenSize, pin_allocs, true);
        h_lm_full_pin   = host_pin_alloc<float>(size_t(std::max(8, opt.echo_draft_steps + 2)) * size_t(mcfg.vocab_size), pin_allocs, false);

        std::vector<float> mm_r, mm_t;
        if (st.contains("model.lm_head.multimax_ranges") && st.contains("model.lm_head.multimax_ts")) {
            mm_r = load_st_as_f32(st, "model.lm_head.multimax_ranges");
            mm_t = load_st_as_f32(st, "model.lm_head.multimax_ts");
            std::memcpy(h_mm_ranges4, mm_r.data(), 16);
            std::memcpy(h_mm_ts4, mm_t.data(), 16);
            d_mm_ranges4 = gpu_upload<float>(mm_r.data(), 4, g1.allocs);
            d_mm_ts4     = gpu_upload<float>(mm_t.data(), 4, g1.allocs);
            has_multimax = true;
        }

        // Shard lm_head.weight [V, 4096] evenly across all MI50 GPUs:
        //   - FP16 (314.4 MiB/GPU) for lossless fdot2 backbone verify (10b mantissa > BF16 7b!)
        //   - FP8 E4M3FN (157.2 MiB/GPU) for 2x-bandwidth DSpark draft-only logits!
        const auto* lm_raw = reinterpret_cast<const uint16_t*>(
            require_st_bytes(st, "lm_head.weight", size_t(mcfg.vocab_size) * kHiddenSize * 2u));
        const int n_tp = int(gpus.size());
        for (int gi = 0; gi < n_tp; ++gi) {
            auto& w = *gpus[size_t(gi)];
            w.lm_v_begin = (mcfg.vocab_size * gi) / n_tp;
            w.lm_v_end   = (mcfg.vocab_size * (gi + 1)) / n_tp;
            const int vc = w.lm_v_count();
            const uint16_t* src_shard = lm_raw + size_t(w.lm_v_begin) * kHiddenSize;
            auto f16_shard = bf16_buf_to_f16(src_shard, size_t(vc) * kHiddenSize);

            std::vector<uint8_t> fp8_shard(size_t(vc) * kHiddenSize);
            std::vector<float>   fp8_scale(size_t(vc), 1.0f);
            #pragma omp parallel for num_threads(kOmpThreads) schedule(static)
            for (int r = 0; r < vc; ++r) {
                const uint16_t* row = src_shard + size_t(r) * kHiddenSize;
                float amax = 0.0f;
                for (int c = 0; c < kHiddenSize; ++c) {
                    const float a = std::fabs(bf16_to_f32(row[c]));
                    if (a > amax) amax = a;
                }
                const float sc = (amax > 1e-12f) ? (amax / 448.0f) : 1.0f;
                const float inv_sc = 1.0f / sc;
                fp8_scale[size_t(r)] = sc;
                uint8_t* dst_r = fp8_shard.data() + size_t(r) * kHiddenSize;
                for (int c = 0; c < kHiddenSize; ++c) {
                    dst_r[c] = f32_to_fp8_e4m3fn(bf16_to_f32(row[c]) * inv_sc);
                }
            }

            hip_check(hipSetDevice(w.hip_dev), "set tp5 lm dev");
            w.d_lm_head_shard     = gpu_upload<uint16_t>(f16_shard.data(), f16_shard.size(), w.allocs);
            w.d_lm_head_fp8       = gpu_upload<uint8_t>(fp8_shard.data(), fp8_shard.size(), w.allocs);
            w.d_lm_head_fp8_scale = gpu_upload<float>(fp8_scale.data(), fp8_scale.size(), w.allocs);
            w.d_lm_x_in           = gpu_alloc<uint16_t>(size_t(kMaxLogitsBatch) * kHiddenSize, w.allocs, true);
            w.d_lm_logits_f32     = gpu_alloc<float>(size_t(kMaxLogitsBatch) * size_t(vc), w.allocs, false);
            w.h_lm_logits_pin     = host_pin_alloc<float>(size_t(kMaxLogitsBatch) * size_t(vc), pin_allocs, false);
            if (has_multimax) {
                w.d_mm_ranges4 = gpu_upload<float>(mm_r.data(), 4, w.allocs);
                w.d_mm_ts4     = gpu_upload<float>(mm_t.data(), 4, w.allocs);
            }
        }
        std::printf("[ERNIE55-RT] TP=%d vocab-sharded lm_head loaded across %d MI50s (FP16 %.1f MiB/GPU + FP8 draft %.1f MiB/GPU)\n",
                    n_tp, n_tp,
                    double(size_t(mcfg.vocab_size) * kHiddenSize * 2u) / double(n_tp) / 1048576.0,
                    double(size_t(mcfg.vocab_size) * kHiddenSize * 1u) / double(n_tp) / 1048576.0);
    }

    void load_spine_layer(int l) {
        char fname[512];
        if (l == mcfg.num_hidden_layers) {
            std::snprintf(fname, sizeof(fname), "%s/packed_spine/spine_echo_%02d.safetensors", opt.weight_root.c_str(), l);
        } else {
            std::snprintf(fname, sizeof(fname), "%s/packed_spine/spine_layer_%02d.safetensors", opt.weight_root.c_str(), l);
        }
        SafetensorsFile st;
        st.open(fname);

        const int owner = (l < split_layer) ? spine_dev0 : spine_dev1;
        auto& L = layers[size_t(l)];
        L.layer_idx = l;
        L.owner_dev = owner;
        L.is_echo   = (l == mcfg.num_hidden_layers);
        L.is_dsa    = mcfg.is_dsa_layer(l);
        L.is_dense  = (l < mcfg.first_k_dense_replace);
        L.tp_size   = L.is_echo ? 1 : tp_size;
        L.owner_tp_rank = 0;
        if (!L.is_echo) {
            for (int r = 0; r < L.tp_size; ++r) {
                if (tp_devs[r] == owner) { L.owner_tp_rank = r; break; }
            }
        }

        // Pre-permute FP32 router_w01 [1024, 4096], corr_bias [512], scale_param [512] once on host if MoE
        std::vector<float> rw_full, cb_full, sp_full;
        if (!L.is_dense) {
            auto rw = load_st_as_f32(st, "moe.router_w01");
            auto cb = load_st_as_f32(st, "moe.corr_bias");
            auto sp = load_st_as_f32(st, "moe.scale_param");
            if (has_router_perm) {
                rw_full.resize(rw.size());
                cb_full.resize(cb.size());
                sp_full.resize(sp.size());
                const auto& perm = slot_to_raw[size_t(l)];
                for (int slot = 0; slot < kNumRoutedExperts; ++slot) {
                    const int raw = int(perm[size_t(slot)]);
                    std::memcpy(rw_full.data() + size_t(slot) * kHiddenSize,
                                rw.data()      + size_t(raw)  * kHiddenSize, size_t(kHiddenSize) * sizeof(float));
                    std::memcpy(rw_full.data() + size_t(kNumRoutedExperts + slot) * kHiddenSize,
                                rw.data()      + size_t(kNumRoutedExperts + raw)  * kHiddenSize, size_t(kHiddenSize) * sizeof(float));
                    cb_full[size_t(slot)] = cb[size_t(raw)];
                    sp_full[size_t(slot)] = sp[size_t(raw)];
                }
            } else {
                rw_full = std::move(rw);
                cb_full = std::move(cb);
                sp_full = std::move(sp);
            }
            std::memcpy(L.h_corr_bias.data(), cb_full.data(), size_t(kNumRoutedExperts) * sizeof(float));
            std::memcpy(L.h_scale_param.data(), sp_full.data(), size_t(kNumRoutedExperts) * sizeof(float));
            auto ln = load_st_as_f32(st, "moe.lat_norm");
            std::memcpy(L.h_lat_norm.data(), ln.data(), size_t(kLatentDim) * sizeof(float));
        }

        std::vector<float> hca_sink_full;
        std::vector<uint16_t> hca_qkv_comp_f16;
        if (!L.is_dsa) {
            hca_sink_full = load_st_as_f32(st, "attn.attn_sink");
            const auto* qd = reinterpret_cast<const uint16_t*>(require_st_bytes(st, "attn.q_down",     1024u * 4096u * 2u));
            const auto* kv = reinterpret_cast<const uint16_t*>(require_st_bytes(st, "attn.kv_proj",     512u * 4096u * 2u));
            const auto* wk = reinterpret_cast<const uint16_t*>(require_st_bytes(st, "attn.comp_wkv",    512u * 4096u * 2u));
            const auto* wg = reinterpret_cast<const uint16_t*>(require_st_bytes(st, "attn.comp_wgate",  512u * 4096u * 2u));
            auto f_qd = bf16_buf_to_f16(qd, 1024u * 4096u);
            auto f_kv = bf16_buf_to_f16(kv,  512u * 4096u);
            auto f_wk = bf16_buf_to_f16(wk,  512u * 4096u);
            auto f_wg = bf16_buf_to_f16(wg,  512u * 4096u);
            hca_qkv_comp_f16.resize(2560u * 4096u);
            std::memcpy(hca_qkv_comp_f16.data() +    0u * 4096u, f_qd.data(), f_qd.size() * 2u);
            std::memcpy(hca_qkv_comp_f16.data() + 1024u * 4096u, f_kv.data(), f_kv.size() * 2u);
            std::memcpy(hca_qkv_comp_f16.data() + 1536u * 4096u, f_wk.data(), f_wk.size() * 2u);
            std::memcpy(hca_qkv_comp_f16.data() + 2048u * 4096u, f_wg.data(), f_wg.size() * 2u);
        }

        const int tp = L.tp_size;
        const size_t max_blocks = size_t((opt.max_kv + 127) / 128);
        for (int r = 0; r < tp; ++r) {
            const int dev_r = L.is_echo ? owner : tp_devs[r];
            auto& g_r = *dev_map[dev_r];
            hip_check(hipSetDevice(g_r.hip_dev), "set spine layer tp dev");

            L.attn_mhc_tp[r]  = upload_mhc(st, "attn_mhc", g_r);
            L.mlp_mhc_tp[r]   = upload_mhc(st, "mlp_mhc",  g_r);
            L.d_in_norm_tp[r] = upload_bf16_tensor(st, "input_layernorm.weight", kHiddenSize, g_r);
            L.d_post_norm_tp[r] = upload_bf16_tensor(st, "post_attention_layernorm.weight", kHiddenSize, g_r);

            if (!L.is_dsa) {
                const int h_loc = kHcaNumHeads / tp;
                const int g_loc = kHcaOGroups / tp;
                auto& H = L.hca_tp[r];
                H.d_qkv_comp   = gpu_upload<uint16_t>(hca_qkv_comp_f16.data(), hca_qkv_comp_f16.size(), g_r.allocs);
                H.d_q_norm     = upload_bf16_tensor(st, "attn.q_norm",     1024u, g_r);
                H.d_q_up       = upload_bf16_rows_as_f16(st, "attn.q_up",  32768u, size_t(r) * size_t(h_loc) * 512u, size_t(h_loc) * 512u, 1024u, g_r);
                H.d_kv_norm    = upload_bf16_tensor(st, "attn.kv_norm",    512u, g_r);
                H.d_comp_norm  = upload_bf16_tensor(st, "attn.comp_norm",  512u, g_r);
                H.d_comp_ape   = upload_f32_tensor(st,  "attn.comp_ape",   128u * 512u, g_r);
                H.d_attn_sink  = gpu_upload<float>(hca_sink_full.data() + size_t(r) * size_t(h_loc), size_t(h_loc), g_r.allocs);
                H.d_vha_m      = upload_f32_tensor(st,  "attn.vha_m",      64u * 64u, g_r);
                H.d_wo_a       = upload_bf16_rows_as_f16(st, "attn.wo_a",  8192u, size_t(r) * size_t(g_loc) * 1024u, size_t(g_loc) * 1024u, 4096u, g_r);
                H.d_gate_proj  = upload_bf16_rows_as_f16(st, "attn.gate_proj", 8192u, size_t(r) * size_t(g_loc) * 1024u, size_t(g_loc) * 1024u, 1024u, g_r);
                H.d_o_proj     = upload_bf16_cols_as_f16(st, "attn.o_proj", 4096u, size_t(r) * size_t(g_loc) * 1024u, size_t(g_loc) * 1024u, 8192u, g_r);

                L.d_hca_comp_kv_tp[r]  = gpu_alloc<uint16_t>(max_blocks * 512u, g_r.allocs, true);
                L.d_hca_swa_ring_tp[r] = gpu_alloc<uint16_t>(256u * 512u, g_r.allocs, true);
                L.d_hca_state_kv_tp[r] = gpu_alloc<float>(128u * 512u, g_r.allocs, true);
                L.d_hca_state_sc_tp[r] = gpu_alloc<float>(128u * 512u, g_r.allocs, true);
                L.d_hca_snap_swa_tp[r] = gpu_alloc<uint16_t>(256u * 512u, g_r.allocs, true);
                L.d_hca_snap_kv_tp[r]  = gpu_alloc<float>(128u * 512u, g_r.allocs, true);
                L.d_hca_snap_sc_tp[r]  = gpu_alloc<float>(128u * 512u, g_r.allocs, true);
            } else {
                const int h_loc = kDsaNumHeads / tp;
                auto& D = L.dsa_tp[r];
                D.d_q_a       = upload_bf16_as_f16(st, "attn.q_a",        1024u * 4096u, g_r);
                D.d_q_a_norm  = upload_bf16_tensor(st, "attn.q_a_norm",   1024u, g_r);
                D.d_q_b       = upload_bf16_rows_as_f16(st, "attn.q_b",   16384u, size_t(r) * size_t(h_loc) * 256u, size_t(h_loc) * 256u, 1024u, g_r);
                D.d_kv_b      = upload_bf16_rows_as_f16(st, "attn.kv_b",  28672u, size_t(r) * size_t(h_loc) * 448u, size_t(h_loc) * 448u, 512u, g_r);
                D.d_vha_m     = upload_f32_tensor(st,  "attn.vha_m",      64u * 64u, g_r);
                D.d_gate_proj = upload_bf16_rows_as_f16(st, "attn.gate_proj", 16384u, size_t(r) * size_t(h_loc) * 256u, size_t(h_loc) * 256u, 1024u, g_r);
                D.d_o_proj    = upload_bf16_cols_as_f16(st, "attn.o_proj", 4096u, size_t(r) * size_t(h_loc) * 256u, size_t(h_loc) * 256u, 16384u, g_r);
                if (r == L.owner_tp_rank) {
                    D.d_kv_a       = upload_bf16_as_f16(st, "attn.kv_a",       576u * 4096u, g_r);
                    D.d_kv_a_norm  = upload_bf16_tensor(st, "attn.kv_a_norm",  512u, g_r);
                    D.d_idx_wq_b   = upload_bf16_as_f16(st, "attn.idx_wq_b",   8192u * 1024u, g_r);
                    D.d_idx_wk     = upload_bf16_as_f16(st, "attn.idx_wk",     128u * 4096u, g_r);
                    D.d_idx_kn_w   = upload_bf16_tensor(st, "attn.idx_kn_w",   128u, g_r);
                    D.d_idx_kn_b   = upload_bf16_tensor(st, "attn.idx_kn_b",   128u, g_r);
                    D.d_idx_w_proj = upload_bf16_as_f16(st, "attn.idx_w_proj", 64u * 4096u, g_r);
                    if (st.contains("attn.flashindexer_gate") && st.contains("attn.flashindexer_bias")) {
                        D.d_fi_gate = upload_bf16_as_f16(st, "attn.flashindexer_gate", 64u * 4096u, g_r);
                        D.d_fi_bias = upload_f32_tensor(st,  "attn.flashindexer_bias", 64u, g_r);
                    }
                    D.d_attn_sink = upload_f32_tensor(st, "attn.attn_sink", 64u, g_r);
                    L.d_dsa_mla_kv   = gpu_alloc<uint16_t>(size_t(opt.max_kv) * 576u, g_r.allocs, true);
                    L.d_dsa_ik_cache = gpu_alloc<uint16_t>(size_t(opt.max_kv) * 128u, g_r.allocs, true);
                }
            }

            if (L.is_dense) {
                const size_t inter_loc = size_t(mcfg.intermediate_size) / size_t(tp);
                L.dense_tp[r].d_gate = upload_bf16_rows_as_f16(st, "mlp.gate", size_t(mcfg.intermediate_size), size_t(r) * inter_loc, inter_loc, kHiddenSize, g_r);
                L.dense_tp[r].d_up   = upload_bf16_rows_as_f16(st, "mlp.up",   size_t(mcfg.intermediate_size), size_t(r) * inter_loc, inter_loc, kHiddenSize, g_r);
                L.dense_tp[r].d_down = upload_bf16_cols_as_f16(st, "mlp.down", kHiddenSize, size_t(r) * inter_loc, inter_loc, size_t(mcfg.intermediate_size), g_r);
            } else if (r == L.owner_tp_rank) {
                // Hybrid Owner-GPU MoE Spine: full unsharded router + fc1/fc2 + shared_experts on owner_dev ONLY
                // (ZERO 4-GPU Router D2H/sync, ZERO EPYC WHT128, ZERO MoE AllReduce, ZERO dev2/dev3 stream_sh contention!)
                auto& M = L.moe_tp[r];
                M.d_fc1_lat     = upload_bf16_as_f16(st, "moe.fc1_lat", size_t(kLatentDim) * kHiddenSize, g_r);
                M.d_fc2_lat     = upload_bf16_as_f16(st, "moe.fc2_lat", size_t(kHiddenSize) * kLatentDim, g_r);
                M.d_sh_w1       = upload_bf16_as_f16(st, "moe.sh_w1",   size_t(kMoeInterSize) * kHiddenSize, g_r);
                M.d_sh_w3       = upload_bf16_as_f16(st, "moe.sh_w3",   size_t(kMoeInterSize) * kHiddenSize, g_r);
                M.d_sh_w2          = upload_bf16_as_f16(st, "moe.sh_w2",   size_t(kHiddenSize) * kMoeInterSize, g_r);
                auto rw_f16        = f32_buf_to_f16(rw_full.data(), rw_full.size());
                M.d_router_w01_f16 = gpu_upload<uint16_t>(rw_f16.data(), rw_f16.size(), g_r.allocs);
                M.d_lat_norm       = upload_bf16_tensor(st, "moe.lat_norm", kLatentDim, g_r);
                M.d_corr_bias   = gpu_upload<float>(cb_full.data(), cb_full.size(), g_r.allocs);
                M.d_scale_param = gpu_upload<float>(sp_full.data(), sp_full.size(), g_r.allocs);
            }
        }

        // Alias primary owner rank pointers (used by save_kvblk_ex / load_kvblk_ex / dump_kv_raw_bytes / ECHO!)
        const int ro = L.owner_tp_rank;
        L.attn_mhc    = L.attn_mhc_tp[ro];
        L.mlp_mhc     = L.mlp_mhc_tp[ro];
        L.d_in_norm   = L.d_in_norm_tp[ro];
        L.d_post_norm = L.d_post_norm_tp[ro];
        L.hca         = L.hca_tp[ro];
        L.dsa         = L.dsa_tp[ro];
        L.dense       = L.dense_tp[ro];
        L.moe         = L.moe_tp[ro];
        if (!L.is_dsa) {
            L.d_hca_comp_kv  = L.d_hca_comp_kv_tp[ro];
            L.d_hca_swa_ring = L.d_hca_swa_ring_tp[ro];
            L.d_hca_state_kv = L.d_hca_state_kv_tp[ro];
            L.d_hca_state_sc = L.d_hca_state_sc_tp[ro];
            L.d_hca_snap_swa = L.d_hca_snap_swa_tp[ro];
            L.d_hca_snap_kv  = L.d_hca_snap_kv_tp[ro];
            L.d_hca_snap_sc  = L.d_hca_snap_sc_tp[ro];
        }

        if (L.is_echo) {
            auto& g = *dev_map[owner];
            hip_check(hipSetDevice(g.hip_dev), "set echo dev");
            L.d_echo_enorm     = upload_bf16_tensor(st, "echo.enorm", kHiddenSize, g);
            L.d_echo_hnorm     = upload_bf16_tensor(st, "echo.hnorm", kHiddenSize, g);
            L.d_echo_eproj     = upload_bf16_as_f16(st, "echo.eproj", size_t(kHiddenSize) * kHiddenSize, g);
            L.d_echo_hproj     = upload_bf16_as_f16(st, "echo.hproj", size_t(kHiddenSize) * kHiddenSize, g);
            L.d_echo_hc_fn     = upload_f32_tensor(st,  "echo.hc_head_fn", size_t(kHcMult) * kHcDim, g);
            L.d_echo_hc_scale  = upload_f32_tensor(st,  "echo.hc_head_scale", 1, g);
            L.d_echo_hc_base   = upload_f32_tensor(st,  "echo.hc_head_base", kHcMult, g);
            L.d_echo_head_norm = upload_bf16_tensor(st, "echo.shared_head_norm", kHiddenSize, g);
            has_echo_l43 = true;
        }
    }

    void load_hlut4_layer_experts(int l) {
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/experts_hlut4/hlut4_layer_%02d.bin", opt.weight_root.c_str(), l);
        const int fd = ::open(fname, O_RDONLY);
        if (fd < 0) throw std::runtime_error(std::string("Missing WHT128-HLUT4 expert shard: ") + fname);

        Ernie55Hlut4LayerHeader hdr{};
        if (::pread(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr)) ||
            hdr.magic != kErnie55Hlut4ShardMagic || hdr.version != kErnie55Hlut4ShardVersion) {
            ::close(fd);
            throw std::runtime_error(std::string("Corrupt HLUT4 header in ") + fname);
        }

        if (l >= 0 && l < kTotalLayersWithEcho) {
            std::memcpy(layers[size_t(l)].h_signs_in.data(), hdr.signs_in, sizeof(hdr.signs_in));
        }

        const bool is_echo_only = (!mcfg.has_bundled_dspark()) && (l == mcfg.num_hidden_layers);
        const bool is_dspark_l  = (l >= mcfg.num_hidden_layers);
        const size_t n_gpus = gpus.size();
        std::vector<std::thread> lthreads;
        std::vector<std::exception_ptr> lerrs(n_gpus);
        lthreads.reserve(n_gpus);
        for (size_t gi = 0; gi < n_gpus; ++gi) {
            lthreads.emplace_back([&, gi]() {
                try {
                    auto& g = *gpus[gi];
                    const int cnt = is_echo_only ? (g.echo_end - g.echo_begin)
                                  : ((is_dspark_l && g.bb_base_count > 0) ? g.bb_base_count : g.bb_count());
                    hip_check(hipSetDevice(g.hip_dev), "set dev hlut4");
                    g.d_signs_in[size_t(l)]    = gpu_upload<uint64_t>(hdr.signs_in,    kLatentDim / 64,    g.allocs);
                    g.d_signs_inter[size_t(l)] = gpu_upload<uint64_t>(hdr.signs_inter, kMoeInterSize / 64, g.allocs);
                    if (cnt <= 0) return;

                    const size_t total_tiles_proj = size_t(cnt) * kExpProjTiles;
                    std::vector<Hlut4Tile32x32> w1(total_tiles_proj), w3(total_tiles_proj), w2(total_tiles_proj);
                    std::vector<Hlut4Tile32x32> raw_one(3u * kExpProjTiles);
                    for (int i = 0; i < cnt; ++i) {
                        const int ge = is_echo_only ? (g.echo_begin + i)
                                     : (!g.bb_global_ids.empty() ? g.bb_global_ids[size_t(i)] : (g.bb_begin + i));
                        const off_t eoff = off_t(sizeof(hdr)) + off_t(ge) * off_t(kHlut4ExpertBytes);
                        size_t got = 0;
                        while (got < kHlut4ExpertBytes) {
                            const ssize_t r = ::pread(fd, reinterpret_cast<uint8_t*>(raw_one.data()) + got,
                                                      kHlut4ExpertBytes - got, eoff + off_t(got));
                            if (r <= 0) throw std::runtime_error(std::string("Short read ") + fname);
                            got += size_t(r);
                        }
                        std::memcpy(w1.data() + size_t(i) * kExpProjTiles, raw_one.data() + 0u * kExpProjTiles, kHlut4ProjBytes);
                        std::memcpy(w3.data() + size_t(i) * kExpProjTiles, raw_one.data() + 1u * kExpProjTiles, kHlut4ProjBytes);
                        std::memcpy(w2.data() + size_t(i) * kExpProjTiles, raw_one.data() + 2u * kExpProjTiles, kHlut4ProjBytes);
                    }
                    g.d_w1[size_t(l)] = gpu_upload<Hlut4Tile32x32>(w1.data(), w1.size(), g.allocs);
                    g.d_w3[size_t(l)] = gpu_upload<Hlut4Tile32x32>(w3.data(), w3.size(), g.allocs);
                    g.d_w2[size_t(l)] = gpu_upload<Hlut4Tile32x32>(w2.data(), w2.size(), g.allocs);
                } catch (...) {
                    lerrs[gi] = std::current_exception();
                }
            });
        }
        for (auto& th : lthreads) th.join();
        for (auto& e : lerrs) {
            if (e) { ::close(fd); std::rethrow_exception(e); }
        }
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }

    void maybe_load_bundled_dspark() {
        const std::string p = opt.weight_root + "/packed_spine/spine_dspark.safetensors";
        if (!mcfg.has_bundled_dspark() || !std::filesystem::is_regular_file(p)) return;
        SafetensorsFile st;
        st.open(p);
        auto& g0 = *dev_map[spine_dev0];
        auto& g1 = *dev_map[spine_dev1];
        hip_check(hipSetDevice(g1.hip_dev), "set dev1 dspark");

        const int n_ds = int(mcfg.dspark_target_layer_ids.size());
        d_dspark_main_proj = upload_bf16_as_f16(st, "dspark.main_proj", size_t(kHiddenSize) * size_t(kHiddenSize * n_ds), g1);
        d_dspark_main_norm = upload_bf16_tensor(st, "dspark.main_norm", kHiddenSize, g1);
        d_dspark_norm      = upload_bf16_tensor(st, "dspark.norm", kHiddenSize, g1);
        d_dspark_hc_fn     = upload_f32_tensor(st,  "dspark.hc_head_fn", size_t(kHcMult) * kHcDim, g1);
        d_dspark_hc_scale  = upload_f32_tensor(st,  "dspark.hc_head_scale", 1, g1);
        d_dspark_hc_base   = upload_f32_tensor(st,  "dspark.hc_head_base", kHcMult, g1);

        d_dspark_aux_cat     = gpu_alloc<uint16_t>(size_t(max_chunk) * size_t(n_ds) * kHiddenSize, g1.allocs, true);
        d_dspark_me_bf16     = gpu_alloc<uint16_t>(size_t(std::max(1, mcfg.dspark_markov_rank)), g1.allocs, true);
        d_dspark_bias_f32    = gpu_alloc<float>(size_t(mcfg.vocab_size), g1.allocs, false);
        d_dspark_step_logits = gpu_alloc<float>(size_t(mcfg.vocab_size), g1.allocs, false);
        d_dspark_sm_stats2   = gpu_alloc<float>(2u, g1.allocs, true);
        d_dspark_cand_cnt    = gpu_alloc<int32_t>(1u, g1.allocs, true);
        d_dspark_cand_idx    = gpu_alloc<int32_t>(size_t(kGpuMaxCands), g1.allocs, false);
        d_dspark_cand_exp    = gpu_alloc<float>(size_t(kGpuMaxCands), g1.allocs, false);
        h_dspark_sm_stats2_pin = host_pin_alloc<float>(2u, pin_allocs, true);
        h_dspark_cand_cnt_pin  = host_pin_alloc<int32_t>(1u, pin_allocs, true);
        h_dspark_cand_idx_pin  = host_pin_alloc<int32_t>(size_t(kGpuMaxCands), pin_allocs, false);
        h_dspark_cand_exp_pin  = host_pin_alloc<float>(size_t(kGpuMaxCands), pin_allocs, false);

        dspark_layers.resize(size_t(n_ds));
        for (int i = 0; i < n_ds; ++i) {
            const int l_id = mcfg.num_hidden_layers + i; // 43, 44, 45
            const std::string pfx = "dspark.layers." + std::to_string(i) + ".";
            if (!st.contains(pfx + "input_layernorm.weight")) break;
            auto& DL = dspark_layers[size_t(i)];
            DL.layer_idx   = l_id;
            DL.attn_mhc    = upload_mhc(st, pfx + "attn_mhc", g1);
            DL.mlp_mhc     = upload_mhc(st, pfx + "mlp_mhc",  g1);
            DL.d_in_norm   = upload_bf16_tensor(st, pfx + "input_layernorm.weight", kHiddenSize, g1);
            DL.d_post_norm = upload_bf16_tensor(st, pfx + "post_attention_layernorm.weight", kHiddenSize, g1);
            DL.swa_attn.d_q_down    = upload_bf16_as_f16(st, pfx + "attn.q_down",    1024u * 4096u, g1);
            DL.swa_attn.d_q_norm    = upload_bf16_tensor(st, pfx + "attn.q_norm",    1024u, g1);
            DL.swa_attn.d_q_up      = upload_bf16_as_f16(st, pfx + "attn.q_up",      32768u * 1024u, g1);
            DL.swa_attn.d_kv_proj   = upload_bf16_as_f16(st, pfx + "attn.kv_proj",   512u * 4096u, g1);
            DL.swa_attn.d_kv_norm   = upload_bf16_tensor(st, pfx + "attn.kv_norm",   512u, g1);
            DL.swa_attn.d_attn_sink = upload_f32_tensor(st,  pfx + "attn.attn_sink", 64u, g1);
            DL.swa_attn.d_vha_m     = upload_f32_tensor(st,  pfx + "attn.vha_m",     64u * 64u, g1);
            DL.swa_attn.d_wo_a      = upload_bf16_as_f16(st, pfx + "attn.wo_a",      8u * 1024u * 4096u, g1);
            DL.swa_attn.d_gate_proj = upload_bf16_as_f16(st, pfx + "attn.gate_proj", 8192u * 1024u, g1);
            DL.swa_attn.d_o_proj    = upload_bf16_as_f16(st, pfx + "attn.o_proj",    4096u * 8192u, g1);
            DL.d_swa_ring = gpu_alloc<uint16_t>(size_t(kDsparkSwaRingSize) * 512u, g1.allocs, true);
            DL.d_swa_snap = gpu_alloc<uint16_t>(size_t(kDsparkSwaRingSize) * 512u, g1.allocs, true);

            // Load 3L MoE DSpark spine (fc1_lat, fc2_lat, lat_norm, shared_experts, permuted dual-sigmoid router!)
            DL.moe.d_fc1_lat  = upload_bf16_as_f16(st, pfx + "moe.fc1_lat",  size_t(kLatentDim) * kHiddenSize, g1);
            DL.moe.d_fc2_lat  = upload_bf16_as_f16(st, pfx + "moe.fc2_lat",  size_t(kHiddenSize) * kLatentDim, g1);
            DL.moe.d_lat_norm = upload_bf16_tensor(st, pfx + "moe.lat_norm", kLatentDim, g1);
            DL.moe.d_sh_w1    = upload_bf16_as_f16(st, pfx + "moe.sh_w1",    size_t(kMoeInterSize) * kHiddenSize, g1);
            DL.moe.d_sh_w3    = upload_bf16_as_f16(st, pfx + "moe.sh_w3",    size_t(kMoeInterSize) * kHiddenSize, g1);
            DL.moe.d_sh_w2    = upload_bf16_as_f16(st, pfx + "moe.sh_w2",    size_t(kHiddenSize) * kMoeInterSize, g1);
            if (has_router_perm) {
                auto rw = load_st_as_f32(st, pfx + "moe.router_w01");
                auto cb = load_st_as_f32(st, pfx + "moe.corr_bias");
                auto sp = load_st_as_f32(st, pfx + "moe.scale_param");
                std::vector<float> rw_p(rw.size()), cb_p(cb.size()), sp_p(sp.size());
                const auto& perm = slot_to_raw[size_t(l_id)];
                for (int slot = 0; slot < kNumRoutedExperts; ++slot) {
                    const int raw = int(perm[size_t(slot)]);
                    std::memcpy(rw_p.data() + size_t(slot) * kHiddenSize,
                                rw.data()   + size_t(raw)  * kHiddenSize, size_t(kHiddenSize) * sizeof(float));
                    std::memcpy(rw_p.data() + size_t(kNumRoutedExperts + slot) * kHiddenSize,
                                rw.data()   + size_t(kNumRoutedExperts + raw)  * kHiddenSize, size_t(kHiddenSize) * sizeof(float));
                    cb_p[size_t(slot)] = cb[size_t(raw)];
                    sp_p[size_t(slot)] = sp[size_t(raw)];
                }
                auto rw_f16 = f32_buf_to_f16(rw_p.data(), rw_p.size());
                DL.moe.d_router_w01_f16 = gpu_upload<uint16_t>(rw_f16.data(), rw_f16.size(), g1.allocs);
                DL.moe.d_corr_bias      = gpu_upload<float>(cb_p.data(), cb_p.size(), g1.allocs);
                DL.moe.d_scale_param    = gpu_upload<float>(sp_p.data(), sp_p.size(), g1.allocs);
            } else {
                auto rw = load_st_as_f32(st, pfx + "moe.router_w01");
                auto rw_f16 = f32_buf_to_f16(rw.data(), rw.size());
                DL.moe.d_router_w01_f16 = gpu_upload<uint16_t>(rw_f16.data(), rw_f16.size(), g1.allocs);
                DL.moe.d_corr_bias      = upload_f32_tensor(st, pfx + "moe.corr_bias",  kNumRoutedExperts, g1);
                DL.moe.d_scale_param    = upload_f32_tensor(st, pfx + "moe.scale_param",kNumRoutedExperts, g1);
            }
            // Load uniform 512-expert WHT128-HLUT4 weights across all 5 MI50s for DSpark layer 43+i!
            load_hlut4_layer_experts(l_id);
            hip_check(hipSetDevice(g1.hip_dev), "restore dev1 dspark");
        }

        // Upload markov_w2 [V, 256] FP16 (98.25 MiB) to dev1 GPU for <0.08ms fdot2 Markov bias GEMV!
        d_dspark_markov_w2 = upload_bf16_as_f16(st, "dspark.markov_w2",
                                                size_t(mcfg.vocab_size) * size_t(mcfg.dspark_markov_rank), g1);
        const size_t w1_elems = size_t(mcfg.vocab_size) * size_t(mcfg.dspark_markov_rank);
        h_dspark_markov_w1_T.resize(w1_elems);
        std::memcpy(h_dspark_markov_w1_T.data(),
                    require_st_bytes(st, "dspark.markov_w1_T", w1_elems * 2u), w1_elems * 2u);

        dspark_heads.vocab_size     = mcfg.vocab_size;
        dspark_heads.hidden_size    = kHiddenSize;
        dspark_heads.markov_rank    = mcfg.dspark_markov_rank;
        dspark_heads.block_size     = std::max(1, std::min(mcfg.dspark_block_size > 0 ? mcfg.dspark_block_size : 7, opt.dspark_block_size));
        dspark_block_size           = dspark_heads.block_size;
        dspark_conf_tau             = opt.dspark_conf_tau;
        dspark_heads.noise_token_id = (mcfg.dspark_noise_token_id >= 0) ? mcfg.dspark_noise_token_id : 1024;
        if (st.contains("dspark.conf_proj_w")) {
            dspark_heads.conf_proj_w = load_st_as_f32(st, "dspark.conf_proj_w");
            dspark_heads.has_confidence = true;
        }
        std::memcpy(dspark_heads.multimax_ranges, h_mm_ranges4, 16);
        std::memcpy(dspark_heads.multimax_ts, h_mm_ts4, 16);
        dspark_heads.has_multimax = has_multimax;

        // Cache noise_token_id embedding [4096] BF16 on dev1 once (8 KiB!)
        std::vector<uint16_t> h_noise(kHiddenSize);
        hip_check(hipSetDevice(g0.hip_dev), "dspark noise dev0");
        hip_check(hipMemcpy(h_noise.data(), d_embed_bf16 + size_t(dspark_heads.noise_token_id) * kHiddenSize,
                            kHiddenSize * 2u, hipMemcpyDeviceToHost), "D2H noise emb");
        hip_check(hipSetDevice(g1.hip_dev), "dspark noise dev1");
        d_dspark_noise_emb = gpu_upload<uint16_t>(h_noise.data(), kHiddenSize, g1.allocs);

        st.release_pages();
        has_dspark = true;
        std::printf("[ERNIE55-RT] Loaded 3L MoE DSpark drafter (L43..L45 WHT128-HLUT4, K=%d, noise_id=%d, markov_w2=GPU BF16, conf=%d)\n",
                    dspark_block_size, dspark_heads.noise_token_id, int(dspark_heads.has_confidence));
    }

    // Commit ONLY accepted tokens' [n_commit, 3, 4096] aux features -> main_x -> 3L DSpark SWA context KV rings!
    void commit_dspark_context_kv(int n_commit, int32_t pos0) {
        if (!has_dspark || n_commit <= 0) return;
        auto& g1 = *dev_map[spine_dev1];
        const int n_ds = int(dspark_layers.size());
        hip_check(hipSetDevice(g1.hip_dev), "commit dspark dev1");
        for (int i = 0; i < n_commit; ++i) h_dspark_pos_pin[i] = pos0 + i;
        hip_check(hipMemcpyAsync(g1.d_pos_ids, h_dspark_pos_pin, size_t(n_commit) * sizeof(int32_t), hipMemcpyHostToDevice, g1.stream), "H2D ds ctx pos");
        bf16_linear_bf16(g1, d_dspark_aux_cat, d_dspark_main_proj, g1.d_sub_out, n_commit, kHiddenSize, n_ds * kHiddenSize);
        launch_rmsnorm_bf16(g1.d_sub_out, d_dspark_main_norm, g1.d_x_in, n_commit, kHiddenSize, mcfg.rms_norm_eps, g1.stream);
        for (int i = 0; i < n_ds; ++i) {
            auto& DL = dspark_layers[size_t(i)];
            if (!DL.d_swa_ring) continue;
            bf16_linear_bf16(g1, g1.d_x_in, DL.swa_attn.d_kv_proj, g1.d_tmp_d, n_commit, 512, kHiddenSize);
            launch_rmsnorm_bf16(g1.d_tmp_d, DL.swa_attn.d_kv_norm, g1.d_kv_tmp, n_commit, 512, mcfg.rms_norm_eps, g1.stream);
            launch_dspark_swa_ring_write(g1.d_kv_tmp, g1.d_pos_ids, g1.dsa_rope, DL.d_swa_ring, n_commit, g1.stream);
        }
        hip_check(hipStreamSynchronize(g1.stream), "sync commit dspark");
    }

    // ---- Bundled 3L MoE DSpark Parallel Block Draft (Preview-BF16!) ----
    DSparkHeadsCpu::DraftProposal run_dspark_draft_block(
        int32_t anchor_tok, int32_t pos0, int B, const SamplingParams& sp, std::mt19937_64& rng)
    {
        const auto t_ds0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool saved_prof = profiling_enabled;
        profiling_enabled = false; // isolate DSpark 3L draft cleanly from backbone profile counters!
        auto& g0 = *dev_map[spine_dev0];
        auto& g1 = *dev_map[spine_dev1];
        const int n_ds = int(dspark_layers.size());
        const int V = mcfg.vocab_size;
        const int R = mcfg.dspark_markov_rank;

        // 1. Copy 1 anchor embedding row (8 KiB) dev0 -> dev1 row 0 via pinned staging, and broadcast cached noise_emb to rows 1..B-1!
        hip_check(hipSetDevice(g0.hip_dev), "dspark anchor dev0");
        hip_check(hipMemcpy(h_dspark_anchor_pin, d_embed_bf16 + size_t(anchor_tok) * kHiddenSize,
                            kHiddenSize * 2u, hipMemcpyDeviceToHost), "D2H anchor emb");

        for (int b = 0; b < B; ++b) h_dspark_pos_pin[b] = pos0 + b;

        hip_check(hipSetDevice(g1.hip_dev), "dspark fwd dev1");
        hip_check(hipMemcpyAsync(g1.d_x_in, h_dspark_anchor_pin, kHiddenSize * 2u, hipMemcpyHostToDevice, g1.stream), "H2D anchor");
        for (int b = 1; b < B; ++b) {
            hip_check(hipMemcpyAsync(g1.d_x_in + size_t(b) * kHiddenSize, d_dspark_noise_emb,
                                     kHiddenSize * 2u, hipMemcpyDeviceToDevice, g1.stream), "D2D noise emb");
        }
        hip_check(hipMemcpyAsync(g1.d_pos_ids, h_dspark_pos_pin, size_t(B) * sizeof(int32_t), hipMemcpyHostToDevice, g1.stream), "H2D dspark pos");

        // 2. Run all 3 MoE DSpark layers: non-causal intra-block SWA + Pure-GPU MI50-resident WHT128-HLUT4 MoE (0 P150A waits!)
        for (int i = 0; i < n_ds; ++i) {
            auto& DL = dspark_layers[size_t(i)];
            if (!DL.d_swa_ring) continue;
            if (i == 0) {
                launch_mhc_pre_broadcast_norm_f16(g1.d_x_in, DL.attn_mhc.d_fn_24x16384_f16, DL.attn_mhc.d_scale3,
                                                  DL.attn_mhc.d_base24, DL.d_in_norm, g1.d_res_a,
                                                  g1.d_post_mix, g1.d_comb_mix, g1.d_x_in, B, kHiddenSize,
                                                  mcfg.rms_norm_eps, mcfg.hc_eps, g1.stream);
            } else {
                run_mhc_pre_norm_fast(g1, g1.d_res_a, DL.attn_mhc, DL.d_in_norm, B);
            }
            // Non-causal SWA attention over [max(0, q_pos - 127) .. pos0 + B - 1] with plain RoPE!
            bf16_linear_bf16(g1, g1.d_x_in, DL.swa_attn.d_q_down, g1.d_tmp_d, B, 1024, kHiddenSize);
            launch_rmsnorm_bf16(g1.d_tmp_d, DL.swa_attn.d_q_norm, g1.d_qr, B, 1024, mcfg.rms_norm_eps, g1.stream);
            bf16_linear_bf16(g1, g1.d_x_in, DL.swa_attn.d_kv_proj, g1.d_tmp_d, B, 512, kHiddenSize);
            launch_rmsnorm_bf16(g1.d_tmp_d, DL.swa_attn.d_kv_norm, g1.d_kv_tmp, B, 512, mcfg.rms_norm_eps, g1.stream);
            bf16_linear_bf16(g1, g1.d_qr, DL.swa_attn.d_q_up, g1.d_tmp_a, B, 32768, 1024);
            launch_hca_q_qknorm_rope(g1.d_tmp_a, g1.d_pos_ids, g1.dsa_rope, B, kHcaNumHeads, mcfg.rms_norm_eps, g1.stream);
            launch_dspark_swa_ring_write(g1.d_kv_tmp, g1.d_pos_ids, g1.dsa_rope, DL.d_swa_ring, B, g1.stream);
            launch_dspark_swa_noncausal_block_attn(
                g1.d_tmp_a, DL.d_swa_ring, DL.swa_attn.d_attn_sink, g1.d_tmp_b,
                pos0, B, kSlidingWindow, g1.stream);
            launch_hca_vha_invrope(g1.d_tmp_b, DL.swa_attn.d_vha_m, g1.d_pos_ids, g1.dsa_rope, g1.d_tmp_a, B, g1.stream);
            bf16_linear_bf16(g1, g1.d_qr, DL.swa_attn.d_gate_proj, g1.d_tmp_d, B, 8192, 1024);
            launch_hca_grouped_wo_a_lrga_bf16_f16(g1.d_tmp_a, DL.swa_attn.d_wo_a, g1.d_tmp_d, g1.d_tmp_b, B, kHcaOGroups, kHcaOLoraRank, g1.stream);
            bf16_linear_bf16(g1, g1.d_tmp_b, DL.swa_attn.d_o_proj, g1.d_sub_out, B, kHiddenSize, 8192);
            launch_mhc_post_update(g1.d_sub_out, g1.d_res_a, g1.d_post_mix, g1.d_comb_mix, g1.d_res_b, B, kHiddenSize, g1.stream);

            // Pure-GPU MI50-resident [0..317] WHT128-HLUT4 MoE (99.33%/97.85%/97.29% hit, 0 P150A waits; verify stays 512-expert lossless!)
            run_mhc_pre_norm_fast(g1, g1.d_res_b, DL.mlp_mhc, DL.d_post_norm, B);
            run_moe_core(DL.layer_idx, DL.moe, false, g1, B, true);
            launch_mhc_post_update(g1.d_sub_out, g1.d_res_b, g1.d_post_mix, g1.d_comb_mix, g1.d_res_a, B, kHiddenSize, g1.stream);
        }

        // 3. Contract -> head_hidden [B, 4096], then draft base_logits = TP5_FP8_lm_head(dspark.norm(head_hidden)) (2x less HBM2!)
        launch_mhc_head_contract(g1.d_res_a, d_dspark_hc_fn, d_dspark_hc_scale, d_dspark_hc_base,
                                 nullptr, g1.d_tmp_a, B, kHiddenSize, mcfg.rms_norm_eps, mcfg.hc_eps, g1.stream);
        launch_rmsnorm_bf16(g1.d_tmp_a, d_dspark_norm, g1.d_x_in, B, kHiddenSize, mcfg.rms_norm_eps, g1.stream);
        compute_lm_head_tp5(g1.d_x_in, B, false, nullptr, g1.d_logits_f32, true);

        const bool want_conf = (dspark_heads.has_confidence && dspark_conf_tau > 0.0f);
        std::vector<uint16_t> h_head_bf16;
        std::vector<float> h_head_f32, me_f32;
        if (want_conf) {
            h_head_bf16.resize(size_t(B) * kHiddenSize);
            h_head_f32.resize(size_t(B) * kHiddenSize);
            me_f32.resize(size_t(std::max(1, R)));
            hip_check(hipMemcpyAsync(h_head_bf16.data(), g1.d_tmp_a, h_head_bf16.size() * 2u, hipMemcpyDeviceToHost, g1.stream), "D2H head_h");
            hip_check(hipStreamSynchronize(g1.stream), "sync head_h");
            for (size_t idx = 0; idx < h_head_bf16.size(); ++idx) h_head_f32[idx] = bf16_to_f32(h_head_bf16[idx]);
        }

        // 4. GPU-Accelerated Sequential Markov Chain + Post-Markov MultiMax + GPU Sparse Top-P Sampling!
        DSparkHeadsCpu::DraftProposal out;
        out.draft_tokens.reserve(size_t(B));
        out.draft_sparse.reserve(size_t(B));
        out.step_confidences.reserve(size_t(B));

        const bool prob_mode = (sp.draft_sample_method == DraftSampleMethod::kProbabilistic && sp.temperature > 1e-6f);
        std::vector<float> h_step_logits;

        int32_t prev_tok = anchor_tok;
        float cum_survival = 1.0f;
        for (int b = 0; b < B; ++b) {
            const int32_t pt = std::clamp(prev_tok, 0, V - 1);
            const uint16_t* me_row = h_dspark_markov_w1_T.data() + size_t(pt) * size_t(R);
            float conf = 1.0f;
            if (want_conf) {
                for (int r = 0; r < R; ++r) me_f32[size_t(r)] = bf16_to_f32(me_row[size_t(r)]);
                conf = dspark_heads.compute_confidence(h_head_f32.data() + size_t(b) * kHiddenSize, me_f32.data());
                cum_survival *= conf;
                if (b > 0 && cum_survival < dspark_conf_tau) break;
            }
            hip_check(hipMemcpyAsync(d_dspark_me_bf16, me_row, size_t(R) * 2u, hipMemcpyHostToDevice, g1.stream), "H2D me");
            bf16_linear_f32(g1, d_dspark_me_bf16, d_dspark_markov_w2, d_dspark_bias_f32, 1, V, R);
            launch_dspark_add_bias_multimax_f32(
                g1.d_logits_f32 + size_t(b) * size_t(V), d_dspark_bias_f32,
                has_multimax ? d_mm_ranges4 : nullptr, has_multimax ? d_mm_ts4 : nullptr,
                d_dspark_step_logits, V, g1.stream);

            int32_t tok = 0;
            if (!prob_mode) {
                launch_argmax_f32_fast(d_dspark_step_logits, g1.d_argmax_v64, g1.d_argmax_i64, g1.d_tok_ids, V, g1.stream);
                hip_check(hipMemcpyAsync(&tok, g1.d_tok_ids, sizeof(int32_t), hipMemcpyDeviceToHost, g1.stream), "D2H ds tok");
                hip_check(hipStreamSynchronize(g1.stream), "sync ds tok");
            } else {
                // Wave-8 GPU Sparse Softmax + Top-Candidates Compaction (0.04 ms vs 3.5 ms CPU! Only copies ~1 KiB D2H!)
                launch_gpu_softmax_sparse_top_candidates(
                    d_dspark_step_logits, V, sp.temperature,
                    g1.d_argmax_v64, d_dspark_sm_stats2, d_dspark_cand_cnt,
                    d_dspark_cand_idx, d_dspark_cand_exp, kGpuMaxCands, g1.stream);
                hip_check(hipMemcpyAsync(h_dspark_sm_stats2_pin, d_dspark_sm_stats2, 2u * sizeof(float), hipMemcpyDeviceToHost, g1.stream), "D2H sm stats");
                hip_check(hipMemcpyAsync(h_dspark_cand_cnt_pin, d_dspark_cand_cnt, sizeof(int32_t), hipMemcpyDeviceToHost, g1.stream), "D2H sm cnt");
                hip_check(hipStreamSynchronize(g1.stream), "sync sm stats");
                const int n_cand = h_dspark_cand_cnt_pin[0];
                const double sum_exp = double(h_dspark_sm_stats2_pin[1]);
                SparseProbs q_b;
                if (n_cand > 0 && n_cand <= kGpuMaxCands && sum_exp > 0.0) {
                    hip_check(hipMemcpyAsync(h_dspark_cand_idx_pin, d_dspark_cand_idx, size_t(n_cand) * sizeof(int32_t), hipMemcpyDeviceToHost, g1.stream), "D2H cidx");
                    hip_check(hipMemcpyAsync(h_dspark_cand_exp_pin, d_dspark_cand_exp, size_t(n_cand) * sizeof(float),   hipMemcpyDeviceToHost, g1.stream), "D2H cexp");
                    hip_check(hipStreamSynchronize(g1.stream), "sync sm cands");
                    gpu_candidates_to_sparse_probs(h_dspark_cand_idx_pin, h_dspark_cand_exp_pin, n_cand, sum_exp,
                                                   sp.top_k, sp.top_p, q_b);
                } else {
                    if (int(h_step_logits.size()) != V) h_step_logits.resize(size_t(V));
                    hip_check(hipMemcpyAsync(h_step_logits.data(), d_dspark_step_logits, size_t(V) * sizeof(float), hipMemcpyDeviceToHost, g1.stream), "D2H ds log fb");
                    hip_check(hipStreamSynchronize(g1.stream), "sync ds log fb");
                    logits_to_sparse_probs_fast(h_step_logits.data(), V, sp.temperature, sp.top_k, sp.top_p, q_b);
                }
                tok = sample_from_sparse_probs(q_b, rng);
                out.draft_sparse.push_back(std::move(q_b));
            }
            out.draft_tokens.push_back(tok);
            out.step_confidences.push_back(conf);
            prev_tok = tok;
        }
        profiling_enabled = saved_prof;
        if (profiling_enabled) {
            prof.dspark_draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ds0).count();
            ++prof.dspark_draft_calls;
        }
        return out;
    }

    // ---- TP=4 / Single-GPU Attention Sublayer (HCA Batched FP16 GEMM + DSA Batched GEMM FlashIndexer & FP16 SGEMM-DSA!) ----
    void run_attention_tp(LayerSpineGpu& L, int M, int32_t pos0) {
        const auto t_a0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int tp = L.tp_size;
        const auto dev_for_r = [&](int r) -> int { return L.is_echo ? L.owner_dev : tp_devs[r]; };
        constexpr int kGemmStrip = 512;

        if (!L.is_dsa) {
            const int h_loc = kHcaNumHeads / tp;
            const int g_loc = kHcaOGroups / tp;
            const int max_comp = (pos0 + M - 1) / kHcaCompressRatio;

            // Phase A: Parallel Q/KV/Compressor/Attention across TP ranks (16 heads/GPU on TP4!)
            const auto t_hp0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& H = L.hca_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "hca A dev");

                bf16_linear_f32(w, w.d_x_in, H.d_qkv_comp, w.d_y_f32, M, 2560, kHiddenSize);
                launch_hca_split_qkv_comp_norm(w.d_y_f32, H.d_q_norm, H.d_kv_norm,
                                               w.d_qr, w.d_kv_tmp, w.d_f32_tmp0, w.d_f32_tmp1,
                                               M, mcfg.rms_norm_eps, w.stream);

                bf16_linear_bf16(w, w.d_qr, H.d_q_up, w.d_tmp_a, M, h_loc * 512, 1024);
                launch_hca_q_qknorm_rope(w.d_tmp_a, w.d_pos_ids, w.yarn_rope, M, h_loc, mcfg.rms_norm_eps, w.stream);
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.hca_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hp0).count();
            }
            const auto t_ha0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& H = L.hca_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "hca A2 dev");

                if (M <= 8) {
                    int t0 = 0;
                    const size_t q_tok_stride = size_t(h_loc) * 512u;
                    while (t0 < M) {
                        const int32_t p0 = pos0 + t0;
                        const int rem_in_blk = kHcaCompressRatio - (p0 % kHcaCompressRatio);
                        const int seg = std::min(M - t0, rem_in_blk);
                        const int num_comp = p0 / kHcaCompressRatio;
                        launch_hca_kv_rope_swa_insert(w.d_kv_tmp + size_t(t0) * 512u, w.d_pos_ids + t0, w.yarn_rope,
                                                      L.d_hca_swa_ring_tp[r], seg, w.stream);
                        launch_hca_compressor_save_state(w.d_f32_tmp0 + size_t(t0) * 512u, w.d_f32_tmp1 + size_t(t0) * 512u,
                                                         H.d_comp_ape, w.d_pos_ids + t0,
                                                         L.d_hca_state_kv_tp[r], L.d_hca_state_sc_tp[r], seg, w.stream);
                        launch_hca_attn_c128_swa_sink_splitk(
                            w.d_tmp_a + size_t(t0) * q_tok_stride, L.d_hca_comp_kv_tp[r], L.d_hca_swa_ring_tp[r],
                            H.d_attn_sink, w.d_hca_o_tot_f32, w.d_hca_ml_f32,
                            w.d_tmp_b + size_t(t0) * q_tok_stride, h_loc, p0, seg, num_comp, w.stream);
                        const int32_t p_last = p0 + seg - 1;
                        if ((p_last % kHcaCompressRatio) == (kHcaCompressRatio - 1)) {
                            const int blk = p_last / kHcaCompressRatio;
                            launch_hca_compressor_reduce_norm_rope(
                                L.d_hca_state_kv_tp[r], L.d_hca_state_sc_tp[r], H.d_comp_norm, w.yarn_rope,
                                blk * kHcaCompressRatio, L.d_hca_comp_kv_tp[r] + size_t(blk) * 512u, mcfg.rms_norm_eps, w.stream);
                        }
                        t0 += seg;
                    }
                } else {
                    launch_hca_kv_rope_inplace(w.d_kv_tmp, w.d_pos_ids, w.yarn_rope, M, w.stream);
                    launch_hca_prefill_compress_blocks(
                        w.d_f32_tmp0, w.d_f32_tmp1, H.d_comp_ape,
                        L.d_hca_state_kv_tp[r], L.d_hca_state_sc_tp[r], H.d_comp_norm, w.yarn_rope,
                        pos0, M, L.d_hca_comp_kv_tp[r], mcfg.rms_norm_eps, w.stream);

                    if (tp == 1 && M > 256) {
                        launch_hca_prefill_attn_c128_swa_sink(
                            w.d_tmp_a, L.d_hca_comp_kv_tp[r], L.d_hca_swa_ring_tp[r], w.d_kv_tmp,
                            H.d_attn_sink, w.d_tmp_b, h_loc, pos0, M, w.stream);
                    } else {
                        // Hardware Batched FP16->FP32 GEMM Compressed-KV (constant 512-row strip!) + 8-Head LDS Exact BF16->FP32 SWA!
                        // Bit-for-bit identical across M=137, M=512, M=1024, and warm -> guarantees cos=1.00000000!
                        constexpr int kBlkTile = 256;
                        const size_t ml_stride = size_t(max_chunk) * size_t(h_loc);
                        float* d_m_tot = w.d_hca_ml_f32;
                        float* d_l_tot = w.d_hca_ml_f32 + ml_stride;
                        float* d_m_blk = w.d_hca_ml_f32 + ml_stride * 2u;
                        float* d_l_blk = w.d_hca_ml_f32 + ml_stride * 3u;
                        const float alpha = 1.0f, beta = 0.0f;

                        if (max_comp > 0) {
                            const int max_comp_pad = ((max_comp + kBlkTile - 1) / kBlkTile) * kBlkTile;
                            launch_bf16_to_f16(L.d_hca_comp_kv_tp[r], w.d_hca_kcomp_f16, max_comp * 512, w.stream);
                            if (max_comp < max_comp_pad) {
                                hip_check(hipMemsetAsync(w.d_hca_kcomp_f16 + size_t(max_comp) * 512u, 0,
                                                         size_t(max_comp_pad - max_comp) * 512u * sizeof(uint16_t), w.stream), "zero hca k pad");
                            }
                            const int n_rows_pad = kGemmStrip * h_loc;
                            for (int m0 = 0; m0 < M; m0 += kGemmStrip) {
                                const int B = std::min(kGemmStrip, M - m0);
                                const int n_rows = B * h_loc;
                                const size_t row_off = size_t(m0) * size_t(h_loc);
                                launch_bf16_to_f16(w.d_tmp_a + row_off * 512u, w.d_x_f16, n_rows * 512, w.stream);
                                if (n_rows < n_rows_pad) {
                                    hip_check(hipMemsetAsync(w.d_x_f16 + size_t(n_rows) * 512u, 0,
                                                             size_t(n_rows_pad - n_rows) * 512u * sizeof(uint16_t), w.stream), "zero hca q pad");
                                    hip_check(hipMemsetAsync(w.d_tmp_d + size_t(n_rows) * size_t(kBlkTile), 0,
                                                             size_t(n_rows_pad - n_rows) * size_t(kBlkTile) * sizeof(uint16_t), w.stream), "zero hca comp p pad");
                                }
                                for (int blk0 = 0; blk0 < max_comp; blk0 += kBlkTile) {
                                    const uint16_t* d_k_tile = w.d_hca_kcomp_f16 + size_t(blk0) * 512u;
                                    rocblas_check(rocblas_gemm_ex(
                                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                        kBlkTile, n_rows_pad, 512, &alpha,
                                        d_k_tile, rocblas_datatype_f16_r, 512,
                                        w.d_x_f16, rocblas_datatype_f16_r, 512,
                                        &beta,
                                        w.d_y_f32, rocblas_datatype_f32_r, kBlkTile,
                                        w.d_y_f32, rocblas_datatype_f32_r, kBlkTile,
                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "hca comp QK gemm_ex");
                                    launch_hca_comp_softmax_f32(w.d_y_f32, d_m_blk, d_l_blk, w.d_tmp_d, B, h_loc, pos0 + m0, blk0, kBlkTile, w.stream);
                                    rocblas_check(rocblas_gemm_ex(
                                        w.blas, rocblas_operation_none, rocblas_operation_none,
                                        512, n_rows_pad, kBlkTile, &alpha,
                                        d_k_tile, rocblas_datatype_f16_r, 512,
                                        w.d_tmp_d, rocblas_datatype_f16_r, kBlkTile,
                                        &beta,
                                        w.d_y_f32, rocblas_datatype_f32_r, 512,
                                        w.d_y_f32, rocblas_datatype_f32_r, 512,
                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "hca comp PV gemm_ex");
                                    launch_hca_comp_merge_tile_f32(
                                        w.d_hca_o_tot_f32 + row_off * 512u, d_m_tot + row_off, d_l_tot + row_off,
                                        w.d_y_f32, d_m_blk, d_l_blk, n_rows, blk0 == 0, w.stream);
                                }
                            }
                        }

                        launch_hca_prefill_swa_merge_sink(
                            w.d_tmp_a, L.d_hca_swa_ring_tp[r], w.d_kv_tmp,
                            (max_comp > 0) ? w.d_hca_o_tot_f32 : nullptr,
                            (max_comp > 0) ? d_m_tot : nullptr,
                            (max_comp > 0) ? d_l_tot : nullptr,
                            H.d_attn_sink, w.d_tmp_b, h_loc, pos0, M, w.stream);
                    }
                    launch_hca_prefill_commit_tail(
                        w.d_kv_tmp, w.d_f32_tmp0, w.d_f32_tmp1, H.d_comp_ape,
                        pos0, M, L.d_hca_swa_ring_tp[r], L.d_hca_state_kv_tp[r], L.d_hca_state_sc_tp[r], w.stream);
                }
            }

            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.hca_attn_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ha0).count();
            }

            // Phase B: 1D Contiguous Rank-Major AllGather [M, h_loc, 512] -> [tp_size, M, h_loc, 512] (ZERO hipMemcpy2DAsync!)
            if (tp > 1) {
                tp_allgather_heads_bf16(M, kHcaNumHeads, 512,
                                        &Mi50ExpertWorker::d_tmp_b, &Mi50ExpertWorker::d_tmp_a, h_tp_oattn_pin);
            }

            // Phase C: Parallel 16x-Reuse Rank-Major VHA + invRoPE + 2-group FP16 wo_a * sigmoid(gate) + o_proj/4
            const auto t_ho0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& H = L.hca_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "hca C dev");

                const uint16_t* d_o_all = (tp > 1) ? w.d_tmp_a : w.d_tmp_b;
                uint16_t* d_vha_out     = (tp > 1) ? w.d_tmp_b : w.d_tmp_a;
                uint16_t* d_wo_out      = (tp > 1) ? w.d_tmp_a : w.d_tmp_b;

                launch_hca_vha_invrope_shard(d_o_all, H.d_vha_m, w.d_pos_ids, w.yarn_rope,
                                             d_vha_out, M, tp, r * h_loc, h_loc, w.stream);
                bf16_linear_bf16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, g_loc * kHcaOLoraRank, 1024);

                if (M <= 8) {
                    launch_hca_grouped_wo_a_lrga_bf16_f16(d_vha_out, H.d_wo_a, w.d_tmp_d, d_wo_out, M, g_loc, kHcaOLoraRank, w.stream);
                } else {
                    const float alpha = 1.0f, beta = 0.0f;
                    for (int m0 = 0; m0 < M; m0 += kGemmStrip) {
                        const int cur = std::min(kGemmStrip, M - m0);
                        launch_bf16_to_f16(d_vha_out + size_t(m0) * size_t(g_loc) * 4096u, w.d_x_f16, cur * g_loc * 4096, w.stream);
                        if (cur < kGemmStrip) {
                            hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(g_loc) * 4096u, 0,
                                                     size_t(kGemmStrip - cur) * size_t(g_loc) * 4096u * sizeof(uint16_t), w.stream), "zero wo_a pad");
                        }
                        rocblas_check(rocblas_gemm_strided_batched_ex(
                            w.blas, rocblas_operation_transpose, rocblas_operation_none,
                            kHcaOLoraRank, kGemmStrip, 4096, &alpha,
                            H.d_wo_a,  rocblas_datatype_f16_r, 4096, int64_t(kHcaOLoraRank) * 4096LL,
                            w.d_x_f16, rocblas_datatype_f16_r, g_loc * 4096, 4096LL,
                            &beta,
                            w.d_y_f32, rocblas_datatype_f32_r, g_loc * kHcaOLoraRank, int64_t(kHcaOLoraRank),
                            w.d_y_f32, rocblas_datatype_f32_r, g_loc * kHcaOLoraRank, int64_t(kHcaOLoraRank),
                            g_loc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "hca wo_a gemm_ex batched");
                        const size_t z_off = size_t(m0) * size_t(g_loc) * size_t(kHcaOLoraRank);
                        launch_f32_mul_sigmoid_to_bf16(w.d_y_f32, w.d_tmp_d + z_off, d_wo_out + z_off, cur * g_loc * kHcaOLoraRank, w.stream);
                    }
                }
                bf16_linear_bf16(w, d_wo_out, H.d_o_proj, w.d_sub_out, M, kHiddenSize, g_loc * kHcaOLoraRank);
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.hca_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ho0).count();
            }

            // Phase D: Canonical TP AllReduce [M, 4096]
            if (tp > 1) {
                tp_allreduce_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.hca_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_a0).count();
            }
        } else {
            // mqa_dsa: TP4 parallel Q/W_UK + single-copy KV / Batched GEMM FlashIndexer / FP16 SGEMM-DSA on owner_dev + TP4 parallel W_UV/VHA/o_proj!
            const int h_loc = kDsaNumHeads / tp;
            const int ro = L.owner_tp_rank;
            auto& g_own = *dev_map[L.owner_dev];
            const auto& D_own = L.dsa_tp[ro];

            // Phase A: Parallel q_a, q_b/4, RoPE, W_UK/4 (+ owner kv_a & indexer prep!)
            const auto t_dp0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dsa_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dsa A dev");

                bf16_linear_bf16(w, w.d_x_in, D.d_q_a, w.d_tmp_d, M, 1024, kHiddenSize);
                launch_rmsnorm_bf16(w.d_tmp_d, D.d_q_a_norm, w.d_qr, M, 1024, mcfg.rms_norm_eps, w.stream);
                bf16_linear_bf16(w, w.d_qr, D.d_q_b, w.d_tmp_a, M, h_loc * 256, 1024);

                if (r == ro) {
                    bf16_linear_bf16(w, w.d_x_in, D.d_kv_a, w.d_kv_tmp, M, 576, kHiddenSize);
                    launch_dsa_prefill_kv_norm_rope_append(
                        w.d_tmp_a, w.d_kv_tmp, D.d_kv_a_norm, w.d_pos_ids, w.dsa_rope,
                        L.d_dsa_mla_kv, M, h_loc, opt.max_kv, mcfg.rms_norm_eps, w.stream);

                    bf16_linear_bf16(w, w.d_qr,   D.d_idx_wq_b,   w.d_tmp_b, M, 8192, 1024);
                    bf16_linear_bf16(w, w.d_x_in, D.d_idx_wk,     w.d_tmp_d, M, 128, kHiddenSize);
                    launch_dsa_indexer_prep_neox_wht128(w.d_tmp_b, w.d_tmp_d, D.d_idx_kn_w, D.d_idx_kn_b,
                                                        w.d_pos_ids, w.dsa_rope, L.d_dsa_ik_cache, M, opt.max_kv, w.stream);
                    bf16_linear_bf16(w, w.d_x_in, D.d_idx_w_proj, w.d_tmp_d, M, 64, kHiddenSize);
                } else {
                    launch_dsa_mla_rope_kv_append(
                        w.d_tmp_a, nullptr, nullptr, w.d_pos_ids, w.dsa_rope,
                        nullptr, M, h_loc, 0, w.stream);
                }

                if (M <= 8) {
                    launch_dsa_q_absorb_uk_f16(w.d_tmp_a, D.d_kv_b, w.d_tmp_c, M, h_loc, w.stream);
                } else {
                    const float alpha = 1.0f, beta = 0.0f;
                    for (int m0 = 0; m0 < M; m0 += kGemmStrip) {
                        const int cur = std::min(kGemmStrip, M - m0);
                        launch_bf16_to_f16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 256u, w.d_x_f16, cur * h_loc * 256, w.stream);
                        if (cur < kGemmStrip) {
                            hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(h_loc) * 256u, 0,
                                                     size_t(kGemmStrip - cur) * size_t(h_loc) * 256u * sizeof(uint16_t), w.stream), "zero uk pad");
                        }
                        rocblas_check(rocblas_gemm_strided_batched_ex(
                            w.blas, rocblas_operation_none, rocblas_operation_none,
                            512, kGemmStrip, 192, &alpha,
                            D.d_kv_b,  rocblas_datatype_f16_r, 512, 448LL * 512LL,
                            w.d_x_f16, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                            &beta,
                            w.d_y_f32, rocblas_datatype_f32_r, h_loc * 576, 576LL,
                            w.d_y_f32, rocblas_datatype_f32_r, h_loc * 576, 576LL,
                            h_loc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa q_absorb gemm_ex batched");
                        launch_f32_to_bf16(w.d_y_f32, w.d_tmp_c + size_t(m0) * size_t(h_loc) * 576u, cur * h_loc * 576, w.stream);
                        launch_dsa_copy_qrope_bf16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 256u,
                                                   w.d_tmp_c + size_t(m0) * size_t(h_loc) * 576u, cur, h_loc, w.stream);
                    }
                }
            }

            // Phase B: 1D Contiguous Gather q_abs [M, h_loc, 576] -> owner g_own.d_tmp_c [M, 64, 576] (ZERO hipMemcpy2DAsync!)
            if (tp > 1) {
                const size_t row_elems = size_t(h_loc) * 576u;
                const size_t slab_elems = size_t(M) * row_elems;
                const size_t slab_b = slab_elems * sizeof(uint16_t);
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    hip_check(hipSetDevice(w.hip_dev), "dsa qabs d2h dev");
                    hip_check(hipMemcpyAsync(h_tp_qabs_rm_pin + size_t(r) * slab_elems, w.d_tmp_c,
                                             slab_b, hipMemcpyDeviceToHost, w.stream), "dsa qabs 1d d2h");
                }
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    hip_check(hipSetDevice(w.hip_dev), "dsa qabs sync dev");
                    hip_check(hipStreamSynchronize(w.stream), "dsa qabs sync");
                }
                #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (M >= 32)
                for (int m = 0; m < M; ++m) {
                    uint16_t* dst = h_tp_qabs_pin + size_t(m) * size_t(kDsaNumHeads) * 576u;
                    for (int r = 0; r < tp; ++r) {
                        std::memcpy(dst + size_t(r) * row_elems,
                                    h_tp_qabs_rm_pin + size_t(r) * slab_elems + size_t(m) * row_elems,
                                    row_elems * sizeof(uint16_t));
                    }
                }
                hip_check(hipSetDevice(g_own.hip_dev), "dsa qabs h2d dev");
                hip_check(hipMemcpyAsync(g_own.d_tmp_c, h_tp_qabs_pin, size_t(tp) * slab_b,
                                         hipMemcpyHostToDevice, g_own.stream), "dsa qabs 1d h2d");
            } else {
                hip_check(hipSetDevice(g_own.hip_dev), "dsa own dev");
            }

            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp0).count();
            }

            // Phase C: Single-copy KV Attention on g_own (Strict pos < 2176 dense vs pos >= 2176 sparse split!)
            constexpr int kStrideSel = kDsaIndexTopK + kSlidingWindow;
            const int n_dense = std::max(0, std::min(M, kStrideSel - pos0));
            if (n_dense > 0) {
                const auto t_dm0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (M <= 8) {
                    launch_dsa_prefill_mla_dense_causal_sink(
                        g_own.d_tmp_c, L.d_dsa_mla_kv, D_own.d_attn_sink, g_own.d_tmp_a, kDsaNumHeads, pos0, n_dense, g_own.stream);
                } else {
                    // Constant-tile Batched FP16->FP32 GEMM DSA Dense Causal (B_strip=64, kTileK=256 -> 100% chunk-invariant!)
                    // Uses disjoint slices inside g_own.d_x_f16 (16 MiB) -> ZERO clobbering of g_own.d_tmp_d / d_tmp_b!
                    constexpr int kBStrip = 64;
                    constexpr int kTileK  = 256;
                    constexpr int kRowsPad = kBStrip * kDsaNumHeads; // 4096
                    const float alpha = 1.0f, beta = 0.0f;
                    uint16_t* d_q_strip_f16 = g_own.d_x_f16;             // [4096, 576] = 2,359,296 halfs
                    uint16_t* d_k_tile_f16  = g_own.d_x_f16 + 4194304u;  // [256, 576]  =   147,456 halfs
                    uint16_t* d_probs_f16   = g_own.d_x_f16 + 5242880u;  // [4096, 256] = 1,048,576 halfs (ends at 6.29M < 8.38M!)
                    float* d_m_tot = g_own.d_hca_ml_f32;
                    float* d_l_tot = g_own.d_hca_ml_f32 + size_t(kRowsPad);
                    float* d_m_blk = g_own.d_hca_ml_f32 + size_t(kRowsPad) * 2u;
                    float* d_l_blk = g_own.d_hca_ml_f32 + size_t(kRowsPad) * 3u;

                    for (int m0 = 0; m0 < n_dense; m0 += kBStrip) {
                        const int B = std::min(kBStrip, n_dense - m0);
                        const int n_rows = B * kDsaNumHeads;
                        const int32_t pos0_strip = pos0 + m0;
                        const int k_lim = pos0_strip + B;
                        launch_bf16_to_f16(g_own.d_tmp_c + size_t(m0) * size_t(kDsaNumHeads) * 576u,
                                           d_q_strip_f16, n_rows * 576, g_own.stream);
                        if (n_rows < kRowsPad) {
                            hip_check(hipMemsetAsync(d_q_strip_f16 + size_t(n_rows) * 576u, 0,
                                                     size_t(kRowsPad - n_rows) * 576u * sizeof(uint16_t), g_own.stream), "zero dsa dense q pad");
                            hip_check(hipMemsetAsync(d_probs_f16 + size_t(n_rows) * size_t(kTileK), 0,
                                                     size_t(kRowsPad - n_rows) * size_t(kTileK) * sizeof(uint16_t), g_own.stream), "zero dsa dense p pad");
                        }
                        for (int k0 = 0; k0 < k_lim; k0 += kTileK) {
                            const int k_cur = std::min(kTileK, k_lim - k0);
                            launch_bf16_to_f16(L.d_dsa_mla_kv + size_t(k0) * 576u, d_k_tile_f16, k_cur * 576, g_own.stream);
                            if (k_cur < kTileK) {
                                hip_check(hipMemsetAsync(d_k_tile_f16 + size_t(k_cur) * 576u, 0,
                                                         size_t(kTileK - k_cur) * 576u * sizeof(uint16_t), g_own.stream), "zero dsa dense k pad");
                            }
                            rocblas_check(rocblas_gemm_ex(
                                g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                kTileK, kRowsPad, 576, &alpha,
                                d_k_tile_f16,  rocblas_datatype_f16_r, 576,
                                d_q_strip_f16, rocblas_datatype_f16_r, 576,
                                &beta,
                                g_own.d_y_f32, rocblas_datatype_f32_r, kTileK,
                                g_own.d_y_f32, rocblas_datatype_f32_r, kTileK,
                                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa dense QK gemm_ex");
                            launch_dsa_dense_softmax_tile_f32(
                                g_own.d_y_f32, d_m_blk, d_l_blk, d_probs_f16,
                                B, kDsaNumHeads, pos0_strip, k0, kTileK, 0.0625f, g_own.stream);
                            rocblas_check(rocblas_gemm_ex(
                                g_own.blas, rocblas_operation_none, rocblas_operation_none,
                                512, kRowsPad, kTileK, &alpha,
                                d_k_tile_f16,  rocblas_datatype_f16_r, 576,
                                d_probs_f16,   rocblas_datatype_f16_r, kTileK,
                                &beta,
                                g_own.d_y_f32, rocblas_datatype_f32_r, 512,
                                g_own.d_y_f32, rocblas_datatype_f32_r, 512,
                                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa dense PV gemm_ex");
                            launch_hca_comp_merge_tile_f32(
                                g_own.d_hca_o_tot_f32, d_m_tot, d_l_tot,
                                g_own.d_y_f32, d_m_blk, d_l_blk, n_rows, k0 == 0, g_own.stream);
                        }
                        launch_dsa_dense_finalize_sink_bf16(
                            g_own.d_hca_o_tot_f32, d_m_tot, d_l_tot, D_own.d_attn_sink,
                            g_own.d_tmp_a + size_t(m0) * size_t(kDsaNumHeads) * 512u,
                            B, kDsaNumHeads, g_own.stream);
                    }
                }
                if (profiling_enabled) {
                    hip_check(hipStreamSynchronize(g_own.stream), "prof dsa dense");
                    prof.dsa_mla_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dm0).count();
                }
            }
            if (n_dense < M) {
                const auto t_df0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const bool do_fi = (use_flashindexer && D_own.d_fi_gate != nullptr && D_own.d_fi_bias != nullptr);
                if (do_fi) {
                    bf16_linear_f32(g_own, g_own.d_x_in, D_own.d_fi_gate, g_own.d_fi_gates, M, 64, kHiddenSize);
                    launch_flashindexer_select_heads(
                        g_own.d_fi_gates, D_own.d_fi_bias, g_own.d_tmp_b, g_own.d_tmp_d, g_own.d_fi_q8, g_own.d_fi_w8, M, g_own.stream);
                }
                const int max_hist = std::max(1, (pos0 + M - 1) - kSlidingWindow + 1);
                const int max_batch_t = std::max(1, std::min(M - n_dense, int(g_own.idx_scores_cap / size_t(max_hist))));
                for (int t0 = n_dense; t0 < M; t0 += max_batch_t) {
                    const int b_sc = std::min(max_batch_t, M - t0);
                    const int b_max_hist = std::max(1, (pos0 + t0 + b_sc - 1) - kSlidingWindow + 1);
                    if (b_max_hist > kDsaIndexTopK) {
                        if (do_fi && M > 8) {
                            // Wave-8 Hardware Batched FP16->FP32 GEMM FlashIndexer (constant kTileK=4096, kSubB=64 -> 100% INV-1d chunk-invariant!)
                            constexpr int kTileK = 4096;
                            constexpr int kSubB  = 64;
                            const int b_pad = ((b_sc + kSubB - 1) / kSubB) * kSubB;
                            uint16_t* d_q_all_f16 = g_own.d_x_f16;             // up to [1024*8, 128] = 1,048,576 halfs (2 MiB)
                            uint16_t* d_k_tile_f16 = g_own.d_x_f16 + 2097152u; // [16384, 128] = 2,097,152 halfs (4 MiB)
                            launch_bf16_to_f16(g_own.d_fi_q8 + size_t(t0) * 8u * 128u, d_q_all_f16, b_sc * 8 * 128, g_own.stream);
                            if (b_sc < b_pad) {
                                hip_check(hipMemsetAsync(d_q_all_f16 + size_t(b_sc) * 8u * 128u, 0,
                                                         size_t(b_pad - b_sc) * 8u * 128u * sizeof(uint16_t), g_own.stream), "zero fi q pad");
                            }
                            const float alpha = 1.0f, beta = 0.0f;
                            for (int k0 = 0; k0 < b_max_hist; k0 += kTileK) {
                                const int k_cur = std::min(kTileK, b_max_hist - k0);
                                launch_bf16_to_f16(L.d_dsa_ik_cache + size_t(k0) * 128u, d_k_tile_f16, k_cur * 128, g_own.stream);
                                if (k_cur < kTileK) {
                                    hip_check(hipMemsetAsync(d_k_tile_f16 + size_t(k_cur) * 128u, 0,
                                                             size_t(kTileK - k_cur) * 128u * sizeof(uint16_t), g_own.stream), "zero fi k pad");
                                }
                                for (int b0 = 0; b0 < b_sc; b0 += kSubB) {
                                    const int b_cur = std::min(kSubB, b_sc - b0);
                                    rocblas_check(rocblas_gemm_ex(
                                        g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                        kTileK, kSubB * 8, 128, &alpha,
                                        d_k_tile_f16, rocblas_datatype_f16_r, 128,
                                        d_q_all_f16 + size_t(b0) * 8u * 128u, rocblas_datatype_f16_r, 128,
                                        &beta,
                                        g_own.d_y_f32, rocblas_datatype_f32_r, kTileK,
                                        g_own.d_y_f32, rocblas_datatype_f32_r, kTileK,
                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "fi gemm_ex");
                                    launch_flashindexer_relu_wsum_f32(
                                        g_own.d_y_f32, g_own.d_fi_w8,
                                        g_own.d_idx_scores + size_t(b0) * size_t(b_max_hist),
                                        pos0, t0 + b0, b_cur, k0, kTileK, b_max_hist, g_own.stream);
                                }
                            }
                        } else if (do_fi) {
                            // Wave-7 4-lane VALU DPP FlashIndexer (0 ds_bpermute, 0 bit-diffs, LDS Q-stationary across B_sc=1..8!)
                            launch_dsa_indexer_score_topk8_dpp4(
                                g_own.d_fi_q8, g_own.d_fi_w8, L.d_dsa_ik_cache, g_own.d_idx_scores,
                                pos0, t0, b_sc, b_max_hist, g_own.stream);
                        } else {
                            launch_dsa_indexer_score_full_batch(
                                g_own.d_tmp_b, g_own.d_tmp_d, L.d_dsa_ik_cache, g_own.d_idx_scores,
                                pos0, t0, b_sc, b_max_hist, g_own.stream);
                        }
                    }
                    // Unified Wave-6 64-row batched zero-skip radix selector across ALL b_sc=1..512 (100% INV-1d bit-exact!)
                    launch_dsa_select_topk2048_wave6(
                        g_own.d_idx_scores, reinterpret_cast<uint32_t*>(g_own.d_y_f32),
                        g_own.d_sel_idx, g_own.d_sel_cnts,
                        pos0, t0, b_sc, b_max_hist, kStrideSel, g_own.stream);
                }
                if (profiling_enabled) {
                    hip_check(hipStreamSynchronize(g_own.stream), "prof dsa fi");
                    prof.dsa_fi_topk_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_df0).count();
                }

                const auto t_dm1 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (M <= 8) {
                    // Wave-8 Batched M<=8 FP16->FP32 GEMM MQA-64 sparse attention (1/6th launches; per-batch GEMM bit-identical!)
                    const int n_sp = M - n_dense;
                    const float alpha = 1.0f, beta = 0.0f;
                    launch_dsa_gather_kv_f16(
                        L.d_dsa_mla_kv, g_own.d_sel_idx + size_t(n_dense) * size_t(kStrideSel),
                        g_own.d_sel_cnts + n_dense, g_own.d_dsa_ksel_f16, n_sp, kStrideSel, g_own.stream);
                    launch_bf16_to_f16(g_own.d_tmp_c + size_t(n_dense) * size_t(kDsaNumHeads) * 576u,
                                       g_own.d_x_f16, n_sp * kDsaNumHeads * 576, g_own.stream);
                    rocblas_check(rocblas_gemm_strided_batched_ex(
                        g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                        kStrideSel, kDsaNumHeads, 576, &alpha,
                        g_own.d_dsa_ksel_f16, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                        g_own.d_x_f16,        rocblas_datatype_f16_r, 576, int64_t(kDsaNumHeads) * 576LL,
                        &beta,
                        g_own.d_y_f32, rocblas_datatype_f32_r, kStrideSel, int64_t(kDsaNumHeads) * int64_t(kStrideSel),
                        g_own.d_y_f32, rocblas_datatype_f32_r, kStrideSel, int64_t(kDsaNumHeads) * int64_t(kStrideSel),
                        n_sp, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa sparse decode QK batched");
                    launch_dsa_scores_softmax_sink_f16(
                        g_own.d_y_f32, D_own.d_attn_sink, g_own.d_sel_cnts + n_dense,
                        g_own.d_x_f16, n_sp, kDsaNumHeads, kStrideSel, 0.0625f, g_own.stream);
                    rocblas_check(rocblas_gemm_strided_batched_ex(
                        g_own.blas, rocblas_operation_none, rocblas_operation_none,
                        512, kDsaNumHeads, kStrideSel, &alpha,
                        g_own.d_dsa_ksel_f16, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                        g_own.d_x_f16,        rocblas_datatype_f16_r, kStrideSel, int64_t(kDsaNumHeads) * int64_t(kStrideSel),
                        &beta,
                        g_own.d_y_f32, rocblas_datatype_f32_r, 512, int64_t(kDsaNumHeads) * 512LL,
                        g_own.d_y_f32, rocblas_datatype_f32_r, 512, int64_t(kDsaNumHeads) * 512LL,
                        n_sp, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa sparse decode PV batched");
                    launch_f32_to_bf16(g_own.d_y_f32, g_own.d_tmp_a + size_t(n_dense) * size_t(kDsaNumHeads) * 512u,
                                       n_sp * kDsaNumHeads * 512, g_own.stream);
                } else {
                    // Hardware FP16->FP32 GEMM-accelerated sparse Top-2048+SWA MLA (constant batch_count=16 -> 100% chunk-invariant!)
                    constexpr int kSparseB = 16;
                    const float alpha = 1.0f, beta = 0.0f;
                    for (int m0 = n_dense; m0 < M; m0 += kSparseB) {
                        const int B = std::min(kSparseB, M - m0);
                        launch_dsa_gather_kv_f16(
                            L.d_dsa_mla_kv, g_own.d_sel_idx + size_t(m0) * size_t(kStrideSel),
                            g_own.d_sel_cnts + m0, g_own.d_dsa_ksel_f16, B, kStrideSel, g_own.stream);
                        launch_bf16_to_f16(g_own.d_tmp_c + size_t(m0) * size_t(kDsaNumHeads) * 576u,
                                           g_own.d_x_f16, B * kDsaNumHeads * 576, g_own.stream);
                        if (B < kSparseB) {
                            hip_check(hipMemsetAsync(g_own.d_dsa_ksel_f16 + size_t(B) * size_t(kStrideSel) * 576u, 0,
                                                     size_t(kSparseB - B) * size_t(kStrideSel) * 576u * sizeof(uint16_t), g_own.stream), "zero dsa ksel pad");
                            hip_check(hipMemsetAsync(g_own.d_x_f16 + size_t(B) * size_t(kDsaNumHeads) * 576u, 0,
                                                     size_t(kSparseB - B) * size_t(kDsaNumHeads) * 576u * sizeof(uint16_t), g_own.stream), "zero dsa qsel pad");
                        }
                        rocblas_check(rocblas_gemm_strided_batched_ex(
                            g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                            kStrideSel, kDsaNumHeads, 576, &alpha,
                            g_own.d_dsa_ksel_f16, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                            g_own.d_x_f16,        rocblas_datatype_f16_r, 576, int64_t(kDsaNumHeads) * 576LL,
                            &beta,
                            g_own.d_y_f32, rocblas_datatype_f32_r, kStrideSel, int64_t(kDsaNumHeads) * int64_t(kStrideSel),
                            g_own.d_y_f32, rocblas_datatype_f32_r, kStrideSel, int64_t(kDsaNumHeads) * int64_t(kStrideSel),
                            kSparseB, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa sparse QK gemm_ex");
                        launch_dsa_scores_softmax_sink_f16(
                            g_own.d_y_f32, D_own.d_attn_sink, g_own.d_sel_cnts + m0,
                            g_own.d_x_f16, B, kDsaNumHeads, kStrideSel, 0.0625f, g_own.stream);
                        if (B < kSparseB) {
                            hip_check(hipMemsetAsync(g_own.d_x_f16 + size_t(B) * size_t(kDsaNumHeads) * size_t(kStrideSel), 0,
                                                     size_t(kSparseB - B) * size_t(kDsaNumHeads) * size_t(kStrideSel) * sizeof(uint16_t), g_own.stream), "zero dsa psel pad");
                        }
                        rocblas_check(rocblas_gemm_strided_batched_ex(
                            g_own.blas, rocblas_operation_none, rocblas_operation_none,
                            512, kDsaNumHeads, kStrideSel, &alpha,
                            g_own.d_dsa_ksel_f16, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                            g_own.d_x_f16,        rocblas_datatype_f16_r, kStrideSel, int64_t(kDsaNumHeads) * int64_t(kStrideSel),
                            &beta,
                            g_own.d_y_f32, rocblas_datatype_f32_r, 512, int64_t(kDsaNumHeads) * 512LL,
                            g_own.d_y_f32, rocblas_datatype_f32_r, 512, int64_t(kDsaNumHeads) * 512LL,
                            kSparseB, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa sparse PV gemm_ex");
                        launch_f32_to_bf16(g_own.d_y_f32, g_own.d_tmp_a + size_t(m0) * size_t(kDsaNumHeads) * 512u,
                                           B * kDsaNumHeads * 512, g_own.stream);
                    }
                }
                if (profiling_enabled) {
                    hip_check(hipStreamSynchronize(g_own.stream), "prof dsa mla");
                    prof.dsa_mla_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dm1).count();
                }
            }

            // Phase D: 1D Contiguous Scatter o_lat [M, 64, 512] -> [M, h_loc, 512] to TP ranks (ZERO hipMemcpy2DAsync!)
            if (tp > 1) {
                const auto t_dc0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const size_t row_elems = size_t(h_loc) * 512u;
                const size_t slab_elems = size_t(M) * row_elems;
                const size_t slab_b = slab_elems * sizeof(uint16_t);
                hip_check(hipMemcpyAsync(h_tp_qabs_pin, g_own.d_tmp_a, size_t(tp) * slab_b,
                                         hipMemcpyDeviceToHost, g_own.stream), "dsa olat 1d d2h");
                hip_check(hipStreamSynchronize(g_own.stream), "dsa olat sync");
                #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (M >= 32)
                for (int m = 0; m < M; ++m) {
                    const uint16_t* src = h_tp_qabs_pin + size_t(m) * size_t(kDsaNumHeads) * 512u;
                    for (int r = 0; r < tp; ++r) {
                        std::memcpy(h_tp_oattn_pin + size_t(r) * slab_elems + size_t(m) * row_elems,
                                    src + size_t(r) * row_elems,
                                    row_elems * sizeof(uint16_t));
                    }
                }
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    hip_check(hipSetDevice(w.hip_dev), "dsa olat h2d dev");
                    hip_check(hipMemcpyAsync(w.d_tmp_a, h_tp_oattn_pin + size_t(r) * slab_elems,
                                             slab_b, hipMemcpyHostToDevice, w.stream), "dsa olat 1d h2d");
                }
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dc0).count();
                }
            }

            // Phase E: Parallel W_UV/4 -> 1D Rank-Major AllGather o_v -> 16x-Reuse VHA+LRGA -> o_proj/4 -> AllReduce
            const auto t_dp1 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dsa_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dsa E1 dev");

                if (M <= 8) {
                    launch_dsa_o_project_uv_f16(w.d_tmp_a, D.d_kv_b, w.d_tmp_b, M, h_loc, w.stream);
                } else {
                    const float alpha = 1.0f, beta = 0.0f;
                    for (int m0 = 0; m0 < M; m0 += kGemmStrip) {
                        const int cur = std::min(kGemmStrip, M - m0);
                        launch_bf16_to_f16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 512u, w.d_x_f16, cur * h_loc * 512, w.stream);
                        if (cur < kGemmStrip) {
                            hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(h_loc) * 512u, 0,
                                                     size_t(kGemmStrip - cur) * size_t(h_loc) * 512u * sizeof(uint16_t), w.stream), "zero uv pad");
                        }
                        rocblas_check(rocblas_gemm_strided_batched_ex(
                            w.blas, rocblas_operation_transpose, rocblas_operation_none,
                            256, kGemmStrip, 512, &alpha,
                            D.d_kv_b + 192u * 512u, rocblas_datatype_f16_r, 512, 448LL * 512LL,
                            w.d_x_f16,              rocblas_datatype_f16_r, h_loc * 512, 512LL,
                            &beta,
                            w.d_y_f32, rocblas_datatype_f32_r, h_loc * 256, 256LL,
                            w.d_y_f32, rocblas_datatype_f32_r, h_loc * 256, 256LL,
                            h_loc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "dsa o_proj gemm_ex batched");
                        launch_f32_to_bf16(w.d_y_f32, w.d_tmp_b + size_t(m0) * size_t(h_loc) * 256u, cur * h_loc * 256, w.stream);
                    }
                }
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp1).count();
            }
            if (tp > 1) {
                tp_allgather_heads_bf16(M, kDsaNumHeads, 256,
                                        &Mi50ExpertWorker::d_tmp_b, &Mi50ExpertWorker::d_tmp_a, h_tp_oattn_pin);
            }
            const auto t_dp2 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dsa_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dsa E2 dev");
                const uint16_t* d_ov_all = (tp > 1) ? w.d_tmp_a : w.d_tmp_b;
                uint16_t* d_vha_out      = (tp > 1) ? w.d_tmp_b : w.d_tmp_a;
                bf16_linear_bf16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                launch_dsa_vha_lrga_shard(d_ov_all, D.d_vha_m, w.d_tmp_d, d_vha_out, M, tp, r * h_loc, h_loc, w.stream);
                bf16_linear_bf16(w, d_vha_out, D.d_o_proj, w.d_sub_out, M, kHiddenSize, h_loc * 256);
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp2).count();
            }
            if (tp > 1) {
                tp_allreduce_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            }
            if (profiling_enabled) {
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    hip_check(hipSetDevice(w.hip_dev), "prof dsa dev");
                    hip_check(hipStreamSynchronize(w.stream), "prof dsa");
                }
                prof.dsa_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_a0).count();
            }
        }
    }

    // ---- Shared Routed-MoE All-Device Dispatch Helper (P150A FIRST + Strided 5x MI50 + Optional L1-Fused Sum+RMSNorm!) ----
    const float* dispatch_routed_moe_all_devices(int l, bool is_echo, Mi50ExpertWorker& g_own, int M,
                                                 const float* fused_rmsnorm_ln = nullptr, float fused_rmsnorm_eps = 0.0f) {
        const bool all_to_host = (fused_rmsnorm_ln != nullptr);
        const auto t_m0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        bool p150_has_work = false;
        uint32_t p150_seq = 0;

        // 1. Single-pass route bucketing for P150A + ALL 5 strided MI50 GPUs!
        const int n_gpus = int(gpus.size());
        int node_cnt[8] = {};
        for (int gi = 0; gi < n_gpus; ++gi) gpus[size_t(gi)]->active_exps.clear();

        auto* ch = (!is_echo && p150_ok) ? p150_shm.ptr : nullptr;
        const int l1_end = ch ? (p150_exp_begin + int(ch->l1_hot_experts)) : 0;

        for (int r = 0; r < M; ++r) {
            for (int s = 0; s < kTopKExperts; ++s) {
                const int32_t eid = h_topk_ids_pin[size_t(r) * kTopKExperts + size_t(s)];
                const float wt    = h_topk_wts_pin[size_t(r) * kTopKExperts + size_t(s)];

                int gi = -1, le = -1;
                if (!is_echo && wt != 0.0f && eid >= 0 && eid < kNumRoutedExperts) {
                    const int cand_g = int(bb_owner_gpu[size_t(eid)]);
                    const int cand_l = int(bb_local_slot[size_t(eid)]);
                    const bool valid_for_layer = (cand_g >= 0) &&
                        (l < mcfg.num_hidden_layers || cand_l < gpus[size_t(cand_g)]->bb_base_count);
                    if (valid_for_layer) {
                        gi = cand_g;
                        le = cand_l;
                    }
                } else if (is_echo && wt != 0.0f && eid >= 0 && eid < kNumRoutedExperts) {
                    for (int g = 0; g < n_gpus; ++g) {
                        if (eid >= gpus[size_t(g)]->echo_begin && eid < gpus[size_t(g)]->echo_end) {
                            gi = g; le = eid - gpus[size_t(g)]->echo_begin; break;
                        }
                    }
                }

                if (gi >= 0 && le >= 0) {
                    if (ch) {
                        ch->row_slots[r][s]   = -1;
                        ch->row_weights[r][s] = 0.0f;
                    }
                    auto& w = *gpus[size_t(gi)];
                    const int nd = node_cnt[gi]++;
                    if (w.b_cnt[size_t(le)] == 0) w.active_exps.push_back(le);
                    w.b_rows[size_t(nd)]  = r;
                    w.b_slots[size_t(nd)] = s;
                    w.b_wts[size_t(nd)]   = wt;
                    w.b_next[size_t(nd)]  = w.b_head[size_t(le)];
                    w.b_head[size_t(le)]  = nd;
                    ++w.b_cnt[size_t(le)];
                    ++prof.mi50_routes;
                } else if (ch) {
                    if (eid >= p150_exp_begin && eid < p150_exp_end && wt != 0.0f) {
                        ch->row_slots[r][s]   = eid - p150_exp_begin;
                        ch->row_weights[r][s] = wt;
                        p150_has_work = true;
                        ++prof.p150_routes;
                        if (eid < l1_end) ++prof.p150_l1_routes;
                    } else {
                        ch->row_slots[r][s]   = -1;
                        ch->row_weights[r][s] = 0.0f;
                    }
                }
            }
        }

        // 2. Ring Blackhole P150A doorbell FIRST (fast-skip on 0 hits!) so it overlaps 100% with MI50_MoE + shared_experts!
        if (p150_has_work && ch) {
            const size_t n_lat = size_t(M) * kLatentDim;
            avx2_f16_to_f32(h_lat_rot_f16_pin, ch->x_rot_f32, n_lat, kOmpThreads);
            ch->op        = ERNIE55_P150_OP_MOE_EXEC;
            ch->layer_idx = uint32_t(l);
            ch->rows      = uint32_t(M);
            ch->top_k     = kTopKExperts;
            p150_seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
            ch->req_seq.store(p150_seq, std::memory_order_release);
        }

        // 3. Launch ALL active strided MI50 expert workers 100% CONCURRENTLY via chunk-invariant WHT128-HLUT4 fdot2 GEMV!
        //    (Keeping HGEMM disabled in forward_chunk preserves 100% INV-1d cos=1.00000000 across M=137 vs M=512!)
        Mi50ExpertWorker* active_mi50s[8];
        int n_active_mi50 = 0;
        bool owner_did_moe = false;
        size_t max_grp_cnt = 0;
        constexpr int kHgemmThreshold = 1000000;

        for (int gi = 0; gi < n_gpus; ++gi) {
            auto& w = *gpus[size_t(gi)];
            w.h_groups.clear();
            w.hgemm_plans.clear();
            if (w.active_exps.empty()) continue;

            hip_check(hipSetDevice(w.hip_dev), "set worker dev");
            if (all_to_host || w.hip_dev != g_own.hip_dev) {
                hip_check(hipMemcpyAsync(w.d_x_rot_f16, h_lat_rot_f16_pin,
                                         size_t(M) * kLatentDim * sizeof(uint16_t), hipMemcpyHostToDevice, w.stream), "H2D worker x_rot");
            }
            hip_check(hipMemsetAsync(w.d_y_slots_f32, 0, size_t(M) * kTopKExperts * kLatentDim * sizeof(float), w.stream), "zero worker y_slots");

            int hgemm_off = 0;
            for (int le : w.active_exps) {
                const int cnt = w.b_cnt[size_t(le)];
                if (M >= 32 && cnt >= kHgemmThreshold && w.d_w_unpack_f16 != nullptr) {
                    ++prof.hgemm_experts;
                    const int off0 = hgemm_off;
                    for (int idx = w.b_head[size_t(le)]; idx >= 0; idx = w.b_next[size_t(idx)]) {
                        w.h_row_ids_pin[hgemm_off]   = w.b_rows[size_t(idx)];
                        w.h_row_slots_pin[hgemm_off] = w.b_slots[size_t(idx)];
                        w.h_row_wts_pin[hgemm_off]   = w.b_wts[size_t(idx)];
                        ++hgemm_off;
                    }
                    w.hgemm_plans.push_back({le, off0, hgemm_off - off0});
                } else {
                    ++prof.gemv_experts;
                    ExpertDispatchGroup grp{};
                    grp.expert = le;
                    grp.count  = 0;
                    for (int idx = w.b_head[size_t(le)]; idx >= 0; idx = w.b_next[size_t(idx)]) {
                        if (grp.count == kMaxGroupTokens) {
                            w.h_groups.push_back(grp);
                            grp = ExpertDispatchGroup{};
                            grp.expert = le;
                            grp.count  = 0;
                        }
                        grp.rows[grp.count]    = w.b_rows[size_t(idx)];
                        grp.slots[grp.count]   = w.b_slots[size_t(idx)];
                        grp.weights[grp.count] = w.b_wts[size_t(idx)];
                        ++grp.count;
                    }
                    if (grp.count > 0) w.h_groups.push_back(grp);
                }
                w.b_head[size_t(le)] = -1;
                w.b_cnt[size_t(le)]  = 0;
            }
            if (hgemm_off > 0) {
                hip_check(hipMemcpyAsync(w.d_row_ids,   w.h_row_ids_pin,   size_t(hgemm_off) * sizeof(int32_t), hipMemcpyHostToDevice, w.stream), "H2D hg rids");
                hip_check(hipMemcpyAsync(w.d_row_slots, w.h_row_slots_pin, size_t(hgemm_off) * sizeof(int32_t), hipMemcpyHostToDevice, w.stream), "H2D hg rslots");
                hip_check(hipMemcpyAsync(w.d_row_wts,   w.h_row_wts_pin,   size_t(hgemm_off) * sizeof(float),   hipMemcpyHostToDevice, w.stream), "H2D hg rwts");
            }
            if (w.h_groups.size() > max_grp_cnt) max_grp_cnt = w.h_groups.size();
        }

        // Launch HGEMM experts (M_e >= 16) concurrently across ALL 5 MI50s first!
        {
            const float alpha = 1.0f, beta = 0.0f;
            for (int gi = 0; gi < n_gpus; ++gi) {
                auto& w = *gpus[size_t(gi)];
                if (w.hgemm_plans.empty()) continue;
                hip_check(hipSetDevice(w.hip_dev), "set worker hgemm dev");
                for (const auto& pl : w.hgemm_plans) {
                    const int C = pl.cnt;
                    const size_t e_off = size_t(pl.le) * kExpProjTiles;
                    launch_gather_rows_f16(w.d_x_rot_f16, w.d_row_ids + pl.off, w.d_xg_f16, C, kLatentDim, w.stream);
                    launch_unpack_hlut4_to_f16(w.d_w1[size_t(l)] + e_off, w.d_levels8_f16, w.d_w_unpack_f16, kMoeInterSize, kLatentDim, w.stream);
                    rocblas_check(rocblas_gemm_ex(
                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                        kMoeInterSize, C, kLatentDim, &alpha,
                        w.d_w_unpack_f16, rocblas_datatype_f16_r, kLatentDim,
                        w.d_xg_f16,       rocblas_datatype_f16_r, kLatentDim, &beta,
                        w.d_g_f32,        rocblas_datatype_f32_r, kMoeInterSize,
                        w.d_g_f32,        rocblas_datatype_f32_r, kMoeInterSize,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "moe hgemm w1");
                    launch_unpack_hlut4_to_f16(w.d_w3[size_t(l)] + e_off, w.d_levels8_f16, w.d_w_unpack_f16, kMoeInterSize, kLatentDim, w.stream);
                    rocblas_check(rocblas_gemm_ex(
                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                        kMoeInterSize, C, kLatentDim, &alpha,
                        w.d_w_unpack_f16, rocblas_datatype_f16_r, kLatentDim,
                        w.d_xg_f16,       rocblas_datatype_f16_r, kLatentDim, &beta,
                        w.d_u_f32,        rocblas_datatype_f32_r, kMoeInterSize,
                        w.d_u_f32,        rocblas_datatype_f32_r, kMoeInterSize,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "moe hgemm w3");
                    launch_situ_glu_f32(w.d_g_f32, w.d_u_f32, w.d_g_f32, C, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, w.stream);
                    launch_wht128_f32_to_f16(w.d_g_f32, w.d_signs_inter[size_t(l)], w.d_xg_f16, C, kMoeInterSize, w.stream);
                    launch_unpack_hlut4_to_f16(w.d_w2[size_t(l)] + e_off, w.d_levels8_f16, w.d_w_unpack_f16, kLatentDim, kMoeInterSize, w.stream);
                    rocblas_check(rocblas_gemm_ex(
                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                        kLatentDim, C, kMoeInterSize, &alpha,
                        w.d_w_unpack_f16, rocblas_datatype_f16_r, kMoeInterSize,
                        w.d_xg_f16,       rocblas_datatype_f16_r, kMoeInterSize, &beta,
                        w.d_yh_f32,       rocblas_datatype_f32_r, kLatentDim,
                        w.d_yh_f32,       rocblas_datatype_f32_r, kLatentDim,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "moe hgemm w2");
                    launch_scatter_slot_weighted_f32(w.d_yh_f32, w.d_row_ids + pl.off, w.d_row_slots + pl.off,
                                                     w.d_row_wts + pl.off, w.d_y_slots_f32, C, kLatentDim, w.stream);
                }
            }
        }

        // Launch remaining GEMV chunk g0 across ALL 5 GPUs simultaneously before any multi-chunk sync!
        for (size_t g0 = 0; g0 < max_grp_cnt; g0 += size_t(max_groups)) {
            for (int gi = 0; gi < n_gpus; ++gi) {
                auto& w = *gpus[size_t(gi)];
                if (g0 >= w.h_groups.size()) continue;
                hip_check(hipSetDevice(w.hip_dev), "set worker grp dev");
                if (g0 > 0) hip_check(hipStreamSynchronize(w.stream), "sync groups chunk");
                const int ng = int(std::min<size_t>(size_t(max_groups), w.h_groups.size() - g0));
                int max_c = 1;
                for (int i = 0; i < ng; ++i) {
                    const int c = w.h_groups[g0 + size_t(i)].count;
                    if (c > max_c) max_c = c;
                }
                std::memcpy(w.h_groups_pin, w.h_groups.data() + g0, size_t(ng) * sizeof(ExpertDispatchGroup));
                hip_check(hipMemcpyAsync(w.d_groups, w.h_groups_pin, size_t(ng) * sizeof(ExpertDispatchGroup),
                                         hipMemcpyHostToDevice, w.stream), "H2D groups");
                launch_hlut4_fdot2_grouped_gu_situ(
                    w.d_x_rot_f16, w.d_groups, ng, w.d_w1[size_t(l)], w.d_w3[size_t(l)],
                    w.d_levels8_f16, w.d_z_f32, kMoeInterSize, kLatentDim,
                    mcfg.situ_beta, mcfg.situ_linear_beta, w.stream);
                launch_wht128_grouped_f32_to_f16(w.d_z_f32, w.d_groups, ng, w.d_signs_inter[size_t(l)],
                                                 w.d_z_rot_f16, kMoeInterSize, w.stream, max_c);
                launch_hlut4_fdot2_grouped_down_slot(
                    w.d_z_rot_f16, w.d_groups, ng, w.d_w2[size_t(l)],
                    w.d_levels8_f16, w.d_y_slots_f32, kLatentDim, kMoeInterSize, w.stream);
            }
        }

        for (int gi = 0; gi < n_gpus; ++gi) {
            auto& w = *gpus[size_t(gi)];
            if (w.h_groups.empty() && w.hgemm_plans.empty()) continue;
            hip_check(hipSetDevice(w.hip_dev), "set worker fin dev");
            launch_reduce_moe_slots_f32(w.d_y_slots_f32, w.d_y_acc_f32, M, kLatentDim, w.stream);
            if (!all_to_host && w.hip_dev == g_own.hip_dev) {
                owner_did_moe = true;
            } else {
                hip_check(hipMemcpyAsync(w.h_y_partial_pin, w.d_y_acc_f32, size_t(M) * kLatentDim * sizeof(float),
                                         hipMemcpyDeviceToHost, w.stream), "D2H worker y");
                active_mi50s[n_active_mi50++] = &w;
            }
        }

        // 4a. Fast L1-Fused TP Path: sync all MI50s + P150A, then ONE single-pass canonical Sum + RMSNorm -> 4 BF16 slabs!
        if (all_to_host) {
            const float* srcs[9];
            int n_srcs = 0;
            for (int wi = 0; wi < n_active_mi50; ++wi) {
                auto* w = active_mi50s[wi];
                hip_check(hipSetDevice(w->hip_dev), "sync worker dev");
                hip_check(hipStreamSynchronize(w->stream), "sync worker");
                srcs[n_srcs++] = w->h_y_partial_pin;
            }
            if (profiling_enabled) {
                prof.mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
            }
            if (p150_has_work && ch) {
                const auto t_p0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                while (ch->resp_seq.load(std::memory_order_acquire) != p150_seq) p150_cpu_relax();
                if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
                if (profiling_enabled) {
                    prof.p150_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
                }
                srcs[n_srcs++] = ch->y_out_f32;
            }
            const int lat_loc = kLatentDim / tp_size;
            #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (M >= 8)
            for (int m = 0; m < M; ++m) {
                alignas(32) float y[kLatentDim];
                const size_t off = size_t(m) * kLatentDim;
                if (n_srcs == 0) {
                    std::memset(y, 0, sizeof(y));
                } else {
                    const float* s0 = srcs[0] + off;
                    for (int c = 0; c < kLatentDim; ++c) y[c] = s0[c];
                    for (int si = 1; si < n_srcs; ++si) {
                        const float* sk = srcs[si] + off;
                        for (int c = 0; c < kLatentDim; ++c) y[c] += sk[c];
                    }
                }
                float ss = 0.0f;
                for (int c = 0; c < kLatentDim; ++c) ss += y[c] * y[c];
                const float inv = 1.0f / std::sqrt(ss * (1.0f / float(kLatentDim)) + fused_rmsnorm_eps);
                for (int r = 0; r < tp_size; ++r) {
                    uint16_t* dst = h_tp_fc2_in_pin[r] + size_t(m) * size_t(lat_loc);
                    const float* ys = y + size_t(r) * size_t(lat_loc);
                    const float* ws = fused_rmsnorm_ln + size_t(r) * size_t(lat_loc);
                    for (int c = 0; c < lat_loc; ++c) {
                        dst[c] = f32_to_bf16_rne(ys[c] * inv * ws[c]);
                    }
                }
            }
            return nullptr;
        }

        // 4b. Owner-GPU Canonical Path (Backbone Hybrid MoE + 3L DSpark L43..L45 & ECHO L43)
        const size_t n_lat = size_t(M) * kLatentDim;
        bool has_remote_sum = false;
        for (int wi = 0; wi < n_active_mi50; ++wi) {
            auto* w = active_mi50s[wi];
            hip_check(hipSetDevice(w->hip_dev), "sync worker dev");
            hip_check(hipStreamSynchronize(w->stream), "sync worker");
            if (!has_remote_sum) {
                std::memcpy(h_y_sum_f32_pin, w->h_y_partial_pin, n_lat * sizeof(float));
                has_remote_sum = true;
            } else {
                avx2_add_f32_inplace(h_y_sum_f32_pin, w->h_y_partial_pin, n_lat, kOmpThreads);
            }
        }
        if (profiling_enabled) {
            if (owner_did_moe) {
                hip_check(hipSetDevice(g_own.hip_dev), "prof owner moe dev");
                hip_check(hipStreamSynchronize(g_own.stream), "prof owner moe");
            }
            prof.mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
        }
        if (p150_has_work && ch) {
            const auto t_p0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            while (ch->resp_seq.load(std::memory_order_acquire) != p150_seq) p150_cpu_relax();
            if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
            if (profiling_enabled) {
                prof.p150_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
            }
            if (!has_remote_sum) {
                std::memcpy(h_y_sum_f32_pin, ch->y_out_f32, n_lat * sizeof(float));
                has_remote_sum = true;
            } else {
                avx2_add_f32_inplace(h_y_sum_f32_pin, ch->y_out_f32, n_lat, kOmpThreads);
            }
        }

        hip_check(hipSetDevice(g_own.hip_dev), "restore spine dev");
        if (owner_did_moe && has_remote_sum) {
            hip_check(hipMemcpyAsync(g_own.d_f32_tmp0, h_y_sum_f32_pin, n_lat * sizeof(float), hipMemcpyHostToDevice, g_own.stream), "H2D y_sum");
            launch_f32_add_inplace(g_own.d_y_acc_f32, g_own.d_f32_tmp0, int(n_lat), g_own.stream);
            return g_own.d_y_acc_f32;
        } else if (owner_did_moe) {
            return g_own.d_y_acc_f32;
        } else if (has_remote_sum) {
            hip_check(hipMemcpyAsync(g_own.d_f32_tmp0, h_y_sum_f32_pin, n_lat * sizeof(float), hipMemcpyHostToDevice, g_own.stream), "H2D y_sum");
            return g_own.d_f32_tmp0;
        } else {
            hip_check(hipMemsetAsync(g_own.d_f32_tmp0, 0, n_lat * sizeof(float), g_own.stream), "zero y_sum");
            return g_own.d_f32_tmp0;
        }
    }

    // ---- Owner-GPU MoE Sublayer (Backbone Hybrid L01..L42 + 3L DSpark L43..L45 & ECHO L43!) ----
    void run_moe_core(int l, const MoeSpineGpuWeights& moe, bool is_echo, Mi50ExpertWorker& g, int M, bool draft_mi50_only = false) {
        const auto t_r0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        hip_check(hipSetDevice(g.hip_dev), "moe single dev");

        // Run FP16 fdot2 router + fc1 + WHT128 FIRST at 100% CU occupancy so P150A doorbell fires asap,
        // THEN launch shared_experts on g.stream_sh so it overlaps 100% with MI50_MoE + P150A!
        bf16_linear_f32(g, g.d_x_in, moe.d_router_w01_f16, g.d_f32_tmp1, M, 1024, kHiddenSize);
        const int max_exp = (draft_mi50_only && p150_ok) ? p150_exp_begin : kNumRoutedExperts;
        launch_split_sigmoid_top10_route(g.d_f32_tmp1, moe.d_corr_bias, moe.d_scale_param,
                                         g.d_topk_ids, g.d_topk_wts, M, g.stream, max_exp);
        bf16_linear_bf16(g, g.d_x_in, moe.d_fc1_lat, g.d_h_lat, M, kLatentDim, kHiddenSize);
        launch_wht128_bf16_to_f16(g.d_h_lat, g.d_signs_in[size_t(l)], g.d_x_rot_f16, M, kLatentDim, g.stream);
        if (g.stream_sh) {
            hip_check(hipEventRecord(g.ev_sh_start, g.stream), "sh start single");
        }

        hip_check(hipMemcpyAsync(h_topk_ids_pin, g.d_topk_ids, size_t(M) * kTopKExperts * sizeof(int32_t), hipMemcpyDeviceToHost, g.stream), "D2H topk_ids");
        hip_check(hipMemcpyAsync(h_topk_wts_pin, g.d_topk_wts, size_t(M) * kTopKExperts * sizeof(float),   hipMemcpyDeviceToHost, g.stream), "D2H topk_wts");
        hip_check(hipMemcpyAsync(h_lat_rot_f16_pin, g.d_x_rot_f16, size_t(M) * kLatentDim * sizeof(uint16_t), hipMemcpyDeviceToHost, g.stream), "D2H lat_rot");
        if (g.stream_sh) {
            hip_check(hipStreamWaitEvent(g.stream_sh, g.ev_sh_start, 0), "sh wait single");
            f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w1, g.d_sh_tmp_a, M, kMoeInterSize, kHiddenSize, 512);
            f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w3, g.d_sh_tmp_b, M, kMoeInterSize, kHiddenSize, 512);
            launch_situ_glu_bf16(g.d_sh_tmp_a, g.d_sh_tmp_b, g.d_sh_tmp_a, M, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream_sh);
            f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_sh_tmp_a, moe.d_sh_w2, g.d_sh_out, M, kHiddenSize, kMoeInterSize, 512);
            hip_check(hipEventRecord(g.ev_sh_done, g.stream_sh), "sh done single");
        }
        hip_check(hipStreamSynchronize(g.stream), "sync router+lat");
        if (router_capture_dst && l >= 0 && l < mcfg.num_hidden_layers && M <= router_capture_stride) {
            const size_t base = size_t(l) * size_t(router_capture_stride) * kTopKExperts;
            if (base + size_t(M) * kTopKExperts <= router_capture_dst->size()) {
                for (int r = 0; r < M; ++r) {
                    for (int s = 0; s < kTopKExperts; ++s) {
                        const int32_t eid = h_topk_ids_pin[size_t(r) * kTopKExperts + size_t(s)];
                        const int16_t raw = (has_router_perm && eid >= 0 && eid < kNumRoutedExperts)
                            ? int16_t(slot_to_raw[size_t(l)][size_t(eid)])
                            : int16_t(eid);
                        (*router_capture_dst)[base + size_t(r) * kTopKExperts + size_t(s)] = raw;
                    }
                }
            }
        }
        if (profiling_enabled) {
            prof.router_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_r0).count();
        }

        const float* d_y_final = dispatch_routed_moe_all_devices(l, is_echo, g, M);
        launch_rmsnorm_f32_to_bf16(d_y_final, moe.d_lat_norm, g.d_h_lat, M, kLatentDim, mcfg.rms_norm_eps, g.stream);
        bf16_linear_bf16(g, g.d_h_lat, moe.d_fc2_lat, g.d_sub_out, M, kHiddenSize, kLatentDim);
        if (g.stream_sh) {
            hip_check(hipStreamWaitEvent(g.stream, g.ev_sh_done, 0), "wait sh single");
            launch_add_bf16_inplace(g.d_sub_out, g.d_sh_out, M * kHiddenSize, g.stream);
        } else {
            bf16_linear_bf16(g, g.d_x_in, moe.d_sh_w1, g.d_tmp_a, M, kMoeInterSize, kHiddenSize);
            bf16_linear_bf16(g, g.d_x_in, moe.d_sh_w3, g.d_tmp_b, M, kMoeInterSize, kHiddenSize);
            launch_situ_glu_bf16(g.d_tmp_a, g.d_tmp_b, g.d_tmp_a, M, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream);
            bf16_linear_bf16(g, g.d_tmp_a, moe.d_sh_w2, g.d_tmp_d, M, kHiddenSize, kMoeInterSize);
            launch_add_bf16_inplace(g.d_sub_out, g.d_tmp_d, M * kHiddenSize, g.stream);
        }
    }

    // ---- TP=4 Dense (L00) / Hybrid Owner-GPU Latent MoE Spine (L01..L42) + Single-Source Broadcast! ----
    void run_moe_tp(LayerSpineGpu& L, int M) {
        const int tp = L.tp_size;
        const auto dev_for_r = [&](int r) -> int { return L.is_echo ? L.owner_dev : tp_devs[r]; };

        if (L.is_dense) {
            const int inter_loc = mcfg.intermediate_size / tp;
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dense_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dense tp dev");
                bf16_linear_bf16(w, w.d_x_in, D.d_gate, w.d_tmp_a, M, inter_loc, kHiddenSize);
                bf16_linear_bf16(w, w.d_x_in, D.d_up,   w.d_tmp_b, M, inter_loc, kHiddenSize);
                launch_situ_glu_bf16(w.d_tmp_a, w.d_tmp_b, w.d_tmp_a, M, inter_loc,
                                     mcfg.situ_beta, mcfg.situ_linear_beta, w.stream);
                bf16_linear_bf16(w, w.d_tmp_a, D.d_down, w.d_sub_out, M, kHiddenSize, inter_loc);
            }
            if (tp > 1) {
                tp_allreduce_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            }
            return;
        }

        auto& g_own = *dev_map[L.owner_dev];
        run_moe_core(L.layer_idx, L.moe, L.is_echo, g_own, M);

        // Single-source 1D contiguous SDMA broadcast of owner's d_sub_out [M, 4096] -> other (tp - 1) ranks!
        if (tp > 1) {
            const auto t_mb0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const size_t n = size_t(M) * kHiddenSize;
            const size_t nb = n * sizeof(uint16_t);
            hip_check(hipSetDevice(g_own.hip_dev), "moe bcast d2h dev");
            hip_check(hipMemcpyAsync(h_tp_ar_out_pin, g_own.d_sub_out, nb, hipMemcpyDeviceToHost, g_own.stream), "moe bcast d2h");
            hip_check(hipStreamSynchronize(g_own.stream), "moe bcast sync");
            for (int r = 0; r < tp; ++r) {
                if (r == L.owner_tp_rank) continue;
                auto& w = *dev_map[tp_devs[r]];
                hip_check(hipSetDevice(w.hip_dev), "moe bcast h2d dev");
                hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_ar_out_pin, nb, hipMemcpyHostToDevice, w.stream), "moe bcast h2d");
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mb0).count();
            }
        }
    }

    // ---- Run one transformer layer across TP ranks (or single GPU for ECHO) ----
    void run_layer_tp(LayerSpineGpu& L, int M, int32_t pos0, bool is_first_layer_broadcast) {
        const int tp = L.tp_size;
        const auto dev_for_r = [&](int r) -> int { return L.is_echo ? L.owner_dev : tp_devs[r]; };

        const auto t_mc0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (int r = 0; r < tp; ++r) {
            auto& w = *dev_map[dev_for_r(r)];
            hip_check(hipSetDevice(w.hip_dev), "layer mhc1 dev");
            if (is_first_layer_broadcast) {
                launch_mhc_pre_broadcast_norm_f16(w.d_x_in, L.attn_mhc_tp[r].d_fn_24x16384_f16, L.attn_mhc_tp[r].d_scale3,
                                                  L.attn_mhc_tp[r].d_base24, L.d_in_norm_tp[r], w.d_res_a,
                                                  w.d_post_mix, w.d_comb_mix, w.d_x_in, M, kHiddenSize,
                                                  mcfg.rms_norm_eps, mcfg.hc_eps, w.stream);
            } else {
                run_mhc_pre_norm_fast(w, w.d_res_a, L.attn_mhc_tp[r], L.d_in_norm_tp[r], M);
            }
        }
        if (profiling_enabled) {
            prof_sync_tp(tp, L.is_echo, L.owner_dev);
            prof.mhc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mc0).count();
        }

        run_attention_tp(L, M, pos0);

        const auto t_mc1 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (int r = 0; r < tp; ++r) {
            auto& w = *dev_map[dev_for_r(r)];
            hip_check(hipSetDevice(w.hip_dev), "layer mhc2 dev");
            launch_mhc_post_update(w.d_sub_out, w.d_res_a, w.d_post_mix, w.d_comb_mix, w.d_res_b, M, kHiddenSize, w.stream);
            run_mhc_pre_norm_fast(w, w.d_res_b, L.mlp_mhc_tp[r], L.d_post_norm_tp[r], M);
        }
        if (profiling_enabled) {
            prof_sync_tp(tp, L.is_echo, L.owner_dev);
            prof.mhc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mc1).count();
        }

        run_moe_tp(L, M);

        const auto t_mc2 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (int r = 0; r < tp; ++r) {
            auto& w = *dev_map[dev_for_r(r)];
            hip_check(hipSetDevice(w.hip_dev), "layer mhc3 dev");
            launch_mhc_post_update(w.d_sub_out, w.d_res_b, w.d_post_mix, w.d_comb_mix, w.d_res_a, M, kHiddenSize, w.stream);
        }
        if (profiling_enabled) {
            prof_sync_tp(tp, L.is_echo, L.owner_dev);
            prof.mhc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mc2).count();
        }
    }

    // ---- Forward M tokens through 43 backbone layers -> logits [M, vocab_size] after MultiMax ----
    void forward_backbone_chunk(const int32_t* tok_ids, int M, int32_t pos0,
                                std::vector<float>* out_logits_MxV,
                                bool only_last_logits,
                                bool capture_last_res_for_echo,
                                bool capture_dspark_features)
    {
        auto& g0 = *dev_map[spine_dev0];
        auto& g1 = *dev_map[spine_dev1];

        for (int i = 0; i < M; ++i) {
            h_tok_pin[i] = tok_ids[i];
            h_pos_pin[i] = pos0 + i;
        }

        hip_check(hipSetDevice(g0.hip_dev), "fwd dev0");
        hip_check(hipMemcpyAsync(g0.d_tok_ids, h_tok_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, g0.stream), "H2D tok_ids");
        hip_check(hipMemcpyAsync(g0.d_pos_ids, h_pos_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, g0.stream), "H2D pos0");
        launch_embed_lookup_bf16(d_embed_bf16, g0.d_tok_ids, g0.d_x_in, M, kHiddenSize, g0.stream);

        // Broadcast initial embedding [M, 4096] + pos_ids to all TP ranks once (ZERO mid-network residual hop!)
        if (tp_size > 1) {
            const size_t emb_b = size_t(M) * kHiddenSize * sizeof(uint16_t);
            hip_check(hipMemcpyAsync(h_tp_ar_out_pin, g0.d_x_in, emb_b, hipMemcpyDeviceToHost, g0.stream), "D2H bcast emb");
            hip_check(hipStreamSynchronize(g0.stream), "sync bcast emb");
            for (int r = 1; r < tp_size; ++r) {
                auto& w = *dev_map[tp_devs[r]];
                hip_check(hipSetDevice(w.hip_dev), "H2D bcast dev");
                hip_check(hipMemcpyAsync(w.d_x_in, h_tp_ar_out_pin, emb_b, hipMemcpyHostToDevice, w.stream), "H2D bcast emb");
                hip_check(hipMemcpyAsync(w.d_pos_ids, h_pos_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, w.stream), "H2D bcast pos");
            }
        }

        const size_t n_ds_t = mcfg.dspark_target_layer_ids.size();
        for (int l = 0; l < mcfg.num_hidden_layers; ++l) {
            if (tp_size == 1 && l == split_layer && spine_dev1 != spine_dev0) {
                const size_t hop_bytes = size_t(M) * kHcDim * sizeof(uint16_t);
                hip_check(hipSetDevice(g0.hip_dev), "hop d2h dev");
                hip_check(hipMemcpyAsync(h_hop_bf16_pin, g0.d_res_a, hop_bytes, hipMemcpyDeviceToHost, g0.stream), "D2H hop");
                hip_check(hipStreamSynchronize(g0.stream), "sync hop");
                hip_check(hipSetDevice(g1.hip_dev), "fwd dev1");
                hip_check(hipMemcpyAsync(g1.d_res_a, h_hop_bf16_pin, hop_bytes, hipMemcpyHostToDevice, g1.stream), "H2D hop");
                hip_check(hipMemcpyAsync(g1.d_pos_ids, h_pos_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, g1.stream), "H2D pos1");
            }

            run_layer_tp(layers[size_t(l)], M, pos0, l == 0);

            if (capture_dspark_features && has_dspark && d_dspark_aux_cat) {
                for (size_t fi = 0; fi < n_ds_t; ++fi) {
                    if (mcfg.dspark_target_layer_ids[fi] == l) {
                        hip_check(hipSetDevice(g1.hip_dev), "dspark aux dev1");
                        launch_mhc_mean4_bf16(g1.d_res_a, g1.d_sub_out, M, kHiddenSize, g1.stream);
                        hip_check(hipMemcpy2DAsync(
                            d_dspark_aux_cat + fi * kHiddenSize,
                            n_ds_t * kHiddenSize * sizeof(uint16_t),
                            g1.d_sub_out,
                            kHiddenSize * sizeof(uint16_t),
                            kHiddenSize * sizeof(uint16_t),
                            size_t(M),
                            hipMemcpyDeviceToDevice, g1.stream), "D2D dspark aux");
                    }
                }
            }
        }

        hip_check(hipSetDevice(g1.hip_dev), "head dev1");
        if (capture_last_res_for_echo && has_echo_l43) {
            hip_check(hipMemcpyAsync(h_echo_res_pin, g1.d_res_a,
                                     size_t(M) * kHcDim * sizeof(uint16_t), hipMemcpyDeviceToHost, g1.stream), "D2H echo res");
        }

        const auto t_hd0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (out_logits_MxV != nullptr) {
            const int n_out = only_last_logits ? 1 : M;
            const uint16_t* src_res = only_last_logits ? (g1.d_res_a + size_t(M - 1) * kHcDim) : g1.d_res_a;
            out_logits_MxV->resize(size_t(n_out) * size_t(mcfg.vocab_size));
            constexpr int kLogitSlice = 64;
            for (int row = 0; row < n_out; row += kLogitSlice) {
                const int cur = std::min(kLogitSlice, n_out - row);
                launch_mhc_head_contract(src_res + size_t(row) * kHcDim, d_hc_head_fn, d_hc_head_scale, d_hc_head_base,
                                         d_final_norm, g1.d_x_in, cur, kHiddenSize,
                                         mcfg.rms_norm_eps, mcfg.hc_eps, g1.stream);
                compute_lm_head_tp5(g1.d_x_in, cur, true,
                                    out_logits_MxV->data() + size_t(row) * size_t(mcfg.vocab_size), nullptr);
            }
        }
        hip_check(hipSetDevice(g1.hip_dev), "sync dev1 set");
        hip_check(hipStreamSynchronize(g1.stream), "sync dev1");
        if (capture_last_res_for_echo && has_echo_l43) {
            h_last_backbone_res_4xH.assign(h_echo_res_pin + size_t(M - 1) * kHcDim,
                                           h_echo_res_pin + size_t(M) * kHcDim);
        }
        if (profiling_enabled) {
            prof.head_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hd0).count();
            ++prof.calls;
        }
    }

    // ---- ECHO (L43) 1-Step Draft Forward (called 2x for 2-step greedy MTP draft!) ----
    int32_t run_echo_draft_step(int32_t tok_id, int32_t pos, std::vector<uint16_t>& inout_res_4xH) {
        auto& g0 = *dev_map[spine_dev0];
        auto& g1 = *dev_map[spine_dev1];
        auto& L43 = layers[size_t(mcfg.num_hidden_layers)];

        // Fetch embedding for tok_id from dev0 -> dev1 via pinned staging!
        hip_check(hipSetDevice(g0.hip_dev), "echo emb dev0");
        hip_check(hipMemcpy(h_echo_emb_pin, d_embed_bf16 + size_t(tok_id) * kHiddenSize, kHiddenSize * sizeof(uint16_t), hipMemcpyDeviceToHost), "D2H echo emb");
        h_echo_pos_pin[0] = pos;

        hip_check(hipSetDevice(g1.hip_dev), "echo dev1");
        hip_check(hipMemcpyAsync(g1.d_res_b, inout_res_4xH.data(), kHcDim * sizeof(uint16_t), hipMemcpyHostToDevice, g1.stream), "H2D echo prev_res");
        hip_check(hipMemcpyAsync(g1.d_x_in, h_echo_emb_pin, kHiddenSize * sizeof(uint16_t), hipMemcpyHostToDevice, g1.stream), "H2D echo emb");
        hip_check(hipMemcpyAsync(g1.d_pos_ids, h_echo_pos_pin, sizeof(int32_t), hipMemcpyHostToDevice, g1.stream), "H2D echo pos");

        // h_proj(hnorm(prev_res[4, 4096])) + e_proj(enorm(emb[4096])).unsqueeze(-2)
        launch_rmsnorm_bf16(g1.d_res_b, L43.d_echo_hnorm, g1.d_tmp_a, kHcMult, kHiddenSize, mcfg.rms_norm_eps, g1.stream);
        bf16_linear_bf16(g1, g1.d_tmp_a, L43.d_echo_hproj, g1.d_tmp_b, kHcMult, kHiddenSize, kHiddenSize);

        launch_rmsnorm_bf16(g1.d_x_in, L43.d_echo_enorm, g1.d_sub_out, 1, kHiddenSize, mcfg.rms_norm_eps, g1.stream);
        bf16_linear_bf16(g1, g1.d_sub_out, L43.d_echo_eproj, g1.d_x_in, 1, kHiddenSize, kHiddenSize);

        launch_echo_fuse_he(g1.d_tmp_b, g1.d_x_in, g1.d_res_a, 1, kHiddenSize, g1.stream);

        // Run L43 decoder block (mHC -> mqa_dsa -> mHC -> 512 WHT128-HLUT4 experts -> mhc_post)
        run_layer_tp(L43, 1, pos, false);

        // Stash updated 4-stream residual for next ECHO spec step (spec_step_index = 1!)
        hip_check(hipMemcpyAsync(inout_res_4xH.data(), g1.d_res_a, kHcDim * sizeof(uint16_t), hipMemcpyDeviceToHost, g1.stream), "D2H echo next_res");

        // Contract via L43.hc_head -> L43.shared_head.norm -> TP5 shared lm_head -> MultiMax -> greedy argmax
        launch_mhc_head_contract(g1.d_res_a, L43.d_echo_hc_fn, L43.d_echo_hc_scale, L43.d_echo_hc_base,
                                 L43.d_echo_head_norm, g1.d_x_in, 1, kHiddenSize,
                                 mcfg.rms_norm_eps, mcfg.hc_eps, g1.stream);
        compute_lm_head_tp5(g1.d_x_in, 1, true, nullptr, g1.d_logits_f32);
        int32_t out_tok = 0;
        launch_argmax_f32_fast(g1.d_logits_f32, g1.d_argmax_v64, g1.d_argmax_i64, g1.d_tok_ids, mcfg.vocab_size, g1.stream);
        hip_check(hipMemcpyAsync(&out_tok, g1.d_tok_ids, sizeof(int32_t), hipMemcpyDeviceToHost, g1.stream), "D2H echo tok");
        hip_check(hipStreamSynchronize(g1.stream), "sync echo step");
        return out_tok;
    }

    // ---- Mirror Replicated HCA KV from owner rank -> other TP ranks (used after load_kvblk_ex!) ----
    void mirror_hca_kv_from_owner(int cur_seq_len, bool include_snaps) {
        if (tp_size <= 1) return;
        const int n_comp = cur_seq_len / kHcaCompressRatio;
        const size_t comp_b = size_t(n_comp) * 512u * sizeof(uint16_t);
        const size_t swa_b  = 256u * 512u * sizeof(uint16_t);
        const size_t st_b   = 128u * 512u * sizeof(float);
        std::vector<uint8_t> buf(std::max<size_t>(comp_b, std::max<size_t>(swa_b, st_b)));
        auto bcast_buf = [&](int src_dev, const void* src_ptr, auto dst_arr, size_t nb) {
            if (nb == 0 || !src_ptr) return;
            hip_check(hipSetDevice(src_dev), "mirror hca d2h dev");
            hip_check(hipMemcpy(buf.data(), src_ptr, nb, hipMemcpyDeviceToHost), "mirror hca d2h");
            for (int r = 0; r < tp_size; ++r) {
                const int d = tp_devs[r];
                if (d == src_dev || !dst_arr[r]) continue;
                hip_check(hipSetDevice(d), "mirror hca h2d dev");
                hip_check(hipMemcpy(dst_arr[r], buf.data(), nb, hipMemcpyHostToDevice), "mirror hca h2d");
            }
        };
        for (int l = 0; l < mcfg.num_hidden_layers; ++l) {
            auto& L = layers[size_t(l)];
            if (L.is_dsa || L.tp_size <= 1) continue;
            bcast_buf(L.owner_dev, L.d_hca_comp_kv,  L.d_hca_comp_kv_tp,  comp_b);
            bcast_buf(L.owner_dev, L.d_hca_swa_ring, L.d_hca_swa_ring_tp, swa_b);
            bcast_buf(L.owner_dev, L.d_hca_state_kv, L.d_hca_state_kv_tp, st_b);
            bcast_buf(L.owner_dev, L.d_hca_state_sc, L.d_hca_state_sc_tp, st_b);
            if (include_snaps) {
                bcast_buf(L.owner_dev, L.d_hca_snap_swa, L.d_hca_snap_swa_tp, swa_b);
                bcast_buf(L.owner_dev, L.d_hca_snap_kv,  L.d_hca_snap_kv_tp,  st_b);
                bcast_buf(L.owner_dev, L.d_hca_snap_sc,  L.d_hca_snap_sc_tp,  st_b);
            }
        }
    }

    // ---- Lightweight GPU D2D Prompt-Boundary Snapshot (replicated across all TP ranks!) ----
    void capture_hca_prompt_snapshot(const std::vector<int32_t>& prompt_toks) {
        for (int l = 0; l < mcfg.num_hidden_layers; ++l) {
            auto& L = layers[size_t(l)];
            if (L.is_dsa) continue;
            for (int r = 0; r < L.tp_size; ++r) {
                if (L.d_hca_swa_ring_tp[r] && L.d_hca_snap_swa_tp[r]) {
                    const int dev = tp_devs[r];
                    auto& g = *dev_map[dev];
                    hip_check(hipSetDevice(dev), "snap cap dev");
                    hip_check(hipMemcpyAsync(L.d_hca_snap_swa_tp[r], L.d_hca_swa_ring_tp[r], 256u * 512u * sizeof(uint16_t), hipMemcpyDeviceToDevice, g.stream), "cap swa");
                    hip_check(hipMemcpyAsync(L.d_hca_snap_kv_tp[r],  L.d_hca_state_kv_tp[r], 128u * 512u * sizeof(float),    hipMemcpyDeviceToDevice, g.stream), "cap kv");
                    hip_check(hipMemcpyAsync(L.d_hca_snap_sc_tp[r],  L.d_hca_state_sc_tp[r], 128u * 512u * sizeof(float),    hipMemcpyDeviceToDevice, g.stream), "cap sc");
                }
            }
        }
        if (has_dspark) {
            auto& g1 = *dev_map[spine_dev1];
            hip_check(hipSetDevice(spine_dev1), "snap cap ds");
            for (auto& DL : dspark_layers) {
                if (DL.d_swa_ring && DL.d_swa_snap) {
                    hip_check(hipMemcpyAsync(DL.d_swa_snap, DL.d_swa_ring, size_t(kDsparkSwaRingSize) * 512u * sizeof(uint16_t), hipMemcpyDeviceToDevice, g1.stream), "cap ds swa");
                }
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& g = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(g.hip_dev), "snap cap sync dev");
            hip_check(hipStreamSynchronize(g.stream), "snap cap sync");
        }
        ckpt_prompt_tokens = prompt_toks;
        ckpt_prompt_len    = int(prompt_toks.size());
    }

    void restore_hca_prompt_snapshot() {
        for (int l = 0; l < mcfg.num_hidden_layers; ++l) {
            auto& L = layers[size_t(l)];
            if (L.is_dsa) continue;
            for (int r = 0; r < L.tp_size; ++r) {
                if (L.d_hca_swa_ring_tp[r] && L.d_hca_snap_swa_tp[r]) {
                    const int dev = tp_devs[r];
                    auto& g = *dev_map[dev];
                    hip_check(hipSetDevice(dev), "snap rst dev");
                    hip_check(hipMemcpyAsync(L.d_hca_swa_ring_tp[r], L.d_hca_snap_swa_tp[r], 256u * 512u * sizeof(uint16_t), hipMemcpyDeviceToDevice, g.stream), "rst swa");
                    hip_check(hipMemcpyAsync(L.d_hca_state_kv_tp[r], L.d_hca_snap_kv_tp[r],  128u * 512u * sizeof(float),    hipMemcpyDeviceToDevice, g.stream), "rst kv");
                    hip_check(hipMemcpyAsync(L.d_hca_state_sc_tp[r], L.d_hca_snap_sc_tp[r],  128u * 512u * sizeof(float),    hipMemcpyDeviceToDevice, g.stream), "rst sc");
                }
            }
        }
        if (has_dspark) {
            auto& g1 = *dev_map[spine_dev1];
            hip_check(hipSetDevice(spine_dev1), "snap rst ds");
            for (auto& DL : dspark_layers) {
                if (DL.d_swa_ring && DL.d_swa_snap) {
                    hip_check(hipMemcpyAsync(DL.d_swa_ring, DL.d_swa_snap, size_t(kDsparkSwaRingSize) * 512u * sizeof(uint16_t), hipMemcpyDeviceToDevice, g1.stream), "rst ds swa");
                }
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& g = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(g.hip_dev), "snap rst sync dev");
            hip_check(hipStreamSynchronize(g.stream), "snap rst sync");
        }
        seq_len       = ckpt_prompt_len;
        cached_tokens = ckpt_prompt_tokens;
    }
};

// ---- Public Ernie55Runtime Methods ----
Ernie55Runtime::Ernie55Runtime() : impl_(std::make_unique<Impl>()) {}
Ernie55Runtime::~Ernie55Runtime() { shutdown(); }

bool Ernie55Runtime::initialized() const noexcept { return impl_ && impl_->is_init; }
const Ernie55ModelConfig& Ernie55Runtime::model_config() const noexcept { return impl_->mcfg; }
const Ernie55TopologyPlan& Ernie55Runtime::topology() const noexcept { return impl_->topo; }
const Ernie55Tokenizer& Ernie55Runtime::tokenizer() const noexcept { return impl_->tok; }
DraftMethod Ernie55Runtime::active_draft_method() const noexcept { return impl_->draft_method; }
int Ernie55Runtime::max_kv() const noexcept { return impl_->opt.max_kv; }
int Ernie55Runtime::current_seq_len() const noexcept { return impl_->seq_len; }
bool Ernie55Runtime::p150_attached() const noexcept { return impl_->p150_ok; }

void Ernie55Runtime::init(const Ernie55RuntimeOptions& opt) {
    shutdown();
    omp_set_num_threads(Impl::kOmpThreads);
    impl_->opt = opt;
    impl_->use_flashindexer  = opt.enable_flashindexer;
    impl_->dspark_block_size = std::max(1, std::min(7, opt.dspark_block_size));
    impl_->dspark_conf_tau   = opt.dspark_conf_tau;
    impl_->max_chunk = std::min(1024, std::max(16, opt.prefill_chunk));

    // Allocate pinned host staging buffers once
    impl_->h_lat_rot_f16_pin = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kLatentDim, impl_->pin_allocs, true);
    impl_->h_topk_ids_pin    = host_pin_alloc<int32_t>(size_t(impl_->max_chunk) * kTopKExperts, impl_->pin_allocs, true);
    impl_->h_topk_wts_pin    = host_pin_alloc<float>(size_t(impl_->max_chunk) * kTopKExperts, impl_->pin_allocs, true);
    impl_->h_y_sum_f32_pin   = host_pin_alloc<float>(size_t(impl_->max_chunk) * kLatentDim, impl_->pin_allocs, true);
    impl_->h_hop_bf16_pin    = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHcDim, impl_->pin_allocs, true);
    impl_->h_echo_res_pin    = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHcDim, impl_->pin_allocs, true);
    impl_->h_sel_idx_batch     = host_pin_alloc<int32_t>(size_t(impl_->max_chunk) * size_t(kDsaIndexTopK + kSlidingWindow), impl_->pin_allocs, false);
    impl_->h_sel_cnts_batch    = host_pin_alloc<int32_t>(size_t(impl_->max_chunk), impl_->pin_allocs, true);
    impl_->h_pos_pin           = host_pin_alloc<int32_t>(size_t(impl_->max_chunk), impl_->pin_allocs, true);
    impl_->h_tok_pin           = host_pin_alloc<int32_t>(size_t(impl_->max_chunk), impl_->pin_allocs, true);
    impl_->h_dspark_pos_pin    = host_pin_alloc<int32_t>(size_t(impl_->max_chunk), impl_->pin_allocs, true);
    impl_->h_dspark_anchor_pin = host_pin_alloc<uint16_t>(kHiddenSize, impl_->pin_allocs, true);
    impl_->h_echo_pos_pin      = host_pin_alloc<int32_t>(1, impl_->pin_allocs, true);
    impl_->h_echo_emb_pin      = host_pin_alloc<uint16_t>(kHiddenSize, impl_->pin_allocs, true);
    for (int r = 0; r < LayerSpineGpu::kMaxTp; ++r) {
        impl_->h_tp_ar_in_pin[r] = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHiddenSize, impl_->pin_allocs, true);
    }
    impl_->h_tp_ar_out_pin  = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHiddenSize, impl_->pin_allocs, true);
    impl_->h_tp_oattn_pin   = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 64u * 512u, impl_->pin_allocs, true);
    impl_->h_tp_qabs_pin    = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 64u * 576u, impl_->pin_allocs, true);
    impl_->h_tp_qabs_rm_pin = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 64u * 576u, impl_->pin_allocs, true);
    for (int r = 0; r < LayerSpineGpu::kMaxTp; ++r) {
        impl_->h_tp_rtr_in_pin[r] = host_pin_alloc<float>(size_t(impl_->max_chunk) * 512u, impl_->pin_allocs, true);
        impl_->h_tp_fc1_in_pin[r] = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 1024u, impl_->pin_allocs, true);
        impl_->h_tp_fc2_in_pin[r] = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 1024u, impl_->pin_allocs, true);
    }
    impl_->h_tp_mapped_slabs = host_mapped_alloc<uint16_t>(4u * 4u * size_t(Impl::kMappedSlabStride), impl_->pin_allocs, true);
    impl_->h_tp_mapped_ready = host_mapped_alloc<uint32_t>(4u * 16u, impl_->pin_allocs, true);
    impl_->h_tp_mapped_done  = host_mapped_alloc<uint32_t>(4u * 16u, impl_->pin_allocs, true);
    impl_->tp_mapped_epoch   = 0u;
    for (int s = 0; s < 2; ++s) {
        for (int b = 0; b < 2; ++b) {
            impl_->h_kv_pin[s][b] = host_pin_alloc<uint8_t>(size_t(Impl::kKvSlabBytes), impl_->pin_allocs, false);
        }
    }

    // Optional router-frequency permutation (weight_root/router_perm.bin -> hottest 88-93% on MI50 HBM2!)
    impl_->has_router_perm = false;
    {
        const std::string pp = opt.weight_root + "/router_perm.bin";
        if (std::filesystem::is_regular_file(pp)) {
            std::ifstream pfs(pp, std::ios::binary);
            uint32_t magic = 0, nl = 0;
            pfs.read(reinterpret_cast<char*>(&magic), 4);
            pfs.read(reinterpret_cast<char*>(&nl), 4);
            if (magic == 0x50524D35u && nl == uint32_t(kTotalLayersWithEcho)) {
                pfs.read(reinterpret_cast<char*>(impl_->slot_to_raw.data()),
                         std::streamsize(sizeof(impl_->slot_to_raw)));
                if (pfs.good()) {
                    impl_->has_router_perm = true;
                    std::printf("[ERNIE55-RT] Loaded router-frequency permutation %s (hot->MI50 HBM2, warm->P150A L1, cold->GDDR6)\n", pp.c_str());
                }
            }
        }
    }

    // 1. Load config.json, topology_*.json, tokenizer.json + tokenizer_config.json
    impl_->mcfg = Ernie55ModelConfig::load_json(opt.weight_root + "/config.json");
    impl_->mcfg.validate_compat();
    impl_->topo = Ernie55TopologyPlan::load_json(opt.topology_path);
    impl_->tok.load(opt.weight_root + "/tokenizer.json", opt.weight_root + "/tokenizer_config.json");

    if (impl_->topo.mi50_tp_ranks.empty()) throw std::runtime_error("Topology has no MI50 spine ranks");
    impl_->spine_dev0 = impl_->topo.mi50_tp_ranks[0].hip_dev;
    impl_->spine_dev1 = (impl_->topo.mi50_tp_ranks.size() > 1) ? impl_->topo.mi50_tp_ranks[1].hip_dev : impl_->spine_dev0;
    impl_->split_layer = 22;  // L00..L21 (3 DSA) on dev0, L22..L43 (3 DSA + DSpark) on dev1

    // 2. Collect all MI50 GPUs & enable TP=4 spine across dev0..dev3 with balanced expert split (70,74,92,32,50 = 318!)
    struct DevPlan { int hip_dev, bb_b, bb_e, echo_b, echo_e; };
    std::vector<DevPlan> plans;
    for (const auto& r : impl_->topo.mi50_tp_ranks) {
        plans.push_back({r.hip_dev, r.expert_begin, r.expert_end, 0, 0});
    }
    for (const auto& e : impl_->topo.mi50_ep_workers) {
        plans.push_back({e.hip_dev, e.expert_begin, e.expert_end, e.echo_begin, e.echo_end});
    }

    impl_->tp_size = (plans.size() >= 4) ? 4 : 1;
    for (int r = 0; r < impl_->tp_size; ++r) impl_->tp_devs[r] = plans[size_t(r)].hip_dev;

    // On 5-MI50 318-expert SB-M5-P2-192G topology, balance experts for Hybrid TP=4 Attn + Owner-GPU MoE Spine + 1024-tok prefill!
    if (impl_->tp_size == 4 && plans.size() == 5 && plans.front().bb_b == 0 && plans.back().bb_e == 318) {
        constexpr int kCounts[5] = {66, 68, 92, 38, 54};
        int cur = 0;
        for (int i = 0; i < 5; ++i) {
            plans[size_t(i)].bb_b = cur;
            cur += kCounts[i];
            plans[size_t(i)].bb_e = cur;
        }
    }

    for (size_t gi = 0; gi < plans.size(); ++gi) {
        const auto& dp = plans[gi];
        const bool is_spine = (int(gi) < impl_->tp_size);
        const bool is_dsa_owner = (dp.hip_dev == impl_->spine_dev0 || dp.hip_dev == impl_->spine_dev1);
        impl_->init_gpu_worker(dp.hip_dev, dp.bb_b, dp.bb_e, dp.echo_b, dp.echo_e, is_spine, is_dsa_owner);
    }

    // Build weighted round-robin (Bresenham/D'Hondt) strided hot-expert assignment across all MI50s!
    // Balances router_perm hot experts [0..318) evenly across all 5 MI50s without touching a single byte on disk!
    impl_->bb_owner_gpu.fill(int8_t(-1));
    impl_->bb_local_slot.fill(int16_t(-1));
    if (impl_->has_router_perm && plans.size() > 1 && plans.front().bb_b == 0) {
        const int n_g = int(plans.size());
        std::vector<int> cap(static_cast<size_t>(n_g), 0);
        std::vector<int> used(static_cast<size_t>(n_g), 0);
        int tot_mi50 = 0;
        for (int g = 0; g < n_g; ++g) {
            cap[size_t(g)] = plans[size_t(g)].bb_e - plans[size_t(g)].bb_b;
            tot_mi50 += cap[size_t(g)];
            impl_->gpus[size_t(g)]->bb_global_ids.clear();
            impl_->gpus[size_t(g)]->bb_global_ids.reserve(size_t(cap[size_t(g)]));
        }
        for (int eid = 0; eid < tot_mi50; ++eid) {
            // Pure equal round-robin (min used[g]) -> top 170 hottest experts split EXACTLY 20.0% across all 5 MI50s!
            int best_g = -1;
            int min_used = 1000000;
            for (int g = 0; g < n_g; ++g) {
                if (used[size_t(g)] >= cap[size_t(g)]) continue;
                if (used[size_t(g)] < min_used) { min_used = used[size_t(g)]; best_g = g; }
            }
            const int lslot = used[size_t(best_g)]++;
            impl_->gpus[size_t(best_g)]->bb_global_ids.push_back(eid);
            impl_->bb_owner_gpu[size_t(eid)]  = int8_t(best_g);
            impl_->bb_local_slot[size_t(eid)] = int16_t(lslot);
        }
    } else {
        for (size_t gi = 0; gi < plans.size(); ++gi) {
            auto& w = *impl_->gpus[gi];
            w.bb_global_ids.clear();
            for (int eid = w.bb_begin; eid < w.bb_end; ++eid) {
                const int lslot = int(w.bb_global_ids.size());
                w.bb_global_ids.push_back(eid);
                impl_->bb_owner_gpu[size_t(eid)]  = int8_t(gi);
                impl_->bb_local_slot[size_t(eid)] = int16_t(lslot);
            }
        }
    }
    for (auto& w : impl_->gpus) w->bb_base_count = int(w->bb_global_ids.size());
    std::printf("[ERNIE55-RT] TP=%d FP16-resident spine on MI50 devs [%d,%d,%d,%d] + %zu-GPU Strided-Hot EP MoE (OMP=%d)\n",
                impl_->tp_size, impl_->tp_devs[0], impl_->tp_devs[1], impl_->tp_devs[2], impl_->tp_devs[3],
                plans.size(), Impl::kOmpThreads);

    // 3. Attach to Blackhole P150A worker SHM if enabled
    if (opt.use_p150 && !impl_->topo.p150_groups.empty()) {
        const auto& pg = impl_->topo.p150_groups[0];
        const std::string shm = (opt.shm_name == "/ernie55_p150_ipc" && !pg.shm_name.empty()) ? pg.shm_name : opt.shm_name;
        impl_->p150_exp_begin = pg.expert_begin;
        impl_->p150_exp_end   = pg.expert_end;
        if (!impl_->p150_shm.attach_client(shm)) {
            throw std::runtime_error("Cannot attach to Blackhole P150A SHM " + shm + " (start bin/p150_ernie55_worker first!)");
        }
        impl_->p150_ok = true;
        std::printf("[ERNIE55-RT] Attached to Blackhole P150A SHM %s (mode=%s, cards=%u, experts=[%u..%u), L1=%.1f MiB)\n",
                    shm.c_str(), impl_->p150_shm.ptr->p150_mode == ERNIE55_P150_MODE_TP2 ? "TP2" : "EP",
                    impl_->p150_shm.ptr->num_p150_cards, impl_->p150_shm.ptr->expert_id_begin,
                    impl_->p150_shm.ptr->expert_id_end, double(impl_->p150_shm.ptr->l1_pinned_bytes_total) / 1048576.0);

        // Wave-8 Shadow-Cache top 13 hottest P150A GDDR6 experts (right past P150A L1 bank!) in free MI50 HBM2
        // for backbone layers L01..L42 ONLY (leaves >= 377 MiB free/GPU, 0 changes to running P150A worker!)
        if (impl_->has_router_perm && impl_->gpus.size() == 5 && impl_->p150_exp_begin == 318) {
            const int gddr_start = impl_->p150_exp_begin + int(impl_->p150_shm.ptr->l1_hot_experts);
            constexpr int kExtraPerGpu[5] = {3, 3, 6, 1, 0}; // 13 hottest GDDR6 experts -> ~272.5 MiB/exp on backbone!
            int eid = gddr_start;
            int added = 0;
            for (int pass = 0; pass < 6 && eid < impl_->p150_exp_end; ++pass) {
                for (int gi = 0; gi < 5 && eid < impl_->p150_exp_end; ++gi) {
                    auto& w = *impl_->gpus[size_t(gi)];
                    if (int(w.bb_global_ids.size()) - w.bb_base_count >= kExtraPerGpu[gi]) continue;
                    const int lslot = int(w.bb_global_ids.size());
                    w.bb_global_ids.push_back(eid);
                    impl_->bb_owner_gpu[size_t(eid)]  = int8_t(gi);
                    impl_->bb_local_slot[size_t(eid)] = int16_t(lslot);
                    ++eid;
                    ++added;
                }
            }
            if (added > 0) {
                std::printf("[ERNIE55-RT] Wave-8 MI50 HBM2 shadow-cached %d hottest P150A GDDR6 experts [%d..%d) on L01..L42 (0 B DDR4 spill)\n",
                            added, gddr_start, gddr_start + added);
            }
        }
    }

    // 4. Load lossless BF16 spine + 100% uniform WHT128-HLUT4 experts across all layers
    const auto t0 = std::chrono::steady_clock::now();
    impl_->load_spine_shared();
    for (int l = 0; l < impl_->mcfg.num_hidden_layers; ++l) {
        impl_->load_spine_layer(l);
        if (l >= impl_->mcfg.first_k_dense_replace) impl_->load_hlut4_layer_experts(l);
    }
    if (opt.load_echo && impl_->mcfg.num_nextn_predict_layers > 0) {
        const int echo_l = impl_->mcfg.num_hidden_layers;
        char ef[512];
        std::snprintf(ef, sizeof(ef), "%s/packed_spine/spine_echo_%02d.safetensors", opt.weight_root.c_str(), echo_l);
        if (std::filesystem::is_regular_file(ef)) {
            impl_->load_spine_layer(echo_l);
            impl_->load_hlut4_layer_experts(echo_l);
        }
    }
    impl_->maybe_load_bundled_dspark();

    // 5. Select active draft method
    if (opt.draft_mode == "none") impl_->draft_method = DraftMethod::kNone;
    else if (opt.draft_mode == "dspark" && impl_->has_dspark) impl_->draft_method = DraftMethod::kDSpark;
    else if (opt.draft_mode == "echo" && impl_->has_echo_l43) impl_->draft_method = DraftMethod::kEcho;
    else {
        if (impl_->has_dspark) impl_->draft_method = DraftMethod::kDSpark;
        else if (impl_->has_echo_l43) impl_->draft_method = DraftMethod::kEcho;
        else impl_->draft_method = DraftMethod::kNone;
    }

    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const char* dm = (impl_->draft_method == DraftMethod::kDSpark) ? "DSpark(probabilistic)"
                   : (impl_->draft_method == DraftMethod::kEcho ? "ECHO(2-step greedy)" : "none");
    std::printf("[ERNIE55-RT] READY in %.1fs | topo=%s | %zu MI50s + %s | max_kv=%d | draft=%s | 0 B DDR4 spill\n",
                load_s, impl_->topo.name.c_str(), impl_->gpus.size(),
                impl_->p150_ok ? "Blackhole P150A" : "no-P150", opt.max_kv, dm);
    std::fflush(stdout);
    impl_->is_init = true;
}

void Ernie55Runtime::shutdown() noexcept {
    if (!impl_) return;
    if (impl_->p150_ok && impl_->p150_shm.ptr && !impl_->opt.keep_p150) {
        auto* ch = impl_->p150_shm.ptr;
        ch->op = ERNIE55_P150_OP_SHUTDOWN;
        const uint32_t seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
        ch->req_seq.store(seq, std::memory_order_release);
        for (int i = 0; i < 500000 && ch->resp_seq.load(std::memory_order_acquire) != seq; ++i) p150_cpu_relax();
    }
    impl_->p150_shm.close_mapping();
    impl_->p150_ok = false;

    for (auto& w : impl_->gpus) {
        if (!w) continue;
        (void)hipSetDevice(w->hip_dev);
        if (w->stream)    (void)hipStreamSynchronize(w->stream);
        if (w->stream_sh) (void)hipStreamSynchronize(w->stream_sh);
        if (w->blas)      (void)rocblas_destroy_handle(w->blas);
        if (w->blas_sh)   (void)rocblas_destroy_handle(w->blas_sh);
        if (w->ev_sh_start) (void)hipEventDestroy(w->ev_sh_start);
        if (w->ev_sh_done)  (void)hipEventDestroy(w->ev_sh_done);
        for (void* p : w->allocs) if (p) (void)hipFree(p);
        w->allocs.clear();
        if (w->stream_sh) (void)hipStreamDestroy(w->stream_sh);
        if (w->stream)    (void)hipStreamDestroy(w->stream);
        (void)hipDeviceSynchronize();
    }
    impl_->gpus.clear();
    impl_->dev_map.clear();
    for (void* p : impl_->pin_allocs) if (p) (void)hipHostFree(p);
    impl_->pin_allocs.clear();
    impl_->is_init = false;
}

void Ernie55Runtime::set_echo_draft_steps(int steps) noexcept {
    if (impl_) impl_->opt.echo_draft_steps = std::max(0, std::min(4, steps));
}

void Ernie55Runtime::set_dspark_block_size(int K) noexcept {
    if (impl_) {
        impl_->dspark_block_size = std::max(0, std::min(7, K));
        impl_->dspark_heads.block_size = impl_->dspark_block_size;
    }
}

void Ernie55Runtime::set_dspark_conf_tau(float tau) noexcept {
    if (impl_) {
        impl_->dspark_conf_tau = std::max(0.0f, std::min(0.99f, tau));
        impl_->dspark_heads.conf_stop_tau = impl_->dspark_conf_tau;
    }
}

void Ernie55Runtime::set_flashindexer(bool enabled) noexcept {
    if (impl_) impl_->use_flashindexer = enabled;
}

void Ernie55Runtime::set_draft_mode(DraftMethod mode) noexcept {
    if (impl_) impl_->draft_method = mode;
}

void Ernie55Runtime::set_prefill_progress_callback(PrefillProgressCallback cb) {
    if (impl_) impl_->prefill_progress_cb = std::move(cb);
}

void Ernie55Runtime::set_profiling(bool enabled) noexcept {
    if (impl_) impl_->profiling_enabled = enabled;
}

void Ernie55Runtime::reset_profile_stats() noexcept {
    if (impl_) impl_->prof = Ernie55ProfileStats{};
}

Ernie55ProfileStats Ernie55Runtime::profile_stats() const noexcept {
    return impl_ ? impl_->prof : Ernie55ProfileStats{};
}

void Ernie55Runtime::set_router_capture(std::vector<int16_t>* out_raw_top10_43xMx10, int stride_tokens) noexcept {
    if (impl_) {
        impl_->router_capture_dst = out_raw_top10_43xMx10;
        impl_->router_capture_stride = stride_tokens;
    }
}

void Ernie55Runtime::reset_kv() {
    impl_->seq_len = 0;
    impl_->echo_seq_len = 0;
    impl_->cached_tokens.clear();
    impl_->ckpt_prompt_tokens.clear();
    impl_->ckpt_prompt_len = 0;
    for (int l = 0; l < kTotalLayersWithEcho; ++l) {
        auto& L = impl_->layers[size_t(l)];
        if (L.is_dsa) continue;
        for (int r = 0; r < L.tp_size; ++r) {
            if (L.d_hca_swa_ring_tp[r]) {
                const int dev = L.is_echo ? L.owner_dev : impl_->tp_devs[r];
                hip_check(hipSetDevice(dev), "reset_kv dev");
                hip_check(hipMemset(L.d_hca_swa_ring_tp[r], 0, 256u * 512u * sizeof(uint16_t)), "zero swa");
                hip_check(hipMemset(L.d_hca_state_kv_tp[r], 0, 128u * 512u * sizeof(float)), "zero state_kv");
                hip_check(hipMemset(L.d_hca_state_sc_tp[r], 0, 128u * 512u * sizeof(float)), "zero state_sc");
            }
        }
    }
    if (impl_->has_dspark) {
        hip_check(hipSetDevice(impl_->spine_dev1), "reset_kv dspark");
        for (auto& DL : impl_->dspark_layers) {
            if (DL.d_swa_ring) {
                hip_check(hipMemset(DL.d_swa_ring, 0, size_t(kDsparkSwaRingSize) * 512u * sizeof(uint16_t)), "zero ds swa");
            }
        }
    }
}

void Ernie55Runtime::wipe_all_kv_vram() {
    reset_kv();
    const size_t max_blocks = size_t((impl_->opt.max_kv + 127) / 128);
    for (int l = 0; l < kTotalLayersWithEcho; ++l) {
        auto& L = impl_->layers[size_t(l)];
        if (!L.is_dsa) {
            for (int r = 0; r < L.tp_size; ++r) {
                if (L.d_hca_comp_kv_tp[r]) {
                    const int dev = L.is_echo ? L.owner_dev : impl_->tp_devs[r];
                    hip_check(hipSetDevice(dev), "wipe hca dev");
                    hip_check(hipMemset(L.d_hca_comp_kv_tp[r],  0, max_blocks * 512u * sizeof(uint16_t)), "wipe comp_kv");
                    if (L.d_hca_snap_swa_tp[r]) hip_check(hipMemset(L.d_hca_snap_swa_tp[r], 0, 256u * 512u * sizeof(uint16_t)), "wipe snap_swa");
                    if (L.d_hca_snap_kv_tp[r])  hip_check(hipMemset(L.d_hca_snap_kv_tp[r],  0, 128u * 512u * sizeof(float)), "wipe snap_kv");
                    if (L.d_hca_snap_sc_tp[r])  hip_check(hipMemset(L.d_hca_snap_sc_tp[r],  0, 128u * 512u * sizeof(float)), "wipe snap_sc");
                }
            }
        }
        if (L.d_dsa_mla_kv) {
            hip_check(hipSetDevice(L.owner_dev), "wipe dsa dev");
            hip_check(hipMemset(L.d_dsa_mla_kv,   0, size_t(impl_->opt.max_kv) * 576u * sizeof(uint16_t)), "wipe mla_kv");
            hip_check(hipMemset(L.d_dsa_ik_cache, 0, size_t(impl_->opt.max_kv) * 128u * sizeof(uint16_t)), "wipe ik_cache");
        }
    }
    if (impl_->has_dspark) {
        hip_check(hipSetDevice(impl_->spine_dev1), "wipe dspark dev");
        for (auto& DL : impl_->dspark_layers) {
            if (DL.d_swa_snap) hip_check(hipMemset(DL.d_swa_snap, 0, size_t(kDsparkSwaRingSize) * 512u * sizeof(uint16_t)), "wipe ds snap");
        }
    }
    for (int r = 0; r < impl_->tp_size; ++r) {
        hip_check(hipSetDevice(impl_->tp_devs[r]), "wipe sync dev");
        hip_check(hipDeviceSynchronize(), "wipe sync");
    }
}

void Ernie55Runtime::capture_prompt_snapshot() {
    if (impl_->seq_len > 0 && int(impl_->cached_tokens.size()) == impl_->seq_len) {
        impl_->capture_hca_prompt_snapshot(impl_->cached_tokens);
    }
}

void Ernie55Runtime::prefill_tokens_chunked(const std::vector<int32_t>& tok_ids, int chunk_size,
                                            std::vector<float>& out_last_logits) {
    if (tok_ids.empty()) return;
    const int n = int(tok_ids.size());
    const int csz = std::max(1, std::min(impl_->max_chunk, chunk_size));
    const int dspark_tail_start = std::max(0, n - kDsparkSwaRingSize);
    for (int off = 0; off < n; ) {
        int m = std::min(csz, n - off);
        const int rem_after = n - (off + m);
        if (rem_after > 0 && rem_after <= 8 && m >= 24) {
            m -= 16; // Keep final chunk >= 17 tokens so INV-1d stays on identical SGEMM math!
        }
        const bool is_last = (off + m == n);
        const bool need_ds = impl_->has_dspark && ((off + m) > dspark_tail_start);
        const int32_t pos0 = impl_->seq_len;
        impl_->forward_backbone_chunk(
            tok_ids.data() + off, m, pos0,
            is_last ? &out_last_logits : nullptr,
            true, is_last, need_ds);
        if (need_ds) impl_->commit_dspark_context_kv(m, pos0);
        impl_->seq_len += m;
        impl_->cached_tokens.insert(impl_->cached_tokens.end(), tok_ids.begin() + off, tok_ids.begin() + off + m);
        off += m;
    }
    impl_->total_prefill_tokens += uint64_t(n);
}

void Ernie55Runtime::prefill_tokens(const std::vector<int32_t>& tok_ids, std::vector<float>& out_last_logits) {
    prefill_tokens_chunked(tok_ids, impl_->max_chunk, out_last_logits);
}

void Ernie55Runtime::prefill_tokens_warm(const std::vector<int32_t>& tok_ids, std::vector<float>& out_last_logits,
                                         int* out_reused_prefix) {
    if (out_reused_prefix) *out_reused_prefix = 0;
    if (tok_ids.empty()) return;

    auto compute_lcp = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) -> int {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return int(i);
    };

    const int target_n = int(tok_ids.size());
    int reused = 0;
    if (impl_->seq_len > 0 && int(impl_->cached_tokens.size()) == impl_->seq_len &&
        impl_->seq_len < target_n && compute_lcp(impl_->cached_tokens, tok_ids) >= impl_->seq_len) {
        // Exact full-end prefix hit!
        reused = impl_->seq_len;
    } else if (impl_->ckpt_prompt_len > 0 && int(impl_->ckpt_prompt_tokens.size()) == impl_->ckpt_prompt_len &&
               impl_->ckpt_prompt_len < target_n &&
               compute_lcp(impl_->ckpt_prompt_tokens, tok_ids) >= impl_->ckpt_prompt_len) {
        // Exact prompt-boundary snapshot hit (e.g., harness stripped prior <ernie:think> block!):
        // Restores exact HCA + DSpark rings in ~0.04 ms D2D without re-prefilling context!
        impl_->restore_hca_prompt_snapshot();
        reused = impl_->seq_len;
    } else {
        reset_kv();
        reused = 0;
    }

    if (out_reused_prefix) *out_reused_prefix = reused;
    const size_t tot_n = tok_ids.size();
    if (impl_->prefill_progress_cb) {
        impl_->prefill_progress_cb(tot_n, size_t(reused), 0u, 0.0);
    }
    const auto run_chunked_cb = [&](const int32_t* ptr, int n) {
        if (n <= 0) return;
        const int csz = std::max(1, impl_->max_chunk);
        const int dspark_tail_start = std::max(0, n - kDsparkSwaRingSize);
        for (int off = 0; off < n; ) {
            int m = std::min(csz, n - off);
            const int rem_after = n - (off + m);
            if (rem_after > 0 && rem_after <= 8 && m >= 24) m -= 16;
            const bool is_last = (off + m == n);
            const bool need_ds = impl_->has_dspark && ((off + m) > dspark_tail_start);
            const int32_t pos0 = impl_->seq_len;
            const auto tc0 = std::chrono::steady_clock::now();
            impl_->forward_backbone_chunk(
                ptr + off, m, pos0,
                is_last ? &out_last_logits : nullptr,
                true, is_last, need_ds);
            if (need_ds) impl_->commit_dspark_context_kv(m, pos0);
            impl_->seq_len += m;
            impl_->cached_tokens.insert(impl_->cached_tokens.end(), ptr + off, ptr + off + m);
            off += m;
            if (impl_->prefill_progress_cb) {
                const double cms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
                impl_->prefill_progress_cb(tot_n, size_t(reused), size_t(off), cms);
            }
        }
        impl_->total_prefill_tokens += uint64_t(n);
    };
    if (reused == 0) {
        run_chunked_cb(tok_ids.data(), int(tok_ids.size()));
    } else {
        run_chunked_cb(tok_ids.data() + reused, int(tok_ids.size()) - reused);
    }
}

std::vector<uint8_t> Ernie55Runtime::dump_kv_raw_bytes() const {
    const int n_comp = impl_->seq_len / kHcaCompressRatio;
    uint64_t total_bytes = 0;
    for (int l = 0; l < impl_->mcfg.num_hidden_layers; ++l) {
        const auto& L = impl_->layers[size_t(l)];
        if (!L.is_dsa) {
            total_bytes += uint64_t(n_comp) * 512ULL * 2ULL + 256ULL * 512ULL * 2ULL + 2ULL * 128ULL * 512ULL * 4ULL;
        } else {
            total_bytes += uint64_t(impl_->seq_len) * (576ULL + 128ULL) * 2ULL;
        }
    }
    if (impl_->has_dspark) {
        for (const auto& DL : impl_->dspark_layers) {
            if (DL.d_swa_ring) total_bytes += uint64_t(kDsparkSwaRingSize) * 512ULL * 2ULL;
        }
    }
    std::vector<uint8_t> out(static_cast<size_t>(total_bytes));
    size_t off = 0;
    auto pull = [&](int dev, const void* dp, size_t nb) {
        if (nb == 0 || !dp) return;
        hip_check(hipSetDevice(dev), "dump_kv dev");
        hip_check(hipMemcpy(out.data() + off, dp, nb, hipMemcpyDeviceToHost), "dump_kv D2H");
        off += nb;
    };
    for (int l = 0; l < impl_->mcfg.num_hidden_layers; ++l) {
        const auto& L = impl_->layers[size_t(l)];
        if (!L.is_dsa) {
            pull(L.owner_dev, L.d_hca_comp_kv,  size_t(n_comp) * 512u * 2u);
            pull(L.owner_dev, L.d_hca_swa_ring, 256u * 512u * 2u);
            pull(L.owner_dev, L.d_hca_state_kv, 128u * 512u * 4u);
            pull(L.owner_dev, L.d_hca_state_sc, 128u * 512u * 4u);
        } else {
            pull(L.owner_dev, L.d_dsa_mla_kv,   size_t(impl_->seq_len) * 576u * 2u);
            pull(L.owner_dev, L.d_dsa_ik_cache, size_t(impl_->seq_len) * 128u * 2u);
        }
    }
    if (impl_->has_dspark) {
        for (const auto& DL : impl_->dspark_layers) {
            pull(impl_->spine_dev1, DL.d_swa_ring, size_t(kDsparkSwaRingSize) * 512u * 2u);
        }
    }
    return out;
}

bool Ernie55Runtime::probe_p150_expert_parity(int layer_idx, int local_slot, const float* x_rot_f32, int M,
                                              std::vector<float>& out_y_f32) {
    if (!impl_->p150_ok || !impl_->p150_shm.ptr || M <= 0 || M > int(ERNIE55_P150_MAX_ROWS)) return false;
    auto* ch = impl_->p150_shm.ptr;
    const int n_p150 = impl_->p150_exp_end - impl_->p150_exp_begin;
    if (local_slot < 0 || local_slot >= n_p150) return false;
    for (int r = 0; r < M; ++r) {
        ch->row_slots[r][0]   = local_slot;
        ch->row_weights[r][0] = 1.0f;
        for (int s = 1; s < kTopKExperts; ++s) {
            ch->row_slots[r][s]   = -1;
            ch->row_weights[r][s] = 0.0f;
        }
    }
    const size_t n_lat = size_t(M) * kLatentDim;
    std::memcpy(ch->x_rot_f32, x_rot_f32, n_lat * sizeof(float));
    ch->op        = ERNIE55_P150_OP_MOE_EXEC;
    ch->layer_idx = uint32_t(layer_idx);
    ch->rows      = uint32_t(M);
    ch->top_k     = kTopKExperts;
    const uint32_t seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
    ch->req_seq.store(seq, std::memory_order_release);
    while (ch->resp_seq.load(std::memory_order_acquire) != seq) p150_cpu_relax();
    if (ch->status != 0) return false;
    out_y_f32.assign(ch->y_out_f32, ch->y_out_f32 + n_lat);
    return true;
}

void Ernie55Runtime::forward_tokens(const int32_t* tok_ids, int M, int32_t pos0,
                                    std::vector<float>& out_logits_MxV,
                                    bool capture_last_res_for_echo,
                                    bool capture_dspark_features) {
    impl_->forward_backbone_chunk(tok_ids, M, pos0, &out_logits_MxV, false,
                                  capture_last_res_for_echo, capture_dspark_features);
}

void Ernie55Runtime::generate_stream(
    const std::vector<int32_t>& prompt_ids,
    int max_new_tokens,
    const SamplingParams& sp,
    const std::vector<int32_t>& extra_stop_ids,
    const std::atomic<bool>* cancel_flag,
    const TokenCallback& on_token,
    GenerateStats& out_stats)
{
    out_stats = GenerateStats{};
    out_stats.prompt_tokens = int(prompt_ids.size());
    out_stats.draft_used = impl_->draft_method;
    if (prompt_ids.empty() || max_new_tokens <= 0) return;

    std::mt19937_64 rng(sp.seed);
    auto is_stop = [&](int32_t tid) -> bool {
        if (impl_->tok.is_eos_token(tid)) return true;
        return std::find(extra_stop_ids.begin(), extra_stop_ids.end(), tid) != extra_stop_ids.end();
    };

    const auto t_start = std::chrono::steady_clock::now();
    std::vector<float> logits;
    prefill_tokens_warm(prompt_ids, logits, &out_stats.cached_prefix_tokens);
    // Capture lightweight ~29.9 MiB GPU D2D prompt-boundary snapshot so next turn reuses exact KV
    // even if the chat harness strips prior <ernie:think>...</ernie:think> reasoning tokens!
    impl_->capture_hca_prompt_snapshot(prompt_ids);
    const auto t_prefill_done = std::chrono::steady_clock::now();
    out_stats.ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_done - t_start).count();

    SparseProbs probs;
    logits_to_sparse_probs_fast(logits.data(), impl_->mcfg.vocab_size, sp.temperature, sp.top_k, sp.top_p, probs);
    int32_t cur_tok = sample_from_sparse_probs(probs, rng);

    auto emit_tok = [&](int32_t tid) -> bool {
        ++out_stats.generated_tokens;
        ++impl_->total_decode_tokens;
        if (is_stop(tid)) {
            out_stats.finish_reason = "stop";
            return false;
        }
        const std::string piece = impl_->tok.decode_single(tid, false);
        if (on_token && !on_token(tid, piece)) {
            out_stats.finish_reason = "abort";
            return false;
        }
        if (out_stats.generated_tokens >= max_new_tokens) {
            out_stats.finish_reason = "length";
            return false;
        }
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
            out_stats.finish_reason = "abort";
            return false;
        }
        return true;
    };

    if (!emit_tok(cur_tok)) {
        out_stats.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prefill_done).count();
        return;
    }

    const bool use_dspark = (impl_->draft_method == DraftMethod::kDSpark &&
                             impl_->has_dspark &&
                             impl_->dspark_block_size > 0);
    const bool use_echo_2step = (!use_dspark &&
                                 impl_->draft_method == DraftMethod::kEcho &&
                                 impl_->has_echo_l43 &&
                                 sp.temperature <= 1e-6f &&
                                 impl_->opt.echo_draft_steps > 0);

    while (impl_->seq_len < impl_->opt.max_kv - 16) {
        // Strict 128-token HCA compression guard: pos0 + K <= 126 (mod 128) ALWAYS!
        // Ensures NO speculative pass ever triggers k_hca_compressor_reduce_norm_rope on a draft token,
        // and combined with the 256-slot HCA SWA ring guarantees 100% zero-leak rollback!
        const int hca_safe_k = (kHcaCompressRatio - 2) - (impl_->seq_len % kHcaCompressRatio);
        if (use_dspark && hca_safe_k >= 1 && (max_new_tokens - out_stats.generated_tokens) >= 2) {
            const int B = std::min({impl_->dspark_block_size, hca_safe_k, max_new_tokens - out_stats.generated_tokens});
            const auto prop = impl_->run_dspark_draft_block(cur_tok, impl_->seq_len, B, sp, rng);
            const int K = int(prop.draft_tokens.size());
            std::vector<int32_t> verify_in;
            verify_in.reserve(size_t(K + 1));
            verify_in.push_back(cur_tok);
            for (int32_t dt : prop.draft_tokens) verify_in.push_back(dt);

            const int32_t pos0 = impl_->seq_len;
            forward_tokens(verify_in.data(), K + 1, pos0, logits, false, true);

            const auto t_sv0 = impl_->profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const auto vr = DSparkHeadsCpu::verify_speculative_sparse(
                prop.draft_tokens, prop.draft_sparse, logits.data(), impl_->mcfg.vocab_size, sp, rng);
            if (impl_->profiling_enabled) {
                impl_->prof.sample_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sv0).count();
            }
            ++out_stats.spec_steps;
            out_stats.spec_drafted  += uint64_t(K);
            out_stats.spec_accepted += uint64_t(vr.n_draft_accepted);
            ++impl_->total_spec_steps;
            impl_->total_spec_drafted  += uint64_t(K);
            impl_->total_spec_accepted += uint64_t(vr.n_draft_accepted);

            // Commit ONLY accepted + anchor tokens [0 .. n_draft_accepted] to DSpark SWA KV & cached_tokens!
            const int n_commit = vr.n_draft_accepted + 1;
            impl_->commit_dspark_context_kv(n_commit, pos0);
            impl_->seq_len = pos0 + n_commit;
            impl_->cached_tokens.insert(impl_->cached_tokens.end(), verify_in.begin(), verify_in.begin() + n_commit);
            bool cont = true;
            for (int32_t tok : vr.accepted_and_bonus) {
                cur_tok = tok;
                if (!emit_tok(tok)) { cont = false; break; }
            }
            if (!cont) break;
        } else if (use_echo_2step && (max_new_tokens - out_stats.generated_tokens) >= 2 &&
                   ((impl_->seq_len % kHcaCompressRatio) + impl_->opt.echo_draft_steps) <= (kHcaCompressRatio - 2)) {
            const int K = std::min(impl_->opt.echo_draft_steps, max_new_tokens - out_stats.generated_tokens);
            std::vector<int32_t> draft_toks;
            draft_toks.reserve(size_t(K));
            std::vector<uint16_t> echo_res = impl_->h_last_backbone_res_4xH;
            int32_t d_in = cur_tok;
            for (int s = 0; s < K; ++s) {
                const int32_t dt = impl_->run_echo_draft_step(d_in, impl_->seq_len + s, echo_res);
                draft_toks.push_back(dt);
                d_in = dt;
            }
            // Verify [cur_tok, d_0, ..., d_{K-1}] in one M=K+1 target forward pass!
            std::vector<int32_t> verify_in;
            verify_in.reserve(size_t(K + 1));
            verify_in.push_back(cur_tok);
            for (int32_t dt : draft_toks) verify_in.push_back(dt);

            const int32_t pos0 = impl_->seq_len;
            forward_tokens(verify_in.data(), K + 1, pos0, logits, true, false);

            const auto vr = DSparkHeadsCpu::verify_speculative(
                draft_toks, {}, logits.data(), impl_->mcfg.vocab_size, sp, rng);
            ++out_stats.spec_steps;
            out_stats.spec_drafted  += uint64_t(K);
            out_stats.spec_accepted += uint64_t(vr.n_draft_accepted);
            ++impl_->total_spec_steps;
            impl_->total_spec_drafted  += uint64_t(K);
            impl_->total_spec_accepted += uint64_t(vr.n_draft_accepted);

            // Roll back BOTH logical seq_len AND exact ECHO 4-stream residual to accepted index!
            const int n_commit = vr.n_draft_accepted + 1;
            impl_->seq_len = pos0 + n_commit;
            impl_->cached_tokens.insert(impl_->cached_tokens.end(), verify_in.begin(), verify_in.begin() + n_commit);
            const int acc_row = std::min(K, std::max(0, vr.n_draft_accepted));
            impl_->h_last_backbone_res_4xH.assign(
                impl_->h_echo_res_pin + size_t(acc_row) * kHcDim,
                impl_->h_echo_res_pin + size_t(acc_row + 1) * kHcDim);
            bool cont = true;
            for (int32_t tok : vr.accepted_and_bonus) {
                cur_tok = tok;
                if (!emit_tok(tok)) { cont = false; break; }
            }
            if (!cont) break;
        } else {
            // Standard M=1 exact decode step (also cleanly flushes 128-token HCA boundary and commits DSpark ctx KV if active!)
            const int32_t pos0 = impl_->seq_len;
            forward_tokens(&cur_tok, 1, pos0, logits, true, use_dspark);
            if (use_dspark) impl_->commit_dspark_context_kv(1, pos0);
            impl_->seq_len = pos0 + 1;
            impl_->cached_tokens.push_back(cur_tok);
            logits_to_sparse_probs_fast(logits.data(), impl_->mcfg.vocab_size, sp.temperature, sp.top_k, sp.top_p, probs);
            cur_tok = sample_from_sparse_probs(probs, rng);
            if (!emit_tok(cur_tok)) break;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    out_stats.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill_done).count();
    if (out_stats.decode_ms > 0.0 && out_stats.generated_tokens > 1) {
        out_stats.decode_tok_s = double(out_stats.generated_tokens - 1) / (out_stats.decode_ms * 1e-3);
    }
}

namespace {

struct alignas(4096) Ernie55KvStripeHdr {
    uint32_t magic;            // 0x4B563536u ("KV56")
    uint32_t version;          // 2 = Striped Dual-NVMe O_DIRECT Lossless BF16 + SSE4.2 CRC32C
    int32_t  seq_len;
    int32_t  layer_begin;
    int32_t  layer_end;
    uint32_t stripe_idx;
    uint64_t payload_bytes;    // exact unpadded payload bytes
    uint64_t padded_bytes;     // 4096 B aligned payload bytes on disk
    uint32_t chunk_crc_xor;    // combined CRC32C over 4 MiB chunks (order-independent parallel verify!)
    uint32_t n_chunks;
    uint32_t has_dual_stripe;
    uint32_t header_crc32c;
    uint8_t  pad[4096 - 56];
};
static_assert(sizeof(Ernie55KvStripeHdr) == 4096, "O_DIRECT header must be 4096 B");

inline uint64_t align_up_4k(uint64_t v) noexcept { return (v + 4095ULL) & ~4095ULL; }

struct KvSeg {
    uint64_t off;
    void*    dev_ptr;
    uint64_t bytes;
};

void issue_kv_slab_dma(const std::vector<KvSeg>& segs, uint64_t slab_off, uint64_t slab_len,
                       uint8_t* host_buf, hipMemcpyKind kind, hipStream_t stream) {
    const uint64_t slab_end = slab_off + slab_len;
    for (const auto& sg : segs) {
        if (sg.off >= slab_end) break;
        const uint64_t sg_end = sg.off + sg.bytes;
        if (sg_end <= slab_off) continue;
        const uint64_t lo = std::max(slab_off, sg.off);
        const uint64_t hi = std::min(slab_end, sg_end);
        const uint64_t nb = hi - lo;
        uint8_t* hp = host_buf + (lo - slab_off);
        uint8_t* dp = static_cast<uint8_t*>(sg.dev_ptr) + (lo - sg.off);
        if (kind == hipMemcpyHostToDevice) {
            hip_check(hipMemcpyAsync(dp, hp, size_t(nb), hipMemcpyHostToDevice, stream), "H2D kvblk slab");
        } else {
            hip_check(hipMemcpyAsync(hp, dp, size_t(nb), hipMemcpyDeviceToHost, stream), "D2H kvblk slab");
        }
    }
}

} // namespace

KvBlkStats Ernie55Runtime::save_kvblk_ex(const std::string& path) const {
    const auto t_all0 = std::chrono::steady_clock::now();
    KvBlkStats st{};
    st.seq_len = impl_->seq_len;
    const int n_comp = impl_->seq_len / kHcaCompressRatio;
    const int split = impl_->split_layer;
    const int n_layers = impl_->mcfg.num_hidden_layers;

    // Pair nvme0n1 (path) with nvme1n1 (kv_stripe_dir/<basename>.s1) for 50/50 dual-NVMe striping!
    std::string stripe_paths[2];
    stripe_paths[0] = path;
    bool use_dual = false;
    if (!impl_->opt.kv_stripe_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(impl_->opt.kv_stripe_dir, ec);
        if (!ec) {
            stripe_paths[1] = impl_->opt.kv_stripe_dir + "/" + std::filesystem::path(path).filename().string() + ".s1";
            use_dual = true;
        }
    }
    st.dual_nvme = use_dual;

    struct StripeSpan { int l0, l1, dev; uint64_t bytes; uint64_t padded; std::vector<KvSeg> segs; };
    StripeSpan spans[2] = {
        {0, use_dual ? split : n_layers, impl_->spine_dev0, 0, 0, {}},
        {use_dual ? split : n_layers, n_layers, impl_->spine_dev1, 0, 0, {}}
    };
    const int n_stripes = use_dual ? 2 : 1;

    for (int s = 0; s < n_stripes; ++s) {
        uint64_t off = 0;
        for (int l = spans[s].l0; l < spans[s].l1; ++l) {
            const auto& L = impl_->layers[size_t(l)];
            auto add = [&](void* dp, uint64_t nb) {
                if (nb > 0 && dp) { spans[s].segs.push_back({off, dp, nb}); off += nb; }
            };
            if (!L.is_dsa) {
                add(L.d_hca_comp_kv,  uint64_t(n_comp) * 512ULL * 2ULL);
                add(L.d_hca_swa_ring, 256ULL * 512ULL * 2ULL);
                add(L.d_hca_state_kv, 128ULL * 512ULL * 4ULL);
                add(L.d_hca_state_sc, 128ULL * 512ULL * 4ULL);
            } else {
                add(L.d_dsa_mla_kv,   uint64_t(impl_->seq_len) * 576ULL * 2ULL);
                add(L.d_dsa_ik_cache, uint64_t(impl_->seq_len) * 128ULL * 2ULL);
            }
        }
        if (s == (n_stripes - 1) && impl_->has_dspark) {
            for (const auto& DL : impl_->dspark_layers) {
                if (DL.d_swa_ring) {
                    spans[s].segs.push_back({off, DL.d_swa_ring, uint64_t(kDsparkSwaRingSize) * 512ULL * 2ULL});
                    off += uint64_t(kDsparkSwaRingSize) * 512ULL * 2ULL;
                }
            }
        }
        spans[s].bytes  = off;
        spans[s].padded = align_up_4k(off);
        st.bytes += off;
    }

    // Ping-pong overlapped D2H DMA + 2x8=16-thread O_DIRECT pwrite + hardware SSE4.2 CRC32C!
    constexpr uint64_t kSlabBytes  = Impl::kKvSlabBytes;
    constexpr uint64_t kChunkBytes = 4ULL * 1024ULL * 1024ULL;
    constexpr int kIoThreadsPerDrive = 8;

    std::mutex err_mtx;
    std::string io_err;
    double dma_ms_arr[2] = {0.0, 0.0};
    double io_ms_arr[2]  = {0.0, 0.0};

    std::vector<std::thread> stripe_threads;
    stripe_threads.reserve(size_t(n_stripes));
    for (int s = 0; s < n_stripes; ++s) {
        stripe_threads.emplace_back([&, s]() {
            try {
                const int dev = spans[s].dev;
                auto& g = *impl_->dev_map.at(dev);
                hip_check(hipSetDevice(dev), "save kv dev");
                hipEvent_t ev[2]{};
                for (int b = 0; b < 2; ++b) hip_check(hipEventCreateWithFlags(&ev[b], hipEventDisableTiming), "ev save");

                const std::string tmp = stripe_paths[s] + ".tmp";
                int fd = ::open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, 0644);
                if (fd < 0) fd = ::open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
                if (fd < 0) throw std::runtime_error("Cannot open " + tmp);

                const uint64_t padded = spans[s].padded;
                const uint64_t payload = spans[s].bytes;
                const uint32_t n_chunks = uint32_t((padded + kChunkBytes - 1ULL) / kChunkBytes);
                const uint32_t n_slabs  = uint32_t((padded + kSlabBytes  - 1ULL) / kSlabBytes);
                std::vector<uint32_t> crcs(n_chunks, 0u);

                auto kick_d2h = [&](uint32_t si) {
                    const uint64_t soff = uint64_t(si) * kSlabBytes;
                    const uint64_t slen = std::min(kSlabBytes, padded - soff);
                    issue_kv_slab_dma(spans[s].segs, soff, slen, impl_->h_kv_pin[s][si & 1u], hipMemcpyDeviceToHost, g.stream);
                    hip_check(hipEventRecord(ev[si & 1u], g.stream), "ev rec save");
                };

                if (n_slabs > 0) kick_d2h(0);
                for (uint32_t si = 0; si < n_slabs; ++si) {
                    const uint32_t slot = si & 1u;
                    const uint64_t soff = uint64_t(si) * kSlabBytes;
                    const uint64_t slen = std::min(kSlabBytes, padded - soff);
                    uint8_t* buf = impl_->h_kv_pin[s][slot];

                    const auto td0 = std::chrono::steady_clock::now();
                    hip_check(hipEventSynchronize(ev[slot]), "ev sync save");
                    dma_ms_arr[s] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();

                    if (si + 1u < n_slabs) kick_d2h(si + 1u);

                    if (soff + slen > payload) {
                        const uint64_t z0 = (payload > soff) ? (payload - soff) : 0ULL;
                        std::memset(buf + z0, 0, size_t(slen - z0));
                    }

                    const auto ti0 = std::chrono::steady_clock::now();
                    const uint32_t c_beg = uint32_t(soff / kChunkBytes);
                    const uint32_t c_end = uint32_t((soff + slen + kChunkBytes - 1ULL) / kChunkBytes);
                    const uint32_t c_cnt = c_end - c_beg;
                    const int nw = std::min<int>(kIoThreadsPerDrive, int(c_cnt));
                    std::atomic<uint32_t> next_c{0};
                    std::atomic<bool> failed{false};
                    std::vector<std::thread> workers;
                    workers.reserve(size_t(nw));
                    for (int w = 0; w < nw; ++w) {
                        workers.emplace_back([&]() {
                            while (!failed.load(std::memory_order_relaxed)) {
                                const uint32_t lc = next_c.fetch_add(1u, std::memory_order_relaxed);
                                if (lc >= c_cnt) break;
                                const uint32_t ci = c_beg + lc;
                                const uint64_t coff = uint64_t(ci) * kChunkBytes;
                                const uint64_t clen = std::min(kChunkBytes, padded - coff);
                                const uint64_t local_off = coff - soff;
                                const uint64_t crc_len = (coff < payload) ? std::min(clen, payload - coff) : 0ULL;
                                if (crc_len > 0) crcs[size_t(ci)] = hw_crc32c(buf + local_off, size_t(crc_len));
                                uint64_t wr_done = 0;
                                while (wr_done < clen) {
                                    const ssize_t wr = ::pwrite(fd, buf + local_off + wr_done, size_t(clen - wr_done),
                                                                off_t(sizeof(Ernie55KvStripeHdr) + coff + wr_done));
                                    if (wr <= 0) { failed.store(true, std::memory_order_relaxed); return; }
                                    wr_done += uint64_t(wr);
                                }
                            }
                        });
                    }
                    for (auto& wt : workers) wt.join();
                    io_ms_arr[s] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti0).count();
                    if (failed.load()) { ::close(fd); ::unlink(tmp.c_str()); throw std::runtime_error("Short O_DIRECT pwrite " + tmp); }
                }
                for (int b = 0; b < 2; ++b) (void)hipEventDestroy(ev[b]);

                alignas(4096) Ernie55KvStripeHdr hdr{};
                hdr.magic           = 0x4B563536u;
                hdr.version         = 2u;
                hdr.seq_len         = impl_->seq_len;
                hdr.layer_begin     = spans[s].l0;
                hdr.layer_end       = spans[s].l1;
                hdr.stripe_idx      = uint32_t(s);
                hdr.payload_bytes   = payload;
                hdr.padded_bytes    = padded;
                hdr.n_chunks        = n_chunks;
                hdr.chunk_crc_xor   = hw_crc32c(crcs.data(), crcs.size() * sizeof(uint32_t));
                hdr.has_dual_stripe = use_dual ? 1u : 0u;
                hdr.header_crc32c   = hw_crc32c(&hdr, offsetof(Ernie55KvStripeHdr, header_crc32c));
                if (::pwrite(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr))) {
                    ::close(fd); ::unlink(tmp.c_str());
                    throw std::runtime_error("Header pwrite failed " + tmp);
                }
                (void)::fdatasync(fd);
                (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                ::close(fd);
                if (::rename(tmp.c_str(), stripe_paths[s].c_str()) != 0) {
                    ::unlink(tmp.c_str());
                    throw std::runtime_error("Rename failed " + stripe_paths[s]);
                }
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(err_mtx);
                io_err = ex.what();
            }
        });
    }
    for (auto& t : stripe_threads) t.join();
    if (!io_err.empty()) throw std::runtime_error(io_err);

    st.dma_ms   = std::max(dma_ms_arr[0], dma_ms_arr[1]);
    st.io_ms    = std::max(io_ms_arr[0],  io_ms_arr[1]);
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all0).count();
    st.gb_s     = (st.total_ms > 0.0) ? (double(st.bytes) / 1e9) / (st.total_ms * 1e-3) : 0.0;
    st.tok_s    = (st.total_ms > 0.0) ? double(st.seq_len) / (st.total_ms * 1e-3) : 0.0;

    // Persist warm-prefix token metadata sidecar (<path>.toks) so cold restart + restore hits 100% warm-prefix cache!
    if (!impl_->cached_tokens.empty()) {
        const std::string tpath = path + ".toks";
        const std::string ttmp  = tpath + ".tmp";
        std::ofstream ofs(ttmp, std::ios::binary | std::ios::trunc);
        if (ofs) {
            const uint32_t magic = 0x544F4B35u; // "TOK5"
            const uint32_t n_cached = uint32_t(impl_->cached_tokens.size());
            const uint32_t n_ckpt   = uint32_t(impl_->ckpt_prompt_tokens.size());
            const uint32_t has_snap = (n_ckpt > 0 && int(n_ckpt) != impl_->seq_len) ? 1u : 0u;
            ofs.write(reinterpret_cast<const char*>(&magic), 4);
            ofs.write(reinterpret_cast<const char*>(&n_cached), 4);
            ofs.write(reinterpret_cast<const char*>(&n_ckpt), 4);
            ofs.write(reinterpret_cast<const char*>(&has_snap), 4);
            ofs.write(reinterpret_cast<const char*>(impl_->cached_tokens.data()), std::streamsize(n_cached * 4u));
            if (n_ckpt > 0) {
                ofs.write(reinterpret_cast<const char*>(impl_->ckpt_prompt_tokens.data()), std::streamsize(n_ckpt * 4u));
            }
            if (has_snap) {
                std::vector<uint8_t> buf(256u * 512u * 4u);
                for (int l = 0; l < impl_->mcfg.num_hidden_layers; ++l) {
                    const auto& L = impl_->layers[size_t(l)];
                    if (!L.is_dsa && L.d_hca_snap_swa) {
                        hip_check(hipSetDevice(L.owner_dev), "save snap dev");
                        hip_check(hipMemcpy(buf.data(), L.d_hca_snap_swa, 256u * 512u * 2u, hipMemcpyDeviceToHost), "save snap swa");
                        ofs.write(reinterpret_cast<const char*>(buf.data()), 256u * 512u * 2u);
                        hip_check(hipMemcpy(buf.data(), L.d_hca_snap_kv,  128u * 512u * 4u, hipMemcpyDeviceToHost), "save snap kv");
                        ofs.write(reinterpret_cast<const char*>(buf.data()), 128u * 512u * 4u);
                        hip_check(hipMemcpy(buf.data(), L.d_hca_snap_sc,  128u * 512u * 4u, hipMemcpyDeviceToHost), "save snap sc");
                        ofs.write(reinterpret_cast<const char*>(buf.data()), 128u * 512u * 4u);
                    }
                }
                if (impl_->has_dspark) {
                    hip_check(hipSetDevice(impl_->spine_dev1), "save ds snap dev");
                    for (const auto& DL : impl_->dspark_layers) {
                        if (DL.d_swa_snap) {
                            hip_check(hipMemcpy(buf.data(), DL.d_swa_snap, size_t(kDsparkSwaRingSize) * 512u * 2u, hipMemcpyDeviceToHost), "save ds snap");
                            ofs.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(size_t(kDsparkSwaRingSize) * 512u * 2u));
                        }
                    }
                }
            }
            ofs.close();
            (void)::rename(ttmp.c_str(), tpath.c_str());
        }
    }
    return st;
}

KvBlkStats Ernie55Runtime::load_kvblk_ex(const std::string& path) {
    const auto t_all0 = std::chrono::steady_clock::now();
    KvBlkStats st{};

    alignas(4096) Ernie55KvStripeHdr hdrs[2]{};
    std::string stripe_paths[2];
    stripe_paths[0] = path;
    {
        const int fd0 = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        const int fd = (fd0 >= 0) ? fd0 : ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open kvblk: " + path);
        if (::pread(fd, &hdrs[0], sizeof(Ernie55KvStripeHdr), 0) != ssize_t(sizeof(Ernie55KvStripeHdr)) ||
            hdrs[0].magic != 0x4B563536u || hdrs[0].version != 2u ||
            hw_crc32c(&hdrs[0], offsetof(Ernie55KvStripeHdr, header_crc32c)) != hdrs[0].header_crc32c) {
            ::close(fd);
            throw std::runtime_error("Invalid or corrupt kvblk stripe0 header: " + path);
        }
        ::close(fd);
    }

    const int n_stripes = (hdrs[0].has_dual_stripe != 0u) ? 2 : 1;
    st.dual_nvme = (n_stripes == 2);
    st.seq_len = hdrs[0].seq_len;
    if (n_stripes == 2) {
        stripe_paths[1] = impl_->opt.kv_stripe_dir + "/" + std::filesystem::path(path).filename().string() + ".s1";
        const int fd1 = ::open(stripe_paths[1].c_str(), O_RDONLY | O_DIRECT);
        const int fd = (fd1 >= 0) ? fd1 : ::open(stripe_paths[1].c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open kvblk stripe1: " + stripe_paths[1]);
        if (::pread(fd, &hdrs[1], sizeof(Ernie55KvStripeHdr), 0) != ssize_t(sizeof(Ernie55KvStripeHdr)) ||
            hdrs[1].magic != 0x4B563536u || hdrs[1].seq_len != hdrs[0].seq_len ||
            hw_crc32c(&hdrs[1], offsetof(Ernie55KvStripeHdr, header_crc32c)) != hdrs[1].header_crc32c) {
            ::close(fd);
            throw std::runtime_error("Invalid or corrupt kvblk stripe1 header: " + stripe_paths[1]);
        }
        ::close(fd);
    }

    const int n_comp = hdrs[0].seq_len / kHcaCompressRatio;
    std::vector<KvSeg> stripe_segs[2];
    int stripe_devs[2] = {impl_->spine_dev0, impl_->spine_dev1};
    for (int s = 0; s < n_stripes; ++s) {
        st.bytes += hdrs[s].payload_bytes;
        uint64_t off = 0;
        for (int l = hdrs[s].layer_begin; l < hdrs[s].layer_end; ++l) {
            auto& L = impl_->layers[size_t(l)];
            stripe_devs[s] = L.owner_dev;
            auto add = [&](void* dp, uint64_t nb) {
                if (nb > 0 && dp) { stripe_segs[s].push_back({off, dp, nb}); off += nb; }
            };
            if (!L.is_dsa) {
                add(L.d_hca_comp_kv,  uint64_t(n_comp) * 512ULL * 2ULL);
                add(L.d_hca_swa_ring, 256ULL * 512ULL * 2ULL);
                add(L.d_hca_state_kv, 128ULL * 512ULL * 4ULL);
                add(L.d_hca_state_sc, 128ULL * 512ULL * 4ULL);
            } else {
                add(L.d_dsa_mla_kv,   uint64_t(hdrs[0].seq_len) * 576ULL * 2ULL);
                add(L.d_dsa_ik_cache, uint64_t(hdrs[0].seq_len) * 128ULL * 2ULL);
            }
        }
        if (s == (n_stripes - 1) && impl_->has_dspark) {
            for (auto& DL : impl_->dspark_layers) {
                if (DL.d_swa_ring) {
                    stripe_segs[s].push_back({off, DL.d_swa_ring, uint64_t(kDsparkSwaRingSize) * 512ULL * 2ULL});
                    off += uint64_t(kDsparkSwaRingSize) * 512ULL * 2ULL;
                }
            }
        }
    }

    // Ping-pong overlapped 2x8=16-thread O_DIRECT pread + L2-hot SSE4.2 CRC32C + concurrent dev0/dev1 H2D DMA!
    constexpr uint64_t kSlabBytes  = Impl::kKvSlabBytes;
    constexpr uint64_t kChunkBytes = 4ULL * 1024ULL * 1024ULL;
    constexpr int kIoThreadsPerDrive = 8;

    std::mutex err_mtx;
    std::string io_err;
    double io_ms_arr[2]  = {0.0, 0.0};
    double dma_ms_arr[2] = {0.0, 0.0};

    std::vector<std::thread> stripe_threads;
    stripe_threads.reserve(size_t(n_stripes));
    for (int s = 0; s < n_stripes; ++s) {
        stripe_threads.emplace_back([&, s]() {
            try {
                const int dev = stripe_devs[s];
                auto& g = *impl_->dev_map.at(dev);
                hip_check(hipSetDevice(dev), "load kv dev");
                hipEvent_t ev[2]{};
                for (int b = 0; b < 2; ++b) hip_check(hipEventCreateWithFlags(&ev[b], hipEventDisableTiming), "ev load");

                int fd = ::open(stripe_paths[s].c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) fd = ::open(stripe_paths[s].c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("Cannot open " + stripe_paths[s]);
                (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

                const uint64_t padded  = hdrs[s].padded_bytes;
                const uint64_t payload = hdrs[s].payload_bytes;
                const uint32_t n_chunks = hdrs[s].n_chunks;
                const uint32_t n_slabs  = uint32_t((padded + kSlabBytes - 1ULL) / kSlabBytes);
                std::vector<uint32_t> crcs(n_chunks, 0u);

                for (uint32_t si = 0; si < n_slabs; ++si) {
                    const uint32_t slot = si & 1u;
                    const uint64_t soff = uint64_t(si) * kSlabBytes;
                    const uint64_t slen = std::min(kSlabBytes, padded - soff);
                    uint8_t* buf = impl_->h_kv_pin[s][slot];

                    if (si >= 2u) {
                        const auto td0 = std::chrono::steady_clock::now();
                        hip_check(hipEventSynchronize(ev[slot]), "ev sync load");
                        dma_ms_arr[s] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                    }

                    const auto ti0 = std::chrono::steady_clock::now();
                    const uint32_t c_beg = uint32_t(soff / kChunkBytes);
                    const uint32_t c_end = uint32_t((soff + slen + kChunkBytes - 1ULL) / kChunkBytes);
                    const uint32_t c_cnt = c_end - c_beg;
                    const int nw = std::min<int>(kIoThreadsPerDrive, int(c_cnt));
                    std::atomic<uint32_t> next_c{0};
                    std::atomic<bool> failed{false};
                    std::vector<std::thread> workers;
                    workers.reserve(size_t(nw));
                    for (int w = 0; w < nw; ++w) {
                        workers.emplace_back([&]() {
                            while (!failed.load(std::memory_order_relaxed)) {
                                const uint32_t lc = next_c.fetch_add(1u, std::memory_order_relaxed);
                                if (lc >= c_cnt) break;
                                const uint32_t ci = c_beg + lc;
                                const uint64_t coff = uint64_t(ci) * kChunkBytes;
                                const uint64_t clen = std::min(kChunkBytes, padded - coff);
                                const uint64_t local_off = coff - soff;
                                uint64_t rd_done = 0;
                                while (rd_done < clen) {
                                    const ssize_t r = ::pread(fd, buf + local_off + rd_done, size_t(clen - rd_done),
                                                              off_t(sizeof(Ernie55KvStripeHdr) + coff + rd_done));
                                    if (r <= 0) { failed.store(true, std::memory_order_relaxed); return; }
                                    rd_done += uint64_t(r);
                                }
                                const uint64_t crc_len = (coff < payload) ? std::min(clen, payload - coff) : 0ULL;
                                if (crc_len > 0) crcs[size_t(ci)] = hw_crc32c(buf + local_off, size_t(crc_len));
                            }
                        });
                    }
                    for (auto& wt : workers) wt.join();
                    io_ms_arr[s] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti0).count();
                    if (failed.load()) { ::close(fd); throw std::runtime_error("Short O_DIRECT pread " + stripe_paths[s]); }

                    issue_kv_slab_dma(stripe_segs[s], soff, slen, buf, hipMemcpyHostToDevice, g.stream);
                    hip_check(hipEventRecord(ev[slot], g.stream), "ev rec load");
                }
                ::close(fd);

                const auto td1 = std::chrono::steady_clock::now();
                hip_check(hipStreamSynchronize(g.stream), "sync kv load stream");
                dma_ms_arr[s] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td1).count();
                for (int b = 0; b < 2; ++b) (void)hipEventDestroy(ev[b]);

                if (hw_crc32c(crcs.data(), crcs.size() * sizeof(uint32_t)) != hdrs[s].chunk_crc_xor) {
                    throw std::runtime_error("SSE4.2 CRC32C mismatch in kvblk stripe " + std::to_string(s));
                }
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(err_mtx);
                io_err = ex.what();
            }
        });
    }
    for (auto& t : stripe_threads) t.join();
    if (!io_err.empty()) throw std::runtime_error(io_err);

    impl_->seq_len = hdrs[0].seq_len;
    impl_->mirror_hca_kv_from_owner(impl_->seq_len, false);
    st.io_ms    = std::max(io_ms_arr[0],  io_ms_arr[1]);
    st.dma_ms   = std::max(dma_ms_arr[0], dma_ms_arr[1]);
    st.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all0).count();
    st.gb_s     = (st.total_ms > 0.0) ? (double(st.bytes) / 1e9) / (st.total_ms * 1e-3) : 0.0;
    st.tok_s    = (st.total_ms > 0.0) ? double(st.seq_len) / (st.total_ms * 1e-3) : 0.0;

    // Restore warm-prefix token metadata sidecar (<path>.toks) if present!
    impl_->cached_tokens.clear();
    impl_->ckpt_prompt_tokens.clear();
    impl_->ckpt_prompt_len = 0;
    {
        const std::string tpath = path + ".toks";
        std::ifstream ifs(tpath, std::ios::binary);
        uint32_t magic = 0, n_cached = 0, n_ckpt = 0, has_snap = 0;
        if (ifs &&
            ifs.read(reinterpret_cast<char*>(&magic), 4) &&
            ifs.read(reinterpret_cast<char*>(&n_cached), 4) &&
            ifs.read(reinterpret_cast<char*>(&n_ckpt), 4) &&
            ifs.read(reinterpret_cast<char*>(&has_snap), 4) &&
            magic == 0x544F4B35u && int(n_cached) == impl_->seq_len) {
            impl_->cached_tokens.resize(n_cached);
            ifs.read(reinterpret_cast<char*>(impl_->cached_tokens.data()), std::streamsize(n_cached * 4u));
            if (n_ckpt > 0) {
                impl_->ckpt_prompt_tokens.resize(n_ckpt);
                ifs.read(reinterpret_cast<char*>(impl_->ckpt_prompt_tokens.data()), std::streamsize(n_ckpt * 4u));
            }
            if (ifs.good()) {
                if (has_snap) {
                    std::vector<uint8_t> buf(256u * 512u * 4u);
                    for (int l = 0; l < impl_->mcfg.num_hidden_layers; ++l) {
                        auto& L = impl_->layers[size_t(l)];
                        if (!L.is_dsa && L.d_hca_snap_swa) {
                            hip_check(hipSetDevice(L.owner_dev), "load snap dev");
                            ifs.read(reinterpret_cast<char*>(buf.data()), 256u * 512u * 2u);
                            hip_check(hipMemcpy(L.d_hca_snap_swa, buf.data(), 256u * 512u * 2u, hipMemcpyHostToDevice), "load snap swa");
                            ifs.read(reinterpret_cast<char*>(buf.data()), 128u * 512u * 4u);
                            hip_check(hipMemcpy(L.d_hca_snap_kv,  buf.data(), 128u * 512u * 4u, hipMemcpyHostToDevice), "load snap kv");
                            ifs.read(reinterpret_cast<char*>(buf.data()), 128u * 512u * 4u);
                            hip_check(hipMemcpy(L.d_hca_snap_sc,  buf.data(), 128u * 512u * 4u, hipMemcpyHostToDevice), "load snap sc");
                        }
                    }
                    if (impl_->has_dspark) {
                        hip_check(hipSetDevice(impl_->spine_dev1), "load ds snap dev");
                        for (auto& DL : impl_->dspark_layers) {
                            if (DL.d_swa_snap) {
                                ifs.read(reinterpret_cast<char*>(buf.data()), std::streamsize(size_t(kDsparkSwaRingSize) * 512u * 2u));
                                hip_check(hipMemcpy(DL.d_swa_snap, buf.data(), size_t(kDsparkSwaRingSize) * 512u * 2u, hipMemcpyHostToDevice), "load ds snap");
                            }
                        }
                    }
                    impl_->mirror_hca_kv_from_owner(impl_->seq_len, true);
                    impl_->ckpt_prompt_len = int(n_ckpt);
                } else if (n_ckpt > 0 && int(n_ckpt) == impl_->seq_len) {
                    impl_->capture_hca_prompt_snapshot(impl_->ckpt_prompt_tokens);
                }
            } else {
                impl_->cached_tokens.clear();
                impl_->ckpt_prompt_tokens.clear();
                impl_->ckpt_prompt_len = 0;
            }
        }
    }
    return st;
}

void Ernie55Runtime::save_kvblk(const std::string& path) const { (void)save_kvblk_ex(path); }
void Ernie55Runtime::load_kvblk(const std::string& path) { (void)load_kvblk_ex(path); }

JsonValue Ernie55Runtime::metrics_json() const {
    JsonValue j = JsonValue::make_object();
    j.set("topology", JsonValue::make_string(impl_->topo.name));
    j.set("total_vram_gib", JsonValue::make_int(impl_->topo.total_vram_gib));
    j.set("ddr4_expert_spill_bytes", JsonValue::make_int(0));
    j.set("max_kv", JsonValue::make_int(impl_->opt.max_kv));
    j.set("seq_len", JsonValue::make_int(impl_->seq_len));
    j.set("p150_attached", JsonValue::make_bool(impl_->p150_ok));
    j.set("has_echo_l43", JsonValue::make_bool(impl_->has_echo_l43));
    j.set("has_bundled_dspark", JsonValue::make_bool(impl_->has_dspark));
    j.set("dspark_block_size", JsonValue::make_int(impl_->dspark_block_size));
    j.set("flashindexer_enabled", JsonValue::make_bool(impl_->use_flashindexer));
    j.set("total_prefill_tokens", JsonValue::make_int(int64_t(impl_->total_prefill_tokens)));
    j.set("total_decode_tokens", JsonValue::make_int(int64_t(impl_->total_decode_tokens)));
    j.set("spec_steps", JsonValue::make_int(int64_t(impl_->total_spec_steps)));
    j.set("spec_drafted", JsonValue::make_int(int64_t(impl_->total_spec_drafted)));
    j.set("spec_accepted", JsonValue::make_int(int64_t(impl_->total_spec_accepted)));

    JsonValue devs = JsonValue::make_array();
    for (const auto& w : impl_->gpus) {
        size_t free_b = 0, total_b = 0;
        (void)hipSetDevice(w->hip_dev);
        (void)hipMemGetInfo(&free_b, &total_b);
        JsonValue d = JsonValue::make_object();
        d.set("hip_dev", JsonValue::make_int(w->hip_dev));
        d.set("used_mib", JsonValue::make_double(double(total_b - free_b) / 1048576.0));
        d.set("free_mib", JsonValue::make_double(double(free_b) / 1048576.0));
        d.set("bb_experts_per_layer", JsonValue::make_int(w->bb_count()));
        d.set("echo_experts", JsonValue::make_int(w->echo_count()));
        devs.arr_val.push_back(std::move(d));
    }
    j.set("mi50_devices", std::move(devs));
    return j;
}

} // namespace ernie55
