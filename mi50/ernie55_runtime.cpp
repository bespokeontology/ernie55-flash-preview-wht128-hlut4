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
#include "ernie55_kvblk_cache.hpp"
#include "ernie55_p150_shm.hpp"
#include "ernie55_ngram_lookahead.hpp"
#include "safetensors_mmap.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
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

inline bool gfx906_rocblas_sol_enabled() noexcept {
    static const bool enabled = []() {
        const char* env = std::getenv("ERNIE55_ROCBLAS_SOL");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

inline bool gfx906_offload_shadow_enabled() noexcept {
    static const bool enabled = []() {
        const char* env = std::getenv("ERNIE55_OFFLOAD_SHADOW");
        return (env && env[0] == '1' && env[1] == '\0');
    }();
    return enabled;
}

inline bool gfx906_gpu_ar_enabled() noexcept {
    static const bool enabled = []() {
        const char* env = std::getenv("ERNIE55_GPU_AR");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

inline bool gfx906_mapped_tp4_enabled() noexcept {
    static const bool enabled = []() {
        const char* env = std::getenv("ERNIE55_MAPPED_TP4");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

inline bool gfx906_mapped_moe_enabled() noexcept {
    static const bool enabled = []() {
        const char* env = std::getenv("ERNIE55_MAPPED_MOE");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

// Wave-12 Turn-17 Cut D12.1: M-fused double-buffered HCA split-K decode pass 1.
// Set ERNIE55_HCA_MFUSED=0 to fall back to the legacy M-loop kernel.
inline bool gfx906_hca_mfused_decode_enabled() noexcept {
    static const bool enabled = []() {
        const char* env = std::getenv("ERNIE55_HCA_MFUSED");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

// Wave-11 Turn-4/5 AR modes:
//   0 = host hipEventSynchronize + SDMA H2D (pre-Turn-4 baseline)
//   1 = device hipStreamWaitEvent + direct GPU kernel reads of mapped pinned host pages
//   2 = device hipStreamWaitEvent + full-bandwidth SDMA H2D (0 host blocks, full PCIe Gen4 SDMA!) [default]
inline int gfx906_ar_device_mode() noexcept {
    static const int mode = []() {
        const char* env = std::getenv("ERNIE55_AR_DEVICE");
        if (!env || env[0] == '\0') return 2;
        if (env[0] == '0' && env[1] == '\0') return 0;
        if (env[0] == '1' && env[1] == '\0') return 1;
        return 2;
    }();
    return mode;
}

// Verified 100% lossless Type_HH_HPA / Type_HS_HPA (compute = f32_r) Tensile solution_index table
// for AMD Radeon Instinct MI50 (gfx906). Automatic fallback to rocblas_gemm_algo_standard on mismatch!
inline int32_t gfx906_tuned_gemm_solution(rocblas_operation opA, rocblas_operation opB,
                                          int m, int n, int k,
                                          rocblas_datatype out_dt) noexcept {
    if (!gfx906_rocblas_sol_enabled()) return 0;
    if (out_dt == rocblas_datatype_f16_r) {
        if (opA == rocblas_operation_transpose && opB == rocblas_operation_none) {
            if (k == 128 && m == 4096 && n >= 512) return 33065; // FlashIndexer (3.1x: 0.80 -> 0.25 ms)
            if (k == 512 && (m == 64 || m == 128 || m == 256) && n >= 2048) return 32975; // HCA Comp-KV QK 64/128/256
            if (k == 512 && (m == 512 || m == 1024 || m == 2048 || m == 4096) && n >= 2048) return 32989; // HCA Comp-KV QK 512..4096
            if (k == 576 && m == 256 && n >= 1024) return 32977; // DSA Dense QK (covers TP4 n=4096!)
            if (n >= 512) {
                if (m == 2560 && k == 4096) return 33064; // HCA qkv_comp (4.44 -> 2.35 ms, 18.3 TF)
                if (m == 8192 && k == 1024) return 33065; // HCA q_up / DSA idx_wq_b (20.4 TF)
                if (m == 4096 && k == 2048) return (n >= 1024) ? 33065 : 32975; // HCA o_proj / MoE fc2_lat
                if (m == 2048 && k == 1024) return 32989; // HCA gate_proj (0.80 -> 0.53 ms)
                if (m == 1024 && k == 4096) return 33068; // DSA q_a (1.73 -> 1.33 ms)
                if (m == 4096 && k == 1024) return 33064; // DSA q_b / gate_proj (1.56 -> 0.95 ms)
                if (m == 576  && k == 4096) return 32975; // DSA kv_a (1.13 -> 0.76 ms)
                if (m == 2048 && k == 4096) return 32990; // MoE fc1_lat (3.56 -> 2.12 ms)
                if (m == 1536 && k == 4096) return 33065; // MoE sh_w1/sh_w3 (2.77 -> 1.59 ms)
                if (m == 4096 && k == 1536) return 33065; // MoE sh_w2 (2.35 -> 1.28 ms, 20.2 TF)
                if (m == 4096 && k == 4096) return (n >= 1024) ? 33064 : 32975; // DSA o_proj
                if (m == 3584 && k == 4096) return 33064; // Dense gate/up (6.13 -> 3.54 ms)
                if (m == 4096 && k == 3584) return 33064; // Dense down (6.01 -> 3.16 ms, 19.0 TF)
            }
        } else if (opA == rocblas_operation_none && opB == rocblas_operation_none) {
            if (m == 512 && (k == 64 || k == 128 || k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096) && n >= 2048) return 33161; // HCA Comp-KV PV (19.3 TF)
            if (m == 512 && k == 256 && n >= 1024) return 33152;               // DSA Dense PV (covers TP4 n=4096!)
        }
    } else if (out_dt == rocblas_datatype_f32_r) {
        if (opA == rocblas_operation_transpose && opB == rocblas_operation_none) {
            if (m == 1024 && k == 4096 && n >= 1024) return 31468; // MoE router f16->f32
        }
    }
    return 0;
}

inline int32_t gfx906_tuned_sbat_solution(rocblas_operation opA, rocblas_operation opB,
                                          int m, int n, int k, int batch) noexcept {
    if (!gfx906_rocblas_sol_enabled()) return 0;
    if (opA == rocblas_operation_transpose && opB == rocblas_operation_none) {
        if (m == 512 && k == 4096 && batch == 4 && n >= 512) return 32975;  // HCA wo_a batched
        if (m == 256 && k == 512 && batch == 16 && n >= 512) return (n >= 2048) ? 32990 : 33066; // DSA W_UV
        if (m == 256 && n == 2048 && k == 512 && batch == 8) return 32989;  // HCA SWA-128 QK b=8
        if (m == 256 && n == 2048 && k == 512 && batch == 16) return 33063; // HCA SWA-128 QK b=16
    } else if (opA == rocblas_operation_none && opB == rocblas_operation_none) {
        if (m == 512 && k == 192 && batch == 16 && n >= 512) return 33181;  // DSA W_UK (5.0x: 2.13 -> 0.43 ms)
        if (m == 512 && n == 2048 && k == 256 && batch == 8) return 33169;  // HCA SWA-128 PV b=8
        if (m == 512 && n == 2048 && k == 256 && batch == 16) return 33161; // HCA SWA-128 PV b=16
    }
    return 0;
}

inline void rocblas_gemm_ex_tuned(
    rocblas_handle blas, rocblas_operation opA, rocblas_operation opB,
    rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda,
    const void* b, rocblas_datatype b_type, rocblas_int ldb,
    const float* beta,
    const void* c, rocblas_datatype c_type, rocblas_int ldc,
    void* d, rocblas_datatype d_type, rocblas_int ldd,
    const char* what)
{
    const int32_t sol = (a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r && c_type == d_type)
        ? gfx906_tuned_gemm_solution(opA, opB, int(m), int(n), int(k), d_type) : 0;
    if (sol != 0) {
        const rocblas_status st = rocblas_gemm_ex(
            blas, opA, opB, m, n, k, alpha,
            a, a_type, lda, b, b_type, ldb, beta,
            c, c_type, ldc, d, d_type, ldd,
            rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, sol, 0);
        if (st == rocblas_status_success) return;
    }
    rocblas_check(rocblas_gemm_ex(
        blas, opA, opB, m, n, k, alpha,
        a, a_type, lda, b, b_type, ldb, beta,
        c, c_type, ldc, d, d_type, ldd,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), what);
}

inline void rocblas_gemm_strided_batched_ex_tuned(
    rocblas_handle blas, rocblas_operation opA, rocblas_operation opB,
    rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda, rocblas_stride stride_a,
    const void* b, rocblas_datatype b_type, rocblas_int ldb, rocblas_stride stride_b,
    const float* beta,
    const void* c, rocblas_datatype c_type, rocblas_int ldc, rocblas_stride stride_c,
    void* d, rocblas_datatype d_type, rocblas_int ldd, rocblas_stride stride_d,
    rocblas_int batch_count, const char* what)
{
    const int32_t sol = (a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
                         c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
        ? gfx906_tuned_sbat_solution(opA, opB, int(m), int(n), int(k), int(batch_count)) : 0;
    if (sol != 0) {
        const rocblas_status st = rocblas_gemm_strided_batched_ex(
            blas, opA, opB, m, n, k, alpha,
            a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta,
            c, c_type, ldc, stride_c, d, d_type, ldd, stride_d,
            batch_count, rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, sol, 0);
        if (st == rocblas_status_success) return;
    }
    rocblas_check(rocblas_gemm_strided_batched_ex(
        blas, opA, opB, m, n, k, alpha,
        a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta,
        c, c_type, ldc, stride_c, d, d_type, ldd, stride_d,
        batch_count, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), what);
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

// Hardware F16C+AVX2 FP16 -> BF16 RNE (24-thread EPYC 7542, 8 MiB in ~0.04 ms for P150A native BF16 DMA!)
inline void avx2_f16_to_bf16(const uint16_t* src, uint16_t* dst, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    const __m256i v_bias = _mm256_set1_epi32(0x7FFF);
    const __m256i v_one  = _mm256_set1_epi32(1);
    const __m256i v_shuf = _mm256_setr_epi8(
        2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1,
        2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n >= 65536)
    for (size_t i = 0; i < n8; i += 8) {
        __m256  f   = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
        __m256i u   = _mm256_castps_si256(f);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), v_one);
        u = _mm256_add_epi32(u, _mm256_add_epi32(v_bias, lsb));
        __m256i p = _mm256_shuffle_epi8(u, v_shuf);
        __m128i out = _mm_unpacklo_epi64(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), out);
    }
    for (size_t i = n8; i < n; ++i) dst[i] = f32_to_bf16_rne(fp16_to_f32(src[i]));
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

// Hardware AVX2 BF16 -> FP32 copy (dst[i] = bf16_to_f32(src[i]))
inline void avx2_bf16_to_f32(const uint16_t* src, float* dst, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n >= 65536)
    for (size_t i = 0; i < n8; i += 8) {
        __m128i u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256  f8  = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u16), 16));
        _mm256_storeu_ps(dst + i, f8);
    }
    for (size_t i = n8; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}

// Hardware AVX2 BF16 -> FP32 accumulate (dst[i] += bf16_to_f32(src[i]))
inline void avx2_add_bf16_to_f32_inplace(float* dst, const uint16_t* src, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n >= 65536)
    for (size_t i = 0; i < n8; i += 8) {
        __m128i u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256  b   = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u16), 16));
        __m256  a   = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }
    for (size_t i = n8; i < n; ++i) dst[i] += bf16_to_f32(src[i]);
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

// Hardware AVX2/F16C canonical left-to-right (((p0+p1)+p2)+p3) FP16 RNE AllReduce sum
inline void avx2_sum4_f16_canonical(const uint16_t* p0, const uint16_t* p1,
                                    const uint16_t* p2, const uint16_t* p3,
                                    uint16_t* dst, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    const size_t chunks = n8 >> 3;
    #pragma omp parallel for num_threads(nt) schedule(static) if (n8 >= 65536u)
    for (size_t c = 0; c < chunks; ++c) {
        const size_t i = c << 3;
        __m256 s = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + i))),
                              _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + i)))),
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p2 + i)))),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p3 + i))));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    for (size_t i = n8; i < n; ++i) {
        float s = fp16_to_f32(p0[i]);
        s += fp16_to_f32(p1[i]);
        s += fp16_to_f32(p2[i]);
        s += fp16_to_f32(p3[i]);
        dst[i] = f32_to_fp16_rne(s);
    }
}

// Wave-10 C2: Hardware AVX2/F16C canonical FP16->BF16 RNE AllReduce sum (0 GPU f16->bf16 bounce on o_proj!)
inline void avx2_sum4_f16_to_bf16_canonical(const uint16_t* p0, const uint16_t* p1,
                                            const uint16_t* p2, const uint16_t* p3,
                                            uint16_t* dst_bf16, size_t n, int nt) noexcept {
    const size_t n8 = n & ~size_t(7);
    const size_t chunks = n8 >> 3;
    const __m256i v_bias = _mm256_set1_epi32(0x7FFF);
    const __m256i v_one  = _mm256_set1_epi32(1);
    const __m256i v_shuf = _mm256_setr_epi8(
        2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1,
        2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
    #pragma omp parallel for num_threads(nt) schedule(static) if (n8 >= 65536u)
    for (size_t c = 0; c < chunks; ++c) {
        const size_t i = c << 3;
        __m256 s = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + i))),
                              _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + i)))),
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p2 + i)))),
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p3 + i))));
        __m256i u = _mm256_castps_si256(s);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), v_one);
        u = _mm256_add_epi32(u, _mm256_add_epi32(v_bias, lsb));
        __m256i p = _mm256_shuffle_epi8(u, v_shuf);
        __m128i out = _mm_unpacklo_epi64(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_bf16 + i), out);
    }
    for (size_t i = n8; i < n; ++i) {
        float s = fp16_to_f32(p0[i]);
        s += fp16_to_f32(p1[i]);
        s += fp16_to_f32(p2[i]);
        s += fp16_to_f32(p3[i]);
        dst_bf16[i] = f32_to_bf16_rne(s);
    }
}

// Exact FP64 Column-Pivoted Modified Gram-Schmidt factorization of A = vha_m - I (64x64, rank <= 16) -> Q[64,16] @ R[16,64]
static bool factor_vha_rank16(const float* vha_m_64x64, float* out_Q_64x16, float* out_R_16x64, double* out_max_err = nullptr) {
    double A[64][64], res[64][64];
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            const double v = double(vha_m_64x64[i * 64 + j]) - ((i == j) ? 1.0 : 0.0);
            A[i][j] = v;
            res[i][j] = v;
        }
    }
    double Q[64][16] = {};
    double R[16][64] = {};
    for (int k = 0; k < 16; ++k) {
        int best_j = 0;
        double best_n2 = -1.0;
        for (int j = 0; j < 64; ++j) {
            double n2 = 0.0;
            for (int i = 0; i < 64; ++i) n2 += res[i][j] * res[i][j];
            if (n2 > best_n2) { best_n2 = n2; best_j = j; }
        }
        const double nrm = std::sqrt(best_n2);
        if (nrm < 1e-12) break;
        for (int i = 0; i < 64; ++i) Q[i][k] = res[i][best_j] / nrm;
        for (int p = 0; p < k; ++p) {
            double dot = 0.0;
            for (int i = 0; i < 64; ++i) dot += Q[i][p] * Q[i][k];
            for (int i = 0; i < 64; ++i) Q[i][k] -= dot * Q[i][p];
        }
        double n2 = 0.0;
        for (int i = 0; i < 64; ++i) n2 += Q[i][k] * Q[i][k];
        const double inv_n = (n2 > 1e-24) ? (1.0 / std::sqrt(n2)) : 0.0;
        for (int i = 0; i < 64; ++i) Q[i][k] *= inv_n;
        for (int j = 0; j < 64; ++j) {
            double rkj = 0.0;
            for (int i = 0; i < 64; ++i) rkj += Q[i][k] * res[i][j];
            R[k][j] = rkj;
            for (int i = 0; i < 64; ++i) res[i][j] -= Q[i][k] * rkj;
        }
    }
    double max_err = 0.0;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            double rec = 0.0;
            for (int k = 0; k < 16; ++k) rec += Q[i][k] * R[k][j];
            max_err = std::max(max_err, std::fabs(A[i][j] - rec));
        }
    }
    if (out_max_err) *out_max_err = max_err;
    for (int i = 0; i < 64; ++i)
        for (int k = 0; k < 16; ++k)
            out_Q_64x16[i * 16 + k] = float(Q[i][k]);
    for (int k = 0; k < 16; ++k)
        for (int j = 0; j < 64; ++j)
            out_R_16x64[k * 64 + j] = float(R[k][j]);
    return (max_err < 1e-4);
}

struct PinAllocEntry {
    void* ptr = nullptr;
    bool  registered = false; // true: posix_memalign + hipHostRegisterPortable (WB cached DDR4!); false: hipHostMalloc
};

template <typename T>
T* host_pin_alloc(size_t count, std::vector<PinAllocEntry>& pin_allocs, bool zero = false,
                  bool mapped = false, T** dev_out = nullptr) {
    if (count == 0) return nullptr;
    const size_t bytes = count * sizeof(T);
    const size_t alloc_bytes = (bytes + 4095u) & ~size_t(4095u);
    const size_t align = (alloc_bytes >= 2097152u) ? 2097152u : 4096u;
    void* p = nullptr;
    if (posix_memalign(&p, align, alloc_bytes) == 0 && p != nullptr) {
        if (alloc_bytes >= 2097152u) (void)madvise(p, alloc_bytes, MADV_HUGEPAGE);
        // Fault in WB cached pages before hipHostRegisterPortable (0 VM_IO uncached penalty on CPU AVX2!)
        std::memset(p, 0, zero ? alloc_bytes : std::min(alloc_bytes, size_t(4096u)));
        if (!zero && alloc_bytes > 4096u) {
            volatile uint8_t* vp = static_cast<volatile uint8_t*>(p);
            for (size_t off = 4096u; off < alloc_bytes; off += 4096u) vp[off] = 0u;
        }
        // Wave-11 Turn-4 Cut B: `mapped` registers the same pinned pages with
        // hipHostRegisterMapped so gfx906 kernels can read them directly over PCIe (system-scope
        // coherent). This removes the host hipEventSynchronize + H2D leg of every cross-RC relay
        // without allocating a single new byte of GPU VRAM.
        const unsigned flags = hipHostRegisterPortable | (mapped ? hipHostRegisterMapped : 0u);
        if (hipHostRegister(p, alloc_bytes, flags) == hipSuccess) {
            if (mapped) {
                void* dp = nullptr;
                hip_check(hipHostGetDevicePointer(&dp, p, 0), "hipHostGetDevicePointer");
                if (dev_out != nullptr) *dev_out = static_cast<T*>(dp);
            }
            pin_allocs.push_back({p, true});
            return static_cast<T*>(p);
        }
        std::free(p);
        p = nullptr;
    }
    hip_check(hipHostMalloc(&p, bytes, hipHostMallocDefault), "hipHostMalloc");
    if (zero) std::memset(p, 0, bytes);
    pin_allocs.push_back({p, false});
    return static_cast<T*>(p);
}

template <typename T>
T* host_mapped_alloc(size_t count, std::vector<PinAllocEntry>& pin_allocs, bool zero = false) {
    if (count == 0) return nullptr;
    void* p = nullptr;
    hip_check(hipHostMalloc(&p, count * sizeof(T), hipHostMallocPortable | hipHostMallocMapped | hipHostMallocCoherent), "hipHostMallocMapped");
    if (zero) std::memset(p, 0, count * sizeof(T));
    pin_allocs.push_back({p, false});
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
    float*    d_vha_q_loc   = nullptr;  // [16, 16] F32 (Commit 4a rank-16 Q shard for TP=4)
    float*    d_vha_r_loc   = nullptr;  // [16, 16] F32 (Commit 4a rank-16 R shard for TP=4)
    bool      has_vha_lr    = false;
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
    float*    d_vha_q_loc   = nullptr;  // [16, 16] F32 (Commit 4a rank-16 Q shard for TP=4)
    float*    d_vha_r_loc   = nullptr;  // [16, 16] F32 (Commit 4a rank-16 R shard for TP=4)
    bool      has_vha_lr    = false;
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
    hipStream_t stream_sh = nullptr;     // secondary stream for concurrent shared_experts + async host-pinned SDMA!
    rocblas_handle blas_sh = nullptr;
    hipEvent_t ev_sh_start = nullptr;
    hipEvent_t ev_sh_done  = nullptr;
    hipEvent_t ev_mb_c2d[4]     = {nullptr, nullptr, nullptr, nullptr}; // Wave-11 C1: 4-MB compute->SDMA events (0 B VRAM)
    hipEvent_t ev_mb_d2c[4]     = {nullptr, nullptr, nullptr, nullptr}; // Wave-11 C1: 4-MB SDMA->compute events
    hipEvent_t ev_mb_vha_d2h[4] = {nullptr, nullptr, nullptr, nullptr}; // Wave-11 C1: 4-MB VHA D2H completion events
    hipEvent_t ev_mb_vha_h2d[4] = {nullptr, nullptr, nullptr, nullptr}; // Wave-11 C1: 4-MB VHA H2D completion events
    hipEvent_t ev_mb_o_d2h[4]   = {nullptr, nullptr, nullptr, nullptr}; // Wave-11 C1: o_proj D2H completion events
    hipEvent_t ev_res_b_ready   = nullptr; // Wave-11 C1: deferred non-owner o_proj + post_mhc + 160 KiB mix ready event
    hipStream_t keep_warm_stream = nullptr;
    uint32_t*   h_dpm_flag       = nullptr;
    uint32_t*   d_dpm_flag       = nullptr;
    uint32_t*   d_dpm_uc_scratch = nullptr;
    std::vector<void*> allocs;

    uint16_t* d_levels8_f16 = nullptr;
    std::array<uint64_t*, kTotalLayersWithEcho> d_signs_in{};
    std::array<uint64_t*, kTotalLayersWithEcho> d_signs_inter{};
    std::array<Hlut4Tile32x32*, kTotalLayersWithEcho> d_w1{};
    std::array<Hlut4Tile32x32*, kTotalLayersWithEcho> d_w3{};
    std::array<Hlut4Tile32x32*, kTotalLayersWithEcho> d_w2{};

    // MoE dispatch scratch on this GPU (up to max_chunk rows)
    uint16_t* d_x_rot_f16 = nullptr;        // [max_chunk, 2048] FP16
    ExpertDispatchGroup* d_groups = nullptr;// [max_groups] (decode M<=8)
    ExpertDispatchGroup* h_groups_pin = nullptr; // [max_groups] pinned H2D (decode M<=8)
    float*    d_z_f32     = nullptr;        // [max(max_chunk*10*2048/2, 80*16*2048)] F32 (80 MiB: aliases FP16 z_f16 on prefill!)
    uint16_t* d_z_rot_f16 = nullptr;        // [80 * 16, 2048] FP16 (5 MiB decode-only!)
    float*    d_y_slots_f32 = nullptr;      // [8, 10, 2048] F32 (0.625 MiB decode-only canonical slot buffer!)
    float*    d_y_acc_f32 = nullptr;        // [max_chunk, 2048] F32
    float*    h_y_partial_pin = nullptr;    // [max_chunk, 2048] F32 pinned!

    // Commit 2: Compact On-GPU Prefill MoE CSR Scratch (~0.5 MiB/GPU, zero HBM2 FP16 unpack!)
    int32_t*  d_moe_topk_ids = nullptr;     // [max_chunk * kTopKExperts] I32
    float*    d_moe_topk_wts = nullptr;     // [max_chunk * kTopKExperts] F32
    int16_t*  d_g2l_bb       = nullptr;     // [kNumRoutedExperts] I16 (global->local expert map)
    int32_t*  d_csr_counts   = nullptr;     // [4] I32 ([0]=n_tiles, [1]=n_pairs)
    Ernie55MoeWorkTile*   d_moe_tiles = nullptr; // [4096]
    Ernie55MoeRoutedPair* d_moe_pairs = nullptr; // [max_chunk * kTopKExperts]

    // Pre-allocated zero-heap flat MoE dispatch buckets (decode M<=8)
    std::array<int32_t, kNumRoutedExperts> b_head{};
    std::array<int32_t, kNumRoutedExperts> b_cnt{};
    std::vector<int32_t> b_next;
    std::vector<int32_t> b_rows;
    std::vector<int32_t> b_slots;
    std::vector<float>   b_wts;
    std::vector<int32_t> active_exps;
    std::vector<ExpertDispatchGroup> h_groups;

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
    // Mapped host-memory TP=4 & 5-MI50 MoE device pointers (zero-sync M<=8 collectives!)
    uint16_t* d_tp_mapped_slabs  = nullptr;
    uint32_t* d_tp_mapped_ready  = nullptr;
    uint32_t* d_tp_mapped_done   = nullptr;
    uint16_t* d_moe_mapped_slabs = nullptr;
    uint32_t* d_moe_mapped_ready = nullptr;
    uint32_t* d_moe_mapped_done  = nullptr;
    uint16_t* d_p150_y_bf16_mapped = nullptr;

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
    int vha_lr_count = 0;
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
    // Wave-10 Bug 4c: Relaxed layer-granularity prefill progress atomics (0 extra hipStreamSynchronize!)
    std::atomic<size_t> live_prefill_base_fresh{0};
    std::atomic<int>    live_prefill_chunk_m{0};
    std::atomic<int>    live_prefill_layer_done{0};
    std::atomic<int>    live_prefill_active{0};
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

    // Pinned host allocations (WB cached hugepages + hipHostRegisterPortable; freed cleanly in shutdown!)
    std::vector<PinAllocEntry> pin_allocs;
    int dpm_warm_depth = 0;
    std::array<bool, 8> dpm_warm_launched{};
    std::atomic<double> last_chunk_prefill_tok_s{0.0};
    std::atomic<double> last_prefill_tok_s{0.0};
    std::atomic<double> last_decode_tok_s{0.0};
    NgramSuffixIndex ngram_idx;
    uint64_t sa_hits = 0, sa_drafted = 0, sa_accepted = 0, dspark_skipped = 0;
    bool sa_cascade = true;
    Ernie55BlockKvCache kv_blk_cache;
    int last_ctx_read_tokens = 0;
    double last_ctx_read_ms = 0.0;
    double last_ctx_read_tok_s = 0.0;
    double peak_ctx_read_tok_s = 0.0;
    std::atomic<int> last_ctx_tier_id{0}; // 0=none, 1=VRAM, 2=DDR4, 3=NVMe_kvblk
    double chunk_hca_ms = 0.0;
    double chunk_hca_a_ms = 0.0;
    double chunk_hca_ar_ms = 0.0;
    double chunk_dsa_ms = 0.0;
    double chunk_dsa_fi_ms = 0.0;
    double chunk_dsa_mqa_ms = 0.0;
    double chunk_moe_ms = 0.0;
    double chunk_moe_pre_ms = 0.0;
    double chunk_moe_post_ms = 0.0;
    double chunk_p150_wait_ms = 0.0;
    double chunk_mi50_moe_ms = 0.0;
    double chunk_gpu_moe_ms[5] = {};
    // Wave-12 Cut P12.1: moe_pre sub-stage x-ray. Populated only when ERNIE55_MOE_PRE_XRAY=1;
    // every timer is a single steady_clock::now() pair with no added host sync.
    double chunk_pre_backlog_ms = 0.0;  // inherited HCA/DSA/o_proj/mhc GPU backlog
    double chunk_pre_router_ms  = 0.0;  // router GEMM + topk D2H sync
    double chunk_pre_pack_ms    = 0.0;  // CPU row_slots / P150A LUT pass
    double chunk_pre_fc1_ms     = 0.0;  // fc1_lat + WHT128 + D2H
    double chunk_pre_sh_ms      = 0.0;  // shared_experts launch on stream_sh
    bool moe_pre_xray = false;
    // Wave-12 Turn-15: backlog sub-stage decomposition (ERNIE55_MOE_PRE_XRAY=1 only).
    double chunk_bk_moe_post = 0.0;   // (a) prev-layer fc2_lat + add_sh + mhc_post + bcast
    double chunk_bk_attn_pre = 0.0;   // (b) attn mhc_pre
    double chunk_bk_qkv      = 0.0;   // (c) hca/dsa qkv + q_b
    double chunk_bk_attn     = 0.0;   // (d) hca/dsa attn (comp + swa / fi + mqa)
    double chunk_bk_vha_op   = 0.0;   // (e) vha + o_proj + tp4_ar
    double chunk_bk_attn_post= 0.0;   // (f) attn mhc_post + mlp mhc_pre
    void reset_backlog_xray() {
        chunk_bk_moe_post = chunk_bk_attn_pre = chunk_bk_qkv = 0.0;
        chunk_bk_attn = chunk_bk_vha_op = chunk_bk_attn_post = 0.0;
    }
    void init_moe_pre_xray() {
        const char* ev = std::getenv("ERNIE55_MOE_PRE_XRAY");
        moe_pre_xray = (ev != nullptr && ev[0] == '1' && ev[1] == '\0');
    }
    void reset_chunk_timers() {
        chunk_hca_ms = chunk_hca_a_ms = chunk_hca_ar_ms = 0.0;
        chunk_dsa_ms = chunk_dsa_fi_ms = chunk_dsa_mqa_ms = 0.0;
        chunk_moe_ms = chunk_moe_pre_ms = chunk_moe_post_ms = 0.0;
        chunk_p150_wait_ms = chunk_mi50_moe_ms = 0.0;
        chunk_pre_backlog_ms = chunk_pre_router_ms = chunk_pre_pack_ms = 0.0;
        chunk_pre_fc1_ms = chunk_pre_sh_ms = 0.0;
        reset_backlog_xray();
        for (int i = 0; i < 5; ++i) chunk_gpu_moe_ms[i] = 0.0;
    }

    void dpm_warm_enter(bool only_non_spine = false) {
        if (dpm_warm_depth++ > 0) return;
        dpm_warm_launched.fill(false);
        auto is_spine_dev = [&](int dev) -> bool {
            for (int r = 0; r < tp_size; ++r) {
                if (tp_devs[r] == dev) return true;
            }
            return false;
        };
        for (size_t gi = 0; gi < gpus.size() && gi < dpm_warm_launched.size(); ++gi) {
            auto& w = gpus[gi];
            if (!w || !w->h_dpm_flag || !w->d_dpm_flag || !w->d_dpm_uc_scratch || !w->keep_warm_stream) continue;
            if (only_non_spine && is_spine_dev(w->hip_dev)) continue;
            *w->h_dpm_flag = 1u;
            dpm_warm_launched[gi] = true;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (size_t gi = 0; gi < gpus.size() && gi < dpm_warm_launched.size(); ++gi) {
            if (!dpm_warm_launched[gi]) continue;
            auto& w = gpus[gi];
            (void)hipSetDevice(w->hip_dev);
            launch_ernie55_keep_dpm_warm(w->d_dpm_flag, w->d_dpm_uc_scratch, 16383u, w->keep_warm_stream);
        }
    }

    void dpm_warm_exit() {
        if (dpm_warm_depth <= 0) return;
        if (--dpm_warm_depth > 0) return;
        for (size_t gi = 0; gi < gpus.size() && gi < dpm_warm_launched.size(); ++gi) {
            auto& w = gpus[gi];
            if (dpm_warm_launched[gi] && w && w->h_dpm_flag) *w->h_dpm_flag = 0u;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (size_t gi = 0; gi < gpus.size() && gi < dpm_warm_launched.size(); ++gi) {
            auto& w = gpus[gi];
            if (dpm_warm_launched[gi] && w && w->keep_warm_stream) {
                (void)hipSetDevice(w->hip_dev);
                (void)hipStreamSynchronize(w->keep_warm_stream);
            }
            dpm_warm_launched[gi] = false;
        }
    }

    struct DpmWarmGuard {
        Ernie55Runtime::Impl* impl;
        explicit DpmWarmGuard(Ernie55Runtime::Impl* p, bool only_non_spine = false) : impl(p) {
            if (impl) impl->dpm_warm_enter(only_non_spine);
        }
        ~DpmWarmGuard() { if (impl) impl->dpm_warm_exit(); }
    };
    uint16_t* h_lat_rot_f16_pin   = nullptr;  // [max_chunk, 2048]
    uint16_t* h_lat_rot_bf16_pin  = nullptr;  // [max_chunk, 2048] native BF16 for zero-cv P150A SHM DMA!
    int32_t*  h_topk_ids_pin      = nullptr;  // [max_chunk, 10]
    float*    h_topk_wts_pin      = nullptr;  // [max_chunk, 10]
    float*    h_y_sum_f32_pin     = nullptr;  // [max_chunk, 2048]
    float*    h_p150_y_pin        = nullptr;  // [max_chunk, 2048] (for M > ERNIE55_P150_MAX_ROWS sliced P150A SHM!)
    // Turn-5 Cut A: active-row x_rot compaction state (host RAM only, 0 B GPU VRAM).
    std::vector<uint16_t> h_p150_y_full;      // [M, kLatentDim] scattered P150A output
    std::vector<int32_t>  p150_row_map;       // compact row i -> original row
    int p150_active_rows = 0;
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
    uint16_t* d_tp_ar_in_pin[LayerSpineGpu::kMaxTp]  = {}; // Turn-4 Cut B: device-side aliases (mapped)
    uint16_t* h_tp_ar_out_pin   = nullptr;                 // [max_chunk, 4096] BF16 canonical sum (MoE post + fallback AR)
    uint16_t* h_tp_o_out_pin    = nullptr;                 // [max_chunk, 4096] BF16 Wave-11 C1 dedicated o_proj sum (zero race with moe_post!)
    bool      o_proj_host_d2h_pending = false;             // Wave-11 C5: g_own.ev_mb_vha_d2h[3] signals h_tp_o_out_pin ready for deferred non-owners
    float*    h_mhc_mix_pin     = nullptr;                 // [max_chunk, 20] F32 Wave-11 C1 owner->non-owner (d_post_mix[4] + d_comb_mix[16] = 160 KiB)
    uint16_t* h_tp_oattn_pin    = nullptr;                 // [tp_size, max_chunk, h_loc, 512] BF16 rank-major contiguous
    uint16_t* d_tp_oattn_pin    = nullptr;                 // Turn-4 Cut B: device-side alias of h_tp_oattn_pin (mapped)
    uint16_t* h_tp_qabs_pin     = nullptr;                 // [max_chunk, 64, 576] BF16 token-major DSA q_abs
    uint16_t* d_tp_qabs_pin     = nullptr;                 // Turn-4 Cut B: device-side alias of h_tp_qabs_pin (mapped)
    uint16_t* h_tp_qabs_rm_pin  = nullptr;                 // [tp_size, max_chunk, 16, 576] BF16 rank-major contiguous
    int32_t*  h_dsa_sel_cnts_pin = nullptr;                // [max_chunk] I32 dedicated Commit 3 DSA sel_cnts host-SDMA staging (0 aliasing!)
    float*    h_tp_rtr_in_pin[LayerSpineGpu::kMaxTp] = {}; // [max_chunk, 256] F32 contiguous per rank
    uint16_t* h_tp_fc1_in_pin[LayerSpineGpu::kMaxTp] = {}; // [max_chunk, 512] BF16 contiguous per rank
    uint16_t* h_tp_fc2_in_pin[LayerSpineGpu::kMaxTp] = {}; // [max_chunk, 512] BF16 contiguous per rank
    // Wave-11 Cut D3.1/D3.2/D3.5: Deep kPingPong=16 Monotonic Mapped Host-Memory TP=4 & 5-MI50+P150A MoE Rings (0 B GPU VRAM!)
    static constexpr int kMappedPingPong   = 16;
    static constexpr int kMappedSlabStride = 8 * 16 * 576; // 73,728 halfs (144 KiB/rank/slot, covers DSA q_abs [M,16,576]!)
    static constexpr int kMoeMappedStride  = 8 * 2048 + 512; // 16,896 halfs (33 KiB/GPU/slot, 16B aligned)
    uint16_t* h_tp_mapped_slabs  = nullptr;
    uint32_t* h_tp_mapped_ready  = nullptr;
    uint32_t* h_tp_mapped_done   = nullptr;
    uint32_t  tp_mapped_epoch    = 0u;
    uint16_t* h_moe_mapped_slabs = nullptr; // 16 slots x 7 slabs/slot (0..4=MI50, 5=xrot/topk, 6=P150A BF16 + has_work)
    uint32_t* h_moe_mapped_ready = nullptr;
    uint32_t* h_moe_mapped_done  = nullptr;
    uint32_t  moe_mapped_epoch   = 0u;

    // Dedicated zero-blocking background P150A decode relay thread (eliminates all 84 host spins/step!)
    struct P150DecodeJob {
        uint32_t epoch      = 0u;
        int32_t  layer      = 0;
        int32_t  M          = 0;
        bool     use_bb_lut = true;
    };
    static constexpr uint32_t kP150RelayQSize = 64u;
    P150DecodeJob           p150_relay_q[kP150RelayQSize]{};
    std::atomic<uint64_t>   p150_relay_head{0u};
    std::atomic<uint64_t>   p150_relay_tail{0u};
    std::atomic<bool>       p150_relay_stop{false};
    std::atomic<bool>       p150_relay_sleeping{false};
    std::atomic<int>        p150_relay_err{0};
    std::mutex              p150_relay_mu;
    std::condition_variable p150_relay_cv;
    std::thread             p150_relay_thr;

    void start_p150_relay() {
        if (p150_relay_thr.joinable()) return;
        p150_relay_head.store(0u, std::memory_order_relaxed);
        p150_relay_tail.store(0u, std::memory_order_relaxed);
        p150_relay_stop.store(false, std::memory_order_relaxed);
        p150_relay_sleeping.store(false, std::memory_order_relaxed);
        p150_relay_err.store(0, std::memory_order_relaxed);
        p150_relay_thr = std::thread([this]() { this->p150_relay_loop(); });
    }

    void stop_p150_relay() {
        if (!p150_relay_thr.joinable()) return;
        p150_relay_stop.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(p150_relay_mu);
            p150_relay_cv.notify_all();
        }
        p150_relay_thr.join();
    }

    void drain_p150_decode_relay() {
        if (!p150_relay_thr.joinable()) return;
        const uint64_t target = p150_relay_head.load(std::memory_order_relaxed);
        while (p150_relay_tail.load(std::memory_order_acquire) != target) {
            p150_cpu_relax();
        }
        const int err = p150_relay_err.exchange(0, std::memory_order_acq_rel);
        if (err != 0) {
            throw std::runtime_error("Blackhole P150A relay worker returned error status=" + std::to_string(err));
        }
    }

    void submit_p150_decode_job(uint32_t ep, int32_t layer, int32_t M, bool use_bb_lut) {
        const uint64_t head = p150_relay_head.load(std::memory_order_relaxed);
        while (head - p150_relay_tail.load(std::memory_order_acquire) >= kP150RelayQSize) {
            p150_cpu_relax();
        }
        p150_relay_q[head & (kP150RelayQSize - 1u)] = P150DecodeJob{ep, layer, M, use_bb_lut};
        p150_relay_head.store(head + 1u, std::memory_order_release);
        if (p150_relay_sleeping.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(p150_relay_mu);
            p150_relay_cv.notify_one();
        }
    }

    void p150_relay_loop() {
        uint64_t spin = 0u;
        while (true) {
            const uint64_t tail = p150_relay_tail.load(std::memory_order_relaxed);
            if (tail == p150_relay_head.load(std::memory_order_acquire)) {
                if (p150_relay_stop.load(std::memory_order_acquire)) break;
                if (++spin < 4000u) {
                    p150_cpu_relax();
                } else {
                    std::unique_lock<std::mutex> lk(p150_relay_mu);
                    p150_relay_sleeping.store(true, std::memory_order_release);
                    p150_relay_cv.wait(lk, [&]() {
                        return p150_relay_stop.load(std::memory_order_acquire) ||
                               p150_relay_tail.load(std::memory_order_relaxed) != p150_relay_head.load(std::memory_order_acquire);
                    });
                    p150_relay_sleeping.store(false, std::memory_order_relaxed);
                    spin = 0u;
                }
                continue;
            }
            spin = 0u;
            const P150DecodeJob job = p150_relay_q[tail & (kP150RelayQSize - 1u)];
            const uint32_t ep   = job.epoch;
            const uint32_t slot = (ep - 1u) & 15u;
            const int M         = int(job.M);
            const int l         = int(job.layer);

            const uint32_t* rdy5 = &h_moe_mapped_ready[5 * 16];
            while (__atomic_load_n(rdy5, __ATOMIC_ACQUIRE) < ep) {
                if (p150_relay_stop.load(std::memory_order_acquire)) break;
                p150_cpu_relax();
            }

            uint16_t* s6_host = h_moe_mapped_slabs + (size_t(slot) * 7u + 6u) * size_t(kMoeMappedStride);
            auto* ch = p150_shm.ptr;
            bool p150_has_work = false;
            if (ch != nullptr && !p150_relay_stop.load(std::memory_order_acquire)) {
                const uint16_t* s5_host = h_moe_mapped_slabs + (size_t(slot) * 7u + 5u) * size_t(kMoeMappedStride);
                const int32_t* ids_h = reinterpret_cast<const int32_t*>(s5_host + 16384u);
                const float*   wts_h = reinterpret_cast<const float*>(s5_host + 16544u);
                const int16_t* lut = p150_slot_lut_bb.data();
                for (int r = 0; r < M; ++r) {
                    for (int s = 0; s < kTopKExperts; ++s) {
                        const int32_t eid = ids_h[r * kTopKExperts + s];
                        const float   wt  = wts_h[r * kTopKExperts + s];
                        int32_t pslot = -1;
                        if (wt != 0.0f && uint32_t(eid) < uint32_t(kNumRoutedExperts)) {
                            if (job.use_bb_lut) {
                                pslot = int32_t(lut[eid]);
                            } else {
                                const int cg = int(bb_owner_gpu[size_t(eid)]);
                                const int cl = (cg >= 0) ? int(bb_local_slot[size_t(eid)]) : -1;
                                const bool mi50_owns = (cg >= 0) && (cl < gpus[size_t(cg)]->bb_base_count);
                                if (!mi50_owns && eid >= p150_exp_begin && eid < p150_exp_end) pslot = eid - p150_exp_begin;
                            }
                        }
                        if (pslot >= 0) {
                            ch->row_slots[r][s]   = pslot;
                            ch->row_weights[r][s] = wt;
                            p150_has_work = true;
                        } else {
                            ch->row_slots[r][s]   = -1;
                            ch->row_weights[r][s] = 0.0f;
                        }
                    }
                }
                if (p150_has_work) {
                    const size_t n_lat = size_t(M) * kLatentDim;
                    avx2_f16_to_f32(s5_host, ch->x_rot_f32, n_lat, 1);
                    ch->flags     = 0u;
                    ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                    ch->layer_idx = uint32_t(l);
                    ch->rows      = uint32_t(M);
                    ch->top_k     = kTopKExperts;
                    const uint32_t seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                    ch->req_seq.store(seq, std::memory_order_release);
                    while (ch->resp_seq.load(std::memory_order_acquire) != seq) p150_cpu_relax();
                    if (ch->status != 0) {
                        p150_relay_err.store(int(ch->status), std::memory_order_release);
                        p150_has_work = false;
                    } else {
                        for (size_t i = 0; i < n_lat; ++i) s6_host[i] = f32_to_bf16_rne(ch->y_out_f32[i]);
                    }
                }
            }
            s6_host[16384u] = p150_has_work ? 1u : 0u;
            std::atomic_thread_fence(std::memory_order_release);
            __atomic_store_n(&h_moe_mapped_ready[6 * 16], ep, __ATOMIC_RELEASE);
            p150_relay_tail.store(tail + 1u, std::memory_order_release);
        }
    }
    double    dspark_alpha_ema   = 0.58; // 16-step EMA for adaptive K in [4..7]
    uint16_t* d_dspark_markov_w1_T_mapped = nullptr;
    bool      h_dspark_markov_reg         = false;
    // Wave-10 C6: Contiguous GPU Sparse Softmax Packet (1 single D2H + 1 sync for n_cand<=128, -817 KiB VRAM vs Wave-8!)
    static constexpr int kGpuMaxCands = 4096;
    static constexpr int kInlineCands = 128;
    struct DsparkSparsePacket {
        DsparkSparsePacketHeader hdr;
        DsparkCandPair           cands[kGpuMaxCands];
    };
    DsparkSparsePacket* d_dspark_sm_pack     = nullptr;
    DsparkSparsePacket* h_dspark_sm_pack_pin = nullptr;
    uint16_t*           h_dspark_me_pin      = nullptr; // [256] BF16 pinned Markov lookup row (0 B GPU VRAM)

    std::vector<std::unique_ptr<Mi50ExpertWorker>> gpus;
    std::unordered_map<int, Mi50ExpertWorker*> dev_map;

    // Shared root weights
    uint16_t* d_embed_bf16    = nullptr;  // on spine_dev0
    std::vector<uint16_t> h_embed_mirror; // [V, 4096] BF16 on host RAM (0 B GPU VRAM, eliminates g0 D2H/sync!)
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
    uint16_t* d_dspark_markov_w2   = nullptr; // [201216, 256] FP16 on spine_dev1 (fused fdot2 Markov+MultiMax!)
    uint16_t* d_dspark_me_bf16     = nullptr; // [256] BF16
    float*    d_dspark_step_logits = nullptr; // [201216] F32
    std::vector<uint16_t> h_dspark_markov_w1_T; // [201216, 256] BF16 on host RAM
    std::vector<DSparkLayerGpu> dspark_layers;

    DraftMethod draft_method = DraftMethod::kNone;

    // Blackhole P150A SHM IPC
    Ernie55P150ShmMapping p150_shm;
    int p150_exp_begin = 512;
    int p150_exp_end   = 512;
    bool p150_ok = false;
    bool p150_shm_x_reg = false;
    bool p150_shm_y_reg = false;
    std::array<int16_t, kNumRoutedExperts> p150_slot_lut_bb{};
    std::array<int16_t, kNumRoutedExperts> p150_slot_lut_bb_prefill{}; // Commit 4: +GPU[2] shadow experts [bb_base_count..bb_count) for M>=256

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
    //      Dynamic-strip 26.8 TFLOPS Type_HH_HPA (FP16 C/D, FP32 accumulate) for M>8! ----
    static void f16_linear_bf16_ex(rocblas_handle blas, hipStream_t stream,
                                   uint16_t* d_x_f16, float* d_y_f32,
                                   const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                   uint16_t* d_y_bf16, int M, int N, int K, int pad_M = 512,
                                   size_t x_cap_half = 512u * 16384u, size_t y_cap_half = 512u * 16384u * 2u) {
        if (M <= 8) {
            launch_bf16x_f16w_gemv_bf16(d_x_bf16, d_w_f16, d_y_bf16, M, N, K, stream);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32);
            const size_t cap_m = std::min(x_cap_half / size_t(K), y_cap_half / size_t(N));
            const int m_step = std::max(64, std::min(M, (int(cap_m) >= M) ? M : ((int(cap_m) / 64) * 64)));
            (void)pad_M;
            for (int m0 = 0; m0 < M; m0 += m_step) {
                const int cur = std::min(m_step, M - m0);
                const int eff_m = std::min(m_step, ((cur + 63) / 64) * 64);
                launch_bf16_to_f16(d_x_bf16 + size_t(m0) * size_t(K), d_x_f16, cur * K, stream);
                if (cur < eff_m) {
                    hip_check(hipMemsetAsync(d_x_f16 + size_t(cur) * size_t(K), 0,
                                             size_t(eff_m - cur) * size_t(K) * sizeof(uint16_t), stream), "zero f16 pad");
                }
                rocblas_gemm_ex_tuned(
                    blas, rocblas_operation_transpose, rocblas_operation_none,
                    N, eff_m, K, &alpha,
                    d_w_f16, rocblas_datatype_f16_r, K,
                    d_x_f16, rocblas_datatype_f16_r, K,
                    &beta,
                    d_y_f16, rocblas_datatype_f16_r, N,
                    d_y_f16, rocblas_datatype_f16_r, N,
                    "f16_linear_bf16 hh_hpa");
                launch_f16_to_bf16(d_y_f16, d_y_bf16 + size_t(m0) * size_t(N), cur * N, stream);
            }
        }
    }

    static void f16_linear_f32_ex(rocblas_handle blas, hipStream_t stream,
                                  uint16_t* d_x_f16,
                                  const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                  float* d_y_f32, int M, int N, int K, int pad_M = 512,
                                  int max_rows = 0, size_t x_cap_half = 512u * 16384u,
                                  uint16_t* d_y_tmp_f16 = nullptr, size_t y_tmp_cap_half = 0u) {
        if (M <= 8) {
            launch_bf16x_f16w_gemv_f32(d_x_bf16, d_w_f16, d_y_f32, M, N, K, stream);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            const bool do_hh = (d_y_tmp_f16 != nullptr && y_tmp_cap_half >= size_t(64) * size_t(N));
            size_t cap_m = x_cap_half / size_t(K);
            if (do_hh) cap_m = std::min(cap_m, y_tmp_cap_half / size_t(N));
            const int lim_m = (max_rows > 0) ? std::min(M, max_rows) : M;
            const int m_step = std::max(64, std::min(lim_m, (int(cap_m) >= lim_m) ? lim_m : ((int(cap_m) / 64) * 64)));
            (void)pad_M;
            for (int m0 = 0; m0 < M; m0 += m_step) {
                const int cur = std::min(m_step, M - m0);
                int eff_m = std::min(m_step, ((cur + 63) / 64) * 64);
                if (!do_hh && max_rows > 0 && m0 + eff_m > max_rows) eff_m = std::max(cur, max_rows - m0);
                launch_bf16_to_f16(d_x_bf16 + size_t(m0) * size_t(K), d_x_f16, cur * K, stream);
                if (cur < eff_m) {
                    hip_check(hipMemsetAsync(d_x_f16 + size_t(cur) * size_t(K), 0,
                                             size_t(eff_m - cur) * size_t(K) * sizeof(uint16_t), stream), "zero f16 pad");
                }
                float* dst_strip = d_y_f32 + size_t(m0) * size_t(N);
                if (do_hh) {
                    rocblas_gemm_ex_tuned(
                        blas, rocblas_operation_transpose, rocblas_operation_none,
                        N, eff_m, K, &alpha,
                        d_w_f16, rocblas_datatype_f16_r, K,
                        d_x_f16, rocblas_datatype_f16_r, K,
                        &beta,
                        d_y_tmp_f16, rocblas_datatype_f16_r, N,
                        d_y_tmp_f16, rocblas_datatype_f16_r, N,
                        "f16_linear_f32 hh_hpa");
                    launch_f16_to_f32(d_y_tmp_f16, dst_strip, cur * N, stream);
                } else {
                    rocblas_gemm_ex_tuned(
                        blas, rocblas_operation_transpose, rocblas_operation_none,
                        N, eff_m, K, &alpha,
                        d_w_f16, rocblas_datatype_f16_r, K,
                        d_x_f16, rocblas_datatype_f16_r, K,
                        &beta,
                        dst_strip, rocblas_datatype_f32_r, N,
                        dst_strip, rocblas_datatype_f32_r, N,
                        "f16_linear_f32 gemm_ex");
                }
            }
        }
    }

    static void bf16_linear_bf16(Mi50ExpertWorker& g, const uint16_t* d_x, const uint16_t* d_w_f16,
                                 uint16_t* d_y, int M, int N, int K) {
        f16_linear_bf16_ex(g.blas, g.stream, g.d_x_f16, g.d_y_f32, d_x, d_w_f16, d_y, M, N, K, 512,
                           512u * 16384u, 512u * 16384u * 2u);
    }

    // Wave-10 C2: Zero-bounce BF16->FP16 Type_HH_HPA (writes directly to d_y_f16 when 64-aligned, 0 f16->bf16!)
    static void bf16_linear_f16(Mi50ExpertWorker& g, const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                uint16_t* d_y_f16, int M, int N, int K) {
        const float alpha = 1.0f, beta = 0.0f;
        uint16_t* d_y_tmp = reinterpret_cast<uint16_t*>(g.d_y_f32);
        const size_t x_cap_half = 512u * 16384u;
        const size_t y_cap_half = 512u * 16384u * 2u;
        const size_t cap_m = std::min(x_cap_half / size_t(K), y_cap_half / size_t(N));
        const int m_step = std::max(64, std::min(M, (int(cap_m) >= M) ? M : ((int(cap_m) / 64) * 64)));
        for (int m0 = 0; m0 < M; m0 += m_step) {
            const int cur = std::min(m_step, M - m0);
            const int eff_m = std::min(m_step, ((cur + 63) / 64) * 64);
            launch_bf16_to_f16(d_x_bf16 + size_t(m0) * size_t(K), g.d_x_f16, cur * K, g.stream);
            if (cur < eff_m) {
                hip_check(hipMemsetAsync(g.d_x_f16 + size_t(cur) * size_t(K), 0,
                                         size_t(eff_m - cur) * size_t(K) * sizeof(uint16_t), g.stream), "zero bf16_f16 pad");
            }
            uint16_t* dst = (cur == eff_m) ? (d_y_f16 + size_t(m0) * size_t(N)) : d_y_tmp;
            rocblas_gemm_ex_tuned(
                g.blas, rocblas_operation_transpose, rocblas_operation_none,
                N, eff_m, K, &alpha,
                d_w_f16, rocblas_datatype_f16_r, K,
                g.d_x_f16, rocblas_datatype_f16_r, K,
                &beta,
                dst, rocblas_datatype_f16_r, N,
                dst, rocblas_datatype_f16_r, N,
                "bf16_linear_f16 hh_hpa");
            if (cur != eff_m) {
                hip_check(hipMemcpyAsync(d_y_f16 + size_t(m0) * size_t(N), d_y_tmp,
                                         size_t(cur) * size_t(N) * sizeof(uint16_t), hipMemcpyDeviceToDevice, g.stream), "d2d f16 tail");
            }
        }
    }

    // Wave-10 C2: Direct 0-conversion FP16->FP16 Type_HH_HPA (1 single rocBLAS call when M%64==0!)
    static void f16_linear_f16(Mi50ExpertWorker& g, const uint16_t* d_x_f16_in, const uint16_t* d_w_f16,
                               uint16_t* d_y_f16_out, int M, int N, int K) {
        const float alpha = 1.0f, beta = 0.0f;
        if ((M & 63) == 0) {
            rocblas_gemm_ex_tuned(
                g.blas, rocblas_operation_transpose, rocblas_operation_none,
                N, M, K, &alpha,
                d_w_f16, rocblas_datatype_f16_r, K,
                d_x_f16_in, rocblas_datatype_f16_r, K,
                &beta,
                d_y_f16_out, rocblas_datatype_f16_r, N,
                d_y_f16_out, rocblas_datatype_f16_r, N,
                "f16_linear_f16 hh_hpa");
            return;
        }
        const int m_aligned = M & ~63;
        if (m_aligned > 0) {
            rocblas_gemm_ex_tuned(
                g.blas, rocblas_operation_transpose, rocblas_operation_none,
                N, m_aligned, K, &alpha,
                d_w_f16, rocblas_datatype_f16_r, K,
                d_x_f16_in, rocblas_datatype_f16_r, K,
                &beta,
                d_y_f16_out, rocblas_datatype_f16_r, N,
                d_y_f16_out, rocblas_datatype_f16_r, N,
                "f16_linear_f16 hh_hpa aligned");
        }
        const int tail = M - m_aligned;
        const int eff_m = ((tail + 63) / 64) * 64;
        uint16_t* d_y_tmp = reinterpret_cast<uint16_t*>(g.d_y_f32);
        hip_check(hipMemcpyAsync(g.d_x_f16, d_x_f16_in + size_t(m_aligned) * size_t(K),
                                 size_t(tail) * size_t(K) * sizeof(uint16_t), hipMemcpyDeviceToDevice, g.stream), "d2d f16x tail");
        hip_check(hipMemsetAsync(g.d_x_f16 + size_t(tail) * size_t(K), 0,
                                 size_t(eff_m - tail) * size_t(K) * sizeof(uint16_t), g.stream), "zero f16x tail");
        rocblas_gemm_ex_tuned(
            g.blas, rocblas_operation_transpose, rocblas_operation_none,
            N, eff_m, K, &alpha,
            d_w_f16, rocblas_datatype_f16_r, K,
            g.d_x_f16, rocblas_datatype_f16_r, K,
            &beta,
            d_y_tmp, rocblas_datatype_f16_r, N,
            d_y_tmp, rocblas_datatype_f16_r, N,
            "f16_linear_f16 hh_hpa tail");
        hip_check(hipMemcpyAsync(d_y_f16_out + size_t(m_aligned) * size_t(N), d_y_tmp,
                                 size_t(tail) * size_t(N) * sizeof(uint16_t), hipMemcpyDeviceToDevice, g.stream), "d2d f16y tail");
    }

    void bf16_linear_f32(Mi50ExpertWorker& g, const uint16_t* d_x, const uint16_t* d_w_f16,
                         float* d_y, int M, int N, int K,
                         uint16_t* d_y_tmp_f16 = nullptr, size_t y_tmp_cap_half = 0u) const {
        f16_linear_f32_ex(g.blas, g.stream, g.d_x_f16, d_x, d_w_f16, d_y, M, N, K, 512,
                          max_chunk, 512u * 16384u, d_y_tmp_f16, y_tmp_cap_half);
    }

    // Helper: identify the high-capacity MI50 straggler GPU (GPU[2], 32 GiB) for prefill-only shadow offload (1:1 partition, 0 B VRAM)
    bool is_prefill_shadow_offload_gpu(int gi) const {
        if (gi < 0 || size_t(gi) >= gpus.size()) return false;
        int max_base = 0;
        for (const auto& gw : gpus) max_base = std::max(max_base, gw->bb_base_count);
        return gpus[size_t(gi)]->bb_base_count == max_base &&
               gpus[size_t(gi)]->bb_count() > gpus[size_t(gi)]->bb_base_count;
    }

    // ---- Fast Multi-CU FP16 fdot2 mHC Pre-Norm Helper (96-block split-K for M<=8, Wave-11 C3 LDS+SS+DualOut for M>8!) ----
    void run_mhc_pre_norm_fast(Mi50ExpertWorker& g, const uint16_t* d_res,
                               const MhcGpuWeights& mhc, const uint16_t* d_norm_w, int M,
                               bool write_x_f16 = false) {
        if (M <= 8) {
            launch_mhc_pre_norm_fast_f16(d_res, mhc.d_fn_24x16384_f16, mhc.d_scale3, mhc.d_base24, d_norm_w,
                                         g.d_mhc_ws, g.d_post_mix, g.d_comb_mix, g.d_x_in,
                                         M, kHiddenSize, mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
        } else {
            launch_mhc_pre_gemm_ss_bf16_f16_f32(d_res, mhc.d_fn_24x16384_f16, g.d_mhc_ws, M, kHiddenSize, g.stream);
            uint16_t* d_xf16 = (write_x_f16 && M <= max_chunk && M <= 2048) ? g.d_x_f16 : nullptr;
            launch_mhc_pre_sinkhorn_from_mixes_ss(
                d_res, g.d_mhc_ws, mhc.d_scale3, mhc.d_base24, d_norm_w,
                g.d_post_mix, g.d_comb_mix, g.d_x_in, d_xf16,
                M, kHiddenSize, mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
        }
    }

    // ---- Wave-11 Cut D3.1: Deep kPingPong=16 Zero-Host-Sync Mapped TP=4 AllReduce (M <= 8) + SDMA Host-Staged (M > 8) ----
    void tp_allreduce_half(int M, int cols,
                           uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m),
                           bool is_f16, bool use_vha_buf) {
        if (tp_size <= 1) return;
        const auto t_c0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const size_t n = size_t(M) * size_t(cols);
        if (M <= 8 && tp_size == 4 && h_tp_mapped_slabs != nullptr &&
            n <= size_t(kMappedSlabStride) && gfx906_mapped_tp4_enabled()) {
            const uint32_t ep = ++tp_mapped_epoch;
            for (int r = 0; r < 4; ++r) {
                auto& w = *dev_map[tp_devs[r]];
                hip_check(hipSetDevice(w.hip_dev), "ar map dev");
                if (is_f16) {
                    launch_tp4_mapped_allreduce_f16(
                        w.*src_m, w.*dst_m, w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                        r, int(n), kMappedSlabStride, ep, w.stream);
                } else {
                    launch_tp4_mapped_allreduce_bf16(
                        w.*src_m, w.*dst_m, w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                        r, int(n), kMappedSlabStride, ep, w.stream);
                }
            }
            if (profiling_enabled) {
                prof_sync_tp(tp_size, false, spine_dev0);
                prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0).count();
            }
            return;
        }
        const size_t nb = n * sizeof(uint16_t);
        const bool big = use_vha_buf || (n > size_t(max_chunk) * kHiddenSize);
        uint16_t* in_pin[LayerSpineGpu::kMaxTp];
        for (int r = 0; r < tp_size; ++r) {
            in_pin[r] = big ? (h_tp_oattn_pin + size_t(r) * n) : h_tp_ar_in_pin[r];
        }
        uint16_t* dst = big ? h_tp_qabs_pin : h_tp_ar_out_pin;

        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar d2h dev");
            hip_check(hipMemcpyAsync(in_pin[r], w.*src_m, nb, hipMemcpyDeviceToHost, w.stream), "ar d2h");
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar sync dev");
            hip_check(hipStreamSynchronize(w.stream), "ar sync");
        }
        if (tp_size == 4) {
            if (is_f16) avx2_sum4_f16_canonical(in_pin[0], in_pin[1], in_pin[2], in_pin[3], dst, n, kOmpThreads);
            else        avx2_sum4_bf16_canonical(in_pin[0], in_pin[1], in_pin[2], in_pin[3], dst, n, kOmpThreads);
        } else {
            #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (n >= 65536u)
            for (size_t i = 0; i < n; ++i) {
                float s = 0.0f;
                for (int r = 0; r < tp_size; ++r) s += is_f16 ? fp16_to_f32(in_pin[r][i]) : bf16_to_f32(in_pin[r][i]);
                dst[i] = is_f16 ? f32_to_fp16_rne(s) : f32_to_bf16_rne(s);
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

    void tp_allreduce_bf16(int M, int H, uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m)) {
        tp_allreduce_half(M, H, src_m, dst_m, false, false);
    }

    void tp_allreduce_f16(int M, int cols, uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m)) {
        tp_allreduce_half(M, cols, src_m, dst_m, true, true);
    }

    // Wave-10 C2 / Wave-11 D3.1: Mapped kPingPong=16 FP16->BF16 AR on M<=8, AVX2 host-staged on M>8!
    void tp_allreduce_f16_to_bf16(int M, int cols, uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m)) {
        if (tp_size <= 1) return;
        const auto t_c0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const size_t n = size_t(M) * size_t(cols);
        if (M <= 8 && tp_size == 4 && h_tp_mapped_slabs != nullptr &&
            n <= size_t(kMappedSlabStride) && gfx906_mapped_tp4_enabled()) {
            const uint32_t ep = ++tp_mapped_epoch;
            for (int r = 0; r < 4; ++r) {
                auto& w = *dev_map[tp_devs[r]];
                hip_check(hipSetDevice(w.hip_dev), "ar f16b map dev");
                launch_tp4_mapped_allreduce_f16_to_bf16(
                    w.*src_m, w.*dst_m, w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                    r, int(n), kMappedSlabStride, ep, w.stream);
            }
            if (profiling_enabled) {
                prof_sync_tp(tp_size, false, spine_dev0);
                prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0).count();
            }
            return;
        }
        const size_t nb = n * sizeof(uint16_t);
        const bool big = (n > size_t(max_chunk) * kHiddenSize);
        uint16_t* in_pin[LayerSpineGpu::kMaxTp];
        for (int r = 0; r < tp_size; ++r) {
            in_pin[r] = big ? (h_tp_oattn_pin + size_t(r) * n) : h_tp_ar_in_pin[r];
        }
        uint16_t* dst = big ? h_tp_qabs_pin : h_tp_ar_out_pin;

        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar f16b d2h dev");
            hip_check(hipMemcpyAsync(in_pin[r], w.*src_m, nb, hipMemcpyDeviceToHost, w.stream), "ar f16b d2h");
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar f16b sync dev");
            hip_check(hipStreamSynchronize(w.stream), "ar f16b sync");
        }
        if (tp_size == 4) {
            avx2_sum4_f16_to_bf16_canonical(in_pin[0], in_pin[1], in_pin[2], in_pin[3], dst, n, kOmpThreads);
        } else {
            #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (n >= 65536u)
            for (size_t i = 0; i < n; ++i) {
                float s = 0.0f;
                for (int r = 0; r < tp_size; ++r) s += fp16_to_f32(in_pin[r][i]);
                dst[i] = f32_to_bf16_rne(s);
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            auto& w = *dev_map[tp_devs[r]];
            hip_check(hipSetDevice(w.hip_dev), "ar f16b h2d dev");
            hip_check(hipMemcpyAsync(w.*dst_m, dst, nb, hipMemcpyHostToDevice, w.stream), "ar f16b h2d");
        }
        if (profiling_enabled) {
            prof_sync_tp(tp_size, false, spine_dev0);
            prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0).count();
        }
    }

    // ---- 100% 1D Contiguous Rank-Major Head AllGather (Mapped kPingPong=16 on M<=8, Self D2D + 3-peer SDMA on M>8) ----
    void tp_allgather_heads_bf16(int M, int total_heads, int head_dim,
                                 uint16_t* (Mi50ExpertWorker::*src_m), uint16_t* (Mi50ExpertWorker::*dst_m),
                                 uint16_t* host_pin_buf) {
        if (tp_size <= 1) return;
        const auto t_c0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int h_loc = total_heads / tp_size;
        const size_t slab_elems = size_t(M) * size_t(h_loc) * size_t(head_dim);
        if (M <= 8 && tp_size == 4 && h_tp_mapped_slabs != nullptr &&
            slab_elems <= size_t(kMappedSlabStride) && gfx906_mapped_tp4_enabled()) {
            const uint32_t ep = ++tp_mapped_epoch;
            for (int r = 0; r < 4; ++r) {
                auto& w = *dev_map[tp_devs[r]];
                hip_check(hipSetDevice(w.hip_dev), "ag map dev");
                launch_tp4_mapped_allgather_heads_bf16(
                    w.*src_m, w.*dst_m, w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                    r, int(slab_elems), kMappedSlabStride, ep, w.stream);
            }
            if (profiling_enabled) {
                prof_sync_tp(tp_size, false, spine_dev0);
                prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0).count();
            }
            return;
        }
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
        const int n_g = int(gpus.size());
        constexpr int kOrdLmLaunch5[5] = {2, 4, 3, 0, 1};
        constexpr int kOrdLmSync5[5]   = {1, 0, 3, 4, 2};
        for (int oi = 0; oi < n_g; ++oi) {
            const int gi = (n_g == 5) ? kOrdLmLaunch5[oi] : oi;
            auto& w = *gpus[size_t(gi)];
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
        for (int oi = 0; oi < n_g; ++oi) {
            const int gi = (n_g == 5) ? kOrdLmSync5[oi] : oi;
            auto& w = *gpus[size_t(gi)];
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

    // ---- Wave-11 Pillar III: Zero-Dense-D2H 5-GPU Sharded Sparse Verification (M <= 8, 0 B new GPU hipMalloc!) ----
    // Greedy: 64 B/GPU D2H (240 B total vs 4.83 MiB!). Probabilistic: 5-shard GPU softmax packet merge (~49 KiB/GPU)!
    void compute_lm_head_tp5_sparse(const uint16_t* d_x_on_dev1, int M, bool apply_multimax,
                                    const SamplingParams& sp, std::vector<SparseProbs>& out_sparse) {
        auto& g1 = *dev_map[spine_dev1];
        const int V = mcfg.vocab_size;
        const size_t x_bytes = size_t(M) * kHiddenSize * sizeof(uint16_t);
        const bool is_greedy = (sp.temperature <= 1e-6f);

        hip_check(hipSetDevice(g1.hip_dev), "tp5 lms dev1");
        hip_check(hipMemcpyAsync(h_lm_x_pin, d_x_on_dev1, x_bytes, hipMemcpyDeviceToHost, g1.stream), "D2H tp5 lms_x");
        hip_check(hipStreamSynchronize(g1.stream), "sync tp5 lms_x");

        const int n_g = int(gpus.size());
        constexpr int kOrdLmLaunch5[5] = {2, 4, 3, 0, 1};
        constexpr int kOrdLmSync5[5]   = {1, 0, 3, 4, 2};
        for (int oi = 0; oi < n_g; ++oi) {
            const int gi = (n_g == 5) ? kOrdLmLaunch5[oi] : oi;
            auto& w = *gpus[size_t(gi)];
            const int vc = w.lm_v_count();
            if (vc <= 0) continue;
            hip_check(hipSetDevice(w.hip_dev), "tp5 lms worker");
            const uint16_t* x_ptr = (w.hip_dev == g1.hip_dev) ? d_x_on_dev1 : w.d_lm_x_in;
            if (w.hip_dev != g1.hip_dev) {
                hip_check(hipMemcpyAsync(w.d_lm_x_in, h_lm_x_pin, x_bytes, hipMemcpyHostToDevice, w.stream), "H2D tp5 lms_x");
            }
            launch_bf16x_f16w_gemv_f32(x_ptr, w.d_lm_head_shard, w.d_lm_logits_f32, M, vc, kHiddenSize, w.stream);
            if (apply_multimax && has_multimax) {
                launch_multimax_inplace_f32(w.d_lm_logits_f32, w.d_mm_ranges4, w.d_mm_ts4, M, vc, w.stream);
            }
            if (is_greedy) {
                // Reuse existing 16 MiB w.d_y_acc_f32 (0 B new VRAM): [0..63] part_v, [64..127] part_i, [128..135] out_i, [136..143] out_v
                float* d_pv   = w.d_y_acc_f32;
                int32_t* d_pi = reinterpret_cast<int32_t*>(w.d_y_acc_f32 + 64u);
                int32_t* d_oi = reinterpret_cast<int32_t*>(w.d_y_acc_f32 + 128u);
                float* d_ov   = w.d_y_acc_f32 + 136u;
                for (int r = 0; r < M; ++r) {
                    launch_argmax_f32_fast(w.d_lm_logits_f32 + size_t(r) * size_t(vc),
                                           d_pv, d_pi, d_oi + r, vc, w.stream, d_ov + r);
                }
                hip_check(hipMemcpyAsync(w.h_lm_logits_pin, w.d_y_acc_f32 + 128u,
                                         16u * sizeof(float), hipMemcpyDeviceToHost, w.stream), "D2H tp5 greedy");
            } else {
                float* d_p64 = w.d_y_acc_f32;
                auto* d_packs = reinterpret_cast<DsparkSparsePacket*>(w.d_y_acc_f32 + 256u);
                for (int r = 0; r < M; ++r) {
                    launch_gpu_softmax_sparse_top_candidates_packed(
                        w.d_lm_logits_f32 + size_t(r) * size_t(vc), vc, sp.temperature,
                        d_p64, &d_packs[r].hdr, d_packs[r].cands, kGpuMaxCands, w.stream);
                }
                hip_check(hipMemcpyAsync(w.h_lm_logits_pin, d_packs,
                                         size_t(M) * sizeof(DsparkSparsePacket),
                                         hipMemcpyDeviceToHost, w.stream), "D2H tp5 sparse packs");
            }
        }

        for (int oi = 0; oi < n_g; ++oi) {
            const int gi = (n_g == 5) ? kOrdLmSync5[oi] : oi;
            auto& w = *gpus[size_t(gi)];
            if (w.lm_v_count() <= 0) continue;
            hip_check(hipSetDevice(w.hip_dev), "tp5 lms sync dev");
            hip_check(hipStreamSynchronize(w.stream), "tp5 lms sync");
        }

        out_sparse.resize(size_t(M));
        if (is_greedy) {
            for (int r = 0; r < M; ++r) {
                float best_v = -std::numeric_limits<float>::infinity();
                int32_t best_id = 0;
                for (int gi = 0; gi < n_g; ++gi) {
                    const auto& w = *gpus[size_t(gi)];
                    if (w.lm_v_count() <= 0) continue;
                    const auto* hi = reinterpret_cast<const int32_t*>(w.h_lm_logits_pin);
                    const float* hv = w.h_lm_logits_pin + 8u;
                    const float v = hv[r];
                    const int32_t gid = w.lm_v_begin + hi[r];
                    if (v > best_v || (v == best_v && gid < best_id)) {
                        best_v = v;
                        best_id = gid;
                    }
                }
                out_sparse[size_t(r)].items.clear();
                out_sparse[size_t(r)].items.emplace_back(best_id, 1.0f);
            }
        } else {
            const double inv_t = 1.0 / double(sp.temperature);
            std::vector<DsparkCandPair> merged;
            merged.reserve(1024u);
            for (int r = 0; r < M; ++r) {
                float m_all = -std::numeric_limits<float>::infinity();
                for (int gi = 0; gi < n_g; ++gi) {
                    const auto& w = *gpus[size_t(gi)];
                    if (w.lm_v_count() <= 0) continue;
                    const auto& pk = reinterpret_cast<const DsparkSparsePacket*>(w.h_lm_logits_pin)[r];
                    if (std::isfinite(pk.hdr.stats2[0]) && pk.hdr.stats2[1] > 0.0f) {
                        if (pk.hdr.stats2[0] > m_all) m_all = pk.hdr.stats2[0];
                    }
                }
                bool overflow = !std::isfinite(m_all);
                if (!overflow) {
                    for (int gi = 0; gi < n_g; ++gi) {
                        const auto& w = *gpus[size_t(gi)];
                        if (w.lm_v_count() <= 0) continue;
                        const auto& pk = reinterpret_cast<const DsparkSparsePacket*>(w.h_lm_logits_pin)[r];
                        const float mg = pk.hdr.stats2[0];
                        if (std::isfinite(mg) && double(mg - m_all) * inv_t > -16.0 &&
                            pk.hdr.cand_cnt > kGpuMaxCands) {
                            overflow = true;
                            break;
                        }
                    }
                }
                if (overflow) {
                    // Rare overflow fallback: D2H only this single row from the 5 shards
                    for (int gi = 0; gi < n_g; ++gi) {
                        auto& w = *gpus[size_t(gi)];
                        const int vc = w.lm_v_count();
                        if (vc <= 0) continue;
                        hip_check(hipSetDevice(w.hip_dev), "tp5 lms fb dev");
                        hip_check(hipMemcpy(h_lm_full_pin + size_t(w.lm_v_begin),
                                            w.d_lm_logits_f32 + size_t(r) * size_t(vc),
                                            size_t(vc) * sizeof(float), hipMemcpyDeviceToHost), "D2H tp5 lms fb");
                    }
                    logits_to_sparse_probs_fast(h_lm_full_pin, V, sp.temperature, sp.top_k, sp.top_p, out_sparse[size_t(r)]);
                    continue;
                }
                merged.clear();
                double sum_all = 0.0;
                for (int gi = 0; gi < n_g; ++gi) {
                    const auto& w = *gpus[size_t(gi)];
                    if (w.lm_v_count() <= 0) continue;
                    const auto& pk = reinterpret_cast<const DsparkSparsePacket*>(w.h_lm_logits_pin)[r];
                    const float mg = pk.hdr.stats2[0];
                    const double sg_sum = double(pk.hdr.stats2[1]);
                    if (!std::isfinite(mg) || sg_sum <= 0.0) continue;
                    const double z = double(mg - m_all) * inv_t;
                    if (z < -24.0) continue;
                    const double sc = std::exp(z);
                    sum_all += sg_sum * sc;
                    if (z > -16.0) {
                        const int nc = std::min(pk.hdr.cand_cnt, kGpuMaxCands);
                        for (int c = 0; c < nc; ++c) {
                            const double ev = double(pk.cands[c].exp) * sc;
                            if (ev > 0.0) {
                                merged.push_back(DsparkCandPair{w.lm_v_begin + pk.cands[c].idx, float(ev)});
                            }
                        }
                    }
                }
                gpu_cand_pairs_to_sparse_probs(merged.data(), int(merged.size()), sum_all,
                                               sp.top_k, sp.top_p, out_sparse[size_t(r)]);
            }
        }
        hip_check(hipSetDevice(g1.hip_dev), "tp5 lms restore dev1");
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
        if (is_spine) {
            hip_check(hipStreamCreate(&w->stream_sh), "hipStreamCreate sh");
            hip_check(hipEventCreateWithFlags(&w->ev_sh_start, hipEventDisableTiming), "ev_sh_start");
            hip_check(hipEventCreateWithFlags(&w->ev_sh_done,  hipEventDisableTiming), "ev_sh_done");
            for (int mb = 0; mb < 4; ++mb) {
                hip_check(hipEventCreateWithFlags(&w->ev_mb_c2d[mb],     hipEventDisableTiming), "ev_mb_c2d");
                hip_check(hipEventCreateWithFlags(&w->ev_mb_d2c[mb],     hipEventDisableTiming), "ev_mb_d2c");
                hip_check(hipEventCreateWithFlags(&w->ev_mb_vha_d2h[mb], hipEventDisableTiming), "ev_mb_vha_d2h");
                hip_check(hipEventCreateWithFlags(&w->ev_mb_vha_h2d[mb], hipEventDisableTiming), "ev_mb_vha_h2d");
                hip_check(hipEventCreateWithFlags(&w->ev_mb_o_d2h[mb],   hipEventDisableTiming), "ev_mb_o_d2h");
            }
            hip_check(hipEventCreateWithFlags(&w->ev_res_b_ready, hipEventDisableTiming), "ev_res_b_ready");
            if (is_dsa_owner) {
                rocblas_check(rocblas_create_handle(&w->blas_sh), "rocblas_create_handle sh");
                rocblas_check(rocblas_set_stream(w->blas_sh, w->stream_sh), "rocblas_set_stream sh");
                rocblas_check(rocblas_set_atomics_mode(w->blas_sh, rocblas_atomics_not_allowed), "rocblas_set_atomics_mode sh");
            }
        }
        {
            int lo_pri = 0, hi_pri = 0;
            (void)hipDeviceGetStreamPriorityRange(&lo_pri, &hi_pri);
            hip_check(hipStreamCreateWithPriority(&w->keep_warm_stream, hipStreamNonBlocking, lo_pri), "hipStreamCreate dpm lo_pri");
            w->h_dpm_flag = host_mapped_alloc<uint32_t>(16u, pin_allocs, true);
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_dpm_flag), w->h_dpm_flag, 0), "dpm flag ptr");
            void* ucp = nullptr;
            hip_check(hipExtMallocWithFlags(&ucp, 65536u, hipDeviceMallocUncached), "dpm uc scratch");
            hip_check(hipMemsetAsync(ucp, 0, 65536u, w->stream), "dpm uc zero");
            w->allocs.push_back(ucp);
            w->d_dpm_uc_scratch = static_cast<uint32_t*>(ucp);
        }

        const auto lv32 = hlut4_levels_f32();
        uint16_t lv16[8];
        for (int i = 0; i < 8; ++i) lv16[i] = f32_to_fp16_rne(lv32[size_t(i)]);
        w->d_levels8_f16 = gpu_upload<uint16_t>(lv16, 8, w->allocs);

        w->d_x_rot_f16     = gpu_alloc<uint16_t>(size_t(max_chunk) * kLatentDim, w->allocs, true);
        w->d_groups        = gpu_alloc<ExpertDispatchGroup>(size_t(max_groups), w->allocs, true);
        w->h_groups_pin    = host_pin_alloc<ExpertDispatchGroup>(size_t(max_groups), pin_allocs, true);
        const size_t z_f32_elems = is_spine
            ? (size_t(80) * kMaxGroupTokens * kMoeInterSize)
            : std::max<size_t>(
                size_t(max_chunk) * kTopKExperts * kMoeInterSize / 2u,
                size_t(80) * kMaxGroupTokens * kMoeInterSize);
        w->d_z_f32         = gpu_alloc<float>(z_f32_elems, w->allocs, true);
        w->d_z_rot_f16     = gpu_alloc<uint16_t>(80u * kMaxGroupTokens * kMoeInterSize, w->allocs, true);
        w->d_y_slots_f32   = gpu_alloc<float>(8u * kTopKExperts * kLatentDim, w->allocs, true);
        w->d_y_acc_f32     = gpu_alloc<float>(size_t(max_chunk) * kLatentDim, w->allocs, true);
        w->h_y_partial_pin = host_pin_alloc<float>(size_t(max_chunk) * kLatentDim, pin_allocs, true);

        w->d_moe_topk_ids  = gpu_alloc<int32_t>(size_t(max_chunk) * kTopKExperts, w->allocs, true);
        w->d_moe_topk_wts  = gpu_alloc<float>(size_t(max_chunk) * kTopKExperts, w->allocs, true);
        w->d_g2l_bb        = gpu_alloc<int16_t>(kNumRoutedExperts, w->allocs, true);
        w->d_csr_counts    = gpu_alloc<int32_t>(4u, w->allocs, true);
        w->d_moe_tiles     = gpu_alloc<Ernie55MoeWorkTile>(4096u, w->allocs, true);
        w->d_moe_pairs     = gpu_alloc<Ernie55MoeRoutedPair>(size_t(max_chunk) * kTopKExperts, w->allocs, true);

        w->b_head.fill(-1);
        w->b_cnt.fill(0);
        w->b_next.resize(8u * kTopKExperts);
        w->b_rows.resize(8u * kTopKExperts);
        w->b_slots.resize(8u * kTopKExperts);
        w->b_wts.resize(8u * kTopKExperts);
        w->active_exps.reserve(kNumRoutedExperts);
        w->h_groups.reserve(size_t(max_groups) * 4u);

        // Compact 512 KiB FP16 input buffer for M>8 rocblas_gemm_ex lm_head (0 MiB d_lm_w_f32!)
        w->d_lm_x_f16 = gpu_alloc<uint16_t>(64u * kHiddenSize, w->allocs, true);

        if (is_spine && h_tp_mapped_slabs) {
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_tp_mapped_slabs), h_tp_mapped_slabs, 0), "mapped slabs ptr");
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_tp_mapped_ready), h_tp_mapped_ready, 0), "mapped ready ptr");
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_tp_mapped_done),  h_tp_mapped_done,  0), "mapped done ptr");
        }
        if (h_moe_mapped_slabs) {
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_moe_mapped_slabs), h_moe_mapped_slabs, 0), "moe mapped slabs ptr");
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_moe_mapped_ready), h_moe_mapped_ready, 0), "moe mapped ready ptr");
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&w->d_moe_mapped_done),  h_moe_mapped_done,  0), "moe mapped done ptr");
        }

        if (is_spine) {
            const auto yarn_inv = compute_hca_yarn_inv_freq(
                kHcaRopeHeadDim, mcfg.compress_rope_theta, mcfg.yarn_factor,
                mcfg.yarn_orig_max_pos, mcfg.yarn_beta_fast, mcfg.yarn_beta_slow);
            const auto dsa_inv = compute_dsa_inv_freq(kDsaQkRopeHeadDim, mcfg.rope_theta);
            w->yarn_rope = upload_rope2d(yarn_inv, w->allocs);
            w->dsa_rope  = upload_rope2d(dsa_inv,  w->allocs);

            // GEMM strip buffers stay compact at 512-row strip size even when max_chunk == 1024!
            // Zero all scratch buffers once at startup (0 B net new VRAM, <1 ms boot) to eliminate cold-boot NaN hazards.
            w->d_x_f16         = gpu_alloc<uint16_t>(512u * 16384u, w->allocs, true);
            w->d_y_f32         = gpu_alloc<float>(512u * (is_dsa_owner ? 36864u : 16384u), w->allocs, true);
            w->d_hca_kcomp_f16 = gpu_alloc<uint16_t>(
                std::max<size_t>(size_t((opt.max_kv + 127) / 128 + 1024), size_t(max_chunk) + 512u) * 512u,
                w->allocs, true);
            const int h_loc    = kHcaNumHeads / std::max(1, tp_size);
            // Cold-boot Request #1 fix: these two buffers are read by k_hca_comp_merge_tile_norm_f16
            // and k_hca_tp4_swa128_merge_f16, whose only guard is `comp_l_f32[row] > 0.0f`. Because
            // max_comp = (pos0 + M - 1) / 128 is 15 even at pos0 == 0, the first chunk after startup
            // reaches the merge before every row has been covered by a strip, so any uncovered row
            // kept uninitialised garbage that passed the `> 0.0f` gate and corrupted the prompt.
            // Zeroing at allocation removes the whole class for 0 B net new VRAM (16 MiB + 128 KiB
            // that were already reserved).
            w->d_hca_o_tot_f32 = gpu_alloc<float>(size_t(max_chunk) * size_t(h_loc) * 512u, w->allocs, true);
            w->d_hca_ml_f32    = gpu_alloc<float>(size_t(max_chunk) * size_t(h_loc) * 4u, w->allocs, true);
            if (is_dsa_owner) {
                w->d_sh_x_f16  = gpu_alloc<uint16_t>(512u * kHiddenSize, w->allocs, true);
                w->d_sh_y_f32  = gpu_alloc<float>(512u * kHiddenSize, w->allocs, true);
                w->d_sh_tmp_a  = gpu_alloc<uint16_t>(size_t(max_chunk) * kMoeInterSize, w->allocs, true);
                w->d_sh_tmp_b  = gpu_alloc<uint16_t>(size_t(max_chunk) * kMoeInterSize, w->allocs, true);
                w->d_sh_out    = gpu_alloc<uint16_t>(size_t(max_chunk) * kHiddenSize, w->allocs, true);
            }
            w->d_mhc_ws        = gpu_alloc<float>(size_t(max_chunk) * 100u, w->allocs, true);
            w->d_argmax_v64    = gpu_alloc<float>(64u, w->allocs, true);
            w->d_argmax_i64    = gpu_alloc<int32_t>(64u, w->allocs, true);

            w->d_res_a         = gpu_alloc<uint16_t>(size_t(max_chunk) * kHcDim, w->allocs, true);
            w->d_res_b         = gpu_alloc<uint16_t>(size_t(max_chunk) * kHcDim, w->allocs, true);
            w->d_post_mix      = gpu_alloc<float>(size_t(max_chunk) * kHcMult, w->allocs, true);
            w->d_comb_mix      = gpu_alloc<float>(size_t(max_chunk) * kHcMult2, w->allocs, true);
            w->d_x_in          = gpu_alloc<uint16_t>(size_t(max_chunk) * kHiddenSize, w->allocs, true);
            w->d_sub_out       = gpu_alloc<uint16_t>(size_t(max_chunk) * kHiddenSize, w->allocs, true);
            w->d_tmp_a         = gpu_alloc<uint16_t>(size_t(max_chunk) * 32768u, w->allocs, true);
            w->d_tmp_b         = gpu_alloc<uint16_t>(std::max<size_t>(size_t(max_chunk) * 16384u, hip_dev == spine_dev1 ? 64u * 32768u : 0u), w->allocs, true);
            w->d_tmp_c         = gpu_alloc<uint16_t>(size_t(max_chunk) * (is_dsa_owner ? 36864u : 9216u), w->allocs, true);
            w->d_tmp_d         = gpu_alloc<uint16_t>(std::max<size_t>(size_t(max_chunk) * 4096u, 64u * 8192u), w->allocs, true);
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
                // Decode M<=8 only (prefill M>8 uses zero-gather k_dsa_sparse_mqa64_fused_vdot2 -> saves 19.25 MiB/DSA GPU!)
                // Cold-boot Request #1 fix (same class as d_hca_o_tot_f32): the decode M<=8 DSA path
                // consumes ksel rows that are only written for the slots a route actually selects, so
                // an unwritten row must read as zero rather than as arbitrary HBM2 garbage.
                w->d_dsa_ksel_f16 = gpu_alloc<uint16_t>(8u * size_t(kDsaIndexTopK + kSlidingWindow) * 576u, w->allocs, true);
                w->idx_scores_cap = std::max<size_t>(size_t(opt.max_kv), 16ULL * 1024ULL * 1024ULL);
                w->d_idx_scores   = gpu_alloc<float>(w->idx_scores_cap, w->allocs, true);
                w->d_fi_gates     = gpu_alloc<float>(size_t(max_chunk) * 64u, w->allocs, true);
                w->d_fi_q8        = gpu_alloc<uint16_t>(size_t(max_chunk) * size_t(kFlashIndexerTopKHeads) * 128u, w->allocs, true);
                w->d_fi_w8        = gpu_alloc<float>(size_t(max_chunk) * size_t(kFlashIndexerTopKHeads), w->allocs, true);
                w->d_sel_idx      = gpu_alloc<int32_t>(size_t(max_chunk) * size_t(kDsaIndexTopK + kSlidingWindow), w->allocs, true);
                w->d_sel_cnts     = gpu_alloc<int32_t>(size_t(max_chunk), w->allocs, true);
            }
            // Wave-10 C1 + Wave-11 Turn-10: Eagerly load ALL tuned Tensile Type_HH_HPA & Type_HS_HPA code objects
            // on BOTH w->blas AND w->blas_sh BEFORE keep_warm_stream runs (eliminates +300 ms cold-boot Request #1
            // hca spike and prevents lazy hipModuleLoad barrier hazards; 0 B net new GPU VRAM!).
            {
                const float alpha = 1.0f, beta = 0.0f;
                uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w->d_y_f32);
                auto warm_handle = [&](rocblas_handle bh, hipStream_t bs) {
                    rocblas_check(rocblas_gemm_ex(
                        bh, rocblas_operation_transpose, rocblas_operation_none,
                        64, 64, 64, &alpha,
                        w->d_x_f16, rocblas_datatype_f16_r, 64,
                        w->d_x_f16, rocblas_datatype_f16_r, 64,
                        &beta,
                        d_y16, rocblas_datatype_f16_r, 64,
                        d_y16, rocblas_datatype_f16_r, 64,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "warmup hh_hpa");
                    rocblas_check(rocblas_gemm_ex(
                        bh, rocblas_operation_transpose, rocblas_operation_none,
                        64, 64, 64, &alpha,
                        w->d_x_f16, rocblas_datatype_f16_r, 64,
                        w->d_x_f16, rocblas_datatype_f16_r, 64,
                        &beta,
                        w->d_y_f32, rocblas_datatype_f32_r, 64,
                        w->d_y_f32, rocblas_datatype_f32_r, 64,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "warmup hs_hpa");
                    if (gfx906_rocblas_sol_enabled()) {
                        const int32_t tn_sols[] = {32975, 32977, 32989, 32990, 33063, 33064, 33065, 33066, 33068};
                        for (int32_t sol : tn_sols) {
                            (void)rocblas_gemm_ex(
                                bh, rocblas_operation_transpose, rocblas_operation_none,
                                256, 256, 256, &alpha,
                                w->d_x_f16, rocblas_datatype_f16_r, 256,
                                w->d_x_f16, rocblas_datatype_f16_r, 256,
                                &beta,
                                d_y16, rocblas_datatype_f16_r, 256,
                                d_y16, rocblas_datatype_f16_r, 256,
                                rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, sol, 0);
                        }
                        const int32_t nn_sols[] = {33152, 33161, 33169, 33181};
                        for (int32_t sol : nn_sols) {
                            (void)rocblas_gemm_ex(
                                bh, rocblas_operation_none, rocblas_operation_none,
                                256, 256, 256, &alpha,
                                w->d_x_f16, rocblas_datatype_f16_r, 256,
                                w->d_x_f16, rocblas_datatype_f16_r, 256,
                                &beta,
                                d_y16, rocblas_datatype_f16_r, 256,
                                d_y16, rocblas_datatype_f16_r, 256,
                                rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, sol, 0);
                        }
                        (void)rocblas_gemm_ex(
                            bh, rocblas_operation_transpose, rocblas_operation_none,
                            256, 256, 256, &alpha,
                            w->d_x_f16, rocblas_datatype_f16_r, 256,
                            w->d_x_f16, rocblas_datatype_f16_r, 256,
                            &beta,
                            w->d_y_f32, rocblas_datatype_f32_r, 256,
                            w->d_y_f32, rocblas_datatype_f32_r, 256,
                            rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 31468, 0);
                    }
                    hip_check(hipStreamSynchronize(bs), "warmup blas sync");
                };
                warm_handle(w->blas, w->stream);
                if (w->blas_sh && w->stream_sh) warm_handle(w->blas_sh, w->stream_sh);
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
        const size_t emb_elems = size_t(mcfg.vocab_size) * kHiddenSize;
        const auto* emb_raw = reinterpret_cast<const uint16_t*>(
            require_st_bytes(st, "model.embed_tokens.weight", emb_elems * 2u));
        h_embed_mirror.assign(emb_raw, emb_raw + emb_elems);
        d_embed_bf16 = gpu_upload<uint16_t>(emb_raw, emb_elems, g0.allocs);

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
        // Commit 4a: Exact FP64 rank-16 factorization of VHA (vha_m - I = Q[64,16] @ R[16,64]) -> 4x less TP=4 VHA PCIe!
        std::array<float, 64 * 16> vha_Q_full{};
        std::array<float, 16 * 64> vha_R_full{};
        bool vha_lr_ok = false;
        if (tp == 4) {
            auto vm = load_st_as_f32(st, "attn.vha_m");
            if (vm.size() == 64u * 64u) {
                double max_err = 0.0;
                vha_lr_ok = factor_vha_rank16(vm.data(), vha_Q_full.data(), vha_R_full.data(), &max_err);
                if (vha_lr_ok) ++vha_lr_count;
            }
        }
        for (int r = 0; r < tp; ++r) {
            const int dev_r = L.is_echo ? owner : tp_devs[r];
            auto& g_r = *dev_map[dev_r];
            hip_check(hipSetDevice(g_r.hip_dev), "set spine layer tp dev");

            L.attn_mhc_tp[r]  = upload_mhc(st, "attn_mhc", g_r);
            L.mlp_mhc_tp[r]   = upload_mhc(st, "mlp_mhc",  g_r);
            L.d_in_norm_tp[r] = upload_bf16_tensor(st, "input_layernorm.weight", kHiddenSize, g_r);
            L.d_post_norm_tp[r] = upload_bf16_tensor(st, "post_attention_layernorm.weight", kHiddenSize, g_r);

            float q_loc[256], r_loc[256];
            if (vha_lr_ok) {
                for (int i = 0; i < 16; ++i) {
                    for (int k = 0; k < 16; ++k) {
                        q_loc[i * 16 + k] = vha_Q_full[size_t(r * 16 + i) * 16u + size_t(k)];
                        r_loc[i * 16 + k] = vha_R_full[size_t(i) * 64u + size_t(r * 16 + k)];
                    }
                }
            }

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
                if (vha_lr_ok) {
                    H.d_vha_q_loc = gpu_upload<float>(q_loc, 256u, g_r.allocs);
                    H.d_vha_r_loc = gpu_upload<float>(r_loc, 256u, g_r.allocs);
                    H.has_vha_lr  = true;
                }
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
                if (vha_lr_ok) {
                    D.d_vha_q_loc = gpu_upload<float>(q_loc, 256u, g_r.allocs);
                    D.d_vha_r_loc = gpu_upload<float>(r_loc, 256u, g_r.allocs);
                    D.has_vha_lr  = true;
                }
                D.d_gate_proj = upload_bf16_rows_as_f16(st, "attn.gate_proj", 16384u, size_t(r) * size_t(h_loc) * 256u, size_t(h_loc) * 256u, 1024u, g_r);
                D.d_o_proj    = upload_bf16_cols_as_f16(st, "attn.o_proj", 4096u, size_t(r) * size_t(h_loc) * 256u, size_t(h_loc) * 256u, 16384u, g_r);
                D.d_attn_sink = upload_f32_tensor(st, "attn.attn_sink", 64u, g_r);
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
        d_dspark_step_logits = gpu_alloc<float>(size_t(mcfg.vocab_size), g1.allocs, false);
        d_dspark_sm_pack     = gpu_alloc<DsparkSparsePacket>(1u, g1.allocs, true);
        h_dspark_sm_pack_pin = host_pin_alloc<DsparkSparsePacket>(1u, pin_allocs, true);
        h_dspark_me_pin      = host_pin_alloc<uint16_t>(size_t(std::max(1, mcfg.dspark_markov_rank)), pin_allocs, true);

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
        d_dspark_markov_w1_T_mapped = nullptr;
        h_dspark_markov_reg = false;
        if (hipHostRegister(h_dspark_markov_w1_T.data(), w1_elems * 2u,
                            hipHostRegisterPortable | hipHostRegisterMapped) == hipSuccess) {
            h_dspark_markov_reg = true;
            hip_check(hipSetDevice(g1.hip_dev), "dspark markov reg dev1");
            if (hipHostGetDevicePointer(reinterpret_cast<void**>(&d_dspark_markov_w1_T_mapped),
                                        h_dspark_markov_w1_T.data(), 0) != hipSuccess) {
                d_dspark_markov_w1_T_mapped = nullptr;
            }
        }

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
    // Wave-10 C6: GPU-side iota pos_ids + fully asynchronous on g1.stream (overlaps with CPU token emit & g0 anchor fetch!).
    void commit_dspark_context_kv(int n_commit, int32_t pos0) {
        if (!has_dspark || n_commit <= 0) return;
        auto& g1 = *dev_map[spine_dev1];
        const int n_ds = int(dspark_layers.size());
        hip_check(hipSetDevice(g1.hip_dev), "commit dspark dev1");
        launch_fill_iota_i32(g1.d_pos_ids, pos0, n_commit, g1.stream);
        bf16_linear_bf16(g1, d_dspark_aux_cat, d_dspark_main_proj, g1.d_sub_out, n_commit, kHiddenSize, n_ds * kHiddenSize);
        launch_rmsnorm_bf16(g1.d_sub_out, d_dspark_main_norm, g1.d_x_in, n_commit, kHiddenSize, mcfg.rms_norm_eps, g1.stream);
        for (int i = 0; i < n_ds; ++i) {
            auto& DL = dspark_layers[size_t(i)];
            if (!DL.d_swa_ring) continue;
            bf16_linear_bf16(g1, g1.d_x_in, DL.swa_attn.d_kv_proj, g1.d_tmp_d, n_commit, 512, kHiddenSize);
            launch_rmsnorm_bf16(g1.d_tmp_d, DL.swa_attn.d_kv_norm, g1.d_kv_tmp, n_commit, 512, mcfg.rms_norm_eps, g1.stream);
            launch_dspark_swa_ring_write(g1.d_kv_tmp, g1.d_pos_ids, g1.dsa_rope, DL.d_swa_ring, n_commit, g1.stream);
        }
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

        // 1. Copy 1 anchor embedding row (8 KiB) directly from host DDR4 mirror (0 g0 D2H, 0 hipStreamSynchronize!),
        //    then enqueue H2D row 0 + fused 1-launch noise broadcast [1..B-1] & pos_ids [0..B-1] on g1.stream!
        const int32_t safe_anchor = std::clamp(anchor_tok, 0, V - 1);
        if (!h_embed_mirror.empty()) {
            std::memcpy(h_dspark_anchor_pin, h_embed_mirror.data() + size_t(safe_anchor) * kHiddenSize,
                        kHiddenSize * 2u);
        } else {
            hip_check(hipSetDevice(g0.hip_dev), "dspark anchor dev0");
            hip_check(hipMemcpyAsync(h_dspark_anchor_pin, d_embed_bf16 + size_t(safe_anchor) * kHiddenSize,
                                     kHiddenSize * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H anchor emb");
            hip_check(hipStreamSynchronize(g0.stream), "sync anchor emb");
        }

        hip_check(hipSetDevice(g1.hip_dev), "dspark fwd dev1");
        hip_check(hipMemcpyAsync(g1.d_x_in, h_dspark_anchor_pin, kHiddenSize * 2u, hipMemcpyHostToDevice, g1.stream), "H2D anchor");
        launch_dspark_broadcast_noise_and_pos(d_dspark_noise_emb, g1.d_x_in, g1.d_pos_ids, pos0, B, kHiddenSize, g1.stream);

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

        if (!prob_mode && !want_conf && d_dspark_markov_w1_T_mapped != nullptr && B <= max_chunk - 1) {
            // Wave-11 Cut D3.4: Zero-sync GPU DSpark Markov chain — 1 H2D (4 B) + B chained launches + 1 D2H (4*B B)!
            h_dspark_pos_pin[0] = std::clamp(anchor_tok, 0, V - 1);
            hip_check(hipMemcpyAsync(g1.d_tok_ids, h_dspark_pos_pin, sizeof(int32_t), hipMemcpyHostToDevice, g1.stream), "H2D ds anchor tok");
            for (int b = 0; b < B; ++b) {
                launch_dspark_markov_bias_multimax_from_tok_f32(
                    d_dspark_markov_w1_T_mapped, g1.d_tok_ids + b,
                    d_dspark_markov_w2,
                    g1.d_logits_f32 + size_t(b) * size_t(V),
                    has_multimax ? d_mm_ranges4 : nullptr, has_multimax ? d_mm_ts4 : nullptr,
                    d_dspark_step_logits, V, R, g1.stream);
                launch_argmax_f32_fast(d_dspark_step_logits, g1.d_argmax_v64, g1.d_argmax_i64,
                                       g1.d_tok_ids + b + 1, V, g1.stream);
            }
            hip_check(hipMemcpyAsync(h_dspark_pos_pin, g1.d_tok_ids + 1, size_t(B) * sizeof(int32_t),
                                     hipMemcpyDeviceToHost, g1.stream), "D2H ds batch toks");
            hip_check(hipStreamSynchronize(g1.stream), "sync ds batch toks");
            for (int b = 0; b < B; ++b) {
                out.draft_tokens.push_back(h_dspark_pos_pin[b]);
                out.step_confidences.push_back(1.0f);
            }
        } else {
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
            // Wave-10 C6: Pinned me_row staging + 1-launch fused fdot2 Markov bias + MultiMax (0 bias HBM2 bounce!)
            std::memcpy(h_dspark_me_pin, me_row, size_t(R) * 2u);
            hip_check(hipMemcpyAsync(d_dspark_me_bf16, h_dspark_me_pin, size_t(R) * 2u, hipMemcpyHostToDevice, g1.stream), "H2D me pin");
            launch_dspark_markov_bias_multimax_f32(
                d_dspark_me_bf16, d_dspark_markov_w2,
                g1.d_logits_f32 + size_t(b) * size_t(V),
                has_multimax ? d_mm_ranges4 : nullptr, has_multimax ? d_mm_ts4 : nullptr,
                d_dspark_step_logits, V, R, g1.stream);

            int32_t tok = 0;
            if (!prob_mode) {
                launch_argmax_f32_fast(d_dspark_step_logits, g1.d_argmax_v64, g1.d_argmax_i64, g1.d_tok_ids, V, g1.stream);
                hip_check(hipMemcpyAsync(h_dspark_pos_pin, g1.d_tok_ids, sizeof(int32_t), hipMemcpyDeviceToHost, g1.stream), "D2H ds tok");
                hip_check(hipStreamSynchronize(g1.stream), "sync ds tok");
                tok = h_dspark_pos_pin[0];
            } else {
                // Wave-10 C6: Single-DMA single-sync coalesced sparse-candidate packet (1040 B D2H covers n_cand<=128!)
                launch_gpu_softmax_sparse_top_candidates_packed(
                    d_dspark_step_logits, V, sp.temperature,
                    g1.d_argmax_v64, &d_dspark_sm_pack->hdr,
                    d_dspark_sm_pack->cands, kGpuMaxCands, g1.stream);
                constexpr size_t kInlineBytes =
                    sizeof(DsparkSparsePacketHeader) + size_t(kInlineCands) * sizeof(DsparkCandPair);
                hip_check(hipMemcpyAsync(h_dspark_sm_pack_pin, d_dspark_sm_pack, kInlineBytes,
                                         hipMemcpyDeviceToHost, g1.stream), "D2H sm inline pack");
                hip_check(hipStreamSynchronize(g1.stream), "sync sm inline pack");
                const int n_cand = h_dspark_sm_pack_pin->hdr.cand_cnt;
                const double sum_exp = double(h_dspark_sm_pack_pin->hdr.stats2[1]);
                SparseProbs q_b;
                if (n_cand > 0 && n_cand <= kGpuMaxCands && sum_exp > 0.0) {
                    if (n_cand > kInlineCands) {
                        const size_t tail_n = size_t(n_cand - kInlineCands);
                        hip_check(hipMemcpyAsync(h_dspark_sm_pack_pin->cands + kInlineCands,
                                                 d_dspark_sm_pack->cands + kInlineCands,
                                                 tail_n * sizeof(DsparkCandPair),
                                                 hipMemcpyDeviceToHost, g1.stream), "D2H sm tail cands");
                        hip_check(hipStreamSynchronize(g1.stream), "sync sm tail cands");
                    }
                    gpu_cand_pairs_to_sparse_probs(h_dspark_sm_pack_pin->cands, n_cand, sum_exp,
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
        }
        profiling_enabled = saved_prof;
        if (profiling_enabled) {
            prof.dspark_draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ds0).count();
            ++prof.dspark_draft_calls;
        }
        return out;
    }

    // ---- TP=4 / Single-GPU Attention Sublayer (HCA Batched FP16 GEMM + DSA Batched GEMM FlashIndexer & FP16 SGEMM-DSA!) ----
    void run_attention_tp(LayerSpineGpu& L, int M, int32_t pos0,
                          bool defer_nonowner_oproj = false,
                          bool* nonowner_oproj_deferred_out = nullptr,
                          bool x_f16_ready = false) {
        if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = false;
        o_proj_host_d2h_pending = false;
        const auto t_a0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int tp = L.tp_size;
        const auto dev_for_r = [&](int r) -> int { return L.is_echo ? L.owner_dev : tp_devs[r]; };
        constexpr int kGemmStrip = 512;

        if (!L.is_dsa) {
            const int h_loc = kHcaNumHeads / tp;
            const int g_loc = kHcaOGroups / tp;
            const int max_comp = (pos0 + M - 1) / kHcaCompressRatio;
            const bool q_stay_f16 = (M > 8 && !(tp == 1 && M > 256));
            const bool use_hca_vha_lr = (tp == 4 && M > 8 && L.hca_tp[0].has_vha_lr);
            const bool vha_stay_f16 = (q_stay_f16 && use_hca_vha_lr);

            // Phase A: Parallel Q/KV/Compressor/Attention across TP ranks (16 heads/GPU on TP4!)
            const bool do_hca_bl_xray = (M >= 256 && moe_pre_xray);
            const auto t_hp0 = (profiling_enabled || do_hca_bl_xray) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& H = L.hca_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "hca A dev");

                if (vha_stay_f16) {
                    if (x_f16_ready && (M & 63) == 0) {
                        f16_linear_f16(w, w.d_x_f16, H.d_qkv_comp, w.d_tmp_a, M, 2560, kHiddenSize);
                    } else {
                        bf16_linear_f16(w, w.d_x_in, H.d_qkv_comp, w.d_tmp_a, M, 2560, kHiddenSize);
                    }
                    launch_hca_split_qkv_comp_norm_f16(w.d_tmp_a, H.d_q_norm, H.d_kv_norm,
                                                       w.d_qr, w.d_kv_tmp, w.d_f32_tmp0, w.d_f32_tmp1,
                                                       M, mcfg.rms_norm_eps, w.stream);
                    f16_linear_f16(w, w.d_qr, H.d_q_up, w.d_tmp_a, M, h_loc * 512, 1024);
                    launch_hca_q_qknorm_rope_f16(w.d_tmp_a, w.d_pos_ids, w.yarn_rope, M, h_loc, mcfg.rms_norm_eps, w.stream);
                } else {
                    bf16_linear_f32(w, w.d_x_in, H.d_qkv_comp, w.d_y_f32, M, 2560, kHiddenSize,
                                    w.d_tmp_a, size_t(max_chunk) * 16384u);
                    launch_hca_split_qkv_comp_norm(w.d_y_f32, H.d_q_norm, H.d_kv_norm,
                                                   w.d_qr, w.d_kv_tmp, w.d_f32_tmp0, w.d_f32_tmp1,
                                                   M, mcfg.rms_norm_eps, w.stream);
                    if (q_stay_f16) {
                        bf16_linear_f16(w, w.d_qr, H.d_q_up, w.d_tmp_a, M, h_loc * 512, 1024);
                        launch_hca_q_qknorm_rope_f16(w.d_tmp_a, w.d_pos_ids, w.yarn_rope, M, h_loc, mcfg.rms_norm_eps, w.stream);
                    } else {
                        bf16_linear_bf16(w, w.d_qr, H.d_q_up, w.d_tmp_a, M, h_loc * 512, 1024);
                        launch_hca_q_qknorm_rope(w.d_tmp_a, w.d_pos_ids, w.yarn_rope, M, h_loc, mcfg.rms_norm_eps, w.stream);
                    }
                }
            }
            if (profiling_enabled || do_hca_bl_xray) {
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.hca_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hp0).count();
                } else {
                    auto& go = *dev_map[L.owner_dev];
                    hip_check(hipSetDevice(go.hip_dev), "bl hca qkv dev");
                    hip_check(hipStreamSynchronize(go.stream), "bl hca qkv sync");
                    chunk_bk_qkv += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hp0).count();
                }
            }
            const auto t_ha0 = (profiling_enabled || do_hca_bl_xray) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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
                        if (gfx906_hca_mfused_decode_enabled()) {
                            // Wave-12 Turn-17 Cut D12.1: M-fused double-buffered pass 1 + pass 2 merge (4x fewer comp_kv reads).
                            launch_hca_decode_splitk_mfused(
                                w.d_tmp_a + size_t(t0) * q_tok_stride, L.d_hca_comp_kv_tp[r], L.d_hca_swa_ring_tp[r],
                                H.d_attn_sink, w.d_hca_o_tot_f32, w.d_hca_ml_f32,
                                w.d_tmp_b + size_t(t0) * q_tok_stride, h_loc, p0, seg, num_comp, w.stream);
                        } else {
                        launch_hca_attn_c128_swa_sink_splitk(
                            w.d_tmp_a + size_t(t0) * q_tok_stride, L.d_hca_comp_kv_tp[r], L.d_hca_swa_ring_tp[r],
                            H.d_attn_sink, w.d_hca_o_tot_f32, w.d_hca_ml_f32,
                            w.d_tmp_b + size_t(t0) * q_tok_stride, h_loc, p0, seg, num_comp, w.stream);
                        }
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
                        // Wave-11 Cut P1.3+ / Wave-12 Cut P12.3: Adaptive per-tile cur_tile in {64,128,256,512,1024}
                        // keeping full comp_strip=512 (n=8192) on ALL 4 TP4 ranks (both 72 MiB owners & 32 MiB non-owners)
                        // with zero tail-padding waste at 544K-1M (at most 63 zero-padded blocks, 0 B net new VRAM!).
                        auto pick_hca_tile = [](int rem) -> int {
                            if (rem <= 64)  return 64;
                            if (rem <= 128) return 128;
                            if (rem <= 256) return 256;
                            if (rem <= 512) return 512;
                            return 1024;
                        };
                        const int blk_tile = pick_hca_tile(max_comp);
                        const int comp_strip = std::min(kGemmStrip, 8192 / h_loc);
                        const size_t ml_stride = size_t(max_chunk) * size_t(h_loc);
                        float* d_m_tot = w.d_hca_ml_f32;
                        float* d_l_tot = w.d_hca_ml_f32 + ml_stride;
                        float* d_m_blk = w.d_hca_ml_f32 + ml_stride * 2u;
                        float* d_l_blk = w.d_hca_ml_f32 + ml_stride * 3u;
                        uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                        const size_t hca_probs_off = size_t(comp_strip) * size_t(h_loc) * size_t(blk_tile);
                        uint16_t* d_probs_comp = (blk_tile >= 1024) ? (d_y16 + hca_probs_off) : (w.d_x_f16 + 4194304u);
                        uint16_t* d_probs_swa  = w.d_x_f16 + 4194304u;
                        const float alpha = 1.0f, beta = 0.0f;
                        const float alpha_hca = 0.04419417382415922f;

                        if (max_comp > 0) {
                            int max_comp_pad = 0;
                            for (int b0 = 0; b0 < max_comp; ) {
                                b0 += pick_hca_tile(max_comp - b0);
                                max_comp_pad = b0;
                            }
                            launch_bf16_to_f16(L.d_hca_comp_kv_tp[r], w.d_hca_kcomp_f16, max_comp * 512, w.stream);
                            if (max_comp < max_comp_pad) {
                                hip_check(hipMemsetAsync(w.d_hca_kcomp_f16 + size_t(max_comp) * 512u, 0,
                                                         size_t(max_comp_pad - max_comp) * 512u * sizeof(uint16_t), w.stream), "zero hca k pad");
                            }
                            for (int m0 = 0; m0 < M; m0 += comp_strip) {
                                const int B = std::min(comp_strip, M - m0);
                                const int B_eff = std::min(comp_strip, ((B + 63) / 64) * 64);
                                const int n_rows = B * h_loc;
                                const int n_rows_eff = B_eff * h_loc;
                                const size_t row_off = size_t(m0) * size_t(h_loc);
                                const uint16_t* d_q_in = w.d_x_f16;
                                if (q_stay_f16 && n_rows == n_rows_eff) {
                                    d_q_in = w.d_tmp_a + row_off * 512u;
                                } else {
                                    if (q_stay_f16) {
                                        hip_check(hipMemcpyAsync(w.d_x_f16, w.d_tmp_a + row_off * 512u,
                                                                 size_t(n_rows) * 512u * sizeof(uint16_t), hipMemcpyDeviceToDevice, w.stream), "d2d hca q pad");
                                    } else {
                                        launch_bf16_to_f16(w.d_tmp_a + row_off * 512u, w.d_x_f16, n_rows * 512, w.stream);
                                    }
                                    if (n_rows < n_rows_eff) {
                                        hip_check(hipMemsetAsync(w.d_x_f16 + size_t(n_rows) * 512u, 0,
                                                                 size_t(n_rows_eff - n_rows) * 512u * sizeof(uint16_t), w.stream), "zero hca q pad");
                                    }
                                }
                                for (int blk0 = 0; blk0 < max_comp; ) {
                                    const int cur_tile = pick_hca_tile(max_comp - blk0);
                                    if (n_rows < n_rows_eff) {
                                        hip_check(hipMemsetAsync(d_probs_comp + size_t(n_rows) * size_t(cur_tile), 0,
                                                                 size_t(n_rows_eff - n_rows) * size_t(cur_tile) * sizeof(uint16_t), w.stream), "zero hca comp p pad");
                                    }
                                    const uint16_t* d_k_tile = w.d_hca_kcomp_f16 + size_t(blk0) * 512u;
                                    rocblas_gemm_ex_tuned(
                                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                        cur_tile, n_rows_eff, 512, &alpha_hca,
                                        d_k_tile, rocblas_datatype_f16_r, 512,
                                        d_q_in,   rocblas_datatype_f16_r, 512,
                                        &beta,
                                        d_y16, rocblas_datatype_f16_r, cur_tile,
                                        d_y16, rocblas_datatype_f16_r, cur_tile, "hca comp QK hh_hpa");
                                    launch_hca_comp_softmax_f16_in_norm(d_y16, d_m_blk, d_l_blk, d_probs_comp, B, h_loc, pos0 + m0, blk0, cur_tile, w.stream);
                                    rocblas_gemm_ex_tuned(
                                        w.blas, rocblas_operation_none, rocblas_operation_none,
                                        512, n_rows_eff, cur_tile, &alpha,
                                        d_k_tile,     rocblas_datatype_f16_r, 512,
                                        d_probs_comp, rocblas_datatype_f16_r, cur_tile,
                                        &beta,
                                        d_y16, rocblas_datatype_f16_r, 512,
                                        d_y16, rocblas_datatype_f16_r, 512, "hca comp PV hh_hpa");
                                    launch_hca_comp_merge_tile_norm_f16(
                                        w.d_hca_o_tot_f32 + row_off * 512u, d_m_tot + row_off, d_l_tot + row_off,
                                        d_y16, d_m_blk, d_l_blk, n_rows, blk0 == 0, w.stream);
                                    blk0 += cur_tile;
                                }
                            }
                        }

                        const int n_pre    = std::min(127, int(pos0));
                        const int base_pos = int(pos0) - n_pre;
                        const int n_pad    = n_pre + M + 256;
                        launch_hca_pack_swa_keys_f16(
                            L.d_hca_swa_ring_tp[r], w.d_kv_tmp, w.d_hca_kcomp_f16,
                            pos0, n_pre, M, n_pad, w.stream);
                        if (q_stay_f16 && vha_stay_f16 && h_loc == 16 && pos0 >= 128 && (M & 127) == 0) {
                            // Wave-11 C5: 2-iteration b<=8 Strided-Batched SWA-128 (Type_HH_HPA sol=32989/33169, 1.20 ms vs 2.19 ms!)
                            // 0 B extra VRAM: scores (8 MiB) & PV (16 MiB) fit 32 MiB d_y_f32; probs (8 MiB) fits d_x_f16[8..16 MiB)!
                            for (int m0 = 0; m0 < M; m0 += 1024) {
                                const int n_strips = std::min(8, (M - m0) / 128);
                                const int B = n_strips * 128;
                                const int k_slot0 = (int(pos0) + m0 - 127) - base_pos;
                                const uint16_t* d_k_win = w.d_hca_kcomp_f16 + size_t(k_slot0) * 512u;
                                const uint16_t* d_q_win = w.d_tmp_a + size_t(m0) * 16u * 512u;
                                rocblas_gemm_strided_batched_ex_tuned(
                                    w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                    256, 2048, 512, &alpha_hca,
                                    d_k_win, rocblas_datatype_f16_r, 512, 128LL * 512LL,
                                    d_q_win, rocblas_datatype_f16_r, 512, 2048LL * 512LL,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, 256, 256LL * 2048LL,
                                    d_y16, rocblas_datatype_f16_r, 256, 256LL * 2048LL,
                                    n_strips, "hca swa QK batched8 hh_hpa");
                                launch_hca_swa_softmax_batched128_f16_in_norm(
                                    d_y16, d_m_blk, d_l_blk, d_probs_swa, B, 16, w.stream);
                                rocblas_gemm_strided_batched_ex_tuned(
                                    w.blas, rocblas_operation_none, rocblas_operation_none,
                                    512, 2048, 256, &alpha,
                                    d_k_win,     rocblas_datatype_f16_r, 512, 128LL * 512LL,
                                    d_probs_swa, rocblas_datatype_f16_r, 256, 256LL * 2048LL,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, 512, 512LL * 2048LL,
                                    d_y16, rocblas_datatype_f16_r, 512, 512LL * 2048LL,
                                    n_strips, "hca swa PV batched8 hh_hpa");
                                launch_hca_swa_finalize_merge_sink_f16_out(
                                    d_y16, d_m_blk, d_l_blk,
                                    (max_comp > 0) ? w.d_hca_o_tot_f32 : nullptr,
                                    (max_comp > 0) ? d_m_tot : nullptr,
                                    (max_comp > 0) ? d_l_tot : nullptr,
                                    H.d_attn_sink, w.d_tmp_b, B, 16, m0 * 16, w.stream);
                            }
                        } else if (q_stay_f16 && vha_stay_f16 && h_loc == 16) {
                            // Fallback for pos0 < 128 or non-128-aligned chunks: fused single-launch TP4 SWA-128!
                            launch_hca_tp4_swa128_merge_f16(
                                w.d_tmp_a, w.d_hca_kcomp_f16,
                                (max_comp > 0) ? w.d_hca_o_tot_f32 : nullptr,
                                (max_comp > 0) ? d_m_tot : nullptr,
                                (max_comp > 0) ? d_l_tot : nullptr,
                                H.d_attn_sink, w.d_tmp_b,
                                M, pos0, base_pos, w.stream);
                        } else {
                            const int swa_strip = std::min(128, 8192 / h_loc);
                            for (int m0 = 0; m0 < M; m0 += swa_strip) {
                                const int B          = std::min(swa_strip, M - m0);
                                const int B_eff      = std::min(swa_strip, ((B + 63) / 64) * 64);
                                const int n_rows     = B * h_loc;
                                const int n_rows_eff = B_eff * h_loc;
                                const size_t row_off = size_t(m0) * size_t(h_loc);
                                const int32_t q_pos0 = pos0 + m0;
                                const int32_t k_start_pos = std::max(0, int(q_pos0) - 127);
                                const int k_slot0    = k_start_pos - base_pos;
                                const uint16_t* d_k_win = w.d_hca_kcomp_f16 + size_t(k_slot0) * 512u;

                                const uint16_t* d_q_swa = w.d_x_f16;
                                if (q_stay_f16 && n_rows == n_rows_eff) {
                                    d_q_swa = w.d_tmp_a + row_off * 512u;
                                } else {
                                    if (q_stay_f16) {
                                        hip_check(hipMemcpyAsync(w.d_x_f16, w.d_tmp_a + row_off * 512u,
                                                                 size_t(n_rows) * 512u * sizeof(uint16_t), hipMemcpyDeviceToDevice, w.stream), "d2d swa q pad");
                                    } else {
                                        launch_bf16_to_f16(w.d_tmp_a + row_off * 512u, w.d_x_f16, n_rows * 512, w.stream);
                                    }
                                    if (n_rows < n_rows_eff) {
                                        hip_check(hipMemsetAsync(w.d_x_f16 + size_t(n_rows) * 512u, 0,
                                                                 size_t(n_rows_eff - n_rows) * 512u * sizeof(uint16_t), w.stream), "zero swa q pad");
                                    }
                                }
                                rocblas_gemm_ex_tuned(
                                    w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                    256, n_rows_eff, 512, &alpha_hca,
                                    d_k_win, rocblas_datatype_f16_r, 512,
                                    d_q_swa, rocblas_datatype_f16_r, 512,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, 256,
                                    d_y16, rocblas_datatype_f16_r, 256, "hca swa QK hh_hpa");
                                launch_hca_swa_softmax_f16_in_norm(
                                    d_y16, d_m_blk, d_l_blk, d_probs_swa,
                                    B, h_loc, q_pos0, k_start_pos, w.stream);
                                if (n_rows < n_rows_eff) {
                                    hip_check(hipMemsetAsync(d_probs_swa + size_t(n_rows) * 256u, 0,
                                                             size_t(n_rows_eff - n_rows) * 256u * sizeof(uint16_t), w.stream), "zero swa p pad");
                                }
                                rocblas_gemm_ex_tuned(
                                    w.blas, rocblas_operation_none, rocblas_operation_none,
                                    512, n_rows_eff, 256, &alpha,
                                    d_k_win,     rocblas_datatype_f16_r, 512,
                                    d_probs_swa, rocblas_datatype_f16_r, 256,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, 512,
                                    d_y16, rocblas_datatype_f16_r, 512, "hca swa PV hh_hpa");
                                if (vha_stay_f16) {
                                    launch_hca_swa_finalize_merge_sink_f16_out(
                                        d_y16, d_m_blk, d_l_blk,
                                        (max_comp > 0) ? w.d_hca_o_tot_f32 : nullptr,
                                        (max_comp > 0) ? d_m_tot : nullptr,
                                        (max_comp > 0) ? d_l_tot : nullptr,
                                        H.d_attn_sink, w.d_tmp_b, B, h_loc, int(row_off), w.stream);
                                } else {
                                    launch_hca_swa_finalize_merge_sink_f16(
                                        d_y16, d_m_blk, d_l_blk,
                                        (max_comp > 0) ? w.d_hca_o_tot_f32 : nullptr,
                                        (max_comp > 0) ? d_m_tot : nullptr,
                                        (max_comp > 0) ? d_l_tot : nullptr,
                                        H.d_attn_sink, w.d_tmp_b, B, h_loc, int(row_off), w.stream);
                                }
                            }
                        }
                    }
                    launch_hca_prefill_commit_tail(
                        w.d_kv_tmp, w.d_f32_tmp0, w.d_f32_tmp1, H.d_comp_ape,
                        pos0, M, L.d_hca_swa_ring_tp[r], L.d_hca_state_kv_tp[r], L.d_hca_state_sc_tp[r], w.stream);
                }
            }

            if (profiling_enabled || do_hca_bl_xray) {
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.hca_attn_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ha0).count();
                } else {
                    auto& go = *dev_map[L.owner_dev];
                    hip_check(hipSetDevice(go.hip_dev), "bl hca attn dev");
                    hip_check(hipStreamSynchronize(go.stream), "bl hca attn sync");
                    chunk_bk_attn += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ha0).count();
                }
            }

            // Phase B/C/D: Wave-10 C2 Zero-Bounce FP16-Stay VHA/wo_a/o_proj + 2-Microbatch Host-SDMA FP16 AllReduce Ping-Pong!
            // (100% host-pinned EPYC DDR4 SDMA — ZERO BAR1 P2P, ZERO extra streams, ZERO extra VRAM!)
            auto run_hca_wo_a_f16 = [&](Mi50ExpertWorker& w, const HcaAttnGpuWeights& H,
                                        const uint16_t* d_vha_f16, const uint16_t* d_gate_bf16,
                                        uint16_t* d_wo_f16, int mb_m) {
                const float alpha = 1.0f, beta = 0.0f;
                uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                // Wave-12 Cut P12.4: d_y16 holds ONLY output (mb_m * g_loc * kHcaOLoraRank = 4 MiB for 1024x4x512 << 32 MiB d_y_f32),
                // and d_x_f16 pad copy is only needed when mb_m is not 64-aligned -> 1 single 1024-row batched GEMM per MB!
                const int wo_strip = ((mb_m & 63) == 0 && size_t(g_loc) * size_t(kHcaOLoraRank) * 1024u <= 16777216u)
                    ? 1024
                    : std::min(1024, int((512u * 16384u) / (size_t(g_loc) * 4096u)));
                for (int m0 = 0; m0 < mb_m; m0 += wo_strip) {
                    const int cur = std::min(wo_strip, mb_m - m0);
                    const int eff = std::min(wo_strip, ((cur + 63) / 64) * 64);
                    const uint16_t* d_in = d_vha_f16 + size_t(m0) * size_t(g_loc) * 4096u;
                    if (cur < eff) {
                        hip_check(hipMemcpyAsync(w.d_x_f16, d_in,
                                                 size_t(cur) * size_t(g_loc) * 4096u * sizeof(uint16_t),
                                                 hipMemcpyDeviceToDevice, w.stream), "d2d wo_a f16 pad");
                        hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(g_loc) * 4096u, 0,
                                                 size_t(eff - cur) * size_t(g_loc) * 4096u * sizeof(uint16_t),
                                                 w.stream), "zero wo_a f16 pad");
                        d_in = w.d_x_f16;
                    }
                    // Wave-11 C2/C5: one tuned strided-batched GEMM across all g_loc groups (sol=32975)
                    rocblas_gemm_strided_batched_ex_tuned(
                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                        kHcaOLoraRank, eff, 4096, &alpha,
                        H.d_wo_a, rocblas_datatype_f16_r, 4096, int64_t(kHcaOLoraRank) * 4096LL,
                        d_in,     rocblas_datatype_f16_r, g_loc * 4096, 4096LL,
                        &beta,
                        d_y16, rocblas_datatype_f16_r, g_loc * kHcaOLoraRank, int64_t(kHcaOLoraRank),
                        d_y16, rocblas_datatype_f16_r, g_loc * kHcaOLoraRank, int64_t(kHcaOLoraRank),
                        g_loc, "hca wo_a f16 hh_hpa batched");
                    const size_t z_off = size_t(m0) * size_t(g_loc) * size_t(kHcaOLoraRank);
                    launch_f16_mul_sigmoid_f16_gate_to_f16(d_y16, d_gate_bf16 + z_off, d_wo_f16 + z_off,
                                                           cur * g_loc * kHcaOLoraRank, w.stream);
                }
            };

            if (vha_stay_f16 && M >= 256) {
                // Wave-11 C5: PCIe 4.0 HBM2-Reduced 4-MB AllReduce (0 CPU DDR4 sum4 stall, 0 B extra VRAM, bit-exact FP32 RNE!)
                // Fallback to CPU AVX2 path when ERNIE55_GPU_AR=0 or d_tmp_a headroom is insufficient.
                const auto t_ho0 = (profiling_enabled || M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                constexpr int kNb = 4;
                int mb_off[kNb]{}, mb_len[kNb]{};
                const int base_mb = (M / kNb) & ~63;
                for (int b = 0; b < kNb; ++b) {
                    mb_off[b] = b * base_mb;
                    mb_len[b] = (b + 1 < kNb) ? base_mb : (M - b * base_mb);
                }
                constexpr int kOrdH2d[4] = {2, 1, 3, 0};
                constexpr int kOrdD2h[4] = {2, 1, 3, 0};
                const size_t vha_cols  = 16u * 512u;
                const size_t gate_cols = size_t(g_loc) * size_t(kHcaOLoraRank);
                const size_t wo_base   = size_t(M) * vha_cols;
                const size_t wo_end    = wo_base + size_t(M) * gate_cols;
                const size_t mb_max_rows = size_t(std::max(mb_len[0], mb_len[kNb - 1]));
                const size_t mb_vha_max  = mb_max_rows * vha_cols;
                const size_t mb_op_max   = mb_max_rows * kHiddenSize;
                const size_t tmp_a_cap   = size_t(max_chunk) * 32768u;
                const bool defer_ok    = (defer_nonowner_oproj && !L.is_echo && tp == 4);
                const bool use_gpu_ar  = gfx906_gpu_ar_enabled() && !L.is_echo && (tp == 4) &&
                                         (L.owner_tp_rank >= 0 && L.owner_tp_rank < 4) &&
                                         (wo_end + 6u * mb_vha_max <= tmp_a_cap) &&
                                         (wo_end + 12u * mb_op_max <= tmp_a_cap);

                std::chrono::steady_clock::time_point t_ar0{};
                if (use_gpu_ar) {
                    // Wave-11 Cut P1.1: Dual-Owner 2 x 1024 Full-Duplex HBM2 VHA AllReduce across g0 (c7:00.0) & g1 (88:00.0)!
                    // Splits VHA 96 MiB H2D + 32 MiB D2H 50/50 across two independent 28 GB/s Gen4 x16 PCIe root complexes,
                    // eliminates g0 D2H serialization behind wo_a/o_proj, and executes wo_a + o_proj at N=1024 (sol=33065/32975)!
                    const int own_r = L.owner_tp_rank;
                    auto& g0 = *dev_map[dev_for_r(0)];
                    auto& g1 = *dev_map[dev_for_r(1)];
                    auto& g_own = *dev_map[L.owner_dev];
                    const int mb2_off[2] = { 0, (M / 2) & ~63 };
                    const int mb2_len[2] = { mb2_off[1], M - mb2_off[1] };
                    const size_t mb2_max_rows = size_t(std::max(mb2_len[0], mb2_len[1]));
                    const size_t mb2_vha_max  = mb2_max_rows * vha_cols;
                    const size_t mb2_op_max   = mb2_max_rows * kHiddenSize;
                    const bool dual_ok = (mb2_len[0] >= 128 && mb2_len[1] >= 128 &&
                                          wo_end + 3u * mb2_vha_max <= tmp_a_cap &&
                                          wo_end + 6u * mb2_op_max  <= tmp_a_cap);

                    if (dual_ok) {
                        const size_t voff0 = 0u, voff1 = size_t(mb2_off[1]) * vha_cols;
                        const size_t goff0 = 0u, goff1 = size_t(mb2_off[1]) * gate_cols;
                        const size_t ooff0 = 0u, ooff1 = size_t(mb2_off[1]) * kHiddenSize;
                        const size_t vha0_n = size_t(mb2_len[0]) * vha_cols, vha1_n = size_t(mb2_len[1]) * vha_cols;
                        const size_t sb_vha0 = vha0_n * sizeof(uint16_t), sb_vha1 = vha1_n * sizeof(uint16_t);
                        const size_t sb_op0  = size_t(mb2_len[0]) * kHiddenSize * sizeof(uint16_t);
                        const size_t sb_op1  = size_t(mb2_len[1]) * kHiddenSize * sizeof(uint16_t);
                        const size_t vha_red = wo_end;
                        const size_t op_red  = wo_end;

                        // Step 1: Slow-first VHA proj + immediate D2H on stream_sh (g0 keeps MB0 local & sends MB1; g1 keeps MB1 local & sends MB0!)
                        constexpr int kLaunchOrd[4] = {0, 1, 3, 2};
                        for (int oi = 0; oi < 4; ++oi) {
                            const int r = kLaunchOrd[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            const auto& H = L.hca_tp[r];
                            hip_check(hipSetDevice(w.hip_dev), "hca dual init dev");
                            uint16_t* h_base = h_tp_oattn_pin + size_t(r) * size_t(M) * vha_cols;
                            if (r == 0) {
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_b + voff1, H.d_vha_r_loc, w.d_tmp_a + voff1, mb2_len[1], 512, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[1], w.stream), "hca dual g0 c2d1");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[1], 0), "hca dual g0 wait1");
                                hip_check(hipMemcpyAsync(h_base + voff1, w.d_tmp_a + voff1, sb_vha1, hipMemcpyDeviceToHost, w.stream_sh), "hca dual g0 d2h1");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[1], w.stream_sh), "hca dual g0 d2h1 rec");
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, mb2_len[0], 512, w.stream);
                            } else if (r == 1) {
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, mb2_len[0], 512, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "hca dual g1 c2d0");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "hca dual g1 wait0");
                                hip_check(hipMemcpyAsync(h_base, w.d_tmp_a, sb_vha0, hipMemcpyDeviceToHost, w.stream_sh), "hca dual g1 d2h0");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "hca dual g1 d2h0 rec");
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_b + voff1, H.d_vha_r_loc, w.d_tmp_a + voff1, mb2_len[1], 512, w.stream);
                            } else {
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, mb2_len[0], 512, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "hca dual rem c2d0");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "hca dual rem wait0");
                                hip_check(hipMemcpyAsync(h_base, w.d_tmp_a, sb_vha0, hipMemcpyDeviceToHost, w.stream_sh), "hca dual rem d2h0");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "hca dual rem d2h0 rec");
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_b + voff1, H.d_vha_r_loc, w.d_tmp_a + voff1, mb2_len[1], 512, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[1], w.stream), "hca dual rem c2d1");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[1], 0), "hca dual rem wait1");
                                hip_check(hipMemcpyAsync(h_base + voff1, w.d_tmp_a + voff1, sb_vha1, hipMemcpyDeviceToHost, w.stream_sh), "hca dual rem d2h1");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[1], w.stream_sh), "hca dual rem d2h1 rec");
                            }
                            f16_linear_f16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, int(gate_cols), 1024);
                        }

                        // Step 2: Fast-first host hipEventSynchronize -> CONCURRENT H2D + HBM2 sum4 + D2H on g0 (MB0) & g1 (MB1)!
                        // (Host hipEventSynchronize before every cross-RC H2D guarantees EPYC I/O-die DDR4 drain!)
                        // Wave-11 Turn-4 Cut B: DEVICE-DRIVEN VHA relay. The 4 rank partials live in
                        // hipHostRegisterMapped pinned pages, so the owner's sum4 kernel reads them
                        // directly over PCIe and only a DEVICE-SIDE hipStreamWaitEvent orders
                        // D2H -> sum4. The host never blocks (was 6 hipEventSynchronize + 6 H2D per
                        // layer = 11.2 ms/L of the 415 ms hca_ar), and the DDR4 round trip is halved.
                        const int ar_mode = gfx906_ar_device_mode();
                        const bool ar_dev = (ar_mode == 1) &&
                                            (d_tp_oattn_pin != nullptr && d_tp_qabs_pin != nullptr &&
                                             d_tp_ar_in_pin[0] != nullptr && d_tp_ar_in_pin[1] != nullptr &&
                                             d_tp_ar_in_pin[2] != nullptr && d_tp_ar_in_pin[3] != nullptr);
                        const bool ar_sdma = (ar_mode == 2);
                        auto relay_vha = [&](int src_r, int mb, Mi50ExpertWorker& dst_g) {
                            auto& ws = *dev_map[dev_for_r(src_r)];
                            hip_check(hipSetDevice(dst_g.hip_dev), "hca dual relay dev");
                            hip_check(hipStreamWaitEvent(dst_g.stream, ws.ev_mb_vha_d2h[mb], 0), "hca dual relay wait");
                        };
                        auto sync_and_h2d = [&](int src_r, int mb, Mi50ExpertWorker& dst_g, size_t dst_slot_idx) {
                            auto& ws = *dev_map[dev_for_r(src_r)];
                            const size_t vo = (mb == 0) ? voff0 : voff1;
                            const size_t sb = (mb == 0) ? sb_vha0 : sb_vha1;
                            if (ar_sdma) {
                                hip_check(hipSetDevice(dst_g.hip_dev), "hca dual h2d dst");
                                hip_check(hipStreamWaitEvent(dst_g.stream_sh, ws.ev_mb_vha_d2h[mb], 0), "hca dual sdma wait");
                            } else {
                                hip_check(hipSetDevice(ws.hip_dev), "hca dual sync src");
                                hip_check(hipEventSynchronize(ws.ev_mb_vha_d2h[mb]), "hca dual sync d2h");
                                hip_check(hipSetDevice(dst_g.hip_dev), "hca dual h2d dst");
                            }
                            uint16_t* h_in = h_tp_oattn_pin + size_t(src_r) * size_t(M) * vha_cols + vo;
                            hip_check(hipMemcpyAsync(dst_g.d_tmp_a + vha_red + dst_slot_idx * mb2_vha_max, h_in,
                                                     sb, hipMemcpyHostToDevice, dst_g.stream_sh), "hca dual h2d");
                        };
                        // Fast Gen4 peers first across BOTH g0 and g1 before syncing 8 GT/s r=2!
                        if (ar_dev) { relay_vha(1, 0, g0); relay_vha(0, 1, g1); }
                        else { sync_and_h2d(1, 0, g0, 0u); sync_and_h2d(0, 1, g1, 0u); }
                        if (M >= 256) {
                            t_ar0 = std::chrono::steady_clock::now();
                            chunk_hca_a_ms += std::chrono::duration<double, std::milli>(t_ar0 - t_ho0).count();
                        }
                        if (ar_dev) { relay_vha(3, 0, g0); relay_vha(3, 1, g1); relay_vha(2, 0, g0); }
                        else { sync_and_h2d(3, 0, g0, 2u); sync_and_h2d(3, 1, g1, 2u); sync_and_h2d(2, 0, g0, 1u); }
                        // Launch g0 MB0 canonical sum4 + immediate stream_sh D2H, then deferred gate_proj + wo_a/o_proj(MB0)!
                        {
                            hip_check(hipSetDevice(g0.hip_dev), "hca dual g0 sum dev");
                            if (ar_dev) {
                                // p0 = own device partial (no PCIe read), p1..p3 = mapped host partials
                                // of ranks 1..3 in strict rank order (bit-exact canonical FP32 order).
                                launch_sum4_f16_canonical(
                                    g0.d_tmp_a,
                                    d_tp_oattn_pin + size_t(1) * size_t(M) * vha_cols + voff0,
                                    d_tp_oattn_pin + size_t(2) * size_t(M) * vha_cols + voff0,
                                    d_tp_oattn_pin + size_t(3) * size_t(M) * vha_cols + voff0,
                                    g0.d_tmp_a, vha0_n, g0.stream);
                            } else {
                            hip_check(hipEventRecord(g0.ev_mb_vha_h2d[0], g0.stream_sh), "hca dual g0 h2d rec");
                            hip_check(hipStreamWaitEvent(g0.stream, g0.ev_mb_vha_h2d[0], 0), "hca dual g0 wait h2d");
                            launch_sum4_f16_canonical(
                                g0.d_tmp_a,
                                g0.d_tmp_a + vha_red + 0u * mb2_vha_max,
                                g0.d_tmp_a + vha_red + 1u * mb2_vha_max,
                                g0.d_tmp_a + vha_red + 2u * mb2_vha_max,
                                g0.d_tmp_a, vha0_n, g0.stream);
                            }
                            hip_check(hipEventRecord(g0.ev_mb_c2d[0], g0.stream), "hca dual g0 sum rec");
                            hip_check(hipStreamWaitEvent(g0.stream_sh, g0.ev_mb_c2d[0], 0), "hca dual g0 sh wait sum");
                            hip_check(hipMemcpyAsync(h_tp_qabs_pin, g0.d_tmp_a, sb_vha0, hipMemcpyDeviceToHost, g0.stream_sh), "hca dual g0 sum d2h");
                            hip_check(hipEventRecord(g0.ev_mb_vha_d2h[0], g0.stream_sh), "hca dual g0 sum d2h rec");
                            const auto& H0 = L.hca_tp[0];
                            launch_hca_vha_lowrank_invrope_f16_io(g0.d_tmp_b, g0.d_tmp_a, H0.d_vha_q_loc,
                                                                  g0.d_pos_ids, g0.yarn_rope, g0.d_tmp_b, mb2_len[0], g0.stream);
                            uint16_t* d_wo0 = g0.d_tmp_a + wo_base;
                            run_hca_wo_a_f16(g0, H0, g0.d_tmp_b, g0.d_tmp_d, d_wo0, mb2_len[0]);
                            f16_linear_f16(g0, d_wo0, H0.d_o_proj, g0.d_sub_out, mb2_len[0], kHiddenSize, int(gate_cols));
                            hip_check(hipEventRecord(g0.ev_mb_d2c[0], g0.stream), "hca dual g0 op0 rec");
                        }
                        if (ar_dev) relay_vha(2, 1, g1);
                        else        sync_and_h2d(2, 1, g1, 1u);
                        // Launch g1 MB1 canonical sum4 (exact p0,p1,p2,p3 order!) + stream_sh D2H, then deferred gate_proj + wo_a/o_proj(MB1)!
                        {
                            hip_check(hipSetDevice(g1.hip_dev), "hca dual g1 sum dev");
                            if (ar_dev) {
                                launch_sum4_f16_canonical(
                                    d_tp_oattn_pin + size_t(0) * size_t(M) * vha_cols + voff1,
                                    g1.d_tmp_a + voff1,
                                    d_tp_oattn_pin + size_t(2) * size_t(M) * vha_cols + voff1,
                                    d_tp_oattn_pin + size_t(3) * size_t(M) * vha_cols + voff1,
                                    g1.d_tmp_a + voff1, vha1_n, g1.stream);
                            } else {
                            hip_check(hipEventRecord(g1.ev_mb_vha_h2d[1], g1.stream_sh), "hca dual g1 h2d rec");
                            hip_check(hipStreamWaitEvent(g1.stream, g1.ev_mb_vha_h2d[1], 0), "hca dual g1 wait h2d");
                            launch_sum4_f16_canonical(
                                g1.d_tmp_a + vha_red + 0u * mb2_vha_max,
                                g1.d_tmp_a + voff1,
                                g1.d_tmp_a + vha_red + 1u * mb2_vha_max,
                                g1.d_tmp_a + vha_red + 2u * mb2_vha_max,
                                g1.d_tmp_a + voff1, vha1_n, g1.stream);
                            }
                            hip_check(hipEventRecord(g1.ev_mb_c2d[1], g1.stream), "hca dual g1 sum rec");
                            hip_check(hipStreamWaitEvent(g1.stream_sh, g1.ev_mb_c2d[1], 0), "hca dual g1 sh wait sum");
                            hip_check(hipMemcpyAsync(h_tp_qabs_pin + voff1, g1.d_tmp_a + voff1, sb_vha1, hipMemcpyDeviceToHost, g1.stream_sh), "hca dual g1 sum d2h");
                            hip_check(hipEventRecord(g1.ev_mb_vha_d2h[1], g1.stream_sh), "hca dual g1 sum d2h rec");
                            const auto& H1 = L.hca_tp[1];
                            launch_hca_vha_lowrank_invrope_f16_io(g1.d_tmp_b + voff1, g1.d_tmp_a + voff1, H1.d_vha_q_loc,
                                                                  g1.d_pos_ids + mb2_off[1], g1.yarn_rope, g1.d_tmp_b + voff1, mb2_len[1], g1.stream);
                            uint16_t* d_wo1 = g1.d_tmp_a + wo_base + goff1;
                            run_hca_wo_a_f16(g1, H1, g1.d_tmp_b + voff1, g1.d_tmp_d + goff1, d_wo1, mb2_len[1]);
                            f16_linear_f16(g1, d_wo1, H1.d_o_proj, g1.d_sub_out + ooff1, mb2_len[1], kHiddenSize, int(gate_cols));
                            hip_check(hipEventRecord(g1.ev_mb_d2c[1], g1.stream), "hca dual g1 op1 rec");
                        }

                        // Step 3: Full-duplex overlapped bcast MB0 & MB1 H2D on w.stream_sh behind MB0 compute on w.stream,
                        // followed by non-owner o_proj D2Hs (0 inter-MB H2D stall on 8 GT/s r=2 & Gen4 r=3!)
                        auto launch_bcast_mb = [&](int mb) {
                            auto& red_g = (mb == 0) ? g0 : g1;
                            const int red_r = (mb == 0) ? 0 : 1;
                            const size_t vo = (mb == 0) ? voff0 : voff1;
                            const size_t go = (mb == 0) ? goff0 : goff1;
                            const size_t oo = (mb == 0) ? ooff0 : ooff1;
                            const size_t sb_v = (mb == 0) ? sb_vha0 : sb_vha1;
                            if (!ar_dev && !ar_sdma) {
                                hip_check(hipSetDevice(red_g.hip_dev), "hca dual red sync dev");
                                hip_check(hipEventSynchronize(red_g.ev_mb_vha_d2h[mb]), "hca dual red sync");
                            }
                            const int targets[3] = { 2, 3, 1 - red_r };
                            for (int ti = 0; ti < 3; ++ti) {
                                const int r = targets[ti];
                                auto& w = *dev_map[dev_for_r(r)];
                                const auto& H = L.hca_tp[r];
                                hip_check(hipSetDevice(w.hip_dev), "hca dual tgt dev");
                                uint16_t* d_vb = w.d_tmp_b + vo;
                                if (ar_dev) {
                                    // Turn-4 Cut B: the reduced VHA half-buffer is read straight out of the
                                    // owner's mapped pinned pages; only a device-side wait orders the D2H.
                                    hip_check(hipStreamWaitEvent(w.stream, red_g.ev_mb_vha_d2h[mb], 0), "hca dual tgt r wait");
                                    launch_hca_vha_lowrank_invrope_f16_io(d_vb, d_tp_qabs_pin + vo, H.d_vha_q_loc,
                                                                          w.d_pos_ids + mb2_off[mb], w.yarn_rope, d_vb, mb2_len[mb], w.stream);
                                } else {
                                    if (ar_sdma) {
                                        hip_check(hipStreamWaitEvent(w.stream_sh, red_g.ev_mb_vha_d2h[mb], 0), "hca dual tgt sdma wait");
                                    }
                                    hip_check(hipMemcpyAsync(w.d_tmp_a + vo, h_tp_qabs_pin + vo, sb_v, hipMemcpyHostToDevice, w.stream_sh), "hca dual tgt h2d sh");
                                    hip_check(hipEventRecord(w.ev_mb_vha_h2d[mb], w.stream_sh), "hca dual tgt h2d rec");
                                    hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_vha_h2d[mb], 0), "hca dual tgt wait h2d");
                                    launch_hca_vha_lowrank_invrope_f16_io(d_vb, w.d_tmp_a + vo, H.d_vha_q_loc,
                                                                          w.d_pos_ids + mb2_off[mb], w.yarn_rope, d_vb, mb2_len[mb], w.stream);
                                }
                                uint16_t* d_wo = w.d_tmp_a + wo_base + go;
                                run_hca_wo_a_f16(w, H, d_vb, w.d_tmp_d + go, d_wo, mb2_len[mb]);
                                f16_linear_f16(w, d_wo, H.d_o_proj, w.d_sub_out + oo, mb2_len[mb], kHiddenSize, int(gate_cols));
                                hip_check(hipEventRecord(w.ev_mb_d2c[mb], w.stream), "hca dual tgt op rec");
                            }
                        };
                        launch_bcast_mb(0);
                        launch_bcast_mb(1);
                        // Queue non-owner o_proj D2Hs strictly AFTER both VHA H2Ds on w.stream_sh (in execution order!)
                        for (int r = 0; r < 4; ++r) {
                            if (r == own_r) continue;
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca dual op d2h q dev");
                            const int mb_ord[2] = { (r == 1) ? 1 : 0, (r == 1) ? 0 : 1 };
                            for (int mi = 0; mi < 2; ++mi) {
                                const int mb = mb_ord[mi];
                                const size_t oo = (mb == 0) ? ooff0 : ooff1;
                                const size_t sb_o = (mb == 0) ? sb_op0 : sb_op1;
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[mb], 0), "hca dual tgt op wait");
                                hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + oo, w.d_sub_out + oo,
                                                         sb_o, hipMemcpyDeviceToHost, w.stream_sh), "hca dual tgt op d2h");
                                hip_check(hipEventRecord(w.ev_mb_o_d2h[mb], w.stream_sh), "hca dual tgt op d2h rec");
                            }
                        }

                        // Step 4: Owner-GPU HBM2 o_proj Reduce across b = 0, 1 + single contiguous h_tp_o_out_pin D2H!
                        // On r=1 (g1), MB1 executes first and MB0 second -> sync r=1 LAST on b=0 and FIRST on b=1!
                        constexpr int kOrdFastSyncB[2][4] = {{0, 3, 2, 1}, {1, 0, 3, 2}};
                        int non_own[3];
                        int ni = 0;
                        for (int oi = 0; oi < 4; ++oi) {
                            const int r = kOrdFastSyncB[0][oi];
                            if (r != own_r) non_own[ni++] = r;
                        }
                        for (int b = 0; b < 2; ++b) {
                            int non_own_b[3];
                            int nib = 0;
                            for (int oi = 0; oi < 4; ++oi) {
                                const int r = kOrdFastSyncB[b][oi];
                                if (r != own_r) non_own_b[nib++] = r;
                            }
                            const size_t ooff  = size_t(mb2_off[b]) * kHiddenSize;
                            const size_t mb_op = size_t(mb2_len[b]) * kHiddenSize;
                            const size_t sb_op = mb_op * sizeof(uint16_t);
                            const size_t op_slot_base = op_red + size_t(b) * 3u * mb2_op_max;
                            const uint16_t* p_rank[4] = {nullptr, nullptr, nullptr, nullptr};
                            p_rank[own_r] = g_own.d_sub_out + ooff;
                            for (int i = 0; i < 3; ++i) {
                                const int r = non_own_b[i];
                                auto& w = *dev_map[dev_for_r(r)];
                                if (ar_dev) {
                                    // Turn-4 Cut B: read the non-owner o_proj partial directly from its
                                    // mapped pinned pages (device-side wait, 0 host block, 0 H2D).
                                    hip_check(hipSetDevice(g_own.hip_dev), "hca dual op red dev");
                                    hip_check(hipStreamWaitEvent(g_own.stream, w.ev_mb_o_d2h[b], 0), "hca dual op r wait");
                                    p_rank[r] = d_tp_ar_in_pin[r] + ooff;
                                } else {
                                if (ar_sdma) {
                                    hip_check(hipSetDevice(g_own.hip_dev), "hca dual op h2d dev");
                                    hip_check(hipStreamWaitEvent(g_own.stream_sh, w.ev_mb_o_d2h[b], 0), "hca dual op sdma wait");
                                } else {
                                    hip_check(hipSetDevice(w.hip_dev), "hca dual op sync dev");
                                    hip_check(hipEventSynchronize(w.ev_mb_o_d2h[b]), "hca dual op sync");
                                    hip_check(hipSetDevice(g_own.hip_dev), "hca dual op h2d dev");
                                }
                                uint16_t* dst_slot = g_own.d_tmp_a + op_slot_base + size_t(i) * mb2_op_max;
                                hip_check(hipMemcpyAsync(dst_slot, h_tp_ar_in_pin[r] + ooff,
                                                         sb_op, hipMemcpyHostToDevice, g_own.stream_sh), "hca dual op h2d");
                                p_rank[r] = dst_slot;
                                }
                            }
                            hip_check(hipSetDevice(g_own.hip_dev), "hca dual op red dev");
                            if (!ar_dev) {
                                hip_check(hipEventRecord(g_own.ev_mb_o_d2h[b], g_own.stream_sh), "hca dual op h2d rec");
                                hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_o_d2h[b], 0), "hca dual op sum wait");
                            }
                            launch_sum4_f16_to_bf16_canonical(p_rank[0], p_rank[1], p_rank[2], p_rank[3],
                                                              g_own.d_sub_out + ooff, mb_op, g_own.stream);
                        }
                        hip_check(hipSetDevice(g_own.hip_dev), "hca dual op host dev");
                        hip_check(hipEventRecord(g_own.ev_mb_c2d[3], g_own.stream), "hca dual op sum rec");
                        hip_check(hipStreamWaitEvent(g_own.stream_sh, g_own.ev_mb_c2d[3], 0), "hca dual op host wait");
                        hip_check(hipMemcpyAsync(h_tp_o_out_pin, g_own.d_sub_out,
                                                 size_t(M) * kHiddenSize * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, g_own.stream_sh), "hca dual op host d2h");
                        hip_check(hipEventRecord(g_own.ev_mb_vha_d2h[3], g_own.stream_sh), "hca dual op host done");

                        if (defer_ok) {
                            o_proj_host_d2h_pending = true;
                            if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = true;
                        } else {
                            hip_check(hipSetDevice(g_own.hip_dev), "hca dual nondef sync dev");
                            hip_check(hipEventSynchronize(g_own.ev_mb_vha_d2h[3]), "hca dual nondef sync");
                            const size_t full_op_b = size_t(M) * kHiddenSize * sizeof(uint16_t);
                            for (int i = 0; i < 3; ++i) {
                                const int r = non_own[i];
                                auto& w = *dev_map[dev_for_r(r)];
                                hip_check(hipSetDevice(w.hip_dev), "hca dual nondef h2d dev");
                                hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_o_out_pin, full_op_b,
                                                         hipMemcpyHostToDevice, w.stream_sh), "hca dual nondef h2d");
                                hip_check(hipEventRecord(w.ev_mb_d2c[3], w.stream_sh), "hca dual nondef rec");
                                hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[3], 0), "hca dual nondef wait");
                            }
                        }
                    } else {
                    constexpr int kRemLaunch[3] = {1, 3, 2}; // Turn-16: fast-first remote ranks (excludes g0=0)
                    constexpr int kRemSync[3]   = {1, 3, 2}; // fast-first sync so g0 starts H2D on 16 GT/s ranks while r=2 finishes
                    const size_t vha_red_base[2] = { wo_end, wo_end + 3u * mb_vha_max };
                    const size_t op_red_base     = wo_end;

                    // Init: remotes launch MB0 VHA proj -> immediate D2H on stream_sh; GPU0 runs MB0..3 VHA proj + gate_proj(M);
                    // then remotes queue MB1..3 VHA proj + ALL remaining VHA D2H(1..3) back-to-back on stream_sh (0 inter-MB gaps)!
                    for (int ri = 0; ri < 3; ++ri) {
                        const int r = kRemLaunch[ri];
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& H = L.hca_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "hca gar init0 dev");
                        launch_vha_lowrank_proj_f16_in(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, mb_len[0], 512, w.stream);
                        hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "hca gar c2d0");
                        hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "hca gar wait0");
                        uint16_t* h_in0 = h_tp_oattn_pin + size_t(r) * size_t(M) * vha_cols;
                        hip_check(hipMemcpyAsync(h_in0, w.d_tmp_a, size_t(mb_len[0]) * vha_cols * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, w.stream_sh), "hca gar vha0 d2h");
                        hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "hca gar vha0 rec");
                    }
                    {
                        const auto& H0 = L.hca_tp[0];
                        hip_check(hipSetDevice(g0.hip_dev), "hca gar g0 init dev");
                        for (int b = 0; b < kNb; ++b) {
                            const size_t voff = size_t(mb_off[b]) * vha_cols;
                            launch_vha_lowrank_proj_f16_in(g0.d_tmp_b + voff, H0.d_vha_r_loc, g0.d_tmp_a + voff,
                                                           mb_len[b], 512, g0.stream);
                        }
                        f16_linear_f16(g0, g0.d_qr, H0.d_gate_proj, g0.d_tmp_d, M, int(gate_cols), 1024);
                    }
                    for (int ri = 0; ri < 3; ++ri) {
                        const int r = kRemLaunch[ri];
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& H = L.hca_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "hca gar init1 dev");
                        for (int b = 1; b < kNb; ++b) {
                            const size_t voff = size_t(mb_off[b]) * vha_cols;
                            launch_vha_lowrank_proj_f16_in(w.d_tmp_b + voff, H.d_vha_r_loc, w.d_tmp_a + voff,
                                                           mb_len[b], 512, w.stream);
                            hip_check(hipEventRecord(w.ev_mb_c2d[b], w.stream), "hca gar c2d b");
                            hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[b], 0), "hca gar wait b");
                            uint16_t* h_in_b = h_tp_oattn_pin + size_t(r) * size_t(M) * vha_cols + voff;
                            hip_check(hipMemcpyAsync(h_in_b, w.d_tmp_a + voff,
                                                     size_t(mb_len[b]) * vha_cols * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, w.stream_sh), "hca gar vha b d2h");
                            hip_check(hipEventRecord(w.ev_mb_vha_d2h[b], w.stream_sh), "hca gar vha b rec");
                        }
                        f16_linear_f16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, int(gate_cols), 1024);
                    }

                    // Helper: sync GPU0 reduced VHA D2H(bm) -> launch remotes H2D + invrope + wo_a + o_proj + early o_proj D2H!
                    auto launch_rem_mb = [&](int bm) {
                        const size_t voff_m = size_t(mb_off[bm]) * vha_cols;
                        const size_t goff_m = size_t(mb_off[bm]) * gate_cols;
                        const size_t ooff_m = size_t(mb_off[bm]) * kHiddenSize;
                        const size_t sb_vha_m = size_t(mb_len[bm]) * vha_cols * sizeof(uint16_t);
                        const size_t sb_op_m  = size_t(mb_len[bm]) * kHiddenSize * sizeof(uint16_t);
                        hip_check(hipSetDevice(g0.hip_dev), "hca gar g0 sum sync dev");
                        hip_check(hipEventSynchronize(g0.ev_mb_vha_d2h[bm]), "hca gar g0 sum sync");
                        for (int ri = 0; ri < 3; ++ri) {
                            const int r = kRemLaunch[ri];
                            auto& w = *dev_map[dev_for_r(r)];
                            const auto& H = L.hca_tp[r];
                            hip_check(hipSetDevice(w.hip_dev), "hca gar rem comp dev");
                            hip_check(hipMemcpyAsync(w.d_tmp_a + voff_m, h_tp_qabs_pin + voff_m,
                                                     sb_vha_m, hipMemcpyHostToDevice, w.stream), "hca gar rem h2d");
                            uint16_t* d_vha_b = w.d_tmp_b + voff_m;
                            launch_hca_vha_lowrank_invrope_f16_io(d_vha_b, w.d_tmp_a + voff_m, H.d_vha_q_loc,
                                                                  w.d_pos_ids + mb_off[bm], w.yarn_rope, d_vha_b, mb_len[bm], w.stream);
                            uint16_t* d_wo_b = w.d_tmp_a + wo_base + goff_m;
                            run_hca_wo_a_f16(w, H, d_vha_b, w.d_tmp_d + goff_m, d_wo_b, mb_len[bm]);
                            f16_linear_f16(w, d_wo_b, H.d_o_proj, w.d_sub_out + ooff_m, mb_len[bm], kHiddenSize, int(gate_cols));
                            hip_check(hipEventRecord(w.ev_mb_d2c[bm], w.stream), "hca gar rem o comp rec");
                            if (r != own_r) {
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[bm], 0), "hca gar op wait");
                                hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + ooff_m, w.d_sub_out + ooff_m,
                                                         sb_op_m, hipMemcpyDeviceToHost, w.stream_sh), "hca gar op d2h");
                                hip_check(hipEventRecord(w.ev_mb_o_d2h[bm], w.stream_sh), "hca gar op rec");
                            }
                        }
                    };

                    // 4-MB Full-Duplex GPU0 HBM2 VHA AllReduce: g0.stream_sh runs H2D(b) concurrently with g0.stream D2H(b-1)!
                    for (int b = 0; b < kNb; ++b) {
                        const size_t voff = size_t(mb_off[b]) * vha_cols;
                        const size_t goff = size_t(mb_off[b]) * gate_cols;
                        const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                        const size_t mb_vha = size_t(mb_len[b]) * vha_cols;
                        const size_t sb_vha = mb_vha * sizeof(uint16_t);
                        const size_t slot_base = vha_red_base[b & 1];

                        if (b >= 2) {
                            hip_check(hipSetDevice(g0.hip_dev), "hca gar g0 slot wait dev");
                            hip_check(hipStreamWaitEvent(g0.stream_sh, g0.ev_mb_c2d[b - 2], 0), "hca gar g0 slot wait");
                        }
                        for (int ri = 0; ri < 3; ++ri) {
                            const int r = kRemSync[ri];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca gar sync d2h dev");
                            hip_check(hipEventSynchronize(w.ev_mb_vha_d2h[b]), "hca gar sync vha d2h");
                            hip_check(hipSetDevice(g0.hip_dev), "hca gar g0 h2d dev");
                            uint16_t* h_in_r = h_tp_oattn_pin + size_t(r) * size_t(M) * vha_cols + voff;
                            hip_check(hipMemcpyAsync(g0.d_tmp_a + slot_base + size_t(r - 1) * mb_vha_max, h_in_r,
                                                     sb_vha, hipMemcpyHostToDevice, g0.stream_sh), "hca gar g0 h2d");
                        }
                        if (b == 0 && M >= 256) {
                            t_ar0 = std::chrono::steady_clock::now();
                            chunk_hca_a_ms += std::chrono::duration<double, std::milli>(t_ar0 - t_ho0).count();
                        }

                        hip_check(hipSetDevice(g0.hip_dev), "hca gar g0 red dev");
                        hip_check(hipEventRecord(g0.ev_mb_vha_h2d[b], g0.stream_sh), "hca gar g0 h2d rec");
                        hip_check(hipStreamWaitEvent(g0.stream, g0.ev_mb_vha_h2d[b], 0), "hca gar g0 wait h2d");
                        launch_sum4_f16_canonical(
                            g0.d_tmp_a + voff,
                            g0.d_tmp_a + slot_base + 0u * mb_vha_max,
                            g0.d_tmp_a + slot_base + 1u * mb_vha_max,
                            g0.d_tmp_a + slot_base + 2u * mb_vha_max,
                            g0.d_tmp_a + voff, mb_vha, g0.stream);
                        hip_check(hipEventRecord(g0.ev_mb_c2d[b], g0.stream), "hca gar g0 sum rec");
                        hip_check(hipMemcpyAsync(h_tp_qabs_pin + voff, g0.d_tmp_a + voff,
                                                 sb_vha, hipMemcpyDeviceToHost, g0.stream), "hca gar g0 sum d2h");
                        hip_check(hipEventRecord(g0.ev_mb_vha_d2h[b], g0.stream), "hca gar g0 sum d2h rec");

                        // GPU0 runs invrope + wo_a + o_proj (~0.45 ms) inside g0.stream_sh's 0.87 ms H2D(b+1) shadow!
                        {
                            const auto& H0 = L.hca_tp[0];
                            uint16_t* d_vha_b = g0.d_tmp_b + voff;
                            launch_hca_vha_lowrank_invrope_f16_io(d_vha_b, g0.d_tmp_a + voff, H0.d_vha_q_loc,
                                                                  g0.d_pos_ids + mb_off[b], g0.yarn_rope, d_vha_b, mb_len[b], g0.stream);
                            uint16_t* d_wo_b = g0.d_tmp_a + wo_base + goff;
                            run_hca_wo_a_f16(g0, H0, d_vha_b, g0.d_tmp_d + goff, d_wo_b, mb_len[b]);
                            f16_linear_f16(g0, d_wo_b, H0.d_o_proj, g0.d_sub_out + ooff, mb_len[b], kHiddenSize, int(gate_cols));
                            hip_check(hipEventRecord(g0.ev_mb_d2c[b], g0.stream), "hca gar g0 o comp rec");
                        }

                        // Sync b-1 AFTER queueing g0 H2D(b) -> 100% full-duplex H2D(b) || D2H(b-1) on g0!
                        if (b >= 1) launch_rem_mb(b - 1);
                    }
                    // On L22..L42 (own_r != 0), immediately stream g0's already-completed o_proj(0..3) on g0.stream_sh!
                    if (own_r != 0) {
                        hip_check(hipSetDevice(g0.hip_dev), "hca gar g0 op d2h dev");
                        for (int bp = 0; bp < kNb; ++bp) {
                            const size_t ooff_p = size_t(mb_off[bp]) * kHiddenSize;
                            const size_t sb_op  = size_t(mb_len[bp]) * kHiddenSize * sizeof(uint16_t);
                            hip_check(hipStreamWaitEvent(g0.stream_sh, g0.ev_mb_d2c[bp], 0), "hca gar g0 op wait");
                            hip_check(hipMemcpyAsync(h_tp_ar_in_pin[0] + ooff_p, g0.d_sub_out + ooff_p,
                                                     sb_op, hipMemcpyDeviceToHost, g0.stream_sh), "hca gar g0 op d2h");
                            hip_check(hipEventRecord(g0.ev_mb_o_d2h[bp], g0.stream_sh), "hca gar g0 op rec");
                        }
                    }
                    launch_rem_mb(kNb - 1);

                    // Owner-GPU HBM2 o_proj Reduce: fast-first sync + pure back-to-back H2D on g_own.stream_sh ->
                    // single contiguous 16 MiB h_tp_o_out_pin D2H AFTER b=3 (100% hidden behind moe_pre!)
                    constexpr int kOrdFastSync[4] = {0, 1, 3, 2};
                    int non_own[3];
                    int ni = 0;
                    for (int oi = 0; oi < 4; ++oi) {
                        const int r = kOrdFastSync[oi];
                        if (r != own_r) non_own[ni++] = r;
                    }
                    for (int b = 0; b < kNb; ++b) {
                        const size_t ooff  = size_t(mb_off[b]) * kHiddenSize;
                        const size_t mb_op = size_t(mb_len[b]) * kHiddenSize;
                        const size_t sb_op = mb_op * sizeof(uint16_t);
                        const size_t op_slot_base = op_red_base + size_t(b) * 3u * mb_op_max;
                        const uint16_t* p_rank[4] = {nullptr, nullptr, nullptr, nullptr};
                        p_rank[own_r] = g_own.d_sub_out + ooff;
                        for (int i = 0; i < 3; ++i) {
                            const int r = non_own[i];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca gar op sync dev");
                            hip_check(hipEventSynchronize(w.ev_mb_o_d2h[b]), "hca gar op sync");
                            hip_check(hipSetDevice(g_own.hip_dev), "hca gar op h2d dev");
                            uint16_t* dst_slot = g_own.d_tmp_a + op_slot_base + size_t(i) * mb_op_max;
                            hip_check(hipMemcpyAsync(dst_slot, h_tp_ar_in_pin[r] + ooff,
                                                     sb_op, hipMemcpyHostToDevice, g_own.stream_sh), "hca gar op h2d");
                            p_rank[r] = dst_slot;
                        }
                        hip_check(hipSetDevice(g_own.hip_dev), "hca gar op red dev");
                        hip_check(hipEventRecord(g_own.ev_mb_o_d2h[b], g_own.stream_sh), "hca gar op h2d rec");
                        hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_o_d2h[b], 0), "hca gar op sum wait");
                        launch_sum4_f16_to_bf16_canonical(p_rank[0], p_rank[1], p_rank[2], p_rank[3],
                                                          g_own.d_sub_out + ooff, mb_op, g_own.stream);
                    }
                    hip_check(hipSetDevice(g_own.hip_dev), "hca gar op host dev");
                    hip_check(hipEventRecord(g_own.ev_mb_c2d[3], g_own.stream), "hca gar op sum3 rec");
                    hip_check(hipStreamWaitEvent(g_own.stream_sh, g_own.ev_mb_c2d[3], 0), "hca gar op host wait");
                    hip_check(hipMemcpyAsync(h_tp_o_out_pin, g_own.d_sub_out,
                                             size_t(M) * kHiddenSize * sizeof(uint16_t),
                                             hipMemcpyDeviceToHost, g_own.stream_sh), "hca gar op host d2h");
                    hip_check(hipEventRecord(g_own.ev_mb_vha_d2h[3], g_own.stream_sh), "hca gar op host done");

                    if (defer_ok) {
                        o_proj_host_d2h_pending = true;
                        if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = true;
                    } else {
                        hip_check(hipSetDevice(g_own.hip_dev), "hca gar nondef sync dev");
                        hip_check(hipEventSynchronize(g_own.ev_mb_vha_d2h[3]), "hca gar nondef sync");
                        const size_t full_op_b = size_t(M) * kHiddenSize * sizeof(uint16_t);
                        for (int i = 0; i < 3; ++i) {
                            const int r = non_own[i];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca gar nondef h2d dev");
                            hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_o_out_pin, full_op_b,
                                                     hipMemcpyHostToDevice, w.stream_sh), "hca gar nondef h2d");
                            hip_check(hipEventRecord(w.ev_mb_d2c[3], w.stream_sh), "hca gar nondef rec");
                            hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[3], 0), "hca gar nondef wait");
                        }
                    }
                    } // !dual_ok
                } else {
                // Init: MB0 VHA proj -> immediate MB0 VHA D2H on stream_sh; MB1..3 VHA proj + gate_proj(M) overlapped on stream!
                for (int oi = 0; oi < tp; ++oi) {
                    const int r = kOrdD2h[oi];
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& H = L.hca_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "hca 4mb init0 dev");
                    launch_vha_lowrank_proj_f16_in(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, mb_len[0], 512, w.stream);
                    hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "hca 4mb c2d0");
                    hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "hca 4mb wait0");
                    uint16_t* h_in0 = h_tp_oattn_pin + size_t(r) * size_t(M) * vha_cols;
                    hip_check(hipMemcpyAsync(h_in0, w.d_tmp_a, size_t(mb_len[0]) * vha_cols * sizeof(uint16_t),
                                             hipMemcpyDeviceToHost, w.stream_sh), "hca 4mb vha0 d2h");
                    hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "hca 4mb vha0 d2h rec");
                }
                for (int oi = 0; oi < tp; ++oi) {
                    const int r = kOrdD2h[oi];
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& H = L.hca_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "hca 4mb init1 dev");
                    for (int b = 1; b < kNb; ++b) {
                        const size_t voff = size_t(mb_off[b]) * vha_cols;
                        launch_vha_lowrank_proj_f16_in(w.d_tmp_b + voff, H.d_vha_r_loc, w.d_tmp_a + voff,
                                                       mb_len[b], 512, w.stream);
                        hip_check(hipEventRecord(w.ev_mb_c2d[b], w.stream), "hca 4mb c2d b");
                    }
                    f16_linear_f16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, int(gate_cols), 1024);
                }

                for (int b = 0; b < kNb; ++b) {
                    const size_t voff = size_t(mb_off[b]) * vha_cols;
                    const size_t goff = size_t(mb_off[b]) * gate_cols;
                    const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdD2h[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "hca 4mb sync d2h dev");
                        hip_check(hipEventSynchronize(w.ev_mb_vha_d2h[b]), "hca 4mb sync vha d2h");
                    }
                    if (b == 0 && M >= 256) {
                        t_ar0 = std::chrono::steady_clock::now();
                        chunk_hca_a_ms += std::chrono::duration<double, std::milli>(t_ar0 - t_ho0).count();
                    }
                    // Queue next VHA D2H immediately on stream_sh BEFORE CPU sum so PCIe never stalls!
                    if (b + 1 < kNb) {
                        const size_t voff_n = size_t(mb_off[b + 1]) * vha_cols;
                        for (int oi = 0; oi < tp; ++oi) {
                            const int r = kOrdD2h[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca 4mb next d2h dev");
                            hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[b + 1], 0), "hca 4mb wait next");
                            uint16_t* h_inn = h_tp_oattn_pin + size_t(r) * size_t(M) * vha_cols + voff_n;
                            hip_check(hipMemcpyAsync(h_inn, w.d_tmp_a + voff_n,
                                                     size_t(mb_len[b + 1]) * vha_cols * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, w.stream_sh), "hca 4mb next vha d2h");
                            hip_check(hipEventRecord(w.ev_mb_vha_d2h[b + 1], w.stream_sh), "hca 4mb next vha rec");
                        }
                    }
                    avx2_sum4_f16_canonical(
                        h_tp_oattn_pin + 0u * size_t(M) * vha_cols + voff,
                        h_tp_oattn_pin + 1u * size_t(M) * vha_cols + voff,
                        h_tp_oattn_pin + 2u * size_t(M) * vha_cols + voff,
                        h_tp_oattn_pin + 3u * size_t(M) * vha_cols + voff,
                        h_tp_qabs_pin + voff, size_t(mb_len[b]) * vha_cols, kOmpThreads);

                    // Queue VHA H2D(b) on stream_sh -> Compute(b) on stream (100% zero SDMA collision!)
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdH2d[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& H = L.hca_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "hca 4mb h2d+comp dev");
                        hip_check(hipMemcpyAsync(w.d_tmp_a + voff, h_tp_qabs_pin + voff,
                                                 size_t(mb_len[b]) * vha_cols * sizeof(uint16_t),
                                                 hipMemcpyHostToDevice, w.stream_sh), "hca 4mb vha h2d");
                        hip_check(hipEventRecord(w.ev_mb_vha_h2d[b], w.stream_sh), "hca 4mb vha h2d rec");
                        hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_vha_h2d[b], 0), "hca 4mb comp wait");
                        uint16_t* d_vha_b = w.d_tmp_b + voff;
                        launch_hca_vha_lowrank_invrope_f16_io(d_vha_b, w.d_tmp_a + voff, H.d_vha_q_loc,
                                                              w.d_pos_ids + mb_off[b], w.yarn_rope, d_vha_b, mb_len[b], w.stream);
                        uint16_t* d_wo_b = w.d_tmp_a + wo_base + goff;
                        run_hca_wo_a_f16(w, H, d_vha_b, w.d_tmp_d + goff, d_wo_b, mb_len[b]);
                        f16_linear_f16(w, d_wo_b, H.d_o_proj, w.d_sub_out + ooff, mb_len[b], kHiddenSize, int(gate_cols));
                        hip_check(hipEventRecord(w.ev_mb_d2c[b], w.stream), "hca 4mb o comp rec");
                    }
                    // Queue o_proj D2H(b-1) on stream_sh (Compute(b-1) already finished behind VHA D2H/H2D!)
                    if (b >= 1) {
                        const int bp = b - 1;
                        const size_t ooff_p = size_t(mb_off[bp]) * kHiddenSize;
                        for (int oi = 0; oi < tp; ++oi) {
                            const int r = kOrdD2h[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca 4mb op d2h dev");
                            hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[bp], 0), "hca 4mb op wait");
                            hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + ooff_p, w.d_sub_out + ooff_p,
                                                     size_t(mb_len[bp]) * kHiddenSize * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, w.stream_sh), "hca 4mb op d2h");
                            hip_check(hipEventRecord(w.ev_mb_o_d2h[bp], w.stream_sh), "hca 4mb op rec");
                        }
                    }
                }
                // Queue final two o_proj D2Hs (MB2, MB3) on stream_sh
                for (int bp = kNb - 2; bp < kNb; ++bp) {
                    const size_t ooff_p = size_t(mb_off[bp]) * kHiddenSize;
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdD2h[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "hca 4mb tail d2h dev");
                        hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[bp], 0), "hca 4mb tail wait");
                        hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + ooff_p, w.d_sub_out + ooff_p,
                                                 size_t(mb_len[bp]) * kHiddenSize * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, w.stream_sh), "hca 4mb tail d2h");
                        hip_check(hipEventRecord(w.ev_mb_o_d2h[bp], w.stream_sh), "hca 4mb tail rec");
                    }
                }
                // Drain o_proj(0..3) -> AVX2 sum4_to_bf16 -> Owner-First H2D on stream_sh!
                for (int b = 0; b < kNb; ++b) {
                    const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdD2h[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "hca 4mb o sync dev");
                        hip_check(hipEventSynchronize(w.ev_mb_o_d2h[b]), "hca 4mb o sync");
                    }
                    avx2_sum4_f16_to_bf16_canonical(
                        h_tp_ar_in_pin[0] + ooff, h_tp_ar_in_pin[1] + ooff,
                        h_tp_ar_in_pin[2] + ooff, h_tp_ar_in_pin[3] + ooff,
                        h_tp_o_out_pin + ooff, size_t(mb_len[b]) * kHiddenSize, kOmpThreads);
                    if (defer_ok) {
                        auto& g_own = *dev_map[L.owner_dev];
                        hip_check(hipSetDevice(g_own.hip_dev), "hca 4mb o own h2d dev");
                        hip_check(hipMemcpyAsync(g_own.d_sub_out + ooff, h_tp_o_out_pin + ooff,
                                                 size_t(mb_len[b]) * kHiddenSize * sizeof(uint16_t),
                                                 hipMemcpyHostToDevice, g_own.stream_sh), "hca 4mb o own h2d");
                        if (b == kNb - 1) {
                            hip_check(hipEventRecord(g_own.ev_mb_d2c[3], g_own.stream_sh), "hca 4mb o own done");
                            hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_d2c[3], 0), "hca 4mb o own wait");
                            if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = true;
                        }
                    } else {
                        for (int oi = 0; oi < tp; ++oi) {
                            const int r = kOrdH2d[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "hca 4mb o all h2d dev");
                            hip_check(hipMemcpyAsync(w.d_sub_out + ooff, h_tp_o_out_pin + ooff,
                                                     size_t(mb_len[b]) * kHiddenSize * sizeof(uint16_t),
                                                     hipMemcpyHostToDevice, w.stream_sh), "hca 4mb o all h2d");
                            if (b == kNb - 1) {
                                hip_check(hipEventRecord(w.ev_mb_d2c[3], w.stream_sh), "hca 4mb o all done");
                                hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[3], 0), "hca 4mb o all wait");
                            }
                        }
                    }
                }
                } // end use_gpu_ar else
                if (M >= 256) {
                    chunk_hca_ar_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ar0).count();
                }
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.hca_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ho0).count();
                }
            } else if (vha_stay_f16) {
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& H = L.hca_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "hca lr proj f16 dev");
                    launch_vha_lowrank_proj_f16_in(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, M, 512, w.stream);
                }
                tp_allreduce_f16(M, 16 * 512, &Mi50ExpertWorker::d_tmp_a, &Mi50ExpertWorker::d_tmp_a);

                const auto t_ho0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& H = L.hca_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "hca C f16 dev");
                    uint16_t* d_vha_out = w.d_tmp_b;
                    uint16_t* d_wo_out  = w.d_tmp_a;
                    launch_hca_vha_lowrank_invrope_f16_io(w.d_tmp_b, w.d_tmp_a, H.d_vha_q_loc,
                                                          w.d_pos_ids, w.yarn_rope, d_vha_out, M, w.stream);
                    f16_linear_f16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, g_loc * kHcaOLoraRank, 1024);
                    run_hca_wo_a_f16(w, H, d_vha_out, w.d_tmp_d, d_wo_out, M);
                    f16_linear_f16(w, d_wo_out, H.d_o_proj, w.d_sub_out, M, kHiddenSize, g_loc * kHcaOLoraRank);
                }
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.hca_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ho0).count();
                }
                tp_allreduce_f16_to_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            } else {
                if (use_hca_vha_lr) {
                    for (int r = 0; r < tp; ++r) {
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& H = L.hca_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "hca lr proj dev");
                        launch_vha_lowrank_proj_f16(w.d_tmp_b, H.d_vha_r_loc, w.d_tmp_a, M, 512, w.stream);
                    }
                    tp_allreduce_f16(M, 16 * 512, &Mi50ExpertWorker::d_tmp_a, &Mi50ExpertWorker::d_tmp_a);
                } else if (tp > 1) {
                    tp_allgather_heads_bf16(M, kHcaNumHeads, 512,
                                            &Mi50ExpertWorker::d_tmp_b, &Mi50ExpertWorker::d_tmp_a, h_tp_oattn_pin);
                }

                // Phase C: Parallel Low-Rank / Rank-Major VHA + invRoPE + 2-group FP16 wo_a * sigmoid(gate) + o_proj/4
                const auto t_ho0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& H = L.hca_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "hca C dev");

                    const uint16_t* d_o_all = (tp > 1) ? w.d_tmp_a : w.d_tmp_b;
                    uint16_t* d_vha_out     = (tp > 1) ? w.d_tmp_b : w.d_tmp_a;
                    uint16_t* d_wo_out      = (tp > 1) ? w.d_tmp_a : w.d_tmp_b;

                    if (use_hca_vha_lr) {
                        launch_hca_vha_lowrank_invrope_f16(w.d_tmp_b, w.d_tmp_a, H.d_vha_q_loc,
                                                           w.d_pos_ids, w.yarn_rope, d_vha_out, M, w.stream);
                    } else {
                        launch_hca_vha_invrope_shard(d_o_all, H.d_vha_m, w.d_pos_ids, w.yarn_rope,
                                                     d_vha_out, M, tp, r * h_loc, h_loc, w.stream);
                    }
                    bf16_linear_bf16(w, w.d_qr, H.d_gate_proj, w.d_tmp_d, M, g_loc * kHcaOLoraRank, 1024);

                    if (M <= 8) {
                        launch_hca_grouped_wo_a_lrga_bf16_f16(d_vha_out, H.d_wo_a, w.d_tmp_d, d_wo_out, M, g_loc, kHcaOLoraRank, w.stream);
                    } else {
                        const float alpha = 1.0f, beta = 0.0f;
                        uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                        const int wo_strip = std::min(1024, int((512u * 16384u) / (size_t(g_loc) * 4096u)));
                        for (int m0 = 0; m0 < M; m0 += wo_strip) {
                            const int cur = std::min(wo_strip, M - m0);
                            const int eff = std::min(wo_strip, ((cur + 63) / 64) * 64);
                            launch_bf16_to_f16(d_vha_out + size_t(m0) * size_t(g_loc) * 4096u, w.d_x_f16, cur * g_loc * 4096, w.stream);
                            if (cur < eff) {
                                hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(g_loc) * 4096u, 0,
                                                         size_t(eff - cur) * size_t(g_loc) * 4096u * sizeof(uint16_t), w.stream), "zero wo_a pad");
                            }
                            rocblas_gemm_strided_batched_ex_tuned(
                                w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                kHcaOLoraRank, eff, 4096, &alpha,
                                H.d_wo_a,  rocblas_datatype_f16_r, 4096, int64_t(kHcaOLoraRank) * 4096LL,
                                w.d_x_f16, rocblas_datatype_f16_r, g_loc * 4096, 4096LL,
                                &beta,
                                d_y16, rocblas_datatype_f16_r, g_loc * kHcaOLoraRank, int64_t(kHcaOLoraRank),
                                d_y16, rocblas_datatype_f16_r, g_loc * kHcaOLoraRank, int64_t(kHcaOLoraRank),
                                g_loc, "hca wo_a hh_hpa batched");
                            const size_t z_off = size_t(m0) * size_t(g_loc) * size_t(kHcaOLoraRank);
                            launch_f16_mul_sigmoid_to_bf16(d_y16, w.d_tmp_d + z_off, d_wo_out + z_off, cur * g_loc * kHcaOLoraRank, w.stream);
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
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.hca_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_a0).count();
            }
        } else {
            // mqa_dsa: Wave-10 C3 TP4 16-head Sparse/Dense DSA + Wave-11 C2 Zero-Bounce FP16 Phase A!
            const int h_loc = kDsaNumHeads / tp;
            const int ro = L.owner_tp_rank;
            auto& g_own = *dev_map[L.owner_dev];
            const auto& D_own = L.dsa_tp[ro];
            const bool use_tp4_dsa = (tp == 4 && M > 8 && L.dsa_tp[0].has_vha_lr);
            constexpr int kStrideSel = kDsaIndexTopK + kSlidingWindow;
            const int n_dense = std::max(0, std::min(M, kStrideSel - pos0));
            const bool do_fi_layer = (use_flashindexer && D_own.d_fi_gate != nullptr && D_own.d_fi_bias != nullptr);
            const bool dsa_stay_f16 = (use_tp4_dsa && do_fi_layer && M >= 64 && (M & 63) == 0 && M <= max_chunk &&
                                       (n_dense >= M || (M - n_dense) >= 64));

            // Phase A: Parallel q_a, q_b/4, RoPE, W_UK/4 (+ owner kv_a & indexer prep!)
            const auto t_dp0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dsa_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dsa A dev");

                if (dsa_stay_f16) {
                    // Wave-11 C2/C3: Reuse direct FP16 w.d_x_f16 written by mHC (or convert once if !x_f16_ready)
                    // and stay 100% FP16 across q_a -> RMSNorm -> q_b -> kv_a -> idx_wq_b -> idx_wk -> idx_w_proj -> W_UK -> fi_gate -> gate_proj!
                    if (!x_f16_ready) {
                        launch_bf16_to_f16(w.d_x_in, w.d_x_f16, M * kHiddenSize, w.stream);
                    }
                    f16_linear_f16(w, w.d_x_f16, D.d_q_a, w.d_tmp_d, M, 1024, kHiddenSize);
                    launch_rmsnorm_f16_io(w.d_tmp_d, D.d_q_a_norm, w.d_qr, M, 1024, mcfg.rms_norm_eps, w.stream);
                    f16_linear_f16(w, w.d_qr, D.d_q_b, w.d_tmp_a, M, h_loc * 256, 1024);

                    if (r == ro) {
                        f16_linear_f16(w, w.d_x_f16, D.d_kv_a, w.d_kv_tmp, M, 576, kHiddenSize);
                        launch_dsa_prefill_kv_norm_rope_append_f16_in(
                            w.d_tmp_a, w.d_kv_tmp, D.d_kv_a_norm, w.d_pos_ids, w.dsa_rope,
                            L.d_dsa_mla_kv, M, h_loc, opt.max_kv, mcfg.rms_norm_eps, w.stream);

                        f16_linear_f16(w, w.d_qr,    D.d_idx_wq_b,   w.d_tmp_b, M, 8192, 1024);
                        f16_linear_f16(w, w.d_x_f16, D.d_idx_wk,     w.d_tmp_d, M, 128, kHiddenSize);
                        launch_dsa_indexer_prep_neox_wht128_f16_in(
                            w.d_tmp_b, w.d_tmp_d, D.d_idx_kn_w, D.d_idx_kn_b,
                            w.d_pos_ids, w.dsa_rope, L.d_dsa_ik_cache, M, opt.max_kv, w.stream);
                        f16_linear_f16(w, w.d_x_f16, D.d_idx_w_proj, w.d_tmp_d, M, 64, kHiddenSize);
                    } else {
                        launch_dsa_mla_rope_q_f16_inplace(w.d_tmp_a, w.d_pos_ids, w.dsa_rope, M, h_loc, w.stream);
                    }

                    // Single-launch direct FP16 W_UK absorb (sol=33181: 2.13 -> 0.43 ms = 5.0x faster!)
                    const float alpha = 1.0f, beta = 0.0f;
                    rocblas_gemm_strided_batched_ex_tuned(
                        w.blas, rocblas_operation_none, rocblas_operation_none,
                        512, M, 192, &alpha,
                        D.d_kv_b,  rocblas_datatype_f16_r, 512, 448LL * 512LL,
                        w.d_tmp_a, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                        &beta,
                        w.d_tmp_c, rocblas_datatype_f16_r, h_loc * 576, 576LL,
                        w.d_tmp_c, rocblas_datatype_f16_r, h_loc * 576, 576LL,
                        h_loc, "dsa q_absorb hh_hpa f16 zero-bounce");
                    launch_dsa_copy_qrope_f16_to_f16(w.d_tmp_a, w.d_tmp_c, M, h_loc, w.stream);
                } else {
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
                        uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                        const int uk_cap = int(std::min((512u * 16384u) / (size_t(h_loc) * 256u),
                                                        (512u * 16384u * 2u) / (size_t(h_loc) * 576u)));
                        const int uk_strip = std::max(64, std::min(1024, (uk_cap / 64) * 64));
                        for (int m0 = 0; m0 < M; m0 += uk_strip) {
                            const int cur = std::min(uk_strip, M - m0);
                            const int eff = std::min(uk_strip, ((cur + 63) / 64) * 64);
                            launch_bf16_to_f16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 256u, w.d_x_f16, cur * h_loc * 256, w.stream);
                            if (cur < eff) {
                                hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(h_loc) * 256u, 0,
                                                         size_t(eff - cur) * size_t(h_loc) * 256u * sizeof(uint16_t), w.stream), "zero uk pad");
                            }
                            if (use_tp4_dsa) {
                                uint16_t* d_dst_c = w.d_tmp_c + size_t(m0) * size_t(h_loc) * 576u;
                                uint16_t* d_out_uk = (cur == eff) ? d_dst_c : d_y16;
                                rocblas_gemm_strided_batched_ex_tuned(
                                    w.blas, rocblas_operation_none, rocblas_operation_none,
                                    512, eff, 192, &alpha,
                                    D.d_kv_b,  rocblas_datatype_f16_r, 512, 448LL * 512LL,
                                    w.d_x_f16, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                                    &beta,
                                    d_out_uk, rocblas_datatype_f16_r, h_loc * 576, 576LL,
                                    d_out_uk, rocblas_datatype_f16_r, h_loc * 576, 576LL,
                                    h_loc, "dsa q_absorb hh_hpa f16 direct");
                                if (cur < eff) {
                                    hip_check(hipMemcpyAsync(d_dst_c, d_y16,
                                                             size_t(cur) * size_t(h_loc) * 576u * sizeof(uint16_t),
                                                             hipMemcpyDeviceToDevice, w.stream), "d2d uk f16 tail");
                                }
                                launch_dsa_copy_qrope_f16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 256u,
                                                          d_dst_c, cur, h_loc, w.stream);
                            } else {
                                rocblas_gemm_strided_batched_ex_tuned(
                                    w.blas, rocblas_operation_none, rocblas_operation_none,
                                    512, eff, 192, &alpha,
                                    D.d_kv_b,  rocblas_datatype_f16_r, 512, 448LL * 512LL,
                                    w.d_x_f16, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, h_loc * 576, 576LL,
                                    d_y16, rocblas_datatype_f16_r, h_loc * 576, 576LL,
                                    h_loc, "dsa q_absorb hh_hpa batched");
                                launch_f16_to_bf16(d_y16, w.d_tmp_c + size_t(m0) * size_t(h_loc) * 576u, cur * h_loc * 576, w.stream);
                                launch_dsa_copy_qrope_bf16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 256u,
                                                           w.d_tmp_c + size_t(m0) * size_t(h_loc) * 576u, cur, h_loc, w.stream);
                            }
                        }
                    }
                }
            }

            // Phase B: Zero-Host-Sync Mapped Gather q_abs (M<=8) or 1D Contiguous Gather (SKIPPED when use_tp4_dsa!)
            if (!use_tp4_dsa && tp > 1) {
                if (M <= 8 && tp == 4 && h_tp_mapped_slabs != nullptr && gfx906_mapped_tp4_enabled()) {
                    const uint32_t ep = ++tp_mapped_epoch;
                    for (int r = 0; r < 4; ++r) {
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "dsa qabs map dev");
                        launch_tp4_mapped_dsa_gather_qabs_bf16(
                            w.d_tmp_c, (r == ro) ? w.d_tmp_c : nullptr,
                            w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                            r, ro, M, kMappedSlabStride, ep, w.stream);
                    }
                    hip_check(hipSetDevice(g_own.hip_dev), "dsa own dev");
                } else {
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
                }
            } else {
                hip_check(hipSetDevice(g_own.hip_dev), "dsa own dev");
            }

            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp0).count();
            }

            // Phase C: Strict pos < 2176 dense vs pos >= 2176 sparse split!
            // When use_tp4_dsa: run FlashIndexer + topk2048 FIRST (while g_own.d_tmp_b still holds idx_wq and g_own.d_tmp_a
            // provides 128 MiB / 33.5M floats = 2x max_batch_t score buffer!), then Phase C1 Dense (4x TP4 parallel into
            // w.d_tmp_b), then Phase C2 Sparse (unique-KV FP16/u16 host-SDMA overlapped with owner MQA into w.d_tmp_b)!

            if (use_tp4_dsa) {
                // C0: FlashIndexer + select_topk2048 on g_own BEFORE Phase C1 touches g_own.d_tmp_b / g_own.d_tmp_a!
                const auto t_c0_start = (profiling_enabled || M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (n_dense < M) {
                    hip_check(hipSetDevice(g_own.hip_dev), "dsa tp4 fi dev");
                    const bool do_fi = do_fi_layer;
                    if (do_fi) {
                        if (dsa_stay_f16) {
                            const float alpha = 1.0f, beta = 0.0f;
                            rocblas_gemm_ex_tuned(
                                g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                64, M, kHiddenSize, &alpha,
                                D_own.d_fi_gate, rocblas_datatype_f16_r, kHiddenSize,
                                g_own.d_x_f16,   rocblas_datatype_f16_r, kHiddenSize,
                                &beta,
                                g_own.d_fi_gates, rocblas_datatype_f32_r, 64,
                                g_own.d_fi_gates, rocblas_datatype_f32_r, 64, "dsa fi_gate f16 direct");
                            launch_flashindexer_select_heads_f16(
                                g_own.d_fi_gates, D_own.d_fi_bias, g_own.d_tmp_b, g_own.d_tmp_d,
                                g_own.d_fi_q8, g_own.d_fi_w8, M, g_own.stream);
                        } else {
                            bf16_linear_f32(g_own, g_own.d_x_in, D_own.d_fi_gate, g_own.d_fi_gates, M, 64, kHiddenSize);
                            launch_flashindexer_select_heads(
                                g_own.d_fi_gates, D_own.d_fi_bias, g_own.d_tmp_b, g_own.d_tmp_d,
                                g_own.d_fi_q8, g_own.d_fi_w8, M, g_own.stream);
                        }
                    }
                    const bool use_tmp_a_scores = (size_t(max_chunk) * 16384u >= g_own.idx_scores_cap);
                    float* d_fi_scores = use_tmp_a_scores ? reinterpret_cast<float*>(g_own.d_tmp_a) : g_own.d_idx_scores;
                    const size_t fi_scores_cap = use_tmp_a_scores ? (size_t(max_chunk) * 16384u) : g_own.idx_scores_cap;
                    const int max_hist = std::max(1, (pos0 + M - 1) - kSlidingWindow + 1);
                    const int max_batch_t = std::max(1, std::min(M - n_dense, int(fi_scores_cap / size_t(max_hist))));
                    // Convert/copy sparse-row FlashIndexer queries (g_own.d_fi_q8[n_dense..M)) into d_q_all_f16!
                    uint16_t* d_q_all_f16  = g_own.d_x_f16;             // up to [2048*8, 128] = 2.09M halfs (4 MiB)
                    uint16_t* d_k_tile_f16 = g_own.d_x_f16 + 3145728u;  // [4096, 128] = 524K halfs (1 MiB)
                    if (do_fi && max_hist > kDsaIndexTopK) {
                        const int n_sp_tot = M - n_dense;
                        const int n_sp_pad = ((n_sp_tot + 63) / 64) * 64;
                        if (dsa_stay_f16) {
                            hip_check(hipMemcpyAsync(d_q_all_f16, g_own.d_fi_q8 + size_t(n_dense) * 8u * 128u,
                                                     size_t(n_sp_tot) * 8u * 128u * sizeof(uint16_t),
                                                     hipMemcpyDeviceToDevice, g_own.stream), "d2d fi q_all f16");
                        } else {
                            launch_bf16_to_f16(g_own.d_fi_q8 + size_t(n_dense) * 8u * 128u, d_q_all_f16, n_sp_tot * 8 * 128, g_own.stream);
                        }
                        if (n_sp_tot < n_sp_pad) {
                            hip_check(hipMemsetAsync(d_q_all_f16 + size_t(n_sp_tot) * 8u * 128u, 0,
                                                     size_t(n_sp_pad - n_sp_tot) * 8u * 128u * sizeof(uint16_t), g_own.stream), "zero fi q_all pad");
                        }
                    }
                    // Wave-10 C5b: inverted single-pass K_tile = 16,384 loop. The K tile
                    // is the OUTER dimension and queries stay active on the inside, so
                    // ik_cache is read & converted strictly ONCE through 262K -> 512K -> 1M
                    // (strictly linear O(L)) and top-2048 state merges ping-pong across tiles.
                    // 0 B extra GPU VRAM: d_tmp_a (128 MiB) = tile scores, d_tmp_b (64 MiB) =
                    // two 32 MiB ping-pong halves, d_x_f16 (16 MiB) = q_all (4 MiB) + k_tile (4 MiB),
                    // d_y_f32 (72 MiB) = 4096x4096 FP16 GEMM slab (32 MiB).
                    const int n_sp_all = M - n_dense;
                    const bool use_streaming_fi = (do_fi && max_hist > kDsaIndexTopK &&
                                                   n_sp_all >= 64 && n_sp_all <= max_chunk);
                    bool tp4_fi_ready = (use_streaming_fi && tp == 4 && max_hist > kDsaIndexTopK && n_sp_all >= 256);
                    if (tp4_fi_ready) {
                        for (int r = 0; r < tp; ++r) {
                            auto& w = *dev_map[dev_for_r(r)];
                            if (!w.stream_sh || !w.ev_mb_c2d[0] || !w.ev_mb_d2c[0] || !w.ev_mb_vha_d2h[0]) {
                                tp4_fi_ready = false; break;
                            }
                        }
                    }
                    if (tp4_fi_ready) {
                        // Wave-11 C4: 4-way query-sharded TP4 FlashIndexer (<=512 queries/GPU, 0 B extra VRAM!)
                        // Double-buffered 4 MiB FP16 d_k_buf[2] inside w.d_x_f16[4..12 MiB) overlapped on w.stream_sh
                        // behind GEMM + min_u floor-bypass streaming_merge on w.stream (set_mism == 0 bit-exact!).
                        constexpr int kTileK = 16384;
                        constexpr int kSubK  = 4096;
                        const float alpha = 1.0f, beta = 0.0f;
                        int q_off[4]{}, q_cnt[4]{}, q_eff[4]{};
                        {
                            int rem = n_sp_all, cur_off = 0;
                            for (int r = 0; r < 4; ++r) {
                                const int c = std::min(512, (rem + (3 - r)) / (4 - r));
                                q_off[r] = cur_off;
                                q_cnt[r] = c;
                                q_eff[r] = std::min(512, ((c + 63) / 64) * 64);
                                cur_off += c;
                                rem -= c;
                            }
                        }
                        // One-time tiny query/weight scatter (<= 4 MiB q_all + <= 64 KiB fi_w8 -> < 0.3 ms!)
                        hip_check(hipSetDevice(g_own.hip_dev), "tp4 fi scatter dev");
                        hip_check(hipMemcpyAsync(h_tp_oattn_pin, g_own.d_x_f16,
                                                 size_t(n_sp_all) * 8u * 128u * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, g_own.stream), "tp4 fi q d2h");
                        hip_check(hipMemcpyAsync(h_mhc_mix_pin, g_own.d_fi_w8,
                                                 size_t(M) * 8u * sizeof(float),
                                                 hipMemcpyDeviceToHost, g_own.stream), "tp4 fi w d2h");
                        hip_check(hipStreamSynchronize(g_own.stream), "tp4 fi scatter sync");
                        for (int r = 0; r < 4; ++r) {
                            auto& w = *dev_map[dev_for_r(r)];
                            if (r == ro) {
                                if (q_cnt[r] < q_eff[r]) {
                                    hip_check(hipSetDevice(w.hip_dev), "tp4 fi own qpad dev");
                                    hip_check(hipMemcpyAsync(w.d_x_f16, h_tp_oattn_pin + size_t(q_off[r]) * 8u * 128u,
                                                             size_t(q_cnt[r]) * 8u * 128u * sizeof(uint16_t),
                                                             hipMemcpyHostToDevice, w.stream), "tp4 fi own q h2d");
                                    hip_check(hipMemsetAsync(w.d_x_f16 + size_t(q_cnt[r]) * 8u * 128u, 0,
                                                             size_t(q_eff[r] - q_cnt[r]) * 8u * 128u * sizeof(uint16_t),
                                                             w.stream), "tp4 fi own q zero");
                                }
                                continue;
                            }
                            hip_check(hipSetDevice(w.hip_dev), "tp4 fi h2d dev");
                            hip_check(hipMemcpyAsync(w.d_x_f16, h_tp_oattn_pin + size_t(q_off[r]) * 8u * 128u,
                                                     size_t(q_cnt[r]) * 8u * 128u * sizeof(uint16_t),
                                                     hipMemcpyHostToDevice, w.stream), "tp4 fi q h2d");
                            if (q_cnt[r] < q_eff[r]) {
                                hip_check(hipMemsetAsync(w.d_x_f16 + size_t(q_cnt[r]) * 8u * 128u, 0,
                                                         size_t(q_eff[r] - q_cnt[r]) * 8u * 128u * sizeof(uint16_t),
                                                         w.stream), "tp4 fi q zero");
                            }
                            hip_check(hipMemcpyAsync(w.d_f32_tmp0, h_mhc_mix_pin,
                                                     size_t(M) * 8u * sizeof(float),
                                                     hipMemcpyHostToDevice, w.stream), "tp4 fi w h2d");
                        }
                        const uint16_t* d_q_rank[4];
                        const float*    d_w_rank[4];
                        float*   bv0_r[4]; int32_t* bi0_r[4];
                        float*   bv1_r[4]; int32_t* bi1_r[4];
                        for (int r = 0; r < 4; ++r) {
                            auto& w = *dev_map[dev_for_r(r)];
                            d_q_rank[r] = (r == ro && q_cnt[r] == q_eff[r]) ? (w.d_x_f16 + size_t(q_off[r]) * 8u * 128u) : w.d_x_f16;
                            d_w_rank[r] = (r == ro) ? w.d_fi_w8 : w.d_f32_tmp0;
                            char* bbase = reinterpret_cast<char*>(w.d_tmp_b);
                            const size_t hs = size_t(q_cnt[r]) * 2048u;
                            bv0_r[r] = reinterpret_cast<float*>(bbase);
                            bi0_r[r] = reinterpret_cast<int32_t*>(bbase + hs * 4u);
                            bv1_r[r] = reinterpret_cast<float*>(bbase + hs * 8u);
                            bi1_r[r] = reinterpret_cast<int32_t*>(bbase + hs * 12u);
                        }
                        // Double-buffered 4 MiB FP16 key tile slots inside d_x_f16[4..12 MiB) & h_tp_qabs_pin[0..8 MiB)
                        auto k_buf_dev = [](Mi50ExpertWorker& w, int slot) -> uint16_t* {
                            return w.d_x_f16 + (slot == 0 ? 2097152u : 4194304u);
                        };
                        uint16_t* h_k_buf[2] = { h_tp_qabs_pin, h_tp_qabs_pin + 2097152u };
                        const int n_tiles = (max_hist + kTileK - 1) / kTileK;

                        auto stage_tile_k = [&](int ti, bool wait_prev_use) {
                            const int p = ti & 1;
                            const int k_blk = ti * kTileK;
                            const int k_cur = std::min(kTileK, max_hist - k_blk);
                            const int k_pad = ((k_cur + kSubK - 1) / kSubK) * kSubK;
                            hip_check(hipSetDevice(g_own.hip_dev), "tp4 fi stage own dev");
                            if (wait_prev_use) {
                                hip_check(hipStreamWaitEvent(g_own.stream_sh, g_own.ev_mb_c2d[p], 0), "tp4 fi own k wait");
                            }
                            uint16_t* d_ko = k_buf_dev(g_own, p);
                            launch_bf16_to_f16(L.d_dsa_ik_cache + size_t(k_blk) * 128u, d_ko, k_cur * 128, g_own.stream_sh);
                            if (k_cur < k_pad) {
                                hip_check(hipMemsetAsync(d_ko + size_t(k_cur) * 128u, 0,
                                                         size_t(k_pad - k_cur) * 128u * sizeof(uint16_t),
                                                         g_own.stream_sh), "tp4 fi own k pad");
                            }
                            hip_check(hipMemcpyAsync(h_k_buf[p], d_ko, size_t(k_pad) * 128u * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, g_own.stream_sh), "tp4 fi k d2h");
                            hip_check(hipEventRecord(g_own.ev_mb_vha_d2h[p], g_own.stream_sh), "tp4 fi k d2h rec");
                            hip_check(hipEventSynchronize(g_own.ev_mb_vha_d2h[p]), "tp4 fi k d2h sync");
                            for (int r = 0; r < 4; ++r) {
                                if (r == ro) continue;
                                auto& w = *dev_map[dev_for_r(r)];
                                hip_check(hipSetDevice(w.hip_dev), "tp4 fi k h2d dev");
                                if (wait_prev_use) {
                                    hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[p], 0), "tp4 fi rem k wait");
                                }
                                hip_check(hipMemcpyAsync(k_buf_dev(w, p), h_k_buf[p],
                                                         size_t(k_pad) * 128u * sizeof(uint16_t),
                                                         hipMemcpyHostToDevice, w.stream_sh), "tp4 fi k h2d");
                                hip_check(hipEventRecord(w.ev_mb_d2c[p], w.stream_sh), "tp4 fi k h2d rec");
                            }
                        };

                        stage_tile_k(0, false);
                        bool odd = false;
                        for (int ti = 0; ti < n_tiles; ++ti) {
                            const int p = ti & 1;
                            const int k_blk = ti * kTileK;
                            const int k_cur = std::min(kTileK, max_hist - k_blk);
                            for (int r = 0; r < 4; ++r) {
                                if (q_cnt[r] <= 0) continue;
                                auto& w = *dev_map[dev_for_r(r)];
                                hip_check(hipSetDevice(w.hip_dev), "tp4 fi comp dev");
                                if (r != ro) {
                                    hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[p], 0), "tp4 fi wait k h2d");
                                }
                                const uint16_t* d_kt = k_buf_dev(w, p);
                                uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                                float* d_ts = reinterpret_cast<float*>(w.d_tmp_a);
                                for (int ks = 0; ks < k_cur; ks += kSubK) {
                                    rocblas_gemm_ex_tuned(
                                        w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                        kSubK, q_eff[r] * 8, 128, &alpha,
                                        d_kt + size_t(ks) * 128u, rocblas_datatype_f16_r, 128,
                                        d_q_rank[r],              rocblas_datatype_f16_r, 128,
                                        &beta,
                                        d_y16, rocblas_datatype_f16_r, kSubK,
                                        d_y16, rocblas_datatype_f16_r, kSubK, "tp4 fi gemm hh_hpa");
                                    launch_flashindexer_relu_wsum_tiled_f16_in(
                                        d_y16, d_w_rank[r], d_ts,
                                        pos0, n_dense + q_off[r], q_cnt[r], k_blk, ks, kSubK, kTileK, w.stream);
                                }
                                hip_check(hipEventRecord(w.ev_mb_c2d[p], w.stream), "tp4 fi k done rec");
                                if (!odd) {
                                    launch_dsa_topk2048_streaming_merge(
                                        d_ts, bv0_r[r], bi0_r[r], bv1_r[r], bi1_r[r],
                                        pos0, n_dense + q_off[r], q_cnt[r], k_blk, kTileK, w.stream);
                                } else {
                                    launch_dsa_topk2048_streaming_merge(
                                        d_ts, bv1_r[r], bi1_r[r], bv0_r[r], bi0_r[r],
                                        pos0, n_dense + q_off[r], q_cnt[r], k_blk, kTileK, w.stream);
                                }
                            }
                            if (ti + 1 < n_tiles) {
                                stage_tile_k(ti + 1, ti >= 1);
                            }
                            odd = !odd;
                        }
                        // Gather 3 non-owners' final [q_cnt[r], 2048] i32 indices (<= 12 MiB) -> finalize on g_own!
                        int32_t* h_bi_gather = reinterpret_cast<int32_t*>(h_tp_oattn_pin);
                        for (int r = 0; r < 4; ++r) {
                            if (r == ro || q_cnt[r] <= 0) continue;
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "tp4 fi gather d2h dev");
                            hip_check(hipMemcpyAsync(h_bi_gather + size_t(q_off[r]) * 2048u,
                                                     odd ? bi1_r[r] : bi0_r[r],
                                                     size_t(q_cnt[r]) * 2048u * sizeof(int32_t),
                                                     hipMemcpyDeviceToHost, w.stream), "tp4 fi gather d2h");
                        }
                        hip_check(hipSetDevice(g_own.hip_dev), "tp4 fi fin own dev");
                        if (q_cnt[ro] > 0) {
                            launch_dsa_topk2048_streaming_finalize(
                                odd ? bi1_r[ro] : bi0_r[ro], g_own.d_sel_idx, g_own.d_sel_cnts,
                                pos0, n_dense + q_off[ro], q_cnt[ro], kStrideSel, g_own.stream);
                        }
                        for (int r = 0; r < 4; ++r) {
                            if (r == ro || q_cnt[r] <= 0) continue;
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "tp4 fi gather sync dev");
                            hip_check(hipStreamSynchronize(w.stream), "tp4 fi gather sync");
                            hip_check(hipSetDevice(g_own.hip_dev), "tp4 fi gather h2d dev");
                            int32_t* d_bi_dst = reinterpret_cast<int32_t*>(g_own.d_tmp_a) + size_t(q_off[r]) * 2048u;
                            hip_check(hipMemcpyAsync(d_bi_dst, h_bi_gather + size_t(q_off[r]) * 2048u,
                                                     size_t(q_cnt[r]) * 2048u * sizeof(int32_t),
                                                     hipMemcpyHostToDevice, g_own.stream), "tp4 fi gather h2d");
                            launch_dsa_topk2048_streaming_finalize(
                                d_bi_dst, g_own.d_sel_idx, g_own.d_sel_cnts,
                                pos0, n_dense + q_off[r], q_cnt[r], kStrideSel, g_own.stream);
                        }
                    } else if (use_streaming_fi) {
                        const int n_sp_tot = n_sp_all;
                        constexpr int kTileK = 16384;
                        constexpr int kSubK  = 4096;
                        constexpr int kSubB  = 512;
                        float* d_tile_scores   = reinterpret_cast<float*>(g_own.d_tmp_a);
                        uint16_t* d_q_all_f16  = g_own.d_x_f16;             // [n_sp_pad, 8, 128] (<=4 MiB)
                        uint16_t* d_k_tile_f16 = g_own.d_x_f16 + 3145728u;  // [16384, 128] (4 MiB inside 16 MiB)
                        uint16_t* d_y16        = reinterpret_cast<uint16_t*>(g_own.d_y_f32);
                        char* b0 = reinterpret_cast<char*>(g_own.d_tmp_b);
                        const size_t half_stride = size_t(n_sp_tot) * 2048u;
                        float*   bv0 = reinterpret_cast<float*>(b0);
                        int32_t* bi0 = reinterpret_cast<int32_t*>(b0 + half_stride * 4u);
                        float*   bv1 = reinterpret_cast<float*>(b0 + half_stride * 8u);
                        int32_t* bi1 = reinterpret_cast<int32_t*>(b0 + half_stride * 12u);
                        const float alpha = 1.0f, beta = 0.0f;
                        bool odd = false;
                        for (int k_blk = 0; k_blk < max_hist; k_blk += kTileK) {
                            const int k_blk_cur = std::min(kTileK, max_hist - k_blk);
                            const int k_blk_pad = ((k_blk_cur + kSubK - 1) / kSubK) * kSubK;
                            // Convert ALL k_blk_cur <= 16,384 keys STRICTLY ONCE per outer tile!
                            launch_bf16_to_f16(L.d_dsa_ik_cache + size_t(k_blk) * 128u,
                                               d_k_tile_f16, k_blk_cur * 128, g_own.stream);
                            if (k_blk_cur < k_blk_pad) {
                                hip_check(hipMemsetAsync(d_k_tile_f16 + size_t(k_blk_cur) * 128u, 0,
                                                         size_t(k_blk_pad - k_blk_cur) * 128u * sizeof(uint16_t),
                                                         g_own.stream), "zero fi k_tile pad");
                            }
                            for (int ks = 0; ks < k_blk_cur; ks += kSubK) {
                                const uint16_t* d_k_sub = d_k_tile_f16 + size_t(ks) * 128u;
                                for (int b0r = 0; b0r < n_sp_tot; b0r += kSubB) {
                                    const int b_cur = std::min(kSubB, n_sp_tot - b0r);
                                    const int b_eff = std::min(kSubB, ((b_cur + 63) / 64) * 64);
                                    const uint16_t* d_q_strip = d_q_all_f16 + size_t(b0r) * 8u * 128u;
                                    rocblas_gemm_ex_tuned(
                                        g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                        kSubK, b_eff * 8, 128, &alpha,
                                        d_k_sub,   rocblas_datatype_f16_r, 128,
                                        d_q_strip, rocblas_datatype_f16_r, 128,
                                        &beta,
                                        d_y16, rocblas_datatype_f16_r, kSubK,
                                        d_y16, rocblas_datatype_f16_r, kSubK, "fi gemm_ex hh_hpa 512");
                                    launch_flashindexer_relu_wsum_tiled_f16_in(
                                        d_y16, g_own.d_fi_w8,
                                        d_tile_scores + size_t(b0r) * size_t(kTileK),
                                        pos0, n_dense + b0r, b_cur, k_blk, ks, kSubK, kTileK, g_own.stream);
                                }
                            }
                            if (!odd) {
                                launch_dsa_topk2048_streaming_merge(
                                    d_tile_scores, bv0, bi0, bv1, bi1,
                                    pos0, n_dense, n_sp_tot, k_blk, kTileK, g_own.stream);
                            } else {
                                launch_dsa_topk2048_streaming_merge(
                                    d_tile_scores, bv1, bi1, bv0, bi0,
                                    pos0, n_dense, n_sp_tot, k_blk, kTileK, g_own.stream);
                            }
                            odd = !odd;
                        }
                        // odd==true means last write went to (bv1, bi1); odd==false -> (bv0, bi0)!
                        launch_dsa_topk2048_streaming_finalize(
                            odd ? bi1 : bi0, g_own.d_sel_idx, g_own.d_sel_cnts,
                            pos0, n_dense, n_sp_tot, kStrideSel, g_own.stream);
                    }
                    for (int t0 = n_dense; !use_streaming_fi && t0 < M; t0 += max_batch_t) {
                        const int b_sc = std::min(max_batch_t, M - t0);
                        const int b_max_hist = std::max(1, (pos0 + t0 + b_sc - 1) - kSlidingWindow + 1);
                        if (b_max_hist > kDsaIndexTopK) {
                            if (do_fi) {
                                // Wave-10 C3: Type_HH_HPA FlashIndexer (kTileK=4096, kSubB=512 -> 32 MiB d_y16 <= 72 MiB!)
                                constexpr int kTileK = 4096;
                                constexpr int kSubB  = 512;
                                uint16_t* d_q_strip_base = d_q_all_f16 + size_t(t0 - n_dense) * 8u * 128u;
                                uint16_t* d_y16 = reinterpret_cast<uint16_t*>(g_own.d_y_f32);
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
                                        const int b_eff = std::min(kSubB, ((b_cur + 63) / 64) * 64);
                                        rocblas_gemm_ex_tuned(
                                            g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                            kTileK, b_eff * 8, 128, &alpha,
                                            d_k_tile_f16,                          rocblas_datatype_f16_r, 128,
                                            d_q_strip_base + size_t(b0) * 8u * 128u, rocblas_datatype_f16_r, 128,
                                            &beta,
                                            d_y16, rocblas_datatype_f16_r, kTileK,
                                            d_y16, rocblas_datatype_f16_r, kTileK, "fi gemm_ex hh_hpa");
                                        launch_flashindexer_relu_wsum_f16_in(
                                            d_y16, g_own.d_fi_w8,
                                            d_fi_scores + size_t(b0) * size_t(b_max_hist),
                                            pos0, t0 + b0, b_cur, k0, kTileK, b_max_hist, g_own.stream);
                                    }
                                }
                            } else {
                                launch_dsa_indexer_score_full_batch(
                                    g_own.d_tmp_b, g_own.d_tmp_d, L.d_dsa_ik_cache, d_fi_scores,
                                    pos0, t0, b_sc, b_max_hist, g_own.stream);
                            }
                        }
                        if (b_sc >= 64) {
                            launch_dsa_select_topk2048_gpu(
                                d_fi_scores, g_own.d_sel_idx, g_own.d_sel_cnts,
                                pos0, t0, b_sc, b_max_hist, kStrideSel, g_own.stream);
                        } else {
                            launch_dsa_select_topk2048_wave6(
                                d_fi_scores, reinterpret_cast<uint32_t*>(g_own.d_y_f32),
                                g_own.d_sel_idx, g_own.d_sel_cnts,
                                pos0, t0, b_sc, b_max_hist, kStrideSel, g_own.stream);
                        }
                    }
                    if (profiling_enabled) {
                        hip_check(hipStreamSynchronize(g_own.stream), "prof dsa tp4 fi");
                        prof.dsa_fi_topk_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0_start).count();
                    }
                }

                // C1: TP4 Parallel Dense Causal DSA (pos < 2176): broadcast <= 2.53 MiB FP16 KV into w.d_tmp_a -> 4x parallel GEMM!
                if (n_dense > 0) {
                    const auto t_dm0 = (profiling_enabled || M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const int k_lim_all = pos0 + n_dense;
                    const int k_pad_all = ((k_lim_all + 255) / 256) * 256;
                    hip_check(hipSetDevice(g_own.hip_dev), "dsa tp4 dense kv dev");
                    launch_bf16_to_f16(L.d_dsa_mla_kv, g_own.d_tmp_a, k_lim_all * 576, g_own.stream);
                    if (k_lim_all < k_pad_all) {
                        hip_check(hipMemsetAsync(g_own.d_tmp_a + size_t(k_lim_all) * 576u, 0,
                                                 size_t(k_pad_all - k_lim_all) * 576u * sizeof(uint16_t), g_own.stream), "zero tp4 dense k pad");
                    }
                    const size_t kv_bytes = size_t(k_pad_all) * 576u * sizeof(uint16_t);
                    hip_check(hipMemcpyAsync(h_tp_qabs_pin, g_own.d_tmp_a, kv_bytes,
                                             hipMemcpyDeviceToHost, g_own.stream), "dsa tp4 dense kv d2h");
                    hip_check(hipStreamSynchronize(g_own.stream), "dsa tp4 dense kv sync");
                    for (int r = 0; r < tp; ++r) {
                        if (r == ro) continue;
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "dsa tp4 dense kv h2d dev");
                        hip_check(hipMemcpyAsync(w.d_tmp_a, h_tp_qabs_pin, kv_bytes,
                                                 hipMemcpyHostToDevice, w.stream), "dsa tp4 dense kv h2d");
                    }

                    constexpr int kBStrip  = 256;
                    constexpr int kTileK   = 256;
                    constexpr int kRowsPad = kBStrip * 16; // 4096 rows = constant Tensile Type_HH_HPA tile across all 4 GPUs!
                    const float alpha = 1.0f, beta = 0.0f;
                    const float alpha_dsa = 0.0625f;
                    for (int r = 0; r < tp; ++r) {
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& D = L.dsa_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "dsa tp4 dense compute dev");
                        uint16_t* d_probs_f16 = w.d_x_f16 + 5242880u; // [4096, 256] = 1M halfs
                        uint16_t* d_y16       = reinterpret_cast<uint16_t*>(w.d_y_f32);
                        float* d_m_tot = w.d_hca_ml_f32;
                        float* d_l_tot = w.d_hca_ml_f32 + size_t(kRowsPad);
                        float* d_m_blk = w.d_hca_ml_f32 + size_t(kRowsPad) * 2u;
                        float* d_l_blk = w.d_hca_ml_f32 + size_t(kRowsPad) * 3u;
                        for (int m0 = 0; m0 < n_dense; m0 += kBStrip) {
                            const int B = std::min(kBStrip, n_dense - m0);
                            const int n_rows = B * 16;
                            const int32_t pos0_strip = pos0 + m0;
                            const int k_lim = pos0_strip + B;
                            const uint16_t* d_q_strip_f16 = w.d_tmp_c + size_t(m0) * 16u * 576u;
                            if (n_rows < kRowsPad) {
                                hip_check(hipMemcpyAsync(w.d_x_f16, d_q_strip_f16,
                                                         size_t(n_rows) * 576u * sizeof(uint16_t),
                                                         hipMemcpyDeviceToDevice, w.stream), "d2d tp4 dense q pad");
                                hip_check(hipMemsetAsync(w.d_x_f16 + size_t(n_rows) * 576u, 0,
                                                         size_t(kRowsPad - n_rows) * 576u * sizeof(uint16_t), w.stream), "zero tp4 dense q pad");
                                hip_check(hipMemsetAsync(d_probs_f16 + size_t(n_rows) * size_t(kTileK), 0,
                                                         size_t(kRowsPad - n_rows) * size_t(kTileK) * sizeof(uint16_t), w.stream), "zero tp4 dense p pad");
                                d_q_strip_f16 = w.d_x_f16;
                            }
                            for (int k0 = 0; k0 < k_lim; k0 += kTileK) {
                                const uint16_t* d_k_tile_f16 = w.d_tmp_a + size_t(k0) * 576u;
                                rocblas_gemm_ex_tuned(
                                    w.blas, rocblas_operation_transpose, rocblas_operation_none,
                                    kTileK, kRowsPad, 576, &alpha_dsa,
                                    d_k_tile_f16,  rocblas_datatype_f16_r, 576,
                                    d_q_strip_f16, rocblas_datatype_f16_r, 576,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, kTileK,
                                    d_y16, rocblas_datatype_f16_r, kTileK, "dsa tp4 dense QK hh_hpa");
                                launch_dsa_dense_softmax_tile_f16_in_norm(
                                    d_y16, d_m_blk, d_l_blk, d_probs_f16,
                                    B, 16, pos0_strip, k0, kTileK, w.stream);
                                rocblas_gemm_ex_tuned(
                                    w.blas, rocblas_operation_none, rocblas_operation_none,
                                    512, kRowsPad, kTileK, &alpha,
                                    d_k_tile_f16, rocblas_datatype_f16_r, 576,
                                    d_probs_f16,  rocblas_datatype_f16_r, kTileK,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, 512,
                                    d_y16, rocblas_datatype_f16_r, 512, "dsa tp4 dense PV hh_hpa");
                                launch_hca_comp_merge_tile_norm_f16(
                                    w.d_hca_o_tot_f32, d_m_tot, d_l_tot,
                                    d_y16, d_m_blk, d_l_blk, n_rows, k0 == 0, w.stream);
                            }
                            launch_dsa_dense_finalize_sink_f16(
                                w.d_hca_o_tot_f32, d_m_tot, d_l_tot, D.d_attn_sink + r * 16,
                                w.d_tmp_b + size_t(m0) * 16u * 512u,
                                B, 16, w.stream);
                        }
                    }
                    if (profiling_enabled) {
                        prof_sync_tp(tp, L.is_echo, L.owner_dev);
                        prof.dsa_mla_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dm0).count();
                    }
                }

                // C2: TP4 Parallel 16-Head Sparse MQA (pos >= 2176):
                //     Unique-KV compaction + adaptive cur_sp halving (<= 116,000 keys -> fits 128 MiB w.d_tmp_a at 1M!)
                //     + dynamic u16/i32 remap (fits disjoint upper 32 MiB of w.d_tmp_b!)
                //     + OVERLAPPED owner MQA on g_own.stream || host-SDMA D2H on g_own.stream_sh -> 3x non-owner H2D + MQA!
                if (n_dense < M) {
                    const auto t_c2_0 = (profiling_enabled || M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const int total_kv = pos0 + M;
                    int32_t* d_map       = reinterpret_cast<int32_t*>(g_own.d_idx_scores);
                    int32_t* d_uniq_kpos = d_map + total_kv;
                    int32_t* d_uniq_cnt  = g_own.d_csr_counts;
                    const size_t sel_rem_off_u16 = size_t(M) * 16u * 512u; // disjoint upper 32 MiB of w.d_tmp_b!
                    const int max_uniq_cap = std::max(1024, int((size_t(max_chunk) * 32768u) / 576u) - 64);
                    int32_t* h_sel_cnts_stage = h_dsa_sel_cnts_pin;

                    for (int sp0 = n_dense; sp0 < M; ) {
                        int cur_sp = M - sp0;
                        int32_t h_uniq_cnt = 0;
                        hip_check(hipSetDevice(g_own.hip_dev), "dsa tp4 compact dev");
                        for (;;) {
                            // launch_dsa_mark_selected_keys & launch_dsa_compact_unique_map zero d_map/d_uniq_cnt on stream!
                            launch_dsa_mark_selected_keys(
                                g_own.d_sel_idx + size_t(sp0) * size_t(kStrideSel),
                                g_own.d_sel_cnts + sp0, d_map, cur_sp, kStrideSel, total_kv, g_own.stream);
                            launch_dsa_compact_unique_map(d_map, d_uniq_kpos, d_uniq_cnt, total_kv, g_own.stream);
                            hip_check(hipMemcpyAsync(&h_uniq_cnt, d_uniq_cnt, sizeof(int32_t),
                                                     hipMemcpyDeviceToHost, g_own.stream), "dsa uniq_cnt d2h");
                            hip_check(hipStreamSynchronize(g_own.stream), "dsa uniq_cnt sync");
                            if (h_uniq_cnt <= max_uniq_cap || cur_sp <= 32) break;
                            cur_sp = std::max(32, cur_sp / 2);
                        }
                        if (M >= 256 && sp0 == n_dense) {
                            chunk_dsa_fi_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c0_start).count();
                        }
                        const auto t_mq0 = (M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                        const bool use_u16 = (h_uniq_cnt <= 65535);
                        const size_t sel_elem_bytes = use_u16 ? sizeof(uint16_t) : sizeof(int32_t);
                        const size_t kv_uniq_bytes  = size_t(h_uniq_cnt) * 576u * sizeof(uint16_t);
                        const size_t sel_rem_bytes  = size_t(cur_sp) * size_t(kStrideSel) * sel_elem_bytes;
                        const size_t sel_cnt_bytes  = size_t(cur_sp) * sizeof(int32_t);
                        void* d_sel_rem_own = static_cast<void*>(g_own.d_tmp_b + sel_rem_off_u16);

                        if (h_uniq_cnt > 0) {
                            launch_dsa_gather_unique_kv_f16(L.d_dsa_mla_kv, d_uniq_kpos, g_own.d_tmp_a, h_uniq_cnt, g_own.stream);
                            launch_dsa_remap_sel_idx(
                                g_own.d_sel_idx + size_t(sp0) * size_t(kStrideSel),
                                g_own.d_sel_cnts + sp0, d_map, d_sel_rem_own, cur_sp, kStrideSel, use_u16, g_own.stream);
                        }
                        hip_check(hipEventRecord(g_own.ev_mb_c2d[0], g_own.stream), "dsa tp4 compact rec");

                        // OVERLAPPED: Launch owner sparse MQA on g_own.stream immediately while stream_sh D2Hs unique KV!
                        launch_dsa_sparse_mqa_tp4_h16(
                            g_own.d_tmp_c + size_t(sp0) * 16u * 576u,
                            L.d_dsa_mla_kv, D_own.d_attn_sink + ro * 16,
                            g_own.d_sel_idx + size_t(sp0) * size_t(kStrideSel),
                            g_own.d_sel_cnts + sp0,
                            g_own.d_tmp_b + size_t(sp0) * 16u * 512u,
                            cur_sp, kStrideSel, /*kv_is_f16=*/false, /*idx_is_u16=*/false, g_own.stream);

                        // Wave-11 Turn-7: 4-Stripe Full-Duplex Device-Ordered PCIe Pipeline (0 host syncs!):
                        // While g_own.stream_sh D2Hs stripe c+1 of unique KV, all 3 non-owners pull stripe c H2D
                        // simultaneously over full-duplex PCIe Gen4 via pre-allocated g_own.ev_mb_o_d2h[0..3]!
                        hip_check(hipStreamWaitEvent(g_own.stream_sh, g_own.ev_mb_c2d[0], 0), "dsa tp4 d2h wait");
                        const int n_pipe = (h_uniq_cnt > 0 && kv_uniq_bytes >= (4u << 20)) ? 4 : 1;
                        size_t pipe_off[4]{}, pipe_len[4]{};
                        if (h_uniq_cnt > 0) {
                            const size_t base_b = ((kv_uniq_bytes / size_t(n_pipe)) + 4095u) & ~size_t(4095u);
                            for (int p = 0; p < n_pipe; ++p) {
                                pipe_off[p] = std::min(kv_uniq_bytes, size_t(p) * base_b);
                                const size_t end_b = (p + 1 == n_pipe) ? kv_uniq_bytes : std::min(kv_uniq_bytes, size_t(p + 1) * base_b);
                                pipe_len[p] = (end_b > pipe_off[p]) ? (end_b - pipe_off[p]) : 0u;
                            }
                            hip_check(hipMemcpyAsync(h_tp_qabs_rm_pin, d_sel_rem_own, sel_rem_bytes,
                                                     hipMemcpyDeviceToHost, g_own.stream_sh), "dsa tp4 sel_rem d2h");
                        }
                        hip_check(hipMemcpyAsync(h_sel_cnts_stage, g_own.d_sel_cnts + sp0, sel_cnt_bytes,
                                                 hipMemcpyDeviceToHost, g_own.stream_sh), "dsa tp4 sel_cnts d2h");
                        for (int p = 0; p < n_pipe; ++p) {
                            if (pipe_len[p] > 0u) {
                                hip_check(hipMemcpyAsync(reinterpret_cast<uint8_t*>(h_tp_qabs_pin) + pipe_off[p],
                                                         reinterpret_cast<const uint8_t*>(g_own.d_tmp_a) + pipe_off[p],
                                                         pipe_len[p], hipMemcpyDeviceToHost, g_own.stream_sh), "dsa tp4 kv_uniq pipe d2h");
                            }
                            hip_check(hipEventRecord(g_own.ev_mb_o_d2h[p], g_own.stream_sh), "dsa tp4 kv_uniq pipe rec");
                        }

                        // Broadcast host-pinned SDMA H2D to 3 non-owners with device-ordered stripe waits + 16-head FP16 sparse MQA!
                        for (int r = 0; r < tp; ++r) {
                            if (r == ro) continue;
                            auto& w = *dev_map[dev_for_r(r)];
                            const auto& D = L.dsa_tp[r];
                            hip_check(hipSetDevice(w.hip_dev), "dsa tp4 remote dev");
                            void* d_sel_rem_w = static_cast<void*>(w.d_tmp_b + sel_rem_off_u16);
                            int32_t* d_sel_cnts_w = reinterpret_cast<int32_t*>(w.d_tmp_d);
                            hip_check(hipStreamWaitEvent(w.stream, g_own.ev_mb_o_d2h[0], 0), "dsa tp4 remote wait0");
                            if (h_uniq_cnt > 0) {
                                hip_check(hipMemcpyAsync(d_sel_rem_w, h_tp_qabs_rm_pin, sel_rem_bytes,
                                                         hipMemcpyHostToDevice, w.stream), "dsa tp4 sel_rem h2d");
                            }
                            hip_check(hipMemcpyAsync(d_sel_cnts_w, h_sel_cnts_stage, sel_cnt_bytes,
                                                     hipMemcpyHostToDevice, w.stream), "dsa tp4 sel_cnts h2d");
                            for (int p = 0; p < n_pipe; ++p) {
                                if (p > 0) {
                                    hip_check(hipStreamWaitEvent(w.stream, g_own.ev_mb_o_d2h[p], 0), "dsa tp4 remote waitp");
                                }
                                if (pipe_len[p] > 0u) {
                                    hip_check(hipMemcpyAsync(reinterpret_cast<uint8_t*>(w.d_tmp_a) + pipe_off[p],
                                                             reinterpret_cast<const uint8_t*>(h_tp_qabs_pin) + pipe_off[p],
                                                             pipe_len[p], hipMemcpyHostToDevice, w.stream), "dsa tp4 kv_uniq pipe h2d");
                                }
                            }
                            launch_dsa_sparse_mqa_tp4_h16(
                                w.d_tmp_c + size_t(sp0) * 16u * 576u,
                                w.d_tmp_a, D.d_attn_sink + r * 16,
                                d_sel_rem_w, d_sel_cnts_w,
                                w.d_tmp_b + size_t(sp0) * 16u * 512u,
                                cur_sp, kStrideSel, /*kv_is_f16=*/true, /*idx_is_u16=*/use_u16, w.stream);
                        }
                        sp0 += cur_sp;
                        if (sp0 < M) {
                            // Multi-slice rare guard (512K-1M adversarial n_uniq > 116K): drain non-owners before next slice overwrites d_tmp_a
                            for (int r = 0; r < tp; ++r) {
                                if (r == ro) continue;
                                auto& w = *dev_map[dev_for_r(r)];
                                hip_check(hipSetDevice(w.hip_dev), "dsa tp4 slice sync dev");
                                hip_check(hipStreamSynchronize(w.stream), "dsa tp4 slice sync");
                            }
                        }
                        if (M >= 256) {
                            chunk_dsa_mqa_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mq0).count();
                        }
                    }
                    if (profiling_enabled) {
                        prof_sync_tp(tp, L.is_echo, L.owner_dev);
                        prof.dsa_mla_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c2_0).count();
                    }
                }
            } else {
                // Canonical Single-Owner / Decode (M <= 8) Path — 100% preserved!
                if (n_dense > 0) {
                    const auto t_dm0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    if (M <= 8) {
                        launch_dsa_prefill_mla_dense_causal_sink(
                            g_own.d_tmp_c, L.d_dsa_mla_kv, D_own.d_attn_sink, g_own.d_tmp_a, kDsaNumHeads, pos0, n_dense, g_own.stream);
                    } else {
                        constexpr int kBStrip = 64;
                        constexpr int kTileK  = 256;
                        constexpr int kRowsPad = kBStrip * kDsaNumHeads;
                        const float alpha = 1.0f, beta = 0.0f;
                        uint16_t* d_q_strip_f16 = g_own.d_x_f16;
                        uint16_t* d_k_tile_f16  = g_own.d_x_f16 + 4194304u;
                        uint16_t* d_probs_f16   = g_own.d_x_f16 + 5242880u;
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
                                uint16_t* d_y16 = reinterpret_cast<uint16_t*>(g_own.d_y_f32);
                                const float alpha_dsa = 0.0625f;
                                launch_bf16_to_f16(L.d_dsa_mla_kv + size_t(k0) * 576u, d_k_tile_f16, k_cur * 576, g_own.stream);
                                if (k_cur < kTileK) {
                                    hip_check(hipMemsetAsync(d_k_tile_f16 + size_t(k_cur) * 576u, 0,
                                                             size_t(kTileK - k_cur) * 576u * sizeof(uint16_t), g_own.stream), "zero dsa dense k pad");
                                }
                                rocblas_gemm_ex_tuned(
                                    g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                    kTileK, kRowsPad, 576, &alpha_dsa,
                                    d_k_tile_f16,  rocblas_datatype_f16_r, 576,
                                    d_q_strip_f16, rocblas_datatype_f16_r, 576,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, kTileK,
                                    d_y16, rocblas_datatype_f16_r, kTileK, "dsa dense QK hh_hpa");
                                launch_dsa_dense_softmax_tile_f16_in_norm(
                                    d_y16, d_m_blk, d_l_blk, d_probs_f16,
                                    B, kDsaNumHeads, pos0_strip, k0, kTileK, g_own.stream);
                                rocblas_gemm_ex_tuned(
                                    g_own.blas, rocblas_operation_none, rocblas_operation_none,
                                    512, kRowsPad, kTileK, &alpha,
                                    d_k_tile_f16,  rocblas_datatype_f16_r, 576,
                                    d_probs_f16,   rocblas_datatype_f16_r, kTileK,
                                    &beta,
                                    d_y16, rocblas_datatype_f16_r, 512,
                                    d_y16, rocblas_datatype_f16_r, 512, "dsa dense PV hh_hpa");
                                launch_hca_comp_merge_tile_norm_f16(
                                    g_own.d_hca_o_tot_f32, d_m_tot, d_l_tot,
                                    d_y16, d_m_blk, d_l_blk, n_rows, k0 == 0, g_own.stream);
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
                                constexpr int kTileK = 4096;
                                constexpr int kSubB  = 256;
                                const int b_pad = ((b_sc + kSubB - 1) / kSubB) * kSubB;
                                uint16_t* d_q_all_f16 = g_own.d_x_f16;
                                uint16_t* d_k_tile_f16 = g_own.d_x_f16 + 3145728u;
                                uint16_t* d_y16 = reinterpret_cast<uint16_t*>(g_own.d_y_f32);
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
                                        const int b_eff = std::min(kSubB, ((b_cur + 63) / 64) * 64);
                                        rocblas_gemm_ex_tuned(
                                            g_own.blas, rocblas_operation_transpose, rocblas_operation_none,
                                            kTileK, b_eff * 8, 128, &alpha,
                                            d_k_tile_f16,                        rocblas_datatype_f16_r, 128,
                                            d_q_all_f16 + size_t(b0) * 8u * 128u, rocblas_datatype_f16_r, 128,
                                            &beta,
                                            d_y16, rocblas_datatype_f16_r, kTileK,
                                            d_y16, rocblas_datatype_f16_r, kTileK, "fi gemm_ex hh_hpa");
                                        launch_flashindexer_relu_wsum_f16_in(
                                            d_y16, g_own.d_fi_w8,
                                            g_own.d_idx_scores + size_t(b0) * size_t(b_max_hist),
                                            pos0, t0 + b0, b_cur, k0, kTileK, b_max_hist, g_own.stream);
                                    }
                                }
                            } else if (do_fi) {
                                // Wave-11 Cut D3.3: Direct-BF16 uint4 + v_dot2_f32_f16 persistent 60-CU FlashIndexer decode (>800 GB/s HBM2!)
                                if (b_sc <= 8) {
                                    launch_flashindexer_decode_m8_splitk(
                                        g_own.d_fi_q8, g_own.d_fi_w8, L.d_dsa_ik_cache, g_own.d_idx_scores,
                                        pos0, t0, b_sc, b_max_hist, g_own.stream);
                                } else {
                                    launch_dsa_indexer_score_topk8_dpp4(
                                        g_own.d_fi_q8, g_own.d_fi_w8, L.d_dsa_ik_cache, g_own.d_idx_scores,
                                        pos0, t0, b_sc, b_max_hist, g_own.stream);
                                }
                            } else {
                                launch_dsa_indexer_score_full_batch(
                                    g_own.d_tmp_b, g_own.d_tmp_d, L.d_dsa_ik_cache, g_own.d_idx_scores,
                                    pos0, t0, b_sc, b_max_hist, g_own.stream);
                            }
                        }
                        if (b_sc >= 64) {
                            launch_dsa_select_topk2048_gpu(
                                g_own.d_idx_scores, g_own.d_sel_idx, g_own.d_sel_cnts,
                                pos0, t0, b_sc, b_max_hist, kStrideSel, g_own.stream);
                        } else {
                            launch_dsa_select_topk2048_wave6(
                                g_own.d_idx_scores, reinterpret_cast<uint32_t*>(g_own.d_y_f32),
                                g_own.d_sel_idx, g_own.d_sel_cnts,
                                pos0, t0, b_sc, b_max_hist, kStrideSel, g_own.stream);
                        }
                    }
                    if (profiling_enabled) {
                        hip_check(hipStreamSynchronize(g_own.stream), "prof dsa fi");
                        prof.dsa_fi_topk_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_df0).count();
                    }

                    const auto t_dm1 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    if (M <= 8) {
                        // Wave-11 Cut D3.3: Zero-Gather 60-CU Split-K Sparse MQA Decode + 1-Wave Combine (0 B new VRAM, 4 strict invariants!)
                        const int n_sp = M - n_dense;
                        void* d_mqa_scratch = static_cast<void*>(g_own.d_tmp_a + 1048576u); // disjoint +2 MiB inside 128 MiB d_tmp_a
                        launch_dsa_sparse_mqa_decode_splitk(
                            g_own.d_tmp_c + size_t(n_dense) * size_t(kDsaNumHeads) * 576u,
                            L.d_dsa_mla_kv,
                            D_own.d_attn_sink,
                            g_own.d_sel_idx + size_t(n_dense) * size_t(kStrideSel),
                            g_own.d_sel_cnts + n_dense,
                            d_mqa_scratch,
                            g_own.d_tmp_a + size_t(n_dense) * size_t(kDsaNumHeads) * 512u,
                            n_sp, kDsaNumHeads / 16, kStrideSel, pos0 + n_dense,
                            /*q_is_f16=*/false, /*kv_is_f16=*/false, /*idx_is_u16=*/false, /*out_is_f16=*/false,
                            g_own.stream);
                    } else {
                        launch_dsa_sparse_mqa64_fused_vdot2(
                            g_own.d_tmp_c + size_t(n_dense) * size_t(kDsaNumHeads) * 576u,
                            L.d_dsa_mla_kv, D_own.d_attn_sink,
                            g_own.d_sel_idx + size_t(n_dense) * size_t(kStrideSel),
                            g_own.d_sel_cnts + n_dense,
                            g_own.d_tmp_a + size_t(n_dense) * size_t(kDsaNumHeads) * 512u,
                            M - n_dense, kStrideSel, g_own.stream);
                    }
                    if (profiling_enabled) {
                        hip_check(hipStreamSynchronize(g_own.stream), "prof dsa mla");
                        prof.dsa_mla_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dm1).count();
                    }
                }
            }

            // Phase D: Zero-Host-Sync Mapped Scatter o_lat (M<=8) or 1D Contiguous Scatter (SKIPPED when use_tp4_dsa!)
            if (!use_tp4_dsa && tp > 1) {
                const auto t_dc0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (M <= 8 && tp == 4 && h_tp_mapped_slabs != nullptr && gfx906_mapped_tp4_enabled()) {
                    const uint32_t ep = ++tp_mapped_epoch;
                    for (int r = 0; r < 4; ++r) {
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "dsa olat map dev");
                        launch_tp4_mapped_dsa_scatter_olat_bf16(
                            (r == ro) ? g_own.d_tmp_a : nullptr, w.d_tmp_a,
                            w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                            r, ro, M, kMappedSlabStride, ep, w.stream);
                    }
                } else {
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
                }
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dc0).count();
                }
            }

            // Phase E: Parallel Type_HH_HPA W_UV/4 -> Rank-16 Low-Rank VHA / AllGather -> LRGA -> o_proj/4 -> AllReduce
            const auto t_dp1 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dsa_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dsa E1 dev");

                if (M <= 8) {
                    launch_dsa_o_project_uv_f16(w.d_tmp_a, D.d_kv_b, w.d_tmp_b, M, h_loc, w.stream);
                } else if (use_tp4_dsa) {
                    // Wave-10 C3: Zero-bounce direct FP16 W_UV (reads w.d_tmp_b FP16 -> writes w.d_tmp_c FP16!)
                    const float alpha = 1.0f, beta = 0.0f;
                    uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                    const int uv_cap = int(std::min((512u * 16384u) / (size_t(h_loc) * 512u),
                                                    (512u * 16384u * 2u) / (size_t(h_loc) * 256u)));
                    const int uv_strip = std::max(64, std::min(1024, (uv_cap / 64) * 64));
                    for (int m0 = 0; m0 < M; m0 += uv_strip) {
                        const int cur = std::min(uv_strip, M - m0);
                        const int eff = std::min(uv_strip, ((cur + 63) / 64) * 64);
                        const uint16_t* d_uv_in = w.d_tmp_b + size_t(m0) * size_t(h_loc) * 512u;
                        uint16_t* d_dst_uv = w.d_tmp_c + size_t(m0) * size_t(h_loc) * 256u;
                        uint16_t* d_uv_out = (cur == eff) ? d_dst_uv : d_y16;
                        if (cur < eff) {
                            hip_check(hipMemcpyAsync(w.d_x_f16, d_uv_in,
                                                     size_t(cur) * size_t(h_loc) * 512u * sizeof(uint16_t),
                                                     hipMemcpyDeviceToDevice, w.stream), "d2d uv f16 pad");
                            hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(h_loc) * 512u, 0,
                                                     size_t(eff - cur) * size_t(h_loc) * 512u * sizeof(uint16_t), w.stream), "zero uv f16 pad");
                            d_uv_in = w.d_x_f16;
                        }
                        rocblas_gemm_strided_batched_ex_tuned(
                            w.blas, rocblas_operation_transpose, rocblas_operation_none,
                            256, eff, 512, &alpha,
                            D.d_kv_b + 192u * 512u, rocblas_datatype_f16_r, 512, 448LL * 512LL,
                            d_uv_in,                rocblas_datatype_f16_r, h_loc * 512, 512LL,
                            &beta,
                            d_uv_out, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                            d_uv_out, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                            h_loc, "dsa o_proj hh_hpa f16 direct");
                        if (cur < eff) {
                            hip_check(hipMemcpyAsync(d_dst_uv, d_y16,
                                                     size_t(cur) * size_t(h_loc) * 256u * sizeof(uint16_t),
                                                     hipMemcpyDeviceToDevice, w.stream), "d2d uv f16 tail");
                        }
                    }
                } else {
                    const float alpha = 1.0f, beta = 0.0f;
                    uint16_t* d_y16 = reinterpret_cast<uint16_t*>(w.d_y_f32);
                    const int uv_cap = int(std::min((512u * 16384u) / (size_t(h_loc) * 512u),
                                                    (512u * 16384u * 2u) / (size_t(h_loc) * 256u)));
                    const int uv_strip = std::max(64, std::min(1024, (uv_cap / 64) * 64));
                    for (int m0 = 0; m0 < M; m0 += uv_strip) {
                        const int cur = std::min(uv_strip, M - m0);
                        const int eff = std::min(uv_strip, ((cur + 63) / 64) * 64);
                        launch_bf16_to_f16(w.d_tmp_a + size_t(m0) * size_t(h_loc) * 512u, w.d_x_f16, cur * h_loc * 512, w.stream);
                        if (cur < eff) {
                            hip_check(hipMemsetAsync(w.d_x_f16 + size_t(cur) * size_t(h_loc) * 512u, 0,
                                                     size_t(eff - cur) * size_t(h_loc) * 512u * sizeof(uint16_t), w.stream), "zero uv pad");
                        }
                        rocblas_gemm_strided_batched_ex_tuned(
                            w.blas, rocblas_operation_transpose, rocblas_operation_none,
                            256, eff, 512, &alpha,
                            D.d_kv_b + 192u * 512u, rocblas_datatype_f16_r, 512, 448LL * 512LL,
                            w.d_x_f16,              rocblas_datatype_f16_r, h_loc * 512, 512LL,
                            &beta,
                            d_y16, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                            d_y16, rocblas_datatype_f16_r, h_loc * 256, 256LL,
                            h_loc, "dsa o_proj hh_hpa batched");
                        launch_f16_to_bf16(d_y16, w.d_tmp_b + size_t(m0) * size_t(h_loc) * 256u, cur * h_loc * 256, w.stream);
                    }
                }
            }
            if (profiling_enabled) {
                prof_sync_tp(tp, L.is_echo, L.owner_dev);
                prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp1).count();
            }
            const bool use_dsa_vha_lr = (tp == 4 && M > 8 && L.dsa_tp[0].has_vha_lr);
            if (use_dsa_vha_lr && M >= 512) {
                // Wave-11 C5: PCIe 4.0 HBM2-Reduced 4-MB DSA AllReduce (0 CPU DDR4 sum4 stall, 0 B extra VRAM, bit-exact FP32 RNE!)
                const auto t_dp2 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                constexpr int kNb = 4;
                int mb_off[kNb]{}, mb_len[kNb]{};
                const int base_mb = (M / kNb) & ~63;
                for (int b = 0; b < kNb; ++b) {
                    mb_off[b] = b * base_mb;
                    mb_len[b] = (b + 1 < kNb) ? base_mb : (M - b * base_mb);
                }
                constexpr int kOrdH2d[4] = {2, 1, 3, 0};
                constexpr int kOrdD2h[4] = {2, 1, 3, 0};
                const size_t dsa_cols = 16u * 256u; // 4096
                const size_t dsa_end  = size_t(M) * dsa_cols;
                const size_t mb_max_rows = size_t(std::max(mb_len[0], mb_len[kNb - 1]));
                const size_t mb_dsa_max  = mb_max_rows * dsa_cols;
                const size_t mb_op_max   = mb_max_rows * kHiddenSize;
                const size_t tmp_a_cap   = size_t(max_chunk) * 32768u;
                const bool defer_ok   = (defer_nonowner_oproj && !L.is_echo && tp == 4);
                const bool use_gpu_ar = gfx906_gpu_ar_enabled() && !L.is_echo && (tp == 4) &&
                                        (L.owner_tp_rank >= 0 && L.owner_tp_rank < 4) &&
                                        (dsa_end + 6u * mb_dsa_max + 12u * mb_op_max <= tmp_a_cap);

                if (use_gpu_ar) {
                    // Wave-11 Cut P1.1: Dual-Owner 2 x 1024 Full-Duplex HBM2 DSA VHA AllReduce across g0 & g1!
                    const int own_r = L.owner_tp_rank;
                    auto& g0 = *dev_map[dev_for_r(0)];
                    auto& g1 = *dev_map[dev_for_r(1)];
                    auto& g_own = *dev_map[L.owner_dev];
                    const int mb2_off[2] = { 0, (M / 2) & ~63 };
                    const int mb2_len[2] = { mb2_off[1], M - mb2_off[1] };
                    const size_t mb2_max_rows = size_t(std::max(mb2_len[0], mb2_len[1]));
                    const size_t mb2_dsa_max  = mb2_max_rows * dsa_cols;
                    const size_t mb2_op_max   = mb2_max_rows * kHiddenSize;
                    const bool dual_ok = (mb2_len[0] >= 256 && mb2_len[1] >= 256 &&
                                          dsa_end + 3u * mb2_dsa_max + 6u * mb2_op_max <= tmp_a_cap);

                    if (dual_ok) {
                        const size_t voff0 = 0u, voff1 = size_t(mb2_off[1]) * dsa_cols;
                        const size_t ooff0 = 0u, ooff1 = size_t(mb2_off[1]) * kHiddenSize;
                        const size_t dsa0_n = size_t(mb2_len[0]) * dsa_cols, dsa1_n = size_t(mb2_len[1]) * dsa_cols;
                        const size_t sb_dsa0 = dsa0_n * sizeof(uint16_t), sb_dsa1 = dsa1_n * sizeof(uint16_t);
                        const size_t sb_op0  = size_t(mb2_len[0]) * kHiddenSize * sizeof(uint16_t);
                        const size_t sb_op1  = size_t(mb2_len[1]) * kHiddenSize * sizeof(uint16_t);
                        const size_t vha_red = dsa_end;
                        const size_t op_red  = dsa_end + 3u * mb2_dsa_max;

                        constexpr int kLaunchOrd[4] = {0, 1, 3, 2};
                        for (int oi = 0; oi < 4; ++oi) {
                            const int r = kLaunchOrd[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            const auto& D = L.dsa_tp[r];
                            hip_check(hipSetDevice(w.hip_dev), "dsa dual init dev");
                            uint16_t* h_base = h_tp_oattn_pin + size_t(r) * size_t(M) * dsa_cols;
                            if (r == 0) {
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_c + voff1, D.d_vha_r_loc, w.d_tmp_a + voff1, mb2_len[1], 256, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[1], w.stream), "dsa dual g0 c2d1");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[1], 0), "dsa dual g0 wait1");
                                hip_check(hipMemcpyAsync(h_base + voff1, w.d_tmp_a + voff1, sb_dsa1, hipMemcpyDeviceToHost, w.stream_sh), "dsa dual g0 d2h1");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[1], w.stream_sh), "dsa dual g0 d2h1 rec");
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_c, D.d_vha_r_loc, w.d_tmp_a, mb2_len[0], 256, w.stream);
                            } else if (r == 1) {
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_c, D.d_vha_r_loc, w.d_tmp_a, mb2_len[0], 256, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "dsa dual g1 c2d0");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "dsa dual g1 wait0");
                                hip_check(hipMemcpyAsync(h_base, w.d_tmp_a, sb_dsa0, hipMemcpyDeviceToHost, w.stream_sh), "dsa dual g1 d2h0");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "dsa dual g1 d2h0 rec");
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_c + voff1, D.d_vha_r_loc, w.d_tmp_a + voff1, mb2_len[1], 256, w.stream);
                            } else {
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_c, D.d_vha_r_loc, w.d_tmp_a, mb2_len[0], 256, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "dsa dual rem c2d0");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "dsa dual rem wait0");
                                hip_check(hipMemcpyAsync(h_base, w.d_tmp_a, sb_dsa0, hipMemcpyDeviceToHost, w.stream_sh), "dsa dual rem d2h0");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "dsa dual rem d2h0 rec");
                                launch_vha_lowrank_proj_f16_in(w.d_tmp_c + voff1, D.d_vha_r_loc, w.d_tmp_a + voff1, mb2_len[1], 256, w.stream);
                                hip_check(hipEventRecord(w.ev_mb_c2d[1], w.stream), "dsa dual rem c2d1");
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[1], 0), "dsa dual rem wait1");
                                hip_check(hipMemcpyAsync(h_base + voff1, w.d_tmp_a + voff1, sb_dsa1, hipMemcpyDeviceToHost, w.stream_sh), "dsa dual rem d2h1");
                                hip_check(hipEventRecord(w.ev_mb_vha_d2h[1], w.stream_sh), "dsa dual rem d2h1 rec");
                            }
                            if (r >= 2) {
                                if (dsa_stay_f16) f16_linear_f16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                                else              bf16_linear_bf16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                            }
                        }

                        const int ar_mode = gfx906_ar_device_mode();
                        const bool ar_dev = (ar_mode == 1) &&
                                            (d_tp_oattn_pin != nullptr && d_tp_qabs_pin != nullptr &&
                                             d_tp_ar_in_pin[0] != nullptr && d_tp_ar_in_pin[1] != nullptr &&
                                             d_tp_ar_in_pin[2] != nullptr && d_tp_ar_in_pin[3] != nullptr);
                        const bool ar_sdma = (ar_mode == 2);
                        auto relay_vha = [&](int src_r, int mb, Mi50ExpertWorker& dst_g) {
                            auto& ws = *dev_map[dev_for_r(src_r)];
                            hip_check(hipSetDevice(dst_g.hip_dev), "dsa dual relay dev");
                            hip_check(hipStreamWaitEvent(dst_g.stream, ws.ev_mb_vha_d2h[mb], 0), "dsa dual relay wait");
                        };
                        auto sync_and_h2d = [&](int src_r, int mb, Mi50ExpertWorker& dst_g, size_t dst_slot_idx) {
                            auto& ws = *dev_map[dev_for_r(src_r)];
                            const size_t vo = (mb == 0) ? voff0 : voff1;
                            const size_t sb = (mb == 0) ? sb_dsa0 : sb_dsa1;
                            if (ar_sdma) {
                                hip_check(hipSetDevice(dst_g.hip_dev), "dsa dual h2d dst");
                                hip_check(hipStreamWaitEvent(dst_g.stream_sh, ws.ev_mb_vha_d2h[mb], 0), "dsa dual sdma wait");
                            } else {
                                hip_check(hipSetDevice(ws.hip_dev), "dsa dual sync src");
                                hip_check(hipEventSynchronize(ws.ev_mb_vha_d2h[mb]), "dsa dual sync d2h");
                                hip_check(hipSetDevice(dst_g.hip_dev), "dsa dual h2d dst");
                            }
                            uint16_t* h_in = h_tp_oattn_pin + size_t(src_r) * size_t(M) * dsa_cols + vo;
                            hip_check(hipMemcpyAsync(dst_g.d_tmp_a + vha_red + dst_slot_idx * mb2_dsa_max, h_in,
                                                     sb, hipMemcpyHostToDevice, dst_g.stream_sh), "dsa dual h2d");
                        };
                        if (ar_dev) {
                            relay_vha(1, 0, g0); relay_vha(0, 1, g1);
                            relay_vha(3, 0, g0); relay_vha(3, 1, g1); relay_vha(2, 0, g0);
                        } else {
                            sync_and_h2d(1, 0, g0, 0u);
                            sync_and_h2d(0, 1, g1, 0u);
                            sync_and_h2d(3, 0, g0, 2u);
                            sync_and_h2d(3, 1, g1, 2u);
                            sync_and_h2d(2, 0, g0, 1u);
                        }
                        {
                            hip_check(hipSetDevice(g0.hip_dev), "dsa dual g0 sum dev");
                            if (ar_dev) {
                                launch_sum4_f16_canonical(
                                    g0.d_tmp_a,
                                    d_tp_oattn_pin + size_t(1) * size_t(M) * dsa_cols + voff0,
                                    d_tp_oattn_pin + size_t(2) * size_t(M) * dsa_cols + voff0,
                                    d_tp_oattn_pin + size_t(3) * size_t(M) * dsa_cols + voff0,
                                    g0.d_tmp_a, dsa0_n, g0.stream);
                            } else {
                                hip_check(hipEventRecord(g0.ev_mb_vha_h2d[0], g0.stream_sh), "dsa dual g0 h2d rec");
                                hip_check(hipStreamWaitEvent(g0.stream, g0.ev_mb_vha_h2d[0], 0), "dsa dual g0 wait h2d");
                                launch_sum4_f16_canonical(
                                    g0.d_tmp_a,
                                    g0.d_tmp_a + vha_red + 0u * mb2_dsa_max,
                                    g0.d_tmp_a + vha_red + 1u * mb2_dsa_max,
                                    g0.d_tmp_a + vha_red + 2u * mb2_dsa_max,
                                    g0.d_tmp_a, dsa0_n, g0.stream);
                            }
                            hip_check(hipEventRecord(g0.ev_mb_c2d[0], g0.stream), "dsa dual g0 sum rec");
                            hip_check(hipStreamWaitEvent(g0.stream_sh, g0.ev_mb_c2d[0], 0), "dsa dual g0 sh wait sum");
                            hip_check(hipMemcpyAsync(h_tp_qabs_pin, g0.d_tmp_a, sb_dsa0, hipMemcpyDeviceToHost, g0.stream_sh), "dsa dual g0 sum d2h");
                            hip_check(hipEventRecord(g0.ev_mb_vha_d2h[0], g0.stream_sh), "dsa dual g0 sum d2h rec");
                            const auto& D0 = L.dsa_tp[0];
                            if (dsa_stay_f16) f16_linear_f16(g0, g0.d_qr, D0.d_gate_proj, g0.d_tmp_d, M, h_loc * 256, 1024);
                            else              bf16_linear_bf16(g0, g0.d_qr, D0.d_gate_proj, g0.d_tmp_d, M, h_loc * 256, 1024);
                            if (dsa_stay_f16) launch_dsa_vha_lowrank_lrga_f16_all(g0.d_tmp_c, g0.d_tmp_a, D0.d_vha_q_loc, g0.d_tmp_d, g0.d_tmp_c, mb2_len[0], g0.stream);
                            else              launch_dsa_vha_lowrank_lrga_f16_io(g0.d_tmp_c, g0.d_tmp_a, D0.d_vha_q_loc, g0.d_tmp_d, g0.d_tmp_c, mb2_len[0], g0.stream);
                            f16_linear_f16(g0, g0.d_tmp_c, D0.d_o_proj, g0.d_sub_out, mb2_len[0], kHiddenSize, h_loc * 256);
                            hip_check(hipEventRecord(g0.ev_mb_d2c[0], g0.stream), "dsa dual g0 op0 rec");
                        }
                        if (ar_dev) relay_vha(2, 1, g1);
                        else        sync_and_h2d(2, 1, g1, 1u);
                        {
                            hip_check(hipSetDevice(g1.hip_dev), "dsa dual g1 sum dev");
                            if (ar_dev) {
                                launch_sum4_f16_canonical(
                                    d_tp_oattn_pin + size_t(0) * size_t(M) * dsa_cols + voff1,
                                    g1.d_tmp_a + voff1,
                                    d_tp_oattn_pin + size_t(2) * size_t(M) * dsa_cols + voff1,
                                    d_tp_oattn_pin + size_t(3) * size_t(M) * dsa_cols + voff1,
                                    g1.d_tmp_a + voff1, dsa1_n, g1.stream);
                            } else {
                                hip_check(hipEventRecord(g1.ev_mb_vha_h2d[1], g1.stream_sh), "dsa dual g1 h2d rec");
                                hip_check(hipStreamWaitEvent(g1.stream, g1.ev_mb_vha_h2d[1], 0), "dsa dual g1 wait h2d");
                                launch_sum4_f16_canonical(
                                    g1.d_tmp_a + vha_red + 0u * mb2_dsa_max,
                                    g1.d_tmp_a + voff1,
                                    g1.d_tmp_a + vha_red + 1u * mb2_dsa_max,
                                    g1.d_tmp_a + vha_red + 2u * mb2_dsa_max,
                                    g1.d_tmp_a + voff1, dsa1_n, g1.stream);
                            }
                            hip_check(hipEventRecord(g1.ev_mb_c2d[1], g1.stream), "dsa dual g1 sum rec");
                            hip_check(hipStreamWaitEvent(g1.stream_sh, g1.ev_mb_c2d[1], 0), "dsa dual g1 sh wait sum");
                            hip_check(hipMemcpyAsync(h_tp_qabs_pin + voff1, g1.d_tmp_a + voff1, sb_dsa1, hipMemcpyDeviceToHost, g1.stream_sh), "dsa dual g1 sum d2h");
                            hip_check(hipEventRecord(g1.ev_mb_vha_d2h[1], g1.stream_sh), "dsa dual g1 sum d2h rec");
                            const auto& D1 = L.dsa_tp[1];
                            if (dsa_stay_f16) f16_linear_f16(g1, g1.d_qr, D1.d_gate_proj, g1.d_tmp_d, M, h_loc * 256, 1024);
                            else              bf16_linear_bf16(g1, g1.d_qr, D1.d_gate_proj, g1.d_tmp_d, M, h_loc * 256, 1024);
                            if (dsa_stay_f16) launch_dsa_vha_lowrank_lrga_f16_all(g1.d_tmp_c + voff1, g1.d_tmp_a + voff1, D1.d_vha_q_loc, g1.d_tmp_d + voff1, g1.d_tmp_c + voff1, mb2_len[1], g1.stream);
                            else              launch_dsa_vha_lowrank_lrga_f16_io(g1.d_tmp_c + voff1, g1.d_tmp_a + voff1, D1.d_vha_q_loc, g1.d_tmp_d + voff1, g1.d_tmp_c + voff1, mb2_len[1], g1.stream);
                            f16_linear_f16(g1, g1.d_tmp_c + voff1, D1.d_o_proj, g1.d_sub_out + ooff1, mb2_len[1], kHiddenSize, h_loc * 256);
                            hip_check(hipEventRecord(g1.ev_mb_d2c[1], g1.stream), "dsa dual g1 op1 rec");
                        }

                        auto launch_bcast_mb = [&](int mb) {
                            auto& red_g = (mb == 0) ? g0 : g1;
                            const int red_r = (mb == 0) ? 0 : 1;
                            const size_t vo = (mb == 0) ? voff0 : voff1;
                            const size_t oo = (mb == 0) ? ooff0 : ooff1;
                            const size_t sb_v = (mb == 0) ? sb_dsa0 : sb_dsa1;
                            if (!ar_dev && !ar_sdma) {
                                hip_check(hipSetDevice(red_g.hip_dev), "dsa dual red sync dev");
                                hip_check(hipEventSynchronize(red_g.ev_mb_vha_d2h[mb]), "dsa dual red sync");
                            }
                            const int targets[3] = { 2, 3, 1 - red_r };
                            for (int ti = 0; ti < 3; ++ti) {
                                const int r = targets[ti];
                                auto& w = *dev_map[dev_for_r(r)];
                                const auto& D = L.dsa_tp[r];
                                hip_check(hipSetDevice(w.hip_dev), "dsa dual tgt dev");
                                uint16_t* d_cb = w.d_tmp_c + vo;
                                const uint16_t* p_vha_in = w.d_tmp_a + vo;
                                if (ar_dev) {
                                    hip_check(hipStreamWaitEvent(w.stream, red_g.ev_mb_vha_d2h[mb], 0), "dsa dual tgt r wait");
                                    p_vha_in = d_tp_qabs_pin + vo;
                                } else {
                                    if (ar_sdma) {
                                        hip_check(hipStreamWaitEvent(w.stream_sh, red_g.ev_mb_vha_d2h[mb], 0), "dsa dual tgt sdma wait");
                                    }
                                    hip_check(hipMemcpyAsync(w.d_tmp_a + vo, h_tp_qabs_pin + vo, sb_v, hipMemcpyHostToDevice, w.stream_sh), "dsa dual tgt h2d sh");
                                    hip_check(hipEventRecord(w.ev_mb_vha_h2d[mb], w.stream_sh), "dsa dual tgt h2d rec");
                                    hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_vha_h2d[mb], 0), "dsa dual tgt wait h2d");
                                }
                                if (dsa_stay_f16) launch_dsa_vha_lowrank_lrga_f16_all(d_cb, p_vha_in, D.d_vha_q_loc, w.d_tmp_d + vo, d_cb, mb2_len[mb], w.stream);
                                else              launch_dsa_vha_lowrank_lrga_f16_io(d_cb, p_vha_in, D.d_vha_q_loc, w.d_tmp_d + vo, d_cb, mb2_len[mb], w.stream);
                                f16_linear_f16(w, d_cb, D.d_o_proj, w.d_sub_out + oo, mb2_len[mb], kHiddenSize, h_loc * 256);
                                hip_check(hipEventRecord(w.ev_mb_d2c[mb], w.stream), "dsa dual tgt op rec");
                            }
                        };
                        launch_bcast_mb(0);
                        launch_bcast_mb(1);
                        for (int r = 0; r < 4; ++r) {
                            if (r == own_r) continue;
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa dual op d2h q dev");
                            const int mb_ord[2] = { (r == 1) ? 1 : 0, (r == 1) ? 0 : 1 };
                            for (int mi = 0; mi < 2; ++mi) {
                                const int mb = mb_ord[mi];
                                const size_t oo = (mb == 0) ? ooff0 : ooff1;
                                const size_t sb_o = (mb == 0) ? sb_op0 : sb_op1;
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[mb], 0), "dsa dual tgt op wait");
                                hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + oo, w.d_sub_out + oo,
                                                         sb_o, hipMemcpyDeviceToHost, w.stream_sh), "dsa dual tgt op d2h");
                                hip_check(hipEventRecord(w.ev_mb_o_d2h[mb], w.stream_sh), "dsa dual tgt op d2h rec");
                            }
                        }

                        constexpr int kOrdFastSyncB[2][4] = {{0, 3, 2, 1}, {1, 0, 3, 2}};
                        int non_own[3];
                        int ni = 0;
                        for (int oi = 0; oi < 4; ++oi) {
                            const int r = kOrdFastSyncB[0][oi];
                            if (r != own_r) non_own[ni++] = r;
                        }
                        for (int b = 0; b < 2; ++b) {
                            int non_own_b[3];
                            int nib = 0;
                            for (int oi = 0; oi < 4; ++oi) {
                                const int r = kOrdFastSyncB[b][oi];
                                if (r != own_r) non_own_b[nib++] = r;
                            }
                            const size_t ooff  = size_t(mb2_off[b]) * kHiddenSize;
                            const size_t mb_op = size_t(mb2_len[b]) * kHiddenSize;
                            const size_t sb_op = mb_op * sizeof(uint16_t);
                            const size_t op_slot_base = op_red + size_t(b) * 3u * mb2_op_max;
                            const uint16_t* p_rank[4] = {nullptr, nullptr, nullptr, nullptr};
                            p_rank[own_r] = g_own.d_sub_out + ooff;
                            for (int i = 0; i < 3; ++i) {
                                const int r = non_own_b[i];
                                auto& w = *dev_map[dev_for_r(r)];
                                if (ar_dev) {
                                    hip_check(hipSetDevice(g_own.hip_dev), "dsa dual op red dev");
                                    hip_check(hipStreamWaitEvent(g_own.stream, w.ev_mb_o_d2h[b], 0), "dsa dual op r wait");
                                    p_rank[r] = d_tp_ar_in_pin[r] + ooff;
                                } else {
                                    if (ar_sdma) {
                                        hip_check(hipSetDevice(g_own.hip_dev), "dsa dual op h2d dev");
                                        hip_check(hipStreamWaitEvent(g_own.stream_sh, w.ev_mb_o_d2h[b], 0), "dsa dual op sdma wait");
                                    } else {
                                        hip_check(hipSetDevice(w.hip_dev), "dsa dual op sync dev");
                                        hip_check(hipEventSynchronize(w.ev_mb_o_d2h[b]), "dsa dual op sync");
                                        hip_check(hipSetDevice(g_own.hip_dev), "dsa dual op h2d dev");
                                    }
                                    uint16_t* dst_slot = g_own.d_tmp_a + op_slot_base + size_t(i) * mb2_op_max;
                                    hip_check(hipMemcpyAsync(dst_slot, h_tp_ar_in_pin[r] + ooff,
                                                             sb_op, hipMemcpyHostToDevice, g_own.stream_sh), "dsa dual op h2d");
                                    p_rank[r] = dst_slot;
                                }
                            }
                            hip_check(hipSetDevice(g_own.hip_dev), "dsa dual op red dev");
                            if (!ar_dev) {
                                hip_check(hipEventRecord(g_own.ev_mb_o_d2h[b], g_own.stream_sh), "dsa dual op h2d rec");
                                hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_o_d2h[b], 0), "dsa dual op sum wait");
                            }
                            launch_sum4_f16_to_bf16_canonical(p_rank[0], p_rank[1], p_rank[2], p_rank[3],
                                                              g_own.d_sub_out + ooff, mb_op, g_own.stream);
                        }
                        hip_check(hipSetDevice(g_own.hip_dev), "dsa dual op host dev");
                        hip_check(hipEventRecord(g_own.ev_mb_c2d[3], g_own.stream), "dsa dual op sum rec");
                        hip_check(hipStreamWaitEvent(g_own.stream_sh, g_own.ev_mb_c2d[3], 0), "dsa dual op host wait");
                        hip_check(hipMemcpyAsync(h_tp_o_out_pin, g_own.d_sub_out,
                                                 size_t(M) * kHiddenSize * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, g_own.stream_sh), "dsa dual op host d2h");
                        hip_check(hipEventRecord(g_own.ev_mb_vha_d2h[3], g_own.stream_sh), "dsa dual op host done");

                        if (defer_ok) {
                            o_proj_host_d2h_pending = true;
                            if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = true;
                        } else {
                            hip_check(hipSetDevice(g_own.hip_dev), "dsa dual nondef sync dev");
                            hip_check(hipEventSynchronize(g_own.ev_mb_vha_d2h[3]), "dsa dual nondef sync");
                            const size_t full_op_b = size_t(M) * kHiddenSize * sizeof(uint16_t);
                            for (int i = 0; i < 3; ++i) {
                                const int r = non_own[i];
                                auto& w = *dev_map[dev_for_r(r)];
                                hip_check(hipSetDevice(w.hip_dev), "dsa dual nondef h2d dev");
                                hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_o_out_pin, full_op_b,
                                                         hipMemcpyHostToDevice, w.stream_sh), "dsa dual nondef h2d");
                                hip_check(hipEventRecord(w.ev_mb_d2c[3], w.stream_sh), "dsa dual nondef rec");
                                hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[3], 0), "dsa dual nondef wait");
                            }
                        }
                    } else {
                    constexpr int kRemLaunch[3] = {1, 3, 2}; // Turn-16: fast-first remote ranks (excludes g0=0)
                    constexpr int kRemSync[3]   = {1, 3, 2};
                    const size_t vha_red_base[2] = { dsa_end, dsa_end + 3u * mb_dsa_max };
                    const size_t op_red_base     = dsa_end + 6u * mb_dsa_max;

                    for (int ri = 0; ri < 3; ++ri) {
                        const int r = kRemLaunch[ri];
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& D = L.dsa_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "dsa gar init0 dev");
                        launch_vha_lowrank_proj_f16_in(w.d_tmp_c, D.d_vha_r_loc, w.d_tmp_a, mb_len[0], 256, w.stream);
                        hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "dsa gar c2d0");
                        hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "dsa gar wait0");
                        uint16_t* h_in0 = h_tp_oattn_pin + size_t(r) * size_t(M) * dsa_cols;
                        hip_check(hipMemcpyAsync(h_in0, w.d_tmp_a, size_t(mb_len[0]) * dsa_cols * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, w.stream_sh), "dsa gar vha0 d2h");
                        hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "dsa gar vha0 rec");
                    }
                    {
                        const auto& D0 = L.dsa_tp[0];
                        hip_check(hipSetDevice(g0.hip_dev), "dsa gar g0 init dev");
                        for (int b = 0; b < kNb; ++b) {
                            const size_t voff = size_t(mb_off[b]) * dsa_cols;
                            launch_vha_lowrank_proj_f16_in(g0.d_tmp_c + voff, D0.d_vha_r_loc, g0.d_tmp_a + voff,
                                                           mb_len[b], 256, g0.stream);
                        }
                        if (dsa_stay_f16) f16_linear_f16(g0, g0.d_qr, D0.d_gate_proj, g0.d_tmp_d, M, h_loc * 256, 1024);
                        else              bf16_linear_bf16(g0, g0.d_qr, D0.d_gate_proj, g0.d_tmp_d, M, h_loc * 256, 1024);
                    }
                    for (int ri = 0; ri < 3; ++ri) {
                        const int r = kRemLaunch[ri];
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& D = L.dsa_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "dsa gar init1 dev");
                        for (int b = 1; b < kNb; ++b) {
                            const size_t voff = size_t(mb_off[b]) * dsa_cols;
                            launch_vha_lowrank_proj_f16_in(w.d_tmp_c + voff, D.d_vha_r_loc, w.d_tmp_a + voff,
                                                           mb_len[b], 256, w.stream);
                            hip_check(hipEventRecord(w.ev_mb_c2d[b], w.stream), "dsa gar c2d b");
                            hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[b], 0), "dsa gar wait b");
                            uint16_t* h_in_b = h_tp_oattn_pin + size_t(r) * size_t(M) * dsa_cols + voff;
                            hip_check(hipMemcpyAsync(h_in_b, w.d_tmp_a + voff,
                                                     size_t(mb_len[b]) * dsa_cols * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, w.stream_sh), "dsa gar vha b d2h");
                            hip_check(hipEventRecord(w.ev_mb_vha_d2h[b], w.stream_sh), "dsa gar vha b rec");
                        }
                        if (dsa_stay_f16) f16_linear_f16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                        else              bf16_linear_bf16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                    }

                    auto launch_rem_mb = [&](int bm) {
                        const size_t voff_m = size_t(mb_off[bm]) * dsa_cols;
                        const size_t ooff_m = size_t(mb_off[bm]) * kHiddenSize;
                        const size_t sb_dsa_m = size_t(mb_len[bm]) * dsa_cols * sizeof(uint16_t);
                        const size_t sb_op_m  = size_t(mb_len[bm]) * kHiddenSize * sizeof(uint16_t);
                        hip_check(hipSetDevice(g0.hip_dev), "dsa gar g0 sum sync dev");
                        hip_check(hipEventSynchronize(g0.ev_mb_vha_d2h[bm]), "dsa gar g0 sum sync");
                        for (int ri = 0; ri < 3; ++ri) {
                            const int r = kRemLaunch[ri];
                            auto& w = *dev_map[dev_for_r(r)];
                            const auto& D = L.dsa_tp[r];
                            hip_check(hipSetDevice(w.hip_dev), "dsa gar rem comp dev");
                            hip_check(hipMemcpyAsync(w.d_tmp_a + voff_m, h_tp_qabs_pin + voff_m,
                                                     sb_dsa_m, hipMemcpyHostToDevice, w.stream), "dsa gar rem h2d");
                            uint16_t* d_cb = w.d_tmp_c + voff_m;
                            if (dsa_stay_f16) launch_dsa_vha_lowrank_lrga_f16_all(d_cb, w.d_tmp_a + voff_m, D.d_vha_q_loc, w.d_tmp_d + voff_m, d_cb, mb_len[bm], w.stream);
                            else              launch_dsa_vha_lowrank_lrga_f16_io(d_cb, w.d_tmp_a + voff_m, D.d_vha_q_loc, w.d_tmp_d + voff_m, d_cb, mb_len[bm], w.stream);
                            f16_linear_f16(w, d_cb, D.d_o_proj, w.d_sub_out + ooff_m, mb_len[bm], kHiddenSize, h_loc * 256);
                            hip_check(hipEventRecord(w.ev_mb_d2c[bm], w.stream), "dsa gar rem o comp rec");
                            if (r != own_r) {
                                hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[bm], 0), "dsa gar op wait");
                                hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + ooff_m, w.d_sub_out + ooff_m,
                                                         sb_op_m, hipMemcpyDeviceToHost, w.stream_sh), "dsa gar op d2h");
                                hip_check(hipEventRecord(w.ev_mb_o_d2h[bm], w.stream_sh), "dsa gar op rec");
                            }
                        }
                    };

                    for (int b = 0; b < kNb; ++b) {
                        const size_t voff   = size_t(mb_off[b]) * dsa_cols;
                        const size_t ooff   = size_t(mb_off[b]) * kHiddenSize;
                        const size_t mb_dsa = size_t(mb_len[b]) * dsa_cols;
                        const size_t sb_dsa = mb_dsa * sizeof(uint16_t);
                        const size_t slot_base = vha_red_base[b & 1];

                        if (b >= 2) {
                            hip_check(hipSetDevice(g0.hip_dev), "dsa gar g0 slot wait dev");
                            hip_check(hipStreamWaitEvent(g0.stream_sh, g0.ev_mb_c2d[b - 2], 0), "dsa gar g0 slot wait");
                        }
                        for (int ri = 0; ri < 3; ++ri) {
                            const int r = kRemSync[ri];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa gar sync d2h dev");
                            hip_check(hipEventSynchronize(w.ev_mb_vha_d2h[b]), "dsa gar sync vha");
                            hip_check(hipSetDevice(g0.hip_dev), "dsa gar g0 h2d dev");
                            uint16_t* h_in_r = h_tp_oattn_pin + size_t(r) * size_t(M) * dsa_cols + voff;
                            hip_check(hipMemcpyAsync(g0.d_tmp_a + slot_base + size_t(r - 1) * mb_dsa_max, h_in_r,
                                                     sb_dsa, hipMemcpyHostToDevice, g0.stream_sh), "dsa gar g0 h2d");
                        }
                        hip_check(hipSetDevice(g0.hip_dev), "dsa gar g0 red dev");
                        hip_check(hipEventRecord(g0.ev_mb_vha_h2d[b], g0.stream_sh), "dsa gar g0 h2d rec");
                        hip_check(hipStreamWaitEvent(g0.stream, g0.ev_mb_vha_h2d[b], 0), "dsa gar g0 wait h2d");
                        launch_sum4_f16_canonical(
                            g0.d_tmp_a + voff,
                            g0.d_tmp_a + slot_base + 0u * mb_dsa_max,
                            g0.d_tmp_a + slot_base + 1u * mb_dsa_max,
                            g0.d_tmp_a + slot_base + 2u * mb_dsa_max,
                            g0.d_tmp_a + voff, mb_dsa, g0.stream);
                        hip_check(hipEventRecord(g0.ev_mb_c2d[b], g0.stream), "dsa gar g0 sum rec");
                        hip_check(hipMemcpyAsync(h_tp_qabs_pin + voff, g0.d_tmp_a + voff,
                                                 sb_dsa, hipMemcpyDeviceToHost, g0.stream), "dsa gar g0 sum d2h");
                        hip_check(hipEventRecord(g0.ev_mb_vha_d2h[b], g0.stream), "dsa gar g0 sum d2h rec");
                        {
                            const auto& D0 = L.dsa_tp[0];
                            uint16_t* d_cb = g0.d_tmp_c + voff;
                            if (dsa_stay_f16) launch_dsa_vha_lowrank_lrga_f16_all(d_cb, g0.d_tmp_a + voff, D0.d_vha_q_loc, g0.d_tmp_d + voff, d_cb, mb_len[b], g0.stream);
                            else              launch_dsa_vha_lowrank_lrga_f16_io(d_cb, g0.d_tmp_a + voff, D0.d_vha_q_loc, g0.d_tmp_d + voff, d_cb, mb_len[b], g0.stream);
                            f16_linear_f16(g0, d_cb, D0.d_o_proj, g0.d_sub_out + ooff, mb_len[b], kHiddenSize, h_loc * 256);
                            hip_check(hipEventRecord(g0.ev_mb_d2c[b], g0.stream), "dsa gar g0 o comp rec");
                        }
                        if (b >= 1) launch_rem_mb(b - 1);
                    }
                    if (own_r != 0) {
                        hip_check(hipSetDevice(g0.hip_dev), "dsa gar g0 op d2h dev");
                        for (int bp = 0; bp < kNb; ++bp) {
                            const size_t ooff_p = size_t(mb_off[bp]) * kHiddenSize;
                            const size_t sb_op  = size_t(mb_len[bp]) * kHiddenSize * sizeof(uint16_t);
                            hip_check(hipStreamWaitEvent(g0.stream_sh, g0.ev_mb_d2c[bp], 0), "dsa gar g0 op wait");
                            hip_check(hipMemcpyAsync(h_tp_ar_in_pin[0] + ooff_p, g0.d_sub_out + ooff_p,
                                                     sb_op, hipMemcpyDeviceToHost, g0.stream_sh), "dsa gar g0 op d2h");
                            hip_check(hipEventRecord(g0.ev_mb_o_d2h[bp], g0.stream_sh), "dsa gar g0 op rec");
                        }
                    }
                    launch_rem_mb(kNb - 1);

                    constexpr int kOrdFastSync[4] = {0, 1, 3, 2};
                    int non_own[3];
                    int ni = 0;
                    for (int oi = 0; oi < 4; ++oi) {
                        const int r = kOrdFastSync[oi];
                        if (r != own_r) non_own[ni++] = r;
                    }
                    for (int b = 0; b < kNb; ++b) {
                        const size_t ooff  = size_t(mb_off[b]) * kHiddenSize;
                        const size_t mb_op = size_t(mb_len[b]) * kHiddenSize;
                        const size_t sb_op = mb_op * sizeof(uint16_t);
                        const size_t op_slot_base = op_red_base + size_t(b) * 3u * mb_op_max;
                        const uint16_t* p_rank[4] = {nullptr, nullptr, nullptr, nullptr};
                        p_rank[own_r] = g_own.d_sub_out + ooff;
                        for (int i = 0; i < 3; ++i) {
                            const int r = non_own[i];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa gar op sync dev");
                            hip_check(hipEventSynchronize(w.ev_mb_o_d2h[b]), "dsa gar op sync");
                            hip_check(hipSetDevice(g_own.hip_dev), "dsa gar op h2d dev");
                            uint16_t* dst_slot = g_own.d_tmp_a + op_slot_base + size_t(i) * mb_op_max;
                            hip_check(hipMemcpyAsync(dst_slot, h_tp_ar_in_pin[r] + ooff,
                                                     sb_op, hipMemcpyHostToDevice, g_own.stream_sh), "dsa gar op h2d");
                            p_rank[r] = dst_slot;
                        }
                        hip_check(hipSetDevice(g_own.hip_dev), "dsa gar op red dev");
                        hip_check(hipEventRecord(g_own.ev_mb_o_d2h[b], g_own.stream_sh), "dsa gar op h2d rec");
                        hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_o_d2h[b], 0), "dsa gar op sum wait");
                        launch_sum4_f16_to_bf16_canonical(p_rank[0], p_rank[1], p_rank[2], p_rank[3],
                                                          g_own.d_sub_out + ooff, mb_op, g_own.stream);
                    }
                    hip_check(hipSetDevice(g_own.hip_dev), "dsa gar op host dev");
                    hip_check(hipEventRecord(g_own.ev_mb_c2d[3], g_own.stream), "dsa gar op sum3 rec");
                    hip_check(hipStreamWaitEvent(g_own.stream_sh, g_own.ev_mb_c2d[3], 0), "dsa gar op host wait");
                    hip_check(hipMemcpyAsync(h_tp_o_out_pin, g_own.d_sub_out,
                                             size_t(M) * kHiddenSize * sizeof(uint16_t),
                                             hipMemcpyDeviceToHost, g_own.stream_sh), "dsa gar op host d2h");
                    hip_check(hipEventRecord(g_own.ev_mb_vha_d2h[3], g_own.stream_sh), "dsa gar op host done");

                    if (defer_ok) {
                        o_proj_host_d2h_pending = true;
                        if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = true;
                    } else {
                        hip_check(hipSetDevice(g_own.hip_dev), "dsa gar nondef sync dev");
                        hip_check(hipEventSynchronize(g_own.ev_mb_vha_d2h[3]), "dsa gar nondef sync");
                        const size_t full_op_b = size_t(M) * kHiddenSize * sizeof(uint16_t);
                        for (int i = 0; i < 3; ++i) {
                            const int r = non_own[i];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa gar nondef h2d dev");
                            hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_o_out_pin, full_op_b,
                                                     hipMemcpyHostToDevice, w.stream_sh), "dsa gar nondef h2d");
                            hip_check(hipEventRecord(w.ev_mb_d2c[3], w.stream_sh), "dsa gar nondef rec");
                            hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[3], 0), "dsa gar nondef wait");
                        }
                    }
                    } // !dual_ok
                } else {
                for (int oi = 0; oi < tp; ++oi) {
                    const int r = kOrdD2h[oi];
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& D = L.dsa_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "dsa 4mb init0 dev");
                    launch_vha_lowrank_proj_f16_in(w.d_tmp_c, D.d_vha_r_loc, w.d_tmp_a, mb_len[0], 256, w.stream);
                    hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "dsa 4mb c2d0");
                    hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "dsa 4mb wait0");
                    uint16_t* h_in0 = h_tp_oattn_pin + size_t(r) * size_t(M) * dsa_cols;
                    hip_check(hipMemcpyAsync(h_in0, w.d_tmp_a, size_t(mb_len[0]) * dsa_cols * sizeof(uint16_t),
                                             hipMemcpyDeviceToHost, w.stream_sh), "dsa 4mb vha0 d2h");
                    hip_check(hipEventRecord(w.ev_mb_vha_d2h[0], w.stream_sh), "dsa 4mb vha0 rec");
                }
                for (int oi = 0; oi < tp; ++oi) {
                    const int r = kOrdD2h[oi];
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& D = L.dsa_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "dsa 4mb init1 dev");
                    for (int b = 1; b < kNb; ++b) {
                        const size_t voff = size_t(mb_off[b]) * dsa_cols;
                        launch_vha_lowrank_proj_f16_in(w.d_tmp_c + voff, D.d_vha_r_loc, w.d_tmp_a + voff,
                                                       mb_len[b], 256, w.stream);
                        hip_check(hipEventRecord(w.ev_mb_c2d[b], w.stream), "dsa 4mb c2d b");
                    }
                    if (dsa_stay_f16) f16_linear_f16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                    else              bf16_linear_bf16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                }

                for (int b = 0; b < kNb; ++b) {
                    const size_t voff = size_t(mb_off[b]) * dsa_cols;
                    const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdD2h[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "dsa 4mb sync d2h dev");
                        hip_check(hipEventSynchronize(w.ev_mb_vha_d2h[b]), "dsa 4mb sync vha");
                    }
                    if (b + 1 < kNb) {
                        const size_t voff_n = size_t(mb_off[b + 1]) * dsa_cols;
                        for (int oi = 0; oi < tp; ++oi) {
                            const int r = kOrdD2h[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa 4mb next d2h dev");
                            hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[b + 1], 0), "dsa 4mb wait next");
                            uint16_t* h_inn = h_tp_oattn_pin + size_t(r) * size_t(M) * dsa_cols + voff_n;
                            hip_check(hipMemcpyAsync(h_inn, w.d_tmp_a + voff_n,
                                                     size_t(mb_len[b + 1]) * dsa_cols * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, w.stream_sh), "dsa 4mb next vha d2h");
                            hip_check(hipEventRecord(w.ev_mb_vha_d2h[b + 1], w.stream_sh), "dsa 4mb next vha rec");
                        }
                    }
                    avx2_sum4_f16_canonical(
                        h_tp_oattn_pin + 0u * size_t(M) * dsa_cols + voff,
                        h_tp_oattn_pin + 1u * size_t(M) * dsa_cols + voff,
                        h_tp_oattn_pin + 2u * size_t(M) * dsa_cols + voff,
                        h_tp_oattn_pin + 3u * size_t(M) * dsa_cols + voff,
                        h_tp_qabs_pin + voff, size_t(mb_len[b]) * dsa_cols, kOmpThreads);

                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdH2d[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        const auto& D = L.dsa_tp[r];
                        hip_check(hipSetDevice(w.hip_dev), "dsa 4mb h2d+comp dev");
                        hip_check(hipMemcpyAsync(w.d_tmp_a + voff, h_tp_qabs_pin + voff,
                                                 size_t(mb_len[b]) * dsa_cols * sizeof(uint16_t),
                                                 hipMemcpyHostToDevice, w.stream_sh), "dsa 4mb vha h2d");
                        hip_check(hipEventRecord(w.ev_mb_vha_h2d[b], w.stream_sh), "dsa 4mb vha h2d rec");
                        hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_vha_h2d[b], 0), "dsa 4mb comp wait");
                        uint16_t* d_cb = w.d_tmp_c + voff;
                        if (dsa_stay_f16) launch_dsa_vha_lowrank_lrga_f16_all(d_cb, w.d_tmp_a + voff, D.d_vha_q_loc, w.d_tmp_d + voff, d_cb, mb_len[b], w.stream);
                        else              launch_dsa_vha_lowrank_lrga_f16_io(d_cb, w.d_tmp_a + voff, D.d_vha_q_loc, w.d_tmp_d + voff, d_cb, mb_len[b], w.stream);
                        f16_linear_f16(w, d_cb, D.d_o_proj, w.d_sub_out + ooff, mb_len[b], kHiddenSize, h_loc * 256);
                        hip_check(hipEventRecord(w.ev_mb_d2c[b], w.stream), "dsa 4mb o comp rec");
                    }
                    if (b >= 1) {
                        const int bp = b - 1;
                        const size_t ooff_p = size_t(mb_off[bp]) * kHiddenSize;
                        for (int oi = 0; oi < tp; ++oi) {
                            const int r = kOrdD2h[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa 4mb op d2h dev");
                            hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[bp], 0), "dsa 4mb op wait");
                            hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + ooff_p, w.d_sub_out + ooff_p,
                                                     size_t(mb_len[bp]) * kHiddenSize * sizeof(uint16_t),
                                                     hipMemcpyDeviceToHost, w.stream_sh), "dsa 4mb op d2h");
                            hip_check(hipEventRecord(w.ev_mb_o_d2h[bp], w.stream_sh), "dsa 4mb op rec");
                        }
                    }
                }
                for (int bp = kNb - 2; bp < kNb; ++bp) {
                    const size_t ooff_p = size_t(mb_off[bp]) * kHiddenSize;
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdD2h[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "dsa 4mb tail d2h dev");
                        hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_d2c[bp], 0), "dsa 4mb tail wait");
                        hip_check(hipMemcpyAsync(h_tp_ar_in_pin[r] + ooff_p, w.d_sub_out + ooff_p,
                                                 size_t(mb_len[bp]) * kHiddenSize * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, w.stream_sh), "dsa 4mb tail d2h");
                        hip_check(hipEventRecord(w.ev_mb_o_d2h[bp], w.stream_sh), "dsa 4mb tail rec");
                    }
                }
                for (int b = 0; b < kNb; ++b) {
                    const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                    for (int oi = 0; oi < tp; ++oi) {
                        const int r = kOrdD2h[oi];
                        auto& w = *dev_map[dev_for_r(r)];
                        hip_check(hipSetDevice(w.hip_dev), "dsa 4mb o sync dev");
                        hip_check(hipEventSynchronize(w.ev_mb_o_d2h[b]), "dsa 4mb o sync");
                    }
                    avx2_sum4_f16_to_bf16_canonical(
                        h_tp_ar_in_pin[0] + ooff, h_tp_ar_in_pin[1] + ooff,
                        h_tp_ar_in_pin[2] + ooff, h_tp_ar_in_pin[3] + ooff,
                        h_tp_o_out_pin + ooff, size_t(mb_len[b]) * kHiddenSize, kOmpThreads);
                    if (defer_ok) {
                        auto& g_own = *dev_map[L.owner_dev];
                        hip_check(hipSetDevice(g_own.hip_dev), "dsa 4mb o own h2d dev");
                        hip_check(hipMemcpyAsync(g_own.d_sub_out + ooff, h_tp_o_out_pin + ooff,
                                                 size_t(mb_len[b]) * kHiddenSize * sizeof(uint16_t),
                                                 hipMemcpyHostToDevice, g_own.stream_sh), "dsa 4mb o own h2d");
                        if (b == kNb - 1) {
                            hip_check(hipEventRecord(g_own.ev_mb_d2c[3], g_own.stream_sh), "dsa 4mb o own done");
                            hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_d2c[3], 0), "dsa 4mb o own wait");
                            if (nonowner_oproj_deferred_out) *nonowner_oproj_deferred_out = true;
                        }
                    } else {
                        for (int oi = 0; oi < tp; ++oi) {
                            const int r = kOrdH2d[oi];
                            auto& w = *dev_map[dev_for_r(r)];
                            hip_check(hipSetDevice(w.hip_dev), "dsa 4mb o all h2d dev");
                            hip_check(hipMemcpyAsync(w.d_sub_out + ooff, h_tp_o_out_pin + ooff,
                                                     size_t(mb_len[b]) * kHiddenSize * sizeof(uint16_t),
                                                     hipMemcpyHostToDevice, w.stream_sh), "dsa 4mb o all h2d");
                            if (b == kNb - 1) {
                                hip_check(hipEventRecord(w.ev_mb_d2c[3], w.stream_sh), "dsa 4mb o all done");
                                hip_check(hipStreamWaitEvent(w.stream, w.ev_mb_d2c[3], 0), "dsa 4mb o all wait");
                            }
                        }
                    }
                }
                } // end dsa use_gpu_ar else
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp2).count();
                }
            } else if (use_dsa_vha_lr) {
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& D = L.dsa_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "dsa lr proj dev");
                    launch_vha_lowrank_proj_f16_in(w.d_tmp_c, D.d_vha_r_loc, w.d_tmp_a, M, 256, w.stream);
                }
                tp_allreduce_f16(M, 16 * 256, &Mi50ExpertWorker::d_tmp_a, &Mi50ExpertWorker::d_tmp_a);
                const auto t_dp2 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                for (int r = 0; r < tp; ++r) {
                    auto& w = *dev_map[dev_for_r(r)];
                    const auto& D = L.dsa_tp[r];
                    hip_check(hipSetDevice(w.hip_dev), "dsa E2 f16 dev");
                    if (dsa_stay_f16) {
                        f16_linear_f16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                        launch_dsa_vha_lowrank_lrga_f16_all(w.d_tmp_c, w.d_tmp_a, D.d_vha_q_loc, w.d_tmp_d, w.d_tmp_c, M, w.stream);
                    } else {
                        bf16_linear_bf16(w, w.d_qr, D.d_gate_proj, w.d_tmp_d, M, h_loc * 256, 1024);
                        launch_dsa_vha_lowrank_lrga_f16_io(w.d_tmp_c, w.d_tmp_a, D.d_vha_q_loc, w.d_tmp_d, w.d_tmp_c, M, w.stream);
                    }
                    f16_linear_f16(w, w.d_tmp_c, D.d_o_proj, w.d_sub_out, M, kHiddenSize, h_loc * 256);
                }
                if (profiling_enabled) {
                    prof_sync_tp(tp, L.is_echo, L.owner_dev);
                    prof.dsa_proj_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dp2).count();
                }
                tp_allreduce_f16_to_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            } else {
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

    // ---- Shared Routed-MoE All-Device Dispatch Helper (P150A FIRST + Strided 5x MI50 + On-GPU Async Merge!) ----
    const float* dispatch_routed_moe_all_devices(int l, bool is_echo, Mi50ExpertWorker& g_own, int M,
                                                 const float* fused_rmsnorm_ln = nullptr, float fused_rmsnorm_eps = 0.0f,
                                                 bool p150_prearmed = false, bool pre_p150_has_work = false,
                                                 uint32_t pre_p150_seq = 0u, const uint16_t* pre_p150_y_bf16 = nullptr,
                                                 const uint16_t* fused_owner_lat_norm = nullptr, float fused_owner_eps = 0.0f,
                                                 int defer_nonowner_tp = 0, int defer_owner_rank = -1,
                                                 hipEvent_t ev_xrot_host_ready = nullptr) {
        const bool all_to_host = (fused_rmsnorm_ln != nullptr);
        const bool track_chunk = (M >= 256);
        const auto t_m0 = (profiling_enabled || track_chunk) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        bool p150_has_work = p150_prearmed ? pre_p150_has_work : false;
        uint32_t p150_seq = p150_prearmed ? pre_p150_seq : 0u;
        p150_active_rows = M;

        // 1. Fast Path for Prefill (M > 8 && !is_echo): Commit 2 On-Chip LDS SRAM M=64 + Direct Weighted FP32 atomicAdd!
        //    - Zero host MoE bucketing/H2D/sync, zero d_y_slots_f32 (14.4 GiB less HBM2/2048 chunk!), M up to 2048!
        const int n_gpus = int(gpus.size());
        auto* ch = (!is_echo && p150_ok) ? p150_shm.ptr : nullptr;
        const int l1_end = ch ? (p150_exp_begin + 1) : 0;

        Mi50ExpertWorker* active_mi50s[8];
        int n_active_mi50 = 0;
        bool owner_did_moe = false;
        const float* p150_y_ptr = nullptr;
        const uint16_t* p150_y_bf16_ptr = p150_prearmed ? pre_p150_y_bf16 : nullptr;

        if (M > 8 && !is_echo) {
            const int max_pairs = M * kTopKExperts;
            constexpr int kMaxTiles = 4096;
            const bool sliced_p150 = (!p150_prearmed && ch != nullptr && M > int(ERNIE55_P150_MAX_ROWS));
            // Commit 4: prefill-only symmetric shadow-expert offload across all 13 Wave-8 shadow experts
            // ([319..332): 6 on GPU[2], 3 on GPU[0], 3 on GPU[1], 1 on GPU[3]).
            // Exact 1:1 partition: when offload_shadow is true, every MI50 caps max_loc to bb_base_count
            // while P150A executes all 13 shadow experts on its GDDR6 slack (0 B extra GPU VRAM,
            // decode M<=8 stays 100% MI50-resident).
            const bool offload_shadow = (gfx906_offload_shadow_enabled() && M >= 256 && ch != nullptr && l < mcfg.num_hidden_layers);

            // Commit 4b / Wave-10 C1 / Wave-11 C5 / Turn-5 Cut A: Single-pass async P150A with
            // 24-thread AVX2 BF16 conversion + ACTIVE-ROW x_rot COMPACTION!
            // Only rows with >= 1 P150-routed expert are transferred and executed (typically ~38%
            // of M at 194/512 P150 experts), so the 8.39 MiB x_rot H2D and the Tensix row loop both
            // shrink ~2.6x. The worker is UNCHANGED (it already honours ch->rows); the compact
            // y_out rows are scattered back to full M rows on the host before the merge, so the
            // canonical FP32 accumulation order and every output bit are preserved.
            if (!p150_prearmed && ch && !sliced_p150) {
                p150_row_map.clear();
                int n_active = 0;
                for (int r = 0; r < M; ++r) {
                    int row_routes = 0;
                    for (int s = 0; s < kTopKExperts; ++s) {
                        const int32_t eid = h_topk_ids_pin[size_t(r) * kTopKExperts + size_t(s)];
                        const float wt    = h_topk_wts_pin[size_t(r) * kTopKExperts + size_t(s)];
                        const int cand_g  = (wt != 0.0f && eid >= 0 && eid < kNumRoutedExperts) ? int(bb_owner_gpu[size_t(eid)]) : -1;
                        const int cand_l  = (cand_g >= 0) ? int(bb_local_slot[size_t(eid)]) : -1;
                        const bool shadow_off = offload_shadow && is_prefill_shadow_offload_gpu(cand_g) && (cand_l >= gpus[size_t(cand_g)]->bb_base_count);
                        const bool mi50_owns = !shadow_off &&
                            (cand_g >= 0) && (l < mcfg.num_hidden_layers || cand_l < gpus[size_t(cand_g)]->bb_base_count);
                        if (!mi50_owns && eid >= p150_exp_begin && eid < p150_exp_end && wt != 0.0f) ++row_routes;
                    }
                    if (row_routes > 0) {
                        // Compact row n_active carries the routes of original row r.
                        for (int s = 0; s < kTopKExperts; ++s) {
                            const int32_t eid = h_topk_ids_pin[size_t(r) * kTopKExperts + size_t(s)];
                            const float wt    = h_topk_wts_pin[size_t(r) * kTopKExperts + size_t(s)];
                            const int cand_g  = (wt != 0.0f && eid >= 0 && eid < kNumRoutedExperts) ? int(bb_owner_gpu[size_t(eid)]) : -1;
                            const int cand_l  = (cand_g >= 0) ? int(bb_local_slot[size_t(eid)]) : -1;
                            const bool shadow_off = offload_shadow && is_prefill_shadow_offload_gpu(cand_g) && (cand_l >= gpus[size_t(cand_g)]->bb_base_count);
                            const bool mi50_owns = !shadow_off &&
                                (cand_g >= 0) && (l < mcfg.num_hidden_layers || cand_l < gpus[size_t(cand_g)]->bb_base_count);
                            if (!mi50_owns && eid >= p150_exp_begin && eid < p150_exp_end && wt != 0.0f) {
                                ch->row_slots[n_active][s]   = eid - p150_exp_begin;
                                ch->row_weights[n_active][s] = wt;
                                p150_has_work = true;
                                ++prof.p150_routes;
                                if (eid < l1_end) ++prof.p150_l1_routes;
                            } else {
                                ch->row_slots[n_active][s]   = -1;
                                ch->row_weights[n_active][s] = 0.0f;
                                if (mi50_owns) ++prof.mi50_routes;
                            }
                        }
                        p150_row_map.push_back(int32_t(r));
                        ++n_active;
                    } else {
                        for (int s = 0; s < kTopKExperts; ++s) {
                            ch->row_slots[r][s]   = -1;
                            ch->row_weights[r][s] = 0.0f;
                        }
                    }
                }
                p150_active_rows = n_active;
                if (p150_has_work) {
                    const bool use_native_half = (M > 32);
                    if (use_native_half) {
                        // Gather only the active rows and convert FP16 -> BF16 into ch->x_rot_f16!
                        #pragma omp parallel for num_threads(8) schedule(static) if (n_active >= 64)
                        for (int i = 0; i < n_active; ++i) {
                            const uint16_t* src = h_lat_rot_f16_pin + size_t(p150_row_map[size_t(i)]) * kLatentDim;
                            avx2_f16_to_bf16(src, ch->x_rot_f16 + size_t(i) * kLatentDim, kLatentDim, 1);
                        }
                        ch->flags = ERNIE55_P150_FLAG_X_BF16 | ERNIE55_P150_FLAG_Y_BF16;
                        p150_y_bf16_ptr = ch->y_out_bf16;
                    } else {
                        for (int i = 0; i < n_active; ++i) {
                            const uint16_t* src = h_lat_rot_f16_pin + size_t(p150_row_map[size_t(i)]) * kLatentDim;
                            avx2_f16_to_f32(src, ch->x_rot_f32 + size_t(i) * kLatentDim, size_t(kLatentDim), 1);
                        }
                        ch->flags = 0u;
                        p150_y_ptr = ch->y_out_f32;
                    }
                    ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                    ch->layer_idx = uint32_t(l);
                    ch->rows      = uint32_t(n_active);
                    ch->top_k     = kTopKExperts;
                    p150_seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                    ch->req_seq.store(p150_seq, std::memory_order_release);
                }
            }

            // Wave-11 C5: MI50 workers always consume native FP16 from WB-pinned h_lat_rot_f16_pin,
            // while P150A consumes 24-thread AVX2-converted BF16 from ch->x_rot_f16 (0 worker CPU stall!).
            const uint16_t* x_rot_host_src = h_lat_rot_f16_pin;
            bool nonown_has_xrot_ev[LayerSpineGpu::kMaxTp] = {};
            bool nonown_has_moe_ev[LayerSpineGpu::kMaxTp]  = {};
            auto launch_one_mi50 = [&](int gi, Mi50ExpertWorker& w) {
                const bool gpu_offload = offload_shadow && is_prefill_shadow_offload_gpu(gi);
                const int max_loc = (l < mcfg.num_hidden_layers && !gpu_offload) ? w.bb_count() : w.bb_base_count;
                if (max_loc <= 0) return;
                const bool is_owner = (!all_to_host && w.hip_dev == g_own.hip_dev);
                hip_check(hipSetDevice(w.hip_dev), "set worker c2 dev");
                const int32_t* topk_ids_dev = w.d_moe_topk_ids;
                const float*   topk_wts_dev = w.d_moe_topk_wts;
                if (is_owner && p150_prearmed) {
                    topk_ids_dev = w.d_topk_ids;
                    topk_wts_dev = w.d_topk_wts;
                } else {
                    if (all_to_host || w.hip_dev != g_own.hip_dev) {
                        hip_check(hipMemcpyAsync(w.d_x_rot_f16, x_rot_host_src,
                                                 size_t(M) * kLatentDim * sizeof(uint16_t), hipMemcpyHostToDevice, w.stream), "H2D worker x_rot c2");
                    }
                    hip_check(hipMemcpyAsync(w.d_moe_topk_ids, h_topk_ids_pin,
                                             size_t(max_pairs) * sizeof(int32_t), hipMemcpyHostToDevice, w.stream), "H2D topk_ids c2");
                    hip_check(hipMemcpyAsync(w.d_moe_topk_wts, h_topk_wts_pin,
                                             size_t(max_pairs) * sizeof(float), hipMemcpyHostToDevice, w.stream), "H2D topk_wts c2");
                    if (defer_nonowner_tp > 1 && w.ev_mb_c2d[0] != nullptr) {
                        for (int rr = 0; rr < defer_nonowner_tp; ++rr) {
                            if (rr != defer_owner_rank && tp_devs[rr] == w.hip_dev) {
                                hip_check(hipEventRecord(w.ev_mb_c2d[0], w.stream), "defer nonown xrot ev");
                                nonown_has_xrot_ev[rr] = true;
                                break;
                            }
                        }
                    }
                }
                hip_check(hipMemsetAsync(w.d_y_acc_f32, 0, size_t(M) * kLatentDim * sizeof(float), w.stream), "zero y_acc c2");

                // Single persistent dynamic-predicated m64 grid (has_m1..has_m7 8-row skip inside
                // m64 beats multi-bucket m64/m32/m8 on gfx906 silicon: 10.10 ms vs 10.47 ms!).
                launch_ernie55_build_expert_csr_prefill(
                    topk_ids_dev, topk_wts_dev, w.d_g2l_bb,
                    M, max_loc, 64, kMaxTiles, max_pairs,
                    w.d_csr_counts, w.d_moe_tiles, w.d_moe_pairs, w.stream, /*hybrid_buckets=*/false);
                uint16_t* z_f16 = (w.d_tmp_a != nullptr) ? w.d_tmp_a : reinterpret_cast<uint16_t*>(w.d_z_f32);
                launch_hlut4_prefill_situ_f16_m64(
                    w.d_x_rot_f16, w.d_csr_counts, w.d_moe_tiles, w.d_moe_pairs,
                    w.d_w1[size_t(l)], w.d_w3[size_t(l)], w.d_levels8_f16, z_f16,
                    kMoeInterSize, kLatentDim, mcfg.situ_beta, mcfg.situ_linear_beta, kMaxTiles, w.stream);
                launch_wht128_f16_inplace_prefill_pairs(
                    z_f16, w.d_csr_counts, w.d_signs_inter[size_t(l)], kMoeInterSize, max_pairs, w.stream);
                launch_hlut4_prefill_down_atomic_m64(
                    z_f16, w.d_csr_counts, w.d_moe_tiles, w.d_moe_pairs,
                    w.d_w2[size_t(l)], w.d_levels8_f16, w.d_y_acc_f32,
                    kLatentDim, kMoeInterSize, kMaxTiles, w.stream);
                // Wave-11 C3: On the straggler MI50 (GPU[2]), defer mhc_post_update until after down_atomic_m64
                // so GPU[2] keeps 100% HBM2/CU bandwidth for critical-path mi50_moe!
                if (!is_owner && defer_nonowner_tp > 1 && is_prefill_shadow_offload_gpu(gi) && w.ev_mb_c2d[1] != nullptr) {
                    for (int rr = 0; rr < defer_nonowner_tp; ++rr) {
                        if (rr != defer_owner_rank && tp_devs[rr] == w.hip_dev) {
                            hip_check(hipEventRecord(w.ev_mb_c2d[1], w.stream), "defer nonown moe ev");
                            nonown_has_moe_ev[rr] = true;
                            break;
                        }
                    }
                }

                if (is_owner) {
                    owner_did_moe = true;
                } else {
                    if (M > 32 && !all_to_host && fused_owner_lat_norm != nullptr) {
                        // Wave-10 C1: Half-bandwidth FP16 remote partial D2H (8 MiB vs 16 MiB) into dead d_x_rot_f16!
                        launch_f32_to_f16(w.d_y_acc_f32, w.d_x_rot_f16, M * kLatentDim, w.stream);
                        hip_check(hipMemcpyAsync(w.h_y_partial_pin, w.d_x_rot_f16, size_t(M) * kLatentDim * sizeof(uint16_t),
                                                 hipMemcpyDeviceToHost, w.stream), "D2H worker y f16 c1");
                    } else {
                        hip_check(hipMemcpyAsync(w.h_y_partial_pin, w.d_y_acc_f32, size_t(M) * kLatentDim * sizeof(float),
                                                 hipMemcpyDeviceToHost, w.stream), "D2H worker y c2");
                    }
                    active_mi50s[n_active_mi50++] = &w;
                }
            };

            if (p150_prearmed) {
                // Launch owner FIRST (reads resident g_own.d_x_rot_f16 -> overlaps host h_lat_rot_f16_pin D2H!),
                // then sync h_lat_rot_f16_pin before launching straggler GPU[2] and remaining MI50s!
                for (int gi = 0; gi < n_gpus; ++gi) {
                    if (gpus[size_t(gi)]->hip_dev == g_own.hip_dev) launch_one_mi50(gi, *gpus[size_t(gi)]);
                }
                if (ev_xrot_host_ready != nullptr) {
                    hip_check(hipSetDevice(g_own.hip_dev), "sync xrot host dev");
                    hip_check(hipEventSynchronize(ev_xrot_host_ready), "sync xrot host ready");
                }
                for (int gi = 0; gi < n_gpus; ++gi) {
                    if (gpus[size_t(gi)]->hip_dev != g_own.hip_dev && is_prefill_shadow_offload_gpu(gi)) {
                        launch_one_mi50(gi, *gpus[size_t(gi)]);
                    }
                }
                for (int gi = 0; gi < n_gpus; ++gi) {
                    if (gpus[size_t(gi)]->hip_dev != g_own.hip_dev && !is_prefill_shadow_offload_gpu(gi)) {
                        launch_one_mi50(gi, *gpus[size_t(gi)]);
                    }
                }
            } else {
                for (int gi = 0; gi < n_gpus; ++gi) {
                    launch_one_mi50(gi, *gpus[size_t(gi)]);
                }
            }

            // Wave-11 C1/C3: Overlap the 3 non-owner o_proj H2Ds (16 MiB each) + attn post-mHC (d_res_b) +
            // 160 KiB owner mlp_mhc mixes asynchronously on w.stream_sh (straggler GPU[2] runs mhc_post_update
            // right after down_atomic_m64 so critical-path mi50_moe sees 0 HBM2 contention!).
            if (defer_nonowner_tp > 1 && defer_owner_rank >= 0) {
                if (o_proj_host_d2h_pending) {
                    hip_check(hipSetDevice(g_own.hip_dev), "defer own o_d2h dev");
                    hip_check(hipEventSynchronize(g_own.ev_mb_vha_d2h[3]), "defer own o_d2h sync");
                    o_proj_host_d2h_pending = false;
                }
                constexpr int kOrdDefer[4] = {2, 1, 3, 0};
                for (int oi = 0; oi < defer_nonowner_tp; ++oi) {
                    const int rr = kOrdDefer[oi];
                    if (rr == defer_owner_rank) continue;
                    auto& w = *dev_map[tp_devs[rr]];
                    hip_check(hipSetDevice(w.hip_dev), "defer nonown dev");
                    if (nonown_has_xrot_ev[rr]) {
                        hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[0], 0), "defer nonown wait xrot");
                    }
                    hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_o_out_pin,
                                             size_t(M) * kHiddenSize * sizeof(uint16_t),
                                             hipMemcpyHostToDevice, w.stream_sh), "defer nonown o_proj h2d");
                    if (nonown_has_moe_ev[rr]) {
                        hip_check(hipStreamWaitEvent(w.stream_sh, w.ev_mb_c2d[1], 0), "defer nonown wait moe");
                    }
                    launch_mhc_post_update(w.d_sub_out, w.d_res_a, w.d_post_mix, w.d_comb_mix,
                                           w.d_res_b, M, kHiddenSize, w.stream_sh);
                    hip_check(hipMemcpyAsync(w.d_post_mix, h_mhc_mix_pin,
                                             size_t(M) * 4u * sizeof(float),
                                             hipMemcpyHostToDevice, w.stream_sh), "defer nonown post_mix h2d");
                    hip_check(hipMemcpyAsync(w.d_comb_mix, h_mhc_mix_pin + size_t(M) * 4u,
                                             size_t(M) * 16u * sizeof(float),
                                             hipMemcpyHostToDevice, w.stream_sh), "defer nonown comb_mix h2d");
                    hip_check(hipEventRecord(w.ev_res_b_ready, w.stream_sh), "defer nonown res_b rec");
                }
            }

            // If M > ERNIE55_P150_MAX_ROWS (e.g. M=2048), run Blackhole P150A in <=1024-row slices CONCURRENTLY with MI50!
            if (sliced_p150) {
                const int max_r = int(ERNIE55_P150_MAX_ROWS);
                for (int r0 = 0; r0 < M; r0 += max_r) {
                    const int cur = std::min(max_r, M - r0);
                    bool slice_has_work = false;
                    for (int r = 0; r < cur; ++r) {
                        const int gr = r0 + r;
                        for (int s = 0; s < kTopKExperts; ++s) {
                            const int32_t eid = h_topk_ids_pin[size_t(gr) * kTopKExperts + size_t(s)];
                            const float wt    = h_topk_wts_pin[size_t(gr) * kTopKExperts + size_t(s)];
                            const int cand_g  = (wt != 0.0f && eid >= 0 && eid < kNumRoutedExperts) ? int(bb_owner_gpu[size_t(eid)]) : -1;
                            const int cand_l  = (cand_g >= 0) ? int(bb_local_slot[size_t(eid)]) : -1;
                            const bool shadow_off = offload_shadow && is_prefill_shadow_offload_gpu(cand_g) && (cand_l >= gpus[size_t(cand_g)]->bb_base_count);
                            const bool mi50_owns = !shadow_off &&
                                (cand_g >= 0) && (l < mcfg.num_hidden_layers || cand_l < gpus[size_t(cand_g)]->bb_base_count);
                            if (!mi50_owns && eid >= p150_exp_begin && eid < p150_exp_end && wt != 0.0f) {
                                ch->row_slots[r][s]   = eid - p150_exp_begin;
                                ch->row_weights[r][s] = wt;
                                slice_has_work = true;
                                ++prof.p150_routes;
                                if (eid < l1_end) ++prof.p150_l1_routes;
                            } else {
                                ch->row_slots[r][s]   = -1;
                                ch->row_weights[r][s] = 0.0f;
                                if (mi50_owns) ++prof.mi50_routes;
                            }
                        }
                    }
                    if (slice_has_work) {
                        avx2_f16_to_f32(h_lat_rot_f16_pin + size_t(r0) * kLatentDim, ch->x_rot_f32, size_t(cur) * kLatentDim, kOmpThreads);
                        ch->flags     = 0u;
                        ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                        ch->layer_idx = uint32_t(l);
                        ch->rows      = uint32_t(cur);
                        ch->top_k     = kTopKExperts;
                        const uint32_t seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                        ch->req_seq.store(seq, std::memory_order_release);
                        while (ch->resp_seq.load(std::memory_order_acquire) != seq) p150_cpu_relax();
                        if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
                        std::memcpy(h_p150_y_pin + size_t(r0) * kLatentDim, ch->y_out_f32, size_t(cur) * kLatentDim * sizeof(float));
                        p150_has_work = true;
                    } else {
                        std::memset(h_p150_y_pin + size_t(r0) * kLatentDim, 0, size_t(cur) * kLatentDim * sizeof(float));
                    }
                }
                if (p150_has_work) {
                    p150_y_ptr = h_p150_y_pin;
                    p150_seq = ch->resp_seq.load(std::memory_order_acquire);
                }
            }
        } else {
            // Canonical Decode Path (M <= 8 || is_echo): 100% untouched v1.0.0 deterministic slot-reduced path!
            int node_cnt[8] = {};
            for (int gi = 0; gi < n_gpus; ++gi) gpus[size_t(gi)]->active_exps.clear();

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

            if (p150_has_work && ch) {
                const size_t n_lat = size_t(M) * kLatentDim;
                avx2_f16_to_f32(h_lat_rot_f16_pin, ch->x_rot_f32, n_lat, kOmpThreads);
                ch->flags     = 0u;
                ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                ch->layer_idx = uint32_t(l);
                ch->rows      = uint32_t(M);
                ch->top_k     = kTopKExperts;
                p150_seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                ch->req_seq.store(p150_seq, std::memory_order_release);
                p150_y_ptr = ch->y_out_f32;
            }

            size_t max_grp_cnt = 0;
            for (int gi = 0; gi < n_gpus; ++gi) {
                auto& w = *gpus[size_t(gi)];
                w.h_groups.clear();
                if (w.active_exps.empty()) continue;

                hip_check(hipSetDevice(w.hip_dev), "set worker dev");
                if (all_to_host || w.hip_dev != g_own.hip_dev) {
                    hip_check(hipMemcpyAsync(w.d_x_rot_f16, h_lat_rot_f16_pin,
                                             size_t(M) * kLatentDim * sizeof(uint16_t), hipMemcpyHostToDevice, w.stream), "H2D worker x_rot");
                }
                hip_check(hipMemsetAsync(w.d_y_slots_f32, 0, size_t(M) * kTopKExperts * kLatentDim * sizeof(float), w.stream), "zero worker y_slots");

                for (int le : w.active_exps) {
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
                    w.b_head[size_t(le)] = -1;
                    w.b_cnt[size_t(le)]  = 0;
                }
                if (w.h_groups.size() > max_grp_cnt) max_grp_cnt = w.h_groups.size();
            }

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
                if (w.h_groups.empty()) continue;
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
            if (p150_has_work && ch && (p150_y_ptr || p150_y_bf16_ptr)) {
                const auto t_p0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                while (ch->resp_seq.load(std::memory_order_acquire) != p150_seq) p150_cpu_relax();
                if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
                if (profiling_enabled) {
                    prof.p150_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
                }
                if (p150_y_ptr) srcs[n_srcs++] = p150_y_ptr;
            }
            const int lat_loc = kLatentDim / tp_size;
            #pragma omp parallel for num_threads(kOmpThreads) schedule(static) if (M >= 8)
            for (int m = 0; m < M; ++m) {
                alignas(32) float y[kLatentDim];
                const size_t off = size_t(m) * kLatentDim;
                if (n_srcs == 0 && !p150_y_bf16_ptr) {
                    std::memset(y, 0, sizeof(y));
                } else if (n_srcs > 0) {
                    const float* s0 = srcs[0] + off;
                    for (int c = 0; c < kLatentDim; ++c) y[c] = s0[c];
                    for (int si = 1; si < n_srcs; ++si) {
                        const float* sk = srcs[si] + off;
                        for (int c = 0; c < kLatentDim; ++c) y[c] += sk[c];
                    }
                    if (p150_y_bf16_ptr) {
                        const uint16_t* sb = p150_y_bf16_ptr + off;
                        for (int c = 0; c < kLatentDim; ++c) y[c] += bf16_to_f32(sb[c]);
                    }
                } else {
                    const uint16_t* sb = p150_y_bf16_ptr + off;
                    for (int c = 0; c < kLatentDim; ++c) y[c] = bf16_to_f32(sb[c]);
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
        if (M > 32 && !is_echo && fused_owner_lat_norm != nullptr && g_own.stream_sh != nullptr) {
            // Wave-11 C5: Fastest-first remote MI50 FP16 + P150A BF16 partial H2Ds run on g_own.stream_sh
            // behind tuned shared_experts (finishes at ~4.8 ms, long before remote MI50s!) -> 100% overlapped
            // with owner MI50 MoE! Signaled to g_own.stream via ev_mb_d2c[0] (0 extra streams/events!).
            std::sort(active_mi50s, active_mi50s + n_active_mi50,
                      [](const Mi50ExpertWorker* a, const Mi50ExpertWorker* b) { return a->bb_count() < b->bb_count(); });
            bool used_sh_h2d = false;
            for (int wi = 0; wi < n_active_mi50; ++wi) {
                auto* w = active_mi50s[wi];
                hip_check(hipSetDevice(w->hip_dev), "sync worker dev");
                hip_check(hipStreamSynchronize(w->stream), "sync worker");
                if (track_chunk) {
                    const double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
                    for (int gi = 0; gi < n_gpus && gi < 5; ++gi) {
                        if (gpus[size_t(gi)].get() == w) { chunk_gpu_moe_ms[gi] += gms; break; }
                    }
                }
                hip_check(hipSetDevice(g_own.hip_dev), "h2d worker f16 dev");
                hip_check(hipMemcpyAsync(g_own.d_tmp_b + size_t(wi) * n_lat, w->h_y_partial_pin,
                                         n_lat * sizeof(uint16_t), hipMemcpyHostToDevice, g_own.stream_sh),
                          "H2D remote partial f16 on stream_sh");
                used_sh_h2d = true;
            }
            if (track_chunk) {
                chunk_mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
            }
            if (profiling_enabled) {
                hip_check(hipSetDevice(g_own.hip_dev), "prof owner moe dev");
                hip_check(hipStreamSynchronize(g_own.stream), "prof owner moe");
                prof.mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
            }
            const uint16_t* d_p150_bf16_slab = nullptr;
            // Turn-3 Cut B1: honest 42-layer owner accounting with ZERO extra syncs. The owner's routed
            // MoE completion is observed at the FIRST host-blocking point that already exists in this
            // layer (the P150A resp_seq spin, or the existing stream_sh sync below), so g0/g1 accumulate
            // all 42 layers instead of only the 21 layers where they are a non-owner remote worker.
            std::chrono::steady_clock::time_point t_owner_end{};
            if (p150_has_work && ch && p150_y_bf16_ptr) {
                const auto t_p0 = (profiling_enabled || track_chunk) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                while (ch->resp_seq.load(std::memory_order_acquire) != p150_seq) p150_cpu_relax();
                if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
                if (track_chunk) t_owner_end = std::chrono::steady_clock::now();
                if (profiling_enabled || track_chunk) {
                    const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
                    if (profiling_enabled) prof.p150_wait_ms += pms;
                    if (track_chunk) chunk_p150_wait_ms += pms;
                }
                hip_check(hipSetDevice(g_own.hip_dev), "merge p150 dev");
                uint16_t* dst_slab = g_own.d_tmp_b + size_t(n_active_mi50) * n_lat;
                const void* src_host = p150_y_bf16_ptr;
                if (p150_active_rows != M && int(p150_row_map.size()) == p150_active_rows) {
                    // Turn-5 Cut A: scatter compact P150A BF16 rows into PINNED h_p150_y_pin (0 unpinned DMA race!).
                    auto* dst_pin = reinterpret_cast<uint16_t*>(h_p150_y_pin);
                    std::memset(dst_pin, 0, n_lat * sizeof(uint16_t));
                    for (int i = 0; i < p150_active_rows; ++i) {
                        std::memcpy(dst_pin + size_t(p150_row_map[size_t(i)]) * kLatentDim,
                                    p150_y_bf16_ptr + size_t(i) * kLatentDim,
                                    size_t(kLatentDim) * sizeof(uint16_t));
                    }
                    src_host = dst_pin;
                } else if (!p150_shm_y_reg) {
                    std::memcpy(h_p150_y_pin, src_host, n_lat * sizeof(uint16_t));
                    src_host = h_p150_y_pin;
                }
                hip_check(hipMemcpyAsync(dst_slab, src_host, n_lat * sizeof(uint16_t),
                                         hipMemcpyHostToDevice, g_own.stream_sh), "H2D p150 bf16 on stream_sh");
                d_p150_bf16_slab = dst_slab;
                used_sh_h2d = true;
            }
            hip_check(hipSetDevice(g_own.hip_dev), "restore spine dev");
            if (used_sh_h2d && g_own.ev_mb_d2c[0] != nullptr) {
                hip_check(hipEventRecord(g_own.ev_mb_d2c[0], g_own.stream_sh), "rec moe partial h2d done");
                hip_check(hipStreamWaitEvent(g_own.stream, g_own.ev_mb_d2c[0], 0), "wait moe partial h2d");
                if (track_chunk && t_owner_end == std::chrono::steady_clock::time_point{}) t_owner_end = std::chrono::steady_clock::now();
            } else if (used_sh_h2d) {
                hip_check(hipStreamSynchronize(g_own.stream_sh), "sync moe partial h2d");
                if (track_chunk && t_owner_end == std::chrono::steady_clock::time_point{}) t_owner_end = std::chrono::steady_clock::now();
            }
            if (track_chunk && t_owner_end != std::chrono::steady_clock::time_point{}) {
                const double oms = std::chrono::duration<double, std::milli>(t_owner_end - t_m0).count();
                for (int gi = 0; gi < n_gpus && gi < 5; ++gi) {
                    if (gpus[size_t(gi)].get() == &g_own) { chunk_gpu_moe_ms[gi] += oms; break; }
                }
            }
            launch_merge_moe_f16_bf16_rmsnorm_to_f16(
                owner_did_moe ? g_own.d_y_acc_f32 : nullptr,
                g_own.d_tmp_b, n_active_mi50, d_p150_bf16_slab,
                fused_owner_lat_norm, g_own.d_h_lat, M, kLatentDim, fused_owner_eps, g_own.stream);
            return nullptr;
        }
        if (M > 32 && !is_echo) {
            // Wave-10 C1: Zero OpenMP CPU float reduction! Pipeline fastest-first MI50 H2D + on-GPU FP32 add
            // while slower MI50s and P150A finish, then merge P150A BF16 via async H2D + k_add_bf16_to_f32_inplace!
            std::sort(active_mi50s, active_mi50s + n_active_mi50,
                      [](const Mi50ExpertWorker* a, const Mi50ExpertWorker* b) { return a->bb_count() < b->bb_count(); });
            if (!owner_did_moe) {
                hip_check(hipSetDevice(g_own.hip_dev), "zero owner y_acc dev");
                hip_check(hipMemsetAsync(g_own.d_y_acc_f32, 0, n_lat * sizeof(float), g_own.stream), "zero owner y_acc");
            }
            for (int wi = 0; wi < n_active_mi50; ++wi) {
                auto* w = active_mi50s[wi];
                hip_check(hipSetDevice(w->hip_dev), "sync worker dev");
                hip_check(hipStreamSynchronize(w->stream), "sync worker");
                hip_check(hipSetDevice(g_own.hip_dev), "merge worker dev");
                hip_check(hipMemcpyAsync(g_own.d_f32_tmp0, w->h_y_partial_pin, n_lat * sizeof(float),
                                         hipMemcpyHostToDevice, g_own.stream), "H2D remote partial");
                launch_f32_add_inplace(g_own.d_y_acc_f32, g_own.d_f32_tmp0, int(n_lat), g_own.stream);
            }
            if (track_chunk) {
                chunk_mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
            }
            if (profiling_enabled) {
                hip_check(hipSetDevice(g_own.hip_dev), "prof owner moe dev");
                hip_check(hipStreamSynchronize(g_own.stream), "prof owner moe");
                prof.mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
            }
            if (p150_has_work && ch && (p150_y_ptr || p150_y_bf16_ptr)) {
                const auto t_p0 = (profiling_enabled || track_chunk) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                while (ch->resp_seq.load(std::memory_order_acquire) != p150_seq) p150_cpu_relax();
                if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
                if (profiling_enabled || track_chunk) {
                    const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
                    if (profiling_enabled) prof.p150_wait_ms += pms;
                    if (track_chunk) chunk_p150_wait_ms += pms;
                }
                hip_check(hipSetDevice(g_own.hip_dev), "merge p150 dev");
                if (p150_y_bf16_ptr) {
                    std::memcpy(h_p150_y_pin, p150_y_bf16_ptr, n_lat * sizeof(uint16_t));
                    hip_check(hipMemcpyAsync(g_own.d_h_lat, h_p150_y_pin, n_lat * sizeof(uint16_t),
                                             hipMemcpyHostToDevice, g_own.stream), "H2D p150 bf16");
                    launch_add_bf16_to_f32_inplace(g_own.d_y_acc_f32, g_own.d_h_lat, int(n_lat), g_own.stream);
                } else {
                    std::memcpy(h_p150_y_pin, p150_y_ptr, n_lat * sizeof(float));
                    hip_check(hipMemcpyAsync(g_own.d_f32_tmp0, h_p150_y_pin, n_lat * sizeof(float),
                                             hipMemcpyHostToDevice, g_own.stream), "H2D p150 f32");
                    launch_f32_add_inplace(g_own.d_y_acc_f32, g_own.d_f32_tmp0, int(n_lat), g_own.stream);
                }
            }
            hip_check(hipSetDevice(g_own.hip_dev), "restore spine dev");
            return g_own.d_y_acc_f32;
        }

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
        if (track_chunk) {
            chunk_mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
        }
        if (profiling_enabled) {
            if (owner_did_moe) {
                hip_check(hipSetDevice(g_own.hip_dev), "prof owner moe dev");
                hip_check(hipStreamSynchronize(g_own.stream), "prof owner moe");
            }
            prof.mi50_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
        }
        if (p150_has_work && ch && (p150_y_ptr || p150_y_bf16_ptr)) {
            const auto t_p0 = (profiling_enabled || track_chunk) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            while (ch->resp_seq.load(std::memory_order_acquire) != p150_seq) p150_cpu_relax();
            if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
            if (profiling_enabled || track_chunk) {
                const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
                if (profiling_enabled) prof.p150_wait_ms += pms;
                if (track_chunk) chunk_p150_wait_ms += pms;
            }
            if (p150_y_bf16_ptr) {
                if (!has_remote_sum) {
                    avx2_bf16_to_f32(p150_y_bf16_ptr, h_y_sum_f32_pin, n_lat, kOmpThreads);
                    has_remote_sum = true;
                } else {
                    avx2_add_bf16_to_f32_inplace(h_y_sum_f32_pin, p150_y_bf16_ptr, n_lat, kOmpThreads);
                }
            } else if (!has_remote_sum) {
                std::memcpy(h_y_sum_f32_pin, p150_y_ptr, n_lat * sizeof(float));
                has_remote_sum = true;
            } else {
                avx2_add_f32_inplace(h_y_sum_f32_pin, p150_y_ptr, n_lat, kOmpThreads);
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
    void run_moe_core(int l, const MoeSpineGpuWeights& moe, bool is_echo, Mi50ExpertWorker& g, int M,
                      bool draft_mi50_only = false, int bcast_tp = 1, int bcast_owner_rank = 0,
                      bool* bcast_done_out = nullptr, bool nonowner_oproj_deferred = false,
                      bool x_f16_ready = false) {
        if (bcast_done_out) *bcast_done_out = false;
        const auto t_r0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        hip_check(hipSetDevice(g.hip_dev), "moe single dev");

        auto* ch = (!is_echo && p150_ok && !draft_mi50_only) ? p150_shm.ptr : nullptr;
        const bool has_sh = (g.stream_sh != nullptr && g.blas_sh != nullptr);
        const bool fast_async_p150 = (M > 32 && !is_echo && has_sh && !draft_mi50_only &&
                                      (!ch || M <= int(ERNIE55_P150_MAX_ROWS)));
        const float* d_y_final = nullptr;

        if (fast_async_p150) {
            drain_p150_decode_relay();
            const auto t_mp0 = (M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const bool do_pre_xray = (M >= 256 && moe_pre_xray);
            std::chrono::steady_clock::time_point t_xr_bl{}, t_xr_rt{}, t_xr_pk{}, t_xr_fc{};
            if (do_pre_xray) {
                hip_check(hipStreamSynchronize(g.stream), "xray pre backlog sync");
                t_xr_bl = std::chrono::steady_clock::now();
                chunk_pre_backlog_ms += std::chrono::duration<double, std::milli>(t_xr_bl - t_mp0).count();
            }
            // Wave-11 C5+: Router on g.stream -> immediate 320 KiB topk+mhc_mix D2H on g.stream_sh ->
            // hide CPU row_slots + non-owner o_proj H2D/mhc_post_update behind 2.12 ms fc1_lat!
            // Dual-output WHT128 writes FP16 (d_x_rot_f16) + BF16 (d_tmp_a) -> direct pinned D2H into ch->x_rot_f16
            // (0 OpenMP threads woken during M=2048 prefill -> 0 CPU contention with Blackhole P150A!).
            const int M_sh = ((M + 63) / 64) * 64;
            const bool sh_direct_f16 = (M_sh <= max_chunk && M_sh <= 2048);
            const bool can_overlap_pre = (g.ev_mb_c2d[0] != nullptr && g.ev_mb_c2d[1] != nullptr &&
                                          g.ev_mb_c2d[2] != nullptr && g.ev_mb_d2c[0] != nullptr);
            const bool gpu_p150_bf16 = (ch && p150_shm_x_reg && g.d_tmp_a != nullptr && can_overlap_pre);

            if (sh_direct_f16 && x_f16_ready) {
                if (M_sh > M) {
                    hip_check(hipMemsetAsync(g.d_x_f16 + size_t(M) * kHiddenSize, 0,
                                             size_t(M_sh - M) * kHiddenSize * sizeof(uint16_t), g.stream), "pad x_f16 router");
                }
                const float alpha = 1.0f, beta = 0.0f;
                rocblas_gemm_ex_tuned(
                    g.blas, rocblas_operation_transpose, rocblas_operation_none,
                    1024, M_sh, kHiddenSize, &alpha,
                    moe.d_router_w01_f16, rocblas_datatype_f16_r, kHiddenSize,
                    g.d_x_f16,            rocblas_datatype_f16_r, kHiddenSize,
                    &beta,
                    g.d_f32_tmp1, rocblas_datatype_f32_r, 1024,
                    g.d_f32_tmp1, rocblas_datatype_f32_r, 1024, "moe router f16 direct");
                launch_split_sigmoid_top10_route(g.d_f32_tmp1, moe.d_corr_bias, moe.d_scale_param,
                                                 g.d_topk_ids, g.d_topk_wts, M, g.stream, kNumRoutedExperts);
            } else {
                bf16_linear_f32(g, g.d_x_in, moe.d_router_w01_f16, g.d_f32_tmp1, M, 1024, kHiddenSize);
                launch_split_sigmoid_top10_route(g.d_f32_tmp1, moe.d_corr_bias, moe.d_scale_param,
                                                 g.d_topk_ids, g.d_topk_wts, M, g.stream, kNumRoutedExperts);
            }

            if (can_overlap_pre) {
                hip_check(hipEventRecord(g.ev_mb_c2d[0], g.stream), "moe pre topk rec");
                hip_check(hipStreamWaitEvent(g.stream_sh, g.ev_mb_c2d[0], 0), "moe pre topk wait");
                hip_check(hipMemcpyAsync(h_topk_ids_pin, g.d_topk_ids, size_t(M) * kTopKExperts * sizeof(int32_t),
                                         hipMemcpyDeviceToHost, g.stream_sh), "D2H topk_ids sh");
                hip_check(hipMemcpyAsync(h_topk_wts_pin, g.d_topk_wts, size_t(M) * kTopKExperts * sizeof(float),
                                         hipMemcpyDeviceToHost, g.stream_sh), "D2H topk_wts sh");
                if (nonowner_oproj_deferred && bcast_tp > 1) {
                    hip_check(hipMemcpyAsync(h_mhc_mix_pin, g.d_post_mix,
                                             size_t(M) * 4u * sizeof(float),
                                             hipMemcpyDeviceToHost, g.stream_sh), "D2H post_mix sh");
                    hip_check(hipMemcpyAsync(h_mhc_mix_pin + size_t(M) * 4u, g.d_comb_mix,
                                             size_t(M) * 16u * sizeof(float),
                                             hipMemcpyDeviceToHost, g.stream_sh), "D2H comb_mix sh");
                }
                hip_check(hipEventRecord(g.ev_mb_d2c[0], g.stream_sh), "moe pre topk d2h rec");
            }

            // Launch fc1_lat + WHT128 + lat_rot D2Hs immediately on g.stream!
            if (sh_direct_f16) {
                if (!x_f16_ready && M_sh > M) {
                    hip_check(hipMemsetAsync(g.d_x_f16 + size_t(M) * kHiddenSize, 0,
                                             size_t(M_sh - M) * kHiddenSize * sizeof(uint16_t), g.stream), "pad x_f16 fc1");
                }
                f16_linear_f16(g, g.d_x_f16, moe.d_fc1_lat, g.d_h_lat, M_sh, kLatentDim, kHiddenSize);
            } else {
                bf16_linear_f16(g, g.d_x_in, moe.d_fc1_lat, g.d_h_lat, M, kLatentDim, kHiddenSize);
            }
            launch_wht128_f16_to_f16_and_bf16(g.d_h_lat, g.d_signs_in[size_t(l)], g.d_x_rot_f16,
                                              gpu_p150_bf16 ? g.d_tmp_a : nullptr, M, kLatentDim, g.stream);
            if (gpu_p150_bf16) {
                hip_check(hipMemcpyAsync(ch->x_rot_f16, g.d_tmp_a, size_t(M) * kLatentDim * sizeof(uint16_t),
                                         hipMemcpyDeviceToHost, g.stream), "D2H p150 lat_rot_bf16 direct");
                hip_check(hipEventRecord(g.ev_mb_c2d[1], g.stream), "moe pre p150 bf16 rec");
                // Offload h_lat_rot_f16_pin 8 MiB D2H onto g.stream_sh so g.stream starts owner mi50_moe immediately!
                hip_check(hipStreamWaitEvent(g.stream_sh, g.ev_mb_c2d[1], 0), "moe pre sh wait wht128");
                hip_check(hipMemcpyAsync(h_lat_rot_f16_pin, g.d_x_rot_f16, size_t(M) * kLatentDim * sizeof(uint16_t),
                                         hipMemcpyDeviceToHost, g.stream_sh), "D2H lat_rot_f16 sh");
                hip_check(hipEventRecord(g.ev_mb_c2d[2], g.stream_sh), "moe pre lat_f16 sh rec");
                // Sync ONLY 320 KiB topk+mhc_mix (+ prior 16 MiB o_proj D2H already ahead on g.stream_sh!)
                // while fc1_lat (2.12 ms) runs on g.stream!
                hip_check(hipEventSynchronize(g.ev_mb_d2c[0]), "sync topk+mhc_mix sh");
                o_proj_host_d2h_pending = false;
            } else {
                if (!can_overlap_pre) {
                    hip_check(hipMemcpyAsync(h_topk_ids_pin, g.d_topk_ids, size_t(M) * kTopKExperts * sizeof(int32_t),
                                             hipMemcpyDeviceToHost, g.stream), "D2H topk_ids");
                    hip_check(hipMemcpyAsync(h_topk_wts_pin, g.d_topk_wts, size_t(M) * kTopKExperts * sizeof(float),
                                             hipMemcpyDeviceToHost, g.stream), "D2H topk_wts");
                    if (nonowner_oproj_deferred && bcast_tp > 1) {
                        hip_check(hipMemcpyAsync(h_mhc_mix_pin, g.d_post_mix,
                                                 size_t(M) * 4u * sizeof(float),
                                                 hipMemcpyDeviceToHost, g.stream), "D2H post_mix");
                        hip_check(hipMemcpyAsync(h_mhc_mix_pin + size_t(M) * 4u, g.d_comb_mix,
                                                 size_t(M) * 16u * sizeof(float),
                                                 hipMemcpyDeviceToHost, g.stream), "D2H comb_mix");
                    }
                }
                hip_check(hipMemcpyAsync(h_lat_rot_f16_pin, g.d_x_rot_f16, size_t(M) * kLatentDim * sizeof(uint16_t),
                                         hipMemcpyDeviceToHost, g.stream), "D2H lat_rot_f16 single");
                if (can_overlap_pre) {
                    hip_check(hipEventRecord(g.ev_mb_c2d[2], g.stream), "moe pre lat_f16 rec");
                    hip_check(hipEventSynchronize(g.ev_mb_d2c[0]), "sync topk+mhc_mix sh");
                    o_proj_host_d2h_pending = false;
                } else {
                    hip_check(hipStreamSynchronize(g.stream), "sync router+lat_f16");
                }
            }
            if (do_pre_xray) {
                t_xr_rt = std::chrono::steady_clock::now();
                chunk_pre_router_ms += std::chrono::duration<double, std::milli>(t_xr_rt - t_xr_bl).count();
            }

            // Launch non-owner o_proj H2D + attn mhc_post_update + mlp_mhc mixes on idle non-owner GPUs
            // DURING fc1_lat (2.12 ms) -> finishes BEFORE mi50_moe starts (0 PCIe/HBM2 contention with mi50_moe!)
            bool nonowner_early_dispatched = false;
            if (can_overlap_pre && nonowner_oproj_deferred && bcast_tp > 1) {
                constexpr int kOrdDefer[4] = {2, 1, 3, 0};
                for (int oi = 0; oi < bcast_tp; ++oi) {
                    const int rr = (bcast_tp == 4) ? kOrdDefer[oi] : oi;
                    if (rr == bcast_owner_rank) continue;
                    auto& w = *dev_map[tp_devs[rr]];
                    hip_check(hipSetDevice(w.hip_dev), "early nonown dev");
                    hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_o_out_pin,
                                             size_t(M) * kHiddenSize * sizeof(uint16_t),
                                             hipMemcpyHostToDevice, w.stream_sh), "early nonown o_proj h2d");
                    launch_mhc_post_update(w.d_sub_out, w.d_res_a, w.d_post_mix, w.d_comb_mix,
                                           w.d_res_b, M, kHiddenSize, w.stream_sh);
                    hip_check(hipMemcpyAsync(w.d_post_mix, h_mhc_mix_pin,
                                             size_t(M) * 4u * sizeof(float),
                                             hipMemcpyHostToDevice, w.stream_sh), "early nonown post_mix h2d");
                    hip_check(hipMemcpyAsync(w.d_comb_mix, h_mhc_mix_pin + size_t(M) * 4u,
                                             size_t(M) * 16u * sizeof(float),
                                             hipMemcpyHostToDevice, w.stream_sh), "early nonown comb_mix h2d");
                    hip_check(hipEventRecord(w.ev_res_b_ready, w.stream_sh), "early nonown res_b rec");
                }
                hip_check(hipSetDevice(g.hip_dev), "early nonown restore dev");
                nonowner_early_dispatched = true;
            }

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

            bool pre_p150_work = false;
            uint32_t pre_p150_seq = 0u;
            const uint16_t* pre_p150_y = nullptr;
            if (ch) {
                const bool offload_shadow = (gfx906_offload_shadow_enabled() && M >= 256 && l < mcfg.num_hidden_layers);
                const int16_t* lut = offload_shadow ? p150_slot_lut_bb_prefill.data() : p150_slot_lut_bb.data();
                const bool use_bb_lut = (l < mcfg.num_hidden_layers);
                for (int r = 0; r < M; ++r) {
                    const int32_t* ids_r = h_topk_ids_pin + size_t(r) * kTopKExperts;
                    const float*   wts_r = h_topk_wts_pin + size_t(r) * kTopKExperts;
                    int32_t* dst_slots   = ch->row_slots[r];
                    float*   dst_wts     = ch->row_weights[r];
                    for (int s = 0; s < kTopKExperts; ++s) {
                        const int32_t eid = ids_r[s];
                        const float wt    = wts_r[s];
                        int32_t pslot = -1;
                        if (wt != 0.0f && uint32_t(eid) < uint32_t(kNumRoutedExperts)) {
                            if (use_bb_lut) {
                                pslot = int32_t(lut[eid]);
                            } else {
                                const int cand_g = int(bb_owner_gpu[size_t(eid)]);
                                const int cand_l = (cand_g >= 0) ? int(bb_local_slot[size_t(eid)]) : -1;
                                const bool mi50_owns = (cand_g >= 0) && (cand_l < gpus[size_t(cand_g)]->bb_base_count);
                                if (!mi50_owns && eid >= p150_exp_begin && eid < p150_exp_end) pslot = eid - p150_exp_begin;
                            }
                        }
                        if (pslot >= 0) {
                            dst_slots[s] = pslot;
                            dst_wts[s]   = wt;
                            pre_p150_work = true;
                        } else {
                            dst_slots[s] = -1;
                            dst_wts[s]   = 0.0f;
                        }
                    }
                }
            }

            if (do_pre_xray) {
                t_xr_pk = std::chrono::steady_clock::now();
                chunk_pre_pack_ms += std::chrono::duration<double, std::milli>(t_xr_pk - t_xr_rt).count();
            }

            // Ring P150A doorbell the instant direct GPU-WHT128 BF16 lands in ch->x_rot_f16 (0 OpenMP threads!)
            if (gpu_p150_bf16) {
                hip_check(hipEventSynchronize(g.ev_mb_c2d[1]), "sync p150 bf16 direct");
                if (pre_p150_work) {
                    ch->flags     = ERNIE55_P150_FLAG_X_BF16 | ERNIE55_P150_FLAG_Y_BF16;
                    ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                    ch->layer_idx = uint32_t(l);
                    ch->rows      = uint32_t(M);
                    ch->top_k     = kTopKExperts;
                    pre_p150_seq  = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                    ch->req_seq.store(pre_p150_seq, std::memory_order_release);
                    pre_p150_y = ch->y_out_bf16;
                }
            } else {
                if (can_overlap_pre) {
                    hip_check(hipEventSynchronize(g.ev_mb_c2d[2]), "sync lat_f16 fallback");
                }
                if (ch && pre_p150_work) {
                    avx2_f16_to_bf16(h_lat_rot_f16_pin, ch->x_rot_f16, size_t(M) * kLatentDim, 8);
                    ch->flags     = ERNIE55_P150_FLAG_X_BF16 | ERNIE55_P150_FLAG_Y_BF16;
                    ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                    ch->layer_idx = uint32_t(l);
                    ch->rows      = uint32_t(M);
                    ch->top_k     = kTopKExperts;
                    pre_p150_seq  = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                    ch->req_seq.store(pre_p150_seq, std::memory_order_release);
                    pre_p150_y = ch->y_out_bf16;
                }
            }
            if (do_pre_xray) {
                t_xr_fc = std::chrono::steady_clock::now();
                chunk_pre_fc1_ms += std::chrono::duration<double, std::milli>(t_xr_fc - t_xr_pk).count();
            }
            if (profiling_enabled) {
                prof.router_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_r0).count();
            }

            // Launch tuned shared_experts on g.stream_sh AFTER P150A doorbell (sol=33065: 4.8 ms vs 7.9 ms!)
            if (sh_direct_f16) {
                // Wave-10 C2/C4 + Wave-11 C5: 3 tuned direct FP16 Type_HH_HPA GEMMs reading g.d_x_f16 directly!
                const float alpha = 1.0f, beta = 0.0f;
                rocblas_gemm_ex_tuned(
                    g.blas_sh, rocblas_operation_transpose, rocblas_operation_none,
                    kMoeInterSize, M_sh, kHiddenSize, &alpha,
                    moe.d_sh_w1, rocblas_datatype_f16_r, kHiddenSize,
                    g.d_x_f16,   rocblas_datatype_f16_r, kHiddenSize,
                    &beta,
                    g.d_sh_tmp_a, rocblas_datatype_f16_r, kMoeInterSize,
                    g.d_sh_tmp_a, rocblas_datatype_f16_r, kMoeInterSize, "sh w1 direct hh_hpa");
                rocblas_gemm_ex_tuned(
                    g.blas_sh, rocblas_operation_transpose, rocblas_operation_none,
                    kMoeInterSize, M_sh, kHiddenSize, &alpha,
                    moe.d_sh_w3, rocblas_datatype_f16_r, kHiddenSize,
                    g.d_x_f16,   rocblas_datatype_f16_r, kHiddenSize,
                    &beta,
                    g.d_sh_tmp_b, rocblas_datatype_f16_r, kMoeInterSize,
                    g.d_sh_tmp_b, rocblas_datatype_f16_r, kMoeInterSize, "sh w3 direct hh_hpa");
                launch_situ_glu_f16(g.d_sh_tmp_a, g.d_sh_tmp_b, g.d_sh_tmp_a, M_sh, kMoeInterSize,
                                    mcfg.situ_beta, mcfg.situ_linear_beta, g.stream_sh);
                rocblas_gemm_ex_tuned(
                    g.blas_sh, rocblas_operation_transpose, rocblas_operation_none,
                    kHiddenSize, M_sh, kMoeInterSize, &alpha,
                    moe.d_sh_w2,  rocblas_datatype_f16_r, kMoeInterSize,
                    g.d_sh_tmp_a, rocblas_datatype_f16_r, kMoeInterSize,
                    &beta,
                    g.d_sh_out, rocblas_datatype_f16_r, kHiddenSize,
                    g.d_sh_out, rocblas_datatype_f16_r, kHiddenSize, "sh w2 direct hh_hpa");
            } else {
                constexpr size_t kShXCap = 512u * kHiddenSize;
                constexpr size_t kShYCap = 512u * kHiddenSize * 2u;
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w1, g.d_sh_tmp_a, M, kMoeInterSize, kHiddenSize, 512, kShXCap, kShYCap);
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w3, g.d_sh_tmp_b, M, kMoeInterSize, kHiddenSize, 512, kShXCap, kShYCap);
                launch_situ_glu_bf16(g.d_sh_tmp_a, g.d_sh_tmp_b, g.d_sh_tmp_a, M, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream_sh);
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_sh_tmp_a, moe.d_sh_w2, g.d_sh_out, M, kHiddenSize, kMoeInterSize, 512, kShXCap, kShYCap);
            }
            // Wave-11 C3: Record ev_sh_done IMMEDIATELY after sh_w2 (decoupled from remote MI50/P150A H2Ds)!
            hip_check(hipEventRecord(g.ev_sh_done, g.stream_sh), "sh done after w2");
            if (M >= 256) {
                const auto t_mp1 = std::chrono::steady_clock::now();
                chunk_moe_pre_ms += std::chrono::duration<double, std::milli>(t_mp1 - t_mp0).count();
                if (do_pre_xray) {
                    chunk_pre_sh_ms += std::chrono::duration<double, std::milli>(t_mp1 - t_xr_fc).count();
                }
            }
            d_y_final = dispatch_routed_moe_all_devices(
                l, is_echo, g, M, nullptr, 0.0f, true, pre_p150_work, pre_p150_seq, pre_p150_y,
                moe.d_lat_norm, mcfg.rms_norm_eps,
                (nonowner_oproj_deferred && !nonowner_early_dispatched) ? bcast_tp : 0,
                (nonowner_oproj_deferred && !nonowner_early_dispatched) ? bcast_owner_rank : -1,
                gpu_p150_bf16 ? g.ev_mb_c2d[2] : nullptr);
            const auto t_po0 = (M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (bcast_tp > 1 && M >= 256) {
                // Wave-11 C5+/T10: 2-MB FP16 fc2_lat + add->BF16 overlapped with host-pinned SDMA moe bcast (M >= 256)!
                const auto t_mb0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const int kNb = 2;
                int mb_off[4]{}, mb_len[4]{};
                const int base_mb = (M / kNb) & ~63;
                for (int b = 0; b < kNb; ++b) {
                    mb_off[b] = b * base_mb;
                    mb_len[b] = (b + 1 < kNb) ? base_mb : (M - b * base_mb);
                }
                constexpr int kOrdH2d[4] = {2, 1, 3, 0};

                const bool moe_bcast_sdma = (gfx906_ar_device_mode() == 2 && kNb == 2 &&
                                             g.ev_mb_vha_h2d[2] != nullptr && g.ev_mb_vha_h2d[3] != nullptr);
                // Queue microbatches: fc2_lat + add->BF16 on g.stream -> continuous D2H pipeline on g.stream_sh!
                for (int b = 0; b < kNb; ++b) {
                    const size_t loff = size_t(mb_off[b]) * kLatentDim;
                    const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                    const size_t olen = size_t(mb_len[b]) * kHiddenSize;
                    f16_linear_f16(g, g.d_h_lat + loff, moe.d_fc2_lat, g.d_sub_out + ooff, mb_len[b], kHiddenSize, kLatentDim);
                    if (b == 0) {
                        hip_check(hipStreamWaitEvent(g.stream, g.ev_sh_done, 0), "wait sh before mb0 add");
                    }
                    if (sh_direct_f16) launch_add_f16_f16_to_bf16(g.d_sub_out + ooff, g.d_sh_out + ooff, int(olen), g.stream);
                    else               launch_add_f16_bf16_to_bf16(g.d_sub_out + ooff, g.d_sh_out + ooff, int(olen), g.stream);
                    hip_check(hipEventRecord(g.ev_mb_c2d[b], g.stream), "moe mb c2d rec");
                    hip_check(hipStreamWaitEvent(g.stream_sh, g.ev_mb_c2d[b], 0), "moe mb c2d wait");
                    hip_check(hipMemcpyAsync(h_tp_ar_out_pin + ooff, g.d_sub_out + ooff, olen * sizeof(uint16_t),
                                             hipMemcpyDeviceToHost, g.stream_sh), "moe mb d2h");
                    hip_check(hipEventRecord(moe_bcast_sdma ? g.ev_mb_vha_h2d[2 + b] : g.ev_mb_d2c[b], g.stream_sh), "moe mb d2h rec");
                }

                // Stream each completed microbatch full-duplex to the 3 non-owner ranks (slow-first)!
                for (int b = 0; b < kNb; ++b) {
                    const size_t ooff = size_t(mb_off[b]) * kHiddenSize;
                    const size_t olen = size_t(mb_len[b]) * kHiddenSize;
                    if (!moe_bcast_sdma) {
                        hip_check(hipSetDevice(g.hip_dev), "moe mb d2h sync dev");
                        hip_check(hipEventSynchronize(g.ev_mb_d2c[b]), "moe mb d2h sync");
                    }
                    for (int oi = 0; oi < bcast_tp; ++oi) {
                        const int r = (bcast_tp == 4) ? kOrdH2d[oi] : oi;
                        if (r == bcast_owner_rank) continue;
                        auto& w = *dev_map[tp_devs[r]];
                        hip_check(hipSetDevice(w.hip_dev), "moe mb h2d dev");
                        if (b == 0 && nonowner_oproj_deferred && w.ev_res_b_ready != nullptr) {
                            hip_check(hipStreamWaitEvent(w.stream, w.ev_res_b_ready, 0), "moe mb wait res_b");
                        }
                        if (moe_bcast_sdma) {
                            hip_check(hipStreamWaitEvent(w.stream, g.ev_mb_vha_h2d[2 + b], 0), "moe mb sdma wait");
                        }
                        hip_check(hipMemcpyAsync(w.d_sub_out + ooff, h_tp_ar_out_pin + ooff, olen * sizeof(uint16_t),
                                                 hipMemcpyHostToDevice, w.stream), "moe mb h2d");
                    }
                }
                if (bcast_done_out) *bcast_done_out = true;
                if (M >= 256) {
                    chunk_moe_post_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_po0).count();
                }
                if (profiling_enabled) {
                    prof_sync_tp(bcast_tp, false, g.hip_dev);
                    prof.tp4_comm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mb0).count();
                }
                return;
            }
            if ((M & 63) != 0) {
                hip_check(hipStreamWaitEvent(g.stream, g.ev_sh_done, 0), "wait sh before fc2 tail");
            }
            f16_linear_f16(g, g.d_h_lat, moe.d_fc2_lat, g.d_sub_out, M, kHiddenSize, kLatentDim);
            if ((M & 63) == 0) {
                hip_check(hipStreamWaitEvent(g.stream, g.ev_sh_done, 0), "wait sh before add");
            }
            if (sh_direct_f16) launch_add_f16_f16_to_bf16(g.d_sub_out, g.d_sh_out, M * kHiddenSize, g.stream);
            else               launch_add_f16_bf16_to_bf16(g.d_sub_out, g.d_sh_out, M * kHiddenSize, g.stream);
            if (M >= 256) {
                chunk_moe_post_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_po0).count();
            }
            return;
        } else if (M <= 8 && !is_echo && int(gpus.size()) == 5 &&
                   h_moe_mapped_slabs != nullptr && router_capture_dst == nullptr &&
                   gfx906_mapped_moe_enabled()) {
            // Wave-11 Cut D3.2: Zero-Host-Sync Mapped 5-MI50 Decode MoE + 1-CTA GPU ExpertDispatchGroup Builder!
            // Eliminates 252+ hipStreamSynchronize/step across backbone (42L) AND 3L DSpark draft (0 B new VRAM!).
            const int n_gpus = 5;
            int own_gi = 0;
            for (int gi = 0; gi < n_gpus; ++gi) {
                if (gpus[size_t(gi)]->hip_dev == g.hip_dev) { own_gi = gi; break; }
            }
            const uint32_t ep = ++moe_mapped_epoch;
            const uint32_t slot = (ep - 1u) & 15u;
            const int max_g = std::min(80, M * kTopKExperts);

            bf16_linear_f32(g, g.d_x_in, moe.d_router_w01_f16, g.d_f32_tmp1, M, 1024, kHiddenSize);
            const int max_exp = (draft_mi50_only && p150_ok) ? p150_exp_begin : kNumRoutedExperts;
            launch_split_sigmoid_top10_route(g.d_f32_tmp1, moe.d_corr_bias, moe.d_scale_param,
                                             g.d_topk_ids, g.d_topk_wts, M, g.stream, max_exp);
            bf16_linear_bf16(g, g.d_x_in, moe.d_fc1_lat, g.d_h_lat, M, kLatentDim, kHiddenSize);
            launch_wht128_bf16_to_f16(g.d_h_lat, g.d_signs_in[size_t(l)], g.d_x_rot_f16, M, kLatentDim, g.stream);
            launch_moe_mapped_publish_xrot_topk(
                g.d_x_rot_f16, g.d_topk_ids, g.d_topk_wts,
                g.d_moe_mapped_slabs, g.d_moe_mapped_ready, g.d_moe_mapped_done,
                M, kLatentDim, kMoeMappedStride, ep, g.stream);

            // Wave-12 Cut D12.1b: Doorbell-First P150A decode overlap — wake background relay thread
            // IMMEDIATELY after publish_xrot_topk (BEFORE shared_experts & 5-GPU MI50 enqueue!) so
            // Blackhole P150A starts the instant rdy5 flips (0 CPU launch queue lag!).
            int wait_p150_mapped = 0;
            bool p150_has_work = false;
            if (ch != nullptr && p150_relay_thr.joinable()) {
                submit_p150_decode_job(ep, int32_t(l), int32_t(M), l < mcfg.num_hidden_layers);
                wait_p150_mapped = 1;
            }

            if (has_sh) {
                hip_check(hipEventRecord(g.ev_sh_start, g.stream), "sh start map");
                hip_check(hipStreamWaitEvent(g.stream_sh, g.ev_sh_start, 0), "sh wait map");
                constexpr size_t kShXCap = 512u * kHiddenSize;
                constexpr size_t kShYCap = 512u * kHiddenSize * 2u;
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w1, g.d_sh_tmp_a, M, kMoeInterSize, kHiddenSize, 512, kShXCap, kShYCap);
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w3, g.d_sh_tmp_b, M, kMoeInterSize, kHiddenSize, 512, kShXCap, kShYCap);
                launch_situ_glu_bf16(g.d_sh_tmp_a, g.d_sh_tmp_b, g.d_sh_tmp_a, M, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream_sh);
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_sh_tmp_a, moe.d_sh_w2, g.d_sh_out, M, kHiddenSize, kMoeInterSize, 512, kShXCap, kShYCap);
                hip_check(hipEventRecord(g.ev_sh_done, g.stream_sh), "sh done map");
            }

            // Enqueue ALL 5 MI50 GPUs completely asynchronously with 4-kernel fused decode MoE chain:
            //   1. Fused mapped wait+copy+groups+80-bit mask (-180 launches/step) & 0 hipMemsetAsync (-225 launches/step!)
            //   2. Fused GU-SiTU + LDS WHT128 -> FP16 (-225 launches/step & 0 d_z_f32 HBM2 bounce!)
            //   3. Persistent down_slot + masked reduce/publish (skips ~80% empty HBM2 slot reads!)
            for (int gi = 0; gi < n_gpus; ++gi) {
                auto& w = *gpus[size_t(gi)];
                const int max_loc = (l < mcfg.num_hidden_layers) ? w.bb_count() : w.bb_base_count;
                hip_check(hipSetDevice(w.hip_dev), "moe map worker dev");
                if (gi == own_gi) {
                    launch_build_decode_expert_groups(g.d_topk_ids, g.d_topk_wts, w.d_g2l_bb,
                                                      w.d_groups, M, max_loc, max_g, w.stream, w.d_csr_counts);
                } else {
                    launch_moe_mapped_wait_copy_and_build_groups(
                        w.d_x_rot_f16, w.d_moe_mapped_slabs, w.d_moe_mapped_ready,
                        w.d_g2l_bb, w.d_groups, w.d_csr_counts,
                        M, kLatentDim, kMoeMappedStride, ep, max_loc, max_g, w.stream);
                }
                // Wave-11 Turn-6 Cut B: restore decoupled 48-CTA/expert persistent GU-SiTU (dim3(48,16))
                // + persistent WHT128 (dim3(12,24)) using device-side w.d_csr_counts (0 host syncs!),
                // keeping full 60-CU saturation at ng=3..10 alongside fused mapped wait+copy+groups
                // and zero-memset 80-bit active-slot mask.
                launch_hlut4_fdot2_grouped_gu_situ_persist(
                    w.d_x_rot_f16, w.d_groups, w.d_csr_counts, max_g,
                    w.d_w1[size_t(l)], w.d_w3[size_t(l)],
                    w.d_levels8_f16, w.d_z_f32, kMoeInterSize, kLatentDim,
                    mcfg.situ_beta, mcfg.situ_linear_beta, w.stream);
                launch_wht128_grouped_f32_to_f16_persist(
                    w.d_z_f32, w.d_groups, w.d_csr_counts, max_g, w.d_signs_inter[size_t(l)],
                    w.d_z_rot_f16, kMoeInterSize, w.stream);
                launch_hlut4_fdot2_grouped_down_slot_persist(
                    w.d_z_rot_f16, w.d_groups, w.d_csr_counts, max_g, w.d_w2[size_t(l)],
                    w.d_levels8_f16, w.d_y_slots_f32, kLatentDim, kMoeInterSize, w.stream);
                if (gi == own_gi) {
                    launch_reduce_moe_slots_f32(w.d_y_slots_f32, w.d_y_acc_f32, M, kLatentDim, w.stream, w.d_csr_counts + 1);
                } else {
                    launch_moe_mapped_reduce_slots_publish_f16(
                        w.d_y_slots_f32, w.d_moe_mapped_slabs, w.d_moe_mapped_ready, w.d_moe_mapped_done,
                        gi, M, kLatentDim, kMoeMappedStride, ep, w.stream, w.d_csr_counts + 1);
                }
            }

            // Fallback inline P150A dispatch only when relay thread is unavailable
            if (wait_p150_mapped == 0 && ch != nullptr) {
                const uint32_t* rdy5 = &h_moe_mapped_ready[5 * 16];
                while (__atomic_load_n(rdy5, __ATOMIC_ACQUIRE) < ep) p150_cpu_relax();
                const uint16_t* s5_host = h_moe_mapped_slabs + (size_t(slot) * 7u + 5u) * size_t(kMoeMappedStride);
                const int32_t* ids_h = reinterpret_cast<const int32_t*>(s5_host + 16384u);
                const float*   wts_h = reinterpret_cast<const float*>(s5_host + 16544u);
                const int16_t* lut = p150_slot_lut_bb.data();
                const bool use_bb_lut = (l < mcfg.num_hidden_layers);
                for (int r = 0; r < M; ++r) {
                    for (int s = 0; s < kTopKExperts; ++s) {
                        const int32_t eid = ids_h[r * kTopKExperts + s];
                        const float   wt  = wts_h[r * kTopKExperts + s];
                        int32_t pslot = -1;
                        if (wt != 0.0f && uint32_t(eid) < uint32_t(kNumRoutedExperts)) {
                            if (use_bb_lut) {
                                pslot = int32_t(lut[eid]);
                            } else {
                                const int cg = int(bb_owner_gpu[size_t(eid)]);
                                const int cl = (cg >= 0) ? int(bb_local_slot[size_t(eid)]) : -1;
                                const bool mi50_owns = (cg >= 0) && (cl < gpus[size_t(cg)]->bb_base_count);
                                if (!mi50_owns && eid >= p150_exp_begin && eid < p150_exp_end) pslot = eid - p150_exp_begin;
                            }
                        }
                        if (pslot >= 0) {
                            ch->row_slots[r][s]   = pslot;
                            ch->row_weights[r][s] = wt;
                            p150_has_work = true;
                        } else {
                            ch->row_slots[r][s]   = -1;
                            ch->row_weights[r][s] = 0.0f;
                        }
                    }
                }
                if (p150_has_work) {
                    const size_t n_lat = size_t(M) * kLatentDim;
                    avx2_f16_to_f32(s5_host, ch->x_rot_f32, n_lat, 1);
                    ch->flags     = 0u;
                    ch->op        = ERNIE55_P150_OP_MOE_EXEC;
                    ch->layer_idx = uint32_t(l);
                    ch->rows      = uint32_t(M);
                    ch->top_k     = kTopKExperts;
                    const uint32_t seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
                    ch->req_seq.store(seq, std::memory_order_release);
                    while (ch->resp_seq.load(std::memory_order_acquire) != seq) p150_cpu_relax();
                    if (ch->status != 0) throw std::runtime_error("Blackhole P150A worker returned error on layer " + std::to_string(l));
                    for (size_t i = 0; i < n_lat; ++i) h_lat_rot_bf16_pin[i] = f32_to_bf16_rne(ch->y_out_f32[i]);
                    hip_check(hipSetDevice(g.hip_dev), "moe map p150 h2d dev");
                    hip_check(hipMemcpyAsync(g.d_tmp_d, h_lat_rot_bf16_pin, n_lat * sizeof(uint16_t),
                                             hipMemcpyHostToDevice, g.stream), "moe map p150 h2d");
                }
            }

            hip_check(hipSetDevice(g.hip_dev), "moe map merge dev");
            launch_moe_mapped_merge_rmsnorm_to_bf16(
                g.d_y_acc_f32, g.d_moe_mapped_slabs, g.d_moe_mapped_ready, g.d_moe_mapped_done,
                p150_has_work ? g.d_tmp_d : nullptr, moe.d_lat_norm, g.d_h_lat,
                own_gi, n_gpus, M, kLatentDim, kMoeMappedStride, ep, mcfg.rms_norm_eps, g.stream, wait_p150_mapped);
            bf16_linear_bf16(g, g.d_h_lat, moe.d_fc2_lat, g.d_sub_out, M, kHiddenSize, kLatentDim);
            if (has_sh) {
                hip_check(hipStreamWaitEvent(g.stream, g.ev_sh_done, 0), "wait sh map");
                launch_add_bf16_inplace(g.d_sub_out, g.d_sh_out, M * kHiddenSize, g.stream);
            } else {
                bf16_linear_bf16(g, g.d_x_in, moe.d_sh_w1, g.d_tmp_a, M, kMoeInterSize, kHiddenSize);
                bf16_linear_bf16(g, g.d_x_in, moe.d_sh_w3, g.d_tmp_b, M, kMoeInterSize, kHiddenSize);
                launch_situ_glu_bf16(g.d_tmp_a, g.d_tmp_b, g.d_tmp_a, M, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream);
                bf16_linear_bf16(g, g.d_tmp_a, moe.d_sh_w2, g.d_tmp_d, M, kHiddenSize, kMoeInterSize);
                launch_add_bf16_inplace(g.d_sub_out, g.d_tmp_d, M * kHiddenSize, g.stream);
            }
            return;
        } else {
            drain_p150_decode_relay();
            // Canonical Decode / DSpark / ECHO Path (M <= 32): 100% untouched v1.1.0 path!
            bf16_linear_f32(g, g.d_x_in, moe.d_router_w01_f16, g.d_f32_tmp1, M, 1024, kHiddenSize);
            const int max_exp = (draft_mi50_only && p150_ok) ? p150_exp_begin : kNumRoutedExperts;
            launch_split_sigmoid_top10_route(g.d_f32_tmp1, moe.d_corr_bias, moe.d_scale_param,
                                             g.d_topk_ids, g.d_topk_wts, M, g.stream, max_exp);
            bf16_linear_bf16(g, g.d_x_in, moe.d_fc1_lat, g.d_h_lat, M, kLatentDim, kHiddenSize);
            launch_wht128_bf16_to_f16(g.d_h_lat, g.d_signs_in[size_t(l)], g.d_x_rot_f16, M, kLatentDim, g.stream);
            if (has_sh) {
                hip_check(hipEventRecord(g.ev_sh_start, g.stream), "sh start single");
            }

            hip_check(hipMemcpyAsync(h_topk_ids_pin, g.d_topk_ids, size_t(M) * kTopKExperts * sizeof(int32_t), hipMemcpyDeviceToHost, g.stream), "D2H topk_ids");
            hip_check(hipMemcpyAsync(h_topk_wts_pin, g.d_topk_wts, size_t(M) * kTopKExperts * sizeof(float),   hipMemcpyDeviceToHost, g.stream), "D2H topk_wts");
            hip_check(hipMemcpyAsync(h_lat_rot_f16_pin, g.d_x_rot_f16, size_t(M) * kLatentDim * sizeof(uint16_t), hipMemcpyDeviceToHost, g.stream), "D2H lat_rot");
            if (has_sh) {
                hip_check(hipStreamWaitEvent(g.stream_sh, g.ev_sh_start, 0), "sh wait single");
                constexpr size_t kShXCap = 512u * kHiddenSize;
                constexpr size_t kShYCap = 512u * kHiddenSize * 2u;
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w1, g.d_sh_tmp_a, M, kMoeInterSize, kHiddenSize, 512, kShXCap, kShYCap);
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_x_in, moe.d_sh_w3, g.d_sh_tmp_b, M, kMoeInterSize, kHiddenSize, 512, kShXCap, kShYCap);
                launch_situ_glu_bf16(g.d_sh_tmp_a, g.d_sh_tmp_b, g.d_sh_tmp_a, M, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream_sh);
                f16_linear_bf16_ex(g.blas_sh, g.stream_sh, g.d_sh_x_f16, g.d_sh_y_f32, g.d_sh_tmp_a, moe.d_sh_w2, g.d_sh_out, M, kHiddenSize, kMoeInterSize, 512, kShXCap, kShYCap);
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
            d_y_final = dispatch_routed_moe_all_devices(l, is_echo, g, M);
        }

        launch_rmsnorm_f32_to_bf16(d_y_final, moe.d_lat_norm, g.d_h_lat, M, kLatentDim, mcfg.rms_norm_eps, g.stream);
        bf16_linear_bf16(g, g.d_h_lat, moe.d_fc2_lat, g.d_sub_out, M, kHiddenSize, kLatentDim);
        if (has_sh) {
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
    void run_moe_tp(LayerSpineGpu& L, int M, bool nonowner_oproj_deferred = false, bool x_f16_ready = false) {
        const int tp = L.tp_size;
        const auto dev_for_r = [&](int r) -> int { return L.is_echo ? L.owner_dev : tp_devs[r]; };

        if (L.is_dense) {
            const int inter_loc = mcfg.intermediate_size / tp;
            const bool dense_f16_ar = (tp > 1 && M > 8);
            const bool dense_direct_f16 = (dense_f16_ar && (M & 63) == 0 && M <= max_chunk && M <= 2048);
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                const auto& D = L.dense_tp[r];
                hip_check(hipSetDevice(w.hip_dev), "dense tp dev");
                if (dense_direct_f16) {
                    if (!x_f16_ready) launch_bf16_to_f16(w.d_x_in, w.d_x_f16, M * kHiddenSize, w.stream);
                    f16_linear_f16(w, w.d_x_f16, D.d_gate, w.d_tmp_a, M, inter_loc, kHiddenSize);
                    f16_linear_f16(w, w.d_x_f16, D.d_up,   w.d_tmp_b, M, inter_loc, kHiddenSize);
                    launch_situ_glu_f16(w.d_tmp_a, w.d_tmp_b, w.d_tmp_a, M, inter_loc,
                                        mcfg.situ_beta, mcfg.situ_linear_beta, w.stream);
                    f16_linear_f16(w, w.d_tmp_a, D.d_down, w.d_sub_out, M, kHiddenSize, inter_loc);
                } else {
                    bf16_linear_bf16(w, w.d_x_in, D.d_gate, w.d_tmp_a, M, inter_loc, kHiddenSize);
                    bf16_linear_bf16(w, w.d_x_in, D.d_up,   w.d_tmp_b, M, inter_loc, kHiddenSize);
                    launch_situ_glu_bf16(w.d_tmp_a, w.d_tmp_b, w.d_tmp_a, M, inter_loc,
                                         mcfg.situ_beta, mcfg.situ_linear_beta, w.stream);
                    if (dense_f16_ar) {
                        bf16_linear_f16(w, w.d_tmp_a, D.d_down, w.d_sub_out, M, kHiddenSize, inter_loc);
                    } else {
                        bf16_linear_bf16(w, w.d_tmp_a, D.d_down, w.d_sub_out, M, kHiddenSize, inter_loc);
                    }
                }
            }
            if (dense_f16_ar) {
                tp_allreduce_f16_to_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            } else if (tp > 1) {
                tp_allreduce_bf16(M, kHiddenSize, &Mi50ExpertWorker::d_sub_out, &Mi50ExpertWorker::d_sub_out);
            }
            return;
        }

        auto& g_own = *dev_map[L.owner_dev];
        bool bcast_done = false;
        run_moe_core(L.layer_idx, L.moe, L.is_echo, g_own, M, false, tp, L.owner_tp_rank,
                     &bcast_done, nonowner_oproj_deferred, x_f16_ready);

        // Single-source 1D contiguous broadcast of owner's d_sub_out [M, 4096] -> other (tp - 1) ranks!
        if (tp > 1 && !bcast_done) {
            const auto t_mb0 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const size_t n = size_t(M) * kHiddenSize;
            if (M <= 8 && tp == 4 && !L.is_echo && h_tp_mapped_slabs != nullptr &&
                n <= size_t(kMappedSlabStride) && !nonowner_oproj_deferred && gfx906_mapped_tp4_enabled()) {
                const uint32_t ep = ++tp_mapped_epoch;
                for (int r = 0; r < 4; ++r) {
                    auto& w = *dev_map[tp_devs[r]];
                    hip_check(hipSetDevice(w.hip_dev), "moe bcast map dev");
                    launch_tp4_mapped_bcast_bf16(
                        (r == L.owner_tp_rank) ? w.d_sub_out : nullptr, w.d_sub_out,
                        w.d_tp_mapped_slabs, w.d_tp_mapped_ready, w.d_tp_mapped_done,
                        r, L.owner_tp_rank, int(n), kMappedSlabStride, ep, w.stream);
                }
            } else {
                const size_t nb = n * sizeof(uint16_t);
                hip_check(hipSetDevice(g_own.hip_dev), "moe bcast d2h dev");
                hip_check(hipMemcpyAsync(h_tp_ar_out_pin, g_own.d_sub_out, nb, hipMemcpyDeviceToHost, g_own.stream), "moe bcast d2h");
                hip_check(hipStreamSynchronize(g_own.stream), "moe bcast sync");
                for (int r = 0; r < tp; ++r) {
                    if (r == L.owner_tp_rank) continue;
                    auto& w = *dev_map[tp_devs[r]];
                    hip_check(hipSetDevice(w.hip_dev), "moe bcast h2d dev");
                    if (nonowner_oproj_deferred && w.ev_res_b_ready != nullptr) {
                        hip_check(hipStreamWaitEvent(w.stream, w.ev_res_b_ready, 0), "moe bcast wait res_b");
                    }
                    hip_check(hipMemcpyAsync(w.d_sub_out, h_tp_ar_out_pin, nb, hipMemcpyHostToDevice, w.stream), "moe bcast h2d");
                }
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

        const bool do_bl_xray = (M >= 256 && moe_pre_xray);
        auto bl_sync_owner = [&]() {
            auto& go = *dev_map[L.owner_dev];
            hip_check(hipSetDevice(go.hip_dev), "bl xray dev");
            hip_check(hipStreamSynchronize(go.stream), "bl xray sync");
        };
        std::chrono::steady_clock::time_point t_bl{};
        if (do_bl_xray) {
            t_bl = std::chrono::steady_clock::now();
            bl_sync_owner();
            const auto t_now = std::chrono::steady_clock::now();
            chunk_bk_moe_post += std::chrono::duration<double, std::milli>(t_now - t_bl).count();
            t_bl = t_now;
        }

        const bool attn_x_f16_ready = (!is_first_layer_broadcast && M > 8 && (M & 63) == 0 && M <= max_chunk && M <= 2048);
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
                run_mhc_pre_norm_fast(w, w.d_res_a, L.attn_mhc_tp[r], L.d_in_norm_tp[r], M, attn_x_f16_ready);
            }
        }
        if (do_bl_xray) {
            bl_sync_owner();
            const auto t_now = std::chrono::steady_clock::now();
            chunk_bk_attn_pre += std::chrono::duration<double, std::milli>(t_now - t_bl).count();
        }
        if (profiling_enabled) {
            prof_sync_tp(tp, L.is_echo, L.owner_dev);
            prof.mhc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mc0).count();
        }

        // Wave-11 C1: On TP4 MoE prefill layers (L01..L42, M>=512), deliver o_proj to g_own first,
        // skip dead non-owner run_mhc_pre_norm_fast(mlp_mhc), and overlap the 3 non-owner o_proj H2Ds +
        // attn post_mhc (d_res_b) + 160 KiB owner mlp_mhc mixes behind mi50_moe!
        bool has_all_sh = (tp == 4);
        if (has_all_sh) {
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                if (!w.stream_sh || !w.ev_res_b_ready) { has_all_sh = false; break; }
            }
        }
        const bool defer_nonowner_req =
            (!L.is_dense && !L.is_echo && tp == 4 && M >= 256 && has_all_sh &&
             (!p150_ok || M <= int(ERNIE55_P150_MAX_ROWS)));
        bool nonowner_oproj_deferred = false;

        const auto t_ca0 = (M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        run_attention_tp(L, M, pos0, defer_nonowner_req, &nonowner_oproj_deferred, attn_x_f16_ready);
        if (M >= 256) {
            const double ams = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ca0).count();
            if (L.is_dsa) chunk_dsa_ms += ams;
            else          chunk_hca_ms += ams;
        }

        if (do_bl_xray) {
            t_bl = std::chrono::steady_clock::now();
            bl_sync_owner();
            const auto t_now = std::chrono::steady_clock::now();
            chunk_bk_vha_op += std::chrono::duration<double, std::milli>(t_now - t_bl).count();
            t_bl = t_now;
        }
        const bool moe_x_f16_ready = (!L.is_dense && M > 32 && M <= max_chunk && M <= 2048);
        const auto t_mc1 = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (nonowner_oproj_deferred) {
            // Only g_own runs attn post-mHC + mlp_mhc pre-norm immediately!
            // Non-owner ranks run attn post-mHC + receive 160 KiB owner mlp_mhc mixes on w.stream_sh behind mi50_moe!
            auto& g_own = *dev_map[L.owner_dev];
            hip_check(hipSetDevice(g_own.hip_dev), "layer mhc2 own dev");
            launch_mhc_post_update(g_own.d_sub_out, g_own.d_res_a, g_own.d_post_mix, g_own.d_comb_mix,
                                   g_own.d_res_b, M, kHiddenSize, g_own.stream);
            run_mhc_pre_norm_fast(g_own, g_own.d_res_b, L.mlp_mhc_tp[L.owner_tp_rank],
                                  L.d_post_norm_tp[L.owner_tp_rank], M, moe_x_f16_ready);
        } else {
            for (int r = 0; r < tp; ++r) {
                auto& w = *dev_map[dev_for_r(r)];
                hip_check(hipSetDevice(w.hip_dev), "layer mhc2 dev");
                launch_mhc_post_update(w.d_sub_out, w.d_res_a, w.d_post_mix, w.d_comb_mix, w.d_res_b, M, kHiddenSize, w.stream);
                run_mhc_pre_norm_fast(w, w.d_res_b, L.mlp_mhc_tp[r], L.d_post_norm_tp[r], M,
                                      moe_x_f16_ready && (r == L.owner_tp_rank));
            }
        }
        if (do_bl_xray) {
            bl_sync_owner();
            const auto t_now = std::chrono::steady_clock::now();
            chunk_bk_attn_post += std::chrono::duration<double, std::milli>(t_now - t_bl).count();
        }
        if (profiling_enabled) {
            prof_sync_tp(tp, L.is_echo, L.owner_dev);
            prof.mhc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mc1).count();
        }

        const auto t_cm0 = (M >= 256) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        run_moe_tp(L, M, nonowner_oproj_deferred, moe_x_f16_ready);
        if (M >= 256) {
            chunk_moe_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cm0).count();
        }

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
                                bool capture_dspark_features,
                                std::vector<SparseProbs>* out_sparse_M = nullptr,
                                const SamplingParams* sparse_sp = nullptr)
    {
        auto& g0 = *dev_map[spine_dev0];
        auto& g1 = *dev_map[spine_dev1];

        for (int i = 0; i < M; ++i) {
            h_tok_pin[i] = tok_ids[i];
            h_pos_pin[i] = pos0 + i;
        }

        // Broadcast initial embedding [M, 4096] + pos_ids to all TP ranks once (host DDR4 mirror -> 0 g0 D2H/sync!)
        if (tp_size > 1 && !h_embed_mirror.empty()) {
            const size_t emb_b = size_t(M) * kHiddenSize * sizeof(uint16_t);
            const int V = mcfg.vocab_size;
            for (int i = 0; i < M; ++i) {
                const int32_t safe_tok = std::clamp(tok_ids[i], 0, V - 1);
                std::memcpy(h_tp_ar_out_pin + size_t(i) * kHiddenSize,
                            h_embed_mirror.data() + size_t(safe_tok) * kHiddenSize,
                            kHiddenSize * sizeof(uint16_t));
            }
            for (int r = 0; r < tp_size; ++r) {
                auto& w = *dev_map[tp_devs[r]];
                hip_check(hipSetDevice(w.hip_dev), "H2D bcast dev");
                hip_check(hipMemcpyAsync(w.d_x_in, h_tp_ar_out_pin, emb_b, hipMemcpyHostToDevice, w.stream), "H2D bcast emb");
                hip_check(hipMemcpyAsync(w.d_pos_ids, h_pos_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, w.stream), "H2D bcast pos");
            }
        } else {
            hip_check(hipSetDevice(g0.hip_dev), "fwd dev0");
            hip_check(hipMemcpyAsync(g0.d_tok_ids, h_tok_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, g0.stream), "H2D tok_ids");
            hip_check(hipMemcpyAsync(g0.d_pos_ids, h_pos_pin, size_t(M) * sizeof(int32_t), hipMemcpyHostToDevice, g0.stream), "H2D pos0");
            launch_embed_lookup_bf16(d_embed_bf16, g0.d_tok_ids, g0.d_x_in, M, kHiddenSize, g0.stream);

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
            if (live_prefill_active.load(std::memory_order_relaxed)) {
                live_prefill_layer_done.store(l + 1, std::memory_order_relaxed);
            }

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
        if (out_sparse_M != nullptr && sparse_sp != nullptr && !only_last_logits && M <= 8) {
            launch_mhc_head_contract(g1.d_res_a, d_hc_head_fn, d_hc_head_scale, d_hc_head_base,
                                     d_final_norm, g1.d_x_in, M, kHiddenSize,
                                     mcfg.rms_norm_eps, mcfg.hc_eps, g1.stream);
            compute_lm_head_tp5_sparse(g1.d_x_in, M, true, *sparse_sp, *out_sparse_M);
        } else if (out_logits_MxV != nullptr) {
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
        // Wave-11 Turn-10: Sync all TP ranks after multi-token chunks so the next chunk's host memcpy
        // into h_tp_ar_out_pin / h_pos_pin can NEVER race with trailing Layer-42 non-owner H2D transfers!
        if (M > 8 && tp_size > 1) {
            for (int r = 0; r < tp_size; ++r) {
                auto& w = *dev_map[tp_devs[r]];
                if (w.hip_dev == g1.hip_dev) continue;
                hip_check(hipSetDevice(w.hip_dev), "sync chunk tp dev");
                hip_check(hipStreamSynchronize(w.stream), "sync chunk tp");
            }
            hip_check(hipSetDevice(g1.hip_dev), "sync chunk restore dev1");
        }
        drain_p150_decode_relay();
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
    impl_->max_chunk = std::min(2048, std::max(16, opt.prefill_chunk));

    // Allocate pinned host staging buffers once
    impl_->h_lat_rot_f16_pin = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kLatentDim, impl_->pin_allocs, true);
    impl_->h_lat_rot_bf16_pin = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kLatentDim, impl_->pin_allocs, true);
    impl_->h_topk_ids_pin    = host_pin_alloc<int32_t>(size_t(impl_->max_chunk) * kTopKExperts, impl_->pin_allocs, true);
    impl_->h_topk_wts_pin    = host_pin_alloc<float>(size_t(impl_->max_chunk) * kTopKExperts, impl_->pin_allocs, true);
    impl_->h_y_sum_f32_pin   = host_pin_alloc<float>(size_t(impl_->max_chunk) * kLatentDim, impl_->pin_allocs, true);
    impl_->h_p150_y_pin      = host_pin_alloc<float>(size_t(impl_->max_chunk) * kLatentDim, impl_->pin_allocs, true);
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
        impl_->h_tp_ar_in_pin[r] = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHiddenSize, impl_->pin_allocs, true,
                                                            /*mapped=*/true, &impl_->d_tp_ar_in_pin[r]);
    }
    impl_->h_tp_ar_out_pin  = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHiddenSize, impl_->pin_allocs, true);
    impl_->h_tp_o_out_pin   = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * kHiddenSize, impl_->pin_allocs, true);
    impl_->h_mhc_mix_pin    = host_pin_alloc<float>(size_t(impl_->max_chunk) * 20u, impl_->pin_allocs, true);
    impl_->h_tp_oattn_pin   = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 64u * 512u, impl_->pin_allocs, true,
                                                        /*mapped=*/true, &impl_->d_tp_oattn_pin);
    impl_->h_tp_qabs_pin    = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 64u * 576u, impl_->pin_allocs, true,
                                                        /*mapped=*/true, &impl_->d_tp_qabs_pin);
    impl_->h_tp_qabs_rm_pin = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 64u * 576u, impl_->pin_allocs, true);
    impl_->h_dsa_sel_cnts_pin = host_pin_alloc<int32_t>(size_t(impl_->max_chunk), impl_->pin_allocs, true);
    for (int r = 0; r < LayerSpineGpu::kMaxTp; ++r) {
        impl_->h_tp_rtr_in_pin[r] = host_pin_alloc<float>(size_t(impl_->max_chunk) * 512u, impl_->pin_allocs, true);
        impl_->h_tp_fc1_in_pin[r] = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 1024u, impl_->pin_allocs, true);
        impl_->h_tp_fc2_in_pin[r] = host_pin_alloc<uint16_t>(size_t(impl_->max_chunk) * 1024u, impl_->pin_allocs, true);
    }
    impl_->h_tp_mapped_slabs = host_mapped_alloc<uint16_t>(size_t(Impl::kMappedPingPong) * 4u * size_t(Impl::kMappedSlabStride), impl_->pin_allocs, true);
    impl_->h_tp_mapped_ready = host_mapped_alloc<uint32_t>(4u * 16u, impl_->pin_allocs, true);
    impl_->h_tp_mapped_done  = host_mapped_alloc<uint32_t>(4u * 16u, impl_->pin_allocs, true);
    impl_->tp_mapped_epoch   = 0u;
    impl_->h_moe_mapped_slabs = host_mapped_alloc<uint16_t>(size_t(Impl::kMappedPingPong) * 7u * size_t(Impl::kMoeMappedStride), impl_->pin_allocs, true);
    impl_->h_moe_mapped_ready = host_mapped_alloc<uint32_t>(8u * 16u, impl_->pin_allocs, true);
    impl_->h_moe_mapped_done  = host_mapped_alloc<uint32_t>(8u * 16u, impl_->pin_allocs, true);
    impl_->moe_mapped_epoch   = 0u;
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
    impl_->init_moe_pre_xray();  // Wave-12 Cut P12.1: ERNIE55_MOE_PRE_XRAY=1 enables the sub-stage timers
    { const char* sa = std::getenv("ERNIE55_SA_CASCADE"); impl_->sa_cascade = !(sa && sa[0] == '0' && sa[1] == '\0'); }
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

    // On 5-MI50 318-expert SB-M5-P2-192G topology (HIP_VISIBLE_DEVICES=0,2,4,1,3):
    // g0=c7:00.0 (32G Gen4), g1=88:00.0 (32G Gen4), g2=08:00.0 (32G Gen4), g3=85:00.0 (16G Gen4), g4=05:00.0 (16G Gen3 + Wayland).
    // Moving 5 base experts from g4 (05:00.0) -> g2 (08:00.0) keeps >= 1,160 MiB sysfs free HBM2 on Wayland renderD131
    // and >= 295 MiB sysfs free HBM2 on g2 (08:00.0, 97+1=98)!
    if (impl_->tp_size == 4 && plans.size() == 5 && plans.front().bb_b == 0 && plans.back().bb_e == 318) {
        constexpr int kCounts[5] = {66, 68, 97, 38, 49};
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

    // Wave-11 C5+: Shadow-weighted m64-tile + pair balanced hot-expert assignment across all 5 MI50s!
    // Keeps 05:00.0 (GPU[2], 8.0 GT/s + desktop, 92+6=98 slots) on the coldest tail (~101 tiles, ~11% pairs vs 129 tiles, 21% pairs!)
    // while balancing all 4 Gen-4 MI50s (0 B extra VRAM, 0 bytes changed on disk!).
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
        std::array<double, kNumRoutedExperts> rank_freq{};
        int hist_layers = 0;
        {
            std::ifstream hfs(opt.weight_root + "/router_hist_ernie55.txt");
            if (hfs) {
                int hl = -1;
                std::array<double, kNumRoutedExperts> row_c{};
                while (hfs >> hl) {
                    double rtot = 0.0;
                    bool ok = true;
                    for (int i = 0; i < kNumRoutedExperts; ++i) {
                        if (!(hfs >> row_c[size_t(i)])) { ok = false; break; }
                        rtot += row_c[size_t(i)];
                    }
                    if (!ok) break;
                    if (hl >= 1 && hl <= 42 && rtot > 0.0) {
                        std::sort(row_c.begin(), row_c.end(), std::greater<double>());
                        for (int i = 0; i < kNumRoutedExperts; ++i) {
                            rank_freq[size_t(i)] += row_c[size_t(i)] / rtot;
                        }
                        ++hist_layers;
                    }
                }
            }
        }
        if (hist_layers > 0) {
            for (double& f : rank_freq) f /= double(hist_layers);
        } else {
            double zsum = 0.0;
            for (int i = 0; i < kNumRoutedExperts; ++i) {
                rank_freq[size_t(i)] = 1.0 / std::pow(double(i) + 15.0, 0.95);
                zsum += rank_freq[size_t(i)];
            }
            for (double& f : rank_freq) f /= zsum;
        }
        // Wave-11 Turn-3->4 Owner-Aware Cost Model (audited true 42-layer ratios + owner shared_experts CU contention):
        // On owner layers (21 each for g0/g1), shared_experts (4.8 ms/L) contends with owner mi50_moe on the same 60 CUs,
        // while non-owner g3 (271 ms) and g4 (350 ms) sit idle. Weighting g0/g1/g2 higher (1.18/1.22/1.50) and g3/g4 lower
        // (0.56/0.78) moves hottest experts onto idle g3/g4, eliminating the moe_post owner straggler (0 B VRAM delta!).
        // Wave-11 Turn-4 Cut A (owner-aware recalibration from the honest 42-layer accounting):
        // live pos0=12288 chunk mi50=501ms g=524/580/501/282/359 with 80/79/94/131/102 tiles gives
        // 6.55/7.34/5.33/2.15/3.52 ms/tile. g0/g1 stay slowest because the 42-layer owner accounting
        // charges them the 21 owner layers (shared_experts + merge + o_proj reduce) on top of their
        // 21 non-owner layers, so their weights rise; g3/g4 (16 GiB HBM2, Gen4 x16) take more.
        // Wave-11 Turn-10 Pillar I: owner-aware recalibration from Turn-9 silicon g=525/615/542/309/402 ms
        // (moves hottest tiles off g1 straggler onto idle 16 GiB Gen4 g3/g4 -> 0 B VRAM delta!).
        // Post-swap Switchbox #2 (32 GiB <-> 16 GiB) owner-aware cost weights:
        // All 4 TP4 ranks (g0=c7:00.0, g1=88:00.0, g2=08:00.0, g3=85:00.0) are 16.0 GT/s x16 with zero Wayland;
        // g0 tuned 1.20 -> 1.38 from live 764/602/574/517/536 ms trace, g2 (08:00.0) at 0.88, g4 (05:00.0 Gen3) at 0.86.
        // Wave-12 Cut P12.2: damped first-order minimax correction. The live post-swap chunk
        // (mi50 = 598ms, g = 554/556/598/327/412) shows the previous weights {1.38,1.44,0.88,
        // 0.44,0.86} over-load g0/g1/g2 and starve g3/g4. Scaling each weight by
        // (measured_ms / mean_ms)^0.7 with mean 469.4 ms gives the damped update below, which
        // equalizes projected TIME rather than tile count. kCounts and kExtraPerGpu are unchanged,
        // so this is 0 B net new hipMalloc and the slot reduction order is preserved.
        double cost_w[5] = {1.55, 1.63, 1.04, 0.34, 0.78};
        if (const char* cw_env = std::getenv("ERNIE55_MOE_COST")) {
            std::string cw(cw_env);
            std::vector<std::string> parts;
            size_t p0 = 0;
            while (p0 <= cw.size()) {
                const size_t p1 = cw.find(',', p0);
                parts.push_back(cw.substr(p0, (p1 == std::string::npos ? cw.size() : p1) - p0));
                if (p1 == std::string::npos) break;
                p0 = p1 + 1;
            }
            if (parts.size() == 5) {
                for (int g = 0; g < 5; ++g) {
                    const double v = std::atof(parts[size_t(g)].c_str());
                    if (v > 0.0) cost_w[g] = v;
                }
            }
        }
        std::vector<double> cur_t(static_cast<size_t>(n_g), 0.0);
        std::vector<double> cur_p(static_cast<size_t>(n_g), 0.0);
        if (n_g == 5 && tot_mi50 == 318 && opt.use_p150) {
            // Post-swap Switchbox #2 shadow cache: 7 hottest GDDR6 experts [319..326) across Gen4 GPUs only
            // (g1 88:00.0 at 2 -> 385 MiB free, g2 08:00.0 at 97+1=98 -> 298 MiB free, g4 05:00.0 Wayland at 49+0 -> 1,163 MiB sysfs free).
            constexpr int kShadowSeed[5] = {3, 2, 1, 1, 0};
            int seid = 319;
            int su[5] = {};
            for (int pass = 0; pass < 3 && seid < kNumRoutedExperts; ++pass) {
                for (int g = 0; g < 5 && seid < kNumRoutedExperts; ++g) {
                    if (su[g] >= kShadowSeed[g]) continue;
                    const double sp = 20480.0 * rank_freq[size_t(seid)];
                    cur_p[size_t(g)] += sp;
                    cur_t[size_t(g)] += std::ceil(sp / 64.0);
                    ++su[g];
                    ++seid;
                }
            }
        }
        for (int eid = 0; eid < tot_mi50; ++eid) {
            const double ep = 20480.0 * rank_freq[size_t(eid)];
            const double et = std::ceil(ep / 64.0);
            int best_g = -1;
            double best_score = 1e18;
            for (int g = 0; g < n_g; ++g) {
                const int rem = cap[size_t(g)] - used[size_t(g)];
                if (rem <= 0) continue;
                const double proj_t = cur_t[size_t(g)] + et + double(rem - 1);
                const double proj_p = cur_p[size_t(g)] + ep + double(rem - 1) * 20.0;
                // Cut A: projected TIME (not tile count) -> the 16 GiB parts and 05:00.0 receive
                // proportionally fewer experts so all 5 MI50s finish together.
                const double score = cost_w[size_t(g)] * (proj_t + proj_p / 65.0);
                if (score < best_score) { best_score = score; best_g = g; }
            }
            const int lslot = used[size_t(best_g)]++;
            cur_t[size_t(best_g)] += et;
            cur_p[size_t(best_g)] += ep;
            impl_->gpus[size_t(best_g)]->bb_global_ids.push_back(eid);
            impl_->bb_owner_gpu[size_t(eid)]  = int8_t(best_g);
            impl_->bb_local_slot[size_t(eid)] = int16_t(lslot);
        }
        std::printf("[ERNIE55-RT] Tile-Balanced 5-MI50 MoE plan (hist=%dL, cost_w=%.2f/%.2f/%.2f/%.2f/%.2f):", hist_layers,
                    cost_w[0], cost_w[1], cost_w[2], cost_w[3], cost_w[4]);
        for (int g = 0; g < n_g; ++g) {
            std::printf(" g%d=%de/%.0ft/%.0fp/%.0fms", g, cap[size_t(g)], cur_t[size_t(g)], cur_p[size_t(g)],
                        cost_w[size_t(g)] * (cur_t[size_t(g)] + cur_p[size_t(g)] / 65.0));
        }
        std::printf("\n");
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
        if (impl_->h_moe_mapped_slabs != nullptr) impl_->start_p150_relay();
        // Wave-10 C1: Register 4K-aligned SHM half buffers for zero-copy host-pinned SDMA directly to/from /dev/shm!
        if (hipHostRegister(impl_->p150_shm.ptr->x_rot_f16, sizeof(impl_->p150_shm.ptr->x_rot_f16),
                            hipHostRegisterPortable) == hipSuccess) {
            impl_->p150_shm_x_reg = true;
        }
        if (hipHostRegister(impl_->p150_shm.ptr->y_out_bf16, sizeof(impl_->p150_shm.ptr->y_out_bf16),
                            hipHostRegisterPortable) == hipSuccess) {
            impl_->p150_shm_y_reg = true;
        }
        std::printf("[ERNIE55-RT] Attached to Blackhole P150A SHM %s (mode=%s, cards=%u, experts=[%u..%u), L1=%.1f MiB, zcopy_dma=%d/%d)\n",
                    shm.c_str(), impl_->p150_shm.ptr->p150_mode == ERNIE55_P150_MODE_TP2 ? "TP2" : "EP",
                    impl_->p150_shm.ptr->num_p150_cards, impl_->p150_shm.ptr->expert_id_begin,
                    impl_->p150_shm.ptr->expert_id_end, double(impl_->p150_shm.ptr->l1_pinned_bytes_total) / 1048576.0,
                    int(impl_->p150_shm_x_reg), int(impl_->p150_shm_y_reg));

        // Post-swap Switchbox #2: Shadow-Cache top 7 hottest P150A GDDR6 experts [319..326) in Gen4 MI50 HBM2
        // (g1 88:00.0 at 2 -> 385 MiB free, g2 08:00.0 at 97+1=98 -> 298 MiB free, g4 05:00.0 Wayland at 49+0 -> >=1,160 MiB sysfs free)
        if (impl_->has_router_perm && impl_->gpus.size() == 5 && impl_->p150_exp_begin == 318) {
            const int gddr_start = impl_->p150_exp_begin + 1;
            constexpr int kExtraPerGpu[5] = {3, 2, 1, 1, 0}; // 7 hottest GDDR6 experts across Gen4 GPUs only
            int eid = gddr_start;
            int added = 0;
            for (int pass = 0; pass < 3 && eid < impl_->p150_exp_end; ++pass) {
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
                std::printf("[ERNIE55-RT] Wave-11 MI50 HBM2 shadow-cached %d hottest P150A GDDR6 experts [%d..%d) on L01..L42 (05:00.0 Wayland >=1.16 GiB sysfs free)\n",
                            added, gddr_start, gddr_start + added);
            }
        }
    }

    // Build branchless P150A backbone slot LUTs (canonical decode + Wave-11 C3 straggler-only GPU[2] prefill shadow offload!)
    impl_->p150_slot_lut_bb.fill(int16_t(-1));
    impl_->p150_slot_lut_bb_prefill.fill(int16_t(-1));
    if (impl_->p150_ok) {
        for (int eid = impl_->p150_exp_begin; eid < impl_->p150_exp_end && eid < kNumRoutedExperts; ++eid) {
            const int owner = int(impl_->bb_owner_gpu[size_t(eid)]);
            if (owner < 0) {
                impl_->p150_slot_lut_bb[size_t(eid)] = int16_t(eid - impl_->p150_exp_begin);
                impl_->p150_slot_lut_bb_prefill[size_t(eid)] = int16_t(eid - impl_->p150_exp_begin);
            } else if (impl_->is_prefill_shadow_offload_gpu(owner) &&
                       int(impl_->bb_local_slot[size_t(eid)]) >= impl_->gpus[size_t(owner)]->bb_base_count) {
                impl_->p150_slot_lut_bb_prefill[size_t(eid)] = int16_t(eid - impl_->p150_exp_begin);
            }
        }
    }

    // Upload per-GPU global->local expert lookup tables for Commit 2 on-GPU CSR builder
    for (size_t gi = 0; gi < impl_->gpus.size(); ++gi) {
        auto& w = *impl_->gpus[gi];
        std::array<int16_t, kNumRoutedExperts> h_g2l;
        h_g2l.fill(int16_t(-1));
        for (int eid = 0; eid < kNumRoutedExperts; ++eid) {
            if (int(impl_->bb_owner_gpu[size_t(eid)]) == int(gi)) {
                h_g2l[size_t(eid)] = impl_->bb_local_slot[size_t(eid)];
            }
        }
        hip_check(hipSetDevice(w.hip_dev), "upload g2l dev");
        hip_check(hipMemcpy(w.d_g2l_bb, h_g2l.data(), h_g2l.size() * sizeof(int16_t), hipMemcpyHostToDevice), "upload d_g2l_bb");
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

    // 5.5. Initialize 3-Tier 2,048-Block .kvblk Engine (100% pre-allocated pinned DDR4 LRU + Dual-NVMe O_DIRECT!)
    if (impl_->tp_size == 4) {
        std::vector<KvLayerDesc> klds(size_t(impl_->mcfg.num_hidden_layers));
        for (int l = 0; l < impl_->mcfg.num_hidden_layers; ++l) {
            const auto& L = impl_->layers[size_t(l)];
            auto& kd = klds[size_t(l)];
            kd.is_dsa    = L.is_dsa;
            kd.owner_dev = L.owner_dev;
            kd.tp_size   = L.tp_size;
            for (int r = 0; r < 4; ++r) {
                kd.d_hca_comp_kv_tp[r]  = L.d_hca_comp_kv_tp[r];
                kd.d_hca_swa_ring_tp[r] = L.d_hca_swa_ring_tp[r];
                kd.d_hca_state_kv_tp[r] = L.d_hca_state_kv_tp[r];
                kd.d_hca_state_sc_tp[r] = L.d_hca_state_sc_tp[r];
            }
            kd.d_dsa_mla_kv   = L.d_dsa_mla_kv;
            kd.d_dsa_ik_cache = L.d_dsa_ik_cache;
        }
        std::vector<uint16_t*> ds_rings;
        if (impl_->has_dspark) {
            for (const auto& DL : impl_->dspark_layers) {
                if (DL.d_swa_ring) ds_rings.push_back(DL.d_swa_ring);
            }
        }
        std::string s0_dir = "kvcache";
        try {
            const auto wr = std::filesystem::path(opt.weight_root);
            if (wr.has_parent_path() && wr.parent_path().has_parent_path()) {
                s0_dir = (wr.parent_path().parent_path() / "kvcache").string();
            }
        } catch (...) {}
        const char* ddr_b = std::getenv("ERNIE55_KV_DDR4_BLKS");
        const int max_ddr_blks = (ddr_b && ddr_b[0]) ? std::max(8, std::atoi(ddr_b)) : 256;
        uint8_t* tp_scratch[4] = {
            reinterpret_cast<uint8_t*>(impl_->dev_map[impl_->tp_devs[0]]->d_tmp_a),
            reinterpret_cast<uint8_t*>(impl_->dev_map[impl_->tp_devs[1]]->d_tmp_a),
            reinterpret_cast<uint8_t*>(impl_->dev_map[impl_->tp_devs[2]]->d_tmp_a),
            reinterpret_cast<uint8_t*>(impl_->dev_map[impl_->tp_devs[3]]->d_tmp_a)
        };
        impl_->kv_blk_cache.init(s0_dir, opt.kv_stripe_dir, impl_->spine_dev0, impl_->spine_dev1,
                                 impl_->split_layer, impl_->tp_devs, klds, ds_rings, max_ddr_blks, tp_scratch);
        std::printf("[ERNIE55-RT] 3-Tier 2048-Block .kvblk cache ready (DDR4_prealloc=%d slots, hbm2_slab=1, s0=%s, s1=%s)\n",
                    impl_->kv_blk_cache.capacity_slots(), s0_dir.c_str(),
                    opt.kv_stripe_dir.empty() ? s0_dir.c_str() : opt.kv_stripe_dir.c_str());
    }

    // Wave-11 Turn-11: Silent 1-chunk (M=max_chunk) self-warmup + full KV wipe BEFORE advertising READY.
    // Preloads every HIP code object, Tensile strided-batched solution, and HLUT4 kernel across all 5 MI50s
    // and Blackhole P150A (0 B net new GPU VRAM, 0 .kvblk entries written) so Cold-Boot Request #1 hits
    // full warm throughput and 100% verbatim dual-needle coherence on the very first request!
    if (impl_->max_chunk >= 64) {
        const char* skip_wu = std::getenv("ERNIE55_SKIP_BOOT_WARMUP");
        if (!(skip_wu && skip_wu[0] == '1' && skip_wu[1] == '\0')) {
            const auto tw0 = std::chrono::steady_clock::now();
            {
                Impl::DpmWarmGuard dpm_guard(impl_.get(), /*only_non_spine=*/false);
                std::vector<int32_t> wu_toks(size_t(impl_->max_chunk), int32_t(1));
                impl_->forward_backbone_chunk(wu_toks.data(), int(wu_toks.size()), 0, nullptr, true, false, false);
            }
            wipe_all_kv_vram();
            // Turn-12: the P150A SHM control block is deliberately NOT reset here. The
            // p150_ernie55_worker daemon is already attached with a non-zero local `last_seq`,
            // so storing 0 back into req_seq/resp_seq would make the worker observe 0 != last_seq
            // and execute a spurious rows == 0 command. The cold-boot Request #1 anomaly also
            // reproduces on builds without this self-warmup, so the warmup SHM state is not the
            // trigger and no SHM write is performed.
            const double wu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
            std::printf("[ERNIE55-RT] Cold-boot self-warmup (M=%d + wipe_all_kv_vram) complete in %.1fms\n",
                        impl_->max_chunk, wu_ms);
        }
    }

    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const char* dm = (impl_->draft_method == DraftMethod::kDSpark) ? "DSpark(probabilistic)"
                   : (impl_->draft_method == DraftMethod::kEcho ? "ECHO(2-step greedy)" : "none");
    std::printf("[ERNIE55-RT] READY in %.1fs | topo=%s | %zu MI50s + %s | max_kv=%d | vha_lr=%d/%d | draft=%s | 0 B DDR4 spill\n",
                load_s, impl_->topo.name.c_str(), impl_->gpus.size(),
                impl_->p150_ok ? "Blackhole P150A" : "no-P150", opt.max_kv,
                impl_->vha_lr_count, impl_->mcfg.num_hidden_layers, dm);
    std::fflush(stdout);
    impl_->is_init = true;
}

void Ernie55Runtime::shutdown() noexcept {
    if (!impl_) return;
    impl_->kv_blk_cache.shutdown();
    if (impl_->p150_ok && impl_->p150_shm.ptr && !impl_->opt.keep_p150) {
        auto* ch = impl_->p150_shm.ptr;
        ch->op = ERNIE55_P150_OP_SHUTDOWN;
        const uint32_t seq = ch->req_seq.load(std::memory_order_relaxed) + 1u;
        ch->req_seq.store(seq, std::memory_order_release);
        for (int i = 0; i < 500000 && ch->resp_seq.load(std::memory_order_acquire) != seq; ++i) p150_cpu_relax();
    }
    if (impl_->h_dspark_markov_reg && !impl_->h_dspark_markov_w1_T.empty()) {
        (void)hipHostUnregister(impl_->h_dspark_markov_w1_T.data());
        impl_->h_dspark_markov_reg = false;
        impl_->d_dspark_markov_w1_T_mapped = nullptr;
    }
    impl_->stop_p150_relay();
    if (impl_->p150_shm.ptr) {
        if (impl_->p150_shm_x_reg) { (void)hipHostUnregister(impl_->p150_shm.ptr->x_rot_f16); impl_->p150_shm_x_reg = false; }
        if (impl_->p150_shm_y_reg) { (void)hipHostUnregister(impl_->p150_shm.ptr->y_out_bf16); impl_->p150_shm_y_reg = false; }
    }
    impl_->p150_shm.close_mapping();
    impl_->p150_ok = false;

    for (auto& w : impl_->gpus) {
        if (w && w->h_dpm_flag) *w->h_dpm_flag = 0u;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (auto& w : impl_->gpus) {
        if (!w) continue;
        (void)hipSetDevice(w->hip_dev);
        if (w->keep_warm_stream) (void)hipStreamSynchronize(w->keep_warm_stream);
        if (w->stream)    (void)hipStreamSynchronize(w->stream);
        if (w->stream_sh) (void)hipStreamSynchronize(w->stream_sh);
        if (w->blas)      (void)rocblas_destroy_handle(w->blas);
        if (w->blas_sh)   (void)rocblas_destroy_handle(w->blas_sh);
        if (w->ev_sh_start) (void)hipEventDestroy(w->ev_sh_start);
        if (w->ev_sh_done)  (void)hipEventDestroy(w->ev_sh_done);
        for (int mb = 0; mb < 4; ++mb) {
            if (w->ev_mb_c2d[mb])     (void)hipEventDestroy(w->ev_mb_c2d[mb]);
            if (w->ev_mb_d2c[mb])     (void)hipEventDestroy(w->ev_mb_d2c[mb]);
            if (w->ev_mb_vha_d2h[mb]) (void)hipEventDestroy(w->ev_mb_vha_d2h[mb]);
            if (w->ev_mb_vha_h2d[mb]) (void)hipEventDestroy(w->ev_mb_vha_h2d[mb]);
            if (w->ev_mb_o_d2h[mb])   (void)hipEventDestroy(w->ev_mb_o_d2h[mb]);
        }
        if (w->ev_res_b_ready) (void)hipEventDestroy(w->ev_res_b_ready);
        for (void* p : w->allocs) if (p) (void)hipFree(p);
        w->allocs.clear();
        if (w->keep_warm_stream) (void)hipStreamDestroy(w->keep_warm_stream);
        if (w->stream_sh) (void)hipStreamDestroy(w->stream_sh);
        if (w->stream)    (void)hipStreamDestroy(w->stream);
        (void)hipDeviceSynchronize();
    }
    impl_->gpus.clear();
    impl_->dev_map.clear();
    for (const auto& pa : impl_->pin_allocs) {
        if (!pa.ptr) continue;
        if (pa.registered) { (void)hipHostUnregister(pa.ptr); std::free(pa.ptr); }
        else               { (void)hipHostFree(pa.ptr); }
    }
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

size_t Ernie55Runtime::live_prefill_fresh_tokens() const noexcept {
    if (!impl_ || !impl_->live_prefill_active.load(std::memory_order_acquire)) return 0u;
    const size_t base = impl_->live_prefill_base_fresh.load(std::memory_order_relaxed);
    const int m = impl_->live_prefill_chunk_m.load(std::memory_order_relaxed);
    const int ldone = impl_->live_prefill_layer_done.load(std::memory_order_relaxed);
    const int nl = std::max(1, impl_->mcfg.num_hidden_layers);
    if (m <= 0 || ldone <= 0) return base;
    // Clamp strictly below base + m until chunk finishes so final commit lands exact-to-the-token!
    const size_t est = size_t((double(m) * double(std::min(ldone, nl))) / double(nl) + 0.5);
    return base + std::min<size_t>(est, size_t(std::max(0, m - 1)));
}

void Ernie55Runtime::last_cache_restore_info(std::string& out_tier, double& out_ms) const {
    if (!impl_) { out_tier = "none"; out_ms = 0.0; return; }
    const int tid = impl_->last_ctx_tier_id.load(std::memory_order_acquire);
    out_tier = (tid == 3) ? "NVMe_kvblk" : (tid == 2) ? "DDR4" : (tid == 1) ? "VRAM" : "none";
    out_ms = impl_->last_ctx_read_ms;
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
    impl_->ngram_idx.reset();
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
    Impl::DpmWarmGuard dpm_guard(impl_.get(), /*only_non_spine=*/false);
    const auto t_pf0 = std::chrono::steady_clock::now();
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
        const auto tc0 = std::chrono::steady_clock::now();
        impl_->reset_chunk_timers();
        impl_->forward_backbone_chunk(
            tok_ids.data() + off, m, pos0,
            is_last ? &out_last_logits : nullptr,
            true, is_last, need_ds);
        if (need_ds) impl_->commit_dspark_context_kv(m, pos0);
        const double cms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
        if (cms > 0.0 && m >= 256) {
            const double c_tps = double(m) / (cms * 1e-3);
            impl_->last_chunk_prefill_tok_s.store(c_tps, std::memory_order_relaxed);
            std::printf("[ERNIE55-CHUNK] pos0=%d m=%d %.1fms %.1f tok/s (hca=%.0fms[a=%.0f,ar=%.0f] dsa=%.0fms[fi=%.0f,mqa=%.0f] moe=%.0fms[pre=%.0f,mi50=%.0f(g=%.0f/%.0f/%.0f/%.0f/%.0f),p150=%.0f,post=%.0f])\n",
                        pos0, m, cms, c_tps, impl_->chunk_hca_ms, impl_->chunk_hca_a_ms, impl_->chunk_hca_ar_ms,
                        impl_->chunk_dsa_ms, impl_->chunk_dsa_fi_ms, impl_->chunk_dsa_mqa_ms,
                        impl_->chunk_moe_ms, impl_->chunk_moe_pre_ms,
                        impl_->chunk_mi50_moe_ms,
                        impl_->chunk_gpu_moe_ms[0], impl_->chunk_gpu_moe_ms[1], impl_->chunk_gpu_moe_ms[2],
                        impl_->chunk_gpu_moe_ms[3], impl_->chunk_gpu_moe_ms[4],
                        impl_->chunk_p150_wait_ms, impl_->chunk_moe_post_ms);
            if (impl_->moe_pre_xray) {
                std::printf("[ERNIE55-MOE-PRE-XRAY] backlog=%.1f router=%.1f pack=%.1f fc1=%.1f sh=%.1f ms (pre_total=%.1f)\n",
                            impl_->chunk_pre_backlog_ms, impl_->chunk_pre_router_ms, impl_->chunk_pre_pack_ms,
                            impl_->chunk_pre_fc1_ms, impl_->chunk_pre_sh_ms, impl_->chunk_moe_pre_ms);
                std::printf("[ERNIE55-BACKLOG-XRAY] moe_post=%.1f attn_pre=%.1f qkv=%.1f attn=%.1f vha_op=%.1f attn_post=%.1f ms (sum=%.1f)\n",
                            impl_->chunk_bk_moe_post, impl_->chunk_bk_attn_pre, impl_->chunk_bk_qkv,
                            impl_->chunk_bk_attn, impl_->chunk_bk_vha_op, impl_->chunk_bk_attn_post,
                            impl_->chunk_bk_moe_post + impl_->chunk_bk_attn_pre + impl_->chunk_bk_qkv +
                            impl_->chunk_bk_attn + impl_->chunk_bk_vha_op + impl_->chunk_bk_attn_post);
                std::fflush(stdout);
            }
            std::fflush(stdout);
        }
        impl_->seq_len += m;
        impl_->cached_tokens.insert(impl_->cached_tokens.end(), tok_ids.begin() + off, tok_ids.begin() + off + m);
        off += m;
    }
    const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
    if (pf_ms > 0.0 && n > 0) {
        impl_->last_prefill_tok_s.store(double(n) / (pf_ms * 1e-3), std::memory_order_relaxed);
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
    double ctx_read_ms = 0.0;
    double ctx_read_tps = 0.0;
    int ctx_tier_id = 0;
    const auto t_rd0 = std::chrono::steady_clock::now();
    if (impl_->seq_len > 0 && int(impl_->cached_tokens.size()) == impl_->seq_len &&
        impl_->seq_len < target_n && compute_lcp(impl_->cached_tokens, tok_ids) >= impl_->seq_len) {
        // Exact full-end prefix hit!
        reused = impl_->seq_len;
        ctx_read_ms = std::max(1e-4, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rd0).count());
        ctx_read_tps = double(reused) / (ctx_read_ms * 1e-3);
        ctx_tier_id = 1;
    } else if (impl_->ckpt_prompt_len > 0 && int(impl_->ckpt_prompt_tokens.size()) == impl_->ckpt_prompt_len &&
               impl_->ckpt_prompt_len < target_n &&
               compute_lcp(impl_->ckpt_prompt_tokens, tok_ids) >= impl_->ckpt_prompt_len) {
        // Exact prompt-boundary snapshot hit (e.g., harness stripped prior <ernie:think> block!):
        // Restores exact HCA + DSpark rings in ~0.04 ms D2D without re-prefilling context!
        impl_->restore_hca_prompt_snapshot();
        reused = impl_->seq_len;
        ctx_read_ms = std::max(1e-4, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rd0).count());
        ctx_read_tps = double(reused) / (ctx_read_ms * 1e-3);
        ctx_tier_id = 1;
    } else {
        reset_kv();
        reused = 0;
    }

    // Tier 1 (Pre-allocated Pinned DDR4 LRU) + Tier 2 (Striped Dual-NVMe O_DIRECT .kvblk) Automatic Block Lookup!
    if (impl_->tp_size == 4 && impl_->kv_blk_cache.initialized() && (target_n - reused) >= Ernie55BlockKvCache::kBlockTokens) {
        hipStream_t tp_streams[4] = {
            impl_->dev_map[impl_->tp_devs[0]]->stream,
            impl_->dev_map[impl_->tp_devs[1]]->stream,
            impl_->dev_map[impl_->tp_devs[2]]->stream,
            impl_->dev_map[impl_->tp_devs[3]]->stream
        };
        KvReadTelemetry tel{};
        const int blk_reused = impl_->kv_blk_cache.restore_longest_prefix(tok_ids, reused, tp_streams, tel);
        if (blk_reused > reused) {
            const int added_toks = blk_reused - reused;
            impl_->seq_len = blk_reused;
            impl_->cached_tokens.assign(tok_ids.begin(), tok_ids.begin() + blk_reused);
            reused = blk_reused;
            ctx_read_ms  = tel.read_ms;
            ctx_read_tps = tel.read_tok_s;
            ctx_tier_id  = (tel.tier == "NVMe_kvblk") ? 3 : 2;
            std::printf("[ERNIE55-KVBLK] RESTORE tier=%s reused=%d (+%d tok / %d blk) %.2fms %.1fK tok/s\n",
                        tel.tier.c_str(), reused, added_toks, added_toks / Ernie55BlockKvCache::kBlockTokens,
                        tel.read_ms, tel.read_tok_s / 1000.0);
            std::fflush(stdout);
        }
    }

    impl_->last_ctx_read_tokens = reused;
    impl_->last_ctx_read_ms     = ctx_read_ms;
    impl_->last_ctx_read_tok_s  = ctx_read_tps;
    if (reused >= Ernie55BlockKvCache::kBlockTokens && ctx_read_tps > impl_->peak_ctx_read_tok_s) {
        impl_->peak_ctx_read_tok_s = ctx_read_tps;
    }
    impl_->last_ctx_tier_id.store(ctx_tier_id, std::memory_order_release);

    if (out_reused_prefix) *out_reused_prefix = reused;
    const size_t tot_n = tok_ids.size();
    if (impl_->prefill_progress_cb) {
        impl_->prefill_progress_cb(tot_n, size_t(reused), 0u, 0.0);
    }
    const auto run_chunked_cb = [&](const int32_t* ptr, int n) {
        if (n <= 0) return;
        Impl::DpmWarmGuard dpm_guard(impl_.get(), /*only_non_spine=*/false);
        const auto t_pf0 = std::chrono::steady_clock::now();
        const int csz = std::max(1, impl_->max_chunk);
        const int dspark_tail_start = std::max(0, n - kDsparkSwaRingSize);
        for (int off = 0; off < n; ) {
            const int32_t pos0 = impl_->seq_len;
            int m = std::min(csz, n - off);
            // Align unaligned start onto exact 2048-token block boundary so every subsequent 2048 chunk
            // lands on an exact k*2048 HCA/SWA block boundary for 100% memcmp == 0 block caching!
            if ((pos0 & 2047) != 0) {
                const int to_blk = 2048 - (pos0 & 2047);
                if (m > to_blk && (n - (off + to_blk)) >= 64) {
                    m = to_blk;
                }
            }
            const int rem_after = n - (off + m);
            if (rem_after > 0 && rem_after <= 8 && m >= 24) m -= 16;
            const bool is_last = (off + m == n);
            const bool need_ds = impl_->has_dspark && ((off + m) > dspark_tail_start);
            const auto tc0 = std::chrono::steady_clock::now();
            impl_->reset_chunk_timers();
            impl_->live_prefill_base_fresh.store(size_t(off), std::memory_order_relaxed);
            impl_->live_prefill_chunk_m.store(m, std::memory_order_relaxed);
            impl_->live_prefill_layer_done.store(0, std::memory_order_relaxed);
            impl_->live_prefill_active.store(1, std::memory_order_release);
            impl_->forward_backbone_chunk(
                ptr + off, m, pos0,
                is_last ? &out_last_logits : nullptr,
                true, is_last, need_ds);
            impl_->live_prefill_active.store(0, std::memory_order_release);
            if (need_ds) impl_->commit_dspark_context_kv(m, pos0);
            const double cms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
            if (cms > 0.0 && m >= 256) {
                const double c_tps = double(m) / (cms * 1e-3);
                impl_->last_chunk_prefill_tok_s.store(c_tps, std::memory_order_relaxed);
                std::printf("[ERNIE55-CHUNK] pos0=%d m=%d %.1fms %.1f tok/s (hca=%.0fms[a=%.0f,ar=%.0f] dsa=%.0fms[fi=%.0f,mqa=%.0f] moe=%.0fms[pre=%.0f,mi50=%.0f(g=%.0f/%.0f/%.0f/%.0f/%.0f),p150=%.0f,post=%.0f])\n",
                            pos0, m, cms, c_tps, impl_->chunk_hca_ms, impl_->chunk_hca_a_ms, impl_->chunk_hca_ar_ms,
                            impl_->chunk_dsa_ms, impl_->chunk_dsa_fi_ms, impl_->chunk_dsa_mqa_ms,
                            impl_->chunk_moe_ms, impl_->chunk_moe_pre_ms,
                            impl_->chunk_mi50_moe_ms,
                            impl_->chunk_gpu_moe_ms[0], impl_->chunk_gpu_moe_ms[1], impl_->chunk_gpu_moe_ms[2],
                            impl_->chunk_gpu_moe_ms[3], impl_->chunk_gpu_moe_ms[4],
                            impl_->chunk_p150_wait_ms, impl_->chunk_moe_post_ms);
                if (impl_->moe_pre_xray) {
                    std::printf("[ERNIE55-MOE-PRE-XRAY] backlog=%.1f router=%.1f pack=%.1f fc1=%.1f sh=%.1f ms (pre_total=%.1f)\n",
                                impl_->chunk_pre_backlog_ms, impl_->chunk_pre_router_ms, impl_->chunk_pre_pack_ms,
                                impl_->chunk_pre_fc1_ms, impl_->chunk_pre_sh_ms, impl_->chunk_moe_pre_ms);
                    std::printf("[ERNIE55-BACKLOG-XRAY] moe_post=%.1f attn_pre=%.1f qkv=%.1f attn=%.1f vha_op=%.1f attn_post=%.1f ms (sum=%.1f)\n",
                                impl_->chunk_bk_moe_post, impl_->chunk_bk_attn_pre,
                                impl_->chunk_bk_qkv, impl_->chunk_bk_attn,
                                impl_->chunk_bk_vha_op, impl_->chunk_bk_attn_post,
                                impl_->chunk_bk_moe_post + impl_->chunk_bk_attn_pre +
                                impl_->chunk_bk_qkv + impl_->chunk_bk_attn +
                                impl_->chunk_bk_vha_op + impl_->chunk_bk_attn_post);
                }
                std::fflush(stdout);
            }
            impl_->seq_len += m;
            impl_->cached_tokens.insert(impl_->cached_tokens.end(), ptr + off, ptr + off + m);
            off += m;
            // Snapshot ALL completed 2048-token blocks [0 .. seq_len/2048) (O(1) no-op if already in DDR4;
            // guarantees 100% gap-free contiguous chains even when decode crosses a 2048-token boundary!).
            if (impl_->tp_size == 4 && impl_->kv_blk_cache.initialized()) {
                const int b_last = impl_->seq_len / 2048;
                if (b_last > 0 && int(impl_->cached_tokens.size()) >= b_last * 2048) {
                    KvBlockKey kcur{};
                    for (int bi = 0; bi < b_last; ++bi) {
                        kcur = step_kv_block_hash(kcur, bi, impl_->cached_tokens.data() + size_t(bi) * 2048u);
                        const bool exact_end = ((bi + 1) * 2048 == impl_->seq_len);
                        impl_->kv_blk_cache.capture_block_from_vram(
                            kcur, exact_end, exact_end && need_ds,
                            impl_->dev_map[impl_->spine_dev0]->stream,
                            impl_->dev_map[impl_->spine_dev1]->stream);
                    }
                }
            }
            if (impl_->prefill_progress_cb) {
                impl_->prefill_progress_cb(tot_n, size_t(reused), size_t(off), cms);
            }
        }
        const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
        if (pf_ms > 0.0 && n > 0) {
            impl_->last_prefill_tok_s.store(double(n) / (pf_ms * 1e-3), std::memory_order_relaxed);
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
    ch->flags     = 0u;
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

    Impl::DpmWarmGuard dpm_guard(impl_.get(), /*only_non_spine=*/false);

    const bool use_dspark = (impl_->draft_method == DraftMethod::kDSpark &&
                             impl_->has_dspark &&
                             impl_->dspark_block_size > 0);
    const bool use_echo_2step = (!use_dspark &&
                                 impl_->draft_method == DraftMethod::kEcho &&
                                 impl_->has_echo_l43 &&
                                 sp.temperature <= 1e-6f &&
                                 impl_->opt.echo_draft_steps > 0);

    if (impl_->sa_cascade) {
        impl_->ngram_idx.sync_to(impl_->cached_tokens.data(), impl_->cached_tokens.size());
    }
    uint64_t req_sa_hits = 0, req_sa_drafted = 0, req_sa_accepted = 0;
    int sa_cooldown = 0;

    while (impl_->seq_len < impl_->opt.max_kv - 16) {
        // Strict 128-token HCA compression guard: pos0 + K <= 126 (mod 128) ALWAYS!
        // Ensures NO speculative pass ever triggers k_hca_compressor_reduce_norm_rope on a draft token,
        // and combined with the 256-slot HCA SWA ring guarantees 100% zero-leak rollback!
        const int hca_safe_k = (kHcaCompressRatio - 2) - (impl_->seq_len % kHcaCompressRatio);
        if (use_dspark && hca_safe_k >= 1 && (max_new_tokens - out_stats.generated_tokens) >= 2) {
            // Wave-11 Cut D3.4/D3.5: Adaptive K in [4..7] driven by 16-step EMA of DSpark acceptance rate
            int target_k = impl_->dspark_block_size;
            if (target_k >= 4) {
                const double a = impl_->dspark_alpha_ema;
                target_k = (a >= 0.62) ? 7 : ((a >= 0.48) ? 6 : ((a >= 0.36) ? 5 : 4));
            }
            const int B = std::min({target_k, hca_safe_k, max_new_tokens - out_stats.generated_tokens});
            // Wave-12 Cut D12.2: Precision-gated 8-gram(K=7)/6-gram(K=5) DSpark+SA 1M Suffix Cascade.
            // Probes [cached_tokens[seq_len-7..seq_len-1], cur_tok] in <2 us across up to 1M history.
            // High-precision 8-gram/6-gram keys + 2-step miss cooldown eliminate 600K-1M prose false hits
            // while capturing 100% of verbatim code/path/signature spans 50K-950K tokens back with exact
            // Dirac-delta draft_sparse q = delta_{sa_buf[i]} (100% lossless for greedy & T>0 verify!).
            DSparkHeadsCpu::DraftProposal prop;
            int sa_len = 0;
            bool used_sa = false;
            int32_t sa_buf[NgramSuffixIndex::kMaxK] = {0};
            const int k_sa = std::min({7, hca_safe_k, max_new_tokens - out_stats.generated_tokens});
            if (sa_cooldown > 0) {
                --sa_cooldown;
            } else if (impl_->sa_cascade && k_sa >= 4) {
                int matched_n = 0;
                sa_len = impl_->ngram_idx.probe_with_cur(cur_tok, k_sa, sa_buf, &matched_n);
                if (sa_len >= 4) {
                    used_sa = true;
                    prop.draft_tokens.reserve(size_t(sa_len));
                    prop.draft_sparse.resize(size_t(sa_len));
                    for (int i = 0; i < sa_len; ++i) {
                        prop.draft_tokens.push_back(sa_buf[i]);
                        prop.draft_sparse[size_t(i)].items.emplace_back(sa_buf[i], 1.0f);
                    }
                    ++impl_->sa_hits;
                    impl_->sa_drafted += uint64_t(sa_len);
                    ++impl_->dspark_skipped;
                    ++req_sa_hits;
                    req_sa_drafted += uint64_t(sa_len);
                }
            }
            if (!used_sa) {
                prop = impl_->run_dspark_draft_block(cur_tok, impl_->seq_len, B, sp, rng);
            }
            const int K = int(prop.draft_tokens.size());
            std::vector<int32_t> verify_in;
            verify_in.reserve(size_t(K + 1));
            verify_in.push_back(cur_tok);
            for (int32_t dt : prop.draft_tokens) verify_in.push_back(dt);

            const int32_t pos0 = impl_->seq_len;
            std::vector<SparseProbs> verify_sparse;
            impl_->forward_backbone_chunk(verify_in.data(), K + 1, pos0, nullptr, false, false, true, &verify_sparse, &sp);

            const auto t_sv0 = impl_->profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const auto vr = DSparkHeadsCpu::verify_speculative_from_target_sparse(
                prop.draft_tokens, prop.draft_sparse, verify_sparse, sp, rng);
            if (impl_->profiling_enabled) {
                impl_->prof.sample_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sv0).count();
            }
            if (used_sa) {
                impl_->sa_accepted += uint64_t(vr.n_draft_accepted);
                req_sa_accepted += uint64_t(vr.n_draft_accepted);
                sa_cooldown = (vr.n_draft_accepted <= 1) ? 2 : 0;
            } else if (K > 0) {
                const double step_alpha = double(vr.n_draft_accepted) / double(K);
                impl_->dspark_alpha_ema = (15.0 / 16.0) * impl_->dspark_alpha_ema + (1.0 / 16.0) * step_alpha;
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
            if (impl_->sa_cascade) impl_->ngram_idx.ingest(verify_in.data(), size_t(n_commit));
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
            impl_->ngram_idx.ingest(verify_in.data(), size_t(n_commit));
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
            if (impl_->sa_cascade) impl_->ngram_idx.push_token(cur_tok);
            logits_to_sparse_probs_fast(logits.data(), impl_->mcfg.vocab_size, sp.temperature, sp.top_k, sp.top_p, probs);
            cur_tok = sample_from_sparse_probs(probs, rng);
            if (!emit_tok(cur_tok)) break;
        }
    }

    if (req_sa_hits > 0) {
        std::printf("[ERNIE55-SA] hits=%llu drafted=%llu accepted=%llu (%.1f%%) dspark_skipped_tot=%llu\n",
                    (unsigned long long)req_sa_hits, (unsigned long long)req_sa_drafted,
                    (unsigned long long)req_sa_accepted,
                    req_sa_drafted > 0 ? 100.0 * double(req_sa_accepted) / double(req_sa_drafted) : 0.0,
                    (unsigned long long)impl_->dspark_skipped);
        std::fflush(stdout);
    }

    if (use_dspark) {
        auto& g1 = *impl_->dev_map[impl_->spine_dev1];
        hip_check(hipSetDevice(g1.hip_dev), "final dspark sync dev");
        hip_check(hipStreamSynchronize(g1.stream), "final dspark sync");
    }

    const auto t_end = std::chrono::steady_clock::now();
    out_stats.decode_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill_done).count();
    if (out_stats.decode_ms > 0.0 && out_stats.generated_tokens > 1) {
        out_stats.decode_tok_s = double(out_stats.generated_tokens - 1) / (out_stats.decode_ms * 1e-3);
        impl_->last_decode_tok_s.store(out_stats.decode_tok_s, std::memory_order_relaxed);
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
    impl_->ngram_idx.reset();
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
void Ernie55Runtime::flush_kvblk_cache() { if (impl_) impl_->kv_blk_cache.flush_async_writes(); }

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
    j.set("last_chunk_prefill_tok_s", JsonValue::make_double(impl_->last_chunk_prefill_tok_s.load(std::memory_order_relaxed)));
    j.set("last_prefill_tok_s", JsonValue::make_double(impl_->last_prefill_tok_s.load(std::memory_order_relaxed)));
    j.set("last_decode_tok_s", JsonValue::make_double(impl_->last_decode_tok_s.load(std::memory_order_relaxed)));
    const int tid = impl_->last_ctx_tier_id.load(std::memory_order_acquire);
    const char* tname = (tid == 1) ? "VRAM" : (tid == 2 ? "DDR4" : (tid == 3 ? "NVMe_kvblk" : "none"));
    j.set("ctx_read_tier", JsonValue::make_string(tname));
    j.set("ctx_read_tokens", JsonValue::make_int(impl_->last_ctx_read_tokens));
    j.set("ctx_read_ms", JsonValue::make_double(impl_->last_ctx_read_ms));
    j.set("ctx_read_tok_s", JsonValue::make_double(impl_->last_ctx_read_tok_s));
    j.set("peak_ctx_read_tok_s", JsonValue::make_double(impl_->peak_ctx_read_tok_s));
    j.set("kvblk_ddr4_slots", JsonValue::make_int(impl_->kv_blk_cache.capacity_slots()));
    j.set("kvblk_ddr4_used", JsonValue::make_int(impl_->kv_blk_cache.used_slots()));

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


// ---- Wave-10 C6: zero-copy Dfly2 shared-sparsity view (read-only live metadata & device handles) ----
Ernie55Runtime::Dfly2SharedSparsityView Ernie55Runtime::dfly2_shared_sparsity_view() const {
    Dfly2SharedSparsityView v;
    const Impl* im = impl_.get();
    if (im == nullptr || !im->is_init || im->gpus.empty()) return v; // valid == false
    v.tp_size          = im->tp_size;
    v.active_seq_len   = im->seq_len;
    v.max_kv           = im->opt.max_kv;
    v.spine_dev0       = im->spine_dev0;
    v.spine_dev1       = im->spine_dev1;
    v.has_dspark       = im->has_dspark;
    v.d_dspark_aux_cat = im->d_dspark_aux_cat;
    auto it_s1 = im->dev_map.find(im->spine_dev1);
    v.d_last_res_a = (it_s1 != im->dev_map.end() && it_s1->second != nullptr) ? it_s1->second->d_res_a : nullptr;

    const int n_layers = im->mcfg.num_hidden_layers;
    v.layers.reserve(size_t(n_layers));
    for (int l = 0; l < n_layers; ++l) {
        const auto& L = im->layers[size_t(l)];
        auto it_own = im->dev_map.find(L.owner_dev);
        const Mi50ExpertWorker* g_own = (it_own != im->dev_map.end()) ? it_own->second : nullptr;
        Dfly2LayerView lv;
        lv.layer_idx        = l;
        lv.owner_dev        = L.owner_dev;
        lv.is_dsa           = L.is_dsa;
        lv.seq_len          = im->seq_len;
        lv.max_seq_len      = im->opt.max_kv;
        lv.d_dsa_mla_kv     = L.d_dsa_mla_kv;
        lv.d_dsa_ik_cache   = L.d_dsa_ik_cache;
        lv.d_hca_comp_kv    = L.d_hca_comp_kv;
        lv.d_hca_swa_ring   = L.d_hca_swa_ring;
        lv.d_dspark_aux_cat = im->d_dspark_aux_cat;
        lv.d_res_a          = (g_own != nullptr) ? g_own->d_res_a : nullptr;
        if (L.is_dsa && g_own != nullptr) {
            lv.d_sel_idx  = g_own->d_sel_idx;
            lv.d_sel_cnts = g_own->d_sel_cnts;
            lv.sel_stride = kDsaIndexTopK + kSlidingWindow;
        }
        v.layers.push_back(lv);
    }
    v.valid = true;
    return v;
}

} // namespace ernie55
