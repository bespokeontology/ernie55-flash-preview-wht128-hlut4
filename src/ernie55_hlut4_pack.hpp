// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 100% Uniform WHT128-HLUT4 (4.125 bpw) Quantizer, FP64 TARQ RAREBAL-LDLQ,
// Exact ERNIE-5.5 Mathematical Reference Primitives (SiTU-GLU, mHC, MultiMax, YaRN/DSA RoPE, CRC32C),
// and Hardware Packers for AMD MI50 (gfx906 fdot2) & Tenstorrent Blackhole P150A (QUANT_MODE=1).
// Zero Python anywhere.

#include "ernie55_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ernie55 {

inline constexpr int kTileRows       = 32;
inline constexpr int kTileCols       = 32;
inline constexpr int kTileWeights    = 1024;
inline constexpr int kScaleGroups    = 8;     // 8 groups of 128 weights (4 rows x 32 cols)
inline constexpr int kGroupWeights   = 128;
inline constexpr int kSuperRows      = 128;
inline constexpr int kSuperCols      = 128;
inline constexpr int kSuperTiles     = 16;    // 4 x 4 subtiles

// ---- CRC32C (Castagnoli) ----
inline uint32_t crc32c_update(uint32_t crc, const void* data, size_t len) noexcept {
    static const std::array<uint32_t, 256> tbl = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    uint32_t c = ~crc;
    for (size_t i = 0; i < len; ++i) c = tbl[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return ~c;
}

// ---- Exact IEEE BF16, FP16, and FP8 E4M3FN conversions ----
inline uint16_t f32_to_bf16_rne(float x) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

inline float bf16_to_f32(uint16_t b) noexcept {
    uint32_t u = uint32_t(b) << 16;
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

inline uint16_t f32_to_fp16_rne(float x) noexcept {
    uint32_t f = 0;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = int32_t((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        const uint32_t half_bit = 1u << (shift - 1);
        const uint32_t mask = (1u << shift) - 1u;
        uint32_t h = mant >> shift;
        if ((mant & mask) > half_bit || ((mant & mask) == half_bit && (h & 1u))) ++h;
        return uint16_t(sign | h);
    }
    if (exp >= 31) return uint16_t(sign | 0x7BFFu);
    uint32_t h = uint32_t(exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        ++h;
        if ((h & 0x7C00u) == 0x7C00u) return uint16_t(sign | 0x7BFFu);
    }
    return uint16_t(sign | h);
}

inline float fp16_to_f32(uint16_t h) noexcept {
    const uint32_t sgn = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0u) {
        if (man == 0u) u = sgn;
        else {
            uint32_t m = man, e = 113u;
            while ((m & 0x0400u) == 0u) { m <<= 1; --e; }
            u = sgn | (e << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exp == 31u) {
        u = sgn | 0x7F800000u | (man << 13);
    } else {
        u = sgn | ((exp + 112u) << 23) | (man << 13);
    }
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

inline float fp8_e4m3fn_to_f32(uint8_t b) noexcept {
    const uint32_t sgn = uint32_t(b & 0x80u) << 24;
    const uint32_t exp = (b >> 3) & 0x0Fu;
    const uint32_t man = b & 0x07u;
    if (exp == 0u) {
        if (man == 0u) {
            float z = 0.0f;
            std::memcpy(&z, &sgn, 4);
            return z;
        }
        float v = std::ldexp(float(man) * 0.125f, -6);
        return sgn ? -v : v;
    }
    if (exp == 15u && man == 7u) return 0.0f;
    const uint32_t u = sgn | ((exp + 120u) << 23) | (man << 20);
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

inline float ue8m0_to_f32(uint8_t e) noexcept {
    if (e == 0u) return 0.0f;
    const uint32_t u = uint32_t(e) << 23;
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

// ---- Exact SiTU-GLU Activation (vLLM SituAndMul: beta=4.0, linear_beta=25.0) ----
inline float ernie55_situ_glu(float g, float u,
                              float beta = kSituBeta,
                              float linear_beta = kSituLinearBeta) noexcept {
    const float sig = 1.0f / (1.0f + std::exp(-g));
    const float gate = beta * std::tanh(g / beta) * sig;
    const float up = (linear_beta > 0.0f) ? (linear_beta * std::tanh(u / linear_beta)) : u;
    return gate * up;
}

// ---- Exact MultiMax Output Head Transform (bit-exact with vLLM-ERNIE model.py) ----
inline float ernie55_multimax_scalar(float x, const float r[4], const float t[4]) noexcept {
    if (std::isinf(x) && x < 0.0f) return x;
    auto relu = [](float v) noexcept -> float { return (v <= 0.0f) ? 0.0f : v; };
    const float a = relu(r[0] - x);
    const float b = relu(x - r[1]);
    const float c = relu(r[2] - x);
    const float d = relu(x - r[3]);
    float y = x + t[0] * a;
    y = y + t[1] * b;
    y = y + t[2] * (c * c);
    y = y + t[3] * (d * d);
    return y;
}

// ---- Deterministic WHT128 Sign Vector & Fast Orthonormal WHT128 ----
inline std::vector<uint64_t> make_wht_signs(int K, uint64_t seed) {
    const int nblk = K / kWhtBlock;
    std::vector<uint64_t> s(size_t(nblk) * 2u, 0ULL);
    uint64_t st = seed;
    auto splitmix64 = [&]() -> uint64_t {
        uint64_t z = (st += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (size_t i = 0; i < s.size(); ++i) s[i] = splitmix64();
    return s;
}

inline void wht128_block(float* v, const uint64_t* signs_2u64) noexcept {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    if (signs_2u64 != nullptr) {
        const uint64_t s0 = signs_2u64[0], s1 = signs_2u64[1];
        for (int i = 0; i < 64; ++i) if ((s0 >> i) & 1ULL) v[i] = -v[i];
        for (int i = 0; i < 64; ++i) if ((s1 >> i) & 1ULL) v[64 + i] = -v[64 + i];
    }
    for (int len = 1; len < 128; len <<= 1) {
        const int step = len << 1;
        for (int base = 0; base < 128; base += step) {
            for (int j = 0; j < len; ++j) {
                const float a = v[base + j];
                const float b = v[base + j + len];
                v[base + j] = a + b;
                v[base + j + len] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) v[i] *= kInvSqrt128;
}

inline void wht128_rows(const float* in, float* out, int rows, int K,
                        const std::vector<uint64_t>& signs) {
    const int nblk = K / 128;
    #pragma omp parallel for schedule(static) if(rows >= 16)
    for (int r = 0; r < rows; ++r) {
        const float* xr = in + size_t(r) * size_t(K);
        float* yr = out + size_t(r) * size_t(K);
        if (yr != xr) std::memcpy(yr, xr, size_t(K) * sizeof(float));
        for (int b = 0; b < nblk; ++b) wht128_block(yr + b * 128, &signs[size_t(b) * 2]);
    }
}

// ---- Exact Analytic FP64 16-Level Symmetric Gaussian Lloyd-Max Solver (HLUT4: 8 positive levels + sign) ----
struct LloydMax16Solution {
    std::array<double, 8> levels{};
    std::array<double, 7> thresholds{};
    double mse = 0.0;
};

inline const LloydMax16Solution& hlut4_lloyd_max() {
    static const LloydMax16Solution sol = []() {
        auto gpdf = [](double x) noexcept -> double {
            static const double kInvSqrt2Pi = 1.0 / std::sqrt(2.0 * std::acos(-1.0));
            return kInvSqrt2Pi * std::exp(-0.5 * x * x);
        };
        auto gcdf = [](double x) noexcept -> double { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
        auto pos_q = [&](double p) noexcept -> double {
            const double target = 0.5 * (1.0 + p);
            double lo = 0.0, hi = 1.0;
            while (gcdf(hi) < target) hi *= 2.0;
            for (int it = 0; it < 256; ++it) {
                double mid = 0.5 * (lo + hi);
                if (gcdf(mid) < target) lo = mid; else hi = mid;
                if (hi - lo <= 1e-15) break;
            }
            return 0.5 * (lo + hi);
        };
        LloydMax16Solution s;
        for (int i = 0; i < 8; ++i) s.levels[size_t(i)] = pos_q((double(i) + 0.5) / 8.0);
        for (int it = 0; it < 4000; ++it) {
            for (int i = 0; i < 7; ++i) s.thresholds[size_t(i)] = 0.5 * (s.levels[size_t(i)] + s.levels[size_t(i + 1)]);
            double max_delta = 0.0;
            for (int i = 0; i < 8; ++i) {
                const double a = (i == 0) ? 0.0 : s.thresholds[size_t(i - 1)];
                const bool inf_b = (i == 7);
                const double b = inf_b ? 1e308 : s.thresholds[size_t(i)];
                const double pb = inf_b ? 1.0 : gcdf(b);
                const double fb = inf_b ? 0.0 : gpdf(b);
                const double pa = gcdf(a);
                const double fa = gpdf(a);
                const double mass = 2.0 * (pb - pa);
                const double first = 2.0 * (fa - fb);
                const double c = (mass > 1e-30) ? (first / mass) : s.levels[size_t(i)];
                max_delta = std::max(max_delta, std::abs(c - s.levels[size_t(i)]));
                s.levels[size_t(i)] = c;
            }
            if (max_delta <= 1e-13) break;
        }
        for (int i = 0; i < 7; ++i) s.thresholds[size_t(i)] = 0.5 * (s.levels[size_t(i)] + s.levels[size_t(i + 1)]);
        double sq = 0.0;
        for (int i = 0; i < 8; ++i) {
            const double a = (i == 0) ? 0.0 : s.thresholds[size_t(i - 1)];
            const bool inf_b = (i == 7);
            const double b = inf_b ? 1e308 : s.thresholds[size_t(i)];
            const double pb = inf_b ? 1.0 : gcdf(b);
            const double pa = gcdf(a);
            const double mass = 2.0 * (pb - pa);
            sq += mass * s.levels[size_t(i)] * s.levels[size_t(i)];
        }
        s.mse = 1.0 - sq;
        return s;
    }();
    return sol;
}

inline std::array<float, 8> hlut4_levels_f32() noexcept {
    const auto& s = hlut4_lloyd_max();
    std::array<float, 8> out{};
    for (int i = 0; i < 8; ++i) out[size_t(i)] = float(s.levels[size_t(i)]);
    return out;
}

// ---- Exact 528 B 32x32 Tile & 8,448 B 128x128 Superblock (identical to Hy4-770B / Hy4-flash!) ----
struct alignas(16) Hlut4Tile32x32 {
    uint32_t q_nib4[128];    // 512 B: 4-bit signed nibble [sign<<3 | mag(0..7)], 4 words/row (8 cols/word)
    uint16_t scales_f16[8];  // 16 B:  FP16 RMS scale per 128-weight group (4 rows x 32 cols)
};
static_assert(sizeof(Hlut4Tile32x32) == kHlut4TileBytes, "Hlut4Tile32x32 must be 528 B");
static_assert(alignof(Hlut4Tile32x32) == 16, "Hlut4Tile32x32 alignment must be 16 B");

struct alignas(256) Hlut4Superblock128x128 {
    Hlut4Tile32x32 subtiles[kSuperTiles];  // nt_sub * 4 + kt_sub
};
static_assert(sizeof(Hlut4Superblock128x128) == kHlut4SuperBytes, "Hlut4Superblock128x128 must be 8,448 B");

inline uint32_t hlut4_quant_mag(double z, const LloydMax16Solution& sol) noexcept {
    if (z >= sol.thresholds[3]) {
        if (z >= sol.thresholds[5]) return (z >= sol.thresholds[6]) ? 7u : 6u;
        return (z >= sol.thresholds[4]) ? 5u : 4u;
    } else {
        if (z >= sol.thresholds[1]) return (z >= sol.thresholds[2]) ? 3u : 2u;
        return (z >= sol.thresholds[0]) ? 1u : 0u;
    }
}

inline void hlut4_set_nibble(Hlut4Tile32x32& tile, int r, int c, uint32_t mag, uint32_t sign) noexcept {
    const int widx = r * 4 + (c >> 3);
    const int sh = (c & 7) * 4;
    const uint32_t nib = ((sign & 1u) << 3) | (mag & 7u);
    tile.q_nib4[widx] = (tile.q_nib4[widx] & ~(0xFu << sh)) | (nib << sh);
}

inline float hlut4_dequant_elem(const Hlut4Tile32x32& tile, int r, int c, const float lv[8]) noexcept {
    const uint32_t nib = (tile.q_nib4[r * 4 + (c >> 3)] >> ((c & 7) * 4)) & 0xFu;
    const float v = lv[nib & 7u];
    const float s = fp16_to_f32(tile.scales_f16[r >> 2]);
    return ((nib & 8u) ? -v : v) * s;
}

// Quantize a [N, K] row-major FP32 matrix (after WHT128 rotation along K) to [N/32, K/32] Hlut4Tile32x32.
// Optional FP64 reverse-Cholesky LDLQ upper-triangular R_up [K, K] (row-major, unit diagonal: R_up[i*K+i]=1)
// applies exact RAREBAL-LDLQ error feedback across columns K-1 -> 0!
inline std::vector<Hlut4Tile32x32> quantize_wht128_hlut4(
    const float* W_rm, int N, int K,
    const std::vector<uint64_t>& signs,
    const double* R_up_ldlq = nullptr)
{
    const auto& sol = hlut4_lloyd_max();
    const int nt_cnt = N / 32;
    const int kt_cnt = K / 32;

    std::vector<float> W_rot(size_t(N) * size_t(K));
    wht128_rows(W_rm, W_rot.data(), N, K, signs);

    std::vector<Hlut4Tile32x32> tiles(size_t(nt_cnt) * size_t(kt_cnt));
    std::memset(tiles.data(), 0, tiles.size() * sizeof(Hlut4Tile32x32));

    #pragma omp parallel for schedule(static)
    for (int nt = 0; nt < nt_cnt; ++nt) {
        // Process 4 rows per scale group together so scale is exact per (4 rows x 32 cols = 128 weights)
        std::vector<double> row_buf(size_t(4) * size_t(K));
        for (int sg = 0; sg < 8; ++sg) {
            const int r0 = nt * 32 + sg * 4;
            for (int dr = 0; dr < 4; ++dr) {
                const float* src = W_rot.data() + size_t(r0 + dr) * size_t(K);
                double* dst = row_buf.data() + size_t(dr) * size_t(K);
                for (int c = 0; c < K; ++c) dst[c] = double(src[c]);
            }
            // Precompute exact per-tile 128-weight RMS scales (from rotated seed before LDLQ feedback)
            std::vector<double> tile_sig(size_t(kt_cnt), 0.0);
            for (int kt = 0; kt < kt_cnt; ++kt) {
                double ss = 0.0;
                for (int dr = 0; dr < 4; ++dr) {
                    const double* rw = row_buf.data() + size_t(dr) * size_t(K) + size_t(kt * 32);
                    for (int dc = 0; dc < 32; ++dc) ss += rw[dc] * rw[dc];
                }
                const float rms = float(std::sqrt(ss / 128.0));
                const uint16_t h_sig = f32_to_fp16_rne(rms);
                tiles[size_t(nt) * size_t(kt_cnt) + size_t(kt)].scales_f16[sg] = h_sig;
                tile_sig[size_t(kt)] = double(fp16_to_f32(h_sig));
            }

            if (R_up_ldlq == nullptr) {
                // H=I exact Lloyd-Max quantization
                for (int kt = 0; kt < kt_cnt; ++kt) {
                    Hlut4Tile32x32& tile = tiles[size_t(nt) * size_t(kt_cnt) + size_t(kt)];
                    const double sig = tile_sig[size_t(kt)];
                    const double inv_sig = (sig > 1e-12) ? (1.0 / sig) : 0.0;
                    for (int dr = 0; dr < 4; ++dr) {
                        const double* rw = row_buf.data() + size_t(dr) * size_t(K) + size_t(kt * 32);
                        for (int dc = 0; dc < 32; ++dc) {
                            const double v = rw[dc];
                            const uint32_t sign = (v < 0.0) ? 1u : 0u;
                            const uint32_t mag = (sig > 1e-12) ? hlut4_quant_mag(std::abs(v) * inv_sig, sol) : 0u;
                            hlut4_set_nibble(tile, sg * 4 + dr, dc, mag, sign);
                        }
                    }
                }
            } else {
                // Reverse-Cholesky FP64 LDLQ error feedback (k = K-1 down to 0)
                // R_up_ldlq has upper-triangular feedback coefficients: w_hat[k] = w[k] + sum_{j>k} (w[j]-q[j]) * R_up[k, j]
                std::vector<double> err(size_t(4) * size_t(K), 0.0);
                for (int k = K - 1; k >= 0; --k) {
                    const int kt = k >> 5;
                    const int dc = k & 31;
                    Hlut4Tile32x32& tile = tiles[size_t(nt) * size_t(kt_cnt) + size_t(kt)];
                    const double sig = tile_sig[size_t(kt)];
                    const double inv_sig = (sig > 1e-12) ? (1.0 / sig) : 0.0;
                    const double* rrow = R_up_ldlq + size_t(k) * size_t(K);
                    for (int dr = 0; dr < 4; ++dr) {
                        double v = row_buf[size_t(dr) * size_t(K) + size_t(k)];
                        const double* erow = err.data() + size_t(dr) * size_t(K);
                        double fb = 0.0;
                        for (int j = k + 1; j < K; ++j) fb += erow[j] * rrow[j];
                        v += fb;
                        const uint32_t sign = (v < 0.0) ? 1u : 0u;
                        const uint32_t mag = (sig > 1e-12) ? hlut4_quant_mag(std::abs(v) * inv_sig, sol) : 0u;
                        hlut4_set_nibble(tile, sg * 4 + dr, dc, mag, sign);
                        const double qv = (sig > 1e-12) ? ((sign ? -sol.levels[mag] : sol.levels[mag]) * sig) : 0.0;
                        err[size_t(dr) * size_t(K) + size_t(k)] = v - qv;
                    }
                }
            }
        }
    }
    return tiles;
}

// Build exact FP64 reverse-Cholesky LDLQ feedback matrix R_up [K, K] from Hessian H = X_rot^T X_rot / M + damp*I
// (where H is already rotated by WHT128!). Solves L D L^T = P H P (reversed order) -> returns R_up where
// R_up[k, j] = U[k, j] / U[k, k] for j > k (unit diagonal).
inline std::vector<double> build_fp64_ldlq_rup(const double* H_rot, int K, double damp_rel = 0.01) {
    // Reverse order P H P: A[i, j] = H_rot[(K-1-i)*K + (K-1-j)]
    std::vector<double> A(size_t(K) * size_t(K));
    double diag_mean = 0.0;
    for (int i = 0; i < K; ++i) diag_mean += H_rot[size_t(i) * size_t(K) + size_t(i)];
    diag_mean = (diag_mean > 0.0) ? (diag_mean / double(K)) : 1.0;
    const double damp = std::max(1e-8, damp_rel * diag_mean);

    for (int i = 0; i < K; ++i) {
        const int ri = K - 1 - i;
        for (int j = 0; j < K; ++j) {
            const int rj = K - 1 - j;
            A[size_t(i) * size_t(K) + size_t(j)] = H_rot[size_t(ri) * size_t(K) + size_t(rj)];
        }
        A[size_t(i) * size_t(K) + size_t(i)] += damp;
    }

    // Standard in-place Cholesky L L^T = A (lower triangle of A holds L)
    for (int j = 0; j < K; ++j) {
        double d = A[size_t(j) * size_t(K) + size_t(j)];
        for (int k = 0; k < j; ++k) {
            const double ljk = A[size_t(j) * size_t(K) + size_t(k)];
            d -= ljk * ljk;
        }
        if (d <= 1e-14) d = 1e-14;
        const double inv_s = 1.0 / std::sqrt(d);
        A[size_t(j) * size_t(K) + size_t(j)] = std::sqrt(d);
        #pragma omp parallel for schedule(static) if(K - j > 64)
        for (int i = j + 1; i < K; ++i) {
            double s = A[size_t(i) * size_t(K) + size_t(j)];
            const double* ai = A.data() + size_t(i) * size_t(K);
            const double* aj = A.data() + size_t(j) * size_t(K);
            for (int k = 0; k < j; ++k) s -= ai[k] * aj[k];
            A[size_t(i) * size_t(K) + size_t(j)] = s * inv_s;
        }
    }

    // Reverse back to upper-triangular R_up[k, j] = L[K-1-k, K-1-j] / L[K-1-k, K-1-k] for j > k
    std::vector<double> R_up(size_t(K) * size_t(K), 0.0);
    for (int k = 0; k < K; ++k) {
        const int rk = K - 1 - k;
        const double diag = A[size_t(rk) * size_t(K) + size_t(rk)];
        const double inv_d = (diag > 1e-14) ? (1.0 / diag) : 0.0;
        R_up[size_t(k) * size_t(K) + size_t(k)] = 1.0;
        for (int j = k + 1; j < K; ++j) {
            const int rj = K - 1 - j;
            R_up[size_t(k) * size_t(K) + size_t(j)] = A[size_t(rk) * size_t(K) + size_t(rj)] * inv_d;
        }
    }
    return R_up;
}

// Repack [N/32, K/32] MI50 Hlut4Tile32x32 layout into Blackhole P150A [N/128, K/128] Hlut4Superblock128x128
// (subtile order nt_sub * 4 + kt_sub, 8,448 B/superblock, exact 33 x 256 B NOC burst!)
inline std::vector<Hlut4Superblock128x128> pack_tiles_to_p150_superblocks(
    const Hlut4Tile32x32* tiles, int N, int K)
{
    const int nsb = N / 128;
    const int ksb = K / 128;
    const int kt_cnt = K / 32;
    std::vector<Hlut4Superblock128x128> out(size_t(nsb) * size_t(ksb));
    for (int ns = 0; ns < nsb; ++ns) {
        for (int ks = 0; ks < ksb; ++ks) {
            Hlut4Superblock128x128& sb = out[size_t(ns) * size_t(ksb) + size_t(ks)];
            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                    const int nt = ns * 4 + nt_sub;
                    const int kt = ks * 4 + kt_sub;
                    sb.subtiles[nt_sub * 4 + kt_sub] = tiles[size_t(nt) * size_t(kt_cnt) + size_t(kt)];
                }
            }
        }
    }
    return out;
}

// ---- Exact HCA DeepSeek-V4 YaRN & DSA Plain RoPE Inv-Freq Builders ----
inline std::vector<float> compute_dsa_inv_freq(int rope_dim = kDsaQkRopeHeadDim,
                                               double theta = kDsaRopeTheta) {
    const int half = rope_dim / 2;
    std::vector<float> inv(static_cast<size_t>(half), 0.0f);
    for (int i = 0; i < half; ++i) {
        inv[size_t(i)] = float(1.0 / std::pow(theta, double(2 * i) / double(rope_dim)));
    }
    return inv;
}

inline std::vector<float> compute_hca_yarn_inv_freq(
    int rope_dim = kHcaRopeHeadDim,
    double theta = kHcaCompressRopeTheta,
    double factor = kHcaYarnFactor,
    int orig_max_pos = kHcaYarnOrigMaxPos,
    double beta_fast = kHcaYarnBetaFast,
    double beta_slow = kHcaYarnBetaSlow)
{
    // Exact DeepSeek-V4 YaRN (deepseek_scaling_rope.py, mscale=0 -> no magnitude boost)
    const int half = rope_dim / 2;
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    auto find_corr_dim = [&](double num_rot) -> double {
        return double(rope_dim) * std::log(double(orig_max_pos) / (num_rot * kTwoPi)) / (2.0 * std::log(theta));
    };
    double low = std::floor(find_corr_dim(beta_fast));
    double high = std::ceil(find_corr_dim(beta_slow));
    low = std::max(low, 0.0);
    high = std::min(high, double(rope_dim - 1));
    if (std::abs(low - high) < 1e-6) high += 0.001;

    std::vector<float> inv(static_cast<size_t>(half), 0.0f);
    for (int i = 0; i < half; ++i) {
        const double pos_freq = std::pow(theta, double(2 * i) / double(rope_dim));
        const double inv_extrap = 1.0 / pos_freq;
        const double inv_interp = 1.0 / (factor * pos_freq);
        double ramp = (double(i) - low) / (high - low);
        ramp = std::max(0.0, std::min(1.0, ramp));
        const double extrap_mask = 1.0 - ramp;
        inv[size_t(i)] = float(inv_interp * (1.0 - extrap_mask) + inv_extrap * extrap_mask);
    }
    return inv;
}

// ---- Exact mHC Reference (matches vLLM-ERNIE test_mhc.py + tilelang_kernels.py bit-for-bit) ----
struct MhcPreOut {
    std::array<float, kHcMult> pre_mix{};
    std::array<float, kHcMult> post_mix{};
    std::array<float, kHcMult2> comb_mix{}; // [i * 4 + j]
    std::vector<float> layer_input;         // [H]
};

inline MhcPreOut ernie55_mhc_pre_ref(
    const float* residual_4xH,   // [4, H]
    const float* fn_24x16384,    // [24, 16384] (transposed from ckpt [16384, 24]!)
    const float hc_scale[3],     // [alpha_pre, alpha_post, alpha_res]
    const float hc_base[kHcMult3],
    const float* norm_w,         // optional [H] RMSNorm weight
    int H = kHiddenSize,
    float rms_eps = kRmsNormEps,
    float hc_eps = kHcEps,
    float post_alpha = kHcPostAlpha,
    int sinkhorn_iters = kHcSinkhornIters)
{
    const int hc_dim = kHcMult * H;
    double sq = 0.0;
    for (int i = 0; i < hc_dim; ++i) sq += double(residual_4xH[i]) * double(residual_4xH[i]);
    // ERNIE 5.5 mHC exact RMS factor: 1 / (sqrt(mean) + eps) -- eps OUTSIDE sqrt!
    const float rms = 1.0f / (std::sqrt(float(sq / double(hc_dim))) + rms_eps);

    std::array<float, kHcMult3> mixes{};
    for (int m = 0; m < kHcMult3; ++m) {
        double acc = 0.0;
        const float* row = fn_24x16384 + size_t(m) * size_t(hc_dim);
        for (int i = 0; i < hc_dim; ++i) acc += double(residual_4xH[i]) * double(row[i]);
        mixes[size_t(m)] = float(acc) * rms;
    }

    auto sigmoid = [](float x) noexcept -> float { return 1.0f / (1.0f + std::exp(-x)); };
    MhcPreOut out;
    for (int j = 0; j < kHcMult; ++j) {
        out.pre_mix[size_t(j)]  = sigmoid(mixes[size_t(j)] * hc_scale[0] + hc_base[j]) + hc_eps;
        out.post_mix[size_t(j)] = sigmoid(mixes[size_t(kHcMult + j)] * hc_scale[1] + hc_base[kHcMult + j]) * post_alpha;
    }

    // 4x4 Sinkhorn-Knopp (20 iterations, exact vLLM-ERNIE order)
    float cm[kHcMult][kHcMult];
    for (int i = 0; i < kHcMult; ++i) {
        float rmax = -1e30f;
        for (int j = 0; j < kHcMult; ++j) {
            cm[i][j] = mixes[size_t(2 * kHcMult + i * kHcMult + j)] * hc_scale[2] + hc_base[2 * kHcMult + i * kHcMult + j];
            if (cm[i][j] > rmax) rmax = cm[i][j];
        }
        float rsum = 0.0f;
        for (int j = 0; j < kHcMult; ++j) { cm[i][j] = std::exp(cm[i][j] - rmax); rsum += cm[i][j]; }
        for (int j = 0; j < kHcMult; ++j) cm[i][j] = cm[i][j] / rsum + hc_eps;
    }
    auto col_norm = [&]() {
        for (int j = 0; j < kHcMult; ++j) {
            float csum = 0.0f;
            for (int i = 0; i < kHcMult; ++i) csum += cm[i][j];
            const float inv = 1.0f / (csum + hc_eps);
            for (int i = 0; i < kHcMult; ++i) cm[i][j] *= inv;
        }
    };
    auto row_norm = [&]() {
        for (int i = 0; i < kHcMult; ++i) {
            float rsum = 0.0f;
            for (int j = 0; j < kHcMult; ++j) rsum += cm[i][j];
            const float inv = 1.0f / (rsum + hc_eps);
            for (int j = 0; j < kHcMult; ++j) cm[i][j] *= inv;
        }
    };
    col_norm();
    for (int it = 0; it < sinkhorn_iters - 1; ++it) {
        row_norm();
        col_norm();
    }
    for (int i = 0; i < kHcMult; ++i)
        for (int j = 0; j < kHcMult; ++j)
            out.comb_mix[size_t(i * kHcMult + j)] = cm[i][j];

    out.layer_input.assign(size_t(H), 0.0f);
    double lsq = 0.0;
    for (int d = 0; d < H; ++d) {
        float v = 0.0f;
        for (int s = 0; s < kHcMult; ++s) v += out.pre_mix[size_t(s)] * residual_4xH[size_t(s * H + d)];
        const float vb = bf16_to_f32(f32_to_bf16_rne(v));
        out.layer_input[size_t(d)] = vb;
        lsq += double(vb) * double(vb);
    }
    if (norm_w != nullptr) {
        const float inv_r = 1.0f / std::sqrt(float(lsq / double(H)) + rms_eps);
        for (int d = 0; d < H; ++d) out.layer_input[size_t(d)] = bf16_to_f32(f32_to_bf16_rne(out.layer_input[size_t(d)] * inv_r * norm_w[d]));
    }
    return out;
}

inline void ernie55_mhc_post_ref(
    const float* sublayer_x,        // [H]
    const float* residual_4xH,      // [4, H]
    const float post_mix[kHcMult],
    const float comb_mix[kHcMult2], // [i * 4 + j] -> einsum("ij,ih->jh")!
    float* residual_out_4xH,        // [4, H]
    int H = kHiddenSize) noexcept
{
    for (int j = 0; j < kHcMult; ++j) {
        for (int d = 0; d < H; ++d) {
            float acc = post_mix[j] * sublayer_x[d];
            for (int i = 0; i < kHcMult; ++i) acc += comb_mix[i * kHcMult + j] * residual_4xH[i * H + d];
            residual_out_4xH[j * H + d] = bf16_to_f32(f32_to_bf16_rne(acc));
        }
    }
}

inline void ernie55_mhc_head_ref(
    const float* residual_4xH,
    const float* hc_head_fn_4x16384, // [4, 16384] (transposed from ckpt [16384, 4]!)
    float hc_head_scale,
    const float hc_head_base[kHcMult],
    float* out_H,
    int H = kHiddenSize,
    float rms_eps = kRmsNormEps,
    float hc_eps = kHcEps) noexcept
{
    const int hc_dim = kHcMult * H;
    double sq = 0.0;
    for (int i = 0; i < hc_dim; ++i) sq += double(residual_4xH[i]) * double(residual_4xH[i]);
    // Upstream tilelang_kernels.py:1114 uses rsqrt(mean + rms_eps) for hc_head
    const float rsqrt_v = 1.0f / std::sqrt(float(sq / double(hc_dim)) + rms_eps);
    float w[kHcMult];
    for (int m = 0; m < kHcMult; ++m) {
        double dot = 0.0;
        const float* row = hc_head_fn_4x16384 + size_t(m) * size_t(hc_dim);
        for (int i = 0; i < hc_dim; ++i) dot += double(residual_4xH[i]) * double(row[i]);
        const float z = float(dot) * rsqrt_v * hc_head_scale + hc_head_base[m];
        w[m] = 1.0f / (1.0f + std::exp(-z)) + hc_eps;
    }
    for (int d = 0; d < H; ++d) {
        float acc = 0.0f;
        for (int m = 0; m < kHcMult; ++m) acc += w[m] * residual_4xH[m * H + d];
        out_H[d] = bf16_to_f32(f32_to_bf16_rne(acc));
    }
}

// ---- Universal On-Disk WHT128-HLUT4 Layer Shard Header (4096-byte page aligned) ----
// Shared identically by ernie55_build_weights, MI50 ernie55_runtime, and Blackhole p150_ernie55_worker.
// File layout:
//   [0 .. 4095]                     : Ernie55Hlut4LayerHeader
//   [4096 .. 4096 + E*3*4096*528)   : Expert e in [0..num_experts):
//                                       w1_gate [4096 Hlut4Tile32x32] (2048x2048)
//                                       w3_up   [4096 Hlut4Tile32x32] (2048x2048)
//                                       w2_down [4096 Hlut4Tile32x32] (2048x2048)
inline constexpr uint32_t kErnie55Hlut4ShardMagic   = 0x34483545u; // 'E5H4'
inline constexpr uint32_t kErnie55Hlut4ShardVersion = 1u;

struct alignas(4096) Ernie55Hlut4LayerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_idx;
    uint32_t num_experts;            // 512
    uint32_t latent_dim;             // 2048
    uint32_t moe_inter_dim;          // 2048
    uint32_t tiles_per_proj;         // 4096 (64 x 64)
    uint32_t ldlq_used;              // 1 = FP64 RAREBAL-LDLQ, 0 = identity
    uint64_t signs_in[kLatentDim / 64];     // 32 uint64_t (256 B)
    uint64_t signs_inter[kMoeInterSize / 64]; // 32 uint64_t (256 B)
    double   avg_sqnr_db;
    double   hessian_cond_in;
    double   hessian_cond_inter;
    double   rarebal_lambda;
    uint32_t header_crc32c;
    uint32_t payload_crc32c;
    uint8_t  pad[4096 - (8 * 4 + 64 * 8 + 4 * 8 + 2 * 4)];
};
static_assert(sizeof(Ernie55Hlut4LayerHeader) == 4096, "Ernie55Hlut4LayerHeader must be exactly 4096 B");

inline uint64_t ernie55_wht_seed_in(int layer_idx) noexcept {
    return 0x453535494E000000ULL ^ (uint64_t(layer_idx + 1) * 0x9E3779B97F4A7C15ULL);
}
inline uint64_t ernie55_wht_seed_inter(int layer_idx) noexcept {
    return 0x4535354D4F000000ULL ^ (uint64_t(layer_idx + 1) * 0xBF58476D1CE4E5B9ULL);
}

inline std::vector<uint64_t> make_ernie55_signs_in(int layer_idx) {
    return make_wht_signs(kLatentDim, ernie55_wht_seed_in(layer_idx));
}
inline std::vector<uint64_t> make_ernie55_signs_inter(int layer_idx) {
    return make_wht_signs(kMoeInterSize, ernie55_wht_seed_inter(layer_idx));
}

// Build TARQ RAREBAL-weighted FP64 Hessian H = X_rot^T diag(w) X_rot [K, K]
struct RarebalStats {
    int    m_cal = 0, m_common = 0, m_rare = 0;
    double tr_common = 0.0, tr_rare = 0.0, lambda_tarq = 1.0, cond_est = 1.0;
};

inline std::vector<double> build_rarebal_fp64_hessian(
    const float* x_rot, int M_cal, int K, bool enable_rarebal = true, RarebalStats* out_stats = nullptr)
{
    if (out_stats) *out_stats = RarebalStats{};
    std::vector<uint8_t> rare(size_t(M_cal), 0u);
    if (enable_rarebal && M_cal >= 8) {
        std::vector<double> mean_dir(size_t(K), 0.0);
        std::vector<float> inv_n(size_t(M_cal), 0.0f);
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot + size_t(m) * size_t(K);
            double n2 = 0.0;
            for (int k = 0; k < K; ++k) n2 += double(xm[k]) * double(xm[k]);
            const double inv = (n2 > 1e-30) ? (1.0 / std::sqrt(n2)) : 0.0;
            inv_n[size_t(m)] = float(inv);
            for (int k = 0; k < K; ++k) mean_dir[size_t(k)] += double(xm[k]) * inv;
        }
        double mn2 = 0.0;
        for (int k = 0; k < K; ++k) mn2 += mean_dir[size_t(k)] * mean_dir[size_t(k)];
        const double inv_mn = (mn2 > 1e-30) ? (1.0 / std::sqrt(mn2)) : 0.0;
        for (int k = 0; k < K; ++k) mean_dir[size_t(k)] *= inv_mn;
        std::vector<std::pair<float, int>> align(static_cast<size_t>(M_cal));
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot + size_t(m) * size_t(K);
            double dot = 0.0;
            for (int k = 0; k < K; ++k) dot += double(xm[k]) * mean_dir[size_t(k)];
            align[size_t(m)] = {float(std::fabs(dot) * double(inv_n[size_t(m)])), m};
        }
        const int rt = std::max(1, M_cal / 8);
        std::nth_element(align.begin(), align.begin() + rt, align.end());
        for (int i = 0; i < rt; ++i) rare[size_t(align[size_t(i)].second)] = 1u;
    }

    double tr_c = 0.0, tr_r = 0.0;
    int mc = 0, mr = 0;
    for (int m = 0; m < M_cal; ++m) {
        const float* xm = x_rot + size_t(m) * size_t(K);
        double sq = 0.0;
        for (int k = 0; k < K; ++k) sq += double(xm[k]) * double(xm[k]);
        if (enable_rarebal && rare[size_t(m)]) { tr_r += sq; ++mr; } else { tr_c += sq; ++mc; }
    }
    const double lam = (enable_rarebal && mc > 0 && mr > 0 && tr_c > 0.0 && tr_r > 0.0)
                     ? std::min(16.0, tr_c / std::max(1e-12, tr_r)) : 1.0;

    const double wc = 1.0 / double(std::max(1, M_cal));
    const double wr = lam * wc;
    std::vector<double> H(size_t(K) * size_t(K), 0.0);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < K; ++i) {
        double* hrow = H.data() + size_t(i) * size_t(K);
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot + size_t(m) * size_t(K);
            const double wi = (rare[size_t(m)] ? wr : wc) * double(xm[i]);
            for (int j = i; j < K; ++j) hrow[j] += wi * double(xm[j]);
        }
    }
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < i; ++j)
            H[size_t(i) * size_t(K) + size_t(j)] = H[size_t(j) * size_t(K) + size_t(i)];

    if (out_stats) {
        out_stats->m_cal = M_cal; out_stats->m_common = mc; out_stats->m_rare = mr;
        out_stats->tr_common = tr_c; out_stats->tr_rare = tr_r; out_stats->lambda_tarq = lam;
    }
    return H;
}

// Fast float32-feedback causal LDLQ (from exact FP64 reverse-Cholesky R_up_f32 [K, K] where
// R_up_f32[col*K + j] for j > col propagates quantization error forward; 4x faster than double scalar loop!)
inline std::vector<float> build_fp64_ldlq_rup_f32(const double* H_rot, int K, double damp_rel = 0.01, double* out_cond = nullptr) {
    // Reverse order P H P: A[i, j] = H_rot[(K-1-i)*K + (K-1-j)]
    double diag_mean = 0.0;
    for (int i = 0; i < K; ++i) diag_mean += H_rot[size_t(i) * size_t(K) + size_t(i)];
    diag_mean = (diag_mean > 0.0) ? (diag_mean / double(K)) : 1.0;

    std::vector<double> L(size_t(K) * size_t(K), 0.0);
    double damp = damp_rel * diag_mean;
    bool ok = false;
    for (int attempt = 0; attempt < 8 && !ok; ++attempt) {
        ok = true;
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j <= i; ++j) {
                double v = H_rot[size_t(K - 1 - i) * size_t(K) + size_t(K - 1 - j)];
                if (i == j) v += damp;
                L[size_t(i) * size_t(K) + size_t(j)] = v;
            }
        }
        for (int j = 0; j < K; ++j) {
            double diag = L[size_t(j) * size_t(K) + size_t(j)];
            for (int k = 0; k < j; ++k) {
                const double ljk = L[size_t(j) * size_t(K) + size_t(k)];
                diag -= ljk * ljk;
            }
            if (!(diag > 1e-18)) { ok = false; damp *= 4.0; break; }
            const double ljj = std::sqrt(diag);
            L[size_t(j) * size_t(K) + size_t(j)] = ljj;
            const double inv_ljj = 1.0 / ljj;
            #pragma omp parallel for schedule(static) if((K - j) > 128)
            for (int i = j + 1; i < K; ++i) {
                double s = L[size_t(i) * size_t(K) + size_t(j)];
                const double* li = L.data() + size_t(i) * size_t(K);
                const double* lj = L.data() + size_t(j) * size_t(K);
                for (int k = 0; k < j; ++k) s -= li[k] * lj[k];
                L[size_t(i) * size_t(K) + size_t(j)] = s * inv_ljj;
            }
        }
    }
    if (!ok) return {};
    if (out_cond) {
        double lmin = 1e300, lmax = 0.0;
        for (int i = 0; i < K; ++i) {
            const double d = L[size_t(i) * size_t(K) + size_t(i)];
            if (d < lmin) lmin = d;
            if (d > lmax) lmax = d;
        }
        const double r = (lmin > 1e-18) ? (lmax / lmin) : 1e18;
        *out_cond = r * r;
    }
    std::vector<float> R_up(size_t(K) * size_t(K), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int col = 0; col < K; ++col) {
        const int ir = K - 1 - col;
        float* rrow = R_up.data() + size_t(col) * size_t(K);
        rrow[col] = 1.0f;
        for (int k = col + 1; k < K; ++k) {
            const int jr = K - 1 - k;
            rrow[k] = float(L[size_t(ir) * size_t(K) + size_t(jr)] / L[size_t(jr) * size_t(K) + size_t(jr)]);
        }
    }
    return R_up;
}

inline std::vector<Hlut4Tile32x32> quantize_wht128_hlut4_fast(
    const float* W_rm, int N, int K,
    const std::vector<uint64_t>& signs,
    const std::vector<float>& R_up_f32,
    double* out_sqnr_db = nullptr)
{
    const auto& sol = hlut4_lloyd_max();
    const int nt_cnt = N / 32;
    const int kt_cnt = K / 32;
    const bool use_ldlq = (R_up_f32.size() == size_t(K) * size_t(K));

    std::vector<float> W_rot(size_t(N) * size_t(K));
    wht128_rows(W_rm, W_rot.data(), N, K, signs);

    std::vector<Hlut4Tile32x32> tiles(size_t(nt_cnt) * size_t(kt_cnt));
    std::memset(tiles.data(), 0, tiles.size() * sizeof(Hlut4Tile32x32));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nt = 0; nt < nt_cnt; ++nt) {
        for (int kt = 0; kt < kt_cnt; ++kt) {
            Hlut4Tile32x32& tile = tiles[size_t(nt) * size_t(kt_cnt) + size_t(kt)];
            for (int sg = 0; sg < 8; ++sg) {
                double sq = 0.0;
                for (int r = 4 * sg; r < 4 * sg + 4; ++r) {
                    const float* src = W_rot.data() + size_t(nt * 32 + r) * size_t(K) + size_t(kt * 32);
                    for (int c = 0; c < 32; ++c) sq += double(src[c]) * double(src[c]);
                }
                tile.scales_f16[sg] = f32_to_fp16_rne(float(std::sqrt(sq / 128.0 + 1e-30)));
            }
        }
    }

    double sig_sum = 0.0, err_sum = 0.0;
    #pragma omp parallel for reduction(+:sig_sum,err_sum) schedule(static)
    for (int r4 = 0; r4 < N; r4 += 4) {
        const int nt = r4 >> 5;
        const int lr0 = r4 & 31;
        const int sg = lr0 >> 2;
        const float* o0 = W_rot.data() + size_t(r4 + 0) * size_t(K);
        const float* o1 = W_rot.data() + size_t(r4 + 1) * size_t(K);
        const float* o2 = W_rot.data() + size_t(r4 + 2) * size_t(K);
        const float* o3 = W_rot.data() + size_t(r4 + 3) * size_t(K);

        std::vector<float> w0(o0, o0 + K), w1(o1, o1 + K), w2(o2, o2 + K), w3(o3, o3 + K);
        for (int col = 0; col < K; ++col) {
            Hlut4Tile32x32& tile = tiles[size_t(nt) * size_t(kt_cnt) + size_t(col >> 5)];
            const float sigma = fp16_to_f32(tile.scales_f16[sg]);
            const double inv_s = (sigma > 0.0f) ? (1.0 / double(sigma)) : 0.0;
            const int lc = col & 31;

            auto quant1 = [&](float wv, float orig, int local_r) -> float {
                const uint32_t mag = hlut4_quant_mag(double(std::fabs(wv)) * inv_s, sol);
                const uint32_t sgn = std::signbit(wv) ? 1u : 0u;
                hlut4_set_nibble(tile, local_r, lc, mag, sgn);
                const float qv = (sgn ? -float(sol.levels[mag]) : float(sol.levels[mag])) * sigma;
                const double d = double(orig) - double(qv);
                sig_sum += double(orig) * double(orig);
                err_sum += d * d;
                return orig - qv;
            };
            const float e0 = quant1(w0[size_t(col)], o0[col], lr0 + 0);
            const float e1 = quant1(w1[size_t(col)], o1[col], lr0 + 1);
            const float e2 = quant1(w2[size_t(col)], o2[col], lr0 + 2);
            const float e3 = quant1(w3[size_t(col)], o3[col], lr0 + 3);

            if (use_ldlq && col + 1 < K) {
                const float* __restrict__ rrow = R_up_f32.data() + size_t(col) * size_t(K);
                float* __restrict__ p0 = w0.data();
                float* __restrict__ p1 = w1.data();
                float* __restrict__ p2 = w2.data();
                float* __restrict__ p3 = w3.data();
                #pragma omp simd
                for (int k = col + 1; k < K; ++k) {
                    const float rv = rrow[k];
                    p0[k] += e0 * rv;
                    p1[k] += e1 * rv;
                    p2[k] += e2 * rv;
                    p3[k] += e3 * rv;
                }
            }
        }
    }
    if (out_sqnr_db) *out_sqnr_db = 10.0 * std::log10((sig_sum + 1e-30) / (err_sum + 1e-30));
    return tiles;
}

} // namespace ernie55
