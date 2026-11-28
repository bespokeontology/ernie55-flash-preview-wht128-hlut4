// SPDX-License-Identifier: Apache-2.0
// Pure gfx906 HIP Zero-Scratch (ScratchSize == 0) Packed v_dot2_f32_f16 (fdot2) WHT128-HLUT4 Kernels
// Zero VGPR/SGPR spills, zero device trig, zero slc/system RMW atomics.

#include "ernie55_hlut4_kernels.hpp"
#include <hip/hip_fp16.h>
#include <type_traits>

namespace ernie55 {

namespace {

typedef _Float16 f16x2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    return __uint_as_float(uint32_t(b) << 16);
}

__device__ __forceinline__ uint16_t d_f32_to_bf16(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

__device__ __forceinline__ float d_f16_to_f32(uint16_t h) {
    __half hv = *reinterpret_cast<const __half*>(&h);
    return __half2float(hv);
}

__device__ __forceinline__ uint16_t d_f32_to_f16(float x) {
    __half hv = __float2half(x);
    return *reinterpret_cast<const uint16_t*>(&hv);
}

__device__ __forceinline__ f16x2_t d_pack_f16x2(uint16_t a, uint16_t b) {
    uint32_t u = uint32_t(a) | (uint32_t(b) << 16);
    union { uint32_t u32; f16x2_t v; } cvt;
    cvt.u32 = u;
    return cvt.v;
}

__device__ __forceinline__ f16x2_t d_u32_to_f16x2(uint32_t u) {
    union { uint32_t u32; f16x2_t v; } cvt;
    cvt.u32 = u;
    return cvt.v;
}

// Overflow-free zero-scratch hardware tanh using single v_exp_f32 (0 libm calls, 0 scratch!)
__device__ __forceinline__ float d_tanh_hw(float x) {
    const float ax = fabsf(x);
    const float e = __expf(-2.0f * ax);
    const float t = (1.0f - e) / (1.0f + e);
    return (x < 0.0f) ? -t : t;
}

// Exact ERNIE-5.5 SiTU-GLU: beta * tanh(g / beta) * sigmoid(g) * (linear_beta * tanh(u / linear_beta))
__device__ __forceinline__ float d_situ_glu(float g, float u, float beta, float linear_beta) {
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float gate = beta * d_tanh_hw(g / beta) * sig;
    const float up = (linear_beta > 0.0f) ? (linear_beta * d_tanh_hw(u / linear_beta)) : u;
    return gate * up;
}

// ---- WHT128 Kernels (128 threads per 128-element block, 0 scratch) ----
__device__ __forceinline__ void d_wht128_shared(float* s_x, int tid, const uint64_t* signs_2u64) {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    if (signs_2u64 != nullptr) {
        const uint64_t sw = (tid < 64) ? signs_2u64[0] : signs_2u64[1];
        if ((sw >> (tid & 63)) & 1ULL) s_x[tid] = -s_x[tid];
    }
    __syncthreads();
    #pragma unroll
    for (int len = 1; len < 128; len <<= 1) {
        const float a = s_x[tid];
        const float b = s_x[tid ^ len];
        __syncthreads();
        s_x[tid] = (tid & len) ? (b - a) : (a + b);
        __syncthreads();
    }
    s_x[tid] *= kInvSqrt128;
}

__global__ __launch_bounds__(128, 8)
void k_wht128_bf16_to_f16(const uint16_t* __restrict__ in_bf16,
                          const uint64_t* __restrict__ signs,
                          uint16_t* __restrict__ out_f16, int K) {
    __shared__ float s_x[128];
    const int blk = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int idx = row * K + blk * 128 + tid;
    s_x[tid] = d_bf16_to_f32(in_bf16[idx]);
    d_wht128_shared(s_x, tid, signs ? (signs + blk * 2) : nullptr);
    out_f16[idx] = d_f32_to_f16(s_x[tid]);
}

// Wave-10 C2 v2: 1-pass WHT128 reading FP16 directly and emitting BOTH FP16 (MI50) + optional BF16 (P150A SHM, safe in-place!)
__global__ __launch_bounds__(128, 8)
void k_wht128_f16_to_f16_and_bf16(const uint16_t* in_f16,
                                  const uint64_t* __restrict__ signs,
                                  uint16_t* __restrict__ out_f16,
                                  uint16_t* out_bf16_or_null, int K) {
    __shared__ float s_x[128];
    const int blk = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int idx = row * K + blk * 128 + tid;
    s_x[tid] = d_f16_to_f32(in_f16[idx]);
    d_wht128_shared(s_x, tid, signs ? (signs + blk * 2) : nullptr);
    const float v = s_x[tid];
    out_f16[idx] = d_f32_to_f16(v);
    if (out_bf16_or_null != nullptr) out_bf16_or_null[idx] = d_f32_to_bf16(v);
}

__global__ __launch_bounds__(128, 8)
void k_wht128_f32_to_f16(const float* __restrict__ in_f32,
                         const uint64_t* __restrict__ signs,
                         uint16_t* __restrict__ out_f16, int K) {
    __shared__ float s_x[128];
    const int blk = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int idx = row * K + blk * 128 + tid;
    s_x[tid] = in_f32[idx];
    d_wht128_shared(s_x, tid, signs ? (signs + blk * 2) : nullptr);
    out_f16[idx] = d_f32_to_f16(s_x[tid]);
}

__global__ __launch_bounds__(128, 8)
void k_wht128_grouped_f32_to_f16(const float* __restrict__ in_f32,
                                 const ExpertDispatchGroup* __restrict__ groups,
                                 const uint64_t* __restrict__ signs,
                                 uint16_t* __restrict__ out_f16, int K, int max_c) {
    const int y = int(blockIdx.y);
    const int g_idx = y / max_c;
    const int c     = y - g_idx * max_c;
    if (c >= groups[g_idx].count) return;
    __shared__ float s_x[128];
    const int blk = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int row = g_idx * kMaxGroupTokens + c;
    const int idx = row * K + blk * 128 + tid;
    s_x[tid] = in_f32[idx];
    d_wht128_shared(s_x, tid, signs ? (signs + blk * 2) : nullptr);
    out_f16[idx] = d_f32_to_f16(s_x[tid]);
}

__global__ __launch_bounds__(128, 8)
void k_wht128_f32_to_f32(const float* __restrict__ in_f32,
                         const uint64_t* __restrict__ signs,
                         float* __restrict__ out_f32, int K) {
    __shared__ float s_x[128];
    const int blk = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int idx = row * K + blk * 128 + tid;
    s_x[tid] = in_f32[idx];
    d_wht128_shared(s_x, tid, signs ? (signs + blk * 2) : nullptr);
    out_f32[idx] = s_x[tid];
}

// ---- Packed v_dot2_f32_f16 (fdot2) WHT128-HLUT4 Single-Expert GEMV (ScratchSize == 0) ----
// Grid: (N/32), Block: 128 threads = 32 rows x 4 lanes/row (each lane handles 8 cols/tile = 4 fdot2 ops!)
__global__ __launch_bounds__(128, 4)
void k_hlut4_fdot2_gemv(const uint16_t* __restrict__ x_rot_f16,
                        const Hlut4Tile32x32* __restrict__ tiles,
                        const uint16_t* __restrict__ levels8_f16,
                        float* __restrict__ y_f32,
                        int N, int K) {
    __shared__ uint16_t s_lut[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_lut[tid]     = pos;
        s_lut[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();

    const int lr = tid >> 2;         // 0..31 local row inside 32x32 tile
    const int w4 = tid & 3;          // 0..3 word within row (8 cols/word)
    const int nt = int(blockIdx.x);
    const int row = nt * 32 + lr;
    if (row >= N) return;

    const int sg = lr >> 2;
    const int kt_cnt = K >> 5;
    const Hlut4Tile32x32* __restrict__ trow = tiles + size_t(nt) * size_t(kt_cnt);
    const uint32_t* __restrict__ x_u32 = reinterpret_cast<const uint32_t*>(x_rot_f16);

    float acc = 0.0f;
    for (int kt = 0; kt < kt_cnt; ++kt) {
        const Hlut4Tile32x32* __restrict__ t = trow + kt;
        const uint32_t w = t->q_nib4[lr * 4 + w4];
        const float sig = d_f16_to_f32(t->scales_f16[sg]);
        const uint32_t* __restrict__ xk = x_u32 + (kt * 16 + w4 * 4);

        float loc = 0.0f;
        const f16x2_t w0 = d_pack_f16x2(s_lut[w & 0xFu],         s_lut[(w >> 4)  & 0xFu]);
        const f16x2_t w1 = d_pack_f16x2(s_lut[(w >> 8)  & 0xFu], s_lut[(w >> 12) & 0xFu]);
        const f16x2_t w2 = d_pack_f16x2(s_lut[(w >> 16) & 0xFu], s_lut[(w >> 20) & 0xFu]);
        const f16x2_t w3 = d_pack_f16x2(s_lut[(w >> 24) & 0xFu], s_lut[(w >> 28) & 0xFu]);
        loc = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(xk[0]), loc, false);
        loc = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(xk[1]), loc, false);
        loc = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(xk[2]), loc, false);
        loc = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(xk[3]), loc, false);
        acc = fmaf(loc, sig, acc);
    }
    acc += __shfl_xor(acc, 1, 64);
    acc += __shfl_xor(acc, 2, 64);
    if (w4 == 0) y_f32[row] = acc;
}

__device__ __forceinline__ float d_reduce4_dpp(float v) {
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xf, 0xf, true)); // quad_perm:[1,0,3,2]
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xf, 0xf, true)); // quad_perm:[2,3,0,1]
    return v;
}

// ---- Grouped WHT128-HLUT4 fdot2 Gate(w1) + Up(w3) + SiTU-GLU (ScratchSize == 0) ----
// 32-bank 256-entry pair LUT (1 ds_read_b32 -> f16x2!) + dual gt/ut prefetch + fused LDS s_x reuse!
__global__ __launch_bounds__(128, 3)
void k_hlut4_fdot2_grouped_gu_situ(
    const uint16_t* __restrict__ x_rot_f16,
    const ExpertDispatchGroup* __restrict__ groups,
    const Hlut4Tile32x32* __restrict__ w1_tiles,
    const Hlut4Tile32x32* __restrict__ w3_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ z_f32,
    int N_inter, int K_in,
    float situ_beta, float situ_linear_beta)
{
    const int g_idx = int(blockIdx.y);
    const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
    const int expert = grp->expert;
    const int count  = grp->count;
    const int nt = int(blockIdx.x);
    if (expert < 0 || count <= 0 || nt * 32 >= N_inter) return;

    __shared__ uint16_t s_l16[16];
    __shared__ uint32_t s_lut256[256];
    __shared__ uint32_t s_x[2][8][16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_l16[tid]     = pos;
        s_l16[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();
    s_lut256[tid]       = uint32_t(s_l16[tid & 0xFu]) | (uint32_t(s_l16[tid >> 4]) << 16);
    s_lut256[tid + 128] = uint32_t(s_l16[(tid + 128) & 0xFu]) | (uint32_t(s_l16[(tid + 128) >> 4]) << 16);
    __syncthreads();

    const int lr = tid >> 2;
    const int w4 = tid & 3;
    const int n  = nt * 32 + lr;
    const int sg = lr >> 2;
    const int nt_cnt = N_inter >> 5;
    const int kt_cnt = K_in >> 5;
    const size_t tbase = size_t(expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const Hlut4Tile32x32* __restrict__ grow = w1_tiles + tbase;
    const Hlut4Tile32x32* __restrict__ urow = w3_tiles + tbase;
    const uint32_t* __restrict__ x_u32_base = reinterpret_cast<const uint32_t*>(x_rot_f16);
    const int k_u32 = K_in >> 1;
    const int tx = tid >> 4; // 0..7
    const int tk = tid & 15; // 0..15

    if (count == 1) {
        // Zero-barrier direct-L1 decode fast path with 32-bank s_lut256 + dual gt/ut load overlap!
        const uint32_t* __restrict__ x0 = x_u32_base + size_t(grp->rows[0]) * size_t(k_u32);
        float ag = 0.0f, au = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const int xoff = kt * 16 + w4 * 4;
            const uint32_t gw = grow[kt].q_nib4[tid];
            const uint32_t uw = urow[kt].q_nib4[tid];
            const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
            const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);
            const uint4 xv4 = reinterpret_cast<const uint4*>(x0 + xoff)[0];
            const f16x2_t xv0 = d_u32_to_f16x2(xv4.x);
            const f16x2_t xv1 = d_u32_to_f16x2(xv4.y);
            const f16x2_t xv2 = d_u32_to_f16x2(xv4.z);
            const f16x2_t xv3 = d_u32_to_f16x2(xv4.w);

            float lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[gw & 0xFFu]),         xv0, 0.0f, false);
            lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]), xv1, lg, false);
            lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]), xv2, lg, false);
            lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[gw >> 24]),           xv3, lg, false);
            ag = fmaf(lg, sig_g, ag);

            float lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[uw & 0xFFu]),         xv0, 0.0f, false);
            lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]), xv1, lu, false);
            lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]), xv2, lu, false);
            lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[uw >> 24]),           xv3, lu, false);
            au = fmaf(lu, sig_u, au);
        }
        ag = d_reduce4_dpp(ag);
        au = d_reduce4_dpp(au);
        if (w4 == 0) {
            z_f32[size_t(g_idx) * size_t(kMaxGroupTokens) * size_t(N_inter) + size_t(n)] =
                d_situ_glu(ag, au, situ_beta, situ_linear_beta);
        }
        return;
    }

    if (count == 2) {
        // Wave-8b Zero-barrier direct-L1 2-token fast path (avoids STEP=4 wasted fdot2 + 64 barriers!)
        const uint32_t* __restrict__ x0 = x_u32_base + size_t(grp->rows[0]) * size_t(k_u32);
        const uint32_t* __restrict__ x1 = x_u32_base + size_t(grp->rows[1]) * size_t(k_u32);
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const int xoff = kt * 16 + w4 * 4;
            const uint32_t gw = grow[kt].q_nib4[tid];
            const uint32_t uw = urow[kt].q_nib4[tid];
            const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
            const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);
            const f16x2_t g0 = d_u32_to_f16x2(s_lut256[gw & 0xFFu]);
            const f16x2_t g1 = d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]);
            const f16x2_t g2 = d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]);
            const f16x2_t g3 = d_u32_to_f16x2(s_lut256[gw >> 24]);
            const f16x2_t u0 = d_u32_to_f16x2(s_lut256[uw & 0xFFu]);
            const f16x2_t u1 = d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]);
            const f16x2_t u2 = d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]);
            const f16x2_t u3 = d_u32_to_f16x2(s_lut256[uw >> 24]);
            {
                const uint4 xv4 = reinterpret_cast<const uint4*>(x0 + xoff)[0];
                const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);
                float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                ag0 = fmaf(lg, sig_g, ag0);
                float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                au0 = fmaf(lu, sig_u, au0);
            }
            {
                const uint4 xv4 = reinterpret_cast<const uint4*>(x1 + xoff)[0];
                const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);
                float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                ag1 = fmaf(lg, sig_g, ag1);
                float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                au1 = fmaf(lu, sig_u, au1);
            }
        }
        ag0 = d_reduce4_dpp(ag0); au0 = d_reduce4_dpp(au0);
        ag1 = d_reduce4_dpp(ag1); au1 = d_reduce4_dpp(au1);
        if (w4 == 0) {
            const size_t zb = size_t(g_idx) * size_t(kMaxGroupTokens) * size_t(N_inter) + size_t(n);
            z_f32[zb]                   = d_situ_glu(ag0, au0, situ_beta, situ_linear_beta);
            z_f32[zb + size_t(N_inter)] = d_situ_glu(ag1, au1, situ_beta, situ_linear_beta);
        }
        return;
    }

    auto run_strip = [&](auto tag_step) {
        constexpr int STEP = decltype(tag_step)::value;
        for (int c0 = 0; c0 < count; c0 += STEP) {
            const uint32_t* __restrict__ x_load =
                (tx < STEP && (c0 + tx) < count) ? (x_u32_base + size_t(grp->rows[c0 + tx]) * size_t(k_u32)) : nullptr;
            float ag[STEP], au[STEP];
            #pragma unroll
            for (int c = 0; c < STEP; ++c) { ag[c] = 0.0f; au[c] = 0.0f; }

            if (x_load) s_x[0][tx][tk] = x_load[tk];
            __syncthreads();

            for (int kt = 0; kt < kt_cnt; ++kt) {
                const int cur = kt & 1;
                const int nxt = cur ^ 1;
                if (kt + 1 < kt_cnt && x_load) s_x[nxt][tx][tk] = x_load[(kt + 1) * 16 + tk];

                const uint32_t gw = grow[kt].q_nib4[tid];
                const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                const uint32_t uw = urow[kt].q_nib4[tid];
                const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);

                const f16x2_t g0 = d_u32_to_f16x2(s_lut256[gw & 0xFFu]);
                const f16x2_t g1 = d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]);
                const f16x2_t g2 = d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]);
                const f16x2_t g3 = d_u32_to_f16x2(s_lut256[gw >> 24]);

                const f16x2_t u0 = d_u32_to_f16x2(s_lut256[uw & 0xFFu]);
                const f16x2_t u1 = d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]);
                const f16x2_t u2 = d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]);
                const f16x2_t u3 = d_u32_to_f16x2(s_lut256[uw >> 24]);

                #pragma unroll
                for (int c = 0; c < STEP; ++c) {
                    const uint4 xv4 = reinterpret_cast<const uint4*>(&s_x[cur][c][w4 * 4])[0];
                    const f16x2_t xv0 = d_u32_to_f16x2(xv4.x);
                    const f16x2_t xv1 = d_u32_to_f16x2(xv4.y);
                    const f16x2_t xv2 = d_u32_to_f16x2(xv4.z);
                    const f16x2_t xv3 = d_u32_to_f16x2(xv4.w);

                    float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                    lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                    lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                    lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                    ag[c] = fmaf(lg, sig_g, ag[c]);

                    float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                    lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                    lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                    lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                    au[c] = fmaf(lu, sig_u, au[c]);
                }
                __syncthreads();
            }

            #pragma unroll
            for (int c = 0; c < STEP; ++c) {
                if (c0 + c < count) {
                    ag[c] = d_reduce4_dpp(ag[c]);
                    au[c] = d_reduce4_dpp(au[c]);
                    if (w4 == 0) {
                        z_f32[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + c)) * size_t(N_inter) + size_t(n)] =
                            d_situ_glu(ag[c], au[c], situ_beta, situ_linear_beta);
                    }
                }
            }
        }
    };

    if (count <= 4) run_strip(std::integral_constant<int, 4>{});
    else            run_strip(std::integral_constant<int, 8>{});
}

// ---- Grouped WHT128-HLUT4 fdot2 Down(w2) + Weighted Device-Local AtomicAdd (ScratchSize == 0) ----
__global__ __launch_bounds__(128, 4)
void k_hlut4_fdot2_grouped_down_atomic(
    const uint16_t* __restrict__ z_rot_f16,
    const ExpertDispatchGroup* __restrict__ groups,
    const Hlut4Tile32x32* __restrict__ w2_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ y_accum_f32,
    int N_out, int K_inter)
{
    __shared__ uint16_t s_lut[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_lut[tid]     = pos;
        s_lut[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();

    const int g_idx = int(blockIdx.y);
    const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
    const int expert = grp->expert;
    const int count  = grp->count;
    if (expert < 0 || count <= 0) return;

    const int lr = tid >> 2;
    const int w4 = tid & 3;
    const int nt = int(blockIdx.x);
    const int n  = nt * 32 + lr;
    if (n >= N_out) return;

    const int sg = lr >> 2;
    const int nt_cnt = N_out >> 5;
    const int kt_cnt = K_inter >> 5;
    const size_t tbase = size_t(expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const Hlut4Tile32x32* __restrict__ drow = w2_tiles + tbase;
    const uint32_t* __restrict__ z_u32_base = reinterpret_cast<const uint32_t*>(z_rot_f16);
    const int k_u32 = K_inter >> 1;

    for (int c0 = 0; c0 < count; c0 += 2) {
        const bool has1 = (c0 + 1 < count);
        const uint32_t* __restrict__ z0 = z_u32_base + (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0)) * size_t(k_u32);
        const uint32_t* __restrict__ z1 = has1 ? (z0 + k_u32) : z0;
        float a0 = 0.0f, a1 = 0.0f;

        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut4Tile32x32* __restrict__ dt = drow + kt;
            const uint32_t dw = dt->q_nib4[lr * 4 + w4];
            const float sig = d_f16_to_f32(dt->scales_f16[sg]);

            const f16x2_t w0 = d_pack_f16x2(s_lut[dw & 0xFu],         s_lut[(dw >> 4)  & 0xFu]);
            const f16x2_t w1 = d_pack_f16x2(s_lut[(dw >> 8)  & 0xFu], s_lut[(dw >> 12) & 0xFu]);
            const f16x2_t w2 = d_pack_f16x2(s_lut[(dw >> 16) & 0xFu], s_lut[(dw >> 20) & 0xFu]);
            const f16x2_t w3 = d_pack_f16x2(s_lut[(dw >> 24) & 0xFu], s_lut[(dw >> 28) & 0xFu]);

            const int zoff = kt * 16 + w4 * 4;
            float l0 = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(z0[zoff + 0]), 0.0f, false);
            l0 = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(z0[zoff + 1]), l0, false);
            l0 = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(z0[zoff + 2]), l0, false);
            l0 = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(z0[zoff + 3]), l0, false);
            a0 = fmaf(l0, sig, a0);

            if (has1) {
                float l1 = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(z1[zoff + 0]), 0.0f, false);
                l1 = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(z1[zoff + 1]), l1, false);
                l1 = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(z1[zoff + 2]), l1, false);
                l1 = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(z1[zoff + 3]), l1, false);
                a1 = fmaf(l1, sig, a1);
            }
        }
        a0 += __shfl_xor(a0, 1, 64); a0 += __shfl_xor(a0, 2, 64);
        if (w4 == 0) atomicAdd(&y_accum_f32[size_t(grp->rows[c0]) * size_t(N_out) + size_t(n)], a0 * grp->weights[c0]);
        if (has1) {
            a1 += __shfl_xor(a1, 1, 64); a1 += __shfl_xor(a1, 2, 64);
            if (w4 == 0) atomicAdd(&y_accum_f32[size_t(grp->rows[c0 + 1]) * size_t(N_out) + size_t(n)], a1 * grp->weights[c0 + 1]);
        }
    }
}

// ---- Deterministic 0-Atomic Grouped WHT128-HLUT4 fdot2 Down(w2) -> per-(row, slot) [M, 10, N_out] ----
// Zero-barrier direct-L1 (STEP=1 decode) + Double-buffered LDS + 32-bank s_lut256 (STEP=4/8 prefill): 0 spills!
__global__ __launch_bounds__(128, 4)
void k_hlut4_fdot2_grouped_down_slot(
    const uint16_t* __restrict__ z_rot_f16,
    const ExpertDispatchGroup* __restrict__ groups,
    const Hlut4Tile32x32* __restrict__ w2_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ y_slots_f32,
    int N_out, int K_inter)
{
    const int g_idx = int(blockIdx.y);
    const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
    const int expert = grp->expert;
    const int count  = grp->count;
    const int nt = int(blockIdx.x);
    if (expert < 0 || count <= 0 || nt * 32 >= N_out) return;

    __shared__ uint16_t s_l16[16];
    __shared__ uint32_t s_lut256[256];
    __shared__ uint32_t s_z[2][8][16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_l16[tid]     = pos;
        s_l16[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();
    s_lut256[tid]       = uint32_t(s_l16[tid & 0xFu]) | (uint32_t(s_l16[tid >> 4]) << 16);
    s_lut256[tid + 128] = uint32_t(s_l16[(tid + 128) & 0xFu]) | (uint32_t(s_l16[(tid + 128) >> 4]) << 16);
    __syncthreads();

    const int lr = tid >> 2;
    const int w4 = tid & 3;
    const int n  = nt * 32 + lr;
    const int sg = lr >> 2;
    const int nt_cnt = N_out >> 5;
    const int kt_cnt = K_inter >> 5;
    const size_t tbase = size_t(expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const Hlut4Tile32x32* __restrict__ drow = w2_tiles + tbase;
    const int k_u32 = K_inter >> 1;
    const uint32_t* __restrict__ z_grp = reinterpret_cast<const uint32_t*>(z_rot_f16) +
                                         size_t(g_idx) * size_t(kMaxGroupTokens) * size_t(k_u32);
    const int tx = tid >> 4; // 0..7
    const int tk = tid & 15; // 0..15

    if (count == 1) {
        // Zero-barrier direct-L1 decode fast path with 32-bank s_lut256 + uint4 vector load!
        float a = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const uint32_t dw = drow[kt].q_nib4[tid];
            const float sig = d_f16_to_f32(drow[kt].scales_f16[sg]);
            const uint4 zv4 = reinterpret_cast<const uint4*>(z_grp + kt * 16 + w4 * 4)[0];
            float lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[dw & 0xFFu]),         d_u32_to_f16x2(zv4.x), 0.0f, false);
            lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(dw >> 8)  & 0xFFu]), d_u32_to_f16x2(zv4.y), lc, false);
            lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(dw >> 16) & 0xFFu]), d_u32_to_f16x2(zv4.z), lc, false);
            lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[dw >> 24]),           d_u32_to_f16x2(zv4.w), lc, false);
            a = fmaf(lc, sig, a);
        }
        a = d_reduce4_dpp(a);
        if (w4 == 0) {
            y_slots_f32[(size_t(grp->rows[0]) * 10u + size_t(grp->slots[0])) * size_t(N_out) + size_t(n)] =
                a * grp->weights[0];
        }
        return;
    }

    if (count == 2) {
        // Wave-8b Zero-barrier direct-L1 2-token fast path (avoids STEP=4 wasted fdot2 + 64 barriers!)
        const uint32_t* __restrict__ z0 = z_grp;
        const uint32_t* __restrict__ z1 = z_grp + size_t(k_u32);
        float a0 = 0.0f, a1 = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const int zoff = kt * 16 + w4 * 4;
            const uint32_t dw = drow[kt].q_nib4[tid];
            const float sig = d_f16_to_f32(drow[kt].scales_f16[sg]);
            const f16x2_t w0 = d_u32_to_f16x2(s_lut256[dw & 0xFFu]);
            const f16x2_t w1 = d_u32_to_f16x2(s_lut256[(dw >> 8)  & 0xFFu]);
            const f16x2_t w2 = d_u32_to_f16x2(s_lut256[(dw >> 16) & 0xFFu]);
            const f16x2_t w3 = d_u32_to_f16x2(s_lut256[dw >> 24]);
            const uint4 zv0 = reinterpret_cast<const uint4*>(z0 + zoff)[0];
            float lc0 = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(zv0.x), 0.0f, false);
            lc0 = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(zv0.y), lc0, false);
            lc0 = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(zv0.z), lc0, false);
            lc0 = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(zv0.w), lc0, false);
            a0 = fmaf(lc0, sig, a0);
            const uint4 zv1 = reinterpret_cast<const uint4*>(z1 + zoff)[0];
            float lc1 = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(zv1.x), 0.0f, false);
            lc1 = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(zv1.y), lc1, false);
            lc1 = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(zv1.z), lc1, false);
            lc1 = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(zv1.w), lc1, false);
            a1 = fmaf(lc1, sig, a1);
        }
        a0 = d_reduce4_dpp(a0);
        a1 = d_reduce4_dpp(a1);
        if (w4 == 0) {
            y_slots_f32[(size_t(grp->rows[0]) * 10u + size_t(grp->slots[0])) * size_t(N_out) + size_t(n)] =
                a0 * grp->weights[0];
            y_slots_f32[(size_t(grp->rows[1]) * 10u + size_t(grp->slots[1])) * size_t(N_out) + size_t(n)] =
                a1 * grp->weights[1];
        }
        return;
    }

    auto run_strip = [&](auto tag_step) {
        constexpr int STEP = decltype(tag_step)::value;
        for (int c0 = 0; c0 < count; c0 += STEP) {
            const uint32_t* __restrict__ z_lo = (tx < STEP && (c0 + tx) < count) ? (z_grp + size_t(c0 + tx) * size_t(k_u32)) : nullptr;
            float a[STEP];
            #pragma unroll
            for (int c = 0; c < STEP; ++c) a[c] = 0.0f;

            if (z_lo) s_z[0][tx][tk] = z_lo[tk];
            __syncthreads();

            for (int kt = 0; kt < kt_cnt; ++kt) {
                const int cur = kt & 1;
                const int nxt = cur ^ 1;
                if (kt + 1 < kt_cnt && z_lo) s_z[nxt][tx][tk] = z_lo[(kt + 1) * 16 + tk];

                const uint32_t dw = drow[kt].q_nib4[tid];
                const float sig = d_f16_to_f32(drow[kt].scales_f16[sg]);
                const f16x2_t w0 = d_u32_to_f16x2(s_lut256[dw & 0xFFu]);
                const f16x2_t w1 = d_u32_to_f16x2(s_lut256[(dw >> 8)  & 0xFFu]);
                const f16x2_t w2 = d_u32_to_f16x2(s_lut256[(dw >> 16) & 0xFFu]);
                const f16x2_t w3 = d_u32_to_f16x2(s_lut256[dw >> 24]);

                #pragma unroll
                for (int c = 0; c < STEP; ++c) {
                    const uint4 zv4 = reinterpret_cast<const uint4*>(&s_z[cur][c][w4 * 4])[0];
                    float lc = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(zv4.x), 0.0f, false);
                    lc = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(zv4.y), lc, false);
                    lc = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(zv4.z), lc, false);
                    lc = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(zv4.w), lc, false);
                    a[c] = fmaf(lc, sig, a[c]);
                }
                __syncthreads();
            }

            #pragma unroll
            for (int c = 0; c < STEP; ++c) {
                if (c0 + c < count) {
                    a[c] = d_reduce4_dpp(a[c]);
                    if (w4 == 0) {
                        y_slots_f32[(size_t(grp->rows[c0 + c]) * 10u + size_t(grp->slots[c0 + c])) * size_t(N_out) + size_t(n)] =
                            a[c] * grp->weights[c0 + c];
                    }
                }
            }
        }
    };

    if (count <= 4) run_strip(std::integral_constant<int, 4>{});
    else            run_strip(std::integral_constant<int, 8>{});
}

// ---- Unpack WHT128-HLUT4 [N, K] -> FP16 Row-Major for rocBLAS Prefill Crossover ----
__global__ __launch_bounds__(256, 8)
void k_unpack_hlut4_to_f16(const Hlut4Tile32x32* __restrict__ tiles,
                           const uint16_t* __restrict__ levels8_f16,
                           uint16_t* __restrict__ out_f16, int N, int K) {
    __shared__ float s_lut[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const float lv = d_f16_to_f32(levels8_f16[tid]);
        s_lut[tid]     = lv;
        s_lut[tid + 8] = -lv;
    }
    __syncthreads();

    const int kt = int(blockIdx.x);
    const int nt = int(blockIdx.y);
    const int kt_cnt = K >> 5;
    const Hlut4Tile32x32* __restrict__ t = tiles + size_t(nt) * size_t(kt_cnt) + size_t(kt);

    // 256 threads x 4 cols = 1024 elems (32x32 tile)
    const int lr = tid >> 3;         // 0..31
    const int c4 = (tid & 7) * 4;    // 0,4,8,...,28
    const int row = nt * 32 + lr;
    if (row >= N) return;
    const float sig = d_f16_to_f32(t->scales_f16[lr >> 2]);
    const uint32_t w = t->q_nib4[lr * 4 + (c4 >> 3)];
    const int sh0 = (c4 & 7) * 4;
    uint16_t* __restrict__ dst = out_f16 + size_t(row) * size_t(K) + size_t(kt * 32 + c4);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        dst[j] = d_f32_to_f16(s_lut[(w >> (sh0 + 4 * j)) & 0xFu] * sig);
    }
}

// ---- Dense / Shared-Expert SiTU-GLU on BF16 ----
__global__ __launch_bounds__(256, 8)
void k_situ_glu_bf16(const uint16_t* __restrict__ gate_bf16,
                     const uint16_t* __restrict__ up_bf16,
                     uint16_t* __restrict__ out_bf16,
                     int total, float beta, float linear_beta) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) {
        const float g = d_bf16_to_f32(gate_bf16[i]);
        const float u = d_bf16_to_f32(up_bf16[i]);
        out_bf16[i] = d_f32_to_bf16(d_situ_glu(g, u, beta, linear_beta));
    }
}

__global__ __launch_bounds__(256, 8)
void k_situ_glu_f16(const uint16_t* gate_f16,
                    const uint16_t* __restrict__ up_f16,
                    uint16_t* out_f16,
                    int total, float beta, float linear_beta) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) {
        const float g = d_f16_to_f32(gate_f16[i]);
        const float u = d_f16_to_f32(up_f16[i]);
        out_f16[i] = d_f32_to_f16(d_situ_glu(g, u, beta, linear_beta));
    }
}

__global__ __launch_bounds__(256, 8)
void k_situ_glu_f32(const float* __restrict__ gate_f32,
                    const float* __restrict__ up_f32,
                    float* __restrict__ out_f32,
                    int total, float beta, float linear_beta) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) {
        out_f32[i] = d_situ_glu(gate_f32[i], up_f32[i], beta, linear_beta);
    }
}

// ---- Fast Zero-Scratch Weight-Stationary uint4-Vectorized BF16 Row-Major GEMV ----
// Loads each weight row ONCE from HBM2 via 128-bit uint4 transactions (8 BF16/lane = 512 BF16/iter/warp)
// and computes up to M=4 tokens simultaneously in registers! Cuts ECHO M=3 HBM2 traffic by 3.0x!
__device__ __forceinline__ float dot8_bf16_u4(uint4 a, uint4 b) {
    float s = 0.0f;
    s = fmaf(d_bf16_to_f32(uint16_t(a.x & 0xFFFFu)), d_bf16_to_f32(uint16_t(b.x & 0xFFFFu)), s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.x >> 16)),     d_bf16_to_f32(uint16_t(b.x >> 16)),     s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.y & 0xFFFFu)), d_bf16_to_f32(uint16_t(b.y & 0xFFFFu)), s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.y >> 16)),     d_bf16_to_f32(uint16_t(b.y >> 16)),     s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.z & 0xFFFFu)), d_bf16_to_f32(uint16_t(b.z & 0xFFFFu)), s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.z >> 16)),     d_bf16_to_f32(uint16_t(b.z >> 16)),     s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.w & 0xFFFFu)), d_bf16_to_f32(uint16_t(b.w & 0xFFFFu)), s);
    s = fmaf(d_bf16_to_f32(uint16_t(a.w >> 16)),     d_bf16_to_f32(uint16_t(b.w >> 16)),     s);
    return s;
}

template <int M_STATIC, bool OUT_F32>
__global__ __launch_bounds__(256, 4)
void k_bf16_gemv_ws(const uint16_t* __restrict__ x_bf16,
                    const uint16_t* __restrict__ w_bf16,
                    void* __restrict__ y_out,
                    int N, int K) {
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int n = int(blockIdx.x) * 4 + warp;
    if (n >= N) return;

    const int K8 = K >> 3;
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(w_bf16 + size_t(n) * size_t(K));
    const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(x_bf16 + size_t(0) * size_t(K));
    const uint4* __restrict__ x1 = (M_STATIC > 1) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(1) * size_t(K)) : x0;
    const uint4* __restrict__ x2 = (M_STATIC > 2) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(2) * size_t(K)) : x0;
    const uint4* __restrict__ x3 = (M_STATIC > 3) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(3) * size_t(K)) : x0;
    const uint4* __restrict__ x4 = (M_STATIC > 4) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(4) * size_t(K)) : x0;
    const uint4* __restrict__ x5 = (M_STATIC > 5) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(5) * size_t(K)) : x0;
    const uint4* __restrict__ x6 = (M_STATIC > 6) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(6) * size_t(K)) : x0;
    const uint4* __restrict__ x7 = (M_STATIC > 7) ? reinterpret_cast<const uint4*>(x_bf16 + size_t(7) * size_t(K)) : x0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int k8 = lane; k8 < K8; k8 += 64) {
        const uint4 wv = w4[k8];
        a0 += dot8_bf16_u4(x0[k8], wv);
        if constexpr (M_STATIC > 1) a1 += dot8_bf16_u4(x1[k8], wv);
        if constexpr (M_STATIC > 2) a2 += dot8_bf16_u4(x2[k8], wv);
        if constexpr (M_STATIC > 3) a3 += dot8_bf16_u4(x3[k8], wv);
        if constexpr (M_STATIC > 4) a4 += dot8_bf16_u4(x4[k8], wv);
        if constexpr (M_STATIC > 5) a5 += dot8_bf16_u4(x5[k8], wv);
        if constexpr (M_STATIC > 6) a6 += dot8_bf16_u4(x6[k8], wv);
        if constexpr (M_STATIC > 7) a7 += dot8_bf16_u4(x7[k8], wv);
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        a0 += __shfl_xor(a0, off, 64);
        if constexpr (M_STATIC > 1) a1 += __shfl_xor(a1, off, 64);
        if constexpr (M_STATIC > 2) a2 += __shfl_xor(a2, off, 64);
        if constexpr (M_STATIC > 3) a3 += __shfl_xor(a3, off, 64);
        if constexpr (M_STATIC > 4) a4 += __shfl_xor(a4, off, 64);
        if constexpr (M_STATIC > 5) a5 += __shfl_xor(a5, off, 64);
        if constexpr (M_STATIC > 6) a6 += __shfl_xor(a6, off, 64);
        if constexpr (M_STATIC > 7) a7 += __shfl_xor(a7, off, 64);
    }
    if (lane == 0) {
        if constexpr (OUT_F32) {
            float* yp = reinterpret_cast<float*>(y_out);
            yp[size_t(0) * size_t(N) + size_t(n)] = a0;
            if constexpr (M_STATIC > 1) yp[size_t(1) * size_t(N) + size_t(n)] = a1;
            if constexpr (M_STATIC > 2) yp[size_t(2) * size_t(N) + size_t(n)] = a2;
            if constexpr (M_STATIC > 3) yp[size_t(3) * size_t(N) + size_t(n)] = a3;
            if constexpr (M_STATIC > 4) yp[size_t(4) * size_t(N) + size_t(n)] = a4;
            if constexpr (M_STATIC > 5) yp[size_t(5) * size_t(N) + size_t(n)] = a5;
            if constexpr (M_STATIC > 6) yp[size_t(6) * size_t(N) + size_t(n)] = a6;
            if constexpr (M_STATIC > 7) yp[size_t(7) * size_t(N) + size_t(n)] = a7;
        } else {
            uint16_t* yp = reinterpret_cast<uint16_t*>(y_out);
            yp[size_t(0) * size_t(N) + size_t(n)] = d_f32_to_bf16(a0);
            if constexpr (M_STATIC > 1) yp[size_t(1) * size_t(N) + size_t(n)] = d_f32_to_bf16(a1);
            if constexpr (M_STATIC > 2) yp[size_t(2) * size_t(N) + size_t(n)] = d_f32_to_bf16(a2);
            if constexpr (M_STATIC > 3) yp[size_t(3) * size_t(N) + size_t(n)] = d_f32_to_bf16(a3);
            if constexpr (M_STATIC > 4) yp[size_t(4) * size_t(N) + size_t(n)] = d_f32_to_bf16(a4);
            if constexpr (M_STATIC > 5) yp[size_t(5) * size_t(N) + size_t(n)] = d_f32_to_bf16(a5);
            if constexpr (M_STATIC > 6) yp[size_t(6) * size_t(N) + size_t(n)] = d_f32_to_bf16(a6);
            if constexpr (M_STATIC > 7) yp[size_t(7) * size_t(N) + size_t(n)] = d_f32_to_bf16(a7);
        }
    }
}

__device__ __forceinline__ float dot4_bf16_f32(uint2 xv, float4 wv) {
    float s = 0.0f;
    s = fmaf(d_bf16_to_f32(uint16_t(xv.x & 0xFFFFu)), wv.x, s);
    s = fmaf(d_bf16_to_f32(uint16_t(xv.x >> 16)),     wv.y, s);
    s = fmaf(d_bf16_to_f32(uint16_t(xv.y & 0xFFFFu)), wv.z, s);
    s = fmaf(d_bf16_to_f32(uint16_t(xv.y >> 16)),     wv.w, s);
    return s;
}

template <int M_STATIC>
__global__ __launch_bounds__(256, 4)
void k_bf16_f32w_gemv_ws(const uint16_t* __restrict__ x_bf16,
                         const float* __restrict__ w_f32,
                         float* __restrict__ y_f32,
                         int N, int K) {
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int n = int(blockIdx.x) * 4 + warp;
    if (n >= N) return;

    const int K4 = K >> 2;
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(w_f32 + size_t(n) * size_t(K));
    const uint2* __restrict__ x0 = reinterpret_cast<const uint2*>(x_bf16 + size_t(0) * size_t(K));
    const uint2* __restrict__ x1 = (M_STATIC > 1) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(1) * size_t(K)) : x0;
    const uint2* __restrict__ x2 = (M_STATIC > 2) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(2) * size_t(K)) : x0;
    const uint2* __restrict__ x3 = (M_STATIC > 3) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(3) * size_t(K)) : x0;
    const uint2* __restrict__ x4 = (M_STATIC > 4) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(4) * size_t(K)) : x0;
    const uint2* __restrict__ x5 = (M_STATIC > 5) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(5) * size_t(K)) : x0;
    const uint2* __restrict__ x6 = (M_STATIC > 6) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(6) * size_t(K)) : x0;
    const uint2* __restrict__ x7 = (M_STATIC > 7) ? reinterpret_cast<const uint2*>(x_bf16 + size_t(7) * size_t(K)) : x0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int k4 = lane; k4 < K4; k4 += 64) {
        const float4 wv = w4[k4];
        a0 += dot4_bf16_f32(x0[k4], wv);
        if constexpr (M_STATIC > 1) a1 += dot4_bf16_f32(x1[k4], wv);
        if constexpr (M_STATIC > 2) a2 += dot4_bf16_f32(x2[k4], wv);
        if constexpr (M_STATIC > 3) a3 += dot4_bf16_f32(x3[k4], wv);
        if constexpr (M_STATIC > 4) a4 += dot4_bf16_f32(x4[k4], wv);
        if constexpr (M_STATIC > 5) a5 += dot4_bf16_f32(x5[k4], wv);
        if constexpr (M_STATIC > 6) a6 += dot4_bf16_f32(x6[k4], wv);
        if constexpr (M_STATIC > 7) a7 += dot4_bf16_f32(x7[k4], wv);
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        a0 += __shfl_xor(a0, off, 64);
        if constexpr (M_STATIC > 1) a1 += __shfl_xor(a1, off, 64);
        if constexpr (M_STATIC > 2) a2 += __shfl_xor(a2, off, 64);
        if constexpr (M_STATIC > 3) a3 += __shfl_xor(a3, off, 64);
        if constexpr (M_STATIC > 4) a4 += __shfl_xor(a4, off, 64);
        if constexpr (M_STATIC > 5) a5 += __shfl_xor(a5, off, 64);
        if constexpr (M_STATIC > 6) a6 += __shfl_xor(a6, off, 64);
        if constexpr (M_STATIC > 7) a7 += __shfl_xor(a7, off, 64);
    }
    if (lane == 0) {
        y_f32[size_t(0) * size_t(N) + size_t(n)] = a0;
        if constexpr (M_STATIC > 1) y_f32[size_t(1) * size_t(N) + size_t(n)] = a1;
        if constexpr (M_STATIC > 2) y_f32[size_t(2) * size_t(N) + size_t(n)] = a2;
        if constexpr (M_STATIC > 3) y_f32[size_t(3) * size_t(N) + size_t(n)] = a3;
        if constexpr (M_STATIC > 4) y_f32[size_t(4) * size_t(N) + size_t(n)] = a4;
        if constexpr (M_STATIC > 5) y_f32[size_t(5) * size_t(N) + size_t(n)] = a5;
        if constexpr (M_STATIC > 6) y_f32[size_t(6) * size_t(N) + size_t(n)] = a6;
        if constexpr (M_STATIC > 7) y_f32[size_t(7) * size_t(N) + size_t(n)] = a7;
    }
}

// ---- Prefill MoE M_e >= 16 rocblas_hgemm Crossover Helper Kernels (Zero Scratch) ----
__global__ __launch_bounds__(256, 4)
void k_gather_rows_f16(const uint16_t* __restrict__ src_MxK,
                       const int32_t* __restrict__ row_ids,
                       uint16_t* __restrict__ dst_CxK,
                       int C, int K) {
    const int c = int(blockIdx.y);
    if (c >= C) return;
    const int r = row_ids[c];
    const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src_MxK + size_t(r) * size_t(K));
    uint4* __restrict__ d4       = reinterpret_cast<uint4*>(dst_CxK + size_t(c) * size_t(K));
    const int K8 = K >> 3;
    for (int k8 = int(blockIdx.x) * 256 + int(threadIdx.x); k8 < K8; k8 += int(gridDim.x) * 256) {
        d4[k8] = s4[k8];
    }
}

__global__ __launch_bounds__(256, 4)
void k_situ_glu_f16_to_f32(const uint16_t* __restrict__ gate_f16,
                           const uint16_t* __restrict__ up_f16,
                           float* __restrict__ out_f32,
                           int total, float situ_beta, float situ_linear_beta) {
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (idx < total) {
        const float g = __half2float(*reinterpret_cast<const __half*>(&gate_f16[idx]));
        const float u = __half2float(*reinterpret_cast<const __half*>(&up_f16[idx]));
        out_f32[idx] = d_situ_glu(g, u, situ_beta, situ_linear_beta);
    }
}

__global__ __launch_bounds__(256, 4)
void k_scatter_add_weighted_f32(const float* __restrict__ src_CxK,
                                const int32_t* __restrict__ row_ids,
                                const float* __restrict__ row_wts,
                                float* __restrict__ acc_MxK,
                                int C, int K) {
    const int c = int(blockIdx.y);
    if (c >= C) return;
    const int r = row_ids[c];
    const float wt = row_wts[c];
    const float* __restrict__ s = src_CxK + size_t(c) * size_t(K);
    float* __restrict__ dst = acc_MxK + size_t(r) * size_t(K);
    for (int k = int(blockIdx.x) * 256 + int(threadIdx.x); k < K; k += int(gridDim.x) * 256) {
        atomicAdd(dst + k, s[k] * wt);
    }
}

__global__ __launch_bounds__(256, 4)
void k_scatter_slot_weighted_f32(const float* __restrict__ src_CxK,
                                 const int32_t* __restrict__ row_ids,
                                 const int32_t* __restrict__ row_slots,
                                 const float* __restrict__ row_wts,
                                 float* __restrict__ y_slots_f32,
                                 int C, int K) {
    const int c = int(blockIdx.y);
    if (c >= C) return;
    const int r = row_ids[c];
    const int slot = row_slots[c];
    const float wt = row_wts[c];
    const float* __restrict__ s = src_CxK + size_t(c) * size_t(K);
    float* __restrict__ dst = y_slots_f32 + (size_t(r) * 10u + size_t(slot)) * size_t(K);
    for (int k = int(blockIdx.x) * 256 + int(threadIdx.x); k < K; k += int(gridDim.x) * 256) {
        dst[k] = s[k] * wt;
    }
}

__global__ __launch_bounds__(256, 8)
void k_reduce_moe_slots_f32(const float* __restrict__ y_slots_f32,
                            float* __restrict__ y_out_f32,
                            const int32_t* __restrict__ slot_mask3,
                            int N_out) {
    const int m = int(blockIdx.y);
    const int n = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (n >= N_out) return;
    const float* __restrict__ base = y_slots_f32 + size_t(m) * 10u * size_t(N_out) + size_t(n);
    float acc = 0.0f;
    if (slot_mask3 != nullptr) {
        const uint32_t m0 = uint32_t(slot_mask3[0]);
        const uint32_t m1 = uint32_t(slot_mask3[1]);
        const uint32_t m2 = uint32_t(slot_mask3[2]);
        const int bit0 = m * 10;
        #pragma unroll 1
        for (int s = 0; s < 10; ++s) {
            const int bit = bit0 + s;
            const uint32_t w = (bit < 32) ? m0 : ((bit < 64) ? m1 : m2);
            if ((w >> (bit & 31)) & 1u) {
                acc += base[size_t(s) * size_t(N_out)];
            }
        }
    } else {
        #pragma unroll
        for (int s = 0; s < 10; ++s) {
            acc += base[size_t(s) * size_t(N_out)];
        }
    }
    y_out_f32[size_t(m) * size_t(N_out) + size_t(n)] = acc;
}

// ---- GPU-Accelerated WHT128-HLUT4 LDLQ Quantizer (1 block = 1 scale-group of 4 rows, 64 threads = 1 wave!) ----
// Keeps 4x2048 working buffer in 32 KiB LDS -> ZERO scratch, 512 blocks/matrix, quantizes 512 experts in ~1s on 5x MI50!
__device__ __forceinline__ uint32_t d_hlut4_quant_mag(float z, const float th[7]) {
    if (z >= th[3]) {
        if (z >= th[5]) return (z >= th[6]) ? 7u : 6u;
        return (z >= th[4]) ? 5u : 4u;
    } else {
        if (z >= th[1]) return (z >= th[2]) ? 3u : 2u;
        return (z >= th[0]) ? 1u : 0u;
    }
}

__global__ __launch_bounds__(64, 2)
void k_wht128_hlut4_ldlq_quant_gpu(
    const float* __restrict__ w_rot_NxK,
    const float* __restrict__ rup_KxK,
    const float* __restrict__ levels8_f32,
    const float* __restrict__ thresh7_f32,
    Hlut4Tile32x32* __restrict__ out_tiles,
    float* __restrict__ sq_sig_err2,
    int N, int K)
{
    // 1 block handles 4 consecutive rows (one 4x32 scale group per kt!) across all K=2048 cols
    const int r4 = int(blockIdx.x) * 4;
    if (r4 >= N) return;
    const int nt = r4 >> 5;
    const int lr0 = r4 & 31;
    const int sg = lr0 >> 2;
    const int kt_cnt = K >> 5;
    const int lane = int(threadIdx.x);

    __shared__ float s_w[4][2048];   // 32 KiB LDS (exact fit for K <= 2048!)
    __shared__ float s_lv[8];
    __shared__ float s_th[7];
    __shared__ float s_sig[64];      // per-kt scale (kt_cnt <= 64 for K <= 2048)

    if (lane < 8) s_lv[lane] = levels8_f32[lane];
    if (lane < 7) s_th[lane] = thresh7_f32[lane];

    // Load 4 rows into LDS
    for (int dr = 0; dr < 4; ++dr) {
        const float* __restrict__ src = w_rot_NxK + size_t(r4 + dr) * size_t(K);
        for (int k = lane; k < K; k += 64) s_w[dr][k] = src[k];
    }
    __syncthreads();

    // Compute exact 128-weight FP16 RMS scale per kt tile (kt = 0..kt_cnt-1)
    if (lane < kt_cnt) {
        float sq = 0.0f;
        const int c0 = lane * 32;
        for (int dr = 0; dr < 4; ++dr) {
            for (int dc = 0; dc < 32; ++dc) {
                const float v = s_w[dr][c0 + dc];
                sq = fmaf(v, v, sq);
            }
        }
        const __half h_sig = __float2half_rn(sqrtf(sq * (1.0f / 128.0f) + 1e-30f));
        const uint16_t u_sig = *reinterpret_cast<const uint16_t*>(&h_sig);
        out_tiles[size_t(nt) * size_t(kt_cnt) + size_t(lane)].scales_f16[sg] = u_sig;
        s_sig[lane] = __half2float(h_sig);
    }
    __syncthreads();

    // Causal column quantization + parallel LDLQ feedback across all remaining cols k > col!
    __shared__ uint32_t s_nib_word[4];
    float local_sig2 = 0.0f, local_err2 = 0.0f;

    for (int col = 0; col < K; ++col) {
        const int kt = col >> 5;
        const int lc = col & 31;
        const int word_in_row = lc >> 3;
        const int nib_sh = (lc & 7) * 4;
        if (nib_sh == 0 && lane < 4) s_nib_word[lane] = 0u;
        __syncthreads();

        __shared__ float s_e[4];
        if (lane < 4) {
            const float orig = w_rot_NxK[size_t(r4 + lane) * size_t(K) + size_t(col)];
            const float wv   = s_w[lane][col];
            const float sig  = s_sig[kt];
            const float inv_s = (sig > 0.0f) ? (1.0f / sig) : 0.0f;
            const uint32_t mag = d_hlut4_quant_mag(fabsf(wv) * inv_s, s_th);
            const uint32_t sgn = signbit(wv) ? 1u : 0u;
            const uint32_t nib = (sgn << 3) | mag;
            s_nib_word[lane] |= (nib << nib_sh);
            const float qv = (sgn ? -s_lv[mag] : s_lv[mag]) * sig;
            const float diff = orig - qv;
            s_e[lane] = diff;
            local_sig2 = fmaf(orig, orig, local_sig2);
            local_err2 = fmaf(diff, diff, local_err2);
        }
        __syncthreads();

        if (nib_sh == 28 && lane < 4) {
            out_tiles[size_t(nt) * size_t(kt_cnt) + size_t(kt)].q_nib4[(lr0 + lane) * 4 + word_in_row] = s_nib_word[lane];
        }

        if (rup_KxK != nullptr && col + 1 < K) {
            const float* __restrict__ rrow = rup_KxK + size_t(col) * size_t(K);
            const float e0 = s_e[0], e1 = s_e[1], e2 = s_e[2], e3 = s_e[3];
            for (int k = col + 1 + lane; k < K; k += 64) {
                const float rv = rrow[k];
                s_w[0][k] = fmaf(e0, rv, s_w[0][k]);
                s_w[1][k] = fmaf(e1, rv, s_w[1][k]);
                s_w[2][k] = fmaf(e2, rv, s_w[2][k]);
                s_w[3][k] = fmaf(e3, rv, s_w[3][k]);
            }
            __syncthreads();
        }
    }

    if (sq_sig_err2 != nullptr && lane < 4) {
        atomicAdd(sq_sig_err2 + 0, local_sig2);
        atomicAdd(sq_sig_err2 + 1, local_err2);
    }
}

} // namespace

void launch_wht128_bf16_to_f16(const uint16_t* d_in_bf16, const uint64_t* d_signs,
                               uint16_t* d_out_f16, int M, int K, hipStream_t stream) {
    hipLaunchKernelGGL(k_wht128_bf16_to_f16, dim3(K / 128, M), dim3(128), 0, stream,
                       d_in_bf16, d_signs, d_out_f16, K);
}

void launch_wht128_f16_to_f16_and_bf16(const uint16_t* d_in_f16, const uint64_t* d_signs,
                                       uint16_t* d_out_f16, uint16_t* d_out_bf16_or_null,
                                       int M, int K, hipStream_t stream) {
    if (M <= 0 || K <= 0) return;
    hipLaunchKernelGGL(k_wht128_f16_to_f16_and_bf16, dim3(K / 128, M), dim3(128), 0, stream,
                       d_in_f16, d_signs, d_out_f16, d_out_bf16_or_null, K);
}

void launch_wht128_f32_to_f16(const float* d_in_f32, const uint64_t* d_signs,
                              uint16_t* d_out_f16, int M, int K, hipStream_t stream) {
    hipLaunchKernelGGL(k_wht128_f32_to_f16, dim3(K / 128, M), dim3(128), 0, stream,
                       d_in_f32, d_signs, d_out_f16, K);
}

void launch_wht128_grouped_f32_to_f16(const float* d_in_f32, const ExpertDispatchGroup* d_groups, int n_groups,
                                      const uint64_t* d_signs, uint16_t* d_out_f16, int K, hipStream_t stream,
                                      int max_c) {
    if (n_groups <= 0) return;
    if (max_c <= 0 || max_c > kMaxGroupTokens) max_c = kMaxGroupTokens;
    hipLaunchKernelGGL(k_wht128_grouped_f32_to_f16, dim3(K / 128, n_groups * max_c), dim3(128), 0, stream,
                       d_in_f32, d_groups, d_signs, d_out_f16, K, max_c);
}

void launch_wht128_f32_to_f32(const float* d_in_f32, const uint64_t* d_signs,
                              float* d_out_f32, int M, int K, hipStream_t stream) {
    hipLaunchKernelGGL(k_wht128_f32_to_f32, dim3(K / 128, M), dim3(128), 0, stream,
                       d_in_f32, d_signs, d_out_f32, K);
}

void launch_hlut4_fdot2_gemv(const uint16_t* d_x_rot_f16, const Hlut4Tile32x32* d_tiles,
                             const uint16_t* d_levels8_f16, float* d_y_f32,
                             int N, int K, hipStream_t stream) {
    hipLaunchKernelGGL(k_hlut4_fdot2_gemv, dim3((N + 31) / 32), dim3(128), 0, stream,
                       d_x_rot_f16, d_tiles, d_levels8_f16, d_y_f32, N, K);
}

void launch_hlut4_fdot2_grouped_gu_situ(
    const uint16_t* d_x_rot_f16,
    const ExpertDispatchGroup* d_groups, int n_groups,
    const Hlut4Tile32x32* d_w1_tiles, const Hlut4Tile32x32* d_w3_tiles,
    const uint16_t* d_levels8_f16,
    float* d_z_f32, int N_inter, int K_in,
    float situ_beta, float situ_linear_beta, hipStream_t stream) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut4_fdot2_grouped_gu_situ, dim3((N_inter + 31) / 32, n_groups), dim3(128), 0, stream,
                       d_x_rot_f16, d_groups, d_w1_tiles, d_w3_tiles, d_levels8_f16,
                       d_z_f32, N_inter, K_in, situ_beta, situ_linear_beta);
}

void launch_hlut4_fdot2_grouped_down_atomic(
    const uint16_t* d_z_rot_f16,
    const ExpertDispatchGroup* d_groups, int n_groups,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_accum_f32, int N_out, int K_inter, hipStream_t stream) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut4_fdot2_grouped_down_atomic, dim3((N_out + 31) / 32, n_groups), dim3(128), 0, stream,
                       d_z_rot_f16, d_groups, d_w2_tiles, d_levels8_f16, d_y_accum_f32, N_out, K_inter);
}

void launch_hlut4_fdot2_grouped_down_slot(
    const uint16_t* d_z_rot_f16,
    const ExpertDispatchGroup* d_groups, int n_groups,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_slots_f32, int N_out, int K_inter, hipStream_t stream) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut4_fdot2_grouped_down_slot, dim3((N_out + 31) / 32, n_groups), dim3(128), 0, stream,
                       d_z_rot_f16, d_groups, d_w2_tiles, d_levels8_f16, d_y_slots_f32, N_out, K_inter);
}

void launch_unpack_hlut4_to_f16(const Hlut4Tile32x32* d_tiles, const uint16_t* d_levels8_f16,
                                uint16_t* d_out_f16, int N, int K, hipStream_t stream) {
    hipLaunchKernelGGL(k_unpack_hlut4_to_f16, dim3(K / 32, N / 32), dim3(256), 0, stream,
                       d_tiles, d_levels8_f16, d_out_f16, N, K);
}

void launch_gather_rows_f16(const uint16_t* d_src_MxK, const int32_t* d_row_ids,
                            uint16_t* d_dst_CxK, int C, int K, hipStream_t stream) {
    if (C <= 0 || K <= 0) return;
    hipLaunchKernelGGL(k_gather_rows_f16, dim3(std::max(1, (K / 8 + 255) / 256), C), dim3(256), 0, stream,
                       d_src_MxK, d_row_ids, d_dst_CxK, C, K);
}

void launch_situ_glu_f16_to_f32(const uint16_t* d_gate_f16, const uint16_t* d_up_f16,
                                float* d_out_f32, int C, int N,
                                float situ_beta, float situ_linear_beta, hipStream_t stream) {
    const int tot = C * N;
    if (tot <= 0) return;
    hipLaunchKernelGGL(k_situ_glu_f16_to_f32, dim3((tot + 255) / 256), dim3(256), 0, stream,
                       d_gate_f16, d_up_f16, d_out_f32, tot, situ_beta, situ_linear_beta);
}

void launch_scatter_add_weighted_f32(const float* d_src_CxK, const int32_t* d_row_ids,
                                     const float* d_row_wts, float* d_acc_MxK,
                                     int C, int K, hipStream_t stream) {
    if (C <= 0 || K <= 0) return;
    hipLaunchKernelGGL(k_scatter_add_weighted_f32, dim3((K + 255) / 256, C), dim3(256), 0, stream,
                       d_src_CxK, d_row_ids, d_row_wts, d_acc_MxK, C, K);
}

void launch_scatter_slot_weighted_f32(const float* d_src_CxK, const int32_t* d_row_ids,
                                      const int32_t* d_row_slots, const float* d_row_wts,
                                      float* d_y_slots_f32, int C, int K, hipStream_t stream) {
    if (C <= 0 || K <= 0) return;
    hipLaunchKernelGGL(k_scatter_slot_weighted_f32, dim3((K + 255) / 256, C), dim3(256), 0, stream,
                       d_src_CxK, d_row_ids, d_row_slots, d_row_wts, d_y_slots_f32, C, K);
}

void launch_reduce_moe_slots_f32(const float* d_y_slots_f32, float* d_y_out_f32,
                                 int M, int N_out, hipStream_t stream,
                                 const int32_t* d_slot_mask3) {
    if (M <= 0 || N_out <= 0) return;
    hipLaunchKernelGGL(k_reduce_moe_slots_f32, dim3((N_out + 255) / 256, M), dim3(256), 0, stream,
                       d_y_slots_f32, d_y_out_f32, d_slot_mask3, N_out);
}

void launch_wht128_hlut4_ldlq_quant_gpu(
    const float* d_w_rot_NxK, const float* d_rup_KxK,
    const float* d_levels8_f32, const float* d_thresh7_f32,
    Hlut4Tile32x32* d_out_tiles, float* d_sq_sig_err2,
    int N, int K, hipStream_t stream) {
    hipLaunchKernelGGL(k_wht128_hlut4_ldlq_quant_gpu, dim3(N / 4), dim3(64), 0, stream,
                       d_w_rot_NxK, d_rup_KxK, d_levels8_f32, d_thresh7_f32,
                       d_out_tiles, d_sq_sig_err2, N, K);
}

void launch_situ_glu_bf16(const uint16_t* d_gate_bf16, const uint16_t* d_up_bf16,
                          uint16_t* d_out_bf16, int M, int N,
                          float situ_beta, float situ_linear_beta, hipStream_t stream) {
    const int tot = M * N;
    if (tot <= 0) return;
    hipLaunchKernelGGL(k_situ_glu_bf16, dim3((tot + 255) / 256), dim3(256), 0, stream,
                       d_gate_bf16, d_up_bf16, d_out_bf16, tot, situ_beta, situ_linear_beta);
}

void launch_situ_glu_f16(const uint16_t* d_gate_f16, const uint16_t* d_up_f16,
                         uint16_t* d_out_f16, int M, int N,
                         float situ_beta, float situ_linear_beta, hipStream_t stream) {
    const int tot = M * N;
    if (tot <= 0) return;
    hipLaunchKernelGGL(k_situ_glu_f16, dim3((tot + 255) / 256), dim3(256), 0, stream,
                       d_gate_f16, d_up_f16, d_out_f16, tot, situ_beta, situ_linear_beta);
}

void launch_bf16_gemv_bf16(const uint16_t* d_x_bf16, const uint16_t* d_w_bf16,
                           uint16_t* d_y_bf16, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + 3) / 4);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_bf16 + size_t(off) * size_t(K);
        uint16_t* yp       = d_y_bf16 + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_bf16_gemv_ws<1, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_bf16_gemv_ws<2, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_bf16_gemv_ws<3, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_bf16_gemv_ws<4, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_bf16_gemv_ws<5, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_bf16_gemv_ws<6, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_bf16_gemv_ws<7, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            default: hipLaunchKernelGGL((k_bf16_gemv_ws<8, false>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
        }
    }
}

void launch_bf16_gemv_f32(const uint16_t* d_x_bf16, const uint16_t* d_w_bf16,
                          float* d_y_f32, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + 3) / 4);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_bf16 + size_t(off) * size_t(K);
        float* yp          = d_y_f32  + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_bf16_gemv_ws<1, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_bf16_gemv_ws<2, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_bf16_gemv_ws<3, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_bf16_gemv_ws<4, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_bf16_gemv_ws<5, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_bf16_gemv_ws<6, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_bf16_gemv_ws<7, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
            default: hipLaunchKernelGGL((k_bf16_gemv_ws<8, true>), grid, dim3(256), 0, stream, xp, d_w_bf16, yp, N, K); break;
        }
    }
}

void launch_bf16_f32w_gemv_f32(const uint16_t* d_x_bf16, const float* d_w_f32,
                               float* d_y_f32, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + 3) / 4);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_bf16 + size_t(off) * size_t(K);
        float* yp          = d_y_f32  + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<1>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<2>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<3>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<4>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<5>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<6>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<7>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
            default: hipLaunchKernelGGL((k_bf16_f32w_gemv_ws<8>), grid, dim3(256), 0, stream, xp, d_w_f32, yp, N, K); break;
        }
    }
}

void launch_situ_glu_f32(const float* d_gate_f32, const float* d_up_f32,
                         float* d_out_f32, int M, int N,
                         float situ_beta, float situ_linear_beta, hipStream_t stream) {
    const int total = M * N;
    if (total <= 0) return;
    hipLaunchKernelGGL(k_situ_glu_f32, dim3((total + 255) / 256), dim3(256), 0, stream,
                       d_gate_f32, d_up_f32, d_out_f32, total, situ_beta, situ_linear_beta);
}

namespace {

__device__ __forceinline__ uint16_t d_bf16_to_f16_clamped(uint16_t b) {
    float v = d_bf16_to_f32(b);
    if (v > 65504.0f) v = 65504.0f;
    else if (v < -65504.0f) v = -65504.0f;
    return d_f32_to_f16(v);
}

__global__ __launch_bounds__(256, 8)
void k_bf16_to_f16_vec(const uint16_t* __restrict__ in_bf16, uint16_t* __restrict__ out_f16, int total) {
    const int idx4 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = idx4 * 4;
    if (base + 3 < total) {
        const uint2 v = reinterpret_cast<const uint2*>(in_bf16)[idx4];
        const uint16_t h0 = d_bf16_to_f16_clamped(uint16_t(v.x & 0xFFFFu));
        const uint16_t h1 = d_bf16_to_f16_clamped(uint16_t(v.x >> 16));
        const uint16_t h2 = d_bf16_to_f16_clamped(uint16_t(v.y & 0xFFFFu));
        const uint16_t h3 = d_bf16_to_f16_clamped(uint16_t(v.y >> 16));
        uint2 o;
        o.x = uint32_t(h0) | (uint32_t(h1) << 16);
        o.y = uint32_t(h2) | (uint32_t(h3) << 16);
        reinterpret_cast<uint2*>(out_f16)[idx4] = o;
    } else {
        for (int i = base; i < total; ++i) out_f16[i] = d_bf16_to_f16_clamped(in_bf16[i]);
    }
}

__device__ __forceinline__ float dot8_f16_u4_fdot2(uint4 xv, uint4 wv, float acc) {
    acc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(xv.x), d_u32_to_f16x2(wv.x), acc, false);
    acc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(xv.y), d_u32_to_f16x2(wv.y), acc, false);
    acc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(xv.z), d_u32_to_f16x2(wv.z), acc, false);
    acc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(xv.w), d_u32_to_f16x2(wv.w), acc, false);
    return acc;
}

// Fast 2-row-per-wavefront K=256 FP16 fdot2 GEMV (100% lane utilization for DSpark markov_w2 [201216, 256]!)
__global__ __launch_bounds__(256, 8)
void k_f16_gemv_k256_m1_f32(const uint16_t* __restrict__ x_f16,
                            const uint16_t* __restrict__ w_f16,
                            float* __restrict__ y_out,
                            int N) {
    const int tid = int(threadIdx.x);
    const int n = int(blockIdx.x) * 8 + (tid >> 5);
    const int l32 = tid & 31; // 32 lanes * 8 halfs = 256!
    if (n >= N) return;
    const uint4 xv = reinterpret_cast<const uint4*>(x_f16)[l32];
    const uint4 wv = reinterpret_cast<const uint4*>(w_f16 + size_t(n) * 256u)[l32];
    float acc = dot8_f16_u4_fdot2(xv, wv, 0.0f);
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc += __shfl_xor(acc, off, 64);
    if (l32 == 0) y_out[n] = acc;
}

template <int M_STATIC, bool OUT_F32>
__global__ __launch_bounds__(256, 4)
void k_f16_gemv_ws(const uint16_t* __restrict__ x_f16,
                   const uint16_t* __restrict__ w_f16,
                   void* __restrict__ y_out,
                   int N, int K) {
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int n = int(blockIdx.x) * 4 + warp;
    if (n >= N) return;

    const int K8 = K >> 3;
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(w_f16 + size_t(n) * size_t(K));
    const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(x_f16 + size_t(0) * size_t(K));
    const uint4* __restrict__ x1 = (M_STATIC > 1) ? reinterpret_cast<const uint4*>(x_f16 + size_t(1) * size_t(K)) : x0;
    const uint4* __restrict__ x2 = (M_STATIC > 2) ? reinterpret_cast<const uint4*>(x_f16 + size_t(2) * size_t(K)) : x0;
    const uint4* __restrict__ x3 = (M_STATIC > 3) ? reinterpret_cast<const uint4*>(x_f16 + size_t(3) * size_t(K)) : x0;
    const uint4* __restrict__ x4 = (M_STATIC > 4) ? reinterpret_cast<const uint4*>(x_f16 + size_t(4) * size_t(K)) : x0;
    const uint4* __restrict__ x5 = (M_STATIC > 5) ? reinterpret_cast<const uint4*>(x_f16 + size_t(5) * size_t(K)) : x0;
    const uint4* __restrict__ x6 = (M_STATIC > 6) ? reinterpret_cast<const uint4*>(x_f16 + size_t(6) * size_t(K)) : x0;
    const uint4* __restrict__ x7 = (M_STATIC > 7) ? reinterpret_cast<const uint4*>(x_f16 + size_t(7) * size_t(K)) : x0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int k8 = lane; k8 < K8; k8 += 64) {
        const uint4 wv = w4[k8];
        a0 = dot8_f16_u4_fdot2(x0[k8], wv, a0);
        if constexpr (M_STATIC > 1) a1 = dot8_f16_u4_fdot2(x1[k8], wv, a1);
        if constexpr (M_STATIC > 2) a2 = dot8_f16_u4_fdot2(x2[k8], wv, a2);
        if constexpr (M_STATIC > 3) a3 = dot8_f16_u4_fdot2(x3[k8], wv, a3);
        if constexpr (M_STATIC > 4) a4 = dot8_f16_u4_fdot2(x4[k8], wv, a4);
        if constexpr (M_STATIC > 5) a5 = dot8_f16_u4_fdot2(x5[k8], wv, a5);
        if constexpr (M_STATIC > 6) a6 = dot8_f16_u4_fdot2(x6[k8], wv, a6);
        if constexpr (M_STATIC > 7) a7 = dot8_f16_u4_fdot2(x7[k8], wv, a7);
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        a0 += __shfl_xor(a0, off, 64);
        if constexpr (M_STATIC > 1) a1 += __shfl_xor(a1, off, 64);
        if constexpr (M_STATIC > 2) a2 += __shfl_xor(a2, off, 64);
        if constexpr (M_STATIC > 3) a3 += __shfl_xor(a3, off, 64);
        if constexpr (M_STATIC > 4) a4 += __shfl_xor(a4, off, 64);
        if constexpr (M_STATIC > 5) a5 += __shfl_xor(a5, off, 64);
        if constexpr (M_STATIC > 6) a6 += __shfl_xor(a6, off, 64);
        if constexpr (M_STATIC > 7) a7 += __shfl_xor(a7, off, 64);
    }
    if (lane == 0) {
        if constexpr (OUT_F32) {
            float* yp = reinterpret_cast<float*>(y_out);
            yp[size_t(0) * size_t(N) + size_t(n)] = a0;
            if constexpr (M_STATIC > 1) yp[size_t(1) * size_t(N) + size_t(n)] = a1;
            if constexpr (M_STATIC > 2) yp[size_t(2) * size_t(N) + size_t(n)] = a2;
            if constexpr (M_STATIC > 3) yp[size_t(3) * size_t(N) + size_t(n)] = a3;
            if constexpr (M_STATIC > 4) yp[size_t(4) * size_t(N) + size_t(n)] = a4;
            if constexpr (M_STATIC > 5) yp[size_t(5) * size_t(N) + size_t(n)] = a5;
            if constexpr (M_STATIC > 6) yp[size_t(6) * size_t(N) + size_t(n)] = a6;
            if constexpr (M_STATIC > 7) yp[size_t(7) * size_t(N) + size_t(n)] = a7;
        } else {
            uint16_t* yp = reinterpret_cast<uint16_t*>(y_out);
            yp[size_t(0) * size_t(N) + size_t(n)] = d_f32_to_bf16(a0);
            if constexpr (M_STATIC > 1) yp[size_t(1) * size_t(N) + size_t(n)] = d_f32_to_bf16(a1);
            if constexpr (M_STATIC > 2) yp[size_t(2) * size_t(N) + size_t(n)] = d_f32_to_bf16(a2);
            if constexpr (M_STATIC > 3) yp[size_t(3) * size_t(N) + size_t(n)] = d_f32_to_bf16(a3);
            if constexpr (M_STATIC > 4) yp[size_t(4) * size_t(N) + size_t(n)] = d_f32_to_bf16(a4);
            if constexpr (M_STATIC > 5) yp[size_t(5) * size_t(N) + size_t(n)] = d_f32_to_bf16(a5);
            if constexpr (M_STATIC > 6) yp[size_t(6) * size_t(N) + size_t(n)] = d_f32_to_bf16(a6);
            if constexpr (M_STATIC > 7) yp[size_t(7) * size_t(N) + size_t(n)] = d_f32_to_bf16(a7);
        }
    }
}

template <int M_STATIC>
__global__ __launch_bounds__(256, 4)
void k_hca_grouped_wo_a_lrga_f16_ws(
    const uint16_t* __restrict__ o_f16,
    const uint16_t* __restrict__ wo_a_f16,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* __restrict__ z_out_bf16,
    int n_groups, int o_lora_rank)
{
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int r = int(blockIdx.x) * 4 + warp;
    const int g = int(blockIdx.y);
    if (r >= o_lora_rank || g >= n_groups) return;

    const size_t o_tok_stride = size_t(n_groups) * 4096u;
    const size_t z_tok_stride = size_t(n_groups) * size_t(o_lora_rank);
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(
        wo_a_f16 + (size_t(g) * size_t(o_lora_rank) + size_t(r)) * 4096u);
    const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(o_f16 + size_t(0) * o_tok_stride + size_t(g) * 4096u);
    const uint4* __restrict__ x1 = (M_STATIC > 1) ? reinterpret_cast<const uint4*>(o_f16 + size_t(1) * o_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x2 = (M_STATIC > 2) ? reinterpret_cast<const uint4*>(o_f16 + size_t(2) * o_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x3 = (M_STATIC > 3) ? reinterpret_cast<const uint4*>(o_f16 + size_t(3) * o_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x4 = (M_STATIC > 4) ? reinterpret_cast<const uint4*>(o_f16 + size_t(4) * o_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x5 = (M_STATIC > 5) ? reinterpret_cast<const uint4*>(o_f16 + size_t(5) * o_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x6 = (M_STATIC > 6) ? reinterpret_cast<const uint4*>(o_f16 + size_t(6) * o_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x7 = (M_STATIC > 7) ? reinterpret_cast<const uint4*>(o_f16 + size_t(7) * o_tok_stride + size_t(g) * 4096u) : x0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int k8 = lane; k8 < 512; k8 += 64) {
        const uint4 wv = w4[k8];
        a0 = dot8_f16_u4_fdot2(x0[k8], wv, a0);
        if constexpr (M_STATIC > 1) a1 = dot8_f16_u4_fdot2(x1[k8], wv, a1);
        if constexpr (M_STATIC > 2) a2 = dot8_f16_u4_fdot2(x2[k8], wv, a2);
        if constexpr (M_STATIC > 3) a3 = dot8_f16_u4_fdot2(x3[k8], wv, a3);
        if constexpr (M_STATIC > 4) a4 = dot8_f16_u4_fdot2(x4[k8], wv, a4);
        if constexpr (M_STATIC > 5) a5 = dot8_f16_u4_fdot2(x5[k8], wv, a5);
        if constexpr (M_STATIC > 6) a6 = dot8_f16_u4_fdot2(x6[k8], wv, a6);
        if constexpr (M_STATIC > 7) a7 = dot8_f16_u4_fdot2(x7[k8], wv, a7);
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        a0 += __shfl_xor(a0, off, 64);
        if constexpr (M_STATIC > 1) a1 += __shfl_xor(a1, off, 64);
        if constexpr (M_STATIC > 2) a2 += __shfl_xor(a2, off, 64);
        if constexpr (M_STATIC > 3) a3 += __shfl_xor(a3, off, 64);
        if constexpr (M_STATIC > 4) a4 += __shfl_xor(a4, off, 64);
        if constexpr (M_STATIC > 5) a5 += __shfl_xor(a5, off, 64);
        if constexpr (M_STATIC > 6) a6 += __shfl_xor(a6, off, 64);
        if constexpr (M_STATIC > 7) a7 += __shfl_xor(a7, off, 64);
    }
    if (lane == 0) {
        auto finish = [&](int m, float val) {
            const size_t idx = size_t(m) * z_tok_stride + size_t(g) * size_t(o_lora_rank) + size_t(r);
            const float gv = d_bf16_to_f32(gate_bf16[idx]);
            z_out_bf16[idx] = d_f32_to_bf16(val * (1.0f / (1.0f + __expf(-gv))));
        };
        finish(0, a0);
        if constexpr (M_STATIC > 1) finish(1, a1);
        if constexpr (M_STATIC > 2) finish(2, a2);
        if constexpr (M_STATIC > 3) finish(3, a3);
        if constexpr (M_STATIC > 4) finish(4, a4);
        if constexpr (M_STATIC > 5) finish(5, a5);
        if constexpr (M_STATIC > 6) finish(6, a6);
        if constexpr (M_STATIC > 7) finish(7, a7);
    }
}

__device__ __forceinline__ uint4 d_bf16_u4_to_f16_u4(uint4 b) {
    auto cvt2 = [](uint32_t u) -> uint32_t {
        const uint16_t h0 = d_bf16_to_f16_clamped(uint16_t(u & 0xFFFFu));
        const uint16_t h1 = d_bf16_to_f16_clamped(uint16_t(u >> 16));
        return uint32_t(h0) | (uint32_t(h1) << 16);
    };
    uint4 o;
    o.x = cvt2(b.x);
    o.y = cvt2(b.y);
    o.z = cvt2(b.z);
    o.w = cvt2(b.w);
    return o;
}

__device__ __forceinline__ float d_reduce32_dpp(float v) {
    // 4-lane intra-quad zero-LDS VALU DPP + 3 xor shuffles (cuts ds_bpermute in half!)
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xf, 0xf, true)); // quad_perm:[1,0,3,2]
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xf, 0xf, true)); // quad_perm:[2,3,0,1]
    v += __shfl_xor(v, 4, 64);
    v += __shfl_xor(v, 8, 64);
    v += __shfl_xor(v, 16, 64);
    return v;
}

// Wave-8 Fused BF16-in FP16-weight K=256 M=1 (DSpark markov_w2 [201216, 256], 0 bf16_to_f16 launch!)
__global__ __launch_bounds__(256, 8)
void k_bf16x_f16w_gemv_k256_m1_f32(const uint16_t* __restrict__ x_bf16,
                                   const uint16_t* __restrict__ w_f16,
                                   float* __restrict__ y_out,
                                   int N) {
    __shared__ uint4 s_x[32];
    const int tid = int(threadIdx.x);
    if (tid < 32) {
        s_x[tid] = d_bf16_u4_to_f16_u4(reinterpret_cast<const uint4*>(x_bf16)[tid]);
    }
    __syncthreads();
    const int n = int(blockIdx.x) * 8 + (tid >> 5);
    const int l32 = tid & 31;
    if (n >= N) return;
    const uint4 wv = reinterpret_cast<const uint4*>(w_f16 + size_t(n) * 256u)[l32];
    float acc = dot8_f16_u4_fdot2(s_x[l32], wv, 0.0f);
    acc = d_reduce32_dpp(acc);
    if (l32 == 0) y_out[n] = acc;
}

// Wave-8 Fused BF16-in FP16-weight LDS-tiled 8-row/block (32 lanes/row) v_dot2_f32_f16 GEMV!
// Eliminates separate launch_bf16_to_f16, shares converted x_f16 across 8 rows in LDS, 100% M=1..8 bit-exact!
template <int M_STATIC, bool OUT_F32>
__global__ __launch_bounds__(256, 3)
void k_bf16x_f16w_gemv_r8_ws(const uint16_t* __restrict__ x_bf16,
                             const uint16_t* __restrict__ w_f16,
                             void* __restrict__ y_out,
                             int N, int K) {
    __shared__ uint4 s_x[M_STATIC][128]; // 1024 halfs/token/tile (<= 16 KiB LDS at M=8!)
    const int tid = int(threadIdx.x);
    const int row_loc = tid >> 5;  // 0..7
    const int l32     = tid & 31;  // 0..31
    const int n       = int(blockIdx.x) * 8 + row_loc;
    const bool valid  = (n < N);
    const int K8      = K >> 3;
    const uint4* __restrict__ w4 = valid ? reinterpret_cast<const uint4*>(w_f16 + size_t(n) * size_t(K)) : nullptr;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    for (int k0 = 0; k0 < K8; k0 += 128) {
        const int t_len = (K8 - k0 < 128) ? (K8 - k0) : 128;
        for (int idx = tid; idx < M_STATIC * t_len; idx += 256) {
            const int m = (t_len == 128) ? (idx >> 7) : (idx / t_len);
            const int k = (t_len == 128) ? (idx & 127) : (idx - m * t_len);
            const uint4 bv = reinterpret_cast<const uint4*>(x_bf16 + size_t(m) * size_t(K))[k0 + k];
            s_x[m][k] = d_bf16_u4_to_f16_u4(bv);
        }
        __syncthreads();

        if (valid) {
            for (int k = l32; k < t_len; k += 32) {
                const uint4 wv = w4[k0 + k];
                a0 = dot8_f16_u4_fdot2(s_x[0][k], wv, a0);
                if constexpr (M_STATIC > 1) a1 = dot8_f16_u4_fdot2(s_x[1][k], wv, a1);
                if constexpr (M_STATIC > 2) a2 = dot8_f16_u4_fdot2(s_x[2][k], wv, a2);
                if constexpr (M_STATIC > 3) a3 = dot8_f16_u4_fdot2(s_x[3][k], wv, a3);
                if constexpr (M_STATIC > 4) a4 = dot8_f16_u4_fdot2(s_x[4][k], wv, a4);
                if constexpr (M_STATIC > 5) a5 = dot8_f16_u4_fdot2(s_x[5][k], wv, a5);
                if constexpr (M_STATIC > 6) a6 = dot8_f16_u4_fdot2(s_x[6][k], wv, a6);
                if constexpr (M_STATIC > 7) a7 = dot8_f16_u4_fdot2(s_x[7][k], wv, a7);
            }
        }
        __syncthreads();
    }

    if (!valid) return;
    a0 = d_reduce32_dpp(a0);
    if constexpr (M_STATIC > 1) a1 = d_reduce32_dpp(a1);
    if constexpr (M_STATIC > 2) a2 = d_reduce32_dpp(a2);
    if constexpr (M_STATIC > 3) a3 = d_reduce32_dpp(a3);
    if constexpr (M_STATIC > 4) a4 = d_reduce32_dpp(a4);
    if constexpr (M_STATIC > 5) a5 = d_reduce32_dpp(a5);
    if constexpr (M_STATIC > 6) a6 = d_reduce32_dpp(a6);
    if constexpr (M_STATIC > 7) a7 = d_reduce32_dpp(a7);

    if (l32 == 0) {
        if constexpr (OUT_F32) {
            float* yp = reinterpret_cast<float*>(y_out);
            yp[size_t(0) * size_t(N) + size_t(n)] = a0;
            if constexpr (M_STATIC > 1) yp[size_t(1) * size_t(N) + size_t(n)] = a1;
            if constexpr (M_STATIC > 2) yp[size_t(2) * size_t(N) + size_t(n)] = a2;
            if constexpr (M_STATIC > 3) yp[size_t(3) * size_t(N) + size_t(n)] = a3;
            if constexpr (M_STATIC > 4) yp[size_t(4) * size_t(N) + size_t(n)] = a4;
            if constexpr (M_STATIC > 5) yp[size_t(5) * size_t(N) + size_t(n)] = a5;
            if constexpr (M_STATIC > 6) yp[size_t(6) * size_t(N) + size_t(n)] = a6;
            if constexpr (M_STATIC > 7) yp[size_t(7) * size_t(N) + size_t(n)] = a7;
        } else {
            uint16_t* yp = reinterpret_cast<uint16_t*>(y_out);
            yp[size_t(0) * size_t(N) + size_t(n)] = d_f32_to_bf16(a0);
            if constexpr (M_STATIC > 1) yp[size_t(1) * size_t(N) + size_t(n)] = d_f32_to_bf16(a1);
            if constexpr (M_STATIC > 2) yp[size_t(2) * size_t(N) + size_t(n)] = d_f32_to_bf16(a2);
            if constexpr (M_STATIC > 3) yp[size_t(3) * size_t(N) + size_t(n)] = d_f32_to_bf16(a3);
            if constexpr (M_STATIC > 4) yp[size_t(4) * size_t(N) + size_t(n)] = d_f32_to_bf16(a4);
            if constexpr (M_STATIC > 5) yp[size_t(5) * size_t(N) + size_t(n)] = d_f32_to_bf16(a5);
            if constexpr (M_STATIC > 6) yp[size_t(6) * size_t(N) + size_t(n)] = d_f32_to_bf16(a6);
            if constexpr (M_STATIC > 7) yp[size_t(7) * size_t(N) + size_t(n)] = d_f32_to_bf16(a7);
        }
    }
}

// Wave-8 Fused BF16-in HCA Grouped wo_a + LRGA (8 rows/block, LDS-tiled o_bf16 -> 0 bf16_to_f16 launch!)
template <int M_STATIC>
__global__ __launch_bounds__(256, 3)
void k_hca_grouped_wo_a_lrga_bf16_f16_ws(
    const uint16_t* __restrict__ o_bf16,
    const uint16_t* __restrict__ wo_a_f16,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* __restrict__ z_out_bf16,
    int n_groups, int o_lora_rank)
{
    __shared__ uint4 s_x[M_STATIC][128];
    const int tid = int(threadIdx.x);
    const int row_loc = tid >> 5; // 0..7
    const int l32     = tid & 31; // 0..31
    const int r = int(blockIdx.x) * 8 + row_loc;
    const int g = int(blockIdx.y);
    const bool valid = (r < o_lora_rank && g < n_groups);

    const size_t o_tok_stride = size_t(n_groups) * 4096u;
    const size_t z_tok_stride = size_t(n_groups) * size_t(o_lora_rank);
    const uint4* __restrict__ w4 = valid ? reinterpret_cast<const uint4*>(
        wo_a_f16 + (size_t(g) * size_t(o_lora_rank) + size_t(r)) * 4096u) : nullptr;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    for (int k0 = 0; k0 < 512; k0 += 128) {
        for (int idx = tid; idx < M_STATIC * 128; idx += 256) {
            const int m = idx >> 7;
            const int k = idx & 127;
            const uint4 bv = reinterpret_cast<const uint4*>(o_bf16 + size_t(m) * o_tok_stride + size_t(g) * 4096u)[k0 + k];
            s_x[m][k] = d_bf16_u4_to_f16_u4(bv);
        }
        __syncthreads();
        if (valid) {
            for (int k = l32; k < 128; k += 32) {
                const uint4 wv = w4[k0 + k];
                a0 = dot8_f16_u4_fdot2(s_x[0][k], wv, a0);
                if constexpr (M_STATIC > 1) a1 = dot8_f16_u4_fdot2(s_x[1][k], wv, a1);
                if constexpr (M_STATIC > 2) a2 = dot8_f16_u4_fdot2(s_x[2][k], wv, a2);
                if constexpr (M_STATIC > 3) a3 = dot8_f16_u4_fdot2(s_x[3][k], wv, a3);
                if constexpr (M_STATIC > 4) a4 = dot8_f16_u4_fdot2(s_x[4][k], wv, a4);
                if constexpr (M_STATIC > 5) a5 = dot8_f16_u4_fdot2(s_x[5][k], wv, a5);
                if constexpr (M_STATIC > 6) a6 = dot8_f16_u4_fdot2(s_x[6][k], wv, a6);
                if constexpr (M_STATIC > 7) a7 = dot8_f16_u4_fdot2(s_x[7][k], wv, a7);
            }
        }
        __syncthreads();
    }

    if (!valid) return;
    a0 = d_reduce32_dpp(a0);
    if constexpr (M_STATIC > 1) a1 = d_reduce32_dpp(a1);
    if constexpr (M_STATIC > 2) a2 = d_reduce32_dpp(a2);
    if constexpr (M_STATIC > 3) a3 = d_reduce32_dpp(a3);
    if constexpr (M_STATIC > 4) a4 = d_reduce32_dpp(a4);
    if constexpr (M_STATIC > 5) a5 = d_reduce32_dpp(a5);
    if constexpr (M_STATIC > 6) a6 = d_reduce32_dpp(a6);
    if constexpr (M_STATIC > 7) a7 = d_reduce32_dpp(a7);

    if (l32 == 0) {
        auto finish = [&](int m, float val) {
            const size_t idx = size_t(m) * z_tok_stride + size_t(g) * size_t(o_lora_rank) + size_t(r);
            const float gv = d_bf16_to_f32(gate_bf16[idx]);
            z_out_bf16[idx] = d_f32_to_bf16(val * (1.0f / (1.0f + __expf(-gv))));
        };
        finish(0, a0);
        if constexpr (M_STATIC > 1) finish(1, a1);
        if constexpr (M_STATIC > 2) finish(2, a2);
        if constexpr (M_STATIC > 3) finish(3, a3);
        if constexpr (M_STATIC > 4) finish(4, a4);
        if constexpr (M_STATIC > 5) finish(5, a5);
        if constexpr (M_STATIC > 6) finish(6, a6);
        if constexpr (M_STATIC > 7) finish(7, a7);
    }
}

// Wave-8 Draft-Only FP8 E4M3FN Per-Row Scaled fdot2 GEMV (2x less HBM2 traffic for DSpark lm_head!)
__device__ __forceinline__ uint16_t d_fp8_e4m3fn_to_f16(uint32_t b) {
    const uint32_t s = (b >> 7) & 1u;
    const uint32_t e = (b >> 3) & 15u;
    const uint32_t m = b & 7u;
    float v = 0.0f;
    if (e == 0u) {
        v = float(m) * (1.0f / 512.0f);
        if (s) v = -v;
    } else if (!(e == 15u && m == 7u)) {
        const uint32_t bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
        v = __uint_as_float(bits);
    }
    return d_f32_to_f16(v);
}

template <int M_STATIC>
__global__ __launch_bounds__(256, 3)
void k_bf16x_fp8w_gemv_r8_f32(const uint16_t* __restrict__ x_bf16,
                              const uint8_t* __restrict__ w_fp8,
                              const float* __restrict__ row_scale,
                              float* __restrict__ y_out,
                              int N, int K) {
    __shared__ uint16_t s_lut[256];
    __shared__ uint4 s_x[M_STATIC][128];
    const int tid = int(threadIdx.x);
    s_lut[tid] = d_fp8_e4m3fn_to_f16(uint32_t(tid));

    const int row_loc = tid >> 5; // 0..7
    const int l32     = tid & 31; // 0..31
    const int n       = int(blockIdx.x) * 8 + row_loc;
    const bool valid  = (n < N);
    const int K8      = K >> 3;
    const uint2* __restrict__ w2 = valid ? reinterpret_cast<const uint2*>(w_fp8 + size_t(n) * size_t(K)) : nullptr;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    for (int k0 = 0; k0 < K8; k0 += 128) {
        const int t_len = (K8 - k0 < 128) ? (K8 - k0) : 128;
        for (int idx = tid; idx < M_STATIC * t_len; idx += 256) {
            const int m = (t_len == 128) ? (idx >> 7) : (idx / t_len);
            const int k = (t_len == 128) ? (idx & 127) : (idx - m * t_len);
            const uint4 bv = reinterpret_cast<const uint4*>(x_bf16 + size_t(m) * size_t(K))[k0 + k];
            s_x[m][k] = d_bf16_u4_to_f16_u4(bv);
        }
        __syncthreads();

        if (valid) {
            for (int k = l32; k < t_len; k += 32) {
                const uint2 q = w2[k0 + k];
                uint4 wv;
                wv.x = uint32_t(s_lut[q.x & 0xFFu])         | (uint32_t(s_lut[(q.x >> 8)  & 0xFFu]) << 16);
                wv.y = uint32_t(s_lut[(q.x >> 16) & 0xFFu]) | (uint32_t(s_lut[q.x >> 24])           << 16);
                wv.z = uint32_t(s_lut[q.y & 0xFFu])         | (uint32_t(s_lut[(q.y >> 8)  & 0xFFu]) << 16);
                wv.w = uint32_t(s_lut[(q.y >> 16) & 0xFFu]) | (uint32_t(s_lut[q.y >> 24])           << 16);

                a0 = dot8_f16_u4_fdot2(s_x[0][k], wv, a0);
                if constexpr (M_STATIC > 1) a1 = dot8_f16_u4_fdot2(s_x[1][k], wv, a1);
                if constexpr (M_STATIC > 2) a2 = dot8_f16_u4_fdot2(s_x[2][k], wv, a2);
                if constexpr (M_STATIC > 3) a3 = dot8_f16_u4_fdot2(s_x[3][k], wv, a3);
                if constexpr (M_STATIC > 4) a4 = dot8_f16_u4_fdot2(s_x[4][k], wv, a4);
                if constexpr (M_STATIC > 5) a5 = dot8_f16_u4_fdot2(s_x[5][k], wv, a5);
                if constexpr (M_STATIC > 6) a6 = dot8_f16_u4_fdot2(s_x[6][k], wv, a6);
                if constexpr (M_STATIC > 7) a7 = dot8_f16_u4_fdot2(s_x[7][k], wv, a7);
            }
        }
        __syncthreads();
    }

    if (!valid) return;
    const float sc = row_scale[n];
    a0 = d_reduce32_dpp(a0) * sc;
    if constexpr (M_STATIC > 1) a1 = d_reduce32_dpp(a1) * sc;
    if constexpr (M_STATIC > 2) a2 = d_reduce32_dpp(a2) * sc;
    if constexpr (M_STATIC > 3) a3 = d_reduce32_dpp(a3) * sc;
    if constexpr (M_STATIC > 4) a4 = d_reduce32_dpp(a4) * sc;
    if constexpr (M_STATIC > 5) a5 = d_reduce32_dpp(a5) * sc;
    if constexpr (M_STATIC > 6) a6 = d_reduce32_dpp(a6) * sc;
    if constexpr (M_STATIC > 7) a7 = d_reduce32_dpp(a7) * sc;

    if (l32 == 0) {
        y_out[size_t(0) * size_t(N) + size_t(n)] = a0;
        if constexpr (M_STATIC > 1) y_out[size_t(1) * size_t(N) + size_t(n)] = a1;
        if constexpr (M_STATIC > 2) y_out[size_t(2) * size_t(N) + size_t(n)] = a2;
        if constexpr (M_STATIC > 3) y_out[size_t(3) * size_t(N) + size_t(n)] = a3;
        if constexpr (M_STATIC > 4) y_out[size_t(4) * size_t(N) + size_t(n)] = a4;
        if constexpr (M_STATIC > 5) y_out[size_t(5) * size_t(N) + size_t(n)] = a5;
        if constexpr (M_STATIC > 6) y_out[size_t(6) * size_t(N) + size_t(n)] = a6;
        if constexpr (M_STATIC > 7) y_out[size_t(7) * size_t(N) + size_t(n)] = a7;
    }
}

} // namespace

void launch_bf16_to_f16(const uint16_t* d_in_bf16, uint16_t* d_out_f16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n4 = (total + 3) / 4;
    hipLaunchKernelGGL(k_bf16_to_f16_vec, dim3((n4 + 255) / 256), dim3(256), 0, stream, d_in_bf16, d_out_f16, total);
}

void launch_f16_gemv_bf16(const uint16_t* d_x_f16, const uint16_t* d_w_f16,
                          uint16_t* d_y_bf16, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + 3) / 4);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_f16 + size_t(off) * size_t(K);
        uint16_t* yp       = d_y_bf16 + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_f16_gemv_ws<1, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_f16_gemv_ws<2, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_f16_gemv_ws<3, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_f16_gemv_ws<4, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_f16_gemv_ws<5, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_f16_gemv_ws<6, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_f16_gemv_ws<7, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            default: hipLaunchKernelGGL((k_f16_gemv_ws<8, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
        }
    }
}

void launch_f16_gemv_f32(const uint16_t* d_x_f16, const uint16_t* d_w_f16,
                         float* d_y_f32, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    if (M == 1 && K == 256) {
        hipLaunchKernelGGL(k_f16_gemv_k256_m1_f32, dim3((N + 7) / 8), dim3(256), 0, stream, d_x_f16, d_w_f16, d_y_f32, N);
        return;
    }
    const dim3 grid((N + 3) / 4);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_f16 + size_t(off) * size_t(K);
        float* yp          = d_y_f32  + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_f16_gemv_ws<1, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_f16_gemv_ws<2, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_f16_gemv_ws<3, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_f16_gemv_ws<4, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_f16_gemv_ws<5, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_f16_gemv_ws<6, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_f16_gemv_ws<7, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            default: hipLaunchKernelGGL((k_f16_gemv_ws<8, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
        }
    }
}

void launch_hca_grouped_wo_a_lrga_f16(const uint16_t* d_o_f16, const uint16_t* d_wo_a_f16,
                                      const uint16_t* d_gate_bf16, uint16_t* d_z_out_bf16,
                                      int M, int n_groups, int o_lora_rank, hipStream_t stream) {
    if (M <= 0 || n_groups <= 0 || o_lora_rank <= 0) return;
    const dim3 grid((o_lora_rank + 3) / 4, n_groups);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* op = d_o_f16    + size_t(off) * size_t(n_groups) * 4096u;
        const uint16_t* gp = d_gate_bf16 + size_t(off) * size_t(n_groups) * size_t(o_lora_rank);
        uint16_t* zp       = d_z_out_bf16 + size_t(off) * size_t(n_groups) * size_t(o_lora_rank);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<1>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 2: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<2>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 3: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<3>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 4: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<4>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 5: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<5>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 6: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<6>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 7: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<7>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            default: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_f16_ws<8>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
        }
    }
}

void launch_bf16x_f16w_gemv_bf16(const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                 uint16_t* d_y_bf16, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + 7) / 8);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_bf16 + size_t(off) * size_t(K);
        uint16_t* yp       = d_y_bf16 + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<1, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<2, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<3, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<4, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<5, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<6, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<7, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            default: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<8, false>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
        }
    }
}

void launch_bf16x_f16w_gemv_f32(const uint16_t* d_x_bf16, const uint16_t* d_w_f16,
                                float* d_y_f32, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    if (M == 1 && K == 256) {
        hipLaunchKernelGGL(k_bf16x_f16w_gemv_k256_m1_f32, dim3((N + 7) / 8), dim3(256), 0, stream, d_x_bf16, d_w_f16, d_y_f32, N);
        return;
    }
    const dim3 grid((N + 7) / 8);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_bf16 + size_t(off) * size_t(K);
        float* yp          = d_y_f32  + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<1, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<2, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<3, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<4, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<5, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<6, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<7, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
            default: hipLaunchKernelGGL((k_bf16x_f16w_gemv_r8_ws<8, true>), grid, dim3(256), 0, stream, xp, d_w_f16, yp, N, K); break;
        }
    }
}

void launch_hca_grouped_wo_a_lrga_bf16_f16(const uint16_t* d_o_bf16, const uint16_t* d_wo_a_f16,
                                           const uint16_t* d_gate_bf16, uint16_t* d_z_out_bf16,
                                           int M, int n_groups, int o_lora_rank, hipStream_t stream) {
    if (M <= 0 || n_groups <= 0 || o_lora_rank <= 0) return;
    const dim3 grid((o_lora_rank + 7) / 8, n_groups);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* op = d_o_bf16     + size_t(off) * size_t(n_groups) * 4096u;
        const uint16_t* gp = d_gate_bf16  + size_t(off) * size_t(n_groups) * size_t(o_lora_rank);
        uint16_t* zp       = d_z_out_bf16 + size_t(off) * size_t(n_groups) * size_t(o_lora_rank);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<1>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 2: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<2>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 3: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<3>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 4: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<4>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 5: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<5>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 6: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<6>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            case 7: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<7>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
            default: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_bf16_f16_ws<8>), grid, dim3(256), 0, stream, op, d_wo_a_f16, gp, zp, n_groups, o_lora_rank); break;
        }
    }
}

void launch_bf16x_fp8w_gemv_f32(const uint16_t* d_x_bf16, const uint8_t* d_w_fp8,
                                const float* d_row_scale, float* d_y_f32,
                                int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + 7) / 8);
    for (int off = 0; off < M; off += 8) {
        const int rem = std::min(8, M - off);
        const uint16_t* xp = d_x_bf16 + size_t(off) * size_t(K);
        float* yp          = d_y_f32  + size_t(off) * size_t(N);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<1>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            case 2: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<2>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            case 3: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<3>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            case 4: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<4>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            case 5: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<5>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            case 6: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<6>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            case 7: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<7>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
            default: hipLaunchKernelGGL((k_bf16x_fp8w_gemv_r8_f32<8>), grid, dim3(256), 0, stream, xp, d_w_fp8, d_row_scale, yp, N, K); break;
        }
    }
}

namespace {

__device__ __forceinline__ float d_fdot2_u32(uint32_t a_u32, uint32_t b_u32, float acc) {
    return __builtin_amdgcn_fdot2(d_u32_to_f16x2(a_u32), d_u32_to_f16x2(b_u32), acc, false);
}

__device__ __forceinline__ f16x2_t d_splat_f16x2_scaled(uint16_t raw_f16, float boost) {
    const float v = d_f16_to_f32(raw_f16) * boost;
    const uint16_t h = d_f32_to_f16(v);
    return d_pack_f16x2(h, h);
}

__device__ __forceinline__ uint32_t d_f16x2_mul_u32(uint32_t a, f16x2_t s) {
    f16x2_t vr = d_u32_to_f16x2(a) * s;
    union { f16x2_t v; uint32_t u32; } cvt;
    cvt.v = vr;
    return cvt.u32;
}

// ---- Commit 2: On-GPU Two-Pass Expert CSR Builder (Full tile_m tiles first, then partial tail!) ----
__global__ __launch_bounds__(256, 4)
void k_ernie55_build_expert_csr_prefill(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const int16_t* __restrict__ global_to_local,
    int M, int max_local_slot, int tile_m,
    int max_tiles, int max_pairs,
    int32_t* __restrict__ d_csr_counts,
    Ernie55MoeWorkTile* __restrict__ d_tiles,
    Ernie55MoeRoutedPair* __restrict__ d_pairs,
    int hybrid_buckets)
{
    __shared__ int32_t s_cnt[256];
    __shared__ int32_t s_cur[256];
    const int tid = int(threadIdx.x);
    const int n_exp = (max_local_slot < 256) ? max_local_slot : 256;
    const int tm = (tile_m > 0 && tile_m <= 64) ? tile_m : 64;
    const int total = M * 10;

    s_cnt[tid] = 0;
    __syncthreads();

    for (int i = tid; i < total; i += 256) {
        const int32_t eid = topk_ids[i];
        const float wt = topk_weights[i];
        if (wt != 0.0f && uint32_t(eid) < 512u) {
            const int le = int(global_to_local[eid]);
            if (le >= 0 && le < n_exp) {
                atomicAdd(&s_cnt[le], 1);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < n_exp; ++e) {
            s_cur[e] = acc;
            acc += s_cnt[e];
        }
        if (hybrid_buckets && tm == 64) {
            // Commit 4: three expert-sorted buckets inside one fixed tile array.
            //   bucket 0 [0, max_tiles/2)          : full 64-row chunks + 49..63 tails -> m64
            //   bucket 1 [max_tiles/2, 3*max/4)    : 9..32-row chunks (strictly <= 32) -> m32
            //   bucket 2 [3*max_tiles/4, max_tiles): 1..8-row tails (strictly <= 8)    -> grouped m8
            // Fixed regions keep every bucket at a compile-time-known offset, so the
            // host never has to read back a tile count to launch the next bucket.
            const int b0_beg = 0,                   b0_end = max_tiles / 2;
            const int b1_beg = max_tiles / 2,       b1_end = (max_tiles * 3) / 4;
            const int b2_beg = (max_tiles * 3) / 4, b2_end = max_tiles;
            int n64 = 0, n32 = 0, n8 = 0;
            // Bucket 0 pass 1: full 64-row tiles first for maximum uniform wavefront occupancy
            for (int e = 0; e < n_exp; ++e) {
                int rem = s_cnt[e];
                int pos = s_cur[e];
                if (pos >= max_pairs) continue;
                if (pos + rem > max_pairs) rem = max_pairs - pos;
                while (rem >= 64 && b0_beg + n64 < b0_end) {
                    d_tiles[b0_beg + n64++] = Ernie55MoeWorkTile{e, pos, 64, 0};
                    pos += 64;
                    rem -= 64;
                }
            }
            // Tail decomposition (1..63 rows per expert):
            //   49..63 -> 1x m64 tile (>=76.6% full, single launch)
            //   32..48 -> 1x full m32 (32 rows) + remainder 0..16 -> m32 (9..16) or m8 (1..8)
            //    9..31 -> 1x m32 tile (strictly <= 32 rows, zero truncation)
            //    1.. 8 -> 1x m8 group (strictly <= kMaxGroupTokens, <=1 group per expert)
            for (int e = 0; e < n_exp; ++e) {
                int rem = s_cnt[e];
                int pos = s_cur[e];
                if (pos >= max_pairs) continue;
                if (pos + rem > max_pairs) rem = max_pairs - pos;
                const int full_done = (rem / 64) * 64;
                pos += full_done;
                rem -= full_done;
                if (rem <= 0) continue;
                if (rem >= 49) {
                    if (b0_beg + n64 < b0_end) {
                        d_tiles[b0_beg + n64++] = Ernie55MoeWorkTile{e, pos, rem, 0};
                    }
                    continue;
                }
                if (rem >= 32) {
                    if (b1_beg + n32 < b1_end) {
                        d_tiles[b1_beg + n32++] = Ernie55MoeWorkTile{e, pos, 32, 1};
                    }
                    pos += 32;
                    rem -= 32;
                }
                if (rem >= 9) {
                    if (b1_beg + n32 < b1_end) {
                        d_tiles[b1_beg + n32++] = Ernie55MoeWorkTile{e, pos, rem, 1};
                    }
                } else if (rem >= 1) {
                    if (b2_beg + n8 < b2_end) {
                        d_tiles[b2_beg + n8++] = Ernie55MoeWorkTile{e, pos, rem, 2};
                    }
                }
            }
            d_csr_counts[0] = n64;
            d_csr_counts[1] = (acc < max_pairs) ? acc : max_pairs;
            d_csr_counts[2] = n32;
            d_csr_counts[3] = n8;
        } else {
            int nt = 0;
            for (int e = 0; e < n_exp; ++e) {
                int rem = s_cnt[e];
                int pos = s_cur[e];
                while (rem >= tm && nt < max_tiles && pos + tm <= max_pairs) {
                    d_tiles[nt++] = Ernie55MoeWorkTile{e, pos, tm, 0};
                    pos += tm;
                    rem -= tm;
                }
            }
            for (int e = 0; e < n_exp; ++e) {
                const int c = s_cnt[e];
                const int full_done = (c / tm) * tm;
                int rem = c - full_done;
                int pos = s_cur[e] + full_done;
                if (pos >= max_pairs) continue;
                if (pos + rem > max_pairs) rem = max_pairs - pos;
                if (rem > 0 && nt < max_tiles) {
                    d_tiles[nt++] = Ernie55MoeWorkTile{e, pos, rem, 0};
                }
            }
            d_csr_counts[0] = nt;
            d_csr_counts[1] = (acc < max_pairs) ? acc : max_pairs;
            d_csr_counts[2] = 0;
            d_csr_counts[3] = 0;
        }
    }
    __syncthreads();

    for (int i = tid; i < total; i += 256) {
        const int32_t eid = topk_ids[i];
        const float wt = topk_weights[i];
        if (wt != 0.0f && uint32_t(eid) < 512u) {
            const int le = int(global_to_local[eid]);
            if (le >= 0 && le < n_exp) {
                const int pos = atomicAdd(&s_cur[le], 1);
                if (pos < max_pairs) {
                    d_pairs[pos] = Ernie55MoeRoutedPair{i / 10, le, wt, 0};
                }
            }
        }
    }
}

// ---- Commit 4: CSR m8 tail tiles -> ExpertDispatchGroup (device-side, 0 B extra) ----
__global__ __launch_bounds__(256, 4)
void k_ernie55_build_groups_from_csr_m8(
    const int32_t* __restrict__ d_csr_counts,
    const Ernie55MoeWorkTile* __restrict__ d_tiles_m8,
    const Ernie55MoeRoutedPair* __restrict__ d_pairs,
    ExpertDispatchGroup* __restrict__ d_groups,
    int max_groups)
{
    const int tid = int(threadIdx.x);
    const int n8 = d_csr_counts[3];
    const int n_groups = (n8 < max_groups) ? n8 : max_groups;
    // Single race-free pass: active groups populated, unused slots parked inert (expert = -1, count = 0).
    for (int t = tid; t < max_groups; t += 256) {
        ExpertDispatchGroup g{};
        if (t < n_groups) {
            const Ernie55MoeWorkTile tl = d_tiles_m8[t];
            g.expert = tl.le;
            g.count  = tl.cnt;
            for (int j = 0; j < kMaxGroupTokens; ++j) {
                if (j < tl.cnt) {
                    const Ernie55MoeRoutedPair pr = d_pairs[tl.p_beg + j];
                    g.rows[j]    = pr.row;
                    g.weights[j] = pr.weight;
                    g.slots[j]   = 0;
                }
            }
        } else {
            g.expert = -1;
            g.count  = 0;
        }
        d_groups[t] = g;
    }
}

// ---- Commit 2: On-Chip LDS SRAM BM=64 x BN=32 x BK=32 Fused Gate+Up+SiTU-GLU (10.25 KiB LDS, 16 acc, 4 WGs/CU, 0 spills!) ----
__global__ __launch_bounds__(256, 4)
void k_hlut4_prefill_situ_f16_m64(
    const uint16_t* __restrict__ x_rot_f16,
    const int32_t* __restrict__ d_csr_counts,
    const Ernie55MoeWorkTile* __restrict__ d_tiles,
    const Ernie55MoeRoutedPair* __restrict__ d_pairs,
    const Hlut4Tile32x32* __restrict__ w1_tiles,
    const Hlut4Tile32x32* __restrict__ w3_tiles,
    const uint16_t* __restrict__ levels8_f16,
    uint16_t* __restrict__ z_f16,
    int N, int K, float situ_beta, float situ_linear_beta)
{
    __shared__ uint32_t s_lut256_h2[256]; // 1,024 B
    __shared__ int32_t  s_rows[64];       // 256 B
    __shared__ uint4    s_x[64][4];       // 4,096 B (all 256 threads load 1 uint4 each!)
    __shared__ uint4    s_wg[32][5];      // 2,560 B (+1 pad -> bank-conflict-free ds_read_b128)
    __shared__ uint4    s_wu[32][5];      // 2,560 B (+1 pad -> bank-conflict-free ds_read_b128)

    const int n_tiles = d_csr_counts[0];
    if (int(blockIdx.y) >= n_tiles) return;

    const int tid = int(threadIdx.x);
    {
        const uint32_t lo = uint32_t(tid) & 0xFu;
        const uint32_t hi = uint32_t(tid) >> 4;
        const uint16_t h0 = levels8_f16[lo & 7u] ^ ((lo & 8u) ? 0x8000u : 0u);
        const uint16_t h1 = levels8_f16[hi & 7u] ^ ((hi & 8u) ? 0x8000u : 0u);
        s_lut256_h2[tid] = uint32_t(h0) | (uint32_t(h1) << 16);
    }
    __syncthreads();

    const int nt = int(blockIdx.x);
    const int nt_count = N >> 5;
    const int kt_count = K >> 5;
    const int k_u4     = K >> 3;
    if (nt >= nt_count) return;

    const bool is_up = (tid >= 128);
    const int rloc   = (tid & 127) >> 2;
    const int wc     = tid & 3;
    const int sg     = rloc >> 2;

    const int ld_t = tid >> 2;
    const int ld_c = tid & 3;
    const int tm   = tid >> 5;
    const int tn   = tid & 31;
    const uint4* __restrict__ x_all4 = reinterpret_cast<const uint4*>(x_rot_f16);
    const size_t col = size_t(nt * 32 + tn);

    #pragma unroll 1
    for (int tile_id = int(blockIdx.y); tile_id < n_tiles; tile_id += int(gridDim.y)) {
        const Ernie55MoeWorkTile wt = d_tiles[tile_id];
        const int cnt = wt.cnt;
        if (cnt <= 0) continue;

        if (tid < 64) s_rows[tid] = (tid < cnt) ? d_pairs[wt.p_beg + tid].row : -1;
        __syncthreads();

        const uint4* x_ld_ptr = nullptr;
        {
            const int r = s_rows[ld_t];
            if (r >= 0) x_ld_ptr = x_all4 + size_t(r) * size_t(k_u4) + size_t(ld_c);
        }

        const size_t base_off = (size_t(wt.le) * size_t(nt_count) + size_t(nt)) * size_t(kt_count);
        const Hlut4Tile32x32* w_row = is_up ? (w3_tiles + base_off) : (w1_tiles + base_off);

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float ag4 = 0.0f, ag5 = 0.0f, ag6 = 0.0f, ag7 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
        float au4 = 0.0f, au5 = 0.0f, au6 = 0.0f, au7 = 0.0f;

        const bool has_m1 = (cnt > 8);
        const bool has_m2 = (cnt > 16);
        const bool has_m3 = (cnt > 24);
        const bool has_m4 = (cnt > 32);
        const bool has_m5 = (cnt > 40);
        const bool has_m6 = (cnt > 48);
        const bool has_m7 = (cnt > 56);

        // Wave-13 Cut W13.2b: the kt+1 packed-weight prefetch is deliberately NOT applied here.
        // This kernel is compiled at exactly 64 VGPRs with __launch_bounds__(256, 4), and the
        // prefetch needs 2 more VGPRs (q_nib4 word + raw scale pair), which would push the
        // footprint to 66 VGPRs and drop occupancy from 4 WGs/CU to 3 WGs/CU. The 25% occupancy
        // loss outweighs the load-latency win, so the zero-extra-register form is kept and the
        // prefetch lands only in k_hlut4_prefill_down_atomic_m64 (69 -> 71 VGPRs, still 3 WGs/CU).
        #pragma unroll 1
        for (int kt = 0; kt < kt_count; ++kt) {
            s_x[ld_t][ld_c] = x_ld_ptr ? x_ld_ptr[size_t(kt) * 4u] : make_uint4(0u, 0u, 0u, 0u);
            {
                const Hlut4Tile32x32* tp = w_row + kt;
                const uint32_t wq = tp->q_nib4[rloc * 4 + wc];
                const f16x2_t sc = d_splat_f16x2_scaled(tp->scales_f16[sg], 64.0f);
                const uint4 dec = make_uint4(
                    d_f16x2_mul_u32(s_lut256_h2[wq & 0xFFu],         sc),
                    d_f16x2_mul_u32(s_lut256_h2[(wq >> 8) & 0xFFu],  sc),
                    d_f16x2_mul_u32(s_lut256_h2[(wq >> 16) & 0xFFu], sc),
                    d_f16x2_mul_u32(s_lut256_h2[wq >> 24],           sc));
                if (is_up) s_wu[rloc][wc] = dec;
                else       s_wg[rloc][wc] = dec;
            }
            __syncthreads();

            #pragma unroll 1
            for (int kc = 0; kc < 4; ++kc) {
                const uint4 wg = s_wg[tn][kc];
                const uint4 wu = s_wu[tn][kc];
                auto mac2 = [&](int r, float& g, float& u) {
                    const uint4 x = s_x[r][kc];
                    g = d_fdot2_u32(x.x, wg.x, g); g = d_fdot2_u32(x.y, wg.y, g);
                    g = d_fdot2_u32(x.z, wg.z, g); g = d_fdot2_u32(x.w, wg.w, g);
                    u = d_fdot2_u32(x.x, wu.x, u); u = d_fdot2_u32(x.y, wu.y, u);
                    u = d_fdot2_u32(x.z, wu.z, u); u = d_fdot2_u32(x.w, wu.w, u);
                };
                mac2(tm, ag0, au0);
                if (has_m1) mac2(tm + 8,  ag1, au1);
                if (has_m2) mac2(tm + 16, ag2, au2);
                if (has_m3) mac2(tm + 24, ag3, au3);
                if (has_m4) mac2(tm + 32, ag4, au4);
                if (has_m5) mac2(tm + 40, ag5, au5);
                if (has_m6) mac2(tm + 48, ag6, au6);
                if (has_m7) mac2(tm + 56, ag7, au7);
            }
            __syncthreads();
        }

        if (tm      < cnt) z_f16[size_t(wt.p_beg + tm     ) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag0 * 0.015625f, au0 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 8  < cnt) z_f16[size_t(wt.p_beg + tm + 8 ) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag1 * 0.015625f, au1 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 16 < cnt) z_f16[size_t(wt.p_beg + tm + 16) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag2 * 0.015625f, au2 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 24 < cnt) z_f16[size_t(wt.p_beg + tm + 24) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag3 * 0.015625f, au3 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 32 < cnt) z_f16[size_t(wt.p_beg + tm + 32) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag4 * 0.015625f, au4 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 40 < cnt) z_f16[size_t(wt.p_beg + tm + 40) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag5 * 0.015625f, au5 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 48 < cnt) z_f16[size_t(wt.p_beg + tm + 48) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag6 * 0.015625f, au6 * 0.015625f, situ_beta, situ_linear_beta));
        if (tm + 56 < cnt) z_f16[size_t(wt.p_beg + tm + 56) * size_t(N) + col] = d_f32_to_f16(d_situ_glu(ag7 * 0.015625f, au7 * 0.015625f, situ_beta, situ_linear_beta));
        __syncthreads();
    }
}

// ---- Commit 2: In-Place FP16 WHT128 on Routed Pair Intermediate z_f16[0..n_pairs-1, N] ----
__global__ __launch_bounds__(128, 8)
void k_wht128_f16_inplace_prefill_pairs(
    uint16_t* __restrict__ z_f16,
    const int32_t* __restrict__ d_csr_counts,
    const uint64_t* __restrict__ signs_i,
    int N)
{
    __shared__ float s_wht[128];
    const int np = d_csr_counts[1];
    const int blk128 = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int lane = tid & 63;
    const int wid  = tid >> 6;
    const bool neg = (signs_i != nullptr) && (((signs_i[blk128 * 2 + wid] >> lane) & 1ULL) != 0ULL);

    for (int p = int(blockIdx.y); p < np; p += int(gridDim.y)) {
        const size_t off = size_t(p) * size_t(N) + size_t(blk128 * 128 + tid);
        float v = d_f16_to_f32(z_f16[off]);
        if (neg) v = -v;
        #pragma unroll
        for (int len = 1; len <= 32; len <<= 1) {
            const float partner = __shfl_xor(v, len, 64);
            v = (lane & len) ? (partner - v) : (v + partner);
        }
        s_wht[tid] = v;
        __syncthreads();

        const float a = s_wht[lane];
        const float b = s_wht[64 + lane];
        const float y = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844f; // 1 / sqrt(128)
        z_f16[off] = d_f32_to_f16(y);
        __syncthreads();
    }
}

// ---- Commit 2: On-Chip LDS SRAM BM=64 x BN=64 x BK=32 Down + Fused Weighted FP32 atomicAdd (10.5 KiB LDS, 16 acc, 3 WGs/CU, 0 spills!) ----
__global__ __launch_bounds__(256, 3)
void k_hlut4_prefill_down_atomic_m64(
    const uint16_t* __restrict__ z_f16,
    const int32_t* __restrict__ d_csr_counts,
    const Ernie55MoeWorkTile* __restrict__ d_tiles,
    const Ernie55MoeRoutedPair* __restrict__ d_pairs,
    const Hlut4Tile32x32* __restrict__ w2_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ y_acc_f32,
    int N_down, int K_inter)
{
    __shared__ uint32_t s_lut256_h2[256]; // 1,024 B
    __shared__ int32_t  s_rows[64];       // 256 B
    __shared__ float    s_wts[64];        // 256 B
    __shared__ uint4    s_z[64][4];       // 4,096 B
    __shared__ uint4    s_wd[64][5];      // 5,120 B (+1 pad -> conflict-free ds_read_b128)

    const int n_tiles = d_csr_counts[0];
    if (int(blockIdx.y) >= n_tiles) return;

    const int tid = int(threadIdx.x);
    {
        const uint32_t lo = uint32_t(tid) & 0xFu;
        const uint32_t hi = uint32_t(tid) >> 4;
        const uint16_t h0 = levels8_f16[lo & 7u] ^ ((lo & 8u) ? 0x8000u : 0u);
        const uint16_t h1 = levels8_f16[hi & 7u] ^ ((hi & 8u) ? 0x8000u : 0u);
        s_lut256_h2[tid] = uint32_t(h0) | (uint32_t(h1) << 16);
    }
    __syncthreads();

    const int nt0 = int(blockIdx.x) * 2;
    const int nt1 = nt0 + 1;
    const int nt_count = N_down >> 5;
    const int kt_count = K_inter >> 5;
    const int k_u4     = K_inter >> 3;
    const bool has_nt1 = (nt1 < nt_count);
    if (nt0 >= nt_count) return;

    const int wr    = tid >> 2;
    const int wc    = tid & 3;
    const bool w_t1 = (wr >= 32);
    const int rloc  = wr & 31;
    const int sg    = rloc >> 2;
    const bool w_ok = (!w_t1 || has_nt1);

    const int tm = tid >> 5;
    const int tn = tid & 31;
    const uint4* __restrict__ z_all4 = reinterpret_cast<const uint4*>(z_f16);
    const size_t col0 = size_t(nt0 * 32 + tn);
    const size_t col1 = size_t(nt1 * 32 + tn);

    #pragma unroll 1
    for (int tile_id = int(blockIdx.y); tile_id < n_tiles; tile_id += int(gridDim.y)) {
        const Ernie55MoeWorkTile wt = d_tiles[tile_id];
        const int cnt = wt.cnt;
        if (cnt <= 0) continue;

        if (tid < 64) {
            if (tid < cnt) {
                const Ernie55MoeRoutedPair pr = d_pairs[wt.p_beg + tid];
                s_rows[tid] = pr.row;
                s_wts[tid]  = pr.weight;
            } else {
                s_rows[tid] = -1;
                s_wts[tid]  = 0.0f;
            }
        }
        __syncthreads();

        const uint4* z_ld_ptr = (wr < cnt) ? (z_all4 + size_t(wt.p_beg + wr) * size_t(k_u4) + size_t(wc)) : nullptr;
        const size_t base_off0 = (size_t(wt.le) * size_t(nt_count) + size_t(nt0)) * size_t(kt_count);
        const Hlut4Tile32x32* d_row = w_t1 ? (w2_tiles + base_off0 + size_t(kt_count)) : (w2_tiles + base_off0);

        float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
        float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
        float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
        float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

        const bool has_m1 = (cnt > 8);
        const bool has_m2 = (cnt > 16);
        const bool has_m3 = (cnt > 24);
        const bool has_m4 = (cnt > 32);
        const bool has_m5 = (cnt > 40);
        const bool has_m6 = (cnt > 48);
        const bool has_m7 = (cnt > 56);

        // Wave-13 Cut W13.2b: register-level kt+1 packed-weight prefetch.
        // The raw q_nib4 word and the raw scale byte pair of K-tile kt+1 are issued at the
        // TOP of iteration kt, so the global load latency of tile kt+1 is overlapped with the
        // 256-entry LUT decode and the 32 fdot2 FMAs of tile kt instead of sitting in front of
        // them. The prefetched values are consumed at the end of the iteration, which keeps the
        // single-buffered s_wd/s_z shared layout and the numerical order bit-exact.
        // Register cost: 2 VGPRs (69 -> 71), which stays inside the 84-VGPR / 3-WG-per-CU
        // headroom of this kernel, so occupancy, ScratchSize == 0 and the zero-spill contract
        // are all preserved.
        uint32_t wd_cur = 0u;
        uint32_t sc_cur = 0u;
        if (w_ok) {
            const Hlut4Tile32x32* tp0 = d_row;
            wd_cur = tp0->q_nib4[rloc * 4 + wc];
            sc_cur = uint32_t(tp0->scales_f16[sg]);
        }

        #pragma unroll 1
        for (int kt = 0; kt < kt_count; ++kt) {
            uint32_t wd_nxt = 0u;
            uint32_t sc_nxt = 0u;
            if (w_ok && (kt + 1 < kt_count)) {
                const Hlut4Tile32x32* tpn = d_row + kt + 1;
                wd_nxt = tpn->q_nib4[rloc * 4 + wc];
                sc_nxt = uint32_t(tpn->scales_f16[sg]);
            }
            s_z[wr][wc] = z_ld_ptr ? z_ld_ptr[size_t(kt) * 4u] : make_uint4(0u, 0u, 0u, 0u);
            if (w_ok) {
                const uint32_t wd = wd_cur;
                const f16x2_t scd = d_splat_f16x2_scaled(uint16_t(sc_cur), 64.0f);
                s_wd[wr][wc] = make_uint4(
                    d_f16x2_mul_u32(s_lut256_h2[wd & 0xFFu],         scd),
                    d_f16x2_mul_u32(s_lut256_h2[(wd >> 8) & 0xFFu],  scd),
                    d_f16x2_mul_u32(s_lut256_h2[(wd >> 16) & 0xFFu], scd),
                    d_f16x2_mul_u32(s_lut256_h2[wd >> 24],           scd));
            }
            __syncthreads();

            #pragma unroll 1
            for (int kc = 0; kc < 4; ++kc) {
                const uint4 w0 = s_wd[tn][kc];
                const uint4 w1 = has_nt1 ? s_wd[32 + tn][kc] : make_uint4(0u, 0u, 0u, 0u);
                auto mac2 = [&](int r, float& v0, float& v1) {
                    const uint4 z = s_z[r][kc];
                    v0 = d_fdot2_u32(z.x, w0.x, v0); v0 = d_fdot2_u32(z.y, w0.y, v0);
                    v0 = d_fdot2_u32(z.z, w0.z, v0); v0 = d_fdot2_u32(z.w, w0.w, v0);
                    if (has_nt1) {
                        v1 = d_fdot2_u32(z.x, w1.x, v1); v1 = d_fdot2_u32(z.y, w1.y, v1);
                        v1 = d_fdot2_u32(z.z, w1.z, v1); v1 = d_fdot2_u32(z.w, w1.w, v1);
                    }
                };
                mac2(tm, a0_0, a1_0);
                if (has_m1) mac2(tm + 8,  a0_1, a1_1);
                if (has_m2) mac2(tm + 16, a0_2, a1_2);
                if (has_m3) mac2(tm + 24, a0_3, a1_3);
                if (has_m4) mac2(tm + 32, a0_4, a1_4);
                if (has_m5) mac2(tm + 40, a0_5, a1_5);
                if (has_m6) mac2(tm + 48, a0_6, a1_6);
                if (has_m7) mac2(tm + 56, a0_7, a1_7);
            }
            __syncthreads();
            wd_cur = wd_nxt;
            sc_cur = sc_nxt;
        }

        auto emit_row = [&](int t, float v0, float v1) {
            if (t < cnt) {
                const int row = s_rows[t];
                if (row >= 0) {
                    const float w_sc = s_wts[t] * 0.015625f;
                    const size_t roff = size_t(row) * size_t(N_down);
                    atomicAdd(&y_acc_f32[roff + col0], v0 * w_sc);
                    if (has_nt1) atomicAdd(&y_acc_f32[roff + col1], v1 * w_sc);
                }
            }
        };
        emit_row(tm,      a0_0, a1_0);
        emit_row(tm + 8,  a0_1, a1_1);
        emit_row(tm + 16, a0_2, a1_2);
        emit_row(tm + 24, a0_3, a1_3);
        emit_row(tm + 32, a0_4, a1_4);
        emit_row(tm + 40, a0_5, a1_5);
        emit_row(tm + 48, a0_6, a1_6);
        emit_row(tm + 56, a0_7, a1_7);
        __syncthreads();
    }
}

// ---- Commit 2 Variant B: BM=32 x BN=64 x BK=32 Gate+Up+SiTU-GLU (13.4 KiB LDS, 16 acc, 3 WGs/CU, 0 spills!) ----
__global__ __launch_bounds__(256, 3)
void k_hlut4_prefill_situ_f16_m32(
    const uint16_t* __restrict__ x_rot_f16,
    const int32_t* __restrict__ d_csr_counts,
    const Ernie55MoeWorkTile* __restrict__ d_tiles,
    const Ernie55MoeRoutedPair* __restrict__ d_pairs,
    const Hlut4Tile32x32* __restrict__ w1_tiles,
    const Hlut4Tile32x32* __restrict__ w3_tiles,
    const uint16_t* __restrict__ levels8_f16,
    uint16_t* __restrict__ z_f16,
    int N, int K, float situ_beta, float situ_linear_beta)
{
    __shared__ uint32_t s_lut256_h2[256];
    __shared__ int32_t  s_rows[32];
    __shared__ uint4    s_x[32][4];
    __shared__ uint4    s_wg[64][5];
    __shared__ uint4    s_wu[64][5];

    const int n_tiles = d_csr_counts[0];
    if (int(blockIdx.y) >= n_tiles) return;

    const int tid = int(threadIdx.x);
    {
        const uint32_t lo = uint32_t(tid) & 0xFu;
        const uint32_t hi = uint32_t(tid) >> 4;
        const uint16_t h0 = levels8_f16[lo & 7u] ^ ((lo & 8u) ? 0x8000u : 0u);
        const uint16_t h1 = levels8_f16[hi & 7u] ^ ((hi & 8u) ? 0x8000u : 0u);
        s_lut256_h2[tid] = uint32_t(h0) | (uint32_t(h1) << 16);
    }
    __syncthreads();

    const int nt0 = int(blockIdx.x) * 2;
    const int nt1 = nt0 + 1;
    const int nt_count = N >> 5;
    const int kt_count = K >> 5;
    const int k_u4     = K >> 3;
    const bool has_nt1 = (nt1 < nt_count);
    if (nt0 >= nt_count) return;

    const int wr    = tid >> 2;
    const int wc    = tid & 3;
    const bool w_t1 = (wr >= 32);
    const int rloc  = wr & 31;
    const int sg    = rloc >> 2;
    const bool w_ok = (!w_t1 || has_nt1);

    const int tm = tid >> 5;
    const int tn = tid & 31;
    const uint4* __restrict__ x_all4 = reinterpret_cast<const uint4*>(x_rot_f16);
    const size_t col0 = size_t(nt0 * 32 + tn);
    const size_t col1 = size_t(nt1 * 32 + tn);

    #pragma unroll 1
    for (int tile_id = int(blockIdx.y); tile_id < n_tiles; tile_id += int(gridDim.y)) {
        const Ernie55MoeWorkTile wt = d_tiles[tile_id];
        const int cnt = wt.cnt;
        if (cnt <= 0) continue;

        if (tid < 32) s_rows[tid] = (tid < cnt) ? d_pairs[wt.p_beg + tid].row : -1;
        __syncthreads();

        const uint4* x_ld_ptr = nullptr;
        if (tid < 128) {
            const int r = s_rows[wr];
            if (r >= 0) x_ld_ptr = x_all4 + size_t(r) * size_t(k_u4) + size_t(wc);
        }

        const size_t base_off0 = (size_t(wt.le) * size_t(nt_count) + size_t(nt0)) * size_t(kt_count);
        const Hlut4Tile32x32* g_row = w_t1 ? (w1_tiles + base_off0 + size_t(kt_count)) : (w1_tiles + base_off0);
        const Hlut4Tile32x32* u_row = w_t1 ? (w3_tiles + base_off0 + size_t(kt_count)) : (w3_tiles + base_off0);

        float ag0_0 = 0.0f, ag0_1 = 0.0f, ag0_2 = 0.0f, ag0_3 = 0.0f;
        float ag1_0 = 0.0f, ag1_1 = 0.0f, ag1_2 = 0.0f, ag1_3 = 0.0f;
        float au0_0 = 0.0f, au0_1 = 0.0f, au0_2 = 0.0f, au0_3 = 0.0f;
        float au1_0 = 0.0f, au1_1 = 0.0f, au1_2 = 0.0f, au1_3 = 0.0f;

        const bool has_m1 = (cnt > 8);
        const bool has_m2 = (cnt > 16);
        const bool has_m3 = (cnt > 24);

        #pragma unroll 1
        for (int kt = 0; kt < kt_count; ++kt) {
            if (tid < 128) {
                s_x[wr][wc] = x_ld_ptr ? x_ld_ptr[size_t(kt) * 4u] : make_uint4(0u, 0u, 0u, 0u);
            }
            if (w_ok) {
                const Hlut4Tile32x32* tg = g_row + kt;
                const Hlut4Tile32x32* tu = u_row + kt;
                const uint32_t wg = tg->q_nib4[rloc * 4 + wc];
                const uint32_t wu = tu->q_nib4[rloc * 4 + wc];
                const f16x2_t scg = d_splat_f16x2_scaled(tg->scales_f16[sg], 64.0f);
                const f16x2_t scu = d_splat_f16x2_scaled(tu->scales_f16[sg], 64.0f);
                s_wg[wr][wc] = make_uint4(
                    d_f16x2_mul_u32(s_lut256_h2[wg & 0xFFu],         scg),
                    d_f16x2_mul_u32(s_lut256_h2[(wg >> 8) & 0xFFu],  scg),
                    d_f16x2_mul_u32(s_lut256_h2[(wg >> 16) & 0xFFu], scg),
                    d_f16x2_mul_u32(s_lut256_h2[wg >> 24],           scg));
                s_wu[wr][wc] = make_uint4(
                    d_f16x2_mul_u32(s_lut256_h2[wu & 0xFFu],         scu),
                    d_f16x2_mul_u32(s_lut256_h2[(wu >> 8) & 0xFFu],  scu),
                    d_f16x2_mul_u32(s_lut256_h2[(wu >> 16) & 0xFFu], scu),
                    d_f16x2_mul_u32(s_lut256_h2[wu >> 24],           scu));
            }
            __syncthreads();

            const int tn1 = 32 + tn;
            #pragma unroll 1
            for (int kc = 0; kc < 4; ++kc) {
                const uint4 x0 = s_x[tm][kc];
                const uint4 x1 = has_m1 ? s_x[tm + 8][kc]  : make_uint4(0u, 0u, 0u, 0u);
                const uint4 x2 = has_m2 ? s_x[tm + 16][kc] : make_uint4(0u, 0u, 0u, 0u);
                const uint4 x3 = has_m3 ? s_x[tm + 24][kc] : make_uint4(0u, 0u, 0u, 0u);
                {
                    const uint4 w = s_wg[tn][kc];
                    ag0_0 = d_fdot2_u32(x0.x, w.x, ag0_0); ag0_0 = d_fdot2_u32(x0.y, w.y, ag0_0); ag0_0 = d_fdot2_u32(x0.z, w.z, ag0_0); ag0_0 = d_fdot2_u32(x0.w, w.w, ag0_0);
                    if (has_m1) { ag0_1 = d_fdot2_u32(x1.x, w.x, ag0_1); ag0_1 = d_fdot2_u32(x1.y, w.y, ag0_1); ag0_1 = d_fdot2_u32(x1.z, w.z, ag0_1); ag0_1 = d_fdot2_u32(x1.w, w.w, ag0_1); }
                    if (has_m2) { ag0_2 = d_fdot2_u32(x2.x, w.x, ag0_2); ag0_2 = d_fdot2_u32(x2.y, w.y, ag0_2); ag0_2 = d_fdot2_u32(x2.z, w.z, ag0_2); ag0_2 = d_fdot2_u32(x2.w, w.w, ag0_2); }
                    if (has_m3) { ag0_3 = d_fdot2_u32(x3.x, w.x, ag0_3); ag0_3 = d_fdot2_u32(x3.y, w.y, ag0_3); ag0_3 = d_fdot2_u32(x3.z, w.z, ag0_3); ag0_3 = d_fdot2_u32(x3.w, w.w, ag0_3); }
                }
                {
                    const uint4 w = s_wu[tn][kc];
                    au0_0 = d_fdot2_u32(x0.x, w.x, au0_0); au0_0 = d_fdot2_u32(x0.y, w.y, au0_0); au0_0 = d_fdot2_u32(x0.z, w.z, au0_0); au0_0 = d_fdot2_u32(x0.w, w.w, au0_0);
                    if (has_m1) { au0_1 = d_fdot2_u32(x1.x, w.x, au0_1); au0_1 = d_fdot2_u32(x1.y, w.y, au0_1); au0_1 = d_fdot2_u32(x1.z, w.z, au0_1); au0_1 = d_fdot2_u32(x1.w, w.w, au0_1); }
                    if (has_m2) { au0_2 = d_fdot2_u32(x2.x, w.x, au0_2); au0_2 = d_fdot2_u32(x2.y, w.y, au0_2); au0_2 = d_fdot2_u32(x2.z, w.z, au0_2); au0_2 = d_fdot2_u32(x2.w, w.w, au0_2); }
                    if (has_m3) { au0_3 = d_fdot2_u32(x3.x, w.x, au0_3); au0_3 = d_fdot2_u32(x3.y, w.y, au0_3); au0_3 = d_fdot2_u32(x3.z, w.z, au0_3); au0_3 = d_fdot2_u32(x3.w, w.w, au0_3); }
                }
                if (has_nt1) {
                    {
                        const uint4 w = s_wg[tn1][kc];
                        ag1_0 = d_fdot2_u32(x0.x, w.x, ag1_0); ag1_0 = d_fdot2_u32(x0.y, w.y, ag1_0); ag1_0 = d_fdot2_u32(x0.z, w.z, ag1_0); ag1_0 = d_fdot2_u32(x0.w, w.w, ag1_0);
                        if (has_m1) { ag1_1 = d_fdot2_u32(x1.x, w.x, ag1_1); ag1_1 = d_fdot2_u32(x1.y, w.y, ag1_1); ag1_1 = d_fdot2_u32(x1.z, w.z, ag1_1); ag1_1 = d_fdot2_u32(x1.w, w.w, ag1_1); }
                        if (has_m2) { ag1_2 = d_fdot2_u32(x2.x, w.x, ag1_2); ag1_2 = d_fdot2_u32(x2.y, w.y, ag1_2); ag1_2 = d_fdot2_u32(x2.z, w.z, ag1_2); ag1_2 = d_fdot2_u32(x2.w, w.w, ag1_2); }
                        if (has_m3) { ag1_3 = d_fdot2_u32(x3.x, w.x, ag1_3); ag1_3 = d_fdot2_u32(x3.y, w.y, ag1_3); ag1_3 = d_fdot2_u32(x3.z, w.z, ag1_3); ag1_3 = d_fdot2_u32(x3.w, w.w, ag1_3); }
                    }
                    {
                        const uint4 w = s_wu[tn1][kc];
                        au1_0 = d_fdot2_u32(x0.x, w.x, au1_0); au1_0 = d_fdot2_u32(x0.y, w.y, au1_0); au1_0 = d_fdot2_u32(x0.z, w.z, au1_0); au1_0 = d_fdot2_u32(x0.w, w.w, au1_0);
                        if (has_m1) { au1_1 = d_fdot2_u32(x1.x, w.x, au1_1); au1_1 = d_fdot2_u32(x1.y, w.y, au1_1); au1_1 = d_fdot2_u32(x1.z, w.z, au1_1); au1_1 = d_fdot2_u32(x1.w, w.w, au1_1); }
                        if (has_m2) { au1_2 = d_fdot2_u32(x2.x, w.x, au1_2); au1_2 = d_fdot2_u32(x2.y, w.y, au1_2); au1_2 = d_fdot2_u32(x2.z, w.z, au1_2); au1_2 = d_fdot2_u32(x2.w, w.w, au1_2); }
                        if (has_m3) { au1_3 = d_fdot2_u32(x3.x, w.x, au1_3); au1_3 = d_fdot2_u32(x3.y, w.y, au1_3); au1_3 = d_fdot2_u32(x3.z, w.z, au1_3); au1_3 = d_fdot2_u32(x3.w, w.w, au1_3); }
                    }
                }
            }
            __syncthreads();
        }

        auto store_r = [&](int t, float g0, float u0, float g1, float u1) {
            if (t < cnt) {
                const size_t roff = size_t(wt.p_beg + t) * size_t(N);
                z_f16[roff + col0] = d_f32_to_f16(d_situ_glu(g0 * 0.015625f, u0 * 0.015625f, situ_beta, situ_linear_beta));
                if (has_nt1) z_f16[roff + col1] = d_f32_to_f16(d_situ_glu(g1 * 0.015625f, u1 * 0.015625f, situ_beta, situ_linear_beta));
            }
        };
        store_r(tm,      ag0_0, au0_0, ag1_0, au1_0);
        store_r(tm + 8,  ag0_1, au0_1, ag1_1, au1_1);
        store_r(tm + 16, ag0_2, au0_2, ag1_2, au1_2);
        store_r(tm + 24, ag0_3, au0_3, ag1_3, au1_3);
        __syncthreads();
    }
}

// ---- Commit 2 Variant B: BM=32 x BN=64 x BK=64 Down + Direct Weighted FP32 atomicAdd (14.6 KiB LDS, 8 acc, 4 WGs/CU, 0 spills!) ----
__global__ __launch_bounds__(256, 4)
void k_hlut4_prefill_down_atomic_m32(
    const uint16_t* __restrict__ z_f16,
    const int32_t* __restrict__ d_csr_counts,
    const Ernie55MoeWorkTile* __restrict__ d_tiles,
    const Ernie55MoeRoutedPair* __restrict__ d_pairs,
    const Hlut4Tile32x32* __restrict__ w2_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ y_acc_f32,
    int N_down, int K_inter)
{
    __shared__ uint32_t s_lut256_h2[256];
    __shared__ int32_t  s_rows[32];
    __shared__ float    s_wts[32];
    __shared__ uint4    s_z[32][8];
    __shared__ uint4    s_wd[64][9];

    const int n_tiles = d_csr_counts[0];
    if (int(blockIdx.y) >= n_tiles) return;

    const int tid = int(threadIdx.x);
    {
        const uint32_t lo = uint32_t(tid) & 0xFu;
        const uint32_t hi = uint32_t(tid) >> 4;
        const uint16_t h0 = levels8_f16[lo & 7u] ^ ((lo & 8u) ? 0x8000u : 0u);
        const uint16_t h1 = levels8_f16[hi & 7u] ^ ((hi & 8u) ? 0x8000u : 0u);
        s_lut256_h2[tid] = uint32_t(h0) | (uint32_t(h1) << 16);
    }
    __syncthreads();

    const int nt0 = int(blockIdx.x) * 2;
    const int nt1 = nt0 + 1;
    const int nt_count = N_down >> 5;
    const int kt_count = K_inter >> 5;
    const int k_u4     = K_inter >> 3;
    const bool has_nt1 = (nt1 < nt_count);
    if (nt0 >= nt_count) return;

    const int ld_t  = tid >> 3; // 0..31
    const int ld_c  = tid & 7;  // 0..7
    const int wr    = tid >> 2;
    const int wc    = tid & 3;
    const bool w_t1 = (wr >= 32);
    const int rloc  = wr & 31;
    const int sg    = rloc >> 2;
    const bool w_ok = (!w_t1 || has_nt1);

    const int tm = tid >> 5;
    const int tn = tid & 31;
    const uint4* __restrict__ z_all4 = reinterpret_cast<const uint4*>(z_f16);
    const size_t col0 = size_t(nt0 * 32 + tn);
    const size_t col1 = size_t(nt1 * 32 + tn);

    #pragma unroll 1
    for (int tile_id = int(blockIdx.y); tile_id < n_tiles; tile_id += int(gridDim.y)) {
        const Ernie55MoeWorkTile wt = d_tiles[tile_id];
        const int cnt = wt.cnt;
        if (cnt <= 0) continue;

        if (tid < 32) {
            if (tid < cnt) {
                const Ernie55MoeRoutedPair pr = d_pairs[wt.p_beg + tid];
                s_rows[tid] = pr.row;
                s_wts[tid]  = pr.weight;
            } else {
                s_rows[tid] = -1;
                s_wts[tid]  = 0.0f;
            }
        }
        __syncthreads();

        const uint4* z_ld_ptr = (ld_t < cnt) ? (z_all4 + size_t(wt.p_beg + ld_t) * size_t(k_u4) + size_t(ld_c)) : nullptr;
        const size_t base_off0 = (size_t(wt.le) * size_t(nt_count) + size_t(nt0)) * size_t(kt_count);
        const Hlut4Tile32x32* d_row = w_t1 ? (w2_tiles + base_off0 + size_t(kt_count)) : (w2_tiles + base_off0);

        float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
        float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
        const bool has_m1 = (cnt > 8);
        const bool has_m2 = (cnt > 16);
        const bool has_m3 = (cnt > 24);

        #pragma unroll 1
        for (int kt = 0; kt < kt_count; kt += 2) {
            s_z[ld_t][ld_c] = z_ld_ptr ? z_ld_ptr[size_t(kt) * 4u] : make_uint4(0u, 0u, 0u, 0u);
            if (w_ok) {
                #pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const Hlut4Tile32x32* tp = d_row + (kt + kk);
                    const uint32_t wd = tp->q_nib4[rloc * 4 + wc];
                    const f16x2_t scd = d_splat_f16x2_scaled(tp->scales_f16[sg], 64.0f);
                    s_wd[wr][kk * 4 + wc] = make_uint4(
                        d_f16x2_mul_u32(s_lut256_h2[wd & 0xFFu],         scd),
                        d_f16x2_mul_u32(s_lut256_h2[(wd >> 8) & 0xFFu],  scd),
                        d_f16x2_mul_u32(s_lut256_h2[(wd >> 16) & 0xFFu], scd),
                        d_f16x2_mul_u32(s_lut256_h2[wd >> 24],           scd));
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int kc = 0; kc < 8; ++kc) {
                const uint4 z0 = s_z[tm][kc];
                const uint4 z1 = has_m1 ? s_z[tm + 8][kc]  : make_uint4(0u, 0u, 0u, 0u);
                const uint4 z2 = has_m2 ? s_z[tm + 16][kc] : make_uint4(0u, 0u, 0u, 0u);
                const uint4 z3 = has_m3 ? s_z[tm + 24][kc] : make_uint4(0u, 0u, 0u, 0u);
                {
                    const uint4 w = s_wd[tn][kc];
                    a0_0 = d_fdot2_u32(z0.x, w.x, a0_0); a0_0 = d_fdot2_u32(z0.y, w.y, a0_0); a0_0 = d_fdot2_u32(z0.z, w.z, a0_0); a0_0 = d_fdot2_u32(z0.w, w.w, a0_0);
                    if (has_m1) { a0_1 = d_fdot2_u32(z1.x, w.x, a0_1); a0_1 = d_fdot2_u32(z1.y, w.y, a0_1); a0_1 = d_fdot2_u32(z1.z, w.z, a0_1); a0_1 = d_fdot2_u32(z1.w, w.w, a0_1); }
                    if (has_m2) { a0_2 = d_fdot2_u32(z2.x, w.x, a0_2); a0_2 = d_fdot2_u32(z2.y, w.y, a0_2); a0_2 = d_fdot2_u32(z2.z, w.z, a0_2); a0_2 = d_fdot2_u32(z2.w, w.w, a0_2); }
                    if (has_m3) { a0_3 = d_fdot2_u32(z3.x, w.x, a0_3); a0_3 = d_fdot2_u32(z3.y, w.y, a0_3); a0_3 = d_fdot2_u32(z3.z, w.z, a0_3); a0_3 = d_fdot2_u32(z3.w, w.w, a0_3); }
                }
                if (has_nt1) {
                    const uint4 w = s_wd[32 + tn][kc];
                    a1_0 = d_fdot2_u32(z0.x, w.x, a1_0); a1_0 = d_fdot2_u32(z0.y, w.y, a1_0); a1_0 = d_fdot2_u32(z0.z, w.z, a1_0); a1_0 = d_fdot2_u32(z0.w, w.w, a1_0);
                    if (has_m1) { a1_1 = d_fdot2_u32(z1.x, w.x, a1_1); a1_1 = d_fdot2_u32(z1.y, w.y, a1_1); a1_1 = d_fdot2_u32(z1.z, w.z, a1_1); a1_1 = d_fdot2_u32(z1.w, w.w, a1_1); }
                    if (has_m2) { a1_2 = d_fdot2_u32(z2.x, w.x, a1_2); a1_2 = d_fdot2_u32(z2.y, w.y, a1_2); a1_2 = d_fdot2_u32(z2.z, w.z, a1_2); a1_2 = d_fdot2_u32(z2.w, w.w, a1_2); }
                    if (has_m3) { a1_3 = d_fdot2_u32(z3.x, w.x, a1_3); a1_3 = d_fdot2_u32(z3.y, w.y, a1_3); a1_3 = d_fdot2_u32(z3.z, w.z, a1_3); a1_3 = d_fdot2_u32(z3.w, w.w, a1_3); }
                }
            }
            __syncthreads();
        }

        auto emit_r = [&](int t, float v0, float v1) {
            if (t < cnt) {
                const int row = s_rows[t];
                if (row >= 0) {
                    const float w_sc = s_wts[t] * 0.015625f;
                    const size_t roff = size_t(row) * size_t(N_down);
                    atomicAdd(&y_acc_f32[roff + col0], v0 * w_sc);
                    if (has_nt1) atomicAdd(&y_acc_f32[roff + col1], v1 * w_sc);
                }
            }
        };
        emit_r(tm,      a0_0, a1_0);
        emit_r(tm + 8,  a0_1, a1_1);
        emit_r(tm + 16, a0_2, a1_2);
        emit_r(tm + 24, a0_3, a1_3);
        __syncthreads();
    }
}

} // anonymous namespace

void launch_ernie55_build_expert_csr_prefill(
    const int32_t* d_topk_ids, const float* d_topk_weights, const int16_t* d_global_to_local,
    int M, int max_local_slot, int tile_m, int max_tiles, int max_pairs,
    int32_t* d_csr_counts, Ernie55MoeWorkTile* d_tiles, Ernie55MoeRoutedPair* d_pairs,
    hipStream_t stream, bool hybrid_buckets) {
    if (M <= 0 || max_local_slot <= 0) return;
    hipLaunchKernelGGL(k_ernie55_build_expert_csr_prefill, dim3(1), dim3(256), 0, stream,
                       d_topk_ids, d_topk_weights, d_global_to_local,
                       M, max_local_slot, tile_m, max_tiles, max_pairs,
                       d_csr_counts, d_tiles, d_pairs, hybrid_buckets ? 1 : 0);
}

void launch_ernie55_build_groups_from_csr_m8(
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles_m8,
    const Ernie55MoeRoutedPair* d_pairs,
    ExpertDispatchGroup* d_groups,
    int max_groups,
    hipStream_t stream) {
    if (max_groups <= 0) return;
    hipLaunchKernelGGL(k_ernie55_build_groups_from_csr_m8, dim3(1), dim3(256), 0, stream,
                       d_csr_counts, d_tiles_m8, d_pairs, d_groups, max_groups);
}

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
    hipStream_t stream) {
    if (N <= 0 || K <= 0 || max_tiles_grid <= 0) return;
    const uint32_t gx = uint32_t(N / 32);
    const uint32_t gy = uint32_t(std::min(max_tiles_grid, 128));
    hipLaunchKernelGGL(k_hlut4_prefill_situ_f16_m64, dim3(gx, gy), dim3(256), 0, stream,
                       d_x_rot_f16, d_csr_counts, d_tiles, d_pairs,
                       d_w1_tiles, d_w3_tiles, d_levels8_f16, d_z_f16,
                       N, K, situ_beta, situ_linear_beta);
}

void launch_wht128_f16_inplace_prefill_pairs(
    uint16_t* d_z_f16,
    const int32_t* d_csr_counts,
    const uint64_t* d_signs_i,
    int N, int max_pairs_grid,
    hipStream_t stream) {
    if (N <= 0 || max_pairs_grid <= 0) return;
    const uint32_t gx = uint32_t(N / 128);
    const uint32_t gy = uint32_t(std::min(max_pairs_grid, 512));
    hipLaunchKernelGGL(k_wht128_f16_inplace_prefill_pairs, dim3(gx, gy), dim3(128), 0, stream,
                       d_z_f16, d_csr_counts, d_signs_i, N);
}

void launch_hlut4_prefill_down_atomic_m64(
    const uint16_t* d_z_f16,
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles,
    const Ernie55MoeRoutedPair* d_pairs,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_acc_f32,
    int N_down, int K_inter, int max_tiles_grid,
    hipStream_t stream) {
    if (N_down <= 0 || K_inter <= 0 || max_tiles_grid <= 0) return;
    const uint32_t gx = uint32_t((N_down + 63) / 64);
    const uint32_t gy = uint32_t(std::min(max_tiles_grid, 128));
    hipLaunchKernelGGL(k_hlut4_prefill_down_atomic_m64, dim3(gx, gy), dim3(256), 0, stream,
                       d_z_f16, d_csr_counts, d_tiles, d_pairs,
                       d_w2_tiles, d_levels8_f16, d_y_acc_f32,
                       N_down, K_inter);
}

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
    hipStream_t stream) {
    if (N <= 0 || K <= 0 || max_tiles_grid <= 0) return;
    const uint32_t gx = uint32_t((N + 63) / 64);
    const uint32_t gy = uint32_t(std::min(max_tiles_grid, 128));
    hipLaunchKernelGGL(k_hlut4_prefill_situ_f16_m32, dim3(gx, gy), dim3(256), 0, stream,
                       d_x_rot_f16, d_csr_counts, d_tiles, d_pairs,
                       d_w1_tiles, d_w3_tiles, d_levels8_f16, d_z_f16,
                       N, K, situ_beta, situ_linear_beta);
}

void launch_hlut4_prefill_down_atomic_m32(
    const uint16_t* d_z_f16,
    const int32_t* d_csr_counts,
    const Ernie55MoeWorkTile* d_tiles,
    const Ernie55MoeRoutedPair* d_pairs,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_acc_f32,
    int N_down, int K_inter, int max_tiles_grid,
    hipStream_t stream) {
    if (N_down <= 0 || K_inter <= 0 || max_tiles_grid <= 0) return;
    const uint32_t gx = uint32_t((N_down + 63) / 64);
    const uint32_t gy = uint32_t(std::min(max_tiles_grid, 128));
    hipLaunchKernelGGL(k_hlut4_prefill_down_atomic_m32, dim3(gx, gy), dim3(256), 0, stream,
                       d_z_f16, d_csr_counts, d_tiles, d_pairs,
                       d_w2_tiles, d_levels8_f16, d_y_acc_f32,
                       N_down, K_inter);
}

// ---- Commit 3: Zero-Scratch Vectorized FP16 Converters for Type_HH_HPA Dense & HCA wo_a ----
__global__ __launch_bounds__(256, 8)
void k_f16_to_bf16_vec(const uint16_t* __restrict__ in_f16, uint16_t* __restrict__ out_bf16, int total) {
    const int i4 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i4 * 4;
    if (base + 3 < total) {
        const uint2 v = reinterpret_cast<const uint2*>(in_f16)[i4];
        uint2 o;
        o.x = uint32_t(d_f32_to_bf16(d_f16_to_f32(uint16_t(v.x & 0xFFFFu)))) |
              (uint32_t(d_f32_to_bf16(d_f16_to_f32(uint16_t(v.x >> 16)))) << 16);
        o.y = uint32_t(d_f32_to_bf16(d_f16_to_f32(uint16_t(v.y & 0xFFFFu)))) |
              (uint32_t(d_f32_to_bf16(d_f16_to_f32(uint16_t(v.y >> 16)))) << 16);
        reinterpret_cast<uint2*>(out_bf16)[i4] = o;
    } else {
        for (int k = base; k < total; ++k) out_bf16[k] = d_f32_to_bf16(d_f16_to_f32(in_f16[k]));
    }
}

__global__ __launch_bounds__(256, 8)
void k_f16_to_f32_vec(const uint16_t* __restrict__ in_f16, float* __restrict__ out_f32, int total) {
    const int i4 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i4 * 4;
    if (base + 3 < total) {
        const uint2 v = reinterpret_cast<const uint2*>(in_f16)[i4];
        float4 o;
        o.x = d_f16_to_f32(uint16_t(v.x & 0xFFFFu));
        o.y = d_f16_to_f32(uint16_t(v.x >> 16));
        o.z = d_f16_to_f32(uint16_t(v.y & 0xFFFFu));
        o.w = d_f16_to_f32(uint16_t(v.y >> 16));
        reinterpret_cast<float4*>(out_f32)[i4] = o;
    } else {
        for (int k = base; k < total; ++k) out_f32[k] = d_f16_to_f32(in_f16[k]);
    }
}

__global__ __launch_bounds__(256, 8)
void k_f16_mul_sigmoid_to_bf16(const uint16_t* __restrict__ z_f16, const uint16_t* __restrict__ gate_bf16,
                               uint16_t* __restrict__ out_bf16, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) {
        const float z = d_f16_to_f32(z_f16[i]);
        const float g = d_bf16_to_f32(gate_bf16[i]);
        out_bf16[i] = d_f32_to_bf16(z / (1.0f + __expf(-g)));
    }
}

void launch_f16_to_bf16(const uint16_t* d_in_f16, uint16_t* d_out_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n4 = (total + 3) / 4;
    hipLaunchKernelGGL(k_f16_to_bf16_vec, dim3(uint32_t((n4 + 255) / 256)), dim3(256), 0, stream, d_in_f16, d_out_bf16, total);
}

void launch_f16_to_f32(const uint16_t* d_in_f16, float* d_out_f32, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n4 = (total + 3) / 4;
    hipLaunchKernelGGL(k_f16_to_f32_vec, dim3(uint32_t((n4 + 255) / 256)), dim3(256), 0, stream, d_in_f16, d_out_f32, total);
}

void launch_f16_mul_sigmoid_to_bf16(const uint16_t* d_z_f16, const uint16_t* d_gate_bf16,
                                    uint16_t* d_out_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    hipLaunchKernelGGL(k_f16_mul_sigmoid_to_bf16, dim3(uint32_t((total + 255) / 256)), dim3(256), 0, stream,
                       d_z_f16, d_gate_bf16, d_out_bf16, total);
}

// Wave-11 Cut D3.2: 1-CTA Zero-Scratch GPU ExpertDispatchGroup Builder for M <= 8 Decode
// Groups up to 80 (row, slot, weight) routes into deterministic ExpertDispatchGroup structs in LDS!
__global__ __launch_bounds__(64, 4)
void k_build_decode_expert_groups(
    const int32_t* __restrict__ topk_ids,
    const float*   __restrict__ topk_wts,
    const int16_t* __restrict__ g2l_bb,
    ExpertDispatchGroup* __restrict__ groups,
    int32_t* __restrict__ num_groups_out,
    int M, int max_loc, int max_out_groups)
{
    constexpr int kMaxPairs = 80; // M <= 8, top_k == 10 -> max 80 routes
    __shared__ int16_t s_le[kMaxPairs];
    __shared__ int16_t s_row[kMaxPairs];
    __shared__ int16_t s_slot[kMaxPairs];
    __shared__ float   s_wt[kMaxPairs];
    __shared__ int16_t s_gexp[kMaxPairs];
    __shared__ int16_t s_gcnt[kMaxPairs];
    __shared__ int16_t s_gidx[kMaxPairs][kMaxGroupTokens];
    __shared__ int32_t s_ng;

    const int tid = int(threadIdx.x);
    const int total_pairs = (M * 10 <= kMaxPairs) ? (M * 10) : kMaxPairs;
    for (int i = tid; i < total_pairs; i += 64) {
        const int r = i / 10;
        const int s = i - r * 10;
        const int32_t eid = topk_ids[i];
        const float   wt  = topk_wts[i];
        int16_t le = -1;
        if (wt != 0.0f && eid >= 0 && eid < 512) {
            const int16_t cand = g2l_bb[eid];
            if (cand >= 0 && int(cand) < max_loc) le = cand;
        }
        s_le[i]   = le;
        s_row[i]  = int16_t(r);
        s_slot[i] = int16_t(s);
        s_wt[i]   = wt;
    }
    __syncthreads();

    if (tid == 0) {
        int ng = 0;
        uint32_t mask0 = 0u, mask1 = 0u, mask2 = 0u;
        for (int i = 0; i < total_pairs; ++i) {
            const int16_t le = s_le[i];
            if (le < 0) continue;
            int found = -1;
            for (int g = 0; g < ng; ++g) {
                if (s_gexp[g] == le && s_gcnt[g] < kMaxGroupTokens) {
                    found = g;
                    break;
                }
            }
            if (found < 0) {
                if (ng >= max_out_groups) continue;
                found = ng++;
                s_gexp[found] = le;
                s_gcnt[found] = 0;
            }
            const int c = int(s_gcnt[found]);
            s_gidx[found][c] = int16_t(i);
            s_gcnt[found]    = int16_t(c + 1);
            if (i < 32)      mask0 |= (1u << i);
            else if (i < 64) mask1 |= (1u << (i - 32));
            else             mask2 |= (1u << (i - 64));
        }
        s_ng = ng;
        if (num_groups_out != nullptr) {
            num_groups_out[0] = ng;
            num_groups_out[1] = int32_t(mask0);
            num_groups_out[2] = int32_t(mask1);
            num_groups_out[3] = int32_t(mask2);
        }
    }
    __syncthreads();

    const int ng = s_ng;
    for (int g = tid; g < max_out_groups; g += 64) {
        ExpertDispatchGroup* dst = groups + g;
        if (g < ng) {
            const int cnt = int(s_gcnt[g]);
            dst->expert = int32_t(s_gexp[g]);
            dst->count  = cnt;
            #pragma unroll
            for (int c = 0; c < kMaxGroupTokens; ++c) {
                if (c < cnt) {
                    const int pi = int(s_gidx[g][c]);
                    dst->rows[c]    = int32_t(s_row[pi]);
                    dst->slots[c]   = int32_t(s_slot[pi]);
                    dst->weights[c] = s_wt[pi];
                } else {
                    dst->rows[c]    = 0;
                    dst->slots[c]   = 0;
                    dst->weights[c] = 0.0f;
                }
            }
        } else {
            dst->expert = -1;
            dst->count  = 0;
        }
    }
}

void launch_build_decode_expert_groups(
    const int32_t* d_topk_ids, const float* d_topk_wts, const int16_t* d_g2l_bb,
    ExpertDispatchGroup* d_groups, int M, int max_loc, int max_out_groups, hipStream_t stream,
    int32_t* d_num_groups) {
    if (M <= 0 || max_out_groups <= 0) return;
    hipLaunchKernelGGL(k_build_decode_expert_groups, dim3(1), dim3(64), 0, stream,
                       d_topk_ids, d_topk_wts, d_g2l_bb, d_groups, d_num_groups, M, max_loc, max_out_groups);
}

// Wave-11 Turn-5/6 Pillar III: Fused mapped wait + uint4 x_rot copy + ExpertDispatchGroup + 80-bit mask builder (-180 launches/step!)
__global__ __launch_bounds__(256, 2)
void k_moe_mapped_wait_copy_and_build_groups(
    uint16_t* __restrict__ x_rot_dst,
    const uint16_t* __restrict__ moe_mapped_slabs,
    const uint32_t* __restrict__ moe_mapped_ready,
    const int16_t* __restrict__ g2l_bb,
    ExpertDispatchGroup* __restrict__ groups,
    int32_t* __restrict__ num_groups_out,
    int M, int K_lat, int slab_stride, uint32_t epoch,
    int max_loc, int max_out_groups)
{
    constexpr int kMaxPairs = 80;
    __shared__ int16_t s_le[kMaxPairs];
    __shared__ int16_t s_row[kMaxPairs];
    __shared__ int16_t s_slot[kMaxPairs];
    __shared__ float   s_wt[kMaxPairs];
    __shared__ int16_t s_gexp[kMaxPairs];
    __shared__ int16_t s_gcnt[kMaxPairs];
    __shared__ int16_t s_gidx[kMaxPairs][kMaxGroupTokens];
    __shared__ int32_t s_ng;

    const int tid = int(threadIdx.x);
    if (tid == 0) {
        while (__hip_atomic_load(&moe_mapped_ready[5 * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
            __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_system();
    }
    __syncthreads();

    const uint32_t slot = (epoch - 1u) & 15u;
    const uint16_t* __restrict__ s5 = moe_mapped_slabs + (size_t(slot) * 7u + 5u) * size_t(slab_stride);
    const int n8 = (M * K_lat) >> 3;
    const uint4* __restrict__ src8 = reinterpret_cast<const uint4*>(s5);
    uint4* __restrict__ dst8 = reinterpret_cast<uint4*>(x_rot_dst);
    for (int i = tid; i < n8; i += 256) dst8[i] = src8[i];

    const int32_t* __restrict__ topk_ids = reinterpret_cast<const int32_t*>(s5 + 16384u);
    const float*   __restrict__ topk_wts = reinterpret_cast<const float*>(s5 + 16544u);
    const int total_pairs = (M * 10 <= kMaxPairs) ? (M * 10) : kMaxPairs;
    if (tid < total_pairs) {
        const int r = tid / 10;
        const int s = tid - r * 10;
        const int32_t eid = topk_ids[tid];
        const float   wt  = topk_wts[tid];
        int16_t le = -1;
        if (wt != 0.0f && eid >= 0 && eid < 512) {
            const int16_t cand = g2l_bb[eid];
            if (cand >= 0 && int(cand) < max_loc) le = cand;
        }
        s_le[tid]   = le;
        s_row[tid]  = int16_t(r);
        s_slot[tid] = int16_t(s);
        s_wt[tid]   = wt;
    }
    __syncthreads();

    if (tid == 0) {
        int ng = 0;
        uint32_t mask0 = 0u, mask1 = 0u, mask2 = 0u;
        for (int i = 0; i < total_pairs; ++i) {
            const int16_t le = s_le[i];
            if (le < 0) continue;
            int found = -1;
            for (int g = 0; g < ng; ++g) {
                if (s_gexp[g] == le && s_gcnt[g] < kMaxGroupTokens) {
                    found = g;
                    break;
                }
            }
            if (found < 0) {
                if (ng >= max_out_groups) continue;
                found = ng++;
                s_gexp[found] = le;
                s_gcnt[found] = 0;
            }
            const int c = int(s_gcnt[found]);
            s_gidx[found][c] = int16_t(i);
            s_gcnt[found]    = int16_t(c + 1);
            if (i < 32)      mask0 |= (1u << i);
            else if (i < 64) mask1 |= (1u << (i - 32));
            else             mask2 |= (1u << (i - 64));
        }
        s_ng = ng;
        if (num_groups_out != nullptr) {
            num_groups_out[0] = ng;
            num_groups_out[1] = int32_t(mask0);
            num_groups_out[2] = int32_t(mask1);
            num_groups_out[3] = int32_t(mask2);
        }
    }
    __syncthreads();

    const int ng = s_ng;
    if (tid < max_out_groups) {
        ExpertDispatchGroup* dst = groups + tid;
        if (tid < ng) {
            const int cnt = int(s_gcnt[tid]);
            dst->expert = int32_t(s_gexp[tid]);
            dst->count  = cnt;
            #pragma unroll
            for (int c = 0; c < kMaxGroupTokens; ++c) {
                if (c < cnt) {
                    const int pi = int(s_gidx[tid][c]);
                    dst->rows[c]    = int32_t(s_row[pi]);
                    dst->slots[c]   = int32_t(s_slot[pi]);
                    dst->weights[c] = s_wt[pi];
                } else {
                    dst->rows[c]    = 0;
                    dst->slots[c]   = 0;
                    dst->weights[c] = 0.0f;
                }
            }
        } else {
            dst->expert = -1;
            dst->count  = 0;
        }
    }
}

void launch_moe_mapped_wait_copy_and_build_groups(
    uint16_t* d_x_rot_dst,
    const uint16_t* d_moe_mapped_slabs,
    const uint32_t* d_moe_mapped_ready,
    const int16_t* d_g2l_bb,
    ExpertDispatchGroup* d_groups,
    int32_t* d_num_groups_and_mask4,
    int M, int K_lat, int slab_stride, uint32_t epoch,
    int max_loc, int max_out_groups, hipStream_t stream)
{
    if (M <= 0 || max_out_groups <= 0) return;
    hipLaunchKernelGGL(k_moe_mapped_wait_copy_and_build_groups, dim3(1), dim3(256), 0, stream,
                       d_x_rot_dst, d_moe_mapped_slabs, d_moe_mapped_ready, d_g2l_bb,
                       d_groups, d_num_groups_and_mask4, M, K_lat, slab_stride, epoch, max_loc, max_out_groups);
}

// Wave-11 Cut D3.5: Persistent Zero-Scratch Dynamic-Count Decode MoE Kernels
// Eliminates ~3.5M wasted empty CTAs/step across 5 MI50s x 42 layers!
__global__ __launch_bounds__(128, 8)
void k_wht128_grouped_f32_to_f16_persist(
    const float* __restrict__ in_f32,
    const ExpertDispatchGroup* __restrict__ groups,
    const int32_t* __restrict__ num_groups,
    const uint64_t* __restrict__ signs,
    uint16_t* __restrict__ out_f16, int K)
{
    const int ng = num_groups[0];
    if (int(blockIdx.y) >= ng) return;
    __shared__ float s_x[128];
    const int blk = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const uint64_t* __restrict__ sb = signs ? (signs + blk * 2) : nullptr;
    for (int g_idx = int(blockIdx.y); g_idx < ng; g_idx += int(gridDim.y)) {
        const int cnt = groups[g_idx].count;
        for (int c = 0; c < cnt; ++c) {
            const int row = g_idx * kMaxGroupTokens + c;
            const int idx = row * K + blk * 128 + tid;
            s_x[tid] = in_f32[idx];
            d_wht128_shared(s_x, tid, sb);
            out_f16[idx] = d_f32_to_f16(s_x[tid]);
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(128, 3)
void k_hlut4_fdot2_grouped_gu_situ_persist(
    const uint16_t* __restrict__ x_rot_f16,
    const ExpertDispatchGroup* __restrict__ groups,
    const int32_t* __restrict__ num_groups,
    const Hlut4Tile32x32* __restrict__ w1_tiles,
    const Hlut4Tile32x32* __restrict__ w3_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ z_f32,
    int N_inter, int K_in,
    float situ_beta, float situ_linear_beta)
{
    const int ng = num_groups[0];
    const int nt = int(blockIdx.x);
    if (int(blockIdx.y) >= ng || nt * 32 >= N_inter) return;

    __shared__ uint16_t s_l16[16];
    __shared__ uint32_t s_lut256[256];
    __shared__ uint32_t s_x[2][8][16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_l16[tid]     = pos;
        s_l16[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();
    s_lut256[tid]       = uint32_t(s_l16[tid & 0xFu]) | (uint32_t(s_l16[tid >> 4]) << 16);
    s_lut256[tid + 128] = uint32_t(s_l16[(tid + 128) & 0xFu]) | (uint32_t(s_l16[(tid + 128) >> 4]) << 16);
    __syncthreads();

    const int lr = tid >> 2;
    const int w4 = tid & 3;
    const int n  = nt * 32 + lr;
    const int sg = lr >> 2;
    const int nt_cnt = N_inter >> 5;
    const int kt_cnt = K_in >> 5;
    const uint32_t* __restrict__ x_u32_base = reinterpret_cast<const uint32_t*>(x_rot_f16);
    const int k_u32 = K_in >> 1;
    const int tx = tid >> 4;
    const int tk = tid & 15;

    for (int g_idx = int(blockIdx.y); g_idx < ng; g_idx += int(gridDim.y)) {
        const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
        const int expert = grp->expert;
        const int count  = grp->count;
        if (expert < 0 || count <= 0) continue;

        const size_t tbase = size_t(expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
        const Hlut4Tile32x32* __restrict__ grow = w1_tiles + tbase;
        const Hlut4Tile32x32* __restrict__ urow = w3_tiles + tbase;

        if (count == 1) {
            const uint32_t* __restrict__ x0 = x_u32_base + size_t(grp->rows[0]) * size_t(k_u32);
            float ag = 0.0f, au = 0.0f;
            for (int kt = 0; kt < kt_cnt; ++kt) {
                const int xoff = kt * 16 + w4 * 4;
                const uint32_t gw = grow[kt].q_nib4[tid];
                const uint32_t uw = urow[kt].q_nib4[tid];
                const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);
                const uint4 xv4 = reinterpret_cast<const uint4*>(x0 + xoff)[0];
                const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);

                float lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[gw & 0xFFu]),         xv0, 0.0f, false);
                lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]), xv1, lg, false);
                lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]), xv2, lg, false);
                lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[gw >> 24]),           xv3, lg, false);
                ag = fmaf(lg, sig_g, ag);

                float lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[uw & 0xFFu]),         xv0, 0.0f, false);
                lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]), xv1, lu, false);
                lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]), xv2, lu, false);
                lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[uw >> 24]),           xv3, lu, false);
                au = fmaf(lu, sig_u, au);
            }
            ag = d_reduce4_dpp(ag);
            au = d_reduce4_dpp(au);
            if (w4 == 0) {
                z_f32[size_t(g_idx) * size_t(kMaxGroupTokens) * size_t(N_inter) + size_t(n)] =
                    d_situ_glu(ag, au, situ_beta, situ_linear_beta);
            }
            continue;
        }

        if (count == 2) {
            const uint32_t* __restrict__ x0 = x_u32_base + size_t(grp->rows[0]) * size_t(k_u32);
            const uint32_t* __restrict__ x1 = x_u32_base + size_t(grp->rows[1]) * size_t(k_u32);
            float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
            for (int kt = 0; kt < kt_cnt; ++kt) {
                const int xoff = kt * 16 + w4 * 4;
                const uint32_t gw = grow[kt].q_nib4[tid];
                const uint32_t uw = urow[kt].q_nib4[tid];
                const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);
                const f16x2_t g0 = d_u32_to_f16x2(s_lut256[gw & 0xFFu]);
                const f16x2_t g1 = d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]);
                const f16x2_t g2 = d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]);
                const f16x2_t g3 = d_u32_to_f16x2(s_lut256[gw >> 24]);
                const f16x2_t u0 = d_u32_to_f16x2(s_lut256[uw & 0xFFu]);
                const f16x2_t u1 = d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]);
                const f16x2_t u2 = d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]);
                const f16x2_t u3 = d_u32_to_f16x2(s_lut256[uw >> 24]);
                {
                    const uint4 xv4 = reinterpret_cast<const uint4*>(x0 + xoff)[0];
                    const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                    const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);
                    float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                    lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                    lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                    lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                    ag0 = fmaf(lg, sig_g, ag0);
                    float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                    lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                    lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                    lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                    au0 = fmaf(lu, sig_u, au0);
                }
                {
                    const uint4 xv4 = reinterpret_cast<const uint4*>(x1 + xoff)[0];
                    const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                    const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);
                    float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                    lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                    lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                    lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                    ag1 = fmaf(lg, sig_g, ag1);
                    float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                    lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                    lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                    lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                    au1 = fmaf(lu, sig_u, au1);
                }
            }
            ag0 = d_reduce4_dpp(ag0); au0 = d_reduce4_dpp(au0);
            ag1 = d_reduce4_dpp(ag1); au1 = d_reduce4_dpp(au1);
            if (w4 == 0) {
                const size_t zb = size_t(g_idx) * size_t(kMaxGroupTokens) * size_t(N_inter) + size_t(n);
                z_f32[zb]                   = d_situ_glu(ag0, au0, situ_beta, situ_linear_beta);
                z_f32[zb + size_t(N_inter)] = d_situ_glu(ag1, au1, situ_beta, situ_linear_beta);
            }
            continue;
        }

        auto run_strip = [&](auto tag_step) {
            constexpr int STEP = decltype(tag_step)::value;
            for (int c0 = 0; c0 < count; c0 += STEP) {
                const uint32_t* __restrict__ x_load =
                    (tx < STEP && (c0 + tx) < count) ? (x_u32_base + size_t(grp->rows[c0 + tx]) * size_t(k_u32)) : nullptr;
                float ag[STEP], au[STEP];
                #pragma unroll
                for (int c = 0; c < STEP; ++c) { ag[c] = 0.0f; au[c] = 0.0f; }

                if (x_load) s_x[0][tx][tk] = x_load[tk];
                __syncthreads();

                for (int kt = 0; kt < kt_cnt; ++kt) {
                    const int cur = kt & 1;
                    const int nxt = cur ^ 1;
                    if (kt + 1 < kt_cnt && x_load) s_x[nxt][tx][tk] = x_load[(kt + 1) * 16 + tk];

                    const uint32_t gw = grow[kt].q_nib4[tid];
                    const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                    const uint32_t uw = urow[kt].q_nib4[tid];
                    const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);

                    const f16x2_t g0 = d_u32_to_f16x2(s_lut256[gw & 0xFFu]);
                    const f16x2_t g1 = d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]);
                    const f16x2_t g2 = d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]);
                    const f16x2_t g3 = d_u32_to_f16x2(s_lut256[gw >> 24]);

                    const f16x2_t u0 = d_u32_to_f16x2(s_lut256[uw & 0xFFu]);
                    const f16x2_t u1 = d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]);
                    const f16x2_t u2 = d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]);
                    const f16x2_t u3 = d_u32_to_f16x2(s_lut256[uw >> 24]);

                    #pragma unroll
                    for (int c = 0; c < STEP; ++c) {
                        const uint4 xv4 = reinterpret_cast<const uint4*>(&s_x[cur][c][w4 * 4])[0];
                        const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                        const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);

                        float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                        lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                        lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                        lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                        ag[c] = fmaf(lg, sig_g, ag[c]);

                        float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                        lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                        lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                        lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                        au[c] = fmaf(lu, sig_u, au[c]);
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int c = 0; c < STEP; ++c) {
                    if (c0 + c < count) {
                        ag[c] = d_reduce4_dpp(ag[c]);
                        au[c] = d_reduce4_dpp(au[c]);
                        if (w4 == 0) {
                            z_f32[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + c)) * size_t(N_inter) + size_t(n)] =
                                d_situ_glu(ag[c], au[c], situ_beta, situ_linear_beta);
                        }
                    }
                }
            }
        };

        if (count <= 4) run_strip(std::integral_constant<int, 4>{});
        else            run_strip(std::integral_constant<int, 8>{});
    }
}

__global__ __launch_bounds__(128, 4)
void k_hlut4_fdot2_grouped_down_slot_persist(
    const uint16_t* __restrict__ z_rot_f16,
    const ExpertDispatchGroup* __restrict__ groups,
    const int32_t* __restrict__ num_groups,
    const Hlut4Tile32x32* __restrict__ w2_tiles,
    const uint16_t* __restrict__ levels8_f16,
    float* __restrict__ y_slots_f32,
    int N_out, int K_inter)
{
    const int ng = num_groups[0];
    const int nt = int(blockIdx.x);
    if (int(blockIdx.y) >= ng || nt * 32 >= N_out) return;

    __shared__ uint16_t s_l16[16];
    __shared__ uint32_t s_lut256[256];
    __shared__ uint32_t s_z[2][8][16];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_l16[tid]     = pos;
        s_l16[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();
    s_lut256[tid]       = uint32_t(s_l16[tid & 0xFu]) | (uint32_t(s_l16[tid >> 4]) << 16);
    s_lut256[tid + 128] = uint32_t(s_l16[(tid + 128) & 0xFu]) | (uint32_t(s_l16[(tid + 128) >> 4]) << 16);
    __syncthreads();

    const int lr = tid >> 2;
    const int w4 = tid & 3;
    const int n  = nt * 32 + lr;
    const int sg = lr >> 2;
    const int nt_cnt = N_out >> 5;
    const int kt_cnt = K_inter >> 5;
    const int k_u32 = K_inter >> 1;
    const int tx = tid >> 4;
    const int tk = tid & 15;

    for (int g_idx = int(blockIdx.y); g_idx < ng; g_idx += int(gridDim.y)) {
        const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
        const int expert = grp->expert;
        const int count  = grp->count;
        if (expert < 0 || count <= 0) continue;

        const size_t tbase = size_t(expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
        const Hlut4Tile32x32* __restrict__ drow = w2_tiles + tbase;
        const uint32_t* __restrict__ z_grp = reinterpret_cast<const uint32_t*>(z_rot_f16) +
                                             size_t(g_idx) * size_t(kMaxGroupTokens) * size_t(k_u32);

        if (count == 1) {
            float a = 0.0f;
            for (int kt = 0; kt < kt_cnt; ++kt) {
                const uint32_t dw = drow[kt].q_nib4[tid];
                const float sig = d_f16_to_f32(drow[kt].scales_f16[sg]);
                const uint4 zv4 = reinterpret_cast<const uint4*>(z_grp + kt * 16 + w4 * 4)[0];
                float lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[dw & 0xFFu]),         d_u32_to_f16x2(zv4.x), 0.0f, false);
                lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(dw >> 8)  & 0xFFu]), d_u32_to_f16x2(zv4.y), lc, false);
                lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(dw >> 16) & 0xFFu]), d_u32_to_f16x2(zv4.z), lc, false);
                lc = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[dw >> 24]),           d_u32_to_f16x2(zv4.w), lc, false);
                a = fmaf(lc, sig, a);
            }
            a = d_reduce4_dpp(a);
            if (w4 == 0) {
                y_slots_f32[(size_t(grp->rows[0]) * 10u + size_t(grp->slots[0])) * size_t(N_out) + size_t(n)] =
                    a * grp->weights[0];
            }
            continue;
        }

        if (count == 2) {
            const uint32_t* __restrict__ z0 = z_grp;
            const uint32_t* __restrict__ z1 = z_grp + size_t(k_u32);
            float a0 = 0.0f, a1 = 0.0f;
            for (int kt = 0; kt < kt_cnt; ++kt) {
                const int zoff = kt * 16 + w4 * 4;
                const uint32_t dw = drow[kt].q_nib4[tid];
                const float sig = d_f16_to_f32(drow[kt].scales_f16[sg]);
                const f16x2_t w0 = d_u32_to_f16x2(s_lut256[dw & 0xFFu]);
                const f16x2_t w1 = d_u32_to_f16x2(s_lut256[(dw >> 8)  & 0xFFu]);
                const f16x2_t w2 = d_u32_to_f16x2(s_lut256[(dw >> 16) & 0xFFu]);
                const f16x2_t w3 = d_u32_to_f16x2(s_lut256[dw >> 24]);
                const uint4 zv0 = reinterpret_cast<const uint4*>(z0 + zoff)[0];
                float lc0 = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(zv0.x), 0.0f, false);
                lc0 = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(zv0.y), lc0, false);
                lc0 = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(zv0.z), lc0, false);
                lc0 = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(zv0.w), lc0, false);
                a0 = fmaf(lc0, sig, a0);
                const uint4 zv1 = reinterpret_cast<const uint4*>(z1 + zoff)[0];
                float lc1 = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(zv1.x), 0.0f, false);
                lc1 = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(zv1.y), lc1, false);
                lc1 = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(zv1.z), lc1, false);
                lc1 = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(zv1.w), lc1, false);
                a1 = fmaf(lc1, sig, a1);
            }
            a0 = d_reduce4_dpp(a0);
            a1 = d_reduce4_dpp(a1);
            if (w4 == 0) {
                y_slots_f32[(size_t(grp->rows[0]) * 10u + size_t(grp->slots[0])) * size_t(N_out) + size_t(n)] =
                    a0 * grp->weights[0];
                y_slots_f32[(size_t(grp->rows[1]) * 10u + size_t(grp->slots[1])) * size_t(N_out) + size_t(n)] =
                    a1 * grp->weights[1];
            }
            continue;
        }

        auto run_strip = [&](auto tag_step) {
            constexpr int STEP = decltype(tag_step)::value;
            for (int c0 = 0; c0 < count; c0 += STEP) {
                const uint32_t* __restrict__ z_lo = (tx < STEP && (c0 + tx) < count) ? (z_grp + size_t(c0 + tx) * size_t(k_u32)) : nullptr;
                float a[STEP];
                #pragma unroll
                for (int c = 0; c < STEP; ++c) a[c] = 0.0f;

                if (z_lo) s_z[0][tx][tk] = z_lo[tk];
                __syncthreads();

                for (int kt = 0; kt < kt_cnt; ++kt) {
                    const int cur = kt & 1;
                    const int nxt = cur ^ 1;
                    if (kt + 1 < kt_cnt && z_lo) s_z[nxt][tx][tk] = z_lo[(kt + 1) * 16 + tk];

                    const uint32_t dw = drow[kt].q_nib4[tid];
                    const float sig = d_f16_to_f32(drow[kt].scales_f16[sg]);
                    const f16x2_t w0 = d_u32_to_f16x2(s_lut256[dw & 0xFFu]);
                    const f16x2_t w1 = d_u32_to_f16x2(s_lut256[(dw >> 8)  & 0xFFu]);
                    const f16x2_t w2 = d_u32_to_f16x2(s_lut256[(dw >> 16) & 0xFFu]);
                    const f16x2_t w3 = d_u32_to_f16x2(s_lut256[dw >> 24]);

                    #pragma unroll
                    for (int c = 0; c < STEP; ++c) {
                        const uint4 zv4 = reinterpret_cast<const uint4*>(&s_z[cur][c][w4 * 4])[0];
                        float lc = __builtin_amdgcn_fdot2(w0, d_u32_to_f16x2(zv4.x), 0.0f, false);
                        lc = __builtin_amdgcn_fdot2(w1, d_u32_to_f16x2(zv4.y), lc, false);
                        lc = __builtin_amdgcn_fdot2(w2, d_u32_to_f16x2(zv4.z), lc, false);
                        lc = __builtin_amdgcn_fdot2(w3, d_u32_to_f16x2(zv4.w), lc, false);
                        a[c] = fmaf(lc, sig, a[c]);
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int c = 0; c < STEP; ++c) {
                    if (c0 + c < count) {
                        a[c] = d_reduce4_dpp(a[c]);
                        if (w4 == 0) {
                            y_slots_f32[(size_t(grp->rows[c0 + c]) * 10u + size_t(grp->slots[c0 + c])) * size_t(N_out) + size_t(n)] =
                                a[c] * grp->weights[c0 + c];
                        }
                    }
                }
            }
        };

        if (count <= 4) run_strip(std::integral_constant<int, 4>{});
        else            run_strip(std::integral_constant<int, 8>{});
    }
}

void launch_hlut4_fdot2_grouped_gu_situ_persist(
    const uint16_t* d_x_rot_f16,
    const ExpertDispatchGroup* d_groups, const int32_t* d_num_groups, int max_groups,
    const Hlut4Tile32x32* d_w1_tiles, const Hlut4Tile32x32* d_w3_tiles,
    const uint16_t* d_levels8_f16,
    float* d_z_f32, int N_inter, int K_in,
    float situ_beta, float situ_linear_beta, hipStream_t stream) {
    if (max_groups <= 0) return;
    const int gy = (max_groups < 16) ? max_groups : 16;
    hipLaunchKernelGGL(k_hlut4_fdot2_grouped_gu_situ_persist, dim3((N_inter + 31) / 32, gy), dim3(128), 0, stream,
                       d_x_rot_f16, d_groups, d_num_groups, d_w1_tiles, d_w3_tiles, d_levels8_f16,
                       d_z_f32, N_inter, K_in, situ_beta, situ_linear_beta);
}

void launch_wht128_grouped_f32_to_f16_persist(
    const float* d_in_f32, const ExpertDispatchGroup* d_groups,
    const int32_t* d_num_groups, int max_groups,
    const uint64_t* d_signs, uint16_t* d_out_f16, int K, hipStream_t stream) {
    if (max_groups <= 0) return;
    const int gy = (max_groups < 24) ? max_groups : 24;
    hipLaunchKernelGGL(k_wht128_grouped_f32_to_f16_persist, dim3(K / 128, gy), dim3(128), 0, stream,
                       d_in_f32, d_groups, d_num_groups, d_signs, d_out_f16, K);
}

void launch_hlut4_fdot2_grouped_down_slot_persist(
    const uint16_t* d_z_rot_f16,
    const ExpertDispatchGroup* d_groups, const int32_t* d_num_groups, int max_groups,
    const Hlut4Tile32x32* d_w2_tiles,
    const uint16_t* d_levels8_f16,
    float* d_y_slots_f32, int N_out, int K_inter, hipStream_t stream) {
    if (max_groups <= 0) return;
    const int gy = (max_groups < 16) ? max_groups : 16;
    hipLaunchKernelGGL(k_hlut4_fdot2_grouped_down_slot_persist, dim3((N_out + 31) / 32, gy), dim3(128), 0, stream,
                       d_z_rot_f16, d_groups, d_num_groups, d_w2_tiles, d_levels8_f16,
                       d_y_slots_f32, N_out, K_inter);
}

// Wave-11 Turn-5/6 Pillar III: Fused GU-SiTU + LDS WHT128 -> FP16 persistent decode kernel
// (-225 launches/step & 0 d_z_f32 HBM2 bounce, 0 scratch, bit-exact 0.00e+00 vs 2-pass!)
__global__ __launch_bounds__(128, 2)
void k_hlut4_fdot2_grouped_gu_situ_wht128_persist(
    const uint16_t* __restrict__ x_rot_f16,
    const ExpertDispatchGroup* __restrict__ groups,
    const int32_t* __restrict__ num_groups,
    const Hlut4Tile32x32* __restrict__ w1_tiles,
    const Hlut4Tile32x32* __restrict__ w3_tiles,
    const uint16_t* __restrict__ levels8_f16,
    const uint64_t* __restrict__ signs_inter,
    uint16_t* __restrict__ z_rot_f16,
    int N_inter, int K_in,
    float situ_beta, float situ_linear_beta)
{
    const int ng = num_groups[0];
    const int blk128 = int(blockIdx.x);
    if (int(blockIdx.y) >= ng || blk128 * 128 >= N_inter) return;

    __shared__ uint16_t s_l16[16];
    __shared__ uint32_t s_lut256[256];
    __shared__ uint32_t s_x[2][8][16];
    __shared__ float    s_z[8][128];
    const int tid = int(threadIdx.x);
    if (tid < 8) {
        const uint16_t pos = levels8_f16[tid];
        s_l16[tid]     = pos;
        s_l16[tid + 8] = uint16_t(pos ^ 0x8000u);
    }
    __syncthreads();
    s_lut256[tid]       = uint32_t(s_l16[tid & 0xFu]) | (uint32_t(s_l16[tid >> 4]) << 16);
    s_lut256[tid + 128] = uint32_t(s_l16[(tid + 128) & 0xFu]) | (uint32_t(s_l16[(tid + 128) >> 4]) << 16);
    __syncthreads();

    const int lr = tid >> 2;
    const int w4 = tid & 3;
    const int sg = lr >> 2;
    const int nt_cnt = N_inter >> 5;
    const int kt_cnt = K_in >> 5;
    const uint32_t* __restrict__ x_u32_base = reinterpret_cast<const uint32_t*>(x_rot_f16);
    const int k_u32 = K_in >> 1;
    const int tx = tid >> 4;
    const int tk = tid & 15;
    const uint64_t* __restrict__ sb = signs_inter ? (signs_inter + blk128 * 2) : nullptr;

    for (int g_idx = int(blockIdx.y); g_idx < ng; g_idx += int(gridDim.y)) {
        const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
        const int expert = grp->expert;
        const int count  = grp->count;
        if (expert < 0 || count <= 0) continue;

        const size_t ebase = size_t(expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(blk128 * 4) * size_t(kt_cnt);

        if (count == 1) {
            const uint32_t* __restrict__ x0 = x_u32_base + size_t(grp->rows[0]) * size_t(k_u32);
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const Hlut4Tile32x32* __restrict__ grow = w1_tiles + ebase + size_t(sub) * size_t(kt_cnt);
                const Hlut4Tile32x32* __restrict__ urow = w3_tiles + ebase + size_t(sub) * size_t(kt_cnt);
                float ag = 0.0f, au = 0.0f;
                for (int kt = 0; kt < kt_cnt; ++kt) {
                    const int xoff = kt * 16 + w4 * 4;
                    const uint32_t gw = grow[kt].q_nib4[tid];
                    const uint32_t uw = urow[kt].q_nib4[tid];
                    const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                    const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);
                    const uint4 xv4 = reinterpret_cast<const uint4*>(x0 + xoff)[0];
                    const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                    const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);

                    float lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[gw & 0xFFu]),         xv0, 0.0f, false);
                    lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]), xv1, lg, false);
                    lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]), xv2, lg, false);
                    lg = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[gw >> 24]),           xv3, lg, false);
                    ag = fmaf(lg, sig_g, ag);

                    float lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[uw & 0xFFu]),         xv0, 0.0f, false);
                    lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]), xv1, lu, false);
                    lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]), xv2, lu, false);
                    lu = __builtin_amdgcn_fdot2(d_u32_to_f16x2(s_lut256[uw >> 24]),           xv3, lu, false);
                    au = fmaf(lu, sig_u, au);
                }
                ag = d_reduce4_dpp(ag);
                au = d_reduce4_dpp(au);
                if (w4 == 0) {
                    s_z[0][sub * 32 + lr] = d_situ_glu(ag, au, situ_beta, situ_linear_beta);
                }
            }
        } else if (count == 2) {
            const uint32_t* __restrict__ x0 = x_u32_base + size_t(grp->rows[0]) * size_t(k_u32);
            const uint32_t* __restrict__ x1 = x_u32_base + size_t(grp->rows[1]) * size_t(k_u32);
            #pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const Hlut4Tile32x32* __restrict__ grow = w1_tiles + ebase + size_t(sub) * size_t(kt_cnt);
                const Hlut4Tile32x32* __restrict__ urow = w3_tiles + ebase + size_t(sub) * size_t(kt_cnt);
                float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
                for (int kt = 0; kt < kt_cnt; ++kt) {
                    const int xoff = kt * 16 + w4 * 4;
                    const uint32_t gw = grow[kt].q_nib4[tid];
                    const uint32_t uw = urow[kt].q_nib4[tid];
                    const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                    const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);
                    const f16x2_t g0 = d_u32_to_f16x2(s_lut256[gw & 0xFFu]);
                    const f16x2_t g1 = d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]);
                    const f16x2_t g2 = d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]);
                    const f16x2_t g3 = d_u32_to_f16x2(s_lut256[gw >> 24]);
                    const f16x2_t u0 = d_u32_to_f16x2(s_lut256[uw & 0xFFu]);
                    const f16x2_t u1 = d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]);
                    const f16x2_t u2 = d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]);
                    const f16x2_t u3 = d_u32_to_f16x2(s_lut256[uw >> 24]);
                    {
                        const uint4 xv4 = reinterpret_cast<const uint4*>(x0 + xoff)[0];
                        const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                        const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);
                        float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                        lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                        lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                        lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                        ag0 = fmaf(lg, sig_g, ag0);
                        float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                        lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                        lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                        lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                        au0 = fmaf(lu, sig_u, au0);
                    }
                    {
                        const uint4 xv4 = reinterpret_cast<const uint4*>(x1 + xoff)[0];
                        const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                        const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);
                        float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                        lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                        lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                        lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                        ag1 = fmaf(lg, sig_g, ag1);
                        float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                        lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                        lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                        lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                        au1 = fmaf(lu, sig_u, au1);
                    }
                }
                ag0 = d_reduce4_dpp(ag0); au0 = d_reduce4_dpp(au0);
                ag1 = d_reduce4_dpp(ag1); au1 = d_reduce4_dpp(au1);
                if (w4 == 0) {
                    s_z[0][sub * 32 + lr] = d_situ_glu(ag0, au0, situ_beta, situ_linear_beta);
                    s_z[1][sub * 32 + lr] = d_situ_glu(ag1, au1, situ_beta, situ_linear_beta);
                }
            }
        } else {
            auto run_strip = [&](auto tag_step) {
                constexpr int STEP = decltype(tag_step)::value;
                for (int c0 = 0; c0 < count; c0 += STEP) {
                    const uint32_t* __restrict__ x_load =
                        (tx < STEP && (c0 + tx) < count) ? (x_u32_base + size_t(grp->rows[c0 + tx]) * size_t(k_u32)) : nullptr;
                    #pragma unroll 1
                    for (int sub = 0; sub < 4; ++sub) {
                        const Hlut4Tile32x32* __restrict__ grow = w1_tiles + ebase + size_t(sub) * size_t(kt_cnt);
                        const Hlut4Tile32x32* __restrict__ urow = w3_tiles + ebase + size_t(sub) * size_t(kt_cnt);
                        float ag[STEP], au[STEP];
                        #pragma unroll
                        for (int c = 0; c < STEP; ++c) { ag[c] = 0.0f; au[c] = 0.0f; }

                        if (x_load) s_x[0][tx][tk] = x_load[tk];
                        __syncthreads();

                        for (int kt = 0; kt < kt_cnt; ++kt) {
                            const int cur = kt & 1;
                            const int nxt = cur ^ 1;
                            if (kt + 1 < kt_cnt && x_load) s_x[nxt][tx][tk] = x_load[(kt + 1) * 16 + tk];

                            const uint32_t gw = grow[kt].q_nib4[tid];
                            const float sig_g = d_f16_to_f32(grow[kt].scales_f16[sg]);
                            const uint32_t uw = urow[kt].q_nib4[tid];
                            const float sig_u = d_f16_to_f32(urow[kt].scales_f16[sg]);

                            const f16x2_t g0 = d_u32_to_f16x2(s_lut256[gw & 0xFFu]);
                            const f16x2_t g1 = d_u32_to_f16x2(s_lut256[(gw >> 8)  & 0xFFu]);
                            const f16x2_t g2 = d_u32_to_f16x2(s_lut256[(gw >> 16) & 0xFFu]);
                            const f16x2_t g3 = d_u32_to_f16x2(s_lut256[gw >> 24]);

                            const f16x2_t u0 = d_u32_to_f16x2(s_lut256[uw & 0xFFu]);
                            const f16x2_t u1 = d_u32_to_f16x2(s_lut256[(uw >> 8)  & 0xFFu]);
                            const f16x2_t u2 = d_u32_to_f16x2(s_lut256[(uw >> 16) & 0xFFu]);
                            const f16x2_t u3 = d_u32_to_f16x2(s_lut256[uw >> 24]);

                            #pragma unroll
                            for (int c = 0; c < STEP; ++c) {
                                const uint4 xv4 = reinterpret_cast<const uint4*>(&s_x[cur][c][w4 * 4])[0];
                                const f16x2_t xv0 = d_u32_to_f16x2(xv4.x), xv1 = d_u32_to_f16x2(xv4.y);
                                const f16x2_t xv2 = d_u32_to_f16x2(xv4.z), xv3 = d_u32_to_f16x2(xv4.w);

                                float lg = __builtin_amdgcn_fdot2(g0, xv0, 0.0f, false);
                                lg = __builtin_amdgcn_fdot2(g1, xv1, lg, false);
                                lg = __builtin_amdgcn_fdot2(g2, xv2, lg, false);
                                lg = __builtin_amdgcn_fdot2(g3, xv3, lg, false);
                                ag[c] = fmaf(lg, sig_g, ag[c]);

                                float lu = __builtin_amdgcn_fdot2(u0, xv0, 0.0f, false);
                                lu = __builtin_amdgcn_fdot2(u1, xv1, lu, false);
                                lu = __builtin_amdgcn_fdot2(u2, xv2, lu, false);
                                lu = __builtin_amdgcn_fdot2(u3, xv3, lu, false);
                                au[c] = fmaf(lu, sig_u, au[c]);
                            }
                            __syncthreads();
                        }

                        #pragma unroll
                        for (int c = 0; c < STEP; ++c) {
                            if (c0 + c < count) {
                                ag[c] = d_reduce4_dpp(ag[c]);
                                au[c] = d_reduce4_dpp(au[c]);
                                if (w4 == 0) {
                                    s_z[c0 + c][sub * 32 + lr] = d_situ_glu(ag[c], au[c], situ_beta, situ_linear_beta);
                                }
                            }
                        }
                    }
                }
            };

            if (count <= 4) run_strip(std::integral_constant<int, 4>{});
            else            run_strip(std::integral_constant<int, 8>{});
        }

        __syncthreads();
        for (int c = 0; c < count; ++c) {
            d_wht128_shared(s_z[c], tid, sb);
            const size_t out_idx = (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(N_inter)
                                 + size_t(blk128 * 128 + tid);
            z_rot_f16[out_idx] = d_f32_to_f16(s_z[c][tid]);
            __syncthreads();
        }
    }
}

void launch_hlut4_fdot2_grouped_gu_situ_wht128_persist(
    const uint16_t* d_x_rot_f16,
    const ExpertDispatchGroup* d_groups, const int32_t* d_num_groups, int max_groups,
    const Hlut4Tile32x32* d_w1_tiles, const Hlut4Tile32x32* d_w3_tiles,
    const uint16_t* d_levels8_f16, const uint64_t* d_signs_inter,
    uint16_t* d_z_rot_f16, int N_inter, int K_in,
    float situ_beta, float situ_linear_beta, hipStream_t stream) {
    if (max_groups <= 0 || N_inter <= 0) return;
    const int gy = (max_groups < 20) ? max_groups : 20;
    hipLaunchKernelGGL(k_hlut4_fdot2_grouped_gu_situ_wht128_persist, dim3(N_inter / 128, gy), dim3(128), 0, stream,
                       d_x_rot_f16, d_groups, d_num_groups, d_w1_tiles, d_w3_tiles,
                       d_levels8_f16, d_signs_inter, d_z_rot_f16, N_inter, K_in, situ_beta, situ_linear_beta);
}

} // namespace ernie55
