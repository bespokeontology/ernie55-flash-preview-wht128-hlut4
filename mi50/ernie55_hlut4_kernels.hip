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

    const int g_idx = int(blockIdx.y);
    const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
    const int expert = grp->expert;
    const int count  = grp->count;
    const int nt = int(blockIdx.x);
    if (expert < 0 || count <= 0 || nt * 32 >= N_inter) return;

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

    const int g_idx = int(blockIdx.y);
    const ExpertDispatchGroup* __restrict__ grp = groups + g_idx;
    const int expert = grp->expert;
    const int count  = grp->count;
    const int nt = int(blockIdx.x);
    if (expert < 0 || count <= 0 || nt * 32 >= N_out) return;

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
                            int N_out) {
    const int m = int(blockIdx.y);
    const int n = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (n >= N_out) return;
    const float* __restrict__ base = y_slots_f32 + size_t(m) * 10u * size_t(N_out) + size_t(n);
    float acc = 0.0f;
    #pragma unroll
    for (int s = 0; s < 10; ++s) {
        acc += base[size_t(s) * size_t(N_out)];
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
                                 int M, int N_out, hipStream_t stream) {
    if (M <= 0 || N_out <= 0) return;
    hipLaunchKernelGGL(k_reduce_moe_slots_f32, dim3((N_out + 255) / 256, M), dim3(256), 0, stream,
                       d_y_slots_f32, d_y_out_f32, N_out);
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

} // namespace ernie55
