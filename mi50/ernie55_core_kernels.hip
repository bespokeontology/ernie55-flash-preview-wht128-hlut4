// SPDX-License-Identifier: Apache-2.0
// Pure gfx906 HIP Zero-Scratch (ScratchSize == 0, 0 trig, 0 spills, 0 slc) Core Architectural Kernels
// for baidu/ERNIE-5.5-Flash (ernie5_5_moe).

#include "ernie55_core_kernels.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cfloat>

namespace ernie55 {

namespace {

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    return __uint_as_float(uint32_t(b) << 16);
}

__device__ __forceinline__ uint16_t d_f32_to_bf16(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float d_warp_sum64(float v) {
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) v += __shfl_xor(v, off, 64);
    return v;
}

// 0-trig 2D factored cos(hi*1024 + lo) lookup for any pos in [0 .. 1,048,575]
__device__ __forceinline__ void d_rope2d_lookup(
    int32_t pos, int pair_idx,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    float& out_c, float& out_s)
{
    const int hi = (pos >> 10) & 1023;
    const int lo = pos & 1023;
    const float ch = cos_hi[hi * 32 + pair_idx];
    const float sh = sin_hi[hi * 32 + pair_idx];
    const float cl = cos_lo[lo * 32 + pair_idx];
    const float sl = sin_lo[lo * 32 + pair_idx];
    out_c = fmaf(ch, cl, -(sh * sl));
    out_s = fmaf(sh, cl,   ch * sl);
}

// ---- Basic Primitives ----
__global__ __launch_bounds__(256, 8)
void k_embed_lookup_bf16(const uint16_t* __restrict__ embed, const int32_t* __restrict__ ids,
                         uint16_t* __restrict__ out, int H) {
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d < H) {
        const int32_t tid = ids[m];
        out[size_t(m) * size_t(H) + size_t(d)] = embed[size_t(tid) * size_t(H) + size_t(d)];
    }
}

__global__ __launch_bounds__(256, 4)
void k_rmsnorm_bf16(const uint16_t* __restrict__ in_bf16, const uint16_t* __restrict__ w_bf16,
                    uint16_t* __restrict__ out_bf16, int D, float eps) {
    __shared__ float s_w[4];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const uint16_t* __restrict__ xr = in_bf16 + size_t(m) * size_t(D);
    uint16_t* __restrict__ yr = out_bf16 + size_t(m) * size_t(D);

    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = d_bf16_to_f32(xr[d]);
        ss = fmaf(v, v, ss);
    }
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(D) + eps);
    for (int d = tid; d < D; d += 256) {
        const float v = d_bf16_to_f32(xr[d]) * inv * d_bf16_to_f32(w_bf16[d]);
        yr[d] = d_f32_to_bf16(v);
    }
}

__global__ __launch_bounds__(256, 4)
void k_rmsnorm_f32_to_bf16(const float* __restrict__ in_f32, const uint16_t* __restrict__ w_bf16,
                           uint16_t* __restrict__ out_bf16, int D, float eps) {
    __shared__ float s_w[4];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const float* __restrict__ xr = in_f32 + size_t(m) * size_t(D);
    uint16_t* __restrict__ yr = out_bf16 + size_t(m) * size_t(D);

    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = xr[d];
        ss = fmaf(v, v, ss);
    }
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(D) + eps);
    for (int d = tid; d < D; d += 256) {
        yr[d] = d_f32_to_bf16(xr[d] * inv * d_bf16_to_f32(w_bf16[d]));
    }
}

__global__ __launch_bounds__(256, 8)
void k_add_bf16_inplace(uint16_t* __restrict__ dst, const uint16_t* __restrict__ src, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) dst[i] = d_f32_to_bf16(d_bf16_to_f32(dst[i]) + d_bf16_to_f32(src[i]));
}

__global__ __launch_bounds__(256, 8)
void k_sigmoid_mul_bf16(uint16_t* __restrict__ x, const uint16_t* __restrict__ g, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) {
        const float xv = d_bf16_to_f32(x[i]);
        const float sv = d_sigmoid(d_bf16_to_f32(g[i]));
        x[i] = d_f32_to_bf16(xv * sv);
    }
}

__global__ __launch_bounds__(256, 4)
void k_argmax_f32(const float* __restrict__ logits, int32_t* __restrict__ out_id, int V) {
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];
    const int tid = int(threadIdx.x);
    float bv = -1e38f;
    int32_t bi = 0;
    for (int i = tid; i < V; i += 256) {
        const float v = logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (int s = 128; s >= 1; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_id = s_idx[0];
}

__global__ __launch_bounds__(256, 4)
void k_argmax_f32_p1(const float* __restrict__ logits, float* __restrict__ part_val,
                     int32_t* __restrict__ part_idx, int V) {
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];
    const int bid = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int stride = int(gridDim.x) * 256;
    float bv = -1e38f;
    int32_t bi = 0;
    for (int i = bid * 256 + tid; i < V; i += stride) {
        const float v = logits[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (int s = 128; s >= 1; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_val[bid] = s_val[0];
        part_idx[bid] = s_idx[0];
    }
}

__global__ __launch_bounds__(64, 4)
void k_argmax_f32_p2(const float* __restrict__ part_val, const int32_t* __restrict__ part_idx,
                     int32_t* __restrict__ out_id, float* __restrict__ out_val) {
    const int lane = int(threadIdx.x);
    float bv = part_val[lane];
    int32_t bi = part_idx[lane];
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        const float ov = __shfl_xor(bv, off, 64);
        const int32_t oi = __shfl_xor(bi, off, 64);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) {
        *out_id = bi;
        if (out_val != nullptr) *out_val = bv;
    }
}

// ---- mHC Shared Sinkhorn-Knopp (20 iters, thread 0 in LDS -> 0 register pressure, 0 scratch!) ----
__device__ __forceinline__ void d_mhc_sinkhorn_in_lds(
    const float* s_mixes24, const float* hc_scale3, const float* hc_base24,
    float* s_pre4, float* s_post4, float* s_cm16, float hc_eps)
{
    for (int j = 0; j < 4; ++j) {
        s_pre4[j]  = d_sigmoid(s_mixes24[j]     * hc_scale3[0] + hc_base24[j])     + hc_eps;
        s_post4[j] = d_sigmoid(s_mixes24[4 + j] * hc_scale3[1] + hc_base24[4 + j]) * 2.0f;
    }
    for (int i = 0; i < 4; ++i) {
        float rmax = -1e30f;
        for (int j = 0; j < 4; ++j) {
            const float v = s_mixes24[8 + i * 4 + j] * hc_scale3[2] + hc_base24[8 + i * 4 + j];
            s_cm16[i * 4 + j] = v;
            if (v > rmax) rmax = v;
        }
        float rsum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const float e = __expf(s_cm16[i * 4 + j] - rmax);
            s_cm16[i * 4 + j] = e;
            rsum += e;
        }
        const float inv_r = 1.0f / rsum;
        for (int j = 0; j < 4; ++j) s_cm16[i * 4 + j] = s_cm16[i * 4 + j] * inv_r + hc_eps;
    }
    // Iter 0 column norm
    for (int j = 0; j < 4; ++j) {
        const float cs = s_cm16[0 * 4 + j] + s_cm16[1 * 4 + j] + s_cm16[2 * 4 + j] + s_cm16[3 * 4 + j];
        const float inv = 1.0f / (cs + hc_eps);
        for (int i = 0; i < 4; ++i) s_cm16[i * 4 + j] *= inv;
    }
    // 19 remaining alternating row/col norms
    for (int it = 0; it < 19; ++it) {
        for (int i = 0; i < 4; ++i) {
            const float rs = s_cm16[i * 4 + 0] + s_cm16[i * 4 + 1] + s_cm16[i * 4 + 2] + s_cm16[i * 4 + 3];
            const float inv = 1.0f / (rs + hc_eps);
            for (int j = 0; j < 4; ++j) s_cm16[i * 4 + j] *= inv;
        }
        for (int j = 0; j < 4; ++j) {
            const float cs = s_cm16[0 * 4 + j] + s_cm16[1 * 4 + j] + s_cm16[2 * 4 + j] + s_cm16[3 * 4 + j];
            const float inv = 1.0f / (cs + hc_eps);
            for (int i = 0; i < 4; ++i) s_cm16[i * 4 + j] *= inv;
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_mhc_pre_broadcast_norm(
    const uint16_t* __restrict__ embed_bf16,
    const float* __restrict__ fn_24x16384,
    const float* __restrict__ hc_scale3,
    const float* __restrict__ hc_base24,
    const uint16_t* __restrict__ norm_w_bf16,
    uint16_t* __restrict__ res_out_bf16,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    uint16_t* __restrict__ layer_in_bf16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_mixes[24];
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_cm[16];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const uint16_t* __restrict__ xr = embed_bf16 + size_t(m) * size_t(H);
    uint16_t* __restrict__ rr = res_out_bf16 + size_t(m) * size_t(4 * H);

    // Broadcast embed -> 4 streams & compute sqrsum
    float ss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        const uint16_t b = xr[d];
        const float v = d_bf16_to_f32(b);
        ss = fmaf(v, v, ss);
        rr[0 * H + d] = b;
        rr[1 * H + d] = b;
        rr[2 * H + d] = b;
        rr[3 * H + d] = b;
    }
    ss = d_warp_sum64(ss);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    // Exact ERNIE 5.5 mHC RMS factor: 1 / (sqrt(mean) + rms_eps)
    const float inv_rms = 1.0f / (sqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H)) + rms_eps);

    // 4 warps compute 24 dot products against fn (6 rows per warp!)
    const int hc_dim = 4 * H;
    for (int row = warp; row < 24; row += 4) {
        const float* __restrict__ frow = fn_24x16384 + size_t(row) * size_t(hc_dim);
        float acc = 0.0f;
        for (int d = lane; d < H; d += 64) {
            const float xv = d_bf16_to_f32(xr[d]);
            const float fsum = frow[0 * H + d] + frow[1 * H + d] + frow[2 * H + d] + frow[3 * H + d];
            acc = fmaf(xv, fsum, acc);
        }
        acc = d_warp_sum64(acc);
        if (lane == 0) s_mixes[row] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        d_mhc_sinkhorn_in_lds(s_mixes, hc_scale3, hc_base24, s_pre, s_post, s_cm, hc_eps);
        for (int j = 0; j < 4; ++j) post_mix[m * 4 + j] = s_post[j];
        for (int k = 0; k < 16; ++k) comb_mix[m * 16 + k] = s_cm[k];
    }
    __syncthreads();

    // Collapse with pre_mix & optional fused RMSNorm (matching mhc_kernels.py BF16 rounding!)
    const float psum = s_pre[0] + s_pre[1] + s_pre[2] + s_pre[3];
    uint16_t* __restrict__ yr = layer_in_bf16 + size_t(m) * size_t(H);
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        const uint16_t vb = d_f32_to_bf16(psum * d_bf16_to_f32(xr[d]));
        yr[d] = vb;
        const float vf = d_bf16_to_f32(vb);
        lss = fmaf(vf, vf, lss);
    }
    if (norm_w_bf16 != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            yr[d] = d_f32_to_bf16(d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(norm_w_bf16[d]));
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_mhc_pre_norm(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ fn_24x16384,
    const float* __restrict__ hc_scale3,
    const float* __restrict__ hc_base24,
    const uint16_t* __restrict__ norm_w_bf16,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    uint16_t* __restrict__ layer_in_bf16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_mixes[24];
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_cm[16];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int hc_dim = 4 * H;
    const uint16_t* __restrict__ rr = res_bf16 + size_t(m) * size_t(hc_dim);

    float ss = 0.0f;
    for (int i = tid; i < hc_dim; i += 256) {
        const float v = d_bf16_to_f32(rr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = d_warp_sum64(ss);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    const float inv_rms = 1.0f / (sqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(hc_dim)) + rms_eps);

    for (int row = warp; row < 24; row += 4) {
        const float* __restrict__ frow = fn_24x16384 + size_t(row) * size_t(hc_dim);
        float acc = 0.0f;
        for (int i = lane; i < hc_dim; i += 64) {
            acc = fmaf(d_bf16_to_f32(rr[i]), frow[i], acc);
        }
        acc = d_warp_sum64(acc);
        if (lane == 0) s_mixes[row] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        d_mhc_sinkhorn_in_lds(s_mixes, hc_scale3, hc_base24, s_pre, s_post, s_cm, hc_eps);
        for (int j = 0; j < 4; ++j) post_mix[m * 4 + j] = s_post[j];
        for (int k = 0; k < 16; ++k) comb_mix[m * 16 + k] = s_cm[k];
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    uint16_t* __restrict__ yr = layer_in_bf16 + size_t(m) * size_t(H);
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        float v = p0 * d_bf16_to_f32(rr[0 * H + d]);
        v = fmaf(p1, d_bf16_to_f32(rr[1 * H + d]), v);
        v = fmaf(p2, d_bf16_to_f32(rr[2 * H + d]), v);
        v = fmaf(p3, d_bf16_to_f32(rr[3 * H + d]), v);
        const uint16_t vb = d_f32_to_bf16(v);
        yr[d] = vb;
        const float vf = d_bf16_to_f32(vb);
        lss = fmaf(vf, vf, lss);
    }
    if (norm_w_bf16 != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            yr[d] = d_f32_to_bf16(d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(norm_w_bf16[d]));
        }
    }
}

template <int M_STATIC>
__global__ __launch_bounds__(256, 2)
void k_mhc_pre_gemv_splitk(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ fn_24x16384,
    float* __restrict__ mhc_ws,
    int H)
{
    __shared__ float s_dot[4][M_STATIC];
    __shared__ float s_sq[4][M_STATIC];
    const int row = int(blockIdx.x); // 0..23
    const int ks  = int(blockIdx.y); // 0..3
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int hc_dim = 4 * H;
    const int hc_dim4 = hc_dim >> 2;
    const int chunk4 = hc_dim4 >> 2;
    const int k4_beg = ks * chunk4;
    const int k4_end = k4_beg + chunk4;

    const float4* __restrict__ f4 = reinterpret_cast<const float4*>(fn_24x16384 + size_t(row) * size_t(hc_dim));
    const uint2* __restrict__ r0 = reinterpret_cast<const uint2*>(res_bf16 + size_t(0) * size_t(hc_dim));
    const uint2* __restrict__ r1 = (M_STATIC > 1) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(1) * size_t(hc_dim)) : r0;
    const uint2* __restrict__ r2 = (M_STATIC > 2) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(2) * size_t(hc_dim)) : r0;
    const uint2* __restrict__ r3 = (M_STATIC > 3) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(3) * size_t(hc_dim)) : r0;
    const uint2* __restrict__ r4 = (M_STATIC > 4) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(4) * size_t(hc_dim)) : r0;
    const uint2* __restrict__ r5 = (M_STATIC > 5) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(5) * size_t(hc_dim)) : r0;
    const uint2* __restrict__ r6 = (M_STATIC > 6) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(6) * size_t(hc_dim)) : r0;
    const uint2* __restrict__ r7 = (M_STATIC > 7) ? reinterpret_cast<const uint2*>(res_bf16 + size_t(7) * size_t(hc_dim)) : r0;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f, d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f, q5 = 0.0f, q6 = 0.0f, q7 = 0.0f;

    auto accum = [&](uint2 rv, float4 fv, float& dot, float& sq) {
        const float x0 = d_bf16_to_f32(uint16_t(rv.x & 0xFFFFu));
        const float x1 = d_bf16_to_f32(uint16_t(rv.x >> 16));
        const float x2 = d_bf16_to_f32(uint16_t(rv.y & 0xFFFFu));
        const float x3 = d_bf16_to_f32(uint16_t(rv.y >> 16));
        dot = fmaf(x0, fv.x, fmaf(x1, fv.y, fmaf(x2, fv.z, fmaf(x3, fv.w, dot))));
        if (row == 0) {
            sq = fmaf(x0, x0, fmaf(x1, x1, fmaf(x2, x2, fmaf(x3, x3, sq))));
        }
    };

    for (int k4 = k4_beg + tid; k4 < k4_end; k4 += 256) {
        const float4 fv = f4[k4];
        accum(r0[k4], fv, d0, q0);
        if constexpr (M_STATIC > 1) accum(r1[k4], fv, d1, q1);
        if constexpr (M_STATIC > 2) accum(r2[k4], fv, d2, q2);
        if constexpr (M_STATIC > 3) accum(r3[k4], fv, d3, q3);
        if constexpr (M_STATIC > 4) accum(r4[k4], fv, d4, q4);
        if constexpr (M_STATIC > 5) accum(r5[k4], fv, d5, q5);
        if constexpr (M_STATIC > 6) accum(r6[k4], fv, d6, q6);
        if constexpr (M_STATIC > 7) accum(r7[k4], fv, d7, q7);
    }

    auto wred = [&](int mi, float dv, float qv) {
        dv = d_warp_sum64(dv);
        if (lane == 0) s_dot[warp][mi] = dv;
        if (row == 0) {
            qv = d_warp_sum64(qv);
            if (lane == 0) s_sq[warp][mi] = qv;
        }
    };
    wred(0, d0, q0);
    if constexpr (M_STATIC > 1) wred(1, d1, q1);
    if constexpr (M_STATIC > 2) wred(2, d2, q2);
    if constexpr (M_STATIC > 3) wred(3, d3, q3);
    if constexpr (M_STATIC > 4) wred(4, d4, q4);
    if constexpr (M_STATIC > 5) wred(5, d5, q5);
    if constexpr (M_STATIC > 6) wred(6, d6, q6);
    if constexpr (M_STATIC > 7) wred(7, d7, q7);
    __syncthreads();

    if (tid < M_STATIC) {
        const float dsum = s_dot[0][tid] + s_dot[1][tid] + s_dot[2][tid] + s_dot[3][tid];
        mhc_ws[(tid * 25 + row) * 4 + ks] = dsum;
        if (row == 0) {
            const float qsum = s_sq[0][tid] + s_sq[1][tid] + s_sq[2][tid] + s_sq[3][tid];
            mhc_ws[(tid * 25 + 24) * 4 + ks] = qsum;
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_mhc_pre_sinkhorn_collapse_ws(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ mhc_ws,
    const float* __restrict__ hc_scale3,
    const float* __restrict__ hc_base24,
    const uint16_t* __restrict__ norm_w_bf16,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    uint16_t* __restrict__ layer_in_bf16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_mixes[24];
    __shared__ float s_sq;
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_cm[16];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int hc_dim = 4 * H;

    if (tid < 25) {
        const float* p = mhc_ws + size_t(m * 25 + tid) * 4u;
        const float sum = p[0] + p[1] + p[2] + p[3];
        if (tid < 24) s_mixes[tid] = sum;
        else          s_sq = sum;
    }
    __syncthreads();

    if (tid == 0) {
        const float inv_rms = 1.0f / (sqrtf(s_sq / float(hc_dim)) + rms_eps);
        for (int r = 0; r < 24; ++r) s_mixes[r] *= inv_rms;
        d_mhc_sinkhorn_in_lds(s_mixes, hc_scale3, hc_base24, s_pre, s_post, s_cm, hc_eps);
        for (int j = 0; j < 4; ++j) post_mix[m * 4 + j] = s_post[j];
        for (int k = 0; k < 16; ++k) comb_mix[m * 16 + k] = s_cm[k];
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    const uint16_t* __restrict__ rr = res_bf16 + size_t(m) * size_t(hc_dim);
    uint16_t* __restrict__ yr = layer_in_bf16 + size_t(m) * size_t(H);
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        float v = p0 * d_bf16_to_f32(rr[0 * H + d]);
        v = fmaf(p1, d_bf16_to_f32(rr[1 * H + d]), v);
        v = fmaf(p2, d_bf16_to_f32(rr[2 * H + d]), v);
        v = fmaf(p3, d_bf16_to_f32(rr[3 * H + d]), v);
        const uint16_t vb = d_f32_to_bf16(v);
        yr[d] = vb;
        const float vf = d_bf16_to_f32(vb);
        lss = fmaf(vf, vf, lss);
    }
    if (norm_w_bf16 != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            yr[d] = d_f32_to_bf16(d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(norm_w_bf16[d]));
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_mhc_pre_sinkhorn_from_mixes(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ raw_mixes_Mx24,
    const float* __restrict__ hc_scale3,
    const float* __restrict__ hc_base24,
    const uint16_t* __restrict__ norm_w_bf16,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    uint16_t* __restrict__ layer_in_bf16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_mixes[24];
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_cm[16];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int hc_dim = 4 * H;
    const uint16_t* __restrict__ rr = res_bf16 + size_t(m) * size_t(hc_dim);

    const uint4* __restrict__ r8 = reinterpret_cast<const uint4*>(rr);
    const int hc_dim8 = hc_dim >> 3;
    float ss = 0.0f;
    for (int i8 = tid; i8 < hc_dim8; i8 += 256) {
        const uint4 v = r8[i8];
        const float f0 = d_bf16_to_f32(uint16_t(v.x & 0xFFFFu));
        const float f1 = d_bf16_to_f32(uint16_t(v.x >> 16));
        const float f2 = d_bf16_to_f32(uint16_t(v.y & 0xFFFFu));
        const float f3 = d_bf16_to_f32(uint16_t(v.y >> 16));
        const float f4 = d_bf16_to_f32(uint16_t(v.z & 0xFFFFu));
        const float f5 = d_bf16_to_f32(uint16_t(v.z >> 16));
        const float f6 = d_bf16_to_f32(uint16_t(v.w & 0xFFFFu));
        const float f7 = d_bf16_to_f32(uint16_t(v.w >> 16));
        ss = fmaf(f0, f0, fmaf(f1, f1, fmaf(f2, f2, fmaf(f3, f3, fmaf(f4, f4, fmaf(f5, f5, fmaf(f6, f6, fmaf(f7, f7, ss))))))));
    }
    ss = d_warp_sum64(ss);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    const float inv_rms = 1.0f / (sqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(hc_dim)) + rms_eps);

    if (tid < 24) s_mixes[tid] = raw_mixes_Mx24[size_t(m) * 24u + size_t(tid)] * inv_rms;
    __syncthreads();
    if (tid == 0) {
        d_mhc_sinkhorn_in_lds(s_mixes, hc_scale3, hc_base24, s_pre, s_post, s_cm, hc_eps);
        for (int j = 0; j < 4; ++j) post_mix[m * 4 + j] = s_post[j];
        for (int k = 0; k < 16; ++k) comb_mix[m * 16 + k] = s_cm[k];
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    uint16_t* __restrict__ yr = layer_in_bf16 + size_t(m) * size_t(H);
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        float v = p0 * d_bf16_to_f32(rr[0 * H + d]);
        v = fmaf(p1, d_bf16_to_f32(rr[1 * H + d]), v);
        v = fmaf(p2, d_bf16_to_f32(rr[2 * H + d]), v);
        v = fmaf(p3, d_bf16_to_f32(rr[3 * H + d]), v);
        const uint16_t vb = d_f32_to_bf16(v);
        yr[d] = vb;
        const float vf = d_bf16_to_f32(vb);
        lss = fmaf(vf, vf, lss);
    }
    if (norm_w_bf16 != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            yr[d] = d_f32_to_bf16(d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(norm_w_bf16[d]));
        }
    }
}

__global__ __launch_bounds__(64, 8)
void k_mhc_post_update(
    const uint16_t* __restrict__ sublayer_bf16,
    const uint16_t* __restrict__ res_in_bf16,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    uint16_t* __restrict__ res_out_bf16,
    int H)
{
    const int m = int(blockIdx.y);
    const int d4 = int(blockIdx.x) * 64 + int(threadIdx.x); // 4 BF16 elems/thread -> 256 elems/block!
    const int H4 = H >> 2;
    if (d4 >= H4) return;

    const uint2 xv = reinterpret_cast<const uint2*>(sublayer_bf16 + size_t(m) * size_t(H))[d4];
    const float x0 = d_bf16_to_f32(uint16_t(xv.x & 0xFFFFu));
    const float x1 = d_bf16_to_f32(uint16_t(xv.x >> 16));
    const float x2 = d_bf16_to_f32(uint16_t(xv.y & 0xFFFFu));
    const float x3 = d_bf16_to_f32(uint16_t(xv.y >> 16));

    const uint2* __restrict__ ri4 = reinterpret_cast<const uint2*>(res_in_bf16 + size_t(m) * size_t(4 * H));
    uint2* __restrict__ ro4 = reinterpret_cast<uint2*>(res_out_bf16 + size_t(m) * size_t(4 * H));
    const uint2 u0 = ri4[0 * H4 + d4];
    const uint2 u1 = ri4[1 * H4 + d4];
    const uint2 u2 = ri4[2 * H4 + d4];
    const uint2 u3 = ri4[3 * H4 + d4];

    const float r00 = d_bf16_to_f32(uint16_t(u0.x & 0xFFFFu)), r01 = d_bf16_to_f32(uint16_t(u0.x >> 16));
    const float r02 = d_bf16_to_f32(uint16_t(u0.y & 0xFFFFu)), r03 = d_bf16_to_f32(uint16_t(u0.y >> 16));
    const float r10 = d_bf16_to_f32(uint16_t(u1.x & 0xFFFFu)), r11 = d_bf16_to_f32(uint16_t(u1.x >> 16));
    const float r12 = d_bf16_to_f32(uint16_t(u1.y & 0xFFFFu)), r13 = d_bf16_to_f32(uint16_t(u1.y >> 16));
    const float r20 = d_bf16_to_f32(uint16_t(u2.x & 0xFFFFu)), r21 = d_bf16_to_f32(uint16_t(u2.x >> 16));
    const float r22 = d_bf16_to_f32(uint16_t(u2.y & 0xFFFFu)), r23 = d_bf16_to_f32(uint16_t(u2.y >> 16));
    const float r30 = d_bf16_to_f32(uint16_t(u3.x & 0xFFFFu)), r31 = d_bf16_to_f32(uint16_t(u3.x >> 16));
    const float r32 = d_bf16_to_f32(uint16_t(u3.y & 0xFFFFu)), r33 = d_bf16_to_f32(uint16_t(u3.y >> 16));

    const float* __restrict__ pm = post_mix + size_t(m) * 4;
    const float* __restrict__ cm = comb_mix + size_t(m) * 16;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float p = pm[j];
        const float c0 = cm[0 * 4 + j], c1 = cm[1 * 4 + j], c2 = cm[2 * 4 + j], c3 = cm[3 * 4 + j];
        const float a0 = fmaf(c3, r30, fmaf(c2, r20, fmaf(c1, r10, fmaf(c0, r00, p * x0))));
        const float a1 = fmaf(c3, r31, fmaf(c2, r21, fmaf(c1, r11, fmaf(c0, r01, p * x1))));
        const float a2 = fmaf(c3, r32, fmaf(c2, r22, fmaf(c1, r12, fmaf(c0, r02, p * x2))));
        const float a3 = fmaf(c3, r33, fmaf(c2, r23, fmaf(c1, r13, fmaf(c0, r03, p * x3))));
        uint2 out;
        out.x = uint32_t(d_f32_to_bf16(a0)) | (uint32_t(d_f32_to_bf16(a1)) << 16);
        out.y = uint32_t(d_f32_to_bf16(a2)) | (uint32_t(d_f32_to_bf16(a3)) << 16);
        ro4[j * H4 + d4] = out;
    }
}

__global__ __launch_bounds__(256, 4)
void k_mhc_head_contract(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ hc_head_fn,
    const float* __restrict__ hc_head_scale,
    const float* __restrict__ hc_head_base,
    const uint16_t* __restrict__ final_norm_w,
    uint16_t* __restrict__ out_bf16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_pre[4];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int hc_dim = 4 * H;
    const uint16_t* __restrict__ rr = res_bf16 + size_t(m) * size_t(hc_dim);

    float ss = 0.0f;
    for (int i = tid; i < hc_dim; i += 256) {
        const float v = d_bf16_to_f32(rr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = d_warp_sum64(ss);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    // Exact upstream tilelang_kernels.py:1114: rsqrt(mean + rms_eps)
    const float rsqrt_v = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(hc_dim) + rms_eps);

    // 4 warps -> 4 rows of hc_head_fn
    const float* __restrict__ frow = hc_head_fn + size_t(warp) * size_t(hc_dim);
    float dot = 0.0f;
    for (int i = lane; i < hc_dim; i += 64) dot = fmaf(d_bf16_to_f32(rr[i]), frow[i], dot);
    dot = d_warp_sum64(dot);
    if (lane == 0) s_pre[warp] = d_sigmoid(dot * rsqrt_v * hc_head_scale[0] + hc_head_base[warp]) + hc_eps;
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    uint16_t* __restrict__ yr = out_bf16 + size_t(m) * size_t(H);
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        float v = p0 * d_bf16_to_f32(rr[0 * H + d]);
        v = fmaf(p1, d_bf16_to_f32(rr[1 * H + d]), v);
        v = fmaf(p2, d_bf16_to_f32(rr[2 * H + d]), v);
        v = fmaf(p3, d_bf16_to_f32(rr[3 * H + d]), v);
        const uint16_t vb = d_f32_to_bf16(v);
        yr[d] = vb;
        const float vf = d_bf16_to_f32(vb);
        lss = fmaf(vf, vf, lss);
    }
    if (final_norm_w != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            yr[d] = d_f32_to_bf16(d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(final_norm_w[d]));
        }
    }
}

// ---- HCA Kernels ----
__global__ __launch_bounds__(256, 4)
void k_hca_q_qknorm_rope(
    uint16_t* __restrict__ q_bf16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    int n_heads, float eps)
{
    // 1 block = 1 (token, head) of 512 dims (256 threads x 2 elems = 512)
    __shared__ float s_w[4];
    const int h = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];
    uint16_t* __restrict__ qh = q_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u;

    const float v0 = d_bf16_to_f32(qh[tid * 2]);
    const float v1 = d_bf16_to_f32(qh[tid * 2 + 1]);
    float ss = fmaf(v0, v0, v1 * v1);
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
    float n0 = v0 * inv;
    float n1 = v1 * inv;
    if (tid >= 224) {
        // Interleaved GPT-J RoPE on trailing 64 dims [448..511] (pairs 224..255 -> 0..31)
        float c, s;
        d_rope2d_lookup(pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(n0, c, -(n1 * s));
        const float r1 = fmaf(n1, c,   n0 * s);
        n0 = r0;
        n1 = r1;
    }
    qh[tid * 2]     = d_f32_to_bf16(n0);
    qh[tid * 2 + 1] = d_f32_to_bf16(n1);
}

__global__ __launch_bounds__(256, 8)
void k_hca_kv_rope_swa_insert(
    const uint16_t* __restrict__ kv_bf16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ swa_ring_bf16)
{
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];
    const uint16_t* __restrict__ src = kv_bf16 + size_t(m) * 512u;
    uint16_t* __restrict__ dst = swa_ring_bf16 + size_t(pos & 255) * 512u;

    float v0 = d_bf16_to_f32(src[tid * 2]);
    float v1 = d_bf16_to_f32(src[tid * 2 + 1]);
    if (tid >= 224) {
        float c, s;
        d_rope2d_lookup(pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(v0, c, -(v1 * s));
        const float r1 = fmaf(v1, c,   v0 * s);
        v0 = r0; v1 = r1;
    }
    dst[tid * 2]     = d_f32_to_bf16(v0);
    dst[tid * 2 + 1] = d_f32_to_bf16(v1);
}

__global__ __launch_bounds__(256, 8)
void k_hca_compressor_save_state(
    const float* __restrict__ wkv_f32,
    const float* __restrict__ wgate_f32,
    const float* __restrict__ ape_128x512,
    const int32_t* __restrict__ pos_ids,
    float* __restrict__ state_kv,
    float* __restrict__ state_score)
{
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int r = pos_ids[m] & 127;
    for (int d = tid; d < 512; d += 256) {
        state_kv[r * 512 + d]    = wkv_f32[size_t(m) * 512u + size_t(d)];
        state_score[r * 512 + d] = wgate_f32[size_t(m) * 512u + size_t(d)] + ape_128x512[r * 512 + d];
    }
}

__global__ __launch_bounds__(256, 4)
void k_hca_compressor_reduce_norm_rope(
    const float* __restrict__ state_kv,
    const float* __restrict__ state_score,
    const uint16_t* __restrict__ comp_norm_w,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    int32_t compressed_pos,
    uint16_t* __restrict__ comp_kv_dst_bf16,
    float eps)
{
    __shared__ float s_comp[512];
    __shared__ float s_w[4];
    const int tid = int(threadIdx.x);

    // 256 threads handle 512 channels (2 channels/thread)
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int d = tid + pass * 256;
        float mx = -1e30f;
        for (int r = 0; r < 128; ++r) {
            const float sc = state_score[r * 512 + d];
            if (sc > mx) mx = sc;
        }
        float esum = 0.0f;
        float acc = 0.0f;
        for (int r = 0; r < 128; ++r) {
            const float w = __expf(state_score[r * 512 + d] - mx);
            esum += w;
            acc = fmaf(w, state_kv[r * 512 + d], acc);
        }
        s_comp[d] = acc / esum;
    }
    __syncthreads();

    const float v0 = s_comp[tid * 2];
    const float v1 = s_comp[tid * 2 + 1];
    float ss = fmaf(v0, v0, v1 * v1);
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
    float n0 = v0 * inv * d_bf16_to_f32(comp_norm_w[tid * 2]);
    float n1 = v1 * inv * d_bf16_to_f32(comp_norm_w[tid * 2 + 1]);
    if (tid >= 224) {
        float c, s;
        d_rope2d_lookup(compressed_pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(n0, c, -(n1 * s));
        const float r1 = fmaf(n1, c,   n0 * s);
        n0 = r0; n1 = r1;
    }
    comp_kv_dst_bf16[tid * 2]     = d_f32_to_bf16(n0);
    comp_kv_dst_bf16[tid * 2 + 1] = d_f32_to_bf16(n1);
}

__device__ __forceinline__ float d_dot8_bf16_u4(uint4 a, uint4 b) {
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

__device__ __forceinline__ uint4 d_pack_bf16x8(float v0, float v1, float v2, float v3,
                                               float v4, float v5, float v6, float v7) {
    uint4 u;
    u.x = uint32_t(d_f32_to_bf16(v0)) | (uint32_t(d_f32_to_bf16(v1)) << 16);
    u.y = uint32_t(d_f32_to_bf16(v2)) | (uint32_t(d_f32_to_bf16(v3)) << 16);
    u.z = uint32_t(d_f32_to_bf16(v4)) | (uint32_t(d_f32_to_bf16(v5)) << 16);
    u.w = uint32_t(d_f32_to_bf16(v6)) | (uint32_t(d_f32_to_bf16(v7)) << 16);
    return u;
}

// HCA Attention: 1 wavefront64 per head (64 threads x 8 dims = 512-d via uint4!), 0 __syncthreads(), 0 LDS!
__global__ __launch_bounds__(64, 4)
void k_hca_attn_c128_swa_sink(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ comp_kv_bf16,
    const uint16_t* __restrict__ swa_ring_bf16,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int32_t pos, int num_comp)
{
    const int h = int(blockIdx.x);
    const int lane = int(threadIdx.x); // 0..63 -> 8 dims/lane!

    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + size_t(h) * 512u)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    constexpr float kScale = 0.04419417382415922f; // 1 / sqrt(512)

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    auto step_kv = [&](const uint16_t* __restrict__ kv) {
        const uint4 kv4 = reinterpret_cast<const uint4*>(kv)[lane];
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = __expf(m_max - m_new);
        const float p = __expf(score - m_new);
        o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
        o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
        o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
        o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    };

    for (int b = 0; b < num_comp; ++b) step_kv(comp_kv_bf16 + size_t(b) * 512u);
    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    for (int p_tok = swa_start; p_tok <= pos; ++p_tok) step_kv(swa_ring_bf16 + size_t(p_tok & 255) * 512u);

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(out_bf16 + size_t(h) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

// 2-Head-Shared LDS-Tiled M<=8 HCA decode/verify kernel (48 blocks @ M=6 = 80% CU occupancy, 100% bit-exact!)
__global__ __launch_bounds__(128, 4)
void k_hca_attn_c128_swa_sink_batch(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ comp_kv_bf16,
    const uint16_t* __restrict__ swa_ring_bf16,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads, int32_t pos0, int num_comp)
{
    __shared__ uint4 s_k[2][64];
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6; // 0..1
    const int lane = tid & 63; // 0..63 -> 8 dims/lane
    const int h    = int(blockIdx.x) * 2 + warp;
    const int m    = int(blockIdx.y);
    const int32_t pos = pos0 + m;

    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    constexpr float kScale = 0.04419417382415922f;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    auto step_kv4 = [&](const uint4 kv4) {
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = __expf(m_max - m_new);
        const float p = __expf(score - m_new);
        o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
        o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
        o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
        o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    };

    for (int b0 = 0; b0 < num_comp; b0 += 2) {
        if (b0 + warp < num_comp) {
            s_k[warp][lane] = reinterpret_cast<const uint4*>(comp_kv_bf16 + size_t(b0 + warp) * 512u)[lane];
        }
        __syncthreads();
        const int cnt = (num_comp - b0 < 2) ? (num_comp - b0) : 2;
        for (int j = 0; j < cnt; ++j) step_kv4(s_k[j][lane]);
        __syncthreads();
    }

    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    for (int p0 = swa_start; p0 <= pos; p0 += 2) {
        const int p_load = p0 + warp;
        if (p_load <= pos) {
            s_k[warp][lane] = reinterpret_cast<const uint4*>(swa_ring_bf16 + size_t(p_load & 255) * 512u)[lane];
        }
        __syncthreads();
        const int cnt = (pos - p0 + 1 < 2) ? (pos - p0 + 1) : 2;
        for (int j = 0; j < cnt; ++j) step_kv4(s_k[j][lane]);
        __syncthreads();
    }

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(out_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

// VHA (64x64) + Inverse GPT-J YaRN RoPE on trailing 64 dims [448..511]
__global__ __launch_bounds__(256, 4)
void k_hca_vha_invrope(
    const uint16_t* __restrict__ o_in,
    const float* __restrict__ vha_m,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ o_out)
{
    const int h = int(blockIdx.x);   // 0..63
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims/thread
    const int pos = pos_ids[m];
    const uint16_t* __restrict__ tok_in = o_in + size_t(m) * 64u * 512u;
    const float* __restrict__ mrow = vha_m + size_t(h) * 64u;

    float y0 = 0.0f, y1 = 0.0f;
    for (int g = 0; g < 64; ++g) {
        const float w = mrow[g];
        y0 = fmaf(w, d_bf16_to_f32(tok_in[size_t(g) * 512u + size_t(tid * 2)]),     y0);
        y1 = fmaf(w, d_bf16_to_f32(tok_in[size_t(g) * 512u + size_t(tid * 2 + 1)]), y1);
    }
    if (tid >= 224) {
        // Inverse interleaved RoPE: y_even*c + y_odd*s, y_odd*c - y_even*s
        float c, s;
        d_rope2d_lookup(pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(y0, c,   y1 * s);
        const float r1 = fmaf(y1, c, -(y0 * s));
        y0 = r0; y1 = r1;
    }
    uint16_t* __restrict__ dst = o_out + (size_t(m) * 64u + size_t(h)) * 512u;
    dst[tid * 2]     = d_f32_to_bf16(y0);
    dst[tid * 2 + 1] = d_f32_to_bf16(y1);
}

template <int M_STATIC>
__global__ __launch_bounds__(256, 4)
void k_hca_grouped_wo_a_lrga_ws(
    const uint16_t* __restrict__ o_bf16,
    const uint16_t* __restrict__ wo_a_bf16,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* __restrict__ z_out_bf16,
    int n_groups, int o_lora_rank)
{
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int r = int(blockIdx.x) * 4 + warp;
    const int g = int(blockIdx.y);
    if (r >= o_lora_rank) return;

    const size_t x_tok_stride = size_t(n_groups) * 4096u;
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(wo_a_bf16 + (size_t(g) * size_t(o_lora_rank) + size_t(r)) * 4096u);
    const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(o_bf16 + size_t(0) * x_tok_stride + size_t(g) * 4096u);
    const uint4* __restrict__ x1 = (M_STATIC > 1) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(1) * x_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x2 = (M_STATIC > 2) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(2) * x_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x3 = (M_STATIC > 3) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(3) * x_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x4 = (M_STATIC > 4) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(4) * x_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x5 = (M_STATIC > 5) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(5) * x_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x6 = (M_STATIC > 6) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(6) * x_tok_stride + size_t(g) * 4096u) : x0;
    const uint4* __restrict__ x7 = (M_STATIC > 7) ? reinterpret_cast<const uint4*>(o_bf16 + size_t(7) * x_tok_stride + size_t(g) * 4096u) : x0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int k8 = lane; k8 < 512; k8 += 64) {
        const uint4 wv = w4[k8];
        a0 += d_dot8_bf16_u4(x0[k8], wv);
        if constexpr (M_STATIC > 1) a1 += d_dot8_bf16_u4(x1[k8], wv);
        if constexpr (M_STATIC > 2) a2 += d_dot8_bf16_u4(x2[k8], wv);
        if constexpr (M_STATIC > 3) a3 += d_dot8_bf16_u4(x3[k8], wv);
        if constexpr (M_STATIC > 4) a4 += d_dot8_bf16_u4(x4[k8], wv);
        if constexpr (M_STATIC > 5) a5 += d_dot8_bf16_u4(x5[k8], wv);
        if constexpr (M_STATIC > 6) a6 += d_dot8_bf16_u4(x6[k8], wv);
        if constexpr (M_STATIC > 7) a7 += d_dot8_bf16_u4(x7[k8], wv);
    }
    a0 = d_warp_sum64(a0);
    if constexpr (M_STATIC > 1) a1 = d_warp_sum64(a1);
    if constexpr (M_STATIC > 2) a2 = d_warp_sum64(a2);
    if constexpr (M_STATIC > 3) a3 = d_warp_sum64(a3);
    if constexpr (M_STATIC > 4) a4 = d_warp_sum64(a4);
    if constexpr (M_STATIC > 5) a5 = d_warp_sum64(a5);
    if constexpr (M_STATIC > 6) a6 = d_warp_sum64(a6);
    if constexpr (M_STATIC > 7) a7 = d_warp_sum64(a7);

    if (lane == 0) {
        const size_t o_stride = size_t(n_groups * o_lora_rank);
        const size_t col = size_t(g * o_lora_rank + r);
        auto emit = [&](int m, float acc) {
            const size_t idx = size_t(m) * o_stride + col;
            z_out_bf16[idx] = d_f32_to_bf16(acc * d_sigmoid(d_bf16_to_f32(gate_bf16[idx])));
        };
        emit(0, a0);
        if constexpr (M_STATIC > 1) emit(1, a1);
        if constexpr (M_STATIC > 2) emit(2, a2);
        if constexpr (M_STATIC > 3) emit(3, a3);
        if constexpr (M_STATIC > 4) emit(4, a4);
        if constexpr (M_STATIC > 5) emit(5, a5);
        if constexpr (M_STATIC > 6) emit(6, a6);
        if constexpr (M_STATIC > 7) emit(7, a7);
    }
}

// ---- DSA (mqa_dsa) Kernels ----
__device__ __forceinline__ void d_wht128_nosign(float* s_x, int tid) {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
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

// Indexer prep: blockIdx.x in [0..63] = iq head h (NeoX RoPE [0..63] + WHT128); blockIdx.x == 64 = ik (LayerNorm + NeoX RoPE + WHT128)
__global__ __launch_bounds__(128, 8)
void k_dsa_indexer_prep_neox_wht128(
    uint16_t* __restrict__ iq_bf16,
    const uint16_t* __restrict__ ik_raw_bf16,
    const uint16_t* __restrict__ knorm_w,
    const uint16_t* __restrict__ knorm_b,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ ik_cache_bf16, int max_kv)
{
    __shared__ float s_x[128];
    __shared__ float s_w[2];
    const int which = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];

    if (which < 64) {
        uint16_t* __restrict__ qh = iq_bf16 + (size_t(m) * 64u + size_t(which)) * 128u;
        s_x[tid] = d_bf16_to_f32(qh[tid]);
        __syncthreads();
        // NeoX split-half RoPE on leading 64 dims [0..63]: pair (tid, tid+32) for tid in [0..31]
        if (tid < 32) {
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = s_x[tid];
            const float x1 = s_x[tid + 32];
            s_x[tid]      = fmaf(x0, c, -(x1 * s));
            s_x[tid + 32] = fmaf(x1, c,   x0 * s);
        }
        d_wht128_nosign(s_x, tid);
        qh[tid] = d_f32_to_bf16(s_x[tid]);
    } else {
        const float v = d_bf16_to_f32(ik_raw_bf16[size_t(m) * 128u + size_t(tid)]);
        float mu = d_warp_sum64(v);
        if ((tid & 63) == 0) s_w[tid >> 6] = mu;
        __syncthreads();
        const float mean = (s_w[0] + s_w[1]) * (1.0f / 128.0f);
        const float diff = v - mean;
        float var = d_warp_sum64(diff * diff);
        if ((tid & 63) == 0) s_w[tid >> 6] = var;
        __syncthreads();
        const float inv = rsqrtf((s_w[0] + s_w[1]) * (1.0f / 128.0f) + 1e-6f);
        s_x[tid] = fmaf(diff * inv, d_bf16_to_f32(knorm_w[tid]), d_bf16_to_f32(knorm_b[tid]));
        __syncthreads();
        if (tid < 32) {
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = s_x[tid];
            const float x1 = s_x[tid + 32];
            s_x[tid]      = fmaf(x0, c, -(x1 * s));
            s_x[tid + 32] = fmaf(x1, c,   x0 * s);
        }
        d_wht128_nosign(s_x, tid);
        if (pos >= 0 && pos < max_kv) ik_cache_bf16[size_t(pos) * 128u + size_t(tid)] = d_f32_to_bf16(s_x[tid]);
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_indexer_score_tokens(
    const uint16_t* __restrict__ iq_bf16,
    const uint16_t* __restrict__ iw_bf16,
    const uint16_t* __restrict__ ik_cache_bf16,
    float* __restrict__ scores_f32,
    int n_hist)
{
    __shared__ float s_w[4];
    const int tok = int(blockIdx.x);
    if (tok >= n_hist) return;
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    // 4 warps each score 16 of the 64 indexer heads (64 threads/warp -> 2 dims/thread across 128!)
    const uint16_t* __restrict__ kt = ik_cache_bf16 + size_t(tok) * 128u;
    const float k0 = d_bf16_to_f32(kt[lane * 2]);
    const float k1 = d_bf16_to_f32(kt[lane * 2 + 1]);
    constexpr float kWScale = 0.011048543456039806f; // 1 / sqrt(128 * 64)

    float sum_h = 0.0f;
    for (int h = warp; h < 64; h += 4) {
        const uint16_t* __restrict__ qh = iq_bf16 + size_t(h) * 128u;
        float dp = fmaf(d_bf16_to_f32(qh[lane * 2]), k0, d_bf16_to_f32(qh[lane * 2 + 1]) * k1);
        dp = d_warp_sum64(dp);
        if (lane == 0) {
            const float r = (dp > 0.0f) ? dp : 0.0f;
            sum_h = fmaf(d_bf16_to_f32(iw_bf16[h]) * kWScale, r, sum_h);
        }
    }
    if (lane == 0) s_w[warp] = sum_h;
    __syncthreads();
    if (tid == 0) scores_f32[tok] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
}

__global__ __launch_bounds__(64, 8)
void k_dsa_mla_rope_kv_append(
    uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ kv_c_norm_bf16,
    const uint16_t* __restrict__ k_pe_bf16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ mla_kv_cache_bf16,
    int n_heads, int max_kv)
{
    const int which = int(blockIdx.x); // 0..n_heads-1 = q_rope; n_heads = kv_c + k_pe append
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);  // 0..63
    const int pos = pos_ids[m];

    if (which < n_heads) {
        if (tid < 32) {
            uint16_t* __restrict__ qr = q_bf16 + (size_t(m) * size_t(n_heads) + size_t(which)) * 256u + 192u;
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = d_bf16_to_f32(qr[tid * 2]);
            const float x1 = d_bf16_to_f32(qr[tid * 2 + 1]);
            qr[tid * 2]     = d_f32_to_bf16(fmaf(x0, c, -(x1 * s)));
            qr[tid * 2 + 1] = d_f32_to_bf16(fmaf(x1, c,   x0 * s));
        }
    } else if (pos >= 0 && pos < max_kv) {
        uint16_t* __restrict__ dst = mla_kv_cache_bf16 + size_t(pos) * 576u;
        const uint16_t* __restrict__ kvc = kv_c_norm_bf16 + size_t(m) * 512u;
        for (int d = tid; d < 512; d += 64) dst[d] = kvc[d];
        if (tid < 32) {
            const uint16_t* __restrict__ kp = k_pe_bf16 + size_t(m) * 64u;
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = d_bf16_to_f32(kp[tid * 2]);
            const float x1 = d_bf16_to_f32(kp[tid * 2 + 1]);
            dst[512 + tid * 2]     = d_f32_to_bf16(fmaf(x0, c, -(x1 * s)));
            dst[512 + tid * 2 + 1] = d_f32_to_bf16(fmaf(x1, c,   x0 * s));
        }
    }
}

// Absorbed Sparse MLA over <= 2,176 selected indices + attn_sink: 1 wavefront64/head, uint4, 0 barriers, 0 LDS!
__global__ __launch_bounds__(64, 4)
void k_dsa_mla_absorbed_sparse_sink(
    const uint16_t* __restrict__ q_abs_bf16,
    const uint16_t* __restrict__ mla_kv_bf16,
    const int32_t* __restrict__ sel_indices,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ o_lat_bf16,
    int n_sel)
{
    const int h = int(blockIdx.x);
    const int lane = int(threadIdx.x); // 0..63 -> 8 kv_c + 1 k_pe!

    const uint16_t* __restrict__ qh = q_abs_bf16 + size_t(h) * 576u;
    const uint4 qv = reinterpret_cast<const uint4*>(qh)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    const float qpe = d_bf16_to_f32(qh[512 + lane]);
    constexpr float kScale = 0.0625f; // 1 / sqrt(256)

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    for (int i = 0; i < n_sel; ++i) {
        const int32_t tok = sel_indices[i];
        if (tok < 0) continue;
        const uint16_t* __restrict__ kv = mla_kv_bf16 + size_t(tok) * 576u;
        const uint4 kv4 = reinterpret_cast<const uint4*>(kv)[lane];
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, fmaf(q7, k7, qpe * d_bf16_to_f32(kv[512 + lane])))))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = __expf(m_max - m_new);
        const float p = __expf(score - m_new);
        o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
        o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
        o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
        o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    }

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(o_lat_bf16 + size_t(h) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

__global__ __launch_bounds__(256, 4)
void k_dsa_vha_lrga(
    const uint16_t* __restrict__ o_in,
    const float* __restrict__ vha_m,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* __restrict__ out_bf16)
{
    const int h = int(blockIdx.x);   // 0..63
    const int m = int(blockIdx.y);
    const int d = int(threadIdx.x);  // 0..255
    const uint16_t* __restrict__ tok = o_in + size_t(m) * 64u * 256u;
    const float* __restrict__ mrow = vha_m + size_t(h) * 64u;
    float y = 0.0f;
    for (int g = 0; g < 64; ++g) y = fmaf(mrow[g], d_bf16_to_f32(tok[size_t(g) * 256u + size_t(d)]), y);
    const size_t idx = (size_t(m) * 64u + size_t(h)) * 256u + size_t(d);
    out_bf16[idx] = d_f32_to_bf16(y * d_sigmoid(d_bf16_to_f32(gate_bf16[idx])));
}

// ---- Multi-Router Split-Feature Sigmoid Top-10 Routing (512 experts, 1 block per token, 0 scratch!) ----
__global__ __launch_bounds__(256, 4)
void k_split_sigmoid_top10_route(
    const float* __restrict__ logits_2E,
    const float* __restrict__ corr_bias,
    const float* __restrict__ scale_param,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int max_expert)
{
    __shared__ float s_scores[512];
    __shared__ float s_sel[512];
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const float* __restrict__ row = logits_2E + size_t(m) * 1024u;

    // 256 threads compute 512 split-sigmoid sums: sigmoid(l0) + sigmoid(l1)
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int e = tid + pass * 256;
        const float sc = d_sigmoid(row[e]) + d_sigmoid(row[512 + e]);
        s_scores[e] = sc;
        s_sel[e]    = (e < max_expert) ? (sc + corr_bias[e]) : -1e38f;
    }
    __syncthreads();

    float w_sum = 0.0f;
    int32_t* __restrict__ oids = topk_ids + size_t(m) * 10u;
    float* __restrict__ ows = topk_weights + size_t(m) * 10u;

    for (int k = 0; k < 10; ++k) {
        float bv = s_sel[tid];
        int32_t bi = tid;
        const float v2 = s_sel[tid + 256];
        if (v2 > bv) { bv = v2; bi = tid + 256; }
        s_val[tid] = bv;
        s_idx[tid] = bi;
        __syncthreads();
        for (int s = 128; s >= 1; s >>= 1) {
            if (tid < s) {
                if (s_val[tid + s] > s_val[tid] || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int32_t win = s_idx[0];
            oids[k] = win;
            const float w = s_scores[win];
            ows[k] = w;
            w_sum += w;
            s_sel[win] = -1e38f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float inv = 1.0f / fmaxf(w_sum, 1e-20f);
        for (int k = 0; k < 10; ++k) {
            ows[k] = (ows[k] * inv) * scale_param[oids[k]];
        }
    }
}

// ---- ECHO Fusion & MultiMax ----
__global__ __launch_bounds__(256, 8)
void k_echo_fuse_he(const uint16_t* __restrict__ hproj_4xH,
                    const uint16_t* __restrict__ eproj_H,
                    uint16_t* __restrict__ out_4xH, int H) {
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d >= H) return;
    const float ev = d_bf16_to_f32(eproj_H[size_t(m) * size_t(H) + size_t(d)]);
    const uint16_t* __restrict__ hp = hproj_4xH + size_t(m) * size_t(4 * H);
    uint16_t* __restrict__ dst = out_4xH + size_t(m) * size_t(4 * H);
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        dst[s * H + d] = d_f32_to_bf16(d_bf16_to_f32(hp[s * H + d]) + ev);
    }
}

// Exact FP32 MultiMax matching vLLM-ERNIE test_multimax.py (no FMA contraction so rounding matches!)
__global__ __launch_bounds__(256, 8)
void k_multimax_inplace_f32(float* __restrict__ logits,
                            const float* __restrict__ r,
                            const float* __restrict__ t,
                            int V) {
    const int m = int(blockIdx.y);
    const int v = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (v >= V) return;
    const size_t idx = size_t(m) * size_t(V) + size_t(v);
    const float x = logits[idx];
    // Preserve -inf masked candidate entries (matching vLLM-ERNIE DSpark postprocess_draft_logits)
    if (isinf(x) && x < 0.0f) return;
    const float r0x = r[0] - x; const float a = (r0x <= 0.0f) ? 0.0f : r0x;
    const float xr1 = x - r[1]; const float b = (xr1 <= 0.0f) ? 0.0f : xr1;
    const float r2x = r[2] - x; const float c = (r2x <= 0.0f) ? 0.0f : r2x;
    const float xr3 = x - r[3]; const float d = (xr3 <= 0.0f) ? 0.0f : xr3;
    float y = __fadd_rn(x, __fmul_rn(t[0], a));
    y = __fadd_rn(y, __fmul_rn(t[1], b));
    y = __fadd_rn(y, __fmul_rn(t[2], __fmul_rn(c, c)));
    y = __fadd_rn(y, __fmul_rn(t[3], __fmul_rn(d, d)));
    logits[idx] = y;
}

} // namespace

// ---- Host Launch Wrappers ----
void launch_embed_lookup_bf16(const uint16_t* d_embed_bf16, const int32_t* d_tok_ids,
                              uint16_t* d_out_bf16, int M, int H, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_embed_lookup_bf16, dim3((H + 255) / 256, M), dim3(256), 0, stream,
                       d_embed_bf16, d_tok_ids, d_out_bf16, H);
}

void launch_rmsnorm_bf16(const uint16_t* d_in_bf16, const uint16_t* d_w_bf16,
                         uint16_t* d_out_bf16, int M, int D, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_rmsnorm_bf16, dim3(M), dim3(256), 0, stream,
                       d_in_bf16, d_w_bf16, d_out_bf16, D, eps);
}

void launch_rmsnorm_f32_to_bf16(const float* d_in_f32, const uint16_t* d_w_bf16,
                                uint16_t* d_out_bf16, int M, int D, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_rmsnorm_f32_to_bf16, dim3(M), dim3(256), 0, stream,
                       d_in_f32, d_w_bf16, d_out_bf16, D, eps);
}

void launch_add_bf16_inplace(uint16_t* d_dst_bf16, const uint16_t* d_src_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    hipLaunchKernelGGL(k_add_bf16_inplace, dim3((total + 255) / 256), dim3(256), 0, stream,
                       d_dst_bf16, d_src_bf16, total);
}

void launch_sigmoid_mul_bf16(uint16_t* d_x_bf16, const uint16_t* d_gate_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    hipLaunchKernelGGL(k_sigmoid_mul_bf16, dim3((total + 255) / 256), dim3(256), 0, stream,
                       d_x_bf16, d_gate_bf16, total);
}

void launch_argmax_f32(const float* d_logits, int32_t* d_out_id, int V, hipStream_t stream) {
    hipLaunchKernelGGL(k_argmax_f32, dim3(1), dim3(256), 0, stream, d_logits, d_out_id, V);
}

void launch_argmax_f32_fast(const float* d_logits, float* d_part_vals64, int32_t* d_part_idx64,
                            int32_t* d_out_id, int V, hipStream_t stream, float* d_out_val) {
    hipLaunchKernelGGL(k_argmax_f32_p1, dim3(64), dim3(256), 0, stream, d_logits, d_part_vals64, d_part_idx64, V);
    hipLaunchKernelGGL(k_argmax_f32_p2, dim3(1),  dim3(64),  0, stream, d_part_vals64, d_part_idx64, d_out_id, d_out_val);
}

void launch_mhc_pre_broadcast_norm(
    const uint16_t* d_embed_bf16, const float* d_fn_24x16384, const float* d_hc_scale3,
    const float* d_hc_base24, const uint16_t* d_norm_w_bf16, uint16_t* d_res_out_bf16,
    float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_broadcast_norm, dim3(M), dim3(256), 0, stream,
                       d_embed_bf16, d_fn_24x16384, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                       d_res_out_bf16, d_post_mix, d_comb_mix, d_layer_in_bf16, H, rms_eps, hc_eps);
}

void launch_mhc_pre_norm(
    const uint16_t* d_res_bf16, const float* d_fn_24x16384, const float* d_hc_scale3,
    const float* d_hc_base24, const uint16_t* d_norm_w_bf16, float* d_post_mix,
    float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_norm, dim3(M), dim3(256), 0, stream,
                       d_res_bf16, d_fn_24x16384, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                       d_post_mix, d_comb_mix, d_layer_in_bf16, H, rms_eps, hc_eps);
}

void launch_mhc_pre_norm_fast(
    const uint16_t* d_res_bf16, const float* d_fn_24x16384, const float* d_hc_scale3,
    const float* d_hc_base24, const uint16_t* d_norm_w_bf16, float* d_mhc_ws,
    float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    for (int off = 0; off < M; off += 8) {
        const int rem = (M - off < 8) ? (M - off) : 8;
        const uint16_t* rp = d_res_bf16 + size_t(off) * size_t(4 * H);
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<1>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            case 2: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<2>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            case 3: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<3>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            case 4: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<4>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            case 5: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<5>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            case 6: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<6>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            case 7: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<7>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
            default: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk<8>), dim3(24, 4), dim3(256), 0, stream, rp, d_fn_24x16384, d_mhc_ws, H); break;
        }
        hipLaunchKernelGGL(k_mhc_pre_sinkhorn_collapse_ws, dim3(rem), dim3(256), 0, stream,
                           rp, d_mhc_ws, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                           d_post_mix + size_t(off) * 4u, d_comb_mix + size_t(off) * 16u,
                           d_layer_in_bf16 + size_t(off) * size_t(H), H, rms_eps, hc_eps);
    }
}

void launch_mhc_pre_sinkhorn_from_mixes(
    const uint16_t* d_res_bf16, const float* d_raw_mixes_Mx24, const float* d_hc_scale3,
    const float* d_hc_base24, const uint16_t* d_norm_w_bf16, float* d_post_mix,
    float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_sinkhorn_from_mixes, dim3(M), dim3(256), 0, stream,
                       d_res_bf16, d_raw_mixes_Mx24, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                       d_post_mix, d_comb_mix, d_layer_in_bf16, H, rms_eps, hc_eps);
}

void launch_mhc_post_update(
    const uint16_t* d_sublayer_bf16, const uint16_t* d_res_in_bf16,
    const float* d_post_mix, const float* d_comb_mix,
    uint16_t* d_res_out_bf16, int M, int H, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_post_update, dim3((H + 255) / 256, M), dim3(64), 0, stream,
                       d_sublayer_bf16, d_res_in_bf16, d_post_mix, d_comb_mix, d_res_out_bf16, H);
}

void launch_mhc_head_contract(
    const uint16_t* d_res_bf16, const float* d_hc_head_fn, const float* d_hc_head_scale,
    const float* d_hc_head_base, const uint16_t* d_final_norm_w, uint16_t* d_out_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_head_contract, dim3(M), dim3(256), 0, stream,
                       d_res_bf16, d_hc_head_fn, d_hc_head_scale, d_hc_head_base,
                       d_final_norm_w, d_out_bf16, H, rms_eps, hc_eps);
}

void launch_hca_q_qknorm_rope(
    uint16_t* d_q_bf16, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    int M, int n_heads, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_q_qknorm_rope, dim3(n_heads, M), dim3(256), 0, stream,
                       d_q_bf16, d_pos_ids, rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       n_heads, eps);
}

void launch_hca_kv_rope_swa_insert(
    const uint16_t* d_kv_normed_bf16, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_swa_ring_bf16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_kv_rope_swa_insert, dim3(M), dim3(256), 0, stream,
                       d_kv_normed_bf16, d_pos_ids, rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_swa_ring_bf16);
}

void launch_hca_compressor_save_state(
    const float* d_wkv_f32, const float* d_wgate_f32, const float* d_ape_128x512,
    const int32_t* d_pos_ids, float* d_state_kv, float* d_state_score, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_compressor_save_state, dim3(M), dim3(256), 0, stream,
                       d_wkv_f32, d_wgate_f32, d_ape_128x512, d_pos_ids, d_state_kv, d_state_score);
}

void launch_hca_compressor_reduce_norm_rope(
    const float* d_state_kv, const float* d_state_score, const uint16_t* d_comp_norm_w,
    const Rope2DTableGpu& rope, int32_t compressed_pos, uint16_t* d_comp_kv_dst_bf16,
    float eps, hipStream_t stream) {
    hipLaunchKernelGGL(k_hca_compressor_reduce_norm_rope, dim3(1), dim3(256), 0, stream,
                       d_state_kv, d_state_score, d_comp_norm_w,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       compressed_pos, d_comp_kv_dst_bf16, eps);
}

void launch_hca_attn_c128_swa_sink(
    const uint16_t* d_q_bf16, const uint16_t* d_comp_kv_bf16, const uint16_t* d_swa_ring_bf16,
    const float* d_attn_sink, uint16_t* d_out_bf16, int n_heads, int32_t pos, int num_comp, hipStream_t stream) {
    hipLaunchKernelGGL(k_hca_attn_c128_swa_sink, dim3(n_heads), dim3(64), 0, stream,
                       d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16, d_attn_sink, d_out_bf16, pos, num_comp);
}

void launch_hca_attn_c128_swa_sink_batch(
    const uint16_t* d_q_bf16, const uint16_t* d_comp_kv_bf16, const uint16_t* d_swa_ring_bf16,
    const float* d_attn_sink, uint16_t* d_out_bf16, int n_heads, int32_t pos0, int M, int num_comp, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_attn_c128_swa_sink_batch, dim3(n_heads / 2, M), dim3(128), 0, stream,
                       d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16, d_attn_sink, d_out_bf16, n_heads, pos0, num_comp);
}

void launch_hca_vha_invrope(
    const uint16_t* d_o_in_bf16, const float* d_vha_m_64x64, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, uint16_t* d_o_out_bf16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_vha_invrope, dim3(64, M), dim3(256), 0, stream,
                       d_o_in_bf16, d_vha_m_64x64, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo, d_o_out_bf16);
}

void launch_hca_grouped_wo_a_lrga(
    const uint16_t* d_o_bf16, const uint16_t* d_wo_a_bf16, const uint16_t* d_gate_bf16,
    uint16_t* d_z_out_bf16, int M, int n_groups, int o_lora_rank, hipStream_t stream) {
    if (M <= 0) return;
    const dim3 grid((o_lora_rank + 3) / 4, n_groups);
    const size_t x_stride = size_t(n_groups) * 4096u;
    const size_t z_stride = size_t(n_groups * o_lora_rank);
    for (int off = 0; off < M; off += 8) {
        const int rem = (M - off < 8) ? (M - off) : 8;
        const uint16_t* xo = d_o_bf16     + size_t(off) * x_stride;
        const uint16_t* go = d_gate_bf16  + size_t(off) * z_stride;
        uint16_t*       zo = d_z_out_bf16 + size_t(off) * z_stride;
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<1>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            case 2: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<2>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            case 3: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<3>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            case 4: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<4>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            case 5: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<5>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            case 6: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<6>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            case 7: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<7>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
            default: hipLaunchKernelGGL((k_hca_grouped_wo_a_lrga_ws<8>), grid, dim3(256), 0, stream, xo, d_wo_a_bf16, go, zo, n_groups, o_lora_rank); break;
        }
    }
}

void launch_dsa_indexer_prep_neox_wht128(
    uint16_t* d_iq_bf16, const uint16_t* d_ik_raw_bf16, const uint16_t* d_knorm_w,
    const uint16_t* d_knorm_b, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_ik_cache_bf16, int M, int max_kv, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_indexer_prep_neox_wht128, dim3(65, M), dim3(128), 0, stream,
                       d_iq_bf16, d_ik_raw_bf16, d_knorm_w, d_knorm_b, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_ik_cache_bf16, max_kv);
}

void launch_dsa_indexer_score_tokens(
    const uint16_t* d_iq_bf16, const uint16_t* d_iw_bf16, const uint16_t* d_ik_cache_bf16,
    float* d_scores_f32, int n_hist, hipStream_t stream) {
    if (n_hist <= 0) return;
    hipLaunchKernelGGL(k_dsa_indexer_score_tokens, dim3(n_hist), dim3(256), 0, stream,
                       d_iq_bf16, d_iw_bf16, d_ik_cache_bf16, d_scores_f32, n_hist);
}

void launch_dsa_mla_rope_kv_append(
    uint16_t* d_q_bf16, const uint16_t* d_kv_c_norm_bf16, const uint16_t* d_k_pe_bf16,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_mla_kv_cache_bf16, int M, int n_heads, int max_kv, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_mla_rope_kv_append, dim3(n_heads + 1, M), dim3(64), 0, stream,
                       d_q_bf16, d_kv_c_norm_bf16, d_k_pe_bf16, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_mla_kv_cache_bf16, n_heads, max_kv);
}

void launch_dsa_mla_absorbed_sparse_sink(
    const uint16_t* d_q_abs_bf16, const uint16_t* d_mla_kv_bf16, const int32_t* d_sel_indices,
    const float* d_attn_sink, uint16_t* d_o_lat_bf16, int n_heads, int n_sel, hipStream_t stream) {
    hipLaunchKernelGGL(k_dsa_mla_absorbed_sparse_sink, dim3(n_heads), dim3(64), 0, stream,
                       d_q_abs_bf16, d_mla_kv_bf16, d_sel_indices, d_attn_sink, d_o_lat_bf16, n_sel);
}

void launch_dsa_vha_lrga(
    const uint16_t* d_o_in_bf16, const float* d_vha_m_64x64, const uint16_t* d_gate_bf16,
    uint16_t* d_out_bf16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_vha_lrga, dim3(64, M), dim3(256), 0, stream,
                       d_o_in_bf16, d_vha_m_64x64, d_gate_bf16, d_out_bf16);
}

void launch_split_sigmoid_top10_route(
    const float* d_logits_2E, const float* d_corr_bias, const float* d_scale_param,
    int32_t* d_topk_ids, float* d_topk_weights, int M, hipStream_t stream,
    int max_expert) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_split_sigmoid_top10_route, dim3(M), dim3(256), 0, stream,
                       d_logits_2E, d_corr_bias, d_scale_param, d_topk_ids, d_topk_weights, max_expert);
}

void launch_echo_fuse_he(
    const uint16_t* d_hproj_4xH, const uint16_t* d_eproj_H, uint16_t* d_out_4xH,
    int M, int H, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_echo_fuse_he, dim3((H + 255) / 256, M), dim3(256), 0, stream,
                       d_hproj_4xH, d_eproj_H, d_out_4xH, H);
}

void launch_multimax_inplace_f32(
    float* d_logits_f32, const float* d_ranges4, const float* d_ts4,
    int M, int V, hipStream_t stream) {
    if (M <= 0 || V <= 0) return;
    hipLaunchKernelGGL(k_multimax_inplace_f32, dim3((V + 255) / 256, M), dim3(256), 0, stream,
                       d_logits_f32, d_ranges4, d_ts4, V);
}

namespace {

// Weight-stationary M<=8 Absorbed MLA query: q_abs[m, h, 0..511] = sum_{d=0..191} q[m, h, d] * W_UK[h, d, r]; [512..575] = q_rope
template <int M_STATIC>
__global__ __launch_bounds__(256, 2)
void k_dsa_q_absorb_uk_ws(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ kv_b_bf16,
    uint16_t* __restrict__ q_abs_bf16,
    int n_heads)
{
    const int h = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    __shared__ float s_qn[M_STATIC][192];
    if (tid < 192) {
        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m) {
            s_qn[m][tid] = d_bf16_to_f32(q_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 256u + size_t(tid)]);
        }
    }
    __syncthreads();

    const uint16_t* __restrict__ wuk = kv_b_bf16 + size_t(h) * 448u * 512u;
    for (int pass = 0; pass < 2; ++pass) {
        const int r = tid + pass * 256;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        for (int d = 0; d < 192; ++d) {
            const float wv = d_bf16_to_f32(wuk[size_t(d) * 512u + size_t(r)]);
            a0 = fmaf(s_qn[0][d], wv, a0);
            if constexpr (M_STATIC > 1) a1 = fmaf(s_qn[1][d], wv, a1);
            if constexpr (M_STATIC > 2) a2 = fmaf(s_qn[2][d], wv, a2);
            if constexpr (M_STATIC > 3) a3 = fmaf(s_qn[3][d], wv, a3);
            if constexpr (M_STATIC > 4) a4 = fmaf(s_qn[4][d], wv, a4);
            if constexpr (M_STATIC > 5) a5 = fmaf(s_qn[5][d], wv, a5);
            if constexpr (M_STATIC > 6) a6 = fmaf(s_qn[6][d], wv, a6);
            if constexpr (M_STATIC > 7) a7 = fmaf(s_qn[7][d], wv, a7);
        }
        q_abs_bf16[(size_t(0) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a0);
        if constexpr (M_STATIC > 1) q_abs_bf16[(size_t(1) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a1);
        if constexpr (M_STATIC > 2) q_abs_bf16[(size_t(2) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a2);
        if constexpr (M_STATIC > 3) q_abs_bf16[(size_t(3) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a3);
        if constexpr (M_STATIC > 4) q_abs_bf16[(size_t(4) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a4);
        if constexpr (M_STATIC > 5) q_abs_bf16[(size_t(5) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a5);
        if constexpr (M_STATIC > 6) q_abs_bf16[(size_t(6) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a6);
        if constexpr (M_STATIC > 7) q_abs_bf16[(size_t(7) * size_t(n_heads) + size_t(h)) * 576u + size_t(r)] = d_f32_to_bf16(a7);
    }
    if (tid < 64) {
        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m) {
            q_abs_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 576u + 512u + size_t(tid)] =
                q_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 256u + 192u + size_t(tid)];
        }
    }
}

// Coalesced 1-uint4-load weight-stationary M<=8 Absorbed MLA output projection: o_v[m, h, d] = sum_{r=0..511} o_lat[m, h, r] * W_UV[h, d, r]
// 512 BF16 = 64 lanes x 8 BF16 (1 uint4 per lane, ZERO loop, 100% coalesced!)
template <int M_STATIC>
__global__ __launch_bounds__(256, 4)
void k_dsa_o_project_uv_ws(
    const uint16_t* __restrict__ o_lat_bf16,
    const uint16_t* __restrict__ kv_b_bf16,
    uint16_t* __restrict__ o_v_bf16,
    int n_heads)
{
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int d = int(blockIdx.x) * 4 + warp; // 0..255
    const int h = int(blockIdx.y);            // 0..n_heads-1
    if (d >= 256) return;

    const uint4 wv = reinterpret_cast<const uint4*>(kv_b_bf16 + (size_t(h) * 448u + 192u + size_t(d)) * 512u)[lane];
    auto proj_row = [&](int m) {
        const uint4 xv = reinterpret_cast<const uint4*>(o_lat_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane];
        float s = d_warp_sum64(d_dot8_bf16_u4(xv, wv));
        if (lane == 0) o_v_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 256u + size_t(d)] = d_f32_to_bf16(s);
    };
    proj_row(0);
    if constexpr (M_STATIC > 1) proj_row(1);
    if constexpr (M_STATIC > 2) proj_row(2);
    if constexpr (M_STATIC > 3) proj_row(3);
    if constexpr (M_STATIC > 4) proj_row(4);
    if constexpr (M_STATIC > 5) proj_row(5);
    if constexpr (M_STATIC > 6) proj_row(6);
    if constexpr (M_STATIC > 7) proj_row(7);
}

__global__ __launch_bounds__(256, 8)
void k_f32_to_bf16(const float* __restrict__ in, uint16_t* __restrict__ out, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int n8 = total >> 3;
    if (i < n8) {
        const float4 a = reinterpret_cast<const float4*>(in)[i * 2];
        const float4 b = reinterpret_cast<const float4*>(in)[i * 2 + 1];
        reinterpret_cast<uint4*>(out)[i] = d_pack_bf16x8(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
    } else if (i == n8) {
        for (int k = n8 * 8; k < total; ++k) out[k] = d_f32_to_bf16(in[k]);
    }
}

__global__ __launch_bounds__(256, 8)
void k_bf16_to_f32(const uint16_t* __restrict__ in, float* __restrict__ out, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int n8 = total >> 3;
    if (i < n8) {
        const uint4 u = reinterpret_cast<const uint4*>(in)[i];
        float4 a, b;
        a.x = d_bf16_to_f32(uint16_t(u.x & 0xFFFFu)); a.y = d_bf16_to_f32(uint16_t(u.x >> 16));
        a.z = d_bf16_to_f32(uint16_t(u.y & 0xFFFFu)); a.w = d_bf16_to_f32(uint16_t(u.y >> 16));
        b.x = d_bf16_to_f32(uint16_t(u.z & 0xFFFFu)); b.y = d_bf16_to_f32(uint16_t(u.z >> 16));
        b.z = d_bf16_to_f32(uint16_t(u.w & 0xFFFFu)); b.w = d_bf16_to_f32(uint16_t(u.w >> 16));
        reinterpret_cast<float4*>(out)[i * 2]     = a;
        reinterpret_cast<float4*>(out)[i * 2 + 1] = b;
    } else if (i == n8) {
        for (int k = n8 * 8; k < total; ++k) out[k] = d_bf16_to_f32(in[k]);
    }
}

__global__ __launch_bounds__(256, 8)
void k_f32_add_inplace(float* __restrict__ dst, const float* __restrict__ src, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) dst[i] += src[i];
}

__global__ __launch_bounds__(256, 8)
void k_f32_mul_sigmoid_to_bf16(const float* __restrict__ z_f32, const uint16_t* __restrict__ gate, uint16_t* __restrict__ out, int total) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < total) {
        const float gv = d_sigmoid(d_bf16_to_f32(gate[i]));
        out[i] = d_f32_to_bf16(z_f32[i] * gv);
    }
}

__global__ __launch_bounds__(256, 8)
void k_dsa_copy_qrope_bf16(const uint16_t* __restrict__ q_bf16, uint16_t* __restrict__ q_abs_bf16, int n_heads) {
    const int m = int(blockIdx.y);
    const int h = int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6);
    const int d = int(threadIdx.x) & 63;
    if (h >= n_heads) return;
    const uint16_t* __restrict__ src = q_bf16     + (size_t(m) * size_t(n_heads) + size_t(h)) * 256u + 192u;
    uint16_t*       __restrict__ dst = q_abs_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 576u + 512u;
    dst[d] = src[d];
}

// ---- Batched HCA Prefill Kernels (Zero Scratch, 100% Bit-Exact with Token-by-Token Decode!) ----
__global__ __launch_bounds__(256, 8)
void k_hca_kv_rope_inplace(
    uint16_t* __restrict__ kv_chunk_bf16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo)
{
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    if (tid < 224) return;
    const int pos = pos_ids[m];
    uint16_t* __restrict__ row = kv_chunk_bf16 + size_t(m) * 512u;
    float v0 = d_bf16_to_f32(row[tid * 2]);
    float v1 = d_bf16_to_f32(row[tid * 2 + 1]);
    float c, s;
    d_rope2d_lookup(pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
    row[tid * 2]     = d_f32_to_bf16(fmaf(v0, c, -(v1 * s)));
    row[tid * 2 + 1] = d_f32_to_bf16(fmaf(v1, c,   v0 * s));
}

__global__ __launch_bounds__(256, 4)
void k_hca_prefill_compress_blocks(
    const float* __restrict__ wkv_chunk_f32,
    const float* __restrict__ wgate_chunk_f32,
    const float* __restrict__ ape_128x512,
    const float* __restrict__ prev_state_kv,
    const float* __restrict__ prev_state_sc,
    const uint16_t* __restrict__ comp_norm_w,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    int32_t pos0, int first_blk,
    uint16_t* __restrict__ comp_kv_cache_bf16,
    float eps)
{
    __shared__ float s_comp[512];
    __shared__ float s_w[4];
    const int blk = first_blk + int(blockIdx.x);
    const int blk_start = blk * 128;
    const int tid = int(threadIdx.x);

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int d = tid + pass * 256;
        float mx = -1e30f;
        for (int r = 0; r < 128; ++r) {
            const int p = blk_start + r;
            const float sc = (p < pos0)
                ? prev_state_sc[r * 512 + d]
                : (wgate_chunk_f32[size_t(p - pos0) * 512u + size_t(d)] + ape_128x512[r * 512 + d]);
            if (sc > mx) mx = sc;
        }
        float esum = 0.0f, acc = 0.0f;
        for (int r = 0; r < 128; ++r) {
            const int p = blk_start + r;
            const float kv = (p < pos0) ? prev_state_kv[r * 512 + d] : wkv_chunk_f32[size_t(p - pos0) * 512u + size_t(d)];
            const float sc = (p < pos0)
                ? prev_state_sc[r * 512 + d]
                : (wgate_chunk_f32[size_t(p - pos0) * 512u + size_t(d)] + ape_128x512[r * 512 + d]);
            const float w = __expf(sc - mx);
            esum += w;
            acc = fmaf(w, kv, acc);
        }
        s_comp[d] = acc / esum;
    }
    __syncthreads();

    const float v0 = s_comp[tid * 2];
    const float v1 = s_comp[tid * 2 + 1];
    float ss = fmaf(v0, v0, v1 * v1);
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
    float n0 = v0 * inv * d_bf16_to_f32(comp_norm_w[tid * 2]);
    float n1 = v1 * inv * d_bf16_to_f32(comp_norm_w[tid * 2 + 1]);
    if (tid >= 224) {
        float c, s;
        d_rope2d_lookup(blk_start, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(n0, c, -(n1 * s));
        const float r1 = fmaf(n1, c,   n0 * s);
        n0 = r0; n1 = r1;
    }
    uint16_t* __restrict__ dst = comp_kv_cache_bf16 + size_t(blk) * 512u;
    dst[tid * 2]     = d_f32_to_bf16(n0);
    dst[tid * 2 + 1] = d_f32_to_bf16(n1);
}

// 1-Wavefront (64-thread, uint4, 0 __syncthreads(), 0 LDS) Batched HCA Prefill Attention
__global__ __launch_bounds__(64, 4)
void k_hca_prefill_attn_c128_swa_sink(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ comp_kv_bf16,
    const uint16_t* __restrict__ swa_ring_bf16,
    const uint16_t* __restrict__ kv_chunk_rot,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads, int32_t pos0)
{
    const int h = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int lane = int(threadIdx.x); // 0..63 -> 8 dims/lane!
    const int32_t pos = pos0 + m;
    const int num_comp = pos / 128;

    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    constexpr float kScale = 0.04419417382415922f; // 1 / sqrt(512)

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    auto step_kv = [&](const uint16_t* __restrict__ kv) {
        const uint4 kv4 = reinterpret_cast<const uint4*>(kv)[lane];
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = __expf(m_max - m_new);
        const float p = __expf(score - m_new);
        o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
        o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
        o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
        o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    };

    for (int b = 0; b < num_comp; ++b) step_kv(comp_kv_bf16 + size_t(b) * 512u);

    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    for (int p_tok = swa_start; p_tok <= pos; ++p_tok) {
        const uint16_t* __restrict__ kv = (p_tok < pos0)
            ? (swa_ring_bf16 + size_t(p_tok & 255) * 512u)
            : (kv_chunk_rot  + size_t(p_tok - pos0) * 512u);
        step_kv(kv);
    }

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(out_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

__global__ __launch_bounds__(256, 8)
void k_hca_prefill_commit_tail(
    const uint16_t* __restrict__ kv_chunk_rot,
    const float* __restrict__ wkv_chunk_f32,
    const float* __restrict__ wgate_chunk_f32,
    const float* __restrict__ ape_128x512,
    int32_t pos0, int M, int swa_cnt, int state_start_t,
    uint16_t* __restrict__ swa_ring_bf16,
    float* __restrict__ state_kv,
    float* __restrict__ state_sc)
{
    const int k = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    if (k < swa_cnt) {
        const int t = M - swa_cnt + k;
        const int slot = (pos0 + t) & 255;
        for (int d = tid; d < 512; d += 256) {
            swa_ring_bf16[size_t(slot) * 512u + size_t(d)] = kv_chunk_rot[size_t(t) * 512u + size_t(d)];
        }
    }
    const int t_st = state_start_t + k;
    if (t_st < M) {
        const int r = (pos0 + t_st) & 127;
        for (int d = tid; d < 512; d += 256) {
            state_kv[r * 512 + d] = wkv_chunk_f32[size_t(t_st) * 512u + size_t(d)];
            state_sc[r * 512 + d] = wgate_chunk_f32[size_t(t_st) * 512u + size_t(d)] + ape_128x512[r * 512 + d];
        }
    }
}

// ---- Batched DSA Prefill Kernels (Zero Scratch, 1 Launch for All M Tokens!) ----
__global__ __launch_bounds__(256, 4)
void k_dsa_prefill_kv_norm_rope_append(
    uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ kv_a_raw_576,
    const uint16_t* __restrict__ kv_a_norm_w,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ mla_kv_cache_bf16,
    int n_heads, int max_kv, float eps)
{
    __shared__ float s_w[4];
    const int which = int(blockIdx.x); // 0..n_heads-1 = q_rope; n_heads = kv_c RMSNorm + k_pe RoPE append
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];

    if (which < n_heads) {
        if (tid < 32) {
            uint16_t* __restrict__ qr = q_bf16 + (size_t(m) * size_t(n_heads) + size_t(which)) * 256u + 192u;
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = d_bf16_to_f32(qr[tid * 2]);
            const float x1 = d_bf16_to_f32(qr[tid * 2 + 1]);
            qr[tid * 2]     = d_f32_to_bf16(fmaf(x0, c, -(x1 * s)));
            qr[tid * 2 + 1] = d_f32_to_bf16(fmaf(x1, c,   x0 * s));
        }
    } else if (pos >= 0 && pos < max_kv) {
        const uint16_t* __restrict__ src = kv_a_raw_576 + size_t(m) * 576u;
        const float v0 = d_bf16_to_f32(src[tid * 2]);
        const float v1 = d_bf16_to_f32(src[tid * 2 + 1]);
        float ss = fmaf(v0, v0, v1 * v1);
        ss = d_warp_sum64(ss);
        if ((tid & 63) == 0) s_w[tid >> 6] = ss;
        __syncthreads();
        const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
        uint16_t* __restrict__ dst = mla_kv_cache_bf16 + size_t(pos) * 576u;
        dst[tid * 2]     = d_f32_to_bf16(v0 * inv * d_bf16_to_f32(kv_a_norm_w[tid * 2]));
        dst[tid * 2 + 1] = d_f32_to_bf16(v1 * inv * d_bf16_to_f32(kv_a_norm_w[tid * 2 + 1]));
        if (tid < 32) {
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = d_bf16_to_f32(src[512 + tid * 2]);
            const float x1 = d_bf16_to_f32(src[512 + tid * 2 + 1]);
            dst[512 + tid * 2]     = d_f32_to_bf16(fmaf(x0, c, -(x1 * s)));
            dst[512 + tid * 2 + 1] = d_f32_to_bf16(fmaf(x1, c,   x0 * s));
        }
    }
}

// 1-Wavefront (64-thread, uint4, 0 __syncthreads(), 0 LDS) Dense Causal & Sparse Batch DSA Prefill MLA!
__global__ __launch_bounds__(64, 4)
void k_dsa_prefill_mla_dense_causal_sink(
    const uint16_t* __restrict__ q_abs_bf16,
    const uint16_t* __restrict__ mla_kv_bf16,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ o_lat_bf16,
    int n_heads, int32_t pos0)
{
    const int h = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int lane = int(threadIdx.x); // 0..63 -> 8 kv_c + 1 k_pe!
    const int32_t pos = pos0 + m;

    const uint16_t* __restrict__ qh = q_abs_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 576u;
    const uint4 qv = reinterpret_cast<const uint4*>(qh)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    const float qpe = d_bf16_to_f32(qh[512 + lane]);
    constexpr float kScale = 0.0625f; // 1 / sqrt(256)

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    for (int tok = 0; tok <= pos; ++tok) {
        const uint16_t* __restrict__ kv = mla_kv_bf16 + size_t(tok) * 576u;
        const uint4 kv4 = reinterpret_cast<const uint4*>(kv)[lane];
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, fmaf(q7, k7, qpe * d_bf16_to_f32(kv[512 + lane])))))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = __expf(m_max - m_new);
        const float p = __expf(score - m_new);
        o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
        o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
        o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
        o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    }

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(o_lat_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

__global__ __launch_bounds__(64, 4)
void k_dsa_prefill_mla_sparse_batch(
    const uint16_t* __restrict__ q_abs_bf16,
    const uint16_t* __restrict__ mla_kv_bf16,
    const int32_t* __restrict__ sel_indices_MxS,
    const int32_t* __restrict__ sel_counts,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ o_lat_bf16,
    int n_heads, int stride_sel)
{
    const int h = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int lane = int(threadIdx.x);

    const uint16_t* __restrict__ qh = q_abs_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 576u;
    const uint4 qv = reinterpret_cast<const uint4*>(qh)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    const float qpe = d_bf16_to_f32(qh[512 + lane]);
    constexpr float kScale = 0.0625f;

    const int32_t* __restrict__ sel = sel_indices_MxS + size_t(m) * size_t(stride_sel);
    const int n_sel = sel_counts[m];
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    for (int i = 0; i < n_sel; ++i) {
        const int32_t tok = sel[i];
        if (tok < 0) continue;
        const uint16_t* __restrict__ kv = mla_kv_bf16 + size_t(tok) * 576u;
        const uint4 kv4 = reinterpret_cast<const uint4*>(kv)[lane];
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, fmaf(q7, k7, qpe * d_bf16_to_f32(kv[512 + lane])))))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = __expf(m_max - m_new);
        const float p = __expf(score - m_new);
        o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
        o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
        o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
        o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    }

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(o_lat_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

// Fast SGEMM-DSA Sparse KV Gather: [B, stride_sel, 576] BF16 -> FP32 (72 uint4/row = 576 elems, 0 scratch!)
__global__ __launch_bounds__(64, 8)
void k_dsa_gather_kv_f32(
    const uint16_t* __restrict__ mla_kv_bf16,
    const int32_t* __restrict__ sel_indices_BxS,
    const int32_t* __restrict__ sel_counts,
    float* __restrict__ k_sel_f32,
    int stride_sel)
{
    const int s = int(blockIdx.x);
    const int b = int(blockIdx.y);
    const int lane = int(threadIdx.x); // 0..63 -> 8 kv_c + 1 k_pe
    const int n_sel = sel_counts[b];
    const int32_t tok = (s < n_sel) ? sel_indices_BxS[size_t(b) * size_t(stride_sel) + size_t(s)] : -1;
    float* __restrict__ dst = k_sel_f32 + (size_t(b) * size_t(stride_sel) + size_t(s)) * 576u;
    if (tok >= 0) {
        const uint16_t* __restrict__ src = mla_kv_bf16 + size_t(tok) * 576u;
        const uint4 u = reinterpret_cast<const uint4*>(src)[lane];
        float4 a, c;
        a.x = d_bf16_to_f32(uint16_t(u.x & 0xFFFFu)); a.y = d_bf16_to_f32(uint16_t(u.x >> 16));
        a.z = d_bf16_to_f32(uint16_t(u.y & 0xFFFFu)); a.w = d_bf16_to_f32(uint16_t(u.y >> 16));
        c.x = d_bf16_to_f32(uint16_t(u.z & 0xFFFFu)); c.y = d_bf16_to_f32(uint16_t(u.z >> 16));
        c.z = d_bf16_to_f32(uint16_t(u.w & 0xFFFFu)); c.w = d_bf16_to_f32(uint16_t(u.w >> 16));
        reinterpret_cast<float4*>(dst)[lane * 2]     = a;
        reinterpret_cast<float4*>(dst)[lane * 2 + 1] = c;
        dst[512 + lane] = d_bf16_to_f32(src[512 + lane]);
    } else {
        const float4 z{0.0f, 0.0f, 0.0f, 0.0f};
        reinterpret_cast<float4*>(dst)[lane * 2]     = z;
        reinterpret_cast<float4*>(dst)[lane * 2 + 1] = z;
        dst[512 + lane] = 0.0f;
    }
}

// Fast 1-Wavefront Row Softmax + Attn Sink (Causal or Sparse Count-Masked) for SGEMM-DSA!
__global__ __launch_bounds__(64, 8)
void k_dsa_scores_softmax_sink_f32(
    float* __restrict__ scores_BxHxL,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ sel_counts_or_null,
    int n_heads, int L, int32_t pos0_causal, float scale)
{
    const int h = int(blockIdx.x);
    const int b = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int valid_len = (sel_counts_or_null != nullptr)
        ? sel_counts_or_null[b]
        : (pos0_causal + b + 1);
    const int lim = (valid_len < L) ? valid_len : L;
    float* __restrict__ row = scores_BxHxL + (size_t(b) * size_t(n_heads) + size_t(h)) * size_t(L);

    float mx = attn_sink[h];
    for (int i = lane; i < lim; i += 64) {
        const float v = row[i] * scale;
        row[i] = v;
        if (v > mx) mx = v;
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));

    float s = 0.0f;
    for (int i = lane; i < lim; i += 64) {
        const float e = __expf(row[i] - mx);
        row[i] = e;
        s += e;
    }
    s = d_warp_sum64(s) + __expf(attn_sink[h] - mx);
    const float inv_s = 1.0f / s;
    for (int i = lane; i < lim; i += 64) row[i] *= inv_s;
    for (int i = lim + lane; i < L; i += 64) row[i] = 0.0f;
}

} // namespace

void launch_hca_kv_rope_inplace(
    uint16_t* d_kv_chunk_bf16, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_kv_rope_inplace, dim3(M), dim3(256), 0, stream,
                       d_kv_chunk_bf16, d_pos_ids, rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo);
}

void launch_hca_prefill_compress_blocks(
    const float* d_wkv_chunk_f32, const float* d_wgate_chunk_f32,
    const float* d_ape_128x512, const float* d_prev_state_kv, const float* d_prev_state_sc,
    const uint16_t* d_comp_norm_w, const Rope2DTableGpu& rope,
    int32_t pos0, int M, uint16_t* d_comp_kv_cache_bf16, float eps, hipStream_t stream) {
    if (M <= 0) return;
    const int first_blk = pos0 / 128;
    const int end_blk   = (pos0 + M) / 128;  // exclusive: block b finishes at pos = b*128 + 127 < pos0 + M
    const int n_blks    = end_blk - first_blk;
    if (n_blks <= 0) return;
    hipLaunchKernelGGL(k_hca_prefill_compress_blocks, dim3(n_blks), dim3(256), 0, stream,
                       d_wkv_chunk_f32, d_wgate_chunk_f32, d_ape_128x512,
                       d_prev_state_kv, d_prev_state_sc, d_comp_norm_w,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       pos0, first_blk, d_comp_kv_cache_bf16, eps);
}

void launch_hca_prefill_attn_c128_swa_sink(
    const uint16_t* d_q_bf16, const uint16_t* d_comp_kv_bf16,
    const uint16_t* d_swa_ring_bf16, const uint16_t* d_kv_chunk_rot,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int n_heads, int32_t pos0, int M, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_prefill_attn_c128_swa_sink, dim3(n_heads, M), dim3(64), 0, stream,
                       d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16, d_kv_chunk_rot,
                       d_attn_sink, d_out_bf16, n_heads, pos0);
}

void launch_hca_prefill_commit_tail(
    const uint16_t* d_kv_chunk_rot, const float* d_wkv_chunk_f32,
    const float* d_wgate_chunk_f32, const float* d_ape_128x512,
    int32_t pos0, int M, uint16_t* d_swa_ring_bf16,
    float* d_state_kv, float* d_state_sc, hipStream_t stream) {
    if (M <= 0) return;
    const int swa_cnt = (M < 128) ? M : 128;
    const int end_pos = pos0 + M;
    const int tail_rem = end_pos & 127;
    const int state_start_t = (tail_rem > 0) ? ((end_pos - tail_rem > pos0) ? (end_pos - tail_rem - pos0) : 0) : M;
    const int n_state = M - state_start_t;
    const int n_blk = (swa_cnt > n_state) ? swa_cnt : n_state;
    if (n_blk <= 0) return;
    hipLaunchKernelGGL(k_hca_prefill_commit_tail, dim3(n_blk), dim3(256), 0, stream,
                       d_kv_chunk_rot, d_wkv_chunk_f32, d_wgate_chunk_f32, d_ape_128x512,
                       pos0, M, swa_cnt, state_start_t, d_swa_ring_bf16, d_state_kv, d_state_sc);
}

void launch_dsa_prefill_kv_norm_rope_append(
    uint16_t* d_q_bf16, const uint16_t* d_kv_a_raw_576, const uint16_t* d_kv_a_norm_w,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_mla_kv_cache_bf16, int M, int n_heads, int max_kv, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_prefill_kv_norm_rope_append, dim3(n_heads + 1, M), dim3(256), 0, stream,
                       d_q_bf16, d_kv_a_raw_576, d_kv_a_norm_w, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_mla_kv_cache_bf16, n_heads, max_kv, eps);
}

void launch_dsa_prefill_mla_dense_causal_sink(
    const uint16_t* d_q_abs_bf16, const uint16_t* d_mla_kv_bf16,
    const float* d_attn_sink, uint16_t* d_o_lat_bf16,
    int n_heads, int32_t pos0, int M, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_prefill_mla_dense_causal_sink, dim3(n_heads, M), dim3(64), 0, stream,
                       d_q_abs_bf16, d_mla_kv_bf16, d_attn_sink, d_o_lat_bf16, n_heads, pos0);
}

void launch_dsa_prefill_mla_sparse_batch(
    const uint16_t* d_q_abs_bf16, const uint16_t* d_mla_kv_bf16,
    const int32_t* d_sel_indices_MxS, const int32_t* d_sel_counts,
    const float* d_attn_sink, uint16_t* d_o_lat_bf16,
    int n_heads, int M, int stride_sel, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_prefill_mla_sparse_batch, dim3(n_heads, M), dim3(64), 0, stream,
                       d_q_abs_bf16, d_mla_kv_bf16, d_sel_indices_MxS, d_sel_counts,
                       d_attn_sink, d_o_lat_bf16, n_heads, stride_sel);
}

void launch_dsa_gather_kv_f32(
    const uint16_t* d_mla_kv_bf16, const int32_t* d_sel_indices_BxS,
    const int32_t* d_sel_counts, float* d_k_sel_f32,
    int B, int stride_sel, hipStream_t stream) {
    if (B <= 0 || stride_sel <= 0) return;
    hipLaunchKernelGGL(k_dsa_gather_kv_f32, dim3(stride_sel, B), dim3(64), 0, stream,
                       d_mla_kv_bf16, d_sel_indices_BxS, d_sel_counts, d_k_sel_f32, stride_sel);
}

void launch_dsa_scores_softmax_sink_f32(
    float* d_scores_BxHxL, const float* d_attn_sink,
    const int32_t* d_sel_counts_or_null,
    int B, int n_heads, int L, int32_t pos0_causal, float scale, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0 || L <= 0) return;
    hipLaunchKernelGGL(k_dsa_scores_softmax_sink_f32, dim3(n_heads, B), dim3(64), 0, stream,
                       d_scores_BxHxL, d_attn_sink, d_sel_counts_or_null, n_heads, L, pos0_causal, scale);
}

void launch_dsa_q_absorb_uk(
    const uint16_t* d_q_bf16, const uint16_t* d_kv_b_bf16, uint16_t* d_q_abs_bf16,
    int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    for (int off = 0; off < M; off += 8) {
        const int rem = (M - off < 8) ? (M - off) : 8;
        const uint16_t* qp = d_q_bf16     + size_t(off) * size_t(n_heads) * 256u;
        uint16_t*       dp = d_q_abs_bf16 + size_t(off) * size_t(n_heads) * 576u;
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<1>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            case 2: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<2>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            case 3: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<3>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            case 4: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<4>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            case 5: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<5>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            case 6: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<6>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            case 7: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<7>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
            default: hipLaunchKernelGGL((k_dsa_q_absorb_uk_ws<8>), dim3(n_heads), dim3(256), 0, stream, qp, d_kv_b_bf16, dp, n_heads); break;
        }
    }
}

void launch_dsa_o_project_uv(
    const uint16_t* d_o_lat_bf16, const uint16_t* d_kv_b_bf16, uint16_t* d_o_v_bf16,
    int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    const dim3 grid(256 / 4, n_heads);
    for (int off = 0; off < M; off += 8) {
        const int rem = (M - off < 8) ? (M - off) : 8;
        const uint16_t* op = d_o_lat_bf16 + size_t(off) * size_t(n_heads) * 512u;
        uint16_t*       vp = d_o_v_bf16   + size_t(off) * size_t(n_heads) * 256u;
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<1>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            case 2: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<2>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            case 3: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<3>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            case 4: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<4>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            case 5: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<5>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            case 6: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<6>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            case 7: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<7>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
            default: hipLaunchKernelGGL((k_dsa_o_project_uv_ws<8>), grid, dim3(256), 0, stream, op, d_kv_b_bf16, vp, n_heads); break;
        }
    }
}

void launch_f32_to_bf16(const float* d_in_f32, uint16_t* d_out_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n8 = (total + 7) / 8;
    hipLaunchKernelGGL(k_f32_to_bf16, dim3((n8 + 255) / 256), dim3(256), 0, stream, d_in_f32, d_out_bf16, total);
}

void launch_bf16_to_f32(const uint16_t* d_in_bf16, float* d_out_f32, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n8 = (total + 7) / 8;
    hipLaunchKernelGGL(k_bf16_to_f32, dim3((n8 + 255) / 256), dim3(256), 0, stream, d_in_bf16, d_out_f32, total);
}

// =========================================================================
// ERNIE-5.5-Flash-Preview: FlashIndexer (Top-8 of 64 DSA Heads) & 3L DSpark
// All kernels strictly ScratchSize == 0, 0 VGPR/SGPR spills, 0 trig, 0 slc.
// =========================================================================

__global__ __launch_bounds__(64, 8)
void k_flashindexer_select_heads(
    const float* __restrict__ gate_logits_f32, // [M, 64]
    const float* __restrict__ bias_f32,        // [64]
    const uint16_t* __restrict__ iq_bf16,      // [M, 64, 128]
    const uint16_t* __restrict__ iw_bf16,      // [M, 64]
    uint16_t* __restrict__ q_sel_bf16,         // [M, 8, 128]
    float* __restrict__ w_sel_f32)             // [M, 8]
{
    __shared__ float s_gate[64];
    __shared__ float s_score[64];
    __shared__ float s_w[64];
    __shared__ int   s_idx[8];

    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);  // 0..63

    const float g = gate_logits_f32[size_t(m) * 64u + size_t(tid)];
    s_gate[tid]  = g;
    s_score[tid] = g + bias_f32[tid];
    s_w[tid]     = d_bf16_to_f32(iw_bf16[size_t(m) * 64u + size_t(tid)]);
    __syncthreads();

    // Exact 8-pass top-k argmax over 64 heads (64 threads in 1 wavefront -> zero barriers!)
    float cur = s_score[tid];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float best_v = cur;
        int   best_i = tid;
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(best_v, off, 64);
            const int   oi = __shfl_xor(best_i, off, 64);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (tid == 0) {
            s_idx[k] = best_i;
            w_sel_f32[size_t(m) * 8u + size_t(k)] = s_w[best_i] * s_gate[best_i];
        }
        if (tid == best_i) cur = -1e38f;
    }
    __syncthreads();

    // Gather 8 selected heads' 128-d Q vectors (64 threads x 2 BF16 elems = 128!)
    const uint16_t* __restrict__ q_src_m = iq_bf16 + size_t(m) * 64u * 128u;
    uint16_t* __restrict__ q_dst_m = q_sel_bf16 + size_t(m) * 8u * 128u;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int h = s_idx[k];
        const uint16_t* __restrict__ src = q_src_m + size_t(h) * 128u;
        uint16_t* __restrict__ dst = q_dst_m + size_t(k) * 128u;
        dst[tid * 2]     = src[tid * 2];
        dst[tid * 2 + 1] = src[tid * 2 + 1];
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_indexer_score_topk8(
    const uint16_t* __restrict__ q_sel_bf16,   // [8, 128]
    const float* __restrict__ w_sel_f32,       // [8]
    const uint16_t* __restrict__ ik_cache_bf16,// [n_hist, 128]
    float* __restrict__ scores_f32,            // [n_hist]
    int n_hist)
{
    // 4 warps/block = 4 tokens/block (1 wavefront64 per token, 2 dims/lane across 128!)
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int tok  = int(blockIdx.x) * 4 + warp;
    if (tok >= n_hist) return;

    const uint16_t* __restrict__ kt = ik_cache_bf16 + size_t(tok) * 128u;
    const float k0 = d_bf16_to_f32(kt[lane * 2]);
    const float k1 = d_bf16_to_f32(kt[lane * 2 + 1]);
    constexpr float kWScale = 0.011048543456039806f; // 1 / sqrt(128 * 64)

    float total = 0.0f;
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        const uint16_t* __restrict__ qh = q_sel_bf16 + size_t(h) * 128u;
        float dp = fmaf(d_bf16_to_f32(qh[lane * 2]), k0, d_bf16_to_f32(qh[lane * 2 + 1]) * k1);
        dp = d_warp_sum64(dp);
        if (lane == 0) {
            const float r = (dp > 0.0f) ? dp : 0.0f;
            total = fmaf(w_sel_f32[h] * kWScale, r, total);
        }
    }
    if (lane == 0) scores_f32[tok] = total;
}

__global__ __launch_bounds__(256, 8)
void k_mhc_mean4_bf16(
    const uint16_t* __restrict__ res_4xH,
    uint16_t* __restrict__ out_H,
    int H)
{
    const int m = int(blockIdx.y);
    const int h = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (h >= H) return;
    const uint16_t* __restrict__ r = res_4xH + size_t(m) * size_t(4 * H);
    const float s = d_bf16_to_f32(r[h])
                  + d_bf16_to_f32(r[H + h])
                  + d_bf16_to_f32(r[2 * H + h])
                  + d_bf16_to_f32(r[3 * H + h]);
    out_H[size_t(m) * size_t(H) + size_t(h)] = d_f32_to_bf16(0.25f * s);
}

__global__ __launch_bounds__(256, 8)
void k_dspark_swa_ring_write(
    const uint16_t* __restrict__ kv_norm_bf16, // [M, 512]
    const int32_t* __restrict__ pos_ids,       // [M]
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ swa_ring_256)       // [256, 512]
{
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);  // 0..255 -> 2 dims/thread = 512!
    const int32_t pos = pos_ids[m];
    if (pos < 0) return;
    const uint16_t* __restrict__ src = kv_norm_bf16 + size_t(m) * 512u;
    uint16_t* __restrict__ dst = swa_ring_256 + size_t(pos & 255) * 512u;

    float v0 = d_bf16_to_f32(src[tid * 2]);
    float v1 = d_bf16_to_f32(src[tid * 2 + 1]);
    if (tid >= 224) { // dims [448..511] interleaved plain RoPE at rope_theta=8,000,000!
        float c, s;
        d_rope2d_lookup(pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(v0, c, -(v1 * s));
        const float r1 = fmaf(v1, c,   v0 * s);
        v0 = r0; v1 = r1;
    }
    dst[tid * 2]     = d_f32_to_bf16(v0);
    dst[tid * 2 + 1] = d_f32_to_bf16(v1);
}

// 3L MoE DSpark Non-Causal Intra-Block SWA Attention (1-wavefront 64-thread, uint4, 0 barriers, 0 LDS!)
__global__ __launch_bounds__(64, 4)
void k_dspark_swa_noncausal_block_attn(
    const uint16_t* __restrict__ q_bf16,       // [K, 64, 512]
    const uint16_t* __restrict__ swa_ring_256, // [256, 512]
    const float* __restrict__ attn_sink,       // [64]
    uint16_t* __restrict__ out_bf16,           // [K, 64, 512]
    int32_t query_pos0, int K, int sliding_window)
{
    const int h = int(blockIdx.x);   // 0..63
    const int qi = int(blockIdx.y);  // 0..K-1
    const int lane = int(threadIdx.x);

    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + (size_t(qi) * 64u + size_t(h)) * 512u)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    constexpr float kScale = 0.04419417382415922f; // 1 / sqrt(512)

    const int32_t q_pos = query_pos0 + qi;
    const int32_t p_beg = (q_pos - sliding_window + 1 > 0) ? (q_pos - sliding_window + 1) : 0;
    const int32_t p_end = query_pos0 + K;

    float m_max = attn_sink ? attn_sink[h] : -1e30f;
    float l_sum = attn_sink ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    for (int32_t p = p_beg; p < p_end; ++p) {
        const uint4 kv4 = reinterpret_cast<const uint4*>(swa_ring_256 + size_t(p & 255) * 512u)[lane];
        const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
        const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
        const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
        const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
        const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
        const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
        const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
        const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
        float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                   fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
        const float score = d_warp_sum64(dp) * kScale;
        const float m_new = (score > m_max) ? score : m_max;
        const float alpha = __expf(m_max - m_new);
        const float beta  = __expf(score - m_new);
        l_sum = fmaf(l_sum, alpha, beta);
        o0 = fmaf(o0, alpha, beta * k0); o1 = fmaf(o1, alpha, beta * k1);
        o2 = fmaf(o2, alpha, beta * k2); o3 = fmaf(o3, alpha, beta * k3);
        o4 = fmaf(o4, alpha, beta * k4); o5 = fmaf(o5, alpha, beta * k5);
        o6 = fmaf(o6, alpha, beta * k6); o7 = fmaf(o7, alpha, beta * k7);
        m_max = m_new;
    }
    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    reinterpret_cast<uint4*>(out_bf16 + (size_t(qi) * 64u + size_t(h)) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l,
                      o4 * inv_l, o5 * inv_l, o6 * inv_l, o7 * inv_l);
}

__global__ __launch_bounds__(256, 8)
void k_dspark_add_bias_multimax_f32(
    const float* __restrict__ base_logits,
    const float* __restrict__ bias,
    const float* __restrict__ r,
    const float* __restrict__ t,
    float* __restrict__ out_logits,
    int V)
{
    const int v = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (v >= V) return;
    const float bx = base_logits[v];
    if (isinf(bx) && bx < 0.0f) { out_logits[v] = bx; return; }
    float x = bx + bias[v];
    if (r != nullptr && t != nullptr) {
        const float r0x = r[0] - x; const float a = (r0x <= 0.0f) ? 0.0f : r0x;
        const float xr1 = x - r[1]; const float b = (xr1 <= 0.0f) ? 0.0f : xr1;
        const float r2x = r[2] - x; const float c = (r2x <= 0.0f) ? 0.0f : r2x;
        const float xr3 = x - r[3]; const float d = (xr3 <= 0.0f) ? 0.0f : xr3;
        float y = __fadd_rn(x, __fmul_rn(t[0], a));
        y = __fadd_rn(y, __fmul_rn(t[1], b));
        y = __fadd_rn(y, __fmul_rn(t[2], __fmul_rn(c, c)));
        y = __fadd_rn(y, __fmul_rn(t[3], __fmul_rn(d, d)));
        x = y;
    }
    out_logits[v] = x;
}

// ---- Batched Zero-Scratch Pure-GPU DSA Indexer Scoring & Exact 3-Pass LDS Radix Top-2048 Selector ----
__device__ __forceinline__ uint32_t d_f32_to_ord(float f) {
    uint32_t u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__global__ __launch_bounds__(256, 4)
void k_dsa_indexer_score_topk8_batch(
    const uint16_t* __restrict__ q_sel_bf16,
    const float* __restrict__ w_sel_f32,
    const uint16_t* __restrict__ ik_cache_bf16,
    float* __restrict__ scores_batch_f32,
    int32_t pos0, int t0, int score_stride)
{
    const int bi  = int(blockIdx.y);
    const int t   = t0 + bi;
    const int pos = pos0 + t;
    const int n_hist = pos - 127;
    if (n_hist <= 2048) return;

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int tok  = int(blockIdx.x) * 4 + warp;
    if (tok >= n_hist) return;

    const uint16_t* __restrict__ kt = ik_cache_bf16 + size_t(tok) * 128u;
    const float k0 = d_bf16_to_f32(kt[lane * 2]);
    const float k1 = d_bf16_to_f32(kt[lane * 2 + 1]);
    constexpr float kWScale = 0.011048543456039806f;

    const uint16_t* __restrict__ q_t = q_sel_bf16 + size_t(t) * 8u * 128u;
    const float*    __restrict__ w_t = w_sel_f32  + size_t(t) * 8u;

    float total = 0.0f;
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        const uint16_t* __restrict__ qh = q_t + size_t(h) * 128u;
        float dp = fmaf(d_bf16_to_f32(qh[lane * 2]), k0, d_bf16_to_f32(qh[lane * 2 + 1]) * k1);
        dp = d_warp_sum64(dp);
        if (lane == 0) {
            const float r = (dp > 0.0f) ? dp : 0.0f;
            total = fmaf(w_t[h] * kWScale, r, total);
        }
    }
    if (lane == 0) scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = total;
}

__global__ __launch_bounds__(256, 4)
void k_dsa_indexer_score_full_batch(
    const uint16_t* __restrict__ iq_bf16,
    const uint16_t* __restrict__ iw_bf16,
    const uint16_t* __restrict__ ik_cache_bf16,
    float* __restrict__ scores_batch_f32,
    int32_t pos0, int t0, int score_stride)
{
    __shared__ float s_w[4];
    const int bi  = int(blockIdx.y);
    const int t   = t0 + bi;
    const int pos = pos0 + t;
    const int n_hist = pos - 127;
    if (n_hist <= 2048) return;

    const int tok = int(blockIdx.x);
    if (tok >= n_hist) return;

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const uint16_t* __restrict__ kt = ik_cache_bf16 + size_t(tok) * 128u;
    const float k0 = d_bf16_to_f32(kt[lane * 2]);
    const float k1 = d_bf16_to_f32(kt[lane * 2 + 1]);
    constexpr float kWScale = 0.011048543456039806f;

    const uint16_t* __restrict__ iq_t = iq_bf16 + size_t(t) * 64u * 128u;
    const uint16_t* __restrict__ iw_t = iw_bf16 + size_t(t) * 64u;

    float sum_h = 0.0f;
    for (int h = warp; h < 64; h += 4) {
        const uint16_t* __restrict__ qh = iq_t + size_t(h) * 128u;
        float dp = fmaf(d_bf16_to_f32(qh[lane * 2]), k0, d_bf16_to_f32(qh[lane * 2 + 1]) * k1);
        dp = d_warp_sum64(dp);
        if (lane == 0) {
            const float r = (dp > 0.0f) ? dp : 0.0f;
            sum_h = fmaf(d_bf16_to_f32(iw_t[h]) * kWScale, r, sum_h);
        }
    }
    if (lane == 0) s_w[warp] = sum_h;
    __syncthreads();
    if (tid == 0) scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
}

__global__ __launch_bounds__(256, 2)
void k_dsa_select_topk2048_gpu(
    const float* __restrict__ scores_batch_f32,
    int32_t* __restrict__ sel_idx_MxS,
    int32_t* __restrict__ sel_cnts,
    int32_t pos0, int t0, int score_stride, int stride_sel)
{
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_meta[6];   // 0=b1, 1=need1, 2=b2, 3=need2, 4=thresh_u, 5=need_eq
    __shared__ uint32_t s_wgt[4];
    __shared__ uint32_t s_weq[4];

    const int bi  = int(blockIdx.x);
    const int t   = t0 + bi;
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int32_t pos = pos0 + t;
    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    const int n_hist = swa_start;
    int32_t* __restrict__ dst = sel_idx_MxS + size_t(t) * size_t(stride_sel);

    if (n_hist <= 2048) {
        for (int p = tid; p <= pos; p += 256) dst[p] = p;
        if (tid == 0) sel_cnts[t] = pos + 1;
        return;
    }

    const float* __restrict__ sc = scores_batch_f32 + size_t(bi) * size_t(score_stride);

    // Pass 1: 11-bit top bits (u >> 21)
    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int i = tid; i < n_hist; i += 256) {
        atomicAdd(&s_hist[d_f32_to_ord(sc[i]) >> 21], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0u, b1 = 0u;
        for (int b = 2047; b >= 0; --b) {
            const uint32_t c = s_hist[b];
            if (above + c >= 2048u) { b1 = uint32_t(b); break; }
            above += c;
        }
        s_meta[0] = b1;
        s_meta[1] = 2048u - above;
    }
    __syncthreads();
    const uint32_t b1    = s_meta[0];
    const uint32_t need1 = s_meta[1];

    // Pass 2: 11-bit mid bits ((u >> 10) & 2047) for elements matching b1
    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int i = tid; i < n_hist; i += 256) {
        const uint32_t u = d_f32_to_ord(sc[i]);
        if ((u >> 21) == b1) atomicAdd(&s_hist[(u >> 10) & 2047u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0u, b2 = 0u;
        for (int b = 2047; b >= 0; --b) {
            const uint32_t c = s_hist[b];
            if (above + c >= need1) { b2 = uint32_t(b); break; }
            above += c;
        }
        s_meta[2] = b2;
        s_meta[3] = need1 - above;
    }
    __syncthreads();
    const uint32_t p22   = (b1 << 11) | s_meta[2];
    const uint32_t need2 = s_meta[3];

    // Pass 3: 10-bit low bits (u & 1023) for elements matching p22
    for (int b = tid; b < 1024; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int i = tid; i < n_hist; i += 256) {
        const uint32_t u = d_f32_to_ord(sc[i]);
        if ((u >> 10) == p22) atomicAdd(&s_hist[u & 1023u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0u, b3 = 0u;
        for (int b = 1023; b >= 0; --b) {
            const uint32_t c = s_hist[b];
            if (above + c >= need2) { b3 = uint32_t(b); break; }
            above += c;
        }
        s_meta[4] = (p22 << 10) | b3;
        s_meta[5] = need2 - above;
    }
    __syncthreads();
    const uint32_t thresh_u = s_meta[4];
    const uint32_t need_eq  = s_meta[5];
    const uint32_t total_gt = 2048u - need_eq;

    // Pass 4: 100% deterministic ascending-index emission (zero global atomics!)
    const uint64_t below = (1ULL << lane) - 1ULL;
    uint32_t gt_so_far = 0u;
    uint32_t eq_so_far = 0u;
    for (int base = 0; base < n_hist; base += 256) {
        const int i = base + tid;
        bool is_gt = false;
        bool is_eq = false;
        if (i < n_hist) {
            const uint32_t u = d_f32_to_ord(sc[i]);
            is_gt = (u > thresh_u);
            is_eq = (u == thresh_u);
        }
        const uint64_t m_gt = __ballot(is_gt);
        const uint32_t w_gt = uint32_t(__popcll(m_gt));
        const uint32_t l_gt = uint32_t(__popcll(m_gt & below));
        const uint64_t m_eq = __ballot(is_eq);
        const uint32_t w_eq = uint32_t(__popcll(m_eq));
        const uint32_t l_eq = uint32_t(__popcll(m_eq & below));

        if (lane == 0) {
            s_wgt[warp] = w_gt;
            s_weq[warp] = w_eq;
        }
        __syncthreads();

        uint32_t wg_off = 0u, we_off = 0u;
        if (warp >= 1) { wg_off += s_wgt[0]; we_off += s_weq[0]; }
        if (warp >= 2) { wg_off += s_wgt[1]; we_off += s_weq[1]; }
        if (warp >= 3) { wg_off += s_wgt[2]; we_off += s_weq[2]; }

        if (is_gt) {
            const uint32_t gi = gt_so_far + wg_off + l_gt;
            if (gi < total_gt) dst[gi] = i;
        }
        if (is_eq) {
            const uint32_t ei = eq_so_far + we_off + l_eq;
            if (ei < need_eq) dst[total_gt + ei] = i;
        }
        gt_so_far += s_wgt[0] + s_wgt[1] + s_wgt[2] + s_wgt[3];
        eq_so_far += s_weq[0] + s_weq[1] + s_weq[2] + s_weq[3];
        __syncthreads();
    }

    // Append local SWA window [swa_start .. pos]
    const int swa_len = pos - swa_start + 1;
    for (int k = tid; k < swa_len; k += 256) dst[2048 + k] = swa_start + k;
    if (tid == 0) sel_cnts[t] = 2048 + swa_len;
}

void launch_flashindexer_select_heads(
    const float* d_gate_logits_f32, const float* d_bias_f32,
    const uint16_t* d_iq_bf16, const uint16_t* d_iw_bf16,
    uint16_t* d_q_sel_bf16, float* d_w_sel_f32, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_flashindexer_select_heads, dim3(M), dim3(64), 0, stream,
                       d_gate_logits_f32, d_bias_f32, d_iq_bf16, d_iw_bf16, d_q_sel_bf16, d_w_sel_f32);
}

void launch_dsa_indexer_score_topk8(
    const uint16_t* d_q_sel_bf16, const float* d_w_sel_f32,
    const uint16_t* d_ik_cache_bf16, float* d_scores_f32, int n_hist, hipStream_t stream) {
    if (n_hist <= 0) return;
    hipLaunchKernelGGL(k_dsa_indexer_score_topk8, dim3((n_hist + 3) / 4), dim3(256), 0, stream,
                       d_q_sel_bf16, d_w_sel_f32, d_ik_cache_bf16, d_scores_f32, n_hist);
}

void launch_dsa_indexer_score_topk8_batch(
    const uint16_t* d_q_sel_bf16, const float* d_w_sel_f32,
    const uint16_t* d_ik_cache_bf16, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream) {
    if (B_sc <= 0 || score_stride <= 0) return;
    hipLaunchKernelGGL(k_dsa_indexer_score_topk8_batch, dim3((score_stride + 3) / 4, B_sc), dim3(256), 0, stream,
                       d_q_sel_bf16, d_w_sel_f32, d_ik_cache_bf16, d_scores_batch_f32, pos0, t0, score_stride);
}

void launch_dsa_indexer_score_full_batch(
    const uint16_t* d_iq_bf16, const uint16_t* d_iw_bf16,
    const uint16_t* d_ik_cache_bf16, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream) {
    if (B_sc <= 0 || score_stride <= 0) return;
    hipLaunchKernelGGL(k_dsa_indexer_score_full_batch, dim3(score_stride, B_sc), dim3(256), 0, stream,
                       d_iq_bf16, d_iw_bf16, d_ik_cache_bf16, d_scores_batch_f32, pos0, t0, score_stride);
}

// ============================================================================
// Wave-6 Zero-Scratch DSA Decode Kernels (ScratchSize == 0, 0 spills, 0 trig, 0 slc):
//   1. k_dsa_indexer_score_topk8_multirow: LDS Q-stationary multi-row FlashIndexer (6x-8x less ik_cache reads, 0.0f ULP diff)
//   2. k_dsa_topk2048_pass{1..4b}: 32-block x B_sc parallel Top-2048 radix selector with VGPR zero-score bypass
// ============================================================================
static constexpr uint32_t kWave6ZeroOrdKey   = 0x80000000u; // d_f32_to_ord(0.0f)
static constexpr uint32_t kWave6ZeroTop11Bin = 1024u;       // kWave6ZeroOrdKey >> 21
static constexpr int      kWave6RadixBlocks  = 32;          // 32 x B_sc blocks across 60 CUs

__device__ __forceinline__ uint32_t d_warp_sum64_u32(uint32_t v) {
    #pragma unroll
    for (int mask = 32; mask >= 1; mask >>= 1) v += __shfl_xor(v, mask, 64);
    return v;
}

__global__ __launch_bounds__(256, 4)
void k_dsa_indexer_score_topk8_multirow(
    const uint16_t* __restrict__ q_sel_bf16,
    const float*    __restrict__ w_sel_f32,
    const uint16_t* __restrict__ ik_cache_bf16,
    float*          __restrict__ scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int score_stride)
{
    // 0 B LDS -> max occupancy; contiguous blocks keep ik_cache coalesced in L2 while Q (<=12 KiB) hits L1 at 100%!
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int tok = int(blockIdx.x) * 4 + warp;
    const int min_hist = (pos0 + t0 >= 127) ? (pos0 + t0 - 127) : 0;
    const int max_hist = (pos0 + t0 + B_sc - 1 >= 127) ? (pos0 + t0 + B_sc - 1 - 127) : 0;
    if (tok >= max_hist) return;

    const uint32_t kp = reinterpret_cast<const uint32_t*>(ik_cache_bf16 + size_t(tok) * 128u)[lane];
    const float k0 = d_bf16_to_f32(uint16_t(kp & 0xFFFFu));
    const float k1 = d_bf16_to_f32(uint16_t(kp >> 16));
    constexpr float kWScale = 0.011048543456039806f;

    const uint32_t* __restrict__ q32_base = reinterpret_cast<const uint32_t*>(q_sel_bf16 + size_t(t0) * 8u * 128u);
    const float*    __restrict__ w_base   = w_sel_f32 + size_t(t0) * 8u;
    const bool all_valid = (min_hist > 2048 && tok < min_hist);

    for (int bi = 0; bi < B_sc; ++bi) {
        if (!all_valid) {
            const int n_hist = (pos0 + t0 + bi >= 127) ? (pos0 + t0 + bi - 127) : 0;
            if (n_hist <= 2048 || tok >= n_hist) continue;
        }
        const uint32_t* __restrict__ qb = q32_base + size_t(bi) * 8u * 64u;
        const float*    __restrict__ wb = w_base   + size_t(bi) * 8u;
        float total = 0.0f;
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            const uint32_t qp = qb[size_t(h) * 64u + size_t(lane)];
            float dp = fmaf(d_bf16_to_f32(uint16_t(qp & 0xFFFFu)), k0,
                            d_bf16_to_f32(uint16_t(qp >> 16)) * k1);
            dp = d_warp_sum64(dp);
            if (lane == 0) {
                const float r = (dp > 0.0f) ? dp : 0.0f;
                total = fmaf(wb[h] * kWScale, r, total);
            }
        }
        if (lane == 0) {
            scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = total;
        }
    }
}

// Wave-7 Zero-Scratch 4-Lane VALU DPP FlashIndexer (0 ds_bpermute, 100% IEEE-754 bit-identical to d_warp_sum64!)
// Persistent 180-block grid (3 blocks/CU x 60 CUs, 16,640 B LDS -> 12 waves/CU, 0 tail imbalance!) caches
// s_q[8*8*128] + s_w[8*8] ONCE per block and unpacks k0[16], k1[16] ONCE per history token across all B_sc x 8 heads.
__global__ __launch_bounds__(256, 3)
void k_dsa_indexer_score_topk8_dpp4(
    const uint16_t* __restrict__ q_sel_bf16,       // [total_M, 8, 128]
    const float*    __restrict__ w_sel_f32,        // [total_M, 8]
    const uint16_t* __restrict__ ik_cache_bf16,    // [max_seq, 128]
    float*          __restrict__ scores_batch_f32, // [B_sc, score_stride]
    int32_t pos0, int t0, int B_sc, int score_stride)
{
    __shared__ uint16_t s_q[8 * 8 * 128]; // 16,384 B
    __shared__ float    s_w[8 * 8];       //    256 B (total 16,640 B -> 3 blocks/CU = 12 waves/CU!)

    const int tid = int(threadIdx.x);
    const int q_pairs = B_sc * 8 * 64;
    const uint32_t* __restrict__ q_src32 =
        reinterpret_cast<const uint32_t*>(q_sel_bf16 + size_t(t0) * 8u * 128u);
    uint32_t* __restrict__ sq32 = reinterpret_cast<uint32_t*>(s_q);
    for (int i = tid; i < q_pairs; i += 256) {
        sq32[i] = q_src32[i];
    }
    if (tid < B_sc * 8) {
        s_w[tid] = w_sel_f32[size_t(t0) * 8u + size_t(tid)];
    }
    __syncthreads();

    const int g        = tid & 3;  // 0..3 within hardware DPP quad
    const int q_in_blk = tid >> 2; // 0..63 tokens per block
    const int min_hist = (pos0 + t0 >= 127) ? (pos0 + t0 - 127) : 0;
    const int max_pos  = pos0 + t0 + B_sc - 1;
    const int max_hist = (max_pos >= 127) ? (max_pos - 127) : 0;
    constexpr float kWScale = 0.011048543456039806f;

    const int stride_tok = int(gridDim.x) * 64;
    for (int tok = int(blockIdx.x) * 64 + q_in_blk; tok < max_hist; tok += stride_tok) {
        // Coalesced 32-bit loads + unpack ONCE per history token across all B_sc x 8 heads!
        const uint32_t* __restrict__ kt32 =
            reinterpret_cast<const uint32_t*>(ik_cache_bf16 + size_t(tok) * 128u);
        float k0[16], k1[16];
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            const uint32_t pk = kt32[4 * m + g];
            k0[m] = __uint_as_float((pk & 0xFFFFu) << 16);
            k1[m] = __uint_as_float(pk & 0xFFFF0000u);
        }

        const bool all_valid = (min_hist > 2048 && tok < min_hist);
        for (int bi = 0; bi < B_sc; ++bi) {
            if (!all_valid) {
                const int n_hist = (pos0 + t0 + bi >= 127) ? (pos0 + t0 + bi - 127) : 0;
                if (n_hist <= 2048 || tok >= n_hist) continue;
            }

            const uint32_t* __restrict__ qb32 = sq32 + bi * 8 * 64;
            const float*    __restrict__ wb   = s_w  + bi * 8;
            float total = 0.0f;

            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const uint32_t* __restrict__ qh32 = qb32 + h * 64;
                float r[8];
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const uint32_t pqa = qh32[4 * m + g];
                    const float dpa = fmaf(__uint_as_float((pqa & 0xFFFFu) << 16), k0[m],
                                           __uint_as_float(pqa & 0xFFFF0000u) * k1[m]);
                    const uint32_t pqb = qh32[4 * (m + 8) + g];
                    const float dpb = fmaf(__uint_as_float((pqb & 0xFFFFu) << 16), k0[m + 8],
                                           __uint_as_float(pqb & 0xFFFF0000u) * k1[m + 8]);
                    // Exact binary-tree step 1 matching d_warp_sum64 mask=32:
                    r[m] = dpa + dpb;
                }
                // Exact binary-tree steps 2..4 matching d_warp_sum64 masks 16, 8, 4:
                r[0] += r[4]; r[1] += r[5]; r[2] += r[6]; r[3] += r[7];
                r[0] += r[2]; r[1] += r[3];
                float dp = r[0] + r[1];

                // Exact VALU DPP quad_perm fold matching d_warp_sum64 masks 2 (0x4E) and 1 (0xB1), 0 ds_bpermute!
                dp += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(dp), 0x4E, 0xf, 0xf, true));
                dp += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(dp), 0xB1, 0xf, 0xf, true));

                if (g == 0) {
                    const float pos_dp = (dp > 0.0f) ? dp : 0.0f;
                    total = fmaf(wb[h] * kWScale, pos_dp, total);
                }
            }
            if (g == 0) {
                scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = total;
            }
        }
    }
}

// Wave-7 Split-K 4-Head HCA Compressed-KV + SWA Decode (M<=8, 60-CU saturation, 0 B new VRAM, M=1==M=6 bit-identical!)
// Pass 1: dim3(n_heads / 4, M, n_comp_splits + 1), dim3(256) -> 4 warps = 4 heads share s_k[4][64] (4 KiB LDS)!
//   s in [0 .. n_comp_splits - 1]: chunk of compressed KV blocks; s == n_comp_splits: <=128 SWA ring tokens in parallel!
__global__ __launch_bounds__(256, 4)
void k_hca_decode_splitk_p1(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ comp_kv_bf16,
    const uint16_t* __restrict__ swa_ring_bf16,
    float* __restrict__ part_o_f32,
    float* __restrict__ part_ml_f32,
    int n_heads, int32_t pos0, int M, int num_comp, int n_comp_splits)
{
    __shared__ uint4 s_k[4][64]; // 4 KiB LDS -> 4 heads share every KV load!
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6; // 0..3
    const int lane = tid & 63; // 0..63 -> 8 dims/lane
    const int h    = int(blockIdx.x) * 4 + warp;
    const int m    = int(blockIdx.y);
    const int s    = int(blockIdx.z); // 0..n_comp_splits (last split = SWA ring!)
    const int32_t pos = pos0 + m;

    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    constexpr float kScale = 0.04419417382415922f;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    auto step_kv_tile4 = [&](int cnt) {
        auto score_kv = [&](const uint4 kv4) -> float {
            const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
            const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
            const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
            const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
            const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
            const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
            const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
            const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
            const float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                             fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
            return d_warp_sum64(dp) * kScale;
        };
        const float sc0 = (cnt > 0) ? score_kv(s_k[0][lane]) : -1e30f;
        const float sc1 = (cnt > 1) ? score_kv(s_k[1][lane]) : -1e30f;
        const float sc2 = (cnt > 2) ? score_kv(s_k[2][lane]) : -1e30f;
        const float sc3 = (cnt > 3) ? score_kv(s_k[3][lane]) : -1e30f;
        float m_blk = fmaxf(m_max, fmaxf(fmaxf(sc0, sc1), fmaxf(sc2, sc3)));
        const float alpha = __expf(m_max - m_blk);
        o0 *= alpha; o1 *= alpha; o2 *= alpha; o3 *= alpha;
        o4 *= alpha; o5 *= alpha; o6 *= alpha; o7 *= alpha;
        l_sum *= alpha;
        m_max = m_blk;
        auto acc_kv = [&](const uint4 kv4, float sc) {
            const float p = __expf(sc - m_blk);
            l_sum += p;
            o0 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu)), o0);
            o1 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.x >> 16)),     o1);
            o2 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu)), o2);
            o3 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.y >> 16)),     o3);
            o4 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu)), o4);
            o5 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.z >> 16)),     o5);
            o6 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu)), o6);
            o7 = fmaf(p, d_bf16_to_f32(uint16_t(kv4.w >> 16)),     o7);
        };
        if (cnt > 0) acc_kv(s_k[0][lane], sc0);
        if (cnt > 1) acc_kv(s_k[1][lane], sc1);
        if (cnt > 2) acc_kv(s_k[2][lane], sc2);
        if (cnt > 3) acc_kv(s_k[3][lane], sc3);
    };

    if (s < n_comp_splits) {
        const int chunk = (num_comp + n_comp_splits - 1) / n_comp_splits;
        const int b_beg = s * chunk;
        const int b_end = (b_beg + chunk < num_comp) ? (b_beg + chunk) : num_comp;
        for (int b0 = b_beg; b0 < b_end; b0 += 4) {
            if (b0 + warp < b_end) {
                s_k[warp][lane] = reinterpret_cast<const uint4*>(comp_kv_bf16 + size_t(b0 + warp) * 512u)[lane];
            }
            __syncthreads();
            const int cnt = (b_end - b0 < 4) ? (b_end - b0) : 4;
            step_kv_tile4(cnt);
            __syncthreads();
        }
    } else {
        const int swa_start = (pos >= 127) ? (pos - 127) : 0;
        for (int p0 = swa_start; p0 <= pos; p0 += 4) {
            const int p_load = p0 + warp;
            if (p_load <= pos) {
                s_k[warp][lane] = reinterpret_cast<const uint4*>(swa_ring_bf16 + size_t(p_load & 255) * 512u)[lane];
            }
            __syncthreads();
            const int cnt = (pos - p0 + 1 < 4) ? (pos - p0 + 1) : 4;
            step_kv_tile4(cnt);
            __syncthreads();
        }
    }

    const size_t slot = (size_t(s) * size_t(M) + size_t(m)) * size_t(n_heads) + size_t(h);
    float4* __restrict__ po = reinterpret_cast<float4*>(part_o_f32 + slot * 512u);
    po[lane * 2]     = make_float4(o0, o1, o2, o3);
    po[lane * 2 + 1] = make_float4(o4, o5, o6, o7);
    if (lane == 0) {
        part_ml_f32[slot * 2u]      = m_max;
        part_ml_f32[slot * 2u + 1u] = l_sum;
    }
}

// Pass 2: Merge n_splits (= n_comp_splits + 1) partial online-softmax states + attn_sink -> BF16 [M, n_heads, 512]
__global__ __launch_bounds__(64, 8)
void k_hca_decode_splitk_merge(
    const float* __restrict__ part_o_f32,
    const float* __restrict__ part_ml_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads, int M, int n_splits)
{
    const int h    = int(blockIdx.x);
    const int m    = int(blockIdx.y);
    const int lane = int(threadIdx.x); // 0..63 -> 8 dims/lane
    const size_t row = size_t(m) * size_t(n_heads) + size_t(h);
    const size_t stride_slots = size_t(M) * size_t(n_heads);

    float m_tot = attn_sink[h];
    for (int s = 0; s < n_splits; ++s) {
        const float ms = part_ml_f32[(size_t(s) * stride_slots + row) * 2u];
        if (ms > m_tot) m_tot = ms;
    }

    float l_tot = __expf(attn_sink[h] - m_tot);
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const size_t slot = size_t(s) * stride_slots + row;
        const float ls = part_ml_f32[slot * 2u + 1u];
        if (ls <= 0.0f) continue;
        const float ms = part_ml_f32[slot * 2u];
        const float sc = __expf(ms - m_tot);
        l_tot = fmaf(ls, sc, l_tot);
        const float4* __restrict__ po = reinterpret_cast<const float4*>(part_o_f32 + slot * 512u);
        const float4 v0 = po[lane * 2];
        const float4 v1 = po[lane * 2 + 1];
        o0 = fmaf(v0.x, sc, o0); o1 = fmaf(v0.y, sc, o1);
        o2 = fmaf(v0.z, sc, o2); o3 = fmaf(v0.w, sc, o3);
        o4 = fmaf(v1.x, sc, o4); o5 = fmaf(v1.y, sc, o5);
        o6 = fmaf(v1.z, sc, o6); o7 = fmaf(v1.w, sc, o7);
    }

    const float inv_d = 1.0f / l_tot;
    reinterpret_cast<uint4*>(out_bf16 + row * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

__global__ __launch_bounds__(256, 4)
void k_dsa_topk2048_pass1_hist(
    const float* __restrict__ scores_batch_f32,
    uint32_t*    __restrict__ g_hist1,
    int32_t*     __restrict__ sel_idx_MxS,
    int32_t*     __restrict__ sel_cnts,
    int32_t pos0, int t0, int score_stride, int stride_sel)
{
    __shared__ uint32_t s_hist[2048];
    const int blk  = int(blockIdx.x);
    const int bi   = int(blockIdx.y);
    const int t    = t0 + bi;
    const int tid  = int(threadIdx.x);
    const int lane = tid & 63;

    const int32_t pos = pos0 + t;
    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    const int n_hist = swa_start;

    if (n_hist <= 2048) {
        if (blk == 0) {
            int32_t* __restrict__ dst = sel_idx_MxS + size_t(t) * size_t(stride_sel);
            for (int p = tid; p <= pos; p += 256) dst[p] = p;
            if (tid == 0) sel_cnts[t] = pos + 1;
        }
        return;
    }

    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();

    const int chunk = (n_hist + kWave6RadixBlocks - 1) / kWave6RadixBlocks;
    const int i_beg = blk * chunk;
    const int i_end = (i_beg + chunk < n_hist) ? (i_beg + chunk) : n_hist;
    const float* __restrict__ sc = scores_batch_f32 + size_t(bi) * size_t(score_stride);

    uint32_t zero_cnt = 0u;
    for (int i = i_beg + tid; i < i_end; i += 256) {
        const uint32_t u = d_f32_to_ord(sc[i]);
        if (u == kWave6ZeroOrdKey) ++zero_cnt;
        else atomicAdd(&s_hist[u >> 21], 1u);
    }
    zero_cnt = d_warp_sum64_u32(zero_cnt);
    if (lane == 0 && zero_cnt != 0u) atomicAdd(&s_hist[kWave6ZeroTop11Bin], zero_cnt);
    __syncthreads();

    uint32_t* __restrict__ out = g_hist1 + (size_t(bi) * kWave6RadixBlocks + size_t(blk)) * 2048u;
    for (int b = tid; b < 2048; b += 256) out[b] = s_hist[b];
}

__global__ __launch_bounds__(256, 4)
void k_dsa_topk2048_pass2_hist(
    const float*    __restrict__ scores_batch_f32,
    const uint32_t* __restrict__ g_hist1,
    uint32_t*       __restrict__ g_hist2,
    uint32_t*       __restrict__ g_meta,
    int32_t pos0, int t0, int score_stride)
{
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_m[2];

    const int blk  = int(blockIdx.x);
    const int bi   = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int lane = tid & 63;

    const int32_t pos = pos0 + t0 + bi;
    const int n_hist = (pos >= 127) ? (pos - 127) : 0;
    if (n_hist <= 2048) return;

    const uint32_t* __restrict__ h1 = g_hist1 + size_t(bi) * kWave6RadixBlocks * 2048u;
    for (int b = tid; b < 2048; b += 256) {
        uint32_t sum = 0u;
        #pragma unroll 8
        for (int c = 0; c < kWave6RadixBlocks; ++c) sum += h1[size_t(c) * 2048u + size_t(b)];
        s_hist[b] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t above = 0u, b1 = 0u;
        for (int b = 2047; b >= 0; --b) {
            const uint32_t cnt = s_hist[b];
            if (above + cnt >= 2048u) { b1 = uint32_t(b); break; }
            above += cnt;
        }
        s_m[0] = b1;
        s_m[1] = 2048u - above;
        if (blk == 0) {
            g_meta[bi * 8 + 0] = b1;
            g_meta[bi * 8 + 1] = 2048u - above;
        }
    }
    __syncthreads();
    const uint32_t b1 = s_m[0];

    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();

    const int chunk = (n_hist + kWave6RadixBlocks - 1) / kWave6RadixBlocks;
    const int i_beg = blk * chunk;
    const int i_end = (i_beg + chunk < n_hist) ? (i_beg + chunk) : n_hist;
    const float* __restrict__ sc = scores_batch_f32 + size_t(bi) * size_t(score_stride);

    uint32_t zero_cnt = 0u;
    for (int i = i_beg + tid; i < i_end; i += 256) {
        const uint32_t u = d_f32_to_ord(sc[i]);
        if ((u >> 21) == b1) {
            if (u == kWave6ZeroOrdKey) ++zero_cnt;
            else atomicAdd(&s_hist[(u >> 10) & 2047u], 1u);
        }
    }
    zero_cnt = d_warp_sum64_u32(zero_cnt);
    if (lane == 0 && zero_cnt != 0u) atomicAdd(&s_hist[0], zero_cnt);
    __syncthreads();

    uint32_t* __restrict__ out = g_hist2 + (size_t(bi) * kWave6RadixBlocks + size_t(blk)) * 2048u;
    for (int b = tid; b < 2048; b += 256) out[b] = s_hist[b];
}

__global__ __launch_bounds__(256, 4)
void k_dsa_topk2048_pass3_hist(
    const float*    __restrict__ scores_batch_f32,
    const uint32_t* __restrict__ g_hist2,
    uint32_t*       __restrict__ g_hist3,
    uint32_t*       __restrict__ g_meta,
    int32_t pos0, int t0, int score_stride)
{
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_m[2];

    const int blk  = int(blockIdx.x);
    const int bi   = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int lane = tid & 63;

    const int32_t pos = pos0 + t0 + bi;
    const int n_hist = (pos >= 127) ? (pos - 127) : 0;
    if (n_hist <= 2048) return;

    const uint32_t b1    = g_meta[bi * 8 + 0];
    const uint32_t need1 = g_meta[bi * 8 + 1];

    const uint32_t* __restrict__ h2 = g_hist2 + size_t(bi) * kWave6RadixBlocks * 2048u;
    for (int b = tid; b < 2048; b += 256) {
        uint32_t sum = 0u;
        #pragma unroll 8
        for (int c = 0; c < kWave6RadixBlocks; ++c) sum += h2[size_t(c) * 2048u + size_t(b)];
        s_hist[b] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t above = 0u, b2 = 0u;
        for (int b = 2047; b >= 0; --b) {
            const uint32_t cnt = s_hist[b];
            if (above + cnt >= need1) { b2 = uint32_t(b); break; }
            above += cnt;
        }
        const uint32_t p22 = (b1 << 11) | b2;
        s_m[0] = p22;
        s_m[1] = need1 - above;
        if (blk == 0) {
            g_meta[bi * 8 + 2] = p22;
            g_meta[bi * 8 + 3] = need1 - above;
        }
    }
    __syncthreads();
    const uint32_t p22 = s_m[0];

    for (int b = tid; b < 1024; b += 256) s_hist[b] = 0u;
    __syncthreads();

    const int chunk = (n_hist + kWave6RadixBlocks - 1) / kWave6RadixBlocks;
    const int i_beg = blk * chunk;
    const int i_end = (i_beg + chunk < n_hist) ? (i_beg + chunk) : n_hist;
    const float* __restrict__ sc = scores_batch_f32 + size_t(bi) * size_t(score_stride);

    uint32_t zero_cnt = 0u;
    for (int i = i_beg + tid; i < i_end; i += 256) {
        const uint32_t u = d_f32_to_ord(sc[i]);
        if ((u >> 10) == p22) {
            if (u == kWave6ZeroOrdKey) ++zero_cnt;
            else atomicAdd(&s_hist[u & 1023u], 1u);
        }
    }
    zero_cnt = d_warp_sum64_u32(zero_cnt);
    if (lane == 0 && zero_cnt != 0u) atomicAdd(&s_hist[0], zero_cnt);
    __syncthreads();

    uint32_t* __restrict__ out = g_hist3 + (size_t(bi) * kWave6RadixBlocks + size_t(blk)) * 1024u;
    for (int b = tid; b < 1024; b += 256) out[b] = s_hist[b];
}

__global__ __launch_bounds__(256, 4)
void k_dsa_topk2048_pass4a_counts(
    const float*    __restrict__ scores_batch_f32,
    const uint32_t* __restrict__ g_hist3,
    uint32_t*       __restrict__ g_meta,
    uint32_t*       __restrict__ g_blk_cnt,
    int32_t pos0, int t0, int score_stride)
{
    __shared__ uint32_t s_hist[1024];
    __shared__ uint32_t s_wgt[4];
    __shared__ uint32_t s_weq[4];
    __shared__ uint32_t s_thresh;

    const int blk  = int(blockIdx.x);
    const int bi   = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int32_t pos = pos0 + t0 + bi;
    const int n_hist = (pos >= 127) ? (pos - 127) : 0;
    if (n_hist <= 2048) return;

    const uint32_t p22   = g_meta[bi * 8 + 2];
    const uint32_t need2 = g_meta[bi * 8 + 3];

    const uint32_t* __restrict__ h3 = g_hist3 + size_t(bi) * kWave6RadixBlocks * 1024u;
    for (int b = tid; b < 1024; b += 256) {
        uint32_t sum = 0u;
        #pragma unroll 8
        for (int c = 0; c < kWave6RadixBlocks; ++c) sum += h3[size_t(c) * 1024u + size_t(b)];
        s_hist[b] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t above = 0u, b3 = 0u;
        for (int b = 1023; b >= 0; --b) {
            const uint32_t cnt = s_hist[b];
            if (above + cnt >= need2) { b3 = uint32_t(b); break; }
            above += cnt;
        }
        const uint32_t thresh_u = (p22 << 10) | b3;
        const uint32_t need_eq  = need2 - above;
        s_thresh = thresh_u;
        if (blk == 0) {
            g_meta[bi * 8 + 4] = thresh_u;
            g_meta[bi * 8 + 5] = need_eq;
            g_meta[bi * 8 + 6] = 2048u - need_eq;
        }
    }
    __syncthreads();
    const uint32_t thresh_u = s_thresh;

    const int chunk = (n_hist + kWave6RadixBlocks - 1) / kWave6RadixBlocks;
    const int i_beg = blk * chunk;
    const int i_end = (i_beg + chunk < n_hist) ? (i_beg + chunk) : n_hist;
    const float* __restrict__ sc = scores_batch_f32 + size_t(bi) * size_t(score_stride);

    uint32_t my_gt = 0u, my_eq = 0u;
    for (int i = i_beg + tid; i < i_end; i += 256) {
        const uint32_t u = d_f32_to_ord(sc[i]);
        if (u > thresh_u) ++my_gt;
        else if (u == thresh_u) ++my_eq;
    }
    my_gt = d_warp_sum64_u32(my_gt);
    my_eq = d_warp_sum64_u32(my_eq);
    if (lane == 0) {
        s_wgt[warp] = my_gt;
        s_weq[warp] = my_eq;
    }
    __syncthreads();
    if (tid == 0) {
        g_blk_cnt[bi * 64 + blk]      = s_wgt[0] + s_wgt[1] + s_wgt[2] + s_wgt[3];
        g_blk_cnt[bi * 64 + 32 + blk] = s_weq[0] + s_weq[1] + s_weq[2] + s_weq[3];
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_topk2048_pass4b_scatter(
    const float*    __restrict__ scores_batch_f32,
    const uint32_t* __restrict__ g_meta,
    const uint32_t* __restrict__ g_blk_cnt,
    int32_t*        __restrict__ sel_idx_MxS,
    int32_t*        __restrict__ sel_cnts,
    int32_t pos0, int t0, int score_stride, int stride_sel)
{
    __shared__ uint32_t s_wgt[4];
    __shared__ uint32_t s_weq[4];

    const int blk  = int(blockIdx.x);
    const int bi   = int(blockIdx.y);
    const int t    = t0 + bi;
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int32_t pos = pos0 + t;
    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    const int n_hist = swa_start;
    if (n_hist <= 2048) return;

    const uint32_t thresh_u = g_meta[bi * 8 + 4];
    const uint32_t need_eq  = g_meta[bi * 8 + 5];
    const uint32_t total_gt = g_meta[bi * 8 + 6];

    uint32_t gt_so_far = 0u;
    uint32_t eq_so_far = 0u;
    for (int c = 0; c < blk; ++c) {
        gt_so_far += g_blk_cnt[bi * 64 + c];
        eq_so_far += g_blk_cnt[bi * 64 + 32 + c];
    }

    const int chunk = (n_hist + kWave6RadixBlocks - 1) / kWave6RadixBlocks;
    const int i_beg = blk * chunk;
    const int i_end = (i_beg + chunk < n_hist) ? (i_beg + chunk) : n_hist;
    const float* __restrict__ sc = scores_batch_f32 + size_t(bi) * size_t(score_stride);
    int32_t* __restrict__ dst = sel_idx_MxS + size_t(t) * size_t(stride_sel);
    const uint64_t below = (1ULL << lane) - 1ULL;

    for (int base = i_beg; base < i_end; base += 256) {
        const int i = base + tid;
        bool is_gt = false;
        bool is_eq = false;
        if (i < i_end) {
            const uint32_t u = d_f32_to_ord(sc[i]);
            is_gt = (u > thresh_u);
            is_eq = (u == thresh_u);
        }
        const uint64_t m_gt = __ballot(is_gt);
        const uint32_t w_gt = uint32_t(__popcll(m_gt));
        const uint32_t l_gt = uint32_t(__popcll(m_gt & below));
        const uint64_t m_eq = __ballot(is_eq);
        const uint32_t w_eq = uint32_t(__popcll(m_eq));
        const uint32_t l_eq = uint32_t(__popcll(m_eq & below));

        if (lane == 0) {
            s_wgt[warp] = w_gt;
            s_weq[warp] = w_eq;
        }
        __syncthreads();

        uint32_t wg_off = 0u, we_off = 0u;
        if (warp >= 1) { wg_off += s_wgt[0]; we_off += s_weq[0]; }
        if (warp >= 2) { wg_off += s_wgt[1]; we_off += s_weq[1]; }
        if (warp >= 3) { wg_off += s_wgt[2]; we_off += s_weq[2]; }

        if (is_gt) {
            const uint32_t gi = gt_so_far + wg_off + l_gt;
            if (gi < total_gt) dst[gi] = i;
        }
        if (is_eq) {
            const uint32_t ei = eq_so_far + we_off + l_eq;
            if (ei < need_eq) dst[total_gt + ei] = i;
        }
        gt_so_far += s_wgt[0] + s_wgt[1] + s_wgt[2] + s_wgt[3];
        eq_so_far += s_weq[0] + s_weq[1] + s_weq[2] + s_weq[3];
        __syncthreads();
    }

    if (blk == 0) {
        const int swa_len = pos - swa_start + 1;
        for (int k = tid; k < swa_len; k += 256) dst[2048 + k] = swa_start + k;
        if (tid == 0) sel_cnts[t] = 2048 + swa_len;
    }
}

void launch_dsa_select_topk2048_gpu(
    const float* d_scores_batch_f32, int32_t* d_sel_idx_MxS, int32_t* d_sel_cnts,
    int32_t pos0, int t0, int B_sc, int score_stride, int stride_sel, hipStream_t stream) {
    if (B_sc <= 0) return;
    hipLaunchKernelGGL(k_dsa_select_topk2048_gpu, dim3(B_sc), dim3(256), 0, stream,
                       d_scores_batch_f32, d_sel_idx_MxS, d_sel_cnts, pos0, t0, score_stride, stride_sel);
}

void launch_dsa_indexer_score_topk8_wave6(
    const uint16_t* d_q_sel_bf16, const float* d_w_sel_f32,
    const uint16_t* d_ik_cache_bf16, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream) {
    if (B_sc <= 0 || score_stride <= 0) return;
    const int grid_x = (score_stride + 3) / 4; // Coalesced contiguous ik_cache streaming (0 B LDS -> max occupancy!)
    for (int sub = 0; sub < B_sc; sub += 8) {
        const int cur_b = (B_sc - sub < 8) ? (B_sc - sub) : 8;
        hipLaunchKernelGGL(
            k_dsa_indexer_score_topk8_multirow,
            dim3(grid_x), dim3(256), 0, stream,
            d_q_sel_bf16, d_w_sel_f32, d_ik_cache_bf16,
            d_scores_batch_f32 + size_t(sub) * size_t(score_stride),
            pos0, t0 + sub, cur_b, score_stride);
    }
}

void launch_dsa_indexer_score_topk8_dpp4(
    const uint16_t* d_q_sel_bf16, const float* d_w_sel_f32,
    const uint16_t* d_ik_cache_bf16, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream) {
    if (B_sc <= 0 || score_stride <= 0) return;
    const int full_blocks = (score_stride + 63) / 64;
    // Exact 180-block persistent grid (3 blocks/CU x 60 CUs, 16,640 B LDS -> 12 waves/CU, 0 tail imbalance!)
    const int grid_x = (full_blocks < 180) ? full_blocks : 180;
    for (int sub = 0; sub < B_sc; sub += 8) {
        const int cur_b = (B_sc - sub < 8) ? (B_sc - sub) : 8;
        hipLaunchKernelGGL(
            k_dsa_indexer_score_topk8_dpp4,
            dim3(grid_x), dim3(256), 0, stream,
            d_q_sel_bf16, d_w_sel_f32, d_ik_cache_bf16,
            d_scores_batch_f32 + size_t(sub) * size_t(score_stride),
            pos0, t0 + sub, cur_b, score_stride);
    }
}

// Wave-12 Turn-17 Cut D12.1: M-FUSED Double-Buffered Split-K HCA Decode.
// comp_kv_bf16 ([num_comp, 512]) carries NO head index and NO M index: it is shared by all
// 16 local heads and all M = 1..8 DSpark verify tokens. The legacy grid dim3(n_heads/4, M,
// n_splits) therefore re-read the whole table 4*M times (up to 32x) with an unpipelined
// __syncthreads() every 4 keys. This kernel fuses M_TILE tokens per block so ONE shared load
// feeds M_TILE tokens, and prefetches the next 4-key tile into the alternate LDS buffer so the
// Wave-12 Turn-17 Cut D12.1: M-fused double-buffered HCA split-K pass 1.
// Fuses M_TILE (1 or 4) verify tokens into each CTA with 8 KiB ping-pong LDS s_k[2][4][64],
// cooperatively loading 4 keys (b0 + warp) per iteration while computing the previous 4-key tile.
// Single __syncthreads() per 4-key tile, 4x fewer comp_kv_bf16 HBM2 reads at M=4..8, ScratchSize=0.
template <int M_TILE, int MIN_BLOCKS>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void k_hca_decode_splitk_p1_mfused(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ comp_kv_bf16,
    const uint16_t* __restrict__ swa_ring_bf16,
    float* __restrict__ part_o_f32,
    float* __restrict__ part_ml_f32,
    int n_heads, int32_t pos0, int M, int num_comp, int n_comp_splits)
{
    __shared__ uint4 s_k[2][4][64]; // 8 KiB ping-pong LDS
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6; // 0..3 -> one head per warp & cooperative key loader
    const int lane = tid & 63;
    const int h    = int(blockIdx.x) * 4 + warp;
    const int m0   = int(blockIdx.y) * M_TILE;
    const int s    = int(blockIdx.z);
    const int m_cnt = (M - m0 < M_TILE) ? (M - m0) : M_TILE;

    float m_max[M_TILE], l_sum[M_TILE];
    float o0[M_TILE], o1[M_TILE], o2[M_TILE], o3[M_TILE];
    float o4[M_TILE], o5[M_TILE], o6[M_TILE], o7[M_TILE];
    #pragma unroll
    for (int mi = 0; mi < M_TILE; ++mi) {
        m_max[mi] = -1e30f; l_sum[mi] = 0.0f;
        o0[mi] = o1[mi] = o2[mi] = o3[mi] = 0.0f;
        o4[mi] = o5[mi] = o6[mi] = o7[mi] = 0.0f;
    }

    int buf = 0;
    if (s < n_comp_splits) {
        const int chunk = (num_comp + n_comp_splits - 1) / n_comp_splits;
        const int b_beg = s * chunk;
        const int b_end = (b_beg + chunk < num_comp) ? (b_beg + chunk) : num_comp;
        if (b_beg + warp < b_end) {
            s_k[0][warp][lane] = reinterpret_cast<const uint4*>(comp_kv_bf16 + size_t(b_beg + warp) * 512u)[lane];
        }
        __syncthreads();
        for (int b0 = b_beg; b0 < b_end; b0 += 4) {
            const int nxt = b0 + 4;
            if (nxt + warp < b_end) {
                s_k[buf ^ 1][warp][lane] = reinterpret_cast<const uint4*>(comp_kv_bf16 + size_t(nxt + warp) * 512u)[lane];
            }
            const int cnt = (b_end - b0 < 4) ? (b_end - b0) : 4;
            #pragma unroll
            for (int mi = 0; mi < M_TILE; ++mi) {
                if (mi < m_cnt) {
                    const int m = m0 + mi;
                    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane];
                    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
                    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
                    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
                    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
                    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
                    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
                    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
                    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
                    float sc[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (j < cnt) {
                            const uint4 kv4 = s_k[buf][j][lane];
                            const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
                            const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
                            const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
                            const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
                            const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
                            const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
                            const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
                            const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
                            const float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                                             fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
                            sc[j] = d_warp_sum64(dp) * 0.04419417382415922f;
                        } else {
                            sc[j] = -1e30f;
                        }
                    }
                    const float m_blk = fmaxf(m_max[mi], fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
                    const float alpha = __expf(m_max[mi] - m_blk);
                    o0[mi] *= alpha; o1[mi] *= alpha; o2[mi] *= alpha; o3[mi] *= alpha;
                    o4[mi] *= alpha; o5[mi] *= alpha; o6[mi] *= alpha; o7[mi] *= alpha;
                    l_sum[mi] *= alpha;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (j < cnt) {
                            const uint4 kv4 = s_k[buf][j][lane];
                            const float p = __expf(sc[j] - m_blk);
                            l_sum[mi] += p;
                            o0[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu)), o0[mi]);
                            o1[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.x >> 16)),     o1[mi]);
                            o2[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu)), o2[mi]);
                            o3[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.y >> 16)),     o3[mi]);
                            o4[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu)), o4[mi]);
                            o5[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.z >> 16)),     o5[mi]);
                            o6[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu)), o6[mi]);
                            o7[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.w >> 16)),     o7[mi]);
                        }
                    }
                    m_max[mi] = m_blk;
                }
            }
            buf ^= 1;
            __syncthreads();
        }
    } else {
        const int swa_start_all = (pos0 + m0 >= 127) ? (pos0 + m0 - 127) : 0;
        const int p_end_all = pos0 + m0 + m_cnt - 1;
        if (swa_start_all + warp <= p_end_all) {
            s_k[0][warp][lane] = reinterpret_cast<const uint4*>(swa_ring_bf16 + size_t((swa_start_all + warp) & 255) * 512u)[lane];
        }
        __syncthreads();
        for (int p0 = swa_start_all; p0 <= p_end_all; p0 += 4) {
            const int nxt = p0 + 4;
            if (nxt + warp <= p_end_all) {
                s_k[buf ^ 1][warp][lane] = reinterpret_cast<const uint4*>(swa_ring_bf16 + size_t((nxt + warp) & 255) * 512u)[lane];
            }
            const int cnt = (p_end_all - p0 + 1 < 4) ? (p_end_all - p0 + 1) : 4;
            #pragma unroll
            for (int mi = 0; mi < M_TILE; ++mi) {
                if (mi < m_cnt) {
                    const int m = m0 + mi;
                    const int32_t q_pos = pos0 + m;
                    const int swa_lo = (q_pos >= 127) ? (q_pos - 127) : 0;
                    if (p0 + cnt - 1 >= swa_lo && p0 <= q_pos) {
                        const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[lane];
                        const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
                        const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
                        const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
                        const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
                        const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
                        const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
                        const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
                        const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
                        float sc[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int key_pos = p0 + j;
                            const bool valid = (j < cnt) && (key_pos >= swa_lo) && (key_pos <= q_pos);
                            if (valid) {
                                const uint4 kv4 = s_k[buf][j][lane];
                                const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
                                const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
                                const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
                                const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
                                const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
                                const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
                                const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
                                const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
                                const float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                                                 fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
                                sc[j] = d_warp_sum64(dp) * 0.04419417382415922f;
                            } else {
                                sc[j] = -1e30f;
                            }
                        }
                        const float m_blk = fmaxf(m_max[mi], fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
                        const float alpha = __expf(m_max[mi] - m_blk);
                        o0[mi] *= alpha; o1[mi] *= alpha; o2[mi] *= alpha; o3[mi] *= alpha;
                        o4[mi] *= alpha; o5[mi] *= alpha; o6[mi] *= alpha; o7[mi] *= alpha;
                        l_sum[mi] *= alpha;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int key_pos = p0 + j;
                            if (j < cnt && key_pos >= swa_lo && key_pos <= q_pos) {
                                const uint4 kv4 = s_k[buf][j][lane];
                                const float p = __expf(sc[j] - m_blk);
                                l_sum[mi] += p;
                                o0[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu)), o0[mi]);
                                o1[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.x >> 16)),     o1[mi]);
                                o2[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu)), o2[mi]);
                                o3[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.y >> 16)),     o3[mi]);
                                o4[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu)), o4[mi]);
                                o5[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.z >> 16)),     o5[mi]);
                                o6[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu)), o6[mi]);
                                o7[mi] = fmaf(p, d_bf16_to_f32(uint16_t(kv4.w >> 16)),     o7[mi]);
                            }
                        }
                        m_max[mi] = m_blk;
                    }
                }
            }
            buf ^= 1;
            __syncthreads();
        }
    }

    #pragma unroll
    for (int mi = 0; mi < M_TILE; ++mi) {
        if (mi < m_cnt) {
            const int m = m0 + mi;
            const size_t slot = (size_t(s) * size_t(M) + size_t(m)) * size_t(n_heads) + size_t(h);
            float4* __restrict__ po = reinterpret_cast<float4*>(part_o_f32 + slot * 512u);
            po[lane * 2]     = make_float4(o0[mi], o1[mi], o2[mi], o3[mi]);
            po[lane * 2 + 1] = make_float4(o4[mi], o5[mi], o6[mi], o7[mi]);
            if (lane == 0) {
                part_ml_f32[slot * 2u]      = m_max[mi];
                part_ml_f32[slot * 2u + 1u] = l_sum[mi];
            }
        }
    }
}

void launch_hca_decode_splitk_mfused(
    const uint16_t* d_q_bf16, const uint16_t* d_comp_kv_bf16, const uint16_t* d_swa_ring_bf16,
    const float* d_attn_sink, float* d_part_o_f32, float* d_part_ml_f32, uint16_t* d_out_bf16,
    int n_heads, int32_t pos0, int M, int num_comp, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    if (num_comp < 128 || (n_heads & 3) != 0) {
        launch_hca_attn_c128_swa_sink_batch(d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16, d_attn_sink,
                                            d_out_bf16, n_heads, pos0, M, num_comp, stream);
        return;
    }
    int n_comp_splits = 4;
    if (num_comp >= 2048)      n_comp_splits = 32;
    else if (num_comp >= 1024) n_comp_splits = 16;
    else if (num_comp >= 512)  n_comp_splits = 8;
    const int n_splits = n_comp_splits + 1;
    if (M == 1) {
        hipLaunchKernelGGL((k_hca_decode_splitk_p1_mfused<1, 4>), dim3(n_heads / 4, 1, n_splits), dim3(256), 0, stream,
                           d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16,
                           d_part_o_f32, d_part_ml_f32, n_heads, pos0, M, num_comp, n_comp_splits);
    } else {
        const int m_tiles = (M + 3) / 4;
        hipLaunchKernelGGL((k_hca_decode_splitk_p1_mfused<4, 2>), dim3(n_heads / 4, m_tiles, n_splits), dim3(256), 0, stream,
                           d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16,
                           d_part_o_f32, d_part_ml_f32, n_heads, pos0, M, num_comp, n_comp_splits);
    }
    hipLaunchKernelGGL(k_hca_decode_splitk_merge, dim3(n_heads, M), dim3(64), 0, stream,
                       d_part_o_f32, d_part_ml_f32, d_attn_sink, d_out_bf16, n_heads, M, n_splits);
}

void launch_hca_attn_c128_swa_sink_splitk(
    const uint16_t* d_q_bf16, const uint16_t* d_comp_kv_bf16, const uint16_t* d_swa_ring_bf16,
    const float* d_attn_sink, float* d_part_o_f32, float* d_part_ml_f32, uint16_t* d_out_bf16,
    int n_heads, int32_t pos0, int M, int num_comp, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    // Fall back to unsplit kernel when num_comp < 128 or n_heads not divisible by 4 (depends ONLY on num_comp & n_heads -> M=1==M=6 exact!)
    if (num_comp < 128 || (n_heads & 3) != 0) {
        launch_hca_attn_c128_swa_sink_batch(d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16, d_attn_sink,
                                            d_out_bf16, n_heads, pos0, M, num_comp, stream);
        return;
    }
    // Split count depends strictly on num_comp (NOT on M) so M=1 and M=2..8 execute bit-identical FP32 reductions!
    int n_comp_splits = 4;
    if (num_comp >= 2048)      n_comp_splits = 32;
    else if (num_comp >= 1024) n_comp_splits = 16;
    else if (num_comp >= 512)  n_comp_splits = 8;

    const int n_splits = n_comp_splits + 1; // last split = SWA ring in parallel!
    hipLaunchKernelGGL(k_hca_decode_splitk_p1, dim3(n_heads / 4, M, n_splits), dim3(256), 0, stream,
                       d_q_bf16, d_comp_kv_bf16, d_swa_ring_bf16,
                       d_part_o_f32, d_part_ml_f32, n_heads, pos0, M, num_comp, n_comp_splits);
    hipLaunchKernelGGL(k_hca_decode_splitk_merge, dim3(n_heads, M), dim3(64), 0, stream,
                       d_part_o_f32, d_part_ml_f32, d_attn_sink, d_out_bf16, n_heads, M, n_splits);
}

void launch_dsa_select_topk2048_wave6(
    const float* d_scores_batch_f32, uint32_t* d_scratch_u32,
    int32_t* d_sel_idx_MxS, int32_t* d_sel_cnts,
    int32_t pos0, int t0, int B_sc, int score_stride, int stride_sel, hipStream_t stream) {
    if (B_sc <= 0) return;
    // Sub-batch <= 64 rows (~40.0 MiB < 75.5 MiB g_own.d_y_f32) -> 8x fewer launches on 262K prefill!
    for (int sub = 0; sub < B_sc; sub += 64) {
        const int cur_b = (B_sc - sub < 64) ? (B_sc - sub) : 64;
        const float* sc_sub = d_scores_batch_f32 + size_t(sub) * size_t(score_stride);
        uint32_t* g_hist1   = d_scratch_u32;
        uint32_t* g_hist2   = g_hist1   + size_t(cur_b) * kWave6RadixBlocks * 2048u;
        uint32_t* g_hist3   = g_hist2   + size_t(cur_b) * kWave6RadixBlocks * 2048u;
        uint32_t* g_meta    = g_hist3   + size_t(cur_b) * kWave6RadixBlocks * 1024u;
        uint32_t* g_blk_cnt = g_meta    + size_t(cur_b) * 8u;

        const dim3 grid(kWave6RadixBlocks, uint32_t(cur_b));
        const dim3 block(256);
        hipLaunchKernelGGL(k_dsa_topk2048_pass1_hist, grid, block, 0, stream,
                           sc_sub, g_hist1, d_sel_idx_MxS, d_sel_cnts,
                           pos0, t0 + sub, score_stride, stride_sel);
        hipLaunchKernelGGL(k_dsa_topk2048_pass2_hist, grid, block, 0, stream,
                           sc_sub, g_hist1, g_hist2, g_meta,
                           pos0, t0 + sub, score_stride);
        hipLaunchKernelGGL(k_dsa_topk2048_pass3_hist, grid, block, 0, stream,
                           sc_sub, g_hist2, g_hist3, g_meta,
                           pos0, t0 + sub, score_stride);
        hipLaunchKernelGGL(k_dsa_topk2048_pass4a_counts, grid, block, 0, stream,
                           sc_sub, g_hist3, g_meta, g_blk_cnt,
                           pos0, t0 + sub, score_stride);
        hipLaunchKernelGGL(k_dsa_topk2048_pass4b_scatter, grid, block, 0, stream,
                           sc_sub, g_meta, g_blk_cnt, d_sel_idx_MxS, d_sel_cnts,
                           pos0, t0 + sub, score_stride, stride_sel);
    }
}

void launch_mhc_mean4_bf16(
    const uint16_t* d_res_4xH, uint16_t* d_out_H, int M, int H, hipStream_t stream) {
    if (M <= 0 || H <= 0) return;
    hipLaunchKernelGGL(k_mhc_mean4_bf16, dim3((H + 255) / 256, M), dim3(256), 0, stream,
                       d_res_4xH, d_out_H, H);
}

void launch_dspark_swa_ring_write(
    const uint16_t* d_kv_norm_bf16, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, uint16_t* d_swa_ring_256, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dspark_swa_ring_write, dim3(M), dim3(256), 0, stream,
                       d_kv_norm_bf16, d_pos_ids, rope.d_cos_hi, rope.d_sin_hi,
                       rope.d_cos_lo, rope.d_sin_lo, d_swa_ring_256);
}

void launch_dspark_swa_noncausal_block_attn(
    const uint16_t* d_q_bf16, const uint16_t* d_swa_ring_256,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int32_t query_pos0, int K, int sliding_window, hipStream_t stream) {
    if (K <= 0) return;
    hipLaunchKernelGGL(k_dspark_swa_noncausal_block_attn, dim3(64, K), dim3(64), 0, stream,
                       d_q_bf16, d_swa_ring_256, d_attn_sink, d_out_bf16, query_pos0, K, sliding_window);
}

void launch_dspark_add_bias_multimax_f32(
    const float* d_base_logits_f32, const float* d_bias_f32,
    const float* d_ranges4, const float* d_ts4,
    float* d_out_logits_f32, int V, hipStream_t stream) {
    if (V <= 0) return;
    hipLaunchKernelGGL(k_dspark_add_bias_multimax_f32, dim3((V + 255) / 256), dim3(256), 0, stream,
                       d_base_logits_f32, d_bias_f32, d_ranges4, d_ts4, d_out_logits_f32, V);
}

void launch_f32_add_inplace(float* d_dst_f32, const float* d_src_f32, int total, hipStream_t stream) {
    if (total <= 0) return;
    hipLaunchKernelGGL(k_f32_add_inplace, dim3((total + 255) / 256), dim3(256), 0, stream, d_dst_f32, d_src_f32, total);
}

void launch_f32_mul_sigmoid_to_bf16(const float* d_z_f32, const uint16_t* d_gate_bf16, uint16_t* d_out_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    hipLaunchKernelGGL(k_f32_mul_sigmoid_to_bf16, dim3((total + 255) / 256), dim3(256), 0, stream, d_z_f32, d_gate_bf16, d_out_bf16, total);
}

void launch_dsa_copy_qrope_bf16(const uint16_t* d_q_bf16, uint16_t* d_q_abs_bf16, int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_copy_qrope_bf16, dim3((n_heads + 3) / 4, M), dim3(256), 0, stream, d_q_bf16, d_q_abs_bf16, n_heads);
}

namespace {

__device__ __forceinline__ float d_f16_to_f32_core(uint16_t h) {
    __half hv = *reinterpret_cast<const __half*>(&h);
    return __half2float(hv);
}

__device__ __forceinline__ uint16_t d_f32_to_f16_core(float x) {
    if (x > 65504.0f) x = 65504.0f;
    else if (x < -65504.0f) x = -65504.0f;
    __half hv = __float2half(x);
    return *reinterpret_cast<const uint16_t*>(&hv);
}

// 16x-HBM2-reuse Rank-Major VHA + invRoPE: reads [tp_size, M, h_loc, 512] ONCE per 16 output heads!
__global__ __launch_bounds__(256, 4)
void k_hca_vha_invrope_shard(
    const uint16_t* __restrict__ o_in_rm,
    const float* __restrict__ vha_m,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ o_out,
    int M, int tp_size, int head_begin, int n_local_heads)
{
    __shared__ float s_m[16][64];
    const int hg  = int(blockIdx.x); // 0 .. n_local_heads/16 - 1
    const int m   = int(blockIdx.y);
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims/thread across 512
    const int h0  = head_begin + hg * 16;

    // Load 16x64 VHA rows into LDS (4 KiB, 4 floats/thread)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = tid + i * 256;
        const int r = idx >> 6;
        const int c = idx & 63;
        s_m[r][c] = vha_m[size_t(h0 + r) * 64u + size_t(c)];
    }
    __syncthreads();

    float y0[16], y1[16];
    #pragma unroll
    for (int hl = 0; hl < 16; ++hl) { y0[hl] = 0.0f; y1[hl] = 0.0f; }

    const int h_rank = 64 / tp_size;
    int g = 0;
    for (int rk = 0; rk < tp_size; ++rk) {
        const uint16_t* __restrict__ rbase = o_in_rm + (size_t(rk) * size_t(M) + size_t(m)) * size_t(h_rank) * 512u;
        for (int gl = 0; gl < h_rank; ++gl, ++g) {
            const uint32_t pair = reinterpret_cast<const uint32_t*>(rbase + size_t(gl) * 512u)[tid];
            const float x0 = d_bf16_to_f32(uint16_t(pair & 0xFFFFu));
            const float x1 = d_bf16_to_f32(uint16_t(pair >> 16));
            #pragma unroll
            for (int hl = 0; hl < 16; ++hl) {
                const float w = s_m[hl][g];
                y0[hl] = fmaf(w, x0, y0[hl]);
                y1[hl] = fmaf(w, x1, y1[hl]);
            }
        }
    }

    if (tid >= 224) {
        float c, s;
        d_rope2d_lookup(pos_ids[m], tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        #pragma unroll
        for (int hl = 0; hl < 16; ++hl) {
            const float r0 = fmaf(y0[hl], c,   y1[hl] * s);
            const float r1 = fmaf(y1[hl], c, -(y0[hl] * s));
            y0[hl] = r0;
            y1[hl] = r1;
        }
    }

    uint16_t* __restrict__ obase = o_out + (size_t(m) * size_t(n_local_heads) + size_t(hg * 16)) * 512u;
    #pragma unroll
    for (int hl = 0; hl < 16; ++hl) {
        const uint32_t out_p = uint32_t(d_f32_to_bf16(y0[hl])) | (uint32_t(d_f32_to_bf16(y1[hl])) << 16);
        reinterpret_cast<uint32_t*>(obase + size_t(hl) * 512u)[tid] = out_p;
    }
}

// 16x-HBM2-reuse Rank-Major DSA VHA + LRGA: reads [tp_size, M, h_loc, 256] ONCE per 16 output heads!
__global__ __launch_bounds__(256, 4)
void k_dsa_vha_lrga_shard(
    const uint16_t* __restrict__ o_in_rm,
    const float* __restrict__ vha_m,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* __restrict__ out_bf16,
    int M, int tp_size, int head_begin, int n_local_heads)
{
    __shared__ float s_m[16][64];
    const int hg  = int(blockIdx.x);
    const int m   = int(blockIdx.y);
    const int d   = int(threadIdx.x); // 0..255
    const int h0  = head_begin + hg * 16;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = d + i * 256;
        s_m[idx >> 6][idx & 63] = vha_m[size_t(h0 + (idx >> 6)) * 64u + size_t(idx & 63)];
    }
    __syncthreads();

    float y[16];
    #pragma unroll
    for (int hl = 0; hl < 16; ++hl) y[hl] = 0.0f;

    const int h_rank = 64 / tp_size;
    int g = 0;
    for (int rk = 0; rk < tp_size; ++rk) {
        const uint16_t* __restrict__ rbase = o_in_rm + (size_t(rk) * size_t(M) + size_t(m)) * size_t(h_rank) * 256u;
        for (int gl = 0; gl < h_rank; ++gl, ++g) {
            const float x = d_bf16_to_f32(rbase[size_t(gl) * 256u + size_t(d)]);
            #pragma unroll
            for (int hl = 0; hl < 16; ++hl) y[hl] = fmaf(s_m[hl][g], x, y[hl]);
        }
    }

    const size_t base = (size_t(m) * size_t(n_local_heads) + size_t(hg * 16)) * 256u;
    #pragma unroll
    for (int hl = 0; hl < 16; ++hl) {
        const size_t idx = base + size_t(hl) * 256u + size_t(d);
        out_bf16[idx] = d_f32_to_bf16(y[hl] * d_sigmoid(d_bf16_to_f32(gate_bf16[idx])));
    }
}

__global__ __launch_bounds__(256, 8)
void k_hca_gather_swa_kv_f16(
    const uint16_t* __restrict__ swa_ring_bf16,
    const uint16_t* __restrict__ kv_chunk_rot,
    uint16_t* __restrict__ k_swa_f16,
    int32_t pos0, int M, int32_t swa_lo)
{
    const int j = int(blockIdx.x);
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims/thread across 512
    const int32_t p = swa_lo + j;
    uint16_t* __restrict__ dst = k_swa_f16 + size_t(j) * 512u;
    if (p < pos0 + M) {
        const uint16_t* __restrict__ src = (p < pos0)
            ? (swa_ring_bf16 + size_t(p & 255) * 512u)
            : (kv_chunk_rot  + size_t(p - pos0) * 512u);
        dst[tid * 2]     = d_f32_to_f16_core(d_bf16_to_f32(src[tid * 2]));
        dst[tid * 2 + 1] = d_f32_to_f16_core(d_bf16_to_f32(src[tid * 2 + 1]));
    } else {
        dst[tid * 2]     = 0u;
        dst[tid * 2 + 1] = 0u;
    }
}

__global__ __launch_bounds__(256, 4)
void k_hca_swa_softmax_f32(
    const float* __restrict__ scores_f32,
    float* __restrict__ m_swa_f32,
    float* __restrict__ l_swa_f32,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int32_t pos0, int32_t swa_lo, int swa_pad)
{
    __shared__ float s_w[4];
    const int row = int(blockIdx.x);
    const int m   = row / n_heads;
    const int32_t pos = pos0 + m;
    const int32_t p_start = (pos >= 127) ? (pos - 127) : 0;
    const int j_start = int(p_start - swa_lo);
    const int j_end   = int(pos - swa_lo); // inclusive

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const float* __restrict__ sc = scores_f32 + size_t(row) * size_t(swa_pad);
    uint16_t* __restrict__ pr    = probs_f16  + size_t(row) * size_t(swa_pad);
    constexpr float kScale = 0.04419417382415922f;

    float lmax = -1e30f;
    for (int j = j_start + tid; j <= j_end; j += 256) {
        const float v = sc[j] * kScale;
        if (v > lmax) lmax = v;
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    float lsum = 0.0f;
    for (int j = tid; j < swa_pad; j += 256) {
        if (j >= j_start && j <= j_end) {
            const float e = __expf(sc[j] * kScale - mx);
            lsum += e;
            pr[j] = d_f32_to_f16_core(e);
        } else {
            pr[j] = 0u;
        }
    }
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    if (tid == 0) {
        m_swa_f32[row] = mx;
        l_swa_f32[row] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    }
}

__global__ __launch_bounds__(256, 8)
void k_hca_merge_comp_swa_sink_bf16(
    const float* __restrict__ comp_o_f32,
    const float* __restrict__ comp_m_f32,
    const float* __restrict__ comp_l_f32,
    const float* __restrict__ swa_o_f32,
    const float* __restrict__ swa_m_f32,
    const float* __restrict__ swa_l_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads, int has_comp)
{
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims/thread across 512
    const int h   = row % n_heads;
    const float sink = attn_sink[h];
    const float ms = swa_m_f32[row];
    const float ls = swa_l_f32[row];
    const float* __restrict__ so = swa_o_f32 + size_t(row) * 512u;
    uint16_t* __restrict__ dst   = out_bf16  + size_t(row) * 512u;

    if (has_comp) {
        const float mc = comp_m_f32[row];
        const float lc = comp_l_f32[row];
        const float m_tot = fmaxf(fmaxf(mc, ms), sink);
        const float ac = (lc > 0.0f) ? __expf(mc - m_tot) : 0.0f;
        const float as = __expf(ms - m_tot);
        const float inv_d = 1.0f / (lc * ac + ls * as + __expf(sink - m_tot));
        const float* __restrict__ co = comp_o_f32 + size_t(row) * 512u;
        const float y0 = fmaf(co[tid * 2],     ac, so[tid * 2]     * as) * inv_d;
        const float y1 = fmaf(co[tid * 2 + 1], ac, so[tid * 2 + 1] * as) * inv_d;
        dst[tid * 2]     = d_f32_to_bf16(y0);
        dst[tid * 2 + 1] = d_f32_to_bf16(y1);
    } else {
        const float m_tot = fmaxf(ms, sink);
        const float as = __expf(ms - m_tot);
        const float inv_d = as / (ls * as + __expf(sink - m_tot));
        dst[tid * 2]     = d_f32_to_bf16(so[tid * 2]     * inv_d);
        dst[tid * 2 + 1] = d_f32_to_bf16(so[tid * 2 + 1] * inv_d);
    }
}

__global__ __launch_bounds__(256, 8)
void k_flashindexer_relu_wsum_f32(
    const float* __restrict__ dots_f32,   // [B_sc * 8, k_cnt] (row-major: row = bi*8 + h, col = k)
    const float* __restrict__ w_sel_f32,  // [M, 8]
    float* __restrict__ scores_batch_f32, // [B_sc, score_stride]
    int32_t pos0, int t0, int k0, int k_cnt, int score_stride)
{
    const int k  = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int bi = int(blockIdx.y);
    if (k >= k_cnt) return;
    const int t = t0 + bi;
    const int n_hist = (pos0 + t) - 127;
    const int tok = k0 + k;
    if (n_hist <= 2048 || tok >= n_hist) return;

    constexpr float kWScale = 0.011048543456039806f;
    const float* __restrict__ wt = w_sel_f32 + size_t(t) * 8u;
    const size_t base = size_t(bi) * 8u * size_t(k_cnt) + size_t(k);
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float dp = dots_f32[base + size_t(i) * size_t(k_cnt)];
        if (dp > 0.0f) total = fmaf(wt[i] * kWScale, dp, total);
    }
    scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = total;
}

__global__ __launch_bounds__(256, 4)
void k_hca_comp_softmax_f32(
    const float* __restrict__ scores_f32, // [M * n_heads, blk_cnt]
    float* __restrict__ m_blk_f32,        // [M * n_heads]
    float* __restrict__ l_blk_f32,        // [M * n_heads]
    uint16_t* __restrict__ probs_f16,     // [M * n_heads, blk_cnt]
    int n_heads, int32_t pos0, int blk0, int blk_cnt)
{
    __shared__ float s_w[4];
    const int row = int(blockIdx.x);
    const int m   = row / n_heads;
    const int num_comp = (pos0 + m) / 128;
    const int valid = (num_comp > blk0) ? ((num_comp - blk0 < blk_cnt) ? (num_comp - blk0) : blk_cnt) : 0;

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const float* __restrict__ sc = scores_f32 + size_t(row) * size_t(blk_cnt);
    uint16_t* __restrict__ pr    = probs_f16  + size_t(row) * size_t(blk_cnt);
    constexpr float kScale = 0.04419417382415922f;

    if (valid <= 0) {
        for (int j = tid; j < blk_cnt; j += 256) pr[j] = 0u;
        if (tid == 0) { m_blk_f32[row] = -1e30f; l_blk_f32[row] = 0.0f; }
        return;
    }

    float lmax = -1e30f;
    for (int j = tid; j < valid; j += 256) {
        const float v = sc[j] * kScale;
        if (v > lmax) lmax = v;
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    float lsum = 0.0f;
    for (int j = tid; j < blk_cnt; j += 256) {
        if (j < valid) {
            const float e = __expf(sc[j] * kScale - mx);
            lsum += e;
            pr[j] = d_f32_to_f16_core(e);
        } else {
            pr[j] = 0u;
        }
    }
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    if (tid == 0) {
        m_blk_f32[row] = mx;
        l_blk_f32[row] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    }
}

__global__ __launch_bounds__(256, 8)
void k_hca_comp_merge_tile_f32(
    float* __restrict__ o_tot_f32,
    float* __restrict__ m_tot_f32,
    float* __restrict__ l_tot_f32,
    const float* __restrict__ o_blk_f32,
    const float* __restrict__ m_blk_f32,
    const float* __restrict__ l_blk_f32,
    int first_tile)
{
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims/thread across 512
    const float mb = m_blk_f32[row];
    const float lb = l_blk_f32[row];
    float* __restrict__ ot = o_tot_f32 + size_t(row) * 512u;
    const float* __restrict__ ob = o_blk_f32 + size_t(row) * 512u;

    if (first_tile) {
        ot[tid * 2]     = ob[tid * 2];
        ot[tid * 2 + 1] = ob[tid * 2 + 1];
        if (tid == 0) { m_tot_f32[row] = mb; l_tot_f32[row] = lb; }
    } else {
        const float mt = m_tot_f32[row];
        const float lt = l_tot_f32[row];
        if (lb <= 0.0f) return;
        const float mn = fmaxf(mt, mb);
        const float a0 = (lt > 0.0f) ? __expf(mt - mn) : 0.0f;
        const float a1 = __expf(mb - mn);
        ot[tid * 2]     = fmaf(ot[tid * 2],     a0, a1 * ob[tid * 2]);
        ot[tid * 2 + 1] = fmaf(ot[tid * 2 + 1], a0, a1 * ob[tid * 2 + 1]);
        if (tid == 0) {
            m_tot_f32[row] = mn;
            l_tot_f32[row] = fmaf(lt, a0, a1 * lb);
        }
    }
}

// 8-Head-Shared LDS-Tiled Exact BF16->FP32 HCA SWA + Online Comp Merge + Sink:
// 8x less K_swa global reads AND bit-for-bit identical accumulation to M<=8 k_hca_attn_c128_swa_sink_batch!
__global__ __launch_bounds__(512, 2)
void k_hca_prefill_swa_merge_sink(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ swa_ring_bf16,
    const uint16_t* __restrict__ kv_chunk_rot,
    const float* __restrict__ comp_o_f32,
    const float* __restrict__ comp_m_f32,
    const float* __restrict__ comp_l_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads, int32_t pos0)
{
    __shared__ uint4 s_k[8][64];
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6; // 0..7
    const int lane = tid & 63; // 0..63 -> 8 dims/lane
    const int h    = int(blockIdx.x) * 8 + warp;
    const int m    = int(blockIdx.y);
    const int32_t pos = pos0 + m;
    const size_t row = size_t(m) * size_t(n_heads) + size_t(h);

    const uint4 qv = reinterpret_cast<const uint4*>(q_bf16 + row * 512u)[lane];
    const float q0 = d_bf16_to_f32(uint16_t(qv.x & 0xFFFFu));
    const float q1 = d_bf16_to_f32(uint16_t(qv.x >> 16));
    const float q2 = d_bf16_to_f32(uint16_t(qv.y & 0xFFFFu));
    const float q3 = d_bf16_to_f32(uint16_t(qv.y >> 16));
    const float q4 = d_bf16_to_f32(uint16_t(qv.z & 0xFFFFu));
    const float q5 = d_bf16_to_f32(uint16_t(qv.z >> 16));
    const float q6 = d_bf16_to_f32(uint16_t(qv.w & 0xFFFFu));
    const float q7 = d_bf16_to_f32(uint16_t(qv.w >> 16));
    constexpr float kScale = 0.04419417382415922f;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;
    if (comp_o_f32 != nullptr) {
        const float lc = comp_l_f32[row];
        if (lc > 0.0f) {
            m_max = comp_m_f32[row];
            l_sum = lc;
            const float4* __restrict__ co4 = reinterpret_cast<const float4*>(comp_o_f32 + row * 512u);
            const float4 c0 = co4[lane * 2];
            const float4 c1 = co4[lane * 2 + 1];
            o0 = c0.x; o1 = c0.y; o2 = c0.z; o3 = c0.w;
            o4 = c1.x; o5 = c1.y; o6 = c1.z; o7 = c1.w;
        }
    }

    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    for (int p0 = swa_start; p0 <= pos; p0 += 8) {
        const int p_load = p0 + warp;
        if (p_load <= pos) {
            const uint16_t* __restrict__ kv = (p_load < pos0)
                ? (swa_ring_bf16 + size_t(p_load & 255) * 512u)
                : (kv_chunk_rot  + size_t(p_load - pos0) * 512u);
            s_k[warp][lane] = reinterpret_cast<const uint4*>(kv)[lane];
        }
        __syncthreads();

        const int cnt = (pos - p0 + 1 < 8) ? (pos - p0 + 1) : 8;
        for (int j = 0; j < cnt; ++j) {
            const uint4 kv4 = s_k[j][lane];
            const float k0 = d_bf16_to_f32(uint16_t(kv4.x & 0xFFFFu));
            const float k1 = d_bf16_to_f32(uint16_t(kv4.x >> 16));
            const float k2 = d_bf16_to_f32(uint16_t(kv4.y & 0xFFFFu));
            const float k3 = d_bf16_to_f32(uint16_t(kv4.y >> 16));
            const float k4 = d_bf16_to_f32(uint16_t(kv4.z & 0xFFFFu));
            const float k5 = d_bf16_to_f32(uint16_t(kv4.z >> 16));
            const float k6 = d_bf16_to_f32(uint16_t(kv4.w & 0xFFFFu));
            const float k7 = d_bf16_to_f32(uint16_t(kv4.w >> 16));
            float dp = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3,
                       fmaf(q4, k4, fmaf(q5, k5, fmaf(q6, k6, q7 * k7)))))));
            const float score = d_warp_sum64(dp) * kScale;
            const float m_new = fmaxf(m_max, score);
            const float alpha = __expf(m_max - m_new);
            const float p = __expf(score - m_new);
            o0 = fmaf(o0, alpha, p * k0); o1 = fmaf(o1, alpha, p * k1);
            o2 = fmaf(o2, alpha, p * k2); o3 = fmaf(o3, alpha, p * k3);
            o4 = fmaf(o4, alpha, p * k4); o5 = fmaf(o5, alpha, p * k5);
            o6 = fmaf(o6, alpha, p * k6); o7 = fmaf(o7, alpha, p * k7);
            l_sum = fmaf(l_sum, alpha, p);
            m_max = m_new;
        }
        __syncthreads();
    }

    const float sink = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(out_bf16 + row * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

__global__ __launch_bounds__(256, 8)
void k_dsa_gather_kv_f16(
    const uint16_t* __restrict__ mla_kv_bf16,
    const int32_t* __restrict__ sel_indices_BxS,
    const int32_t* __restrict__ sel_counts,
    uint16_t* __restrict__ k_sel_f16,
    int stride_sel)
{
    const int s = int(blockIdx.x);
    const int b = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int cnt = sel_counts[b];
    uint16_t* __restrict__ dst = k_sel_f16 + (size_t(b) * size_t(stride_sel) + size_t(s)) * 576u;
    if (s < cnt) {
        const int tok = sel_indices_BxS[size_t(b) * size_t(stride_sel) + size_t(s)];
        const uint16_t* __restrict__ src = mla_kv_bf16 + size_t(tok) * 576u;
        for (int d = tid; d < 576; d += 256) dst[d] = d_f32_to_f16_core(d_bf16_to_f32(src[d]));
    } else {
        for (int d = tid; d < 576; d += 256) dst[d] = 0u;
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_scores_softmax_sink_f16(
    const float* __restrict__ scores_BxHxL,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ sel_counts,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int L, float scale)
{
    __shared__ float s_w[4];
    const int h = int(blockIdx.x);
    const int b = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int valid = sel_counts[b];
    const size_t off = (size_t(b) * size_t(n_heads) + size_t(h)) * size_t(L);
    const float* __restrict__ sc = scores_BxHxL + off;
    uint16_t* __restrict__ pr    = probs_f16 + off;

    float lmax = attn_sink[h];
    for (int k = tid; k < valid; k += 256) {
        const float v = sc[k] * scale;
        if (v > lmax) lmax = v;
    }
    #pragma unroll
    for (int sh = 32; sh >= 1; sh >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, sh, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float m_tot = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    float lsum = 0.0f;
    for (int k = tid; k < valid; k += 256) lsum += __expf(sc[k] * scale - m_tot);
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    const float inv_d = 1.0f / (s_w[0] + s_w[1] + s_w[2] + s_w[3] + __expf(attn_sink[h] - m_tot));

    for (int k = tid; k < L; k += 256) {
        pr[k] = (k < valid) ? d_f32_to_f16_core(__expf(sc[k] * scale - m_tot) * inv_d) : 0u;
    }
}

typedef _Float16 core_f16x2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ core_f16x2_t d_u32_to_core_f16x2(uint32_t u) {
    union { uint32_t u32; core_f16x2_t v; } c;
    c.u32 = u;
    return c.v;
}

__device__ __forceinline__ float d_core_dot8_f16_u4_fdot2(uint4 xv, uint4 wv, float acc) {
    acc = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(xv.x), d_u32_to_core_f16x2(wv.x), acc, false);
    acc = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(xv.y), d_u32_to_core_f16x2(wv.y), acc, false);
    acc = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(xv.z), d_u32_to_core_f16x2(wv.z), acc, false);
    acc = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(xv.w), d_u32_to_core_f16x2(wv.w), acc, false);
    return acc;
}

__device__ __forceinline__ uint4 d_core_bf16_u4_to_f16_u4(uint4 b) {
    auto c2 = [](uint32_t u) -> uint32_t {
        return uint32_t(d_f32_to_f16_core(d_bf16_to_f32(uint16_t(u & 0xFFFFu))))
             | (uint32_t(d_f32_to_f16_core(d_bf16_to_f32(uint16_t(u >> 16)))) << 16);
    };
    uint4 o;
    o.x = c2(b.x); o.y = c2(b.y); o.z = c2(b.z); o.w = c2(b.w);
    return o;
}

__device__ __forceinline__ float d_core_reduce32_dpp(float v) {
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xf, 0xf, true));
    v += __shfl_xor(v, 4, 64);
    v += __shfl_xor(v, 8, 64);
    v += __shfl_xor(v, 16, 64);
    return v;
}

// Wave-8 Weight-Stationary 64-Block Coalesced DSA q_absorb_uk_f16 (reads W_UK ONCE across M=1..8!)
template <int M_STATIC>
__global__ __launch_bounds__(128, 4)
void k_dsa_q_absorb_uk_f16_ws(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ kv_b_f16,
    uint16_t* __restrict__ q_abs_bf16,
    int n_heads)
{
    __shared__ float s_qn[M_STATIC][192];
    const int blk = int(blockIdx.x); // 0..3 -> 128 cols/block = 512 cols total!
    const int h   = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int c   = blk * 128 + tid;

    for (int idx = tid; idx < M_STATIC * 192; idx += 128) {
        const int m = idx / 192;
        const int d = idx - m * 192;
        s_qn[m][d] = d_bf16_to_f32(q_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 256u + size_t(d)]);
    }
    if (blk == 0 && tid < 64) {
        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m) {
            q_abs_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 576u + 512u + size_t(tid)] =
                q_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 256u + 192u + size_t(tid)];
        }
    }
    __syncthreads();

    const uint16_t* __restrict__ wuk = kv_b_f16 + size_t(h) * 448u * 512u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int r = 0; r < 192; ++r) {
        const float wv = d_f16_to_f32_core(wuk[size_t(r) * 512u + size_t(c)]);
        a0 = fmaf(s_qn[0][r], wv, a0);
        if constexpr (M_STATIC > 1) a1 = fmaf(s_qn[1][r], wv, a1);
        if constexpr (M_STATIC > 2) a2 = fmaf(s_qn[2][r], wv, a2);
        if constexpr (M_STATIC > 3) a3 = fmaf(s_qn[3][r], wv, a3);
        if constexpr (M_STATIC > 4) a4 = fmaf(s_qn[4][r], wv, a4);
        if constexpr (M_STATIC > 5) a5 = fmaf(s_qn[5][r], wv, a5);
        if constexpr (M_STATIC > 6) a6 = fmaf(s_qn[6][r], wv, a6);
        if constexpr (M_STATIC > 7) a7 = fmaf(s_qn[7][r], wv, a7);
    }
    q_abs_bf16[(size_t(0) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a0);
    if constexpr (M_STATIC > 1) q_abs_bf16[(size_t(1) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a1);
    if constexpr (M_STATIC > 2) q_abs_bf16[(size_t(2) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a2);
    if constexpr (M_STATIC > 3) q_abs_bf16[(size_t(3) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a3);
    if constexpr (M_STATIC > 4) q_abs_bf16[(size_t(4) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a4);
    if constexpr (M_STATIC > 5) q_abs_bf16[(size_t(5) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a5);
    if constexpr (M_STATIC > 6) q_abs_bf16[(size_t(6) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a6);
    if constexpr (M_STATIC > 7) q_abs_bf16[(size_t(7) * size_t(n_heads) + size_t(h)) * 576u + size_t(c)] = d_f32_to_bf16(a7);
}

// Wave-8 Weight-Stationary Coalesced fdot2 DSA o_project_uv_f16 (8 rows/block, 2x uint4/lane, 0 loops!)
template <int M_STATIC>
__global__ __launch_bounds__(256, 4)
void k_dsa_o_project_uv_f16_ws(
    const uint16_t* __restrict__ o_lat_bf16,
    const uint16_t* __restrict__ kv_b_f16,
    uint16_t* __restrict__ o_v_bf16,
    int n_heads)
{
    __shared__ uint4 s_o[M_STATIC][64]; // [M_STATIC, 512 halfs] <= 8 KiB LDS!
    const int tid = int(threadIdx.x);
    const int row_loc = tid >> 5; // 0..7
    const int l32     = tid & 31; // 0..31
    const int d = int(blockIdx.x) * 8 + row_loc; // 0..255
    const int h = int(blockIdx.y);               // 0..n_heads-1

    for (int idx = tid; idx < M_STATIC * 64; idx += 256) {
        const int m = idx >> 6;
        const int k = idx & 63;
        const uint4 bv = reinterpret_cast<const uint4*>(o_lat_bf16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u)[k];
        s_o[m][k] = d_core_bf16_u4_to_f16_u4(bv);
    }
    __syncthreads();
    if (d >= 256) return;

    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(kv_b_f16 + (size_t(h) * 448u + 192u + size_t(d)) * 512u);
    const uint4 w0 = w4[l32];
    const uint4 w1 = w4[l32 + 32];

    auto do_row = [&](int m) {
        float acc = d_core_dot8_f16_u4_fdot2(s_o[m][l32], w0, 0.0f);
        acc = d_core_dot8_f16_u4_fdot2(s_o[m][l32 + 32], w1, acc);
        acc = d_core_reduce32_dpp(acc);
        if (l32 == 0) o_v_bf16[(size_t(m) * size_t(n_heads) + size_t(h)) * 256u + size_t(d)] = d_f32_to_bf16(acc);
    };
    do_row(0);
    if constexpr (M_STATIC > 1) do_row(1);
    if constexpr (M_STATIC > 2) do_row(2);
    if constexpr (M_STATIC > 3) do_row(3);
    if constexpr (M_STATIC > 4) do_row(4);
    if constexpr (M_STATIC > 5) do_row(5);
    if constexpr (M_STATIC > 6) do_row(6);
    if constexpr (M_STATIC > 7) do_row(7);
}

__device__ __forceinline__ float d_mhc_dot8(uint4 xv, float4 f0, float4 f1) {
    float d = fmaf(d_bf16_to_f32(uint16_t(xv.x & 0xFFFFu)), f0.x, 0.0f);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.x >> 16)),     f0.y, d);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.y & 0xFFFFu)), f0.z, d);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.y >> 16)),     f0.w, d);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.z & 0xFFFFu)), f1.x, d);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.z >> 16)),     f1.y, d);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.w & 0xFFFFu)), f1.z, d);
    d = fmaf(d_bf16_to_f32(uint16_t(xv.w >> 16)),     f1.w, d);
    return d;
}

// 60-CU direct BF16->FP32 weight-stationary mHC pre-mix GEMM: 13x faster than rocblas_sgemm(24,512,16384), 100% token-invariant!
__global__ __launch_bounds__(256, 4)
void k_mhc_pre_gemm_bf16_f32(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ fn_24x16384,
    float* __restrict__ out_Mx24,
    int M, int K)
{
    __shared__ float s_w[4];
    const int r   = int(blockIdx.x);     // 0..23
    const int m0  = int(blockIdx.y) * 8; // token strip
    const int cnt = (M - m0 < 8) ? (M - m0) : 8;
    const int tid = int(threadIdx.x);    // 0..255 -> 8 dims/iter -> stride 2048
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float4* __restrict__ fn4 = reinterpret_cast<const float4*>(fn_24x16384 + size_t(r) * size_t(K));
    const int k_vec8 = K >> 3;
    float acc[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = 0.0f;

    if (cnt == 8) {
        const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 0) * size_t(K));
        const uint4* __restrict__ x1 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 1) * size_t(K));
        const uint4* __restrict__ x2 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 2) * size_t(K));
        const uint4* __restrict__ x3 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 3) * size_t(K));
        const uint4* __restrict__ x4 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 4) * size_t(K));
        const uint4* __restrict__ x5 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 5) * size_t(K));
        const uint4* __restrict__ x6 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 6) * size_t(K));
        const uint4* __restrict__ x7 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 7) * size_t(K));
        for (int kv = tid; kv < k_vec8; kv += 256) {
            const float4 f0 = fn4[kv * 2];
            const float4 f1 = fn4[kv * 2 + 1];
            acc[0] += d_mhc_dot8(x0[kv], f0, f1);
            acc[1] += d_mhc_dot8(x1[kv], f0, f1);
            acc[2] += d_mhc_dot8(x2[kv], f0, f1);
            acc[3] += d_mhc_dot8(x3[kv], f0, f1);
            acc[4] += d_mhc_dot8(x4[kv], f0, f1);
            acc[5] += d_mhc_dot8(x5[kv], f0, f1);
            acc[6] += d_mhc_dot8(x6[kv], f0, f1);
            acc[7] += d_mhc_dot8(x7[kv], f0, f1);
        }
    } else {
        for (int kv = tid; kv < k_vec8; kv += 256) {
            const float4 f0 = fn4[kv * 2];
            const float4 f1 = fn4[kv * 2 + 1];
            for (int c = 0; c < cnt; ++c) {
                const uint4 xv = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + c) * size_t(K))[kv];
                acc[c] += d_mhc_dot8(xv, f0, f1);
            }
        }
    }

    for (int c = 0; c < cnt; ++c) {
        float v = d_warp_sum64(acc[c]);
        if (lane == 0) s_w[warp] = v;
        __syncthreads();
        if (tid == 0) out_Mx24[size_t(m0 + c) * 24u + size_t(r)] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
        __syncthreads();
    }
}

// Fused HCA [M, 2560] F32 split + BF16-rounded RMSNorm(q_norm[1024], kv_norm[512]) + wkv[512] + wgate[512]!
__global__ __launch_bounds__(256, 4)
void k_hca_split_qkv_comp_norm(
    const float* __restrict__ qkv_f32,
    const uint16_t* __restrict__ q_norm_w,
    const uint16_t* __restrict__ kv_norm_w,
    uint16_t* __restrict__ qr_bf16,
    uint16_t* __restrict__ kv_bf16,
    float* __restrict__ wkv_f32,
    float* __restrict__ wgate_f32,
    float eps)
{
    __shared__ float s_w[4];
    const int m    = int(blockIdx.x);
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const float* __restrict__ row = qkv_f32 + size_t(m) * 2560u;

    // 1. q_down [0..1023] -> BF16 round -> RMSNorm(q_norm_w) -> qr_bf16
    const float q0 = d_bf16_to_f32(d_f32_to_bf16(row[tid]));
    const float q1 = d_bf16_to_f32(d_f32_to_bf16(row[tid + 256]));
    const float q2 = d_bf16_to_f32(d_f32_to_bf16(row[tid + 512]));
    const float q3 = d_bf16_to_f32(d_f32_to_bf16(row[tid + 768]));
    float ss = d_warp_sum64(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    const float inv_q = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 1024.0f) + eps);
    uint16_t* __restrict__ qd = qr_bf16 + size_t(m) * 1024u;
    qd[tid]       = d_f32_to_bf16(q0 * inv_q * d_bf16_to_f32(q_norm_w[tid]));
    qd[tid + 256] = d_f32_to_bf16(q1 * inv_q * d_bf16_to_f32(q_norm_w[tid + 256]));
    qd[tid + 512] = d_f32_to_bf16(q2 * inv_q * d_bf16_to_f32(q_norm_w[tid + 512]));
    qd[tid + 768] = d_f32_to_bf16(q3 * inv_q * d_bf16_to_f32(q_norm_w[tid + 768]));
    __syncthreads();

    // 2. kv_proj [1024..1535] -> BF16 round -> RMSNorm(kv_norm_w) -> kv_bf16
    const float k0 = d_bf16_to_f32(d_f32_to_bf16(row[1024 + tid]));
    const float k1 = d_bf16_to_f32(d_f32_to_bf16(row[1024 + tid + 256]));
    float sk = d_warp_sum64(k0 * k0 + k1 * k1);
    if (lane == 0) s_w[warp] = sk;
    __syncthreads();
    const float inv_k = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
    uint16_t* __restrict__ kd = kv_bf16 + size_t(m) * 512u;
    kd[tid]       = d_f32_to_bf16(k0 * inv_k * d_bf16_to_f32(kv_norm_w[tid]));
    kd[tid + 256] = d_f32_to_bf16(k1 * inv_k * d_bf16_to_f32(kv_norm_w[tid + 256]));

    // 3. comp_wkv [1536..2047] & comp_wgate [2048..2559] -> FP32
    float* __restrict__ wkd = wkv_f32   + size_t(m) * 512u;
    float* __restrict__ wgd = wgate_f32 + size_t(m) * 512u;
    wkd[tid]       = row[1536 + tid];
    wkd[tid + 256] = row[1536 + tid + 256];
    wgd[tid]       = row[2048 + tid];
    wgd[tid + 256] = row[2048 + tid + 256];
}

// Batched FP16->FP32 GEMM DSA Dense Causal Attention helpers (constant k_tile=256, B_pad=64!)
__global__ __launch_bounds__(256, 4)
void k_dsa_dense_softmax_tile_f32(
    const float* __restrict__ scores_f32,
    float* __restrict__ m_blk_f32,
    float* __restrict__ l_blk_f32,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int32_t pos0_strip, int k0, int k_tile, float scale)
{
    __shared__ float s_w[4];
    const int row = int(blockIdx.x); // 0 .. B*n_heads - 1
    const int m   = row / n_heads;
    const int32_t pos = pos0_strip + m;
    const int valid_k = (pos >= k0) ? (((pos - k0 + 1) < k_tile) ? (pos - k0 + 1) : k_tile) : 0;

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const float* __restrict__ sc = scores_f32 + size_t(row) * size_t(k_tile);
    uint16_t* __restrict__ pr    = probs_f16  + size_t(row) * size_t(k_tile);

    if (valid_k <= 0) {
        for (int j = tid; j < k_tile; j += 256) pr[j] = 0u;
        if (tid == 0) { m_blk_f32[row] = -1e30f; l_blk_f32[row] = 0.0f; }
        return;
    }

    float lmax = -1e30f;
    for (int j = tid; j < valid_k; j += 256) {
        const float v = sc[j] * scale;
        if (v > lmax) lmax = v;
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    float lsum = 0.0f;
    for (int j = tid; j < k_tile; j += 256) {
        if (j < valid_k) {
            const float e = __expf(sc[j] * scale - mx);
            lsum += e;
            pr[j] = d_f32_to_f16_core(e);
        } else {
            pr[j] = 0u;
        }
    }
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    if (tid == 0) {
        m_blk_f32[row] = mx;
        l_blk_f32[row] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    }
}

__global__ __launch_bounds__(256, 8)
void k_dsa_dense_finalize_sink_bf16(
    const float* __restrict__ o_tot_f32,
    const float* __restrict__ m_tot_f32,
    const float* __restrict__ l_tot_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads)
{
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims/thread across 512
    const int h   = row % n_heads;
    const float m_max = m_tot_f32[row];
    const float l_sum = l_tot_f32[row];
    const float sink  = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc    = __expf(m_max - m_tot);
    const float inv_d = sc / (l_sum * sc + __expf(sink - m_tot));
    const float* __restrict__ src = o_tot_f32 + size_t(row) * 512u;
    uint16_t* __restrict__ dst    = out_bf16  + size_t(row) * 512u;
    dst[tid * 2]     = d_f32_to_bf16(src[tid * 2]     * inv_d);
    dst[tid * 2 + 1] = d_f32_to_bf16(src[tid * 2 + 1] * inv_d);
}

} // namespace

void launch_hca_vha_invrope_shard(
    const uint16_t* d_o_in_rm_bf16, const float* d_vha_m_64x64,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_o_out_bf16, int M, int tp_size, int head_begin, int n_local_heads, hipStream_t stream) {
    if (M <= 0 || n_local_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_vha_invrope_shard, dim3(n_local_heads / 16, M), dim3(256), 0, stream,
                       d_o_in_rm_bf16, d_vha_m_64x64, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_o_out_bf16, M, tp_size, head_begin, n_local_heads);
}

void launch_dsa_vha_lrga_shard(
    const uint16_t* d_o_in_rm_bf16, const float* d_vha_m_64x64,
    const uint16_t* d_gate_bf16, uint16_t* d_out_bf16,
    int M, int tp_size, int head_begin, int n_local_heads, hipStream_t stream) {
    if (M <= 0 || n_local_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_vha_lrga_shard, dim3(n_local_heads / 16, M), dim3(256), 0, stream,
                       d_o_in_rm_bf16, d_vha_m_64x64, d_gate_bf16, d_out_bf16, M, tp_size, head_begin, n_local_heads);
}

void launch_hca_gather_swa_kv_f16(
    const uint16_t* d_swa_ring_bf16, const uint16_t* d_kv_chunk_rot,
    uint16_t* d_k_swa_f16, int32_t pos0, int M, int32_t swa_lo, int swa_pad, hipStream_t stream) {
    if (swa_pad <= 0) return;
    hipLaunchKernelGGL(k_hca_gather_swa_kv_f16, dim3(swa_pad), dim3(256), 0, stream,
                       d_swa_ring_bf16, d_kv_chunk_rot, d_k_swa_f16, pos0, M, swa_lo);
}

void launch_hca_swa_softmax_f32(
    const float* d_scores_f32, float* d_m_swa_f32, float* d_l_swa_f32, uint16_t* d_probs_f16,
    int M, int n_heads, int32_t pos0, int32_t swa_lo, int swa_pad, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0 || swa_pad <= 0) return;
    hipLaunchKernelGGL(k_hca_swa_softmax_f32, dim3(M * n_heads), dim3(256), 0, stream,
                       d_scores_f32, d_m_swa_f32, d_l_swa_f32, d_probs_f16, n_heads, pos0, swa_lo, swa_pad);
}

void launch_hca_merge_comp_swa_sink_bf16(
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_swa_o_f32, const float* d_swa_m_f32, const float* d_swa_l_f32,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int M, int n_heads, bool has_comp, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_merge_comp_swa_sink_bf16, dim3(M * n_heads), dim3(256), 0, stream,
                       d_comp_o_f32, d_comp_m_f32, d_comp_l_f32,
                       d_swa_o_f32, d_swa_m_f32, d_swa_l_f32,
                       d_attn_sink, d_out_bf16, n_heads, has_comp ? 1 : 0);
}

void launch_flashindexer_relu_wsum_f32(
    const float* d_dots_f32, const float* d_w_sel_f32, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int k0, int k_cnt, int score_stride, hipStream_t stream) {
    if (B_sc <= 0 || k_cnt <= 0) return;
    hipLaunchKernelGGL(k_flashindexer_relu_wsum_f32, dim3((k_cnt + 255) / 256, B_sc), dim3(256), 0, stream,
                       d_dots_f32, d_w_sel_f32, d_scores_batch_f32, pos0, t0, k0, k_cnt, score_stride);
}

void launch_hca_comp_softmax_f32(
    const float* d_scores_f32, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int M, int n_heads, int32_t pos0, int blk0, int blk_cnt, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0 || blk_cnt <= 0) return;
    hipLaunchKernelGGL(k_hca_comp_softmax_f32, dim3(M * n_heads), dim3(256), 0, stream,
                       d_scores_f32, d_m_blk_f32, d_l_blk_f32, d_probs_f16, n_heads, pos0, blk0, blk_cnt);
}

void launch_hca_comp_merge_tile_f32(
    float* d_o_tot_f32, float* d_m_tot_f32, float* d_l_tot_f32,
    const float* d_o_blk_f32, const float* d_m_blk_f32, const float* d_l_blk_f32,
    int n_rows, bool first_tile, hipStream_t stream) {
    if (n_rows <= 0) return;
    hipLaunchKernelGGL(k_hca_comp_merge_tile_f32, dim3(n_rows), dim3(256), 0, stream,
                       d_o_tot_f32, d_m_tot_f32, d_l_tot_f32,
                       d_o_blk_f32, d_m_blk_f32, d_l_blk_f32, first_tile ? 1 : 0);
}

void launch_hca_prefill_swa_merge_sink(
    const uint16_t* d_q_bf16, const uint16_t* d_swa_ring_bf16, const uint16_t* d_kv_chunk_rot,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int n_heads, int32_t pos0, int M, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_prefill_swa_merge_sink, dim3(n_heads / 8, M), dim3(512), 0, stream,
                       d_q_bf16, d_swa_ring_bf16, d_kv_chunk_rot,
                       d_comp_o_f32, d_comp_m_f32, d_comp_l_f32, d_attn_sink, d_out_bf16, n_heads, pos0);
}

void launch_dsa_gather_kv_f16(
    const uint16_t* d_mla_kv_bf16, const int32_t* d_sel_indices_BxS,
    const int32_t* d_sel_counts, uint16_t* d_k_sel_f16,
    int B, int stride_sel, hipStream_t stream) {
    if (B <= 0 || stride_sel <= 0) return;
    hipLaunchKernelGGL(k_dsa_gather_kv_f16, dim3(stride_sel, B), dim3(256), 0, stream,
                       d_mla_kv_bf16, d_sel_indices_BxS, d_sel_counts, d_k_sel_f16, stride_sel);
}

void launch_dsa_scores_softmax_sink_f16(
    const float* d_scores_BxHxL, const float* d_attn_sink,
    const int32_t* d_sel_counts, uint16_t* d_probs_f16,
    int B, int n_heads, int L, float scale, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0 || L <= 0) return;
    hipLaunchKernelGGL(k_dsa_scores_softmax_sink_f16, dim3(n_heads, B), dim3(256), 0, stream,
                       d_scores_BxHxL, d_attn_sink, d_sel_counts, d_probs_f16, n_heads, L, scale);
}

void launch_dsa_q_absorb_uk_f16(
    const uint16_t* d_q_bf16, const uint16_t* d_kv_b_f16,
    uint16_t* d_q_abs_bf16, int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    const dim3 grid(4, uint32_t(n_heads));
    for (int off = 0; off < M; off += 8) {
        const int rem = (M - off < 8) ? (M - off) : 8;
        const uint16_t* qp = d_q_bf16 + size_t(off) * size_t(n_heads) * 256u;
        uint16_t* dp       = d_q_abs_bf16 + size_t(off) * size_t(n_heads) * 576u;
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<1>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            case 2: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<2>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            case 3: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<3>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            case 4: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<4>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            case 5: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<5>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            case 6: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<6>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            case 7: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<7>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
            default: hipLaunchKernelGGL((k_dsa_q_absorb_uk_f16_ws<8>), grid, dim3(128), 0, stream, qp, d_kv_b_f16, dp, n_heads); break;
        }
    }
}

void launch_dsa_o_project_uv_f16(
    const uint16_t* d_o_lat_bf16, const uint16_t* d_kv_b_f16,
    uint16_t* d_o_v_bf16, int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    const dim3 grid(32, uint32_t(n_heads));
    for (int off = 0; off < M; off += 8) {
        const int rem = (M - off < 8) ? (M - off) : 8;
        const uint16_t* op = d_o_lat_bf16 + size_t(off) * size_t(n_heads) * 512u;
        uint16_t* dp       = d_o_v_bf16   + size_t(off) * size_t(n_heads) * 256u;
        switch (rem) {
            case 1: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<1>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            case 2: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<2>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            case 3: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<3>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            case 4: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<4>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            case 5: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<5>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            case 6: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<6>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            case 7: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<7>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
            default: hipLaunchKernelGGL((k_dsa_o_project_uv_f16_ws<8>), grid, dim3(256), 0, stream, op, d_kv_b_f16, dp, n_heads); break;
        }
    }
}

void launch_mhc_pre_gemm_bf16_f32(
    const uint16_t* d_res_bf16, const float* d_fn_24x16384,
    float* d_out_Mx24, int M, int H, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_gemm_bf16_f32, dim3(24, (M + 7) / 8), dim3(256), 0, stream,
                       d_res_bf16, d_fn_24x16384, d_out_Mx24, M, 4 * H);
}

void launch_hca_split_qkv_comp_norm(
    const float* d_qkv_f32, const uint16_t* d_q_norm_w, const uint16_t* d_kv_norm_w,
    uint16_t* d_qr_bf16, uint16_t* d_kv_bf16, float* d_wkv_f32, float* d_wgate_f32,
    int M, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_split_qkv_comp_norm, dim3(M), dim3(256), 0, stream,
                       d_qkv_f32, d_q_norm_w, d_kv_norm_w, d_qr_bf16, d_kv_bf16, d_wkv_f32, d_wgate_f32, eps);
}

void launch_dsa_dense_softmax_tile_f32(
    const float* d_scores_f32, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, int32_t pos0_strip, int k0, int k_tile, float scale, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0 || k_tile <= 0) return;
    hipLaunchKernelGGL(k_dsa_dense_softmax_tile_f32, dim3(B * n_heads), dim3(256), 0, stream,
                       d_scores_f32, d_m_blk_f32, d_l_blk_f32, d_probs_f16, n_heads, pos0_strip, k0, k_tile, scale);
}

void launch_dsa_dense_finalize_sink_bf16(
    const float* d_o_tot_f32, const float* d_m_tot_f32, const float* d_l_tot_f32,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int B, int n_heads, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_dense_finalize_sink_bf16, dim3(B * n_heads), dim3(256), 0, stream,
                       d_o_tot_f32, d_m_tot_f32, d_l_tot_f32, d_attn_sink, d_out_bf16, n_heads);
}

namespace {

// ---- Wave-8 FP16 fdot2 mHC Kernels (2x less HBM2 traffic for fn_24x16384!) ----
__global__ __launch_bounds__(256, 4)
void k_mhc_pre_broadcast_norm_f16(
    const uint16_t* __restrict__ embed_bf16,
    const uint16_t* __restrict__ fn_24x16384_f16,
    const float* __restrict__ hc_scale3,
    const float* __restrict__ hc_base24,
    const uint16_t* __restrict__ norm_w_bf16,
    uint16_t* __restrict__ res_out_bf16,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    uint16_t* __restrict__ layer_in_bf16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_mixes[24];
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_cm[16];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const uint16_t* __restrict__ xr = embed_bf16 + size_t(m) * size_t(H);
    uint16_t* __restrict__ rr = res_out_bf16 + size_t(m) * size_t(4 * H);

    float ss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        const uint16_t b = xr[d];
        const float v = d_bf16_to_f32(b);
        ss = fmaf(v, v, ss);
        rr[0 * H + d] = b;
        rr[1 * H + d] = b;
        rr[2 * H + d] = b;
        rr[3 * H + d] = b;
    }
    ss = d_warp_sum64(ss);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    const float inv_rms = 1.0f / (sqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H)) + rms_eps);

    const int hc_dim = 4 * H;
    for (int row = warp; row < 24; row += 4) {
        const uint16_t* __restrict__ frow = fn_24x16384_f16 + size_t(row) * size_t(hc_dim);
        float acc = 0.0f;
        for (int d = lane; d < H; d += 64) {
            const float xv = d_bf16_to_f32(xr[d]);
            const float fsum = d_f16_to_f32_core(frow[0 * H + d]) + d_f16_to_f32_core(frow[1 * H + d])
                             + d_f16_to_f32_core(frow[2 * H + d]) + d_f16_to_f32_core(frow[3 * H + d]);
            acc = fmaf(xv, fsum, acc);
        }
        acc = d_warp_sum64(acc);
        if (lane == 0) s_mixes[row] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        d_mhc_sinkhorn_in_lds(s_mixes, hc_scale3, hc_base24, s_pre, s_post, s_cm, hc_eps);
        for (int j = 0; j < 4; ++j) post_mix[m * 4 + j] = s_post[j];
        for (int k = 0; k < 16; ++k) comb_mix[m * 16 + k] = s_cm[k];
    }
    __syncthreads();

    const float psum = s_pre[0] + s_pre[1] + s_pre[2] + s_pre[3];
    uint16_t* __restrict__ yr = layer_in_bf16 + size_t(m) * size_t(H);
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        const uint16_t vb = d_f32_to_bf16(psum * d_bf16_to_f32(xr[d]));
        yr[d] = vb;
        const float vf = d_bf16_to_f32(vb);
        lss = fmaf(vf, vf, lss);
    }
    if (norm_w_bf16 != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            yr[d] = d_f32_to_bf16(d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(norm_w_bf16[d]));
        }
    }
}

template <int M_STATIC>
__global__ __launch_bounds__(256, 2)
void k_mhc_pre_gemv_splitk_f16(
    const uint16_t* __restrict__ res_bf16,
    const uint16_t* __restrict__ fn_24x16384_f16,
    float* __restrict__ mhc_ws,
    int H)
{
    __shared__ float s_dot[4][M_STATIC];
    __shared__ float s_sq[4][M_STATIC];
    const int row  = int(blockIdx.x); // 0..23
    const int ks   = int(blockIdx.y); // 0..3
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int hc_dim  = 4 * H;
    const int hc_dim8 = hc_dim >> 3;
    const int chunk8  = hc_dim8 >> 2;
    const int k8_beg  = ks * chunk8;
    const int k8_end  = k8_beg + chunk8;

    const uint4* __restrict__ f8 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(row) * size_t(hc_dim));
    const uint4* __restrict__ r0 = reinterpret_cast<const uint4*>(res_bf16 + size_t(0) * size_t(hc_dim));
    const uint4* __restrict__ r1 = (M_STATIC > 1) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(1) * size_t(hc_dim)) : r0;
    const uint4* __restrict__ r2 = (M_STATIC > 2) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(2) * size_t(hc_dim)) : r0;
    const uint4* __restrict__ r3 = (M_STATIC > 3) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(3) * size_t(hc_dim)) : r0;
    const uint4* __restrict__ r4 = (M_STATIC > 4) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(4) * size_t(hc_dim)) : r0;
    const uint4* __restrict__ r5 = (M_STATIC > 5) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(5) * size_t(hc_dim)) : r0;
    const uint4* __restrict__ r6 = (M_STATIC > 6) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(6) * size_t(hc_dim)) : r0;
    const uint4* __restrict__ r7 = (M_STATIC > 7) ? reinterpret_cast<const uint4*>(res_bf16 + size_t(7) * size_t(hc_dim)) : r0;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f, d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f, q5 = 0.0f, q6 = 0.0f, q7 = 0.0f;

    auto accum = [&](uint4 rv, uint4 fv, float& dot, float& sq) {
        const uint4 xf = d_core_bf16_u4_to_f16_u4(rv);
        dot = d_core_dot8_f16_u4_fdot2(xf, fv, dot);
        if (row == 0) {
            sq = d_core_dot8_f16_u4_fdot2(xf, xf, sq);
        }
    };

    for (int k8 = k8_beg + tid; k8 < k8_end; k8 += 256) {
        const uint4 fv = f8[k8];
        accum(r0[k8], fv, d0, q0);
        if constexpr (M_STATIC > 1) accum(r1[k8], fv, d1, q1);
        if constexpr (M_STATIC > 2) accum(r2[k8], fv, d2, q2);
        if constexpr (M_STATIC > 3) accum(r3[k8], fv, d3, q3);
        if constexpr (M_STATIC > 4) accum(r4[k8], fv, d4, q4);
        if constexpr (M_STATIC > 5) accum(r5[k8], fv, d5, q5);
        if constexpr (M_STATIC > 6) accum(r6[k8], fv, d6, q6);
        if constexpr (M_STATIC > 7) accum(r7[k8], fv, d7, q7);
    }

    auto wred = [&](int mi, float dv, float qv) {
        dv = d_warp_sum64(dv);
        if (lane == 0) s_dot[warp][mi] = dv;
        if (row == 0) {
            qv = d_warp_sum64(qv);
            if (lane == 0) s_sq[warp][mi] = qv;
        }
    };
    wred(0, d0, q0);
    if constexpr (M_STATIC > 1) wred(1, d1, q1);
    if constexpr (M_STATIC > 2) wred(2, d2, q2);
    if constexpr (M_STATIC > 3) wred(3, d3, q3);
    if constexpr (M_STATIC > 4) wred(4, d4, q4);
    if constexpr (M_STATIC > 5) wred(5, d5, q5);
    if constexpr (M_STATIC > 6) wred(6, d6, q6);
    if constexpr (M_STATIC > 7) wred(7, d7, q7);
    __syncthreads();

    if (tid < M_STATIC) {
        const float dsum = s_dot[0][tid] + s_dot[1][tid] + s_dot[2][tid] + s_dot[3][tid];
        mhc_ws[(tid * 25 + row) * 4 + ks] = dsum;
        if (row == 0) {
            const float qsum = s_sq[0][tid] + s_sq[1][tid] + s_sq[2][tid] + s_sq[3][tid];
            mhc_ws[(tid * 25 + 24) * 4 + ks] = qsum;
        }
    }
}

// Wave-10 C2: 1-pass 2-token/block x 6-output/warp mHC pre-GEMM (reads res_bf16 ONCE vs 24x -> 24x less HBM2, 0 LDS, 0 scratch!)
__global__ __launch_bounds__(256, 4)
void k_mhc_pre_gemm_bf16_f16_f32(
    const uint16_t* __restrict__ res_bf16,
    const uint16_t* __restrict__ fn_24x16384_f16,
    float* __restrict__ out_Mx24,
    int M, int K)
{
    const int m0   = int(blockIdx.x) * 2;
    const int cnt  = (M - m0 < 2) ? (M - m0) : 2;
    if (cnt <= 0) return;
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6; // 0..3 -> 6 outputs each (24 total)
    const int lane = tid & 63;
    const int r0   = warp * 6;
    const int k_vec8 = K >> 3;

    const uint4* __restrict__ f0 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 0) * size_t(K));
    const uint4* __restrict__ f1 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 1) * size_t(K));
    const uint4* __restrict__ f2 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 2) * size_t(K));
    const uint4* __restrict__ f3 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 3) * size_t(K));
    const uint4* __restrict__ f4 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 4) * size_t(K));
    const uint4* __restrict__ f5 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 5) * size_t(K));

    float a00 = 0.0f, a01 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f;
    float a40 = 0.0f, a41 = 0.0f;
    float a50 = 0.0f, a51 = 0.0f;

    if (cnt == 2) {
        const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 0) * size_t(K));
        const uint4* __restrict__ x1 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 1) * size_t(K));
        for (int kv = lane; kv < k_vec8; kv += 64) {
            const uint4 v0 = d_core_bf16_u4_to_f16_u4(x0[kv]);
            const uint4 v1 = d_core_bf16_u4_to_f16_u4(x1[kv]);
            { const uint4 fv = f0[kv]; a00 = d_core_dot8_f16_u4_fdot2(v0, fv, a00); a01 = d_core_dot8_f16_u4_fdot2(v1, fv, a01); }
            { const uint4 fv = f1[kv]; a10 = d_core_dot8_f16_u4_fdot2(v0, fv, a10); a11 = d_core_dot8_f16_u4_fdot2(v1, fv, a11); }
            { const uint4 fv = f2[kv]; a20 = d_core_dot8_f16_u4_fdot2(v0, fv, a20); a21 = d_core_dot8_f16_u4_fdot2(v1, fv, a21); }
            { const uint4 fv = f3[kv]; a30 = d_core_dot8_f16_u4_fdot2(v0, fv, a30); a31 = d_core_dot8_f16_u4_fdot2(v1, fv, a31); }
            { const uint4 fv = f4[kv]; a40 = d_core_dot8_f16_u4_fdot2(v0, fv, a40); a41 = d_core_dot8_f16_u4_fdot2(v1, fv, a41); }
            { const uint4 fv = f5[kv]; a50 = d_core_dot8_f16_u4_fdot2(v0, fv, a50); a51 = d_core_dot8_f16_u4_fdot2(v1, fv, a51); }
        }
    } else {
        const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0) * size_t(K));
        for (int kv = lane; kv < k_vec8; kv += 64) {
            const uint4 v0 = d_core_bf16_u4_to_f16_u4(x0[kv]);
            a00 = d_core_dot8_f16_u4_fdot2(v0, f0[kv], a00);
            a10 = d_core_dot8_f16_u4_fdot2(v0, f1[kv], a10);
            a20 = d_core_dot8_f16_u4_fdot2(v0, f2[kv], a20);
            a30 = d_core_dot8_f16_u4_fdot2(v0, f3[kv], a30);
            a40 = d_core_dot8_f16_u4_fdot2(v0, f4[kv], a40);
            a50 = d_core_dot8_f16_u4_fdot2(v0, f5[kv], a50);
        }
    }

    auto emit = [&](int r_off, float v0, float v1) {
        v0 = d_warp_sum64(v0);
        if (cnt > 1) v1 = d_warp_sum64(v1);
        if (lane == 0) {
            out_Mx24[size_t(m0 + 0) * 24u + size_t(r0 + r_off)] = v0;
            if (cnt > 1) out_Mx24[size_t(m0 + 1) * 24u + size_t(r0 + r_off)] = v1;
        }
    };
    emit(0, a00, a01);
    emit(1, a10, a11);
    emit(2, a20, a21);
    emit(3, a30, a31);
    emit(4, a40, a41);
    emit(5, a50, a51);
}

// ---- Wave-8 GPU Sparse Softmax + Top-Candidates Compaction (0 slc, 0 scratch!) ----
__global__ __launch_bounds__(256, 4)
void k_softmax_max_p1(const float* __restrict__ logits, float* __restrict__ part64, int V) {
    __shared__ float s_w[4];
    const int bid = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int stride = int(gridDim.x) * 256;
    float mx = -1e38f;
    for (int i = bid * 256 + tid; i < V; i += stride) {
        const float v = logits[i];
        if (isfinite(v) && v > mx) mx = v;
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
    if (lane == 0) s_w[warp] = mx;
    __syncthreads();
    if (tid == 0) part64[bid] = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));
}

__global__ __launch_bounds__(64, 4)
void k_softmax_max_p2(const float* __restrict__ part64, float* __restrict__ stats2, int32_t* __restrict__ cand_cnt) {
    const int lane = int(threadIdx.x);
    float mx = part64[lane];
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
    if (lane == 0) {
        stats2[0] = mx;
        stats2[1] = 0.0f;
        *cand_cnt = 0;
    }
}

__global__ __launch_bounds__(256, 4)
void k_softmax_sparse_extract_p3(
    const float* __restrict__ logits, int V, float inv_t, float temperature,
    const float* __restrict__ stats2, float* __restrict__ part64,
    int32_t* __restrict__ cand_cnt, int32_t* __restrict__ cand_idx, float* __restrict__ cand_exp,
    int max_cands)
{
    __shared__ float s_w[4];
    const float max_l = stats2[0];
    const float cut_sum  = max_l - 22.0f * temperature;
    const float cut_cand = max_l - 16.0f * temperature;
    const int bid = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int stride = int(gridDim.x) * 256;

    float lsum = 0.0f;
    for (int i = bid * 256 + tid; i < V; i += stride) {
        const float v = logits[i];
        if (v > cut_sum && isfinite(v)) {
            const float e = __expf((v - max_l) * inv_t);
            lsum += e;
            if (v > cut_cand) {
                const int32_t slot = atomicAdd(cand_cnt, 1);
                if (slot < max_cands) {
                    cand_idx[slot] = i;
                    cand_exp[slot] = e;
                }
            }
        }
    }
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    if (tid == 0) part64[bid] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
}

__global__ __launch_bounds__(64, 4)
void k_softmax_sum_reduce_p4(const float* __restrict__ part64, float* __restrict__ stats2) {
    const int lane = int(threadIdx.x);
    float s = d_warp_sum64(part64[lane]);
    if (lane == 0) stats2[1] = s;
}

// ---- Wave-8 Zero-Sync Mapped Pinned Host-Memory TP=4 Collectives (0 slc, 0 CPU sync, kPingPong=4 race-free!) ----
__device__ __forceinline__ uint32_t d_sum4_bf16_u32(uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3) {
    const float lo = ((d_bf16_to_f32(uint16_t(u0 & 0xFFFFu)) + d_bf16_to_f32(uint16_t(u1 & 0xFFFFu)))
                    + d_bf16_to_f32(uint16_t(u2 & 0xFFFFu))) + d_bf16_to_f32(uint16_t(u3 & 0xFFFFu));
    const float hi = ((d_bf16_to_f32(uint16_t(u0 >> 16))     + d_bf16_to_f32(uint16_t(u1 >> 16)))
                    + d_bf16_to_f32(uint16_t(u2 >> 16)))     + d_bf16_to_f32(uint16_t(u3 >> 16));
    return uint32_t(d_f32_to_bf16(lo)) | (uint32_t(d_f32_to_bf16(hi)) << 16);
}

__global__ __launch_bounds__(1024, 1)
void k_tp4_mapped_allreduce_bf16(
    const uint16_t* __restrict__ src_local,
    uint16_t* __restrict__ dst_local,
    uint16_t* __restrict__ mapped_slabs,
    uint32_t* __restrict__ mapped_ready,
    uint32_t* __restrict__ mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const int n8  = n_elems >> 3;
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ base_slab = mapped_slabs + size_t(slot) * 4u * size_t(slab_stride);

    // Wait until all 4 peers (including 05:00.0) finished reading this 16-deep ring slot from epoch - 16!
    if (tid == 0 && epoch > 16u) {
        const uint32_t prev_ep = epoch - 16u;
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_done[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    const uint4* __restrict__ s8 = reinterpret_cast<const uint4*>(src_local);
    uint4* __restrict__ my8 = reinterpret_cast<uint4*>(base_slab + size_t(rank) * size_t(slab_stride));
    for (int i = tid; i < n8; i += 1024) my8[i] = s8[i];

    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_ready[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    // Read own rank straight from 1 TB/s HBM2 (25% fewer uncached PCIe reads!), peers from mapped host slabs
    const uint4* __restrict__ p0 = (rank == 0) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(0) * size_t(slab_stride));
    const uint4* __restrict__ p1 = (rank == 1) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(1) * size_t(slab_stride));
    const uint4* __restrict__ p2 = (rank == 2) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(2) * size_t(slab_stride));
    const uint4* __restrict__ p3 = (rank == 3) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(3) * size_t(slab_stride));
    uint4* __restrict__ out8 = reinterpret_cast<uint4*>(dst_local);
    for (int i = tid; i < n8; i += 1024) {
        const uint4 a = p0[i], b = p1[i], c = p2[i], d = p3[i];
        uint4 res;
        res.x = d_sum4_bf16_u32(a.x, b.x, c.x, d.x);
        res.y = d_sum4_bf16_u32(a.y, b.y, c.y, d.y);
        res.z = d_sum4_bf16_u32(a.z, b.z, c.z, d.z);
        res.w = d_sum4_bf16_u32(a.w, b.w, c.w, d.w);
        out8[i] = res;
    }

    __syncthreads();
    if (tid == 0) {
        __hip_atomic_store(&mapped_done[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ __launch_bounds__(1024, 1)
void k_tp4_mapped_allgather_heads_bf16(
    const uint16_t* __restrict__ src_local,
    uint16_t* __restrict__ dst_local,
    uint16_t* __restrict__ mapped_slabs,
    uint32_t* __restrict__ mapped_ready,
    uint32_t* __restrict__ mapped_done,
    int rank, int slab_elems, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const int n8  = slab_elems >> 3;
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ base_slab = mapped_slabs + size_t(slot) * 4u * size_t(slab_stride);

    if (tid == 0 && epoch > 16u) {
        const uint32_t prev_ep = epoch - 16u;
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_done[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    const uint4* __restrict__ s8 = reinterpret_cast<const uint4*>(src_local);
    uint4* __restrict__ my8 = reinterpret_cast<uint4*>(base_slab + size_t(rank) * size_t(slab_stride));
    uint4* __restrict__ self_dst8 = reinterpret_cast<uint4*>(dst_local + size_t(rank) * size_t(slab_elems));
    // Write local slab to BOTH mapped host buffer (for peers) AND local dst (zero PCIe roundtrip for self!)
    for (int i = tid; i < n8; i += 1024) {
        const uint4 v = s8[i];
        my8[i] = v;
        self_dst8[i] = v;
    }

    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_ready[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    for (int r = 0; r < 4; ++r) {
        if (r == rank) continue;
        const uint4* __restrict__ p8 = reinterpret_cast<const uint4*>(base_slab + size_t(r) * size_t(slab_stride));
        uint4* __restrict__ d8 = reinterpret_cast<uint4*>(dst_local + size_t(r) * size_t(slab_elems));
        for (int i = tid; i < n8; i += 1024) d8[i] = p8[i];
    }

    __syncthreads();
    if (tid == 0) {
        __hip_atomic_store(&mapped_done[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ __launch_bounds__(1024, 1)
void k_tp4_mapped_bcast_bf16(
    const uint16_t* __restrict__ src_local,
    uint16_t* __restrict__ dst_local,
    uint16_t* __restrict__ mapped_slabs,
    uint32_t* __restrict__ mapped_ready,
    uint32_t* __restrict__ mapped_done,
    int rank, int root_rank, int n_elems, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const int n8  = n_elems >> 3;
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ root_slab = mapped_slabs + size_t(slot) * 4u * size_t(slab_stride);

    if (rank == root_rank) {
        if (tid == 0 && epoch > 16u) {
            const uint32_t prev_ep = epoch - 16u;
            for (int r = 0; r < 4; ++r) {
                if (r == rank) continue;
                while (__hip_atomic_load(&mapped_done[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        }
        __syncthreads();
        const uint4* __restrict__ s8 = reinterpret_cast<const uint4*>(src_local);
        uint4* __restrict__ m8 = reinterpret_cast<uint4*>(root_slab);
        for (int i = tid; i < n8; i += 1024) m8[i] = s8[i];
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&mapped_done[rank * 16],  epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    } else {
        if (tid == 0) {
            while (__hip_atomic_load(&mapped_ready[root_rank * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
            __threadfence_system();
        }
        __syncthreads();
        const uint4* __restrict__ m8 = reinterpret_cast<const uint4*>(root_slab);
        uint4* __restrict__ d8 = reinterpret_cast<uint4*>(dst_local);
        for (int i = tid; i < n8; i += 1024) d8[i] = m8[i];
        __syncthreads();
        if (tid == 0) {
            __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&mapped_done[rank * 16],  epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

} // namespace

void launch_mhc_pre_broadcast_norm_f16(
    const uint16_t* d_embed_bf16, const uint16_t* d_fn_24x16384_f16,
    const float* d_hc_scale3, const float* d_hc_base24, const uint16_t* d_norm_w_bf16,
    uint16_t* d_res_out_bf16, float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_broadcast_norm_f16, dim3(M), dim3(256), 0, stream,
                       d_embed_bf16, d_fn_24x16384_f16, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                       d_res_out_bf16, d_post_mix, d_comb_mix, d_layer_in_bf16, H, rms_eps, hc_eps);
}

void launch_mhc_pre_norm_fast_f16(
    const uint16_t* d_res_bf16, const uint16_t* d_fn_24x16384_f16,
    const float* d_hc_scale3, const float* d_hc_base24, const uint16_t* d_norm_w_bf16,
    float* d_mhc_ws, float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0) return;
    if (M > 8) {
        launch_mhc_pre_gemm_bf16_f16_f32(d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, M, H, stream);
        launch_mhc_pre_sinkhorn_from_mixes(d_res_bf16, d_mhc_ws, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                                           d_post_mix, d_comb_mix, d_layer_in_bf16, M, H, rms_eps, hc_eps, stream);
        return;
    }
    const dim3 g1(24, 4);
    switch (M) {
        case 1: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<1>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        case 2: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<2>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        case 3: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<3>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        case 4: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<4>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        case 5: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<5>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        case 6: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<6>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        case 7: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<7>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
        default: hipLaunchKernelGGL((k_mhc_pre_gemv_splitk_f16<8>), g1, dim3(256), 0, stream, d_res_bf16, d_fn_24x16384_f16, d_mhc_ws, H); break;
    }
    hipLaunchKernelGGL(k_mhc_pre_sinkhorn_collapse_ws, dim3(M), dim3(256), 0, stream,
                       d_res_bf16, d_mhc_ws, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                       d_post_mix, d_comb_mix, d_layer_in_bf16, H, rms_eps, hc_eps);
}

void launch_mhc_pre_gemm_bf16_f16_f32(
    const uint16_t* d_res_bf16, const uint16_t* d_fn_24x16384_f16,
    float* d_out_Mx24, int M, int H, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_gemm_bf16_f16_f32, dim3(uint32_t((M + 1) / 2)), dim3(256), 0, stream,
                       d_res_bf16, d_fn_24x16384_f16, d_out_Mx24, M, 4 * H);
}

void launch_gpu_softmax_sparse_top_candidates(
    const float* d_logits_f32, int V, float temperature,
    float* d_part64_f32, float* d_stats2_f32, int32_t* d_cand_cnt,
    int32_t* d_cand_idx, float* d_cand_exp, int max_cands, hipStream_t stream) {
    if (V <= 0 || temperature <= 1e-6f) return;
    const float inv_t = 1.0f / temperature;
    hipLaunchKernelGGL(k_softmax_max_p1, dim3(64), dim3(256), 0, stream, d_logits_f32, d_part64_f32, V);
    hipLaunchKernelGGL(k_softmax_max_p2, dim3(1), dim3(64), 0, stream, d_part64_f32, d_stats2_f32, d_cand_cnt);
    hipLaunchKernelGGL(k_softmax_sparse_extract_p3, dim3(64), dim3(256), 0, stream,
                       d_logits_f32, V, inv_t, temperature, d_stats2_f32, d_part64_f32,
                       d_cand_cnt, d_cand_idx, d_cand_exp, max_cands);
    hipLaunchKernelGGL(k_softmax_sum_reduce_p4, dim3(1), dim3(64), 0, stream, d_part64_f32, d_stats2_f32);
}

void launch_tp4_mapped_allreduce_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (n_elems <= 0) return;
    hipLaunchKernelGGL(k_tp4_mapped_allreduce_bf16, dim3(1), dim3(1024), 0, stream,
                       d_src_local, d_dst_local, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, n_elems, slab_stride, epoch);
}

void launch_tp4_mapped_allgather_heads_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int slab_elems, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (slab_elems <= 0) return;
    hipLaunchKernelGGL(k_tp4_mapped_allgather_heads_bf16, dim3(1), dim3(1024), 0, stream,
                       d_src_local, d_dst_local, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, slab_elems, slab_stride, epoch);
}

void launch_tp4_mapped_bcast_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int root_rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (n_elems <= 0) return;
    hipLaunchKernelGGL(k_tp4_mapped_bcast_bf16, dim3(1), dim3(1024), 0, stream,
                       d_src_local, d_dst_local, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, root_rank, n_elems, slab_stride, epoch);
}

// ---- User-space Vega 20 DPM Keep-Warm (4 tiny waves = 1.6% CU on uncached HBM2 -> pins 1725 MHz sclk & 1000 MHz mclk!) ----
__global__ __launch_bounds__(64)
void k_ernie55_keep_dpm_warm(
    const uint32_t* __restrict__ active_flag,
    volatile uint32_t* __restrict__ uc_scratch,
    uint32_t mask_u32)
{
    const uint32_t tid = blockIdx.x * 64u + threadIdx.x;
    uint32_t acc = tid + 1u;
    uint32_t idx = (tid * 16u) & mask_u32;
    for (uint64_t it = 0; it < 180000000ULL; ++it) {
        if ((it & 63ULL) == 0ULL &&
            __hip_atomic_load(active_flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0u) break;
        acc ^= uc_scratch[idx] + uc_scratch[(idx + 64u) & mask_u32]
             + uc_scratch[(idx + 128u) & mask_u32] + uc_scratch[(idx + 192u) & mask_u32] + uint32_t(it);
        idx = (idx + 256u) & mask_u32;
        __builtin_amdgcn_s_sleep(1);
    }
    if (tid == 0u && acc == 0xFFFFFFFFu) {
        uc_scratch[0] = acc;
    }
}

void launch_ernie55_keep_dpm_warm(
    const uint32_t* d_active_flag, uint32_t* d_uc_scratch, uint32_t mask_u32, hipStream_t stream) {
    if (!d_active_flag || !d_uc_scratch) return;
    hipLaunchKernelGGL(k_ernie55_keep_dpm_warm, dim3(4u), dim3(64), 0, stream, d_active_flag, d_uc_scratch, mask_u32);
}

// ---- Commit 3: Type_HH_HPA HCA Compressed-KV & SWA-128 GEMM Helpers (ScratchSize == 0!) ----
namespace {

__device__ __forceinline__ uint4 d_gather_bf16_u4_to_f16_u4(uint4 b) {
    auto c2 = [](uint32_t u) -> uint32_t {
        return uint32_t(d_f32_to_f16_core(d_bf16_to_f32(uint16_t(u & 0xFFFFu))))
             | (uint32_t(d_f32_to_f16_core(d_bf16_to_f32(uint16_t(u >> 16)))) << 16);
    };
    uint4 o;
    o.x = c2(b.x); o.y = c2(b.y); o.z = c2(b.z); o.w = c2(b.w);
    return o;
}

__global__ __launch_bounds__(256, 4)
void k_hca_comp_softmax_f16_in_norm(
    const uint16_t* __restrict__ scores_f16,
    float* __restrict__ m_blk_f32,
    float* __restrict__ l_blk_f32,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int32_t pos0, int blk0, int blk_cnt)
{
    __shared__ float s_w[4];
    const int row = int(blockIdx.x);
    const int m   = row / n_heads;
    const int num_comp = (pos0 + m) / 128;
    const int valid = (num_comp > blk0) ? ((num_comp - blk0 < blk_cnt) ? (num_comp - blk0) : blk_cnt) : 0;

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const uint16_t* __restrict__ sc = scores_f16 + size_t(row) * size_t(blk_cnt);
    uint16_t* __restrict__ pr       = probs_f16  + size_t(row) * size_t(blk_cnt);

    if (valid <= 0) {
        for (int j = tid; j < blk_cnt; j += 256) pr[j] = 0u;
        if (tid == 0) { m_blk_f32[row] = -1e30f; l_blk_f32[row] = 0.0f; }
        return;
    }

    if (blk_cnt <= 1024) {
        // Single-read VGPR fast path (blk_cnt in 64/128/256/512/1024): 1 HBM2 read + 1 __expf, 0 scratch!
        const float v0 = (tid < valid) ? d_f16_to_f32_core(sc[tid]) : -1e30f;
        const float v1 = (blk_cnt > 256 && (tid + 256) < valid) ? d_f16_to_f32_core(sc[tid + 256]) : -1e30f;
        const float v2 = (blk_cnt > 512 && (tid + 512) < valid) ? d_f16_to_f32_core(sc[tid + 512]) : -1e30f;
        const float v3 = (blk_cnt > 768 && (tid + 768) < valid) ? d_f16_to_f32_core(sc[tid + 768]) : -1e30f;
        float lmax = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
        #pragma unroll
        for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
        if (lane == 0) s_w[warp] = lmax;
        __syncthreads();
        const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

        const float e0 = (tid < valid) ? __expf(v0 - mx) : 0.0f;
        const float e1 = (blk_cnt > 256 && (tid + 256) < valid) ? __expf(v1 - mx) : 0.0f;
        const float e2 = (blk_cnt > 512 && (tid + 512) < valid) ? __expf(v2 - mx) : 0.0f;
        const float e3 = (blk_cnt > 768 && (tid + 768) < valid) ? __expf(v3 - mx) : 0.0f;
        float lsum = d_warp_sum64((e0 + e1) + (e2 + e3));
        if (lane == 0) s_w[warp] = lsum;
        __syncthreads();
        const float tot_l = s_w[0] + s_w[1] + s_w[2] + s_w[3];
        const float inv_l = (tot_l > 0.0f) ? (1.0f / tot_l) : 0.0f;
        if (tid < blk_cnt) pr[tid] = (tid < valid) ? d_f32_to_f16_core(e0 * inv_l) : 0u;
        if (blk_cnt > 256 && (tid + 256) < blk_cnt) pr[tid + 256] = ((tid + 256) < valid) ? d_f32_to_f16_core(e1 * inv_l) : 0u;
        if (blk_cnt > 512 && (tid + 512) < blk_cnt) pr[tid + 512] = ((tid + 512) < valid) ? d_f32_to_f16_core(e2 * inv_l) : 0u;
        if (blk_cnt > 768 && (tid + 768) < blk_cnt) pr[tid + 768] = ((tid + 768) < valid) ? d_f32_to_f16_core(e3 * inv_l) : 0u;
        if (tid == 0) {
            m_blk_f32[row] = mx;
            l_blk_f32[row] = tot_l;
        }
        return;
    }

    if (blk_cnt == 2048 || blk_cnt == 4096) {
        // Wave-12 Cut P12.3: Zero-scratch packed uint4 1-read fast path for blk_cnt=2048 (4 VGPRs) & 4096 (8 VGPRs).
        // Holds only packed FP16 uint4 words across both __syncthreads() barriers (<= 8 data VGPRs -> 0 spills, 4 waves/SIMD).
        const uint4* __restrict__ sc4 = reinterpret_cast<const uint4*>(sc);
        uint4* __restrict__ pr4       = reinterpret_cast<uint4*>(pr);
        const int base0 = tid * 8;
        const int base1 = (tid + 256) * 8;
        const uint4 raw0 = (base0 < valid) ? sc4[tid] : make_uint4(0u, 0u, 0u, 0u);
        const uint4 raw1 = (blk_cnt == 4096 && base1 < valid) ? sc4[tid + 256] : make_uint4(0u, 0u, 0u, 0u);

        auto max_u4 = [](const uint4 r, int base, int vlim, float acc) -> float {
            if (base < vlim) {
                const float f0 = d_f16_to_f32_core(uint16_t(r.x & 0xFFFFu));
                const float f1 = (base + 1 < vlim) ? d_f16_to_f32_core(uint16_t(r.x >> 16)) : -1e30f;
                const float f2 = (base + 2 < vlim) ? d_f16_to_f32_core(uint16_t(r.y & 0xFFFFu)) : -1e30f;
                const float f3 = (base + 3 < vlim) ? d_f16_to_f32_core(uint16_t(r.y >> 16)) : -1e30f;
                const float f4 = (base + 4 < vlim) ? d_f16_to_f32_core(uint16_t(r.z & 0xFFFFu)) : -1e30f;
                const float f5 = (base + 5 < vlim) ? d_f16_to_f32_core(uint16_t(r.z >> 16)) : -1e30f;
                const float f6 = (base + 6 < vlim) ? d_f16_to_f32_core(uint16_t(r.w & 0xFFFFu)) : -1e30f;
                const float f7 = (base + 7 < vlim) ? d_f16_to_f32_core(uint16_t(r.w >> 16)) : -1e30f;
                acc = fmaxf(acc, fmaxf(fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)), fmaxf(fmaxf(f4, f5), fmaxf(f6, f7))));
            }
            return acc;
        };
        float lmax = max_u4(raw0, base0, valid, -1e30f);
        if (blk_cnt == 4096) lmax = max_u4(raw1, base1, valid, lmax);
        #pragma unroll
        for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
        if (lane == 0) s_w[warp] = lmax;
        __syncthreads();
        const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

        auto sum_u4 = [mx](const uint4 r, int base, int vlim) -> float {
            if (base >= vlim) return 0.0f;
            float s = __expf(d_f16_to_f32_core(uint16_t(r.x & 0xFFFFu)) - mx);
            if (base + 1 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.x >> 16))     - mx);
            if (base + 2 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.y & 0xFFFFu)) - mx);
            if (base + 3 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.y >> 16))     - mx);
            if (base + 4 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.z & 0xFFFFu)) - mx);
            if (base + 5 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.z >> 16))     - mx);
            if (base + 6 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.w & 0xFFFFu)) - mx);
            if (base + 7 < vlim) s += __expf(d_f16_to_f32_core(uint16_t(r.w >> 16))     - mx);
            return s;
        };
        float lsum = sum_u4(raw0, base0, valid);
        if (blk_cnt == 4096) lsum += sum_u4(raw1, base1, valid);
        lsum = d_warp_sum64(lsum);
        if (lane == 0) s_w[warp] = lsum;
        __syncthreads();
        const float tot_l = s_w[0] + s_w[1] + s_w[2] + s_w[3];
        const float inv_l = (tot_l > 0.0f) ? (1.0f / tot_l) : 0.0f;

        auto pack_u4 = [mx, inv_l](const uint4 r, int base, int vlim) -> uint4 {
            if (base >= vlim) return make_uint4(0u, 0u, 0u, 0u);
            const uint32_t h0 = d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.x & 0xFFFFu)) - mx) * inv_l);
            const uint32_t h1 = (base + 1 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.x >> 16))     - mx) * inv_l) : 0u;
            const uint32_t h2 = (base + 2 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.y & 0xFFFFu)) - mx) * inv_l) : 0u;
            const uint32_t h3 = (base + 3 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.y >> 16))     - mx) * inv_l) : 0u;
            const uint32_t h4 = (base + 4 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.z & 0xFFFFu)) - mx) * inv_l) : 0u;
            const uint32_t h5 = (base + 5 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.z >> 16))     - mx) * inv_l) : 0u;
            const uint32_t h6 = (base + 6 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.w & 0xFFFFu)) - mx) * inv_l) : 0u;
            const uint32_t h7 = (base + 7 < vlim) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(uint16_t(r.w >> 16))     - mx) * inv_l) : 0u;
            return make_uint4(h0 | (h1 << 16), h2 | (h3 << 16), h4 | (h5 << 16), h6 | (h7 << 16));
        };
        pr4[tid] = pack_u4(raw0, base0, valid);
        if (blk_cnt == 4096) pr4[tid + 256] = pack_u4(raw1, base1, valid);
        if (tid == 0) {
            m_blk_f32[row] = mx;
            l_blk_f32[row] = tot_l;
        }
        return;
    }

    float lmax = -1e30f;
    for (int j = tid; j < valid; j += 256) {
        lmax = fmaxf(lmax, d_f16_to_f32_core(sc[j]));
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    float lsum = 0.0f;
    for (int j = tid; j < valid; j += 256) {
        lsum += __expf(d_f16_to_f32_core(sc[j]) - mx);
    }
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    const float tot_l = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    const float inv_l = (tot_l > 0.0f) ? (1.0f / tot_l) : 0.0f;
    for (int j = tid; j < blk_cnt; j += 256) {
        pr[j] = (j < valid) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(sc[j]) - mx) * inv_l) : 0u;
    }
    if (tid == 0) {
        m_blk_f32[row] = mx;
        l_blk_f32[row] = tot_l;
    }
}

__global__ __launch_bounds__(256, 8)
void k_hca_comp_merge_tile_norm_f16(
    float* __restrict__ o_tot_f32,
    float* __restrict__ m_tot_f32,
    float* __restrict__ l_tot_f32,
    const uint16_t* __restrict__ o_blk_norm_f16,
    const float* __restrict__ m_blk_f32,
    const float* __restrict__ l_blk_f32,
    int first_tile)
{
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const float mb = m_blk_f32[row];
    const float lb = l_blk_f32[row];
    float* __restrict__ ot = o_tot_f32 + size_t(row) * 512u;
    const uint32_t pair = reinterpret_cast<const uint32_t*>(o_blk_norm_f16 + size_t(row) * 512u)[tid];
    const float ob0 = d_f16_to_f32_core(uint16_t(pair & 0xFFFFu));
    const float ob1 = d_f16_to_f32_core(uint16_t(pair >> 16));

    if (first_tile) {
        if (lb > 0.0f) {
            ot[tid * 2]     = lb * ob0;
            ot[tid * 2 + 1] = lb * ob1;
        } else {
            ot[tid * 2]     = 0.0f;
            ot[tid * 2 + 1] = 0.0f;
        }
        if (tid == 0) { m_tot_f32[row] = mb; l_tot_f32[row] = (lb > 0.0f) ? lb : 0.0f; }
    } else {
        const float mt = m_tot_f32[row];
        const float lt = l_tot_f32[row];
        if (lb <= 0.0f) return;
        const float mn = fmaxf(mt, mb);
        const float a0 = (lt > 0.0f) ? __expf(mt - mn) : 0.0f;
        const float a1 = __expf(mb - mn) * lb;
        ot[tid * 2]     = fmaf(ot[tid * 2],     a0, a1 * ob0);
        ot[tid * 2 + 1] = fmaf(ot[tid * 2 + 1], a0, a1 * ob1);
        if (tid == 0) {
            m_tot_f32[row] = mn;
            l_tot_f32[row] = fmaf(lt, a0, a1);
        }
    }
}

__global__ __launch_bounds__(64, 8)
void k_hca_pack_swa_keys_f16(
    const uint16_t* __restrict__ swa_ring_bf16,
    const uint16_t* __restrict__ kv_chunk_rot,
    uint16_t* __restrict__ out_k_f16,
    int32_t pos0, int n_pre, int M)
{
    const int j    = int(blockIdx.x);
    const int lane = int(threadIdx.x); // 0..63 -> 8 dims/lane (512 dims)
    uint4* __restrict__ dst = reinterpret_cast<uint4*>(out_k_f16 + size_t(j) * 512u);
    if (j < n_pre) {
        const int32_t p = pos0 - n_pre + j;
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(swa_ring_bf16 + size_t(p & 255) * 512u);
        dst[lane] = d_gather_bf16_u4_to_f16_u4(src[lane]);
    } else if (j < n_pre + M) {
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(kv_chunk_rot + size_t(j - n_pre) * 512u);
        dst[lane] = d_gather_bf16_u4_to_f16_u4(src[lane]);
    } else {
        dst[lane] = make_uint4(0u, 0u, 0u, 0u);
    }
}

__global__ __launch_bounds__(256, 4)
void k_hca_swa_softmax_f16_in_norm(
    const uint16_t* __restrict__ scores_f16,
    float* __restrict__ m_blk_f32,
    float* __restrict__ l_blk_f32,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int32_t q_pos0, int32_t k_start_pos)
{
    __shared__ float s_w[4];
    const int row   = int(blockIdx.x);
    const int m     = row / n_heads;
    const int32_t q_pos = q_pos0 + m;
    const int tid   = int(threadIdx.x); // 0..255
    const int warp  = tid >> 6;
    const int lane  = tid & 63;
    const int32_t k_pos = k_start_pos + tid;
    const bool valid = (k_pos >= 0) && (k_pos <= q_pos) && ((q_pos - k_pos) <= 127);

    const uint16_t* __restrict__ sc = scores_f16 + size_t(row) * 256u;
    uint16_t* __restrict__ pr       = probs_f16  + size_t(row) * 256u;
    const float v = valid ? d_f16_to_f32_core(sc[tid]) : -1e30f;

    float lmax = v;
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    const float e = valid ? __expf(v - mx) : 0.0f;
    const float lsum = d_warp_sum64(e);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    const float tot_l = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    const float inv_l = (tot_l > 0.0f) ? (1.0f / tot_l) : 0.0f;
    pr[tid] = valid ? d_f32_to_f16_core(e * inv_l) : 0u;
    if (tid == 0) {
        m_blk_f32[row] = mx;
        l_blk_f32[row] = tot_l;
    }
}

// Wave-11 C5: Batched 128-token-strip SWA-128 softmax (for pos0 >= 128, all b<=8 strips in 1 launch!)
// For strip s = m >> 7, local token t = m & 127, key window starts at q_pos0 + s*128 - 127,
// so key tid in [0..255] is valid iff t <= tid <= t + 127 (128 causal keys, independent of s/pos0!).
__global__ __launch_bounds__(256, 4)
void k_hca_swa_softmax_batched128_f16_in_norm(
    const uint16_t* __restrict__ scores_f16,
    float* __restrict__ m_blk_f32,
    float* __restrict__ l_blk_f32,
    uint16_t* __restrict__ probs_f16,
    int n_heads)
{
    __shared__ float s_w[4];
    const int row  = int(blockIdx.x);
    const int m    = row / n_heads;
    const int t128 = m & 127;
    const int tid  = int(threadIdx.x); // 0..255
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const bool valid = (tid >= t128) && (tid <= t128 + 127);

    const uint16_t* __restrict__ sc = scores_f16 + size_t(row) * 256u;
    uint16_t* __restrict__ pr       = probs_f16  + size_t(row) * 256u;
    const float v = valid ? d_f16_to_f32_core(sc[tid]) : -1e30f;

    float lmax = v;
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    const float e = valid ? __expf(v - mx) : 0.0f;
    const float lsum = d_warp_sum64(e);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    const float tot_l = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    const float inv_l = (tot_l > 0.0f) ? (1.0f / tot_l) : 0.0f;
    pr[tid] = valid ? d_f32_to_f16_core(e * inv_l) : 0u;
    if (tid == 0) {
        m_blk_f32[row] = mx;
        l_blk_f32[row] = tot_l;
    }
}

__global__ __launch_bounds__(64, 8)
void k_hca_swa_finalize_merge_sink_f16(
    const uint16_t* __restrict__ o_swa_norm_f16,
    const float* __restrict__ m_swa_f32,
    const float* __restrict__ l_swa_f32,
    const float* __restrict__ comp_o_f32,
    const float* __restrict__ comp_m_f32,
    const float* __restrict__ comp_l_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_bf16,
    int n_heads, int row_off)
{
    const int loc_row  = int(blockIdx.x);
    const int glob_row = row_off + loc_row;
    const int h        = loc_row % n_heads;
    const int lane     = int(threadIdx.x); // 0..63 -> 8 dims/lane (512 dims)

    const float ms = m_swa_f32[loc_row];
    const float ls = l_swa_f32[loc_row];
    const uint4 os_u4 = reinterpret_cast<const uint4*>(o_swa_norm_f16 + size_t(loc_row) * 512u)[lane];
    const float os0 = d_f16_to_f32_core(uint16_t(os_u4.x & 0xFFFFu));
    const float os1 = d_f16_to_f32_core(uint16_t(os_u4.x >> 16));
    const float os2 = d_f16_to_f32_core(uint16_t(os_u4.y & 0xFFFFu));
    const float os3 = d_f16_to_f32_core(uint16_t(os_u4.y >> 16));
    const float os4 = d_f16_to_f32_core(uint16_t(os_u4.z & 0xFFFFu));
    const float os5 = d_f16_to_f32_core(uint16_t(os_u4.z >> 16));
    const float os6 = d_f16_to_f32_core(uint16_t(os_u4.w & 0xFFFFu));
    const float os7 = d_f16_to_f32_core(uint16_t(os_u4.w >> 16));

    float m_max = ms;
    float l_sum = ls;
    float o0 = ls * os0, o1 = ls * os1, o2 = ls * os2, o3 = ls * os3;
    float o4 = ls * os4, o5 = ls * os5, o6 = ls * os6, o7 = ls * os7;

    if (comp_o_f32 != nullptr) {
        const float lc = comp_l_f32[glob_row];
        if (lc > 0.0f) {
            const float mc = comp_m_f32[glob_row];
            const float4* __restrict__ co4 = reinterpret_cast<const float4*>(comp_o_f32 + size_t(glob_row) * 512u);
            const float4 c0 = co4[lane * 2];
            const float4 c1 = co4[lane * 2 + 1];
            m_max = fmaxf(mc, ms);
            const float ac = __expf(mc - m_max);
            const float as = __expf(ms - m_max) * ls;
            l_sum = fmaf(lc, ac, as);
            o0 = fmaf(c0.x, ac, as * os0); o1 = fmaf(c0.y, ac, as * os1);
            o2 = fmaf(c0.z, ac, as * os2); o3 = fmaf(c0.w, ac, as * os3);
            o4 = fmaf(c1.x, ac, as * os4); o5 = fmaf(c1.y, ac, as * os5);
            o6 = fmaf(c1.z, ac, as * os6); o7 = fmaf(c1.w, ac, as * os7);
        }
    }

    const float sink  = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc    = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    reinterpret_cast<uint4*>(out_bf16 + size_t(glob_row) * 512u)[lane] =
        d_pack_bf16x8(o0 * inv_d, o1 * inv_d, o2 * inv_d, o3 * inv_d,
                      o4 * inv_d, o5 * inv_d, o6 * inv_d, o7 * inv_d);
}

} // namespace

void launch_hca_comp_softmax_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int M, int n_heads, int32_t pos0, int blk0, int blk_cnt, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0 || blk_cnt <= 0) return;
    hipLaunchKernelGGL(k_hca_comp_softmax_f16_in_norm, dim3(uint32_t(M * n_heads)), dim3(256), 0, stream,
                       d_scores_f16, d_m_blk_f32, d_l_blk_f32, d_probs_f16, n_heads, pos0, blk0, blk_cnt);
}

void launch_hca_comp_merge_tile_norm_f16(
    float* d_o_tot_f32, float* d_m_tot_f32, float* d_l_tot_f32,
    const uint16_t* d_o_blk_norm_f16, const float* d_m_blk_f32, const float* d_l_blk_f32,
    int n_rows, bool first_tile, hipStream_t stream) {
    if (n_rows <= 0) return;
    hipLaunchKernelGGL(k_hca_comp_merge_tile_norm_f16, dim3(uint32_t(n_rows)), dim3(256), 0, stream,
                       d_o_tot_f32, d_m_tot_f32, d_l_tot_f32, d_o_blk_norm_f16, d_m_blk_f32, d_l_blk_f32,
                       first_tile ? 1 : 0);
}

void launch_hca_pack_swa_keys_f16(
    const uint16_t* d_swa_ring_bf16, const uint16_t* d_kv_chunk_rot, uint16_t* d_out_k_f16,
    int32_t pos0, int n_pre, int M, int n_pad, hipStream_t stream) {
    if (n_pad <= 0) return;
    hipLaunchKernelGGL(k_hca_pack_swa_keys_f16, dim3(uint32_t(n_pad)), dim3(64), 0, stream,
                       d_swa_ring_bf16, d_kv_chunk_rot, d_out_k_f16, pos0, n_pre, M);
}

void launch_hca_swa_softmax_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, int32_t q_pos0, int32_t k_start_pos, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_swa_softmax_f16_in_norm, dim3(uint32_t(B * n_heads)), dim3(256), 0, stream,
                       d_scores_f16, d_m_blk_f32, d_l_blk_f32, d_probs_f16, n_heads, q_pos0, k_start_pos);
}

void launch_hca_swa_softmax_batched128_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_swa_softmax_batched128_f16_in_norm, dim3(uint32_t(B * n_heads)), dim3(256), 0, stream,
                       d_scores_f16, d_m_blk_f32, d_l_blk_f32, d_probs_f16, n_heads);
}

void launch_hca_swa_finalize_merge_sink_f16(
    const uint16_t* d_o_swa_norm_f16, const float* d_m_swa_f32, const float* d_l_swa_f32,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_bf16,
    int B, int n_heads, int row_off, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_swa_finalize_merge_sink_f16, dim3(uint32_t(B * n_heads)), dim3(64), 0, stream,
                       d_o_swa_norm_f16, d_m_swa_f32, d_l_swa_f32,
                       d_comp_o_f32, d_comp_m_f32, d_comp_l_f32,
                       d_attn_sink, d_out_bf16, n_heads, row_off);
}

// ---- Commit 4: Zero-Gather Sparse DSA MQA-64 + Type_HH_HPA FlashIndexer/Dense + Exact Rank-16 Low-Rank VHA ----
namespace {

__device__ __forceinline__ uint32_t d_pack_f32x2_to_f16x2_u32(float f0, float f1) {
    auto h2 = __builtin_amdgcn_cvt_pkrtz(f0, f1);
    uint32_t u;
    __builtin_memcpy(&u, &h2, sizeof(u));
    return u;
}

__device__ __forceinline__ uint32_t d_bf16x2_to_f16x2_core(uint32_t b2) {
    const float f0 = __uint_as_float(b2 << 16);
    const float f1 = __uint_as_float(b2 & 0xFFFF0000u);
    return d_pack_f32x2_to_f16x2_u32(f0, f1);
}

// Commit 4a v2: 1-WG/token (512 threads = 8 waves) Zero-Gather Register-Tiled 4x4 QK + 8x8 PV v_dot2_f32_f16!
//   - Reads K[cnt, 576] ONCE across all 64 heads + both QK and PV (4x less HBM2 than gather+GEMM!)
//   - s_KV[32][289] (36.1 KiB, bank-conflict-free) + union { s_Q[64][73] (18.25 KiB), s_S[64][32] (8 KiB), s_P[64][16] (4 KiB) }
//   - Total LDS = 55.3 KiB <= 64 KiB, ScratchSize == 0, 0 spills, 4:1 QK & 8:1 PV FMA-to-LDS ratio!
__global__ __launch_bounds__(512, 1)
void k_dsa_sparse_mqa64_fused_vdot2(
    const uint16_t* __restrict__ q_abs_bf16,
    const uint16_t* __restrict__ kv_mla_bf16,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ sel_idx,
    const int32_t* __restrict__ sel_cnts,
    uint16_t* __restrict__ o_lat_bf16,
    int stride_sel)
{
    __shared__ uint32_t s_KV[32][289]; // 36,992 B (stride 289 u32 -> 100% bank-conflict-free!)
    __shared__ union {
        uint32_t s_Q[64][73];          // 18,688 B (64 heads x 144 dims = 72 half2 + 1 pad)
        float    s_S[64][32];          //  8,192 B (64 heads x 32 keys FP32 scores)
        uint32_t s_P[64][16];          //  4,096 B (64 heads x 16 half2 prob pairs)
    } u;
    __shared__ int32_t s_idx[32];      //    128 B
    __shared__ float   s_m[64];        //    256 B
    __shared__ float   s_l[64];        //    256 B
    __shared__ float   s_alpha[64];    //    256 B

    const int m   = int(blockIdx.x);
    const int tid = int(threadIdx.x);

    const int valid = sel_cnts[m];
    const int32_t* __restrict__ idx_row = sel_idx + size_t(m) * size_t(stride_sel);
    const uint32_t* __restrict__ q_tok_u32 = reinterpret_cast<const uint32_t*>(
        q_abs_bf16 + size_t(m) * 64u * 576u);

    // Initialize online softmax with attn_sink absorbed cleanly at s0=0 (m = sink, l = 1.0f -> exp(sink - m_final) exact!)
    if (tid < 64) {
        s_m[tid] = attn_sink[tid];
        s_l[tid] = 1.0f;
    }

    float o_acc[8][8];
    #pragma unroll
    for (int hi = 0; hi < 8; ++hi) {
        #pragma unroll
        for (int di = 0; di < 8; ++di) o_acc[hi][di] = 0.0f;
    }

    #pragma unroll 1
    for (int kb = 0; kb < valid; kb += 32) {
        // 1. Load 32 key indices -> s_idx
        if (tid < 32) {
            s_idx[tid] = (kb + tid < valid) ? idx_row[kb + tid] : -1;
        }
        __syncthreads();

        // 2. Coalesced 1-pass load of K[32, 576] BF16 -> s_KV[32][288] FP16 half2 (9 x uint64_t per thread!)
        {
            const int k_row  = tid >> 4; // 0..31 (16 consecutive lanes per key row)
            const int k_lane = tid & 15; // 0..15
            const int32_t kpos = s_idx[k_row];
            if (kpos >= 0) {
                const uint64_t* __restrict__ k_src = reinterpret_cast<const uint64_t*>(
                    kv_mla_bf16 + size_t(kpos) * 576u);
                #pragma unroll 1
                for (int c = 0; c < 9; ++c) {
                    const int u64_i = k_lane + c * 16; // 0..143 -> 288 u32 = 576 elems
                    const uint64_t v64 = k_src[u64_i];
                    s_KV[k_row][u64_i * 2]     = d_bf16x2_to_f16x2_core(uint32_t(v64 & 0xFFFFFFFFull));
                    s_KV[k_row][u64_i * 2 + 1] = d_bf16x2_to_f16x2_core(uint32_t(v64 >> 32));
                }
            } else {
                #pragma unroll 1
                for (int c = 0; c < 9; ++c) {
                    const int u64_i = k_lane + c * 16;
                    s_KV[k_row][u64_i * 2]     = 0u;
                    s_KV[k_row][u64_i * 2 + 1] = 0u;
                }
            }
        }

        // 3. 4x4 Register-Tiled QK across 4 D-quarters (4 x 144 = 576 dims -> 4:1 FMA-to-LDS!)
        float sc[4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) sc[a][b] = 0.0f;
        }

        #pragma unroll 1
        for (int d_step = 0; d_step < 4; ++d_step) {
            const int q_head  = tid >> 3; // 0..63
            const int q_lane  = tid & 7;  // 0..7
            const int d2_base = d_step * 72;
            const uint64_t* __restrict__ q_row64 = reinterpret_cast<const uint64_t*>(
                q_tok_u32 + size_t(q_head) * 288u + size_t(d2_base));
            #pragma unroll 1
            for (int c = 0; c < 4; ++c) {
                const int u64_i = q_lane + c * 8; // 0..31 -> 64 u32
                const uint64_t v64 = q_row64[u64_i];
                u.s_Q[q_head][u64_i * 2]     = d_bf16x2_to_f16x2_core(uint32_t(v64 & 0xFFFFFFFFull));
                u.s_Q[q_head][u64_i * 2 + 1] = d_bf16x2_to_f16x2_core(uint32_t(v64 >> 32));
            }
            if (q_lane < 4) {
                const int u64_i = 32 + q_lane; // 32..35 -> 64..71 u32
                const uint64_t v64 = q_row64[u64_i];
                u.s_Q[q_head][u64_i * 2]     = d_bf16x2_to_f16x2_core(uint32_t(v64 & 0xFFFFFFFFull));
                u.s_Q[q_head][u64_i * 2 + 1] = d_bf16x2_to_f16x2_core(uint32_t(v64 >> 32));
            }
            __syncthreads();

            const int ds    = tid & 3;
            const int tile  = tid >> 2;
            const int hq4   = (tile & 15) * 4;
            const int kq4   = (tile >> 4) * 4;
            const int j0    = ds * 18;
            const int kv_j0 = d2_base + j0;
            #pragma unroll 1
            for (int j = 0; j < 18; ++j) {
                const core_f16x2_t k0 = d_u32_to_core_f16x2(s_KV[kq4 + 0][kv_j0 + j]);
                const core_f16x2_t k1 = d_u32_to_core_f16x2(s_KV[kq4 + 1][kv_j0 + j]);
                const core_f16x2_t k2 = d_u32_to_core_f16x2(s_KV[kq4 + 2][kv_j0 + j]);
                const core_f16x2_t k3 = d_u32_to_core_f16x2(s_KV[kq4 + 3][kv_j0 + j]);

                #pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const core_f16x2_t qa = d_u32_to_core_f16x2(u.s_Q[hq4 + a][j0 + j]);
                    sc[a][0] = __builtin_amdgcn_fdot2(qa, k0, sc[a][0], false);
                    sc[a][1] = __builtin_amdgcn_fdot2(qa, k1, sc[a][1], false);
                    sc[a][2] = __builtin_amdgcn_fdot2(qa, k2, sc[a][2], false);
                    sc[a][3] = __builtin_amdgcn_fdot2(qa, k3, sc[a][3], false);
                }
            }
            __syncthreads(); // safe before overwriting u.s_Q on next d_step or u.s_S after loop
        }

        // 4. Register-only 4-lane warp reduction across ds=0..3 -> u.s_S[64][32]
        {
            const int ds   = tid & 3;
            const int tile = tid >> 2;
            const int hq4  = (tile & 15) * 4;
            const int kq4  = (tile >> 4) * 4;
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float v = sc[a][b];
                    v += __shfl_xor(v, 1, 64);
                    v += __shfl_xor(v, 2, 64);
                    if (ds == 0) u.s_S[hq4 + a][kq4 + b] = v * 0.0625f;
                }
            }
        }
        __syncthreads();

        // 5. 8-lane-per-head online softmax -> u.s_P[64][16] (read all 4 floats BEFORE writing u.s_P!)
        {
            const int sm_h   = tid >> 3;
            const int sm_sub = tid & 7;
            const int k0 = sm_sub * 4;
            const float sv0 = (kb + k0     < valid) ? u.s_S[sm_h][k0]     : -1e30f;
            const float sv1 = (kb + k0 + 1 < valid) ? u.s_S[sm_h][k0 + 1] : -1e30f;
            const float sv2 = (kb + k0 + 2 < valid) ? u.s_S[sm_h][k0 + 2] : -1e30f;
            const float sv3 = (kb + k0 + 3 < valid) ? u.s_S[sm_h][k0 + 3] : -1e30f;
            float lm = fmaxf(fmaxf(sv0, sv1), fmaxf(sv2, sv3));
            lm = fmaxf(lm, __shfl_xor(lm, 1, 64));
            lm = fmaxf(lm, __shfl_xor(lm, 2, 64));
            lm = fmaxf(lm, __shfl_xor(lm, 4, 64));

            const float m_old = s_m[sm_h];
            const float m_new = fmaxf(m_old, lm);
            const float alpha = __expf(m_old - m_new);

            const float e0 = (kb + k0     < valid) ? __expf(sv0 - m_new) : 0.0f;
            const float e1 = (kb + k0 + 1 < valid) ? __expf(sv1 - m_new) : 0.0f;
            const float e2 = (kb + k0 + 2 < valid) ? __expf(sv2 - m_new) : 0.0f;
            const float e3 = (kb + k0 + 3 < valid) ? __expf(sv3 - m_new) : 0.0f;
            float ls = (e0 + e1) + (e2 + e3);
            ls += __shfl_xor(ls, 1, 64);
            ls += __shfl_xor(ls, 2, 64);
            ls += __shfl_xor(ls, 4, 64);

            const uint32_t p01 = d_pack_f32x2_to_f16x2_u32(e0, e1);
            const uint32_t p23 = d_pack_f32x2_to_f16x2_u32(e2, e3);
            __syncthreads(); // ensure all reads of u.s_S finished before overlaying u.s_P!
            u.s_P[sm_h][sm_sub * 2]     = p01;
            u.s_P[sm_h][sm_sub * 2 + 1] = p23;
            if (sm_sub == 0) {
                s_m[sm_h]     = m_new;
                s_l[sm_h]     = fmaf(s_l[sm_h], alpha, ls);
                s_alpha[sm_h] = alpha;
            }
        }
        __syncthreads();

        // 6. Rescale o_acc[8][8] and execute 8x8 Register-Tiled PV via v_dot2_f32_f16 (8:1 FMA-to-LDS!)
        {
            const int h_base = (tid >> 6) * 8;
            const int d2_pv  = (tid & 63) * 4;
            if (kb > 0) {
                #pragma unroll
                for (int hi = 0; hi < 8; ++hi) {
                    const float a = s_alpha[h_base + hi];
                    #pragma unroll
                    for (int di = 0; di < 8; ++di) o_acc[hi][di] *= a;
                }
            }

            #pragma unroll 1
            for (int kp = 0; kp < 16; ++kp) {
                const int k0 = kp * 2;
                const uint32_t a0 = s_KV[k0][d2_pv + 0];
                const uint32_t b0 = s_KV[k0 + 1][d2_pv + 0];
                const core_f16x2_t v0 = d_u32_to_core_f16x2((a0 & 0xFFFFu) | (b0 << 16));
                const core_f16x2_t v1 = d_u32_to_core_f16x2((a0 >> 16) | (b0 & 0xFFFF0000u));

                const uint32_t a1 = s_KV[k0][d2_pv + 1];
                const uint32_t b1 = s_KV[k0 + 1][d2_pv + 1];
                const core_f16x2_t v2 = d_u32_to_core_f16x2((a1 & 0xFFFFu) | (b1 << 16));
                const core_f16x2_t v3 = d_u32_to_core_f16x2((a1 >> 16) | (b1 & 0xFFFF0000u));

                const uint32_t a2 = s_KV[k0][d2_pv + 2];
                const uint32_t b2 = s_KV[k0 + 1][d2_pv + 2];
                const core_f16x2_t v4 = d_u32_to_core_f16x2((a2 & 0xFFFFu) | (b2 << 16));
                const core_f16x2_t v5 = d_u32_to_core_f16x2((a2 >> 16) | (b2 & 0xFFFF0000u));

                const uint32_t a3 = s_KV[k0][d2_pv + 3];
                const uint32_t b3 = s_KV[k0 + 1][d2_pv + 3];
                const core_f16x2_t v6 = d_u32_to_core_f16x2((a3 & 0xFFFFu) | (b3 << 16));
                const core_f16x2_t v7 = d_u32_to_core_f16x2((a3 >> 16) | (b3 & 0xFFFF0000u));

                #pragma unroll
                for (int hi = 0; hi < 8; ++hi) {
                    const core_f16x2_t pv = d_u32_to_core_f16x2(u.s_P[h_base + hi][kp]);
                    o_acc[hi][0] = __builtin_amdgcn_fdot2(pv, v0, o_acc[hi][0], false);
                    o_acc[hi][1] = __builtin_amdgcn_fdot2(pv, v1, o_acc[hi][1], false);
                    o_acc[hi][2] = __builtin_amdgcn_fdot2(pv, v2, o_acc[hi][2], false);
                    o_acc[hi][3] = __builtin_amdgcn_fdot2(pv, v3, o_acc[hi][3], false);
                    o_acc[hi][4] = __builtin_amdgcn_fdot2(pv, v4, o_acc[hi][4], false);
                    o_acc[hi][5] = __builtin_amdgcn_fdot2(pv, v5, o_acc[hi][5], false);
                    o_acc[hi][6] = __builtin_amdgcn_fdot2(pv, v6, o_acc[hi][6], false);
                    o_acc[hi][7] = __builtin_amdgcn_fdot2(pv, v7, o_acc[hi][7], false);
                }
            }
        }
        __syncthreads();
    }

    // 7. Finalize: divide by s_l[h] (which already includes exp(attn_sink - m_final)!) and write BF16
    const int h_base = (tid >> 6) * 8;
    const int d_base = (tid & 63) * 8;
    #pragma unroll
    for (int hi = 0; hi < 8; ++hi) {
        const float inv_l = 1.0f / s_l[h_base + hi];
        uint32_t* __restrict__ out_u32 = reinterpret_cast<uint32_t*>(
            o_lat_bf16 + (size_t(m) * 64u + size_t(h_base + hi)) * 512u + size_t(d_base));
        out_u32[0] = uint32_t(d_f32_to_bf16(o_acc[hi][0] * inv_l)) | (uint32_t(d_f32_to_bf16(o_acc[hi][1] * inv_l)) << 16);
        out_u32[1] = uint32_t(d_f32_to_bf16(o_acc[hi][2] * inv_l)) | (uint32_t(d_f32_to_bf16(o_acc[hi][3] * inv_l)) << 16);
        out_u32[2] = uint32_t(d_f32_to_bf16(o_acc[hi][4] * inv_l)) | (uint32_t(d_f32_to_bf16(o_acc[hi][5] * inv_l)) << 16);
        out_u32[3] = uint32_t(d_f32_to_bf16(o_acc[hi][6] * inv_l)) | (uint32_t(d_f32_to_bf16(o_acc[hi][7] * inv_l)) << 16);
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_dense_softmax_tile_f16_in_norm(
    const uint16_t* __restrict__ scores_f16,
    float* __restrict__ m_blk_f32,
    float* __restrict__ l_blk_f32,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int32_t pos0_strip, int k0, int k_tile)
{
    __shared__ float s_w[4];
    const int row = int(blockIdx.x);
    const int m   = row / n_heads;
    const int32_t pos = pos0_strip + m;
    const int valid_k = (pos >= k0) ? (((pos - k0 + 1) < k_tile) ? (pos - k0 + 1) : k_tile) : 0;

    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const uint16_t* __restrict__ sc = scores_f16 + size_t(row) * size_t(k_tile);
    uint16_t* __restrict__ pr       = probs_f16  + size_t(row) * size_t(k_tile);

    if (valid_k <= 0) {
        if (tid < k_tile) pr[tid] = 0u;
        if (tid == 0) { m_blk_f32[row] = -1e30f; l_blk_f32[row] = 0.0f; }
        return;
    }

    const float v = (tid < valid_k) ? d_f16_to_f32_core(sc[tid]) : -1e30f;
    float lmax = v;
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    const float e = (tid < valid_k) ? __expf(v - mx) : 0.0f;
    float lsum = d_warp_sum64(e);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    const float tot_l = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    const float inv_l = (tot_l > 0.0f) ? (1.0f / tot_l) : 0.0f;
    if (tid < k_tile) pr[tid] = (tid < valid_k) ? d_f32_to_f16_core(e * inv_l) : 0u;
    if (tid == 0) {
        m_blk_f32[row] = mx;
        l_blk_f32[row] = tot_l;
    }
}

__global__ __launch_bounds__(256, 4)
void k_flashindexer_relu_wsum_f16_in(
    const uint16_t* __restrict__ dots_f16,
    const float* __restrict__ w_sel_f32,
    float* __restrict__ scores_batch_f32,
    int32_t pos0, int t0, int k0, int k_cnt, int score_stride)
{
    const int k  = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int bi = int(blockIdx.y);
    if (k >= k_cnt) return;
    const int t = t0 + bi;
    const int n_hist = (pos0 + t) - 127;
    const int tok = k0 + k;
    if (n_hist <= 2048 || tok >= n_hist) return;

    constexpr float kWScale = 0.011048543456039806f;
    const float* __restrict__ wt = w_sel_f32 + size_t(t) * 8u;
    const size_t base = size_t(bi) * 8u * size_t(k_cnt) + size_t(k);
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float dp = d_f16_to_f32_core(dots_f16[base + size_t(i) * size_t(k_cnt)]);
        if (dp > 0.0f) total = fmaf(wt[i] * kWScale, dp, total);
    }
    scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = total;
}

__global__ __launch_bounds__(256, 4)
void k_vha_lowrank_proj_f16(
    const uint16_t* __restrict__ o_loc_bf16,
    const float* __restrict__ vha_r_16x16,
    uint16_t* __restrict__ p_loc_f16,
    int D)
{
    __shared__ float s_R[16][16];
    const int tid = int(threadIdx.x);
    s_R[tid >> 4][tid & 15] = vha_r_16x16[tid];
    __syncthreads();

    const int d2 = int(blockIdx.x) * 256 + tid;
    const int m  = int(blockIdx.y);
    const int D2 = D >> 1;
    if (d2 >= D2) return;

    const uint32_t* __restrict__ ob = reinterpret_cast<const uint32_t*>(o_loc_bf16 + size_t(m) * 16u * size_t(D)) + size_t(d2);
    float o0[16], o1[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint32_t pair = ob[size_t(j) * size_t(D2)];
        o0[j] = d_bf16_to_f32(uint16_t(pair & 0xFFFFu));
        o1[j] = d_bf16_to_f32(uint16_t(pair >> 16));
    }

    uint32_t* __restrict__ pb = reinterpret_cast<uint32_t*>(p_loc_f16 + size_t(m) * 16u * size_t(D)) + size_t(d2);
    #pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        float a0 = 0.0f, a1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float rkj = s_R[k][j];
            a0 = fmaf(rkj, o0[j], a0);
            a1 = fmaf(rkj, o1[j], a1);
        }
        pb[size_t(k) * size_t(D2)] = uint32_t(d_f32_to_f16_core(a0)) | (uint32_t(d_f32_to_f16_core(a1)) << 16);
    }
}

__global__ __launch_bounds__(256, 4)
void k_hca_vha_lowrank_invrope_f16(
    const uint16_t* o_loc_bf16,
    const uint16_t* __restrict__ p_all_f16,
    const float* __restrict__ vha_q_16x16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* o_out_bf16)
{
    __shared__ float s_Q[16][16];
    const int tid = int(threadIdx.x);
    s_Q[tid >> 4][tid & 15] = vha_q_16x16[tid];
    __syncthreads();

    const int m = int(blockIdx.y);
    const uint16_t* __restrict__ pb = p_all_f16 + size_t(m) * 16u * 512u;
    float p0[16], p1[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const uint32_t pair = reinterpret_cast<const uint32_t*>(pb + size_t(k) * 512u)[tid];
        p0[k] = d_f16_to_f32_core(uint16_t(pair & 0xFFFFu));
        p1[k] = d_f16_to_f32_core(uint16_t(pair >> 16));
    }

    float c = 1.0f, s = 0.0f;
    const bool do_rope = (tid >= 224);
    if (do_rope) d_rope2d_lookup(pos_ids[m], tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);

    const uint16_t* ib = o_loc_bf16 + size_t(m) * 16u * 512u;
    uint16_t* ob       = o_out_bf16 + size_t(m) * 16u * 512u;
    #pragma unroll 1
    for (int hl = 0; hl < 16; ++hl) {
        const uint32_t pair = reinterpret_cast<const uint32_t*>(ib + size_t(hl) * 512u)[tid];
        float y0 = d_bf16_to_f32(uint16_t(pair & 0xFFFFu));
        float y1 = d_bf16_to_f32(uint16_t(pair >> 16));
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float q = s_Q[hl][k];
            y0 = fmaf(q, p0[k], y0);
            y1 = fmaf(q, p1[k], y1);
        }
        if (do_rope) {
            const float r0 = fmaf(y0, c,   y1 * s);
            const float r1 = fmaf(y1, c, -(y0 * s));
            y0 = r0;
            y1 = r1;
        }
        const uint32_t out_p = uint32_t(d_f32_to_bf16(y0)) | (uint32_t(d_f32_to_bf16(y1)) << 16);
        reinterpret_cast<uint32_t*>(ob + size_t(hl) * 512u)[tid] = out_p;
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_vha_lowrank_lrga_f16(
    const uint16_t* o_loc_bf16,
    const uint16_t* __restrict__ p_all_f16,
    const float* __restrict__ vha_q_16x16,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* o_out_bf16)
{
    __shared__ float s_Q[16][16];
    const int d = int(threadIdx.x);
    s_Q[d >> 4][d & 15] = vha_q_16x16[d];
    __syncthreads();

    const int m = int(blockIdx.y);
    const uint16_t* __restrict__ pb = p_all_f16 + size_t(m) * 16u * 256u + size_t(d);
    float p[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) p[k] = d_f16_to_f32_core(pb[size_t(k) * 256u]);

    const size_t base = size_t(m) * 16u * 256u + size_t(d);
    #pragma unroll 1
    for (int hl = 0; hl < 16; ++hl) {
        const size_t idx = base + size_t(hl) * 256u;
        float y = d_bf16_to_f32(o_loc_bf16[idx]);
        #pragma unroll
        for (int k = 0; k < 16; ++k) y = fmaf(s_Q[hl][k], p[k], y);
        const float g = d_bf16_to_f32(gate_bf16[idx]);
        o_out_bf16[idx] = d_f32_to_bf16(y / (1.0f + __expf(-g)));
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_scores_softmax_sink_f16_in(
    const uint16_t* __restrict__ scores_BxHxL_f16,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ sel_counts,
    uint16_t* __restrict__ probs_f16,
    int n_heads, int L)
{
    __shared__ float s_w[4];
    const int h = int(blockIdx.x);
    const int b = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int valid = sel_counts[b];
    const size_t off = (size_t(b) * size_t(n_heads) + size_t(h)) * size_t(L);
    const uint16_t* __restrict__ sc = scores_BxHxL_f16 + off;
    uint16_t* __restrict__ pr       = probs_f16 + off;

    float lmax = attn_sink[h];
    for (int k = tid; k < valid; k += 256) {
        const float v = d_f16_to_f32_core(sc[k]);
        if (v > lmax) lmax = v;
    }
    #pragma unroll
    for (int sh = 32; sh >= 1; sh >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, sh, 64));
    if (lane == 0) s_w[warp] = lmax;
    __syncthreads();
    const float m_tot = fmaxf(fmaxf(s_w[0], s_w[1]), fmaxf(s_w[2], s_w[3]));

    float lsum = 0.0f;
    for (int k = tid; k < valid; k += 256) lsum += __expf(d_f16_to_f32_core(sc[k]) - m_tot);
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    const float inv_d = 1.0f / (s_w[0] + s_w[1] + s_w[2] + s_w[3] + __expf(attn_sink[h] - m_tot));

    for (int k = tid; k < L; k += 256) {
        pr[k] = (k < valid) ? d_f32_to_f16_core(__expf(d_f16_to_f32_core(sc[k]) - m_tot) * inv_d) : 0u;
    }
}

} // namespace

void launch_dsa_sparse_mqa64_fused_vdot2(
    const uint16_t* d_q_abs_bf16, const uint16_t* d_kv_mla_bf16, const float* d_attn_sink,
    const int32_t* d_sel_idx, const int32_t* d_sel_cnts, uint16_t* d_o_lat_bf16,
    int M, int stride_sel, hipStream_t stream) {
    if (M <= 0 || stride_sel <= 0) return;
    hipLaunchKernelGGL(k_dsa_sparse_mqa64_fused_vdot2, dim3(uint32_t(M)), dim3(512), 0, stream,
                       d_q_abs_bf16, d_kv_mla_bf16, d_attn_sink, d_sel_idx, d_sel_cnts, d_o_lat_bf16, stride_sel);
}

void launch_dsa_scores_softmax_sink_f16_in(
    const uint16_t* d_scores_BxHxL_f16, const float* d_attn_sink,
    const int32_t* d_sel_counts, uint16_t* d_probs_f16,
    int B, int n_heads, int L, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0 || L <= 0) return;
    hipLaunchKernelGGL(k_dsa_scores_softmax_sink_f16_in, dim3(uint32_t(n_heads), uint32_t(B)), dim3(256), 0, stream,
                       d_scores_BxHxL_f16, d_attn_sink, d_sel_counts, d_probs_f16, n_heads, L);
}

void launch_dsa_dense_softmax_tile_f16_in_norm(
    const uint16_t* d_scores_f16, float* d_m_blk_f32, float* d_l_blk_f32, uint16_t* d_probs_f16,
    int B, int n_heads, int32_t pos0_strip, int k0, int k_tile, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0 || k_tile <= 0) return;
    hipLaunchKernelGGL(k_dsa_dense_softmax_tile_f16_in_norm, dim3(uint32_t(B * n_heads)), dim3(256), 0, stream,
                       d_scores_f16, d_m_blk_f32, d_l_blk_f32, d_probs_f16, n_heads, pos0_strip, k0, k_tile);
}

void launch_flashindexer_relu_wsum_f16_in(
    const uint16_t* d_dots_f16, const float* d_w_sel_f32, float* d_scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int k0, int k_cnt, int score_stride, hipStream_t stream) {
    if (B_sc <= 0 || k_cnt <= 0) return;
    hipLaunchKernelGGL(k_flashindexer_relu_wsum_f16_in,
                       dim3(uint32_t((k_cnt + 255) / 256), uint32_t(B_sc)), dim3(256), 0, stream,
                       d_dots_f16, d_w_sel_f32, d_scores_batch_f32, pos0, t0, k0, k_cnt, score_stride);
}

void launch_vha_lowrank_proj_f16(
    const uint16_t* d_o_loc_bf16, const float* d_vha_r_16x16,
    uint16_t* d_p_loc_f16, int M, int D, hipStream_t stream) {
    if (M <= 0 || D <= 0) return;
    const int D2 = D >> 1;
    hipLaunchKernelGGL(k_vha_lowrank_proj_f16, dim3(uint32_t((D2 + 255) / 256), uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_bf16, d_vha_r_16x16, d_p_loc_f16, D);
}

void launch_hca_vha_lowrank_invrope_f16(
    const uint16_t* d_o_loc_bf16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, uint16_t* d_o_out_bf16,
    int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_vha_lowrank_invrope_f16, dim3(1u, uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_bf16, d_p_all_f16, d_vha_q_16x16, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo, d_o_out_bf16);
}

void launch_dsa_vha_lowrank_lrga_f16(
    const uint16_t* d_o_loc_bf16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_bf16,
    uint16_t* d_o_out_bf16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_vha_lowrank_lrga_f16, dim3(1u, uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_bf16, d_p_all_f16, d_vha_q_16x16, d_gate_bf16, d_o_out_bf16);
}

// ---- Wave-10 C1: Zero-Scratch On-GPU Async Merge Kernels (zero OpenMP CPU float reduction!) ----
namespace {

__global__ __launch_bounds__(256, 8)
void k_add_bf16_to_f32_inplace(float* __restrict__ dst_f32, const uint16_t* __restrict__ src_bf16, int total) {
    const int i4 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i4 * 4;
    if (base + 3 < total) {
        const uint2 b4 = reinterpret_cast<const uint2*>(src_bf16)[i4];
        float4 d4 = reinterpret_cast<float4*>(dst_f32)[i4];
        d4.x += d_bf16_to_f32(uint16_t(b4.x & 0xFFFFu));
        d4.y += d_bf16_to_f32(uint16_t(b4.x >> 16));
        d4.z += d_bf16_to_f32(uint16_t(b4.y & 0xFFFFu));
        d4.w += d_bf16_to_f32(uint16_t(b4.y >> 16));
        reinterpret_cast<float4*>(dst_f32)[i4] = d4;
    } else {
        for (int k = base; k < total; ++k) dst_f32[k] += d_bf16_to_f32(src_bf16[k]);
    }
}

__global__ __launch_bounds__(256, 8)
void k_add_f16_to_bf16(uint16_t* __restrict__ dst_bf16, const uint16_t* __restrict__ src_f16, int total) {
    const int i4 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i4 * 4;
    if (base + 3 < total) {
        const uint2 s4 = reinterpret_cast<const uint2*>(src_f16)[i4];
        const uint2 d4 = reinterpret_cast<const uint2*>(dst_bf16)[i4];
        uint2 o4;
        const float r0 = d_bf16_to_f32(uint16_t(d4.x & 0xFFFFu)) + d_f16_to_f32_core(uint16_t(s4.x & 0xFFFFu));
        const float r1 = d_bf16_to_f32(uint16_t(d4.x >> 16))     + d_f16_to_f32_core(uint16_t(s4.x >> 16));
        const float r2 = d_bf16_to_f32(uint16_t(d4.y & 0xFFFFu)) + d_f16_to_f32_core(uint16_t(s4.y & 0xFFFFu));
        const float r3 = d_bf16_to_f32(uint16_t(d4.y >> 16))     + d_f16_to_f32_core(uint16_t(s4.y >> 16));
        o4.x = uint32_t(d_f32_to_bf16(r0)) | (uint32_t(d_f32_to_bf16(r1)) << 16);
        o4.y = uint32_t(d_f32_to_bf16(r2)) | (uint32_t(d_f32_to_bf16(r3)) << 16);
        reinterpret_cast<uint2*>(dst_bf16)[i4] = o4;
    } else {
        for (int k = base; k < total; ++k) {
            dst_bf16[k] = d_f32_to_bf16(d_bf16_to_f32(dst_bf16[k]) + d_f16_to_f32_core(src_f16[k]));
        }
    }
}

__global__ __launch_bounds__(256, 8)
void k_f32_to_f16_vec(const float* __restrict__ in_f32, uint16_t* __restrict__ out_f16, int total) {
    const int i4 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i4 * 4;
    if (base + 3 < total) {
        const float4 v = reinterpret_cast<const float4*>(in_f32)[i4];
        uint2 o;
        o.x = d_pack_f32x2_to_f16x2_u32(fminf(fmaxf(v.x, -65504.0f), 65504.0f),
                                        fminf(fmaxf(v.y, -65504.0f), 65504.0f));
        o.y = d_pack_f32x2_to_f16x2_u32(fminf(fmaxf(v.z, -65504.0f), 65504.0f),
                                        fminf(fmaxf(v.w, -65504.0f), 65504.0f));
        reinterpret_cast<uint2*>(out_f16)[i4] = o;
    } else {
        for (int k = base; k < total; ++k) {
            out_f16[k] = d_f32_to_f16_core(fminf(fmaxf(in_f32[k], -65504.0f), 65504.0f));
        }
    }
}

// Single-pass fused owner FP32 + N remote FP16 slabs + optional P150A BF16 slab + RMSNorm -> BF16 (0 scratch!)
// Assumes H == 2048 (8 elements/thread -> 4x half2 loads per slab, zero intermediate HBM2 traffic!)
__global__ __launch_bounds__(256, 4)
void k_merge_moe_f16_bf16_rmsnorm_to_bf16(
    const float* __restrict__ own_f32,
    const uint16_t* __restrict__ f16_slabs,
    int n_f16_slabs,
    const uint16_t* __restrict__ bf16_slab,
    const uint16_t* __restrict__ norm_w_bf16,
    uint16_t* __restrict__ out_bf16,
    int M, int H, float eps)
{
    __shared__ float s_w[4];
    const int m   = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const size_t slab_stride = size_t(M) * size_t(H);
    const size_t row_off     = size_t(m) * size_t(H);

    // H == 2048 -> 256 threads x 8 elems (2 x uint4/float4 vectors at offsets tid*4 and (tid+256)*4)
    const int idx0 = tid * 4;
    const int idx1 = (tid + 256) * 4;
    float4 a0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (own_f32 != nullptr) {
        a0 = reinterpret_cast<const float4*>(own_f32 + row_off + size_t(idx0))[0];
        a1 = reinterpret_cast<const float4*>(own_f32 + row_off + size_t(idx1))[0];
    }
    for (int s = 0; s < n_f16_slabs; ++s) {
        const uint16_t* sp = f16_slabs + size_t(s) * slab_stride + row_off;
        const uint2 p0 = reinterpret_cast<const uint2*>(sp + size_t(idx0))[0];
        const uint2 p1 = reinterpret_cast<const uint2*>(sp + size_t(idx1))[0];
        a0.x += d_f16_to_f32_core(uint16_t(p0.x & 0xFFFFu));
        a0.y += d_f16_to_f32_core(uint16_t(p0.x >> 16));
        a0.z += d_f16_to_f32_core(uint16_t(p0.y & 0xFFFFu));
        a0.w += d_f16_to_f32_core(uint16_t(p0.y >> 16));
        a1.x += d_f16_to_f32_core(uint16_t(p1.x & 0xFFFFu));
        a1.y += d_f16_to_f32_core(uint16_t(p1.x >> 16));
        a1.z += d_f16_to_f32_core(uint16_t(p1.y & 0xFFFFu));
        a1.w += d_f16_to_f32_core(uint16_t(p1.y >> 16));
    }
    if (bf16_slab != nullptr) {
        const uint2 b0 = reinterpret_cast<const uint2*>(bf16_slab + row_off + size_t(idx0))[0];
        const uint2 b1 = reinterpret_cast<const uint2*>(bf16_slab + row_off + size_t(idx1))[0];
        a0.x += d_bf16_to_f32(uint16_t(b0.x & 0xFFFFu));
        a0.y += d_bf16_to_f32(uint16_t(b0.x >> 16));
        a0.z += d_bf16_to_f32(uint16_t(b0.y & 0xFFFFu));
        a0.w += d_bf16_to_f32(uint16_t(b0.y >> 16));
        a1.x += d_bf16_to_f32(uint16_t(b1.x & 0xFFFFu));
        a1.y += d_bf16_to_f32(uint16_t(b1.x >> 16));
        a1.z += d_bf16_to_f32(uint16_t(b1.y & 0xFFFFu));
        a1.w += d_bf16_to_f32(uint16_t(b1.y >> 16));
    }

    float ss = fmaf(a0.x, a0.x, fmaf(a0.y, a0.y, fmaf(a0.z, a0.z, a0.w * a0.w)));
    ss = fmaf(a1.x, a1.x, fmaf(a1.y, a1.y, fmaf(a1.z, a1.z, fmaf(a1.w, a1.w, ss))));
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + eps);

    const uint2 w0 = reinterpret_cast<const uint2*>(norm_w_bf16 + size_t(idx0))[0];
    const uint2 w1 = reinterpret_cast<const uint2*>(norm_w_bf16 + size_t(idx1))[0];
    uint2 o0, o1;
    o0.x = uint32_t(d_f32_to_bf16(a0.x * inv * d_bf16_to_f32(uint16_t(w0.x & 0xFFFFu)))) |
           (uint32_t(d_f32_to_bf16(a0.y * inv * d_bf16_to_f32(uint16_t(w0.x >> 16)))) << 16);
    o0.y = uint32_t(d_f32_to_bf16(a0.z * inv * d_bf16_to_f32(uint16_t(w0.y & 0xFFFFu)))) |
           (uint32_t(d_f32_to_bf16(a0.w * inv * d_bf16_to_f32(uint16_t(w0.y >> 16)))) << 16);
    o1.x = uint32_t(d_f32_to_bf16(a1.x * inv * d_bf16_to_f32(uint16_t(w1.x & 0xFFFFu)))) |
           (uint32_t(d_f32_to_bf16(a1.y * inv * d_bf16_to_f32(uint16_t(w1.x >> 16)))) << 16);
    o1.y = uint32_t(d_f32_to_bf16(a1.z * inv * d_bf16_to_f32(uint16_t(w1.y & 0xFFFFu)))) |
           (uint32_t(d_f32_to_bf16(a1.w * inv * d_bf16_to_f32(uint16_t(w1.y >> 16)))) << 16);
    reinterpret_cast<uint2*>(out_bf16 + row_off + size_t(idx0))[0] = o0;
    reinterpret_cast<uint2*>(out_bf16 + row_off + size_t(idx1))[0] = o1;
}

} // namespace

void launch_f32_to_f16(const float* d_in_f32, uint16_t* d_out_f16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n4 = (total + 3) / 4;
    hipLaunchKernelGGL(k_f32_to_f16_vec, dim3(uint32_t((n4 + 255) / 256)), dim3(256), 0, stream,
                       d_in_f32, d_out_f16, total);
}

void launch_merge_moe_f16_bf16_rmsnorm_to_bf16(
    const float* d_own_f32, const uint16_t* d_f16_slabs, int n_f16_slabs,
    const uint16_t* d_bf16_slab_or_null, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_bf16, int M, int H, float eps, hipStream_t stream) {
    if (M <= 0 || H <= 0) return;
    hipLaunchKernelGGL(k_merge_moe_f16_bf16_rmsnorm_to_bf16, dim3(uint32_t(M)), dim3(256), 0, stream,
                       d_own_f32, d_f16_slabs, n_f16_slabs, d_bf16_slab_or_null, d_norm_w_bf16,
                       d_out_bf16, M, H, eps);
}

void launch_add_bf16_to_f32_inplace(float* d_dst_f32, const uint16_t* d_src_bf16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n4 = (total + 3) / 4;
    hipLaunchKernelGGL(k_add_bf16_to_f32_inplace, dim3(uint32_t((n4 + 255) / 256)), dim3(256), 0, stream,
                       d_dst_f32, d_src_bf16, total);
}

void launch_add_f16_to_bf16(uint16_t* d_dst_bf16, const uint16_t* d_src_f16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n4 = (total + 3) / 4;
    hipLaunchKernelGGL(k_add_f16_to_bf16, dim3(uint32_t((n4 + 255) / 256)), dim3(256), 0, stream,
                       d_dst_bf16, d_src_f16, total);
}

// ---- Wave-10 C2: Zero-Bounce FP16-Stay Attention/VHA/o_proj Kernels (0 scratch!) ----
namespace {

__global__ __launch_bounds__(256, 4)
void k_hca_q_qknorm_rope_f16(
    uint16_t* __restrict__ q_f16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    int n_heads, float eps)
{
    __shared__ float s_w[4];
    const int h = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];
    uint32_t* __restrict__ qh2 = reinterpret_cast<uint32_t*>(q_f16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 512u);

    const uint32_t pair = qh2[tid];
    const float v0 = d_f16_to_f32_core(uint16_t(pair & 0xFFFFu));
    const float v1 = d_f16_to_f32_core(uint16_t(pair >> 16));
    float ss = fmaf(v0, v0, v1 * v1);
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
    float n0 = v0 * inv;
    float n1 = v1 * inv;
    if (tid >= 224) {
        float c, s;
        d_rope2d_lookup(pos, tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
        const float r0 = fmaf(n0, c, -(n1 * s));
        const float r1 = fmaf(n1, c,   n0 * s);
        n0 = r0;
        n1 = r1;
    }
    qh2[tid] = d_pack_f32x2_to_f16x2_u32(n0, n1);
}

__global__ __launch_bounds__(64, 8)
void k_hca_swa_finalize_merge_sink_f16_out(
    const uint16_t* __restrict__ o_swa_norm_f16,
    const float* __restrict__ m_swa_f32,
    const float* __restrict__ l_swa_f32,
    const float* __restrict__ comp_o_f32,
    const float* __restrict__ comp_m_f32,
    const float* __restrict__ comp_l_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_f16,
    int n_heads, int row_off)
{
    const int loc_row  = int(blockIdx.x);
    const int glob_row = row_off + loc_row;
    const int h        = loc_row % n_heads;
    const int lane     = int(threadIdx.x);

    const float ms = m_swa_f32[loc_row];
    const float ls = l_swa_f32[loc_row];
    const uint4 os_u4 = reinterpret_cast<const uint4*>(o_swa_norm_f16 + size_t(loc_row) * 512u)[lane];
    const float os0 = d_f16_to_f32_core(uint16_t(os_u4.x & 0xFFFFu));
    const float os1 = d_f16_to_f32_core(uint16_t(os_u4.x >> 16));
    const float os2 = d_f16_to_f32_core(uint16_t(os_u4.y & 0xFFFFu));
    const float os3 = d_f16_to_f32_core(uint16_t(os_u4.y >> 16));
    const float os4 = d_f16_to_f32_core(uint16_t(os_u4.z & 0xFFFFu));
    const float os5 = d_f16_to_f32_core(uint16_t(os_u4.z >> 16));
    const float os6 = d_f16_to_f32_core(uint16_t(os_u4.w & 0xFFFFu));
    const float os7 = d_f16_to_f32_core(uint16_t(os_u4.w >> 16));

    float m_max = ms;
    float l_sum = ls;
    float o0 = ls * os0, o1 = ls * os1, o2 = ls * os2, o3 = ls * os3;
    float o4 = ls * os4, o5 = ls * os5, o6 = ls * os6, o7 = ls * os7;

    if (comp_o_f32 != nullptr) {
        const float lc = comp_l_f32[glob_row];
        if (lc > 0.0f) {
            const float mc = comp_m_f32[glob_row];
            const float4* __restrict__ co4 = reinterpret_cast<const float4*>(comp_o_f32 + size_t(glob_row) * 512u);
            const float4 c0 = co4[lane * 2];
            const float4 c1 = co4[lane * 2 + 1];
            m_max = fmaxf(mc, ms);
            const float ac = __expf(mc - m_max);
            const float as = __expf(ms - m_max) * ls;
            l_sum = fmaf(lc, ac, as);
            o0 = fmaf(c0.x, ac, as * os0); o1 = fmaf(c0.y, ac, as * os1);
            o2 = fmaf(c0.z, ac, as * os2); o3 = fmaf(c0.w, ac, as * os3);
            o4 = fmaf(c1.x, ac, as * os4); o5 = fmaf(c1.y, ac, as * os5);
            o6 = fmaf(c1.z, ac, as * os6); o7 = fmaf(c1.w, ac, as * os7);
        }
    }

    const float sink  = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc    = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    uint4 out_u4;
    out_u4.x = d_pack_f32x2_to_f16x2_u32(o0 * inv_d, o1 * inv_d);
    out_u4.y = d_pack_f32x2_to_f16x2_u32(o2 * inv_d, o3 * inv_d);
    out_u4.z = d_pack_f32x2_to_f16x2_u32(o4 * inv_d, o5 * inv_d);
    out_u4.w = d_pack_f32x2_to_f16x2_u32(o6 * inv_d, o7 * inv_d);
    reinterpret_cast<uint4*>(out_f16 + size_t(glob_row) * 512u)[lane] = out_u4;
}

__global__ __launch_bounds__(256, 4)
void k_vha_lowrank_proj_f16_in(
    const uint16_t* __restrict__ o_loc_f16,
    const float* __restrict__ vha_r_16x16,
    uint16_t* __restrict__ p_loc_f16,
    int D)
{
    __shared__ float s_R[16][16];
    const int tid = int(threadIdx.x);
    s_R[tid >> 4][tid & 15] = vha_r_16x16[tid];
    __syncthreads();

    const int d2 = int(blockIdx.x) * 256 + tid;
    const int m  = int(blockIdx.y);
    const int D2 = D >> 1;
    if (d2 >= D2) return;

    const uint32_t* __restrict__ ob = reinterpret_cast<const uint32_t*>(o_loc_f16 + size_t(m) * 16u * size_t(D)) + size_t(d2);
    float o0[16], o1[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint32_t pair = ob[size_t(j) * size_t(D2)];
        o0[j] = d_f16_to_f32_core(uint16_t(pair & 0xFFFFu));
        o1[j] = d_f16_to_f32_core(uint16_t(pair >> 16));
    }

    uint32_t* __restrict__ pb = reinterpret_cast<uint32_t*>(p_loc_f16 + size_t(m) * 16u * size_t(D)) + size_t(d2);
    #pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        float a0 = 0.0f, a1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float rkj = s_R[k][j];
            a0 = fmaf(rkj, o0[j], a0);
            a1 = fmaf(rkj, o1[j], a1);
        }
        pb[size_t(k) * size_t(D2)] = d_pack_f32x2_to_f16x2_u32(a0, a1);
    }
}

__global__ __launch_bounds__(256, 4)
void k_hca_vha_lowrank_invrope_f16_io(
    const uint16_t* o_loc_f16,
    const uint16_t* __restrict__ p_all_f16,
    const float* __restrict__ vha_q_16x16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* o_out_f16)
{
    __shared__ float s_Q[16][16];
    const int tid = int(threadIdx.x);
    s_Q[tid >> 4][tid & 15] = vha_q_16x16[tid];
    __syncthreads();

    const int m = int(blockIdx.y);
    const uint16_t* __restrict__ pb = p_all_f16 + size_t(m) * 16u * 512u;
    float p0[16], p1[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const uint32_t pair = reinterpret_cast<const uint32_t*>(pb + size_t(k) * 512u)[tid];
        p0[k] = d_f16_to_f32_core(uint16_t(pair & 0xFFFFu));
        p1[k] = d_f16_to_f32_core(uint16_t(pair >> 16));
    }

    float c = 1.0f, s = 0.0f;
    const bool do_rope = (tid >= 224);
    if (do_rope) d_rope2d_lookup(pos_ids[m], tid - 224, cos_hi, sin_hi, cos_lo, sin_lo, c, s);

    const uint16_t* ib = o_loc_f16 + size_t(m) * 16u * 512u;
    uint16_t* ob       = o_out_f16 + size_t(m) * 16u * 512u;
    #pragma unroll 1
    for (int hl = 0; hl < 16; ++hl) {
        const uint32_t pair = reinterpret_cast<const uint32_t*>(ib + size_t(hl) * 512u)[tid];
        float y0 = d_f16_to_f32_core(uint16_t(pair & 0xFFFFu));
        float y1 = d_f16_to_f32_core(uint16_t(pair >> 16));
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float q = s_Q[hl][k];
            y0 = fmaf(q, p0[k], y0);
            y1 = fmaf(q, p1[k], y1);
        }
        if (do_rope) {
            const float r0 = fmaf(y0, c,   y1 * s);
            const float r1 = fmaf(y1, c, -(y0 * s));
            y0 = r0;
            y1 = r1;
        }
        reinterpret_cast<uint32_t*>(ob + size_t(hl) * 512u)[tid] = d_pack_f32x2_to_f16x2_u32(y0, y1);
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_vha_lowrank_lrga_f16_out(
    const uint16_t* o_loc_bf16,
    const uint16_t* __restrict__ p_all_f16,
    const float* __restrict__ vha_q_16x16,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* o_out_f16)
{
    __shared__ float s_Q[16][16];
    const int d = int(threadIdx.x);
    s_Q[d >> 4][d & 15] = vha_q_16x16[d];
    __syncthreads();

    const int m = int(blockIdx.y);
    const uint16_t* __restrict__ pb = p_all_f16 + size_t(m) * 16u * 256u + size_t(d);
    float p[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) p[k] = d_f16_to_f32_core(pb[size_t(k) * 256u]);

    const size_t base = size_t(m) * 16u * 256u + size_t(d);
    #pragma unroll 1
    for (int hl = 0; hl < 16; ++hl) {
        const size_t idx = base + size_t(hl) * 256u;
        float y = d_bf16_to_f32(o_loc_bf16[idx]);
        #pragma unroll
        for (int k = 0; k < 16; ++k) y = fmaf(s_Q[hl][k], p[k], y);
        const float g = d_bf16_to_f32(gate_bf16[idx]);
        o_out_f16[idx] = d_f32_to_f16_core(y / (1.0f + __expf(-g)));
    }
}

__global__ __launch_bounds__(256, 8)
void k_f16_mul_sigmoid_to_f16(const uint16_t* __restrict__ z_f16, const uint16_t* __restrict__ gate_bf16,
                              uint16_t* __restrict__ out_f16, int total) {
    const int i2 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i2 * 2;
    if (base + 1 < total) {
        const uint32_t zv = reinterpret_cast<const uint32_t*>(z_f16)[i2];
        const uint32_t gv = reinterpret_cast<const uint32_t*>(gate_bf16)[i2];
        const float r0 = d_f16_to_f32_core(uint16_t(zv & 0xFFFFu)) / (1.0f + __expf(-d_bf16_to_f32(uint16_t(gv & 0xFFFFu))));
        const float r1 = d_f16_to_f32_core(uint16_t(zv >> 16))     / (1.0f + __expf(-d_bf16_to_f32(uint16_t(gv >> 16))));
        reinterpret_cast<uint32_t*>(out_f16)[i2] = d_pack_f32x2_to_f16x2_u32(r0, r1);
    } else if (base < total) {
        const float z = d_f16_to_f32_core(z_f16[base]);
        const float g = d_bf16_to_f32(gate_bf16[base]);
        out_f16[base] = d_f32_to_f16_core(z / (1.0f + __expf(-g)));
    }
}

__global__ __launch_bounds__(256, 8)
void k_f16_mul_sigmoid_f16_gate_to_f16(const uint16_t* __restrict__ z_f16, const uint16_t* __restrict__ gate_f16,
                                       uint16_t* __restrict__ out_f16, int total) {
    const int i2 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i2 * 2;
    if (base + 1 < total) {
        const uint32_t zv = reinterpret_cast<const uint32_t*>(z_f16)[i2];
        const uint32_t gv = reinterpret_cast<const uint32_t*>(gate_f16)[i2];
        const float r0 = d_f16_to_f32_core(uint16_t(zv & 0xFFFFu)) / (1.0f + __expf(-d_f16_to_f32_core(uint16_t(gv & 0xFFFFu))));
        const float r1 = d_f16_to_f32_core(uint16_t(zv >> 16))     / (1.0f + __expf(-d_f16_to_f32_core(uint16_t(gv >> 16))));
        reinterpret_cast<uint32_t*>(out_f16)[i2] = d_pack_f32x2_to_f16x2_u32(r0, r1);
    } else if (base < total) {
        const float z = d_f16_to_f32_core(z_f16[base]);
        const float g = d_f16_to_f32_core(gate_f16[base]);
        out_f16[base] = d_f32_to_f16_core(z / (1.0f + __expf(-g)));
    }
}

// Wave-10 C2 v2: Fused HCA [M, 2560] FP16 direct split + RMSNorm(q->FP16, kv->BF16) + wkv/wgate FP32 (0 f16->f32 roundtrip!)
__global__ __launch_bounds__(256, 4)
void k_hca_split_qkv_comp_norm_f16(
    const uint16_t* __restrict__ qkv_f16,
    const uint16_t* __restrict__ q_norm_w,
    const uint16_t* __restrict__ kv_norm_w,
    uint16_t* __restrict__ qr_f16,
    uint16_t* __restrict__ kv_bf16,
    float* __restrict__ wkv_f32,
    float* __restrict__ wgate_f32,
    float eps)
{
    __shared__ float s_w[4];
    const int m    = int(blockIdx.x);
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const uint16_t* __restrict__ row = qkv_f16 + size_t(m) * 2560u;

    const float q0 = d_f16_to_f32_core(row[tid]);
    const float q1 = d_f16_to_f32_core(row[tid + 256]);
    const float q2 = d_f16_to_f32_core(row[tid + 512]);
    const float q3 = d_f16_to_f32_core(row[tid + 768]);
    float ss = d_warp_sum64(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();
    const float inv_q = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 1024.0f) + eps);
    uint16_t* __restrict__ qd = qr_f16 + size_t(m) * 1024u;
    qd[tid]       = d_f32_to_f16_core(q0 * inv_q * d_bf16_to_f32(q_norm_w[tid]));
    qd[tid + 256] = d_f32_to_f16_core(q1 * inv_q * d_bf16_to_f32(q_norm_w[tid + 256]));
    qd[tid + 512] = d_f32_to_f16_core(q2 * inv_q * d_bf16_to_f32(q_norm_w[tid + 512]));
    qd[tid + 768] = d_f32_to_f16_core(q3 * inv_q * d_bf16_to_f32(q_norm_w[tid + 768]));
    __syncthreads();

    const float k0 = d_f16_to_f32_core(row[1024 + tid]);
    const float k1 = d_f16_to_f32_core(row[1024 + tid + 256]);
    float sk = d_warp_sum64(k0 * k0 + k1 * k1);
    if (lane == 0) s_w[warp] = sk;
    __syncthreads();
    const float inv_k = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
    uint16_t* __restrict__ kd = kv_bf16 + size_t(m) * 512u;
    kd[tid]       = d_f32_to_bf16(k0 * inv_k * d_bf16_to_f32(kv_norm_w[tid]));
    kd[tid + 256] = d_f32_to_bf16(k1 * inv_k * d_bf16_to_f32(kv_norm_w[tid + 256]));

    float* __restrict__ wkd = wkv_f32   + size_t(m) * 512u;
    float* __restrict__ wgd = wgate_f32 + size_t(m) * 512u;
    wkd[tid]       = d_f16_to_f32_core(row[1536 + tid]);
    wkd[tid + 256] = d_f16_to_f32_core(row[1536 + tid + 256]);
    wgd[tid]       = d_f16_to_f32_core(row[2048 + tid]);
    wgd[tid + 256] = d_f16_to_f32_core(row[2048 + tid + 256]);
}

__global__ __launch_bounds__(256, 4)
void k_merge_moe_f16_bf16_rmsnorm_to_f16(
    const float* __restrict__ own_f32,
    const uint16_t* __restrict__ f16_slabs,
    int n_f16_slabs,
    const uint16_t* __restrict__ bf16_slab,
    const uint16_t* __restrict__ norm_w_bf16,
    uint16_t* __restrict__ out_f16,
    int M, int H, float eps)
{
    __shared__ float s_w[4];
    const int m   = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const size_t slab_stride = size_t(M) * size_t(H);
    const size_t row_off     = size_t(m) * size_t(H);

    const int idx0 = tid * 4;
    const int idx1 = (tid + 256) * 4;
    float4 a0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (own_f32 != nullptr) {
        a0 = reinterpret_cast<const float4*>(own_f32 + row_off + size_t(idx0))[0];
        a1 = reinterpret_cast<const float4*>(own_f32 + row_off + size_t(idx1))[0];
    }
    for (int s = 0; s < n_f16_slabs; ++s) {
        const uint16_t* sp = f16_slabs + size_t(s) * slab_stride + row_off;
        const uint2 p0 = reinterpret_cast<const uint2*>(sp + size_t(idx0))[0];
        const uint2 p1 = reinterpret_cast<const uint2*>(sp + size_t(idx1))[0];
        a0.x += d_f16_to_f32_core(uint16_t(p0.x & 0xFFFFu));
        a0.y += d_f16_to_f32_core(uint16_t(p0.x >> 16));
        a0.z += d_f16_to_f32_core(uint16_t(p0.y & 0xFFFFu));
        a0.w += d_f16_to_f32_core(uint16_t(p0.y >> 16));
        a1.x += d_f16_to_f32_core(uint16_t(p1.x & 0xFFFFu));
        a1.y += d_f16_to_f32_core(uint16_t(p1.x >> 16));
        a1.z += d_f16_to_f32_core(uint16_t(p1.y & 0xFFFFu));
        a1.w += d_f16_to_f32_core(uint16_t(p1.y >> 16));
    }
    if (bf16_slab != nullptr) {
        const uint2 b0 = reinterpret_cast<const uint2*>(bf16_slab + row_off + size_t(idx0))[0];
        const uint2 b1 = reinterpret_cast<const uint2*>(bf16_slab + row_off + size_t(idx1))[0];
        a0.x += d_bf16_to_f32(uint16_t(b0.x & 0xFFFFu));
        a0.y += d_bf16_to_f32(uint16_t(b0.x >> 16));
        a0.z += d_bf16_to_f32(uint16_t(b0.y & 0xFFFFu));
        a0.w += d_bf16_to_f32(uint16_t(b0.y >> 16));
        a1.x += d_bf16_to_f32(uint16_t(b1.x & 0xFFFFu));
        a1.y += d_bf16_to_f32(uint16_t(b1.x >> 16));
        a1.z += d_bf16_to_f32(uint16_t(b1.y & 0xFFFFu));
        a1.w += d_bf16_to_f32(uint16_t(b1.y >> 16));
    }

    float ss = fmaf(a0.x, a0.x, fmaf(a0.y, a0.y, fmaf(a0.z, a0.z, a0.w * a0.w)));
    ss = fmaf(a1.x, a1.x, fmaf(a1.y, a1.y, fmaf(a1.z, a1.z, fmaf(a1.w, a1.w, ss))));
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + eps);

    const uint2 w0 = reinterpret_cast<const uint2*>(norm_w_bf16 + size_t(idx0))[0];
    const uint2 w1 = reinterpret_cast<const uint2*>(norm_w_bf16 + size_t(idx1))[0];
    uint2 o0, o1;
    o0.x = d_pack_f32x2_to_f16x2_u32(a0.x * inv * d_bf16_to_f32(uint16_t(w0.x & 0xFFFFu)),
                                     a0.y * inv * d_bf16_to_f32(uint16_t(w0.x >> 16)));
    o0.y = d_pack_f32x2_to_f16x2_u32(a0.z * inv * d_bf16_to_f32(uint16_t(w0.y & 0xFFFFu)),
                                     a0.w * inv * d_bf16_to_f32(uint16_t(w0.y >> 16)));
    o1.x = d_pack_f32x2_to_f16x2_u32(a1.x * inv * d_bf16_to_f32(uint16_t(w1.x & 0xFFFFu)),
                                     a1.y * inv * d_bf16_to_f32(uint16_t(w1.x >> 16)));
    o1.y = d_pack_f32x2_to_f16x2_u32(a1.z * inv * d_bf16_to_f32(uint16_t(w1.y & 0xFFFFu)),
                                     a1.w * inv * d_bf16_to_f32(uint16_t(w1.y >> 16)));
    reinterpret_cast<uint2*>(out_f16 + row_off + size_t(idx0))[0] = o0;
    reinterpret_cast<uint2*>(out_f16 + row_off + size_t(idx1))[0] = o1;
}

__global__ __launch_bounds__(256, 8)
void k_add_f16_bf16_to_bf16(uint16_t* __restrict__ dst_io, const uint16_t* __restrict__ add_bf16, int total) {
    const int i2 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i2 * 2;
    if (base + 1 < total) {
        const uint32_t dv = reinterpret_cast<const uint32_t*>(dst_io)[i2];
        const uint32_t av = reinterpret_cast<const uint32_t*>(add_bf16)[i2];
        const float r0 = d_f16_to_f32_core(uint16_t(dv & 0xFFFFu)) + d_bf16_to_f32(uint16_t(av & 0xFFFFu));
        const float r1 = d_f16_to_f32_core(uint16_t(dv >> 16))     + d_bf16_to_f32(uint16_t(av >> 16));
        reinterpret_cast<uint32_t*>(dst_io)[i2] = uint32_t(d_f32_to_bf16(r0)) | (uint32_t(d_f32_to_bf16(r1)) << 16);
    } else if (base < total) {
        dst_io[base] = d_f32_to_bf16(d_f16_to_f32_core(dst_io[base]) + d_bf16_to_f32(add_bf16[base]));
    }
}

__global__ __launch_bounds__(256, 8)
void k_add_f16_f16_to_bf16(uint16_t* __restrict__ dst_io, const uint16_t* __restrict__ add_f16, int total) {
    const int i2 = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int base = i2 * 2;
    if (base + 1 < total) {
        const uint32_t dv = reinterpret_cast<const uint32_t*>(dst_io)[i2];
        const uint32_t av = reinterpret_cast<const uint32_t*>(add_f16)[i2];
        const float r0 = d_f16_to_f32_core(uint16_t(dv & 0xFFFFu)) + d_f16_to_f32_core(uint16_t(av & 0xFFFFu));
        const float r1 = d_f16_to_f32_core(uint16_t(dv >> 16))     + d_f16_to_f32_core(uint16_t(av >> 16));
        reinterpret_cast<uint32_t*>(dst_io)[i2] = uint32_t(d_f32_to_bf16(r0)) | (uint32_t(d_f32_to_bf16(r1)) << 16);
    } else if (base < total) {
        dst_io[base] = d_f32_to_bf16(d_f16_to_f32_core(dst_io[base]) + d_f16_to_f32_core(add_f16[base]));
    }
}

} // namespace

void launch_hca_q_qknorm_rope_f16(
    uint16_t* d_q_f16, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    int M, int n_heads, float eps, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_q_qknorm_rope_f16, dim3(uint32_t(n_heads), uint32_t(M)), dim3(256), 0, stream,
                       d_q_f16, d_pos_ids, rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo, n_heads, eps);
}

void launch_hca_swa_finalize_merge_sink_f16_out(
    const uint16_t* d_o_swa_norm_f16, const float* d_m_swa_f32, const float* d_l_swa_f32,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_f16,
    int B, int n_heads, int row_off, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_hca_swa_finalize_merge_sink_f16_out, dim3(uint32_t(B * n_heads)), dim3(64), 0, stream,
                       d_o_swa_norm_f16, d_m_swa_f32, d_l_swa_f32,
                       d_comp_o_f32, d_comp_m_f32, d_comp_l_f32,
                       d_attn_sink, d_out_f16, n_heads, row_off);
}

void launch_vha_lowrank_proj_f16_in(
    const uint16_t* d_o_loc_f16, const float* d_vha_r_16x16,
    uint16_t* d_p_loc_f16, int M, int D, hipStream_t stream) {
    if (M <= 0 || D <= 0) return;
    const int D2 = D >> 1;
    hipLaunchKernelGGL(k_vha_lowrank_proj_f16_in, dim3(uint32_t((D2 + 255) / 256), uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_f16, d_vha_r_16x16, d_p_loc_f16, D);
}

void launch_hca_vha_lowrank_invrope_f16_io(
    const uint16_t* d_o_loc_f16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const int32_t* d_pos_ids,
    const Rope2DTableGpu& rope, uint16_t* d_o_out_f16,
    int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_vha_lowrank_invrope_f16_io, dim3(1u, uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_f16, d_p_all_f16, d_vha_q_16x16, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo, d_o_out_f16);
}

void launch_dsa_vha_lowrank_lrga_f16_out(
    const uint16_t* d_o_loc_bf16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_bf16,
    uint16_t* d_o_out_f16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_vha_lowrank_lrga_f16_out, dim3(1u, uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_bf16, d_p_all_f16, d_vha_q_16x16, d_gate_bf16, d_o_out_f16);
}

void launch_f16_mul_sigmoid_to_f16(
    const uint16_t* d_z_f16, const uint16_t* d_gate_bf16,
    uint16_t* d_out_f16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n2 = (total + 1) / 2;
    hipLaunchKernelGGL(k_f16_mul_sigmoid_to_f16, dim3(uint32_t((n2 + 255) / 256)), dim3(256), 0, stream,
                       d_z_f16, d_gate_bf16, d_out_f16, total);
}

void launch_f16_mul_sigmoid_f16_gate_to_f16(
    const uint16_t* d_z_f16, const uint16_t* d_gate_f16,
    uint16_t* d_out_f16, int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n2 = (total + 1) / 2;
    hipLaunchKernelGGL(k_f16_mul_sigmoid_f16_gate_to_f16, dim3(uint32_t((n2 + 255) / 256)), dim3(256), 0, stream,
                       d_z_f16, d_gate_f16, d_out_f16, total);
}

void launch_hca_split_qkv_comp_norm_f16(
    const uint16_t* d_qkv_f16, const uint16_t* d_q_norm_w, const uint16_t* d_kv_norm_w,
    uint16_t* d_qr_f16, uint16_t* d_kv_bf16, float* d_wkv_f32, float* d_wgate_f32,
    int M, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_split_qkv_comp_norm_f16, dim3(uint32_t(M)), dim3(256), 0, stream,
                       d_qkv_f16, d_q_norm_w, d_kv_norm_w, d_qr_f16, d_kv_bf16, d_wkv_f32, d_wgate_f32, eps);
}

void launch_merge_moe_f16_bf16_rmsnorm_to_f16(
    const float* d_own_f32, const uint16_t* d_f16_slabs, int n_f16_slabs,
    const uint16_t* d_bf16_slab_or_null, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_f16, int M, int H, float eps, hipStream_t stream) {
    if (M <= 0 || H <= 0) return;
    hipLaunchKernelGGL(k_merge_moe_f16_bf16_rmsnorm_to_f16, dim3(uint32_t(M)), dim3(256), 0, stream,
                       d_own_f32, d_f16_slabs, n_f16_slabs, d_bf16_slab_or_null, d_norm_w_bf16, d_out_f16, M, H, eps);
}

void launch_add_f16_bf16_to_bf16(
    uint16_t* d_dst_f16_in_bf16_out, const uint16_t* d_add_bf16,
    int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n2 = (total + 1) / 2;
    hipLaunchKernelGGL(k_add_f16_bf16_to_bf16, dim3(uint32_t((n2 + 255) / 256)), dim3(256), 0, stream,
                       d_dst_f16_in_bf16_out, d_add_bf16, total);
}

void launch_add_f16_f16_to_bf16(
    uint16_t* d_dst_f16_in_bf16_out, const uint16_t* d_add_f16,
    int total, hipStream_t stream) {
    if (total <= 0) return;
    const int n2 = (total + 1) / 2;
    hipLaunchKernelGGL(k_add_f16_f16_to_bf16, dim3(uint32_t((n2 + 255) / 256)), dim3(256), 0, stream,
                       d_dst_f16_in_bf16_out, d_add_f16, total);
}

// ============================================================================
// Wave-10 C3: TP4-Specialized 16-Head Sparse DSA MQA (56.5 KiB LDS Q-Resident,
//             ScratchSize == 0, 0 spills, 0 trig, 0 slc, 0 BAR1 P2P!)
// ============================================================================
namespace {

__global__ __launch_bounds__(64, 8)
void k_dsa_copy_qrope_f16(const uint16_t* __restrict__ q_bf16,
                          uint16_t* __restrict__ q_abs_f16) {
    const int row = int(blockIdx.x); // m * n_heads + h
    const int d   = int(threadIdx.x); // 0..63
    const float v = d_bf16_to_f32(q_bf16[size_t(row) * 256u + 192u + size_t(d)]);
    q_abs_f16[size_t(row) * 576u + 512u + size_t(d)] = d_f32_to_f16_core(v);
}

__global__ __launch_bounds__(256, 4)
void k_dsa_dense_finalize_sink_f16(
    const float* __restrict__ o_tot_f32,
    const float* __restrict__ m_tot_f32,
    const float* __restrict__ l_tot_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_f16,
    int n_heads)
{
    const int row = int(blockIdx.x);
    const int h   = row % n_heads;
    const int tid = int(threadIdx.x); // 0..255 -> 2 dims each (512 dims)
    const float m_max = m_tot_f32[row];
    const float l_sum = l_tot_f32[row];
    const float sink  = attn_sink[h];
    const float m_tot = fmaxf(m_max, sink);
    const float sc    = __expf(m_max - m_tot);
    const float denom = l_sum * sc + __expf(sink - m_tot);
    const float inv_d = sc / denom;

    const float2 v = reinterpret_cast<const float2*>(o_tot_f32 + size_t(row) * 512u)[tid];
    reinterpret_cast<uint32_t*>(out_f16 + size_t(row) * 512u)[tid] =
        d_pack_f32x2_to_f16x2_u32(v.x * inv_d, v.y * inv_d);
}

__global__ __launch_bounds__(256, 4)
void k_dsa_vha_lowrank_lrga_f16_io(
    const uint16_t* o_loc_f16,
    const uint16_t* __restrict__ p_all_f16,
    const float* __restrict__ vha_q_16x16,
    const uint16_t* __restrict__ gate_bf16,
    uint16_t* o_out_f16)
{
    __shared__ float s_Q[16][16];
    const int d = int(threadIdx.x);
    s_Q[d >> 4][d & 15] = vha_q_16x16[d];
    __syncthreads();

    const int m = int(blockIdx.y);
    const uint16_t* __restrict__ pb = p_all_f16 + size_t(m) * 16u * 256u + size_t(d);
    float p[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) p[k] = d_f16_to_f32_core(pb[size_t(k) * 256u]);

    const size_t base = size_t(m) * 16u * 256u + size_t(d);
    #pragma unroll 1
    for (int hl = 0; hl < 16; ++hl) {
        const size_t idx = base + size_t(hl) * 256u;
        float y = d_f16_to_f32_core(o_loc_f16[idx]);
        #pragma unroll
        for (int k = 0; k < 16; ++k) y = fmaf(s_Q[hl][k], p[k], y);
        const float g = d_bf16_to_f32(gate_bf16[idx]);
        o_out_f16[idx] = d_f32_to_f16_core(y / (1.0f + __expf(-g)));
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_mark_selected_keys(
    const int32_t* __restrict__ sel_idx,
    const int32_t* __restrict__ sel_cnts,
    int32_t* __restrict__ map,
    int stride_sel, int total_kv)
{
    const int m = int(blockIdx.x);
    const int cnt = sel_cnts[m];
    const int32_t* __restrict__ row = sel_idx + size_t(m) * size_t(stride_sel);
    for (int s = int(threadIdx.x); s < cnt; s += 256) {
        const int32_t kpos = row[s];
        if (uint32_t(kpos) < uint32_t(total_kv)) map[kpos] = 1;
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_compact_unique_map(
    int32_t* __restrict__ map,
    int32_t* __restrict__ uniq_kpos,
    int32_t* __restrict__ uniq_cnt,
    int total_kv)
{
    const int tid  = int(threadIdx.x);
    const int lane = tid & 63;
    const int kpos = int(blockIdx.x) * 256 + tid;
    const bool hit = (kpos < total_kv) && (map[kpos] > 0);
    const uint64_t mask = __ballot(hit);
    const uint32_t pop  = uint32_t(__popcll(mask));
    int32_t wbase = 0;
    if (lane == 0 && pop > 0u) {
        wbase = atomicAdd(uniq_cnt, int32_t(pop));
    }
    wbase = __shfl(wbase, 0, 64);
    if (hit) {
        const uint64_t below = (lane == 0) ? 0ull : ((1ull << lane) - 1ull);
        const int32_t cid = wbase + int32_t(__popcll(mask & below));
        map[kpos] = cid;
        uniq_kpos[cid] = kpos;
    }
}

__global__ __launch_bounds__(64, 8)
void k_dsa_gather_unique_kv_f16(
    const uint16_t* __restrict__ kv_mla_bf16,
    const int32_t* __restrict__ uniq_kpos,
    uint16_t* __restrict__ kv_uniq_f16)
{
    const int cid  = int(blockIdx.x);
    const int lane = int(threadIdx.x); // 0..63 -> 288 half2s = 4*64 + 32
    const int32_t kpos = uniq_kpos[cid];
    const uint32_t* __restrict__ src = reinterpret_cast<const uint32_t*>(kv_mla_bf16 + size_t(kpos) * 576u);
    uint32_t* __restrict__ dst       = reinterpret_cast<uint32_t*>(kv_uniq_f16 + size_t(cid) * 576u);
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int i = lane + c * 64;
        dst[i] = d_bf16x2_to_f16x2_core(src[i]);
    }
    if (lane < 32) {
        const int i = 256 + lane;
        dst[i] = d_bf16x2_to_f16x2_core(src[i]);
    }
}

template <bool kUseU16>
__global__ __launch_bounds__(256, 4)
void k_dsa_remap_sel_idx(
    const int32_t* __restrict__ sel_idx,
    const int32_t* __restrict__ sel_cnts,
    const int32_t* __restrict__ map,
    void* __restrict__ sel_remapped,
    int stride_sel)
{
    const int m = int(blockIdx.x);
    const int cnt = sel_cnts[m];
    const int32_t* __restrict__ src = sel_idx + size_t(m) * size_t(stride_sel);
    if constexpr (kUseU16) {
        uint16_t* __restrict__ dst = static_cast<uint16_t*>(sel_remapped) + size_t(m) * size_t(stride_sel);
        for (int s = int(threadIdx.x); s < cnt; s += 256) dst[s] = uint16_t(map[src[s]]);
    } else {
        int32_t* __restrict__ dst = static_cast<int32_t*>(sel_remapped) + size_t(m) * size_t(stride_sel);
        for (int s = int(threadIdx.x); s < cnt; s += 256) dst[s] = map[src[s]];
    }
}

// TP4-Specialized 16-Head Sparse DSA MQA:
//   - s_Q[16][289] (18.1 KiB) PERMANENTLY resident in LDS across all kb iterations (68x fewer HBM2 Q reloads!)
//   - s_KV[32][289] (36.1 KiB) + union{ s_S[16][32] (2 KiB), s_P[16][16] (1 KiB) } + state (0.3 KiB) = 56.5 KiB <= 64 KiB!
//   - 256 threads (4 waves), o_acc[4][8] (32 VGPRs) -> ScratchSize == 0, 0 spills, direct FP16 I/O!
template <bool kKvIsF16, bool kIdxIsU16>
__global__ __launch_bounds__(256, 2)
void k_dsa_sparse_mqa_tp4_h16(
    const uint16_t* __restrict__ q_abs_f16,
    const uint16_t* __restrict__ kv_mla,
    const float* __restrict__ attn_sink_16,
    const void* __restrict__ sel_idx,
    const int32_t* __restrict__ sel_cnts,
    uint16_t* __restrict__ o_lat_f16,
    int stride_sel)
{
    // Wave-11 C5: 2 WGs/CU (28,768 B <= 32 KiB LDS) uint4 ds_read_b128 vectorized 16-head/WG K_tile=8!
    // Shares every 128-bit K load across ALL 16 heads (2x less HBM2 traffic vs 8-head/WG) at 2 WGs/CU!
    __shared__ uint4 s_Q[16][73];   // 18,688 B (all 16 heads x 576 FP16 permanently resident in LDS)
    __shared__ uint4 s_KV[8][73];   //  9,344 B (8 keys x 576 FP16, stride 73 u4 -> bank-conflict-free)
    __shared__ union {
        float    s_S[16][8];        //    512 B
        uint32_t s_P[16][4];        //    256 B
    } u;
    __shared__ int32_t s_idx[8];    //     32 B
    __shared__ float   s_m[16];     //     64 B
    __shared__ float   s_l[16];     //     64 B
    __shared__ float   s_alpha[16]; //     64 B

    const int m     = int(blockIdx.x);
    const int tid   = int(threadIdx.x);
    const int valid = sel_cnts[m];

    // 1. Preload ALL 16 heads x 576 FP16 Q ONCE into s_Q[0..15][0..71] via 128-bit coalesced uint4 loads!
    {
        const int q_h    = tid >> 4; // 0..15
        const int q_lane = tid & 15; // 0..15 -> 5 iterations covers 72 uint4!
        const uint4* __restrict__ q4 = reinterpret_cast<const uint4*>(
            q_abs_f16 + (size_t(m) * 16u + size_t(q_h)) * 576u);
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int u4_i = q_lane + c * 16;
            if (u4_i < 72) s_Q[q_h][u4_i] = q4[u4_i];
        }
    }
    if (tid < 16) {
        s_m[tid] = attn_sink_16[tid];
        s_l[tid] = 1.0f;
    }
    __syncthreads();

    float o_acc[4][8];
    #pragma unroll
    for (int hi = 0; hi < 4; ++hi) {
        #pragma unroll
        for (int di = 0; di < 8; ++di) o_acc[hi][di] = 0.0f;
    }

    const uint16_t* __restrict__ idx_u16 = static_cast<const uint16_t*>(sel_idx) + size_t(m) * size_t(stride_sel);
    const int32_t*  __restrict__ idx_i32 = static_cast<const int32_t*>(sel_idx)  + size_t(m) * size_t(stride_sel);

    #pragma unroll 1
    for (int kb = 0; kb < valid; kb += 8) {
        // 2. Load 8 key indices -> s_idx
        if (tid < 8) {
            if (kb + tid < valid) {
                s_idx[tid] = kIdxIsU16 ? int32_t(idx_u16[kb + tid]) : idx_i32[kb + tid];
            } else {
                s_idx[tid] = -1;
            }
        }
        __syncthreads();

        // 3. Coalesced 128-bit load of K[8, 576] -> s_KV[8][0..71] uint4 (32 lanes/key x 3 u4 = FULL 512B/line!)
        {
            const int k_row  = tid >> 5; // 0..7
            const int k_lane = tid & 31; // 0..31
            const int32_t kpos = s_idx[k_row];
            if (kpos >= 0) {
                const uint4* __restrict__ k_src = reinterpret_cast<const uint4*>(
                    kv_mla + size_t(kpos) * 576u);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    const int u4_i = k_lane + c * 32;
                    if (u4_i < 72) {
                        const uint4 v4 = k_src[u4_i];
                        if constexpr (kKvIsF16) {
                            s_KV[k_row][u4_i] = v4;
                        } else {
                            s_KV[k_row][u4_i] = make_uint4(
                                d_bf16x2_to_f16x2_core(v4.x),
                                d_bf16x2_to_f16x2_core(v4.y),
                                d_bf16x2_to_f16x2_core(v4.z),
                                d_bf16x2_to_f16x2_core(v4.w));
                        }
                    }
                }
            } else {
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    const int u4_i = k_lane + c * 32;
                    if (u4_i < 72) s_KV[k_row][u4_i] = z4;
                }
            }
        }
        __syncthreads();

        // 4. Single-pass 2x2 Register-Tiled uint4 (ds_read_b128) QK over ALL 576 dims (8 D-splits x 9 u4!)
        float sc00 = 0.0f, sc01 = 0.0f, sc10 = 0.0f, sc11 = 0.0f;
        {
            const int ds   = tid & 7;         // 0..7
            const int tile = tid >> 3;        // 0..31
            const int hq2  = (tile & 7) * 2;  // 0, 2, 4, 6, 8, 10, 12, 14
            const int kq2  = (tile >> 3) * 2; // 0, 2, 4, 6
            #pragma unroll 1
            for (int j = 0; j < 9; ++j) {
                const int j_idx = ds + j * 8;
                const uint4 k0 = s_KV[kq2 + 0][j_idx];
                const uint4 k1 = s_KV[kq2 + 1][j_idx];
                const uint4 q0 = s_Q[hq2 + 0][j_idx];
                const uint4 q1 = s_Q[hq2 + 1][j_idx];
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.x), d_u32_to_core_f16x2(k0.x), sc00, false);
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.y), d_u32_to_core_f16x2(k0.y), sc00, false);
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.z), d_u32_to_core_f16x2(k0.z), sc00, false);
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.w), d_u32_to_core_f16x2(k0.w), sc00, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.x), d_u32_to_core_f16x2(k1.x), sc01, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.y), d_u32_to_core_f16x2(k1.y), sc01, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.z), d_u32_to_core_f16x2(k1.z), sc01, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.w), d_u32_to_core_f16x2(k1.w), sc01, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.x), d_u32_to_core_f16x2(k0.x), sc10, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.y), d_u32_to_core_f16x2(k0.y), sc10, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.z), d_u32_to_core_f16x2(k0.z), sc10, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.w), d_u32_to_core_f16x2(k0.w), sc10, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.x), d_u32_to_core_f16x2(k1.x), sc11, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.y), d_u32_to_core_f16x2(k1.y), sc11, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.z), d_u32_to_core_f16x2(k1.z), sc11, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.w), d_u32_to_core_f16x2(k1.w), sc11, false);
            }
            sc00 += __shfl_xor(sc00, 1, 64); sc00 += __shfl_xor(sc00, 2, 64); sc00 += __shfl_xor(sc00, 4, 64);
            sc01 += __shfl_xor(sc01, 1, 64); sc01 += __shfl_xor(sc01, 2, 64); sc01 += __shfl_xor(sc01, 4, 64);
            sc10 += __shfl_xor(sc10, 1, 64); sc10 += __shfl_xor(sc10, 2, 64); sc10 += __shfl_xor(sc10, 4, 64);
            sc11 += __shfl_xor(sc11, 1, 64); sc11 += __shfl_xor(sc11, 2, 64); sc11 += __shfl_xor(sc11, 4, 64);
            if (ds == 0) {
                u.s_S[hq2 + 0][kq2 + 0] = sc00 * 0.0625f;
                u.s_S[hq2 + 0][kq2 + 1] = sc01 * 0.0625f;
                u.s_S[hq2 + 1][kq2 + 0] = sc10 * 0.0625f;
                u.s_S[hq2 + 1][kq2 + 1] = sc11 * 0.0625f;
            }
        }
        __syncthreads();

        // 5. 8-lane-per-head online softmax (first 128 threads cover 16 heads x 8 keys) -> u.s_P[16][4]
        {
            uint32_t p01 = 0u;
            float m_new = 0.0f, ls = 0.0f, alpha = 1.0f;
            const int sm_h = tid >> 3;
            const int sm_k = tid & 7;
            if (tid < 128) {
                const float sv = (kb + sm_k < valid) ? u.s_S[sm_h][sm_k] : -1e30f;
                float lm = sv;
                lm = fmaxf(lm, __shfl_xor(lm, 1, 64));
                lm = fmaxf(lm, __shfl_xor(lm, 2, 64));
                lm = fmaxf(lm, __shfl_xor(lm, 4, 64));

                const float m_old = s_m[sm_h];
                m_new = fmaxf(m_old, lm);
                alpha = __expf(m_old - m_new);

                const float e0 = (kb + sm_k < valid) ? __expf(sv - m_new) : 0.0f;
                const float e1 = __shfl_xor(e0, 1, 64);
                ls = e0;
                ls += __shfl_xor(ls, 1, 64);
                ls += __shfl_xor(ls, 2, 64);
                ls += __shfl_xor(ls, 4, 64);
                p01 = d_pack_f32x2_to_f16x2_u32(e0, e1);
            }
            __syncthreads();
            if (tid < 128) {
                if ((sm_k & 1) == 0) u.s_P[sm_h][sm_k >> 1] = p01;
                if (sm_k == 0) {
                    s_m[sm_h]     = m_new;
                    s_l[sm_h]     = fmaf(s_l[sm_h], alpha, ls);
                    s_alpha[sm_h] = alpha;
                }
            }
        }
        __syncthreads();

        // 6. Rescale o_acc[4][8] + 0-Bank-Conflict uint4 (ds_read_b128) 4x8 Register-Tiled PV
        {
            const int h_base = (tid >> 6) * 4; // 0, 4, 8, 12
            const int lane   = tid & 63;       // 0..63 -> dims [lane*8 .. lane*8+7]
            if (kb > 0) {
                #pragma unroll
                for (int hi = 0; hi < 4; ++hi) {
                    const float a = s_alpha[h_base + hi];
                    #pragma unroll
                    for (int di = 0; di < 8; ++di) o_acc[hi][di] *= a;
                }
            }
            #pragma unroll 1
            for (int kp = 0; kp < 4; ++kp) {
                const int k0 = kp * 2;
                const uint4 a = s_KV[k0][lane];
                const uint4 b = s_KV[k0 + 1][lane];
                const core_f16x2_t v0 = d_u32_to_core_f16x2((a.x & 0xFFFFu) | (b.x << 16));
                const core_f16x2_t v1 = d_u32_to_core_f16x2((a.x >> 16) | (b.x & 0xFFFF0000u));
                const core_f16x2_t v2 = d_u32_to_core_f16x2((a.y & 0xFFFFu) | (b.y << 16));
                const core_f16x2_t v3 = d_u32_to_core_f16x2((a.y >> 16) | (b.y & 0xFFFF0000u));
                const core_f16x2_t v4 = d_u32_to_core_f16x2((a.z & 0xFFFFu) | (b.z << 16));
                const core_f16x2_t v5 = d_u32_to_core_f16x2((a.z >> 16) | (b.z & 0xFFFF0000u));
                const core_f16x2_t v6 = d_u32_to_core_f16x2((a.w & 0xFFFFu) | (b.w << 16));
                const core_f16x2_t v7 = d_u32_to_core_f16x2((a.w >> 16) | (b.w & 0xFFFF0000u));

                #pragma unroll
                for (int hi = 0; hi < 4; ++hi) {
                    const core_f16x2_t pv = d_u32_to_core_f16x2(u.s_P[h_base + hi][kp]);
                    o_acc[hi][0] = __builtin_amdgcn_fdot2(pv, v0, o_acc[hi][0], false);
                    o_acc[hi][1] = __builtin_amdgcn_fdot2(pv, v1, o_acc[hi][1], false);
                    o_acc[hi][2] = __builtin_amdgcn_fdot2(pv, v2, o_acc[hi][2], false);
                    o_acc[hi][3] = __builtin_amdgcn_fdot2(pv, v3, o_acc[hi][3], false);
                    o_acc[hi][4] = __builtin_amdgcn_fdot2(pv, v4, o_acc[hi][4], false);
                    o_acc[hi][5] = __builtin_amdgcn_fdot2(pv, v5, o_acc[hi][5], false);
                    o_acc[hi][6] = __builtin_amdgcn_fdot2(pv, v6, o_acc[hi][6], false);
                    o_acc[hi][7] = __builtin_amdgcn_fdot2(pv, v7, o_acc[hi][7], false);
                }
            }
        }
        __syncthreads();
    }

    // 7. Finalize & write FP16 directly via 128-bit coalesced uint4 stores (512 dims)
    const int h_base = (tid >> 6) * 4;
    const int lane   = tid & 63;
    #pragma unroll
    for (int hi = 0; hi < 4; ++hi) {
        const float inv_l = 1.0f / s_l[h_base + hi];
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(
            o_lat_f16 + (size_t(m) * 16u + size_t(h_base + hi)) * 512u);
        dst[lane] = make_uint4(
            d_pack_f32x2_to_f16x2_u32(o_acc[hi][0] * inv_l, o_acc[hi][1] * inv_l),
            d_pack_f32x2_to_f16x2_u32(o_acc[hi][2] * inv_l, o_acc[hi][3] * inv_l),
            d_pack_f32x2_to_f16x2_u32(o_acc[hi][4] * inv_l, o_acc[hi][5] * inv_l),
            d_pack_f32x2_to_f16x2_u32(o_acc[hi][6] * inv_l, o_acc[hi][7] * inv_l));
    }
}

} // namespace

void launch_dsa_copy_qrope_f16(
    const uint16_t* d_q_bf16, uint16_t* d_q_abs_f16,
    int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_copy_qrope_f16, dim3(uint32_t(M * n_heads)), dim3(64), 0, stream,
                       d_q_bf16, d_q_abs_f16);
}

void launch_dsa_dense_finalize_sink_f16(
    const float* d_o_tot_f32, const float* d_m_tot_f32, const float* d_l_tot_f32,
    const float* d_attn_sink, uint16_t* d_out_f16,
    int B, int n_heads, hipStream_t stream) {
    if (B <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_dense_finalize_sink_f16, dim3(uint32_t(B * n_heads)), dim3(256), 0, stream,
                       d_o_tot_f32, d_m_tot_f32, d_l_tot_f32, d_attn_sink, d_out_f16, n_heads);
}

void launch_dsa_vha_lowrank_lrga_f16_io(
    const uint16_t* d_o_loc_f16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_bf16,
    uint16_t* d_o_out_f16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_vha_lowrank_lrga_f16_io, dim3(1u, uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_f16, d_p_all_f16, d_vha_q_16x16, d_gate_bf16, d_o_out_f16);
}

void launch_dsa_mark_selected_keys(
    const int32_t* d_sel_idx, const int32_t* d_sel_cnts,
    int32_t* d_map, int n_sp, int stride_sel, int total_kv, hipStream_t stream) {
    if (n_sp <= 0 || total_kv <= 0) return;
    // k_dsa_mark_selected_keys ORs 1s into d_map and k_dsa_compact_unique_map reads
    // `map[kpos] > 0`, so the caller must not have to remember to clear it: a stale
    // map silently turns into phantom unique keys (and a stale uniq_cnt corrupts the
    // compaction prefix sum). Zero both producers' state on the same stream.
    (void)hipMemsetAsync(d_map, 0, size_t(total_kv) * sizeof(int32_t), stream);
    hipLaunchKernelGGL(k_dsa_mark_selected_keys, dim3(uint32_t(n_sp)), dim3(256), 0, stream,
                       d_sel_idx, d_sel_cnts, d_map, stride_sel, total_kv);
}

void launch_dsa_compact_unique_map(
    int32_t* d_map, int32_t* d_uniq_kpos, int32_t* d_uniq_cnt,
    int total_kv, hipStream_t stream) {
    if (total_kv <= 0) return;
    (void)hipMemsetAsync(d_uniq_cnt, 0, sizeof(int32_t), stream);
    hipLaunchKernelGGL(k_dsa_compact_unique_map, dim3(uint32_t((total_kv + 255) / 256)), dim3(256), 0, stream,
                       d_map, d_uniq_kpos, d_uniq_cnt, total_kv);
}

void launch_dsa_gather_unique_kv_f16(
    const uint16_t* d_kv_mla_bf16, const int32_t* d_uniq_kpos,
    uint16_t* d_kv_uniq_f16, int n_uniq, hipStream_t stream) {
    if (n_uniq <= 0) return;
    hipLaunchKernelGGL(k_dsa_gather_unique_kv_f16, dim3(uint32_t(n_uniq)), dim3(64), 0, stream,
                       d_kv_mla_bf16, d_uniq_kpos, d_kv_uniq_f16);
}

void launch_dsa_remap_sel_idx(
    const int32_t* d_sel_idx, const int32_t* d_sel_cnts, const int32_t* d_map,
    void* d_sel_remapped, int n_sp, int stride_sel, bool use_u16, hipStream_t stream) {
    if (n_sp <= 0) return;
    if (use_u16) {
        hipLaunchKernelGGL(((k_dsa_remap_sel_idx<true>)), dim3(uint32_t(n_sp)), dim3(256), 0, stream,
                           d_sel_idx, d_sel_cnts, d_map, d_sel_remapped, stride_sel);
    } else {
        hipLaunchKernelGGL(((k_dsa_remap_sel_idx<false>)), dim3(uint32_t(n_sp)), dim3(256), 0, stream,
                           d_sel_idx, d_sel_cnts, d_map, d_sel_remapped, stride_sel);
    }
}

void launch_dsa_sparse_mqa_tp4_h16(
    const uint16_t* d_q_abs_f16, const uint16_t* d_kv_mla, const float* d_attn_sink_16,
    const void* d_sel_idx, const int32_t* d_sel_cnts, uint16_t* d_o_lat_f16,
    int n_sp, int stride_sel, bool kv_is_f16, bool idx_is_u16, hipStream_t stream) {
    if (n_sp <= 0) return;
    const dim3 grid(static_cast<uint32_t>(n_sp));
    const dim3 block(256u);
    if (kv_is_f16) {
        if (idx_is_u16) {
            hipLaunchKernelGGL(((k_dsa_sparse_mqa_tp4_h16<true, true>)), grid, block, 0, stream,
                               d_q_abs_f16, d_kv_mla, d_attn_sink_16, d_sel_idx, d_sel_cnts, d_o_lat_f16, stride_sel);
        } else {
            hipLaunchKernelGGL(((k_dsa_sparse_mqa_tp4_h16<true, false>)), grid, block, 0, stream,
                               d_q_abs_f16, d_kv_mla, d_attn_sink_16, d_sel_idx, d_sel_cnts, d_o_lat_f16, stride_sel);
        }
    } else {
        if (idx_is_u16) {
            hipLaunchKernelGGL(((k_dsa_sparse_mqa_tp4_h16<false, true>)), grid, block, 0, stream,
                               d_q_abs_f16, d_kv_mla, d_attn_sink_16, d_sel_idx, d_sel_cnts, d_o_lat_f16, stride_sel);
        } else {
            hipLaunchKernelGGL(((k_dsa_sparse_mqa_tp4_h16<false, false>)), grid, block, 0, stream,
                               d_q_abs_f16, d_kv_mla, d_attn_sink_16, d_sel_idx, d_sel_cnts, d_o_lat_f16, stride_sel);
        }
    }
}


// ---- Wave-10 C5b: single-pass K_tile = 16,384 FlashIndexer (0 scratch, 0 slc) ----
// Score slab writer: 8-head weighted ReLU sum of d_y16 [4096, b_eff*8] into the
// tiled slab d_tile_scores[bi * kTileK + k_off_in_tile + tok]. Keys at or beyond
// n_hist are written as -FLT_MAX so causal-masked tails never reach the top-2048.
__global__ __launch_bounds__(256, 4)
void k_flashindexer_relu_wsum_tiled_f16_in(
    const uint16_t* __restrict__ dots_f16,
    const float* __restrict__ w_sel_f32,
    float* __restrict__ tile_scores,
    int32_t pos0, int t0, int k_blk, int k_off_in_tile, int k_cnt, int kTileK)
{
    const int k  = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int bi = int(blockIdx.y);
    if (k >= k_cnt) return;
    const int t  = t0 + bi;
    const int n_hist = (pos0 + t) - 127;
    const int tok  = k_off_in_tile + k;
    float total = -FLT_MAX;
    if ((k_blk + tok) < n_hist) {
        constexpr float kWScale = 0.011048543456039806f;
        const float* __restrict__ wt = w_sel_f32 + size_t(t) * 8u;
        const size_t base = size_t(bi) * 8u * size_t(k_cnt) + size_t(k);
        total = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float dp = d_f16_to_f32_core(dots_f16[base + size_t(i) * size_t(k_cnt)]);
            if (dp > 0.0f) total = fmaf(wt[i] * kWScale, dp, total);
        }
    }
    tile_scores[size_t(bi) * size_t(kTileK) + size_t(tok)] = total;
}

// Wave-11 C4: Streaming top-2048 merge with exact min_u floor bypass + parallel 256-thread
// reverse prefix threshold scan (0 scratch, 0 slc, LDS=24.6 KiB -> 2 WGs/CU, set_mism==0 guaranteed!).
// Because every running key has virtual index < 2048 and every new key has virtual index >= 2048,
// any new key with ord(score) <= min_u is strictly outranked by all 2048 running keys under (score DESC, idx ASC).
__global__ __launch_bounds__(256, 2)
void k_dsa_topk2048_streaming_merge(
    const float* __restrict__ tile_scores,
    const float* __restrict__ best_val_in,
    const int32_t* __restrict__ best_idx_in,
    float* __restrict__ best_val_out,
    int32_t* __restrict__ best_idx_out,
    int32_t pos0, int t0, int k_blk, int kTileK)
{
    __shared__ uint32_t s_hist[2048];
    __shared__ float    s_cval[2048];
    __shared__ int32_t  s_cidx[2048];
    __shared__ uint32_t s_meta[8];
    __shared__ uint32_t s_wgt[4];
    __shared__ uint32_t s_weq[4];

    const int bi   = int(blockIdx.x);
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const int t   = t0 + bi;
    const int pos = pos0 + t;
    const int n_hist = pos - 127;
    if (n_hist <= 2048) return; // finalizer emits 0..pos

    const int n_new = std::min(kTileK, std::max(0, n_hist - k_blk));
    const float* __restrict__ bv_in = best_val_in + size_t(bi) * 2048u;
    const int32_t* __restrict__ bi_in = best_idx_in + size_t(bi) * 2048u;
    float* __restrict__ bv_out = best_val_out + size_t(bi) * 2048u;
    int32_t* __restrict__ bi_out = best_idx_out + size_t(bi) * 2048u;
    const float* __restrict__ sc = tile_scores + size_t(bi) * size_t(kTileK);

    if (n_new <= 0) { // pass through unchanged
        for (int j = tid; j < 2048; j += 256) { bv_out[j] = bv_in[j]; bi_out[j] = bi_in[j]; }
        return;
    }

    const bool has_running = (k_blk > 0);
    const uint64_t below = (1ULL << lane) - 1ULL;
    bool use_lds_cands = false;
    int nv = has_running ? (2048 + n_new) : n_new;

    // Exact min_u floor bypass on subsequent tiles (k_blk > 0):
    // 1) Find min_u = min_{0<=j<2048} ord(bv_in[j]) in 8 iterations/thread.
    // 2) Scan sc[0..n_new) ONCE, compacting strictly winning keys (ord > min_u) into s_cval/s_cidx.
    //    - cand_cnt == 0: 100% skip! Copy running top-2048 and return immediately.
    //    - 0 < cand_cnt <= 2048: run exact radix select over only 2048 + cand_cnt elements (new keys in LDS!).
    //    - cand_cnt > 2048: seamless fallback to exact radix select over full 2048 + n_new.
    if (has_running) {
        uint32_t my_min = 0xFFFFFFFFu;
        #pragma unroll
        for (int j = tid; j < 2048; j += 256) {
            const uint32_t u = d_f32_to_ord(bv_in[j]);
            if (u < my_min) my_min = u;
        }
        #pragma unroll
        for (int off = 32; off >= 1; off >>= 1) {
            const uint32_t ov = __shfl_xor(my_min, off, 64);
            if (ov < my_min) my_min = ov;
        }
        if (lane == 0) s_wgt[warp] = my_min;
        __syncthreads();
        uint32_t min_u = s_wgt[0];
        if (s_wgt[1] < min_u) min_u = s_wgt[1];
        if (s_wgt[2] < min_u) min_u = s_wgt[2];
        if (s_wgt[3] < min_u) min_u = s_wgt[3];

        uint32_t cand_so_far = 0u;
        for (int base = 0; base < n_new; base += 256) {
            const int j = base + tid;
            float sv = -FLT_MAX;
            bool keep = false;
            if (j < n_new) {
                sv = sc[j];
                keep = (d_f32_to_ord(sv) > min_u);
            }
            const uint64_t m_k = __ballot(keep);
            const uint32_t w_k = uint32_t(__popcll(m_k));
            const uint32_t l_k = uint32_t(__popcll(m_k & below));
            if (lane == 0) s_wgt[warp] = w_k;
            __syncthreads();
            uint32_t w_off = 0u;
            if (warp >= 1) w_off += s_wgt[0];
            if (warp >= 2) w_off += s_wgt[1];
            if (warp >= 3) w_off += s_wgt[2];
            const uint32_t strip_tot = s_wgt[0] + s_wgt[1] + s_wgt[2] + s_wgt[3];
            if (keep) {
                const uint32_t ci = cand_so_far + w_off + l_k;
                if (ci < 2048u) {
                    s_cval[ci] = sv;
                    s_cidx[ci] = k_blk + j;
                }
            }
            cand_so_far += strip_tot;
            __syncthreads();
            if (cand_so_far > 2048u) break;
        }
        if (cand_so_far == 0u) {
            #pragma unroll
            for (int j = tid; j < 2048; j += 256) { bv_out[j] = bv_in[j]; bi_out[j] = bi_in[j]; }
            return;
        }
        if (cand_so_far <= 2048u) {
            use_lds_cands = true;
            nv = 2048 + int(cand_so_far);
        }
    }

    auto score_of = [&](int v) -> float {
        if (has_running && v < 2048) return bv_in[v];
        const int idx = v - (has_running ? 2048 : 0);
        return use_lds_cands ? s_cval[idx] : sc[idx];
    };
    auto key_of = [&](int v) -> int32_t {
        if (has_running && v < 2048) return bi_in[v];
        const int idx = v - (has_running ? 2048 : 0);
        return use_lds_cands ? s_cidx[idx] : (k_blk + idx);
    };

    // Parallel 256-thread descending prefix threshold finder over s_hist[0..nbins-1]
    auto find_bin_desc = [&](int nbins, uint32_t target, int meta_b, int meta_need) {
        const int per  = nbins >> 8; // 8 for 2048, 4 for 1024
        const int b_hi = (nbins - 1) - tid * per;
        uint32_t my_sum = 0u;
        for (int s = 0; s < per; ++s) my_sum += s_hist[b_hi - s];
        uint32_t w_incl = my_sum;
        #pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            const uint32_t pv = __shfl_up(w_incl, off, 64);
            if (lane >= off) w_incl += pv;
        }
        if (lane == 63) s_wgt[warp] = w_incl;
        __syncthreads();
        uint32_t w_base = 0u;
        if (warp >= 1) w_base += s_wgt[0];
        if (warp >= 2) w_base += s_wgt[1];
        if (warp >= 3) w_base += s_wgt[2];
        const uint32_t incl = w_base + w_incl;
        const uint32_t excl = incl - my_sum;
        if (excl < target && incl >= target) {
            uint32_t above = excl;
            uint32_t b_win = uint32_t(b_hi);
            for (int s = 0; s < per; ++s) {
                const uint32_t c = s_hist[b_hi - s];
                if (above + c >= target) { b_win = uint32_t(b_hi - s); break; }
                above += c;
            }
            s_meta[meta_b]    = b_win;
            s_meta[meta_need] = target - above;
        }
        __syncthreads();
    };

    // Pass 1: 11-bit top bits
    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int v = tid; v < nv; v += 256) atomicAdd(&s_hist[d_f32_to_ord(score_of(v)) >> 21], 1u);
    __syncthreads();
    find_bin_desc(2048, 2048u, 0, 1);
    const uint32_t b1 = s_meta[0], need1 = s_meta[1];

    // Pass 2: 11-bit mid bits
    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int v = tid; v < nv; v += 256) {
        const uint32_t u = d_f32_to_ord(score_of(v));
        if ((u >> 21) == b1) atomicAdd(&s_hist[(u >> 10) & 2047u], 1u);
    }
    __syncthreads();
    find_bin_desc(2048, need1, 2, 3);
    const uint32_t p22 = (b1 << 11) | s_meta[2], need2 = s_meta[3];

    // Pass 3: 10-bit low bits
    for (int b = tid; b < 1024; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int v = tid; v < nv; v += 256) {
        const uint32_t u = d_f32_to_ord(score_of(v));
        if ((u >> 10) == p22) atomicAdd(&s_hist[u & 1023u], 1u);
    }
    __syncthreads();
    find_bin_desc(1024, need2, 4, 5);
    const uint32_t thresh_u = (p22 << 10) | s_meta[4], need_eq = s_meta[5];
    const uint32_t total_gt = 2048u - need_eq;

    // Pass 4: deterministic ascending-virtual-index emission (zero global atomics)
    uint32_t gt_so_far = 0u, eq_so_far = 0u;
    for (int base = 0; base < nv; base += 256) {
        const int v = base + tid;
        bool is_gt = false, is_eq = false;
        if (v < nv) {
            const uint32_t u = d_f32_to_ord(score_of(v));
            is_gt = (u > thresh_u);
            is_eq = (u == thresh_u);
        }
        const uint64_t m_gt = __ballot(is_gt);
        const uint32_t w_gt = uint32_t(__popcll(m_gt));
        const uint32_t l_gt = uint32_t(__popcll(m_gt & below));
        const uint64_t m_eq = __ballot(is_eq);
        const uint32_t w_eq = uint32_t(__popcll(m_eq));
        const uint32_t l_eq = uint32_t(__popcll(m_eq & below));
        if (lane == 0) { s_wgt[warp] = w_gt; s_weq[warp] = w_eq; }
        __syncthreads();
        uint32_t wg_off = 0u, we_off = 0u;
        if (warp >= 1) { wg_off += s_wgt[0]; we_off += s_weq[0]; }
        if (warp >= 2) { wg_off += s_wgt[1]; we_off += s_weq[1]; }
        if (warp >= 3) { wg_off += s_wgt[2]; we_off += s_weq[2]; }
        if (is_gt) {
            const uint32_t gi = gt_so_far + wg_off + l_gt;
            if (gi < total_gt) { bv_out[gi] = score_of(v); bi_out[gi] = key_of(v); }
        }
        if (is_eq) {
            const uint32_t ei = eq_so_far + we_off + l_eq;
            if (ei < need_eq) { bv_out[total_gt + ei] = score_of(v); bi_out[total_gt + ei] = key_of(v); }
        }
        gt_so_far += s_wgt[0] + s_wgt[1] + s_wgt[2] + s_wgt[3];
        eq_so_far += s_weq[0] + s_weq[1] + s_weq[2] + s_weq[3];
        __syncthreads();
    }
}

// Final selection indices + sliding-window tail for every sparse query row.
__global__ __launch_bounds__(256, 4)
void k_dsa_topk2048_streaming_finalize(
    const int32_t* __restrict__ best_idx_final,
    int32_t* __restrict__ sel_idx_MxS,
    int32_t* __restrict__ sel_cnts,
    int32_t pos0, int t0, int stride_sel)
{
    const int bi  = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int t   = t0 + bi;
    const int pos = pos0 + t;
    const int swa_start = (pos >= 127) ? (pos - 127) : 0;
    const int n_hist = swa_start;
    int32_t* __restrict__ dst = sel_idx_MxS + size_t(t) * size_t(stride_sel);

    if (n_hist <= 2048) {
        for (int p = tid; p <= pos; p += 256) dst[p] = p;
        if (tid == 0) sel_cnts[t] = pos + 1;
        return;
    }
    const int32_t* __restrict__ src = best_idx_final + size_t(bi) * 2048u;
    for (int j = tid; j < 2048; j += 256) dst[j] = src[j];
    const int swa_len = pos - swa_start + 1;
    for (int k = tid; k < swa_len; k += 256) dst[2048 + k] = swa_start + k;
    if (tid == 0) sel_cnts[t] = 2048 + swa_len;
}

void launch_flashindexer_relu_wsum_tiled_f16_in(
    const uint16_t* d_dots_f16, const float* d_w_sel_f32, float* d_tile_scores,
    int32_t pos0, int t0, int B_sc, int k_blk, int k_off_in_tile, int k_cnt, int kTileK,
    hipStream_t stream) {
    if (B_sc <= 0 || k_cnt <= 0) return;
    hipLaunchKernelGGL(k_flashindexer_relu_wsum_tiled_f16_in,
                       dim3(uint32_t((k_cnt + 255) / 256), uint32_t(B_sc)), dim3(256), 0, stream,
                       d_dots_f16, d_w_sel_f32, d_tile_scores, pos0, t0, k_blk, k_off_in_tile, k_cnt, kTileK);
}

void launch_dsa_topk2048_streaming_merge(
    const float* d_tile_scores, const float* d_best_val_in, const int32_t* d_best_idx_in,
    float* d_best_val_out, int32_t* d_best_idx_out,
    int32_t pos0, int t0, int n_sp, int k_blk, int kTileK, hipStream_t stream) {
    if (n_sp <= 0) return;
    hipLaunchKernelGGL(k_dsa_topk2048_streaming_merge, dim3(uint32_t(n_sp)), dim3(256), 0, stream,
                       d_tile_scores, d_best_val_in, d_best_idx_in,
                       d_best_val_out, d_best_idx_out, pos0, t0, k_blk, kTileK);
}

void launch_dsa_topk2048_streaming_finalize(
    const int32_t* d_best_idx_final, int32_t* d_sel_idx_MxS, int32_t* d_sel_cnts,
    int32_t pos0, int t0, int n_sp, int stride_sel, hipStream_t stream) {
    if (n_sp <= 0) return;
    hipLaunchKernelGGL(k_dsa_topk2048_streaming_finalize, dim3(uint32_t(n_sp)), dim3(256), 0, stream,
                       d_best_idx_final, d_sel_idx_MxS, d_sel_cnts, pos0, t0, stride_sel);
}


// ---- Wave-10 C6: DSpark launch fusion (iota pos, noise+pos broadcast, fused fdot2 Markov+MultiMax, packed sparse softmax) ----
__global__ __launch_bounds__(256, 8)
void k_fill_iota_i32(int32_t* __restrict__ out, int32_t start_val, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = start_val + i;
}

__global__ __launch_bounds__(256, 8)
void k_dspark_broadcast_noise_and_pos(
    const uint16_t* __restrict__ noise_emb,
    uint16_t* __restrict__ x_in,
    int32_t* __restrict__ pos_ids,
    int32_t pos0, int B, int H)
{
    const int bx  = int(blockIdx.x);
    const int by  = int(blockIdx.y); // 0 .. max(0, B - 2) -> target row (by + 1)
    const int tid = int(threadIdx.x);
    if (bx == 0 && by == 0) {
        for (int i = tid; i < B; i += 256) pos_ids[i] = pos0 + i;
    }
    const int dst_row = by + 1;
    if (dst_row < B) {
        const int h8  = H >> 3;
        const int idx = bx * 256 + tid;
        if (idx < h8) {
            const uint4* __restrict__ src8 = reinterpret_cast<const uint4*>(noise_emb);
            uint4* __restrict__ dst8 = reinterpret_cast<uint4*>(x_in + size_t(dst_row) * size_t(H));
            dst8[idx] = src8[idx];
        }
    }
}

// Fused DSpark markov_w2 ([V, R] FP16) * me ([R] BF16->FP16 in LDS) + base_logits + exact 4-segment MultiMax.
// 8 rows/block (32 threads/row -> coalesced 512 B/row reads at R=256), hardware v_dot2_f32_f16, 0 scratch, 0 slc.
__global__ __launch_bounds__(256, 4)
void k_dspark_markov_bias_multimax_f32(
    const uint16_t* __restrict__ me_bf16,
    const uint16_t* __restrict__ w2_VxR_f16,
    const float*    __restrict__ base_logits,
    const float*    __restrict__ mm_r,
    const float*    __restrict__ mm_t,
    float*          __restrict__ out_logits,
    int V, int R)
{
    __shared__ uint4 s_me_f16[128]; // up to R=1024 FP16 values (2 KiB LDS; R=256 uses 512 B)
    const int tid  = int(threadIdx.x);
    const int R8   = R >> 3;
    const bool fast_u4 = ((R & 7) == 0 && R8 <= 128);

    if (fast_u4) {
        const uint4* __restrict__ me_u4 = reinterpret_cast<const uint4*>(me_bf16);
        for (int j = tid; j < R8; j += 256) {
            s_me_f16[j] = d_core_bf16_u4_to_f16_u4(me_u4[j]);
        }
        __syncthreads();
    }

    const int sub  = tid >> 5; // 0..7 subwarps per block
    const int lane = tid & 31; // 0..31 lanes per row
    constexpr int kRowsPerSub = 4;
    const int v_base = int(blockIdx.x) * (8 * kRowsPerSub) + sub;

    #pragma unroll 1
    for (int ri = 0; ri < kRowsPerSub; ++ri) {
        const int v = v_base + ri * 8;
        if (v >= V) break;

        const uint16_t* __restrict__ w_row = w2_VxR_f16 + size_t(v) * size_t(R);
        float acc = 0.0f;
        if (fast_u4) {
            const uint4* __restrict__ w_u4 = reinterpret_cast<const uint4*>(w_row);
            for (int j = lane; j < R8; j += 32) {
                acc = d_core_dot8_f16_u4_fdot2(s_me_f16[j], w_u4[j], acc);
            }
        } else {
            for (int r = lane; r < R; r += 32) {
                const float mf = d_f16_to_f32_core(d_f32_to_f16_core(d_bf16_to_f32(me_bf16[r])));
                acc = fmaf(d_f16_to_f32_core(w_row[r]), mf, acc);
            }
        }
        acc = d_core_reduce32_dpp(acc);

        if (lane == 0) {
            const float bx = base_logits[v];
            if (isinf(bx) && bx < 0.0f) {
                out_logits[v] = bx;
            } else {
                float x = bx + acc;
                if (mm_r != nullptr && mm_t != nullptr) {
                    const float r0x = mm_r[0] - x; const float a = (r0x <= 0.0f) ? 0.0f : r0x;
                    const float xr1 = x - mm_r[1]; const float b = (xr1 <= 0.0f) ? 0.0f : xr1;
                    const float r2x = mm_r[2] - x; const float c = (r2x <= 0.0f) ? 0.0f : r2x;
                    const float xr3 = x - mm_r[3]; const float d = (xr3 <= 0.0f) ? 0.0f : xr3;
                    float y = __fadd_rn(x, __fmul_rn(mm_t[0], a));
                    y = __fadd_rn(y, __fmul_rn(mm_t[1], b));
                    y = __fadd_rn(y, __fmul_rn(mm_t[2], __fmul_rn(c, c)));
                    y = __fadd_rn(y, __fmul_rn(mm_t[3], __fmul_rn(d, d)));
                    x = y;
                }
                out_logits[v] = x;
            }
        }
    }
}

__global__ __launch_bounds__(64, 4)
void k_softmax_max_p2_packed(const float* __restrict__ part64, DsparkSparsePacketHeader* __restrict__ hdr) {
    const int lane = int(threadIdx.x);
    float mx = part64[lane];
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
    if (lane == 0) {
        hdr->stats2[0] = mx;
        hdr->stats2[1] = 0.0f;
        hdr->cand_cnt  = 0;
        hdr->pad       = 0;
    }
}

__global__ __launch_bounds__(256, 4)
void k_softmax_sparse_extract_p3_packed(
    const float* __restrict__ logits, int V, float inv_t, float temperature,
    DsparkSparsePacketHeader* __restrict__ hdr, float* __restrict__ part64,
    DsparkCandPair* __restrict__ cands, int max_cands)
{
    __shared__ float s_w[4];
    const float max_l    = hdr->stats2[0];
    const float cut_sum  = max_l - 22.0f * temperature;
    const float cut_cand = max_l - 16.0f * temperature;
    const int bid = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int stride = int(gridDim.x) * 256;

    float lsum = 0.0f;
    for (int i = bid * 256 + tid; i < V; i += stride) {
        const float v = logits[i];
        if (v > cut_sum && isfinite(v)) {
            const float e = __expf((v - max_l) * inv_t);
            lsum += e;
            if (v > cut_cand) {
                const int32_t slot = atomicAdd(&hdr->cand_cnt, 1);
                if (slot < max_cands) {
                    cands[slot].idx = i;
                    cands[slot].exp = e;
                }
            }
        }
    }
    lsum = d_warp_sum64(lsum);
    if (lane == 0) s_w[warp] = lsum;
    __syncthreads();
    if (tid == 0) part64[bid] = s_w[0] + s_w[1] + s_w[2] + s_w[3];
}

__global__ __launch_bounds__(64, 4)
void k_softmax_sum_reduce_p4_packed(const float* __restrict__ part64, DsparkSparsePacketHeader* __restrict__ hdr) {
    const int lane = int(threadIdx.x);
    float s = d_warp_sum64(part64[lane]);
    if (lane == 0) hdr->stats2[1] = s;
}

void launch_fill_iota_i32(int32_t* d_out, int32_t start_val, int n, hipStream_t stream) {
    if (n <= 0) return;
    hipLaunchKernelGGL(k_fill_iota_i32, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, stream,
                       d_out, start_val, n);
}

void launch_dspark_broadcast_noise_and_pos(
    const uint16_t* d_noise_emb, uint16_t* d_x_in, int32_t* d_pos_ids,
    int32_t pos0, int B, int H, hipStream_t stream) {
    if (B <= 0 || H <= 0) return;
    const int h8 = H >> 3;
    const uint32_t gx = uint32_t((h8 + 255) / 256);
    const uint32_t gy = uint32_t(B > 1 ? (B - 1) : 1);
    hipLaunchKernelGGL(k_dspark_broadcast_noise_and_pos, dim3(gx, gy), dim3(256), 0, stream,
                       d_noise_emb, d_x_in, d_pos_ids, pos0, B, H);
}

void launch_dspark_markov_bias_multimax_f32(
    const uint16_t* d_me_bf16, const uint16_t* d_w2_VxR_f16,
    const float* d_base_logits_f32,
    const float* d_mm_ranges4, const float* d_mm_ts4,
    float* d_out_logits_f32,
    int V, int R, hipStream_t stream) {
    if (V <= 0 || R <= 0) return;
    const uint32_t gx = uint32_t((V + 31) / 32);
    hipLaunchKernelGGL(k_dspark_markov_bias_multimax_f32, dim3(gx), dim3(256), 0, stream,
                       d_me_bf16, d_w2_VxR_f16, d_base_logits_f32, d_mm_ranges4, d_mm_ts4,
                       d_out_logits_f32, V, R);
}

void launch_gpu_softmax_sparse_top_candidates_packed(
    const float* d_logits_f32, int V, float temperature,
    float* d_part64_f32, DsparkSparsePacketHeader* d_hdr,
    DsparkCandPair* d_cands, int max_cands, hipStream_t stream) {
    if (V <= 0 || temperature <= 1e-6f) return;
    const float inv_t = 1.0f / temperature;
    hipLaunchKernelGGL(k_softmax_max_p1, dim3(64), dim3(256), 0, stream, d_logits_f32, d_part64_f32, V);
    hipLaunchKernelGGL(k_softmax_max_p2_packed, dim3(1), dim3(64), 0, stream, d_part64_f32, d_hdr);
    hipLaunchKernelGGL(k_softmax_sparse_extract_p3_packed, dim3(64), dim3(256), 0, stream,
                       d_logits_f32, V, inv_t, temperature, d_hdr, d_part64_f32, d_cands, max_cands);
    hipLaunchKernelGGL(k_softmax_sum_reduce_p4_packed, dim3(1), dim3(64), 0, stream, d_part64_f32, d_hdr);
}


// ---- Wave-11 C2: single-launch TP4 HCA SWA-128 + compressed merge (0 scratch, 0 slc) ----
// 16 heads x 32 keys x 512 dims v_dot2_f32_f16, Q permanently LDS-resident, online softmax,
// 0-bank-conflict lane+0/64/128/192 PV layout, exact per-head merge with (comp_o,comp_m,comp_l)
// and single attn_sink[h] in the epilogue.
__global__ __launch_bounds__(256, 1)
void k_hca_tp4_swa128_merge_f16(
    const uint16_t* __restrict__ q_f16,
    const uint16_t* __restrict__ kv_f16,
    const float* __restrict__ comp_o_f32,
    const float* __restrict__ comp_m_f32,
    const float* __restrict__ comp_l_f32,
    const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_f16,
    int32_t pos0, int32_t base_pos)
{
    __shared__ uint32_t s_KV[32][257];
    __shared__ uint32_t s_Q[16][257];
    __shared__ union { float s_S[16][32]; uint32_t s_P[16][16]; } u;
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];

    const int m   = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int pos = pos0 + m;
    const int n_keys = (pos + 1 < 128) ? (pos + 1) : 128;
    const int base_k_slot = (pos - n_keys + 1) - base_pos;

    {   // Q [16][512] FP16 -> s_Q (16 lanes x 8 u64 per head = 128 u64 = 256 u32)
        const int q_h = tid >> 4;
        const int q_lane = tid & 15;
        const uint64_t* __restrict__ q64 = reinterpret_cast<const uint64_t*>(
            q_f16 + (size_t(m) * 16u + size_t(q_h)) * 512u);
        #pragma unroll 1
        for (int c = 0; c < 8; ++c) {
            const int u64_i = q_lane + c * 16;
            const uint64_t v64 = q64[u64_i];
            s_Q[q_h][u64_i * 2]     = uint32_t(v64 & 0xFFFFFFFFull);
            s_Q[q_h][u64_i * 2 + 1] = uint32_t(v64 >> 32);
        }
    }
    if (tid < 16) {
        s_m[tid] = -1e30f;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    float o_acc[4][8];
    #pragma unroll
    for (int hi = 0; hi < 4; ++hi) {
        #pragma unroll
        for (int di = 0; di < 8; ++di) o_acc[hi][di] = 0.0f;
    }

    #pragma unroll 1
    for (int kb = 0; kb < n_keys; kb += 32) {
        const int rem = n_keys - kb;
        const int cnt = (rem < 32) ? rem : 32;
        {   // K[32][512] FP16 -> s_KV (8 lanes x 16 u64 per key = 128 u64 = 256 u32)
            const int k_row = tid >> 3;
            const int k_lane = tid & 7;
            if (k_row < cnt) {
                const uint64_t* __restrict__ k_src = reinterpret_cast<const uint64_t*>(
                    kv_f16 + size_t(base_k_slot + kb + k_row) * 512u);
                #pragma unroll 1
                for (int c = 0; c < 16; ++c) {
                    const int u64_i = k_lane + c * 8;
                    const uint64_t v64 = k_src[u64_i];
                    s_KV[k_row][u64_i * 2]     = uint32_t(v64 & 0xFFFFFFFFull);
                    s_KV[k_row][u64_i * 2 + 1] = uint32_t(v64 >> 32);
                }
            } else {
                #pragma unroll 1
                for (int c = 0; c < 16; ++c) {
                    const int u64_i = k_lane + c * 8;
                    s_KV[k_row][u64_i * 2]     = 0u;
                    s_KV[k_row][u64_i * 2 + 1] = 0u;
                }
            }
        }
        __syncthreads();

        // 4x4 Register-Tiled QK over all 512 dims (8 D-splits x 32 half2)
        float sc[4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) sc[a][b] = 0.0f;
        }
        {
            const int ds = tid & 7;
            const int tile = tid >> 3;
            const int hq4 = (tile & 3) * 4;
            const int kq4 = (tile >> 2) * 4;
            #pragma unroll 1
            for (int j = 0; j < 32; ++j) {
                const int j_idx = ds + j * 8;
                const core_f16x2_t k0 = d_u32_to_core_f16x2(s_KV[kq4 + 0][j_idx]);
                const core_f16x2_t k1 = d_u32_to_core_f16x2(s_KV[kq4 + 1][j_idx]);
                const core_f16x2_t k2 = d_u32_to_core_f16x2(s_KV[kq4 + 2][j_idx]);
                const core_f16x2_t k3 = d_u32_to_core_f16x2(s_KV[kq4 + 3][j_idx]);
                #pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const core_f16x2_t qa = d_u32_to_core_f16x2(s_Q[hq4 + a][j_idx]);
                    sc[a][0] = __builtin_amdgcn_fdot2(qa, k0, sc[a][0], false);
                    sc[a][1] = __builtin_amdgcn_fdot2(qa, k1, sc[a][1], false);
                    sc[a][2] = __builtin_amdgcn_fdot2(qa, k2, sc[a][2], false);
                    sc[a][3] = __builtin_amdgcn_fdot2(qa, k3, sc[a][3], false);
                }
            }
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float v = sc[a][b];
                    v += __shfl_xor(v, 1, 64);
                    v += __shfl_xor(v, 2, 64);
                    v += __shfl_xor(v, 4, 64);
                    if (ds == 0) u.s_S[hq4 + a][kq4 + b] = v * 0.04419417382415922f;
                }
            }
        }
        __syncthreads();

        // 16-lane-per-head online softmax (strict k0 < cnt within current tile!)
        {
            const int sm_h = tid >> 4;
            const int sm_sub = tid & 15;
            const int k0 = sm_sub * 2;
            const float sv0 = (k0     < cnt) ? u.s_S[sm_h][k0]     : -1e30f;
            const float sv1 = (k0 + 1 < cnt) ? u.s_S[sm_h][k0 + 1] : -1e30f;
            float lm = fmaxf(sv0, sv1);
            lm = fmaxf(lm, __shfl_xor(lm, 1, 64));
            lm = fmaxf(lm, __shfl_xor(lm, 2, 64));
            lm = fmaxf(lm, __shfl_xor(lm, 4, 64));
            lm = fmaxf(lm, __shfl_xor(lm, 8, 64));
            const float m_old = s_m[sm_h];
            const float m_new = fmaxf(m_old, lm);
            const float alpha = (kb > 0) ? __expf(m_old - m_new) : 0.0f;
            const float e0 = (k0     < cnt) ? __expf(sv0 - m_new) : 0.0f;
            const float e1 = (k0 + 1 < cnt) ? __expf(sv1 - m_new) : 0.0f;
            float ls = e0 + e1;
            ls += __shfl_xor(ls, 1, 64);
            ls += __shfl_xor(ls, 2, 64);
            ls += __shfl_xor(ls, 4, 64);
            ls += __shfl_xor(ls, 8, 64);
            const uint32_t p01 = d_pack_f32x2_to_f16x2_u32(e0, e1);
            __syncthreads();
            u.s_P[sm_h][sm_sub] = p01;
            if (sm_sub == 0) {
                s_m[sm_h] = m_new;
                s_l[sm_h] = fmaf(s_l[sm_h], alpha, ls);
                s_alpha[sm_h] = alpha;
            }
        }
        __syncthreads();

        // 0-bank-conflict 4x8 Register-Tiled PV (lane + 0/64/128/192) via v_dot2_f32_f16
        {
            const int h_base = (tid >> 6) * 4;
            const int lane = tid & 63;
            if (kb > 0) {
                #pragma unroll
                for (int hi = 0; hi < 4; ++hi) {
                    const float a = s_alpha[h_base + hi];
                    #pragma unroll
                    for (int di = 0; di < 8; ++di) o_acc[hi][di] *= a;
                }
            }
            #pragma unroll 1
            for (int kp = 0; kp < 16; ++kp) {
                const int k0 = kp * 2;
                const uint32_t a0 = s_KV[k0][lane + 0];
                const uint32_t b0 = s_KV[k0 + 1][lane + 0];
                const core_f16x2_t v0 = d_u32_to_core_f16x2((a0 & 0xFFFFu) | (b0 << 16));
                const core_f16x2_t v1 = d_u32_to_core_f16x2((a0 >> 16) | (b0 & 0xFFFF0000u));

                const uint32_t a1 = s_KV[k0][lane + 64];
                const uint32_t b1 = s_KV[k0 + 1][lane + 64];
                const core_f16x2_t v2 = d_u32_to_core_f16x2((a1 & 0xFFFFu) | (b1 << 16));
                const core_f16x2_t v3 = d_u32_to_core_f16x2((a1 >> 16) | (b1 & 0xFFFF0000u));

                const uint32_t a2 = s_KV[k0][lane + 128];
                const uint32_t b2 = s_KV[k0 + 1][lane + 128];
                const core_f16x2_t v4 = d_u32_to_core_f16x2((a2 & 0xFFFFu) | (b2 << 16));
                const core_f16x2_t v5 = d_u32_to_core_f16x2((a2 >> 16) | (b2 & 0xFFFF0000u));

                const uint32_t a3 = s_KV[k0][lane + 192];
                const uint32_t b3 = s_KV[k0 + 1][lane + 192];
                const core_f16x2_t v6 = d_u32_to_core_f16x2((a3 & 0xFFFFu) | (b3 << 16));
                const core_f16x2_t v7 = d_u32_to_core_f16x2((a3 >> 16) | (b3 & 0xFFFF0000u));

                #pragma unroll
                for (int hi = 0; hi < 4; ++hi) {
                    const core_f16x2_t pv = d_u32_to_core_f16x2(u.s_P[h_base + hi][kp]);
                    o_acc[hi][0] = __builtin_amdgcn_fdot2(pv, v0, o_acc[hi][0], false);
                    o_acc[hi][1] = __builtin_amdgcn_fdot2(pv, v1, o_acc[hi][1], false);
                    o_acc[hi][2] = __builtin_amdgcn_fdot2(pv, v2, o_acc[hi][2], false);
                    o_acc[hi][3] = __builtin_amdgcn_fdot2(pv, v3, o_acc[hi][3], false);
                    o_acc[hi][4] = __builtin_amdgcn_fdot2(pv, v4, o_acc[hi][4], false);
                    o_acc[hi][5] = __builtin_amdgcn_fdot2(pv, v5, o_acc[hi][5], false);
                    o_acc[hi][6] = __builtin_amdgcn_fdot2(pv, v6, o_acc[hi][6], false);
                    o_acc[hi][7] = __builtin_amdgcn_fdot2(pv, v7, o_acc[hi][7], false);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: per-head merge of unnormalized SWA o_acc with compressed (comp_o, comp_m, comp_l) + single attn_sink[h]
    const int h_base = (tid >> 6) * 4;
    const int lane   = tid & 63;
    #pragma unroll
    for (int hi = 0; hi < 4; ++hi) {
        const int h = h_base + hi;
        const size_t glob_row = size_t(m) * 16u + size_t(h);
        const float ms = s_m[h];
        const float ls = s_l[h];
        float m_max = ms;
        float l_sum = ls;
        float y0 = o_acc[hi][0], y1 = o_acc[hi][1];
        float y2 = o_acc[hi][2], y3 = o_acc[hi][3];
        float y4 = o_acc[hi][4], y5 = o_acc[hi][5];
        float y6 = o_acc[hi][6], y7 = o_acc[hi][7];

        if (comp_o_f32 != nullptr) {
            const float lc = comp_l_f32[glob_row];
            if (lc > 0.0f && isfinite(lc)) {
                const float mc = comp_m_f32[glob_row];
                m_max = fmaxf(mc, ms);
                const float ac = __expf(mc - m_max);
                const float as = __expf(ms - m_max);
                l_sum = fmaf(lc, ac, ls * as);
                const float2* __restrict__ co2 = reinterpret_cast<const float2*>(comp_o_f32 + glob_row * 512u);
                const float2 c0 = co2[lane + 0];
                const float2 c1 = co2[lane + 64];
                const float2 c2 = co2[lane + 128];
                const float2 c3 = co2[lane + 192];
                y0 = fmaf(c0.x, ac, as * y0); y1 = fmaf(c0.y, ac, as * y1);
                y2 = fmaf(c1.x, ac, as * y2); y3 = fmaf(c1.y, ac, as * y3);
                y4 = fmaf(c2.x, ac, as * y4); y5 = fmaf(c2.y, ac, as * y5);
                y6 = fmaf(c3.x, ac, as * y6); y7 = fmaf(c3.y, ac, as * y7);
            }
        }

        const float sink  = attn_sink[h];
        const float m_tot = fmaxf(m_max, sink);
        const float sc    = __expf(m_max - m_tot);
        const float denom = l_sum * sc + __expf(sink - m_tot);
        const float inv_d = sc / denom;

        uint32_t* __restrict__ dst = reinterpret_cast<uint32_t*>(out_f16 + glob_row * 512u);
        dst[lane + 0]   = d_pack_f32x2_to_f16x2_u32(y0 * inv_d, y1 * inv_d);
        dst[lane + 64]  = d_pack_f32x2_to_f16x2_u32(y2 * inv_d, y3 * inv_d);
        dst[lane + 128] = d_pack_f32x2_to_f16x2_u32(y4 * inv_d, y5 * inv_d);
        dst[lane + 192] = d_pack_f32x2_to_f16x2_u32(y6 * inv_d, y7 * inv_d);
    }
}

void launch_hca_tp4_swa128_merge_f16(
    const uint16_t* d_q_f16, const uint16_t* d_kv_f16,
    const float* d_comp_o_f32, const float* d_comp_m_f32, const float* d_comp_l_f32,
    const float* d_attn_sink, uint16_t* d_out_f16,
    int M, int32_t pos0, int32_t base_pos, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hca_tp4_swa128_merge_f16, dim3(uint32_t(M)), dim3(256), 0, stream,
                       d_q_f16, d_kv_f16, d_comp_o_f32, d_comp_m_f32, d_comp_l_f32,
                       d_attn_sink, d_out_f16, pos0, base_pos);
}

// ---- Wave-11 C2: Zero-Bounce FP16 DSA Phase A Kernels (0 scratch, 0 slc, 0 trig) ----
__global__ __launch_bounds__(256, 4)
void k_rmsnorm_f16_io(
    const uint16_t* __restrict__ x_f16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ out_f16,
    int D, float eps)
{
    __shared__ float s_w[4];
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const uint16_t* __restrict__ xr = x_f16 + size_t(row) * size_t(D);
    uint16_t* __restrict__ yr = out_f16 + size_t(row) * size_t(D);
    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = d_f16_to_f32_core(xr[d]);
        ss = fmaf(v, v, ss);
    }
    ss = d_warp_sum64(ss);
    if ((tid & 63) == 0) s_w[tid >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(D) + eps);
    for (int d = tid; d < D; d += 256) {
        const float v = d_f16_to_f32_core(xr[d]) * inv * d_bf16_to_f32(w_bf16[d]);
        yr[d] = d_f32_to_f16_core(v);
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_prefill_kv_norm_rope_append_f16_in(
    uint16_t* __restrict__ q_f16,
    const uint16_t* __restrict__ kv_a_raw_f16,
    const uint16_t* __restrict__ kv_a_norm_w,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ mla_kv_cache_bf16,
    int n_heads, int max_kv, float eps)
{
    __shared__ float s_w[4];
    const int which = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];

    if (which < n_heads) {
        if (tid < 32) {
            uint16_t* __restrict__ qr = q_f16 + (size_t(m) * size_t(n_heads) + size_t(which)) * 256u + 192u;
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = d_f16_to_f32_core(qr[tid * 2]);
            const float x1 = d_f16_to_f32_core(qr[tid * 2 + 1]);
            qr[tid * 2]     = d_f32_to_f16_core(fmaf(x0, c, -(x1 * s)));
            qr[tid * 2 + 1] = d_f32_to_f16_core(fmaf(x1, c,   x0 * s));
        }
    } else if (pos >= 0 && pos < max_kv) {
        const uint16_t* __restrict__ src = kv_a_raw_f16 + size_t(m) * 576u;
        const float v0 = d_f16_to_f32_core(src[tid * 2]);
        const float v1 = d_f16_to_f32_core(src[tid * 2 + 1]);
        float ss = fmaf(v0, v0, v1 * v1);
        ss = d_warp_sum64(ss);
        if ((tid & 63) == 0) s_w[tid >> 6] = ss;
        __syncthreads();
        const float inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 512.0f) + eps);
        uint16_t* __restrict__ dst = mla_kv_cache_bf16 + size_t(pos) * 576u;
        dst[tid * 2]     = d_f32_to_bf16(v0 * inv * d_bf16_to_f32(kv_a_norm_w[tid * 2]));
        dst[tid * 2 + 1] = d_f32_to_bf16(v1 * inv * d_bf16_to_f32(kv_a_norm_w[tid * 2 + 1]));
        if (tid < 32) {
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = d_f16_to_f32_core(src[512 + tid * 2]);
            const float x1 = d_f16_to_f32_core(src[512 + tid * 2 + 1]);
            dst[512 + tid * 2]     = d_f32_to_bf16(fmaf(x0, c, -(x1 * s)));
            dst[512 + tid * 2 + 1] = d_f32_to_bf16(fmaf(x1, c,   x0 * s));
        }
    }
}

__global__ __launch_bounds__(64, 8)
void k_dsa_mla_rope_q_f16_inplace(
    uint16_t* __restrict__ q_f16,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    int n_heads)
{
    const int h   = int(blockIdx.x);
    const int m   = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (tid >= 32) return;
    const int pos = pos_ids[m];
    uint16_t* __restrict__ qr = q_f16 + (size_t(m) * size_t(n_heads) + size_t(h)) * 256u + 192u;
    float c, s;
    d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
    const float x0 = d_f16_to_f32_core(qr[tid * 2]);
    const float x1 = d_f16_to_f32_core(qr[tid * 2 + 1]);
    qr[tid * 2]     = d_f32_to_f16_core(fmaf(x0, c, -(x1 * s)));
    qr[tid * 2 + 1] = d_f32_to_f16_core(fmaf(x1, c,   x0 * s));
}

__global__ __launch_bounds__(128, 8)
void k_dsa_indexer_prep_neox_wht128_f16_in(
    uint16_t* __restrict__ iq_f16,
    const uint16_t* __restrict__ ik_raw_f16,
    const uint16_t* __restrict__ knorm_w,
    const uint16_t* __restrict__ knorm_b,
    const int32_t* __restrict__ pos_ids,
    const float* __restrict__ cos_hi, const float* __restrict__ sin_hi,
    const float* __restrict__ cos_lo, const float* __restrict__ sin_lo,
    uint16_t* __restrict__ ik_cache_bf16, int max_kv)
{
    __shared__ float s_x[128];
    __shared__ float s_w[2];
    const int which = int(blockIdx.x);
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos_ids[m];

    if (which < 64) {
        uint16_t* __restrict__ qh = iq_f16 + (size_t(m) * 64u + size_t(which)) * 128u;
        s_x[tid] = d_f16_to_f32_core(qh[tid]);
        __syncthreads();
        if (tid < 32) {
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = s_x[tid];
            const float x1 = s_x[tid + 32];
            s_x[tid]      = fmaf(x0, c, -(x1 * s));
            s_x[tid + 32] = fmaf(x1, c,   x0 * s);
        }
        d_wht128_nosign(s_x, tid);
        qh[tid] = d_f32_to_f16_core(s_x[tid]);
    } else {
        const float v = d_f16_to_f32_core(ik_raw_f16[size_t(m) * 128u + size_t(tid)]);
        float mu = d_warp_sum64(v);
        if ((tid & 63) == 0) s_w[tid >> 6] = mu;
        __syncthreads();
        const float mean = (s_w[0] + s_w[1]) * (1.0f / 128.0f);
        const float diff = v - mean;
        float var = d_warp_sum64(diff * diff);
        if ((tid & 63) == 0) s_w[tid >> 6] = var;
        __syncthreads();
        const float inv = rsqrtf((s_w[0] + s_w[1]) * (1.0f / 128.0f) + 1e-6f);
        s_x[tid] = fmaf(diff * inv, d_bf16_to_f32(knorm_w[tid]), d_bf16_to_f32(knorm_b[tid]));
        __syncthreads();
        if (tid < 32) {
            float c, s;
            d_rope2d_lookup(pos, tid, cos_hi, sin_hi, cos_lo, sin_lo, c, s);
            const float x0 = s_x[tid];
            const float x1 = s_x[tid + 32];
            s_x[tid]      = fmaf(x0, c, -(x1 * s));
            s_x[tid + 32] = fmaf(x1, c,   x0 * s);
        }
        d_wht128_nosign(s_x, tid);
        if (pos >= 0 && pos < max_kv) ik_cache_bf16[size_t(pos) * 128u + size_t(tid)] = d_f32_to_bf16(s_x[tid]);
    }
}

__global__ __launch_bounds__(64, 8)
void k_dsa_copy_qrope_f16_to_f16(const uint16_t* __restrict__ q_f16,
                                 uint16_t* __restrict__ q_abs_f16)
{
    const int row = int(blockIdx.x);
    const int d   = int(threadIdx.x);
    q_abs_f16[size_t(row) * 576u + 512u + size_t(d)] = q_f16[size_t(row) * 256u + 192u + size_t(d)];
}

__global__ __launch_bounds__(64, 8)
void k_flashindexer_select_heads_f16(
    const float* __restrict__ gate_logits_f32,
    const float* __restrict__ bias_f32,
    const uint16_t* __restrict__ iq_f16,
    const uint16_t* __restrict__ iw_f16,
    uint16_t* __restrict__ q_sel_f16,
    float* __restrict__ w_sel_f32)
{
    __shared__ float s_gate[64];
    __shared__ float s_score[64];
    __shared__ float s_w[64];
    __shared__ int   s_idx[8];

    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const float g = gate_logits_f32[size_t(m) * 64u + size_t(tid)];
    s_gate[tid]  = g;
    s_score[tid] = g + bias_f32[tid];
    s_w[tid]     = d_f16_to_f32_core(iw_f16[size_t(m) * 64u + size_t(tid)]);
    __syncthreads();

    float cur = s_score[tid];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float best_v = cur;
        int   best_i = tid;
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(best_v, off, 64);
            const int   oi = __shfl_xor(best_i, off, 64);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (tid == 0) {
            s_idx[k] = best_i;
            w_sel_f32[size_t(m) * 8u + size_t(k)] = s_w[best_i] * s_gate[best_i];
        }
        if (tid == best_i) cur = -1e38f;
    }
    __syncthreads();

    const uint16_t* __restrict__ q_src_m = iq_f16 + size_t(m) * 64u * 128u;
    uint16_t* __restrict__ q_dst_m = q_sel_f16 + size_t(m) * 8u * 128u;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int h = s_idx[k];
        const uint16_t* __restrict__ src = q_src_m + size_t(h) * 128u;
        uint16_t* __restrict__ dst = q_dst_m + size_t(k) * 128u;
        dst[tid * 2]     = src[tid * 2];
        dst[tid * 2 + 1] = src[tid * 2 + 1];
    }
}

__global__ __launch_bounds__(256, 4)
void k_dsa_vha_lowrank_lrga_f16_all(
    const uint16_t* o_loc_f16,
    const uint16_t* __restrict__ p_all_f16,
    const float* __restrict__ vha_q_16x16,
    const uint16_t* __restrict__ gate_f16,
    uint16_t* o_out_f16)
{
    __shared__ float s_Q[16][16];
    const int d = int(threadIdx.x);
    s_Q[d >> 4][d & 15] = vha_q_16x16[d];
    __syncthreads();

    const int m = int(blockIdx.y);
    const uint16_t* __restrict__ pb = p_all_f16 + size_t(m) * 16u * 256u + size_t(d);
    float p[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) p[k] = d_f16_to_f32_core(pb[size_t(k) * 256u]);

    const size_t base = size_t(m) * 16u * 256u + size_t(d);
    #pragma unroll 1
    for (int hl = 0; hl < 16; ++hl) {
        const size_t idx = base + size_t(hl) * 256u;
        float y = d_f16_to_f32_core(o_loc_f16[idx]);
        #pragma unroll
        for (int k = 0; k < 16; ++k) y = fmaf(s_Q[hl][k], p[k], y);
        const float g = d_f16_to_f32_core(gate_f16[idx]);
        o_out_f16[idx] = d_f32_to_f16_core(y / (1.0f + __expf(-g)));
    }
}

void launch_rmsnorm_f16_io(
    const uint16_t* d_x_f16, const uint16_t* d_w_bf16, uint16_t* d_out_f16,
    int M, int D, float eps, hipStream_t stream) {
    if (M <= 0 || D <= 0) return;
    hipLaunchKernelGGL(k_rmsnorm_f16_io, dim3(uint32_t(M)), dim3(256), 0, stream,
                       d_x_f16, d_w_bf16, d_out_f16, D, eps);
}

void launch_dsa_prefill_kv_norm_rope_append_f16_in(
    uint16_t* d_q_f16, const uint16_t* d_kv_a_raw_f16, const uint16_t* d_kv_a_norm_w,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_mla_kv_cache_bf16, int M, int n_heads, int max_kv, float eps, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_prefill_kv_norm_rope_append_f16_in,
                       dim3(uint32_t(n_heads + 1), uint32_t(M)), dim3(256), 0, stream,
                       d_q_f16, d_kv_a_raw_f16, d_kv_a_norm_w, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_mla_kv_cache_bf16, n_heads, max_kv, eps);
}

void launch_dsa_mla_rope_q_f16_inplace(
    uint16_t* d_q_f16, const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_mla_rope_q_f16_inplace,
                       dim3(uint32_t(n_heads), uint32_t(M)), dim3(64), 0, stream,
                       d_q_f16, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo, n_heads);
}

void launch_dsa_indexer_prep_neox_wht128_f16_in(
    uint16_t* d_iq_f16, const uint16_t* d_ik_raw_f16,
    const uint16_t* d_knorm_w, const uint16_t* d_knorm_b,
    const int32_t* d_pos_ids, const Rope2DTableGpu& rope,
    uint16_t* d_ik_cache_bf16, int M, int max_kv, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_indexer_prep_neox_wht128_f16_in,
                       dim3(65u, uint32_t(M)), dim3(128), 0, stream,
                       d_iq_f16, d_ik_raw_f16, d_knorm_w, d_knorm_b, d_pos_ids,
                       rope.d_cos_hi, rope.d_sin_hi, rope.d_cos_lo, rope.d_sin_lo,
                       d_ik_cache_bf16, max_kv);
}

void launch_dsa_copy_qrope_f16_to_f16(
    const uint16_t* d_q_f16, uint16_t* d_q_abs_f16,
    int M, int n_heads, hipStream_t stream) {
    if (M <= 0 || n_heads <= 0) return;
    hipLaunchKernelGGL(k_dsa_copy_qrope_f16_to_f16, dim3(uint32_t(M * n_heads)), dim3(64), 0, stream,
                       d_q_f16, d_q_abs_f16);
}

void launch_flashindexer_select_heads_f16(
    const float* d_gate_logits_f32, const float* d_bias_f32,
    const uint16_t* d_iq_f16, const uint16_t* d_iw_f16,
    uint16_t* d_q_sel_f16, float* d_w_sel_f32,
    int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_flashindexer_select_heads_f16, dim3(uint32_t(M)), dim3(64), 0, stream,
                       d_gate_logits_f32, d_bias_f32, d_iq_f16, d_iw_f16, d_q_sel_f16, d_w_sel_f32);
}

void launch_dsa_vha_lowrank_lrga_f16_all(
    const uint16_t* d_o_loc_f16, const uint16_t* d_p_all_f16,
    const float* d_vha_q_16x16, const uint16_t* d_gate_f16,
    uint16_t* d_o_out_f16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_dsa_vha_lowrank_lrga_f16_all, dim3(1u, uint32_t(M)), dim3(256), 0, stream,
                       d_o_loc_f16, d_p_all_f16, d_vha_q_16x16, d_gate_f16, d_o_out_f16);
}

// ---- Wave-11 C3: Single-Read LDS-Staged mHC Pre-GEMM + Fused ss + Dual BF16/FP16 Output ----
// Cuts res_bf16 HBM2 reads 6x -> 2x across all 85 mHC calls/chunk and eliminates 85 bf16_to_f16 passes!
// LDS = 2 * 256 * 16 + 32 = 8,224 B -> 4 WGs/CU (16 waves/CU), 0 scratch, 0 slc, 0 trig.
__global__ __launch_bounds__(256, 4)
void k_mhc_pre_gemm_ss_bf16_f16_f32(
    const uint16_t* __restrict__ res_bf16,
    const uint16_t* __restrict__ fn_24x16384_f16,
    float* __restrict__ out_Mx25,
    int M, int K)
{
    __shared__ uint4 s_x0[256];
    __shared__ uint4 s_x1[256];
    __shared__ float s_ss[2][4];

    const int m0   = int(blockIdx.x) * 2;
    const int cnt  = (M - m0 < 2) ? (M - m0) : 2;
    if (cnt <= 0) return;
    const int tid  = int(threadIdx.x);
    const int warp = tid >> 6; // 0..3 -> 6 outputs each (24 total)
    const int lane = tid & 63;
    const int r0   = warp * 6;
    const int k_vec8 = K >> 3;

    const uint4* __restrict__ f0 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 0) * size_t(K));
    const uint4* __restrict__ f1 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 1) * size_t(K));
    const uint4* __restrict__ f2 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 2) * size_t(K));
    const uint4* __restrict__ f3 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 3) * size_t(K));
    const uint4* __restrict__ f4 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 4) * size_t(K));
    const uint4* __restrict__ f5 = reinterpret_cast<const uint4*>(fn_24x16384_f16 + size_t(r0 + 5) * size_t(K));

    float a00 = 0.0f, a01 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f;
    float a40 = 0.0f, a41 = 0.0f;
    float a50 = 0.0f, a51 = 0.0f;
    float ss0 = 0.0f, ss1 = 0.0f;

    auto acc_ss_u4 = [](uint4 v, float s) -> float {
        const float f0 = d_bf16_to_f32(uint16_t(v.x & 0xFFFFu));
        const float f1 = d_bf16_to_f32(uint16_t(v.x >> 16));
        const float f2 = d_bf16_to_f32(uint16_t(v.y & 0xFFFFu));
        const float f3 = d_bf16_to_f32(uint16_t(v.y >> 16));
        const float f4 = d_bf16_to_f32(uint16_t(v.z & 0xFFFFu));
        const float f5 = d_bf16_to_f32(uint16_t(v.z >> 16));
        const float f6 = d_bf16_to_f32(uint16_t(v.w & 0xFFFFu));
        const float f7 = d_bf16_to_f32(uint16_t(v.w >> 16));
        return fmaf(f0, f0, fmaf(f1, f1, fmaf(f2, f2, fmaf(f3, f3, fmaf(f4, f4, fmaf(f5, f5, fmaf(f6, f6, fmaf(f7, f7, s))))))));
    };

    if (cnt == 2) {
        const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 0) * size_t(K));
        const uint4* __restrict__ x1 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0 + 1) * size_t(K));
        #pragma unroll 1
        for (int k_blk = 0; k_blk < k_vec8; k_blk += 256) {
            const int g_kv = k_blk + tid;
            if (g_kv < k_vec8) {
                const uint4 bx0 = x0[g_kv];
                const uint4 bx1 = x1[g_kv];
                ss0 = acc_ss_u4(bx0, ss0);
                ss1 = acc_ss_u4(bx1, ss1);
                s_x0[tid] = d_core_bf16_u4_to_f16_u4(bx0);
                s_x1[tid] = d_core_bf16_u4_to_f16_u4(bx1);
            }
            __syncthreads();
            const int rem = (k_vec8 - k_blk < 256) ? (k_vec8 - k_blk) : 256;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int idx = lane + t * 64;
                if (idx < rem) {
                    const int kv = k_blk + idx;
                    const uint4 v0 = s_x0[idx];
                    const uint4 v1 = s_x1[idx];
                    { const uint4 fv = f0[kv]; a00 = d_core_dot8_f16_u4_fdot2(v0, fv, a00); a01 = d_core_dot8_f16_u4_fdot2(v1, fv, a01); }
                    { const uint4 fv = f1[kv]; a10 = d_core_dot8_f16_u4_fdot2(v0, fv, a10); a11 = d_core_dot8_f16_u4_fdot2(v1, fv, a11); }
                    { const uint4 fv = f2[kv]; a20 = d_core_dot8_f16_u4_fdot2(v0, fv, a20); a21 = d_core_dot8_f16_u4_fdot2(v1, fv, a21); }
                    { const uint4 fv = f3[kv]; a30 = d_core_dot8_f16_u4_fdot2(v0, fv, a30); a31 = d_core_dot8_f16_u4_fdot2(v1, fv, a31); }
                    { const uint4 fv = f4[kv]; a40 = d_core_dot8_f16_u4_fdot2(v0, fv, a40); a41 = d_core_dot8_f16_u4_fdot2(v1, fv, a41); }
                    { const uint4 fv = f5[kv]; a50 = d_core_dot8_f16_u4_fdot2(v0, fv, a50); a51 = d_core_dot8_f16_u4_fdot2(v1, fv, a51); }
                }
            }
            __syncthreads();
        }
    } else {
        const uint4* __restrict__ x0 = reinterpret_cast<const uint4*>(res_bf16 + size_t(m0) * size_t(K));
        #pragma unroll 1
        for (int k_blk = 0; k_blk < k_vec8; k_blk += 256) {
            const int g_kv = k_blk + tid;
            if (g_kv < k_vec8) {
                const uint4 bx0 = x0[g_kv];
                ss0 = acc_ss_u4(bx0, ss0);
                s_x0[tid] = d_core_bf16_u4_to_f16_u4(bx0);
            }
            __syncthreads();
            const int rem = (k_vec8 - k_blk < 256) ? (k_vec8 - k_blk) : 256;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int idx = lane + t * 64;
                if (idx < rem) {
                    const int kv = k_blk + idx;
                    const uint4 v0 = s_x0[idx];
                    a00 = d_core_dot8_f16_u4_fdot2(v0, f0[kv], a00);
                    a10 = d_core_dot8_f16_u4_fdot2(v0, f1[kv], a10);
                    a20 = d_core_dot8_f16_u4_fdot2(v0, f2[kv], a20);
                    a30 = d_core_dot8_f16_u4_fdot2(v0, f3[kv], a30);
                    a40 = d_core_dot8_f16_u4_fdot2(v0, f4[kv], a40);
                    a50 = d_core_dot8_f16_u4_fdot2(v0, f5[kv], a50);
                }
            }
            __syncthreads();
        }
    }

    auto emit = [&](int r_off, float v0, float v1) {
        v0 = d_warp_sum64(v0);
        if (cnt > 1) v1 = d_warp_sum64(v1);
        if (lane == 0) {
            out_Mx25[size_t(m0 + 0) * 25u + size_t(r0 + r_off)] = v0;
            if (cnt > 1) out_Mx25[size_t(m0 + 1) * 25u + size_t(r0 + r_off)] = v1;
        }
    };
    emit(0, a00, a01);
    emit(1, a10, a11);
    emit(2, a20, a21);
    emit(3, a30, a31);
    emit(4, a40, a41);
    emit(5, a50, a51);

    ss0 = d_warp_sum64(ss0);
    if (cnt > 1) ss1 = d_warp_sum64(ss1);
    if (lane == 0) {
        s_ss[0][warp] = ss0;
        if (cnt > 1) s_ss[1][warp] = ss1;
    }
    __syncthreads();
    if (tid == 0) {
        out_Mx25[size_t(m0 + 0) * 25u + 24u] = s_ss[0][0] + s_ss[0][1] + s_ss[0][2] + s_ss[0][3];
        if (cnt > 1) {
            out_Mx25[size_t(m0 + 1) * 25u + 24u] = s_ss[1][0] + s_ss[1][1] + s_ss[1][2] + s_ss[1][3];
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_mhc_pre_sinkhorn_from_mixes_ss(
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ raw_mixes_Mx25,
    const float* __restrict__ hc_scale3,
    const float* __restrict__ hc_base24,
    const uint16_t* __restrict__ norm_w_bf16,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    uint16_t* __restrict__ layer_in_bf16,
    uint16_t* __restrict__ layer_in_f16,
    int H, float rms_eps, float hc_eps)
{
    __shared__ float s_w[4];
    __shared__ float s_mixes[24];
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_cm[16];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int hc_dim = 4 * H;
    const uint16_t* __restrict__ rr = res_bf16 + size_t(m) * size_t(hc_dim);

    const float ss = raw_mixes_Mx25[size_t(m) * 25u + 24u];
    const float inv_rms = 1.0f / (sqrtf(ss / float(hc_dim)) + rms_eps);

    if (tid < 24) s_mixes[tid] = raw_mixes_Mx25[size_t(m) * 25u + size_t(tid)] * inv_rms;
    __syncthreads();
    if (tid == 0) {
        d_mhc_sinkhorn_in_lds(s_mixes, hc_scale3, hc_base24, s_pre, s_post, s_cm, hc_eps);
        for (int j = 0; j < 4; ++j) post_mix[m * 4 + j] = s_post[j];
        for (int k = 0; k < 16; ++k) comb_mix[m * 16 + k] = s_cm[k];
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    uint16_t* __restrict__ yr = layer_in_bf16 ? (layer_in_bf16 + size_t(m) * size_t(H)) : nullptr;
    uint16_t* __restrict__ yf = layer_in_f16  ? (layer_in_f16  + size_t(m) * size_t(H)) : nullptr;

    if (H == 4096) {
        const uint4* __restrict__ r0 = reinterpret_cast<const uint4*>(rr + 0u * 4096u);
        const uint4* __restrict__ r1 = reinterpret_cast<const uint4*>(rr + 1u * 4096u);
        const uint4* __restrict__ r2 = reinterpret_cast<const uint4*>(rr + 2u * 4096u);
        const uint4* __restrict__ r3 = reinterpret_cast<const uint4*>(rr + 3u * 4096u);
        float vf[16];
        float lss = 0.0f;
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int i8 = tid + it * 256;
            const uint4 u0 = r0[i8], u1 = r1[i8], u2 = r2[i8], u3 = r3[i8];
            auto comb_bf16 = [&](uint16_t b0, uint16_t b1, uint16_t b2, uint16_t b3) -> float {
                float v = p0 * d_bf16_to_f32(b0);
                v = fmaf(p1, d_bf16_to_f32(b1), v);
                v = fmaf(p2, d_bf16_to_f32(b2), v);
                v = fmaf(p3, d_bf16_to_f32(b3), v);
                const float vr = d_bf16_to_f32(d_f32_to_bf16(v));
                lss = fmaf(vr, vr, lss);
                return vr;
            };
            const int base = it * 8;
            vf[base + 0] = comb_bf16(uint16_t(u0.x & 0xFFFFu), uint16_t(u1.x & 0xFFFFu), uint16_t(u2.x & 0xFFFFu), uint16_t(u3.x & 0xFFFFu));
            vf[base + 1] = comb_bf16(uint16_t(u0.x >> 16),     uint16_t(u1.x >> 16),     uint16_t(u2.x >> 16),     uint16_t(u3.x >> 16));
            vf[base + 2] = comb_bf16(uint16_t(u0.y & 0xFFFFu), uint16_t(u1.y & 0xFFFFu), uint16_t(u2.y & 0xFFFFu), uint16_t(u3.y & 0xFFFFu));
            vf[base + 3] = comb_bf16(uint16_t(u0.y >> 16),     uint16_t(u1.y >> 16),     uint16_t(u2.y >> 16),     uint16_t(u3.y >> 16));
            vf[base + 4] = comb_bf16(uint16_t(u0.z & 0xFFFFu), uint16_t(u1.z & 0xFFFFu), uint16_t(u2.z & 0xFFFFu), uint16_t(u3.z & 0xFFFFu));
            vf[base + 5] = comb_bf16(uint16_t(u0.z >> 16),     uint16_t(u1.z >> 16),     uint16_t(u2.z >> 16),     uint16_t(u3.z >> 16));
            vf[base + 6] = comb_bf16(uint16_t(u0.w & 0xFFFFu), uint16_t(u1.w & 0xFFFFu), uint16_t(u2.w & 0xFFFFu), uint16_t(u3.w & 0xFFFFu));
            vf[base + 7] = comb_bf16(uint16_t(u0.w >> 16),     uint16_t(u1.w >> 16),     uint16_t(u2.w >> 16),     uint16_t(u3.w >> 16));
        }

        float n_inv = 1.0f;
        if (norm_w_bf16 != nullptr) {
            lss = d_warp_sum64(lss);
            if (lane == 0) s_w[warp] = lss;
            __syncthreads();
            n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) * (1.0f / 4096.0f) + rms_eps);
        }

        const uint4* __restrict__ w8 = norm_w_bf16 ? reinterpret_cast<const uint4*>(norm_w_bf16) : nullptr;
        uint4* __restrict__ yb8 = yr ? reinterpret_cast<uint4*>(yr) : nullptr;
        uint4* __restrict__ yf8 = yf ? reinterpret_cast<uint4*>(yf) : nullptr;
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int i8 = tid + it * 256;
            const int base = it * 8;
            float y0 = vf[base + 0], y1 = vf[base + 1], y2 = vf[base + 2], y3 = vf[base + 3];
            float y4 = vf[base + 4], y5 = vf[base + 5], y6 = vf[base + 6], y7 = vf[base + 7];
            if (w8 != nullptr) {
                const uint4 wv = w8[i8];
                y0 = y0 * n_inv * d_bf16_to_f32(uint16_t(wv.x & 0xFFFFu));
                y1 = y1 * n_inv * d_bf16_to_f32(uint16_t(wv.x >> 16));
                y2 = y2 * n_inv * d_bf16_to_f32(uint16_t(wv.y & 0xFFFFu));
                y3 = y3 * n_inv * d_bf16_to_f32(uint16_t(wv.y >> 16));
                y4 = y4 * n_inv * d_bf16_to_f32(uint16_t(wv.z & 0xFFFFu));
                y5 = y5 * n_inv * d_bf16_to_f32(uint16_t(wv.z >> 16));
                y6 = y6 * n_inv * d_bf16_to_f32(uint16_t(wv.w & 0xFFFFu));
                y7 = y7 * n_inv * d_bf16_to_f32(uint16_t(wv.w >> 16));
            }
            if (yb8 != nullptr) {
                uint4 ob;
                ob.x = uint32_t(d_f32_to_bf16(y0)) | (uint32_t(d_f32_to_bf16(y1)) << 16);
                ob.y = uint32_t(d_f32_to_bf16(y2)) | (uint32_t(d_f32_to_bf16(y3)) << 16);
                ob.z = uint32_t(d_f32_to_bf16(y4)) | (uint32_t(d_f32_to_bf16(y5)) << 16);
                ob.w = uint32_t(d_f32_to_bf16(y6)) | (uint32_t(d_f32_to_bf16(y7)) << 16);
                yb8[i8] = ob;
            }
            if (yf8 != nullptr) {
                uint4 oh;
                oh.x = d_pack_f32x2_to_f16x2_u32(y0, y1);
                oh.y = d_pack_f32x2_to_f16x2_u32(y2, y3);
                oh.z = d_pack_f32x2_to_f16x2_u32(y4, y5);
                oh.w = d_pack_f32x2_to_f16x2_u32(y6, y7);
                yf8[i8] = oh;
            }
        }
        return;
    }

    // Generic fallback for H != 4096
    float lss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        float v = p0 * d_bf16_to_f32(rr[0 * H + d]);
        v = fmaf(p1, d_bf16_to_f32(rr[1 * H + d]), v);
        v = fmaf(p2, d_bf16_to_f32(rr[2 * H + d]), v);
        v = fmaf(p3, d_bf16_to_f32(rr[3 * H + d]), v);
        const uint16_t vb = d_f32_to_bf16(v);
        if (yr) yr[d] = vb;
        const float vr = d_bf16_to_f32(vb);
        lss = fmaf(vr, vr, lss);
    }
    if (norm_w_bf16 != nullptr) {
        lss = d_warp_sum64(lss);
        if (lane == 0) s_w[warp] = lss;
        __syncthreads();
        const float n_inv = rsqrtf((s_w[0] + s_w[1] + s_w[2] + s_w[3]) / float(H) + rms_eps);
        for (int d = tid; d < H; d += 256) {
            const float val = d_bf16_to_f32(yr[d]) * n_inv * d_bf16_to_f32(norm_w_bf16[d]);
            if (yr) yr[d] = d_f32_to_bf16(val);
            if (yf) yf[d] = d_f32_to_f16_core(val);
        }
    } else if (yf != nullptr && yr != nullptr) {
        for (int d = tid; d < H; d += 256) yf[d] = d_f32_to_f16_core(d_bf16_to_f32(yr[d]));
    }
}

void launch_mhc_pre_gemm_ss_bf16_f16_f32(
    const uint16_t* d_res_bf16, const uint16_t* d_fn_24x16384_f16,
    float* d_out_Mx25, int M, int H, hipStream_t stream) {
    if (M <= 0 || H <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_gemm_ss_bf16_f16_f32, dim3(uint32_t((M + 1) / 2)), dim3(256), 0, stream,
                       d_res_bf16, d_fn_24x16384_f16, d_out_Mx25, M, 4 * H);
}

void launch_mhc_pre_sinkhorn_from_mixes_ss(
    const uint16_t* d_res_bf16, const float* d_raw_mixes_Mx25,
    const float* d_hc_scale3, const float* d_hc_base24, const uint16_t* d_norm_w_bf16,
    float* d_post_mix, float* d_comb_mix, uint16_t* d_layer_in_bf16, uint16_t* d_layer_in_f16,
    int M, int H, float rms_eps, float hc_eps, hipStream_t stream) {
    if (M <= 0 || H <= 0) return;
    hipLaunchKernelGGL(k_mhc_pre_sinkhorn_from_mixes_ss, dim3(uint32_t(M)), dim3(256), 0, stream,
                       d_res_bf16, d_raw_mixes_Mx25, d_hc_scale3, d_hc_base24, d_norm_w_bf16,
                       d_post_mix, d_comb_mix, d_layer_in_bf16, d_layer_in_f16, H, rms_eps, hc_eps);
}

// ---- Wave-11 C5: Zero-Scratch HBM2 Canonical 4-Way FP16 AllReduce Sum Kernels ----
// Bit-exact left-to-right (((p0 + p1) + p2) + p3) in FP32 -> RNE FP16 or RNE BF16; safe in-place when dst aliases p0..p3.
__global__ __launch_bounds__(256, 4)
void k_sum4_f16_canonical(const uint16_t* p0, const uint16_t* p1,
                          const uint16_t* p2, const uint16_t* p3,
                          uint16_t* dst_f16, size_t n) {
    const size_t idx8 = (size_t(blockIdx.x) * 256u + size_t(threadIdx.x)) * 8u;
    if (idx8 + 8u <= n) {
        const uint4 a0 = *reinterpret_cast<const uint4*>(p0 + idx8);
        const uint4 a1 = *reinterpret_cast<const uint4*>(p1 + idx8);
        const uint4 a2 = *reinterpret_cast<const uint4*>(p2 + idx8);
        const uint4 a3 = *reinterpret_cast<const uint4*>(p3 + idx8);
        auto sum_pair = [](uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3) -> uint32_t {
            float lo = d_f16_to_f32_core(uint16_t(u0 & 0xFFFFu));
            lo += d_f16_to_f32_core(uint16_t(u1 & 0xFFFFu));
            lo += d_f16_to_f32_core(uint16_t(u2 & 0xFFFFu));
            lo += d_f16_to_f32_core(uint16_t(u3 & 0xFFFFu));
            float hi = d_f16_to_f32_core(uint16_t(u0 >> 16));
            hi += d_f16_to_f32_core(uint16_t(u1 >> 16));
            hi += d_f16_to_f32_core(uint16_t(u2 >> 16));
            hi += d_f16_to_f32_core(uint16_t(u3 >> 16));
            return uint32_t(d_f32_to_f16_core(lo)) | (uint32_t(d_f32_to_f16_core(hi)) << 16);
        };
        uint4 out;
        out.x = sum_pair(a0.x, a1.x, a2.x, a3.x);
        out.y = sum_pair(a0.y, a1.y, a2.y, a3.y);
        out.z = sum_pair(a0.z, a1.z, a2.z, a3.z);
        out.w = sum_pair(a0.w, a1.w, a2.w, a3.w);
        *reinterpret_cast<uint4*>(dst_f16 + idx8) = out;
    } else {
        for (size_t i = idx8; i < n; ++i) {
            float s = d_f16_to_f32_core(p0[i]);
            s += d_f16_to_f32_core(p1[i]);
            s += d_f16_to_f32_core(p2[i]);
            s += d_f16_to_f32_core(p3[i]);
            dst_f16[i] = d_f32_to_f16_core(s);
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_sum4_f16_to_bf16_canonical(const uint16_t* p0, const uint16_t* p1,
                                  const uint16_t* p2, const uint16_t* p3,
                                  uint16_t* dst_bf16, size_t n) {
    const size_t idx8 = (size_t(blockIdx.x) * 256u + size_t(threadIdx.x)) * 8u;
    if (idx8 + 8u <= n) {
        const uint4 a0 = *reinterpret_cast<const uint4*>(p0 + idx8);
        const uint4 a1 = *reinterpret_cast<const uint4*>(p1 + idx8);
        const uint4 a2 = *reinterpret_cast<const uint4*>(p2 + idx8);
        const uint4 a3 = *reinterpret_cast<const uint4*>(p3 + idx8);
        auto sum_pair = [](uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3) -> uint32_t {
            float lo = d_f16_to_f32_core(uint16_t(u0 & 0xFFFFu));
            lo += d_f16_to_f32_core(uint16_t(u1 & 0xFFFFu));
            lo += d_f16_to_f32_core(uint16_t(u2 & 0xFFFFu));
            lo += d_f16_to_f32_core(uint16_t(u3 & 0xFFFFu));
            float hi = d_f16_to_f32_core(uint16_t(u0 >> 16));
            hi += d_f16_to_f32_core(uint16_t(u1 >> 16));
            hi += d_f16_to_f32_core(uint16_t(u2 >> 16));
            hi += d_f16_to_f32_core(uint16_t(u3 >> 16));
            return uint32_t(d_f32_to_bf16(lo)) | (uint32_t(d_f32_to_bf16(hi)) << 16);
        };
        uint4 out;
        out.x = sum_pair(a0.x, a1.x, a2.x, a3.x);
        out.y = sum_pair(a0.y, a1.y, a2.y, a3.y);
        out.z = sum_pair(a0.z, a1.z, a2.z, a3.z);
        out.w = sum_pair(a0.w, a1.w, a2.w, a3.w);
        *reinterpret_cast<uint4*>(dst_bf16 + idx8) = out;
    } else {
        for (size_t i = idx8; i < n; ++i) {
            float s = d_f16_to_f32_core(p0[i]);
            s += d_f16_to_f32_core(p1[i]);
            s += d_f16_to_f32_core(p2[i]);
            s += d_f16_to_f32_core(p3[i]);
            dst_bf16[i] = d_f32_to_bf16(s);
        }
    }
}

void launch_sum4_f16_canonical(
    const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
    uint16_t* dst_f16, size_t n, hipStream_t stream) {
    if (n == 0) return;
    const size_t n8 = (n + 7u) / 8u;
    const uint32_t grid = uint32_t((n8 + 255u) / 256u);
    hipLaunchKernelGGL(k_sum4_f16_canonical, dim3(grid), dim3(256), 0, stream,
                       p0, p1, p2, p3, dst_f16, n);
}

void launch_sum4_f16_to_bf16_canonical(
    const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
    uint16_t* dst_bf16, size_t n, hipStream_t stream) {
    if (n == 0) return;
    const size_t n8 = (n + 7u) / 8u;
    const uint32_t grid = uint32_t((n8 + 255u) / 256u);
    hipLaunchKernelGGL(k_sum4_f16_to_bf16_canonical, dim3(grid), dim3(256), 0, stream,
                       p0, p1, p2, p3, dst_bf16, n);
}

// ============================================================================
// Wave-11 Omni Turbo Pillar II (K2.1/K2.2): Zero-Scratch 1 TB/s HBM2 .kvblk Scatter/Gather
// ============================================================================
__global__ __launch_bounds__(256, 4)
void k_kvblk_unpack_scatter_hbm2(const uint8_t* __restrict__ d_slab, KvBlkCopyTable tbl) {
    const uint32_t tid     = blockIdx.x * 256u + threadIdx.x;
    const uint32_t gstride = gridDim.x * 256u;
    const uint32_t cnt     = tbl.count;
    for (uint32_t e = 0u; e < cnt; ++e) {
        const uint32_t nb   = tbl.nbytes[e];
        const uint32_t n16  = nb >> 4;
        const uint8_t* src8 = d_slab + tbl.slab_off[e];
        uint8_t*       dst8 = reinterpret_cast<uint8_t*>(tbl.dev_ptr[e]);
        const uint4*   s16  = reinterpret_cast<const uint4*>(src8);
        uint4*         d16  = reinterpret_cast<uint4*>(dst8);
        for (uint32_t i = tid; i < n16; i += gstride) {
            d16[i] = s16[i];
        }
        const uint32_t rem0 = n16 << 4;
        for (uint32_t b = rem0 + tid; b < nb; b += gstride) {
            dst8[b] = src8[b];
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_kvblk_pack_gather_hbm2(uint8_t* __restrict__ d_slab, KvBlkCopyTable tbl) {
    const uint32_t tid     = blockIdx.x * 256u + threadIdx.x;
    const uint32_t gstride = gridDim.x * 256u;
    const uint32_t cnt     = tbl.count;
    for (uint32_t e = 0u; e < cnt; ++e) {
        const uint32_t nb   = tbl.nbytes[e];
        const uint32_t n16  = nb >> 4;
        const uint8_t* src8 = reinterpret_cast<const uint8_t*>(tbl.dev_ptr[e]);
        uint8_t*       dst8 = d_slab + tbl.slab_off[e];
        const uint4*   s16  = reinterpret_cast<const uint4*>(src8);
        uint4*         d16  = reinterpret_cast<uint4*>(dst8);
        for (uint32_t i = tid; i < n16; i += gstride) {
            d16[i] = s16[i];
        }
        const uint32_t rem0 = n16 << 4;
        for (uint32_t b = rem0 + tid; b < nb; b += gstride) {
            dst8[b] = src8[b];
        }
    }
}

void launch_kvblk_unpack_scatter_hbm2(const uint8_t* d_slab, const KvBlkCopyTable& tbl, hipStream_t stream) {
    if (tbl.count == 0u) return;
    hipLaunchKernelGGL(k_kvblk_unpack_scatter_hbm2, dim3(240), dim3(256), 0, stream, d_slab, tbl);
}

void launch_kvblk_pack_gather_hbm2(uint8_t* d_slab, const KvBlkCopyTable& tbl, hipStream_t stream) {
    if (tbl.count == 0u) return;
    hipLaunchKernelGGL(k_kvblk_pack_gather_hbm2, dim3(240), dim3(256), 0, stream, d_slab, tbl);
}

// ============================================================================
// Wave-11 Omni Turbo Pillar III (D3.1/D3.2): Deep kPingPong=16 Monotonic Mapped TP4 & 5-MI50 Decode MoE
// ============================================================================
__device__ __forceinline__ uint32_t d_sum4_f16_u32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    float lo = d_f16_to_f32_core(uint16_t(a & 0xFFFFu));
    lo += d_f16_to_f32_core(uint16_t(b & 0xFFFFu));
    lo += d_f16_to_f32_core(uint16_t(c & 0xFFFFu));
    lo += d_f16_to_f32_core(uint16_t(d & 0xFFFFu));
    float hi = d_f16_to_f32_core(uint16_t(a >> 16));
    hi += d_f16_to_f32_core(uint16_t(b >> 16));
    hi += d_f16_to_f32_core(uint16_t(c >> 16));
    hi += d_f16_to_f32_core(uint16_t(d >> 16));
    return uint32_t(d_f32_to_f16_core(lo)) | (uint32_t(d_f32_to_f16_core(hi)) << 16);
}

__device__ __forceinline__ uint32_t d_sum4_f16_to_bf16_u32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    float lo = d_f16_to_f32_core(uint16_t(a & 0xFFFFu));
    lo += d_f16_to_f32_core(uint16_t(b & 0xFFFFu));
    lo += d_f16_to_f32_core(uint16_t(c & 0xFFFFu));
    lo += d_f16_to_f32_core(uint16_t(d & 0xFFFFu));
    float hi = d_f16_to_f32_core(uint16_t(a >> 16));
    hi += d_f16_to_f32_core(uint16_t(b >> 16));
    hi += d_f16_to_f32_core(uint16_t(c >> 16));
    hi += d_f16_to_f32_core(uint16_t(d >> 16));
    return uint32_t(d_f32_to_bf16(lo)) | (uint32_t(d_f32_to_bf16(hi)) << 16);
}

template <bool kOutBf16>
__global__ __launch_bounds__(1024, 1)
void k_tp4_mapped_allreduce_f16_impl(
    const uint16_t* __restrict__ src_local,
    uint16_t* __restrict__ dst_local,
    uint16_t* __restrict__ mapped_slabs,
    uint32_t* __restrict__ mapped_ready,
    uint32_t* __restrict__ mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const int n8  = n_elems >> 3;
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ base_slab = mapped_slabs + size_t(slot) * 4u * size_t(slab_stride);

    if (tid == 0 && epoch > 16u) {
        const uint32_t prev_ep = epoch - 16u;
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_done[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    const uint4* __restrict__ s8 = reinterpret_cast<const uint4*>(src_local);
    uint4* __restrict__ my8 = reinterpret_cast<uint4*>(base_slab + size_t(rank) * size_t(slab_stride));
    for (int i = tid; i < n8; i += 1024) my8[i] = s8[i];

    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_ready[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    const uint4* __restrict__ p0 = (rank == 0) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(0) * size_t(slab_stride));
    const uint4* __restrict__ p1 = (rank == 1) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(1) * size_t(slab_stride));
    const uint4* __restrict__ p2 = (rank == 2) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(2) * size_t(slab_stride));
    const uint4* __restrict__ p3 = (rank == 3) ? s8 : reinterpret_cast<const uint4*>(base_slab + size_t(3) * size_t(slab_stride));
    uint4* __restrict__ out8 = reinterpret_cast<uint4*>(dst_local);
    for (int i = tid; i < n8; i += 1024) {
        const uint4 a = p0[i], b = p1[i], c = p2[i], d = p3[i];
        uint4 res;
        if constexpr (kOutBf16) {
            res.x = d_sum4_f16_to_bf16_u32(a.x, b.x, c.x, d.x);
            res.y = d_sum4_f16_to_bf16_u32(a.y, b.y, c.y, d.y);
            res.z = d_sum4_f16_to_bf16_u32(a.z, b.z, c.z, d.z);
            res.w = d_sum4_f16_to_bf16_u32(a.w, b.w, c.w, d.w);
        } else {
            res.x = d_sum4_f16_u32(a.x, b.x, c.x, d.x);
            res.y = d_sum4_f16_u32(a.y, b.y, c.y, d.y);
            res.z = d_sum4_f16_u32(a.z, b.z, c.z, d.z);
            res.w = d_sum4_f16_u32(a.w, b.w, c.w, d.w);
        }
        out8[i] = res;
    }

    __syncthreads();
    if (tid == 0) {
        __hip_atomic_store(&mapped_done[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

void launch_tp4_mapped_allreduce_f16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (n_elems <= 0) return;
    hipLaunchKernelGGL((k_tp4_mapped_allreduce_f16_impl<false>), dim3(1), dim3(1024), 0, stream,
                       d_src_local, d_dst_local, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, n_elems, slab_stride, epoch);
}

void launch_tp4_mapped_allreduce_f16_to_bf16(
    const uint16_t* d_src_local, uint16_t* d_dst_local,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int n_elems, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (n_elems <= 0) return;
    hipLaunchKernelGGL((k_tp4_mapped_allreduce_f16_impl<true>), dim3(1), dim3(1024), 0, stream,
                       d_src_local, d_dst_local, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, n_elems, slab_stride, epoch);
}

// Zero-Host-Sync Mapped TP4 DSA q_abs Gather ([M, 16, 576] x 4 -> owner [M, 64, 576] token-major, 16-wave PCIe burst!)
__global__ __launch_bounds__(1024, 1)
void k_tp4_mapped_dsa_gather_qabs_bf16(
    const uint16_t* __restrict__ src_loc,
    uint16_t* __restrict__ dst_own_or_null,
    uint16_t* __restrict__ mapped_slabs,
    uint32_t* __restrict__ mapped_ready,
    uint32_t* __restrict__ mapped_done,
    int rank, int owner_rank, int M, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ base_slab = mapped_slabs + size_t(slot) * 4u * size_t(slab_stride);

    if (tid == 0 && epoch > 16u) {
        const uint32_t prev_ep = epoch - 16u;
        for (int r = 0; r < 4; ++r) {
            if (r == rank) continue;
            while (__hip_atomic_load(&mapped_done[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    constexpr int kRowU4 = (16 * 576) / 8; // 1152 uint4 per token per rank
    const int n8 = M * kRowU4;
    const uint4* __restrict__ s8 = reinterpret_cast<const uint4*>(src_loc);
    uint4* __restrict__ my8 = reinterpret_cast<uint4*>(base_slab + size_t(rank) * size_t(slab_stride));
    for (int i = tid; i < n8; i += 1024) my8[i] = s8[i];

    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }

    if (rank == owner_rank) {
        if (tid == 0) {
            for (int r = 0; r < 4; ++r) {
                if (r == rank) continue;
                while (__hip_atomic_load(&mapped_ready[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            __threadfence_system();
        }
        __syncthreads();
        uint4* __restrict__ out8 = reinterpret_cast<uint4*>(dst_own_or_null);
        for (int r = 0; r < 4; ++r) {
            const uint4* __restrict__ p8 = reinterpret_cast<const uint4*>(base_slab + size_t(r) * size_t(slab_stride));
            for (int m = 0; m < M; ++m) {
                const uint4* __restrict__ sr = p8 + size_t(m) * size_t(kRowU4);
                uint4* __restrict__ dr = out8 + size_t(m * 4 + r) * size_t(kRowU4);
                for (int i = tid; i < kRowU4; i += 1024) dr[i] = sr[i];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        __hip_atomic_store(&mapped_done[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// Zero-Host-Sync Mapped TP4 DSA o_lat Scatter (owner [M, 64, 512] token-major -> [M, 16, 512] per rank, 16-wave PCIe burst!)
__global__ __launch_bounds__(1024, 1)
void k_tp4_mapped_dsa_scatter_olat_bf16(
    const uint16_t* __restrict__ src_own_or_null,
    uint16_t* __restrict__ dst_loc,
    uint16_t* __restrict__ mapped_slabs,
    uint32_t* __restrict__ mapped_ready,
    uint32_t* __restrict__ mapped_done,
    int rank, int owner_rank, int M, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ base_slab = mapped_slabs + size_t(slot) * 4u * size_t(slab_stride);
    constexpr int kRowU4 = (16 * 512) / 8; // 1024 uint4 per token per rank
    const int n8 = M * kRowU4;

    if (rank == owner_rank) {
        if (tid == 0 && epoch > 16u) {
            const uint32_t prev_ep = epoch - 16u;
            for (int r = 0; r < 4; ++r) {
                if (r == rank) continue;
                while (__hip_atomic_load(&mapped_done[r * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        }
        __syncthreads();
        const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(src_own_or_null);
        for (int r = 0; r < 4; ++r) {
            uint4* __restrict__ p8 = reinterpret_cast<uint4*>(base_slab + size_t(r) * size_t(slab_stride));
            for (int m = 0; m < M; ++m) {
                const uint4* __restrict__ sr = in8 + size_t(m * 4 + r) * size_t(kRowU4);
                uint4* __restrict__ dr = p8 + size_t(m) * size_t(kRowU4);
                dr[tid] = sr[tid];
            }
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        const uint4* __restrict__ my8 = reinterpret_cast<const uint4*>(base_slab + size_t(rank) * size_t(slab_stride));
        uint4* __restrict__ d8 = reinterpret_cast<uint4*>(dst_loc);
        for (int i = tid; i < n8; i += 1024) d8[i] = my8[i];
        __syncthreads();
        if (tid == 0) {
            __hip_atomic_store(&mapped_done[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    } else {
        if (tid == 0) {
            while (__hip_atomic_load(&mapped_ready[owner_rank * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
            __threadfence_system();
        }
        __syncthreads();
        const uint4* __restrict__ my8 = reinterpret_cast<const uint4*>(base_slab + size_t(rank) * size_t(slab_stride));
        uint4* __restrict__ d8 = reinterpret_cast<uint4*>(dst_loc);
        for (int i = tid; i < n8; i += 1024) d8[i] = my8[i];
        __syncthreads();
        if (tid == 0) {
            __hip_atomic_store(&mapped_ready[rank * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&mapped_done[rank * 16],  epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

void launch_tp4_mapped_dsa_gather_qabs_bf16(
    const uint16_t* d_src_loc, uint16_t* d_dst_own_or_null,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int owner_rank, int M, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_tp4_mapped_dsa_gather_qabs_bf16, dim3(1), dim3(1024), 0, stream,
                       d_src_loc, d_dst_own_or_null, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, owner_rank, M, slab_stride, epoch);
}

void launch_tp4_mapped_dsa_scatter_olat_bf16(
    const uint16_t* d_src_own_or_null, uint16_t* d_dst_loc,
    uint16_t* d_mapped_slabs, uint32_t* d_mapped_ready, uint32_t* d_mapped_done,
    int rank, int owner_rank, int M, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_tp4_mapped_dsa_scatter_olat_bf16, dim3(1), dim3(1024), 0, stream,
                       d_src_own_or_null, d_dst_loc, d_mapped_slabs, d_mapped_ready, d_mapped_done,
                       rank, owner_rank, M, slab_stride, epoch);
}

// 5-MI50 + Blackhole P150A Mapped Decode MoE Ring (16 slots x 7 slabs/slot, 16-wave uint4 PCIe bursts, 0 host syncs!)
__global__ __launch_bounds__(1024, 1)
void k_moe_mapped_publish_xrot_topk(
    const uint16_t* __restrict__ x_rot_f16,
    const int32_t*  __restrict__ topk_ids,
    const float*    __restrict__ topk_wts,
    uint16_t* __restrict__ moe_mapped_slabs,
    uint32_t* __restrict__ moe_mapped_ready,
    uint32_t* __restrict__ moe_mapped_done,
    int M, int K_lat, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const uint32_t slot = (epoch - 1u) & 15u;
    if (tid == 0 && epoch > 16u) {
        const uint32_t prev_ep = epoch - 16u;
        for (int g = 0; g < 5; ++g) {
            while (__hip_atomic_load(&moe_mapped_done[g * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < prev_ep) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    uint16_t* __restrict__ s5 = moe_mapped_slabs + (size_t(slot) * 7u + 5u) * size_t(slab_stride);
    const int n8 = (M * K_lat) >> 3;
    const uint4* __restrict__ src8 = reinterpret_cast<const uint4*>(x_rot_f16);
    uint4* __restrict__ dst8 = reinterpret_cast<uint4*>(s5);
    for (int i = tid; i < n8; i += 1024) dst8[i] = src8[i];

    const int np = M * 10;
    int32_t* __restrict__ ids_dst = reinterpret_cast<int32_t*>(s5 + 16384u);
    float*   __restrict__ wts_dst = reinterpret_cast<float*>(s5 + 16544u);
    if (tid < np) {
        ids_dst[tid] = topk_ids[tid];
        wts_dst[tid] = topk_wts[tid];
    }

    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&moe_mapped_ready[5 * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ __launch_bounds__(1024, 1)
void k_moe_mapped_wait_xrot_copy(
    uint16_t* __restrict__ x_rot_dst,
    const uint16_t* __restrict__ moe_mapped_slabs,
    const uint32_t* __restrict__ moe_mapped_ready,
    int M, int K_lat, int slab_stride, uint32_t epoch)
{
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
    for (int i = tid; i < n8; i += 1024) dst8[i] = src8[i];
}

__global__ __launch_bounds__(1024, 1)
void k_moe_mapped_publish_partial_f16(
    const float* __restrict__ y_acc_f32,
    uint16_t* __restrict__ moe_mapped_slabs,
    uint32_t* __restrict__ moe_mapped_ready,
    uint32_t* __restrict__ moe_mapped_done,
    int gi, int M, int K_lat, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ dst = moe_mapped_slabs + (size_t(slot) * 7u + size_t(gi)) * size_t(slab_stride);
    const int n8 = (M * K_lat) >> 3;
    const float4* __restrict__ s4 = reinterpret_cast<const float4*>(y_acc_f32);
    uint4* __restrict__ d8 = reinterpret_cast<uint4*>(dst);
    for (int i = tid; i < n8; i += 1024) {
        const float4 v0 = s4[i * 2];
        const float4 v1 = s4[i * 2 + 1];
        uint4 o;
        o.x = uint32_t(d_f32_to_f16_core(v0.x)) | (uint32_t(d_f32_to_f16_core(v0.y)) << 16);
        o.y = uint32_t(d_f32_to_f16_core(v0.z)) | (uint32_t(d_f32_to_f16_core(v0.w)) << 16);
        o.z = uint32_t(d_f32_to_f16_core(v1.x)) | (uint32_t(d_f32_to_f16_core(v1.y)) << 16);
        o.w = uint32_t(d_f32_to_f16_core(v1.z)) | (uint32_t(d_f32_to_f16_core(v1.w)) << 16);
        d8[i] = o;
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&moe_mapped_ready[gi * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&moe_mapped_done[gi * 16],  epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// Fused canonical 10-slot FP32 reduction + 16-byte uint4 FP16 mapped PCIe publish (-168 launches & HBM2 bounces/step!)
// Optional 80-bit slot_mask3 skips ~80% empty slots and eliminates hipMemsetAsync(d_y_slots_f32)!
__global__ __launch_bounds__(1024, 1)
void k_moe_mapped_reduce_slots_publish_f16(
    const float* __restrict__ y_slots_f32,
    uint16_t* __restrict__ moe_mapped_slabs,
    uint32_t* __restrict__ moe_mapped_ready,
    uint32_t* __restrict__ moe_mapped_done,
    const int32_t* __restrict__ slot_mask3,
    int gi, int M, int K_lat, int slab_stride, uint32_t epoch)
{
    const int tid = int(threadIdx.x);
    const uint32_t slot = (epoch - 1u) & 15u;
    uint16_t* __restrict__ dst = moe_mapped_slabs + (size_t(slot) * 7u + size_t(gi)) * size_t(slab_stride);
    const int row_u4 = K_lat >> 3;
    const int n8 = M * row_u4;
    uint4* __restrict__ d8 = reinterpret_cast<uint4*>(dst);
    const uint32_t m0 = slot_mask3 ? uint32_t(slot_mask3[0]) : 0xFFFFFFFFu;
    const uint32_t m1 = slot_mask3 ? uint32_t(slot_mask3[1]) : 0xFFFFFFFFu;
    const uint32_t m2 = slot_mask3 ? uint32_t(slot_mask3[2]) : 0xFFFFFFFFu;
    for (int idx = tid; idx < n8; idx += 1024) {
        const int m = idx / row_u4;
        const int col8 = (idx - m * row_u4) << 3;
        const float* __restrict__ base = y_slots_f32 + size_t(m) * 10u * size_t(K_lat) + size_t(col8);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        const int bit0 = m * 10;
        #pragma unroll 1
        for (int s = 0; s < 10; ++s) {
            const int bit = bit0 + s;
            const uint32_t w = (bit < 32) ? m0 : ((bit < 64) ? m1 : m2);
            if (!((w >> (bit & 31)) & 1u)) continue;
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(base + size_t(s) * size_t(K_lat));
            const float4 v0 = p4[0];
            const float4 v1 = p4[1];
            a0 += v0.x; a1 += v0.y; a2 += v0.z; a3 += v0.w;
            a4 += v1.x; a5 += v1.y; a6 += v1.z; a7 += v1.w;
        }
        uint4 o;
        o.x = uint32_t(d_f32_to_f16_core(a0)) | (uint32_t(d_f32_to_f16_core(a1)) << 16);
        o.y = uint32_t(d_f32_to_f16_core(a2)) | (uint32_t(d_f32_to_f16_core(a3)) << 16);
        o.z = uint32_t(d_f32_to_f16_core(a4)) | (uint32_t(d_f32_to_f16_core(a5)) << 16);
        o.w = uint32_t(d_f32_to_f16_core(a6)) | (uint32_t(d_f32_to_f16_core(a7)) << 16);
        d8[idx] = o;
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&moe_mapped_ready[gi * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&moe_mapped_done[gi * 16],  epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// 4-Row-Parallel 16-Byte uint4 Vectorized Mapped Merge + RMSNorm (64 B LDS, 0 scratch, 140x fewer PCIe reads!)
__global__ __launch_bounds__(1024, 1)
void k_moe_mapped_merge_rmsnorm_to_bf16(
    const float*    __restrict__ own_f32,
    const uint16_t* __restrict__ moe_mapped_slabs,
    const uint32_t* __restrict__ moe_mapped_ready,
    uint32_t*       __restrict__ moe_mapped_done,
    const uint16_t* __restrict__ p150_bf16_or_null,
    const uint16_t* __restrict__ norm_w_bf16,
    uint16_t*       __restrict__ out_bf16,
    int own_gi, int n_gpus, int M, int H, int slab_stride, uint32_t epoch, float eps, int wait_p150_mapped)
{
    __shared__ float s_wsum[4][4];
    const int tid = int(threadIdx.x);

    if (tid == 0) {
        for (int g = 0; g < n_gpus; ++g) {
            if (g == own_gi) continue;
            while (__hip_atomic_load(&moe_mapped_ready[g * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        if (wait_p150_mapped != 0) {
            while (__hip_atomic_load(&moe_mapped_ready[6 * 16], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    const uint32_t slot = (epoch - 1u) & 15u;
    const uint16_t* __restrict__ slot_base = moe_mapped_slabs + size_t(slot) * 7u * size_t(slab_stride);
    const uint16_t* __restrict__ p150_ptr = p150_bf16_or_null;
    if (wait_p150_mapped != 0) {
        const uint16_t* __restrict__ s6 = slot_base + 6u * size_t(slab_stride);
        p150_ptr = (s6[16384u] != 0u) ? s6 : nullptr;
    }

    const int grp  = tid >> 8;       // 0..3 -> 4 rows processed in parallel!
    const int lt   = tid & 255;      // 0..255 -> 256 uint4s == 2048 halfs in ONE vectorized burst!
    const int warp = lt >> 6;        // 0..3
    const int lane = lt & 63;        // 0..63
    const int n_iters = (M + 3) >> 2;
    const uint4* __restrict__ nw4 = reinterpret_cast<const uint4*>(norm_w_bf16);

    for (int it = 0; it < n_iters; ++it) {
        const int m = it * 4 + grp;
        const bool active = (m < M);
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        float v4 = 0.0f, v5 = 0.0f, v6 = 0.0f, v7 = 0.0f;
        if (active) {
            const size_t row_off = size_t(m) * size_t(H);
            const size_t col8 = size_t(lt) * 8u;
            if (own_f32 != nullptr) {
                const float4* __restrict__ o4 = reinterpret_cast<const float4*>(own_f32 + row_off + col8);
                const float4 a = o4[0], b = o4[1];
                v0 = a.x; v1 = a.y; v2 = a.z; v3 = a.w;
                v4 = b.x; v5 = b.y; v6 = b.z; v7 = b.w;
            }
            for (int g = 0; g < n_gpus; ++g) {
                if (g == own_gi) continue;
                const uint4 u = reinterpret_cast<const uint4*>(slot_base + size_t(g) * size_t(slab_stride) + row_off)[lt];
                v0 += d_f16_to_f32_core(uint16_t(u.x & 0xFFFFu));
                v1 += d_f16_to_f32_core(uint16_t(u.x >> 16));
                v2 += d_f16_to_f32_core(uint16_t(u.y & 0xFFFFu));
                v3 += d_f16_to_f32_core(uint16_t(u.y >> 16));
                v4 += d_f16_to_f32_core(uint16_t(u.z & 0xFFFFu));
                v5 += d_f16_to_f32_core(uint16_t(u.z >> 16));
                v6 += d_f16_to_f32_core(uint16_t(u.w & 0xFFFFu));
                v7 += d_f16_to_f32_core(uint16_t(u.w >> 16));
            }
            if (p150_ptr != nullptr) {
                const uint4 u = reinterpret_cast<const uint4*>(p150_ptr + row_off)[lt];
                v0 += d_bf16_to_f32(uint16_t(u.x & 0xFFFFu));
                v1 += d_bf16_to_f32(uint16_t(u.x >> 16));
                v2 += d_bf16_to_f32(uint16_t(u.y & 0xFFFFu));
                v3 += d_bf16_to_f32(uint16_t(u.y >> 16));
                v4 += d_bf16_to_f32(uint16_t(u.z & 0xFFFFu));
                v5 += d_bf16_to_f32(uint16_t(u.z >> 16));
                v6 += d_bf16_to_f32(uint16_t(u.w & 0xFFFFu));
                v7 += d_bf16_to_f32(uint16_t(u.w >> 16));
            }
            float ss = fmaf(v0, v0, fmaf(v1, v1, fmaf(v2, v2, fmaf(v3, v3,
                       fmaf(v4, v4, fmaf(v5, v5, fmaf(v6, v6, v7 * v7)))))));
            ss = d_warp_sum64(ss);
            if (lane == 0) s_wsum[grp][warp] = ss;
        }
        __syncthreads();
        if (active) {
            const float tot_ss = (s_wsum[grp][0] + s_wsum[grp][1]) + (s_wsum[grp][2] + s_wsum[grp][3]);
            const float inv = rsqrtf(tot_ss * (1.0f / float(H)) + eps);
            const uint4 nw = nw4[lt];
            uint4 o;
            o.x = uint32_t(d_f32_to_bf16(v0 * inv * d_bf16_to_f32(uint16_t(nw.x & 0xFFFFu)))) |
                 (uint32_t(d_f32_to_bf16(v1 * inv * d_bf16_to_f32(uint16_t(nw.x >> 16)))) << 16);
            o.y = uint32_t(d_f32_to_bf16(v2 * inv * d_bf16_to_f32(uint16_t(nw.y & 0xFFFFu)))) |
                 (uint32_t(d_f32_to_bf16(v3 * inv * d_bf16_to_f32(uint16_t(nw.y >> 16)))) << 16);
            o.z = uint32_t(d_f32_to_bf16(v4 * inv * d_bf16_to_f32(uint16_t(nw.z & 0xFFFFu)))) |
                 (uint32_t(d_f32_to_bf16(v5 * inv * d_bf16_to_f32(uint16_t(nw.z >> 16)))) << 16);
            o.w = uint32_t(d_f32_to_bf16(v6 * inv * d_bf16_to_f32(uint16_t(nw.w & 0xFFFFu)))) |
                 (uint32_t(d_f32_to_bf16(v7 * inv * d_bf16_to_f32(uint16_t(nw.w >> 16)))) << 16);
            reinterpret_cast<uint4*>(out_bf16 + size_t(m) * size_t(H))[lt] = o;
        }
        __syncthreads();
    }
    if (tid == 0) {
        __hip_atomic_store(&moe_mapped_done[own_gi * 16], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

void launch_moe_mapped_publish_xrot_topk(
    const uint16_t* d_x_rot_f16, const int32_t* d_topk_ids, const float* d_topk_wts,
    uint16_t* d_moe_mapped_slabs, uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_moe_mapped_publish_xrot_topk, dim3(1), dim3(1024), 0, stream,
                       d_x_rot_f16, d_topk_ids, d_topk_wts, d_moe_mapped_slabs,
                       d_moe_mapped_ready, d_moe_mapped_done, M, K_lat, slab_stride, epoch);
}

void launch_moe_mapped_wait_xrot_copy(
    uint16_t* d_x_rot_dst, const uint16_t* d_moe_mapped_slabs, const uint32_t* d_moe_mapped_ready,
    int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_moe_mapped_wait_xrot_copy, dim3(1), dim3(1024), 0, stream,
                       d_x_rot_dst, d_moe_mapped_slabs, d_moe_mapped_ready, M, K_lat, slab_stride, epoch);
}

void launch_moe_mapped_publish_partial_f16(
    const float* d_y_acc_f32, uint16_t* d_moe_mapped_slabs,
    uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    int gi, int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_moe_mapped_publish_partial_f16, dim3(1), dim3(1024), 0, stream,
                       d_y_acc_f32, d_moe_mapped_slabs, d_moe_mapped_ready, d_moe_mapped_done,
                       gi, M, K_lat, slab_stride, epoch);
}

void launch_moe_mapped_reduce_slots_publish_f16(
    const float* d_y_slots_f32, uint16_t* d_moe_mapped_slabs,
    uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    int gi, int M, int K_lat, int slab_stride, uint32_t epoch, hipStream_t stream,
    const int32_t* d_slot_mask3) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_moe_mapped_reduce_slots_publish_f16, dim3(1), dim3(1024), 0, stream,
                       d_y_slots_f32, d_moe_mapped_slabs, d_moe_mapped_ready, d_moe_mapped_done,
                       d_slot_mask3, gi, M, K_lat, slab_stride, epoch);
}

void launch_moe_mapped_merge_rmsnorm_to_bf16(
    const float* d_own_f32, const uint16_t* d_moe_mapped_slabs,
    const uint32_t* d_moe_mapped_ready, uint32_t* d_moe_mapped_done,
    const uint16_t* d_p150_bf16_or_null, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_bf16, int own_gi, int n_gpus, int M, int H,
    int slab_stride, uint32_t epoch, float eps, hipStream_t stream,
    int wait_p150_mapped) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_moe_mapped_merge_rmsnorm_to_bf16, dim3(1), dim3(1024), 0, stream,
                       d_own_f32, d_moe_mapped_slabs, d_moe_mapped_ready, d_moe_mapped_done,
                       d_p150_bf16_or_null, d_norm_w_bf16, d_out_bf16,
                       own_gi, n_gpus, M, H, slab_stride, epoch, eps, wait_p150_mapped);
}

// ============================================================================
// Wave-11 Omni Turbo Pillar III (D3.3): 60-CU Split-K DSA Sparse MQA Decode
// Strict Invariants 1..4:
//   1. Worker CTA (blockIdx.x=s, blockIdx.y=m, blockIdx.z=hg) initializes m_part=-INF,
//      l_part=0, o_part=0 and emits UNNORMALIZED o_part without touching attn_sink;
//      1-wave combine folds attn_sink[h] strictly ONCE.
//   2. Combine guards empty splits with `if (l_s > 0.0f)` before expf(m_s - m_fin) (0 NaN).
//   3. Inside every K_tile=8 iteration (q_pos = pos0 + m), lanes with k >= k_end, idx < 0,
//      or idx > q_pos clamp gather index to 0 before global_load_dwordx4 and force score=-INF, p=0.
//   4. Adaptive S = (M <= 2) ? 32 : 16 inside existing w.d_tmp_a (0 B new hipMalloc).
// ============================================================================
template <bool kQIsF16, bool kKvIsF16, bool kIdxIsU16>
__global__ __launch_bounds__(256, 2)
void k_dsa_sparse_mqa_decode_tp4_h16_splitk(
    const uint16_t* __restrict__ q_in,
    const uint16_t* __restrict__ kv_mla,
    const void*     __restrict__ sel_idx,
    const int32_t*  __restrict__ sel_cnts,
    float*          __restrict__ o_part_f32,
    float*          __restrict__ m_part_f32,
    float*          __restrict__ l_part_f32,
    int H_total, int stride_sel, int32_t pos0, int S)
{
    __shared__ uint4 s_Q[16][73];   // 18,688 B
    __shared__ uint4 s_KV[8][73];   //  9,344 B (total 28,768 B <= 32 KiB -> 2 WGs/CU!)
    __shared__ union {
        float    s_S[16][8];
        uint32_t s_P[16][4];
    } u;
    __shared__ int32_t s_idx[8];
    __shared__ float   s_m[16];
    __shared__ float   s_l[16];
    __shared__ float   s_alpha[16];

    const int s     = int(blockIdx.x);
    const int m     = int(blockIdx.y);
    const int hg    = int(blockIdx.z);
    const int h0    = hg * 16;
    const int tid   = int(threadIdx.x);
    const int valid = sel_cnts[m];
    const int chunk = (valid + S - 1) / S;
    const int k_beg = s * chunk;
    const int k_end = (k_beg + chunk < valid) ? (k_beg + chunk) : valid;

    if (k_beg >= k_end) {
        if (tid < 16) {
            const size_t ml_idx = (size_t(m) * size_t(H_total) + size_t(h0 + tid)) * size_t(S) + size_t(s);
            m_part_f32[ml_idx] = -INFINITY;
            l_part_f32[ml_idx] = 0.0f;
        }
        return;
    }

    // 1. Preload 16 heads x 576 Q into s_Q[0..15][0..71] as FP16 uint4
    {
        const int q_h    = tid >> 4;
        const int q_lane = tid & 15;
        const uint4* __restrict__ q4 = reinterpret_cast<const uint4*>(
            q_in + (size_t(m) * size_t(H_total) + size_t(h0 + q_h)) * 576u);
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int u4_i = q_lane + c * 16;
            if (u4_i < 72) {
                const uint4 qv = q4[u4_i];
                if constexpr (kQIsF16) s_Q[q_h][u4_i] = qv;
                else                   s_Q[q_h][u4_i] = d_core_bf16_u4_to_f16_u4(qv);
            }
        }
    }
    // Invariant 1: initialize m_part = -INFINITY, l_part = 0.0f (do NOT touch attn_sink!)
    if (tid < 16) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    float o_acc[4][8];
    #pragma unroll
    for (int hi = 0; hi < 4; ++hi) {
        #pragma unroll
        for (int di = 0; di < 8; ++di) o_acc[hi][di] = 0.0f;
    }

    const uint16_t* __restrict__ idx_u16 = static_cast<const uint16_t*>(sel_idx) + size_t(m) * size_t(stride_sel);
    const int32_t*  __restrict__ idx_i32 = static_cast<const int32_t*>(sel_idx)  + size_t(m) * size_t(stride_sel);
    const int32_t q_pos = (pos0 >= 0) ? (pos0 + m) : 0x7FFFFFFF;

    #pragma unroll 1
    for (int kb = k_beg; kb < k_end; kb += 8) {
        // Invariant 3: validate k < k_end, idx >= 0, idx <= q_pos
        if (tid < 8) {
            int32_t raw_idx = -1;
            if (kb + tid < k_end) {
                raw_idx = kIdxIsU16 ? int32_t(idx_u16[kb + tid]) : idx_i32[kb + tid];
                if (raw_idx < 0 || raw_idx > q_pos) raw_idx = -1;
            }
            s_idx[tid] = raw_idx;
        }
        __syncthreads();

        // Coalesced 128-bit K load with Invariant 3 clamp to 0 before global_load_dwordx4
        {
            const int k_row  = tid >> 5;
            const int k_lane = tid & 31;
            const int32_t raw_kpos  = s_idx[k_row];
            const int32_t safe_kpos = (raw_kpos >= 0) ? raw_kpos : 0;
            const uint4* __restrict__ k_src = reinterpret_cast<const uint4*>(
                kv_mla + size_t(safe_kpos) * 576u);
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                const int u4_i = k_lane + c * 32;
                if (u4_i < 72) {
                    const uint4 v4 = k_src[u4_i];
                    if (raw_kpos >= 0) {
                        if constexpr (kKvIsF16) s_KV[k_row][u4_i] = v4;
                        else                    s_KV[k_row][u4_i] = d_core_bf16_u4_to_f16_u4(v4);
                    } else {
                        s_KV[k_row][u4_i] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
        }
        __syncthreads();

        // 2x2 Register-Tiled uint4 (ds_read_b128) QK over 576 dims
        float sc00 = 0.0f, sc01 = 0.0f, sc10 = 0.0f, sc11 = 0.0f;
        {
            const int ds   = tid & 7;
            const int tile = tid >> 3;
            const int hq2  = (tile & 7) * 2;
            const int kq2  = (tile >> 3) * 2;
            #pragma unroll 1
            for (int j = 0; j < 9; ++j) {
                const int j_idx = ds + j * 8;
                const uint4 k0 = s_KV[kq2 + 0][j_idx];
                const uint4 k1 = s_KV[kq2 + 1][j_idx];
                const uint4 q0 = s_Q[hq2 + 0][j_idx];
                const uint4 q1 = s_Q[hq2 + 1][j_idx];
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.x), d_u32_to_core_f16x2(k0.x), sc00, false);
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.y), d_u32_to_core_f16x2(k0.y), sc00, false);
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.z), d_u32_to_core_f16x2(k0.z), sc00, false);
                sc00 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.w), d_u32_to_core_f16x2(k0.w), sc00, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.x), d_u32_to_core_f16x2(k1.x), sc01, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.y), d_u32_to_core_f16x2(k1.y), sc01, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.z), d_u32_to_core_f16x2(k1.z), sc01, false);
                sc01 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q0.w), d_u32_to_core_f16x2(k1.w), sc01, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.x), d_u32_to_core_f16x2(k0.x), sc10, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.y), d_u32_to_core_f16x2(k0.y), sc10, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.z), d_u32_to_core_f16x2(k0.z), sc10, false);
                sc10 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.w), d_u32_to_core_f16x2(k0.w), sc10, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.x), d_u32_to_core_f16x2(k1.x), sc11, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.y), d_u32_to_core_f16x2(k1.y), sc11, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.z), d_u32_to_core_f16x2(k1.z), sc11, false);
                sc11 = __builtin_amdgcn_fdot2(d_u32_to_core_f16x2(q1.w), d_u32_to_core_f16x2(k1.w), sc11, false);
            }
            sc00 += __shfl_xor(sc00, 1, 64); sc00 += __shfl_xor(sc00, 2, 64); sc00 += __shfl_xor(sc00, 4, 64);
            sc01 += __shfl_xor(sc01, 1, 64); sc01 += __shfl_xor(sc01, 2, 64); sc01 += __shfl_xor(sc01, 4, 64);
            sc10 += __shfl_xor(sc10, 1, 64); sc10 += __shfl_xor(sc10, 2, 64); sc10 += __shfl_xor(sc10, 4, 64);
            sc11 += __shfl_xor(sc11, 1, 64); sc11 += __shfl_xor(sc11, 2, 64); sc11 += __shfl_xor(sc11, 4, 64);
            if (ds == 0) {
                u.s_S[hq2 + 0][kq2 + 0] = sc00 * 0.0625f;
                u.s_S[hq2 + 0][kq2 + 1] = sc01 * 0.0625f;
                u.s_S[hq2 + 1][kq2 + 0] = sc10 * 0.0625f;
                u.s_S[hq2 + 1][kq2 + 1] = sc11 * 0.0625f;
            }
        }
        __syncthreads();

        // Online softmax with strict NaN-free empty-tile & Invariant 3 guards
        {
            uint32_t p01 = 0u;
            float m_new = -INFINITY, ls = 0.0f, alpha = 0.0f;
            const int sm_h = tid >> 3;
            const int sm_k = tid & 7;
            if (tid < 128) {
                const bool key_ok = (kb + sm_k < k_end) && (s_idx[sm_k] >= 0);
                const float sv = key_ok ? u.s_S[sm_h][sm_k] : -INFINITY;
                float lm = sv;
                lm = fmaxf(lm, __shfl_xor(lm, 1, 64));
                lm = fmaxf(lm, __shfl_xor(lm, 2, 64));
                lm = fmaxf(lm, __shfl_xor(lm, 4, 64));

                const float m_old = s_m[sm_h];
                const float l_old = s_l[sm_h];
                m_new = fmaxf(m_old, lm);
                alpha = (l_old > 0.0f) ? __expf(m_old - m_new) : 0.0f;

                const float e0_raw = (key_ok && m_new > -INFINITY) ? __expf(sv - m_new) : 0.0f;
                const uint16_t e0_h = d_f32_to_f16_core(e0_raw);
                const float e0 = d_f16_to_f32_core(e0_h);
                const uint32_t e1_h = __shfl_xor(uint32_t(e0_h), 1, 64);
                ls = e0;
                ls += __shfl_xor(ls, 1, 64);
                ls += __shfl_xor(ls, 2, 64);
                ls += __shfl_xor(ls, 4, 64);
                p01 = uint32_t(e0_h) | (e1_h << 16);
            }
            __syncthreads();
            if (tid < 128) {
                if ((sm_k & 1) == 0) u.s_P[sm_h][sm_k >> 1] = p01;
                if (sm_k == 0) {
                    s_m[sm_h]     = m_new;
                    s_l[sm_h]     = fmaf(s_l[sm_h], alpha, ls);
                    s_alpha[sm_h] = alpha;
                }
            }
        }
        __syncthreads();

        // Rescale o_acc[4][8] + 0-Bank-Conflict uint4 PV
        {
            const int h_base = (tid >> 6) * 4;
            const int lane   = tid & 63;
            #pragma unroll
            for (int hi = 0; hi < 4; ++hi) {
                const float a = s_alpha[h_base + hi];
                #pragma unroll
                for (int di = 0; di < 8; ++di) o_acc[hi][di] *= a;
            }
            #pragma unroll 1
            for (int kp = 0; kp < 4; ++kp) {
                const int k0 = kp * 2;
                const uint4 a = s_KV[k0][lane];
                const uint4 b = s_KV[k0 + 1][lane];
                const core_f16x2_t v0 = d_u32_to_core_f16x2((a.x & 0xFFFFu) | (b.x << 16));
                const core_f16x2_t v1 = d_u32_to_core_f16x2((a.x >> 16) | (b.x & 0xFFFF0000u));
                const core_f16x2_t v2 = d_u32_to_core_f16x2((a.y & 0xFFFFu) | (b.y << 16));
                const core_f16x2_t v3 = d_u32_to_core_f16x2((a.y >> 16) | (b.y & 0xFFFF0000u));
                const core_f16x2_t v4 = d_u32_to_core_f16x2((a.z & 0xFFFFu) | (b.z << 16));
                const core_f16x2_t v5 = d_u32_to_core_f16x2((a.z >> 16) | (b.z & 0xFFFF0000u));
                const core_f16x2_t v6 = d_u32_to_core_f16x2((a.w & 0xFFFFu) | (b.w << 16));
                const core_f16x2_t v7 = d_u32_to_core_f16x2((a.w >> 16) | (b.w & 0xFFFF0000u));

                #pragma unroll
                for (int hi = 0; hi < 4; ++hi) {
                    const core_f16x2_t pv = d_u32_to_core_f16x2(u.s_P[h_base + hi][kp]);
                    o_acc[hi][0] = __builtin_amdgcn_fdot2(pv, v0, o_acc[hi][0], false);
                    o_acc[hi][1] = __builtin_amdgcn_fdot2(pv, v1, o_acc[hi][1], false);
                    o_acc[hi][2] = __builtin_amdgcn_fdot2(pv, v2, o_acc[hi][2], false);
                    o_acc[hi][3] = __builtin_amdgcn_fdot2(pv, v3, o_acc[hi][3], false);
                    o_acc[hi][4] = __builtin_amdgcn_fdot2(pv, v4, o_acc[hi][4], false);
                    o_acc[hi][5] = __builtin_amdgcn_fdot2(pv, v5, o_acc[hi][5], false);
                    o_acc[hi][6] = __builtin_amdgcn_fdot2(pv, v6, o_acc[hi][6], false);
                    o_acc[hi][7] = __builtin_amdgcn_fdot2(pv, v7, o_acc[hi][7], false);
                }
            }
        }
        __syncthreads();
    }

    // Invariant 1: emit UNNORMALIZED o_part + (m_part, l_part) without touching attn_sink!
    {
        const int h_base = (tid >> 6) * 4;
        const int lane   = tid & 63;
        #pragma unroll
        for (int hi = 0; hi < 4; ++hi) {
            const int h_loc = h_base + hi;
            const size_t ml_idx = (size_t(m) * size_t(H_total) + size_t(h0 + h_loc)) * size_t(S) + size_t(s);
            if (lane == 0) {
                m_part_f32[ml_idx] = s_m[h_loc];
                l_part_f32[ml_idx] = s_l[h_loc];
            }
            float4* dst4 = reinterpret_cast<float4*>(o_part_f32 + ml_idx * 512u + size_t(lane) * 8u);
            dst4[0] = make_float4(o_acc[hi][0], o_acc[hi][1], o_acc[hi][2], o_acc[hi][3]);
            dst4[1] = make_float4(o_acc[hi][4], o_acc[hi][5], o_acc[hi][6], o_acc[hi][7]);
        }
    }
}

// 1-Wave Combine Epilogue: grid = dim3(H_total, M), 64 lanes, 8 floats/lane in registers, 0 LDS, 0 scratch!
template <bool kOutIsF16>
__global__ __launch_bounds__(64, 8)
void k_dsa_sparse_mqa_splitk_combine_tp4_h16(
    const float* __restrict__ o_part_f32,
    const float* __restrict__ m_part_f32,
    const float* __restrict__ l_part_f32,
    const float* __restrict__ attn_sink,
    uint16_t*    __restrict__ o_out,
    int H_total, int S)
{
    const int h    = int(blockIdx.x);
    const int m    = int(blockIdx.y);
    const int lane = int(threadIdx.x); // 0..63 -> 8 dims [lane*8 .. lane*8+7]

    const size_t ml_base = (size_t(m) * size_t(H_total) + size_t(h)) * size_t(S);
    const float sink = attn_sink[h];

    // Pass 1: global max across attn_sink (once!) and all non-empty splits (Invariant 2 guard!)
    float m_fin = sink;
    for (int s = 0; s < S; ++s) {
        const float ls = l_part_f32[ml_base + size_t(s)];
        if (ls > 0.0f) {
            m_fin = fmaxf(m_fin, m_part_f32[ml_base + size_t(s)]);
        }
    }

    // Pass 2: fold attn_sink strictly ONCE + accumulate rescaled o_part strictly for ls > 0.0f
    float l_fin = __expf(sink - m_fin);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float ls = l_part_f32[ml_base + size_t(s)];
        if (ls > 0.0f) {
            const float w = __expf(m_part_f32[ml_base + size_t(s)] - m_fin);
            l_fin = fmaf(ls, w, l_fin);
            const float4* op4 = reinterpret_cast<const float4*>(
                o_part_f32 + (ml_base + size_t(s)) * 512u + size_t(lane) * 8u);
            const float4 v0 = op4[0];
            const float4 v1 = op4[1];
            a0 = fmaf(v0.x, w, a0); a1 = fmaf(v0.y, w, a1);
            a2 = fmaf(v0.z, w, a2); a3 = fmaf(v0.w, w, a3);
            a4 = fmaf(v1.x, w, a4); a5 = fmaf(v1.y, w, a5);
            a6 = fmaf(v1.z, w, a6); a7 = fmaf(v1.w, w, a7);
        }
    }

    const float inv_l = (l_fin > 0.0f) ? (1.0f / l_fin) : 0.0f;
    uint4 out4;
    if constexpr (kOutIsF16) {
        out4.x = d_pack_f32x2_to_f16x2_u32(a0 * inv_l, a1 * inv_l);
        out4.y = d_pack_f32x2_to_f16x2_u32(a2 * inv_l, a3 * inv_l);
        out4.z = d_pack_f32x2_to_f16x2_u32(a4 * inv_l, a5 * inv_l);
        out4.w = d_pack_f32x2_to_f16x2_u32(a6 * inv_l, a7 * inv_l);
    } else {
        out4.x = uint32_t(d_f32_to_bf16(a0 * inv_l)) | (uint32_t(d_f32_to_bf16(a1 * inv_l)) << 16);
        out4.y = uint32_t(d_f32_to_bf16(a2 * inv_l)) | (uint32_t(d_f32_to_bf16(a3 * inv_l)) << 16);
        out4.z = uint32_t(d_f32_to_bf16(a4 * inv_l)) | (uint32_t(d_f32_to_bf16(a5 * inv_l)) << 16);
        out4.w = uint32_t(d_f32_to_bf16(a6 * inv_l)) | (uint32_t(d_f32_to_bf16(a7 * inv_l)) << 16);
    }
    reinterpret_cast<uint4*>(o_out + (size_t(m) * size_t(H_total) + size_t(h)) * 512u)[lane] = out4;
}

void launch_dsa_sparse_mqa_decode_splitk(
    const uint16_t* d_q_in, const uint16_t* d_kv_mla, const float* d_attn_sink,
    const void* d_sel_idx, const int32_t* d_sel_cnts, void* d_scratch, uint16_t* d_o_out,
    int M, int n_hgroups, int stride_sel, int32_t pos0,
    bool q_is_f16, bool kv_is_f16, bool idx_is_u16, bool out_is_f16, hipStream_t stream)
{
    if (M <= 0 || n_hgroups <= 0) return;
    const int H_total = n_hgroups * 16;
    // Invariant 4: adaptive S = (M <= 2) ? 32 : 16 -> 32..128 CTAs (16h) or 128..384 CTAs (64h)!
    const int S = (M <= 2) ? 32 : 16;
    const size_t ml_elems = size_t(M) * size_t(H_total) * size_t(S);
    const size_t o_elems  = ml_elems * 512u;
    float* d_o_part = static_cast<float*>(d_scratch);
    float* d_m_part = d_o_part + o_elems;
    float* d_l_part = d_m_part + ml_elems;

    const dim3 grid_w{uint32_t(S), uint32_t(M), uint32_t(n_hgroups)};
    if (q_is_f16 && kv_is_f16 && idx_is_u16) {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_decode_tp4_h16_splitk<true, true, true>), grid_w, dim3(256), 0, stream,
                           d_q_in, d_kv_mla, d_sel_idx, d_sel_cnts, d_o_part, d_m_part, d_l_part, H_total, stride_sel, pos0, S);
    } else if (q_is_f16 && kv_is_f16 && !idx_is_u16) {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_decode_tp4_h16_splitk<true, true, false>), grid_w, dim3(256), 0, stream,
                           d_q_in, d_kv_mla, d_sel_idx, d_sel_cnts, d_o_part, d_m_part, d_l_part, H_total, stride_sel, pos0, S);
    } else if (q_is_f16 && !kv_is_f16 && !idx_is_u16) {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_decode_tp4_h16_splitk<true, false, false>), grid_w, dim3(256), 0, stream,
                           d_q_in, d_kv_mla, d_sel_idx, d_sel_cnts, d_o_part, d_m_part, d_l_part, H_total, stride_sel, pos0, S);
    } else if (!q_is_f16 && !kv_is_f16 && !idx_is_u16) {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_decode_tp4_h16_splitk<false, false, false>), grid_w, dim3(256), 0, stream,
                           d_q_in, d_kv_mla, d_sel_idx, d_sel_cnts, d_o_part, d_m_part, d_l_part, H_total, stride_sel, pos0, S);
    } else {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_decode_tp4_h16_splitk<false, true, false>), grid_w, dim3(256), 0, stream,
                           d_q_in, d_kv_mla, d_sel_idx, d_sel_cnts, d_o_part, d_m_part, d_l_part, H_total, stride_sel, pos0, S);
    }

    const dim3 grid_c{uint32_t(H_total), uint32_t(M)};
    if (out_is_f16) {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_splitk_combine_tp4_h16<true>), grid_c, dim3(64), 0, stream,
                           d_o_part, d_m_part, d_l_part, d_attn_sink, d_o_out, H_total, S);
    } else {
        hipLaunchKernelGGL((k_dsa_sparse_mqa_splitk_combine_tp4_h16<false>), grid_c, dim3(64), 0, stream,
                           d_o_part, d_m_part, d_l_part, d_attn_sink, d_o_out, H_total, S);
    }
}

// ============================================================================
// Wave-11 Omni Turbo Pillar III (D3.3): Direct-BF16 uint4 + v_dot2_f32_f16 FlashIndexer (M <= 8)
// 180 persistent CTAs (3 CTAs/CU x 60 CUs), 16,640 B LDS, 0 scratch, >800 GB/s HBM2!
// ============================================================================
__global__ __launch_bounds__(256, 3)
void k_flashindexer_decode_m8_splitk(
    const uint16_t* __restrict__ q_sel_bf16,
    const float*    __restrict__ w_sel_f32,
    const uint16_t* __restrict__ ik_cache_bf16,
    float*          __restrict__ scores_batch_f32,
    int32_t pos0, int t0, int B_sc, int score_stride)
{
    __shared__ uint4 s_q[8 * 8 * 16]; // 16,384 B (8 tokens x 8 heads x 16 uint4 = 128 FP16)
    __shared__ float s_w[8 * 8];      //    256 B (total 16,640 B -> 3 CTAs/CU!)

    const int tid = int(threadIdx.x);
    const int q_u4 = B_sc * 8 * 16;
    const uint4* __restrict__ q_src4 =
        reinterpret_cast<const uint4*>(q_sel_bf16 + size_t(t0) * 8u * 128u);
    for (int i = tid; i < q_u4; i += 256) {
        s_q[i] = d_core_bf16_u4_to_f16_u4(q_src4[i]);
    }
    if (tid < B_sc * 8) {
        s_w[tid] = w_sel_f32[size_t(t0) * 8u + size_t(tid)];
    }
    __syncthreads();

    const int g        = tid & 3;  // 0..3 within DPP quad (each lane loads 4 uint4 = 32 dims!)
    const int q_in_blk = tid >> 2; // 0..63 history tokens per block iteration
    const int min_hist = (pos0 + t0 >= 127) ? (pos0 + t0 - 127) : 0;
    const int max_pos  = pos0 + t0 + B_sc - 1;
    const int max_hist = (max_pos >= 127) ? (max_pos - 127) : 0;
    constexpr float kWScale = 0.011048543456039806f;

    const int stride_tok = int(gridDim.x) * 64;
    for (int tok = int(blockIdx.x) * 64 + q_in_blk; tok < max_hist; tok += stride_tok) {
        // 4 coalesced 128-bit loads (global_load_dwordx4) + in-register BF16->FP16 ONCE across all B_sc x 8 heads!
        const uint4* __restrict__ kt4 =
            reinterpret_cast<const uint4*>(ik_cache_bf16 + size_t(tok) * 128u);
        const uint4 k0 = d_core_bf16_u4_to_f16_u4(kt4[g +  0]);
        const uint4 k1 = d_core_bf16_u4_to_f16_u4(kt4[g +  4]);
        const uint4 k2 = d_core_bf16_u4_to_f16_u4(kt4[g +  8]);
        const uint4 k3 = d_core_bf16_u4_to_f16_u4(kt4[g + 12]);

        const bool all_valid = (min_hist > 2048 && tok < min_hist);
        for (int bi = 0; bi < B_sc; ++bi) {
            if (!all_valid) {
                const int n_hist = (pos0 + t0 + bi >= 127) ? (pos0 + t0 + bi - 127) : 0;
                if (n_hist <= 2048 || tok >= n_hist) continue;
            }
            const uint4* __restrict__ qb4 = s_q + bi * 8 * 16;
            const float* __restrict__ wb  = s_w + bi * 8;
            float total = 0.0f;

            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const uint4* __restrict__ qh4 = qb4 + h * 16;
                float dp = 0.0f;
                dp = d_core_dot8_f16_u4_fdot2(qh4[g +  0], k0, dp);
                dp = d_core_dot8_f16_u4_fdot2(qh4[g +  4], k1, dp);
                dp = d_core_dot8_f16_u4_fdot2(qh4[g +  8], k2, dp);
                dp = d_core_dot8_f16_u4_fdot2(qh4[g + 12], k3, dp);

                dp += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(dp), 0x4E, 0xf, 0xf, true));
                dp += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(dp), 0xB1, 0xf, 0xf, true));
                if (g == 0) {
                    const float pos_dp = (dp > 0.0f) ? dp : 0.0f;
                    total = fmaf(wb[h] * kWScale, pos_dp, total);
                }
            }
            if (g == 0) {
                scores_batch_f32[size_t(bi) * size_t(score_stride) + size_t(tok)] = total;
            }
        }
    }
}

void launch_flashindexer_decode_m8_splitk(
    const uint16_t* d_q_sel_bf16, const float* d_w_sel_f32,
    const uint16_t* d_ik_cache_bf16, float* d_scores_f32,
    int32_t pos0, int t0, int B_sc, int score_stride, hipStream_t stream)
{
    if (B_sc <= 0 || B_sc > 8 || score_stride <= 0) return;
    const int max_pos  = pos0 + t0 + B_sc - 1;
    const int max_hist = (max_pos >= 127) ? (max_pos - 127) : 0;
    if (max_hist <= 2048) return;
    const int blocks = (max_hist + 63) / 64;
    const int grid_x = (blocks < 180) ? blocks : 180;
    hipLaunchKernelGGL(k_flashindexer_decode_m8_splitk, dim3(uint32_t(grid_x)), dim3(256), 0, stream,
                       d_q_sel_bf16, d_w_sel_f32, d_ik_cache_bf16, d_scores_f32, pos0, t0, B_sc, score_stride);
}

// ============================================================================
// Wave-11 Omni Turbo Pillar III (D3.4): Zero-Sync On-GPU DSpark Markov+MultiMax from GPU prev_tok
// ============================================================================
__global__ __launch_bounds__(256, 4)
void k_dspark_markov_bias_multimax_from_tok_f32(
    const uint16_t* __restrict__ markov_w1_T_mapped,
    const int32_t*  __restrict__ d_prev_tok_ptr,
    const uint16_t* __restrict__ w2_VxR_f16,
    const float*    __restrict__ base_logits,
    const float*    __restrict__ mm_r,
    const float*    __restrict__ mm_t,
    float*          __restrict__ out_logits,
    int V, int R)
{
    __shared__ uint4 s_me_f16[128];
    const int tid = int(threadIdx.x);
    int32_t pt = d_prev_tok_ptr[0];
    if (pt < 0) pt = 0;
    if (pt >= V) pt = V - 1;
    const uint16_t* __restrict__ me_bf16 = markov_w1_T_mapped + size_t(pt) * size_t(R);
    const int R8 = R >> 3;
    const bool fast_u4 = ((R & 7) == 0 && R8 <= 128);

    if (fast_u4) {
        const uint4* __restrict__ me_u4 = reinterpret_cast<const uint4*>(me_bf16);
        for (int j = tid; j < R8; j += 256) {
            s_me_f16[j] = d_core_bf16_u4_to_f16_u4(me_u4[j]);
        }
        __syncthreads();
    }

    const int sub  = tid >> 5;
    const int lane = tid & 31;
    constexpr int kRowsPerSub = 4;
    const int v_base = int(blockIdx.x) * (8 * kRowsPerSub) + sub;

    #pragma unroll 1
    for (int ri = 0; ri < kRowsPerSub; ++ri) {
        const int v = v_base + ri * 8;
        if (v >= V) break;

        const uint16_t* __restrict__ w_row = w2_VxR_f16 + size_t(v) * size_t(R);
        float acc = 0.0f;
        if (fast_u4) {
            const uint4* __restrict__ w_u4 = reinterpret_cast<const uint4*>(w_row);
            for (int j = lane; j < R8; j += 32) {
                acc = d_core_dot8_f16_u4_fdot2(s_me_f16[j], w_u4[j], acc);
            }
        } else {
            for (int r = lane; r < R; r += 32) {
                const float mf = d_f16_to_f32_core(d_f32_to_f16_core(d_bf16_to_f32(me_bf16[r])));
                acc = fmaf(d_f16_to_f32_core(w_row[r]), mf, acc);
            }
        }
        acc = d_core_reduce32_dpp(acc);

        if (lane == 0) {
            const float bx = base_logits[v];
            if (isinf(bx) && bx < 0.0f) {
                out_logits[v] = bx;
            } else {
                float x = bx + acc;
                if (mm_r != nullptr && mm_t != nullptr) {
                    const float ax = fabsf(x);
                    if (ax > mm_r[0]) {
                        const float s = (x < 0.0f) ? -1.0f : 1.0f;
                        float r = ax;
                        if (r <= mm_r[1])      r = mm_r[0] + (r - mm_r[0]) * mm_t[0];
                        else if (r <= mm_r[2]) r = mm_r[0] + (mm_r[1] - mm_r[0]) * mm_t[0] + (r - mm_r[1]) * mm_t[1];
                        else if (r <= mm_r[3]) r = mm_r[0] + (mm_r[1] - mm_r[0]) * mm_t[0] + (mm_r[2] - mm_r[1]) * mm_t[1] + (r - mm_r[2]) * mm_t[2];
                        else                   r = mm_r[0] + (mm_r[1] - mm_r[0]) * mm_t[0] + (mm_r[2] - mm_r[1]) * mm_t[1] + (mm_r[3] - mm_r[2]) * mm_t[2] + (r - mm_r[3]) * mm_t[3];
                        x = s * r;
                    }
                }
                out_logits[v] = x;
            }
        }
    }
}

void launch_dspark_markov_bias_multimax_from_tok_f32(
    const uint16_t* d_markov_w1_T_mapped, const int32_t* d_prev_tok_ptr,
    const uint16_t* d_w2_VxR_f16, const float* d_base_logits_f32,
    const float* d_mm_ranges4, const float* d_mm_ts4,
    float* d_out_logits_f32, int V, int R, hipStream_t stream)
{
    if (V <= 0 || R <= 0) return;
    const uint32_t gx = uint32_t((V + 31) / 32);
    hipLaunchKernelGGL(k_dspark_markov_bias_multimax_from_tok_f32, dim3(gx), dim3(256), 0, stream,
                       d_markov_w1_T_mapped, d_prev_tok_ptr, d_w2_VxR_f16,
                       d_base_logits_f32, d_mm_ranges4, d_mm_ts4, d_out_logits_f32, V, R);
}

} // namespace ernie55
