// SPDX-License-Identifier: Apache-2.0
// tools/bench_single_layer_gfx906.cpp
//
// Standalone pure-C++17 / HIP single-layer prefill micro-benchmark for the
// ERNIE-5.5-Flash-Preview v1.0.0-preview-hlut4-192g WHT128-HLUT4 gfx906 (MI50)
// baseline. Zero Python. It links against the shipped v1.0.0 kernel objects
// (build/ernie55_hlut4_kernels.o + build/ernie55_core_kernels.o), so every
// number printed below comes out of the production kernels themselves.
//
// Measured on GPU[0] (HIP_VISIBLE_DEVICES=0), per phase, at M = 1024 and 2048:
//   * MI50_MoE     : 512-expert routing distribution, 8 local HLUT4 experts
//                    (local_eid = e & 7) -> realistic 20-40 tokens/expert,
//                    un-chunked M, calling
//                      launch_hlut4_fdot2_grouped_gu_situ
//                      launch_wht128_grouped_f32_to_f16
//                      launch_hlut4_fdot2_grouped_down_slot
//                      launch_reduce_moe_slots_f32
//   * Dense        : qkv_comp [2560x4096], q_up [8192x1024], o_proj [4096x1024],
//                    fc2_lat [4096x512], shared [2048x4096] comparing the v1.0.0
//                    HS_HPA path (f32 C/D, f32 compute) against the aligned
//                    Type_HH_HPA path (f16 C/D, f32 compute)
//   * HCA SWA-128  : launch_hca_prefill_swa_merge_sink at pos0 = 16384
//   * DSA Top-2048 : launch_dsa_select_topk2048_wave6 vs
//                    launch_dsa_select_topk2048_gpu (0 index mismatches)
//
// Strict VRAM budget: 256 MiB total on GPU[0] (8 experts = 49.5 MiB), so it is
// safe to run while the :8052 server is live. Each phase frees its buffers
// before the next phase starts; the peak is enforced and reported.
//
// Build:
//   g++ -std=c++17 -O3 -march=native -fopenmp -pthread -Wall -Wextra -Werror
//       -Wno-unused-parameter -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -Isrc -Imi50
//       tools/bench_single_layer_gfx906.cpp build/ernie55_hlut4_kernels.o
//       build/ernie55_core_kernels.o -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib
//       -lamdhip64 -lrocblas -o bin/bench_single_layer_gfx906
// Run:
//   HIP_VISIBLE_DEVICES=0 ./bin/bench_single_layer_gfx906

#include "ernie55_hlut4_kernels.hpp"
#include "ernie55_core_kernels.hpp"
#include "ernie55_config.hpp"
#include "ernie55_hlut4_pack.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace ernie55;

// ---------------------------------------------------------------------------
// Constants / strict VRAM budget
// ---------------------------------------------------------------------------
static constexpr size_t kMaxVramBytes = size_t(256) << 20; // 256 MiB hard cap on GPU[0]
static constexpr int    kLocalExperts = 8;                 // local_eid = e & 7
static constexpr size_t kExpertBytes  = size_t(kHlut4ExpertBytes);
static constexpr int    kStrideSel    = kDsaIndexTopK + kSlidingWindow; // 2176

static void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("HIP failure at ") + what + ": " + hipGetErrorString(e));
    }
}
#define HIP_CHECK(x) hip_ok((x), #x)

static void rocblas_ok(rocblas_status s, const char* what) {
    if (s != rocblas_status_success) {
        throw std::runtime_error(std::string("rocBLAS failure at ") + what + ": " + rocblas_status_to_string(s));
    }
}
#define ROC_CHECK(x) rocblas_ok((x), #x)

struct VramBudget {
    size_t used = 0;
    size_t peak = 0;
    std::vector<void*> live;

    ~VramBudget() { release(); }

    void* grab_raw(size_t n) {
        if (n == 0) return nullptr;
        if (used + n > kMaxVramBytes) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "VRAM budget exceeded: %zu B requested, %zu B used, cap %zu B",
                          n, used, kMaxVramBytes);
            throw std::runtime_error(msg);
        }
        void* p = nullptr;
        HIP_CHECK(hipMalloc(&p, n));
        used += n;
        peak = std::max(peak, used);
        live.push_back(p);
        return p;
    }
    template <typename T>
    T* grab(size_t count) {
        return reinterpret_cast<T*>(grab_raw(count * sizeof(T)));
    }
    void release() {
        for (void* p : live) {
            if (p) (void)hipFree(p);
        }
        live.clear();
        used = 0;
    }
};

// ---------------------------------------------------------------------------
// Deterministic host RNG (splitmix64)
// ---------------------------------------------------------------------------
static inline uint64_t sm64(uint64_t& st) noexcept {
    uint64_t z = (st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline float rnd_unit(uint64_t& st) noexcept {
    return float((sm64(st) >> 40) & 0xFFFFFFu) / float(0x1000000u);
}

// ---------------------------------------------------------------------------
// hipEvent timing -> exact microseconds
// ---------------------------------------------------------------------------
template <typename Body>
static double time_us(hipStream_t stream, int iters, Body&& body) {
    hipEvent_t a = nullptr, b = nullptr;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    HIP_CHECK(hipEventRecord(a, stream));
    for (int i = 0; i < iters; ++i) body(i);
    HIP_CHECK(hipEventRecord(b, stream));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return double(ms) * 1000.0; // ms -> us
}

// ---------------------------------------------------------------------------
// Result table
// ---------------------------------------------------------------------------
struct BenchRow {
    std::string phase;
    std::string item;
    int    M = 0;
    int    iters = 0;
    double total_us = 0.0;
    double tflops = 0.0;
    double hbm_gbs = 0.0;
    double max_abs_diff = 0.0;
    double speedup = 0.0;
    std::string note;
};

static std::vector<BenchRow> g_rows;
static void add_row(BenchRow r) { g_rows.push_back(std::move(r)); }

static void print_table() {
    std::printf("\n");
    std::printf("===========================================================================================================================\n");
    std::printf(" ERNIE-5.5 single-layer gfx906 prefill table (v1.0.0-preview-hlut4-192g baseline, 8 local HLUT4 experts, 256 MiB cap)\n");
    std::printf("===========================================================================================================================\n");
    std::printf("%-10s %-32s %6s %6s %11s %11s %9s %11s %13s %9s  %s\n",
                "phase", "item", "M", "iters", "total_us", "per_iter_ms", "TFLOPS", "HBM2_GB/s",
                "max_abs_diff", "speedup", "note");
    std::printf("---------------------------------------------------------------------------------------------------------------------------\n");
    for (const auto& r : g_rows) {
        char tf[32], hb[32], sp[32];
        if (r.tflops > 0.0) std::snprintf(tf, sizeof(tf), "%9.3f", r.tflops);
        else std::snprintf(tf, sizeof(tf), "%9s", "n/a");
        if (r.hbm_gbs > 0.0) std::snprintf(hb, sizeof(hb), "%11.2f", r.hbm_gbs);
        else std::snprintf(hb, sizeof(hb), "%11s", "n/a");
        if (r.speedup > 0.0) std::snprintf(sp, sizeof(sp), "%8.3fx", r.speedup);
        else std::snprintf(sp, sizeof(sp), "%9s", "n/a");
        std::printf("%-10s %-32s %6d %6d %11.1f %11.4f %s %s %13.3e %s  %s\n",
                    r.phase.c_str(), r.item.c_str(), r.M, r.iters, r.total_us,
                    r.total_us / 1000.0 / double(std::max(1, r.iters)), tf, hb,
                    r.max_abs_diff, sp, r.note.c_str());
    }
    std::printf("===========================================================================================================================\n");
}

// ===========================================================================
// MoE phase
// ===========================================================================
struct MoeHostExperts {
    std::vector<Hlut4Tile32x32> w1;  // [8, 4096]
    std::vector<Hlut4Tile32x32> w3;  // [8, 4096]
    std::vector<Hlut4Tile32x32> w2;  // [8, 4096]
    std::vector<uint64_t> signs_in;     // 32
    std::vector<uint64_t> signs_inter;  // 32
};

static void moe_host_experts_synthetic(MoeHostExperts& h, int layer_idx) {
    const size_t per = size_t(kExpProjTiles);
    h.w1.resize(size_t(kLocalExperts) * per);
    h.w3.resize(size_t(kLocalExperts) * per);
    h.w2.resize(size_t(kLocalExperts) * per);
    h.signs_in = make_ernie55_signs_in(layer_idx);
    h.signs_inter = make_ernie55_signs_inter(layer_idx);
    const uint16_t s = f32_to_fp16_rne(0.05f);
    for (int e = 0; e < kLocalExperts; ++e) {
        for (size_t t = 0; t < per; ++t) {
            Hlut4Tile32x32* dst[3] = {&h.w1[size_t(e) * per + t], &h.w3[size_t(e) * per + t],
                                      &h.w2[size_t(e) * per + t]};
            for (int p = 0; p < 3; ++p) {
                uint64_t st = 0x9E3779B97F4A7C15ULL * (t + 1u) ^
                              (0x1000003u * (uint64_t(e) + 1u)) ^ (0x5DEECE66DULL * (uint64_t(p) + 1u));
                for (int i = 0; i < 128; ++i) dst[p]->q_nib4[i] = uint32_t(sm64(st));
                for (int i = 0; i < 8; ++i) dst[p]->scales_f16[i] = s;
            }
        }
    }
}

static void moe_host_experts_from_file(MoeHostExperts& h, const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open HLUT4 expert shard: " + path);
    Ernie55Hlut4LayerHeader hdr{};
    if (::pread(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr)) ||
        hdr.magic != kErnie55Hlut4ShardMagic || hdr.version != kErnie55Hlut4ShardVersion) {
        ::close(fd);
        throw std::runtime_error("Corrupt HLUT4 header in " + path);
    }
    if (int(hdr.num_experts) < kLocalExperts || int(hdr.latent_dim) != kLatentDim ||
        int(hdr.moe_inter_dim) != kMoeInterSize || int(hdr.tiles_per_proj) != int(kExpProjTiles)) {
        ::close(fd);
        throw std::runtime_error("Unexpected HLUT4 shard geometry in " + path);
    }
    h.signs_in.assign(hdr.signs_in, hdr.signs_in + kLatentDim / 64);
    h.signs_inter.assign(hdr.signs_inter, hdr.signs_inter + kMoeInterSize / 64);
    const size_t per = size_t(kExpProjTiles);
    h.w1.resize(size_t(kLocalExperts) * per);
    h.w3.resize(size_t(kLocalExperts) * per);
    h.w2.resize(size_t(kLocalExperts) * per);
    std::vector<Hlut4Tile32x32> raw(3u * per);
    for (int e = 0; e < kLocalExperts; ++e) {
        const off_t eoff = off_t(sizeof(hdr)) + off_t(e) * off_t(kExpertBytes);
        size_t got = 0;
        while (got < kExpertBytes) {
            const ssize_t r = ::pread(fd, reinterpret_cast<uint8_t*>(raw.data()) + got,
                                      kExpertBytes - got, eoff + off_t(got));
            if (r <= 0) {
                ::close(fd);
                throw std::runtime_error("Short read on " + path);
            }
            got += size_t(r);
        }
        std::memcpy(h.w1.data() + size_t(e) * per, raw.data() + 0u * per, per * sizeof(Hlut4Tile32x32));
        std::memcpy(h.w3.data() + size_t(e) * per, raw.data() + 1u * per, per * sizeof(Hlut4Tile32x32));
        std::memcpy(h.w2.data() + size_t(e) * per, raw.data() + 2u * per, per * sizeof(Hlut4Tile32x32));
    }
    ::close(fd);
}

struct MoeRouting {
    std::vector<ExpertDispatchGroup> groups;
    std::vector<int> row_group_begin; // [M+1]
    std::vector<int> row_groups;
    std::vector<int32_t> topk_ids;    // [M * kTopKExperts]
    std::vector<float>   topk_wts;    // [M * kTopKExperts]
    uint64_t assignments = 0;
    int max_c = 1;
};

// Deterministic 512-expert top-10 routing. Only global experts [0, kLocalExperts)
// are resident on this GPU (local_eid = e & 7), giving 20-40 tokens/expert.
static MoeRouting moe_build_routing(int M, uint64_t seed) {
    MoeRouting R;
    R.topk_ids.assign(static_cast<size_t>(M) * kTopKExperts, -1);
    R.topk_wts.assign(static_cast<size_t>(M) * kTopKExperts, 0.0f);
    std::vector<std::vector<std::pair<int,int>>> per_expert(static_cast<size_t>(kLocalExperts));
    std::vector<std::array<int, kTopKExperts>> sel(static_cast<size_t>(M));
    for (int r = 0; r < M; ++r) {
        uint64_t st = seed ^ (0x9E3779B97F4A7C15ULL * uint64_t(r + 1));
        for (int k = 0; k < kTopKExperts; ++k) {
            int e = -1;
            for (int tries = 0; tries < 64; ++tries) {
                const int cand = int(sm64(st) % uint64_t(kNumRoutedExperts));
                bool dup = false;
                for (int j = 0; j < k; ++j) if (sel[size_t(r)][size_t(j)] == cand) dup = true;
                if (!dup) { e = cand; break; }
            }
            if (e < 0) e = int(sm64(st) % uint64_t(kNumRoutedExperts));
            sel[size_t(r)][size_t(k)] = e;
            R.topk_ids[static_cast<size_t>(r) * kTopKExperts + static_cast<size_t>(k)] = e;
            R.topk_wts[static_cast<size_t>(r) * kTopKExperts + static_cast<size_t>(k)] = 0.1f;
        }
    }
    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < kTopKExperts; ++k) {
            const int e = sel[size_t(r)][size_t(k)];
            if (e >= kLocalExperts) continue; // served by another GPU
            per_expert[size_t(e)].push_back({r, k});
            ++R.assignments;
        }
    }
    for (int e = 0; e < kLocalExperts; ++e) {
        const auto& rows = per_expert[size_t(e)];
        for (size_t i = 0; i < rows.size(); i += size_t(kMaxGroupTokens)) {
            ExpertDispatchGroup g{};
            g.expert = e;
            g.count = int(std::min<size_t>(size_t(kMaxGroupTokens), rows.size() - i));
            for (int c = 0; c < g.count; ++c) {
                g.rows[size_t(c)] = rows[i + size_t(c)].first;
                g.weights[size_t(c)] = 0.1f;
                g.slots[size_t(c)] = rows[i + size_t(c)].second;
            }
            R.max_c = std::max(R.max_c, g.count);
            R.groups.push_back(g);
        }
    }
    std::vector<std::vector<int>> rg(static_cast<size_t>(M));
    for (size_t gi = 0; gi < R.groups.size(); ++gi) {
        for (int c = 0; c < R.groups[gi].count; ++c) {
            rg[size_t(R.groups[gi].rows[size_t(c)])].push_back(int(gi));
        }
    }
    R.row_group_begin.assign(size_t(M) + 1, 0);
    for (int r = 0; r < M; ++r) {
        R.row_group_begin[size_t(r)] = int(R.row_groups.size());
        for (int gi : rg[size_t(r)]) R.row_groups.push_back(gi);
    }
    R.row_group_begin[size_t(M)] = int(R.row_groups.size());
    return R;
}

// v1.0.0 CPU reference path: exact WHT128 + HLUT4 dequant + FP32 SiTU-GLU + weighted sum
static void moe_cpu_reference(const MoeHostExperts& he, const MoeRouting& R,
                              const std::vector<uint16_t>& x_bf16, int rows,
                              const std::vector<int>& row_ids,
                              const float lv[8], float beta, float lbeta, float* y_out) {
    const int K = kLatentDim, N = kMoeInterSize;
    #pragma omp parallel
    {
        std::vector<float> xrow(static_cast<size_t>(K)), xr(static_cast<size_t>(K));
        std::vector<float> z(static_cast<size_t>(N)), z_rot(static_cast<size_t>(N));
        #pragma omp for schedule(dynamic, 1)
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < K; ++k) xrow[size_t(k)] = bf16_to_f32(x_bf16[size_t(r) * size_t(K) + size_t(k)]);
            wht128_rows(xrow.data(), xr.data(), 1, K, he.signs_in);
            for (int k = 0; k < K; ++k) xr[size_t(k)] = fp16_to_f32(f32_to_fp16_rne(xr[size_t(k)]));
            float* y = y_out + size_t(r) * size_t(N);
            for (int d = 0; d < N; ++d) y[size_t(d)] = 0.0f;
            const int grow = row_ids[static_cast<size_t>(r)];
            for (int gii = R.row_group_begin[static_cast<size_t>(grow)];
                 gii < R.row_group_begin[static_cast<size_t>(grow) + 1]; ++gii) {
                const auto& g = R.groups[static_cast<size_t>(R.row_groups[static_cast<size_t>(gii)])];
                const int e = g.expert;
                int c = 0;
                while (c < g.count && g.rows[static_cast<size_t>(c)] != grow) ++c;
                if (c >= g.count) continue;
                const float w = g.weights[size_t(c)];
                const Hlut4Tile32x32* w1 = he.w1.data() + size_t(e) * size_t(kExpProjTiles);
                const Hlut4Tile32x32* w3 = he.w3.data() + size_t(e) * size_t(kExpProjTiles);
                const Hlut4Tile32x32* w2 = he.w2.data() + size_t(e) * size_t(kExpProjTiles);
                for (int n = 0; n < N; ++n) {
                    const Hlut4Tile32x32* t1 = w1 + size_t(n >> 5) * size_t(K >> 5);
                    const Hlut4Tile32x32* t3 = w3 + size_t(n >> 5) * size_t(K >> 5);
                    float ag = 0.0f, au = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        const int kt = k >> 5;
                        ag = fmaf(hlut4_dequant_elem(t1[kt], n & 31, k & 31, lv), xr[size_t(k)], ag);
                        au = fmaf(hlut4_dequant_elem(t3[kt], n & 31, k & 31, lv), xr[size_t(k)], au);
                    }
                    z[size_t(n)] = ernie55_situ_glu(ag, au, beta, lbeta);
                }
                wht128_rows(z.data(), z_rot.data(), 1, N, he.signs_inter);
                for (int k = 0; k < N; ++k) z_rot[size_t(k)] = fp16_to_f32(f32_to_fp16_rne(z_rot[size_t(k)]));
                for (int d = 0; d < N; ++d) {
                    const Hlut4Tile32x32* t2 = w2 + size_t(d >> 5) * size_t(N >> 5);
                    float acc = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        acc = fmaf(hlut4_dequant_elem(t2[k >> 5], d & 31, k & 31, lv), z_rot[size_t(k)], acc);
                    }
                    y[size_t(d)] = fmaf(acc, w, y[size_t(d)]);
                }
            }
        }
    }
}

struct MoeOptions {
    int  M = 1024;
    int  iters = 3;
    int  warmup = 1;
    bool synthetic = false;
    std::string ckpt_path = "checkpoints/active/experts_hlut4/hlut4_layer_01.bin";
    int  verify_rows = 4;
};

static void bench_moe(VramBudget& bud, hipStream_t stream, const MoeOptions& opt) {
    const int M = opt.M;
    const int K = kLatentDim, N = kMoeInterSize;

    MoeHostExperts he;
    if (opt.synthetic) moe_host_experts_synthetic(he, 1);
    else moe_host_experts_from_file(he, opt.ckpt_path);

    const auto lv32 = hlut4_levels_f32();
    float lv[8];
    uint16_t lv16[8];
    for (int i = 0; i < 8; ++i) {
        lv16[i] = f32_to_fp16_rne(lv32[size_t(i)]);
        lv[i] = fp16_to_f32(lv16[i]); // the GPU LUT holds FP16 levels
    }

    // 8 local HLUT4 experts = 49.5 MiB
    Hlut4Tile32x32* d_w1 = bud.grab<Hlut4Tile32x32>(size_t(kLocalExperts) * kExpProjTiles);
    Hlut4Tile32x32* d_w3 = bud.grab<Hlut4Tile32x32>(size_t(kLocalExperts) * kExpProjTiles);
    Hlut4Tile32x32* d_w2 = bud.grab<Hlut4Tile32x32>(size_t(kLocalExperts) * kExpProjTiles);
    HIP_CHECK(hipMemcpy(d_w1, he.w1.data(), he.w1.size() * sizeof(Hlut4Tile32x32), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w3, he.w3.data(), he.w3.size() * sizeof(Hlut4Tile32x32), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w2, he.w2.data(), he.w2.size() * sizeof(Hlut4Tile32x32), hipMemcpyHostToDevice));

    uint16_t* d_levels8 = bud.grab<uint16_t>(8);
    HIP_CHECK(hipMemcpy(d_levels8, lv16, sizeof(lv16), hipMemcpyHostToDevice));
    uint64_t* d_signs_in = bud.grab<uint64_t>(he.signs_in.size());
    HIP_CHECK(hipMemcpy(d_signs_in, he.signs_in.data(), he.signs_in.size() * sizeof(uint64_t), hipMemcpyHostToDevice));
    uint64_t* d_signs_inter = bud.grab<uint64_t>(he.signs_inter.size());
    HIP_CHECK(hipMemcpy(d_signs_inter, he.signs_inter.data(), he.signs_inter.size() * sizeof(uint64_t), hipMemcpyHostToDevice));

    // Deterministic 512-expert routing, un-chunked over all M rows
    MoeRouting R = moe_build_routing(M, 0x45524E35355F494EULL);
    const int ng = int(R.groups.size());
    ExpertDispatchGroup* d_groups = bud.grab<ExpertDispatchGroup>(size_t(std::max(1, ng)));
    if (ng > 0) {
        HIP_CHECK(hipMemcpy(d_groups, R.groups.data(), R.groups.size() * sizeof(ExpertDispatchGroup),
                            hipMemcpyHostToDevice));
    }

    std::vector<uint16_t> x_bf16(static_cast<size_t>(M) * static_cast<size_t>(K));
    {
        uint64_t st = 0x4535354D4F000000ULL;
        for (auto& v : x_bf16) v = f32_to_bf16_rne(0.5f * (rnd_unit(st) - 0.5f));
    }
    uint16_t* d_x_bf16 = bud.grab<uint16_t>(size_t(M) * size_t(K));
    HIP_CHECK(hipMemcpy(d_x_bf16, x_bf16.data(), x_bf16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    uint16_t* d_x_rot = bud.grab<uint16_t>(size_t(M) * size_t(K));
    launch_wht128_bf16_to_f16(d_x_bf16, d_signs_in, d_x_rot, M, K, stream);

    float* d_z = bud.grab<float>(size_t(std::max(1, ng)) * kMaxGroupTokens * size_t(N));
    uint16_t* d_z_rot = bud.grab<uint16_t>(size_t(std::max(1, ng)) * kMaxGroupTokens * size_t(N));
    float* d_y_slots = bud.grab<float>(size_t(M) * kTopKExperts * size_t(N));
    float* d_y_acc = bud.grab<float>(size_t(M) * size_t(N));
    HIP_CHECK(hipMemsetAsync(d_y_slots, 0, size_t(M) * kTopKExperts * size_t(N) * sizeof(float), stream));

    const int max_c = std::max(1, R.max_c);
    auto body = [&](int) {
        launch_hlut4_fdot2_grouped_gu_situ(d_x_rot, d_groups, ng, d_w1, d_w3, d_levels8, d_z,
                                           N, K, kSituBeta, kSituLinearBeta, stream);
        launch_wht128_grouped_f32_to_f16(d_z, d_groups, ng, d_signs_inter, d_z_rot, N, stream, max_c);
        launch_hlut4_fdot2_grouped_down_slot(d_z_rot, d_groups, ng, d_w2, d_levels8, d_y_slots,
                                             N, K, stream);
        launch_reduce_moe_slots_f32(d_y_slots, d_y_acc, M, N, stream);
    };

    for (int i = 0; i < std::max(1, opt.warmup); ++i) body(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    const double total_us = time_us(stream, std::max(1, opt.iters), body);

    // ---- verification against the v1.0.0 CPU reference path ----
    // Verify only rows that actually received a local expert (others are all-zero by construction)
    std::vector<int> vrows_list;
    for (int r = 0; r < M && int(vrows_list.size()) < std::max(0, opt.verify_rows); ++r) {
        if (R.row_group_begin[static_cast<size_t>(r) + 1] > R.row_group_begin[static_cast<size_t>(r)]) {
            vrows_list.push_back(r);
        }
    }
    const int vrows = int(vrows_list.size());
    double max_diff = 0.0;
    if (vrows > 0 && ng > 0) {
        std::vector<float> y_gpu(static_cast<size_t>(vrows) * static_cast<size_t>(N), 0.0f);
        std::vector<float> y_ref(static_cast<size_t>(vrows) * static_cast<size_t>(N), 0.0f);
        std::vector<uint16_t> x_sel(static_cast<size_t>(vrows) * static_cast<size_t>(K));
        for (int i = 0; i < vrows; ++i) {
            const int r = vrows_list[static_cast<size_t>(i)];
            HIP_CHECK(hipMemcpy(y_gpu.data() + static_cast<size_t>(i) * static_cast<size_t>(N),
                                d_y_acc + static_cast<size_t>(r) * static_cast<size_t>(N),
                                static_cast<size_t>(N) * sizeof(float), hipMemcpyDeviceToHost));
            std::memcpy(x_sel.data() + static_cast<size_t>(i) * static_cast<size_t>(K),
                        x_bf16.data() + static_cast<size_t>(r) * static_cast<size_t>(K),
                        static_cast<size_t>(K) * sizeof(uint16_t));
        }
        moe_cpu_reference(he, R, x_sel, vrows, vrows_list, lv, kSituBeta, kSituLinearBeta, y_ref.data());
        double ymag = 0.0;
        size_t argmax = 0;
        for (size_t i = 0; i < y_gpu.size(); ++i) {
            ymag = std::max(ymag, double(std::fabs(y_gpu[i])));
            if (double(std::fabs(y_gpu[i] - y_ref[i])) > max_diff) {
                max_diff = double(std::fabs(y_gpu[i] - y_ref[i]));
                argmax = i;
            }
        }
        (void)ymag;
        (void)argmax;
    }

    const double A = double(R.assignments);
    const double flops = A * 3.0 * 2.0 * double(K) * double(N);
    const double expert_bytes = double(ng) * double(kExpertBytes);
    const double act_bytes = A * double(K) * 2.0 + A * double(N) * 4.0 + A * double(N) * 2.0 +
                             A * double(N) * 4.0 + double(M) * double(N) * 4.0 + A * double(N) * 4.0;
    const double secs = total_us / 1e6;
    BenchRow r;
    r.phase = "MI50_MoE";
    r.item = "fdot2 grouped gu/wht/down";
    r.M = M;
    r.iters = std::max(1, opt.iters);
    r.total_us = total_us;
    r.tflops = flops / secs / 1e12;
    r.hbm_gbs = (expert_bytes + act_bytes) / secs / 1e9;
    r.max_abs_diff = max_diff;
    char note[128];
    std::snprintf(note, sizeof(note), "%d exp %.1f tok/exp %d grp", kLocalExperts,
                  double(R.assignments) / double(kLocalExperts), ng);
    r.note = note;
    add_row(std::move(r));
    std::printf("[moe]      M=%4d  experts=%d (%.1f MiB)  assign=%llu  tok/expert=%.1f  groups=%d  "
                "total=%.1f us  max_abs_diff=%.3e\n",
                M, kLocalExperts, double(kLocalExperts) * double(kExpertBytes) / double(1u << 20),
                (unsigned long long)R.assignments, double(R.assignments) / double(kLocalExperts), ng,
                total_us, max_diff);

    // ---- Commit 2: On-chip gfx906 LDS SRAM M=64 MoE + Direct Weighted FP32 atomicAdd (0 d_y_slots!) ----
    {
        std::vector<float> y_v100_all(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        HIP_CHECK(hipMemcpy(y_v100_all.data(), d_y_acc, y_v100_all.size() * sizeof(float), hipMemcpyDeviceToHost));

        const int max_pairs = M * kTopKExperts;
        const int max_tiles = std::max(64, max_pairs / 16 + kLocalExperts);
        int32_t* d_topk_ids = bud.grab<int32_t>(R.topk_ids.size());
        float*   d_topk_wts = bud.grab<float>(R.topk_wts.size());
        int16_t* d_g2l      = bud.grab<int16_t>(kNumRoutedExperts);
        int32_t* d_csr_cnt  = bud.grab<int32_t>(4);
        Ernie55MoeWorkTile*   d_tiles = bud.grab<Ernie55MoeWorkTile>(size_t(max_tiles));
        Ernie55MoeRoutedPair* d_pairs = bud.grab<Ernie55MoeRoutedPair>(size_t(max_pairs));

        std::vector<int16_t> h_g2l(kNumRoutedExperts, int16_t(-1));
        for (int e = 0; e < kLocalExperts; ++e) h_g2l[size_t(e)] = int16_t(e);
        HIP_CHECK(hipMemcpy(d_topk_ids, R.topk_ids.data(), R.topk_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_topk_wts, R.topk_wts.data(), R.topk_wts.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_g2l, h_g2l.data(), h_g2l.size() * sizeof(int16_t), hipMemcpyHostToDevice));

        // Reuse d_y_slots buffer (already allocated above!) as z_f16 for zero extra VRAM overhead
        uint16_t* d_z_f16 = reinterpret_cast<uint16_t*>(d_y_slots);
        auto body_c2 = [&](int) {
            HIP_CHECK(hipMemsetAsync(d_y_acc, 0, size_t(M) * size_t(N) * sizeof(float), stream));
            launch_ernie55_build_expert_csr_prefill(
                d_topk_ids, d_topk_wts, d_g2l, M, kLocalExperts, 64, max_tiles, max_pairs,
                d_csr_cnt, d_tiles, d_pairs, stream);
            launch_hlut4_prefill_situ_f16_m64(
                d_x_rot, d_csr_cnt, d_tiles, d_pairs, d_w1, d_w3, d_levels8, d_z_f16,
                N, K, kSituBeta, kSituLinearBeta, max_tiles, stream);
            launch_wht128_f16_inplace_prefill_pairs(
                d_z_f16, d_csr_cnt, d_signs_inter, N, max_pairs, stream);
            launch_hlut4_prefill_down_atomic_m64(
                d_z_f16, d_csr_cnt, d_tiles, d_pairs, d_w2, d_levels8, d_y_acc,
                N, K, max_tiles, stream);
        };
        for (int i = 0; i < std::max(1, opt.warmup); ++i) body_c2(i);
        HIP_CHECK(hipStreamSynchronize(stream));
        const double c2_us = time_us(stream, std::max(1, opt.iters), body_c2);

        std::vector<float> y_c2_all(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        HIP_CHECK(hipMemcpy(y_c2_all.data(), d_y_acc, y_c2_all.size() * sizeof(float), hipMemcpyDeviceToHost));
        double c2_max_diff = 0.0;
        for (size_t i = 0; i < y_c2_all.size(); ++i) {
            c2_max_diff = std::max(c2_max_diff, double(std::fabs(y_c2_all[i] - y_v100_all[i])));
        }
        int32_t h_cnt[2] = {};
        HIP_CHECK(hipMemcpy(h_cnt, d_csr_cnt, 2 * sizeof(int32_t), hipMemcpyDeviceToHost));

        const double c2_secs = c2_us / 1e6;
        const double c2_expert_bytes = double(h_cnt[0]) * double(kExpertBytes);
        const double c2_act_bytes = A * double(K) * 2.0 + A * double(N) * 4.0 + double(M) * double(N) * 4.0;
        BenchRow rc2;
        rc2.phase = "MI50_MoE";
        rc2.item = "Commit2 LDS M=64 atomic";
        rc2.M = M;
        rc2.iters = std::max(1, opt.iters);
        rc2.total_us = c2_us;
        rc2.tflops = flops / c2_secs / 1e12;
        rc2.hbm_gbs = (c2_expert_bytes + c2_act_bytes) / c2_secs / 1e9;
        rc2.max_abs_diff = c2_max_diff;
        rc2.speedup = total_us / std::max(1e-6, c2_us);
        char c2_note[128];
        std::snprintf(c2_note, sizeof(c2_note), "%d tiles %d pairs (0 d_y_slots)", h_cnt[0], h_cnt[1]);
        rc2.note = c2_note;
        add_row(std::move(rc2));
        std::printf("[moe-c2]   M=%4d  tiles=%d  pairs=%d  total=%.1f us  speedup=%.2fx  max_abs_diff_all=%.3e\n",
                    M, h_cnt[0], h_cnt[1], c2_us, total_us / std::max(1e-6, c2_us), c2_max_diff);

        // Variant B: BM=32 x BN=64 (3-4 WGs/CU, BK=64 on down) + Direct Weighted FP32 atomicAdd
        auto body_c2_m32 = [&](int) {
            HIP_CHECK(hipMemsetAsync(d_y_acc, 0, size_t(M) * size_t(N) * sizeof(float), stream));
            launch_ernie55_build_expert_csr_prefill(
                d_topk_ids, d_topk_wts, d_g2l, M, kLocalExperts, 32, max_tiles, max_pairs,
                d_csr_cnt, d_tiles, d_pairs, stream);
            launch_hlut4_prefill_situ_f16_m32(
                d_x_rot, d_csr_cnt, d_tiles, d_pairs, d_w1, d_w3, d_levels8, d_z_f16,
                N, K, kSituBeta, kSituLinearBeta, max_tiles, stream);
            launch_wht128_f16_inplace_prefill_pairs(
                d_z_f16, d_csr_cnt, d_signs_inter, N, max_pairs, stream);
            launch_hlut4_prefill_down_atomic_m32(
                d_z_f16, d_csr_cnt, d_tiles, d_pairs, d_w2, d_levels8, d_y_acc,
                N, K, max_tiles, stream);
        };
        for (int i = 0; i < std::max(1, opt.warmup); ++i) body_c2_m32(i);
        HIP_CHECK(hipStreamSynchronize(stream));
        const double c2_m32_us = time_us(stream, std::max(1, opt.iters), body_c2_m32);

        HIP_CHECK(hipMemcpy(y_c2_all.data(), d_y_acc, y_c2_all.size() * sizeof(float), hipMemcpyDeviceToHost));
        double c2_m32_diff = 0.0;
        for (size_t i = 0; i < y_c2_all.size(); ++i) {
            c2_m32_diff = std::max(c2_m32_diff, double(std::fabs(y_c2_all[i] - y_v100_all[i])));
        }
        HIP_CHECK(hipMemcpy(h_cnt, d_csr_cnt, 2 * sizeof(int32_t), hipMemcpyDeviceToHost));
        const double m32_secs = c2_m32_us / 1e6;
        BenchRow rc2b;
        rc2b.phase = "MI50_MoE";
        rc2b.item = "Commit2 LDS M=32 atomic";
        rc2b.M = M;
        rc2b.iters = std::max(1, opt.iters);
        rc2b.total_us = c2_m32_us;
        rc2b.tflops = flops / m32_secs / 1e12;
        rc2b.hbm_gbs = (double(h_cnt[0]) * double(kExpertBytes) + c2_act_bytes) / m32_secs / 1e9;
        rc2b.max_abs_diff = c2_m32_diff;
        rc2b.speedup = total_us / std::max(1e-6, c2_m32_us);
        std::snprintf(c2_note, sizeof(c2_note), "%d tiles %d pairs (0 d_y_slots)", h_cnt[0], h_cnt[1]);
        rc2b.note = c2_note;
        add_row(std::move(rc2b));
        std::printf("[moe-m32]  M=%4d  tiles=%d  pairs=%d  total=%.1f us  speedup=%.2fx  max_abs_diff_all=%.3e\n",
                    M, h_cnt[0], h_cnt[1], c2_m32_us, total_us / std::max(1e-6, c2_m32_us), c2_m32_diff);
    }
    bud.release();
}

// ===========================================================================
// Dense projection phase (HS_HPA vs Type_HH_HPA)
// ===========================================================================
struct DenseSpec {
    const char* name;
    int N;
    int K;
};

static const DenseSpec kDenseSpecs[] = {
    {"qkv_comp", 2560, 4096},
    {"q_up",     8192, 1024},
    {"o_proj",   4096, 1024},
    {"fc2_lat",  4096,  512},
    {"shared",   2048, 4096},
};

struct DenseOptions {
    int iters = 2;
    int warmup = 1;
    int strip = 512;
};

static void bench_dense(VramBudget& bud, hipStream_t stream, rocblas_handle blas,
                        const std::vector<int>& Ms, const DenseOptions& opt) {
    const float alpha = 1.0f, beta = 0.0f;
    const int Mmax = *std::max_element(Ms.begin(), Ms.end());

    for (const auto& spec : kDenseSpecs) {
        const int N = spec.N, K = spec.K;
        uint16_t* d_w16 = bud.grab<uint16_t>(size_t(N) * size_t(K));
        float* d_w32 = bud.grab<float>(size_t(N) * size_t(K));
        uint16_t* d_x16 = bud.grab<uint16_t>(size_t(Mmax) * size_t(K));
        float* d_x32 = bud.grab<float>(size_t(Mmax) * size_t(K));
        float* d_y32 = bud.grab<float>(size_t(opt.strip) * size_t(N));
        uint16_t* d_y16 = bud.grab<uint16_t>(size_t(opt.strip) * size_t(N));

        std::vector<uint16_t> h_w16(static_cast<size_t>(N) * static_cast<size_t>(K));
        std::vector<float> h_w32(static_cast<size_t>(N) * static_cast<size_t>(K));
        {
            uint64_t st = 0x9E3779B97F4A7C15ULL ^ (uint64_t(N) << 21) ^ uint64_t(K);
            for (size_t i = 0; i < h_w16.size(); ++i) {
                h_w16[i] = f32_to_fp16_rne(0.02f * (rnd_unit(st) - 0.5f));
                h_w32[i] = fp16_to_f32(h_w16[i]); // aligned: identical FP16 values in both paths
            }
        }
        HIP_CHECK(hipMemcpy(d_w16, h_w16.data(), h_w16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w32, h_w32.data(), h_w32.size() * sizeof(float), hipMemcpyHostToDevice));

        std::vector<uint16_t> h_x16(static_cast<size_t>(Mmax) * static_cast<size_t>(K));
        std::vector<float> h_x32(static_cast<size_t>(Mmax) * static_cast<size_t>(K));
        {
            uint64_t st = 0x4535354D4F000000ULL ^ (uint64_t(K) << 7);
            for (size_t i = 0; i < h_x32.size(); ++i) {
                const float v = 0.5f * (rnd_unit(st) - 0.5f);
                h_x32[i] = bf16_to_f32(f32_to_bf16_rne(v)); // BF16-rounded activation
                h_x16[i] = f32_to_fp16_rne(h_x32[i]);       // aligned FP16 copy of the same value
            }
        }

        for (int M : Ms) {
        const int strip = std::min(std::max(1, opt.strip), M);
        const int nstrip = (M + strip - 1) / strip;
        HIP_CHECK(hipMemcpy(d_x32, h_x32.data(), size_t(M) * size_t(K) * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_x16, h_x16.data(), size_t(M) * size_t(K) * sizeof(uint16_t), hipMemcpyHostToDevice));

        auto run_hs = [&](int s) {
            const int rows = std::min(strip, M - s * strip);
            const float* xs = d_x32 + size_t(s) * size_t(strip) * size_t(K);
            ROC_CHECK(rocblas_gemm_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                N, rows, K, &alpha,
                d_w32, rocblas_datatype_f32_r, K,
                xs, rocblas_datatype_f32_r, K, &beta,
                d_y32, rocblas_datatype_f32_r, N,
                d_y32, rocblas_datatype_f32_r, N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
        };
        auto run_hh = [&](int s) {
            const int rows = std::min(strip, M - s * strip);
            const uint16_t* xs = d_x16 + size_t(s) * size_t(strip) * size_t(K);
            ROC_CHECK(rocblas_gemm_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                N, rows, K, &alpha,
                d_w16, rocblas_datatype_f16_r, K,
                xs, rocblas_datatype_f16_r, K, &beta,
                d_y16, rocblas_datatype_f16_r, N,
                d_y16, rocblas_datatype_f16_r, N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
        };

        // ---- max_abs_diff pass (untimed, first strip only) ----
        double max_diff = 0.0;
        {
            std::vector<float> y32(static_cast<size_t>(strip) * static_cast<size_t>(N));
            std::vector<uint16_t> y16(static_cast<size_t>(strip) * static_cast<size_t>(N));
            for (int s = 0; s < std::min(nstrip, 1); ++s) {
                const int rows = std::min(strip, M - s * strip);
                run_hs(s);
                run_hh(s);
                HIP_CHECK(hipStreamSynchronize(stream));
                HIP_CHECK(hipMemcpy(y32.data(), d_y32, y32.size() * sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(y16.data(), d_y16, y16.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
                for (int i = 0; i < rows * N; ++i) {
                    max_diff = std::max(max_diff, double(std::fabs(y32[size_t(i)] - fp16_to_f32(y16[size_t(i)]))));
                }
            }
        }

        // ---- timed HS_HPA ----
        for (int i = 0; i < std::max(1, opt.warmup); ++i) {
            for (int s = 0; s < nstrip; ++s) run_hs(s);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        const double us_hs = time_us(stream, std::max(1, opt.iters),
                                     [&](int) { for (int s = 0; s < nstrip; ++s) run_hs(s); });

        // ---- timed Type_HH_HPA ----
        for (int i = 0; i < std::max(1, opt.warmup); ++i) {
            for (int s = 0; s < nstrip; ++s) run_hh(s);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        const double us_hh = time_us(stream, std::max(1, opt.iters),
                                     [&](int) { for (int s = 0; s < nstrip; ++s) run_hh(s); });

        const double flops = 2.0 * double(M) * double(N) * double(K);
        BenchRow r;
        r.phase = "Dense";
        r.item = std::string(spec.name) + " HS_HPA";
        r.M = M;
        r.iters = std::max(1, opt.iters);
        r.total_us = us_hs;
        r.tflops = flops / (us_hs / 1e6) / 1e12;
        r.max_abs_diff = max_diff;
        r.speedup = us_hs / us_hh;
        char note[96];
        std::snprintf(note, sizeof(note), "[%dx%d] vs HH speedup", N, K);
        r.note = note;
        add_row(std::move(r));

        BenchRow r2;
        r2.phase = "Dense";
        r2.item = std::string(spec.name) + " Type_HH_HPA";
        r2.M = M;
        r2.iters = std::max(1, opt.iters);
        r2.total_us = us_hh;
        r2.tflops = flops / (us_hh / 1e6) / 1e12;
        r2.max_abs_diff = max_diff;
        r2.note = "[f16 C/D, f32 compute]";
        add_row(std::move(r2));

        std::printf("[dense]    M=%4d  %-9s [%4dx%4d]  HS_HPA=%9.1f us  HH_HPA=%9.1f us  "
                    "speedup=%.3fx  max_abs_diff=%.3e\n",
                    M, spec.name, N, K, us_hs, us_hh, us_hs / us_hh, max_diff);
        } // for M
        bud.release();
    }
}

// ===========================================================================
// HCA SWA-128 prefill phase
// ===========================================================================
struct HcaOptions {
    int iters = 2;
    int warmup = 3;
    int n_heads = 16; // TP4 local heads (64 / 4)
    int pos0 = 16384;
};

static void bench_hca(VramBudget& bud, hipStream_t stream, rocblas_handle blas, int M, const HcaOptions& opt) {
    const int H = opt.n_heads;
    const int pos0 = opt.pos0;
    const int n_comp = std::max(1, pos0 / kHcaCompressRatio);

    uint16_t* d_q = bud.grab<uint16_t>(size_t(M) * size_t(H) * size_t(kHcaHeadDim));
    uint16_t* d_swa = bud.grab<uint16_t>(size_t(256) * size_t(kHcaHeadDim));
    uint16_t* d_kv = bud.grab<uint16_t>(size_t(M) * size_t(kHcaHeadDim));
    float* d_comp_o = bud.grab<float>(size_t(M) * size_t(H) * size_t(kHcaHeadDim));
    float* d_comp_m = bud.grab<float>(size_t(M) * size_t(H));
    float* d_comp_l = bud.grab<float>(size_t(M) * size_t(H));
    float* d_sink = bud.grab<float>(size_t(H));
    uint16_t* d_out = bud.grab<uint16_t>(size_t(M) * size_t(H) * size_t(kHcaHeadDim));
    uint16_t* d_out_hh = bud.grab<uint16_t>(size_t(M) * size_t(H) * size_t(kHcaHeadDim));
    const int n_pre = std::min(127, pos0);
    const int n_pad = n_pre + M + 256;
    constexpr int kSwaStrip = 128;
    uint16_t* d_kpack = bud.grab<uint16_t>(size_t(n_pad) * size_t(kHcaHeadDim));
    uint16_t* d_q_f16 = bud.grab<uint16_t>(size_t(kSwaStrip) * size_t(H) * size_t(kHcaHeadDim));
    uint16_t* d_probs = bud.grab<uint16_t>(size_t(kSwaStrip) * size_t(H) * 256u);
    uint16_t* d_y16   = bud.grab<uint16_t>(size_t(kSwaStrip) * size_t(H) * size_t(kHcaHeadDim));
    float* d_m_blk    = bud.grab<float>(size_t(kSwaStrip) * size_t(H));
    float* d_l_blk    = bud.grab<float>(size_t(kSwaStrip) * size_t(H));

    {
        uint64_t st = 0x48434100ULL;
        std::vector<uint16_t> h(static_cast<size_t>(M) * static_cast<size_t>(H) * static_cast<size_t>(kHcaHeadDim));
        for (auto& v : h) v = f32_to_bf16_rne(0.2f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_q, h.data(), h.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        h.resize(size_t(256) * size_t(kHcaHeadDim));
        for (auto& v : h) v = f32_to_bf16_rne(0.2f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_swa, h.data(), h.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        h.resize(size_t(M) * size_t(kHcaHeadDim));
        for (auto& v : h) v = f32_to_bf16_rne(0.2f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_kv, h.data(), h.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        std::vector<float> f(static_cast<size_t>(M) * static_cast<size_t>(H) * static_cast<size_t>(kHcaHeadDim));
        for (auto& v : f) v = 0.1f * (rnd_unit(st) - 0.5f);
        HIP_CHECK(hipMemcpy(d_comp_o, f.data(), f.size() * sizeof(float), hipMemcpyHostToDevice));
        f.resize(size_t(M) * size_t(H));
        for (auto& v : f) v = rnd_unit(st);
        HIP_CHECK(hipMemcpy(d_comp_m, f.data(), f.size() * sizeof(float), hipMemcpyHostToDevice));
        for (auto& v : f) v = 1.0f + rnd_unit(st);
        HIP_CHECK(hipMemcpy(d_comp_l, f.data(), f.size() * sizeof(float), hipMemcpyHostToDevice));
        f.resize(size_t(H));
        for (auto& v : f) v = -2.0f * rnd_unit(st);
        HIP_CHECK(hipMemcpy(d_sink, f.data(), f.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    auto body = [&](int) {
        launch_hca_prefill_swa_merge_sink(d_q, d_swa, d_kv, d_comp_o, d_comp_m, d_comp_l,
                                          d_sink, d_out, H, pos0, M, stream);
    };
    auto body_hh = [&](int) {
        const int base_pos = pos0 - n_pre;
        launch_hca_pack_swa_keys_f16(d_swa, d_kv, d_kpack, pos0, n_pre, M, n_pad, stream);
        const float alpha_hca = 0.04419417382415922f;
        const float alpha = 1.0f, beta = 0.0f;
        for (int m0 = 0; m0 < M; m0 += kSwaStrip) {
            const int B = std::min(kSwaStrip, M - m0);
            const int B_eff = ((B + 63) / 64) * 64;
            const int n_rows = B * H;
            const int n_rows_eff = B_eff * H;
            const size_t row_off = size_t(m0) * size_t(H);
            const int32_t q_pos0 = pos0 + m0;
            const int32_t k_start_pos = std::max(0, int(q_pos0) - 127);
            const int k_slot0 = k_start_pos - base_pos;
            const uint16_t* d_k_win = d_kpack + size_t(k_slot0) * 512u;
            launch_bf16_to_f16(d_q + row_off * 512u, d_q_f16, n_rows * 512, stream);
            if (n_rows < n_rows_eff) {
                HIP_CHECK(hipMemsetAsync(d_q_f16 + size_t(n_rows) * 512u, 0,
                                         size_t(n_rows_eff - n_rows) * 512u * sizeof(uint16_t), stream));
            }
            ROC_CHECK(rocblas_gemm_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                256, n_rows_eff, 512, &alpha_hca,
                d_k_win, rocblas_datatype_f16_r, 512,
                d_q_f16, rocblas_datatype_f16_r, 512,
                &beta,
                d_y16, rocblas_datatype_f16_r, 256,
                d_y16, rocblas_datatype_f16_r, 256,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            launch_hca_swa_softmax_f16_in_norm(d_y16, d_m_blk, d_l_blk, d_probs, B, H, q_pos0, k_start_pos, stream);
            if (n_rows < n_rows_eff) {
                HIP_CHECK(hipMemsetAsync(d_probs + size_t(n_rows) * 256u, 0,
                                         size_t(n_rows_eff - n_rows) * 256u * sizeof(uint16_t), stream));
            }
            ROC_CHECK(rocblas_gemm_ex(
                blas, rocblas_operation_none, rocblas_operation_none,
                512, n_rows_eff, 256, &alpha,
                d_k_win, rocblas_datatype_f16_r, 512,
                d_probs, rocblas_datatype_f16_r, 256,
                &beta,
                d_y16, rocblas_datatype_f16_r, 512,
                d_y16, rocblas_datatype_f16_r, 512,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            launch_hca_swa_finalize_merge_sink_f16(
                d_y16, d_m_blk, d_l_blk, d_comp_o, d_comp_m, d_comp_l,
                d_sink, d_out_hh, B, H, int(row_off), stream);
        }
    };

    for (int i = 0; i < std::max(1, opt.warmup); ++i) body(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    const double total_us = time_us(stream, std::max(1, opt.iters), body);

    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_hh(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    const double hh_us = time_us(stream, std::max(1, opt.iters), body_hh);

    // determinism + Commit 3 HH_HPA parity check vs exact BF16->FP32 reference
    std::vector<uint16_t> a(static_cast<size_t>(M) * static_cast<size_t>(H) * static_cast<size_t>(kHcaHeadDim)), b(a.size());
    HIP_CHECK(hipMemcpy(a.data(), d_out, a.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b.data(), d_out_hh, b.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    double hh_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) hh_diff = std::max(hh_diff, double(std::fabs(bf16_to_f32(a[i]) - bf16_to_f32(b[i]))));

    const double swa = double(M) * double(H) * double(kHcaHeadDim) * 128.0;
    const double comp = double(M) * double(H) * double(kHcaHeadDim) * double(n_comp);
    const double flops = 2.0 * (swa + comp) * 2.0; // QK + PV
    const double bytes = double(M) * double(H) * double(kHcaHeadDim) * 2.0 +
                         double(256) * double(kHcaHeadDim) * 2.0 +
                         double(M) * double(kHcaHeadDim) * 2.0 +
                         double(M) * double(H) * double(kHcaHeadDim) * 4.0 +
                         double(M) * double(H) * double(kHcaHeadDim) * 2.0;
    const double secs = total_us / 1e6;
    BenchRow r;
    r.phase = "HCA";
    r.item = "prefill_swa_merge_sink";
    r.M = M;
    r.iters = std::max(1, opt.iters);
    r.total_us = total_us;
    r.tflops = flops / secs / 1e12;
    r.hbm_gbs = bytes / secs / 1e9;
    r.max_abs_diff = 0.0;
    char note[96];
    std::snprintf(note, sizeof(note), "SWA-128 pos0=%d H=%d comp=%d", pos0, H, n_comp);
    r.note = note;
    add_row(std::move(r));

    BenchRow r_hh;
    r_hh.phase = "HCA";
    r_hh.item = "Commit3 SWA-128 HH_HPA";
    r_hh.M = M;
    r_hh.iters = std::max(1, opt.iters);
    r_hh.total_us = hh_us;
    r_hh.tflops = flops / (hh_us / 1e6) / 1e12;
    r_hh.hbm_gbs = bytes / (hh_us / 1e6) / 1e9;
    r_hh.max_abs_diff = hh_diff;
    r_hh.speedup = total_us / hh_us;
    r_hh.note = "Type_HH_HPA GEMM SWA-128";
    add_row(std::move(r_hh));
    std::printf("[hca]      M=%4d  heads=%d  pos0=%d  ref=%8.1f us  HH_HPA=%7.1f us  speedup=%.2fx  max_abs_diff=%.3e\n",
                M, H, pos0, total_us, hh_us, total_us / hh_us, hh_diff);
    bud.release();
}

// ===========================================================================
// DSA Top-2048 selector phase (wave6 vs gpu)
// ===========================================================================
struct DsaOptions {
    int iters = 2;
    int warmup = 1;
    int batch = 64; // rows per launch (wave6 sub-batch granularity)
    int pos0 = 16384;
};

static void bench_dsa(VramBudget& bud, hipStream_t stream, int M, int L, const DsaOptions& opt) {
    const int stride = std::max(L, opt.pos0 + M - kSlidingWindow + 1); // keep reads in range
    const int B = std::min(std::max(1, opt.batch), M);
    const int nbatch = (M + B - 1) / B;

    float* d_scores = bud.grab<float>(size_t(B) * size_t(stride));
    // sel_idx / sel_cnts are indexed by the GLOBAL row id t = t0 + bi in both selectors
    int32_t* d_idx_a = bud.grab<int32_t>(size_t(M) * size_t(kStrideSel));
    int32_t* d_idx_b = bud.grab<int32_t>(size_t(M) * size_t(kStrideSel));
    int32_t* d_cnt_a = bud.grab<int32_t>(size_t(M));
    int32_t* d_cnt_b = bud.grab<int32_t>(size_t(M));
    // g_hist1/2/3: B*32*2048/1024 u32 each, g_meta: B*8, g_blk_cnt: B*64 (pass4a uses bi*64+blk)
    const size_t scratch_u32 = size_t(B) * 32u * 2048u * 3u + size_t(B) * 32u * 1024u +
                               size_t(B) * 64u + 4096u;
    uint32_t* d_scratch = bud.grab<uint32_t>(scratch_u32);

    std::vector<float> h_scores(static_cast<size_t>(B) * static_cast<size_t>(stride));
    {
        uint64_t st = 0x44534100ULL ^ uint64_t(L);
        for (auto& v : h_scores) v = rnd_unit(st);
    }
    HIP_CHECK(hipMemcpy(d_scores, h_scores.data(), h_scores.size() * sizeof(float), hipMemcpyHostToDevice));

    auto run_wave6 = [&](int t0) {
        launch_dsa_select_topk2048_wave6(d_scores, d_scratch, d_idx_a, d_cnt_a,
                                         opt.pos0, t0, B, stride, kStrideSel, stream);
    };
    auto run_gpu = [&](int t0) {
        launch_dsa_select_topk2048_gpu(d_scores, d_idx_b, d_cnt_b,
                                       opt.pos0, t0, B, stride, kStrideSel, stream);
    };

    // warmup both
    for (int b = 0; b < nbatch; ++b) {
        run_wave6(b * B);
        run_gpu(b * B);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    const double us_wave6 = time_us(stream, std::max(1, opt.iters),
                                    [&](int) { for (int b = 0; b < nbatch; ++b) run_wave6(b * B); });
    const double us_gpu = time_us(stream, std::max(1, opt.iters),
                                  [&](int) { for (int b = 0; b < nbatch; ++b) run_gpu(b * B); });

    // ---- 0 index mismatches ----
    long long mismatches = 0;
    {
        std::vector<int32_t> ia(static_cast<size_t>(M) * static_cast<size_t>(kStrideSel)), ib(ia.size());
        std::vector<int32_t> ca(static_cast<size_t>(M)), cb(ca.size());
        for (int b = 0; b < nbatch; ++b) {
            run_wave6(b * B);
            run_gpu(b * B);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipMemcpy(ia.data(), d_idx_a, ia.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(ib.data(), d_idx_b, ib.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(ca.data(), d_cnt_a, ca.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(cb.data(), d_cnt_b, cb.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
            const int rows = std::min(B, M - b * B);
            for (int r = 0; r < rows; ++r) {
                const int gr = b * B + r;
                if (ca[static_cast<size_t>(gr)] != cb[static_cast<size_t>(gr)]) ++mismatches;
                const int cnt = std::max(0, ca[static_cast<size_t>(gr)]);
                for (int s = 0; s < cnt; ++s) {
                    if (ia[static_cast<size_t>(gr) * static_cast<size_t>(kStrideSel) + static_cast<size_t>(s)] !=
                        ib[static_cast<size_t>(gr) * static_cast<size_t>(kStrideSel) + static_cast<size_t>(s)]) ++mismatches;
                }
            }
        }
    }

    const double bytes = double(M) * double(stride) * 4.0 + double(M) * double(kStrideSel) * 4.0 * 2.0;
    BenchRow r;
    r.phase = "DSA";
    r.item = "select_topk2048_wave6";
    r.M = M;
    r.iters = std::max(1, opt.iters);
    r.total_us = us_wave6;
    r.hbm_gbs = bytes / (us_wave6 / 1e6) / 1e9;
    r.max_abs_diff = double(mismatches); // index mismatches
    r.speedup = us_gpu / us_wave6;
    char note[96];
    std::snprintf(note, sizeof(note), "L=%d stride=%d mism=%lld", L, stride, mismatches);
    r.note = note;
    add_row(std::move(r));

    BenchRow r2;
    r2.phase = "DSA";
    r2.item = "select_topk2048_gpu";
    r2.M = M;
    r2.iters = std::max(1, opt.iters);
    r2.total_us = us_gpu;
    r2.hbm_gbs = bytes / (us_gpu / 1e6) / 1e9;
    r2.note = "reference selector";
    add_row(std::move(r2));

    std::printf("[dsa]      M=%4d  L=%5d  stride=%d  wave6=%9.1f us  gpu=%9.1f us  "
                "speedup=%.3fx  mismatches=%lld\n",
                M, L, stride, us_wave6, us_gpu, us_gpu / us_wave6, mismatches);
    bud.release();
}

// ===========================================================================
// Commit 4a: Zero-Gather Sparse DSA MQA-64 (k_dsa_sparse_mqa64_fused_vdot2)
// vs v1.0.0 gathered kSparseB=8 strided-batched GEMM baseline
// ===========================================================================
static void bench_dsa_sparse(VramBudget& bud, hipStream_t stream, rocblas_handle blas,
                             int M, int L, const DsaOptions& opt) {
    constexpr int H = 64;
    constexpr int kSparseB = 16;
    // Reuse a single d_out [M, 64, 512] and alias d_y16 onto d_y32 so peak VRAM <= 240 MiB <= 256 MiB!
    uint16_t* d_q_abs   = bud.grab<uint16_t>(size_t(M) * size_t(H) * 576u);
    uint16_t* d_kv_mla  = bud.grab<uint16_t>(size_t(L) * 576u);
    int32_t*  d_sel_idx = bud.grab<int32_t>(size_t(M) * size_t(kStrideSel));
    int32_t*  d_sel_cnt = bud.grab<int32_t>(size_t(M));
    float*    d_sink    = bud.grab<float>(size_t(H));
    uint16_t* d_out     = bud.grab<uint16_t>(size_t(M) * size_t(H) * 512u);
    uint16_t* d_ksel    = bud.grab<uint16_t>(size_t(kSparseB) * size_t(kStrideSel) * 576u);
    uint16_t* d_x16     = bud.grab<uint16_t>(size_t(kSparseB) * size_t(H) * size_t(kStrideSel));
    float*    d_y32     = bud.grab<float>(size_t(kSparseB) * size_t(H) * size_t(kStrideSel));
    uint16_t* d_y16     = reinterpret_cast<uint16_t*>(d_y32);

    {
        uint64_t st = 0x535041525345ULL ^ uint64_t(L);
        std::vector<uint16_t> hq(size_t(M) * size_t(H) * 576u);
        for (auto& v : hq) v = f32_to_bf16_rne(0.25f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_q_abs, hq.data(), hq.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

        std::vector<uint16_t> hkv(size_t(L) * 576u);
        for (auto& v : hkv) v = f32_to_bf16_rne(0.25f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_kv_mla, hkv.data(), hkv.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

        std::vector<int32_t> hidx(static_cast<size_t>(M) * static_cast<size_t>(kStrideSel));
        std::vector<int32_t> hcnt(static_cast<size_t>(M), kStrideSel);
        // Exercise realistic mixed counts (full 2176 + non-multiple-of-32 tail rows)
        for (int t = 0; t < M; ++t) {
            const int cnt = (t == M - 1) ? 2153 : kStrideSel;
            hcnt[static_cast<size_t>(t)] = cnt;
            for (int s = 0; s < kStrideSel; ++s) {
                const uint64_t jitter = static_cast<uint64_t>(rnd_unit(st) * 32.0f);
                int32_t kpos = int32_t((uint64_t(t) * 131u + uint64_t(s) * 7u + jitter) % uint64_t(L));
                hidx[static_cast<size_t>(t) * static_cast<size_t>(kStrideSel) + static_cast<size_t>(s)] = (s < cnt) ? kpos : 0;
            }
        }
        HIP_CHECK(hipMemcpy(d_sel_idx, hidx.data(), hidx.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sel_cnt, hcnt.data(), hcnt.size() * sizeof(int32_t), hipMemcpyHostToDevice));

        std::vector<float> hsink(static_cast<size_t>(H));
        for (auto& v : hsink) v = -1.5f * rnd_unit(st);
        HIP_CHECK(hipMemcpy(d_sink, hsink.data(), hsink.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    auto body_ref = [&](int) {
        const float alpha = 1.0f, beta = 0.0f;
        for (int m0 = 0; m0 < M; m0 += kSparseB) {
            const int B = std::min(kSparseB, M - m0);
            launch_dsa_gather_kv_f16(d_kv_mla, d_sel_idx + size_t(m0) * size_t(kStrideSel),
                                     d_sel_cnt + m0, d_ksel, B, kStrideSel, stream);
            launch_bf16_to_f16(d_q_abs + size_t(m0) * size_t(H) * 576u, d_x16, B * H * 576, stream);
            ROC_CHECK(rocblas_gemm_strided_batched_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                kStrideSel, H, 576, &alpha,
                d_ksel, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                d_x16,  rocblas_datatype_f16_r, 576, int64_t(H) * 576LL,
                &beta,
                d_y32, rocblas_datatype_f32_r, kStrideSel, int64_t(H) * int64_t(kStrideSel),
                d_y32, rocblas_datatype_f32_r, kStrideSel, int64_t(H) * int64_t(kStrideSel),
                B, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            launch_dsa_scores_softmax_sink_f16(
                d_y32, d_sink, d_sel_cnt + m0, d_x16, B, H, kStrideSel, 0.0625f, stream);
            ROC_CHECK(rocblas_gemm_strided_batched_ex(
                blas, rocblas_operation_none, rocblas_operation_none,
                512, H, kStrideSel, &alpha,
                d_ksel, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                d_x16,  rocblas_datatype_f16_r, kStrideSel, int64_t(H) * int64_t(kStrideSel),
                &beta,
                d_y32, rocblas_datatype_f32_r, 512, int64_t(H) * 512LL,
                d_y32, rocblas_datatype_f32_r, 512, int64_t(H) * 512LL,
                B, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            launch_f32_to_bf16(d_y32, d_out + size_t(m0) * size_t(H) * 512u, B * H * 512, stream);
        }
    };

    auto body_hh = [&](int) {
        const float alpha_qk = 0.0625f, alpha_pv = 1.0f, beta = 0.0f;
        for (int m0 = 0; m0 < M; m0 += kSparseB) {
            const int B = std::min(kSparseB, M - m0);
            launch_dsa_gather_kv_f16(d_kv_mla, d_sel_idx + size_t(m0) * size_t(kStrideSel),
                                     d_sel_cnt + m0, d_ksel, B, kStrideSel, stream);
            launch_bf16_to_f16(d_q_abs + size_t(m0) * size_t(H) * 576u, d_x16, B * H * 576, stream);
            ROC_CHECK(rocblas_gemm_strided_batched_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                kStrideSel, H, 576, &alpha_qk,
                d_ksel, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                d_x16,  rocblas_datatype_f16_r, 576, int64_t(H) * 576LL,
                &beta,
                d_y16, rocblas_datatype_f16_r, kStrideSel, int64_t(H) * int64_t(kStrideSel),
                d_y16, rocblas_datatype_f16_r, kStrideSel, int64_t(H) * int64_t(kStrideSel),
                B, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            launch_dsa_scores_softmax_sink_f16_in(
                d_y16, d_sink, d_sel_cnt + m0, d_x16, B, H, kStrideSel, stream);
            ROC_CHECK(rocblas_gemm_strided_batched_ex(
                blas, rocblas_operation_none, rocblas_operation_none,
                512, H, kStrideSel, &alpha_pv,
                d_ksel, rocblas_datatype_f16_r, 576, int64_t(kStrideSel) * 576LL,
                d_x16,  rocblas_datatype_f16_r, kStrideSel, int64_t(H) * int64_t(kStrideSel),
                &beta,
                d_y16, rocblas_datatype_f16_r, 512, int64_t(H) * 512LL,
                d_y16, rocblas_datatype_f16_r, 512, int64_t(H) * 512LL,
                B, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            launch_f16_to_bf16(d_y16, d_out + size_t(m0) * size_t(H) * 512u, B * H * 512, stream);
        }
    };

    auto body_fused = [&](int) {
        launch_dsa_sparse_mqa64_fused_vdot2(d_q_abs, d_kv_mla, d_sink, d_sel_idx, d_sel_cnt,
                                            d_out, M, kStrideSel, stream);
    };

    // Warmup & capture reference output
    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_ref(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_ref(size_t(M) * size_t(H) * 512u);
    HIP_CHECK(hipMemcpy(h_ref.data(), d_out, h_ref.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const double us_ref = time_us(stream, std::max(1, opt.iters), body_ref);

    // Warmup & capture Type_HH_HPA gather+GEMM output
    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_hh(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_hh(size_t(M) * size_t(H) * 512u);
    HIP_CHECK(hipMemcpy(h_hh.data(), d_out, h_hh.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const double us_hh = time_us(stream, std::max(1, opt.iters), body_hh);

    // Warmup & capture fused output
    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_fused(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_fused(size_t(M) * size_t(H) * 512u);
    HIP_CHECK(hipMemcpy(h_fused.data(), d_out, h_fused.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const double us_fused = time_us(stream, std::max(1, opt.iters), body_fused);

    double max_diff_hh = 0.0, max_diff_f = 0.0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        const float rv = bf16_to_f32(h_ref[i]);
        max_diff_hh = std::max(max_diff_hh, double(std::fabs(rv - bf16_to_f32(h_hh[i]))));
        max_diff_f  = std::max(max_diff_f,  double(std::fabs(rv - bf16_to_f32(h_fused[i]))));
    }

    const double flops = 2.0 * double(M) * double(H) * double(kStrideSel) * (576.0 + 512.0);
    const double bytes = double(M) * (double(H) * (576.0 + 512.0) * 2.0 + double(kStrideSel) * 576.0 * 2.0);

    BenchRow r_ref;
    r_ref.phase = "DSA";
    r_ref.item = "sparse_gather_HS_HPA";
    r_ref.M = M;
    r_ref.iters = std::max(1, opt.iters);
    r_ref.total_us = us_ref;
    r_ref.tflops = flops / (us_ref / 1e6) / 1e12;
    r_ref.hbm_gbs = bytes / (us_ref / 1e6) / 1e9;
    r_ref.note = "v1.0.0 gather + HS_HPA GEMM";
    add_row(std::move(r_ref));

    BenchRow r_hh;
    r_hh.phase = "DSA";
    r_hh.item = "sparse_gather_HH_HPA";
    r_hh.M = M;
    r_hh.iters = std::max(1, opt.iters);
    r_hh.total_us = us_hh;
    r_hh.tflops = flops / (us_hh / 1e6) / 1e12;
    r_hh.hbm_gbs = bytes / (us_hh / 1e6) / 1e9;
    r_hh.max_abs_diff = max_diff_hh;
    r_hh.speedup = us_ref / us_hh;
    char note_hh[96];
    std::snprintf(note_hh, sizeof(note_hh), "L=%d gather + HH_HPA B=16", L);
    r_hh.note = note_hh;
    add_row(std::move(r_hh));

    BenchRow r_f;
    r_f.phase = "DSA";
    r_f.item = "Commit4a fused_vdot2";
    r_f.M = M;
    r_f.iters = std::max(1, opt.iters);
    r_f.total_us = us_fused;
    r_f.tflops = flops / (us_fused / 1e6) / 1e12;
    r_f.hbm_gbs = bytes / (us_fused / 1e6) / 1e9;
    r_f.max_abs_diff = max_diff_f;
    r_f.speedup = us_ref / us_fused;
    char note[96];
    std::snprintf(note, sizeof(note), "L=%d zero-gather 1-launch", L);
    r_f.note = note;
    add_row(std::move(r_f));

    std::printf("[dsa-sp]   M=%4d  L=%5d  HS=%9.1f us  HH=%9.1f us (%.2fx, diff=%.2e)  fused=%9.1f us (%.2fx, diff=%.2e)\n",
                M, L, us_ref, us_hh, us_ref / us_hh, max_diff_hh, us_fused, us_ref / us_fused, max_diff_f);
    bud.release();
}

// ===========================================================================
// Wave-10 Commit 3: TP4-Specialized 16-Head Sparse DSA MQA (64 heads total)
//   (i)   owner rank     : kv_is_f16=false, idx_is_u16=false against the full
//                           BF16 MLA KV cache
//   (ii)  non-owner rank : mark_selected_keys -> compact_unique_map ->
//                           gather_unique_kv_f16 -> remap_sel_idx ->
//                           sparse_mqa_tp4_h16(kv_is_f16=true, idx_is_u16=true)
//   Both paths are compared against the existing reference sparse MQA output
//   (gather + strided-batched QK GEMM + sink softmax + PV GEMM, FP16 out).
// ===========================================================================
static uint16_t f32_to_f16_rtz(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        const uint32_t sig = (0x800000u | man);
        return uint16_t(sign | (sig >> (14 - exp)));
    }
    if (exp >= 31) return uint16_t(sign | 0x7BFFu);
    return uint16_t(sign | (uint32_t(exp) << 10) | (man >> 13));
}

static float f16_to_f32_host(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t(h) >> 10) & 0x1Fu;
    const uint32_t man = uint32_t(h) & 0x3FFu;
    uint32_t u = 0;
    if (exp == 0) u = sign;
    else if (exp == 31) u = sign | 0x7F800000u | (man << 13);
    else u = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static void bench_dsa_tp4_h16(VramBudget& bud, hipStream_t stream, rocblas_handle blas,
                              int M, int L, const DsaOptions& opt) {
    constexpr int kRanks = 4;
    constexpr int kRH = 16;                    // heads per TP rank
    constexpr int H = kRanks * kRH;            // 64 heads total
    constexpr int kSparseB = 8;                // reference GEMM batch
    constexpr int kTailCnt = 2153;             // non-multiple-of-32 tail row
    const int stride = kStrideSel;

    // Per-rank packed FP16 Q [4][M][16][576] (the TP4 kernel's native layout).
    uint16_t* d_q = bud.grab<uint16_t>(size_t(kRanks) * size_t(M) * size_t(kRH) * 576u);
    uint16_t* d_kv_bf16 = bud.grab<uint16_t>(size_t(L) * 576u);      // full BF16 MLA KV
    int32_t* d_sel_idx = bud.grab<int32_t>(size_t(M) * size_t(stride));
    int32_t* d_sel_cnt = bud.grab<int32_t>(size_t(M));
    float* d_sink = bud.grab<float>(size_t(H));
    uint16_t* d_out_ref = bud.grab<uint16_t>(size_t(M) * size_t(H) * 512u);
    uint16_t* d_out_test = bud.grab<uint16_t>(size_t(M) * size_t(H) * 512u);
    // Non-owner compaction workspace
    int32_t* d_map = bud.grab<int32_t>(size_t(L));
    int32_t* d_uniq_kpos = bud.grab<int32_t>(size_t(L));
    int32_t* d_uniq_cnt = bud.grab<int32_t>(1);
    uint16_t* d_kv_uniq = bud.grab<uint16_t>(size_t(L) * 576u);
    uint16_t* d_sel_remap = bud.grab<uint16_t>(size_t(M) * size_t(stride));
    // Reference workspace
    uint16_t* d_ksel = bud.grab<uint16_t>(size_t(kSparseB) * size_t(stride) * 576u);
    uint16_t* d_probs = bud.grab<uint16_t>(size_t(kSparseB) * size_t(kRH) * size_t(stride));
    float* d_y32 = bud.grab<float>(size_t(kSparseB) * size_t(kRH) * size_t(stride));

    std::vector<int32_t> hidx(size_t(M) * size_t(stride));
    std::vector<int32_t> hcnt(static_cast<size_t>(M));
    std::vector<uint16_t> hkv_host;
    int uniq_n = 0;
    {
        uint64_t st = 0x43332D544350ULL ^ uint64_t(L);
        std::vector<uint16_t> hq(size_t(kRanks) * size_t(M) * size_t(kRH) * 576u);
        for (auto& v : hq) v = f32_to_f16_rtz(0.25f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_q, hq.data(), hq.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

        std::vector<uint16_t> hkv(size_t(L) * 576u);
        for (auto& v : hkv) v = f32_to_bf16_rne(0.25f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_kv_bf16, hkv.data(), hkv.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        hkv_host = std::move(hkv);

        for (int t = 0; t < M; ++t) {
            // rows: full 2176, one short tail row, one tiny row, and a cnt==0 row
            // (the cnt==0 row exercises the valid==0 path where the kb loop and all
            //  of its barriers are skipped entirely).
            int cnt = stride;
            if (t == M - 1) cnt = 0;
            else if (t == M - 2) cnt = kTailCnt;
            else if (t == 1) cnt = 32;
            hcnt[size_t(t)] = cnt;
            for (int s = 0; s < stride; ++s) {
                const uint64_t jitter = static_cast<uint64_t>(rnd_unit(st) * 32.0f);
                const int32_t kpos = int32_t((uint64_t(t) * 131u + uint64_t(s) * 7u + jitter) % uint64_t(L));
                hidx[size_t(t) * size_t(stride) + size_t(s)] = (s < cnt) ? kpos : 0;
            }
        }
        HIP_CHECK(hipMemcpy(d_sel_idx, hidx.data(), hidx.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sel_cnt, hcnt.data(), hcnt.size() * sizeof(int32_t), hipMemcpyHostToDevice));

        std::vector<float> hsink(static_cast<size_t>(H));
        for (auto& v : hsink) v = -1.5f * rnd_unit(st);
        HIP_CHECK(hipMemcpy(d_sink, hsink.data(), hsink.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    // ---- Reference: existing sparse MQA math, per TP rank, FP16 output -------
    auto body_ref = [&](int) {
        const float alpha = 1.0f, beta = 0.0f;
        for (int r = 0; r < kRanks; ++r) {
            const uint16_t* q_r = d_q + (size_t(r) * size_t(M) * size_t(kRH) * 576u);
            for (int m0 = 0; m0 < M; m0 += kSparseB) {
                const int B = std::min(kSparseB, M - m0);
                launch_dsa_gather_kv_f16(d_kv_bf16, d_sel_idx + size_t(m0) * size_t(stride),
                                         d_sel_cnt + m0, d_ksel, B, stride, stream);
                ROC_CHECK(rocblas_gemm_strided_batched_ex(
                    blas, rocblas_operation_transpose, rocblas_operation_none,
                    stride, kRH, 576, &alpha,
                    d_ksel, rocblas_datatype_f16_r, 576, int64_t(stride) * 576LL,
                    q_r + size_t(m0) * size_t(kRH) * 576u, rocblas_datatype_f16_r, 576,
                    int64_t(kRH) * 576LL,
                    &beta,
                    d_y32, rocblas_datatype_f32_r, stride, int64_t(kRH) * int64_t(stride),
                    d_y32, rocblas_datatype_f32_r, stride, int64_t(kRH) * int64_t(stride),
                    B, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
                launch_dsa_scores_softmax_sink_f16(
                    d_y32, d_sink + size_t(r) * size_t(kRH), d_sel_cnt + m0, d_probs,
                    B, kRH, stride, 0.0625f, stream);
                ROC_CHECK(rocblas_gemm_strided_batched_ex(
                    blas, rocblas_operation_none, rocblas_operation_none,
                    512, kRH, stride, &alpha,
                    d_ksel, rocblas_datatype_f16_r, 576, int64_t(stride) * 576LL,
                    d_probs, rocblas_datatype_f16_r, stride, int64_t(kRH) * int64_t(stride),
                    &beta,
                    d_y32, rocblas_datatype_f32_r, 512, int64_t(kRH) * 512LL,
                    d_y32, rocblas_datatype_f32_r, 512, int64_t(kRH) * 512LL,
                    B, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
                launch_f32_to_f16(d_y32,
                                  d_out_ref + (size_t(r) * size_t(M) + size_t(m0)) * size_t(kRH) * 512u,
                                  B * kRH * 512, stream);
            }
        }
    };

    // ---- (i) owner rank path: full BF16 KV, int32 indices -------------------
    auto body_owner = [&](int) {
        for (int r = 0; r < kRanks; ++r) {
            launch_dsa_sparse_mqa_tp4_h16(
                d_q + size_t(r) * size_t(M) * size_t(kRH) * 576u, d_kv_bf16,
                d_sink + size_t(r) * size_t(kRH), d_sel_idx, d_sel_cnt,
                d_out_test + size_t(r) * size_t(M) * size_t(kRH) * 512u,
                M, stride, /*kv_is_f16=*/false, /*idx_is_u16=*/false, stream);
        }
    };

    // ---- (ii) non-owner path: compacted unique KV + u16 indices -------------
    auto body_nonowner = [&](int) {
        launch_dsa_mark_selected_keys(d_sel_idx, d_sel_cnt, d_map, M, stride, L, stream);
        launch_dsa_compact_unique_map(d_map, d_uniq_kpos, d_uniq_cnt, L, stream);
        launch_dsa_gather_unique_kv_f16(d_kv_bf16, d_uniq_kpos, d_kv_uniq, uniq_n, stream);
        launch_dsa_remap_sel_idx(d_sel_idx, d_sel_cnt, d_map, d_sel_remap, M, stride,
                                 /*use_u16=*/true, stream);
        for (int r = 0; r < kRanks; ++r) {
            launch_dsa_sparse_mqa_tp4_h16(
                d_q + size_t(r) * size_t(M) * size_t(kRH) * 576u, d_kv_uniq,
                d_sink + size_t(r) * size_t(kRH), d_sel_remap, d_sel_cnt,
                d_out_test + size_t(r) * size_t(M) * size_t(kRH) * 512u,
                M, stride, /*kv_is_f16=*/true, /*idx_is_u16=*/true, stream);
        }
    };

    // Warmup + capture the reference output, then derive the exact unique count.
    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_ref(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_ref(size_t(M) * size_t(H) * 512u);
    HIP_CHECK(hipMemcpy(h_ref.data(), d_out_ref, h_ref.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const double us_ref = time_us(stream, std::max(1, opt.iters), body_ref);

    launch_dsa_mark_selected_keys(d_sel_idx, d_sel_cnt, d_map, M, stride, L, stream);
    launch_dsa_compact_unique_map(d_map, d_uniq_kpos, d_uniq_cnt, L, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    int32_t h_uniq_cnt = 0;
    HIP_CHECK(hipMemcpy(&h_uniq_cnt, d_uniq_cnt, sizeof(int32_t), hipMemcpyDeviceToHost));
    uniq_n = h_uniq_cnt;
    launch_dsa_gather_unique_kv_f16(d_kv_bf16, d_uniq_kpos, d_kv_uniq, uniq_n, stream);
    launch_dsa_remap_sel_idx(d_sel_idx, d_sel_cnt, d_map, d_sel_remap, M, stride, /*use_u16=*/true, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    // Exact host-side unique-key compaction count + remap bijection check.
    long long uniq_mismatch = 0;
    {
        std::vector<uint8_t> seen(size_t(L), 0u);
        long long expect = 0;
        for (int t = 0; t < M; ++t) {
            const int cnt = hcnt[size_t(t)];
            for (int s = 0; s < cnt; ++s) {
                const int32_t kpos = hidx[size_t(t) * size_t(stride) + size_t(s)];
                if (!seen[size_t(kpos)]) { seen[size_t(kpos)] = 1u; ++expect; }
            }
        }
        if (expect != int64_t(h_uniq_cnt)) ++uniq_mismatch;
        std::vector<int32_t> h_kpos(static_cast<size_t>(h_uniq_cnt));
        std::vector<uint16_t> h_remap(static_cast<size_t>(M) * static_cast<size_t>(stride));
        HIP_CHECK(hipMemcpy(h_kpos.data(), d_uniq_kpos, h_kpos.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_remap.data(), d_sel_remap, h_remap.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for (int t = 0; t < M; ++t) {
            const int cnt = hcnt[size_t(t)];
            for (int s = 0; s < cnt; ++s) {
                const int32_t kpos = hidx[size_t(t) * size_t(stride) + size_t(s)];
                const uint32_t cid = uint32_t(h_remap[size_t(t) * size_t(stride) + size_t(s)]);
                if (cid >= uint32_t(h_uniq_cnt) || h_kpos[size_t(cid)] != kpos) ++uniq_mismatch;
            }
        }
        // The compacted KV must be a bit-exact FP16 gather of the selected BF16 rows.
        std::vector<uint16_t> h_kv_uniq(size_t(h_uniq_cnt) * 576u);
        HIP_CHECK(hipMemcpy(h_kv_uniq.data(), d_kv_uniq, h_kv_uniq.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for (int32_t c = 0; c < h_uniq_cnt; ++c) {
            const int32_t kpos = h_kpos[size_t(c)];
            for (int d = 0; d < 576; ++d) {
                const uint16_t want = f32_to_f16_rtz(bf16_to_f32(hkv_host[size_t(kpos) * 576u + size_t(d)]));
                if (h_kv_uniq[size_t(c) * 576u + size_t(d)] != want) ++uniq_mismatch;
            }
        }
    }
    uniq_n = h_uniq_cnt;

    // Warmup & capture both TP4 paths.
    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_owner(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_owner(size_t(M) * size_t(H) * 512u);
    HIP_CHECK(hipMemcpy(h_owner.data(), d_out_test, h_owner.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const double us_owner = time_us(stream, std::max(1, opt.iters), body_owner);

    for (int i = 0; i < std::max(1, opt.warmup); ++i) body_nonowner(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_nonowner(size_t(M) * size_t(H) * 512u);
    HIP_CHECK(hipMemcpy(h_nonowner.data(), d_out_test, h_nonowner.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const double us_nonowner = time_us(stream, std::max(1, opt.iters), body_nonowner);

    // Both reference and TP4 rank outputs are [kRanks, M, 16, 512] rank-major contiguous.
    double max_diff_owner = 0.0, max_diff_nonowner = 0.0;
    for (int r = 0; r < kRanks; ++r) {
        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < kRH; ++h) {
                const size_t base = (size_t(r) * size_t(M) + size_t(m)) * size_t(kRH) * 512u + size_t(h) * 512u;
                for (int d = 0; d < 512; ++d) {
                    const float rv = f16_to_f32_host(h_ref[base + size_t(d)]);
                    max_diff_owner = std::max(max_diff_owner,
                        double(std::fabs(rv - f16_to_f32_host(h_owner[base + size_t(d)]))));
                    max_diff_nonowner = std::max(max_diff_nonowner,
                        double(std::fabs(rv - f16_to_f32_host(h_nonowner[base + size_t(d)]))));
                }
            }
        }
    }
    // The cnt==0 row must be exactly zero (the valid==0 barrier fix).
    long long zero_row_bad = 0;
    for (int r = 0; r < kRanks; ++r) {
        const size_t base = (size_t(r) * size_t(M) + size_t(M - 1)) * size_t(kRH) * 512u;
        for (size_t i = 0; i < size_t(kRH) * 512u; ++i) {
            if (h_owner[base + i] != 0u || h_nonowner[base + i] != 0u) ++zero_row_bad;
        }
    }

    const double flops = 2.0 * double(M) * double(H) * double(stride) * (576.0 + 512.0);
    const double bytes = double(M) * (double(H) * (576.0 + 512.0) * 2.0 + double(stride) * 576.0 * 2.0);

    BenchRow r_ref;
    r_ref.phase = "DSA";
    r_ref.item = "tp4_h16_reference";
    r_ref.M = M;
    r_ref.iters = std::max(1, opt.iters);
    r_ref.total_us = us_ref;
    r_ref.tflops = flops / (r_ref.total_us / 1e6) / 1e12;
    r_ref.hbm_gbs = bytes / (r_ref.total_us / 1e6) / 1e9;
    r_ref.note = "gather + HH_HPA per rank";
    add_row(std::move(r_ref));

    BenchRow r_own;
    r_own.phase = "DSA";
    r_own.item = "tp4_h16_owner";
    r_own.M = M;
    r_own.iters = std::max(1, opt.iters);
    r_own.total_us = us_owner;
    r_own.tflops = flops / (us_owner / 1e6) / 1e12;
    r_own.hbm_gbs = bytes / (us_owner / 1e6) / 1e9;
    r_own.max_abs_diff = max_diff_owner;
    char note_own[96];
    std::snprintf(note_own, sizeof(note_own), "kv=bf16 idx=i32 uniq=%d", uniq_n);
    r_own.note = note_own;
    add_row(std::move(r_own));

    BenchRow r_non;
    r_non.phase = "DSA";
    r_non.item = "tp4_h16_nonowner";
    r_non.M = M;
    r_non.iters = std::max(1, opt.iters);
    r_non.total_us = us_nonowner;
    r_non.tflops = flops / (us_nonowner / 1e6) / 1e12;
    r_non.hbm_gbs = bytes / (us_nonowner / 1e6) / 1e9;
    r_non.max_abs_diff = max_diff_nonowner;
    char note_non[96];
    std::snprintf(note_non, sizeof(note_non), "kv=f16 idx=u16 uniq=%d", uniq_n);
    r_non.note = note_non;
    add_row(std::move(r_non));

    std::printf("[dsa-tp4]  M=%4d  L=%5d  ref=%9.1f us  owner=%9.1f us (diff=%.2e)  "
                "nonowner=%9.1f us (diff=%.2e)  uniq=%d mism=%lld zero_row_bad=%lld\n",
                M, L, r_ref.total_us, us_owner, max_diff_owner, us_nonowner, max_diff_nonowner,
                uniq_n, uniq_mismatch, zero_row_bad);
    bud.release();
}


// ===========================================================================
// Wave-10 Commit 4: hybrid m64 / m32 / m8 expert-sorted CSR MoE parity & speedup
//   reference : original Commit 2 m64-only CSR (hybrid_buckets=false)
//   hybrid    : Commit 4 three-bucket CSR (49..64 -> m64, 9..32 -> m32,
//               1..8 -> grouped m8) with single-pass WHT128 over n_pairs
// ===========================================================================
static void bench_moe_hybrid(VramBudget& bud, hipStream_t stream, const MoeOptions& opt) {
    constexpr int kExp = 24;              // 24 real-dim (2048x2048) experts fit under 256 MiB VramBudget
    constexpr int kTopK = 10;
    constexpr int kLat = 2048;            // real ERNIE-5.5 latent dim
    constexpr int kInter = 2048;          // real ERNIE-5.5 moe inter size
    constexpr int kMaxTiles = 4096;
    constexpr int kM8Chunk = 24;
    const int M = 2048;                   // real Wave-10 chunk size
    const int slots = M * kTopK;

    // Covers every single bucket boundary: 0, 1..8 (m8), 9..32 (m32), 33..48 (m32+m32/m8),
    // 49..63 (m64), 64 (m64), and 65..137 (m64 + all tail classes).
    const int hist[kExp] = {0, 1, 3, 7, 8, 9, 12, 16, 20, 31, 32, 33, 40, 48, 49, 55, 63, 64, 65, 72, 95, 96, 127, 137};
    int assigned = 0;
    for (int e = 0; e < kExp; ++e) assigned += hist[e];

    uint16_t* d_x = bud.grab<uint16_t>(size_t(M) * size_t(kLat));
    int32_t* d_ids = bud.grab<int32_t>(size_t(slots));
    float* d_wts = bud.grab<float>(size_t(slots));
    int16_t* d_g2l = bud.grab<int16_t>(512u);
    int32_t* d_csr = bud.grab<int32_t>(4u);
    Ernie55MoeWorkTile* d_tiles = bud.grab<Ernie55MoeWorkTile>(kMaxTiles);
    Ernie55MoeRoutedPair* d_pairs = bud.grab<Ernie55MoeRoutedPair>(size_t(assigned) + 64u);
    Hlut4Tile32x32* d_w1 = bud.grab<Hlut4Tile32x32>(size_t(kExp) * size_t(kInter / 32) * size_t(kLat / 32));
    Hlut4Tile32x32* d_w3 = bud.grab<Hlut4Tile32x32>(size_t(kExp) * size_t(kInter / 32) * size_t(kLat / 32));
    Hlut4Tile32x32* d_w2 = bud.grab<Hlut4Tile32x32>(size_t(kExp) * size_t(kLat / 32) * size_t(kInter / 32));
    uint16_t* d_levels = bud.grab<uint16_t>(8u);
    uint64_t* d_signs = bud.grab<uint64_t>(size_t(kInter / 64));
    uint16_t* d_z = bud.grab<uint16_t>((size_t(assigned) + 64u) * size_t(kInter));
    float* d_y_ref = bud.grab<float>(size_t(M) * size_t(kLat));
    float* d_y_hyb = bud.grab<float>(size_t(M) * size_t(kLat));
    ExpertDispatchGroup* d_groups = bud.grab<ExpertDispatchGroup>(kExp);
    float* d_z_f32 = bud.grab<float>(size_t(kM8Chunk) * size_t(kMaxGroupTokens) * size_t(kInter));
    uint16_t* d_z_rot = bud.grab<uint16_t>(size_t(kM8Chunk) * size_t(kMaxGroupTokens) * size_t(kInter));

    std::vector<int32_t> h_ids(static_cast<size_t>(slots), -1);
    std::vector<float> h_wts(static_cast<size_t>(slots), 0.0f);
    {
        uint64_t st = 0x43342D484259ULL;
        std::vector<uint16_t> hx(static_cast<size_t>(M) * static_cast<size_t>(kLat));
        for (auto& v : hx) v = f32_to_f16_rtz(0.2f * (rnd_unit(st) - 0.5f));
        HIP_CHECK(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        std::vector<uint16_t> hl(8u);
        for (auto& v : hl) v = f32_to_f16_rtz(0.5f * rnd_unit(st));
        HIP_CHECK(hipMemcpy(d_levels, hl.data(), hl.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        std::vector<uint64_t> hs(size_t(kInter / 64), 0xA55A5AA53C3CC3C3ULL);
        HIP_CHECK(hipMemcpy(d_signs, hs.data(), hs.size() * sizeof(uint64_t), hipMemcpyHostToDevice));
        const size_t w_u16 = size_t(kExp) * size_t(kInter / 32) * size_t(kLat / 32) * 528u / 2u;
        std::vector<uint16_t> hw(w_u16);
        for (size_t i = 0; i < w_u16; ++i) {
            hw[i] = uint16_t((i * 1103515245u + 12345u) >> 16);
        }
        // Keep FP16 tile scales well-conditioned
        auto* wtiles = reinterpret_cast<Hlut4Tile32x32*>(hw.data());
        const size_t n_wt = size_t(kExp) * size_t(kInter / 32) * size_t(kLat / 32);
        for (size_t t = 0; t < n_wt; ++t) {
            for (int g = 0; g < 8; ++g) wtiles[t].scales_f16[g] = f32_to_f16_rtz(0.05f);
        }
        HIP_CHECK(hipMemcpy(d_w1, hw.data(), hw.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w3, hw.data(), hw.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w2, hw.data(), hw.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        std::vector<int16_t> hg(512u, int16_t(-1));
        for (int e = 0; e < kExp; ++e) hg[size_t(e)] = int16_t(e);
        HIP_CHECK(hipMemcpy(d_g2l, hg.data(), hg.size() * sizeof(int16_t), hipMemcpyHostToDevice));

        int p = 0;
        for (int e = 0; e < kExp; ++e) {
            for (int c = 0; c < hist[size_t(e)] && p < slots; ++c, ++p) {
                const int slot_idx = ((p * 17) % slots);
                int idx = slot_idx;
                while (h_ids[size_t(idx)] >= 0) idx = (idx + 1) % slots;
                h_ids[size_t(idx)] = e;
                h_wts[size_t(idx)] = 0.05f + 0.15f * rnd_unit(st);
            }
        }
        assigned = p;
        HIP_CHECK(hipMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_wts, h_wts.data(), h_wts.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    auto run_m64 = [&](int) {
        HIP_CHECK(hipMemsetAsync(d_y_ref, 0, size_t(M) * size_t(kLat) * sizeof(float), stream));
        launch_ernie55_build_expert_csr_prefill(d_ids, d_wts, d_g2l, M, kExp, 64, kMaxTiles, assigned + 64,
                                                d_csr, d_tiles, d_pairs, stream, /*hybrid_buckets=*/false);
        launch_hlut4_prefill_situ_f16_m64(d_x, d_csr, d_tiles, d_pairs, d_w1, d_w3, d_levels, d_z,
                                          kInter, kLat, 4.0f, 25.0f, kMaxTiles, stream);
        launch_wht128_f16_inplace_prefill_pairs(d_z, d_csr, d_signs, kInter, assigned + 64, stream);
        launch_hlut4_prefill_down_atomic_m64(d_z, d_csr, d_tiles, d_pairs, d_w2, d_levels, d_y_ref,
                                             kLat, kInter, kMaxTiles, stream);
    };

    for (int i = 0; i < std::max(1, opt.warmup); ++i) run_m64(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    const double us_ref = time_us(stream, std::max(1, opt.iters), run_m64);

    auto run_hybrid = [&](int) {
        HIP_CHECK(hipMemsetAsync(d_y_hyb, 0, size_t(M) * size_t(kLat) * sizeof(float), stream));
        launch_ernie55_build_expert_csr_prefill(d_ids, d_wts, d_g2l, M, kExp, 64, kMaxTiles, assigned + 64,
                                                d_csr, d_tiles, d_pairs, stream, /*hybrid_buckets=*/true);
        constexpr int kB1Beg = kMaxTiles / 2;
        constexpr int kB2Beg = (kMaxTiles * 3) / 4;
        launch_hlut4_prefill_situ_f16_m64(d_x, d_csr, d_tiles, d_pairs, d_w1, d_w3, d_levels, d_z,
                                          kInter, kLat, 4.0f, 25.0f, kMaxTiles / 2, stream);
        launch_hlut4_prefill_situ_f16_m32(d_x, d_csr + 2, d_tiles + kB1Beg, d_pairs, d_w1, d_w3,
                                          d_levels, d_z, kInter, kLat, 4.0f, 25.0f, kMaxTiles / 4, stream);
        launch_wht128_f16_inplace_prefill_pairs(d_z, d_csr, d_signs, kInter, assigned + 64, stream);
        launch_hlut4_prefill_down_atomic_m64(d_z, d_csr, d_tiles, d_pairs, d_w2, d_levels, d_y_hyb,
                                             kLat, kInter, kMaxTiles / 2, stream);
        launch_hlut4_prefill_down_atomic_m32(d_z, d_csr + 2, d_tiles + kB1Beg, d_pairs, d_w2,
                                             d_levels, d_y_hyb, kLat, kInter, kMaxTiles / 4, stream);
        launch_ernie55_build_groups_from_csr_m8(d_csr, d_tiles + kB2Beg, d_pairs, d_groups,
                                                kExp, stream);
        for (int g0 = 0; g0 < kExp; g0 += kM8Chunk) {
            const int ng = std::min(kM8Chunk, kExp - g0);
            launch_hlut4_fdot2_grouped_gu_situ(d_x, d_groups + g0, ng, d_w1, d_w3, d_levels, d_z_f32,
                                               kInter, kLat, 4.0f, 25.0f, stream);
            launch_wht128_grouped_f32_to_f16(d_z_f32, d_groups + g0, ng, d_signs, d_z_rot, kInter,
                                             stream, kMaxGroupTokens);
            launch_hlut4_fdot2_grouped_down_atomic(d_z_rot, d_groups + g0, ng, d_w2, d_levels,
                                                   d_y_hyb, kLat, kInter, stream);
        }
    };

    for (int i = 0; i < std::max(1, opt.warmup); ++i) run_hybrid(i);
    HIP_CHECK(hipStreamSynchronize(stream));
    const double us_hyb = time_us(stream, std::max(1, opt.iters), run_hybrid);

    std::vector<float> h_ref(size_t(M) * size_t(kLat)), h_hyb(size_t(M) * size_t(kLat));
    HIP_CHECK(hipMemcpy(h_ref.data(), d_y_ref, h_ref.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_hyb.data(), d_y_hyb, h_hyb.size() * sizeof(float), hipMemcpyDeviceToHost));
    double max_diff = 0.0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        max_diff = std::max(max_diff, double(std::fabs(h_ref[i] - h_hyb[i])));
    }
    int32_t h_cnt[4] = {0, 0, 0, 0};
    HIP_CHECK(hipMemcpy(h_cnt, d_csr, sizeof(h_cnt), hipMemcpyDeviceToHost));

    const double flops = 2.0 * double(assigned) * double(kLat) * (double(kInter) * 2.0 + double(kInter));
    BenchRow r_ref;
    r_ref.phase = "MoE";
    r_ref.item = "hybrid_m64_reference";
    r_ref.M = M;
    r_ref.iters = std::max(1, opt.iters);
    r_ref.total_us = us_ref;
    r_ref.tflops = flops / (us_ref / 1e6) / 1e12;
    r_ref.note = "Commit 2 m64-only CSR";
    add_row(std::move(r_ref));

    BenchRow r_hyb;
    r_hyb.phase = "MoE";
    r_hyb.item = "hybrid_m64_m32_m8";
    r_hyb.M = M;
    r_hyb.iters = std::max(1, opt.iters);
    r_hyb.total_us = us_hyb;
    r_hyb.tflops = flops / (us_hyb / 1e6) / 1e12;
    r_hyb.max_abs_diff = max_diff;
    char note[96];
    std::snprintf(note, sizeof(note), "n64=%d n32=%d n8=%d", h_cnt[0], h_cnt[2], h_cnt[3]);
    r_hyb.note = note;
    add_row(std::move(r_hyb));

    std::printf("[moe-hyb]  M=%4d  exp=%d  ref=%9.1f us  hybrid=%9.1f us (%.2fx)  diff=%.2e  "
                "n64=%d n32=%d n8=%d pairs=%d/%d\n",
                M, kExp, us_ref, us_hyb, us_ref / us_hyb, max_diff, h_cnt[0], h_cnt[2], h_cnt[3],
                h_cnt[1], assigned);
    bud.release();
}


// ===========================================================================
// Wave-10 C5b: single-pass K_tile = 16,384 FlashIndexer streaming top-2048
//   reference : k_dsa_select_topk2048_gpu over the full score row
//   streaming : k_dsa_topk2048_streaming_merge ping-pong across K tiles +
//               k_dsa_topk2048_streaming_finalize
// ===========================================================================
static void bench_dsa_flashindexer_streaming(VramBudget& bud, hipStream_t stream,
                                             int n_sp, int32_t pos0, const char* tag) {
    constexpr int kTileK = 16384;
    constexpr int kTopK  = 2048;
    const int max_hist = std::max(1, (pos0 + n_sp - 1) - 127);
    const int n_tiles  = (max_hist + kTileK - 1) / kTileK;

    float* d_tiles_all = bud.grab<float>(size_t(n_tiles) * size_t(n_sp) * size_t(kTileK));
    float* d_ref       = bud.grab<float>(size_t(n_sp) * size_t(max_hist));
    float* d_bv0       = bud.grab<float>(size_t(n_sp) * size_t(kTopK));
    int32_t* d_bi0     = bud.grab<int32_t>(size_t(n_sp) * size_t(kTopK));
    float* d_bv1       = bud.grab<float>(size_t(n_sp) * size_t(kTopK));
    int32_t* d_bi1     = bud.grab<int32_t>(size_t(n_sp) * size_t(kTopK));
    int32_t* d_idx_ref = bud.grab<int32_t>(size_t(n_sp) * size_t(kStrideSel));
    int32_t* d_cnt_ref = bud.grab<int32_t>(size_t(n_sp));
    int32_t* d_idx_str = bud.grab<int32_t>(size_t(n_sp) * size_t(kStrideSel));
    int32_t* d_cnt_str = bud.grab<int32_t>(size_t(n_sp));

    std::vector<float> h_ref(static_cast<size_t>(n_sp) * static_cast<size_t>(max_hist));
    std::vector<float> h_tiles(static_cast<size_t>(n_tiles) * static_cast<size_t>(n_sp) * static_cast<size_t>(kTileK), -FLT_MAX);
    {
        uint64_t st = 0x4635424C4B5354ULL ^ uint64_t(pos0);
        for (int t = 0; t < n_sp; ++t) {
            const int n_hist = (pos0 + t) - 127;
            for (int k = 0; k < max_hist; ++k) {
                const float v = (k < n_hist) ? (rnd_unit(st) - 0.35f) : -FLT_MAX;
                h_ref[size_t(t) * size_t(max_hist) + size_t(k)] = v;
                const int ti = k / kTileK;
                const int kl = k % kTileK;
                h_tiles[(size_t(ti) * size_t(n_sp) + size_t(t)) * size_t(kTileK) + size_t(kl)] = v;
            }
        }
        HIP_CHECK(hipMemcpy(d_ref, h_ref.data(), h_ref.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_tiles_all, h_tiles.data(), h_tiles.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemsetAsync(d_bv0, 0, size_t(n_sp) * size_t(kTopK) * sizeof(float), stream));
        HIP_CHECK(hipMemsetAsync(d_bi0, 0, size_t(n_sp) * size_t(kTopK) * sizeof(int32_t), stream));
    }

    auto run_ref = [&](int) {
        launch_dsa_select_topk2048_gpu(d_ref, d_idx_ref, d_cnt_ref, pos0, 0, n_sp, max_hist,
                                       kStrideSel, stream);
    };
    auto run_stream = [&](int) {
        bool odd = false;
        for (int ti = 0; ti < n_tiles; ++ti) {
            const int k_blk = ti * kTileK;
            const float* d_tile = d_tiles_all + size_t(ti) * size_t(n_sp) * size_t(kTileK);
            if (!odd) {
                launch_dsa_topk2048_streaming_merge(d_tile, d_bv0, d_bi0, d_bv1, d_bi1,
                                                    pos0, 0, n_sp, k_blk, kTileK, stream);
            } else {
                launch_dsa_topk2048_streaming_merge(d_tile, d_bv1, d_bi1, d_bv0, d_bi0,
                                                    pos0, 0, n_sp, k_blk, kTileK, stream);
            }
            odd = !odd;
        }
        launch_dsa_topk2048_streaming_finalize(odd ? d_bi1 : d_bi0, d_idx_str, d_cnt_str,
                                               pos0, 0, n_sp, kStrideSel, stream);
    };

    for (int i = 0; i < 2; ++i) { run_ref(i); run_stream(i); }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<int32_t> h_idx_ref(size_t(n_sp) * size_t(kStrideSel)), h_idx_str(h_idx_ref.size());
    std::vector<int32_t> h_cnt_ref(static_cast<size_t>(n_sp)), h_cnt_str(static_cast<size_t>(n_sp));
    HIP_CHECK(hipMemcpy(h_idx_ref.data(), d_idx_ref, h_idx_ref.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_idx_str.data(), d_idx_str, h_idx_str.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_cnt_ref.data(), d_cnt_ref, h_cnt_ref.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_cnt_str.data(), d_cnt_str, h_cnt_str.size() * sizeof(int32_t), hipMemcpyDeviceToHost));

    long long set_mismatch = 0, cnt_mismatch = 0, tail_mismatch = 0;
    for (int t = 0; t < n_sp; ++t) {
        const int pos = pos0 + t;
        const int swa_start = (pos >= 127) ? (pos - 127) : 0;
        const int n_hist = swa_start;
        if (h_cnt_ref[size_t(t)] != h_cnt_str[size_t(t)]) ++cnt_mismatch;
        if (n_hist <= kTopK) {
            for (int p = 0; p <= pos; ++p) {
                if (h_idx_ref[size_t(t) * size_t(kStrideSel) + size_t(p)] !=
                    h_idx_str[size_t(t) * size_t(kStrideSel) + size_t(p)]) ++tail_mismatch;
            }
            continue;
        }
        std::vector<int32_t> a(h_idx_ref.begin() + size_t(t) * size_t(kStrideSel),
                               h_idx_ref.begin() + size_t(t) * size_t(kStrideSel) + kTopK);
        std::vector<int32_t> b(h_idx_str.begin() + size_t(t) * size_t(kStrideSel),
                               h_idx_str.begin() + size_t(t) * size_t(kStrideSel) + kTopK);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) ++set_mismatch;
        const int swa_len = pos - swa_start + 1;
        for (int k = 0; k < swa_len; ++k) {
            if (h_idx_ref[size_t(t) * size_t(kStrideSel) + size_t(kTopK + k)] !=
                h_idx_str[size_t(t) * size_t(kStrideSel) + size_t(kTopK + k)]) ++tail_mismatch;
        }
    }

    const double us_ref = time_us(stream, 3, run_ref);
    const double us_str = time_us(stream, 3, run_stream);
    BenchRow r;
    r.phase = "DSA";
    r.item = "fi_streaming_merge";
    r.M = n_sp;
    r.iters = 3;
    r.total_us = us_str;
    r.max_abs_diff = double(set_mismatch + cnt_mismatch + tail_mismatch);
    r.speedup = us_ref / us_str;
    char note[96];
    std::snprintf(note, sizeof(note), "%s n_hist=%d tiles=%d mism=%lld", tag, max_hist, n_tiles,
                  set_mismatch + cnt_mismatch + tail_mismatch);
    r.note = note;
    add_row(std::move(r));
    std::printf("[fi-str]   %-10s n_sp=%4d n_hist=%6d tiles=%d  ref=%9.1f us  stream=%9.1f us "
                "(%.2fx)  set_mism=%lld cnt_mism=%lld tail_mism=%lld\n",
                tag, n_sp, max_hist, n_tiles, us_ref, us_str, us_ref / us_str,
                set_mismatch, cnt_mismatch, tail_mismatch);
    bud.release();
}

// ===========================================================================
// Wave-10 C6: DSpark launch fusion silicon parity (fused Markov+MultiMax,
// packed sparse-candidate readout, broadcast noise+pos, iota pos)
// ===========================================================================
static void bench_dspark_c6(VramBudget& bud, hipStream_t stream) {
    constexpr int V = 201216;
    constexpr int R = 256;
    constexpr int B = 5;
    constexpr int H = 4096;
    constexpr int kMaxCands = 4096;

    uint16_t* d_me   = bud.grab<uint16_t>(R);
    uint16_t* d_w2   = bud.grab<uint16_t>(size_t(V) * size_t(R));
    float* d_base    = bud.grab<float>(V);
    float* d_bias    = bud.grab<float>(V);
    float* d_out_ref = bud.grab<float>(V);
    float* d_out_fus = bud.grab<float>(V);
    float* d_r4      = bud.grab<float>(4);
    float* d_t4      = bud.grab<float>(4);
    float* d_part64  = bud.grab<float>(64);
    float* d_stats2  = bud.grab<float>(2);
    int32_t* d_cnt   = bud.grab<int32_t>(1);
    int32_t* d_cidx  = bud.grab<int32_t>(kMaxCands);
    float* d_cexp    = bud.grab<float>(kMaxCands);
    DsparkSparsePacketHeader* d_hdr = bud.grab<DsparkSparsePacketHeader>(1);
    DsparkCandPair* d_cands = bud.grab<DsparkCandPair>(kMaxCands);
    uint16_t* d_noise = bud.grab<uint16_t>(H);
    uint16_t* d_xin   = bud.grab<uint16_t>(size_t(B) * size_t(H));
    int32_t* d_pos    = bud.grab<int32_t>(B);

    uint64_t st = 0xC6D59A80ULL;
    std::vector<uint16_t> h_me(R), h_w2(size_t(V) * size_t(R)), h_noise(H);
    std::vector<float> h_base(V);
    for (auto& x : h_me)    x = f32_to_bf16_rne(0.25f * (rnd_unit(st) - 0.5f));
    for (auto& x : h_w2)    x = f32_to_fp16_rne(0.15f * (rnd_unit(st) - 0.5f));
    for (auto& x : h_noise) x = f32_to_bf16_rne(rnd_unit(st) - 0.5f);
    for (int v = 0; v < V; ++v) h_base[size_t(v)] = 4.0f * (rnd_unit(st) - 0.5f);
    // Inject a sharp top-12 peak + -inf sentinel to exercise sparse extraction & -inf passthrough
    for (int k = 0; k < 12; ++k) h_base[size_t(k * 997 + 13)] = 18.0f - 0.4f * float(k);
    h_base[7] = -std::numeric_limits<float>::infinity();
    const float h_r4[4] = {-6.0f, 6.0f, -10.0f, 10.0f};
    const float h_t4[4] = {0.15f, -0.10f, 0.02f, -0.015f};

    HIP_CHECK(hipMemcpy(d_me,    h_me.data(),    h_me.size() * 2u, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w2,    h_w2.data(),    h_w2.size() * 2u, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_base,  h_base.data(),  h_base.size() * 4u, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_r4,    h_r4,           16u, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_t4,    h_t4,           16u, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_noise, h_noise.data(), h_noise.size() * 2u, hipMemcpyHostToDevice));

    // 1. Parity & timing: Ref (bf16x_f16w_gemv_f32 + add_bias_multimax) vs Fused (markov_bias_multimax_f32)
    auto run_ref_mm = [&](int) {
        launch_bf16x_f16w_gemv_f32(d_me, d_w2, d_bias, 1, V, R, stream);
        launch_dspark_add_bias_multimax_f32(d_base, d_bias, d_r4, d_t4, d_out_ref, V, stream);
    };
    auto run_fus_mm = [&](int) {
        launch_dspark_markov_bias_multimax_f32(d_me, d_w2, d_base, d_r4, d_t4, d_out_fus, V, R, stream);
    };
    run_ref_mm(0);
    run_fus_mm(0);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> y_ref(V), y_fus(V);
    HIP_CHECK(hipMemcpy(y_ref.data(), d_out_ref, size_t(V) * 4u, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(y_fus.data(), d_out_fus, size_t(V) * 4u, hipMemcpyDeviceToHost));
    double mm_max_diff = 0.0;
    int inf_mism = 0;
    for (int v = 0; v < V; ++v) {
        if (std::isinf(y_ref[size_t(v)]) && y_ref[size_t(v)] < 0.0f) {
            if (!(std::isinf(y_fus[size_t(v)]) && y_fus[size_t(v)] < 0.0f)) ++inf_mism;
            continue;
        }
        mm_max_diff = std::max(mm_max_diff, std::fabs(double(y_ref[size_t(v)]) - double(y_fus[size_t(v)])));
    }
    const double us_ref_mm = time_us(stream, 20, run_ref_mm);
    const double us_fus_mm = time_us(stream, 20, run_fus_mm);

    // 2. Parity: Unpacked vs Packed GPU sparse softmax candidate extraction
    const float temp = 0.7f;
    launch_gpu_softmax_sparse_top_candidates(d_out_fus, V, temp, d_part64, d_stats2, d_cnt, d_cidx, d_cexp, kMaxCands, stream);
    launch_gpu_softmax_sparse_top_candidates_packed(d_out_fus, V, temp, d_part64, d_hdr, d_cands, kMaxCands, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    float h_st2[2] = {};
    int32_t h_cnt = 0;
    DsparkSparsePacketHeader h_hdr{};
    HIP_CHECK(hipMemcpy(h_st2, d_stats2, 8u, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_cnt, d_cnt, 4u, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_hdr, d_hdr, sizeof(h_hdr), hipMemcpyDeviceToHost));
    int pack_mism = (h_cnt != h_hdr.cand_cnt) || (h_st2[0] != h_hdr.stats2[0]) || (h_st2[1] != h_hdr.stats2[1]);
    if (!pack_mism && h_cnt > 0 && h_cnt <= kMaxCands) {
        std::vector<int32_t> ci(static_cast<size_t>(h_cnt));
        std::vector<float>   ce(static_cast<size_t>(h_cnt));
        std::vector<DsparkCandPair> cp(static_cast<size_t>(h_cnt));
        HIP_CHECK(hipMemcpy(ci.data(), d_cidx, size_t(h_cnt) * 4u, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ce.data(), d_cexp, size_t(h_cnt) * 4u, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(cp.data(), d_cands, size_t(h_cnt) * sizeof(DsparkCandPair), hipMemcpyDeviceToHost));
        std::vector<std::pair<int32_t, float>> a(static_cast<size_t>(h_cnt));
        std::vector<std::pair<int32_t, float>> b(static_cast<size_t>(h_cnt));
        for (int i = 0; i < h_cnt; ++i) { a[size_t(i)] = {ci[size_t(i)], ce[size_t(i)]}; b[size_t(i)] = {cp[size_t(i)].idx, cp[size_t(i)].exp}; }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) ++pack_mism;
    }

    // 3. Parity: broadcast_noise_and_pos + fill_iota_i32
    launch_dspark_broadcast_noise_and_pos(d_noise, d_xin, d_pos, 321000, B, H, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<uint16_t> h_xin(size_t(B) * size_t(H));
    std::vector<int32_t>  h_pos(B);
    HIP_CHECK(hipMemcpy(h_xin.data(), d_xin, h_xin.size() * 2u, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_pos.data(), d_pos, h_pos.size() * 4u, hipMemcpyDeviceToHost));
    int prep_mism = 0;
    for (int b = 0; b < B; ++b) if (h_pos[size_t(b)] != 321000 + b) ++prep_mism;
    for (int b = 1; b < B; ++b) {
        if (std::memcmp(h_xin.data() + size_t(b) * size_t(H), h_noise.data(), size_t(H) * 2u) != 0) ++prep_mism;
    }
    launch_fill_iota_i32(d_pos, 400000, B, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(h_pos.data(), d_pos, h_pos.size() * 4u, hipMemcpyDeviceToHost));
    for (int b = 0; b < B; ++b) if (h_pos[size_t(b)] != 400000 + b) ++prep_mism;

    BenchRow r;
    r.phase = "DSpark";
    r.item = "C6_markov_bias_multimax";
    r.M = 1;
    r.iters = 20;
    r.total_us = us_fus_mm;
    r.max_abs_diff = mm_max_diff;
    r.speedup = us_ref_mm / us_fus_mm;
    char note[96];
    std::snprintf(note, sizeof(note), "V=%d R=%d cands=%d pack_mism=%d prep_mism=%d", V, R, h_cnt, pack_mism, prep_mism);
    r.note = note;
    add_row(std::move(r));
    std::printf("[dspark-c6] V=%d R=%d  ref=%.1f us  fused=%.1f us (%.2fx)  mm_max_diff=%.3e inf_mism=%d  cands=%d pack_mism=%d prep_mism=%d\n",
                V, R, us_ref_mm, us_fus_mm, us_ref_mm / us_fus_mm, mm_max_diff, inf_mism, h_cnt, pack_mism, prep_mism);
    if (mm_max_diff > 1e-4 || inf_mism != 0 || pack_mism != 0 || prep_mism != 0) {
        throw std::runtime_error("DSpark C6 silicon parity mismatch!");
    }
    bud.release();
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// Coarse phase selector: lets Commits 2/3/4 iterate on a single phase in well
// under a second instead of re-running every unrelated phase. Default is
// kAll, which reproduces the unfiltered benchmark byte-for-byte.
enum class Phase { kAll, kMoe, kDense, kHca, kDsa, kDspark };

static bool phase_on(Phase sel, Phase want) { return sel == Phase::kAll || sel == want; }

static const char* phase_name(Phase p) {
    switch (p) {
        case Phase::kAll:    return "all";
        case Phase::kMoe:    return "moe";
        case Phase::kDense:  return "dense";
        case Phase::kHca:    return "hca";
        case Phase::kDsa:    return "dsa";
        case Phase::kDspark: return "dspark";
    }
    return "all";
}

static Phase parse_phase(const std::string& v) {
    if (v == "all")    return Phase::kAll;
    if (v == "moe")    return Phase::kMoe;
    if (v == "dense")  return Phase::kDense;
    if (v == "hca")    return Phase::kHca;
    if (v == "dsa")    return Phase::kDsa;
    if (v == "dspark") return Phase::kDspark;
    throw std::runtime_error("Unknown --phase value: " + v + " (expected all|moe|dense|hca|dsa|dspark)");
}

int main(int argc, char** argv) {
    MoeOptions moe;
    DenseOptions dense;
    HcaOptions hca;
    DsaOptions dsa;
    bool synthetic = false;
    bool have_m = false;
    Phase phase = Phase::kAll;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&](const char* what) -> const char* {
                if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + what);
                return argv[++i];
            };
            if (a == "--synthetic") synthetic = true;
            else if (a == "--iters") moe.iters = dense.iters = std::atoi(next("--iters"));
            else if (a == "--warmup") moe.warmup = dense.warmup = std::atoi(next("--warmup"));
            else if (a == "--hca-iters") hca.iters = std::atoi(next("--hca-iters"));
            else if (a == "--dsa-iters") dsa.iters = std::atoi(next("--dsa-iters"));
            else if (a == "--heads") hca.n_heads = std::atoi(next("--heads"));
            else if (a == "--verify-rows") moe.verify_rows = std::atoi(next("--verify-rows"));
            else if (a == "--ckpt") moe.ckpt_path = next("--ckpt");
            else if (a == "--m") {
                moe.M = std::atoi(next("--m"));
                have_m = true;
            }
            else if (a == "--phase") phase = parse_phase(next("--phase"));
            else if (a == "--help") {
                std::printf("usage: %s [--phase all|moe|dense|hca|dsa] [--synthetic] [--iters N] [--warmup N] "
                            "[--hca-iters N] [--dsa-iters N] [--heads N] [--verify-rows N] [--ckpt PATH] [--m M]\n",
                            argv[0]);
                return 0;
            }
            else throw std::runtime_error("Unknown argument: " + a);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        std::fprintf(stderr, "usage: %s [--phase all|moe|dense|hca|dsa] [--synthetic] [--iters N] [--warmup N] "
                             "[--hca-iters N] [--dsa-iters N] [--heads N] [--verify-rows N] [--ckpt PATH] [--m M]\n",
                     argv[0]);
        return 2;
    }
    moe.synthetic = synthetic;
    if (!have_m) moe.M = 1024;

    try {
        int dev = 0;
        HIP_CHECK(hipGetDevice(&dev));
        HIP_CHECK(hipSetDevice(dev));
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));
        size_t free_b = 0, total_b = 0;
        HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
        std::printf("==========================================================================================================\n");
        std::printf(" ERNIE-5.5 single-layer gfx906 prefill benchmark (v1.0.0-preview-hlut4-192g baseline)\n");
        std::printf("==========================================================================================================\n");
        std::printf(" device          : GPU[%d] %s (%s), VRAM %.2f GiB total / %.2f GiB free\n",
                    dev, prop.name, prop.gcnArchName, double(total_b) / double(1u << 30),
                    double(free_b) / double(1u << 30));
        std::printf(" experts         : %s (%d local experts, local_eid = e & 7, %.2f MiB)\n",
                    synthetic ? "deterministic synthetic HLUT4" : moe.ckpt_path.c_str(),
                    kLocalExperts, double(kLocalExperts) * double(kExpertBytes) / double(1u << 20));
        std::printf(" vram budget     : %.0f MiB hard cap on GPU[%d]\n", double(kMaxVramBytes) / double(1u << 20), dev);
        std::printf(" phase filter    : %s%s\n", phase_name(phase),
                    phase == Phase::kAll ? "" : " (only the selected phase runs)");
        std::printf("----------------------------------------------------------------------------------------------------------\n");

        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreate(&stream));
        VramBudget bud;

        const auto t0 = std::chrono::steady_clock::now();
        double moe_ms = 0.0, dense_ms = 0.0, hca_ms = 0.0, dsa_ms = 0.0, dspark_ms = 0.0;
        if (phase_on(phase, Phase::kDspark)) {
            const auto t5 = std::chrono::steady_clock::now();
            bench_dspark_c6(bud, stream);
            dspark_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t5).count();
        }
        if (phase_on(phase, Phase::kMoe)) {
            const auto t1 = std::chrono::steady_clock::now();
            for (int M : {1024, 2048}) {
                moe.M = M;
                bench_moe(bud, stream, moe);
            }
            // Wave-10 Commit 4: hybrid m64/m32/m8 expert-sorted CSR parity
            bench_moe_hybrid(bud, stream, moe);
            moe_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        }

        if (phase_on(phase, Phase::kDense) || phase_on(phase, Phase::kHca) || phase_on(phase, Phase::kDsa)) {
            rocblas_handle blas = nullptr;
            ROC_CHECK(rocblas_create_handle(&blas));
            ROC_CHECK(rocblas_set_stream(blas, stream));
            ROC_CHECK(rocblas_set_atomics_mode(blas, rocblas_atomics_not_allowed));
            if (phase_on(phase, Phase::kDense)) {
                const auto t2 = std::chrono::steady_clock::now();
                bench_dense(bud, stream, blas, {1024, 2048}, dense);
                dense_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
            }
            if (phase_on(phase, Phase::kHca)) {
                const auto t3 = std::chrono::steady_clock::now();
                for (int M : {1024, 2048}) bench_hca(bud, stream, blas, M, hca);
                hca_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3).count();
            }
            if (phase_on(phase, Phase::kDsa)) {
                const auto t4 = std::chrono::steady_clock::now();
                for (int M : {1024, 2048}) {
                    bench_dsa(bud, stream, M, 16384, dsa);
                    bench_dsa(bud, stream, M, 32768, dsa);
                }
                bench_dsa_sparse(bud, stream, blas, 1024, 16384, dsa);
                bench_dsa_sparse(bud, stream, blas, 1024, 32768, dsa);
                // Wave-10 Commit 3: TP4 16-head sparse DSA (owner + compacted non-owner)
                bench_dsa_tp4_h16(bud, stream, blas, 512, 16384, dsa);
                // Wave-10 C5b: single-pass K_tile=16384 streaming FI parity
                bench_dsa_flashindexer_streaming(bud, stream, 256, 2038, "edge<=2048");
                bench_dsa_flashindexer_streaming(bud, stream, 256, 16512, "edge=16385");
                bench_dsa_flashindexer_streaming(bud, stream, 256, 49279, "multi>=49152");
                dsa_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t4).count();
            }
            ROC_CHECK(rocblas_destroy_handle(blas));
        }

        bud.release();
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipMemGetInfo(&free_b, &total_b));

        print_table();
        const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("\n peak VRAM tracked : %.2f MiB (cap %.0f MiB) -> %s\n",
                    double(bud.peak) / double(1u << 20), double(kMaxVramBytes) / double(1u << 20),
                    (bud.peak <= kMaxVramBytes) ? "OK" : "VIOLATION");
        auto wall = [](const char* label, double ms, bool ran) {
            if (ran) std::printf(" %-16s : %.3f s\n", label, ms / 1000.0);
            else     std::printf(" %-16s : skipped (--phase)\n", label);
        };
        wall("MoE phase wall", moe_ms, phase_on(phase, Phase::kMoe));
        wall("dense phase wall", dense_ms, phase_on(phase, Phase::kDense));
        wall("HCA phase wall", hca_ms, phase_on(phase, Phase::kHca));
        wall("DSA phase wall", dsa_ms, phase_on(phase, Phase::kDsa));
        wall("DSpark phase wall", dspark_ms, phase_on(phase, Phase::kDspark));
        std::printf(" total wall        : %.3f s\n", total_s);
        std::printf(" free VRAM now     : %.2f GiB\n", double(free_b) / double(1u << 30));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
