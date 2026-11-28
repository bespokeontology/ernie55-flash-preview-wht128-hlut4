// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 / 5x MI50 gfx906 HIP Sequential Quantized-Prefix 512K-Token FP64 RAREBAL-LDLQ Builder,
// Empirical Router-Frequency Permutation Harvester, and Exact BF16 64-Probe GATE-2 Reference Logit Streamer
// for baidu/ERNIE-5.5-Flash-Base-BF16 (and Nov 25 post-trained Apache 2.0 baidu/ERNIE-5.5-Flash).
// Zero Python anywhere. 0 B raw BF16 shard disk retention.

#include "../mi50/ernie55_core_kernels.hpp"
#include "../mi50/ernie55_hlut4_kernels.hpp"
#include "../src/ernie55_config.hpp"
#include "../src/ernie55_hlut4_pack.hpp"
#include "../src/ernie55_json.hpp"
#include "../src/ernie55_range_stream.hpp"
#include "../src/safetensors_mmap.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ernie55 {
namespace {

namespace fs = std::filesystem;

inline void hip_chk(hipError_t e, const char* w) {
    if (e != hipSuccess) throw std::runtime_error(std::string("HIP ") + w + ": " + hipGetErrorString(e));
}
inline void blas_chk(rocblas_status s, const char* w) {
    if (s != rocblas_status_success) throw std::runtime_error(std::string("rocBLAS ") + w + ": " + std::to_string(int(s)));
}

template <typename T>
T* dev_alloc(size_t n, std::vector<void*>& pool, bool zero = false) {
    if (n == 0) return nullptr;
    T* p = nullptr;
    hip_chk(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
    if (zero) hip_chk(hipMemset(p, 0, n * sizeof(T)), "hipMemset");
    pool.push_back(p);
    return p;
}

template <typename T>
T* dev_upload(const void* src, size_t n, std::vector<void*>& pool) {
    T* p = dev_alloc<T>(n, pool, false);
    if (n > 0 && src) hip_chk(hipMemcpy(p, src, n * sizeof(T), hipMemcpyHostToDevice), "H2D");
    return p;
}

std::vector<float> st_to_f32(const hy4::SafetensorsMmap& st, const std::string& name) {
    const auto& e = st.at(name);
    const uint8_t* p = st.tensor_bytes(name);
    if (e.dtype == "F32") {
        std::vector<float> v(size_t(e.byte_size()) / 4u);
        std::memcpy(v.data(), p, size_t(e.byte_size()));
        return v;
    }
    std::vector<float> v(size_t(e.byte_size()) / 2u);
    const auto* b = reinterpret_cast<const uint16_t*>(p);
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_to_f32(b[i]);
    return v;
}

Rope2DTableGpu build_rope_gpu(const std::vector<float>& inv32, std::vector<void*>& pool) {
    constexpr int kDim = 1024 * 32;
    std::vector<float> ch(kDim), sh(kDim), cl(kDim), sl(kDim);
    for (int hi = 0; hi < 1024; ++hi) {
        const double p = double(hi) * 1024.0;
        for (int i = 0; i < 32; ++i) {
            const double a = p * double(inv32[size_t(i)]);
            ch[size_t(hi * 32 + i)] = float(std::cos(a));
            sh[size_t(hi * 32 + i)] = float(std::sin(a));
        }
    }
    for (int lo = 0; lo < 1024; ++lo) {
        const double p = double(lo);
        for (int i = 0; i < 32; ++i) {
            const double a = p * double(inv32[size_t(i)]);
            cl[size_t(lo * 32 + i)] = float(std::cos(a));
            sl[size_t(lo * 32 + i)] = float(std::sin(a));
        }
    }
    Rope2DTableGpu r{};
    r.d_cos_hi = dev_upload<float>(ch.data(), ch.size(), pool);
    r.d_sin_hi = dev_upload<float>(sh.data(), sh.size(), pool);
    r.d_cos_lo = dev_upload<float>(cl.data(), cl.size(), pool);
    r.d_sin_lo = dev_upload<float>(sl.data(), sl.size(), pool);
    return r;
}

// Compute SHA-256 hex via system sha256sum (pure shell utility, 0 Python)
std::string file_sha256(const std::string& path) {
    std::string cmd = "sha256sum '" + path + "'";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return "";
    char buf[128]{};
    std::string hex;
    if (std::fgets(buf, sizeof(buf), fp)) {
        for (int i = 0; i < 64 && buf[i] && buf[i] != ' '; ++i) hex.push_back(buf[i]);
    }
    (void)pclose(fp);
    return hex;
}

struct CalCorpusHeader {
    uint32_t magic, version, n_sequences, seq_len, vocab_size, crc32c;
    uint64_t total_tokens;
};
struct ProbeFileHeader {
    uint32_t magic, version, n_probes, probe_len, vocab_size, crc32c;
};

struct LayerSpineDev {
    std::vector<void*> allocs;
    bool is_dsa = false;
    bool is_dense = false;
    bool is_echo = false;
    bool is_dspark = false;
    // mHC
    float *d_am_fn = nullptr, *d_am_sc = nullptr, *d_am_bs = nullptr;
    float *d_mm_fn = nullptr, *d_mm_sc = nullptr, *d_mm_bs = nullptr;
    uint16_t *d_in_norm = nullptr, *d_post_norm = nullptr;
    // HCA
    uint16_t *h_q_down = nullptr, *h_q_norm = nullptr, *h_q_up = nullptr;
    uint16_t *h_kv_proj = nullptr, *h_kv_norm = nullptr;
    uint16_t *h_comp_wkv = nullptr, *h_comp_wgate = nullptr, *h_comp_norm = nullptr;
    float *h_comp_ape = nullptr, *h_attn_sink = nullptr, *h_vha_m = nullptr;
    uint16_t *h_wo_a = nullptr, *h_gate_proj = nullptr, *h_o_proj = nullptr;
    // DSA
    uint16_t *d_q_a = nullptr, *d_q_a_norm = nullptr, *d_q_b = nullptr;
    uint16_t *d_kv_a = nullptr, *d_kv_a_norm = nullptr, *d_kv_b = nullptr;
    uint16_t *d_idx_wq_b = nullptr, *d_idx_wk = nullptr, *d_idx_kn_w = nullptr, *d_idx_kn_b = nullptr, *d_idx_w_proj = nullptr;
    float *d_attn_sink = nullptr, *d_vha_m = nullptr;
    uint16_t *d_gate_proj = nullptr, *d_o_proj = nullptr;
    // Dense (L00)
    uint16_t *dn_gate = nullptr, *dn_up = nullptr, *dn_down = nullptr;
    // MoE spine
    uint16_t *m_fc1 = nullptr, *m_fc2 = nullptr, *m_lat_norm = nullptr;
    uint16_t *m_sh_w1 = nullptr, *m_sh_w3 = nullptr, *m_sh_w2 = nullptr;
    float *m_router_w01 = nullptr, *m_corr_bias = nullptr, *m_scale_param = nullptr;
    // ECHO L43
    uint16_t *e_enorm = nullptr, *e_hnorm = nullptr, *e_eproj = nullptr, *e_hproj = nullptr;

    void free_all(int dev) {
        (void)hipSetDevice(dev);
        for (void* p : allocs) if (p) (void)hipFree(p);
        allocs.clear();
    }
};

struct GpuBuilderCtx {
    int dev = 0;
    hipStream_t stream = nullptr;
    rocblas_handle blas = nullptr;
    std::vector<void*> perm_allocs;
    Rope2DTableGpu yarn_rope{}, dsa_rope{};

    // Persistent scratch for M <= 512 chunked forward + quantization
    uint16_t *d_res_a = nullptr, *d_res_b = nullptr;
    float    *d_post_mix = nullptr, *d_comb_mix = nullptr;
    uint16_t *d_x_in = nullptr, *d_sub_out = nullptr, *d_dspark_main_x = nullptr;
    uint16_t *d_tmp_a = nullptr, *d_tmp_b = nullptr, *d_tmp_c = nullptr, *d_tmp_d = nullptr;
    uint16_t *d_qr = nullptr, *d_kv_tmp = nullptr, *d_h_lat = nullptr;
    float    *d_f32_tmp0 = nullptr, *d_f32_tmp1 = nullptr;
    int32_t  *d_pos_ids = nullptr, *d_topk_ids = nullptr;
    float    *d_topk_wts = nullptr;
    // Small KV buffers for seq_len <= 1024 calibration / 64 probe forward
    uint16_t *d_hca_comp_kv = nullptr, *d_hca_swa_ring = nullptr;
    float    *d_hca_state_kv = nullptr, *d_hca_state_sc = nullptr;
    uint16_t *d_dsa_mla_kv = nullptr, *d_dsa_ik_cache = nullptr;

    // Quantization & FP64 Gram buffers
    float    *d_levels8_f32 = nullptr, *d_thresh7_f32 = nullptr;
    uint16_t *d_levels8_f16 = nullptr;
    uint64_t *d_signs_in = nullptr, *d_signs_inter = nullptr;
    float    *d_rup_in = nullptr, *d_rup_inter = nullptr;
    uint16_t *d_w_bf16 = nullptr;      // [2048, 2048]
    float    *d_w_f32 = nullptr;       // [2048, 2048]
    float    *d_w_rot = nullptr;       // [2048, 2048]
    Hlut4Tile32x32 *d_tiles_out = nullptr; // [4096]
    float    *d_sq_sig_err2 = nullptr; // [2]
    // For BF16 GATE-2 ref MoE & quantized-prefix MoE
    uint16_t *d_x_rot_f16 = nullptr;   // [512, 2048]
    float    *d_z_f32 = nullptr;       // [512, 2048]
    float    *d_u_f32 = nullptr;       // [512, 2048]
    float    *d_y_acc_f32 = nullptr;   // [512, 2048]
    uint16_t *d_z_rot_f16 = nullptr;   // [512, 2048]
    ExpertDispatchGroup *d_groups = nullptr;
    // FP64 Gram accumulator [2048, 2048] + chunk FP64 matrix [512, 2048]
    double   *d_gram_f64 = nullptr;    // [2048, 2048] (32 MiB)
    double   *d_x_f64 = nullptr;       // [512, 2048]  (8 MiB)

    void init(int hip_dev, const Ernie55ModelConfig& mcfg) {
        dev = hip_dev;
        hip_chk(hipSetDevice(dev), "set dev");
        hip_chk(hipStreamCreate(&stream), "stream");
        blas_chk(rocblas_create_handle(&blas), "blas");
        blas_chk(rocblas_set_stream(blas, stream), "blas stream");

        yarn_rope = build_rope_gpu(compute_hca_yarn_inv_freq(
            kHcaRopeHeadDim, mcfg.compress_rope_theta, mcfg.yarn_factor,
            mcfg.yarn_orig_max_pos, mcfg.yarn_beta_fast, mcfg.yarn_beta_slow), perm_allocs);
        dsa_rope = build_rope_gpu(compute_dsa_inv_freq(kDsaQkRopeHeadDim, mcfg.rope_theta), perm_allocs);

        constexpr size_t kM = 512;
        d_res_a    = dev_alloc<uint16_t>(kM * kHcDim, perm_allocs, true);
        d_res_b    = dev_alloc<uint16_t>(kM * kHcDim, perm_allocs, true);
        d_post_mix = dev_alloc<float>(kM * kHcMult, perm_allocs, true);
        d_comb_mix = dev_alloc<float>(kM * kHcMult2, perm_allocs, true);
        d_x_in     = dev_alloc<uint16_t>(kM * kHiddenSize, perm_allocs, true);
        d_sub_out  = dev_alloc<uint16_t>(kM * kHiddenSize, perm_allocs, true);
        d_dspark_main_x = dev_alloc<uint16_t>(kM * kHiddenSize, perm_allocs, true);
        d_tmp_a    = dev_alloc<uint16_t>(kM * 32768u, perm_allocs, true);
        d_tmp_b    = dev_alloc<uint16_t>(kM * 32768u, perm_allocs, true);
        d_tmp_c    = dev_alloc<uint16_t>(kM * 36864u, perm_allocs, true);
        d_tmp_d    = dev_alloc<uint16_t>(kM * 16384u, perm_allocs, true);
        d_qr       = dev_alloc<uint16_t>(kM * 1024u, perm_allocs, true);
        d_kv_tmp   = dev_alloc<uint16_t>(kM * 576u, perm_allocs, true);
        d_h_lat    = dev_alloc<uint16_t>(kM * kLatentDim, perm_allocs, true);
        d_f32_tmp0 = dev_alloc<float>(kM * kLatentDim, perm_allocs, true);
        d_f32_tmp1 = dev_alloc<float>(kM * 1024u, perm_allocs, true);
        d_pos_ids  = dev_alloc<int32_t>(kM, perm_allocs, true);
        d_topk_ids = dev_alloc<int32_t>(kM * kTopKExperts, perm_allocs, true);
        d_topk_wts = dev_alloc<float>(kM * kTopKExperts, perm_allocs, true);

        d_hca_comp_kv  = dev_alloc<uint16_t>(16u * 512u, perm_allocs, true);
        d_hca_swa_ring = dev_alloc<uint16_t>(256u * 512u, perm_allocs, true);
        d_hca_state_kv = dev_alloc<float>(128u * 512u, perm_allocs, true);
        d_hca_state_sc = dev_alloc<float>(128u * 512u, perm_allocs, true);
        d_dsa_mla_kv   = dev_alloc<uint16_t>(1024u * 576u, perm_allocs, true);
        d_dsa_ik_cache = dev_alloc<uint16_t>(1024u * 128u, perm_allocs, true);

        const auto& sol = hlut4_lloyd_max();
        float lv8[8], th7[7];
        uint16_t lv16[8];
        for (int i = 0; i < 8; ++i) { lv8[i] = float(sol.levels[size_t(i)]); lv16[i] = f32_to_fp16_rne(lv8[i]); }
        for (int i = 0; i < 7; ++i) th7[i] = float(sol.thresholds[size_t(i)]);
        d_levels8_f32 = dev_upload<float>(lv8, 8, perm_allocs);
        d_thresh7_f32 = dev_upload<float>(th7, 7, perm_allocs);
        d_levels8_f16 = dev_upload<uint16_t>(lv16, 8, perm_allocs);
        d_signs_in    = dev_alloc<uint64_t>(kLatentDim / 64, perm_allocs, true);
        d_signs_inter = dev_alloc<uint64_t>(kMoeInterSize / 64, perm_allocs, true);
        d_rup_in      = dev_alloc<float>(size_t(kLatentDim) * kLatentDim, perm_allocs, true);
        d_rup_inter   = dev_alloc<float>(size_t(kMoeInterSize) * kMoeInterSize, perm_allocs, true);

        d_w_bf16      = dev_alloc<uint16_t>(size_t(kMoeInterSize) * kLatentDim, perm_allocs, false);
        d_w_f32       = dev_alloc<float>(size_t(kMoeInterSize) * kLatentDim, perm_allocs, false);
        d_w_rot       = dev_alloc<float>(size_t(kMoeInterSize) * kLatentDim, perm_allocs, false);
        d_tiles_out   = dev_alloc<Hlut4Tile32x32>(kExpProjTiles, perm_allocs, false);
        d_sq_sig_err2 = dev_alloc<float>(2, perm_allocs, true);

        d_x_rot_f16 = dev_alloc<uint16_t>(kM * kLatentDim, perm_allocs, true);
        d_z_f32     = dev_alloc<float>(256u * kMaxGroupTokens * kMoeInterSize, perm_allocs, true);
        d_u_f32     = dev_alloc<float>(kM * kMoeInterSize, perm_allocs, true);
        d_y_acc_f32 = dev_alloc<float>(kM * kLatentDim, perm_allocs, true);
        d_z_rot_f16 = dev_alloc<uint16_t>(256u * kMaxGroupTokens * kMoeInterSize, perm_allocs, true);
        d_groups    = dev_alloc<ExpertDispatchGroup>(256u, perm_allocs, true);

        d_gram_f64 = dev_alloc<double>(size_t(kLatentDim) * kLatentDim, perm_allocs, true);
        d_x_f64    = dev_alloc<double>(kM * kLatentDim, perm_allocs, true);
    }

    void destroy() {
        (void)hipSetDevice(dev);
        if (stream) (void)hipStreamSynchronize(stream);
        if (blas)   (void)rocblas_destroy_handle(blas);
        for (void* p : perm_allocs) if (p) (void)hipFree(p);
        perm_allocs.clear();
        if (stream) (void)hipStreamDestroy(stream);
    }

    void bf16_gemm(const uint16_t* dx, const uint16_t* dw, uint16_t* dy, int M, int N, int K) {
        if (M <= 4) {
            launch_bf16_gemv_bf16(dx, dw, dy, M, N, K, stream);
        } else {
            const float a = 1.0f, b = 0.0f;
            blas_chk(rocblas_gemm_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                N, M, K, &a,
                dw, rocblas_datatype_bf16_r, K,
                dx, rocblas_datatype_bf16_r, K,
                &b,
                dy, rocblas_datatype_bf16_r, N,
                dy, rocblas_datatype_bf16_r, N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "bf16_gemm");
        }
    }
    void bf16_gemm_f32(const uint16_t* dx, const uint16_t* dw, float* dy, int M, int N, int K) {
        if (M <= 4) {
            launch_bf16_gemv_f32(dx, dw, dy, M, N, K, stream);
        } else {
            const float a = 1.0f, b = 0.0f;
            blas_chk(rocblas_gemm_ex(
                blas, rocblas_operation_transpose, rocblas_operation_none,
                N, M, K, &a,
                dw, rocblas_datatype_bf16_r, K,
                dx, rocblas_datatype_bf16_r, K,
                &b,
                dy, rocblas_datatype_f32_r, N,
                dy, rocblas_datatype_f32_r, N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "bf16_gemm_f32");
        }
    }
    void f32_gemm(const float* dx, const float* dw, float* dy, int M, int N, int K, float beta = 0.0f) {
        const float a = 1.0f;
        blas_chk(rocblas_sgemm(blas, rocblas_operation_transpose, rocblas_operation_none,
                               N, M, K, &a, dw, K, dx, K, &beta, dy, N), "sgemm");
    }
};

void upload_spine_to_gpu(const hy4::SafetensorsMmap& st, const Ernie55ModelConfig& mcfg, int l,
                         GpuBuilderCtx& g, LayerSpineDev& L) {
    hip_chk(hipSetDevice(g.dev), "set dev spine");
    L.is_echo  = (l == mcfg.num_hidden_layers);
    L.is_dsa   = L.is_echo || mcfg.is_dsa_layer(l);
    L.is_dense = (l < mcfg.first_k_dense_replace);

    auto up_bf16 = [&](const std::string& nm) -> uint16_t* {
        const auto& e = st.at(nm);
        return dev_upload<uint16_t>(st.tensor_bytes(nm), size_t(e.byte_size()) / 2u, L.allocs);
    };
    auto up_f32 = [&](const std::string& nm) -> float* {
        auto v = st_to_f32(st, nm);
        return dev_upload<float>(v.data(), v.size(), L.allocs);
    };

    L.d_am_fn = up_f32("attn_mhc.mapping_proj.weight");
    L.d_am_sc = up_f32("attn_mhc.hc_scale3");
    L.d_am_bs = up_f32("attn_mhc.bias");
    L.d_mm_fn = up_f32("mlp_mhc.mapping_proj.weight");
    L.d_mm_sc = up_f32("mlp_mhc.hc_scale3");
    L.d_mm_bs = up_f32("mlp_mhc.bias");
    L.d_in_norm   = up_bf16("input_layernorm.weight");
    L.d_post_norm = up_bf16("post_attention_layernorm.weight");

    if (!L.is_dsa) {
        L.h_q_down = up_bf16("attn.q_down"); L.h_q_norm = up_bf16("attn.q_norm"); L.h_q_up = up_bf16("attn.q_up");
        L.h_kv_proj = up_bf16("attn.kv_proj"); L.h_kv_norm = up_bf16("attn.kv_norm");
        L.h_comp_wkv = up_bf16("attn.comp_wkv"); L.h_comp_wgate = up_bf16("attn.comp_wgate");
        L.h_comp_norm = up_bf16("attn.comp_norm"); L.h_comp_ape = up_f32("attn.comp_ape");
        L.h_attn_sink = up_f32("attn.attn_sink"); L.h_vha_m = up_f32("attn.vha_m");
        L.h_wo_a = up_bf16("attn.wo_a"); L.h_gate_proj = up_bf16("attn.gate_proj"); L.h_o_proj = up_bf16("attn.o_proj");
    } else {
        L.d_q_a = up_bf16("attn.q_a"); L.d_q_a_norm = up_bf16("attn.q_a_norm"); L.d_q_b = up_bf16("attn.q_b");
        L.d_kv_a = up_bf16("attn.kv_a"); L.d_kv_a_norm = up_bf16("attn.kv_a_norm"); L.d_kv_b = up_bf16("attn.kv_b");
        L.d_idx_wq_b = up_bf16("attn.idx_wq_b"); L.d_idx_wk = up_bf16("attn.idx_wk");
        L.d_idx_kn_w = up_bf16("attn.idx_kn_w"); L.d_idx_kn_b = up_bf16("attn.idx_kn_b");
        L.d_idx_w_proj = up_bf16("attn.idx_w_proj");
        L.d_attn_sink = up_f32("attn.attn_sink"); L.d_vha_m = up_f32("attn.vha_m");
        L.d_gate_proj = up_bf16("attn.gate_proj"); L.d_o_proj = up_bf16("attn.o_proj");
    }

    if (L.is_dense) {
        L.dn_gate = up_bf16("mlp.gate"); L.dn_up = up_bf16("mlp.up"); L.dn_down = up_bf16("mlp.down");
    } else {
        L.m_fc1 = up_bf16("moe.fc1_lat"); L.m_fc2 = up_bf16("moe.fc2_lat"); L.m_lat_norm = up_bf16("moe.lat_norm");
        L.m_sh_w1 = up_bf16("moe.sh_w1"); L.m_sh_w3 = up_bf16("moe.sh_w3"); L.m_sh_w2 = up_bf16("moe.sh_w2");
        L.m_router_w01 = up_f32("moe.router_w01"); L.m_corr_bias = up_f32("moe.corr_bias"); L.m_scale_param = up_f32("moe.scale_param");
    }
    if (L.is_echo) {
        L.e_enorm = up_bf16("echo.enorm"); L.e_hnorm = up_bf16("echo.hnorm");
        L.e_eproj = up_bf16("echo.eproj"); L.e_hproj = up_bf16("echo.hproj");
    }
}

void upload_dspark_spine_to_gpu(const hy4::SafetensorsMmap& st, int dspark_i,
                                GpuBuilderCtx& g, LayerSpineDev& L) {
    hip_chk(hipSetDevice(g.dev), "set dev dspark spine");
    L.is_echo   = false;
    L.is_dspark = true;
    L.is_dsa    = false;
    L.is_dense  = false;
    const std::string p = "dspark.layers." + std::to_string(dspark_i) + ".";

    auto up_bf16 = [&](const std::string& nm) -> uint16_t* {
        const auto& e = st.at(p + nm);
        return dev_upload<uint16_t>(st.tensor_bytes(p + nm), size_t(e.byte_size()) / 2u, L.allocs);
    };
    auto up_f32 = [&](const std::string& nm) -> float* {
        auto v = st_to_f32(st, p + nm);
        return dev_upload<float>(v.data(), v.size(), L.allocs);
    };

    L.d_am_fn = up_f32("attn_mhc.mapping_proj.weight");
    L.d_am_sc = up_f32("attn_mhc.hc_scale3");
    L.d_am_bs = up_f32("attn_mhc.bias");
    L.d_mm_fn = up_f32("mlp_mhc.mapping_proj.weight");
    L.d_mm_sc = up_f32("mlp_mhc.hc_scale3");
    L.d_mm_bs = up_f32("mlp_mhc.bias");
    L.d_in_norm   = up_bf16("input_layernorm.weight");
    L.d_post_norm = up_bf16("post_attention_layernorm.weight");

    L.h_q_down = up_bf16("attn.q_down"); L.h_q_norm = up_bf16("attn.q_norm"); L.h_q_up = up_bf16("attn.q_up");
    L.h_kv_proj = up_bf16("attn.kv_proj"); L.h_kv_norm = up_bf16("attn.kv_norm");
    L.h_attn_sink = up_f32("attn.attn_sink"); L.h_vha_m = up_f32("attn.vha_m");
    L.h_wo_a = up_bf16("attn.wo_a"); L.h_gate_proj = up_bf16("attn.gate_proj"); L.h_o_proj = up_bf16("attn.o_proj");

    L.m_fc1 = up_bf16("moe.fc1_lat"); L.m_fc2 = up_bf16("moe.fc2_lat"); L.m_lat_norm = up_bf16("moe.lat_norm");
    L.m_sh_w1 = up_bf16("moe.sh_w1"); L.m_sh_w3 = up_bf16("moe.sh_w3"); L.m_sh_w2 = up_bf16("moe.sh_w2");
    L.m_router_w01 = up_f32("moe.router_w01"); L.m_corr_bias = up_f32("moe.corr_bias"); L.m_scale_param = up_f32("moe.scale_param");
}

void run_attn_chunk_gpu(GpuBuilderCtx& g, const LayerSpineDev& L, const Ernie55ModelConfig& mcfg, int M, int32_t pos0) {
    if (!L.is_dsa) {
        const auto& rope_use = L.is_dspark ? g.dsa_rope : g.yarn_rope;
        g.bf16_gemm(g.d_x_in, L.h_q_down, g.d_tmp_d, M, 1024, kHiddenSize);
        launch_rmsnorm_bf16(g.d_tmp_d, L.h_q_norm, g.d_qr, M, 1024, mcfg.rms_norm_eps, g.stream);
        g.bf16_gemm(L.is_dspark ? g.d_dspark_main_x : g.d_x_in, L.h_kv_proj, g.d_tmp_d, M, 512, kHiddenSize);
        launch_rmsnorm_bf16(g.d_tmp_d, L.h_kv_norm, g.d_kv_tmp, M, 512, mcfg.rms_norm_eps, g.stream);
        g.bf16_gemm(g.d_qr, L.h_q_up, g.d_tmp_a, M, 32768, 1024);
        launch_hca_q_qknorm_rope(g.d_tmp_a, g.d_pos_ids, rope_use, M, kHcaNumHeads, mcfg.rms_norm_eps, g.stream);
        launch_hca_kv_rope_inplace(g.d_kv_tmp, g.d_pos_ids, rope_use, M, g.stream);

        if (!L.is_dspark) {
            g.bf16_gemm_f32(g.d_x_in, L.h_comp_wkv, g.d_f32_tmp0, M, 512, kHiddenSize);
            g.bf16_gemm_f32(g.d_x_in, L.h_comp_wgate, g.d_f32_tmp1, M, 512, kHiddenSize);
            launch_hca_prefill_compress_blocks(
                g.d_f32_tmp0, g.d_f32_tmp1, L.h_comp_ape,
                g.d_hca_state_kv, g.d_hca_state_sc, L.h_comp_norm, g.yarn_rope,
                pos0, M, g.d_hca_comp_kv, mcfg.rms_norm_eps, g.stream);
            launch_hca_prefill_attn_c128_swa_sink(
                g.d_tmp_a, g.d_hca_comp_kv, g.d_hca_swa_ring, g.d_kv_tmp,
                L.h_attn_sink, g.d_tmp_b, kHcaNumHeads, pos0, M, g.stream);
            launch_hca_prefill_commit_tail(
                g.d_kv_tmp, g.d_f32_tmp0, g.d_f32_tmp1, L.h_comp_ape,
                pos0, M, g.d_hca_swa_ring, g.d_hca_state_kv, g.d_hca_state_sc, g.stream);
        } else {
            // Pure 128-token SWA attention (M <= 128, pos0 == 0 => num_comp == 0 -> exact SWA-only!)
            launch_hca_prefill_attn_c128_swa_sink(
                g.d_tmp_a, g.d_hca_comp_kv, g.d_hca_swa_ring, g.d_kv_tmp,
                L.h_attn_sink, g.d_tmp_b, kHcaNumHeads, 0, M, g.stream);
        }

        launch_hca_vha_invrope(g.d_tmp_b, L.h_vha_m, g.d_pos_ids, rope_use, g.d_tmp_a, M, g.stream);
        g.bf16_gemm(g.d_qr, L.h_gate_proj, g.d_tmp_d, M, 8192, 1024);
        launch_hca_grouped_wo_a_lrga(g.d_tmp_a, L.h_wo_a, g.d_tmp_d, g.d_tmp_b, M, kHcaOGroups, kHcaOLoraRank, g.stream);
        g.bf16_gemm(g.d_tmp_b, L.h_o_proj, g.d_sub_out, M, kHiddenSize, 8192);
    } else {
        g.bf16_gemm(g.d_x_in, L.d_q_a, g.d_tmp_d, M, 1024, kHiddenSize);
        launch_rmsnorm_bf16(g.d_tmp_d, L.d_q_a_norm, g.d_qr, M, 1024, mcfg.rms_norm_eps, g.stream);
        g.bf16_gemm(g.d_qr, L.d_q_b, g.d_tmp_a, M, 16384, 1024);
        g.bf16_gemm(g.d_x_in, L.d_kv_a, g.d_kv_tmp, M, 576, kHiddenSize);
        launch_dsa_prefill_kv_norm_rope_append(
            g.d_tmp_a, g.d_kv_tmp, L.d_kv_a_norm, g.d_pos_ids, g.dsa_rope,
            g.d_dsa_mla_kv, M, kDsaNumHeads, 1024, mcfg.rms_norm_eps, g.stream);
        launch_dsa_q_absorb_uk(g.d_tmp_a, L.d_kv_b, g.d_tmp_c, M, kDsaNumHeads, g.stream);
        launch_dsa_prefill_mla_dense_causal_sink(
            g.d_tmp_c, g.d_dsa_mla_kv, L.d_attn_sink, g.d_tmp_a, kDsaNumHeads, pos0, M, g.stream);
        launch_dsa_o_project_uv(g.d_tmp_a, L.d_kv_b, g.d_tmp_b, M, kDsaNumHeads, g.stream);
        g.bf16_gemm(g.d_qr, L.d_gate_proj, g.d_tmp_d, M, 16384, 1024);
        launch_dsa_vha_lrga(g.d_tmp_b, L.d_vha_m, g.d_tmp_d, g.d_tmp_a, M, g.stream);
        g.bf16_gemm(g.d_tmp_a, L.d_o_proj, g.d_sub_out, M, kHiddenSize, 16384);
    }
}

} // namespace
} // namespace ernie55

int main(int argc, char** argv) {
    using namespace ernie55;
    namespace fs = std::filesystem;

    std::string hf_repo     = "baidu/ERNIE-5.5-Flash-Preview-BF16";
    std::string revision    = "main";
    std::string root_dir    = ".";
    std::string meta_dir    = "./meta/preview-bf16";
    std::string out_dir     = "./checkpoints/preview-bf16";
    std::string experts_dir = "./weights_hlut4/experts_hlut4";
    std::string cal_bin     = "./artifacts/cal_ernie55_preview_disjoint.bin";
    std::string probe_bin   = "./artifacts/gate2_preview_heldout_probes.bin";
    int http_workers        = 24;
    int max_cal_seqs        = 512;  // 512 x 1024 = 524,288 (512K) tokens!
    int layer_begin         = 1;
    int layer_end           = 45;   // L01..L42 backbone + L43..L45 (3L MoE DSpark)!
    int num_gpus            = 5;
    double ldlq_damp        = 0.01;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg ") + f);
            return argv[++i];
        };
        if (a == "--hf-repo")            hf_repo = need("--hf-repo");
        else if (a == "--revision")      revision = need("--revision");
        else if (a == "--root")          root_dir = need("--root");
        else if (a == "--meta-dir")      meta_dir = need("--meta-dir");
        else if (a == "--out-dir")       out_dir = need("--out-dir");
        else if (a == "--experts-dir")   experts_dir = need("--experts-dir");
        else if (a == "--cal-corpus")    cal_bin = need("--cal-corpus");
        else if (a == "--probe-bin")     probe_bin = need("--probe-bin");
        else if (a == "--http-workers")  http_workers = std::stoi(need("--http-workers"));
        else if (a == "--cal-seqs")      max_cal_seqs = std::stoi(need("--cal-seqs"));
        else if (a == "--layer-begin")   layer_begin = std::stoi(need("--layer-begin"));
        else if (a == "--layer-end")     layer_end = std::stoi(need("--layer-end"));
        else if (a == "--num-gpus")      num_gpus = std::stoi(need("--num-gpus"));
        else if (a == "--damp")          ldlq_damp = std::stod(need("--damp"));
    }

    try {
        const auto t_start_all = std::chrono::steady_clock::now();
        int avail_gpus = 0;
        hip_chk(hipGetDeviceCount(&avail_gpus), "hipGetDeviceCount");
        num_gpus = std::min(num_gpus, avail_gpus);
        if (num_gpus <= 0) throw std::runtime_error("No MI50 GPUs available!");

        const auto mcfg = Ernie55ModelConfig::load_json(meta_dir + "/config.json");
        mcfg.validate_compat();

        RemoteRangeIndex idx;
        idx.repo = hf_repo;
        idx.revision = revision;
        const std::string tsv = meta_dir + "/tensor_ranges.tsv";
        if (fs::is_regular_file(tsv)) {
            idx.load_tsv(tsv);
        } else {
            idx.build_from_hf_index(meta_dir + "/model.safetensors.index.json", tsv, http_workers);
        }
        std::printf("[ERNIE55-GPU-BUILD] %d MI50 GPUs | manifest=%zu tensors | cal=%s | probes=%s\n",
                    num_gpus, idx.all_names.size(), cal_bin.c_str(), probe_bin.c_str());

        // 1. Load disjoint calibration corpus (up to 524,288 tokens) & 64 held-out GATE-2 probes (4,096 tokens)
        CalCorpusHeader ch{};
        std::vector<int32_t> cal_ids;
        {
            std::ifstream ifs(cal_bin, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open cal corpus: " + cal_bin);
            ifs.read(reinterpret_cast<char*>(&ch), sizeof(ch));
            if (ch.magic != 0x43414C35u) throw std::runtime_error("Invalid cal corpus magic");
            const int use_seqs = std::min(int(ch.n_sequences), max_cal_seqs);
            ch.n_sequences = uint32_t(use_seqs);
            cal_ids.resize(size_t(use_seqs) * size_t(ch.seq_len));
            ifs.read(reinterpret_cast<char*>(cal_ids.data()), std::streamsize(cal_ids.size() * sizeof(int32_t)));
        }
        const int n_cal_seqs = int(ch.n_sequences);
        const int cal_seq_len = int(ch.seq_len);
        const size_t n_cal_toks = size_t(n_cal_seqs) * size_t(cal_seq_len);

        ProbeFileHeader ph{};
        std::vector<int32_t> probe_ids;
        {
            std::ifstream ifs(probe_bin, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open probe bin: " + probe_bin);
            ifs.read(reinterpret_cast<char*>(&ph), sizeof(ph));
            if (ph.magic != 0x50524235u) throw std::runtime_error("Invalid probe bin magic");
            probe_ids.resize(size_t(ph.n_probes) * size_t(ph.probe_len));
            ifs.read(reinterpret_cast<char*>(probe_ids.data()), std::streamsize(probe_ids.size() * sizeof(int32_t)));
        }
        const int n_probes = int(ph.n_probes);
        const int probe_len = int(ph.probe_len);
        const size_t n_probe_toks = size_t(n_probes) * size_t(probe_len);
        std::printf("[ERNIE55-GPU-BUILD] Calibration: %zu tokens (%d x %d) | GATE-2 BF16 Oracle: %zu tokens (%d x %d)\n",
                    n_cal_toks, n_cal_seqs, cal_seq_len, n_probe_toks, n_probes, probe_len);
        std::fflush(stdout);

        // 2. Initialize all 5 MI50 builder contexts
        std::vector<GpuBuilderCtx> gpus(static_cast<size_t>(num_gpus));
        for (int d = 0; d < num_gpus; ++d) gpus[size_t(d)].init(d, mcfg);

        // 3. Load embeddings from local packed_spine/spine_shared.safetensors and run Embed + L00 Dense!
        std::vector<uint16_t> cal_res(n_cal_toks * kHcDim, 0u);     // 16 GiB @ 512K tokens
        std::vector<uint16_t> ref_res(n_probe_toks * kHcDim, 0u);   // 128 MiB @ 4,096 probe tokens
        std::vector<uint16_t> cal_x_emb_shift;                      // optional for ECHO L43

        {
            hy4::SafetensorsMmap st_sh(out_dir + "/packed_spine/spine_shared.safetensors");
            const auto* emb = reinterpret_cast<const uint16_t*>(st_sh.tensor_bytes("model.embed_tokens.weight"));

            // Populate initial token embeddings (we pack them into stream 0 before L00 broadcast!)
            std::vector<uint16_t> cal_x0(n_cal_toks * kHiddenSize);
            #pragma omp parallel for schedule(static)
            for (size_t t = 0; t < n_cal_toks; ++t) {
                const int32_t tid = std::clamp(cal_ids[t], 0, mcfg.vocab_size - 1);
                std::memcpy(cal_x0.data() + t * kHiddenSize, emb + size_t(tid) * kHiddenSize, kHiddenSize * 2u);
            }
            std::vector<uint16_t> ref_x0(n_probe_toks * kHiddenSize);
            for (size_t t = 0; t < n_probe_toks; ++t) {
                const int32_t tid = std::clamp(probe_ids[t], 0, mcfg.vocab_size - 1);
                std::memcpy(ref_x0.data() + t * kHiddenSize, emb + size_t(tid) * kHiddenSize, kHiddenSize * 2u);
            }

            // Also stash shifted next-token embeddings for ECHO L43 calibration (first 32K tokens = 256 MiB!)
            const size_t echo_toks = std::min<size_t>(n_cal_toks, 32768u);
            cal_x_emb_shift.resize(echo_toks * kHiddenSize);
            #pragma omp parallel for schedule(static)
            for (size_t t = 0; t < echo_toks; ++t) {
                const int32_t tid = std::clamp(cal_ids[(t + 1) % n_cal_toks], 0, mcfg.vocab_size - 1);
                std::memcpy(cal_x_emb_shift.data() + t * kHiddenSize, emb + size_t(tid) * kHiddenSize, kHiddenSize * 2u);
            }
            st_sh.release_pages();

            // Run L00 Dense (HCA + Dense MLP) 5-way data-parallel across all 5 MI50s!
            const auto t_l0 = std::chrono::steady_clock::now();
            hy4::SafetensorsMmap st0(out_dir + "/packed_spine/spine_layer_00.safetensors");
            std::vector<LayerSpineDev> l0_devs(static_cast<size_t>(num_gpus));
            for (int d = 0; d < num_gpus; ++d) upload_spine_to_gpu(st0, mcfg, 0, gpus[size_t(d)], l0_devs[size_t(d)]);
            st0.release_pages();

            auto run_l0_seqs = [&](const uint16_t* x0_in, uint16_t* res_out, int n_seqs, int slen) {
                #pragma omp parallel for num_threads(num_gpus) schedule(static)
                for (int d = 0; d < num_gpus; ++d) {
                    auto& g = gpus[size_t(d)];
                    auto& L0 = l0_devs[size_t(d)];
                    hip_chk(hipSetDevice(g.dev), "l0 dev");
                    std::vector<int32_t> pos_v(512);
                    for (int s = d; s < n_seqs; s += num_gpus) {
                        hip_chk(hipMemsetAsync(g.d_hca_swa_ring, 0, 256u * 512u * 2u, g.stream), "z swa");
                        hip_chk(hipMemsetAsync(g.d_hca_state_kv, 0, 128u * 512u * 4u, g.stream), "z skv");
                        hip_chk(hipMemsetAsync(g.d_hca_state_sc, 0, 128u * 512u * 4u, g.stream), "z ssc");
                        for (int off = 0; off < slen; off += 512) {
                            const int m = std::min(512, slen - off);
                            for (int i = 0; i < m; ++i) pos_v[size_t(i)] = off + i;
                            const size_t tok_off = size_t(s) * size_t(slen) + size_t(off);
                            hip_chk(hipMemcpyAsync(g.d_x_in, x0_in + tok_off * kHiddenSize, size_t(m) * kHiddenSize * 2u, hipMemcpyHostToDevice, g.stream), "H2D x0");
                            hip_chk(hipMemcpyAsync(g.d_pos_ids, pos_v.data(), size_t(m) * 4u, hipMemcpyHostToDevice, g.stream), "H2D pos");
                            launch_mhc_pre_broadcast_norm(g.d_x_in, L0.d_am_fn, L0.d_am_sc, L0.d_am_bs, L0.d_in_norm,
                                                          g.d_res_a, g.d_post_mix, g.d_comb_mix, g.d_x_in, m, kHiddenSize,
                                                          mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
                            run_attn_chunk_gpu(g, L0, mcfg, m, off);
                            launch_mhc_post_update(g.d_sub_out, g.d_res_a, g.d_post_mix, g.d_comb_mix, g.d_res_b, m, kHiddenSize, g.stream);
                            launch_mhc_pre_norm(g.d_res_b, L0.d_mm_fn, L0.d_mm_sc, L0.d_mm_bs, L0.d_post_norm,
                                                g.d_post_mix, g.d_comb_mix, g.d_x_in, m, kHiddenSize,
                                                mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
                            g.bf16_gemm(g.d_x_in, L0.dn_gate, g.d_tmp_a, m, mcfg.intermediate_size, kHiddenSize);
                            g.bf16_gemm(g.d_x_in, L0.dn_up,   g.d_tmp_b, m, mcfg.intermediate_size, kHiddenSize);
                            launch_situ_glu_bf16(g.d_tmp_a, g.d_tmp_b, g.d_tmp_a, m, mcfg.intermediate_size,
                                                 mcfg.situ_beta, mcfg.situ_linear_beta, g.stream);
                            g.bf16_gemm(g.d_tmp_a, L0.dn_down, g.d_sub_out, m, kHiddenSize, mcfg.intermediate_size);
                            launch_mhc_post_update(g.d_sub_out, g.d_res_b, g.d_post_mix, g.d_comb_mix, g.d_res_a, m, kHiddenSize, g.stream);
                            hip_chk(hipMemcpyAsync(res_out + tok_off * kHcDim, g.d_res_a, size_t(m) * kHcDim * 2u, hipMemcpyDeviceToHost, g.stream), "D2H l0");
                        }
                    }
                    hip_chk(hipStreamSynchronize(g.stream), "sync l0");
                }
            };
            run_l0_seqs(cal_x0.data(), cal_res.data(), n_cal_seqs, cal_seq_len);
            run_l0_seqs(ref_x0.data(), ref_res.data(), n_probes, probe_len);
            for (int d = 0; d < num_gpus; ++d) l0_devs[size_t(d)].free_all(d);
            std::printf("[ERNIE55-GPU-BUILD] L00 Dense forward complete (%zu cal + %zu ref tokens in %.1fs)\n",
                        n_cal_toks, n_probe_toks,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_l0).count());
            std::fflush(stdout);
        }

        // Allocate host buffers reused across layers L01..L43
        std::vector<uint16_t> cal_lat_bf16(n_cal_toks * kLatentDim);         // 2.0 GiB @ 512K
        std::vector<uint16_t> cal_shared_bf16(n_cal_toks * kHiddenSize);     // 4.0 GiB @ 512K
        std::vector<int16_t>  cal_topk_ids(n_cal_toks * kTopKExperts);       // 10 MiB
        std::vector<float>    cal_topk_wts(n_cal_toks * kTopKExperts);       // 20 MiB
        std::vector<float>    cal_mlp_post_mix(n_cal_toks * kHcMult);        // 8 MiB
        std::vector<float>    cal_mlp_comb_mix(n_cal_toks * kHcMult2);       // 32 MiB

        // GATE-2 exact BF16 oracle layer state (4,096 tokens!)
        std::vector<float>    ref_lat_f32(n_probe_toks * kLatentDim);
        std::vector<uint16_t> ref_shared_bf16(n_probe_toks * kHiddenSize);
        std::vector<int32_t>  ref_topk_ids(n_probe_toks * kTopKExperts);
        std::vector<float>    ref_topk_wts(n_probe_toks * kTopKExperts);
        std::vector<float>    ref_mlp_post_mix(n_probe_toks * kHcMult);
        std::vector<float>    ref_mlp_comb_mix(n_probe_toks * kHcMult2);
        std::vector<float>    ref_moe_acc(n_probe_toks * kLatentDim, 0.0f);

        // Save per-layer GATE-2 reference router top-10 IDs for EAR@10!
        std::vector<int16_t> all_ref_router_top10(size_t(mcfg.num_hidden_layers) * n_probe_toks * kTopKExperts, 0);

        // Global router histogram & frequency permutation tables across all 43 MoE layers!
        std::array<std::array<uint64_t, kNumRoutedExperts>, kTotalLayersWithEcho> router_hist{};
        std::array<std::array<uint16_t, kNumRoutedExperts>, kTotalLayersWithEcho> slot_to_raw{};
        for (int l = 0; l < kTotalLayersWithEcho; ++l)
            for (int e = 0; e < kNumRoutedExperts; ++e) slot_to_raw[size_t(l)][size_t(e)] = uint16_t(e);

        const std::string store_dir = experts_dir.empty() ? (out_dir + "/experts_hlut4") : experts_dir;
        const std::string link_dir  = out_dir + "/experts_hlut4";
        fs::create_directories(store_dir);
        fs::create_directories(link_dir);
        fs::create_directories(root_dir + "/artifacts");

        struct LayerReceipt {
            int layer = 0;
            double sqnr_db = 0.0, cond_in = 0.0, cond_inter = 0.0, rarebal_lam = 1.0;
            double mi50_hit_pct = 0.0, mi50_l1_hit_pct = 0.0;
            int rare_fallback_experts = 0;
            double stream_s = 0.0, total_s = 0.0;
            std::string sha256;
        };
        std::vector<LayerReceipt> receipts;

        // DSpark authentic calibration buffers (first 32,768 tokens = 32 x 1024):
        // aux_l40, aux_l41, aux_l42 = mean4(cal_res) after L40, L41, L42 -> main_x = main_norm(main_proj(cat))!
        const size_t dspark_cal_toks = std::min<size_t>(n_cal_toks, 32768u);
        std::vector<uint16_t> dspark_aux_cat(dspark_cal_toks * 3u * kHiddenSize, 0u); // 768 MiB
        std::vector<uint16_t> dspark_main_x(dspark_cal_toks * kHiddenSize, 0u);       // 256 MiB
        bool dspark_seeded = false;

        const int total_moe_last = mcfg.has_bundled_dspark()
            ? (mcfg.num_hidden_layers + mcfg.num_dspark_layers() - 1) // 43 + 3 - 1 = 45 (L43, L44, L45)
            : mcfg.num_hidden_layers;                                 // 43 (legacy ECHO)

        // 4. Sequential Layer-by-Layer Loop: L01 .. L42 (Backbone) + L43..L45 (3L MoE DSpark) or L43 (ECHO)!
        for (int l = std::max(1, layer_begin); l <= std::min(layer_end, total_moe_last); ++l) {
            const auto t_layer0 = std::chrono::steady_clock::now();
            const bool is_dspark = mcfg.has_bundled_dspark() && (l >= mcfg.num_hidden_layers);
            const bool is_echo   = (!mcfg.has_bundled_dspark()) && (l == mcfg.num_hidden_layers);
            const bool is_draft  = is_dspark || is_echo;
            const int  dspark_i  = is_dspark ? (l - mcfg.num_hidden_layers) : -1;
            const size_t active_cal_seqs = is_draft ? std::min<size_t>(size_t(n_cal_seqs), 32u) : size_t(n_cal_seqs);
            const size_t active_cal_toks = active_cal_seqs * size_t(cal_seq_len);

            // On entry to DSpark 0 (l == 43): project main_x = main_norm(main_proj(L40||L41||L42)) and broadcast embed tokens into cal_res!
            if (is_dspark && !dspark_seeded) {
                dspark_seeded = true;
                auto& g0 = gpus[0];
                hip_chk(hipSetDevice(0), "dspark seed dev0");
                hy4::SafetensorsMmap st_ds(out_dir + "/packed_spine/spine_dspark.safetensors");
                std::vector<void*> ds_allocs;
                const auto& emp = st_ds.at("dspark.main_proj");
                auto* d_mp = dev_upload<uint16_t>(st_ds.tensor_bytes("dspark.main_proj"), size_t(emp.byte_size()) / 2u, ds_allocs);
                auto* d_mn = dev_upload<uint16_t>(st_ds.tensor_bytes("dspark.main_norm"), kHiddenSize, ds_allocs);
                st_ds.release_pages();
                auto* d_cat = dev_alloc<uint16_t>(512u * 3u * kHiddenSize, ds_allocs, false);
                for (size_t off = 0; off < dspark_cal_toks; off += 512u) {
                    const int m = int(std::min<size_t>(512u, dspark_cal_toks - off));
                    hip_chk(hipMemcpyAsync(d_cat, dspark_aux_cat.data() + off * 3u * kHiddenSize,
                                           size_t(m) * 3u * kHiddenSize * 2u, hipMemcpyHostToDevice, g0.stream), "H2D auxcat");
                    g0.bf16_gemm(d_cat, d_mp, g0.d_x_in, m, kHiddenSize, 3 * kHiddenSize);
                    launch_rmsnorm_bf16(g0.d_x_in, d_mn, g0.d_sub_out, m, kHiddenSize, mcfg.rms_norm_eps, g0.stream);
                    hip_chk(hipMemcpyAsync(dspark_main_x.data() + off * kHiddenSize, g0.d_sub_out,
                                           size_t(m) * kHiddenSize * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H mainx");
                }
                hip_chk(hipStreamSynchronize(g0.stream), "sync mainx");
                for (void* p : ds_allocs) (void)hipFree(p);
                // Broadcast query token embeddings (cal_x_emb_shift) to 4 mHC streams of cal_res!
                #pragma omp parallel for schedule(static)
                for (size_t t = 0; t < dspark_cal_toks; ++t) {
                    const uint16_t* src = cal_x_emb_shift.data() + t * kHiddenSize;
                    uint16_t* dst = cal_res.data() + t * kHcDim;
                    for (int s = 0; s < kHcMult; ++s) {
                        std::memcpy(dst + size_t(s) * kHiddenSize, src, kHiddenSize * 2u);
                    }
                }
                std::printf("[ERNIE55-GPU-BUILD] Seeded 3L MoE DSpark calibration: main_x (%zu toks) + 4-stream mHC query embeddings!\n",
                            dspark_cal_toks);
                std::fflush(stdout);
            }

            char spine_fn[512];
            if (is_dspark)    std::snprintf(spine_fn, sizeof(spine_fn), "%s/packed_spine/spine_dspark.safetensors", out_dir.c_str());
            else if (is_echo) std::snprintf(spine_fn, sizeof(spine_fn), "%s/packed_spine/spine_echo_%02d.safetensors", out_dir.c_str(), l);
            else              std::snprintf(spine_fn, sizeof(spine_fn), "%s/packed_spine/spine_layer_%02d.safetensors", out_dir.c_str(), l);
            hy4::SafetensorsMmap st_l(spine_fn);
            std::vector<LayerSpineDev> l_devs(static_cast<size_t>(num_gpus));
            for (int d = 0; d < num_gpus; ++d) {
                if (is_dspark) upload_dspark_spine_to_gpu(st_l, dspark_i, gpus[size_t(d)], l_devs[size_t(d)]);
                else           upload_spine_to_gpu(st_l, mcfg, l, gpus[size_t(d)], l_devs[size_t(d)]);
            }
            st_l.release_pages();

            const auto signs_in    = make_ernie55_signs_in(l);
            const auto signs_inter = make_ernie55_signs_inter(l);
            for (int d = 0; d < num_gpus; ++d) {
                hip_chk(hipSetDevice(d), "signs dev");
                hip_chk(hipMemcpy(gpus[size_t(d)].d_signs_in,    signs_in.data(),    signs_in.size() * 8u, hipMemcpyHostToDevice), "H2D sin");
                hip_chk(hipMemcpy(gpus[size_t(d)].d_signs_inter, signs_inter.data(), signs_inter.size() * 8u, hipMemcpyHostToDevice), "H2D sinter");
            }

            // 4a. Run Layer l Attention + mHC_pre_mlp + Router + fc1_lat + shared_experts across ALL 5 MI50s!
            std::vector<std::array<uint64_t, kNumRoutedExperts>> dev_hist(static_cast<size_t>(num_gpus));
            for (auto& h : dev_hist) h.fill(0ULL);
            std::vector<std::vector<double>> dev_H_in(static_cast<size_t>(num_gpus), std::vector<double>(size_t(kLatentDim) * kLatentDim, 0.0));

            #pragma omp parallel for num_threads(num_gpus) schedule(static)
            for (int d = 0; d < num_gpus; ++d) {
                auto& g = gpus[size_t(d)];
                auto& LD = l_devs[size_t(d)];
                hip_chk(hipSetDevice(g.dev), "attn phase dev");
                hip_chk(hipMemsetAsync(g.d_gram_f64, 0, size_t(kLatentDim) * kLatentDim * sizeof(double), g.stream), "zero H_in");

                std::vector<int32_t> pos_v(512);
                std::vector<int32_t> h_ids(512 * kTopKExperts);
                std::vector<float>   h_wts(512 * kTopKExperts);
                std::vector<float>   h_zrot(512 * kLatentDim);
                std::vector<double>  h_xf64(512 * kLatentDim);

                for (size_t s = size_t(d); s < active_cal_seqs; s += size_t(num_gpus)) {
                    if (!LD.is_dsa) {
                        hip_chk(hipMemsetAsync(g.d_hca_swa_ring, 0, 256u * 512u * 2u, g.stream), "z swa");
                        hip_chk(hipMemsetAsync(g.d_hca_state_kv, 0, 128u * 512u * 4u, g.stream), "z skv");
                        hip_chk(hipMemsetAsync(g.d_hca_state_sc, 0, 128u * 512u * 4u, g.stream), "z ssc");
                    }
                    const int chunk_step = is_dspark ? 128 : 512;
                    for (int off = 0; off < cal_seq_len; off += chunk_step) {
                        const int m = std::min(chunk_step, cal_seq_len - off);
                        for (int i = 0; i < m; ++i) pos_v[size_t(i)] = is_dspark ? i : (off + i);
                        const size_t tok_off = s * size_t(cal_seq_len) + size_t(off);

                        hip_chk(hipMemcpyAsync(g.d_pos_ids, pos_v.data(), size_t(m) * 4u, hipMemcpyHostToDevice, g.stream), "H2D pos");
                        if (is_dspark) {
                            hip_chk(hipMemcpyAsync(g.d_res_a, cal_res.data() + tok_off * kHcDim,
                                                   size_t(m) * kHcDim * 2u, hipMemcpyHostToDevice, g.stream), "H2D ds_res_a");
                            hip_chk(hipMemcpyAsync(g.d_dspark_main_x, dspark_main_x.data() + tok_off * kHiddenSize,
                                                   size_t(m) * kHiddenSize * 2u, hipMemcpyHostToDevice, g.stream), "H2D ds_mainx");
                        } else if (!is_echo) {
                            hip_chk(hipMemcpyAsync(g.d_res_a, cal_res.data() + tok_off * kHcDim,
                                                   size_t(m) * kHcDim * 2u, hipMemcpyHostToDevice, g.stream), "H2D res_a");
                        } else {
                            // ECHO input fusion: h_proj(hnorm(res_L42)) + e_proj(enorm(emb_next))
                            hip_chk(hipMemcpyAsync(g.d_res_b, cal_res.data() + tok_off * kHcDim,
                                                   size_t(m) * kHcDim * 2u, hipMemcpyHostToDevice, g.stream), "H2D echo res");
                            hip_chk(hipMemcpyAsync(g.d_x_in, cal_x_emb_shift.data() + tok_off * kHiddenSize,
                                                   size_t(m) * kHiddenSize * 2u, hipMemcpyHostToDevice, g.stream), "H2D echo emb");
                            launch_rmsnorm_bf16(g.d_res_b, LD.e_hnorm, g.d_tmp_a, m * kHcMult, kHiddenSize, mcfg.rms_norm_eps, g.stream);
                            g.bf16_gemm(g.d_tmp_a, LD.e_hproj, g.d_tmp_b, m * kHcMult, kHiddenSize, kHiddenSize);
                            launch_rmsnorm_bf16(g.d_x_in, LD.e_enorm, g.d_sub_out, m, kHiddenSize, mcfg.rms_norm_eps, g.stream);
                            g.bf16_gemm(g.d_sub_out, LD.e_eproj, g.d_x_in, m, kHiddenSize, kHiddenSize);
                            launch_echo_fuse_he(g.d_tmp_b, g.d_x_in, g.d_res_a, m, kHiddenSize, g.stream);
                        }

                        launch_mhc_pre_norm(g.d_res_a, LD.d_am_fn, LD.d_am_sc, LD.d_am_bs, LD.d_in_norm,
                                            g.d_post_mix, g.d_comb_mix, g.d_x_in, m, kHiddenSize,
                                            mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
                        run_attn_chunk_gpu(g, LD, mcfg, m, is_dspark ? 0 : off);
                        launch_mhc_post_update(g.d_sub_out, g.d_res_a, g.d_post_mix, g.d_comb_mix, g.d_res_b, m, kHiddenSize, g.stream);

                        // Save post-attention residual into cal_res (for backbone AND DSpark 0..1!)
                        if (!is_echo) {
                            hip_chk(hipMemcpyAsync(cal_res.data() + tok_off * kHcDim, g.d_res_b,
                                                   size_t(m) * kHcDim * 2u, hipMemcpyDeviceToHost, g.stream), "D2H res_b");
                        }

                        // MLP pre-norm + router + fc1_lat + shared_experts
                        launch_mhc_pre_norm(g.d_res_b, LD.d_mm_fn, LD.d_mm_sc, LD.d_mm_bs, LD.d_post_norm,
                                            g.d_post_mix, g.d_comb_mix, g.d_x_in, m, kHiddenSize,
                                            mcfg.rms_norm_eps, mcfg.hc_eps, g.stream);
                        if (!is_echo) {
                            hip_chk(hipMemcpyAsync(cal_mlp_post_mix.data() + tok_off * kHcMult, g.d_post_mix,
                                                   size_t(m) * kHcMult * 4u, hipMemcpyDeviceToHost, g.stream), "D2H pm");
                            hip_chk(hipMemcpyAsync(cal_mlp_comb_mix.data() + tok_off * kHcMult2, g.d_comb_mix,
                                                   size_t(m) * kHcMult2 * 4u, hipMemcpyDeviceToHost, g.stream), "D2H cm");
                        }

                        launch_bf16_f32w_gemv_f32(g.d_x_in, LD.m_router_w01, g.d_f32_tmp1, m, 1024, kHiddenSize, g.stream);
                        launch_split_sigmoid_top10_route(g.d_f32_tmp1, LD.m_corr_bias, LD.m_scale_param,
                                                         g.d_topk_ids, g.d_topk_wts, m, g.stream);
                        g.bf16_gemm(g.d_x_in, LD.m_fc1, g.d_h_lat, m, kLatentDim, kHiddenSize);
                        launch_bf16_to_f32(g.d_h_lat, g.d_f32_tmp0, m * kLatentDim, g.stream);
                        launch_wht128_f32_to_f32(g.d_f32_tmp0, g.d_signs_in, g.d_z_f32, m, kLatentDim, g.stream);

                        // Shared experts (for backbone AND DSpark 0..1!)
                        if (!is_echo) {
                            g.bf16_gemm(g.d_x_in, LD.m_sh_w1, g.d_tmp_a, m, kMoeInterSize, kHiddenSize);
                            g.bf16_gemm(g.d_x_in, LD.m_sh_w3, g.d_tmp_b, m, kMoeInterSize, kHiddenSize);
                            launch_situ_glu_bf16(g.d_tmp_a, g.d_tmp_b, g.d_tmp_a, m, kMoeInterSize, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream);
                            g.bf16_gemm(g.d_tmp_a, LD.m_sh_w2, g.d_sub_out, m, kHiddenSize, kMoeInterSize);
                            hip_chk(hipMemcpyAsync(cal_shared_bf16.data() + tok_off * kHiddenSize, g.d_sub_out,
                                                   size_t(m) * kHiddenSize * 2u, hipMemcpyDeviceToHost, g.stream), "D2H sh");
                        }
                        hip_chk(hipMemcpyAsync(cal_lat_bf16.data() + tok_off * kLatentDim, g.d_h_lat,
                                               size_t(m) * kLatentDim * 2u, hipMemcpyDeviceToHost, g.stream), "D2H lat");
                        hip_chk(hipMemcpyAsync(h_ids.data(), g.d_topk_ids, size_t(m) * kTopKExperts * 4u, hipMemcpyDeviceToHost, g.stream), "D2H ids");
                        hip_chk(hipMemcpyAsync(h_wts.data(), g.d_topk_wts, size_t(m) * kTopKExperts * 4u, hipMemcpyDeviceToHost, g.stream), "D2H wts");
                        hip_chk(hipMemcpyAsync(h_zrot.data(), g.d_z_f32, size_t(m) * kLatentDim * 4u, hipMemcpyDeviceToHost, g.stream), "D2H zrot");
                        hip_chk(hipStreamSynchronize(g.stream), "sync chunk");

                        // Record router histogram + top-k arrays
                        for (int r = 0; r < m; ++r) {
                            for (int k = 0; k < kTopKExperts; ++k) {
                                const int eid = h_ids[size_t(r) * kTopKExperts + size_t(k)];
                                if (eid >= 0 && eid < kNumRoutedExperts) ++dev_hist[size_t(d)][size_t(eid)];
                                cal_topk_ids[(tok_off + size_t(r)) * kTopKExperts + size_t(k)] = int16_t(eid);
                                cal_topk_wts[(tok_off + size_t(r)) * kTopKExperts + size_t(k)] = h_wts[size_t(r) * kTopKExperts + size_t(k)];
                            }
                        }

                        // Exact TARQ RAREBAL FP64 Gram accumulation H_in += X_rot^T diag(w_rarebal) X_rot on MI50 FP64!
                        std::array<double, kLatentDim> mu{};
                        std::vector<double> norms(static_cast<size_t>(m), 0.0);
                        for (int r = 0; r < m; ++r) {
                            const float* xr = h_zrot.data() + size_t(r) * kLatentDim;
                            double sq = 0.0;
                            for (int k = 0; k < kLatentDim; ++k) sq += double(xr[k]) * double(xr[k]);
                            norms[size_t(r)] = sq;
                            const double invn = 1.0 / std::sqrt(sq + 1e-18);
                            for (int k = 0; k < kLatentDim; ++k) mu[size_t(k)] += double(xr[k]) * invn;
                        }
                        double mu_sq = 0.0;
                        for (int k = 0; k < kLatentDim; ++k) mu_sq += mu[size_t(k)] * mu[size_t(k)];
                        const double inv_mu = 1.0 / std::sqrt(mu_sq + 1e-18);
                        std::vector<std::pair<double, int>> align(static_cast<size_t>(m));
                        for (int r = 0; r < m; ++r) {
                            const float* xr = h_zrot.data() + size_t(r) * kLatentDim;
                            double dp = 0.0;
                            for (int k = 0; k < kLatentDim; ++k) dp += double(xr[k]) * mu[size_t(k)];
                            align[size_t(r)] = {std::fabs(dp) * inv_mu / std::sqrt(norms[size_t(r)] + 1e-18), r};
                        }
                        const int rt = std::max(1, m / 8);
                        std::nth_element(align.begin(), align.begin() + rt, align.end());
                        std::vector<uint8_t> is_rare(static_cast<size_t>(m), 0u);
                        double tr_c = 0.0, tr_r = 0.0;
                        for (int i = 0; i < rt; ++i) is_rare[size_t(align[size_t(i)].second)] = 1u;
                        for (int r = 0; r < m; ++r) { if (is_rare[size_t(r)]) tr_r += norms[size_t(r)]; else tr_c += norms[size_t(r)]; }
                        const double lam_tarq = (tr_r > 1e-12 && tr_c > 0.0) ? std::min(16.0, tr_c / tr_r) : 1.0;
                        const double sqrt_lam = std::sqrt(lam_tarq);
                        for (int r = 0; r < m; ++r) {
                            const double sc = is_rare[size_t(r)] ? sqrt_lam : 1.0;
                            const float* xr = h_zrot.data() + size_t(r) * kLatentDim;
                            double* dst = h_xf64.data() + size_t(r) * kLatentDim;
                            for (int k = 0; k < kLatentDim; ++k) dst[k] = double(xr[k]) * sc;
                        }
                        hip_chk(hipMemcpyAsync(g.d_x_f64, h_xf64.data(), size_t(m) * kLatentDim * sizeof(double), hipMemcpyHostToDevice, g.stream), "H2D xf64");
                        const double alpha_g = 1.0 / double(active_cal_toks), beta_g = 1.0;
                        blas_chk(rocblas_dgemm(g.blas, rocblas_operation_none, rocblas_operation_transpose,
                                               kLatentDim, kLatentDim, m, &alpha_g,
                                               g.d_x_f64, kLatentDim, g.d_x_f64, kLatentDim,
                                               &beta_g, g.d_gram_f64, kLatentDim), "dgemm H_in");
                    }
                }
                hip_chk(hipMemcpyAsync(dev_H_in[size_t(d)].data(), g.d_gram_f64,
                                       size_t(kLatentDim) * kLatentDim * sizeof(double), hipMemcpyDeviceToHost, g.stream), "D2H H_in");
                hip_chk(hipStreamSynchronize(g.stream), "sync H_in");
            }

            // Combine empirical router histogram & FP64 H_in across all 5 MI50s!
            auto& hist_l = router_hist[size_t(l)];
            hist_l.fill(0ULL);
            uint64_t total_routes = 0ULL;
            for (int d = 0; d < num_gpus; ++d) {
                for (int e = 0; e < kNumRoutedExperts; ++e) {
                    hist_l[size_t(e)] += dev_hist[size_t(d)][size_t(e)];
                    total_routes += dev_hist[size_t(d)][size_t(e)];
                }
            }
            std::vector<double> H_in(size_t(kLatentDim) * kLatentDim, 0.0);
            for (int d = 0; d < num_gpus; ++d) {
                for (size_t i = 0; i < H_in.size(); ++i) H_in[i] += dev_H_in[size_t(d)][i];
            }

            // Sort 512 experts by descending empirical hit frequency -> slot_to_raw[l]!
            std::array<uint16_t, kNumRoutedExperts> raw_to_slot{};
            {
                std::vector<int> ord(kNumRoutedExperts);
                std::iota(ord.begin(), ord.end(), 0);
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
                    return hist_l[size_t(a)] > hist_l[size_t(b)];
                });
                for (int slot = 0; slot < kNumRoutedExperts; ++slot) {
                    slot_to_raw[size_t(l)][size_t(slot)] = uint16_t(ord[size_t(slot)]);
                    raw_to_slot[size_t(ord[size_t(slot)])] = uint16_t(slot);
                }
            }
            uint64_t mi50_hits = 0, mi50_l1_hits = 0;
            for (int slot = 0; slot < 318; ++slot) mi50_hits += hist_l[slot_to_raw[size_t(l)][size_t(slot)]];
            for (int slot = 0; slot < 386; ++slot) mi50_l1_hits += hist_l[slot_to_raw[size_t(l)][size_t(slot)]];
            const double mi50_hit_pct    = 100.0 * double(mi50_hits)    / double(std::max<uint64_t>(1ULL, total_routes));
            const double mi50_l1_hit_pct = 100.0 * double(mi50_l1_hits) / double(std::max<uint64_t>(1ULL, total_routes));

            // Solve exact FP64 reverse-Cholesky R_up_in!
            double cond_in = 1.0, cond_inter = 1.0;
            auto R_up_in = build_fp64_ldlq_rup_f32(H_in.data(), kLatentDim, ldlq_damp, &cond_in);
            for (int d = 0; d < num_gpus; ++d) {
                hip_chk(hipSetDevice(d), "rup_in dev");
                hip_chk(hipMemcpy(gpus[size_t(d)].d_rup_in, R_up_in.data(), R_up_in.size() * 4u, hipMemcpyHostToDevice), "H2D rup_in");
            }

            // 4b. Simultaneously run exact BF16 Attention + Pre-MLP + Router on the 64 GATE-2 held-out probes (dev 0)!
            if (!is_draft) {
                auto& g0 = gpus[0];
                auto& L0 = l_devs[0];
                hip_chk(hipSetDevice(g0.dev), "ref g2 dev");
                std::vector<int32_t> pos_v(static_cast<size_t>(probe_len));
                for (int i = 0; i < probe_len; ++i) pos_v[size_t(i)] = i;
                hip_chk(hipMemcpy(g0.d_pos_ids, pos_v.data(), size_t(probe_len) * 4u, hipMemcpyHostToDevice), "H2D ppos");
                std::fill(ref_moe_acc.begin(), ref_moe_acc.end(), 0.0f);

                for (int p = 0; p < n_probes; ++p) {
                    const size_t toff = size_t(p) * size_t(probe_len);
                    if (!L0.is_dsa) {
                        hip_chk(hipMemsetAsync(g0.d_hca_swa_ring, 0, 256u * 512u * 2u, g0.stream), "z rswa");
                        hip_chk(hipMemsetAsync(g0.d_hca_state_kv, 0, 128u * 512u * 4u, g0.stream), "z rskv");
                        hip_chk(hipMemsetAsync(g0.d_hca_state_sc, 0, 128u * 512u * 4u, g0.stream), "z rssc");
                    }
                    hip_chk(hipMemcpyAsync(g0.d_res_a, ref_res.data() + toff * kHcDim,
                                           size_t(probe_len) * kHcDim * 2u, hipMemcpyHostToDevice, g0.stream), "H2D rres");
                    launch_mhc_pre_norm(g0.d_res_a, L0.d_am_fn, L0.d_am_sc, L0.d_am_bs, L0.d_in_norm,
                                        g0.d_post_mix, g0.d_comb_mix, g0.d_x_in, probe_len, kHiddenSize,
                                        mcfg.rms_norm_eps, mcfg.hc_eps, g0.stream);
                    run_attn_chunk_gpu(g0, L0, mcfg, probe_len, 0);
                    launch_mhc_post_update(g0.d_sub_out, g0.d_res_a, g0.d_post_mix, g0.d_comb_mix, g0.d_res_b,
                                           probe_len, kHiddenSize, g0.stream);
                    hip_chk(hipMemcpyAsync(ref_res.data() + toff * kHcDim, g0.d_res_b,
                                           size_t(probe_len) * kHcDim * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H rres");
                    launch_mhc_pre_norm(g0.d_res_b, L0.d_mm_fn, L0.d_mm_sc, L0.d_mm_bs, L0.d_post_norm,
                                        g0.d_post_mix, g0.d_comb_mix, g0.d_x_in, probe_len, kHiddenSize,
                                        mcfg.rms_norm_eps, mcfg.hc_eps, g0.stream);
                    hip_chk(hipMemcpyAsync(ref_mlp_post_mix.data() + toff * kHcMult, g0.d_post_mix,
                                           size_t(probe_len) * kHcMult * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H rpm");
                    hip_chk(hipMemcpyAsync(ref_mlp_comb_mix.data() + toff * kHcMult2, g0.d_comb_mix,
                                           size_t(probe_len) * kHcMult2 * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H rcm");

                    launch_bf16_f32w_gemv_f32(g0.d_x_in, L0.m_router_w01, g0.d_f32_tmp1, probe_len, 1024, kHiddenSize, g0.stream);
                    launch_split_sigmoid_top10_route(g0.d_f32_tmp1, L0.m_corr_bias, L0.m_scale_param,
                                                     g0.d_topk_ids, g0.d_topk_wts, probe_len, g0.stream);
                    g0.bf16_gemm(g0.d_x_in, L0.m_fc1, g0.d_h_lat, probe_len, kLatentDim, kHiddenSize);
                    launch_bf16_to_f32(g0.d_h_lat, g0.d_f32_tmp0, probe_len * kLatentDim, g0.stream);

                    g0.bf16_gemm(g0.d_x_in, L0.m_sh_w1, g0.d_tmp_a, probe_len, kMoeInterSize, kHiddenSize);
                    g0.bf16_gemm(g0.d_x_in, L0.m_sh_w3, g0.d_tmp_b, probe_len, kMoeInterSize, kHiddenSize);
                    launch_situ_glu_bf16(g0.d_tmp_a, g0.d_tmp_b, g0.d_tmp_a, probe_len, kMoeInterSize,
                                         mcfg.situ_beta, mcfg.situ_linear_beta, g0.stream);
                    g0.bf16_gemm(g0.d_tmp_a, L0.m_sh_w2, g0.d_sub_out, probe_len, kHiddenSize, kMoeInterSize);

                    hip_chk(hipMemcpyAsync(ref_lat_f32.data() + toff * kLatentDim, g0.d_f32_tmp0,
                                           size_t(probe_len) * kLatentDim * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H rlat");
                    hip_chk(hipMemcpyAsync(ref_shared_bf16.data() + toff * kHiddenSize, g0.d_sub_out,
                                           size_t(probe_len) * kHiddenSize * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H rsh");
                    hip_chk(hipMemcpyAsync(ref_topk_ids.data() + toff * kTopKExperts, g0.d_topk_ids,
                                           size_t(probe_len) * kTopKExperts * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H rids");
                    hip_chk(hipMemcpyAsync(ref_topk_wts.data() + toff * kTopKExperts, g0.d_topk_wts,
                                           size_t(probe_len) * kTopKExperts * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H rwts");
                }
                hip_chk(hipStreamSynchronize(g0.stream), "sync ref g2");
                for (size_t i = 0; i < n_probe_toks * kTopKExperts; ++i) {
                    all_ref_router_top10[size_t(l) * n_probe_toks * kTopKExperts + i] = int16_t(ref_topk_ids[i]);
                }
            }

            // Build per-expert lookup of GATE-2 held-out tokens routed to raw expert e!
            std::array<std::vector<std::pair<int, float>>, kNumRoutedExperts> ref_by_exp;
            if (!is_draft) {
                for (int t = 0; t < int(n_probe_toks); ++t) {
                    for (int k = 0; k < kTopKExperts; ++k) {
                        const int eid = ref_topk_ids[size_t(t) * kTopKExperts + size_t(k)];
                        const float wt = ref_topk_wts[size_t(t) * kTopKExperts + size_t(k)];
                        if (eid >= 0 && eid < kNumRoutedExperts && wt != 0.0f) {
                            ref_by_exp[size_t(eid)].push_back({t, wt});
                        }
                    }
                }
            }

            // 4c. Stream all 512 BF16 experts in 8 shard-contiguous double-buffered batches of 64 -> 5x MI50 LDLQ Quant + BF16 GATE-2!
            const std::string pfx = is_dspark
                ? ("model.mtp." + std::to_string(dspark_i) + ".block_sparse_moe")
                : ("model.layers." + std::to_string(l) + ".block_sparse_moe");
            std::vector<int> exp_order(kNumRoutedExperts);
            std::iota(exp_order.begin(), exp_order.end(), 0);
            std::sort(exp_order.begin() + 1, exp_order.end(), [&](int a, int b) {
                const auto& sa = idx.by_name.at(pfx + ".experts." + std::to_string(a) + ".w1.weight");
                const auto& sb = idx.by_name.at(pfx + ".experts." + std::to_string(b) + ".w1.weight");
                if (sa.shard_file != sb.shard_file) return sa.shard_file < sb.shard_file;
                return sa.file_offset < sb.file_offset;
            });

            auto fetch_batch = [&](int b0, int b1) {
                std::vector<std::string> req;
                req.reserve(size_t(b1 - b0) * 3u);
                for (int bi = b0; bi < b1; ++bi) {
                    const std::string ep = pfx + ".experts." + std::to_string(exp_order[size_t(bi)]);
                    req.push_back(ep + ".w1.weight");
                    req.push_back(ep + ".w3.weight");
                    req.push_back(ep + ".w2.weight");
                }
                return stream_fetch_tensors(idx, req, http_workers);
            };

            constexpr int kBatchE = 64;
            double stream_s = 0.0;
            auto bundle = fetch_batch(0, kBatchE);
            stream_s += bundle.elapsed_sec;

            // Compute true empirical FP64 RAREBAL H_inter from first 32 experts over 16,384 real routed tokens on GPU 0!
            std::vector<float> R_up_inter;
            {
                auto& g0 = gpus[0];
                hip_chk(hipSetDevice(g0.dev), "H_inter dev");
                hip_chk(hipMemsetAsync(g0.d_gram_f64, 0, size_t(kMoeInterSize) * kMoeInterSize * sizeof(double), g0.stream), "zero H_inter");
                const int n_probe_e = 32;
                const int m_per_e = 512;
                const double norm_div = 1.0 / double(n_probe_e * m_per_e);
                std::vector<uint16_t> h_sub_lat(size_t(m_per_e) * kLatentDim);
                std::vector<float> h_zrot(size_t(m_per_e) * kMoeInterSize);
                std::vector<double> h_zf64(size_t(m_per_e) * kMoeInterSize);
                for (int pi = 0; pi < n_probe_e; ++pi) {
                    const int e = exp_order[size_t(pi)];
                    const std::string ep = pfx + ".experts." + std::to_string(e);
                    const auto w1_b = bundle.to_bf16(ep + ".w1.weight");
                    const auto w3_b = bundle.to_bf16(ep + ".w3.weight");
                    // Pick 512 real calibration tokens strided across entire 512K corpus!
                    const size_t stride = std::max<size_t>(1, active_cal_toks / size_t(m_per_e));
                    for (int mi = 0; mi < m_per_e; ++mi) {
                        const size_t idx_t = (size_t(mi) * stride + size_t(pi) * 131u) % active_cal_toks;
                        std::memcpy(h_sub_lat.data() + size_t(mi) * kLatentDim,
                                    cal_lat_bf16.data() + idx_t * kLatentDim, kLatentDim * 2u);
                    }
                    hip_chk(hipMemcpyAsync(g0.d_h_lat, h_sub_lat.data(), h_sub_lat.size() * 2u, hipMemcpyHostToDevice, g0.stream), "H2D sublat");
                    hip_chk(hipMemcpyAsync(g0.d_w_bf16, w1_b.data(), w1_b.size() * 2u, hipMemcpyHostToDevice, g0.stream), "H2D w1");
                    g0.bf16_gemm(g0.d_h_lat, g0.d_w_bf16, g0.d_tmp_a, m_per_e, kMoeInterSize, kLatentDim);
                    hip_chk(hipMemcpyAsync(g0.d_w_bf16, w3_b.data(), w3_b.size() * 2u, hipMemcpyHostToDevice, g0.stream), "H2D w3");
                    g0.bf16_gemm(g0.d_h_lat, g0.d_w_bf16, g0.d_tmp_b, m_per_e, kMoeInterSize, kLatentDim);
                    launch_situ_glu_bf16(g0.d_tmp_a, g0.d_tmp_b, g0.d_tmp_a, m_per_e, kMoeInterSize,
                                         mcfg.situ_beta, mcfg.situ_linear_beta, g0.stream);
                    launch_bf16_to_f32(g0.d_tmp_a, g0.d_f32_tmp0, m_per_e * kMoeInterSize, g0.stream);
                    launch_wht128_f32_to_f32(g0.d_f32_tmp0, g0.d_signs_inter, g0.d_z_f32, m_per_e, kMoeInterSize, g0.stream);
                    hip_chk(hipMemcpyAsync(h_zrot.data(), g0.d_z_f32, h_zrot.size() * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H iz");
                    hip_chk(hipStreamSynchronize(g0.stream), "sync iz");
                    for (size_t i = 0; i < h_zrot.size(); ++i) h_zf64[i] = double(h_zrot[i]);
                    hip_chk(hipMemcpyAsync(g0.d_x_f64, h_zf64.data(), h_zf64.size() * 8u, hipMemcpyHostToDevice, g0.stream), "H2D iz64");
                    const double beta1 = 1.0;
                    blas_chk(rocblas_dgemm(g0.blas, rocblas_operation_none, rocblas_operation_transpose,
                                           kMoeInterSize, kMoeInterSize, m_per_e, &norm_div,
                                           g0.d_x_f64, kMoeInterSize, g0.d_x_f64, kMoeInterSize,
                                           &beta1, g0.d_gram_f64, kMoeInterSize), "dgemm H_inter");
                }
                std::vector<double> H_inter(size_t(kMoeInterSize) * kMoeInterSize);
                hip_chk(hipMemcpyAsync(H_inter.data(), g0.d_gram_f64, H_inter.size() * 8u, hipMemcpyDeviceToHost, g0.stream), "D2H H_inter");
                hip_chk(hipStreamSynchronize(g0.stream), "sync H_inter");
                R_up_inter = build_fp64_ldlq_rup_f32(H_inter.data(), kMoeInterSize, ldlq_damp, &cond_inter);
                for (int d = 0; d < num_gpus; ++d) {
                    hip_chk(hipSetDevice(d), "rup_inter dev");
                    hip_chk(hipMemcpy(gpus[size_t(d)].d_rup_inter, R_up_inter.data(), R_up_inter.size() * 4u, hipMemcpyHostToDevice), "H2D rup_inter");
                }
            }

            // Save GATE-1 BF16 ref snapshot for raw expert 0 on L01, L07, L43 (stored at slot raw_to_slot[0]!)
            if (l == 1 || l == 7 || l == 43) {
                const std::string ref_dir = out_dir + "/gate1_refs";
                fs::create_directories(ref_dir);
                char rp[512];
                std::snprintf(rp, sizeof(rp), "%s/ref_layer_%02d_exp0.bin", ref_dir.c_str(), l);
                auto w1 = bundle.to_f32(pfx + ".experts.0.w1.weight");
                auto w3 = bundle.to_f32(pfx + ".experts.0.w3.weight");
                auto w2 = bundle.to_f32(pfx + ".experts.0.w2.weight");
                std::vector<float> x8(8u * kLatentDim);
                for (size_t i = 0; i < x8.size(); ++i) x8[i] = bf16_to_f32(cal_lat_bf16[i]);
                std::ofstream rfs(rp, std::ios::binary | std::ios::trunc);
                rfs.write(reinterpret_cast<const char*>(x8.data()), std::streamsize(x8.size() * 4u));
                rfs.write(reinterpret_cast<const char*>(w1.data()), std::streamsize(w1.size() * 4u));
                rfs.write(reinterpret_cast<const char*>(w3.data()), std::streamsize(w3.size() * 4u));
                rfs.write(reinterpret_cast<const char*>(w2.data()), std::streamsize(w2.size() * 4u));
            }

            // Open atomic output HLUT4 shard (written in ROUTER-FREQUENCY-SORTED slot order!)
            char rel_name[64];
            std::snprintf(rel_name, sizeof(rel_name), "hlut4_layer_%02d.bin", l);
            const std::string real_path = store_dir + "/" + rel_name;
            const std::string link_path = link_dir + "/" + rel_name;
            const std::string tmp_path  = real_path + ".tmp";
            const int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (fd < 0) throw std::runtime_error("Cannot create " + tmp_path);

            Ernie55Hlut4LayerHeader hdr{};
            hdr.magic = kErnie55Hlut4ShardMagic;
            hdr.version = kErnie55Hlut4ShardVersion;
            hdr.layer_idx = uint32_t(l);
            hdr.num_experts = uint32_t(kNumRoutedExperts);
            hdr.latent_dim = uint32_t(kLatentDim);
            hdr.moe_inter_dim = uint32_t(kMoeInterSize);
            hdr.tiles_per_proj = uint32_t(kExpProjTiles);
            hdr.ldlq_used = 1u;
            std::memcpy(hdr.signs_in,    signs_in.data(),    sizeof(hdr.signs_in));
            std::memcpy(hdr.signs_inter, signs_inter.data(), sizeof(hdr.signs_inter));
            hdr.hessian_cond_in = cond_in;
            hdr.hessian_cond_inter = cond_inter;
            hdr.rarebal_lambda = 1.0;
            if (::pwrite(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr))) {
                ::close(fd); ::unlink(tmp_path.c_str()); throw std::runtime_error("Header write failed");
            }

            // Keep quantized layer tiles in RAM briefly (3.09 GiB) for Step 4d quantized-prefix advance!
            std::vector<Hlut4Tile32x32> layer_tiles_slot(size_t(kNumRoutedExperts) * 3u * kExpProjTiles);
            std::array<uint32_t, kNumRoutedExperts> exp_crcs{};
            double sqnr_sum = 0.0;
            int rare_fallback_cnt = 0;

            for (int b0 = 0; b0 < kNumRoutedExperts; b0 += kBatchE) {
                const int b1 = std::min(b0 + kBatchE, kNumRoutedExperts);
                const int nb0 = b1, nb1 = std::min(nb0 + kBatchE, kNumRoutedExperts);

                StreamedTensorBundle next_bundle;
                std::exception_ptr ferr;
                std::thread pth;
                if (nb0 < kNumRoutedExperts) {
                    pth = std::thread([&]() {
                        try { next_bundle = fetch_batch(nb0, nb1); } catch (...) { ferr = std::current_exception(); }
                    });
                }

                // Parallelize 64 experts across ALL 5 MI50 GPUs!
                #pragma omp parallel for num_threads(num_gpus) schedule(dynamic, 1) reduction(+:sqnr_sum,rare_fallback_cnt)
                for (int bi = b0; bi < b1; ++bi) {
                    const int tid = omp_get_thread_num();
                    auto& g = gpus[size_t(tid)];
                    hip_chk(hipSetDevice(g.dev), "quant dev");

                    const int e = exp_order[size_t(bi)];
                    const int slot = int(raw_to_slot[size_t(e)]);
                    const bool use_ldlq_e = (hist_l[size_t(e)] >= 64ULL);
                    if (!use_ldlq_e) ++rare_fallback_cnt;

                    const std::string ep = pfx + ".experts." + std::to_string(e);
                    const auto w1_b = bundle.to_bf16(ep + ".w1.weight");
                    const auto w3_b = bundle.to_bf16(ep + ".w3.weight");
                    const auto w2_b = bundle.to_bf16(ep + ".w2.weight");

                    // 1. Exact unquantized BF16 GATE-2 oracle contribution for expert e (chunked <= 512 rows!)
                    if (!is_draft && !ref_by_exp[size_t(e)].empty()) {
                        const auto& hits = ref_by_exp[size_t(e)];
                        const int C_tot = int(hits.size());
                        for (int c0 = 0; c0 < C_tot; c0 += 512) {
                            const int C = std::min(512, C_tot - c0);
                            std::vector<float> x_h(size_t(C) * kLatentDim), y_h(size_t(C) * kLatentDim);
                            for (int c = 0; c < C; ++c) {
                                std::memcpy(x_h.data() + size_t(c) * kLatentDim,
                                            ref_lat_f32.data() + size_t(hits[size_t(c0 + c)].first) * kLatentDim, kLatentDim * 4u);
                            }
                            hip_chk(hipMemcpyAsync(g.d_f32_tmp0, x_h.data(), x_h.size() * 4u, hipMemcpyHostToDevice, g.stream), "H2D rx");
                            hip_chk(hipMemcpyAsync(g.d_w_bf16, w1_b.data(), w1_b.size() * 2u, hipMemcpyHostToDevice, g.stream), "H2D rw1");
                            launch_bf16_to_f32(g.d_w_bf16, g.d_w_f32, kMoeInterSize * kLatentDim, g.stream);
                            g.f32_gemm(g.d_f32_tmp0, g.d_w_f32, g.d_z_f32, C, kMoeInterSize, kLatentDim);

                            hip_chk(hipMemcpyAsync(g.d_w_bf16, w3_b.data(), w3_b.size() * 2u, hipMemcpyHostToDevice, g.stream), "H2D rw3");
                            launch_bf16_to_f32(g.d_w_bf16, g.d_w_f32, kMoeInterSize * kLatentDim, g.stream);
                            g.f32_gemm(g.d_f32_tmp0, g.d_w_f32, g.d_u_f32, C, kMoeInterSize, kLatentDim);

                            std::vector<float> gh(size_t(C) * kMoeInterSize), uh(size_t(C) * kMoeInterSize);
                            hip_chk(hipMemcpyAsync(gh.data(), g.d_z_f32, gh.size() * 4u, hipMemcpyDeviceToHost, g.stream), "D2H rg");
                            hip_chk(hipMemcpyAsync(uh.data(), g.d_u_f32, uh.size() * 4u, hipMemcpyDeviceToHost, g.stream), "D2H ru");
                            hip_chk(hipStreamSynchronize(g.stream), "sync rgu");
                            for (size_t i = 0; i < gh.size(); ++i) gh[i] = ernie55_situ_glu(gh[i], uh[i], mcfg.situ_beta, mcfg.situ_linear_beta);
                            hip_chk(hipMemcpyAsync(g.d_z_f32, gh.data(), gh.size() * 4u, hipMemcpyHostToDevice, g.stream), "H2D ract");

                            hip_chk(hipMemcpyAsync(g.d_w_bf16, w2_b.data(), w2_b.size() * 2u, hipMemcpyHostToDevice, g.stream), "H2D rw2");
                            launch_bf16_to_f32(g.d_w_bf16, g.d_w_f32, kLatentDim * kMoeInterSize, g.stream);
                            g.f32_gemm(g.d_z_f32, g.d_w_f32, g.d_y_acc_f32, C, kLatentDim, kMoeInterSize);
                            hip_chk(hipMemcpyAsync(y_h.data(), g.d_y_acc_f32, y_h.size() * 4u, hipMemcpyDeviceToHost, g.stream), "D2H ry");
                            hip_chk(hipStreamSynchronize(g.stream), "sync ry");

                            #pragma omp critical
                            {
                                for (int c = 0; c < C; ++c) {
                                    const int tok_i = hits[size_t(c0 + c)].first;
                                    const float wt  = hits[size_t(c0 + c)].second;
                                    float* dst = ref_moe_acc.data() + size_t(tok_i) * kLatentDim;
                                    const float* src = y_h.data() + size_t(c) * kLatentDim;
                                    for (int k = 0; k < kLatentDim; ++k) dst[k] += src[k] * wt;
                                }
                            }
                        }
                    }

                    // 2. GPU-accelerated FP64 RAREBAL-LDLQ WHT128-HLUT4 quantization of (w1, w3, w2)!
                    Hlut4Tile32x32* dst_exp = layer_tiles_slot.data() + size_t(slot) * 3u * kExpProjTiles;
                    auto quant_proj = [&](const std::vector<uint16_t>& wb, const uint64_t* d_signs,
                                          const float* d_rup, Hlut4Tile32x32* dst_tiles) -> double {
                        hip_chk(hipMemcpyAsync(g.d_w_bf16, wb.data(), wb.size() * 2u, hipMemcpyHostToDevice, g.stream), "H2D qw");
                        launch_bf16_to_f32(g.d_w_bf16, g.d_w_f32, kMoeInterSize * kLatentDim, g.stream);
                        launch_wht128_f32_to_f32(g.d_w_f32, d_signs, g.d_w_rot, kMoeInterSize, kLatentDim, g.stream);
                        hip_chk(hipMemsetAsync(g.d_sq_sig_err2, 0, 8u, g.stream), "z sq");
                        launch_wht128_hlut4_ldlq_quant_gpu(
                            g.d_w_rot, use_ldlq_e ? d_rup : nullptr, g.d_levels8_f32, g.d_thresh7_f32,
                            g.d_tiles_out, g.d_sq_sig_err2, kMoeInterSize, kLatentDim, g.stream);
                        float se[2]{};
                        hip_chk(hipMemcpyAsync(dst_tiles, g.d_tiles_out, kHlut4ProjBytes, hipMemcpyDeviceToHost, g.stream), "D2H tiles");
                        hip_chk(hipMemcpyAsync(se, g.d_sq_sig_err2, 8u, hipMemcpyDeviceToHost, g.stream), "D2H sq");
                        hip_chk(hipStreamSynchronize(g.stream), "sync q");
                        return 10.0 * std::log10((double(se[0]) + 1e-30) / (double(se[1]) + 1e-30));
                    };

                    const double s1 = quant_proj(w1_b, g.d_signs_in,    g.d_rup_in,    dst_exp + 0u * kExpProjTiles);
                    const double s3 = quant_proj(w3_b, g.d_signs_in,    g.d_rup_in,    dst_exp + 1u * kExpProjTiles);
                    const double s2 = quant_proj(w2_b, g.d_signs_inter, g.d_rup_inter, dst_exp + 2u * kExpProjTiles);
                    sqnr_sum += (s1 + s3 + s2) / 3.0;
                    exp_crcs[size_t(slot)] = crc32c_update(0u, dst_exp, kHlut4ExpertBytes);

                    const off_t off = off_t(sizeof(hdr)) + off_t(slot) * off_t(kHlut4ExpertBytes);
                    size_t wr_done = 0;
                    while (wr_done < kHlut4ExpertBytes) {
                        const ssize_t w = ::pwrite(fd, reinterpret_cast<const uint8_t*>(dst_exp) + wr_done,
                                                   kHlut4ExpertBytes - wr_done, off + off_t(wr_done));
                        if (w <= 0) throw std::runtime_error("Short pwrite " + tmp_path);
                        wr_done += size_t(w);
                    }
                }

                bundle = {};
                (void)::fdatasync(fd);
                (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                if (pth.joinable()) {
                    pth.join();
                    if (ferr) { ::close(fd); ::unlink(tmp_path.c_str()); std::rethrow_exception(ferr); }
                    stream_s += next_bundle.elapsed_sec;
                    bundle = std::move(next_bundle);
                }
            }

            hdr.avg_sqnr_db    = sqnr_sum / double(kNumRoutedExperts);
            hdr.payload_crc32c = crc32c_update(0u, exp_crcs.data(), sizeof(exp_crcs));
            hdr.header_crc32c  = crc32c_update(0u, &hdr, offsetof(Ernie55Hlut4LayerHeader, header_crc32c));
            if (::pwrite(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr))) {
                ::close(fd); ::unlink(tmp_path.c_str()); throw std::runtime_error("Header finalize failed");
            }
            (void)::fdatasync(fd);
            (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
            if (::rename(tmp_path.c_str(), real_path.c_str()) != 0) throw std::runtime_error("Rename failed " + real_path);
            if (real_path != link_path) {
                if (fs::exists(link_path) || fs::is_symlink(link_path)) fs::remove(link_path);
                fs::create_symlink(real_path, link_path);
            }
            const std::string sha = file_sha256(real_path);

            // 4d. Advance BOTH X_cal (quantized-prefix WHT128-HLUT4 MoE!) AND X_ref_g2 (exact BF16 MoE!) for Layer l+1!
            if (!is_draft) {
                // First finalize X_ref_g2 on GPU 0 using exact BF16 ref_moe_acc!
                {
                    auto& g0 = gpus[0];
                    auto& L0 = l_devs[0];
                    hip_chk(hipSetDevice(g0.dev), "ref fin dev");
                    for (int off = 0; off < int(n_probe_toks); off += 512) {
                        const int m = std::min(512, int(n_probe_toks) - off);
                        hip_chk(hipMemcpyAsync(g0.d_f32_tmp0, ref_moe_acc.data() + size_t(off) * kLatentDim,
                                               size_t(m) * kLatentDim * 4u, hipMemcpyHostToDevice, g0.stream), "H2D rma");
                        hip_chk(hipMemcpyAsync(g0.d_sub_out, ref_shared_bf16.data() + size_t(off) * kHiddenSize,
                                               size_t(m) * kHiddenSize * 2u, hipMemcpyHostToDevice, g0.stream), "H2D rsh");
                        hip_chk(hipMemcpyAsync(g0.d_res_b, ref_res.data() + size_t(off) * kHcDim,
                                               size_t(m) * kHcDim * 2u, hipMemcpyHostToDevice, g0.stream), "H2D rrb");
                        hip_chk(hipMemcpyAsync(g0.d_post_mix, ref_mlp_post_mix.data() + size_t(off) * kHcMult,
                                               size_t(m) * kHcMult * 4u, hipMemcpyHostToDevice, g0.stream), "H2D rpm");
                        hip_chk(hipMemcpyAsync(g0.d_comb_mix, ref_mlp_comb_mix.data() + size_t(off) * kHcMult2,
                                               size_t(m) * kHcMult2 * 4u, hipMemcpyHostToDevice, g0.stream), "H2D rcm");
                        launch_rmsnorm_f32_to_bf16(g0.d_f32_tmp0, L0.m_lat_norm, g0.d_h_lat, m, kLatentDim, mcfg.rms_norm_eps, g0.stream);
                        g0.bf16_gemm(g0.d_h_lat, L0.m_fc2, g0.d_tmp_d, m, kHiddenSize, kLatentDim);
                        launch_add_bf16_inplace(g0.d_sub_out, g0.d_tmp_d, m * kHiddenSize, g0.stream);
                        launch_mhc_post_update(g0.d_sub_out, g0.d_res_b, g0.d_post_mix, g0.d_comb_mix, g0.d_res_a, m, kHiddenSize, g0.stream);
                        hip_chk(hipMemcpyAsync(ref_res.data() + size_t(off) * kHcDim, g0.d_res_a,
                                               size_t(m) * kHcDim * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H rra");
                    }
                    hip_chk(hipStreamSynchronize(g0.stream), "sync ref fin");
                }
            }

            // Now advance X_cal through the JUST-QUANTIZED WHT128-HLUT4 MoE (split 512 experts across 5 MI50s!)
            // Runs for L01..L42 (backbone -> captures aux L40, L41, L42!) AND L43..L44 (DSpark 0 -> 1 -> 2!)
            if (!is_echo && l < std::min(layer_end, total_moe_last)) {
                    // Upload each GPU's ~103 quantized experts temporarily (only 635 MiB/GPU!)
                    struct TempExpBank {
                        int s0 = 0, s1 = 0;
                        Hlut4Tile32x32 *w1 = nullptr, *w3 = nullptr, *w2 = nullptr;
                    };
                    std::vector<TempExpBank> banks(static_cast<size_t>(num_gpus));
                    for (int d = 0; d < num_gpus; ++d) {
                        const int s0 = (kNumRoutedExperts * d) / num_gpus;
                        const int s1 = (kNumRoutedExperts * (d + 1)) / num_gpus;
                        const int cnt = s1 - s0;
                        banks[size_t(d)].s0 = s0; banks[size_t(d)].s1 = s1;
                        hip_chk(hipSetDevice(d), "bank dev");
                        std::vector<Hlut4Tile32x32> tw1(size_t(cnt) * kExpProjTiles), tw3(size_t(cnt) * kExpProjTiles), tw2(size_t(cnt) * kExpProjTiles);
                        for (int i = 0; i < cnt; ++i) {
                            const Hlut4Tile32x32* src = layer_tiles_slot.data() + size_t(s0 + i) * 3u * kExpProjTiles;
                            std::memcpy(tw1.data() + size_t(i) * kExpProjTiles, src + 0u * kExpProjTiles, kHlut4ProjBytes);
                            std::memcpy(tw3.data() + size_t(i) * kExpProjTiles, src + 1u * kExpProjTiles, kHlut4ProjBytes);
                            std::memcpy(tw2.data() + size_t(i) * kExpProjTiles, src + 2u * kExpProjTiles, kHlut4ProjBytes);
                        }
                        hip_chk(hipMalloc(&banks[size_t(d)].w1, tw1.size() * sizeof(Hlut4Tile32x32)), "m tw1");
                        hip_chk(hipMalloc(&banks[size_t(d)].w3, tw3.size() * sizeof(Hlut4Tile32x32)), "m tw3");
                        hip_chk(hipMalloc(&banks[size_t(d)].w2, tw2.size() * sizeof(Hlut4Tile32x32)), "m tw2");
                        hip_chk(hipMemcpy(banks[size_t(d)].w1, tw1.data(), tw1.size() * sizeof(Hlut4Tile32x32), hipMemcpyHostToDevice), "up tw1");
                        hip_chk(hipMemcpy(banks[size_t(d)].w3, tw3.data(), tw3.size() * sizeof(Hlut4Tile32x32), hipMemcpyHostToDevice), "up tw3");
                        hip_chk(hipMemcpy(banks[size_t(d)].w2, tw2.data(), tw2.size() * sizeof(Hlut4Tile32x32), hipMemcpyHostToDevice), "up tw2");
                    }

                    // Process X_cal in chunks of 512 tokens across all 5 MI50s concurrently!
                    std::vector<float> y_sum(512u * kLatentDim);
                    std::vector<std::vector<float>> y_parts(static_cast<size_t>(num_gpus), std::vector<float>(512u * kLatentDim));
                    std::vector<uint16_t> h_xrot_f16(512u * kLatentDim);
                    for (size_t off = 0; off < active_cal_toks; off += 512u) {
                        const int m = int(std::min<size_t>(512u, active_cal_toks - off));
                        auto& g0 = gpus[0];
                        hip_chk(hipSetDevice(0), "rot dev0");
                        hip_chk(hipMemcpyAsync(g0.d_h_lat, cal_lat_bf16.data() + off * kLatentDim,
                                               size_t(m) * kLatentDim * 2u, hipMemcpyHostToDevice, g0.stream), "H2D clat");
                        launch_wht128_bf16_to_f16(g0.d_h_lat, g0.d_signs_in, g0.d_x_rot_f16, m, kLatentDim, g0.stream);
                        hip_chk(hipMemcpyAsync(h_xrot_f16.data(), g0.d_x_rot_f16, size_t(m) * kLatentDim * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H cxrot");
                        hip_chk(hipStreamSynchronize(g0.stream), "sync cxrot");

                        #pragma omp parallel for num_threads(num_gpus) schedule(static)
                        for (int d = 0; d < num_gpus; ++d) {
                            auto& g = gpus[size_t(d)];
                            const auto& bk = banks[size_t(d)];
                            hip_chk(hipSetDevice(d), "adv dev");
                            if (d != 0) {
                                hip_chk(hipMemcpyAsync(g.d_x_rot_f16, h_xrot_f16.data(), size_t(m) * kLatentDim * 2u, hipMemcpyHostToDevice, g.stream), "H2D xr");
                            }
                            hip_chk(hipMemsetAsync(g.d_y_acc_f32, 0, size_t(m) * kLatentDim * 4u, g.stream), "z yacc");

                            // Bucket tokens by local slot [s0..s1)
                            std::vector<std::vector<std::pair<int, float>>> buckets(size_t(bk.s1 - bk.s0));
                            for (int r = 0; r < m; ++r) {
                                for (int k = 0; k < kTopKExperts; ++k) {
                                    const int raw = int(cal_topk_ids[(off + size_t(r)) * kTopKExperts + size_t(k)]);
                                    const float wt = cal_topk_wts[(off + size_t(r)) * kTopKExperts + size_t(k)];
                                    if (raw >= 0 && raw < kNumRoutedExperts && wt != 0.0f) {
                                        const int sl = int(raw_to_slot[size_t(raw)]);
                                        if (sl >= bk.s0 && sl < bk.s1) buckets[size_t(sl - bk.s0)].push_back({r, wt});
                                    }
                                }
                            }
                            std::vector<ExpertDispatchGroup> grps;
                            for (int le = 0; le < bk.s1 - bk.s0; ++le) {
                                const auto& vec = buckets[size_t(le)];
                                for (size_t o = 0; o < vec.size(); o += kMaxGroupTokens) {
                                    ExpertDispatchGroup grp{};
                                    grp.expert = le;
                                    grp.count = int32_t(std::min<size_t>(kMaxGroupTokens, vec.size() - o));
                                    for (int c = 0; c < grp.count; ++c) { grp.rows[c] = vec[o + size_t(c)].first; grp.weights[c] = vec[o + size_t(c)].second; }
                                    grps.push_back(grp);
                                }
                            }
                            for (size_t g0i = 0; g0i < grps.size(); g0i += 256u) {
                                const int ng = int(std::min<size_t>(256u, grps.size() - g0i));
                                hip_chk(hipMemcpyAsync(g.d_groups, grps.data() + g0i, size_t(ng) * sizeof(ExpertDispatchGroup), hipMemcpyHostToDevice, g.stream), "H2D grps");
                                launch_hlut4_fdot2_grouped_gu_situ(g.d_x_rot_f16, g.d_groups, ng, bk.w1, bk.w3, g.d_levels8_f16,
                                                                   g.d_z_f32, kMoeInterSize, kLatentDim, mcfg.situ_beta, mcfg.situ_linear_beta, g.stream);
                                launch_wht128_f32_to_f16(g.d_z_f32, g.d_signs_inter, g.d_z_rot_f16, ng * kMaxGroupTokens, kMoeInterSize, g.stream);
                                launch_hlut4_fdot2_grouped_down_atomic(g.d_z_rot_f16, g.d_groups, ng, bk.w2, g.d_levels8_f16,
                                                                       g.d_y_acc_f32, kLatentDim, kMoeInterSize, g.stream);
                            }
                            hip_chk(hipMemcpyAsync(y_parts[size_t(d)].data(), g.d_y_acc_f32, size_t(m) * kLatentDim * 4u, hipMemcpyDeviceToHost, g.stream), "D2H yp");
                            hip_chk(hipStreamSynchronize(g.stream), "sync yp");
                        }

                        std::fill(y_sum.begin(), y_sum.begin() + ptrdiff_t(size_t(m) * kLatentDim), 0.0f);
                        for (int d = 0; d < num_gpus; ++d) {
                            for (size_t i = 0; i < size_t(m) * kLatentDim; ++i) y_sum[i] += y_parts[size_t(d)][i];
                        }
                        auto& L0 = l_devs[0];
                        hip_chk(hipSetDevice(0), "fin cal dev0");
                        hip_chk(hipMemcpyAsync(g0.d_f32_tmp0, y_sum.data(), size_t(m) * kLatentDim * 4u, hipMemcpyHostToDevice, g0.stream), "H2D ys");
                        hip_chk(hipMemcpyAsync(g0.d_sub_out, cal_shared_bf16.data() + off * kHiddenSize, size_t(m) * kHiddenSize * 2u, hipMemcpyHostToDevice, g0.stream), "H2D csh");
                        hip_chk(hipMemcpyAsync(g0.d_res_b, cal_res.data() + off * kHcDim, size_t(m) * kHcDim * 2u, hipMemcpyHostToDevice, g0.stream), "H2D crb");
                        hip_chk(hipMemcpyAsync(g0.d_post_mix, cal_mlp_post_mix.data() + off * kHcMult, size_t(m) * kHcMult * 4u, hipMemcpyHostToDevice, g0.stream), "H2D cpm");
                        hip_chk(hipMemcpyAsync(g0.d_comb_mix, cal_mlp_comb_mix.data() + off * kHcMult2, size_t(m) * kHcMult2 * 4u, hipMemcpyHostToDevice, g0.stream), "H2D ccm");
                        launch_rmsnorm_f32_to_bf16(g0.d_f32_tmp0, L0.m_lat_norm, g0.d_h_lat, m, kLatentDim, mcfg.rms_norm_eps, g0.stream);
                        g0.bf16_gemm(g0.d_h_lat, L0.m_fc2, g0.d_tmp_d, m, kHiddenSize, kLatentDim);
                        launch_add_bf16_inplace(g0.d_sub_out, g0.d_tmp_d, m * kHiddenSize, g0.stream);
                        launch_mhc_post_update(g0.d_sub_out, g0.d_res_b, g0.d_post_mix, g0.d_comb_mix, g0.d_res_a, m, kHiddenSize, g0.stream);
                        hip_chk(hipMemcpyAsync(cal_res.data() + off * kHcDim, g0.d_res_a, size_t(m) * kHcDim * 2u, hipMemcpyDeviceToHost, g0.stream), "D2H cra");
                        hip_chk(hipStreamSynchronize(g0.stream), "sync cra");
                    }

                    for (int d = 0; d < num_gpus; ++d) {
                        (void)hipSetDevice(d);
                        (void)hipFree(banks[size_t(d)].w1);
                        (void)hipFree(banks[size_t(d)].w3);
                        (void)hipFree(banks[size_t(d)].w2);
                    }
                    // Capture exact mean4(cal_res) after backbone L40, L41, L42 for 3L MoE DSpark main_x!
                    if (mcfg.has_bundled_dspark() && l >= 40 && l <= 42) {
                        const int aux_slot = l - 40;
                        #pragma omp parallel for schedule(static)
                        for (size_t t = 0; t < dspark_cal_toks; ++t) {
                            const uint16_t* src = cal_res.data() + t * kHcDim;
                            uint16_t* dst = dspark_aux_cat.data() + (t * 3u + size_t(aux_slot)) * kHiddenSize;
                            for (int c = 0; c < kHiddenSize; ++c) {
                                const float m4 = 0.25f * (
                                    bf16_to_f32(src[c]) +
                                    bf16_to_f32(src[kHiddenSize + c]) +
                                    bf16_to_f32(src[2 * kHiddenSize + c]) +
                                    bf16_to_f32(src[3 * kHiddenSize + c]));
                                dst[c] = f32_to_bf16_rne(m4);
                            }
                        }
                    }
            }

            for (int d = 0; d < num_gpus; ++d) l_devs[size_t(d)].free_all(d);

            const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_layer0).count();
            receipts.push_back({l, hdr.avg_sqnr_db, cond_in, cond_inter, 1.0, mi50_hit_pct, mi50_l1_hit_pct, rare_fallback_cnt, stream_s, tot_s, sha});
            char tag[32]{};
            if (is_dspark) std::snprintf(tag, sizeof(tag), "(DSPARK%d)", dspark_i);
            else if (is_echo) std::snprintf(tag, sizeof(tag), "(ECHO)");
            std::printf("[ERNIE55-GPU-BUILD] L%02d%s | SQNR=%.2f dB | cond=%.1f/%.1f | MI50(318)=%.1f%% MI50+L1(386)=%.1f%% | rare_fb=%d | stream=%.1fs tot=%.1fs | sha=%s\n",
                        l, tag, hdr.avg_sqnr_db, cond_in, cond_inter,
                        mi50_hit_pct, mi50_l1_hit_pct, rare_fallback_cnt, stream_s, tot_s, sha.substr(0, 12).c_str());
            std::fflush(stdout);
        }

        // 5. After L42: Compute EXACT BF16 GATE-2 Reference Logits [64, 64, 201216] via lossless lm_head + MultiMax!
        if (layer_begin <= 1 && layer_end >= mcfg.num_hidden_layers - 1) {
            auto& g0 = gpus[0];
            hip_chk(hipSetDevice(0), "g2 head dev");
            hy4::SafetensorsMmap st_sh(out_dir + "/packed_spine/spine_shared.safetensors");
            std::vector<void*> head_allocs;
            auto* d_lm   = dev_upload<uint16_t>(st_sh.tensor_bytes("lm_head.weight"), size_t(mcfg.vocab_size) * kHiddenSize, head_allocs);
            auto* d_fn   = dev_upload<uint16_t>(st_sh.tensor_bytes("model.norm.weight"), kHiddenSize, head_allocs);
            auto hcf     = st_to_f32(st_sh, "model.mhc_contract.hc_head_fn");
            auto hcs     = st_to_f32(st_sh, "model.mhc_contract.hc_head_scale");
            auto hcb     = st_to_f32(st_sh, "model.mhc_contract.hc_head_base");
            auto* d_hcf  = dev_upload<float>(hcf.data(), hcf.size(), head_allocs);
            auto* d_hcs  = dev_upload<float>(hcs.data(), hcs.size(), head_allocs);
            auto* d_hcb  = dev_upload<float>(hcb.data(), hcb.size(), head_allocs);
            float *d_mmr = nullptr, *d_mmt = nullptr;
            const bool has_mm = st_sh.contains("model.lm_head.multimax_ranges") && st_sh.contains("model.lm_head.multimax_ts");
            if (has_mm) {
                auto mr = st_to_f32(st_sh, "model.lm_head.multimax_ranges");
                auto mt = st_to_f32(st_sh, "model.lm_head.multimax_ts");
                d_mmr = dev_upload<float>(mr.data(), 4, head_allocs);
                d_mmt = dev_upload<float>(mt.data(), 4, head_allocs);
            }
            st_sh.release_pages();

            auto* d_log = dev_alloc<float>(size_t(probe_len) * size_t(mcfg.vocab_size), head_allocs, false);
            std::vector<float> h_log(size_t(probe_len) * size_t(mcfg.vocab_size));

            const std::string g2_log_path = root_dir + "/artifacts/gate2_preview_ref_bf16_logits.bin";
            const int lfd = ::open(g2_log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            struct RefLogHdr { uint32_t magic, n_probes, probe_len, vocab_size; } rlh{0x47324C35u, uint32_t(n_probes), uint32_t(probe_len), uint32_t(mcfg.vocab_size)};
            if (::write(lfd, &rlh, sizeof(rlh)) != ssize_t(sizeof(rlh))) {
                ::close(lfd);
                throw std::runtime_error("Failed writing g2_ref_bf16_logits header");
            }
            for (int p = 0; p < n_probes; ++p) {
                const size_t toff = size_t(p) * size_t(probe_len);
                hip_chk(hipMemcpyAsync(g0.d_res_a, ref_res.data() + toff * kHcDim,
                                       size_t(probe_len) * kHcDim * 2u, hipMemcpyHostToDevice, g0.stream), "H2D g2res");
                launch_mhc_head_contract(g0.d_res_a, d_hcf, d_hcs, d_hcb, d_fn, g0.d_x_in,
                                         probe_len, kHiddenSize, mcfg.rms_norm_eps, mcfg.hc_eps, g0.stream);
                g0.bf16_gemm_f32(g0.d_x_in, d_lm, d_log, probe_len, mcfg.vocab_size, kHiddenSize);
                if (has_mm) launch_multimax_inplace_f32(d_log, d_mmr, d_mmt, probe_len, mcfg.vocab_size, g0.stream);
                hip_chk(hipMemcpyAsync(h_log.data(), d_log, h_log.size() * 4u, hipMemcpyDeviceToHost, g0.stream), "D2H g2log");
                hip_chk(hipStreamSynchronize(g0.stream), "sync g2log");
                const size_t nb = h_log.size() * 4u;
                size_t wd = 0;
                while (wd < nb) {
                    const ssize_t w = ::write(lfd, reinterpret_cast<const uint8_t*>(h_log.data()) + wd, nb - wd);
                    if (w <= 0) throw std::runtime_error("Short write gate2_preview_ref_bf16_logits.bin");
                    wd += size_t(w);
                }
            }
            (void)::fdatasync(lfd);
            (void)::posix_fadvise(lfd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(lfd);
            for (void* p : head_allocs) (void)hipFree(p);

            const std::string g2_rt_path = root_dir + "/artifacts/gate2_preview_ref_router_top10.bin";
            std::ofstream rto(g2_rt_path, std::ios::binary | std::ios::trunc);
            rto.write(reinterpret_cast<const char*>(all_ref_router_top10.data()),
                      std::streamsize(all_ref_router_top10.size() * sizeof(int16_t)));
            std::printf("[ERNIE55-GPU-BUILD] Saved EXACT BF16 GATE-2 oracle logits (%s, %.2f GiB) + router top-10!\n",
                        g2_log_path.c_str(), double(n_probe_toks) * double(mcfg.vocab_size) * 4.0 / 1073741824.0);
        }

        // 6. Save router_perm.bin (+ router_perm_preview.bin symlink) + router_hist_ernie55.txt + hlut4_preview_pack_manifest.json!
        {
            const std::string pp = out_dir + "/router_perm.bin";
            std::ofstream pfs(pp, std::ios::binary | std::ios::trunc);
            const uint32_t magic = 0x50524D35u, nl = uint32_t(kTotalLayersWithEcho);
            pfs.write(reinterpret_cast<const char*>(&magic), 4);
            pfs.write(reinterpret_cast<const char*>(&nl), 4);
            pfs.write(reinterpret_cast<const char*>(slot_to_raw.data()), std::streamsize(sizeof(slot_to_raw)));
            const std::string pp_alias = out_dir + "/router_perm_preview.bin";
            if (fs::exists(pp_alias) || fs::is_symlink(pp_alias)) fs::remove(pp_alias);
            fs::create_symlink(pp, pp_alias);
        }
        {
            const std::string hp = out_dir + "/router_hist_ernie55.txt";
            std::ofstream hfs(hp, std::ios::trunc);
            for (int l = 1; l <= total_moe_last; ++l) {
                hfs << l;
                for (int e = 0; e < kNumRoutedExperts; ++e) hfs << ' ' << router_hist[size_t(l)][size_t(e)];
                hfs << '\n';
            }
        }
        {
            const std::string mp = root_dir + "/artifacts/hlut4_preview_pack_manifest.json";
            std::ofstream mfs(mp, std::ios::trunc);
            double mean_sqnr = 0.0, mean_mi50 = 0.0, mean_l1 = 0.0;
            for (const auto& r : receipts) { mean_sqnr += r.sqnr_db; mean_mi50 += r.mi50_hit_pct; mean_l1 += r.mi50_l1_hit_pct; }
            if (!receipts.empty()) { mean_sqnr /= receipts.size(); mean_mi50 /= receipts.size(); mean_l1 /= receipts.size(); }
            mfs << "{\n"
                << "  \"schema\": \"ernie55_preview_hlut4_pack_manifest_v1\",\n"
                << "  \"source_repo\": \"" << hf_repo << "\",\n"
                << "  \"cal_tokens\": " << n_cal_toks << ",\n"
                << "  \"ldlq_damp\": " << ldlq_damp << ",\n"
                << "  \"layers_packed\": " << receipts.size() << ",\n"
                << "  \"mean_sqnr_db\": " << std::fixed << std::setprecision(4) << mean_sqnr << ",\n"
                << "  \"mean_mi50_top318_hit_pct\": " << std::setprecision(2) << mean_mi50 << ",\n"
                << "  \"mean_mi50_plus_p150_l1_top386_hit_pct\": " << mean_l1 << ",\n"
                << "  \"total_wall_s\": " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start_all).count() << ",\n"
                << "  \"layers\": [\n";
            for (size_t i = 0; i < receipts.size(); ++i) {
                const auto& r = receipts[i];
                mfs << "    {\"layer\": " << r.layer << ", \"sqnr_db\": " << std::setprecision(4) << r.sqnr_db
                    << ", \"cond_in\": " << std::setprecision(2) << r.cond_in << ", \"cond_inter\": " << r.cond_inter
                    << ", \"mi50_hit_pct\": " << r.mi50_hit_pct << ", \"mi50_l1_hit_pct\": " << r.mi50_l1_hit_pct
                    << ", \"rare_fb\": " << r.rare_fallback_experts << ", \"stream_s\": " << r.stream_s
                    << ", \"total_s\": " << r.total_s << ", \"sha256\": \"" << r.sha256 << "\"}"
                    << (i + 1 < receipts.size() ? "," : "") << "\n";
            }
            mfs << "  ]\n}\n";
        }

        for (int d = 0; d < num_gpus; ++d) gpus[size_t(d)].destroy();
        std::printf("[ERNIE55-GPU-BUILD] === COMPLETE (%zu layers in %.1fs) ===\n",
                    receipts.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start_all).count());
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[ERNIE55-GPU-BUILD] FATAL: %s\n", ex.what());
        return 1;
    }
}
