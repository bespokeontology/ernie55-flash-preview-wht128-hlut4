// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Zero-Disk-Bloat HTTP/2 Range-Streaming Weight Builder & FP64 RAREBAL-LDLQ WHT128-HLUT4 Quantizer
// for baidu/ERNIE-5.5-Flash-Base-BF16 and Nov 25 Apache 2.0 post-trained baidu/ERNIE-5.5-Flash (BF16 or UE8M0 FP8).
// Zero Python anywhere!
//
// Features:
//   - Streams exact byte ranges over HTTP/2 directly from HuggingFace (0 B raw 573 GB shard disk bloat!)
//   - Seamlessly handles BF16, 128x128 UE8M0 FP8, and 32x32 UE8M0 MXFP8 source checkpoints
//   - Packs lossless BF16/FP32 spine (embed, lm_head, mhc_contract, MultiMax, L00..L42, L43 ECHO, and bundled DSpark model.mtp.*)
//   - Propagates real calibration activations layer-by-layer to compute exact FP64 TARQ RAREBAL Hessians (H_in & H_inter)
//     and exact FP64 reverse-Cholesky LDLQ WHT128-HLUT4 quantization across ALL 22,016 (43 x 512) routed experts
//   - Resumable per-layer atomic writes + POSIX_FADV_DONTNEED + optional --experts-dir split storage symlinks!

#include "../src/ernie55_config.hpp"
#include "../src/ernie55_hlut4_pack.hpp"
#include "../src/ernie55_json.hpp"
#include "../src/ernie55_range_stream.hpp"
#include "../src/ernie55_safetensors_writer.hpp"
#include "../src/ernie55_tokenizer.hpp"
#include "../src/safetensors_mmap.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie55 {
namespace {

namespace fs = std::filesystem;

struct BuildOpts {
    std::string hf_repo      = "baidu/ERNIE-5.5-Flash-Base-BF16";
    std::string revision     = "main";
    std::string out_dir      = "./checkpoints/base-bf16";
    std::string experts_dir;  // optional external storage (e.g. ./weights_hlut4/experts_hlut4)
    std::string meta_dir;     // defaults to out_dir
    int http_workers         = 16;
    int cal_tokens           = 256;
    int layer_begin          = 1;
    int layer_end            = 43;  // includes L43 ECHO!
    bool scan_only           = false;
    bool spine_only          = false;
    bool skip_spine          = false;
    bool skip_existing       = true;
    bool use_ldlq            = true;
    bool save_gate1_ref      = true;
};

// Diverse calibration corpus spanning prose, math, code, JSON, reasoning, and CJK
const char* kCalCorpus[] = {
    "The Kolmogorov complexity K(x) of a finite binary string x is the length of the shortest program that outputs x on a universal Turing machine. "
    "In spectral quantization, applying an orthonormal Walsh-Hadamard transform WHT128 spreads heavy-tailed coordinate outliers uniformly into Gaussian N(0,1) coordinates.",
    "def rarebal_cholesky(H, damp=0.01):\n    import math\n    n = len(H)\n    # Reverse Cholesky LDLQ error feedback minimizes tr((W - Q) H (W - Q)^T)\n    return n\n",
    "{\"tool\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3, \"unit\": \"celsius\", \"include_hourly\": true, \"notes\": null}}\n"
    "<ernie:think>Compute 128 * 42 = 5376, then verify 21504 * 6488064 = 139519328256 bytes = 129.94 GiB.</ernie:think>The answer is 129.94 GiB.",
    "Theorem: For any positive definite matrix H = X^T X + lambda I, the reverse-order Cholesky factorization P H P = L L^T yields the exact causal error-feedback operator U.",
    "SELECT layer_id, AVG(sqnr_db) AS mean_sqnr, MAX(cond_est) FROM hlut4_ledger WHERE bits_per_weight = 4.125 GROUP BY layer_id ORDER BY layer_id ASC;",
    "ERNIE 5.5 Flash uses 37 Hybrid Compressed Attention (HCA) layers with 128:1 block compression and 128-token sliding windows plus 6 MLA+DSA layers and Multi-Router Latent MoE with SiTU-GLU."
};

std::vector<float> transpose_if_needed(const std::vector<float>& v, int64_t s0, int64_t s1, int expect_rows, int expect_cols) {
    if (s0 == expect_rows && s1 == expect_cols) return v;
    if (s0 == expect_cols && s1 == expect_rows) return transpose_f32(v.data(), int(s0), int(s1));
    throw std::runtime_error("Unexpected 2D shape [" + std::to_string(s0) + "," + std::to_string(s1) +
                             "], expected [" + std::to_string(expect_rows) + "," + std::to_string(expect_cols) + "]");
}

void add_mhc_to_writer(Ernie55SafetensorsWriter& w, const StreamedTensorBundle& b,
                       const std::string& src_pfx, const std::string& dst_pfx) {
    const std::string fn_k = src_pfx + ".mapping_proj.weight";
    const auto& fn_sp = b.spec(fn_k);
    auto fn = transpose_if_needed(b.to_f32(fn_k), fn_sp.shape.at(0), fn_sp.shape.at(1), kHcMult3, kHcDim);
    w.add_f32(dst_pfx + ".mapping_proj.weight", {kHcMult3, kHcDim}, fn);

    auto a_pre  = b.to_f32(src_pfx + ".alpha_pre");
    auto a_post = b.to_f32(src_pfx + ".alpha_post");
    auto a_res  = b.to_f32(src_pfx + ".alpha_res");
    std::vector<float> sc3 = {a_pre.at(0), a_post.at(0), a_res.at(0)};
    w.add_f32(dst_pfx + ".hc_scale3", {3}, sc3);

    auto bias = b.to_f32(src_pfx + ".bias");
    w.add_f32(dst_pfx + ".bias", {int64_t(bias.size())}, bias);
}

void ensure_metadata_files(const BuildOpts& opt, const std::string& meta_dir) {
    fs::create_directories(meta_dir);
    RemoteRangeIndex tmp;
    tmp.repo = opt.hf_repo;
    tmp.revision = opt.revision;

    const char* required[] = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"};
    for (const char* f : required) {
        const std::string dst = meta_dir + "/" + f;
        if (!fs::is_regular_file(dst) || fs::file_size(dst) == 0) {
            std::printf("[ERNIE55-BUILD] Fetching %s from %s...\n", f, opt.hf_repo.c_str());
            std::fflush(stdout);
            if (!RemoteRangeIndex::curl_download_file(tmp.file_url(f), dst)) {
                throw std::runtime_error(std::string("Failed to fetch ") + f + " from " + opt.hf_repo);
            }
        }
    }
    const char* optional[] = {"generation_config.json", "chat_template.jinja"};
    for (const char* f : optional) {
        const std::string dst = meta_dir + "/" + f;
        if (!fs::is_regular_file(dst)) {
            (void)RemoteRangeIndex::curl_download_file(tmp.file_url(f), dst);
        }
    }
}

void pack_spine_shared(const RemoteRangeIndex& idx, const BuildOpts& opt, const Ernie55ModelConfig& mcfg) {
    const std::string out_p = opt.out_dir + "/packed_spine/spine_shared.safetensors";
    if (opt.skip_existing && fs::is_regular_file(out_p)) {
        std::printf("[ERNIE55-BUILD] Skipping existing %s\n", out_p.c_str());
        return;
    }
    std::vector<std::string> req = {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.mhc_contract.hc_head_fn",
        "model.mhc_contract.hc_head_scale",
        "model.mhc_contract.hc_head_base",
    };
    const bool has_mm = idx.has("model.lm_head.multimax_ranges") && idx.has("model.lm_head.multimax_ts");
    if (has_mm) {
        req.push_back("model.lm_head.multimax_ranges");
        req.push_back("model.lm_head.multimax_ts");
    }

    std::printf("[ERNIE55-BUILD] Streaming spine_shared (%zu tensors)...\n", req.size());
    std::fflush(stdout);
    auto b = stream_fetch_tensors(idx, req, opt.http_workers);

    Ernie55SafetensorsWriter w;
    w.add_bf16("model.embed_tokens.weight", {mcfg.vocab_size, kHiddenSize}, b.to_bf16("model.embed_tokens.weight"));
    w.add_bf16("lm_head.weight",            {mcfg.vocab_size, kHiddenSize}, b.to_bf16("lm_head.weight"));
    w.add_bf16("model.norm.weight",         {kHiddenSize},                  b.to_bf16("model.norm.weight"));

    const auto& hf_sp = b.spec("model.mhc_contract.hc_head_fn");
    auto hf = transpose_if_needed(b.to_f32("model.mhc_contract.hc_head_fn"), hf_sp.shape.at(0), hf_sp.shape.at(1), kHcMult, kHcDim);
    w.add_f32("model.mhc_contract.hc_head_fn",    {kHcMult, kHcDim}, hf);
    w.add_f32("model.mhc_contract.hc_head_scale", {1},               b.to_f32("model.mhc_contract.hc_head_scale"));
    w.add_f32("model.mhc_contract.hc_head_base",  {kHcMult},         b.to_f32("model.mhc_contract.hc_head_base"));
    if (has_mm) {
        w.add_f32("model.lm_head.multimax_ranges", {4}, b.to_f32("model.lm_head.multimax_ranges"));
        w.add_f32("model.lm_head.multimax_ts",     {4}, b.to_f32("model.lm_head.multimax_ts"));
    }
    w.write_atomic(out_p);
    std::printf("[ERNIE55-BUILD] Wrote %s (%.2f MiB wire in %.1fs)\n",
                out_p.c_str(), double(b.total_wire_bytes) / 1048576.0, b.elapsed_sec);
    std::fflush(stdout);
}

void pack_spine_layer(const RemoteRangeIndex& idx, const BuildOpts& opt, const Ernie55ModelConfig& mcfg, int l) {
    const bool is_echo = (l == mcfg.num_hidden_layers);
    char fname[512];
    if (is_echo) std::snprintf(fname, sizeof(fname), "%s/packed_spine/spine_echo_%02d.safetensors", opt.out_dir.c_str(), l);
    else         std::snprintf(fname, sizeof(fname), "%s/packed_spine/spine_layer_%02d.safetensors", opt.out_dir.c_str(), l);
    if (opt.skip_existing && fs::is_regular_file(fname)) return;

    const std::string pfx = "model.layers." + std::to_string(l);
    const std::string mhc_pfx = is_echo ? (pfx + ".transformer_layer") : pfx;
    const bool is_dsa = mcfg.is_dsa_layer(l);
    const bool is_dense = (l < mcfg.first_k_dense_replace);

    std::vector<std::string> req = {
        mhc_pfx + ".self_attention_hyper_connection.mapping_proj.weight",
        mhc_pfx + ".self_attention_hyper_connection.alpha_pre",
        mhc_pfx + ".self_attention_hyper_connection.alpha_post",
        mhc_pfx + ".self_attention_hyper_connection.alpha_res",
        mhc_pfx + ".self_attention_hyper_connection.bias",
        mhc_pfx + ".mlp_hyper_connection.mapping_proj.weight",
        mhc_pfx + ".mlp_hyper_connection.alpha_pre",
        mhc_pfx + ".mlp_hyper_connection.alpha_post",
        mhc_pfx + ".mlp_hyper_connection.alpha_res",
        mhc_pfx + ".mlp_hyper_connection.bias",
        pfx + ".input_layernorm.weight",
        pfx + ".post_attention_layernorm.weight",
        pfx + ".self_attn.vha_postmix_U",
        pfx + ".self_attn.vha_postmix_V",
        pfx + ".self_attn.gate_proj.weight",
        pfx + ".self_attn.o_proj.weight",
    };

    if (!is_dsa) {
        req.push_back(pfx + ".self_attn.linear_q_down_proj.weight");
        req.push_back(pfx + ".self_attn.q_layernorm.weight");
        req.push_back(pfx + ".self_attn.linear_q_up_proj.weight");
        req.push_back(pfx + ".self_attn.linear_kv_proj.weight");
        req.push_back(pfx + ".self_attn.kv_layernorm.weight");
        req.push_back(pfx + ".self_attn.core_attention.compressor.linear_wkv.weight");
        req.push_back(pfx + ".self_attn.core_attention.compressor.linear_wgate.weight");
        req.push_back(pfx + ".self_attn.core_attention.compressor.norm.weight");
        req.push_back(pfx + ".self_attn.core_attention.compressor.ape");
        req.push_back(pfx + ".self_attn.core_attention.attn_sink");
        req.push_back(pfx + ".self_attn.linear_o_group_proj");
    } else {
        req.push_back(pfx + ".self_attn.q_a_proj.weight");
        req.push_back(pfx + ".self_attn.q_a_layernorm.weight");
        req.push_back(pfx + ".self_attn.q_b_proj.weight");
        req.push_back(pfx + ".self_attn.kv_a_proj_with_mqa.weight");
        req.push_back(pfx + ".self_attn.kv_a_layernorm.weight");
        req.push_back(pfx + ".self_attn.kv_b_proj.weight");
        req.push_back(pfx + ".self_attn.core_attention.indexer.wq_b.weight");
        req.push_back(pfx + ".self_attn.core_attention.indexer.wk.weight");
        req.push_back(pfx + ".self_attn.core_attention.indexer.k_norm.weight");
        req.push_back(pfx + ".self_attn.core_attention.indexer.k_norm.bias");
        req.push_back(pfx + ".self_attn.core_attention.indexer.weights_proj.weight");
        req.push_back(pfx + ".self_attn.core_attention.softmax_offset");
    }

    if (is_dense) {
        req.push_back(pfx + ".mlp.gate_proj.weight");
        req.push_back(pfx + ".mlp.up_proj.weight");
        req.push_back(pfx + ".mlp.down_proj.weight");
    } else {
        req.push_back(pfx + ".block_sparse_moe.fc1_latent_proj.weight");
        req.push_back(pfx + ".block_sparse_moe.fc2_latent_proj.weight");
        req.push_back(pfx + ".block_sparse_moe.latent_norm.weight");
        req.push_back(pfx + ".block_sparse_moe.shared_experts.w1.weight");
        req.push_back(pfx + ".block_sparse_moe.shared_experts.w3.weight");
        req.push_back(pfx + ".block_sparse_moe.shared_experts.w2.weight");
        req.push_back(pfx + ".block_sparse_moe.gate.weight");
        req.push_back(pfx + ".block_sparse_moe.gate.weight_1");
        req.push_back(pfx + ".block_sparse_moe.e_score_correction_bias");
        req.push_back(pfx + ".block_sparse_moe.gate.routed_scaling_factor_param");
    }

    if (is_echo) {
        req.push_back(pfx + ".enorm.weight");
        req.push_back(pfx + ".hnorm.weight");
        req.push_back(pfx + ".e_proj.weight");
        req.push_back(pfx + ".h_proj.weight");
        req.push_back(pfx + ".hc_head_fn");
        req.push_back(pfx + ".hc_head_scale");
        req.push_back(pfx + ".hc_head_base");
        req.push_back(pfx + ".shared_head.norm.weight");
    }

    auto b = stream_fetch_tensors(idx, req, opt.http_workers);
    Ernie55SafetensorsWriter w;

    add_mhc_to_writer(w, b, mhc_pfx + ".self_attention_hyper_connection", "attn_mhc");
    add_mhc_to_writer(w, b, mhc_pfx + ".mlp_hyper_connection",            "mlp_mhc");
    w.add_bf16("input_layernorm.weight",          {kHiddenSize}, b.to_bf16(pfx + ".input_layernorm.weight"));
    w.add_bf16("post_attention_layernorm.weight", {kHiddenSize}, b.to_bf16(pfx + ".post_attention_layernorm.weight"));

    // Pre-fold VHA M = I + V @ U^T [64, 64] FP32
    auto U = b.to_f32(pfx + ".self_attn.vha_postmix_U");
    auto V = b.to_f32(pfx + ".self_attn.vha_postmix_V");
    auto vha_m = fold_vha_matrix_64x64(U.data(), V.data(), 64, kVhaRank);
    w.add_f32("attn.vha_m", {64, 64}, vha_m);

    if (!is_dsa) {
        w.add_bf16("attn.q_down",     {1024, 4096},     b.to_bf16(pfx + ".self_attn.linear_q_down_proj.weight"));
        w.add_bf16("attn.q_norm",     {1024},           b.to_bf16(pfx + ".self_attn.q_layernorm.weight"));
        w.add_bf16("attn.q_up",       {32768, 1024},    b.to_bf16(pfx + ".self_attn.linear_q_up_proj.weight"));
        w.add_bf16("attn.kv_proj",    {512, 4096},      b.to_bf16(pfx + ".self_attn.linear_kv_proj.weight"));
        w.add_bf16("attn.kv_norm",    {512},            b.to_bf16(pfx + ".self_attn.kv_layernorm.weight"));
        w.add_bf16("attn.comp_wkv",   {512, 4096},      b.to_bf16(pfx + ".self_attn.core_attention.compressor.linear_wkv.weight"));
        w.add_bf16("attn.comp_wgate", {512, 4096},      b.to_bf16(pfx + ".self_attn.core_attention.compressor.linear_wgate.weight"));
        w.add_bf16("attn.comp_norm",  {512},            b.to_bf16(pfx + ".self_attn.core_attention.compressor.norm.weight"));
        w.add_f32("attn.comp_ape",    {128, 512},       b.to_f32(pfx + ".self_attn.core_attention.compressor.ape"));
        w.add_f32("attn.attn_sink",   {64},             b.to_f32(pfx + ".self_attn.core_attention.attn_sink"));
        w.add_bf16("attn.wo_a",       {8, 1024, 4096},  b.to_bf16(pfx + ".self_attn.linear_o_group_proj"));
        w.add_bf16("attn.gate_proj",  {8192, 1024},     b.to_bf16(pfx + ".self_attn.gate_proj.weight"));
        w.add_bf16("attn.o_proj",     {4096, 8192},     b.to_bf16(pfx + ".self_attn.o_proj.weight"));
    } else {
        w.add_bf16("attn.q_a",        {1024, 4096},     b.to_bf16(pfx + ".self_attn.q_a_proj.weight"));
        w.add_bf16("attn.q_a_norm",   {1024},           b.to_bf16(pfx + ".self_attn.q_a_layernorm.weight"));
        w.add_bf16("attn.q_b",        {16384, 1024},    b.to_bf16(pfx + ".self_attn.q_b_proj.weight"));
        w.add_bf16("attn.kv_a",       {576, 4096},      b.to_bf16(pfx + ".self_attn.kv_a_proj_with_mqa.weight"));
        w.add_bf16("attn.kv_a_norm",  {512},            b.to_bf16(pfx + ".self_attn.kv_a_layernorm.weight"));
        w.add_bf16("attn.kv_b",       {28672, 512},     b.to_bf16(pfx + ".self_attn.kv_b_proj.weight"));
        w.add_bf16("attn.idx_wq_b",   {8192, 1024},     b.to_bf16(pfx + ".self_attn.core_attention.indexer.wq_b.weight"));
        w.add_bf16("attn.idx_wk",     {128, 4096},      b.to_bf16(pfx + ".self_attn.core_attention.indexer.wk.weight"));
        w.add_bf16("attn.idx_kn_w",   {128},            b.to_bf16(pfx + ".self_attn.core_attention.indexer.k_norm.weight"));
        w.add_bf16("attn.idx_kn_b",   {128},            b.to_bf16(pfx + ".self_attn.core_attention.indexer.k_norm.bias"));
        w.add_bf16("attn.idx_w_proj", {64, 4096},       b.to_bf16(pfx + ".self_attn.core_attention.indexer.weights_proj.weight"));
        w.add_f32("attn.attn_sink",   {64},             b.to_f32(pfx + ".self_attn.core_attention.softmax_offset"));
        w.add_bf16("attn.gate_proj",  {16384, 1024},    b.to_bf16(pfx + ".self_attn.gate_proj.weight"));
        w.add_bf16("attn.o_proj",     {4096, 16384},    b.to_bf16(pfx + ".self_attn.o_proj.weight"));

        // Bake FlashIndexer dynamic head-routing gate + bias from meta_dir/flashindexer.safetensors if present!
        const std::string fi_path = opt.meta_dir + "/flashindexer.safetensors";
        if (fs::is_regular_file(fi_path)) {
            hy4::SafetensorsMmap fi(fi_path);
            const std::string gk = pfx + ".self_attn.core_attention.indexer.linear_gate.weight";
            const std::string bk = pfx + ".self_attn.core_attention.indexer.flashindexer_bias";
            if (fi.contains(gk) && fi.contains(bk)) {
                const auto& ge = fi.at(gk);
                const auto& be = fi.at(bk);
                std::vector<uint16_t> gw(size_t(ge.byte_size()) / 2u);
                std::memcpy(gw.data(), fi.tensor_bytes(gk), size_t(ge.byte_size()));
                std::vector<float> bv(64, 0.0f);
                if (be.dtype == "F32") {
                    std::memcpy(bv.data(), fi.tensor_bytes(bk), 64u * 4u);
                } else {
                    const auto* bp = reinterpret_cast<const uint16_t*>(fi.tensor_bytes(bk));
                    for (int hi = 0; hi < 64; ++hi) bv[size_t(hi)] = bf16_to_f32(bp[hi]);
                }
                w.add_bf16("attn.flashindexer_gate", {64, kHiddenSize}, gw);
                w.add_f32("attn.flashindexer_bias",  {64},              bv);
            }
            fi.release_pages();
        }
    }

    if (is_dense) {
        w.add_bf16("mlp.gate", {mcfg.intermediate_size, kHiddenSize}, b.to_bf16(pfx + ".mlp.gate_proj.weight"));
        w.add_bf16("mlp.up",   {mcfg.intermediate_size, kHiddenSize}, b.to_bf16(pfx + ".mlp.up_proj.weight"));
        w.add_bf16("mlp.down", {kHiddenSize, mcfg.intermediate_size}, b.to_bf16(pfx + ".mlp.down_proj.weight"));
    } else {
        w.add_bf16("moe.fc1_lat",  {kLatentDim, kHiddenSize},    b.to_bf16(pfx + ".block_sparse_moe.fc1_latent_proj.weight"));
        w.add_bf16("moe.fc2_lat",  {kHiddenSize, kLatentDim},    b.to_bf16(pfx + ".block_sparse_moe.fc2_latent_proj.weight"));
        w.add_bf16("moe.lat_norm", {kLatentDim},                 b.to_bf16(pfx + ".block_sparse_moe.latent_norm.weight"));
        w.add_bf16("moe.sh_w1",    {kMoeInterSize, kHiddenSize}, b.to_bf16(pfx + ".block_sparse_moe.shared_experts.w1.weight"));
        w.add_bf16("moe.sh_w3",    {kMoeInterSize, kHiddenSize}, b.to_bf16(pfx + ".block_sparse_moe.shared_experts.w3.weight"));
        w.add_bf16("moe.sh_w2",    {kHiddenSize, kMoeInterSize}, b.to_bf16(pfx + ".block_sparse_moe.shared_experts.w2.weight"));

        auto rw0 = b.to_f32(pfx + ".block_sparse_moe.gate.weight");
        auto rw1 = b.to_f32(pfx + ".block_sparse_moe.gate.weight_1");
        std::vector<float> rw01;
        rw01.reserve(rw0.size() + rw1.size());
        rw01.insert(rw01.end(), rw0.begin(), rw0.end());
        rw01.insert(rw01.end(), rw1.begin(), rw1.end());
        w.add_f32("moe.router_w01",  {1024, kHiddenSize},    rw01);
        w.add_f32("moe.corr_bias",   {kNumRoutedExperts},    b.to_f32(pfx + ".block_sparse_moe.e_score_correction_bias"));
        w.add_f32("moe.scale_param", {kNumRoutedExperts},    b.to_f32(pfx + ".block_sparse_moe.gate.routed_scaling_factor_param"));
    }

    if (is_echo) {
        w.add_bf16("echo.enorm",            {kHiddenSize},              b.to_bf16(pfx + ".enorm.weight"));
        w.add_bf16("echo.hnorm",            {kHiddenSize},              b.to_bf16(pfx + ".hnorm.weight"));
        w.add_bf16("echo.eproj",            {kHiddenSize, kHiddenSize}, b.to_bf16(pfx + ".e_proj.weight"));
        w.add_bf16("echo.hproj",            {kHiddenSize, kHiddenSize}, b.to_bf16(pfx + ".h_proj.weight"));
        const auto& ehf_sp = b.spec(pfx + ".hc_head_fn");
        auto ehf = transpose_if_needed(b.to_f32(pfx + ".hc_head_fn"), ehf_sp.shape.at(0), ehf_sp.shape.at(1), kHcMult, kHcDim);
        w.add_f32("echo.hc_head_fn",        {kHcMult, kHcDim},          ehf);
        w.add_f32("echo.hc_head_scale",     {1},                        b.to_f32(pfx + ".hc_head_scale"));
        w.add_f32("echo.hc_head_base",      {kHcMult},                  b.to_f32(pfx + ".hc_head_base"));
        w.add_bf16("echo.shared_head_norm", {kHiddenSize},              b.to_bf16(pfx + ".shared_head.norm.weight"));
    }

    w.write_atomic(fname);
    std::printf("[ERNIE55-BUILD] Spine L%02d (%s%s) -> %.1f MiB wire in %.1fs\n",
                l, is_dsa ? "DSA" : "HCA", is_echo ? "+ECHO" : (is_dense ? "+DENSE" : "+MoE"),
                double(b.total_wire_bytes) / 1048576.0, b.elapsed_sec);
    std::fflush(stdout);
}

// Auto-ingest bundled Nov 25 DSpark weights under model.mtp.* / dspark.* / model.dspark.* if present!
void maybe_pack_bundled_dspark(const RemoteRangeIndex& idx, const BuildOpts& opt, const Ernie55ModelConfig& mcfg) {
    const std::string out_p = opt.out_dir + "/packed_spine/spine_dspark.safetensors";
    if (opt.skip_existing && fs::is_regular_file(out_p)) return;

    std::string root_pfx;
    std::vector<std::string> req;
    for (const auto& nm : idx.all_names) {
        if (nm.find("_scale_inv") != std::string::npos || nm.find(".experts.") != std::string::npos) continue;
        if (nm.rfind("model.mtp.", 0) == 0) {
            if (root_pfx.empty()) root_pfx = "model.mtp";
            req.push_back(nm);
        } else if (nm.rfind("dspark.", 0) == 0) {
            if (root_pfx.empty()) root_pfx = "dspark";
            req.push_back(nm);
        } else if (nm.rfind("model.dspark.", 0) == 0) {
            if (root_pfx.empty()) root_pfx = "model.dspark";
            req.push_back(nm);
        }
    }
    if (req.empty() || !mcfg.has_bundled_dspark()) {
        if (!req.empty() && !mcfg.has_bundled_dspark()) {
            std::printf("[ERNIE55-BUILD] Notice: found %zu extra '%s.*' keys without dspark_config in config.json — safely ignoring.\n",
                        req.size(), root_pfx.c_str());
        }
        return;
    }

    std::printf("[ERNIE55-BUILD] Detected bundled Nov 25 DSpark draft (%zu tensors under %s.*) — packing spine_dspark.safetensors...\n",
                req.size(), root_pfx.c_str());
    std::fflush(stdout);
    auto b = stream_fetch_tensors(idx, req, opt.http_workers);
    Ernie55SafetensorsWriter w;

    auto opt_pick = [&](const std::string& a, const std::string& b_alt) -> std::string {
        if (b.has(a)) return a;
        if (b.has(b_alt)) return b_alt;
        return "";
    };
    const int n_ds = int(mcfg.dspark_target_layer_ids.size());
    const std::string mp = opt_pick(root_pfx + ".feature_projection.weight", root_pfx + ".main_proj.weight");
    const std::string mn = opt_pick(root_pfx + ".feature_norm.weight",       root_pfx + ".main_norm.weight");
    if (!mp.empty()) w.add_bf16("dspark.main_proj", {kHiddenSize, int64_t(n_ds) * kHiddenSize}, b.to_bf16(mp));
    if (!mn.empty()) w.add_bf16("dspark.main_norm", {kHiddenSize}, b.to_bf16(mn));
    if (b.has(root_pfx + ".norm.weight")) w.add_bf16("dspark.norm", {kHiddenSize}, b.to_bf16(root_pfx + ".norm.weight"));
    const std::string hcf = opt_pick(root_pfx + ".mhc_contract.hc_head_fn",    root_pfx + ".hc_head_fn");
    const std::string hcs = opt_pick(root_pfx + ".mhc_contract.hc_head_scale", root_pfx + ".hc_head_scale");
    const std::string hcb = opt_pick(root_pfx + ".mhc_contract.hc_head_base",  root_pfx + ".hc_head_base");
    if (!hcf.empty() && !hcs.empty() && !hcb.empty()) {
        const auto& sp = b.spec(hcf);
        auto fn = transpose_if_needed(b.to_f32(hcf), sp.shape.at(0), sp.shape.at(1), kHcMult, kHcDim);
        w.add_f32("dspark.hc_head_fn",    {kHcMult, kHcDim}, fn);
        w.add_f32("dspark.hc_head_scale", {1},               b.to_f32(hcs));
        w.add_f32("dspark.hc_head_base",  {kHcMult},         b.to_f32(hcb));
    }
    if (b.has(root_pfx + ".markov_head.markov_w1.weight")) {
        const auto& sp = b.spec(root_pfx + ".markov_head.markov_w1.weight");
        auto w1t = transpose_if_needed(b.to_f32(root_pfx + ".markov_head.markov_w1.weight"),
                                       sp.shape.at(0), sp.shape.at(1), mcfg.vocab_size, mcfg.dspark_markov_rank);
        std::vector<uint16_t> w1t_bf16(w1t.size());
        for (size_t k = 0; k < w1t.size(); ++k) w1t_bf16[k] = f32_to_bf16_rne(w1t[k]);
        w.add_bf16("dspark.markov_w1_T", {mcfg.vocab_size, mcfg.dspark_markov_rank}, w1t_bf16);
    }
    if (b.has(root_pfx + ".markov_head.markov_w2.weight")) {
        w.add_bf16("dspark.markov_w2", {mcfg.vocab_size, mcfg.dspark_markov_rank},
                   b.to_bf16(root_pfx + ".markov_head.markov_w2.weight"));
    }
    const std::string cf = opt_pick(root_pfx + ".confidence_head.projection.weight", root_pfx + ".confidence_head.proj.weight");
    if (!cf.empty()) {
        auto cv = b.to_f32(cf);
        w.add_f32("dspark.conf_proj_w", {int64_t(cv.size())}, cv);
    }

    std::vector<uint16_t> ctx_wkv;
    for (int i = 0; i < n_ds; ++i) {
        std::string spfx = root_pfx + "." + std::to_string(i);
        if (!b.has(spfx + ".input_layernorm.weight")) spfx = root_pfx + ".layers." + std::to_string(i);
        if (!b.has(spfx + ".input_layernorm.weight")) break;
        const std::string dpfx = "dspark.layers." + std::to_string(i) + ".";
        add_mhc_to_writer(w, b, spfx + ".self_attention_hyper_connection", dpfx + "attn_mhc");
        add_mhc_to_writer(w, b, spfx + ".mlp_hyper_connection",            dpfx + "mlp_mhc");
        w.add_bf16(dpfx + "input_layernorm.weight",          {kHiddenSize}, b.to_bf16(spfx + ".input_layernorm.weight"));
        w.add_bf16(dpfx + "post_attention_layernorm.weight", {kHiddenSize}, b.to_bf16(spfx + ".post_attention_layernorm.weight"));
        auto U = b.to_f32(spfx + ".self_attn.vha_postmix_U");
        auto V = b.to_f32(spfx + ".self_attn.vha_postmix_V");
        w.add_f32(dpfx + "attn.vha_m", {64, 64}, fold_vha_matrix_64x64(U.data(), V.data(), 64, kVhaRank));
        auto kvp = b.to_bf16(spfx + ".self_attn.linear_kv_proj.weight");
        ctx_wkv.insert(ctx_wkv.end(), kvp.begin(), kvp.end());
        w.add_bf16(dpfx + "attn.q_down",    {1024, 4096},    b.to_bf16(spfx + ".self_attn.linear_q_down_proj.weight"));
        w.add_bf16(dpfx + "attn.q_norm",    {1024},          b.to_bf16(spfx + ".self_attn.q_layernorm.weight"));
        w.add_bf16(dpfx + "attn.q_up",      {32768, 1024},   b.to_bf16(spfx + ".self_attn.linear_q_up_proj.weight"));
        w.add_bf16(dpfx + "attn.kv_proj",   {512, 4096},     kvp);
        w.add_bf16(dpfx + "attn.kv_norm",   {512},           b.to_bf16(spfx + ".self_attn.kv_layernorm.weight"));
        w.add_f32(dpfx + "attn.attn_sink",  {64},            b.to_f32(spfx + ".self_attn.core_attention.attn_sink"));
        w.add_bf16(dpfx + "attn.wo_a",      {8, 1024, 4096}, b.to_bf16(spfx + ".self_attn.linear_o_group_proj"));
        w.add_bf16(dpfx + "attn.gate_proj", {8192, 1024},    b.to_bf16(spfx + ".self_attn.gate_proj.weight"));
        w.add_bf16(dpfx + "attn.o_proj",    {4096, 8192},    b.to_bf16(spfx + ".self_attn.o_proj.weight"));

        // Pack full 512-expert Latent MoE spine for this DSpark layer!
        if (b.has(spfx + ".block_sparse_moe.fc1_latent_proj.weight")) {
            w.add_bf16(dpfx + "moe.fc1_lat",  {kLatentDim, kHiddenSize},    b.to_bf16(spfx + ".block_sparse_moe.fc1_latent_proj.weight"));
            w.add_bf16(dpfx + "moe.fc2_lat",  {kHiddenSize, kLatentDim},    b.to_bf16(spfx + ".block_sparse_moe.fc2_latent_proj.weight"));
            w.add_bf16(dpfx + "moe.lat_norm", {kLatentDim},                 b.to_bf16(spfx + ".block_sparse_moe.latent_norm.weight"));
            w.add_bf16(dpfx + "moe.sh_w1",    {kMoeInterSize, kHiddenSize}, b.to_bf16(spfx + ".block_sparse_moe.shared_experts.w1.weight"));
            w.add_bf16(dpfx + "moe.sh_w3",    {kMoeInterSize, kHiddenSize}, b.to_bf16(spfx + ".block_sparse_moe.shared_experts.w3.weight"));
            w.add_bf16(dpfx + "moe.sh_w2",    {kHiddenSize, kMoeInterSize}, b.to_bf16(spfx + ".block_sparse_moe.shared_experts.w2.weight"));
            auto rw0 = b.to_f32(spfx + ".block_sparse_moe.gate.weight");
            auto rw1 = b.to_f32(spfx + ".block_sparse_moe.gate.weight_1");
            std::vector<float> rw01;
            rw01.reserve(rw0.size() + rw1.size());
            rw01.insert(rw01.end(), rw0.begin(), rw0.end());
            rw01.insert(rw01.end(), rw1.begin(), rw1.end());
            w.add_f32(dpfx + "moe.router_w01",  {1024, kHiddenSize}, rw01);
            w.add_f32(dpfx + "moe.corr_bias",   {kNumRoutedExperts}, b.to_f32(spfx + ".block_sparse_moe.e_score_correction_bias"));
            w.add_f32(dpfx + "moe.scale_param", {kNumRoutedExperts}, b.to_f32(spfx + ".block_sparse_moe.gate.routed_scaling_factor_param"));
        }
    }
    if (!ctx_wkv.empty()) {
        w.add_bf16("dspark.context_wkv_proj", {int64_t(ctx_wkv.size() / kHiddenSize), kHiddenSize}, ctx_wkv);
    }
    w.write_atomic(out_p);
    std::printf("[ERNIE55-BUILD] Wrote bundled DSpark spine %s (%zu tensors)\n", out_p.c_str(), w.count());
    std::fflush(stdout);
}

// Compute real calibration activations -> FP64 RAREBAL-LDLQ WHT128-HLUT4 for one MoE layer (512 experts)
void build_hlut4_layer(const RemoteRangeIndex& idx, const BuildOpts& opt, const Ernie55ModelConfig& mcfg,
                       int l, const std::vector<float>& cal_x_in_MxH, std::vector<float>& cal_x_next_MxH)
{
    char rel_name[64];
    std::snprintf(rel_name, sizeof(rel_name), "hlut4_layer_%02d.bin", l);
    const std::string link_dir = opt.out_dir + "/experts_hlut4";
    const std::string store_dir = opt.experts_dir.empty() ? link_dir : opt.experts_dir;
    fs::create_directories(link_dir);
    fs::create_directories(store_dir);

    const std::string real_path = store_dir + "/" + rel_name;
    const std::string link_path = link_dir + "/" + rel_name;

    if (opt.skip_existing && fs::is_regular_file(real_path) &&
        fs::file_size(real_path) == sizeof(Ernie55Hlut4LayerHeader) + uint64_t(kNumRoutedExperts) * kHlut4ExpertBytes)
    {
        if (real_path != link_path && !fs::exists(link_path)) fs::create_symlink(real_path, link_path);
        std::printf("[ERNIE55-BUILD] Skipping existing verified %s\n", real_path.c_str());
        std::fflush(stdout);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const std::string pfx = "model.layers." + std::to_string(l) + ".block_sparse_moe";
    constexpr int kBatchExperts = 64;  // Double-buffered 8 x 64 experts (1.5 GiB cur + 1.5 GiB prefetch = 3.0 GiB RAM!)

    // Sort experts by (shard_file, file_offset) so every 64-expert batch coalesces into huge contiguous 256 MiB bursts!
    // Keep expert 0 at index 0 so GATE-1 ref always uses expert 0 cleanly.
    std::vector<int> exp_order(kNumRoutedExperts);
    std::iota(exp_order.begin(), exp_order.end(), 0);
    std::sort(exp_order.begin() + 1, exp_order.end(), [&](int a, int b) {
        const auto& sa = idx.by_name.at(pfx + ".experts." + std::to_string(a) + ".w1.weight");
        const auto& sb = idx.by_name.at(pfx + ".experts." + std::to_string(b) + ".w1.weight");
        if (sa.shard_file != sb.shard_file) return sa.shard_file < sb.shard_file;
        return sa.file_offset < sb.file_offset;
    });

    auto fetch_expert_batch = [&](int b0, int b1, bool include_fc) {
        std::vector<std::string> req;
        req.reserve(size_t(b1 - b0) * 3u + (include_fc ? 2u : 0u));
        if (include_fc) {
            req.push_back(pfx + ".fc1_latent_proj.weight");
            req.push_back(pfx + ".fc2_latent_proj.weight");
        }
        for (int bi = b0; bi < b1; ++bi) {
            const int e = exp_order[size_t(bi)];
            const std::string ep = pfx + ".experts." + std::to_string(e);
            req.push_back(ep + ".w1.weight");
            req.push_back(ep + ".w3.weight");
            req.push_back(ep + ".w2.weight");
        }
        return stream_fetch_tensors(idx, req, opt.http_workers);
    };

    // Batch 0: first 64 shard-contiguous experts (incl. exp0) + fc1/fc2 (~1.5 GiB BF16)
    double stream_s = 0.0;
    auto bundle = fetch_expert_batch(0, std::min(kBatchExperts, kNumRoutedExperts), true);
    stream_s += bundle.elapsed_sec;

    const auto signs_in    = make_ernie55_signs_in(l);
    const auto signs_inter = make_ernie55_signs_inter(l);

    // Build real layer-l latent activations h_lat = fc1_latent_proj(cal_x_in) -> FP64 RAREBAL Hessian H_in & R_up_in!
    const int M = int(cal_x_in_MxH.size() / kHiddenSize);
    std::vector<float> R_up_in, R_up_inter;
    RarebalStats st_in{}, st_inter{};
    double cond_in = 1.0, cond_inter = 1.0;

    std::vector<float> h_lat(size_t(M) * kLatentDim, 0.0f);
    {
        auto fc1 = bundle.to_f32(pfx + ".fc1_latent_proj.weight");
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const float* xr = cal_x_in_MxH.data() + size_t(m) * kHiddenSize;
            float* yr = h_lat.data() + size_t(m) * kLatentDim;
            for (int d = 0; d < kLatentDim; ++d) {
                const float* wr = fc1.data() + size_t(d) * kHiddenSize;
                float acc = 0.0f;
                for (int k = 0; k < kHiddenSize; ++k) acc += wr[k] * xr[k];
                yr[d] = acc;
            }
        }
    }

    if (opt.use_ldlq && M >= 16) {
        std::vector<float> h_lat_rot(h_lat.size());
        wht128_rows(h_lat.data(), h_lat_rot.data(), M, kLatentDim, signs_in);
        auto H_in = build_rarebal_fp64_hessian(h_lat_rot.data(), M, kLatentDim, true, &st_in);
        // Scale damping cleanly when M < K so LDLQ improves activation loss without nullspace blowup!
        const double damp_in = (M < kLatentDim) ? 0.12 : 0.02;
        R_up_in = build_fp64_ldlq_rup_f32(H_in.data(), kLatentDim, damp_in, &cond_in);

        // Compute representative post-SiTU-GLU activations across first 16 experts of Batch 0 -> full-rank H_inter!
        const int n_probe_e = 16;
        const int m_per_e = std::min(M, 128);
        std::vector<float> z_samples(size_t(n_probe_e * m_per_e) * kMoeInterSize, 0.0f);
        for (int pi = 0; pi < n_probe_e; ++pi) {
            const int e = exp_order[size_t(pi)];
            const std::string ep = pfx + ".experts." + std::to_string(e);
            auto w1 = bundle.to_f32(ep + ".w1.weight");
            auto w3 = bundle.to_f32(ep + ".w3.weight");
            #pragma omp parallel for schedule(static)
            for (int mi = 0; mi < m_per_e; ++mi) {
                const float* hr = h_lat.data() + size_t((mi + pi * 7) % M) * kLatentDim;
                float* zr = z_samples.data() + size_t(pi * m_per_e + mi) * kMoeInterSize;
                for (int i = 0; i < kMoeInterSize; ++i) {
                    const float* g_row = w1.data() + size_t(i) * kLatentDim;
                    const float* u_row = w3.data() + size_t(i) * kLatentDim;
                    float g = 0.0f, u = 0.0f;
                    for (int k = 0; k < kLatentDim; ++k) { g += g_row[k] * hr[k]; u += u_row[k] * hr[k]; }
                    zr[i] = ernie55_situ_glu(g, u, mcfg.situ_beta, mcfg.situ_linear_beta);
                }
            }
        }
        std::vector<float> z_rot(z_samples.size());
        wht128_rows(z_samples.data(), z_rot.data(), n_probe_e * m_per_e, kMoeInterSize, signs_inter);
        auto H_inter = build_rarebal_fp64_hessian(z_rot.data(), n_probe_e * m_per_e, kMoeInterSize, true, &st_inter);
        R_up_inter = build_fp64_ldlq_rup_f32(H_inter.data(), kMoeInterSize, 0.05, &cond_inter);
    }

    // Save GATE-1 BF16 reference snapshot on L01, L07 (DSA), and L43 (ECHO) for 0-extra-download GPU cos verification!
    if (opt.save_gate1_ref && (l == 1 || l == 7 || l == 43)) {
        const std::string ref_dir = opt.out_dir + "/gate1_refs";
        fs::create_directories(ref_dir);
        char rp[512];
        std::snprintf(rp, sizeof(rp), "%s/ref_layer_%02d_exp0.bin", ref_dir.c_str(), l);
        auto w1 = bundle.to_f32(pfx + ".experts.0.w1.weight");
        auto w3 = bundle.to_f32(pfx + ".experts.0.w3.weight");
        auto w2 = bundle.to_f32(pfx + ".experts.0.w2.weight");
        std::ofstream rfs(rp, std::ios::binary);
        rfs.write(reinterpret_cast<const char*>(h_lat.data()), std::streamsize(size_t(std::min(M, 8)) * kLatentDim * sizeof(float)));
        rfs.write(reinterpret_cast<const char*>(w1.data()), std::streamsize(w1.size() * sizeof(float)));
        rfs.write(reinterpret_cast<const char*>(w3.data()), std::streamsize(w3.size() * sizeof(float)));
        rfs.write(reinterpret_cast<const char*>(w2.data()), std::streamsize(w2.size() * sizeof(float)));
    }

    // Propagate lightweight residual activation drift for next layer's calibration Hessian
    cal_x_next_MxH = cal_x_in_MxH;
    {
        auto fc2 = bundle.to_f32(pfx + ".fc2_latent_proj.weight");
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const float* hr = h_lat.data() + size_t(m) * kLatentDim;
            float* xr = cal_x_next_MxH.data() + size_t(m) * kHiddenSize;
            for (int d = 0; d < kHiddenSize; ++d) {
                const float* wr = fc2.data() + size_t(d) * kLatentDim;
                float acc = 0.0f;
                for (int k = 0; k < kLatentDim; ++k) acc += wr[k] * hr[k];
                xr[d] += 0.1f * acc;
            }
        }
    }

    // Quantize all 512 experts (w1, w3, w2) with overlapped double-buffered HTTP/2 prefetch!
    const std::string tmp_path = real_path + ".tmp";
    const int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) throw std::runtime_error("Cannot create " + tmp_path);

    Ernie55Hlut4LayerHeader hdr{};
    hdr.magic          = kErnie55Hlut4ShardMagic;
    hdr.version        = kErnie55Hlut4ShardVersion;
    hdr.layer_idx      = uint32_t(l);
    hdr.num_experts    = uint32_t(kNumRoutedExperts);
    hdr.latent_dim     = uint32_t(kLatentDim);
    hdr.moe_inter_dim  = uint32_t(kMoeInterSize);
    hdr.tiles_per_proj = uint32_t(kExpProjTiles);
    hdr.ldlq_used      = (!R_up_in.empty() && !R_up_inter.empty()) ? 1u : 0u;
    std::memcpy(hdr.signs_in,    signs_in.data(),    sizeof(hdr.signs_in));
    std::memcpy(hdr.signs_inter, signs_inter.data(), sizeof(hdr.signs_inter));
    hdr.hessian_cond_in    = cond_in;
    hdr.hessian_cond_inter = cond_inter;
    hdr.rarebal_lambda     = st_in.lambda_tarq;

    if (::pwrite(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr))) {
        ::close(fd); ::unlink(tmp_path.c_str());
        throw std::runtime_error("Failed writing header to " + tmp_path);
    }

    double sqnr_acc = 0.0;
    std::array<uint32_t, kNumRoutedExperts> exp_crcs{};
    std::vector<Hlut4Tile32x32> exp_buf(3u * kExpProjTiles);

    for (int b0 = 0; b0 < kNumRoutedExperts; b0 += kBatchExperts) {
        const int b1 = std::min(b0 + kBatchExperts, kNumRoutedExperts);
        const int nb0 = b1;
        const int nb1 = std::min(nb0 + kBatchExperts, kNumRoutedExperts);

        // Launch async HTTP/2 prefetch of next batch while CPU quantizes current batch!
        StreamedTensorBundle next_bundle;
        std::exception_ptr fetch_err;
        std::thread prefetch_th;
        if (nb0 < kNumRoutedExperts) {
            prefetch_th = std::thread([&]() {
                try {
                    next_bundle = fetch_expert_batch(nb0, nb1, false);
                } catch (...) {
                    fetch_err = std::current_exception();
                }
            });
        }

        for (int bi = b0; bi < b1; ++bi) {
            const int e = exp_order[size_t(bi)];
            const std::string ep = pfx + ".experts." + std::to_string(e);
            double s1 = 0.0, s3 = 0.0, s2 = 0.0;
            auto w1_f = bundle.to_f32(ep + ".w1.weight");
            auto q1   = quantize_wht128_hlut4_fast(w1_f.data(), kMoeInterSize, kLatentDim, signs_in, R_up_in, &s1);
            auto w3_f = bundle.to_f32(ep + ".w3.weight");
            auto q3   = quantize_wht128_hlut4_fast(w3_f.data(), kMoeInterSize, kLatentDim, signs_in, R_up_in, &s3);
            auto w2_f = bundle.to_f32(ep + ".w2.weight");
            auto q2   = quantize_wht128_hlut4_fast(w2_f.data(), kLatentDim, kMoeInterSize, signs_inter, R_up_inter, &s2);

            sqnr_acc += (s1 + s3 + s2) / 3.0;
            std::memcpy(exp_buf.data() + 0u * kExpProjTiles, q1.data(), kHlut4ProjBytes);
            std::memcpy(exp_buf.data() + 1u * kExpProjTiles, q3.data(), kHlut4ProjBytes);
            std::memcpy(exp_buf.data() + 2u * kExpProjTiles, q2.data(), kHlut4ProjBytes);

            exp_crcs[size_t(e)] = crc32c_update(0u, exp_buf.data(), kHlut4ExpertBytes);
            const off_t off = off_t(sizeof(hdr)) + off_t(e) * off_t(kHlut4ExpertBytes);
            size_t written = 0;
            while (written < kHlut4ExpertBytes) {
                const ssize_t wr = ::pwrite(fd, reinterpret_cast<const uint8_t*>(exp_buf.data()) + written,
                                            kHlut4ExpertBytes - written, off + off_t(written));
                if (wr <= 0) {
                    if (prefetch_th.joinable()) prefetch_th.join();
                    ::close(fd); ::unlink(tmp_path.c_str());
                    throw std::runtime_error("Short write " + tmp_path);
                }
                written += size_t(wr);
            }
        }

        // Release current batch RAM immediately before joining prefetched next batch!
        bundle = {};
        (void)::fdatasync(fd);
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        if (prefetch_th.joinable()) {
            prefetch_th.join();
            if (fetch_err) {
                ::close(fd); ::unlink(tmp_path.c_str());
                std::rethrow_exception(fetch_err);
            }
            stream_s += next_bundle.elapsed_sec;
            bundle = std::move(next_bundle);
        }
    }

    const uint32_t payload_crc = crc32c_update(0u, exp_crcs.data(), sizeof(exp_crcs));

    hdr.avg_sqnr_db    = sqnr_acc / double(kNumRoutedExperts);
    hdr.payload_crc32c = payload_crc;
    hdr.header_crc32c  = crc32c_update(0u, &hdr, offsetof(Ernie55Hlut4LayerHeader, header_crc32c));
    if (::pwrite(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr))) {
        ::close(fd); ::unlink(tmp_path.c_str());
        throw std::runtime_error("Failed finalizing header " + tmp_path);
    }
    (void)::fdatasync(fd);
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    if (::rename(tmp_path.c_str(), real_path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        throw std::runtime_error("Atomic rename failed: " + real_path);
    }
    if (real_path != link_path) {
        if (fs::exists(link_path) || fs::is_symlink(link_path)) fs::remove(link_path);
        fs::create_symlink(real_path, link_path);
    }

    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[ERNIE55-BUILD] HLUT4 L%02d (512 experts, 3.09 GiB) | SQNR=%.2f dB | LDLQ=%u (cond=%.1f/%.1f, lam=%.2f) | stream=%.1fs total=%.1fs\n",
                l, hdr.avg_sqnr_db, hdr.ldlq_used, cond_in, cond_inter, st_in.lambda_tarq, stream_s, tot_s);
    std::fflush(stdout);
}

} // namespace
} // namespace ernie55

int main(int argc, char** argv) {
    using namespace ernie55;
    BuildOpts opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg for ") + f);
            return argv[++i];
        };
        if (a == "--hf-repo")            opt.hf_repo = need("--hf-repo");
        else if (a == "--revision")      opt.revision = need("--revision");
        else if (a == "--out-dir")       opt.out_dir = need("--out-dir");
        else if (a == "--experts-dir")   opt.experts_dir = need("--experts-dir");
        else if (a == "--meta-dir")      opt.meta_dir = need("--meta-dir");
        else if (a == "--http-workers")  opt.http_workers = std::stoi(need("--http-workers"));
        else if (a == "--cal-tokens")    opt.cal_tokens = std::stoi(need("--cal-tokens"));
        else if (a == "--layer-begin")   opt.layer_begin = std::stoi(need("--layer-begin"));
        else if (a == "--layer-end")     opt.layer_end = std::stoi(need("--layer-end"));
        else if (a == "--scan-only")     opt.scan_only = true;
        else if (a == "--spine-only")    opt.spine_only = true;
        else if (a == "--skip-spine")    opt.skip_spine = true;
        else if (a == "--force")         opt.skip_existing = false;
        else if (a == "--no-ldlq")       opt.use_ldlq = false;
    }

    try {
        if (opt.meta_dir.empty()) opt.meta_dir = opt.out_dir;
        fs::create_directories(opt.out_dir + "/packed_spine");
        fs::create_directories(opt.out_dir + "/experts_hlut4");

        ensure_metadata_files(opt, opt.meta_dir);
        if (opt.meta_dir != opt.out_dir) {
            for (const char* f : {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"}) {
                const std::string src = opt.meta_dir + "/" + f;
                const std::string dst = opt.out_dir + "/" + f;
                if (fs::is_regular_file(src) && !fs::exists(dst)) fs::copy_file(src, dst);
            }
        }

        const auto mcfg = Ernie55ModelConfig::load_json(opt.meta_dir + "/config.json");
        mcfg.validate_compat();

        RemoteRangeIndex idx;
        idx.repo = opt.hf_repo;
        idx.revision = opt.revision;
        const std::string tsv = opt.meta_dir + "/tensor_ranges.tsv";
        if (fs::is_regular_file(tsv) && fs::file_size(tsv) > 1000) {
            idx.load_tsv(tsv);
            std::printf("[ERNIE55-BUILD] Loaded cached byte-range manifest %s (%zu tensors, fp8=%d)\n",
                        tsv.c_str(), idx.all_names.size(), int(idx.is_fp8));
        } else {
            std::printf("[ERNIE55-BUILD] Scanning 128 remote safetensors headers over HTTP/2 (%d workers)...\n", opt.http_workers);
            std::fflush(stdout);
            idx.build_from_hf_index(opt.meta_dir + "/model.safetensors.index.json", tsv, opt.http_workers);
            std::printf("[ERNIE55-BUILD] Built %s (%zu tensors across %zu shards, fp8=%d)\n",
                        tsv.c_str(), idx.all_names.size(), idx.shard_files.size(), int(idx.is_fp8));
        }
        std::fflush(stdout);
        if (opt.scan_only) return 0;

        if (!opt.skip_spine) {
            pack_spine_shared(idx, opt, mcfg);
            const int max_l = mcfg.num_hidden_layers + (mcfg.num_nextn_predict_layers > 0 ? 1 : 0);
            for (int l = 0; l < max_l; ++l) {
                pack_spine_layer(idx, opt, mcfg, l);
            }
            maybe_pack_bundled_dspark(idx, opt, mcfg);
        }
        if (opt.spine_only) return 0;

        // Seed calibration hidden states from real tokenized corpus embeddings!
        Ernie55Tokenizer tok;
        tok.load(opt.meta_dir + "/tokenizer.json", opt.meta_dir + "/tokenizer_config.json");
        std::vector<int32_t> cal_ids;
        for (const char* s : kCalCorpus) {
            auto ids = tok.encode(s, false);
            cal_ids.insert(cal_ids.end(), ids.begin(), ids.end());
        }
        while (int(cal_ids.size()) < opt.cal_tokens) cal_ids.push_back(cal_ids[cal_ids.size() % std::max<size_t>(1, cal_ids.size())]);
        cal_ids.resize(size_t(opt.cal_tokens));

        // Load real embeddings from packed spine_shared.safetensors for calibration seed
        std::vector<float> cal_x(size_t(opt.cal_tokens) * kHiddenSize, 0.0f);
        {
            const std::string sp_path = opt.out_dir + "/packed_spine/spine_shared.safetensors";
            if (fs::is_regular_file(sp_path)) {
                hy4::SafetensorsMmap st(sp_path);
                const auto* emb = reinterpret_cast<const uint16_t*>(st.tensor_bytes("model.embed_tokens.weight"));
                for (int m = 0; m < opt.cal_tokens; ++m) {
                    const int32_t tid = std::clamp(cal_ids[size_t(m)], 0, mcfg.vocab_size - 1);
                    const uint16_t* row = emb + size_t(tid) * kHiddenSize;
                    for (int d = 0; d < kHiddenSize; ++d) cal_x[size_t(m) * kHiddenSize + size_t(d)] = bf16_to_f32(row[d]);
                }
                st.release_pages();
            } else {
                std::mt19937 rng(5523);
                std::normal_distribution<float> nd(0.0f, 0.02f);
                for (float& v : cal_x) v = nd(rng);
            }
        }

        const int l_end = std::min(opt.layer_end, mcfg.num_hidden_layers + (mcfg.num_nextn_predict_layers > 0 ? 0 : -1));
        std::vector<float> cal_next;
        for (int l = std::max(1, opt.layer_begin); l <= l_end; ++l) {
            build_hlut4_layer(idx, opt, mcfg, l, cal_x, cal_next);
            if (!cal_next.empty()) cal_x.swap(cal_next);
        }

        std::printf("[ERNIE55-BUILD] === COMPLETE: 100%% WHT128-HLUT4 build finished in %s ===\n", opt.out_dir.c_str());
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[ERNIE55-BUILD] FATAL: %s\n", ex.what());
        return 1;
    }
}
