// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 Exact Architectural Specification & Dynamic Topology Loader for baidu/ERNIE-5.5-Flash (ernie5_5_moe)
// Zero Python anywhere. Supports 100% uniform WHT128-HLUT4 across all 21,504 backbone + 512 ECHO experts
// with 0 B CPU DDR4 spill across 192G (5x MI50 + 2x P150A), 224G (5x MI50 + 3x P150A), and 240G (6x MI50 + 3x P150A).

#include "ernie55_json.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie55 {

// ---- Exact Ground-Truth ERNIE-5.5-Flash-Preview Constants (verified against config.json & vLLM-ERNIE) ----
inline constexpr int kHiddenSize              = 4096;
inline constexpr int kNumHiddenLayers         = 43;       // L00..L42
inline constexpr int kNumDsparkLayers         = 3;        // L43..L45 (model.mtp.0..2 3L MoE DSpark)
inline constexpr int kNumEchoLayers           = 1;        // legacy Base single-layer ECHO MTP
inline constexpr int kTotalLayersWithEcho     = 46;       // L00..L42 backbone + L43..L45 3L DSpark (46 total slots)
inline constexpr int kTotalLayersWithDraft    = kTotalLayersWithEcho;
inline constexpr int kFirstKDenseReplace      = 1;        // L00 dense FFN; L01..L42 + L43..L45 MoE
inline constexpr int kNumMoeBackboneLayers    = 42;       // L01..L42
inline constexpr int kNumMoeTotalWithDspark   = 45;       // L01..L42 (42) + DSpark 0..2 (3) = 45 MoE layers
inline constexpr int kDenseInterSize          = 14336;    // L00 dense SiTU-GLU MLP
inline constexpr int kVocabSize               = 201216;   // R2L 3-digit grouping BPE (pad_token_id=0)
inline constexpr int kMaxPositionEmbeddings   = 1048576;  // 1M context
inline constexpr int kFlashIndexerTopKHeads   = 8;        // FlashIndexer dynamic top-8 of 64 DSA indexer heads
inline constexpr int kDsparkSwaRingSize       = 256;      // 256-slot power-of-2 ring (> 128 SWA + 7 block queries)
inline constexpr int kDsparkNoiseTokenIdDef   = 1024;
inline constexpr int kDsparkBlockSizeDef      = 7;
inline constexpr int kDsparkMarkovRankDef     = 256;

// ---- mHC Manifold-Constrained Hyper-Connections ----
inline constexpr int   kHcMult                = 4;
inline constexpr int   kHcMult2               = kHcMult * kHcMult;            // 16
inline constexpr int   kHcMult3               = kHcMult * (2 + kHcMult);      // 24
inline constexpr int   kHcDim                 = kHcMult * kHiddenSize;        // 16384
inline constexpr int   kHcSinkhornIters       = 20;
inline constexpr float kHcEps                 = 1e-6f;
inline constexpr float kHcPostAlpha           = 2.0f;
inline constexpr float kRmsNormEps            = 1e-6f;

// ---- Hybrid Attention: 37 HCA (128:1 + 128-SWA) + 6 Backbone mqa_dsa + 1 ECHO mqa_dsa ----
inline constexpr int   kHcaNumHeads           = 64;
inline constexpr int   kHcaHeadDim            = 512;
inline constexpr int   kHcaNopeHeadDim        = 448;
inline constexpr int   kHcaRopeHeadDim        = 64;
inline constexpr int   kHcaQLoraRank          = 1024;
inline constexpr int   kHcaKvLoraRank         = 512;
inline constexpr int   kHcaOGroups            = 8;
inline constexpr int   kHcaOLoraRank          = 1024;
inline constexpr int   kHcaCompressRatio      = 128;
inline constexpr int   kSlidingWindow         = 128;
inline constexpr double kHcaCompressRopeTheta = 160000.0;
inline constexpr double kHcaYarnFactor        = 16.0;
inline constexpr int   kHcaYarnOrigMaxPos     = 65536;
inline constexpr double kHcaYarnBetaFast      = 32.0;
inline constexpr double kHcaYarnBetaSlow      = 1.0;

// ---- mqa_dsa (MLA + Lightning Indexer DSA) ----
inline constexpr int   kDsaNumHeads           = 64;
inline constexpr int   kDsaQLoraRank          = 1024;
inline constexpr int   kDsaKvLoraRank         = 512;
inline constexpr int   kDsaQkNopeHeadDim      = 192;
inline constexpr int   kDsaQkRopeHeadDim      = 64;
inline constexpr int   kDsaQkHeadDim          = 256;
inline constexpr int   kDsaVHeadDim           = 256;
inline constexpr int   kDsaIndexNumHeads      = 64;
inline constexpr int   kDsaIndexHeadDim       = 128;
inline constexpr int   kDsaIndexTopK          = 2048;
inline constexpr double kDsaRopeTheta         = 8000000.0;
inline constexpr int   kVhaRank               = 16;

// ---- Stable Multi-Router Latent MoE (SiTU-GLU) ----
inline constexpr int   kNumRoutedExperts      = 512;
inline constexpr int   kTopKExperts           = 10;
inline constexpr int   kNumExpertsPerTok      = kTopKExperts;
inline constexpr int   kNumSharedExperts      = 1;
inline constexpr int   kMoeLatentSize         = 2048;     // 16 x 128
inline constexpr int   kLatentDim             = kMoeLatentSize;
inline constexpr int   kMoeInterSize          = 2048;     // 16 x 128
inline constexpr int   kNumMoeLayers          = kNumMoeBackboneLayers; // 42
inline constexpr int   kTotalBackboneExperts  = kNumMoeBackboneLayers * kNumRoutedExperts; // 21,504
inline constexpr int   kTotalExpertsWithEcho  = (kNumMoeBackboneLayers + kNumDsparkLayers) * kNumRoutedExperts; // 23,040
inline constexpr int   kTotalExpertsWithDspark= kTotalExpertsWithEcho;
inline constexpr float kSituBeta              = 4.0f;
inline constexpr float kSituLinearBeta        = 25.0f;

// ---- Exact WHT128-HLUT4 Tile & Expert Byte Geometry (2048 x 2048 latent experts) ----
inline constexpr int      kWhtBlock             = 128;
inline constexpr int      kTileDim              = 32;
inline constexpr int      kHlut4TileBytes       = 528;     // 512 B q_nib4[128] + 16 B scales_f16[8] = 4.125 bpw
inline constexpr int      kHlut4SuperBytes      = 8448;    // 16 x 528 B = 33 x 256 B
inline constexpr size_t   kExpProjTiles         = (kMoeInterSize / 32) * (kMoeLatentSize / 32); // 64 x 64 = 4,096
inline constexpr size_t   kExpProjSuperblocks   = (kMoeInterSize / 128) * (kMoeLatentSize / 128); // 16 x 16 = 256
inline constexpr size_t   kExpTp2ProjSupers     = ((kMoeInterSize / 2) / 128) * (kMoeLatentSize / 128); // 8 x 16 = 128
inline constexpr uint64_t kHlut4ProjBytes       = kExpProjTiles * uint64_t(kHlut4TileBytes);    // 2,162,688 B
inline constexpr uint64_t kHlut4ExpertBytes     = 3ULL * kHlut4ProjBytes;                       // 6,488,064 B = 6.1875 MiB
inline constexpr uint64_t kHlut4Tp2CardExpBytes = 3ULL * kExpTp2ProjSupers * uint64_t(kHlut4SuperBytes); // 3,244,032 B

static_assert(kHlut4ExpertBytes == 6'488'064ULL, "ERNIE-5.5 2048x2048 WHT128-HLUT4 expert must be 6,488,064 B (6.1875 MiB)");
static_assert(kHlut4Tp2CardExpBytes * 2ULL == kHlut4ExpertBytes, "TP2 P150A shard sum must match full expert");
static_assert(uint64_t(kTotalBackboneExperts) * kHlut4ExpertBytes == 139'519'328'256ULL,
              "21,504 backbone experts in WHT128-HLUT4 must equal exact 139,519,328,256 B (129.9375 GiB)");

// ---- Exact compress_ratios schedule (L00..L45): 37 HCA (128) + 6 Backbone DSA (-2) + 3 DSpark SWA (128) ----
inline constexpr std::array<int, kTotalLayersWithEcho> kDefaultCompressRatios = {
    128, 128, 128, 128, 128, 128, 128, -2,   // L00..L07
    128, 128, 128, 128, 128, 128, -2,        // L08..L14
    128, 128, 128, 128, 128, 128, -2,        // L15..L21
    128, 128, 128, 128, 128, 128, -2,        // L22..L28
    128, 128, 128, 128, 128, 128, -2,        // L29..L35
    128, 128, 128, 128, 128, 128, -2,        // L36..L42
    128, 128, 128                            // L43..L45 (3L MoE DSpark SWA-only)
};

inline constexpr bool is_dsa_layer_default(int layer_idx) noexcept {
    return layer_idx >= 0 && layer_idx < kTotalLayersWithEcho && kDefaultCompressRatios[size_t(layer_idx)] <= 0;
}

// Deterministic per-layer WHT128 sign seeds (16 blocks of 128 -> 32 uint64_t = 2,048 bits)
inline uint64_t layer_wht_seed_in(int layer_id) noexcept {
    return 0x45524E35355F494EULL ^ (uint64_t(layer_id + 1) * 0x9E3779B97F4A7C15ULL);
}
inline uint64_t layer_wht_seed_mid(int layer_id) noexcept {
    return 0x45524E35355F4D44ULL ^ (uint64_t(layer_id + 1) * 0xBF58476D1CE4E5B9ULL);
}

// ---- Runtime Model Config (loaded dynamically from config.json for zero-code-edit Nov 25 hot-swap!) ----
struct Ernie55ModelConfig {
    std::string model_type = "ernie5_5_moe";
    int vocab_size = kVocabSize;
    int hidden_size = kHiddenSize;
    int num_hidden_layers = kNumHiddenLayers;
    int num_nextn_predict_layers = kNumEchoLayers;
    int first_k_dense_replace = kFirstKDenseReplace;
    int intermediate_size = kDenseInterSize;
    int moe_latent_size = kMoeLatentSize;
    int moe_intermediate_size = kMoeInterSize;
    int n_routed_experts = kNumRoutedExperts;
    int num_experts_per_tok = kTopKExperts;
    int n_shared_experts = kNumSharedExperts;
    int hc_mult = kHcMult;
    int hc_sinkhorn_iters = kHcSinkhornIters;
    float hc_eps = kHcEps;
    float rms_norm_eps = kRmsNormEps;
    float situ_beta = kSituBeta;
    float situ_linear_beta = kSituLinearBeta;
    int sliding_window = kSlidingWindow;
    int index_topk = kDsaIndexTopK;
    int max_position_embeddings = kMaxPositionEmbeddings;
    int bos_token_id = 1;
    int pad_token_id = 0;
    std::vector<int> eos_token_ids = {2, 8};
    double rope_theta = kDsaRopeTheta;
    double compress_rope_theta = kHcaCompressRopeTheta;
    double yarn_factor = kHcaYarnFactor;
    int yarn_orig_max_pos = kHcaYarnOrigMaxPos;
    double yarn_beta_fast = kHcaYarnBetaFast;
    double yarn_beta_slow = kHcaYarnBetaSlow;
    std::vector<int> compress_ratios{kDefaultCompressRatios.begin(), kDefaultCompressRatios.end()};
    bool source_is_fp8 = false;
    int fp8_block_size = 128;
    int fp8_block_n = 128;
    int fp8_block_k = 128;
    std::string fp8_scale_fmt;

    // Bundled 3-Layer MoE DSpark probabilistic draft config (ERNIE-5.5-Flash-Preview Apache-2.0)
    std::vector<int> dspark_target_layer_ids; // bundled under model.mtp.* (e.g. [40, 41, 42])
    int  dspark_noise_token_id  = -1;         // 1024 in Preview-BF16
    int  dspark_markov_rank     = 0;          // 256 in Preview-BF16
    int  dspark_block_size      = 0;          // 7 in Preview-BF16
    bool enable_confidence_head = false;
    bool enable_flashindexer    = true;       // auto-enabled when flashindexer.safetensors is present

    bool has_bundled_dspark() const noexcept {
        return !dspark_target_layer_ids.empty() && num_hidden_layers > int(dspark_target_layer_ids.size());
    }
    int num_dspark_layers() const noexcept {
        return has_bundled_dspark() ? int(dspark_target_layer_ids.size()) : 0;
    }

    bool is_dsa_layer(int l) const noexcept {
        return l >= 0 && l < int(compress_ratios.size()) && compress_ratios[size_t(l)] <= 0;
    }

    int count_hca_backbone() const noexcept {
        int c = 0;
        for (int l = 0; l < num_hidden_layers; ++l) if (!is_dsa_layer(l)) ++c;
        return c;
    }
    int count_dsa_backbone() const noexcept {
        int c = 0;
        for (int l = 0; l < num_hidden_layers; ++l) if (is_dsa_layer(l)) ++c;
        return c;
    }

    static Ernie55ModelConfig load_json(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Cannot open config.json: " + path);
        std::ostringstream ss;
        ss << ifs.rdbuf();
        JsonValue root = JsonParser::parse(ss.str());

        Ernie55ModelConfig c;
        if (root.has("model_type")) c.model_type = root.get("model_type").as_string();
        if (root.has("vocab_size")) c.vocab_size = int(root.get("vocab_size").as_int());
        if (root.has("hidden_size")) c.hidden_size = int(root.get("hidden_size").as_int());
        if (root.has("num_hidden_layers")) c.num_hidden_layers = int(root.get("num_hidden_layers").as_int());
        if (root.has("num_nextn_predict_layers")) c.num_nextn_predict_layers = int(root.get("num_nextn_predict_layers").as_int());
        if (root.has("first_k_dense_replace")) c.first_k_dense_replace = int(root.get("first_k_dense_replace").as_int());
        if (root.has("intermediate_size")) c.intermediate_size = int(root.get("intermediate_size").as_int());
        if (root.has("moe_latent_size")) c.moe_latent_size = int(root.get("moe_latent_size").as_int());
        if (root.has("moe_intermediate_size")) c.moe_intermediate_size = int(root.get("moe_intermediate_size").as_int());
        if (root.has("n_routed_experts")) c.n_routed_experts = int(root.get("n_routed_experts").as_int());
        if (root.has("num_experts_per_tok")) c.num_experts_per_tok = int(root.get("num_experts_per_tok").as_int());
        if (root.has("n_shared_experts")) c.n_shared_experts = int(root.get("n_shared_experts").as_int());
        if (root.has("hc_mult")) c.hc_mult = int(root.get("hc_mult").as_int());
        if (root.has("hc_sinkhorn_iters")) c.hc_sinkhorn_iters = int(root.get("hc_sinkhorn_iters").as_int());
        if (root.has("hc_eps")) c.hc_eps = float(root.get("hc_eps").as_double());
        if (root.has("rms_norm_eps")) c.rms_norm_eps = float(root.get("rms_norm_eps").as_double());
        if (root.has("activation_situ_beta")) c.situ_beta = float(root.get("activation_situ_beta").as_double());
        if (root.has("activation_situ_linear_beta")) c.situ_linear_beta = float(root.get("activation_situ_linear_beta").as_double());
        if (root.has("sliding_window")) c.sliding_window = int(root.get("sliding_window").as_int());
        if (root.has("index_topk")) c.index_topk = int(root.get("index_topk").as_int());
        if (root.has("max_position_embeddings")) c.max_position_embeddings = int(root.get("max_position_embeddings").as_int());
        if (root.has("bos_token_id")) c.bos_token_id = int(root.get("bos_token_id").as_int());
        if (root.has("pad_token_id")) c.pad_token_id = int(root.get("pad_token_id").as_int());
        if (root.has("rope_theta")) c.rope_theta = root.get("rope_theta").as_double();
        if (root.has("compress_rope_theta")) c.compress_rope_theta = root.get("compress_rope_theta").as_double();
        if (root.has("rope_scaling") && root.get("rope_scaling").is_object()) {
            const auto& rs = root.get("rope_scaling");
            if (rs.has("factor")) c.yarn_factor = rs.get("factor").as_double();
            if (rs.has("original_max_position_embeddings")) c.yarn_orig_max_pos = int(rs.get("original_max_position_embeddings").as_int());
            if (rs.has("beta_fast")) c.yarn_beta_fast = rs.get("beta_fast").as_double();
            if (rs.has("beta_slow")) c.yarn_beta_slow = rs.get("beta_slow").as_double();
        }
        if (root.has("compress_ratios") && root.get("compress_ratios").is_array()) {
            c.compress_ratios.clear();
            for (const auto& v : root.get("compress_ratios").arr_val) c.compress_ratios.push_back(int(v.as_int()));
        }
        if (root.has("quantization_config") && root.get("quantization_config").is_object()) {
            const auto& qc = root.get("quantization_config");
            if (qc.has("quant_method") && qc.get("quant_method").as_string() == "fp8") {
                c.source_is_fp8 = true;
            }
            if (qc.has("scale_fmt") && qc.get("scale_fmt").is_string()) {
                c.fp8_scale_fmt = qc.get("scale_fmt").as_string();
            }
            if (qc.has("weight_block_size") && qc.get("weight_block_size").is_array() &&
                qc.get("weight_block_size").arr_val.size() >= 2) {
                c.fp8_block_n = int(qc.get("weight_block_size").arr_val[0].as_int());
                c.fp8_block_k = int(qc.get("weight_block_size").arr_val[1].as_int());
            }
        }
        // Bundled DSpark fields (top-level or nested under "dspark_draft")
        if (root.has("dspark_target_layer_ids") && root.get("dspark_target_layer_ids").is_array()) {
            c.dspark_target_layer_ids.clear();
            for (const auto& v : root.get("dspark_target_layer_ids").arr_val)
                c.dspark_target_layer_ids.push_back(int(v.as_int()));
        }
        if (root.has("dspark_noise_token_id")) c.dspark_noise_token_id = int(root.get("dspark_noise_token_id").as_int());
        if (root.has("mask_token_id") && c.dspark_noise_token_id < 0) c.dspark_noise_token_id = int(root.get("mask_token_id").as_int());
        if (root.has("dspark_markov_rank")) c.dspark_markov_rank = int(root.get("dspark_markov_rank").as_int());
        if (root.has("dspark_block_size")) c.dspark_block_size = int(root.get("dspark_block_size").as_int());
        bool explicit_conf = false;
        if (root.has("enable_confidence_head")) { c.enable_confidence_head = root.get("enable_confidence_head").as_bool(); explicit_conf = true; }
        if (root.has("enable_flashindexer")) c.enable_flashindexer = root.get("enable_flashindexer").as_bool();
        for (const char* dkey : {"dspark_draft", "dspark_config", "mtp_config"}) {
            if (root.has(dkey) && root.get(dkey).is_object()) {
                const auto& dd = root.get(dkey);
                for (const char* lkey : {"feature_layer_ids", "target_layer_ids"}) {
                    if (dd.has(lkey) && dd.get(lkey).is_array()) {
                        c.dspark_target_layer_ids.clear();
                        for (const auto& v : dd.get(lkey).arr_val)
                            c.dspark_target_layer_ids.push_back(int(v.as_int()));
                    }
                }
                if (dd.has("mask_token_id"))  c.dspark_noise_token_id = int(dd.get("mask_token_id").as_int());
                if (dd.has("noise_token_id")) c.dspark_noise_token_id = int(dd.get("noise_token_id").as_int());
                if (dd.has("markov_rank"))    c.dspark_markov_rank = int(dd.get("markov_rank").as_int());
                if (dd.has("block_size"))     c.dspark_block_size = int(dd.get("block_size").as_int());
                if (dd.has("enable_confidence_head")) { c.enable_confidence_head = dd.get("enable_confidence_head").as_bool(); explicit_conf = true; }
            }
        }
        if (c.has_bundled_dspark() && !explicit_conf) c.enable_confidence_head = true;
        return c;
    }

    void validate_compat() const {
        if (model_type != "ernie5_5_moe") throw std::runtime_error("Unsupported model_type: " + model_type);
        if (hidden_size != kHiddenSize) throw std::runtime_error("hidden_size mismatch");
        if (moe_latent_size != kMoeLatentSize || moe_intermediate_size != kMoeInterSize)
            throw std::runtime_error("MoE latent/inter size mismatch (expected 2048x2048)");
        if (n_routed_experts != kNumRoutedExperts || num_experts_per_tok != kTopKExperts)
            throw std::runtime_error("MoE expert count / topk mismatch");
        if (hc_mult != kHcMult) throw std::runtime_error("mHC hc_mult mismatch");
    }
};

// ---- Dynamic Multi-Accelerator Topology Plan (192G / 224G / 240G, Zero Code Edits!) ----
struct Mi50TpRankSpec {
    int tp_rank = 0;
    int hip_dev = 0;
    std::string bdf;
    int hbm_mib = 32752;
    int expert_begin = 0;
    int expert_end = 0;
    int count() const noexcept { return expert_end - expert_begin; }
};

struct Mi50EpWorkerSpec {
    int ep_id = 0;
    int hip_dev = 0;
    std::string bdf;
    int hbm_mib = 16368;
    int cpu_core = -1;
    int expert_begin = 0;
    int expert_end = 0;
    int echo_begin = 0;
    int echo_end = 0;
    int count() const noexcept { return expert_end - expert_begin; }
    int echo_count() const noexcept { return echo_end - echo_begin; }
};

struct P150GroupSpec {
    int group_id = 0;
    std::string mode = "tp2";  // "tp2" (2x P150A I_shard=1024) or "ep3" (3x P150A unsharded I=2048)
    int num_cards = 2;
    std::string shm_name = "/ernie55_p150_ipc";
    std::string cpu_cores = "25,27";
    int expert_begin = 314;
    int expert_end = 512;
    int trace_region_mib = 256;
    std::vector<int> card_splits;
    int count() const noexcept { return expert_end - expert_begin; }
};

enum class ExpertBackend : uint8_t {
    kMi50Tp = 0,
    kMi50Ep = 1,
    kP150   = 2,
};

struct ExpertRouteEntry {
    ExpertBackend backend = ExpertBackend::kMi50Tp;
    int device_idx = 0;   // tp_rank, ep_id, or p150 group_id
    int local_slot = 0;   // [0 .. count)
};

struct Ernie55TopologyPlan {
    std::string name;
    int total_vram_gib = 192;
    uint64_t ddr4_expert_spill_bytes = 0;
    std::string hip_visible_devices = "0,2,3,1,4";
    std::vector<Mi50TpRankSpec> mi50_tp_ranks;
    std::vector<Mi50EpWorkerSpec> mi50_ep_workers;
    std::vector<P150GroupSpec> p150_groups;
    std::array<ExpertRouteEntry, kNumRoutedExperts> bb_route{};
    std::array<ExpertRouteEntry, kNumRoutedExperts> echo_route{};

    static Ernie55TopologyPlan load_json(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Cannot open topology config: " + path);
        std::ostringstream ss;
        ss << ifs.rdbuf();
        JsonValue root = JsonParser::parse(ss.str());

        Ernie55TopologyPlan p;
        if (root.has("name")) p.name = root.get("name").as_string();
        if (root.has("total_vram_gib")) p.total_vram_gib = int(root.get("total_vram_gib").as_int());
        if (root.has("ddr4_expert_spill_bytes")) p.ddr4_expert_spill_bytes = uint64_t(root.get("ddr4_expert_spill_bytes").as_int());
        if (root.has("hip_visible_devices")) p.hip_visible_devices = root.get("hip_visible_devices").as_string();

        if (root.has("mi50_tp_ranks")) {
            for (const auto& j : root.get("mi50_tp_ranks").arr_val) {
                Mi50TpRankSpec s;
                s.tp_rank = int(j.get("tp_rank").as_int());
                s.hip_dev = int(j.get("hip_dev").as_int());
                if (j.has("bdf")) s.bdf = j.get("bdf").as_string();
                if (j.has("hbm_mib")) s.hbm_mib = int(j.get("hbm_mib").as_int());
                s.expert_begin = int(j.get("expert_begin").as_int());
                s.expert_end = int(j.get("expert_end").as_int());
                p.mi50_tp_ranks.push_back(std::move(s));
            }
        }
        if (root.has("mi50_ep_workers")) {
            for (const auto& j : root.get("mi50_ep_workers").arr_val) {
                Mi50EpWorkerSpec s;
                s.ep_id = int(j.get("ep_id").as_int());
                s.hip_dev = int(j.get("hip_dev").as_int());
                if (j.has("bdf")) s.bdf = j.get("bdf").as_string();
                if (j.has("hbm_mib")) s.hbm_mib = int(j.get("hbm_mib").as_int());
                if (j.has("cpu_core")) s.cpu_core = int(j.get("cpu_core").as_int());
                s.expert_begin = int(j.get("expert_begin").as_int());
                s.expert_end = int(j.get("expert_end").as_int());
                if (j.has("echo_begin")) s.echo_begin = int(j.get("echo_begin").as_int());
                if (j.has("echo_end")) s.echo_end = int(j.get("echo_end").as_int());
                p.mi50_ep_workers.push_back(std::move(s));
            }
        }
        if (root.has("p150_groups")) {
            for (const auto& j : root.get("p150_groups").arr_val) {
                P150GroupSpec s;
                s.group_id = int(j.get("group_id").as_int());
                if (j.has("mode")) s.mode = j.get("mode").as_string();
                if (j.has("num_cards")) s.num_cards = int(j.get("num_cards").as_int());
                if (j.has("shm_name")) s.shm_name = j.get("shm_name").as_string();
                if (j.has("cpu_cores")) s.cpu_cores = j.get("cpu_cores").as_string();
                s.expert_begin = int(j.get("expert_begin").as_int());
                s.expert_end = int(j.get("expert_end").as_int());
                if (j.has("trace_region_mib")) s.trace_region_mib = int(j.get("trace_region_mib").as_int());
                if (j.has("card_splits")) {
                    for (const auto& sp : j.get("card_splits").arr_val) s.card_splits.push_back(int(sp.as_int()));
                }
                p.p150_groups.push_back(std::move(s));
            }
        }
        p.finalize_and_verify();
        return p;
    }

    void finalize_and_verify() {
        if (ddr4_expert_spill_bytes != 0ULL) {
            throw std::runtime_error("ERNIE-5.5 topology must enforce 0 B CPU DDR4 expert spill!");
        }
        std::array<int, kNumRoutedExperts> bb_seen{};
        std::array<int, kNumRoutedExperts> echo_seen{};
        for (size_t i = 0; i < mi50_tp_ranks.size(); ++i) {
            const auto& r = mi50_tp_ranks[i];
            for (int e = r.expert_begin; e < r.expert_end; ++e) {
                if (e < 0 || e >= kNumRoutedExperts || bb_seen[size_t(e)]++)
                    throw std::runtime_error("Invalid/overlapping MI50 TP expert range");
                bb_route[size_t(e)] = {ExpertBackend::kMi50Tp, int(i), e - r.expert_begin};
            }
        }
        for (size_t i = 0; i < mi50_ep_workers.size(); ++i) {
            const auto& w = mi50_ep_workers[i];
            for (int e = w.expert_begin; e < w.expert_end; ++e) {
                if (e < 0 || e >= kNumRoutedExperts || bb_seen[size_t(e)]++)
                    throw std::runtime_error("Invalid/overlapping MI50 EP expert range");
                bb_route[size_t(e)] = {ExpertBackend::kMi50Ep, int(i), e - w.expert_begin};
            }
            for (int e = w.echo_begin; e < w.echo_end; ++e) {
                if (e < 0 || e >= kNumRoutedExperts || echo_seen[size_t(e)]++)
                    throw std::runtime_error("Invalid/overlapping ECHO expert range");
                echo_route[size_t(e)] = {ExpertBackend::kMi50Ep, int(i), e - w.echo_begin};
            }
        }
        for (size_t i = 0; i < p150_groups.size(); ++i) {
            const auto& g = p150_groups[i];
            for (int e = g.expert_begin; e < g.expert_end; ++e) {
                if (e < 0 || e >= kNumRoutedExperts || bb_seen[size_t(e)]++)
                    throw std::runtime_error("Invalid/overlapping P150A expert range");
                bb_route[size_t(e)] = {ExpertBackend::kP150, int(i), e - g.expert_begin};
            }
            if (!g.card_splits.empty()) {
                int sum = 0;
                for (int s : g.card_splits) sum += s;
                if (sum != g.count()) throw std::runtime_error("P150 card_splits sum != group count");
            }
        }
        int echo_total = 0;
        for (int e = 0; e < kNumRoutedExperts; ++e) {
            if (bb_seen[size_t(e)] != 1) throw std::runtime_error("Topology does not cover all 512 experts/layer");
            echo_total += echo_seen[size_t(e)];
        }
        if (echo_total == 0) {
            // Uniform 45-layer split (L01..L42 backbone + L43..L45 3L MoE DSpark share bb_route!)
            echo_route = bb_route;
        } else {
            for (int e = 0; e < kNumRoutedExperts; ++e) {
                if (echo_seen[size_t(e)] != 1) throw std::runtime_error("Topology does not cover all 512 ECHO experts");
            }
        }
    }

    double estimate_mi50_expert_gib(int count_per_layer, int echo_count) const noexcept {
        const uint64_t n_layers = (echo_count == 0) ? uint64_t(kNumMoeTotalWithDspark) : uint64_t(kNumMoeBackboneLayers);
        return double(uint64_t(count_per_layer) * n_layers + uint64_t(echo_count)) *
               double(kHlut4ExpertBytes) / double(1ULL << 30);
    }

    double estimate_p150_card_gib(const P150GroupSpec& g) const noexcept {
        const uint64_t n_layers = uint64_t(kNumMoeTotalWithDspark);
        if (g.mode == "tp2") {
            return double(uint64_t(g.count()) * n_layers) *
                   double(kHlut4Tp2CardExpBytes) / double(1ULL << 30);
        }
        int max_per_card = (g.count() + g.num_cards - 1) / g.num_cards;
        for (int s : g.card_splits) if (s > max_per_card) max_per_card = s;
        return double(uint64_t(max_per_card) * n_layers) *
               double(kHlut4ExpertBytes) / double(1ULL << 30);
    }
};

} // namespace ernie55
