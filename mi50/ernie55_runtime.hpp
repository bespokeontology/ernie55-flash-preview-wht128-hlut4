// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 / gfx906 HIP / Blackhole P150A Multi-Accelerator Runtime for baidu/ERNIE-5.5-Flash (ernie5_5_moe)
// 100% uniform WHT128-HLUT4 across all 21,504 backbone + 512 ECHO experts (0 B CPU DDR4 spill!).
// Supports dynamic topologies (192G / 224G / 240G), full 1M lossless BF16 HCA+MLA KV cache,
// BOTH ECHO 2-step greedy MTP draft (Base) AND bundled DSpark probabilistic block draft (Nov 25 post-trained).

#include "ernie55_config.hpp"
#include "ernie55_dspark.hpp"
#include "ernie55_hlut4_pack.hpp"
#include "ernie55_tokenizer.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ernie55 {

struct Ernie55RuntimeOptions {
    std::string weight_root   = "./checkpoints/active";
    std::string topology_path = "./config/topology_192g.json";
    std::string shm_name      = "/ernie55_p150_ipc";
    std::string kv_stripe_dir = "/var/tmp/ernie55_kvcache"; // Physical nvme1n1 stripe dir (paired with nvme0n1!)
    int  max_kv               = kMaxPositionEmbeddings;     // 1,048,576 full 1M context
    int  prefill_chunk        = 1024;                       // 1024-tok chunks amortize P150A + MI50 MoE while keeping 512-row GEMM invariance!
    int  echo_draft_steps     = 2;                          // ECHO MTP 2-step greedy draft (do_sample=false)
    int  dspark_block_size    = 5;                          // Default 3L MoE DSpark draft K (tuned via GATE-4 K=2..7 sweep!)
    float dspark_conf_tau     = 0.0f;                       // Confidence early-stop threshold (0.0 = disabled / fixed K)
    bool enable_flashindexer  = true;                       // Top-8/64 dynamic DSA head routing (Preview-BF16)
    std::string draft_mode    = "auto";                     // "auto", "echo", "dspark", "none"
    bool use_p150             = true;
    bool keep_p150            = true;
    bool load_echo            = true;
};

struct Ernie55ProfileStats {
    uint64_t calls             = 0;
    uint64_t dspark_draft_calls = 0;
    double   mhc_ms            = 0.0;
    double   hca_ms            = 0.0;
    double   hca_proj_ms       = 0.0;
    double   hca_attn_ms       = 0.0;
    double   dsa_ms            = 0.0;
    double   dsa_proj_ms       = 0.0;
    double   dsa_fi_topk_ms    = 0.0;
    double   dsa_mla_ms        = 0.0;
    double   router_ms         = 0.0;
    double   mi50_moe_ms       = 0.0;
    double   p150_wait_ms      = 0.0;
    double   shared_ms         = 0.0;
    double   tp4_comm_ms       = 0.0;
    double   dspark_draft_ms   = 0.0;
    double   head_ms           = 0.0;
    double   sample_cpu_ms     = 0.0;
    uint64_t mi50_routes       = 0;
    uint64_t p150_routes       = 0;
    uint64_t p150_l1_routes    = 0;
    uint64_t hgemm_experts     = 0;
    uint64_t gemv_experts      = 0;
};

struct KvBlkStats {
    int32_t  seq_len    = 0;
    uint64_t bytes      = 0;
    double   io_ms      = 0.0;
    double   crc_ms     = 0.0;
    double   dma_ms     = 0.0;
    double   total_ms   = 0.0;
    double   gb_s       = 0.0;
    double   tok_s      = 0.0;
    bool     dual_nvme  = false;
};

struct GenerateStats {
    int    prompt_tokens        = 0;
    int    cached_prefix_tokens = 0;
    int    generated_tokens     = 0;
    double ttft_ms              = 0.0;
    double decode_ms            = 0.0;
    double decode_tok_s         = 0.0;
    uint64_t spec_steps         = 0;
    uint64_t spec_drafted       = 0;
    uint64_t spec_accepted      = 0;
    DraftMethod draft_used      = DraftMethod::kNone;
    std::string finish_reason   = "stop";
};

using TokenCallback = std::function<bool(int32_t tok_id, const std::string& piece)>;
using PrefillProgressCallback = std::function<void(size_t prompt_total, size_t cached_prefix, size_t fresh_done, double chunk_ms)>;

class Ernie55Runtime {
public:
    Ernie55Runtime();
    ~Ernie55Runtime();
    Ernie55Runtime(const Ernie55Runtime&) = delete;
    Ernie55Runtime& operator=(const Ernie55Runtime&) = delete;

    void init(const Ernie55RuntimeOptions& opt);
    void shutdown() noexcept;

    bool initialized() const noexcept;
    const Ernie55ModelConfig& model_config() const noexcept;
    const Ernie55TopologyPlan& topology() const noexcept;
    const Ernie55Tokenizer& tokenizer() const noexcept;
    DraftMethod active_draft_method() const noexcept;
    int max_kv() const noexcept;
    int current_seq_len() const noexcept;
    bool p150_attached() const noexcept;

    void reset_kv();
    void wipe_all_kv_vram();
    void capture_prompt_snapshot();
    void set_echo_draft_steps(int steps) noexcept;
    void set_dspark_block_size(int K) noexcept;
    void set_dspark_conf_tau(float tau) noexcept;
    void set_flashindexer(bool enabled) noexcept;
    void set_draft_mode(DraftMethod mode) noexcept;
    void set_prefill_progress_callback(PrefillProgressCallback cb);
    void set_profiling(bool enabled) noexcept;
    void reset_profile_stats() noexcept;
    Ernie55ProfileStats profile_stats() const noexcept;
    void set_router_capture(std::vector<int16_t>* out_raw_top10_43xMx10, int stride_tokens) noexcept;

    // Forward prompt tokens and return final token logits [vocab_size] (after MultiMax)
    void prefill_tokens(const std::vector<int32_t>& tok_ids, std::vector<float>& out_last_logits);
    void prefill_tokens_chunked(const std::vector<int32_t>& tok_ids, int chunk_size, std::vector<float>& out_last_logits);
    void prefill_tokens_warm(const std::vector<int32_t>& tok_ids, std::vector<float>& out_last_logits,
                             int* out_reused_prefix = nullptr);

    // Raw active KV VRAM byte dump (all 43 backbone + 3 DSpark layers) for byte-exact memcmp == 0 verification
    std::vector<uint8_t> dump_kv_raw_bytes() const;

    // Direct single-expert live Blackhole P150A parity probe (local_slot in [0 .. p150_count))
    bool probe_p150_expert_parity(int layer_idx, int local_slot, const float* x_rot_f32, int M,
                                  std::vector<float>& out_y_f32);

    // Forward M tokens starting at pos0 (for decode M=1 or speculative verify M=K+1)
    // Writes logits [M, vocab_size] after MultiMax.
    void forward_tokens(const int32_t* tok_ids, int M, int32_t pos0,
                        std::vector<float>& out_logits_MxV,
                        bool capture_last_res_for_echo = true,
                        bool capture_dspark_features = false);

    // Full streaming generation with automatic ECHO (2-step greedy) or DSpark (probabilistic) speculative decoding
    void generate_stream(
        const std::vector<int32_t>& prompt_ids,
        int max_new_tokens,
        const SamplingParams& sp,
        const std::vector<int32_t>& extra_stop_ids,
        const std::atomic<bool>* cancel_flag,
        const TokenCallback& on_token,
        GenerateStats& out_stats);

    // Striped Dual-NVMe O_DIRECT Cold KV Snapshot (.kvblk with hardware SSE4.2 CRC32C)
    void save_kvblk(const std::string& path) const;
    void load_kvblk(const std::string& path);
    KvBlkStats save_kvblk_ex(const std::string& path) const;
    KvBlkStats load_kvblk_ex(const std::string& path);

    // Inspect VRAM ledger & live metrics as JSON
    JsonValue metrics_json() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ernie55
