// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 Exact Speculative Decoding Engine for baidu/ERNIE-5.5-Flash:
//   1. ECHO (L43 MTP 2-step greedy draft, do_sample=false) — active on ERNIE-5.5-Flash-Base today!
//   2. DSpark Probabilistic Block Draft (draft_sample_method="probabilistic") — auto-activated on Nov 25
//      post-trained ERNIE-5.5-Flash (bundled under model.mtp.*):
//        - Feature combine: main_norm(main_proj(concat(target_layer_hiddens)))
//        - Context KV projection: context_wkv_proj -> kv_norm -> SWA KV cache (plain RoPE at rope_theta)
//        - Parallel block forward: B = dspark_block_size positions [anchor, mask, ..., mask] through SWA mHC layers
//        - Sequential Low-Rank Markov Head (markov_w1^T [V, R] + markov_w2 [V, R])
//        - MultiMax postprocess_draft_logits AFTER Markov bias (preserving -inf masked entries!)
//        - Optional DSparkConfidenceHead early stopping: sigmoid(proj(concat(head_hidden, markov_embed)))
//        - Exact Leviathan/Chen lossless probabilistic rejection sampling & residual bonus token!

#include "ernie55_config.hpp"
#include "ernie55_hlut4_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace ernie55 {

enum class DraftMethod : uint8_t {
    kNone   = 0,
    kEcho   = 1,  // ECHO MTP 2-step greedy draft (do_sample=false)
    kDSpark = 2,  // Bundled DSpark probabilistic block draft (Nov 25 post-trained)
};

enum class DraftSampleMethod : uint8_t {
    kGreedy        = 0,
    kProbabilistic = 1,
};

struct SamplingParams {
    float temperature        = 0.0f;   // 0.0 => greedy (do_sample=false)
    float top_p              = 1.0f;
    int   top_k              = 0;      // <= 0 => disabled
    float repetition_penalty = 1.0f;
    uint64_t seed            = 42ULL;
    DraftSampleMethod draft_sample_method = DraftSampleMethod::kProbabilistic;
};

// Compact sparse probability distribution for fast O(k) probabilistic speculative decoding (k << V)
struct SparseProbs {
    // Sorted descending by probability; sum(prob) == 1.0
    std::vector<std::pair<int32_t, float>> items;

    float prob_of(int32_t tok) const noexcept {
        for (const auto& kv : items) {
            if (kv.first == tok) return kv.second;
        }
        return 0.0f;
    }
};

inline int32_t sample_from_sparse_probs(const SparseProbs& sp, std::mt19937_64& rng) noexcept {
    if (sp.items.empty()) return 0;
    if (sp.items.size() == 1u) return sp.items[0].first;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const double u = ud(rng);
    double cdf = 0.0;
    for (const auto& kv : sp.items) {
        cdf += double(kv.second);
        if (u <= cdf) return kv.first;
    }
    return sp.items.back().first;
}

// Fast exact threshold-pruned sparse probability distribution from logits[0..V):
// Prunes entries with (logits[i] - max_l)/T < -22.0f (exp(-22) = 2.8e-10 < 1e-9 cutoff!),
// avoiding 201K scalar std::exp + 805 KiB dense vector allocs/memsets while staying 100% INV-1c exact!
inline void logits_to_sparse_probs_fast(
    const float* logits, int V,
    float temperature, int top_k, float top_p,
    SparseProbs& out)
{
    out.items.clear();
    if (V <= 0) return;

    if (temperature <= 1e-6f) {
        int best = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < V; ++i) {
            const float v = logits[i];
            if (v > best_v) { best_v = v; best = i; }
        }
        out.items.emplace_back(best, 1.0f);
        return;
    }

    float max_l = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < V; ++i) {
        const float v = logits[i];
        if (std::isfinite(v) && v > max_l) max_l = v;
    }
    if (!std::isfinite(max_l)) {
        out.items.emplace_back(0, 1.0f);
        return;
    }

    const float inv_t = 1.0f / temperature;
    // Any token below max_l - 22*T has unnormalized exp < 2.8e-10 (< 1e-9 pruning threshold!)
    const float cut = max_l - 22.0f * temperature;
    out.items.reserve(256);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        const float v = logits[i];
        if (v > cut && std::isfinite(v)) {
            const float e = std::exp((v - max_l) * inv_t);
            sum += double(e);
            out.items.emplace_back(i, e);
        }
    }
    if (out.items.empty() || sum <= 0.0) {
        int best = int(std::max_element(logits, logits + V) - logits);
        out.items.clear();
        out.items.emplace_back(best, 1.0f);
        return;
    }

    const float inv_sum = float(1.0 / sum);
    for (auto& kv : out.items) kv.second *= inv_sum;

    const bool do_topk = (top_k > 0 && top_k < V);
    const bool do_topp = (top_p > 0.0f && top_p < 1.0f);
    if (!do_topk && !do_topp) {
        return;
    }

    // Drop negligible entries < 1e-9f (identical to legacy logits_to_probs_filtered!)
    size_t w = 0;
    for (size_t i = 0; i < out.items.size(); ++i) {
        if (out.items[i].second > 1e-9f) out.items[w++] = out.items[i];
    }
    out.items.resize(w);
    if (out.items.empty()) {
        int best = int(std::max_element(logits, logits + V) - logits);
        out.items.emplace_back(best, 1.0f);
        return;
    }

    auto cmp = [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) {
        return a.second > b.second;
    };
    if (do_topk && int(out.items.size()) > top_k) {
        std::nth_element(out.items.begin(), out.items.begin() + top_k, out.items.end(), cmp);
        out.items.resize(size_t(top_k));
    }
    std::sort(out.items.begin(), out.items.end(), cmp);

    size_t keep_n = out.items.size();
    if (do_topp) {
        double cum = 0.0;
        keep_n = 0;
        for (size_t k = 0; k < out.items.size(); ++k) {
            cum += double(out.items[k].second);
            ++keep_n;
            if (cum >= double(top_p)) break;
        }
    }
    out.items.resize(keep_n);
    double nsum = 0.0;
    for (const auto& kv : out.items) nsum += double(kv.second);
    const float inv_n = (nsum > 0.0) ? float(1.0 / nsum) : 0.0f;
    for (auto& kv : out.items) kv.second *= inv_n;
}

// Build normalized SparseProbs directly from GPU-extracted (idx, exp_val) candidates + total_sum_exp!
inline void gpu_candidates_to_sparse_probs(
    const int32_t* cand_idx, const float* cand_exp, int n_cand, double total_sum_exp,
    int top_k, float top_p, SparseProbs& out)
{
    out.items.clear();
    if (n_cand <= 0 || total_sum_exp <= 0.0) {
        out.items.emplace_back(0, 1.0f);
        return;
    }
    const float inv_sum = float(1.0 / total_sum_exp);
    out.items.reserve(size_t(n_cand));
    for (int i = 0; i < n_cand; ++i) {
        const float p = cand_exp[i] * inv_sum;
        if (p > 1e-9f) out.items.emplace_back(cand_idx[i], p);
    }
    if (out.items.empty()) {
        out.items.emplace_back(cand_idx[0], 1.0f);
        return;
    }
    auto cmp = [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) {
        return a.second > b.second;
    };
    const bool do_topk = (top_k > 0 && int(out.items.size()) > top_k);
    if (do_topk) {
        std::nth_element(out.items.begin(), out.items.begin() + top_k, out.items.end(), cmp);
        out.items.resize(size_t(top_k));
    }
    std::sort(out.items.begin(), out.items.end(), cmp);

    size_t keep_n = out.items.size();
    if (top_p > 0.0f && top_p < 1.0f) {
        double cum = 0.0;
        keep_n = 0;
        for (size_t k = 0; k < out.items.size(); ++k) {
            cum += double(out.items[k].second);
            ++keep_n;
            if (cum >= double(top_p)) break;
        }
    }
    out.items.resize(keep_n);
    double nsum = 0.0;
    for (const auto& kv : out.items) nsum += double(kv.second);
    const float inv_n = (nsum > 0.0) ? float(1.0 / nsum) : 0.0f;
    for (auto& kv : out.items) kv.second *= inv_n;
}

// Compute normalized probability distribution p[0..V) from logits with temperature, top_k, top_p
inline void logits_to_probs_filtered(
    const float* logits, int V,
    float temperature, int top_k, float top_p,
    std::vector<float>& probs_out)
{
    probs_out.assign(size_t(V), 0.0f);
    if (V <= 0) return;
    SparseProbs sp;
    logits_to_sparse_probs_fast(logits, V, temperature, top_k, top_p, sp);
    for (const auto& kv : sp.items) {
        if (kv.first >= 0 && kv.first < V) probs_out[size_t(kv.first)] = kv.second;
    }
}

inline int32_t sample_from_probs(const std::vector<float>& probs, std::mt19937_64& rng) noexcept {
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const double u = ud(rng);
    double cdf = 0.0;
    int32_t last_pos = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > 0.0f) last_pos = int32_t(i);
        cdf += double(probs[i]);
        if (u <= cdf) return int32_t(i);
    }
    return last_pos;
}

// ============================================================================
// Exact DSpark Low-Rank Markov Head + Confidence Head + Post-Markov MultiMax
// Matching vLLM-ERNIE src/vllm_ernie/models/ernie5_5/nvidia/dspark.py
// ============================================================================
struct DSparkHeadsCpu {
    int vocab_size       = 0;
    int hidden_size      = 0;
    int markov_rank      = 0;
    int block_size       = 7;
    int noise_token_id   = 2;
    bool has_confidence  = false;
    float conf_stop_tau  = 0.35f;

    // markov_w1_T: [vocab_size, markov_rank] (transposed from ckpt [markov_rank, vocab_size]!)
    std::vector<float> markov_w1_T;
    // markov_w2:   [vocab_size, markov_rank] (row v dotted with m_emb gives Markov bias for vocab v)
    std::vector<float> markov_w2;
    // confidence_head.proj.weight: [hidden_size + markov_rank] (+ optional bias)
    std::vector<float> conf_proj_w;
    float conf_proj_b = 0.0f;

    // Shared target MultiMax parameters applied AFTER Markov bias!
    float multimax_ranges[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float multimax_ts[4]     = {0.0f, 0.0f, 0.0f, 0.0f};
    bool  has_multimax       = false;

    void markov_embed(int32_t prev_token_id, float* out_rank) const noexcept {
        if (prev_token_id < 0 || prev_token_id >= vocab_size || markov_rank <= 0) {
            std::fill(out_rank, out_rank + markov_rank, 0.0f);
            return;
        }
        const float* row = markov_w1_T.data() + size_t(prev_token_id) * size_t(markov_rank);
        std::memcpy(out_rank, row, size_t(markov_rank) * sizeof(float));
    }

    // Add low-rank Markov bias in-place to base_logits[0..V) and apply MultiMax (preserving -inf!)
    void add_markov_bias_and_multimax(const float* m_emb, float* logits_inout) const noexcept {
        const int V = vocab_size;
        const int R = markov_rank;
        #pragma omp parallel for schedule(static) if(V >= 8192)
        for (int v = 0; v < V; ++v) {
            float x = logits_inout[v];
            if (std::isinf(x) && x < 0.0f) continue;  // preserve -inf mask!
            if (R > 0 && !markov_w2.empty()) {
                const float* w2 = markov_w2.data() + size_t(v) * size_t(R);
                float bias = 0.0f;
                for (int r = 0; r < R; ++r) bias += w2[r] * m_emb[r];
                x += bias;
            }
            if (has_multimax) {
                x = ernie55_multimax_scalar(x, multimax_ranges, multimax_ts);
            }
            logits_inout[v] = x;
        }
    }

    // Exact DSparkConfidenceHead: sigmoid(w · [head_hidden, markov_embed] + b)
    float compute_confidence(const float* head_hidden, const float* m_emb) const noexcept {
        if (!has_confidence || int(conf_proj_w.size()) < hidden_size + markov_rank) return 1.0f;
        double acc = double(conf_proj_b);
        for (int d = 0; d < hidden_size; ++d) acc += double(conf_proj_w[size_t(d)]) * double(head_hidden[d]);
        for (int r = 0; r < markov_rank; ++r) acc += double(conf_proj_w[size_t(hidden_size + r)]) * double(m_emb[r]);
        return float(1.0 / (1.0 + std::exp(-acc)));
    }

    struct DraftProposal {
        std::vector<int32_t> draft_tokens;              // proposed [d_0 .. d_{K-1}], K <= block_size
        std::vector<std::vector<float>> draft_probs;    // legacy dense q_b[0..V) (used by propose_block unit tests)
        std::vector<SparseProbs> draft_sparse;          // Wave-8 compact sparse q_b (O(k), 0 dense 201K allocs!)
        std::vector<float> step_confidences;            // per-step survival confidence
    };

    // Given parallel block outputs head_hiddens[B, H] and base_logits[B, V] (BEFORE MultiMax!),
    // sequentially roll the Markov chain + post-Markov MultiMax + ConfidenceHead + probabilistic sampling!
    DraftProposal propose_block(
        int32_t anchor_token,
        const float* head_hiddens_BxH,
        const float* base_logits_BxV,
        int B,
        const SamplingParams& sp,
        std::mt19937_64& rng) const
    {
        DraftProposal out;
        out.draft_tokens.reserve(size_t(B));
        out.draft_probs.reserve(size_t(B));
        out.step_confidences.reserve(size_t(B));

        std::vector<float> m_emb(size_t(std::max(1, markov_rank)), 0.0f);
        std::vector<float> logits_b(static_cast<size_t>(vocab_size));
        int32_t prev_tok = anchor_token;
        float cum_survival = 1.0f;

        for (int b = 0; b < B; ++b) {
            markov_embed(prev_tok, m_emb.data());
            const float conf = compute_confidence(head_hiddens_BxH + size_t(b) * size_t(hidden_size), m_emb.data());
            cum_survival *= conf;
            if (b > 0 && has_confidence && cum_survival < conf_stop_tau) break;

            std::memcpy(logits_b.data(), base_logits_BxV + size_t(b) * size_t(vocab_size),
                        size_t(vocab_size) * sizeof(float));
            add_markov_bias_and_multimax(m_emb.data(), logits_b.data());

            std::vector<float> q_b;
            const bool prob_mode = (sp.draft_sample_method == DraftSampleMethod::kProbabilistic && sp.temperature > 1e-6f);
            logits_to_probs_filtered(logits_b.data(), vocab_size,
                                     prob_mode ? sp.temperature : 0.0f,
                                     sp.top_k, sp.top_p, q_b);
            const int32_t tok = sample_from_probs(q_b, rng);

            out.draft_tokens.push_back(tok);
            out.draft_probs.push_back(std::move(q_b));
            out.step_confidences.push_back(conf);
            prev_tok = tok;
        }
        return out;
    }

    // Exact Lossless Speculative Verification (works for BOTH ECHO greedy and DSpark probabilistic!)
    // target_logits_Kp1xV has K+1 rows (after target MultiMax!):
    //   row b in [0..K-1] verifies draft_tokens[b]
    //   row K provides bonus token if all K draft tokens are accepted!
    struct VerifyResult {
        std::vector<int32_t> accepted_and_bonus;  // 1 .. K+1 tokens (guaranteed >= 1!)
        int n_draft_accepted = 0;
    };

    static VerifyResult verify_speculative(
        const std::vector<int32_t>& draft_tokens,
        const std::vector<std::vector<float>>& draft_probs,
        const float* target_logits_Kp1xV,
        int V,
        const SamplingParams& sp,
        std::mt19937_64& rng)
    {
        VerifyResult vr;
        const int K = int(draft_tokens.size());
        const bool probabilistic = (sp.draft_sample_method == DraftSampleMethod::kProbabilistic &&
                                    sp.temperature > 1e-6f &&
                                    int(draft_probs.size()) == K);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        std::vector<float> p_b;

        for (int b = 0; b < K; ++b) {
            logits_to_probs_filtered(target_logits_Kp1xV + size_t(b) * size_t(V),
                                     V, sp.temperature, sp.top_k, sp.top_p, p_b);
            const int32_t d_tok = draft_tokens[size_t(b)];
            if (!probabilistic) {
                // Greedy exact match verification (ECHO do_sample=false or DSpark greedy)
                const int32_t t_tok = int32_t(std::max_element(p_b.begin(), p_b.end()) - p_b.begin());
                if (d_tok == t_tok) {
                    vr.accepted_and_bonus.push_back(d_tok);
                    ++vr.n_draft_accepted;
                    continue;
                }
                vr.accepted_and_bonus.push_back(t_tok);
                return vr;
            }

            // Exact Leviathan / Chen probabilistic rejection sampling:
            // Accept d_tok w.p. min(1, p_b(d_tok) / q_b(d_tok))
            const auto& q_b = draft_probs[size_t(b)];
            const double p_d = (d_tok >= 0 && d_tok < V) ? double(p_b[size_t(d_tok)]) : 0.0;
            const double q_d = (d_tok >= 0 && d_tok < V) ? double(q_b[size_t(d_tok)]) : 0.0;
            const double accept_prob = (q_d <= 1e-30) ? (p_d > 0.0 ? 1.0 : 0.0) : std::min(1.0, p_d / q_d);
            if (ud(rng) <= accept_prob) {
                vr.accepted_and_bonus.push_back(d_tok);
                ++vr.n_draft_accepted;
                continue;
            }

            // Rejected at step b: sample exact residual norm(max(0, p_b - q_b))!
            std::vector<float> resid(size_t(V), 0.0f);
            double rsum = 0.0;
            for (int v = 0; v < V; ++v) {
                const float diff = p_b[size_t(v)] - q_b[size_t(v)];
                if (diff > 0.0f) { resid[size_t(v)] = diff; rsum += double(diff); }
            }
            if (rsum > 1e-12) {
                const float inv_r = float(1.0 / rsum);
                for (float& rv : resid) rv *= inv_r;
                vr.accepted_and_bonus.push_back(sample_from_probs(resid, rng));
            } else {
                vr.accepted_and_bonus.push_back(sample_from_probs(p_b, rng));
            }
            return vr;
        }

        // All K draft tokens accepted -> sample bonus token from row K!
        logits_to_probs_filtered(target_logits_Kp1xV + size_t(K) * size_t(V),
                                 V, sp.temperature, sp.top_k, sp.top_p, p_b);
        vr.accepted_and_bonus.push_back(sample_from_probs(p_b, rng));
        return vr;
    }

    // Wave-8 Zero-Dense-Alloc Sparse Probabilistic Speculative Verification:
    // Exact Leviathan/Chen rejection sampling + residual norm(max(0, p_b - q_b)) in O(k) time (k << 201,216)!
    static VerifyResult verify_speculative_sparse(
        const std::vector<int32_t>& draft_tokens,
        const std::vector<SparseProbs>& draft_sparse,
        const float* target_logits_Kp1xV,
        int V,
        const SamplingParams& sp,
        std::mt19937_64& rng)
    {
        VerifyResult vr;
        const int K = int(draft_tokens.size());
        const bool probabilistic = (sp.draft_sample_method == DraftSampleMethod::kProbabilistic &&
                                    sp.temperature > 1e-6f &&
                                    int(draft_sparse.size()) == K);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        SparseProbs p_b;

        for (int b = 0; b < K; ++b) {
            logits_to_sparse_probs_fast(target_logits_Kp1xV + size_t(b) * size_t(V),
                                        V, sp.temperature, sp.top_k, sp.top_p, p_b);
            const int32_t d_tok = draft_tokens[size_t(b)];
            if (!probabilistic) {
                const int32_t t_tok = p_b.items.empty() ? 0 : p_b.items[0].first;
                if (d_tok == t_tok) {
                    vr.accepted_and_bonus.push_back(d_tok);
                    ++vr.n_draft_accepted;
                    continue;
                }
                vr.accepted_and_bonus.push_back(t_tok);
                return vr;
            }

            const auto& q_b = draft_sparse[size_t(b)];
            const double p_d = double(p_b.prob_of(d_tok));
            const double q_d = double(q_b.prob_of(d_tok));
            const double accept_prob = (q_d <= 1e-30) ? (p_d > 0.0 ? 1.0 : 0.0) : std::min(1.0, p_d / q_d);
            if (ud(rng) <= accept_prob) {
                vr.accepted_and_bonus.push_back(d_tok);
                ++vr.n_draft_accepted;
                continue;
            }

            // Rejected at step b: exact residual norm(max(0, p_b - q_b)) over p_b's sparse support ONLY!
            // (Where p_b[v] == 0, max(0, p_b[v] - q_b[v]) == 0 identically!)
            SparseProbs resid;
            resid.items.reserve(p_b.items.size());
            double rsum = 0.0;
            for (const auto& kv : p_b.items) {
                const float diff = kv.second - q_b.prob_of(kv.first);
                if (diff > 0.0f) {
                    resid.items.emplace_back(kv.first, diff);
                    rsum += double(diff);
                }
            }
            if (rsum > 1e-12) {
                const float inv_r = float(1.0 / rsum);
                for (auto& kv : resid.items) kv.second *= inv_r;
                vr.accepted_and_bonus.push_back(sample_from_sparse_probs(resid, rng));
            } else {
                vr.accepted_and_bonus.push_back(sample_from_sparse_probs(p_b, rng));
            }
            return vr;
        }

        // All K draft tokens accepted -> sample bonus token from row K!
        logits_to_sparse_probs_fast(target_logits_Kp1xV + size_t(K) * size_t(V),
                                    V, sp.temperature, sp.top_k, sp.top_p, p_b);
        vr.accepted_and_bonus.push_back(sample_from_sparse_probs(p_b, rng));
        return vr;
    }
};

} // namespace ernie55
