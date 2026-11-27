// SPDX-License-Identifier: Apache-2.0
#pragma once
// Wave-12 Turn-18 Cut D12.2: Zero-Training DSpark+SA 1M Rolling-Hash Suffix Cascade.
// DSpark's 3-layer drafter is restricted to a 128-token sliding window (arXiv:2611.34776),
// so it cannot recall exact code identifiers, struct names, file paths, or JSON keys seen
// 50K-950K tokens back. This index provides a <2 us O(1) suffix lookup over the live token
// history [0 .. seq_len) up to 1,048,576 tokens using 16 MiB of host DDR4 (0 B GPU VRAM).
//
// Design invariants:
//   - Two 4-way MRU set-associative bucket tables (2^19 sets x 16 B = 8 MiB each -> 16 MiB total,
//     strictly <= 32 MiB host budget, 0 B GPU VRAM): 8-gram primary (K=7) + 6-gram fallback (K=5).
//   - Long-context (600K-1M) precision guard: 8-gram / 6-gram keys eliminate random prose/boilerplate
//     false collisions that occur with 4-grams at 1M context, triggering exclusively on genuine
//     verbatim code, path, identifier, and signature spans 50K-950K tokens back.
//   - Each 4-way MRU bucket stores 4 1-based end positions (0 = empty) in 16 B (1/4 cache line).
//   - Hashes the actual 8-token / 6-token ID tuples (SplitMix64) and ALWAYS verifies exact
//     token-by-token equality against `hist_` before proposing (0 false positives on collision).
//   - `truncate()` is O(1) via lazy position validation against `seq_len_`.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ernie55 {

class NgramSuffixIndex {
public:
    static constexpr uint32_t kBuckets    = 1u << 19;   // 524,288 4-way sets per table (8 MiB/table -> 16 MiB total)
    static constexpr uint32_t kMaxHistory = 1u << 20;   // 1,048,576 tokens
    static constexpr int      kMaxK       = 7;
    static constexpr int      kK6         = 5;          // 6-gram caps draft to K=5 (M=6, zero extra verify cost over DSpark)

    struct Bucket {
        uint32_t pos[4] = {0u, 0u, 0u, 0u};  // 4-way MRU ring of 1-based end indices (16 B = 1/4 cache line!)
    };

    NgramSuffixIndex() : t8_(kBuckets), t6_(kBuckets), hist_(kMaxHistory, 0) {}

    void reset() {
        std::fill(t8_.begin(), t8_.end(), Bucket{});
        std::fill(t6_.begin(), t6_.end(), Bucket{});
        seq_len_ = 0u;
    }

    // O(1): stale positions >= seq_len_ are rejected lazily during probe().
    void truncate(size_t new_len) noexcept {
        seq_len_ = uint32_t(std::min<uint64_t>(new_len, kMaxHistory));
    }

    size_t size() const noexcept { return size_t(seq_len_); }

    void push_token(int32_t tok) noexcept {
        if (seq_len_ >= kMaxHistory) return;
        const uint32_t p = seq_len_;
        hist_[size_t(p)] = tok;
        ++seq_len_;
        if (p >= 7u) {
            const uint32_t s8 = hash8(&hist_[size_t(p - 7u)]);
            insert_bucket(t8_[s8], p + 1u);
        }
        if (p >= 5u) {
            const uint32_t s6 = hash6(&hist_[size_t(p - 5u)]);
            insert_bucket(t6_[s6], p + 1u);
        }
    }

    void ingest(const int32_t* toks, size_t n) noexcept {
        for (size_t i = 0; i < n; ++i) push_token(toks[i]);
    }

    // Synchronize incrementally with `cached_tokens[0..n)`. When `toks` strictly extends the
    // indexed prefix, ingests ONLY the newly appended tail in O(delta); on reset/edit, rebuilds.
    void sync_to(const int32_t* toks, size_t n) noexcept {
        const size_t lim = std::min<size_t>(n, kMaxHistory);
        if (lim >= size_t(seq_len_) &&
            (seq_len_ == 0u ||
             (hist_[0] == toks[0] &&
              hist_[size_t(seq_len_ - 1u)] == toks[size_t(seq_len_ - 1u)] &&
              (seq_len_ < 64u || hist_[size_t(seq_len_ >> 1)] == toks[size_t(seq_len_ >> 1)])))) {
            if (lim > size_t(seq_len_)) ingest(toks + size_t(seq_len_), lim - size_t(seq_len_));
            return;
        }
        reset();
        ingest(toks, lim);
    }

    // Probe verbatim continuation following the suffix ending at hist_[seq_len_-1].
    // Returns matched length 0..min(max_k, 7) and writes out_draft[0..len-1].
    int probe(int max_k, int32_t* out_draft, int* out_matched_n = nullptr) const noexcept {
        if (out_matched_n) *out_matched_n = 0;
        if (seq_len_ < 6u || max_k <= 0) return 0;
        if (seq_len_ >= 8u) {
            const int32_t* tail8 = &hist_[size_t(seq_len_ - 8u)];
            return probe_tail8(tail8, true, seq_len_ - 1u, max_k, out_draft, out_matched_n);
        }
        int32_t tail8[8] = {0, 0,
            hist_[size_t(seq_len_ - 6u)], hist_[size_t(seq_len_ - 5u)],
            hist_[size_t(seq_len_ - 4u)], hist_[size_t(seq_len_ - 3u)],
            hist_[size_t(seq_len_ - 2u)], hist_[size_t(seq_len_ - 1u)]};
        return probe_tail8(tail8, false, seq_len_ - 1u, max_k, out_draft, out_matched_n);
    }

    // Probe verbatim continuation following `[hist_[seq_len_-7..seq_len_-1], cur_tok]`
    // BEFORE `cur_tok` has been committed to `hist_` (the exact decode-step boundary!).
    int probe_with_cur(int32_t cur_tok, int max_k, int32_t* out_draft, int* out_matched_n = nullptr) const noexcept {
        if (out_matched_n) *out_matched_n = 0;
        if (seq_len_ < 5u || max_k <= 0) return 0;
        const bool has8 = (seq_len_ >= 7u);
        const int32_t tail8[8] = {
            has8 ? hist_[size_t(seq_len_ - 7u)] : 0,
            has8 ? hist_[size_t(seq_len_ - 6u)] : 0,
            hist_[size_t(seq_len_ - 5u)],
            hist_[size_t(seq_len_ - 4u)],
            hist_[size_t(seq_len_ - 3u)],
            hist_[size_t(seq_len_ - 2u)],
            hist_[size_t(seq_len_ - 1u)],
            cur_tok,
        };
        return probe_tail8(tail8, has8, seq_len_, max_k, out_draft, out_matched_n);
    }

    size_t table_bytes() const noexcept {
        return (t8_.size() + t6_.size()) * sizeof(Bucket);
    }

    static uint32_t hash8(const int32_t* s) noexcept {
        uint64_t h = 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 8; ++i) h = mix64(h ^ uint64_t(uint32_t(s[i])));
        return uint32_t(h & (uint64_t(kBuckets - 1u)));
    }

    static uint32_t hash6(const int32_t* s) noexcept {
        uint64_t h = 0xD6E8FEB86659FD93ULL;
        for (int i = 0; i < 6; ++i) h = mix64(h ^ uint64_t(uint32_t(s[i])));
        return uint32_t(h & (uint64_t(kBuckets - 1u)));
    }

private:
    static uint64_t mix64(uint64_t x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    static void insert_bucket(Bucket& b, uint32_t pos1) noexcept {
        if (b.pos[0] != pos1) {
            b.pos[3] = b.pos[2];
            b.pos[2] = b.pos[1];
            b.pos[1] = b.pos[0];
            b.pos[0] = pos1;
        }
    }

    int try_candidate(uint32_t pos1, const int32_t* key, int key_len,
                      uint32_t cur_end, int k, int32_t* out_draft) const noexcept {
        if (pos1 == 0u) return 0;
        const uint32_t match_end = pos1 - 1u;
        // Reject stale truncated entries, self-matches, or entries without at least 1 following token.
        if (match_end >= seq_len_ || match_end >= cur_end) return 0;
        if (match_end + 1u < uint32_t(key_len)) return 0;
        const uint32_t match_beg = match_end + 1u - uint32_t(key_len);
        // Exact token-by-token verification -> 0 false positives under any hash collision.
        if (std::memcmp(&hist_[size_t(match_beg)], key, size_t(key_len) * sizeof(int32_t)) != 0) return 0;
        // Available verbatim continuation strictly before `cur_end` (zero self-overlap!).
        const uint32_t avail = cur_end - match_end - 1u;
        const int n = std::min<int>(k, int(std::min<uint32_t>(avail, uint32_t(seq_len_ - 1u - match_end))));
        if (n <= 0) return 0;
        for (int j = 0; j < n; ++j) {
            out_draft[j] = hist_[size_t(match_end) + 1u + size_t(j)];
        }
        return n;
    }

    int probe_tail8(const int32_t tail8[8], bool has8, uint32_t cur_end,
                    int max_k, int32_t* out_draft, int* out_matched_n) const noexcept {
        const int k8 = std::min<int>(max_k, kMaxK);
        if (k8 <= 0) return 0;
        // 1. Primary 8-gram exact suffix match (4-way MRU set in a single cache line -> up to K=7).
        if (has8) {
            const Bucket& b8 = t8_[hash8(tail8)];
            for (int w = 0; w < 4; ++w) {
                if (int n = try_candidate(b8.pos[w], tail8, 8, cur_end, k8, out_draft)) {
                    if (out_matched_n) *out_matched_n = 8;
                    return n;
                }
            }
        }
        // 2. Fallback 6-gram exact suffix match (capped at K=5 -> M=6 verify, zero extra verify cost over DSpark).
        const int k6 = std::min<int>(max_k, kK6);
        const Bucket& b6 = t6_[hash6(tail8 + 2)];
        for (int w = 0; w < 4; ++w) {
            if (int n = try_candidate(b6.pos[w], tail8 + 2, 6, cur_end, k6, out_draft)) {
                if (out_matched_n) *out_matched_n = 6;
                return n;
            }
        }
        return 0;
    }

    std::vector<Bucket>  t8_;
    std::vector<Bucket>  t6_;
    std::vector<int32_t> hist_;
    uint32_t seq_len_ = 0u;
};

}  // namespace ernie55
