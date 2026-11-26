// SPDX-License-Identifier: Apache-2.0
#pragma once
// 3-Tier Content-Addressed 2048-Token Block-Hashed .kvblk Prefix Store
//   Tier 0: Active GPU VRAM linear/snapshot prefix (20M-85M+ tok/s)
//   Tier 1: Pinned Host DDR4 LRU Block Pool (up to 160 blocks = 327,680 tokens, 3M-5.5M+ tok/s)
//   Tier 2: Persistent Striped Dual-NVMe O_DIRECT .kvblk Store (nvme0n1 + nvme1n1, 800K-1.3M+ tok/s)
// Zero critical-path stall via asynchronous background Dual-NVMe O_DIRECT writeback worker.

#include <hip/hip_runtime.h>
#include <nmmintrin.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ernie55 {

struct KvBlockKey {
    uint64_t h_lo    = 0ULL;
    uint64_t h_hi    = 0ULL;
    int32_t  blk_idx = 0;
    int32_t  pad     = 0;

    bool operator==(const KvBlockKey& o) const noexcept {
        return h_lo == o.h_lo && h_hi == o.h_hi && blk_idx == o.blk_idx;
    }
};

struct KvBlockKeyHasher {
    size_t operator()(const KvBlockKey& k) const noexcept {
        uint64_t x = k.h_lo ^ (k.h_hi * 0x9E3779B97F4A7C15ULL) ^ (uint64_t(uint32_t(k.blk_idx)) * 0xBF58476D1CE4E5B9ULL);
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        return static_cast<size_t>(x);
    }
};

struct alignas(4096) Ernie55KvBlockDiskHdr {
    uint32_t magic;          // 0x4B423539u ("KB59")
    uint32_t version;        // 1
    uint64_t h_lo;
    uint64_t h_hi;
    int32_t  blk_idx;
    uint32_t stripe_idx;
    uint64_t delta_bytes;
    uint64_t delta_padded;
    uint64_t tip_bytes;
    uint64_t tip_padded;
    uint32_t has_dspark_tip; // bit 0: dspark_tip, bit 1: valid_hca_tip
    uint32_t delta_crc32c;
    uint32_t tip_crc32c;
    uint32_t hdr_crc32c;
    uint8_t  pad[4096 - 80];
};
static_assert(sizeof(Ernie55KvBlockDiskHdr) == 4096, "KvBlockDiskHdr must be 4096 B");

inline uint32_t kvblk_crc32c(const void* data, size_t len) noexcept {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t crc = 0xFFFFFFFFu;
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t v = 0;
        std::memcpy(&v, p + i, 8);
        crc = _mm_crc32_u64(crc, v);
    }
    for (; i < len; ++i) {
        crc = _mm_crc32_u8(static_cast<uint32_t>(crc), p[i]);
    }
    return static_cast<uint32_t>(crc ^ 0xFFFFFFFFu);
}

inline KvBlockKey step_kv_block_hash(const KvBlockKey& prev, int32_t blk_idx, const int32_t* toks_2048) noexcept {
    uint64_t lo = (blk_idx == 0) ? 0xCBF29CE484222325ULL : prev.h_lo;
    uint64_t hi = (blk_idx == 0) ? 0x9E3779B97F4A7C15ULL : prev.h_hi;
    lo = (lo ^ uint64_t(uint32_t(blk_idx))) * 0x100000001B3ULL;
    hi = _mm_crc32_u64(hi, uint64_t(uint32_t(blk_idx)) ^ 0xA5A5A5A55A5A5A5AULL) | (hi << 32);
    for (int i = 0; i < 2048; i += 2) {
        const uint64_t pair = uint64_t(uint32_t(toks_2048[i])) | (uint64_t(uint32_t(toks_2048[i + 1])) << 32);
        lo ^= uint64_t(uint32_t(toks_2048[i]));
        lo *= 0x100000001B3ULL;
        lo ^= uint64_t(uint32_t(toks_2048[i + 1]));
        lo *= 0x100000001B3ULL;
        const uint64_t c = _mm_crc32_u64(hi, pair);
        hi = ((hi * 0xD6E8FEB86659FD93ULL) ^ (hi >> 32)) ^ (c * 0x9E3779B97F4A7C15ULL);
    }
    KvBlockKey out{};
    out.h_lo    = lo;
    out.h_hi    = hi;
    out.blk_idx = blk_idx;
    return out;
}

struct KvLayerDesc {
    bool      is_dsa    = false;
    int       owner_dev = 0;
    int       tp_size   = 4;
    uint16_t* d_hca_comp_kv_tp[4]  = {};
    uint16_t* d_hca_swa_ring_tp[4] = {};
    float*    d_hca_state_kv_tp[4] = {};
    float*    d_hca_state_sc_tp[4] = {};
    uint16_t* d_dsa_mla_kv         = nullptr;
    uint16_t* d_dsa_ik_cache       = nullptr;
};

struct KvReadTelemetry {
    int         read_tokens = 0;
    double      read_ms     = 0.0;
    double      read_tok_s  = 0.0;
    std::string tier        = "none"; // "VRAM", "DDR4", "NVMe_kvblk", "none"
};

class Ernie55BlockKvCache {
public:
    static constexpr int      kBlockTokens = 2048;
    static constexpr int      kCompPerBlk  = 16;   // 2048 / 128
    static constexpr uint64_t kAlign4K     = 4096ULL;

    struct DdrSlot {
        KvBlockKey key{};
        bool       valid          = false;
        bool       has_valid_tip  = false;
        bool       has_dspark_tip = false;
        uint64_t   lru_tick       = 0ULL;
        uint8_t*   h_s0           = nullptr; // 4K-aligned pinned [delta_padded[0] + tip_padded[0]]
        uint8_t*   h_s1           = nullptr; // 4K-aligned pinned [delta_padded[1] + tip_padded[1]]
    };

    Ernie55BlockKvCache() = default;
    ~Ernie55BlockKvCache() { shutdown(); }
    Ernie55BlockKvCache(const Ernie55BlockKvCache&) = delete;
    Ernie55BlockKvCache& operator=(const Ernie55BlockKvCache&) = delete;

    void init(const std::string& dir_s0, const std::string& dir_s1,
              int spine_dev0, int spine_dev1, int split_layer,
              const int* tp_devs4, const std::vector<KvLayerDesc>& layers,
              const std::vector<uint16_t*>& dspark_swa_rings,
              int max_ddr_blocks = 64)
    {
        shutdown();
        dir_s_[0]     = dir_s0;
        dir_s_[1]     = dir_s1.empty() ? dir_s0 : dir_s1;
        spine_dev_[0] = spine_dev0;
        spine_dev_[1] = spine_dev1;
        split_layer_  = split_layer;
        for (int r = 0; r < 4; ++r) {
            tp_devs_[r] = tp_devs4[r];
            if (tp_devs4[r] == spine_dev0) r_s_[0] = r;
            if (tp_devs4[r] == spine_dev1) r_s_[1] = r;
        }
        layers_       = layers;
        dspark_rings_ = dspark_swa_rings;
        max_slots_    = std::max(8, max_ddr_blocks);

        std::error_code ec;
        std::filesystem::create_directories(dir_s_[0], ec);
        std::filesystem::create_directories(dir_s_[1], ec);

        const int n_layers = int(layers_.size());
        const int l_beg[2] = {0, split_layer_};
        const int l_end[2] = {split_layer_, n_layers};
        for (int s = 0; s < 2; ++s) {
            uint64_t db = 0, tb = 0;
            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                if (!layers_[size_t(l)].is_dsa) {
                    db += uint64_t(kCompPerBlk) * 512ULL * 2ULL;     // 16 KiB comp_kv
                    tb += 256ULL * 512ULL * 2ULL;                    // 256 KiB swa_ring
                    tb += 2ULL * 128ULL * 512ULL * 4ULL;             // 512 KiB state_kv + state_sc
                } else {
                    db += uint64_t(kBlockTokens) * (576ULL + 128ULL) * 2ULL; // 2,816 KiB mla_kv + ik_cache
                }
            }
            if (s == 1) {
                tb += uint64_t(dspark_rings_.size()) * 256ULL * 512ULL * 2ULL;
            }
            delta_bytes_[s]  = db;
            delta_padded_[s] = (db + 4095ULL) & ~4095ULL;
            tip_bytes_[s]    = tb;
            tip_padded_[s]   = (tb + 4095ULL) & ~4095ULL;
            slot_bytes_[s]   = delta_padded_[s] + tip_padded_[s];
        }

        // Commit 5: Pre-allocate 100% of pinned DDR4 LRU ring slots upfront (zero hot-path hipHostMalloc!)
        const int prealloc_n = max_slots_;
        slots_.reserve(size_t(max_slots_));
        for (int i = 0; i < prealloc_n; ++i) {
            DdrSlot ns{};
            if (hipHostMalloc(reinterpret_cast<void**>(&ns.h_s0), size_t(slot_bytes_[0]), hipHostMallocDefault) == hipSuccess &&
                hipHostMalloc(reinterpret_cast<void**>(&ns.h_s1), size_t(slot_bytes_[1]), hipHostMallocDefault) == hipSuccess) {
                slots_.push_back(ns);
            } else {
                if (ns.h_s0) (void)hipHostFree(ns.h_s0);
                if (ns.h_s1) (void)hipHostFree(ns.h_s1);
                break;
            }
        }

        stop_writer_.store(false, std::memory_order_relaxed);
        writer_thr_ = std::thread([this]() { writer_loop(); });
        inited_ = true;
    }

    void shutdown() noexcept {
        if (!inited_) return;
        {
            std::lock_guard<std::mutex> lk(q_mtx_);
            stop_writer_.store(true, std::memory_order_release);
        }
        q_cv_.notify_all();
        if (writer_thr_.joinable()) writer_thr_.join();

        std::lock_guard<std::mutex> lk(mtx_);
        index_.clear();
        for (auto& sl : slots_) {
            if (sl.h_s0) (void)hipHostFree(sl.h_s0);
            if (sl.h_s1) (void)hipHostFree(sl.h_s1);
        }
        slots_.clear();
        inited_ = false;
    }

    bool initialized() const noexcept { return inited_; }
    int capacity_slots() const noexcept { return int(slots_.size()); }
    int used_slots() const noexcept {
        std::lock_guard<std::mutex> lk(mtx_);
        return int(index_.size());
    }

    // Drain async NVMe write queue (used before cold restart test so all blocks are durable on disk!)
    void flush_async_writes() {
        if (!inited_) return;
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(q_mtx_);
                if (write_q_.empty() && active_writes_.load(std::memory_order_acquire) == 0) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // Capture newly computed block `blk_idx` ([blk_idx*2048 .. (blk_idx+1)*2048)) from GPU VRAM into
    // Tier-1 Pinned DDR4 LRU and enqueue zero-stall background Dual-NVMe O_DIRECT persistence!
    void capture_block_from_vram(const KvBlockKey& key, bool has_valid_tip, bool has_dspark_tip,
                                 hipStream_t s0_d2h, hipStream_t s1_d2h)
    {
        if (!inited_) return;
        int slot_id = -1;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = index_.find(key);
            if (it != index_.end() && slots_[size_t(it->second)].valid) {
                auto& ex = slots_[size_t(it->second)];
                ex.lru_tick = ++lru_clock_;
                if ((!has_valid_tip || ex.has_valid_tip) && (!has_dspark_tip || ex.has_dspark_tip)) return;
                slot_id = it->second;
            } else {
                slot_id = alloc_or_evict_slot_locked(key);
                if (slot_id < 0) return;
            }
        }

        auto& sl = slots_[size_t(slot_id)];
        const int n_layers = int(layers_.size());
        const int l_beg[2] = {0, split_layer_};
        const int l_end[2] = {split_layer_, n_layers};
        hipStream_t strms[2] = {s0_d2h, s1_d2h};

        for (int s = 0; s < 2; ++s) {
            (void)hipSetDevice(spine_dev_[s]);
            uint8_t* dst = (s == 0) ? sl.h_s0 : sl.h_s1;
            uint64_t off = 0;
            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                const auto& L = layers_[size_t(l)];
                if (!L.is_dsa) {
                    const size_t comp_off = size_t(key.blk_idx) * size_t(kCompPerBlk) * 512u;
                    const size_t nb = size_t(kCompPerBlk) * 512u * 2u;
                    (void)hipMemcpyAsync(dst + off, L.d_hca_comp_kv_tp[r_s_[s]] + comp_off, nb, hipMemcpyDeviceToHost, strms[s]);
                    off += nb;
                } else {
                    const size_t tok_off = size_t(key.blk_idx) * size_t(kBlockTokens);
                    const size_t nb_m = size_t(kBlockTokens) * 576u * 2u;
                    const size_t nb_i = size_t(kBlockTokens) * 128u * 2u;
                    (void)hipMemcpyAsync(dst + off, L.d_dsa_mla_kv + tok_off * 576u, nb_m, hipMemcpyDeviceToHost, strms[s]);
                    off += nb_m;
                    (void)hipMemcpyAsync(dst + off, L.d_dsa_ik_cache + tok_off * 128u, nb_i, hipMemcpyDeviceToHost, strms[s]);
                    off += nb_i;
                }
            }
            if (has_valid_tip) {
                uint64_t toff = delta_padded_[s];
                for (int l = l_beg[s]; l < l_end[s]; ++l) {
                    const auto& L = layers_[size_t(l)];
                    if (L.is_dsa) continue;
                    const int r_own = r_s_[s];
                    (void)hipMemcpyAsync(dst + toff, L.d_hca_swa_ring_tp[r_own], 256u * 512u * 2u, hipMemcpyDeviceToHost, strms[s]);
                    toff += 256u * 512u * 2u;
                    (void)hipMemcpyAsync(dst + toff, L.d_hca_state_kv_tp[r_own], 128u * 512u * 4u, hipMemcpyDeviceToHost, strms[s]);
                    toff += 128u * 512u * 4u;
                    (void)hipMemcpyAsync(dst + toff, L.d_hca_state_sc_tp[r_own], 128u * 512u * 4u, hipMemcpyDeviceToHost, strms[s]);
                    toff += 128u * 512u * 4u;
                }
                if (s == 1 && has_dspark_tip) {
                    for (uint16_t* dr : dspark_rings_) {
                        if (dr) {
                            (void)hipMemcpyAsync(dst + toff, dr, 256u * 512u * 2u, hipMemcpyDeviceToHost, strms[1]);
                        }
                        toff += 256u * 512u * 2u;
                    }
                }
            } else {
                std::memset(dst + delta_padded_[s], 0, size_t(tip_padded_[s]));
            }
        }
        for (int s = 0; s < 2; ++s) {
            (void)hipSetDevice(spine_dev_[s]);
            (void)hipStreamSynchronize(strms[s]);
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            sl.key            = key;
            sl.has_valid_tip  = has_valid_tip;
            sl.has_dspark_tip = has_valid_tip && has_dspark_tip;
            sl.valid          = true;
            sl.lru_tick       = ++lru_clock_;
            index_[key]       = slot_id;
        }
        {
            std::lock_guard<std::mutex> qlk(q_mtx_);
            write_q_.push_back(slot_id);
            q_cv_.notify_one();
        }
    }

    // Lookup longest matching prefix of 2048-token blocks in Tier 1 (Pinned DDR4) + Tier 2 (Dual-NVMe O_DIRECT),
    // leaving >= 1 fresh token (`max_blocks = (target_n - 1) / 2048`) so forward produces final token logits!
    int restore_longest_prefix(const std::vector<int32_t>& tok_ids,
                               int existing_vram_tokens,
                               hipStream_t tp_h2d_streams[4],
                               KvReadTelemetry& out_tel)
    {
        out_tel = KvReadTelemetry{};
        if (!inited_ || int(tok_ids.size()) <= kBlockTokens) return existing_vram_tokens;

        const int target_n  = int(tok_ids.size());
        const int max_blks  = (target_n - 1) / kBlockTokens;
        const int start_blk = existing_vram_tokens / kBlockTokens;
        if (max_blks <= start_blk) return existing_vram_tokens;

        std::vector<KvBlockKey> keys(static_cast<size_t>(max_blks));
        KvBlockKey cur{};
        for (int b = 0; b < max_blks; ++b) {
            cur = step_kv_block_hash(cur, b, tok_ids.data() + size_t(b) * size_t(kBlockTokens));
            keys[size_t(b)] = cur;
        }

        // Probe Tier 1 (DDR4) then Tier 2 (Dual-NVMe disk) for contiguous prefix [0 .. raw_blks)
        int raw_blks = 0;
        std::vector<uint32_t> blk_flags(static_cast<size_t>(max_blks), 0u);
        std::vector<bool>     blk_on_disk_only(static_cast<size_t>(max_blks), false);
        for (int b = 0; b < max_blks; ++b) {
            bool in_ddr = false;
            uint32_t flg = 0u;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                auto it = index_.find(keys[size_t(b)]);
                if (it != index_.end() && slots_[size_t(it->second)].valid) {
                    in_ddr = true;
                    const auto& sref = slots_[size_t(it->second)];
                    flg = (sref.has_dspark_tip ? 1u : 0u) | (sref.has_valid_tip ? 2u : 0u);
                }
            }
            if (in_ddr) {
                blk_flags[size_t(b)] = flg;
                raw_blks = b + 1;
            } else if (disk_block_exists(keys[size_t(b)], &flg)) {
                blk_flags[size_t(b)] = flg;
                blk_on_disk_only[size_t(b)] = true;
                raw_blks = b + 1;
            } else {
                break;
            }
        }

        // Step back to highest block with a valid HCA SWA tip (and DSpark tip if fresh remainder < 256)
        int hit_blks = raw_blks;
        while (hit_blks > start_blk) {
            const uint32_t flg = blk_flags[size_t(hit_blks - 1)];
            const bool valid_hca = (flg & 2u) != 0u;
            const bool valid_ds  = (flg & 1u) != 0u;
            const int  rem       = target_n - hit_blks * kBlockTokens;
            if (valid_hca && (dspark_rings_.empty() || rem >= 256 || valid_ds)) break;
            --hit_blks;
        }
        int disk_misses = 0;
        for (int b = start_blk; b < hit_blks; ++b) {
            if (blk_on_disk_only[size_t(b)]) ++disk_misses;
        }
        if (hit_blks <= start_blk) return existing_vram_tokens;

        const auto t0 = std::chrono::steady_clock::now();

        // Ensure all blocks [start_blk .. hit_blks) are resident in Tier-1 DDR4 slots (loading any Tier-2 misses
        // via 2x4-thread parallel Dual-NVMe O_DIRECT pread + SSE4.2 CRC32C verification!)
        std::vector<int> slot_ids(static_cast<size_t>(hit_blks - start_blk), -1);
        for (int b = start_blk; b < hit_blks; ++b) {
            const bool need_tip = (b == hit_blks - 1);
            int sid = -1;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                auto it = index_.find(keys[size_t(b)]);
                if (it != index_.end() && slots_[size_t(it->second)].valid) {
                    if (!need_tip || slots_[size_t(it->second)].has_valid_tip) {
                        sid = it->second;
                        slots_[size_t(sid)].lru_tick = ++lru_clock_;
                    }
                }
            }
            if (sid < 0) {
                sid = load_block_from_disk(keys[size_t(b)], need_tip);
                if (sid < 0) {
                    hit_blks = b;
                    break;
                }
            }
            slot_ids[size_t(b - start_blk)] = sid;
        }
        if (hit_blks <= start_blk) return existing_vram_tokens;

        // Parallel 4-GPU async H2D from pinned DDR4 slots onto tp_devs_[0..3]!
        // - Rank 0 (spine_dev0): DSA L00..L21 + HCA L00..L42
        // - Rank 2 (spine_dev1): DSA L22..L42 + HCA L00..L42 + DSpark tip rings
        // - Ranks 1 & 3:         HCA L00..L42 only (~652 KiB/block -> <7 ms for 262K!)
        const int n_layers = int(layers_.size());
        const int l_beg[2] = {0, split_layer_};
        const int l_end[2] = {split_layer_, n_layers};

        std::thread workers[4];
        for (int r = 0; r < 4; ++r) {
            workers[r] = std::thread([&, r]() {
                (void)hipSetDevice(tp_devs_[r]);
                hipStream_t st = tp_h2d_streams[r];
                for (int b = start_blk; b < hit_blks; ++b) {
                    const auto& sl = slots_[size_t(slot_ids[size_t(b - start_blk)])];
                    const bool is_last_blk = (b == hit_blks - 1);
                    for (int s = 0; s < 2; ++s) {
                        const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                        uint64_t off = 0;
                        for (int l = l_beg[s]; l < l_end[s]; ++l) {
                            const auto& L = layers_[size_t(l)];
                            if (!L.is_dsa) {
                                const size_t comp_off = size_t(b) * size_t(kCompPerBlk) * 512u;
                                const size_t nb = size_t(kCompPerBlk) * 512u * 2u;
                                if (L.d_hca_comp_kv_tp[r]) {
                                    (void)hipMemcpyAsync(L.d_hca_comp_kv_tp[r] + comp_off, src + off, nb, hipMemcpyHostToDevice, st);
                                }
                                off += nb;
                            } else {
                                const size_t tok_off = size_t(b) * size_t(kBlockTokens);
                                const size_t nb_m = size_t(kBlockTokens) * 576u * 2u;
                                const size_t nb_i = size_t(kBlockTokens) * 128u * 2u;
                                if (tp_devs_[r] == L.owner_dev) {
                                    (void)hipMemcpyAsync(L.d_dsa_mla_kv + tok_off * 576u, src + off, nb_m, hipMemcpyHostToDevice, st);
                                    (void)hipMemcpyAsync(L.d_dsa_ik_cache + tok_off * 128u, src + off + nb_m, nb_i, hipMemcpyHostToDevice, st);
                                }
                                off += nb_m + nb_i;
                            }
                        }
                        if (is_last_blk) {
                            uint64_t toff = delta_padded_[s];
                            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                const auto& L = layers_[size_t(l)];
                                if (L.is_dsa) continue;
                                if (L.d_hca_swa_ring_tp[r]) {
                                    (void)hipMemcpyAsync(L.d_hca_swa_ring_tp[r], src + toff, 256u * 512u * 2u, hipMemcpyHostToDevice, st);
                                    (void)hipMemcpyAsync(L.d_hca_state_kv_tp[r], src + toff + 256u * 512u * 2u, 128u * 512u * 4u, hipMemcpyHostToDevice, st);
                                    (void)hipMemcpyAsync(L.d_hca_state_sc_tp[r], src + toff + 256u * 512u * 2u + 128u * 512u * 4u, 128u * 512u * 4u, hipMemcpyHostToDevice, st);
                                }
                                toff += 256u * 512u * 2u + 2u * 128u * 512u * 4u;
                            }
                            if (s == 1 && tp_devs_[r] == spine_dev_[1] && sl.has_dspark_tip) {
                                for (uint16_t* dr : dspark_rings_) {
                                    if (dr) {
                                        (void)hipMemcpyAsync(dr, src + toff, 256u * 512u * 2u, hipMemcpyHostToDevice, st);
                                    }
                                    toff += 256u * 512u * 2u;
                                }
                            }
                        }
                    }
                }
                (void)hipStreamSynchronize(st);
            });
        }
        for (int r = 0; r < 4; ++r) workers[r].join();

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const int new_read_toks = (hit_blks - start_blk) * kBlockTokens;
        out_tel.read_tokens = hit_blks * kBlockTokens;
        out_tel.read_ms     = ms;
        out_tel.read_tok_s  = (ms > 0.0) ? (double(new_read_toks) / (ms * 1e-3)) : 0.0;
        out_tel.tier        = (disk_misses > 0) ? "NVMe_kvblk" : "DDR4";
        std::printf("[KVBLK-READ] tier=%s blks=%d/%d (disk=%d) tokens=%d ms=%.2f tok_s=%.0f\n",
                    out_tel.tier.c_str(), hit_blks - start_blk, hit_blks, disk_misses,
                    new_read_toks, ms, out_tel.read_tok_s);
        std::fflush(stdout);
        return hit_blks * kBlockTokens;
    }

    size_t ddr_resident_blocks() const noexcept {
        std::lock_guard<std::mutex> lk(mtx_);
        return index_.size();
    }

private:
    std::string make_blk_path(const KvBlockKey& k, int s) const {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "/blk2k_%016llx_%016llx_b%04d.s%d",
                      static_cast<unsigned long long>(k.h_hi),
                      static_cast<unsigned long long>(k.h_lo),
                      int(k.blk_idx), s);
        return dir_s_[s] + buf;
    }

    bool disk_block_exists(const KvBlockKey& k, uint32_t* out_flags = nullptr) const {
        const std::string p0 = make_blk_path(k, 0);
        const std::string p1 = make_blk_path(k, 1);
        struct stat st0{}, st1{};
        if (::stat(p0.c_str(), &st0) != 0 || ::stat(p1.c_str(), &st1) != 0) return false;
        if (uint64_t(st0.st_size) < 4096ULL + delta_padded_[0] ||
            uint64_t(st1.st_size) < 4096ULL + delta_padded_[1]) return false;
        if (out_flags) {
            *out_flags = 0u;
            const int fd = ::open(p0.c_str(), O_RDONLY);
            if (fd >= 0) {
                Ernie55KvBlockDiskHdr h{};
                if (::pread(fd, &h, 80, 0) == 80 && h.magic == 0x4B423539u) {
                    *out_flags = h.has_dspark_tip;
                }
                ::close(fd);
            }
        }
        return true;
    }

    int alloc_or_evict_slot_locked(const KvBlockKey& key) {
        auto it_ex = index_.find(key);
        if (it_ex != index_.end()) return it_ex->second;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (!slots_[i].valid) return int(i);
        }
        // Commit 5: All DDR4 LRU ring slots are pre-allocated in init() — zero hot-path hipHostMalloc!
        if (slots_.empty()) return -1;
        size_t victim = 0;
        uint64_t best_tick = UINT64_MAX;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].lru_tick < best_tick) {
                best_tick = slots_[i].lru_tick;
                victim = i;
            }
        }
        index_.erase(slots_[victim].key);
        slots_[victim].valid = false;
        slots_[victim].key   = key;
        return int(victim);
    }

    int load_block_from_disk(const KvBlockKey& key, bool need_tip) {
        int sid = -1;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            sid = alloc_or_evict_slot_locked(key);
            if (sid < 0) return -1;
        }
        auto& sl = slots_[size_t(sid)];
        std::atomic<bool> ok{true};
        uint32_t ds_flag = 0u;
        std::thread rth[2];
        for (int s = 0; s < 2; ++s) {
            rth[s] = std::thread([&, s]() {
                const std::string p = make_blk_path(key, s);
                int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) fd = ::open(p.c_str(), O_RDONLY);
                if (fd < 0) { ok.store(false, std::memory_order_relaxed); return; }

                alignas(4096) Ernie55KvBlockDiskHdr hdr{};
                if (::pread(fd, &hdr, 4096, 0) != 4096 ||
                    hdr.magic != 0x4B423539u || hdr.version != 1u ||
                    hdr.h_lo != key.h_lo || hdr.h_hi != key.h_hi || hdr.blk_idx != key.blk_idx ||
                    kvblk_crc32c(&hdr, offsetof(Ernie55KvBlockDiskHdr, hdr_crc32c)) != hdr.hdr_crc32c) {
                    ::close(fd);
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                if (s == 0) ds_flag = hdr.has_dspark_tip;

                uint8_t* dst = (s == 0) ? sl.h_s0 : sl.h_s1;
                // Wave-9: Non-tip blocks only need delta_padded_[s] (8.96 MiB vs 23.9 MiB = 2.67x less NVMe I/O!)
                const uint64_t tot_rd = need_tip ? (delta_padded_[s] + tip_padded_[s]) : delta_padded_[s];
                constexpr uint64_t kChunk = 2ULL * 1024ULL * 1024ULL;
                const int n_ch = int((tot_rd + kChunk - 1ULL) / kChunk);
                std::atomic<int> next_c{0};
                std::vector<std::thread> iow;
                const int nw = std::min(4, std::max(1, n_ch));
                iow.reserve(size_t(nw));
                for (int w = 0; w < nw; ++w) {
                    iow.emplace_back([&]() {
                        while (ok.load(std::memory_order_relaxed)) {
                            const int ci = next_c.fetch_add(1, std::memory_order_relaxed);
                            if (ci >= n_ch) break;
                            const uint64_t off = uint64_t(ci) * kChunk;
                            const uint64_t len = std::min(kChunk, tot_rd - off);
                            uint64_t done = 0;
                            while (done < len) {
                                const ssize_t r = ::pread(fd, dst + off + done, size_t(len - done), off_t(4096ULL + off + done));
                                if (r <= 0) { ok.store(false, std::memory_order_relaxed); return; }
                                done += uint64_t(r);
                            }
                        }
                    });
                }
                for (auto& t : iow) t.join();
                ::close(fd);
                if (!ok.load(std::memory_order_relaxed)) return;
                if (kvblk_crc32c(dst, size_t(delta_bytes_[s])) != hdr.delta_crc32c) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                if (need_tip &&
                    kvblk_crc32c(dst + delta_padded_[s], size_t(tip_bytes_[s])) != hdr.tip_crc32c) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        rth[0].join();
        rth[1].join();
        if (!ok.load(std::memory_order_relaxed)) return -1;

        std::lock_guard<std::mutex> lk(mtx_);
        sl.key            = key;
        sl.has_dspark_tip = need_tip && ((ds_flag & 1u) != 0u);
        sl.has_valid_tip  = need_tip && ((ds_flag & 2u) != 0u);
        sl.valid          = true;
        sl.lru_tick       = ++lru_clock_;
        index_[key]       = sid;
        return sid;
    }

    void write_slot_to_disk(int sid) {
        KvBlockKey key{};
        uint32_t tip_flags = 0u;
        const uint8_t* s_buf[2] = {nullptr, nullptr};
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (sid < 0 || sid >= int(slots_.size()) || !slots_[size_t(sid)].valid) return;
            key       = slots_[size_t(sid)].key;
            tip_flags = (slots_[size_t(sid)].has_dspark_tip ? 1u : 0u) |
                        (slots_[size_t(sid)].has_valid_tip  ? 2u : 0u);
            s_buf[0]  = slots_[size_t(sid)].h_s0;
            s_buf[1]  = slots_[size_t(sid)].h_s1;
        }
        uint32_t ex_flags = 0u;
        if (disk_block_exists(key, &ex_flags) && ((ex_flags & tip_flags) == tip_flags)) return;

        std::thread wth[2];
        for (int s = 0; s < 2; ++s) {
            wth[s] = std::thread([&, s]() {
                const std::string final_p = make_blk_path(key, s);
                const std::string tmp_p   = final_p + ".tmp." + std::to_string(::getpid());
                int fd = ::open(tmp_p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
                if (fd < 0) fd = ::open(tmp_p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) return;

                alignas(4096) Ernie55KvBlockDiskHdr hdr{};
                hdr.magic          = 0x4B423539u;
                hdr.version        = 1u;
                hdr.h_lo           = key.h_lo;
                hdr.h_hi           = key.h_hi;
                hdr.blk_idx        = key.blk_idx;
                hdr.stripe_idx     = uint32_t(s);
                hdr.delta_bytes    = delta_bytes_[s];
                hdr.delta_padded   = delta_padded_[s];
                hdr.tip_bytes      = tip_bytes_[s];
                hdr.tip_padded     = tip_padded_[s];
                hdr.has_dspark_tip = tip_flags;
                hdr.delta_crc32c   = kvblk_crc32c(s_buf[s], size_t(delta_bytes_[s]));
                hdr.tip_crc32c     = kvblk_crc32c(s_buf[s] + delta_padded_[s], size_t(tip_bytes_[s]));
                hdr.hdr_crc32c     = kvblk_crc32c(&hdr, offsetof(Ernie55KvBlockDiskHdr, hdr_crc32c));

                if (::pwrite(fd, &hdr, 4096, 0) != 4096) {
                    ::close(fd); ::unlink(tmp_p.c_str()); return;
                }
                const uint64_t tot_wr = delta_padded_[s] + tip_padded_[s];
                uint64_t done = 0;
                while (done < tot_wr) {
                    const ssize_t w = ::pwrite(fd, s_buf[s] + done, size_t(tot_wr - done), off_t(4096ULL + done));
                    if (w <= 0) { ::close(fd); ::unlink(tmp_p.c_str()); return; }
                    done += uint64_t(w);
                }
                ::close(fd);
                if (::rename(tmp_p.c_str(), final_p.c_str()) != 0) {
                    ::unlink(tmp_p.c_str());
                }
            });
        }
        wth[0].join();
        wth[1].join();
    }

    void writer_loop() {
        for (;;) {
            int sid = -1;
            {
                std::unique_lock<std::mutex> lk(q_mtx_);
                q_cv_.wait(lk, [&]() {
                    return stop_writer_.load(std::memory_order_acquire) || !write_q_.empty();
                });
                if (write_q_.empty() && stop_writer_.load(std::memory_order_acquire)) break;
                sid = write_q_.front();
                write_q_.pop_front();
                active_writes_.fetch_add(1, std::memory_order_release);
            }
            write_slot_to_disk(sid);
            active_writes_.fetch_sub(1, std::memory_order_release);
        }
    }

    bool inited_ = false;
    std::string dir_s_[2];
    int spine_dev_[2]   = {0, 1};
    int r_s_[2]         = {0, 1};
    int split_layer_    = 22;
    int tp_devs_[4]     = {0, 1, 2, 3};
    int max_slots_      = 160;
    uint64_t delta_bytes_[2]  = {0, 0};
    uint64_t delta_padded_[2] = {0, 0};
    uint64_t tip_bytes_[2]    = {0, 0};
    uint64_t tip_padded_[2]   = {0, 0};
    uint64_t slot_bytes_[2]   = {0, 0};
    std::vector<KvLayerDesc> layers_;
    std::vector<uint16_t*>   dspark_rings_;

    mutable std::mutex mtx_;
    uint64_t lru_clock_ = 0ULL;
    std::vector<DdrSlot> slots_;
    std::unordered_map<KvBlockKey, int, KvBlockKeyHasher> index_;

    std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::deque<int> write_q_;
    std::atomic<bool> stop_writer_{false};
    std::atomic<int>  active_writes_{0};
    std::thread writer_thr_;
};

} // namespace ernie55
