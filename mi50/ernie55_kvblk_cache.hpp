// SPDX-License-Identifier: Apache-2.0
#pragma once
// 3-Tier Content-Addressed 2048-Token Block-Hashed .kvblk Prefix Store
//   Tier 0: Active GPU VRAM linear/snapshot prefix (20M-85M+ tok/s)
//   Tier 1: Pinned Host DDR4 LRU Block Pool (up to 160 blocks = 327,680 tokens, 3M-5.5M+ tok/s)
//   Tier 2: Persistent Striped Dual-NVMe O_DIRECT .kvblk Store (nvme0n1 + nvme1n1, 800K-1.3M+ tok/s)
// Zero critical-path stall via asynchronous background Dual-NVMe O_DIRECT writeback worker.

#include "ernie55_core_kernels.hpp"
#include <hip/hip_runtime.h>
#include <nmmintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fcntl.h>
#include <sys/mman.h>
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

inline uint32_t kvblk_gf2_mat_vec(const uint32_t mat[32], uint32_t vec) noexcept {
    uint32_t sum = 0u;
    int idx = 0;
    while (vec != 0u) {
        if (vec & 1u) sum ^= mat[idx];
        vec >>= 1;
        ++idx;
    }
    return sum;
}

inline void kvblk_gf2_mat_sq(uint32_t sq[32], const uint32_t mat[32]) noexcept {
    for (int n = 0; n < 32; ++n) sq[n] = kvblk_gf2_mat_vec(mat, mat[n]);
}

inline uint32_t kvblk_crc32c_shift_xor(uint32_t crc1, uint32_t crc2, size_t len2) noexcept {
    if (len2 == 0u) return crc1 ^ crc2;
    uint32_t odd[32], even[32];
    odd[0] = 0x82F63B78u; // Reflected Castagnoli CRC32C polynomial
    uint32_t row = 1u;
    for (int n = 1; n < 32; ++n) { odd[n] = row; row <<= 1; }
    kvblk_gf2_mat_sq(even, odd); // 2 bits
    kvblk_gf2_mat_sq(odd, even); // 4 bits
    kvblk_gf2_mat_sq(even, odd); // 8 bits (1 byte)
    while (len2 > 0u) {
        if (len2 & 1u) crc1 = kvblk_gf2_mat_vec(even, crc1);
        len2 >>= 1;
        if (len2 == 0u) break;
        kvblk_gf2_mat_sq(odd, even);
        if (len2 & 1u) crc1 = kvblk_gf2_mat_vec(odd, crc1);
        len2 >>= 1;
        if (len2 == 0u) break;
        kvblk_gf2_mat_sq(even, odd);
    }
    return crc1 ^ crc2;
}

// Bit-exact 3-way hardware ILP SSE4.2 Castagnoli CRC32C (saturates Zen 2 3-cycle latency / 1-cycle throughput: ~21 GB/s!)
inline uint32_t kvblk_crc32c(const void* data, size_t len) noexcept {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t crc = 0xFFFFFFFFu;
    size_t i = 0;
    if (len >= 12288u) {
        const size_t chunk = (len / 24u) * 8u;
        uint64_t c0 = 0xFFFFFFFFu, c1 = 0u, c2 = 0u;
        const uint8_t* p0 = p;
        const uint8_t* p1 = p + chunk;
        const uint8_t* p2 = p + 2u * chunk;
        for (size_t off = 0; off < chunk; off += 8u) {
            uint64_t v0 = 0, v1 = 0, v2 = 0;
            std::memcpy(&v0, p0 + off, 8);
            std::memcpy(&v1, p1 + off, 8);
            std::memcpy(&v2, p2 + off, 8);
            c0 = _mm_crc32_u64(c0, v0);
            c1 = _mm_crc32_u64(c1, v1);
            c2 = _mm_crc32_u64(c2, v2);
        }
        uint32_t comb = kvblk_crc32c_shift_xor(uint32_t(c0), uint32_t(c1), chunk);
        comb = kvblk_crc32c_shift_xor(comb, uint32_t(c2), chunk);
        crc = uint64_t(comb);
        i = 3u * chunk;
    }
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
        uint8_t*   raw_s0         = nullptr; // 4K-aligned [4096 hdr + delta_padded[0] + tip_padded[0]]
        uint8_t*   raw_s1         = nullptr; // 4K-aligned [4096 hdr + delta_padded[1] + tip_padded[1]]
        uint8_t*   h_s0           = nullptr; // raw_s0 + 4096 (4K-aligned payload for single-pread O_DIRECT!)
        uint8_t*   h_s1           = nullptr; // raw_s1 + 4096 (4K-aligned payload for single-pread O_DIRECT!)
        bool       reg_s0         = false;   // hipHostRegister(Portable) succeeded for raw_s0
        bool       reg_s1         = false;   // hipHostRegister(Portable) succeeded for raw_s1
        // Commit 5: in-flight pin (guarded by mtx_). Incremented AT RESERVATION TIME
        // (restore load/hit or capture write_q_ enqueue) so a slot can NEVER be evicted
        // or overwritten until all 4 GPU H2Ds or background NVMe write complete.
        uint32_t   in_flight      = 0;
    };

    Ernie55BlockKvCache() = default;
    ~Ernie55BlockKvCache() { shutdown(); }
    Ernie55BlockKvCache(const Ernie55BlockKvCache&) = delete;
    Ernie55BlockKvCache& operator=(const Ernie55BlockKvCache&) = delete;

    void init(const std::string& dir_s0, const std::string& dir_s1,
              int spine_dev0, int spine_dev1, int split_layer,
              const int* tp_devs4, const std::vector<KvLayerDesc>& layers,
              const std::vector<uint16_t*>& dspark_swa_rings,
              int max_ddr_blocks = 64,
              uint8_t* const tp_scratch_d_tmp_a[4] = nullptr)
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
            tp_scratch_d_tmp_a_[r] = tp_scratch_d_tmp_a ? tp_scratch_d_tmp_a[r] : nullptr;
            h_hca_pack_[r] = nullptr;
            h_hca_pack_reg_[r] = false;
            (void)alloc_wb_slot(&h_hca_pack_[r], 32ULL * 1024ULL * 1024ULL, &h_hca_pack_reg_[r]);
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

        // Commit 5: Pre-allocate 100% of the DDR4 LRU ring slots upfront (zero hot-path
        // allocation) as 4 KiB-aligned Write-Back hugepage DDR4. hipHostMalloc returns
        // Write-Combining memory on gfx906, which stalls CPU SSE4.2 CRC32C reads; WB +
        // MADV_HUGEPAGE + hipHostRegister(Portable) keeps the CRC32C loop at full speed
        // while staying O_DIRECT-safe and 4 KiB aligned. 0 B net new GPU hipMalloc.
        const int prealloc_n = max_slots_;
        slots_.reserve(size_t(max_slots_));
        for (int i = 0; i < prealloc_n; ++i) {
            DdrSlot ns{};
            bool r0 = false, r1 = false;
            if (alloc_wb_slot(&ns.raw_s0, size_t(4096ULL + slot_bytes_[0]), &r0) &&
                alloc_wb_slot(&ns.raw_s1, size_t(4096ULL + slot_bytes_[1]), &r1)) {
                ns.h_s0   = ns.raw_s0 + 4096u;
                ns.h_s1   = ns.raw_s1 + 4096u;
                ns.reg_s0 = r0;
                ns.reg_s1 = r1;
                slots_.push_back(ns);
            } else {
                free_wb_slot(ns.raw_s0, r0);
                free_wb_slot(ns.raw_s1, r1);
                break;
            }
        }

        stop_writer_.store(false, std::memory_order_relaxed);
        writer_thr_ = std::thread([this]() { writer_loop(); });
        // Wave-11 Turn-9 Pillar II: dedicated independent Dual-NVMe O_DIRECT physical-drive workers
        // (default QD=16 per physical NVMe drive, overridable via ERNIE55_KVBLK_QD=2..24) +
        // 4 dedicated hardware-ILP CRC32C verifier threads so NVMe readers never stall on CPU CRC!
        int qd = kReadersPerStripe;
        if (const char* ev = std::getenv("ERNIE55_KVBLK_QD")) {
            const int v = std::atoi(ev);
            if (v >= 2 && v <= 24) qd = v;
        }
        stop_pool_.store(false, std::memory_order_relaxed);
        for (int s = 0; s < 2; ++s) {
            readers_[s].clear();
            for (int w = 0; w < qd; ++w) {
                readers_[s].emplace_back([this, s]() { reader_loop(s); });
            }
        }
        crc_workers_.clear();
        for (int c = 0; c < kCrcWorkers; ++c) {
            crc_workers_.emplace_back([this]() { crc_loop(); });
        }
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

        {
            std::lock_guard<std::mutex> lk(pool_mtx_);
            stop_pool_.store(true, std::memory_order_release);
        }
        pool_cv_[0].notify_all();
        pool_cv_[1].notify_all();
        {
            std::lock_guard<std::mutex> clk(crc_mtx_);
        }
        crc_cv_.notify_all();
        block_cv_.notify_all();
        for (int s = 0; s < 2; ++s) {
            for (auto& t : readers_[s]) if (t.joinable()) t.join();
            readers_[s].clear();
        }
        for (auto& t : crc_workers_) if (t.joinable()) t.join();
        crc_workers_.clear();
        {
            std::lock_guard<std::mutex> clk(crc_mtx_);
            crc_q_.clear();
        }
        {
            std::lock_guard<std::mutex> lk(pool_mtx_);
            job_q_[0].clear();
            job_q_[1].clear();
            free_jobs_.clear();
            jobs_.clear();
        }

        std::lock_guard<std::mutex> lk(mtx_);
        index_.clear();
        for (auto& sl : slots_) {
            free_wb_slot(sl.raw_s0, sl.reg_s0);
            free_wb_slot(sl.raw_s1, sl.reg_s1);
        }
        slots_.clear();
        for (int r = 0; r < 4; ++r) {
            free_wb_slot(h_hca_pack_[r], h_hca_pack_reg_[r]);
            h_hca_pack_[r] = nullptr;
            h_hca_pack_reg_[r] = false;
            tp_scratch_d_tmp_a_[r] = nullptr;
        }
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
            std::unique_lock<std::mutex> lk(mtx_);
            for (;;) {
                auto it = index_.find(key);
                if (it != index_.end() && slots_[size_t(it->second)].valid) {
                    auto& ex = slots_[size_t(it->second)];
                    ex.lru_tick = ++lru_clock_;
                    if ((!has_valid_tip || ex.has_valid_tip) && (!has_dspark_tip || ex.has_dspark_tip)) return;
                    if (ex.in_flight == 0u) {
                        slot_id = it->second;
                        ex.in_flight++;
                        break;
                    }
                } else {
                    slot_id = alloc_or_evict_slot_locked(key);
                    if (slot_id >= 0) {
                        // alloc_or_evict_slot_locked already incremented in_flight!
                        break;
                    }
                }
                if (slots_.empty()) return;
                slot_cv_.wait_for(lk, std::chrono::microseconds(200));
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
            uint8_t* d_slab = tp_scratch_d_tmp_a_[r_s_[s]];
            if (d_slab != nullptr) {
                // Cut K2.2: 1 TB/s HBM2 pack-gather into idle d_tmp_a + 1 single contiguous D2H (0 B extra VRAM!)
                KvBlkCopyTable tbl{};
                uint32_t off = 0u;
                for (int l = l_beg[s]; l < l_end[s]; ++l) {
                    const auto& L = layers_[size_t(l)];
                    if (!L.is_dsa) {
                        const size_t comp_off = size_t(key.blk_idx) * size_t(kCompPerBlk) * 512u;
                        const uint32_t nb = uint32_t(kCompPerBlk) * 512u * 2u;
                        const uint32_t idx = tbl.count++;
                        tbl.slab_off[idx] = off;
                        tbl.nbytes[idx]   = nb;
                        tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_comp_kv_tp[r_s_[s]] + comp_off);
                        off += nb;
                    } else {
                        const size_t tok_off = size_t(key.blk_idx) * size_t(kBlockTokens);
                        const uint32_t nb_m = uint32_t(kBlockTokens) * 576u * 2u;
                        const uint32_t nb_i = uint32_t(kBlockTokens) * 128u * 2u;
                        uint32_t idx = tbl.count++;
                        tbl.slab_off[idx] = off;
                        tbl.nbytes[idx]   = nb_m;
                        tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_dsa_mla_kv + tok_off * 576u);
                        off += nb_m;
                        idx = tbl.count++;
                        tbl.slab_off[idx] = off;
                        tbl.nbytes[idx]   = nb_i;
                        tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_dsa_ik_cache + tok_off * 128u);
                        off += nb_i;
                    }
                }
                launch_kvblk_pack_gather_hbm2(d_slab, tbl, strms[s]);
                (void)hipMemcpyAsync(dst, d_slab, size_t(delta_bytes_[s]), hipMemcpyDeviceToHost, strms[s]);
                if (has_valid_tip) {
                    KvBlkCopyTable ttbl{};
                    uint32_t toff = 0u;
                    const int r_own = r_s_[s];
                    for (int l = l_beg[s]; l < l_end[s]; ++l) {
                        const auto& L = layers_[size_t(l)];
                        if (L.is_dsa) continue;
                        uint32_t idx = ttbl.count++;
                        ttbl.slab_off[idx] = toff; ttbl.nbytes[idx] = 256u * 512u * 2u;
                        ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_swa_ring_tp[r_own]);
                        toff += 256u * 512u * 2u;
                        idx = ttbl.count++;
                        ttbl.slab_off[idx] = toff; ttbl.nbytes[idx] = 128u * 512u * 4u;
                        ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_state_kv_tp[r_own]);
                        toff += 128u * 512u * 4u;
                        idx = ttbl.count++;
                        ttbl.slab_off[idx] = toff; ttbl.nbytes[idx] = 128u * 512u * 4u;
                        ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_state_sc_tp[r_own]);
                        toff += 128u * 512u * 4u;
                    }
                    if (s == 1 && has_dspark_tip) {
                        for (uint16_t* dr : dspark_rings_) {
                            if (dr) {
                                const uint32_t idx = ttbl.count++;
                                ttbl.slab_off[idx] = toff; ttbl.nbytes[idx] = 256u * 512u * 2u;
                                ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(dr);
                            }
                            toff += 256u * 512u * 2u;
                        }
                    }
                    uint8_t* d_tip_slab = d_slab + size_t(delta_padded_[s]);
                    launch_kvblk_pack_gather_hbm2(d_tip_slab, ttbl, strms[s]);
                    (void)hipMemcpyAsync(dst + delta_padded_[s], d_tip_slab, size_t(tip_bytes_[s]), hipMemcpyDeviceToHost, strms[s]);
                }
            } else {
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
                }
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
        restore_t0_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count(),
                             std::memory_order_relaxed);
        restore_s_end_ns_[0].store(0, std::memory_order_relaxed);
        restore_s_end_ns_[1].store(0, std::memory_order_relaxed);
        restore_io_us_[0].store(0, std::memory_order_relaxed);
        restore_io_us_[1].store(0, std::memory_order_relaxed);
        restore_io_cnt_[0].store(0, std::memory_order_relaxed);
        restore_io_cnt_[1].store(0, std::memory_order_relaxed);
        restore_crc_us_[0].store(0, std::memory_order_relaxed);
        restore_crc_us_[1].store(0, std::memory_order_relaxed);

        // Wave-10 C4 safety fix: stream restore in waves of <= slots_.size() blocks so
        // >131K (>64-block) cold Dual-NVMe restores NEVER evict an in-wave DDR4 slot
        // before all 4 GPUs' async H2D copies from that slot have completed!
        const int n_layers = int(layers_.size());
        const int l_beg[2] = {0, split_layer_};
        const int l_end[2] = {split_layer_, n_layers};
        const int max_wave = std::max(1, int(slots_.size()));

        for (int b0 = start_blk; b0 < hit_blks; b0 += max_wave) {
            const int b1 = std::min(hit_blks, b0 + max_wave);
            const int nw = b1 - b0;
            std::vector<int> slot_ids(static_cast<size_t>(nw), -1);
            std::vector<BlockReadJob*> block_jobs(static_cast<size_t>(nw), nullptr);

            // Commit 5: pin every wave slot under mtx_ AT RESERVATION TIME (zero sid=0 clobber
            // or premature eviction), skip disk I/O on Tier-1 DDR4 hits, and fan out independent
            // stripe-0 / stripe-1 jobs across the 8+8 persistent Dual-NVMe pool (QD=8/drive).
            auto release_wave_pins = [&](int count) {
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    for (int k = 0; k < count; ++k) {
                        const int s_id = slot_ids[size_t(k)];
                        if (s_id >= 0 && slots_[size_t(s_id)].in_flight > 0u) {
                            slots_[size_t(s_id)].in_flight--;
                        }
                    }
                }
                slot_cv_.notify_all();
                for (int k = 0; k < count; ++k) {
                    if (block_jobs[size_t(k)]) retire_block_job(block_jobs[size_t(k)]);
                }
            };

            WaveReadyQueue wave_rq;
            wave_rq.ready[0].reserve(size_t(nw));
            wave_rq.ready[1].reserve(size_t(nw));

            // Submit tip block (b == hit_blks - 1, 45.6 MiB) FIRST so it overlaps the wave instead of
            // sitting as a 25 ms tail at the end of the Dual-NVMe read queue!
            auto submit_one = [&](int i) -> bool {
                const int b = b0 + i;
                const bool need_tip = (b == hit_blks - 1);
                bool was_ddr_hit = false;
                int sid = acquire_or_hit_slot(keys[size_t(b)], need_tip, &was_ddr_hit);
                if (sid < 0) return false;
                slot_ids[size_t(i)] = sid;
                if (!was_ddr_hit) {
                    block_jobs[size_t(i)] = submit_block_read(keys[size_t(b)], need_tip, sid, &wave_rq, i);
                } else {
                    wave_rq.push(0, i, true);
                    wave_rq.push(1, i, true);
                }
                return true;
            };
            const bool tip_in_wave = (nw > 1 && (b0 + nw - 1 == hit_blks - 1));
            bool sub_ok = true;
            if (tip_in_wave) {
                if (!submit_one(nw - 1)) sub_ok = false;
            }
            for (int i = 0; sub_ok && i < (tip_in_wave ? (nw - 1) : nw); ++i) {
                if (!submit_one(i)) { sub_ok = false; break; }
            }
            if (!sub_ok) {
                for (int k = 0; k < nw; ++k) {
                    if (block_jobs[size_t(k)] != nullptr) (void)wait_block_job(block_jobs[size_t(k)]);
                }
                release_wave_pins(nw);
                return existing_vram_tokens;
            }

            // Wave-11 Turn-10 Pillar II: Merged Completion-Order Out-of-Order 4-GPU H2D + 1 TB/s HBM2 Unpack!
            // Drains BOTH stripes concurrently (prioritizing each GPU's owned 8.55 MiB stripe and filling
            // wait gaps with non-owner 304 KiB HCA packs -> 0 second-pass tail!).
            std::atomic<bool> h2d_ok{true};
            std::thread workers[4];
            for (int r = 0; r < 4; ++r) {
                workers[r] = std::thread([&, r]() {
                    (void)hipSetDevice(tp_devs_[r]);
                    hipStream_t st = tp_h2d_streams[r];
                    uint8_t* d_scratch = tp_scratch_d_tmp_a_[r];
                    const bool is_s0 = (tp_devs_[r] == spine_dev_[0]);
                    const bool is_s1 = (tp_devs_[r] == spine_dev_[1]);
                    const int s_pri = is_s1 ? 1 : 0;
                    const int s_sec = 1 - s_pri;
                    uint32_t pk_seq = 0u;
                    int done_seq[2] = {0, 0};

                    while (done_seq[0] < nw || done_seq[1] < nw) {
                        int s = -1;
                        int i = -1;
                        int cur_seq = 0;
                        {
                            std::unique_lock<std::mutex> wlk(wave_rq.mtx);
                            wave_rq.cv.wait(wlk, [&]() {
                                const bool pri_rdy = (done_seq[s_pri] < nw) && (int(wave_rq.ready[s_pri].size()) > done_seq[s_pri]);
                                const bool sec_rdy = (done_seq[s_sec] < nw) && (int(wave_rq.ready[s_sec].size()) > done_seq[s_sec]);
                                return pri_rdy || sec_rdy || wave_rq.failed ||
                                       stop_pool_.load(std::memory_order_acquire);
                            });
                            if (wave_rq.failed || stop_pool_.load(std::memory_order_acquire)) {
                                h2d_ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                            if (done_seq[s_pri] < nw && int(wave_rq.ready[s_pri].size()) > done_seq[s_pri]) {
                                s = s_pri;
                            } else if (done_seq[s_sec] < nw && int(wave_rq.ready[s_sec].size()) > done_seq[s_sec]) {
                                s = s_sec;
                            } else {
                                h2d_ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                            cur_seq = done_seq[s]++;
                            i = wave_rq.ready[s][size_t(cur_seq)];
                        }
                        const int b   = b0 + i;
                        const int sid = slot_ids[size_t(i)];
                        const auto& sl = slots_[size_t(sid)];
                        const bool is_last_blk = (b == hit_blks - 1);

                        const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                        const bool owns_this_stripe = (s == 0 && is_s0) || (s == 1 && is_s1);

                        if (d_scratch != nullptr && owns_this_stripe) {
                            // 1 contiguous 8.55 MiB H2D into 8-slot ring inside 128 MiB d_tmp_a + 1 TB/s HBM2 unpack!
                            uint8_t* d_slab = d_scratch + size_t(cur_seq & 7) * size_t(delta_padded_[s]);
                            (void)hipMemcpyAsync(d_slab, src, size_t(delta_bytes_[s]), hipMemcpyHostToDevice, st);
                            KvBlkCopyTable tbl{};
                            uint32_t off = 0u;
                            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                const auto& L = layers_[size_t(l)];
                                if (!L.is_dsa) {
                                    const size_t comp_off = size_t(b) * size_t(kCompPerBlk) * 512u;
                                    const uint32_t nb = uint32_t(kCompPerBlk) * 512u * 2u;
                                    if (L.d_hca_comp_kv_tp[r]) {
                                        const uint32_t idx = tbl.count++;
                                        tbl.slab_off[idx] = off;
                                        tbl.nbytes[idx]   = nb;
                                        tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_comp_kv_tp[r] + comp_off);
                                    }
                                    off += nb;
                                } else {
                                    const size_t tok_off = size_t(b) * size_t(kBlockTokens);
                                    const uint32_t nb_m = uint32_t(kBlockTokens) * 576u * 2u;
                                    const uint32_t nb_i = uint32_t(kBlockTokens) * 128u * 2u;
                                    uint32_t idx = tbl.count++;
                                    tbl.slab_off[idx] = off;
                                    tbl.nbytes[idx]   = nb_m;
                                    tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_dsa_mla_kv + tok_off * 576u);
                                    off += nb_m;
                                    idx = tbl.count++;
                                    tbl.slab_off[idx] = off;
                                    tbl.nbytes[idx]   = nb_i;
                                    tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_dsa_ik_cache + tok_off * 128u);
                                    off += nb_i;
                                }
                            }
                            launch_kvblk_unpack_scatter_hbm2(d_slab, tbl, st);
                        } else if (d_scratch != nullptr && h_hca_pack_[r] != nullptr) {
                            // Non-owner HCA comp_kv only (~288-304 KiB): pack in L3 -> 1 H2D -> 1 HBM2 scatter!
                            if (pk_seq > 0u && (pk_seq & 63u) == 0u) (void)hipStreamSynchronize(st);
                            uint8_t* h_pk = h_hca_pack_[r] + size_t(pk_seq & 63u) * (512ULL * 1024ULL);
                            uint8_t* d_pk = d_scratch + (88ULL * 1024ULL * 1024ULL) + size_t(pk_seq & 15u) * (512ULL * 1024ULL);
                            ++pk_seq;
                            KvBlkCopyTable tbl{};
                            uint32_t pk_off = 0u;
                            uint64_t src_off = 0u;
                            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                const auto& L = layers_[size_t(l)];
                                if (!L.is_dsa) {
                                    const size_t comp_off = size_t(b) * size_t(kCompPerBlk) * 512u;
                                    const uint32_t nb = uint32_t(kCompPerBlk) * 512u * 2u;
                                    if (L.d_hca_comp_kv_tp[r]) {
                                        std::memcpy(h_pk + pk_off, src + src_off, nb);
                                        const uint32_t idx = tbl.count++;
                                        tbl.slab_off[idx] = pk_off;
                                        tbl.nbytes[idx]   = nb;
                                        tbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_comp_kv_tp[r] + comp_off);
                                        pk_off += nb;
                                    }
                                    src_off += nb;
                                } else {
                                    src_off += uint64_t(kBlockTokens) * (576ULL + 128ULL) * 2ULL;
                                }
                            }
                            if (pk_off > 0u) {
                                (void)hipMemcpyAsync(d_pk, h_pk, pk_off, hipMemcpyHostToDevice, st);
                                launch_kvblk_unpack_scatter_hbm2(d_pk, tbl, st);
                            }
                        } else {
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
                        }

                        if (is_last_blk) {
                            // Ensure full tip metadata is committed (for has_dspark_tip check)
                            if (block_jobs[size_t(i)] != nullptr && !wait_block_job(block_jobs[size_t(i)])) {
                                h2d_ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                            if (d_scratch != nullptr && tip_bytes_[s] > 0u) {
                                uint8_t* d_tip_slab = d_scratch + (100ULL * 1024ULL * 1024ULL);
                                (void)hipMemcpyAsync(d_tip_slab, src + delta_padded_[s], size_t(tip_bytes_[s]), hipMemcpyHostToDevice, st);
                                KvBlkCopyTable ttbl{};
                                uint32_t toff = 0u;
                                for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                    const auto& L = layers_[size_t(l)];
                                    if (L.is_dsa) continue;
                                    if (L.d_hca_swa_ring_tp[r]) {
                                        uint32_t idx = ttbl.count++;
                                        ttbl.slab_off[idx] = toff; ttbl.nbytes[idx] = 256u * 512u * 2u;
                                        ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_swa_ring_tp[r]);
                                        idx = ttbl.count++;
                                        ttbl.slab_off[idx] = toff + 256u * 512u * 2u; ttbl.nbytes[idx] = 128u * 512u * 4u;
                                        ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_state_kv_tp[r]);
                                        idx = ttbl.count++;
                                        ttbl.slab_off[idx] = toff + 256u * 512u * 2u + 128u * 512u * 4u; ttbl.nbytes[idx] = 128u * 512u * 4u;
                                        ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_state_sc_tp[r]);
                                    }
                                    toff += 256u * 512u * 2u + 2u * 128u * 512u * 4u;
                                }
                                const bool has_ds_tip = (block_jobs[size_t(i)] != nullptr)
                                                      ? ((blk_flags[size_t(b)] & 1u) != 0u)
                                                      : sl.has_dspark_tip;
                                if (s == 1 && tp_devs_[r] == spine_dev_[1] && has_ds_tip) {
                                    for (uint16_t* dr : dspark_rings_) {
                                        if (dr) {
                                            const uint32_t idx = ttbl.count++;
                                            ttbl.slab_off[idx] = toff; ttbl.nbytes[idx] = 256u * 512u * 2u;
                                            ttbl.dev_ptr[idx]  = reinterpret_cast<uint64_t>(dr);
                                        }
                                        toff += 256u * 512u * 2u;
                                    }
                                }
                                launch_kvblk_unpack_scatter_hbm2(d_tip_slab, ttbl, st);
                            } else {
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
                                const bool has_ds_tip = (block_jobs[size_t(i)] != nullptr)
                                                      ? ((blk_flags[size_t(b)] & 1u) != 0u)
                                                      : sl.has_dspark_tip;
                                if (s == 1 && tp_devs_[r] == spine_dev_[1] && has_ds_tip) {
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
            // Ensure all overlapped payload CRC32C checks finish before releasing slot pins / committing restore!
            for (int i = 0; i < nw; ++i) {
                if (block_jobs[size_t(i)] != nullptr && !wait_block_job(block_jobs[size_t(i)])) {
                    h2d_ok.store(false, std::memory_order_relaxed);
                }
            }
            release_wave_pins(nw);
            if (!h2d_ok.load(std::memory_order_relaxed)) return existing_vram_tokens;
        }

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const int new_read_toks = (hit_blks - start_blk) * kBlockTokens;
        out_tel.read_tokens = hit_blks * kBlockTokens;
        out_tel.read_ms     = ms;
        out_tel.read_tok_s  = (ms > 0.0) ? (double(new_read_toks) / (ms * 1e-3)) : 0.0;
        out_tel.tier        = (disk_misses > 0) ? "NVMe_kvblk" : "DDR4";
        const int64_t t0_ns = restore_t0_ns_.load(std::memory_order_relaxed);
        const int64_t e0_ns = restore_s_end_ns_[0].load(std::memory_order_relaxed);
        const int64_t e1_ns = restore_s_end_ns_[1].load(std::memory_order_relaxed);
        const double d0_end_ms = (e0_ns > t0_ns) ? (double(e0_ns - t0_ns) * 1e-6) : 0.0;
        const double d1_end_ms = (e1_ns > t0_ns) ? (double(e1_ns - t0_ns) * 1e-6) : 0.0;
        const double cnt0 = std::max(1.0, double(restore_io_cnt_[0].load(std::memory_order_relaxed)));
        const double cnt1 = std::max(1.0, double(restore_io_cnt_[1].load(std::memory_order_relaxed)));
        const double denom = (disk_misses > 0) ? double(disk_misses) : 1.0;
        const double io0_avg = double(restore_io_us_[0].load(std::memory_order_relaxed)) / (1000.0 * cnt0);
        const double io1_avg = double(restore_io_us_[1].load(std::memory_order_relaxed)) / (1000.0 * cnt1);
        const double crc_avg = double(restore_crc_us_[0].load(std::memory_order_relaxed) +
                                      restore_crc_us_[1].load(std::memory_order_relaxed)) / (2000.0 * denom);
        std::printf("[KVBLK-READ] tier=%s blks=%d/%d (disk=%d) tokens=%d ms=%.2f tok_s=%.0f (d0_end=%.1f d1_end=%.1f io0=%.2f io1=%.2f crc=%.2f)\n",
                    out_tel.tier.c_str(), hit_blks - start_blk, hit_blks, disk_misses,
                    new_read_toks, ms, out_tel.read_tok_s,
                    d0_end_ms, d1_end_ms, io0_avg, io1_avg, crc_avg);
        std::fflush(stdout);
        return hit_blks * kBlockTokens;
    }

    size_t ddr_resident_blocks() const noexcept {
        std::lock_guard<std::mutex> lk(mtx_);
        return index_.size();
    }

private:
    // Wave-11 Turn-11 Pillar II: 40/60 Asymmetric Dual-NVMe physical-drive mapping tuned to live 4-GPU H2D!
    // Under concurrent 4-GPU PCIe H2D, nvme0n1 delivers 3.20 GB/s (40.6%) and nvme1n1 delivers 4.69 GB/s (59.4%),
    // so routing b % 5 == 0 .s0 stripes to nvme1n1 equalizes d0_end and d1_end while tip-first OOO H2D cuts tail to 1.5 ms!
    static int preferred_drive(int32_t blk_idx, int stripe) noexcept {
        return (stripe == 0 && (blk_idx % 5) == 0) ? 1 : stripe;
    }

    std::string make_blk_path_on(const KvBlockKey& k, int stripe, int drive) const {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "/blk2k_%016llx_%016llx_b%04d.s%d",
                      static_cast<unsigned long long>(k.h_hi),
                      static_cast<unsigned long long>(k.h_lo),
                      int(k.blk_idx), stripe);
        return dir_s_[drive] + buf;
    }

    bool resolve_blk_path(const KvBlockKey& k, int stripe, std::string& out_path, int& out_drive, uint64_t* out_size = nullptr) const {
        const int pref = preferred_drive(k.blk_idx, stripe);
        const int order[2] = {pref, 1 - pref};
        for (int oi = 0; oi < 2; ++oi) {
            const int d = order[oi];
            std::string p = make_blk_path_on(k, stripe, d);
            struct stat st{};
            if (::stat(p.c_str(), &st) == 0 && uint64_t(st.st_size) >= 4096ULL + delta_padded_[stripe]) {
                out_path = std::move(p);
                out_drive = d;
                if (out_size) *out_size = uint64_t(st.st_size);
                return true;
            }
        }
        return false;
    }

    bool disk_block_exists(const KvBlockKey& k, uint32_t* out_flags = nullptr) const {
        std::string p0, p1;
        int d0 = 0, d1 = 1;
        if (!resolve_blk_path(k, 0, p0, d0) || !resolve_blk_path(k, 1, p1, d1)) return false;
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

    // Reserves a slot (always incrementing in_flight under mtx_ BEFORE returning!)
    int alloc_or_evict_slot_locked(const KvBlockKey& key) {
        auto it_ex = index_.find(key);
        if (it_ex != index_.end()) {
            const int sid = it_ex->second;
            if (slots_[size_t(sid)].in_flight != 0u) return -1;
            slots_[size_t(sid)].in_flight++;
            return sid;
        }
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (!slots_[i].valid && slots_[i].in_flight == 0u) {
                slots_[i].in_flight++;
                slots_[i].key = key;
                return int(i);
            }
        }
        // Commit 5: All DDR4 LRU ring slots are pre-allocated in init() — zero hot-path
        // allocation. A slot with in_flight != 0 is actively pinned for NVMe I/O or 4-GPU
        // async H2D scatter and must never be evicted or overwritten.
        if (slots_.empty()) return -1;
        size_t victim = 0;
        uint64_t best_tick = UINT64_MAX;
        bool found = false;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].in_flight != 0u) continue;
            if (!found || slots_[i].lru_tick < best_tick) {
                best_tick = slots_[i].lru_tick;
                victim = i;
                found = true;
            }
        }
        if (!found) return -1; // every slot is pinned -> caller waits on slot_cv_
        index_.erase(slots_[victim].key);
        slots_[victim].valid = false;
        slots_[victim].key   = key;
        slots_[victim].in_flight++;
        return int(victim);
    }

    // ---- Commit 5: Write-Back hugepage-backed pinned DDR4 slot allocation ----
    // hipHostMalloc hands out Write-Combining memory on gfx906, which stalls the CPU
    // SSE4.2 CRC32C read loop. posix_memalign(4096) + MADV_HUGEPAGE + hipHostRegister
    // (Portable) gives 4 KiB-aligned Write-Back DDR4 that is still O_DIRECT-safe and
    // still 0 B net new GPU hipMalloc. If hipHostRegister fails we fall back cleanly to
    // plain WB DDR4 (free() only) instead of leaking or aborting.
    static bool alloc_wb_slot(uint8_t** out, size_t bytes, bool* registered) {
        void* p = nullptr;
        if (::posix_memalign(&p, 4096, bytes) != 0 || p == nullptr) return false;
        *out         = static_cast<uint8_t*>(p);
        *registered  = false;
        (void)::madvise(p, bytes, MADV_HUGEPAGE);
        // Cut K2.1: OpenMP 8-channel parallel 4 KiB page first-touch before hipHostRegisterPortable!
        auto* bp = static_cast<uint8_t*>(p);
        #pragma omp parallel for schedule(static) if (bytes >= 1048576u)
        for (size_t off = 0; off < bytes; off += 4096u) {
            bp[off] = 0u;
        }
        if (hipHostRegister(p, bytes, hipHostRegisterPortable) == hipSuccess) *registered = true;
        return true;
    }

    static void free_wb_slot(uint8_t* p, bool registered) {
        if (p == nullptr) return;
        if (registered) (void)hipHostUnregister(p);
        std::free(p);
    }

    struct WaveReadyQueue {
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<int> ready[2];
        bool failed = false;
        void push(int stripe, int wave_idx, bool ok) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (!ok) failed = true;
                else ready[stripe].push_back(wave_idx);
            }
            cv.notify_all();
        }
    };

    // One Dual-NVMe block read spanning independent physical-drive worker queues
    // (drive-0 = nvme0n1, drive-1 = nvme1n1). Heap-allocated via unique_ptr so pointers stay stable.
    struct BlockReadJob {
        KvBlockKey key{};
        bool     need_tip = false;
        int      sid      = -1;
        WaveReadyQueue* wave_rq = nullptr;
        int      wave_idx = 0;
        std::string path[2];
        int      drive[2] = {0, 1};
        std::atomic<uint32_t> tip_flags{0u};
        std::atomic<bool>     ok{true};
        std::atomic<int>      stripe_state[2]{{0}, {0}}; // 0=pending, 1=ok, 2=failed
        std::atomic<int>      stripes_done{0};
        std::atomic<int>      state{0}; // 0=in-flight, 1=committed-ok, 2=failed
    };

    struct StripeTask {
        BlockReadJob* jp = nullptr;
        int stripe = 0;
    };

    struct CrcTask {
        BlockReadJob* jp = nullptr;
        int stripe = 0;
        bool io_ok = false;
        uint32_t delta_crc32c = 0u;
        uint32_t tip_crc32c   = 0u;
    };

    BlockReadJob* submit_block_read(const KvBlockKey& key, bool need_tip, int sid,
                                    WaveReadyQueue* wave_rq = nullptr, int wave_idx = 0) {
        std::string p0, p1;
        int d0 = 0, d1 = 1;
        if (!resolve_blk_path(key, 0, p0, d0)) { d0 = preferred_drive(key.blk_idx, 0); p0 = make_blk_path_on(key, 0, d0); }
        if (!resolve_blk_path(key, 1, p1, d1)) { d1 = preferred_drive(key.blk_idx, 1); p1 = make_blk_path_on(key, 1, d1); }
        BlockReadJob* jp = nullptr;
        {
            std::lock_guard<std::mutex> lk(pool_mtx_);
            if (!free_jobs_.empty()) {
                jp = free_jobs_.back();
                free_jobs_.pop_back();
            } else {
                jobs_.push_back(std::make_unique<BlockReadJob>());
                jp = jobs_.back().get();
            }
            jp->key      = key;
            jp->need_tip = need_tip;
            jp->sid      = sid;
            jp->wave_rq  = wave_rq;
            jp->wave_idx = wave_idx;
            jp->path[0]  = std::move(p0);
            jp->path[1]  = std::move(p1);
            jp->drive[0] = d0;
            jp->drive[1] = d1;
            jp->tip_flags.store(0u, std::memory_order_relaxed);
            jp->ok.store(true, std::memory_order_relaxed);
            jp->stripe_state[0].store(0, std::memory_order_relaxed);
            jp->stripe_state[1].store(0, std::memory_order_relaxed);
            jp->stripes_done.store(0, std::memory_order_relaxed);
            jp->state.store(0, std::memory_order_relaxed);
            job_q_[d0].push_back(StripeTask{jp, 0});
            job_q_[d1].push_back(StripeTask{jp, 1});
        }
        pool_cv_[d0].notify_one();
        pool_cv_[d1].notify_one();
        return jp;
    }

    void retire_block_job(BlockReadJob* jp) {
        if (jp == nullptr) return;
        std::lock_guard<std::mutex> lk(pool_mtx_);
        free_jobs_.push_back(jp);
    }

    bool wait_stripe_job(BlockReadJob* jp, int stripe) {
        if (jp == nullptr) return true;
        int st = jp->stripe_state[stripe].load(std::memory_order_acquire);
        if (st == 0) {
            std::unique_lock<std::mutex> lk(block_mtx_);
            block_cv_.wait(lk, [&]() {
                st = jp->stripe_state[stripe].load(std::memory_order_acquire);
                return st != 0 || stop_pool_.load(std::memory_order_acquire);
            });
        }
        return st == 1;
    }

    bool wait_block_job(BlockReadJob* jp) {
        if (jp == nullptr) return true;
        int st = jp->state.load(std::memory_order_acquire);
        if (st == 0) {
            std::unique_lock<std::mutex> lk(block_mtx_);
            block_cv_.wait(lk, [&]() {
                st = jp->state.load(std::memory_order_acquire);
                return st != 0 || stop_pool_.load(std::memory_order_acquire);
            });
        }
        return st == 1;
    }

    void reader_loop(int drive) {
        for (;;) {
            StripeTask task{};
            {
                std::unique_lock<std::mutex> lk(pool_mtx_);
                pool_cv_[drive].wait(lk, [this, drive] {
                    return stop_pool_.load(std::memory_order_acquire) || !job_q_[drive].empty();
                });
                if (job_q_[drive].empty()) {
                    if (stop_pool_.load(std::memory_order_acquire)) return;
                    continue;
                }
                task = job_q_[drive].front();
                job_q_[drive].pop_front();
            }
            BlockReadJob* jp = task.jp;
            const int stripe = task.stripe;
            Ernie55KvBlockDiskHdr hdr{};
            const auto t_io0 = std::chrono::steady_clock::now();
            const bool io_ok = process_read_stripe_io(*jp, stripe, hdr);
            const auto t_io1 = std::chrono::steady_clock::now();
            restore_io_us_[drive].fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t_io1 - t_io0).count(),
                std::memory_order_relaxed);
            restore_io_cnt_[drive].fetch_add(1, std::memory_order_relaxed);
            const int64_t end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_io1.time_since_epoch()).count();
            int64_t prev_end = restore_s_end_ns_[drive].load(std::memory_order_relaxed);
            while (end_ns > prev_end &&
                   !restore_s_end_ns_[drive].compare_exchange_weak(prev_end, end_ns, std::memory_order_relaxed)) {}

            // Hand off 8.55 MiB payload CRC32C to dedicated CRC pool so NVMe readers never stall!
            {
                std::lock_guard<std::mutex> clk(crc_mtx_);
                crc_q_.push_back(CrcTask{jp, stripe, io_ok, hdr.delta_crc32c, hdr.tip_crc32c});
            }
            crc_cv_.notify_one();
        }
    }

    void crc_loop() {
        for (;;) {
            CrcTask ct{};
            {
                std::unique_lock<std::mutex> clk(crc_mtx_);
                crc_cv_.wait(clk, [this]() {
                    return stop_pool_.load(std::memory_order_acquire) || !crc_q_.empty();
                });
                if (crc_q_.empty()) {
                    if (stop_pool_.load(std::memory_order_acquire)) return;
                    continue;
                }
                ct = crc_q_.front();
                crc_q_.pop_front();
            }
            BlockReadJob* jp = ct.jp;
            const int stripe = ct.stripe;
            bool stripe_ok = ct.io_ok;
            if (ct.io_ok) {
                const auto t_c0 = std::chrono::steady_clock::now();
                const auto& sl = slots_[size_t(jp->sid)];
                const uint8_t* dst = (stripe == 0) ? sl.h_s0 : sl.h_s1;
                const bool crc_ok =
                    (kvblk_crc32c(dst, size_t(delta_bytes_[stripe])) == ct.delta_crc32c) &&
                    (!jp->need_tip ||
                     kvblk_crc32c(dst + delta_padded_[stripe], size_t(tip_bytes_[stripe])) == ct.tip_crc32c);
                const auto t_c1 = std::chrono::steady_clock::now();
                restore_crc_us_[stripe].fetch_add(
                    std::chrono::duration_cast<std::chrono::microseconds>(t_c1 - t_c0).count(),
                    std::memory_order_relaxed);
                stripe_ok = crc_ok;
            }
            if (!stripe_ok) jp->ok.store(false, std::memory_order_release);
            // Wake spine GPU IMMEDIATELY in completion order (strictly AFTER CPU CRC finishes -> 0 DDR4 snoop contention!)
            {
                std::lock_guard<std::mutex> blk(block_mtx_);
                jp->stripe_state[stripe].store(stripe_ok ? 1 : 2, std::memory_order_release);
            }
            block_cv_.notify_all();
            if (jp->wave_rq != nullptr) {
                jp->wave_rq->push(stripe, jp->wave_idx, stripe_ok);
            }
            // Once-per-block completion: whichever stripe finishes second commits DdrSlot metadata BEFORE state=1!
            if (jp->stripes_done.fetch_add(1, std::memory_order_acq_rel) == 1) {
                const bool both_ok = jp->ok.load(std::memory_order_acquire);
                if (both_ok) {
                    const uint32_t flg = jp->tip_flags.load(std::memory_order_acquire);
                    std::lock_guard<std::mutex> lk(mtx_);
                    auto& sl = slots_[size_t(jp->sid)];
                    sl.key            = jp->key;
                    sl.has_dspark_tip = jp->need_tip && ((flg & 1u) != 0u);
                    sl.has_valid_tip  = jp->need_tip && ((flg & 2u) != 0u);
                    sl.valid          = true;
                    sl.lru_tick       = ++lru_clock_;
                    index_[jp->key]   = jp->sid;
                }
                {
                    std::lock_guard<std::mutex> blk(block_mtx_);
                    jp->state.store(both_ok ? 1 : 2, std::memory_order_release);
                }
                block_cv_.notify_all();
            }
        }
    }

    bool process_read_stripe_io(BlockReadJob& J, int s, Ernie55KvBlockDiskHdr& hdr_out) {
        auto& sl = slots_[size_t(J.sid)];
        const std::string& p = J.path[s];
        int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) return false;

        // Wave-11 Turn-10 Pillar II: 2 MiB O_DIRECT chunks at QD=4/drive (measured 9.05 GB/s combined on Dual-NVMe, overridable via ERNIE55_KVBLK_CHUNK_MB).
        static const uint64_t kChunkBytes = []() -> uint64_t {
            if (const char* ev = std::getenv("ERNIE55_KVBLK_CHUNK_MB")) {
                const int mb = std::atoi(ev);
                if (mb >= 1 && mb <= 16) return uint64_t(mb) * 1024ULL * 1024ULL;
            }
            return 2ULL * 1024ULL * 1024ULL;
        }();
        uint8_t* raw = (s == 0) ? sl.raw_s0 : sl.raw_s1;
        const uint64_t tot_rd = 4096ULL + (J.need_tip ? (delta_padded_[s] + tip_padded_[s]) : delta_padded_[s]);
        uint64_t done = 0;
        while (done < tot_rd) {
            const size_t want = size_t(std::min<uint64_t>(kChunkBytes, tot_rd - done));
            const ssize_t r = ::pread(fd, raw + done, want, off_t(done));
            if (r <= 0) { ::close(fd); return false; }
            done += uint64_t(r);
        }
        ::close(fd);

        std::memcpy(&hdr_out, raw, sizeof(Ernie55KvBlockDiskHdr));
        if (hdr_out.magic != 0x4B423539u || hdr_out.version != 1u ||
            hdr_out.h_lo != J.key.h_lo || hdr_out.h_hi != J.key.h_hi || hdr_out.blk_idx != J.key.blk_idx ||
            kvblk_crc32c(&hdr_out, offsetof(Ernie55KvBlockDiskHdr, hdr_crc32c)) != hdr_out.hdr_crc32c) {
            return false;
        }
        if (s == 0) J.tip_flags.store(hdr_out.has_dspark_tip, std::memory_order_release);
        return true;
    }

    // Lookup Tier-1 DDR4 hit OR reserve a free/victim slot, always pinning in_flight++ under mtx_!
    int acquire_or_hit_slot(const KvBlockKey& key, bool need_tip, bool* was_ddr_hit) {
        *was_ddr_hit = false;
        std::unique_lock<std::mutex> lk(mtx_);
        for (;;) {
            auto it = index_.find(key);
            if (it != index_.end() && slots_[size_t(it->second)].valid) {
                auto& ex = slots_[size_t(it->second)];
                if (!need_tip || ex.has_valid_tip) {
                    ex.lru_tick = ++lru_clock_;
                    ex.in_flight++;
                    *was_ddr_hit = true;
                    return it->second;
                }
            }
            const int sid = alloc_or_evict_slot_locked(key);
            if (sid >= 0) return sid;
            if (slots_.empty()) return -1;
            slot_cv_.wait_for(lk, std::chrono::microseconds(200));
        }
    }

    void write_slot_to_disk(int sid) {
        KvBlockKey key{};
        uint32_t tip_flags = 0u;
        uint8_t* s_buf[2] = {nullptr, nullptr};
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
                if ((tip_flags & 2u) == 0u) {
                    std::memset(s_buf[s] + delta_padded_[s], 0, size_t(tip_padded_[s]));
                }
                const int drv = preferred_drive(key.blk_idx, s);
                const std::string final_p = make_blk_path_on(key, s, drv);
                const std::string tmp_p   = final_p + ".tmp." + std::to_string(::getpid());
                int fd = ::open(tmp_p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
                if (fd < 0) fd = ::open(tmp_p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) return;
                const uint64_t tot_wr = delta_padded_[s] + tip_padded_[s];
                (void)::posix_fallocate(fd, 0, off_t(4096ULL + tot_wr));

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
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (sid >= 0 && sid < int(slots_.size()) && slots_[size_t(sid)].in_flight > 0u) {
                    slots_[size_t(sid)].in_flight--;
                }
            }
            slot_cv_.notify_all();
            active_writes_.fetch_sub(1, std::memory_order_release);
        }
    }

    bool inited_ = false;
    std::string dir_s_[2];
    int spine_dev_[2]   = {0, 1};
    int r_s_[2]         = {0, 1};
    int split_layer_    = 22;
    int tp_devs_[4]     = {0, 1, 2, 3};
    uint8_t* tp_scratch_d_tmp_a_[4] = {nullptr, nullptr, nullptr, nullptr};
    uint8_t* h_hca_pack_[4]         = {nullptr, nullptr, nullptr, nullptr};
    bool     h_hca_pack_reg_[4]     = {false, false, false, false};
    int max_slots_      = 160;
    uint64_t delta_bytes_[2]  = {0, 0};
    uint64_t delta_padded_[2] = {0, 0};
    uint64_t tip_bytes_[2]    = {0, 0};
    uint64_t tip_padded_[2]   = {0, 0};
    uint64_t slot_bytes_[2]   = {0, 0};
    std::vector<KvLayerDesc> layers_;
    std::vector<uint16_t*>   dspark_rings_;

    mutable std::mutex mtx_;
    std::condition_variable slot_cv_;
    uint64_t lru_clock_ = 0ULL;
    std::vector<DdrSlot> slots_;
    std::unordered_map<KvBlockKey, int, KvBlockKeyHasher> index_;

    // Wave-11 Turn-11 Pillar II: 6+6=12 Dual-NVMe O_DIRECT physical-drive pool + 4 dedicated CRC32C verifiers!
    static constexpr int kReadersPerStripe = 6;
    static constexpr int kCrcWorkers       = 4;
    std::atomic<int64_t>   restore_t0_ns_{0};
    std::atomic<int64_t>   restore_s_end_ns_[2]{{0}, {0}};
    std::atomic<int64_t>   restore_io_us_[2]{{0}, {0}};
    std::atomic<int64_t>   restore_io_cnt_[2]{{0}, {0}};
    std::atomic<int64_t>   restore_crc_us_[2]{{0}, {0}};
    std::vector<std::unique_ptr<BlockReadJob>> jobs_;
    std::vector<BlockReadJob*> free_jobs_;
    std::deque<StripeTask>  job_q_[2];
    std::mutex             pool_mtx_;
    std::condition_variable pool_cv_[2];
    std::deque<CrcTask>    crc_q_;
    std::mutex             crc_mtx_;
    std::condition_variable crc_cv_;
    std::vector<std::thread> crc_workers_;
    std::mutex             block_mtx_;
    std::condition_variable block_cv_;
    std::vector<std::thread> readers_[2];
    std::atomic<bool>      stop_pool_{false};

    std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::deque<int> write_q_;
    std::atomic<bool> stop_writer_{false};
    std::atomic<int>  active_writes_{0};
    std::thread writer_thr_;
};

} // namespace ernie55
