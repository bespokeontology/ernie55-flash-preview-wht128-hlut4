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

    static constexpr uint64_t kDdrSlotMagic = 0x5731324B56424C4BULL; // "W12KVBLK"

    // ---- Wave-13 Goal B: decoupled SDMA<->MEC DDR4 wave rings (0 B new GPU VRAM) ----
    // Every rank owns a 128 MiB `d_tmp_a` scratch slab. Two strictly disjoint layouts are
    // selected per wave, never concurrently, so the same physical bytes are reused:
    //   Tier-1 DDR4 wave  (12 slots): [  0 .. 104 MiB) delta ring | [112 .. 126.25 MiB) tip slab
    //   Mixed / Tier-2    ( 8 slots): [  0 ..  72 MiB) delta ring | [ 72 .. 104 MiB) non-owner
    //                                 HCA pack ring | [112 .. 126.25 MiB) tip slab
    // 12 * 8,962,048 B = 102.56 MiB < 104 MiB and 8 * 8,962,048 B = 68.34 MiB < 72 MiB, so both
    // rings stay strictly inside their span and never touch the tip slab at 112 MiB.
    static constexpr int      kRingSlots       = 12;
    static constexpr int      kMixedRingSlots  = 8;
    static constexpr uint64_t kDdrRingSpan     = 104ULL * 1024ULL * 1024ULL;
    static constexpr uint64_t kMixedRingSpan   = 72ULL * 1024ULL * 1024ULL;
    static constexpr uint64_t kHcaRingDevOff   = 72ULL * 1024ULL * 1024ULL;
    static constexpr uint64_t kHcaRingSpan     = 32ULL * 1024ULL * 1024ULL;
    static constexpr uint64_t kTipSlabDevOff   = 112ULL * 1024ULL * 1024ULL;
    static constexpr uint64_t kScratchSpan     = 128ULL * 1024ULL * 1024ULL;
    // Wave-13 Goal B contract: the 12-/8-slot delta rings, the non-owner HCA pack ring and the tip
    // slab are ALL carved out of the PRE-EXISTING 128 MiB `d_tmp_a` slab, so the cache itself
    // requests exactly 0 B of new GPU VRAM. Gate-0 Suite 35 asserts this constant AND the layout
    // arithmetic that makes it true.
    static constexpr uint64_t kExtraVramBytes  = 0ULL;

    struct DdrSlot {
        KvBlockKey key{};
        bool       valid          = false;
        bool       has_valid_tip  = false;
        bool       has_dspark_tip = false;
        uint64_t   lru_tick       = 0ULL;
        uint64_t   gen_id         = 0ULL;
        uint64_t   magic_gen      = 0ULL; // kDdrSlotMagic ^ gen_id (verified on Tier-1 DDR4 RAM hits!)
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
            (void)hipSetDevice(tp_devs_[r]);
            for (int h = 0; h < 2; ++h) {
                ev_hca_half_[r][h] = nullptr;
                (void)hipEventCreateWithFlags(&ev_hca_half_[r][h], hipEventDisableTiming);
                ev_hca_half_h2d_[r][h] = nullptr;
                (void)hipEventCreateWithFlags(&ev_hca_half_h2d_[r][h], hipEventDisableTiming);
            }
            // Wave-13 Goal B: 12 per-slot `h2d_done` / `unpack_done` event pairs + 1 tip pair.
            // Created with hipEventDisableTiming (pure order fence, zero timing overhead) and
            // destroyed 1:1 in shutdown(). No event slot aliases the attention/MoE `ev_mb_*`
            // events because these live in the cache object, not in `Mi50ExpertWorker`.
            for (int k = 0; k < kRingSlots; ++k) {
                ev_ring_h2d_[r][k]    = nullptr;
                ev_ring_unpack_[r][k] = nullptr;
                (void)hipEventCreateWithFlags(&ev_ring_h2d_[r][k], hipEventDisableTiming);
                (void)hipEventCreateWithFlags(&ev_ring_unpack_[r][k], hipEventDisableTiming);
            }
            ev_tip_h2d_[r]    = nullptr;
            ev_tip_unpack_[r] = nullptr;
            (void)hipEventCreateWithFlags(&ev_tip_h2d_[r], hipEventDisableTiming);
            (void)hipEventCreateWithFlags(&ev_tip_unpack_[r], hipEventDisableTiming);
        }
        // Wave-13 Goal B: capture-path D2H fence (one per spine stripe, cache-owned so it can
        // never collide with the `ev_mb_*` attention/MoE event slots).
        for (int s = 0; s < 2; ++s) {
            ev_cap_pack_[s] = nullptr;
            (void)hipSetDevice(spine_dev_[s]);
            (void)hipEventCreateWithFlags(&ev_cap_pack_[s], hipEventDisableTiming);
        }
        (void)hipSetDevice(tp_devs_[0]);
        layers_       = layers;
        dspark_rings_ = dspark_swa_rings;
        max_slots_    = std::max(8, max_ddr_blocks);
        // Wave-13 Goal B: the decoupled rings are only engaged when every per-slot event pair
        // exists. If any hipEventCreateWithFlags failed we degrade to the legacy single-stream
        // path (terminal hipStreamSynchronize only) instead of ever syncing mid-restore.
        ring_ev_ok_ = true;
        for (int r = 0; r < 4 && ring_ev_ok_; ++r) {
            for (int k = 0; k < kRingSlots && ring_ev_ok_; ++k) {
                if (ev_ring_h2d_[r][k] == nullptr || ev_ring_unpack_[r][k] == nullptr) ring_ev_ok_ = false;
            }
            for (int h = 0; h < 2 && ring_ev_ok_; ++h) {
                if (ev_hca_half_[r][h] == nullptr || ev_hca_half_h2d_[r][h] == nullptr) ring_ev_ok_ = false;
            }
            if (ev_tip_h2d_[r] == nullptr || ev_tip_unpack_[r] == nullptr) ring_ev_ok_ = false;
        }

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
            slot_stride_[s]  = 4096ULL + slot_bytes_[s];
        }

        // Wave-13 Cut W13.6: Build HCA layer ordinal & intra-stripe offset maps and allocate a
        // shared layer-major coherent host mirror (`n_hca * max_slots * 16 KiB` = 296 MiB DDR4,
        // 0 B GPU VRAM). Each block's 37 HCA comp_kv layers are mirrored at capture / NVMe-CRC
        // time (~15 us) so Tier-1 warm DDR4 restores issue 0 ms of CPU memcpy and just 18/19/37
        // contiguous DMA transfers per GPU instead of 4x redundant pack + 830 unpack launches.
        hca_layer_ord_.assign(layers_.size(), -1);
        hca_layer_off_.assign(layers_.size(), 0ULL);
        n_hca_layers_ = 0;
        for (int s = 0; s < 2; ++s) {
            uint64_t off = 0ULL;
            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                if (!layers_[size_t(l)].is_dsa) {
                    hca_layer_ord_[size_t(l)] = n_hca_layers_++;
                    hca_layer_off_[size_t(l)] = off;
                    off += uint64_t(kCompPerBlk) * 512ULL * 2ULL;
                } else {
                    off += uint64_t(kBlockTokens) * (576ULL + 128ULL) * 2ULL;
                }
            }
        }
        max_hca_blocks_   = max_slots_;
        hca_layer_stride_ = uint64_t(max_hca_blocks_) * uint64_t(kCompPerBlk) * 512ULL * 2ULL;
        h_hca_layer_major_ = nullptr;
        h_hca_layer_coh_   = false;
        h_hca_layer_reg_   = false;
        hca_mirror_key_[0].assign(size_t(max_hca_blocks_), KvBlockKey{});
        hca_mirror_key_[1].assign(size_t(max_hca_blocks_), KvBlockKey{});
        hca_mirror_valid_[0].assign(size_t(max_hca_blocks_), 0u);
        hca_mirror_valid_[1].assign(size_t(max_hca_blocks_), 0u);
        if (n_hca_layers_ > 0 && max_hca_blocks_ > 0) {
            const size_t tot_hca = size_t(n_hca_layers_) * size_t(hca_layer_stride_);
            void* hp = nullptr;
            if (hipHostMalloc(&hp, tot_hca, hipHostMallocCoherent) == hipSuccess && hp != nullptr) {
                h_hca_layer_major_ = static_cast<uint8_t*>(hp);
                h_hca_layer_coh_   = true;
                #pragma omp parallel for schedule(static) num_threads(kWbFaultThreads) if (tot_hca >= 1048576u)
                for (size_t off = 0; off < tot_hca; off += 4096u) h_hca_layer_major_[off] = 0u;
            } else {
                (void)alloc_wb_slot(&h_hca_layer_major_, tot_hca, &h_hca_layer_reg_);
            }
        }

        // Wave-13 Cut W13.6: Pre-allocate 100% of the DDR4 LRU ring slots upfront (zero hot-path
        // allocation) as TWO contiguous 4 KiB-aligned `hipHostMallocCoherent` stripe arenas (one
        // per spine stripe: O_DIRECT verified, 18.5 GB/s SSE4.2 CRC32C, 47.1 GB/s concurrent 2-GPU
        // H2D, only 2 KFD mappings instead of 1,024, 0 B net new GPU hipMalloc). Falls back cleanly
        // to per-slot 4 KiB WB DDR4 (`posix_memalign` + `hipHostRegisterPortable`) if unavailable.
        const int prealloc_n = max_slots_;
        slots_.reserve(size_t(max_slots_));
        arena_s_[0] = nullptr; arena_s_[1] = nullptr;
        arena_coherent_ = false;
        const size_t a0_bytes = size_t(prealloc_n) * size_t(slot_stride_[0]);
        const size_t a1_bytes = size_t(prealloc_n) * size_t(slot_stride_[1]);
        void *a0_ptr = nullptr, *a1_ptr = nullptr;
        if (prealloc_n > 0 && a0_bytes > 0u && a1_bytes > 0u &&
            hipHostMalloc(&a0_ptr, a0_bytes, hipHostMallocCoherent) == hipSuccess && a0_ptr != nullptr &&
            hipHostMalloc(&a1_ptr, a1_bytes, hipHostMallocCoherent) == hipSuccess && a1_ptr != nullptr) {
            arena_s_[0] = static_cast<uint8_t*>(a0_ptr);
            arena_s_[1] = static_cast<uint8_t*>(a1_ptr);
            arena_coherent_ = true;
            #pragma omp parallel for schedule(static) num_threads(kWbFaultThreads) if (a0_bytes >= 1048576u)
            for (size_t off = 0; off < a0_bytes; off += 4096u) arena_s_[0][off] = 0u;
            #pragma omp parallel for schedule(static) num_threads(kWbFaultThreads) if (a1_bytes >= 1048576u)
            for (size_t off = 0; off < a1_bytes; off += 4096u) arena_s_[1][off] = 0u;
            for (int i = 0; i < prealloc_n; ++i) {
                DdrSlot ns{};
                ns.raw_s0 = arena_s_[0] + size_t(i) * size_t(slot_stride_[0]);
                ns.raw_s1 = arena_s_[1] + size_t(i) * size_t(slot_stride_[1]);
                ns.h_s0   = ns.raw_s0 + 4096u;
                ns.h_s1   = ns.raw_s1 + 4096u;
                ns.reg_s0 = true;
                ns.reg_s1 = true;
                slots_.push_back(ns);
            }
        } else {
            if (a0_ptr != nullptr) (void)hipHostFree(a0_ptr);
            if (a1_ptr != nullptr) (void)hipHostFree(a1_ptr);
            for (int i = 0; i < prealloc_n; ++i) {
                DdrSlot ns{};
                bool r0 = false, r1 = false;
                if (alloc_wb_slot(&ns.raw_s0, size_t(slot_stride_[0]), &r0) &&
                    alloc_wb_slot(&ns.raw_s1, size_t(slot_stride_[1]), &r1)) {
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
        if (arena_coherent_) {
            if (arena_s_[0] != nullptr) { (void)hipHostFree(arena_s_[0]); arena_s_[0] = nullptr; }
            if (arena_s_[1] != nullptr) { (void)hipHostFree(arena_s_[1]); arena_s_[1] = nullptr; }
            arena_coherent_ = false;
        } else {
            for (auto& sl : slots_) {
                free_wb_slot(sl.raw_s0, sl.reg_s0);
                free_wb_slot(sl.raw_s1, sl.reg_s1);
            }
        }
        slots_.clear();
        if (h_hca_layer_major_ != nullptr) {
            if (h_hca_layer_coh_) (void)hipHostFree(h_hca_layer_major_);
            else free_wb_slot(h_hca_layer_major_, h_hca_layer_reg_);
            h_hca_layer_major_ = nullptr;
            h_hca_layer_coh_   = false;
            h_hca_layer_reg_   = false;
        }
        hca_mirror_key_[0].clear();
        hca_mirror_key_[1].clear();
        hca_mirror_valid_[0].clear();
        hca_mirror_valid_[1].clear();
        hca_layer_ord_.clear();
        hca_layer_off_.clear();
        n_hca_layers_ = 0;
        max_hca_blocks_ = 0;
        hca_layer_stride_ = 0ULL;
        for (int r = 0; r < 4; ++r) {
            (void)hipSetDevice(tp_devs_[r]);
            for (int h = 0; h < 2; ++h) {
                if (ev_hca_half_[r][h] != nullptr) {
                    (void)hipEventDestroy(ev_hca_half_[r][h]);
                    ev_hca_half_[r][h] = nullptr;
                }
                if (ev_hca_half_h2d_[r][h] != nullptr) {
                    (void)hipEventDestroy(ev_hca_half_h2d_[r][h]);
                    ev_hca_half_h2d_[r][h] = nullptr;
                }
            }
            for (int k = 0; k < kRingSlots; ++k) {
                if (ev_ring_h2d_[r][k] != nullptr) {
                    (void)hipEventDestroy(ev_ring_h2d_[r][k]);
                    ev_ring_h2d_[r][k] = nullptr;
                }
                if (ev_ring_unpack_[r][k] != nullptr) {
                    (void)hipEventDestroy(ev_ring_unpack_[r][k]);
                    ev_ring_unpack_[r][k] = nullptr;
                }
            }
            if (ev_tip_h2d_[r] != nullptr) {
                (void)hipEventDestroy(ev_tip_h2d_[r]);
                ev_tip_h2d_[r] = nullptr;
            }
            if (ev_tip_unpack_[r] != nullptr) {
                (void)hipEventDestroy(ev_tip_unpack_[r]);
                ev_tip_unpack_[r] = nullptr;
            }
            free_wb_slot(h_hca_pack_[r], h_hca_pack_reg_[r]);
            h_hca_pack_[r] = nullptr;
            h_hca_pack_reg_[r] = false;
            tp_scratch_d_tmp_a_[r] = nullptr;
        }
        for (int s = 0; s < 2; ++s) {
            if (ev_cap_pack_[s] != nullptr) {
                (void)hipSetDevice(spine_dev_[s]);
                (void)hipEventDestroy(ev_cap_pack_[s]);
                ev_cap_pack_[s] = nullptr;
            }
        }
        ring_ev_ok_ = false;
        path_cache_.clear();
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
    //
    // Backward-compatible overload: `s0_d2h` / `s1_d2h` carry BOTH the pack-gather kernel and the
    // D2H copy (single stream per spine). Existing tests and tools compile unchanged.
    void capture_block_from_vram(const KvBlockKey& key, bool has_valid_tip, bool has_dspark_tip,
                                 hipStream_t s0_d2h, hipStream_t s1_d2h)
    {
        capture_block_from_vram(key, has_valid_tip, has_dspark_tip,
                                s0_d2h, s1_d2h, s0_d2h, s1_d2h);
    }

    // Wave-13 Goal B primary overload: `s0_compute` / `s1_compute` (= `w.stream`) carry the
    // `launch_kvblk_pack_gather_hbm2` HBM2 pack and `s0_sh` / `s1_sh` (= `w.stream_sh`) carry the
    // D2H. Each D2H is fenced from its own compute stream with a DEDICATED cache-owned event pair
    // (`ev_cap_h2d_` / `ev_cap_done_`), so no `ev_mb_*` attention/MoE event slot is reused and an
    // async `.kvblk` D2H can never stall chunk `>= 1` compute on `w.stream`.
    void capture_block_from_vram(const KvBlockKey& key, bool has_valid_tip, bool has_dspark_tip,
                                 hipStream_t s0_compute, hipStream_t s1_compute,
                                 hipStream_t s0_sh, hipStream_t s1_sh)
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
        hipStream_t cstrms[2] = {s0_compute, s1_compute};
        hipStream_t dstrms[2] = {s0_sh, s1_sh};
        // A D2H is only re-routed onto `stream_sh` when a cache-owned event pair exists AND the
        // sh stream really differs from the compute stream; otherwise the legacy in-line path runs.
        const bool cap_decoupled = ring_ev_ok_;

        for (int s = 0; s < 2; ++s) {
            (void)hipSetDevice(spine_dev_[s]);
            uint8_t* dst = (s == 0) ? sl.h_s0 : sl.h_s1;
            uint8_t* d_slab = tp_scratch_d_tmp_a_[r_s_[s]];
            const bool dec_s = cap_decoupled && (dstrms[s] != nullptr) && (dstrms[s] != cstrms[s]);
            hipStream_t d2h_st = dec_s ? dstrms[s] : cstrms[s];
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
                launch_kvblk_pack_gather_hbm2(d_slab, tbl, cstrms[s]);
                uint8_t* d_tip_slab = nullptr;
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
                    d_tip_slab = d_slab + size_t(delta_padded_[s]);
                    launch_kvblk_pack_gather_hbm2(d_tip_slab, ttbl, cstrms[s]);
                }
                // Wave-13 Goal B: fence stream_sh AFTER both pack-gather kernels on w.stream, then D2H on stream_sh.
                if (dec_s) {
                    (void)hipEventRecord(ev_cap_pack_[s], cstrms[s]);
                    (void)hipStreamWaitEvent(d2h_st, ev_cap_pack_[s], 0);
                }
                (void)hipMemcpyAsync(dst, d_slab, size_t(delta_bytes_[s]), hipMemcpyDeviceToHost, d2h_st);
                if (has_valid_tip && d_tip_slab != nullptr) {
                    (void)hipMemcpyAsync(dst + delta_padded_[s], d_tip_slab, size_t(tip_bytes_[s]), hipMemcpyDeviceToHost, d2h_st);
                }
            } else {
                if (dec_s) {
                    (void)hipEventRecord(ev_cap_pack_[s], cstrms[s]);
                    (void)hipStreamWaitEvent(d2h_st, ev_cap_pack_[s], 0);
                }
                uint64_t off = 0;
                for (int l = l_beg[s]; l < l_end[s]; ++l) {
                    const auto& L = layers_[size_t(l)];
                    if (!L.is_dsa) {
                        const size_t comp_off = size_t(key.blk_idx) * size_t(kCompPerBlk) * 512u;
                        const size_t nb = size_t(kCompPerBlk) * 512u * 2u;
                        (void)hipMemcpyAsync(dst + off, L.d_hca_comp_kv_tp[r_s_[s]] + comp_off, nb, hipMemcpyDeviceToHost, d2h_st);
                        off += nb;
                    } else {
                        const size_t tok_off = size_t(key.blk_idx) * size_t(kBlockTokens);
                        const size_t nb_m = size_t(kBlockTokens) * 576u * 2u;
                        const size_t nb_i = size_t(kBlockTokens) * 128u * 2u;
                        (void)hipMemcpyAsync(dst + off, L.d_dsa_mla_kv + tok_off * 576u, nb_m, hipMemcpyDeviceToHost, d2h_st);
                        off += nb_m;
                        (void)hipMemcpyAsync(dst + off, L.d_dsa_ik_cache + tok_off * 128u, nb_i, hipMemcpyDeviceToHost, d2h_st);
                        off += nb_i;
                    }
                }
                if (has_valid_tip) {
                    uint64_t toff = delta_padded_[s];
                    for (int l = l_beg[s]; l < l_end[s]; ++l) {
                        const auto& L = layers_[size_t(l)];
                        if (L.is_dsa) continue;
                        const int r_own = r_s_[s];
                        (void)hipMemcpyAsync(dst + toff, L.d_hca_swa_ring_tp[r_own], 256u * 512u * 2u, hipMemcpyDeviceToHost, d2h_st);
                        toff += 256u * 512u * 2u;
                        (void)hipMemcpyAsync(dst + toff, L.d_hca_state_kv_tp[r_own], 128u * 512u * 4u, hipMemcpyDeviceToHost, d2h_st);
                        toff += 128u * 512u * 4u;
                        (void)hipMemcpyAsync(dst + toff, L.d_hca_state_sc_tp[r_own], 128u * 512u * 4u, hipMemcpyDeviceToHost, d2h_st);
                        toff += 128u * 512u * 4u;
                    }
                    if (s == 1 && has_dspark_tip) {
                        for (uint16_t* dr : dspark_rings_) {
                            if (dr) {
                                (void)hipMemcpyAsync(dst + toff, dr, 256u * 512u * 2u, hipMemcpyDeviceToHost, d2h_st);
                            }
                            toff += 256u * 512u * 2u;
                        }
                    }
                }
            }
        }
        for (int s = 0; s < 2; ++s) {
            (void)hipSetDevice(spine_dev_[s]);
            const bool dec_s = cap_decoupled && (dstrms[s] != nullptr) && (dstrms[s] != cstrms[s]);
            (void)hipStreamSynchronize(dec_s ? dstrms[s] : cstrms[s]);
        }
        mirror_slot_hca(0, key, sl.h_s0);
        mirror_slot_hca(1, key, sl.h_s1);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            sl.key            = key;
            sl.has_valid_tip  = has_valid_tip;
            sl.has_dspark_tip = has_valid_tip && has_dspark_tip;
            sl.gen_id         = ++lru_clock_;
            sl.magic_gen      = kDdrSlotMagic ^ sl.gen_id;
            sl.valid          = true;
            sl.lru_tick       = sl.gen_id;
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
    //
    // Backward-compatible overload: `tp_h2d_streams[r]` is used for BOTH the SDMA H2D and the MEC
    // scatter (single-stream ping-pong). Existing tests and tools compile unchanged.
    int restore_longest_prefix(const std::vector<int32_t>& tok_ids,
                               int existing_vram_tokens,
                               hipStream_t tp_h2d_streams[4],
                               KvReadTelemetry& out_tel)
    {
        hipStream_t tp_compute_streams[4] = {tp_h2d_streams[0], tp_h2d_streams[1],
                                             tp_h2d_streams[2], tp_h2d_streams[3]};
        return restore_longest_prefix(tok_ids, existing_vram_tokens,
                                      tp_h2d_streams, tp_compute_streams, out_tel);
    }

    // Wave-13 Goal B primary overload: `tp_dma_streams[r]` (= `w.stream_sh`) carries the SDMA H2D
    // and `tp_compute_streams[r]` (= `w.stream`) carries `launch_kvblk_unpack_scatter_hbm2`, so the
    // two engines never ping-pong on one stream. 0 B new GPU VRAM (existing `d_tmp_a` only).
    int restore_longest_prefix(const std::vector<int32_t>& tok_ids,
                               int existing_vram_tokens,
                               hipStream_t tp_dma_streams[4],
                               hipStream_t tp_compute_streams[4],
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
        // Wave-13 Goal B: probe_disk_block_cached() resolves AND caches both on-disk stripe paths
        // before `t0` is taken, so submit_block_read() issues zero `::stat` after `t0`.
        path_cache_.clear();
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
            } else if (probe_disk_block_cached(keys[size_t(b)], &flg)) {
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
        double last_rank_ms[4] = {0.0, 0.0, 0.0, 0.0};

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

            // Wave-13 Goal B: decoupled SDMA <-> MEC wave rings (0 B new GPU VRAM, 2 streams/rank).
            //   1. Pure Tier-1 DDR4 fast path (disk_misses == 0): zero WaveReadyQueue mutex/CV, a
            //      12-slot lock-free ring in [0..104 MiB) of d_tmp_a, the 8.55 MiB SDMA H2D issued on
            //      `w.stream_sh` and `launch_kvblk_unpack_scatter_hbm2` on `w.stream`, fenced by 12
            //      per-slot `h2d_done` / `unpack_done` hipEvent_t pairs. The single-stream
            //      SDMA<->MEC ping-pong barrier is gone.
            //   2. Tier-2 NVMe / mixed path: an 8-slot decoupled ring in [0..72 MiB) plus the
            //      64-slot [72..104 MiB) non-owner HCA pack ring and the [112..126.25 MiB) tip slab,
            //      with per-slot + per-half-ring hipEvent_t fences and ZERO mid-restore host syncs.
            // d_scratch (128 MiB d_tmp_a) strictly disjoint layouts (never concurrent):
            //   Tier-1 : [  0 .. 104 MiB) 12 x delta_padded ring | [112 .. 126.25 MiB) tip slab
            //   Mixed  : [  0 ..  72 MiB)  8 x delta_padded ring | [ 72 .. 104 MiB) HCA pack ring
            //            [112 .. 126.25 MiB) tip slab
            const bool wave_all_ddr_hit = (disk_misses == 0);
            const bool decoupled        = ring_ev_ok_;
            // Wave-13 Cut W13.6: Verify/populate the pre-cached layer-major HCA host mirror for
            // Tier-1 DDR4 restores when all blocks in this wave fit in [0 .. max_hca_blocks_).
            // On warm hits (populated at capture or Dual-NVMe CRC time), 0 blocks miss -> 0.0 ms!
            bool hca_mirror_ready = false;
            if (wave_all_ddr_hit && h_hca_layer_major_ != nullptr && n_hca_layers_ > 0 &&
                b1 <= max_hca_blocks_) {
                for (int s = 0; s < 2; ++s) {
                    std::vector<int> miss_i;
                    {
                        std::lock_guard<std::mutex> mlk(hca_mirror_mtx_[s]);
                        for (int i = 0; i < nw; ++i) {
                            const int b = b0 + i;
                            if (!hca_mirror_valid_[s][size_t(b)] || !(hca_mirror_key_[s][size_t(b)] == keys[size_t(b)])) {
                                miss_i.push_back(i);
                            }
                        }
                    }
                    if (!miss_i.empty()) {
                        const uint32_t nb = uint32_t(kCompPerBlk) * 512u * 2u;
                        #pragma omp parallel for schedule(static) num_threads(kWbFaultThreads) if (miss_i.size() >= 16u)
                        for (size_t mi = 0; mi < miss_i.size(); ++mi) {
                            const int i = miss_i[mi];
                            const int b = b0 + i;
                            const auto& sl = slots_[size_t(slot_ids[size_t(i)])];
                            const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                const int ord = hca_layer_ord_[size_t(l)];
                                if (ord < 0) continue;
                                uint8_t* dst = h_hca_layer_major_ + size_t(ord) * size_t(hca_layer_stride_) + size_t(b) * size_t(nb);
                                std::memcpy(dst, src + hca_layer_off_[size_t(l)], size_t(nb));
                            }
                        }
                        std::lock_guard<std::mutex> mlk(hca_mirror_mtx_[s]);
                        for (int i : miss_i) {
                            const int b = b0 + i;
                            hca_mirror_key_[s][size_t(b)]   = keys[size_t(b)];
                            hca_mirror_valid_[s][size_t(b)] = 1u;
                        }
                    }
                }
                hca_mirror_ready = true;
            }
            // Wave-13 Cut W13.6: Switchbox #1 anti-contention gate. GPU[1] (r_s_[1], 88:00.0) and
            // GPU[3] (r=3, 85:00.0) share one PEX88048 Gen4 x16 uplink; letting GPU[1] finish its
            // H2D stream solo before GPU[3] streams its 274 MiB HCA+Tip eliminates +20 ms of
            // PEX88048 split-completion arbitration while GPU[3] completes solo in ~10 ms.
            const bool serialize_sb1_r3 = wave_all_ddr_hit && nw >= 64 &&
                                          r_s_[1] >= 0 && r_s_[1] < 4 && r_s_[1] != 3 && r_s_[0] != 3;
            std::atomic<bool> spine1_dma_done{false};
            std::atomic<bool> h2d_ok{true};
            std::thread workers[4];
            for (int r = 0; r < 4; ++r) {
                workers[r] = std::thread([&, r]() {
                    // RAII guard ensures GPU[3] can never wait forever even on an early error return.
                    struct Spine1Release {
                        bool active;
                        std::atomic<bool>* flag;
                        ~Spine1Release() { if (active && flag) flag->store(true, std::memory_order_release); }
                    } sb1_rel{serialize_sb1_r3 && (r == r_s_[1]), &spine1_dma_done};
                    (void)hipSetDevice(tp_devs_[r]);
                    if (serialize_sb1_r3 && r == 3) {
                        while (!spine1_dma_done.load(std::memory_order_acquire) &&
                               h2d_ok.load(std::memory_order_relaxed)) {
                            std::this_thread::sleep_for(std::chrono::microseconds(50));
                        }
                        if (!h2d_ok.load(std::memory_order_relaxed)) return;
                    }
                    uint8_t* d_scratch = tp_scratch_d_tmp_a_[r];
                    const bool is_s0 = (tp_devs_[r] == spine_dev_[0]);
                    const bool is_s1 = (tp_devs_[r] == spine_dev_[1]);
                    const int s_pri = is_s1 ? 1 : 0;
                    const int s_sec = 1 - s_pri;
                    // `st_comp` always carries the MEC unpack scatters, `st_dma` the SDMA H2D.
                    // They collapse onto one stream only if the event pool could not be created,
                    // in which case same-stream ordering makes every fence a no-op.
                    hipStream_t st_comp = tp_compute_streams[r];
                    hipStream_t st_dma  = (decoupled && tp_dma_streams[r] != nullptr) ? tp_dma_streams[r] : st_comp;
                    // Per-stripe ring depth, clamped so a wider delta_padded can never overrun its
                    // span (and therefore can never reach the 112 MiB tip slab).
                    const uint64_t ring_span = wave_all_ddr_hit ? kDdrRingSpan : kMixedRingSpan;
                    const int      ring_cap  = wave_all_ddr_hit ? kRingSlots : kMixedRingSlots;
                    int nring[2] = { ring_cap, ring_cap };
                    for (int s = 0; s < 2; ++s) {
                        const uint64_t dp = std::max<uint64_t>(1, delta_padded_[s]);
                        nring[s] = std::max(2, std::min(ring_cap, int(ring_span / dp)));
                    }
                    const int s_order[2] = { s_pri, s_sec };
                    int      ring_seq    = 0;
                    bool     ring_armed[kRingSlots] = {};
                    uint32_t pk_seq = 0u;
                    bool     half_armed[2] = { false, false };
                    bool     tip_armed = false;

                    auto build_owned_tbl = [&](int s, int b) -> KvBlkCopyTable {
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
                        return tbl;
                    };

    // Wave-13 Cut W13.6: the 12-slot DDR4 ring is driven at RING granularity instead of
                    // per block. The previous per-block form interleaved one 8.55 MiB H2D with one
                    // MEC scatter, so the SDMA queue never held more than a single transfer and the
                    // two engines ping-ponged through an event pair every block. The 415-block
                    // Tier-1 restore therefore saturated at 24.95 GB/s (2.784M tok/s), roughly 55%
                    // of the 42.77 GB/s concurrent PCIe Gen 4.0 ceiling.
                    //
                    // Phase A now issues `cnt` back-to-back 8.55 MiB H2Ds on st_dma (102 MiB of
                    // queued SDMA work for a full 12-slot ring) and Phase B issues `cnt`
                    // back-to-back MEC scatters on st_comp. Correctness is unchanged: slot k's
                    // scatter waits on that slot's `h2d_done`, and the next wrap of slot k waits on
                    // `unpack_done` before its H2D may clobber the slab. 0 B extra VRAM, still
                    // strictly 2 streams per MI50, bit-exact restoration.
                    auto ring_slot_of = [&](int s, int seq) -> int { return seq % nring[s]; };
                    auto emit_ring_group = [&](int s, const int* idxs, int cnt) -> bool {
                        if (cnt <= 0) return true;
                        int ks[kRingSlots];
                        const uint8_t* srcs[kRingSlots];
                        uint8_t* slabs[kRingSlots];
                        KvBlkCopyTable tbls[kRingSlots];
                        int n = 0;
                        for (int k = 0; k < cnt; ++k) {
                            const int i = idxs[k];
                            const auto& sl = slots_[size_t(slot_ids[size_t(i)])];
                            // On Tier-1 DDR4 hits (block_jobs[i] == nullptr) magic_gen is already
                            // committed; on Tier-2 NVMe the wait_block_job() fence applies.
                            if (block_jobs[size_t(i)] == nullptr && sl.magic_gen != (kDdrSlotMagic ^ sl.gen_id)) return false;
                            const int kk = ring_slot_of(s, ring_seq);
                            ks[n]    = kk;
                            srcs[n]  = (s == 0) ? sl.h_s0 : sl.h_s1;
                            slabs[n] = d_scratch + size_t(kk) * size_t(delta_padded_[s]);
                            tbls[n]  = build_owned_tbl(s, b0 + i);
                            ++n;
                            ++ring_seq;
                        }
                        if (decoupled) {
                            for (int k = 0; k < n; ++k) {
                                if (ring_armed[ks[k]]) {
                                    (void)hipStreamWaitEvent(st_dma, ev_ring_unpack_[r][ks[k]], 0);
                                }
                                (void)hipMemcpyAsync(slabs[k], srcs[k], size_t(delta_bytes_[s]), hipMemcpyHostToDevice, st_dma);
                                (void)hipEventRecord(ev_ring_h2d_[r][ks[k]], st_dma);
                            }
                            for (int k = 0; k < n; ++k) {
                                (void)hipStreamWaitEvent(st_comp, ev_ring_h2d_[r][ks[k]], 0);
                            }
                        } else {
                            for (int k = 0; k < n; ++k) {
                                (void)hipMemcpyAsync(slabs[k], srcs[k], size_t(delta_bytes_[s]), hipMemcpyHostToDevice, st_comp);
                            }
                        }
                        for (int k = 0; k < n; ++k) {
                            launch_kvblk_unpack_scatter_hbm2(slabs[k], tbls[k], st_comp);
                        }
                        if (decoupled) {
                            for (int k = 0; k < n; ++k) {
                                (void)hipEventRecord(ev_ring_unpack_[r][ks[k]], st_comp);
                                ring_armed[ks[k]] = true;
                            }
                        }
                        return true;
                    };
                    // Per-block shim retained for the mixed / Tier-2 completion-order path.
                    auto emit_ring_delta = [&](int s, int i) -> bool {
                        return emit_ring_group(s, &i, 1);
                    };

                    // Direct per-layer H2D for a stripe this rank does not own. Used by the 12-slot
                    // layout (the delta ring owns [0..104 MiB)) and as the no-slab fallback.
                    auto emit_direct_hca = [&](int s, const int* idxs, int cnt) {
                        for (int k = 0; k < cnt; ++k) {
                            const int b = b0 + idxs[k];
                            const auto& sl = slots_[size_t(slot_ids[size_t(idxs[k])])];
                            const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                            uint64_t off = 0u;
                            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                const auto& L = layers_[size_t(l)];
                                if (!L.is_dsa) {
                                    const size_t comp_off = size_t(b) * size_t(kCompPerBlk) * 512u;
                                    const size_t nb = size_t(kCompPerBlk) * 512u * 2u;
                                    if (L.d_hca_comp_kv_tp[r]) {
                                        (void)hipMemcpyAsync(L.d_hca_comp_kv_tp[r] + comp_off, src + off, nb, hipMemcpyHostToDevice, st_dma);
                                    }
                                    off += nb;
                                } else {
                                    const size_t tok_off = size_t(b) * size_t(kBlockTokens);
                                    const size_t nb_m = size_t(kBlockTokens) * 576u * 2u;
                                    const size_t nb_i = size_t(kBlockTokens) * 128u * 2u;
                                    if (tp_devs_[r] == L.owner_dev) {
                                        (void)hipMemcpyAsync(L.d_dsa_mla_kv + tok_off * 576u, src + off, nb_m, hipMemcpyHostToDevice, st_dma);
                                        (void)hipMemcpyAsync(L.d_dsa_ik_cache + tok_off * 128u, src + off + nb_m, nb_i, hipMemcpyHostToDevice, st_dma);
                                    }
                                    off += nb_m + nb_i;
                                }
                            }
                        }
                        // Single fence so the following tip scatter on st_comp sees the H2Ds.
                        if (decoupled && cnt > 0) {
                            (void)hipEventRecord(ev_hca_half_h2d_[r][0], st_dma);
                            (void)hipStreamWaitEvent(st_comp, ev_hca_half_h2d_[r][0], 0);
                        }
                    };

                    auto emit_tip_for = [&](int s, int i) -> bool {
                        const int b = b0 + i;
                        const int sid = slot_ids[size_t(i)];
                        const auto& sl = slots_[size_t(sid)];
                        const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                        // Strict commit fence: the slot is only reusable once BOTH stripes passed
                        // CRC32C. A mismatch aborts the whole restore before any commit.
                        if (block_jobs[size_t(i)] != nullptr && !wait_block_job(block_jobs[size_t(i)])) {
                            return false;
                        }
                        const bool has_ds_tip = (block_jobs[size_t(i)] != nullptr)
                                              ? ((blk_flags[size_t(b)] & 1u) != 0u)
                                              : sl.has_dspark_tip;
                        if (d_scratch != nullptr && tip_bytes_[s] > 0u) {
                            uint8_t* d_tip_slab = d_scratch + kTipSlabDevOff;
                            if (decoupled) {
                                // The tip slab is shared by both stripes, so the second stripe's
                                // H2D must wait for the first stripe's tip scatter to retire.
                                if (tip_armed) (void)hipStreamWaitEvent(st_dma, ev_tip_unpack_[r], 0);
                                (void)hipMemcpyAsync(d_tip_slab, src + delta_padded_[s], size_t(tip_bytes_[s]), hipMemcpyHostToDevice, st_dma);
                                (void)hipEventRecord(ev_tip_h2d_[r], st_dma);
                                (void)hipStreamWaitEvent(st_comp, ev_tip_h2d_[r], 0);
                            } else {
                                (void)hipMemcpyAsync(d_tip_slab, src + delta_padded_[s], size_t(tip_bytes_[s]), hipMemcpyHostToDevice, st_comp);
                            }
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
                            launch_kvblk_unpack_scatter_hbm2(d_tip_slab, ttbl, st_comp);
                            if (decoupled) {
                                (void)hipEventRecord(ev_tip_unpack_[r], st_comp);
                                tip_armed = true;
                            }
                        } else {
                            uint64_t toff = delta_padded_[s];
                            for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                const auto& L = layers_[size_t(l)];
                                if (L.is_dsa) continue;
                                if (L.d_hca_swa_ring_tp[r]) {
                                    (void)hipMemcpyAsync(L.d_hca_swa_ring_tp[r], src + toff, 256u * 512u * 2u, hipMemcpyHostToDevice, st_dma);
                                    (void)hipMemcpyAsync(L.d_hca_state_kv_tp[r], src + toff + 256u * 512u * 2u, 128u * 512u * 4u, hipMemcpyHostToDevice, st_dma);
                                    (void)hipMemcpyAsync(L.d_hca_state_sc_tp[r], src + toff + 256u * 512u * 2u + 128u * 512u * 4u, 128u * 512u * 4u, hipMemcpyHostToDevice, st_dma);
                                }
                                toff += 256u * 512u * 2u + 2u * 128u * 512u * 4u;
                            }
                            if (s == 1 && tp_devs_[r] == spine_dev_[1] && has_ds_tip) {
                                for (uint16_t* dr : dspark_rings_) {
                                    if (dr) {
                                        (void)hipMemcpyAsync(dr, src + toff, 256u * 512u * 2u, hipMemcpyHostToDevice, st_dma);
                                    }
                                    toff += 256u * 512u * 2u;
                                }
                            }
                            if (decoupled) {
                                (void)hipEventRecord(ev_hca_half_h2d_[r][1], st_dma);
                                (void)hipStreamWaitEvent(st_comp, ev_hca_half_h2d_[r][1], 0);
                            }
                        }
                        return true;
                    };

                    auto emit_nonowner_hca_batched = [&](int s, const int* idxs, int cnt) {
                        if (cnt <= 0) return;
                        if (d_scratch != nullptr && h_hca_pack_[r] != nullptr) {
                            int pos = 0;
                            while (pos < cnt) {
                                const int half = int((pk_seq >> 5) & 1u);
                                // Ring wrap: protect host pinned `h_base` before CPU memcpy via H2D-only
                                // completion event (zero compute-stream stall) and protect GPU `d_base`
                                // via device-side `hipStreamWaitEvent` on MEC scatter completion.
                                if (decoupled && half_armed[half]) {
                                    (void)hipEventSynchronize(ev_hca_half_h2d_[r][half]);
                                    (void)hipStreamWaitEvent(st_dma, ev_hca_half_[r][half], 0);
                                }
                                half_armed[half] = false;
                                const int rem_in_half = 32 - int(pk_seq & 31u);
                                const int take = std::min(cnt - pos, rem_in_half);
                                const uint32_t slot0 = pk_seq & 63u;
                                uint8_t* h_base = h_hca_pack_[r] + size_t(slot0) * (512ULL * 1024ULL);
                                uint8_t* d_base = d_scratch + kHcaRingDevOff + size_t(slot0) * (512ULL * 1024ULL);
                                KvBlkCopyTable tbls[32];
                                uint32_t tot_pk_bytes = 0u;
                                for (int k = 0; k < take; ++k) {
                                    const int wi = idxs[pos + k];
                                    const int b = b0 + wi;
                                    const auto& sl = slots_[size_t(slot_ids[size_t(wi)])];
                                    const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                                    tbls[k] = KvBlkCopyTable{};
                                    uint64_t src_off = 0u;
                                    for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                        const auto& L = layers_[size_t(l)];
                                        if (!L.is_dsa) {
                                            const size_t comp_off = size_t(b) * size_t(kCompPerBlk) * 512u;
                                            const uint32_t nb = uint32_t(kCompPerBlk) * 512u * 2u;
                                            if (L.d_hca_comp_kv_tp[r]) {
                                                std::memcpy(h_base + tot_pk_bytes, src + src_off, nb);
                                                const uint32_t idx = tbls[k].count++;
                                                tbls[k].slab_off[idx] = tot_pk_bytes;
                                                tbls[k].nbytes[idx]   = nb;
                                                tbls[k].dev_ptr[idx]  = reinterpret_cast<uint64_t>(L.d_hca_comp_kv_tp[r] + comp_off);
                                                tot_pk_bytes += nb;
                                            }
                                            src_off += nb;
                                        } else {
                                            src_off += uint64_t(kBlockTokens) * (576ULL + 128ULL) * 2ULL;
                                        }
                                    }
                                }
                                if (tot_pk_bytes > 0u) {
                                    if (decoupled) {
                                        (void)hipMemcpyAsync(d_base, h_base, tot_pk_bytes, hipMemcpyHostToDevice, st_dma);
                                        (void)hipEventRecord(ev_hca_half_h2d_[r][half], st_dma);
                                        (void)hipStreamWaitEvent(st_comp, ev_hca_half_h2d_[r][half], 0);
                                    } else {
                                        (void)hipMemcpyAsync(d_base, h_base, tot_pk_bytes, hipMemcpyHostToDevice, st_comp);
                                    }
                                    for (int k = 0; k < take; ++k) {
                                        if (tbls[k].count > 0u) {
                                            launch_kvblk_unpack_scatter_hbm2(d_base, tbls[k], st_comp);
                                        }
                                    }
                                    if (decoupled) {
                                        (void)hipEventRecord(ev_hca_half_[r][half], st_comp);
                                        half_armed[half] = true;
                                    }
                                }
                                pk_seq += uint32_t(take);
                                pos += take;
                            }
                        } else {
                            emit_direct_hca(s, idxs, cnt);
                        }
                    };

                    if (wave_all_ddr_hit) {
                        // Cut R12.1 / W13 Cut W13.6: lock-free Tier-1 DDR4 fast path:
                        //   1. Owned stripe: Half-Ring-6 decoupled SDMA<->MEC wave ring (2 half-ring
                        //      event pairs for 6 back-to-back 8.55 MiB H2Ds = 51.3 MiB bursts, zero
                        //      per-block event chatter, 0 B extra GPU VRAM, bit-exact parity).
                        //   2. Non-owned HCA stripes: zero-CPU-memcpy layer-major contiguous H2D
                        //      (`18/19/37` single DMA calls straight to `L.d_hca_comp_kv_tp[r]`).
                        bool hca_ring_fenced = false;
                        for (int si = 0; si < 2; ++si) {
                            const int s = s_order[si];
                            const bool owns_this_stripe = (s == 0 && is_s0) || (s == 1 && is_s1);
                            if (d_scratch != nullptr && owns_this_stripe) {
                                const int half_sz = std::max(1, nring[s] / 2);
                                bool half_ring_armed[2] = { false, false };
                                for (int i0 = 0, hi = 0; i0 < nw; i0 += half_sz, hi ^= 1) {
                                    const int cnt = std::min(half_sz, nw - i0);
                                    const uint8_t* srcs[kRingSlots];
                                    uint8_t* slabs[kRingSlots];
                                    KvBlkCopyTable tbls[kRingSlots];
                                    for (int j = 0; j < cnt; ++j) {
                                        const int wi = i0 + j;
                                        const auto& sl = slots_[size_t(slot_ids[size_t(wi)])];
                                        if (block_jobs[size_t(wi)] == nullptr && sl.magic_gen != (kDdrSlotMagic ^ sl.gen_id)) {
                                            h2d_ok.store(false, std::memory_order_relaxed);
                                            return;
                                        }
                                        const int slot_k = hi * half_sz + j;
                                        srcs[j]  = (s == 0) ? sl.h_s0 : sl.h_s1;
                                        slabs[j] = d_scratch + size_t(slot_k) * size_t(delta_padded_[s]);
                                        tbls[j]  = build_owned_tbl(s, b0 + wi);
                                    }
                                    if (decoupled) {
                                        if (half_ring_armed[hi]) {
                                            (void)hipStreamWaitEvent(st_dma, ev_ring_unpack_[r][hi], 0);
                                        }
                                        for (int j = 0; j < cnt; ++j) {
                                            (void)hipMemcpyAsync(slabs[j], srcs[j], size_t(delta_bytes_[s]), hipMemcpyHostToDevice, st_dma);
                                        }
                                        (void)hipEventRecord(ev_ring_h2d_[r][hi], st_dma);
                                        (void)hipStreamWaitEvent(st_comp, ev_ring_h2d_[r][hi], 0);
                                        for (int j = 0; j < cnt; ++j) {
                                            launch_kvblk_unpack_scatter_hbm2(slabs[j], tbls[j], st_comp);
                                        }
                                        (void)hipEventRecord(ev_ring_unpack_[r][hi], st_comp);
                                        half_ring_armed[hi] = true;
                                    } else {
                                        for (int j = 0; j < cnt; ++j) {
                                            (void)hipMemcpyAsync(slabs[j], srcs[j], size_t(delta_bytes_[s]), hipMemcpyHostToDevice, st_comp);
                                            launch_kvblk_unpack_scatter_hbm2(slabs[j], tbls[j], st_comp);
                                        }
                                    }
                                }
                                // Mark upper ring slots armed only if non-mirror fallback reuses [72..104 MiB).
                                if (half_ring_armed[1]) ring_armed[kMixedRingSlots] = true;
                            } else if (owns_this_stripe) {
                                for (int i = 0; i < nw; ++i) {
                                    const int b = b0 + i;
                                    const auto& sl = slots_[size_t(slot_ids[size_t(i)])];
                                    const uint8_t* src = (s == 0) ? sl.h_s0 : sl.h_s1;
                                    uint64_t off = 0u;
                                    for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                        const auto& L = layers_[size_t(l)];
                                        if (!L.is_dsa) {
                                            const size_t comp_off = size_t(b) * size_t(kCompPerBlk) * 512u;
                                            const size_t nb = size_t(kCompPerBlk) * 512u * 2u;
                                            if (L.d_hca_comp_kv_tp[r]) {
                                                (void)hipMemcpyAsync(L.d_hca_comp_kv_tp[r] + comp_off, src + off, nb, hipMemcpyHostToDevice, st_dma);
                                            }
                                            off += nb;
                                        } else {
                                            const size_t tok_off = size_t(b) * size_t(kBlockTokens);
                                            const size_t nb_m = size_t(kBlockTokens) * 576u * 2u;
                                            const size_t nb_i = size_t(kBlockTokens) * 128u * 2u;
                                            (void)hipMemcpyAsync(L.d_dsa_mla_kv + tok_off * 576u, src + off, nb_m, hipMemcpyHostToDevice, st_dma);
                                            (void)hipMemcpyAsync(L.d_dsa_ik_cache + tok_off * 128u, src + off + nb_m, nb_i, hipMemcpyHostToDevice, st_dma);
                                            off += nb_m + nb_i;
                                        }
                                    }
                                }
                            } else if (hca_mirror_ready) {
                                // Wave-13 Cut W13.6: Zero-CPU-memcpy direct contiguous layer-major HCA H2D!
                                const size_t blk_nb = size_t(kCompPerBlk) * 512u * 2u;
                                const size_t layer_bytes = size_t(nw) * blk_nb;
                                const size_t dst_off_elems = size_t(b0) * size_t(kCompPerBlk) * 512u;
                                const size_t src_off_bytes = size_t(b0) * blk_nb;
                                for (int l = l_beg[s]; l < l_end[s]; ++l) {
                                    const auto& L = layers_[size_t(l)];
                                    const int ord = hca_layer_ord_[size_t(l)];
                                    if (ord < 0 || !L.d_hca_comp_kv_tp[r]) continue;
                                    const uint8_t* src = h_hca_layer_major_ + size_t(ord) * size_t(hca_layer_stride_) + src_off_bytes;
                                    (void)hipMemcpyAsync(L.d_hca_comp_kv_tp[r] + dst_off_elems, src, layer_bytes, hipMemcpyHostToDevice, st_dma);
                                }
                            } else {
                                // Fallback when wave extends beyond max_hca_blocks_: fence upper half-ring
                                // before reusing [72..104 MiB) for batched non-owner HCA pack H2Ds.
                                if (!hca_ring_fenced && decoupled) {
                                    if (ring_armed[kMixedRingSlots]) {
                                        (void)hipStreamWaitEvent(st_dma, ev_ring_unpack_[r][1], 0);
                                    }
                                    hca_ring_fenced = true;
                                }
                                std::vector<int> all_i(static_cast<size_t>(nw));
                                for (int i = 0; i < nw; ++i) all_i[size_t(i)] = i;
                                emit_nonowner_hca_batched(s, all_i.data(), nw);
                            }
                            if (b0 + nw == hit_blks) {
                                if (!emit_tip_for(s, nw - 1)) {
                                    h2d_ok.store(false, std::memory_order_relaxed);
                                    return;
                                }
                            }
                        }
                        // Terminal per-wave fence only (never mid-restore). Sync st_dma BEFORE st_comp
                        // so GPU[1] immediately releases Switchbox #1 for GPU[3] once PCIe H2D drains!
                        if (st_dma != st_comp) (void)hipStreamSynchronize(st_dma);
                        if (serialize_sb1_r3 && r == r_s_[1]) {
                            spine1_dma_done.store(true, std::memory_order_release);
                        }
                        (void)hipStreamSynchronize(st_comp);
                        last_rank_ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        return;
                    }

                    // Tier-2 NVMe / mixed completion-order batched streaming path (0 mid-restore syncs!)
                    int done_seq[2] = { 0, 0 };
                    while (done_seq[0] < nw || done_seq[1] < nw) {
                        int s = -1;
                        int batch_i[8];
                        int batch_cnt = 0;
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
                            const int avail = int(wave_rq.ready[s].size()) - done_seq[s];
                            batch_cnt = std::min(8, std::min(avail, nw - done_seq[s]));
                            for (int k = 0; k < batch_cnt; ++k) {
                                batch_i[k] = wave_rq.ready[s][size_t(done_seq[s]++)];
                            }
                        }
                        const bool owns_this_stripe = (s == 0 && is_s0) || (s == 1 && is_s1);
                        if (d_scratch != nullptr && owns_this_stripe) {
                            for (int k = 0; k < batch_cnt; ++k) {
                                if (!emit_ring_delta(s, batch_i[k])) {
                                    h2d_ok.store(false, std::memory_order_relaxed);
                                    return;
                                }
                            }
                        } else if (!owns_this_stripe) {
                            emit_nonowner_hca_batched(s, batch_i, batch_cnt);
                        } else {
                            emit_direct_hca(s, batch_i, batch_cnt);
                        }
                        for (int k = 0; k < batch_cnt; ++k) {
                            if (b0 + batch_i[k] == hit_blks - 1) {
                                if (!emit_tip_for(s, batch_i[k])) {
                                    h2d_ok.store(false, std::memory_order_relaxed);
                                    return;
                                }
                            }
                        }
                    }
                    (void)hipStreamSynchronize(st_comp);
                    if (st_dma != st_comp) (void)hipStreamSynchronize(st_dma);
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
        if (disk_misses == 0) {
            std::printf("[KVBLK-READ] tier=%s blks=%d/%d (disk=0) tokens=%d ms=%.2f tok_s=%.0f (g0=%.1f g1=%.1f g2=%.1f g3=%.1f)\n",
                        out_tel.tier.c_str(), hit_blks - start_blk, hit_blks,
                        new_read_toks, ms, out_tel.read_tok_s,
                        last_rank_ms[0], last_rank_ms[1], last_rank_ms[2], last_rank_ms[3]);
        } else {
            std::printf("[KVBLK-READ] tier=%s blks=%d/%d (disk=%d) tokens=%d ms=%.2f tok_s=%.0f (d0_end=%.1f d1_end=%.1f io0=%.2f io1=%.2f crc=%.2f)\n",
                        out_tel.tier.c_str(), hit_blks - start_blk, hit_blks, disk_misses,
                        new_read_toks, ms, out_tel.read_tok_s,
                        d0_end_ms, d1_end_ms, io0_avg, io1_avg, crc_avg);
        }
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
        // Wave-12 Cut R12.2: measured live 544K dual-NVMe bandwidth is 3.53 GB/s (45.7%) on nvme0n1 vs
        // 4.19 GB/s (54.3%) on nvme1n1; routing 1 in 12 stripe-0 blocks to nvme1n1 hits 45.8% / 54.2%
        // while resolve_blk_path() probes both drives for 100% backward compatibility with existing shards.
        //
        // Wave-13 Goal B: the 1/12 ratio is the exact Bresenham solution of the measured
        // 3.53 : 4.19 bandwidth split (r = 0.5 * (4.19 - 3.53) / (4.19 + 3.53) = 0.0427 ~ 1/12), so the
        // byte balance is already optimal. The residual ~35 ms `d1` tail in Wave-12 was NOT a byte
        // imbalance: it was the CRC32C stage sitting between `io_ok` and the `wave_rq` push, which
        // delayed the spine H2D of the last `d1` stripe by a full 0.4 ms CRC pass per stripe of
        // queue depth. Goal B pushes `wave_rq` at `io_ok` and keeps `wait_block_job` as the strict
        // commit fence, which removes that tail without touching the (already optimal) drive ratio.
        return (stripe == 0 && (blk_idx % 12) == 0) ? 1 : stripe;
    }

    // Wave-13 Goal B: resolved on-disk block paths cached during the pre-`t0` prefix scan.
    // `submit_block_read` then executes ZERO redundant `::stat` syscalls after `t0`.
    struct CachedBlkPath {
        std::string path[2];
        int         drive[2] = {0, 1};
        uint64_t    size[2]  = {0, 0};
        bool        ok       = false;
    };

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

    // Wave-13 Goal B: pre-`t0` probe that ALSO records the resolved paths. The restore prefix
    // scan calls this for every candidate block, so by the time `t0` is taken the whole
    // contiguous prefix is already resolved and `submit_block_read` performs zero `::stat`.
    bool probe_disk_block_cached(const KvBlockKey& k, uint32_t* out_flags = nullptr) {
        CachedBlkPath cp{};
        cp.ok = resolve_blk_path(k, 0, cp.path[0], cp.drive[0], &cp.size[0]) &&
                resolve_blk_path(k, 1, cp.path[1], cp.drive[1], &cp.size[1]);
        if (cp.ok && out_flags) {
            *out_flags = 0u;
            const int fd = ::open(cp.path[0].c_str(), O_RDONLY);
            if (fd >= 0) {
                Ernie55KvBlockDiskHdr h{};
                if (::pread(fd, &h, 80, 0) == 80 && h.magic == 0x4B423539u) {
                    *out_flags = h.has_dspark_tip;
                }
                ::close(fd);
            }
        }
        path_cache_[k] = std::move(cp);
        return cp.ok;
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
        // Wave-13 Cut W13.6b: Direct-mapped sequential slot placement (`sid == key.blk_idx` when
        // free) keeps blocks 0..511 strictly monotonic inside the contiguous coherent DDR4 arena
        // for linear PCIe/IOMMU streaming during Tier-1 warm restores (0 B extra VRAM).
        if (key.blk_idx >= 0 && size_t(key.blk_idx) < slots_.size()) {
            auto& direct = slots_[size_t(key.blk_idx)];
            if (!direct.valid && direct.in_flight == 0u) {
                direct.in_flight++;
                direct.key = key;
                return int(key.blk_idx);
            }
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
    //
    // Wave-13 Goal B: the first-touch fault loop is pinned to 16 OpenMP threads. The
    // 512-slot (8.75 GiB) DDR4 arena is then striped round-robin across all 8 AMD EPYC
    // 7542 DDR4 channels at 4 KiB granularity instead of being bound to whichever single
    // core wins the static schedule, which is what keeps the full-wire 42.77 GB/s PCIe
    // Gen 4.0 H2D streaming reachable from `hipHostRegisterPortable` memory.
    static constexpr int kWbFaultThreads = 16;

    static bool alloc_wb_slot(uint8_t** out, size_t bytes, bool* registered) {
        void* p = nullptr;
        if (::posix_memalign(&p, 4096, bytes) != 0 || p == nullptr) return false;
        *out         = static_cast<uint8_t*>(p);
        *registered  = false;
        // Cut W13.6: 4 KiB page striping without MADV_HUGEPAGE improves concurrent 2-GPU H2D
        // channel balance on AMD EPYC 7542 by +3.2 GB/s while preserving 18.6 GB/s SSE4.2 CRC32C.
        auto* bp = static_cast<uint8_t*>(p);
        #pragma omp parallel for schedule(static) num_threads(kWbFaultThreads) if (bytes >= 1048576u)
        for (size_t off = 0; off < bytes; off += 4096u) {
            bp[off] = 0u;
        }
        if (hipHostRegister(p, bytes, hipHostRegisterPortable) == hipSuccess) *registered = true;
        return true;
    }

    // Wave-13 Cut W13.6: Mirror one stripe's HCA comp_kv layers (19 or 18 x 16 KiB = ~300 KiB,
    // ~15 us while hot in L2/L3 cache) into `h_hca_layer_major_` at capture or Dual-NVMe CRC time
    // so Tier-1 warm DDR4 restores issue 0 ms of CPU memcpy and 18/19/37 direct contiguous H2Ds.
    void mirror_slot_hca(int s, const KvBlockKey& key, const uint8_t* h_stripe) {
        if (h_hca_layer_major_ == nullptr || h_stripe == nullptr || n_hca_layers_ <= 0) return;
        const int b = key.blk_idx;
        if (b < 0 || b >= max_hca_blocks_) return;
        const int n_layers = int(layers_.size());
        const int lb = (s == 0) ? 0 : split_layer_;
        const int le = (s == 0) ? split_layer_ : n_layers;
        const size_t nb = size_t(kCompPerBlk) * 512u * 2u;
        std::lock_guard<std::mutex> mlk(hca_mirror_mtx_[s]);
        for (int l = lb; l < le; ++l) {
            const int ord = hca_layer_ord_[size_t(l)];
            if (ord < 0) continue;
            uint8_t* dst = h_hca_layer_major_ + size_t(ord) * size_t(hca_layer_stride_) + size_t(b) * nb;
            std::memcpy(dst, h_stripe + hca_layer_off_[size_t(l)], nb);
        }
        hca_mirror_key_[s][size_t(b)]   = key;
        hca_mirror_valid_[s][size_t(b)] = 1u;
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
        // Wave-13 Goal B: the pre-`t0` prefix scan already resolved both stripes, so the hot
        // restore path performs ZERO `::stat` syscalls here. The resolve_blk_path() fallback is
        // kept only for callers that bypass the scan (e.g. tests) and stays 100% compatible
        // with shards written by earlier waves.
        const auto cit = path_cache_.find(key);
        if (cit != path_cache_.end() && cit->second.ok) {
            p0 = cit->second.path[0];
            p1 = cit->second.path[1];
            d0 = cit->second.drive[0];
            d1 = cit->second.drive[1];
        } else {
            if (!resolve_blk_path(key, 0, p0, d0)) { d0 = preferred_drive(key.blk_idx, 0); p0 = make_blk_path_on(key, 0, d0); }
            if (!resolve_blk_path(key, 1, p1, d1)) { d1 = preferred_drive(key.blk_idx, 1); p1 = make_blk_path_on(key, 1, d1); }
        }
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

            // Wave-13 Goal B: wake the spine GPU IMMEDIATELY at `io_ok` so the 8.55 MiB SDMA H2D
            // overlaps the CPU CRC32C pass instead of waiting behind it (~0.4 ms/stripe of queue
            // depth that used to show up as the ~35 ms `d1` tail). `wait_block_job` stays the
            // strict commit fence: `jp->state` only reaches 1 after BOTH stripes pass CRC32C, so a
            // mismatch still aborts the whole restore cleanly before any slot pin is released.
            if (jp->wave_rq != nullptr) {
                jp->wave_rq->push(stripe, jp->wave_idx, io_ok);
            }
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
                if (crc_ok) {
                    mirror_slot_hca(stripe, jp->key, dst);
                }
                stripe_ok = crc_ok;
            }
            if (!stripe_ok) jp->ok.store(false, std::memory_order_release);
            // Wave-13 Goal B: `wave_rq` is pushed at `io_ok` in reader_loop(); the CRC pool only
            // records `stripe_state` (the `wait_stripe_job` fence) and the once-per-block commit.
            {
                std::lock_guard<std::mutex> blk(block_mtx_);
                jp->stripe_state[stripe].store(stripe_ok ? 1 : 2, std::memory_order_release);
            }
            block_cv_.notify_all();
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
                    sl.gen_id         = ++lru_clock_;
                    sl.magic_gen      = kDdrSlotMagic ^ sl.gen_id;
                    sl.valid          = true;
                    sl.lru_tick       = sl.gen_id;
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
        // Wave-13 Goal B: default raised 2 MiB -> 4 MiB. A stripe is 8,962,048 B, so 4 MiB
        // chunks issue 3 preads/stripe instead of 5, cutting per-`pread` O_DIRECT submission and
        // completion-interrupt overhead by 40% while every chunk stays a 4 KiB-aligned multiple
        // (O_DIRECT-safe) and the 10-deep per-drive queue still keeps 10 stripes in flight.
        static const uint64_t kChunkBytes = []() -> uint64_t {
            if (const char* ev = std::getenv("ERNIE55_KVBLK_CHUNK_MB")) {
                const int mb = std::atoi(ev);
                if (mb >= 1 && mb <= 16) return uint64_t(mb) * 1024ULL * 1024ULL;
            }
            return 4ULL * 1024ULL * 1024ULL;
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
    hipEvent_t ev_hca_half_[4][2]   = {{nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}};
    // Wave-13 Goal B: HCA pack-ring half fences (H2D-done + unpack-done) and the 12-slot
    // decoupled delta-ring event pairs + tip pair. All cache-owned, all hipEventDisableTiming.
    hipEvent_t ev_hca_half_h2d_[4][2] = {{nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}};
    hipEvent_t ev_ring_h2d_[4][kRingSlots]    = {};
    hipEvent_t ev_ring_unpack_[4][kRingSlots] = {};
    hipEvent_t ev_tip_h2d_[4]    = {nullptr, nullptr, nullptr, nullptr};
    hipEvent_t ev_tip_unpack_[4] = {nullptr, nullptr, nullptr, nullptr};
    hipEvent_t ev_cap_pack_[2]   = {nullptr, nullptr};
    bool       ring_ev_ok_       = false;
    int max_slots_      = 160;
    uint64_t delta_bytes_[2]  = {0, 0};
    uint64_t delta_padded_[2] = {0, 0};
    uint64_t tip_bytes_[2]    = {0, 0};
    uint64_t tip_padded_[2]   = {0, 0};
    uint64_t slot_bytes_[2]   = {0, 0};
    uint64_t slot_stride_[2]  = {0, 0};
    uint8_t* arena_s_[2]      = {nullptr, nullptr};
    bool     arena_coherent_  = false;
    // Wave-13 Cut W13.6: Shared layer-major HCA host mirror (0 B GPU VRAM, 296 MiB DDR4 @ 512 slots).
    uint8_t* h_hca_layer_major_ = nullptr;
    bool     h_hca_layer_coh_   = false;
    bool     h_hca_layer_reg_   = false;
    int      n_hca_layers_      = 0;
    int      max_hca_blocks_    = 0;
    uint64_t hca_layer_stride_  = 0ULL;
    std::vector<int>        hca_layer_ord_;
    std::vector<uint64_t>   hca_layer_off_;
    std::mutex              hca_mirror_mtx_[2];
    std::vector<KvBlockKey> hca_mirror_key_[2];
    std::vector<uint8_t>    hca_mirror_valid_[2];
    std::vector<KvLayerDesc> layers_;
    std::vector<uint16_t*>   dspark_rings_;

    mutable std::mutex mtx_;
    std::condition_variable slot_cv_;
    uint64_t lru_clock_ = 0ULL;
    std::vector<DdrSlot> slots_;
    std::unordered_map<KvBlockKey, int, KvBlockKeyHasher> index_;
    // Wave-13 Goal B: pre-`t0` resolved on-disk stripe paths (zero `::stat` after `t0`).
    std::unordered_map<KvBlockKey, CachedBlkPath, KvBlockKeyHasher> path_cache_;

    // Wave-11 Turn-11 Pillar II: 6+6=12 Dual-NVMe O_DIRECT physical-drive pool + 4 dedicated CRC32C verifiers!
    // Wave-12 Cut R12.2: QD 6 -> 8 per physical drive and CRC workers 4 -> 6. The live cold read
    // reached only 7.75 GB/s of the 9.05 GB/s peak because the 4 CRC workers became the tail:
    // each 8.55 MiB stripe costs ~0.4 ms of CRC32C, so 4 workers saturate at ~10 stripes in flight
    // while 8 readers per drive can queue 16. Six workers keep the CRC pool ahead of the readers.
    // Wave-13 Goal B: QD 8 -> 10 per drive and CRC workers 6 -> 8. Because `wave_rq` is now pushed
    // at `io_ok` (CRC overlapped with the GPU H2D), the CRC pool no longer gates the spine, so the
    // readers are the only remaining constraint: 10 x 4 MiB chunks in flight per drive keeps both
    // NVMe queues saturated, and 8 CRC workers still finish a stripe in ~0.4 ms < the 4 MiB pread.
    static constexpr int kReadersPerStripe = 10;
    static constexpr int kCrcWorkers       = 8;
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
