// SPDX-License-Identifier: Apache-2.0
#pragma once
// Lock-Free POSIX /dev/shm IPC Channel between MI50 Host (ernie55_runtime) and Tenstorrent Blackhole P150A
// Supports dynamic topologies:
//   - 192G: 2x Blackhole P150A FABRIC_1D TP2 (198 WHT128-HLUT4 experts/layer, 0 B CPU DDR4)
//   - 224G/240G: 3x Blackhole P150A EP3 (100 experts/layer/card = 300/layer, 0 AllReduce)
// Latent MoE geometry: H_lat = 2048, I_moe = 2048, top_k = 10, 100% uniform WHT128-HLUT4.

#include "../src/ernie55_config.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__HIP_DEVICE_COMPILE__)
#include <immintrin.h>
#endif

namespace ernie55 {

inline constexpr uint32_t ERNIE55_P150_SHM_MAGIC   = 0x45353550u;  // 'E55P'
inline constexpr uint32_t ERNIE55_P150_SHM_VERSION = 3u;
inline constexpr uint32_t ERNIE55_P150_MAX_ROWS    = 2048u;
inline constexpr uint32_t ERNIE55_P150_MAX_TOPK    = uint32_t(kNumExpertsPerTok);  // 10

inline constexpr uint32_t ERNIE55_P150_FLAG_X_F16  = 1u;  // input is native FP16 in x_rot_f16 (0 CPU f16->f32!)
inline constexpr uint32_t ERNIE55_P150_FLAG_Y_BF16 = 2u;  // output is native BF16 in y_out_bf16 (0 CPU bf16->f32 on worker!)
inline constexpr uint32_t ERNIE55_P150_FLAG_X_BF16 = 4u;  // input is native BF16 in x_rot_f16 (0 CPU f16->bf16 on worker!)

enum Ernie55P150Op : uint32_t {
    ERNIE55_P150_OP_MOE_EXEC = 1u,
    ERNIE55_P150_OP_PING     = 2u,
    ERNIE55_P150_OP_SHUTDOWN = 99u,
};

enum Ernie55P150Mode : uint32_t {
    ERNIE55_P150_MODE_TP2 = 0u,  // 2x P150A FABRIC_1D TP2 (I_shard = 1024)
    ERNIE55_P150_MODE_EP  = 1u,  // Nx P150A independent EP (I = 2048, zero AllReduce)
};

struct alignas(4096) Ernie55P150ShmChannel {
    uint32_t magic;
    uint32_t version;
    int32_t  worker_pid;
    uint32_t p150_mode;             // Ernie55P150Mode (0=TP2, 1=EP)
    uint32_t num_p150_cards;        // 2 (192G) or 3 (224G/240G)
    uint32_t n_experts_per_layer;   // total experts owned by P150 bank per MoE layer
    uint32_t expert_id_begin;       // global expert start [begin .. end)
    uint32_t expert_id_end;
    uint32_t ready;
    uint32_t fabric_active;
    uint32_t l1_hot_experts;
    uint32_t reserved0;
    uint64_t l1_pinned_bytes_total;
    uint64_t gddr6_bytes_per_card;

    alignas(64) std::atomic<uint32_t> req_seq;
    uint32_t op;
    uint32_t layer_idx;             // 1..43
    uint32_t rows;                  // M in [1..ERNIE55_P150_MAX_ROWS]
    uint32_t top_k;                 // 10
    uint32_t flags;                 // bitmask of ERNIE55_P150_FLAG_*

    alignas(64) std::atomic<uint32_t> resp_seq;
    int32_t  status;
    uint64_t last_device_ns;
    std::atomic<uint64_t> total_device_ns;
    std::atomic<uint64_t> stat_total_slots;
    std::atomic<uint64_t> stat_l1_hits;

    // Input: local P150 slot IDs in [0, n_experts_per_layer) or -1 if routed to MI50
    alignas(64) int32_t  row_slots[ERNIE55_P150_MAX_ROWS][ERNIE55_P150_MAX_TOPK];
    alignas(64) float    row_weights[ERNIE55_P150_MAX_ROWS][ERNIE55_P150_MAX_TOPK];
    // WHT128-rotated FP32 latent activations [M, 2048] (decode / legacy path)
    alignas(64) float    x_rot_f32[ERNIE55_P150_MAX_ROWS * kLatentDim];
    // Wave-9 Pass 6 / Wave-10 C1: Native FP16 or BF16 latent activations [M, 2048] -> 8 MiB (4K-aligned for direct DMA!)
    alignas(4096) uint16_t x_rot_f16[ERNIE55_P150_MAX_ROWS * kLatentDim];
    // Accumulated weighted expert latent output [M, 2048] FP32 (decode / legacy path)
    alignas(64) float    y_out_f32[ERNIE55_P150_MAX_ROWS * kLatentDim];
    // Wave-9 Pass 6 / Wave-10 C1: Native BF16 prefill expert latent output [M, 2048] -> 8 MiB (4K-aligned for direct DMA!)
    alignas(4096) uint16_t y_out_bf16[ERNIE55_P150_MAX_ROWS * kLatentDim];
};

inline void p150_cpu_relax() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__HIP_DEVICE_COMPILE__)
    _mm_pause();
#endif
}

class Ernie55P150ShmMapping {
public:
    Ernie55P150ShmChannel* ptr = nullptr;
    int fd = -1;
    bool owner = false;
    std::string shm_name;

    Ernie55P150ShmMapping() = default;
    ~Ernie55P150ShmMapping() { close_mapping(); }
    Ernie55P150ShmMapping(const Ernie55P150ShmMapping&) = delete;
    Ernie55P150ShmMapping& operator=(const Ernie55P150ShmMapping&) = delete;

    void close_mapping() noexcept {
        if (ptr) { ::munmap(ptr, sizeof(Ernie55P150ShmChannel)); ptr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
        if (owner && !shm_name.empty()) { ::shm_unlink(shm_name.c_str()); owner = false; }
    }

    bool create_server(const std::string& name = "/ernie55_p150_ipc") {
        close_mapping();
        shm_name = name;
        ::shm_unlink(shm_name.c_str());
        fd = ::shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        if (::ftruncate(fd, sizeof(Ernie55P150ShmChannel)) != 0) { close_mapping(); return false; }
        void* m = ::mmap(nullptr, sizeof(Ernie55P150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { close_mapping(); return false; }
        (void)::mlock(m, sizeof(Ernie55P150ShmChannel));
        std::memset(m, 0, sizeof(Ernie55P150ShmChannel));
        ptr = static_cast<Ernie55P150ShmChannel*>(m);
        owner = true;
        ptr->worker_pid = int32_t(::getpid());
        ptr->version = ERNIE55_P150_SHM_VERSION;
        ptr->req_seq.store(0u, std::memory_order_relaxed);
        ptr->resp_seq.store(0u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        ptr->magic = ERNIE55_P150_SHM_MAGIC;
        return true;
    }

    bool attach_client(const std::string& name = "/ernie55_p150_ipc") {
        close_mapping();
        shm_name = name;
        fd = ::shm_open(shm_name.c_str(), O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        void* m = ::mmap(nullptr, sizeof(Ernie55P150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { close_mapping(); return false; }
        auto* ch = static_cast<Ernie55P150ShmChannel*>(m);
        if (ch->magic != ERNIE55_P150_SHM_MAGIC || ch->version != ERNIE55_P150_SHM_VERSION || !ch->ready) {
            ::munmap(m, sizeof(Ernie55P150ShmChannel));
            ::close(fd);
            fd = -1;
            return false;
        }
        volatile const uint8_t* pb = static_cast<volatile const uint8_t*>(m);
        uint8_t sink = 0;
        for (size_t off = 0; off < sizeof(Ernie55P150ShmChannel); off += 4096u) sink ^= pb[off];
        (void)sink;
        ptr = ch;
        owner = false;
        return true;
    }
};

}  // namespace ernie55
