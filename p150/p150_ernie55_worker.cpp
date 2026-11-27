// SPDX-License-Identifier: Apache-2.0
// Pure C++20 Tenstorrent Blackhole P150A On-Device Tensix Worker for baidu/ERNIE-5.5-Flash (ernie5_5_moe)
// 100% uniform WHT128-HLUT4 (4.125 bpw, 8,448 B/superblock) + Native Blackhole SFPU SiTU-GLU(beta=4, lin=25).
// Zero host CPU expert math, zero Python anywhere!
// Dynamic topology support (zero code edits):
//   - 192G ("tp2"): 2x Blackhole P150A FABRIC_1D TP2 (I_shard=1024, 128 sb/proj, 64 cores/card, 34 L1 hot shards/card)
//   - 224G/240G ("ep3"): 3x Blackhole P150A independent EP cards (I=2048, 256 sb/proj, 64 cores/card, 0 AllReduce)

#include "ernie55_p150_shm.hpp"
#include "ernie55_p150_spec.hpp"
#include "../src/ernie55_hlut4_pack.hpp"

#include <ttnn/device.hpp>
#include <ttnn/cluster.hpp>
#include <ttnn/config.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/operations/generic/generic_op.hpp>
#include <ttnn/operations/ccl/all_reduce/all_reduce.hpp>
#include <ttnn/operations/ccl/broadcast/broadcast.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <immintrin.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ernie55 {
namespace {

using namespace tt::tt_metal;

volatile sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

struct TraceStorageContextGuard {
    TraceStorageContextGuard()  { tt::tt_metal::push_allocation_context("trace_storage"); }
    ~TraceStorageContextGuard() { tt::tt_metal::pop_allocation_context(); }
    TraceStorageContextGuard(const TraceStorageContextGuard&) = delete;
    TraceStorageContextGuard& operator=(const TraceStorageContextGuard&) = delete;
};

static constexpr uint32_t kBatchSlots        = 64u;  // Wave-10 C1: 64-slot batch halves prefill dispatches (9->5 passes/L)!
static constexpr uint32_t kMaxCompactSlots   = 640u;
static constexpr uint32_t kSlotIdsBytes      = 1344u;  // [0..1279]=640 uint16_t slot_ids, [1280..1343]=64 B ldesc
static constexpr uint32_t kL1SmallSizeBytes  = 0u;
static constexpr uint32_t kPrefillMaxRows    = 2048u;  // Wave-9 Pass 4: Single-pass 2048-row prefill (2x less GDDR6 weight streaming!)
static constexpr uint32_t kPageBytesHlut4    = 8448u;

// Verified via tt-smi + lspci/sysfs on SB-M5-P2-192G:
//   UMD chip_id 0 (mesh_coord {0, 0}) = 0000:0a:00.0 (PCIe Gen4 x4)
//   UMD chip_id 1 (mesh_coord {0, 1}) = 0000:41:00.0 (PCIe Gen4 x16)
// Hard-pin x16_relay to {0, 1} in TP2 mode so 100% of host H2D/D2H crosses the Gen4 x16 root port!
constexpr uint32_t detect_x16_mesh_col(bool is_tp2) noexcept { return is_tp2 ? 1u : 0u; }

inline uint32_t float_bits(float v) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return u;
}

void check_no_existing_p150_owner() {
    namespace fs = std::filesystem;
    const std::string self_pid = std::to_string(::getpid());
    for (const auto& proc : fs::directory_iterator("/proc")) {
        const auto pid = proc.path().filename().string();
        if (pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos || pid == self_pid) continue;
        std::error_code ec;
        fs::directory_iterator fd_it(proc.path() / "fd", fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;
        for (const auto& entry : fd_it) {
            const auto link = fs::read_symlink(entry.path(), ec);
            if (ec) { ec.clear(); continue; }
            if (link.string().rfind("/dev/tenstorrent/", 0) == 0) {
                throw std::runtime_error("Blackhole P150A already owned by PID " + pid);
            }
        }
    }
}

struct LockedBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    explicit LockedBuffer(size_t nbytes) : size((nbytes + 4095ULL) & ~4095ULL) {
        if (::posix_memalign(&ptr, 4096, size) != 0) throw std::bad_alloc();
        (void)::mlock(ptr, size);
        std::memset(ptr, 0, size);
    }
    ~LockedBuffer() {
        if (ptr) {
            (void)::munlock(ptr, size);
            ::free(ptr);
        }
    }
    LockedBuffer(const LockedBuffer&) = delete;
    LockedBuffer& operator=(const LockedBuffer&) = delete;
};

CoreRangeSet make_core_set(const std::vector<CoreCoord>& coords) {
    std::set<CoreRange> ranges;
    for (const auto& c : coords) ranges.emplace(c, c);
    return CoreRangeSet(ranges);
}

CBDescriptor make_cb(uint32_t id, uint32_t pages, uint32_t page_bytes, tt::DataFormat fmt, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = pages * page_bytes;
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{static_cast<uint8_t>(id), fmt, page_bytes, {}, {}});
    return cb;
}

std::vector<uint32_t> make_accessor_args(std::initializer_list<const ttnn::Tensor*> tensors, std::vector<uint32_t> prefix = {}) {
    std::vector<uint32_t> out = std::move(prefix);
    for (const auto* t : tensors) {
        const auto a = TensorAccessorArgs(t->buffer()).get_compile_time_args();
        out.insert(out.end(), a.begin(), a.end());
    }
    return out;
}

KernelDescriptor make_kernel(
    const std::string& path,
    const CoreRangeSet& cores,
    KernelDescriptor::ConfigDescriptor config,
    std::vector<uint32_t> ct_args = {})
{
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Missing P150 kernel file: " + path);
    }
    KernelDescriptor kd;
    kd.kernel_source = path;
    kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kd.core_ranges = cores;
    kd.config = std::move(config);
    kd.compile_time_args = std::move(ct_args);
    return kd;
}

ttnn::Tensor run_generic_preserving_topology(
    const std::vector<ttnn::Tensor>& tensors,
    const ProgramDescriptor& program)
{
    if (tensors.empty()) throw std::invalid_argument("run_generic_preserving_topology: empty tensor list");
    const auto topo = tensors.back().tensor_topology();
    auto res = ttnn::generic_op(tensors, program);
    if (res.buffer() != tensors.back().buffer()) {
        throw std::runtime_error("ttnn::generic_op replaced caller-owned output buffer");
    }
    res.update_tensor_topology(topo);
    return res;
}

ttnn::Tensor alloc_l1(distributed::MeshDevice& mesh, const Shape& shape, DataType dtype, Layout layout, bool sharded_axis0) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::L1_MEMORY_CONFIG)),
        &mesh,
        sharded_axis0
            ? TensorTopology::create_sharded_tensor_topology(mesh.shape(), 0)
            : TensorTopology::create_fully_replicated_tensor_topology(mesh.shape()));
}

ttnn::Tensor alloc_dram_sharded0(distributed::MeshDevice& mesh, const Shape& shape, DataType dtype, Layout layout) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)),
        &mesh,
        TensorTopology::create_sharded_tensor_topology(mesh.shape(), 0));
}

ttnn::Tensor alloc_dram_replicated(distributed::MeshDevice& mesh, const Shape& shape, DataType dtype, Layout layout) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)),
        &mesh,
        TensorTopology::create_fully_replicated_tensor_topology(mesh.shape()));
}

void write_sharded_pair_blocking(distributed::MeshDevice& mesh, ttnn::Tensor& dst, const void* tp0_data, const void* tp1_data) {
    auto borrowed = std::shared_ptr<distributed::MeshBuffer>(
        const_cast<distributed::MeshBuffer*>(&dst.mesh_buffer()), [](auto*) {});
    mesh.mesh_command_queue().enqueue_write_shards(
        borrowed,
        {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}.host_data(const_cast<void*>(tp0_data)),
         distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 1}}.host_data(const_cast<void*>(tp1_data))},
        true);
}

void write_single_shard_blocking(distributed::MeshDevice& mesh, ttnn::Tensor& dst, const void* data) {
    auto borrowed = std::shared_ptr<distributed::MeshBuffer>(
        const_cast<distributed::MeshBuffer*>(&dst.mesh_buffer()), [](auto*) {});
    mesh.mesh_command_queue().enqueue_write_shards(
        borrowed,
        {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}.host_data(const_cast<void*>(data))},
        true);
}

inline constexpr uint32_t tensix_tile_offset(int r, int c) noexcept {
    const int face = ((r >= 16) ? 2 : 0) + ((c >= 16) ? 1 : 0);
    return uint32_t(face * 256 + (r & 15) * 16 + (c & 15));
}

const std::array<std::array<uint16_t, 32>, 32>& tile_offset_lut() {
    static const auto lut = []() {
        std::array<std::array<uint16_t, 32>, 32> t{};
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 32; ++c)
                t[size_t(r)][size_t(c)] = static_cast<uint16_t>(tensix_tile_offset(r, c));
        return t;
    }();
    return lut;
}

inline void f32_row_add_avx2(const float* src_f32, float* dst_f32) noexcept {
    for (int j = 0; j < kLatentDim; j += 8) {
        _mm256_storeu_ps(dst_f32 + j, _mm256_add_ps(_mm256_loadu_ps(dst_f32 + j), _mm256_loadu_ps(src_f32 + j)));
    }
}

inline void f32_row_to_bf16_avx2(const float* src_f32, uint16_t* dst_bf16) noexcept {
    const __m256i v_7fff = _mm256_set1_epi32(0x00007FFFu);
    const __m256i v_one  = _mm256_set1_epi32(1);
    for (int j = 0; j < kLatentDim; j += 16) {
        const __m256i u0 = _mm256_castps_si256(_mm256_loadu_ps(src_f32 + j));
        const __m256i u1 = _mm256_castps_si256(_mm256_loadu_ps(src_f32 + j + 8));
        const __m256i r0 = _mm256_srli_epi32(_mm256_add_epi32(u0, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u0, 16), v_one))), 16);
        const __m256i r1 = _mm256_srli_epi32(_mm256_add_epi32(u1, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u1, 16), v_one))), 16);
        const __m256i pk = _mm256_packus_epi32(r0, r1);
        const __m256i ord = _mm256_permute4x64_epi64(pk, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bf16 + j), ord);
    }
}

inline void f16_row_to_bf16_avx2(const uint16_t* src_f16, uint16_t* dst_bf16) noexcept {
    const __m256i v_7fff = _mm256_set1_epi32(0x00007FFFu);
    const __m256i v_one  = _mm256_set1_epi32(1);
    for (int j = 0; j < kLatentDim; j += 16) {
        const __m256i u0 = _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src_f16 + j))));
        const __m256i u1 = _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src_f16 + j + 8))));
        const __m256i r0 = _mm256_srli_epi32(_mm256_add_epi32(u0, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u0, 16), v_one))), 16);
        const __m256i r1 = _mm256_srli_epi32(_mm256_add_epi32(u1, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u1, 16), v_one))), 16);
        const __m256i pk = _mm256_packus_epi32(r0, r1);
        const __m256i ord = _mm256_permute4x64_epi64(pk, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bf16 + j), ord);
    }
}

inline void bf16_row_add_bf16_avx2(const uint16_t* src_bf16, uint16_t* dst_bf16) noexcept {
    const __m256i v_7fff = _mm256_set1_epi32(0x00007FFFu);
    const __m256i v_one  = _mm256_set1_epi32(1);
    for (int j = 0; j < kLatentDim; j += 16) {
        const __m256 s0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src_bf16 + j))), 16));
        const __m256 d0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst_bf16 + j))), 16));
        const __m256 s1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src_bf16 + j + 8))), 16));
        const __m256 d1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst_bf16 + j + 8))), 16));
        const __m256i u0 = _mm256_castps_si256(_mm256_add_ps(s0, d0));
        const __m256i u1 = _mm256_castps_si256(_mm256_add_ps(s1, d1));
        const __m256i r0 = _mm256_srli_epi32(_mm256_add_epi32(u0, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u0, 16), v_one))), 16);
        const __m256i r1 = _mm256_srli_epi32(_mm256_add_epi32(u1, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u1, 16), v_one))), 16);
        const __m256i pk = _mm256_packus_epi32(r0, r1);
        const __m256i ord = _mm256_permute4x64_epi64(pk, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bf16 + j), ord);
    }
}

inline void bf16_row_to_f32_avx2(const uint16_t* src_bf16, float* dst_f32) noexcept {
    for (int j = 0; j < kLatentDim; j += 8) {
        const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_bf16 + j));
        const __m256i f8 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b8), 16);
        _mm256_storeu_ps(dst_f32 + j, _mm256_castsi256_ps(f8));
    }
}

inline void bf16_row_add_to_f32_avx2(const uint16_t* src_bf16, float* dst_f32) noexcept {
    for (int j = 0; j < kLatentDim; j += 8) {
        const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_bf16 + j));
        const __m256  f8 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b8), 16));
        _mm256_storeu_ps(dst_f32 + j, _mm256_add_ps(_mm256_loadu_ps(dst_f32 + j), f8));
    }
}

void build_wht128_block_matrix_local(int blk, const uint64_t* signs_tp, float q_mat[128][128]) {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    const uint64_t s0 = signs_tp[size_t(blk) * 2u + 0u];
    const uint64_t s1 = signs_tp[size_t(blk) * 2u + 1u];
    for (int r = 0; r < 128; ++r) {
        const bool neg = (r < 64) ? (((s0 >> r) & 1ULL) != 0ULL) : (((s1 >> (r - 64)) & 1ULL) != 0ULL);
        const float sr = neg ? -kInvSqrt128 : kInvSqrt128;
        for (int c = 0; c < 128; ++c) {
            const int pop = __builtin_popcount(unsigned(r & c));
            q_mat[r][c] = (pop & 1) ? -sr : sr;
        }
    }
}

void build_wht_tiles_bf16(const uint64_t* signs_blk, int num_blocks, std::vector<uint16_t>& out_tiles) {
    const auto& t_lut = tile_offset_lut();
    out_tiles.assign(size_t(num_blocks) * 16u * 1024u, 0u);
    float q_mat[128][128];
    for (int blk = 0; blk < num_blocks; ++blk) {
        build_wht128_block_matrix_local(blk, signs_blk, q_mat);
        for (int rt = 0; rt < 4; ++rt) {
            for (int ct = 0; ct < 4; ++ct) {
                uint16_t* dst = out_tiles.data() + (size_t(blk) * 16u + size_t(rt * 4 + ct)) * 1024u;
                for (int r = 0; r < 32; ++r)
                    for (int c = 0; c < 32; ++c)
                        dst[t_lut[size_t(r)][size_t(c)]] = f32_to_bf16_rne(q_mat[rt * 32 + r][ct * 32 + c]);
            }
        }
    }
}

uint32_t choose_decode_slot_bucket(uint32_t active) {
    // Wave-11 Turn-8 Pillar III: exact 1:1 buckets for active <= 16 eliminates 100% of dummy padded
    // decode slots (-2 to -4 ms/step, 100% bit-exact, well within the 128-entry trace cache).
    if (active <= 16u) return std::max<uint32_t>(1u, active);
    static constexpr uint32_t kBuckets[] = {18u, 20u, 24u, 28u, 32u, 40u, 48u, 64u};
    for (uint32_t b : kBuckets) if (active <= b) return b;
    return std::min<uint32_t>(((active + kBatchSlots - 1u) / kBatchSlots) * kBatchSlots, kMaxCompactSlots);
}

uint32_t choose_compact_slot_bucket(uint32_t active) {
    // Wave-11 Turn-8 Pillar I: uniform 16-slot buckets halve wasted prefill padding slots.
    constexpr uint32_t step = 16u;
    const uint32_t rounded = ((active + step - 1u) / step) * step;
    return std::min<uint32_t>(std::max<uint32_t>(16u, rounded), kMaxCompactSlots);
}

// ============================================================================
// Universal WHT128-HLUT4 Superblock Swizzlers (canonical Hlut4Tile32x32 [64x64] -> Tensix 8,448 B face pages)
// ============================================================================
void swizzle_hlut4_superblock_4x4(const Hlut4Tile32x32* tp16[4][4], uint8_t* pg) noexcept {
    const auto& t_lut = tile_offset_lut();
    uint32_t* w_nib0 = reinterpret_cast<uint32_t*>(pg + 0u);
    uint32_t* w_nib1 = reinterpret_cast<uint32_t*>(pg + 4096u);
    uint16_t* sc_dst = reinterpret_cast<uint16_t*>(pg + 8192u);
    for (int ks = 0; ks < 4; ++ks)
        for (int ns = 0; ns < 4; ++ns)
            std::memcpy(sc_dst + (ks * 4 + ns) * 8, tp16[ks][ns]->scales_f16, 16u);

    for (int rk = 0; rk < 32; ++rk) {
        const auto& roff = t_lut[size_t(rk)];
        const int wi_k = rk >> 3;
        const int sh_k = 4 * (rk & 7);
        for (int cn = 0; cn < 32; ++cn) {
            const int w_idx = cn * 4 + wi_k;
            uint32_t nib0 = 0u, nib1 = 0u;
            for (int ks = 0; ks < 2; ++ks) {
                for (int ns = 0; ns < 4; ++ns) {
                    const uint32_t nb0 = (tp16[ks][ns]->q_nib4[w_idx] >> sh_k) & 0xFu;
                    const uint32_t nb1 = (tp16[ks + 2][ns]->q_nib4[w_idx] >> sh_k) & 0xFu;
                    nib0 |= (nb0 << (4u * uint32_t(ks * 4 + ns)));
                    nib1 |= (nb1 << (4u * uint32_t(ks * 4 + ns)));
                }
            }
            const uint32_t tidx = roff[size_t(cn)];
            w_nib0[tidx] = nib0;
            w_nib1[tidx] = nib1;
        }
    }
}

void pack_hlut4_expert_tp2(const Hlut4Tile32x32* tiles, bool is_down, uint8_t* tp0_dst, uint8_t* tp1_dst) noexcept {
    constexpr int kt_full = 64;
    const int nt4 = is_down ? 16 : 8;
    const int kt4 = is_down ? 8  : 16;
    for (int tp = 0; tp < 2; ++tp) {
        uint8_t* dst = (tp == 0) ? tp0_dst : tp1_dst;
        for (int nb4 = 0; nb4 < nt4; ++nb4) {
            for (int kb4 = 0; kb4 < kt4; ++kb4) {
                const int nt_base = (is_down ? nb4 : (tp * 8 + nb4)) * 4;
                const int kt_base = (is_down ? (tp * 8 + kb4) : kb4) * 4;
                const Hlut4Tile32x32* tp16[4][4];
                for (int ks = 0; ks < 4; ++ks)
                    for (int ns = 0; ns < 4; ++ns)
                        tp16[ks][ns] = &tiles[size_t(nt_base + ns) * size_t(kt_full) + size_t(kt_base + ks)];
                swizzle_hlut4_superblock_4x4(tp16, dst + size_t(nb4 * kt4 + kb4) * kPageBytesHlut4);
            }
        }
    }
}

void pack_hlut4_expert_ep(const Hlut4Tile32x32* tiles, uint8_t* dst) noexcept {
    constexpr int kt_full = 64;
    for (int nb4 = 0; nb4 < 16; ++nb4) {
        for (int kb4 = 0; kb4 < 16; ++kb4) {
            const int nt_base = nb4 * 4;
            const int kt_base = kb4 * 4;
            const Hlut4Tile32x32* tp16[4][4];
            for (int ks = 0; ks < 4; ++ks)
                for (int ns = 0; ns < 4; ++ns)
                    tp16[ks][ns] = &tiles[size_t(nt_base + ns) * size_t(kt_full) + size_t(kt_base + ks)];
            swizzle_hlut4_superblock_4x4(tp16, dst + size_t(nb4 * 16 + kb4) * kPageBytesHlut4);
        }
    }
}

struct WorkerConfig {
    std::string weight_root     = "./checkpoints/active";
    std::string topology_path   = "./config/topology_192g.json";
    std::string config_path     = "./meta/bf16/config.json";
    std::string kernels_dir     = "./p150/kernels";
    std::string shm_name        = "/ernie55_p150_ipc";
    int group_id                = 0;
    bool enable_trace           = true;
    bool x16_relay              = true;
    bool self_test              = false;
    bool synthetic_self_test    = false;
    int max_layers_load         = kNumMoeTotalWithDspark; // 45 (L01..L42 backbone + L43..L45 3L DSpark)
    float situ_beta             = kSituBeta;
    float situ_linear_beta      = kSituLinearBeta;
};

struct ResidentLayerBank {
    uint32_t layer = 0;
    uint32_t n_experts = 0;
    ttnn::Tensor gate_pages;
    ttnn::Tensor up_pages;
    ttnn::Tensor down_pages;
    ttnn::Tensor wht_q_i;
    std::array<uint16_t, 256> hot_slot_for_eid = []() {
        std::array<uint16_t, 256> a{};
        a.fill(0xFFFFu);
        return a;
    }();
};

struct TraceSlot {
    std::optional<tt::tt_metal::distributed::MeshTraceId> id;
    bool warmed  = false;
    bool skipped = false;
};

// ============================================================================
// Unified Blackhole P150A Engine Instance
//   is_tp2 == true  : 1x2 MeshDevice (FABRIC_1D TP2, 2 cards, I_shard=1024, 128 sb/proj)
//   is_tp2 == false : 1x1 MeshDevice (single independent EP card, I=2048, 256 sb/proj, 0 AllReduce)
// ============================================================================
class P150Ernie55Mesh {
public:
    P150Ernie55Mesh(
        const WorkerConfig& cfg,
        bool is_tp2,
        std::vector<int> device_ids,
        int expert_begin,
        int expert_end,
        uint32_t trace_mib)
        : cfg_(cfg),
          is_tp2_(is_tp2),
          device_ids_(std::move(device_ids)),
          expert_begin_(expert_begin),
          expert_end_(expert_end),
          n_experts_(uint32_t(std::max(0, expert_end - expert_begin))),
          sb_per_proj_(is_tp2 ? kP150Tp2SbPerProj : kP150EpSbPerProj),
          inter_tiles_(is_tp2 ? kP150Tp2InterTiles : kP150EpInterTiles),
          nt4_gu_(is_tp2 ? 8u : 16u),
          kt4_dn_(is_tp2 ? 8u : 16u),
          k_splits_gu_(is_tp2 ? 8u : 4u),
          k_splits_dn_(4u),
          num_sb_core_(is_tp2 ? 2u : 4u),
          beta_bits_(float_bits(cfg.situ_beta)),
          beta_recip_bits_(float_bits(1.0f / cfg.situ_beta)),
          lin_bits_(float_bits(cfg.situ_linear_beta)),
          lin_recip_bits_(float_bits(1.0f / cfg.situ_linear_beta)),
          x16_coord_(0, detect_x16_mesh_col(is_tp2))
    {
        const auto dispatch = ttnn::device::create_cluster_aware_dispatch_config();
        trace_region_bytes_ = cfg_.enable_trace ? (trace_mib << 20) : 0u;
        const ttnn::MeshShape mshape = is_tp2_ ? ttnn::MeshShape{1, 2} : ttnn::MeshShape{1, 1};
        mesh_ = ttnn::distributed::open_mesh_device(
            mshape, kL1SmallSizeBytes, trace_region_bytes_, 1, dispatch, std::nullopt, device_ids_);

        grid_ = mesh_->compute_with_storage_grid_size();
        total_cores_ = uint32_t(grid_.x * grid_.y);
        if (total_cores_ < 64u) {
            throw std::runtime_error("Blackhole P150A grid too small for 64-core ERNIE-5.5: " + std::to_string(total_cores_));
        }

        const uint32_t inter_dim = inter_tiles_ * 32u;
        x_tilized_           = alloc_dram_replicated(*mesh_, Shape({1, kBatchSlots, 32u, kP150LatentK}), DataType::BFLOAT16, Layout::TILE);
        x_decode_src_        = alloc_dram_replicated(*mesh_, Shape({1, 1u, 32u, kP150LatentK}), DataType::BFLOAT16, Layout::TILE);
        slot_ids_src_        = alloc_dram_replicated(*mesh_, Shape({1, 1, 1, kSlotIdsBytes / 2u}), DataType::UINT16, Layout::ROW_MAJOR);
        weights_decode64_src_= alloc_dram_replicated(*mesh_, Shape({1, 64u, 32u, 32u}), DataType::FLOAT32, Layout::TILE);
        weights_tiled_src_   = alloc_dram_replicated(*mesh_, Shape({1, kMaxCompactSlots, 32u, 32u}), DataType::FLOAT32, Layout::TILE);
        weights_tiled_       = weights_tiled_src_;
        gate_partials_       = alloc_dram_sharded0(*mesh_, Shape({1, k_splits_gu_, kBatchSlots * 32u, inter_dim}), DataType::FLOAT32, Layout::TILE);
        up_partials_         = alloc_dram_sharded0(*mesh_, Shape({1, k_splits_gu_, kBatchSlots * 32u, inter_dim}), DataType::FLOAT32, Layout::TILE);
        z_rot_bf16_          = alloc_dram_sharded0(*mesh_, Shape({1, 1, kBatchSlots * 32u, inter_dim}), DataType::BFLOAT16, Layout::TILE);
        down_partials_       = gate_partials_;  // Exact 16 MiB alias in BOTH TP2 (8*16*32) and EP (4*16*64) == 4*16*64!
        combined_f32_        = alloc_dram_sharded0(*mesh_, Shape({1, 1, 32u, kP150LatentK}), DataType::FLOAT32, Layout::TILE);
        x_rm_src_            = alloc_dram_replicated(*mesh_, Shape({1, 1, kPrefillMaxRows, kP150LatentK}), DataType::BFLOAT16, Layout::TILE);
        slot_meta_src_       = alloc_dram_replicated(*mesh_, Shape({1, 1, kMaxCompactSlots, 64u}), DataType::UINT32, Layout::ROW_MAJOR);
        y_accum_f32_         = alloc_dram_sharded0(*mesh_, Shape({1, 1, kPrefillMaxRows, kP150LatentK}), DataType::FLOAT32, Layout::TILE);
        l1_lut_              = alloc_dram_replicated(*mesh_, Shape({1, uint32_t(kTotalLayersWithEcho), 256u}), DataType::UINT16, Layout::ROW_MAJOR);
        dummy_hot_           = alloc_dram_sharded0(*mesh_, Shape({1, 1u, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);

        if (!is_tp2_) {
            x_decode_         = x_decode_src_;
            slot_ids_         = slot_ids_src_;
            weights_decode64_ = weights_decode64_src_;
            weights_tiled_    = weights_tiled_src_;
            slot_meta_dst_    = slot_meta_src_;
            x_rm_dst_         = x_rm_src_;
        }

        // Pin ultra-hot WHT128-HLUT4 experts into on-die Tensix L1 SRAM while guaranteeing 256 KiB/core CB guard!
        const uint32_t max_try = is_tp2_ ? kP150MaxHotL1Tp2 : kP150MaxHotL1Ep;
        const uint32_t guard_pages = (total_cores_ * kP150Fabric1DCbGuardPerCore) / 4096u;
        for (uint32_t k_try = max_try; k_try >= 2u; --k_try) {
            try {
                auto hg = alloc_l1(*mesh_, Shape({1, k_try * sb_per_proj_, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto hu = alloc_l1(*mesh_, Shape({1, k_try * sb_per_proj_, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto hd = alloc_l1(*mesh_, Shape({1, k_try * sb_per_proj_, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto cb_guard = alloc_l1(*mesh_, Shape({1, guard_pages, 1024u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                (void)cb_guard;
                hot_gate_l1_ = std::move(hg);
                hot_up_l1_   = std::move(hu);
                hot_down_l1_ = std::move(hd);
                num_hot_shards_ = k_try;
                pinned_hot_l1_bytes_per_card_ = hot_gate_l1_->buffer()->size() + hot_up_l1_->buffer()->size() + hot_down_l1_->buffer()->size();
                break;
            } catch (...) {
                hot_gate_l1_.reset();
                hot_up_l1_.reset();
                hot_down_l1_.reset();
            }
        }

        std::memset(host_slot_lut_.ptr, 0xFF, host_slot_lut_.size);
        const distributed::MeshCoordinateRange all_devs(mesh_->shape());
        auto& cq = mesh_->mesh_command_queue();
        cq.enqueue_write_shard_to_sub_grid(l1_lut_->mesh_buffer(), host_slot_lut_.ptr, all_devs, false, BufferRegion(0, l1_lut_->buffer()->size()));

        if (is_tp2_) {
            run_bcast_decode64_untraced();
            run_bcast_prefill_untraced();
        }
        run_reduce_decode_untraced();
        run_reduce_prefill_untraced();
        cq.finish();

        std::printf("[P150-ERNIE55] Opened %s Blackhole mesh (grid=%ux%u=%u cores, experts=[%d..%d) (%u/L), "
                    "sb/proj=%u, hot_L1=%u shards/card [%.2f MiB total], SiTU(%.1f,%.1f), max_rows=%u, x16_coord={0,%u}, trace=%d)\n",
                    is_tp2_ ? "1x2 TP2 FABRIC_1D" : "1x1 EP",
                    unsigned(grid_.x), unsigned(grid_.y), total_cores_,
                    expert_begin_, expert_end_, n_experts_, sb_per_proj_,
                    num_hot_shards_, double(pinned_hot_l1_bytes_per_card_ * uint64_t(is_tp2_ ? 2u : 1u)) / 1048576.0,
                    cfg_.situ_beta, cfg_.situ_linear_beta, kPrefillMaxRows, unsigned(x16_coord_[1]), int(cfg_.enable_trace));
        std::fflush(stdout);
    }

    ~P150Ernie55Mesh() {
        if (mesh_) {
            mesh_->mesh_command_queue().finish();
            for (auto& [k, tr] : op_traces_) { if (tr.id) mesh_->release_mesh_trace(*tr.id); }
            for (auto& tr : reduce_traces_)  { if (tr.id) mesh_->release_mesh_trace(*tr.id); }
            for (auto& tr : bcast_traces_)   { if (tr.id) mesh_->release_mesh_trace(*tr.id); }
            hot_down_l1_.reset();
            hot_up_l1_.reset();
            hot_gate_l1_.reset();
            dummy_hot_.reset();
            l1_lut_.reset();
            y_accum_f32_.reset();
            slot_meta_dst_.reset();
            slot_meta_src_.reset();
            x_rm_dst_.reset();
            x_rm_src_.reset();
            for (auto& t : finish_decode_) t = ttnn::Tensor();
            for (auto& t : finish_prefill_) t = ttnn::Tensor();
            combined_f32_ = ttnn::Tensor();
            down_partials_ = ttnn::Tensor();
            z_rot_bf16_ = ttnn::Tensor();
            up_partials_ = ttnn::Tensor();
            gate_partials_ = ttnn::Tensor();
            weights_tiled_ = ttnn::Tensor();
            weights_decode64_ = ttnn::Tensor();
            slot_ids_ = ttnn::Tensor();
            x_decode_ = ttnn::Tensor();
            weights_tiled_src_ = ttnn::Tensor();
            weights_decode64_src_ = ttnn::Tensor();
            slot_ids_src_ = ttnn::Tensor();
            x_decode_src_ = ttnn::Tensor();
            x_tilized_ = ttnn::Tensor();
            for (auto& l : layers_) l.reset();
            mesh_->mesh_command_queue().finish();
            ttnn::distributed::close_mesh_device(mesh_);
        }
    }

    uint32_t n_experts() const noexcept { return n_experts_; }
    uint32_t num_hot_shards() const noexcept { return num_hot_shards_; }
    uint64_t pinned_l1_bytes_total() const noexcept { return pinned_hot_l1_bytes_per_card_ * uint64_t(is_tp2_ ? 2u : 1u); }
    uint64_t gddr6_bytes_per_card() const noexcept { return gddr6_bytes_per_card_; }
    uint64_t stat_total_slots() const noexcept { return stat_total_slots_; }
    uint64_t stat_l1_hits() const noexcept { return stat_l1_hits_; }

    void load_synthetic_layer1(const std::vector<Hlut4Tile32x32>& all_exp_tiles,
                               const std::vector<uint64_t>& signs_inter) {
        const size_t exp_card_b = size_t(sb_per_proj_) * kPageBytesHlut4;
        const size_t bank_b = size_t(n_experts_) * exp_card_b;
        std::vector<uint8_t> g0(bank_b), g1(is_tp2_ ? bank_b : 0u);
        std::vector<uint8_t> u0(bank_b), u1(is_tp2_ ? bank_b : 0u);
        std::vector<uint8_t> d0(bank_b), d1(is_tp2_ ? bank_b : 0u);
        constexpr size_t kExpTiles3 = 3u * kExpProjTiles;

        #pragma omp parallel for schedule(static)
        for (uint32_t slot = 0u; slot < n_experts_; ++slot) {
            const Hlut4Tile32x32* base = all_exp_tiles.data() + size_t(slot) * kExpTiles3;
            const size_t off = size_t(slot) * exp_card_b;
            if (is_tp2_) {
                pack_hlut4_expert_tp2(base + 0u * kExpProjTiles, false, g0.data() + off, g1.data() + off);
                pack_hlut4_expert_tp2(base + 1u * kExpProjTiles, false, u0.data() + off, u1.data() + off);
                pack_hlut4_expert_tp2(base + 2u * kExpProjTiles, true,  d0.data() + off, d1.data() + off);
            } else {
                pack_hlut4_expert_ep(base + 0u * kExpProjTiles, g0.data() + off);
                pack_hlut4_expert_ep(base + 1u * kExpProjTiles, u0.data() + off);
                pack_hlut4_expert_ep(base + 2u * kExpProjTiles, d0.data() + off);
            }
        }
        upload_layer_bank(1, g0, g1, u0, u1, d0, d1, signs_inter.data());
        pin_hot_from_layer1(g0, g1, u0, u1, d0, d1);
        prewarm_and_capture_traces(*layers_[1]);
    }

    void load_layers(int first_layer, int last_layer) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t exp_card_b = size_t(sb_per_proj_) * kPageBytesHlut4;
        const size_t arena_b = size_t(num_hot_shards_) * exp_card_b;
        std::vector<uint8_t> hg0(arena_b, 0u), hg1(is_tp2_ ? arena_b : 0u, 0u);
        std::vector<uint8_t> hu0(arena_b, 0u), hu1(is_tp2_ ? arena_b : 0u, 0u);
        std::vector<uint8_t> hd0(arena_b, 0u), hd1(is_tp2_ ? arena_b : 0u, 0u);
        std::vector<std::pair<uint16_t, uint16_t>> hot_list;
        hot_list.reserve(num_hot_shards_);

        const int total_layers = last_layer - first_layer + 1;
        uint32_t loaded = 0u;
        for (int l = first_layer; l <= last_layer && !g_stop; ++l) {
            load_one_layer_file(l, hg0, hg1, hu0, hu1, hd0, hd1, hot_list, total_layers);
            ++loaded;
            if (loaded == 1u || loaded == uint32_t(total_layers) || (loaded % 7u) == 0u) {
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("[P150-ERNIE55] Uploaded %u/%d layers (%.2f GiB/card GDDR6) in %.1fs\n",
                            loaded, total_layers, double(gddr6_bytes_per_card_) / 1073741824.0, sec);
                std::fflush(stdout);
            }
        }

        if (num_hot_shards_ > 0u && !hot_list.empty() && hot_gate_l1_.has_value()) {
            if (is_tp2_) {
                write_sharded_pair_blocking(*mesh_, *hot_gate_l1_, hg0.data(), hg1.data());
                write_sharded_pair_blocking(*mesh_, *hot_up_l1_,   hu0.data(), hu1.data());
                write_sharded_pair_blocking(*mesh_, *hot_down_l1_, hd0.data(), hd1.data());
            } else {
                write_single_shard_blocking(*mesh_, *hot_gate_l1_, hg0.data());
                write_single_shard_blocking(*mesh_, *hot_up_l1_,   hu0.data());
                write_single_shard_blocking(*mesh_, *hot_down_l1_, hd0.data());
            }
            auto* lut_u16 = static_cast<uint16_t*>(host_slot_lut_.ptr);
            std::memset(lut_u16, 0xFF, uint32_t(kTotalLayersWithEcho) * 256u * sizeof(uint16_t));
            for (size_t k = 0; k < hot_list.size(); ++k) {
                lut_u16[size_t(hot_list[k].first) * 256u + size_t(hot_list[k].second)] = uint16_t(k);
                if (layers_[hot_list[k].first].has_value()) {
                    layers_[hot_list[k].first]->hot_slot_for_eid[hot_list[k].second] = uint16_t(k);
                }
            }
            const distributed::MeshCoordinateRange all_devs(mesh_->shape());
            mesh_->mesh_command_queue().enqueue_write_shard_to_sub_grid(
                l1_lut_->mesh_buffer(), lut_u16, all_devs, true, BufferRegion(0, l1_lut_->buffer()->size()));
            std::printf("[P150-ERNIE55] Pinned %zu/%u hot WHT128-HLUT4 shards in Tensix L1 SRAM (%.2f MiB total)\n",
                        hot_list.size(), num_hot_shards_,
                        double(hot_list.size() * exp_card_b * uint64_t(is_tp2_ ? 6u : 3u)) / 1048576.0);
            std::fflush(stdout);
        }

        if (loaded > 0u && layers_[size_t(first_layer)].has_value()) {
            prewarm_and_capture_traces(*layers_[size_t(first_layer)]);
        }
    }

    // 100% On-Device Blackhole Tensix Forward Pass (x_rot is ALREADY WHT128-rotated [rows, 2048])
    void forward_layer(
        int layer_idx,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][ERNIE55_P150_MAX_TOPK],
        const float row_weights[][ERNIE55_P150_MAX_TOPK],
        const float* x_rot_f32,
        float* y_out_f32,
        bool accumulate_into_y = false,
        const uint16_t* x_rot_f16 = nullptr,
        uint16_t* y_out_bf16 = nullptr,
        const uint16_t* x_rot_bf16 = nullptr)
    {
        if (layer_idx < 1 || layer_idx >= kTotalLayersWithEcho || !layers_[size_t(layer_idx)].has_value()) {
            throw std::runtime_error("P150Ernie55Mesh: layer not loaded: " + std::to_string(layer_idx));
        }
        if (rows == 0u || n_experts_ == 0u) {
            if (!accumulate_into_y && rows > 0u) {
                if (y_out_bf16) std::memset(y_out_bf16, 0, size_t(rows) * kLatentDim * sizeof(uint16_t));
                else            std::memset(y_out_f32,  0, size_t(rows) * kLatentDim * sizeof(float));
            }
            return;
        }

        const auto& bank = *layers_[size_t(layer_idx)];
        if (rows <= 32u) {
            if (!accumulate_into_y) std::memset(y_out_f32, 0, size_t(rows) * kLatentDim * sizeof(float));
            const uint16_t* x_b16_in = x_rot_bf16;
            if (!x_b16_in) {
                x_bf16_scratch_.resize(size_t(rows) * kLatentDim);
                for (uint32_t r = 0u; r < rows; ++r) {
                    if (x_rot_f16) f16_row_to_bf16_avx2(x_rot_f16 + size_t(r) * kLatentDim, x_bf16_scratch_.data() + size_t(r) * kLatentDim);
                    else           f32_row_to_bf16_avx2(x_rot_f32 + size_t(r) * kLatentDim, x_bf16_scratch_.data() + size_t(r) * kLatentDim);
                }
                x_b16_in = x_bf16_scratch_.data();
            }
            run_decode_small(bank, rows, top_k, row_slots, row_weights, x_b16_in, y_out_f32);
            if (y_out_bf16) {
                for (uint32_t r = 0u; r < rows; ++r) {
                    f32_row_to_bf16_avx2(y_out_f32 + size_t(r) * kLatentDim, y_out_bf16 + size_t(r) * kLatentDim);
                }
            }
        } else {
            // Wave-10 C1: Zero-copy native BF16 DMA directly from 4K-aligned SHM when FLAG_X_BF16 is set (cv=0.00 ms!)
            void* x_dma_ptr = host_x_rm_.ptr;
            if (x_rot_bf16) {
                x_dma_ptr = const_cast<uint16_t*>(x_rot_bf16);
            } else {
                const auto t_cv0 = std::chrono::steady_clock::now();
                auto* x_rm_bf16 = static_cast<uint16_t*>(host_x_rm_.ptr);
                if (x_rot_f16) {
                    #pragma omp parallel for schedule(static) num_threads(16) if (rows >= 128u)
                    for (uint32_t r = 0u; r < rows; ++r) {
                        f16_row_to_bf16_avx2(x_rot_f16 + size_t(r) * kLatentDim, x_rm_bf16 + size_t(r) * kLatentDim);
                    }
                } else {
                    #pragma omp parallel for schedule(static) num_threads(16) if (rows >= 128u)
                    for (uint32_t r = 0u; r < rows; ++r) {
                        f32_row_to_bf16_avx2(x_rot_f32 + size_t(r) * kLatentDim, x_rm_bf16 + size_t(r) * kLatentDim);
                    }
                }
                prof_cv_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cv0).count();
            }
            run_compact_prefill(bank, rows, top_k, row_slots, row_weights, y_out_f32, accumulate_into_y, y_out_bf16, x_dma_ptr);
        }
    }

private:
    CoreCoord core_at(uint32_t idx) const noexcept {
        return CoreCoord(idx % uint32_t(grid_.x), idx / uint32_t(grid_.x));
    }

    void upload_layer_bank(
        int layer_idx,
        const std::vector<uint8_t>& g0, const std::vector<uint8_t>& g1,
        const std::vector<uint8_t>& u0, const std::vector<uint8_t>& u1,
        const std::vector<uint8_t>& d0, const std::vector<uint8_t>& d1,
        const uint64_t* signs_inter)
    {
        ResidentLayerBank bank;
        bank.layer = uint32_t(layer_idx);
        bank.n_experts = n_experts_;
        const Shape pshape({1, n_experts_ * sb_per_proj_, kPageBytesHlut4 / 4u});
        bank.gate_pages = alloc_dram_sharded0(*mesh_, pshape, DataType::UINT32, Layout::ROW_MAJOR);
        bank.up_pages   = alloc_dram_sharded0(*mesh_, pshape, DataType::UINT32, Layout::ROW_MAJOR);
        bank.down_pages = alloc_dram_sharded0(*mesh_, pshape, DataType::UINT32, Layout::ROW_MAJOR);

        const uint32_t wht_blks = is_tp2_ ? 8u : 16u;
        bank.wht_q_i = alloc_dram_sharded0(*mesh_, Shape({1, wht_blks, 128u, 128u}), DataType::BFLOAT16, Layout::TILE);

        if (is_tp2_) {
            write_sharded_pair_blocking(*mesh_, bank.gate_pages, g0.data(), g1.data());
            write_sharded_pair_blocking(*mesh_, bank.up_pages,   u0.data(), u1.data());
            write_sharded_pair_blocking(*mesh_, bank.down_pages, d0.data(), d1.data());
            std::vector<uint16_t> q0, q1;
            build_wht_tiles_bf16(signs_inter + 0u,  8, q0);
            build_wht_tiles_bf16(signs_inter + 16u, 8, q1);
            write_sharded_pair_blocking(*mesh_, bank.wht_q_i, q0.data(), q1.data());
        } else {
            write_single_shard_blocking(*mesh_, bank.gate_pages, g0.data());
            write_single_shard_blocking(*mesh_, bank.up_pages,   u0.data());
            write_single_shard_blocking(*mesh_, bank.down_pages, d0.data());
            std::vector<uint16_t> q0;
            build_wht_tiles_bf16(signs_inter, 16, q0);
            write_single_shard_blocking(*mesh_, bank.wht_q_i, q0.data());
        }

        gddr6_bytes_per_card_ += uint64_t(g0.size() + u0.size() + d0.size()) + uint64_t(wht_blks) * 16u * 2048u;
        layers_[size_t(layer_idx)] = std::move(bank);
    }

    void pin_hot_from_layer1(
        const std::vector<uint8_t>& g0, const std::vector<uint8_t>& g1,
        const std::vector<uint8_t>& u0, const std::vector<uint8_t>& u1,
        const std::vector<uint8_t>& d0, const std::vector<uint8_t>& d1)
    {
        if (num_hot_shards_ == 0u || !hot_gate_l1_.has_value()) return;
        const size_t exp_card_b = size_t(sb_per_proj_) * kPageBytesHlut4;
        const uint32_t pin_n = std::min<uint32_t>(num_hot_shards_, n_experts_);
        const size_t arena_b = size_t(num_hot_shards_) * exp_card_b;
        std::vector<uint8_t> hg0(arena_b, 0u), hg1(is_tp2_ ? arena_b : 0u, 0u);
        std::vector<uint8_t> hu0(arena_b, 0u), hu1(is_tp2_ ? arena_b : 0u, 0u);
        std::vector<uint8_t> hd0(arena_b, 0u), hd1(is_tp2_ ? arena_b : 0u, 0u);
        std::memcpy(hg0.data(), g0.data(), size_t(pin_n) * exp_card_b);
        std::memcpy(hu0.data(), u0.data(), size_t(pin_n) * exp_card_b);
        std::memcpy(hd0.data(), d0.data(), size_t(pin_n) * exp_card_b);
        if (is_tp2_) {
            std::memcpy(hg1.data(), g1.data(), size_t(pin_n) * exp_card_b);
            std::memcpy(hu1.data(), u1.data(), size_t(pin_n) * exp_card_b);
            std::memcpy(hd1.data(), d1.data(), size_t(pin_n) * exp_card_b);
            write_sharded_pair_blocking(*mesh_, *hot_gate_l1_, hg0.data(), hg1.data());
            write_sharded_pair_blocking(*mesh_, *hot_up_l1_,   hu0.data(), hu1.data());
            write_sharded_pair_blocking(*mesh_, *hot_down_l1_, hd0.data(), hd1.data());
        } else {
            write_single_shard_blocking(*mesh_, *hot_gate_l1_, hg0.data());
            write_single_shard_blocking(*mesh_, *hot_up_l1_,   hu0.data());
            write_single_shard_blocking(*mesh_, *hot_down_l1_, hd0.data());
        }
        auto* lut_u16 = static_cast<uint16_t*>(host_slot_lut_.ptr);
        std::memset(lut_u16, 0xFF, uint32_t(kTotalLayersWithEcho) * 256u * sizeof(uint16_t));
        for (uint32_t i = 0u; i < pin_n; ++i) {
            lut_u16[1u * 256u + i] = uint16_t(i);
            layers_[1]->hot_slot_for_eid[i] = uint16_t(i);
        }
        const distributed::MeshCoordinateRange all_devs(mesh_->shape());
        mesh_->mesh_command_queue().enqueue_write_shard_to_sub_grid(
            l1_lut_->mesh_buffer(), lut_u16, all_devs, true, BufferRegion(0, l1_lut_->buffer()->size()));
    }

    void load_one_layer_file(
        int layer_idx,
        std::vector<uint8_t>& hg0, std::vector<uint8_t>& hg1,
        std::vector<uint8_t>& hu0, std::vector<uint8_t>& hu1,
        std::vector<uint8_t>& hd0, std::vector<uint8_t>& hd1,
        std::vector<std::pair<uint16_t, uint16_t>>& hot_list,
        int total_layers)
    {
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/experts_hlut4/hlut4_layer_%02d.bin", cfg_.weight_root.c_str(), layer_idx);
        const int fd = ::open(fname, O_RDONLY);
        if (fd < 0) throw std::runtime_error(std::string("Cannot open HLUT4 shard: ") + fname);

        Ernie55Hlut4LayerHeader hdr{};
        if (::pread(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr)) ||
            hdr.magic != kErnie55Hlut4ShardMagic || hdr.version != kErnie55Hlut4ShardVersion) {
            ::close(fd);
            throw std::runtime_error(std::string("Invalid HLUT4 shard header: ") + fname);
        }

        const size_t exp_bytes_disk = kHlut4ExpertBytes;
        const size_t read_bytes = size_t(n_experts_) * exp_bytes_disk;
        const off_t file_off = off_t(sizeof(hdr)) + off_t(expert_begin_) * off_t(exp_bytes_disk);
        std::vector<Hlut4Tile32x32> raw(size_t(n_experts_) * 3u * kExpProjTiles);
        size_t got = 0;
        while (got < read_bytes) {
            const ssize_t r = ::pread(fd, reinterpret_cast<uint8_t*>(raw.data()) + got, read_bytes - got, file_off + off_t(got));
            if (r <= 0) { ::close(fd); throw std::runtime_error(std::string("Short read in ") + fname); }
            got += size_t(r);
        }
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);

        const size_t exp_card_b = size_t(sb_per_proj_) * kPageBytesHlut4;
        const size_t bank_b = size_t(n_experts_) * exp_card_b;
        std::vector<uint8_t> g0(bank_b), g1(is_tp2_ ? bank_b : 0u);
        std::vector<uint8_t> u0(bank_b), u1(is_tp2_ ? bank_b : 0u);
        std::vector<uint8_t> d0(bank_b), d1(is_tp2_ ? bank_b : 0u);

        #pragma omp parallel for schedule(static)
        for (uint32_t slot = 0u; slot < n_experts_; ++slot) {
            const Hlut4Tile32x32* base = raw.data() + size_t(slot) * (3u * kExpProjTiles);
            const size_t off = size_t(slot) * exp_card_b;
            if (is_tp2_) {
                pack_hlut4_expert_tp2(base + 0u * kExpProjTiles, false, g0.data() + off, g1.data() + off);
                pack_hlut4_expert_tp2(base + 1u * kExpProjTiles, false, u0.data() + off, u1.data() + off);
                pack_hlut4_expert_tp2(base + 2u * kExpProjTiles, true,  d0.data() + off, d1.data() + off);
            } else {
                pack_hlut4_expert_ep(base + 0u * kExpProjTiles, g0.data() + off);
                pack_hlut4_expert_ep(base + 1u * kExpProjTiles, u0.data() + off);
                pack_hlut4_expert_ep(base + 2u * kExpProjTiles, d0.data() + off);
            }
        }

        upload_layer_bank(layer_idx, g0, g1, u0, u1, d0, d1, hdr.signs_inter);

        // Wave-10 C1: Pin hottest slot-0 expert across backbone layers L01..L42 first (up to num_hot_shards_=41 -> ~253.7 MiB L1!)
        if (hot_list.size() < num_hot_shards_ && n_experts_ > 0u && layer_idx <= kNumMoeBackboneLayers) {
            const uint32_t per_l = std::max<uint32_t>(1u, num_hot_shards_ / uint32_t(std::max(1, std::min(total_layers, kNumMoeBackboneLayers))));
            for (uint32_t s = 0u; s < per_l && s < n_experts_ && hot_list.size() < num_hot_shards_; ++s) {
                const size_t hidx = hot_list.size();
                std::memcpy(hg0.data() + hidx * exp_card_b, g0.data() + s * exp_card_b, exp_card_b);
                std::memcpy(hu0.data() + hidx * exp_card_b, u0.data() + s * exp_card_b, exp_card_b);
                std::memcpy(hd0.data() + hidx * exp_card_b, d0.data() + s * exp_card_b, exp_card_b);
                if (is_tp2_) {
                    std::memcpy(hg1.data() + hidx * exp_card_b, g1.data() + s * exp_card_b, exp_card_b);
                    std::memcpy(hu1.data() + hidx * exp_card_b, u1.data() + s * exp_card_b, exp_card_b);
                    std::memcpy(hd1.data() + hidx * exp_card_b, d1.data() + s * exp_card_b, exp_card_b);
                }
                hot_list.emplace_back(uint16_t(layer_idx), uint16_t(s));
            }
        }
    }

    void write_layer_desc(const ResidentLayerBank& bank) noexcept {
        auto* ldesc = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(host_slot_ids_.ptr) + (kSlotIdsBytes - 64u));
        const bool use_l1 = hot_gate_l1_.has_value();
        ldesc[0] = bank.gate_pages.buffer()->address();
        ldesc[1] = bank.up_pages.buffer()->address();
        ldesc[2] = bank.down_pages.buffer()->address();
        ldesc[3] = 0u;
        ldesc[4] = bank.wht_q_i.buffer()->address();
        ldesc[5] = use_l1 ? hot_gate_l1_->buffer()->address() : dummy_hot_->buffer()->address();
        ldesc[6] = use_l1 ? hot_up_l1_->buffer()->address()   : dummy_hot_->buffer()->address();
        ldesc[7] = use_l1 ? hot_down_l1_->buffer()->address() : dummy_hot_->buffer()->address();
        ldesc[8] = l1_lut_->buffer()->address();
        ldesc[9] = bank.layer;
        for (uint32_t k = 10u; k < 16u; ++k) ldesc[k] = 0u;
    }

    void run_bcast_decode64_untraced() {
        auto k0 = x_decode_; auto k1 = slot_ids_; auto k2 = weights_decode64_;
        (void)k0; (void)k1; (void)k2;
        x_decode_ = ttnn::broadcast(x_decode_src_, x16_coord_, std::nullopt,
                                    ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        slot_ids_ = ttnn::broadcast(slot_ids_src_, x16_coord_, 1,
                                    ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        weights_decode64_ = ttnn::broadcast(weights_decode64_src_, x16_coord_, std::nullopt,
                                            ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
    }

    void run_bcast_prefill_untraced() {
        auto k1 = slot_meta_dst_; auto k2 = x_rm_dst_;
        (void)k1; (void)k2;
        slot_meta_dst_ = ttnn::broadcast(*slot_meta_src_, x16_coord_, std::nullopt,
                                         ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        x_rm_dst_ = ttnn::broadcast(*x_rm_src_, x16_coord_, std::nullopt,
                                    ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
    }

    void run_reduce_decode_untraced() {
        auto k = finish_decode_; (void)k;
        finish_decode_[0] = combined_f32_;
        if (is_tp2_) {
            finish_decode_[1] = ttnn::all_reduce(
                combined_f32_, 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
            finish_decode_[2] = ttnn::untilize(finish_decode_[1], ttnn::DRAM_MEMORY_CONFIG);
        } else {
            finish_decode_[1] = combined_f32_;
            finish_decode_[2] = ttnn::untilize(combined_f32_, ttnn::DRAM_MEMORY_CONFIG);
        }
    }

    void run_reduce_prefill_untraced() {
        auto k = finish_prefill_; (void)k;
        // Wave-9 Pass 5: On-device F32->BF16 typecast before FABRIC_1D all_reduce & PCIe readback (2.0x less bandwidth!)
        finish_prefill_[0] = ttnn::typecast(*y_accum_f32_, DataType::BFLOAT16, ttnn::DRAM_MEMORY_CONFIG);
        if (is_tp2_) {
            finish_prefill_[1] = ttnn::all_reduce(
                finish_prefill_[0], 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
        } else {
            finish_prefill_[1] = finish_prefill_[0];
        }
    }

    void enqueue_projection(
        const ResidentLayerBank& bank,
        const ttnn::Tensor& act_l1,
        const ttnn::Tensor& pages_dram,
        ttnn::Tensor& partials_l1,
        uint32_t slot_start,
        uint32_t batch_cnt,
        uint32_t act_slot_stride,
        uint32_t proj_idx)
    {
        const bool is_down     = (proj_idx == 2u);
        const bool is_fused_gu = (proj_idx == 3u);
        const uint32_t num_proj = is_fused_gu ? 2u : 1u;
        const uint32_t keep_act = (act_slot_stride == 0u) ? 1u : 0u;
        const uint32_t nt4      = is_down ? 16u : nt4_gu_;
        const uint32_t kt4      = is_down ? kt4_dn_ : 16u;
        const uint32_t k_splits = is_down ? k_splits_dn_ : k_splits_gu_;
        const uint32_t nt_total = nt4 * 4u;
        const uint32_t kt_total = kt4 * 4u;
        const uint32_t kt_core  = num_sb_core_ * 4u;

        const bool use_l1 = hot_gate_l1_.has_value();
        const ttnn::Tensor& hot_t = use_l1 ? (is_down ? *hot_down_l1_ : *hot_gate_l1_) : *dummy_hot_;
        const ttnn::Tensor& partials0 = is_fused_gu ? gate_partials_ : partials_l1;

        std::vector<CoreCoord> all_coords;
        all_coords.reserve(nt4 * k_splits);
        uint32_t ci = 0u;
        for (uint32_t nb4 = 0u; nb4 < nt4; ++nb4)
            for (uint32_t ks = 0u; ks < k_splits; ++ks)
                all_coords.push_back(core_at(ci++));
        const CoreRangeSet all_cores = make_core_set(all_coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_reader_a.cpp",
            all_cores, ReaderConfigDescriptor{}, make_accessor_args({&act_l1}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_writer_b.cpp",
            all_cores, WriterConfigDescriptor{},
            make_accessor_args({&pages_dram, &slot_ids_, &partials0, &hot_t, &*l1_lut_}, {sb_per_proj_}));

        ComputeConfigDescriptor cfg;
        // Wave-9 Pass 7: Both CB 0 (act) & CB 1 (unpacked HLUT4 weights) are Float16_b (BF16, 7-bit mantissa) ->
        // HiFi2 is mathematically bit-identical to HiFi4 for BF16xBF16->FP32 while doubling Tensix FPU throughput!
        cfg.math_fidelity = MathFidelity::HiFi2;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = true;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[4] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[5] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/ernie55_p150_compute.cpp", all_cores, cfg);

        ci = 0u;
        for (uint32_t nb4 = 0u; nb4 < nt4; ++nb4) {
            for (uint32_t ks = 0u; ks < k_splits; ++ks) {
                const uint32_t kb4_start = ks * num_sb_core_;
                const CoreCoord& c = all_coords[ci++];
                reader.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{act_l1.buffer()->address(), kt_total, kb4_start * 4u, kt_core,
                                             batch_cnt, act_slot_stride, slot_start, num_proj});
                writer.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{slot_ids_.buffer()->address(), partials0.buffer()->address(),
                                             slot_start, proj_idx, nb4, kt4, kb4_start, num_sb_core_, ks,
                                             nt_total, kBatchSlots, kSlotIdsBytes, batch_cnt,
                                             partials_l1.buffer()->address()});
                compute.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{num_sb_core_, batch_cnt, num_proj, keep_act, slot_start});
            }
        }

        const uint32_t cb0_pages = (keep_act != 0u || is_down) ? kt_core : (2u * kt_core);
        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  cb0_pages,           2048u, tt::DataFormat::Float16_b, all_cores),
            make_cb(1,  num_sb_core_ * 16u,  2048u, tt::DataFormat::Float16_b, all_cores),
            make_cb(4,  4u * num_sb_core_,   4096u, tt::DataFormat::UInt32,    all_cores),
            make_cb(5,  4u * num_sb_core_,   4096u, tt::DataFormat::UInt32,    all_cores),
            make_cb(6,  1u,                  4608u, tt::DataFormat::UInt32,    all_cores),
            make_cb(16, 8u,                  4096u, tt::DataFormat::Float32,   all_cores),
        };

        if (is_fused_gu) {
            const ttnn::Tensor& hot_up = use_l1 ? *hot_up_l1_ : *dummy_hot_;
            run_generic_preserving_topology(
                {act_l1, slot_ids_, pages_dram, bank.up_pages, hot_t, hot_up, *l1_lut_, gate_partials_, partials_l1}, prog);
        } else {
            run_generic_preserving_topology(
                {act_l1, slot_ids_, pages_dram, hot_t, *l1_lut_, partials_l1}, prog);
        }
    }

    void enqueue_situ_wht128(const ResidentLayerBank& bank, uint32_t batch_cnt) {
        const uint32_t max_groups = total_cores_ / nt4_gu_;
        const uint32_t m_per_grp  = (batch_cnt + max_groups - 1u) / max_groups;
        const uint32_t ngroups    = (batch_cnt + m_per_grp - 1u) / m_per_grp;
        const uint32_t ncores     = nt4_gu_ * ngroups;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_situ_wht128_reader.cpp",
            cores, ReaderConfigDescriptor{},
            make_accessor_args({&gate_partials_, &up_partials_, &bank.wht_q_i, &slot_ids_}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_situ_wht128_writer.cpp",
            cores, WriterConfigDescriptor{},
            make_accessor_args({&z_rot_bf16_}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi2;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/ernie55_p150_situ_wht128_compute.cpp", cores, cfg);

        for (uint32_t g = 0u; g < ngroups; ++g) {
            const uint32_t m_start = g * m_per_grp;
            const uint32_t m_cnt   = std::min<uint32_t>(m_per_grp, batch_cnt - m_start);
            for (uint32_t nb4 = 0u; nb4 < nt4_gu_; ++nb4) {
                const CoreCoord& c = coords[g * nt4_gu_ + nb4];
                reader.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{gate_partials_.buffer()->address(), up_partials_.buffer()->address(),
                                             slot_ids_.buffer()->address(), nb4, k_splits_gu_, kBatchSlots,
                                             m_start, m_cnt, kSlotIdsBytes, inter_tiles_});
                writer.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{z_rot_bf16_.buffer()->address(), nb4, m_start, m_cnt, inter_tiles_});
                compute.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{k_splits_gu_, m_cnt, beta_bits_, beta_recip_bits_, lin_bits_, lin_recip_bits_});
            }
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  k_splits_gu_, 4096u, tt::DataFormat::Float32,   cores),
            make_cb(1,  k_splits_gu_, 4096u, tt::DataFormat::Float32,   cores),
            make_cb(2,  16u, 2048u, tt::DataFormat::Float16_b, cores),
            make_cb(24, 4u,  2048u, tt::DataFormat::Float16_b, cores),
            make_cb(16, 8u,  2048u, tt::DataFormat::Float16_b, cores),
        };
        run_generic_preserving_topology({gate_partials_, up_partials_, bank.wht_q_i, slot_ids_, z_rot_bf16_}, prog);
    }

    void enqueue_ksplit_combine_decode(uint32_t slot_start, uint32_t batch_cnt, bool use_weights64) {
        constexpr uint32_t ncores = 32u;  // 64 tiles / 32 cores = 2 tiles/core
        const uint32_t acc_prev = (slot_start > 0u) ? 1u : 0u;
        const ttnn::Tensor& w_tensor = use_weights64 ? weights_decode64_ : weights_tiled_;

        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_ksplit_combine_reader.cpp",
            cores, ReaderConfigDescriptor{},
            make_accessor_args({&down_partials_, &w_tensor, &combined_f32_}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_ksplit_combine_writer.cpp",
            cores, WriterConfigDescriptor{},
            make_accessor_args({&combined_f32_}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi2;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[2] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/ernie55_p150_ksplit_combine_compute.cpp", cores, cfg);

        for (uint32_t i = 0u; i < ncores; ++i) {
            reader.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{down_partials_.buffer()->address(), w_tensor.buffer()->address(),
                                                 combined_f32_.buffer()->address(), slot_start, k_splits_dn_,
                                                 kBatchSlots, 2u * i, 2u, acc_prev, batch_cnt});
            writer.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{combined_f32_.buffer()->address(), 2u * i, 2u});
            compute.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{k_splits_dn_, 2u, acc_prev, batch_cnt});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  k_splits_dn_, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1,  2u,           4096u, tt::DataFormat::Float32, cores),
            make_cb(2,  2u,           4096u, tt::DataFormat::Float32, cores),
            make_cb(16, 2u,           4096u, tt::DataFormat::Float32, cores),
        };
        run_generic_preserving_topology({down_partials_, w_tensor, combined_f32_}, prog);
    }

    void enqueue_zero_y_accum(uint32_t rows) {
        const uint32_t total_pages = rows * 2u;  // 2 exact 4096 B pages/row at H_lat=2048
        const uint32_t ncores = std::min<uint32_t>(total_cores_, total_pages);
        const uint32_t per_c = (total_pages + ncores - 1u) / ncores;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto writer = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_zero_pages_writer.cpp",
            cores, WriterConfigDescriptor{}, make_accessor_args({&*y_accum_f32_}));
        for (uint32_t i = 0u; i < ncores; ++i) {
            const uint32_t p0 = i * per_c;
            const uint32_t cnt = (p0 < total_pages) ? std::min<uint32_t>(per_c, total_pages - p0) : 0u;
            writer.runtime_args.emplace_back(coords[i], std::vector<uint32_t>{y_accum_f32_->buffer()->address(), p0, cnt});
        }
        ProgramDescriptor prog;
        prog.kernels = {std::move(writer)};
        prog.cbs = {make_cb(0, 1u, 4096u, tt::DataFormat::UInt32, cores)};
        run_generic_preserving_topology({slot_ids_, *y_accum_f32_}, prog);
    }

    void enqueue_compact_gather_tilize(uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t ncores = 64u;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto writer = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_compact_gather_tilize.cpp",
            cores, WriterConfigDescriptor{},
            make_accessor_args({&*x_rm_dst_, &*slot_meta_dst_, &x_tilized_}));
        for (uint32_t i = 0u; i < ncores; ++i) {
            writer.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{x_rm_dst_->buffer()->address(), slot_meta_dst_->buffer()->address(),
                                                 x_tilized_.buffer()->address(), slot_start, i, batch_cnt});
        }
        ProgramDescriptor prog;
        prog.kernels = {std::move(writer)};
        prog.cbs = {make_cb(0, 1u, 14592u, tt::DataFormat::UInt32, cores)};
        run_generic_preserving_topology({*x_rm_dst_, *slot_meta_dst_, x_tilized_}, prog);
    }

    void enqueue_compact_scatter_accum(uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t ncores = 64u;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_compact_scatter_reader.cpp",
            cores, ReaderConfigDescriptor{},
            make_accessor_args({&down_partials_, &*y_accum_f32_, &*slot_meta_dst_}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/ernie55_p150_compact_scatter_writer.cpp",
            cores, WriterConfigDescriptor{},
            make_accessor_args({&*y_accum_f32_, &*slot_meta_dst_}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi2;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[2] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/ernie55_p150_ksplit_combine_compute.cpp", cores, cfg);

        for (uint32_t i = 0u; i < ncores; ++i) {
            reader.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{down_partials_.buffer()->address(),
                                                 y_accum_f32_->buffer()->address(), slot_meta_dst_->buffer()->address(),
                                                 slot_start, k_splits_dn_, i, kBatchSlots, batch_cnt});
            writer.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{y_accum_f32_->buffer()->address(), slot_meta_dst_->buffer()->address(),
                                                 slot_start, i, batch_cnt});
            compute.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{k_splits_dn_, batch_cnt, 1u, 1u});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  k_splits_dn_, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1,  1u,           4096u, tt::DataFormat::Float32, cores),
            make_cb(2,  1u,           4096u, tt::DataFormat::Float32, cores),
            make_cb(5,  1u,           20736u, tt::DataFormat::UInt32,  cores),
            make_cb(6,  1u,           16640u, tt::DataFormat::UInt32,  cores),
            make_cb(7,  kBatchSlots,  64u,   tt::DataFormat::UInt32,  cores),
            make_cb(16, 1u,           4096u, tt::DataFormat::Float32, cores),
        };
        run_generic_preserving_topology({down_partials_, *slot_meta_dst_, *y_accum_f32_}, prog);
    }

    void prewarm_and_capture_traces(const ResidentLayerBank& bank) {
        if (programs_prewarmed_) return;
        programs_prewarmed_ = true;

        auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids_.ptr);
        for (uint32_t i = 0u; i < kMaxCompactSlots; ++i) slot_u16[i] = 0xFFFFu;
        slot_u16[0] = 0u;
        std::memset(host_x_tilized_.ptr, 0, host_x_tilized_.size);
        std::memset(host_slot_meta_.ptr, 0, host_slot_meta_.size);
        std::memset(host_weights_.ptr, 0, host_weights_.size);
        write_layer_desc(bank);

        auto& cq = mesh_->mesh_command_queue();
        const distributed::MeshCoordinateRange all_devs(mesh_->shape());
        cq.enqueue_write_shard_to_sub_grid(x_decode_.mesh_buffer(), host_x_tilized_.ptr, all_devs, false, BufferRegion(0, x_decode_.buffer()->size()));
        cq.enqueue_write_shard_to_sub_grid(slot_ids_.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
        cq.enqueue_write_shard_to_sub_grid(weights_decode64_.mesh_buffer(), host_weights_.ptr, all_devs, false, BufferRegion(0, weights_decode64_.buffer()->size()));
        cq.enqueue_write_shard_to_sub_grid(slot_meta_dst_->mesh_buffer(), host_slot_meta_.ptr, all_devs, false, BufferRegion(0, slot_meta_dst_->buffer()->size()));
        cq.finish();

        // Prewarm decode + compact kernels once so TRISC binaries are cached before trace capture
        enqueue_projection(bank, x_decode_, bank.gate_pages, up_partials_, 0u, 2u, 0u, 3u);
        enqueue_situ_wht128(bank, 2u);
        enqueue_projection(bank, z_rot_bf16_, bank.down_pages, down_partials_, 0u, 2u, inter_tiles_, 2u);
        enqueue_ksplit_combine_decode(0u, 2u, true);
        run_reduce_decode_untraced();
        enqueue_zero_y_accum(64u);
        enqueue_compact_gather_tilize(0u, 2u);
        enqueue_compact_scatter_accum(0u, 2u);
        run_reduce_prefill_untraced();
        cq.finish();

        if (!cfg_.enable_trace) return;
        TraceStorageContextGuard guard;
        if (is_tp2_) {
            {
                x_decode_ = ttnn::Tensor(); slot_ids_ = ttnn::Tensor(); weights_decode64_ = ttnn::Tensor();
                cq.finish();
                const auto tid = mesh_->begin_mesh_trace(cq);
                x_decode_ = ttnn::broadcast(x_decode_src_, x16_coord_, std::nullopt,
                                            ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
                slot_ids_ = ttnn::broadcast(slot_ids_src_, x16_coord_, 1,
                                            ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
                weights_decode64_ = ttnn::broadcast(weights_decode64_src_, x16_coord_, std::nullopt,
                                                    ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
                mesh_->end_mesh_trace(cq, tid);
                bcast_traces_[0].id = tid; bcast_traces_[0].warmed = true;
            }
            {
                slot_meta_dst_ = std::nullopt; x_rm_dst_ = std::nullopt;
                cq.finish();
                const auto tid = mesh_->begin_mesh_trace(cq);
                slot_meta_dst_ = ttnn::broadcast(*slot_meta_src_, x16_coord_, std::nullopt,
                                                 ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
                x_rm_dst_ = ttnn::broadcast(*x_rm_src_, x16_coord_, std::nullopt,
                                            ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
                mesh_->end_mesh_trace(cq, tid);
                bcast_traces_[1].id = tid; bcast_traces_[1].warmed = true;
            }
        }
        {
            for (auto& t : finish_decode_) t = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            finish_decode_[0] = combined_f32_;
            if (is_tp2_) {
                finish_decode_[1] = ttnn::all_reduce(
                    combined_f32_, 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
                finish_decode_[2] = ttnn::untilize(finish_decode_[1], ttnn::DRAM_MEMORY_CONFIG);
            } else {
                finish_decode_[1] = combined_f32_;
                finish_decode_[2] = ttnn::untilize(combined_f32_, ttnn::DRAM_MEMORY_CONFIG);
            }
            mesh_->end_mesh_trace(cq, tid);
            reduce_traces_[0].id = tid; reduce_traces_[0].warmed = true;
        }
        {
            for (auto& t : finish_prefill_) t = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            finish_prefill_[0] = ttnn::typecast(*y_accum_f32_, DataType::BFLOAT16, ttnn::DRAM_MEMORY_CONFIG);
            if (is_tp2_) {
                finish_prefill_[1] = ttnn::all_reduce(
                    finish_prefill_[0], 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
            } else {
                finish_prefill_[1] = finish_prefill_[0];
            }
            mesh_->end_mesh_trace(cq, tid);
            reduce_traces_[1].id = tid; reduce_traces_[1].warmed = true;
        }
        // Wave-9 Build #4: Pre-capture ALL decode slot buckets at startup BEFORE any prefill trace can consume trace DRAM!
        static constexpr uint32_t kDecBuckets[] = {2u, 4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u};
        uint32_t dec_captured = 0u;
        for (uint32_t bkt : kDecBuckets) {
            const uint64_t est_b = uint64_t((bkt + kBatchSlots - 1u) / kBatchSlots) * 131072ULL + (2ULL << 20);
            if (uint64_t(mesh_->get_trace_buffers_size()) + est_b + (8ULL << 20) > uint64_t(trace_region_bytes_)) break;
            auto run_bkt = [&]() {
                for (uint32_t s = 0u; s < bkt; s += kBatchSlots) {
                    const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, bkt - s);
                    enqueue_projection(bank, x_decode_, bank.gate_pages, up_partials_, s, bcnt, 0u, 3u);
                    enqueue_situ_wht128(bank, bcnt);
                    enqueue_projection(bank, z_rot_bf16_, bank.down_pages, down_partials_, s, bcnt, inter_tiles_, 2u);
                    enqueue_ksplit_combine_decode(s, bcnt, true);
                }
            };
            run_bkt();
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            run_bkt();
            mesh_->end_mesh_trace(cq, tid);
            auto& slot = op_traces_[std::make_pair(0u, bkt)];
            slot.warmed = true;
            slot.id = tid;
            ++dec_captured;
        }
        cq.finish();
        std::printf("[P150-ERNIE55] Pre-captured %u/%u decode traces (trace_used=%.1f/%.1f MiB)\n",
                    dec_captured, unsigned(sizeof(kDecBuckets) / sizeof(kDecBuckets[0])),
                    double(mesh_->get_trace_buffers_size()) / 1048576.0,
                    double(trace_region_bytes_) / 1048576.0);
        std::fflush(stdout);
    }

    template <typename ExecFn, typename RedFn>
    void run_traced(uint32_t path_id, uint32_t scheduled_slots, uint64_t est_bytes, ExecFn&& exec_fn, RedFn&& red_fn) {
        TraceStorageContextGuard guard;
        auto& cq = mesh_->mesh_command_queue();
        TraceSlot* op_tr = nullptr;
        if (cfg_.enable_trace) {
            auto key = std::make_pair(path_id, scheduled_slots);
            auto it = op_traces_.find(key);
            if (it != op_traces_.end()) op_tr = &it->second;
            else if (op_traces_.size() < 128u) op_tr = &op_traces_[key];
        }
        if (op_tr && op_tr->warmed && !op_tr->skipped) {
            if (!op_tr->id) {
                if (uint64_t(mesh_->get_trace_buffers_size()) + est_bytes + (8ULL << 20) <= uint64_t(trace_region_bytes_)) {
                    cq.finish();
                    const auto tid = mesh_->begin_mesh_trace(cq);
                    exec_fn();
                    mesh_->end_mesh_trace(cq, tid);
                    op_tr->id = tid;
                } else {
                    op_tr->skipped = true;
                }
            }
            if (op_tr->id) mesh_->replay_mesh_trace(cq, *op_tr->id, false);
            else exec_fn();
        } else {
            exec_fn();
            if (op_tr) op_tr->warmed = true;
        }
        TraceSlot& red_tr = reduce_traces_[path_id];
        if (cfg_.enable_trace && red_tr.id.has_value()) mesh_->replay_mesh_trace(cq, *red_tr.id, false);
        else red_fn();
    }

    void run_decode_small(
        const ResidentLayerBank& bank,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][ERNIE55_P150_MAX_TOPK],
        const float row_weights[][ERNIE55_P150_MAX_TOPK],
        const uint16_t* x_rot_bf16,
        float* y_out_f32)
    {
        struct SmallPair { uint16_t row; float weight; };
        std::array<std::array<SmallPair, 32>, 256> exp_pairs{};
        std::array<uint8_t, 256> exp_cnt{};
        std::vector<uint16_t> active_experts;
        active_experts.reserve(64);

        for (uint32_t r = 0u; r < rows; ++r) {
            for (uint32_t s = 0u; s < top_k; ++s) {
                const int32_t eid = row_slots[r][s];
                const float w = row_weights[r][s];
                if (eid >= 0 && uint32_t(eid) < bank.n_experts && w != 0.0f) {
                    uint8_t& c = exp_cnt[uint32_t(eid)];
                    if (c == 0u) active_experts.push_back(uint16_t(eid));
                    if (c < 32u) exp_pairs[uint32_t(eid)][c++] = SmallPair{uint16_t(r), w};
                }
            }
        }
        if (active_experts.empty()) return;
        // Canonical ascending expert order guarantees 100% bit-exact row invariance between M=1 and M=K+1!
        std::sort(active_experts.begin(), active_experts.end());

        const uint32_t n_act = uint32_t(active_experts.size());
        stat_total_slots_ += n_act;
        for (uint16_t eid : active_experts) {
            if (bank.hot_slot_for_eid[eid] != 0xFFFFu) ++stat_l1_hits_;
        }

        auto* x_tiles = static_cast<uint16_t*>(host_x_tilized_.ptr);
        const auto& t_lut = tile_offset_lut();
        for (uint32_t r = rows; r < dirty_mt0_; ++r) {
            const auto& row_off = t_lut[r];
            for (uint32_t kt = 0u; kt < kP150LatentTiles; ++kt) {
                uint16_t* dst_t = x_tiles + kt * 1024u;
                for (int c = 0; c < 32; ++c) dst_t[row_off[size_t(c)]] = 0u;
            }
        }
        dirty_mt0_ = rows;

        for (uint32_t r = 0u; r < rows; ++r) {
            const uint16_t* src_row = x_rot_bf16 + size_t(r) * kLatentDim;
            const auto& row_off = t_lut[r];
            const uint32_t off0 = row_off[0], off16 = row_off[16];
            for (uint32_t kt = 0u; kt < kP150LatentTiles; ++kt) {
                uint16_t* dst_t = x_tiles + kt * 1024u;
                const uint16_t* sb = src_row + kt * 32u;
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_t + off0),  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sb)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_t + off16), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sb + 16)));
            }
        }

        const uint32_t scheduled_slots = choose_decode_slot_bucket(n_act);
        auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids_.ptr);
        for (uint32_t i = 0u; i < kMaxCompactSlots; ++i) slot_u16[i] = 0xFFFFu;
        for (uint32_t i = 0u; i < n_act; ++i) slot_u16[i] = active_experts[i];
        write_layer_desc(bank);

        const size_t weights_bytes = size_t(scheduled_slots) * 4096ULL;
        auto* w_f32 = static_cast<float*>(host_weights_.ptr);
        std::memset(w_f32, 0, weights_bytes);
        for (uint32_t i = 0u; i < n_act; ++i) {
            const uint16_t eid = active_experts[i];
            const uint8_t cnt = exp_cnt[eid];
            float* dst_t = w_f32 + size_t(i) * 1024u;
            for (uint8_t k = 0u; k < cnt; ++k) {
                const auto& row_off = t_lut[exp_pairs[eid][k].row];
                const float w = exp_pairs[eid][k].weight;
                for (int c = 0; c < 32; ++c) dst_t[row_off[size_t(c)]] += w;
            }
        }

        const auto t_dp0 = std::chrono::steady_clock::now();
        const bool use_weights64 = (scheduled_slots <= 64u);
        auto& cq = mesh_->mesh_command_queue();
        if (is_tp2_ && cfg_.x16_relay && rows > 8u) {
            TraceStorageContextGuard guard;
            const distributed::MeshCoordinateRange card_x16(x16_coord_, x16_coord_);
            cq.enqueue_write_shard_to_sub_grid(x_decode_src_.mesh_buffer(), host_x_tilized_.ptr, card_x16, false, BufferRegion(0, kP150LatentTiles * 2048u));
            cq.enqueue_write_shard_to_sub_grid(slot_ids_src_.mesh_buffer(), host_slot_ids_.ptr, card_x16, false, BufferRegion(0, kSlotIdsBytes));
            auto& ws_src = use_weights64 ? weights_decode64_src_ : weights_tiled_src_;
            cq.enqueue_write_shard_to_sub_grid(ws_src.mesh_buffer(), host_weights_.ptr, card_x16, false, BufferRegion(0, weights_bytes));
            if (bcast_traces_[0].id.has_value()) mesh_->replay_mesh_trace(cq, *bcast_traces_[0].id, false);
            else run_bcast_decode64_untraced();
            if (!use_weights64) {
                weights_tiled_ = ttnn::broadcast(weights_tiled_src_, x16_coord_, std::nullopt,
                                                 ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            }
        } else {
            const distributed::MeshCoordinateRange all_devs(mesh_->shape());
            cq.enqueue_write_shard_to_sub_grid(x_decode_.mesh_buffer(), host_x_tilized_.ptr, all_devs, false, BufferRegion(0, kP150LatentTiles * 2048u));
            cq.enqueue_write_shard_to_sub_grid(slot_ids_.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
            auto& wt = use_weights64 ? weights_decode64_ : weights_tiled_;
            cq.enqueue_write_shard_to_sub_grid(wt.mesh_buffer(), host_weights_.ptr, all_devs, false, BufferRegion(0, weights_bytes));
        }
        const auto t_dh1 = std::chrono::steady_clock::now();
        auto dtr_it = op_traces_.find(std::make_pair(0u, scheduled_slots));
        const uint32_t dec_was_traced = (dtr_it != op_traces_.end() && dtr_it->second.id.has_value()) ? 1u : 0u;

        run_traced(
            0u, scheduled_slots,
            uint64_t((scheduled_slots + kBatchSlots - 1u) / kBatchSlots) * 131072ULL + (2ULL << 20),
            [&]() {
                for (uint32_t s = 0u; s < scheduled_slots; s += kBatchSlots) {
                    const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, scheduled_slots - s);
                    enqueue_projection(bank, x_decode_, bank.gate_pages, up_partials_, s, bcnt, 0u, 3u);
                    enqueue_situ_wht128(bank, bcnt);
                    enqueue_projection(bank, z_rot_bf16_, bank.down_pages, down_partials_, s, bcnt, inter_tiles_, 2u);
                    enqueue_ksplit_combine_decode(s, bcnt, use_weights64);
                }
            },
            [&]() { run_reduce_decode_untraced(); });

        const size_t out_bytes = size_t(rows) * kLatentDim * sizeof(float);
        const auto& out_rm = finish_decode_[2];
        auto borrowed = std::shared_ptr<distributed::MeshBuffer>(const_cast<distributed::MeshBuffer*>(&out_rm.mesh_buffer()), [](auto*) {});
        cq.enqueue_read_shards(
            {distributed::ShardDataTransfer{x16_coord_}.host_data(host_out_f32_.ptr).region(BufferRegion(0, out_bytes))},
            borrowed, true);
        const auto t_dr1 = std::chrono::steady_clock::now();

        const auto* out_f32 = static_cast<const float*>(host_out_f32_.ptr);
        for (uint32_t r = 0u; r < rows; ++r) {
            f32_row_add_avx2(out_f32 + size_t(r) * kLatentDim, y_out_f32 + size_t(r) * kLatentDim);
        }
        ++dprof_calls_;
        dprof_slots_ += n_act;
        dprof_sched_ += scheduled_slots;
        dprof_traced_ += dec_was_traced;
        dprof_h2d_ms_ += std::chrono::duration<double, std::milli>(t_dh1 - t_dp0).count();
        dprof_dev_ms_ += std::chrono::duration<double, std::milli>(t_dr1 - t_dh1).count();
        if (dprof_calls_ >= 42u) {
            const double inv = 1.0 / double(dprof_calls_);
            std::printf("[P150-DECODE-XRAY] %u calls (rows=%u) | avg ms/L: h2d=%.2f dev+red+d2h=%.2f (tot=%.2f) | slots=%.1f/%.1f traced=%u/%u trace_mib=%.1f\n",
                        dprof_calls_, rows, dprof_h2d_ms_ * inv, dprof_dev_ms_ * inv, (dprof_h2d_ms_ + dprof_dev_ms_) * inv,
                        double(dprof_slots_) * inv, double(dprof_sched_) * inv, dprof_traced_, dprof_calls_,
                        double(mesh_->get_trace_buffers_size()) / 1048576.0);
            std::fflush(stdout);
            dprof_calls_ = 0u; dprof_slots_ = 0u; dprof_sched_ = 0u; dprof_traced_ = 0u;
            dprof_h2d_ms_ = 0.0; dprof_dev_ms_ = 0.0;
        }
    }

    void run_compact_prefill(
        const ResidentLayerBank& bank,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][ERNIE55_P150_MAX_TOPK],
        const float row_weights[][ERNIE55_P150_MAX_TOPK],
        float* y_out_f32,
        bool accumulate_into_y,
        uint16_t* y_out_bf16 = nullptr,
        void* x_dma_ptr = nullptr)
    {
        if (!x_dma_ptr) x_dma_ptr = host_x_rm_.ptr;
        const auto t_pk0 = std::chrono::steady_clock::now();
        for (uint32_t e = 0u; e < bank.n_experts; ++e) per_exp_[e].clear();
        uint32_t total_pairs = 0u;
        for (uint32_t r = 0u; r < rows; ++r) {
            for (uint32_t s = 0u; s < top_k; ++s) {
                const int32_t eid = row_slots[r][s];
                const float w = row_weights[r][s];
                if (eid >= 0 && uint32_t(eid) < bank.n_experts && w != 0.0f) {
                    per_exp_[uint32_t(eid)].push_back({uint16_t(r), w});
                    ++total_pairs;
                }
            }
        }
        if (total_pairs == 0u) {
            if (!accumulate_into_y) {
                if (y_out_bf16) std::memset(y_out_bf16, 0, size_t(rows) * kLatentDim * sizeof(uint16_t));
                else            std::memset(y_out_f32,  0, size_t(rows) * kLatentDim * sizeof(float));
            }
            return;
        }

        // Wave-11 Turn-8 Pillar I: Bit-Exact Ascending-eid Conflict-Free Wave Packing!
        // Assign each tile of expert e (processed in strictly ascending eid order) to wave
        // w = 1 + max_{r in tile} row_last_wave[r]. Because any shared token row r satisfies
        // w(e1) < w(e2) whenever e1 < e2, every row accumulates in 100% bit-identical ascending
        // eid FP32 order while tiles inside each wave are strictly row-disjoint (slashing
        // compact_scatter_accum DRAM barriers ~3.5x from 28-40 down to ~8-11 per layer!).
        struct PlacedTile { uint16_t eid; uint16_t off; uint16_t cnt; };
        std::array<int16_t, kPrefillMaxRows> row_last_wave;
        row_last_wave.fill(-1);
        std::vector<std::vector<PlacedTile>> waves;
        waves.reserve(16u);
        size_t total_tiles = 0u;
        for (uint32_t e = 0u; e < bank.n_experts; ++e) {
            const auto& vec = per_exp_[e];
            const size_t tot_n = vec.size();
            if (tot_n == 0u) continue;
            // Assign all tiles of expert e to the SAME wave w so contiguous per-expert GDDR6 weight reuse
            // (runs) stays 100% intact while preserving bit-exact ascending-eid FP32 accumulation order!
            int16_t max_w = -1;
            for (size_t k = 0u; k < tot_n; ++k) {
                const int16_t lw = row_last_wave[vec[k].row];
                if (lw > max_w) max_w = lw;
            }
            const size_t w = size_t(max_w + 1);
            if (w >= waves.size()) waves.resize(w + 1u);
            for (size_t off = 0u; off < tot_n; off += 32u) {
                const uint16_t cnt = uint16_t(std::min<size_t>(32u, tot_n - off));
                waves[w].push_back(PlacedTile{uint16_t(e), uint16_t(off), cnt});
                ++total_tiles;
            }
            for (size_t k = 0u; k < tot_n; ++k) {
                row_last_wave[vec[k].row] = int16_t(w);
            }
        }
        std::vector<PlacedTile> placed;
        placed.reserve(total_tiles);
        for (const auto& wv : waves) {
            for (const auto& pt : wv) placed.push_back(pt);
        }

        const size_t x_bytes = size_t(rows) * kLatentDim * sizeof(uint16_t);
        const size_t out_bf16_bytes = size_t(rows) * kLatentDim * sizeof(uint16_t);
        auto& cq = mesh_->mesh_command_queue();
        const distributed::MeshCoordinateRange all_devs(mesh_->shape());
        const distributed::MeshCoordinateRange card_x16(x16_coord_, x16_coord_);

        std::array<uint32_t, kPrefillMaxRows> row_seen_gen{};
        uint32_t cur_gen = 0u;
        uint32_t call_barriers = 0u, call_runs = 0u, call_slots = 0u, call_sched = 0u, call_traced = 0u;
        for (size_t s0 = 0; s0 < placed.size(); s0 += kMaxCompactSlots) {
            const uint32_t chunk_n = uint32_t(std::min<size_t>(kMaxCompactSlots, placed.size() - s0));
            const uint32_t scheduled = choose_compact_slot_bucket(chunk_n);
            stat_total_slots_ += chunk_n;
            call_slots += chunk_n;
            call_sched += scheduled;

            auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids_.ptr);
            for (uint32_t i = 0u; i < kMaxCompactSlots; ++i) slot_u16[i] = 0xFFFFu;
            auto* meta_u32 = static_cast<uint32_t*>(host_slot_meta_.ptr);
            std::memset(meta_u32, 0, size_t(scheduled) * 256u);

            uint16_t prev_e = 0xFFFFu;
            for (uint32_t i = 0u; i < chunk_n; ++i) {
                const auto& pt = placed[s0 + i];
                const auto& vec = per_exp_[pt.eid];
                slot_u16[i] = pt.eid;
                if ((i % kBatchSlots) == 0u || pt.eid != prev_e) ++call_runs;
                prev_e = pt.eid;
                if (bank.hot_slot_for_eid[pt.eid] != 0xFFFFu) ++stat_l1_hits_;
                uint32_t* m = meta_u32 + size_t(i) * 64u;
                m[0] = pt.cnt;
                uint32_t need_barrier = 0u;
                if ((i % kBatchSlots) == 0u) {
                    ++cur_gen;
                } else {
                    for (uint32_t r = 0u; r < pt.cnt; ++r) {
                        if (row_seen_gen[vec[pt.off + r].row] == cur_gen) { need_barrier = 1u; break; }
                    }
                    if (need_barrier != 0u) { ++cur_gen; ++call_barriers; }
                }
                m[1] = need_barrier;
                auto* ridx = reinterpret_cast<uint16_t*>(m + 2);
                auto* mw   = reinterpret_cast<float*>(m + 32);
                for (uint32_t r = 0u; r < pt.cnt; ++r) {
                    const uint16_t rr = vec[pt.off + r].row;
                    ridx[r] = rr;
                    mw[r]   = vec[pt.off + r].weight;
                    row_seen_gen[rr] = cur_gen;
                }
            }
            write_layer_desc(bank);
            const auto t_hd0 = std::chrono::steady_clock::now();
            prof_pack_ms_ += std::chrono::duration<double, std::milli>(t_hd0 - t_pk0).count();

            if (is_tp2_ && cfg_.x16_relay) {
                // Wave-11 Cut P1.2: Inline 256 B slot_meta weights delete 1.25 MiB weights_tiled_src_ H2D & FABRIC bcast!
                TraceStorageContextGuard guard;
                cq.enqueue_write_shard_to_sub_grid(x_rm_src_->mesh_buffer(), x_dma_ptr, card_x16, false, BufferRegion(0, x_bytes));
                cq.enqueue_write_shard_to_sub_grid(slot_meta_src_->mesh_buffer(), host_slot_meta_.ptr, card_x16, false, BufferRegion(0, size_t(scheduled) * 256u));
                cq.enqueue_write_shard_to_sub_grid(slot_ids_.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
                if (bcast_traces_[1].id.has_value()) mesh_->replay_mesh_trace(cq, *bcast_traces_[1].id, false);
                else run_bcast_prefill_untraced();
            } else {
                cq.enqueue_write_shard_to_sub_grid(x_rm_dst_->mesh_buffer(), x_dma_ptr, all_devs, false, BufferRegion(0, x_bytes));
                cq.enqueue_write_shard_to_sub_grid(slot_ids_.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
                cq.enqueue_write_shard_to_sub_grid(slot_meta_dst_->mesh_buffer(), host_slot_meta_.ptr, all_devs, false, BufferRegion(0, size_t(scheduled) * 256u));
            }
            const auto t_ex0 = std::chrono::steady_clock::now();
            prof_h2d_ms_ += std::chrono::duration<double, std::milli>(t_ex0 - t_hd0).count();

            auto tr_it = op_traces_.find(std::make_pair(1u, scheduled));
            if (tr_it != op_traces_.end() && tr_it->second.id.has_value()) call_traced = 1u;

            run_traced(
                1u, scheduled,
                uint64_t((scheduled + kBatchSlots - 1u) / kBatchSlots) * 262144ULL + (4ULL << 20),
                [&]() {
                    enqueue_zero_y_accum(kPrefillMaxRows);
                    for (uint32_t s = 0u; s < scheduled; s += kBatchSlots) {
                        const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, scheduled - s);
                        enqueue_compact_gather_tilize(s, bcnt);
                        enqueue_projection(bank, x_tilized_, bank.gate_pages, up_partials_, s, bcnt, kP150LatentTiles, 3u);
                        enqueue_situ_wht128(bank, bcnt);
                        enqueue_projection(bank, z_rot_bf16_, bank.down_pages, down_partials_, s, bcnt, inter_tiles_, 2u);
                        enqueue_compact_scatter_accum(s, bcnt);
                    }
                },
                [&]() { run_reduce_prefill_untraced(); });

            const bool overwrite_y = (s0 == 0u && !accumulate_into_y);
            const bool direct_bf16_read = (overwrite_y && y_out_bf16 != nullptr);
            void* rd_dst = direct_bf16_read ? static_cast<void*>(y_out_bf16) : host_out_bf16_.ptr;
            const auto& out_t = finish_prefill_[1];
            auto borrowed = std::shared_ptr<distributed::MeshBuffer>(const_cast<distributed::MeshBuffer*>(&out_t.mesh_buffer()), [](auto*) {});
            cq.enqueue_read_shards(
                {distributed::ShardDataTransfer{x16_coord_}.host_data(rd_dst).region(BufferRegion(0, out_bf16_bytes))},
                borrowed, true);
            const auto t_rd1 = std::chrono::steady_clock::now();
            prof_dev_ms_ += std::chrono::duration<double, std::milli>(t_rd1 - t_ex0).count();

            if (!direct_bf16_read) {
                const auto* out_bf16 = static_cast<const uint16_t*>(host_out_bf16_.ptr);
                if (y_out_bf16 != nullptr) {
                    for (uint32_t r = 0u; r < rows; ++r) {
                        bf16_row_add_bf16_avx2(out_bf16 + size_t(r) * kLatentDim, y_out_bf16 + size_t(r) * kLatentDim);
                    }
                } else if (overwrite_y) {
                    for (uint32_t r = 0u; r < rows; ++r) {
                        bf16_row_to_f32_avx2(out_bf16 + size_t(r) * kLatentDim, y_out_f32 + size_t(r) * kLatentDim);
                    }
                } else {
                    for (uint32_t r = 0u; r < rows; ++r) {
                        bf16_row_add_to_f32_avx2(out_bf16 + size_t(r) * kLatentDim, y_out_f32 + size_t(r) * kLatentDim);
                    }
                }
            }
            prof_fin_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rd1).count();
        }
        ++prof_calls_;
        prof_slots_    += call_slots;
        prof_sched_    += call_sched;
        prof_runs_     += call_runs;
        prof_barriers_ += call_barriers;
        prof_traced_   += call_traced;
        if (prof_calls_ >= 42u) {
            const double inv = 1.0 / double(prof_calls_);
            std::printf("[P150-PREFILL-XRAY] %u calls (rows=%u) | avg ms/L: cv=%.2f pack=%.2f h2d=%.2f dev+red+d2h=%.2f fin=%.2f (tot=%.2f) | slots=%.1f/%.1f runs=%.1f bar=%.1f traced=%u/%u\n",
                        prof_calls_, rows,
                        prof_cv_ms_ * inv, prof_pack_ms_ * inv, prof_h2d_ms_ * inv, prof_dev_ms_ * inv, prof_fin_ms_ * inv,
                        (prof_cv_ms_ + prof_pack_ms_ + prof_h2d_ms_ + prof_dev_ms_ + prof_fin_ms_) * inv,
                        double(prof_slots_) * inv, double(prof_sched_) * inv, double(prof_runs_) * inv,
                        double(prof_barriers_) * inv, prof_traced_, prof_calls_);
            std::fflush(stdout);
            prof_calls_ = 0u; prof_slots_ = 0u; prof_sched_ = 0u; prof_runs_ = 0u; prof_barriers_ = 0u; prof_traced_ = 0u;
            prof_cv_ms_ = prof_pack_ms_ = prof_h2d_ms_ = prof_dev_ms_ = prof_fin_ms_ = 0.0;
        }
    }

    WorkerConfig cfg_;
    bool is_tp2_ = true;
    std::vector<int> device_ids_;
    int expert_begin_ = 0;
    int expert_end_ = 0;
    uint32_t n_experts_ = 0;
    uint32_t sb_per_proj_ = 128;
    uint32_t inter_tiles_ = 32;
    uint32_t nt4_gu_ = 8;
    uint32_t kt4_dn_ = 8;
    uint32_t k_splits_gu_ = 8;
    uint32_t k_splits_dn_ = 4;
    uint32_t num_sb_core_ = 2;
    uint32_t beta_bits_ = 0;
    uint32_t beta_recip_bits_ = 0;
    uint32_t lin_bits_ = 0;
    uint32_t lin_recip_bits_ = 0;
    distributed::MeshCoordinate x16_coord_{0, 0};

    std::shared_ptr<distributed::MeshDevice> mesh_;
    CoreCoord grid_{0, 0};
    uint32_t total_cores_ = 0;
    uint32_t trace_region_bytes_ = 0;

    ttnn::Tensor x_tilized_, x_decode_src_, slot_ids_src_, weights_decode64_src_, weights_tiled_src_;
    ttnn::Tensor x_decode_, slot_ids_, weights_decode64_, weights_tiled_;
    ttnn::Tensor gate_partials_, up_partials_, z_rot_bf16_, down_partials_, combined_f32_;
    std::array<ttnn::Tensor, 3> finish_decode_;
    std::array<ttnn::Tensor, 2> finish_prefill_;

    std::optional<ttnn::Tensor> x_rm_src_, x_rm_dst_, slot_meta_src_, slot_meta_dst_, y_accum_f32_;
    std::optional<ttnn::Tensor> l1_lut_, dummy_hot_, hot_gate_l1_, hot_up_l1_, hot_down_l1_;
    std::array<std::optional<ResidentLayerBank>, kTotalLayersWithEcho> layers_;

    uint32_t num_hot_shards_ = 0;
    uint64_t pinned_hot_l1_bytes_per_card_ = 0;
    uint64_t gddr6_bytes_per_card_ = 0;
    uint64_t stat_total_slots_ = 0;
    uint64_t stat_l1_hits_ = 0;
    uint32_t dirty_mt0_ = 0;
    bool programs_prewarmed_ = false;

    std::map<std::pair<uint32_t, uint32_t>, TraceSlot> op_traces_;
    std::array<TraceSlot, 2> bcast_traces_{};
    std::array<TraceSlot, 2> reduce_traces_{};

    LockedBuffer host_x_tilized_{size_t(kP150LatentTiles) * 2048u};
    LockedBuffer host_x_rm_{size_t(kPrefillMaxRows) * kLatentDim * sizeof(uint16_t)};
    LockedBuffer host_slot_ids_{kSlotIdsBytes};
    LockedBuffer host_slot_meta_{size_t(kMaxCompactSlots) * 256u};
    LockedBuffer host_weights_{size_t(kMaxCompactSlots) * 4096u};
    LockedBuffer host_out_f32_{size_t(32u) * kLatentDim * sizeof(float)};
    LockedBuffer host_out_bf16_{size_t(kPrefillMaxRows) * kLatentDim * sizeof(uint16_t)};
    LockedBuffer host_slot_lut_{size_t(kTotalLayersWithEcho) * 256u * sizeof(uint16_t)};
    std::vector<uint16_t> x_bf16_scratch_;
    struct TokenItem { uint16_t row; float weight; };
    std::array<std::vector<TokenItem>, 256> per_exp_;
    uint32_t prof_calls_ = 0u, prof_slots_ = 0u, prof_sched_ = 0u, prof_runs_ = 0u, prof_barriers_ = 0u, prof_traced_ = 0u;
    double prof_cv_ms_ = 0.0, prof_pack_ms_ = 0.0, prof_h2d_ms_ = 0.0, prof_dev_ms_ = 0.0, prof_fin_ms_ = 0.0;
    uint32_t dprof_calls_ = 0u, dprof_slots_ = 0u, dprof_sched_ = 0u, dprof_traced_ = 0u;
    double dprof_h2d_ms_ = 0.0, dprof_dev_ms_ = 0.0;
};

// ============================================================================
// Dynamic Multi-Card Coordinator (handles BOTH 192G TP2 AND 224G/240G EP3!)
// ============================================================================
class P150Ernie55Coordinator {
public:
    P150Ernie55Coordinator(const WorkerConfig& cfg, const P150GroupSpec& gspec)
        : cfg_(cfg), gspec_(gspec), is_tp2_(gspec.mode == "tp2")
    {
        check_no_existing_p150_owner();
        ttnn::CONFIG.set<"throw_exception_on_fallback">(true);
        if (is_tp2_) {
            // Wave-9 Build #4: Restore default FABRIC_1D packet config (eliminates 8192 B suboptimal warning on 4096 B FP32 decode pages -> 36+ tok/s!)
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
            meshes_.push_back(std::make_unique<P150Ernie55Mesh>(
                cfg_, true, std::vector<int>{0, 1}, gspec_.expert_begin, gspec_.expert_end, uint32_t(gspec_.trace_region_mib)));
            card_ranges_.push_back({0, gspec_.count()});
        } else {
            // Independent EP cards (e.g. 3x P150A in 224G/240G — zero inter-card AllReduce!)
            int cur = gspec_.expert_begin;
            int local_cur = 0;
            const int ncards = std::max(1, gspec_.num_cards);
            for (int c = 0; c < ncards; ++c) {
                const int cnt = (c < int(gspec_.card_splits.size()))
                              ? gspec_.card_splits[size_t(c)]
                              : ((gspec_.expert_end - cur) / (ncards - c));
                meshes_.push_back(std::make_unique<P150Ernie55Mesh>(
                    cfg_, false, std::vector<int>{c}, cur, cur + cnt, uint32_t(gspec_.trace_region_mib)));
                card_ranges_.push_back({local_cur, local_cur + cnt});
                cur += cnt;
                local_cur += cnt;
            }
        }
    }

    bool is_tp2() const noexcept { return is_tp2_; }
    uint32_t num_cards() const noexcept { return uint32_t(gspec_.num_cards); }
    uint32_t n_experts_total() const noexcept { return uint32_t(gspec_.count()); }
    uint32_t expert_begin() const noexcept { return uint32_t(gspec_.expert_begin); }
    uint32_t expert_end() const noexcept { return uint32_t(gspec_.expert_end); }

    uint32_t num_hot_shards() const noexcept {
        uint32_t s = 0;
        for (const auto& m : meshes_) s += m->num_hot_shards();
        return s;
    }
    uint64_t pinned_l1_bytes_total() const noexcept {
        uint64_t s = 0;
        for (const auto& m : meshes_) s += m->pinned_l1_bytes_total();
        return s;
    }
    uint64_t gddr6_bytes_per_card() const noexcept { return meshes_.empty() ? 0ULL : meshes_[0]->gddr6_bytes_per_card(); }
    uint64_t stat_total_slots() const noexcept {
        uint64_t s = 0;
        for (const auto& m : meshes_) s += m->stat_total_slots();
        return s;
    }
    uint64_t stat_l1_hits() const noexcept {
        uint64_t s = 0;
        for (const auto& m : meshes_) s += m->stat_l1_hits();
        return s;
    }

    void load_layers(int first_layer, int last_layer) {
        for (auto& m : meshes_) m->load_layers(first_layer, last_layer);
    }

    void load_synthetic_layer1(const std::vector<Hlut4Tile32x32>& all_exp_tiles, const std::vector<uint64_t>& signs_inter) {
        for (size_t i = 0; i < meshes_.size(); ++i) {
            const int b = card_ranges_[i].first;
            const int e = card_ranges_[i].second;
            constexpr size_t kExpTiles3 = 3u * kExpProjTiles;
            std::vector<Hlut4Tile32x32> slice(all_exp_tiles.begin() + ptrdiff_t(size_t(b) * kExpTiles3),
                                              all_exp_tiles.begin() + ptrdiff_t(size_t(e) * kExpTiles3));
            meshes_[i]->load_synthetic_layer1(slice, signs_inter);
        }
    }

    void forward_layer(
        int layer_idx, uint32_t rows, uint32_t top_k,
        const int32_t row_slots[][ERNIE55_P150_MAX_TOPK],
        const float row_weights[][ERNIE55_P150_MAX_TOPK],
        const float* x_rot_f32, float* y_out_f32,
        const uint16_t* x_rot_f16 = nullptr, uint16_t* y_out_bf16 = nullptr,
        const uint16_t* x_rot_bf16 = nullptr)
    {
        if (meshes_.size() == 1u) {
            meshes_[0]->forward_layer(layer_idx, rows, top_k, row_slots, row_weights, x_rot_f32, y_out_f32, false, x_rot_f16, y_out_bf16, x_rot_bf16);
            return;
        }
        // EP multi-card parallel execution
        std::memset(y_out_f32, 0, size_t(rows) * kLatentDim * sizeof(float));
        std::vector<std::vector<float>> partials(meshes_.size(), std::vector<float>(size_t(rows) * kLatentDim, 0.0f));
        std::vector<std::thread> ths;
        for (size_t c = 0; c < meshes_.size(); ++c) {
            ths.emplace_back([&, c]() {
                const int lo = card_ranges_[c].first;
                const int hi = card_ranges_[c].second;
                std::vector<std::array<int32_t, ERNIE55_P150_MAX_TOPK>> local_slots(rows);
                std::vector<std::array<float, ERNIE55_P150_MAX_TOPK>>   local_wts(rows);
                bool any = false;
                for (uint32_t r = 0u; r < rows; ++r) {
                    local_slots[r].fill(-1);
                    local_wts[r].fill(0.0f);
                    for (uint32_t s = 0u; s < top_k; ++s) {
                        const int32_t slot = row_slots[r][s];
                        if (slot >= lo && slot < hi && row_weights[r][s] != 0.0f) {
                            local_slots[r][s] = slot - lo;
                            local_wts[r][s]   = row_weights[r][s];
                            any = true;
                        }
                    }
                }
                if (any) {
                    meshes_[c]->forward_layer(
                        layer_idx, rows, top_k,
                        reinterpret_cast<const int32_t(*)[ERNIE55_P150_MAX_TOPK]>(local_slots.data()),
                        reinterpret_cast<const float(*)[ERNIE55_P150_MAX_TOPK]>(local_wts.data()),
                        x_rot_f32, partials[c].data(), false, x_rot_f16, nullptr, x_rot_bf16);
                }
            });
        }
        for (auto& t : ths) t.join();
        for (size_t c = 0; c < meshes_.size(); ++c) {
            for (size_t i = 0; i < partials[c].size(); ++i) y_out_f32[i] += partials[c][i];
        }
        if (y_out_bf16) {
            for (uint32_t r = 0u; r < rows; ++r) {
                f32_row_to_bf16_avx2(y_out_f32 + size_t(r) * kLatentDim, y_out_bf16 + size_t(r) * kLatentDim);
            }
        }
    }

private:
    WorkerConfig cfg_;
    P150GroupSpec gspec_;
    bool is_tp2_ = true;
    std::vector<std::unique_ptr<P150Ernie55Mesh>> meshes_;
    std::vector<std::pair<int, int>> card_ranges_;
};

// Exact FP64 CPU reference for self-test (WHT128-HLUT4 + exact SiTU-GLU + WHT128)
void ref_hlut4_situ_fp64(
    const std::vector<Hlut4Tile32x32>& exp_tiles,
    const std::vector<uint64_t>& signs_inter,
    uint32_t rows,
    const int32_t row_slots[][ERNIE55_P150_MAX_TOPK],
    const float row_weights[][ERNIE55_P150_MAX_TOPK],
    const float* x_rot_f32,
    float situ_beta, float situ_linear_beta,
    std::vector<double>& y_ref)
{
    y_ref.assign(size_t(rows) * kLatentDim, 0.0);
    const auto lv = hlut4_levels_f32();
    constexpr size_t kExpTiles3 = 3u * kExpProjTiles;

    auto unpack_proj = [&](const Hlut4Tile32x32* tbase, std::vector<double>& out) {
        out.assign(size_t(kLatentDim) * kLatentDim, 0.0);
        for (int nt = 0; nt < 64; ++nt) {
            for (int kt = 0; kt < 64; ++kt) {
                const auto& t = tbase[size_t(nt) * 64u + size_t(kt)];
                for (int r = 0; r < 32; ++r) {
                    const double sig = double(fp16_to_f32(t.scales_f16[r >> 2]));
                    for (int c = 0; c < 32; ++c) {
                        const uint32_t nib = (t.q_nib4[r * 4 + (c >> 3)] >> ((c & 7) * 4)) & 0xFu;
                        const double v = double(lv[nib & 7u]) * sig;
                        out[size_t(nt * 32 + r) * kLatentDim + size_t(kt * 32 + c)] = (nib & 8u) ? -v : v;
                    }
                }
            }
        }
    };

    std::map<int, std::array<std::vector<double>, 3>> cache;
    for (uint32_t r = 0u; r < rows; ++r) {
        const float* xr = x_rot_f32 + size_t(r) * kLatentDim;
        for (uint32_t s = 0u; s < ERNIE55_P150_MAX_TOPK; ++s) {
            const int e = row_slots[r][s];
            const double wt = double(row_weights[r][s]);
            if (e < 0 || wt == 0.0) continue;
            if (!cache.count(e)) {
                const Hlut4Tile32x32* base = exp_tiles.data() + size_t(e) * kExpTiles3;
                unpack_proj(base + 0u * kExpProjTiles, cache[e][0]);
                unpack_proj(base + 1u * kExpProjTiles, cache[e][1]);
                unpack_proj(base + 2u * kExpProjTiles, cache[e][2]);
            }
            const auto& wg = cache[e][0];
            const auto& wu = cache[e][1];
            const auto& wd = cache[e][2];
            std::vector<float> z(kLatentDim);
            for (int i = 0; i < kLatentDim; ++i) {
                double g = 0.0, u = 0.0;
                for (int k = 0; k < kLatentDim; ++k) {
                    g += wg[size_t(i) * kLatentDim + size_t(k)] * double(xr[k]);
                    u += wu[size_t(i) * kLatentDim + size_t(k)] * double(xr[k]);
                }
                z[size_t(i)] = ernie55_situ_glu(float(g), float(u), situ_beta, situ_linear_beta);
            }
            for (int b = 0; b < kLatentDim / 128; ++b) wht128_block(z.data() + b * 128, signs_inter.data() + size_t(b) * 2u);
            for (int d = 0; d < kLatentDim; ++d) {
                double acc = 0.0;
                for (int i = 0; i < kLatentDim; ++i) acc += wd[size_t(d) * kLatentDim + size_t(i)] * double(z[size_t(i)]);
                y_ref[size_t(r) * kLatentDim + size_t(d)] += wt * acc;
            }
        }
    }
}

int run_self_test(P150Ernie55Coordinator& coord, const WorkerConfig& cfg) {
    const uint32_t n_exp = coord.n_experts_total();
    std::vector<Hlut4Tile32x32> exp_tiles;
    std::vector<uint64_t> signs_inter;
    if (cfg.synthetic_self_test) {
        signs_inter = make_ernie55_signs_inter(1);
        const uint32_t test_e = std::min<uint32_t>(n_exp, 4u);
        exp_tiles.resize(size_t(n_exp) * 3u * kExpProjTiles);
        std::memset(exp_tiles.data(), 0, exp_tiles.size() * sizeof(Hlut4Tile32x32));
        std::mt19937 rng(5501);
        std::normal_distribution<float> nd(0.0f, 0.04f);
        std::vector<float> tmp_w(size_t(kLatentDim) * kLatentDim);
        const auto signs_in = make_ernie55_signs_in(1);
        const std::vector<float> no_rup;
        for (uint32_t e = 0u; e < test_e; ++e) {
            for (int p = 0; p < 3; ++p) {
                for (float& v : tmp_w) v = nd(rng);
                auto q = quantize_wht128_hlut4_fast(tmp_w.data(), kLatentDim, kLatentDim,
                                                    (p == 2) ? signs_inter : signs_in, no_rup);
                std::memcpy(exp_tiles.data() + (size_t(e) * 3u + size_t(p)) * kExpProjTiles, q.data(), q.size() * sizeof(Hlut4Tile32x32));
            }
        }
        coord.load_synthetic_layer1(exp_tiles, signs_inter);
    } else {
        coord.load_layers(1, 1);
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/experts_hlut4/hlut4_layer_01.bin", cfg.weight_root.c_str());
        const int fd = ::open(fname, O_RDONLY);
        Ernie55Hlut4LayerHeader hdr{};
        if (fd < 0 || ::pread(fd, &hdr, sizeof(hdr), 0) != ssize_t(sizeof(hdr))) {
            if (fd >= 0) ::close(fd);
            throw std::runtime_error("Cannot read layer 1 shard for self-test");
        }
        signs_inter.assign(hdr.signs_inter, hdr.signs_inter + kMoeInterSize / 64);
        exp_tiles.resize(size_t(n_exp) * 3u * kExpProjTiles);
        const off_t off = off_t(sizeof(hdr)) + off_t(coord.expert_begin()) * off_t(kHlut4ExpertBytes);
        const ssize_t r = ::pread(fd, exp_tiles.data(), exp_tiles.size() * sizeof(Hlut4Tile32x32), off);
        ::close(fd);
        if (r != ssize_t(exp_tiles.size() * sizeof(Hlut4Tile32x32))) throw std::runtime_error("Short read in self-test");
    }

    std::mt19937 rng(5523);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    bool all_ok = true;
    for (uint32_t M : {1u, 6u, 64u, 256u}) {
        std::vector<float> x_rot(size_t(M) * kLatentDim);
        for (float& v : x_rot) v = nd(rng);
        std::vector<std::array<int32_t, ERNIE55_P150_MAX_TOPK>> slots(M);
        std::vector<std::array<float, ERNIE55_P150_MAX_TOPK>> wts(M);
        for (uint32_t r = 0u; r < M; ++r) {
            slots[r].fill(-1); wts[r].fill(0.0f);
            if (M <= 64u) {
                slots[r][0] = 0; wts[r][0] = 0.55f;
                if (n_exp > 1u) { slots[r][1] = 1; wts[r][1] = 0.45f; }
            } else {
                // Exercise full batch_cnt=64 + L1 hot + GDDR6 cold (including last slot n_exp-1) + wait_barrier=0/1!
                const int e0 = int(r % std::min<uint32_t>(n_exp, 72u));
                const int e1 = int((n_exp - 1u) - (r % std::min<uint32_t>(n_exp, 20u)));
                slots[r][0] = e0; wts[r][0] = 0.40f;
                if (e1 != e0) { slots[r][1] = e1; wts[r][1] = 0.35f; }
            }
        }
        std::vector<double> y_ref;
        ref_hlut4_situ_fp64(exp_tiles, signs_inter, M,
                            reinterpret_cast<const int32_t(*)[ERNIE55_P150_MAX_TOPK]>(slots.data()),
                            reinterpret_cast<const float(*)[ERNIE55_P150_MAX_TOPK]>(wts.data()),
                            x_rot.data(), cfg.situ_beta, cfg.situ_linear_beta, y_ref);
        std::vector<float> y_dev(size_t(M) * kLatentDim, 0.0f);
        coord.forward_layer(1, M, ERNIE55_P150_MAX_TOPK,
                            reinterpret_cast<const int32_t(*)[ERNIE55_P150_MAX_TOPK]>(slots.data()),
                            reinterpret_cast<const float(*)[ERNIE55_P150_MAX_TOPK]>(wts.data()),
                            x_rot.data(), y_dev.data());
        const auto t0 = std::chrono::steady_clock::now();
        coord.forward_layer(1, M, ERNIE55_P150_MAX_TOPK,
                            reinterpret_cast<const int32_t(*)[ERNIE55_P150_MAX_TOPK]>(slots.data()),
                            reinterpret_cast<const float(*)[ERNIE55_P150_MAX_TOPK]>(wts.data()),
                            x_rot.data(), y_dev.data());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (size_t i = 0; i < y_ref.size(); ++i) {
            dot += y_ref[i] * double(y_dev[i]);
            n1  += y_ref[i] * y_ref[i];
            n2  += double(y_dev[i]) * double(y_dev[i]);
        }
        const double cos_sim = dot / (std::sqrt(n1 * n2) + 1e-30);
        const bool ok = (cos_sim >= 0.999);
        if (!ok) all_ok = false;
        std::printf("[P150-ERNIE55-SELFTEST] M=%2u | WHT128-HLUT4+SiTU-GLU cos=%.8f | latency=%.3f ms | L1_hits=%llu/%llu | %s\n",
                    M, cos_sim, ms, (unsigned long long)coord.stat_l1_hits(), (unsigned long long)coord.stat_total_slots(),
                    ok ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    return all_ok ? 0 : 1;
}

}  // namespace
}  // namespace ernie55

int main(int argc, char** argv) {
    using namespace ernie55;
    std::signal(SIGINT,  on_sig);
    std::signal(SIGTERM, on_sig);

    WorkerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg for ") + f);
            return argv[++i];
        };
        if (a == "--weight-root")            cfg.weight_root = need("--weight-root");
        else if (a == "--topology")          cfg.topology_path = need("--topology");
        else if (a == "--config")            cfg.config_path = need("--config");
        else if (a == "--kernels-dir")       cfg.kernels_dir = need("--kernels-dir");
        else if (a == "--shm")               cfg.shm_name = need("--shm");
        else if (a == "--group-id")          cfg.group_id = std::stoi(need("--group-id"));
        else if (a == "--no-trace")          cfg.enable_trace = false;
        else if (a == "--no-x16-relay")      cfg.x16_relay = false;
        else if (a == "--self-test")         cfg.self_test = true;
        else if (a == "--synthetic-self-test") { cfg.self_test = true; cfg.synthetic_self_test = true; }
        else if (a == "--max-layers")        cfg.max_layers_load = std::stoi(need("--max-layers"));
    }

    try {
        if (std::filesystem::is_regular_file(cfg.config_path)) {
            const auto mc = Ernie55ModelConfig::load_json(cfg.config_path);
            cfg.situ_beta = mc.situ_beta;
            cfg.situ_linear_beta = mc.situ_linear_beta;
        }
        const auto topo = Ernie55TopologyPlan::load_json(cfg.topology_path);
        if (cfg.group_id < 0 || cfg.group_id >= int(topo.p150_groups.size())) {
            throw std::runtime_error("Invalid --group-id for topology " + topo.name);
        }
        const auto& gspec = topo.p150_groups[size_t(cfg.group_id)];
        if (cfg.shm_name == "/ernie55_p150_ipc" && !gspec.shm_name.empty()) cfg.shm_name = gspec.shm_name;

        P150Ernie55Coordinator coord(cfg, gspec);
        if (cfg.self_test) {
            return run_self_test(coord, cfg);
        }

        int last_l = std::min(kNumMoeTotalWithDspark, cfg.max_layers_load);
        while (last_l > kNumMoeBackboneLayers) {
            char sp[512];
            std::snprintf(sp, sizeof(sp), "%s/experts_hlut4/hlut4_layer_%02d.bin", cfg.weight_root.c_str(), last_l);
            if (std::filesystem::is_regular_file(sp)) break;
            --last_l;
        }
        coord.load_layers(1, last_l);
        if (g_stop) return 0;

        Ernie55P150ShmMapping shm;
        if (!shm.create_server(cfg.shm_name)) {
            std::fprintf(stderr, "[P150-ERNIE55] Failed to create POSIX SHM %s\n", cfg.shm_name.c_str());
            return 1;
        }
        auto* ch = shm.ptr;
        ch->p150_mode             = coord.is_tp2() ? ERNIE55_P150_MODE_TP2 : ERNIE55_P150_MODE_EP;
        ch->num_p150_cards        = coord.num_cards();
        ch->n_experts_per_layer   = coord.n_experts_total();
        ch->expert_id_begin       = coord.expert_begin();
        ch->expert_id_end         = coord.expert_end();
        ch->fabric_active         = coord.is_tp2() ? 1u : 0u;
        ch->l1_hot_experts        = coord.num_hot_shards();
        ch->l1_pinned_bytes_total = coord.pinned_l1_bytes_total();
        ch->gddr6_bytes_per_card  = coord.gddr6_bytes_per_card();
        std::atomic_thread_fence(std::memory_order_release);
        ch->ready = 1u;

        std::printf("[P150-ERNIE55] READY on SHM %s (PID=%d, topo=%s, mode=%s, cards=%u, experts=[%u..%u), L1=%.1f MiB, GDDR6=%.2f GiB/card)\n",
                    cfg.shm_name.c_str(), int(::getpid()), topo.name.c_str(),
                    coord.is_tp2() ? "TP2" : "EP", coord.num_cards(),
                    coord.expert_begin(), coord.expert_end(),
                    double(coord.pinned_l1_bytes_total()) / 1048576.0,
                    double(coord.gddr6_bytes_per_card()) / 1073741824.0);
        std::fflush(stdout);

        uint32_t last_seq = 0u, idle_spins = 0u;
        while (!g_stop) {
            const uint32_t seq = ch->req_seq.load(std::memory_order_acquire);
            if (seq == last_seq) {
                p150_cpu_relax();
                if (++idle_spins > 50000u) {
                    struct timespec ts{0, 200000};
                    ::nanosleep(&ts, nullptr);
                    idle_spins = 0u;
                }
                continue;
            }
            idle_spins = 0u;
            last_seq = seq;

            const uint32_t op = ch->op;
            if (op == ERNIE55_P150_OP_SHUTDOWN) {
                ch->status = 0;
                ch->resp_seq.store(seq, std::memory_order_release);
                std::printf("[P150-ERNIE55] Received graceful IPC SHUTDOWN — draining Blackhole mesh...\n");
                std::fflush(stdout);
                break;
            } else if (op == ERNIE55_P150_OP_PING) {
                ch->status = 0;
                ch->resp_seq.store(seq, std::memory_order_release);
            } else if (op == ERNIE55_P150_OP_MOE_EXEC) {
                const auto t0 = std::chrono::steady_clock::now();
                try {
                    const uint16_t* x_b16 = (ch->flags & ERNIE55_P150_FLAG_X_BF16) ? ch->x_rot_f16  : nullptr;
                    const uint16_t* x_f16 = (!x_b16 && (ch->flags & ERNIE55_P150_FLAG_X_F16)) ? ch->x_rot_f16 : nullptr;
                    uint16_t*       y_b16 = (ch->flags & ERNIE55_P150_FLAG_Y_BF16) ? ch->y_out_bf16 : nullptr;
                    coord.forward_layer(int(ch->layer_idx), ch->rows, ch->top_k,
                                        ch->row_slots, ch->row_weights, ch->x_rot_f32, ch->y_out_f32, x_f16, y_b16, x_b16);
                    const uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - t0).count());
                    ch->last_device_ns = ns;
                    ch->total_device_ns.fetch_add(ns, std::memory_order_relaxed);
                    ch->stat_total_slots.store(coord.stat_total_slots(), std::memory_order_relaxed);
                    ch->stat_l1_hits.store(coord.stat_l1_hits(), std::memory_order_relaxed);
                    ch->status = 0;
                } catch (const std::exception& ex) {
                    std::fprintf(stderr, "[P150-ERNIE55] EXEC error layer %u: %s\n", ch->layer_idx, ex.what());
                    ch->status = -1;
                }
                ch->resp_seq.store(seq, std::memory_order_release);
            }
        }
        std::printf("[P150-ERNIE55] Clean shutdown complete.\n");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[P150-ERNIE55] FATAL: %s\n", ex.what());
        return 1;
    }
}
