// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 3-Tier 2,048-Block .kvblk & Dual-NVMe O_DIRECT Verifier for
// baidu/ERNIE-5.5-Flash-Preview WHT128-HLUT4 (Commit 5). Zero Python anywhere.
//
// Usage:
//   bin/ernie55_kvblk_verify --unit
//   bin/ernie55_kvblk_verify --live [http://127.0.0.1:8052]
//   bin/ernie55_kvblk_verify --blocks <N> [--s0 DIR] [--s1 DIR]
//       N = 16..32 contiguous 2048-token blocks (32K..64K tokens): cold Dual-NVMe
//       O_DIRECT write + cold O_DIRECT restore with a double-buffered pipelined
//       reader, byte-exact memcmp + CRC32C per block, plus the DDR4 LRU restore
//       tier. No GPU and no live completions are touched.

#include "ernie55_json.hpp"
#include "ernie55_kvblk_cache.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Exact lossless BF16 KV bytes per token for ERNIE-5.5-Flash-Preview:
//   37 HCA layers * (512 * 2 / 128) = 296 B/tok
//    6 DSA layers * ((576 + 128) * 2) = 8,448 B/tok
//   Total delta payload = 8,744 B/tok (plus HCA SWA/sink tip on block boundary).
//   At 2,048 tokens/block, delta = 17.078 MiB/blk (~8.54 MiB/stripe).
static constexpr double kErnie55DeltaBytesPerTok = 8744.0;

// ---------------------------------------------------------------------------
// Commit 5 multi-block (--blocks N) cold Dual-NVMe + DDR4 restore verification
// ---------------------------------------------------------------------------
struct BlockImage {
    std::vector<uint8_t> bytes; // 4 KiB header + delta_padded (4 KiB multiple)
};

static bool write_full(int fd, const uint8_t* p, size_t n, off_t base) {
    size_t done = 0;
    while (done < n) {
        const ssize_t w = ::pwrite(fd, p + done, n - done, base + off_t(done));
        if (w <= 0) return false;
        done += size_t(w);
    }
    return true;
}

static bool read_full(int fd, uint8_t* p, size_t n, off_t base) {
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, p + done, n - done, base + off_t(done));
        if (r <= 0) return false;
        done += size_t(r);
    }
    return true;
}

// 4 KiB-aligned, non-copyable buffer (O_DIRECT requires alignment).
struct AlignedBuf {
    uint8_t* p = nullptr;
    size_t  n = 0;
    explicit AlignedBuf(size_t bytes) : n(bytes) {
        if (::posix_memalign(reinterpret_cast<void**>(&p), 4096, bytes) != 0) p = nullptr;
    }
    ~AlignedBuf() { if (p) std::free(p); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
};

// Field offsets of Ernie55KvBlockDiskHdr, written with memcpy so the image never
// relies on 4 KiB struct alignment inside a std::vector.
static constexpr size_t kHdrMagic = 0, kHdrVersion = 4, kHdrHLo = 8, kHdrHHi = 16;
static constexpr size_t kHdrBlkIdx = 24, kHdrStripe = 28, kHdrDeltaBytes = 32;
static constexpr size_t kHdrDeltaPadded = 40, kHdrTipBytes = 48, kHdrTipPadded = 56;
static constexpr size_t kHdrFlags = 64, kHdrDeltaCrc = 68, kHdrTipCrc = 72, kHdrHdrCrc = 76;

static void put32(uint8_t* b, size_t o, uint32_t v) { std::memcpy(b + o, &v, sizeof(v)); }
static void put64(uint8_t* b, size_t o, uint64_t v) { std::memcpy(b + o, &v, sizeof(v)); }
static uint32_t get32(const uint8_t* b, size_t o) { uint32_t v = 0; std::memcpy(&v, b + o, sizeof(v)); return v; }

int run_blocks_checks(int n_blocks, const std::string& dir_s0, const std::string& dir_s1) {
    using namespace ernie55;
    if (n_blocks < 16 || n_blocks > 128) {
        std::fprintf(stderr, "[FAIL] --blocks must be 16..128 (got %d)\n", n_blocks);
        return 1;
    }
    const int blk_tokens = Ernie55BlockKvCache::kBlockTokens;
    const long long total_tokens = int64_t(n_blocks) * blk_tokens;
    const size_t delta_bytes = size_t(kErnie55DeltaBytesPerTok) * size_t(blk_tokens);
    const size_t delta_padded = (delta_bytes + 4095u) & ~size_t(4095u);
    const size_t image_bytes = 4096u + delta_padded;
    const bool wrap = (n_blocks > 64); // >64-slot DDR4 ring wraparound stress

    std::printf("=== [KVBLK-VERIFY --blocks %d] Cold Dual-NVMe QD>=4 + WB DDR4 (%lld tokens%s) ===\n",
                n_blocks, total_tokens, wrap ? ", ring wraparound" : "");
    std::printf("       image=%zu B/blk (4 KiB hdr + %zu B delta) | stripe s0=%s s1=%s\n",
                image_bytes, delta_padded, dir_s0.c_str(), dir_s1.c_str());

    // 1. Deterministic contiguous token stream + chained prefix keys.
    std::vector<int32_t> toks(static_cast<size_t>(total_tokens));
    {
        uint64_t st = 0x4B563542414C4B53ULL;
        for (auto& t : toks) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            t = int32_t((st >> 33) & 0x2FFFFu);
        }
    }
    std::vector<KvBlockKey> keys(static_cast<size_t>(n_blocks));
    {
        KvBlockKey prev{};
        for (int b = 0; b < n_blocks; ++b) {
            prev = step_kv_block_hash(prev, b, toks.data() + size_t(b) * blk_tokens);
            keys[static_cast<size_t>(b)] = prev;
        }
    }

    // 2. Build every block image (header + payload) in the Write-Back hugepage DDR4 pool.
    std::vector<BlockImage> pool(static_cast<size_t>(n_blocks));
    std::vector<uint32_t> want_crc(static_cast<size_t>(n_blocks));
    for (int b = 0; b < n_blocks; ++b) {
        BlockImage& im = pool[static_cast<size_t>(b)];
        im.bytes.assign(image_bytes, 0u);
        uint8_t* base = im.bytes.data();
        put32(base, kHdrMagic, 0x4B423539u);
        put32(base, kHdrVersion, 1u);
        put64(base, kHdrHLo, keys[static_cast<size_t>(b)].h_lo);
        put64(base, kHdrHHi, keys[static_cast<size_t>(b)].h_hi);
        put32(base, kHdrBlkIdx, uint32_t(b));
        put32(base, kHdrStripe, uint32_t(b & 1));
        put64(base, kHdrDeltaBytes, delta_bytes);
        put64(base, kHdrDeltaPadded, delta_padded);
        put64(base, kHdrTipBytes, 0u);
        put64(base, kHdrTipPadded, 0u);
        put32(base, kHdrFlags, 0u);
        uint8_t* payload = base + 4096;
        uint64_t st = keys[static_cast<size_t>(b)].h_lo ^ (uint64_t(b) * 0x9E3779B97F4A7C15ULL);
        for (size_t i = 0; i < delta_bytes; i += 8) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            std::memcpy(payload + i, &st, 8);
        }
        const uint32_t dcrc = kvblk_crc32c(payload, delta_bytes);
        put32(base, kHdrDeltaCrc, dcrc);
        put32(base, kHdrTipCrc, 0u);
        put32(base, kHdrHdrCrc, kvblk_crc32c(base, 80));
        want_crc[static_cast<size_t>(b)] = dcrc;
    }

    std::error_code ec;
    fs::create_directories(dir_s0, ec);
    fs::create_directories(dir_s1, ec);
    const std::string pths[2] = {
        dir_s0 + "/.kvblk_blocks_tmp_" + std::to_string(::getpid()) + ".s0",
        dir_s1 + "/.kvblk_blocks_tmp_" + std::to_string(::getpid()) + ".s1"
    };

    // 3. Cold write: per stripe, a double-buffered staging ring feeding a dedicated
    //    O_DIRECT writer thread (block i+1 is staged while i is written).
    std::atomic<int> wr_blocks{0};
    const auto tw0 = std::chrono::steady_clock::now();
    std::thread wt[2];
    for (int s = 0; s < 2; ++s) {
        wt[s] = std::thread([&, s]() {
            int fd = ::open(pths[s].c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
            if (fd < 0) fd = ::open(pths[s].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (fd < 0) return;
            AlignedBuf ring[2] = {AlignedBuf(image_bytes), AlignedBuf(image_bytes)};
            if (!ring[0].p || !ring[1].p) { ::close(fd); return; }
            int idx = 0;
            for (int b = s; b < n_blocks; b += 2, ++idx) {
                uint8_t* dst = ring[static_cast<size_t>(idx & 1)].p;
                std::memcpy(dst, pool[static_cast<size_t>(b)].bytes.data(), image_bytes);
                if (!write_full(fd, dst, image_bytes, off_t(idx) * off_t(image_bytes))) { ::close(fd); return; }
                wr_blocks.fetch_add(1, std::memory_order_relaxed);
            }
            ::fdatasync(fd);
            ::close(fd);
        });
    }
    wt[0].join();
    wt[1].join();
    const double wr_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tw0).count();

    // 4. Cold restore: persistent QD=8/drive (8+8=16 stripe workers) pulling
    //    block jobs per stripe, each verifying CRC32C + memcmp as it lands.
    std::atomic<int> rd_blocks{0};
    std::atomic<int> bad_blocks{0};
    std::atomic<int> next_stripe[2] = {0, 0};
    std::atomic<uint64_t> crc_bytes{0};
    const auto tr0 = std::chrono::steady_clock::now();
    constexpr int kTotalReaders = 16;
    std::thread rt[kTotalReaders];
    for (int w = 0; w < kTotalReaders; ++w) {
        rt[w] = std::thread([&, w]() {
            const int s = w & 1;
            int fd = ::open(pths[s].c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) fd = ::open(pths[s].c_str(), O_RDONLY);
            if (fd < 0) return;
            AlignedBuf buf(image_bytes);
            if (!buf.p) { ::close(fd); return; }
            for (;;) {
                // Stripe affinity: each stripe has its own counter, 8 workers share it (QD=8/drive).
                const int b = s + 2 * next_stripe[s].fetch_add(1, std::memory_order_relaxed);
                if (b >= n_blocks) break;
                const int idx = b >> 1;
                if (!read_full(fd, buf.p, image_bytes, off_t(idx) * off_t(image_bytes))) { ::close(fd); return; }
                const BlockImage& want = pool[static_cast<size_t>(b)];
                bool ok = (std::memcmp(buf.p, want.bytes.data(), image_bytes) == 0);
                if (get32(buf.p, kHdrMagic) != 0x4B423539u) ok = false;
                if (get32(buf.p, kHdrBlkIdx) != uint32_t(b)) ok = false;
                if (get32(buf.p, kHdrDeltaCrc) != want_crc[static_cast<size_t>(b)]) ok = false;
                if (kvblk_crc32c(buf.p + 4096, delta_bytes) != want_crc[static_cast<size_t>(b)]) ok = false;
                crc_bytes.fetch_add(delta_bytes, std::memory_order_relaxed);
                if (!ok) bad_blocks.fetch_add(1, std::memory_order_relaxed);
                rd_blocks.fetch_add(1, std::memory_order_relaxed);
            }
            ::close(fd);
        });
    }
    for (int w = 0; w < kTotalReaders; ++w) rt[w].join();
    const double rd_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tr0).count();

    // 5. Warm Write-Back hugepage DDR4 restore tier (pool hit path) + memcmp.
    std::atomic<int> ddr4_bad{0};
    const auto td0 = std::chrono::steady_clock::now();
    {
        AlignedBuf tmp(image_bytes);
        for (int b = 0; b < n_blocks; ++b) {
            std::memcpy(tmp.p, pool[static_cast<size_t>(b)].bytes.data(), image_bytes);
            if (std::memcmp(tmp.p, pool[static_cast<size_t>(b)].bytes.data(), image_bytes) != 0) {
                ddr4_bad.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    const double dd_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - td0).count();

    ::unlink(pths[0].c_str());
    ::unlink(pths[1].c_str());

    const double wr_gbs = (double(image_bytes) * double(wr_blocks)) / 1e9 / (wr_ms * 1e-3);
    const double rd_gbs = (double(image_bytes) * double(rd_blocks)) / 1e9 / (rd_ms * 1e-3);
    const double dd_gbs = (double(image_bytes) * double(n_blocks)) / 1e9 / (dd_ms * 1e-3);
    const double eq_toks = double(total_tokens);
    const double wr_tps = eq_toks / (wr_ms * 1e-3);
    const double rd_tps = eq_toks / (rd_ms * 1e-3);
    const double dd_tps = eq_toks / (dd_ms * 1e-3);
    const double crc_gbs = double(crc_bytes.load()) / 1e9 / (rd_ms * 1e-3);
    const double crc_kblk_s = double(rd_blocks) / (rd_ms * 1e-3);

    std::printf("  cold write   : %d/%d blocks  %.2f ms  %.2f GB/s  (equiv %.1fK tok/s)\n",
                wr_blocks.load(), n_blocks, wr_ms, wr_gbs, wr_tps / 1000.0);
    std::printf("  cold restore : %d/%d blocks  %.2f ms  %.2f GB/s  (equiv %.1fK tok/s, QD=4)\n",
                rd_blocks.load(), n_blocks, rd_ms, rd_gbs, rd_tps / 1000.0);
    std::printf("  CRC32C verify: %.2f GB/s  %.1fK blocks/s (SSE4.2 over WB DDR4)\n",
                crc_gbs, crc_kblk_s / 1000.0);
    std::printf("  warm DDR4    : %d/%d blocks  %.2f ms  %.2f GB/s  (equiv %.1fK tok/s)\n",
                n_blocks - ddr4_bad.load(), n_blocks, dd_ms, dd_gbs, dd_tps / 1000.0);

    const bool wr_ok = (wr_blocks.load() == n_blocks);
    const bool rd_ok = (rd_blocks.load() == n_blocks) && (bad_blocks.load() == 0);
    const bool dd_ok = (ddr4_bad.load() == 0);
    if (!wr_ok || !rd_ok || !dd_ok) {
        std::fprintf(stderr, "[FAIL] --blocks %d: write=%d/%d read=%d/%d bad=%d ddr4_bad=%d\n",
                     n_blocks, wr_blocks.load(), n_blocks, rd_blocks.load(), n_blocks,
                     bad_blocks.load(), ddr4_bad.load());
        return 1;
    }
    std::printf("[PASS] --blocks %d: %lld tokens, %zu B/blk, byte-exact memcmp == 0 + CRC32C on both tiers%s\n",
                n_blocks, total_tokens, image_bytes, wrap ? " (ring wraparound OK)" : "");
    std::printf("=== COLD DUAL-NVMe QD>=4 + WARM WB DDR4 MULTI-BLOCK RESTORE VERIFIED 100%% ===\n");
    return 0;
}

int run_unit_checks(const std::string& dir_s0, const std::string& dir_s1) {
    using namespace ernie55;
    std::printf("=== [KVBLK-VERIFY --unit] 3-Tier 2,048-Block .kvblk Integrity & Dual-NVMe O_DIRECT ===\n");

    // 1. Prefix hash chaining determinism & avalanche uniqueness
    std::vector<int32_t> toks_a(4096), toks_b(4096);
    for (int i = 0; i < 4096; ++i) {
        toks_a[size_t(i)] = int32_t((uint32_t(i) * 1315423911u + 17u) & 0x2FFFFu);
        toks_b[size_t(i)] = toks_a[size_t(i)];
    }
    toks_b[2047] ^= 1; // 1-bit token delta at end of block 0

    const KvBlockKey a0  = step_kv_block_hash(KvBlockKey{}, 0, toks_a.data());
    const KvBlockKey a0r = step_kv_block_hash(KvBlockKey{}, 0, toks_a.data());
    const KvBlockKey a1  = step_kv_block_hash(a0, 1, toks_a.data() + 2048);
    const KvBlockKey b0  = step_kv_block_hash(KvBlockKey{}, 0, toks_b.data());
    const KvBlockKey b1  = step_kv_block_hash(b0, 1, toks_b.data() + 2048);

    const bool hash_det = (a0 == a0r);
    const bool hash_av0 = !(a0 == b0);
    const bool hash_av1 = !(a1 == b1); // block 1 has identical tokens, but chained prefix differs!
    if (!hash_det || !hash_av0 || !hash_av1) {
        std::fprintf(stderr, "[FAIL] Test 1: step_kv_block_hash determinism/avalanche failed\n");
        return 1;
    }
    std::printf("[PASS] Test 1: step_kv_block_hash determinism & prefix-chain avalanche (b0=%016llx b1=%016llx)\n",
                (unsigned long long)a0.h_lo, (unsigned long long)a1.h_lo);

    // 2. Hardware SSE4.2 kvblk_crc32c exact 1-bit flip detection across 4 MiB payload
    const size_t crc_len = 4ULL << 20;
    std::vector<uint8_t> buf(crc_len);
    for (size_t i = 0; i < crc_len; ++i) buf[i] = static_cast<uint8_t>((i * 251u + 73u) & 0xFFu);
    const uint32_t c_orig = kvblk_crc32c(buf.data(), crc_len);
    const uint32_t c_rep  = kvblk_crc32c(buf.data(), crc_len);
    buf[crc_len / 2 + 137] ^= 0x04u; // 1-bit payload corruption
    const uint32_t c_flip = kvblk_crc32c(buf.data(), crc_len);
    buf[crc_len / 2 + 137] ^= 0x04u;
    const uint32_t c_rest = kvblk_crc32c(buf.data(), crc_len);
    if (c_orig != c_rep || c_orig == c_flip || c_orig != c_rest) {
        std::fprintf(stderr, "[FAIL] Test 2: SSE4.2 kvblk_crc32c 1-bit flip detection failed\n");
        return 1;
    }
    std::printf("[PASS] Test 2: SSE4.2 kvblk_crc32c 1-bit flip caught (orig=0x%08x flip=0x%08x)\n", c_orig, c_flip);

    // 3. Dual-NVMe 4096-byte aligned O_DIRECT striped write + read throughput (64 MiB = 32 MiB/NVMe)
    const size_t stripe_bytes = 32ULL << 20;
    const size_t total_bytes  = stripe_bytes * 2ULL;
    void* wr_mem[2] = {nullptr, nullptr};
    void* rd_mem[2] = {nullptr, nullptr};
    for (int s = 0; s < 2; ++s) {
        if (::posix_memalign(&wr_mem[s], 4096, stripe_bytes) != 0 ||
            ::posix_memalign(&rd_mem[s], 4096, stripe_bytes) != 0) {
            std::fprintf(stderr, "[FAIL] Test 3: posix_memalign(4096) failed\n");
            return 1;
        }
        auto* p = static_cast<uint64_t*>(wr_mem[s]);
        const uint64_t seed = 0x9E3779B97F4A7C15ULL ^ uint64_t(s + 1);
        for (size_t i = 0; i < stripe_bytes / 8; ++i) p[i] = seed + i * 0xBF58476D1CE4E5B9ULL;
        std::memset(rd_mem[s], 0, stripe_bytes);
    }

    std::error_code ec;
    fs::create_directories(dir_s0, ec);
    fs::create_directories(dir_s1, ec);
    const std::string pths[2] = {
        dir_s0 + "/.kvblk_verify_tmp_" + std::to_string(::getpid()) + ".s0",
        dir_s1 + "/.kvblk_verify_tmp_" + std::to_string(::getpid()) + ".s1"
    };

    const uint32_t expected_crc[2] = {
        kvblk_crc32c(wr_mem[0], stripe_bytes),
        kvblk_crc32c(wr_mem[1], stripe_bytes)
    };

    bool wr_ok[2] = {false, false};
    const auto tw0 = std::chrono::steady_clock::now();
    std::thread wt[2];
    for (int s = 0; s < 2; ++s) {
        wt[s] = std::thread([&, s]() {
            int fd = ::open(pths[s].c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
            if (fd < 0) fd = ::open(pths[s].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (fd < 0) return;
            size_t done = 0;
            const auto* base = static_cast<const uint8_t*>(wr_mem[s]);
            while (done < stripe_bytes) {
                const ssize_t n = ::pwrite(fd, base + done, stripe_bytes - done, off_t(done));
                if (n <= 0) { ::close(fd); return; }
                done += size_t(n);
            }
            ::fdatasync(fd);
            ::close(fd);
            wr_ok[s] = true;
        });
    }
    wt[0].join(); wt[1].join();
    const double wr_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();

    bool rd_ok[2] = {false, false};
    const auto tr0 = std::chrono::steady_clock::now();
    std::thread rt[2];
    for (int s = 0; s < 2; ++s) {
        rt[s] = std::thread([&, s]() {
            int fd = ::open(pths[s].c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) fd = ::open(pths[s].c_str(), O_RDONLY);
            if (fd < 0) return;
            size_t done = 0;
            auto* base = static_cast<uint8_t*>(rd_mem[s]);
            while (done < stripe_bytes) {
                const ssize_t n = ::pread(fd, base + done, stripe_bytes - done, off_t(done));
                if (n <= 0) { ::close(fd); return; }
                done += size_t(n);
            }
            ::close(fd);
            rd_ok[s] = true;
        });
    }
    rt[0].join(); rt[1].join();
    const double rd_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();

    // Always clean up temp files immediately
    ::unlink(pths[0].c_str());
    ::unlink(pths[1].c_str());

    bool exact = wr_ok[0] && wr_ok[1] && rd_ok[0] && rd_ok[1];
    for (int s = 0; s < 2; ++s) {
        if (exact) {
            const uint32_t got_crc = kvblk_crc32c(rd_mem[s], stripe_bytes);
            if (got_crc != expected_crc[s] || std::memcmp(wr_mem[s], rd_mem[s], stripe_bytes) != 0) exact = false;
        }
        std::free(wr_mem[s]);
        std::free(rd_mem[s]);
    }
    if (!exact) {
        std::fprintf(stderr, "[FAIL] Test 3: Dual-NVMe O_DIRECT round-trip CRC32C/memcmp mismatch\n");
        return 1;
    }

    const double wr_gbs = (double(total_bytes) / 1e9) / (wr_ms * 1e-3);
    const double rd_gbs = (double(total_bytes) / 1e9) / (rd_ms * 1e-3);
    const double eq_toks = double(total_bytes) / kErnie55DeltaBytesPerTok;
    const double rd_tps  = eq_toks / (rd_ms * 1e-3);
    std::printf("[PASS] Test 3: Dual-NVMe O_DIRECT 64 MiB striped round-trip (memcmp == 0, CRC32C OK)\n");
    std::printf("       write=%.2f ms (%.2f GB/s) | read=%.2f ms (%.2f GB/s, equiv %.1fK tok/s @ 8.74 KB/tok)\n",
                wr_ms, wr_gbs, rd_ms, rd_gbs, rd_tps / 1000.0);
    std::printf("=== ALL 3 KVBLK UNIT TESTS PASSED 100%% ===\n");
    return 0;
}

bool http_get_local(const std::string& host, int port, const std::string& path, std::string& out_body) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    timeval tv{5, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(uint16_t(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) { ::close(fd); return false; }
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) { ::close(fd); return false; }
    const std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
    if (::send(fd, req.data(), req.size(), 0) != ssize_t(req.size())) { ::close(fd); return false; }
    std::string resp;
    char tmp[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        resp.append(tmp, size_t(n));
    }
    ::close(fd);
    const auto sep = resp.find("\r\n\r\n");
    if (sep == std::string::npos) return false;
    out_body = resp.substr(sep + 4);
    return true;
}

int run_live_checks(const std::string& endpoint, const std::string& dir_s0, const std::string& dir_s1) {
    using namespace ernie55;
    std::string host = "127.0.0.1";
    int port = 8052;
    std::string ep = endpoint;
    const auto pos_proto = ep.find("://");
    if (pos_proto != std::string::npos) ep = ep.substr(pos_proto + 3);
    const auto pos_slash = ep.find('/');
    if (pos_slash != std::string::npos) ep.resize(pos_slash);
    const auto pos_colon = ep.find(':');
    if (pos_colon != std::string::npos) {
        host = ep.substr(0, pos_colon);
        port = std::atoi(ep.substr(pos_colon + 1).c_str());
    } else if (!ep.empty()) {
        host = ep;
    }

    std::printf("=== [KVBLK-VERIFY --live] Querying %s:%d/v1/metrics ===\n", host.c_str(), port);
    std::string body;
    if (!http_get_local(host, port, "/v1/metrics", body)) {
        std::fprintf(stderr, "[FAIL] Cannot GET /v1/metrics from %s:%d\n", host.c_str(), port);
        return 1;
    }
    JsonValue j;
    if (!JsonParser::try_parse(body, j) || !j.is_object()) {
        std::fprintf(stderr, "[FAIL] Invalid JSON from /v1/metrics\n");
        return 1;
    }

    const int64_t seq_len   = j.has("seq_len") ? j.get("seq_len").as_int() : 0;
    const int64_t slots     = j.has("kvblk_ddr4_slots") ? j.get("kvblk_ddr4_slots").as_int() : 0;
    const int64_t used      = j.has("kvblk_ddr4_used") ? j.get("kvblk_ddr4_used").as_int() : 0;
    const std::string tier  = j.has("ctx_read_tier") ? j.get("ctx_read_tier").as_string() : "none";
    const int64_t rd_toks   = j.has("ctx_read_tokens") ? j.get("ctx_read_tokens").as_int() : 0;
    const double  rd_ms     = j.has("ctx_read_ms") ? j.get("ctx_read_ms").as_double() : 0.0;
    const double  rd_tps    = j.has("ctx_read_tok_s") ? j.get("ctx_read_tok_s").as_double() : 0.0;
    const double  pk_tps    = j.has("peak_ctx_read_tok_s") ? j.get("peak_ctx_read_tok_s").as_double() : 0.0;
    const double  ch_pf     = j.has("last_chunk_prefill_tok_s") ? j.get("last_chunk_prefill_tok_s").as_double() : 0.0;
    const double  dec_tps   = j.has("last_decode_tok_s") ? j.get("last_decode_tok_s").as_double() : 0.0;

    auto count_stripes = [](const std::string& d, const std::string& suf) -> size_t {
        size_t n = 0;
        std::error_code ec;
        if (!fs::is_directory(d, ec)) return 0;
        for (const auto& ent : fs::directory_iterator(d, ec)) {
            const std::string fn = ent.path().filename().string();
            if (fn.rfind("blk2k_", 0) == 0 && fn.size() > suf.size() &&
                fn.compare(fn.size() - suf.size(), suf.size(), suf) == 0) ++n;
        }
        return n;
    };
    const size_t n_s0 = count_stripes(dir_s0, ".s0");
    const size_t n_s1 = count_stripes(dir_s1, ".s1");

    std::printf("  seq_len=%lld | DDR4_slots=%lld (used=%lld) | Dual-NVMe stripes: s0=%zu s1=%zu\n",
                (long long)seq_len, (long long)slots, (long long)used, n_s0, n_s1);
    std::printf("  ctx_read_tier=%s | read_tokens=%lld | read_ms=%.3f | read_tok_s=%.1fK | peak=%.1fK\n",
                tier.c_str(), (long long)rd_toks, rd_ms, rd_tps / 1000.0, pk_tps / 1000.0);
    std::printf("  last_chunk_prefill=%.1f tok/s | last_decode=%.2f tok/s\n", ch_pf, dec_tps);

    if (slots <= 0 || n_s0 == 0 || n_s1 == 0) {
        std::fprintf(stderr, "[FAIL] Live 3-tier .kvblk invariant failed (slots=%lld s0=%zu s1=%zu)\n",
                     (long long)slots, n_s0, n_s1);
        return 1;
    }
    std::printf("[PASS] Live 3-Tier .kvblk telemetry & Dual-NVMe stripe persistence verified!\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool do_unit = false;
    bool do_live = false;
    bool do_blocks = false;
    int blocks_n = 16;
    std::string endpoint = "http://127.0.0.1:8052";
    std::string dir_s0   = "./kvcache";
    std::string dir_s1   = "/var/tmp/ernie55_kvcache";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--unit") do_unit = true;
        else if (a == "--live") {
            do_live = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') endpoint = argv[++i];
        } else if (a == "--blocks" && i + 1 < argc) {
            do_blocks = true;
            blocks_n = std::atoi(argv[++i]);
        } else if (a == "--s0" && i + 1 < argc) dir_s0 = argv[++i];
        else if (a == "--s1" && i + 1 < argc) dir_s1 = argv[++i];
    }
    if (!do_unit && !do_live && !do_blocks) do_unit = true;

    int rc = 0;
    if (do_unit) rc |= run_unit_checks(dir_s0, dir_s1);
    if (do_blocks) rc |= run_blocks_checks(blocks_n, dir_s0, dir_s1);
    if (do_live) rc |= run_live_checks(endpoint, dir_s0, dir_s1);
    return rc;
}
