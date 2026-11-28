// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 PCIe Gen 4.0 x16 DMA Stress Gate for 5x AMD Instinct MI50 (gfx906)
// Executes 500-iteration 16 MiB solo + ALL-5 concurrent pinned H2D/D2H transfers,
// checks per-card PCIe link speed/width, PCIe replay counts, and global AER_TOTAL == 0.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string read_trim(const std::string& p) {
    std::ifstream ifs(p);
    if (!ifs) return "";
    std::string s;
    std::getline(ifs, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

std::string to_lower_str(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    }
    return s;
}

int64_t read_aer_total() {
    int64_t total = 0;
    std::error_code ec;
    for (const auto& ent : std::filesystem::directory_iterator("/sys/bus/pci/devices", ec)) {
        if (ec) break;
        const std::string base = ent.path().string();
        for (const char* fname : {"/aer_dev_fatal", "/aer_dev_nonfatal"}) {
            std::ifstream ifs(base + fname);
            if (!ifs) continue;
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.rfind("TOTAL_ERR_FATAL", 0) == 0 || line.rfind("TOTAL_ERR_NONFATAL", 0) == 0) {
                    std::istringstream iss(line);
                    std::string k;
                    int64_t v = 0;
                    if (iss >> k >> v) total += v;
                }
            }
        }
    }
    return total;
}

int64_t read_replay_count(const std::string& bdf) {
    const std::string s = read_trim("/sys/bus/pci/devices/" + bdf + "/pcie_replay_count");
    if (s.empty()) return -1;
    return std::strtoll(s.c_str(), nullptr, 10);
}

void check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "[PCIE-DMA-GATE] HIP error at %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

struct DevCtx {
    int dev_id = 0;
    std::string bdf;
    std::string speed;
    std::string width;
    int64_t replay_before = -1;
    int64_t replay_after = -1;
    void* h_buf = nullptr;
    void* d_buf = nullptr;
    hipStream_t stream = nullptr;
    double solo_h2d_gbps = 0.0;
    double solo_d2h_gbps = 0.0;
    double conc_h2d_gbps = 0.0;
    double conc_d2h_gbps = 0.0;
    bool integrity_ok = true;
};

}  // namespace

int main(int argc, char** argv) {
    int iters = 500;
    int warmup = 20;
    size_t bytes = 16ull * 1024ull * 1024ull;
    std::string target_bdf = "0000:08:00.0";
    double min_target_gbps = 26.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* fl) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", fl);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--iters") iters = std::max(1, std::atoi(need("--iters")));
        else if (a == "--warmup") warmup = std::max(0, std::atoi(need("--warmup")));
        else if (a == "--mib") bytes = size_t(std::max(1, std::atoi(need("--mib")))) * 1024ull * 1024ull;
        else if (a == "--target-bdf") target_bdf = to_lower_str(need("--target-bdf"));
        else if (a == "--min-gbps") min_target_gbps = std::atof(need("--min-gbps"));
    }

    int ndev = 0;
    check_hip(hipGetDeviceCount(&ndev), "hipGetDeviceCount");
    if (ndev <= 0) {
        std::fprintf(stderr, "[PCIE-DMA-GATE] No HIP devices found.\n");
        return 2;
    }

    const int64_t aer_before = read_aer_total();
    std::vector<DevCtx> devs(static_cast<size_t>(ndev));

    for (int i = 0; i < ndev; ++i) {
        auto& c = devs[size_t(i)];
        c.dev_id = i;
        check_hip(hipSetDevice(i), "hipSetDevice");
        char bus[64]{};
        check_hip(hipDeviceGetPCIBusId(bus, sizeof(bus), i), "hipDeviceGetPCIBusId");
        c.bdf = to_lower_str(bus);
        c.speed = read_trim("/sys/bus/pci/devices/" + c.bdf + "/current_link_speed");
        c.width = read_trim("/sys/bus/pci/devices/" + c.bdf + "/current_link_width");
        c.replay_before = read_replay_count(c.bdf);
        check_hip(hipHostMalloc(&c.h_buf, bytes, hipHostMallocDefault), "hipHostMalloc");
        check_hip(hipMalloc(&c.d_buf, bytes), "hipMalloc");
        check_hip(hipStreamCreateWithFlags(&c.stream, hipStreamNonBlocking), "hipStreamCreate");
        auto* p = static_cast<uint8_t*>(c.h_buf);
        const uint8_t pat = uint8_t(0x5A ^ (i * 17));
        std::memset(p, pat, bytes);
        p[0] = uint8_t(0xA5 ^ i);
        p[bytes - 1] = uint8_t(0x3C ^ i);
    }

    std::printf("=== ERNIE-5.5 PCIe Gen 4.0 DMA Stress Gate (ndev=%d, size=%zu MiB, iters=%d, warmup=%d) ===\n",
                ndev, static_cast<size_t>(bytes / (1024ull * 1024ull)), iters, warmup);
    std::printf("AER_TOTAL_BEFORE=%lld\n", (long long)aer_before);

    // 1. Solo pinned H2D + D2H per GPU
    for (int i = 0; i < ndev; ++i) {
        auto& c = devs[size_t(i)];
        check_hip(hipSetDevice(c.dev_id), "solo hipSetDevice");

        for (int w = 0; w < warmup; ++w) {
            check_hip(hipMemcpyAsync(c.d_buf, c.h_buf, bytes, hipMemcpyHostToDevice, c.stream), "warm H2D");
        }
        check_hip(hipStreamSynchronize(c.stream), "warm H2D sync");

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            check_hip(hipMemcpyAsync(c.d_buf, c.h_buf, bytes, hipMemcpyHostToDevice, c.stream), "solo H2D");
        }
        check_hip(hipStreamSynchronize(c.stream), "solo H2D sync");
        const auto t1 = std::chrono::steady_clock::now();
        const double h2d_s = std::chrono::duration<double>(t1 - t0).count();
        c.solo_h2d_gbps = (double(bytes) * double(iters)) / (h2d_s * 1e9);

        // Zero host sentinels to verify D2H round-trip integrity
        auto* p = static_cast<uint8_t*>(c.h_buf);
        const uint8_t exp0 = uint8_t(0xA5 ^ i);
        const uint8_t exp1 = uint8_t(0x3C ^ i);
        p[0] = 0;
        p[bytes - 1] = 0;

        for (int w = 0; w < warmup; ++w) {
            check_hip(hipMemcpyAsync(c.h_buf, c.d_buf, bytes, hipMemcpyDeviceToHost, c.stream), "warm D2H");
        }
        check_hip(hipStreamSynchronize(c.stream), "warm D2H sync");

        const auto t2 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            check_hip(hipMemcpyAsync(c.h_buf, c.d_buf, bytes, hipMemcpyDeviceToHost, c.stream), "solo D2H");
        }
        check_hip(hipStreamSynchronize(c.stream), "solo D2H sync");
        const auto t3 = std::chrono::steady_clock::now();
        const double d2h_s = std::chrono::duration<double>(t3 - t2).count();
        c.solo_d2h_gbps = (double(bytes) * double(iters)) / (d2h_s * 1e9);
        if (p[0] != exp0 || p[bytes - 1] != exp1) c.integrity_ok = false;

        std::printf("  [SOLO] dev=%d bdf=%s link=%s x%s  H2D=%.2f GB/s  D2H=%.2f GB/s  integrity=%s\n",
                    c.dev_id, c.bdf.c_str(), c.speed.c_str(), c.width.c_str(),
                    c.solo_h2d_gbps, c.solo_d2h_gbps, c.integrity_ok ? "OK" : "FAIL");
    }

    // 2. ALL-N Concurrent pinned H2D and D2H
    auto run_concurrent = [&](bool is_h2d) -> double {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> thrs;
        thrs.reserve(size_t(ndev));
        std::chrono::steady_clock::time_point wall_t0{};
        for (int i = 0; i < ndev; ++i) {
            thrs.emplace_back([&, i]() {
                auto& c = devs[size_t(i)];
                check_hip(hipSetDevice(c.dev_id), "conc hipSetDevice");
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (!go.load(std::memory_order_acquire)) {}
                const auto lt0 = std::chrono::steady_clock::now();
                for (int it = 0; it < iters; ++it) {
                    if (is_h2d) {
                        check_hip(hipMemcpyAsync(c.d_buf, c.h_buf, bytes, hipMemcpyHostToDevice, c.stream), "conc H2D");
                    } else {
                        check_hip(hipMemcpyAsync(c.h_buf, c.d_buf, bytes, hipMemcpyDeviceToHost, c.stream), "conc D2H");
                    }
                }
                check_hip(hipStreamSynchronize(c.stream), "conc sync");
                const auto lt1 = std::chrono::steady_clock::now();
                const double sec = std::chrono::duration<double>(lt1 - lt0).count();
                const double gbps = (double(bytes) * double(iters)) / (sec * 1e9);
                if (is_h2d) c.conc_h2d_gbps = gbps;
                else c.conc_d2h_gbps = gbps;
            });
        }
        while (ready.load(std::memory_order_acquire) < ndev) {
            std::this_thread::yield();
        }
        wall_t0 = std::chrono::steady_clock::now();
        go.store(true, std::memory_order_release);
        for (auto& t : thrs) t.join();
        const auto wall_t1 = std::chrono::steady_clock::now();
        const double wall_s = std::chrono::duration<double>(wall_t1 - wall_t0).count();
        return (double(bytes) * double(iters) * double(ndev)) / (wall_s * 1e9);
    };

    const double agg_h2d = run_concurrent(true);
    const double agg_d2h = run_concurrent(false);

    int64_t replay_delta_sum = 0;
    bool target_seen = false;
    bool target_pass = false;
    bool all_integrity = true;

    std::printf("  [ALL-%d CONCURRENT] aggregate_H2D=%.2f GB/s  aggregate_D2H=%.2f GB/s\n", ndev, agg_h2d, agg_d2h);
    for (int i = 0; i < ndev; ++i) {
        auto& c = devs[size_t(i)];
        c.replay_after = read_replay_count(c.bdf);
        const int64_t rdelta = (c.replay_before >= 0 && c.replay_after >= 0) ? (c.replay_after - c.replay_before) : 0;
        replay_delta_sum += rdelta;
        if (!c.integrity_ok) all_integrity = false;
        if (c.bdf == target_bdf) {
            target_seen = true;
            target_pass = (c.solo_h2d_gbps >= min_target_gbps && c.solo_d2h_gbps >= min_target_gbps);
        }
        std::printf("    dev=%d bdf=%s conc_H2D=%.2f GB/s  conc_D2H=%.2f GB/s  replay=%lld->%lld (delta=%lld)\n",
                    c.dev_id, c.bdf.c_str(), c.conc_h2d_gbps, c.conc_d2h_gbps,
                    (long long)c.replay_before, (long long)c.replay_after, (long long)rdelta);

        check_hip(hipSetDevice(c.dev_id), "cleanup hipSetDevice");
        (void)hipStreamDestroy(c.stream);
        (void)hipFree(c.d_buf);
        (void)hipHostFree(c.h_buf);
    }

    const int64_t aer_after = read_aer_total();
    std::printf("AER_TOTAL_AFTER=%lld (delta=%lld)  REPLAY_DELTA_SUM=%lld\n",
                (long long)aer_after, (long long)(aer_after - aer_before), (long long)replay_delta_sum);

    const bool ok = all_integrity && (aer_after == 0) && (replay_delta_sum == 0) && (!target_seen || target_pass);
    std::printf("=== PCIE DMA GATE RESULT: %s (target=%s threshold=%.2f GB/s seen=%d pass=%d) ===\n",
                ok ? "PASS" : "FAIL", target_bdf.c_str(), min_target_gbps, int(target_seen), int(target_pass));
    return ok ? 0 : 1;
}
