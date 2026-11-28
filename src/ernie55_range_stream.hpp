// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 Zero-Disk-Bloat HTTP/2 206 Byte-Range Coalesced Safetensors Streamer for ERNIE-5.5
// Builds exact byte-offset manifests straight from Hugging Face model.safetensors.index.json + shard headers
// (zero Python!) and seamlessly handles BOTH BF16 direct shards (baidu/ERNIE-5.5-Flash-Base-BF16) and
// Block-128 FP8 shards (baidu/ERNIE-5.5-Flash-Base or Nov 25 post-trained checkpoint with weight_scale_inv).

#include "ernie55_hlut4_pack.hpp"
#include "ernie55_json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ernie55 {

struct RemoteTensorSpec {
    std::string name;
    std::string shard_file;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t file_offset = 0;
    uint64_t nbytes = 0;
};

struct RemoteRangeIndex {
    std::string repo = "baidu/ERNIE-5.5-Flash-Base-BF16";
    std::string revision = "main";
    std::unordered_map<std::string, RemoteTensorSpec> by_name;
    std::vector<std::string> all_names;
    std::vector<std::string> shard_files;
    bool is_fp8 = false;

    std::string file_url(const std::string& relpath) const {
        return "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + relpath;
    }

    static bool curl_fetch_range(const std::string& url, uint64_t start, uint64_t end_excl,
                                 uint8_t* dst, int max_retries = 5) {
        const uint64_t expect = end_excl - start;
        if (expect == 0ULL) return true;
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            char cmd[1024];
            std::snprintf(cmd, sizeof(cmd),
                "curl -fsSL --http2 --retry 2 --retry-delay 2 --connect-timeout 20 --max-time 600 "
                "-r %llu-%llu '%s'",
                (unsigned long long)start, (unsigned long long)(end_excl - 1ULL), url.c_str());
            FILE* fp = popen(cmd, "r");
            if (!fp) continue;
            size_t got = 0;
            while (got < expect) {
                size_t n = std::fread(dst + got, 1, size_t(expect - got), fp);
                if (n == 0) break;
                got += n;
            }
            const int rc = pclose(fp);
            if (rc == 0 && got == expect) return true;
            std::this_thread::sleep_for(std::chrono::seconds(2 + attempt * 2));
        }
        return false;
    }

    static bool curl_download_file(const std::string& url, const std::string& out_path) {
        char cmd[1024];
        std::snprintf(cmd, sizeof(cmd),
            "curl -fsSL --http2 --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 300 '%s' -o '%s'",
            url.c_str(), out_path.c_str());
        return std::system(cmd) == 0;
    }

    // Build exact tensor byte-offset TSV directly from model.safetensors.index.json + parallel 206 shard header reads
    void build_from_hf_index(const std::string& index_json_path, const std::string& cache_tsv_path, int workers = 16) {
        std::ifstream ifs(index_json_path);
        if (!ifs) throw std::runtime_error("Cannot open index JSON: " + index_json_path);
        std::ostringstream ss;
        ss << ifs.rdbuf();
        JsonValue root = JsonParser::parse(ss.str());
        const auto& wm = root.get("weight_map");
        if (!wm.is_object()) throw std::runtime_error("Missing weight_map in " + index_json_path);

        std::set<std::string> shards_set;
        for (const auto& kv : wm.obj_val) shards_set.insert(kv.second.as_string());
        shard_files.assign(shards_set.begin(), shards_set.end());

        std::atomic<size_t> next_idx{0};
        std::atomic<int> fails{0};
        std::mutex mu;
        std::vector<RemoteTensorSpec> collected;
        collected.reserve(wm.obj_val.size());

        auto worker = [&]() {
            std::vector<uint8_t> hdr_buf;
            while (true) {
                const size_t i = next_idx.fetch_add(1);
                if (i >= shard_files.size()) break;
                const std::string& sf = shard_files[i];
                const std::string url = file_url(sf);

                uint64_t hlen = 0;
                uint8_t b8[8];
                if (!curl_fetch_range(url, 0, 8, b8)) { ++fails; continue; }
                std::memcpy(&hlen, b8, 8);
                if (hlen == 0 || hlen > 64ULL * 1024ULL * 1024ULL) { ++fails; continue; }

                hdr_buf.resize(size_t(hlen));
                if (!curl_fetch_range(url, 8, 8 + hlen, hdr_buf.data())) { ++fails; continue; }

                std::string hdr_str(reinterpret_cast<const char*>(hdr_buf.data()), size_t(hlen));
                JsonValue hjson = JsonParser::parse(hdr_str);
                std::vector<RemoteTensorSpec> local;
                for (const auto& kv : hjson.obj_val) {
                    if (kv.first == "__metadata__") continue;
                    const auto& info = kv.second;
                    RemoteTensorSpec sp;
                    sp.name = kv.first;
                    sp.shard_file = sf;
                    sp.dtype = info.get("dtype").as_string();
                    for (const auto& d : info.get("shape").arr_val) sp.shape.push_back(d.as_int());
                    const auto& offs = info.get("data_offsets").arr_val;
                    const uint64_t b0 = uint64_t(offs.at(0).as_int());
                    const uint64_t b1 = uint64_t(offs.at(1).as_int());
                    sp.file_offset = 8ULL + hlen + b0;
                    sp.nbytes = b1 - b0;
                    local.push_back(std::move(sp));
                }
                std::lock_guard<std::mutex> lk(mu);
                for (auto& sp : local) collected.push_back(std::move(sp));
            }
        };

        const int nw = std::min<int>(workers, std::max<int>(1, int(shard_files.size())));
        std::vector<std::thread> ths;
        for (int t = 0; t < nw; ++t) ths.emplace_back(worker);
        for (auto& t : ths) t.join();
        if (fails.load() > 0) throw std::runtime_error("Failed fetching shard headers for " + repo);

        std::sort(collected.begin(), collected.end(), [](const RemoteTensorSpec& a, const RemoteTensorSpec& b) {
            return a.name < b.name;
        });

        by_name.clear();
        all_names.clear();
        is_fp8 = false;
        std::ofstream ofs(cache_tsv_path);
        ofs << "name\tshard_file\tdtype\tshape\tfile_offset\tnbytes\n";
        for (auto& sp : collected) {
            if (sp.dtype == "F8_E4M3") is_fp8 = true;
            ofs << sp.name << '\t' << sp.shard_file << '\t' << sp.dtype << '\t';
            for (size_t k = 0; k < sp.shape.size(); ++k) {
                if (k) ofs << 'x';
                ofs << sp.shape[k];
            }
            ofs << '\t' << sp.file_offset << '\t' << sp.nbytes << '\n';
            all_names.push_back(sp.name);
            by_name.emplace(sp.name, std::move(sp));
        }
    }

    void load_tsv(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Cannot open tensor range manifest: " + path);
        std::string line;
        std::getline(ifs, line);
        by_name.clear();
        all_names.clear();
        is_fp8 = false;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            RemoteTensorSpec sp;
            size_t p0 = 0, p1 = line.find('\t');
            sp.name = line.substr(p0, p1 - p0);
            p0 = p1 + 1; p1 = line.find('\t', p0);
            sp.shard_file = line.substr(p0, p1 - p0);
            p0 = p1 + 1; p1 = line.find('\t', p0);
            sp.dtype = line.substr(p0, p1 - p0);
            if (sp.dtype == "F8_E4M3") is_fp8 = true;
            p0 = p1 + 1; p1 = line.find('\t', p0);
            std::string shape_s = line.substr(p0, p1 - p0);
            {
                size_t a = 0;
                while (a < shape_s.size()) {
                    size_t x = shape_s.find('x', a);
                    if (x == std::string::npos) x = shape_s.size();
                    sp.shape.push_back(std::stoll(shape_s.substr(a, x - a)));
                    a = x + 1;
                }
            }
            p0 = p1 + 1; p1 = line.find('\t', p0);
            sp.file_offset = std::stoull(line.substr(p0, p1 - p0));
            p0 = p1 + 1;
            sp.nbytes = std::stoull(line.substr(p0));
            all_names.push_back(sp.name);
            by_name.emplace(sp.name, std::move(sp));
        }
    }

    bool has(const std::string& name) const noexcept { return by_name.find(name) != by_name.end(); }
};

struct StreamedTensorBundle {
    std::vector<std::vector<uint8_t>> chunks;
    std::unordered_map<std::string, const uint8_t*> ptrs;
    std::unordered_map<std::string, RemoteTensorSpec> specs;
    uint64_t total_tensor_bytes = 0;
    uint64_t total_wire_bytes = 0;
    double elapsed_sec = 0.0;

    bool has(const std::string& name) const noexcept { return ptrs.find(name) != ptrs.end(); }
    const uint8_t* data(const std::string& name) const {
        auto it = ptrs.find(name);
        if (it == ptrs.end()) throw std::runtime_error("Missing streamed tensor: " + name);
        return it->second;
    }
    const RemoteTensorSpec& spec(const std::string& name) const {
        auto it = specs.find(name);
        if (it == specs.end()) throw std::runtime_error("Missing streamed tensor spec: " + name);
        return it->second;
    }

    // Unified BF16/F32/Block-128-FP8 -> FP32 extractor (handles both Base-BF16 and FP8 Nov 25 checkpoints!)
    std::vector<float> to_f32(const std::string& name) const {
        const auto& sp = spec(name);
        const uint8_t* p = data(name);
        size_t numel = 1;
        for (int64_t d : sp.shape) numel *= size_t(d);
        std::vector<float> out(numel);
        if (sp.dtype == "F32") {
            std::memcpy(out.data(), p, numel * sizeof(float));
        } else if (sp.dtype == "BF16") {
            const uint16_t* bp = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < numel; ++i) out[i] = bf16_to_f32(bp[i]);
        } else if (sp.dtype == "F16") {
            const uint16_t* hp = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < numel; ++i) out[i] = fp16_to_f32(hp[i]);
        } else if (sp.dtype == "F8_E4M3") {
            const std::string sname = name + "_scale_inv";
            if (!has(sname)) throw std::runtime_error("FP8 tensor missing scale_inv: " + sname);
            const auto& ssp = spec(sname);
            const uint8_t* sp_bytes = data(sname);
            const int64_t rows = sp.shape.at(0);
            const int64_t cols = sp.shape.at(1);
            const int64_t s_rows = ssp.shape.at(0);
            const int64_t s_cols = ssp.shape.at(1);
            const int64_t blk_r = std::max<int64_t>(1, (rows + s_rows - 1) / std::max<int64_t>(1, s_rows));
            const int64_t blk_c = std::max<int64_t>(1, (cols + s_cols - 1) / std::max<int64_t>(1, s_cols));
            const bool u8_scale = (ssp.dtype == "U8" || ssp.dtype == "I8");
            const float* f32_scales = u8_scale ? nullptr : reinterpret_cast<const float*>(sp_bytes);
            for (int64_t r = 0; r < rows; ++r) {
                const int64_t sr = std::min(s_rows - 1, r / blk_r);
                for (int64_t c = 0; c < cols; ++c) {
                    const int64_t sc = std::min(s_cols - 1, c / blk_c);
                    const float s = u8_scale ? ue8m0_to_f32(sp_bytes[size_t(sr * s_cols + sc)])
                                             : f32_scales[size_t(sr * s_cols + sc)];
                    out[size_t(r * cols + c)] = fp8_e4m3fn_to_f32(p[size_t(r * cols + c)]) * s;
                }
            }
        } else {
            throw std::runtime_error("Unsupported dtype " + sp.dtype + " for " + name);
        }
        return out;
    }

    std::vector<uint16_t> to_bf16(const std::string& name) const {
        const auto& sp = spec(name);
        if (sp.dtype == "BF16") {
            size_t numel = 1;
            for (int64_t d : sp.shape) numel *= size_t(d);
            std::vector<uint16_t> out(numel);
            std::memcpy(out.data(), data(name), numel * 2u);
            return out;
        }
        const auto f = to_f32(name);
        std::vector<uint16_t> out(f.size());
        for (size_t i = 0; i < f.size(); ++i) out[i] = f32_to_bf16_rne(f[i]);
        return out;
    }
};

inline StreamedTensorBundle stream_fetch_tensors(
    const RemoteRangeIndex& idx,
    const std::vector<std::string>& req_names,
    int num_workers = 16,
    uint64_t max_gap_coalesce = 1024ULL * 1024ULL,
    uint64_t max_chunk_bytes  = 256ULL * 1024ULL * 1024ULL)
{
    const auto t0 = std::chrono::steady_clock::now();
    StreamedTensorBundle bundle;

    // Automatically include paired *_scale_inv for any FP8 tensors!
    std::vector<std::string> names;
    std::set<std::string> seen;
    for (const auto& nm : req_names) {
        if (seen.insert(nm).second) names.push_back(nm);
        auto it = idx.by_name.find(nm);
        if (it != idx.by_name.end() && it->second.dtype == "F8_E4M3") {
            const std::string sn = nm + "_scale_inv";
            if (idx.has(sn) && seen.insert(sn).second) names.push_back(sn);
        }
    }

    std::map<std::string, std::vector<const RemoteTensorSpec*>> by_shard;
    for (const auto& nm : names) {
        auto it = idx.by_name.find(nm);
        if (it == idx.by_name.end()) throw std::runtime_error("Tensor not in index: " + nm);
        by_shard[it->second.shard_file].push_back(&it->second);
        bundle.specs.emplace(nm, it->second);
        bundle.total_tensor_bytes += it->second.nbytes;
    }

    struct RangeJob {
        std::string shard_file;
        uint64_t start = 0;
        uint64_t end = 0;
        std::vector<const RemoteTensorSpec*> items;
    };
    std::vector<RangeJob> jobs;
    for (auto& kv : by_shard) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(), [](const RemoteTensorSpec* a, const RemoteTensorSpec* b) {
            return a->file_offset < b->file_offset;
        });
        for (const auto* sp : vec) {
            const uint64_t s = sp->file_offset;
            const uint64_t e = s + sp->nbytes;
            if (!jobs.empty() && jobs.back().shard_file == kv.first &&
                s >= jobs.back().end && (s - jobs.back().end) <= max_gap_coalesce &&
                (e - jobs.back().start) <= max_chunk_bytes)
            {
                jobs.back().end = std::max(jobs.back().end, e);
                jobs.back().items.push_back(sp);
            } else {
                RangeJob j;
                j.shard_file = kv.first;
                j.start = s;
                j.end = e;
                j.items.push_back(sp);
                jobs.push_back(std::move(j));
            }
        }
    }

    std::sort(jobs.begin(), jobs.end(), [](const RangeJob& a, const RangeJob& b) {
        return (a.end - a.start) > (b.end - b.start);
    });

    bundle.chunks.resize(jobs.size());
    std::atomic<size_t> next_job{0};
    std::atomic<int> failures{0};
    std::atomic<uint64_t> wire_bytes{0};

    auto worker = [&]() {
        while (true) {
            const size_t ji = next_job.fetch_add(1);
            if (ji >= jobs.size()) break;
            const auto& j = jobs[ji];
            const uint64_t sz = j.end - j.start;
            bundle.chunks[ji].resize(size_t(sz));
            const std::string url = idx.file_url(j.shard_file);
            uint64_t cur = j.start, off = 0;
            bool ok = true;
            while (cur < j.end) {
                const uint64_t step = std::min<uint64_t>(j.end - cur, 256ULL * 1024ULL * 1024ULL);
                if (!RemoteRangeIndex::curl_fetch_range(url, cur, cur + step, bundle.chunks[ji].data() + off)) {
                    ok = false;
                    break;
                }
                cur += step;
                off += step;
            }
            if (!ok) { ++failures; continue; }
            wire_bytes.fetch_add(sz);
        }
    };

    const int nw = std::min<int>(num_workers, std::max<int>(1, int(jobs.size())));
    std::vector<std::thread> ths;
    for (int t = 0; t < nw; ++t) ths.emplace_back(worker);
    for (auto& t : ths) t.join();
    if (failures.load() > 0) throw std::runtime_error("HTTP/2 range streaming failed!");

    for (size_t ji = 0; ji < jobs.size(); ++ji) {
        const auto& j = jobs[ji];
        for (const auto* sp : j.items) {
            bundle.ptrs.emplace(sp->name, bundle.chunks[ji].data() + (sp->file_offset - j.start));
        }
    }
    bundle.total_wire_bytes = wire_bytes.load();
    bundle.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return bundle;
}

} // namespace ernie55
