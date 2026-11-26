// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 Streaming Safetensors Writer for ERNIE-5.5-Flash Packed Spine Shards
// Zero Python anywhere. Writes standard HuggingFace Safetensors v1 binary files
// with 64-byte aligned tensor payloads for zero-copy mmap loading.

#include "ernie55_json.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace ernie55 {

struct SafetensorsOutEntry {
    std::string name;
    std::string dtype;              // "BF16", "F32", "F16", "U8", "I32"
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

class Ernie55SafetensorsWriter {
public:
    void add_tensor(std::string name, std::string dtype, std::vector<int64_t> shape,
                    const void* bytes, size_t nbytes) {
        SafetensorsOutEntry e;
        e.name = std::move(name);
        e.dtype = std::move(dtype);
        e.shape = std::move(shape);
        e.data.resize(nbytes);
        if (nbytes > 0) std::memcpy(e.data.data(), bytes, nbytes);
        entries_.push_back(std::move(e));
    }

    void add_f32(const std::string& name, const std::vector<int64_t>& shape, const std::vector<float>& v) {
        add_tensor(name, "F32", shape, v.data(), v.size() * sizeof(float));
    }

    void add_bf16(const std::string& name, const std::vector<int64_t>& shape, const std::vector<uint16_t>& v) {
        add_tensor(name, "BF16", shape, v.data(), v.size() * sizeof(uint16_t));
    }

    size_t count() const noexcept { return entries_.size(); }

    void write_atomic(const std::string& out_path) const {
        JsonValue hdr = JsonValue::make_object();
        uint64_t offset = 0;
        for (const auto& e : entries_) {
            JsonValue meta = JsonValue::make_object();
            meta.set("dtype", JsonValue::make_string(e.dtype));
            JsonValue sh = JsonValue::make_array();
            for (int64_t d : e.shape) sh.arr_val.push_back(JsonValue::make_int(d));
            meta.set("shape", std::move(sh));
            JsonValue offs = JsonValue::make_array();
            const uint64_t start = offset;
            const uint64_t end = start + uint64_t(e.data.size());
            offs.arr_val.push_back(JsonValue::make_int(int64_t(start)));
            offs.arr_val.push_back(JsonValue::make_int(int64_t(end)));
            meta.set("data_offsets", std::move(offs));
            hdr.set(e.name, std::move(meta));
            offset = (end + 63ULL) & ~63ULL;
        }
        std::string hdr_json = hdr.dumps_compact();
        while ((8ULL + hdr_json.size()) % 64ULL != 0ULL) hdr_json.push_back(' ');
        const uint64_t hdr_len = uint64_t(hdr_json.size());

        const std::string tmp_path = out_path + ".tmp";
        const int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("Cannot create safetensors file: " + tmp_path);

        auto write_all = [&](const void* p, size_t n) {
            const uint8_t* bp = static_cast<const uint8_t*>(p);
            size_t done = 0;
            while (done < n) {
                const ssize_t w = ::write(fd, bp + done, n - done);
                if (w <= 0) {
                    ::close(fd);
                    ::unlink(tmp_path.c_str());
                    throw std::runtime_error("Short write to safetensors file: " + tmp_path);
                }
                done += size_t(w);
            }
        };

        write_all(&hdr_len, sizeof(hdr_len));
        write_all(hdr_json.data(), hdr_json.size());

        const uint8_t zeros[64]{};
        uint64_t cur = 0;
        for (const auto& e : entries_) {
            write_all(e.data.data(), e.data.size());
            cur += uint64_t(e.data.size());
            const uint64_t aligned = (cur + 63ULL) & ~63ULL;
            if (aligned > cur) {
                write_all(zeros, size_t(aligned - cur));
                cur = aligned;
            }
        }
        (void)::fdatasync(fd);
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
        if (::rename(tmp_path.c_str(), out_path.c_str()) != 0) {
            ::unlink(tmp_path.c_str());
            throw std::runtime_error("Atomic rename failed: " + out_path);
        }
    }

private:
    std::vector<SafetensorsOutEntry> entries_;
};

// Precompute VHA post-head mix matrix M[64, 64] = I + V @ U^T in FP32
// (U: [64, 16], V: [64, 16]) matching vLLM-ERNIE attention.py::fold_vha!
inline std::vector<float> fold_vha_matrix_64x64(const float* U_64x16, const float* V_64x16, int H = 64, int R = 16) {
    std::vector<float> M(size_t(H) * size_t(H), 0.0f);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            double acc = (i == j) ? 1.0 : 0.0;
            for (int r = 0; r < R; ++r) {
                acc += double(V_64x16[size_t(i) * size_t(R) + size_t(r)]) *
                       double(U_64x16[size_t(j) * size_t(R) + size_t(r)]);
            }
            M[size_t(i) * size_t(H) + size_t(j)] = float(acc);
        }
    }
    return M;
}

// Transpose [rows, cols] -> [cols, rows] in FP32 (for mHC mapping_proj [16384,24], hc_head_fn [16384,4], markov_w1 [R,V])
inline std::vector<float> transpose_f32(const float* src, int rows, int cols) {
    std::vector<float> dst(size_t(rows) * size_t(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[size_t(c) * size_t(rows) + size_t(r)] = src[size_t(r) * size_t(cols) + size_t(c)];
        }
    }
    return dst;
}

} // namespace ernie55
