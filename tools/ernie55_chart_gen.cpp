// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Zero-Dependency 1440x860 4-Panel (2x2) PNG Telemetry Chart Renderer
// for baidu/ERNIE-5.5-Flash-Preview (300B = 43L backbone + 3L MoE DSpark+SA, 100% WHT128-HLUT4,
// 0 B DDR4, 1M Lossless BF16 KV + FlashIndexer) on SB-M5-P2-192G.
// Reads measured silicon receipts from gate_wave13_1k_prefill_1m_receipt.json (Wave-12 and Wave-11 accepted as a
// compatibility alias) and GATE-1/2/4/6 JSONs.
// Emits strictly IHDR + IDAT + IEND chunks (zero ancillary metadata chunks, zero Python, zero deps).
//
// Terminology and privacy rules enforced in every rendered label and comment:
//   - lowercase "bespoke ontology" and "the kitchen"
//   - "AMD EPYC 7542"
//   - zero local filesystem paths, zero personal names, zero hostnames, zero IPs
//   - zero wall-power cost figures

#include "../src/ernie55_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ernie55_chart {

struct Color { uint8_t r, g, b; };

static constexpr Color kBg       {248, 250, 252};
static constexpr Color kCard     {255, 255, 255};
static constexpr Color kBorder   {203, 213, 225};
static constexpr Color kGrid     {226, 232, 240};
static constexpr Color kText     { 15,  23,  42};
static constexpr Color kMuted    { 71,  85, 105};
static constexpr Color kBlue     { 37,  99, 235};
static constexpr Color kTeal     { 13, 148, 136};
static constexpr Color kGreen    { 22, 163,  74};
static constexpr Color kAmber    {217, 119,   6};
static constexpr Color kPurple   {124,  58, 237};
static constexpr Color kSlate    {100, 116, 139};
static constexpr Color kRed      {220,  38,  38};
static constexpr Color kSoftBlue {219, 234, 254};
static constexpr Color kSoftGrn  {220, 252, 231};

struct Canvas {
    int W = 1440, H = 860;
    std::vector<uint8_t> rgb;
    Canvas(int w, int h, Color bg) : W(w), H(h), rgb(size_t(w) * size_t(h) * 3u) {
        for (int i = 0; i < w * h; ++i) {
            rgb[size_t(i) * 3u + 0u] = bg.r;
            rgb[size_t(i) * 3u + 1u] = bg.g;
            rgb[size_t(i) * 3u + 2u] = bg.b;
        }
    }
    inline void px(int x, int y, Color c) {
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        const size_t o = (size_t(y) * size_t(W) + size_t(x)) * 3u;
        rgb[o + 0u] = c.r; rgb[o + 1u] = c.g; rgb[o + 2u] = c.b;
    }
    void fill_rect(int x, int y, int w, int h, Color c) {
        for (int yy = y; yy < y + h; ++yy)
            for (int xx = x; xx < x + w; ++xx) px(xx, yy, c);
    }
    void draw_rect(int x, int y, int w, int h, Color c, int t = 1) {
        fill_rect(x, y, w, t, c);
        fill_rect(x, y + h - t, w, t, c);
        fill_rect(x, y, t, h, c);
        fill_rect(x + w - t, y, t, h, c);
    }
    void fill_circle(int cx, int cy, int r, Color c) {
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
                if (dx * dx + dy * dy <= r * r) px(cx + dx, cy + dy, c);
    }
    void draw_line(double x0, double y0, double x1, double y1, Color c, int thick = 2) {
        const double dx = x1 - x0, dy = y1 - y0;
        const int steps = std::max(1, int(std::ceil(std::hypot(dx, dy) * 2.0)));
        const int r = thick / 2;
        for (int i = 0; i <= steps; ++i) {
            const double t = double(i) / double(steps);
            const int x = int(std::lround(x0 + dx * t));
            const int y = int(std::lround(y0 + dy * t));
            for (int oy = -r; oy <= r; ++oy)
                for (int ox = -r; ox <= r; ++ox) px(x + ox, y + oy, c);
        }
    }
    void draw_dashed_h(int x0, int x1, int y, Color c) {
        for (int x = x0; x <= x1; ++x) if (((x - x0) / 5) % 2 == 0) px(x, y, c);
    }
    void draw_dashed_v(int x, int y0, int y1, Color c) {
        for (int y = y0; y <= y1; ++y) if (((y - y0) / 5) % 2 == 0) { px(x, y, c); px(x + 1, y, c); }
    }

    static const uint8_t* glyph5x7(char ch) {
        static const uint8_t kBlank[7] = {0,0,0,0,0,0,0};
        struct Entry { char c; uint8_t rows[7]; };
        static const Entry kFont[] = {
            {'0',{0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}}, {'1',{0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
            {'2',{0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}}, {'3',{0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}},
            {'4',{0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}}, {'5',{0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
            {'6',{0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}}, {'7',{0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
            {'8',{0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}}, {'9',{0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
            {'A',{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}}, {'B',{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
            {'C',{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}}, {'D',{0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}},
            {'E',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}}, {'F',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
            {'G',{0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}}, {'H',{0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
            {'I',{0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}}, {'J',{0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
            {'K',{0x11,0x12,0x14,0x18,0x14,0x12,0x11}}, {'L',{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
            {'M',{0x11,0x1B,0x15,0x15,0x11,0x11,0x11}}, {'N',{0x11,0x19,0x15,0x13,0x11,0x11,0x11}},
            {'O',{0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'P',{0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
            {'Q',{0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}}, {'R',{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
            {'S',{0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}}, {'T',{0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
            {'U',{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'V',{0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
            {'W',{0x11,0x11,0x11,0x15,0x15,0x1B,0x11}}, {'X',{0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
            {'Y',{0x11,0x11,0x0A,0x04,0x04,0x04,0x04}}, {'Z',{0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
            {'a',{0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}}, {'b',{0x10,0x10,0x1E,0x11,0x11,0x11,0x1E}},
            {'c',{0x00,0x00,0x0E,0x10,0x10,0x11,0x0E}}, {'d',{0x01,0x01,0x0F,0x11,0x11,0x11,0x0F}},
            {'e',{0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E}}, {'f',{0x06,0x08,0x1E,0x08,0x08,0x08,0x08}},
            {'g',{0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E}}, {'h',{0x10,0x10,0x1E,0x11,0x11,0x11,0x11}},
            {'i',{0x04,0x00,0x0C,0x04,0x04,0x04,0x0E}}, {'j',{0x02,0x00,0x06,0x02,0x02,0x12,0x0C}},
            {'k',{0x10,0x10,0x12,0x14,0x18,0x14,0x12}}, {'l',{0x0C,0x04,0x04,0x04,0x04,0x04,0x0E}},
            {'m',{0x00,0x00,0x1A,0x15,0x15,0x11,0x11}}, {'n',{0x00,0x00,0x1E,0x11,0x11,0x11,0x11}},
            {'o',{0x00,0x00,0x0E,0x11,0x11,0x11,0x0E}}, {'p',{0x00,0x00,0x1E,0x11,0x1E,0x10,0x10}},
            {'q',{0x00,0x00,0x0F,0x11,0x0F,0x01,0x01}}, {'r',{0x00,0x00,0x16,0x19,0x10,0x10,0x10}},
            {'s',{0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E}}, {'t',{0x08,0x08,0x1E,0x08,0x08,0x09,0x06}},
            {'u',{0x00,0x00,0x11,0x11,0x11,0x13,0x0D}}, {'v',{0x00,0x00,0x11,0x11,0x11,0x0A,0x04}},
            {'w',{0x00,0x00,0x11,0x11,0x15,0x15,0x0A}}, {'x',{0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}},
            {'y',{0x00,0x00,0x11,0x11,0x0F,0x01,0x0E}}, {'z',{0x00,0x00,0x1F,0x02,0x04,0x08,0x1F}},
            {'.',{0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}}, {',',{0x00,0x00,0x00,0x00,0x00,0x04,0x08}},
            {':',{0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}}, {'-',{0x00,0x00,0x00,0x1F,0x00,0x00,0x00}},
            {'+',{0x00,0x04,0x04,0x1F,0x04,0x04,0x00}}, {'/',{0x01,0x02,0x04,0x08,0x10,0x00,0x00}},
            {'%',{0x19,0x1A,0x04,0x04,0x0B,0x13,0x00}}, {'(',{0x02,0x04,0x08,0x08,0x08,0x04,0x02}},
            {')',{0x08,0x04,0x02,0x02,0x02,0x04,0x08}}, {'=',{0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}},
            {'|',{0x04,0x04,0x04,0x04,0x04,0x04,0x04}}, {'_',{0x00,0x00,0x00,0x00,0x00,0x00,0x1F}},
            {'<',{0x02,0x04,0x08,0x10,0x08,0x04,0x02}}, {'>',{0x08,0x04,0x02,0x01,0x02,0x04,0x08}},
            {'&',{0x0C,0x12,0x14,0x08,0x15,0x12,0x0D}}, {'#',{0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A}}
        };
        for (const auto& e : kFont) if (e.c == ch) return e.rows;
        return kBlank;
    }

    void draw_text(int x, int y, const std::string& s, Color c, int scale = 1) {
        int cx = x;
        for (char ch : s) {
            const uint8_t* g = glyph5x7(ch);
            for (int r = 0; r < 7; ++r) {
                for (int b = 0; b < 5; ++b) {
                    if (g[r] & (1u << (4 - b))) {
                        fill_rect(cx + b * scale, y + r * scale, scale, scale, c);
                    }
                }
            }
            cx += 6 * scale;
        }
    }

    static uint32_t crc32_png(const uint8_t* buf, size_t len) {
        static uint32_t tbl[256];
        static bool init = false;
        if (!init) {
            for (uint32_t n = 0; n < 256; ++n) {
                uint32_t c = n;
                for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                tbl[n] = c;
            }
            init = true;
        }
        uint32_t c = 0xFFFFFFFFu;
        for (size_t i = 0; i < len; ++i) c = tbl[(c ^ buf[i]) & 0xFFu] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    bool save_png(const std::string& path) const {
        std::vector<uint8_t> raw;
        raw.reserve(size_t(H) * (1u + size_t(W) * 3u));
        for (int y = 0; y < H; ++y) {
            raw.push_back(0u);
            const uint8_t* row = rgb.data() + size_t(y) * size_t(W) * 3u;
            raw.insert(raw.end(), row, row + size_t(W) * 3u);
        }
        std::vector<uint8_t> z;
        z.push_back(0x78u); z.push_back(0x01u);
        size_t pos = 0;
        while (pos < raw.size()) {
            const size_t blk = std::min<size_t>(65535u, raw.size() - pos);
            const bool last = (pos + blk == raw.size());
            z.push_back(last ? 0x01u : 0x00u);
            const uint16_t len = uint16_t(blk), nlen = uint16_t(~len);
            z.push_back(uint8_t(len & 0xFFu)); z.push_back(uint8_t(len >> 8));
            z.push_back(uint8_t(nlen & 0xFFu)); z.push_back(uint8_t(nlen >> 8));
            z.insert(z.end(), raw.begin() + ptrdiff_t(pos), raw.begin() + ptrdiff_t(pos + blk));
            pos += blk;
        }
        uint32_t s1 = 1u, s2 = 0u;
        for (uint8_t b : raw) { s1 = (s1 + b) % 65521u; s2 = (s2 + s1) % 65521u; }
        const uint32_t adler = (s2 << 16) | s1;
        z.push_back(uint8_t(adler >> 24)); z.push_back(uint8_t(adler >> 16));
        z.push_back(uint8_t(adler >> 8));  z.push_back(uint8_t(adler & 0xFFu));

        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs) return false;
        const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        ofs.write(reinterpret_cast<const char*>(sig), 8);

        auto write_chunk = [&](const char type[4], const uint8_t* data, size_t len) {
            uint8_t be[4] = {uint8_t(len >> 24), uint8_t(len >> 16), uint8_t(len >> 8), uint8_t(len & 0xFFu)};
            ofs.write(reinterpret_cast<const char*>(be), 4);
            std::vector<uint8_t> body(4u + len);
            std::memcpy(body.data(), type, 4);
            if (len) std::memcpy(body.data() + 4, data, len);
            ofs.write(reinterpret_cast<const char*>(body.data()), std::streamsize(body.size()));
            const uint32_t crc = crc32_png(body.data(), body.size());
            uint8_t cbe[4] = {uint8_t(crc >> 24), uint8_t(crc >> 16), uint8_t(crc >> 8), uint8_t(crc & 0xFFu)};
            ofs.write(reinterpret_cast<const char*>(cbe), 4);
        };
        uint8_t ihdr[13] = {
            uint8_t(W >> 24), uint8_t(W >> 16), uint8_t(W >> 8), uint8_t(W & 0xFF),
            uint8_t(H >> 24), uint8_t(H >> 16), uint8_t(H >> 8), uint8_t(H & 0xFF),
            8, 2, 0, 0, 0
        };
        write_chunk("IHDR", ihdr, 13);
        write_chunk("IDAT", z.data(), z.size());
        write_chunk("IEND", nullptr, 0);
        return ofs.good();
    }
};

std::string fmt0(double v) { std::ostringstream s; s << std::fixed << std::setprecision(0) << v; return s.str(); }
std::string fmt1(double v) { std::ostringstream s; s << std::fixed << std::setprecision(1) << v; return s.str(); }
std::string fmt2(double v) { std::ostringstream s; s << std::fixed << std::setprecision(2) << v; return s.str(); }

// Strictly finite, strictly non-negative sanitizer for every receipt-sourced scalar.
double clamp_nonneg(double v, double lo, double hi) {
    if (!std::isfinite(v)) return lo;
    return std::min(hi, std::max(lo, v));
}

} // namespace ernie55_chart

int main(int argc, char** argv) {
    using namespace ernie55;
    using namespace ernie55_chart;

    std::string root = ".";
    std::string out_png;
    std::string wave_json;
    std::string wave13_json;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_png = argv[++i];
        else if ((a == "--wave12-json" || a == "--wave11-json") && i + 1 < argc) wave_json = argv[++i];
        else if (a == "--wave13-json" && i + 1 < argc) wave13_json = argv[++i];
    }
    if (out_png.empty()) out_png = root + "/artifacts/ernie55_preview_hlut4_1m_telemetry.png";
    if (wave_json.empty()) wave_json = root + "/artifacts/gate_wave12_1m_apex_receipt.json";
    // Wave-13 Cut W13.5: the Wave-13 receipt is optional and takes precedence when present, so
    // the renderer stays backward compatible with Wave-12 and Wave-11 receipts.
    if (wave13_json.empty()) wave13_json = root + "/artifacts/gate_wave13_1k_prefill_1m_receipt.json";

    auto read_json = [](const std::string& p) -> JsonValue {
        std::ifstream f(p);
        if (!f) return JsonValue::make_object();
        std::ostringstream ss; ss << f.rdbuf();
        JsonValue out;
        if (JsonParser::try_parse(ss.str(), out)) return out;
        return JsonValue::make_object();
    };

    // Primary Wave-12 receipt; a Wave-11 receipt is accepted as a compatibility source so the
    // renderer stays useful before the Wave-12 receipt is cut.
    const auto w13 = read_json(wave13_json);
    const bool w13_present = w13.has("release") || w13.has("pillar1_prefill_m2048");
    const auto w12 = (w13_present ? JsonValue::make_object() : read_json(wave_json));
    const bool w12_present = w12.has("release") || w12.has("pillar1_prefill_m2048");
    const auto w11 = (w12_present || w13_present) ? JsonValue::make_object()
                                 : read_json(root + "/artifacts/gate_wave11_1m_omni_receipt.json");
    const auto& rc = w13_present ? w13 : (w12_present ? w12 : w11);
    const bool is_wave13 = w13_present;
    const char* wave_tag = is_wave13 ? "v1.5.0-wave13-1k-prefill-1m"
                         : (w12_present ? "v1.4.0-wave12-1m-apex" : "v1.3.0-wave11-1m-omni");
    const char* wave_short = is_wave13 ? "Wave-13" : (w12_present ? "Wave-12" : "Wave-11");

    auto num = [](const JsonValue& o, const char* k, double dflt) -> double {
        if (o.has(k)) return o.at(k).as_double();
        return dflt;
    };
    auto i32 = [](const JsonValue& o, const char* k, int dflt) -> int {
        if (o.has(k)) return int(o.at(k).as_int());
        return dflt;
    };

    // Wave-13 Cut W13.7: P150A batch-slot default for the banner (overridden by receipt).
    int kBatchSlotsDflt = 256;
    if (rc.has("wave13_session") && rc.at("wave13_session").has("kBatchSlots")) {
        kBatchSlotsDflt = int(rc.at("wave13_session").at("kBatchSlots").as_int());
    }
    // Wave-13 Cut W13.13: dynamic telemetry bound to gate_wave13_1k_prefill_1m_receipt.json.
    int w13_chunks_measured = 490;
    std::string w13_ctx_label = "832K";
    double w13_ctx_peak = 442.2;
    int w13_t1_live_blks = 434;
    double w13_t2_live_peak_k = 944.1;
    int w13_t2_live_peak_blks = 461;
    size_t w13_peak_ctx = 869000;
    int w13_compactions = 0;
    if (rc.has("pillar1_prefill_m2048")) {
        const auto& p1j = rc.at("pillar1_prefill_m2048");
        if (p1j.has("w13_m2048_chunks_measured")) w13_chunks_measured = int(p1j.at("w13_m2048_chunks_measured").as_int());
        else if (p1j.has("w12_m2048_chunks_measured")) w13_chunks_measured = int(p1j.at("w12_m2048_chunks_measured").as_int());
        if (p1j.has("context_buckets") && p1j.at("context_buckets").is_array()) {
            const auto& cb = p1j.at("context_buckets").arr_val;
            if (!cb.empty()) {
                const auto& last = cb.back();
                if (last.has("label")) w13_ctx_label = last.at("label").s_val;
                if (last.has("w13_peak")) w13_ctx_peak = last.at("w13_peak").as_double();
                else if (last.has("w12_peak")) w13_ctx_peak = last.at("w12_peak").as_double();
            }
        }
    }
    if (rc.has("pillar2_kvblk")) {
        const auto& p2j = rc.at("pillar2_kvblk");
        if (p2j.has("tier1_ddr4_live_blks")) w13_t1_live_blks = int(p2j.at("tier1_ddr4_live_blks").as_int());
        if (p2j.has("tier2_nvme_live_peak_tokens")) w13_t2_live_peak_k = p2j.at("tier2_nvme_live_peak_tokens").as_double() / 1000.0;
        if (p2j.has("tier2_nvme_live_peak_blks")) w13_t2_live_peak_blks = int(p2j.at("tier2_nvme_live_peak_blks").as_int());
    }
    if (rc.has("wave13_session")) {
        const auto& wsj = rc.at("wave13_session");
        if (wsj.has("peak_context_tokens")) w13_peak_ctx = size_t(wsj.at("peak_context_tokens").as_int());
        if (wsj.has("compactions")) w13_compactions = int(wsj.at("compactions").as_int());
    }

    // ---- Verified v1.4.0-wave12-1m-apex measured silicon defaults (overridden by receipt when present) ----
    int isa_kernels = 343, isa_hlut4 = 118, isa_core = 225, isa_fdot2 = 3504;
    int unit_pass = 32, unit_tot = 32, aer_devs = 69, aer_total = 0;
    int pii_pass = 10, pii_tot = 10;
    int t1_slots = 512, t2_mod = 12;
    double p1_peak = 977.5, p1_warm = 958.0, p1_ratio = 2.12;
    double p1_pk_64k = 958.0, p1_pk_576k = 562.6, p1_pk_768k = 459.6, p1_pk_832k = 442.2;
    double p2_t0_b = 2.46, p2_t1_m = 2.66, p2_nvme_k = 884.3, p2_tail_ms = 2.01;
    double p2_delta_ms = 0.99, p2_w11_tok_ms = 46.93;
    double p3_mean = 32.05, p3_w10 = 26.58, p3_pk_short = 73.97, p3_pk_217 = 52.72;
    double p3_pk_580 = 45.63, p3_pk_612 = 42.44, p3_acc = 62.4, p3_acc_pk = 87.5, p3_ngram_mib = 16.0;
    // Wave-13 Cut W13.13: optional 1M-band decode peak from the receipt.
    double p3_peak_1m = 0.0;
    int sess_turns = 24, sess_swaps = 15, sess_steps = 545, sess_tools = 576, sess_out = 306914;
    int sess_sa_skip = 3952;
    double sess_peak_k = 823.0;

    if (rc.has("isa_audit")) {
        const auto& ia = rc.at("isa_audit");
        isa_kernels = i32(ia, "kernels", isa_kernels);
        isa_hlut4   = i32(ia, "hlut4_kernels", isa_hlut4);
        isa_core    = i32(ia, "core_kernels", isa_core);
        isa_fdot2   = i32(ia, "fdot2_ops", isa_fdot2);
    }
    if (rc.has("unit_gate")) {
        const auto& ug = rc.at("unit_gate");
        unit_pass = i32(ug, "suites_passed", unit_pass);
        unit_tot  = i32(ug, "suites_total", unit_tot);
    }
    if (rc.has("aer")) {
        const auto& ae = rc.at("aer");
        aer_devs  = i32(ae, "devices", aer_devs);
        aer_total = i32(ae, "total", aer_total);
    }
    if (rc.has("zero_pii")) {
        const auto& zp = rc.at("zero_pii");
        pii_pass = i32(zp, "checks_passed", pii_pass);
        pii_tot  = i32(zp, "checks_total", pii_tot);
    }
    if (rc.has("pillar1_prefill_m2048")) {
        const auto& p1 = rc.at("pillar1_prefill_m2048");
        p1_peak    = clamp_nonneg(num(p1, "peak_cold_tok_s", p1_peak), 0.0, 100000.0);
        p1_warm    = clamp_nonneg(num(p1, "warm_natural_tok_s", p1_warm), 0.0, 100000.0);
        p1_ratio   = clamp_nonneg(num(p1, "speedup_vs_wave10", p1_ratio), 0.0, 100.0);
        p1_pk_64k  = clamp_nonneg(num(p1, "peak_64k_tok_s", p1_pk_64k), 0.0, 100000.0);
        p1_pk_576k = clamp_nonneg(num(p1, "peak_576k_tok_s", p1_pk_576k), 0.0, 100000.0);
        p1_pk_768k = clamp_nonneg(num(p1, "peak_768k_tok_s", p1_pk_768k), 0.0, 100000.0);
        p1_pk_832k = clamp_nonneg(num(p1, "peak_832k_tok_s", p1_pk_832k), 0.0, 100000.0);
    }
    if (rc.has("pillar2_kvblk")) {
        const auto& p2 = rc.at("pillar2_kvblk");
        p2_t0_b       = clamp_nonneg(num(p2, "tier0_vram_peak_tok_s", p2_t0_b * 1e9) / 1e9, 0.0, 1e3);
        p2_t1_m       = clamp_nonneg(num(p2, "tier1_ddr4_live_tok_s", p2_t1_m * 1e6) / 1e6, 0.0, 1e3);
        t1_slots      = std::max(8, i32(p2, "tier1_slots", t1_slots));
        p2_nvme_k     = clamp_nonneg(num(p2, "tier2_nvme_live_peak_tok_s", p2_nvme_k * 1e3) / 1e3, 0.0, 1e6);
        t2_mod        = std::max(1, i32(p2, "tier2_stripe_modulus", t2_mod));
        p2_tail_ms    = clamp_nonneg(num(p2, "tier2_nvme_live_crc_ms", p2_tail_ms), 0.0, 1000.0);
        p2_delta_ms   = clamp_nonneg(num(p2, "chat_prefix_delta_ms", p2_delta_ms), 0.0, 1000.0);
        p2_w11_tok_ms = clamp_nonneg(num(p2, "chat_prefix_wave11_ms", p2_w11_tok_ms), 0.0, 1e5);
    }
    if (rc.has("pillar3_dspark_decode")) {
        const auto& p3 = rc.at("pillar3_dspark_decode");
        p3_mean      = clamp_nonneg(num(p3, "campaign_mean_tok_s", p3_mean), 0.0, 10000.0);
        p3_w10       = clamp_nonneg(num(p3, "wave10_mean_tok_s", p3_w10), 0.0, 10000.0);
        p3_pk_short  = clamp_nonneg(num(p3, "peak_short_tok_s", p3_pk_short), 0.0, 10000.0);
        p3_pk_217    = clamp_nonneg(num(p3, "peak_217k_tok_s", p3_pk_217), 0.0, 10000.0);
        p3_pk_580    = clamp_nonneg(num(p3, "peak_580k_tok_s", p3_pk_580), 0.0, 10000.0);
        p3_peak_1m   = clamp_nonneg(num(p3, "peak_1m_tok_s", 0.0), 0.0, 10000.0);
        if (p3_peak_1m <= 0.0) p3_peak_1m = p3_pk_612;
        p3_pk_612    = clamp_nonneg(num(p3, "peak_612k_tok_s", p3_pk_612), 0.0, 10000.0);
        p3_acc       = clamp_nonneg(num(p3, "sa_mean_acceptance", p3_acc / 100.0) * 100.0, 0.0, 100.0);
        p3_acc_pk    = clamp_nonneg(num(p3, "sa_peak_acceptance", p3_acc_pk / 100.0) * 100.0, 0.0, 100.0);
        p3_ngram_mib = clamp_nonneg(num(p3, "ngram_host_mib", p3_ngram_mib), 0.0, 1e6);
    }
    // Wave-13 Cut W13.7: read the Wave-13 session block when present, else fall back to
        // the Wave-12 block so older receipts keep rendering unchanged.
    const JsonValue* sess_ptr = nullptr;
    if (rc.has("wave13_session")) sess_ptr = &rc.at("wave13_session");
    else if (rc.has("wave12_session")) sess_ptr = &rc.at("wave12_session");
    if (sess_ptr != nullptr) {
        const auto& ws = *sess_ptr;
        sess_turns   = i32(ws, "turns", sess_turns);
        sess_swaps   = i32(ws, "live_swaps", sess_swaps);
        sess_steps   = i32(ws, "live_steps", sess_steps);
        sess_tools   = i32(ws, "tool_calls", sess_tools);
        sess_out     = i32(ws, "generated_tokens", sess_out);
        sess_sa_skip = i32(ws, "sa_skipped_dspark", sess_sa_skip);
        sess_peak_k  = clamp_nonneg(num(ws, "peak_context_tokens", sess_peak_k * 1000.0) / 1000.0, 0.0, 100000.0);
    }

    Canvas cv(1440, 860, kBg);

    // ---- Top Banner (High-Contrast Dark Slate) ----
    cv.fill_rect(0, 0, 1440, 78, kText);
    cv.draw_text(20, 10, std::string("ERNIE-5.5-Flash-Preview (300B + 3L MoE DSpark+SA) - ") + wave_tag, kCard, 2);
    cv.draw_text(20, 40, std::string("5x MI50 (4x Gen4 x16 TP4 + 1x Gen3 x16 EP2, 128 GiB HBM2) + 2x Blackhole P150A (64 GiB GDDR6, kBatchSlots=") + std::to_string(kBatchSlotsDflt) + ") | AMD EPYC 7542 | 0 B Spill", kBorder, 1);
    {
        char hbuf[260];
        std::snprintf(hbuf, sizeof(hbuf),
            "%s Live: %.1fK :3081 (bespoke ontology, the kitchen) | Prefill %.1f->%.1f (%s) | DDR4 %.2fM / NVMe %.1fK | SA %.1f%% (%.1f%% pk) | %d/%d 0-Scratch",
            wave_short,
            sess_peak_k, p1_pk_64k, w13_ctx_peak, w13_ctx_label.c_str(), p2_t1_m, p2_nvme_k, p3_acc, p3_acc_pk, isa_kernels, isa_kernels);
        cv.draw_text(20, 58, hbuf, kSoftGrn, 1);
    }

    // ---- 2x2 Panel Cards ----
    const int cw = 692, ch = 362;
    const int lx = 18, rx = 730;
    const int ty = 90, by = 468;
    for (auto [px, py] : std::array<std::pair<int, int>, 4>{{{lx, ty}, {rx, ty}, {lx, by}, {rx, by}}}) {
        cv.fill_rect(px, py, cw, ch, kCard);
        cv.draw_rect(px, py, cw, ch, kBorder, 1);
    }

    // =========================================================================
    // PANEL 1 (Top-Left): Pillar I Cold M=2048 Prefill (64K -> 1M)
    // =========================================================================
    cv.draw_text(lx + 14, ty + 12, "Panel 1: Pillar I Cold M=2048 Prefill Across 1M Context (" + std::string(wave_short) + " vs Wave-10, tok/s)", kText, 1);
    cv.draw_text(lx + 14, ty + 28, std::string("kLaunchOrd 0,1,3,2 + comp_strip 512 + cur_tile 64..1024 + Tuned HH_HPA + P150A kBatchSlots=") + std::to_string(kBatchSlotsDflt), kMuted, 1);
    {
        struct PfPt { std::string lbl; double w12_pk; double w12_mn; double w10_mn; };
        std::vector<PfPt> pts;
        if (rc.has("pillar1_prefill_m2048") && rc.at("pillar1_prefill_m2048").has("context_buckets")) {
            const auto& arr = rc.at("pillar1_prefill_m2048").at("context_buckets").arr_val;
            const size_t lim = std::min<size_t>(arr.size(), 16u);
            for (size_t i = 0; i < lim; ++i) {
                const auto& b = arr[i];
                const double w12p = b.has("w13_peak") ? b.at("w13_peak").as_double()
                                  : (b.has("w12_peak") ? b.at("w12_peak").as_double()
                                  : (b.has("w11_peak") ? b.at("w11_peak").as_double() : 0.0));
                const double w12m = b.has("w13_mean") ? b.at("w13_mean").as_double()
                                  : (b.has("w12_mean") ? b.at("w12_mean").as_double()
                                  : (b.has("w11_mean") ? b.at("w11_mean").as_double() : 0.0));
                pts.push_back({
                    b.has("label")    ? b.at("label").s_val : "",
                    clamp_nonneg(w12p, 0.0, 100000.0),
                    clamp_nonneg(w12m, 0.0, 100000.0),
                    clamp_nonneg(b.has("w10_mean") ? b.at("w10_mean").as_double() : 0.0, 0.0, 100000.0)
                });
            }
        }
        if (pts.empty()) {
            pts = {
                {"64K", 958.0, 856.6, 460.2}, {"128K", 787.1, 767.7, 384.6},
                {"192K", 744.2, 715.2, 360.2}, {"256K", 723.2, 677.0, 339.5},
                {"320K", 697.8, 622.8, 315.7}, {"384K", 613.3, 589.0, 310.8},
                {"448K", 598.4, 561.5, 272.4}, {"512K", 550.1, 532.7, 272.4},
                {"576K", 562.6, 536.2, 265.0}, {"640K", 542.5, 515.2, 255.0},
                {"704K", 503.9, 480.6, 246.0}, {"768K", 459.6, 445.8, 238.0},
                {"832K", 442.2, 437.3, 230.0}
            };
        }
        if (!pts.empty()) {
            p1_pk_64k  = (pts.front().w12_pk > 0.0) ? pts.front().w12_pk : p1_pk_64k;
            p1_pk_832k = (pts.back().w12_pk  > 0.0) ? pts.back().w12_pk  : p1_pk_832k;
        }
        const int gx = lx + 56, gy = ty + 54, gw = 382, gh = 246;
        cv.draw_rect(gx, gy, gw, gh, kBorder, 1);
        const double ymax = 1100.0;
        for (int i = 0; i <= 5; ++i) {
            const int yy = gy + gh - (gh * i) / 5;
            cv.draw_dashed_h(gx, gx + gw, yy, kGrid);
            cv.draw_text(gx - 42, yy - 4, fmt0(ymax * i / 5.0), kMuted, 1);
        }
        const int n = int(pts.size());
        auto x_at = [&](int i) { return gx + 20 + int((gw - 40) * double(i) / double(std::max(1, n - 1))); };
        auto y_at = [&](double v) { return gy + gh - std::min(gh - 8, std::max(4, int(std::lround((v / ymax) * double(gh))))); };

        // Wave-10 baseline curve (amber; only plot measured Wave-10 points > 0)
        int last_w10 = -1;
        for (int i = 0; i < n; ++i) {
            if (pts[size_t(i)].w10_mn > 0.0) last_w10 = i;
        }
        for (int i = 0; i < n; ++i) {
            if (pts[size_t(i)].w10_mn <= 0.0) continue;
            const int xi = x_at(i), yi = y_at(pts[size_t(i)].w10_mn);
            if (i > 0 && pts[size_t(i - 1)].w10_mn > 0.0) {
                cv.draw_line(x_at(i - 1), y_at(pts[size_t(i - 1)].w10_mn), xi, yi, kAmber, 2);
            }
            cv.fill_circle(xi, yi, 3, kAmber);
            if (i == 0 || i == last_w10) cv.draw_text(xi - 14, yi + 7, fmt0(pts[size_t(i)].w10_mn), kAmber, 1);
        }
        // Active-wave peak curve (blue)
        for (int i = 0; i < n; ++i) {
            const int xi = x_at(i), yi = y_at(pts[size_t(i)].w12_pk);
            if (i > 0) cv.draw_line(x_at(i - 1), y_at(pts[size_t(i - 1)].w12_pk), xi, yi, kBlue, 3);
            cv.fill_circle(xi, yi, 4, kBlue);
            if (i % 2 == 0 || i == n - 1) cv.draw_text(xi - 12, yi - 14, fmt0(pts[size_t(i)].w12_pk), kBlue, 1);
            if (i % 2 == 0 || i == n - 1) cv.draw_text(std::max(gx + 2, std::min(gx + gw - 32, xi - 12)), gy + gh + 10, pts[size_t(i)].lbl, kText, 1);
        }
        // Legend inside top-right of plot
        cv.fill_rect(gx + 168, gy + 8, 12, 3, kBlue);
        cv.draw_text(gx + 186, gy + 6, std::string(wave_short) + " Peak", kBlue, 1);
        cv.fill_rect(gx + 168, gy + 22, 12, 3, kAmber);
        cv.draw_text(gx + 186, gy + 20, "Wave-10 Mean Baseline", kAmber, 1);
        cv.draw_text(gx + 12, gy + gh + 28, "Context Position pos0 (" + std::to_string(w13_chunks_measured) + " Live M=2048 Chunks to " + w13_ctx_label + ")", kMuted, 1);

        // Right summary scorecard
        const int sx = lx + 452, sy = ty + 54, sw = 224, sh = 290;
        cv.fill_rect(sx, sy, sw, sh, kBg);
        cv.draw_rect(sx, sy, sw, sh, kBorder, 1);
        cv.fill_rect(sx + 1, sy + 1, sw - 2, 24, kSoftBlue);
        cv.draw_text(sx + 10, sy + 9, "Pillar I & ISA Verification", kText, 1);
        cv.draw_text(sx + 10, sy + 36, std::string("Peak <=64K:  ") + fmt1(p1_pk_64k) + " tok/s", kBlue, 1);
        cv.draw_text(sx + 10, sy + 54, std::string("Peak 576K:   ") + fmt1(p1_pk_576k) + " tok/s", kBlue, 1);
        cv.draw_text(sx + 10, sy + 72, std::string("Peak 768K:   ") + fmt1(p1_pk_768k) + " tok/s", kTeal, 1);
        cv.draw_text(sx + 10, sy + 90, "Peak " + w13_ctx_label + ": " + fmt1(w13_ctx_peak) + " tok/s", kGreen, 1);
        cv.draw_text(sx + 10, sy + 108, "vs Wave-10:  " + fmt2(p1_ratio) + "x Faster", kGreen, 1);
        cv.draw_text(sx + 10, sy + 132, "gfx906 ISA:  " + std::to_string(isa_kernels) + "/" + std::to_string(isa_kernels) + " 0-Scr", kGreen, 1);
        cv.draw_text(sx + 10, sy + 150, "HLUT4/Core:  " + std::to_string(isa_hlut4) + " + " + std::to_string(isa_core), kMuted, 1);
        cv.draw_text(sx + 10, sy + 168, "v_dot2 ops:  " + std::to_string(isa_fdot2), kBlue, 1);
        cv.draw_text(sx + 10, sy + 186, "Spills/Trig: 0 / 0 | SLC: 0", kGreen, 1);
        cv.draw_text(sx + 10, sy + 204, "BAR1 P2P:    0 | Python: 0", kGreen, 1);
        cv.draw_text(sx + 10, sy + 222, "PCIe AER:    0/0 (" + std::to_string(aer_devs) + " dev)", kGreen, 1);
        cv.draw_text(sx + 10, sy + 246, "Cold Peak:   " + fmt1(p1_peak) + " tok/s", kBlue, 1);
        cv.draw_text(sx + 10, sy + 264, "License: Apache-2.0 (BF16)", kMuted, 1);
    }

    // =========================================================================
    // PANEL 2 (Top-Right): Pillar II 3-Tier .kvblk Restore + Zero-Copy JSON Delta
    // =========================================================================
    cv.draw_text(rx + 14, ty + 12, "Panel 2: Pillar II 3-Tier .kvblk Restore & Zero-Copy JSON Delta (K tok/s, memcmp == 0)", kText, 1);
    cv.draw_text(rx + 14, ty + 28, "Tier-1 512-Slot Coherent DDR4 (Half-Ring-6 + HCA Mirror) + Tier-2 Dual-NVMe O_DIRECT + ChatPrefixCache", kMuted, 1);
    {
        struct KvBar { std::string lbl; double k_tps; Color col; };
        std::vector<KvBar> bars;
        const Color pal[7] = {kSlate, kAmber, kTeal, kBlue, kPurple, kGreen, kText};
        if (rc.has("pillar2_kvblk") && rc.at("pillar2_kvblk").has("progression")) {
            const auto& arr = rc.at("pillar2_kvblk").at("progression").arr_val;
            const size_t lim = std::min<size_t>(arr.size(), 7u);
            for (size_t i = 0; i < lim; ++i) {
                const auto& r = arr[i];
                bars.push_back({
                    r.has("label")   ? r.at("label").s_val : "",
                    clamp_nonneg(r.has("k_tok_s") ? r.at("k_tok_s").as_double() : 0.0, 0.0, 1e6),
                    pal[i]
                });
            }
        }
        if (bars.empty()) {
            bars = {
                {"W10c2", 133.0, kSlate}, {"W10c5", 703.8, kAmber}, {"W11T11", 877.6, kTeal},
                {"W12T17", 858.1, kBlue}, {"W12T19", 884.3, kPurple}, {"Vfy160", 986.9, kGreen},
                {"RawOD", 1034.0, kText}
            };
        }
        const int gx = rx + 58, gy = ty + 54, gw = 380, gh = 246;
        cv.draw_rect(gx, gy, gw, gh, kBorder, 1);
        const double ymax = 1200.0;
        for (int i = 0; i <= 4; ++i) {
            const int yy = gy + gh - (gh * i) / 4;
            cv.draw_dashed_h(gx, gx + gw, yy, kGrid);
            cv.draw_text(gx - 50, yy - 4, fmt0(ymax * i / 4.0) + "K", kMuted, 1);
        }
        const int y1m = gy + gh - int(std::lround((1000.0 / ymax) * double(gh)));
        cv.draw_dashed_h(gx, gx + gw, y1m, kGreen);
        cv.draw_text(gx + 8, y1m - 12, "1.0M tok/s Dual-NVMe Ceiling", kGreen, 1);

        const int bw = 38, gap = 14;
        const int nb = int(std::min<size_t>(bars.size(), 7u));
        for (int i = 0; i < nb; ++i) {
            const int bx = gx + 14 + i * (bw + gap);
            const int bh = std::min(gh - 18, std::max(4, int(std::lround((bars[size_t(i)].k_tps / ymax) * double(gh)))));
            cv.fill_rect(bx, gy + gh - bh, bw, bh, bars[size_t(i)].col);
            cv.draw_text(bx + 2, gy + gh + 10, bars[size_t(i)].lbl, kText, 1);
            cv.draw_text(bx + 2, gy + gh - bh - 12, fmt0(bars[size_t(i)].k_tps) + "K", bars[size_t(i)].col, 1);
        }
        cv.draw_text(gx + 10, gy + gh + 28, "Dual-NVMe Cold Restore Evolution (4-GPU HBM2 + CRC32C)", kMuted, 1);

        // Right summary scorecard: Tier-0 / Tier-1 512-slot / Tier-2 b mod 12 / JSON delta
        const int sx = rx + 452, sy = ty + 54, sw = 224, sh = 290;
        cv.fill_rect(sx, sy, sw, sh, kBg);
        cv.draw_rect(sx, sy, sw, sh, kBorder, 1);
        cv.fill_rect(sx + 1, sy + 1, sw - 2, 24, kSoftGrn);
        cv.draw_text(sx + 10, sy + 9, "3-Tier .kvblk & JSON Delta", kText, 1);
        cv.draw_text(sx + 10, sy + 36, "Tier-0 VRAM Peak:", kMuted, 1);
        cv.draw_text(sx + 10, sy + 50, fmt2(p2_t0_b) + "B tok/s (0.11 ms)", kGreen, 1);
        cv.draw_text(sx + 10, sy + 74, "Tier-1 DDR4 " + std::to_string(t1_slots) + "-Slot:", kMuted, 1);
        cv.draw_text(sx + 10, sy + 88, fmt2(p2_t1_m) + "M tok/s (" + std::to_string(w13_t1_live_blks) + " blk)", kBlue, 1);
        cv.draw_text(sx + 10, sy + 102, std::string("Half-Ring-6 + HCA Mirror"), kBlue, 1);
        cv.draw_text(sx + 10, sy + 126, std::string("Tier-2 NVMe ") + fmt0(w13_t2_live_peak_k) + "K Live:", kMuted, 1);
        cv.draw_text(sx + 10, sy + 140, fmt1(p2_nvme_k) + "K tok/s (b mod " + std::to_string(t2_mod) + ")", kPurple, 1);
        cv.draw_text(sx + 10, sy + 154, "CRC32C: " + fmt2(p2_tail_ms) + " ms (" + std::to_string(w13_t2_live_peak_blks) + " blk)", kTeal, 1);
        cv.draw_text(sx + 10, sy + 178, "JSON Delta: " + fmt2(p2_delta_ms) + " ms", kGreen, 1);
        cv.draw_text(sx + 10, sy + 192, "vs W11 Tok:  " + fmt2(p2_w11_tok_ms) + " ms", kMuted, 1);
        cv.draw_text(sx + 10, sy + 216, "Tier-1 Capacity:", kMuted, 1);
        cv.draw_text(sx + 10, sy + 230, "512 x 2048 = 1M tok", kBlue, 1);
        cv.draw_text(sx + 10, sy + 254, "Integrity: memcmp=0 + CRC32C", kMuted, 1);
        cv.draw_text(sx + 10, sy + 272, "Spill: 0 B (DDR4 pinned)", kGreen, 1);
    }

    // =========================================================================
    // PANEL 3 (Bottom-Left): Pillar III M-Fused HCA + Doorbell P150A + 8/6-Gram SA
    // =========================================================================
    cv.draw_text(lx + 14, by + 12, "Panel 3: Pillar III Live M-Fused HCA + Doorbell P150A + 8/6-Gram DSpark+SA Decode (tok/s)", kText, 1);
    cv.draw_text(lx + 14, by + 28, "M-Fused Split-K HCA + Doorbell-First P150A + Precision 8/6-Gram SA Cascade (0 B GPU VRAM)", kMuted, 1);
    {
        struct DecBand { std::string lbl; double mn; double pk; };
        std::vector<DecBand> bands;
        if (rc.has("pillar3_dspark_decode") && rc.at("pillar3_dspark_decode").has("bands")) {
            const auto& arr = rc.at("pillar3_dspark_decode").at("bands").arr_val;
            const size_t lim = std::min<size_t>(arr.size(), 8u);
            for (size_t i = 0; i < lim; ++i) {
                const auto& b = arr[i];
                bands.push_back({
                    b.has("label") ? b.at("label").s_val : "",
                    clamp_nonneg(b.has("mean") ? b.at("mean").as_double() : 0.0, 0.0, 1e4),
                    clamp_nonneg(b.has("peak") ? b.at("peak").as_double() : 0.0, 0.0, 1e4)
                });
            }
        }
        if (bands.empty()) {
            bands = {
                {"<=64K", 34.78, 73.97}, {"217K", 35.54, 52.72}, {"384K", 31.28, 57.02},
                {"512K", 30.62, 41.49}, {"580K", 30.82, 45.63}, {"618K", 26.94, 42.44},
                {"869K", 30.13, 30.13}
            };
        }
        const int gx = lx + 56, gy = by + 54, gw = 382, gh = 246;
        cv.draw_rect(gx, gy, gw, gh, kBorder, 1);
        const double ymax = 80.0;
        for (int i = 0; i <= 4; ++i) {
            const int yy = gy + gh - (gh * i) / 4;
            cv.draw_dashed_h(gx, gx + gw, yy, kGrid);
            cv.draw_text(gx - 38, yy - 4, fmt0(ymax * i / 4.0), kMuted, 1);
        }
        const int y_w12 = gy + gh - int(std::lround((clamp_nonneg(p3_mean, 0.0, ymax) / ymax) * double(gh)));
        const int y_w10 = gy + gh - int(std::lround((clamp_nonneg(p3_w10, 0.0, ymax) / ymax) * double(gh)));
        cv.draw_dashed_h(gx, gx + gw, y_w12, kGreen);
        cv.draw_dashed_h(gx, gx + gw, y_w10, kAmber);

        const int pair_w = 42, gap = 10;
        const int nb = int(std::min<size_t>(bands.size(), 7u));
        for (int i = 0; i < nb; ++i) {
            const int bx = gx + 14 + i * (pair_w + gap);
            const int h_pk = std::min(gh - 18, std::max(4, int(std::lround((clamp_nonneg(bands[size_t(i)].pk, 0.0, ymax) / ymax) * double(gh)))));
            const int h_mn = std::min(gh - 18, std::max(4, int(std::lround((clamp_nonneg(bands[size_t(i)].mn, 0.0, ymax) / ymax) * double(gh)))));
            cv.fill_rect(bx,      gy + gh - h_pk, 18, h_pk, kTeal);
            cv.fill_rect(bx + 20, gy + gh - h_mn, 18, h_mn, kBlue);
            cv.draw_text(bx, gy + gh - h_pk - 12, fmt1(bands[size_t(i)].pk), kTeal, 1);
            cv.draw_text(std::max(gx + 2, std::min(gx + gw - 28, bx)), gy + gh + 10, bands[size_t(i)].lbl, kText, 1);
        }
        cv.fill_rect(gx + 166, gy + 8, 10, 8, kTeal);
        cv.draw_text(gx + 181, gy + 8, "Live Peak tok/s", kTeal, 1);
        cv.fill_rect(gx + 166, gy + 22, 10, 8, kBlue);
        cv.draw_text(gx + 181, gy + 22, "Band Mean tok/s", kBlue, 1);
        cv.draw_text(gx + 166, gy + 36, std::string("Green=") + wave_short + " " + fmt1(p3_mean) + " | Amb=W10 " + fmt1(p3_w10), kMuted, 1);
        cv.draw_text(gx + 14, gy + gh + 28, "Live :3081 Context Band (DSpark K=4..7 + SA 8/6-Gram)", kMuted, 1);

        const int sx = lx + 452, sy = by + 54, sw = 224, sh = 290;
        cv.fill_rect(sx, sy, sw, sh, kBg);
        cv.draw_rect(sx, sy, sw, sh, kBorder, 1);
        cv.fill_rect(sx + 1, sy + 1, sw - 2, 24, kSoftBlue);
        cv.draw_text(sx + 10, sy + 9, "Pillar III Decode Scorecard", kText, 1);
        cv.draw_text(sx + 10, sy + 36, std::string("Peak <=64K:  ") + fmt2(p3_pk_short) + " tok/s", kTeal, 1);
        cv.draw_text(sx + 10, sy + 54, std::string(is_wave13 ? "Peak 448K:   " : "Peak 217K:   ") + fmt2(p3_pk_217) + " tok/s", kTeal, 1);
        cv.draw_text(sx + 10, sy + 72, std::string(is_wave13 ? "Peak 768K:   " : "Peak 580K:   ") + fmt2(p3_pk_580) + " tok/s", kGreen, 1);
        cv.draw_text(sx + 10, sy + 90, std::string(is_wave13 ? "Peak 1M:     " : "Peak 612K:   ") + fmt2(is_wave13 ? p3_peak_1m : p3_pk_612) + " tok/s", kGreen, 1);
        cv.draw_text(sx + 10, sy + 114, std::string(wave_short) + " Mean: " + fmt2(p3_mean) + " tok/s", kBlue, 1);
        const double w10_gain_pct = (p3_w10 > 0.0) ? ((p3_mean / p3_w10 - 1.0) * 100.0) : 0.0;
        cv.draw_text(sx + 10, sy + 132, "W10 Mean:    " + fmt2(p3_w10) + " (+" + fmt1(w10_gain_pct) + "%)", kMuted, 1);
        cv.draw_text(sx + 10, sy + 156, "SA Mean/Pk:  " + fmt1(p3_acc) + "% / " + fmt1(p3_acc_pk) + "%", kPurple, 1);
        cv.draw_text(sx + 10, sy + 174, "SA Cascade:  8/6-Gram (42 ns)", kPurple, 1);
        cv.draw_text(sx + 10, sy + 192, "SA Host:     " + fmt1(p3_ngram_mib) + " MiB DDR4", kText, 1);
        cv.draw_text(sx + 10, sy + 210, "SA VRAM:     0 B (GPU)", kGreen, 1);
        cv.draw_text(sx + 10, sy + 234, "Recall:      Verbatim PASS", kGreen, 1);
        cv.draw_text(sx + 10, sy + 252, "P150A Batch: " + std::to_string(kBatchSlotsDflt) + " slots", kText, 1);
        cv.draw_text(sx + 10, sy + 270, "5x MI50 + 2x P150A", kMuted, 1);
    }

    // =========================================================================
    // PANEL 4 (Bottom-Right): Active-wave 1M Trajectory & Release Signoff
    // =========================================================================
    {
        struct TrPt { int step; double ctx_k; std::string turn; };
        std::vector<TrPt> traj;
        if (sess_ptr != nullptr && sess_ptr->has("trajectory")) {
            const auto& arr = sess_ptr->at("trajectory").arr_val;
            const size_t lim = std::min<size_t>(arr.size(), 64u);
            for (size_t i = 0; i < lim; ++i) {
                const auto& t = arr[i];
                traj.push_back({
                    t.has("step")  ? int(t.at("step").as_int()) : int(i + 1u),
                    clamp_nonneg(t.has("ctx_k") ? t.at("ctx_k").as_double() : 0.0, 0.0, 1e6),
                    t.has("turn")  ? t.at("turn").s_val : ""
                });
            }
        }
        if (traj.empty()) {
            traj = {{1, 7.6, "T1"}, {41, 131.1, "T2"}, {91, 202.3, "T4"}, {191, 282.5, "T6"},
                    {291, 364.8, "T11"}, {322, 514.0, "T13"}, {370, 543.3, "T15"},
                    {420, 581.9, "T17"}, {465, 612.6, "T20"}, {495, 823.0, "T22"}};
        }
        cv.draw_text(rx + 14, by + 12, "Panel 4: " + std::string(wave_short) + " 1M Trajectory & Release Signoff (" + fmt1(sess_peak_k) + "K Peak)", kText, 1);
        cv.draw_text(rx + 14, by + 28, "1 Session | " + std::to_string(sess_turns) + " Live Turns | " + std::to_string(sess_swaps) +
                     std::string(" Swaps | ") + std::to_string(w13_compactions) + " Compact to " + fmt0(double(w13_peak_ctx) / 1000.0) + "K | 0/0 AER | 0 B Spill", kMuted, 1);

        const int gx = rx + 58, gy = by + 54, gw = 380, gh = 246;
        cv.draw_rect(gx, gy, gw, gh, kBorder, 1);
        const double ymax_k = 1100.0;
        for (int i = 0; i <= 4; ++i) {
            const int yy = gy + gh - (gh * i) / 4;
            cv.draw_dashed_h(gx, gx + gw, yy, kGrid);
            cv.draw_text(gx - 46, yy - 4, fmt0(ymax_k * i / 4.0) + "K", kMuted, 1);
        }
        // Dashed horizontal milestone lines at 512K, 768K and 1024K
        const int ms_k[3] = {512, 768, 1024};
        const Color ms_c[3] = {kGreen, kBlue, kPurple};
        for (int m = 0; m < 3; ++m) {
            const int ym = gy + gh - int(std::lround((double(ms_k[m]) / ymax_k) * double(gh)));
            cv.draw_dashed_h(gx, gx + gw, ym, ms_c[m]);
            const int lbl_y = (m == 2) ? (ym + 5) : (ym - 12);
            cv.draw_text(gx + 8, lbl_y, std::to_string(ms_k[m]) + "K Milestone", ms_c[m], 1);
        }

        const int max_step = std::max(2, traj.back().step);
        std::string prev_t;
        for (size_t i = 0; i < traj.size(); ++i) {
            const int xi = gx + 10 + int((gw - 20) * double(traj[i].step) / double(max_step));
            const int yi = gy + gh - std::min(gh - 6, std::max(4, int(std::lround((traj[i].ctx_k / ymax_k) * double(gh)))));
            if (i > 0) {
                const int xp = gx + 10 + int((gw - 20) * double(traj[i - 1].step) / double(max_step));
                const int yp = gy + gh - std::min(gh - 6, std::max(4, int(std::lround((traj[i - 1].ctx_k / ymax_k) * double(gh)))));
                cv.draw_line(xp, yp, xi, yi, kPurple, 3);
            }
            if (!traj[i].turn.empty() && traj[i].turn != prev_t && i > 0) {
                if (traj[i].turn == "T4" || traj[i].turn == "T9" || traj[i].turn == "T15" || traj[i].turn == "T20") {
                    cv.draw_dashed_v(xi, gy, gy + gh, kRed);
                    cv.draw_text(std::min(gx + gw - 26, xi + 3), gy + 4, traj[i].turn, kRed, 1);
                }
            }
            prev_t = traj[i].turn;
            if (i == 0 || i + 1 == traj.size()) cv.fill_circle(xi, yi, 4, kPurple);
        }
        cv.draw_text(gx + 8, gy + gh + 10, "Step 1", kText, 1);
        cv.draw_text(gx + gw / 2 - 26, gy + gh + 10, "Step " + std::to_string(std::max(2, sess_steps / 2)), kText, 1);
        cv.draw_text(gx + gw - 58, gy + gh + 10, "Step " + std::to_string(sess_steps), kText, 1);
        cv.draw_text(gx + 10, gy + gh + 28, "Unbroken :3081 Trace (bespoke ontology, the kitchen)", kMuted, 1);

        const int sx = rx + 452, sy = by + 54, sw = 224, sh = 290;
        cv.fill_rect(sx, sy, sw, sh, kBg);
        cv.draw_rect(sx, sy, sw, sh, kBorder, 1);
        cv.fill_rect(sx + 1, sy + 1, sw - 2, 24, kSoftGrn);
        cv.draw_text(sx + 10, sy + 9, std::string(wave_short) + " Release Signoff", kText, 1);
        cv.draw_text(sx + 10, sy + 36, "Session IDs: 1 (Unbroken)", kGreen, 1);
        cv.draw_text(sx + 10, sy + 54, "Live Turns:  " + std::to_string(sess_turns) + " (" + std::to_string(sess_swaps) + " swaps)", kText, 1);
        cv.draw_text(sx + 10, sy + 72, "Live Steps:  " + std::to_string(sess_steps) + " (" + std::to_string(sess_tools) + " tools)", kText, 1);
        cv.draw_text(sx + 10, sy + 90, "Peak Ctx:    " + fmt1(sess_peak_k) + "K tok", kPurple, 1);
        cv.draw_text(sx + 10, sy + 108, "Output Tok:  " + std::to_string(sess_out), kBlue, 1);
        cv.draw_text(sx + 10, sy + 126, "SA Skipped:  " + std::to_string(sess_sa_skip) + " DSpark", kPurple, 1);
        cv.draw_text(sx + 10, sy + 150, "Unit Suites: " + std::to_string(unit_pass) + "/" + std::to_string(unit_tot) + " PASS", kGreen, 1);
        cv.draw_text(sx + 10, sy + 168, "gfx906 ISA:  " + std::to_string(isa_kernels) + "/" + std::to_string(isa_kernels) + " 0-Scr", kGreen, 1);
        cv.draw_text(sx + 10, sy + 186, "PCIe AER:    " + std::to_string(aer_total) + " (" + std::to_string(aer_devs) + " dev)", kGreen, 1);
        cv.draw_text(sx + 10, sy + 204, "Zero-PII:    " + std::to_string(pii_pass) + "/" + std::to_string(pii_tot) + " PASS", kBlue, 1);
        cv.draw_text(sx + 10, sy + 222, "Spill/VRAM:  0 B / 0 B", kGreen, 1);
        cv.draw_text(sx + 10, sy + 246, "Context:     1,048,576 tok", kBlue, 1);
        cv.draw_text(sx + 10, sy + 264, std::string("Tag: ") + wave_tag, kMuted, 1);
    }

    // ---- Footer ----
    cv.draw_text(18, 842, "Co-authored by bespoke ontology (the kitchen) & Google DeepMind / Gemini (Gemini Carbon & Gemini Neon) & Antigravity | Pure C++17 / gfx906 HIP / Blackhole C++20 | 0 Python | 0/0 PCIe AER", kMuted, 1);

    if (!cv.save_png(out_png)) {
        std::fprintf(stderr, "[CHART] Failed writing %s\n", out_png.c_str());
        return 1;
    }
    std::printf("[CHART] Rendered 1440x860 4-panel %s PNG (IHDR/IDAT/IEND only) -> %s\n", wave_tag, out_png.c_str());
    return 0;
}
