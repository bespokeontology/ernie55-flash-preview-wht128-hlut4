// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Exact ERNIE-5.5 R2L 3-Digit Grouping 201K ByteLevel BPE Tokenizer & Streaming Parser
// Zero Python anywhere. Uses libpcre2-8 with PCRE2_UTF | PCRE2_UCP for exact HF tokenizers parity.

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "ernie55_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ernie55 {

namespace {

struct ByteLevelTables {
    std::array<std::string, 256> byte_to_utf8;
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;

    ByteLevelTables() {
        std::array<bool, 256> direct{};
        for (int b = 33; b <= 126; ++b) direct[b] = true;
        for (int b = 161; b <= 172; ++b) direct[b] = true;
        for (int b = 174; b <= 255; ++b) direct[b] = true;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const uint32_t cp = direct[b] ? uint32_t(b) : uint32_t(256 + n++);
            std::string u8;
            if (cp <= 0x7Fu) {
                u8.push_back(char(cp));
            } else {
                u8.push_back(char(0xC0u | (cp >> 6)));
                u8.push_back(char(0x80u | (cp & 0x3Fu)));
            }
            byte_to_utf8[b] = u8;
            cp_to_byte[cp] = uint8_t(b);
        }
    }
};

const ByteLevelTables& byte_tables() {
    static const ByteLevelTables t;
    return t;
}

uint32_t next_utf8_cp(const std::string& s, size_t& i) {
    const size_t n = s.size();
    const uint8_t b0 = uint8_t(s[i]);
    if (b0 < 0x80u) { ++i; return b0; }
    if (b0 >= 0xC2u && b0 <= 0xDFu && i + 1 < n) {
        uint32_t cp = ((uint32_t(b0 & 0x1Fu) << 6) | uint32_t(uint8_t(s[i + 1]) & 0x3Fu));
        i += 2;
        return cp;
    }
    if (b0 >= 0xE0u && b0 <= 0xEFu && i + 2 < n) {
        uint32_t cp = (uint32_t(b0 & 0x0Fu) << 12) |
                      (uint32_t(uint8_t(s[i + 1]) & 0x3Fu) << 6) |
                      uint32_t(uint8_t(s[i + 2]) & 0x3Fu);
        i += 3;
        return cp;
    }
    if (b0 >= 0xF0u && b0 <= 0xF4u && i + 3 < n) {
        uint32_t cp = (uint32_t(b0 & 0x07u) << 18) |
                      (uint32_t(uint8_t(s[i + 1]) & 0x3Fu) << 12) |
                      (uint32_t(uint8_t(s[i + 2]) & 0x3Fu) << 6) |
                      uint32_t(uint8_t(s[i + 3]) & 0x3Fu);
        i += 4;
        return cp;
    }
    ++i;
    return 0xFFFDu;
}

std::string bytelevel_encode(const std::string& raw) {
    const auto& tbl = byte_tables();
    std::string out;
    out.reserve(raw.size() * 2u);
    for (unsigned char c : raw) out += tbl.byte_to_utf8[c];
    return out;
}

std::string bytelevel_decode(const std::string& tok) {
    const auto& tbl = byte_tables();
    std::string out;
    out.reserve(tok.size());
    size_t i = 0;
    while (i < tok.size()) {
        const size_t start = i;
        const uint32_t cp = next_utf8_cp(tok, i);
        auto it = tbl.cp_to_byte.find(cp);
        if (it != tbl.cp_to_byte.end()) out.push_back(char(it->second));
        else out.append(tok, start, i - start);
    }
    return out;
}

struct Pcre2Step {
    enum class Kind { kDigits, kSplitIsolated };
    Kind kind = Kind::kSplitIsolated;
    bool individual_digits = false;
    pcre2_code* re = nullptr;

    ~Pcre2Step() {
        if (re) pcre2_code_free(re);
    }
    Pcre2Step() = default;
    Pcre2Step(const Pcre2Step&) = delete;
    Pcre2Step& operator=(const Pcre2Step&) = delete;
    Pcre2Step(Pcre2Step&& o) noexcept
        : kind(o.kind), individual_digits(o.individual_digits), re(o.re) {
        o.re = nullptr;
    }
};

pcre2_code* compile_pcre2(const std::string& raw_pat) {
    // Translate Oniguruma/Rust-tokenizers `{N}+` repetition (e.g. `\p{Nd}{3}+` = 1+ groups of 3 digits)
    // to PCRE2 `(?:...)+` because PCRE2 interprets `{3}+` as a possessive quantifier matching exactly 3!
    std::string pat = raw_pat;
    const std::string needle = "\\p{Nd}{3}+";
    const std::string repl   = "(?:\\p{Nd}{3})+";
    size_t pos = 0;
    while ((pos = pat.find(needle, pos)) != std::string::npos) {
        pat.replace(pos, needle.size(), repl);
        pos += repl.size();
    }
    int errcode = 0;
    PCRE2_SIZE erroff = 0;
    pcre2_code* re = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(pat.c_str()), pat.size(),
        PCRE2_UTF | PCRE2_UCP, &errcode, &erroff, nullptr);
    if (!re) {
        PCRE2_UCHAR msg[256];
        pcre2_get_error_message(errcode, msg, sizeof(msg));
        throw std::runtime_error("PCRE2 compile failed for pattern [" + pat + "] at " +
                                 std::to_string(erroff) + ": " + reinterpret_cast<const char*>(msg));
    }
    (void)pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
    return re;
}

static uint64_t xxh64_rotl(uint64_t x, int r) noexcept {
    return (x << r) | (x >> (64 - r));
}

static uint64_t xxh64_round(uint64_t acc, uint64_t input) noexcept {
    constexpr uint64_t kP1 = 11400714785074694791ULL;
    constexpr uint64_t kP2 = 14029467366897019727ULL;
    acc += input * kP2;
    acc = xxh64_rotl(acc, 31);
    acc *= kP1;
    return acc;
}

static uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) noexcept {
    constexpr uint64_t kP1 = 11400714785074694791ULL;
    constexpr uint64_t kP4 = 9650029242287828579ULL;
    val = xxh64_round(0ULL, val);
    acc ^= val;
    acc = acc * kP1 + kP4;
    return acc;
}

static uint64_t xxh64_bytes(const void* data, size_t len, uint64_t seed = 0ULL) noexcept {
    constexpr uint64_t kP1 = 11400714785074694791ULL;
    constexpr uint64_t kP2 = 14029467366897019727ULL;
    constexpr uint64_t kP3 = 1609587929392839161ULL;
    constexpr uint64_t kP4 = 9650029242287828579ULL;
    constexpr uint64_t kP5 = 2870177450012600261ULL;

    const auto* p = static_cast<const uint8_t*>(data);
    const uint8_t* const end = p + len;
    uint64_t h = 0ULL;

    if (len >= 32u) {
        const uint8_t* const lim = end - 32;
        uint64_t v1 = seed + kP1 + kP2;
        uint64_t v2 = seed + kP2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - kP1;
        do {
            uint64_t k1, k2, k3, k4;
            std::memcpy(&k1, p + 0, 8);
            std::memcpy(&k2, p + 8, 8);
            std::memcpy(&k3, p + 16, 8);
            std::memcpy(&k4, p + 24, 8);
            v1 = xxh64_round(v1, k1);
            v2 = xxh64_round(v2, k2);
            v3 = xxh64_round(v3, k3);
            v4 = xxh64_round(v4, k4);
            p += 32;
        } while (p <= lim);
        h = xxh64_rotl(v1, 1) + xxh64_rotl(v2, 7) + xxh64_rotl(v3, 12) + xxh64_rotl(v4, 18);
        h = xxh64_merge_round(h, v1);
        h = xxh64_merge_round(h, v2);
        h = xxh64_merge_round(h, v3);
        h = xxh64_merge_round(h, v4);
    } else {
        h = seed + kP5;
    }

    h += uint64_t(len);
    while (p + 8 <= end) {
        uint64_t k1 = 0ULL;
        std::memcpy(&k1, p, 8);
        h ^= xxh64_round(0ULL, k1);
        h = xxh64_rotl(h, 27) * kP1 + kP4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t k1 = 0u;
        std::memcpy(&k1, p, 4);
        h ^= uint64_t(k1) * kP1;
        h = xxh64_rotl(h, 23) * kP2 + kP3;
        p += 4;
    }
    while (p < end) {
        h ^= uint64_t(*p) * kP5;
        h = xxh64_rotl(h, 11) * kP1;
        ++p;
    }
    h ^= h >> 33;
    h *= kP2;
    h ^= h >> 29;
    h *= kP3;
    h ^= h >> 32;
    return h;
}

void split_isolated_piece_md(pcre2_code* re, pcre2_match_data* md,
                             const char* data, size_t n, std::vector<std::string>& out) {
    if (n == 0u) return;
    PCRE2_SIZE off = 0;
    while (off < n) {
        int rc = pcre2_match(re, reinterpret_cast<PCRE2_SPTR>(data), n, off, 0, md, nullptr);
        if (rc == PCRE2_ERROR_NOMATCH) {
            out.emplace_back(data + off, n - off);
            break;
        }
        if (rc < 0) {
            // Invalid UTF-8 byte sequence encountered: isolate bad byte at error offset and continue
            const PCRE2_SIZE bad = pcre2_get_startchar(md);
            if (bad > off && bad < n) {
                split_isolated_piece_md(re, md, data + off, size_t(bad - off), out);
                out.emplace_back(data + bad, 1u);
                off = bad + 1;
                continue;
            }
            out.emplace_back(data + off, 1u);
            ++off;
            continue;
        }
        PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
        const PCRE2_SIZE m0 = ov[0];
        const PCRE2_SIZE m1 = ov[1];
        if (m0 > off) out.emplace_back(data + off, size_t(m0 - off));
        if (m1 > m0) {
            out.emplace_back(data + m0, size_t(m1 - m0));
            off = m1;
        } else {
            // Zero-length match guard: advance one UTF-8 character
            const uint8_t b0 = uint8_t(data[off]);
            size_t adv = 1u;
            if (b0 >= 0xC2u && b0 <= 0xDFu && off + 2u <= n) adv = 2u;
            else if (b0 >= 0xE0u && b0 <= 0xEFu && off + 3u <= n) adv = 3u;
            else if (b0 >= 0xF0u && b0 <= 0xF4u && off + 4u <= n) adv = 4u;
            out.emplace_back(data + off, adv);
            off += adv;
        }
    }
}

std::string lower_trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    std::string r = s.substr(a, b - a);
    for (char& c : r) c = char(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

} // namespace

// ---- Exact vLLM-ERNIE parser.py JSON Schema Coercion & Helper Functions ----
namespace {

const JsonValue* lookup_json_pointer(const JsonValue* root, const std::string& ref) {
    if (!root || ref.size() < 2 || ref[0] != '#' || ref[1] != '/') return nullptr;
    const JsonValue* node = root;
    size_t pos = 2;
    while (pos <= ref.size() && node != nullptr) {
        size_t slash = ref.find('/', pos);
        std::string part = (slash == std::string::npos) ? ref.substr(pos) : ref.substr(pos, slash - pos);
        // Unescape RFC 6901 ~1 -> / and ~0 -> ~
        std::string dec;
        for (size_t i = 0; i < part.size(); ++i) {
            if (part[i] == '~' && i + 1 < part.size()) {
                if (part[i + 1] == '1') { dec.push_back('/'); ++i; continue; }
                if (part[i + 1] == '0') { dec.push_back('~'); ++i; continue; }
            }
            dec.push_back(part[i]);
        }
        if (node->is_object()) {
            node = node->find(dec);
        } else if (node->is_array()) {
            bool all_dig = !dec.empty();
            for (char c : dec) if (!std::isdigit(static_cast<unsigned char>(c))) { all_dig = false; break; }
            if (!all_dig) return nullptr;
            size_t idx = size_t(std::strtoull(dec.c_str(), nullptr, 10));
            node = (idx < node->arr_val.size()) ? &node->arr_val[idx] : nullptr;
        } else {
            return nullptr;
        }
        if (slash == std::string::npos) break;
        pos = slash + 1;
    }
    return node;
}

void collect_schema_types(const JsonValue* schema, const JsonValue* root,
                          std::vector<std::string>& out_types,
                          std::vector<std::string>& seen_refs) {
    if (!schema || !schema->is_object()) return;
    if (const JsonValue* rv = schema->find("$ref"); rv && rv->is_string()) {
        const std::string& ref = rv->as_string();
        if (!ref.empty() && ref[0] == '#' &&
            std::find(seen_refs.begin(), seen_refs.end(), ref) == seen_refs.end()) {
            seen_refs.push_back(ref);
            collect_schema_types(lookup_json_pointer(root, ref), root, out_types, seen_refs);
        }
    }
    if (const JsonValue* tv = schema->find("type")) {
        if (tv->is_string()) {
            out_types.push_back(lower_trim(tv->as_string()));
        } else if (tv->is_array()) {
            for (const auto& item : tv->arr_val) {
                if (item.is_string()) out_types.push_back(lower_trim(item.as_string()));
            }
        }
    }
    static constexpr const char* kCombinators[3] = {"anyOf", "oneOf", "allOf"};
    for (const char* kw : kCombinators) {
        if (const JsonValue* arr = schema->find(kw); arr && arr->is_array()) {
            for (const auto& branch : arr->arr_val) {
                collect_schema_types(&branch, root, out_types, seen_refs);
            }
        }
    }
}

const JsonValue* find_schema_property(const JsonValue* schema, const std::string& key, const JsonValue* root,
                                      std::vector<std::string>& seen_refs) {
    if (!schema || !schema->is_object()) return nullptr;
    if (const JsonValue* rv = schema->find("$ref"); rv && rv->is_string()) {
        const std::string& ref = rv->as_string();
        if (!ref.empty() && ref[0] == '#' &&
            std::find(seen_refs.begin(), seen_refs.end(), ref) == seen_refs.end()) {
            seen_refs.push_back(ref);
            if (const JsonValue* p = find_schema_property(lookup_json_pointer(root, ref), key, root, seen_refs)) return p;
        }
    }
    if (const JsonValue* props = schema->find("properties"); props && props->is_object()) {
        if (const JsonValue* p = props->find(key)) return p;
    }
    static constexpr const char* kCombinators[3] = {"anyOf", "oneOf", "allOf"};
    for (const char* kw : kCombinators) {
        if (const JsonValue* arr = schema->find(kw); arr && arr->is_array()) {
            for (const auto& branch : arr->arr_val) {
                if (const JsonValue* p = find_schema_property(&branch, key, root, seen_refs)) return p;
            }
        }
    }
    return nullptr;
}

bool has_type(const std::vector<std::string>& types, const char* t) {
    return std::find(types.begin(), types.end(), t) != types.end();
}

std::string strip_ws_edges(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

} // namespace

bool is_ernie5_thinking_enabled(const JsonValue* chat_template_kwargs) noexcept {
    if (!chat_template_kwargs || !chat_template_kwargs->is_object()) return true;
    if (const JsonValue* et = chat_template_kwargs->find("enable_thinking")) {
        if (et->is_bool() && !et->as_bool()) return false;
        if (et->is_string()) {
            const std::string v = lower_trim(et->as_string());
            if (v == "false") return false;
        }
    }
    if (const JsonValue* opts = chat_template_kwargs->find("options"); opts && opts->is_object()) {
        if (const JsonValue* tm = opts->find("thinking_mode"); tm && tm->is_string()) {
            const std::string v = lower_trim(tm->as_string());
            if (v == "close" || v == "false") return false;
        }
    }
    return true;
}

JsonValue ernie5_coerce_param(const std::string& raw_value, const JsonValue* prop_schema, const JsonValue* root_schema) {
    const std::string lv = lower_trim(raw_value);
    if (lv == "null" || lv == "none" || lv == "nil") {
        return JsonValue::make_null();
    }

    std::vector<std::string> types;
    std::vector<std::string> seen_refs;
    if (prop_schema) collect_schema_types(prop_schema, root_schema, types, seen_refs);

    // Schema-guided coercion matching vLLM-ERNIE parser.py::coerce_param
    if (!types.empty()) {
        if (has_type(types, "boolean") || has_type(types, "bool")) {
            if (lv == "true" || lv == "yes" || lv == "on") return JsonValue::make_bool(true);
            if (lv == "false" || lv == "no" || lv == "off") return JsonValue::make_bool(false);
        }
        if (has_type(types, "integer") || has_type(types, "int")) {
            const std::string ts = strip_ws_edges(raw_value);
            char* endp = nullptr;
            const long long iv = std::strtoll(ts.c_str(), &endp, 10);
            if (!ts.empty() && endp && *endp == '\0') return JsonValue::make_int(int64_t(iv));
        }
        if (has_type(types, "number") || has_type(types, "float")) {
            const std::string ts = strip_ws_edges(raw_value);
            char* endp = nullptr;
            const double dv = std::strtod(ts.c_str(), &endp);
            if (!ts.empty() && endp && *endp == '\0') return JsonValue::make_double(dv);
        }
        if (has_type(types, "object") || has_type(types, "array")) {
            JsonValue parsed;
            if (JsonParser::try_parse(raw_value, parsed) && (parsed.is_object() || parsed.is_array())) {
                return parsed;
            }
        }
        if (has_type(types, "string") || has_type(types, "str")) {
            return JsonValue::make_string(raw_value);
        }
        // Fallback when schema exists (e.g. boolean="maybe" -> keep string)
        return JsonValue::make_string(raw_value);
    }

    // No schema provided: best-effort JSON parsing (still keeps "yes" as string unless bool schema)
    if (lv == "true") return JsonValue::make_bool(true);
    if (lv == "false") return JsonValue::make_bool(false);
    JsonValue parsed;
    if (!raw_value.empty() &&
        (raw_value.front() == '{' || raw_value.front() == '[' ||
         raw_value.front() == '-' || std::isdigit(static_cast<unsigned char>(raw_value.front()))) &&
        JsonParser::try_parse(raw_value, parsed)) {
        return parsed;
    }
    return JsonValue::make_string(raw_value);
}

bool Ernie55StreamParser::is_null_word_prefix(const std::string& partial_val) noexcept {
    if (partial_val.empty()) return true;
    const std::string lv = lower_trim(partial_val);
    static constexpr const char* kNullWords[3] = {"null", "none", "nil"};
    for (const char* w : kNullWords) {
        if (std::strncmp(w, lv.c_str(), lv.size()) == 0) return true;
    }
    return false;
}

// ---- Streaming ERNIE-5.5 <ernie:think> & <ernie:tool_call><invoke> Parser ----
Ernie55StreamParser::Ernie55StreamParser(bool thinking_enabled,
                                         const JsonValue* tools,
                                         const Ernie55SpecialTokenIds* special_ids) {
    reset(thinking_enabled, tools, special_ids);
}

void Ernie55StreamParser::reset(bool thinking_enabled,
                                const JsonValue* tools,
                                const Ernie55SpecialTokenIds* special_ids) {
    thinking_enabled_ = thinking_enabled;
    tools_ = tools;
    if (special_ids) specials_ = *special_ids;
    state_ = thinking_enabled ? State::kReasoning : State::kContent;
    tool_sub_ = ToolSubState::kSeekInvoke;
    buf_.clear();
    tool_buf_.clear();
    call_counter_ = 0;
    cur_call_idx_ = -1;
    cur_invoke_open_ = false;
    cur_param_count_ = 0;
    cur_fname_.clear();
    cur_root_schema_ = nullptr;
    cur_pname_.clear();
    cur_prop_schema_ = nullptr;
    param_buf_.clear();
    str_param_opened_ = false;
}

const JsonValue* Ernie55StreamParser::find_tool_params(const std::string& name) const {
    if (!tools_ || !tools_->is_array()) return nullptr;
    for (const auto& t : tools_->arr_val) {
        if (!t.is_object()) continue;
        const JsonValue* fn = t.find("function");
        const JsonValue* obj = (fn && fn->is_object()) ? fn : &t;
        const JsonValue* nm = obj->find("name");
        if (nm && nm->is_string() && nm->as_string() == name) {
            return obj->find("parameters");
        }
    }
    return nullptr;
}

bool Ernie55StreamParser::is_reasoning_end(const std::vector<int32_t>& token_ids) const noexcept {
    if (!thinking_enabled_) return true;
    // Scope strictly to the current turn (after the last <|ernie:im_start|> or <|ernie:im_end|>)!
    size_t start = 0;
    for (size_t i = token_ids.size(); i > 0; --i) {
        const int32_t id = token_ids[i - 1];
        if (id == specials_.im_start_id || id == specials_.im_end_id) {
            start = i;
            break;
        }
    }
    for (size_t i = start; i < token_ids.size(); ++i) {
        const int32_t id = token_ids[i];
        if (specials_.think_end_id >= 0 && id == specials_.think_end_id) {
            return true;
        }
    }
    return false;
}

void Ernie55StreamParser::parse_tool_block(const std::string& body, std::vector<ParsedToolCall>& out_calls) {
    size_t pos = 0;
    while (true) {
        size_t inv0 = body.find("<invoke", pos);
        if (inv0 == std::string::npos) break;
        size_t tag_end = body.find('>', inv0);
        if (tag_end == std::string::npos) break;
        size_t inv1 = body.find("</invoke>", tag_end);
        if (inv1 == std::string::npos) break;

        std::string header = body.substr(inv0, tag_end - inv0);
        std::string fname;
        size_t np = header.find("name=");
        if (np != std::string::npos) {
            np += 5;
            if (np < header.size() && (header[np] == '"' || header[np] == '\'')) {
                char q = header[np++];
                size_t ne = header.find(q, np);
                if (ne != std::string::npos) fname = header.substr(np, ne - np);
            }
        }

        const JsonValue* root_schema = find_tool_params(fname);
        JsonValue args = JsonValue::make_object();
        std::string inner = body.substr(tag_end + 1, inv1 - (tag_end + 1));
        size_t ppos = 0;
        while (true) {
            size_t p0 = inner.find("<parameter", ppos);
            if (p0 == std::string::npos) break;
            size_t pe = inner.find('>', p0);
            if (pe == std::string::npos) break;
            size_t p1 = inner.find("</parameter>", pe);
            if (p1 == std::string::npos) break;

            std::string phdr = inner.substr(p0, pe - p0);
            std::string pname;
            size_t k = phdr.find("name=");
            if (k != std::string::npos) {
                k += 5;
                if (k < phdr.size() && (phdr[k] == '"' || phdr[k] == '\'')) {
                    char q = phdr[k++];
                    size_t ke = phdr.find(q, k);
                    if (ke != std::string::npos) pname = phdr.substr(k, ke - k);
                }
            }
            std::string raw_val = inner.substr(pe + 1, p1 - (pe + 1));
            if (!pname.empty()) {
                std::vector<std::string> seen;
                const JsonValue* prop_schema = find_schema_property(root_schema, pname, root_schema, seen);
                args.set(pname, ernie5_coerce_param(raw_val, prop_schema, root_schema));
            }
            ppos = p1 + 12;
        }

        if (!fname.empty()) {
            ParsedToolCall tc;
            tc.id = "call_ernie55_" + std::to_string(++call_counter_);
            tc.name = fname;
            tc.arguments_json = args.dump();
            out_calls.push_back(std::move(tc));
            if (!parallel_tool_calls_) break;
        }
        pos = inv1 + 9;
    }
}

void Ernie55StreamParser::feed_token(int32_t token_id,
                                     const std::string& piece,
                                     std::string& out_reasoning_delta,
                                     std::string& out_content_delta,
                                     std::vector<ParsedToolCall>& out_tool_calls) {
    if (specials_.think_end_id >= 0 && token_id == specials_.think_end_id && state_ == State::kReasoning) {
        out_reasoning_delta += buf_;
        buf_.clear();
        state_ = State::kContent;
        return;
    }
    if (specials_.tool_call_start_id >= 0 && token_id == specials_.tool_call_start_id &&
        (state_ == State::kContent || (!thinking_enabled_ && state_ == State::kReasoning))) {
        if (state_ == State::kReasoning) out_reasoning_delta += buf_;
        else                             out_content_delta += buf_;
        buf_.clear();
        tool_buf_.clear();
        state_ = State::kToolCall;
        return;
    }
    if (specials_.tool_call_end_id >= 0 && token_id == specials_.tool_call_end_id && state_ == State::kToolCall) {
        tool_buf_ += buf_;
        buf_.clear();
        parse_tool_block(tool_buf_, out_tool_calls);
        tool_buf_.clear();
        state_ = State::kContent;
        return;
    }
    feed(piece, out_reasoning_delta, out_content_delta, out_tool_calls);
}

namespace {

// Never slice a streaming SSE delta mid-UTF-8 codepoint (prevents U+FFFD U+FFFD on multi-byte chars like em-dash '—').
size_t safe_utf8_prefix_len(const std::string& s, size_t max_emit) noexcept {
    const size_t n = std::min(max_emit, s.size());
    if (n == 0) return 0;
    size_t i = n;
    while (i > 0 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) --i;
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    size_t need = 1;
    if ((lead & 0x80u) == 0x00u)      need = 1;
    else if ((lead & 0xE0u) == 0xC0u) need = 2;
    else if ((lead & 0xF0u) == 0xE0u) need = 3;
    else if ((lead & 0xF8u) == 0xF0u) need = 4;
    else return n;
    return ((i - 1) + need > n) ? (i - 1) : n;
}

std::string json_escape_contents(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 8);
    for (unsigned char c : raw) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) { char h[8]; std::snprintf(h, sizeof(h), "\\u%04x", c); out += h; }
        else                out.push_back(char(c));
    }
    return out;
}

std::string extract_xml_attr_name(const std::string& tag_header) {
    size_t np = tag_header.find("name=");
    if (np == std::string::npos) return "";
    np += 5;
    if (np < tag_header.size() && (tag_header[np] == '"' || tag_header[np] == '\'')) {
        const char q = tag_header[np++];
        const size_t ne = tag_header.find(q, np);
        if (ne != std::string::npos) return tag_header.substr(np, ne - np);
    }
    return "";
}

bool is_strictly_string_schema(const JsonValue* prop_schema, const JsonValue* root_schema) {
    if (!prop_schema) return false;
    std::vector<std::string> types;
    std::vector<std::string> seen;
    collect_schema_types(prop_schema, root_schema, types, seen);
    if (types.empty()) return false;
    bool has_str = false;
    for (const auto& t : types) {
        if (t == "string" || t == "str") has_str = true;
        else if (t == "null") continue;
        else return false;
    }
    return has_str;
}

void push_tool_delta(std::vector<StreamToolCallDelta>& out, int idx,
                     const std::string& id, const std::string& name,
                     const std::string& arg_frag) {
    if (id.empty() && name.empty() && arg_frag.empty()) return;
    if (id.empty() && name.empty() && !out.empty() &&
        out.back().index == idx && out.back().id.empty() && out.back().name.empty()) {
        out.back().arguments_delta += arg_frag;
        return;
    }
    StreamToolCallDelta d;
    d.index = idx;
    d.id = id;
    d.name = name;
    d.arguments_delta = arg_frag;
    out.push_back(std::move(d));
}

} // namespace

void Ernie55StreamParser::feed(const std::string& piece,
                               std::string& out_reasoning_delta,
                               std::string& out_content_delta,
                               std::vector<ParsedToolCall>& out_tool_calls) {
    buf_ += piece;
    constexpr const char* kThinkEnd  = "</ernie:think>";
    constexpr const char* kToolStart = "<ernie:tool_call>";
    constexpr const char* kToolEnd   = "</ernie:tool_call>";

    while (!buf_.empty()) {
        if (state_ == State::kReasoning) {
            const size_t p_te = buf_.find(kThinkEnd);
            // Wave-6 Step-19 guard: while thinking_enabled_ == true and inside kReasoning,
            // keep <ernie:tool_call> literals/examples inside reasoning until </ernie:think>!
            const size_t p_ts = thinking_enabled_ ? std::string::npos : buf_.find(kToolStart);
            if (p_te != std::string::npos && (p_ts == std::string::npos || p_te <= p_ts)) {
                out_reasoning_delta.append(buf_, 0, p_te);
                buf_.erase(0, p_te + std::strlen(kThinkEnd));
                state_ = State::kContent;
                continue;
            }
            if (p_ts != std::string::npos) {
                out_reasoning_delta.append(buf_, 0, p_ts);
                buf_.erase(0, p_ts + std::strlen(kToolStart));
                tool_buf_.clear();
                state_ = State::kToolCall;
                continue;
            }
            const size_t max_tag = thinking_enabled_
                ? std::strlen(kThinkEnd)
                : std::max(std::strlen(kThinkEnd), std::strlen(kToolStart));
            const size_t keep = std::min(buf_.size(), max_tag - 1);
            if (buf_.size() > keep) {
                const size_t emit = safe_utf8_prefix_len(buf_, buf_.size() - keep);
                if (emit > 0) {
                    out_reasoning_delta.append(buf_, 0, emit);
                    buf_.erase(0, emit);
                }
            }
            break;
        } else if (state_ == State::kContent) {
            const size_t p = buf_.find(kToolStart);
            if (p != std::string::npos) {
                out_content_delta.append(buf_, 0, p);
                buf_.erase(0, p + std::strlen(kToolStart));
                tool_buf_.clear();
                state_ = State::kToolCall;
                continue;
            }
            const size_t keep = std::min(buf_.size(), std::strlen(kToolStart) - 1);
            if (buf_.size() > keep) {
                const size_t emit = safe_utf8_prefix_len(buf_, buf_.size() - keep);
                if (emit > 0) {
                    out_content_delta.append(buf_, 0, emit);
                    buf_.erase(0, emit);
                }
            }
            break;
        } else {
            const size_t p = buf_.find(kToolEnd);
            if (p != std::string::npos) {
                tool_buf_.append(buf_, 0, p);
                parse_tool_block(tool_buf_, out_tool_calls);
                tool_buf_.clear();
                buf_.erase(0, p + std::strlen(kToolEnd));
                state_ = State::kContent;
                continue;
            }
            const size_t keep = std::min(buf_.size(), std::strlen(kToolEnd) - 1);
            if (buf_.size() > keep) {
                tool_buf_.append(buf_, 0, buf_.size() - keep);
                buf_.erase(0, buf_.size() - keep);
            }
            break;
        }
    }
}

void Ernie55StreamParser::flush(std::string& out_reasoning_delta,
                                std::string& out_content_delta,
                                std::vector<ParsedToolCall>& out_tool_calls) {
    if (state_ == State::kReasoning) out_reasoning_delta += buf_;
    else if (state_ == State::kContent) out_content_delta += buf_;
    else {
        tool_buf_ += buf_;
        parse_tool_block(tool_buf_, out_tool_calls);
        tool_buf_.clear();
    }
    buf_.clear();
}

void Ernie55StreamParser::process_tool_stream(std::vector<StreamToolCallDelta>& out_deltas, bool eof_block) {
    constexpr const char* kToolEnd  = "</ernie:tool_call>";
    constexpr const char* kParamEnd = "</parameter>";
    constexpr size_t kParamEndLen   = 12u; // strlen("</parameter>")

    while (state_ == State::kToolCall) {
        if (tool_sub_ == ToolSubState::kSeekInvoke) {
            const size_t p_inv = buf_.find("<invoke");
            const size_t p_end = buf_.find(kToolEnd);
            if (p_end != std::string::npos && (p_inv == std::string::npos || p_end < p_inv)) {
                buf_.erase(0, p_end + std::strlen(kToolEnd));
                state_ = State::kContent;
                return;
            }
            if (p_inv != std::string::npos) {
                const size_t p_gt = buf_.find('>', p_inv);
                if (p_gt == std::string::npos) {
                    if (p_inv > 0) buf_.erase(0, p_inv);
                    break;
                }
                const std::string fname = extract_xml_attr_name(buf_.substr(p_inv, p_gt - p_inv));
                buf_.erase(0, p_gt + 1);
                if (!fname.empty()) {
                    cur_call_idx_ = call_counter_++;
                    cur_fname_ = fname;
                    cur_root_schema_ = find_tool_params(fname);
                    cur_param_count_ = 0;
                    cur_invoke_open_ = true;
                    push_tool_delta(out_deltas, cur_call_idx_,
                                    "call_ernie55_" + std::to_string(cur_call_idx_ + 1),
                                    cur_fname_, "{");
                    tool_sub_ = ToolSubState::kSeekParam;
                }
                continue;
            }
            const size_t keep = std::min(buf_.size(), std::strlen(kToolEnd) - 1u);
            if (buf_.size() > keep) buf_.erase(0, buf_.size() - keep);
            break;
        } else if (tool_sub_ == ToolSubState::kSeekParam) {
            const size_t p_par = buf_.find("<parameter");
            const size_t p_cie = buf_.find("</invoke>");
            const size_t p_end = buf_.find(kToolEnd);
            size_t first = std::min(p_par, std::min(p_cie, p_end));
            if (first == std::string::npos) {
                const size_t keep = std::min(buf_.size(), std::strlen(kToolEnd) - 1u);
                if (buf_.size() > keep) buf_.erase(0, buf_.size() - keep);
                break;
            }
            if (first == p_par) {
                const size_t p_gt = buf_.find('>', p_par);
                if (p_gt == std::string::npos) {
                    if (p_par > 0) buf_.erase(0, p_par);
                    break;
                }
                cur_pname_ = extract_xml_attr_name(buf_.substr(p_par, p_gt - p_par));
                buf_.erase(0, p_gt + 1);
                std::vector<std::string> seen;
                cur_prop_schema_ = cur_pname_.empty()
                    ? nullptr
                    : find_schema_property(cur_root_schema_, cur_pname_, cur_root_schema_, seen);
                param_buf_.clear();
                str_param_opened_ = false;
                if (!cur_pname_.empty() && is_strictly_string_schema(cur_prop_schema_, cur_root_schema_)) {
                    tool_sub_ = ToolSubState::kStreamStringVal;
                } else {
                    tool_sub_ = ToolSubState::kBufferParamVal;
                }
                continue;
            }
            if (first == p_cie) {
                buf_.erase(0, p_cie + 9u);
                if (cur_invoke_open_) {
                    push_tool_delta(out_deltas, cur_call_idx_, "", "", "}");
                    cur_invoke_open_ = false;
                }
                tool_sub_ = parallel_tool_calls_ ? ToolSubState::kSeekInvoke : ToolSubState::kSkipRest;
                continue;
            }
            // first == p_end
            buf_.erase(0, p_end + std::strlen(kToolEnd));
            if (cur_invoke_open_) {
                push_tool_delta(out_deltas, cur_call_idx_, "", "", "}");
                cur_invoke_open_ = false;
            }
            state_ = State::kContent;
            return;
        } else if (tool_sub_ == ToolSubState::kStreamStringVal) {
            const size_t p_pe = buf_.find(kParamEnd);
            if (p_pe != std::string::npos) {
                if (!str_param_opened_) {
                    param_buf_.append(buf_, 0, p_pe);
                    buf_.erase(0, p_pe + kParamEndLen);
                    const std::string dumped = ernie5_coerce_param(param_buf_, cur_prop_schema_, cur_root_schema_).dump();
                    std::string frag = (cur_param_count_++ > 0 ? "," : "");
                    frag += JsonValue::make_string(cur_pname_).dump();
                    frag += ":";
                    frag += dumped;
                    push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
                } else {
                    const std::string rem = buf_.substr(0, p_pe);
                    buf_.erase(0, p_pe + kParamEndLen);
                    std::string frag = json_escape_contents(rem);
                    frag.push_back('"');
                    push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
                }
                param_buf_.clear();
                str_param_opened_ = false;
                tool_sub_ = ToolSubState::kSeekParam;
                continue;
            }
            // Hold back 12 bytes (strlen("</parameter>")) + UTF-8 safe prefix so </parameter> never leaks!
            if (buf_.size() > kParamEndLen) {
                const size_t emit = safe_utf8_prefix_len(buf_, buf_.size() - kParamEndLen);
                if (emit > 0) {
                    if (!str_param_opened_) {
                        param_buf_.append(buf_, 0, emit);
                        buf_.erase(0, emit);
                        if (!is_null_word_prefix(param_buf_)) {
                            std::string frag = (cur_param_count_++ > 0 ? "," : "");
                            frag += JsonValue::make_string(cur_pname_).dump();
                            frag += ":\"";
                            frag += json_escape_contents(param_buf_);
                            push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
                            param_buf_.clear();
                            str_param_opened_ = true;
                        }
                    } else {
                        const std::string chunk = buf_.substr(0, emit);
                        buf_.erase(0, emit);
                        push_tool_delta(out_deltas, cur_call_idx_, "", "", json_escape_contents(chunk));
                    }
                }
            }
            break;
        } else if (tool_sub_ == ToolSubState::kBufferParamVal) {
            const size_t p_pe = buf_.find(kParamEnd);
            if (p_pe != std::string::npos) {
                param_buf_.append(buf_, 0, p_pe);
                buf_.erase(0, p_pe + kParamEndLen);
                if (!cur_pname_.empty()) {
                    const std::string dumped = ernie5_coerce_param(param_buf_, cur_prop_schema_, cur_root_schema_).dump();
                    std::string frag = (cur_param_count_++ > 0 ? "," : "");
                    frag += JsonValue::make_string(cur_pname_).dump();
                    frag += ":";
                    frag += dumped;
                    push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
                }
                param_buf_.clear();
                tool_sub_ = ToolSubState::kSeekParam;
                continue;
            }
            if (buf_.size() > kParamEndLen) {
                const size_t move_n = buf_.size() - kParamEndLen;
                param_buf_.append(buf_, 0, move_n);
                buf_.erase(0, move_n);
            }
            break;
        } else { // kSkipRest
            const size_t p_end = buf_.find(kToolEnd);
            if (p_end != std::string::npos) {
                buf_.erase(0, p_end + std::strlen(kToolEnd));
                state_ = State::kContent;
                return;
            }
            const size_t keep = std::min(buf_.size(), std::strlen(kToolEnd) - 1u);
            if (buf_.size() > keep) buf_.erase(0, buf_.size() - keep);
            break;
        }
    }

    if (eof_block && state_ == State::kToolCall) {
        if (tool_sub_ == ToolSubState::kStreamStringVal) {
            if (str_param_opened_) {
                std::string frag = json_escape_contents(buf_);
                frag.push_back('"');
                push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
                str_param_opened_ = false;
            } else if (!cur_pname_.empty() && (!param_buf_.empty() || !buf_.empty())) {
                param_buf_ += buf_;
                const std::string dumped = ernie5_coerce_param(param_buf_, cur_prop_schema_, cur_root_schema_).dump();
                std::string frag = (cur_param_count_++ > 0 ? "," : "");
                frag += JsonValue::make_string(cur_pname_).dump();
                frag += ":";
                frag += dumped;
                push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
            }
            param_buf_.clear();
        } else if (tool_sub_ == ToolSubState::kBufferParamVal && !cur_pname_.empty()) {
            param_buf_ += buf_;
            if (!param_buf_.empty()) {
                const std::string dumped = ernie5_coerce_param(param_buf_, cur_prop_schema_, cur_root_schema_).dump();
                std::string frag = (cur_param_count_++ > 0 ? "," : "");
                frag += JsonValue::make_string(cur_pname_).dump();
                frag += ":";
                frag += dumped;
                push_tool_delta(out_deltas, cur_call_idx_, "", "", frag);
            }
            param_buf_.clear();
        }
        if (cur_invoke_open_) {
            push_tool_delta(out_deltas, cur_call_idx_, "", "", "}");
            cur_invoke_open_ = false;
        }
        buf_.clear();
        tool_sub_ = ToolSubState::kSeekInvoke;
        state_ = State::kContent;
    }
}

void Ernie55StreamParser::feed_token_stream(int32_t token_id,
                                            const std::string& piece,
                                            std::string& out_reasoning_delta,
                                            std::string& out_content_delta,
                                            std::vector<StreamToolCallDelta>& out_tool_deltas) {
    if (specials_.think_end_id >= 0 && token_id == specials_.think_end_id && state_ == State::kReasoning) {
        out_reasoning_delta += buf_;
        buf_.clear();
        state_ = State::kContent;
        return;
    }
    if (specials_.tool_call_start_id >= 0 && token_id == specials_.tool_call_start_id &&
        (state_ == State::kContent || (!thinking_enabled_ && state_ == State::kReasoning))) {
        if (state_ == State::kReasoning) out_reasoning_delta += buf_;
        else                             out_content_delta += buf_;
        buf_.clear();
        tool_sub_ = ToolSubState::kSeekInvoke;
        cur_invoke_open_ = false;
        state_ = State::kToolCall;
        return;
    }
    if (specials_.tool_call_end_id >= 0 && token_id == specials_.tool_call_end_id && state_ == State::kToolCall) {
        process_tool_stream(out_tool_deltas, true);
        return;
    }
    feed_stream(piece, out_reasoning_delta, out_content_delta, out_tool_deltas);
}

void Ernie55StreamParser::feed_stream(const std::string& piece,
                                      std::string& out_reasoning_delta,
                                      std::string& out_content_delta,
                                      std::vector<StreamToolCallDelta>& out_tool_deltas) {
    buf_ += piece;
    constexpr const char* kThinkEnd  = "</ernie:think>";
    constexpr const char* kToolStart = "<ernie:tool_call>";

    while (!buf_.empty()) {
        if (state_ == State::kReasoning) {
            const size_t p_te = buf_.find(kThinkEnd);
            const size_t p_ts = thinking_enabled_ ? std::string::npos : buf_.find(kToolStart);
            if (p_te != std::string::npos && (p_ts == std::string::npos || p_te <= p_ts)) {
                out_reasoning_delta.append(buf_, 0, p_te);
                buf_.erase(0, p_te + std::strlen(kThinkEnd));
                state_ = State::kContent;
                continue;
            }
            if (p_ts != std::string::npos) {
                out_reasoning_delta.append(buf_, 0, p_ts);
                buf_.erase(0, p_ts + std::strlen(kToolStart));
                tool_sub_ = ToolSubState::kSeekInvoke;
                cur_invoke_open_ = false;
                state_ = State::kToolCall;
                continue;
            }
            const size_t max_tag = thinking_enabled_
                ? std::strlen(kThinkEnd)
                : std::max(std::strlen(kThinkEnd), std::strlen(kToolStart));
            const size_t keep = std::min(buf_.size(), max_tag - 1);
            if (buf_.size() > keep) {
                const size_t emit = safe_utf8_prefix_len(buf_, buf_.size() - keep);
                if (emit > 0) {
                    out_reasoning_delta.append(buf_, 0, emit);
                    buf_.erase(0, emit);
                }
            }
            break;
        } else if (state_ == State::kContent) {
            const size_t p = buf_.find(kToolStart);
            if (p != std::string::npos) {
                out_content_delta.append(buf_, 0, p);
                buf_.erase(0, p + std::strlen(kToolStart));
                tool_sub_ = ToolSubState::kSeekInvoke;
                cur_invoke_open_ = false;
                state_ = State::kToolCall;
                continue;
            }
            const size_t keep = std::min(buf_.size(), std::strlen(kToolStart) - 1);
            if (buf_.size() > keep) {
                const size_t emit = safe_utf8_prefix_len(buf_, buf_.size() - keep);
                if (emit > 0) {
                    out_content_delta.append(buf_, 0, emit);
                    buf_.erase(0, emit);
                }
            }
            break;
        } else {
            const size_t prev_sz = buf_.size();
            const ToolSubState prev_sub = tool_sub_;
            process_tool_stream(out_tool_deltas, false);
            if (state_ == State::kToolCall && buf_.size() == prev_sz && tool_sub_ == prev_sub) break;
        }
    }
}

void Ernie55StreamParser::flush_stream(std::string& out_reasoning_delta,
                                       std::string& out_content_delta,
                                       std::vector<StreamToolCallDelta>& out_tool_deltas) {
    if (state_ == State::kReasoning) {
        out_reasoning_delta += buf_;
        buf_.clear();
    } else if (state_ == State::kContent) {
        out_content_delta += buf_;
        buf_.clear();
    } else {
        process_tool_stream(out_tool_deltas, true);
    }
}

Ernie55StreamParser::CompleteResult Ernie55StreamParser::parse_complete(
    const std::string& text,
    const JsonValue* tools,
    bool thinking_enabled,
    bool parallel_tool_calls)
{
    Ernie55StreamParser p(thinking_enabled, tools, nullptr);
    p.parallel_tool_calls_ = parallel_tool_calls;
    CompleteResult res;
    p.feed(text, res.reasoning, res.content, res.tool_calls);
    p.flush(res.reasoning, res.content, res.tool_calls);
    if (!res.tool_calls.empty()) res.content = strip_ws_edges(res.content);
    return res;
}

// ---- Ernie55Tokenizer Implementation ----
struct Ernie55Tokenizer::Impl {
    static constexpr size_t kMaxCacheBytes      = 64ULL * 1024ULL * 1024ULL;
    static constexpr size_t kMaxCacheEntryBytes = 16ULL * 1024ULL * 1024ULL;
    static constexpr size_t kOmpMinBytes        = 32768u;

    struct PlainCacheEntry {
        uint64_t hash = 0ULL;
        std::string text;
        std::vector<int32_t> tokens;
        size_t bytes = 0u;
    };

    bool is_loaded = false;
    bool has_chat_tpl = false;
    std::vector<std::string> id_to_token;
    std::vector<uint8_t> id_is_special;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> added_token_to_id;
    std::vector<std::pair<std::string, int32_t>> added_sorted_desc;
    std::unordered_map<std::string, uint32_t> merge_ranks;
    std::vector<Pcre2Step> pretok_steps;
    std::vector<int32_t> eos_ids = {2, 8};

    mutable std::mutex cache_mtx;
    mutable std::list<PlainCacheEntry> lru_list;
    mutable std::unordered_multimap<uint64_t, std::list<PlainCacheEntry>::iterator> lru_index;
    mutable size_t cache_bytes = 0u;
    mutable std::atomic<uint64_t> cache_hits{0ULL};
    mutable std::atomic<uint64_t> cache_misses{0ULL};

    std::vector<std::string> run_pretokenizer_slice(const char* data, size_t len) const {
        std::vector<std::string> cur;
        if (len == 0u) return cur;
        cur.emplace_back(data, len);
        std::vector<std::string> next;
        for (const auto& step : pretok_steps) {
            next.clear();
            pcre2_match_data* md = pcre2_match_data_create_from_pattern(step.re, nullptr);
            for (const auto& piece : cur) {
                split_isolated_piece_md(step.re, md, piece.data(), piece.size(), next);
            }
            pcre2_match_data_free(md);
            cur.swap(next);
        }
        return cur;
    }

    std::vector<std::string> run_pretokenizer(const std::string& text) const {
        return run_pretokenizer_slice(text.data(), text.size());
    }

    void bpe_encode_piece(const std::string& raw_piece, std::vector<int32_t>& out) const {
        if (raw_piece.empty()) return;
        const std::string bl = bytelevel_encode(raw_piece);
        // Fast path: whole word in vocab
        auto direct_it = token_to_id.find(bl);
        if (direct_it != token_to_id.end()) {
            out.push_back(direct_it->second);
            return;
        }
        // Split into UTF-8 symbols
        std::vector<std::string> syms;
        size_t i = 0;
        while (i < bl.size()) {
            const size_t s0 = i;
            (void)next_utf8_cp(bl, i);
            syms.push_back(bl.substr(s0, i - s0));
        }
        while (syms.size() >= 2) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            size_t best_pos = std::string::npos;
            for (size_t k = 0; k + 1 < syms.size(); ++k) {
                std::string key;
                key.reserve(syms[k].size() + 1 + syms[k + 1].size());
                key += syms[k];
                key.push_back('\x01');
                key += syms[k + 1];
                auto it = merge_ranks.find(key);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = k;
                }
            }
            if (best_pos == std::string::npos) break;
            syms[best_pos] += syms[best_pos + 1];
            syms.erase(syms.begin() + ptrdiff_t(best_pos + 1));
        }
        for (const auto& sym : syms) {
            auto it = token_to_id.find(sym);
            if (it != token_to_id.end()) out.push_back(it->second);
        }
    }

    void encode_plain_serial(const char* data, size_t len, std::vector<int32_t>& out) const {
        if (len == 0u) return;
        for (const auto& piece : run_pretokenizer_slice(data, len)) {
            bpe_encode_piece(piece, out);
        }
    }

    // Cut K2.3: Bit-exact OpenMP parallel pretokenizer + BPE on >= 32 KiB buffers.
    // Cuts land strictly at byte offsets `pos` where data[pos-1] == '\n' and
    // 0x21 <= uint8_t(data[pos]) <= 0x7E (printable non-space ASCII), which is
    // provably non-straddling across all 6 pre_tokenizer stages in tokenizer.json.
    void encode_plain_parallel(const char* data, size_t len, std::vector<int32_t>& out) const {
        if (len == 0u) return;
#ifdef _OPENMP
        int n_thr = std::min(omp_get_max_threads(), 8);
        if (len >= kOmpMinBytes && n_thr > 1 && !omp_in_parallel()) {
            const int max_chunks = std::min(n_thr * 2, int(len / 16384u));
            if (max_chunks >= 2) {
                std::vector<size_t> cuts;
                cuts.reserve(size_t(max_chunks) + 1u);
                cuts.push_back(0u);
                const size_t stride = len / size_t(max_chunks);
                for (int c = 1; c < max_chunks; ++c) {
                    const size_t target = size_t(c) * stride;
                    const size_t lo = std::max(cuts.back() + 1u, (target > stride / 2u) ? (target - stride / 2u) : 1u);
                    const size_t hi = std::min(len - 1u, target + stride / 2u);
                    size_t best = 0u;
                    for (size_t p = target; p <= hi; ++p) {
                        const unsigned char nxt = static_cast<unsigned char>(data[p]);
                        if (data[p - 1u] == '\n' && nxt >= 0x21u && nxt <= 0x7Eu) {
                            best = p;
                            break;
                        }
                    }
                    if (best == 0u) {
                        for (size_t p = target; p > lo; --p) {
                            const unsigned char nxt = static_cast<unsigned char>(data[p]);
                            if (data[p - 1u] == '\n' && nxt >= 0x21u && nxt <= 0x7Eu) {
                                best = p;
                                break;
                            }
                        }
                    }
                    if (best > cuts.back() && best < len) cuts.push_back(best);
                }
                cuts.push_back(len);

                const int n_chunks = int(cuts.size()) - 1;
                if (n_chunks >= 2) {
                    std::vector<std::vector<int32_t>> partials(static_cast<size_t>(n_chunks));
                    const int use_thr = std::min(n_thr, n_chunks);
                    #pragma omp parallel for num_threads(use_thr) schedule(dynamic, 1)
                    for (int ci = 0; ci < n_chunks; ++ci) {
                        const size_t b0 = cuts[size_t(ci)];
                        const size_t b1 = cuts[size_t(ci) + 1u];
                        auto& dst = partials[size_t(ci)];
                        dst.reserve((b1 - b0) / 3u + 16u);
                        encode_plain_serial(data + b0, b1 - b0, dst);
                    }
                    size_t tot = 0u;
                    for (const auto& pv : partials) tot += pv.size();
                    out.reserve(out.size() + tot);
                    for (const auto& pv : partials) {
                        out.insert(out.end(), pv.begin(), pv.end());
                    }
                    return;
                }
            }
        }
#endif
        encode_plain_serial(data, len, out);
    }

    // Cut K2.3: Thread-safe <= 64 MiB LRU cache keyed by XXH64 + exact byte length + memcmp.
    void encode_plain_cached(const std::string& plain_buf, std::vector<int32_t>& out) const {
        if (plain_buf.empty()) return;
        const uint64_t h = xxh64_bytes(plain_buf.data(), plain_buf.size());
        {
            std::lock_guard<std::mutex> lk(cache_mtx);
            auto range = lru_index.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                auto lit = it->second;
                if (lit->text.size() == plain_buf.size() &&
                    std::memcmp(lit->text.data(), plain_buf.data(), plain_buf.size()) == 0) {
                    lru_list.splice(lru_list.begin(), lru_list, lit);
                    out.insert(out.end(), lit->tokens.begin(), lit->tokens.end());
                    cache_hits.fetch_add(1ULL, std::memory_order_relaxed);
                    return;
                }
            }
        }
        cache_misses.fetch_add(1ULL, std::memory_order_relaxed);
        std::vector<int32_t> toks;
        toks.reserve(plain_buf.size() / 3u + 16u);
        encode_plain_parallel(plain_buf.data(), plain_buf.size(), toks);
        out.insert(out.end(), toks.begin(), toks.end());

        const size_t entry_bytes = plain_buf.size() + toks.size() * sizeof(int32_t) + sizeof(PlainCacheEntry) + 64u;
        if (entry_bytes > kMaxCacheEntryBytes) return;

        std::lock_guard<std::mutex> lk(cache_mtx);
        // Re-check after unlock to avoid duplicate entries under concurrency
        auto range = lru_index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            auto lit = it->second;
            if (lit->text.size() == plain_buf.size() &&
                std::memcmp(lit->text.data(), plain_buf.data(), plain_buf.size()) == 0) {
                lru_list.splice(lru_list.begin(), lru_list, lit);
                return;
            }
        }
        while (!lru_list.empty() && cache_bytes + entry_bytes > kMaxCacheBytes) {
            auto last = std::prev(lru_list.end());
            const uint64_t eh = last->hash;
            auto er = lru_index.equal_range(eh);
            for (auto it = er.first; it != er.second; ++it) {
                if (it->second == last) {
                    lru_index.erase(it);
                    break;
                }
            }
            cache_bytes -= last->bytes;
            lru_list.erase(last);
        }
        if (cache_bytes + entry_bytes <= kMaxCacheBytes) {
            lru_list.push_front(PlainCacheEntry{h, plain_buf, std::move(toks), entry_bytes});
            lru_index.emplace(h, lru_list.begin());
            cache_bytes += entry_bytes;
        }
    }
};

Ernie55Tokenizer::Ernie55Tokenizer() : impl_(std::make_unique<Impl>()) {}
Ernie55Tokenizer::~Ernie55Tokenizer() = default;

bool Ernie55Tokenizer::loaded() const noexcept { return impl_->is_loaded; }
int Ernie55Tokenizer::vocab_size() const noexcept { return int(impl_->id_to_token.size()); }
bool Ernie55Tokenizer::has_chat_template() const noexcept { return impl_->has_chat_tpl; }

bool Ernie55Tokenizer::is_eos_token(int32_t id) const noexcept {
    for (int32_t e : impl_->eos_ids) if (e == id) return true;
    return false;
}

void Ernie55Tokenizer::load(const std::string& tokenizer_json_path,
                            const std::string& tokenizer_config_path) {
    std::ifstream ifs(tokenizer_json_path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Ernie55Tokenizer: cannot open " + tokenizer_json_path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    JsonValue root = JsonParser::parse(ss.str());

    // 1. Load vocab
    const auto& model = root.get("model");
    const auto& vocab = model.get("vocab");
    int max_id = -1;
    for (const auto& kv : vocab.obj_val) {
        const int id = int(kv.second.as_int());
        if (id > max_id) max_id = id;
    }
    if (root.has("added_tokens") && root.get("added_tokens").is_array()) {
        for (const auto& at : root.get("added_tokens").arr_val) {
            const int id = int(at.get("id").as_int());
            if (id > max_id) max_id = id;
        }
    }
    if (max_id < 0) throw std::runtime_error("Ernie55Tokenizer: empty vocab");

    impl_->id_to_token.assign(size_t(max_id + 1), "");
    impl_->id_is_special.assign(size_t(max_id + 1), 0u);
    impl_->token_to_id.clear();
    impl_->token_to_id.reserve(size_t(max_id + 1) * 2u);
    for (const auto& kv : vocab.obj_val) {
        const int32_t id = int32_t(kv.second.as_int());
        impl_->id_to_token[size_t(id)] = kv.first;
        impl_->token_to_id.emplace(kv.first, id);
    }

    // 2. Load added_tokens (handles pad_token_id=0 cleanly!)
    impl_->added_token_to_id.clear();
    impl_->added_sorted_desc.clear();
    if (root.has("added_tokens") && root.get("added_tokens").is_array()) {
        for (const auto& at : root.get("added_tokens").arr_val) {
            const int32_t id = int32_t(at.get("id").as_int());
            const std::string content = at.get("content").as_string();
            const bool special = at.has("special") && at.get("special").as_bool();
            impl_->id_to_token[size_t(id)] = content;
            impl_->id_is_special[size_t(id)] = special ? 1u : 0u;
            impl_->token_to_id[content] = id;
            impl_->added_token_to_id[content] = id;
            impl_->added_sorted_desc.emplace_back(content, id);
        }
        std::sort(impl_->added_sorted_desc.begin(), impl_->added_sorted_desc.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    auto find_added = [&](const char* tok, int32_t def) -> int32_t {
        auto it = impl_->added_token_to_id.find(tok);
        return (it != impl_->added_token_to_id.end()) ? it->second : def;
    };
    pad_id_      = find_added("<|ERNIE_PAD|>", 0);
    bos_id_      = find_added("<|ERNIE_BEGIN|>", 1);
    eos_id_      = find_added("<|ERNIE_MASK|>", 2);
    im_start_id_ = find_added("<|ernie:im_start|>", 7);
    im_end_id_   = find_added("<|ernie:im_end|>", 8);
    impl_->eos_ids = {eos_id_, im_end_id_};

    // 3. Load merges (supports both "a b" string format and ["a", "b"] array format!)
    impl_->merge_ranks.clear();
    const auto& merges = model.get("merges").arr_val;
    impl_->merge_ranks.reserve(merges.size() * 2u);
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        std::string key;
        if (merges[rank].is_string()) {
            const std::string& s = merges[rank].as_string();
            size_t sp = s.find(' ');
            if (sp == std::string::npos) continue;
            key.reserve(s.size());
            key.append(s, 0, sp);
            key.push_back('\x01');
            key.append(s, sp + 1, std::string::npos);
        } else if (merges[rank].is_array() && merges[rank].arr_val.size() == 2) {
            const std::string& a = merges[rank].arr_val[0].as_string();
            const std::string& b = merges[rank].arr_val[1].as_string();
            key.reserve(a.size() + 1 + b.size());
            key += a;
            key.push_back('\x01');
            key += b;
        }
        if (!key.empty()) impl_->merge_ranks.emplace(std::move(key), uint32_t(rank));
    }

    // 4. Dynamically compile exact R2L 3-digit grouping pre_tokenizer chain from tokenizer.json!
    impl_->pretok_steps.clear();
    if (root.has("pre_tokenizer") && root.get("pre_tokenizer").is_object()) {
        const auto& pt = root.get("pre_tokenizer");
        auto add_pt = [&](const JsonValue& item) {
            const std::string tp = item.get("type").as_string();
            if (tp == "Digits") {
                const bool indiv = item.has("individual_digits") && item.get("individual_digits").as_bool();
                Pcre2Step st;
                st.kind = Pcre2Step::Kind::kDigits;
                st.individual_digits = indiv;
                st.re = compile_pcre2(indiv ? "\\p{Nd}" : "\\p{Nd}+");
                impl_->pretok_steps.push_back(std::move(st));
            } else if (tp == "Split") {
                const auto& pat = item.get("pattern");
                if (pat.has("Regex")) {
                    Pcre2Step st;
                    st.kind = Pcre2Step::Kind::kSplitIsolated;
                    st.re = compile_pcre2(pat.get("Regex").as_string());
                    impl_->pretok_steps.push_back(std::move(st));
                }
            }
        };
        if (pt.get("type").as_string() == "Sequence" && pt.has("pretokenizers")) {
            for (const auto& item : pt.get("pretokenizers").arr_val) add_pt(item);
        } else {
            add_pt(pt);
        }
    }

    // 5. Inspect tokenizer_config.json / chat_template.jinja for automatic chat-mode detection
    impl_->has_chat_tpl = false;
    std::string cfg_path = tokenizer_config_path;
    if (cfg_path.empty()) {
        std::filesystem::path p(tokenizer_json_path);
        auto cand = p.parent_path() / "tokenizer_config.json";
        if (std::filesystem::exists(cand)) cfg_path = cand.string();
    }
    if (!cfg_path.empty() && std::filesystem::exists(cfg_path)) {
        std::ifstream cfs(cfg_path);
        if (cfs) {
            std::ostringstream css;
            css << cfs.rdbuf();
            JsonValue cj = JsonParser::parse(css.str());
            if ((cj.has("chat_template") && cj.get("chat_template").is_string() &&
                 !cj.get("chat_template").as_string().empty()) ||
                (cj.has("tokenizer_class") && cj.get("tokenizer_class").is_string() &&
                 cj.get("tokenizer_class").as_string() == "ErnieChatTokenizer")) {
                impl_->has_chat_tpl = true;
            }
        }
        std::filesystem::path p(cfg_path);
        if (std::filesystem::exists(p.parent_path() / "chat_template.jinja")) {
            impl_->has_chat_tpl = true;
        }
    }
    impl_->is_loaded = true;
}

std::vector<std::string> Ernie55Tokenizer::pretokenize_debug(const std::string& text) const {
    return impl_->run_pretokenizer(text);
}

std::vector<int32_t> Ernie55Tokenizer::encode(const std::string& text, bool allow_special) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    if (!allow_special || impl_->added_sorted_desc.empty()) {
        impl_->encode_plain_parallel(text.data(), text.size(), out);
        return out;
    }

    // Split on added tokens first (longest match first), BPE-encode non-special spans
    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& kv : impl_->added_sorted_desc) {
            const auto& tok = kv.first;
            if (tok.size() <= text.size() - pos && text.compare(pos, tok.size(), tok) == 0) {
                if (pos > plain_start) {
                    impl_->encode_plain_parallel(text.data() + plain_start, pos - plain_start, out);
                }
                out.push_back(kv.second);
                pos += tok.size();
                plain_start = pos;
                matched = true;
                break;
            }
        }
        if (!matched) ++pos;
    }
    if (plain_start < text.size()) {
        impl_->encode_plain_parallel(text.data() + plain_start, text.size() - plain_start, out);
    }
    return out;
}

std::vector<int32_t> Ernie55Tokenizer::encode_serial_reference(const std::string& text, bool allow_special) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    if (!allow_special || impl_->added_sorted_desc.empty()) {
        impl_->encode_plain_serial(text.data(), text.size(), out);
        return out;
    }

    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& kv : impl_->added_sorted_desc) {
            const auto& tok = kv.first;
            if (tok.size() <= text.size() - pos && text.compare(pos, tok.size(), tok) == 0) {
                if (pos > plain_start) {
                    impl_->encode_plain_serial(text.data() + plain_start, pos - plain_start, out);
                }
                out.push_back(kv.second);
                pos += tok.size();
                plain_start = pos;
                matched = true;
                break;
            }
        }
        if (!matched) ++pos;
    }
    if (plain_start < text.size()) {
        impl_->encode_plain_serial(text.data() + plain_start, text.size() - plain_start, out);
    }
    return out;
}

void Ernie55Tokenizer::cache_stats(uint64_t& hits, uint64_t& misses, size_t& bytes) const noexcept {
    if (!impl_) { hits = 0; misses = 0; bytes = 0; return; }
    hits   = impl_->cache_hits.load(std::memory_order_relaxed);
    misses = impl_->cache_misses.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(impl_->cache_mtx);
    bytes  = impl_->cache_bytes;
}

void Ernie55Tokenizer::clear_cache() const {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->cache_mtx);
    impl_->lru_index.clear();
    impl_->lru_list.clear();
    impl_->cache_bytes = 0u;
    impl_->cache_hits.store(0ULL, std::memory_order_relaxed);
    impl_->cache_misses.store(0ULL, std::memory_order_relaxed);
}

std::vector<int32_t> Ernie55Tokenizer::encode_segments(const std::vector<ChatSegment>& segs) const {
    std::vector<int32_t> out;
    if (!impl_) return out;
    std::string plain_buf;
    auto flush_plain = [&]() {
        if (plain_buf.empty()) return;
        impl_->encode_plain_cached(plain_buf, out);
        plain_buf.clear();
    };
    for (const auto& seg : segs) {
        if (seg.text.empty()) continue;
        if (!seg.allow_special) {
            plain_buf += seg.text;
            continue;
        }
        size_t pos = 0;
        while (pos < seg.text.size()) {
            bool matched = false;
            for (const auto& kv : impl_->added_sorted_desc) {
                const auto& tok = kv.first;
                if (tok.size() <= seg.text.size() - pos && seg.text.compare(pos, tok.size(), tok) == 0) {
                    flush_plain();
                    out.push_back(kv.second);
                    pos += tok.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                plain_buf.push_back(seg.text[pos++]);
            }
        }
    }
    flush_plain();
    return out;
}

std::string Ernie55Tokenizer::decode_single(int32_t id, bool skip_special) const {
    if (id < 0 || size_t(id) >= impl_->id_to_token.size()) return "";
    if (skip_special && (impl_->id_is_special[size_t(id)] || id == pad_id_)) return "";
    if (impl_->added_token_to_id.find(impl_->id_to_token[size_t(id)]) != impl_->added_token_to_id.end()) {
        return impl_->id_to_token[size_t(id)];
    }
    return bytelevel_decode(impl_->id_to_token[size_t(id)]);
}

std::string Ernie55Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) out += decode_single(id, skip_special);
    return out;
}

namespace {

// Identifier-first property priority for tool JSON schemas so ERNIE emits file_path / path / description
// BEFORE bulk payload parameters (content / new_string / old_string / command), letting :3081 render the
// exact Write <file> / Edit <file> / Bash <desc> card from Burst 1!
int tool_prop_order_rank(const std::string& k) noexcept {
    static constexpr const char* kFirst[] = {
        "file_path", "path", "filepath", "filename", "description", "name",
        "title", "pattern", "query", "glob", "url", "id", "action", "mode",
        "offset", "limit", "timeout", "timeoutMs", "replace_all"
    };
    for (const char* f : kFirst) if (k == f) return 0;
    static constexpr const char* kLast[] = {
        "old_string", "new_string", "content", "command", "edits", "patch", "body", "code", "text", "data"
    };
    for (const char* l : kLast) if (k == l) return 2;
    return 1;
}

// Python-exact json.dumps(val, ensure_ascii=False, sort_keys=True) matching encoding_ernie.py::_json,
// with optional identifier-first ordering for JSON-schema "properties" maps!
std::string py_json_dumps_sorted(const JsonValue& v, bool reorder_tool_props = false, bool in_props_map = false) {
    switch (v.type) {
        case JsonValue::Type::Null:   return "null";
        case JsonValue::Type::Bool:   return v.b_val ? "true" : "false";
        case JsonValue::Type::Int:    return std::to_string(v.i_val);
        case JsonValue::Type::Double: {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", v.d_val);
            return buf;
        }
        case JsonValue::Type::String: {
            std::string s = "\"";
            for (unsigned char c : v.s_val) {
                if (c == '"') s += "\\\"";
                else if (c == '\\') s += "\\\\";
                else if (c == '\b') s += "\\b";
                else if (c == '\f') s += "\\f";
                else if (c == '\n') s += "\\n";
                else if (c == '\r') s += "\\r";
                else if (c == '\t') s += "\\t";
                else if (c < 0x20) { char h[8]; std::snprintf(h, sizeof(h), "\\u%04x", c); s += h; }
                else s.push_back(char(c));
            }
            s.push_back('"');
            return s;
        }
        case JsonValue::Type::Array: {
            std::string s = "[";
            for (size_t i = 0; i < v.arr_val.size(); ++i) {
                if (i > 0) s += ", ";
                s += py_json_dumps_sorted(v.arr_val[i], reorder_tool_props, false);
            }
            s.push_back(']');
            return s;
        }
        case JsonValue::Type::Object: {
            std::vector<size_t> ord(v.obj_val.size());
            for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
            std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
                if (reorder_tool_props && in_props_map) {
                    const int ra = tool_prop_order_rank(v.obj_val[a].first);
                    const int rb = tool_prop_order_rank(v.obj_val[b].first);
                    if (ra != rb) return ra < rb;
                }
                return v.obj_val[a].first < v.obj_val[b].first;
            });
            std::string s = "{";
            for (size_t i = 0; i < ord.size(); ++i) {
                if (i > 0) s += ", ";
                const std::string& kname = v.obj_val[ord[i]].first;
                JsonValue ks = JsonValue::make_string(kname);
                s += py_json_dumps_sorted(ks, false, false);
                s += ": ";
                const bool child_is_props = reorder_tool_props && (kname == "properties");
                s += py_json_dumps_sorted(v.obj_val[ord[i]].second, reorder_tool_props, child_is_props);
            }
            s.push_back('}');
            return s;
        }
    }
    return "null";
}

void strip_outer_newlines(std::vector<ChatSegment>& segs) {
    // Strip leading '\n'
    while (!segs.empty()) {
        size_t i = 0;
        while (i < segs.front().text.size() && segs.front().text[i] == '\n') ++i;
        if (i == segs.front().text.size()) segs.erase(segs.begin());
        else { if (i > 0) segs.front().text.erase(0, i); break; }
    }
    // Strip trailing '\n'
    while (!segs.empty()) {
        auto& t = segs.back().text;
        while (!t.empty() && t.back() == '\n') t.pop_back();
        if (t.empty()) segs.pop_back();
        else break;
    }
}

bool has_non_ws(const std::string& s) {
    for (unsigned char c : s) if (c > ' ') return true;
    return false;
}

} // namespace

std::vector<ChatSegment> Ernie55Tokenizer::render_chat_segments(
    const std::vector<ChatMessage>& messages,
    const JsonValue* tools,
    bool enable_thinking,
    bool add_generation_prompt,
    const std::string& reasoning_effort,
    bool parallel_tool_calls) const
{
    static constexpr const char* kMask     = "<|ERNIE_MASK|>";
    static constexpr const char* kStart    = "<|ernie:im_start|>";
    static constexpr const char* kEnd      = "<|ernie:im_end|>";
    static constexpr const char* kThink    = "<ernie:think>";
    static constexpr const char* kThinkEnd = "</ernie:think>";
    static constexpr const char* kCall     = "<ernie:tool_call>";
    static constexpr const char* kCallEnd  = "</ernie:tool_call>";
    static constexpr const char* kOut      = "<ernie:tool_output>";
    static constexpr const char* kOutEnd   = "</ernie:tool_output>";

    const bool has_tools = (tools && tools->is_array() && !tools->arr_val.empty());
    std::vector<ChatSegment> out;
    out.push_back({std::string(kMask) + kStart + "system\n", true});

    if (!messages.empty() && messages[0].role == "system") {
        out.push_back({"<ernie:system_setting>", true});
        out.push_back({"\n", true});
        out.push_back({messages[0].content, false});
        out.push_back({"\n</ernie:system_setting>\n", true});
    }

    out.push_back({"<ernie:global_setting>", true});
    if (enable_thinking) {
        const std::string eff = reasoning_effort.empty() ? "high" : reasoning_effort;
        out.push_back({"\nreasoning_effort=", true});
        out.push_back({eff, false});
    }
    if (parallel_tool_calls && has_tools) {
        out.push_back({"\nparallel_tool_calls=True", true});
    }
    out.push_back({"\n</ernie:global_setting>", true});

    if (has_tools) {
        std::string tstr;
        for (size_t i = 0; i < tools->arr_val.size(); ++i) {
            if (i > 0) tstr += ", ";
            tstr += py_json_dumps_sorted(tools->arr_val[i], true, false);
        }
        out.push_back({"\n<ernie:tool_list>\n[", true});
        out.push_back({tstr, false});
        out.push_back({"]\n</ernie:tool_list>", true});
    }
    out.push_back({std::string(kEnd) + "\n", true});

    // Render role-by-role conversation body matching encoding_ernie.py::_Renderer::messages()
    std::vector<ChatSegment> body_msgs;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        if (m.role == "user" || (m.role == "system" && i > 0)) {
            body_msgs.push_back({std::string(kStart) + m.role + "\n", true});
            body_msgs.push_back({m.content, false});
            body_msgs.push_back({std::string(kEnd) + "\n", true});
        } else if (m.role == "assistant") {
            std::string reasoning = m.reasoning_content;
            std::string content   = m.content;
            if (reasoning.empty()) {
                const auto te = content.find(kThinkEnd);
                if (te != std::string::npos) {
                    const auto ts = content.rfind(kThink, te);
                    const size_t r0 = (ts != std::string::npos) ? (ts + std::strlen(kThink)) : 0u;
                    reasoning = content.substr(r0, te - r0);
                    const auto te_last = content.rfind(kThinkEnd);
                    content = content.substr(te_last + std::strlen(kThinkEnd));
                }
            }
            body_msgs.push_back({std::string(kStart) + "assistant\n", true});
            body_msgs.push_back({kThink, true});
            if (enable_thinking && has_non_ws(reasoning)) {
                body_msgs.push_back({reasoning, false});
            }
            body_msgs.push_back({kThinkEnd, true});
            body_msgs.push_back({content, false});
            if (!m.tool_calls.empty()) {
                body_msgs.push_back({std::string(kCall) + "\n", true});
                for (const auto& tc : m.tool_calls) {
                    body_msgs.push_back({"<invoke name=\"", true});
                    body_msgs.push_back({tc.name, false});
                    body_msgs.push_back({"\">", true});
                    JsonValue aj;
                    if (JsonParser::try_parse(tc.arguments_json, aj) && aj.is_object()) {
                        std::vector<size_t> ord(aj.obj_val.size());
                        for (size_t ai = 0; ai < ord.size(); ++ai) ord[ai] = ai;
                        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
                            return aj.obj_val[a].first < aj.obj_val[b].first;
                        });
                        for (size_t oi : ord) {
                            const auto& k   = aj.obj_val[oi].first;
                            const auto& val = aj.obj_val[oi].second;
                            body_msgs.push_back({"<parameter name=\"", true});
                            body_msgs.push_back({k, false});
                            body_msgs.push_back({"\">", true});
                            body_msgs.push_back({val.is_string() ? val.as_string() : py_json_dumps_sorted(val), false});
                            body_msgs.push_back({"</parameter>", true});
                        }
                    }
                    body_msgs.push_back({"</invoke>\n", true});
                }
                body_msgs.push_back({kCallEnd, true});
            }
            body_msgs.push_back({kEnd, true});
            body_msgs.push_back({"\n", true});
        } else if (m.role == "tool") {
            if (i == 0 || messages[i - 1].role != "tool") {
                body_msgs.push_back({std::string(kStart) + "tool", true});
            }
            body_msgs.push_back({std::string("\n") + kOut, true});
            body_msgs.push_back({m.content, false});
            body_msgs.push_back({kOutEnd, true});
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") {
                body_msgs.push_back({std::string(kEnd) + "\n", true});
            }
        }
    }
    strip_outer_newlines(body_msgs);
    out.insert(out.end(), body_msgs.begin(), body_msgs.end());

    if (add_generation_prompt) {
        out.push_back({std::string("\n") + kStart + "assistant\n", true});
        out.push_back({enable_thinking ? std::string(kThink) : (std::string(kThink) + kThinkEnd), true});
    }
    return out;
}

std::vector<ChatSegment> Ernie55Tokenizer::render_chat_tail_segments(
    const std::vector<ChatMessage>& tail_messages,
    bool enable_thinking,
    bool add_generation_prompt) const
{
    static constexpr const char* kStart    = "<|ernie:im_start|>";
    static constexpr const char* kEnd      = "<|ernie:im_end|>";
    static constexpr const char* kThink    = "<ernie:think>";
    static constexpr const char* kThinkEnd = "</ernie:think>";
    static constexpr const char* kCall     = "<ernie:tool_call>";
    static constexpr const char* kCallEnd  = "</ernie:tool_call>";
    static constexpr const char* kOut      = "<ernie:tool_output>";
    static constexpr const char* kOutEnd   = "</ernie:tool_output>";

    std::vector<ChatSegment> body_msgs;
    for (size_t i = 0; i < tail_messages.size(); ++i) {
        const auto& m = tail_messages[i];
        if (m.role == "user" || m.role == "system") {
            body_msgs.push_back({std::string(kStart) + m.role + "\n", true});
            body_msgs.push_back({m.content, false});
            body_msgs.push_back({std::string(kEnd) + "\n", true});
        } else if (m.role == "assistant") {
            std::string reasoning = m.reasoning_content;
            std::string content   = m.content;
            if (reasoning.empty()) {
                const auto te = content.find(kThinkEnd);
                if (te != std::string::npos) {
                    const auto ts = content.rfind(kThink, te);
                    const size_t r0 = (ts != std::string::npos) ? (ts + std::strlen(kThink)) : 0u;
                    reasoning = content.substr(r0, te - r0);
                    const auto te_last = content.rfind(kThinkEnd);
                    content = content.substr(te_last + std::strlen(kThinkEnd));
                }
            }
            body_msgs.push_back({std::string(kStart) + "assistant\n", true});
            body_msgs.push_back({kThink, true});
            if (enable_thinking && has_non_ws(reasoning)) {
                body_msgs.push_back({reasoning, false});
            }
            body_msgs.push_back({kThinkEnd, true});
            body_msgs.push_back({content, false});
            if (!m.tool_calls.empty()) {
                body_msgs.push_back({std::string(kCall) + "\n", true});
                for (const auto& tc : m.tool_calls) {
                    body_msgs.push_back({"<invoke name=\"", true});
                    body_msgs.push_back({tc.name, false});
                    body_msgs.push_back({"\">", true});
                    JsonValue aj;
                    if (JsonParser::try_parse(tc.arguments_json, aj) && aj.is_object()) {
                        std::vector<size_t> ord(aj.obj_val.size());
                        for (size_t ai = 0; ai < ord.size(); ++ai) ord[ai] = ai;
                        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
                            return aj.obj_val[a].first < aj.obj_val[b].first;
                        });
                        for (size_t oi : ord) {
                            const auto& k   = aj.obj_val[oi].first;
                            const auto& val = aj.obj_val[oi].second;
                            body_msgs.push_back({"<parameter name=\"", true});
                            body_msgs.push_back({k, false});
                            body_msgs.push_back({"\">", true});
                            body_msgs.push_back({val.is_string() ? val.as_string() : py_json_dumps_sorted(val), false});
                            body_msgs.push_back({"</parameter>", true});
                        }
                    }
                    body_msgs.push_back({"</invoke>\n", true});
                }
                body_msgs.push_back({kCallEnd, true});
            }
            body_msgs.push_back({kEnd, true});
            body_msgs.push_back({"\n", true});
        } else if (m.role == "tool") {
            if (i == 0 || tail_messages[i - 1].role != "tool") {
                body_msgs.push_back({std::string(kStart) + "tool", true});
            }
            body_msgs.push_back({std::string("\n") + kOut, true});
            body_msgs.push_back({m.content, false});
            body_msgs.push_back({kOutEnd, true});
            if (i + 1 == tail_messages.size() || tail_messages[i + 1].role != "tool") {
                body_msgs.push_back({std::string(kEnd) + "\n", true});
            }
        }
    }
    strip_outer_newlines(body_msgs);
    std::vector<ChatSegment> out;
    if (!body_msgs.empty()) {
        // Restore the '\n' stripped from the cached prefix's final <|ernie:im_end|>\n before the new message!
        out.push_back({"\n", true});
        out.insert(out.end(), body_msgs.begin(), body_msgs.end());
    }
    if (add_generation_prompt) {
        out.push_back({std::string("\n") + kStart + "assistant\n", true});
        out.push_back({enable_thinking ? std::string(kThink) : (std::string(kThink) + kThinkEnd), true});
    }
    return out;
}

int32_t Ernie55Tokenizer::token_to_id_opt(const std::string& tok) const noexcept {
    if (!impl_) return -1;
    auto it = impl_->added_token_to_id.find(tok);
    if (it != impl_->added_token_to_id.end()) return it->second;
    auto it2 = impl_->token_to_id.find(tok);
    return (it2 != impl_->token_to_id.end()) ? it2->second : -1;
}

Ernie55SpecialTokenIds Ernie55Tokenizer::special_ids() const noexcept {
    Ernie55SpecialTokenIds s;
    s.im_start_id        = im_start_id_;
    s.im_end_id          = im_end_id_;
    s.tool_call_start_id = token_to_id_opt("<ernie:tool_call>");
    s.tool_call_end_id   = token_to_id_opt("</ernie:tool_call>");
    s.think_start_id     = token_to_id_opt("<ernie:think>");
    s.think_end_id       = token_to_id_opt("</ernie:think>");
    return s;
}

} // namespace ernie55
