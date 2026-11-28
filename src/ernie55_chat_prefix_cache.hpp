// SPDX-License-Identifier: Apache-2.0
#pragma once
// Wave-12 Cut R12.3: Sub-8 ms Zero-Copy Multi-Turn JSON Prefix Span Matcher.
// Caches the previous request's raw `"messages":[ ... ]` inner bytes, header config signature
// (tools JSON + enable_thinking + parallel_tool_calls + reasoning_effort), ending-tool flag,
// and rendered token IDs before add_generation_prompt. On subsequent multi-turn requests:
//   1. Locate `"messages":[` at top-level depth 1 without parsing the message array.
//   2. Check cached entries via O(1) length + fast memcmp over the unchanged prefix (~80 us at 512K)
//      and scan ONLY the tiny newly appended tail to find the closing `]`.
//   3. Parse ONLY the stripped outer JSON (`"messages":[]`) and the tiny `[tail]` array, verify
//      exact header signature and non-tool-continuation seam, and BPE-encode ONLY the tail.
// Any mismatch, edit, truncation, or consecutive tool-chain continuation falls back cleanly to
// full JsonParser::parse + render_chat_segments + encode_segments (100% bit-exact parity).

#include "ernie55_json.hpp"
#include "ernie55_tokenizer.hpp"

#include <cstdint>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <vector>

namespace ernie55 {

inline bool chat_json_is_ws(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

// Advance `pos` past a JSON string starting at raw[pos] == '"'. Returns false if unterminated.
inline bool chat_json_skip_string(const char* raw, size_t len, size_t& pos) noexcept {
    if (pos >= len || raw[pos] != '"') return false;
    ++pos;
    while (pos < len) {
        const char c = raw[pos++];
        if (c == '\\') {
            if (pos >= len) return false;
            ++pos;
        } else if (c == '"') {
            return true;
        }
    }
    return false;
}

// Advance `pos` past any JSON value starting at raw[pos]. Returns false if malformed.
inline bool chat_json_skip_value(const char* raw, size_t len, size_t& pos) noexcept {
    while (pos < len && chat_json_is_ws(raw[pos])) ++pos;
    if (pos >= len) return false;
    const char c = raw[pos];
    if (c == '"') return chat_json_skip_string(raw, len, pos);
    if (c == '{' || c == '[') {
        int depth = 0;
        while (pos < len) {
            const char ch = raw[pos];
            if (ch == '"') {
                if (!chat_json_skip_string(raw, len, pos)) return false;
                continue;
            }
            if (ch == '{' || ch == '[') ++depth;
            else if (ch == '}' || ch == ']') {
                --depth;
                ++pos;
                if (depth == 0) return true;
                continue;
            }
            ++pos;
        }
        return false;
    }
    while (pos < len && !chat_json_is_ws(raw[pos]) && raw[pos] != ',' && raw[pos] != '}' && raw[pos] != ']') {
        ++pos;
    }
    return true;
}

// Locate the opening '[' of the top-level `"messages"` array (depth 1).
// Returns true with `arr_beg` pointing to '[' and `first_elem` pointing to first non-ws byte inside '['.
inline bool chat_json_find_messages_open(const char* raw, size_t len,
                                         size_t& arr_beg, size_t& first_elem) noexcept {
    size_t pos = 0;
    while (pos < len && chat_json_is_ws(raw[pos])) ++pos;
    if (pos >= len || raw[pos] != '{') return false;
    ++pos;
    while (pos < len) {
        while (pos < len && chat_json_is_ws(raw[pos])) ++pos;
        if (pos >= len || raw[pos] == '}') return false;
        if (raw[pos] != '"') return false;
        const size_t key_beg = pos;
        if (!chat_json_skip_string(raw, len, pos)) return false;
        const size_t key_end = pos;
        while (pos < len && chat_json_is_ws(raw[pos])) ++pos;
        if (pos >= len || raw[pos] != ':') return false;
        ++pos;
        while (pos < len && chat_json_is_ws(raw[pos])) ++pos;
        if (key_end - key_beg == 10u && std::memcmp(raw + key_beg, "\"messages\"", 10u) == 0) {
            if (pos >= len || raw[pos] != '[') return false;
            arr_beg = pos;
            first_elem = pos + 1u;
            while (first_elem < len && chat_json_is_ws(raw[first_elem])) ++first_elem;
            return true;
        }
        if (!chat_json_skip_value(raw, len, pos)) return false;
        while (pos < len && chat_json_is_ws(raw[pos])) ++pos;
        if (pos < len && raw[pos] == ',') ++pos;
    }
    return false;
}

// Starting from `pos` inside an open `[` array at depth 1, scan forward to the matching `]` at depth 0.
inline bool chat_json_find_array_close_from(const char* raw, size_t len, size_t pos, size_t& arr_end) noexcept {
    int depth = 1;
    while (pos < len) {
        const char c = raw[pos];
        if (c == '"') {
            if (!chat_json_skip_string(raw, len, pos)) return false;
            continue;
        }
        if (c == '[' || c == '{') {
            ++depth;
        } else if (c == ']' || c == '}') {
            --depth;
            if (depth == 0) {
                if (c != ']') return false;
                arr_end = pos;
                return true;
            }
        }
        ++pos;
    }
    return false;
}

// Canonical header configuration signature (exact string equality -> 0 collision risk).
inline std::string make_chat_header_sig(const JsonValue* tools, bool enable_thinking,
                                        bool parallel_tc, const std::string& reasoning_effort) {
    std::string s;
    s.reserve(256u);
    s += enable_thinking ? "T1|" : "T0|";
    s += parallel_tc ? "P1|" : "P0|";
    s += reasoning_effort.empty() ? "high" : reasoning_effort;
    s += "|";
    if (tools && tools->is_array() && !tools->arr_val.empty()) {
        s += tools->dumps_compact();
    }
    return s;
}

inline std::vector<ChatMessage> parse_chat_messages_vec(const JsonValue& arr) {
    std::vector<ChatMessage> msgs;
    if (!arr.is_array()) return msgs;
    for (const auto& m : arr.arr_val) {
        if (!m.is_object()) continue;
        ChatMessage cm;
        if (m.has("role") && m.get("role").is_string()) cm.role = m.get("role").as_string();
        if (m.has("name") && m.get("name").is_string()) cm.name = m.get("name").as_string();
        if (m.has("tool_call_id") && m.get("tool_call_id").is_string()) cm.tool_call_id = m.get("tool_call_id").as_string();
        if (m.has("reasoning_content") && m.get("reasoning_content").is_string()) {
            cm.reasoning_content = m.get("reasoning_content").as_string();
        }
        if (m.has("content")) {
            const auto& c = m.get("content");
            if (c.is_string()) {
                cm.content = c.as_string();
            } else if (c.is_array()) {
                for (const auto& part : c.arr_val) {
                    if (part.is_object() && part.has("text") && part.get("text").is_string()) {
                        cm.content += part.get("text").as_string();
                    }
                }
            }
        }
        if (m.has("tool_calls") && m.get("tool_calls").is_array()) {
            for (const auto& tc : m.get("tool_calls").arr_val) {
                if (!tc.is_object()) continue;
                ChatMessage::ToolCall ctc;
                if (tc.has("id") && tc.get("id").is_string()) ctc.id = tc.get("id").as_string();
                if (tc.has("function") && tc.get("function").is_object()) {
                    const auto& fn = tc.get("function");
                    if (fn.has("name") && fn.get("name").is_string()) ctc.name = fn.get("name").as_string();
                    if (fn.has("arguments")) {
                        ctc.arguments_json = fn.get("arguments").is_string()
                            ? fn.get("arguments").as_string()
                            : fn.get("arguments").dumps_compact();
                    }
                }
                cm.tool_calls.push_back(std::move(ctc));
            }
        }
        msgs.push_back(std::move(cm));
    }
    return msgs;
}

struct ChatPrefixParsedRequest {
    JsonValue            outer_json;       // Full top-level JSON (with `"messages":[]` on delta hit!)
    std::vector<int32_t> prompt_ids;       // 100% bit-exact rendered & BPE-tokenized prompt IDs
    bool                 delta_hit = false;
    bool                 enable_thinking = true;
    bool                 parallel_tc = true;
    std::string          reasoning_effort = "high";
};

class ChatPrefixCache {
public:
    static constexpr size_t kMaxEntries = 4u;

    // Fast path + automatic full-parse fallback: returns 100% bit-exact prompt_ids + outer JSON.
    bool parse_and_tokenize(const std::string& raw_body,
                            const Ernie55Tokenizer& tok,
                            ChatPrefixParsedRequest& out) {
        out = ChatPrefixParsedRequest{};
        const char* raw = raw_body.data();
        const size_t len = raw_body.size();

        auto extract_flags = [](const JsonValue& j, bool& enable_thinking,
                                bool& parallel_tc, std::string& effort,
                                const JsonValue*& tools) {
            tools = j.has("tools") ? &j.get("tools") : nullptr;
            const JsonValue* ctk = j.has("chat_template_kwargs") ? &j.get("chat_template_kwargs") : nullptr;
            enable_thinking = is_ernie5_thinking_enabled(ctk);
            if (j.has("enable_thinking") && j.get("enable_thinking").is_bool()) {
                enable_thinking = j.get("enable_thinking").as_bool();
            }
            parallel_tc = !(j.has("parallel_tool_calls") && j.get("parallel_tool_calls").is_bool() &&
                            !j.get("parallel_tool_calls").as_bool());
            effort = "high";
            if (ctk && ctk->is_object() && ctk->has("reasoning_effort") && ctk->get("reasoning_effort").is_string()) {
                effort = ctk->get("reasoning_effort").as_string();
            } else if (j.has("reasoning_effort") && j.get("reasoning_effort").is_string()) {
                effort = j.get("reasoning_effort").as_string();
            }
        };

        size_t arr_beg = 0, first_elem = 0;
        const bool found_msgs = chat_json_find_messages_open(raw, len, arr_beg, first_elem);

        if (found_msgs) {
            // Probe up to 4 cached LRU entries via fast memcmp over unchanged prefix (~80 us at 512K!)
            // Copy ONLY metadata + token IDs (never copy multi-MB prefix_bytes out of LRU!).
            std::string matched_sig;
            std::vector<int32_t> matched_tokens;
            bool matched_ends_in_tool = false;
            size_t tail_comma = 0;
            bool exact_repeat = false;
            bool hit_prefix = false;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                for (auto it = lru_.begin(); it != lru_.end(); ++it) {
                    const size_t plen = it->prefix_bytes.size();
                    if (plen == 0u || first_elem + plen >= len) continue;
                    size_t after = first_elem + plen;
                    while (after < len && chat_json_is_ws(raw[after])) ++after;
                    if (after >= len) continue;
                    if (raw[after] != ',' && raw[after] != ']') continue;
                    if (std::memcmp(raw + first_elem, it->prefix_bytes.data(), plen) != 0) continue;
                    matched_sig          = it->header_sig;
                    matched_tokens       = it->tokens_no_gen;
                    matched_ends_in_tool = it->ends_in_tool;
                    exact_repeat = (raw[after] == ']');
                    tail_comma = after;
                    lru_.splice(lru_.begin(), lru_, it);
                    hit_prefix = true;
                    break;
                }
            }

            if (hit_prefix) {
                size_t arr_end = tail_comma;
                const bool closed = exact_repeat
                    ? true
                    : chat_json_find_array_close_from(raw, len, tail_comma + 1u, arr_end);
                if (closed && arr_end > arr_beg) {
                    std::string outer_raw;
                    outer_raw.reserve((arr_beg + 1u) + (len - arr_end));
                    outer_raw.append(raw, arr_beg + 1u);
                    outer_raw.append(raw + arr_end, len - arr_end);
                    JsonValue outer_j;
                    if (JsonParser::try_parse(outer_raw, outer_j) && outer_j.is_object()) {
                        const JsonValue* tools = nullptr;
                        extract_flags(outer_j, out.enable_thinking, out.parallel_tc, out.reasoning_effort, tools);
                        const std::string sig = make_chat_header_sig(tools, out.enable_thinking, out.parallel_tc, out.reasoning_effort);
                        if (sig == matched_sig) {
                            if (exact_repeat) {
                                const auto gen_segs = tok.render_chat_tail_segments({}, out.enable_thinking, true);
                                const auto gen_ids  = tok.encode_segments(gen_segs);
                                out.prompt_ids = std::move(matched_tokens);
                                out.prompt_ids.insert(out.prompt_ids.end(), gen_ids.begin(), gen_ids.end());
                                out.outer_json = std::move(outer_j);
                                out.delta_hit  = true;
                                return true;
                            }
                            std::string tail_raw;
                            tail_raw.reserve((arr_end - tail_comma) + 2u);
                            tail_raw.push_back('[');
                            tail_raw.append(raw + tail_comma + 1u, arr_end - tail_comma - 1u);
                            tail_raw.push_back(']');
                            JsonValue tail_j;
                            if (JsonParser::try_parse(tail_raw, tail_j) && tail_j.is_array() && !tail_j.arr_val.empty()) {
                                const auto tail_msgs = parse_chat_messages_vec(tail_j);
                                // Seam safety: consecutive tool-role messages merge into a single <|ernie:im_start|>tool block
                                // inside render_chat_segments, so fall back to full parse ONLY if prefix ended in "tool" AND tail starts with "tool".
                                if (!tail_msgs.empty() && !(matched_ends_in_tool && tail_msgs.front().role == "tool")) {
                                    const auto tail_segs_no_gen = tok.render_chat_tail_segments(tail_msgs, out.enable_thinking, false);
                                    const auto tail_ids_no_gen  = tok.encode_segments(tail_segs_no_gen);
                                    const auto gen_segs         = tok.render_chat_tail_segments({}, out.enable_thinking, true);
                                    const auto gen_ids          = tok.encode_segments(gen_segs);

                                    matched_tokens.insert(matched_tokens.end(), tail_ids_no_gen.begin(), tail_ids_no_gen.end());

                                    out.prompt_ids = matched_tokens;
                                    out.prompt_ids.insert(out.prompt_ids.end(), gen_ids.begin(), gen_ids.end());
                                    out.outer_json = std::move(outer_j);
                                    out.delta_hit  = true;

                                    // Trim trailing whitespace before ']' and extend LRU entry in place (O(tail_len)!)
                                    size_t trim_end = arr_end;
                                    while (trim_end > first_elem && chat_json_is_ws(raw[trim_end - 1u])) --trim_end;
                                    commit_locked(raw + first_elem, trim_end - first_elem, sig,
                                                  std::move(matched_tokens), tail_msgs.back().role == "tool");
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Full parse fallback (cold turn, prefix edit/truncation, or consecutive tool-chain across turns)
        JsonValue full_j = JsonParser::parse(raw_body);
        const auto msgs = parse_chat_messages_vec(full_j.get("messages"));
        const JsonValue* tools = nullptr;
        extract_flags(full_j, out.enable_thinking, out.parallel_tc, out.reasoning_effort, tools);
        const std::string sig = make_chat_header_sig(tools, out.enable_thinking, out.parallel_tc, out.reasoning_effort);

        const auto segs_no_gen = tok.render_chat_segments(
            msgs, tools, out.enable_thinking, false, out.reasoning_effort, out.parallel_tc);
        std::vector<int32_t> ids_no_gen = tok.encode_segments(segs_no_gen);
        const auto gen_segs = tok.render_chat_tail_segments({}, out.enable_thinking, true);
        const auto gen_ids  = tok.encode_segments(gen_segs);

        out.prompt_ids = ids_no_gen;
        out.prompt_ids.insert(out.prompt_ids.end(), gen_ids.begin(), gen_ids.end());
        out.outer_json = std::move(full_j);
        out.delta_hit  = false;

        // Commit prefix if at least one non-system body message exists (guarantees prefix ends in <|ernie:im_end|>)
        const bool has_body = !msgs.empty() && (msgs.size() > 1u || msgs[0].role != "system");
        if (found_msgs && has_body) {
            size_t arr_end = 0;
            if (chat_json_find_array_close_from(raw, len, first_elem, arr_end) && arr_end > first_elem) {
                size_t trim_end = arr_end;
                while (trim_end > first_elem && chat_json_is_ws(raw[trim_end - 1u])) --trim_end;
                if (trim_end > first_elem) {
                    commit_locked(raw + first_elem, trim_end - first_elem, sig,
                                  std::move(ids_no_gen), msgs.back().role == "tool");
                }
            }
        }
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return lru_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        lru_.clear();
    }

private:
    struct Entry {
        std::string          prefix_bytes;
        std::string          header_sig;
        std::vector<int32_t> tokens_no_gen;
        bool                 ends_in_tool = false;
    };

    void commit_locked(const char* bytes, size_t len, const std::string& sig,
                       std::vector<int32_t> ids_no_gen, bool ends_in_tool) {
        std::lock_guard<std::mutex> lk(mtx_);
        // Replace existing entry if same header_sig and strictly extending its prefix (O(tail_len) append!), else push front
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            const size_t old_len = it->prefix_bytes.size();
            if (it->header_sig == sig && len >= old_len &&
                std::memcmp(bytes, it->prefix_bytes.data(), old_len) == 0) {
                if (len > old_len) {
                    it->prefix_bytes.append(bytes + old_len, len - old_len);
                }
                it->tokens_no_gen = std::move(ids_no_gen);
                it->ends_in_tool  = ends_in_tool;
                lru_.splice(lru_.begin(), lru_, it);
                return;
            }
        }
        Entry e;
        e.prefix_bytes.assign(bytes, len);
        e.header_sig    = sig;
        e.tokens_no_gen = std::move(ids_no_gen);
        e.ends_in_tool  = ends_in_tool;
        lru_.push_front(std::move(e));
        while (lru_.size() > kMaxEntries) lru_.pop_back();
    }

    mutable std::mutex mtx_;
    std::list<Entry> lru_;
};

}  // namespace ernie55
