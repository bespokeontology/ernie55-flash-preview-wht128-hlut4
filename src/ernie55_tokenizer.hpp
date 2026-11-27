// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure-C++17 Exact ERNIE-5.5 R2L 3-Digit Grouping 201K ByteLevel BPE Tokenizer,
// Segment-Safe Chat Renderer, and <ernie:think> / <ernie:tool_call><invoke> Streaming Parser.
// Zero Python anywhere. Links libpcre2-8.

#include "ernie55_json.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ernie55 {

struct ChatSegment {
    std::string text;
    bool allow_special = false;  // true ONLY for renderer-emitted control markers!
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::string tool_call_id;
    std::string name;
    struct ToolCall {
        std::string id;
        std::string name;
        std::string arguments_json;
    };
    std::vector<ToolCall> tool_calls;
};

struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// OpenAI-compatible incremental SSE tool-call delta (Wave-7 schema-safe online string-param streaming)
struct StreamToolCallDelta {
    int index = 0;               // 0-based monotonic per-response call index
    std::string id;              // populated on opening delta for this call
    std::string name;            // populated on opening delta for this call
    std::string arguments_delta; // raw JSON fragment to append to function.arguments
};

struct Ernie55SpecialTokenIds {
    int32_t im_start_id        = 7;
    int32_t im_end_id          = 8;
    int32_t tool_call_start_id = -1;
    int32_t tool_call_end_id   = -1;
    int32_t think_start_id     = -1;
    int32_t think_end_id       = -1;
};

// Mirror vLLM-ERNIE parser.py::is_thinking_enabled(chat_template_kwargs)
bool is_ernie5_thinking_enabled(const JsonValue* chat_template_kwargs) noexcept;

// Exact vLLM-ERNIE parser.py schema-aware parameter coercion ($ref, anyOf/oneOf/allOf, null/none/nil, yes/on/no/off)
JsonValue ernie5_coerce_param(const std::string& raw_value, const JsonValue* prop_schema, const JsonValue* root_schema);

// Streaming & complete ERNIE-5 reasoning (<ernie:think>...</ernie:think>) + tool-call (<ernie:tool_call><invoke...>)
// parser matching vLLM-ERNIE src/vllm_ernie/models/ernie5_5/parser.py & tests/models/ernie5_5/test_parser.py.
class Ernie55StreamParser {
public:
    explicit Ernie55StreamParser(bool thinking_enabled = true,
                                 const JsonValue* tools = nullptr,
                                 const Ernie55SpecialTokenIds* special_ids = nullptr);
    void reset(bool thinking_enabled = true,
               const JsonValue* tools = nullptr,
               const Ernie55SpecialTokenIds* special_ids = nullptr);

    void feed(const std::string& piece,
              std::string& out_reasoning_delta,
              std::string& out_content_delta,
              std::vector<ParsedToolCall>& out_tool_calls);

    // Token-ID-aware feed: recognizes control token IDs directly when special_ids are bound
    void feed_token(int32_t token_id,
                    const std::string& piece,
                    std::string& out_reasoning_delta,
                    std::string& out_content_delta,
                    std::vector<ParsedToolCall>& out_tool_calls);

    void flush(std::string& out_reasoning_delta,
               std::string& out_content_delta,
               std::vector<ParsedToolCall>& out_tool_calls);

    // Wave-7 schema-safe live streaming feed: streams "type":"string" tool parameters online
    // (12-byte </parameter> hold + UTF-8 boundary + null-word prefix guard) while keeping exact
    // buffered ernie5_coerce_param for all number/integer/boolean/object/array parameters.
    void set_parallel_tool_calls(bool v) noexcept { parallel_tool_calls_ = v; }

    void feed_token_stream(int32_t token_id,
                           const std::string& piece,
                           std::string& out_reasoning_delta,
                           std::string& out_content_delta,
                           std::vector<StreamToolCallDelta>& out_tool_deltas);

    void feed_stream(const std::string& piece,
                     std::string& out_reasoning_delta,
                     std::string& out_content_delta,
                     std::vector<StreamToolCallDelta>& out_tool_deltas);

    void flush_stream(std::string& out_reasoning_delta,
                      std::string& out_content_delta,
                      std::vector<StreamToolCallDelta>& out_tool_deltas);

    // Turn-scoped reasoning-end check (ignores </ernie:think> from earlier assistant turns!)
    bool is_reasoning_end(const std::vector<int32_t>& token_ids) const noexcept;

    // Single-pass complete parse matching Ernie5Parser.parse() (preserves trailing content after </ernie:tool_call>)
    struct CompleteResult {
        std::string reasoning;
        std::string content;
        std::vector<ParsedToolCall> tool_calls;
    };
    static CompleteResult parse_complete(const std::string& text,
                                         const JsonValue* tools = nullptr,
                                         bool thinking_enabled = true,
                                         bool parallel_tool_calls = true);

    // Streaming null-prefix check matching _ernie5_arg_converter (holds back open prefixes of null/none/nil)
    static bool is_null_word_prefix(const std::string& partial_val) noexcept;

private:
    enum class State { kReasoning, kContent, kToolCall };
    enum class ToolSubState { kSeekInvoke, kSeekParam, kStreamStringVal, kBufferParamVal, kSkipRest };
    State state_ = State::kReasoning;
    ToolSubState tool_sub_ = ToolSubState::kSeekInvoke;
    bool thinking_enabled_ = true;
    bool parallel_tool_calls_ = true;
    Ernie55SpecialTokenIds specials_{};
    const JsonValue* tools_ = nullptr;
    std::string buf_;
    std::string tool_buf_;
    int call_counter_ = 0;

    // Live incremental tool-call streaming state (used only by feed_*_stream / flush_stream)
    int cur_call_idx_ = -1;
    bool cur_invoke_open_ = false;
    int cur_param_count_ = 0;
    std::string cur_fname_;
    const JsonValue* cur_root_schema_ = nullptr;
    std::string cur_pname_;
    const JsonValue* cur_prop_schema_ = nullptr;
    std::string param_buf_;
    bool str_param_opened_ = false;

    const JsonValue* find_tool_params(const std::string& name) const;
    void parse_tool_block(const std::string& body, std::vector<ParsedToolCall>& out_calls);
    void process_tool_stream(std::vector<StreamToolCallDelta>& out_deltas, bool eof_block);
};

class Ernie55Tokenizer {
public:
    Ernie55Tokenizer();
    ~Ernie55Tokenizer();
    Ernie55Tokenizer(const Ernie55Tokenizer&) = delete;
    Ernie55Tokenizer& operator=(const Ernie55Tokenizer&) = delete;

    // Load tokenizer.json + optional tokenizer_config.json (auto-detects chat template & special IDs!)
    void load(const std::string& tokenizer_json_path, const std::string& tokenizer_config_path = "");

    bool loaded() const noexcept;
    int vocab_size() const noexcept;
    bool has_chat_template() const noexcept;

    int32_t pad_id() const noexcept { return pad_id_; }
    int32_t bos_id() const noexcept { return bos_id_; }
    int32_t eos_id() const noexcept { return eos_id_; }
    int32_t im_start_id() const noexcept { return im_start_id_; }
    int32_t im_end_id() const noexcept { return im_end_id_; }
    Ernie55SpecialTokenIds special_ids() const noexcept;
    int32_t token_to_id_opt(const std::string& tok) const noexcept;
    bool is_eos_token(int32_t id) const noexcept;

    // Exact R2L 3-digit grouping + ByteLevel BPE encode/decode
    std::vector<int32_t> encode(const std::string& text, bool allow_special = false) const;
    std::vector<int32_t> encode_serial_reference(const std::string& text, bool allow_special = false) const;
    std::vector<int32_t> encode_segments(const std::vector<ChatSegment>& segs) const;
    std::vector<std::string> pretokenize_debug(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;
    std::string decode_single(int32_t id, bool skip_special = false) const;

    // Cut K2.3: Thread-safe <= 64 MiB plain_buf XXH64 LRU cache telemetry & reset
    void cache_stats(uint64_t& hits, uint64_t& misses, size_t& bytes) const noexcept;
    void clear_cache() const;

    // Segment-safe ERNIE-5.5-Flash-Preview chat prompt renderer (100% token-ID exact match to encoding_ernie.py!)
    std::vector<ChatSegment> render_chat_segments(
        const std::vector<ChatMessage>& messages,
        const JsonValue* tools = nullptr,
        bool enable_thinking = true,
        bool add_generation_prompt = true,
        const std::string& reasoning_effort = "high",
        bool parallel_tool_calls = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int32_t pad_id_ = 0;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t im_start_id_ = 7;
    int32_t im_end_id_ = 8;
};

} // namespace ernie55
