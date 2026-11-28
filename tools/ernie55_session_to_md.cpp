// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 DeepSeek Harness Sanitized Session JSONL -> Readable Markdown Replay Renderer.
// Zero Python anywhere. Compiles cleanly under -std=c++17 -O3 -Wall -Wextra -Werror.

#include "../src/ernie55_json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

std::string clip_text(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) return s;
    const size_t head = max_chars * 2u / 3u;
    const size_t tail = max_chars - head;
    return s.substr(0, head) + "\n... [truncated " + std::to_string(s.size() - max_chars) +
           " chars; full verbatim payload in session.jsonl] ...\n" +
           s.substr(s.size() - tail);
}

struct StepRecord {
    int turn = 1;
    int step = 0;
    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    int64_t cache_read_tokens = 0;
    int64_t t_start_ms = 0;
    int64_t t_end_ms = 0;
    std::string reasoning;
    std::string text;
    struct ToolCall {
        std::string id;
        std::string name;
        std::string arguments;
        std::string result;
    };
    std::vector<ToolCall> calls;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <sanitized_session.jsonl> <output_replay.md>\n", argv[0]);
        return 2;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];

    std::ifstream ifs(in_path);
    if (!ifs) {
        std::fprintf(stderr, "[ERROR] Cannot open %s\n", in_path.c_str());
        return 1;
    }

    std::string session_id;
    std::string title;
    std::string user_prompt;
    std::string finish_reason = "completed";
    int64_t created_ms = 0;
    int64_t ended_ms = 0;

    std::map<std::pair<int, int>, StepRecord> steps;
    std::map<std::string, std::pair<std::pair<int, int>, size_t>> call_index;

    std::string line;
    size_t line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        if (line.empty()) continue;
        // Fast pre-filter: skip high-volume streaming chunk events except usage!
        if (line.find("\"type\":\"assistant/chunk\"") != std::string::npos &&
            line.find("\"type\":\"usage\"") == std::string::npos) {
            continue;
        }
        ernie55::JsonValue obj;
        try {
            obj = ernie55::parse_json(line);
        } catch (...) {
            continue;
        }
        if (!obj.is_object()) continue;
        const std::string typ = obj.has("type") ? obj.get("type").as_str() : "";
        const int64_t ts = obj.has("time") ? obj.get("time").as_int() : 0;

        if (typ == "session") {
            if (obj.has("id")) session_id = obj.get("id").as_str();
            if (obj.has("createdAt")) created_ms = obj.get("createdAt").as_int();
        } else if (typ == "session/title" && obj.has("data")) {
            const auto& d = obj.get("data");
            if (d.has("title")) title = d.get("title").as_str();
        } else if (typ == "user/message" && user_prompt.empty() && obj.has("data")) {
            const auto& d = obj.get("data");
            if (d.has("content") && d.get("content").is_array()) {
                for (const auto& c : d.get("content").arr_val) {
                    if (c.is_object() && c.has("type") && c.get("type").as_str() == "text" && c.has("text")) {
                        const std::string t = c.get("text").as_str();
                        if (t.find("Current runtime context.") != 0) {
                            user_prompt = t;
                            break;
                        }
                    }
                }
            }
        } else if (typ == "step/start" && obj.has("data")) {
            const auto& d = obj.get("data");
            const int turn = int(d.has("turn") ? d.get("turn").as_int(1) : 1);
            const int step = int(d.has("step") ? d.get("step").as_int(0) : 0);
            auto& rec = steps[{turn, step}];
            rec.turn = turn;
            rec.step = step;
            rec.t_start_ms = ts;
        } else if (typ == "step/end" && obj.has("data")) {
            const auto& d = obj.get("data");
            const int turn = int(d.has("turn") ? d.get("turn").as_int(1) : 1);
            const int step = int(d.has("step") ? d.get("step").as_int(0) : 0);
            auto& rec = steps[{turn, step}];
            rec.turn = turn;
            rec.step = step;
            rec.t_end_ms = ts;
        } else if (typ == "turn/end" && obj.has("data")) {
            ended_ms = ts;
            const auto& d = obj.get("data");
            if (d.has("reason") && d.get("reason").is_object() && d.get("reason").has("kind")) {
                finish_reason = d.get("reason").get("kind").as_str();
            }
        } else if (typ == "assistant/chunk" && obj.has("data")) {
            const auto& d = obj.get("data");
            const int turn = int(d.has("turn") ? d.get("turn").as_int(1) : 1);
            const int step = int(d.has("step") ? d.get("step").as_int(0) : 0);
            if (d.has("chunk") && d.get("chunk").is_object()) {
                const auto& ch = d.get("chunk");
                if (ch.has("type") && ch.get("type").as_str() == "usage" && ch.has("usage") && ch.get("usage").is_object()) {
                    const auto& u = ch.get("usage");
                    auto& rec = steps[{turn, step}];
                    rec.turn = turn;
                    rec.step = step;
                    if (u.has("inputTokens")) rec.input_tokens = u.get("inputTokens").as_int();
                    if (u.has("outputTokens")) rec.output_tokens = u.get("outputTokens").as_int();
                    if (u.has("cacheReadTokens")) rec.cache_read_tokens = u.get("cacheReadTokens").as_int();
                }
            }
        } else if (typ == "assistant/message" && obj.has("data")) {
            const auto& d = obj.get("data");
            const int turn = int(d.has("turn") ? d.get("turn").as_int(1) : 1);
            const int step = int(d.has("step") ? d.get("step").as_int(0) : 0);
            auto& rec = steps[{turn, step}];
            rec.turn = turn;
            rec.step = step;
            if (d.has("message") && d.get("message").is_object() &&
                d.get("message").has("content") && d.get("message").get("content").is_array()) {
                for (const auto& blk : d.get("message").get("content").arr_val) {
                    if (!blk.is_object() || !blk.has("type")) continue;
                    const std::string bt = blk.get("type").as_str();
                    if (bt == "reasoning" && blk.has("text")) {
                        if (!rec.reasoning.empty()) rec.reasoning += "\n";
                        rec.reasoning += blk.get("text").as_str();
                    } else if (bt == "text" && blk.has("text")) {
                        if (!rec.text.empty()) rec.text += "\n";
                        rec.text += blk.get("text").as_str();
                    } else if (bt == "tool-call") {
                        StepRecord::ToolCall tc;
                        if (blk.has("id")) tc.id = blk.get("id").as_str();
                        if (blk.has("name")) tc.name = blk.get("name").as_str();
                        if (blk.has("arguments")) tc.arguments = blk.get("arguments").as_str();
                        const size_t idx = rec.calls.size();
                        rec.calls.push_back(std::move(tc));
                        if (!rec.calls[idx].id.empty()) {
                            call_index[rec.calls[idx].id] = {{turn, step}, idx};
                        }
                    }
                }
            }
        } else if (typ == "tool/result" && obj.has("data")) {
            const auto& d = obj.get("data");
            if (d.has("message") && d.get("message").is_object() &&
                d.get("message").has("content") && d.get("message").get("content").is_array()) {
                for (const auto& blk : d.get("message").get("content").arr_val) {
                    if (!blk.is_object() || !blk.has("toolCallId")) continue;
                    const std::string cid = blk.get("toolCallId").as_str();
                    auto it = call_index.find(cid);
                    if (it == call_index.end()) continue;
                    std::string rtxt;
                    if (blk.has("content") && blk.get("content").is_array()) {
                        for (const auto& item : blk.get("content").arr_val) {
                            if (item.is_object() && item.has("text")) {
                                if (!rtxt.empty()) rtxt += "\n";
                                rtxt += item.get("text").as_str();
                            }
                        }
                    }
                    steps[it->second.first].calls[it->second.second].result = std::move(rtxt);
                }
            }
        }
    }

    int64_t peak_ctx = 0;
    int64_t tot_out = 0;
    int64_t tot_calls = 0;
    for (const auto& kv : steps) {
        const auto& s = kv.second;
        peak_ctx = std::max(peak_ctx, s.input_tokens + s.cache_read_tokens + s.output_tokens);
        tot_out += s.output_tokens;
        tot_calls += int64_t(s.calls.size());
    }

    std::ofstream ofs(out_path);
    if (!ofs) {
        std::fprintf(stderr, "[ERROR] Cannot write %s\n", out_path.c_str());
        return 1;
    }
    ofs << "# ERNIE-5.5-Flash-Preview-WHT128-HLUT4 Autonomous Co-Engineering Transcript Replay\n\n";
    ofs << "- **Session ID:** `" << session_id << "`\n";
    if (!title.empty()) ofs << "- **Title:** " << title << "\n";
    ofs << "- **Model:** `ERNIE-5.5-Flash-Preview-WHT128-HLUT4` (300B total / 14B active, 4.125 bpw, Apache-2.0)\n";
    ofs << "- **Hardware:** `SB-M5-P2-192G` (5x AMD Radeon Instinct MI50 128 GiB HBM2 + 2x Tenstorrent Blackhole P150A 64 GiB GDDR6 + AMD EPYC 7542)\n";
    ofs << "- **Steps:** `" << steps.size() << "` | **Tool Calls:** `" << tot_calls
        << "` | **Peak Context:** `" << peak_ctx << "` tokens | **Total Generated Output:** `" << tot_out
        << "` tokens | **Status:** `" << finish_reason << "`\n";
    if (created_ms > 0 && ended_ms > created_ms) {
        const double mins = double(ended_ms - created_ms) / 60000.0;
        char tbuf[64];
        std::snprintf(tbuf, sizeof(tbuf), "%.1f minutes", mins);
        ofs << "- **Wall Duration:** `" << tbuf << "`\n";
    }
    ofs << "\n## Initial Prompt\n\n```text\n" << user_prompt << "\n```\n\n";
    ofs << "## Step Telemetry Summary\n\n";
    ofs << "| Step | Input Tokens | Cache Read | Output Tokens | Tool Calls | Step Wall (s) |\n";
    ofs << "|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& kv : steps) {
        const auto& s = kv.second;
        const double sec = (s.t_end_ms > s.t_start_ms) ? double(s.t_end_ms - s.t_start_ms) / 1000.0 : 0.0;
        char row[256];
        std::snprintf(row, sizeof(row), "| %d | %lld | %lld | %lld | %zu | %.1f |\n",
                      s.step, (long long)s.input_tokens, (long long)s.cache_read_tokens,
                      (long long)s.output_tokens, s.calls.size(), sec);
        ofs << row;
    }
    ofs << "\n## Step-by-Step Replay\n\n";
    for (const auto& kv : steps) {
        const auto& s = kv.second;
        ofs << "### Step " << s.step << " (`in=" << s.input_tokens << "`, `cache_read="
            << s.cache_read_tokens << "`, `out=" << s.output_tokens << "`)\n\n";
        if (!s.reasoning.empty()) {
            ofs << "<details>\n<summary>Reasoning (" << s.reasoning.size() << " chars)</summary>\n\n```text\n"
                << clip_text(s.reasoning, 4000) << "\n```\n\n</details>\n\n";
        }
        if (!s.text.empty()) {
            ofs << s.text << "\n\n";
        }
        for (const auto& tc : s.calls) {
            ofs << "**Tool Call:** `" << tc.name << "` (`" << tc.id << "`)\n\n```json\n"
                << clip_text(tc.arguments, 2000) << "\n```\n\n";
            if (!tc.result.empty()) {
                ofs << "<details>\n<summary>Tool Output (" << tc.result.size() << " chars)</summary>\n\n```text\n"
                    << clip_text(tc.result, 2500) << "\n```\n\n</details>\n\n";
            }
        }
    }
    ofs.close();
    std::printf("[OK] Rendered %zu steps (%lld peak ctx, %lld output toks, %lld tool calls) -> %s\n",
                steps.size(), (long long)peak_ctx, (long long)tot_out, (long long)tot_calls, out_path.c_str());
    return 0;
}
