// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 GATE-6 Sustained Long-Horizon :3081 DeepSeek Harness Campaign Driver
// Drives baidu/ERNIE-5.5-Flash-Preview (:8052) inside a SINGLE persistent :3081 session_id
// (zero new windows, zero compaction) past 512K -> 1M active context across real C++17 /
// gfx906 HIP / Blackhole TT-Metalium engineering tasks, performs Mid-Campaign Cold .kvblk
// Restart Survival (save -> stop :8052 + VRAM wipe -> cold restart -> <2.0s O_DIRECT restore),
// and resumes the EXACT SAME session_id for 15+ more real engineering tool-call steps to
// clean -Wall -Wextra -Werror build & test pass. Zero Python anywhere.

#include "../src/ernie55_json.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ernie55_g6 {

using ernie55::JsonParser;
using ernie55::JsonValue;
namespace fs = std::filesystem;

struct HttpResp {
    int status = 0;
    std::string body;
};

HttpResp http_request(const std::string& host, int port, const std::string& method,
                      const std::string& path, const std::string& json_body, int timeout_sec = 600) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket() failed");
    timeval tv{};
    tv.tv_sec = timeout_sec;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return {0, ""};
    }

    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Origin: http://" << host << ":" << port << "\r\n"
        << "Connection: close\r\n";
    if (!json_body.empty() || method == "POST") {
        req << "Content-Type: application/json\r\n"
            << "Content-Length: " << json_body.size() << "\r\n";
    }
    req << "\r\n" << json_body;
    const std::string s = req.str();
    size_t sent = 0;
    while (sent < s.size()) {
        const ssize_t n = ::send(fd, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) { ::close(fd); return {0, ""}; }
        sent += size_t(n);
    }

    std::string raw;
    char buf[8192];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, size_t(n));
    }
    ::close(fd);

    HttpResp r{};
    const size_t sp1 = raw.find(' ');
    if (sp1 != std::string::npos) r.status = std::atoi(raw.c_str() + sp1 + 1);
    const size_t he = raw.find("\r\n\r\n");
    if (he == std::string::npos) return r;
    std::string hdr = raw.substr(0, he);
    std::string body = raw.substr(he + 4);
    if (hdr.find("Transfer-Encoding: chunked") != std::string::npos ||
        hdr.find("transfer-encoding: chunked") != std::string::npos) {
        std::string dec;
        size_t p = 0;
        while (p < body.size()) {
            const size_t eol = body.find("\r\n", p);
            if (eol == std::string::npos) break;
            const size_t clen = std::strtoull(body.substr(p, eol - p).c_str(), nullptr, 16);
            p = eol + 2;
            if (clen == 0 || p + clen > body.size()) break;
            dec.append(body, p, clen);
            p += clen + 2;
        }
        r.body = std::move(dec);
    } else {
        r.body = std::move(body);
    }
    return r;
}

JsonValue dsh_rpc(int dsh_port, const std::string& method, const JsonValue& payload, uint64_t& rpc_seq) {
    JsonValue env = JsonValue::make_object();
    env.set("type", JsonValue::make_string("client-request"));
    env.set("rpcId", JsonValue::make_string("g6-" + std::to_string(++rpc_seq)));
    env.set("method", JsonValue::make_string(method));
    env.set("payload", payload);

    const auto resp = http_request("127.0.0.1", dsh_port, "POST", "/api/" + method, env.dumps_compact(), 120);
    if (resp.status != 200) {
        throw std::runtime_error("DSH RPC " + method + " HTTP " + std::to_string(resp.status) + ": " + resp.body);
    }
    const auto j = JsonParser::parse(resp.body);
    if (!j.has("result") || !j.at("result").has("ok") || !j.at("result").at("ok").as_bool()) {
        throw std::runtime_error("DSH RPC " + method + " failed: " + resp.body);
    }
    return j.at("result").has("value") ? j.at("result").at("value") : JsonValue::make_object();
}

JsonValue fe_get(int fe_port, const std::string& path) {
    const auto r = http_request("127.0.0.1", fe_port, "GET", path, "", 30);
    if (r.status != 200) return JsonValue::make_object();
    JsonValue j;
    if (JsonParser::try_parse(r.body, j)) return j;
    return JsonValue::make_object();
}

JsonValue fe_post(int fe_port, const std::string& path, const JsonValue& body) {
    const auto r = http_request("127.0.0.1", fe_port, "POST", path, body.dumps_compact(), 180);
    if (r.status != 200) throw std::runtime_error("FE POST " + path + " HTTP " + std::to_string(r.status) + ": " + r.body);
    return JsonParser::parse(r.body);
}

struct SessionPoll {
    bool running = false;
    int  last_seq = -1;
    int  total_tool_calls = 0;
    int  total_asst_msgs  = 0;
    int  compactions      = 0;
    int  parser_errors    = 0;
    bool repetition_loop  = false;
    std::string last_assistant_text;
};

bool has_repetition_loop(const std::string& s) {
    if (s.size() < 400) return false;
    const size_t win = 48;
    const std::string tail = s.substr(s.size() - win);
    int hits = 0;
    size_t pos = 0;
    while ((pos = s.find(tail, pos)) != std::string::npos) {
        ++hits;
        pos += win;
    }
    return hits >= 6;
}

SessionPoll poll_session(int dsh_port, const std::string& sid, uint64_t& rpc_seq) {
    SessionPoll sp{};
    {
        const auto lst = dsh_rpc(dsh_port, "session.list", JsonValue::make_object(), rpc_seq);
        if (lst.has("items") && lst.at("items").is_array()) {
            for (const auto& it : lst.at("items").arr_val) {
                if (it.has("sessionId") && it.at("sessionId").as_string() == sid) {
                    sp.running = it.has("running") && it.at("running").as_bool();
                    if (it.has("projections") && it.at("projections").has("asOfSeq")) {
                        sp.last_seq = int(it.at("projections").at("asOfSeq").as_int());
                    }
                    break;
                }
            }
        }
    }
    JsonValue hp = JsonValue::make_object();
    hp.set("sessionId", JsonValue::make_string(sid));
    hp.set("maxMessages", JsonValue::make_int(200));
    const auto hist = dsh_rpc(dsh_port, "session.history", hp, rpc_seq);
    if (hist.has("events") && hist.at("events").is_array()) {
        for (const auto& entry : hist.at("events").arr_val) {
            if (!entry.has("event")) continue;
            const auto& ev = entry.at("event");
            const std::string ty = ev.has("type") ? ev.at("type").as_string() : "";
            if (ty == "tool/call") ++sp.total_tool_calls;
            else if (ty == "compaction/start") ++sp.compactions;
            else if (ty == "assistant/message") {
                ++sp.total_asst_msgs;
                if (ev.has("data") && ev.at("data").has("message") && ev.at("data").at("message").has("content") &&
                    ev.at("data").at("message").at("content").is_array()) {
                    std::string txt;
                    for (const auto& blk : ev.at("data").at("message").at("content").arr_val) {
                        if (blk.has("type") && blk.at("type").as_string() == "text" && blk.has("text")) {
                            txt += blk.at("text").as_string();
                        }
                    }
                    if (!txt.empty()) {
                        sp.last_assistant_text = txt;
                        if (txt.find("<ernie:tool_call>") != std::string::npos ||
                            txt.find("<ernie:think>") != std::string::npos) {
                            ++sp.parser_errors;
                        }
                        if (has_repetition_loop(txt)) sp.repetition_loop = true;
                    }
                }
            }
        }
    }
    return sp;
}

struct StepRecord {
    int turn = 0;
    int step = 0;
    std::string phase;
    int context_tokens = 0;
    int delta_tool_calls = 0;
    int cumulative_tool_calls = 0;
    double wall_s = 0.0;
    double decode_tok_s = 0.0;
    double spec_accept_pct = 0.0;
    int compactions = 0;
    int parser_drops = 0;
    bool repetition = false;
};

int run_cmd_rc(const std::string& cmd) {
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

void restart_fe_cold(int fe_port, const std::string& root, int dspark_k) {
    std::printf("[GATE-6] Triggering graceful POST /v1/shutdown on :%d (100%% VRAM wipe)...\n", fe_port);
    std::fflush(stdout);
    (void)fe_post(fe_port, "/v1/shutdown", JsonValue::make_object());
    for (int i = 0; i < 60; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto h = fe_get(fe_port, "/health");
        if (!h.has("status")) break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::printf("[GATE-6] Starting cold :%d frontend (DSpark K=%d)...\n", fe_port, dspark_k);
    std::fflush(stdout);
    const std::string launch =
        "nohup env ERNIE55_PORT=" + std::to_string(fe_port) +
        " ERNIE55_DSPARK_K=" + std::to_string(dspark_k) +
        " bash " + root + "/scripts/start_ernie55_fe.sh >> " + root + "/logs/ernie55_fe_8052.log 2>&1 < /dev/null &";
    const int rc_l = std::system(launch.c_str());
    if (rc_l != 0) std::fprintf(stderr, "[GATE-6] Warning: launch rc=%d\n", rc_l);

    for (int i = 0; i < 300; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        const auto h = fe_get(fe_port, "/health");
        if (h.has("status") && h.at("status").as_string() == "ok") {
            const int sl = h.has("seq_len") ? int(h.at("seq_len").as_int()) : -1;
            std::printf("[GATE-6] Cold :%d READY (seq_len=%d after clean VRAM wipe)!\n", fe_port, sl);
            std::fflush(stdout);
            return;
        }
    }
    throw std::runtime_error("Timed out waiting for cold :8052 restart");
}

} // namespace ernie55_g6

int main(int argc, char** argv) {
    using namespace ernie55_g6;
    std::string root        = ".";
    std::string provider_id = "ernie-5.5-flash-preview-hlut4-1m";
    std::string model_id    = "ERNIE-5.5-Flash-Preview-WHT128-HLUT4";
    int dsh_port            = 3081;
    int fe_port             = 8052;
    int dspark_k            = 5;
    int target_ctx          = 524288;
    int post_restart_target = 16;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg ") + f);
            return argv[++i];
        };
        if (a == "--root")               root = need("--root");
        else if (a == "--dsh-port")      dsh_port = std::stoi(need("--dsh-port"));
        else if (a == "--fe-port")       fe_port = std::stoi(need("--fe-port"));
        else if (a == "--dspark-k")      dspark_k = std::stoi(need("--dspark-k"));
        else if (a == "--target-ctx")    target_ctx = std::stoi(need("--target-ctx"));
        else if (a == "--post-steps")    post_restart_target = std::stoi(need("--post-steps"));
        else if (a == "--provider")      provider_id = need("--provider");
        else if (a == "--model")         model_id = need("--model");
    }

    try {
        const auto t_campaign0 = std::chrono::steady_clock::now();
        uint64_t rpc_seq = 0;

        // 1. Create ONE persistent :3081 session (zero new windows, zero compaction!)
        JsonValue cp = JsonValue::make_object();
        cp.set("cwd", JsonValue::make_string(root));
        cp.set("agentPreset", JsonValue::make_string("standard"));
        const auto cr = dsh_rpc(dsh_port, "session.create", cp, rpc_seq);
        const std::string sid = cr.at("sessionId").as_string();

        JsonValue rp = JsonValue::make_object();
        rp.set("sessionId", JsonValue::make_string(sid));
        rp.set("title", JsonValue::make_string("GATE-6: ERNIE-5.5-Flash-Preview 1M Unbroken C++/HIP/TT-Metal Campaign"));
        (void)dsh_rpc(dsh_port, "session.rename", rp, rpc_seq);

        JsonValue mp = JsonValue::make_object();
        mp.set("sessionId", JsonValue::make_string(sid));
        mp.set("provider", JsonValue::make_string(provider_id));
        mp.set("model", JsonValue::make_string(model_id));
        mp.set("reasoningEffort", JsonValue::make_string("high"));
        (void)dsh_rpc(dsh_port, "session.selectModel", mp, rpc_seq);

        std::printf("[GATE-6] Created SINGLE persistent :%d session_id=%s (%s / %s)\n",
                    dsh_port, sid.c_str(), provider_id.c_str(), model_id.c_str());
        std::printf("[GATE-6] LIVE BROWSER GUI URL: http://127.0.0.1:%d/?session=%s\n",
                    dsh_port, sid.c_str());
        std::fflush(stdout);

        // Reset :8052 KV once at token 0 so we track the exact unbroken session from token 0 -> 512K-1M!
        (void)fe_post(fe_port, "/v1/kv/reset", JsonValue::make_object());

        // Build real multi-phase engineering directives referencing our actual C++17 / gfx906 HIP / Blackhole codebase!
        struct FileSlice { std::string path; int start; int end; std::string focus; };
        std::vector<FileSlice> audit_slices;
        auto add_slices = [&](const std::string& rel, int total_lines, int step_lines, const std::string& focus) {
            for (int s = 1; s <= total_lines; s += step_lines) {
                audit_slices.push_back({rel, s, std::min(total_lines, s + step_lines - 1), focus});
            }
        };
        add_slices("mi50/ernie55_runtime.cpp",       4096, 160, "hybrid TP4 FP16 spine, owner-GPU MoE, 256-slot HCA ring, constant-tile GEMM, warm-prefix snapshot, Dual-NVMe .kvblk");
        add_slices("mi50/ernie55_core_kernels.hip",  4060, 160, "zero-scratch gfx906 HCA/DSA/FlashIndexer/DSpark/mHC HIP kernels");
        add_slices("mi50/ernie55_hlut4_kernels.hip", 1445, 160, "fdot2 WHT128-HLUT4 MoE & weight-stationary BF16/FP16 GEMV<1..8>");
        add_slices("src/ernie55_config.hpp",          503, 170, "Preview 43L+3L config & 192G/224G/240G topology planning");
        add_slices("src/ernie55_dspark.hpp",          343, 175, "Post-Markov MultiMax & lossless probabilistic verification");
        add_slices("src/ernie55_tokenizer.cpp",      1196, 160, "R2L 3-digit BPE, segment-safe chat renderer & ernie5 parser");
        add_slices("p150/p150_ernie55_worker.cpp",   2031, 160, "Blackhole P150A TP2 L1-pinned + GDDR6 45-layer SFPU worker");
        add_slices("tools/ernie55_verify_gates.cpp", 1547, 160, "6-point lossless invariants & GATE-1..4 verification harness");
        add_slices("build/ernie55_hlut4_kernels.s", 27458, 220, "gfx906 LLVM ISA scratch/spill/fdot2 verification");
        add_slices("build/ernie55_core_kernels.s",  47122, 220, "gfx906 LLVM ISA HCA/DSA/FlashIndexer zero-scratch verification");

        std::vector<StepRecord> records;
        int turn_idx = 0;
        int cum_tool_calls = 0;
        int peak_ctx = 0;
        int total_compactions = 0;
        int total_parser_drops = 0;
        bool any_repetition = false;

        auto run_turn = [&](const std::string& phase, const std::string& prompt_text) -> StepRecord {
            ++turn_idx;
            const auto m0 = fe_get(fe_port, "/v1/metrics");
            const uint64_t dec0  = m0.has("total_decode_tokens") ? uint64_t(m0.at("total_decode_tokens").as_int()) : 0ULL;
            const uint64_t acc0  = m0.has("spec_accepted")       ? uint64_t(m0.at("spec_accepted").as_int())       : 0ULL;
            const uint64_t dft0  = m0.has("spec_drafted")        ? uint64_t(m0.at("spec_drafted").as_int())        : 0ULL;
            const auto t0 = std::chrono::steady_clock::now();

            JsonValue part = JsonValue::make_object();
            part.set("type", JsonValue::make_string("text"));
            part.set("text", JsonValue::make_string(prompt_text));
            JsonValue content = JsonValue::make_array();
            content.arr_val.push_back(std::move(part));

            JsonValue pp = JsonValue::make_object();
            pp.set("sessionId", JsonValue::make_string(sid));
            pp.set("mode", JsonValue::make_string("queue"));
            pp.set("content", std::move(content));
            (void)dsh_rpc(dsh_port, "session.prompt", pp, rpc_seq);

            // Wait for :3081 turn to start and finish cleanly
            SessionPoll sp{};
            bool saw_running = false;
            const int prev_asst = records.empty() ? 0 : records.back().step;
            (void)prev_asst;
            for (int w = 0; w < 1800; ++w) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                sp = poll_session(dsh_port, sid, rpc_seq);
                if (sp.running) saw_running = true;
                if (!sp.running && (saw_running || w >= 4)) break;
            }

            const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const auto m1 = fe_get(fe_port, "/v1/metrics");
            const int ctx_now   = m1.has("seq_len")             ? int(m1.at("seq_len").as_int())             : peak_ctx;
            const uint64_t dec1 = m1.has("total_decode_tokens") ? uint64_t(m1.at("total_decode_tokens").as_int()) : dec0;
            const uint64_t acc1 = m1.has("spec_accepted")       ? uint64_t(m1.at("spec_accepted").as_int())       : acc0;
            const uint64_t dft1 = m1.has("spec_drafted")        ? uint64_t(m1.at("spec_drafted").as_int())        : dft0;

            StepRecord rec{};
            rec.turn = turn_idx;
            rec.step = sp.total_tool_calls + sp.total_asst_msgs;
            rec.phase = phase;
            rec.context_tokens = ctx_now;
            rec.delta_tool_calls = std::max(0, sp.total_tool_calls - cum_tool_calls);
            rec.cumulative_tool_calls = sp.total_tool_calls;
            cum_tool_calls = sp.total_tool_calls;
            rec.wall_s = wall_s;
            rec.decode_tok_s = (wall_s > 0.05 && dec1 > dec0) ? (double(dec1 - dec0) / wall_s) : 0.0;
            rec.spec_accept_pct = (dft1 > dft0) ? (100.0 * double(acc1 - acc0) / double(dft1 - dft0)) : 0.0;
            rec.compactions = sp.compactions;
            rec.parser_drops = sp.parser_errors;
            rec.repetition = sp.repetition_loop;

            peak_ctx = std::max(peak_ctx, ctx_now);
            total_compactions = std::max(total_compactions, sp.compactions);
            total_parser_drops = std::max(total_parser_drops, sp.parser_errors);
            if (sp.repetition_loop) any_repetition = true;

            records.push_back(rec);
            std::printf("[GATE-6] Turn %3d (%s) | steps=%3d (+%d tools) | ctx=%7d tok | wall=%.1fs | eff=%.1f tok/s | spec=%.1f%% | comp=%d drops=%d\n",
                        rec.turn, phase.c_str(), rec.step, rec.delta_tool_calls, rec.context_tokens,
                        rec.wall_s, rec.decode_tok_s, rec.spec_accept_pct, rec.compactions, rec.parser_drops);
            std::fflush(stdout);
            return rec;
        };

        // ---- PHASE A & B: Sustained Engineering Campaign Growing Active Context Past target_ctx (512K+) ----
        // Step 1: Kickoff + initial topology & GATE-4 hotspot inspection
        run_turn("pre_restart",
            "You are executing GATE-6 (sustained unbroken 1M C++17/gfx906 HIP/Blackhole TT-Metalium engineering campaign in a single persistent session). "
            "Use your read and bash tools to inspect config/topology_192g.json, config/topology_224g.json, config/topology_240g.json, and artifacts/gate4_ernie55_preview_hlut4_1m.json. "
            "Summarize the exact 45-layer expert splits across 5x MI50 + 2x P150A and identify the top 2 latency hotspots from GATE-4. Be concise.");

        // Interleave real code authoring (tools/ernie55_topo_tp3_audit.cpp) + multi-slice deep source/ISA audits until ctx >= target_ctx!
        size_t slice_idx = 0;
        bool authored_verifier = false;
        while (peak_ctx < target_ctx && slice_idx < audit_slices.size()) {
            if (!authored_verifier && peak_ctx >= 120000) {
                authored_verifier = true;
                run_turn("pre_restart",
                    "Using your write and bash tools, create a pure-C++17 SPDX Apache-2.0 standalone verifier at tools/ernie55_topo_tp3_audit.cpp "
                    "that loads config/topology_192g.json, config/topology_224g.json, and config/topology_240g.json via src/ernie55_config.hpp, "
                    "verifies 100% disjoint expert coverage [0..512) with 0 B DDR4 spill for all 3 topologies, checks 45-layer uniform HLUT4 footprint, "
                    "and exits 0 on PASS. Compile it with: g++ -std=c++17 -O3 -Wall -Wextra -Werror -ffile-prefix-map=.=. -Isrc tools/ernie55_topo_tp3_audit.cpp -o bin/ernie55_topo_tp3_audit "
                    "and run ./bin/ernie55_topo_tp3_audit. Report the exact output.");
                continue;
            }
            // Batch 3 real source slices per engineering audit turn (~6K-8K real source tokens per turn, zero pruning!)
            std::ostringstream p;
            p << "Continue the deep C++17/HIP/Blackhole engine audit in this same session. "
              << "Use the read tool to inspect each of the following 3 exact source slices (use start/limit lines so each read stays crisp) and verify memory safety, zero-scratch invariants, and 128-boundary exactness:\n";
            for (int k = 0; k < 3 && slice_idx < audit_slices.size(); ++k, ++slice_idx) {
                const auto& sl = audit_slices[slice_idx];
                p << "  " << (k + 1) << ". " << sl.path << " lines " << sl.start << "-" << sl.end << " (focus: " << sl.focus << ")\n";
            }
            p << "Reply with a concise 4-line bullet summary of verified invariants and exact symbol names seen.";
            run_turn("pre_restart", p.str());
        }

        // Ensure verifier is authored before cold restart if we reached target_ctx fast
        if (!authored_verifier) {
            authored_verifier = true;
            run_turn("pre_restart",
                "Using your write and bash tools, create a pure-C++17 SPDX Apache-2.0 standalone verifier at tools/ernie55_topo_tp3_audit.cpp "
                "that loads config/topology_192g.json, config/topology_224g.json, and config/topology_240g.json via src/ernie55_config.hpp, "
                "verifies 100% disjoint expert coverage [0..512) with 0 B DDR4 spill across all 45 layers, compiles cleanly with "
                "g++ -std=c++17 -O3 -Wall -Wextra -Werror -ffile-prefix-map=.=. -Isrc tools/ernie55_topo_tp3_audit.cpp -o bin/ernie55_topo_tp3_audit, "
                "and passes ./bin/ernie55_topo_tp3_audit. Report exact output.");
        }

        // ---- MID-CAMPAIGN COLD .KVBLK RESTART SURVIVAL (>= 512K CONTEXT!) ----
        const int restart_turn = turn_idx;
        const int restart_step = records.empty() ? 0 : records.back().step;
        const std::string kv_path = root + "/kvcache/gate6_mid_campaign.kvblk";
        fs::create_directories(root + "/kvcache");

        std::printf("[GATE-6] === MID-CAMPAIGN COLD .KVBLK RESTART AT TURN %d (ctx=%d tok) ===\n", restart_turn, peak_ctx);
        std::fflush(stdout);
        JsonValue sp_save = JsonValue::make_object();
        sp_save.set("path", JsonValue::make_string(kv_path));
        const auto kv_sv = fe_post(fe_port, "/v1/kv/save", sp_save);
        const double kv_save_ms = kv_sv.at("total_ms").as_double();
        const int    kv_seq_len = int(kv_sv.at("seq_len").as_int());
        const uint64_t kv_bytes = uint64_t(kv_sv.at("bytes").as_int());
        std::printf("[GATE-6] Saved live session .kvblk: seq_len=%d (%.2f GiB) in %.1f ms (%.2f GB/s)\n",
                    kv_seq_len, double(kv_bytes) / 1073741824.0, kv_save_ms, kv_sv.at("gb_s").as_double());
        std::fflush(stdout);

        restart_fe_cold(fe_port, root, dspark_k);

        JsonValue sp_rst = JsonValue::make_object();
        sp_rst.set("path", JsonValue::make_string(kv_path));
        const auto kv_ld = fe_post(fe_port, "/v1/kv/restore", sp_rst);
        const double kv_restore_ms  = kv_ld.at("total_ms").as_double();
        const double kv_restore_gbs = kv_ld.at("gb_s").as_double();
        const double kv_restore_tps = kv_ld.at("tok_s").as_double();
        const int    kv_restored_sl = int(kv_ld.at("seq_len").as_int());
        std::printf("[GATE-6] COLD .kvblk RESTORE: seq_len=%d in %.1f ms (%.2f GB/s, %.0f tok/s E2E) — RESUMING EXACT SAME session_id=%s!\n",
                    kv_restored_sl, kv_restore_ms, kv_restore_gbs, kv_restore_tps, sid.c_str());
        std::fflush(stdout);

        // ---- PHASE C: Post-Restart Continuation on EXACT SAME session_id (16+ real engineering steps!) ----
        const std::vector<std::string> post_prompts = {
            "We just performed a cold Dual-NVMe .kvblk save, full :8052 VRAM wipe, and cold O_DIRECT restore in this EXACT SAME session. "
            "Use bash to verify ./bin/ernie55_topo_tp3_audit still exists and passes, and also state the exact expert splits you read earlier in Phase A for 192G, 224G, and 240G.",
            "Use bash to run `llvm-objdump` / ISA grep check on build/ernie55_hlut4_kernels.s to count all kernels, confirm ScratchSize == 0, 0 spills, 0 trig, 0 slc, and count v_dot2_f32_f16 instructions.",
            "Use bash to run the exact same ISA scratch/spill/trig/slc audit on build/ernie55_core_kernels.s and verify all 101 core kernels have ScratchSize == 0.",
            "Use read to inspect tools/ernie55_topo_tp3_audit.cpp and identify 3 mathematical invariants we should add (mHC Sinkhorn doubly-stochastic error < 1e-5, HCA 128-boundary safe_k guard for all pos%128 in 0..127, and P150A L1 330 MiB budget).",
            "Use edit or write to add the mHC Sinkhorn 20-iteration doubly-stochastic invariant test into tools/ernie55_topo_tp3_audit.cpp, then use bash to recompile with -Wall -Wextra -Werror and run it.",
            "Use edit or write to add the HCA 128-boundary guard invariant (pos%128 + safe_k <= 126 for all 0..127) and 256-slot SWA ring non-collision proof into tools/ernie55_topo_tp3_audit.cpp, recompile -Wall -Wextra -Werror, and run it.",
            "Use edit or write to add a 224G and 240G TP3 VRAM headroom ledger check (confirming 100% 45-layer uniform HLUT4 + 1M BF16 KV fits with 0 B DDR4 spill) into tools/ernie55_topo_tp3_audit.cpp, recompile, and run it.",
            "Use bash to run `./bin/ernie55_verify_unit` (GATE-0) and report the exact 6/6 test suite output.",
            "Use read to inspect artifacts/gate1_ernie55_preview_hlut4_1m.json and verify all 6 lossless invariants (1a..1d) are PASS.",
            "Use read to inspect artifacts/gate2_ernie55_preview_hlut4_1m.json and compare the measured Mean KL nats and Top-1% against frozen Hy4-flash (<=0.045 nats) and Hy4-770B (0.0558 nats).",
            "Use read to inspect artifacts/gate3_ernie55_preview_hlut4_1m.json and report GSM8K, HumanEval pass@1, 20/20 protocol, and 18/18 1M needle results.",
            "Use read to inspect artifacts/gate4_ernie55_preview_hlut4_1m.json and confirm 262K/524K/1M .kvblk memcmp_diff_bytes == 0 and peak DSpark decode throughput.",
            "Use bash to check PCIe AER counters across /sys/bus/pci/devices/*/aer_dev_{nonfatal,fatal} and confirm 0/0 errors plus 0 Python (.py) files in ..",
            "Use bash to check `make audit-gfx906` and confirm 100% pass with zero scratch and zero spills.",
            "Use bash to verify `./bin/ernie55_topo_tp3_audit` passes cleanly and verify zero leaked paths via `./bin/ernie55_sanitize_verifier ./bin --mode workspace || true`.",
            "Use bash to inspect `ls -lh artifacts/` and, if `artifacts/ernie55_preview_hlut4_1m_telemetry.png` exists, run `gemini-api analyze -i artifacts/ernie55_preview_hlut4_1m_telemetry.png 'Audit layout legibility in 2 sentences'`; otherwise report the exact artifact checklist ready for Phase 6 chart generation. Conclude with a concise final GATE-6 sign-off."
        };

        int post_turns_done = 0;
        for (size_t i = 0; i < post_prompts.size() || post_turns_done < post_restart_target; ++i) {
            const std::string& pr = (i < post_prompts.size()) ? post_prompts[i]
                : "Use bash to run `./bin/ernie55_topo_tp3_audit && ./bin/ernie55_verify_unit` and report final exit status.";
            run_turn("post_restart", pr);
            ++post_turns_done;
            if (i + 1 >= post_prompts.size() && post_turns_done >= post_restart_target) break;
        }

        // Independent host verification of final build & test pass!
        const int rc_build = run_cmd_rc(
            "g++ -std=c++17 -O3 -Wall -Wextra -Werror -ffile-prefix-map=" + root + "=. -I" + root + "/src " +
            root + "/tools/ernie55_topo_tp3_audit.cpp -o " + root + "/bin/ernie55_topo_tp3_audit 2>&1");
        const int rc_run = (rc_build == 0) ? run_cmd_rc(root + "/bin/ernie55_topo_tp3_audit") : 1;
        const bool final_build_pass = (rc_build == 0 && rc_run == 0);

        // Clean up intermediate .kvblk files after survival verification
        std::error_code ec;
        fs::remove(kv_path, ec);
        fs::remove(kv_path + ".toks", ec);
        fs::remove("/var/tmp/ernie55_kvcache/" + fs::path(kv_path).filename().string() + ".s1", ec);

        const double total_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_campaign0).count();
        const int total_steps = records.empty() ? 0 : records.back().step;
        const int post_steps  = total_steps - restart_step;
        const bool pass = (peak_ctx >= target_ctx) &&
                          (kv_restored_sl == kv_seq_len) &&
                          (kv_restore_ms < 2000.0) &&
                          (post_turns_done >= 15) &&
                          (total_compactions == 0) &&
                          (total_parser_drops == 0) &&
                          (!any_repetition) &&
                          final_build_pass;

        // Write sanitized receipt artifacts/gate6_long_horizon_1m_campaign.json (zero PII!)
        {
            std::ofstream ofs(root + "/artifacts/gate6_long_horizon_1m_campaign.json", std::ios::trunc);
            ofs << "{\n"
                << "  \"schema\": \"ernie55_preview_gate6_v1\",\n"
                << "  \"harness_port\": " << dsh_port << ",\n"
                << "  \"frontend_port\": " << fe_port << ",\n"
                << "  \"persistent_session_count\": 1,\n"
                << "  \"session_id\": \"" << sid << "\",\n"
                << "  \"total_turns\": " << turn_idx << ",\n"
                << "  \"total_steps\": " << total_steps << ",\n"
                << "  \"total_tool_calls\": " << cum_tool_calls << ",\n"
                << "  \"peak_context_tokens\": " << peak_ctx << ",\n"
                << "  \"cold_restart_at_turn\": " << restart_turn << ",\n"
                << "  \"cold_restart_at_step\": " << restart_step << ",\n"
                << "  \"cold_save_seq_len\": " << kv_seq_len << ",\n"
                << "  \"cold_save_bytes\": " << kv_bytes << ",\n"
                << "  \"cold_save_ms\": " << std::fixed << std::setprecision(2) << kv_save_ms << ",\n"
                << "  \"cold_restore_seq_len\": " << kv_restored_sl << ",\n"
                << "  \"cold_restore_ms\": " << kv_restore_ms << ",\n"
                << "  \"cold_restore_gb_s\": " << kv_restore_gbs << ",\n"
                << "  \"cold_restore_tok_s\": " << std::setprecision(0) << kv_restore_tps << ",\n"
                << "  \"post_restart_turns\": " << post_turns_done << ",\n"
                << "  \"post_restart_steps\": " << post_steps << ",\n"
                << "  \"compaction_events\": " << total_compactions << ",\n"
                << "  \"parser_drops\": " << total_parser_drops << ",\n"
                << "  \"repetition_loops\": " << (any_repetition ? 1 : 0) << ",\n"
                << "  \"final_build_pass\": " << (final_build_pass ? "true" : "false") << ",\n"
                << "  \"wall_s\": " << std::setprecision(1) << total_wall_s << ",\n"
                << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
                << "  \"steps\": [\n";
            for (size_t i = 0; i < records.size(); ++i) {
                const auto& r = records[i];
                if (i > 0) ofs << ",\n";
                ofs << "    {\"turn\": " << r.turn << ", \"step\": " << r.step
                    << ", \"phase\": \"" << r.phase << "\", \"context_tokens\": " << r.context_tokens
                    << ", \"delta_tool_calls\": " << r.delta_tool_calls
                    << ", \"cumulative_tool_calls\": " << r.cumulative_tool_calls
                    << ", \"wall_s\": " << std::fixed << std::setprecision(2) << r.wall_s
                    << ", \"decode_tok_s\": " << r.decode_tok_s
                    << ", \"spec_accept_pct\": " << r.spec_accept_pct
                    << ", \"compactions\": " << r.compactions
                    << ", \"parser_drops\": " << r.parser_drops << "}";
            }
            ofs << "\n  ]\n}\n";
        }

        std::printf("[GATE-6] === CAMPAIGN COMPLETE: %s | turns=%d steps=%d tools=%d | peak_ctx=%d | cold_restore=%.1fms | post_turns=%d | build=%d ===\n",
                    pass ? "PASS" : "FAIL", turn_idx, total_steps, cum_tool_calls, peak_ctx,
                    kv_restore_ms, post_turns_done, int(final_build_pass));
        return pass ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[GATE-6] FATAL: %s\n", ex.what());
        return 1;
    }
}
