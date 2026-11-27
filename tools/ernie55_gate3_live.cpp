// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Live :3081 DeepSeek Harness Web GUI GATE-3 Suite Driver (Zero Headless Bypass, Zero Python).
// Creates ONE visible :3081 session and executes all 4 official GATE-3 suites visibly through :3081 -> :8052:
//   1. GSM8K 0-Shot (N=100, thinking=on / reasoningEffort=medium)
//   2. HumanEval (N=164, pass@1 via zero-file POSIX pipe, thinking=off)
//   3. 20/20 Protocol Audit (10 live structured tool calls + 10 enable_thinking=false immediate-answer probes)
//   4. 18/18 Real-Codebase Multi-Needle Synthesis monotonically up to 1M context (4K, 16K, 64K, 256K, 512K, ~1M)
// Writes ./artifacts/gate3_ernie55_preview_hlut4_1m.json with session_id receipt.

#include "ernie55_json.hpp"
#include "ernie55_tokenizer.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
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

namespace ernie55_g3 {

using ernie55::Ernie55Tokenizer;
using ernie55::JsonParser;
using ernie55::JsonValue;
namespace fs = std::filesystem;

struct HttpResp {
    int status = 0;
    std::string body;
};

HttpResp http_request(const std::string& host, int port, const std::string& method,
                      const std::string& path, const std::string& json_body, int timeout_sec = 300) {
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
    char buf[16384];
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
    env.set("rpcId", JsonValue::make_string("g3-" + std::to_string(++rpc_seq)));
    env.set("method", JsonValue::make_string(method));
    env.set("payload", payload);
    const auto resp = http_request("127.0.0.1", dsh_port, "POST", "/api/" + method, env.dumps_compact(), 180);
    if (resp.status != 200) {
        throw std::runtime_error("DSH RPC " + method + " HTTP " + std::to_string(resp.status) + ": " + resp.body.substr(0, 300));
    }
    const auto j = JsonParser::parse(resp.body);
    if (!j.has("result") || !j.at("result").has("ok") || !j.at("result").at("ok").as_bool()) {
        throw std::runtime_error("DSH RPC " + method + " failed: " + resp.body.substr(0, 300));
    }
    return j.at("result").has("value") ? j.at("result").at("value") : JsonValue::make_object();
}

std::string read_file_str(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string extract_gsm8k_num(const std::string& s) {
    std::string sub = s;
    const auto bp = s.rfind("\\boxed{");
    if (bp != std::string::npos) {
        const auto be = s.find('}', bp + 7);
        if (be != std::string::npos) sub = s.substr(bp + 7, be - (bp + 7));
    } else {
        const auto hp = s.rfind("####");
        if (hp != std::string::npos) sub = s.substr(hp + 4);
    }
    std::string best;
    for (size_t i = 0; i < sub.size(); ) {
        if ((sub[i] >= '0' && sub[i] <= '9') ||
            (sub[i] == '-' && i + 1 < sub.size() && sub[i + 1] >= '0' && sub[i + 1] <= '9')) {
            std::string cur;
            if (sub[i] == '-') cur.push_back(sub[i++]);
            while (i < sub.size() && ((sub[i] >= '0' && sub[i] <= '9') || sub[i] == ',' || sub[i] == '.')) {
                if (sub[i] != ',') cur.push_back(sub[i]);
                ++i;
            }
            if (!cur.empty()) best = cur;
        } else {
            ++i;
        }
    }
    if (best.find('.') != std::string::npos) {
        while (!best.empty() && best.back() == '0') best.pop_back();
        if (!best.empty() && best.back() == '.') best.pop_back();
    }
    return best;
}

bool eval_humaneval_candidate_pipe(const std::string& code) {
    int pfd[2];
    if (::pipe(pfd) != 0) return false;
    const pid_t pid = ::fork();
    if (pid < 0) { ::close(pfd[0]); ::close(pfd[1]); return false; }
    if (pid == 0) {
        ::close(pfd[1]);
        ::dup2(pfd[0], STDIN_FILENO);
        ::close(pfd[0]);
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
        char* const argv[] = {
            const_cast<char*>("timeout"),
            const_cast<char*>("3"),
            const_cast<char*>("python3"),
            const_cast<char*>("-"),
            nullptr
        };
        ::execvp("timeout", argv);
        ::_exit(127);
    }
    ::close(pfd[0]);
    size_t off = 0;
    while (off < code.size()) {
        const ssize_t w = ::write(pfd[1], code.data() + off, code.size() - off);
        if (w <= 0) break;
        off += size_t(w);
    }
    ::close(pfd[1]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::string extract_python_code(const std::string& raw, const std::string& prompt, const std::string& entry_point) {
    std::string body = raw;
    const auto f0 = raw.find("```");
    if (f0 != std::string::npos) {
        size_t s = raw.find('\n', f0 + 3);
        if (s != std::string::npos) {
            ++s;
            const auto f1 = raw.find("```", s);
            body = (f1 != std::string::npos) ? raw.substr(s, f1 - s) : raw.substr(s);
        }
    }
    if (body.find("def " + entry_point) != std::string::npos) {
        std::string imports;
        std::istringstream iss(prompt);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.rfind("from ", 0) == 0 || line.rfind("import ", 0) == 0) imports += line + "\n";
        }
        return imports + "\n" + body + "\n";
    }
    return prompt + body + "\n";
}

std::vector<std::string> load_real_codebase_lines(const std::string& root_dir) {
    std::vector<std::string> src_paths;
    const char* subdirs[4] = {"/src", "/mi50", "/p150", "/tools"};
    std::error_code ec;
    for (const char* sd : subdirs) {
        const fs::path dir(root_dir + sd);
        if (!fs::is_directory(dir, ec)) continue;
        for (const auto& ent : fs::directory_iterator(dir, ec)) {
            if (!ent.is_regular_file(ec)) continue;
            const std::string ext = ent.path().extension().string();
            if (ext == ".hpp" || ext == ".h" || ext == ".cpp" || ext == ".hip") {
                src_paths.push_back(ent.path().string());
            }
        }
    }
    std::sort(src_paths.begin(), src_paths.end());
    std::vector<std::string> lines;
    for (const auto& sp : src_paths) {
        const std::string body = read_file_str(sp);
        if (body.empty()) continue;
        lines.push_back("// === MODULE: " + fs::path(sp).filename().string() + " ===");
        std::istringstream iss(body);
        std::string ln;
        while (std::getline(iss, ln)) lines.push_back(ln);
    }
    if (lines.empty()) lines.push_back("// fallback C++17 codebase line");
    return lines;
}

struct TurnOutcome {
    std::string reasoning;
    std::string text;
    std::vector<std::string> tool_names;
    int valid_tool_json = 0;
    int pressure_tokens = 0;
    double wall_ms = 0.0;
};

struct LiveHarnessClient {
    int dsh_port = 3081;
    std::string sid;
    uint64_t rpc_seq = 0;

    void select_model(const std::string& effort) {
        JsonValue p = JsonValue::make_object();
        p.set("sessionId", JsonValue::make_string(sid));
        p.set("provider", JsonValue::make_string("ernie-5.5-flash-preview-hlut4-1m"));
        p.set("model", JsonValue::make_string("ERNIE-5.5-Flash-Preview-WHT128-HLUT4"));
        p.set("reasoningEffort", JsonValue::make_string(effort));
        (void)dsh_rpc(dsh_port, "session.selectModel", p, rpc_seq);
    }

    int get_pressure_and_seq(int& out_seq, bool& out_running) {
        const auto lst = dsh_rpc(dsh_port, "session.list", JsonValue::make_object(), rpc_seq);
        out_running = false;
        out_seq = 0;
        int pt = 0;
        if (lst.has("items") && lst.at("items").is_array()) {
            for (const auto& it : lst.at("items").arr_val) {
                if (it.has("sessionId") && it.at("sessionId").as_string() == sid) {
                    out_running = it.has("running") && it.at("running").as_bool();
                    if (it.has("projections")) {
                        const auto& pr = it.at("projections");
                        if (pr.has("asOfSeq")) out_seq = int(pr.at("asOfSeq").as_int());
                        if (pr.has("values") && pr.at("values").has("contextPressure") &&
                            pr.at("values").at("contextPressure").has("pressureTokens")) {
                            pt = int(pr.at("values").at("contextPressure").at("pressureTokens").as_int());
                        }
                    }
                    break;
                }
            }
        }
        return pt;
    }

    TurnOutcome prompt_and_wait(const std::string& prompt_text, int max_wait_sec = 3600) {
        int seq_before = 0;
        bool running = false;
        (void)get_pressure_and_seq(seq_before, running);

        JsonValue item = JsonValue::make_object();
        item.set("type", JsonValue::make_string("text"));
        item.set("text", JsonValue::make_string(prompt_text));
        JsonValue content = JsonValue::make_array();
        content.arr_val.push_back(std::move(item));

        JsonValue pp = JsonValue::make_object();
        pp.set("sessionId", JsonValue::make_string(sid));
        pp.set("mode", JsonValue::make_string("queue"));
        pp.set("content", std::move(content));

        const auto t0 = std::chrono::steady_clock::now();
        (void)dsh_rpc(dsh_port, "session.prompt", pp, rpc_seq);

        int cur_seq = seq_before;
        int cur_pt = 0;
        bool saw_progress = false;
        const auto deadline = t0 + std::chrono::seconds(max_wait_sec);
        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(350));
            cur_pt = get_pressure_and_seq(cur_seq, running);
            if (cur_seq > seq_before + 1) saw_progress = true;
            if (!running && saw_progress) break;
            if (!running && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 5.0) break;
        }

        TurnOutcome out{};
        out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out.pressure_tokens = cur_pt;

        JsonValue hp = JsonValue::make_object();
        hp.set("sessionId", JsonValue::make_string(sid));
        hp.set("maxMessages", JsonValue::make_int(8));
        const auto hist = dsh_rpc(dsh_port, "session.history", hp, rpc_seq);
        if (hist.has("events") && hist.at("events").is_array()) {
            for (const auto& entry : hist.at("events").arr_val) {
                if (!entry.has("event")) continue;
                const auto& ev = entry.at("event");
                const int seq = ev.has("seq") ? int(ev.at("seq").as_int()) : 0;
                if (seq <= seq_before) continue;
                const std::string ty = ev.has("type") ? ev.at("type").as_string() : "";
                if (ty == "assistant/message" && ev.has("data") && ev.at("data").has("message")) {
                    const auto& msg = ev.at("data").at("message");
                    if (msg.has("content") && msg.at("content").is_array()) {
                        for (const auto& blk : msg.at("content").arr_val) {
                            const std::string bt = blk.has("type") ? blk.at("type").as_string() : "";
                            if (bt == "reasoning" && blk.has("text")) {
                                out.reasoning += blk.at("text").as_string();
                            } else if (bt == "text" && blk.has("text")) {
                                out.text += blk.at("text").as_string();
                            } else if (bt == "tool-call") {
                                const std::string nm = blk.has("name") ? blk.at("name").as_string() : "";
                                if (!nm.empty()) out.tool_names.push_back(nm);
                                if (blk.has("arguments")) {
                                    const auto& a = blk.at("arguments");
                                    if (a.is_object()) {
                                        ++out.valid_tool_json;
                                    } else if (a.is_string()) {
                                        JsonValue parsed;
                                        if (JsonParser::try_parse(a.as_string(), parsed) && parsed.is_object()) {
                                            ++out.valid_tool_json;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out;
    }
};

} // namespace ernie55_g3

int main(int argc, char** argv) {
    using namespace ernie55_g3;
    std::string root = ".";
    int dsh_port = 3081;
    int gsm_n = 100;
    int he_n = 164;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--dsh-port" && i + 1 < argc) dsh_port = std::atoi(argv[++i]);
        else if (a == "--gsm-n" && i + 1 < argc) gsm_n = std::atoi(argv[++i]);
        else if (a == "--he-n" && i + 1 < argc) he_n = std::atoi(argv[++i]);
    }

    Ernie55Tokenizer tok;
    tok.load(root + "/checkpoints/active/tokenizer.json", root + "/checkpoints/active/tokenizer_config.json");
    const auto code_lines = load_real_codebase_lines(root);
    size_t code_cursor = 0;

    auto build_code_span = [&](int target_tokens, const std::string& n10, const std::string& n50, const std::string& n90) -> std::string {
        // Estimate ~11 tokens/line and calibrate cleanly with tokenizer on the assembled span
        std::vector<std::string> chunk;
        int approx_tok = 0;
        const int goal = std::max(256, target_tokens);
        while (approx_tok < goal) {
            const std::string& ln = code_lines[code_cursor % code_lines.size()];
            ++code_cursor;
            chunk.push_back(ln);
            approx_tok += std::max(1, int(ln.size() / 3));
        }
        // Refine exact token count via real BPE tokenizer
        auto join_with_needles = [&](size_t n_lines) {
            const size_t i10 = (n_lines * 10u) / 100u;
            const size_t i50 = (n_lines * 50u) / 100u;
            const size_t i90 = (n_lines * 90u) / 100u;
            std::string s;
            s.reserve(n_lines * 56u + 512u);
            for (size_t i = 0; i < n_lines; ++i) {
                if (i == i10) s += n10;
                if (i == i50) s += n50;
                if (i == i90) s += n90;
                s += chunk[i];
                s.push_back('\n');
            }
            return s;
        };
        std::string text = join_with_needles(chunk.size());
        int actual = int(tok.encode(text, false).size());
        while (actual < goal * 92 / 100) {
            const int extra = std::max(64, (goal - actual) / 4);
            for (int k = 0; k < extra; ++k) {
                chunk.push_back(code_lines[code_cursor % code_lines.size()]);
                ++code_cursor;
            }
            text = join_with_needles(chunk.size());
            actual = int(tok.encode(text, false).size());
        }
        if (actual > goal * 104 / 100 && chunk.size() > 64) {
            const size_t trimmed = std::max<size_t>(32, size_t(double(chunk.size()) * double(goal) / double(actual)));
            text = join_with_needles(std::min(trimmed, chunk.size()));
        }
        return text;
    };

    const auto t_start = std::chrono::steady_clock::now();
    LiveHarnessClient cli;
    cli.dsh_port = dsh_port;

    // 1. Create dedicated visible :3081 session
    {
        JsonValue cp = JsonValue::make_object();
        cp.set("cwd", JsonValue::make_string(root));
        cp.set("agentPreset", JsonValue::make_string("standard"));
        const auto cv = dsh_rpc(dsh_port, "session.create", cp, cli.rpc_seq);
        cli.sid = cv.at("sessionId").as_string();

        JsonValue rp = JsonValue::make_object();
        rp.set("sessionId", JsonValue::make_string(cli.sid));
        rp.set("title", JsonValue::make_string(
            "ERNIE-5.5-Flash-Preview · Official Live GATE-3 Suite (GSM8K / HumanEval / Tool+ThinkOff / 1M Needles)"));
        (void)dsh_rpc(dsh_port, "session.rename", rp, cli.rpc_seq);
        cli.select_model("medium");
        std::printf("[GATE-3-LIVE] Created visible :%d session: %s\n", dsh_port, cli.sid.c_str());
        std::fflush(stdout);
    }

    int gsm_ok = 0, gsm_tot = 0;
    int he_pass = 0, he_tot = 0;
    int tool_ok = 0, think_off_ok = 0;
    int needle_ok = 0, needle_tot = 0;
    std::ostringstream needle_json;
    const double depths[3] = {0.10, 0.50, 0.90};

    auto write_receipt = [&]() {
        const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::ofstream ofs(root + "/artifacts/gate3_ernie55_preview_hlut4_1m.json", std::ios::trunc);
        ofs << "{\n"
            << "  \"schema\": \"ernie55_preview_gate3_v1\",\n"
            << "  \"session_id\": \"" << cli.sid << "\",\n"
            << "  \"execution_mode\": \"live_3081_web_gui\",\n"
            << "  \"gsm8k_0shot_correct\": " << gsm_ok << ", \"gsm8k_0shot_total\": " << gsm_tot
            << ", \"gsm8k_0shot_pct\": " << std::fixed << std::setprecision(2) << (100.0 * gsm_ok / std::max(1, gsm_tot)) << ",\n"
            << "  \"humaneval_pass1\": " << he_pass << ", \"humaneval_total\": " << he_tot
            << ", \"humaneval_pass1_pct\": " << (100.0 * he_pass / std::max(1, he_tot)) << ",\n"
            << "  \"tool_calling_passed\": " << tool_ok << ", \"tool_calling_total\": 10,\n"
            << "  \"thinking_off_passed\": " << think_off_ok << ", \"thinking_off_total\": 10,\n"
            << "  \"protocol_passed\": " << (tool_ok + think_off_ok) << ", \"protocol_total\": 20,\n"
            << "  \"needles_passed\": " << needle_ok << ", \"needles_total\": " << needle_tot << ",\n"
            << "  \"wall_s\": " << wall_s << ",\n"
            << "  \"needles\": [\n" << needle_json.str() << "\n  ]\n}\n";
    };

    auto run_needle_rung = [&](int ri, int target_total_ctx) {
        int cur_seq = 0; bool run_f = false;
        const int cur_ctx = cli.get_pressure_and_seq(cur_seq, run_f);
        const int delta_tok = std::max(1024, std::min(target_total_ctx - cur_ctx - 512, 1032000 - cur_ctx));
        const int c10 = 417200 + ri * 1000 + 0 * 137;
        const int c50 = 417200 + ri * 1000 + 1 * 137;
        const int c90 = 417200 + ri * 1000 + 2 * 137;
        const std::string s10 = std::to_string(c10), s50 = std::to_string(c50), s90 = std::to_string(c90);
        const std::string tag = "R" + std::to_string(ri + 1);
        const std::string n10 = "\n// [HARDWARE_SPEC_" + tag + "_10] constexpr uint32_t kSpecKey" + tag + "Depth10 = " + s10 + "u;\n";
        const std::string n50 = "\n// [HARDWARE_SPEC_" + tag + "_50] constexpr uint32_t kSpecKey" + tag + "Depth50 = " + s50 + "u;\n";
        const std::string n90 = "\n// [HARDWARE_SPEC_" + tag + "_90] constexpr uint32_t kSpecKey" + tag + "Depth90 = " + s90 + "u;\n";
        const std::string body = build_code_span(delta_tok, n10, n50, n90);

        const std::string prompt =
            "Do NOT call any tools. Review the C++17 / gfx906 HIP / Blackhole codebase excerpt below (Rung " + tag + "):\n\n" +
            body +
            "\n\nExtract the exact 6-digit integer literals assigned to kSpecKey" + tag + "Depth10, kSpecKey" + tag +
            "Depth50, and kSpecKey" + tag + "Depth90 in this excerpt and compute their exact integer sum. "
            "Reply on one line: kSpecKey" + tag + "Depth10=<val>, kSpecKey" + tag + "Depth50=<val>, kSpecKey" + tag + "Depth90=<val>, sum=<sum>";

        const auto res = cli.prompt_and_wait(prompt, 3600);
        const std::string combined = res.text + "\n" + res.reasoning;
        const std::string codes[3] = {s10, s50, s90};
        for (int di = 0; di < 3; ++di) {
            const bool found = (combined.find(codes[di]) != std::string::npos);
            if (found) ++needle_ok;
            if (needle_tot > 0) needle_json << ",\n";
            needle_json << "    {\"ctx_len\": " << res.pressure_tokens << ", \"depth\": " << depths[di]
                        << ", \"code\": \"" << codes[di] << "\", \"found\": " << (found ? "true" : "false")
                        << ", \"wall_ms\": " << std::fixed << std::setprecision(1) << res.wall_ms << "}";
            ++needle_tot;
            std::printf("[GATE-3-LIVE] Needle %2d/18 | rung=%s ctx=%7d depth=%.0f%% | code=%s | %s (%.1f s)\n",
                        needle_tot, tag.c_str(), res.pressure_tokens, depths[di] * 100.0,
                        codes[di].c_str(), found ? "PASS" : "FAIL", res.wall_ms / 1000.0);
        }
        std::fflush(stdout);
        write_receipt();
    };

    // Suite 1: Early monotonic needle rungs (4K, 16K) while context is small
    run_needle_rung(0, 12000);
    run_needle_rung(1, 24000);

    // Suite 2: GSM8K 0-Shot (N=100, thinking ON / medium)
    {
        const auto test_j = JsonParser::parse(read_file_str(root + "/bench_data/gsm8k_test_100.json"));
        const auto& rows = test_j.at("rows").arr_val;
        gsm_tot = std::min<int>(gsm_n, int(rows.size()));
        for (int i = 0; i < gsm_tot; ++i) {
            const std::string q = rows[size_t(i)].at("row").at("question").as_string();
            const std::string gold = extract_gsm8k_num(rows[size_t(i)].at("row").at("answer").as_string());
            const std::string prompt =
                "Do NOT call any tools. Solve GSM8K #" + std::to_string(i + 1) +
                " cleanly and conclude with your final numeric answer after ####.\n\n" + q;
            const auto res = cli.prompt_and_wait(prompt, 300);
            const std::string pred = extract_gsm8k_num(!res.text.empty() ? res.text : res.reasoning);
            if (!gold.empty() && pred == gold) ++gsm_ok;
            if ((i + 1) % 10 == 0 || i + 1 == gsm_tot) {
                std::printf("[GATE-3-LIVE] GSM8K %d/%d | correct=%d (%.1f%%) | ctx=%d\n",
                            i + 1, gsm_tot, gsm_ok, 100.0 * gsm_ok / (i + 1), res.pressure_tokens);
                std::fflush(stdout);
                write_receipt();
            }
        }
    }

    // Suite 3: 64K Needle Rung
    run_needle_rung(2, 68000);

    // Suite 4a: 10/10 Live Structured Tool-Calling Probes inside :3081 (thinking ON)
    {
        const char* tool_prompts[10] = {
            "Use the glob tool to list *.hpp files in src/ and report the count.",
            "Use the read tool to inspect lines 1 to 15 of Makefile.",
            "Use the grep tool to find 'k_dsa_indexer_score_topk8_dpp4' in mi50/.",
            "Use the bash tool to run 'uname -m' and report the output.",
            "Use the glob tool to list *.json files in config/.",
            "Use the read tool to inspect lines 1 to 12 of scripts/start_ernie55_fe.sh.",
            "Use the grep tool to find 'launch_hca_attn_c128_swa_sink_splitk' in mi50/.",
            "Use the bash tool to run 'ls -1 artifacts/ | wc -l' and report the number.",
            "Use the read tool to inspect lines 1 to 10 of NOTICE.",
            "Use the grep tool to find 'Ernie55StreamParser' in src/ernie55_tokenizer.hpp."
        };
        for (int i = 0; i < 10; ++i) {
            const auto res = cli.prompt_and_wait(tool_prompts[i], 300);
            if (!res.tool_names.empty() && res.valid_tool_json > 0) ++tool_ok;
            std::printf("[GATE-3-LIVE] ToolProbe %2d/10 | tools=%zu valid_json=%d | pass=%d/%d\n",
                        i + 1, res.tool_names.size(), res.valid_tool_json, tool_ok, i + 1);
            std::fflush(stdout);
        }
        write_receipt();
    }

    // Switch to reasoningEffort="off" (enable_thinking=false) for Think-Off Audit + HumanEval + 256K..1M Needles
    cli.select_model("off");

    // Suite 4b: 10/10 enable_thinking=false Immediate-Answer Audit
    {
        const char* no_think_prompts[10] = {
            "Do NOT call any tools. What is 17 * 23? Answer with just the number.",
            "Do NOT call any tools. Name the capital city of Japan in one word.",
            "Do NOT call any tools. What is the atomic number of carbon?",
            "Do NOT call any tools. List the 3 primary colors of light separated by commas.",
            "Do NOT call any tools. What is 2 to the power of 10?",
            "Do NOT call any tools. Translate 'hello world' into French.",
            "Do NOT call any tools. What protocol port does HTTPS use by default?",
            "Do NOT call any tools. Name the largest planet in our solar system.",
            "Do NOT call any tools. What is the derivative of x^3 with respect to x?",
            "Do NOT call any tools. Give the 2-letter ISO code for Germany."
        };
        for (int i = 0; i < 10; ++i) {
            const auto res = cli.prompt_and_wait(no_think_prompts[i], 600);
            const bool clean = res.reasoning.empty() && !res.text.empty() &&
                               res.text.find("<ernie:think>") == std::string::npos &&
                               res.text.find("</ernie:think>") == std::string::npos;
            if (clean) ++think_off_ok;
            std::printf("[GATE-3-LIVE] ThinkOff %2d/10 | reasoning_len=%zu text_len=%zu | %s\n",
                        i + 1, res.reasoning.size(), res.text.size(), clean ? "PASS" : "FAIL");
            std::fflush(stdout);
        }
        write_receipt();
    }

    // Suite 5: HumanEval (N=164, pass@1 via zero-file POSIX pipe, thinking=off)
    {
        std::vector<JsonValue> all_rows;
        for (const char* fn : {"/bench_data/humaneval_test_64.json", "/bench_data/humaneval_test_64_164.json"}) {
            const std::string raw = read_file_str(root + fn);
            if (raw.empty()) continue;
            const auto j = JsonParser::parse(raw);
            for (const auto& r : j.at("rows").arr_val) all_rows.push_back(r.at("row"));
        }
        he_tot = std::min<int>(he_n, int(all_rows.size()));
        for (int i = 0; i < he_tot; ++i) {
            const auto& row = all_rows[size_t(i)];
            const std::string prompt = row.at("prompt").as_string();
            const std::string test   = row.at("test").as_string();
            const std::string entry  = row.at("entry_point").as_string();
            const std::string up =
                "Do NOT call any tools. Complete the Python function below and output ONLY valid Python code inside a ```python code block.\n\n```python\n" +
                prompt + "\n```";
            const auto res = cli.prompt_and_wait(up, 300);
            const std::string full_code = extract_python_code(res.text, prompt, entry) + "\n" + test + "\ncheck(" + entry + ")\n";
            if (eval_humaneval_candidate_pipe(full_code)) ++he_pass;
            if ((i + 1) % 10 == 0 || i + 1 == he_tot) {
                std::printf("[GATE-3-LIVE] HumanEval %d/%d | pass@1=%d (%.1f%%) | ctx=%d\n",
                            i + 1, he_tot, he_pass, 100.0 * he_pass / (i + 1), res.pressure_tokens);
                std::fflush(stdout);
                write_receipt();
            }
        }
    }

    // Suite 6: Long-Context Real-Codebase Multi-Needle Rungs (256K, 512K, ~1M)
    run_needle_rung(3, 262144);
    run_needle_rung(4, 524288);
    run_needle_rung(5, 1028000);

    write_receipt();
    std::printf("[GATE-3-LIVE] DONE | session=%s | GSM8K=%d/%d | HumanEval=%d/%d | Protocol=%d/20 | Needles=%d/%d\n",
                cli.sid.c_str(), gsm_ok, gsm_tot, he_pass, he_tot, tool_ok + think_off_ok, needle_ok, needle_tot);
    return (needle_ok == needle_tot && (tool_ok + think_off_ok) >= 18) ? 0 : 1;
}
