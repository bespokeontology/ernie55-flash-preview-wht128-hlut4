// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 OpenAI-Compatible HTTP/1.1 + SSE Server (:8052) for baidu/ERNIE-5.5-Flash (ernie5_5_moe)
// Supports:
//   - POST /v1/completions           (for ERNIE-5.5-Flash-Base today!)
//   - POST /v1/chat/completions      (streaming SSE & non-streaming JSON; auto chat template + ernie5 reasoning & tool parser!)
//   - GET  /health, /v1/models, /v1/metrics
//   - POST /v1/kv/save, /v1/kv/load
//   - POST /v1/shutdown              (strict graceful teardown: syncs MI50s + optional P150A IPC shutdown)
// Zero Python anywhere.

#include "ernie55_runtime.hpp"
#include "ernie55_graceful.hpp"
#include "ernie55_json.hpp"
#include "ernie55_live_progress.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ernie55 {
namespace {

std::atomic<bool> g_shutdown_req{false};
int g_listen_fd = -1;

void handle_sig(int) {
    g_shutdown_req.store(true, std::memory_order_relaxed);
    if (g_listen_fd >= 0) ::shutdown(g_listen_fd, SHUT_RDWR);
}

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += size_t(n);
    }
    return true;
}

void send_http_json(int fd, int status, const char* reason, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << reason << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    const std::string s = oss.str();
    (void)send_all(fd, s.data(), s.size());
}

bool start_sse(int fd) {
    static constexpr const char* kHdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n";
    return send_all(fd, kHdr, std::strlen(kHdr));
}

bool send_sse_json(int fd, const JsonValue& obj) {
    const std::string line = "data: " + obj.dumps_compact() + "\n\n";
    return send_all(fd, line.data(), line.size());
}

bool send_sse_done(int fd) {
    static constexpr const char* kDone = "data: [DONE]\n\n";
    return send_all(fd, kDone, std::strlen(kDone));
}

bool send_sse_keepalive(int fd) {
    static constexpr const char* kKeep = ": keepalive\n\n";
    return send_all(fd, kKeep, std::strlen(kKeep));
}

bool client_socket_closed(int fd) noexcept {
    char probe = 0;
    const ssize_t pk = ::recv(fd, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
    if (pk == 0) return true;
    if (pk < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
    return false;
}

int64_t steady_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

bool read_http_request(int fd, HttpRequest& req) {
    std::string raw;
    raw.reserve(4096);
    char buf[4096];
    size_t hdr_end = std::string::npos;
    size_t content_len = 0;

    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        raw.append(buf, size_t(n));
        if (hdr_end == std::string::npos) {
            hdr_end = raw.find("\r\n\r\n");
            if (hdr_end != std::string::npos) {
                std::string hdr = raw.substr(0, hdr_end);
                size_t cl = hdr.find("Content-Length:");
                if (cl == std::string::npos) cl = hdr.find("content-length:");
                if (cl != std::string::npos) {
                    content_len = size_t(std::strtoull(hdr.c_str() + cl + 15, nullptr, 10));
                }
            }
        }
        if (hdr_end != std::string::npos && raw.size() >= hdr_end + 4 + content_len) break;
        if (raw.size() > 64u * 1024u * 1024u) return false;
    }

    const size_t sp1 = raw.find(' ');
    const size_t sp2 = (sp1 != std::string::npos) ? raw.find(' ', sp1 + 1) : std::string::npos;
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = raw.substr(0, sp1);
    req.path   = raw.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t qm = req.path.find('?');
    if (qm != std::string::npos) req.path.resize(qm);
    req.body = raw.substr(hdr_end + 4, content_len);
    return true;
}

int64_t unix_now() noexcept {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

SamplingParams parse_sampling(const JsonValue& req, bool chat_default_stochastic, uint64_t default_seed) {
    SamplingParams sp;
    if (chat_default_stochastic) {
        // Official ERNIE-5.5-Flash-Preview generation_config.json defaults (when caller omits temperature/top_p!)
        sp.temperature = 1.0f;
        sp.top_p       = 0.95f;
        sp.seed        = default_seed;
    }
    if (req.has("temperature")) sp.temperature = float(req.get("temperature").as_double());
    if (req.has("top_p"))       sp.top_p       = float(req.get("top_p").as_double());
    if (req.has("top_k"))       sp.top_k       = int(req.get("top_k").as_int());
    if (req.has("seed"))        sp.seed        = uint64_t(req.get("seed").as_int());
    if (req.has("do_sample") && req.get("do_sample").is_bool() && !req.get("do_sample").as_bool()) {
        sp.temperature = 0.0f;
    }
    if (req.has("draft_sample_method") && req.get("draft_sample_method").is_string()) {
        const std::string m = req.get("draft_sample_method").as_string();
        sp.draft_sample_method = (m == "greedy") ? DraftSampleMethod::kGreedy : DraftSampleMethod::kProbabilistic;
    }
    return sp;
}

std::vector<ChatMessage> parse_chat_messages(const JsonValue& arr) {
    std::vector<ChatMessage> msgs;
    if (!arr.is_array()) return msgs;
    for (const auto& item : arr.arr_val) {
        if (!item.is_object()) continue;
        ChatMessage m;
        if (item.has("role") && item.get("role").is_string()) m.role = item.get("role").as_string();
        if (item.has("name") && item.get("name").is_string()) m.name = item.get("name").as_string();
        if (item.has("tool_call_id") && item.get("tool_call_id").is_string()) m.tool_call_id = item.get("tool_call_id").as_string();
        if (item.has("reasoning_content") && item.get("reasoning_content").is_string()) m.reasoning_content = item.get("reasoning_content").as_string();
        if (item.has("content")) {
            const auto& c = item.get("content");
            if (c.is_string()) {
                m.content = c.as_string();
            } else if (c.is_array()) {
                for (const auto& part : c.arr_val) {
                    if (part.is_object() && part.has("text") && part.get("text").is_string()) {
                        m.content += part.get("text").as_string();
                    }
                }
            }
        }
        if (item.has("tool_calls") && item.get("tool_calls").is_array()) {
            for (const auto& tc : item.get("tool_calls").arr_val) {
                if (!tc.is_object()) continue;
                ChatMessage::ToolCall c;
                if (tc.has("id") && tc.get("id").is_string()) c.id = tc.get("id").as_string();
                if (tc.has("function") && tc.get("function").is_object()) {
                    const auto& fn = tc.get("function");
                    if (fn.has("name") && fn.get("name").is_string()) c.name = fn.get("name").as_string();
                    if (fn.has("arguments") && fn.get("arguments").is_string()) c.arguments_json = fn.get("arguments").as_string();
                }
                m.tool_calls.push_back(std::move(c));
            }
        }
        msgs.push_back(std::move(m));
    }
    return msgs;
}

JsonValue make_usage_json_live(size_t prompt_toks, size_t cached_toks, size_t comp_toks, size_t reasoning_toks = 0) {
    JsonValue u = JsonValue::make_object();
    u.set("prompt_tokens", JsonValue::make_int(int64_t(prompt_toks)));
    u.set("cached_prefix_tokens", JsonValue::make_int(int64_t(cached_toks)));
    u.set("completion_tokens", JsonValue::make_int(int64_t(comp_toks)));
    u.set("total_tokens", JsonValue::make_int(int64_t(prompt_toks + comp_toks)));
    JsonValue det = JsonValue::make_object();
    det.set("cached_tokens", JsonValue::make_int(int64_t(cached_toks)));
    u.set("prompt_tokens_details", std::move(det));
    JsonValue cdet = JsonValue::make_object();
    cdet.set("reasoning_tokens", JsonValue::make_int(int64_t(reasoning_toks)));
    u.set("completion_tokens_details", std::move(cdet));
    return u;
}

JsonValue make_usage_json(const GenerateStats& st, size_t reasoning_toks = 0) {
    JsonValue u = make_usage_json_live(size_t(st.prompt_tokens), size_t(st.cached_prefix_tokens),
                                       size_t(st.generated_tokens), reasoning_toks);
    u.set("ttft_ms", JsonValue::make_double(st.ttft_ms));
    u.set("decode_tok_s", JsonValue::make_double(st.decode_tok_s));
    u.set("spec_accepted", JsonValue::make_int(int64_t(st.spec_accepted)));
    u.set("spec_drafted", JsonValue::make_int(int64_t(st.spec_drafted)));
    return u;
}

JsonValue tool_calls_to_json(const std::vector<ParsedToolCall>& tcs, bool with_index, int* io_sse_idx = nullptr) {
    JsonValue arr = JsonValue::make_array();
    for (size_t i = 0; i < tcs.size(); ++i) {
        JsonValue tc = JsonValue::make_object();
        if (with_index) {
            const int idx = io_sse_idx ? (*io_sse_idx)++ : int(i);
            tc.set("index", JsonValue::make_int(int64_t(idx)));
        }
        tc.set("id", JsonValue::make_string(tcs[i].id));
        tc.set("type", JsonValue::make_string("function"));
        JsonValue fn = JsonValue::make_object();
        fn.set("name", JsonValue::make_string(tcs[i].name));
        fn.set("arguments", JsonValue::make_string(tcs[i].arguments_json));
        tc.set("function", std::move(fn));
        arr.arr_val.push_back(std::move(tc));
    }
    return arr;
}

JsonValue tool_deltas_to_json(const std::vector<StreamToolCallDelta>& tds) {
    JsonValue arr = JsonValue::make_array();
    for (const auto& d : tds) {
        JsonValue tc = JsonValue::make_object();
        tc.set("index", JsonValue::make_int(int64_t(d.index)));
        if (!d.id.empty()) {
            tc.set("id", JsonValue::make_string(d.id));
            tc.set("type", JsonValue::make_string("function"));
        }
        JsonValue fn = JsonValue::make_object();
        if (!d.name.empty()) fn.set("name", JsonValue::make_string(d.name));
        fn.set("arguments", JsonValue::make_string(d.arguments_delta));
        tc.set("function", std::move(fn));
        arr.arr_val.push_back(std::move(tc));
    }
    return arr;
}

} // namespace
} // namespace ernie55

int main(int argc, char** argv) {
    using namespace ernie55;
    std::signal(SIGINT,  handle_sig);
    std::signal(SIGTERM, handle_sig);
    std::signal(SIGPIPE, SIG_IGN);

    std::string host = "127.0.0.1";
    int port = 8052;
    std::string model_id = "ERNIE-5.5-Flash-Preview-WHT128-HLUT4";
    Ernie55RuntimeOptions opt;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg for ") + f);
            return argv[++i];
        };
        if (a == "--host")               host = need("--host");
        else if (a == "--port")          port = std::stoi(need("--port"));
        else if (a == "--model-id")      model_id = need("--model-id");
        else if (a == "--weight-root")   opt.weight_root = need("--weight-root");
        else if (a == "--topology")      opt.topology_path = need("--topology");
        else if (a == "--shm")           opt.shm_name = need("--shm");
        else if (a == "--max-kv")        opt.max_kv = std::stoi(need("--max-kv"));
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(need("--prefill-chunk"));
        else if (a == "--draft-mode")    opt.draft_mode = need("--draft-mode");
        else if (a == "--echo-steps")    opt.echo_draft_steps = std::stoi(need("--echo-steps"));
        else if (a == "--dspark-k")      opt.dspark_block_size = std::stoi(need("--dspark-k"));
        else if (a == "--dspark-conf-tau") opt.dspark_conf_tau = std::stof(need("--dspark-conf-tau"));
        else if (a == "--no-flashindexer") opt.enable_flashindexer = false;
        else if (a == "--no-p150")       opt.use_p150 = false;
        else if (a == "--no-keep-p150")  opt.keep_p150 = false;
        else if (a == "--no-echo")       opt.load_echo = false;
    }

    try {
        Ernie55Runtime rt;
        rt.init(opt);
        const auto special_ids = rt.tokenizer().special_ids();

        const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (lfd < 0) throw std::runtime_error("socket() failed");
        int one = 1;
        ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(uint16_t(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) throw std::runtime_error("Invalid --host");
        if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) throw std::runtime_error("bind() failed on port " + std::to_string(port));
        if (::listen(lfd, 16) != 0) throw std::runtime_error("listen() failed");
        g_listen_fd = lfd;

        std::printf("[ERNIE55-FE] Serving %s on http://%s:%d (chat_template=%s, draft=%s)\n",
                    model_id.c_str(), host.c_str(), port,
                    rt.tokenizer().has_chat_template() ? "enabled" : "base+chat-ready",
                    rt.active_draft_method() == DraftMethod::kDSpark ? "dspark-probabilistic"
                    : (rt.active_draft_method() == DraftMethod::kEcho ? "echo-2step" : "none"));
        std::fflush(stdout);

        std::mutex infer_mtx;
        uint64_t req_seq = 0;

        while (!g_shutdown_req.load(std::memory_order_relaxed)) {
            sockaddr_in cli{};
            socklen_t clen = sizeof(cli);
            const int cfd = ::accept(lfd, reinterpret_cast<sockaddr*>(&cli), &clen);
            if (cfd < 0) {
                if (g_shutdown_req.load(std::memory_order_relaxed)) break;
                continue;
            }
            ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

            HttpRequest req;
            if (!read_http_request(cfd, req)) { ::close(cfd); continue; }

            if (req.method == "OPTIONS") {
                send_http_json(cfd, 200, "OK", "{}");
                ::close(cfd);
                continue;
            }
            if (req.method == "GET" && req.path == "/health") {
                JsonValue h = rt.metrics_json();
                h.set("status", JsonValue::make_string("ok"));
                h.set("model_id", JsonValue::make_string(model_id));
                h.set("chat_template_detected", JsonValue::make_bool(rt.tokenizer().has_chat_template()));
                uint64_t th = 0, tm = 0; size_t tb = 0;
                rt.tokenizer().cache_stats(th, tm, tb);
                h.set("tok_cache_hits", JsonValue::make_int(int64_t(th)));
                h.set("tok_cache_misses", JsonValue::make_int(int64_t(tm)));
                h.set("tok_cache_mib", JsonValue::make_double(double(tb) / 1048576.0));
                send_http_json(cfd, 200, "OK", h.dumps_compact());
                ::close(cfd);
                continue;
            }
            if (req.method == "GET" && (req.path == "/v1/metrics" || req.path == "/metrics")) {
                send_http_json(cfd, 200, "OK", rt.metrics_json().dumps_compact());
                ::close(cfd);
                continue;
            }
            if (req.method == "GET" && req.path == "/v1/models") {
                JsonValue root = JsonValue::make_object();
                root.set("object", JsonValue::make_string("list"));
                JsonValue arr = JsonValue::make_array();
                JsonValue m = JsonValue::make_object();
                m.set("id", JsonValue::make_string(model_id));
                m.set("object", JsonValue::make_string("model"));
                m.set("owned_by", JsonValue::make_string("baidu"));
                m.set("max_model_len", JsonValue::make_int(rt.max_kv()));
                arr.arr_val.push_back(std::move(m));
                root.set("data", std::move(arr));
                send_http_json(cfd, 200, "OK", root.dumps_compact());
                ::close(cfd);
                continue;
            }
            if (req.method == "POST" && req.path == "/v1/shutdown") {
                send_http_json(cfd, 200, "OK", "{\"status\":\"shutting_down\"}");
                ::close(cfd);
                g_shutdown_req.store(true, std::memory_order_relaxed);
                break;
            }
            if (req.method == "POST" && req.path == "/v1/kv/reset") {
                std::lock_guard<std::mutex> lk(infer_mtx);
                rt.reset_kv();
                send_http_json(cfd, 200, "OK", "{\"status\":\"ok\",\"seq_len\":0}");
                ::close(cfd);
                continue;
            }
            if (req.method == "POST" && (req.path == "/v1/kv/save" || req.path == "/v1/kv/load" || req.path == "/v1/kv/restore")) {
                try {
                    JsonValue j = JsonParser::parse(req.body);
                    const std::string p = j.get("path").as_string();
                    std::lock_guard<std::mutex> lk(infer_mtx);
                    KvBlkStats kst{};
                    if (req.path == "/v1/kv/save") {
                        rt.flush_kvblk_cache();
                        kst = rt.save_kvblk_ex(p);
                    } else {
                        kst = rt.load_kvblk_ex(p);
                    }
                    JsonValue ok = JsonValue::make_object();
                    ok.set("status",    JsonValue::make_string("ok"));
                    ok.set("seq_len",   JsonValue::make_int(kst.seq_len));
                    ok.set("bytes",     JsonValue::make_int(int64_t(kst.bytes)));
                    ok.set("io_ms",     JsonValue::make_double(kst.io_ms));
                    ok.set("dma_ms",    JsonValue::make_double(kst.dma_ms));
                    ok.set("total_ms",  JsonValue::make_double(kst.total_ms));
                    ok.set("gb_s",      JsonValue::make_double(kst.gb_s));
                    ok.set("tok_s",     JsonValue::make_double(kst.tok_s));
                    ok.set("dual_nvme", JsonValue::make_bool(kst.dual_nvme));
                    send_http_json(cfd, 200, "OK", ok.dumps_compact());
                } catch (const std::exception& ex) {
                    JsonValue err = JsonValue::make_object();
                    err.set("error", JsonValue::make_string(ex.what()));
                    send_http_json(cfd, 400, "Bad Request", err.dumps_compact());
                }
                ::close(cfd);
                continue;
            }

            if (req.method == "POST" && req.path == "/v1/completions") {
                try {
                    JsonValue j = JsonParser::parse(req.body);
                    const std::string prompt = j.has("prompt") && j.get("prompt").is_string() ? j.get("prompt").as_string() : "";
                    const int max_tok = j.has("max_tokens") ? int(j.get("max_tokens").as_int()) : 128;
                    const bool stream = j.has("stream") && j.get("stream").as_bool();
                    const SamplingParams sp = parse_sampling(j, false, uint64_t(unix_now()) ^ (req_seq + 1ULL));
                    const std::string cid = "cmpl-ernie55-" + std::to_string(++req_seq);
                    const int64_t ts = unix_now();

                    std::vector<int32_t> prompt_ids;
                    prompt_ids.push_back(rt.tokenizer().bos_id());
                    auto enc = rt.tokenizer().encode(prompt, false);
                    prompt_ids.insert(prompt_ids.end(), enc.begin(), enc.end());

                    std::lock_guard<std::mutex> lk(infer_mtx);
                    GenerateStats stats;
                    if (stream) {
                        start_sse(cfd);
                        rt.generate_stream(prompt_ids, max_tok, sp, {}, &g_shutdown_req,
                            [&](int32_t, const std::string& piece) -> bool {
                                JsonValue ev = JsonValue::make_object();
                                ev.set("id", JsonValue::make_string(cid));
                                ev.set("object", JsonValue::make_string("text_completion"));
                                ev.set("created", JsonValue::make_int(ts));
                                ev.set("model", JsonValue::make_string(model_id));
                                JsonValue ch = JsonValue::make_object();
                                ch.set("index", JsonValue::make_int(0));
                                ch.set("text", JsonValue::make_string(piece));
                                ch.set("finish_reason", JsonValue::make_null());
                                JsonValue chs = JsonValue::make_array();
                                chs.arr_val.push_back(std::move(ch));
                                ev.set("choices", std::move(chs));
                                return send_sse_json(cfd, ev);
                            }, stats);
                        JsonValue fin = JsonValue::make_object();
                        fin.set("id", JsonValue::make_string(cid));
                        fin.set("object", JsonValue::make_string("text_completion"));
                        fin.set("created", JsonValue::make_int(ts));
                        fin.set("model", JsonValue::make_string(model_id));
                        JsonValue ch = JsonValue::make_object();
                        ch.set("index", JsonValue::make_int(0));
                        ch.set("text", JsonValue::make_string(""));
                        ch.set("finish_reason", JsonValue::make_string(stats.finish_reason));
                        JsonValue chs = JsonValue::make_array();
                        chs.arr_val.push_back(std::move(ch));
                        fin.set("choices", std::move(chs));
                        fin.set("usage", make_usage_json(stats));
                        send_sse_json(cfd, fin);
                        send_sse_done(cfd);
                    } else {
                        std::string full_text;
                        rt.generate_stream(prompt_ids, max_tok, sp, {}, &g_shutdown_req,
                            [&](int32_t, const std::string& piece) -> bool {
                                full_text += piece;
                                return true;
                            }, stats);
                        JsonValue resp = JsonValue::make_object();
                        resp.set("id", JsonValue::make_string(cid));
                        resp.set("object", JsonValue::make_string("text_completion"));
                        resp.set("created", JsonValue::make_int(ts));
                        resp.set("model", JsonValue::make_string(model_id));
                        JsonValue ch = JsonValue::make_object();
                        ch.set("index", JsonValue::make_int(0));
                        ch.set("text", JsonValue::make_string(full_text));
                        ch.set("finish_reason", JsonValue::make_string(stats.finish_reason));
                        JsonValue chs = JsonValue::make_array();
                        chs.arr_val.push_back(std::move(ch));
                        resp.set("choices", std::move(chs));
                        resp.set("usage", make_usage_json(stats));
                        send_http_json(cfd, 200, "OK", resp.dumps_compact());
                    }
                    std::printf("[ERNIE55-FE] /v1/completions prompt=%d gen=%d ttft=%.1fms dec=%.2f tok/s spec=%llu/%llu\n",
                                stats.prompt_tokens, stats.generated_tokens, stats.ttft_ms, stats.decode_tok_s,
                                (unsigned long long)stats.spec_accepted, (unsigned long long)stats.spec_drafted);
                    std::fflush(stdout);
                } catch (const std::exception& ex) {
                    JsonValue err = JsonValue::make_object();
                    err.set("error", JsonValue::make_string(ex.what()));
                    send_http_json(cfd, 400, "Bad Request", err.dumps_compact());
                }
                ::close(cfd);
                continue;
            }

            if (req.method == "POST" && req.path == "/v1/chat/completions") {
                try {
                    JsonValue j = JsonParser::parse(req.body);
                    const auto msgs = parse_chat_messages(j.get("messages"));
                    const JsonValue* tools = j.has("tools") ? &j.get("tools") : nullptr;
                    const JsonValue* ctk = j.has("chat_template_kwargs") ? &j.get("chat_template_kwargs") : nullptr;
                    bool enable_thinking = is_ernie5_thinking_enabled(ctk);
                    if (j.has("enable_thinking") && j.get("enable_thinking").is_bool()) {
                        enable_thinking = j.get("enable_thinking").as_bool();
                    }
                    const bool parallel_tc = !(j.has("parallel_tool_calls") && j.get("parallel_tool_calls").is_bool() &&
                                               !j.get("parallel_tool_calls").as_bool());
                    std::string reasoning_effort = "high";
                    if (ctk && ctk->is_object() && ctk->has("reasoning_effort") && ctk->get("reasoning_effort").is_string()) {
                        reasoning_effort = ctk->get("reasoning_effort").as_string();
                    } else if (j.has("reasoning_effort") && j.get("reasoning_effort").is_string()) {
                        reasoning_effort = j.get("reasoning_effort").as_string();
                    }
                    const int max_tok = j.has("max_tokens") ? int(j.get("max_tokens").as_int())
                                      : (j.has("max_completion_tokens") ? int(j.get("max_completion_tokens").as_int()) : 1024);
                    const bool stream = j.has("stream") && j.get("stream").as_bool();
                    const int64_t ts = unix_now();
                    const SamplingParams sp = parse_sampling(j, true, uint64_t(ts) ^ (req_seq + 1ULL));
                    const std::string cid = "chatcmpl-ernie55-" + std::to_string(++req_seq);

                    const auto t_tok0 = std::chrono::steady_clock::now();
                    const auto segs = rt.tokenizer().render_chat_segments(
                        msgs, tools, enable_thinking, true, reasoning_effort, parallel_tc);
                    const auto prompt_ids = rt.tokenizer().encode_segments(segs);
                    const double tok_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_tok0).count();

                    std::lock_guard<std::mutex> lk(infer_mtx);
                    GenerateStats stats;
                    Ernie55StreamParser parser(enable_thinking, tools, &special_ids);
                    parser.set_parallel_tool_calls(parallel_tc);

                    if (stream) {
                        start_sse(cfd);
                        bool saw_tools = false;
                        std::mutex sse_mtx;
                        std::atomic<bool> sse_done{false};
                        std::atomic<bool> sse_dead{false};
                        std::atomic<int64_t> last_send_ms{steady_ms()};
                        std::atomic<size_t> live_gen_toks{0};
                        std::atomic<size_t> live_reasoning_toks{0};
                        std::atomic<size_t> live_cached_toks{0};
                        const int64_t req_start_ns = Ernie55LiveProgress::now_ns();
                        Ernie55LiveProgress live_prog;

                        // Emit opening role chunk immediately (matches Qwen 3.8 flash_fe.cpp:1057-1060)
                        {
                            JsonValue open_ev = JsonValue::make_object();
                            open_ev.set("id", JsonValue::make_string(cid));
                            open_ev.set("object", JsonValue::make_string("chat.completion.chunk"));
                            open_ev.set("created", JsonValue::make_int(ts));
                            open_ev.set("model", JsonValue::make_string(model_id));
                            JsonValue od = JsonValue::make_object();
                            od.set("role", JsonValue::make_string("assistant"));
                            od.set("content", JsonValue::make_string(""));
                            JsonValue ch = JsonValue::make_object();
                            ch.set("index", JsonValue::make_int(0));
                            ch.set("delta", std::move(od));
                            ch.set("finish_reason", JsonValue::make_null());
                            JsonValue chs = JsonValue::make_array();
                            chs.arr_val.push_back(std::move(ch));
                            open_ev.set("choices", std::move(chs));
                            std::lock_guard<std::mutex> slk(sse_mtx);
                            if (!send_sse_json(cfd, open_ev)) sse_dead.store(true, std::memory_order_relaxed);
                            else last_send_ms.store(steady_ms(), std::memory_order_relaxed);
                        }

                        auto emit_progress_frame = [&](bool is_final) -> bool {
                            if (sse_dead.load(std::memory_order_relaxed)) return false;
                            if (!is_final && !live_prog.input_ready.load(std::memory_order_acquire) &&
                                live_prog.gen_count.load(std::memory_order_acquire) == 0) {
                                return true;
                            }
                            // Bug 4c Fix: Advance layer-granularity fresh prefill tokens every 150ms (0 extra GPU syncs!)
                            if (!is_final && live_prog.gen_count.load(std::memory_order_acquire) == 0) {
                                const size_t live_f = rt.live_prefill_fresh_tokens();
                                if (live_f > 0) live_prog.observe_prefill(live_f, 0.0);
                            }
                            JsonValue pev = JsonValue::make_object();
                            pev.set("id", JsonValue::make_string(cid));
                            pev.set("object", JsonValue::make_string("chat.completion.chunk"));
                            pev.set("created", JsonValue::make_int(ts));
                            pev.set("model", JsonValue::make_string(model_id));
                            pev.set("choices", JsonValue::make_array());
                            pev.set("x_flash_progress", live_prog.sample(cid, is_final));
                            pev.set("usage", make_usage_json_live(
                                prompt_ids.size(),
                                live_cached_toks.load(std::memory_order_relaxed),
                                live_gen_toks.load(std::memory_order_relaxed),
                                live_reasoning_toks.load(std::memory_order_relaxed)));
                            std::lock_guard<std::mutex> slk(sse_mtx);
                            if (sse_done.load(std::memory_order_relaxed) && !is_final) return true;
                            const bool ok = send_sse_json(cfd, pev);
                            if (ok) last_send_ms.store(steady_ms(), std::memory_order_relaxed);
                            else sse_dead.store(true, std::memory_order_relaxed);
                            return ok;
                        };

                        rt.set_prefill_progress_callback(
                            [&](size_t p_tot, size_t p_cached, size_t fresh_done, double chunk_ms) {
                                live_cached_toks.store(p_cached, std::memory_order_relaxed);
                                if (fresh_done == 0) {
                                    std::string ctier;
                                    double cms_r = 0.0;
                                    rt.last_cache_restore_info(ctier, cms_r);
                                    live_prog.configure_input(p_tot, p_cached, req_start_ns, ctier, cms_r);
                                } else {
                                    live_prog.observe_prefill(fresh_done, chunk_ms);
                                }
                                (void)emit_progress_frame(false);
                            });

                        // Background 150ms native x_flash_progress + 2s SSE heartbeat keeps :3081 counters
                        // ticking live across prefills & decode bursts and detects dead sockets immediately!
                        std::thread ka_thr([&]() {
                            while (!sse_done.load(std::memory_order_relaxed)) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                                if (sse_done.load(std::memory_order_relaxed)) break;
                                if (client_socket_closed(cfd)) {
                                    sse_dead.store(true, std::memory_order_relaxed);
                                    break;
                                }
                                if (!emit_progress_frame(false)) break;
                                const int64_t now = steady_ms();
                                if (now - last_send_ms.load(std::memory_order_relaxed) >= 2000) {
                                    std::lock_guard<std::mutex> slk(sse_mtx);
                                    if (sse_done.load(std::memory_order_relaxed)) break;
                                    if (!send_sse_keepalive(cfd)) {
                                        sse_dead.store(true, std::memory_order_relaxed);
                                        break;
                                    }
                                    last_send_ms.store(steady_ms(), std::memory_order_relaxed);
                                }
                            }
                        });

                        auto emit_delta = [&](const std::string& rd, const std::string& cd,
                                              const std::vector<StreamToolCallDelta>& tds) -> bool {
                            if (sse_dead.load(std::memory_order_relaxed) || client_socket_closed(cfd)) {
                                sse_dead.store(true, std::memory_order_relaxed);
                                return false;
                            }
                            if (rd.empty() && cd.empty() && tds.empty()) {
                                const int64_t now = steady_ms();
                                if (now - last_send_ms.load(std::memory_order_relaxed) >= 2000) {
                                    std::lock_guard<std::mutex> slk(sse_mtx);
                                    if (!send_sse_keepalive(cfd)) {
                                        sse_dead.store(true, std::memory_order_relaxed);
                                        return false;
                                    }
                                    last_send_ms.store(steady_ms(), std::memory_order_relaxed);
                                }
                                return true;
                            }
                            JsonValue ev = JsonValue::make_object();
                            ev.set("id", JsonValue::make_string(cid));
                            ev.set("object", JsonValue::make_string("chat.completion.chunk"));
                            ev.set("created", JsonValue::make_int(ts));
                            ev.set("model", JsonValue::make_string(model_id));
                            JsonValue delta = JsonValue::make_object();
                            if (!rd.empty()) delta.set("reasoning_content", JsonValue::make_string(rd));
                            if (!cd.empty()) delta.set("content", JsonValue::make_string(cd));
                            if (!tds.empty()) {
                                saw_tools = true;
                                delta.set("tool_calls", tool_deltas_to_json(tds));
                            }
                            JsonValue ch = JsonValue::make_object();
                            ch.set("index", JsonValue::make_int(0));
                            ch.set("delta", std::move(delta));
                            ch.set("finish_reason", JsonValue::make_null());
                            JsonValue chs = JsonValue::make_array();
                            chs.arr_val.push_back(std::move(ch));
                            ev.set("choices", std::move(chs));
                            ev.set("usage", make_usage_json_live(
                                prompt_ids.size(),
                                live_cached_toks.load(std::memory_order_relaxed),
                                live_gen_toks.load(std::memory_order_relaxed),
                                live_reasoning_toks.load(std::memory_order_relaxed)));
                            std::lock_guard<std::mutex> slk(sse_mtx);
                            const bool ok = send_sse_json(cfd, ev);
                            if (ok) last_send_ms.store(steady_ms(), std::memory_order_relaxed);
                            else sse_dead.store(true, std::memory_order_relaxed);
                            return ok;
                        };

                        rt.generate_stream(prompt_ids, max_tok, sp, {}, &g_shutdown_req,
                            [&](int32_t tid, const std::string& piece) -> bool {
                                const size_t cur_g = live_gen_toks.fetch_add(1, std::memory_order_relaxed) + 1;
                                live_prog.observe_decode_tokens(cur_g);
                                if (sse_dead.load(std::memory_order_relaxed) || client_socket_closed(cfd)) {
                                    sse_dead.store(true, std::memory_order_relaxed);
                                    return false;
                                }
                                std::string rd, cd;
                                std::vector<StreamToolCallDelta> tds;
                                parser.feed_token_stream(tid, piece, rd, cd, tds);
                                if (!rd.empty() && cd.empty() && tds.empty()) {
                                    live_reasoning_toks.fetch_add(1, std::memory_order_relaxed);
                                }
                                return emit_delta(rd, cd, tds);
                            }, stats);

                        rt.set_prefill_progress_callback({});
                        // Bug 4d Fix: Reconcile live_gen_toks & live_prog with exact stats.generated_tokens!
                        if (size_t(stats.generated_tokens) > live_gen_toks.load(std::memory_order_relaxed)) {
                            live_gen_toks.store(size_t(stats.generated_tokens), std::memory_order_relaxed);
                        }
                        live_prog.finish(size_t(stats.generated_tokens));

                        std::string rd, cd;
                        std::vector<StreamToolCallDelta> tds;
                        parser.flush_stream(rd, cd, tds);
                        (void)emit_delta(rd, cd, tds);

                        sse_done.store(true, std::memory_order_relaxed);
                        if (ka_thr.joinable()) ka_thr.join();

                        if (!sse_dead.load(std::memory_order_relaxed)) {
                            const size_t r_toks = live_reasoning_toks.load(std::memory_order_relaxed);
                            (void)emit_progress_frame(true);
                            JsonValue fin = JsonValue::make_object();
                            fin.set("id", JsonValue::make_string(cid));
                            fin.set("object", JsonValue::make_string("chat.completion.chunk"));
                            fin.set("created", JsonValue::make_int(ts));
                            fin.set("model", JsonValue::make_string(model_id));
                            JsonValue ch = JsonValue::make_object();
                            ch.set("index", JsonValue::make_int(0));
                            ch.set("delta", JsonValue::make_object());
                            ch.set("finish_reason", JsonValue::make_string(saw_tools ? "tool_calls" : stats.finish_reason));
                            JsonValue chs = JsonValue::make_array();
                            chs.arr_val.push_back(std::move(ch));
                            fin.set("choices", std::move(chs));
                            fin.set("usage", make_usage_json(stats, r_toks));
                            send_sse_json(cfd, fin);

                            // Final OpenAI empty-choices usage chunk matching Qwen 3.8 flash_fe.cpp:1291-1299!
                            JsonValue fin_u = JsonValue::make_object();
                            fin_u.set("id", JsonValue::make_string(cid));
                            fin_u.set("object", JsonValue::make_string("chat.completion.chunk"));
                            fin_u.set("created", JsonValue::make_int(ts));
                            fin_u.set("model", JsonValue::make_string(model_id));
                            fin_u.set("choices", JsonValue::make_array());
                            fin_u.set("usage", make_usage_json(stats, r_toks));
                            send_sse_json(cfd, fin_u);
                            send_sse_done(cfd);
                        }
                    } else {
                        std::string raw_out;
                        rt.generate_stream(prompt_ids, max_tok, sp, {}, &g_shutdown_req,
                            [&](int32_t, const std::string& piece) -> bool {
                                raw_out += piece;
                                return true;
                            }, stats);
                        const auto parsed = Ernie55StreamParser::parse_complete(raw_out, tools, enable_thinking, parallel_tc);
                        JsonValue msg = JsonValue::make_object();
                        msg.set("role", JsonValue::make_string("assistant"));
                        if (!parsed.reasoning.empty()) msg.set("reasoning_content", JsonValue::make_string(parsed.reasoning));
                        msg.set("content", JsonValue::make_string(parsed.content));
                        if (!parsed.tool_calls.empty()) msg.set("tool_calls", tool_calls_to_json(parsed.tool_calls, false));

                        JsonValue ch = JsonValue::make_object();
                        ch.set("index", JsonValue::make_int(0));
                        ch.set("message", std::move(msg));
                        ch.set("finish_reason", JsonValue::make_string(!parsed.tool_calls.empty() ? "tool_calls" : stats.finish_reason));
                        JsonValue chs = JsonValue::make_array();
                        chs.arr_val.push_back(std::move(ch));

                        JsonValue resp = JsonValue::make_object();
                        resp.set("id", JsonValue::make_string(cid));
                        resp.set("object", JsonValue::make_string("chat.completion"));
                        resp.set("created", JsonValue::make_int(ts));
                        resp.set("model", JsonValue::make_string(model_id));
                        resp.set("choices", std::move(chs));
                        resp.set("usage", make_usage_json(stats));
                        send_http_json(cfd, 200, "OK", resp.dumps_compact());
                    }
                    std::printf("[ERNIE55-FE] /v1/chat/completions prompt=%d cached=%d gen=%d tok=%.2fms ttft=%.1fms dec=%.2f tok/s spec=%llu/%llu\n",
                                stats.prompt_tokens, stats.cached_prefix_tokens, stats.generated_tokens, tok_ms, stats.ttft_ms, stats.decode_tok_s,
                                (unsigned long long)stats.spec_accepted, (unsigned long long)stats.spec_drafted);
                    std::fflush(stdout);
                } catch (const std::exception& ex) {
                    JsonValue err = JsonValue::make_object();
                    err.set("error", JsonValue::make_string(ex.what()));
                    send_http_json(cfd, 400, "Bad Request", err.dumps_compact());
                }
                ::close(cfd);
                continue;
            }

            send_http_json(cfd, 404, "Not Found", "{\"error\":\"not found\"}");
            ::close(cfd);
        }

        if (lfd >= 0) ::close(lfd);
        std::printf("[ERNIE55-FE] Graceful shutdown: flushing .kvblk cache & draining MI50 + P150A state...\n");
        rt.flush_kvblk_cache();
        rt.shutdown();
        std::printf("[ERNIE55-FE] Clean exit.\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[ERNIE55-FE] FATAL: %s\n", ex.what());
        return 1;
    }
}
