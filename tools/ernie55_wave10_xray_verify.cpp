// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Wave-10 Release Gate & X-Ray Verifier for
// baidu/ERNIE-5.5-Flash-Preview WHT128-HLUT4 (gfx906 / MI50, zero Python anywhere).
//
// Zero dependencies beyond libstdc++ / POSIX. Four independent gates:
//   (a) Server-log telemetry X-ray: [ERNIE55-CHUNK] / [ERNIE55-KVBLK] / [KVBLK-READ] /
//       [ERNIE55-FE] parsing with min/median/p95/max M=2048 prefill ms & tok/s,
//       hca/dsa/moe sub-breakdowns, DDR4 + NVMe .kvblk restore tok/s, and
//       DSpark acceptance ratio / decode tok/s.
//   (b) Static zero-BAR1-P2P scan across mi50/ and tools/ (hard fail on any peer
//       access / peer memcpy call site).
//   (c) gfx906 ISA zero-scratch & zero-slc audit over compiled HIP assembly
//       (build/*.s) and, when present, AMDGPU metadata notes inside ELF objects.
//   (d) Live read-only /health probe (GET only - never POSTs to /v1/chat/completions)
//       plus sysfs PCIe AER counters and a GPU temperature / VRAM headroom summary.
//
// Usage:
//   bin/ernie55_wave10_xray_verify [--log PATH] [--root DIR] [--health URL]
//                                  [--json PATH] [--isa-dir DIR]
//                                  [--no-aer] [--no-health] [--no-isa]

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
std::string now_utc() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    ::gmtime_r(&t, &tm);
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '\"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20u) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", unsigned(c));
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

// Floating point value that follows `key` (e.g. "hca="), with optional K/M suffix.
// A suffix only scales when it is not part of a unit word ("133.0K tok/s" scales,
// "46.21ms" does not).
bool num_after(const std::string& s, const std::string& key, double& out) {
    const size_t p = s.find(key);
    if (p == std::string::npos) return false;
    const char* b = s.c_str() + p + key.size();
    char* e = nullptr;
    const double v = std::strtod(b, &e);
    if (e == b) return false;
    double scale = 1.0;
    if ((*e == 'K' || *e == 'k' || *e == 'M' || *e == 'm') && !std::isalpha(static_cast<unsigned char>(e[1]))) {
        scale = (*e == 'K' || *e == 'k') ? 1e3 : 1e6;
    }
    out = v * scale;
    return true;
}

bool i64_after(const std::string& s, const std::string& key, long long& out) {
    const size_t p = s.find(key);
    if (p == std::string::npos) return false;
    const char* b = s.c_str() + p + key.size();
    char* e = nullptr;
    const long long v = std::strtoll(b, &e, 10);
    if (e == b) return false;
    out = v;
    return true;
}

std::string str_after(const std::string& s, const std::string& key) {
    const size_t p = s.find(key);
    if (p == std::string::npos) return std::string();
    const size_t b = p + key.size();
    size_t e = s.find(' ', b);
    if (e == std::string::npos) e = s.size();
    return s.substr(b, e - b);
}

struct Stats {
    int n = 0;
    double mn = 0.0, med = 0.0, p95 = 0.0, mx = 0.0;
};

Stats stats_of(std::vector<double> v) {
    Stats s;
    s.n = int(v.size());
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.mn = v.front();
    s.mx = v.back();
    s.med = v[(v.size() - 1) / 2];
    const size_t idx = (v.size() * 95u + 99u) / 100u;
    s.p95 = v[std::min(idx, v.size() - 1u)];
    return s;
}

// ---------------------------------------------------------------------------
// (a) Log telemetry
// ---------------------------------------------------------------------------
struct ChunkRec {
    long long pos0 = 0;
    int m = 0;
    double ms = 0.0, tps = 0.0;
    double hca = 0.0, hca_a = 0.0, hca_ar = 0.0, dsa = 0.0, dsa_fi = 0.0, dsa_mqa = 0.0;
    double moe = 0.0, moe_pre = 0.0, moe_mi50 = 0.0, moe_p150 = 0.0, moe_post = 0.0;
};

struct RestoreRec {
    bool     canonical = false; // true only for [ERNIE55-KVBLK] RESTORE
    std::string tier;
    long long reused = 0, tokens = 0, blks = 0, disk = 0;
    double ms = 0.0, tps = 0.0;
};

struct FeRec {
    long long prompt = 0, cached = 0, gen = 0, spec_acc = 0, spec_tot = 0;
    double ttft = 0.0, dec = 0.0;
};

struct LogXray {
    std::string path;
    int lines = 0;
    int n_chunk = 0, n_kvblk = 0, n_kvr = 0, n_fe = 0;
    std::vector<ChunkRec> chunks;
    std::vector<RestoreRec> restores;
    std::vector<FeRec> fes;
};

bool parse_chunk(const std::string& ln, ChunkRec& c) {
    if (ln.rfind("[ERNIE55-CHUNK]", 0) != 0) return false;
    if (!i64_after(ln, "pos0=", c.pos0)) return false;
    long long m_val = 0;
    if (!i64_after(ln, "m=", m_val)) return false;
    c.m = int(m_val);
    const size_t pm = ln.find("m=");
    if (pm == std::string::npos) return false;
    // Layout: "m=<rows> <ms>ms <tok/s> tok/s (hca=...)" - skip the row count first.
    const char* b = ln.c_str() + pm + 2;
    char* e = nullptr;
    (void)std::strtoll(b, &e, 10);
    if (e == b) return false;
    c.ms = std::strtod(e, &e);
    if (e[0] == 'm' && e[1] == 's') e += 2; // skip the "ms" unit before the tok/s column
    c.tps = std::strtod(e, &e);
    if (!num_after(ln, "hca=", c.hca)) return false;
    if (!num_after(ln, "[a=", c.hca_a)) return false;
    if (!num_after(ln, ",ar=", c.hca_ar)) return false;
    if (!num_after(ln, "dsa=", c.dsa)) return false;
    // Commit 3+ emits dsa=Xms[fi=Y,mqa=Z]; pre-C3 lines carry dsa=Xms only.
    if (ln.find("[fi=") != std::string::npos) {
        num_after(ln, "[fi=", c.dsa_fi);
        num_after(ln, "mqa=", c.dsa_mqa);
    }
    if (!num_after(ln, "moe=", c.moe)) return false;
    if (!num_after(ln, "pre=", c.moe_pre)) return false;
    if (!num_after(ln, "mi50=", c.moe_mi50)) return false;
    if (!num_after(ln, "p150=", c.moe_p150)) return false;
    if (!num_after(ln, "post=", c.moe_post)) return false;
    return true;
}

bool parse_restore(const std::string& ln, RestoreRec& r) {
    if (ln.rfind("[ERNIE55-KVBLK] RESTORE", 0) != 0) return false;
    r.canonical = true;
    r.tier = str_after(ln, "tier=");
    if (!i64_after(ln, "reused=", r.reused)) return false;
    if (!i64_after(ln, "(+", r.tokens)) return false;
    if (!i64_after(ln, "/ ", r.blks)) return false;
    if (!num_after(ln, ") ", r.ms)) return false;
    if (!num_after(ln, "ms ", r.tps)) return false;
    return true;
}

bool parse_kvread(const std::string& ln, RestoreRec& r) {
    if (ln.rfind("[KVBLK-READ]", 0) != 0) return false;
    r.tier = str_after(ln, "tier=");
    if (!i64_after(ln, "blks=", r.blks)) return false;
    if (!i64_after(ln, "(disk=", r.disk)) return false;
    if (!i64_after(ln, "tokens=", r.tokens)) return false;
    if (!num_after(ln, "ms=", r.ms)) return false;
    if (!num_after(ln, "tok_s=", r.tps)) return false;
    return true;
}

bool parse_fe(const std::string& ln, FeRec& f) {
    if (ln.rfind("[ERNIE55-FE] /v1/chat/completions", 0) != 0) return false;
    if (!i64_after(ln, "prompt=", f.prompt)) return false;
    if (!i64_after(ln, "cached=", f.cached)) return false;
    if (!i64_after(ln, "gen=", f.gen)) return false;
    if (!num_after(ln, "ttft=", f.ttft)) return false;
    if (!num_after(ln, "dec=", f.dec)) return false;
    if (!i64_after(ln, "spec=", f.spec_acc)) return false;
    const size_t ps = ln.find("spec=");
    if (ps == std::string::npos) return false;
    if (!i64_after(ln.substr(ps), "/", f.spec_tot)) return false;
    return true;
}

bool parse_log(const std::string& path, LogXray& lx) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string ln;
    while (std::getline(ifs, ln)) {
        ++lx.lines;
        ChunkRec c;
        if (parse_chunk(ln, c)) { lx.chunks.push_back(c); ++lx.n_chunk; continue; }
        RestoreRec r;
        if (parse_restore(ln, r)) { lx.restores.push_back(r); ++lx.n_kvblk; continue; }
        if (parse_kvread(ln, r)) { lx.restores.push_back(r); ++lx.n_kvr; continue; }
        FeRec f;
        if (parse_fe(ln, f)) { lx.fes.push_back(f); ++lx.n_fe; continue; }
    }
    lx.path = path;
    return true;
}

// ---------------------------------------------------------------------------
// (b) Zero-BAR1-P2P static scan
// ---------------------------------------------------------------------------
struct ScanHit {
    std::string file;
    int line = 0;
    std::string what;
};

bool has_text_ext(const fs::path& p) {
    static const char* kExt[] = {".cpp", ".hpp", ".hip", ".h", ".c", ".cc", ".md", ".txt", ".json", ".sh"};
    const std::string e = p.extension().string();
    for (const char* x : kExt) if (e == x) return true;
    return p.filename() == "Makefile";
}

std::vector<ScanHit> scan_peer_calls(const std::string& root) {
    // Needles are assembled at runtime so this file never matches its own scan.
    const std::vector<std::string> needles = {
        std::string("hipDeviceCan") + "AccessPeer",
        std::string("hipDeviceEnable") + "PeerAccess",
        std::string("hipMemcpy") + "Peer",
        std::string("hipMemcpy") + "PeerAsync",
    };
    std::vector<ScanHit> hits;
    for (const char* sub : {"mi50", "tools"}) {
        const fs::path base = fs::path(root) / sub;
        std::error_code ec;
        if (!fs::is_directory(base, ec)) continue;
        for (const auto& ent : fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied)) {
            if (!ent.is_regular_file() || !has_text_ext(ent.path())) continue;
            std::ifstream ifs(ent.path());
            if (!ifs) continue;
            std::string ln;
            int no = 0;
            while (std::getline(ifs, ln)) {
                ++no;
                for (const auto& n : needles) {
                    if (ln.find(n) != std::string::npos) hits.push_back({ent.path().string(), no, n});
                }
            }
        }
    }
    return hits;
}

// ---------------------------------------------------------------------------
// (c) gfx906 ISA zero-scratch / zero-slc audit
// ---------------------------------------------------------------------------
struct IsaAudit {
    int s_files = 0;
    int elf_files = 0;
    int kernels = 0;
    int scratch_nonzero = 0;
    int spills = 0;
    int slc = 0;
    int trig = 0;
    int notes_parsed = 0;
    std::vector<std::string> problems;
};

bool is_elf(const fs::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) return false;
    unsigned char h[4]{};
    ifs.read(reinterpret_cast<char*>(h), 4);
    return ifs.gcount() == 4 && h[0] == 0x7Fu && h[1] == 'E' && h[2] == 'L' && h[3] == 'F';
}

// Decode the msgpack value starting at `p` (bounded by `end`); only the encodings
// the AMDGPU metadata note uses are accepted.
bool mp_uint(const uint8_t* p, const uint8_t* end, uint64_t& out) {
    if (p >= end) return false;
    const uint8_t t = *p;
    if (t <= 0x7Fu) { out = t; return true; }
    if (t == 0xCCu) { if (p + 1 >= end) return false; out = p[1]; return true; }
    if (t == 0xCDu) {
        if (p + 2 >= end) return false;
        out = (uint64_t(p[1]) << 8) | uint64_t(p[2]);
        return true;
    }
    if (t == 0xCEu) {
        if (p + 4 >= end) return false;
        out = (uint64_t(p[1]) << 24) | (uint64_t(p[2]) << 16) | (uint64_t(p[3]) << 8) | uint64_t(p[4]);
        return true;
    }
    if (t == 0xCFu) {
        if (p + 8 >= end) return false;
        out = 0;
        for (int i = 0; i < 8; ++i) out = (out << 8) | uint64_t(p[1 + i]);
        return true;
    }
    return false;
}

void scan_metadata(const uint8_t* p, size_t n, IsaAudit& au, const std::string& where) {
    static const char* kScratch = "ScratchSize";
    static const char* kPriv = ".private_segment_fixed_size";
    static const char* kGrp = ".group_segment_fixed_size";
    static const char* kSgpr = ".sgpr_count";
    static const char* kVgpr = ".vgpr_count";
    const uint8_t* end = p + n;
    auto find_key = [&](const char* key, size_t& pos) -> bool {
        const size_t kl = std::strlen(key);
        for (size_t i = 0; i + kl < n; ++i) {
            if (std::memcmp(p + i, key, kl) == 0) { pos = i + kl; return true; }
        }
        return false;
    };
    bool saw_any = false;
    for (const char* k : {kScratch, kPriv, kGrp}) {
        size_t pos = 0;
        if (!find_key(k, pos)) continue;
        uint64_t v = 0;
        if (!mp_uint(p + pos, end, v)) continue;
        saw_any = true;
        if (v != 0) {
            ++au.scratch_nonzero;
            if (au.problems.size() < 32) au.problems.push_back(where + ": " + k + "=" + std::to_string(v));
        }
    }
    for (const char* k : {kSgpr, kVgpr}) {
        size_t pos = 0;
        if (!find_key(k, pos)) continue;
        uint64_t v = 0;
        if (!mp_uint(p + pos, end, v)) continue;
        saw_any = true;
        if (k == kSgpr) au.kernels += int(v > 0 ? 1 : 0);
    }
    if (saw_any) ++au.notes_parsed;
}

void audit_elf_notes(const fs::path& p, IsaAudit& au) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) return;
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (b.size() < 64 || b[0] != 0x7Fu) return;
    const bool is64 = b[4] == 2;
    auto rd16 = [&](size_t o) -> uint32_t { return uint32_t(b[o]) | (uint32_t(b[o + 1]) << 8); };
    auto rd32 = [&](size_t o) -> uint32_t {
        return uint32_t(b[o]) | (uint32_t(b[o + 1]) << 8) | (uint32_t(b[o + 2]) << 16) | (uint32_t(b[o + 3]) << 24);
    };
    auto rd64 = [&](size_t o) -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(b[o + i]) << (8 * i);
        return v;
    };
    uint64_t shoff = 0, shentsize = 0;
    uint32_t shnum = 0;
    if (is64) {
        shoff = rd64(0x28);
        shentsize = rd16(0x3A);
        shnum = rd16(0x3C);
    } else {
        shoff = rd32(0x20);
        shentsize = rd16(0x2E);
        shnum = rd16(0x30);
    }
    if (shoff == 0 || shentsize < 64 || shnum == 0 || shoff + shentsize * uint64_t(shnum) > b.size()) return;
    for (uint32_t i = 0; i < shnum; ++i) {
        const size_t so = size_t(shoff) + size_t(i) * shentsize;
        if (rd32(so + 4) != 7 /* SHT_NOTE */) continue;
        const uint64_t off = is64 ? rd64(so + 0x18) : rd32(so + 0x10);
        const uint64_t sz = is64 ? rd64(so + 0x20) : rd32(so + 0x14);
        if (off + sz > b.size() || sz < 12) continue;
        size_t o = size_t(off);
        const size_t oend = size_t(off + sz);
        while (o + 12 <= oend) {
            const uint32_t namesz = rd32(o);
            const uint32_t descsz = rd32(o + 4);
            const uint32_t ntype = rd32(o + 8);
            const size_t nameoff = o + 12;
            const size_t desc = nameoff + ((namesz + 3u) & ~3u);
            if (desc + descsz > oend) break;
            const std::string name(reinterpret_cast<const char*>(b.data() + nameoff),
                                   namesz ? std::min<size_t>(namesz, oend - nameoff) : 0);
            if (ntype == 0x100u && (name.find("AMDGPU") != std::string::npos || name.find("amdgcn") != std::string::npos)) {
                scan_metadata(b.data() + desc, size_t(descsz), au, p.filename().string());
            }
            o = desc + ((descsz + 3u) & ~3u);
        }
    }
}

void audit_isa(const std::string& root, const std::string& isa_dir, IsaAudit& au) {
    std::vector<fs::path> sfiles;
    const fs::path primary = isa_dir.empty() ? (fs::path(root) / "build") : fs::path(isa_dir);
    std::error_code ec;
    if (fs::is_directory(primary, ec)) {
        for (const auto& ent : fs::directory_iterator(primary, ec)) {
            if (ent.is_regular_file() && ent.path().extension() == ".s") sfiles.push_back(ent.path());
        }
    }
    std::sort(sfiles.begin(), sfiles.end());
    for (const auto& sf : sfiles) {
        std::ifstream ifs(sf);
        if (!ifs) continue;
        ++au.s_files;
        std::string ln;
        while (std::getline(ifs, ln)) {
            if (ln.find(".amdhsa_kernel") != std::string::npos) ++au.kernels;
            if (ln.find("ScratchSize:") != std::string::npos) {
                const size_t c = ln.rfind(':');
                if (c != std::string::npos && std::strtol(ln.c_str() + c + 1, nullptr, 10) != 0) {
                    ++au.scratch_nonzero;
                    if (au.problems.size() < 32) au.problems.push_back(sf.filename().string() + ": ScratchSize != 0");
                }
            }
            if (ln.find(".amdhsa_private_segment_fixed_size") != std::string::npos) {
                const size_t c = ln.rfind(' ');
                if (c != std::string::npos && std::strtol(ln.c_str() + c + 1, nullptr, 10) != 0) {
                    ++au.scratch_nonzero;
                    if (au.problems.size() < 32) au.problems.push_back(sf.filename().string() + ": private_segment_fixed_size != 0");
                }
            }
            if (ln.find("scratch_load") != std::string::npos || ln.find("scratch_store") != std::string::npos) ++au.spills;
            if (ln.find("v_sin_f32") != std::string::npos || ln.find("v_cos_f32") != std::string::npos) ++au.trig;
            if (ln.find("slc") != std::string::npos) {
                ++au.slc;
                if (au.problems.size() < 32) au.problems.push_back(sf.filename().string() + ": unauthorized slc");
            }
        }
    }
    for (const char* sub : {"build", "bin"}) {
        const fs::path base = fs::path(root) / sub;
        if (!fs::is_directory(base, ec)) continue;
        for (const auto& ent : fs::directory_iterator(base, ec)) {
            if (!ent.is_regular_file() || !is_elf(ent.path())) continue;
            ++au.elf_files;
            audit_elf_notes(ent.path(), au);
        }
    }
}

// ---------------------------------------------------------------------------
// (d) Live read-only /health probe + AER + GPU thermal / VRAM
// ---------------------------------------------------------------------------
bool http_get(const std::string& url, std::string& body, std::string& err) {
    std::string host = "127.0.0.1";
    int port = 80;
    std::string path = "/";
    std::string rest = url;
    const size_t pp = rest.find("://");
    if (pp != std::string::npos) rest = rest.substr(pp + 3);
    const size_t slash = rest.find('/');
    if (slash != std::string::npos) {
        path = rest.substr(slash);
        rest.resize(slash);
    }
    const size_t colon = rest.find(':');
    if (colon != std::string::npos) {
        host = rest.substr(0, colon);
        port = std::atoi(rest.c_str() + colon + 1);
    } else if (!rest.empty()) {
        host = rest;
    }
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string sport = std::to_string(port);
    if (::getaddrinfo(host.c_str(), sport.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "getaddrinfo failed for " + host;
        return false;
    }
    int fd = -1;
    for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        err = "connect failed for " + host + ":" + sport;
        return false;
    }
    timeval tv{5, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    // Read-only probe: GET only. This tool never issues POSTs of any kind.
    const std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host +
                            "\r\nAccept: application/json\r\nConnection: close\r\n\r\n";
    size_t sent = 0;
    while (sent < req.size()) {
        const ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, 0);
        if (n <= 0) { ::close(fd); err = "send failed"; return false; }
        sent += size_t(n);
    }
    std::string resp;
    char tmp[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        resp.append(tmp, size_t(n));
    }
    ::close(fd);
    const size_t sep = resp.find("\r\n\r\n");
    if (sep == std::string::npos) { err = "malformed HTTP response"; return false; }
    const size_t sp = resp.find(' ');
    if (sp != std::string::npos && resp.compare(sp + 1, 3, "200") != 0) {
        err = "HTTP status " + resp.substr(sp + 1, 3);
        return false;
    }
    body = resp.substr(sep + 4);
    return true;
}

bool json_num(const std::string& j, const std::string& key, double& out) {
    const std::string pat = "\"" + key + "\"";
    const size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    const size_t c = j.find(':', p + pat.size());
    if (c == std::string::npos) return false;
    const char* b = j.c_str() + c + 1;
    char* e = nullptr;
    const double v = std::strtod(b, &e);
    if (e == b) return false;
    out = v;
    return true;
}

bool json_str(const std::string& j, const std::string& key, std::string& out) {
    const std::string pat = "\"" + key + "\"";
    const size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    const size_t c = j.find(':', p + pat.size());
    if (c == std::string::npos) return false;
    const size_t q0 = j.find('\"', c);
    if (q0 == std::string::npos) return false;
    const size_t q1 = j.find('\"', q0 + 1);
    if (q1 == std::string::npos) return false;
    out = j.substr(q0 + 1, q1 - q0 - 1);
    return true;
}

struct HealthInfo {
    bool ok = false;
    std::string err;
    std::string status;
    double seq_len = 0.0, spec_acc = 0.0, spec_drafted = 0.0, spec_steps = 0.0;
    double last_decode = 0.0, last_prefill = 0.0;
    double ddr4_slots = 0.0, ddr4_used = 0.0, spill_bytes = 0.0;
    int n_gpu = 0;
    double min_free_mib = 0.0, max_used_mib = 0.0;
};

HealthInfo probe_health(const std::string& url) {
    HealthInfo h;
    std::string body;
    if (!http_get(url, body, h.err)) return h;
    if (!json_str(body, "status", h.status)) h.status = "unknown";
    json_num(body, "seq_len", h.seq_len);
    json_num(body, "spec_accepted", h.spec_acc);
    json_num(body, "spec_drafted", h.spec_drafted);
    json_num(body, "spec_steps", h.spec_steps);
    json_num(body, "last_decode_tok_s", h.last_decode);
    json_num(body, "last_chunk_prefill_tok_s", h.last_prefill);
    json_num(body, "kvblk_ddr4_slots", h.ddr4_slots);
    json_num(body, "kvblk_ddr4_used", h.ddr4_used);
    json_num(body, "ddr4_expert_spill_bytes", h.spill_bytes);
    const size_t a0 = body.find("\"mi50_devices\"");
    if (a0 != std::string::npos) {
        const size_t ob = body.find('[', a0);
        const size_t cb = (ob == std::string::npos) ? std::string::npos : body.find(']', ob);
        if (ob != std::string::npos && cb != std::string::npos) {
            const std::string arr = body.substr(ob, cb - ob);
            size_t p = 0;
            double free_mib = 0.0, used_mib = 0.0;
            while (p < arr.size()) {
                const size_t e = arr.find('}', p);
                if (e == std::string::npos) break;
                const std::string obj = arr.substr(p, e - p);
                if (json_num(obj, "free_mib", free_mib)) {
                    ++h.n_gpu;
                    if (h.n_gpu == 1 || free_mib < h.min_free_mib) h.min_free_mib = free_mib;
                }
                if (json_num(obj, "used_mib", used_mib)) h.max_used_mib = std::max(h.max_used_mib, used_mib);
                p = e + 1;
            }
        }
    }
    h.ok = true;
    return h;
}

struct AerInfo {
    int devices = 0;
    long long nonfatal = 0;
    long long fatal = 0;
    bool ok = true;
};

AerInfo check_aer() {
    AerInfo a;
    std::error_code ec;
    const fs::path base("/sys/bus/pci/devices");
    if (!fs::is_directory(base, ec)) return a;
    for (const auto& ent : fs::directory_iterator(base, ec)) {
        bool any = false;
        for (const char* f : {"aer_dev_nonfatal", "aer_dev_fatal"}) {
            std::ifstream ifs(ent.path() / f);
            if (!ifs) continue;
            any = true;
            std::string ln;
            while (std::getline(ifs, ln)) {
                if (ln.rfind("TOTAL_ERR_", 0) != 0) continue;
                const size_t sp = ln.rfind(' ');
                if (sp == std::string::npos) continue;
                const long long v = std::strtoll(ln.c_str() + sp + 1, nullptr, 10);
                if (f[9] == 'n') a.nonfatal += v; else a.fatal += v;
            }
        }
        if (any) ++a.devices;
    }
    a.ok = (a.nonfatal == 0 && a.fatal == 0);
    return a;
}

struct GpuInfo {
    int n_temp = 0;
    double max_temp_c = 0.0;
    int n_vram = 0;
    double min_free_mib = 0.0;
};

GpuInfo check_gpu() {
    GpuInfo g;
    std::error_code ec;
    const fs::path base("/sys/class/drm");
    if (!fs::is_directory(base, ec)) return g;
    for (const auto& ent : fs::directory_iterator(base, ec)) {
        if (ent.path().filename().string().rfind("card", 0) != 0) continue;
        const fs::path dev = ent.path() / "device";
        const fs::path hw = dev / "hwmon";
        if (fs::is_directory(hw, ec)) {
            for (const auto& h : fs::directory_iterator(hw, ec)) {
                std::ifstream ifs(h.path() / "temp1_input");
                if (!ifs) continue;
                long long milli = 0;
                ifs >> milli;
                if (milli > 0) {
                    ++g.n_temp;
                    g.max_temp_c = std::max(g.max_temp_c, double(milli) / 1000.0);
                }
            }
        }
        double used = 0.0, total = 0.0;
        {
            std::ifstream ifs(dev / "mem_info_vram_used");
            if (ifs) ifs >> used;
        }
        {
            std::ifstream ifs(dev / "mem_info_vram_total");
            if (ifs) ifs >> total;
        }
        if (total > 0.0) {
            ++g.n_vram;
            const double free_mib = (total - used) / double(1u << 20);
            if (g.n_vram == 1 || free_mib < g.min_free_mib) g.min_free_mib = free_mib;
        }
    }
    return g;
}

struct Report {
    bool log_ok = false;
    bool p2p_ok = true;
    bool isa_ok = true;
    bool health_ok = false;
    bool aer_ok = true;
    int failures = 0;
};

std::string jnum(double v, int prec = 3) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.*f", prec, v);
    return std::string(b);
}

} // namespace

int main(int argc, char** argv) {
    std::string root = ".";
    std::string log_path = "logs/ernie55_fe_w10c2_v3.log";
    std::string health_url = "http://127.0.0.1:8052/health";
    std::string json_path;
    std::string isa_dir;
    std::vector<std::string> log_paths;
    std::vector<std::pair<std::string, std::string>> commit_tags;
    bool do_aer = true, do_health = true, do_isa = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--log" && i + 1 < argc) { log_path = argv[++i]; log_paths.push_back(log_path); }
        else if (a == "--commit-tag" && i + 1 < argc) {
            const std::string t = argv[++i];
            const size_t eq = t.find('=');
            if (eq != std::string::npos) commit_tags.emplace_back(t.substr(0, eq), t.substr(eq + 1));
        }
        else if (a == "--health" && i + 1 < argc) health_url = argv[++i];
        else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (a == "--isa-dir" && i + 1 < argc) isa_dir = argv[++i];
        else if (a == "--no-aer") do_aer = false;
        else if (a == "--no-health") do_health = false;
        else if (a == "--no-isa") do_isa = false;
        else if (a == "--help") {
            std::printf("usage: %s [--log PATH] [--root DIR] [--health URL] [--json PATH]\n"
                        "          [--isa-dir DIR] [--no-aer] [--no-health] [--no-isa]\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    const std::string abs_log =
        fs::path(log_path).is_absolute() ? log_path : (fs::path(root) / log_path).string();
    Report rep;
    IsaAudit au;
    const std::vector<ScanHit> p2p = scan_peer_calls(root);
    const GpuInfo gpu = check_gpu();
    const AerInfo aer = do_aer ? check_aer() : AerInfo{};
    HealthInfo hh;
    LogXray lx;
    const bool have_log = parse_log(abs_log, lx);

    std::printf("==========================================================================================================\n");
    std::printf(" ERNIE-5.5 Wave-10 RELEASE GATE & X-RAY VERIFIER (gfx906 / MI50, zero-BAR1-P2P, zero-scratch)\n");
    std::printf(" generated_utc=%s  root=%s\n", now_utc().c_str(), root.c_str());
    std::printf("==========================================================================================================\n");

    // ---- (a) Log telemetry -------------------------------------------------
    std::printf("\n[a] SERVER-LOG TELEMETRY X-RAY  (%s)\n", abs_log.c_str());
    if (!have_log) {
        std::printf("  [FAIL] cannot read log file\n");
        ++rep.failures;
    } else if (lx.n_chunk == 0) {
        std::printf("  [FAIL] no [ERNIE55-CHUNK] records parsed from %d lines\n", lx.lines);
        ++rep.failures;
    } else {
        std::vector<double> all_ms, all_tps, m2048_ms, m2048_tps;
        std::vector<double> v_hca, v_a, v_ar, v_dsa, v_moe, v_pre, v_mi50, v_p150, v_post;
        for (const auto& c : lx.chunks) {
            all_ms.push_back(c.ms);
            all_tps.push_back(c.tps);
            v_hca.push_back(c.hca);
            v_a.push_back(c.hca_a);
            v_ar.push_back(c.hca_ar);
            v_dsa.push_back(c.dsa);
            v_moe.push_back(c.moe);
            v_pre.push_back(c.moe_pre);
            v_mi50.push_back(c.moe_mi50);
            v_p150.push_back(c.moe_p150);
            v_post.push_back(c.moe_post);
            if (c.m == 2048) {
                m2048_ms.push_back(c.ms);
                m2048_tps.push_back(c.tps);
            }
        }
        const Stats s_all_ms = stats_of(all_ms), s_all_tps = stats_of(all_tps);
        const Stats s2_ms = stats_of(m2048_ms), s2_tps = stats_of(m2048_tps);
        const Stats sh = stats_of(v_hca), sa = stats_of(v_a), sar = stats_of(v_ar), sd = stats_of(v_dsa);
        const Stats sm = stats_of(v_moe), sp = stats_of(v_pre), smi = stats_of(v_mi50);
        const Stats spp = stats_of(v_p150), spo = stats_of(v_post);
        std::printf("  chunks=%d (M=2048: %d)  kvblk=%d  kvblk_read=%d  fe=%d  lines=%d\n",
                    lx.n_chunk, int(m2048_ms.size()), lx.n_kvblk, lx.n_kvr, lx.n_fe, lx.lines);
        std::printf("  M=2048 prefill ms   : min=%.1f med=%.1f p95=%.1f max=%.1f  (n=%d)\n",
                    s2_ms.mn, s2_ms.med, s2_ms.p95, s2_ms.mx, s2_ms.n);
        std::printf("  M=2048 prefill tok/s: min=%.1f med=%.1f p95=%.1f max=%.1f  (n=%d)\n",
                    s2_tps.mn, s2_tps.med, s2_tps.p95, s2_tps.mx, s2_tps.n);
        std::printf("  all-chunk ms        : min=%.1f med=%.1f p95=%.1f max=%.1f  (n=%d)\n",
                    s_all_ms.mn, s_all_ms.med, s_all_ms.p95, s_all_ms.mx, s_all_ms.n);
        std::printf("  all-chunk tok/s     : min=%.1f med=%.1f p95=%.1f max=%.1f  (n=%d)\n",
                    s_all_tps.mn, s_all_tps.med, s_all_tps.p95, s_all_tps.mx, s_all_tps.n);
        std::printf("  hca ms med=%.1f p95=%.1f max=%.1f | attn med=%.1f p95=%.1f | allreduce med=%.1f p95=%.1f\n",
                    sh.med, sh.p95, sh.mx, sa.med, sa.p95, sar.med, sar.p95);
        std::printf("  dsa ms med=%.1f p95=%.1f max=%.1f\n", sd.med, sd.p95, sd.mx);
        std::printf("  moe ms med=%.1f p95=%.1f max=%.1f | pre=%.1f mi50=%.1f p150=%.1f post=%.1f\n",
                    sm.med, sm.p95, sm.mx, sp.med, smi.med, spp.med, spo.med);
        for (const auto& r : lx.restores) {
            std::printf("  restore tier=%-12s tokens=%-7lld blks=%-3lld disk=%-3lld %8.2f ms  %.1fK tok/s\n",
                        r.tier.c_str(), r.tokens, r.blks, r.disk, r.ms, r.tps / 1000.0);
        }
        std::vector<double> nvme_tps, ddr4_tps;
        for (const auto& r : lx.restores) {
            if (r.tier.find("NVMe") != std::string::npos) nvme_tps.push_back(r.tps);
            else if (r.tier.find("DDR4") != std::string::npos) ddr4_tps.push_back(r.tps);
        }
        if (!nvme_tps.empty()) {
            const Stats s = stats_of(nvme_tps);
            std::printf("  NVMe .kvblk restore  : n=%d min=%.1fK med=%.1fK max=%.1fK tok/s\n",
                        s.n, s.mn / 1000.0, s.med / 1000.0, s.mx / 1000.0);
        }
        if (!ddr4_tps.empty()) {
            const Stats s = stats_of(ddr4_tps);
            std::printf("  DDR4 .kvblk restore  : n=%d min=%.1fK med=%.1fK max=%.1fK tok/s\n",
                        s.n, s.mn / 1000.0, s.med / 1000.0, s.mx / 1000.0);
        }
        long long spec_acc = 0, spec_tot = 0;
        std::vector<double> dec;
        for (const auto& f : lx.fes) {
            spec_acc += f.spec_acc;
            spec_tot += f.spec_tot;
            dec.push_back(f.dec);
        }
        if (!dec.empty()) {
            const Stats s = stats_of(dec);
            std::printf("  DSpark acceptance    : %lld/%lld = %.4f | decode tok/s min=%.2f med=%.2f max=%.2f\n",
                        spec_acc, spec_tot, spec_tot > 0 ? double(spec_acc) / double(spec_tot) : 0.0,
                        s.mn, s.med, s.mx);
        }
        rep.log_ok = true;
    }

    // ---- (b) Zero-BAR1-P2P static scan -------------------------------------
    std::printf("\n[b] ZERO-BAR1-P2P STATIC SCAN  (mi50/ + tools/)\n");
    if (p2p.empty()) {
        std::printf("  [PASS] 0 peer-access / peer-memcpy call sites\n");
    } else {
        rep.p2p_ok = false;
        ++rep.failures;
        std::printf("  [FAIL] %zu P2P call site(s):\n", p2p.size());
        for (size_t i = 0; i < p2p.size() && i < 16; ++i) {
            std::printf("         %s:%d  %s\n", p2p[i].file.c_str(), p2p[i].line, p2p[i].what.c_str());
        }
    }

    // ---- (c) ISA zero-scratch / zero-slc audit ------------------------------
    std::printf("\n[c] GFX906 ISA ZERO-SCRATCH & ZERO-SLC AUDIT\n");
    if (!do_isa) {
        std::printf("  [SKIP] --no-isa\n");
    } else {
        audit_isa(root, isa_dir, au);
        std::printf("  .s files=%d  elf files=%d  kernels=%d  metadata notes=%d\n",
                    au.s_files, au.elf_files, au.kernels, au.notes_parsed);
        std::printf("  scratch>0=%d  spills=%d  trig=%d  slc=%d\n",
                    au.scratch_nonzero, au.spills, au.trig, au.slc);
        if (au.s_files == 0 && au.notes_parsed == 0) {
            std::printf("  [SKIP] no build/*.s and no AMDGPU metadata notes; run `make audit-gfx906` first\n");
        } else if (au.scratch_nonzero == 0 && au.spills == 0 && au.slc == 0 && au.trig == 0) {
            std::printf("  [PASS] every kernel: ScratchSize==0, 0 spills, 0 slc, 0 trig\n");
        } else {
            rep.isa_ok = false;
            ++rep.failures;
            std::printf("  [FAIL] ISA violations:\n");
            for (const auto& p : au.problems) std::printf("         %s\n", p.c_str());
        }
    }

    // ---- (d) Live /health + AER + GPU ---------------------------------------
    std::printf("\n[d] LIVE READ-ONLY HEALTH PROBE + PCIe AER + GPU THERMAL/VRAM\n");
    if (!do_health) {
        std::printf("  [SKIP] --no-health\n");
    } else {
        hh = probe_health(health_url);
        if (!hh.ok) {
            ++rep.failures;
            std::printf("  [FAIL] GET %s -> %s\n", health_url.c_str(), hh.err.c_str());
        } else {
            rep.health_ok = true;
            const double acc = hh.spec_drafted > 0.0 ? hh.spec_acc / hh.spec_drafted : 0.0;
            std::printf("  [PASS] GET %s -> status=%s seq_len=%.0f\n",
                        health_url.c_str(), hh.status.c_str(), hh.seq_len);
            std::printf("         DSpark spec %.0f/%.0f = %.4f (steps=%.0f) | last_prefill=%.1f tok/s last_decode=%.2f tok/s\n",
                        hh.spec_acc, hh.spec_drafted, acc, hh.spec_steps, hh.last_prefill, hh.last_decode);
            std::printf("         kvblk_ddr4 slots=%.0f used=%.0f | ddr4_expert_spill_bytes=%.0f\n",
                        hh.ddr4_slots, hh.ddr4_used, hh.spill_bytes);
            std::printf("         MI50s=%d  max_used=%.0f MiB  min_free=%.0f MiB\n",
                        hh.n_gpu, hh.max_used_mib, hh.min_free_mib);
        }
    }
    if (!do_aer) {
        std::printf("  [SKIP] --no-aer\n");
    } else if (aer.ok) {
        std::printf("  [PASS] PCIe AER: nonfatal=0 fatal=0 over %d devices\n", aer.devices);
    } else {
        rep.aer_ok = false;
        ++rep.failures;
        std::printf("  [FAIL] PCIe AER: nonfatal=%lld fatal=%lld over %d devices\n",
                    aer.nonfatal, aer.fatal, aer.devices);
    }
    std::printf("  GPU thermal/VRAM: temp nodes=%d max=%.1f C | vram nodes=%d min_free=%.0f MiB\n",
                gpu.n_temp, gpu.max_temp_c, gpu.n_vram, gpu.min_free_mib);
    if (gpu.n_temp > 0 && gpu.max_temp_c > 85.0) std::printf("  [WARN] GPU temperature above 85 C\n");
    if (gpu.n_vram > 0 && gpu.min_free_mib < 256.0) {
        std::printf("  [WARN] GPU VRAM headroom below 256 MiB (live server resident)\n");
    }


    // ---- Wave-10 campaign aggregation across every --log ----
    struct BucketStat { double tps = 0.0; long long n = 0; double hca = 0.0, dsa = 0.0, fi = 0.0, mqa = 0.0, moe = 0.0; };
    const char* kBuckets[8] = {"<=64K", "64K-128K", "128K-192K", "192K-256K",
                               "256K-320K", "320K-384K", "384K-512K", ">512K"};
    BucketStat bstat[8];
    long long c_spec_acc = 0, c_spec_drafted = 0, c_dec_n = 0;
    double c_dec_sum = 0.0;
    std::vector<std::pair<std::string, LogXray>> campaign;
    auto label_for = [&](const std::string& path, size_t idx) -> std::string {
        for (const auto& t : commit_tags) if (t.second == path) return t.first;
        const std::string b = fs::path(path).filename().string();
        const size_t p = b.find("w10c");
        if (p != std::string::npos && p + 4 < b.size()) return std::string("c") + b.substr(p + 4, 1);
        return "log" + std::to_string(idx);
    };
    for (size_t i = 0; i < log_paths.size(); ++i) {
        LogXray lx2;
        if (parse_log(log_paths[i], lx2)) campaign.emplace_back(label_for(log_paths[i], i), std::move(lx2));
    }
    if (campaign.empty()) {
        LogXray lx2;
        if (parse_log(abs_log, lx2)) campaign.emplace_back(label_for(abs_log, 0), std::move(lx2));
    }
    auto bucket_of = [](long long ctx) -> int {
        if (ctx <= 65536) return 0;
        if (ctx <= 131072) return 1;
        if (ctx <= 196608) return 2;
        if (ctx <= 262144) return 3;
        if (ctx <= 327680) return 4;
        if (ctx <= 393216) return 5;
        if (ctx <= 524288) return 6;
        return 7;
    };
    for (auto& ent : campaign) {
        for (const auto& c : ent.second.chunks) {
            if (c.m < 1024) continue;
            const int b = bucket_of(c.pos0 + c.m);
            bstat[b].tps += c.tps;
            ++bstat[b].n;
            bstat[b].hca += c.hca;
            bstat[b].dsa += c.dsa;
            bstat[b].fi  += c.dsa_fi;
            bstat[b].mqa += c.dsa_mqa;
            bstat[b].moe += c.moe;
        }
        for (const auto& f : ent.second.fes) {
            c_spec_acc += f.spec_acc;
            c_spec_drafted += f.spec_tot;
            c_dec_sum += f.dec;
            ++c_dec_n;
        }
    }
    std::printf("\n[e] WAVE-10 CAMPAIGN AGGREGATION (%zu logs)\n", campaign.size());
    for (const auto& ent : campaign) {
        std::printf("  %-8s chunks=%-4d M>=1024=%-4d kvblk=%-3d fe=%-4d\n",
                    ent.first.c_str(), ent.second.n_chunk,
                    int(std::count_if(ent.second.chunks.begin(), ent.second.chunks.end(),
                                      [](const ChunkRec& c) { return c.m >= 1024; })),
                    ent.second.n_kvblk + ent.second.n_kvr, ent.second.n_fe);
    }
    std::printf("  %-11s %6s %9s %9s %9s %9s %9s %9s\n",
                "bucket", "n", "tok/s", "hca", "dsa", "fi", "mqa", "moe");
    for (int b = 0; b < 8; ++b) {
        if (bstat[b].n == 0) continue;
        const double inv = 1.0 / double(bstat[b].n);
        std::printf("  %-11s %6lld %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n",
                    kBuckets[b], bstat[b].n, bstat[b].tps * inv, bstat[b].hca * inv,
                    bstat[b].dsa * inv, bstat[b].fi * inv, bstat[b].mqa * inv, bstat[b].moe * inv);
    }
    std::printf("  DSpark cumulative: %lld/%lld = %.4f | decode tok/s mean=%.2f (n=%lld)\n",
                c_spec_acc, c_spec_drafted,
                c_spec_drafted > 0 ? double(c_spec_acc) / double(c_spec_drafted) : 0.0,
                c_dec_n > 0 ? c_dec_sum / double(c_dec_n) : 0.0, c_dec_n);
    std::vector<std::string> kv_recs;
    std::vector<std::string> traj;
    long long peak_ctx = 0;
    int traj_step = 0;
    for (const auto& ent : campaign) {
        for (const auto& f : ent.second.fes) {
            char tb[256];
            std::snprintf(tb, sizeof(tb),
                "{\"step\": %d, \"commit\": \"%s\", \"prompt_tokens\": %lld, \"cached_tokens\": %lld, "
                "\"generated_tokens\": %lld, \"decode_tok_s\": %.2f, \"spec_accepted\": %lld, \"spec_drafted\": %lld}",
                traj_step++, ent.first.c_str(), f.prompt, f.cached, f.gen, f.dec, f.spec_acc, f.spec_tot);
            traj.emplace_back(tb);
            peak_ctx = std::max(peak_ctx, f.prompt);
        }
        for (const auto& r : ent.second.restores) {
            if (!r.canonical) continue; // one record per restore, from [ERNIE55-KVBLK] only
            char buf[160];
            std::snprintf(buf, sizeof(buf), "%s,%s,%lld,%lld,%.2f,%.1f",
                          ent.first.c_str(), r.tier.c_str(), r.tokens, r.blks, r.ms, r.tps);
            kv_recs.emplace_back(buf);
        }
    }

    // ---- Release-hygiene gate (SPDX / zero-Python / Dfly2 hook) ----
    int n_hyg = 0, spdx_bad = 0, py_bad = 0;
    bool dfly2_decl = false;
    for (const char* d : {"src", "mi50", "p150", "tools"}) {
        const fs::path base = fs::path(root) / d;
        std::error_code hec;
        if (!fs::is_directory(base, hec)) continue;
        for (const auto& ent : fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied)) {
            if (!ent.is_regular_file()) continue;
            const std::string ext = ent.path().extension().string();
            if (ext == ".py") { ++py_bad; continue; }
            if (ext != ".cpp" && ext != ".hpp" && ext != ".h" && ext != ".hip") continue;
            ++n_hyg;
            std::ifstream ifs(ent.path());
            std::string head;
            for (int i = 0; i < 40 && std::getline(ifs, head); ++i) {
                if (head.find("SPDX-License-Identifier: Apache-2.0") != std::string::npos) break;
                head.clear();
            }
            if (head.find("SPDX-License-Identifier: Apache-2.0") == std::string::npos) {
                ++spdx_bad;
                std::printf("  [FAIL H1] missing SPDX header: %s\n", ent.path().filename().string().c_str());
            }
        }
    }
    {
        std::ifstream ifs(fs::path(root) / "mi50/ernie55_runtime.hpp");
        std::string all((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        dfly2_decl = (all.find("Dfly2SharedSparsityView") != std::string::npos);
    }
    const bool hyg_ok = (spdx_bad == 0 && py_bad == 0 && dfly2_decl);
    std::printf("\n[f] RELEASE HYGIENE: sources=%d spdx_missing=%d py_files=%d dfly2_hook=%s -> %s\n",
                n_hyg, spdx_bad, py_bad, dfly2_decl ? "yes" : "no", hyg_ok ? "PASS" : "FAIL");
    if (!hyg_ok) ++rep.failures;

    std::string campaign_json;
    {
        char buf[512];
        campaign_json += "  \"wave10_campaign\": {\n";
        campaign_json += "    \"log_count\": " + std::to_string(campaign.size()) + ",\n";
        campaign_json += "    \"commits\": [";
        for (size_t i = 0; i < campaign.size(); ++i) {
            if (i) campaign_json += ", ";
            campaign_json += "\"" + json_escape(campaign[i].first) + "\"";
        }
        campaign_json += "],\n";
        std::vector<std::string> bucket_json;
        for (int b = 0; b < 8; ++b) {
            if (bstat[b].n == 0) continue;
            const double inv = 1.0 / double(bstat[b].n);
            std::snprintf(buf, sizeof(buf),
                "{\"bucket\": \"%s\", \"n\": %lld, \"prefill_tok_s\": %.3f, \"hca_ms\": %.3f, "
                "\"dsa_ms\": %.3f, \"dsa_fi_ms\": %.3f, \"dsa_mqa_ms\": %.3f, \"moe_ms\": %.3f}",
                kBuckets[b], bstat[b].n, bstat[b].tps * inv, bstat[b].hca * inv,
                bstat[b].dsa * inv, bstat[b].fi * inv, bstat[b].mqa * inv, bstat[b].moe * inv);
            bucket_json.emplace_back(buf);
        }
        campaign_json += "    \"context_buckets\": [";
        for (size_t i = 0; i < bucket_json.size(); ++i) {
            if (i) campaign_json += ", ";
            campaign_json += bucket_json[i];
        }
        campaign_json += "],\n";
        std::snprintf(buf, sizeof(buf),
            "    \"dspark_spec_accepted\": %lld, \"dspark_spec_drafted\": %lld, "
            "\"dspark_acceptance\": %.6f, \"decode_tok_s_mean\": %.4f,\n",
            c_spec_acc, c_spec_drafted,
            c_spec_drafted > 0 ? double(c_spec_acc) / double(c_spec_drafted) : 0.0,
            c_dec_n > 0 ? c_dec_sum / double(c_dec_n) : 0.0);
        campaign_json += buf;
        campaign_json += "    \"trajectory\": [";
        for (size_t i = 0; i < traj.size(); ++i) {
            if (i) campaign_json += ", ";
            campaign_json += traj[i];
        }
        campaign_json += "],\n";
        std::snprintf(buf, sizeof(buf), "    \"peak_context_tokens\": %lld,\n", peak_ctx);
        campaign_json += buf;
        campaign_json += "    \"kvblk_restore\": [";
        for (size_t i = 0; i < kv_recs.size(); ++i) {
            if (i) campaign_json += ", ";
            std::string rec = kv_recs[i];
            std::vector<std::string> f;
            size_t s0 = 0;
            for (int k = 0; k < 6; ++k) { size_t c = rec.find(',', s0); f.push_back(rec.substr(s0, c - s0)); s0 = c + 1; }
            campaign_json += std::string("{\"commit\": \"") + json_escape(f[0]) + "\", \"tier\": \"" + json_escape(f[1]) +
                             "\", \"tokens\": " + f[2] + ", \"blks\": " + f[3] +
                             ", \"ms\": " + f[4] + ", \"tok_s\": " + f[5] + "}";
        }
        campaign_json += "],\n";
        std::snprintf(buf, sizeof(buf),
            "    \"hygiene\": {\"sources\": %d, \"spdx_missing\": %d, \"py_files\": %d, \"dfly2_hook\": %s, \"ok\": %s}\n",
            n_hyg, spdx_bad, py_bad, dfly2_decl ? "true" : "false", hyg_ok ? "true" : "false");
        campaign_json += buf;
        campaign_json += "  },\n";
    }

    // ---- Optional JSON report ----------------------------------------------
    if (!json_path.empty()) {
        std::ofstream js(json_path, std::ios::binary);
        if (!js) {
            std::fprintf(stderr, "[FAIL] cannot write JSON report %s\n", json_path.c_str());
            ++rep.failures;
        } else {
            js << "{\n";
            js << "  \"tool\": \"ernie55_wave10_xray_verify\",\n";
            js << "  \"generated_utc\": \"" << now_utc() << "\",\n";
            js << "  \"root\": \"" << json_escape(root) << "\",\n";
            js << "  \"log\": {\n";
            js << "    \"path\": \"" << json_escape(abs_log) << "\",\n";
            js << "    \"lines\": " << lx.lines << ",\n";
            js << "    \"chunks\": " << lx.n_chunk << ",\n";
            js << "    \"kvblk_restore\": " << lx.n_kvblk << ",\n";
            js << "    \"kvblk_read\": " << lx.n_kvr << ",\n";
            js << "    \"fe\": " << lx.n_fe << ",\n";
            js << "    \"ok\": " << (rep.log_ok ? "true" : "false") << "\n";
            js << "  },\n";
            js << "  \"p2p_scan\": {\"hits\": " << p2p.size() << ", \"ok\": " << (rep.p2p_ok ? "true" : "false") << "},\n";
            js << "  \"isa_audit\": {\"s_files\": " << au.s_files << ", \"elf_files\": " << au.elf_files
               << ", \"kernels\": " << au.kernels << ", \"scratch_nonzero\": " << au.scratch_nonzero
               << ", \"spills\": " << au.spills << ", \"slc\": " << au.slc << ", \"trig\": " << au.trig
               << ", \"ok\": " << (rep.isa_ok ? "true" : "false") << "},\n";
            js << "  \"health\": {\"ok\": " << (rep.health_ok ? "true" : "false")
               << ", \"url\": \"" << json_escape(health_url) << "\"";
            if (hh.ok) {
                js << ", \"status\": \"" << json_escape(hh.status) << "\""
                   << ", \"seq_len\": " << jnum(hh.seq_len, 0)
                   << ", \"spec_accepted\": " << jnum(hh.spec_acc, 0)
                   << ", \"spec_drafted\": " << jnum(hh.spec_drafted, 0)
                   << ", \"last_decode_tok_s\": " << jnum(hh.last_decode)
                   << ", \"last_prefill_tok_s\": " << jnum(hh.last_prefill)
                   << ", \"n_gpu\": " << hh.n_gpu
                   << ", \"min_free_mib\": " << jnum(hh.min_free_mib, 1)
                   << ", \"max_used_mib\": " << jnum(hh.max_used_mib, 1);
            } else {
                js << ", \"error\": \"" << json_escape(hh.err) << "\"";
            }
            js << "},\n";
            js << "  \"aer\": {\"devices\": " << aer.devices << ", \"nonfatal\": " << aer.nonfatal
               << ", \"fatal\": " << aer.fatal << ", \"ok\": " << (aer.ok ? "true" : "false") << "},\n";
            js << "  \"gpu\": {\"temp_nodes\": " << gpu.n_temp << ", \"max_temp_c\": " << jnum(gpu.max_temp_c, 1)
               << ", \"vram_nodes\": " << gpu.n_vram << ", \"min_free_mib\": " << jnum(gpu.min_free_mib, 1) << "},\n";
            js << campaign_json;
            js << "  \"failures\": " << rep.failures << ",\n";
            js << "  \"result\": \"" << (rep.failures == 0 ? "PASS" : "FAIL") << "\"\n";
            js << "}\n";
            std::printf("\n  JSON report written: %s\n", json_path.c_str());
        }
    }

    std::printf("\n==========================================================================================================\n");
    std::printf(" WAVE-10 X-RAY GATE: %s (failures=%d)\n", rep.failures == 0 ? "PASS" : "FAIL", rep.failures);
    std::printf("==========================================================================================================\n");
    return rep.failures == 0 ? 0 : 1;
}
