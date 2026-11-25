// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Disjoint 512K-Token Calibration Corpus & 64-Probe Held-Out GATE-2 Builder + 8-Gram Auditor
// Zero Python anywhere. Uses verified ERNIE-5.5 PCRE2 R2L 3-digit grouping 201K BPE tokenizer.

#include "../src/ernie55_config.hpp"
#include "../src/ernie55_hlut4_pack.hpp"
#include "../src/ernie55_json.hpp"
#include "../src/ernie55_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ernie55 {
namespace {

namespace fs = std::filesystem;

constexpr uint32_t kCalBinMagic   = 0x43414C35u;  // "CAL5"
constexpr uint32_t kProbeBinMagic = 0x50524235u;  // "PRB5"

struct CalCorpusHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_sequences;
    uint32_t seq_len;
    uint32_t vocab_size;
    uint32_t crc32c;
    uint64_t total_tokens;
};

struct ProbeFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_probes;
    uint32_t probe_len;
    uint32_t vocab_size;
    uint32_t crc32c;
};

inline uint64_t hash_8gram(const int32_t* p) noexcept {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int i = 0; i < 8; ++i) {
        uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(p[i])) + 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        h ^= x ^ (x >> 31);
        h *= 0x100000001B3ULL;
    }
    return h;
}

std::string read_text_file(const std::string& path, size_t max_bytes = 64ULL * 1024ULL * 1024ULL) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    std::string out;
    ifs.seekg(0, std::ios::end);
    const auto sz = ifs.tellg();
    if (sz <= 0) return {};
    const size_t n = std::min<size_t>(size_t(sz), max_bytes);
    out.resize(n);
    ifs.seekg(0, std::ios::beg);
    ifs.read(&out[0], std::streamsize(n));
    return out;
}

// Extract row strings from HF datasets-server JSON (rows[*].row.<field>)
std::vector<std::string> load_hf_rows_field(const std::string& path, const std::string& field) {
    std::vector<std::string> out;
    const std::string raw = read_text_file(path);
    if (raw.empty()) return out;
    const JsonValue j = JsonParser::parse(raw);
    if (!j.has("rows") || !j.at("rows").is_array()) return out;
    for (const auto& item : j.at("rows").arr_val) {
        if (!item.has("row") || !item.at("row").is_object()) continue;
        const auto& r = item.at("row");
        if (const JsonValue* fv = r.find(field); fv && fv->is_string() && !fv->as_string().empty()) {
            out.push_back(fv->as_string());
        }
    }
    return out;
}

// 16 held-out systems/GPU/Rust/C++ probes (NEVER included in calibration!)
const std::array<const char*, 16> kHeldoutCodeProbes = {
    "template <int BLOCK_N, int BLOCK_K>\n__global__ void tiled_gemm_bf16(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C, int M, int N, int K) {\n    __shared__ float smA[BLOCK_N][BLOCK_K + 1];\n    int row = blockIdx.y * BLOCK_N + threadIdx.y;\n    int col = blockIdx.x * BLOCK_N + threadIdx.x;\n    float acc = 0.0f;\n    for (int kt = 0; kt < K; kt += BLOCK_K) {\n",
    "pub struct RingBuffer<T, const CAP: usize> {\n    buf: [Option<T>; CAP],\n    head: usize,\n    tail: usize,\n    len: usize,\n}\nimpl<T, const CAP: usize> RingBuffer<T, CAP> {\n    pub fn push(&mut self, item: T) -> Result<(), T> {\n        if self.len == CAP { return Err(item); }\n        self.buf[self.tail] = Some(item);\n        self.tail = (self.tail + 1) % CAP;\n",
    "struct PciBarWindow {\n    int fd = -1;\n    void* mapped = nullptr;\n    size_t length = 0;\n    uint32_t read32(size_t offset) const noexcept {\n        volatile const uint32_t* reg = reinterpret_cast<volatile const uint32_t*>(static_cast<const uint8_t*>(mapped) + offset);\n        return *reg;\n    }\n    void write32(size_t offset, uint32_t val) noexcept {\n",
    "static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }\nvoid chacha20_quarter_round(uint32_t* st, int a, int b, int c, int d) {\n    st[a] += st[b]; st[d] = rotl32(st[d] ^ st[a], 16);\n    st[c] += st[d]; st[b] = rotl32(st[b] ^ st[c], 12);\n    st[a] += st[b]; st[d] = rotl32(st[d] ^ st[a], 8);\n    st[c] += st[d]; st[b] = rotl32(st[b] ^ st[c], 7);\n}\n",
    "fn radix_sort_u64(data: &mut [u64]) {\n    let mut aux = vec![0u64; data.len()];\n    for shift in (0..64).step_by(8) {\n        let mut counts = [0usize; 256];\n        for &v in data.iter() { counts[((v >> shift) & 0xFF) as usize] += 1; }\n        let mut offset = 0usize;\n        for c in counts.iter_mut() { let n = *c; *c = offset; offset += n; }\n",
    "class WorkStealingDeque {\n    std::atomic<int64_t> top_{0}, bottom_{0};\n    std::vector<std::function<void()>> ring_;\npublic:\n    explicit WorkStealingDeque(size_t pow2_cap) : ring_(pow2_cap) {}\n    void push_bottom(std::function<void()> task) {\n        int64_t b = bottom_.load(std::memory_order_relaxed);\n        ring_[size_t(b) & (ring_.size() - 1)] = std::move(task);\n        std::atomic_thread_fence(std::memory_order_release);\n",
    "__device__ __forceinline__ float warp_reduce_max64(float v) {\n    v = fmaxf(v, __shfl_xor(v, 32, 64));\n    v = fmaxf(v, __shfl_xor(v, 16, 64));\n    v = fmaxf(v, __shfl_xor(v, 8, 64));\n    v = fmaxf(v, __shfl_xor(v, 4, 64));\n    v = fmaxf(v, __shfl_xor(v, 2, 64));\n    v = fmaxf(v, __shfl_xor(v, 1, 64));\n    return v;\n}\n",
    "class LruPageCache {\n    size_t capacity_;\n    std::list<std::pair<uint64_t, std::vector<uint8_t>>> lru_;\n    std::unordered_map<uint64_t, decltype(lru_)::iterator> map_;\npublic:\n    const std::vector<uint8_t>* get(uint64_t page_id) {\n        auto it = map_.find(page_id);\n        if (it == map_.end()) return nullptr;\n        lru_.splice(lru_.begin(), lru_, it->second);\n        return &it->second->second;\n    }\n",
    "def dijkstra_shortest_paths(adj: dict[int, list[tuple[int, float]]], src: int) -> dict[int, float]:\n    import heapq\n    dist = {src: 0.0}\n    pq = [(0.0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist.get(u, float('inf')):\n            continue\n        for v, w in adj.get(u, []):\n            nd = d + w\n",
    "struct BTreeNode {\n    bool leaf = true;\n    std::vector<int64_t> keys;\n    std::vector<BTreeNode*> children;\n    void split_child(int idx, int min_deg) {\n        BTreeNode* full = children[size_t(idx)];\n        auto* right = new BTreeNode();\n        right->leaf = full->leaf;\n        right->keys.assign(full->keys.begin() + min_deg, full->keys.end());\n",
    "void fast_walsh_hadamard_64(float* x) {\n    for (int len = 1; len < 64; len <<= 1) {\n        for (int i = 0; i < 64; i += (len << 1)) {\n            for (int j = 0; j < len; ++j) {\n                float u = x[i + j];\n                float v = x[i + j + len];\n                x[i + j] = u + v;\n                x[i + j + len] = u - v;\n            }\n        }\n    }\n}\n",
    "struct ArenaAllocator {\n    uint8_t* base;\n    size_t cap, off;\n    void* alloc_aligned(size_t bytes, size_t align) {\n        size_t aligned = (off + align - 1) & ~(align - 1);\n        if (aligned + bytes > cap) return nullptr;\n        off = aligned + bytes;\n        return base + aligned;\n    }\n    void reset() noexcept { off = 0; }\n};\n",
    "async fn forward_tcp_stream(mut client: tokio::net::TcpStream, upstream_addr: &str) -> std::io::Result<(u64, u64)> {\n    let mut server = tokio::net::TcpStream::connect(upstream_addr).await?;\n    client.set_nodelay(true)?;\n    server.set_nodelay(true)?;\n    let (rx, tx) = tokio::io::copy_bidirectional(&mut client, &mut server).await?;\n    Ok((rx, tx))\n}\n",
    "int parse_http_content_length(const char* headers, size_t len) {\n    const char* p = headers;\n    const char* end = headers + len;\n    while (p < end) {\n        const char* nl = static_cast<const char*>(std::memchr(p, '\\n', size_t(end - p)));\n        size_t line_len = nl ? size_t(nl - p) : size_t(end - p);\n        if (line_len > 15 && strncasecmp(p, \"Content-Length:\", 15) == 0) return std::atoi(p + 15);\n",
    "static uint64_t crc64_ecma(const uint8_t* data, size_t len, uint64_t seed) {\n    uint64_t crc = ~seed;\n    for (size_t i = 0; i < len; ++i) {\n        crc ^= uint64_t(data[i]) << 56;\n        for (int b = 0; b < 8; ++b) {\n            crc = (crc & 0x8000000000000000ULL) ? ((crc << 1) ^ 0x42F0E1EBA9EA3693ULL) : (crc << 1);\n        }\n    }\n    return ~crc;\n}\n",
    "class KalmanFilter1D {\n    double x_, p_, q_, r_;\npublic:\n    KalmanFilter1D(double x0, double p0, double q, double r) : x_(x0), p_(p0), q_(q), r_(r) {}\n    double update(double z) {\n        p_ += q_;\n        double k = p_ / (p_ + r_);\n        x_ += k * (z - x_);\n        p_ *= (1.0 - k);\n        return x_;\n    }\n};\n"
};

// 16 held-out formal math / physics / algorithms probes
const std::array<const char*, 16> kHeldoutMathProbes = {
    "Theorem (Spectral Decomposition of Real Symmetric Matrices). Let A in R^{n x n} satisfy A^T = A. Then there exists an orthogonal matrix Q with Q^T Q = I and a real diagonal matrix Lambda = diag(lambda_1, ..., lambda_n) such that A = Q Lambda Q^T. Proof: By induction on n, pick a unit eigenvector v_1 of A corresponding to the real eigenvalue lambda_1 = max_{||x||=1} x^T A x.",
    "Lemma (Kullback-Leibler Non-Negativity). For any discrete probability distributions P and Q supported on a finite alphabet V with Q(v) > 0 whenever P(v) > 0, D_KL(P || Q) = sum_{v in V} P(v) log(P(v)/Q(v)) >= 0, with equality if and only if P = Q everywhere. Proof: Apply Jensen's inequality to the strictly concave logarithm function on the ratio Q(v)/P(v).",
    "Consider the second-order linear ordinary differential equation y''(t) + 2 zeta omega_0 y'(t) + omega_0^2 y(t) = 0 with damping ratio 0 < zeta < 1. The characteristic polynomial r^2 + 2 zeta omega_0 r + omega_0^2 = 0 has complex conjugate roots r = -zeta omega_0 +/- i omega_0 sqrt(1 - zeta^2), yielding underdamped oscillatory solutions.",
    "In analytic number theory, the Euler product formula for the Riemann zeta function states that for Re(s) > 1, zeta(s) = sum_{n=1}^{infty} n^{-s} = prod_{p prime} (1 - p^{-s})^{-1}. Taking logarithms and differentiating yields -zeta'(s)/zeta(s) = sum_{n=1}^{infty} Lambda(n) n^{-s}, where Lambda is the von Mangoldt function.",
    "Let X_1, ..., X_n be independent sub-Gaussian random variables with mean zero and variance proxy sigma^2. By Hoeffding's lemma and the Chernoff bound, for every threshold t > 0 we have P(|(1/n) sum_{i=1}^n X_i| >= t) <= 2 exp(-n t^2 / (2 sigma^2)). Consequently, sample averages concentrate exponentially around their expectation.",
    "In convex optimization, suppose f: R^d -> R is continuously differentiable with L-Lipschitz gradient and mu-strongly convex. Gradient descent with step size eta = 1/L satisfies f(x_{k+1}) - f(x^*) <= (1 - mu/L) (f(x_k) - f(x^*)), giving linear convergence with iteration complexity O((L/mu) log(1/epsilon)).",
    "The discrete Fourier transform of a length-N complex sequence x_0, ..., x_{N-1} is defined by X_k = sum_{n=0}^{N-1} x_n exp(-2 pi i k n / N). When N = 2^m, the Cooley-Tukey radix-2 decimation-in-time factorization splits the sum into even and odd indices, reducing arithmetic cost from O(N^2) to O(N log N).",
    "For a Hamiltonian system with canonical coordinates (q, p) and smooth energy function H(q, p), Hamilton's equations dq/dt = partial H / partial p and dp/dt = -partial H / partial q preserve the symplectic 2-form omega = sum_i dq_i wedge dp_i. By Liouville's theorem, phase-space volume is strictly invariant under Hamiltonian flow.",
    "Let G = (V, E) be a connected undirected graph with positive edge weights w: E -> R_{>0}. Every cut (S, V \\ S) has a unique minimum-weight crossing edge when weights are distinct, and this light edge belongs to every minimum spanning tree of G. Both Prim's and Kruskal's algorithms exploit this cut property greedily.",
    "In numerical linear algebra, the condition number kappa_2(A) = sigma_max(A) / sigma_min(A) bounds the relative sensitivity of the linear system A x = b to perturbations in A or b. Householder QR factorization achieves backward stability in floating-point arithmetic without squaring the condition number, unlike forming normal equations A^T A.",
    "Consider a Markov chain on finite state space S with irreducible aperiodic transition matrix P. By the Perron-Frobenius theorem, P has a unique stationary distribution pi satisfying pi^T P = pi^T, and the total variation distance ||P^t(x, .) - pi||_TV decays geometrically at a rate controlled by the spectral gap 1 - |lambda_2|.",
    "In thermodynamics, the Helmholtz free energy F(T, V, N) = U - T S satisfies the differential relation dF = -S dT - P dV + mu dN. Consequently, pressure P = -(partial F / partial V)_{T,N} and entropy S = -(partial F / partial T)_{V,N} follow directly from partial derivatives of the canonical partition function Z = exp(-beta F).",
    "Let p be an odd prime and let a be an integer not divisible by p. By Euler's criterion, the Legendre symbol (a/p) equals a^{(p-1)/2} mod p, taking value +1 when a is a quadratic residue modulo p and -1 otherwise. Quadratic reciprocity relates (p/q)(q/p) = (-1)^{(p-1)(q-1)/4} for distinct odd primes p and q.",
    "In information theory, the rate-distortion function R(D) of a memoryless Gaussian source N(0, sigma^2) under squared-error distortion D <= sigma^2 is R(D) = (1/2) log_2(sigma^2 / D) bits per symbol. Scalar Lloyd-Max quantization with b bits approaches within roughly 0.255 bits of this Shannon bound at high resolution.",
    "For a smooth manifold M equipped with a Riemannian metric g, the Levi-Civita connection nabla is the unique affine connection that is torsion-free and metric-compatible. Geodesics satisfy nabla_{gamma'} gamma' = 0 and locally extremize arc length, governed in coordinates by the Christoffel symbols Gamma^k_{ij}.",
    "Suppose a binary linear error-correcting code C subset F_2^n has parity-check matrix H in F_2^{(n-k) x n} with minimum Hamming distance d = 2t + 1. Syndrome decoding computes s = H y^T for received word y = c + e and uniquely identifies every error pattern e of weight at most t."
};

// 16 held-out structured JSON / SQL / tool-use / tabular probes
const std::array<const char*, 16> kHeldoutStructuredProbes = {
    "{\"service\":\"order-router\",\"version\":\"4.2.1\",\"region\":\"eu-central-1\",\"limits\":{\"max_inflight\":4096,\"timeout_ms\":250,\"retry_backoff_ms\":[15,45,135]},\"circuit_breaker\":{\"enabled\":true,\"error_rate_threshold\":0.05,\"window_sec\":60},\"routes\":[{\"prefix\":\"/v2/orders\",\"pool\":\"primary-a\",\"weight\":80}]}",
    "SELECT d.department_name, COUNT(e.employee_id) AS headcount, ROUND(AVG(e.salary_usd), 2) AS avg_salary, PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY e.salary_usd) AS p95_salary FROM employees e INNER JOIN departments d ON d.department_id = e.department_id WHERE e.active = TRUE GROUP BY d.department_name HAVING COUNT(e.employee_id) >= 10 ORDER BY avg_salary DESC;",
    "{\"jsonrpc\":\"2.0\",\"id\":94821,\"method\":\"storage.write_batch\",\"params\":{\"namespace\":\"telemetry_2026\",\"compression\":\"zstd\",\"fsync\":true,\"records\":[{\"key\":\"node_07/temp_c\",\"ts\":1795401200,\"val\":61.4},{\"key\":\"node_07/power_w\",\"ts\":1795401200,\"val\":184.2}]}}",
    "| Epoch | Train Loss | Val Loss | Val Perplexity | Learning Rate | Grad Norm | Tokens/s |\n|------:|-----------:|---------:|---------------:|--------------:|----------:|---------:|\n|     1 |     3.1842 |   3.0915 |          22.01 |      3.00e-04 |     1.142 |  142,880 |\n|     2 |     2.6410 |   2.6104 |          13.60 |      2.85e-04 |     0.987 |  144,120 |\n",
    "apiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: metadata-catalog\n  namespace: storage-core\nspec:\n  serviceName: metadata-catalog\n  replicas: 3\n  selector:\n    matchLabels:\n      app: metadata-catalog\n  template:\n    spec:\n      containers:\n        - name: catalog\n          ports:\n            - containerPort: 5432\n",
    "CREATE TABLE sensor_readings (\n    device_uuid UUID NOT NULL,\n    recorded_at TIMESTAMPTZ NOT NULL,\n    voltage_mv INTEGER NOT NULL CHECK (voltage_mv BETWEEN 0 AND 60000),\n    current_ma INTEGER NOT NULL,\n    status_flags SMALLINT NOT NULL DEFAULT 0,\n    PRIMARY KEY (device_uuid, recorded_at)\n) PARTITION BY RANGE (recorded_at);",
    "{\"openapi\":\"3.1.0\",\"info\":{\"title\":\"Inventory Ledger API\",\"version\":\"2.4.0\"},\"paths\":{\"/items/{sku}/reserve\":{\"post\":{\"operationId\":\"reserveInventory\",\"parameters\":[{\"name\":\"sku\",\"in\":\"path\",\"required\":true,\"schema\":{\"type\":\"string\",\"pattern\":\"^[A-Z0-9-]{6,24}$\"}}]}}}}",
    "WITH ranked_flights AS (\n    SELECT carrier_code, origin_iata, dest_iata, dep_delay_min,\n           ROW_NUMBER() OVER (PARTITION BY origin_iata ORDER BY dep_delay_min DESC) AS rn\n    FROM flight_ops\n    WHERE flight_date BETWEEN DATE '2026-10-01' AND DATE '2026-10-31'\n)\nSELECT carrier_code, origin_iata, dest_iata, dep_delay_min FROM ranked_flights WHERE rn <= 5;",
    "{\"invoice_id\":\"INV-2026-084912\",\"currency\":\"USD\",\"issued_date\":\"2026-11-18\",\"line_items\":[{\"sku\":\"NET-QSFP-800G\",\"qty\":4,\"unit_price\":640.00,\"subtotal\":2560.00},{\"sku\":\"CBL-DAC-1M\",\"qty\":4,\"unit_price\":95.50,\"subtotal\":382.00}],\"tax_rate\":0.0825,\"total_due\":3184.72}",
    "{\n  \"type\": \"object\",\n  \"required\": [\"query\", \"filters\", \"page_size\"],\n  \"properties\": {\n    \"query\": {\"type\": \"string\", \"minLength\": 1},\n    \"page_size\": {\"type\": \"integer\", \"minimum\": 1, \"maximum\": 100},\n    \"filters\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}\n  }\n}",
    "timestamp_utc,cluster,rack,gpu_index,sm_util_pct,mem_used_mib,temp_c,pcie_aer_err\n2026-11-23T18:00:00Z,lab-a,r04,0,98.4,28412,67,0\n2026-11-23T18:00:00Z,lab-a,r04,1,97.9,28390,66,0\n2026-11-23T18:00:00Z,lab-a,r04,2,99.1,28405,68,0\n2026-11-23T18:00:00Z,lab-a,r04,3,96.8,14201,62,0\n",
    "{\"event\":\"deployment.promoted\",\"pipeline_id\":\"rel-7741\",\"commit_sha\":\"9f2b14c08d3e\",\"checks\":{\"unit\":true,\"integration\":true,\"canary_p99_ms\":18.4,\"error_budget_remaining_pct\":94.2},\"approvers\":[\"ops-bot\",\"release-lead\"]}",
    "EXPLAIN (ANALYZE, BUFFERS)\nSELECT o.account_id, SUM(o.amount_cents) AS gross_cents\nFROM ledger_entries o\nWHERE o.posted_at >= NOW() - INTERVAL '30 days' AND o.status = 'SETTLED'\nGROUP BY o.account_id\nORDER BY gross_cents DESC\nLIMIT 25;",
    "{\"tool_calls\":[{\"id\":\"call_01\",\"type\":\"function\",\"function\":{\"name\":\"query_telemetry\",\"arguments\":{\"metric\":\"pcie_rx_gbps\",\"host\":\"node-12\",\"interval_sec\":30,\"include_histograms\":false}}}],\"finish_reason\":\"tool_calls\",\"usage\":{\"prompt_tokens\":412,\"completion_tokens\":68}}",
    "[storage.nvme0]\ndevice = \"/dev/nvme0n1\"\nqueue_depth = 128\nblock_size_bytes = 4096\ndirect_io = true\ncrc32c_verify = true\n\n[storage.nvme1]\ndevice = \"/dev/nvme1n1\"\nqueue_depth = 128\nblock_size_bytes = 4096\ndirect_io = true\ncrc32c_verify = true\n",
    "{\"graph_id\":\"dag-compute-09\",\"nodes\":[{\"id\":\"ingest\",\"op\":\"stream_read\",\"threads\":8},{\"id\":\"norm\",\"op\":\"rmsnorm_f32\",\"eps\":1e-6},{\"id\":\"proj\",\"op\":\"gemm_bf16\",\"shape\":[4096,2048]}],\"edges\":[[\"ingest\",\"norm\"],[\"norm\",\"proj\"]]}"
};

void add_text_8grams(const Ernie55Tokenizer& tok, const std::string& text,
                     std::unordered_set<uint64_t>& banned, size_t& total_eval_8grams) {
    if (text.empty()) return;
    const auto ids = tok.encode(text, false);
    if (ids.size() < 8) return;
    for (size_t i = 0; i + 8 <= ids.size(); ++i) {
        banned.insert(hash_8gram(ids.data() + i));
        ++total_eval_8grams;
    }
}

} // namespace
} // namespace ernie55

int main(int argc, char** argv) {
    using namespace ernie55;
    namespace fs = std::filesystem;

    std::string root = ".";
    int target_seqs  = 512;   // 512 x 1024 = 524,288 (512K) tokens!
    int seq_len      = 1024;
    int probe_len    = 64;    // 64 probes x 64 = 4,096 evaluated next-token positions!

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--seqs" && i + 1 < argc) target_seqs = std::stoi(argv[++i]);
        else if (a == "--seq-len" && i + 1 < argc) seq_len = std::stoi(argv[++i]);
    }

    try {
        Ernie55Tokenizer tok;
        tok.load(root + "/meta/preview-bf16/tokenizer.json", root + "/meta/preview-bf16/tokenizer_config.json");
        std::printf("[CAL-BUILD] Loaded ERNIE-5.5-Flash-Preview tokenizer (vocab=%d, chat_tpl=%d)\n",
                    tok.vocab_size(), int(tok.has_chat_template()));

        // Sample OpenAI tool schema for authentic agentic tool-call calibration & held-out probes
        const JsonValue sample_tools = JsonParser::parse(R"JSON([
          {"type":"function","function":{"name":"read_telemetry","description":"Read hardware telemetry counter","parameters":{"type":"object","properties":{"device":{"type":"string"},"metric":{"type":"string"}},"required":["device"]}}},
          {"type":"function","function":{"name":"run_check","description":"Run verification command","parameters":{"type":"object","properties":{"suite":{"type":"string"},"strict":{"type":"boolean"}},"required":["suite"]}}}
        ])JSON");

        auto utf8_align_down = [](const std::string& s, size_t pos) -> size_t {
            if (pos >= s.size()) return s.size();
            while (pos > 0 && (static_cast<unsigned char>(s[pos]) & 0xC0u) == 0x80u) --pos;
            return pos;
        };
        auto utf8_substr_cap = [&](const std::string& s, size_t max_bytes) -> std::string {
            if (s.size() <= max_bytes) return s;
            return s.substr(0, utf8_align_down(s, max_bytes));
        };
        // 1. Construct 64 disjoint held-out multi-domain GATE-2 chat/reasoning/coding/tool probes
        std::vector<std::string> probe_texts;
        probe_texts.reserve(64);
        for (const char* s : kHeldoutCodeProbes)       probe_texts.emplace_back(s);
        for (const char* s : kHeldoutMathProbes)       probe_texts.emplace_back(s);
        {
            auto wen = load_hf_rows_field(root + "/bench_data/gate2_wiki_en_heldout.json", "text");
            auto wzh = load_hf_rows_field(root + "/bench_data/gate2_wiki_zh_heldout.json", "text");
            auto wru = load_hf_rows_field(root + "/bench_data/gate2_wiki_ru_heldout.json", "text");
            auto take_prose = [&](const std::vector<std::string>& src, int count) {
                int added = 0;
                for (const auto& s : src) {
                    if (s.size() < 400) continue;
                    probe_texts.push_back(utf8_substr_cap(s, 2048));
                    if (++added >= count) break;
                }
            };
            take_prose(wen, 6);
            take_prose(wzh, 5);
            take_prose(wru, 5);
        }
        for (const char* s : kHeldoutStructuredProbes) probe_texts.emplace_back(s);
        if (probe_texts.size() != 64) {
            throw std::runtime_error("Expected 64 GATE-2 held-out probes, got " + std::to_string(probe_texts.size()));
        }

        // Render all 64 held-out probes through the OFFICIAL ERNIE-5.5-Flash-Preview segment-safe chat renderer!
        std::vector<int32_t> probe_ids(size_t(64) * size_t(probe_len), tok.pad_id());
        for (int p = 0; p < 64; ++p) {
            std::vector<ChatMessage> msgs;
            if ((p % 4) == 0) {
                ChatMessage sm; sm.role = "system";
                sm.content = "You are a precise technical assistant.";
                msgs.push_back(std::move(sm));
            }
            ChatMessage um; um.role = "user";
            um.content = probe_texts[size_t(p)];
            msgs.push_back(std::move(um));
            const bool think_on = ((p % 8) != 7);
            const JsonValue* use_tools = ((p % 6) == 5) ? &sample_tools : nullptr;
            auto segs = tok.render_chat_segments(msgs, use_tools, think_on, true, "high", true);
            auto ids  = tok.encode_segments(segs);
            for (int t = 0; t < probe_len; ++t) {
                probe_ids[size_t(p) * size_t(probe_len) + size_t(t)] =
                    ids[size_t(t) % std::max<size_t>(1, ids.size())];
            }
        }

        // Write artifacts/gate2_preview_heldout_probes.bin
        fs::create_directories(root + "/artifacts");
        const std::string probe_path = root + "/artifacts/gate2_preview_heldout_probes.bin";
        {
            ProbeFileHeader ph{};
            ph.magic = kProbeBinMagic;
            ph.version = 1;
            ph.n_probes = 64;
            ph.probe_len = uint32_t(probe_len);
            ph.vocab_size = uint32_t(tok.vocab_size());
            ph.crc32c = crc32c_update(0u, probe_ids.data(), probe_ids.size() * sizeof(int32_t));
            std::ofstream ofs(probe_path, std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(&ph), sizeof(ph));
            ofs.write(reinterpret_cast<const char*>(probe_ids.data()), std::streamsize(probe_ids.size() * sizeof(int32_t)));
        }

        // 2. Collect EVERY content 8-gram across ALL 64 GATE-2 probes + GSM8K + MMLU-Pro + HumanEval!
        std::unordered_set<uint64_t> banned_8grams;
        banned_8grams.reserve(1 << 20);
        size_t eval_8grams_checked = 0;

        for (int p = 0; p < 64; ++p) {
            add_text_8grams(tok, probe_texts[size_t(p)], banned_8grams, eval_8grams_checked);
        }
        for (const auto& s : load_hf_rows_field(root + "/bench_data/gsm8k_test_100.json", "question"))
            add_text_8grams(tok, s, banned_8grams, eval_8grams_checked);
        for (const auto& s : load_hf_rows_field(root + "/bench_data/gsm8k_test_100.json", "answer"))
            add_text_8grams(tok, s, banned_8grams, eval_8grams_checked);
        for (const auto& s : load_hf_rows_field(root + "/bench_data/mmlu_pro_test_100.json", "question"))
            add_text_8grams(tok, s, banned_8grams, eval_8grams_checked);
        for (const auto& s : load_hf_rows_field(root + "/bench_data/humaneval_test_64.json", "prompt"))
            add_text_8grams(tok, s, banned_8grams, eval_8grams_checked);
        for (const auto& s : load_hf_rows_field(root + "/bench_data/humaneval_test_64_164.json", "prompt"))
            add_text_8grams(tok, s, banned_8grams, eval_8grams_checked);

        std::printf("[CAL-BUILD] Indexed %zu eval 8-grams (%zu unique) across GATE-2 + GSM8K + MMLU-Pro + HumanEval\n",
                    eval_8grams_checked, banned_8grams.size());

        // 3. Harvest diverse authentic calibration documents (EN/ZH/RU Wikipedia + C++/HIP system headers + JSON/tool examples)
        struct DomainPool {
            std::string name;
            std::vector<std::string> docs;
            size_t tokens_kept = 0;
        };
        std::vector<DomainPool> pools;
        pools.push_back({"wiki_en", load_hf_rows_field(root + "/bench_data/cal_raw/wiki_en_cal.json", "text"), 0});
        pools.push_back({"wiki_zh", load_hf_rows_field(root + "/bench_data/cal_raw/wiki_zh_cal.json", "text"), 0});
        pools.push_back({"wiki_ru", load_hf_rows_field(root + "/bench_data/cal_raw/wiki_ru_cal.json", "text"), 0});

        DomainPool code_pool{"cpp_hip_headers", {}, 0};
        for (const char* dir : {"/opt/rocm/include/hip", "/usr/include/c++/11/bits"}) {
            if (!fs::is_directory(dir)) continue;
            for (const auto& ent : fs::directory_iterator(dir)) {
                if (!ent.is_regular_file()) continue;
                const auto ext = ent.path().extension().string();
                if (ext != ".h" && ext != ".hpp" && ext != ".tcc") continue;
                std::string txt = read_text_file(ent.path().string(), 48ULL * 1024ULL);
                if (txt.size() >= 1024) code_pool.docs.push_back(std::move(txt));
                if (code_pool.docs.size() >= 80) break;
            }
        }
        pools.push_back(std::move(code_pool));

        // Filter and pack strictly 8-gram-disjoint Preview chat/thinking/tool sequences!
        const size_t total_target = size_t(target_seqs) * size_t(seq_len);
        std::vector<int32_t> cal_tokens;
        cal_tokens.reserve(total_target);

        std::vector<size_t> doc_cursor(pools.size(), 0);
        size_t chunks_rejected = 0;
        size_t seq_idx_counter = 0;
        bool made_progress = true;
        while (cal_tokens.size() < total_target && made_progress) {
            made_progress = false;
            for (size_t pi = 0; pi < pools.size() && cal_tokens.size() < total_target; ++pi) {
                auto& pool = pools[pi];
                while (doc_cursor[pi] < pool.docs.size() && cal_tokens.size() < total_target) {
                    const std::string& doc_full = pool.docs[doc_cursor[pi]++];
                    if (doc_full.size() < 2500) continue;
                    const std::string doc = utf8_substr_cap(doc_full, 18000);
                    // Build authentic ERNIE-5.5-Flash-Preview chat/thinking/tool sequence wrapping this document!
                    const size_t cut1 = utf8_align_down(doc, std::min<size_t>(doc.size() / 3, 1200u));
                    const size_t cut2 = std::max(cut1, utf8_align_down(doc, std::min<size_t>((doc.size() * 2) / 3, 2800u)));
                    std::vector<ChatMessage> msgs;
                    ChatMessage um; um.role = "user";
                    um.content = "Analyze and summarize the following technical excerpt:\n" + doc.substr(0, cut1);
                    msgs.push_back(std::move(um));

                    ChatMessage am; am.role = "assistant";
                    am.reasoning_content = "Step-by-step analysis of the provided technical material:\n" + doc.substr(cut1, cut2 - cut1);
                    am.content = doc.substr(cut2);
                    if ((seq_idx_counter % 4) == 3) {
                        ChatMessage::ToolCall tc;
                        tc.name = "read_telemetry";
                        tc.arguments_json = "{\"device\":\"gpu0\",\"metric\":\"utilization\"}";
                        am.tool_calls.push_back(std::move(tc));
                    }
                    msgs.push_back(std::move(am));
                    if ((seq_idx_counter % 4) == 3) {
                        ChatMessage tm; tm.role = "tool";
                        tm.content = "{\"status\":\"ok\",\"value\":98.4}";
                        msgs.push_back(std::move(tm));
                    }
                    const JsonValue* use_t = ((seq_idx_counter % 4) == 3) ? &sample_tools : nullptr;
                    const bool think_on = ((seq_idx_counter % 8) != 7);
                    const char* eff = ((seq_idx_counter % 3) == 0) ? "high" : (((seq_idx_counter % 3) == 1) ? "medium" : "low");
                    auto segs = tok.render_chat_segments(msgs, use_t, think_on, true, eff, true);
                    auto ids  = tok.encode_segments(segs);
                    if (int(ids.size()) < seq_len) continue;
                    // Extract up to 4 non-overlapping seq_len windows with ZERO 8-gram overlap
                    int taken_doc = 0;
                    for (size_t off = 0; off + size_t(seq_len) <= ids.size() && cal_tokens.size() < total_target; off += size_t(seq_len)) {
                        std::vector<int32_t> seq(ids.begin() + ptrdiff_t(off), ids.begin() + ptrdiff_t(off + size_t(seq_len)));
                        bool clean = true;
                        for (int k = 0; k + 8 <= seq_len; ++k) {
                            if (banned_8grams.find(hash_8gram(seq.data() + k)) != banned_8grams.end()) {
                                clean = false;
                                break;
                            }
                        }
                        if (!clean) { ++chunks_rejected; continue; }
                        cal_tokens.insert(cal_tokens.end(), seq.begin(), seq.end());
                        pool.tokens_kept += size_t(seq_len);
                        ++taken_doc;
                        ++seq_idx_counter;
                        made_progress = true;
                        if (taken_doc >= 4) break;
                    }
                    if (taken_doc > 0) break;
                }
            }
        }

        const int actual_seqs = int(cal_tokens.size() / size_t(seq_len));
        cal_tokens.resize(size_t(actual_seqs) * size_t(seq_len));

        // 4. Independent full audit pass verifying ZERO 8-gram overlap!
        size_t overlap_8grams = 0;
        for (int s = 0; s < actual_seqs; ++s) {
            const int32_t* row = cal_tokens.data() + size_t(s) * size_t(seq_len);
            for (int k = 0; k + 8 <= seq_len; ++k) {
                if (banned_8grams.find(hash_8gram(row + k)) != banned_8grams.end()) ++overlap_8grams;
            }
        }
        if (overlap_8grams != 0) {
            throw std::runtime_error("Disjointness audit FAILED: overlap_8grams=" + std::to_string(overlap_8grams));
        }

        // 5. Write artifacts/cal_ernie55_preview_disjoint.bin
        const std::string cal_path = root + "/artifacts/cal_ernie55_preview_disjoint.bin";
        const uint32_t crc = crc32c_update(0u, cal_tokens.data(), cal_tokens.size() * sizeof(int32_t));
        {
            CalCorpusHeader ch{};
            ch.magic        = kCalBinMagic;
            ch.version      = 1;
            ch.n_sequences  = uint32_t(actual_seqs);
            ch.seq_len      = uint32_t(seq_len);
            ch.vocab_size   = uint32_t(tok.vocab_size());
            ch.crc32c       = crc;
            ch.total_tokens = uint64_t(cal_tokens.size());
            std::ofstream ofs(cal_path, std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(&ch), sizeof(ch));
            ofs.write(reinterpret_cast<const char*>(cal_tokens.data()), std::streamsize(cal_tokens.size() * sizeof(int32_t)));
        }

        // 6. Write artifacts/cal_disjoint_receipt.json
        const std::string rec_path = root + "/artifacts/cal_disjoint_receipt.json";
        {
            std::ofstream ofs(rec_path, std::ios::trunc);
            ofs << "{\n"
                << "  \"schema\": \"ernie55_cal_disjoint_receipt_v1\",\n"
                << "  \"cal_corpus_bin\": \"" << cal_path << "\",\n"
                << "  \"gate2_probes_bin\": \"" << probe_path << "\",\n"
                << "  \"n_cal_tokens\": " << cal_tokens.size() << ",\n"
                << "  \"n_cal_sequences\": " << actual_seqs << ",\n"
                << "  \"seq_len\": " << seq_len << ",\n"
                << "  \"domain_tokens\": {\n";
            for (size_t i = 0; i < pools.size(); ++i) {
                ofs << "    \"" << pools[i].name << "\": " << pools[i].tokens_kept << (i + 1 < pools.size() ? "," : "") << "\n";
            }
            ofs << "  },\n"
                << "  \"gate2_heldout_probes\": 64,\n"
                << "  \"gate2_probe_len\": " << probe_len << ",\n"
                << "  \"gate2_eval_positions\": " << (64 * probe_len) << ",\n"
                << "  \"eval_8grams_indexed\": " << eval_8grams_checked << ",\n"
                << "  \"eval_8grams_unique\": " << banned_8grams.size() << ",\n"
                << "  \"chunks_rejected_for_overlap\": " << chunks_rejected << ",\n"
                << "  \"overlap_8grams\": " << overlap_8grams << ",\n"
                << "  \"cal_crc32c\": \"0x" << std::hex << crc << std::dec << "\",\n"
                << "  \"status\": \"PASS\"\n"
                << "}\n";
        }

        std::printf("[CAL-BUILD] PASS! Built %zu tokens (%d x %d) | overlap_8grams=%zu (checked %zu eval 8-grams) | crc=0x%08x\n",
                    cal_tokens.size(), actual_seqs, seq_len, overlap_8grams, eval_8grams_checked, crc);
        for (const auto& p : pools) {
            std::printf("  domain %-16s : %zu tokens\n", p.name.c_str(), p.tokens_kept);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[CAL-BUILD] FATAL: %s\n", ex.what());
        return 1;
    }
}
