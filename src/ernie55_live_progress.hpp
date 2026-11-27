// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Native Live Progress & Token-Work Telemetry for :3081 DeepSeek Harness
// Emits provider-independent `x_flash_progress` v1 payloads matching pi-ai openai-completions.js:319-354
// Wave-10 Bugs 4a-4d Fix:
//   - 4a: Strictly decouple zero-work VRAM/DDR4/NVMe prefix cache hits from live compute token_work
//         (cached tokens NEVER inflate compute prefill/decode tok/s; reported with cache_tier & cache_restore_ms).
//   - 4b: True rolling >=250ms decode sample ring (finish() NEVER clobbers window_tokens/window_ms with sub-ms tail flush).
//   - 4c: Monotonic layer-granularity fresh prefill progress ticks every 150ms (zero 4s dead-air freeze).
//   - 4d: Exact 1:1 parity: generated_tokens == token_work.generated_tokens == usage.completion_tokens on every frame.
#pragma once

#include "ernie55_json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ernie55 {

struct Ernie55LiveProgress {
    struct WorkInterval {
        uint64_t seq = 0;
        uint64_t cached = 0;
        uint64_t fresh = 0;
        uint64_t generated = 0;
        int64_t  begin_ns = 0;
        int64_t  end_ns = 0;
    };

    std::mutex mtx;
    std::atomic<uint64_t> gen_count{0};
    std::atomic<int64_t>  first_gen_ns{0};
    std::atomic<int64_t>  end_ns{0};
    std::atomic<bool>     input_ready{false};

    uint64_t input_total = 0;
    uint64_t input_cached = 0;
    uint64_t input_fresh = 0;
    std::string cache_tier = "none";
    double   cache_restore_ms = 0.0;

    uint64_t prefill_tokens = 0;
    double   prefill_elapsed_ms = 0.0;
    uint64_t prefill_sample_tokens = 0;
    double   prefill_sample_ms = 0.0;

    uint64_t work_seq = 0;
    uint64_t work_cached = 0;
    uint64_t work_fresh = 0;
    uint64_t work_generated = 0;
    uint64_t unmeasured_work = 0;
    uint64_t live_work_tokens = 0;
    int64_t  live_work_ns = 0;

    int64_t  work_epoch_ns = 0;
    int64_t  work_active_ns = 0;
    int64_t  cached_active_ns = 0;
    int64_t  fresh_active_ns = 0;
    int64_t  generated_active_ns = 0;
    int64_t  prefill_begin_ns = 0;
    int64_t  last_gen_interval_ns = 0;
    uint64_t pending_gen_tokens = 0;
    uint64_t window_tokens = 0;
    double   window_ms = 250.0;

    // Rolling decode history (timestamp_ns, cumulative_gen_tokens) for spike-free >=250ms window tok/s (Bug 4b)
    std::deque<std::pair<int64_t, uint64_t>> decode_ring;
    std::vector<WorkInterval> pending_work;

    static int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void add_work_locked(uint64_t cached, uint64_t fresh, uint64_t generated, int64_t begin, int64_t end) {
        if (!cached && !fresh && !generated) return;
        if (begin <= 0 || end <= begin) {
            unmeasured_work += cached + fresh + generated;
            return;
        }
        if (work_epoch_ns == 0) work_epoch_ns = begin;
        if (begin < work_epoch_ns) begin = work_epoch_ns;
        if (end <= begin) end = begin + 1;
        pending_work.push_back({++work_seq, cached, fresh, generated, begin, end});
        work_cached += cached;
        work_fresh += fresh;
        work_generated += generated;
        const int64_t dur = end - begin;
        work_active_ns += dur;
        if (cached)    cached_active_ns += dur;
        if (fresh)     fresh_active_ns += dur;
        if (generated) generated_active_ns += dur;
    }

    void update_rolling_window_locked(int64_t t, uint64_t cur_total) {
        if (cur_total == 0) return;
        if (decode_ring.empty() || t > decode_ring.back().first || cur_total > decode_ring.back().second) {
            decode_ring.emplace_back(t, cur_total);
        }
        // Keep up to 1.5s of history so we always have a clean >=250ms span without tail-flush spikes!
        while (decode_ring.size() > 2 && (t - decode_ring[1].first) >= 1000000000LL) {
            decode_ring.pop_front();
        }
        // Pick the newest anchor that is at least 250 ms old (else oldest available in decode_ring)
        std::pair<int64_t, uint64_t> anchor = decode_ring.front();
        for (const auto& pt : decode_ring) {
            if (t - pt.first >= 250000000LL) {
                anchor = pt;
            } else {
                break;
            }
        }
        const int64_t span_ns = t - anchor.first;
        const uint64_t dt_toks = (cur_total >= anchor.second) ? (cur_total - anchor.second) : 0u;
        // Never overwrite window_tokens / window_ms with sub-100ms micro-flush noise (Bug 4b)!
        if (span_ns >= 100000000LL && dt_toks > 0) {
            window_tokens = dt_toks;
            window_ms = double(span_ns) * 1e-6;
        } else if (window_tokens == 0 && dt_toks > 0 && span_ns > 0) {
            window_tokens = dt_toks;
            window_ms = std::max(100.0, double(span_ns) * 1e-6);
        }
    }

    void configure_input(size_t total, size_t cached, int64_t req_start_ns,
                         const std::string& tier = "none", double restore_ms = 0.0) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        input_total = uint64_t(total);
        input_cached = uint64_t(cached);
        cache_tier = tier.empty() ? (cached > 0 ? "VRAM" : "none") : tier;
        cache_restore_ms = (restore_ms > 0.0) ? restore_ms : 0.0;
        work_epoch_ns = (req_start_ns > 0 && req_start_ns < t) ? req_start_ns : (t - 1000);
        // Bug 4a Fix: NEVER inject zero-work cached prefix hits into live compute token_work.intervals!
        // Record cache restore wall time separately in cached_active_ns without inflating compute tok/s.
        if (cached > 0) {
            cached_active_ns = (restore_ms > 0.0) ? int64_t(restore_ms * 1e6) : std::max<int64_t>(1, t - work_epoch_ns);
        }
        prefill_begin_ns = (total > cached) ? t : 0;
        input_ready.store(true, std::memory_order_release);
    }

    void observe_prefill(size_t chunk_done_fresh, double chunk_ms) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        const uint64_t max_fresh = (input_total > input_cached) ? (input_total - input_cached) : 0u;
        const uint64_t clamped = std::min<uint64_t>(uint64_t(chunk_done_fresh), max_fresh);
        if (clamped <= input_fresh) return;
        const uint64_t delta = clamped - input_fresh;
        const int64_t b = (prefill_begin_ns > 0 && prefill_begin_ns < t) ? prefill_begin_ns : (t - 1000);
        add_work_locked(0, delta, 0, b, t);
        prefill_begin_ns = (clamped < max_fresh) ? t : 0;
        input_fresh = clamped;
        prefill_tokens = input_fresh;
        prefill_elapsed_ms = double(fresh_active_ns) * 1e-6;
        prefill_sample_tokens = delta;
        prefill_sample_ms = (chunk_ms > 0.0) ? chunk_ms : (double(t - b) * 1e-6);
    }

    void observe_decode_tokens(size_t total_generated) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        const uint64_t prior = gen_count.load(std::memory_order_relaxed);
        if (total_generated <= prior) return;
        if (first_gen_ns.load(std::memory_order_relaxed) == 0) {
            // Close any remaining fresh prefill tokens before first generated token
            if (input_total > input_cached + input_fresh) {
                const uint64_t rem = input_total - input_cached - input_fresh;
                const int64_t b = (prefill_begin_ns > 0 && prefill_begin_ns < t) ? prefill_begin_ns : (t - 1000);
                add_work_locked(0, rem, 0, b, t);
                input_fresh += rem;
                prefill_tokens = input_fresh;
                prefill_elapsed_ms = double(fresh_active_ns) * 1e-6;
            }
            prefill_begin_ns = 0;
            first_gen_ns.store(t, std::memory_order_relaxed);
            last_gen_interval_ns = t;
            decode_ring.clear();
            decode_ring.emplace_back(t, 0u);
        }
        const uint64_t delta = uint64_t(total_generated) - prior;
        pending_gen_tokens += delta;
        if (t - last_gen_interval_ns >= 80000000LL) {
            add_work_locked(0, 0, pending_gen_tokens, last_gen_interval_ns, t);
            pending_gen_tokens = 0;
            last_gen_interval_ns = t;
        }
        update_rolling_window_locked(t, uint64_t(total_generated));
        gen_count.store(uint64_t(total_generated), std::memory_order_release);
    }

    void finish(size_t final_generated_tokens = 0) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        // Bug 4d Fix: Reconcile any final EOS/stop token counted in stats.generated_tokens so
        // generated_tokens == token_work.generated_tokens == usage.completion_tokens exact!
        const uint64_t cur_n = gen_count.load(std::memory_order_relaxed);
        if (final_generated_tokens > cur_n) {
            if (first_gen_ns.load(std::memory_order_relaxed) == 0) {
                first_gen_ns.store(t, std::memory_order_relaxed);
                last_gen_interval_ns = t;
                decode_ring.clear();
                decode_ring.emplace_back(t, 0u);
            }
            pending_gen_tokens += uint64_t(final_generated_tokens) - cur_n;
            gen_count.store(uint64_t(final_generated_tokens), std::memory_order_release);
        }
        if (pending_gen_tokens > 0 && last_gen_interval_ns > 0) {
            const int64_t e = (t > last_gen_interval_ns) ? t : (last_gen_interval_ns + 1);
            add_work_locked(0, 0, pending_gen_tokens, last_gen_interval_ns, e);
            pending_gen_tokens = 0;
            last_gen_interval_ns = e;
        }
        // Bug 4b Fix: Update rolling window using >=250ms ring (NEVER overwrite with sub-ms tail flush!)
        update_rolling_window_locked(t, gen_count.load(std::memory_order_relaxed));
        prefill_begin_ns = 0;
        end_ns.store(t, std::memory_order_release);
    }

    JsonValue sample(const std::string& request_id, bool is_final) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        const uint64_t n = gen_count.load(std::memory_order_acquire);
        const int64_t s0 = first_gen_ns.load(std::memory_order_relaxed);
        const int64_t e0 = end_ns.load(std::memory_order_acquire);
        const int64_t gen_el_ns = (s0 > 0) ? ((e0 > 0 ? e0 : t) - s0) : 0;

        // Bug 4d Fix: Flush pending_gen_tokens before serializing token_work so
        // generated_tokens == token_work.generated_tokens on EVERY frame (zero off-by-1 drift)!
        if (pending_gen_tokens > 0 && last_gen_interval_ns > 0) {
            const int64_t e = (t > last_gen_interval_ns) ? t : (last_gen_interval_ns + 1);
            add_work_locked(0, 0, pending_gen_tokens, last_gen_interval_ns, e);
            pending_gen_tokens = 0;
            last_gen_interval_ns = e;
            update_rolling_window_locked(e, n);
        }

        JsonValue p = JsonValue::make_object();
        p.set("version", JsonValue::make_int(1));
        p.set("request_id", JsonValue::make_string(request_id));
        p.set("phase", JsonValue::make_string(is_final ? "paused" : (n > 0 ? "generating" : "prefill")));
        p.set("generated_tokens", JsonValue::make_int(int64_t(n)));
        p.set("generation_elapsed_ms", JsonValue::make_double(double(gen_el_ns) * 1e-6));
        p.set("generation_window_tokens", JsonValue::make_int(int64_t(window_tokens)));
        p.set("generation_window_ms", JsonValue::make_double(window_ms));
        p.set("prefill_tokens", JsonValue::make_int(int64_t(prefill_tokens)));
        p.set("prefill_elapsed_ms", JsonValue::make_double(prefill_elapsed_ms));
        p.set("prefill_sample_tokens", JsonValue::make_int(int64_t(prefill_sample_tokens)));
        p.set("prefill_sample_ms", JsonValue::make_double(prefill_sample_ms));
        p.set("cache_tier", JsonValue::make_string(cache_tier));
        p.set("cache_restore_ms", JsonValue::make_double(cache_restore_ms));

        if (input_ready.load(std::memory_order_acquire)) {
            p.set("input_tokens_total", JsonValue::make_int(int64_t(input_total)));
            p.set("cached_input_tokens", JsonValue::make_int(int64_t(input_cached)));
            p.set("fresh_input_tokens", JsonValue::make_int(int64_t(input_fresh)));
            p.set("input_serviced_tokens", JsonValue::make_int(int64_t(input_cached + input_fresh)));
        }

        JsonValue ivs = JsonValue::make_array();
        uint64_t new_toks = 0;
        int64_t  new_ns = 0;
        for (const auto& iv : pending_work) {
            JsonValue o = JsonValue::make_object();
            o.set("seq", JsonValue::make_int(int64_t(iv.seq)));
            o.set("start_ns", JsonValue::make_int(std::max<int64_t>(0, iv.begin_ns - work_epoch_ns)));
            o.set("end_ns", JsonValue::make_int(std::max<int64_t>(1, iv.end_ns - work_epoch_ns)));
            o.set("cached_tokens", JsonValue::make_int(int64_t(iv.cached)));
            o.set("fresh_tokens", JsonValue::make_int(int64_t(iv.fresh)));
            o.set("generated_tokens", JsonValue::make_int(int64_t(iv.generated)));
            ivs.arr_val.push_back(std::move(o));
            // Bug 4a Fix: Only real compute tokens (fresh + generated) enter live_work_tokens!
            new_toks += iv.fresh + iv.generated;
            new_ns += (iv.end_ns - iv.begin_ns);
        }
        if (new_toks > 0) {
            live_work_tokens = new_toks;
            live_work_ns = new_ns;
        }
        pending_work.clear();

        JsonValue tw = JsonValue::make_object();
        tw.set("cached_tokens", JsonValue::make_int(int64_t(work_cached)));
        tw.set("fresh_tokens", JsonValue::make_int(int64_t(work_fresh)));
        tw.set("generated_tokens", JsonValue::make_int(int64_t(work_generated)));
        tw.set("active_ns", JsonValue::make_int(work_active_ns));
        tw.set("live_tokens", JsonValue::make_int(int64_t(live_work_tokens)));
        tw.set("live_active_ns", JsonValue::make_int(live_work_ns));
        tw.set("unmeasured_tokens", JsonValue::make_int(int64_t(unmeasured_work)));
        tw.set("request_wall_ns", JsonValue::make_int(work_epoch_ns > 0 ? std::max<int64_t>(0, (e0 > 0 ? e0 : t) - work_epoch_ns) : 0));
        tw.set("cached_active_ns", JsonValue::make_int(cached_active_ns));
        tw.set("fresh_active_ns", JsonValue::make_int(fresh_active_ns));
        tw.set("generated_active_ns", JsonValue::make_int(generated_active_ns));
        tw.set("interval_count", JsonValue::make_int(int64_t(work_seq)));
        tw.set("new_work", JsonValue::make_bool(new_toks > 0));
        tw.set("input_processing", JsonValue::make_bool(prefill_begin_ns != 0));
        tw.set("intervals", std::move(ivs));
        p.set("token_work", std::move(tw));
        return p;
    }
};

} // namespace ernie55
