// SPDX-License-Identifier: Apache-2.0
// Pure-C++17 Native Live Progress & Token-Work Telemetry for :3081 DeepSeek Harness
// Emits provider-independent `x_flash_progress` v1 payloads matching pi-ai openai-completions.js:319-354
// so :3081 displays live prefill progress, cached vs fresh tokens, and real-time decode tok/s counters.
#pragma once

#include "ernie55_json.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
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
    double   window_ms = 100.0;

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

    void configure_input(size_t total, size_t cached, int64_t req_start_ns) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        input_total = uint64_t(total);
        input_cached = uint64_t(cached);
        work_epoch_ns = (req_start_ns > 0 && req_start_ns < t) ? req_start_ns : (t - 1000);
        if (cached > 0) {
            add_work_locked(uint64_t(cached), 0, 0, work_epoch_ns, t);
        }
        prefill_begin_ns = t;
        input_ready.store(true, std::memory_order_release);
    }

    void observe_prefill(size_t chunk_done_fresh, double chunk_ms) {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        if (chunk_done_fresh <= input_fresh) return;
        const uint64_t delta = uint64_t(chunk_done_fresh) - input_fresh;
        const int64_t b = (prefill_begin_ns > 0 && prefill_begin_ns < t) ? prefill_begin_ns : (t - 1000);
        add_work_locked(0, delta, 0, b, t);
        prefill_begin_ns = t;
        input_fresh = uint64_t(chunk_done_fresh);
        prefill_tokens = input_fresh;
        prefill_elapsed_ms += (chunk_ms > 0.0) ? chunk_ms : (double(t - b) * 1e-6);
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
                prefill_elapsed_ms += double(t - b) * 1e-6;
            }
            prefill_begin_ns = 0;
            first_gen_ns.store(t, std::memory_order_relaxed);
            last_gen_interval_ns = t;
        }
        const uint64_t delta = uint64_t(total_generated) - prior;
        pending_gen_tokens += delta;
        if (t - last_gen_interval_ns >= 80000000LL) { // flush interval every >= 80 ms (aligned to DSpark bursts)
            add_work_locked(0, 0, pending_gen_tokens, last_gen_interval_ns, t);
            window_tokens = pending_gen_tokens;
            window_ms = double(t - last_gen_interval_ns) * 1e-6;
            pending_gen_tokens = 0;
            last_gen_interval_ns = t;
        }
        gen_count.store(uint64_t(total_generated), std::memory_order_release);
    }

    void finish() {
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t t = now_ns();
        if (pending_gen_tokens > 0 && last_gen_interval_ns > 0) {
            const int64_t e = (t > last_gen_interval_ns) ? t : (last_gen_interval_ns + 1);
            add_work_locked(0, 0, pending_gen_tokens, last_gen_interval_ns, e);
            window_tokens = pending_gen_tokens;
            window_ms = double(e - last_gen_interval_ns) * 1e-6;
            pending_gen_tokens = 0;
            last_gen_interval_ns = e;
        }
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

        // Flush pending decode interval if >= 100 ms elapsed
        if (!is_final && pending_gen_tokens > 0 && last_gen_interval_ns > 0 && (t - last_gen_interval_ns) >= 100000000LL) {
            add_work_locked(0, 0, pending_gen_tokens, last_gen_interval_ns, t);
            window_tokens = pending_gen_tokens;
            window_ms = double(t - last_gen_interval_ns) * 1e-6;
            pending_gen_tokens = 0;
            last_gen_interval_ns = t;
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
            new_toks += iv.cached + iv.fresh + iv.generated;
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
