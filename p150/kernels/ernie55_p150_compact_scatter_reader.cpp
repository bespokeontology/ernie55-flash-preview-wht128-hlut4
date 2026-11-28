// SPDX-License-Identifier: Apache-2.0
// Blackhole P150A ERNIE-5.5 Expert-Compact K-Split + Scatter-Accumulate Reader (RISCV_1 / NOC_0 across 64 cores)
// y_accum_f32 is [1, 1, max_rows, 2048] FLOAT32 TILE -> 2 exact 4096 B pages per row (nt_total = 64, 1 tile/core).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t partials_addr  = get_arg_val<uint32_t>(0);
    const uint32_t y_accum_addr   = get_arg_val<uint32_t>(1);
    const uint32_t slot_meta_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot_start     = get_arg_val<uint32_t>(3);
    const uint32_t k_splits       = get_arg_val<uint32_t>(4);  // 4
    const uint32_t core_i         = get_arg_val<uint32_t>(5);  // 0..63
    const uint32_t m_tiles_ws     = get_arg_val<uint32_t>(6);  // 32
    const uint32_t batch_cnt      = get_arg_val<uint32_t>(7);  // 1..32

    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto y_args = TensorAccessorArgs<p_args.next_compile_time_args_offset(), p_args.next_common_runtime_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<y_args.next_compile_time_args_offset(), y_args.next_common_runtime_args_offset()>();

    const auto part_acc = TensorAccessor(p_args, partials_addr, 4096);
    const auto y_acc    = TensorAccessor(y_args, y_accum_addr, 4096);
    const auto meta_acc = TensorAccessor(m_args, slot_meta_addr, 256);

    cb_reserve_back(5, 1);
    const uint32_t meta_base = get_write_ptr(5);
    // Wave-13 Cut W13.4a: 256 * 256 B = 65,536 B metadata arena for kBatchSlots = 256
    // (total CB 5 = 65,536 + 4,096 gather = 69,632 B).
    const uint32_t gather_l1 = meta_base + 65536u;
    const uint32_t page_grp  = core_i >> 5;
    const uint32_t byte_off  = (core_i & 31u) * 128u;
    constexpr uint32_t nt_total = 64u;
    uint32_t unacked = 0u;

    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        noc_async_read(meta_acc.get_noc_addr(slot_start + b, 0), meta_base + b * 256u, 256u);
    }
    if (batch_cnt > 0u) noc_async_read_barrier();

    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint32_t meta_l1  = meta_base + b * 256u;
        if (b > 0u) ++unacked;

        const volatile uint32_t* meta_u32 = reinterpret_cast<const volatile uint32_t*>(meta_l1);
        uint32_t cnt = meta_u32[0];
        if (cnt > 32u) cnt = 32u;
        // Wave-13 Cut W13.4a: bit 0 = wait_barrier (unchanged), bit 1 = is_first_wave.
        const uint32_t flags        = meta_u32[1];
        const uint32_t wait_barrier = flags & 1u;
        const bool     is_first_wave = (flags & 2u) != 0u;
        const volatile uint16_t* row_idx = reinterpret_cast<const volatile uint16_t*>(meta_l1 + 8u);
        const volatile uint32_t* w_bits  = meta_u32 + 32u;

        // Overlap partials DRAM read with writer drain when wait_barrier == 1, or fuse BOTH part_acc + y_acc
        // into ONE single noc_async_read_barrier() when wait_barrier == 0!
        cb_reserve_back(0, k_splits);
        const uint32_t p_wr_base = get_write_ptr(0);
        for (uint32_t ks = 0u; ks < k_splits; ++ks) {
            noc_async_read_tile((ks * m_tiles_ws + b) * nt_total + core_i, part_acc, p_wr_base + ks * 4096u);
        }

        if (wait_barrier != 0u && unacked > 0u) {
            cb_wait_front(7, unacked);
            cb_pop_front(7, unacked);
            unacked = 0u;
        }
        // Wave-13 Cut W13.4a: every wave-0 row is guaranteed zero in GDDR6 because the
        // scatter writer stores each wave-0 tile immediately after enqueue_zero_y_accum, so
        // the 32 random 128 B y_acc reads are skipped for the whole first wave.
        if (!is_first_wave) {
            for (uint32_t r = 0u; r < cnt; ++r) {
                const uint32_t src_row = row_idx[r];
                noc_async_read(y_acc.get_noc_addr(src_row * 2u + page_grp, byte_off), gather_l1 + r * 128u, 128u);
            }
        }

        // While part_acc + y_acc NOC DRAM reads are in flight, synthesize the 32x32 FP32 weight tile in L1 CB 1
        // from inline meta weights (0 DRAM reads!) and zero unused gather_l1 rows!
        cb_reserve_back(1, 1);
        uint64_t* w_tile = reinterpret_cast<uint64_t*>(get_write_ptr(1));
        for (uint32_t fr = 0u; fr < 16u; ++fr) {
            const uint32_t wb0 = (fr < cnt) ? w_bits[fr] : 0u;
            const uint64_t ww0 = uint64_t(wb0) | (uint64_t(wb0) << 32);
            const uint32_t d_off = fr * 8u;
            #pragma GCC unroll 8
            for (uint32_t c = 0u; c < 8u; ++c) {
                w_tile[0u   + d_off + c] = ww0;
                w_tile[128u + d_off + c] = ww0;
            }
        }
        if (cnt <= 16u) {
            for (uint32_t w = 256u; w < 512u; ++w) w_tile[w] = 0ULL;
        } else {
            for (uint32_t fr = 0u; fr < 16u; ++fr) {
                const uint32_t wb1 = ((16u + fr) < cnt) ? w_bits[16u + fr] : 0u;
                const uint64_t ww1 = uint64_t(wb1) | (uint64_t(wb1) << 32);
                const uint32_t d_off = fr * 8u;
                #pragma GCC unroll 8
                for (uint32_t c = 0u; c < 8u; ++c) {
                    w_tile[256u + d_off + c] = ww1;
                    w_tile[384u + d_off + c] = ww1;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(0, k_splits);
        cb_push_back(1, 1);

        cb_reserve_back(2, 1);
        uint64_t* dst_tile = reinterpret_cast<uint64_t*>(get_write_ptr(2));
        if (is_first_wave) {
            // Wave-13 Cut W13.4a: wave-0 y_acc rows are guaranteed zero in GDDR6, so the
            // RV32 tilize loop is replaced by a direct 4 KiB zero store into the combine tile.
            for (uint32_t w = 0u; w < 512u; ++w) dst_tile[w] = 0ULL;
            cb_push_back(2, 1);
            continue;
        }
        const uint64_t* src_u64 = reinterpret_cast<const uint64_t*>(gather_l1);
        // Wave-13 Cut W13.4a: copy strictly live rows [0..fr_top) and [16..16+fr_bot), and
        // zero unused rows directly in dst_tile (zero gather_l1 zeroing waste, 100% exact).
        const uint32_t fr_top = (cnt < 16u) ? cnt : 16u;
        const uint32_t fr_bot = (cnt > 16u) ? (cnt - 16u) : 0u;
        for (uint32_t fr = 0u; fr < fr_top; ++fr) {
            const uint64_t* s_top = src_u64 + fr * 16u;
            const uint32_t d_off  = fr * 8u;
            #pragma GCC unroll 8
            for (uint32_t c = 0u; c < 8u; ++c) {
                dst_tile[0u   + d_off + c] = s_top[0u + c];
                dst_tile[128u + d_off + c] = s_top[8u + c];
            }
        }
        for (uint32_t fr = fr_top; fr < 16u; ++fr) {
            const uint32_t d_off = fr * 8u;
            #pragma GCC unroll 8
            for (uint32_t c = 0u; c < 8u; ++c) {
                dst_tile[0u   + d_off + c] = 0ULL;
                dst_tile[128u + d_off + c] = 0ULL;
            }
        }
        if (cnt <= 16u) {
            for (uint32_t w = 256u; w < 512u; ++w) dst_tile[w] = 0ULL;
        } else {
            for (uint32_t fr = 0u; fr < fr_bot; ++fr) {
                const uint64_t* s_bot = src_u64 + (16u + fr) * 16u;
                const uint32_t d_off  = fr * 8u;
                #pragma GCC unroll 8
                for (uint32_t c = 0u; c < 8u; ++c) {
                    dst_tile[256u + d_off + c] = s_bot[0u + c];
                    dst_tile[384u + d_off + c] = s_bot[8u + c];
                }
            }
            for (uint32_t fr = fr_bot; fr < 16u; ++fr) {
                const uint32_t d_off = fr * 8u;
                #pragma GCC unroll 8
                for (uint32_t c = 0u; c < 8u; ++c) {
                    dst_tile[256u + d_off + c] = 0ULL;
                    dst_tile[384u + d_off + c] = 0ULL;
                }
            }
        }
        cb_push_back(2, 1);
    }
    if (unacked > 0u) {
        cb_wait_front(7, unacked);
        cb_pop_front(7, unacked);
    }
}
