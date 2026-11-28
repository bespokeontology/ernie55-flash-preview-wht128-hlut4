// SPDX-License-Identifier: Apache-2.0
// Blackhole P150A ERNIE-5.5 Expert-Compact Row Gather + Tilizer (RISCV_0 / NOC_1 across 64 cores)
// H_lat = 2048 -> 64 tiles = 64 cores x 1 tile/core (32 cols = 64 B BF16/row).
// x_rm is [1, 1, max_rows, 2048] BFLOAT16 TILE -> 2 exact 2048 B pages per row!

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_rm_addr      = get_arg_val<uint32_t>(0);
    const uint32_t slot_meta_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_tilized_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot_start     = get_arg_val<uint32_t>(3);
    const uint32_t core_i         = get_arg_val<uint32_t>(4);  // 0..63
    const uint32_t batch_cnt      = get_arg_val<uint32_t>(5);  // 1..32

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset(), m_args.next_common_runtime_args_offset()>();

    const auto x_rm_acc = TensorAccessor(x_args, x_rm_addr, 2048);
    const auto meta_acc = TensorAccessor(m_args, slot_meta_addr, 256);
    const auto out_acc  = TensorAccessor(o_args, x_tilized_addr, 2048);

    cb_reserve_back(0, 1);
    const uint32_t base_l1   = get_write_ptr(0);
    // Wave-13 Cut W13.4a: 256 * 128 B = 32,768 B metadata arena for kBatchSlots = 256
    // (total CB 0 = 32,768 + 2,048 gather + 2 * 2,048 tiles = 38,912 B < 200 KiB guard).
    const uint32_t gather_l1 = base_l1 + 32768u;
    const uint32_t tile0_l1  = base_l1 + 34816u;
    const uint32_t tile1_l1  = base_l1 + 36864u;

    const uint32_t page_grp = core_i >> 5;
    const uint32_t byte_off = (core_i & 31u) * 64u;

    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        noc_async_read(meta_acc.get_noc_addr(slot_start + b, 0), base_l1 + b * 128u, 128u);
    }
    if (batch_cnt > 0u) noc_async_read_barrier();

    uint32_t issued_wr = 0u;
    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint32_t meta_b = base_l1 + b * 128u;
        const volatile uint32_t* meta_u32 = reinterpret_cast<const volatile uint32_t*>(meta_b);
        uint32_t cnt = meta_u32[0];
        if (cnt > 32u) cnt = 32u;
        if (cnt == 0u) continue;
        const volatile uint16_t* row_idx = reinterpret_cast<const volatile uint16_t*>(meta_b + 8u);

        for (uint32_t r = 0u; r < cnt; ++r) {
            const uint32_t src_row = row_idx[r];
            noc_async_read(x_rm_acc.get_noc_addr(src_row * 2u + page_grp, byte_off), gather_l1 + r * 64u, 64u);
        }

        const uint32_t cur_tile_l1 = (issued_wr & 1u) ? tile1_l1 : tile0_l1;
        if (issued_wr >= 2u) noc_async_writes_flushed();
        uint64_t* dst0 = reinterpret_cast<uint64_t*>(cur_tile_l1);
        const uint32_t fr_top = (cnt < 16u) ? cnt : 16u;
        const uint32_t fr_bot = (cnt > 16u) ? (cnt - 16u) : 0u;
        // Zero unused tile rows directly in dst0 while the gather NOC reads are in flight.
        for (uint32_t fr = fr_top; fr < 16u; ++fr) {
            const uint32_t d_off = fr * 4u;
            #pragma GCC unroll 4
            for (uint32_t c4 = 0u; c4 < 4u; ++c4) {
                dst0[0u  + d_off + c4] = 0ULL;
                dst0[64u + d_off + c4] = 0ULL;
            }
        }
        if (cnt <= 16u) {
            for (uint32_t w = 128u; w < 256u; ++w) dst0[w] = 0ULL;
        } else {
            for (uint32_t fr = fr_bot; fr < 16u; ++fr) {
                const uint32_t d_off = fr * 4u;
                #pragma GCC unroll 4
                for (uint32_t c4 = 0u; c4 < 4u; ++c4) {
                    dst0[128u + d_off + c4] = 0ULL;
                    dst0[192u + d_off + c4] = 0ULL;
                }
            }
        }
        if (cnt > 0u) noc_async_read_barrier();

        const uint64_t* src_u64 = reinterpret_cast<const uint64_t*>(gather_l1);
        // Wave-13 Cut W13.4a: copy strictly live rows [0..fr_top) and [16..16+fr_bot).
        for (uint32_t fr = 0u; fr < fr_top; ++fr) {
            const uint64_t* s_top = src_u64 + fr * 8u;
            const uint32_t d_off  = fr * 4u;
            #pragma GCC unroll 4
            for (uint32_t c4 = 0u; c4 < 4u; ++c4) {
                dst0[0u  + d_off + c4] = s_top[0u + c4];
                dst0[64u + d_off + c4] = s_top[4u + c4];
            }
        }
        for (uint32_t fr = 0u; fr < fr_bot; ++fr) {
            const uint64_t* s_bot = src_u64 + (16u + fr) * 8u;
            const uint32_t d_off  = fr * 4u;
            #pragma GCC unroll 4
            for (uint32_t c4 = 0u; c4 < 4u; ++c4) {
                dst0[128u + d_off + c4] = s_bot[0u + c4];
                dst0[192u + d_off + c4] = s_bot[4u + c4];
            }
        }

        noc_async_write_tile(b * 64u + core_i, out_acc, cur_tile_l1);
        ++issued_wr;
    }
    noc_async_write_barrier();
}
