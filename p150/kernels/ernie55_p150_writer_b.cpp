// SPDX-License-Identifier: Apache-2.0
// Blackhole P150A ERNIE-5.5 Uniform WHT128-HLUT4 Weight Feeder (L1 Hot Pin + GDDR6) & Partial Writer (RISCV_0 / NOC_1)
// 100% QUANT_MODE == 1: 8,448 B/superblock (2 x 4096 B nibble tiles + 256 B FP16 scales).
// Parameterized sb_per_proj (128 for TP2 2xP150A, 256 for EP3 3xP150A).

#include "api/dataflow/dataflow_api.h"

static inline __attribute__((always_inline)) uint32_t fp16_to_f32_bits_fast(uint32_t h) {
    const uint32_t mag = h & 0x7FFFu;
    const uint32_t sgn = (h & 0x8000u) << 16;
    if (__builtin_expect((mag & 0x7C00u) != 0u && (mag & 0x7C00u) != 0x7C00u, 1)) {
        return sgn | ((mag + 0x1C000u) << 13);
    }
    if (mag == 0u) return sgn;
    if ((mag & 0x7C00u) == 0u) {
        uint32_t e = 113u;
        uint32_t m = mag;
        while ((m & 0x0400u) == 0u) { m <<= 1; --e; }
        return sgn | (e << 23) | ((m & 0x03FFu) << 13);
    }
    return sgn | 0x7F800000u | ((mag & 0x03FFu) << 13);
}

static inline __attribute__((always_inline)) void expand_hlut_scales_tile(
    const volatile uint16_t* raw_f16_base,
    uint64_t* dst_u64)
{
    for (uint32_t kl = 0u; kl < 2u; ++kl) {
        const uint32_t f_even_base = kl * 256u;
        const uint32_t f_odd_base  = kl * 256u + 128u;
        for (uint32_t nt_sub = 0u; nt_sub < 4u; ++nt_sub) {
            const volatile uint16_t* sc8 = raw_f16_base + (kl * 4u + nt_sub) * 8u;
            uint64_t* f0 = dst_u64 + f_even_base + (nt_sub * 4u) * 8u;
            uint64_t* f1 = dst_u64 + f_odd_base  + (nt_sub * 4u) * 8u;
            #pragma GCC unroll 4
            for (uint32_t sg = 0u; sg < 4u; ++sg) {
                const uint32_t f32 = fp16_to_f32_bits_fast(sc8[sg]);
                const uint64_t pair = uint64_t(f32) | (uint64_t(f32) << 32);
                const uint32_t c0 = sg * 2u;
                f0[0u  + c0] = pair; f0[0u  + c0 + 1u] = pair;
                f0[8u  + c0] = pair; f0[8u  + c0 + 1u] = pair;
                f0[16u + c0] = pair; f0[16u + c0 + 1u] = pair;
                f0[24u + c0] = pair; f0[24u + c0 + 1u] = pair;
            }
            #pragma GCC unroll 4
            for (uint32_t sg = 0u; sg < 4u; ++sg) {
                const uint32_t f32 = fp16_to_f32_bits_fast(sc8[4u + sg]);
                const uint64_t pair = uint64_t(f32) | (uint64_t(f32) << 32);
                const uint32_t c0 = sg * 2u;
                f1[0u  + c0] = pair; f1[0u  + c0 + 1u] = pair;
                f1[8u  + c0] = pair; f1[8u  + c0 + 1u] = pair;
                f1[16u + c0] = pair; f1[16u + c0 + 1u] = pair;
                f1[24u + c0] = pair; f1[24u + c0 + 1u] = pair;
            }
        }
    }
}

void kernel_main() {
    constexpr uint32_t kPageBytes  = 8448u;
    constexpr uint32_t kSbPerProj  = get_compile_time_arg_val(0);  // 128 (TP2) or 256 (EP3)

    const uint32_t slot_ids_addr   = get_arg_val<uint32_t>(0);
    const uint32_t partials0_addr  = get_arg_val<uint32_t>(1);
    const uint32_t slot_start      = get_arg_val<uint32_t>(2);
    const uint32_t proj_idx        = get_arg_val<uint32_t>(3);  // 3=fused gate+up, 2=down
    const uint32_t nb4             = get_arg_val<uint32_t>(4);  // 0..7 (TP2 gu) or 0..15 (down / EP3)
    const uint32_t kt4_total       = get_arg_val<uint32_t>(5);  // 16 (gu / EP3) or 8 (TP2 down)
    const uint32_t kb4_start       = get_arg_val<uint32_t>(6);  // ks * num_sb
    const uint32_t num_sb          = get_arg_val<uint32_t>(7);  // 2 (TP2) or 4 (EP3)
    const uint32_t ks              = get_arg_val<uint32_t>(8);  // 0..k_splits-1
    const uint32_t nt_total        = get_arg_val<uint32_t>(9);  // 32 (TP2 gu) or 64 (down / EP3)
    const uint32_t m_tiles_ws      = get_arg_val<uint32_t>(10); // 16
    const uint32_t slot_ids_bytes  = get_arg_val<uint32_t>(11); // 1344
    const uint32_t batch_cnt       = get_arg_val<uint32_t>(12); // 1..16
    const uint32_t partials1_addr  = get_arg_val<uint32_t>(13);

    constexpr auto p_args = TensorAccessorArgs<1>();
    constexpr auto i_args = TensorAccessorArgs<p_args.next_compile_time_args_offset(), p_args.next_common_runtime_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset(), i_args.next_common_runtime_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<o_args.next_compile_time_args_offset(), o_args.next_common_runtime_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<h_args.next_compile_time_args_offset(), h_args.next_common_runtime_args_offset()>();

    const bool fused_gu      = (proj_idx == 3u);
    const uint32_t num_projs = fused_gu ? 2u : 1u;
    const uint32_t base_proj = fused_gu ? 0u : proj_idx;

    const auto ids_acc  = TensorAccessor(i_args, slot_ids_addr, slot_ids_bytes);
    const auto out_acc0 = TensorAccessor(o_args, partials0_addr, 4096);
    const auto out_acc1 = TensorAccessor(o_args, partials1_addr, 4096);

    cb_reserve_back(6, 1);
    const uint32_t scratch_ptr = get_write_ptr(6);
    noc_async_read(ids_acc.get_noc_addr(0, 0), scratch_ptr, slot_ids_bytes);
    noc_async_read_barrier();
    cb_push_back(6, 1);

    uint16_t local_eids[256];  // Wave-13 Cut W13.4a: matches kBatchSlots = 256
    const volatile uint16_t* slot_u16 = reinterpret_cast<const volatile uint16_t*>(scratch_ptr);
    bool any_valid = false;
    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint16_t e = slot_u16[slot_start + b];
        local_eids[b] = e;
        if (e < 0xFFFEu) any_valid = true;
    }
    if (!any_valid) return;

    const volatile uint32_t* ldesc = reinterpret_cast<const volatile uint32_t*>(scratch_ptr + (slot_ids_bytes - 64u));
    const uint32_t lut_addr  = ldesc[8];
    const uint32_t layer_idx = ldesc[9];

    const auto pages_acc0 = TensorAccessor(p_args, ldesc[base_proj], kPageBytes);
    const auto pages_acc1 = TensorAccessor(p_args, ldesc[fused_gu ? 1u : base_proj], kPageBytes);
    const auto hot_acc0   = TensorAccessor(h_args, ldesc[5u + base_proj], kPageBytes);
    const auto hot_acc1   = TensorAccessor(h_args, ldesc[5u + (fused_gu ? 1u : base_proj)], kPageBytes);
    const auto lut_acc    = TensorAccessor(l_args, lut_addr, 512);

    const uint32_t lut_l1    = scratch_ptr + 1408u;
    const uint32_t scales_l1 = scratch_ptr + 1920u;
    noc_async_read(lut_acc.get_noc_addr(layer_idx, 0), lut_l1, 512u);
    noc_async_read_barrier();
    const volatile uint16_t* hot_lut = reinterpret_cast<const volatile uint16_t*>(lut_l1);

    const uint32_t local_sb_off  = nb4 * kt4_total + kb4_start;
    const uint32_t col_tile_base = nb4 * 4u;

    auto drain_out_step = [&](uint32_t b_done, uint32_t p_done) __attribute__((always_inline)) {
        cb_wait_front(16, 4);
        const uint32_t rd_l1 = get_read_ptr(16);
        const uint32_t out_tile_base = (ks * m_tiles_ws + b_done) * nt_total + col_tile_base;
        if (p_done == 0u) {
            for (uint32_t n = 0u; n < 4u; ++n) {
                noc_async_write(rd_l1 + n * 4096u, out_acc0.get_noc_addr(out_tile_base + n, 0), 4096u);
            }
        } else {
            for (uint32_t n = 0u; n < 4u; ++n) {
                noc_async_write(rd_l1 + n * 4096u, out_acc1.get_noc_addr(out_tile_base + n, 0), 4096u);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(16, 4);
    };

    uint32_t prev_b = 0u, prev_p = 0u;
    bool has_prev = false;

    uint32_t b0 = 0u;
    while (b0 < batch_cnt) {
        const uint32_t expert_slot = local_eids[b0];
        if (expert_slot >= 0xFFFEu) { ++b0; continue; }
        uint32_t run_len = 1u;
        while (b0 + run_len < batch_cnt && local_eids[b0 + run_len] == expert_slot) ++run_len;

        const uint16_t hs = hot_lut[expert_slot];
        const int32_t hot_slot = (hs != 0xFFFFu) ? static_cast<int32_t>(hs) : -1;
        const uint32_t dram_pid_base = expert_slot * kSbPerProj + local_sb_off;
        const uint32_t hot_pid_base  = (hot_slot >= 0) ? (uint32_t(hot_slot) * kSbPerProj + local_sb_off) : 0u;

        for (uint32_t p = 0u; p < num_projs; ++p) {
            cb_reserve_back(4, 2u * num_sb);
            const uint32_t cb4_wr = get_write_ptr(4);
            for (uint32_t sb = 0u; sb < num_sb; ++sb) {
                const uint32_t dram_pid = dram_pid_base + sb;
                const uint32_t hot_pid  = hot_pid_base + sb;
                auto read_slice = [&](uint32_t off, uint32_t dst_l1, uint32_t nbytes) __attribute__((always_inline)) {
                    if (p == 0u) {
                        if (hot_slot >= 0) noc_async_read(hot_acc0.get_noc_addr(hot_pid, off), dst_l1, nbytes);
                        else               noc_async_read(pages_acc0.get_noc_addr(dram_pid, off), dst_l1, nbytes);
                    } else {
                        if (hot_slot >= 0) noc_async_read(hot_acc1.get_noc_addr(hot_pid, off), dst_l1, nbytes);
                        else               noc_async_read(pages_acc1.get_noc_addr(dram_pid, off), dst_l1, nbytes);
                    }
                };
                // Wave-13 Cut W13.4a: the two 4096 B nibble halves are contiguous in
                // GDDR6, so a single 8192 B NOC read replaces two 4096 B reads and halves
                // the per-superblock read-issue overhead with identical bytes.
                read_slice(0u,    cb4_wr + sb * 8192u,          8192u);
                read_slice(8192u, scales_l1 + sb * 256u,       256u);
            }
            noc_async_read_barrier();

            for (uint32_t sb = 0u; sb < num_sb; ++sb) {
                cb_reserve_back(5, 2);
                const uint32_t cb5_wr = get_write_ptr(5);
                const volatile uint16_t* sc16 = reinterpret_cast<const volatile uint16_t*>(scales_l1 + sb * 256u);
                expand_hlut_scales_tile(sc16 + 0u,  reinterpret_cast<uint64_t*>(cb5_wr + 0u));
                expand_hlut_scales_tile(sc16 + 64u, reinterpret_cast<uint64_t*>(cb5_wr + 4096u));
                cb_push_back(4, 2);
                cb_push_back(5, 2);
            }

            for (uint32_t r = 0u; r < run_len; ++r) {
                if (has_prev) drain_out_step(prev_b, prev_p);
                prev_b = b0 + r;
                prev_p = p;
                has_prev = true;
            }
        }
        b0 += run_len;
    }

    if (has_prev) drain_out_step(prev_b, prev_p);
    noc_async_write_barrier();
}
