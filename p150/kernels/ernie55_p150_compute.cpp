// SPDX-License-Identifier: Apache-2.0
// Blackhole P150A ERNIE-5.5 Pure WHT128-HLUT4 On-Chip SFPU Dequant + Tensix FPU Matmul Compute Kernel
// 100% QUANT_MODE == 1: Exact 16-level Lloyd-Max N(0,1) WHT128-HLUT4 (3 LREGs peak!) -> Float16_b CB 1 -> HiFi4 matmul_block.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

template <int KT_LOCAL, int NT_SUB>
sfpi_inline sfpi::vFloat decode_hlut4_lut(
    const sfpi::vUInt& w,
    const sfpi::vLut16ss& s01, const sfpi::vLut16ii& i01,
    const sfpi::vLut16ss& s23, const sfpi::vLut16ii& i23,
    const sfpi::vLut16ss& s45, const sfpi::vLut16ii& i45)
{
    constexpr int sh = 4 * (KT_LOCAL * 4 + NT_SUB);
    constexpr int sh_l = 29 - sh;
    constexpr int sh_sgn = 28 - sh;

    sfpi::vUInt mag = (sh_l > 0) ? ((w << sh_l) >> 29) : (w >> 29);
    sfpi::vFloat v = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(mag), sfpi::RoundMode::Nearest);
    v = v * 0.5f; // 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 (SFPMULI: zero-safe! addexp wraps 0->Inf)
    v = sfpi::lut<sfpi::LutMode::Fp16x6_HWM3>(v, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
    if constexpr (sh_sgn == 0) {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w));
    } else {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w << sh_sgn));
    }
}

template <int KT_LOCAL, int NT_SUB>
__attribute__((noinline)) void unpack_hlut4_1nt() {
    constexpr int sc_base = (KT_LOCAL == 0) ? 224 : 240;

    // 6-entry Fp16x6_HWM3 hardware SFPLUTFP32 table (cutoffs: 0.5, 1.0, 1.5, 2.0, 3.0):
    //   0.0 -> 0.12839780 (exact)
    //   0.5 -> 0.38809040 (exact)
    //   1.0 -> 0.65678227 (exact)
    //   1.5 -> 0.94235235 (exact)
    //   2.0, 2.5 -> 1.25623083, 1.61804903 (s4=0.72363640, i4=-0.19104197)
    //   3.0, 3.5 -> 2.06901598, 2.73259377 (s5=1.32715558, i5=-1.91245076)
    const sfpi::vLut16ss s01(0.0f,        0.0f);
    const sfpi::vLut16ii i01(0.12839780f, 0.38809040f);
    const sfpi::vLut16ss s23(0.0f,        0.0f);
    const sfpi::vLut16ii i23(0.65678227f, 0.94235235f);
    const sfpi::vLut16ss s45(0.72363640f, 1.32715558f);
    const sfpi::vLut16ii i45(-0.19104197f, -1.91245076f);

    #pragma GCC unroll 2
    for (int col_face = 0; col_face < 2; ++col_face) {
        #pragma GCC unroll 2
        for (int odd = 0; odd < 2; ++odd) {
            const int sc_idx   = sc_base + col_face * 8 + NT_SUB * 2 + odd;
            const int face_top = col_face * 8 + odd;
            const int face_bot = 16 + col_face * 8 + odd;

            #pragma GCC unroll 4
            for (int r_grp = 0; r_grp < 4; ++r_grp) {
                const int row_off = face_top + 2 * r_grp;
                const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
                const sfpi::vUInt w = sfpi::as<sfpi::vUInt>(w_i);
                sfpi::vFloat v = decode_hlut4_lut<KT_LOCAL, NT_SUB>(w, s01, i01, s23, i23, s45, i45);
                v = v * sfpi::vFloat(sfpi::dst_reg[sc_idx]);
                sfpi::dst_reg[NT_SUB * 32 + row_off] = v;
            }
            #pragma GCC unroll 4
            for (int r_grp = 0; r_grp < 4; ++r_grp) {
                const int row_off = face_bot + 2 * r_grp;
                const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
                const sfpi::vUInt w = sfpi::as<sfpi::vUInt>(w_i);
                sfpi::vFloat v = decode_hlut4_lut<KT_LOCAL, NT_SUB>(w, s01, i01, s23, i23, s45, i45);
                v = v * sfpi::vFloat(sfpi::dst_reg[sc_idx]);
                sfpi::dst_reg[NT_SUB * 32 + row_off] = v;
            }
        }
    }
}

}  // namespace ckernel::sfpu
#endif

#define CALL_HLUT4_4NT(KL) \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, 1>())); \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, 2>())); \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, 3>()))

inline void pack_4nt_to_cb1(uint32_t base_tile) {
    tile_regs_wait();
    pack_tile<true>(0, 1, base_tile + 0u);
    pack_tile<true>(1, 1, base_tile + 1u);
    pack_tile<true>(2, 1, base_tile + 2u);
    pack_tile<true>(3, 1, base_tile + 3u);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t num_sb        = get_arg_val<uint32_t>(0);
    const uint32_t batch_cnt     = get_arg_val<uint32_t>(1);
    const uint32_t num_projs     = get_arg_val<uint32_t>(2);
    const uint32_t keep_act      = get_arg_val<uint32_t>(3);
    const uint32_t slot_start    = get_arg_val<uint32_t>(4);
    const uint32_t kt_core       = num_sb * 4u;
    const uint32_t total_w_tiles = num_sb * 16u;

    cb_wait_front(6, 1);
    const uint32_t scratch_addr = get_tile_address(6, 0);
    const volatile uint16_t* slot_u16 = reinterpret_cast<const volatile uint16_t*>(scratch_addr);
    uint16_t local_eids[24];
    bool any_valid = false;
    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint16_t e = slot_u16[slot_start + b];
        local_eids[b] = e;
        if (e < 0xFFFEu) any_valid = true;
    }
    if (!any_valid) return;

    compute_kernel_hw_startup(4, 4, 1);
    reconfig_data_format_srca(4);
    copy_init(4);
    MATH((SFPU_BINARY_INIT(unused)));
    bool need_sfpu_reconfig = false;

    if (keep_act != 0u) cb_wait_front(0, kt_core);

    auto unpack_weights_to_cb1 = [&]() __attribute__((always_inline)) {
        if (need_sfpu_reconfig) {
            reconfig_full_operand(4, 4);
            pack_reconfig_data_format(1);
            reconfig_data_format_srca(4);
            copy_init(4);
            MATH((SFPU_BINARY_INIT(unused)));
        }
        cb_reserve_back(1, total_w_tiles);
        for (uint32_t sb = 0u; sb < num_sb; ++sb) {
            const uint32_t sb_base = sb * 16u;
            cb_wait_front(4, 2);
            cb_wait_front(5, 2);
            for (uint32_t half = 0u; half < 2u; ++half) {
                tile_regs_acquire();
                copy_tile(4, half, 6);
                copy_tile(5, half, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0)));
                CALL_HLUT4_4NT(0);
                MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit();
                pack_4nt_to_cb1(sb_base + (half * 2u + 0u) * 4u);

                tile_regs_acquire();
                copy_tile(4, half, 6);
                copy_tile(5, half, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0)));
                CALL_HLUT4_4NT(1);
                MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit();
                pack_4nt_to_cb1(sb_base + (half * 2u + 1u) * 4u);
            }
            cb_pop_front(4, 2);
            cb_pop_front(5, 2);
        }
        cb_push_back(1, total_w_tiles);
        cb_wait_front(1, total_w_tiles);
    };

    auto matmul_one_strip = [&]() __attribute__((always_inline)) {
        tile_regs_acquire();
        for (uint32_t inner_dim_idx = 0u; inner_dim_idx < kt_core; ++inner_dim_idx) {
            matmul_block(0, 1, inner_dim_idx, inner_dim_idx * 4u, 0, 0, 4, 1, kt_core);
        }
        tile_regs_commit();
        cb_reserve_back(16, 4);
        tile_regs_wait();
        pack_block(0, 16, 4);
        tile_regs_release();
        cb_push_back(16, 4);
    };

    auto matmul_two_strips = [&]() __attribute__((always_inline)) {
        tile_regs_acquire();
        for (uint32_t inner_dim_idx = 0u; inner_dim_idx < kt_core; ++inner_dim_idx) {
            matmul_block(0, 1, inner_dim_idx, inner_dim_idx * 4u, 0, 0, 4, 2, kt_core);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(16, 4);
        pack_block(0, 16, 4);
        cb_push_back(16, 4);
        cb_reserve_back(16, 4);
        pack_block(4, 16, 4);
        cb_push_back(16, 4);
        tile_regs_release();
    };

    const bool can_rt2 = (keep_act == 0u && num_projs == 2u);
    uint32_t cb0_half = 0u;
    uint32_t b0 = 0u;
    while (b0 < batch_cnt) {
        const uint32_t expert_slot = local_eids[b0];
        if (expert_slot >= 0xFFFEu) { ++b0; continue; }
        uint32_t run_len = 1u;
        while (b0 + run_len < batch_cnt && local_eids[b0 + run_len] == expert_slot) ++run_len;

        const bool hold_2 = (can_rt2 && run_len == 2u && cb0_half == 0u);
        if (keep_act == 0u && run_len == 1u) cb_wait_front(0, kt_core);
        else if (hold_2)                     cb_wait_front(0, 2u * kt_core);

        for (uint32_t p = 0u; p < num_projs; ++p) {
            unpack_weights_to_cb1();
            reconfig_full_operand<SrcOrder::Reverse>(0, 1);
            pack_reconfig_data_format(1, 16);
            need_sfpu_reconfig = true;

            if (keep_act != 0u || run_len == 1u) {
                matmul_block_init(0, 1, 0, 4, 1, kt_core);
                for (uint32_t r = 0u; r < run_len; ++r) matmul_one_strip();
            } else if (hold_2) {
                matmul_block_init(0, 1, 0, 4, 2, kt_core);
                matmul_two_strips();
            } else if (can_rt2) {
                uint32_t cur_rt = 0u, r = 0u;
                while (r < run_len) {
                    if (cb0_half == 0u && r + 2u <= run_len) {
                        if (cur_rt != 2u) { matmul_block_init(0, 1, 0, 4, 2, kt_core); cur_rt = 2u; }
                        cb_wait_front(0, 2u * kt_core);
                        matmul_two_strips();
                        cb_pop_front(0, 2u * kt_core);
                        r += 2u;
                    } else {
                        if (cur_rt != 1u) { matmul_block_init(0, 1, 0, 4, 1, kt_core); cur_rt = 1u; }
                        cb_wait_front(0, kt_core);
                        matmul_one_strip();
                        cb_pop_front(0, kt_core);
                        cb0_half ^= 1u;
                        r += 1u;
                    }
                }
            } else {
                matmul_block_init(0, 1, 0, 4, 1, kt_core);
                for (uint32_t r = 0u; r < run_len; ++r) {
                    cb_wait_front(0, kt_core);
                    matmul_one_strip();
                    cb_pop_front(0, kt_core);
                }
            }
            cb_pop_front(1, total_w_tiles);
        }

        if (keep_act == 0u && run_len == 1u) {
            cb_pop_front(0, kt_core);
            if (can_rt2) cb0_half ^= 1u;
        } else if (hold_2) {
            cb_pop_front(0, 2u * kt_core);
        }
        b0 += run_len;
    }

    if (keep_act != 0u) cb_pop_front(0, kt_core);
}
