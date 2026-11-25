// SPDX-License-Identifier: Apache-2.0
// Blackhole P150A ERNIE-5.5 Fused K-Split Reduction + Native Hardware SFPU SiTU-GLU + L1 WHT128 Compute Kernel
// Exact baidu/ERNIE-5.5 SiTU-GLU in FP32 DST registers (zero L1 spill!):
//   SiTU-GLU(g, u) = (beta * tanh(g / beta)) * sigmoid(g) * (linear_beta * tanh(u / linear_beta))
// Uses Blackhole native softcap_tile + copy_dest_values + sigmoid_tile + mul_binary_tile,
// followed by exact 4x4 L1 WHT128 Q_i matmul into CB 16!

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/softcap.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t k_splits         = get_arg_val<uint32_t>(0);  // 8 (TP2) or 4 (EP3)
    const uint32_t m_cnt            = get_arg_val<uint32_t>(1);
    const uint32_t beta_bits        = get_arg_val<uint32_t>(2);  // 4.0f  = 0x40800000
    const uint32_t beta_recip_bits  = get_arg_val<uint32_t>(3);  // 0.25f = 0x3E800000
    const uint32_t lin_bits         = get_arg_val<uint32_t>(4);  // 25.0f = 0x41C80000
    const uint32_t lin_recip_bits   = get_arg_val<uint32_t>(5);  // 0.04f = 0x3D23D70A

    compute_kernel_hw_startup(0, 1, 24);
    cb_wait_front(2, 16);

    for (uint32_t dm = 0u; dm < m_cnt; ++dm) {
        reconfig_full_operand(0, 1);
        pack_reconfig_data_format(24);
        cb_reserve_back(24, 4);

        for (uint32_t nt_sub = 0u; nt_sub < 4u; ++nt_sub) {
            tile_regs_acquire();
            reconfig_data_format_srca(0);

            cb_wait_front(0, 1);
            cb_wait_front(1, 1);
            copy_init(0);
            copy_tile(0, 0, 0);
            copy_tile(1, 0, 1);
            cb_pop_front(0, 1);
            cb_pop_front(1, 1);

            add_binary_tile_init();
            for (uint32_t ks = 1u; ks < k_splits; ++ks) {
                cb_wait_front(0, 1);
                cb_wait_front(1, 1);
                copy_tile(0, 0, 2);
                copy_tile(1, 0, 3);
                cb_pop_front(0, 1);
                cb_pop_front(1, 1);
                add_binary_tile(0, 2, 0);
                add_binary_tile(1, 3, 1);
            }

            // Native Blackhole FP32 DST SiTU-GLU(dst[0]=g, dst[1]=u) -> dst[0]:
            //   dst[2] = sigmoid(g)
            //   dst[0] = beta * tanh(g / beta)
            //   dst[1] = linear_beta * tanh(u / linear_beta)
            //   dst[0] = dst[0] * dst[2] * dst[1]
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(0, 2);
            sigmoid_tile_init();
            sigmoid_tile(2);
            softcap_tile_init();
            softcap_tile(0, beta_bits, beta_recip_bits);
            softcap_tile(1, lin_bits,  lin_recip_bits);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 24);
            tile_regs_release();
        }
        cb_push_back(24, 4);

        // Apply exact 4x4 L1 WHT128 rotation Q_i into CB 16
        cb_wait_front(24, 4);
        reconfig_full_operand<SrcOrder::Reverse>(24, 2);
        pack_reconfig_data_format(24, 16);
        matmul_block_init(24, 2, 0, 4, 1, 4);

        tile_regs_acquire();
        for (uint32_t k = 0u; k < 4u; ++k) {
            matmul_block(24, 2, k, k * 4u, 0, 0, 4, 1, 4);
        }
        tile_regs_commit();

        cb_reserve_back(16, 4);
        tile_regs_wait();
        pack_block(0, 16, 4);
        tile_regs_release();
        cb_push_back(16, 4);
        cb_pop_front(24, 4);
    }

    cb_pop_front(2, 16);
}
