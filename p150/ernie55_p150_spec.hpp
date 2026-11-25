// SPDX-License-Identifier: Apache-2.0
#pragma once
// Compile-Checked Tenstorrent Blackhole P150A Hardware Contract for baidu/ERNIE-5.5-Flash (ernie5_5_moe)
// 100% uniform WHT128-HLUT4 (4.125 bpw, 8,448 B/superblock) across ALL experts (0 B CPU DDR4).
// Supports BOTH:
//   - 192G (2x P150A FABRIC_1D TP2): I_shard=1024, 128 sb/proj/card, 3.09375 MiB/expert/card (198/L = 25.12 GiB/card)
//   - 224G/240G (3x P150A EP3):      I=2048,       256 sb/proj/card, 6.18750 MiB/expert/card (100/L/card = 25.37 GiB/card)

#include "../src/ernie55_config.hpp"

#include <cstddef>
#include <cstdint>

namespace ernie55 {

inline constexpr uint32_t kP150CoresPerCard           = 110u;       // Blackhole P150A active Tensix grid under FABRIC_1D
inline constexpr uint32_t kP150Fabric1DBankBytes      = 1'459'200u; // Live verified main-L1 bank ceiling per core
inline constexpr uint32_t kP150Fabric1DCbGuardPerCore = 262'144u;   // 256 KiB/core Circular Buffer guard
inline constexpr uint64_t kP150Gddr6CapPerCard        = 31ULL * 1024ULL * 1024ULL * 1024ULL;

// Latent MoE geometry: H_lat = 2048, I_moe = 2048
inline constexpr uint32_t kP150LatentK                = uint32_t(kLatentDim);         // 2048
inline constexpr uint32_t kP150LatentTiles            = kP150LatentK / 32u;           // 64
inline constexpr uint32_t kP150Tp2InterShard          = uint32_t(kMoeInterSize / 2);  // 1024
inline constexpr uint32_t kP150Tp2InterTiles          = kP150Tp2InterShard / 32u;     // 32
inline constexpr uint32_t kP150EpInterFull            = uint32_t(kMoeInterSize);      // 2048
inline constexpr uint32_t kP150EpInterTiles           = kP150EpInterFull / 32u;       // 64

inline constexpr uint32_t kP150Tp2SbPerProj           = (kP150Tp2InterShard / 128u) * (kP150LatentK / 128u);  // 8 * 16 = 128
inline constexpr uint32_t kP150EpSbPerProj            = (kP150EpInterFull / 128u)   * (kP150LatentK / 128u);  // 16 * 16 = 256
static_assert(kP150Tp2SbPerProj == 128u, "ERNIE-5.5 TP2 projection must be 128 superblocks (8x16)");
static_assert(kP150EpSbPerProj  == 256u, "ERNIE-5.5 EP projection must be 256 superblocks (16x16)");

inline constexpr uint64_t kP150Hlut4Tp2CardExpBytes   = 3ULL * kP150Tp2SbPerProj * kHlut4SuperBytes;  // 3,244,032 B (3.09375 MiB)
inline constexpr uint64_t kP150Hlut4EpCardExpBytes    = 3ULL * kP150EpSbPerProj  * kHlut4SuperBytes;  // 6,488,064 B (6.18750 MiB)
static_assert(kP150Hlut4Tp2CardExpBytes == kHlut4ExpertBytes / 2ULL, "TP2 card bytes must be exact half of expert bytes");
static_assert(kP150Hlut4EpCardExpBytes  == kHlut4ExpertBytes,        "EP card bytes must equal full expert bytes");

// Compile-time GDDR6 fit checks for 192G (2xP150A TP2: 198/L) and 224G/240G (3xP150A EP3: 100/L/card = 300/L)
static_assert(uint64_t(kNumMoeLayers) * 198ULL * kP150Hlut4Tp2CardExpBytes <= kP150Gddr6CapPerCard,
              "192G TP2: 42 x 198 WHT128-HLUT4 experts (25.12 GiB/card) must fit 31 GiB P150A GDDR6");
static_assert(uint64_t(kNumMoeLayers) * 100ULL * kP150Hlut4EpCardExpBytes <= kP150Gddr6CapPerCard,
              "224G/240G EP3: 42 x 100 WHT128-HLUT4 experts (25.37 GiB/card) must fit 31 GiB P150A GDDR6");

// Exact Tensix L1 SRAM Hot-Expert Pinning Capacity per card (across 110 active cores)
inline constexpr size_t p150_l1_per_bank_bytes(uint32_t hot_shards, uint32_t sb_per_proj) noexcept {
    const size_t pages = size_t(hot_shards) * size_t(sb_per_proj);
    const size_t per_core_pages = (pages + kP150CoresPerCard - 1u) / kP150CoresPerCard;
    // Workspace + CB guard 256 KiB/core + 3 projections (gate/up/down)
    return 131'072u + kP150Fabric1DCbGuardPerCore + 3u * per_core_pages * size_t(kHlut4SuperBytes);
}

inline constexpr uint32_t kP150MaxHotL1Tp2 = 34u;  // 34 * 3.094 MiB = 105.2 MiB/card (210.4 MiB total across 2x P150A)
inline constexpr uint32_t kP150MaxHotL1Ep  = 17u;  // 17 * 6.188 MiB = 105.2 MiB/card (315.6 MiB total across 3x P150A)
static_assert(p150_l1_per_bank_bytes(kP150MaxHotL1Tp2, kP150Tp2SbPerProj) <= kP150Fabric1DBankBytes,
              "34 hot HLUT4 TP2 shards/card must fit 1,459,200 B FABRIC_1D L1 bank with 256 KiB CB guard");
static_assert(p150_l1_per_bank_bytes(kP150MaxHotL1Ep,  kP150EpSbPerProj)  <= kP150Fabric1DBankBytes,
              "17 hot HLUT4 EP shards/card must fit 1,459,200 B L1 bank with 256 KiB CB guard");

}  // namespace ernie55
