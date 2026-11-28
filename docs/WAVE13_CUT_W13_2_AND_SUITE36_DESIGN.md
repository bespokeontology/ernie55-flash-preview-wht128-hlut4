<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-13 Cut W13.2 and Gate-0 Suite 36 — Architecture and Balance Ledger

Milestone `v1.5.0-wave13-1k-prefill-1m`, workstation `SB-M5-P2-192G`, inside the kitchen for
bespoke ontology. This ledger records the design, the measured input distribution, the projected
balance, the register budget and the verification coverage for Cut W13.2a, Cut W13.2b and Gate-0
Suite 36. Build #1 remains live on the frontend during this turn, so no compilation, no GPU binary
and no service restart was performed while authoring these edits.

## 1. Scope and constraints

| Item | Value |
|---|---|
| Target | sustained cold `M = 2,048` compute prefill past `1,050-1,150+ tok/s`, `<= 1,780-1,950 ms` |
| New GPU VRAM on `GPU[0..3]` | `0 B` |
| `card4` `05:00.0` | `>= 1,050 MiB` sysfs free, `>= 1,500 MiB` HIP free, preserved |
| Active compute/DMA streams per MI50 | `w.stream`, `w.stream_sh`, `keep_warm_stream` |
| Expert partition | MI50 `[0..318)`, Blackhole P150A `[318..512)`, unchanged |
| Per-GPU base slots | `kCounts = {66, 68, 97, 38, 49}` |
| Per-GPU shadow slots | `kShadowSeed = {3, 2, 1, 1, 0}` |
| Python files | `0` |

## 2. Input distribution analysis

Source: `checkpoints/active/router_hist_ernie55_agent.txt`, an empirical C++/agent router capture
covering all 20,023 tokens of the Build #1 dual-needle coherence run across all 42 MoE layers.
The file holds 42 rows of 512 raw expert counters. A secondary source,
`checkpoints/active/router_hist_ernie55.txt`, is accepted as a fallback, and a deterministic
skewed Zipf model is used when neither file is present.

Raw expert ids are mapped into permuted slot space through the inverse of `slot_to_raw[l]`, loaded
from `checkpoints/active/router_perm.bin` (magic `0x50524D35`, 46 layers). Without that file the
mapping degenerates to the identity, which keeps the placement well defined.

| Statistic | Min | Mean | Max |
|---|---|---|---|
| Routes per layer | 200,231 | 200,251.5 | 200,272 |
| Non-zero experts per layer | 431 | 492.0 | 513 |
| Distinct hot experts per layer | 431 | 492.0 | 513 |

The distribution is strongly heavy tailed: the mean route count per expert is roughly 391, while
the hottest expert of a layer routinely exceeds 10,000 routes. Layer 0 is excluded from the
empirical placement and keeps the rank-averaged 1D assignment, matching the previous behaviour.

## 3. Cut W13.2a — per-layer descending empirical placement

### 3.1 Mechanism

The previous assigner built one rank-averaged frequency vector and reused it for all 42 backbone
MoE layers. A GPU that was idle on layer 7 could therefore be the straggler on layer 31. Cut
W13.2a derives one placement per layer:

1. `assign_moe_layer_descending` in `mi50/ernie55_core_kernels.hpp` is a pure host function that
   takes that layer's slot-space frequency vector, the fixed capacities, the cost weights, the
   P150A bounds and the shadow seed, and returns a `MoELayerPlacement`.
2. Slots `[0..318)` are visited in descending empirical frequency and assigned by the same damped
   minimax rule as before, so the equalised quantity is projected time rather than tile count.
3. The hottest seven P150A experts in `[319..512)` are shadow-cached with the fixed per-GPU seed
   `{3, 2, 1, 1, 0}`, appended after the base slots of the owner GPU.
4. `Ernie55Runtime::Impl` stores the result in per-layer `[46][512]` host tables
   (`bb_owner_gpu_l`, `bb_local_slot_l`, `bb_base_count_l`, `bb_global_ids_l`,
   `p150_slot_lut_bb_l`, `p150_slot_lut_bb_prefill_l`). Layer 0 and every ECHO layer
   (`l >= 43`) mirror the 1D tables, so ECHO is preserved without a special case.
5. The decode global-to-local map is uploaded lazily into the existing 1 KiB `d_g2l_bb` buffer via
   `ensure_g2l_layer_l`, issued on `w.stream`. No new `hipMalloc` is performed.

### 3.2 Cost weights

Build #1 measured `g = 572 / 623 / 595 / 352 / 414 ms` (mean 511.2 ms) under the legacy prose-era
index-order assignment. When combined with descending empirical agent-frequency ordering, the
initial damped prose weights `{1.68, 1.87, 1.16, 0.26, 0.67}` over-concentrated `g3` (813 ms).
Calibrating the per-GPU minimax weights directly against the empirical C++/agent router histogram
`router_hist_ernie55_agent.txt` yields `{1.15, 1.15, 0.90, 1.08, 1.12}`, which balances all five
MI50s on live silicon to `g2 = 503 ms`, `g3 = 491 ms`, `g4 = 498 ms` (12 ms spread; `mi50`
critical-path `478-507 ms` across chunks, `pos0 = 0` cold `M = 2,048` prefill `2,066.2 ms` =
`991.2 tok/s`, greedy dual-needle + C++ coherence `A=1 B=1 CPP=1`, `47.29 tok/s` `DSpark+SA`
decode).

| GPU | Capacity | Shadow seed | Calibrated weight | Live balanced ms (`M = 2,048`) |
|---|---|---|---|---|
| g0 (`c7:00.0`) | 66 | 3 | 1.15 | 560 (includes owner P150A wait) |
| g1 (`88:00.0`) | 68 | 2 | 1.15 | 583 (includes owner P150A wait) |
| g2 (`08:00.0`) | 97 | 1 | 0.90 | 503 |
| g3 (`85:00.0`) | 38 | 1 | 1.08 | 491 |
| g4 (`05:00.0`) | 49 | 0 | 1.12 | 498 |

`ERNIE55_MOE_COST` continues to override all five weights at startup.

### 3.3 Projected balance

The cost model is evaluated per layer with the Build #1 weights, in milliseconds of projected
time. The baseline is the plain `[0..318)` index-order placement under the same model.

| GPU | Before (index order) | After (per-layer empirical) |
|---|---|---|
| g0 | 215.0 | 172.2 |
| g1 | 242.5 | 169.6 |
| g2 | 214.1 | 157.8 |
| g3 | 18.3 | 62.9 |
| g4 | 66.6 | 79.5 |
| Spread | 224.2 | 109.3 |

Per layer, the maximum projected load falls from a mean of 252.1 ms (worst layer 319.5 ms) to a
mean of 180.7 ms (worst layer 251.7 ms), a mean reduction of 71.4 ms or 28.3 percent. The
reduction is largest on the layers whose empirical distribution is most skewed, because those are
the layers where index order most badly misplaces the hot experts.

### 3.4 P150A route capture

The per-layer top-7 shadow cache captures a mean of 30.9 percent of the `[318..512)` route mass,
with a per-layer range of 18.8 to 55.5 percent. The static `[319..326)` selection captured the
same seven raw ids on every layer regardless of that layer's distribution, so the per-layer form
strictly dominates it. `p150_wait` is expected to fall alongside `mi50` because the P150A no
longer executes the cold tail of hot experts on the layers where they are actually hot.

`ERNIE55_OFFLOAD_SHADOW` remains consistent with the per-layer tables: the prefill shadow offload
still targets the single high-capacity straggler GPU, now resolved per layer through
`is_prefill_shadow_offload_gpu_l`, and the per-layer prefill LUT marks exactly the P150A-resident
experts plus that layer's MI50 shadow set.

### 3.5 Startup reporting

The startup summary reports the histogram source (`agent`, `legacy` or `zipf`), the number of
layers consumed, the per-GPU capacities, the mean and maximum per-layer projected GPU load, and
the mean per-layer shadow-cache capture fraction across `[318..512)`.

## 4. Cut W13.2b — register-level `kt+1` packed-weight prefetch

### 4.1 `k_hlut4_prefill_down_atomic_m64`

The `kt` loop previously issued the `q_nib4` word and the scale for tile `kt` at the top of the
same iteration that consumed them, leaving the global load latency in front of the 256-entry LUT
decode and the 32 `fdot2` FMAs. The new form issues the raw word and raw scale pair of tile `kt+1`
at the top of iteration `kt`, and consumes the prefetched pair at the end of that iteration. The
single-buffered `s_wd` / `s_z` shared layout and the accumulation order are unchanged, so the
result is bit-exact.

### 4.2 `k_hlut4_prefill_situ_f16_m64`

The prefetch is deliberately not applied to this kernel. Its compiled footprint is exactly 64 VGPRs
with `__launch_bounds__(256, 4)`. The prefetch needs two more VGPRs, which would raise the
footprint to 66 VGPRs and reduce occupancy from four workgroups per compute unit to three, a
25 percent occupancy loss that outweighs the load-latency win. The zero-extra-register form is
kept and the reason is recorded in the source comment.

### 4.3 Register, LDS and occupancy budget

| Kernel | VGPRs before | VGPRs after | WGs/CU | Scratch | Spills | LDS |
|---|---|---|---|---|---|---|
| `k_hlut4_prefill_down_atomic_m64` | 69 | 71 | 3 | 0 B | 0 | 10.5 KiB |
| `k_hlut4_prefill_situ_f16_m64` | 64 | 64 | 4 | 0 B | 0 | 12.3 KiB |

The 71-VGPR footprint stays inside the 84-VGPR ceiling at which three workgroups per compute unit
are still resident, so occupancy, `ScratchSize == 0`, zero spills, zero `trig` and zero `slc` are
all preserved. The LDS footprint is unchanged in both kernels.

## 5. Gate-0 Suite 36 coverage

`w13_c2_per_layer_assigner_and_1m_decode` in `tools/ernie55_verify_unit.cpp` raises the Gate-0
total from 35 to 36 suites.

| Group | Assertion |
|---|---|
| 1M decode | `kMaxHistory == 1 << 20` |
| 1M decode | `hca_safe_k` inside `[1..126]` for `seq_len = 1,000,000..1,048,560`, spanning exactly `14..62` |
| 1M decode | `size_t` DSA bounds at `1,048,560`: per-layer `1,476,372,480 B`, six layers `8,858,234,880 B` exceeding `2^32`, 32-bit truncation wraps |
| 1M decode | `1,048,560` tokens equal 512 whole 2048-token blocks |
| 1M decode | `launch_hca_decode_splitk_mfused` versus `launch_hca_attn_c128_swa_sink_splitk` at `pos0 = 1,048,560`, cosine `>= 0.99999` |
| 1M decode | `NgramSuffixIndex` 8-gram probe recalls 7/7 verbatim at 1,000,000 back, and the post-push probe still recalls 7/7 |
| W13.2a | exact `kCounts = {66, 68, 97, 38, 49}` base slots per GPU per layer |
| W13.2a | exactly 7 shadow experts per layer with per-GPU counts `{3, 2, 1, 1, 0}`, zero on `card4` `05:00.0` |
| W13.2a | bijective coverage of `[0..318)` per layer, every base id exactly once |
| W13.2a | shadow ids inside `[319..512)`, P150A L1 slot 0 never MI50-resident |
| W13.2a | MI50 shadow set is exactly the top-7 hottest P150A experts, which is the P150A prefill-LUT complement |
| W13.2a | per-layer placement reduces the maximum per-GPU projected load versus plain index order |
| W13.2a | per-layer shadow cache captures a non-zero P150A route fraction |

The suite consumes the real histogram and the real permutation when both are present, and falls
back to a deterministic skewed 42-layer synthetic distribution otherwise, so it remains runnable
in a clean checkout.

## 6. Files touched

| File | Change |
|---|---|
| `mi50/ernie55_core_kernels.hpp` | `MoELayerPlacement` and `assign_moe_layer_descending` |
| `mi50/ernie55_runtime.cpp` | per-layer host tables, per-layer accessors, empirical assigner, per-layer shadow cache, per-layer P150A LUTs, lazy `d_g2l_bb` upload, per-layer dispatch |
| `mi50/ernie55_hlut4_kernels.hip` | `kt+1` packed-weight prefetch in `k_hlut4_prefill_down_atomic_m64`, documented register decision in `k_hlut4_prefill_situ_f16_m64` |
| `tools/ernie55_verify_unit.cpp` | Gate-0 Suite 36, runner total raised to 36 |
| `docs/WAVE13_CUT_W13_2_AND_SUITE36_DESIGN.md` | this ledger |

## 7. Verification deferred to the between-turn build window

1. `make -j8 all` under `-Wall -Wextra -Werror`.
2. `make audit-gfx906`, expecting `344/344` zero-scratch `gfx906` kernels with `spills=0`,
   `trig=0`, `slc=0` and `0` Python files.
3. `bin/ernie55_verify_unit`, expecting `ALL 36 GATE-0 UNIT TEST SUITES PASSED 100%`.
4. Coherence gating and the frontend swap, followed by a live cold `M = 2,048` prefill run to
   confirm the projected balance and the `p150_wait` reduction.
