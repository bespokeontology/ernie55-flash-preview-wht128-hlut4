<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-14 Turn 3 — Cut W14.3 Cooled 8-Channel `cost_w` Minimax Re-Calibration and `512K–1M` Single-Stream Decode Tightening (Gate-0 Suite 49)

Workstation `SB-M5-P2-192G`, Build W14-2 live on `:8052` with the persistent `/dev/shm/ernie55_kvblk_ddr4`
arena re-attaching in `16,825.84 ms`.

No `make`, no `hipcc`, no daemon / PCIe sysfs / Python access was performed during this turn.

## 1. Cooled derivation of `cost_w[5]`

Inputs: `logs/w14_b2_cooled_uninst.log`, `logs/w14_b2_cooled_xray.log`,
`logs/w14_b2_cooled_thermals.log`, `checkpoints/active/router_hist_ernie55_agent.txt` (42 layers x 512),
`checkpoints/active/router_perm.bin`.

Per-chunk `g0..g4` are **not** comparable as printed (`mi50/ernie55_runtime.cpp:6884-6960`): `g0`/`g1`
accumulate `t_owner_end - t_m0` (which includes `p150_wait`) on their 21 owner layers, while
`g2`/`g3`/`g4` accumulate pure remote MI50 completion over all 42 layers. The cooled data therefore
splits into

* pure remote completion per layer: `g2 = 485/42 = 11.55 ms`, `g3 = 456/42 = 10.86 ms`,
  `g4 = 475/42 = 11.31 ms` — and `mi50 == g2` on **all 10** chunks, i.e. `g2` is the per-layer
  critical path;
* `g0`/`g1` corrected for thermals (`g0` 93C, `g1` 97C with `1606 MHz` residency, `g4` 62C).

Dividing the measured per-layer ms by the assigner's raw load (`proj_ms / cost_w`) gives the
hardware cost per raw unit

```
ms_per_raw = [0.0775, 0.0828, 0.07164, 0.07263, 0.07958]   // g0..g4
```

which reproduces the measured `mi50` almost exactly (proxy `491.0 ms` vs measured `485 ms`, +1.2%).
Re-running the minimax over all 42 layers with `cost_w[g] = ms_per_raw[g]` (scale-invariant, `w[2]`
pinned at `0.90`) and rounding to two decimals gives

```cpp
double cost_w[5] = {1.05, 1.15, 0.90, 1.03, 1.10};   // was {1.15, 1.15, 0.90, 1.08, 1.12}
```

| metric | Wave-13 `{1.15,1.15,0.90,1.08,1.12}` | W14.3 `{1.05,1.15,0.90,1.03,1.10}` |
|---|---|---|
| chunk `mi50` proxy (sum over layers of per-layer max) | `491.0 ms` | **`478.6 ms`** (`-2.5%`) |
| worst single layer | `13.286 ms` | **`12.388 ms`** (`-6.8%`) |
| `g0` | `10.46 ms` | `11.08 ms` (balanced up) |
| `g1` (97C, warmest) | `11.32 ms` | **`11.02 ms`** (relieved) |
| `g2` (the `mi50` bottleneck) | `11.55 ms` | **`11.05 ms`** (relieved) |
| `g3` (80C, underutilized) | `10.86 ms` | **`11.08 ms`** (loaded) |
| `g4` (62C, coolest) | `11.31 ms` | `11.27 ms` (held) |
| per-GPU spread | `1.09 ms` | **`0.19 ms`** |
| P150A shadow capture | `29.14%` | `29.14%` (unchanged) |

Fixed slot caps `{66, 68, 97, 38, 49}` and the `318 + 7` shadow layout are untouched, the delta is
`0 B` of GPU VRAM, and `ERNIE55_MOE_COST` still overrides any element.

## 2. `512K–1M` decode tightening: occupancy-aware HCA split count

`mi50/ernie55_core_kernels.hip`, `launch_hca_decode_splitk_mfused` (final Build W14-3 form).

On TP4 MI50 the HCA head count per rank is `h_loc = kHcaNumHeads / 4 = 16`, so the pass-1 grid is
`(n_heads/4, m_tiles, n_splits) = (4, m_tiles, n_splits)`. `part_o_f32` is
`[n_splits, M, n_heads, 512]` F32 and `k_hca_decode_splitk_merge` reads every term, so the split
count **is** the HBM2 traffic of the decode pass:

| `n_splits` | `part_o_f32` per HCA layer per GPU at `M = 6`, `n_heads = 16` | over 37 HCA layers (write + read) | across TP4 |
|---|---|---|---|
| `33` (Wave-13, `num_comp >= 2048`) | `6,488,064 B` = `6.19 MiB` | `458 MiB` | `1.92 GB` |
| `17` (W14.3) | `3,342,336 B` = `3.19 MiB` | `236 MiB` | `0.99 GB` |

The launcher now picks the **smallest power-of-two rung `<= base_splits`** whose `M = 1` pass-1
grid already fills one full MI50 wave of 60 CUs, and the choice is `M`-independent so `M = 1` and
`M = 6` use the identical partition (bit-exact `M = 1 == M = 6`):

```
for (cand = 4; cand <= base_splits; cand <<= 1) { n = cand; if (x_blocks*(cand+1) >= 60) break; }
```

With `x_blocks = 4` this selects `n_comp_splits = 16` (`17` terms) for every `num_comp >= 2048`,
i.e. across the whole `256K..1M` range: `1.94x` lower `part_o_f32` HBM2 traffic and merge loop
iterations, `zero` regression below `256K` (the ladder is unchanged there), and `n_comp_splits` is
capped strictly `<= base_splits`. `ERNIE55_HCA_DEC_SPLITS=4..32` pins a rung so the silicon A/B can
be run both ways.

The partition only **regroups** the exact max-rescale reduction; Suite 49 proves the `33`-term and
`17`-term partitions agree to `cosine = 1.000000000` on a scalar model of
`k_hca_decode_splitk_p1_mfused` + `k_hca_decode_splitk_merge`, and the `M = 1 == M = 6` identity
was verified on live MI50 silicon at `1M` (`num_comp = 8192`, `M = 6`, `h_loc = 16`,
`min_cos = 1.0000000`). The greedy-coherence gate (`A=1 B=1 CPP=1`) is the acceptance test.

### Byte / VGPR / LDS accounting

* `part_o_f32` is pre-existing scratch carved from `d_y_f32` — **no new allocation**, `0 B` net new
  `hipMalloc`, `ScratchSize == 0`, `0` spills / trig / `slc`.
* `s_k[2][4][64]` u4 = `8 KiB` ping-pong LDS, unchanged.
* `__launch_bounds__(256, MIN_BLOCKS)` with `MIN_BLOCKS = 4` (`M_TILE = 1`) / `2` (`M_TILE = 4`) is
  unchanged, so the VGPR cap and the `0`-spill result are unchanged.
* No new kernel is added, so the `344/344` zero-scratch `gfx906` census is unchanged.

## 3. Gate-0 Suite 49

`tools/ernie55_verify_unit.cpp` — `w14_c3_costw_minimax_and_decode_splits()`, registered in `main()`
(banner now `ALL 49`). Host-only, no GPU required.

1. **Placement**: loads the real `router_perm.bin` + `router_hist_ernie55_agent.txt`, then for both
   the Wave-13 and W14.3 weights runs the real `assign_moe_layer_descending` over all 42 layers and
   asserts the `512/512` bijection (`318 + 7 = 325` owned experts, every owned expert carries a
   local slot), the fixed caps `{66, 68, 97, 38, 49}` via `base_count` and `ids` (`cap + shadow_seed`),
   `mi50` proxy and worst-layer both down, `g2`/`g1` relieved, `g3` loaded, `g4`/`g0` balanced,
   spread `<= 0.35 ms`, minimax gain `> 5 ms`, unchanged base counts and shadow capture, and
   `kExtraVramBytes == 0`.
2. **Split count**: mirrors the launcher rule (`h_loc = 16`, `x_blocks = 4`) and asserts
   `n_comp_splits = 16` (`17` terms) for every `num_comp >= 2048` independent of `M`, a pass-1 grid
   `>= 60` CUs at `M = 1`, and the exact `part_o_f32` byte figures (`6,488,064` -> `3,342,336` per
   layer per GPU, a `1.94x` drop).
3. **Parity**: a scalar model of the split-K online-softmax reduction over `16`/`64` heads and
   `4096`/`8192` keys, comparing the `33`-way and `17`-way partitions — `cosine >= 0.999999`
   (measured `1.000000000`).

Final Build W14-3 receipt: `[SUITE49] cost_w=1.05/1.15/0.90/1.03/1.10 mi50proxy 491.0->478.6 ms
worstlayer 13.29->12.39 ms spread=0.191 ms hw=[11.08 11.02 11.05 11.08 11.27] splits 33->17 part_o
6.19->3.19 MiB/L/GPU min_cos=1.0000000 vram=0 B`.

## 4. Files edited

* `mi50/ernie55_runtime.cpp` — cooled `cost_w[5]` default `{1.05, 1.15, 0.90, 1.03, 1.10}` + derivation comment.
* `mi50/ernie55_core_kernels.hip` — occupancy-aware `n_comp_splits` in `launch_hca_decode_splitk_mfused`
  (+ `ERNIE55_HCA_DEC_SPLITS` A/B override, `<cstdint>`/`<cstdlib>`).
* `tools/ernie55_verify_unit.cpp` — Suite 49 + registration, banner `ALL 49`.
