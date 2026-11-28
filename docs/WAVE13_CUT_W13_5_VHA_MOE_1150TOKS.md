<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wave-13 Cut W13.5 — `vha_op` Acceleration, `moe_post` Pipeline Compression and Tile-Wave Tightening

Milestone `v1.5.0-wave13-1k-prefill-1m`, workstation `SB-M5-P2-192G`, inside the kitchen for
bespoke ontology. Build #3 is live on the frontend and `p150_ernie55_worker` is live during this
turn, so no compilation, no GPU binary and no service restart was performed. The edits are staged
for the between-turn Build #4 window.

## 1. Build #3 X-ray root-cause analysis

Build #3 broke `1,000 tok/s` on cold `M = 2,048` prefill (`pos0 = 0`: `2,045.4 ms` =
`1,001.3 tok/s`) while improving `vha_op` by `36-39 ms` and the P150A barrier count from about
`33` to `11.5-13.6`. The remaining gap to `1,080-1,150+ tok/s` (`<= 1,780-1,895 ms`) is
`220-340 ms/chunk`, dominated by four buckets.

| Bucket | Build #3 | Delta vs Build #2 |
|---|---|---|
| `moe_post` | `178.6-184.6 ms` | `+20.7 / +24.1 ms` (regression) |
| `vha_op` | `541.5 / 630.6 ms` | `-38.7 / -36.5 ms` (improved) |
| `qkv` | `171.1 / 171.4 ms` | unchanged |
| `attn` | `124.5 / 100.2 ms` | improved |
| `mi50` | `511-533 ms` (`g = 557/569/510/499/506`) | `g0`/`g1` lead by `+50-70 ms` |
| `p150_wait` | `92-187 ms` | improved |

### 1.1 The `moe_post` regression

Cut W13.4b moved BOTH the `8 MiB` `ch->x_rot_f16` D2H and the `8 MiB` `h_lat_rot_f16_pin` D2H
onto `g.stream_sh` before `shared_experts`. `shared_experts` is enqueued on the same
`g.stream_sh`, so it was queued behind `16 MiB` of D2H, stretching `moe_post` from `154.5` to
`178.6 ms`.

The dependency is the key observation: `shared_experts` (`sh_w1`, `sh_w3`, `SiTU`, `sh_w2`)
depends ONLY on `g.d_x_f16`, the post-`mlp_mhc` RMSNorm input. It does not depend on `router`,
on `fc1_lat` or on `WHT128`. The `ch->x_rot_f16` D2H is the P150A doorbell payload, so it belongs
on the compute stream; the `h_lat_rot_f16_pin` D2H is the only one that must be on the DMA
stream, and it is now enqueued after `shared_experts`.

### 1.2 The `vha_op` asymmetry

VHA AllReduce is dual-owner: `g0` reduces `MB0` and `g1` reduces `MB1`, splitting the volume
across two root complexes. The Step 4 `o_proj` AllReduce still funnels `3 x 16 MiB = 48 MiB` of
H2D onto one `g_own` GPU and then D2Hs `16 MiB` back. The measured asymmetry is quantified in
Gate-0 Suite 38 (`50,331,648 B` total, `25,165,824 B` per owner under a dual-owner split).

### 1.3 The P150A tile-wave granularity

Colouring at whole-expert granularity forces an expert with `60` tokens into a fresh colour
because of a single conflicting row. Colouring at `32`-row tile granularity, with the constraint
that a tile is never placed earlier than its predecessor, keeps per-expert order and adjacent
same-expert tiles together whenever that wave is free, so GDDR6 weight reuse is unchanged.

## 2. Cut W13.5 changes

### 2.1 `mi50/ernie55_runtime.cpp` — `moe_post` regression removal

| Change | Effect |
|---|---|
| `ch->x_rot_f16` D2H returns to `g.stream`, with `ev_mb_d2c[1]` recorded there | the P150A doorbell host sync drains only that `8 MiB` copy instead of also draining `shared_experts` |
| `h_lat_rot_f16_pin` D2H stays on `g.stream_sh` but is enqueued after `shared_experts` | `shared_experts` is no longer queued behind `8 MiB` of D2H |
| `ev_mb_c2d[2]` / `ev_xrot_host_ready` still recorded on `g.stream_sh` after the D2H | the non-owner device-side release is unchanged, no host sync before the non-owner H2D |

Expected recovery: `+24 ms` regression removed, plus the `shared_experts` queueing delay, so
`moe_post` returns toward `150-160 ms` and the `g0`/`g1` CU contention that inflated their MoE
wall time by `+50-70 ms` is reduced.

### 2.2 `mi50/ernie55_runtime.cpp` — `cost_w` recalibration

The Build #3 per-GPU wall times `557/569/510/499/506 ms` (mean `528.2 ms`) are folded in with
the same damped first-order rule used by Cut P12.2, `w *= (measured / mean)^0.7`, giving
`{1.194, 1.211, 0.878, 1.038, 1.087}`. Because the greedy minimises `cost_w[g] * projected_time`,
the stragglers `g0`/`g1` become more expensive and `g2`/`g3`/`g4` cheaper, which moves work off
the tail. `ERNIE55_MOE_COST` still overrides all five weights.

### 2.3 `p150/p150_ernie55_worker.cpp` — per-tile wave colouring

`run_compact_prefill` now colours at `32`-row tile granularity. A tile may move to a later wave
than its predecessor, but never an earlier one, so per-expert order is preserved and adjacent
same-expert tiles still land in the same wave whenever that wave is free. On the deterministic
path-graph model the wave count falls from `511` to `2`; on the live `M = 2,048` routing the
target is `13-15` down to `5-7` waves and `bar` down to `4-6`.

### 2.4 `tools/ernie55_chart_gen.cpp` — Wave-13 rendering

`--wave13-json` is accepted and defaults to
`artifacts/gate_wave13_1k_prefill_1m_receipt.json`. When that receipt carries `release` or
`pillar1_prefill_m2048` it becomes the primary source and the banner, panel titles, legend and
signoff tag switch to `v1.5.0-wave13-1k-prefill-1m`. Without it the renderer is byte-identical to
the Wave-12 and Wave-11 behaviour. Output remains IHDR/IDAT/IEND only.

## 3. Expected millisecond savings

| Stage | Expected |
|---|---|
| `moe_post` | `-24 ms` regression removed, `-10 to -20 ms` further from the `shared_experts` queueing fix |
| `g0`/`g1` MoE wall time | `-20 to -40 ms` from the `cost_w` rebalance |
| P150A `waves` / `bar` | `13-15 -> 5-7` waves, `11.5-13.6 -> 4-6` barriers, `-40 to -80 ms` per chunk |
| `vha_op` | quantified for a follow-up dual-owner `o_proj` reduction (`-100 to -170 ms` available) |

## 4. Verification checklist

Gate-0 Suite 38 (`w13_c5_vha_moe_pipeline_and_tile_coloring`) asserts:

| Item | Assertion |
|---|---|
| `o_proj` AllReduce volume | exactly `48 MiB`, `3 x 2 x 8 MiB`, `24 MiB` per owner under a dual-owner split |
| VHA microbatch slab | exactly `4 MiB` |
| `x_rot` D2H size | exactly `8 MiB` each, `16 MiB` total for the Build #3 regression |
| `shared_experts` shape | three tuned FP16 GEMMs plus one fused SiTU-GLU |
| `fc2_lat` sizing | single-shot `M = 2048` is `0.238 ms` cheaper than `2 x 1024` |
| `cost_w` direction | `g0`/`g1` weights rise, `g2`/`g3`/`g4` fall |
| Tile conservation | both colourings conserve every tile and every pair |
| Per-tile ordering | a tile is never placed earlier than its predecessor |
| Conflict freedom | every wave is strictly row-disjoint |
| Wave-0 safety | wave 0 is a bounded strict prefix |

Build #4 must additionally confirm:

1. `make -j8 all` under `-Wall -Wextra -Werror`.
2. `make audit-gfx906` with `344/344` zero-scratch `gfx906` kernels, `0` spills, `trig`, `slc` and Python files.
3. `bin/ernie55_verify_unit` reporting `ALL 38 GATE-0 UNIT TEST SUITES PASSED 100%`.
4. `p150_ernie55_worker --self-test` and `--synthetic-self-test` both passing bit-exact.
5. A live cold `M = 2,048` prefill run with `moe_post <= 160 ms`, `waves` in `5-7` and total time inside `1,780-1,895 ms`.
6. `bin/ernie55_chart_gen --wave13-json ...` rendering the 4-panel Wave-13 PNG.

## 5. Files touched

| File | Change |
|---|---|
| `mi50/ernie55_runtime.cpp` | `ch->x_rot_f16` D2H back on `g.stream`, `h_lat_rot_f16_pin` D2H after `shared_experts`, damped `cost_w` recalibration |
| `p150/p150_ernie55_worker.cpp` | per-tile wave colouring in `run_compact_prefill` |
| `tools/ernie55_chart_gen.cpp` | `--wave13-json` Wave-13 receipt support and active-wave labels |
| `tools/ernie55_verify_unit.cpp` | Gate-0 Suite 38, runner total raised to 38 |
| `docs/WAVE13_CUT_W13_5_VHA_MOE_1150TOKS.md` | this ledger |

## 6. Live silicon Build #4 A/B telemetry and locked selection

Build #4 compiled cleanly (`-Wall -Wextra -Werror`), passed `38/38` Gate-0 suites, `344/344`
zero-scratch `gfx906` ISA audit, `3/3` `ernie55_kvblk_verify --unit`, and `4/4` live P150A
`--self-test` (`cos = 0.99997326` at `M = 256`). Full cold `20,023`-token `M = 2,048` A/B on
live silicon isolated each experimental cut against Build #3 (`2,045.4 ms = 1,001.3 tok/s` at
`pos0 = 0`, `2,118-2,123 ms` at `pos0 = 2048..6144`):

- **Per-tile wave coloring on P150A**: `waves` remained `12.6-15.7` (`bar = 11.6-14.6`) while
  splitting multi-tile experts raised GDDR6 weight `runs` from `133.6` to `150.0-158.1`, adding
  `+0.55 ms/L` (`+23 ms/chunk`). Reverted to Build #3 whole-expert minimum-coloring (`13.88 ms/L`).
- **Dual-Owner `o_proj` AllReduce**: splitting `o_proj` across `g0` (`MB0`) and `g1` (`MB1`)
  added a second cross-device PCIe hop (`peer_g -> g_own`) on the `g_own` critical path (`+17 ms`).
  Reverted to single-owner direct `o_proj` AllReduce.
- **Moving `ch->x_rot_f16` D2H back onto `g.stream`**: stretched `g0`/`g1` MoE by `+15 ms`.
  Reverted to Build #3 `g.stream_sh` D2H placement.
- **Aggressive `cost_w = {1.194, 1.211, 0.878, 1.038, 1.087}`**: overloaded `g2`/`g3`/`g4` by
  `+19 ms`. Reverted to Build #3 `{1.15, 1.15, 0.90, 1.08, 1.12}`.
- **Retained for Wave-13 lock**: pipelined owner `mhc_post_update` behind `MB1 fc2_lat` D2H
  trigger, Wave-13 `tools/ernie55_chart_gen.cpp` (`--wave13-json`), and Gate-0 Suite 38 (`38/38`).


---

## 8. Addendum — Build #4b `pos0 >= 2,048` delta trace and the `ev_dsa_uniq_cnt` fence

### 8.1 The `+61.0 ms` delta, attributed

| Stage | `pos0 = 0` | `pos0 = 2048` | Delta |
|---|---|---|---|
| `hca` | 74 ms | 79 ms | `+5` |
| `dsa` | 91 ms (`fi=0`, `mqa=0`) | 106 ms (`fi=96`, `mqa=2`) | `+15` |
| `pre` (MoE pre) | 1,213 ms | 1,268 ms | `+55` |
| `mi50` | 531 ms | 508 ms | `-23` |
| `p150` | 101 ms | 109 ms | `+8` |
| `post` | 8 ms | 8 ms | `0` |
| total | 2,043.2 ms | 2,104.2 ms | `+61.0` |

The single largest attributable jump is `fi = 0 -> 96 ms`. At `pos0 = 0`, `max_hist = M - 128 = 1,920 < kDsaIndexTopK = 2,048`, so the whole TP4 FlashIndexer branch is skipped. At `pos0 = 2,048`, `max_hist = 3,968 > 2,048`, so the branch activates and pays the unique-KV compaction plus its host readback.

### 8.2 The fix

`mi50/ernie55_runtime.cpp` previously drained the entire compute stream with `hipStreamSynchronize(g_own.stream)` immediately after the 4-byte `d_uniq_cnt` D2H, once per slice and up to ~7× per slice while halving `cur_sp`. That drain also waited on unrelated work queued earlier on `g_own.stream`, so the `fi` timer measured the whole backlog rather than the readback.

The readback is now fenced by a dedicated cache/worker-owned event `ev_dsa_uniq_cnt`, created with `hipEventDisableTiming` alongside `ev_res_b_ready` and destroyed with it. The host waits on that event only; the gather is still enqueued on `g_own.stream` afterwards, so ordering and every output bit are unchanged. If event creation failed, the code degrades to the previous stream sync instead of waiting on a null event.

Cost: `0 B` new GPU VRAM, no new stream, no new scratch, no change to the decode path.

### 8.3 Expected effect

`fi` should fall from `96 ms` toward the true readback latency (a few hundred microseconds), and `pre` should stop absorbing the `g_own.stream` backlog, which is the second `+55 ms` contributor. Together with the unchanged `mi50 = 508 ms` and `p150 = 109 ms`, the target band `<= 1,930-2,015 ms` (`>= 1,015-1,060+ tok/s`) is reachable for `pos0 = 2,048..16,384`.

### 8.4 Verification

Suite 36 was extended (still `38` suites) with the Cut W13.5 invariants: the per-layer GPU bound and expert space are unchanged, `max_hist` at `1,048,560` is exactly `1,048,432`, and the FlashIndexer activation boundary is exactly `pos0 = 2,048` — the measured `+61 ms` cliff.
