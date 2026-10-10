# Pollard 3.5 on the fixed-rate EXL3 path: what it costs and the candidates

Pollard declares `"projection_K": [3, 3, 4]`-style triples per layer (some layers uniform K3). This note traces
what a [3, 3, 4] layer and a [3, 3, 3] layer in the same pack run through today (`feat/exl3-integer-rates`), counts
the per-tile cost of the paths that differ from a K3 pack's, says which of the Spark (compact) candidates carry over,
and describes the candidate branches. Nothing here was timed: the kernels were built and checked on Linux by CPU
emulation; the device work is tier 1-3 of `docs/exl3-compact-bench.md` (on `opt/exl3-compact-tools`) with
`scripts/bench_exl3_compact.py --fixed`.

## Where a Pollard layer is "padded"

`ProjectionLayout.tight()` (`src/exl3_quant.zig`) is `code_row_words == 320 x 144 x 16 K`: the cache slot's physical
expert row is exactly the logical one. The streamer gives each bank its stride in `StreamSource.bankArrays`
(`src/expert_stream.zig`): the per-layer `base` / `ext` banks take the layer's own segment lengths, so a [3, 3, 4]
layer's gate / up slots are tight K3 (48 shorts per tile row) and its down slots tight K4 (64). The shared
`transient` bank is `StorageGeometry.transient` (`src/sdk_ext/expert/stream.zig`): per component, the widest segment
over all layers. So in a pack with any K4 down layer, every layer's transient down rows are 64-short rows; if any
layer has a K4 gate or up, every layer's transient gate / up rows are too. Logical tiles stay a contiguous prefix of the
row (`tile = base + (tk x 144 + tn) x 16 K`); only `base = code + id x code_strides[0]` changes.

The decode route the served tier runs is stack4's: `routeForms(down_pair, gu_one)` then `routeBanked`
(`src/deepseek_v41_module.zig`), one launch per stage over a wave's rows of all three banks. `Banked.gateUp` /
`Banked.down` choose per launch, not per bank: `tightBanks(proj)` must hold for base, ext and transient together.

| layer | stage | K3 pack | Pollard pack (K4 down somewhere) | Pollard pack (K4 gate / up somewhere too) |
|---|---|---|---|---|
| [3, 3, 3] | gate / up | `b3_guone_k3_2304` (one launch) | same | `b3_mul1h_rate_2304__k3` twice |
| [3, 3, 3] | down | `b3_pair_k3_5120` | `b3_mul1h_rate_5120__k3` | same |
| [3, 3, 4] | gate / up | - | `b3_guone_k3_2304` | `b3_mul1h_rate_2304__k3` twice |
| [3, 3, 4] | down | - | `b3_mul1h_rate_5120__k4` | same |

Without the banked route (`Gemv.projectGu` / `project`, per bank) the base and ext banks of a [3, 3, 4] layer keep
their forms for gate / up; the transient rows and every K4 down go direct. `Gemv.initRates` prepares the direct
plans for the rates `rateMask` finds (K3 and K4 here; a pure-K3 pack prepares none).

## Decode: the `_rate_` kernels against the tight ones

`dsv41_exl3_mul1h_rate_{2304,5120}` (and `_b3_`) are templates on `K` registered as the base (K2) and the `__k3` /
`__k4` variants (`src/exl3_kernels.zig` `rate_kernels`, `ratesOf`). Their body is the K3 kernel's with two changes:
`base` through `code_strides[0]` (one constant-buffer read per threadgroup) and the tile at `16 K` shorts; the loop is
`if constexpr (K == 3)` the K3 kernel's two-word funnel extraction, verbatim, else `rate_window<K>` per window.

So **a padded K3 projection on `__k3` costs the same per tile as tight K3** (CPU emulation: `mul1h_k3` on a tight bank
and `mul1h_rate<3>` on a 64-word padded bank with the same logical codes are bit-identical, reference error 0). What
it loses is the forms: `gu_one` (gate and up in one launch of 2 x 144 x rows threadgroups instead of two launches)
and `down_pair` (rows of one slot pair up, the leader decodes each weight once for both rows, and its codebook step is
`mul1q`'s `rr - q x 255` form). The `pair_k3` / `guone_k3` texts and their `b3` twins hard-code `2211840ul`.

**K4 on the current `rate_window<4>` path**, per lane per 16 x 16 tile (8 windows, positions `8 lane + 7 - m`):

| | u32 loads | window ALU | notes |
|---|---|---|---|
| tight K3 (`mul1h_k3`, and `__k3`) | 2 | 2 funnel shifts + 8 x (shift, and) | lane plan (`wa`, `wb`, `lp`) outside the loop |
| K4 today (`rate_window<4>`) | 11 | 8 x (mul, div / mod by 32 and 32-word wrap, 1 or 2 shifts, or, and) | 3 of the 8 windows straddle a word; `% 32` is cheap, `% 24` (K3) would not be |
| K2 today | 11.5 avg (8 / 15 by lane parity) | as above | |
| K4 with the word cache (A) | 2 | 1 funnel (`v1`) + 8 x (shift, and) | `E = 32 (lane + 1)` is word aligned: `r = 0`, no funnel for `v0` |

Everything after the window (the `dig_cb` codebook multiply-add, the `half` fma, the 8 `xp` loads and 8 float fmas,
the `red[16][32]` reduction) is identical in every variant. The code bytes a K4 tile streams are 4/3 of a K3 tile's;
the memory-bound floor for a K4 down is 4/3 of the K3 down's, so the word cache is what lets K4 approach that floor.

## Prefill: the `_rate_m128` GEMMs against the tuned texts

`DigX` (`src/exl3_quant.zig`) launches `gemm_gu128` / `gemm_dn128` only for tight K3 operands; otherwise
`rate_gu[K0 - 2][K1 - 2]` / `rate_dn[K - 2]`, the `_xmul1h_rate_m128` texts at their `__k3_k4`-style variants. The
rate GEMMs are the tuned `_xmul1hk3_m128` texts with `WPT = 8 K` and `row_words = code*_strides[0]`, and the B
loader's job is `rate_dec_job<K, H>`: at K == 3 it is `dig_dec_job<half, H>` (the tuned job; the `(3 c7 + 23 + t) % 24`
word indices are computed per call rather than hoisted as `o0..o3`, a loop-invariant the compiler should hoist), so
**a [3, 3, 4] layer's gate operand decodes exactly as in a K3 pack; its up operand (K4) pays the rate path**:

| job (16 weights of one tile column) | u32 loads | threadgroup stores | window extraction |
|---|---|---|---|
| K3 `dig_dec_job` | 4 | 2 x `uint4` | 8 constant-shift pairs (two windows per 19-bit read) |
| K4 today (`rate_window<4>` x 16) | 28 (H 0) / 16 (H 1) | 16 scalar `half` | 16 gathers with `/ 32`, `% 32` and the circular wrap |
| K4 with the word cache (A) | 5 (H 0) / 4 (H 1) | 2 x `uint4` | 8 `rate_pair<4, Q>` with compile-time shifts; H 1 windows are word-aligned (`used >= 20`), no second word |

Not lost for Pollard: `m128lut` (the gate|up GEMM with the codebook through a threadgroup table) is registered but
not launched by the route, for K3 packs either. Lost for every K4 down layer: the fused down GEMM
`_xmul1hk3_m128w1` (`routeFusedDown`, the `prefill_fused_down` override): `downFused` falls back to `downChain` for
a down layout that is not tight K3, so the wave runs the down GEMM and `rot_widen1` as two launches with `z`
[rows, 5120] f32 round-tripping through device memory (5 launches per wave instead of 4). `routeExpertBf16` is
unaffected (it swaps `widen1_e`, which the chain still runs).

## Which Spark candidates apply

- `opt/decode-wordcache` rewrote the four `_rate_` decode texts too (`gemv_rate_loop<K, TILES, COLS, UNROLL>` with
  `gemv_lane_windows<K>`): the diff against `feat/exl3-mixed-experts` touches `dsv41_exl3_{,b3_}mul1h_rate_{2304,5120}`
  and appends the lane-window block to `header_mul1h_rate.metal`. `gemv_lane_windows<4>` is already the K4-aligned
  two-word read, so candidate B (a K4-specialised decode window path) has nothing left to take: skipped.
- `opt/prefill-wordcache` rewrote `rate_dec_job` in place (`rate_pair<K, Q>`, `W[K + 1]` loaded once, two `uint4`
  stores); the `_rate_m128` texts call it unchanged. Both branches' header edits compose (one in place at
  `rate_dec_job`, one appended).
- The rate texts and header are byte-identical between `feat/exl3-integer-rates` and `feat/exl3-mixed-experts`, so the
  hunks apply directly; the compact kernels do not exist on this base and their hunks are dropped.

## The branches (each off `opt/pollard-rates`)

| branch | what | Zig | CPU check |
|---|---|---|---|
| `opt/pollard-wordcache` (A) | the `_rate_` decode texts and `rate_dec_job` from the two Spark branches; `scripts/repin_exl3_manifest.py`; `scripts/bench_exl3_compact.py` with `--fixed` | none (manifest re-pinned) | decode: orig vs new bit-identical, K2-K5 x {2304, 5120} x {single, b3} on padded strides (odd, 64-word and the pack-widest), 36 cases, 735k outputs; windows vs `rate_window<K>` 22.3M comparisons; `rate_dec_job` K2-K5 x H x c7, 1.4M jobs per codebook mode, with clang++ and g++ |
| `opt/pollard-k4-fast` (B) | not created: see above | - | - |
| `opt/pollard-tight-forms` (C) | on A: `dsv41_exl3_{guone_k3s_2304,pair_k3s_5120}` and `_b3_` twins = the tight form texts with `base` through `code_*_strides[0]`; `Forms.strided` (`DSV41_CELL_ROUTED_FORMS=down_pair,gu_one,strided`) binds them beside the forms for K3 layouts that are not tight (banked: banks all K3, not all tight) | `exl3_quant.zig` (Gemv / Banked plans and routing), `exl3_kernels.zig` (4 tags), `exl3_selfcheck.zig` (twins), `deepseek_v41_ar.zig` (the form name), the GPU gate's forms cases; `zig fmt` / `ast-check` only | strided forms on tight, 64-word and odd strides vs the tight forms and vs stock `mul1h_k3`, grouped ids, single and banked: bit-identical, 48 cases |
| D (prefill) | not implemented; see below | - | - |

C's device twin check runs the strided texts at the 48-short stride (so the stock text reads the same arrays); the
padded stride is the CPU check's. The GPU gate (`exl3_kernel_ops_gate.zig`, "dsv41 kernels ops gpu") adds the
`{gu_one, down_pair, strided}` case over its padded third bank, which exercises both strided routes on the device.

### D: what the rate GEMMs would need to match the tuned K3 route

- The `m128lut` table codebook is not routed for K3 either; nothing to port.
- The fused down GEMM is the real gap for K4 down layers. A `dsv41_prefill_dig_gemm_2304x5120_xmul1h_rate_m128w1`
  text is mechanical: the `m128w1` text with `WPT = 8 K`, `row_words = code0_strides[0]`, `cd` through `row_words / 2`
  and `rate_dec_job<K, H>` jobs (exactly the three edits that turn `_xmul1hk3_m128` into `_xmul1h_rate_m128`), on
  `header_mul1h_rate.metal`, template `K` with `__k3` / `__k4` variants. The route side is not: `DigX.rate_dn_w1[3]`,
  `downFused` taking it for non-tight layouts (and `tiles_w` / the 40-tile table are unchanged), manifest entries with
  the `code0_words` var and `rout`, and `exl3_selfcheck.fusedOf` generalised so each variant's `fused` reference is the
  `_rate_m128` variant at the same K followed by `rot_widen1`. Four files plus the manifest; not done here.
- `rate_dec_job<3, H>` could take the hoisted `o0..o3` like the tuned text; only worth it if the compiler does not
  hoist them (check the Metal assembly, or time `__k3` against `_xmul1hk3_m128` with `--fixed`: the `K3/K3 tight`
  prefill rows do exactly that).

## Testing on the Mac

Tier 1 (`scripts/repin_exl3_manifest.py --check`, `scripts/compile_kernels_offline.py`, the pre-ship gate), tier 2
(`--fixed` prints `exact` / `DIFF` per tree; the K2-K4 GPU gate with the sushi fixtures and `DSV41_PUBLIC_DOWN_FIXTURE`)
and tier 3 as in `docs/exl3-compact-bench.md`, with:

```sh
python3 scripts/bench_exl3_compact.py --fixed feat/exl3-integer-rates opt/pollard-wordcache opt/pollard-tight-forms \
    --rows 1,8,48 --json bench-pollard.json
```

How to read it: the `K4 tight` decode rows against the `K3 tight ref` rows scaled by 4/3 (bytes) say whether the K4
down is at its memory floor after A; the `K3 padded 64` rows against `ref mul1h_k3` say the stride costs nothing
(expected within noise); the `ref guone_k3 (gate+up)` and `guone_k3s (gate+up)` rows are two rate launches' time
against the one-launch forms, tight and padded; `ref pair_k3 (sorted ids)` and `pair_k3s` show down_pair's gain at
rows 8 / 48 (grouped ids). For prefill, `K3/K4 tight` against `K3/K3 tight` is the up operand's K4 cost; `dn K4 tight`
against `dn K3 tight` and its `m128w1 (+widen1)` row is what D would recover.

Not verified here: Metal compilation of every text (tier 1 first), register pressure of `gemv_rate_loop` and
`rate_dec_job` at K4 (the compact handoff's caveats apply), and every Zig change on C (`zig fmt` and `ast-check`
pass with 0.17.0; nothing was built against the host).
