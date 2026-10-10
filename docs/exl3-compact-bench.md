# Testing the Spark 2.77 (compact EXL3) kernel candidates

A handoff for whoever has the Apple Silicon machine. It says why the compact path is slow, which branches try to
fix it, and how to find out which ones actually do, in the order that wastes the least GPU time. Nothing below was
timed yet: every candidate was built on Linux without a Metal compiler or GPU, so **the first thing each branch
needs is a Metal compile**, and the second is a bit-identity check against the current kernels on the device.

## Why the Spark pack is slow

Spark 2.77 mixes K2 / K3 / K5 experts in each layer (`expert-manifest-v3.json` classes). Those banks take the
"compact" route: `dsv41_exl3_mul1h_compact_{2304,5120}` and `dsv41_exl3_b3_mul1h_compact_*` for decode,
`dsv41_prefill_dig_gemm_*_xmul1h_compact_m128` for prefill, all built on `header_mul1h_rate.metal`. Each threadgroup
reads its expert's rate `K` from the descriptor array at run time. For K == 3 the kernels keep the tuned path (two
device loads per lane per 16 x 16 trellis tile, windows cut out by 32-bit funnel shifts). For every other rate they
fall back to `rate_window<K>` per window: 8 windows x 1-2 loads each in decode (16 windows, 16 scalar 2-byte
threadgroup stores in prefill), with circular `% (8K)` index math per load. On the fixed-K3 pack that path never
runs; on Spark it runs for most experts. Two second-order costs: the K dispatch sits inside the unrolled hot loop,
so registers are allocated for the union of all four rate paths even when the expert is K3, and the 32-lane decode
threadgroups re-read the whole activation row from device memory for every 16-output tile column.

## The branches

All are on `origin` (github.com/jasontitus/mlx-stream). Every candidate branch contains `opt/exl3-compact-tools`
(= `feat/exl3-mixed-experts` + `scripts/bench_exl3_compact.py` + `scripts/repin_exl3_manifest.py`) and has its
kernel manifest re-pinned, so it builds like any other tree. Each isolates one idea so the benchmark can tell which
effects are real; combining winners is a separate step (below).

| branch | phase | what it changes | needs Zig changes | status |
|---|---|---|---|---|
| `opt/exl3-compact-tools` | - | baseline: current kernels + the two scripts | no | pushed |
| `opt/prefill-wordcache` | prefill | `rate_dec_job` for K != 3 loads its K (or K+1) words once, extracts with compile-time shifts, stores two `uint4` instead of 16 halves. The fixed-rate `_rate_m128` kernels share the helper and gain too. | no | pushed; CPU harness bit-identical (K2-K5, both halves, all columns, ~350k jobs per K) |
| `opt/prefill-prefetch` | prefill | on top of wordcache: each lane loads stage s+1's code words before stage s's matmul (+3..6 live registers). Also restructures the fixed-rate `_rate_m128` GEMMs. | no | pushed; CPU harness bit-identical at the Bs level and the 256-thread stage loop's read schedule (K2-K5, both shapes, 3 row counts) |
| `opt/prefill-per-k` | prefill | compile-time-K GEMM variants; the wave is split by rate and each variant launched once | yes | agent running |
| `opt/prefill-staged-words` | prefill | the stage's 16 tiles are copied to threadgroup memory with coalesced `uint4` loads (one per thread), decoded from there; +2.5 KB threadgroup memory, one more barrier per stage. Its author expects it to lose or tie (the scalar device loads already hit L1; K2/K4 decode reads are 2-/4-way bank-conflicted), built so the benchmark decides. | no | pushed; CPU harness bit-identical (256-thread stage emulation, K2-K5, ASan/UBSan clean) |
| `opt/decode-hoist` | decode | K dispatch hoisted out of the tk loop only; K2/K4/K5 windows still read one at a time. Isolates the dispatch / register effect. | no | pushed; CPU harness bit-identical (20 kernel cases, 1.07M outputs) |
| `opt/decode-wordcache` | decode | the hoist plus a per-lane word cache: K2/K4/K5 lanes load 2 words per tile (3 for K5) and funnel-shift them, like the K3 path. Also covers the fixed-rate `_rate_` decode kernels. | no | pushed; same harness, bit-identical |
| `opt/decode-per-k` | decode | four `compactk` kernels (template K = 2/3/4/5), launched as a chain: each covers every row, computes rows at its K and copies the previous launch's output for the rest (+3 launches, +3 output reads/writes per stage, about 3 % of the code bytes). Enable with `DSV41_CELL_ROUTED_FORMS=per_k` (combinable with `down_pair,gu_one`; a no-op on fixed-K3 storage). | yes | pushed; CPU harness: 4-launch chain bit-identical to the compact launch (stock + banked, both shapes, rows 1/5/8, sentinel rows); Zig semantic analysis passes cross-target (`zig test --test-no-exec -target aarch64-macos`); new Trace test not executed (needs macOS link) |
| `opt/decode-staged-x` | decode | 4 simdgroups per threadgroup share the activation row staged in threadgroup memory (20 KB for the 2304-out kernel, 9 KB for 5120-out); K hoisted. Removes the 8 scattered activation loads per lane per tile; costs a serialized staging prologue and, at 20 KB, only one 2304-out group per core if the budget is 32 KB. | no (manifest `threadgroup` 32 -> 128) | pushed; CPU harness bit-identical (all four kernels, 5 seeds, rows 1-9, sentinel rows) |
| `opt/decode-staged-x8` | decode | the same at 8 simdgroups per threadgroup (`threadgroup` 256): twice the resident simdgroups per KB of staged memory. Its author's pick for the 2304-out kernel. | no | pushed; same harness passes at NSG 8 |

The two `per-k` branches add kernels and a construction-time switch (`DSV41_CELL_ROUTED_FORMS=per_k` for decode; the
prefill one names its own in its commit message). `opt/decode-per-k` also extends `scripts/bench_exl3_compact.py`: with
that branch checked out, the bench adds a `<tree> per_k` row that times the 4-launch chain, so the per-K decode idea
can be judged in tier 3 after all; run the bench from that branch's script. Everything else is a drop-in text
change behind the same kernel names and manifest `grid` / `threadgroup` rules.

## Prerequisites

- Apple Silicon Mac with the mlx-serve host checkout the plugin builds against (`HOST_PIN` in this repo names the
  commit), Zig per `build.zig.zon`, Xcode command line tools (`xcrun -sdk macosx metal`).
- Python 3 with `mlx` (the same MLX the host links, for the microbench) and `numpy`.
- A checked Spark streaming repack (v3 bank) for the end-to-end tier, and the K3 3.0-bpw pack as the reference.
- Nothing else on the GPU while timing. Run each timing three times and use the median; treat differences under
  about 5 % as noise.

## Tier 1: compile gate (minutes, no GPU time)

For every branch:

```sh
git checkout <branch>
python3 scripts/repin_exl3_manifest.py --check                     # REPIN OK: texts match the manifest
python3 scripts/compile_kernels_offline.py --mlx-include <mlx-stage>/include
```

`compile_kernels_offline.py` assembles every kernel instantiation the way `mx.fast.metal_kernel` does and runs the
Metal compiler on the CPU. A branch that prints `KERNELS_OFFLINE_COMPILE_ERROR` is fixed or dropped here. The
things most likely to trip: `if constexpr` on `constexpr uint` locals inside templates (used by
`opt/prefill-wordcache`), arrays with an unread element (`uint W[K + 1]` with `W[0]` unused at H = 1), template
arguments like `rate_pair<K, 4 * H + 26>`, and the staged-x branch's threadgroup memory (20 KB of floats plus
`red[4][16][32]`; the limit is 32 KB).

Then the plugin's own construction checks, which compile every kernel on the device and run the manifest's
self-checks (compile, row invariance, plans):

```sh
zig build test-build -Dmlx-serve=../mlx-serve -Dtest-filter="dsv41 pre-ship gate:"
DSV41_SELFCHECK_DEVICE=1 ./zig-out/bin/<test binary>
```

## Tier 2: numerical identity on the device

Every candidate claims bit-identical output to the current kernels (same fma order, same reduction). Two
independent checks:

1. **Microbench identity.** `scripts/bench_exl3_compact.py` runs each kernel on every tree with the same random
   codes and prints `exact` or `DIFF max <abs>` against the first tree. Any `DIFF` fails the branch (exit code 1).
2. **The compact GPU gate** against the PonyExl3 CPU reference (README, "Quant correctness checks"):

   ```sh
   zig build test-build -Dmlx-serve=../mlx-serve -Dtest-filter="exl3 mixed"
   DSV41_KERNELS_GPU=1 SUSHI_EXL3_K2_FIXTURE=... SUSHI_EXL3_K3_FIXTURE=... SUSHI_EXL3_K4_FIXTURE=... \
     SUSHI_EXL3_K5_FIXTURE=... DSV41_SPARK_K5_REFERENCE=... ./zig-out/bin/<test binary>
   ```

   This covers K2-K5 mixtures, all three banks, carried BF16 waves across 128-row boundaries, poisoned missing
   slots and a real Spark K5 expert. Missing fixtures fail rather than skip, so get the assets first.

The `per-k` branches also need the plugin test suite (`zig build test -Dmlx-serve=../mlx-serve`) because they
change Zig routing.

## Tier 3: kernel microbench (one GPU session, ~10 minutes)

All non-per-k branches in one invocation so they share codes, clocks and thermal state:

```sh
python3 scripts/bench_exl3_compact.py \
    opt/exl3-compact-tools \
    opt/prefill-wordcache opt/prefill-prefetch opt/prefill-staged-words \
    opt/decode-hoist opt/decode-wordcache opt/decode-staged-x opt/decode-staged-x8 \
    --mix-from <spark-repack>/expert-manifest-v3.json \
    --rows 1,8,48 --json bench-$(date +%Y%m%d-%H%M).json
```

Without a repack at hand, `--mix 2:0.45,3:0.40,5:0.15` approximates Spark's class shares (an assumption; take the
real shares from the pack's manifest when possible). `--cap 96` experts per arena (default) exceeds the system
cache so codes stream from DRAM as they do in serving.

The script builds every kernel from the branch's own texts and manifest (`git show <ref>:src/kernels/exl3/...`),
so geometry changes like staged-x's are picked up. One row per (case, tree):

```
case                                         tree                           time        stream   speed  output
decode mul1h_compact_2304 K5 rows=1          opt/exl3-compact-tools        123.4 us    35.9 GB/s  1.00x  first
decode mul1h_compact_2304 K5 rows=1          opt/decode-wordcache           61.2 us    72.4 GB/s  2.02x  exact
decode mul1h_compact_2304 K3 rows=1          ref mul1h_k3_2304              40.1 us   110.5 GB/s  3.08x  exact
```

How to read it:

- **time / speed** - microseconds per launch and the speedup over the first tree. Decode at `rows=1` is the
  latency-bound case that sets tok/s; `rows=48` is the batched draft/verify shape. Prefill cases time one wave of
  `--experts` experts x `--prefill-rows` rows each.
- **stream** - the code bytes the launch had to read per second. The decode GEMVs are memory-bound; a kernel at the
  SoC's DRAM bandwidth cannot go faster whatever the ALU does.
- **ref** rows - the tight-K3 kernel on the very same K3 codes (only in the `K3` cases). This is the ceiling the
  compact path can reach for a K3 expert. If a candidate's `K3` row is within noise of `ref`, the hoist worked and
  the only remaining compact cost is the non-K3 rates' extraction; if its `K2`/`K5` rows are also near the K3
  `ref` scaled by bytes (K2 reads 2/3, K5 5/3 of a K3 expert), the kernels are done and any end-to-end gap left
  on Spark is in reads or residency, not kernels.
- **output** - `exact` or `DIFF`. A `DIFF` is a bug in the candidate even if it is fast.

Keep the branch with the best `rows=1` decode time and the best prefill time for the Spark mix, provided it is
`exact`. `opt/decode-wordcache` contains `opt/decode-hoist`'s change, so their difference is the word cache alone; if `opt/decode-staged-x` and `opt/decode-wordcache` both beat the baseline, they are independent effects
(activation traffic vs. code extraction) and worth composing; `opt/prefill-prefetch` already contains
`opt/prefill-wordcache`, so compare the two directly.

## Tier 4: the per-k branches and end to end

The `per-k` idea pays in register allocation (each kernel compiled for one rate) and costs extra launches per
layer plus the wave split; a single-kernel microbench cannot see that trade. Judge them, and confirm the tier-3
winners, with the served cell on the Spark repack (README, "Memory and context length"), against the same build
on the K3 pack:

```sh
zig build cell -Dmlx-serve=../mlx-serve                       # the timed cell harness
# baseline, then each candidate; the per-k switch per the branch's report
DSV41_BANK=<spark-repack> ./zig-out/bin/<cell> ...            # 16K prompt: prefill tok/s, TTFT, decode tok/s
```

Report the 16K headline cell (prefill tok/s, TTFT, decode tok/s, prompt / decode rows) per branch next to the K3
pack's 668 / 24.5 s / 37.6 from the README ladder. Decode tok/s is what the Spark users feel; prefill tok/s is where
the GEMM branches show. If the kernels are near the K3 ceiling in tier 3 but the cell still trails the K3 pack by
more than the byte ratio explains, look at the read pool and admission (`src/expert_stream.zig`,
`src/expert_admission.zig`) next, not the kernels.

## Composing winners

Each candidate is a separate branch off the same base, so a combination is a merge:

```sh
git checkout -b opt/compact-best opt/prefill-<winner>
git merge opt/decode-<winner>            # header_mul1h_rate.metal: both append; resolve, keep both blocks
python3 scripts/repin_exl3_manifest.py   # re-pins texts and manifest_sha256 after the merge
```

Re-run tier 1 and tier 2 on the merged branch before timing it; the manifest must be re-pinned after any text
change or `Registry.init` refuses the load by name.

## What is already known

- Every candidate's decode arithmetic was checked bit-identical to the current kernels in a CPU emulation of the
  kernel body (32 lanes per threadgroup, Metal's half fma emulated with a single rounding). Those harnesses are in
  the Linux session's scratchpad, not in the repo; the device checks above supersede them.
- Not known: whether the Metal compiler accepts each text, register pressure (K5 keeps 6 code words live in
  prefill; the decode word cache keeps 3), occupancy under staged-x's 20 KB threadgroup footprint (the two NSG branches bracket it), and every
  number in this document's tables. Expect at least one branch to need a compile fix.
