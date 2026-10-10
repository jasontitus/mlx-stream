#!/usr/bin/env python3
"""bench_exl3_compact.py: time the compact (per-expert K2-K5) EXL3 decode GEMVs and prefill GEMMs on the GPU, per rate
and against the tight-K3 kernels on the same codes, for one or more kernel trees; every tree's outputs are compared
bit for bit with the first's.

Each tree is a git ref of this repository (`HEAD`, `main`, `opt/decode-wordcache`, ...) or a directory holding the
kernels/exl3 texts and their manifest. The kernels are built the way the plugin builds them (MLX
fast.metal_kernel, the manifest's inputs, header, geometry and row contiguity), on random codes: a compact arena per
case whose experts are uniform K2 / K3 / K4 / K5 or a mixture (`--mix`, or the class shares of a repacked bank's
expert-manifest-v3.json with `--mix-from`). Every timed launch reads a different expert set, so the codes stream
from DRAM rather than the system cache.

  python3 scripts/bench_exl3_compact.py main opt/decode-wordcache opt/prefill-wordcache
  python3 scripts/bench_exl3_compact.py --only decode --rows 1,8 --mix-from /models/spark/expert-manifest-v3.json main HEAD

Needs Apple silicon, `pip install mlx numpy`. Prints one row per (case, tree): microseconds per launch, the code
stream's effective GB/s, the speedup over the first tree and whether the output is bit-identical to the first tree's
(`ref`: the tight-K3 kernel, which reads the same K3 codes through its fixed stride). A tree whose manifest carries the
compactk texts (opt/decode-per-k) also times their per-rate chain as `<tree> per_k`, four launches composed as the
plugin's ROUTED_FORMS=per_k route composes them, checked against the compact launch's words."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
KREL = "src/kernels/exl3"
TILE_SHORTS = 16  # x K: one 16 x 16 trellis tile
EXPERT_TILES = 320 * 144  # hidden 5120 / 16 x inter 2304 / 16, either orientation


def expert_shorts(k):
    return EXPERT_TILES * TILE_SHORTS * k


class Tree:
    """One kernel tree: its manifest and texts, from a directory or a git ref of this repository."""

    def __init__(self, spec):
        self.label = spec
        if os.path.isdir(spec):
            d = spec if os.path.exists(os.path.join(spec, "manifest.json")) else os.path.join(spec, KREL)
            self.read = lambda f: open(os.path.join(d, f)).read()
        else:
            self.read = lambda f: subprocess.run(["git", "-C", ROOT, "show", f"{spec}:{KREL}/{f}"], check=True,
                                                 capture_output=True, text=True).stdout
        self.manifest = json.loads(self.read("manifest.json"))
        self.specs = {k["name"]: k for k in self.manifest["kernels"]}
        self.headers = {h["id"]: h["file"] for h in self.manifest["headers"]}
        self.addenda = {a["id"]: a["file"] for a in self.manifest.get("port_addenda", [])}
        self.built = {}

    def kernel(self, name):
        import mlx.core as mx
        if name not in self.built:
            k = self.specs[name]
            header = ""
            if k.get("header"):
                hid = k["header"]["id"]
                header = self.read(self.headers[hid]) + (self.read(self.addenda[hid]) if hid in self.addenda else "")
            tag = "".join(c if c.isalnum() else "_" for c in self.label)
            self.built[name] = mx.fast.metal_kernel(
                name=f"{name}_{tag}", input_names=[i["name"] for i in k["inputs"]],
                output_names=[o["name"] for o in k["outputs"]], source=self.read(k["source"]["file"]), header=header,
                ensure_row_contiguous=k.get("ensure_row_contiguous", True), atomic_outputs=False)
        return self.specs[name], self.built[name]


def dim(d, vs):
    return d.get("m", 1) * (vs[d["v"]] if "v" in d else 1)


def launch(tree, name, args, vs):
    """args: the kernel's inputs by name (mx arrays); vs: the manifest vars of this launch. A templated entry (the
    compactk texts' K / ACC) is instantiated at its recorded ints."""
    import mlx.core as mx
    k, fn = tree.kernel(name)
    template = [(t["name"], t["int"]) for t in k.get("template", []) if "int" in t]
    return fn(inputs=[args[i["name"]] for i in k["inputs"]], template=template or None, grid=tuple(dim(d, vs) for d in k["grid"]),
              threadgroup=tuple(k["threadgroup"]),
              output_shapes=[tuple(dim(d, vs) for d in o["shape"]) for o in k["outputs"]],
              output_dtypes=[getattr(mx, {"float32": "float32", "float16": "float16"}[o["dtype"]]) for o in k["outputs"]])


class Arena:
    """A compact code arena: experts' codes concatenated at their own rate, descriptors (offset in int16, K)."""

    def __init__(self, ks, rng):
        import mlx.core as mx
        self.ks = list(ks)
        offs = np.concatenate([[0], np.cumsum([expert_shorts(k) for k in self.ks])]).astype(np.uint64)
        total = int(offs[-1])
        rows = (total + 255) // 256
        codes = rng.integers(-32768, 32768, size=rows * 256, dtype=np.int16)
        self.code = mx.array(codes.reshape(rows, 256))
        self.desc = mx.array(np.stack([offs[:-1], np.array(self.ks, dtype=np.uint64)], axis=1))
        self.uniform_k3 = all(k == 3 for k in self.ks)
        if self.uniform_k3:  # the tight kernels' [cap, 320, 144, 48] view of the same codes (K3 stride)
            self.tight = self.code.reshape(-1)[: total].reshape(len(self.ks), 320, 144, 48)
        self.bytes = [2 * expert_shorts(k) for k in self.ks]
        mx.eval(self.code, self.desc)


def wave_table(slots, rows_each, bm, tiles):
    t = np.zeros(80, dtype=np.int32)
    row0 = tg = 0
    for j in range(16):
        if j >= len(slots):
            t[48 + j] = np.iinfo(np.int32).max
            continue
        t[j], t[16 + j], t[32 + j], t[48 + j] = slots[j], row0, rows_each, tg
        row0 += rows_each
        tg += (rows_each + bm - 1) // bm * tiles
    t[64], t[65] = len(slots), tg
    return t, tg


def timed(make, iters, warm=3):
    """make(i) -> outputs of launch i (a list); returns (seconds per launch, outputs of launch 0)."""
    import mlx.core as mx
    first = make(0)
    mx.eval(first)
    for i in range(warm):
        mx.eval(make(i + 1))
    outs = [make(i) for i in range(iters)]
    t0 = time.perf_counter()
    mx.eval(outs)
    return (time.perf_counter() - t0) / iters, [np.array(o) for o in first]


def k_sets(args, rng):
    sets = [(f"K{k}", [k] * args.cap) for k in (2, 3, 4, 5) if str(k) in args.uniform.split(",")]
    if args.mix_from:
        m = json.load(open(args.mix_from))
        cls = {c["class"]: c["projection_K"] for c in m["classes"]}
        triples = [cls[r["class"]] for r in m["records"]]
        pick = rng.integers(0, len(triples), size=args.cap)
        sets.append(("mix(gate)", [triples[i][0] for i in pick]))
        sets.append(("mix(down)", [triples[i][2] for i in pick]))
    elif args.mix:
        shares = {int(k): float(v) for k, v in (p.split(":") for p in args.mix.split(","))}
        ks, ps = list(shares), np.array(list(shares.values()))
        sets.append((f"mix({args.mix})", list(rng.choice(ks, size=args.cap, p=ps / ps.sum()))))
    return sets


def report(rows, case, label, sec, nbytes, base, exact):
    rows.append((case, label, sec * 1e6, nbytes / sec / 1e9, base / sec if base else 1.0, exact))
    print(f"{case:<44} {label:<28} {sec * 1e6:9.1f} us {nbytes / sec / 1e9:7.1f} GB/s {base / sec if base else 1.0:6.2f}x  {exact}",
          flush=True)


def same(a, b):
    return "exact" if all(np.array_equal(x, y) for x, y in zip(a, b)) else \
        "DIFF max %.3g" % max(float(np.max(np.abs(x.astype(np.float64) - y))) for x, y in zip(a, b))


def bench_decode(trees, args, rng, rows_out):
    import mlx.core as mx
    for set_name, ks in k_sets(args, rng):
        arena = Arena(ks, rng)
        for shape, xd, ref in (("2304", 5120, "dsv41_exl3_mul1h_k3_2304"), ("5120", 2304, "dsv41_exl3_mul1h_k3_5120")):
            for rows in args.rows:
                xh = mx.array(rng.standard_normal((rows, xd)).astype(np.float32))
                picks = [mx.array(rng.choice(args.cap, size=rows, replace=False).astype(np.uint32)) for _ in range(args.iters + 4)]
                nbytes = rows * float(np.mean(arena.bytes))
                vs = {"rows": rows, "cap": args.cap}
                for banked in (False, True):
                    name = f"dsv41_exl3_{'b3_' if banked else ''}mul1h_compact_{shape}"
                    case = f"decode {name[11:]} {set_name} rows={rows}"
                    base = first = None
                    def inputs(i):
                        if banked:  # one arena behind all three banks; bank b's ids carry b << 24
                            ids = picks[i] | mx.array((np.arange(rows) % 3 << 24).astype(np.uint32))
                            return {"xh": xh, "ids": ids, "code0": arena.code, "code1": arena.code, "code2": arena.code,
                                    "desc0": arena.desc, "desc1": arena.desc, "desc2": arena.desc}
                        return {"xh": xh, "ids": picks[i], "code": arena.code, "desc": arena.desc}
                    for tree in trees:
                        def make(i, tree=tree):
                            return launch(tree, name, inputs(i), vs)
                        sec, out = timed(make, args.iters)
                        exact = "first" if first is None else same(first, out)
                        first = first if first is not None else out
                        base = base or sec
                        report(rows_out, case, tree.label, sec, nbytes, base, exact)
                        # The per-rate chain where the tree has it (ROUTED_FORMS per_k): the compactk text at K 2 .. 5 in
                        # turn, each carrying the other rows' words from the previous launch (the head's carry slot: xh).
                        pk = name.replace("_compact_", "_compactk_")
                        if pk in tree.specs:
                            def make_pk(i, tree=tree, pk=pk):
                                a, carry = inputs(i), xh
                                for K in (2, 3, 4, 5):
                                    carry = launch(tree, pk if K == 2 else f"{pk}__k{K}", dict(a, carry=carry), vs)[0]
                                return [carry]
                            sec, out = timed(make_pk, args.iters)
                            report(rows_out, case, f"{tree.label} per_k", sec, nbytes, base, same(first, out))
                    if arena.uniform_k3 and not banked:
                        zeros = {n: mx.zeros((s,), dtype=mx.int32) for n, s in (("lane_a", 32), ("lane_b", 32), ("lane_p", 32), ("pos_r", 256), ("pos_c", 256))}
                        cb = mx.array(np.array([3417055213, 0, 2415890431, 996162400], dtype=np.uint32))

                        def make_ref(i):
                            return launch(trees[0], ref, dict(zeros, xh=xh, ids=picks[i], code=arena.tight, cb=cb), vs)
                        sec, out = timed(make_ref, args.iters)
                        report(rows_out, case, f"ref {ref[11:]}", sec, nbytes, base, same(first, out))
        del arena


def bench_prefill(trees, args, rng, rows_out):
    import mlx.core as mx
    E, R = args.experts, args.prefill_rows
    for set_name, ks in k_sets(args, rng):
        arena = Arena(ks, rng)
        waves = [rng.choice(args.cap, size=E, replace=False) for _ in range(args.iters + 4)]
        for name, ref, kd, tiles, gu in (
                ("dsv41_prefill_dig_gemm_5120x2304_gu_xmul1h_compact_m128", "dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128", 5120, 72, True),
                ("dsv41_prefill_dig_gemm_2304x5120_xmul1h_compact_m128", "dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128", 2304, 80, False)):
            x = mx.array((rng.standard_normal((E * R, 1, kd)) * 0.5).astype(np.float16))
            tbls = []
            for w in waves:
                t, tgs = wave_table(list(w), R, 128, tiles)
                tbls.append(mx.array(t))
            vs = {"rows": E * R, "cap": args.cap, "tgs": tgs, "experts": E}
            nbytes = E * float(np.mean(arena.bytes)) * (2 if gu else 1)
            case = f"prefill {'gu' if gu else 'dn'} {set_name} E={E} rows/E={R}"
            base = first = None
            for tree in trees:
                def make(i, tree=tree):
                    a = {"tbl": tbls[i]}
                    if gu:
                        a.update(x0=x, x1=x, code0=arena.code, code1=arena.code, desc0=arena.desc, desc1=arena.desc)
                    else:
                        a.update(x=x, code0=arena.code, desc=arena.desc)
                    return launch(tree, name, a, vs)
                sec, out = timed(make, args.iters)
                exact = "first" if first is None else same(first, out)
                first = first if first is not None else out
                base = base or sec
                report(rows_out, case, tree.label, sec, nbytes, base, exact)
            if arena.uniform_k3:
                def make_ref(i):
                    a = {"tbl": tbls[i]}
                    if gu:
                        a.update(x0=x, x1=x, code0=arena.tight, code1=arena.tight)
                    else:
                        a.update(x=x, code0=arena.tight)
                    return launch(trees[0], ref, a, vs)
                sec, out = timed(make_ref, args.iters)
                report(rows_out, case, "ref tight K3 m128", sec, nbytes, base, same(first, out))
        del arena


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trees", nargs="+", help="git refs or kernel directories; the first is the baseline")
    ap.add_argument("--only", choices=("decode", "prefill"), default=None)
    ap.add_argument("--cap", type=int, default=96, help="experts per arena (more than the system cache holds)")
    ap.add_argument("--rows", default="1,8", help="decode rows (routed assignments) per launch")
    ap.add_argument("--experts", type=int, default=8, help="prefill: experts per wave")
    ap.add_argument("--prefill-rows", type=int, default=256, help="prefill: rows routed to each expert")
    ap.add_argument("--uniform", default="2,3,4,5", help="uniform-rate arenas to time (comma list, '' for none)")
    ap.add_argument("--mix", default="2:0.45,3:0.40,5:0.15", help="K:share list for the mixed arena")
    ap.add_argument("--mix-from", default=None, help="a repacked bank's expert-manifest-v3.json: its class shares")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the rows here")
    args = ap.parse_args()
    args.rows = [int(r) for r in args.rows.split(",") if r]
    import mlx.core as mx
    if not mx.metal.is_available():
        print("bench_exl3_compact: needs a Metal device")
        return 2
    trees = [Tree(t) for t in args.trees]
    rng = np.random.default_rng(args.seed)
    rows_out = []
    print(f"{'case':<44} {'tree':<28} {'time':>12} {'stream':>12} {'speed':>7}  output")
    if args.only in (None, "decode"):
        bench_decode(trees, args, rng, rows_out)
    if args.only in (None, "prefill"):
        bench_prefill(trees, args, rng, rows_out)
    if args.json:
        json.dump([dict(zip(("case", "tree", "us", "gbps", "speedup", "output"), r)) for r in rows_out], open(args.json, "w"), indent=1)
    return 1 if any(r[5].startswith("DIFF") for r in rows_out) else 0


if __name__ == "__main__":
    sys.exit(main())
