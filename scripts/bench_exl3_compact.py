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

`--fixed` times the fixed-rate (`_rate_`) kernels instead: the Pollard path, one logical rate per launch (K2 / K3 / K4
through the manifest's `__k3` / `__k4` template variants), each code bank at its tight stride (16 K shorts per tile
row) and at the 64-short padded stride a mixed bank's transient rows take, plus the tight-K3 kernels on the same K3
codes (`ref`: `mul1h_k3`, the gate|up / down 128-row GEMMs and the fused down GEMM). A tree without the compact
kernels (feat/exl3-integer-rates and its candidates) is benched with `--fixed` only.

  python3 scripts/bench_exl3_compact.py --fixed feat/exl3-integer-rates opt/pollard-wordcache

Needs Apple silicon, `pip install mlx numpy`. Prints one row per (case, tree): microseconds per launch, the code
stream's effective GB/s, the speedup over the first tree and whether the output is bit-identical to the first tree's
(`ref`: the tight-K3 kernel, which reads the same K3 codes through its fixed stride)."""
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
    """args: the kernel's inputs by name (mx arrays); vs: the manifest vars of this launch. Static inputs (the codebook
    words, the unused lane tables) the caller leaves out are generated from their manifest domain."""
    import mlx.core as mx
    k, fn = tree.kernel(name)
    ins = []
    for i in k["inputs"]:
        if i["name"] in args:
            ins.append(args[i["name"]])
        elif i["role"] == "static":
            shape = tuple(dim(d, vs) for d in i["shape"])
            dt = getattr(mx, i["dtype"])
            ins.append(mx.array(np.array(i["domain"]["ints"], dtype=np.uint32)) if i["domain"]["kind"] == "values" else mx.zeros(shape, dtype=dt))
        else:
            raise KeyError(f"{name}: input {i['name']} not given")
    extra = {"template": [(t["name"], t["int"]) for t in k["template"]]} if k.get("template") else {}
    return fn(inputs=ins, grid=tuple(dim(d, vs) for d in k["grid"]), threadgroup=tuple(k["threadgroup"]),
              output_shapes=[tuple(dim(d, vs) for d in o["shape"]) for o in k["outputs"]],
              output_dtypes=[getattr(mx, {"float32": "float32", "float16": "float16"}[o["dtype"]]) for o in k["outputs"]], **extra)


def code_shape(tree, name, inp):
    """The int16 code input's manifest shape with a var in its last axis: (cap, n_i, n_j)."""
    k = tree.specs[name]
    s = next(i for i in k["inputs"] if i["name"] == inp)["shape"]
    return s[1]["m"], s[2]["m"]


class FixedBank:
    """A fixed-rate bank at logical rate K: code [cap, n_i, n_j, W] int16, W = 16 K (tight) or 64 (the padded stride of
    a mixed bank's transient rows); every expert's logical tiles are a contiguous prefix of its row, so `tight` (the
    [cap, n_i, n_j, 16 K] copy of the same codes) is what the tight kernels read."""

    def __init__(self, k, cap, n_i, n_j, words, rng):
        import mlx.core as mx
        self.k, self.words = k, words
        flat = rng.integers(-32768, 32768, size=(cap, n_i * n_j * words), dtype=np.int16)
        self.code = mx.array(flat.reshape(cap, n_i, n_j, words))
        logical = n_i * n_j * 16 * k
        self.tight = mx.array(np.ascontiguousarray(flat[:, :logical]).reshape(cap, n_i, n_j, 16 * k))
        self.bytes = 2 * logical
        mx.eval(self.code, self.tight)


def rate_name(base, k):
    return base if k == 2 else f"{base}__k{k}"


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
                    for tree in trees:
                        def make(i, tree=tree):
                            if banked:  # one arena behind all three banks; bank b's ids carry b << 24
                                ids = picks[i] | mx.array((np.arange(rows) % 3 << 24).astype(np.uint32))
                                a = {"xh": xh, "ids": ids, "code0": arena.code, "code1": arena.code, "code2": arena.code,
                                     "desc0": arena.desc, "desc1": arena.desc, "desc2": arena.desc}
                            else:
                                a = {"xh": xh, "ids": picks[i], "code": arena.code, "desc": arena.desc}
                            return launch(tree, name, a, vs)
                        sec, out = timed(make, args.iters)
                        exact = "first" if first is None else same(first, out)
                        first = first if first is not None else out
                        base = base or sec
                        report(rows_out, case, tree.label, sec, nbytes, base, exact)
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


def bench_fixed_decode(trees, args, rng, rows_out):
    """The `_rate_` decode GEMVs at K2 / K3 / K4 on tight and padded banks; at K3 also the tight kernel on the same codes."""
    import mlx.core as mx
    for k in args.fixed_rates:
        for shape, xd, base in (("2304", 5120, "dsv41_exl3_mul1h_rate_2304"), ("5120", 2304, "dsv41_exl3_mul1h_rate_5120")):
            n_i, n_j = code_shape(trees[0], base, "code")
            for words in sorted({16 * k, 64}):
                banks = [FixedBank(k, args.cap, n_i, n_j, words, rng) for _ in range(3)]
                stride = "tight" if words == 16 * k else f"padded {words}"
                for rows in args.rows:
                    xh = mx.array(rng.standard_normal((rows, xd)).astype(np.float32))
                    picks = [mx.array(rng.choice(args.cap, size=rows, replace=False).astype(np.uint32)) for _ in range(args.iters + 4)]
                    nbytes = rows * banks[0].bytes
                    for banked in (False, True):
                        name = rate_name(f"dsv41_exl3_{'b3_' if banked else ''}mul1h_rate_{shape}", k)
                        case = f"fixed {name[11:]} K{k} {stride} rows={rows}"
                        vs = {"rows": rows, "cap": args.cap, "code_words": words, "code0_words": words, "code1_words": words, "code2_words": words}
                        base_t = first = None
                        for tree in trees:
                            def make(i, tree=tree):
                                if banked:
                                    ids = picks[i] | mx.array((np.arange(rows) % 3 << 24).astype(np.uint32))
                                    a = {"xh": xh, "ids": ids, "code0": banks[0].code, "code1": banks[1].code, "code2": banks[2].code}
                                else:
                                    a = {"xh": xh, "ids": picks[i], "code": banks[0].code}
                                return launch(tree, name, a, vs)
                            sec, out = timed(make, args.iters)
                            exact = "first" if first is None else same(first, out)
                            first = first if first is not None else out
                            base_t = base_t or sec
                            report(rows_out, case, tree.label, sec, nbytes, base_t, exact)
                        if k == 3:
                            ref = f"dsv41_exl3_{'b3_' if banked else ''}mul1h_k3_{shape}"

                            def make_ref(i):
                                if banked:
                                    ids = picks[i] | mx.array((np.arange(rows) % 3 << 24).astype(np.uint32))
                                    a = {"xh": xh, "ids": ids, "code0": banks[0].tight, "code1": banks[1].tight, "code2": banks[2].tight}
                                else:
                                    a = {"xh": xh, "ids": picks[i], "code": banks[0].tight}
                                return launch(trees[0], ref, a, {"rows": rows, "cap": args.cap})
                            sec, out = timed(make_ref, args.iters)
                            report(rows_out, case, f"ref {ref[11:]}", sec, nbytes, base_t, same(first, out))
                            if not banked:  # the routed forms: gate|up in one launch, the pair text on grouped ids
                                guone = "dsv41_exl3_guone_k3_2304" if shape == "2304" else None
                                pair = "dsv41_exl3_pair_k3_5120" if shape == "5120" else None
                                if guone:
                                    def make_gu(i):
                                        return launch(trees[0], guone, {"xh_g": xh, "xh_u": xh, "ids": picks[i], "code_g": banks[0].tight, "code_u": banks[1].tight}, {"rows": rows, "cap": args.cap})
                                    sec, out = timed(make_gu, args.iters)
                                    report(rows_out, case, "ref guone_k3 (gate+up)", sec, 2 * nbytes, 2 * base_t, same(first, out[:1]))
                                if pair:
                                    def make_pair(i):
                                        return launch(trees[0], pair, {"xh": xh, "ids": mx.sort(picks[i]), "code": banks[0].tight}, {"rows": rows, "cap": args.cap})
                                    sec, out = timed(make_pair, args.iters)
                                    report(rows_out, case, "ref pair_k3 (sorted ids)", sec, nbytes, base_t, "-")
                            # the strided forms (opt/pollard-tight-forms): the same forms through the bank's own stride
                            for tree in trees:
                                s_gu = "dsv41_exl3_b3_guone_k3s_2304" if banked else "dsv41_exl3_guone_k3s_2304"
                                s_pair = "dsv41_exl3_b3_pair_k3s_5120" if banked else "dsv41_exl3_pair_k3s_5120"
                                if shape == "2304" and s_gu in tree.specs:
                                    def make_sgu(i, tree=tree):
                                        a = {"xh_g": xh, "xh_u": xh}
                                        if banked:
                                            a["ids"] = picks[i] | mx.array((np.arange(rows) % 3 << 24).astype(np.uint32))
                                            a.update({f"code_g{b}": banks[b].code for b in range(3)}, **{f"code_u{b}": banks[b].code for b in range(3)})
                                        else:
                                            a.update(ids=picks[i], code_g=banks[0].code, code_u=banks[1].code)
                                        return launch(tree, s_gu, a, vs)
                                    sec, out = timed(make_sgu, args.iters)
                                    report(rows_out, case, f"{tree.label} {s_gu[11:]} (gate+up)", sec, 2 * nbytes, 2 * base_t, same(first, out[:1]))
                                if shape == "5120" and s_pair in tree.specs:
                                    def make_spair(i, tree=tree):
                                        a = {"xh": xh}
                                        if banked:
                                            a["ids"] = mx.sort(picks[i]) | mx.array((np.arange(rows) % 3 << 24).astype(np.uint32))
                                            a.update({f"code{b}": banks[b].code for b in range(3)})
                                        else:
                                            a.update(ids=mx.sort(picks[i]), code=banks[0].code)
                                        return launch(tree, s_pair, a, vs)
                                    sec, out = timed(make_spair, args.iters)
                                    report(rows_out, case, f"{tree.label} {s_pair[11:]} (sorted ids)", sec, nbytes, base_t, "-")
                del banks


def bench_fixed_prefill(trees, args, rng, rows_out):
    """The `_rate_m128` GEMMs per rate pair on tight and padded banks; at K3 the tuned m128 texts (and the fused down)."""
    import mlx.core as mx
    E, R = args.experts, args.prefill_rows
    gu_base = "dsv41_prefill_dig_gemm_5120x2304_gu_xmul1h_rate_m128"
    dn_base = "dsv41_prefill_dig_gemm_2304x5120_xmul1h_rate_m128"
    pairs = sorted({(k, k) for k in args.fixed_rates} | {(3, 4), (4, 3)} & {(a, b) for a in args.fixed_rates for b in args.fixed_rates})
    waves = [rng.choice(args.cap, size=E, replace=False) for _ in range(args.iters + 4)]
    for k0, k1 in pairs:
        name = gu_base if (k0, k1) == (2, 2) else (f"{gu_base}__k{k0}" if k0 == k1 else f"{gu_base}__k{k0}_k{k1}")
        n_i, n_j = code_shape(trees[0], gu_base, "code0")
        for padded in (False, True):
            w0, w1 = (64, 64) if padded else (16 * k0, 16 * k1)
            bg, bu = FixedBank(k0, args.cap, n_i, n_j, w0, rng), FixedBank(k1, args.cap, n_i, n_j, w1, rng)
            x = mx.array((rng.standard_normal((E * R, 1, 5120)) * 0.5).astype(np.float16))
            tbls = []
            for w in waves:
                t, tgs = wave_table(list(w), R, 128, 72)
                tbls.append(mx.array(t))
            vs = {"rows": E * R, "cap": args.cap, "tgs": tgs, "experts": E, "code0_words": w0, "code1_words": w1}
            nbytes = E * (bg.bytes + bu.bytes)
            case = f"fixed prefill gu K{k0}/K{k1} {'padded 64' if padded else 'tight'} E={E} rows/E={R}"
            base_t = first = None
            for tree in trees:
                def make(i, tree=tree):
                    return launch(tree, name, {"tbl": tbls[i], "x0": x, "x1": x, "code0": bg.code, "code1": bu.code}, vs)
                sec, out = timed(make, args.iters)
                exact = "first" if first is None else same(first, out)
                first = first if first is not None else out
                base_t = base_t or sec
                report(rows_out, case, tree.label, sec, nbytes, base_t, exact)
            if (k0, k1) == (3, 3):
                for ref in ("dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128", "dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut"):
                    def make_ref(i, ref=ref):
                        return launch(trees[0], ref, {"tbl": tbls[i], "x0": x, "x1": x, "code0": bg.tight, "code1": bu.tight}, {"rows": E * R, "cap": args.cap, "tgs": tgs, "experts": E})
                    sec, out = timed(make_ref, args.iters)
                    report(rows_out, case, f"ref {ref[27:]}", sec, nbytes, base_t, same(first, out))
            del bg, bu
    for k in args.fixed_rates:
        name = rate_name(dn_base, k)
        n_i, n_j = code_shape(trees[0], dn_base, "code0")
        for words in sorted({16 * k, 64}):
            bd = FixedBank(k, args.cap, n_i, n_j, words, rng)
            x = mx.array((rng.standard_normal((E * R, 1, 2304)) * 0.5).astype(np.float16))
            tbls, tbls_w = [], []
            for w in waves:
                t, tgs = wave_table(list(w), R, 128, 80)
                tbls.append(mx.array(t))
                t, tgs_w = wave_table(list(w), R, 128, 40)
                tbls_w.append(mx.array(t))
            vs = {"rows": E * R, "cap": args.cap, "tgs": tgs, "experts": E, "code0_words": words}
            nbytes = E * bd.bytes
            case = f"fixed prefill dn K{k} {'tight' if words == 16 * k else f'padded {words}'} E={E} rows/E={R}"
            base_t = first = None
            for tree in trees:
                def make(i, tree=tree):
                    return launch(tree, name, {"tbl": tbls[i], "x": x, "code0": bd.code}, vs)
                sec, out = timed(make, args.iters)
                exact = "first" if first is None else same(first, out)
                first = first if first is not None else out
                base_t = base_t or sec
                report(rows_out, case, tree.label, sec, nbytes, base_t, exact)
            if k == 3:
                ref = "dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128"

                def make_ref(i):
                    return launch(trees[0], ref, {"tbl": tbls[i], "x": x, "code0": bd.tight}, {"rows": E * R, "cap": args.cap, "tgs": tgs, "experts": E})
                sec, out = timed(make_ref, args.iters)
                report(rows_out, case, f"ref {ref[27:]}", sec, nbytes, base_t, same(first, out))
                # the fused down GEMM (widen1 in its epilogue): the route's whole down stage in one launch; its output
                # is widen1(z), so no bit comparison here
                rout = mx.array((rng.standard_normal((args.cap, 5120)) * 0.1).astype(np.float16))
                fused = "dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1"

                def make_w1(i):
                    return launch(trees[0], fused, {"tbl": tbls_w[i], "x": x, "code0": bd.tight, "rout": rout}, {"rows": E * R, "cap": args.cap, "tgs": tgs_w, "experts": E})
                sec, out = timed(make_w1, args.iters)
                report(rows_out, case, f"ref {fused[27:]} (+widen1)", sec, nbytes, base_t, "-")
            del bd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trees", nargs="+", help="git refs or kernel directories; the first is the baseline")
    ap.add_argument("--only", choices=("decode", "prefill"), default=None)
    ap.add_argument("--fixed", action="store_true", help="time the fixed-rate `_rate_` kernels (Pollard) instead of the compact ones")
    ap.add_argument("--fixed-rates", default="2,3,4", help="--fixed: the logical rates to time")
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
    args.fixed_rates = [int(r) for r in args.fixed_rates.split(",") if r]
    import mlx.core as mx
    if not mx.metal.is_available():
        print("bench_exl3_compact: needs a Metal device")
        return 2
    trees = [Tree(t) for t in args.trees]
    if not args.fixed and any("dsv41_exl3_mul1h_compact_2304" not in t.specs for t in trees):
        print("bench_exl3_compact: a tree has no compact kernels; use --fixed")
        return 2
    rng = np.random.default_rng(args.seed)
    rows_out = []
    print(f"{'case':<44} {'tree':<28} {'time':>12} {'stream':>12} {'speed':>7}  output")
    if args.only in (None, "decode"):
        (bench_fixed_decode if args.fixed else bench_decode)(trees, args, rng, rows_out)
    if args.only in (None, "prefill"):
        (bench_fixed_prefill if args.fixed else bench_prefill)(trees, args, rng, rows_out)
    if args.json:
        json.dump([dict(zip(("case", "tree", "us", "gbps", "speedup", "output"), r)) for r in rows_out], open(args.json, "w"), indent=1)
    return 1 if any(r[5].startswith("DIFF") for r in rows_out) else 0


if __name__ == "__main__":
    sys.exit(main())
