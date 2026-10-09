#!/usr/bin/env python3
"""repin_exl3_manifest.py: re-pin kernels/exl3/manifest.json after kernel or header texts change on disk.

For every header, port addendum and kernel source it rewrites `bytes` / `sha256` from the file, refreshes each
kernel's embedded header record, and recomputes the lane pins of kernels whose text or header changed (keeping a
pin's recorded prefix length). Only pins whose symbol starts with `mlx-stream` are recomputed: a pin recorded from
another lane's install record is refused instead, since its text must not change here.

The superseded manifest is appended to `predecessors` with the kernels and headers that are byte-identical in the new
one (the registry accepts its fixtures for those), and every older predecessor drops the kernels and headers that
changed. The manifest is written in its published form (sorted keys, no spaces, trailing newline) and
`manifest_sha256` in src/exl3_kernels.zig is rewritten to the new digest.

  scripts/repin_exl3_manifest.py [--check]
--check changes nothing and exits 1 when the manifest is stale."""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
KDIR = os.path.join(ROOT, "src", "kernels", "exl3")
MANIFEST = os.path.join(KDIR, "manifest.json")
REGISTRY = os.path.join(ROOT, "src", "exl3_kernels.zig")
PIN_RE = re.compile(r'(pub const manifest_sha256 = ")([0-9a-f]{64})(";)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    with open(os.path.join(KDIR, name), "rb") as f:
        return f.read()


def text_record(j):
    data = read(j["file"])
    return dict(j, bytes=len(data), sha256=sha(data))


def published(m):
    return (json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pin(p, source, hdr):
    h = hashlib.sha256()
    if p["hash_of"] == "source":
        h.update(source)
    elif p["hash_of"] == "header":
        h.update(hdr)
    elif p["hash_of"] == "header+source":
        h.update(hdr)
        h.update(source)
    elif p["hash_of"] == "header+join+source":
        h.update(hdr)
        h.update(p.get("join", "").encode())
        h.update(source)
    else:
        raise SystemExit(f"repin: pin over {p['hash_of']!r} is not one this tool knows")
    return h.hexdigest()


def repin(old):
    m = json.loads(old)
    headers = {h["id"]: text_record(h) for h in m["headers"]}
    changed_headers = {h["id"] for h in m["headers"] if headers[h["id"]] != h}
    m["headers"] = [headers[h["id"]] for h in m["headers"]]
    m["port_addenda"] = [text_record(a) for a in m.get("port_addenda", [])]
    changed = set()
    for k in m["kernels"]:
        before = json.dumps(k, sort_keys=True)
        k["source"] = text_record(k["source"])
        source = read(k["source"]["file"])
        hdr = b""
        if k.get("header"):
            k["header"] = dict(k["header"], **{f: headers[k["header"]["id"]][f] for f in ("bytes", "sha256")})
            hdr = read(k["header"]["file"])
        for p in k["lane_pins"]:
            got = pin(p, source, hdr)
            if got.startswith(p["sha256"]):
                continue
            if not p["symbol"].startswith("mlx-stream"):
                raise SystemExit(f"repin: {k['name']}: the pin {p['symbol']!r} is another lane's record; its text must not change")
            p["sha256"] = got[: len(p["sha256"])]
        if json.dumps(k, sort_keys=True) != before:
            changed.add(k["name"])
    if not changed and not changed_headers and published(m) == old:
        return old, changed, changed_headers
    for p in m["predecessors"]:
        p["kernels"] = [n for n in p["kernels"] if n not in changed]
        p["headers"] = [n for n in p["headers"] if n not in changed_headers]
    m["predecessors"].append({
        "headers": [h["id"] for h in m["headers"] if h["id"] not in changed_headers],
        "kernels": [k["name"] for k in m["kernels"] if k["name"] not in changed],
        "manifest_sha256": sha(old),
        "widened": [],
    })
    return published(m), changed, changed_headers


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the manifest is stale; write nothing")
    args = ap.parse_args()
    with open(MANIFEST, "rb") as f:
        old = f.read()
    new, changed, changed_headers = repin(old)
    with open(REGISTRY) as f:
        zig = f.read()
    pinned = PIN_RE.search(zig)
    if pinned is None:
        raise SystemExit("repin: manifest_sha256 not found in src/exl3_kernels.zig")
    stale = new != old or pinned.group(2) != sha(old)
    if args.check:
        print(f"REPIN {'STALE' if stale else 'OK'} kernels={len(changed)} headers={len(changed_headers)}")
        return 1 if stale else 0
    if new == old and not stale:
        print(f"REPIN OK manifest {sha(old)} unchanged")
        return 0
    with open(MANIFEST, "wb") as f:
        f.write(new)
    with open(REGISTRY, "w") as f:
        f.write(PIN_RE.sub(lambda g: g.group(1) + sha(new) + g.group(3), zig, count=1))
    for name in sorted(changed_headers):
        print(f"header  {name}")
    for name in sorted(changed):
        print(f"kernel  {name}")
    print(f"REPIN {sha(old)} -> {sha(new)} (predecessor keeps {len(json.loads(new)['predecessors'][-1]['kernels'])} kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
