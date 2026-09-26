#!/usr/bin/env python3
"""Split a qwen35 GGUF with an embedded MTP block into the D-006 "separate file" packaging.

DECISIONS.md D-006: the ggml-org pack ships the trunk as a plain `block_count = n_layer`
model and the MTP block as its own GGUF with `block_count = n_layer + 1`,
`nextn_predict_layers = 1`, only `blk.<n_layer>.*` plus its own copies of `token_embd`,
`output` and `output_norm`. This tool produces that packaging from a single-file model
(e.g. /root/halo-ref/tiny/tiny-f32.gguf) so the engine's `mtp_path` route can be tested
against the single-file goldens: the tensors are copied byte for byte, only the file
layout and the structural keys change.

Outputs (default --out /root/halo-ref/tiny/split):
  <stem>-trunk.gguf           trunk: block_count n, no nextn key, recurrent_layers[:n], no blk.n.*
  <stem>-mtp.gguf             MTP:   block_count n+1, nextn_predict_layers 1, blk.n.* + embd/head copies
  <stem>-mtp-bad-dims.gguf    MTP file whose attention.head_count differs from the trunk
  <stem>-mtp-bad-arch.gguf    MTP file whose general.architecture is not qwen35
  <stem>-mtp-with-trunk.gguf  MTP file that also carries a trunk tensor (blk.0.attn_norm.weight)

Only F32 / F16 tensors are supported (the tiny F32 model); quantized files are refused.
Run with the reference venv: /root/halo-py/.venv/bin/python python/tools/split_mtp_gguf.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gguf
import numpy as np

ARCH = "qwen35"
BLK = re.compile(r"^blk\.(\d+)\.")


def field_value(f):
    return f.contents()


def copy_kvs(reader: gguf.GGUFReader, w: gguf.GGUFWriter, overrides: dict, drop: set[str]) -> None:
    for key, f in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key in drop or key in overrides:
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field_value(f), vtype, sub)
    for key, (val, vtype, sub) in overrides.items():
        w.add_key_value(key, val, vtype, sub)


def write(path: Path, reader: gguf.GGUFReader, arch: str, overrides: dict, drop: set[str], keep) -> int:
    w = gguf.GGUFWriter(str(path), arch)
    copy_kvs(reader, w, overrides, drop)
    n = 0
    for t in reader.tensors:
        if not keep(t.name):
            continue
        if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            raise SystemExit(f"{t.name}: {t.tensor_type.name} is not supported (F32/F16 only)")
        w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path, nargs="?", default=Path("/root/halo-ref/tiny/tiny-f32.gguf"))
    ap.add_argument("--out", type=Path, default=Path("/root/halo-ref/tiny/split"))
    args = ap.parse_args()
    r = gguf.GGUFReader(str(args.model))
    assert field_value(r.fields["general.architecture"]) == ARCH, "not a qwen35 GGUF"
    blocks = int(field_value(r.fields[f"{ARCH}.block_count"]))
    nextn = int(field_value(r.fields.get(f"{ARCH}.nextn_predict_layers"))) if f"{ARCH}.nextn_predict_layers" in r.fields else 0
    if nextn != 1:
        raise SystemExit(f"{args.model}: expected one embedded MTP block (nextn_predict_layers = 1), found {nextn}")
    n = blocks - 1
    rl_key = f"{ARCH}.attention.recurrent_layers"
    rl = list(field_value(r.fields[rl_key])) if rl_key in r.fields else None
    U32, ARR, BOOL = gguf.GGUFValueType.UINT32, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.BOOL
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.model.stem

    def blk(name):
        m = BLK.match(name)
        return int(m.group(1)) if m else None

    shared = {"token_embd.weight", "output.weight", "output_norm.weight"}

    # ---- trunk ----------------------------------------------------------------------------
    t_over = {f"{ARCH}.block_count": (n, U32, None)}
    if rl is not None:
        t_over[rl_key] = (rl[:n], ARR, BOOL)
    k = write(args.out / f"{stem}-trunk.gguf", r, ARCH, t_over, {f"{ARCH}.nextn_predict_layers"},
              lambda name: blk(name) is None or blk(name) < n)
    print(f"trunk: {k} tensors, block_count {n}")

    # ---- MTP-only file (ggml-org mtp-*.gguf layout) ---------------------------------------
    def mtp_only(name):
        return blk(name) == n or name in shared

    m_over = {f"{ARCH}.block_count": (blocks, U32, None), f"{ARCH}.nextn_predict_layers": (1, U32, None)}
    k = write(args.out / f"{stem}-mtp.gguf", r, ARCH, m_over, set(), mtp_only)
    print(f"mtp: {k} tensors, block_count {blocks}")

    # ---- deliberately mismatched MTP files (rejection tests) -------------------------------
    heads = int(field_value(r.fields[f"{ARCH}.attention.head_count"]))
    bad_dims = dict(m_over)
    bad_dims[f"{ARCH}.attention.head_count"] = (heads * 2, U32, None)
    write(args.out / f"{stem}-mtp-bad-dims.gguf", r, ARCH, bad_dims, set(), mtp_only)
    # Wrong architecture: the qwen35.* keys stay, general.architecture says llama.
    write(args.out / f"{stem}-mtp-bad-arch.gguf", r, "llama", m_over, set(), mtp_only)
    write(args.out / f"{stem}-mtp-with-trunk.gguf", r, ARCH, m_over, set(),
          lambda name: mtp_only(name) or name == "blk.0.attn_norm.weight")
    print("mismatched MTP files written")


if __name__ == "__main__":
    main()
