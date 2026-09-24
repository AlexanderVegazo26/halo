"""Build the dequantization oracle for HALO's tensor tests (DECISIONS.md D-007).

For every tensor type present in the real published Qwen3.8-27B GGUF files, this tool
HTTP-Range-fetches a few whole rows from one tensor of that type, and writes the raw
bytes plus gguf-py's ``gguf.quants.dequantize`` float32 output. It also covers:

* every type present in the local tiny models (real llama-quantize output: Q5_0, Q5_1, ...);
* synthetic seeded random blocks for every type HALO dequantizes (scale fields forced
  to finite fp16), plus fp16/bf16 special values.

Before trusting a remote file it checks that the file still matches the header saved by
``fetch_gguf_headers.py`` (byte-compare of a prefix and of the header tail), that the bytes
between the header end and the aligned data start are zero padding, and that
``data_start + max(offset + nbytes)`` equals the remote file size (up to the final
alignment pad). That identity validates the data-offset rule (header length rounded up
to general.alignment) and the byte size of the last tensor only; per-dtype geometry is
checked by the C++ tests' table-order packing assertion. The file size is recorded so the
C++ header-only tests can assert the same identity.

Output: ``~/halo-ref/quant_blocks/`` (``manifest.json`` + ``*.raw`` / ``*.f32``).
Run with the reference venv: ``/root/halo-py/.venv/bin/python python/tools/fetch_quant_blocks.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import struct
import sys
import urllib.request
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

REF = Path.home() / "halo-ref"
OUT = REF / "quant_blocks"

# Same URLs as python/tools/fetch_gguf_headers.py (that script runs on import, so the
# table is duplicated here; keep the two in sync).
FILES: dict[str, str] = {
    "ggml-org-q4km": "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf",
    "ggml-org-mtp-q4_0": "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/mtp-Qwen3.8-27B-Q4_0.gguf",
    "unsloth-ud-q4kxl": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf",
}

# Types HALO's CPU reference dequantizes (include/halo/tensor/dtype.h dequant_supported).
SUPPORTED = ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K",
             "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "IQ3_S"]

# Byte offsets of fp16 scale fields inside one block (ggml/src/ggml-common.h). Synthetic
# blocks get finite scales there so outputs are finite and comparisons meaningful.
FP16_FIELDS: dict[str, tuple[int, ...]] = {
    "Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,),
    "Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,),
    "IQ4_NL": (0,), "IQ4_XS": (0,), "IQ3_S": (0,),
}

ROWS_PER_CHUNK = 3
USER_AGENT = "halo-dev"


class OracleError(RuntimeError):
    """A consistency check on remote or local reference data failed."""


@dataclass
class Http:
    """Minimal HTTP Range client that records headers of redirect hops."""

    hops: list[dict[str, str]] = field(default_factory=list)

    def get(self, url: str, start: int, end: int) -> tuple[bytes, int, dict[str, str]]:
        """Fetch bytes [start, end] (inclusive). Returns (data, total_file_size, headers)."""
        hops = self.hops

        class Recorder(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Message,
                                 newurl: str) -> Any:
                hops.append({k.lower(): v for k, v in headers.items()})
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(Recorder())
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        with opener.open(req, timeout=300) as r:
            if r.status != 206:
                raise OracleError(f"{url}: expected 206 Partial Content, got {r.status}")
            data = r.read()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
        m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", hdrs.get("content-range", ""))
        if not m or int(m.group(1)) != start or len(data) != end - start + 1:
            raise OracleError(f"{url}: bad range response {hdrs.get('content-range')} len={len(data)}")
        return data, int(m.group(3)), hdrs


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def tensor_nbytes(dims: list[int], type_name: str) -> int:
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType[type_name]]
    if dims[0] % block:
        raise OracleError(f"ne0={dims[0]} not a multiple of {type_name} block {block}")
    rows = int(np.prod(dims[1:], dtype=np.int64)) if len(dims) > 1 else 1
    return rows * (dims[0] // block) * size


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


@dataclass
class Writer:
    out: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, entry_id: str, type_name: str, raw: bytes, ne0: int, rows: int, **meta: Any) -> None:
        qtype = GGMLQuantizationType[type_name]
        block, size = GGML_QUANT_SIZES[qtype]
        row_bytes = ne0 // block * size
        if len(raw) != rows * row_bytes:
            raise OracleError(f"{entry_id}: {len(raw)} bytes != {rows} x {row_bytes}")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_bytes)
        ref = np.asarray(dequantize(arr, qtype), dtype=np.float32)
        if ref.shape != (rows, ne0):
            raise OracleError(f"{entry_id}: gguf-py returned shape {ref.shape}, expected {(rows, ne0)}")
        stem = safe(entry_id)
        (self.out / f"{stem}.raw").write_bytes(raw)
        ref.astype("<f4").tofile(self.out / f"{stem}.f32")
        self.entries.append({
            "id": entry_id, "type": type_name, "type_id": int(qtype), "ne0": ne0, "rows": rows,
            "row_bytes": row_bytes, "raw": f"{stem}.raw", "ref": f"{stem}.f32",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "n_nan": int(np.isnan(ref).sum()), **meta,
        })


def remote(writer: Writer, http: Http, key: str, url: str) -> dict[str, Any]:
    summary = json.loads((REF / f"{key}.summary.json").read_text(encoding="utf-8"))
    header = (REF / f"{key}.header.gguf").read_bytes()
    header_len = int(summary["header_len"])
    if len(header) != header_len:
        raise OracleError(f"{key}: local header is {len(header)} bytes, summary says {header_len}")
    alignment = int(summary["kv"].get("general.alignment", 32))
    data_start = align_up(header_len, alignment)

    # 1) The remote file still is the file whose header we saved.
    http.hops.clear()
    prefix, file_size, hdrs = http.get(url, 0, 65535)
    if prefix != header[:65536]:
        raise OracleError(f"{key}: remote prefix differs from saved header (file changed upstream?)")
    ident = {k: v for hop in [*http.hops, hdrs] for k, v in hop.items()
             if k in ("etag", "x-repo-commit", "x-linked-etag", "last-modified", "x-linked-size")}
    tail_start = header_len - 4096
    tail, _, _ = http.get(url, tail_start, data_start + 63)
    if tail[: header_len - tail_start] != header[tail_start:]:
        raise OracleError(f"{key}: remote header tail differs from saved header")
    pad = tail[header_len - tail_start: data_start - tail_start]
    if any(pad):
        raise OracleError(f"{key}: bytes [{header_len}, {data_start}) are not zero padding")

    # 2) data_start + max(offset + nbytes) == file size (up to the final alignment pad).
    tensors = summary["tensors"]
    if len(tensors) != int(summary["n_tensors"]):
        raise OracleError(f"{key}: summary lists {len(tensors)} tensors, header says {summary['n_tensors']}")
    data_end = max(int(t["offset"]) + tensor_nbytes([int(d) for d in t["dims"]], t["type"]) for t in tensors)
    if not (data_start + data_end <= file_size <= data_start + align_up(data_end, alignment)):
        raise OracleError(f"{key}: data_start {data_start} + data_end {data_end} inconsistent with size {file_size}")
    print(f"{key}: size {file_size} header {header_len} data_start {data_start} data_end {data_end} OK")

    # 3) A few whole rows (start + middle of the tensor) for each type present.
    seen: set[str] = set()
    for t in tensors:
        tname = t["type"]
        dims = [int(d) for d in t["dims"]]
        if tname in seen or (tname == "F32" and dims[0] < 1024):
            continue
        seen.add(tname)
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[tname]]
        row_bytes = dims[0] // block * size
        n_rows = int(np.prod(dims[1:], dtype=np.int64)) if len(dims) > 1 else 1
        for first in sorted({0, n_rows // 2}):
            rows = min(ROWS_PER_CHUNK, n_rows - first)
            abs_off = data_start + int(t["offset"]) + first * row_bytes
            raw, _, _ = http.get(url, abs_off, abs_off + rows * row_bytes - 1)
            writer.add(f"{key}.{t['name']}.r{first}", tname, raw, dims[0], rows, kind="remote", source=key,
                       tensor=t["name"], first_row=first, abs_offset=abs_off)
            print(f"  {tname:7s} {t['name']} rows {first}..{first + rows - 1} ({len(raw)} B)")
    return {"url": url, "file_size": file_size, "header_len": header_len, "alignment": alignment,
            "data_start": data_start, "data_end": data_end, "types": sorted(seen), "identity": ident}


def local_tiny(writer: Writer) -> list[str]:
    covered: list[str] = []
    for path in sorted((REF / "tiny").glob("tiny-*.gguf")):
        reader = GGUFReader(path)
        for t in reader.tensors:
            tname = t.tensor_type.name
            if tname in covered or tname == "F32":
                continue
            covered.append(tname)
            data = np.asarray(t.data)
            rows = min(ROWS_PER_CHUNK, data.shape[0] if data.ndim > 1 else 1)
            raw = data.reshape(-1, data.shape[-1])[:rows].view(np.uint8).tobytes()
            ne0 = int(t.shape[0])
            writer.add(f"tiny.{path.stem}.{t.name}", tname, raw, ne0, rows, kind="local", source=path.name,
                       tensor=t.name, first_row=0)
    return covered


def synthetic(writer: Writer, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for tname in SUPPORTED:
        if tname == "F32":
            continue
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[tname]]
        rows, n_blocks = 2, max(8, 256 // block)
        ne0 = n_blocks * block
        raw = bytearray(rng.integers(0, 256, rows * n_blocks * size, dtype=np.uint8).tobytes())
        if tname == "F16":
            specials = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
                        0x7E00, 0x7C01, 0xFE00, 0x3555, 0x0200]
            raw[: 2 * len(specials)] = struct.pack(f"<{len(specials)}H", *specials)
        for f in FP16_FIELDS.get(tname, ()):
            for b in range(rows * n_blocks):
                mag = np.float16(2.0 ** rng.uniform(-12, 2) * rng.choice([-1.0, 1.0]))
                raw[b * size + f: b * size + f + 2] = mag.tobytes()
        writer.add(f"synthetic.{tname}", tname, bytes(raw), ne0, rows, kind="synthetic", seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--offline", action="store_true", help="skip remote files (local + synthetic only)")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out)
    http = Http()
    files = {} if args.offline else {k: remote(writer, http, k, u) for k, u in FILES.items()}
    tiny = local_tiny(writer)
    synthetic(writer, args.seed)
    manifest = {
        "generator": "python/tools/fetch_quant_blocks.py",
        "gguf_py": importlib.metadata.version("gguf"),
        "quant_sizes": {q.name: {"id": int(q), "block_elems": b, "block_bytes": s}
                        for q, (b, s) in GGML_QUANT_SIZES.items()},
        "files": files,
        "tiny_types": tiny,
        "entries": writer.entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by_kind: dict[str, int] = {}
    for e in writer.entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    print(f"wrote {len(writer.entries)} entries {by_kind} to {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OracleError as e:
        print(f"ORACLE CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(2)
