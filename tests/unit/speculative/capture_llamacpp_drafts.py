#!/usr/bin/env python3
"""Capture llama.cpp's MTP draft chain on the tiny qwen35 model (review R-5(b), D-005).

llama.cpp is the only executable reference for the qwen35 MTP drafter (DECISIONS D-005). This
tool runs llama-server with `--spec-type draft-mtp` on tiny-f32.gguf, greedy, for each golden
prompt, and records every draft call's top-3 candidates at every draft depth from the server's
debug log (common/speculative.cpp, `common_speculative_impl_draft_mtp::draft`, SPC_DBG
"draft candidate" lines) plus the target's acceptance count after each call. One fresh server
per prompt: llama.cpp does not reset its MTP carry-over hidden between requests on a slot.
The C++ test Spec.DraftChainMatchesLlamaCpp (test_speculative.cpp) rebuilds llama.cpp's MTP
state for every recorded call and compares HALO's draft chain, depth by depth, with these
records.

Output (default: llamacpp_draft_chain.json next to this file) is committed test data.
Re-run it when tiny-f32.gguf, the llama.cpp reference commit or the prompts change:

  python3 tests/unit/speculative/capture_llamacpp_drafts.py \
      --server /root/llama.cpp/build/bin/llama-server --model /root/halo-ref/tiny/tiny-f32.gguf

Standard library only (runs with the system python3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

N_DRAFT = 3
N_PREDICT = 12
PROMPTS = ("p0", "p1", "p2")
CAND = re.compile(
    r"seq_id\s+(\d+), draft candidate\s+(\d+), pos\s+(\d+):\s+(-?\d+) \(\s*([-0-9.naif]+)\)"
)
ACC = re.compile(r"accepted (\d+)/(\d+) draft tokens")


def load_i32(golden: Path, name: str) -> list[int]:
    manifest = json.loads((golden / "manifest.json").read_text())
    ent = manifest["tensors"][name]
    assert ent["dtype"] == "int32", ent
    raw = (golden / ent["file"]).read_bytes()
    return [int.from_bytes(raw[i : i + 4], "little", signed=True) for i in range(0, len(raw), 4)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit("llama-server did not become healthy")


def parse_calls(log: str) -> tuple[list[list[list[tuple[int, float]]]], list[int]]:
    """Draft calls in order; each call = list over depth of [(id, p) x top-3]. Also the
    number of drafts the target accepted after each call (server "accepted N/M" lines)."""
    calls: list[list[list[tuple[int, float]]]] = []
    accepted: list[int] = []
    for line in log.splitlines():
        a = ACC.search(line)
        if a:
            accepted.append(int(a.group(1)))
            assert len(accepted) == len(calls), f"acceptance line without a draft call: {line}"
            continue
        m = CAND.search(line)
        if not m:
            continue
        cand, depth, tok, p = int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))
        if depth == 0 and cand == 0:
            calls.append([])
        assert calls, line
        call = calls[-1]
        if cand == 0:
            assert depth == len(call), f"depth {depth} out of order in call {len(calls) - 1}: {line}"
            call.append([])
        call[depth].append((tok, p))
    assert len(accepted) == len(calls), f"{len(calls)} draft calls but {len(accepted)} acceptance lines"
    return calls, accepted


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", type=Path, default=Path("/root/llama.cpp/build/bin/llama-server"))
    ap.add_argument("--model", type=Path, default=Path("/root/halo-ref/tiny/tiny-f32.gguf"))
    ap.add_argument("--golden", type=Path, default=Path("/root/halo-ref/tiny/golden"))
    ap.add_argument("--out", type=Path, default=here / "llamacpp_draft_chain.json")
    ap.add_argument("--log", type=Path, default=None, help="keep the raw server log here")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.server.parent.parent.parent), "rev-parse", "--short=9", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    port = free_port()
    flags = [
        "--host", "127.0.0.1", "--port", str(port), "-m", str(args.model),
        "-c", "1024", "-np", "1", "-t", str(args.threads), "-tb", str(args.threads), "-ngl", "0",
        "--no-cache-prompt", "--no-warmup", "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(N_DRAFT), "--spec-draft-n-min", "0", "--draft-p-min", "0",
        "-lv", "99",
    ]
    log_path = args.log or args.out.with_suffix(".log.tmp")
    results = {}
    with log_path.open("w") as logf:
        for name in PROMPTS:
            # A fresh server per prompt: llama.cpp's MTP carry-over row (pending_h) lives per
            # slot and is not reset between requests, so a second request on the same slot
            # would pair its first prompt token with the previous request's last hidden.
            # Fresh, it is zero-initialised (see the MTP position-0 note in the C++ test).
            proc = subprocess.Popen([str(args.server), *flags], stdout=logf, stderr=subprocess.STDOUT)
            try:
                wait_healthy(port, proc)
                prompt = load_i32(args.golden, f"{name}.tokens")
                logf.flush()
                start = log_path.stat().st_size
                r = post(port, "/completion", {
                    "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0.0, "top_k": 1,
                    "cache_prompt": False, "return_tokens": True, "ignore_eos": True, "seed": 1,
                })
                time.sleep(0.5)
                logf.flush()
                with log_path.open(encoding="utf-8", errors="replace") as lf:
                    lf.seek(start)
                    chunk = lf.read()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            calls, accepted = parse_calls(chunk)
            tokens = r.get("tokens", [])
            n_prompt = r.get("timings", {}).get("prompt_n") or r.get("tokens_evaluated")
            results[name] = {
                "prompt_len": len(prompt),
                "server_prompt_tokens": n_prompt,
                "tokens": tokens,
                "golden_decode_tokens": load_i32(args.golden, f"{name}.decode_tokens"),
                "draft_calls": [[[[t, p] for t, p in depth] for depth in call] for call in calls],
                "accepted": accepted,
            }
            print(f"{name}: {len(tokens)} tokens, {len(calls)} draft calls, prompt {len(prompt)} "
                  f"(server evaluated {n_prompt})", file=sys.stderr)
    out = {
        "what": "llama.cpp MTP draft chain (top-3 candidates per draft depth, per draft call) on tiny-f32, greedy",
        "llama_cpp_commit": commit,
        "server_flags": ["<port>" if f == str(port) else "<model>" if f == str(args.model) else f for f in flags],
        "model": args.model.name,
        "model_sha256": sha256(args.model),
        "n_draft": N_DRAFT,
        "n_predict": N_PREDICT,
        "note": "probabilities are llama.cpp's top-k=10 draft sampler softmax, printed with 3 decimals",
        "prompts": results,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    if args.log is None:
        log_path.unlink()
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
