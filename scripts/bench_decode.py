#!/usr/bin/env python3
"""Decode tok/s sweep over the next.md speed flags, measured only on an idle GPU.

Runs next.md's "Verification (every step)" speed command once per variant (the baseline,
plain decode, deeper drafts, and each opt-in HALO_* flag alone) and writes a table with
tok/s, the change against the baseline, the request's draft acceptance, and whether the
greedy output is identical to the baseline's. `--variants subgroup-reduce+draft-vocab-32k`
measures several flags together (env and args merged; bitwise only if every part is).

    scripts/bench_decode.py --halo build-dev/tools/halo/halo --model Qwen3.8-27B-UD-Q4_K_XL.gguf \
        [--variants all|NAME,NAME+NAME] [-n 128] [--ctx 4096] [--repeat 1] [--busy-max 5] [--busy-settle 10] \
        [--report-dir .test-report/bench-decode]

Numbers taken while another process keeps the GPU busy are not benchmarks (2026-11-14: a
24-token prompt + 4 tokens took 62 s beside a busy model server). So before every run the
script samples gpu_busy_percent; above --busy-max it runs nothing more, marks the remaining
variants "not measured", and exits 3. It samples again after every run: gpu_busy_percent is
smoothed by amdgpu, so halo's own load takes a few seconds to fade once it has exited (measured
2026-11-17 on the idle EVO-X2: 72 -> 28 after 1 s, <= 5 only after ~2.75 s). The script polls
until the reading is back under --busy-max, for up to --busy-settle seconds; a GPU still busy
after that means another process overlapped the run, so that run's tok/s is discarded the same
way instead of being reported as measured.
Every inherited HALO_* speed flag is removed so the baseline really is the default build
behaviour.

Exit: 0 every variant measured, no failure · 1 a run failed, printed no tok/s, or a variant
that must be bitwise the baseline produced different output · 2 usage · 3 GPU busy (or
gpu_busy_percent unreadable): some variants not measured. stdlib only; safe to rerun.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. Write a short story about it."
DEFAULT_BUSY_GLOB = "/sys/class/drm/card*/device/gpu_busy_percent"

# Every env var that changes decode speed or numerics. They are stripped from the inherited
# environment so only the variant's own setting is in effect.
SPEED_FLAGS = (
    "HALO_SUBGROUP_REDUCE", "HALO_MTP_DRAFT_VOCAB", "HALO_GDN_REG", "HALO_GDN_DV_TILE",
    "HALO_STREAM_REUSE", "HALO_MTP_DEVICE_HIDDEN", "HALO_MTP_CHAIN", "HALO_MTP_TREE",
    "HALO_FUSE_GEMV", "HALO_FUSE_NORM", "HALO_COOPMAT", "HALO_REPACK", "HALO_KV_FP16",
    "HALO_KV_TYPE", "HALO_NGRAM_DRAFT", "HALO_VK_GEMV_WG", "HALO_VK_SYNC_EVERY", "HALO_VK_OP_TIMINGS",
    "HALO_VK_PUSH_DESCRIPTORS",
)


@dataclass(frozen=True)
class Variant:
    name: str
    step: str  # next.md step it measures
    env: dict = field(default_factory=dict)
    args: tuple = ()
    # False when the variant may legitimately change greedy output (lossy KV storage, fp16
    # GEMM products): its output is reported, not required to equal the baseline's.
    bitwise: bool = True


VARIANTS = (
    Variant("baseline", "-", {}, ()),
    Variant("plain", "-", {}, ("--mtp-draft", "0")),
    Variant("mtp-draft-3", "5", {}, ("--mtp-draft", "3")),
    Variant("subgroup-reduce", "1", {"HALO_SUBGROUP_REDUCE": "1"}),
    Variant("draft-vocab-32k", "2", {"HALO_MTP_DRAFT_VOCAB": "32768"}),
    Variant("gdn-reg", "3", {"HALO_GDN_REG": "1"}),
    Variant("gdn-dv-tile-32", "3", {"HALO_GDN_DV_TILE": "32"}),
    Variant("stream-reuse", "4", {"HALO_STREAM_REUSE": "1"}),
    Variant("push-descriptors", "4", {"HALO_VK_PUSH_DESCRIPTORS": "1"}),
    Variant("mtp-device-hidden", "4", {"HALO_MTP_DEVICE_HIDDEN": "1"}),
    Variant("mtp-chain", "4", {"HALO_MTP_CHAIN": "1"}),
    Variant("mtp-tree", "5", {"HALO_MTP_TREE": "1"}),
    Variant("fuse-gemv", "6", {"HALO_FUSE_GEMV": "all"}),
    Variant("coopmat", "7", {"HALO_COOPMAT": "1"}, bitwise=False),
    Variant("repack", "8", {"HALO_REPACK": "1"}),
    Variant("kv-fp16", "9", {"HALO_KV_FP16": "1"}, bitwise=False),
    Variant("kv-q8", "9", {"HALO_KV_TYPE": "q8"}, bitwise=False),
    Variant("fuse-norm", "10", {"HALO_FUSE_NORM": "1"}),
    Variant("ngram", "11", {"HALO_NGRAM_DRAFT": "1"}),
)

_TPS = re.compile(r"\[halo\] prompt (\d+) tokens, generated (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s decode\), finish (\w+)")
_DRAFTS = re.compile(r"\[halo\] drafts (\d+) accepted of (\d+) proposed(?: \(([\d.]+)%\))?(.*)$")


@dataclass
class Result:
    name: str
    step: str
    env: dict
    args: list
    bitwise: bool
    status: str = "not measured"  # measured | failed | not measured
    note: str = ""
    tps: list = field(default_factory=list)
    tps_median: float | None = None
    vs_baseline_pct: float | None = None
    prompt_tokens: int | None = None
    generated: int | None = None
    drafts: str = ""
    same_output: bool | None = None
    busy_before: int | None = None
    busy_after: int | None = None  # highest sample right after the last run that finished
    command: str = ""


def gpu_busy(pattern: str, samples: int, interval: float) -> int | None:
    """Highest gpu_busy_percent over `samples` reads of every matching file; None if unreadable."""
    files = sorted(glob.glob(pattern))
    worst = None
    for i in range(max(1, samples)):
        for f in files:
            try:
                v = int(Path(f).read_text().strip())
            except (OSError, ValueError):
                continue
            worst = v if worst is None else max(worst, v)
        if i + 1 < samples:
            time.sleep(interval)
    return worst


def clean_env(base: dict) -> tuple[dict, list]:
    env = {k: v for k, v in base.items() if k not in SPEED_FLAGS}
    return env, sorted(k for k in base if k in SPEED_FLAGS)


def parse_run(stderr: str) -> dict:
    out: dict = {}
    for line in stderr.splitlines():
        if m := _TPS.search(line):
            out.update(prompt_tokens=int(m[1]), generated=int(m[2]), tps=float(m[4]), finish=m[5])
        elif m := _DRAFTS.search(line):
            out["drafts"] = line.strip().removeprefix("[halo] ")
    return out


def wait_settled(a) -> int | None:
    """Busy reading once halo's own load has faded: polls until <= --busy-max or --busy-settle
    seconds have passed, and returns the last reading (None if unreadable)."""
    deadline = time.monotonic() + a.busy_settle
    while True:
        busy = gpu_busy(a.busy_file, a.busy_samples, a.busy_interval)
        if busy is None or busy <= a.busy_max or time.monotonic() >= deadline:
            return busy
        time.sleep(max(a.busy_interval, 0.05))


def speed_command(v: Variant, a) -> tuple[list, str]:
    """The variant's speed command as argv, and as the shell line that reruns it alone."""
    cmd = [a.halo, "run", a.model, "-p", a.prompt, "--raw", "--temperature", "0", "-n", str(a.max_tokens),
           "--ctx", str(a.ctx), "--parallel", "1", *v.args]
    return cmd, " ".join([*(f"{k}={shlex.quote(x)}" for k, x in v.env.items()), *map(shlex.quote, cmd)])


def run_variant(v: Variant, a, env: dict) -> tuple[Result, list]:
    cmd, line = speed_command(v, a)
    r = Result(v.name, v.step, dict(v.env), list(v.args), v.bitwise, command=line)
    outputs = []
    for i in range(a.repeat):
        busy = gpu_busy(a.busy_file, a.busy_samples, a.busy_interval)
        if i == 0:
            r.busy_before = busy
        if busy is None:
            r.note = f"gpu_busy_percent unreadable ({a.busy_file}): idle GPU not confirmed"
            return r, outputs
        if busy > a.busy_max:
            r.note = f"GPU busy {busy}% > {a.busy_max}% before run {i + 1}"
            r.tps = []
            return r, outputs
        try:
            p = subprocess.run(cmd, env={**env, **v.env}, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            r.status, r.note = "failed", f"timed out after {a.timeout} s"
            return r, outputs
        info = parse_run(p.stderr)
        if p.returncode != 0 or "tps" not in info:
            tail = " | ".join(p.stderr.strip().splitlines()[-3:])
            r.status, r.note = "failed", f"exit {p.returncode}{'' if 'tps' in info else ', no tok/s line'}: {tail}"
            return r, outputs
        # halo has exited, so its own work is done once its smoothed load has faded: a GPU
        # still busy after the settle window was shared with another process during the run,
        # and the run's tok/s is not a benchmark.
        after = wait_settled(a)
        r.busy_after = after
        if after is None or after > a.busy_max:
            seen = "unreadable" if after is None else f"{after}% > {a.busy_max}%"
            r.note = (f"GPU busy {seen} right after run {i + 1} (still after {a.busy_settle:g} s settle): "
                      "another process overlapped it, tok/s discarded")
            r.tps = []
            return r, []
        r.tps.append(info["tps"])
        r.prompt_tokens, r.generated = info["prompt_tokens"], info["generated"]
        r.drafts = info.get("drafts", "")
        outputs.append(p.stdout)
    if len(set(outputs)) > 1:
        r.status, r.note = "failed", "greedy output changed between repeats"
        return r, outputs
    r.status = "measured"
    r.tps_median = statistics.median(r.tps)
    return r, outputs


def combine(name: str, parts: list[Variant]) -> Variant:
    """One variant that applies every part's flags at once (`a+b`): its env and args merged, bitwise
    only if every part is. Parts that set the same HALO_* knob or both pass CLI args (e.g. two
    --mtp-draft depths) can't be combined: one would silently override the other."""
    env: dict = {}
    args: list = []
    for p in parts:
        clash = sorted(k for k in p.env if k in env)
        if clash:
            raise SystemExit(f"bench_decode: variant {name}: {p.name} sets {', '.join(clash)} again")
        if p.args and args:
            raise SystemExit(f"bench_decode: variant {name}: {p.name} passes {' '.join(p.args)} after "
                             f"{' '.join(args)}; only one part may set CLI args")
        env.update(p.env)
        args += p.args
    return Variant(name, "+".join(p.step for p in parts), env, tuple(args), all(p.bitwise for p in parts))


def select(names: str) -> list[Variant]:
    if names == "all":
        return list(VARIANTS)
    by_name = {v.name: v for v in VARIANTS}
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [p for n in wanted for p in n.split("+") if p not in by_name]
    if unknown:
        raise SystemExit(f"bench_decode: unknown variant(s) {', '.join(unknown)}; known: {', '.join(by_name)}")
    chosen = []
    for n in wanted:
        parts = n.split("+")
        if len(parts) == 1:
            chosen.append(by_name[n])
            continue
        if len(set(parts)) != len(parts) or "baseline" in parts:
            raise SystemExit(f"bench_decode: variant {n}: a combination needs distinct parts other than baseline")
        chosen.append(combine(n, [by_name[p] for p in parts]))
    # The baseline always runs first: every comparison is against it.
    return [by_name["baseline"], *(v for v in chosen if v.name != "baseline")]


def write_report(a, results: list[Result], stripped: list, rc: int) -> Path:
    d = Path(a.report_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"halo": a.halo, "model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens, "ctx": a.ctx,
            "repeat": a.repeat, "busy_max": a.busy_max, "busy_settle": a.busy_settle, "stripped_env": stripped,
            "exit": rc,
            "date": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (d / "bench-decode.json").write_text(json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, indent=2) + "\n")
    lines = [f"# Decode sweep: {'PASS' if rc == 0 else 'GPU BUSY: not measured' if rc == 3 else 'FAIL'}", "",
             f"- model `{a.model}`, `-n {a.max_tokens} --ctx {a.ctx}`, {a.repeat} run(s) per variant (median), "
             f"busy limit {a.busy_max}%, {meta['date']}"]
    if stripped:
        lines.append(f"- removed from the inherited environment: {', '.join(stripped)}")
    lines += ["", "| variant | step | tok/s | vs baseline | drafts | output == baseline | status |", "|---|---|---|---|---|---|---|"]
    for r in results:
        tps = f"{r.tps_median:.2f}" if r.tps_median is not None else "-"
        vs = f"{r.vs_baseline_pct:+.1f}%" if r.vs_baseline_pct is not None else "-"
        same = "-" if r.same_output is None else "yes" if r.same_output else ("NO" if r.bitwise else "no (lossy, allowed)")
        status = r.status + (f": {r.note}" if r.note else "")
        lines.append(f"| {r.name} | {r.step} | {tps} | {vs} | {r.drafts.replace('|', '/') or '-'} | {same} | {status} |")
    lines += ["", "Rerun one variant:", ""]
    lines += [f"- {r.name}: `{r.command}`" for r in results]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--halo", required=True, help="halo CLI binary (Vulkan build)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("-n", "--max-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--variants", default="all",
                    help="all, or a comma list; a+b runs those variants' flags together (baseline always runs first)")
    ap.add_argument("--busy-max", type=int, default=5, help="highest gpu_busy_percent allowed before a run")
    ap.add_argument("--busy-file", default=DEFAULT_BUSY_GLOB, help="glob of gpu_busy_percent files")
    ap.add_argument("--busy-samples", type=int, default=5)
    ap.add_argument("--busy-interval", type=float, default=0.2, help="seconds between busy samples")
    ap.add_argument("--busy-settle", type=float, default=10.0,
                    help="seconds a run's own smoothed GPU load may take to fade before the GPU counts as "
                         "shared with another process")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per run")
    ap.add_argument("--report-dir", default=".test-report/bench-decode")
    a = ap.parse_args(argv)
    if a.repeat < 1 or a.max_tokens < 1:
        ap.error("--repeat and -n must be >= 1")
    if a.busy_settle < 0:
        ap.error("--busy-settle must be >= 0")
    variants = select(a.variants)
    env, stripped = clean_env(dict(os.environ))

    results: list[Result] = []
    base_out = None
    busy = False
    for v in variants:
        if busy:  # once busy, measure nothing more: a later idle moment would mix conditions
            results.append(Result(v.name, v.step, dict(v.env), list(v.args), v.bitwise, note="skipped after GPU busy",
                                  command=speed_command(v, a)[1]))
            continue
        r, outs = run_variant(v, a, env)
        results.append(r)
        if r.status == "not measured":
            busy = True
            print(f"[bench-decode] {r.name}: not measured: {r.note}", file=sys.stderr)
            continue
        if r.status != "measured":
            print(f"[bench-decode] {r.name}: {r.status}: {r.note}", file=sys.stderr)
            continue
        if v.name == "baseline":
            base_out = outs[0]
        if base_out is not None:
            r.same_output = outs[0] == base_out
            base_tps = results[0].tps_median
            if base_tps:
                r.vs_baseline_pct = 100.0 * (r.tps_median - base_tps) / base_tps
            if not r.same_output and r.bitwise:
                r.status, r.note = "failed", "greedy output differs from the baseline (must be bitwise)"
        print(f"[bench-decode] {r.name}: {r.tps_median:.2f} tok/s" if r.status == "measured"
              else f"[bench-decode] {r.name}: {r.status}: {r.note}", file=sys.stderr)

    rc = 1 if any(r.status == "failed" for r in results) else 3 if any(r.status != "measured" for r in results) else 0
    d = write_report(a, results, stripped, rc)
    print(f"[bench-decode] {'PASS' if rc == 0 else 'FAIL' if rc == 1 else 'NOT MEASURED (GPU busy)'}: {d / 'summary.md'}",
          file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
