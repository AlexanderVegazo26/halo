#!/usr/bin/env python3
"""Aggregate HALO_VK_OP_TIMINGS profiles by kernel, and compare two of them (next.md step 3).

HALO_VK_OP_TIMINGS=<path> makes the Vulkan backend append one `<kernel> <ns>` line per dispatch
(GPU timestamps) to <path> at every Stream::wait(), followed by one `# wait` line per completed
fence wait. The file is appended to, so delete it before each profiled run.

    scripts/op_profile.py summary PROFILE [--per N] [--top K] [--json OUT]
    scripts/op_profile.py compare BEFORE AFTER [--per-before N] [--per-after N] [--top K] [--json OUT]

`summary` prints, per kernel: dispatches, total ms, average and max us, and the share of all GPU
time, largest first, plus the matvec_* share (process.md section 13.2's table) and the fence
waits (with dispatches per wait). `compare` prints the per-kernel change in total ms and
dispatches between two profiles, largest change first, with kernels that only one side ran marked
new/gone, plus the fence-wait change: the attribution check of next.md ("re-profile with
HALO_VK_OP_TIMINGS and aggregate by kernel ... confirm the per-tick wall time and fence count
drop"). `--per N` divides every total and count by N (e.g. the generated tokens or decode ticks
of that run), so runs of different lengths compare per token. Profiles written before the
`# wait` markers existed report fence waits as not recorded. Other `#` lines are comments.

Exit: 0 ok · 2 usage, unreadable file, or a malformed line (named as file:line). stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


class ProfileError(Exception):
    """A profile that can't be read; the message names the file (and line)."""


@dataclass
class KernelStat:
    kernel: str
    dispatches: int
    total_ns: float
    max_ns: float

    @property
    def avg_ns(self) -> float:
        return self.total_ns / self.dispatches if self.dispatches else 0.0


WAIT_MARKER = "# wait"


@dataclass
class Profile:
    kernels: dict[str, KernelStat]
    waits: int  # `# wait` lines = completed Stream fence waits; 0 = not recorded


def parse(path: Path) -> Profile:
    """Per-kernel totals and fence waits of one HALO_VK_OP_TIMINGS file. Blank lines and `#`
    comments are skipped (`# wait` counts a fence wait); any other line must be `<kernel> <ns>`
    with a finite ns >= 0."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ProfileError(f"{path}: cannot read: {e.strerror or e}") from e
    stats: dict[str, KernelStat] = {}
    waits = 0
    for no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if line.strip() == WAIT_MARKER:
                waits += 1
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ProfileError(f"{path}:{no}: expected '<kernel> <ns>', got {line!r}")
        name, raw = parts
        try:
            ns = float(raw)
        except ValueError:
            ns = math.nan
        if not math.isfinite(ns) or ns < 0:
            raise ProfileError(f"{path}:{no}: ns must be a finite number >= 0, got {raw!r}")
        s = stats.get(name)
        if s is None:
            stats[name] = KernelStat(name, 1, ns, ns)
        else:
            s.dispatches += 1
            s.total_ns += ns
            s.max_ns = max(s.max_ns, ns)
    if not stats:
        raise ProfileError(f"{path}: no dispatches (was HALO_VK_OP_TIMINGS set for the run?)")
    return Profile(stats, waits)


def _ms(ns: float) -> str:
    return f"{ns / 1e6:.2f}"


def _us(ns: float) -> str:
    return f"{ns / 1e3:.1f}"


def _num(x: float) -> str:
    """Counts stay integers unless --per made them fractional."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def fence_line(waits: int, dispatches: int, per: float) -> str:
    if not waits:
        return "fence waits: not recorded (no '# wait' lines; profile predates the markers)."
    unit = f" per {_num(per)}" if per != 1 else ""
    return f"{_num(waits / per)} fence waits{unit}, {dispatches / waits:.1f} dispatches per wait."


def summary_rows(stats: dict[str, KernelStat], per: float) -> list[dict]:
    total = sum(s.total_ns for s in stats.values())
    rows = []
    for s in sorted(stats.values(), key=lambda s: (-s.total_ns, s.kernel)):
        rows.append(
            {
                "kernel": s.kernel,
                "dispatches": s.dispatches / per,
                "total_ns": s.total_ns / per,
                "avg_ns": s.avg_ns,
                "max_ns": s.max_ns,
                "share_pct": 100.0 * s.total_ns / total if total else 0.0,
            }
        )
    return rows


def render_summary(path: Path, prof: Profile, per: float, top: int) -> str:
    stats = prof.kernels
    rows = summary_rows(stats, per)
    total_ns = sum(r["total_ns"] for r in rows)
    total_d = sum(r["dispatches"] for r in rows)
    matvec_ns = sum(r["total_ns"] for r in rows if r["kernel"].startswith("matvec_"))
    unit = f" per {_num(per)}" if per != 1 else ""
    out = [
        f"# GPU op profile: {path}",
        "",
        f"{_num(total_d)} dispatches, {_ms(total_ns)} ms GPU time{unit}; "
        f"matvec_* {100.0 * matvec_ns / total_ns if total_ns else 0.0:.1f}% of it.",
        fence_line(prof.waits, sum(s.dispatches for s in stats.values()), per),
        "",
        "| kernel | dispatches | total ms | avg us | max us | % GPU |",
        "|---|---|---|---|---|---|",
    ]
    shown = rows if top <= 0 else rows[:top]
    for r in shown:
        out.append(
            f"| {r['kernel']} | {_num(r['dispatches'])} | {_ms(r['total_ns'])} | {_us(r['avg_ns'])} "
            f"| {_us(r['max_ns'])} | {r['share_pct']:.1f} |"
        )
    rest = rows[len(shown):]
    if rest:
        out.append(
            f"| ({len(rest)} more kernels) | {_num(sum(r['dispatches'] for r in rest))} "
            f"| {_ms(sum(r['total_ns'] for r in rest))} | | | {sum(r['share_pct'] for r in rest):.1f} |"
        )
    return "\n".join(out) + "\n"


def compare_rows(
    before: dict[str, KernelStat], after: dict[str, KernelStat], per_before: float, per_after: float
) -> list[dict]:
    rows = []
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name), after.get(name)
        b_ns = b.total_ns / per_before if b else 0.0
        a_ns = a.total_ns / per_after if a else 0.0
        rows.append(
            {
                "kernel": name,
                "status": "new" if b is None else "gone" if a is None else "",
                "before_ns": b_ns,
                "after_ns": a_ns,
                "delta_ns": a_ns - b_ns,
                "delta_pct": (100.0 * (a_ns - b_ns) / b_ns) if b_ns else None,
                "before_dispatches": b.dispatches / per_before if b else 0.0,
                "after_dispatches": a.dispatches / per_after if a else 0.0,
            }
        )
    rows.sort(key=lambda r: (-abs(r["delta_ns"]), r["kernel"]))
    return rows


def _pct(p: float | None) -> str:
    return "n/a" if p is None else f"{p:+.1f}%"


def fence_change(waits_before: int, waits_after: int, per_before: float, per_after: float) -> str:
    missing = [side for side, w in (("before", waits_before), ("after", waits_after)) if not w]
    if missing:
        return f"fence waits: not recorded in {' and '.join(missing)} (no '# wait' lines)."
    wb, wa = waits_before / per_before, waits_after / per_after
    return f"fence waits {_num(wb)} -> {_num(wa)} ({_pct(100.0 * (wa - wb) / wb)})."


def render_compare(
    before_path: Path,
    after_path: Path,
    rows: list[dict],
    per_before: float,
    per_after: float,
    top: int,
    waits: tuple[int, int] = (0, 0),
) -> str:
    tb = sum(r["before_ns"] for r in rows)
    ta = sum(r["after_ns"] for r in rows)
    db = sum(r["before_dispatches"] for r in rows)
    da = sum(r["after_dispatches"] for r in rows)
    norm = ""
    if per_before != 1 or per_after != 1:
        norm = f" (before / {_num(per_before)}, after / {_num(per_after)})"
    out = [
        f"# GPU op profile change: {before_path} -> {after_path}{norm}",
        "",
        f"GPU time {_ms(tb)} -> {_ms(ta)} ms ({_pct(100.0 * (ta - tb) / tb if tb else None)}), "
        f"dispatches {_num(db)} -> {_num(da)} ({_pct(100.0 * (da - db) / db if db else None)}).",
        fence_change(waits[0], waits[1], per_before, per_after),
        "",
        "| kernel | before ms | after ms | change ms | change | dispatches before -> after |",
        "|---|---|---|---|---|---|",
    ]
    shown = rows if top <= 0 else rows[:top]
    for r in shown:
        label = f"{r['kernel']} ({r['status']})" if r["status"] else r["kernel"]
        out.append(
            f"| {label} | {_ms(r['before_ns'])} | {_ms(r['after_ns'])} | {r['delta_ns'] / 1e6:+.2f} "
            f"| {_pct(r['delta_pct'])} | {_num(r['before_dispatches'])} -> {_num(r['after_dispatches'])} |"
        )
    rest = rows[len(shown):]
    if rest:
        rb = sum(r["before_ns"] for r in rest)
        ra = sum(r["after_ns"] for r in rest)
        out.append(
            f"| ({len(rest)} more kernels) | {_ms(rb)} | {_ms(ra)} | {(ra - rb) / 1e6:+.2f} | | "
            f"{_num(sum(r['before_dispatches'] for r in rest))} -> {_num(sum(r['after_dispatches'] for r in rest))} |"
        )
    return "\n".join(out) + "\n"


def _positive(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        v = math.nan
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be a number > 0, got {text!r}")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="per-kernel table of one profile")
    s.add_argument("profile", type=Path)
    s.add_argument("--per", type=_positive, default=1.0, help="divide totals and counts by N")
    c = sub.add_parser("compare", help="per-kernel change between two profiles")
    c.add_argument("before", type=Path)
    c.add_argument("after", type=Path)
    c.add_argument("--per-before", type=_positive, default=1.0)
    c.add_argument("--per-after", type=_positive, default=1.0)
    for p in (s, c):
        p.add_argument("--top", type=int, default=0, help="rows to show (0 = all; the rest are summed)")
        p.add_argument("--json", type=Path, help="also write the rows as JSON here")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "summary":
            prof = parse(args.profile)
            sys.stdout.write(render_summary(args.profile, prof, args.per, args.top))
            data = {
                "profile": str(args.profile),
                "per": args.per,
                "fence_waits": prof.waits / args.per if prof.waits else None,
                "kernels": summary_rows(prof.kernels, args.per),
            }
        else:
            before, after = parse(args.before), parse(args.after)
            rows = compare_rows(before.kernels, after.kernels, args.per_before, args.per_after)
            waits = (before.waits, after.waits)
            sys.stdout.write(
                render_compare(args.before, args.after, rows, args.per_before, args.per_after, args.top, waits)
            )
            data = {
                "before": str(args.before),
                "after": str(args.after),
                "per_before": args.per_before,
                "per_after": args.per_after,
                "fence_waits_before": before.waits / args.per_before if before.waits else None,
                "fence_waits_after": after.waits / args.per_after if after.waits else None,
                "kernels": rows,
            }
    except ProfileError as e:
        print(f"op_profile: {e}", file=sys.stderr)
        return 2
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
