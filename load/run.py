#!/usr/bin/env python3
"""Run a k6 script against a local halo_load_server it starts itself.

    python3 load/run.py --server build/tests/unit/api/halo_load_server load/smoke.js
    python3 load/run.py --server ... --server-arg=--token-delay-ms=2 load/load.js
    python3 load/run.py --server build-asan/... --budget-scale=20 load/smoke.js   # sanitizer/Debug build

Starts the server (the real ApiServer over the FakeEngine, 127.0.0.1, ephemeral port), runs
`k6 run` with BASE_URL pointing at it, stops the server and checks it exited cleanly (under
ASan a leak or memory error at shutdown fails the run). k6's end-of-test summary is written to
$HALO_LOAD_REPORT_DIR (scripts/test-agent sets it to .test-report/load) or ./load-report as
k6-<script>.json; the failed thresholds and checks are printed with the rerun command. The
server's stderr is passed through, and the connections its ConnectionGuard closed mid-request are
summed per reason ('[WARN] api: closed N connection(s) ...', src/api/conn_guard.cpp) and printed
with the result, so a run failed by resets says why without reading the whole log.
--budget-scale multiplies the script's latency budgets (HALO_LOAD_BUDGET_SCALE): 1 for an optimized
build, larger for sanitizer/Debug builds (CMake picks it per build, tests/unit/api/CMakeLists.txt).

Exit: 0 pass, 1 fail, 77 skipped (k6 not installed; ctest reports SKIPPED), 2 usage.
Never targets anything but the server it started (no production, no third parties).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SKIP = 77
PORT_TIMEOUT_S = 30
STOP_TIMEOUT_S = 30

# ConnectionGuard::scan's warning (src/api/conn_guard.cpp), text or JSON log format alike.
CLOSE_LINE = re.compile(
    r"closed (\d+) connection\(s\) while reading a request: (\d+) past the read deadline, (\d+) over the "
    r"per-peer limit, (\d+) shed under overload, (\d+) over the global stalled-connection limit"
)
CLOSE_REASONS = ("past the read deadline", "over the per-peer limit", "shed under overload",
                 "over the global stalled-connection limit")


class CloseTally:
    """Connections the server closed mid-request, summed per reason over its log lines."""

    def __init__(self) -> None:
        self.counts = [0] * len(CLOSE_REASONS)
        self.lines = 0

    def feed(self, line: str) -> None:
        m = CLOSE_LINE.search(line)
        if m is None:
            return
        self.lines += 1
        for i, n in enumerate(m.groups()[1:]):
            self.counts[i] += int(n)

    def total(self) -> int:
        return sum(self.counts)

    def describe(self) -> str | None:
        """One line naming each reason that closed a connection, or None when none were closed."""
        if self.total() == 0:
            return None
        parts = ", ".join(f"{n} {why}" for n, why in zip(self.counts, CLOSE_REASONS) if n)
        return (f"server closed {self.total()} connection(s) mid-request: {parts} "
                f"({self.lines} '[WARN] api: closed ...' line(s) in its stderr above)")


def pump_stderr(stream, tally: CloseTally) -> None:
    """Forward the server's stderr line by line (it stays visible as before) and tally close lines."""
    for line in stream:
        sys.stderr.write(line)
        tally.feed(line)


def start_server(server: str, server_args: list[str],
                 tally: CloseTally) -> tuple[subprocess.Popen[str], int, threading.Thread]:
    # errors="replace": an undecodable stderr byte must not kill the pump (the server would then
    # block on a full stderr pipe).
    proc = subprocess.Popen([server, *server_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            errors="replace")
    assert proc.stderr is not None
    pump = threading.Thread(target=pump_stderr, args=(proc.stderr, tally), daemon=True)
    pump.start()
    assert proc.stdout is not None
    deadline = time.monotonic() + PORT_TIMEOUT_S
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], max(0.0, deadline - time.monotonic()))
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:  # EOF: the server exited before printing its port
            break
        if line.startswith("PORT "):
            return proc, int(line.split()[1]), pump
    proc.kill()
    proc.wait()
    pump.join(timeout=STOP_TIMEOUT_S)
    raise RuntimeError(
        f"{server} did not print 'PORT <n>' within {PORT_TIMEOUT_S}s (exit code {proc.returncode}); "
        "its stderr is above"
    )


def stop_server(proc: subprocess.Popen[str]) -> int:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"load: server ignored SIGTERM for {STOP_TIMEOUT_S}s and was killed", file=sys.stderr)
            return -9
    return proc.returncode


def failures_from_summary(summary: dict) -> list[str]:
    """Failed thresholds and checks from a k6 --summary-export JSON, one readable line each."""
    out: list[str] = []
    for name, metric in sorted(summary.get("metrics", {}).items()):
        for expr, ok in sorted((metric.get("thresholds") or {}).items()):
            # --summary-export marks a threshold true when it was crossed (failed).
            if ok:
                values = {k: v for k, v in metric.items() if k != "thresholds"}
                out.append(f"threshold {name} '{expr}' failed: actual {json.dumps(values, sort_keys=True)}")

    def walk(group: dict, path: str) -> None:
        for check in (group.get("checks") or {}).values():
            if check.get("fails", 0):
                out.append(
                    f"check '{path}{check.get('name')}' failed {check['fails']} of "
                    f"{check['fails'] + check.get('passes', 0)} times"
                )
        for sub in (group.get("groups") or {}).values():
            walk(sub, f"{path}{sub.get('name')} > ")

    walk(summary.get("root_group") or {}, "")
    return out


def budget_scale(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number in [1, 1000], got '{text}'") from None
    if not 1.0 <= value <= 1000.0:  # also rejects nan/inf: a budget must stay a real bound
        raise argparse.ArgumentTypeError(f"expected a number in [1, 1000], got '{text}'")
    return value


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="path to the halo_load_server binary")
    ap.add_argument("--server-arg", action="append", default=[], help="extra server argument (repeatable)")
    ap.add_argument("--budget-scale", type=budget_scale, default=1.0,
                    help="multiply the script's latency budgets (1 = optimized build; sanitizer/Debug builds need more)")
    ap.add_argument("script", help="k6 script, e.g. load/smoke.js")
    args = ap.parse_args(argv)

    k6 = shutil.which("k6")
    if k6 is None:
        print("load: SKIPPED, k6 is not on PATH (https://grafana.com/docs/k6/latest/set-up/install-k6/)")
        return SKIP

    script = Path(args.script).resolve()
    report_dir = Path(os.environ.get("HALO_LOAD_REPORT_DIR", "load-report")).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"k6-{script.stem}.json"
    summary_path.unlink(missing_ok=True)
    rerun = shlex.join([sys.executable, str(Path(__file__).resolve()), "--server", str(Path(args.server).resolve()),
                        *(f"--server-arg={a}" for a in args.server_arg), f"--budget-scale={args.budget_scale:g}",
                        str(script)])

    tally = CloseTally()
    try:
        proc, port, pump = start_server(args.server, args.server_arg, tally)
    except (OSError, RuntimeError) as e:
        print(f"load: FAIL, could not start the server: {e}\nrerun: {rerun}")
        return 1

    env = {**os.environ, "BASE_URL": f"http://127.0.0.1:{port}", "K6_NO_USAGE_REPORT": "true",
           "HALO_LOAD_BUDGET_SCALE": f"{args.budget_scale:g}"}
    try:
        k6_rc = subprocess.run(
            [k6, "run", "--quiet", "--no-usage-report", "--summary-export", str(summary_path), str(script)],
            env=env, check=False,
        ).returncode
    finally:
        server_rc = stop_server(proc)
        pump.join(timeout=STOP_TIMEOUT_S)  # EOF once the server exited: every line is tallied

    problems: list[str] = []
    if summary_path.exists():
        problems += failures_from_summary(json.loads(summary_path.read_text()))
    if k6_rc != 0 and not problems:
        problems.append(f"k6 exited {k6_rc} (script error or aborted run; see the k6 output above)")
    if server_rc != 0:
        problems.append(f"halo_load_server exited {server_rc} after SIGTERM, expected 0 (its stderr is above)")

    closed = tally.describe()
    if problems:
        print(f"load: FAIL {script.name} against halo_load_server")
        for p in problems:
            print(f"  - {p}")
        if closed is not None:
            print(closed)
        print(f"summary: {summary_path}\nrerun: {rerun}")
        return 1
    print(f"load: PASS {script.name}, summary {summary_path}")
    if closed is not None:
        print(closed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
