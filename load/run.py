#!/usr/bin/env python3
"""Run a k6 script against a local halo_load_server it starts itself.

    python3 load/run.py --server build/tests/unit/api/halo_load_server load/smoke.js
    python3 load/run.py --server ... --server-arg=--token-delay-ms=2 load/load.js

Starts the server (the real ApiServer over the FakeEngine, 127.0.0.1, ephemeral port), runs
`k6 run` with BASE_URL pointing at it, stops the server and checks it exited cleanly (under
ASan a leak or memory error at shutdown fails the run). k6's end-of-test summary is written to
$HALO_LOAD_REPORT_DIR (scripts/test-agent sets it to .test-report/load) or ./load-report as
k6-<script>.json; the failed thresholds and checks are printed with the rerun command.

Exit: 0 pass, 1 fail, 77 skipped (k6 not installed; ctest reports SKIPPED), 2 usage.
Never targets anything but the server it started (no production, no third parties).
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SKIP = 77
PORT_TIMEOUT_S = 30
STOP_TIMEOUT_S = 30


def start_server(server: str, server_args: list[str]) -> tuple[subprocess.Popen[str], int]:
    proc = subprocess.Popen([server, *server_args], stdout=subprocess.PIPE, text=True)
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
            return proc, int(line.split()[1])
    proc.kill()
    proc.wait()
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="path to the halo_load_server binary")
    ap.add_argument("--server-arg", action="append", default=[], help="extra server argument (repeatable)")
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
                        *(f"--server-arg={a}" for a in args.server_arg), str(script)])

    try:
        proc, port = start_server(args.server, args.server_arg)
    except (OSError, RuntimeError) as e:
        print(f"load: FAIL, could not start the server: {e}\nrerun: {rerun}")
        return 1

    env = {**os.environ, "BASE_URL": f"http://127.0.0.1:{port}", "K6_NO_USAGE_REPORT": "true"}
    try:
        k6_rc = subprocess.run(
            [k6, "run", "--quiet", "--no-usage-report", "--summary-export", str(summary_path), str(script)],
            env=env, check=False,
        ).returncode
    finally:
        server_rc = stop_server(proc)

    problems: list[str] = []
    if summary_path.exists():
        problems += failures_from_summary(json.loads(summary_path.read_text()))
    if k6_rc != 0 and not problems:
        problems.append(f"k6 exited {k6_rc} (script error or aborted run; see the k6 output above)")
    if server_rc != 0:
        problems.append(f"halo_load_server exited {server_rc} after SIGTERM, expected 0 (its stderr is above)")

    if problems:
        print(f"load: FAIL {script.name} against halo_load_server")
        for p in problems:
            print(f"  - {p}")
        print(f"summary: {summary_path}\nrerun: {rerun}")
        return 1
    print(f"load: PASS {script.name}, summary {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
