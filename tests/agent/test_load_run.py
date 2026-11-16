"""Contract tests for load/run.py's connection-close report.

A fake halo_load_server (Python) prints `PORT <n>`, writes ConnectionGuard's real warning lines to
stderr and exits 0 on SIGTERM; a fake `k6` on PATH writes a --summary-export JSON with a failed
check. Run: python3 -m unittest discover -s tests/agent (ctest runs it as `test_agent_report`).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "load"))

import run as load_run  # noqa: E402

FAKE_SERVER = textwrap.dedent(
    """\
    #!{python}
    import json, signal, sys, time
    spec = json.load(open(sys.argv[1]))
    print("PORT 1", flush=True)
    for line in spec["stderr"]:
        if line.startswith("hex:"):  # raw bytes, e.g. invalid UTF-8
            sys.stderr.flush()
            sys.stderr.buffer.write(bytes.fromhex(line[4:]) + b"\\n")
            sys.stderr.buffer.flush()
        else:
            sys.stderr.write(line + "\\n")
    sys.stderr.flush()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.05)
    """
)

FAKE_K6 = textwrap.dedent(
    """\
    #!{python}
    import json, sys
    out = sys.argv[sys.argv.index("--summary-export") + 1]
    fails = {fails}
    json.dump({{"metrics": {{}}, "root_group": {{"checks": {{"c": {{"name": "chat 200", "passes": 90,
               "fails": fails}}}}}}}}, open(out, "w"))
    sys.exit(99 if fails else 0)
    """
)


def close_line(deadline: int, peer: int, shed: int, global_: int) -> str:
    """The text-format line ConnectionGuard::scan logs (src/api/conn_guard.cpp)."""
    total = deadline + peer + shed + global_
    return (f"[WARN] api: closed {total} connection(s) while reading a request: {deadline} past the read "
            f"deadline, {peer} over the per-peer limit, {shed} shed under overload, {global_} over the "
            f"global stalled-connection limit")


class CloseTallyTest(unittest.TestCase):
    def test_sums_every_scan_line_per_reason_and_names_only_nonzero_reasons(self) -> None:
        t = load_run.CloseTally()
        for line in [close_line(0, 100, 0, 0), "[INFO] api: listening", close_line(2, 59, 0, 1)]:
            t.feed(line)
        self.assertEqual(t.counts, [2, 159, 0, 1], "per-reason sums over both close lines")
        self.assertEqual(
            t.describe(),
            "server closed 162 connection(s) mid-request: 2 past the read deadline, 159 over the per-peer "
            "limit, 1 over the global stalled-connection limit (2 '[WARN] api: closed ...' line(s) in its "
            "stderr above)",
        )

    def test_json_log_format_lines_are_tallied_too(self) -> None:
        t = load_run.CloseTally()
        msg = close_line(0, 0, 4, 0).removeprefix("[WARN] api: ")
        t.feed(json.dumps({"ts_ms": 1, "level": "WARN", "component": "api", "msg": msg}))
        self.assertEqual(t.counts, [0, 0, 4, 0], f"HALO_LOG_FORMAT=json line {msg!r} should count as 4 shed")

    def test_no_close_lines_means_no_report_line(self) -> None:
        t = load_run.CloseTally()
        t.feed("[WARN] api: something else entirely")
        self.assertIsNone(t.describe(), "a run without resets must not print a close line")

    def test_pattern_matches_the_format_string_in_conn_guard_cpp(self) -> None:
        src = (REPO / "src/api/conn_guard.cpp").read_text()
        call = re.search(r'HALO_WARN\("api",\s*((?:"[^"]*"\s*)+),', src)
        self.assertIsNotNone(call, "src/api/conn_guard.cpp: HALO_WARN(\"api\", ...) close line not found")
        fmt = "".join(re.findall(r'"([^"]*)"', call.group(1)))
        self.assertIn("closed {} connection(s)", fmt, f"unexpected HALO_WARN format {fmt!r}")
        sample = fmt.replace("{}", "7", 1)
        for n in (1, 2, 3, 4):
            sample = sample.replace("{}", str(n), 1)
        m = load_run.CLOSE_LINE.search(sample)
        self.assertIsNotNone(m, f"load/run.py CLOSE_LINE no longer matches conn_guard.cpp's format {fmt!r}: "
                                "update the regex and CLOSE_REASONS together with the C++ message")
        self.assertEqual(m.groups(), ("7", "1", "2", "3", "4"), f"group order vs format {fmt!r}")


class RunReportsClosesTest(unittest.TestCase):
    def run_main(self, stderr_lines: list[str], fails: int) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory(prefix=f"halo_load_run_{os.getpid()}_") as d:
            tmp = Path(d)
            server = tmp / "fake_server.py"
            server.write_text(FAKE_SERVER.format(python=sys.executable))
            spec = tmp / "spec.json"
            spec.write_text(json.dumps({"stderr": stderr_lines}))
            k6 = tmp / "bin" / "k6"
            k6.parent.mkdir()
            k6.write_text(FAKE_K6.format(python=sys.executable, fails=fails))
            for f in (server, k6):
                f.chmod(0o755)
            script = tmp / "smoke.js"
            script.write_text("// fake\n")
            out, err = io.StringIO(), io.StringIO()
            env = {"PATH": f"{k6.parent}{os.pathsep}{os.environ.get('PATH', '')}",
                   "HALO_LOAD_REPORT_DIR": str(tmp / "report")}
            with mock.patch.dict(os.environ, env), redirect_stdout(out), redirect_stderr(err):
                rc = load_run.main(["--server", str(server), f"--server-arg={spec}", str(script)])
            return rc, out.getvalue(), err.getvalue()

    def test_failed_run_names_the_reasons_the_server_closed_connections(self) -> None:
        rc, out, err = self.run_main([close_line(0, 100, 0, 0), close_line(0, 59, 0, 0)], fails=3)
        self.assertEqual(rc, 1, f"failed check should fail the run; stdout:\n{out}")
        self.assertIn("check 'chat 200' failed 3 of 93 times", out, f"stdout:\n{out}")
        self.assertIn("server closed 159 connection(s) mid-request: 159 over the per-peer limit", out,
                      f"FAIL report should name the close reason; stdout:\n{out}")
        self.assertIn(close_line(0, 100, 0, 0), err, "the server's stderr must still be passed through")

    def test_invalid_utf8_on_server_stderr_does_not_stop_the_tally(self) -> None:
        rc, out, err = self.run_main(["hex:" + (b"[WARN] api: bad " + bytes([0xFF, 0xFE]) + b" bytes").hex(),
                                      close_line(3, 0, 0, 0)], fails=3)
        self.assertEqual(rc, 1, f"stdout:\n{out}")
        self.assertIn("server closed 3 connection(s) mid-request: 3 past the read deadline", out,
                      f"the close line after an undecodable byte must still be counted; stdout:\n{out}\n"
                      f"stderr:\n{err}")
        self.assertIn("bad " + chr(0xFFFD) * 2 + " bytes", err, "undecodable bytes are passed through as U+FFFD")

    def test_clean_run_prints_no_close_line(self) -> None:
        rc, out, _ = self.run_main(["[INFO] api: listening"], fails=0)
        self.assertEqual(rc, 0, f"stdout:\n{out}")
        self.assertNotIn("server closed", out, f"no resets, so no close line; stdout:\n{out}")


if __name__ == "__main__":
    unittest.main()
