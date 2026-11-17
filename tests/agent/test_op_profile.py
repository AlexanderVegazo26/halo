"""Contract tests for scripts/op_profile.py (HALO_VK_OP_TIMINGS aggregation, next.md step 3).

Profiles are temp files in the `<kernel> <ns>` + `# wait` format Stream::wait() appends
(backends/vulkan/src/stream.cpp; VkStream.OpTimingsProfileMarksEachFenceWaitAfterItsDispatches
pins the writer side on the GPU). Run: python3 -m unittest discover -s tests/agent
(ctest runs it as `test_agent_report`).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import op_profile as op  # noqa: E402


class OpProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                rc = op.main(list(argv))
            except SystemExit as e:  # argparse usage errors
                rc = int(e.code or 0)
        return rc, out.getvalue(), err.getvalue()

    def test_summary_totals_each_kernel_and_ranks_by_gpu_time(self) -> None:
        p = self.write(
            "p.txt",
            "rms_norm 2000\nmatvec_q5_k 1000000\nmatvec_q5_k 3000000\n\nattention 1000000\n",
        )
        rc, out, err = self.run_cli("summary", str(p), "--json", str(self.dir / "s.json"))
        self.assertEqual(rc, 0, f"summary should succeed; stderr: {err}")
        self.assertIn("4 dispatches, 5.00 ms GPU time;", out, out)
        self.assertIn("matvec_* 80.0% of it", out, f"matvec share of 4 ms in 5.002 ms; got:\n{out}")
        lines = [l for l in out.splitlines() if l.startswith("| ") and "kernel" not in l]
        self.assertEqual(
            [l.split(" | ")[0][2:] for l in lines],
            ["matvec_q5_k", "attention", "rms_norm"],
            f"rows must be ordered by total GPU time, largest first; got:\n{out}",
        )
        self.assertEqual(lines[0], "| matvec_q5_k | 2 | 4.00 | 2000.0 | 3000.0 | 80.0 |", out)
        data = json.loads((self.dir / "s.json").read_text())
        q5 = data["kernels"][0]
        self.assertEqual((q5["kernel"], q5["dispatches"], q5["total_ns"], q5["max_ns"]),
                         ("matvec_q5_k", 2, 4e6, 3e6), f"JSON row: {q5}")

    def test_per_divides_totals_and_counts_but_not_averages(self) -> None:
        p = self.write("p.txt", "eltwise 1000\neltwise 3000\neltwise 2000\neltwise 2000\n")
        rc, out, err = self.run_cli("summary", str(p), "--per", "4")
        self.assertEqual(rc, 0, err)
        self.assertIn("1 dispatches, 0.00 ms GPU time per 4", out, out)
        self.assertIn("| eltwise | 1 | 0.00 | 2.0 | 3.0 | 100.0 |", out,
                      f"--per 4 divides count and total, avg/max stay per dispatch; got:\n{out}")

    def test_top_sums_the_rest_into_one_row(self) -> None:
        p = self.write("p.txt", "a 4000000\nb 3000000\nc 2000000\nd 1000000\n")
        rc, out, _ = self.run_cli("summary", str(p), "--top", "2")
        self.assertEqual(rc, 0)
        self.assertIn("| (2 more kernels) | 2 | 3.00 | | | 30.0 |", out, out)
        self.assertNotIn("| c |", out, f"kernel c is beyond --top 2; got:\n{out}")

    def test_compare_orders_by_change_and_marks_new_and_gone_kernels(self) -> None:
        before = self.write("b.txt", "matvec_q5_k 3000000\ngdn_gates 10000\nrms_norm 500000\n")
        after = self.write("a.txt", "matvec_q5_k 2000000\nrms_norm 500000\nmatvec_q8_0_gates 20000\n")
        rc, out, err = self.run_cli("compare", str(before), str(after), "--json", str(self.dir / "c.json"))
        self.assertEqual(rc, 0, err)
        self.assertIn("GPU time 3.51 -> 2.52 ms (-28.2%), dispatches 3 -> 3 (+0.0%)", out, out)
        rows = [l for l in out.splitlines() if l.startswith("| ") and "before ms" not in l]
        self.assertEqual(
            rows[0], "| matvec_q5_k | 3.00 | 2.00 | -1.00 | -33.3% | 1 -> 1 |",
            f"largest absolute change first; got:\n{out}",
        )
        self.assertEqual(
            [r.split(" | ")[0][2:].split(" (")[0] for r in rows],
            ["matvec_q5_k", "matvec_q8_0_gates", "gdn_gates", "rms_norm"],
            f"rows by |change| (a +0.02 ms gain outranks a -0.01 ms loss); got:\n{out}",
        )
        self.assertIn("| matvec_q8_0_gates (new) | 0.00 | 0.02 | +0.02 | n/a | 0 -> 1 |", out, out)
        self.assertIn("| gdn_gates (gone) | 0.01 | 0.00 | -0.01 | -100.0% | 1 -> 0 |", out, out)
        self.assertIn("| rms_norm | 0.50 | 0.50 | +0.00 | +0.0% | 1 -> 1 |", out, out)
        data = json.loads((self.dir / "c.json").read_text())
        status = {r["kernel"]: r["status"] for r in data["kernels"]}
        self.assertEqual(status, {"matvec_q5_k": "", "rms_norm": "", "gdn_gates": "gone",
                                  "matvec_q8_0_gates": "new"}, f"JSON statuses: {status}")

    def test_compare_normalizes_each_side_by_its_own_per(self) -> None:
        before = self.write("b.txt", "eltwise 1000000\n" * 4)  # 4 ms over 2 tokens
        after = self.write("a.txt", "eltwise 1000000\n" * 6)  # 6 ms over 6 tokens
        rc, out, _ = self.run_cli("compare", str(before), str(after), "--per-before", "2", "--per-after", "6")
        self.assertEqual(rc, 0)
        self.assertIn("| eltwise | 2.00 | 1.00 | -1.00 | -50.0% | 2 -> 1 |", out,
                      f"per-token: 2 ms/token before, 1 ms/token after; got:\n{out}")

    def test_the_recorded_2026_09_29_profile_matches_process_md(self) -> None:
        """process.md section 13.2's hand-made table came from this file."""
        rc, out, _ = self.run_cli("summary", str(REPO / "process_tools" / "op_timings.txt"))
        self.assertEqual(rc, 0)
        self.assertIn("9396 dispatches, 4299.73 ms GPU time; matvec_* 96.0% of it.", out, out[:400])
        self.assertIn("fence waits: not recorded", out,
                      f"the 2026-09-29 file predates the '# wait' markers; got:\n{out[:400]}")
        self.assertIn("| matvec_q5_k | 1719 | 1866.75 | 1086.0 |", out, out[:800])

    def test_summary_counts_fence_waits_and_dispatches_per_wait(self) -> None:
        p = self.write(
            "p.txt",
            "matvec_q5_k 1000\nrms_norm 10\nmatvec_q5_k 1000\n# wait\n"
            "  # a comment line is skipped\nrms_norm 10\n# wait\n# wait\n",
        )
        rc, out, err = self.run_cli("summary", str(p), "--json", str(self.dir / "s.json"))
        self.assertEqual(rc, 0, err)
        self.assertIn("4 dispatches, 0.00 ms GPU time;", out, f"'#' lines are not dispatches; got:\n{out}")
        self.assertIn("3 fence waits, 1.3 dispatches per wait.", out,
                      f"3 '# wait' lines over 4 dispatches; got:\n{out}")
        self.assertNotIn("comment", out, f"the comment is not a kernel row; got:\n{out}")
        self.assertEqual(json.loads((self.dir / "s.json").read_text())["fence_waits"], 3)
        rc, out, _ = self.run_cli("summary", str(p), "--per", "2")
        self.assertEqual(rc, 0)
        self.assertIn("1.5 fence waits per 2, 1.3 dispatches per wait.", out,
                      f"--per divides the wait count, not the dispatches-per-wait ratio; got:\n{out}")

    def test_compare_reports_the_fence_wait_change_per_token(self) -> None:
        before = self.write("b.txt", "eltwise 1000\n# wait\n" * 8)  # 8 waits over 2 tokens
        after = self.write("a.txt", "eltwise 1000\neltwise 1000\n# wait\n" * 3)  # 3 waits over 3 tokens
        rc, out, err = self.run_cli("compare", str(before), str(after), "--per-before", "2", "--per-after", "3",
                                    "--json", str(self.dir / "c.json"))
        self.assertEqual(rc, 0, err)
        self.assertIn("fence waits 4 -> 1 (-75.0%).", out, f"4 waits/token before, 1 after; got:\n{out}")
        data = json.loads((self.dir / "c.json").read_text())
        self.assertEqual((data["fence_waits_before"], data["fence_waits_after"]), (4, 1), f"JSON: {data}")

    def test_compare_says_which_side_has_no_wait_markers(self) -> None:
        old = self.write("old.txt", "eltwise 1000\n")
        new = self.write("new.txt", "eltwise 1000\n# wait\n")
        for argv, needle in [
            ((old, new), "not recorded in before"),
            ((new, old), "not recorded in after"),
            ((old, old), "not recorded in before and after"),
        ]:
            with self.subTest(argv=[a.name for a in argv]):
                rc, out, _ = self.run_cli("compare", *map(str, argv), "--json", str(self.dir / "c.json"))
                self.assertEqual(rc, 0)
                self.assertIn(f"fence waits: {needle} (no '# wait' lines).", out, out)
                data = json.loads((self.dir / "c.json").read_text())
                self.assertIsNone(
                    data["fence_waits_before"] if "before" in needle else data["fence_waits_after"],
                    f"an unrecorded side is null in the JSON, not 0: {data}",
                )

    def test_summary_counts_transfer_fences_and_their_bytes_apart_from_fence_waits(self) -> None:
        p = self.write(
            "p.txt",
            "# transfer 4096\nrms_norm 10\n# wait\n# transfer 1024\n# transfer 1024\n# wait\n",
        )
        rc, out, err = self.run_cli("summary", str(p), "--json", str(self.dir / "s.json"))
        self.assertEqual(rc, 0, err)
        self.assertIn("2 fence waits, 0.5 dispatches per wait.", out,
                      f"'# transfer' lines are not Stream fence waits; got:\n{out}")
        self.assertIn("3 transfer fences (6.0 KiB staged).", out, f"3 transfers of 4096+1024+1024 B; got:\n{out}")
        self.assertNotIn("| transfer", out, f"a transfer marker is not a kernel row; got:\n{out}")
        data = json.loads((self.dir / "s.json").read_text())
        self.assertEqual((data["transfer_fences"], data["transfer_bytes"]), (3, 6144), f"JSON: {data}")
        rc, out, _ = self.run_cli("summary", str(p), "--per", "3")
        self.assertEqual(rc, 0)
        self.assertIn("1 transfer fences per 3 (2.0 KiB staged per 3).", out,
                      f"--per divides the transfer count and bytes; got:\n{out}")
        old = self.write("old.txt", "eltwise 1\n# wait\n")
        rc, out, _ = self.run_cli("summary", str(old), "--json", str(self.dir / "o.json"))
        self.assertEqual(rc, 0)
        self.assertIn("transfer fences: not recorded (no '# transfer' lines;", out, out)
        self.assertIsNone(json.loads((self.dir / "o.json").read_text())["transfer_fences"],
                          "an unrecorded count is null in the JSON, not 0")

    def test_compare_reports_the_transfer_fence_change_per_token(self) -> None:
        before = self.write("b.txt", "eltwise 1000\n" + "# transfer 20480\n" * 6)  # 6 over 2 tokens
        after = self.write("a.txt", "eltwise 1000\n# transfer 4\n")  # 1 over 1 token
        rc, out, err = self.run_cli("compare", str(before), str(after), "--per-before", "2",
                                    "--json", str(self.dir / "c.json"))
        self.assertEqual(rc, 0, err)
        self.assertIn("transfer fences 3 -> 1 (-66.7%).", out, f"3 transfers/token before, 1 after; got:\n{out}")
        data = json.loads((self.dir / "c.json").read_text())
        self.assertEqual((data["transfer_fences_before"], data["transfer_fences_after"]), (3, 1), f"JSON: {data}")
        rc, out, _ = self.run_cli("compare", str(after), str(self.write("o.txt", "eltwise 1\n")),
                                  "--json", str(self.dir / "c.json"))
        self.assertEqual(rc, 0)
        self.assertIn("transfer fences: not recorded in after (no '# transfer' lines).", out, out)
        self.assertIsNone(json.loads((self.dir / "c.json").read_text())["transfer_fences_after"])

    def test_malformed_transfer_marker_is_named_with_file_and_line(self) -> None:
        for marker in ("# transfer", "# transfer x", "# transfer -4", "# transfer 4 5", "# transfer ²"):
            with self.subTest(marker=marker):
                p = self.write("bad.txt", f"eltwise 1\n{marker}\n")
                rc, out, err = self.run_cli("summary", str(p))
                self.assertEqual(rc, 2, f"{marker!r} should exit 2; stdout: {out}")
                self.assertIn(f"{p}:2: expected '# transfer <bytes>'", err, f"{marker!r}: got {err!r}")

    def test_malformed_line_is_named_with_file_and_line(self) -> None:
        for body, needle in [
            ("eltwise 100\nmatvec 1 2\n", ":2: expected '<kernel> <ns>'"),
            ("eltwise abc\n", ":1: ns must be a finite number >= 0, got 'abc'"),
            ("eltwise -5\n", ":1: ns must be a finite number >= 0, got '-5'"),
            ("eltwise nan\n", ":1: ns must be a finite number >= 0, got 'nan'"),
            ("\n\n", ": no dispatches"),
        ]:
            with self.subTest(body=body):
                p = self.write("bad.txt", body)
                rc, out, err = self.run_cli("summary", str(p))
                self.assertEqual(rc, 2, f"{body!r} should exit 2; stdout: {out}")
                self.assertIn(f"{p}{needle}", err, f"{body!r}: stderr should name the problem; got {err!r}")

    def test_missing_file_and_bad_per_exit_2(self) -> None:
        rc, _, err = self.run_cli("summary", str(self.dir / "nope.txt"))
        self.assertEqual(rc, 2)
        self.assertIn("nope.txt: cannot read", err, err)
        p = self.write("p.txt", "eltwise 1\n")
        for bad in ("0", "-1", "x", "inf"):
            with self.subTest(per=bad):
                rc, _, err = self.run_cli("summary", str(p), "--per", bad)
                self.assertEqual(rc, 2, f"--per {bad} is a usage error")
                self.assertIn("must be a number > 0", err, err)


if __name__ == "__main__":
    unittest.main()
