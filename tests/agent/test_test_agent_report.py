"""Contract tests for scripts/test-agent and scripts/test_agent_report.py.

The fixtures are real ctest 4.2 output (--output-junit, --show-only=json-v1) from injected
tests: a failing EXPECT_EQ, a segfault, a hang hit by --timeout, a GTEST_SKIP, and two passes,
with the checkout path rewritten to /src. Run: python3 -m unittest discover -s tests/agent
(ctest also runs it as `test_agent_report`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(REPO / "scripts"))

import test_agent_report as rep  # noqa: E402


class ReportFromCtestOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # The failing assertion's source file, so the summary can quote the line.
        src = self.root / "src" / "tests" / "unit" / "test_core.cpp"
        src.parent.mkdir(parents=True)
        src.write_text("\n" * 26 + 'TEST(Inject, FailsEq) { int got = 3; EXPECT_EQ(got, 4) << "context: id=7"; }\n')
        # The fixtures name files under /src: rewrite them to this temp source root.
        for name in ("ctest-junit.xml", "ctest-tests.json"):
            text = (FIXTURES / name).read_text().replace("/src/", f"{self.root / 'src'}/")
            (self.root / name).write_text(text)
        self.report = self.root / "report"

    def run_report(self, *extra: str) -> tuple[int, str, dict]:
        code = rep.main(["--report-dir", str(self.report), "--build-dir", "build",
                         "--source-root", str(self.root / "src"),
                         "--junit", str(self.root / "ctest-junit.xml"),
                         "--tests-json", str(self.root / "ctest-tests.json"), *extra])
        summary = (self.report / "summary.md").read_text()
        results = json.loads((self.report / "results.json").read_text())
        return code, summary, results

    def section(self, summary: str, test: str) -> str:
        start = summary.find(f"## FAILED {test} ")
        self.assertNotEqual(start, -1, f"summary.md has no section for {test}:\n{summary}")
        end = summary.find("\n## ", start + 1)
        return summary[start:end if end != -1 else None]

    def test_counts_and_verdict_cover_every_outcome(self):
        code, summary, results = self.run_report("--exit-code", "8")
        self.assertEqual(code, 1, f"3 failing tests must exit 1, got {code}")
        self.assertTrue(summary.startswith("# test-agent: FAIL\n"), f"verdict line wrong:\n{summary[:200]}")
        self.assertIn("6 tests: 2 passed, 3 failed, 1 skipped.", summary)
        self.assertEqual(results["counts"], {"passed": 2, "failed": 3, "skipped": 1}, results)

    def test_assertion_failure_names_file_line_source_and_expected_vs_actual(self):
        _, summary, results = self.run_report("--exit-code", "8")
        sec = self.section(summary, "Inject.FailsEq")
        self.assertIn("assertion at `tests/unit/test_core.cpp:27`", sec, f"file:line missing:\n{sec}")
        self.assertIn('EXPECT_EQ(got, 4) << "context: id=7"', sec, f"source line not quoted:\n{sec}")
        self.assertIn("Expected equality of these values:\n  got\n    Which is: 3\n  4\ncontext: id=7", sec,
                      f"gtest expected/actual message not kept verbatim:\n{sec}")
        failed = {t["name"]: t for t in results["tests"] if t["status"] == "failed"}
        self.assertEqual(failed["Inject.FailsEq"]["assertions"][0]["line"], 27, failed["Inject.FailsEq"])

    def test_rerun_commands_select_only_the_failing_test(self):
        _, summary, _ = self.run_report("--exit-code", "8")
        sec = self.section(summary, "Inject.FailsEq")
        self.assertIn(r"ctest --test-dir build -R '^Inject\.FailsEq$' --output-on-failure", sec, sec)
        self.assertIn("tests/test_core --gtest_filter=Inject.FailsEq --gtest_also_run_disabled_tests", sec,
                      f"direct gtest command (from ctest's listing) missing:\n{sec}")

    def test_crash_and_timeout_get_a_hint_and_the_output_tail(self):
        _, summary, _ = self.run_report("--exit-code", "8")
        crash = self.section(summary, "Inject.Crashes")
        self.assertIn("(SEGFAULT)", crash)
        self.assertIn("SIGSEGV", crash, f"crash hint missing:\n{crash}")
        self.assertIn("[ RUN      ] Inject.Crashes", crash, f"output tail missing:\n{crash}")
        self.assertNotIn("Running main() from", crash, "gtest_main banner noise should be dropped")
        hang = self.section(summary, "Inject.Hangs")
        self.assertIn("--timeout", hang, f"timeout hint missing:\n{hang}")

    def test_skips_are_grouped_by_gtest_skip_reason(self):
        _, summary, results = self.run_report("--exit-code", "8")
        self.assertIn("## Skipped\n\n- 1 x needs HALO_REF_DIR", summary, summary)
        skipped = [t for t in results["tests"] if t["status"] == "skipped"]
        self.assertEqual(skipped[0]["skip_reason"], "needs HALO_REF_DIR", skipped)
        self.assertEqual(skipped[0]["assertions"], [], f"a GTEST_SKIP is not a failed assertion: {skipped}")

    def test_nonzero_ctest_exit_without_failing_cases_is_still_a_failure(self):
        junit = self.root / "ctest-junit.xml"
        text = junit.read_text()
        # Keep only the two passing Error.* cases.
        cut = text.index('\t<testcase name="Inject.FailsEq"')
        junit.write_text(text[:cut] + "</testsuite>\n")
        code, summary, results = self.run_report("--exit-code", "8")
        self.assertEqual(code, 1, "ctest exit 8 (e.g. a test that could not start) must not report PASS")
        self.assertIn("ctest exited 8 without a failing test case", summary, summary)
        self.assertEqual(results["verdict"], "fail")

    def test_all_green_is_pass(self):
        junit = self.root / "ctest-junit.xml"
        text = junit.read_text()
        junit.write_text(text[:text.index('\t<testcase name="Inject.FailsEq"')] + "</testsuite>\n")
        code, summary, results = self.run_report("--exit-code", "0")
        self.assertEqual(code, 0, summary)
        self.assertTrue(summary.startswith("# test-agent: PASS\n"), summary)
        self.assertEqual(results["counts"], {"passed": 2, "failed": 0, "skipped": 0})

    def test_missing_junit_is_not_a_pass(self):
        (self.root / "ctest-junit.xml").unlink()
        code, summary, _ = self.run_report("--exit-code", "0")
        self.assertEqual(code, 1, "no JUnit file means no tests ran; must not exit 0")
        self.assertIn("FAIL (no tests ran)", summary)


class ReportFromBuildFailure(unittest.TestCase):
    def test_compiler_and_linker_errors_are_listed_with_file_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "lib").mkdir(parents=True)
            (root / "src" / "lib" / "a.cpp").write_text("int f() {\n  return undefined_name;\n}\n")
            log = root / "run.log"
            log.write_text(textwrap.dedent(f"""\
                [1/3] Building CXX object a.o
                FAILED: a.o
                {root}/src/lib/a.cpp:2:10: error: use of undeclared identifier 'undefined_name'
                {root}/src/lib/a.cpp:2:10: error: use of undeclared identifier 'undefined_name'
                1 error generated.
                ld.lld: error: undefined symbol: halo::g()
                """))
            code = rep.main(["--report-dir", str(root / "r"), "--source-root", str(root / "src"),
                             "--build-log", str(log)])
            summary = (root / "r" / "summary.md").read_text()
            self.assertEqual(code, 1)
            self.assertTrue(summary.startswith("# test-agent: FAIL (build)"), summary)
            self.assertIn("- `lib/a.cpp:2`: use of undeclared identifier 'undefined_name'", summary, summary)
            self.assertIn("source: `return undefined_name;`", summary, summary)
            self.assertEqual(summary.count("undeclared identifier"), 1, f"duplicate errors not folded:\n{summary}")
            self.assertIn("ld.lld: error: undefined symbol: halo::g()", summary, summary)


class Wrapper(unittest.TestCase):
    """scripts/test-agent against stub cmake/ctest on PATH (no real build)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.log"
        self.build = self.root / "build"
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text("")

    def stub(self, name: str, body: str):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + f'echo "{name} $*" >> {self.calls}\n' + body)
        path.chmod(0o755)

    def run_wrapper(self, *args: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}"}
        return subprocess.run([str(REPO / "scripts" / "test-agent"), f"--build-dir={self.build}", *args],
                              env=env, capture_output=True, text=True, timeout=60)

    def ctest_stub(self, junit: str, code: int) -> str:
        # Writes the fixture JUnit to the --output-junit path, mimics ctest's exit code.
        return textwrap.dedent(f"""\
            for a in "$@"; do [[ $a == --show-only=json-v1 ]] && {{ cat {FIXTURES / 'ctest-tests.json'}; exit 0; }}; done
            prev=""; for a in "$@"; do [[ $prev == --output-junit ]] && cp {junit} "$a"; prev=$a; done
            exit {code}
            """)

    def test_passes_ctest_args_after_separator_and_reports_failures(self):
        self.stub("cmake", "exit 0\n")
        self.stub("ctest", self.ctest_stub(str(FIXTURES / "ctest-junit.xml"), 8))
        report = self.root / "report"
        p = self.run_wrapper(f"--report-dir={report}", "--", "-R", "^Inject")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        calls = self.calls.read_text()
        run_line = [l for l in calls.splitlines() if "--output-junit" in l][0]
        self.assertIn("-R ^Inject", run_line, f"ctest args after -- not forwarded:\n{calls}")
        self.assertIn(f"--output-junit {report}/junit.xml", run_line, "JUnit path must be absolute")
        self.assertIn("## FAILED Inject.FailsEq (Failed)", p.stdout, "summary.md should be printed")
        for f in ("summary.md", "results.json", "junit.xml", "run.log", "tests.json"):
            self.assertTrue((report / f).is_file(), f"{f} missing from the report dir")

    def test_runs_without_ctest_args_under_set_u(self):
        self.stub("cmake", "exit 0\n")
        passing = self.root / "pass.xml"
        text = (FIXTURES / "ctest-junit.xml").read_text()
        passing.write_text(text[:text.index('\t<testcase name="Inject.FailsEq"')] + "</testsuite>\n")
        self.stub("ctest", self.ctest_stub(str(passing), 0))
        p = self.run_wrapper(f"--report-dir={self.root / 'report'}")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("# test-agent: PASS", p.stdout)

    def test_build_failure_reports_compiler_errors_and_skips_ctest(self):
        self.stub("cmake", f'echo "{REPO}/src/x.cpp:3:1: error: expected \';\'"; exit 1\n')
        self.stub("ctest", "exit 0\n")
        p = self.run_wrapper(f"--report-dir={self.root / 'report'}")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("# test-agent: FAIL (build)", p.stdout)
        self.assertIn("- `src/x.cpp:3`: expected ';'", p.stdout, p.stdout)
        self.assertNotIn("ctest", self.calls.read_text(), "ctest must not run after a failed build")

    def test_report_dir_keeps_unrelated_files(self):
        self.stub("cmake", "exit 0\n")
        self.stub("ctest", self.ctest_stub(str(FIXTURES / "ctest-junit.xml"), 8))
        report = self.root / "report"
        report.mkdir()
        (report / "keep.txt").write_text("not ours")
        (report / "summary.md").write_text("stale")
        self.run_wrapper(f"--report-dir={report}")
        self.assertTrue((report / "keep.txt").is_file(), "test-agent deleted a file it does not own")
        self.assertNotEqual((report / "summary.md").read_text(), "stale", "stale summary.md not replaced")


if __name__ == "__main__":
    unittest.main()
