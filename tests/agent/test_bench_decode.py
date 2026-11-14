"""Contract tests for scripts/bench_decode.py (the next.md tok/s sweep).

A fake `halo` (Python) prints `halo run`'s real summary lines (tools/halo/commands.cpp) and logs
every call; a temp file stands in for gpu_busy_percent. Run: python3 -m unittest discover -s
tests/agent (ctest runs it as `test_agent_report`).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import bench_decode as bd  # noqa: E402

FAKE_HALO = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    spec = json.load(open(os.environ["FAKE_HALO_SPEC"]))
    flags = {{k: v for k, v in os.environ.items() if k.startswith("HALO_")}}
    with open(spec["log"], "a") as f:
        f.write(json.dumps({{"argv": sys.argv[1:], "halo_env": flags}}) + "\\n")
    hit = lambda key: [k for k in flags if k in spec.get(key, {{}})]
    if spec.get("fail_env") in flags:
        sys.stderr.write("halo: vulkan: device lost\\n"); sys.exit(4)
    for k in hit("busy_after"):
        open(spec["busy_after"][k], "w").write("100\\n")
    tps = spec.get("default_tps", 10.0)
    if "tps_seq" in spec:  # per-call values, in call order
        tps = spec["tps_seq"][sum(1 for _ in open(spec["log"])) - 1]
    for k in hit("tps_by_env"):
        tps = spec["tps_by_env"][k]
    if "--mtp-draft" in sys.argv and sys.argv[sys.argv.index("--mtp-draft") + 1] == "0":
        tps = spec.get("plain_tps", 8.0)
    out = "391 is the answer"
    for k in hit("output_by_env"):
        out = spec["output_by_env"][k]
    sys.stdout.write(out + "\\n")
    sys.stderr.write("[halo] prompt 17 tokens, generated 128 tokens in 9.31 s (%.2f tok/s decode), finish length\\n" % tps)
    if not ("--mtp-draft" in sys.argv and sys.argv[sys.argv.index("--mtp-draft") + 1] == "0"):
        sys.stderr.write("[halo] drafts 90 accepted of 120 proposed (75.0%)\\n")
    """
)


class BenchDecode(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.halo = self.root / "halo"
        self.halo.write_text(FAKE_HALO.format(python=sys.executable))
        self.halo.chmod(0o755)
        self.busy = self.root / "gpu_busy_percent"
        self.busy.write_text("0\n")
        self.log = self.root / "calls.jsonl"
        self.spec = {"log": str(self.log)}
        self.report = self.root / "report"

    def run_bench(self, *extra, env=None):
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(self.spec))
        argv = ["--halo", str(self.halo), "--model", "/models/m.gguf", "--busy-file", str(self.busy),
                "--busy-samples", "1", "--busy-interval", "0", "--report-dir", str(self.report), *extra]
        # Start from no HALO_* at all: scripts/test-agent exports HALO_*_REPORT_DIR, which would
        # otherwise reach the fake halo and the env assertions.
        outer = {k: v for k, v in os.environ.items() if not k.startswith("HALO_")}
        full_env = {**outer, "FAKE_HALO_SPEC": str(spec_path), **(env or {})}
        err = io.StringIO()
        with mock.patch.dict(os.environ, full_env, clear=True), redirect_stderr(err):
            rc = bd.main(argv)
        data = json.loads((self.report / "bench-decode.json").read_text())
        self.results = {r["name"]: r for r in data["results"]}
        self.meta = data["meta"]
        self.summary = (self.report / "summary.md").read_text()
        self.stderr = err.getvalue()
        return rc

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_busy_gpu_runs_nothing_and_reports_not_measured(self):
        self.busy.write_text("100\n")
        rc = self.run_bench()
        self.assertEqual(rc, 3, f"a busy GPU must exit 3 (not measured); stderr: {self.stderr}")
        self.assertEqual(self.calls(), [], "no halo run may start while gpu_busy_percent is above the limit")
        self.assertTrue(all(r["status"] == "not measured" and r["tps_median"] is None for r in self.results.values()),
                        f"every variant must be 'not measured' with no tok/s: {self.results}")
        self.assertIn("GPU busy 100% > 5%", self.results["baseline"]["note"])
        self.assertIn("GPU BUSY: not measured", self.summary)

    def test_unreadable_busy_file_is_not_an_idle_gpu(self):
        rc = self.run_bench("--busy-file", str(self.root / "missing*"))
        self.assertEqual(rc, 3, "without gpu_busy_percent the GPU is not known to be idle")
        self.assertEqual(self.calls(), [])
        self.assertIn("unreadable", self.results["baseline"]["note"])

    def test_idle_sweep_measures_every_variant_against_the_baseline(self):
        self.spec.update(default_tps=10.0, plain_tps=8.0, tps_by_env={"HALO_MTP_TREE": 12.5})
        rc = self.run_bench()
        self.assertEqual(rc, 0, f"stderr: {self.stderr}\n{self.summary}")
        self.assertEqual([c["argv"][0] for c in self.calls()], ["run"] * len(bd.VARIANTS))
        self.assertEqual(list(self.results), [v.name for v in bd.VARIANTS], "baseline first, then every variant")
        tree = self.results["mtp-tree"]
        self.assertEqual(tree["tps_median"], 12.5)
        self.assertAlmostEqual(tree["vs_baseline_pct"], 25.0)
        self.assertEqual(self.results["plain"]["tps_median"], 8.0)
        self.assertAlmostEqual(self.results["plain"]["vs_baseline_pct"], -20.0)
        self.assertEqual(self.results["plain"]["drafts"], "", "plain decode prints no drafts line")
        self.assertEqual(self.results["baseline"]["drafts"], "drafts 90 accepted of 120 proposed (75.0%)")
        self.assertTrue(all(r["same_output"] for r in self.results.values()), self.results)
        self.assertIn("| mtp-tree | 5 | 12.50 | +25.0% |", self.summary)

    def test_each_variant_runs_the_speed_command_with_only_its_own_flag(self):
        rc = self.run_bench("--variants", "mtp-tree,mtp-draft-3", "-n", "64", "--ctx", "2048",
                            env={"HALO_SUBGROUP_REDUCE": "1", "HALO_KV_TYPE": "q8", "HALO_CONFIG": "/x.toml"})
        self.assertEqual(rc, 0, self.stderr)
        base, tree, deep = self.calls()
        self.assertEqual(base["argv"], ["run", "/models/m.gguf", "-p", bd.DEFAULT_PROMPT, "--raw", "--temperature", "0",
                                        "-n", "64", "--ctx", "2048", "--parallel", "1"])
        self.assertEqual(base["halo_env"], {"HALO_CONFIG": "/x.toml"},
                         "inherited speed flags must be stripped; other HALO_ settings kept")
        self.assertEqual(tree["halo_env"], {"HALO_CONFIG": "/x.toml", "HALO_MTP_TREE": "1"})
        self.assertEqual(deep["argv"][-2:], ["--mtp-draft", "3"])
        self.assertEqual(self.meta["stripped_env"], ["HALO_KV_TYPE", "HALO_SUBGROUP_REDUCE"])
        self.assertIn("removed from the inherited environment: HALO_KV_TYPE, HALO_SUBGROUP_REDUCE", self.summary)

    def test_bitwise_variant_with_different_output_fails(self):
        self.spec["output_by_env"] = {"HALO_FUSE_GEMV": "390"}
        rc = self.run_bench("--variants", "fuse-gemv")
        self.assertEqual(rc, 1, "a bitwise flag that changes greedy output is a correctness bug")
        r = self.results["fuse-gemv"]
        self.assertEqual((r["status"], r["same_output"]), ("failed", False))
        self.assertIn("differs from the baseline", r["note"])
        self.assertIn("| NO |", self.summary)

    def test_lossy_variant_may_change_output(self):
        self.spec["output_by_env"] = {"HALO_KV_FP16": "390"}
        rc = self.run_bench("--variants", "kv-fp16")
        self.assertEqual(rc, 0, self.summary)
        self.assertEqual(self.results["kv-fp16"]["same_output"], False)
        self.assertIn("no (lossy, allowed)", self.summary)

    def test_failed_run_is_reported_with_its_stderr(self):
        self.spec["fail_env"] = "HALO_REPACK"
        rc = self.run_bench("--variants", "repack,ngram")
        self.assertEqual(rc, 1)
        self.assertEqual(self.results["repack"]["status"], "failed")
        self.assertIn("exit 4", self.results["repack"]["note"])
        self.assertIn("device lost", self.results["repack"]["note"])
        self.assertEqual(self.results["ngram"]["status"], "measured", "a failed variant does not stop the sweep")

    def test_gpu_turning_busy_mid_sweep_stops_measuring(self):
        self.spec["busy_after"] = {"HALO_GDN_REG": str(self.busy)}
        rc = self.run_bench("--variants", "gdn-reg,stream-reuse,ngram")
        self.assertEqual(rc, 3)
        self.assertEqual(len(self.calls()), 2, "nothing runs after the GPU turned busy")
        self.assertEqual(self.results["gdn-reg"]["status"], "measured")
        self.assertEqual(self.results["stream-reuse"]["status"], "not measured")
        self.assertEqual(self.results["ngram"]["note"], "skipped after GPU busy")

    def test_repeat_takes_the_median(self):
        self.spec["tps_seq"] = [9.0, 30.0, 11.0]
        rc = self.run_bench("--variants", "baseline", "--repeat", "3")
        self.assertEqual(rc, 0)
        self.assertEqual(self.results["baseline"]["tps"], [9.0, 30.0, 11.0])
        self.assertEqual(self.results["baseline"]["tps_median"], 11.0, "median of the repeats, not mean/first/last")
        self.assertEqual(len(self.calls()), 3)

    def test_unknown_variant_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            bd.select("baseline,warp-drive")
        self.assertIn("warp-drive", str(cm.exception))

    def test_every_speed_flag_halo_reads_is_stripped(self):
        # A new HALO_* knob in the sources must be added to SPEED_FLAGS (or knowingly left out).
        import re

        found = set()
        for d in ("src", "backends", "include", "tools"):
            for p in (REPO / d).rglob("*"):
                if p.suffix in (".cpp", ".h", ".hpp"):
                    found |= set(re.findall(r'getenv\("(HALO_[A-Z0-9_]+)"', p.read_text(errors="replace")))
        missing = sorted(found - set(bd.SPEED_FLAGS))
        self.assertEqual(missing, [], f"HALO_* env knobs read by the sources but not stripped by bench_decode: {missing}")


if __name__ == "__main__":
    unittest.main()
