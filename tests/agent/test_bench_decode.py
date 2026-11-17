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
    calls = sum(1 for _ in open(spec["log"]))
    if spec.get("busy_on_call") == calls:  # another process takes the GPU during this call
        open(spec["busy_file"], "w").write("100\\n")
    if spec.get("busy_gone_on_call") == calls:  # gpu_busy_percent disappears during this call
        os.remove(spec["busy_file"])
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
        self.spec = {"log": str(self.log), "busy_file": str(self.busy)}
        self.report = self.root / "report"

    def run_bench(self, *extra, env=None):
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(self.spec))
        argv = ["--halo", str(self.halo), "--model", "/models/m.gguf", "--busy-file", str(self.busy),
                "--busy-samples", "1", "--busy-interval", "0", "--busy-settle", "0", "--report-dir", str(self.report),
                *extra]
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
        self.assertTrue(all(r["busy_after"] == 0 for r in self.results.values()),
                        f"every run must be checked idle right after it finished: {self.results}")
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
        # Another process takes the GPU while gdn-reg runs: its tok/s was shared with that
        # process, so it is discarded rather than reported as measured, and nothing more runs.
        self.spec["busy_after"] = {"HALO_GDN_REG": str(self.busy)}
        rc = self.run_bench("--variants", "gdn-reg,stream-reuse,ngram")
        self.assertEqual(rc, 3, f"stderr: {self.stderr}\n{self.summary}")
        self.assertEqual(len(self.calls()), 2, "nothing runs after the GPU turned busy")
        self.assertEqual(self.results["baseline"]["status"], "measured", self.results["baseline"])
        gdn = self.results["gdn-reg"]
        self.assertEqual((gdn["status"], gdn["tps_median"], gdn["tps"]), ("not measured", None, []),
                         f"a run that overlapped another GPU user must not count as measured: {gdn}")
        self.assertEqual(gdn["busy_after"], 100)
        self.assertIn("GPU busy 100% > 5% right after run 1", gdn["note"])
        self.assertEqual(self.results["stream-reuse"]["note"], "skipped after GPU busy")
        self.assertEqual(self.results["ngram"]["note"], "skipped after GPU busy")
        self.assertIn("| gdn-reg | 3 | - | - |", self.summary)

    def test_skipped_variants_still_print_their_rerun_command(self):
        # The sweep stopped at gdn-reg, so stream-reuse and ngram never ran: the summary must still
        # say how to measure each of them alone later (it printed an empty `` before).
        self.spec["busy_after"] = {"HALO_GDN_REG": str(self.busy)}
        self.run_bench("--variants", "gdn-reg,stream-reuse,ngram")
        for name, flag in (("stream-reuse", "HALO_STREAM_REUSE=1"), ("ngram", "HALO_NGRAM_DRAFT=1")):
            cmd = self.results[name]["command"]
            self.assertTrue(cmd.startswith(f"{flag} {self.halo} run /models/m.gguf "),
                            f"skipped {name} must keep its own speed command, got {cmd!r}")
            self.assertIn(f"- {name}: `{cmd}`", self.summary, f"summary lacks {name}'s rerun line:\n{self.summary}")
        self.assertEqual(self.results["stream-reuse"]["command"].split(" ", 1)[1],
                         self.results["baseline"]["command"],
                         "a skipped variant's command is the baseline's plus only its own flag")

    def test_overlap_in_any_repeat_discards_the_whole_variant(self):
        # Repeats 1 and 2 are clean, repeat 3 overlaps: a median over the clean two would hide
        # that the sweep's conditions changed, so the variant reports no tok/s at all.
        self.spec["busy_on_call"] = 3
        self.spec["tps_seq"] = [10.0, 10.2, 4.0]
        rc = self.run_bench("--variants", "baseline", "--repeat", "3")
        self.assertEqual(rc, 3, f"stderr: {self.stderr}")
        b = self.results["baseline"]
        self.assertEqual((b["status"], b["tps"], b["tps_median"]), ("not measured", [], None), b)
        self.assertIn("right after run 3", b["note"])
        self.assertIn("GPU BUSY: not measured", self.summary)

    def test_busy_file_vanishing_during_a_run_is_not_an_idle_gpu(self):
        self.spec["busy_gone_on_call"] = 1
        rc = self.run_bench("--variants", "baseline")
        self.assertEqual(rc, 3, f"stderr: {self.stderr}")
        b = self.results["baseline"]
        self.assertEqual((b["status"], b["tps_median"], b["busy_after"]), ("not measured", None, None), b)
        self.assertIn("GPU busy unreadable right after run 1", b["note"])

    def busy_readings(self, seq):
        """Patch gpu_busy to return `seq` in call order (then its last value): amdgpu smooths
        gpu_busy_percent, so a run's own load fades over a few seconds after halo exits."""
        it = iter(seq)
        return mock.patch.object(bd, "gpu_busy", side_effect=lambda *_: next(it, seq[-1]))

    def test_own_load_fading_after_exit_is_still_an_idle_gpu(self):
        # Idle EVO-X2, 2026-11-17: 0 before the run, then 72 -> 43 -> 20 -> 3 after halo exited.
        # Reading once 1 s after exit (28%) discarded every run of a sweep on an idle machine.
        with self.busy_readings([0, 72, 43, 20, 3]):
            rc = self.run_bench("--variants", "baseline", "--busy-settle", "30")
        b = self.results["baseline"]
        self.assertEqual(rc, 0, f"a fading reading on an idle GPU must not discard the run: {b}\n{self.stderr}")
        self.assertEqual((b["status"], b["tps_median"], b["busy_after"]), ("measured", 10.0, 3), b)

    def test_gpu_still_busy_after_the_settle_window_discards_the_run(self):
        with self.busy_readings([0, 100]):
            rc = self.run_bench("--variants", "baseline", "--busy-settle", "0.2")
        b = self.results["baseline"]
        self.assertEqual(rc, 3, f"a GPU that stays busy after halo exited was shared: {b}")
        self.assertEqual((b["status"], b["tps"], b["busy_after"]), ("not measured", [], 100), b)
        self.assertIn("GPU busy 100% > 5% right after run 1 (still after 0.2 s settle)", b["note"])

    def test_busy_settle_must_not_be_negative(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            bd.main(["--halo", "h", "--model", "m", "--busy-settle", "-1"])

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

    def test_combined_variant_runs_every_parts_flags_in_one_command(self):
        self.spec["tps_by_env"] = {"HALO_MTP_DRAFT_VOCAB": 11.0}
        rc = self.run_bench("--variants", "subgroup-reduce+draft-vocab-32k+mtp-draft-3,repack")
        self.assertEqual(rc, 0, f"stderr: {self.stderr}\n{self.summary}")
        base, combo, repack = self.calls()
        self.assertEqual(combo["halo_env"], {"HALO_SUBGROUP_REDUCE": "1", "HALO_MTP_DRAFT_VOCAB": "32768"},
                         "a+b must run with both parts' HALO_* flags at once")
        self.assertEqual(combo["argv"][-2:], ["--mtp-draft", "3"], f"the part's CLI args must be kept: {combo['argv']}")
        self.assertEqual(repack["halo_env"], {"HALO_REPACK": "1"}, "a variant after a combination runs alone")
        r = self.results["subgroup-reduce+draft-vocab-32k+mtp-draft-3"]
        self.assertEqual((r["step"], r["tps_median"], r["bitwise"]), ("1+2+5", 11.0, True), r)
        self.assertIn("HALO_SUBGROUP_REDUCE=1 HALO_MTP_DRAFT_VOCAB=32768 ", r["command"])
        self.assertIn("| subgroup-reduce+draft-vocab-32k+mtp-draft-3 | 1+2+5 | 11.00 | +10.0% |", self.summary)

    def test_combination_with_a_lossy_part_is_lossy(self):
        self.spec["output_by_env"] = {"HALO_KV_FP16": "390"}
        rc = self.run_bench("--variants", "subgroup-reduce+kv-fp16")
        self.assertEqual(rc, 0, f"a lossy part may change the combination's output: {self.summary}")
        self.assertEqual(self.results["subgroup-reduce+kv-fp16"]["bitwise"], False)

    def test_combinations_that_would_override_a_part_are_usage_errors(self):
        for names, why in (("plain+mtp-draft-3", "--mtp-draft 3 after --mtp-draft 0"),
                           ("kv-fp16+kv-fp16", "distinct parts"),
                           ("baseline+repack", "other than baseline"),
                           ("repack+warp-drive", "warp-drive")):
            with self.subTest(names=names), self.assertRaises(SystemExit) as cm:
                bd.select(names)
            self.assertIn(why, str(cm.exception), f"select({names!r}) must refuse and say why")

    def test_parts_setting_the_same_knob_cannot_be_combined(self):
        a = bd.Variant("a", "1", {"HALO_GDN_DV_TILE": "32"})
        b = bd.Variant("b", "3", {"HALO_GDN_DV_TILE": "64"})
        with self.assertRaises(SystemExit) as cm:
            bd.combine("a+b", [a, b])
        self.assertIn("b sets HALO_GDN_DV_TILE again", str(cm.exception))

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

    @staticmethod
    def sent(call, flag):
        return call["argv"][call["argv"].index(flag) + 1]

    def test_long_context_workload_sends_its_fixed_prompt_with_room_in_the_context(self):
        rc = self.run_bench("--workload", "long-context", "--variants", "kv-fp16")
        self.assertEqual(rc, 0, f"stderr: {self.stderr}\n{self.summary}")
        calls = self.calls()
        prompt = bd.WORKLOADS["long-context"].prompt
        self.assertEqual([self.sent(c, "-p") for c in calls], [prompt, prompt],
                         "baseline and kv-fp16 must both get the long-context prompt, unchanged")
        self.assertEqual(bd._long_context_prompt(), prompt, "the long-context prompt must be the same text every time")
        self.assertGreater(len(prompt), 20000, "long-context must be thousands of tokens, not the 17-token story")
        self.assertEqual([self.sent(c, "--ctx") for c in calls], ["8192", "8192"],
                         "a ~6k-token prompt needs the workload's 8192 context, not the 4096 default")
        self.assertEqual(self.meta["workload"], "long-context")
        self.assertIn("workload `long-context`", self.summary)

    def test_explicit_ctx_overrides_the_workloads(self):
        self.run_bench("--workload", "long-context", "--ctx", "16384", "--variants", "baseline")
        self.assertEqual(self.sent(self.calls()[0], "--ctx"), "16384")

    def test_story_stays_the_default_workload(self):
        self.run_bench("--variants", "baseline")
        call = self.calls()[0]
        self.assertEqual((self.sent(call, "-p"), self.sent(call, "--ctx")), (bd.DEFAULT_PROMPT, "4096"))
        self.assertEqual(self.meta["workload"], "story")

    def test_custom_prompt_is_reported_as_custom(self):
        self.run_bench("--prompt", "Count to five.", "--variants", "baseline")
        self.assertEqual(self.sent(self.calls()[0], "-p"), "Count to five.")
        self.assertEqual(self.meta["workload"], "custom", "a --prompt run must not be labelled as a built-in workload")

    def test_prompt_and_workload_cannot_both_be_given(self):
        with self.assertRaises(SystemExit) as cm, redirect_stderr(io.StringIO()):
            bd.main(["--halo", "h", "--model", "m", "--prompt", "x", "--workload", "code-edit"])
        self.assertEqual(cm.exception.code, 2)

    def test_code_edit_reply_can_start_copying_the_prompt_at_once(self):
        # n-gram drafts look up the context's recurring suffix: the prompt must already end inside a
        # copy of a span it contains earlier, or step 11 measures only lookup overhead again.
        prompt = bd.WORKLOADS["code-edit"].prompt
        last_line = prompt.rsplit("\n", 1)[1]
        self.assertGreater(len(last_line.split()), 3, f"the prompt must end with a copied line, got {last_line!r}")
        self.assertEqual(prompt.count(last_line), 2, "its last line must appear once in the source and once at the end")
        self.assertEqual(prompt.count(bd._CODE_EDIT_SOURCE), 1, "the source to copy must be in the prompt verbatim")

    def test_long_prompt_rerun_line_reads_the_saved_prompt_back_exactly(self):
        import subprocess

        # "$(cat f)" drops trailing newlines, so a long prompt ending in one must stay inline.
        newline_end = "Repeat after me:\n" + "one two three four. " * 20 + "\n"
        cases = [(name, ("--workload", name), bd.WORKLOADS[name].prompt, True) for name in ("long-context", "code-edit")]
        cases.append(("custom ending in a newline", ("--prompt", newline_end), newline_end, False))
        tricky = "It's a\tshort one:\n'quoted' \\n back\\slash $HOME `x`\r\n"
        cases.append(("short multi-line with quotes", ("--prompt", tricky), tricky, False))
        for name, args, prompt, from_file in cases:
            with self.subTest(workload=name):
                self.log.unlink(missing_ok=True)
                (self.report / "prompt.txt").unlink(missing_ok=True)
                self.run_bench(*args, "--variants", "ngram")
                line = next(x for x in self.summary.splitlines() if x.startswith("- ngram: "))
                if from_file:
                    self.assertNotIn(prompt[:200], line, "the rerun line must not inline a long prompt")
                    self.assertIn("prompt.txt", line)
                    self.assertEqual((self.report / "prompt.txt").read_text(), prompt)
                else:
                    self.assertNotIn("prompt.txt", line, "a prompt ending in a newline can't round-trip through $(cat)")
                bench_call = self.calls()[-1]
                env = {k: v for k, v in os.environ.items() if not k.startswith("HALO_")}
                env["FAKE_HALO_SPEC"] = str(self.root / "spec.json")
                p = subprocess.run(["bash", "-c", line.removeprefix("- ngram: `").removesuffix("`")], env=env,
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(self.calls()[-1], bench_call,
                                 "rerunning the summary's line must give halo exactly the argv and HALO_* env the sweep did")


if __name__ == "__main__":
    unittest.main()
