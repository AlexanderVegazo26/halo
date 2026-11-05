#!/usr/bin/env python3
"""Runs one libFuzzer target (a HALO_BUILD_FUZZERS build) over its seed corpus.

The checked-in corpus stays read-only: libFuzzer writes the inputs it finds to a work dir
and crash/timeout/oom reproducers to an artifacts dir, both under
$HALO_FUZZ_REPORT_DIR/<target> (scripts/test-agent sets it to .test-report/fuzz; default
./fuzz-out). On a finding it prints the reproducer path and the commands to replay it.

  run.py --smoke build-fuzz/tests/fuzz/fuzz_gguf tests/fuzz/corpus/gguf   # ctest, CI: fixed seed + runs
  run.py --time 600 build-fuzz/tests/fuzz/fuzz_gguf tests/fuzz/corpus/gguf  # opt-in long campaign
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Per-target smoke budget: a fixed number of executions with a fixed seed, so a run is
# reproducible (same corpus + same binary = same inputs) and takes seconds, not minutes.
SMOKE_RUNS = {"fuzz_gguf": 20000, "fuzz_api_request": 20000, "fuzz_output_parser": 100000, "fuzz_chat_template": 20000, "fuzz_stop_matcher": 100000}
MAX_LEN = {"fuzz_gguf": 16384, "fuzz_api_request": 8192, "fuzz_output_parser": 4096, "fuzz_chat_template": 4096, "fuzz_stop_matcher": 1024}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="fixed seed and run count (blocking)")
    mode.add_argument("--time", type=int, metavar="SECONDS", help="time-boxed campaign (opt-in)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel fuzzing jobs for --time")
    ap.add_argument("target", type=Path)
    ap.add_argument("corpus", type=Path)
    args = ap.parse_args(argv)

    target, corpus = args.target.resolve(), args.corpus.resolve()  # libFuzzer runs in the out dir
    name = target.name
    out = Path(os.environ.get("HALO_FUZZ_REPORT_DIR", "fuzz-out")).resolve() / name
    work, artifacts = out / "corpus", out / "artifacts"
    shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    cmd = [str(target), f"-artifact_prefix={artifacts}/", f"-max_len={MAX_LEN.get(name, 4096)}",
           "-timeout=10", "-rss_limit_mb=2048", "-print_final_stats=1", "-detect_leaks=1"]
    if args.smoke:
        cmd += ["-seed=1", f"-runs={SMOKE_RUNS.get(name, 20000)}"]
    else:
        cmd += [f"-max_total_time={args.time}"]
        if args.jobs > 1:
            cmd += [f"-jobs={args.jobs}", f"-workers={args.jobs}"]
    cmd += [str(work), str(corpus)]
    print("+", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=out).returncode

    found = sorted(p for p in artifacts.iterdir() if p.is_file())
    if rc == 0 and not found:
        print(f"fuzz: {name} clean ({len(list(work.iterdir()))} new corpus inputs in {work})")
        return 0
    print(f"\nfuzz: {name} FAILED (exit {rc})", file=sys.stderr)
    for p in found:
        print(f"  reproducer: {p}\n"
              f"    replay:   {target} {p}\n"
              f"    keep it:  cp {p} {corpus}/  (then the replay test guards the fix in every build)",
              file=sys.stderr)
    return rc or 1


if __name__ == "__main__":
    sys.exit(main())
