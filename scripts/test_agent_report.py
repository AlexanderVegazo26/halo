#!/usr/bin/env python3
"""Turn a ctest run into .test-report/summary.md + results.json for a fixing agent.

Reads ctest's JUnit output (--output-junit) and its test listing (--show-only=json-v1),
and, when the build itself failed, the compiler errors in the build log. Every failure in
summary.md carries the test name, the failing file:line and its source line, the GoogleTest
message (expected vs actual), and the exact commands to rerun only that test.

    scripts/test_agent_report.py --report-dir .test-report --build-dir build \
        [--junit F] [--tests-json F] [--build-log F] [--source-root DIR] [--exit-code N]

Used by scripts/test-agent; stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

# "/path/file.cpp:27: Failure" (gtest), "/path/file.cpp:27: error: ..." (gtest on some
# platforms), "/path/file.cpp:30: Skipped".
_GTEST_LOC = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+): (?P<kind>Failure|error|Skipped)\b.*$")
# clang/gcc: "path:line:col: error: message" (also "fatal error:").
_COMPILER_ERR = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?:\d+:)? (?:fatal )?error: (?P<msg>.*)$")
_LINKER_ERR = re.compile(r"(undefined reference to|ld(?:\.lld)?: error:|collect2: error:)")
_GTEST_END = re.compile(r"^\[ {2}(FAILED|SKIPPED|  OK  ) {1,2}\] ")
# Sanitizer reports (the HALO_ENABLE_ASAN build): "==123==ERROR: AddressSanitizer: heap-buffer-overflow
# on address ...", UBSan's "file:line:col: runtime error: ...", and stack frames
# "    #0 0x55 in func /path/file.cpp:42:7".
_SAN_ERROR = re.compile(r"^==\d+==ERROR: (?P<tool>\w+Sanitizer): (?P<what>.*)$")
_UBSAN = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):\d+: runtime error: (?P<what>.*)$")
_FRAME = re.compile(r"^\s+#\d+ 0x[0-9a-f]+ in (?P<func>.+?) (?P<file>/[^\s:]+):(?P<line>\d+)(?::\d+)?$")
_SAN_END = re.compile(r"^(SUMMARY: \w+Sanitizer|Shadow bytes around|==\d+==ABORTING)")

# ctest's <failure message=...> for a test that never reported a gtest failure.
_CRASH_HINT = {
    "Timeout": "the test hit ctest's --timeout (hang or far too slow); the output before the cutoff is below",
    "SEGFAULT": "the test process crashed with SIGSEGV; rerun the binary under a debugger or the ASan build (scripts/build.sh --asan)",
    "Subprocess aborted": "the process aborted (assert/abort/uncaught exception)",
    "Exception": "the process died on a signal",
}
_SANITIZER_HINT = ("a sanitizer stopped the test: fix the memory/UB bug at the first in-repo frame below "
                   "(the report is from the ASan+UBSan build, `-DHALO_ENABLE_ASAN=ON`; rerun with the "
                   "same build dir)")


@dataclass
class Assertion:
    file: str
    line: int
    source: str | None
    message: str


@dataclass
class Frame:
    label: str  # the report line the frame belongs to: "WRITE of size 4 ...", "allocated by thread T0 here:"
    func: str
    file: str
    line: int
    source: str | None


@dataclass
class SanitizerFinding:
    kind: str  # "AddressSanitizer: heap-buffer-overflow", "UndefinedBehaviorSanitizer: signed integer overflow ..."
    frames: list[Frame] = field(default_factory=list)


@dataclass
class TestResult:
    name: str
    status: str  # passed | failed | skipped
    reason: str = ""  # ctest's failure/skip message: Failed, SEGFAULT, Timeout, ...
    time_s: float = 0.0
    defined_at: str | None = None
    assertions: list[Assertion] = field(default_factory=list)
    sanitizer: list[SanitizerFinding] = field(default_factory=list)
    skip_reason: str = ""
    output_tail: str = ""
    rerun: list[str] = field(default_factory=list)


def _source_line(source_root: Path, file: str, line: int) -> str | None:
    path = Path(file)
    if not path.is_absolute():
        path = source_root / path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return lines[line - 1].strip() if 0 < line <= len(lines) else None


def _relative(source_root: Path, file: str) -> str:
    try:
        return str(Path(file).relative_to(source_root))
    except ValueError:
        return file


def parse_gtest_output(text: str, source_root: Path) -> tuple[list[Assertion], str]:
    """Failure blocks and the skip reason from one test's GoogleTest stdout."""
    assertions: list[Assertion] = []
    skip_reason = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _GTEST_LOC.match(lines[i])
        if not m:
            i += 1
            continue
        body: list[str] = []
        i += 1
        while i < len(lines) and not _GTEST_LOC.match(lines[i]) and not _GTEST_END.match(lines[i]):
            body.append(lines[i])
            i += 1
        message = "\n".join(body).strip("\n")
        if m["kind"] == "Skipped":
            skip_reason = message.strip()
            continue
        file = _relative(source_root, m["file"])
        line = int(m["line"])
        assertions.append(Assertion(file, line, _source_line(source_root, file, line), message))
    return assertions, skip_reason


def _in_repo(source_root: Path, file: str) -> bool:
    """Our code, not the toolchain's or a FetchContent dependency (build/_deps lives under the root)."""
    rel = _relative(source_root, file)
    return not Path(rel).is_absolute() and "_deps" not in Path(rel).parts


def parse_sanitizer_output(text: str, source_root: Path) -> tuple[list[SanitizerFinding], str]:
    """Sanitizer findings with the first in-repo frame of each stack, and the report text minus
    the shadow-byte dump (which is all a fixed-size output tail would otherwise show)."""
    findings: list[SanitizerFinding] = []
    report: list[str] = []
    current: SanitizerFinding | None = None
    label = ""
    labelled: set[str] = set()  # one frame per stack: the first in-repo one
    in_report = skipping = False
    for raw in text.splitlines():
        m = _SAN_ERROR.match(raw)
        u = _UBSAN.match(raw)
        if m or u:
            if m:
                kind = f"{m['tool']}: {m['what'].split(' on address ')[0].strip()}"
            else:
                kind = f"UndefinedBehaviorSanitizer: {u['what'].strip()}"
            current = SanitizerFinding(kind)
            findings.append(current)
            label, labelled, in_report, skipping = (kind if u else ""), set(), True, False
            report.append(raw)
            continue
        if not in_report:
            continue
        if raw.startswith("Shadow bytes around"):
            skipping = True
        if _SAN_END.match(raw) and raw.startswith("SUMMARY"):
            skipping = False
        f = _FRAME.match(raw)
        if not skipping:
            if f and not _in_repo(source_root, f["file"]) or raw.lstrip().startswith("#") and not f:
                # toolchain/gtest/libc frames: one marker per run of them instead of 10+ lines
                if not report or report[-1] != "    ...":
                    report.append("    ...")
            else:
                report.append(raw)
        if f and current is not None:
            if label not in labelled and _in_repo(source_root, f["file"]):
                labelled.add(label)
                rel = _relative(source_root, f["file"])
                line = int(f["line"])
                current.frames.append(Frame(label or current.kind, f["func"], rel, line,
                                            _source_line(source_root, rel, line)))
        elif raw.strip() and not raw.startswith((" ", "==", "SUMMARY", "Shadow", "=>")) and not skipping:
            label = raw.strip().rstrip(":")
    return findings, "\n".join(report)


def _tail(text: str, n: int = 30) -> str:
    keep = [l for l in text.splitlines() if not l.startswith(("Running main() from", "Note: Google Test filter"))]
    return "\n".join(keep[-n:])


def _ctest_regex(name: str) -> str:
    return "^" + re.escape(name) + "$"


def load_listing(tests_json: Path | None, source_root: Path) -> dict[str, dict]:
    if not tests_json or not tests_json.is_file():
        return {}
    data = json.loads(tests_json.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for t in data.get("tests", []):
        props = {p["name"]: p["value"] for p in t.get("properties", [])}
        defined = props.get("DEF_SOURCE_LINE")
        if defined:
            defined = _relative(source_root, defined)
        out[t["name"]] = {"command": t.get("command") or [], "defined_at": defined}
    return out


def parse_junit(junit: Path, listing: dict[str, dict], source_root: Path, build_dir: str) -> list[TestResult]:
    root = ET.parse(junit).getroot()
    results: list[TestResult] = []
    for case in root.iter("testcase"):
        name = case.get("name", "")
        out = case.findtext("system-out") or ""
        failure = case.find("failure")
        skipped = case.find("skipped")
        if failure is not None:
            status, reason = "failed", failure.get("message", "Failed")
        elif skipped is not None or case.get("status") == "notrun":
            status, reason = "skipped", (skipped.get("message", "") if skipped is not None else "notrun")
        else:
            status, reason = "passed", ""
        info = listing.get(name, {})
        r = TestResult(name=name, status=status, reason=reason, time_s=float(case.get("time") or 0),
                       defined_at=info.get("defined_at"))
        if status != "passed":
            r.assertions, r.skip_reason = parse_gtest_output(out, source_root)
        if status == "failed":
            r.rerun = [f"ctest --test-dir {shlex.quote(build_dir)} -R {shlex.quote(_ctest_regex(name))} --output-on-failure"]
            if info.get("command"):
                r.rerun.append(shlex.join(info["command"]))
            r.sanitizer, san_report = parse_sanitizer_output(out, source_root)
            if r.sanitizer:
                r.output_tail = _tail(san_report, 40)
            elif not r.assertions:
                r.output_tail = _tail(out)
        results.append(r)
    return results


def parse_build_errors(build_log: Path, source_root: Path, limit: int = 20) -> list[dict]:
    errors: list[dict] = []
    seen: set[tuple] = set()
    for raw in build_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _COMPILER_ERR.match(raw)
        if m:
            file = _relative(source_root, m["file"])
            key = (file, m["line"], m["msg"])
            if key in seen:
                continue
            seen.add(key)
            errors.append({"file": file, "line": int(m["line"]), "message": m["msg"],
                           "source": _source_line(source_root, file, int(m["line"]))})
        elif _LINKER_ERR.search(raw) and raw not in seen:
            seen.add(raw)
            errors.append({"file": None, "line": None, "message": raw.strip(), "source": None})
        if len(errors) >= limit:
            break
    return errors


def _fence(text: str) -> str:
    return "```\n" + text.rstrip() + "\n```"


def render(results: list[TestResult], build_errors: list[dict] | None, exit_code: int, rerun_all: str,
           log_name: str) -> str:
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]
    passed = sum(r.status == "passed" for r in results)
    out: list[str] = []
    if build_errors is not None:
        out += ["# test-agent: FAIL (build)", "",
                f"The build failed, so no tests ran. Fix these first, then rerun `{rerun_all}`. Full output: `{log_name}`.", ""]
        for e in build_errors:
            where = f"{e['file']}:{e['line']}" if e["file"] else "link"
            out.append(f"- `{where}`: {e['message']}")
            if e["source"]:
                out.append(f"  - source: `{e['source']}`")
        if not build_errors:
            out.append(f"- no compiler error lines recognised; read the end of `{log_name}`")
        return "\n".join(out) + "\n"

    if failed or exit_code != 0:
        verdict = "FAIL"
    elif not results:
        verdict = "FAIL (no tests ran)"
    else:
        verdict = "PASS"
    out += [f"# test-agent: {verdict}", "",
            f"{len(results)} tests: {passed} passed, {len(failed)} failed, {len(skipped)} skipped. "
            f"Rerun everything: `{rerun_all}`. Full output: `{log_name}`.", ""]
    if exit_code != 0 and not failed:
        out += [f"ctest exited {exit_code} without a failing test case: read `{log_name}`.", ""]
    for r in failed:
        out += [f"## FAILED {r.name} ({r.reason})", ""]
        if r.defined_at:
            out.append(f"- defined at `{r.defined_at}`")
        hint = _SANITIZER_HINT if r.sanitizer else _CRASH_HINT.get(r.reason)
        if hint:
            out.append(f"- {hint}")
        for sf in r.sanitizer:
            out.append(f"- **{sf.kind}**")
            if not sf.frames:
                out.append("  - no frame in this repo's sources: read the report below")
            for fr in sf.frames:
                where = f"`{fr.file}:{fr.line}` in `{fr.func}`" + (f": `{fr.source}`" if fr.source else "")
                out.append(f"  - {fr.label}: {where}" if fr.label != sf.kind else f"  - at {where}")
        for a in r.assertions:
            out.append(f"- assertion at `{a.file}:{a.line}`" + (f": `{a.source}`" if a.source else ""))
            if a.message:
                out += ["", _fence(a.message), ""]
        if r.output_tail:
            out += ["", "Output tail:", "", _fence(r.output_tail), ""]
        out += ["", "Rerun only this test:", ""] + [f"    {c}" for c in r.rerun] + [""]
    if skipped:
        reasons: dict[str, int] = {}
        for r in skipped:
            key = r.skip_reason.splitlines()[0] if r.skip_reason else r.reason
            reasons[key] = reasons.get(key, 0) + 1
        out += ["## Skipped", ""]
        out += [f"- {n} x {why}" for why, n in sorted(reasons.items(), key=lambda kv: -kv[1])]
        out.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--junit", type=Path)
    ap.add_argument("--tests-json", type=Path)
    ap.add_argument("--build-log", type=Path, help="set when the build failed: report its errors instead of tests")
    ap.add_argument("--source-root", type=Path, default=Path.cwd())
    ap.add_argument("--exit-code", type=int, default=0, help="ctest's exit code")
    ap.add_argument("--rerun-all", default="scripts/test-agent")
    a = ap.parse_args(argv)

    a.report_dir.mkdir(parents=True, exist_ok=True)
    source_root = a.source_root.resolve()
    log_name = str(a.report_dir / "run.log")
    if a.build_log:
        errors = parse_build_errors(a.build_log, source_root)
        summary = render([], errors, 1, a.rerun_all, log_name)
        payload = {"verdict": "build-failed", "build_errors": errors, "tests": []}
        code = 1
    else:
        listing = load_listing(a.tests_json, source_root)
        results = parse_junit(a.junit, listing, source_root, a.build_dir) if a.junit and a.junit.is_file() else []
        summary = render(results, None, a.exit_code, a.rerun_all, log_name)
        failed = any(r.status == "failed" for r in results)
        code = 1 if failed or a.exit_code != 0 or not results else 0
        payload = {"verdict": "fail" if code else "pass", "exit_code": a.exit_code,
                   "counts": {s: sum(r.status == s for r in results) for s in ("passed", "failed", "skipped")},
                   "tests": [asdict(r) for r in results if r.status != "passed"]}
    (a.report_dir / "summary.md").write_text(summary, encoding="utf-8")
    (a.report_dir / "results.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
