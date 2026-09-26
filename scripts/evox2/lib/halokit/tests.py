"""`run-tests`: runs every gtest binary of the kit with XML output and summarizes the result.

Crash- and hang-tolerant: a binary that dies or times out is re-run one test at a time, so a
single GPU fault does not hide the outcome of the rest of the binary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from . import common as C

PER_TEST_TIMEOUT_MAX = 600


def binary_category(name: str) -> str:
    if name.startswith("test_hip"):
        return "hip"
    if name.startswith("test_vulkan") or name.startswith("test_vk"):
        return "vulkan"
    return "cpu"


def kernel_log() -> list[str] | None:
    """Kernel ring buffer lines, or None when this user may not read it."""
    for argv in (["dmesg"], ["sudo", "-n", "dmesg"]):
        if C.which(argv[0]) is None:
            continue
        import subprocess
        try:
            r = subprocess.run(argv, capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace").splitlines()
    return None


GPU_KMSG = re.compile(r"amdgpu|kfd|drm|gpu|hsa|iommu|ttm|vram|gtt|mes|gfxhub|mmhub|page fault|ring", re.I)


def run_binary(b: dict[str, Any], anchor: Path, out: Path, env: dict[str, str], timeout: float) -> dict[str, Any]:
    name = b["name"]
    exe = str(anchor / b["path"])
    cwd = str(anchor / b["working_dir"])
    xml = out / "xml" / f"{name}.xml"
    log = out / "logs" / f"{name}.log"
    xml.parent.mkdir(parents=True, exist_ok=True)
    if xml.exists():
        xml.unlink()
    # Only the HIP binaries link ROCm: give just them the target's ROCm lib dir first
    # (C.rocm_lib_env). Their RUNPATH already has /opt/rocm/lib; this adds $ROCM_PATH support.
    hip_env = C.rocm_lib_env(env) if binary_category(name) == "hip" else {}
    if hip_env:
        env = {**env, **hip_env}
    kbefore = kernel_log() if binary_category(name) != "cpu" else None
    rec = C.run_cmd([exe, f"--gtest_output=xml:{xml}"], stdout_path=log, timeout=timeout, cwd=cwd, env=env,
                    time_v_path=out / "logs" / f"{name}.time-v.txt")
    res: dict[str, Any] = {"name": name, "category": binary_category(name), "command": rec["command"],
                           "cwd": cwd, "exit_code": rec.get("exit_code"), "run_status": rec["status"],
                           "duration_s": rec.get("duration_s"), "max_rss_kb": rec.get("max_rss_kb"),
                           "log": str(log), "xml": str(xml)}
    if hip_env:
        res["env_overrides"] = hip_env
    if rec.get("signal"):
        res["signal"] = rec.get("signal")
    tests: list[dict[str, Any]] = []
    if rec["status"] in ("pass", "fail") and xml.is_file():
        try:
            tests = C.parse_gtest_xml(xml)
        except Exception as e:  # noqa: BLE001 - a broken XML is itself a finding
            res["xml_error"] = str(e)
    if rec["status"] in ("crashed", "timeout", "not-available", "error") or not xml.is_file() or "xml_error" in res:
        res["rerun_per_test"] = True
        tests = rerun_per_test(b, exe, cwd, out, env, min(timeout, PER_TEST_TIMEOUT_MAX))
    if kbefore is not None:
        kafter = kernel_log() or []
        new = kafter[len(kbefore):] if len(kafter) >= len(kbefore) else kafter
        gpu_lines = [ln for ln in new if GPU_KMSG.search(ln)]
        if gpu_lines:
            kp = out / "logs" / f"{name}.dmesg-delta.txt"
            kp.write_text("\n".join(gpu_lines) + "\n", encoding="utf-8")
            res["dmesg_delta"] = str(kp)
            res["dmesg_delta_lines"] = len(gpu_lines)
    elif binary_category(name) != "cpu":
        res["dmesg_delta"] = "not readable (dmesg restricted; run `sudo -v` first or run as root)"
    for t in tests:
        if t["status"] == "skipped":
            t["skip_category"] = C.skip_category(t["message"])
    res["tests"] = tests
    counts = {k: 0 for k in ("passed", "failed", "skipped", "disabled", "crashed", "timeout")}
    for t in tests:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    res["counts"] = counts
    if rec["status"] in ("crashed", "timeout", "error"):
        # Even if every test passes when run alone, the whole-binary run died (a teardown crash,
        # a GPU fault triggered by an earlier test, a hang): that is a finding, never a pass.
        res["whole_run_problem"] = (f"the full run {rec['status']} (exit {rec.get('exit_code')}, signal "
                                    f"{rec.get('signal_name', rec.get('signal'))}); tests were re-run one at a time")
    if rec["status"] == "not-available":
        res["status"] = "missing"
    elif counts["failed"] or counts["crashed"] or counts["timeout"] or not tests or "whole_run_problem" in res:
        res["status"] = "fail"
    elif rec["status"] != "pass" and not res.get("rerun_per_test"):
        res["status"] = "fail"
    else:
        res["status"] = "pass"
    return res


def rerun_per_test(b: dict[str, Any], exe: str, cwd: str, out: Path, env: dict[str, str],
                   timeout: float) -> list[dict[str, Any]]:
    name = b["name"]
    d = out / "rerun" / name
    d.mkdir(parents=True, exist_ok=True)
    names = C.list_gtests(exe, cwd, env, d / "list.txt")
    results = []
    for i, t in enumerate(names):
        xml = d / f"{i:04d}.xml"
        log = d / f"{i:04d}.log"
        rec = C.run_cmd([exe, f"--gtest_filter={t}", f"--gtest_output=xml:{xml}"], stdout_path=log,
                        timeout=timeout, cwd=cwd, env=env)
        if rec["status"] in ("crashed", "timeout") or not xml.is_file():
            st = "timeout" if rec["status"] == "timeout" else "crashed"
            msg = f"{st}: exit {rec.get('exit_code')} signal {rec.get('signal_name', rec.get('signal'))}; log {log}"
            results.append({"name": t, "status": st, "message": msg, "time_s": rec.get("duration_s", 0)})
            continue
        try:
            parsed = C.parse_gtest_xml(xml)
        except Exception as e:  # noqa: BLE001
            results.append({"name": t, "status": "crashed", "message": f"bad XML: {e}", "time_s": 0})
            continue
        for p in parsed:
            p["log"] = str(log)
        results.extend(parsed)
    return results


def expectation_failures(binaries: list[dict[str, Any]], gpu_host: bool, env_notes: dict[str, Any]) -> list[str]:
    out = []
    if env_notes.get("HSA_OVERRIDE_GFX_VERSION"):
        out.append(f"HSA_OVERRIDE_GFX_VERSION is set ({env_notes['HSA_OVERRIDE_GFX_VERSION']}); HALO builds "
                   "for gfx1151 and must run without an override")
    for b in binaries:
        if b["status"] == "missing":
            out.append(f"{b['name']}: binary missing from the kit")
        if b.get("whole_run_problem"):
            out.append(f"{b['name']}: {b['whole_run_problem']}")
        for t in b.get("tests", []):
            if t["status"] == "skipped" and t.get("skip_category") == "no-device" and gpu_host:
                out.append(f"{b['name']}: {t['name']} SKIPPED on a gfx1151 host: {t['message'][:300]}")
    return out


def summarize(binaries: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    totals = {k: 0 for k in ("passed", "failed", "skipped", "disabled", "crashed", "timeout")}
    skips: dict[str, int] = {}
    for b in binaries:
        for k, v in b["counts"].items():
            totals[k] = totals.get(k, 0) + v
        for t in b.get("tests", []):
            if t["status"] == "skipped":
                skips[t["skip_category"]] = skips.get(t["skip_category"], 0) + 1
    exp = expectation_failures(binaries, meta["gpu_host"], meta.get("gpu_env", {}))
    ok = (totals["failed"] == 0 and totals["crashed"] == 0 and totals["timeout"] == 0 and not exp
          and all(b["status"] == "pass" for b in binaries))
    return {"schema": "halo.kit.test-run/1", **meta, "totals": totals, "skips_by_category": skips,
            "expectation_failures": exp, "ok": ok, "binaries": binaries}


def write_markdown(s: dict[str, Any], path: Path) -> None:
    L = [f"# Test run: {s['label']}", "",
         f"- kit: {s['kit_sha']} ({'dirty' if s.get('kit_dirty') else 'clean'}); host gfx1151: {s['gpu_host']}",
         f"- Vulkan ICD: {s['icd_note']}", f"- full bandwidth benchmark: {s['full_bandwidth']}",
         f"- verdict: **{'OK' if s['ok'] else 'NOT OK'}**", "",
         "| totals | passed | failed | skipped | crashed | timeout |", "|---|---|---|---|---|---|",
         "| all | {passed} | {failed} | {skipped} | {crashed} | {timeout} |".format(**s["totals"]), "",
         "Skips by category: " + (", ".join(f"{k} {v}" for k, v in sorted(s["skips_by_category"].items())) or "none"), ""]
    if s["expectation_failures"]:
        L += ["## Expectation failures", ""] + [f"- {e}" for e in s["expectation_failures"]] + [""]
    whole = [b for b in s["binaries"] if b.get("whole_run_problem")]
    if whole:
        L += ["## Binaries whose full run crashed or hung", ""]
        L += [f"- `{b['name']}`: {b['whole_run_problem']}; log `{b['log']}`" for b in whole] + [""]
    bad = [(b, t) for b in s["binaries"] for t in b.get("tests", []) if t["status"] in ("failed", "crashed", "timeout")]
    if bad:
        L += ["## Failed tests", ""]
        for b, t in bad:
            L.append(f"- `{b['name']}` **{t['name']}** ({t['status']}): {t['message'][:400]!s}  \n  log: `{t.get('log', b['log'])}`")
        L.append("")
    L += ["## Binaries", "", "| binary | status | exit | passed | failed | skipped | s | log |", "|---|---|---|---|---|---|---|---|"]
    for b in s["binaries"]:
        c = b["counts"]
        L.append(f"| {b['name']} | {b['status']} | {b.get('exit_code')} | {c['passed']} | {c['failed'] + c['crashed'] + c['timeout']} "
                 f"| {c['skipped']} | {b.get('duration_s')} | `{b['log']}` |")
    L += ["", "## Skipped tests", ""]
    for b in s["binaries"]:
        for t in b.get("tests", []):
            if t["status"] == "skipped":
                L.append(f"- [{t['skip_category']}] `{b['name']}` {t['name']}: {t['message'][:200]}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def compare_main(argv: list[str]) -> int:
    """Per-test status/skip-message differences between two run-tests summaries."""
    ap = argparse.ArgumentParser(prog="halo_kit.py compare")
    ap.add_argument("a")
    ap.add_argument("b")
    a = ap.parse_args(argv)

    def load(p: str) -> dict[str, tuple[str, str]]:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        out = {}
        for b in s["binaries"]:
            for t in b.get("tests", []):
                msg = t["message"] if t["status"] == "skipped" else ""
                out[f"{b['name']}::{t['name']}"] = (t["status"], msg)
        return out

    A, B = load(a.a), load(a.b)
    diffs = []
    for k in sorted(set(A) | set(B)):
        if A.get(k) != B.get(k):
            diffs.append(f"{k}: {A.get(k, ('absent', ''))} -> {B.get(k, ('absent', ''))}")
    for d in diffs:
        print(d)
    print(f"compare: {len(A)} vs {len(B)} tests, {len(diffs)} difference(s)")
    return 0 if not diffs else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="run-tests.sh", description="Run the kit's gtest binaries (docs/evox2.md).")
    ap.add_argument("--kit", help="unpacked kit directory (default: the kit this script is in)")
    ap.add_argument("--ref", help="reference data directory (default: $HALO_KIT_REF, <kit>/ref, <kit>/../halo-ref)")
    ap.add_argument("--out", default=None, help="output directory (default ./halo-tests-<utc>)")
    ap.add_argument("--label", default="default", help="name of this run in the summaries")
    ap.add_argument("--category", choices=("all", "cpu", "hip", "vulkan"), default="all")
    ap.add_argument("--only", default=None, help="regex over binary names")
    ap.add_argument("--vk-icd", default="default", choices=("default", "radv", "amdvlk", "lavapipe"),
                    help="restrict the Vulkan loader to one driver family (VK_DRIVER_FILES/VK_ICD_FILENAMES)")
    ap.add_argument("--full-bandwidth", action="store_true", help="HALO_BANDWIDTH_FULL=1 (TRD §50 benchmark)")
    ap.add_argument("--timeout", type=float, default=1800, help="per binary, seconds (default 1800)")
    ap.add_argument("--expect-gpu", choices=("auto", "yes", "no"), default="auto",
                    help="treat device-test skips as failures (auto: yes when KFD reports gfx1151)")
    ap.add_argument("--llama-dir", default=None, help="llama.cpp bin dir for the baseline tests (HALO_LLAMA_BIN_DIR)")
    ap.add_argument("--no-anchor", action="store_true",
                    help="(package.sh) the anchor is the real build tree: only link its ref/")
    a = ap.parse_args(argv)

    kit = Path(a.kit).resolve() if a.kit else C.kit_root_from(__file__)
    info = C.load_kit(kit)
    ref = C.find_ref(kit, a.ref)
    if a.no_anchor:
        anchor = Path(info["anchor"])
        if not (anchor / ".halo-kit-staging").exists():
            raise C.KitError(f"--no-anchor: {anchor} is not a package build tree")
        link = anchor / "ref"
        if link.is_symlink():
            link.unlink()
        if ref is not None:
            link.symlink_to(ref.resolve())
        notes = [f"in-tree run; {link} -> {ref}"]
    else:
        notes = C.setup_anchor(kit, info["anchor"], ref)
    out = Path(a.out or f"halo-tests-{C.utc_stamp()}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu_host = a.expect_gpu == "yes" or (a.expect_gpu == "auto" and C.is_gfx1151_host())

    env = dict(os.environ)
    env_extra: dict[str, str] = {}
    icd, icd_note = C.icd_env(a.vk_icd)
    if icd is None:
        C.write_json(out / "summary.json", {"schema": "halo.kit.test-run/1", "label": a.label, "ok": None,
                                            "not_available": icd_note})
        print(f"run-tests: {a.label}: not available: {icd_note}")
        return 3
    env_extra.update(icd)
    if a.full_bandwidth:
        env_extra["HALO_BANDWIDTH_FULL"] = "1"
    if a.llama_dir:
        env_extra["HALO_LLAMA_BIN_DIR"] = str(Path(a.llama_dir).resolve())
    env.update(env_extra)

    rx = re.compile(a.only) if a.only else None
    sel = [b for b in info["tests"]["binaries"]
           if (a.category == "all" or binary_category(b["name"]) == a.category) and (rx is None or rx.search(b["name"]))]
    gpu_env = {k: v for k, v in C.redacted_env().items() if k.startswith(C.GPU_ENV_PREFIXES)}
    meta = {"label": a.label, "kit": str(kit), "kit_sha": info.get("git_sha"), "kit_dirty": info.get("dirty"),
            "anchor": info["anchor"], "anchor_notes": notes, "ref": str(ref) if ref else None,
            "gpu_host": gpu_host, "gfx_targets": C.kfd_gfx_targets(), "icd": a.vk_icd, "icd_note": icd_note,
            "full_bandwidth": a.full_bandwidth, "env_overrides": env_extra, "gpu_env": gpu_env,
            "started_utc": C.utc_now()}
    t0 = time.monotonic()
    results = []
    for b in sel:
        print(f"run-tests[{a.label}]: {b['name']} ...", flush=True)
        r = run_binary(b, Path(info["anchor"]), out, env, a.timeout)
        c = r["counts"]
        print(f"run-tests[{a.label}]: {b['name']}: {r['status']} (passed {c['passed']}, failed {c['failed']}, "
              f"skipped {c['skipped']}, crashed {c['crashed']}, timeout {c['timeout']}) {r['duration_s']} s", flush=True)
        results.append(r)
    meta["duration_s"] = round(time.monotonic() - t0, 1)
    s = summarize(results, meta)
    C.write_json(out / "summary.json", s)
    write_markdown(s, out / "summary.md")
    t = s["totals"]
    print(f"run-tests[{a.label}]: {'OK' if s['ok'] else 'NOT OK'}: passed {t['passed']}, failed {t['failed']}, "
          f"skipped {t['skipped']} {json.dumps(s['skips_by_category'])}, crashed {t['crashed']}, timeout {t['timeout']}; "
          f"{len(s['expectation_failures'])} expectation failure(s); summary {out / 'summary.md'}")
    return 0 if s["ok"] else 1
