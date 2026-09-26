"""`verify-env`: pre-flight checks of the target machine and the unpacked kit (docs/evox2.md step 1).

Each check prints one line: OK, WARN, FAIL, INFO or N/A. Exit 1 if any check FAILs.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import common as C


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, verdict: str, name: str, detail: str) -> None:
        self.items.append({"verdict": verdict, "check": name, "detail": detail})
        print(f"{verdict:<4} {name}: {detail}", flush=True)


def elf_files(kit: Path) -> list[Path]:
    out = []
    for root in (kit / "bin", kit / "build"):
        for p in sorted(root.rglob("*")) if root.is_dir() else []:
            if p.is_file() and os.access(p, os.X_OK):
                with open(p, "rb") as f:
                    if f.read(4) == b"\x7fELF":
                        out.append(p)
    return out


def check_sums(kit: Path) -> tuple[int, list[str]]:
    sums = kit / "SHA256SUMS"
    bad = []
    n = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        p = kit / name
        n += 1
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                bad.append(f"{name}: checksum mismatch")
        except OSError as e:
            bad.append(f"{name}: {e.strerror}")
    return n, bad


def ldd(p: Path, extra_env: dict[str, str] | None = None) -> str:
    env = dict(os.environ)
    env.update(extra_env or {})
    try:
        r = subprocess.run(["ldd", str(p)], capture_output=True, timeout=30, env=env)
        return r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"ldd failed: {e}"


def run(kit: Path, ref_arg: str | None, check_ref: bool) -> tuple[Checks, dict[str, Any]]:
    ck = Checks()
    facts: dict[str, Any] = {}
    info = C.load_kit(kit)
    gpu_host = C.is_gfx1151_host()
    facts["gfx_targets"] = C.kfd_gfx_targets()
    ck.add("OK", "python", sys.version.split()[0])
    ck.add("INFO", "kit", f"{info.get('kit_name')} built {info.get('build_date_utc')} "
                         f"(-march={info.get('march')}, ROCm {info.get('rocm_version_built_with')})")

    n, bad = check_sums(kit)
    ck.add("FAIL" if bad else "OK", "kit integrity", f"{len(bad)} of {n} files bad: {bad[:5]}" if bad else f"{n} files match SHA256SUMS")

    halo = kit / "bin" / "halo"
    try:
        r = subprocess.run([str(halo), "version"], capture_output=True, timeout=30)
        ok = r.returncode == 0
        ck.add("OK" if ok else "FAIL", "halo version", (r.stdout.decode(errors="replace").strip().splitlines() or ["(no output)"])[0]
               if ok else f"exit {r.returncode}: {r.stderr.decode(errors='replace').strip()[:300]}")
    except OSError as e:
        ck.add("FAIL", "halo version", f"cannot execute {halo}: {e} (wrong CPU target? see docs/evox2.md troubleshooting)")

    missing: dict[str, list[str]] = {}
    sys_missing: list[str] = []
    hip_lib = None
    kit_env = C.rocm_lib_env()  # what run-tests.sh adds for the test_hip_* binaries only
    for p in elf_files(kit):
        out = ldd(p, kit_env if p.name.startswith("test_hip") else None)
        nf = [ln.strip() for ln in out.splitlines() if "not found" in ln]
        if nf:
            missing[str(p.relative_to(kit))] = nf
        m = re.search(r"libamdhip64\.so\S*\s+=>\s+(\S+)", out)
        if m:
            hip_lib = m.group(1)
        if kit_env and "libamdhip64" in out and "not found" in ldd(p):
            sys_missing.append(p.name)
    facts["libamdhip64"] = hip_lib
    facts["kit_ld_env"] = kit_env
    how = f" (test_hip_* with {kit_env})" if kit_env else ""
    if missing:
        ck.add("FAIL", "shared libraries", f"unresolved{how}: " + "; ".join(f"{k}: {v}" for k, v in list(missing.items())[:6]))
    else:
        ck.add("OK", "shared libraries", f"every binary resolves its libraries{how} "
                                         f"(libamdhip64 -> {hip_lib or 'not needed/not found'})")
    if sys_missing:
        ck.add("INFO", "libamdhip64 lookup", f"{len(sys_missing)} binaries find libamdhip64 only through the kit's "
               "LD_LIBRARY_PATH (not through their RUNPATH /opt/rocm/lib or ld.so.conf); run them through the kit scripts")
    if hip_lib:
        real = os.path.realpath(hip_lib)
        ck.add("INFO", "libamdhip64", f"{hip_lib} -> {real}")

    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        ck.add("FAIL", "HSA_OVERRIDE_GFX_VERSION", f"set to {os.environ['HSA_OVERRIDE_GFX_VERSION']!r}; unset it "
               "(HALO's device code is built for gfx1151 and must run without an override)")
    else:
        ck.add("OK", "HSA_OVERRIDE_GFX_VERSION", "not set")

    if gpu_host:
        ck.add("OK", "GPU", f"KFD reports {facts['gfx_targets']}")
    elif facts["gfx_targets"]:
        ck.add("WARN", "GPU", f"KFD reports {facts['gfx_targets']}, not gfx1151")
    else:
        ck.add("N/A", "GPU", "no AMD GPU in the KFD topology (expected on the dev host, D-001)")

    kfd = Path("/dev/kfd")
    if kfd.exists():
        acc = os.access(kfd, os.R_OK | os.W_OK)
        ck.add("OK" if acc else "FAIL", "/dev/kfd", "read/write" if acc else
               "not accessible: add this user to the render group (sudo usermod -aG render,video $USER; log out and in)")
    else:
        ck.add("FAIL" if gpu_host else "N/A", "/dev/kfd", "missing")
    renders = sorted(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    if renders:
        acc = [r.name for r in renders if os.access(r, os.R_OK | os.W_OK)]
        ck.add("OK" if acc else ("FAIL" if gpu_host else "WARN"), "/dev/dri/renderD*",
               f"accessible: {acc}" if acc else f"none of {[r.name for r in renders]} accessible")
    else:
        ck.add("FAIL" if gpu_host else "N/A", "/dev/dri/renderD*", "none")
    groups = sorted({grp.getgrgid(g).gr_name for g in os.getgroups() if _has_gid(g)})
    ck.add("INFO", "groups", ", ".join(groups))

    rocm_v = C.read_sysfs("/opt/rocm/.info/version")
    facts["rocm_version"] = rocm_v
    if rocm_v:
        built = info.get("rocm_version_built_with")
        ck.add("INFO", "ROCm", f"/opt/rocm/.info/version = {rocm_v}; the kit was built with ROCm {built}"
               + ("" if rocm_v == built else " (different; the first run is the compatibility check)"))
    else:
        ck.add("WARN" if gpu_host else "N/A", "ROCm", "/opt/rocm/.info/version not found")
    hipcfg = C.which("hipconfig") or ("/opt/rocm/bin/hipconfig" if os.access("/opt/rocm/bin/hipconfig", os.X_OK) else None)
    if hipcfg:
        try:
            r = subprocess.run([hipcfg, "--version"], capture_output=True, timeout=30)
            ck.add("INFO", "hipconfig --version", r.stdout.decode(errors="replace").strip()[:200])
        except (OSError, subprocess.TimeoutExpired) as e:
            ck.add("WARN", "hipconfig --version", str(e))

    icds = C.vulkan_icds()
    facts["vulkan_icds"] = icds
    ck.add("OK" if icds["radv"] else ("FAIL" if gpu_host else "INFO"), "Vulkan RADV ICD",
           ", ".join(icds["radv"]) or "not installed (sudo apt install mesa-vulkan-drivers)")
    ck.add("INFO", "Vulkan AMDVLK ICD", ", ".join(icds["amdvlk"]) or "not installed (optional)")

    for dev in C.amdgpu_cards():
        vt = C.read_sysfs(dev / "mem_info_vram_total")
        gt = C.read_sysfs(dev / "mem_info_gtt_total")
        if vt:
            ck.add("INFO", f"{dev.parent.name} memory", f"VRAM {int(vt) / 2**30:.1f} GiB, GTT {int(gt or 0) / 2**30:.1f} GiB "
                   "(the engine report expects ~96 GiB VRAM carveout, D-002)")

    ref = C.find_ref(kit, ref_arg)
    if ref is None:
        ck.add("WARN", "reference data", "not found; tests that need it will SKIP (unpack halo-ref-*.tar.gz next to the kit, or pass --ref)")
    else:
        sums = ref / "REF_SHA256SUMS"
        h = hashlib.sha256(sums.read_bytes()).hexdigest()[:12] if sums.is_file() else "no REF_SHA256SUMS"
        ck.add("OK", "reference data", f"{ref} (content hash {h})")
        if check_ref and sums.is_file():
            nbad = 0
            for line in sums.read_text(encoding="utf-8").splitlines():
                digest, _, name = line.partition("  ")
                try:
                    hh = hashlib.sha256()
                    with open(ref / name, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            hh.update(chunk)
                    nbad += hh.hexdigest() != digest
                except OSError:
                    nbad += 1
            ck.add("FAIL" if nbad else "OK", "reference data integrity", f"{nbad} bad file(s)")
    try:
        notes = C.setup_anchor(kit, info["anchor"], ref)
        ck.add("OK", "test path anchor", f"{info['anchor']} ({len(notes)} links)")
    except C.KitError as e:
        ck.add("FAIL", "test path anchor", str(e))

    ck.add("OK" if os.access(C.GNU_TIME, os.X_OK) else "WARN", "GNU time",
           C.GNU_TIME if os.access(C.GNU_TIME, os.X_OK) else "missing: no peak-RSS figures (sudo apt install time)")
    free = shutil.disk_usage(os.getcwd()).free
    ck.add("OK" if free > 5 * 2**30 else "WARN", "free disk here", f"{free / 2**30:.1f} GiB in {os.getcwd()}")
    return ck, facts


def _has_gid(g: int) -> bool:
    try:
        grp.getgrgid(g)
        return True
    except KeyError:
        return False


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verify-env.sh", description="Pre-flight checks (docs/evox2.md).")
    ap.add_argument("--kit")
    ap.add_argument("--ref")
    ap.add_argument("--check-ref", action="store_true", help="also verify every reference file's SHA-256 (slow)")
    ap.add_argument("--json", help="write the checks as JSON to this file")
    a = ap.parse_args(argv)
    kit = Path(a.kit).resolve() if a.kit else C.kit_root_from(__file__)
    ck, facts = run(kit, a.ref, a.check_ref)
    nfail = sum(i["verdict"] == "FAIL" for i in ck.items)
    if a.json:
        C.write_json(Path(a.json), {"schema": "halo.kit.verify/1", "checks": ck.items, "facts": facts, "failures": nfail})
    print(f"verify-env: {'OK' if nfail == 0 else f'{nfail} FAIL'}")
    return 0 if nfail == 0 else 1
