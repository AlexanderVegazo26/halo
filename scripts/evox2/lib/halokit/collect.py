"""`collect`: the diagnostic bundle (scripts/evox2/collect.sh, docs/evox2.md step 3).

One command that records everything needed to judge a HALO run on the EVO-X2:
system / GPU / driver facts, HALO facts, every test binary with gtest XML, and (with --model)
inspect, a greedy trace-level generation, benchmarks, the tune lookup and llama.cpp baselines.

Every step runs with a timeout, never aborts the collection, and records its exact command
line, exit code, stdout, stderr, duration, peak RSS (GNU time) and a thermal/power snapshot
before and after. The bundle is halo-diag-<host>-<utc>/ plus its .tar.gz, with manifest.json
(machine-readable) and SUMMARY.md (failures first, then the expected-vs-actual checklist).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import common as C

SCHEMA = "halo.diag.bundle/1"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # colour codes (rocminfo) kept out of SUMMARY.md
GPU_KMSG = re.compile(r"amdgpu|kfd|\bdrm\b|\bgpu\b|hsa|iommu|\bttm\b|vram|gtt", re.I)
# Whole words only: a bare "fault" also matches "Default domain type" (seen on the dev host).
KMSG_BAD = re.compile(r"\b(?:errors?|fail\w*|fault\w*|timeouts?|timed out|resets?|reset\w*|hangs?|hung|oom|bug|"
                      r"warning|call trace|segfault)\b", re.I)


# --------------------------------------------------------------------------------------------
# Snapshots

def _num(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def thermal_snapshot() -> dict[str, Any]:
    """Cheap (tens of sysfs reads): temperatures, power, GPU busy and memory use, profile."""
    snap: dict[str, Any] = {"utc": C.utc_now()}
    temps = {}
    for tz in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = _num(C.read_sysfs(f"{tz}/temp"))
        if t is not None:
            temps[f"{os.path.basename(tz)}:{C.read_sysfs(f'{tz}/type')}"] = round(t / 1000, 1)
    hw: dict[str, Any] = {}
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = C.read_sysfs(f"{h}/name") or os.path.basename(h)
        vals = {}
        for f in sorted(glob.glob(f"{h}/temp*_input")):
            v = _num(C.read_sysfs(f))
            lab = C.read_sysfs(f.replace("_input", "_label")) or os.path.basename(f)
            if v is not None:
                vals[f"temp:{lab}"] = round(v / 1000, 1)
        for f in sorted(glob.glob(f"{h}/power*_average") + glob.glob(f"{h}/power*_input")):
            v = _num(C.read_sysfs(f))
            if v is not None:
                vals[os.path.basename(f)] = round(v / 1e6, 2)  # microwatts -> W
        for f in sorted(glob.glob(f"{h}/freq*_input")):
            v = _num(C.read_sysfs(f))
            lab = C.read_sysfs(f.replace("_input", "_label")) or os.path.basename(f)
            if v is not None:
                vals[f"freq:{lab}"] = round(v / 1e6)  # Hz -> MHz
        if vals:
            hw[f"{os.path.basename(h)}:{name}"] = vals
    snap["thermal_zones_c"] = temps
    snap["hwmon"] = hw
    gpus = {}
    for dev in C.amdgpu_cards():
        g: dict[str, Any] = {}
        for f in ("gpu_busy_percent", "mem_busy_percent", "mem_info_vram_used", "mem_info_gtt_used",
                  "power_dpm_force_performance_level"):
            v = C.read_sysfs(dev / f)
            if v is not None:
                g[f] = v
        sclk = C.read_sysfs(dev / "pp_dpm_sclk") or ""
        cur = [ln for ln in sclk.splitlines() if ln.rstrip().endswith("*")]
        if cur:
            g["sclk"] = cur[0]
        gpus[dev.parent.name] = g
    snap["amdgpu"] = gpus
    snap["platform_profile"] = C.read_sysfs("/sys/firmware/acpi/platform_profile")
    snap["loadavg"] = C.read_sysfs("/proc/loadavg")
    return snap


def sysfs_dump() -> str:
    """Full text of the amdgpu / power / memory-tier sysfs files (gpu/amdgpu-sysfs.txt)."""
    lines: list[str] = []

    def add(path: str) -> None:
        v = C.read_sysfs(path)
        if v is None:
            return
        if "\n" in v:
            lines.append(f"{path}:")
            lines.extend("    " + ln for ln in v.splitlines())
        else:
            lines.append(f"{path}: {v}")

    for dev in C.amdgpu_cards():
        d = str(dev)
        for f in ("pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_fclk", "pp_dpm_socclk", "pp_dpm_dcefclk", "pp_dpm_pcie",
                  "power_dpm_force_performance_level", "power_dpm_state", "pp_power_profile_mode", "pp_features",
                  "pp_od_clk_voltage", "gpu_busy_percent", "mem_busy_percent", "mem_info_vram_total",
                  "mem_info_vram_used", "mem_info_vis_vram_total", "mem_info_vis_vram_used", "mem_info_gtt_total",
                  "mem_info_gtt_used", "mem_info_vram_vendor", "vbios_version", "current_link_speed",
                  "current_link_width", "device", "vendor", "revision", "subsystem_device"):
            add(f"{d}/{f}")
        for h in sorted(glob.glob(f"{d}/hwmon/hwmon*")):
            for f in sorted(os.listdir(h)):
                p = f"{h}/{f}"
                if os.path.isfile(p) and not f.startswith(("uevent",)):
                    add(p)
    for p in sorted(glob.glob("/sys/module/amdgpu/parameters/*")):
        add(p)
    for p in ("/sys/module/ttm/parameters/pages_limit", "/sys/module/ttm/parameters/page_pool_size",
              "/sys/firmware/acpi/platform_profile", "/sys/firmware/acpi/platform_profile_choices",
              "/sys/devices/system/cpu/amd_pstate/status", "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver",
              "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
              "/sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference",
              "/sys/devices/system/cpu/cpufreq/boost", "/sys/kernel/mm/transparent_hugepage/enabled",
              "/proc/sys/kernel/numa_balancing", "/sys/module/amdgpu/version"):
        add(p)
    for node in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*")):
        add(f"{node}/name")
        add(f"{node}/properties")
        for mb in sorted(glob.glob(f"{node}/mem_banks/*")):
            add(f"{mb}/properties")
    if not lines:
        lines.append("no amdgpu / KFD sysfs on this host")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# The bundle

class Bundle:
    def __init__(self, root: Path, a: argparse.Namespace, gpu_host: bool, kit: Path | None) -> None:
        self.root = root
        self.a = a
        self.gpu_host = gpu_host
        self.kit = kit
        self.steps: list[dict[str, Any]] = []
        self.n = 0
        self.per_step_snapshots = not a.no_step_snapshots
        self.facts: dict[str, Any] = {}
        self.kitinfo: dict[str, Any] | None = None

    def _dir(self, category: str, name: str) -> Path:
        self.n += 1
        d = self.root / category / f"{self.n:02d}-{name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def step(self, name: str, argv: list[str], *, category: str, timeout: float, expect: str = "pass",
             env_extra: dict[str, str] | None = None, cwd: str | None = None, note: str = "",
             stdout_name: str = "stdout.txt", accept_rc: tuple[int, ...] = (0,)) -> dict[str, Any]:
        """expect: pass | pass-on-gpu-host | fail (a clean failure is the expected result) | info."""
        if name in self.a.skip:
            return self._record_skip(name, category, argv, "skipped by --skip")
        d = self._dir(category, name)
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        before = thermal_snapshot() if self.per_step_snapshots else None
        print(f"collect: [{category}] {name} ...", flush=True)
        rec = C.run_cmd(argv, stdout_path=d / stdout_name, stderr_path=d / "stderr.txt",
                        timeout=timeout * self.a.timeout_scale, cwd=cwd, env=env, time_v_path=d / "time-v.txt")
        if rec["status"] == "fail" and rec.get("exit_code") in accept_rc:
            rec["status"] = "pass"
        rec.update(name=name, category=category, expect=expect, dir=str(d.relative_to(self.root)),
                   env_overrides={k: ("<redacted>" if C.is_secret_name(k) else v) for k, v in (env_extra or {}).items()})
        if note:
            rec["note"] = (rec.get("note", "") + " " + note).strip()
        if before is not None:
            rec["thermal_before"] = before
            rec["thermal_after"] = thermal_snapshot()
        rec["verdict"] = self._verdict(rec)
        rec["stdout"] = str(Path(rec["stdout"]).relative_to(self.root))
        rec["stderr"] = str(Path(rec["stderr"]).relative_to(self.root))
        if "time_v" in rec:
            rec["time_v"] = str(Path(rec["time_v"]).relative_to(self.root))
        (d / "command.txt").write_text(rec["command"] + "\n", encoding="utf-8")
        C.write_json(d / "result.json", rec)
        print(f"collect: [{category}] {name}: {rec['status']} (exit {rec.get('exit_code')}, {rec.get('duration_s')} s) -> {rec['verdict']}",
              flush=True)
        self.steps.append(rec)
        return rec

    def internal(self, name: str, category: str, desc: str, fn: Callable[[Path], str | None],
                 expect: str = "pass") -> dict[str, Any]:
        """A step done in Python (reading sysfs, env): fn writes into the step dir, returns a note."""
        if name in self.a.skip:
            return self._record_skip(name, category, [f"internal:{desc}"], "skipped by --skip")
        d = self._dir(category, name)
        t0 = time.monotonic()
        rec: dict[str, Any] = {"name": name, "category": category, "command": f"internal: {desc}", "expect": expect,
                               "started_utc": C.utc_now(), "dir": str(d.relative_to(self.root))}
        try:
            note = fn(d)
            rec.update(status="pass", exit_code=0)
            if note:
                rec["note"] = note
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            rec.update(status="fail", exit_code=1, note=f"{type(e).__name__}: {e}")
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        rec["verdict"] = self._verdict(rec)
        (d / "command.txt").write_text(rec["command"] + "\n", encoding="utf-8")
        C.write_json(d / "result.json", rec)
        self.steps.append(rec)
        return rec

    def _record_skip(self, name: str, category: str, argv: list[str], why: str) -> dict[str, Any]:
        rec = {"name": name, "category": category, "command": C.shell_join(argv), "status": "skipped",
               "verdict": "SKIPPED", "note": why, "exit_code": None, "duration_s": 0}
        self.steps.append(rec)
        return rec

    def _verdict(self, r: dict[str, Any]) -> str:
        st, exp = r["status"], r.get("expect", "pass")
        if st == "not-available":
            return "NOT-AVAILABLE"
        if st == "timeout":
            return "FAIL"
        if exp == "info":
            return "OK" if st == "pass" else "INFO"
        if exp == "fail":
            return "EXPECTED-FAIL" if st in ("fail",) else ("UNEXPECTED-PASS" if st == "pass" else "FAIL")
        if st == "pass":
            return "OK"
        if exp == "pass-on-gpu-host" and not self.gpu_host:
            return "EXPECTED-FAIL"
        return "FAIL"


# --------------------------------------------------------------------------------------------
# Steps

def kernel_log_step(b: Bundle) -> None:
    def fn(d: Path) -> str:
        import subprocess
        for argv in (["dmesg", "-T"], ["sudo", "-n", "dmesg", "-T"], ["journalctl", "-k", "-b", "--no-pager", "-q"]):
            if C.which(argv[0]) is None:
                continue
            try:
                r = subprocess.run(argv, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0 and r.stdout.strip():
                lines = r.stdout.decode("utf-8", errors="replace").splitlines()
                gpu = [ln for ln in lines if GPU_KMSG.search(ln)]
                bad = [ln for ln in gpu if KMSG_BAD.search(ln)]
                (d / "dmesg-gpu.txt").write_text("\n".join(gpu) + "\n", encoding="utf-8")
                (d / "dmesg-gpu-problems.txt").write_text("\n".join(bad) + "\n", encoding="utf-8")
                b.facts["kmsg"] = {"method": C.shell_join(argv), "lines": len(lines), "gpu_lines": len(gpu),
                                   "gpu_problem_lines": len(bad)}
                return f"via `{C.shell_join(argv)}`: {len(gpu)} GPU lines, {len(bad)} with error/fault/reset words"
        b.facts["kmsg"] = None
        raise RuntimeError("kernel log not readable (dmesg_restrict, no passwordless sudo, no journal access): "
                           "run `sudo -v` right before collect.sh, or run it as root")
    # expect=info: an unreadable kernel log (no sudo) is reported (INFO here, UNKNOWN in the
    # checklist), not a failed collection.
    b.internal("kernel-log", "gpu", "dmesg | sudo -n dmesg | journalctl -k, filtered to amdgpu/kfd/drm/iommu", fn,
               expect="info")


def system_steps(b: Bundle) -> None:
    T = 60
    b.step("uname", ["uname", "-a"], category="system", timeout=T)
    b.step("os-release", ["cat", "/etc/os-release"], category="system", timeout=T)
    b.step("kernel-cmdline", ["cat", "/proc/cmdline"], category="system", timeout=T)
    b.step("lscpu", ["lscpu"], category="system", timeout=T)
    b.step("lscpu-json", ["lscpu", "-J"], category="system", timeout=T, expect="info")
    b.step("free", ["free", "-h", "-w"], category="system", timeout=T)
    b.step("free-bytes", ["free", "-b", "-w"], category="system", timeout=T)
    b.step("meminfo", ["cat", "/proc/meminfo"], category="system", timeout=T)
    b.step("numactl", ["numactl", "-H"], category="system", timeout=T, expect="info")
    b.step("lsmem", ["lsmem"], category="system", timeout=T, expect="info")
    b.step("virt", ["systemd-detect-virt"], category="system", timeout=T, expect="info")
    b.step("id", ["id"], category="system", timeout=T)

    def env_fn(d: Path) -> str:
        env = C.redacted_env() if b.a.redact else dict(sorted(os.environ.items()))
        (d / "env.txt").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
        gpu = {k: v for k, v in env.items() if k.startswith(C.GPU_ENV_PREFIXES)}
        C.write_json(d / "gpu-env.json", gpu)
        b.facts["gpu_env"] = gpu
        n = sum(1 for k in os.environ if C.is_secret_name(k))
        return f"{len(env)} variables, {n} secret-named values redacted" if b.a.redact else "REDACTION DISABLED (--no-redact)"
    b.internal("environment", "system", "os.environ, secret-named values redacted", env_fn)


def gpu_steps(b: Bundle, when: str) -> None:
    T = 120
    if when == "before":
        kernel_log_step(b)
        b.step("rocm-version", ["cat", "/opt/rocm/.info/version"], category="gpu", timeout=T, expect="pass-on-gpu-host")
        rocm_dirs = sorted(glob.glob("/opt/rocm*"))
        if rocm_dirs:
            b.step("rocm-dirs", ["ls", "-ld"] + rocm_dirs, category="gpu", timeout=T, expect="info")
        # hipconfig --full took 116 s on the (loaded) dev host: give it room.
        b.step("hipconfig", [C.which("hipconfig") or "/opt/rocm/bin/hipconfig", "--full"], category="gpu", timeout=600,
               expect="pass-on-gpu-host")
        b.step("rocminfo", [C.which("rocminfo") or "/opt/rocm/bin/rocminfo"], category="gpu", timeout=T,
               expect="pass-on-gpu-host")
        b.step("rocm-smi-showall", [C.which("rocm-smi") or "/opt/rocm/bin/rocm-smi", "--showall"], category="gpu",
               timeout=T, expect="pass-on-gpu-host")
        if C.which("amd-smi") or os.access("/opt/rocm/bin/amd-smi", os.X_OK):
            smi = C.which("amd-smi") or "/opt/rocm/bin/amd-smi"
            b.step("amd-smi-static", [smi, "static"], category="gpu", timeout=T, expect="info")
            b.step("amd-smi-metric", [smi, "metric"], category="gpu", timeout=T, expect="info")
        b.step("lspci-gpu", ["lspci", "-nnk", "-d", "1002:"], category="gpu", timeout=T, expect="info")
        b.step("amdgpu-module-version", ["modinfo", "-F", "version", "amdgpu"], category="gpu", timeout=T, expect="info")
        b.step("dev-nodes", ["ls", "-l", "/dev/kfd", "/dev/dri"], category="gpu", timeout=T, expect="pass-on-gpu-host")
        b.step("vulkaninfo-summary", ["vulkaninfo", "--summary"], category="gpu", timeout=T)
        b.step("vulkaninfo-full", ["vulkaninfo"], category="gpu", timeout=T, stdout_name="vulkaninfo.txt")
        for fam in ("radv", "amdvlk"):
            env, why = C.icd_env(fam)
            if env is None:
                b._record_skip(f"vulkaninfo-{fam}", "gpu", ["vulkaninfo", "--summary"], why)
            else:
                b.step(f"vulkaninfo-{fam}", ["vulkaninfo", "--summary"], category="gpu", timeout=T, env_extra=env,
                       expect="pass-on-gpu-host" if fam == "radv" else "info", note=why)
        b.step("clinfo", ["clinfo"], category="gpu", timeout=T, expect="info")

        def sysfs_fn(d: Path) -> str:
            (d / "amdgpu-sysfs.txt").write_text(sysfs_dump(), encoding="utf-8")
            vram = None
            for dev in C.amdgpu_cards():
                vram = C.read_sysfs(dev / "mem_info_vram_total")
                b.facts["vram_total"] = int(vram) if vram else None
                b.facts["gtt_total"] = int(C.read_sysfs(dev / "mem_info_gtt_total") or 0)
            return f"VRAM total {int(vram) / 2**30:.1f} GiB" if vram else "no amdgpu sysfs"
        b.internal("amdgpu-sysfs", "gpu", "read amdgpu/hwmon/KFD/power sysfs", sysfs_fn)
    b.step(f"rocm-smi-mem-{when}", [C.which("rocm-smi") or "/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram", "gtt"],
           category="gpu", timeout=T, expect="pass-on-gpu-host")


def halo_steps(b: Bundle, halo: str, kit: Path) -> None:
    T = 300
    kpy = str(kit / "scripts" / "evox2" / "lib" / "halo_kit.py")
    ref = ["--ref", b.a.ref] if b.a.ref else []
    b.step("verify-env", [sys.executable, kpy, "verify-env", "--kit", str(kit), "--json",
                          str(b.root / "halo" / "verify-env.json")] + ref, category="halo", timeout=T,
           expect="pass-on-gpu-host")
    b.step("halo-version", [halo, "version"], category="halo", timeout=T)
    b.step("halo-devices", [halo, "devices"], category="halo", timeout=T)
    b.step("halo-devices-json", [halo, "devices", "--json"], category="halo", timeout=T, stdout_name="devices.json")
    b.step("halo-devices-verify", [halo, "devices", "--verify", "--json"], category="halo", timeout=T,
           expect="pass-on-gpu-host", stdout_name="devices-verify.json")

    def ldd_fn(d: Path) -> str:
        import subprocess
        from .verify import elf_files
        out, nf = [], 0
        kit_env = C.rocm_lib_env()  # the LD_LIBRARY_PATH run-tests.sh gives the test_hip_* binaries
        hip_env = {**os.environ, **kit_env}
        out.append(f"# ldd; test_hip_* with the kit's run environment {kit_env or '(no change)'}\n")
        for p in elf_files(kit):
            r = subprocess.run(["ldd", str(p)], capture_output=True, timeout=30,
                               env=hip_env if p.name.startswith("test_hip") else None)
            txt = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            nf += txt.count("not found")
            out.append(f"== {p.relative_to(kit)}\n{txt}")
        (d / "ldd.txt").write_text("\n".join(out), encoding="utf-8")
        if nf:
            raise RuntimeError(f"{nf} unresolved librar(ies); see ldd.txt")
        return f"{len(out) - 1} binaries, all libraries resolved"
    b.internal("ldd-all", "halo", "ldd of every kit binary", ldd_fn)
    b.step("bench-micro", [halo, "bench", "micro", "--host-label", b.a.host_label, "--power-mode", b.a.power_mode,
                           "--out", str(b.root / "halo" / "bench-micro.json")], category="halo", timeout=1800)


def test_steps(b: Bundle, kit: Path) -> None:
    kpy = str(kit / "scripts" / "evox2" / "lib" / "halo_kit.py")
    common = ["--kit", str(kit)] + (["--ref", b.a.ref] if b.a.ref else []) + \
             (["--llama-dir", b.a.llama_dir] if b.a.llama_dir else []) + ["--expect-gpu", b.a.expect_gpu]
    tdir = b.root / "tests"
    main = [sys.executable, kpy, "run-tests", "--label", "main", "--out", str(tdir / "main")] + common
    if not b.a.quick:
        main.append("--full-bandwidth")
    b.step("tests-main", main, category="tests", timeout=7200,
           note="all binaries; HIP/Vulkan device tests enabled" + ("" if b.a.quick else "; HALO_BANDWIDTH_FULL=1"))
    if b.a.quick:
        return
    for fam in ("radv", "amdvlk"):
        b.step(f"tests-vulkan-{fam}", [sys.executable, kpy, "run-tests", "--label", f"vulkan-{fam}", "--category", "vulkan",
                                       "--vk-icd", fam, "--out", str(tdir / f"vulkan-{fam}")] + common,
               category="tests", timeout=3600, expect="pass-on-gpu-host" if fam == "radv" else "info",
               note=f"Vulkan tests with the loader restricted to {fam} (VK_DRIVER_FILES/VK_ICD_FILENAMES)")


def model_steps(b: Bundle, halo: str) -> None:
    a = b.a
    mtp = ["--mtp", a.mtp] if a.mtp else []
    mdir = b.root / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    trace = {"HALO_LOG_LEVEL": "trace", "HALO_LOG_FORMAT": "json"}
    b.step("inspect-json", [halo, "inspect", a.model, "--json"] + mtp, category="model", timeout=600,
           stdout_name="inspect.json")
    b.step("inspect", [halo, "inspect", a.model] + mtp, category="model", timeout=600)
    run_common = [halo, "run", a.model] + mtp + ["-p", a.prompt, "-n", str(a.max_tokens), "--temperature", "0",
                                                  "--seed", "1", "--ctx", str(a.ctx), "--parallel", "1"]
    b.step("run-greedy", run_common, category="model", timeout=7200,
           env_extra={**trace, "HALO_LOG_FILE": str(mdir / "run-greedy.log.jsonl")},
           note="backend auto; the backend that actually ran is read from the trace log (SUMMARY.md checklist)")
    for be in ("hip", "vulkan"):
        b.step(f"run-backend-{be}", [halo, "run", a.model] + mtp + ["-p", "hi", "-n", "1", "--temperature", "0",
                                                                    "--ctx", "512", "--parallel", "1", "--backend", be],
               category="model", timeout=1800, expect="fail",
               env_extra={**trace, "HALO_LOG_FILE": str(mdir / f"run-backend-{be}.log.jsonl")},
               note="probe: the model forward is not wired to the GPU yet (ADR-001 BI-1..BI-6); a clean "
                    "'not available' error is the expected result today")
    bench = [halo, "bench", "model", "--model", a.model] + mtp + [
        "--host-label", a.host_label, "--power-mode", a.power_mode, "--modes", "decode", "--contexts", str(a.ctx),
        "--concurrency", "1", "--prompt-tokens", str(a.bench_prompt_tokens), "--decode-tokens", str(a.bench_decode_tokens),
        "--warmup", "1", "--repetitions", "1", "--allow-nonconformant", "--out", str(mdir / "bench-model.json")]
    if a.bench_args:
        bench += a.bench_args.split()
    b.step("bench-model", bench, category="model", timeout=10800, env_extra={"HALO_LOG_FORMAT": "json",
           "HALO_LOG_FILE": str(mdir / "bench-model.log.jsonl")},
           note="smoke-sized (--allow-nonconformant); on the CPU engine, so NOT a GPU performance number")
    if a.bench_system:
        wl = Path(b.kit or ".") / "src" / "bench" / "workloads" / "concurrency_4agents.json"
        b.step("bench-system", [halo, "bench", "system", "--model", a.model] + mtp + [
            "--host-label", a.host_label, "--power-mode", a.power_mode, "--workload", str(wl), "--agents", "4",
            "--warmup", "0", "--repetitions", "1", "--allow-nonconformant", "--out", str(mdir / "bench-system.json")],
            category="model", timeout=14400)
    b.step("tune-list", [halo, "tune", "--list", "--model", a.model] + mtp + [
        "--host-label", a.host_label, "--power-mode", a.power_mode, "--report", str(mdir / "tune-list.json")],
        category="model", timeout=600, expect="info",
        note="read-only lookup; a missing profile DB is reported, not created"
             + ("; halo tune rejects power mode 'unknown' (CONFIG_ERROR): pass --power-mode" if a.power_mode == "unknown" else ""))


def baseline_steps(b: Bundle) -> None:
    a = b.a
    d = Path(a.llama_dir)
    cands = [d / "llama-bench", d / "bin" / "llama-bench", d / "build" / "bin" / "llama-bench"]
    lb = next((str(c) for c in cands if os.access(c, os.X_OK)), None)
    if lb is None:
        b._record_skip("llama-bench", "baselines", ["llama-bench"], f"no llama-bench under {d}")
        return
    b.step("llama-bench", [lb, "-m", a.model, "-ngl", "999", "-fa", "1", "-p", "512", "-n", "128", "-r", "3", "-o", "json"]
           + (a.llama_args.split() if a.llama_args else []),
           category="baselines", timeout=3600, stdout_name="llama-bench.json",
           note="llama.cpp same-file baseline (docs/strix-halo-qwen38-engine-report.md: ~20 tok/s decode)")
    cli = Path(lb).with_name("llama-cli")
    if os.access(cli, os.X_OK):
        b.step("llama-cli-version", [str(cli), "--version"], category="baselines", timeout=60, expect="info")


# --------------------------------------------------------------------------------------------
# Checklist and summary

def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _step(b: Bundle, name: str) -> dict[str, Any] | None:
    return next((s for s in b.steps if s["name"] == name), None)


def _out(b: Bundle, name: str) -> str:
    s = _step(b, name)
    return C.read_text(b.root / s["stdout"]) if s and "stdout" in s and s.get("status") != "skipped" else ""


BACKEND_MSG = re.compile(r"backend|cpu reference|\bhip\b|vulkan|gfx1151|device", re.I)


def backend_evidence(log: Path) -> tuple[str, list[str]]:
    """(backend that ran, the log lines that say so) from a HALO JSON-lines trace log.

    HALO has no single "backend = X" line yet, so this reads what the engine does log: the
    CPU reference engine says "CPU reference" when it plans memory. Anything that mentions a
    GPU backend is returned too, so a GPU run (once ADR-001 BI-6 lands) shows up as evidence
    rather than being hidden behind the old expectation.
    """
    lines: list[str] = []
    for raw in C.read_text(log, limit=64 << 20).splitlines():
        try:
            j = json.loads(raw)
        except ValueError:
            continue
        msg = str(j.get("msg", "")) if isinstance(j, dict) else ""
        if BACKEND_MSG.search(msg):
            lines.append(f"{j.get('component')}: {msg}"[:240])
    text = "\n".join(lines).lower()
    if re.search(r"\b(hip|vulkan)\b.*(forward|decode|step|selected|using)", text):
        ran = "gpu (see evidence)"
    elif "cpu reference" in text:
        ran = "cpu"
    else:
        ran = "unknown (no backend line in the log)"
    return ran, lines[:6]


def checklist(b: Bundle) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    gh = b.gpu_host

    def add(check: str, expected: str, actual: str, verdict: str, where: str = "") -> None:
        items.append({"check": check, "expected": expected, "actual": actual, "verdict": verdict, "where": where})

    gfx = C.kfd_gfx_targets()
    add("GPU in the KFD topology", "gfx1151 (Radeon 8060S)", ", ".join(gfx) or "none",
        "OK" if "gfx1151" in gfx else ("MISMATCH" if gfx or gh else "N/A (dev host, D-001)"), "gpu/*amdgpu-sysfs")
    hov = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    add("HSA_OVERRIDE_GFX_VERSION", "unset", hov or "unset", "OK" if not hov else "MISMATCH", "system/*environment")
    rv = C.read_sysfs("/opt/rocm/.info/version")
    built = (b.kitinfo or {}).get("rocm_version_built_with")
    add("ROCm on this host vs the kit build", f"target HIP 7.15.26333 / ROCm runtime 1.21; kit built with {built}",
        rv or "no /opt/rocm", "INFO" if rv else ("N/A" if not gh else "MISMATCH"), "gpu/*rocm-version, gpu/*hipconfig")
    ri = _out(b, "rocminfo")
    add("rocminfo lists a gfx1151 agent", "yes", "yes" if "gfx1151" in ri else ("no" if ri else "rocminfo did not run"),
        "OK" if "gfx1151" in ri else ("N/A (dev host)" if not gh else "MISMATCH"), "gpu/*rocminfo")
    vt = b.facts.get("vram_total")
    if vt:
        gib = vt / 2**30
        add("VRAM carveout (D-002)", "~96 GiB", f"{gib:.1f} GiB", "OK" if 90 <= gib <= 100 else "MISMATCH", "gpu/*amdgpu-sysfs")
    else:
        add("VRAM carveout (D-002)", "~96 GiB", "no amdgpu sysfs", "N/A (dev host)" if not gh else "MISMATCH", "gpu/*amdgpu-sysfs")
    vs = _step(b, "vulkaninfo-radv")
    vr = _out(b, "vulkaninfo-radv")
    radv_ok = "RADV" in vr and re.search(r"gfx1151|8060S|GFX1151|STRIX", vr, re.I) is not None
    if radv_ok:
        radv_actual = "yes"
    elif vs is None or vs.get("status") == "skipped":
        radv_actual = "RADV ICD not installed"
    else:
        err = C.read_text(b.root / vs["stderr"]).strip().splitlines() if vs.get("stderr") else []
        radv_actual = f"RADV ICD installed; no RADV gfx1151 device (exit {vs.get('exit_code')}" + (f": {err[-1][:120]}" if err else "") + ")"
    add("Vulkan RADV sees the GPU", "RADV device for gfx1151 / 8060S", radv_actual,
        "OK" if radv_ok else ("N/A (dev host)" if not gh else "MISMATCH"), "gpu/*vulkaninfo-radv")
    ver = _read_json(b.root / "halo" / "verify-env.json")
    if ver:
        fails = [c for c in ver["checks"] if c["verdict"] == "FAIL"]
        libs = next((c for c in ver["checks"] if c["check"] == "shared libraries"), None)
        add("Kit binaries resolve their libraries", "no 'not found'", libs["detail"] if libs else "?",
            "OK" if libs and libs["verdict"] == "OK" else "MISMATCH", "halo/verify-env.json")
        add("verify-env", "no FAIL", f"{len(fails)} FAIL: " + "; ".join(f"{c['check']}: {c['detail'][:80]}" for c in fails[:4]),
            "OK" if not fails else ("EXPECTED (dev host)" if not gh else "MISMATCH"), "halo/verify-env.json")
    dv = _read_json(b.root / (_step(b, "halo-devices-verify") or {}).get("stdout", "none")) if _step(b, "halo-devices-verify") else None
    if isinstance(dv, dict):
        failed = [c for c in dv.get("checklist", []) if isinstance(c, dict) and c.get("status") == "fail"]
        add("halo devices --verify", "overall not fail", f"overall {dv.get('overall')}; failed: "
            + ", ".join(f"{c.get('check') or c.get('id') or c.get('name')} ({str(c.get('detail', ''))[:60]})" for c in failed[:6]),
            "OK" if dv.get("overall") != "fail" else ("EXPECTED (dev host)" if not gh else "MISMATCH"), "halo/*halo-devices-verify")
    tm = _read_json(b.root / "tests" / "main" / "summary.json")
    if tm and "totals" in tm:
        t = tm["totals"]
        add("Unit tests (all binaries)", "0 failed, 0 crashed",
            f"passed {t['passed']}, failed {t['failed']}, crashed {t['crashed']}, timeout {t['timeout']}, skipped {t['skipped']} {tm['skips_by_category']}",
            "OK" if t["failed"] == t["crashed"] == t["timeout"] == 0 else "MISMATCH", "tests/main/summary.md")
        hip = [x for bb in tm["binaries"] if bb["category"] == "hip" for x in bb["tests"]]
        hip_skip = [x for x in hip if x["status"] == "skipped" and x.get("skip_category") == "no-device"]
        # Device tests live in suites named *Device (HipGemvDevice, HipRuntimeDevice, ...);
        # HipRuntime.ProbeNeverThrowsAndExplainsZeroDevices is a no-device test and must not count.
        hip_ran = [x for x in hip if x["name"].split(".", 1)[0].endswith("Device") and x["status"] == "passed"]
        add("HIP device tests ran on the GPU", "every *Device* test passed, none skipped (gfx1151 host)",
            f"{len(hip_ran)} device tests passed, {len(hip_skip)} skipped for no device",
            ("OK" if not hip_skip and hip_ran else "MISMATCH") if gh else ("N/A (dev host: skips expected, D-001)" if hip_skip else "INFO"),
            "tests/main/summary.md")
        bw = [x for bb in tm["binaries"] for x in bb["tests"] if x["name"] == "Bandwidth.ConformantDefaultsDevHost"]
        bw_line = ""
        hw_log = next((bb["log"] for bb in tm["binaries"] if bb["name"] == "test_hardware"), None)
        if hw_log:
            m = re.search(r"\[dev-host, 256 MiB, NOT a HALO performance claim\] (\{.*\})", C.read_text(Path(hw_log)))
            if m:
                try:
                    j = json.loads(m.group(1))
                    bw_line = f"host read median {j['read_gbps']['median']:.1f} GB/s, copy {j['copy_gbps']['median']:.1f} GB/s"
                except (ValueError, KeyError, TypeError):
                    bw_line = m.group(1)[:200]
        add("Full host-bandwidth benchmark (HALO_BANDWIDTH_FULL=1)", "ran", (bw[0]["status"] if bw else "absent") + (f"; {bw_line}" if bw_line else ""),
            "OK" if bw and bw[0]["status"] == "passed" else ("SKIPPED (--quick)" if b.a.quick else "MISMATCH"), "tests/main/logs/test_hardware.log")
    else:
        add("Unit tests (all binaries)", "ran", "no summary (see tests/*tests-main)", "MISMATCH" if not b.a.no_tests else "SKIPPED", "tests/")
    for fam in ("radv", "amdvlk"):
        ts = _read_json(b.root / "tests" / f"vulkan-{fam}" / "summary.json")
        if not ts:
            continue
        if ts.get("not_available"):
            add(f"Vulkan tests on {fam}", "ran" if fam == "radv" else "ran if installed", ts["not_available"],
                "MISMATCH" if (fam == "radv" and gh) else "N/A", f"tests/vulkan-{fam}")
            continue
        t = ts["totals"]
        nodev = ts["skips_by_category"].get("no-device", 0)
        add(f"Vulkan tests on {fam}", "all pass, no 'Vulkan unavailable' skip",
            f"passed {t['passed']}, failed {t['failed']}, skipped {t['skipped']} (no-device {nodev})",
            ("OK" if t["failed"] == 0 and nodev == 0 and t["passed"] > 0 else "MISMATCH") if gh else
            ("N/A (dev host)" if nodev else ("OK" if t["failed"] == 0 else "MISMATCH")), f"tests/vulkan-{fam}/summary.md")
    km = b.facts.get("kmsg")
    add("Kernel log: GPU errors/faults/resets", "none", "not readable" if km is None else f"{km['gpu_problem_lines']} of {km['gpu_lines']} GPU lines",
        "UNKNOWN" if km is None else ("OK" if km["gpu_problem_lines"] == 0 else "CHECK"), "gpu/*kernel-log/dmesg-gpu-problems.txt")
    if b.a.model:
        rg = _step(b, "run-greedy")
        err = C.read_text(b.root / rg["stderr"]) if rg and "stderr" in rg else ""
        m = re.search(r"\[halo\] prompt .*", err)
        add("halo run (greedy, trace log)", "exit 0; backend that ran = cpu (GPU forward not wired, BI-6)",
            (f"exit {rg.get('exit_code')}; " if rg else "") + (m.group(0) if m else "no summary line"),
            "OK" if rg and rg["verdict"] == "OK" else "MISMATCH", "model/*run-greedy")
        if rg and rg.get("status") != "skipped":
            ran, ev = backend_evidence(b.root / "model" / "run-greedy.log.jsonl")
            if ran.startswith("unknown"):
                # A GPU host plans GPU tiers, so the "CPU reference" plan line may be absent. The
                # --backend probes then say whether this build can run anything but the CPU engine.
                probes = [C.read_text(b.root / s["stderr"]) for s in (_step(b, "run-backend-hip"), _step(b, "run-backend-vulkan"))
                          if s and s.get("stderr")]
                if probes and all("cpu reference only" in p for p in probes):
                    ran = "cpu"
                    ev = ["inferred: --backend hip and --backend vulkan both fail with 'cpu reference only'"] + ev
            b.facts["run_greedy_backend"] = {"ran": ran, "evidence": ev}
            add("Backend that ran `halo run` (from the trace log)", "cpu until ADR-001 BI-6 wires the GPU forward",
                f"{ran}; " + (" / ".join(ev[:2]) if ev else "no backend-related log line"),
                "OK" if ran == "cpu" else ("CHECK (GPU forward now runs?)" if ran.startswith("gpu") else "UNKNOWN"),
                "model/run-greedy.log.jsonl")
        for be in ("hip", "vulkan"):
            s = _step(b, f"run-backend-{be}")
            if s:
                e = C.read_text(b.root / s["stderr"]).strip().splitlines()
                add(f"halo run --backend {be}", "clean 'not available' error until BI-6",
                    f"exit {s.get('exit_code')}: {(e[-1] if e else '')[:160]}",
                    {"EXPECTED-FAIL": "OK", "UNEXPECTED-PASS": "CHECK (GPU backend now runs?)"}.get(s["verdict"], "MISMATCH"),
                    f"model/*run-backend-{be}")
        bm = _read_json(b.root / "model" / "bench-model.json")
        if isinstance(bm, dict):
            add("halo bench model artifact", "written (smoke; CPU engine)", f"valid={bm.get('valid')}, conformant={bm.get('conformant')}",
                "INFO", "model/bench-model.json")
        lbj = _read_json(b.root / (_step(b, "llama-bench") or {}).get("stdout", "none")) if _step(b, "llama-bench") else None
        if isinstance(lbj, list):
            rows = [f"{r.get('n_prompt')}/{r.get('n_gen')}: {r.get('avg_ts', 0):.1f} t/s" for r in lbj if isinstance(r, dict)]
            add("llama.cpp baseline (llama-bench)", "pp512 / tg128 recorded (report: ~20 tok/s decode)", "; ".join(rows) or "?",
                "INFO", "baselines/*llama-bench")
    temps = []
    for s in b.steps:
        for key in ("thermal_before", "thermal_after"):
            for v in (s.get(key) or {}).get("thermal_zones_c", {}).values():
                temps.append(v)
    if temps:
        add("Thermal range over the run", "record", f"{min(temps):.1f}-{max(temps):.1f} C (thermal zones)", "INFO", "manifest.json thermal_*")
    return items


def write_summary(b: Bundle, manifest: dict[str, Any]) -> None:
    L: list[str] = []
    m = manifest
    L += [f"# HALO diagnostic bundle: {m['host']} {m['created_utc']}", "",
          "Read section 1 first; each entry names the file to open. Section 2 compares what this run",
          "saw with what HALO expects at the current MVP stage (docs/evox2.md, \"What to expect\").", "",
          f"- kit: `{m['kit'].get('kit_name')}` (git {m['kit'].get('git_sha')}, dirty={m['kit'].get('dirty')}, "
          f"-march={m['kit'].get('march')}, built with ROCm {m['kit'].get('rocm_version_built_with')})",
          f"- host is a gfx1151 GPU host: **{m['gpu_host']}**; host label `{m['args'].get('host_label')}`, "
          f"power mode `{m['args'].get('power_mode')}`",
          f"- model: `{m['args'].get('model')}`; llama.cpp: `{m['args'].get('llama_dir')}`",
          f"- steps: {len(b.steps)}; duration {m['duration_s']} s; verdict **{m['verdict']}**", ""]
    fails = [s for s in b.steps if s.get("verdict") in ("FAIL",)]
    notes = [s for s in b.steps if s.get("verdict") in ("UNEXPECTED-PASS",)]
    L += ["## 1. Failures (read these first)", ""]
    cerr = m.get("collector_errors", [])
    if not fails and not m["test_failures"] and not m["expectation_failures"] and not notes and not cerr:
        L += ["None.", ""]
    for e in cerr:
        L.append(f"- **collector error** (a bug in collect.sh; tracebacks in `collector-errors.txt`): {e}")
    for s in fails:
        tail = C.read_text(b.root / s["stderr"]).strip().splitlines()[-3:] if s.get("stderr") else []
        if not tail and s.get("stdout"):
            tail = C.read_text(b.root / s["stdout"]).strip().splitlines()[-3:]
        L.append(f"- **{s['category']}/{s['name']}**: {s['status']} (exit {s.get('exit_code')}, {s.get('duration_s')} s). "
                 f"Command: `{s['command'][:300]}`")
        L.append(f"  - logs: `{s.get('stdout', '')}`, `{s.get('stderr', '')}`" + (f"; note: {s['note']}" if s.get("note") else ""))
        for t in tail:
            L.append(f"  - > {ANSI_RE.sub('', t)[:300]}")
    for f in m["test_failures"]:
        L.append(f"- **test** `{f['run']}/{f['binary']}` {f['test']} ({f['status']}): {f['message'][:300]}  \n  log: `{f['log']}`")
    for e in m["expectation_failures"]:
        L.append(f"- **expectation**: {e}")
    for s in notes:
        L.append(f"- **unexpected pass** {s['category']}/{s['name']}: {s.get('note', '')} (`{s.get('stdout')}`)")
    L += ["", "## 2. Expected vs actual", "", "| # | check | expected | actual | verdict | where |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(m["checklist"], 1):
        L.append(f"| {i} | {c['check']} | {c['expected']} | {c['actual']} | **{c['verdict']}** | `{c['where']}` |")
    na = [s for s in b.steps if s.get("verdict") in ("NOT-AVAILABLE", "SKIPPED")]
    exp = [s for s in b.steps if s.get("verdict") == "EXPECTED-FAIL"]
    L += ["", "## 3. Not available, skipped, expected failures", ""]
    for s in na:
        L.append(f"- {s['category']}/{s['name']}: {s['verdict'].lower()}: {s.get('note', '')}")
    for s in exp:
        why = "no GPU on this host (D-001)" if s.get("expect") == "pass-on-gpu-host" else s.get("note", "")
        L.append(f"- {s['category']}/{s['name']}: expected failure (exit {s.get('exit_code')}): {why}")
    if not na and not exp:
        L.append("None.")
    L += ["", "## 4. All steps", "", "| step | status | verdict | exit | s | peak RSS MiB | stdout |", "|---|---|---|---|---|---|---|"]
    for s in b.steps:
        rss = f"{s['max_rss_kb'] / 1024:.0f}" if s.get("max_rss_kb") else ""
        L.append(f"| {s['category']}/{s['name']} | {s['status']} | {s.get('verdict')} | {s.get('exit_code')} | "
                 f"{s.get('duration_s')} | {rss} | `{s.get('stdout', s.get('dir', ''))}` |")
    L += ["", "## 5. Layout", "",
          "- `manifest.json`: every step (argv, cwd, env overrides, exit, signal, duration, peak RSS, thermal before/after), the checklist, the kit's BUILDINFO.",
          "- `system/`, `gpu/`, `halo/`, `tests/`, `model/`, `baselines/`: one `NN-step/` directory per step with `command.txt`, `stdout.txt`, `stderr.txt`, `time-v.txt`, `result.json`.",
          "- `tests/<run>/summary.md|json`, `tests/<run>/xml/*.xml` (gtest XML), `tests/<run>/logs/*.log`.",
          "- `model/*.log.jsonl`: HALO's own log at trace level, JSON lines (HALO_LOG_LEVEL / HALO_LOG_FORMAT / HALO_LOG_FILE).",
          "- `snapshots/`: thermal and power state before and after the whole collection.", ""]
    (b.root / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def scrub(root: Path, replacements: list[tuple[str, str]]) -> int:
    """Replaces every occurrence of each needle in every text file of the bundle."""
    changed = 0
    if not replacements:
        return 0
    for p in root.rglob("*"):
        if not p.is_file() or p.stat().st_size > 64 << 20:
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue
        new = data
        for needle, rep in replacements:
            if needle:
                new = new.replace(needle.encode(), rep.encode())
        if new != data:
            p.write_bytes(new)
            changed += 1
    return changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="collect.sh", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=".", help="directory for the bundle (default: current directory)")
    ap.add_argument("--kit", help="unpacked kit (default: the kit this script is in)")
    ap.add_argument("--ref", help="reference data directory for the tests")
    ap.add_argument("--model", help="GGUF to inspect/run/bench (e.g. Qwen3.8-27B-UD-Q4_K_XL.gguf on ext4)")
    ap.add_argument("--mtp", help="separate MTP GGUF (ggml-org pack)")
    ap.add_argument("--llama-dir", help="llama.cpp build (dir with llama-bench, or its build/bin)")
    ap.add_argument("--llama-args", default="", help="extra llama-bench arguments")
    ap.add_argument("--host-label", default=None, help="bench/tune host label (default evo-x2 on a gfx1151 host, else dev-host)")
    ap.add_argument("--power-mode", default="unknown", help="EVO-X2 BIOS/EC performance mode (not visible in sysfs)")
    ap.add_argument("--prompt", default="Explain speculative decoding in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--bench-prompt-tokens", type=int, default=128)
    ap.add_argument("--bench-decode-tokens", type=int, default=32)
    ap.add_argument("--bench-args", default="", help="extra `halo bench model` arguments")
    ap.add_argument("--bench-system", action="store_true", help="also run `halo bench system` (4 agents; slow on the CPU engine)")
    ap.add_argument("--expect-gpu", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--quick", action="store_true", help="no full bandwidth benchmark, no per-ICD Vulkan passes")
    ap.add_argument("--no-tests", action="store_true", help="skip the test binaries")
    ap.add_argument("--skip", default="", help="comma list of step names to skip")
    ap.add_argument("--timeout-scale", type=float, default=1.0, help="multiply every step timeout")
    ap.add_argument("--no-step-snapshots", action="store_true", help="thermal snapshots only before/after the whole run")
    ap.add_argument("--anonymize", action="store_true", help="replace the hostname, user name and home directory everywhere")
    ap.add_argument("--no-redact", dest="redact", action="store_false",
                    help="DEBUG ONLY: keep secret-named environment values (never use for a bundle you share)")
    ap.add_argument("--no-tar", action="store_true", help="leave the bundle as a directory only")
    a = ap.parse_args(argv)
    a.skip = {s for s in a.skip.split(",") if s}
    if a.model and not Path(a.model).is_file():
        raise C.KitError(f"--model {a.model}: not a file")
    os.umask(0o077)

    kit: Path | None = Path(a.kit).resolve() if a.kit else C.kit_root_from(__file__)
    try:
        kitinfo = C.load_kit(kit)
    except C.KitError as e:
        print(f"collect: WARNING {e}; HALO and test steps are skipped", file=sys.stderr)
        kitinfo, kit = None, None
    gpu_host = a.expect_gpu == "yes" or (a.expect_gpu == "auto" and C.is_gfx1151_host())
    if a.host_label is None:
        a.host_label = "evo-x2" if gpu_host else "dev-host"
    if a.power_mode == "unknown":
        print("collect: WARNING --power-mode not given: bench artifacts record 'unknown' and `halo tune --list` "
              "refuses it; pass the BIOS/EC mode you set (e.g. --power-mode performance)", file=sys.stderr)
    host = socket.gethostname()
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())
    home = os.path.expanduser("~")
    host_tag = "anon" if a.anonymize else re.sub(r"[^A-Za-z0-9_.-]", "_", host)
    name = f"halo-diag-{host_tag}-{C.utc_stamp()}"
    root = Path(a.out).resolve() / name
    root.mkdir(parents=True)

    b = Bundle(root, a, gpu_host, kit)
    b.kitinfo = kitinfo
    t0 = time.monotonic()
    C.write_json(root / "snapshots" / "thermal-before.json", thermal_snapshot())
    print(f"collect: bundle {root} (gpu host: {gpu_host})", flush=True)

    collector_errors: list[str] = []

    def guarded(label: str, fn: Callable[[], None]) -> None:
        # A bug in the collector itself must cost one group of steps, never the bundle.
        try:
            fn()
        except Exception:  # noqa: BLE001 - recorded in SUMMARY.md section 1 and manifest.json
            tb = traceback.format_exc()
            collector_errors.append(f"{label}: {tb.strip().splitlines()[-1]}")
            (root / "collector-errors.txt").open("a", encoding="utf-8").write(f"== {label}\n{tb}\n")
            print(f"collect: INTERNAL ERROR in {label} (see collector-errors.txt); continuing", file=sys.stderr, flush=True)

    guarded("system steps", lambda: system_steps(b))
    guarded("gpu steps (before)", lambda: gpu_steps(b, "before"))
    halo = str(kit / "bin" / "halo") if kit else None
    if kit and halo:
        guarded("halo steps", lambda: halo_steps(b, halo, kit))
        if not a.no_tests:
            guarded("test steps", lambda: test_steps(b, kit))
        if a.model:
            guarded("model steps", lambda: model_steps(b, halo))
    if a.model and a.llama_dir:
        guarded("baseline steps", lambda: baseline_steps(b))
    guarded("gpu steps (after)", lambda: gpu_steps(b, "after"))
    if "kernel-log" not in a.skip:
        # A second kernel-log capture shows what the run itself added (GPU faults, resets).
        guarded("kernel log (after)", lambda: kernel_log_step(b))
    guarded("thermal snapshot (after)",
            lambda: C.write_json(root / "snapshots" / "thermal-after.json", thermal_snapshot()))

    test_failures: list[dict[str, Any]] = []
    exp_failures: list[str] = []
    items: list[dict[str, str]] = []

    def collect_test_results() -> None:
        for run in sorted((root / "tests").glob("*/summary.json")) if (root / "tests").is_dir() else []:
            s = _read_json(run) or {}
            for bb in s.get("binaries", []):
                for t in bb.get("tests", []):
                    if t["status"] in ("failed", "crashed", "timeout"):
                        test_failures.append({"run": run.parent.name, "binary": bb["name"], "test": t["name"],
                                              "status": t["status"], "message": t["message"],
                                              "log": os.path.relpath(t.get("log", bb["log"]), root)})
            exp_failures.extend(f"{run.parent.name}: {e}" for e in s.get("expectation_failures", []))

    guarded("test result parsing", collect_test_results)
    guarded("checklist", lambda: items.extend(checklist(b)))
    bad_steps = [s for s in b.steps if s.get("verdict") == "FAIL"]
    verdict = "OK" if not bad_steps and not test_failures and not exp_failures and not collector_errors and \
        not any(c["verdict"] == "MISMATCH" for c in items) else "NEEDS ATTENTION"
    manifest = {
        "schema": SCHEMA, "created_utc": C.utc_now(), "host": "<host>" if a.anonymize else host,
        "user": "<user>" if a.anonymize else user, "gpu_host": gpu_host, "gfx_targets": C.kfd_gfx_targets(),
        "duration_s": round(time.monotonic() - t0, 1), "verdict": verdict,
        "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(a).items()},
        "kit": kitinfo and {k: v for k, v in kitinfo.items() if k != "tests"} or {},
        "facts": b.facts, "checklist": items, "test_failures": test_failures, "expectation_failures": exp_failures,
        "collector_errors": collector_errors,
        "steps": b.steps, "python": sys.version.split()[0], "tool": "scripts/evox2/collect.sh",
    }
    guarded("manifest.json", lambda: C.write_json(root / "manifest.json", manifest))
    try:
        write_summary(b, manifest)
    except Exception:  # noqa: BLE001 - fall back to a minimal SUMMARY.md, never lose the bundle
        tb = traceback.format_exc()
        collector_errors.append(f"SUMMARY.md: {tb.strip().splitlines()[-1]}")
        (root / "SUMMARY.md").write_text(
            f"# HALO diagnostic bundle (MINIMAL SUMMARY: the summary writer failed)\n\nverdict: NEEDS ATTENTION\n\n"
            "## 1. Failures\n\n- **collector error**: see the traceback below; every step's record is in manifest.json "
            "and the NN-step/result.json files.\n" + "".join(f"- {e}\n" for e in collector_errors)
            + f"\n```\n{tb}```\n", encoding="utf-8")
        verdict = "NEEDS ATTENTION"

    reps: list[tuple[str, str]] = []
    if a.redact:
        reps += [(v, "<redacted>") for v in C.secret_values()]
    if a.anonymize:
        if home and home != "/":
            reps.append((home, "/home/<user>"))
        fq = socket.getfqdn()
        for h in sorted({fq, host}, key=len, reverse=True):
            if h and len(h) >= 3 and h not in ("localhost",):
                reps.append((h, "<host>"))
        if len(user) >= 3:
            reps.append((user, "<user>"))
    try:
        n = scrub(root, reps)
    except Exception as e:  # noqa: BLE001
        # Never package what may still hold a secret or a hostname: leave the directory only.
        print(f"collect: ERROR scrubbing the bundle ({type(e).__name__}: {e}); NOT writing the .tar.gz. "
              f"Inspect {root} before sharing it.", file=sys.stderr, flush=True)
        return 1
    print(f"collect: scrubbed {n} file(s) ({'redacted' if a.redact else 'NOT redacted'}"
          f"{', anonymized' if a.anonymize else ''})", flush=True)
    if not a.no_tar:
        tgz = root.with_name(name + ".tar.gz")
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(root, arcname=name)
        print(f"collect: wrote {tgz}")
    print(f"collect: {verdict}: {len(bad_steps)} failed step(s), {len(test_failures)} failed test(s), "
          f"{len(exp_failures)} expectation failure(s), {len(collector_errors)} collector error(s); "
          f"read {root / 'SUMMARY.md'}")
    return 0 if verdict == "OK" else 1
