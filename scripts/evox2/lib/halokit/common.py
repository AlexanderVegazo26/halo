"""Shared helpers for the EVO-X2 field kit (docs/evox2.md).

Python standard library only: the kit runs on a stock Ubuntu with python3 and nothing else.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

GNU_TIME = "/usr/bin/time"

# Environment variable names whose values are never written anywhere (collect.sh, run-tests).
# PAT only as a whole word (GITHUB_PAT), so PATH is not caught. SSH_CONNECTION / SSH_CLIENT carry
# the client and server IP addresses (review S-44).
SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSW|AUTH|CREDENTIAL|COOKIE|DSN|PRIVATE|(?:^|_)PAT(?:_|$)|"
                            r"^SSH_(?:CONNECTION|CLIENT)$", re.IGNORECASE)
# Credentials inside a URL-valued variable of any name: http_proxy=http://user:pass@host:3128,
# DATABASE_URL=postgres://u:p@db/x (review S-44). Group 1 keeps the scheme.
URL_CRED_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+:[^@\s/]+@")
URL_CRED_BYTES_RE = re.compile(URL_CRED_RE.pattern.encode())
# Dropped from the environment dump under --anonymize (they identify the session or the network).
ANON_DROP_RE = re.compile(r"^(?:SSH_\w*|DISPLAY|XDG_SESSION_\w*|MAIL|SUDO_\w*)$")
# GPU runtime variables whose values are recorded (they change what the GPU stack does).
GPU_ENV_PREFIXES = ("HSA_", "HIP_", "ROCR_", "ROCM_", "HCC_", "AMD_", "GPU_", "RADV_", "VK_", "MESA_",
                    "ACO_", "LLVM_", "HALO_", "GGML_", "OMP_", "LD_LIBRARY_PATH")


class KitError(RuntimeError):
    """A condition the kit cannot work around; the message says what to do."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def is_secret_name(name: str) -> bool:
    return bool(SECRET_NAME_RE.search(name))


def redact_url_credentials(text: str) -> str:
    return URL_CRED_RE.sub(r"\1<redacted>@", text)


def redacted_env(env: dict[str, str] | None = None, anonymize: bool = False) -> dict[str, str]:
    """A copy of env: secret-looking names' values -> <redacted>, URL credentials in any value
    -> scheme://<redacted>@, and with anonymize the session/network variables dropped."""
    src = os.environ if env is None else env
    out = {}
    for k, v in sorted(src.items()):
        if anonymize and ANON_DROP_RE.match(k):
            continue
        out[k] = "<redacted>" if is_secret_name(k) else redact_url_credentials(v)
    return out


def secret_values(env: dict[str, str] | None = None) -> list[str]:
    """Values of secret-named variables, longest first, for scrubbing captured output."""
    src = os.environ if env is None else env
    vals = {v for k, v in src.items() if is_secret_name(k) and len(v) >= 4}
    return sorted(vals, key=len, reverse=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_text(path: Path, limit: int = 1 << 20) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_sysfs(path: str | Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return f.read(1 << 16).decode("utf-8", errors="replace").strip()
    except OSError:
        return None


def which(tool: str) -> str | None:
    if os.sep in tool:
        return tool if os.access(tool, os.X_OK) else None
    return shutil.which(tool)


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def parse_time_v(path: Path) -> dict[str, Any]:
    """Fields of GNU time -v output that the summaries use."""
    out: dict[str, Any] = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size (kbytes):"):
            out["max_rss_kb"] = int(line.rsplit(":", 1)[1])
        elif line.startswith("Elapsed (wall clock) time"):
            out["wall"] = line.split("):", 1)[-1].strip()
        elif line.startswith("User time (seconds):"):
            out["user_s"] = float(line.rsplit(":", 1)[1])
        elif line.startswith("System time (seconds):"):
            out["sys_s"] = float(line.rsplit(":", 1)[1])
        elif line.startswith("Exit status:"):
            out["exit_status"] = int(line.rsplit(":", 1)[1])
        elif line.startswith("Command terminated by signal"):
            out["signal"] = int(line.rsplit(" ", 1)[1])
    return out


def run_cmd(argv: list[str], *, stdout_path: Path, stderr_path: Path | None = None, timeout: float,
            cwd: str | Path | None = None, env: dict[str, str] | None = None,
            time_v_path: Path | None = None, stdin_data: bytes | None = None) -> dict[str, Any]:
    """Runs argv with a hard timeout (the whole process group is killed) and records the result.

    Never raises for a failing or missing program: the result dict says what happened.
    stderr_path None sends stderr to the stdout file.
    """
    rec: dict[str, Any] = {
        "argv": list(argv),
        "command": shell_join(argv),
        "cwd": str(cwd) if cwd else os.getcwd(),
        "timeout_s": timeout,
        "started_utc": utc_now(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path if stderr_path else stdout_path),
    }
    exe = which(argv[0])
    if exe is None:
        rec.update(status="not-available", exit_code=None, duration_s=0.0,
                   note=f"{argv[0]}: not installed or not on PATH")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(rec["note"] + "\n", encoding="utf-8")
        return rec
    full = list(argv)
    use_time = time_v_path is not None and os.access(GNU_TIME, os.X_OK)
    if use_time:
        full = [GNU_TIME, "-v", "-o", str(time_v_path), "--"] + full
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    so = open(stdout_path, "wb")
    se = open(stderr_path, "wb") if stderr_path else None
    t0 = time.monotonic()
    timed_out = False
    try:
        try:
            p = subprocess.Popen(full, cwd=cwd, env=env, stdout=so, stderr=se if se else subprocess.STDOUT,
                                 stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                                 start_new_session=True)
        except OSError as e:
            rec.update(status="error", exit_code=None, duration_s=0.0, note=f"could not start: {e}")
            return rec
        try:
            if stdin_data is not None and p.stdin:
                try:
                    p.stdin.write(stdin_data)
                    p.stdin.close()
                except OSError:
                    pass
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(p)
            rc = p.returncode
    finally:
        so.close()
        if se:
            se.close()
    rec["duration_s"] = round(time.monotonic() - t0, 3)
    sig = None
    if rc is not None and rc < 0:
        sig = -rc
    if use_time and time_v_path is not None:
        tv = parse_time_v(time_v_path)
        rec["time_v"] = str(time_v_path)
        rec.update({k: v for k, v in tv.items() if k in ("max_rss_kb", "user_s", "sys_s")})
        if "signal" in tv:
            sig = tv["signal"]
        if "exit_status" in tv and not timed_out and sig is None:
            rc = tv["exit_status"]
    rec["exit_code"] = rc
    rec["signal"] = sig
    rec["timed_out"] = timed_out
    if timed_out:
        rec["status"] = "timeout"
    elif sig is not None:
        rec["status"] = "crashed"
        try:
            rec["signal_name"] = signal.Signals(sig).name
        except ValueError:
            pass
    else:
        rec["status"] = "pass" if rc == 0 else "fail"
    return rec


def _kill_group(p: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# ---------------------------------------------------------------------------------------------
# Host facts

def kfd_gfx_targets() -> list[str]:
    """gfx names of the GPU nodes in the KFD topology (empty when there is no amdgpu/KFD)."""
    names = []
    root = Path("/sys/class/kfd/kfd/topology/nodes")
    if not root.is_dir():
        return names
    for node in sorted(root.iterdir()):
        props = read_sysfs(node / "properties") or ""
        m = re.search(r"^gfx_target_version\s+(\d+)", props, re.MULTILINE)
        if not m or m.group(1) == "0":
            continue
        v = int(m.group(1))
        major, minor, step = v // 10000, (v // 100) % 100, v % 100
        names.append(f"gfx{major}{minor:x}{step:x}")
    return names


def is_gfx1151_host() -> bool:
    return "gfx1151" in kfd_gfx_targets()


def amdgpu_cards() -> list[Path]:
    """/sys/class/drm/cardN/device directories driven by amdgpu."""
    out = []
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return out
    for card in sorted(drm.glob("card[0-9]*")):
        if "-" in card.name:
            continue
        dev = card / "device"
        drv = dev / "driver"
        try:
            if drv.is_symlink() and os.path.basename(os.readlink(drv)) == "amdgpu":
                out.append(dev)
        except OSError:
            continue
    return out


def vulkan_icds() -> dict[str, list[str]]:
    """ICD manifest files by driver family (radv, amdvlk, lavapipe, other)."""
    dirs = ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d",
            "/opt/amdgpu/etc/vulkan/icd.d"]
    out: dict[str, list[str]] = {"radv": [], "amdvlk": [], "lavapipe": [], "other": []}
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.json")):
            n = f.name
            if n.startswith("radeon_icd"):
                out["radv"].append(str(f))
            elif n.startswith("amd_icd") or "amdvlk" in n:
                out["amdvlk"].append(str(f))
            elif n.startswith("lvp_icd"):
                out["lavapipe"].append(str(f))
            else:
                out["other"].append(str(f))
    return out


def icd_env(family: str) -> tuple[dict[str, str] | None, str]:
    """Environment that makes the Vulkan loader see only one driver family."""
    if family in ("", "default"):
        return {}, "default ICD selection (the loader sees every installed driver)"
    files = vulkan_icds().get(family, [])
    if not files:
        return None, f"no {family} ICD manifest installed"
    joined = ":".join(files)
    return {"VK_DRIVER_FILES": joined, "VK_ICD_FILENAMES": joined}, f"{family} only: {joined}"


def rocm_lib_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """LD_LIBRARY_PATH with the target's ROCm library directory first, or {} when there is none.

    The kit sets no RPATH to ROCm (the build host's /opt/rocm-7.1.x does not exist on the
    target). The test_hip_* binaries need libamdhip64 from the TARGET's ROCm, so the kit's
    runners put ${ROCM_PATH:-/opt/rocm}/lib first in LD_LIBRARY_PATH, which the loader searches
    before any RUNPATH. The value is recorded with every run (env_overrides).
    """
    src = os.environ if env is None else env
    lib = Path(src.get("ROCM_PATH") or "/opt/rocm") / "lib"
    if not any(lib.glob("libamdhip64.so*")):
        return {}
    cur = [p for p in src.get("LD_LIBRARY_PATH", "").split(":") if p]
    if cur and cur[0] == str(lib):
        return {}
    return {"LD_LIBRARY_PATH": ":".join([str(lib)] + [p for p in cur if p != str(lib)])}


# ---------------------------------------------------------------------------------------------
# Kit layout

def kit_root_from(script_file: str) -> Path:
    """<kit>/scripts/evox2/lib/halokit/common.py -> <kit>."""
    return Path(script_file).resolve().parents[4]


def load_kit(kit: Path) -> dict[str, Any]:
    info = kit / "kit.json"
    if not info.is_file():
        raise KitError(f"{kit} is not an unpacked HALO kit (no kit.json). Pass --kit DIR, or run the "
                       "script from inside the unpacked halo-evox2-* directory.")
    return json.loads(info.read_text(encoding="utf-8"))


def setup_anchor(kit: Path, anchor: str, ref: Path | None) -> list[str]:
    """Creates the fixed path the test binaries were built against (docs/evox2.md, "layout").

    <anchor> is a directory owned by the current user holding three symlinks: src and build
    point into the kit, ref at the reference data (absent -> ref-dependent tests skip).
    """
    notes: list[str] = []
    a = Path(anchor)
    if a.is_symlink():
        raise KitError(f"{a} is a symlink; the kit refuses to use it. Remove it (rm {a}) and re-run.")
    import stat as _stat
    if not a.exists():
        # No parents, no exist_ok: fails if another user creates it first (review S-40).
        a.mkdir(mode=0o700, parents=False)
        os.chmod(a, 0o700, follow_symlinks=False)  # mkdir's mode is filtered by the umask
    # Verify whatever is there now: a directory, not a symlink, owned by the effective user, mode
    # 0700. A group- or world-writable anchor would let another user swap the build/ symlink that
    # run-tests executes through.
    st = a.lstat()
    if not _stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
        raise KitError(f"{a} exists and is not a directory owned by uid {os.geteuid()}. Remove it "
                       "(it may be left over from another user or from package.sh) and re-run.")
    if _stat.S_IMODE(st.st_mode) != 0o700:
        raise KitError(f"{a} has mode {_stat.S_IMODE(st.st_mode):o}, not 700; refusing to run binaries through it. "
                       f"Remove it (rm -r {a}) and re-run.")
    if (a / ".halo-kit-staging").exists():
        raise KitError(f"{a} holds a package build (scripts/package.sh --keep-build). Remove it or "
                       "run the tests from that build directory instead.")
    for name, target in (("src", kit / "src"), ("build", kit / "build"), ("ref", ref)):
        link = a / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise KitError(f"{link} exists and is not a symlink; remove {a} and re-run.")
        if target is not None:
            if not target.exists():
                raise KitError(f"{target} does not exist")
            link.symlink_to(target.resolve())
            notes.append(f"{link} -> {target.resolve()}")
        else:
            notes.append(f"{link}: no reference data (tests that need it will SKIP)")
    return notes


def find_ref(kit: Path, explicit: str | None) -> Path | None:
    cands = []
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise KitError(f"--ref {explicit}: not a directory")
        return _ref_root(p)
    env = os.environ.get("HALO_KIT_REF")
    if env:
        cands.append(Path(env))
    cands += [kit / "ref", kit / "halo-ref", kit.parent / "halo-ref"]
    for c in cands:
        if c.is_dir():
            return _ref_root(c)
    return None


def _ref_root(p: Path) -> Path:
    # Accept either the data directory itself or a directory holding the unpacked halo-ref/.
    if not (p / "tiny").exists() and (p / "halo-ref").is_dir():
        return p / "halo-ref"
    return p


# ---------------------------------------------------------------------------------------------
# gtest

def parse_gtest_xml(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    tests = []
    for ts in root.iter("testsuite"):
        for tc in ts.findall("testcase"):
            name = f"{tc.get('classname', ts.get('name', '?'))}.{tc.get('name', '?')}"
            fails = tc.findall("failure")
            skip = tc.find("skipped")
            if fails:
                status = "failed"
                msg = "\n".join((f.get("message") or f.text or "") for f in fails)
            elif skip is not None or tc.get("result") == "skipped":
                status = "skipped"
                msg = (skip.get("message") if skip is not None else "") or (skip.text if skip is not None else "") or ""
            elif tc.get("status") == "notrun":
                status = "disabled"
                msg = ""
            else:
                status = "passed"
                msg = ""
            tests.append({"name": name, "status": status, "message": msg.strip()[:4000],
                          "time_s": float(tc.get("time", "0") or 0)})
    return tests


def list_gtests(exe: str, cwd: str, env: dict[str, str], log: Path) -> list[str]:
    rec = run_cmd([exe, "--gtest_list_tests"], stdout_path=log, timeout=120, cwd=cwd, env=env)
    if rec["status"] != "pass":
        return []
    names = []
    suite = ""
    for line in read_text(log).splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            suite = line.split("#", 1)[0].strip()
        else:
            names.append(suite + line.split("#", 1)[0].strip())
    return names


SKIP_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("device-present", re.compile(r"a HIP device is present", re.I)),
    ("no-device", re.compile(r"no HIP device|D-001|Vulkan unavailable|zero physical devices|"
                             r"no Vulkan device|hipErrorNoDevice|no ROCm", re.I)),
    ("opt-in", re.compile(r"HALO_BANDWIDTH_FULL|set HALO_[A-Z_]+=1", re.I)),
    ("tool-missing", re.compile(r"llama|ollama|HALO_LLAMA_BIN_DIR", re.I)),
    ("ref-missing", re.compile(r"not found|missing|HALO_REF|halo-ref|reference|golden|fetch_", re.I)),
]


def skip_category(message: str) -> str:
    for cat, rx in SKIP_CATEGORIES:
        if rx.search(message):
            return cat
    return "other"
