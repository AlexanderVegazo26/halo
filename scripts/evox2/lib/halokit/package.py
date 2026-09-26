"""Package-time helpers used by scripts/package.sh (they run on the build host).

`assemble` copies the build-tree binaries and the fixtures they read into the kit directory
and writes kit.json. `check` verifies that every path a shipped binary embeds resolves
inside the kit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import common as C

# Source-tree data the tests read through HALO_SOURCE_DIR / HALO_API_FIXTURES (grep of tests/:
# tests/fixtures/sysfs, tests/unit/profiling/fixtures, tests/unit/api/fixtures, bench/workloads).
# Every tests/**/fixtures directory is taken, so a new fixture directory is picked up without an edit here.
FIXTURE_GLOBS = ["tests/fixtures", "tests/unit/*/fixtures", "bench/workloads"]
SOURCE_SUFFIXES = (".cpp", ".cc", ".h", ".hpp", ".hip", ".inc", ".ipp")
FOREIGN_RE =re.compile(rb"(?:/root/|/mnt/[a-z]/|/home/)[A-Za-z0-9_.+/-]{2,}")


def ctest_tests(build: Path) -> list[dict[str, Any]]:
    r = subprocess.run(["ctest", "--test-dir", str(build), "--show-only=json-v1"], capture_output=True, check=True)
    return json.loads(r.stdout)["tests"]


def embedded_paths(exe: Path, anchor: str) -> tuple[list[str], list[str]]:
    data = exe.read_bytes()
    rx = re.compile(re.escape(anchor.encode()) + rb"(/[A-Za-z0-9_.+/-]*)?")
    mine = sorted({(m.group(1) or b"").decode().lstrip("/") for m in rx.finditer(data)})
    foreign = sorted({m.group(0).decode(errors="replace") for m in FOREIGN_RE.finditer(data)})
    return mine, foreign


def assemble(build: Path, src: Path, anchor: str, kit: Path) -> dict[str, Any]:
    anchor_build = f"{anchor}/build/"
    tests = ctest_tests(build)
    bins: dict[str, dict[str, Any]] = {}
    for t in tests:
        cmd = t.get("command") or []
        if not cmd:
            continue
        exe = cmd[0]
        props = {p["name"]: p.get("value") for p in t.get("properties", [])}
        wd = props.get("WORKING_DIRECTORY") or str(Path(exe).parent)
        if not exe.startswith(anchor_build) or not str(wd).startswith(anchor_build):
            raise C.KitError(f"test {t.get('name')}: {exe} (cwd {wd}) is not under {anchor_build}; "
                             "package.sh must build inside the anchor")
        b = bins.setdefault(exe, {"name": Path(exe).name, "path": os.path.relpath(exe, anchor),
                                  "working_dir": os.path.relpath(wd, anchor), "discovered_tests": 0,
                                  "environment": props.get("ENVIRONMENT")})
        b["discovered_tests"] += 1

    extra: set[str] = set()
    foreign_all: dict[str, list[str]] = {}
    for exe, b in bins.items():
        mine, foreign = embedded_paths(Path(exe), anchor)
        b["embedded_paths"] = mine
        b["needs_ref"] = any(p == "ref" or p.startswith("ref/") for p in mine)
        if foreign:
            foreign_all[b["name"]] = foreign
        for p in mine:
            if p.startswith("build/") and p != b["path"] and (Path(anchor) / p).is_file():
                extra.add(p)
    halo_exe = build / "tools" / "halo" / "halo"
    if not halo_exe.is_file():
        raise C.KitError(f"{halo_exe} was not built")
    _, foreign = embedded_paths(halo_exe, anchor)
    if foreign:
        foreign_all["halo"] = foreign

    # Copy: bin/halo, the test binaries and helpers at their build-tree paths, the fixtures.
    (kit / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(halo_exe, kit / "bin" / "halo")
    for rel in sorted({b["path"] for b in bins.values()} | extra):
        dst = kit / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(anchor) / rel, dst)
    for b in bins.values():
        (kit / b["working_dir"]).mkdir(parents=True, exist_ok=True)
    fixtures = []
    for g in FIXTURE_GLOBS:
        for d in sorted(src.glob(g)):
            if d.is_dir():
                rel = d.relative_to(src)
                shutil.copytree(d, kit / "src" / rel, dirs_exist_ok=True)
                fixtures.append(f"src/{rel}")
    return {"tests": {"binaries": sorted(bins.values(), key=lambda x: x["name"]), "helpers": sorted(extra)},
            "fixtures": fixtures, "foreign_paths": foreign_all}


def check(kit: Path, anchor: str) -> list[str]:
    """Problems: embedded anchor paths (src/, build/) that do not exist in the kit."""
    info = C.load_kit(kit)
    problems = []
    for b in info["tests"]["binaries"]:
        for p in b["embedded_paths"]:
            if p in ("", "ref") or p.startswith("ref/") or p.endswith(SOURCE_SUFFIXES):
                continue  # __FILE__ of a test source (gtest failure messages) is never opened
            if not (kit / p).exists():
                problems.append(f"{b['name']} embeds {anchor}/{p}, which is not in the kit")
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="halo_kit.py package")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("assemble")
    s1.add_argument("--build", required=True)
    s1.add_argument("--src", required=True)
    s1.add_argument("--anchor", required=True)
    s1.add_argument("--kit", required=True)
    s1.add_argument("--meta", required=True, help="JSON file with the build facts to merge into kit.json")
    s2 = sub.add_parser("check")
    s2.add_argument("--kit", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        kit = Path(a.kit)
        res = assemble(Path(a.build), Path(a.src), a.anchor, kit)
        meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
        info = {"schema": "halo.kit/1", **meta, "anchor": a.anchor, **res}
        C.write_json(kit / "kit.json", info)
        nb = len(res["tests"]["binaries"])
        nt = sum(b["discovered_tests"] for b in res["tests"]["binaries"])
        print(f"assemble: {nb} test binaries ({nt} ctest tests), helpers {res['tests']['helpers']}, "
              f"fixtures {res['fixtures']}")
        for k, v in res["foreign_paths"].items():
            print(f"assemble: WARNING {k} embeds non-kit paths: {v[:5]}")
        return 0
    probs = check(Path(a.kit), C.load_kit(Path(a.kit))["anchor"])
    for p in probs:
        print(f"check: {p}")
    print(f"check: {'OK' if not probs else f'{len(probs)} problem(s)'}")
    return 0 if not probs else 1
