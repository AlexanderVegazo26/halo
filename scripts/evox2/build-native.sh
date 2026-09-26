#!/usr/bin/env bash
# Builds HALO from source ON the EVO-X2, against the target's own ROCm, with -march=native and
# gfx1151, and assembles the result as a kit directory that run-tests.sh / collect.sh use exactly
# like the prebuilt one. The fallback when the prebuilt kit's ROCm 7.1 build does not load on
# ROCm 7.15 (docs/evox2.md, "Troubleshooting").
#
#   build-native.sh [--src DIR] [--out DIR] [--no-werror] [--in-tree-tests] [--jobs N] [--cc C --cxx CXX]
#     --src DIR        a HALO source tree (default: unpack the kit's source/halo-src-*.tar.gz)
#     --out DIR        where the native kit goes (default: ./native)
#     --no-werror      warnings are not errors (a newer clang than the dev host's clang 18 may warn more)
#     --in-tree-tests  also run the tests on the build tree before assembling
#     --cc/--cxx       compilers (default clang / clang++; e.g. --cc clang-20 --cxx clang++-20)
#
# Needs network access: CMake fetches nlohmann/json, minja, cpp-httplib and googletest (pinned by
# SHA-256, cmake/HaloDeps.cmake). Apt packages: see docs/evox2.md, "Prerequisites".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT="$(cd "$HERE/../.." && pwd)"
SRC=""
OUT="$PWD/native"
PKG_ARGS=()
CC_=clang
CXX_=clang++
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-werror) PKG_ARGS+=(--no-werror); shift ;;
    --in-tree-tests) PKG_ARGS+=(--in-tree-tests); shift ;;
    --jobs) PKG_ARGS+=(--jobs "$2"); shift 2 ;;
    --cc) CC_="$2"; shift 2 ;;
    --cxx) CXX_="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "build-native.sh: unknown option $1" >&2; exit 2 ;;
  esac
done

fail=0
need() {  # need <command> <apt hint>
  if ! command -v "$1" >/dev/null 2>&1; then echo "MISSING $1  ($2)"; fail=1; else echo "ok      $1: $(command -v "$1")"; fi
}
echo "== build-native.sh: prerequisites"
need cmake "sudo apt install cmake (>= 3.25)"
need ninja "sudo apt install ninja-build"
need "$CC_" "sudo apt install clang"
need "$CXX_" "sudo apt install clang"
need glslc "sudo apt install glslc"
need python3 "sudo apt install python3"
need tar "coreutils"
need readelf "sudo apt install binutils"
need pkg-config "sudo apt install pkg-config"
command -v spirv-val >/dev/null || echo "note    spirv-val missing: shaders are not validated at build time (sudo apt install spirv-tools)"
if command -v cmake >/dev/null; then
  cv="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  if [[ "$(printf '%s\n3.25\n' "$cv" | sort -V | head -1)" != "3.25" ]]; then echo "MISSING cmake >= 3.25 (found $cv)"; fail=1; fi
fi
[[ -f /usr/include/vulkan/vulkan.h ]] || { echo "MISSING Vulkan headers (sudo apt install libvulkan-dev)"; fail=1; }
[[ -f /usr/include/sqlite3.h ]] || { echo "MISSING SQLite headers (sudo apt install libsqlite3-dev)"; fail=1; }
ROCM="${ROCM_PATH:-/opt/rocm}"
[[ -x "$ROCM/llvm/bin/clang++" ]] || { echo "MISSING $ROCM/llvm/bin/clang++ (the ROCm HIP compiler; install the HIP SDK of your ROCm, or set ROCM_PATH)"; fail=1; }
HIPCFG="$(ls "$ROCM"/lib/cmake/hip/hip-config.cmake 2>/dev/null || true)"
[[ -n "$HIPCFG" ]] || { echo "MISSING $ROCM/lib/cmake/hip/hip-config.cmake (HIP CMake package; install hip-dev / the HIP SDK)"; fail=1; }
echo "ROCm:   $(cat "$ROCM/.info/version" 2>/dev/null || echo unknown) at $ROCM"
echo "clang:  $("$CXX_" --version 2>/dev/null | head -1 || echo missing)"
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  echo "WARNING HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set; unset it before running the tests"
fi
if [[ $fail -ne 0 ]]; then echo "build-native.sh: install the missing prerequisites above and re-run" >&2; exit 1; fi

WORK="$(mkdir -p "$OUT" && cd "$OUT" && pwd)/work"
mkdir -p "$WORK"
if [[ -z "$SRC" ]]; then
  TGZ="$(ls "$KIT"/source/halo-src-*.tar.gz 2>/dev/null | head -1 || true)"
  [[ -n "$TGZ" ]] || { echo "build-native.sh: no --src and no source tarball in $KIT/source" >&2; exit 1; }
  rm -rf "$WORK/src" && mkdir -p "$WORK/src"
  tar -xzf "$TGZ" -C "$WORK/src" --strip-components=1
  SRC="$WORK/src"
fi
[[ -f "$SRC/CMakeLists.txt" && -f "$SRC/scripts/package.sh" ]] || { echo "build-native.sh: $SRC is not a HALO source tree" >&2; exit 1; }

echo "== build-native.sh: building $SRC (this takes a while)"
exec bash "$SRC/scripts/package.sh" --source copy --march native --name-prefix halo-native \
  --no-ref --no-tarball --out "$OUT" --cc "$CC_" --cxx "$CXX_" "${PKG_ARGS[@]}"
