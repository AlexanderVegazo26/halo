#!/usr/bin/env bash
# Builds the EVO-X2 field kit (docs/evox2.md): dist/halo-evox2-<sha12>[-dirty].tar.gz, plus the
# reference-data tarball dist/halo-ref-<hash12>.tar.gz that the tests read. Runs on a Linux build
# host: WSL on the dev machine, or the EVO-X2 itself through scripts/evox2/build-native.sh.
#
#   scripts/package.sh [options]
#     --source head|worktree|copy  what to build. head (default) = `git archive HEAD` (reproducible);
#                             worktree = every tracked and untracked, non-ignored file as it is now;
#                             copy = this directory as it is (an unpacked source tarball, no git).
#     --overlay PATH          with --source head: copy PATH (file or directory) from the working
#                             tree over HEAD. Repeatable. Marks the kit dirty if PATH differs.
#     --march TARGET          CPU target (default x86-64-v3; docs/evox2.md explains why)
#     --build-type TYPE       CMake build type (default Release)
#     --name-prefix P         kit name prefix (default halo-evox2; build-native.sh uses halo-native)
#     --out DIR               output directory (default <repo>/dist)
#     --ref DIR               reference data to package (default $HOME/halo-ref)
#     --no-ref                do not build the reference-data tarball
#     --no-tarball            leave the kit as the directory <out>/<name> instead of a .tar.gz
#     --no-werror             configure with -DHALO_WERROR=OFF
#     --cc C / --cxx CXX      host compilers (default clang / clang++)
#     --in-tree-tests         run the kit's test runner on the build tree before packaging, and
#                             save its summary next to the kit (to compare with a run of the kit)
#     --incremental           reuse an existing build tree at the anchor (development only)
#     --keep-build            leave the build tree at the anchor afterwards (the kit's tests then
#                             refuse to run on this host until it is removed)
#     --jobs N                build parallelism (default $CMAKE_BUILD_PARALLEL_LEVEL, else nproc)
#
# The test binaries embed absolute paths at compile time (HALO_SOURCE_DIR, HALO_REF_DIR, the API
# fixtures, the profiling child helper). The build therefore happens under one fixed prefix, the
# "anchor" /tmp/<prefix>-<sha12>-anchor, and the kit recreates that prefix on the target as a
# directory of symlinks (scripts/evox2/run-tests.sh). Nothing from ROCm or the system is bundled.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE=head
OVERLAYS=()
MARCH=x86-64-v3
BUILD_TYPE=Release
PREFIX=halo-evox2
OUT="$REPO/dist"
REF="${HOME}/halo-ref"
DO_REF=1
TARBALL=1
WERROR=ON
CC_=clang
CXX_=clang++
IN_TREE=0
INCREMENTAL=0
KEEP=0
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

die() { echo "package.sh: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --overlay) OVERLAYS+=("$2"); shift 2 ;;
    --march) MARCH="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --name-prefix) PREFIX="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --no-ref) DO_REF=0; shift ;;
    --no-tarball) TARBALL=0; shift ;;
    --no-werror) WERROR=OFF; shift ;;
    --cc) CC_="$2"; shift 2 ;;
    --cxx) CXX_="$2"; shift 2 ;;
    --in-tree-tests) IN_TREE=1; shift ;;
    --incremental) INCREMENTAL=1; shift ;;
    --keep-build) KEEP=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
[[ "$SOURCE" =~ ^(head|worktree|copy)$ ]] || die "--source must be head, worktree or copy"
[[ ${#OVERLAYS[@]} -eq 0 || "$SOURCE" == head ]] || die "--overlay needs --source head"
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || die "--name-prefix: letters, digits, _ and - only"
[[ "$(uname -s)" == Linux ]] || die "run this on a Linux build host (WSL), not on Windows"
TOOLS=(cmake ninja "$CC_" "$CXX_" python3 tar gzip sha256sum readelf ldd)
[[ "$SOURCE" != copy ]] && TOOLS+=(git)
for t in "${TOOLS[@]}"; do command -v "$t" >/dev/null || die "missing build tool: $t"; done

# --- identity: git sha + dirty flag (from git, or from SOURCE_INFO in a source tarball) --------
DIRTY_PATHS=""
if [[ "$SOURCE" == copy ]]; then
  # Only a checkout whose top level IS this directory counts: an unpacked source tarball that
  # happens to sit inside another checkout must not take that checkout's sha.
  if [[ "$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null)" == "$REPO" ]]; then
    SHA="$(git -C "$REPO" rev-parse HEAD)"
    DIRTY_PATHS="$(git -C "$REPO" status --porcelain --untracked-files=all)"
  elif [[ -f "$REPO/SOURCE_INFO" ]]; then
    SHA="$(sed -n 's/^git_sha=//p' "$REPO/SOURCE_INFO")"
    if [[ "$(sed -n 's/^dirty=//p' "$REPO/SOURCE_INFO")" == true ]]; then
      DIRTY_PATHS="$(sed -n 's/^dirty_paths=//p' "$REPO/SOURCE_INFO")"
      [[ -n "$DIRTY_PATHS" ]] || DIRTY_PATHS="(dirty per SOURCE_INFO)"
    fi
  else
    SHA=unknown000000
  fi
else
  SHA="$(git -C "$REPO" rev-parse HEAD)"
  case "$SOURCE" in
    head) [[ ${#OVERLAYS[@]} -gt 0 ]] && DIRTY_PATHS="$(git -C "$REPO" status --porcelain --untracked-files=all -- "${OVERLAYS[@]}")" ;;
    worktree) DIRTY_PATHS="$(git -C "$REPO" status --porcelain --untracked-files=all)" ;;
  esac
fi
SHA12="${SHA:0:12}"
DIRTY=false
[[ -n "$DIRTY_PATHS" ]] && DIRTY=true
NAME="${PREFIX}-${SHA12}"
[[ "$DIRTY" == true ]] && NAME="${NAME}-dirty"
ANCHOR="/tmp/${PREFIX}-${SHA12}-anchor"
SRC="$ANCHOR/src"
BUILD="$ANCHOR/build"

echo "package.sh: $NAME (source $SOURCE, -march=$MARCH, $BUILD_TYPE), anchor $ANCHOR"

# --- stage the source at the anchor --------------------------------------------------------
if [[ -e "$ANCHOR" || -L "$ANCHOR" ]]; then
  [[ -L "$ANCHOR" ]] && die "$ANCHOR is a symlink; remove it first"
  [[ -O "$ANCHOR" ]] || die "$ANCHOR exists and is not owned by $(id -un); remove it first"
  if [[ $INCREMENTAL -eq 1 && -f "$ANCHOR/.halo-kit-staging" && -d "$BUILD" && -d "$SRC" ]] \
      && command -v rsync >/dev/null; then
    echo "package.sh: --incremental: reusing $BUILD"
    rm -rf -- "$ANCHOR/stage" "$ANCHOR/ref"
  else
    echo "package.sh: removing the previous $ANCHOR"
    rm -rf -- "$ANCHOR"
  fi
fi
# The anchor is a predictable /tmp path (review S-40): create it with plain mkdir (no -p), which
# fails if another user created it in the meantime, and then verify what we got before staging.
if [[ ! -e "$ANCHOR" && ! -L "$ANCHOR" ]]; then
  mkdir -m 0700 -- "$ANCHOR" 2>/dev/null \
    || die "cannot create $ANCHOR (it appeared after the check: another user may own it); remove it and re-run"
fi
[[ -d "$ANCHOR" && ! -L "$ANCHOR" && -O "$ANCHOR" ]] || die "$ANCHOR is not a directory owned by $(id -un); refusing"
[[ "$(stat -c %a -- "$ANCHOR")" == 700 ]] || die "$ANCHOR has mode $(stat -c %a -- "$ANCHOR"), not 700; refusing"
touch "$ANCHOR/.halo-kit-staging"
# Stage into src.new. Files get the extraction time as mtime (tar -m, cp without -a): git
# archive would otherwise stamp every file with the commit time, older than existing objects.
NEW="$ANCHOR/src.new"
rm -rf -- "$NEW"
mkdir -p "$NEW"
case "$SOURCE" in
  head) git -C "$REPO" archive HEAD | tar -x -m -C "$NEW" ;;
  worktree)
    (cd "$REPO" && git ls-files -z -co --exclude-standard | while IFS= read -r -d '' f; do
        [[ -e "$f" ]] && printf '%s\0' "$f"; done | tar --null -T - -cf -) | tar -x -m -C "$NEW" ;;
  copy) tar -C "$REPO" --exclude=./dist --exclude=./build --exclude=./native --exclude=./.git \
          --exclude=__pycache__ -cf - . | tar -x -m -C "$NEW" ;;
esac
for p in "${OVERLAYS[@]}"; do
  [[ "$p" != /* && "$p" != *..* ]] || die "--overlay $p: must be a relative path inside the repository"
  [[ -e "$REPO/$p" ]] || die "--overlay $p: not in the working tree"
  mkdir -p "$NEW/$(dirname "$p")"
  rm -rf -- "${NEW:?}/$p"
  cp -R "$REPO/$p" "$NEW/$(dirname "$p")/"
done
find "$NEW" -name __pycache__ -type d -prune -exec rm -rf {} +
if [[ -d "$SRC" ]]; then
  # --incremental: copy only the files whose content changed, so ninja rebuilds exactly those.
  rsync -rlc --delete "$NEW/" "$SRC/"
  rm -rf -- "$NEW"
else
  mv "$NEW" "$SRC"
fi
{
  echo "git_sha=$SHA"
  echo "dirty=$DIRTY"
  echo "dirty_paths=$(printf '%s' "$DIRTY_PATHS" | tr '\n' ';')"
  echo "source_mode=$SOURCE"
} >"$SRC/SOURCE_INFO"
# The kit tooling that is shipped is also the tooling that assembles the kit.
KIT_PY="$SRC/scripts/evox2/lib/halo_kit.py"
[[ -f "$KIT_PY" ]] || die "the staged source has no scripts/evox2 (build a commit that has it, or --overlay scripts/evox2)"

# --- configure + build ---------------------------------------------------------------------
CMAKE_OPTS=(
  -G Ninja
  -DCMAKE_C_COMPILER="$CC_" -DCMAKE_CXX_COMPILER="$CXX_"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DHALO_MARCH="$MARCH"
  -DHALO_WERROR="$WERROR"
  -DHALO_BUILD_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151
  -DHALO_BUILD_VULKAN=ON
  -DHALO_BUILD_SERVER=ON
  -DHALO_BUILD_TESTS=ON
  -DHALO_BUILD_BENCHMARKS=ON
  -DHALO_REF_DIR="$ANCHOR/ref"
)
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
LOGDIR="$OUT/logs-$NAME"
mkdir -p "$LOGDIR"
echo "package.sh: configure (log $LOGDIR/configure.log)"
cmake -S "$SRC" -B "$BUILD" "${CMAKE_OPTS[@]}" >"$LOGDIR/configure.log" 2>&1 \
  || { tail -30 "$LOGDIR/configure.log"; die "configure failed"; }
echo "package.sh: build (log $LOGDIR/build.log)"
cmake --build "$BUILD" -j "$JOBS" >"$LOGDIR/build.log" 2>&1 \
  || { grep -E "error|FAILED" "$LOGDIR/build.log" | head -30; die "build failed"; }

# --- build facts ---------------------------------------------------------------------------
first() { "$@" 2>/dev/null | head -n1 || true; }
# backends/hip sets CMAKE_HIP_COMPILER as a normal variable, so it is not in CMakeCache.txt;
# the compiler CMake actually used is recorded in CMakeFiles/<version>/CMakeHIPCompiler.cmake.
HIPCXX="$(sed -n 's/^set(CMAKE_HIP_COMPILER "\(.*\)")$/\1/p' "$BUILD"/CMakeFiles/*/CMakeHIPCompiler.cmake 2>/dev/null | head -n1 || true)"
META="$ANCHOR/meta.json"
GIT_SHA="$SHA" GIT_SHA12="$SHA12" DIRTY="$DIRTY" DIRTY_PATHS="$DIRTY_PATHS" SOURCE_MODE="$SOURCE" \
OVERLAYS="${OVERLAYS[*]:-}" MARCH="$MARCH" BUILD_TYPE="$BUILD_TYPE" KIT_NAME="$NAME" \
CXX_VERSION="$(first "$CXX_" --version)" HIP_CXX="$HIPCXX" HIP_CXX_VERSION="$(first "${HIPCXX:-false}" --version)" \
ROCM_VERSION="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)" \
HIP_VERSION="$(first /opt/rocm/bin/hipconfig --version)" CMAKE_VERSION="$(first cmake --version)" \
GLIBC_VERSION="$(first ldd --version)" BUILD_OS="$(. /etc/os-release && echo "$PRETTY_NAME")" \
BUILD_KERNEL="$(uname -r)" CMAKE_OPTS="${CMAKE_OPTS[*]}" python3 - "$META" <<'PY'
import datetime, json, os, sys
e = os.environ
meta = {
    "kit_name": e["KIT_NAME"], "git_sha": e["GIT_SHA"], "git_sha12": e["GIT_SHA12"],
    "dirty": e["DIRTY"] == "true", "dirty_paths": [l for l in e["DIRTY_PATHS"].replace(";", "\n").splitlines() if l],
    "source_mode": e["SOURCE_MODE"], "overlays": e["OVERLAYS"].split(),
    "build_date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "march": e["MARCH"], "build_type": e["BUILD_TYPE"], "cmake_options": e["CMAKE_OPTS"].split(),
    "cxx_compiler": e["CXX_VERSION"], "hip_compiler": e["HIP_CXX"], "hip_compiler_version": e["HIP_CXX_VERSION"],
    "rocm_version_built_with": e["ROCM_VERSION"], "hip_version_built_with": e["HIP_VERSION"],
    "cmake": e["CMAKE_VERSION"], "glibc_built_with": e["GLIBC_VERSION"],
    "build_os": e["BUILD_OS"], "build_kernel": e["BUILD_KERNEL"],
    "rocm_note": ("Built against ROCm %s on %s. The EVO-X2 runs HIP 7.15.26333 / ROCm runtime 1.21. "
                  "Whether the gfx1151 fatbin and the libamdhip64 dependency of the test_hip_* binaries load "
                  "under 7.15 is UNVERIFIED until the first EVO-X2 run; if they do not, build on the target "
                  "with scripts/evox2/build-native.sh." % (e["ROCM_VERSION"], e["BUILD_OS"])),
}
json.dump(meta, open(sys.argv[1], "w"), indent=2)
PY

# --- assemble the kit ----------------------------------------------------------------------
STAGE="$ANCHOR/stage"
KIT="$STAGE/$NAME"
mkdir -p "$KIT"
python3 "$KIT_PY" package assemble --build "$BUILD" --src "$SRC" --anchor "$ANCHOR" --kit "$KIT" --meta "$META"
mkdir -p "$KIT/scripts" "$KIT/docs" "$KIT/source"
cp -a "$SRC/scripts/evox2" "$KIT/scripts/"
find "$KIT/scripts" -name __pycache__ -type d -prune -exec rm -rf {} +
chmod 0755 "$KIT"/scripts/evox2/*.sh "$KIT/scripts/evox2/lib/halo_kit.py"
[[ -f "$SRC/docs/evox2.md" ]] && cp "$SRC/docs/evox2.md" "$KIT/docs/"
# The exact source that was built, for scripts/evox2/build-native.sh on the target.
tar -C "$ANCHOR" -czf "$KIT/source/halo-src-$SHA12.tar.gz" --transform "s,^src,halo-src-$SHA12," src
python3 "$KIT_PY" package check --kit "$KIT"

# BUILDINFO (human-readable) and LIBS.txt (what every binary needs at run time)
python3 - "$KIT/kit.json" >"$KIT/BUILDINFO" <<'PY'
import json, sys
k = json.load(open(sys.argv[1]))
order = ["kit_name", "git_sha", "dirty", "dirty_paths", "source_mode", "overlays", "build_date_utc", "build_type",
         "march", "cxx_compiler", "hip_compiler", "hip_compiler_version", "rocm_version_built_with",
         "hip_version_built_with", "cmake", "glibc_built_with", "build_os", "build_kernel", "cmake_options",
         "anchor", "rocm_note"]
for key in order:
    v = k.get(key)
    if isinstance(v, list):
        v = " ".join(v) if key != "dirty_paths" else "; ".join(v)
    print(f"{key}={v}")
print(f"test_binaries={len(k['tests']['binaries'])}")
print(f"ctest_tests={sum(b['discovered_tests'] for b in k['tests']['binaries'])}")
PY
{
  echo "# Shared libraries of every shipped binary, as seen on the BUILD host (readelf -d, ldd)."
  echo "# On the target, scripts/evox2/verify-env.sh runs ldd again; nothing here is bundled."
  while IFS= read -r -d '' f; do
    if head -c4 "$f" | grep -q $'\x7fELF'; then
      rel="${f#"$KIT"/}"
      echo "== $rel"
      readelf -d "$f" | grep -E 'NEEDED|RPATH|RUNPATH' | sed 's/^ */  /'
      ldd "$f" 2>&1 | sed 's/^\s*/  ldd: /'
    fi
  done < <(find "$KIT/bin" "$KIT/build" -type f -perm -u+x -print0 | sort -z)
} >"$KIT/LIBS.txt"

if [[ $IN_TREE -eq 1 ]]; then
  echo "package.sh: in-tree test run (the kit's own runner, on the build tree)"
  REFARG=()
  [[ -d "$REF" ]] && REFARG=(--ref "$REF")
  set +e
  python3 "$KIT_PY" run-tests --kit "$KIT" --no-anchor "${REFARG[@]}" --label in-tree --out "$OUT/intree-$NAME" \
    >"$LOGDIR/intree-tests.log" 2>&1
  echo "package.sh: in-tree tests exit $? (summary $OUT/intree-$NAME/summary.md)"
  set -e
  tail -1 "$LOGDIR/intree-tests.log"
fi

(cd "$KIT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
if [[ $TARBALL -eq 1 ]]; then
  echo "package.sh: writing $OUT/$NAME.tar.gz"
  tar -C "$STAGE" -czf "$OUT/$NAME.tar.gz.part" "$NAME"
  mv "$OUT/$NAME.tar.gz.part" "$OUT/$NAME.tar.gz"
  (cd "$OUT" && sha256sum "$NAME.tar.gz" >"$NAME.tar.gz.sha256")
  cp "$KIT/BUILDINFO" "$OUT/$NAME.BUILDINFO"
  RESULT="$OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
else
  rm -rf -- "${OUT:?}/$NAME"
  mv "$KIT" "$OUT/$NAME"
  RESULT="$OUT/$NAME/ (kit directory; run $OUT/$NAME/scripts/evox2/run-tests.sh)"
fi

# --- reference data ------------------------------------------------------------------------
if [[ $DO_REF -eq 1 ]]; then
  if [[ -d "$REF" ]]; then
    TMPS="$(mktemp -d)"
    (cd "$REF" && find . -type f ! -name REF_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) >"$TMPS/REF_SHA256SUMS"
    RH="$(sha256sum "$TMPS/REF_SHA256SUMS" | cut -c1-12)"
    RNAME="halo-ref-$RH"
    if [[ -f "$OUT/$RNAME.tar.gz" ]]; then
      echo "package.sh: $OUT/$RNAME.tar.gz already exists (same content hash)"
    else
      echo "package.sh: writing $OUT/$RNAME.tar.gz from $REF ($(du -sh "$REF" | cut -f1))"
      tar -czf "$OUT/$RNAME.tar.gz.part" --transform 's,^\.\(/\|$\),halo-ref\1,' \
          -C "$REF" . -C "$TMPS" ./REF_SHA256SUMS
      mv "$OUT/$RNAME.tar.gz.part" "$OUT/$RNAME.tar.gz"
      (cd "$OUT" && sha256sum "$RNAME.tar.gz" >"$RNAME.tar.gz.sha256")
    fi
    rm -rf "$TMPS"
  else
    echo "package.sh: WARNING no reference data at $REF; tests that need it will SKIP on the target"
  fi
fi

if [[ $KEEP -eq 0 ]]; then
  rm -rf -- "$ANCHOR"
else
  echo "package.sh: build kept at $ANCHOR (remove it before running a kit on this host)"
fi
echo "package.sh: done: $RESULT"
