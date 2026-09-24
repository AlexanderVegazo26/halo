#!/usr/bin/env bash
# Build + test HALO inside WSL/Linux. Sources may live on /mnt/c; the build tree lives on
# the Linux filesystem for speed.
#   scripts/build.sh [--asan] [--no-test] [extra cmake args...]
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${HALO_BUILD_DIR:-$HOME/halo-build}"
TEST=1
ARGS=()
for a in "$@"; do
  case "$a" in
    --asan) BUILD="${BUILD}-asan"; ARGS+=(-DHALO_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug) ;;
    --no-test) TEST=0 ;;
    *) ARGS+=("$a") ;;
  esac
done
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "${ARGS[@]}"
cmake --build "$BUILD"
if [[ $TEST == 1 ]]; then
  ctest --test-dir "$BUILD" --output-on-failure -j"$(nproc)"
fi
