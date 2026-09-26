# Sourced by the scripts/evox2/*.sh wrappers. Finds python3 and the kit tooling.
HALO_KIT_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib"
if ! command -v python3 >/dev/null 2>&1; then
  echo "$(basename "$0"): python3 is required (sudo apt install python3)" >&2
  exit 2
fi
halo_kit() { exec python3 "$HALO_KIT_LIB/halo_kit.py" "$@"; }
