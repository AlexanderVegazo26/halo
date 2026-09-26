#!/usr/bin/env bash
# Pre-flight checks of this machine and the unpacked kit (docs/evox2.md, step 1): kit integrity,
# shared libraries, /dev/kfd and render-node access, gfx1151, ROCm version, HSA_OVERRIDE_GFX_VERSION,
# Vulkan ICDs, reference data, and the test path anchor.
#   verify-env.sh [--ref DIR] [--check-ref] [--json FILE]
# Exit 0 = no FAIL; 1 = at least one FAIL (the line says what to do).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
halo_kit verify-env "$@"
