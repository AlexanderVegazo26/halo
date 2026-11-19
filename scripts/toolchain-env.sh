# Sourced by scripts/build.sh and scripts/evox2/build-native.sh (not run on its own).
#
# HALO_TOOLCHAIN=<prefix> points at a glslc installed under <prefix>/bin (e.g. a shaderc tarball
# unpacked as ~/halo-toolchain/usr). Its bin/ goes on PATH for CMake's find_program(glslc), and its
# libshaderc lives in <prefix>/lib/x86_64-linux-gnu, which must be on LD_LIBRARY_PATH or glslc
# fails with "libshaderc.so.1: cannot open shared object file". Unset: nothing changes.
if [[ -n "${HALO_TOOLCHAIN:-}" ]]; then
  if [[ ! -x "$HALO_TOOLCHAIN/bin/glslc" ]]; then
    echo "HALO_TOOLCHAIN=$HALO_TOOLCHAIN has no executable bin/glslc" >&2
    exit 2
  fi
  export PATH="$HALO_TOOLCHAIN/bin:$PATH"
  export LD_LIBRARY_PATH="$HALO_TOOLCHAIN/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
