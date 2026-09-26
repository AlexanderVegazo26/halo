# Install rules (`cmake --install <build> --prefix <dir>`), used by scripts/evox2/build-native.sh.
#
# Installs the `halo` CLI and the reference benchmark workloads. The EVO-X2 field kit
# (scripts/package.sh) does not use these rules: it ships the build-tree binaries unchanged,
# because the test binaries embed build-tree paths (see docs/evox2.md, "How the kit is laid out").
#
# No RPATH is added: the `halo` binary links no ROCm library (only the test_hip_* binaries do,
# and they are not installed). System libraries come from the target system.
include(GNUInstallDirs)

if(TARGET halo_cli_main)
  install(TARGETS halo_cli_main RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY ${PROJECT_SOURCE_DIR}/bench/workloads/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/halo/workloads
        FILES_MATCHING PATTERN "*.json")
