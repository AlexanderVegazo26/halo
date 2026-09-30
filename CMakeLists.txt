cmake_minimum_required(VERSION 3.25)
project(halo VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HALO_BUILD_TESTS      "Build unit/integration tests"          ON)
option(HALO_BUILD_BENCHMARKS "Build benchmark tools"                 ON)
option(HALO_BUILD_SERVER     "Build the HTTP API server"             ON)
option(HALO_BUILD_VULKAN     "Build the Vulkan/RADV backend"         ON)
option(HALO_BUILD_HIP        "Build the HIP backend (needs ROCm)"    OFF)
# Off by default: this hasn't been benchmarked yet (no A/B build comparison exists), and the
# multi-static-lib-per-module layout means a from-scratch LTO build is materially slower --
# not a rejected-for-a-bug default, just an unevaluated one (cpp23-efficiency-review.md §H/§K).
option(HALO_ENABLE_LTO       "Enable link-time optimization"         OFF)
option(HALO_ENABLE_ASAN      "Enable AddressSanitizer + UBSan"       OFF)
option(HALO_NATIVE           "Tune for the build machine CPU"        OFF)
# CPU target passed as -march=<value> to every HALO target (C, C++ and the host side of HIP).
# Empty (the default) passes no -march, i.e. the compiler's baseline x86-64; HALO_NATIVE=ON
# is the same as HALO_MARCH=native. Portable packages use a psABI level such as x86-64-v3
# (scripts/package.sh; docs/evox2.md explains the choice).
set(HALO_MARCH "" CACHE STRING "CPU target for -march (empty = compiler default; e.g. x86-64-v3, native)")
option(HALO_WERROR           "Treat compiler warnings as errors"    ON)
# Semicolon list of module / backend / test-dir names to build (empty = everything).
# Lets parallel workstreams build only their own code in a shared tree, e.g.
#   -DHALO_ONLY="tensor;model"
set(HALO_ONLY "" CACHE STRING "Build only these modules (empty = all)")

function(halo_selected name out)
  if(HALO_ONLY STREQUAL "" OR name IN_LIST HALO_ONLY)
    set(${out} TRUE PARENT_SCOPE)
  else()
    set(${out} FALSE PARENT_SCOPE)
  endif()
endfunction()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(HaloWarnings)
if(NOT HALO_WERROR)
  # For building on a newer compiler whose new warnings would otherwise stop the build
  # (scripts/evox2/build-native.sh). The warnings are still printed.
  target_compile_options(halo_warnings INTERFACE -Wno-error)
  message(STATUS "HALO_WERROR=OFF: warnings are not errors")
endif()
include(HaloDeps)

if(HALO_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
if(HALO_ENABLE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()
if(NOT HALO_MARCH STREQUAL "")
  if(NOT HALO_MARCH MATCHES "^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    message(FATAL_ERROR "HALO_MARCH='${HALO_MARCH}' is not a CPU target name (e.g. x86-64-v3, znver5, native)")
  endif()
  if(HALO_NATIVE AND NOT HALO_MARCH STREQUAL "native")
    message(FATAL_ERROR "HALO_NATIVE=ON conflicts with HALO_MARCH='${HALO_MARCH}': set only one")
  endif()
  add_compile_options(-march=${HALO_MARCH})
  message(STATUS "CPU target: -march=${HALO_MARCH}")
elseif(HALO_NATIVE)
  add_compile_options(-march=native)
  message(STATUS "CPU target: -march=native (HALO_NATIVE)")
endif()

# Symbol visibility: only the public API (HALO_API) is exported (TRD §35).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_subdirectory(src)
add_subdirectory(backends)
add_subdirectory(tools)
include(HaloInstall)

if(HALO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
