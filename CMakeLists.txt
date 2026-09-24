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
option(HALO_ENABLE_LTO       "Enable link-time optimization"         OFF)
option(HALO_ENABLE_ASAN      "Enable AddressSanitizer + UBSan"       OFF)
option(HALO_NATIVE           "Tune for the build machine CPU"        OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(HaloWarnings)
include(HaloDeps)

if(HALO_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
if(HALO_ENABLE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()
if(HALO_NATIVE)
  add_compile_options(-march=native)
endif()

# Symbol visibility: only the public API (HALO_API) is exported (TRD §35).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_subdirectory(src)
add_subdirectory(backends)
add_subdirectory(tools)

if(HALO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
