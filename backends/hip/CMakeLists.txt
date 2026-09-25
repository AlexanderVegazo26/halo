# HALO HIP backend (WS-K; TRD §17, §18, §63). HALO's own kernels only: no rocBLAS,
# hipBLASLt or MIOpen (TRD §3.3). Enabled by -DHALO_BUILD_HIP=ON (backends/CMakeLists.txt).
#
# Two libraries:
#   halo_hip_device  — the .hip translation units (ROCm clang, CMake's HIP language):
#                      __global__ wrappers around the shared bodies in kernels/*.h.
#   halo_backend_hip — host C++ (HALO's clang, -Werror): runtime, registry, op validation,
#                      and the host emulation of the same kernel bodies (tests, D-001).

if(NOT DEFINED CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX})
  set(CMAKE_HIP_COMPILER /opt/rocm/llvm/bin/clang++)
endif()
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP offload architectures (TRD §3.3: gfx1151)")
option(HALO_HIP_ALLOW_GENERIC "Allow building the HIP kernels for gfx11-generic instead of gfx1151" OFF)
enable_language(HIP)
find_package(hip CONFIG REQUIRED GLOBAL PATHS /opt/rocm)  # GLOBAL: tests use hip::host

set(_halo_hip_generic 0)
if(CMAKE_HIP_ARCHITECTURES MATCHES "generic")
  if(NOT HALO_HIP_ALLOW_GENERIC)
    message(FATAL_ERROR "HIP: CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}' is a generic ISA. "
                        "TRD §3.3 requires gfx1151 unless the generic target is chosen explicitly: "
                        "pass -DHALO_HIP_ALLOW_GENERIC=ON (recorded in the build and logged at runtime).")
  endif()
  set(_halo_hip_generic 1)
  message(WARNING "HIP: building device code for generic ISA '${CMAKE_HIP_ARCHITECTURES}' (TRD §63)")
elseif(NOT CMAKE_HIP_ARCHITECTURES MATCHES "gfx1151")
  message(WARNING "HIP: CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}' does not include gfx1151; "
                  "Context::create will refuse a gfx1151 device")
endif()
message(STATUS "HIP: device code for '${CMAKE_HIP_ARCHITECTURES}' (HIP ${hip_VERSION})")

# Every file that includes kernels/*.h (device and host emulation) is built without FP
# contraction, like the CPU reference, so the emulation reproduces the device's roundings.
add_library(halo_hip_device STATIC src/launch_kernels.hip)
set_target_properties(halo_hip_device PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF)
# ${PROJECT_SOURCE_DIR}/src: kernels/gemv.h uses halo::tensor's own IQ tables (src/tensor/ggml_tables.h).
target_include_directories(halo_hip_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                   ${PROJECT_SOURCE_DIR}/src)
target_compile_options(halo_hip_device PRIVATE -ffp-contract=off -Rpass-analysis=kernel-resource-usage)
# ASan builds (HALO_ENABLE_ASAN): the host side of the .hip TU is sanitized; device-side
# sanitizers need an xnack+ target, which gfx1151 is not, so they are disabled explicitly.
if(HALO_ENABLE_ASAN)
  target_compile_options(halo_hip_device PRIVATE -fno-gpu-sanitize)
endif()
target_link_libraries(halo_hip_device PRIVATE halo_warnings)

add_library(halo_backend_hip STATIC
  src/runtime.cpp
  src/registry.cpp
  src/ops.cpp
  src/emulate.cpp)
target_include_directories(halo_backend_hip PUBLIC ${PROJECT_SOURCE_DIR}/include
                                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                    ${PROJECT_SOURCE_DIR}/src)
string(REPLACE ";" "," _halo_hip_archs "${CMAKE_HIP_ARCHITECTURES}")
target_compile_definitions(halo_backend_hip PRIVATE HALO_HIP_OFFLOAD_ARCHS="${_halo_hip_archs}"
                                                    HALO_HIP_GENERIC_ISA=${_halo_hip_generic})
target_compile_options(halo_backend_hip PRIVATE -ffp-contract=off)
# PUBLIC halo_tensor: ops.h uses halo::DType (review N-2 pattern: include => link).
target_link_libraries(halo_backend_hip PUBLIC halo_core halo_tensor PRIVATE hip::host halo_hip_device halo_warnings)
