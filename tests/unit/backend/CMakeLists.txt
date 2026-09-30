# WS-BI-1: the CPU backend adapter against the halo::cpu ops it composes (ADR-001 §5, §6.5).
halo_add_test(test_backend
  SOURCES test_cpu_backend.cpp
  LIBS halo_backend)
target_compile_options(test_backend PRIVATE -ffp-contract=off)
# WS-F2 V5: the Vulkan backend adapter vs the CPU backend, op by op (lavapipe). The adapter
# library itself is created at the end of configuration (src/backend/CMakeLists.txt), so the
# guard is on the Vulkan backend target, which exists by the time tests/ is configured.
if(TARGET halo_backend_vulkan)
  halo_add_test(test_vulkan_backend GPU
    SOURCES test_vulkan_backend.cpp
    LIBS halo_backend_vulkan_adapter halo_tensor halo_state)
  target_include_directories(test_vulkan_backend PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit)
endif()
# WS-BI-3 (ADR-001 §6.3): the HIP backend adapter in host emulation vs the CPU backend, per op
# and for the whole qwen35 forward on the tiny model. Built only with -DHALO_BUILD_HIP=ON.
if(TARGET halo_backend_hip_adapter)
  halo_add_test(test_hip_backend GPU
    SOURCES test_hip_backend.cpp
    LIBS halo_backend_hip_adapter halo_backend halo_tensor)
  target_compile_options(test_hip_backend PRIVATE -ffp-contract=off)
  if(TARGET halo_models)
    # tiny_golden.h (tests/unit/models) loads the tiny model and its test sequences.
    halo_add_test(test_hip_forward GPU
      SOURCES test_hip_forward.cpp
      LIBS halo_backend_hip_adapter halo_models)
    target_include_directories(test_hip_forward PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit/models)
    target_compile_options(test_hip_forward PRIVATE -ffp-contract=off)
  endif()
endif()
# WS-F2 V5: ADR-001 §6.3 forward differential, Qwen35 over the Vulkan backend vs over the CPU
# backend on the tiny models (needs the models module in HALO_ONLY).
if(TARGET halo_backend_vulkan AND TARGET halo_models)
  halo_add_test(test_vulkan_forward GPU
    SOURCES test_vulkan_forward.cpp
    LIBS halo_backend_vulkan_adapter halo_models)
  target_include_directories(test_vulkan_forward PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit)
endif()
