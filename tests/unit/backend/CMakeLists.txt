# WS-BI-1: the CPU backend adapter against the halo::cpu ops it composes (ADR-001 §5, §6.5).
halo_add_test(test_backend
  SOURCES test_cpu_backend.cpp
  LIBS halo_backend)
target_compile_options(test_backend PRIVATE -ffp-contract=off)
# WS-BI-3 (ADR-001 §6.3): the HIP backend adapter in host emulation vs the CPU backend, per op
# and for the whole qwen35 forward on the tiny model. Built only with -DHALO_BUILD_HIP=ON.
if(TARGET halo_backend_hip_adapter)
  halo_add_test(test_hip_backend
    SOURCES test_hip_backend.cpp
    LIBS halo_backend_hip_adapter halo_backend halo_tensor)
  target_compile_options(test_hip_backend PRIVATE -ffp-contract=off)
  if(TARGET halo_models)
    # tiny_golden.h (tests/unit/models) loads the tiny model and its test sequences.
    halo_add_test(test_hip_forward
      SOURCES test_hip_forward.cpp
      LIBS halo_backend_hip_adapter halo_models)
    target_include_directories(test_hip_forward PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit/models)
    target_compile_options(test_hip_forward PRIVATE -ffp-contract=off)
  endif()
endif()
