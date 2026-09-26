# Backend interface + CPU reference adapter (ADR-001, WS-BI-1). The model forward is
# composed over halo::backend::Backend; the CPU backend calls the frozen halo::cpu ops.
halo_module(backend
  SOURCES backend.cpp cpu_backend.cpp
  DEPS halo_backend_cpu halo_tensor halo_core)
# Reference results must not depend on HALO_NATIVE (same reason as halo_backend_cpu).
target_compile_options(halo_backend PRIVATE -ffp-contract=off)

# WS-BI-3: the HIP backend adapter (ADR-001 §9) in its own library, so halo_models (which
# links halo_backend) never links HIP (ADR §5.1 layering). src/ is configured before
# backends/, so `if(TARGET halo_backend_hip)` would always be false here: guard on the same
# conditions that create that target; the link name resolves at generate time.
halo_selected(hip _halo_bi3_hip)
if(HALO_BUILD_HIP AND _halo_bi3_hip)
  add_library(halo_backend_hip_adapter STATIC hip_adapter.cpp)
  target_include_directories(halo_backend_hip_adapter PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(halo_backend_hip_adapter PUBLIC halo_backend halo_backend_hip PRIVATE halo_tensor halo_warnings)
  target_compile_options(halo_backend_hip_adapter PRIVATE -ffp-contract=off)
endif()
