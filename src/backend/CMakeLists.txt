# Backend interface + CPU reference adapter (ADR-001, WS-BI-1). The model forward is
# composed over halo::backend::Backend; the CPU backend calls the frozen halo::cpu ops.
halo_module(backend
  SOURCES backend.cpp cpu_backend.cpp
  DEPS halo_backend_cpu halo_tensor halo_core)
# Reference results must not depend on HALO_NATIVE (same reason as halo_backend_cpu).
target_compile_options(halo_backend PRIVATE -ffp-contract=off)
