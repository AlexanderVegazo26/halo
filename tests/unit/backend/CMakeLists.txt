# WS-BI-1: the CPU backend adapter against the halo::cpu ops it composes (ADR-001 §5, §6.5).
halo_add_test(test_backend
  SOURCES test_cpu_backend.cpp
  LIBS halo_backend)
target_compile_options(test_backend PRIVATE -ffp-contract=off)
