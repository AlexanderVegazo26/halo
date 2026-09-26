# WS-G Milestone 1: qwen35 CPU forward + MTP vs golden, GGUF adapters.
halo_add_test(test_models
  SOURCES test_adapters.cpp test_qwen35_golden.cpp
  LIBS halo_models)
# WS-BI-1 migration gate (ADR-001 §6.2): the forward over the CPU backend vs the test-only
# legacy copy of the pre-backend forward, bitwise. The legacy copy is compiled exactly as
# halo_models compiled it (-ffp-contract=off). Delete legacy_qwen35.* at the end of WS-BI-2.
halo_add_test(test_models_gate
  SOURCES legacy_qwen35.cpp test_backend_gate.cpp
  LIBS halo_models)
target_compile_options(test_models_gate PRIVATE -ffp-contract=off)
