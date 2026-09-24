# WS-G Milestone 1: qwen35 CPU forward + MTP vs golden, GGUF adapters.
halo_add_test(test_models
  SOURCES test_adapters.cpp test_qwen35_golden.cpp
  LIBS halo_models)
