# WS-G M2: MTP speculative decoding invariants on the tiny model (shares tests/unit/models/tiny_golden.h).
halo_add_test(test_speculative
  SOURCES test_speculative.cpp
  LIBS halo_speculative)
# next.md step 4 on the device: the MTP host round-trip flags (device hidden, chained drafts, stream
# reuse) over the Vulkan backend must keep drafts, greedy tokens and state bitwise unchanged.
if(TARGET halo_backend_vulkan AND TARGET halo_models)
  halo_add_test(test_vulkan_speculative GPU
    SOURCES test_vulkan_speculative.cpp
    LIBS halo_backend_vulkan_adapter halo_speculative)
  target_include_directories(test_vulkan_speculative PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit)
endif()
