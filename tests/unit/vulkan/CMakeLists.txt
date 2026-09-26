# Vulkan backend tests. GPU kernels are accepted only by differential tests against the
# CPU reference (halo::cpu) and halo::tensor dequantization on the same input buffers
# (TRD §30, DECISIONS D-016, code review M-2), so both are linked here.
if(TARGET halo_backend_vulkan)
  foreach(_dep IN ITEMS halo_backend_cpu halo_tensor)
    if(NOT TARGET ${_dep})
      message(FATAL_ERROR "tests/unit/vulkan needs ${_dep}: add 'cpu' and 'tensor' to HALO_ONLY")
    endif()
  endforeach()
  halo_add_test(test_vulkan
    SOURCES test_vk_host.cpp test_vk_device.cpp test_vk_ops.cpp test_vk_differential.cpp test_vk_views.cpp test_vk_layer.cpp test_vk_rows.cpp test_vk_head.cpp
    LIBS halo_backend_vulkan halo_backend_cpu halo_tensor)
  # Shared fp64 reference + tolerance models of the CPU kernel tests (error-bound scaling).
  target_include_directories(test_vulkan PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit)
  # KV write and paged attention read their inputs from the real halo::kv_cache pool (layout parity).
  if(TARGET halo_kv_cache)
    halo_add_test(test_vulkan_kv SOURCES test_vk_kv.cpp LIBS halo_backend_vulkan halo_backend_cpu halo_kv_cache)
  else()
    message(WARNING "Vulkan: halo_kv_cache is not selected (add kv_cache to HALO_ONLY); test_vulkan_kv (KV write "
                    "and paged attention vs halo::kv_cache / cpu::attention_gqa) is NOT built")
  endif()
endif()
