# Runtime Engine (WS-G M3): CPU reference engine, batched scheduler, prefix cache.
halo_module(runtime
  SOURCES engine.cpp profile.cpp
  DEPS halo_autotune halo_profiling halo_speculative halo_models halo_state halo_kv_cache halo_tokenizer halo_template halo_memory halo_hardware
       halo_sampling halo_core)

# BI-6: the GPU backend adapters, linked into the runtime so engine.cpp can construct them.
# The adapter targets are created after src/ is configured (src/backend/CMakeLists.txt
# explains), so the guard mirrors their creation condition and the link names resolve at
# generate time.
halo_selected(hip _halo_rt_hip)
if(HALO_BUILD_HIP AND _halo_rt_hip)
  target_link_libraries(halo_runtime PUBLIC halo_backend_hip_adapter)
  target_compile_definitions(halo_runtime PRIVATE HALO_RUNTIME_HIP=1)
endif()
halo_selected(vulkan _halo_rt_vulkan)
if(HALO_BUILD_VULKAN AND _halo_rt_vulkan)
  target_link_libraries(halo_runtime PUBLIC halo_backend_vulkan_adapter)
  target_compile_definitions(halo_runtime PRIVATE HALO_RUNTIME_VULKAN=1)
endif()
