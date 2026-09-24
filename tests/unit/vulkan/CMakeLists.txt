if(TARGET halo_backend_vulkan)
  halo_add_test(test_vulkan
    SOURCES test_vk_host.cpp test_vk_device.cpp test_vk_ops.cpp
    LIBS halo_backend_vulkan)
endif()
