# halo_memory uses the header-only tier vocabulary and the plain MemoryTiers struct from
# include/halo/hardware/, but does not link halo_hardware.
halo_module(memory
  SOURCES planner.cpp
  DEPS halo_core nlohmann_json::nlohmann_json)
