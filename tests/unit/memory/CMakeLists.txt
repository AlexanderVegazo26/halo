halo_add_test(test_memory SOURCES test_planner.cpp LIBS halo_memory)

# The fixture-driven planner tests need hardware discovery; only build them when the
# hardware module is part of this build (HALO_ONLY may exclude it).
if(TARGET halo_hardware)
  halo_add_test(test_memory_fixtures SOURCES test_planner_fixtures.cpp LIBS halo_memory halo_hardware)
endif()
