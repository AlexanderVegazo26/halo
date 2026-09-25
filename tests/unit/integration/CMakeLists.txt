# WS-G M4: the separately tested halves meet. `halo run` / `halo bench` through the CLI
# library (tools/halo run_cli), the HTTP ApiServer over the real CPU Engine on a loopback
# port, and the TRD §64 tune -> persist -> next-launch lookup chain.
#
# A missing component is a reported skip, never a silently shorter test list.
set(_needed halo_runtime halo_cli halo_api halo_httplib halo_autotune halo_autotune_cpu halo_profiling)
set(_missing "")
foreach(t IN LISTS _needed)
  if(NOT TARGET ${t})
    list(APPEND _missing ${t})
  endif()
endforeach()
if(_missing)
  string(REPLACE ";" ", " _missing_str "${_missing}")
  halo_add_test(test_integration SOURCES test_missing.cpp LIBS halo_core)
  target_compile_definitions(test_integration PRIVATE HALO_INTEGRATION_MISSING="${_missing_str}")
  return()
endif()
halo_add_test(test_integration
  SOURCES test_integration.cpp
  LIBS halo_runtime halo_cli halo_api halo_httplib halo_autotune halo_autotune_cpu halo_profiling)
