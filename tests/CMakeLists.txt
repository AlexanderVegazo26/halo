include(GoogleTest)

# Reference data (real GGUF headers, golden tensors) lives outside the repo; see
# python/tools/ (fetch_gguf_headers.py, fetch_hf_files.py, fetch_quant_blocks.py and the
# make_*.py generators, e.g. make_tiny_model.py). Tests that need it read HALO_REF_DIR and call
# GTEST_SKIP() with a message when it is absent — a skip is reported, never a pass.
set(HALO_REF_DIR "$ENV{HOME}/halo-ref" CACHE PATH "Reference data directory")

# GPU: the test opens the real GPU (Vulkan or HIP). Such tests hold the ctest resource lock
# "gpu" and never run concurrently under `ctest -j`: several GPU processes at once (Vulkan +
# HIP) wedged a compute ring on gfx1151 and forced a full GPU reset, which also kills the
# desktop compositor (2026-09-29, MES failed to respond / VRAM lost).
#
# PROPERTIES: extra ctest properties for every discovered test (e.g. RUN_SERIAL TRUE LABELS perf).
function(halo_add_test name)
  cmake_parse_arguments(T "GPU" "" "SOURCES;LIBS;PROPERTIES" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE ${T_LIBS} GTest::gtest_main halo_warnings)
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE HALO_REF_DIR="${HALO_REF_DIR}"
                                             HALO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  set(_props ${T_PROPERTIES})
  if(T_GPU)
    list(APPEND _props RESOURCE_LOCK gpu)
  endif()
  if(_props)
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60 PROPERTIES ${_props})
  else()
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
  endif()
endfunction()

halo_add_test(test_core SOURCES unit/test_core.cpp LIBS halo_core)

# scripts/test-agent's report parser and wrapper (stdlib Python, no build needed).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME test_agent_report
           COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/agent)
endif()

file(GLOB _test_dirs LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/unit/*)
foreach(d IN LISTS _test_dirs)
  get_filename_component(_dname ${d} NAME)
  halo_selected(${_dname} _sel)
  if(_sel AND IS_DIRECTORY ${d} AND EXISTS ${d}/CMakeLists.txt)
    add_subdirectory(${d})
  endif()
endforeach()

# Fuzz targets: seed-corpus replay in every build, libFuzzer campaigns with HALO_BUILD_FUZZERS.
halo_selected(fuzz _sel)
if(_sel)
  add_subdirectory(fuzz)
endif()
