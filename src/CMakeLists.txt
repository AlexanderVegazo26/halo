# Each module is a static library `halo_<module>`; `halo` aggregates them.
# Module sources live in src/<module>/; public headers in include/halo/<module>/.

function(halo_module name)
  cmake_parse_arguments(M "" "" "SOURCES;DEPS" ${ARGN})
  add_library(halo_${name} STATIC ${M_SOURCES})
  target_include_directories(halo_${name} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                          PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(halo_${name} PUBLIC ${M_DEPS} PRIVATE halo_warnings)
endfunction()

halo_module(core
  SOURCES core/log.cpp
  DEPS nlohmann_json::nlohmann_json Threads::Threads)

# Modules are added below as they land (model, tensor, tokenizer, template, ...).
foreach(mod IN ITEMS tensor model tokenizer template memory hardware kv_cache state
                     sampling speculative runtime scheduler profiling api autotune)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${mod}/CMakeLists.txt)
    add_subdirectory(${mod})
  endif()
endforeach()
