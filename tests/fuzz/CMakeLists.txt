# Coverage-guided fuzz targets (libFuzzer entry points; tests/fuzz/README.md).
#
# Every build: each target links replay_main.cpp and ctest `fuzz_replay_<target>` runs it over
# the checked-in seed corpus (tests/fuzz/corpus/<target>), so the plain, ASan and TSan lanes
# all replay the seeds and every reproducer added to the corpus.
#
# HALO_BUILD_FUZZERS=ON (clang only; set at the top level, which also builds the whole tree
# with -fsanitize=fuzzer-no-link): the targets link libFuzzer instead, and ctest
# `fuzz_smoke_<target>` (label `fuzz`) runs a short campaign with a fixed seed and run count
# (deterministic, ~seconds) through tests/fuzz/run.py, which keeps the checked-in corpus
# read-only and writes new inputs and crash reproducers under the report dir.
if(NOT TARGET halo_model OR NOT TARGET halo_template)
  return()
endif()

function(halo_add_fuzzer name)
  cmake_parse_arguments(F "" "" "SOURCES;LIBS" ${ARGN})
  set(_corpus ${CMAKE_CURRENT_SOURCE_DIR}/corpus/${name})
  if(HALO_BUILD_FUZZERS)
    add_executable(fuzz_${name} ${F_SOURCES})
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
  else()
    add_executable(fuzz_${name} ${F_SOURCES} replay_main.cpp)
  endif()
  target_link_libraries(fuzz_${name} PRIVATE ${F_LIBS} halo_warnings)
  target_include_directories(fuzz_${name} PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  if(HALO_BUILD_FUZZERS)
    if(Python3_Interpreter_FOUND)
      add_test(NAME fuzz_smoke_${name}
               COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py --smoke
                       $<TARGET_FILE:fuzz_${name}> ${_corpus})
      set_tests_properties(fuzz_smoke_${name} PROPERTIES LABELS fuzz TIMEOUT 600)
    endif()
    # libFuzzer runs each corpus file once with `-runs=0`: the same replay as a normal build.
    add_test(NAME fuzz_replay_${name} COMMAND fuzz_${name} -runs=0 ${_corpus})
  else()
    add_test(NAME fuzz_replay_${name} COMMAND fuzz_${name} ${_corpus})
  endif()
  set_property(TEST fuzz_replay_${name} APPEND PROPERTY LABELS fuzz)
endfunction()

halo_add_fuzzer(gguf SOURCES fuzz_gguf.cpp LIBS halo_model)
halo_add_fuzzer(output_parser SOURCES fuzz_output_parser.cpp LIBS halo_template)
halo_add_fuzzer(chat_template SOURCES fuzz_chat_template.cpp LIBS halo_template)
if(TARGET halo_api)
  halo_add_fuzzer(api_request
    SOURCES fuzz_api_request.cpp ${PROJECT_SOURCE_DIR}/tests/unit/api/fake_engine.cpp
    LIBS halo_api halo_template halo_tokenizer halo_core)
  target_include_directories(fuzz_api_request PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit/api)
  target_compile_definitions(fuzz_api_request PRIVATE HALO_REF_DIR="${HALO_REF_DIR}")
endif()

# Regenerates the binary GGUF seeds (corpus/gguf) from tests/unit/model/gguf_builder.h.
if(NOT HALO_BUILD_FUZZERS)
  add_executable(fuzz_make_seeds fuzz_make_seeds.cpp)
  target_link_libraries(fuzz_make_seeds PRIVATE halo_model halo_warnings)
  target_include_directories(fuzz_make_seeds PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit/model)
endif()
