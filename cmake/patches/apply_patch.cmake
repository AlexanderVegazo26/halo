# Applies a pinned patch to a FetchContent dependency's source tree (the working directory
# of the patch step). Run as:
#   cmake -DPATCH_FILE=<file> -DPATCH_SHA256=<hex> -P apply_patch.cmake
#
# - The patch file must match PATCH_SHA256 (recorded in HaloDeps.cmake), so an edited patch
#   cannot be applied silently. The upstream archive itself is still verified by URL_HASH.
# - Idempotent: a tree that already carries the patch is left alone, so re-running the
#   patch step (ExternalProject re-runs it when its command line changes) does not fail.
# - A tree carrying neither the pristine nor the patched text fails loudly: delete the
#   dependency's <build>/_deps/<name>-* directories and reconfigure (the error says so).
if(NOT PATCH_FILE OR NOT PATCH_SHA256)
  message(FATAL_ERROR "apply_patch.cmake: PATCH_FILE and PATCH_SHA256 are required")
endif()
file(SHA256 "${PATCH_FILE}" _actual)
if(NOT _actual STREQUAL PATCH_SHA256)
  message(FATAL_ERROR "apply_patch.cmake: SHA256 of ${PATCH_FILE} is ${_actual}, expected ${PATCH_SHA256}")
endif()
find_program(_patch_exe patch REQUIRED)
execute_process(COMMAND "${_patch_exe}" -p1 -R --fuzz=0 --dry-run --silent --force -i "${PATCH_FILE}"
                RESULT_VARIABLE _already OUTPUT_QUIET ERROR_QUIET)
if(_already EQUAL 0)
  message(STATUS "apply_patch.cmake: ${PATCH_FILE} already applied")
  return()
endif()
execute_process(COMMAND "${_patch_exe}" -p1 --forward --fuzz=0 --dry-run --silent -i "${PATCH_FILE}"
                RESULT_VARIABLE _fits OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _fits EQUAL 0)
  # The patch step runs in <build>/_deps/<name>-src.
  get_filename_component(_src_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  string(REGEX REPLACE "-src$" "" _dep "${_src_name}")
  message(FATAL_ERROR "apply_patch.cmake: ${PATCH_FILE} does not apply to ${CMAKE_CURRENT_SOURCE_DIR}: the tree "
                      "is neither pristine nor patched (hand-edited or another version). "
                      "Delete <build>/_deps/${_dep}-* and reconfigure.\n${_out}${_err}")
endif()
execute_process(COMMAND "${_patch_exe}" -p1 --forward --fuzz=0 --silent -i "${PATCH_FILE}" RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "apply_patch.cmake: applying ${PATCH_FILE} failed (${_rc})")
endif()
message(STATUS "apply_patch.cmake: applied ${PATCH_FILE}")
