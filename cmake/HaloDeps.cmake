# Third-party dependencies (TRD §43: one of each, pinned by version + SHA256).
# Hashes computed 2026-09-23 from the archives at these exact URLs.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# JSON (MIT). The second URL is the same release asset (id 244339406) through the REST API:
# GitHub serves it separately from the github.com download URL, which returned 504 for
# minutes on 2026-11-08 and failed CI's configure step. CMake tries the URLs in order and
# checks URL_HASH whichever one served the file; the Accept header makes the API return the
# file instead of its JSON metadata (the download URL ignores it).
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
      https://api.github.com/repos/nlohmann/json/releases/assets/244339406
  HTTP_HEADER "Accept: application/octet-stream"
  DOWNLOAD_NAME json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# Header-only deps below: FetchContent_MakeAvailable (FetchContent_Populate with declared
# details is deprecated, CMP0169) with SOURCE_SUBDIR pointing at a directory that doesn't
# exist, so the archive is downloaded, hash-checked and patched but its own CMakeLists.txt
# (tests, install rules, options) is never added to the build. We only need the include dir.
set(HALO_HEADER_ONLY_SUBDIR halo-header-only-no-cmake)

# Jinja-subset template engine (MIT) — header-only; we only need the include dir.
# HALO carries a pinned patch (security review S-3; docs/security-hardening.md): resource
# limits, linear lexing, iterative Value teardown. URL_HASH still verifies the upstream
# archive; apply_patch.cmake verifies the patch file against its own SHA256 and is
# idempotent. src/template/chat_template.cpp refuses to compile against unpatched minja, so a
# FETCHCONTENT_SOURCE_DIR_MINJA override must point at a patched tree.
set(HALO_MINJA_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/minja-halo-limits.patch")
set(HALO_MINJA_PATCH_SHA256 54da3b56abf72b152a15db5ecd308af727ab5b416f84fbdff700f2e720f5e275)
FetchContent_Declare(minja
  URL https://github.com/google/minja/archive/021c2293c187789ef13d56c6cfd89c9b134fd80f.tar.gz
  URL_HASH SHA256=dc3ddd37497b79a4cd35fd41550e22b3b0139439de8be6eab3d2d024fed47bb4
  PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH_FILE=${HALO_MINJA_PATCH} -DPATCH_SHA256=${HALO_MINJA_PATCH_SHA256}
                -P ${CMAKE_CURRENT_LIST_DIR}/patches/apply_patch.cmake
  SOURCE_SUBDIR ${HALO_HEADER_ONLY_SUBDIR})
FetchContent_MakeAvailable(minja)
add_library(halo_minja INTERFACE)
target_include_directories(halo_minja SYSTEM INTERFACE ${minja_SOURCE_DIR}/include)
target_link_libraries(halo_minja INTERFACE nlohmann_json::nlohmann_json)

# HTTP/1.1 + SSE server (MIT) — header-only.
if(HALO_BUILD_SERVER)
  FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.65.0.tar.gz
    URL_HASH SHA256=6fda8f23ff811a751c2b6e694fe2f1aa1625df53a317110329ef57a18f9fa5fb
    SOURCE_SUBDIR ${HALO_HEADER_ONLY_SUBDIR})
  FetchContent_MakeAvailable(httplib)
  add_library(halo_httplib INTERFACE)
  target_include_directories(halo_httplib SYSTEM INTERFACE ${httplib_SOURCE_DIR})
endif()

find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

if(HALO_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5)
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  set(BUILD_GMOCK OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(googletest)
endif()
