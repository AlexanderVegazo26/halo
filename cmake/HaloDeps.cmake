# Third-party dependencies (TRD §43: one of each, pinned by version + SHA256).
# Hashes computed 2026-09-23 from the archives at these exact URLs.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# JSON (MIT)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# Jinja-subset template engine (MIT) — header-only; we only need the include dir.
FetchContent_Declare(minja
  URL https://github.com/google/minja/archive/021c2293c187789ef13d56c6cfd89c9b134fd80f.tar.gz
  URL_HASH SHA256=dc3ddd37497b79a4cd35fd41550e22b3b0139439de8be6eab3d2d024fed47bb4)
FetchContent_Populate(minja)
add_library(halo_minja INTERFACE)
target_include_directories(halo_minja SYSTEM INTERFACE ${minja_SOURCE_DIR}/include)
target_link_libraries(halo_minja INTERFACE nlohmann_json::nlohmann_json)

# HTTP/1.1 + SSE server (MIT) — header-only.
if(HALO_BUILD_SERVER)
  FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.57.1.tar.gz
    URL_HASH SHA256=5c9e56b6638eb415dc451ac57531e84931fcb50e261ca3dd74eec6b484fb4712)
  FetchContent_Populate(httplib)
  add_library(halo_httplib INTERFACE)
  target_include_directories(halo_httplib SYSTEM INTERFACE ${httplib_SOURCE_DIR})
endif()

find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

if(HALO_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  set(BUILD_GMOCK OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(googletest)
endif()
