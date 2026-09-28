# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 4.2.3)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(minja-populate NONE)



include(ExternalProject)
ExternalProject_Add(minja-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/google/minja/archive/021c2293c187789ef13d56c6cfd89c9b134fd80f.tar.gz" "URL_HASH" "SHA256=dc3ddd37497b79a4cd35fd41550e22b3b0139439de8be6eab3d2d024fed47bb4" "PATCH_COMMAND" "/usr/bin/cmake" "-DPATCH_FILE=/home/alexander-vegazo/Documents/repos/halo/cmake/patches/minja-halo-limits.patch" "-DPATCH_SHA256=54da3b56abf72b152a15db5ecd308af727ab5b416f84fbdff700f2e720f5e275" "-P" "/home/alexander-vegazo/Documents/repos/halo/cmake/patches/apply_patch.cmake"
                    SOURCE_DIR          "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-src"
                    BINARY_DIR          "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


