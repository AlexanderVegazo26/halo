# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-src")
  file(MAKE_DIRECTORY "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-src")
endif()
file(MAKE_DIRECTORY
  "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-build"
  "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix"
  "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix/tmp"
  "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix/src/minja-populate-stamp"
  "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix/src"
  "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix/src/minja-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix/src/minja-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/alexander-vegazo/Documents/repos/halo/build-m2/_deps/minja-subbuild/minja-populate-prefix/src/minja-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
