# Warnings as an interface target; linked PRIVATE by every HALO target.
add_library(halo_warnings INTERFACE)
target_compile_options(halo_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
  -Wcast-align -Wnull-dereference -Wdouble-promotion -Wformat=2
  -Wimplicit-fallthrough -Werror)
