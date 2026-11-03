#pragma once
// Invariant failure in a fuzz target: print what broke and abort, so libFuzzer saves the
// input as a crash-* reproducer and the replay driver dies on the input it just named.

#include <cstdio>
#include <cstdlib>

#define HALO_FUZZ_FAIL(...)                                              \
    do {                                                                 \
        std::fprintf(stderr, "FUZZ INVARIANT BROKEN (%s:%d): ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__);                               \
        std::fprintf(stderr, "\n");                                      \
        std::fflush(stderr);                                             \
        std::abort();                                                    \
    } while (0)
