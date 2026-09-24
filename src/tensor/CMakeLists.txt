halo_module(tensor
  SOURCES dtype.cpp quant.cpp
  DEPS halo_core)
# Defensive: keep dequantization rounding identical to ggml/gguf-py even if a future format
# has a non-exact product that an FMA could fuse (see the header comment in quant.cpp).
target_compile_options(halo_tensor PRIVATE -ffp-contract=off)
