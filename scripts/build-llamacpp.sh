#!/usr/bin/env bash
# Build llama.cpp (CPU-only) at the commit the EVO-X2 runs (bd4f514db), used as a
# reference implementation and for llama-quantize. Not part of HALO's runtime.
set -euo pipefail
PIN=bd4f514db14d87fded667787a7a963bfbaa98e89
cd /root
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch -q --depth 1 origin "$PIN"
git checkout -q "$PIN"
actual=$(git rev-parse HEAD)
if [ "$actual" != "$PIN" ]; then
  echo "build-llamacpp.sh: HEAD is $actual, expected pinned commit $PIN" >&2
  exit 1
fi
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-quantize llama-cli llama-server llama-eval-callback llama-tokenize -j"$(nproc)"
ls build/bin
