#pragma once
// Device side of the kernel execution model (kernels/hd.h): one phase = the calling
// thread's share of the work followed by a workgroup barrier. HIP translation units only.

#include <hip/hip_runtime.h>

#include "kernels/hd.h"

namespace halo::hip::dev {

template <class Regs>
struct DeviceExec {
    Regs r;
    __device__ unsigned block_dim() const { return blockDim.x; }
    template <class F>
    __device__ void phase(F&& f) {
        f(threadIdx.x, r);
        __syncthreads();
    }
    __device__ static void atomic_or(std::uint32_t* p, std::uint32_t v) { atomicOr(p, v); }
};

}  // namespace halo::hip::dev
