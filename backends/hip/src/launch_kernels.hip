// Device kernels of the HIP backend: thin __global__ wrappers around the shared bodies in
// kernels/*.h, plus their launchers. Built for CMAKE_HIP_ARCHITECTURES (default gfx1151)
// with -ffp-contract=off; resource usage is printed at build time (docs/hip.md).

#include <hip/hip_runtime.h>

#include "device_exec.h"
#include "launch.h"

namespace halo::hip::detail {

using namespace kern;

namespace {

__global__ void __launch_bounds__(256) k_gdn_recurrent(GdnRecParams p) {
    __shared__ GdnRecShared sh;
    dev::DeviceExec<GdnRecRegs> ex;
    gdn_recurrent_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(1024) k_gdn_check_g(GdnCheckParams p) {
    __shared__ GdnNoShared sh;
    dev::DeviceExec<GdnNoRegs> ex;
    gdn_check_g_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(256) k_gdn_chunk_intra(GdnChunkParams p) {
    __shared__ GdnIntraShared sh;
    dev::DeviceExec<GdnNoRegs> ex;
    gdn_chunk_intra_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(kGdnStateMaxBlock) k_gdn_chunk_state(GdnChunkParams p) {
    __shared__ GdnStateShared sh;
    dev::DeviceExec<GdnNoRegs> ex;
    gdn_chunk_state_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(1024) k_conv1d_silu(ConvParams p) {
    __shared__ ConvShared sh;
    dev::DeviceExec<ConvRegs> ex;
    conv1d_silu_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(1024) k_norm(NormParams p) {
    __shared__ NormShared sh;
    dev::DeviceExec<NormRegs> ex;
    norm_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(kGemvMaxBlock) k_gemv_generic(GemvParams p) {
    __shared__ GemvGenericShared sh;
    dev::DeviceExec<GemvNoRegs> ex;
    gemv_generic_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(kGemvMaxBlock) k_gemv_wave(GemvParams p) {
    __shared__ GemvWaveShared sh;
    dev::DeviceExec<GemvNoRegs> ex;
    gemv_wave_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(kArgMaxBlock) k_argmax_partial(ArgmaxParams p) {
    __shared__ ArgShared sh;
    dev::DeviceExec<ArgNoRegs> ex;
    argmax_partial_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(kArgMaxBlock) k_argmax_reduce(ArgmaxParams p) {
    __shared__ ArgShared sh;
    dev::DeviceExec<ArgNoRegs> ex;
    argmax_reduce_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(256) k_topk(TopkParams p) {
    __shared__ TopkShared sh;
    dev::DeviceExec<ArgNoRegs> ex;
    topk_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(256) k_rope(RopeParams p) {
    __shared__ RopeShared sh;
    dev::DeviceExec<ArgNoRegs> ex;
    rope_body(ex, sh, p, blockIdx.x, blockIdx.y);
}

__global__ void __launch_bounds__(1024) k_eltwise(EwParams p) {
    __shared__ EwShared sh;
    dev::DeviceExec<ArgNoRegs> ex;
    ew_body(ex, sh, p, blockIdx.x, blockIdx.y);
}


template <class K, class P>
int launch(K kernel, const P& p, const Launch& l, void* stream) {
    hipLaunchKernelGGL(kernel, dim3(l.grid_x, l.grid_y, 1), dim3(l.block, 1, 1), 0,
                       static_cast<hipStream_t>(stream), p);
    return static_cast<int>(hipGetLastError());
}

}  // namespace

int launch_gdn_recurrent(const GdnRecParams& p, const Launch& l, void* s) { return launch(k_gdn_recurrent, p, l, s); }
int launch_gdn_check_g(const GdnCheckParams& p, const Launch& l, void* s) { return launch(k_gdn_check_g, p, l, s); }
int launch_gdn_chunk_intra(const GdnChunkParams& p, const Launch& l, void* s) { return launch(k_gdn_chunk_intra, p, l, s); }
int launch_gdn_chunk_state(const GdnChunkParams& p, const Launch& l, void* s) { return launch(k_gdn_chunk_state, p, l, s); }
int launch_conv1d_silu(const ConvParams& p, const Launch& l, void* s) { return launch(k_conv1d_silu, p, l, s); }
int launch_norm(const NormParams& p, const Launch& l, void* s) { return launch(k_norm, p, l, s); }
int launch_gemv_generic(const GemvParams& p, const Launch& l, void* s) { return launch(k_gemv_generic, p, l, s); }
int launch_gemv_wave(const GemvParams& p, const Launch& l, void* s) { return launch(k_gemv_wave, p, l, s); }
int launch_argmax_partial(const ArgmaxParams& p, const Launch& l, void* s) { return launch(k_argmax_partial, p, l, s); }
int launch_argmax_reduce(const ArgmaxParams& p, const Launch& l, void* s) { return launch(k_argmax_reduce, p, l, s); }
int launch_topk(const TopkParams& p, const Launch& l, void* s) { return launch(k_topk, p, l, s); }
int launch_rope(const RopeParams& p, const Launch& l, void* s) { return launch(k_rope, p, l, s); }
int launch_eltwise(const EwParams& p, const Launch& l, void* s) { return launch(k_eltwise, p, l, s); }

}  // namespace halo::hip::detail
