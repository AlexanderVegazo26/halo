#pragma once
// Internal seam between the host op layer (ops.cpp, clang 18) and the kernels: each kernel
// has a device launcher (launch_*.hip, ROCm clang; returns the hipError_t as int, 0 =
// success) and a host emulator (emulate.cpp) that runs the identical body on host memory.

#include "kernels/conv_norm.h"
#include "kernels/gdn.h"
#include "kernels/gemv.h"
#include "kernels/attention.h"
#include "kernels/decode.h"
#include "kernels/gemm.h"
#include "kernels/head.h"

namespace halo::hip::detail {

// Device launchers: enqueue on `stream` (a hipStream_t), return hipGetLastError().
int launch_gdn_recurrent(const kern::GdnRecParams& p, const kern::Launch& l, void* stream);
int launch_gdn_check_g(const kern::GdnCheckParams& p, const kern::Launch& l, void* stream);
int launch_gdn_chunk_intra(const kern::GdnChunkParams& p, const kern::Launch& l, void* stream);
int launch_gdn_chunk_state(const kern::GdnChunkParams& p, const kern::Launch& l, void* stream);
int launch_conv1d_silu(const kern::ConvParams& p, const kern::Launch& l, void* stream);
int launch_norm(const kern::NormParams& p, const kern::Launch& l, void* stream);
int launch_gemv_generic(const kern::GemvParams& p, const kern::Launch& l, void* stream);
int launch_gemv_wave(const kern::GemvParams& p, const kern::Launch& l, void* stream);
int launch_argmax_partial(const kern::ArgmaxParams& p, const kern::Launch& l, void* stream);
int launch_argmax_reduce(const kern::ArgmaxParams& p, const kern::Launch& l, void* stream);
int launch_topk(const kern::TopkParams& p, const kern::Launch& l, void* stream);
int launch_rope(const kern::RopeParams& p, const kern::Launch& l, void* stream);
int launch_eltwise(const kern::EwParams& p, const kern::Launch& l, void* stream);
int launch_attn_check(const kern::AttnParams& p, const kern::Launch& l, void* stream);
int launch_attn_exact(const kern::AttnParams& p, const kern::Launch& l, void* stream);
int launch_attn_online(const kern::AttnParams& p, const kern::Launch& l, void* stream);
int launch_kv_write(const kern::KvWriteParams& p, const kern::Launch& l, void* stream);
int launch_get_rows(const kern::GetRowsParams& p, const kern::Launch& l, void* stream);
int launch_add_norm(const kern::AddNormParams& p, const kern::Launch& l, void* stream);
int launch_gdn_gate(const kern::GdnGateParams& p, const kern::Launch& l, void* stream);
int launch_gemm(const kern::GemvParams& p, const kern::Launch& l, void* stream);

// Host emulators (reverse = run threads and blocks in reverse order).
void emulate_gdn_recurrent(const kern::GdnRecParams& p, const kern::Launch& l, bool reverse);
void emulate_gdn_check_g(const kern::GdnCheckParams& p, const kern::Launch& l, bool reverse);
void emulate_gdn_chunk_intra(const kern::GdnChunkParams& p, const kern::Launch& l, bool reverse);
void emulate_gdn_chunk_state(const kern::GdnChunkParams& p, const kern::Launch& l, bool reverse);
void emulate_conv1d_silu(const kern::ConvParams& p, const kern::Launch& l, bool reverse);
void emulate_norm(const kern::NormParams& p, const kern::Launch& l, bool reverse);
void emulate_gemv_generic(const kern::GemvParams& p, const kern::Launch& l, bool reverse);
void emulate_gemv_wave(const kern::GemvParams& p, const kern::Launch& l, bool reverse);
void emulate_argmax_partial(const kern::ArgmaxParams& p, const kern::Launch& l, bool reverse);
void emulate_argmax_reduce(const kern::ArgmaxParams& p, const kern::Launch& l, bool reverse);
void emulate_topk(const kern::TopkParams& p, const kern::Launch& l, bool reverse);
void emulate_rope(const kern::RopeParams& p, const kern::Launch& l, bool reverse);
void emulate_eltwise(const kern::EwParams& p, const kern::Launch& l, bool reverse);
void emulate_attn_check(const kern::AttnParams& p, const kern::Launch& l, bool reverse);
void emulate_attn_exact(const kern::AttnParams& p, const kern::Launch& l, bool reverse);
void emulate_attn_online(const kern::AttnParams& p, const kern::Launch& l, bool reverse);
void emulate_kv_write(const kern::KvWriteParams& p, const kern::Launch& l, bool reverse);
void emulate_get_rows(const kern::GetRowsParams& p, const kern::Launch& l, bool reverse);
void emulate_add_norm(const kern::AddNormParams& p, const kern::Launch& l, bool reverse);
void emulate_gdn_gate(const kern::GdnGateParams& p, const kern::Launch& l, bool reverse);
void emulate_gemm(const kern::GemvParams& p, const kern::Launch& l, bool reverse);

}  // namespace halo::hip::detail
