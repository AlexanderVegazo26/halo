// Backend-neutral parts of the backend interface (ADR-001 §5.1): names, run_op, result decode.

#include "halo/backend/backend.h"

#include <array>
#include <bit>
#include <cstring>
#include <limits>

#include "halo/core/error.h"

namespace halo::backend {

std::string_view to_string(Kind k) noexcept {
    switch (k) {
        case Kind::Cpu: return "cpu";
        case Kind::Hip: return "hip";
        case Kind::HipEmulation: return "hip-emulation";
        case Kind::Vulkan: return "vulkan";
    }
    return "unknown";
}

std::string_view op_name(OpId op) noexcept {
    switch (op) {
        case OpId::GetRows: return "GET_ROWS";
        case OpId::RmsNorm: return "RMS_NORM";
        case OpId::AddRmsNorm: return "ADD_RMS_NORM";
        case OpId::Gemv: return "MATMUL";
        case OpId::GdnGates: return "GDN_GATES";
        case OpId::Conv1dSilu: return "CONV1D_SHORT";
        case OpId::GatedDeltaRule: return "GATED_DELTANET";
        case OpId::GatedRmsNorm: return "GATED_NORM";
        case OpId::PartialRope: return "PARTIAL_ROPE";
        case OpId::KvWrite: return "KV_WRITE";
        case OpId::Attention: return "ATTENTION";
        case OpId::Swiglu: return "SWIGLU";
        case OpId::MulSigmoid: return "MUL_SIGMOID";
        case OpId::Add: return "ADD";
        case OpId::LmHead: return "LM_HEAD";
        case OpId::Argmax: return "ARGMAX_FUSED";
        case OpId::TopK: return "TOP_K";
        case OpId::Copy: return "COPY";
    }
    return "UNKNOWN";
}

namespace {

template <class A>
const A& args_of(const OpInvocation& inv) {
    const A* a = std::get_if<A>(&inv.args);
    HALO_CHECK(a != nullptr, ErrorCode::Api, "run_op: {} called with the args of another op", op_name(inv.op));
    return *a;
}

}  // namespace

void Backend::run_op(Stream& s, const OpInvocation& inv) {
    switch (inv.op) {
        case OpId::GetRows: return get_rows(s, args_of<GetRowsArgs>(inv));
        case OpId::RmsNorm: return rms_norm(s, args_of<RmsNormArgs>(inv));
        case OpId::AddRmsNorm: return add_rms_norm(s, args_of<AddRmsNormArgs>(inv));
        case OpId::Gemv: return gemv(s, args_of<GemvArgs>(inv));
        case OpId::GdnGates: return gdn_gates(s, args_of<GdnGateArgs>(inv));
        case OpId::Conv1dSilu: return conv1d_silu(s, args_of<Conv1dArgs>(inv));
        case OpId::GatedDeltaRule: return gated_delta_rule(s, args_of<GdnArgs>(inv));
        case OpId::GatedRmsNorm: return gated_rms_norm(s, args_of<GatedNormArgs>(inv));
        case OpId::PartialRope: return partial_rope(s, args_of<RopeArgs>(inv));
        case OpId::KvWrite: return kv_write(s, args_of<KvWriteArgs>(inv));
        case OpId::Attention: return attention(s, args_of<AttentionArgs>(inv));
        case OpId::Swiglu: return swiglu(s, args_of<EltwiseArgs>(inv));
        case OpId::MulSigmoid: return mul_sigmoid(s, args_of<EltwiseArgs>(inv));
        case OpId::Add: return add(s, args_of<EltwiseArgs>(inv));
        case OpId::LmHead: return lm_head(s, args_of<LmHeadArgs>(inv));
        case OpId::Argmax: return argmax(s, args_of<ArgmaxArgs>(inv));
        case OpId::TopK: return top_k(s, args_of<TopKArgs>(inv));
        case OpId::Copy: return copy(s, args_of<CopyArgs>(inv));
    }
    throw_error(ErrorCode::Api, "run_op: unknown op id {}", static_cast<unsigned>(inv.op));
}

std::vector<ArgmaxResult> decode_argmax(std::span<const std::byte> words) {
    HALO_CHECK(words.size() % kArgmaxResultBytes == 0, ErrorCode::Kernel, "decode_argmax: {} bytes is not a multiple of {}",
               words.size(), kArgmaxResultBytes);
    const std::size_t n = words.size() / kArgmaxResultBytes;
    std::vector<ArgmaxResult> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<std::uint32_t, 3> w{};
        std::memcpy(w.data(), words.data() + i * kArgmaxResultBytes, sizeof(w));
        // H1: a NaN word poisons only its own vector (ArgmaxResult{-1, NaN}), not the whole
        // decode -- a batched LM-head call spans multiple sequences, and one bad row must not
        // stop the caller from reading the others'. The caller decides what a poisoned row
        // means for the sequence it belongs to.
        out[i] = w[2] != 0 ? ArgmaxResult{-1, std::numeric_limits<float>::quiet_NaN()}
                           : ArgmaxResult{static_cast<std::int32_t>(w[0]), std::bit_cast<float>(w[1])};
    }
    return out;
}

}  // namespace halo::backend
