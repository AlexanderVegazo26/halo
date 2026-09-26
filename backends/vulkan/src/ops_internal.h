#pragma once
// Internal operand helpers shared by the Vulkan op implementations (ops.cpp, ops_layer.cpp):
// BufferView validation and binding (code review S-1) and byte-range aliasing checks.

#include <algorithm>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "vk_util.h"

namespace halo::vulkan::detail {

inline constexpr std::uint64_t k_f32 = 4;

inline std::uint64_t round_up4(std::uint64_t v) { return (v + 3) / 4 * 4; }

/// A validated view, bound for one dispatch (see ops.h: BufferView).
struct Operand {
    BufferBinding binding;
    std::uint32_t off = 0;     ///< sub-alignment remainder of the view offset, in units
    std::uint32_t stride = 0;  ///< row stride, in units
    VkBuffer handle = VK_NULL_HANDLE;
    std::uint64_t begin = 0;  ///< touched byte range [begin, end) of the buffer
    std::uint64_t end = 0;
};

enum class Access {
    Floats,  ///< fp32 elements: offset and stride must be multiples of 4; unit = 4 bytes
    Words,   ///< byte-addressed data read as whole 32-bit words (ggml blocks); unit = 1 byte
};

/// Validates `v` for `rows` rows of `row_bytes` bytes and binds it at its offset rounded
/// down to `align`, covering only its own extent.
inline Operand operand(const BufferView& v, std::uint64_t rows, std::uint64_t row_bytes, Access access,
                std::uint64_t align, std::string_view op, std::string_view what) {
    HALO_CHECK(v.buffer != nullptr && v.buffer->valid(), ErrorCode::Kernel, "{}: '{}' has no buffer", op, what);
    HALO_CHECK(rows > 0 && row_bytes > 0, ErrorCode::Kernel, "{}: '{}' is empty", op, what);
    const std::uint64_t size = v.buffer->size();
    HALO_CHECK(v.offset <= size, ErrorCode::Kernel, "{}: '{}' offset {} is past the buffer end ({} bytes)", op, what,
               v.offset, size);
    const std::uint64_t avail = v.bytes == 0 ? size - v.offset : v.bytes;
    HALO_CHECK(avail <= size - v.offset, ErrorCode::Kernel, "{}: '{}' view [{}, +{}) exceeds the buffer ({} bytes)",
               op, what, v.offset, avail, size);
    const std::uint64_t stride = (rows == 1 || v.row_stride == 0) ? row_bytes : v.row_stride;
    HALO_CHECK(stride >= row_bytes, ErrorCode::Kernel, "{}: '{}' row_stride {} < row size {} bytes", op, what,
               stride, row_bytes);
    const std::uint64_t unit = access == Access::Floats ? k_f32 : 1;
    HALO_CHECK(v.offset % unit == 0 && stride % unit == 0, ErrorCode::Kernel,
               "{}: '{}' offset {} / row_stride {} must be multiples of {} bytes", op, what, v.offset, stride, unit);
    const std::uint64_t need = checked_add(checked_mul(rows - 1, stride, op), row_bytes, op);
    HALO_CHECK(need <= avail, ErrorCode::Kernel, "{}: '{}' needs {} bytes, the view has {}", op, what, need, avail);

    const std::uint64_t desc = v.offset / align * align;
    const std::uint64_t rem = v.offset - desc;
    const std::uint64_t span_bytes = round_up4(rem + need);
    if (access == Access::Words) {
        // The word holding the last byte read must lie inside the buffer.
        HALO_CHECK(span_bytes <= size - desc, ErrorCode::Kernel,
                   "{}: '{}' is read as 32-bit words; the buffer must extend to byte {} (it has {})", op, what,
                   desc + span_bytes, size);
    }
    const std::uint64_t range = std::min(span_bytes, size - desc);
    // Shader indices (remainder + extent, and the stride) are 32-bit.
    (void)to_u32((rem + need + unit - 1) / unit, std::format("{} '{}' index range", op, what));
    Operand o;
    o.binding = BufferBinding{v.buffer, desc, range};
    o.off = static_cast<std::uint32_t>(rem / unit);
    o.stride = to_u32(stride / unit, std::format("{} '{}' row stride", op, what));
    o.handle = v.buffer->handle();
    o.begin = v.offset;
    o.end = v.offset + need;
    return o;
}

inline bool overlaps(const Operand& a, const Operand& b) {
    return a.handle == b.handle && a.begin < b.end && b.begin < a.end;
}

/// An output must not overlap any of `others` (byte ranges in the same buffer).
inline void require_disjoint(const Operand& out, std::string_view out_name,
                      std::initializer_list<std::pair<const Operand*, std::string_view>> others, std::string_view op) {
    for (const auto& [o, name] : others) {
        if (o == nullptr) continue;
        HALO_CHECK(!overlaps(out, *o), ErrorCode::Kernel,
                   "{}: output '{}' bytes [{}, {}) overlap '{}' bytes [{}, {}) of the same buffer", op, out_name,
                   out.begin, out.end, name, o->begin, o->end);
    }
}


/// True when `a` and `b` name exactly the same elements (same buffer, same first byte, same
/// stride and extent): the only aliasing an op that says "may alias exactly" accepts.
inline bool same_range(const Operand& a, const Operand& b) {
    return a.handle == b.handle && a.begin == b.begin && a.end == b.end && a.stride == b.stride;
}

/// `out` must be disjoint from `in` or exactly alias it.
inline void require_disjoint_or_exact(const Operand& out, std::string_view out_name, const Operand& in,
                                      std::string_view in_name, std::string_view op) {
    HALO_CHECK(!overlaps(out, in) || same_range(out, in), ErrorCode::Kernel,
               "{}: output '{}' bytes [{}, {}) partially overlap '{}' bytes [{}, {}) (only an exact alias is allowed)",
               op, out_name, out.begin, out.end, in_name, in.begin, in.end);
}

/// Largest byte range one dispatch binds for an operand that can be split into row slabs
/// (gemv / get_rows weights): OpsOptions::max_binding_bytes, maxStorageBufferRange, 2^32 - 1.
inline std::uint64_t binding_limit(const OpsOptions& o, const DeviceInfo& info) {
    std::uint64_t l = info.max_storage_buffer_range;
    if (o.max_binding_bytes != 0) l = std::min(l, o.max_binding_bytes);
    return std::min<std::uint64_t>(l, 0xFFFFFFFFull);
}

/// Rows per slab such that `n` rows `stride` bytes apart (the last `row_bytes` long), bound at
/// an offset remainder < align and rounded up to a word, fit in `limit` bytes.
inline std::uint64_t rows_per_slab(std::uint64_t limit, std::uint64_t align, std::uint64_t stride, std::uint64_t row_bytes,
                                   std::string_view op) {
    HALO_CHECK(limit > align + 4 + row_bytes, ErrorCode::Unsupported,
               "{}: one {}-byte row exceeds the per-dispatch binding limit of {} bytes", op, row_bytes, limit);
    return 1 + (limit - align - 4 - row_bytes) / stride;
}

/// "<prefix>_<type>" kernel name for a weight type of matvec_row_bytes (matvec_*, get_rows_*).
inline std::string weight_shader(std::string_view prefix, DType t) {
    std::string_view suffix;
    switch (t) {
        case DType::F32: suffix = "f32"; break;
        case DType::Q8_0: suffix = "q8_0"; break;
        case DType::Q4_K: suffix = "q4_k"; break;
        case DType::Q5_K: suffix = "q5_k"; break;
        case DType::Q6_K: suffix = "q6_k"; break;
        case DType::IQ4_XS: suffix = "iq4_xs"; break;
        case DType::IQ4_NL: suffix = "iq4_nl"; break;
        case DType::Q3_K: suffix = "q3_k"; break;
        case DType::IQ3_S: suffix = "iq3_s"; break;
        default:
            throw_error(ErrorCode::Unsupported, "Vulkan {}: weight type id {} is not supported", prefix,
                        static_cast<std::uint32_t>(t));
    }
    return std::string(prefix) + "_" + std::string(suffix);
}

}  // namespace halo::vulkan::detail
