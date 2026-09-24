#include "halo/tensor/dtype.h"

#include <array>

#include "halo/core/error.h"

namespace halo {
namespace {

struct Entry {
    bool known;
    DTypeTraits traits;
};

// Geometry matches ggml type_traits (ggml/src/ggml.c) and the block structs in
// ggml/src/ggml-common.h at llama.cpp bd4f514db. Cross-checked in tests against gguf-py's
// GGML_QUANT_SIZES (written into /root/halo-ref/quant_blocks/manifest.json).
constexpr std::size_t kQK_K = 256;
constexpr std::array<Entry, 31> kTable = {{
    /*  0 */ {true, {"F32", 1, 4, false}},
    /*  1 */ {true, {"F16", 1, 2, false}},
    /*  2 */ {true, {"Q4_0", 32, 18, true}},
    /*  3 */ {true, {"Q4_1", 32, 20, true}},
    /*  4 */ {false, {}},
    /*  5 */ {false, {}},
    /*  6 */ {true, {"Q5_0", 32, 22, true}},
    /*  7 */ {true, {"Q5_1", 32, 24, true}},
    /*  8 */ {true, {"Q8_0", 32, 34, true}},
    /*  9 */ {true, {"Q8_1", 32, 36, true}},
    /* 10 */ {true, {"Q2_K", kQK_K, 84, true}},
    /* 11 */ {true, {"Q3_K", kQK_K, 110, true}},
    /* 12 */ {true, {"Q4_K", kQK_K, 144, true}},
    /* 13 */ {true, {"Q5_K", kQK_K, 176, true}},
    /* 14 */ {true, {"Q6_K", kQK_K, 210, true}},
    /* 15 */ {true, {"Q8_K", kQK_K, 292, true}},
    /* 16 */ {true, {"IQ2_XXS", kQK_K, 66, true}},
    /* 17 */ {true, {"IQ2_XS", kQK_K, 74, true}},
    /* 18 */ {true, {"IQ3_XXS", kQK_K, 98, true}},
    /* 19 */ {true, {"IQ1_S", kQK_K, 50, true}},
    /* 20 */ {true, {"IQ4_NL", 32, 18, true}},
    /* 21 */ {true, {"IQ3_S", kQK_K, 110, true}},
    /* 22 */ {true, {"IQ2_S", kQK_K, 82, true}},
    /* 23 */ {true, {"IQ4_XS", kQK_K, 136, true}},
    /* 24 */ {true, {"I8", 1, 1, false}},
    /* 25 */ {true, {"I16", 1, 2, false}},
    /* 26 */ {true, {"I32", 1, 4, false}},
    /* 27 */ {true, {"I64", 1, 8, false}},
    /* 28 */ {true, {"F64", 1, 8, false}},
    /* 29 */ {true, {"IQ1_M", kQK_K, 56, true}},
    /* 30 */ {true, {"BF16", 1, 2, false}},
}};

// ggml type ids that exist at llama.cpp bd4f514db but have no DType (see dtype.h).
constexpr std::string_view extra_ggml_name(std::uint32_t id) noexcept {
    switch (id) {
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        case 39: return "MXFP4";
        case 40: return "NVFP4";
        case 41: return "Q1_0";
        case 42: return "Q2_0";
        default: return {};
    }
}

}  // namespace

std::optional<DType> dtype_from_id(std::uint32_t id) noexcept {
    if (id < kTable.size() && kTable[id].known) {
        return static_cast<DType>(id);
    }
    return std::nullopt;
}

const DTypeTraits& traits(DType t) {
    const auto id = static_cast<std::uint32_t>(t);
    if (id >= kTable.size() || !kTable[id].known) {
        throw_error(ErrorCode::Unsupported, "unknown tensor type id {}", id);
    }
    return kTable[id].traits;
}

std::string_view ggml_type_name(std::uint32_t id) noexcept {
    if (id < kTable.size() && kTable[id].known) {
        return kTable[id].traits.name;
    }
    return extra_ggml_name(id);
}

bool dequant_supported(DType t) noexcept {
    switch (t) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q5_0:
        case DType::Q5_1:
        case DType::Q8_0:
        case DType::Q2_K:
        case DType::Q3_K:
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K:
        case DType::IQ4_NL:
        case DType::IQ4_XS:
        case DType::IQ3_S:
            return true;
        default:
            return false;
    }
}

}  // namespace halo
