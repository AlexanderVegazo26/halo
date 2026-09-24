// GGUF -> CPU-operator adapters (code review S-4): WeightRef -> WeightMatrix / fp32 vectors,
// embedding rows, and GGUF tokenizer metadata -> tokenizer::VocabSpec.

#include <cstdint>
#include <limits>
#include <string>

#include "halo/core/error.h"
#include "halo/models/qwen35.h"
#include "halo/tensor/quant.h"

namespace halo::models {

namespace {

const float* as_floats(const std::byte* p, const std::string& name) {
    HALO_CHECK(reinterpret_cast<std::uintptr_t>(p) % alignof(float) == 0, ErrorCode::Model,
               "tensor {}: F32 data is not 4-byte aligned", name);
    return static_cast<const float*>(static_cast<const void*>(p));
}

}  // namespace

cpu::WeightMatrix weight_matrix(const model::WeightRef& w, std::size_t first_row, std::optional<std::size_t> n_rows) {
    const auto& info = w.info();
    HALO_CHECK(info.n_dims == 2 && info.ne[2] == 1 && info.ne[3] == 1 && info.ne[0] > 0 && info.ne[1] > 0, ErrorCode::Model,
               "tensor {}: expected a 2-D matrix, n_dims {}, ne = [{}, {}, {}, {}]", info.name, info.n_dims, info.ne[0],
               info.ne[1], info.ne[2], info.ne[3]);
    const auto cols = static_cast<std::size_t>(info.ne[0]);
    const auto total_rows = static_cast<std::size_t>(info.ne[1]);
    HALO_CHECK(first_row <= total_rows, ErrorCode::Model, "tensor {}: row {} of {}", info.name, first_row, total_rows);
    const std::size_t rows = n_rows.value_or(total_rows - first_row);
    HALO_CHECK(rows <= total_rows - first_row, ErrorCode::Model, "tensor {}: rows [{}, {}) of {}", info.name, first_row,
               first_row + rows, total_rows);
    HALO_CHECK(dequant_supported(info.type), ErrorCode::Unsupported, "tensor {}: no dequantization for {}", info.name,
               traits(info.type).name);
    const std::size_t rb = tensor::row_bytes(info.type, info.ne[0]);
    const std::span<const std::byte> data = w.data();
    HALO_CHECK(data.size() / rb >= total_rows, ErrorCode::Model, "tensor {}: {} bytes for {} rows of {}", info.name,
               data.size(), total_rows, rb);
    const std::byte* base = data.data() + first_row * rb;
    if (info.type == DType::F32) {
        return cpu::WeightMatrix::dense(cpu::ConstRows(as_floats(base, info.name), rows, cols, cols));
    }
    const DType type = info.type;
    return cpu::WeightMatrix::dequantized(rows, cols, [type, base, rb, cols](std::size_t first, std::size_t n, std::span<float> out) {
        tensor::dequantize_row(type, base + first * rb, out.data(), static_cast<std::int64_t>(n * cols));
    });
}

std::vector<float> weight_vector(const model::WeightRef& w) {
    const auto& info = w.info();
    HALO_CHECK(info.n_elements <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()), ErrorCode::Model,
               "tensor {}: too many elements", info.name);
    const auto n = static_cast<std::int64_t>(info.n_elements);
    const std::span<const std::byte> data = w.data();
    HALO_CHECK(data.size() == tensor::row_bytes(info.type, n), ErrorCode::Model, "tensor {}: {} bytes for {} elements of {}",
               info.name, data.size(), n, traits(info.type).name);
    std::vector<float> out(static_cast<std::size_t>(n));
    tensor::dequantize_row(info.type, data, out);
    return out;
}

void embedding_row(const model::WeightRef& embd, std::int32_t token, std::span<float> out) {
    const auto& info = embd.info();
    HALO_CHECK(token >= 0 && token < info.ne[1], ErrorCode::Api, "token id {} outside the vocabulary [0, {})", token,
               info.ne[1]);
    HALO_CHECK(out.size() == static_cast<std::size_t>(info.ne[0]), ErrorCode::Kernel, "embedding_row: {} floats for n_embd {}",
               out.size(), info.ne[0]);
    const std::size_t rb = tensor::row_bytes(info.type, info.ne[0]);
    const std::span<const std::byte> data = embd.data();
    HALO_CHECK(data.size() / rb > static_cast<std::size_t>(token), ErrorCode::Model, "tensor {}: truncated", info.name);
    tensor::dequantize_row(info.type, data.subspan(static_cast<std::size_t>(token) * rb, rb), out);
}

tokenizer::VocabSpec vocab_spec(const model::TokenizerMetadata& meta) {
    HALO_CHECK(meta.model == "gpt2", ErrorCode::Unsupported, "tokenizer.ggml.model '{}' is not supported (need gpt2)",
               meta.model);
    HALO_CHECK(meta.pre == "qwen35", ErrorCode::Unsupported, "tokenizer.ggml.pre '{}' is not supported (need qwen35)",
               meta.pre);
    HALO_CHECK(!meta.tokens.empty(), ErrorCode::Model, "GGUF has no tokenizer.ggml.tokens");
    HALO_CHECK(meta.tokens.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()), ErrorCode::Model,
               "vocabulary of {} tokens exceeds int32 ids", meta.tokens.size());
    HALO_CHECK(meta.token_types.empty() || meta.token_types.size() == meta.tokens.size(), ErrorCode::Model,
               "tokenizer.ggml.token_type has {} entries for {} tokens", meta.token_types.size(), meta.tokens.size());
    tokenizer::VocabSpec spec;
    spec.tokens.assign(meta.tokens.begin(), meta.tokens.end());
    spec.merges.assign(meta.merges.begin(), meta.merges.end());
    spec.types.reserve(meta.token_types.size());
    for (std::size_t i = 0; i < meta.token_types.size(); ++i) {
        const std::int64_t t = meta.token_types[i];
        HALO_CHECK(t >= 1 && t <= 6, ErrorCode::Model, "token {} has type {} (valid: 1..6)", i, t);
        spec.types.push_back(static_cast<tokenizer::TokenType>(t));
    }
    const auto id = [&](const std::optional<std::uint32_t>& v, const char* what) -> std::optional<std::int32_t> {
        if (!v) return std::nullopt;
        HALO_CHECK(*v < meta.tokens.size(), ErrorCode::Model, "{} id {} outside the vocabulary of {}", what, *v,
                   meta.tokens.size());
        return static_cast<std::int32_t>(*v);
    };
    spec.bos = id(meta.bos_id, "bos");
    spec.eos = id(meta.eos_id, "eos");
    spec.pad = id(meta.pad_id, "pad");
    spec.pre_tokenizer = tokenizer::PreTokenizer::Qwen35;
    spec.normalizer = tokenizer::Normalizer::Nfc;  // D-015: match tokenizer.json
    return spec;
}

}  // namespace halo::models
