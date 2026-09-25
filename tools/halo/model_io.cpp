#include "model_io.h"

#include <limits>

#include "halo/core/error.h"
#include "halo/model/model.h"

namespace halo::cli {

// GGUF tokenizer metadata -> VocabSpec. This mirrors halo::models::vocab_spec
// (src/models/adapters.cpp) on purpose: the CLI's tokenize/template helpers must not link
// the model forward (halo_models pulls in the CPU backend, KV cache and state modules).
// Keep the two in step (D-008, D-015).
namespace {

tokenizer::VocabSpec vocab_from(const model::TokenizerMetadata& meta) {
    HALO_CHECK(meta.model == "gpt2", ErrorCode::Unsupported, "tokenizer.ggml.model '{}' is not supported (need gpt2)",
               meta.model);
    HALO_CHECK(meta.pre == "qwen35", ErrorCode::Unsupported, "tokenizer.ggml.pre '{}' is not supported (need qwen35)",
               meta.pre);
    HALO_CHECK(!meta.tokens.empty(), ErrorCode::Model, "GGUF has no tokenizer.ggml.tokens");
    HALO_CHECK(meta.tokens.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
               ErrorCode::Model, "vocabulary of {} tokens exceeds int32 ids", meta.tokens.size());
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
    spec.normalizer = tokenizer::Normalizer::Nfc;
    return spec;
}

}  // namespace

ModelText load_model_text(const std::filesystem::path& gguf) {
    const auto m = model::NormalizedModel::load(gguf, model::GgufMode::HeaderOnly);
    const auto meta = m.tokenizer();
    ModelText out;
    out.tokenizer = std::make_unique<tokenizer::Tokenizer>(tokenizer::Tokenizer::from_spec(vocab_from(meta)));
    if (meta.chat_template && !meta.chat_template->empty()) {
        const auto piece = [&](const std::optional<std::uint32_t>& id) {
            return id ? out.tokenizer->token_to_piece(static_cast<std::int32_t>(*id)) : std::string{};
        };
        out.chat_template = std::make_unique<chat::ChatTemplate>(std::string(*meta.chat_template),
                                                                 piece(meta.bos_id), piece(meta.eos_id));
    }
    return out;
}

}  // namespace halo::cli
