#pragma once
// Tokenizer + chat template straight from a GGUF's metadata (header-only open, so the
// real *.header.gguf files work), for `halo tokenize` / `halo template`.

#include <filesystem>
#include <memory>
#include <string>

#include "halo/template/chat_template.h"
#include "halo/tokenizer/tokenizer.h"

namespace halo::cli {

struct ModelText {
    std::unique_ptr<tokenizer::Tokenizer> tokenizer;
    std::unique_ptr<chat::ChatTemplate> chat_template;  ///< null when the GGUF has none
};

/// Throws halo::Error (Model/Unsupported/Io/Config) for unusable files.
[[nodiscard]] ModelText load_model_text(const std::filesystem::path& gguf);

}  // namespace halo::cli
