#pragma once
// HALO tokenizer (PRD FR-014, TRD §10, D-008): byte-level BPE with the qwen35
// pre-tokenizer and NFC normalizer, matching HF `tokenizers` token-for-token.
//
// Pipeline for encode(text, parse_special):
//   1. Invalid UTF-8 is replaced by U+FFFD per maximal subpart (the policy of Python's
//      `bytes.decode('utf-8', 'replace')`); the model never sees raw invalid bytes.
//   2. Added tokens are matched first on the raw text, leftmost-longest. With
//      parse_special=true every added token (Control and UserDefined) is matched; with
//      parse_special=false only UserDefined ones (e.g. `<think>`, `<tool_call>`) are —
//      the same as HF's `encode_special_tokens=True`. BPE never merges across them.
//   3. Each remaining segment is NFC-normalized, split by the qwen35 regex (hand-written,
//      no std::regex) and BPE-encoded (HF merge order: lowest rank, then leftmost).
//
// Thread safety: a Tokenizer is immutable after construction; all const members may be
// called concurrently. StreamDecoder is per-sequence state and not thread-safe.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace halo::tokenizer {

/// GGUF `tokenizer.ggml.token_type` values.
enum class TokenType : std::uint8_t {
    Normal = 1,
    Unknown = 2,      ///< rejected (Unsupported): byte-level BPE has no unk
    Control = 3,      ///< special: matched only with parse_special, skipped by skip_special
    UserDefined = 4,  ///< added token: always matched, never skipped
    Unused = 5,       ///< padding entries (e.g. `[PAD248077]`): never produced, decode to ""
    Byte = 6,         ///< rejected (Unsupported): SentencePiece byte tokens
};

enum class PreTokenizer : std::uint8_t {
    Qwen35,  ///< GGUF `tokenizer.ggml.pre = qwen35`
};

enum class Normalizer : std::uint8_t {
    None,
    Nfc,  ///< HF tokenizer.json `{"type": "NFC"}`; llama.cpp does not apply it
};

/// Raw vocabulary, as a GGUF loader reads it (`tokenizer.ggml.*`).
struct VocabSpec {
    /// Token strings indexed by id. Normal tokens use the GPT-2 byte-to-unicode mapping
    /// (`Ġ` = space); Control/UserDefined tokens are literal text.
    std::vector<std::string> tokens;
    /// Merge rules "left right" in the same mapped form, highest priority first.
    std::vector<std::string> merges;
    /// Per-token types; empty means all Normal. Size must match `tokens` otherwise.
    std::vector<TokenType> types;
    std::optional<std::int32_t> bos;
    std::optional<std::int32_t> eos;
    std::optional<std::int32_t> pad;
    PreTokenizer pre_tokenizer = PreTokenizer::Qwen35;
    Normalizer normalizer = Normalizer::Nfc;
};

/// Token ids plus, for each token, the byte offset in the *input* text where the text that
/// produced it begins. Offsets are non-decreasing. They are exact for added tokens and for
/// text NFC leaves unchanged; a token that starts inside a multi-byte character (byte-level
/// BPE may split one) or inside a sequence NFC rewrote reports that character's /
/// sequence's start. Use `token_index_at` to map a byte offset to a token index.
struct Encoding {
    std::vector<std::int32_t> ids;
    std::vector<std::size_t> offsets;

    /// Index of the first token whose offset is >= `byte_offset` (ids.size() if none).
    [[nodiscard]] std::size_t token_index_at(std::size_t byte_offset) const noexcept;
};

class Tokenizer {
public:
    /// Builds from raw vectors (the GGUF path). Throws halo::Error: Model for malformed
    /// data (bad merge, missing byte token, out-of-range special id, duplicate added
    /// token), Unsupported for token types this implementation does not handle.
    static Tokenizer from_spec(VocabSpec spec);

    /// Builds from HF `tokenizer.json` text; `tokenizer_config_json` (optional) supplies
    /// bos/eos/pad. Configs other than NFC + qwen35 Split + ByteLevel + BPE (no dropout,
    /// no unk, no byte fallback, merges not ignored) are rejected with Unsupported.
    static Tokenizer from_hf_json(std::string_view tokenizer_json, std::string_view tokenizer_config_json = {});

    /// Reads `dir/tokenizer.json` (+ `dir/tokenizer_config.json` if present). Io on read
    /// failure; files larger than 256 MiB are rejected.
    static Tokenizer from_hf_dir(const std::filesystem::path& dir);

    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    ~Tokenizer();

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view utf8, bool parse_special) const;
    [[nodiscard]] Encoding encode_with_offsets(std::string_view utf8, bool parse_special) const;

    /// Concatenates token bytes (see token_to_piece) and converts the result to valid
    /// UTF-8, replacing ill-formed subparts with U+FFFD (HF's lossy decode). With
    /// skip_special, Control tokens are omitted. Throws Api for an id outside the vocab.
    [[nodiscard]] std::string decode(std::span<const std::int32_t> ids, bool skip_special) const;

    /// Raw bytes of one token; may be an incomplete UTF-8 sequence. Unused tokens map to
    /// "". Throws Api for an id outside the vocab.
    [[nodiscard]] const std::string& token_to_piece(std::int32_t id) const;

    /// Id of the token whose raw bytes are exactly `piece` (Normal or added), if any.
    [[nodiscard]] std::optional<std::int32_t> piece_to_id(std::string_view piece) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] TokenType token_type(std::int32_t id) const;
    [[nodiscard]] bool is_special(std::int32_t id) const;  ///< Control
    [[nodiscard]] std::optional<std::int32_t> bos() const noexcept;
    [[nodiscard]] std::optional<std::int32_t> eos() const noexcept;
    [[nodiscard]] std::optional<std::int32_t> pad() const noexcept;
    [[nodiscard]] Normalizer normalizer() const noexcept;

    struct Impl;

private:
    explicit Tokenizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// Incremental detokenizer: feed ids one at a time, get back only complete UTF-8.
/// Bytes of a character split across tokens are held until it completes; a sequence that
/// can no longer become valid is emitted as U+FFFD. Concatenating every push() result
/// and a final flush() equals Tokenizer::decode of all ids.
/// The Tokenizer must outlive the decoder.
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer& tokenizer, bool skip_special = false);

    [[nodiscard]] std::string push(std::int32_t id);
    /// Emits U+FFFD for a held incomplete tail and resets.
    [[nodiscard]] std::string flush();
    /// Bytes currently held back (an incomplete UTF-8 prefix).
    [[nodiscard]] std::size_t pending_bytes() const noexcept { return pending_.size(); }

private:
    const Tokenizer* tok_;
    bool skip_special_;
    std::string pending_;
};

}  // namespace halo::tokenizer
