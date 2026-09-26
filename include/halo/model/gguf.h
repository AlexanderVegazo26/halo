#pragma once
// GGUF v2/v3 container parser (little-endian). The file is untrusted input: every length,
// count, offset and size is bounds- and overflow-checked, and any violation throws a typed
// halo::Error (MODEL_ERROR for malformed files, UNSUPPORTED_ERROR for well-formed files
// using features HALO does not support, IO_ERROR for filesystem failures). Nothing in this
// parser relies on the file being well formed for memory safety.
//
// Two modes:
//  - GgufMode::Full: the whole file is mmap'ed read-only; every tensor's byte range must
//    lie inside the file; tensor_data() returns views into the mapping.
//  - GgufMode::HeaderOnly: only KV metadata + tensor infos are required to be present
//    (e.g. the truncated *.header.gguf reference files, which end before the aligned data
//    start). Tensor ranges cannot be checked against EOF, and tensor_data() throws
//    MODEL_ERROR.
//
// Thread-safety: a GgufFile is immutable after construction; concurrent const access is safe.
// Lifetime: tensor_data() spans and all returned references are valid while the GgufFile
// lives (moving a GgufFile keeps them valid).

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "halo/model/mapped_file.h"
#include "halo/tensor/dtype.h"

namespace halo::model {

/// GGUF metadata value types (gguf spec numbering).
enum class GgufType : std::uint32_t {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
};

[[nodiscard]] std::string_view to_string(GgufType t) noexcept;

/// A metadata array. Elements live in the vector matching `elem_type`:
/// unsigned ints -> uints, signed ints -> ints, F32/F64 -> floats, Bool -> bools (0/1),
/// String -> strings, Array -> arrays (nested). The other vectors are empty.
struct GgufArray {
    GgufType elem_type = GgufType::U8;
    std::vector<std::uint64_t> uints;
    std::vector<std::int64_t> ints;
    std::vector<double> floats;
    std::vector<std::uint8_t> bools;
    std::vector<std::string> strings;
    std::vector<GgufArray> arrays;

    [[nodiscard]] std::size_t size() const noexcept;
};

/// A metadata value; the member matching `type` holds it (ints widened to 64 bits, floats
/// to double — the original width is preserved in `type`).
struct GgufValue {
    GgufType type = GgufType::U8;
    std::uint64_t u = 0;
    std::int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    GgufArray arr;
};

struct GgufTensorInfo {
    std::string name;
    std::uint32_t n_dims = 0;
    std::array<std::int64_t, 4> ne{1, 1, 1, 1};  // ggml order: ne[0] is the contiguous dim
    DType type = DType::F32;
    std::uint64_t offset = 0;  // relative to the data section; multiple of the alignment
    std::uint64_t n_elements = 0;
    std::uint64_t n_bytes = 0;

    /// Number of ne[0]-element rows. Computed in uint64_t (L1): a hostile header with
    /// ne[0] == 0 passes the full-product overflow check trivially, and signed ne[1..3]
    /// products can then overflow int64_t (UB) before any caller sees them.
    [[nodiscard]] std::int64_t rows() const noexcept {
        const auto r = static_cast<std::uint64_t>(ne[1]) * static_cast<std::uint64_t>(ne[2]) *
                       static_cast<std::uint64_t>(ne[3]);
        return static_cast<std::int64_t>(r);
    }
};

enum class GgufMode { Full, HeaderOnly };

/// Parser limits. Chosen well above any real model (Qwen3.8: 248,320 tokens, ~9 KB chat
/// template, 851-866 tensors) while keeping a hostile header from requesting absurd
/// allocations. Counts are additionally bounded by the bytes actually remaining.
struct GgufLimits {
    static constexpr std::uint64_t kMaxKv = 1u << 16;
    static constexpr std::uint64_t kMaxTensors = 1u << 20;
    static constexpr std::uint64_t kMaxStringBytes = 16u << 20;
    static constexpr std::uint64_t kMaxArrayElems = 1u << 26;
    static constexpr int kMaxArrayDepth = 8;
    static constexpr std::uint64_t kMaxTensorNameBytes = 63;  // ggml GGML_MAX_NAME - 1
    static constexpr std::uint64_t kMaxAlignment = 1u << 20;
    /// Default total allocation budget for the parsed header (S-4), see GgufOptions.
    static constexpr std::uint64_t kMaxMetadataBytes = std::uint64_t{1} << 30;
};

/// Parse options (additive to GgufMode).
struct GgufOptions {
    /// Total bytes the parsed header may allocate: every metadata value and tensor info is
    /// charged elements x stored size (8 per integer/float element, sizeof(std::string) +
    /// its bytes per string, sizeof(GgufArray) per nested array, plus per-KV and per-tensor
    /// bookkeeping) before it is allocated. The on-disk encoding is up to ~13x smaller than
    /// the stored form, so the per-array caps alone let a large "model" request hundreds of
    /// GB (security review S-4). Exceeding it throws Error(Model). Real Qwen3.8 headers use
    /// ~30 MB (GgufFile::metadata_bytes()). 0 disables the budget.
    std::uint64_t max_metadata_bytes = GgufLimits::kMaxMetadataBytes;
};

class GgufFile {
public:
    /// Opens and parses `path`. Throws Error(Io) if the file cannot be mapped, Error(Model)
    /// or Error(Unsupported) if it is not a valid/supported GGUF for `mode`.
    static GgufFile open(const std::filesystem::path& path, GgufMode mode = GgufMode::Full);
    static GgufFile open(const std::filesystem::path& path, GgufMode mode, const GgufOptions& options);

    /// Parses an in-memory image (takes ownership). Same validation as open().
    static GgufFile parse(std::vector<std::byte> bytes, GgufMode mode, std::string source_name = "<memory>");
    static GgufFile parse(std::vector<std::byte> bytes, GgufMode mode, std::string source_name,
                          const GgufOptions& options);

    GgufFile(GgufFile&&) noexcept = default;
    GgufFile& operator=(GgufFile&&) noexcept = default;
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;
    ~GgufFile() = default;

    [[nodiscard]] const std::string& source() const noexcept { return source_; }
    [[nodiscard]] GgufMode mode() const noexcept { return mode_; }
    [[nodiscard]] bool has_tensor_data() const noexcept { return mode_ == GgufMode::Full; }
    [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
    [[nodiscard]] std::uint64_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] std::uint64_t header_size() const noexcept { return header_size_; }  // end of tensor infos
    [[nodiscard]] std::uint64_t data_offset() const noexcept { return data_offset_; }  // aligned data start
    [[nodiscard]] std::uint64_t file_size() const noexcept { return bytes_.size(); }
    /// Bytes charged against GgufOptions::max_metadata_bytes while parsing.
    [[nodiscard]] std::uint64_t metadata_bytes() const noexcept { return metadata_bytes_; }

    // ---- metadata -------------------------------------------------------------------
    [[nodiscard]] const std::vector<std::pair<std::string, GgufValue>>& kvs() const noexcept { return kvs_; }
    [[nodiscard]] const GgufValue* find(std::string_view key) const noexcept;

    // Typed getters: nullopt/nullptr if absent; Error(Model) if present with a type that
    // cannot represent the request (e.g. a string where an integer is expected, a negative
    // value for get_uint, an out-of-range value for get_u32).
    [[nodiscard]] std::optional<std::uint64_t> get_uint(std::string_view key) const;
    [[nodiscard]] std::optional<std::uint32_t> get_u32(std::string_view key) const;
    [[nodiscard]] std::optional<std::int64_t> get_int(std::string_view key) const;
    [[nodiscard]] std::optional<double> get_float(std::string_view key) const;  // F32/F64 only
    [[nodiscard]] std::optional<bool> get_bool(std::string_view key) const;
    [[nodiscard]] const std::string* get_string(std::string_view key) const;
    [[nodiscard]] const GgufArray* get_array(std::string_view key) const;

    // ---- tensors --------------------------------------------------------------------
    [[nodiscard]] const std::vector<GgufTensorInfo>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const GgufTensorInfo* find_tensor(std::string_view name) const noexcept;

    /// The tensor's bytes. Throws Error(Model) in HeaderOnly mode.
    [[nodiscard]] std::span<const std::byte> tensor_data(const GgufTensorInfo& t) const;

private:
    GgufFile() = default;
    void parse_all(const GgufOptions& options);

    std::string source_;
    GgufMode mode_ = GgufMode::Full;
    MappedFile map_;
    std::vector<std::byte> owned_;
    std::span<const std::byte> bytes_;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = 32;
    std::uint64_t header_size_ = 0;
    std::uint64_t data_offset_ = 0;
    std::uint64_t metadata_bytes_ = 0;
    std::vector<std::pair<std::string, GgufValue>> kvs_;
    std::map<std::string, std::size_t, std::less<>> kv_index_;
    std::vector<GgufTensorInfo> tensors_;
    std::map<std::string, std::size_t, std::less<>> tensor_index_;
};

}  // namespace halo::model
