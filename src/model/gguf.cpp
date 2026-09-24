#include "halo/model/gguf.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#include "halo/core/error.h"
#include "halo/tensor/quant.h"

namespace halo::model {
namespace {

constexpr std::uint64_t kI64Max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool valid_type(std::uint32_t t) noexcept { return t <= static_cast<std::uint32_t>(GgufType::F64); }

// Minimum encoded size of one value of type t (used to bound counts before allocating).
std::uint64_t min_encoded_size(GgufType t) noexcept {
    switch (t) {
        case GgufType::U8:
        case GgufType::I8:
        case GgufType::Bool: return 1;
        case GgufType::U16:
        case GgufType::I16: return 2;
        case GgufType::U32:
        case GgufType::I32:
        case GgufType::F32: return 4;
        case GgufType::U64:
        case GgufType::I64:
        case GgufType::F64:
        case GgufType::String: return 8;   // u64 length prefix
        case GgufType::Array: return 12;   // u32 type + u64 count
    }
    return 1;
}

bool is_unsigned(GgufType t) noexcept {
    return t == GgufType::U8 || t == GgufType::U16 || t == GgufType::U32 || t == GgufType::U64;
}
bool is_signed(GgufType t) noexcept {
    return t == GgufType::I8 || t == GgufType::I16 || t == GgufType::I32 || t == GgufType::I64;
}

class Reader {
public:
    Reader(std::span<const std::byte> b, const std::string& src) : b_(b), src_(src) {}

    [[nodiscard]] std::uint64_t pos() const noexcept { return pos_; }
    [[nodiscard]] std::uint64_t remaining() const noexcept { return b_.size() - pos_; }

    void need(std::uint64_t n, std::string_view what) const {
        if (n > remaining()) {
            throw_error(ErrorCode::Model, "{}: truncated at offset {} reading {} ({} bytes needed, {} left)", src_,
                        pos_, what, n, remaining());
        }
    }

    template <typename T>
    T scalar(std::string_view what) {
        need(sizeof(T), what);
        T v{};
        std::memcpy(&v, b_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string string(std::string_view what, std::uint64_t max_len) {
        const auto len = scalar<std::uint64_t>(what);
        if (len > max_len) {
            throw_error(ErrorCode::Model, "{}: {} length {} at offset {} exceeds limit {}", src_, what, len,
                        pos_ - 8, max_len);
        }
        need(len, what);
        std::string s(reinterpret_cast<const char*>(b_.data() + pos_), static_cast<std::size_t>(len));
        pos_ += len;
        return s;
    }

    // Validates that `count` elements of at least `min_size` bytes each can still fit.
    void check_count(std::uint64_t count, std::uint64_t min_size, std::uint64_t cap, std::string_view what) const {
        if (count > cap) {
            throw_error(ErrorCode::Model, "{}: {} count {} exceeds limit {}", src_, what, count, cap);
        }
        if (min_size != 0 && count > remaining() / min_size) {
            throw_error(ErrorCode::Model, "{}: {} count {} cannot fit in the {} remaining bytes", src_, what, count,
                        remaining());
        }
    }

    [[nodiscard]] const std::string& src() const noexcept { return src_; }

private:
    std::span<const std::byte> b_;
    const std::string& src_;
    std::uint64_t pos_ = 0;
};

GgufType read_type(Reader& r, std::string_view what) {
    const auto t = r.scalar<std::uint32_t>(what);
    if (!valid_type(t)) {
        throw_error(ErrorCode::Model, "{}: invalid GGUF value type {} for {} at offset {}", r.src(), t, what,
                    r.pos() - 4);
    }
    return static_cast<GgufType>(t);
}

bool read_bool(Reader& r) {
    const auto v = r.scalar<std::uint8_t>("bool");
    if (v > 1) {
        throw_error(ErrorCode::Model, "{}: invalid bool value {} at offset {}", r.src(), v, r.pos() - 1);
    }
    return v != 0;
}

void read_array(Reader& r, GgufArray& a, int depth);

void read_scalar_into(Reader& r, GgufType t, GgufValue& v) {
    switch (t) {
        case GgufType::U8: v.u = r.scalar<std::uint8_t>("u8"); break;
        case GgufType::I8: v.i = r.scalar<std::int8_t>("i8"); break;
        case GgufType::U16: v.u = r.scalar<std::uint16_t>("u16"); break;
        case GgufType::I16: v.i = r.scalar<std::int16_t>("i16"); break;
        case GgufType::U32: v.u = r.scalar<std::uint32_t>("u32"); break;
        case GgufType::I32: v.i = r.scalar<std::int32_t>("i32"); break;
        case GgufType::F32: v.f = static_cast<double>(r.scalar<float>("f32")); break;
        case GgufType::Bool: v.b = read_bool(r); break;
        case GgufType::String: v.s = r.string("string value", GgufLimits::kMaxStringBytes); break;
        case GgufType::U64: v.u = r.scalar<std::uint64_t>("u64"); break;
        case GgufType::I64: v.i = r.scalar<std::int64_t>("i64"); break;
        case GgufType::F64: v.f = r.scalar<double>("f64"); break;
        case GgufType::Array: read_array(r, v.arr, 1); break;
    }
}

void read_array(Reader& r, GgufArray& a, int depth) {
    if (depth > GgufLimits::kMaxArrayDepth) {
        throw_error(ErrorCode::Model, "{}: array nesting deeper than {} at offset {}", r.src(),
                    GgufLimits::kMaxArrayDepth, r.pos());
    }
    a.elem_type = read_type(r, "array element type");
    const auto n = r.scalar<std::uint64_t>("array length");
    r.check_count(n, min_encoded_size(a.elem_type), GgufLimits::kMaxArrayElems, "array element");
    const auto count = static_cast<std::size_t>(n);
    switch (a.elem_type) {
        case GgufType::U8:
            a.uints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.uints.push_back(r.scalar<std::uint8_t>("u8"));
            break;
        case GgufType::U16:
            a.uints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.uints.push_back(r.scalar<std::uint16_t>("u16"));
            break;
        case GgufType::U32:
            a.uints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.uints.push_back(r.scalar<std::uint32_t>("u32"));
            break;
        case GgufType::U64:
            a.uints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.uints.push_back(r.scalar<std::uint64_t>("u64"));
            break;
        case GgufType::I8:
            a.ints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.ints.push_back(r.scalar<std::int8_t>("i8"));
            break;
        case GgufType::I16:
            a.ints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.ints.push_back(r.scalar<std::int16_t>("i16"));
            break;
        case GgufType::I32:
            a.ints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.ints.push_back(r.scalar<std::int32_t>("i32"));
            break;
        case GgufType::I64:
            a.ints.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.ints.push_back(r.scalar<std::int64_t>("i64"));
            break;
        case GgufType::F32:
            a.floats.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.floats.push_back(static_cast<double>(r.scalar<float>("f32")));
            break;
        case GgufType::F64:
            a.floats.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.floats.push_back(r.scalar<double>("f64"));
            break;
        case GgufType::Bool:
            a.bools.reserve(count);
            for (std::size_t k = 0; k < count; ++k) a.bools.push_back(read_bool(r) ? 1 : 0);
            break;
        case GgufType::String:
            a.strings.reserve(count);
            for (std::size_t k = 0; k < count; ++k) {
                a.strings.push_back(r.string("array string", GgufLimits::kMaxStringBytes));
            }
            break;
        case GgufType::Array:
            a.arrays.reserve(count);
            for (std::size_t k = 0; k < count; ++k) {
                a.arrays.emplace_back();
                read_array(r, a.arrays.back(), depth + 1);
            }
            break;
    }
}

std::uint64_t align_up(std::uint64_t x, std::uint64_t a, const std::string& src) {
    std::uint64_t t = 0;
    if (!checked_add(x, a - 1, t)) throw_error(ErrorCode::Model, "{}: offset overflow", src);
    return t / a * a;
}

}  // namespace

std::string_view to_string(GgufType t) noexcept {
    switch (t) {
        case GgufType::U8: return "UINT8";
        case GgufType::I8: return "INT8";
        case GgufType::U16: return "UINT16";
        case GgufType::I16: return "INT16";
        case GgufType::U32: return "UINT32";
        case GgufType::I32: return "INT32";
        case GgufType::F32: return "FLOAT32";
        case GgufType::Bool: return "BOOL";
        case GgufType::String: return "STRING";
        case GgufType::Array: return "ARRAY";
        case GgufType::U64: return "UINT64";
        case GgufType::I64: return "INT64";
        case GgufType::F64: return "FLOAT64";
    }
    return "UNKNOWN";
}

std::size_t GgufArray::size() const noexcept {
    switch (elem_type) {
        case GgufType::U8:
        case GgufType::U16:
        case GgufType::U32:
        case GgufType::U64: return uints.size();
        case GgufType::I8:
        case GgufType::I16:
        case GgufType::I32:
        case GgufType::I64: return ints.size();
        case GgufType::F32:
        case GgufType::F64: return floats.size();
        case GgufType::Bool: return bools.size();
        case GgufType::String: return strings.size();
        case GgufType::Array: return arrays.size();
    }
    return 0;
}

GgufFile GgufFile::open(const std::filesystem::path& path, GgufMode mode) {
    GgufFile f;
    f.source_ = path.string();
    f.mode_ = mode;
    f.map_ = MappedFile::open(path);
    f.bytes_ = f.map_.bytes();
    f.parse_all();
    return f;
}

GgufFile GgufFile::parse(std::vector<std::byte> bytes, GgufMode mode, std::string source_name) {
    GgufFile f;
    f.source_ = std::move(source_name);
    f.mode_ = mode;
    f.owned_ = std::move(bytes);
    f.bytes_ = f.owned_;
    f.parse_all();
    return f;
}

void GgufFile::parse_all() {
    Reader r(bytes_, source_);
    const auto& src = source_;

    // ---- header -------------------------------------------------------------------------
    r.need(4, "magic");
    const auto magic = r.scalar<std::uint32_t>("magic");
    if (magic != 0x46554747u) {  // "GGUF" little-endian
        throw_error(ErrorCode::Model, "{}: not a GGUF file (bad magic)", src);
    }
    version_ = r.scalar<std::uint32_t>("version");
    if (version_ != 2 && version_ != 3) {
        if (version_ == 1) throw_error(ErrorCode::Unsupported, "{}: GGUF v1 is not supported", src);
        if (std::byteswap(version_) == 2 || std::byteswap(version_) == 3) {
            throw_error(ErrorCode::Unsupported, "{}: big-endian GGUF is not supported", src);
        }
        throw_error(ErrorCode::Model, "{}: unknown GGUF version {}", src, version_);
    }
    const auto n_tensors = r.scalar<std::uint64_t>("tensor count");
    const auto n_kv = r.scalar<std::uint64_t>("kv count");
    r.check_count(n_kv, 8 + 4 + 1, GgufLimits::kMaxKv, "kv");

    // ---- metadata -----------------------------------------------------------------------
    kvs_.reserve(static_cast<std::size_t>(n_kv));
    for (std::uint64_t k = 0; k < n_kv; ++k) {
        std::string key = r.string("kv key", GgufLimits::kMaxStringBytes);
        GgufValue v;
        v.type = read_type(r, "kv value type");
        read_scalar_into(r, v.type, v);
        if (kv_index_.contains(key)) {
            throw_error(ErrorCode::Model, "{}: duplicate metadata key '{}'", src, key);
        }
        kv_index_.emplace(key, kvs_.size());
        kvs_.emplace_back(std::move(key), std::move(v));
    }

    if (const GgufValue* al = find("general.alignment")) {
        if (al->type != GgufType::U32) {
            throw_error(ErrorCode::Model, "{}: general.alignment has type {}, expected UINT32", src,
                        to_string(al->type));
        }
        if (al->u == 0 || !std::has_single_bit(al->u) || al->u > GgufLimits::kMaxAlignment) {
            throw_error(ErrorCode::Model, "{}: invalid general.alignment {}", src, al->u);
        }
        alignment_ = al->u;
    }

    // ---- tensor infos -------------------------------------------------------------------
    r.check_count(n_tensors, 8 + 4 + 8 + 4 + 8, GgufLimits::kMaxTensors, "tensor info");
    tensors_.reserve(static_cast<std::size_t>(n_tensors));
    for (std::uint64_t k = 0; k < n_tensors; ++k) {
        GgufTensorInfo t;
        t.name = r.string("tensor name", GgufLimits::kMaxTensorNameBytes);
        t.n_dims = r.scalar<std::uint32_t>("tensor n_dims");
        if (t.n_dims < 1 || t.n_dims > 4) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' has {} dims (expected 1..4)", src, t.name, t.n_dims);
        }
        std::uint64_t n_el = 1;
        for (std::uint32_t d = 0; d < t.n_dims; ++d) {
            const auto ne = r.scalar<std::uint64_t>("tensor dim");
            if (ne > kI64Max || !checked_mul(n_el, ne, n_el) || n_el > kI64Max) {
                throw_error(ErrorCode::Model, "{}: tensor '{}' element count overflows", src, t.name);
            }
            t.ne[d] = static_cast<std::int64_t>(ne);
        }
        t.n_elements = n_el;
        const auto type_id = r.scalar<std::uint32_t>("tensor type");
        const auto dt = dtype_from_id(type_id);
        if (!dt) {
            const auto name = ggml_type_name(type_id);
            if (!name.empty()) {
                throw_error(ErrorCode::Unsupported, "{}: tensor '{}' uses ggml type {} (id {}), not supported by HALO",
                            src, t.name, name, type_id);
            }
            throw_error(ErrorCode::Model, "{}: tensor '{}' has invalid type id {}", src, t.name, type_id);
        }
        t.type = *dt;
        const auto& tr = traits(t.type);
        if (static_cast<std::uint64_t>(t.ne[0]) % tr.block_elems != 0) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' ne[0]={} is not a multiple of the {} block size {}", src,
                        t.name, t.ne[0], tr.name, tr.block_elems);
        }
        const std::uint64_t row = tensor::row_bytes(t.type, t.ne[0]);
        if (!checked_mul(row, static_cast<std::uint64_t>(t.rows()), t.n_bytes) || t.n_bytes > kI64Max) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' byte size overflows", src, t.name);
        }
        t.offset = r.scalar<std::uint64_t>("tensor offset");
        if (t.offset % alignment_ != 0) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' offset {} is not a multiple of the alignment {}", src,
                        t.name, t.offset, alignment_);
        }
        std::uint64_t end = 0;
        if (!checked_add(t.offset, t.n_bytes, end) || end > kI64Max) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' data range overflows", src, t.name);
        }
        if (tensor_index_.contains(t.name)) {
            throw_error(ErrorCode::Model, "{}: duplicate tensor name '{}'", src, t.name);
        }
        tensor_index_.emplace(t.name, tensors_.size());
        tensors_.push_back(std::move(t));
    }
    header_size_ = r.pos();
    data_offset_ = align_up(header_size_, alignment_, src);

    // ---- data ranges: no overlap; inside the file (Full mode) ---------------------------
    std::vector<std::size_t> order(tensors_.size());
    for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return tensors_[a].offset < tensors_[b].offset;
    });
    for (std::size_t k = 1; k < order.size(); ++k) {
        const auto& prev = tensors_[order[k - 1]];
        const auto& cur = tensors_[order[k]];
        if (prev.offset + prev.n_bytes > cur.offset) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' [{}, {}) overlaps '{}' starting at {}", src, prev.name,
                        prev.offset, prev.offset + prev.n_bytes, cur.name, cur.offset);
        }
    }
    if (mode_ == GgufMode::Full) {
        const std::uint64_t size = bytes_.size();
        if (data_offset_ > size && !tensors_.empty()) {
            throw_error(ErrorCode::Model, "{}: data section starts at {}, past end of file ({} bytes)", src,
                        data_offset_, size);
        }
        for (const auto& t : tensors_) {
            std::uint64_t end = 0;
            if (!checked_add(data_offset_, t.offset + t.n_bytes, end) || end > size) {
                throw_error(ErrorCode::Model, "{}: tensor '{}' data [{}, {}) extends past end of file ({} bytes)",
                            src, t.name, data_offset_ + t.offset, data_offset_ + t.offset + t.n_bytes, size);
            }
        }
    }
}

const GgufValue* GgufFile::find(std::string_view key) const noexcept {
    const auto it = kv_index_.find(key);
    return it == kv_index_.end() ? nullptr : &kvs_[it->second].second;
}

std::optional<std::uint64_t> GgufFile::get_uint(std::string_view key) const {
    const GgufValue* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (is_unsigned(v->type)) return v->u;
    if (is_signed(v->type)) {
        if (v->i < 0) throw_error(ErrorCode::Model, "{}: key {} = {} must be non-negative", source_, key, v->i);
        return static_cast<std::uint64_t>(v->i);
    }
    throw_error(ErrorCode::Model, "{}: key {} has type {}, expected an integer", source_, key, to_string(v->type));
}

std::optional<std::uint32_t> GgufFile::get_u32(std::string_view key) const {
    const auto v = get_uint(key);
    if (!v) return std::nullopt;
    if (*v > std::numeric_limits<std::uint32_t>::max()) {
        throw_error(ErrorCode::Model, "{}: key {} = {} does not fit in 32 bits", source_, key, *v);
    }
    return static_cast<std::uint32_t>(*v);
}

std::optional<std::int64_t> GgufFile::get_int(std::string_view key) const {
    const GgufValue* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (is_signed(v->type)) return v->i;
    if (is_unsigned(v->type)) {
        if (v->u > kI64Max) throw_error(ErrorCode::Model, "{}: key {} = {} overflows int64", source_, key, v->u);
        return static_cast<std::int64_t>(v->u);
    }
    throw_error(ErrorCode::Model, "{}: key {} has type {}, expected an integer", source_, key, to_string(v->type));
}

std::optional<double> GgufFile::get_float(std::string_view key) const {
    const GgufValue* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (v->type == GgufType::F32 || v->type == GgufType::F64) return v->f;
    throw_error(ErrorCode::Model, "{}: key {} has type {}, expected a float", source_, key, to_string(v->type));
}

std::optional<bool> GgufFile::get_bool(std::string_view key) const {
    const GgufValue* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (v->type == GgufType::Bool) return v->b;
    throw_error(ErrorCode::Model, "{}: key {} has type {}, expected BOOL", source_, key, to_string(v->type));
}

const std::string* GgufFile::get_string(std::string_view key) const {
    const GgufValue* v = find(key);
    if (v == nullptr) return nullptr;
    if (v->type == GgufType::String) return &v->s;
    throw_error(ErrorCode::Model, "{}: key {} has type {}, expected STRING", source_, key, to_string(v->type));
}

const GgufArray* GgufFile::get_array(std::string_view key) const {
    const GgufValue* v = find(key);
    if (v == nullptr) return nullptr;
    if (v->type == GgufType::Array) return &v->arr;
    throw_error(ErrorCode::Model, "{}: key {} has type {}, expected ARRAY", source_, key, to_string(v->type));
}

const GgufTensorInfo* GgufFile::find_tensor(std::string_view name) const noexcept {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

std::span<const std::byte> GgufFile::tensor_data(const GgufTensorInfo& t) const {
    if (mode_ != GgufMode::Full) {
        throw_error(ErrorCode::Model, "{}: tensor data for '{}' is unavailable (file opened header-only)", source_,
                    t.name);
    }
    // Ranges were validated at parse time; re-check that `t` belongs to this file.
    const GgufTensorInfo* own = find_tensor(t.name);
    HALO_CHECK(own == &t, ErrorCode::Api, "tensor '{}' does not belong to {}", t.name, source_);
    return bytes_.subspan(static_cast<std::size_t>(data_offset_ + t.offset), static_cast<std::size_t>(t.n_bytes));
}

}  // namespace halo::model
