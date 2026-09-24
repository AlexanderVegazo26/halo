// GGUF parser unit tests on synthetic in-memory files.

#include <gtest/gtest.h>

#include <limits>

#include "gguf_builder.h"
#include "halo/core/error.h"
#include "halo/model/gguf.h"

using halo::DType;
using halo::ErrorCode;
using halo::model::GgufFile;
using halo::model::GgufMode;
using halo::model::GgufType;
using namespace halo::test;

namespace {

GgufFile parse(const GgufSpec& s, GgufMode mode = GgufMode::Full, bool with_data = true) {
    return GgufFile::parse(s.build(with_data), mode, "test.gguf");
}

// Returns the error code and asserts the message contains `needle`.
template <typename F>
ErrorCode err(F&& f, std::string_view needle = {}) {
    try {
        f();
    } catch (const halo::Error& e) {
        if (!needle.empty()) {
            EXPECT_NE(std::string_view(e.what()).find(needle), std::string_view::npos) << e.what();
        }
        return e.code();
    }
    ADD_FAILURE() << "expected halo::Error";
    return ErrorCode::Cancelled;
}

GgufSpec basic() {
    GgufSpec s;
    s.kv("general.architecture", v_str("test")).kv("a.u32", v_u32(7));
    s.tensor("w", {64, 3}, DType::F32).tensor("q", {64, 2}, DType::Q8_0);
    return s;
}

}  // namespace

TEST(Gguf, ParsesAllScalarTypesArraysAndTensors) {
    GgufSpec s = basic();
    Bytes u8; u8.put<std::uint32_t>(0).put<std::uint8_t>(200);
    Bytes i8; i8.put<std::uint32_t>(1).put<std::int8_t>(-5);
    Bytes u16; u16.put<std::uint32_t>(2).put<std::uint16_t>(60000);
    Bytes i16; i16.put<std::uint32_t>(3).put<std::int16_t>(-30000);
    Bytes i64; i64.put<std::uint32_t>(11).put<std::int64_t>(-(1LL << 40));
    Bytes f64; f64.put<std::uint32_t>(12).put<double>(0.25);
    Bytes nested;  // array of 2 arrays of u16
    nested.put<std::uint32_t>(9).put<std::uint32_t>(9).put<std::uint64_t>(2);
    nested.put<std::uint32_t>(2).put<std::uint64_t>(2).put<std::uint16_t>(1).put<std::uint16_t>(2);
    nested.put<std::uint32_t>(2).put<std::uint64_t>(1).put<std::uint16_t>(3);
    s.kv("t.u8", u8).kv("t.i8", i8).kv("t.u16", u16).kv("t.i16", i16).kv("t.i64", i64).kv("t.f64", f64)
        .kv("t.u64", v_u64(1ULL << 50)).kv("t.f32", v_f32(1.5f)).kv("t.bool", v_bool(true))
        .kv("t.str", v_str("héllo")).kv("t.nested", nested).kv("t.strs", v_arr_str({"a", "", "ccc"}));
    const GgufFile f = parse(s);
    EXPECT_EQ(f.version(), 3u);
    EXPECT_EQ(f.get_uint("t.u8"), 200u);
    EXPECT_EQ(f.get_int("t.i8"), -5);
    EXPECT_EQ(f.get_uint("t.u16"), 60000u);
    EXPECT_EQ(f.get_int("t.i16"), -30000);
    EXPECT_EQ(f.get_int("t.i64"), -(1LL << 40));
    EXPECT_EQ(f.get_uint("t.u64"), 1ULL << 50);
    EXPECT_EQ(f.get_float("t.f64"), 0.25);
    EXPECT_EQ(f.get_float("t.f32"), 1.5);
    EXPECT_EQ(f.get_bool("t.bool"), true);
    EXPECT_EQ(*f.get_string("t.str"), "héllo");
    const auto* n = f.get_array("t.nested");
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->elem_type, GgufType::Array);
    ASSERT_EQ(n->size(), 2u);
    EXPECT_EQ(n->arrays[0].uints, (std::vector<std::uint64_t>{1, 2}));
    EXPECT_EQ(n->arrays[1].uints, (std::vector<std::uint64_t>{3}));
    EXPECT_EQ(f.get_array("t.strs")->strings, (std::vector<std::string>{"a", "", "ccc"}));
    EXPECT_FALSE(f.get_uint("absent").has_value());
    EXPECT_EQ(err([&] { (void)f.get_uint("t.str"); }), ErrorCode::Model);
    EXPECT_EQ(err([&] { (void)f.get_uint("t.i8"); }), ErrorCode::Model);  // negative
    EXPECT_EQ(err([&] { (void)f.get_u32("t.u64"); }), ErrorCode::Model);  // > 32 bits
    EXPECT_EQ(err([&] { (void)f.get_float("t.u8"); }), ErrorCode::Model);

    ASSERT_EQ(f.tensors().size(), 2u);
    const auto* w = f.find_tensor("w");
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->n_elements, 192u);
    EXPECT_EQ(w->n_bytes, 768u);
    EXPECT_EQ(w->offset, 0u);
    const auto* q = f.find_tensor("q");
    EXPECT_EQ(q->n_bytes, 2u * 2u * 34u);
    EXPECT_EQ(q->offset, 768u);
    EXPECT_EQ(f.data_offset() % 32, 0u);
    EXPECT_EQ(f.tensor_data(*w).size(), 768u);
}

TEST(Gguf, CustomAlignment) {
    GgufSpec s = basic();
    s.alignment = 256;
    s.kv("general.alignment", v_u32(256));
    const GgufFile f = parse(s);
    EXPECT_EQ(f.alignment(), 256u);
    EXPECT_EQ(f.data_offset() % 256, 0u);
    EXPECT_EQ(f.find_tensor("q")->offset, 768u);  // 768 is a multiple of 256
    s.kv("general.alignment", v_u32(48));
    EXPECT_EQ(err([&] { (void)parse(s); }, "alignment"), ErrorCode::Model);
    s.kv("general.alignment", v_u64(32));  // wrong type
    EXPECT_EQ(err([&] { (void)parse(s); }, "alignment"), ErrorCode::Model);
}

TEST(Gguf, HeaderOnlyModeAcceptsTruncatedFileButRefusesData) {
    const GgufSpec s = basic();
    const auto header = s.build(false);
    EXPECT_EQ(err([&] { (void)GgufFile::parse(header, GgufMode::Full, "h"); }, "past end of file"), ErrorCode::Model);
    const GgufFile f = GgufFile::parse(header, GgufMode::HeaderOnly, "h");
    EXPECT_FALSE(f.has_tensor_data());
    EXPECT_EQ(f.header_size(), header.size());
    EXPECT_GE(f.data_offset(), f.header_size());
    EXPECT_EQ(err([&] { (void)f.tensor_data(*f.find_tensor("w")); }, "header-only"), ErrorCode::Model);
    // Full file also parses fine in header-only mode.
    EXPECT_NO_THROW((void)parse(s, GgufMode::HeaderOnly));
}

TEST(Gguf, EveryTruncationOfAValidHeaderIsATypedError) {
    const auto header = basic().build(false);
    for (std::size_t n = 0; n < header.size(); ++n) {
        std::vector<std::byte> cut(header.begin(), header.begin() + static_cast<std::ptrdiff_t>(n));
        EXPECT_EQ(err([&] { (void)GgufFile::parse(cut, GgufMode::HeaderOnly, "cut"); }), ErrorCode::Model) << n;
    }
}

TEST(Gguf, RejectsBadMagicAndVersions) {
    auto img = basic().build();
    auto bad = img;
    bad[0] = std::byte{'X'};
    EXPECT_EQ(err([&] { (void)GgufFile::parse(bad, GgufMode::Full); }, "magic"), ErrorCode::Model);
    GgufSpec s = basic();
    s.version = 1;
    EXPECT_EQ(err([&] { (void)parse(s); }, "v1"), ErrorCode::Unsupported);
    s.version = 0x03000000;  // byte-swapped 3
    EXPECT_EQ(err([&] { (void)parse(s); }, "big-endian"), ErrorCode::Unsupported);
    s.version = 7;
    EXPECT_EQ(err([&] { (void)parse(s); }, "version"), ErrorCode::Model);
    s.version = 2;
    EXPECT_NO_THROW((void)parse(s));
}

TEST(Gguf, RejectsMalformedMetadata) {
    {
        GgufSpec s = basic();
        s.kvs.push_back({"a.u32", v_u32(1)});  // duplicate
        EXPECT_EQ(err([&] { (void)parse(s); }, "duplicate metadata key"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(13).put<std::uint32_t>(0);
        s.kv("bad.type", b);
        EXPECT_EQ(err([&] { (void)parse(s); }, "invalid GGUF value type"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(7).put<std::uint8_t>(2);
        s.kv("bad.bool", b);
        EXPECT_EQ(err([&] { (void)parse(s); }, "invalid bool"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(8).put<std::uint64_t>(1ULL << 62);  // absurd string length
        s.kv("bad.str", b);
        EXPECT_EQ(err([&] { (void)parse(s); }, "exceeds limit"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(8).put<std::uint64_t>(1000);  // within limit, past EOF
        s.kv("bad.str", b);
        EXPECT_EQ(err([&] { (void)parse(s, GgufMode::HeaderOnly, false); }, "truncated"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(9).put<std::uint32_t>(10).put<std::uint64_t>(1ULL << 60);  // huge u64 array
        s.kv("bad.arr", b);
        EXPECT_EQ(err([&] { (void)parse(s); }, "count"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(9).put<std::uint32_t>(10).put<std::uint64_t>(1000);  // fits limit, not bytes
        s.kv("bad.arr", b);
        EXPECT_EQ(err([&] { (void)parse(s, GgufMode::HeaderOnly, false); }, "cannot fit"), ErrorCode::Model);
    }
    {
        GgufSpec s = basic();
        Bytes b; b.put<std::uint32_t>(9);
        for (int d = 0; d < 10; ++d) b.put<std::uint32_t>(9).put<std::uint64_t>(1);
        b.put<std::uint32_t>(0).put<std::uint64_t>(0);
        s.kv("deep", b);
        EXPECT_EQ(err([&] { (void)parse(s); }, "nesting"), ErrorCode::Model);
    }
    {
        // n_kv larger than the file could hold: rejected before any allocation.
        auto img = basic().build();
        const std::uint64_t huge = 1ULL << 40;
        std::memcpy(img.data() + 16, &huge, 8);
        EXPECT_EQ(err([&] { (void)GgufFile::parse(img, GgufMode::Full); }, "kv count"), ErrorCode::Model);
    }
}

TEST(Gguf, RejectsMalformedTensorInfos) {
    auto expect = [](GgufSpec s, ErrorCode code, std::string_view needle, GgufMode mode = GgufMode::Full) {
        EXPECT_EQ(err([&] { (void)parse(s, mode); }, needle), code) << needle;
    };
    {
        GgufSpec s = basic();
        s.find("w")->n_dims = 0;
        s.find("w")->ne.clear();
        expect(s, ErrorCode::Model, "dims");
    }
    {
        GgufSpec s = basic();
        s.tensor("five", {32, 1, 1, 1, 1});
        expect(s, ErrorCode::Model, "dims");
    }
    {
        GgufSpec s = basic();
        s.tensor("huge", {1LL << 40, 1LL << 40});
        expect(s, ErrorCode::Model, "overflows", GgufMode::HeaderOnly);
    }
    {
        GgufSpec s = basic();
        s.tensor("neg", {-1});
        expect(s, ErrorCode::Model, "overflows", GgufMode::HeaderOnly);
    }
    {
        GgufSpec s = basic();
        s.find("q")->type = 4;  // removed Q4_2: never valid
        expect(s, ErrorCode::Model, "invalid type id");
    }
    {
        GgufSpec s = basic();
        s.find("q")->type = 39;  // MXFP4: real ggml type HALO lacks
        expect(s, ErrorCode::Unsupported, "MXFP4");
    }
    {
        GgufSpec s = basic();
        s.find("q")->ne = {48, 2};  // not a multiple of 32
        expect(s, ErrorCode::Model, "multiple of the Q8_0 block size");
    }
    {
        GgufSpec s = basic();
        s.find("q")->offset = 16;  // misaligned
        expect(s, ErrorCode::Model, "not a multiple of the alignment");
    }
    {
        GgufSpec s = basic();
        s.find("q")->offset = 736;  // aligned but overlaps w [0, 768)
        expect(s, ErrorCode::Model, "overlaps");
    }
    {
        GgufSpec s = basic();
        s.tensor("w", {32});
        expect(s, ErrorCode::Model, "duplicate tensor name");
    }
    {
        GgufSpec s = basic();
        s.find("w")->name = std::string(64, 'n');  // ggml GGML_MAX_NAME is 64 incl. NUL
        expect(s, ErrorCode::Model, "exceeds limit");
    }
    {
        GgufSpec s = basic();
        s.find("q")->offset = 1ULL << 62;  // aligned, far past EOF
        expect(s, ErrorCode::Model, "past end of file");
        EXPECT_NO_THROW((void)parse(s, GgufMode::HeaderOnly));  // cannot be checked header-only
    }
    {
        GgufSpec s = basic();
        s.find("q")->offset = std::numeric_limits<std::uint64_t>::max() - 31;  // offset + size overflows
        expect(s, ErrorCode::Model, "overflows", GgufMode::HeaderOnly);
    }
}

TEST(Gguf, OpenReportsIoErrors) {
    EXPECT_EQ(err([] { (void)GgufFile::open("/nonexistent/halo.gguf"); }, "cannot open"), ErrorCode::Io);
    EXPECT_EQ(err([] { (void)GgufFile::open("/"); }), ErrorCode::Io);
}
