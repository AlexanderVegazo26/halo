// GET_ROWS (embedding lookup, WS-F2 V2) against halo::tensor::dequantize_row on the SAME weight
// bytes (TRD §30; code review M-2). The kernels evaluate dequantize_row's expressions
// uncontracted, so every row must be BIT-IDENTICAL. Verified on lavapipe only (D-001); the
// fp16 scales are normal numbers on purpose (reference.h: random_normal_fp16).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/vulkan/ops.h"
#include "halo/core/error.h"
#include "reference.h"
#include "vk_test_util.h"

namespace hv = halo::vulkan;
namespace ref = halo::vulkan::ref;
using hv::test::download;
using hv::test::upload;

namespace {

constexpr float k_sentinel = -12345.5f;

const char* type_name(ref::WType t) {
    switch (t) {
        case ref::WType::F32: return "F32";
        case ref::WType::Q8_0: return "Q8_0";
        case ref::WType::Q4_K: return "Q4_K";
        case ref::WType::Q5_K: return "Q5_K";
        case ref::WType::Q6_K: return "Q6_K";
        case ref::WType::IQ4_XS: return "IQ4_XS";
    }
    return "?";
}

struct RowsFixture {
    std::vector<std::uint8_t> arena;  // `lead` bytes, then the weight rows, then padding
    std::uint64_t lead = 0;
    std::uint64_t row_bytes = 0;
    std::vector<float> deq;  // tensor::dequantize_row of every row [rows * cols]
};

/// Weights of `rows` x `cols` placed at byte `lead` of an arena (any byte for quantized types,
/// a multiple of 4 for F32), so the kernel's sub-alignment remainder is exercised.
RowsFixture make_rows(ref::WType t, std::uint32_t rows, std::uint32_t cols, std::uint64_t lead, std::mt19937& rng) {
    RowsFixture f;
    const ref::Weights w = ref::random_weights(t, rows, cols, rng, 14);
    f.lead = lead;
    f.row_bytes = hv::matvec_row_bytes(ref::dtype(t), cols);
    const std::uint64_t payload = f.row_bytes * rows;
    f.arena.assign((lead + payload + 3 + 16) / 4 * 4, 0xA5);
    std::memcpy(f.arena.data() + lead, w.bytes.data(), payload);
    f.deq = w.deq;
    return f;
}

/// Runs get_rows over `ids`; returns the out rows (sentinel-initialised) and the status word.
std::pair<std::vector<float>, std::uint32_t> run_get_rows(const std::shared_ptr<hv::Context>& ctx, ref::WType t,
                                                          const RowsFixture& f, std::uint32_t rows,
                                                          std::uint32_t cols, const std::vector<std::int32_t>& ids) {
    hv::Ops ops(ctx);
    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(f.arena));
    // ids at byte 4, out at byte 8, status at byte 12: leads that are not multiples of the
    // device alignment, so the kernel's ids / out / status offset terms are load-bearing.
    std::vector<std::int32_t> ids_buf(1 + ids.size(), -7);
    std::copy(ids.begin(), ids.end(), ids_buf.begin() + 1);
    hv::Buffer bi = upload(ctx, std::span<const std::int32_t>(ids_buf));
    const std::vector<float> init(2 + ids.size() * cols, k_sentinel);
    hv::Buffer bo = upload(ctx, std::span<const float>(init), hv::MemoryUsage::HostCached);
    const std::vector<std::uint32_t> st_init{7u, 7u, 7u, 0xDEADBEEFu};  // the op must zero word 3
    hv::Buffer bs = upload(ctx, std::span<const std::uint32_t>(st_init), hv::MemoryUsage::HostCached);
    hv::GetRowsArgs a;
    a.wtype = ref::dtype(t);
    a.w = hv::BufferView(bw, f.lead, f.row_bytes * rows);
    a.n_rows = rows;
    a.cols = cols;
    a.ids = hv::BufferView(bi, 4, ids.size() * 4);
    a.n_ids = static_cast<std::uint32_t>(ids.size());
    a.out = hv::BufferView(bo, 8, ids.size() * cols * 4);
    a.status = hv::BufferView(bs, 12, 4);
    hv::Stream s(ctx);
    ops.get_rows(s, a);
    s.submit_and_wait();
    std::vector<float> out(ids.size() * cols);
    bo.download(std::span<float>(out), 8);
    EXPECT_EQ(hv::read_status(bs, 0), 7u) << "bytes before the status view were touched";
    return {out, hv::read_status(bs, 12)};
}

bool row_bitwise(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * 4) == 0; }

}  // namespace

TEST(VkRows, GetRowsIsBitIdenticalToDequantizeRowForEveryType) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    struct Case {
        ref::WType t;
        std::uint32_t cols;
        std::uint64_t lead;
    };
    const Case cases[] = {{ref::WType::F32, 77, 12},    {ref::WType::Q8_0, 160, 5},  {ref::WType::Q4_K, 512, 7},
                          {ref::WType::Q5_K, 512, 3},   {ref::WType::Q6_K, 768, 1},  {ref::WType::IQ4_XS, 512, 9}};
    const std::uint32_t rows = 37;
    const std::vector<std::int32_t> ids{0, 36, 5, 5, 17, 36, 1, 2, 30};
    for (const Case& c : cases) {
        SCOPED_TRACE(type_name(c.t));
        std::mt19937 rng(100 + static_cast<unsigned>(c.t));
        const RowsFixture f = make_rows(c.t, rows, c.cols, c.lead, rng);
        const auto [out, status] = run_get_rows(ctx, c.t, f, rows, c.cols, ids);
        EXPECT_EQ(status, 0u);
        std::size_t bad_rows = 0;
        for (std::size_t r = 0; r < ids.size(); ++r) {
            const float* want = f.deq.data() + static_cast<std::size_t>(ids[r]) * c.cols;
            if (!row_bitwise(out.data() + r * c.cols, want, c.cols)) {
                ++bad_rows;
                ADD_FAILURE() << "row " << r << " (id " << ids[r] << ") differs from tensor::dequantize_row";
            }
        }
        std::cout << "[vk-rows] " << type_name(c.t) << " cols " << c.cols << " lead " << c.lead << ": " << bad_rows
                  << " of " << ids.size() << " rows differ from tensor::dequantize_row\n";
    }
}

TEST(VkRows, GetRowsTokenEmbeddingShapeQ4K) {
    // token_embd of the 27B pack: Q4_K, 5120 columns (fewer rows than the 248320 vocabulary).
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint32_t rows = 1000, cols = 5120;
    std::mt19937 rng(7);
    const RowsFixture f = make_rows(ref::WType::Q4_K, rows, cols, 0, rng);
    const std::vector<std::int32_t> ids{999, 0, 512, 999};
    const auto [out, status] = run_get_rows(ctx, ref::WType::Q4_K, f, rows, cols, ids);
    EXPECT_EQ(status, 0u);
    for (std::size_t r = 0; r < ids.size(); ++r) {
        EXPECT_TRUE(row_bitwise(out.data() + r * cols, f.deq.data() + static_cast<std::size_t>(ids[r]) * cols, cols))
            << "row " << r;
    }
}

TEST(VkRows, GetRowsBadIdSetsStatusAndLeavesOnlyThatRowUnwritten) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::uint32_t rows = 37, cols = 256;
    std::mt19937 rng(8);
    const RowsFixture f = make_rows(ref::WType::Q6_K, rows, cols, 2, rng);
    const std::vector<std::int32_t> ids{3, -1, 37, 36, 2147483647, 4};
    const auto [out, status] = run_get_rows(ctx, ref::WType::Q6_K, f, rows, cols, ids);
    EXPECT_EQ(status, hv::k_status_bad_index);
    for (std::size_t r = 0; r < ids.size(); ++r) {
        const bool bad = ids[r] < 0 || ids[r] >= static_cast<std::int32_t>(rows);
        if (bad) {
            for (std::size_t c = 0; c < cols; ++c) ASSERT_EQ(out[r * cols + c], k_sentinel) << "bad row " << r << " written";
        } else {
            EXPECT_TRUE(row_bitwise(out.data() + r * cols, f.deq.data() + static_cast<std::size_t>(ids[r]) * cols, cols))
                << "good row " << r;
        }
    }
    EXPECT_THROW(hv::check_status(status, "get_rows"), halo::Error);
    EXPECT_NO_THROW(hv::check_status(0, "get_rows"));
    try {
        hv::check_status(status, "get_rows");
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Kernel);
        EXPECT_NE(std::string(e.what()).find("bad-index"), std::string::npos) << e.what();
    }
}

TEST(VkRows, GetRowsStatusWordIsZeroedByEachCall) {
    // One status word reused by two calls in one recording: the second (clean) call's fill
    // must clear the first call's bit.
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    const std::uint32_t rows = 4, cols = 32;
    std::mt19937 rng(9);
    const RowsFixture f = make_rows(ref::WType::Q8_0, rows, cols, 0, rng);
    hv::Buffer bw = upload(ctx, std::span<const std::uint8_t>(f.arena));
    const std::vector<std::int32_t> bad{9}, good{1};
    hv::Buffer bbad = upload(ctx, std::span<const std::int32_t>(bad));
    hv::Buffer bgood = upload(ctx, std::span<const std::int32_t>(good));
    hv::Buffer bo = hv::Buffer::create(ctx, cols * 4, hv::MemoryUsage::HostCached);
    hv::Buffer bs = hv::Buffer::create(ctx, 16, hv::MemoryUsage::HostCached);
    hv::GetRowsArgs a;
    a.wtype = halo::DType::Q8_0;
    a.w = hv::BufferView(bw, 0, f.row_bytes * rows);
    a.n_rows = rows;
    a.cols = cols;
    a.out = bo;
    a.status = hv::BufferView(bs, 8, 4);
    hv::Stream s(ctx);
    a.ids = bbad;
    ops.get_rows(s, a);
    s.submit_and_wait();
    EXPECT_EQ(hv::read_status(bs, 8), hv::k_status_bad_index);
    a.ids = bbad;
    ops.get_rows(s, a);
    a.ids = bgood;
    ops.get_rows(s, a);
    s.submit_and_wait();
    EXPECT_EQ(hv::read_status(bs, 8), 0u);
    EXPECT_TRUE(row_bitwise(download<float>(bo, cols).data(), f.deq.data() + cols, cols));
}

TEST(VkRows, GetRowsValidation) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Ops ops(ctx);
    hv::Buffer w = hv::Buffer::create(ctx, 4 * 34, hv::MemoryUsage::DeviceLocal);
    hv::Buffer ids = hv::Buffer::create(ctx, 2 * 4, hv::MemoryUsage::DeviceLocal);
    hv::Buffer o = hv::Buffer::create(ctx, 2 * 32 * 4 + 16, hv::MemoryUsage::DeviceLocal);
    hv::Buffer st = hv::Buffer::create(ctx, 4, hv::MemoryUsage::DeviceLocal);
    hv::Stream s(ctx);
    auto base = [&] {
        hv::GetRowsArgs a;
        a.wtype = halo::DType::Q8_0;
        a.w = w;
        a.n_rows = 4;
        a.cols = 32;
        a.ids = ids;
        a.n_ids = 2;
        a.out = hv::BufferView(o, 0, 2 * 32 * 4);
        a.status = st;
        return a;
    };
    EXPECT_NO_THROW(ops.get_rows(s, base()));
    auto a = base();
    a.wtype = halo::DType::Q4_0;
    try {
        ops.get_rows(s, a);
        ADD_FAILURE() << "Q4_0 accepted";
    } catch (const halo::Error& e) {
        EXPECT_EQ(e.code(), halo::ErrorCode::Unsupported);
    }
    a = base();
    a.cols = 48;  // not a multiple of the Q8_0 block
    EXPECT_THROW(ops.get_rows(s, a), halo::Error);
    a = base();
    a.n_rows = 5;  // w too small
    EXPECT_THROW(ops.get_rows(s, a), halo::Error);
    a = base();
    a.n_ids = 3;  // ids too small
    EXPECT_THROW(ops.get_rows(s, a), halo::Error);
    a = base();
    a.status = hv::BufferView(o, 2 * 32 * 4 - 4, 4);  // status overlaps out
    EXPECT_THROW(ops.get_rows(s, a), halo::Error);
    a = base();
    a.status = hv::BufferView(o, 2 * 32 * 4, 4);  // same buffer, disjoint: fine
    EXPECT_NO_THROW(ops.get_rows(s, a));
    a = base();
    a.status = hv::BufferView();  // no status
    EXPECT_THROW(ops.get_rows(s, a), halo::Error);
    s.submit_and_wait();
}
