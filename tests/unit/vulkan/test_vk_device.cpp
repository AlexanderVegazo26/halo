// Device-level tests: context, ICD report, buffers (direct + staging), streams, kernel
// validation, pipeline cache. Skip (reported) when no Vulkan device exists.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/kernel.h"
#include "halo/backends/vulkan/ops.h"
#include "halo/backends/vulkan/shaders.h"
#include "vk_test_util.h"

namespace hv = halo::vulkan;
using hv::test::download;
using hv::test::upload;

TEST(VkDevice, DescribeReportsDriverPolicyAndMemoryHonestly) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const hv::DeviceInfo& d = ctx->info();
    const nlohmann::json j = ctx->describe();
    std::cout << "[vk-device] " << j.dump(2) << "\n";
    EXPECT_EQ(j["available"], true);
    EXPECT_EQ(j["selected"], ctx->selection().index);
    ASSERT_TRUE(j["devices"].is_array());
    ASSERT_LT(ctx->selection().index, j["devices"].size());
    const auto& jd = j["devices"][ctx->selection().index];
    EXPECT_EQ(jd["name"], d.name);
    EXPECT_EQ(jd["driver"], std::string(hv::to_string(d.driver)));
    EXPECT_FALSE(jd["memory_heaps"].empty());
    EXPECT_FALSE(jd["memory_types"].empty());
    EXPECT_GT(d.subgroup_size, 0u);
    EXPECT_GE(d.api_version, hv::k_min_api_version);
    EXPECT_TRUE(d.compute_queue_family.has_value());
    if (d.driver == hv::DriverKind::Lavapipe) {
        EXPECT_TRUE(d.correctness_only);
        EXPECT_EQ(j["correctness_only"], true);
        bool warned = false;
        for (const auto& w : ctx->selection().warnings) warned = warned || w.find("correctness-only") != std::string::npos;
        EXPECT_TRUE(warned);
    }
    // describe_devices() works without creating a logical device.
    const nlohmann::json dd = hv::describe_devices();
    EXPECT_EQ(dd["available"], true);
    EXPECT_EQ(dd["devices"].size(), j["devices"].size());
    EXPECT_EQ(dd["selected"], j["selected"]);
}

TEST(VkBuffer, DirectRoundTripAllUsages) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    for (const auto usage : {hv::MemoryUsage::DeviceLocal, hv::MemoryUsage::HostVisible, hv::MemoryUsage::HostCached}) {
        std::vector<std::uint32_t> src(1000);
        std::iota(src.begin(), src.end(), 7u);
        hv::Buffer b = upload(ctx, std::span<const std::uint32_t>(src), usage);
        EXPECT_EQ(b.size(), src.size() * 4);
        std::cout << "[vk-buffer] usage=" << hv::to_string(usage) << " type=" << b.memory_type()
                  << " tier=" << hv::to_string(b.tier()) << " host_visible=" << b.host_visible() << "\n";
        EXPECT_EQ(download<std::uint32_t>(b, src.size()), src);
        // Partial upload/download at an offset.
        const std::vector<std::uint32_t> patch{0xDEADBEEFu, 0xCAFEF00Du};
        b.upload(std::span<const std::uint32_t>(patch), 40);
        std::vector<std::uint32_t> got(2);
        b.download(std::span<std::uint32_t>(got), 40);
        EXPECT_EQ(got, patch);
    }
}

TEST(VkBuffer, StagingRoundTripExercisesGpuCopyPath) {
    hv::ContextOptions opts;
    opts.force_staging = true;
    HALO_VK_CONTEXT_OR_SKIP(ctx, opts);
    ASSERT_TRUE(ctx->force_staging());
    std::vector<float> src(4099);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) * 0.5f - 3.0f;
    hv::Buffer b = upload(ctx, std::span<const float>(src));
    EXPECT_EQ(download<float>(b, src.size()), src);
    const std::vector<float> patch{1.0f, 2.0f, 3.0f};
    b.upload(std::span<const float>(patch), 12);  // offset 12 bytes, 3 floats
    std::vector<float> got(5);
    b.download(std::span<float>(got), 8);
    EXPECT_EQ(got, (std::vector<float>{src[2], 1.0f, 2.0f, 3.0f, src[6]}));
}

TEST(VkBuffer, StagingIsBoundedReusedAndChunked) {
    // Code review N-1: one staging buffer per context, capacity ContextOptions::staging_bytes,
    // reused by every staged transfer; larger transfers are split into chunks. A 4 KiB
    // capacity forces many chunks, including a partial last one and odd offsets.
    hv::ContextOptions opts;
    opts.force_staging = true;
    opts.staging_bytes = 4096;
    HALO_VK_CONTEXT_OR_SKIP(ctx, opts);
    EXPECT_EQ(ctx->staging_stats().allocations, 0u) << "staging is created lazily";
    const std::size_t n = 100003;  // bytes: 24 full chunks + a 1715-byte tail
    std::vector<std::uint8_t> src(n);
    for (std::size_t i = 0; i < n; ++i) src[i] = static_cast<std::uint8_t>((i * 131u + 7u) >> 3);
    hv::Buffer b = hv::Buffer::create(ctx, n + 16, hv::MemoryUsage::DeviceLocal);
    const std::vector<std::uint8_t> zeros(n + 16, 0);
    b.upload(std::span<const std::uint8_t>(zeros));
    const std::uint64_t chunks0 = ctx->staging_stats().chunks;
    b.upload(std::span<const std::uint8_t>(src), 5);  // odd destination offset
    const std::uint64_t per = (n + 4095) / 4096;
    EXPECT_EQ(ctx->staging_stats().chunks - chunks0, per);
    std::vector<std::uint8_t> got(n);
    b.download(std::span<std::uint8_t>(got), 5);
    EXPECT_EQ(got, src) << "chunked upload + download round trip";
    std::vector<std::uint8_t> edges(16);
    b.download(std::span<std::uint8_t>(edges.data(), 5), 0);
    b.download(std::span<std::uint8_t>(edges.data() + 5, 11), n + 5);
    EXPECT_TRUE(std::all_of(edges.begin(), edges.end(), [](std::uint8_t x) { return x == 0; }))
        << "bytes outside [5, 5 + n) were written";
    // A small transfer after the large ones reuses the same buffer.
    const std::vector<std::uint8_t> tiny{1, 2, 3};
    b.upload(std::span<const std::uint8_t>(tiny), 4097);
    std::vector<std::uint8_t> tiny_back(3);
    b.download(std::span<std::uint8_t>(tiny_back), 4097);
    EXPECT_EQ(tiny_back, tiny);
    const hv::Context::StagingStats st = ctx->staging_stats();
    EXPECT_EQ(st.allocations, 1u) << "one staging buffer for the whole context";
    EXPECT_EQ(st.capacity, 4096u);
    std::cout << "[vk-staging] capacity " << st.capacity << " B, allocations " << st.allocations << ", chunks "
              << st.chunks << "\n";

    hv::ContextOptions bad;
    bad.staging_bytes = 0;
    EXPECT_THROW((void)hv::Context::create(bad), halo::Error);
}

TEST(VkBuffer, RejectsZeroSizeAndOutOfRangeTransfers) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    EXPECT_THROW((void)hv::Buffer::create(ctx, 0, hv::MemoryUsage::DeviceLocal), halo::Error);
    hv::Buffer b = hv::Buffer::create(ctx, 64, hv::MemoryUsage::DeviceLocal);
    std::vector<std::byte> big(65);
    EXPECT_THROW(b.upload(std::span<const std::byte>(big)), halo::Error);
    std::vector<std::byte> four(4);
    EXPECT_THROW(b.upload(std::span<const std::byte>(four), 61), halo::Error);
    EXPECT_THROW(b.download(std::span<std::byte>(four), 62), halo::Error);
    EXPECT_NO_THROW(b.download(std::span<std::byte>(four), 60));
    const hv::DeviceInfo& d = ctx->info();
    if (d.max_memory_allocation_size > 0 && d.max_memory_allocation_size < UINT64_MAX) {
        EXPECT_THROW((void)hv::Buffer::create(ctx, d.max_memory_allocation_size + 1, hv::MemoryUsage::DeviceLocal),
                     halo::Error);
    }
}

TEST(VkBuffer, StagedTransfersRefuseOnAHungDeviceWithoutTouchingTheTransferCommandBuffer) {
    // After a fence wait fails, the transfer command buffer may still be pending on the GPU:
    // re-recording it (or refilling the staging buffer it reads) is invalid and can corrupt the
    // command stream the GPU is executing. Staged transfers must refuse instead, like streams.
    hv::ContextOptions opts;
    opts.force_staging = true;
    opts.staging_bytes = 4096;
    HALO_VK_CONTEXT_OR_SKIP(ctx, opts);
    hv::Buffer b = hv::Buffer::create(ctx, 64, hv::MemoryUsage::DeviceLocal);
    const std::vector<std::uint32_t> before{1, 2, 3, 4};
    b.upload(std::span<const std::uint32_t>(before));
    const std::uint64_t chunks0 = ctx->staging_stats().chunks;

    ctx->mark_hung();
    const auto expect_hung_refusal = [](const auto& transfer, const char* what) {
        try {
            transfer();
            ADD_FAILURE() << what << " on a hung device: expected Error(Backend) 'device is hung', got no throw";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), halo::ErrorCode::Backend) << what << ": " << e.what();
            EXPECT_NE(std::string(e.what()).find("hung"), std::string::npos)
                << what << ": message should say the device is hung, got: " << e.what();
        }
    };
    const std::vector<std::uint32_t> after{9, 9, 9, 9};
    expect_hung_refusal([&] { b.upload(std::span<const std::uint32_t>(after)); }, "staged upload");
    std::vector<std::uint32_t> got(4);
    expect_hung_refusal([&] { b.download(std::span<std::uint32_t>(got)); }, "staged download");
    expect_hung_refusal([&] { ctx->copy_buffer_sync(b.handle(), b.handle(), 0, 32, 16); }, "copy_buffer_sync");
    EXPECT_EQ(ctx->staging_stats().chunks, chunks0) << "no GPU copy may be issued once the device is hung";
}

TEST(VkBuffer, MoveTransfersOwnership) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Buffer a = hv::Buffer::create(ctx, 256, hv::MemoryUsage::HostVisible);
    const VkBuffer h = a.handle();
    hv::Buffer b = std::move(a);
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move): moved-from state is specified
    EXPECT_FALSE(a.valid());
    EXPECT_EQ(b.handle(), h);
    hv::Buffer c = hv::Buffer::create(ctx, 128, hv::MemoryUsage::DeviceLocal);
    c = std::move(b);
    EXPECT_EQ(c.handle(), h);
    EXPECT_EQ(c.size(), 256u);
}

TEST(VkBuffer, ResourcesMayOutliveTheirContextHandle) {
    // Buffers keep the device alive (shared ownership, TRD §32).
    hv::Buffer b;
    {
        HALO_VK_CONTEXT_OR_SKIP(ctx);
        b = hv::Buffer::create(ctx, 16, hv::MemoryUsage::HostVisible);
    }
    const std::vector<std::uint32_t> v{1, 2, 3, 4};
    b.upload(std::span<const std::uint32_t>(v));
    EXPECT_EQ(download<std::uint32_t>(b, 4), v);
}

TEST(VkStream, CopyTimestampsAndReuse) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    std::vector<std::uint32_t> src(1 << 16);
    std::iota(src.begin(), src.end(), 0u);
    hv::Buffer a = upload(ctx, std::span<const std::uint32_t>(src));
    hv::Buffer b = hv::Buffer::create(ctx, a.size(), hv::MemoryUsage::HostCached);
    hv::Stream s(ctx);
    const auto t0 = s.timestamp();
    s.copy(a, 0, b, 0, a.size());
    const auto t1 = s.timestamp();
    s.submit_and_wait();
    EXPECT_EQ(download<std::uint32_t>(b, src.size()), src);
    if (ctx->info().timestamps_supported()) {
        if (!t0 || !t1) {
            GTEST_FAIL() << "timestamps are supported but timestamp() returned no query (t0 "
                         << (t0 ? "set" : "empty") << ", t1 " << (t1 ? "set" : "empty") << ")";
        }
        const auto ns = s.elapsed_ns(*t0, *t1);
        if (!ns) GTEST_FAIL() << "elapsed_ns(t0, t1) around a 256 KiB copy returned no value";
        EXPECT_GT(*ns, 0.0) << "timestamps around a 256 KiB copy did not advance";
        std::cout << "[vk-timing] 256 KiB copy: " << *ns << " ns — "
                  << (ctx->info().correctness_only ? "lavapipe (CPU) timing — not a GPU performance number"
                                                   : "unvalidated GPU timing")
                  << "\n";
    } else {
        EXPECT_FALSE(t0.has_value());
    }
    // Second recording on the same stream (pool/fence reuse), copying back a shifted range.
    s.copy(b, 4, a, 0, 16);
    s.submit_and_wait();
    const auto back = download<std::uint32_t>(a, 4);
    EXPECT_EQ(back, (std::vector<std::uint32_t>{1, 2, 3, 4}));
    EXPECT_THROW(s.copy(a, 0, b, 1, a.size()), halo::Error);
}

// HALO_VK_OP_TIMINGS (next.md step 3): each completed fence wait appends that submission's
// "<kernel> <ns>" lines and then one "# wait" line, so scripts/op_profile.py can count fences
// (and dispatches per fence) from the profile alone. A wait() with nothing submitted adds nothing.
TEST(VkStream, OpTimingsProfileMarksEachFenceWaitAfterItsDispatches) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const std::filesystem::path path =
        std::filesystem::path(::testing::TempDir()) /
        ("halo_op_timings_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + ".txt");
    std::filesystem::remove(path);
    const char* old_env = std::getenv("HALO_VK_OP_TIMINGS");
    const std::optional<std::string> old = old_env ? std::optional<std::string>(old_env) : std::nullopt;
    ::setenv("HALO_VK_OP_TIMINGS", path.c_str(), 1);
    {
        constexpr std::uint32_t rows = 2, cols = 64;
        const std::vector<float> x(rows * cols, 0.5f), w(cols, 1.0f);
        hv::Buffer xb = upload(ctx, std::span<const float>(x));
        hv::Buffer wb = upload(ctx, std::span<const float>(w));
        hv::Buffer yb = hv::Buffer::create(ctx, xb.size(), hv::MemoryUsage::DeviceLocal);
        hv::Ops ops(ctx);
        hv::Stream s(ctx);
        ops.rms_norm(s, xb, wb, yb, rows, cols, 1e-6f);
        ops.rms_norm(s, yb, wb, xb, rows, cols, 1e-6f);
        s.submit_and_wait();  // submission 1: two dispatches
        s.wait();             // nothing in flight: no marker
        s.copy(xb, 0, yb, 0, xb.size());
        s.submit_and_wait();  // submission 2: a copy, no dispatch
    }
    if (old) {
        ::setenv("HALO_VK_OP_TIMINGS", old->c_str(), 1);
    } else {
        ::unsetenv("HALO_VK_OP_TIMINGS");
    }
    std::vector<std::string> lines;
    {
        std::ifstream in(path);
        for (std::string l; std::getline(in, l);) lines.push_back(l);
    }
    std::filesystem::remove(path);
    std::string got;
    for (const std::string& l : lines) got += "\n  " + l;
    // Kernel lines reduced to their name. The input uploads may be staged ("# transfer <bytes>",
    // depending on the device's memory types); VkContext.OpTimingsProfileMarksEachTransferFence*
    // pins those, so this Stream test ignores them.
    std::vector<std::string> shape;
    shape.reserve(lines.size());
    for (const std::string& l : lines) {
        if (l.starts_with("# transfer ")) continue;
        shape.push_back(l.starts_with('#') ? l : l.substr(0, l.find(' ')));
    }
    const std::vector<std::string> want =
        ctx->info().timestamps_supported()
            ? std::vector<std::string>{"rms_norm", "rms_norm", "# wait", "# wait"}
            : std::vector<std::string>{"# wait", "# wait"};
    EXPECT_EQ(shape, want) << "profile " << path << " should hold each submission's dispatch lines followed by one "
                           << "'# wait' per completed fence wait (2 here; the idle wait() adds none); got:" << got;
}

// HALO_VK_OP_TIMINGS (next.md step 4): staged uploads/downloads wait on the context's transfer
// fence, not a Stream's, so each completed staged copy appends "# transfer <bytes>" (one per
// staging chunk, in order) and a Stream fence wait still appends only "# wait". Nothing is
// written while the variable is unset.
TEST(VkContext, OpTimingsProfileMarksEachTransferFenceWithItsBytes) {
    hv::ContextOptions opts;
    opts.force_staging = true;
    opts.staging_bytes = 4096;
    HALO_VK_CONTEXT_OR_SKIP(ctx, opts);
    const std::filesystem::path path =
        std::filesystem::path(::testing::TempDir()) /
        ("halo_op_timings_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + ".txt");
    std::filesystem::remove(path);
    const char* old_env = std::getenv("HALO_VK_OP_TIMINGS");
    const std::optional<std::string> old = old_env ? std::optional<std::string>(old_env) : std::nullopt;
    std::vector<std::uint8_t> src(10000);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>(i * 37u + 11u);
    hv::Buffer a = hv::Buffer::create(ctx, src.size(), hv::MemoryUsage::DeviceLocal);
    hv::Buffer b = hv::Buffer::create(ctx, src.size(), hv::MemoryUsage::DeviceLocal);
    ::unsetenv("HALO_VK_OP_TIMINGS");
    a.upload(std::span<const std::uint8_t>(src));  // unset: no profile written
    const bool written_while_unset = std::filesystem::exists(path);
    ::setenv("HALO_VK_OP_TIMINGS", path.c_str(), 1);
    std::vector<std::uint8_t> got(5000);
    {
        b.upload(std::span<const std::uint8_t>(src));          // 4096 + 4096 + 1808
        b.download(std::span<std::uint8_t>(got), 3000);        // 4096 + 904
        hv::Stream s(ctx);
        s.copy(a, 0, b, 0, a.size());
        s.submit_and_wait();  // a Stream fence: "# wait", not a transfer
    }
    if (old) {
        ::setenv("HALO_VK_OP_TIMINGS", old->c_str(), 1);
    } else {
        ::unsetenv("HALO_VK_OP_TIMINGS");
    }
    std::vector<std::string> lines;
    {
        std::ifstream in(path);
        for (std::string l; std::getline(in, l);) lines.push_back(l);
    }
    std::filesystem::remove(path);
    EXPECT_FALSE(written_while_unset) << "a staged upload with HALO_VK_OP_TIMINGS unset created " << path;
    EXPECT_EQ(got, std::vector<std::uint8_t>(src.begin() + 3000, src.begin() + 8000)) << "staged round trip";
    const std::vector<std::string> want{"# transfer 4096", "# transfer 4096", "# transfer 1808",
                                        "# transfer 4096", "# transfer 904",  "# wait"};
    std::string shown;
    for (const std::string& l : lines) shown += "\n  " + l;
    EXPECT_EQ(lines, want) << "profile " << path << " should hold one '# transfer <bytes>' per staged chunk "
                           << "(upload 10000 B, download 5000 B at 4 KiB staging) and one '# wait' for the "
                           << "Stream fence; got:" << shown;
}

TEST(VkKernel, DispatchValidatesBindingsPushAndGroups) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::KernelDesc d;
    d.shader = &hv::find_shader("matvec_f32");
    d.num_buffers = 3;
    d.push_constant_bytes = 8;
    d.spec_constants = {{0, 64}};
    d.local_size = {64, 1, 1};
    hv::Kernel k(ctx, d);
    hv::Buffer w = hv::Buffer::create(ctx, 4096, hv::MemoryUsage::DeviceLocal);
    hv::Buffer x = hv::Buffer::create(ctx, 4096, hv::MemoryUsage::DeviceLocal);
    hv::Buffer y = hv::Buffer::create(ctx, 4096, hv::MemoryUsage::DeviceLocal);
    const std::uint32_t push[2] = {1, 1};
    const std::array<hv::BufferBinding, 3> ok{hv::BufferBinding{&w}, hv::BufferBinding{&x}, hv::BufferBinding{&y}};
    hv::Stream s(ctx);
    const auto bytes = std::as_bytes(std::span(push));
    EXPECT_THROW(s.dispatch(k, std::span(ok).first(2), bytes, {1, 1, 1}), halo::Error);  // binding count
    EXPECT_THROW(s.dispatch(k, ok, bytes.first(4), {1, 1, 1}), halo::Error);              // push size
    EXPECT_THROW(s.dispatch(k, ok, bytes, {0, 1, 1}), halo::Error);                       // zero groups
    EXPECT_THROW(s.dispatch(k, ok, bytes, {ctx->info().max_workgroup_count[0] + 1u, 1, 1}), halo::Error);
    std::array<hv::BufferBinding, 3> bad = ok;
    bad[1].offset = 4096;  // offset == size
    EXPECT_THROW(s.dispatch(k, bad, bytes, {1, 1, 1}), halo::Error);
    bad[1] = {&x, 0, 8192};  // range past end
    EXPECT_THROW(s.dispatch(k, bad, bytes, {1, 1, 1}), halo::Error);
    if (ctx->info().min_storage_buffer_offset_alignment > 1) {
        bad[1] = {&x, 1, 0};  // misaligned offset
        EXPECT_THROW(s.dispatch(k, bad, bytes, {1, 1, 1}), halo::Error);
    }
    bad[1] = {nullptr};
    EXPECT_THROW(s.dispatch(k, bad, bytes, {1, 1, 1}), halo::Error);
    EXPECT_NO_THROW(s.dispatch(k, ok, bytes, {1, 1, 1}));
    EXPECT_NO_THROW(s.submit_and_wait());
    EXPECT_EQ(s.dispatch_count(), 1u);

    hv::KernelDesc too_big = d;
    too_big.local_size = {ctx->info().max_workgroup_size[0] + 1u, 1, 1};
    EXPECT_THROW(hv::Kernel(ctx, too_big), halo::Error);
    hv::KernelDesc bad_push = d;
    bad_push.push_constant_bytes = 6;
    EXPECT_THROW(hv::Kernel(ctx, bad_push), halo::Error);
}

namespace {

/// `n` dependent rms_norm dispatches (600 = past the stream's 512-dispatch submit split, so the
/// recording continues in the stream's second command buffer with its own descriptor pools)
/// ping-ponging between two offset views of one buffer; `flush_every` > 0 submits and waits after
/// every that many dispatches instead (no split). Returns the final rows and the stream's
/// descriptor pool count.
std::pair<std::vector<float>, std::size_t> rms_norm_chain(const std::shared_ptr<hv::Context>& ctx, int n = 600,
                                                          int flush_every = 0) {
    constexpr std::uint32_t rows = 4, cols = 256;
    constexpr std::uint64_t region = 4096;  // bytes; rows * cols * 4, aligned for any device's offset rule
    std::vector<float> x0(2 * rows * cols), w(cols);
    for (std::size_t i = 0; i < x0.size(); ++i) x0[i] = static_cast<float>((i * 2654435761u) % 1000u) / 250.0f - 2.0f;
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.75f + static_cast<float>(i % 7u) / 16.0f;
    hv::Buffer xb = upload(ctx, std::span<const float>(x0));
    hv::Buffer wb = upload(ctx, std::span<const float>(w));
    hv::Ops ops(ctx);
    hv::Stream s(ctx);
    for (int i = 0; i < n; ++i) {
        const std::uint64_t src = (i % 2) * region, dst = region - src;
        ops.rms_norm(s, hv::BufferView(xb, src, region), wb, hv::BufferView(xb, dst, region), rows, cols, 1e-6f);
        if (flush_every > 0 && (i + 1) % flush_every == 0) s.submit_and_wait();
    }
    s.submit_and_wait();
    return {download<float>(xb, x0.size()), s.descriptor_pool_count()};
}

hv::ContextOptions push_options(bool on) {
    hv::ContextOptions o;
    o.push_descriptors = on;
    return o;
}

}  // namespace

// next.md step 4 (per-dispatch CPU cost): with ContextOptions::push_descriptors the stream pushes each
// dispatch's bindings instead of allocating and writing a descriptor set. Same buffers and offsets bound,
// so every output is bit for bit the descriptor-set path's, and the stream never creates a pool.
TEST(VkStream, PushDescriptorsBindTheSameBuffersWithoutDescriptorSets) {
    HALO_VK_CONTEXT_OR_SKIP(off, push_options(false));
    EXPECT_FALSE(off->push_descriptors_enabled()) << "ContextOptions::push_descriptors = false enabled them";
    const auto on = hv::Context::create(push_options(true));
    if (!on->push_descriptors_enabled())
        GTEST_SKIP() << "device '" << on->info().name << "' lacks VK_KHR_push_descriptor (maxPushDescriptors >= "
                     << hv::Context::kMaxBindings << ")";
    const auto [want, pools_off] = rms_norm_chain(off);
    const auto [got, pools_on] = rms_norm_chain(on);
    EXPECT_GE(pools_off, 1u) << "the descriptor-set path allocated no pool: the comparison is vacuous";
    EXPECT_EQ(pools_on, 0u) << "push descriptors on, yet the stream created " << pools_on << " descriptor pool(s)";
    ASSERT_EQ(got.size(), want.size());
    std::size_t diff = 0, first = got.size();
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i])) {
            if (diff++ == 0) first = i;
        }
    }
    EXPECT_EQ(diff, 0u) << diff << " of " << got.size() << " floats differ after 600 rms_norm dispatches; first at "
                        << first << ": push " << (first < got.size() ? got[first] : 0.0f) << " vs set "
                        << (first < got.size() ? want[first] : 0.0f);
}

namespace {

/// Runs `body` with HALO_VK_OP_TIMINGS pointed at a fresh file named after the current test plus
/// `tag`, restores the variable (also when `body` throws) and returns the profile's lines.
std::vector<std::string> op_timings_profile(const std::string& tag, const std::function<void()>& body) {
    const std::filesystem::path path =
        std::filesystem::path(::testing::TempDir()) /
        ("halo_op_timings_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + "_" +
         tag + ".txt");
    std::filesystem::remove(path);
    const char* old_env = std::getenv("HALO_VK_OP_TIMINGS");
    struct Restore {
        std::optional<std::string> old;
        ~Restore() {
            if (old) {
                ::setenv("HALO_VK_OP_TIMINGS", old->c_str(), 1);
            } else {
                ::unsetenv("HALO_VK_OP_TIMINGS");
            }
        }
    } restore{old_env ? std::optional<std::string>(old_env) : std::nullopt};
    ::setenv("HALO_VK_OP_TIMINGS", path.c_str(), 1);
    body();
    ::unsetenv("HALO_VK_OP_TIMINGS");
    std::vector<std::string> lines;
    {
        std::ifstream in(path);
        for (std::string l; std::getline(in, l);) lines.push_back(l);
    }
    std::filesystem::remove(path);
    return lines;
}

}  // namespace

// next.md step 4: the 512-dispatch submit split no longer makes the host wait mid-recording. The
// full command buffer is submitted and recording goes on in the stream's second one behind a full
// barrier; only a third buffer's worth waits (for the oldest, which is then reused). So a 600-dispatch
// chain completes with one fence wait (two before) and an 1100-dispatch chain with two (three before),
// and every dependent dispatch still sees its predecessor's output across the splits: the result is
// bit for bit a chain that submits and waits every 100 dispatches.
TEST(VkStream, SubmitSplitKeepsRecordingWithoutAFenceWait) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const bool timestamps = ctx->info().timestamps_supported();
    for (const int n : {600, 1100}) {
        const std::vector<float> want = rms_norm_chain(ctx, n, 100).first;
        std::vector<float> got;
        const std::vector<std::string> lines = op_timings_profile(std::to_string(n), [&] {
            got = rms_norm_chain(ctx, n).first;
        });
        std::size_t waits = 0, kernels = 0, other = 0;
        std::string shown;
        for (const std::string& l : lines) {
            if (l == "# wait") {
                ++waits;
                shown += "\n  # wait (after " + std::to_string(kernels) + " kernel lines)";
            } else if (l.starts_with("rms_norm ")) {
                ++kernels;
            } else if (!l.starts_with("# transfer ")) {  // staged input uploads, see VkContext.OpTimings*
                ++other;
                shown += "\n  " + l;
            }
        }
        const std::size_t want_waits = n > 1024 ? 2 : 1;
        EXPECT_EQ(waits, want_waits) << n << " dispatches in one recording should need " << want_waits
                                     << " fence wait(s): the split submits without waiting, only a third "
                                     << "command buffer waits for the oldest; profile:" << shown;
        EXPECT_EQ(kernels, timestamps ? static_cast<std::size_t>(n) : 0u)
            << "every dispatch of the " << n << "-dispatch chain should have its kernel line in the profile";
        EXPECT_EQ(other, 0u) << "unexpected profile lines:" << shown;
        ASSERT_EQ(got.size(), want.size());
        std::size_t diff = 0, first = got.size();
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i])) {
                if (diff++ == 0) first = i;
            }
        }
        EXPECT_EQ(diff, 0u) << diff << " of " << got.size() << " floats differ after " << n
                            << " dependent rms_norm dispatches split without a wait vs flushed every 100; first at "
                            << first << ": split " << (first < got.size() ? got[first] : 0.0f) << " vs flushed "
                            << (first < got.size() ? want[first] : 0.0f);
    }
}

TEST(VkKernel, PipelineCacheRoundTripsIntoANewContext) {
    std::vector<std::byte> data;
    {
        HALO_VK_CONTEXT_OR_SKIP(ctx);
        hv::Ops ops(ctx);
        hv::Buffer x = hv::Buffer::create(ctx, 1024, hv::MemoryUsage::DeviceLocal);
        hv::Buffer w = hv::Buffer::create(ctx, 1024, hv::MemoryUsage::DeviceLocal);
        hv::Buffer y = hv::Buffer::create(ctx, 1024, hv::MemoryUsage::DeviceLocal);
        const std::size_t empty_size = ctx->pipeline_cache_data().size();
        hv::Stream s(ctx);
        ops.rms_norm(s, x, w, y, 1, 256, 1e-6f);  // builds the pipeline through the cache
        data = ctx->pipeline_cache_data();
        std::cout << "[vk-cache] pipeline cache " << empty_size << " B empty -> " << data.size()
                  << " B after one pipeline\n";
        EXPECT_FALSE(data.empty());
        if (ctx->info().correctness_only) {
            // Measured: lavapipe keeps only the 32-byte header, so cache population is not
            // observable on it; the growth check applies to real GPU drivers.
            std::cout << "[vk-cache] CPU device: pipeline-cache population not observable here\n";
        } else {
            EXPECT_GT(data.size(), empty_size) << "pipeline creation did not go through the VkPipelineCache";
        }
    }
    hv::ContextOptions opts;
    opts.pipeline_cache_data = data;
    auto ctx2 = hv::Context::create(opts);
    EXPECT_FALSE(ctx2->pipeline_cache_data().empty());
    // Garbage cache data is rejected by the driver header check and the context still works.
    hv::ContextOptions junk;
    junk.pipeline_cache_data.assign(64, std::byte{0x5A});
    auto ctx3 = hv::Context::create(junk);
    EXPECT_NE(ctx3->pipeline_cache(), VK_NULL_HANDLE);
}
