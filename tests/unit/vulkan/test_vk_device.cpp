// Device-level tests: context, ICD report, buffers (direct + staging), streams, kernel
// validation, pipeline cache. Skip (reported) when no Vulkan device exists.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
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

TEST(VkBuffer, MoveTransfersOwnership) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    hv::Buffer a = hv::Buffer::create(ctx, 256, hv::MemoryUsage::HostVisible);
    const VkBuffer h = a.handle();
    hv::Buffer b = std::move(a);
    EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move): moved-from state is specified
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
        ASSERT_TRUE(t0 && t1);
        const auto ns = s.elapsed_ns(*t0, *t1);
        ASSERT_TRUE(ns.has_value());
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
