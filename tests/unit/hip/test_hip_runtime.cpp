// HIP host runtime: discovery, error translation, memory tiers, streams/events, registry.
// Device-dependent tests skip with "no HIP device (dev host, D-001)".

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <set>
#include <string>

#include "halo/backends/hip/registry.h"
#include "hip_test_util.h"

namespace halo::hip::test {
namespace {

TEST(HipRuntime, ProbeNeverThrowsAndExplainsZeroDevices) {
    const Availability a = probe();
    if (a.available()) {
        EXPECT_GT(a.device_count, 0);
        EXPECT_TRUE(a.reason.empty());
    } else {
        EXPECT_EQ(a.device_count, 0);
        EXPECT_FALSE(a.reason.empty());
        std::printf("probe: %s\n", a.reason.c_str());  // dev host: hipErrorNoDevice
    }
}

TEST(HipRuntime, ContextCreateWithoutDeviceIsDeviceError) {
    if (probe().available()) GTEST_SKIP() << "a HIP device is present; the no-device path is not reachable";
    try {
        static_cast<void>(Context::create(0));
        FAIL() << "Context::create succeeded without a device";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Device) << e.what();
    }
}

TEST(HipRuntime, ErrorTranslationUsesTypedCodes) {
    EXPECT_EQ(error_code_for(hipErrorOutOfMemory), ErrorCode::Memory);
    EXPECT_EQ(error_code_for(hipErrorNoDevice), ErrorCode::Device);
    EXPECT_EQ(error_code_for(hipErrorInvalidDevice), ErrorCode::Device);
    EXPECT_EQ(error_code_for(hipErrorInsufficientDriver), ErrorCode::Device);
    EXPECT_EQ(error_code_for(hipErrorLaunchFailure), ErrorCode::Kernel);
    EXPECT_EQ(error_code_for(hipErrorIllegalAddress), ErrorCode::Kernel);
    EXPECT_EQ(error_code_for(hipErrorNoBinaryForGpu), ErrorCode::Kernel);
    EXPECT_EQ(error_code_for(hipErrorInvalidConfiguration), ErrorCode::Kernel);
    EXPECT_EQ(error_code_for(hipErrorNotSupported), ErrorCode::Unsupported);
    EXPECT_EQ(error_code_for(hipErrorInvalidValue), ErrorCode::Backend);
    EXPECT_NO_THROW(check(hipSuccess, "noop"));
    try {
        check(hipErrorOutOfMemory, "hipMalloc");
        FAIL() << "check did not throw";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Memory);
        const std::string w = e.what();
        EXPECT_NE(w.find("MEMORY_ERROR"), std::string::npos) << w;
        EXPECT_NE(w.find("hipMalloc"), std::string::npos) << w;
        EXPECT_NE(w.find("hipErrorOutOfMemory"), std::string::npos) << w;
    }
}

TEST(HipRuntime, CompiledForGfx1151) {
    EXPECT_NE(compiled_offload_archs().find("gfx1151"), std::string_view::npos) << compiled_offload_archs();
    EXPECT_FALSE(compiled_for_generic_isa());
}

TEST(HipRuntime, HostBufferRoundTripAndRangeChecks) {
    std::vector<float> host(8, 0.0f);
    const Buffer b = Buffer::wrap_host(host.data(), host.size() * sizeof(float));
    EXPECT_EQ(b.tier(), MemoryTier::Host);
    const float in[2] = {1.5f, -2.0f};
    b.upload(in, sizeof(in), 4 * sizeof(float));
    EXPECT_EQ(host[4], 1.5f);
    EXPECT_EQ(host[5], -2.0f);
    float out[2] = {};
    b.download(out, sizeof(out), 4 * sizeof(float));
    EXPECT_EQ(out[0], 1.5f);
    try {
        b.upload(in, sizeof(in), 7 * sizeof(float));
        FAIL() << "out-of-range upload accepted";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Memory);
    }
    EXPECT_THROW(static_cast<void>(Buffer::wrap_host(nullptr, 4)), Error);
}

TEST(HipRuntime, RegistryListsGdnVariantsAndRejectsUnknownNames) {
    std::set<std::string_view> names;
    for (const KernelVariant& v : kernel_variants()) {
        EXPECT_TRUE(names.insert(v.name).second) << "duplicate variant " << v.name;
        EXPECT_GT(v.block, 0u);
        EXPECT_LE(v.block, 256u) << v.name << ": above the kernels' __launch_bounds__";
        EXPECT_FALSE(v.decisions.empty()) << v.name;
    }
    EXPECT_EQ(default_variant("GATED_DELTANET", "recurrent").name, "gdn_recurrent_b128");
    EXPECT_EQ(default_variant("GATED_DELTANET", "chunked").name, "gdn_chunked_b64");
    EXPECT_EQ(default_variant("CONV1D_SHORT").name, "conv1d_silu_b256");
    EXPECT_EQ(default_variant("GATED_NORM").name, "gated_norm_b128");
    EXPECT_GE(kernel_variants("GATED_DELTANET").size(), 4u);
    EXPECT_THROW(static_cast<void>(find_variant("GATED_DELTANET", "nope")), Error);
    OpsOptions bad;
    bad.gdn_recurrent = "gdn_chunked_b64";  // right op, wrong form
    try {
        const Ops ops(bad);
        FAIL() << "wrong-form variant accepted";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Config) << e.what();
    }
    OpsOptions unknown;
    unknown.conv1d = "conv1d_silu_b999";
    EXPECT_THROW(Ops{unknown}, Error);
}

// ---- device-only ----------------------------------------------------------------------

TEST(HipRuntimeDevice, ContextReportsGfx1151Properties) {
    HALO_REQUIRE_HIP_DEVICE();
    const auto ctx = Context::create(0);
    const DeviceInfo& i = ctx->info();
    std::printf("device: %s %s CUs %d wave%d mem %llu MiB driver %d runtime %d integrated %d\n", i.name.c_str(),
                i.gcn_arch_full.c_str(), i.compute_units, i.warp_size,
                static_cast<unsigned long long>(i.total_global_mem >> 20), i.driver_version, i.runtime_version,
                i.integrated ? 1 : 0);
    EXPECT_EQ(i.gcn_arch, "gfx1151");
    EXPECT_GT(i.compute_units, 0);
    EXPECT_GT(i.total_global_mem, 0u);
}

TEST(HipRuntimeDevice, AllocatesEveryTierAndRoundTrips) {
    HALO_REQUIRE_HIP_DEVICE();
    const auto ctx = Context::create(0);
    for (MemoryTier t : {MemoryTier::Device, MemoryTier::HostPinned, MemoryTier::Managed}) {
        SCOPED_TRACE(std::string(to_string(t)));
        const Buffer b = Buffer::allocate(*ctx, 1u << 20, t);
        std::vector<std::uint32_t> in(256), out(256);
        for (std::uint32_t i = 0; i < in.size(); ++i) in[i] = i * 2654435761u;
        b.upload(in.data(), in.size() * 4, 4096);
        b.download(out.data(), out.size() * 4, 4096);
        EXPECT_EQ(in, out);
    }
    try {
        static_cast<void>(Buffer::allocate(*ctx, 1ull << 62, MemoryTier::Device));
        FAIL() << "4 EiB allocation succeeded";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Memory) << e.what();
    }
}

TEST(HipRuntimeDevice, StreamEventsTime) {
    HALO_REQUIRE_HIP_DEVICE();
    const auto ctx = Context::create(0);
    Stream s(*ctx);
    Event a(*ctx), b(*ctx);
    a.record(s);
    b.record(s);
    b.synchronize();
    EXPECT_GE(Event::elapsed_ms(a, b), 0.0f);
}

TEST(HipRuntimeDevice, TargetRejectsWrongMemoryKind) {
    HALO_REQUIRE_HIP_DEVICE();
    const auto ctx = Context::create(0);
    Stream s(*ctx);
    std::vector<float> host(64, 0.0f);
    const Buffer hb = Buffer::wrap_host(host.data(), host.size() * 4);
    const Buffer db = Buffer::allocate(*ctx, 256, MemoryTier::Device);
    const Ops ops;
    GatedNormArgs a{.x = hb, .z = hb, .w = hb, .out = hb, .rows = 1, .cols = 4};
    EXPECT_THROW(ops.gated_rms_norm(Target::device(s), a), Error);
    GatedNormArgs d{.x = db, .z = db, .w = db, .out = db, .rows = 1, .cols = 4};
    EXPECT_THROW(ops.gated_rms_norm(Target::emulation(), d), Error);
}

}  // namespace
}  // namespace halo::hip::test
