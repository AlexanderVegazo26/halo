#pragma once
// Test helpers for the HIP backend: one scenario runs either on the host emulation (both
// thread/workgroup orders; compared bit for bit against halo::cpu) or on a real device
// (compared against halo::cpu within a stated tolerance; skipped without a device, D-001).

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "halo/backends/hip/ops.h"
#include "halo/backends/hip/runtime.h"

namespace halo::hip::test {

inline constexpr const char* kNoDevice = "no HIP device (dev host, D-001)";

/// Where a scenario runs.
class Runner {
public:
    virtual ~Runner() = default;
    /// A Buffer holding `host` (emulation: wraps it; device: a device copy).
    virtual Buffer make(std::vector<float>& host) = 0;
    virtual Buffer make_words(std::vector<std::uint32_t>& host) = 0;
    /// Brings a buffer's content back into `host` (emulation: nothing to do).
    virtual void fetch(const Buffer& b, std::vector<float>& host) = 0;
    [[nodiscard]] virtual Target target() const = 0;
    virtual void finish() = 0;
    /// Bitwise comparison (emulation) or tolerance (device).
    [[nodiscard]] virtual bool exact() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

class EmuRunner final : public Runner {
public:
    explicit EmuRunner(bool reverse) : reverse_(reverse) {}
    Buffer make(std::vector<float>& host) override {
        return Buffer::wrap_host(host.data(), host.size() * sizeof(float));
    }
    Buffer make_words(std::vector<std::uint32_t>& host) override {
        return Buffer::wrap_host(host.data(), host.size() * sizeof(std::uint32_t));
    }
    void fetch(const Buffer&, std::vector<float>&) override {}
    [[nodiscard]] Target target() const override { return Target::emulation(reverse_); }
    void finish() override {}
    [[nodiscard]] bool exact() const override { return true; }
    [[nodiscard]] std::string name() const override { return reverse_ ? "emulation(reverse)" : "emulation(forward)"; }

private:
    bool reverse_;
};

class DeviceRunner final : public Runner {
public:
    DeviceRunner() : ctx_(Context::create(0)), stream_(*ctx_) {}
    Buffer make(std::vector<float>& host) override {
        Buffer b = Buffer::allocate(*ctx_, host.size() * sizeof(float), MemoryTier::Device);
        b.upload(host.data(), host.size() * sizeof(float));
        return b;
    }
    Buffer make_words(std::vector<std::uint32_t>& host) override {
        Buffer b = Buffer::allocate(*ctx_, host.size() * sizeof(std::uint32_t), MemoryTier::Device);
        b.upload(host.data(), host.size() * sizeof(std::uint32_t));
        return b;
    }
    void fetch(const Buffer& b, std::vector<float>& host) override {
        b.download(host.data(), host.size() * sizeof(float));
    }
    [[nodiscard]] Target target() const override { return Target::device(stream_); }
    void finish() override { stream_.synchronize(); }
    [[nodiscard]] bool exact() const override { return false; }
    [[nodiscard]] std::string name() const override { return "device " + ctx_->info().gcn_arch; }

private:
    std::shared_ptr<Context> ctx_;
    Stream stream_;
};

/// Skips the calling test when no HIP device is present (never passes silently).
#define HALO_REQUIRE_HIP_DEVICE()                                                   \
    do {                                                                            \
        const ::halo::hip::Availability av_ = ::halo::hip::probe();                 \
        if (!av_.available()) GTEST_SKIP() << ::halo::hip::test::kNoDevice << ": " << av_.reason; \
    } while (0)

/// Device tolerance (unmeasured, to be confirmed on the EVO-X2): the device differs from
/// halo::cpu only in expf (ocml, a few ulp) — every other operation is the same fp32
/// sequence. |err| <= kDevAbs + kDevRel * |ref| allows ~100 ulp of relative drift.
inline constexpr float kDevAbs = 2e-5f;
inline constexpr float kDevRel = 1e-5f;

/// Compares `got` against `ref` element by element.
inline ::testing::AssertionResult matches(const std::vector<float>& ref, const std::vector<float>& got,
                                          bool exact, const std::string& what) {
    if (ref.size() != got.size()) {
        return ::testing::AssertionFailure() << what << ": size " << got.size() << " != " << ref.size();
    }
    std::size_t bad = 0;
    std::size_t first = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        bool ok = false;
        if (exact) {
            ok = std::bit_cast<std::uint32_t>(ref[i]) == std::bit_cast<std::uint32_t>(got[i]);
        } else {
            const float d = std::fabs(ref[i] - got[i]);
            ok = d <= kDevAbs + kDevRel * std::fabs(ref[i]);
            worst = std::max(worst, static_cast<double>(d));
        }
        if (!ok) {
            if (bad == 0) first = i;
            ++bad;
        }
    }
    if (bad == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << what << ": " << bad << " of " << ref.size() << " elements differ ("
                                         << (exact ? "bitwise" : "tolerance") << "); first at " << first
                                         << ": ref " << ref[first] << " got " << got[first]
                                         << (exact ? "" : " worst |d| " + std::to_string(worst));
}

inline std::vector<float> uniform(std::mt19937& rng, std::size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

/// The runners a scenario is executed on: both emulation orders, plus the device when
/// `device` is set (the caller skips first when there is none).
inline std::vector<std::unique_ptr<Runner>> emulation_runners() {
    std::vector<std::unique_ptr<Runner>> r;
    r.push_back(std::make_unique<EmuRunner>(false));
    r.push_back(std::make_unique<EmuRunner>(true));
    return r;
}

}  // namespace halo::hip::test
