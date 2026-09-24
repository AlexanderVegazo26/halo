#pragma once
// Shared helpers for the Vulkan device tests.
//
// Skip policy: a test skips (reported as skipped, never passed) only when there is no
// Vulkan loader/instance, zero physical devices, or no device meeting HALO's minimum
// (API 1.3 + compute queue). Any failure after a suitable device exists is a failure.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "halo/backends/vulkan/buffer.h"
#include "halo/backends/vulkan/context.h"
#include "halo/backends/vulkan/device.h"
#include "halo/core/error.h"

namespace halo::vulkan::test {

// Why no device is usable, or nullopt when one is.
inline std::optional<std::string> no_device_reason() {
    std::shared_ptr<Instance> inst;
    try {
        inst = Instance::create();
    } catch (const Error& e) {
        return std::string("no Vulkan instance: ") + e.what();
    }
    if (inst->devices().empty()) return std::string("Vulkan instance has zero physical devices");
    bool any = false;
    for (const DeviceInfo& d : inst->devices()) any = any || d.suitable();
    if (!any) return std::string("no Vulkan device with API >= 1.3 and a compute queue");
    return std::nullopt;
}

// Creates a context or records a skip (a context-creation failure on a machine that has
// a suitable device propagates as a test failure).
#define HALO_VK_CONTEXT_OR_SKIP(var, ...)                                             \
    std::shared_ptr<::halo::vulkan::Context> var;                                   \
    if (auto why_ = ::halo::vulkan::test::no_device_reason()) {                     \
        GTEST_SKIP() << "Vulkan unavailable: " << *why_;                            \
    }                                                                               \
    var = ::halo::vulkan::Context::create(__VA_ARGS__)

template <typename T>
Buffer upload(const std::shared_ptr<Context>& ctx, std::span<const T> data,
              MemoryUsage usage = MemoryUsage::DeviceLocal, std::size_t min_bytes = 0) {
    const std::size_t bytes = std::max(data.size_bytes(), min_bytes);
    Buffer b = Buffer::create(ctx, bytes, usage);
    b.upload(data);
    return b;
}

template <typename T>
std::vector<T> download(const Buffer& b, std::size_t count) {
    std::vector<T> out(count);
    b.download(std::span<T>(out));
    return out;
}

// Error summary for GPU (float) vs reference (double) arrays.
struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;       // |d| / |ref| over elements with |ref| >= rel_floor
    double max_ratio = 0.0;     // |d| / tolerance_i  (must be <= 1)
    std::size_t worst = 0;      // index of max_ratio
    std::size_t count = 0;
};

// tol(i) is the per-element a priori bound; rel_floor excludes near-zero refs from max_rel.
template <typename TolFn>
ErrorStats compare(std::span<const float> got, std::span<const double> ref, TolFn tol,
                   double rel_floor = 1e-3) {
    ErrorStats s;
    s.count = got.size();
    double ref_max = 0.0;
    for (double r : ref) ref_max = std::max(ref_max, std::fabs(r));
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (!std::isfinite(static_cast<double>(got[i]))) {
            s.max_ratio = std::numeric_limits<double>::infinity();
            s.worst = i;
            continue;
        }
        s.max_abs = std::max(s.max_abs, d);
        if (std::fabs(ref[i]) >= rel_floor * ref_max && ref[i] != 0.0) {
            s.max_rel = std::max(s.max_rel, d / std::fabs(ref[i]));
        }
        const double t = tol(i);
        const double ratio = t > 0.0 ? d / t : (d == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        if (ratio > s.max_ratio) {
            s.max_ratio = ratio;
            s.worst = i;
        }
    }
    return s;
}

inline void report(const std::string& what, const ErrorStats& s) {
    std::cout << "[vk-err] " << what << ": n=" << s.count << " max_abs=" << s.max_abs
              << " max_rel=" << s.max_rel << " max_err/bound=" << s.max_ratio << "\n";
}

}  // namespace halo::vulkan::test
