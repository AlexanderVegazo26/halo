#include "halo/hardware/bandwidth.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstring>
#include <new>
#include <system_error>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"

namespace halo::hardware {

Stats compute_stats(std::span<const double> samples) {
    HALO_CHECK(!samples.empty(), ErrorCode::Config, "compute_stats: empty sample");
    for (const double v : samples) HALO_CHECK(std::isfinite(v), ErrorCode::Config, "compute_stats: non-finite sample");
    std::vector<double> s(samples.begin(), samples.end());
    std::ranges::sort(s);
    Stats st;
    st.n = s.size();
    st.min = s.front();
    st.max = s.back();
    double sum = 0;
    for (const double v : s) sum += v;
    st.mean = sum / static_cast<double>(st.n);
    st.median = st.n % 2 == 1 ? s[st.n / 2] : (s[st.n / 2 - 1] + s[st.n / 2]) / 2.0;
    if (st.n >= 2) {
        double ss = 0;
        for (const double v : s) ss += (v - st.mean) * (v - st.mean);
        st.stddev = std::sqrt(ss / static_cast<double>(st.n - 1));
    }
    // Nearest-rank percentile: the smallest value with at least 95% of samples <= it.
    const auto rank = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(st.n)));
    st.p95 = s[std::max<std::size_t>(rank, 1) - 1];
    return st;
}

namespace {

enum class Op : int { Read, Write, Copy };

inline void escape(std::uint64_t v) { __asm__ volatile("" : : "r"(v) : "memory"); }

struct Slice {
    std::size_t begin = 0;
    std::size_t end = 0;
};

}  // namespace

TierBandwidth measure_host_bandwidth(const BandwidthOptions& o) {
    HALO_CHECK(!o.label.empty(), ErrorCode::Config,
               "bandwidth results must be labelled with where they were measured (e.g. \"dev-host\")");
    HALO_CHECK(o.iterations >= 1, ErrorCode::Config, "iterations must be >= 1");
    HALO_CHECK(o.allow_nonconformant || (o.warmup >= 5 && o.iterations >= 20), ErrorCode::Config,
               "TRD §50 requires >= 5 warm-up and >= 20 measured runs (got {} / {})", o.warmup, o.iterations);
    const unsigned threads = o.threads != 0 ? o.threads : std::max(1U, std::thread::hardware_concurrency());
    HALO_CHECK(threads <= 1024, ErrorCode::Config, "threads {} exceeds 1024", threads);
    const std::size_t words = o.buffer_bytes / sizeof(std::uint64_t);
    HALO_CHECK(words >= threads, ErrorCode::Config, "buffer of {} bytes too small for {} threads", o.buffer_bytes,
               threads);
    const std::size_t bytes = words * sizeof(std::uint64_t);

    std::vector<std::uint64_t> src;
    std::vector<std::uint64_t> dst;
    try {
        src.assign(words, 0x5a5a5a5a5a5a5a5aULL);  // touches every page before timing
        dst.assign(words, 0);
    } catch (const std::bad_alloc&) {
        throw_error(ErrorCode::Memory, "cannot allocate 2 x {} bytes for the bandwidth benchmark", bytes);
    }

    std::vector<Slice> slices(threads);
    for (unsigned t = 0; t < threads; ++t) {
        slices[t].begin = words * t / threads;
        slices[t].end = words * (t + 1) / threads;
    }

    std::atomic<int> op{0};
    std::atomic<std::uint64_t> fill_value{1};
    std::atomic<bool> stop{false};
    std::barrier start(static_cast<std::ptrdiff_t>(threads) + 1);
    std::barrier done(static_cast<std::ptrdiff_t>(threads) + 1);

    auto worker = [&](unsigned t) {
        const Slice sl = slices[t];
        for (;;) {
            start.arrive_and_wait();
            if (stop.load(std::memory_order_acquire)) return;
            switch (static_cast<Op>(op.load(std::memory_order_relaxed))) {
                case Op::Read: {
                    std::uint64_t acc = 0;
                    for (std::size_t i = sl.begin; i < sl.end; ++i) acc += src[i];
                    escape(acc);
                    break;
                }
                case Op::Write: {
                    const std::uint64_t v = fill_value.load(std::memory_order_relaxed);
                    std::fill(dst.begin() + static_cast<std::ptrdiff_t>(sl.begin),
                              dst.begin() + static_cast<std::ptrdiff_t>(sl.end), v);
                    escape(dst[sl.begin]);
                    break;
                }
                case Op::Copy:
                    std::memcpy(dst.data() + sl.begin, src.data() + sl.begin,
                                (sl.end - sl.begin) * sizeof(std::uint64_t));
                    escape(dst[sl.begin]);
                    break;
            }
            done.arrive_and_wait();
        }
    };

    std::vector<double> samples[3];
    for (auto& s : samples) s.reserve(o.iterations);
    {
        std::vector<std::jthread> pool;
        pool.reserve(threads);
        // Guard: whatever happens, release the workers so the jthreads can join.
        struct Release {
            std::atomic<bool>& stop;
            std::barrier<>& start;
            bool armed = true;
            ~Release() {
                if (!armed) return;
                stop.store(true, std::memory_order_release);
                start.arrive_and_wait();
            }
        };
        try {
            for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker, t);
        } catch (const std::system_error& e) {
            // Release the workers that did start: stand in for the missing ones, then stop.
            stop.store(true, std::memory_order_release);
            for (std::size_t k = pool.size(); k < threads; ++k) (void)start.arrive_and_drop();
            start.arrive_and_wait();
            throw_error(ErrorCode::Config, "cannot start {} benchmark threads: {}", threads, e.what());
        }
        Release release{stop, start};

        for (const Op which : {Op::Read, Op::Write, Op::Copy}) {
            op.store(static_cast<int>(which), std::memory_order_relaxed);
            for (unsigned i = 0; i < o.warmup + o.iterations; ++i) {
                fill_value.store(i + 2, std::memory_order_relaxed);
                const auto t0 = std::chrono::steady_clock::now();
                start.arrive_and_wait();
                done.arrive_and_wait();
                const auto t1 = std::chrono::steady_clock::now();
                if (i < o.warmup) continue;
                const double secs = std::chrono::duration<double>(t1 - t0).count();
                const double moved = static_cast<double>(which == Op::Copy ? 2 * bytes : bytes);
                // A timer tick of 0 would be infinite bandwidth; clamp to 1 ns.
                samples[static_cast<int>(which)].push_back(moved / std::max(secs, 1e-9) / 1e9);
            }
        }
    }  // Release runs before the jthreads join (reverse declaration order).

    TierBandwidth r;
    r.tier = MemoryTier::Host;
    r.processor = Processor::Cpu;
    r.label = o.label;
    r.buffer_bytes = bytes;
    r.threads = threads;
    r.warmup = o.warmup;
    r.iterations = o.iterations;
    r.read_gbps = compute_stats(samples[0]);
    r.write_gbps = compute_stats(samples[1]);
    r.copy_gbps = compute_stats(samples[2]);
    return r;
}

void to_json(nlohmann::json& j, const Stats& v) {
    j = {{"n", v.n},           {"min", v.min},     {"max", v.max}, {"mean", v.mean},
         {"median", v.median}, {"stddev", v.stddev}, {"p95", v.p95}};
}

void to_json(nlohmann::json& j, const TierBandwidth& v) {
    j = {{"tier", std::string(to_string(v.tier))},
         {"processor", std::string(to_string(v.processor))},
         {"label", v.label},
         {"buffer_bytes", v.buffer_bytes},
         {"threads", v.threads},
         {"warmup", v.warmup},
         {"iterations", v.iterations},
         {"read_gbps", v.read_gbps},
         {"write_gbps", v.write_gbps},
         {"copy_gbps", v.copy_gbps},
         {"units", "GB/s (1e9 bytes/s); copy counts bytes read + written"}};
}

}  // namespace halo::hardware
