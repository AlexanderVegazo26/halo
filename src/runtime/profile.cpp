// Profile key + CPU tunable op keys for the Engine (TRD §57, §64). See profile.h.
//
// Known debt: SHA-256 exists three times until WS-J's profiling::sha256_file is committed
// (tools/halo/hashing.cpp, include/halo/profiling/sha256.h in flight, and this file). All
// three implement FIPS 180-4 and the same PACK_ID text; switch to the shared one then.

#include "halo/runtime/profile.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "halo/core/error.h"
#include "halo/hardware/hardware.h"
#include "halo/profiling/hw_state.h"

namespace halo::runtime {

namespace {

class Sha256 {
public:
    void update(const unsigned char* p, std::size_t n) {
        len_ += n;
        while (n > 0) {
            const std::size_t take = std::min(n, 64 - fill_);
            std::memcpy(buf_.data() + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == 64) {
                block(buf_.data());
                fill_ = 0;
            }
        }
    }
    std::string hex() {
        const std::uint64_t bits = len_ * 8;
        const unsigned char one = 0x80;
        update(&one, 1);
        const unsigned char zero = 0;
        while (fill_ != 56) update(&zero, 1);
        std::array<unsigned char, 8> l{};
        for (int i = 0; i < 8; ++i) l[static_cast<std::size_t>(i)] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(l.data(), 8);
        std::string out;
        for (const std::uint32_t v : h_) out += std::format("{:08x}", v);
        return out;
    }

private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
            0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
            0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t{p[4 * i]} << 24) | (std::uint32_t{p[4 * i + 1]} << 16) | (std::uint32_t{p[4 * i + 2]} << 8) |
                   std::uint32_t{p[4 * i + 3]};
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::array<std::uint32_t, 8> v = h_;
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
            const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const std::uint32_t t1 = v[7] + S1 + ch + k[i] + w[i];
            const std::uint32_t S0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
            const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            const std::uint32_t t2 = S0 + maj;
            v = {t1 + t2, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
        }
        for (std::size_t i = 0; i < 8; ++i) h_[i] += v[i];
    }

    std::array<std::uint32_t, 8> h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> buf_{};
    std::size_t fill_ = 0;
    std::uint64_t len_ = 0;
};

/// Per-process cache of file hashes by (canonical path, size, mtime).
std::string cached_file_hash(const std::filesystem::path& p) {
    static std::mutex mu;
    static std::map<std::tuple<std::string, std::uintmax_t, std::int64_t>, std::string> cache;
    std::error_code ec;
    const auto canon = std::filesystem::weakly_canonical(p, ec).string();
    const auto size = std::filesystem::file_size(p, ec);
    HALO_CHECK(!ec, ErrorCode::Io, "cannot stat {}: {}", p.string(), ec.message());
    const auto mtime = std::filesystem::last_write_time(p, ec).time_since_epoch().count();
    const auto key = std::make_tuple(canon, size, static_cast<std::int64_t>(mtime));
    {
        const std::lock_guard lk(mu);
        if (auto it = cache.find(key); it != cache.end()) return it->second;
    }
    std::string h = sha256_file_hex(p);
    const std::lock_guard lk(mu);
    cache[key] = h;
    return h;
}

}  // namespace

std::string sha256_hex(std::string_view data) {
    Sha256 s;
    s.update(reinterpret_cast<const unsigned char*>(data.data()), data.size());
    return s.hex();
}

std::string sha256_file_hex(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    HALO_CHECK(f.good(), ErrorCode::Io, "cannot open {} for hashing", path.string());
    Sha256 s;
    std::vector<char> buf(4u << 20);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = f.gcount();
        if (got > 0) s.update(reinterpret_cast<const unsigned char*>(buf.data()), static_cast<std::size_t>(got));
    }
    HALO_CHECK(f.eof(), ErrorCode::Io, "read error while hashing {}", path.string());
    return s.hex();
}

std::string pack_id(std::string_view trunk_sha256, const std::optional<std::string>& mtp_sha256) {
    return sha256_hex(std::format("halo.pack/1\ntrunk={}\nmtp={}\n", trunk_sha256, mtp_sha256 ? *mtp_sha256 : "none"));
}

std::string cpu_isa_label() {
    const auto s = hardware::runtime_simd_flags();
#if defined(__x86_64__)
    if (s && s->avx512f) return "x86-64-avx512";
    if (s && s->avx2) return "x86-64-avx2";
    return "x86-64";
#else
    (void)s;
    return "cpu";
#endif
}

autotune::ProfileKey engine_profile_key(const EngineConfig& cfg, const std::string& hardware_root) {
    HALO_CHECK(!cfg.platform_power_mode.empty(), ErrorCode::Config,
               "profile_db needs platform_power_mode: the power mode is part of the profile key (TRD §57)");
    hardware::DiscoveryOptions d;
    d.root = hardware_root;
    const profiling::HardwareState hw = profiling::capture_hardware_state(d, cfg.platform_power_mode);
    const std::string trunk = cached_file_hash(cfg.model_path);
    std::optional<std::string> mtp;
    if (cfg.mtp_path) mtp = cached_file_hash(*cfg.mtp_path);
    return autotune::make_profile_key(hw, trunk, pack_id(trunk, mtp), cfg.isa_target.value_or(cpu_isa_label()));
}

autotune::OpKey matmul_op_key(const model::Qwen35HParams& hp) {
    return {"MATMUL", std::format("T=1,K={},N={},w=f32", hp.n_embd, hp.n_ff)};
}

autotune::OpKey gdn_op_key(const model::Qwen35HParams& hp) {
    return {"GATED_DELTANET", std::format("T={},Hk={},Hv={},dk={},dv={},impl=chunked", kGdnTuneTokens, hp.gdn_n_k_heads,
                                          hp.gdn_n_v_heads, hp.gdn_head_k_dim, hp.gdn_head_v_dim)};
}

namespace {

std::vector<std::int64_t> thread_values() {
    const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
    std::vector<std::int64_t> t;
    for (unsigned n = 1; n <= hw; ++n) t.push_back(n);
    return t;
}

std::vector<std::int64_t> chunk_values() {
    std::vector<std::int64_t> c;
    for (std::int64_t v = 1; v <= 1024; v *= 2) c.push_back(v);
    for (std::int64_t v = 16; v <= 512; v += 16) {
        if (std::find(c.begin(), c.end(), v) == c.end()) c.push_back(v);
    }
    return c;
}

}  // namespace

std::vector<autotune::Candidate> matmul_candidates() {
    std::vector<autotune::Candidate> out;
    for (const auto t : thread_values()) out.push_back(autotune::Candidate{{{"threads", t}}});
    return out;
}

std::vector<autotune::Candidate> gdn_candidates() {
    std::vector<autotune::Candidate> out;
    for (const auto c : chunk_values()) {
        for (const auto t : thread_values()) out.push_back(autotune::Candidate{{{"chunk", c}, {"threads", t}}});
    }
    return out;
}

}  // namespace halo::runtime
