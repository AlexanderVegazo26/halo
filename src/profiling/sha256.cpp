#include "halo/profiling/sha256.h"

#include <bit>
#include <cstring>
#include <fstream>
#include <vector>

#include "halo/core/error.h"

namespace halo::profiling {

namespace {

constexpr std::array<std::uint32_t, 64> kK{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) noexcept { return std::rotr(x, n); }

}  // namespace

void Sha256::reset() noexcept {
    h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    buf_len_ = 0;
    total_ = 0;
}

void Sha256::block(const std::uint8_t* p) noexcept {
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
        w[static_cast<std::size_t>(i)] =
            (static_cast<std::uint32_t>(p[4 * i]) << 24) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16) |
            (static_cast<std::uint32_t>(p[4 * i + 2]) << 8) | static_cast<std::uint32_t>(p[4 * i + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + s1 + ch + kK[i] + w[i];
        const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
}

void Sha256::update(std::span<const std::uint8_t> data) noexcept {
    total_ += data.size();
    std::size_t i = 0;
    if (buf_len_ > 0) {
        const std::size_t take = std::min(data.size(), 64 - buf_len_);
        std::memcpy(buf_.data() + buf_len_, data.data(), take);
        buf_len_ += take;
        i = take;
        if (buf_len_ < 64) return;
        block(buf_.data());
        buf_len_ = 0;
    }
    for (; i + 64 <= data.size(); i += 64) block(data.data() + i);
    if (i < data.size()) {
        buf_len_ = data.size() - i;
        std::memcpy(buf_.data(), data.data() + i, buf_len_);
    }
}

void Sha256::update(std::string_view text) noexcept {
    update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

std::array<std::uint8_t, 32> Sha256::finish() noexcept {
    const std::uint64_t bits = total_ * 8;
    std::array<std::uint8_t, 72> pad{};
    pad[0] = 0x80;
    const std::size_t pad_len = (buf_len_ < 56) ? (56 - buf_len_) : (120 - buf_len_);
    for (int i = 0; i < 8; ++i) pad[pad_len + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    const std::uint64_t saved = total_;
    update(std::span<const std::uint8_t>(pad.data(), pad_len + 8));
    total_ = saved;
    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 8; ++i) {
        out[4 * i] = static_cast<std::uint8_t>(h_[i] >> 24);
        out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
        out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
        out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
    }
    return out;
}

std::string Sha256::finish_hex() {
    const auto d = finish();
    return to_hex(d);
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s;
    s.reserve(bytes.size() * 2);
    for (const std::uint8_t b : bytes) {
        s.push_back(kHex[b >> 4]);
        s.push_back(kHex[b & 0xf]);
    }
    return s;
}

std::string sha256_hex(std::string_view text) {
    Sha256 h;
    h.update(text);
    return h.finish_hex();
}

std::string sha256_file(const std::filesystem::path& path, std::size_t chunk_bytes,
                        const std::function<void(std::uint64_t)>& progress) {
    HALO_CHECK(chunk_bytes >= 1 && chunk_bytes <= (256U << 20), ErrorCode::Config, "sha256_file: chunk {} invalid",
               chunk_bytes);
    std::ifstream in(path, std::ios::binary);
    HALO_CHECK(in.is_open(), ErrorCode::Io, "sha256_file: cannot open {}", path.string());
    std::vector<char> buf(chunk_bytes);
    Sha256 h;
    std::uint64_t done = 0;
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize n = in.gcount();
        if (n > 0) {
            h.update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(buf.data()),
                                                   static_cast<std::size_t>(n)));
            done += static_cast<std::uint64_t>(n);
            if (progress) progress(done);
        }
    }
    HALO_CHECK(in.eof() && !in.bad(), ErrorCode::Io, "sha256_file: read error on {}", path.string());
    return h.finish_hex();
}

std::string make_pack_id(std::string_view trunk_sha256, const std::optional<std::string>& mtp_sha256) {
    const auto hex64 = [](std::string_view s) {
        if (s.size() != 64) return false;
        for (const char c : s) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        }
        return true;
    };
    HALO_CHECK(hex64(trunk_sha256), ErrorCode::Config, "make_pack_id: trunk hash is not 64 lowercase hex chars");
    HALO_CHECK(!mtp_sha256 || hex64(*mtp_sha256), ErrorCode::Config,
               "make_pack_id: MTP hash is not 64 lowercase hex chars");
    std::string text = "halo.pack/1\ntrunk=";
    text += trunk_sha256;
    text += "\nmtp=";
    text += mtp_sha256 ? *mtp_sha256 : std::string("none");
    text += "\n";
    return sha256_hex(text);
}

}  // namespace halo::profiling
