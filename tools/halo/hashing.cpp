#include "hashing.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <vector>

#include "halo/core/error.h"

namespace halo::cli {

namespace {

constexpr std::array<std::uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

class Sha256 {
public:
    void update(const unsigned char* p, std::size_t n) {
        total_ += n;
        while (n > 0) {
            const std::size_t take = std::min(n, buf_.size() - len_);
            std::copy(p, p + take, buf_.begin() + static_cast<std::ptrdiff_t>(len_));
            len_ += take;
            p += take;
            n -= take;
            if (len_ == buf_.size()) {
                block(buf_.data());
                len_ = 0;
            }
        }
    }
    std::string hex() {
        const std::uint64_t bits = total_ * 8;
        const unsigned char pad = 0x80;
        update(&pad, 1);
        const unsigned char zero = 0;
        while (len_ != 56) update(&zero, 1);
        std::array<unsigned char, 8> len{};
        for (int i = 0; i < 8; ++i) len[static_cast<std::size_t>(i)] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(len.data(), len.size());
        static constexpr char digits[] = "0123456789abcdef";
        std::string out;
        for (const auto w : h_) {
            for (int s = 28; s >= 0; s -= 4) out.push_back(digits[(w >> s) & 0xF]);
        }
        return out;
    }

private:
    void block(const unsigned char* p) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t{p[4 * i]} << 24) | (std::uint32_t{p[4 * i + 1]} << 16) |
                   (std::uint32_t{p[4 * i + 2]} << 8) | std::uint32_t{p[4 * i + 3]};
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = h_;
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        const std::array<std::uint32_t, 8> add{a, b, c, d, e, f, g, h};
        for (std::size_t i = 0; i < 8; ++i) h_[i] += add[i];
    }

    std::array<std::uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> buf_{};
    std::size_t len_ = 0;
    std::uint64_t total_ = 0;
};

}  // namespace

std::string sha256_hex(std::string_view data) {
    Sha256 s;
    s.update(reinterpret_cast<const unsigned char*>(data.data()), data.size());  // NOLINT: byte view of text
    return s.hex();
}

std::string sha256_file_hex(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    HALO_CHECK(f.good(), ErrorCode::Io, "cannot open {} for hashing", path.string());
    Sha256 s;
    std::vector<char> buf(4u << 20);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto n = static_cast<std::size_t>(f.gcount());
        if (n > 0) s.update(reinterpret_cast<const unsigned char*>(buf.data()), n);  // NOLINT: byte view
    }
    HALO_CHECK(f.eof(), ErrorCode::Io, "reading {} failed", path.string());
    return s.hex();
}

std::string pack_id(const std::string& trunk_sha256, const std::optional<std::string>& mtp_sha256) {
    return sha256_hex("halo.pack/1\ntrunk=" + trunk_sha256 + "\nmtp=" + mtp_sha256.value_or("none") + "\n");
}

}  // namespace halo::cli
