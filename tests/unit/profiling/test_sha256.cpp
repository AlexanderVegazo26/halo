#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "halo/profiling/sha256.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;

// FIPS 180-4 / NIST CAVS example vectors.
TEST(Sha256, NistVectors) {
    EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    Sha256 h;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) h.update(chunk);  // one million 'a'
    EXPECT_EQ(h.finish_hex(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, StreamingEqualsOneShotForEveryChunkSize) {
    std::string data;
    for (int i = 0; i < 5000; ++i) data.push_back(static_cast<char>((i * 7 + 3) & 0xff));
    const std::string ref = sha256_hex(data);
    for (const std::size_t chunk : {1UL, 55UL, 56UL, 63UL, 64UL, 65UL, 119UL, 4096UL}) {
        Sha256 h;
        for (std::size_t off = 0; off < data.size(); off += chunk) {
            h.update(std::string_view(data).substr(off, chunk));
        }
        EXPECT_EQ(h.finish_hex(), ref) << chunk;
    }
    // Lengths around the padding boundary (55/56/64 bytes) against one-shot.
    for (const std::size_t n : {55UL, 56UL, 57UL, 63UL, 64UL, 65UL}) {
        Sha256 a;
        a.update(std::string_view(data).substr(0, 1));
        a.update(std::string_view(data).substr(1, n - 1));
        EXPECT_EQ(a.finish_hex(), sha256_hex(std::string_view(data).substr(0, n))) << n;
    }
}

TEST(Sha256, FileStreamingAndErrors) {
    const auto p = std::filesystem::temp_directory_path() / ("halo_wsj_sha_" + std::to_string(::getpid()));
    {
        std::ofstream out(p, std::ios::binary);
        const std::string chunk(1000, 'a');
        for (int i = 0; i < 1000; ++i) out << chunk;
    }
    std::uint64_t last = 0;
    EXPECT_EQ(sha256_file(p, 4097, [&](std::uint64_t n) { last = n; }),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    EXPECT_EQ(last, 1000000u);
    std::filesystem::remove(p);
    test::expect_error(ErrorCode::Io, [&] { (void)sha256_file(p); });
    test::expect_error(ErrorCode::Config, [&] { (void)sha256_file(p, 0); });
}

TEST(Sha256, PackIdDefinition) {
    const std::string t = sha256_hex("trunk");
    const std::string m = sha256_hex("mtp");
    EXPECT_EQ(make_pack_id(t, std::nullopt), sha256_hex("halo.pack/1\ntrunk=" + t + "\nmtp=none\n"));
    EXPECT_EQ(make_pack_id(t, m), sha256_hex("halo.pack/1\ntrunk=" + t + "\nmtp=" + m + "\n"));
    EXPECT_NE(make_pack_id(t, m), make_pack_id(m, t));
    test::expect_error(ErrorCode::Config, [&] { (void)make_pack_id("ABC", std::nullopt); });
    std::string upper = t;
    upper[0] = 'F';
    test::expect_error(ErrorCode::Config, [&] { (void)make_pack_id(upper, std::nullopt); });
    test::expect_error(ErrorCode::Config, [&] { (void)make_pack_id(t, std::string("xyz")); });
}
