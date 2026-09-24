// Mutation fuzzing of the GGUF parser + qwen35 validation. Invariant: for any input, parsing
// and model validation either succeed or throw halo::Error — never another exception
// (bad_alloc, length_error, out_of_range, ...), never a crash or UB. Run under ASan/UBSan
// (scripts/build.sh --asan) to turn memory errors into failures.

#include <gtest/gtest.h>

#include <cstdio>
#include <exception>
#include <random>

#include "gguf_builder.h"
#include "halo/core/error.h"
#include "halo/model/gguf.h"
#include "halo/model/model.h"

using halo::model::GgufFile;
using halo::model::GgufMode;
using halo::model::NormalizedModel;

namespace {

struct Outcome {
    int ok = 0;
    int typed_error = 0;
};

// Parses in both modes and, if parsing succeeds, validates as a model.
void exercise(const std::vector<std::byte>& img, Outcome& out, std::size_t iter) {
    for (GgufMode mode : {GgufMode::HeaderOnly, GgufMode::Full}) {
        try {
            GgufFile f = GgufFile::parse(img, mode, "fuzz");
            NormalizedModel m = NormalizedModel::from_gguf(std::move(f));
            (void)m.tokenizer();
            ++out.ok;
        } catch (const halo::Error&) {
            ++out.typed_error;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "iteration " << iter << ": non-halo exception: " << e.what();
        }
    }
}

std::vector<std::byte> mutate(const std::vector<std::byte>& base, std::mt19937_64& rng) {
    std::vector<std::byte> v = base;
    const auto pick = [&](std::size_t n) { return static_cast<std::size_t>(rng() % n); };
    switch (rng() % 6) {
        case 0: {  // flip 1-8 random bytes
            const int n = 1 + static_cast<int>(rng() % 8);
            for (int k = 0; k < n; ++k) v[pick(v.size())] = static_cast<std::byte>(rng() & 0xFF);
            break;
        }
        case 1: {  // overwrite an 8-byte window with an extreme value (lengths, counts, offsets)
            static constexpr std::uint64_t kVals[] = {0, 1, 0xFF, 0xFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull,
                                                      0xFFFFFFFFFFFFFFFFull, 1ull << 32, 1ull << 62};
            const std::uint64_t val = kVals[rng() % std::size(kVals)];
            const std::size_t at = pick(v.size() - 8);
            std::memcpy(v.data() + at, &val, 8);
            break;
        }
        case 2:  // truncate
            v.resize(pick(v.size()));
            break;
        case 3: {  // overwrite a 4-byte window (types, n_dims, ggml type ids)
            const std::uint32_t val = static_cast<std::uint32_t>(rng() % 64);
            std::memcpy(v.data() + pick(v.size() - 4), &val, 4);
            break;
        }
        case 4: {  // insert random bytes
            const std::size_t at = pick(v.size());
            const std::size_t n = 1 + pick(16);
            std::vector<std::byte> ins(n);
            for (auto& b : ins) b = static_cast<std::byte>(rng() & 0xFF);
            v.insert(v.begin() + static_cast<std::ptrdiff_t>(at), ins.begin(), ins.end());
            break;
        }
        default: {  // delete a range
            const std::size_t at = pick(v.size());
            const std::size_t n = std::min(v.size() - at, 1 + pick(32));
            v.erase(v.begin() + static_cast<std::ptrdiff_t>(at), v.begin() + static_cast<std::ptrdiff_t>(at + n));
            break;
        }
    }
    return v;
}

}  // namespace

TEST(GgufFuzz, MutatedQwen35FilesNeverCrash) {
    const auto spec = halo::test::tiny_qwen35();
    const auto base = spec.build(true);
    // Mutations are concentrated in the header region (that is where the parser reads);
    // the data section is ~0.5 MB, so 3 of 4 mutants start from the header-only image.
    const auto header = spec.build(false);
    {
        Outcome sanity;
        exercise(base, sanity, 0);
        ASSERT_EQ(sanity.ok, 2) << "unmutated file must load in both modes";
    }
    std::mt19937_64 rng(0x4841'4C4F);  // fixed seed: deterministic, reproducible
    constexpr std::size_t kIters = 6000;
    Outcome out;
    for (std::size_t i = 0; i < kIters; ++i) {
        exercise(mutate(i % 4 == 0 ? base : header, rng), out, i);
        if (HasFailure()) break;
    }
    std::printf("[ info ] %zu mutants: %d loads ok, %d typed errors\n", kIters, out.ok, out.typed_error);
    EXPECT_GT(out.typed_error, 0);
}
