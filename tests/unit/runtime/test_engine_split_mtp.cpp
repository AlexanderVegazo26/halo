// WS-G M6: the D-006 "separate MTP file" packaging at engine level.
//
// python/tools/split_mtp_gguf.py splits tiny-f32.gguf (one file, embedded MTP block) into the
// ggml-org layout: a trunk GGUF (block_count n, no MTP) plus an MTP-only GGUF (block_count
// n+1, blk.n.* and its own token_embd / output / output_norm copies). The tensors are copied
// byte for byte, so an engine on trunk + mtp_path must reproduce the single-file goldens, and
// its MTP must produce the same drafts as the single-file engine's.
//
// The golden tokens alone cannot show that the MTP file was used: the trunk reproduces them
// with no MTP at all, and greedy output is invariant to the drafts. So the tests also assert
// has_mtp, that drafting ran, and that the drafts equal the single-file engine's drafts.
//
// Note: the MTP file's embedding / head copies are byte-identical to the trunk's here, so
// the D-006 choice between them (trunk by default) is invisible to these tests; it is
// covered by the model tests (MtpTensorOrigin).

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include "../models/tiny_golden.h"
#include "halo/core/error.h"
#include "halo/model/model.h"
#include "halo/runtime/cpu_engine.h"

using halo::runtime::CpuEngineOptions;
using halo::runtime::Engine;
using halo::runtime::EngineConfig;
using halo::runtime::GenerateRequest;
using halo::runtime::GenerateResult;
using halo::runtime::TickInfo;
using halo::speculative::GateMode;
using halo::test::Golden;

namespace fs = std::filesystem;

namespace {

using Toks = std::vector<std::int32_t>;

constexpr const char* kRegen =
    "generate it with: /root/halo-py/.venv/bin/python python/tools/split_mtp_gguf.py "
    "(writes $HALO_REF_DIR/tiny/split)";

fs::path single() { return halo::test::tiny_dir() / "tiny-f32.gguf"; }
fs::path split(const char* part) { return halo::test::tiny_dir() / "split" / (std::string("tiny-f32-") + part + ".gguf"); }

/// An engine plus the drafts its ticks reported (one request at a time).
struct Probe {
    std::mutex mu;
    Toks drafts;
    std::unique_ptr<Engine> engine;  // last: destroyed (worker joined) before mu / drafts

    static std::unique_ptr<Probe> make(EngineConfig c) {
        auto p = std::make_unique<Probe>();
        CpuEngineOptions o;
        o.gate_mode = GateMode::Always;  // draft on every greedy decode step
        o.on_tick = [raw = p.get()](const TickInfo& t) {
            const std::lock_guard lk(raw->mu);
            raw->drafts.insert(raw->drafts.end(), t.drafts.begin(), t.drafts.end());
        };
        p->engine = halo::runtime::create_cpu_engine(c, o);
        return p;
    }
    /// Greedy generation; returns the tokens and the drafts proposed during it.
    std::pair<GenerateResult, Toks> run(const Toks& prompt, std::size_t n) {
        {
            const std::lock_guard lk(mu);
            drafts.clear();
        }
        GenerateRequest r;
        r.prompt = prompt;
        r.sampling.temperature = 0.0f;
        r.max_tokens = n;
        GenerateResult res = engine->generate(r, {});
        const std::lock_guard lk(mu);
        return {std::move(res), drafts};
    }
};

EngineConfig cfg(const fs::path& model) {
    EngineConfig c;
    c.model_path = model.string();
    c.max_context = 512;
    c.max_sequences = 1;
    c.mtp_max_draft = 3;
    c.prefix_cache = false;  // every request computes from scratch (no cross-request state)
    return c;
}

class SplitMtp : public ::testing::Test {
protected:
    void SetUp() override {
        golden_ = Golden::load();
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        if (!fs::exists(single())) GTEST_SKIP() << "tiny-f32.gguf missing under " << halo::test::tiny_dir();
        for (const char* p : {"trunk", "mtp", "mtp-bad-dims", "mtp-bad-arch", "mtp-with-trunk"}) {
            if (!fs::exists(split(p))) GTEST_SKIP() << split(p) << " missing; " << kRegen;
        }
    }
    Toks prompt(const char* p) const { return golden_->i32(std::string(p) + ".tokens"); }
    Toks gold(const char* p) const { return golden_->i32(std::string(p) + ".decode_tokens"); }

    std::optional<Golden> golden_;
};

std::optional<halo::ErrorCode> code_of(const std::function<void()>& f, std::string& what) {
    try {
        f();
    } catch (const halo::Error& e) {
        what = e.what();
        return e.code();
    }
    what = "(no exception)";
    return std::nullopt;
}

}  // namespace

// (a) trunk + separate MTP file == single file: golden greedy tokens, MTP on == MTP off, and
// the MTP from the separate file drafts exactly what the embedded MTP drafts.
// One instance per prompt keeps each ctest within the ASan budget.
class SplitMtpPrompt : public SplitMtp, public ::testing::WithParamInterface<const char*> {};

TEST_P(SplitMtpPrompt, TrunkPlusMtpFileReproducesTheSingleFileEngine) {
    EngineConfig split_cfg = cfg(split("trunk"));
    split_cfg.mtp_path = split("mtp").string();
    const auto with_mtp = Probe::make(split_cfg);
    const auto embedded = Probe::make(cfg(single()));
    const auto trunk_only = Probe::make(cfg(split("trunk")));  // MTP off: the trunk file has none

    EXPECT_TRUE(with_mtp->engine->model().has_mtp);
    EXPECT_TRUE(embedded->engine->model().has_mtp);
    EXPECT_FALSE(trunk_only->engine->model().has_mtp);

    const char* p = GetParam();
    const Toks g = gold(p);
    const auto [r_split, d_split] = with_mtp->run(prompt(p), g.size());
    const auto [r_emb, d_emb] = embedded->run(prompt(p), g.size());
    const auto [r_off, d_off] = trunk_only->run(prompt(p), g.size());
    EXPECT_EQ(r_split.tokens, g) << r_split.error;
    EXPECT_EQ(r_off.tokens, g) << r_off.error;
    EXPECT_EQ(r_emb.tokens, g) << r_emb.error;
    // The MTP from the separate file ran (the goldens alone would pass without it) ...
    EXPECT_GT(r_split.draft_tokens, 0u);
    EXPECT_EQ(r_split.draft_tokens, d_split.size());
    EXPECT_EQ(r_off.draft_tokens, 0u);
    EXPECT_TRUE(d_off.empty());
    // ... and proposed exactly the embedded MTP's drafts, draft for draft.
    EXPECT_EQ(d_split, d_emb);
    EXPECT_EQ(r_split.accepted_draft_tokens, r_emb.accepted_draft_tokens);
}

INSTANTIATE_TEST_SUITE_P(Prompts, SplitMtpPrompt, ::testing::Values("p0", "p1", "p2"));

// (a) an MTP file whose dimensions disagree with the trunk fails engine creation with a
// typed Model error naming the mismatched key (not some other load failure).
TEST_F(SplitMtp, MtpFileWithMismatchedDimsIsRejectedWithATypedError) {
    EngineConfig c = cfg(split("trunk"));
    c.mtp_path = split("mtp-bad-dims").string();
    std::string what;
    EXPECT_EQ(code_of([&] { (void)halo::runtime::create_engine(c); }, what), halo::ErrorCode::Model) << what;
    EXPECT_NE(what.find("attention.head_count"), std::string::npos) << what;
    EXPECT_NE(what.find("does not match the trunk"), std::string::npos) << what;
}

// The other malformed MTP files, at model level (header-only: cheap), plus attaching a second
// MTP to a file that already embeds one.
TEST_F(SplitMtp, MalformedMtpFilesAreRejectedAtLoad) {
    using halo::model::GgufMode;
    using halo::model::NormalizedModel;
    const auto trunk = [] { return NormalizedModel::load(split("trunk"), GgufMode::HeaderOnly); };
    std::string what;
    {
        NormalizedModel m = trunk();
        EXPECT_EQ(code_of([&] { m.attach_mtp(split("mtp-bad-arch"), GgufMode::HeaderOnly); }, what),
                  halo::ErrorCode::Unsupported)
            << what;
        EXPECT_NE(what.find("architecture 'llama'"), std::string::npos) << what;
        EXPECT_EQ(m.mtp_source(), halo::model::MtpSource::None);
    }
    {
        NormalizedModel m = trunk();
        EXPECT_EQ(code_of([&] { m.attach_mtp(split("mtp-with-trunk"), GgufMode::HeaderOnly); }, what),
                  halo::ErrorCode::Model)
            << what;
        EXPECT_NE(what.find("blk.0.attn_norm.weight"), std::string::npos) << what;
        EXPECT_EQ(m.mtp_source(), halo::model::MtpSource::None);
    }
    {
        NormalizedModel m = NormalizedModel::load(single(), GgufMode::HeaderOnly);
        EXPECT_EQ(code_of([&] { m.attach_mtp(split("mtp"), GgufMode::HeaderOnly); }, what), halo::ErrorCode::Config)
            << what;
        EXPECT_EQ(m.mtp_source(), halo::model::MtpSource::Embedded);
    }
    {
        NormalizedModel m = trunk();
        EXPECT_EQ(m.mtp_source(), halo::model::MtpSource::None);
        m.attach_mtp(split("mtp"), GgufMode::HeaderOnly);
        EXPECT_EQ(m.mtp_source(), halo::model::MtpSource::SeparateFile);
    }
}
