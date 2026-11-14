// next.md step 4 on the device: the host round-trip flags of MTP speculative decoding, run by the
// Speculator over the Vulkan backend on the tiny f32 model.
//
//   SpecConfig::device_hidden  (HALO_MTP_DEVICE_HIDDEN)  the MTP hidden stays on the device between depths
//   SpecConfig::chain          (HALO_MTP_CHAIN)          every draft depth in one submit (mtp_draft_chain)
//   Qwen35Options::reuse_streams (HALO_STREAM_REUSE)     pooled backend streams instead of one per forward
//
// and next.md step 6, Qwen35Options::fuse_gemv (HALO_FUSE_GEMV): gate|up, beta|alpha and q|k|v
// loaded row-concatenated as one gemv each. Every W row is reduced by its own workgroup in a fixed
// order, so fused rows must equal the separate gemvs' bit for bit, at R == 1 (decode, one fused
// dispatch for qkv) and R > 1 (prefill / MTP verify, row slices of the fused buffer).
//
// None of them may change what greedy decoding emits: with each flag (and combined), for k = 1..3,
// the drafts of every tick must equal the per-depth host path's drafts bit for bit, the emitted
// tokens must equal plain greedy decoding (MTP off), and the sequence state (trunk KV, GDN state,
// carried hidden, MTP KV after a flush) must equal the plain sequence's exactly.
//
// On this random tiny model the real MTP almost never agrees with the trunk, so acceptance stays
// near 0 and the drafts themselves carry the signal: a wrong device hidden or a wrong chained
// embedding lookup changes depth >= 2 drafts, which the per-tick draft comparison catches.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "backend/vulkan_adapter.h"
#include "halo/backend/backend.h"
#include "halo/kv_cache/paged_kv.h"
#include "halo/models/qwen35.h"
#include "halo/speculative/speculative.h"
#include "models/tiny_golden.h"
#include "vulkan/vk_test_util.h"

namespace hb = halo::backend;
namespace hm = halo::models;
using halo::kv_cache::KvPool;
using halo::speculative::GateMode;
using halo::speculative::SpecConfig;
using halo::speculative::Speculator;
using halo::speculative::StepRequest;
using halo::speculative::Tick;
using halo::state::SequenceState;

namespace {

using Toks = std::vector<std::int32_t>;
constexpr std::size_t k_blocks = 512;
constexpr std::size_t k_slots = 4;  // k + 1 GDN slots for k <= 3
constexpr std::size_t k_tokens = 16;

/// The tiny model over one Vulkan backend, with its own pools.
struct VkModel {
    std::unique_ptr<hb::Backend> be;
    std::unique_ptr<hm::Qwen35> model;
    std::unique_ptr<KvPool> kv_pool, mtp_pool;

    VkModel(const std::shared_ptr<halo::vulkan::Context>& ctx, const halo::model::NormalizedModel& nm, bool reuse)
        : VkModel(ctx, nm, hm::Qwen35Options{.reuse_streams = reuse, .fuse_gemv = ""}) {}
    VkModel(const std::shared_ptr<halo::vulkan::Context>& ctx, const halo::model::NormalizedModel& nm, const hm::Qwen35Options& o)
        : be(hb::make_vulkan_backend(ctx)) {
        model = std::make_unique<hm::Qwen35>(nm, *be, o);
        kv_pool = std::make_unique<KvPool>(model->kv_layout(), k_blocks);
        mtp_pool = std::make_unique<KvPool>(model->mtp_kv_layout(), k_blocks);
    }
    [[nodiscard]] std::unique_ptr<SequenceState> seq() const {
        return std::make_unique<SequenceState>(*kv_pool, mtp_pool.get(), model->gdn_shape(), k_slots);
    }
    [[nodiscard]] Speculator spec(GateMode mode, bool device_hidden, bool chain) const {
        SpecConfig c;
        c.max_draft = 3;
        c.gate.mode = mode;
        c.device_hidden = device_hidden;
        c.chain = chain;
        return Speculator(*model, c);
    }
};

struct Flags {
    const char* name;
    bool device_hidden;
    bool chain;
};
constexpr Flags k_host{"host per-depth", false, false};
constexpr Flags k_variants[] = {{"device_hidden", true, false}, {"chain", false, true}, {"chain+device_hidden", true, true}};

/// Emitted tokens plus the drafts each tick verified, in order.
struct SpecRun {
    Toks tokens;
    std::vector<Toks> drafts;
};

Tick step1(Speculator& sp, const StepRequest& r) {
    Tick t;
    sp.step(std::span(&r, 1), t);
    return t;
}

/// Greedy generation of n tokens (the first from the prefill), k drafts per tick.
SpecRun generate(Speculator& sp, SequenceState& s, std::span<const std::int32_t> prompt, std::size_t n, std::size_t k) {
    SpecRun run;
    StepRequest pre;
    pre.seq = &s;
    pre.prefill = prompt;
    run.tokens = step1(sp, pre).out.at(0).tokens;
    while (run.tokens.size() < n) {
        StepRequest r;
        r.seq = &s;
        r.token = run.tokens.back();
        r.max_draft = k;
        r.max_emit = n - run.tokens.size();
        const Tick t = step1(sp, r);
        if (t.out.at(0).tokens.empty()) {
            ADD_FAILURE() << "tick " << run.drafts.size() << " emitted no token";
            break;
        }
        run.drafts.push_back(t.out[0].drafts);
        run.tokens.insert(run.tokens.end(), t.out[0].tokens.begin(), t.out[0].tokens.end());
    }
    return run;
}

/// The sequence's K and V rows [0, length) of every layer. The pool is device-resident on Vulkan,
/// so this reads the logical block-major image (KvPool::read_all: block, then layer, then K|V,
/// then the block's token rows) through the sequence's block table.
std::vector<float> dump_kv(const halo::kv_cache::SequenceKv& kv) {
    const std::vector<float> image = kv.pool().read_all();
    const auto& l = kv.pool().layout();
    const std::size_t block_floats = 2 * l.n_layers * l.block_tokens * l.kv_dim;
    EXPECT_EQ(image.size(), block_floats * kv.pool().total_blocks()) << "unexpected fp32 KV image size";
    std::vector<float> out;
    for (std::size_t layer = 0; layer < l.n_layers; ++layer) {
        for (std::size_t half = 0; half < 2; ++half) {  // 0 = K, 1 = V
            for (std::size_t r = 0; r < kv.length(); ++r) {
                const auto block = static_cast<std::size_t>(kv.blocks()[r / l.block_tokens]);
                const std::size_t at = block * block_floats + (layer * 2 + half) * l.block_tokens * l.kv_dim + (r % l.block_tokens) * l.kv_dim;
                if (at + l.kv_dim > image.size()) {
                    ADD_FAILURE() << "KV row " << r << " of layer " << layer << " lies outside the pool image";
                    return out;
                }
                out.insert(out.end(), image.begin() + static_cast<std::ptrdiff_t>(at),
                           image.begin() + static_cast<std::ptrdiff_t>(at + l.kv_dim));
            }
        }
    }
    return out;
}

/// Bitwise equality of two sequences' state; each side's MTP queue is flushed by its own speculator.
void expect_same_state(Speculator& sa, SequenceState& a, Speculator& sb, SequenceState& b, const std::string& what) {
    SequenceState* pa[] = {&a};
    SequenceState* pb[] = {&b};
    sa.flush_mtp(pa);
    sb.flush_mtp(pb);
    EXPECT_EQ(a.length(), b.length()) << what << ": sequence length";
    EXPECT_TRUE(dump_kv(a.kv) == dump_kv(b.kv)) << what << ": trunk KV rows differ from the reference";
    EXPECT_TRUE(a.gdn.snapshot().data == b.gdn.snapshot().data) << what << ": GDN recurrent/conv state differs";
    EXPECT_TRUE(a.last_hidden == b.last_hidden) << what << ": carried trunk hidden differs";
    ASSERT_TRUE(a.mtp_kv && b.mtp_kv) << what;
    EXPECT_EQ(a.mtp_kv->length(), b.mtp_kv->length()) << what << ": MTP KV length";
    EXPECT_TRUE(dump_kv(*a.mtp_kv) == dump_kv(*b.mtp_kv)) << what << ": MTP KV rows differ from the reference";
}

std::string str(const Toks& t) {
    std::string s = "[";
    for (std::size_t i = 0; i < t.size(); ++i) s += (i ? " " : "") + std::to_string(t[i]);
    return s + "]";
}

/// First tick whose drafts differ, or -1.
std::ptrdiff_t first_draft_diff(const std::vector<Toks>& a, const std::vector<Toks>& b) {
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (a[i] != b[i]) return static_cast<std::ptrdiff_t>(i);
    return a.size() == b.size() ? -1 : static_cast<std::ptrdiff_t>(std::min(a.size(), b.size()));
}

class VulkanSpec : public ::testing::Test {
protected:
    void SetUp() override {
        golden_ = halo::test::Golden::load();
        if (!golden_) GTEST_SKIP() << "golden data missing under " << halo::test::tiny_dir();
        const auto p = halo::test::tiny_dir() / "tiny-f32.gguf";
        if (!std::filesystem::exists(p)) GTEST_SKIP() << "tiny model missing: " << p;
        nm_ = std::make_unique<halo::model::NormalizedModel>(halo::model::NormalizedModel::load(p));
    }
    [[nodiscard]] Toks prompt(const char* p) const { return golden_->i32(std::string(p) + ".tokens"); }

    std::optional<halo::test::Golden> golden_;
    std::unique_ptr<halo::model::NormalizedModel> nm_;
};

}  // namespace

// The per-depth host path is the reference; each flag must reproduce its drafts and plain greedy exactly.
TEST_F(VulkanSpec, DeviceHiddenAndChainedDraftsEqualTheHostPathAndPlainGreedy) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel vk(ctx, *nm_, /*reuse=*/false);
    ASSERT_TRUE(vk.model->has_mtp()) << "tiny-f32.gguf has no MTP head";
    ASSERT_TRUE(vk.be->limits().lm_head_ids) << "the Vulkan backend must support device token ids, or chain silently falls back";
    ASSERT_FALSE(vk.model->reuse_streams());
    for (const char* p : {"p0", "p1", "p2"}) {
        SCOPED_TRACE(p);
        const Toks pr = prompt(p);
        auto off = vk.spec(GateMode::Off, false, false);
        auto so = vk.seq();
        const SpecRun plain = generate(off, *so, pr, k_tokens, 0);
        ASSERT_EQ(plain.tokens.size(), k_tokens);
        for (std::size_t k = 1; k <= 3; ++k) {
            SCOPED_TRACE("k=" + std::to_string(k));
            auto host = vk.spec(GateMode::Always, k_host.device_hidden, k_host.chain);
            ASSERT_FALSE(*host.config().chain || *host.config().device_hidden) << "explicit false must override the env";
            auto sh = vk.seq();
            const SpecRun ref = generate(host, *sh, pr, k_tokens, k);
            EXPECT_EQ(ref.tokens, plain.tokens) << "host-path MTP greedy " << str(ref.tokens) << " != plain " << str(plain.tokens);
            EXPECT_GT(host.metrics().drafted, 0u) << "MTP drafting never ran";
            expect_same_state(host, *sh, off, *so, std::string(k_host.name));
            for (const Flags& f : k_variants) {
                SCOPED_TRACE(f.name);
                auto sp = vk.spec(GateMode::Always, f.device_hidden, f.chain);
                ASSERT_EQ(*sp.config().chain, f.chain);
                ASSERT_EQ(*sp.config().device_hidden, f.device_hidden);
                auto s = vk.seq();
                const SpecRun got = generate(sp, *s, pr, k_tokens, k);
                EXPECT_EQ(got.tokens, plain.tokens) << f.name << " greedy " << str(got.tokens) << " != plain " << str(plain.tokens);
                const auto d = first_draft_diff(got.drafts, ref.drafts);
                EXPECT_EQ(d, -1) << f.name << ": drafts of tick " << d << " differ from the host path: got "
                                 << (d >= 0 && std::size_t(d) < got.drafts.size() ? str(got.drafts[d]) : "<none>") << ", host "
                                 << (d >= 0 && std::size_t(d) < ref.drafts.size() ? str(ref.drafts[d]) : "<none>");
                EXPECT_EQ(sp.metrics().drafted, host.metrics().drafted) << f.name << ": drafted count";
                expect_same_state(sp, *s, off, *so, f.name);
            }
            std::printf("[%s k=%zu] ticks %zu, drafted %llu, accepted %llu\n", p, k, ref.drafts.size(),
                        static_cast<unsigned long long>(host.metrics().drafted),
                        static_cast<unsigned long long>(host.metrics().accepted));
        }
    }
}

// Pooled streams (one model, many forwards reusing the same command pools, fences and descriptor
// pools) must give the same tokens, drafts and state as a model that creates a stream per forward.
TEST_F(VulkanSpec, StreamReuseIsBitwiseNeutral) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel fresh(ctx, *nm_, /*reuse=*/false);
    const VkModel pooled(ctx, *nm_, /*reuse=*/true);
    ASSERT_TRUE(pooled.model->reuse_streams()) << "Qwen35Options::reuse_streams = true was not applied";
    ASSERT_FALSE(fresh.model->reuse_streams()) << "Qwen35Options::reuse_streams = false must override HALO_STREAM_REUSE";
    for (const char* p : {"p0", "p1", "p2"}) {
        SCOPED_TRACE(p);
        const Toks pr = prompt(p);
        for (const Flags& f : {k_host, k_variants[2]}) {
            for (const std::size_t k : {std::size_t{0}, std::size_t{3}}) {
                SCOPED_TRACE(std::string(f.name) + " k=" + std::to_string(k));
                const GateMode mode = k == 0 ? GateMode::Off : GateMode::Always;
                auto sf = fresh.spec(mode, f.device_hidden, f.chain);
                auto sp = pooled.spec(mode, f.device_hidden, f.chain);
                auto qf = fresh.seq();
                auto qp = pooled.seq();
                const SpecRun a = generate(sf, *qf, pr, k_tokens, k);
                const SpecRun b = generate(sp, *qp, pr, k_tokens, k);
                EXPECT_EQ(b.tokens, a.tokens) << "pooled " << str(b.tokens) << " != fresh " << str(a.tokens);
                const auto d = first_draft_diff(b.drafts, a.drafts);
                EXPECT_EQ(d, -1) << "pooled drafts of tick " << d << " differ from fresh streams";
                expect_same_state(sp, *qp, sf, *qf, "pooled vs fresh");
            }
        }
    }
}

// next.md step 6: each fused projection form (and all three) must give the same tokens, drafts and
// sequence state as the separate gemvs, with MTP off (R == 1 decode ticks) and k = 3 (R = 4 verify
// forwards); the prompt prefill runs R > 1 in both.
TEST_F(VulkanSpec, FusedProjectionsAreBitwiseTheSeparateGemvs) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel plain(ctx, *nm_, hm::Qwen35Options{.reuse_streams = false, .fuse_gemv = ""});
    const hm::FusedProjections none = plain.model->fused_projections();
    ASSERT_FALSE(none.ffn || none.ba || none.qkv) << "fuse_gemv = \"\" must override HALO_FUSE_GEMV and load no fused form";
    struct Fuse {
        const char* spec;
        hm::FusedProjections expect;
    };
    const Fuse fusions[] = {{"ffn", {.ffn = true}}, {"ba", {.ba = true}}, {"qkv", {.qkv = true}},
                            {"all", {.ffn = true, .ba = true, .qkv = true}}};
    for (const Fuse& fu : fusions) {
        SCOPED_TRACE(std::string("fuse_gemv=") + fu.spec);
        const VkModel fused(ctx, *nm_, hm::Qwen35Options{.reuse_streams = false, .fuse_gemv = fu.spec});
        const hm::FusedProjections got = fused.model->fused_projections();
        ASSERT_EQ(got.ffn, fu.expect.ffn) << fu.spec << ": fused gate|up loaded";
        ASSERT_EQ(got.ba, fu.expect.ba) << fu.spec << ": fused beta|alpha loaded (tiny model needs GDN layers)";
        ASSERT_EQ(got.qkv, fu.expect.qkv) << fu.spec << ": fused q|k|v loaded (tiny model needs attention layers)";
        for (const char* p : {"p0", "p1", "p2"}) {
            SCOPED_TRACE(p);
            const Toks pr = prompt(p);
            for (const std::size_t k : {std::size_t{0}, std::size_t{3}}) {
                SCOPED_TRACE("k=" + std::to_string(k));
                const GateMode mode = k == 0 ? GateMode::Off : GateMode::Always;
                auto sa = plain.spec(mode, false, false);
                auto sb = fused.spec(mode, false, false);
                auto qa = plain.seq();
                auto qb = fused.seq();
                const SpecRun a = generate(sa, *qa, pr, k_tokens, k);
                const SpecRun b = generate(sb, *qb, pr, k_tokens, k);
                ASSERT_EQ(a.tokens.size(), k_tokens);
                EXPECT_EQ(b.tokens, a.tokens) << fu.spec << " greedy " << str(b.tokens) << " != separate gemvs " << str(a.tokens);
                const auto d = first_draft_diff(b.drafts, a.drafts);
                EXPECT_EQ(d, -1) << fu.spec << ": drafts of tick " << d << " differ from the separate gemvs: got "
                                 << (d >= 0 && std::size_t(d) < b.drafts.size() ? str(b.drafts[d]) : "<none>") << ", separate "
                                 << (d >= 0 && std::size_t(d) < a.drafts.size() ? str(a.drafts[d]) : "<none>");
                expect_same_state(sb, *qb, sa, *qa, std::string("fused ") + fu.spec + " vs separate");
            }
        }
    }
}
