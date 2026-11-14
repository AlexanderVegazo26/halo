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
// And next.md step 5, SpecConfig::tree (HALO_MTP_TREE): the verify forward carries [x, d1..dk, d1']
// with d1' a second child of x, attended through attention_tree.comp (virtual keys in plain-decode
// order), and a kept runner-up moves its KV row and GDN ring slot into place. Greedy output and
// the committed state must equal plain decoding exactly, whichever branch wins.
//
// And next.md step 2, Qwen35Options::mtp_draft_vocab (HALO_MTP_DRAFT_VOCAB): an Argmax MTP step
// scores only the first N token rows of the MTP head. Each draft must be exactly the argmax of the
// full head's logits over [0, min(N, valid_vocab)) (lowest index on ties), a Full step must still get
// the whole head, and the chained drafts must equal the per-depth path's under the same N.
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
#include <array>
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
        : VkModel(ctx, nm, hm::Qwen35Options{.reuse_streams = reuse, .fuse_gemv = "", .mtp_draft_vocab = 0}) {}
    VkModel(const std::shared_ptr<halo::vulkan::Context>& ctx, const halo::model::NormalizedModel& nm, const hm::Qwen35Options& o)
        : be(hb::make_vulkan_backend(ctx)) {
        model = std::make_unique<hm::Qwen35>(nm, *be, o);
        kv_pool = std::make_unique<KvPool>(model->kv_layout(), k_blocks);
        mtp_pool = std::make_unique<KvPool>(model->mtp_kv_layout(), k_blocks);
    }
    [[nodiscard]] std::unique_ptr<SequenceState> seq(std::size_t slots = k_slots) const {
        return std::make_unique<SequenceState>(*kv_pool, mtp_pool.get(), model->gdn_shape(), slots);
    }
    [[nodiscard]] Speculator spec(GateMode mode, bool device_hidden, bool chain) const {
        SpecConfig c;
        c.max_draft = 3;
        c.gate.mode = mode;
        c.device_hidden = device_hidden;
        c.chain = chain;
        return Speculator(*model, c);
    }
    /// Tree verification on (next.md step 5); the sequence needs max_draft + 2 GDN slots for the leaf.
    /// chain stays off explicitly: HALO_MTP_CHAIN drafts yield no runner-up, so tree_ok refuses them.
    [[nodiscard]] Speculator spec_tree(std::size_t max_draft, bool device_hidden) const {
        SpecConfig c;
        c.max_draft = max_draft;
        c.gate.mode = GateMode::Always;
        c.device_hidden = device_hidden;
        c.chain = false;
        c.tree = true;
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

/// Plain decode of `toks` one at a time (the reference history); returns the argmaxes.
Toks feed_plain(Speculator& sp, SequenceState& s, std::span<const std::int32_t> toks) {
    Toks out;
    for (const std::int32_t t : toks) {
        StepRequest r;
        r.seq = &s;
        r.token = t;
        out.push_back(step1(sp, r).out.at(0).tokens.at(0));
    }
    return out;
}

/// Prefill `p` (the first generated token is discarded: the caller already knows it).
void prefill(Speculator& sp, SequenceState& s, std::span<const std::int32_t> p) {
    StepRequest r;
    r.seq = &s;
    r.prefill = p;
    (void)step1(sp, r);
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

// ---- next.md step 5: tree verification (SpecConfig::tree, HALO_MTP_TREE) on the device -------

// Greedy generation with trees on (real MTP drafts plus their runner-up leaves) equals plain greedy
// decoding, token for token and state for state, for k = 1..3, with the host or device hidden. On
// this random tiny model the chain is almost always rejected at d1, so most ticks also test whether
// the runner-up d1' is the trunk's argmax: either way the leaf row went through attention_tree.comp.
TEST_F(VulkanSpec, TreeGreedyEqualsPlainGreedy) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel vk(ctx, *nm_, /*reuse=*/false);
    ASSERT_TRUE(vk.model->has_mtp()) << "tiny-f32.gguf has no MTP head";
    ASSERT_EQ(vk.kv_pool->layout().type, hb::KvType::F32) << "attention_tree.comp is fp32-only: tree_ok refuses other KV";
    for (const char* p : {"p0", "p1", "p2"}) {
        SCOPED_TRACE(p);
        const Toks pr = prompt(p);
        auto off = vk.spec(GateMode::Off, false, false);
        auto so = vk.seq();
        const SpecRun plain = generate(off, *so, pr, k_tokens, 0);
        ASSERT_EQ(plain.tokens.size(), k_tokens);
        for (const bool dh : {false, true}) {
            for (std::size_t k = 1; k <= 3; ++k) {
                SCOPED_TRACE(std::string(dh ? "device_hidden" : "host hidden") + " k=" + std::to_string(k));
                auto on = vk.spec_tree(k, dh);
                ASSERT_TRUE(on.config().tree);
                ASSERT_FALSE(*on.config().chain) << "explicit chain = false must override HALO_MTP_CHAIN";
                auto s = vk.seq(k + 2);
                const SpecRun got = generate(on, *s, pr, k_tokens, k);
                EXPECT_EQ(got.tokens, plain.tokens) << "tree greedy " << str(got.tokens) << " != plain " << str(plain.tokens);
                EXPECT_GT(on.metrics().tree_steps, 0u) << "no tick verified a tree on Vulkan (tree_ok refused?)";
                expect_same_state(on, *s, off, *so, "tree vs plain");
                std::printf("[%s %s k=%zu] ticks %zu, tree steps %llu, leaf kept %llu, accepted %llu\n", p,
                            dh ? "dh" : "host", k, got.drafts.size(), static_cast<unsigned long long>(on.metrics().tree_steps),
                            static_cast<unsigned long long>(on.metrics().tree_leaf_kept),
                            static_cast<unsigned long long>(on.metrics().accepted));
            }
        }
    }
    // A ring with no room for the leaf state (k + 1 slots) silently verifies a plain chain, still identical.
    const Toks pr = prompt("p1");
    auto off = vk.spec(GateMode::Off, false, false);
    auto so = vk.seq();
    const SpecRun plain = generate(off, *so, pr, k_tokens, 0);
    auto on = vk.spec_tree(2, false);
    auto s = vk.seq(3);
    EXPECT_EQ(generate(on, *s, pr, k_tokens, 2).tokens, plain.tokens) << "k + 1 slots: plain-chain fallback";
    EXPECT_EQ(on.metrics().tree_steps, 0u) << "a tree was verified without a GDN slot for its leaf";
}

// The runner-up branch on the device: force d1 wrong and d1' == the trunk's true next token. verify
// must accept the leaf and emit [t0, argmax(leaf row)] -- the leaf row's logits come from
// attention_tree.comp seeing [history, x, d1'] only -- and commit must leave the sequence
// bit-identical to plain decoding [x, d1'] (KV row moved from L+k+1 to L+1, GDN state = the leaf's
// ring slot, hidden of the leaf row). Keeping only x (n_keep = 1) is decode-1.
TEST_F(VulkanSpec, TreeRunnerUpBranchIsKeptAndEqualsPlainDecode) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel vk(ctx, *nm_, /*reuse=*/false);
    const std::size_t V = vk.model->n_vocab();
    for (const char* p : {"p0", "p1"}) {  // p0: 150 tokens, the history crosses KV blocks
        SCOPED_TRACE(p);
        const Toks pr = prompt(p);
        auto off = vk.spec(GateMode::Off, false, false);
        auto gs = vk.seq();
        const Toks g = generate(off, *gs, pr, 12, 0).tokens;
        for (const std::size_t kK : {std::size_t{1}, std::size_t{3}}) {
            for (const std::size_t keep : {std::size_t{2}, std::size_t{1}}) {
                SCOPED_TRACE("k=" + std::to_string(kK) + " keep=" + std::to_string(keep));
                const std::size_t m = 3;  // pending token g[m], history = prompt + g[0..m)
                auto sp = vk.spec_tree(kK, false);
                auto s = vk.seq(kK + 2);
                prefill(sp, *s, pr);
                (void)feed_plain(sp, *s, std::span(g).first(m));
                StepRequest q;
                q.seq = s.get();
                q.token = g[m];
                q.max_draft = kK;
                const std::span<const StepRequest> reqs(&q, 1);
                Tick t;
                sp.draft(reqs, t);
                ASSERT_TRUE(t.seqs.at(0).tree) << "the tree was not armed on Vulkan";
                ASSERT_EQ(t.out.at(0).drafts.size(), kK);
                const std::int32_t truth = g[m + 1];
                t.out[0].drafts[0] = static_cast<std::int32_t>((static_cast<std::size_t>(truth) + 1) % V);  // d1 wrong
                t.seqs[0].alt = truth;                                                                     // d1' right
                sp.verify(reqs, t);
                ASSERT_TRUE(t.seqs[0].alt_win) << "verify did not take the runner-up d1' == " << truth;
                EXPECT_EQ(t.out[0].accepted, 1u);
                EXPECT_EQ(t.out[0].n_keep, 2u);
                EXPECT_EQ(t.out[0].tokens, (Toks{g[m + 1], g[m + 2]}))
                    << "leaf row argmax (attention_tree.comp) != plain decode's next token";
                const std::array<std::size_t, 1> nk = {keep};
                sp.commit(reqs, t, keep == 2 ? std::span<const std::size_t>() : std::span<const std::size_t>(nk));
                EXPECT_EQ(s->length(), pr.size() + m + keep);
                EXPECT_EQ(sp.metrics().tree_leaf_kept, keep == 2 ? 1u : 0u);
                auto ref = vk.seq();
                auto off2 = vk.spec(GateMode::Off, false, false);
                prefill(off2, *ref, pr);
                (void)feed_plain(off2, *ref, std::span(g).first(m + keep));  // x, and d1' == g[m+1] when kept
                expect_same_state(sp, *s, off2, *ref, "kept leaf vs plain decode");
                // The next tick continues correctly from the committed state.
                StepRequest q2;
                q2.seq = s.get();
                q2.token = keep == 2 ? g[m + 2] : g[m + 1];
                q2.max_draft = kK;
                const Tick t2 = step1(sp, q2);
                EXPECT_EQ(t2.out.at(0).tokens.at(0), keep == 2 ? g[m + 3] : g[m + 2]) << "tick after the commit";
            }
        }
    }
}

// The runner-up is never kept when it would be wrong: d1 right (the chain wins, all k + 1 tokens
// kept), or d1 and d1' both wrong (one token, plain decode).
TEST_F(VulkanSpec, TreeChainWinsOrBothWrongEqualsPlainDecode) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel vk(ctx, *nm_, /*reuse=*/false);
    const std::size_t V = vk.model->n_vocab();
    constexpr std::size_t kK = 2;
    const Toks pr = prompt("p1");
    auto off = vk.spec(GateMode::Off, false, false);
    auto gs = vk.seq();
    const Toks g = generate(off, *gs, pr, k_tokens, 0).tokens;
    const std::size_t m = 4;
    for (const bool chain_right : {true, false}) {
        SCOPED_TRACE(chain_right ? "chain right" : "both wrong");
        auto sp = vk.spec_tree(kK, false);
        auto s = vk.seq(kK + 2);
        prefill(sp, *s, pr);
        (void)feed_plain(sp, *s, std::span(g).first(m));
        StepRequest q;
        q.seq = s.get();
        q.token = g[m];
        q.max_draft = kK;
        const std::span<const StepRequest> reqs(&q, 1);
        Tick t;
        sp.draft(reqs, t);
        ASSERT_TRUE(t.seqs.at(0).tree);
        Toks d(g.begin() + static_cast<std::ptrdiff_t>(m + 1), g.begin() + static_cast<std::ptrdiff_t>(m + 1 + kK));
        if (!chain_right) d[0] = static_cast<std::int32_t>((static_cast<std::size_t>(d[0]) + 1) % V);
        t.out.at(0).drafts = d;
        t.seqs[0].alt = static_cast<std::int32_t>((static_cast<std::size_t>(g[m + 1]) + 2) % V);  // wrong, distinct from a forged d1
        sp.verify(reqs, t);
        EXPECT_FALSE(t.seqs[0].alt_win) << "a wrong runner-up was kept";
        sp.commit(reqs, t);
        const std::size_t kept = chain_right ? kK + 1 : 1;
        EXPECT_EQ(t.out[0].tokens, Toks(g.begin() + static_cast<std::ptrdiff_t>(m + 1), g.begin() + static_cast<std::ptrdiff_t>(m + 1 + kept)));
        auto ref = vk.seq();
        auto off2 = vk.spec(GateMode::Off, false, false);
        prefill(off2, *ref, pr);
        (void)feed_plain(off2, *ref, std::span(g).first(m + kept));
        expect_same_state(sp, *s, off2, *ref, chain_right ? "chain win vs plain" : "both wrong vs plain");
    }
}

// ---- next.md step 2: reduced MTP draft vocabulary (Qwen35Options::mtp_draft_vocab) on the device ----

namespace {

hm::Qwen35Options draft_vocab_options(std::size_t n, std::size_t valid_vocab = 0) {
    return hm::Qwen35Options{.valid_vocab = valid_vocab, .reuse_streams = false, .fuse_gemv = "", .mtp_draft_vocab = n};
}

/// Index of the first maximum of row[0, n) (the LM head's tie rule: lowest index wins).
std::int32_t prefix_argmax(std::span<const float> row, std::size_t n) {
    return static_cast<std::int32_t>(std::max_element(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(n)) - row.begin());
}

}  // namespace

// One MTP step over prompt p1's teacher-forced rows (the trunk hidden of the full-vocab model, fed to
// every model so the head inputs are identical): with draft vocab N, an Argmax row's winner and its
// value are the full head's logits row restricted to [0, min(N, valid_vocab)); a Full step on the
// reduced model is the full head, logits bit for bit.
TEST_F(VulkanSpec, ReducedDraftVocabArgmaxIsTheFullHeadsPrefixArgmax) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel full(ctx, *nm_, draft_vocab_options(0));
    ASSERT_TRUE(full.model->has_mtp()) << "tiny-f32.gguf has no MTP head";
    const std::size_t V = full.model->n_vocab();
    const std::size_t E = full.model->n_embd();
    ASSERT_GE(V, 16u) << "tiny vocab too small for prefix cases";
    EXPECT_EQ(full.model->mtp_draft_vocab(), V) << "mtp_draft_vocab = 0 must override HALO_MTP_DRAFT_VOCAB and mean the full head";
    EXPECT_EQ(VkModel(ctx, *nm_, draft_vocab_options(V + 5)).model->mtp_draft_vocab(), V) << "N > n_vocab clamps to the full head";

    // The shared head input: trunk hidden rows of p1, then the MTP step tokens[1..] on hidden[0..n-1).
    const Toks pr = prompt("p1");
    ASSERT_GE(pr.size(), 3u);
    std::vector<float> hidden;
    {
        auto s = full.seq();
        const std::vector<std::size_t> last{pr.size() - 1};
        hm::SeqStep st;
        st.tokens = pr;
        st.kv = &s->kv;
        st.gdn = &s->gdn;
        st.logit_rows = last;
        st.want_hidden = true;
        hm::StepResult r;
        full.model->forward(std::span(&st, 1), r);
        hidden = r.seqs.at(0).hidden;
        ASSERT_EQ(hidden.size(), pr.size() * E);
    }
    const std::size_t n = pr.size() - 1;
    const Toks toks(pr.begin() + 1, pr.end());
    const std::span<const float> h(hidden.data(), n * E);
    std::vector<std::size_t> rows(n);
    for (std::size_t i = 0; i < n; ++i) rows[i] = i;
    const auto mtp = [&](const VkModel& m, hm::LogitsMode mode) {
        auto s = m.seq();
        hm::MtpStep st{.tokens = toks, .hidden = h, .kv = &*s->mtp_kv, .first_position = 1, .logit_rows = rows, .logits = mode};
        hm::StepResult r;
        m.model->mtp_forward(std::span(&st, 1), r);
        return r.seqs.at(0);
    };
    const hm::SeqOutput ref = mtp(full, hm::LogitsMode::Full);
    ASSERT_EQ(ref.logits.size(), n * V);
    const hm::SeqOutput ref_argmax = mtp(full, hm::LogitsMode::Argmax);

    struct Case {
        std::size_t draft, valid;
    };
    const Case cases[] = {{1, 0}, {7, 0}, {V / 3, 0}, {V - 1, 0}, {V / 3, V / 2}, {2 * V / 3, V / 2}};
    for (const Case& c : cases) {
        const std::size_t want_n = c.valid ? std::min(c.draft, c.valid) : c.draft;
        SCOPED_TRACE("draft_vocab=" + std::to_string(c.draft) + " valid_vocab=" + std::to_string(c.valid));
        const VkModel red(ctx, *nm_, draft_vocab_options(c.draft, c.valid));
        ASSERT_EQ(red.model->mtp_draft_vocab(), c.draft) << "Qwen35Options::mtp_draft_vocab was not applied";
        const hm::SeqOutput got = mtp(red, hm::LogitsMode::Argmax);
        ASSERT_EQ(got.argmax.size(), n);
        std::size_t outside = 0;  // rows whose full-head winner lies outside the prefix
        for (std::size_t i = 0; i < n; ++i) {
            const std::span<const float> row(ref.logits.data() + i * V, V);
            const std::int32_t want = prefix_argmax(row, want_n);
            EXPECT_EQ(got.argmax[i].index, want) << "row " << i << " (token " << toks[i] << "): reduced-head draft " << got.argmax[i].index
                                                 << " != argmax of the full logits over [0, " << want_n << ") = " << want;
            if (got.argmax[i].index == want) {
                EXPECT_EQ(got.argmax[i].value, row[static_cast<std::size_t>(want)])
                    << "row " << i << ": winning logit differs from the full head's logit " << want;
            }
            if (static_cast<std::size_t>(ref_argmax.argmax[i].index) >= want_n) ++outside;
        }
        if (want_n <= V / 3) EXPECT_GT(outside, 0u) << "no row's full-head winner lies outside the prefix: the case tests nothing";
        // Full logits ignore the draft vocabulary: the whole head, bit for bit the full model's.
        const hm::SeqOutput all = mtp(red, hm::LogitsMode::Full);
        EXPECT_TRUE(all.logits == ref.logits) << "a Full step on the reduced model must score every token row";
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t lim = c.valid ? c.valid : V;
            EXPECT_EQ(all.argmax[i].index, prefix_argmax(std::span<const float>(ref.logits.data() + i * V, V), lim)) << "Full row " << i;
        }
    }
}

// Speculative decoding with a reduced draft head: every draft lies in the prefix, the chained drafts
// (mtp_draft_chain's own head) equal the per-depth path's, and greedy output and sequence state equal
// plain decoding, since the trunk head verifies every draft.
TEST_F(VulkanSpec, ReducedDraftVocabKeepsGreedyAndChainEqualsHostPath) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const VkModel full(ctx, *nm_, draft_vocab_options(0));
    const std::size_t V = full.model->n_vocab();
    const std::size_t N = std::max<std::size_t>(1, V / 8);
    const VkModel red(ctx, *nm_, draft_vocab_options(N));
    ASSERT_EQ(red.model->mtp_draft_vocab(), N);
    std::size_t changed = 0;  // drafts of the full head that the prefix cannot produce
    for (const char* p : {"p0", "p1", "p2"}) {
        SCOPED_TRACE(p);
        const Toks pr = prompt(p);
        auto off = full.spec(GateMode::Off, false, false);
        auto so = full.seq();
        const SpecRun plain = generate(off, *so, pr, k_tokens, 0);
        ASSERT_EQ(plain.tokens.size(), k_tokens);
        for (std::size_t k = 1; k <= 3; ++k) {
            SCOPED_TRACE("k=" + std::to_string(k));
            auto fh = full.spec(GateMode::Always, false, false);
            auto sf = full.seq();
            for (const Toks& d : generate(fh, *sf, pr, k_tokens, k).drafts)
                changed += static_cast<std::size_t>(std::count_if(d.begin(), d.end(), [&](std::int32_t t) { return std::size_t(t) >= N; }));
            auto host = red.spec(GateMode::Always, false, false);
            auto sh = red.seq();
            const SpecRun ref = generate(host, *sh, pr, k_tokens, k);
            EXPECT_EQ(ref.tokens, plain.tokens) << "reduced-head MTP greedy " << str(ref.tokens) << " != plain " << str(plain.tokens);
            EXPECT_GT(host.metrics().drafted, 0u) << "MTP drafting never ran";
            for (std::size_t t = 0; t < ref.drafts.size(); ++t)
                for (const std::int32_t d : ref.drafts[t])
                    EXPECT_TRUE(d >= 0 && std::size_t(d) < N) << "tick " << t << " drafted " << d << ", outside the first " << N << " rows";
            expect_same_state(host, *sh, off, *so, "reduced head host path vs plain");
            for (const Flags& f : k_variants) {
                SCOPED_TRACE(f.name);
                auto sp = red.spec(GateMode::Always, f.device_hidden, f.chain);
                auto s = red.seq();
                const SpecRun got = generate(sp, *s, pr, k_tokens, k);
                EXPECT_EQ(got.tokens, plain.tokens) << f.name << " greedy " << str(got.tokens) << " != plain " << str(plain.tokens);
                const auto d = first_draft_diff(got.drafts, ref.drafts);
                EXPECT_EQ(d, -1) << f.name << ": drafts of tick " << d << " differ from the reduced-head host path: got "
                                 << (d >= 0 && std::size_t(d) < got.drafts.size() ? str(got.drafts[d]) : "<none>") << ", host "
                                 << (d >= 0 && std::size_t(d) < ref.drafts.size() ? str(ref.drafts[d]) : "<none>");
                expect_same_state(sp, *s, off, *so, std::string("reduced head ") + f.name + " vs plain");
            }
        }
    }
    EXPECT_GT(changed, 0u) << "the full head never drafted outside the first " << N << " rows: the prefix restriction went untested";
    std::printf("[draft vocab %zu of %zu] full-head drafts outside the prefix: %zu\n", N, V, changed);
}
