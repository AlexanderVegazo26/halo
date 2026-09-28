// ADR-001 §6.3 "forward over X == forward over CPU" for X = Vulkan (lavapipe, WS-F2 V5): the one
// models::Qwen35 forward composed over the Vulkan backend against the same forward over the CPU
// backend, on every tiny-*.gguf whose tensor types the Vulkan backend implements. Verified on
// lavapipe only (DECISIONS D-001).
//
// Two sides, each with its own KV pools (filled with a sentinel), sequences and GDN states, run
// the same operation sequence (the §6.2 fixture's dimensions): a 70-token prefill crossing the
// GDN chunk (64) and KV block (16) boundaries; decode; a ragged S = 3 batch mixing prefill and
// decode; a K = 3 verify with state slots followed by commit_rows_kept(3, 2) and truncate; a
// decode after the rollback; an MTP step. Logits modes Full, Argmax and None. GDN path: first
// Recurrent on both sides (GdnPath::Recurrent), then Chunked (GdnPath::Auto, which both
// backends resolve to Chunked for slot-free multi-row steps; asserted).
//
// Pass condition (ADR §6.3 mode (b)), fixed BEFORE the first measurement:
//   - every compared fp32 tensor: relative L2 error ||vk - cpu||_2 / ||cpu||_2 <= tau = 1e-4.
//     Compared after every call: per-layer inputs (capture_layer_inputs), hidden rows, full
//     logits rows, the whole KV pool storage of both pools (sentinel positions must match
//     exactly, i.e. both sides wrote the same rows), and every GDN layer's live recurrent and
//     conv state plus the whole ring slab (every physical slot, ADR-001 §5.3).
//     Derivation: the Vulkan kernels are within ~1e2 u (u = 2^-24, ~6e-6 relative) of the CPU
//     op per op (a-priori bounds, tests/unit/vulkan; observed 1e-3..1e-1 of those bounds). A
//     forward of the tiny model is ~9 layers x ~15 ops; rounding differences propagate through
//     norms (scale-invariant) and non-expansive mixers, so they add as a random walk:
//     sqrt(135) x 6e-6 ~ 7e-5 -> tau = 1e-4. (The worst-case linear sum, ~8e-4, would be too
//     loose to see a small systematic error; the check is a model, and a 1e-4 miss is a
//     finding to investigate, never a tolerance to widen.)
//   - argmax: the same token for every requested row, unless the CPU's top-2 logit margin is
//     within 1e-3 x max|logit| (a near-tie; only decidable for Full rows) — near-ties are
//     printed. Argmax-mode rows must always match.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "backend/vulkan_adapter.h"
#include "halo/backend/backend.h"
#include "halo/backends/vulkan/context.h"
#include "halo/core/error.h"
#include "halo/models/qwen35.h"
#include "models/tiny_golden.h"
#include "vulkan/vk_test_util.h"

namespace hb = halo::backend;
namespace hm = halo::models;
using halo::kv_cache::KvPool;
using halo::test::TestSeq;

namespace {

constexpr double k_tau = 1e-4;
constexpr float k_sentinel = -7.25f;
constexpr std::size_t k_blocks = 64;
constexpr std::size_t k_slots = 4;

/// Weight types the Vulkan backend implements (vulkan::matvec_row_bytes).
const std::set<halo::DType> k_vk_types{halo::DType::F32,    halo::DType::Q8_0,   halo::DType::Q4_K,
                                       halo::DType::Q5_K,   halo::DType::Q6_K,   halo::DType::IQ4_XS,
                                       halo::DType::IQ4_NL, halo::DType::Q3_K,   halo::DType::IQ3_S};

/// The first matrix / embedding type the Vulkan backend does not implement, or empty.
std::string unsupported_type(const halo::model::NormalizedModel& m) {
    std::vector<const halo::model::WeightRef*> w{&m.token_embd(), &m.output()};
    auto layer = [&](const halo::model::LayerWeights& l) {
        for (const auto* r : {&l.ffn_gate, &l.ffn_up, &l.ffn_down}) w.push_back(r);
        if (l.kind == halo::model::LayerKind::FullAttention) {
            for (const auto* r : {&l.attn.q, &l.attn.k, &l.attn.v, &l.attn.output}) w.push_back(r);
        } else {
            for (const auto* r : {&l.gdn.qkv, &l.gdn.gate, &l.gdn.beta, &l.gdn.alpha, &l.gdn.out}) w.push_back(r);
        }
    };
    for (const auto& l : m.layers()) layer(l);
    if (const auto* mtp = m.mtp()) {
        layer(mtp->block);
        for (const auto* r : {&mtp->eh_proj, &mtp->embedding, &mtp->lm_head}) w.push_back(r);
    }
    for (const auto* r : w) {
        if (r->present() && !k_vk_types.contains(r->type())) return std::string(halo::traits(r->type()).name) + " (" + std::string(r->name()) + ")";
    }
    return {};
}

double rel_l2(std::span<const float> vk, std::span<const float> cpu) {
    double e = 0, n = 0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        const double d = double(vk[i]) - double(cpu[i]);
        e += d * d;
        n += double(cpu[i]) * double(cpu[i]);
    }
    if (!std::isfinite(e)) return 1e300;
    return n == 0 ? (e == 0 ? 0 : 1e300) : std::sqrt(e / n);
}

/// One side: a backend, the model over it, pools and sequences.
struct Side {
    std::unique_ptr<hb::Backend> be;
    std::unique_ptr<hm::Qwen35> model;
    std::unique_ptr<KvPool> kv_pool, mtp_pool;
    std::vector<std::unique_ptr<TestSeq>> seqs;

    Side(std::unique_ptr<hb::Backend> b, const halo::model::NormalizedModel& nm) : be(std::move(b)) {
        model = std::make_unique<hm::Qwen35>(nm, *be);
        kv_pool = std::make_unique<KvPool>(model->kv_layout(), k_blocks);
        mtp_pool = std::make_unique<KvPool>(model->mtp_kv_layout(), k_blocks);
        for (KvPool* p : {kv_pool.get(), mtp_pool.get()}) {
            std::fill_n(p->k_rows(0, 0), p->total_blocks() * p->layout().block_floats(), k_sentinel);
        }
        for (std::size_t i = 0; i < 3; ++i) seqs.push_back(std::make_unique<TestSeq>(*kv_pool, *mtp_pool, model->gdn_shape(), k_slots));
    }
};

/// Worst relative L2 over the whole run (printed at the end).
struct Worst {
    double v = 0;
    std::string where;
    void add(double r, const std::string& w) {
        if (r > v) {
            v = r;
            where = w;
        }
    }
};

class Diff {
public:
    Diff(Side& vk, Side& cpu, hm::GdnPath path) : vk_(vk), cpu_(cpu), path_(path) {}

    /// make(Side&) -> steps over that side's sequences; runs both and compares everything.
    template <class Make>
    std::vector<hm::GdnPath> forward(const std::string& what, Make&& make, bool capture = false) {
        SCOPED_TRACE(what);
        std::vector<hm::SeqStep> sv = make(vk_), sc = make(cpu_);
        for (auto* s : {&sv, &sc})
            for (auto& st : *s)
                if (st.gdn_path == hm::GdnPath::Auto && path_ == hm::GdnPath::Recurrent) st.gdn_path = hm::GdnPath::Recurrent;
        hm::StepResult rv, rc;
        const hm::ForwardOptions opts{.capture_layer_inputs = capture};
        vk_.model->forward(sv, rv, opts);
        cpu_.model->forward(sc, rc, opts);
        EXPECT_EQ(rv.gdn_paths, rc.gdn_paths) << what;
        compare_results(what, rv, rc, sc);
        compare_state(what);
        return rc.gdn_paths;
    }

    void mtp(const std::string& what, const std::vector<std::int32_t>& tokens, const std::vector<float>& hidden_vk,
             const std::vector<float>& hidden_cpu, std::size_t seq, std::int32_t first_pos) {
        SCOPED_TRACE(what);
        const std::vector<std::size_t> rows{0};
        auto step = [&](Side& s, const std::vector<float>& h) {
            return std::vector<hm::MtpStep>{hm::MtpStep{tokens, h, &s.seqs[seq]->mtp_kv, first_pos, rows, hm::LogitsMode::Full, true}};
        };
        const auto sv = step(vk_, hidden_vk), sc = step(cpu_, hidden_cpu);
        hm::StepResult rv, rc;
        vk_.model->mtp_forward(sv, rv);
        cpu_.model->mtp_forward(sc, rc);
        std::vector<hm::SeqStep> fake(1);
        fake[0].logit_rows = rows;
        fake[0].logits = hm::LogitsMode::Full;
        compare_results(what, rv, rc, fake);
        check("mtp pool", rel_l2_masked(pool_span(*vk_.mtp_pool), pool_span(*cpu_.mtp_pool)));
    }

    /// The same host-side state operation on both sides.
    void both(const std::function<void(Side&)>& f) {
        f(vk_);
        f(cpu_);
    }

    [[nodiscard]] const Worst& worst() const { return worst_; }
    std::size_t near_ties = 0;

private:
    static std::span<const float> pool_span(const KvPool& p) {
        return {p.k_rows(0, 0), p.total_blocks() * p.layout().block_floats()};
    }

    /// Sentinel positions must match exactly; relative L2 over the written ones.
    double rel_l2_masked(std::span<const float> vk, std::span<const float> cpu) {
        std::vector<float> a, b;
        std::size_t mismatched = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            const bool sc = std::bit_cast<std::uint32_t>(cpu[i]) == std::bit_cast<std::uint32_t>(k_sentinel);
            const bool sv = std::bit_cast<std::uint32_t>(vk[i]) == std::bit_cast<std::uint32_t>(k_sentinel);
            if (sc != sv) ++mismatched;
            if (!sc) {
                a.push_back(vk[i]);
                b.push_back(cpu[i]);
            }
        }
        EXPECT_EQ(mismatched, 0u) << "KV rows written on one side only";
        return rel_l2(a, b);
    }

    void check(const std::string& what, double r) {
        worst_.add(r, what);
        EXPECT_LE(r, k_tau) << what << ": relative L2 " << r;
    }

    void compare_results(const std::string& what, const hm::StepResult& rv, const hm::StepResult& rc,
                         const std::vector<hm::SeqStep>& steps) {
        ASSERT_EQ(rv.seqs.size(), rc.seqs.size());
        ASSERT_EQ(rv.layer_inputs.size(), rc.layer_inputs.size());
        for (std::size_t l = 0; l < rc.layer_inputs.size(); ++l) {
            check(what + " layer " + std::to_string(l) + " input", rel_l2(rv.layer_inputs[l], rc.layer_inputs[l]));
        }
        for (std::size_t s = 0; s < rc.seqs.size(); ++s) {
            const hm::SeqOutput& v = rv.seqs[s];
            const hm::SeqOutput& c = rc.seqs[s];
            ASSERT_EQ(v.hidden.size(), c.hidden.size());
            if (!c.hidden.empty()) check(what + " seq " + std::to_string(s) + " hidden", rel_l2(v.hidden, c.hidden));
            ASSERT_EQ(v.logits.size(), c.logits.size());
            ASSERT_EQ(v.argmax.size(), c.argmax.size());
            const std::size_t nvoc = vk_.model->n_vocab();
            for (std::size_t j = 0; j < c.argmax.size(); ++j) {
                const std::string row = what + " seq " + std::to_string(s) + " row " + std::to_string(j);
                if (!c.logits.empty()) {
                    const std::span<const float> lc(c.logits.data() + j * nvoc, nvoc), lv(v.logits.data() + j * nvoc, nvoc);
                    check(row + " logits", rel_l2(lv, lc));
                }
                if (v.argmax[j].index == c.argmax[j].index) continue;
                bool near_tie = false;
                if (!c.logits.empty()) {
                    std::vector<float> l(c.logits.begin() + static_cast<std::ptrdiff_t>(j * nvoc),
                                         c.logits.begin() + static_cast<std::ptrdiff_t>((j + 1) * nvoc));
                    std::partial_sort(l.begin(), l.begin() + 2, l.end(), std::greater<>());
                    double mx = 0;
                    for (float x : l) mx = std::max(mx, std::abs(double(x)));
                    near_tie = double(l[0]) - double(l[1]) <= 1e-3 * mx;
                }
                if (near_tie) {
                    ++near_ties;
                    std::cout << "[vk-forward] near-tie " << row << ": vk " << v.argmax[j].index << " cpu " << c.argmax[j].index << "\n";
                } else {
                    ADD_FAILURE() << row << ": vk token " << v.argmax[j].index << " != cpu token " << c.argmax[j].index
                                  << (steps[s].logits == hm::LogitsMode::Full ? "" : " (argmax-only row)");
                }
            }
        }
    }

    void compare_state(const std::string& what) {
        check(what + " trunk KV pool", rel_l2_masked(pool_span(*vk_.kv_pool), pool_span(*cpu_.kv_pool)));
        const auto shape = vk_.model->gdn_shape();
        for (std::size_t si = 0; si < vk_.seqs.size(); ++si) {
            auto& gv = vk_.seqs[si]->gdn;
            auto& gc = cpu_.seqs[si]->gdn;
            EXPECT_EQ(vk_.seqs[si]->kv.length(), cpu_.seqs[si]->kv.length());
            EXPECT_EQ(gv.live(), gc.live()) << what << " seq " << si << " ring live";
            // The Vulkan slab is device-resident: refresh its host mirror, then compare the
            // whole ring (live state + every physical slot, ADR-001 §5.3).
            gv.pull();
            const std::size_t p = gv.ring_size();
            for (std::size_t l = 0; l < shape.n_layers; ++l) {
                const std::string pr = what + " seq " + std::to_string(si) + " gdn layer " + std::to_string(l);
                check(pr + " recurrent", rel_l2(gv.recurrent(l), gc.recurrent(l)));
                const auto cv = gv.conv(l), cc = gc.conv(l);
                check(pr + " conv", rel_l2(std::span<const float>(cv.data(), cv.rows() * cv.cols()),
                                           std::span<const float>(cc.data(), cc.rows() * cc.cols())));
                check(pr + " recurrent ring", rel_l2(gv.slab().subspan(l * p * shape.recurrent_floats(),
                                                                       p * shape.recurrent_floats()),
                                                       gc.slab().subspan(l * p * shape.recurrent_floats(),
                                                                         p * shape.recurrent_floats())));
                const std::size_t cb = shape.n_layers * p * shape.recurrent_floats() + l * p * shape.conv_floats();
                check(pr + " conv ring", rel_l2(gv.slab().subspan(cb, p * shape.conv_floats()),
                                                gc.slab().subspan(cb, p * shape.conv_floats())));
            }
        }
    }

    Side& vk_;
    Side& cpu_;
    hm::GdnPath path_;
    Worst worst_;
};

std::vector<std::int32_t> tokens(std::size_t n, std::size_t vocab, std::uint32_t seed) {
    std::vector<std::int32_t> t(n);
    std::uint32_t x = seed * 2654435761u + 12345u;
    for (auto& v : t) {
        x = x * 1664525u + 1013904223u;
        v = static_cast<std::int32_t>((x >> 8) % vocab);
    }
    return t;
}

struct Case {
    std::string file;
    hm::GdnPath path;
};

void PrintTo(const Case& c, std::ostream* os) { *os << c.file << (c.path == hm::GdnPath::Recurrent ? " recurrent" : " chunked"); }

class VulkanForward : public ::testing::TestWithParam<Case> {};

}  // namespace

TEST_P(VulkanForward, MatchesCpuBackend) {
    HALO_VK_CONTEXT_OR_SKIP(ctx);
    const Case& c = GetParam();
    const auto p = halo::test::tiny_dir() / c.file;
    if (!std::filesystem::exists(p)) GTEST_SKIP() << "tiny model missing: " << p;
    const auto nm = halo::model::NormalizedModel::load(p);
    if (const std::string t = unsupported_type(nm); !t.empty()) {
        GTEST_SKIP() << c.file << ": weight type " << t << " is not implemented by the Vulkan backend";
    }
    Side vk(hb::make_vulkan_backend(ctx), nm);
    Side cpu(hb::make_cpu_backend(nullptr), nm);
    const std::size_t V = vk.model->n_vocab();
    Diff d(vk, cpu, c.path);
    const bool chunked = c.path != hm::GdnPath::Recurrent;

    // 1. Prefill 70 rows on seq 0 (GDN chunk 64 + partial, KV blocks of 16 + partial).
    const auto t0 = tokens(70, V, 1);
    const std::vector<std::size_t> last{69};
    const auto paths = d.forward("prefill 70", [&](Side& s) {
        return std::vector<hm::SeqStep>{hm::SeqStep{t0, {}, &s.seqs[0]->kv, &s.seqs[0]->gdn, last, hm::LogitsMode::Full, true, 0,
                                                    hm::GdnPath::Auto}};
    }, /*capture=*/true);
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_EQ(paths[0], chunked ? hm::GdnPath::Chunked : hm::GdnPath::Recurrent) << "the prefill ran the intended GDN form";

    // 2. Decode 3 tokens.
    for (int i = 0; i < 3; ++i) {
        const auto t = tokens(1, V, 10 + i);
        const std::vector<std::size_t> r0{0};
        d.forward("decode " + std::to_string(i), [&](Side& s) {
            return std::vector<hm::SeqStep>{hm::SeqStep{t, {}, &s.seqs[0]->kv, &s.seqs[0]->gdn, r0, hm::LogitsMode::Argmax, false, 0,
                                                        hm::GdnPath::Auto}};
        });
    }

    // 3. Ragged S = 3: prefill 20 (no logits), decode 1 (Full), prefill 5 (Argmax rows 0, 4).
    const auto ta = tokens(20, V, 20), tb = tokens(1, V, 21), tc = tokens(5, V, 22);
    const std::vector<std::size_t> none{}, r0{0}, r04{0, 4};
    d.forward("ragged S=3", [&](Side& s) {
        return std::vector<hm::SeqStep>{
            hm::SeqStep{ta, {}, &s.seqs[1]->kv, &s.seqs[1]->gdn, none, hm::LogitsMode::None, false, 0, hm::GdnPath::Auto},
            hm::SeqStep{tb, {}, &s.seqs[0]->kv, &s.seqs[0]->gdn, r0, hm::LogitsMode::Full, false, 0, hm::GdnPath::Auto},
            hm::SeqStep{tc, {}, &s.seqs[2]->kv, &s.seqs[2]->gdn, r04, hm::LogitsMode::Argmax, false, 0, hm::GdnPath::Auto}};
    });

    // 4. Verify K = 3 with slots (the recurrent path on both), keep 2 rows.
    const auto tv = tokens(3, V, 30);
    const std::vector<std::size_t> all3{0, 1, 2};
    std::size_t len0 = 0;
    d.both([&](Side& s) { len0 = s.seqs[0]->kv.length(); });
    d.forward("verify K=3", [&](Side& s) {
        return std::vector<hm::SeqStep>{hm::SeqStep{tv, {}, &s.seqs[0]->kv, &s.seqs[0]->gdn, all3, hm::LogitsMode::Full, false, 3,
                                                    hm::GdnPath::Auto}};
    });
    d.both([&](Side& s) {
        s.seqs[0]->kv.truncate(len0 + 2);
        s.seqs[0]->gdn.commit_rows_kept(3, 2);
    });

    // 5. Decode after the rollback (hidden kept for the MTP step).
    const auto td = tokens(1, V, 40);
    std::vector<float> hv_, hc_;
    {
        const std::vector<std::size_t> rr{0};
        hm::StepResult rv, rc;
        std::vector<hm::SeqStep> sv{hm::SeqStep{td, {}, &vk.seqs[0]->kv, &vk.seqs[0]->gdn, rr, hm::LogitsMode::Full, true, 0,
                                                chunked ? hm::GdnPath::Auto : hm::GdnPath::Recurrent}};
        std::vector<hm::SeqStep> sc{hm::SeqStep{td, {}, &cpu.seqs[0]->kv, &cpu.seqs[0]->gdn, rr, hm::LogitsMode::Full, true, 0,
                                                chunked ? hm::GdnPath::Auto : hm::GdnPath::Recurrent}};
        vk.model->forward(sv, rv);
        cpu.model->forward(sc, rc);
        ASSERT_EQ(rv.seqs[0].argmax.size(), 1u);
        EXPECT_EQ(rv.seqs[0].argmax[0].index, rc.seqs[0].argmax[0].index) << "decode after rollback";
        EXPECT_LE(rel_l2(rv.seqs[0].hidden, rc.seqs[0].hidden), k_tau) << "decode after rollback hidden";
        EXPECT_LE(rel_l2(rv.seqs[0].logits, rc.seqs[0].logits), k_tau) << "decode after rollback logits";
        hv_ = rv.seqs[0].hidden;
        hc_ = rc.seqs[0].hidden;
    }

    // 6. MTP step on seq 0 (the separate MTP KV pool).
    if (vk.model->has_mtp()) {
        const auto tm = tokens(1, V, 50);
        d.mtp("mtp", tm, hv_, hc_, 0, static_cast<std::int32_t>(vk.seqs[0]->kv.length()));
    }

    std::cout << "[vk-forward] " << c.file << (chunked ? " chunked" : " recurrent") << ": worst relative L2 = " << d.worst().v
              << " (" << d.worst().where << "), near-ties " << d.near_ties << ", tau = " << k_tau << "\n";
}

INSTANTIATE_TEST_SUITE_P(Tiny, VulkanForward,
                         ::testing::Values(Case{"tiny-f32.gguf", hm::GdnPath::Recurrent}, Case{"tiny-f32.gguf", hm::GdnPath::Auto},
                                           Case{"tiny-q8_0.gguf", hm::GdnPath::Recurrent}, Case{"tiny-q8_0.gguf", hm::GdnPath::Auto},
                                           Case{"tiny-q6_k.gguf", hm::GdnPath::Recurrent}, Case{"tiny-q6_k.gguf", hm::GdnPath::Auto},
                                           Case{"tiny-iq4_xs.gguf", hm::GdnPath::Recurrent},
                                           Case{"tiny-iq4_xs.gguf", hm::GdnPath::Auto},
                                           Case{"tiny-q4_k_m.gguf", hm::GdnPath::Auto}, Case{"tiny-q3_k_m.gguf", hm::GdnPath::Auto}),
                         [](const auto& info) {
                             std::string n = info.param.file.substr(5, info.param.file.size() - 10);
                             std::replace(n.begin(), n.end(), '.', '_');
                             return n + (info.param.path == hm::GdnPath::Recurrent ? "_recurrent" : "_chunked");
                         });
