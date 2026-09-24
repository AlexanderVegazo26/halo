// GATED_DELTANET (recurrent + chunked), CONV1D_SHORT and GATED_NORM of the HIP backend,
// differentially tested against halo::cpu on the same inputs (TRD §30, D-016, review M-2).
//
//   Emu*    : the host emulation of the device kernels (same bodies, same tiling and
//             reduction order), run in forward and reverse thread/workgroup order; each run
//             must be BIT-IDENTICAL to halo::cpu. This verifies the kernels' math and
//             indexing on the dev host. It does not verify device scheduling, the memory
//             model, wave-level behaviour or the device expf.
//   Device* : the same scenarios launched on a HIP device, compared against halo::cpu
//             within kDevAbs/kDevRel. Skipped on the dev host (D-001).

#include <algorithm>
#include <cmath>
#include <limits>

#include "halo/backends/cpu/ops.h"
#include "hip_test_util.h"

namespace halo::hip::test {
namespace {

// ---------------------------------------------------------------------------------------
// GATED_DELTANET scenario
// ---------------------------------------------------------------------------------------

struct GdnCase {
    std::uint32_t n_k = 2, n_v = 6, d_k = 16, d_v = 16;
    std::uint32_t T = 1;
    std::uint32_t n_slots = 0;
    bool grouped = false;
    bool l2 = true;
    std::optional<float> q_scale{};
    bool in_place = true;
    bool fused = false;          // q/k/v strided views into one qkv row; g/beta interleaved
    bool chunked = false;
    std::uint32_t chunk = 64;
    std::uint32_t chunks_per_group = 1u << 30;  // workspace size in chunk groups
    std::string variant{};       // empty = default
    std::uint32_t seed = 1;
    float g_override = 0.0f;     // chunked error cases: set g[bad_t][bad_j] to this
    bool poison_g = false;
    std::uint32_t bad_t = 0, bad_j = 0;
};

struct GdnData {
    std::uint64_t qk_cols, v_cols, row;  // row = fused row length
    std::vector<float> qkv;              // fused: [T, row] (q | k | v); else unused
    std::vector<float> q, k, v;          // non-fused
    std::vector<float> gb;               // [T, 2*n_v] interleaved (g | beta) when fused
    std::vector<float> g, beta;
    std::vector<float> state, state_out, slots, out;
};

GdnData make_gdn_data(const GdnCase& c) {
    std::mt19937 rng(c.seed);
    GdnData d;
    d.qk_cols = static_cast<std::uint64_t>(c.n_k) * c.d_k;
    d.v_cols = static_cast<std::uint64_t>(c.n_v) * c.d_v;
    d.row = 2 * d.qk_cols + d.v_cols;
    // Without in-kernel normalization keep |k|^2 small so the delta rule stays contractive.
    const float qk_amp = c.l2 ? 1.0f : 0.8f / std::sqrt(static_cast<float>(c.d_k));
    auto q = uniform(rng, c.T * d.qk_cols, -qk_amp, qk_amp);
    auto k = uniform(rng, c.T * d.qk_cols, -qk_amp, qk_amp);
    auto v = uniform(rng, c.T * d.v_cols, -1.0f, 1.0f);
    auto g = uniform(rng, static_cast<std::size_t>(c.T) * c.n_v, -1.5f, 0.0f);
    auto beta = uniform(rng, static_cast<std::size_t>(c.T) * c.n_v, 0.0f, 1.0f);
    if (c.T > 2) g[static_cast<std::size_t>(1) * c.n_v] = 0.0f;  // exp(0) = 1 edge
    if (c.g_override != 0.0f || c.poison_g) {
        g[static_cast<std::size_t>(c.bad_t) * c.n_v + c.bad_j] =
            c.poison_g ? std::numeric_limits<float>::quiet_NaN() : c.g_override;
    }
    if (c.fused) {
        d.qkv.resize(c.T * d.row);
        d.gb.resize(static_cast<std::size_t>(c.T) * 2 * c.n_v);
        for (std::uint32_t t = 0; t < c.T; ++t) {
            std::copy_n(&q[t * d.qk_cols], d.qk_cols, &d.qkv[t * d.row]);
            std::copy_n(&k[t * d.qk_cols], d.qk_cols, &d.qkv[t * d.row + d.qk_cols]);
            std::copy_n(&v[t * d.v_cols], d.v_cols, &d.qkv[t * d.row + 2 * d.qk_cols]);
            std::copy_n(&g[t * c.n_v], c.n_v, &d.gb[t * 2 * c.n_v]);
            std::copy_n(&beta[t * c.n_v], c.n_v, &d.gb[t * 2 * c.n_v + c.n_v]);
        }
    } else {
        d.q = std::move(q);
        d.k = std::move(k);
        d.v = std::move(v);
        d.g = std::move(g);
        d.beta = std::move(beta);
    }
    const std::size_t state_n = d.v_cols * c.d_k;
    d.state = uniform(rng, state_n, -0.5f, 0.5f);  // a continuing sequence
    d.state_out.assign(c.in_place ? 0 : state_n, 777.0f);
    d.slots.assign(state_n * c.n_slots, -555.0f);  // sentinel: untouched slots must keep it
    d.out.assign(c.T * d.v_cols, 999.0f);
    return d;
}

/// The CPU reference on its own copy of the data. Returns false if it threw (the caller
/// checks that the HIP op raised the same error).
bool run_cpu(const GdnCase& c, GdnData& d) {
    cpu::GdnDims dims{c.n_k, c.n_v, c.d_k, c.d_v, c.grouped ? cpu::GdnHeadMapping::Grouped : cpu::GdnHeadMapping::Tiled};
    cpu::GdnInputs in;
    if (c.fused) {
        in.q = cpu::ConstRows(d.qkv.data(), c.T, d.qk_cols, d.row);
        in.k = cpu::ConstRows(d.qkv.data() + d.qk_cols, c.T, d.qk_cols, d.row);
        in.v = cpu::ConstRows(d.qkv.data() + 2 * d.qk_cols, c.T, d.v_cols, d.row);
        in.g = cpu::ConstRows(d.gb.data(), c.T, c.n_v, 2 * c.n_v);
        in.beta = cpu::ConstRows(d.gb.data() + c.n_v, c.T, c.n_v, 2 * c.n_v);
    } else {
        in.q = cpu::ConstRows(d.q.data(), c.T, d.qk_cols, d.qk_cols);
        in.k = cpu::ConstRows(d.k.data(), c.T, d.qk_cols, d.qk_cols);
        in.v = cpu::ConstRows(d.v.data(), c.T, d.v_cols, d.v_cols);
        in.g = cpu::ConstRows(d.g.data(), c.T, c.n_v, c.n_v);
        in.beta = cpu::ConstRows(d.beta.data(), c.T, c.n_v, c.n_v);
    }
    // The CPU op is in place; an out-of-place HIP call is compared against the CPU's
    // final state, and its input state must stay unchanged.
    std::vector<float> st = d.state;
    cpu::Rows out(d.out.data(), c.T, d.v_cols, d.v_cols);
    const cpu::GdnQkParams qk{.qk_l2norm = c.l2, .q_scale = c.q_scale};
    try {
        if (c.chunked) {
            cpu::gated_delta_rule_chunked(dims, in, st, out, qk, c.chunk, nullptr, d.slots);
        } else {
            cpu::gated_delta_rule_recurrent(dims, in, st, out, qk, nullptr, d.slots);
        }
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::Kernel);
        return false;
    }
    if (c.in_place) {
        d.state = st;
    } else {
        d.state_out = st;
    }
    return true;
}

/// Runs the HIP op; returns false if it raised Error(Kernel) (after check_status).
bool run_hip(const GdnCase& c, GdnData& d, Runner& r) {
    std::vector<Buffer> keep;
    keep.reserve(16);  // stable addresses: views hold Buffer pointers
    GdnArgs a;
    a.n_k = c.n_k;
    a.n_v = c.n_v;
    a.d_k = c.d_k;
    a.d_v = c.d_v;
    a.n_tokens = c.T;
    a.mapping = c.grouped ? GdnHeadMapping::Grouped : GdnHeadMapping::Tiled;
    a.qk_l2norm = c.l2;
    a.q_scale = c.q_scale;
    if (c.fused) {
        const std::uint64_t rs = d.row * sizeof(float);
        const Buffer& qkv = keep.emplace_back(r.make(d.qkv));
        a.q = BufferView(qkv, 0, 0, rs);
        a.k = BufferView(qkv, d.qk_cols * sizeof(float), 0, rs);
        a.v = BufferView(qkv, 2 * d.qk_cols * sizeof(float), 0, rs);
        const Buffer& gb = keep.emplace_back(r.make(d.gb));
        a.g = BufferView(gb, 0, 0, 2 * c.n_v * sizeof(float));
        a.beta = BufferView(gb, c.n_v * sizeof(float), 0, 2 * c.n_v * sizeof(float));
    } else {
        a.q = keep.emplace_back(r.make(d.q));
        a.k = keep.emplace_back(r.make(d.k));
        a.v = keep.emplace_back(r.make(d.v));
        a.g = keep.emplace_back(r.make(d.g));
        a.beta = keep.emplace_back(r.make(d.beta));
    }
    const Buffer& st = keep.emplace_back(r.make(d.state));
    a.state = st;
    const Buffer* st_out = nullptr;
    if (!c.in_place) {
        st_out = &keep.emplace_back(r.make(d.state_out));
        a.state_out = BufferView(*st_out);
    }
    const Buffer* slots = nullptr;
    if (c.n_slots > 0) {
        slots = &keep.emplace_back(r.make(d.slots));
        a.state_slots = *slots;
        a.n_slots = c.n_slots;
    }
    const Buffer& out = keep.emplace_back(r.make(d.out));
    a.out = out;
    const Ops ops = [&] {
        OpsOptions o;
        if (!c.variant.empty()) (c.chunked ? o.gdn_chunked : o.gdn_recurrent) = c.variant;
        return Ops(o);
    }();
    bool ok = true;
    if (c.chunked) {
        std::vector<float> ws(gdn_chunked_workspace_bytes(a, c.chunk, c.chunks_per_group) / sizeof(float), 0.0f);
        std::vector<std::uint32_t> status_host{0xDEADBEEFu};  // stale garbage: the op must zero it
        const Buffer& wsb = keep.emplace_back(r.make(ws));
        const Buffer& sb = keep.emplace_back(r.make_words(status_host));
        GdnChunkedArgs ca{a, c.chunk, BufferView(wsb), BufferView(sb)};
        ops.gated_delta_rule_chunked(r.target(), ca);
        r.finish();
        try {
            Ops::check_status(r.target(), BufferView(sb));
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Kernel);
            ok = false;
        }
    } else {
        ops.gated_delta_rule_recurrent(r.target(), a);
        r.finish();
    }
    r.fetch(st, d.state);
    if (st_out != nullptr) r.fetch(*st_out, d.state_out);
    if (slots != nullptr) r.fetch(*slots, d.slots);
    r.fetch(out, d.out);
    return ok;
}

void check_gdn(const GdnCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    GdnData ref = make_gdn_data(c);
    const GdnData orig = ref;
    GdnData got = ref;
    const bool cpu_ok = run_cpu(c, ref);
    const bool hip_ok = run_hip(c, got, r);
    ASSERT_EQ(cpu_ok, hip_ok) << "CPU and HIP disagree on raising Error(Kernel)";
    if (!cpu_ok) {
        // Nothing may be written when the op raises (CPU: validated before writing; HIP:
        // the kernels skip on the status bit).
        EXPECT_TRUE(matches(orig.out, got.out, true, "out untouched"));
        EXPECT_TRUE(matches(orig.state, got.state, true, "state untouched"));
        EXPECT_TRUE(matches(orig.state_out, got.state_out, true, "state_out untouched"));
        EXPECT_TRUE(matches(orig.slots, got.slots, true, "slots untouched"));
        return;
    }
    EXPECT_TRUE(matches(ref.out, got.out, r.exact(), "out"));
    EXPECT_TRUE(matches(ref.state, got.state, r.exact(), c.in_place ? "state (in place)" : "state (input, unchanged)"));
    if (!c.in_place) EXPECT_TRUE(matches(ref.state_out, got.state_out, r.exact(), "state_out"));
    if (c.n_slots > 0) EXPECT_TRUE(matches(ref.slots, got.slots, r.exact(), "state_slots"));
}

void check_gdn_emulated(const GdnCase& c) {
    for (auto& r : emulation_runners()) check_gdn(c, *r);
}

void check_gdn_device(const GdnCase& c) {
    DeviceRunner r;
    check_gdn(c, r);
}

// Real qwen35 dims (D-003): 16 key heads, 48 value heads, d_k = d_v = 128.
GdnCase real_dims(GdnCase c) {
    c.n_k = 16;
    c.n_v = 48;
    c.d_k = 128;
    c.d_v = 128;
    return c;
}

// ---- recurrent ------------------------------------------------------------------------

std::vector<GdnCase> recurrent_cases() {
    std::vector<GdnCase> v;
    v.push_back(real_dims({.T = 1, .seed = 11}));                                           // decode step
    v.push_back(real_dims({.T = 5, .n_slots = 3, .in_place = false, .fused = true, .seed = 12}));  // MTP verify
    v.push_back({.T = 3, .n_slots = 5, .seed = 13});                                        // slots s >= T untouched
    v.push_back({.n_k = 2, .n_v = 6, .d_k = 20, .d_v = 150, .T = 7, .n_slots = 2, .grouped = true, .l2 = false,
                 .q_scale = 0.37f, .in_place = false, .variant = "gdn_recurrent_b64", .seed = 14});  // odd dims, 3 column blocks
    v.push_back({.n_k = 1, .n_v = 3, .d_k = 128, .d_v = 100, .T = 9, .q_scale = 1.0f, .fused = true,
                 .variant = "gdn_recurrent_b64", .seed = 15});
    v.push_back({.n_k = 4, .n_v = 4, .d_k = 7, .d_v = 5, .T = 4, .n_slots = 4, .seed = 16});  // d_k < 8: dot tail only
    return v;
}

TEST(HipGdnEmu, RecurrentBitExactVsCpu) {
    for (const GdnCase& c : recurrent_cases()) {
        SCOPED_TRACE(testing::Message() << "n_k " << c.n_k << " n_v " << c.n_v << " d_k " << c.d_k << " d_v " << c.d_v
                                        << " T " << c.T << " slots " << c.n_slots << " seed " << c.seed);
        check_gdn_emulated(c);
    }
}

TEST(HipGdnDevice, RecurrentVsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    for (const GdnCase& c : recurrent_cases()) check_gdn_device(c);
}

// ---- chunked --------------------------------------------------------------------------

std::vector<GdnCase> chunked_cases() {
    std::vector<GdnCase> v;
    v.push_back(real_dims({.T = 70, .n_slots = 2, .chunked = true, .seed = 21}));  // partial 2nd chunk
    v.push_back({.T = 37, .n_slots = 8, .fused = true, .chunked = true, .chunk = 16, .seed = 22});  // slots straddle a chunk boundary
    // Multi-group workspace (groups of 2 chunks => 4 groups), in place and out of place,
    // slots straddling the group boundary at row 96.
    v.push_back({.T = 100, .n_slots = 8, .chunked = true, .chunk = 16, .chunks_per_group = 2, .seed = 23});
    v.push_back({.T = 100, .n_slots = 8, .in_place = false, .chunked = true, .chunk = 16, .chunks_per_group = 2,
                 .seed = 24});
    v.push_back({.n_k = 2, .n_v = 6, .d_k = 20, .d_v = 150, .T = 45, .grouped = true, .l2 = false, .q_scale = 0.5f,
                 .in_place = false, .chunked = true, .chunk = 8, .chunks_per_group = 1, .variant = "gdn_chunked_b32",
                 .seed = 25});
    v.push_back({.T = 1, .n_slots = 3, .chunked = true, .seed = 26});  // one row
    v.push_back({.n_k = 4, .n_v = 4, .d_k = 7, .d_v = 5, .T = 12, .chunked = true, .chunk = 5, .seed = 27});
    return v;
}

TEST(HipGdnEmu, ChunkedBitExactVsCpu) {
    for (const GdnCase& c : chunked_cases()) {
        SCOPED_TRACE(testing::Message() << "n_k " << c.n_k << " n_v " << c.n_v << " d_k " << c.d_k << " d_v " << c.d_v
                                        << " T " << c.T << " chunk " << c.chunk << " groups of " << c.chunks_per_group
                                        << " slots " << c.n_slots << " seed " << c.seed);
        check_gdn_emulated(c);
    }
}

TEST(HipGdnDevice, ChunkedVsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    for (const GdnCase& c : chunked_cases()) check_gdn_device(c);
}

// Positive or NaN g: the CPU raises Error(Kernel) before writing; the HIP op flags the
// status word and writes nothing (out, state, state_out, slots bit-for-bit untouched).
std::vector<GdnCase> bad_g_cases() {
    return {
        {.T = 40, .n_slots = 2, .chunked = true, .chunk = 16, .chunks_per_group = 1, .seed = 31,
         .g_override = 0.25f, .bad_t = 39, .bad_j = 5},  // in the last group only
        {.T = 40, .in_place = false, .chunked = true, .chunk = 16, .seed = 32, .poison_g = true, .bad_t = 3,
         .bad_j = 0},
    };
}

TEST(HipGdnEmu, ChunkedPositiveOrNanGRaisesAndWritesNothing) {
    for (const GdnCase& c : bad_g_cases()) check_gdn_emulated(c);
}

TEST(HipGdnDevice, ChunkedPositiveOrNanGRaisesAndWritesNothing) {
    HALO_REQUIRE_HIP_DEVICE();
    for (const GdnCase& c : bad_g_cases()) check_gdn_device(c);
}

// A failed call leaves the status bit set; the next valid call on the same status buffer
// must zero it and run normally (a stale bit must not turn later calls into no-ops).
void status_reuse(Runner& r) {
    GdnCase c{.T = 20, .chunked = true, .chunk = 8, .seed = 41};
    GdnData d = make_gdn_data(c);
    GdnData ref = d;
    ASSERT_TRUE(run_cpu(c, ref));
    std::vector<float> bad_g = d.g;
    bad_g[7] = 1.0f;
    std::vector<float> ws(gdn_chunked_workspace_bytes(GdnArgs{.n_k = c.n_k, .n_v = c.n_v, .d_k = c.d_k, .d_v = c.d_v,
                                                              .n_tokens = c.T},
                                                      c.chunk, 99) /
                              sizeof(float),
                          0.0f);
    std::vector<std::uint32_t> status{0};
    Buffer q = r.make(d.q), k = r.make(d.k), v = r.make(d.v), g = r.make(d.g), gbad = r.make(bad_g),
           beta = r.make(d.beta), st = r.make(d.state), out = r.make(d.out), wsb = r.make(ws),
           sb = r.make_words(status);
    GdnArgs a{.q = q, .k = k, .v = v, .g = gbad, .beta = beta, .state = st, .out = out, .n_k = c.n_k, .n_v = c.n_v,
              .d_k = c.d_k, .d_v = c.d_v, .n_tokens = c.T};
    const Ops ops;
    ops.gated_delta_rule_chunked(r.target(), GdnChunkedArgs{a, c.chunk, wsb, sb});
    r.finish();
    EXPECT_THROW(Ops::check_status(r.target(), sb), Error);
    a.g = g;
    ops.gated_delta_rule_chunked(r.target(), GdnChunkedArgs{a, c.chunk, wsb, sb});
    r.finish();
    EXPECT_NO_THROW(Ops::check_status(r.target(), sb));
    r.fetch(out, d.out);
    r.fetch(st, d.state);
    EXPECT_TRUE(matches(ref.out, d.out, r.exact(), "out after status reuse"));
    EXPECT_TRUE(matches(ref.state, d.state, r.exact(), "state after status reuse"));
}

TEST(HipGdnEmu, StatusWordIsResetByEachCall) {
    for (auto& r : emulation_runners()) status_reuse(*r);
}

TEST(HipGdnDevice, StatusWordIsResetByEachCall) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    status_reuse(r);
}

// TRD §29/§30, review S-5: a long trajectory (T = 4096, chunk 64 = 64 chunk-boundary
// crossings, and 4096 recurrent steps) — output rows and the final state bit-identical to
// halo::cpu. Reduced head count (2 value heads of 128 x 128) to keep the host run short.
TEST(HipGdnEmu, LongTrajectory4096BitExactVsCpu) {
    const GdnCase chunked{.n_k = 1, .n_v = 2, .d_k = 128, .d_v = 128, .T = 4096, .n_slots = 2, .fused = true,
                          .chunked = true, .chunks_per_group = 16, .seed = 51};
    check_gdn_emulated(chunked);
    GdnCase recurrent = chunked;
    recurrent.chunked = false;
    recurrent.seed = 52;
    check_gdn_emulated(recurrent);
}

TEST(HipGdnDevice, LongTrajectory4096VsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    const GdnCase chunked = real_dims({.T = 4096, .n_slots = 2, .fused = true, .chunked = true, .seed = 53});
    check_gdn_device(chunked);
    GdnCase recurrent = chunked;
    recurrent.chunked = false;
    check_gdn_device(recurrent);
}

// ---- contract divergences of this backend (documented in ops.h / docs/hip.md) --------

TEST(HipGdnEmu, RejectsUnsupportedShapes) {
    std::vector<float> f(1u << 16, 0.0f);
    std::vector<std::uint32_t> w{0};
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * sizeof(float));
    std::vector<float> f2(1u << 16, 0.0f);
    const Buffer b2 = Buffer::wrap_host(f2.data(), f2.size() * sizeof(float));
    const Buffer sb = Buffer::wrap_host(w.data(), sizeof(std::uint32_t));
    const Ops ops;
    const Target t = Target::emulation();
    auto expect_kernel_error = [](const char* needle, auto&& fn) {
        try {
            fn();
            ADD_FAILURE() << "no error; expected: " << needle;
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Kernel) << e.what();
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    // d_k above the register-resident cap.
    GdnArgs a{.q = BufferView(b, 0, 4096), .k = BufferView(b, 4096, 4096), .v = BufferView(b, 8192, 4096),
              .g = BufferView(b, 12288, 16), .beta = BufferView(b, 12304, 16), .state = BufferView(b2, 0, 256),
              .out = BufferView(b, 16384, 4096), .n_k = 1, .n_v = 1, .d_k = 129, .d_v = 1};
    expect_kernel_error("register-resident limit", [&] { ops.gated_delta_rule_recurrent(t, a); });
    a.d_k = 8;
    a.d_v = 8;
    ASSERT_NO_THROW(ops.gated_delta_rule_recurrent(t, a));  // the baseline below is valid
    // chunk_size above the LDS tile cap (the CPU op accepts up to 1024).
    expect_kernel_error("chunk_size 65", [&] {
        ops.gated_delta_rule_chunked(t, GdnChunkedArgs{a, 65, BufferView(b2, 1024), sb});
    });
    // state_out partially overlapping state.
    a.state_out = BufferView(b2, 128, 256);
    expect_kernel_error("state_out partially overlaps state", [&] { ops.gated_delta_rule_recurrent(t, a); });
    a.state_out.reset();
    // out overlapping v.
    a.out = BufferView(b, 8192 + 16, 256);
    expect_kernel_error("out overlaps v", [&] { ops.gated_delta_rule_recurrent(t, a); });
    // Misaligned offset.
    a.out = BufferView(b, 16386, 256);
    expect_kernel_error("not 4-byte aligned", [&] { ops.gated_delta_rule_recurrent(t, a); });
    // View too small for T rows.
    a.out = BufferView(b, 16384, 256);
    a.n_tokens = 2;
    a.g = BufferView(b, 12288, 4);
    expect_kernel_error("g needs 8 bytes", [&] { ops.gated_delta_rule_recurrent(t, a); });
    // Slot count without a slot buffer.
    a.g = BufferView(b, 12288, 16);
    a.n_slots = 1;
    expect_kernel_error("n_slots 1 but state_slots is empty", [&] { ops.gated_delta_rule_recurrent(t, a); });
}

// ---------------------------------------------------------------------------------------
// CONV1D_SHORT
// ---------------------------------------------------------------------------------------

struct ConvCase {
    std::uint32_t T = 1, C = 64, K = 4, n_slots = 0;
    bool alias = false;     // out == x
    std::uint64_t x_pad = 0;  // extra elements per x row (strided view)
    std::string variant{};
    std::uint32_t seed = 1;
};

void check_conv(const ConvCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "T " << c.T << " C " << c.C << " K " << c.K << " slots " << c.n_slots);
    std::mt19937 rng(c.seed);
    const std::uint64_t xs = c.C + c.x_pad;
    std::vector<float> x = uniform(rng, c.T * xs, -2.0f, 2.0f);
    std::vector<float> w = uniform(rng, static_cast<std::size_t>(c.C) * c.K, -1.0f, 1.0f);
    std::vector<float> st = uniform(rng, static_cast<std::size_t>(c.K - 1) * c.C + 1, -1.0f, 1.0f);  // +1: never empty
    std::vector<float> out(c.alias ? 1 : c.T * c.C, 999.0f);
    std::vector<float> slots(static_cast<std::size_t>(c.n_slots) * (c.K - 1) * c.C + 1, -555.0f);

    auto rx = x, rst = st, rout = out, rslots = slots;
    {
        cpu::Rows xr(rx.data(), c.T, c.C, xs);
        cpu::Rows yr = c.alias ? xr : cpu::Rows(rout.data(), c.T, c.C, c.C);
        cpu::Rows sr(rst.data(), c.K - 1, c.C, c.C);
        cpu::causal_conv1d_silu(xr, cpu::ConstRows(w.data(), c.C, c.K, c.K), sr, yr, nullptr,
                                std::span<float>(rslots.data(), rslots.size() - 1));
    }
    const Buffer bx = r.make(x), bw = r.make(w), bst = r.make(st), bout = r.make(out), bsl = r.make(slots);
    Conv1dArgs a;
    a.x = BufferView(bx, 0, 0, xs * sizeof(float));
    a.weight = bw;
    if (c.K > 1) a.conv_state = BufferView(bst, 0, static_cast<std::uint64_t>(c.K - 1) * c.C * sizeof(float));
    a.out = c.alias ? a.x : BufferView(bout);
    if (c.n_slots > 0) {
        a.state_slots = BufferView(bsl, 0, (slots.size() - 1) * sizeof(float));
        a.n_slots = c.n_slots;
    }
    a.n_tokens = c.T;
    a.channels = c.C;
    a.kernel = c.K;
    OpsOptions o;
    if (!c.variant.empty()) o.conv1d = c.variant;
    Ops(o).causal_conv1d_silu(r.target(), a);
    r.finish();
    r.fetch(bx, x);
    r.fetch(bst, st);
    r.fetch(bout, out);
    r.fetch(bsl, slots);
    EXPECT_TRUE(matches(rx, x, r.exact(), c.alias ? "out (aliasing x)" : "x (unchanged)"));
    EXPECT_TRUE(matches(rout, out, r.exact(), "out"));
    EXPECT_TRUE(matches(rst, st, r.exact(), "conv_state"));
    EXPECT_TRUE(matches(rslots, slots, r.exact(), "state_slots"));
}

std::vector<ConvCase> conv_cases() {
    return {
        {.T = 1, .C = 10240, .K = 4, .seed = 61},                                    // qwen35 decode step
        {.T = 9, .C = 300, .K = 4, .n_slots = 4, .alias = true, .seed = 62},         // MTP verify, in place
        {.T = 3, .C = 70, .K = 2, .n_slots = 5, .x_pad = 13, .variant = "conv1d_silu_b64", .seed = 63},
        {.T = 5, .C = 33, .K = 1, .seed = 64},                                       // no history
        {.T = 17, .C = 129, .K = 8, .n_slots = 2, .seed = 65},
    };
}

TEST(HipConvEmu, BitExactVsCpu) {
    for (const ConvCase& c : conv_cases()) {
        for (auto& r : emulation_runners()) check_conv(c, *r);
    }
}

// Divergence from the CPU op: this backend's conv keeps the history in registers (K <= 8).
TEST(HipConvEmu, RejectsKernelSizeOutsideOneToEight) {
    std::vector<float> f(4096, 0.0f);
    const Buffer b = Buffer::wrap_host(f.data(), f.size() * sizeof(float));
    Conv1dArgs a;
    a.x = BufferView(b, 0, 64);             // T 1 x C 16
    a.out = BufferView(b, 64, 64);
    a.conv_state = BufferView(b, 1024, 1024);
    a.weight = BufferView(b, 2048, 1024);
    a.n_tokens = 1;
    a.channels = 16;
    a.kernel = 8;
    const Ops ops;
    ASSERT_NO_THROW(ops.causal_conv1d_silu(Target::emulation(), a));  // K = 8: the valid baseline
    for (std::uint32_t k : {9u, 0u}) {
        a.kernel = k;
        try {
            ops.causal_conv1d_silu(Target::emulation(), a);
            ADD_FAILURE() << "kernel size " << k << " accepted";
        } catch (const Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Kernel);
            EXPECT_NE(std::string(e.what()).find("kernel size " + std::to_string(k) + " not in [1, 8]"),
                      std::string::npos)
                << e.what();
        }
    }
}

TEST(HipConvDevice, VsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const ConvCase& c : conv_cases()) check_conv(c, r);
}

// ---------------------------------------------------------------------------------------
// GATED_NORM
// ---------------------------------------------------------------------------------------

struct NormCase {
    std::uint32_t rows = 48, cols = 128;
    int alias = 0;  // 0 none, 1 out == x, 2 out == z
    std::uint64_t pad = 0;
    std::string variant{};
    std::uint32_t seed = 1;
};

void check_norm(const NormCase& c, Runner& r) {
    SCOPED_TRACE(r.name());
    SCOPED_TRACE(testing::Message() << "rows " << c.rows << " cols " << c.cols << " alias " << c.alias);
    std::mt19937 rng(c.seed);
    const std::uint64_t s = c.cols + c.pad;
    std::vector<float> x = uniform(rng, c.rows * s, -3.0f, 3.0f);
    std::vector<float> z = uniform(rng, c.rows * s, -4.0f, 4.0f);
    std::vector<float> w = uniform(rng, c.cols, 0.5f, 1.5f);
    std::vector<float> out(c.rows * s, 999.0f);
    constexpr float kEps = 1e-6f;
    auto rx = x, rz = z, rout = out;
    {
        cpu::Rows xr(rx.data(), c.rows, c.cols, s);
        cpu::Rows zr(rz.data(), c.rows, c.cols, s);
        cpu::Rows yr = c.alias == 1 ? xr : c.alias == 2 ? zr : cpu::Rows(rout.data(), c.rows, c.cols, s);
        cpu::gated_rms_norm(xr, w, zr, kEps, yr);
    }
    const Buffer bx = r.make(x), bz = r.make(z), bw = r.make(w), bout = r.make(out);
    GatedNormArgs a;
    a.x = BufferView(bx, 0, 0, s * sizeof(float));
    a.z = BufferView(bz, 0, 0, s * sizeof(float));
    a.w = bw;
    a.out = c.alias == 1 ? a.x : c.alias == 2 ? a.z : BufferView(bout, 0, 0, s * sizeof(float));
    a.rows = c.rows;
    a.cols = c.cols;
    a.eps = kEps;
    OpsOptions o;
    if (!c.variant.empty()) o.gated_norm = c.variant;
    Ops(o).gated_rms_norm(r.target(), a);
    r.finish();
    r.fetch(bx, x);
    r.fetch(bz, z);
    r.fetch(bout, out);
    EXPECT_TRUE(matches(rx, x, r.exact(), "x"));
    EXPECT_TRUE(matches(rz, z, r.exact(), "z"));
    EXPECT_TRUE(matches(rout, out, r.exact(), "out"));
}

std::vector<NormCase> norm_cases() {
    return {
        {.rows = 48, .cols = 128, .seed = 71},                                   // one token, 48 heads
        {.rows = 10, .cols = 37, .alias = 1, .variant = "gated_norm_b32", .seed = 72},  // tail of the 8-lane sum
        {.rows = 7, .cols = 128, .alias = 2, .pad = 5, .seed = 73},
        {.rows = 3, .cols = 5, .seed = 74},                                      // shorter than 8: tail only
    };
}

TEST(HipNormEmu, GatedBitExactVsCpu) {
    for (const NormCase& c : norm_cases()) {
        for (auto& r : emulation_runners()) check_norm(c, *r);
    }
}

TEST(HipNormDevice, GatedVsCpu) {
    HALO_REQUIRE_HIP_DEVICE();
    DeviceRunner r;
    for (const NormCase& c : norm_cases()) check_norm(c, r);
}

}  // namespace
}  // namespace halo::hip::test
