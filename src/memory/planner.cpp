#include "halo/memory/planner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>

#include <nlohmann/json.hpp>

#include "halo/core/error.h"

namespace halo::memory {

// ---- checked arithmetic ---------------------------------------------------------------------

std::optional<std::uint64_t> checked_add(std::uint64_t a, std::uint64_t b) noexcept {
    std::uint64_t r = 0;
    if (__builtin_add_overflow(a, b, &r)) return std::nullopt;
    return r;
}

std::optional<std::uint64_t> checked_mul(std::uint64_t a, std::uint64_t b) noexcept {
    std::uint64_t r = 0;
    if (__builtin_mul_overflow(a, b, &r)) return std::nullopt;
    return r;
}

std::uint64_t add_or_throw(std::uint64_t a, std::uint64_t b, std::string_view what) {
    const auto r = checked_add(a, b);
    HALO_CHECK(r.has_value(), ErrorCode::Config, "size arithmetic overflow computing {}", what);
    return *r;
}

std::uint64_t mul_or_throw(std::uint64_t a, std::uint64_t b, std::string_view what) {
    const auto r = checked_mul(a, b);
    HALO_CHECK(r.has_value(), ErrorCode::Config, "size arithmetic overflow computing {}", what);
    return *r;
}

namespace {
std::uint64_t mul3(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::string_view what) {
    return mul_or_throw(mul_or_throw(a, b, what), c, what);
}

std::string gib(std::uint64_t b) { return std::format("{:.2f} GiB", static_cast<double>(b) / (1ULL << 30)); }
}  // namespace

std::uint64_t DtypeSize::row_bytes(std::uint64_t elems) const {
    HALO_CHECK(block_elems > 0 && block_bytes > 0, ErrorCode::Config, "dtype {} has an empty block", name);
    HALO_CHECK(elems % block_elems == 0, ErrorCode::Config, "{} elements are not a multiple of the {} block size {}",
               elems, name, block_elems);
    return mul_or_throw(elems / block_elems, block_bytes, "dtype row bytes");
}

std::uint64_t WeightBytes::total() const {
    std::uint64_t t = 0;
    for (const auto v : {embeddings, lm_head, attention, gdn, ffn, norms, mtp}) t = add_or_throw(t, v, "weight total");
    return t;
}

ModelShape qwen38_27b_shape(const WeightBytes& weights) {
    ModelShape s;
    s.weights = weights;
    s.n_attn_layers = 16;     // 64 layers, full attention every 4th (D-004)
    s.n_head = 24;
    s.n_head_kv = 4;
    s.key_dim = 256;
    s.value_dim = 256;
    s.n_gdn_layers = 48;
    s.n_v_heads = 48;         // linear_num_value_heads (D-003)
    s.d_k = 128;
    s.d_v = 128;
    s.conv_kernel = 4;
    s.conv_channels = 10240;  // 2048 q + 2048 k + 6144 v (D-004)
    s.hidden = 5120;
    s.ffn_intermediate = 17408;
    s.vocab = 248320;
    s.max_trained_context = 262144;
    s.mtp_attn_layers = 1;    // one NextN block (D-005)
    return s;
}

std::string_view to_string(Component c) noexcept {
    switch (c) {
        case Component::GdnRecurrentState: return "gdn_recurrent_state";
        case Component::GdnConvState: return "gdn_conv_state";
        case Component::GdnRollbackState: return "gdn_rollback_state";
        case Component::KvCache: return "kv_cache";
        case Component::MtpKvCache: return "mtp_kv_cache";
        case Component::Logits: return "logits";
        case Component::Activations: return "activations";
        case Component::Workspace: return "workspace";
        case Component::GpuRuntimeOverhead: return "gpu_runtime_overhead";
        case Component::Norms: return "weights.norms";
        case Component::AttentionWeights: return "weights.attention";
        case Component::GdnWeights: return "weights.gdn";
        case Component::FfnWeights: return "weights.ffn";
        case Component::LmHead: return "weights.lm_head";
        case Component::Embeddings: return "weights.embeddings";
        case Component::MtpWeights: return "weights.mtp";
        case Component::PrefixCheckpoints: return "prefix_checkpoints";
        case Component::HostRuntimeOverhead: return "host_runtime_overhead";
    }
    return "?";
}

namespace {
// Placement priority order (fastest tier first to the hottest per-token state).
constexpr std::array kAllComponents{
    Component::GdnRecurrentState, Component::GdnConvState,     Component::GdnRollbackState,
    Component::KvCache,           Component::MtpKvCache,       Component::Logits,
    Component::Activations,       Component::Workspace,        Component::GpuRuntimeOverhead,
    Component::Norms,             Component::AttentionWeights, Component::GdnWeights,
    Component::FfnWeights,        Component::LmHead,           Component::Embeddings,
    Component::MtpWeights,        Component::PrefixCheckpoints, Component::HostRuntimeOverhead,
};
}  // namespace

std::uint64_t SizeBreakdown::bytes(Component c) const noexcept {
    switch (c) {
        case Component::GdnRecurrentState: return gdn_recurrent;
        case Component::GdnConvState: return gdn_conv;
        case Component::GdnRollbackState: return gdn_rollback;
        case Component::KvCache: return kv;
        case Component::MtpKvCache: return mtp_kv;
        case Component::Logits: return logits;
        case Component::Activations: return activations;
        case Component::Workspace: return workspace;
        case Component::GpuRuntimeOverhead: return gpu_runtime_overhead;
        case Component::Norms: return weights.norms;
        case Component::AttentionWeights: return weights.attention;
        case Component::GdnWeights: return weights.gdn;
        case Component::FfnWeights: return weights.ffn;
        case Component::LmHead: return weights.lm_head;
        case Component::Embeddings: return weights.embeddings;
        case Component::MtpWeights: return weights.mtp;
        case Component::PrefixCheckpoints: return prefix_checkpoints;
        case Component::HostRuntimeOverhead: return host_runtime_overhead;
    }
    return 0;
}

std::uint64_t SizeBreakdown::total() const {
    std::uint64_t t = 0;
    for (const auto c : kAllComponents) t = add_or_throw(t, bytes(c), "planned total");
    return t;
}

SizeBreakdown compute_sizes(const ModelShape& m, const PlanRequest& r) {
    HALO_CHECK(r.max_context >= 1, ErrorCode::Config, "max_context must be >= 1");
    HALO_CHECK(r.max_sequences >= 1, ErrorCode::Config, "max_sequences must be >= 1");
    HALO_CHECK(r.ubatch >= 1 && r.ubatch <= r.batch, ErrorCode::Config, "need 1 <= ubatch ({}) <= batch ({})", r.ubatch,
               r.batch);
    HALO_CHECK(!r.mtp_enabled || r.mtp_draft_depth >= 1, ErrorCode::Config, "MTP enabled with draft depth 0");
    HALO_CHECK(!r.prefix_checkpoints.enabled || r.prefix_checkpoints.spacing_tokens >= 1, ErrorCode::Config,
               "prefix checkpoint spacing must be >= 1 token");
    HALO_CHECK(m.n_attn_layers == 0 || (m.n_head_kv > 0 && m.key_dim > 0 && m.value_dim > 0), ErrorCode::Config,
               "attention layers present but KV head shape is empty");
    HALO_CHECK(m.vocab > 0 && m.hidden > 0, ErrorCode::Config, "model shape has empty vocab/hidden");

    SizeBreakdown s;
    s.weights = m.weights;
    (void)m.weights.total();  // overflow check

    const std::uint64_t kv_row = add_or_throw(r.kv_dtype.row_bytes(m.key_dim), r.kv_dtype.row_bytes(m.value_dim),
                                              "KV bytes per head");
    const std::uint64_t kv_per_layer_token = mul_or_throw(m.n_head_kv, kv_row, "KV bytes per layer-token");
    s.kv_bytes_per_token = mul_or_throw(m.n_attn_layers, kv_per_layer_token, "KV bytes per token");
    s.kv_cells = r.kv_layout == KvLayout::Unified ? r.max_context
                                                  : mul_or_throw(r.max_context, r.max_sequences, "KV cells");
    s.kv = mul_or_throw(s.kv_bytes_per_token, s.kv_cells, "KV cache bytes");
    if (r.mtp_enabled)
        s.mtp_kv = mul3(m.mtp_attn_layers, kv_per_layer_token, s.kv_cells, "MTP KV cache bytes");

    // GDN state: one copy = recurrent (per value head d_k × d_v) + conv history, per sequence.
    const std::uint64_t rec_elems = mul3(m.n_v_heads, m.d_k, m.d_v, "GDN recurrent elements");
    const std::uint64_t rec_per_seq = mul_or_throw(m.n_gdn_layers, r.state_dtype.row_bytes(rec_elems), "GDN recurrent state");
    const std::uint64_t conv_hist = m.conv_kernel > 0 ? m.conv_kernel - 1 : 0;
    const std::uint64_t conv_elems = mul_or_throw(conv_hist, m.conv_channels, "GDN conv elements");
    const std::uint64_t conv_per_seq = mul_or_throw(m.n_gdn_layers, r.state_dtype.row_bytes(conv_elems), "GDN conv state");
    s.gdn_state_per_copy = add_or_throw(rec_per_seq, conv_per_seq, "GDN state per copy");
    s.gdn_recurrent = mul_or_throw(rec_per_seq, r.max_sequences, "GDN recurrent state");
    s.gdn_conv = mul_or_throw(conv_per_seq, r.max_sequences, "GDN conv state");
    s.gdn_rollback_copies = r.gdn_rollback_copies.value_or(r.mtp_enabled ? r.mtp_draft_depth + 1 : 0);
    s.gdn_rollback = mul3(s.gdn_state_per_copy, s.gdn_rollback_copies, r.max_sequences, "GDN rollback state");
    if (r.prefix_checkpoints.enabled && m.n_gdn_layers > 0) {
        const auto& pc = r.prefix_checkpoints;
        const std::uint64_t derived = add_or_throw(r.max_context / pc.spacing_tokens, 1, "prefix checkpoints per slot");
        const std::uint64_t per_slot = pc.max_per_slot ? *pc.max_per_slot : derived;
        HALO_CHECK(per_slot <= std::numeric_limits<std::uint32_t>::max(), ErrorCode::Config,
                   "size arithmetic overflow computing prefix checkpoints per slot");
        s.prefix_checkpoints_per_slot = static_cast<std::uint32_t>(per_slot);
        s.prefix_checkpoints = mul3(s.gdn_state_per_copy, per_slot, r.max_sequences, "prefix checkpoint bytes");
    }

    const std::uint64_t draft = r.mtp_enabled ? r.mtp_draft_depth : 0;
    const std::uint64_t rows =
        r.logits_rows ? *r.logits_rows : mul_or_throw(r.max_sequences, add_or_throw(1, draft, "logits rows"), "logits rows");
    s.logits = mul3(rows, m.vocab, 4, "logits bytes");

    if (r.activations_override) {
        s.activations = *r.activations_override;
    } else {
        // Widest simultaneously-live per-token fp32 intermediates: residual + normed input,
        // plus the widest mixer/FFN projection output. Estimate, not a measurement.
        const std::uint64_t attn_w = std::uint64_t{m.n_head} * m.key_dim * 2 +
                                     std::uint64_t{m.n_head_kv} * (std::uint64_t{m.key_dim} + m.value_dim);
        const std::uint64_t gdn_w = std::uint64_t{m.conv_channels} + std::uint64_t{m.n_v_heads} * m.d_v +
                                    2 * std::uint64_t{m.n_v_heads};
        const std::uint64_t ffn_w = 2 * std::uint64_t{m.ffn_intermediate};
        const std::uint64_t width = 2 * std::uint64_t{m.hidden} + std::max({attn_w, gdn_w, ffn_w});
        s.activations = mul3(r.ubatch, width, 4, "activation bytes");
    }
    if (r.workspace_override) {
        s.workspace = *r.workspace_override;
    } else {
        // Chunked delta-rule scratch per value head: a chunk×chunk tile plus k/v rows.
        const std::uint64_t gdn_ws_per_tok =
            mul_or_throw(m.n_v_heads, std::uint64_t{m.gdn_chunk} + m.d_k + m.d_v, "GDN workspace");
        s.workspace = mul3(r.ubatch, gdn_ws_per_tok, 4, "GDN workspace");
        if (!r.flash_attention && m.n_attn_layers > 0) {
            const auto scores = mul3(mul_or_throw(r.ubatch, m.n_head, "attention scores"), r.max_context, 4,
                                     "attention score workspace");
            s.workspace = add_or_throw(s.workspace, scores, "workspace");
        }
    }
    // The MTP block's weights are only resident when MTP drafting is enabled.
    if (!r.mtp_enabled) s.weights.mtp = 0;
    s.gpu_runtime_overhead = r.gpu_runtime_overhead;
    s.host_runtime_overhead = r.host_runtime_overhead;
    (void)s.total();  // overflow check of the grand total
    return s;
}

// ---- placement ------------------------------------------------------------------------------

namespace {

bool is_host_only(Component c) { return c == Component::HostRuntimeOverhead; }

std::uint64_t apply_safety(std::uint64_t available, double sf) {
    const long double v = std::floor(static_cast<long double>(available) * static_cast<long double>(sf));
    if (v <= 0) return 0;
    if (v >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t>(v);
}

std::uint64_t sat_sub(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }

struct Ledger {
    std::optional<TierBudget> vram;
    std::optional<TierBudget> gtt;
    TierBudget host;

    TierBudget* get(MemoryTier t) {
        switch (t) {
            case MemoryTier::Vram: return vram ? &*vram : nullptr;
            case MemoryTier::Gtt: return gtt ? &*gtt : nullptr;
            case MemoryTier::Host:
            case MemoryTier::Pinned: return &host;  // pinned staging is part of the OS pool
        }
        return nullptr;
    }
    // Headroom under every constraint that a placement on `t` touches.
    std::uint64_t remaining(MemoryTier t) {
        const std::uint64_t host_used = host.planned + (gtt ? gtt->planned : 0);
        const std::uint64_t host_left = sat_sub(host.budget, host_used);
        switch (t) {
            case MemoryTier::Vram: return vram ? sat_sub(vram->budget, vram->planned) : 0;
            case MemoryTier::Gtt: return gtt ? std::min(sat_sub(gtt->budget, gtt->planned), host_left) : 0;
            case MemoryTier::Host:
            case MemoryTier::Pinned: return host_left;
        }
        return 0;
    }
};

TierBudget make_budget(MemoryTier t, std::uint64_t total, std::uint64_t available, double sf) {
    TierBudget b;
    b.tier = t;
    b.total = total;
    b.available = available;
    b.budget = apply_safety(available, sf);
    b.reserve = available - b.budget;
    return b;
}

std::string constraint_label(const std::string& c) {
    if (c == "HOST") return "HOST (system RAM, incl. GTT pages)";
    if (c == "MAX_MEMORY") return "max_memory (user cap)";
    return c;
}

}  // namespace

std::uint64_t MemoryPlan::placed_on(MemoryTier t) const noexcept {
    std::uint64_t s = 0;
    for (const auto& a : allocations)
        if (a.tier == t) s += a.bytes;  // bounded by sizes.total(), which was overflow-checked
    return s;
}

std::optional<MemoryTier> MemoryPlan::tier_of(Component c) const noexcept {
    for (const auto& a : allocations)
        if (a.component == c) return a.tier;
    return std::nullopt;
}

namespace {

MemoryPlan plan_once(const ModelShape& shape, const PlanRequest& req, const hardware::MemoryTiers& hw) {
    HALO_CHECK(req.safety_factor > 0.0 && req.safety_factor <= kMaxSafetyFactor, ErrorCode::Config,
               "safety_factor {} outside (0, {}]: HALO never plans all of a memory tier", req.safety_factor,
               kMaxSafetyFactor);
    HALO_CHECK(hw.host_available.has_value() || hw.host_total.has_value(), ErrorCode::Config,
               "host memory size unknown (/proc/meminfo unreadable)");

    MemoryPlan plan;
    plan.safety_factor = req.safety_factor;
    plan.sizes = compute_sizes(shape, req);
    plan.prefix_checkpoints_requested = plan.sizes.prefix_checkpoints_per_slot;

    Ledger L;
    const bool gpu = hw.topology != hardware::MemoryTopology::NoGpu;
    if (gpu && hw.vram_total.value_or(0) > 0)
        L.vram = make_budget(MemoryTier::Vram, *hw.vram_total, sat_sub(*hw.vram_total, hw.vram_used.value_or(0)),
                             req.safety_factor);
    if (gpu && hw.gtt_total.value_or(0) > 0)
        L.gtt = make_budget(MemoryTier::Gtt, *hw.gtt_total, sat_sub(*hw.gtt_total, hw.gtt_used.value_or(0)),
                            req.safety_factor);
    const std::uint64_t host_total = hw.host_total.value_or(hw.host_available.value_or(0));
    L.host = make_budget(MemoryTier::Host, host_total, hw.host_available.value_or(host_total), req.safety_factor);
    if (!hw.host_available) plan.notes.emplace_back("MemAvailable unknown: HOST budget uses MemTotal");

    std::vector<MemoryTier> gpu_tiers;
    if (L.vram) gpu_tiers.push_back(MemoryTier::Vram);
    if (L.gtt) gpu_tiers.push_back(MemoryTier::Gtt);
    // Without GPU tiers the CPU reference path runs everything from HOST.
    const std::vector<MemoryTier> gpu_pref = gpu_tiers.empty() ? std::vector{MemoryTier::Host} : gpu_tiers;

    for (const auto& [c, t] : req.tier_overrides) {
        HALO_CHECK(L.get(t) != nullptr, ErrorCode::Config, "tier override {} -> {}: tier not present on this system",
                   to_string(c), hardware::to_string(t));
        HALO_CHECK(c != Component::PrefixCheckpoints || t == MemoryTier::Vram || t == MemoryTier::Gtt,
                   ErrorCode::Config, "tier override {} -> {}: prefix checkpoints are GPU-tier only", to_string(c),
                   hardware::to_string(t));
    }
    std::vector<MemoryTier> checkpoint_tiers = gpu_tiers;
    if (hw.topology == hardware::MemoryTopology::CarveoutPrimary) std::erase(checkpoint_tiers, MemoryTier::Gtt);
    if (plan.sizes.prefix_checkpoints > 0 && checkpoint_tiers.empty()) {
        HALO_CHECK(!req.prefix_checkpoints.max_per_slot.has_value(), ErrorCode::Config,
                   "prefix checkpoints require a GPU tier that is not the OS RAM pool; none was discovered");
        // Derived (default) budget: fall back to always-recompute rather than failing (D-013).
        plan.notes.push_back(std::format("prefix checkpoints disabled: no GPU tier outside the OS RAM pool "
                                         "({} per slot requested); prefix reuse falls back to always-recompute",
                                         plan.sizes.prefix_checkpoints_per_slot));
        plan.sizes.prefix_checkpoints = 0;
        plan.sizes.prefix_checkpoints_per_slot = 0;
    }

    for (const auto c : kAllComponents) {
        const std::uint64_t bytes = plan.sizes.bytes(c);
        if (bytes == 0) continue;
        if (c == Component::GpuRuntimeOverhead && !gpu) continue;
        Allocation a;
        a.component = c;
        a.bytes = bytes;
        std::vector<MemoryTier> cand;
        if (const auto it = req.tier_overrides.find(c); it != req.tier_overrides.end()) {
            cand = {it->second};
            a.overridden = true;
        } else if (is_host_only(c)) {
            cand = {MemoryTier::Host};
        } else if (c == Component::PrefixCheckpoints) {
            // Never the OS pool (architecture review M-3). In the carveout-primary layout GTT
            // pages *are* the OS pool, so checkpoints are VRAM-only there.
            cand = checkpoint_tiers;
        } else {
            cand = gpu_pref;
        }
        bool placed = false;
        for (const auto t : cand) {
            if (bytes <= L.remaining(t)) {
                a.tier = t;
                placed = true;
                break;
            }
        }
        if (!placed) {
            a.tier = cand.back();
            a.overcommitted = true;
        }
        L.get(a.tier)->planned += bytes;  // bounded by sizes.total(), which is overflow-checked
        plan.allocations.push_back(a);
    }

    // Constraints.
    auto check = [&](std::string name, std::uint64_t demand, std::uint64_t budget) {
        if (demand > budget) plan.overflows.push_back(Overflow{std::move(name), demand, budget, demand - budget});
    };
    if (L.vram) check("VRAM", L.vram->planned, L.vram->budget);
    if (L.gtt) check("GTT", L.gtt->planned, L.gtt->budget);
    check("HOST", L.host.planned + (L.gtt ? L.gtt->planned : 0), L.host.budget);
    plan.planned_total = plan.sizes.total() - (gpu ? 0 : plan.sizes.gpu_runtime_overhead);
    if (req.max_memory) check("MAX_MEMORY", plan.planned_total, *req.max_memory);
    std::ranges::stable_sort(plan.overflows, std::ranges::greater{}, &Overflow::excess);

    if (L.vram) plan.tiers.push_back(*L.vram);
    if (L.gtt) plan.tiers.push_back(*L.gtt);
    plan.tiers.push_back(L.host);

    // Notes.
    if (!gpu) plan.notes.emplace_back("no GPU tiers discovered: all classes planned in HOST (CPU reference path)");
    if (hw.topology == hardware::MemoryTopology::CarveoutPrimary && L.gtt && L.gtt->planned > 0)
        plan.notes.push_back(std::format("{} spilled to GTT; GTT pages are drawn from the OS RAM pool", gib(L.gtt->planned)));
    if (const auto t = plan.tier_of(Component::GdnRecurrentState);
        t && *t != gpu_pref.front() && !req.tier_overrides.contains(Component::GdnRecurrentState))
        plan.notes.push_back(std::format("GDN recurrent state ({}) does not fit {}; placed in {}",
                                         gib(plan.sizes.gdn_recurrent), hardware::to_string(gpu_pref.front()),
                                         hardware::to_string(*t)));
    if (const auto t = plan.tier_of(Component::PrefixCheckpoints))
        plan.notes.push_back(std::format("prefix checkpoints: {} per slot x {} slots x {:.3f} MiB = {} in {}",
                                         plan.sizes.prefix_checkpoints_per_slot, req.max_sequences,
                                         static_cast<double>(plan.sizes.gdn_state_per_copy) / (1ULL << 20),
                                         gib(plan.sizes.prefix_checkpoints),
                                         hardware::to_string(*t)));
    if (!req.activations_override || !req.workspace_override)
        plan.notes.emplace_back("activations/workspace are formula estimates until a backend reports measured sizes");

    plan.ok = plan.overflows.empty();
    if (plan.ok) {
        plan.message = std::format("fits: {} planned (safety factor {:.2f})", gib(plan.planned_total), req.safety_factor);
    } else {
        const auto& o = plan.overflows.front();
        plan.message = std::format("memory plan refused: {} needs {} but its budget is {} (safety factor {:.2f}): "
                                   "short by {} ({} bytes)",
                                   constraint_label(o.constraint), gib(o.demand), gib(o.budget), req.safety_factor,
                                   gib(o.excess), o.excess);
        for (std::size_t i = 1; i < plan.overflows.size(); ++i)
            plan.message += std::format("; also {} short by {}", constraint_label(plan.overflows[i].constraint),
                                        gib(plan.overflows[i].excess));
    }
    return plan;
}

}  // namespace

MemoryPlan plan_memory(const ModelShape& shape, const PlanRequest& req, const hardware::MemoryTiers& hw) {
    MemoryPlan plan = plan_once(shape, req, hw);
    const auto& pc = req.prefix_checkpoints;
    const std::uint32_t requested = plan.sizes.prefix_checkpoints_per_slot;
    if (plan.ok || !pc.enabled || requested == 0) return plan;
    if (pc.max_per_slot || !pc.shrink_to_fit) {
        plan.notes.push_back(std::format("prefix checkpoints ({} per slot) were not shrunk: {}", requested,
                                         pc.max_per_slot ? "max_per_slot is explicit configuration"
                                                         : "shrink_to_fit is disabled"));
        return plan;
    }
    // Shrink the derived checkpoint count to the largest that fits (demand is monotone in it).
    PlanRequest r = req;
    auto attempt = [&](std::uint32_t n) {
        r.prefix_checkpoints.max_per_slot = n;
        return plan_once(shape, r, hw);
    };
    if (!attempt(0).ok) {
        plan.notes.emplace_back("does not fit even with zero prefix checkpoints");
        return plan;
    }
    std::uint32_t lo = 0;          // fits
    std::uint32_t hi = requested;  // does not fit
    while (hi - lo > 1) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (attempt(mid).ok) lo = mid;
        else hi = mid;
    }
    MemoryPlan best = attempt(lo);
    best.prefix_checkpoints_requested = requested;
    best.notes.push_back(std::format("prefix checkpoints shrunk from {} to {} per slot to fit the budget", requested, lo));
    return best;
}

MemoryPlan plan_memory_or_throw(const ModelShape& shape, const PlanRequest& request, const hardware::MemoryTiers& tiers) {
    auto p = plan_memory(shape, request, tiers);
    p.require_ok();
    return p;
}

void MemoryPlan::require_ok() const {
    if (!ok) throw_error(ErrorCode::Memory, "{}\n{}", message, table());
}

std::string MemoryPlan::table() const {
    std::string out;
    out += std::format("{:<24} {:>14} {:>8}  {}\n", "component", "bytes", "tier", "");
    for (const auto& a : allocations) {
        std::string flags;
        if (a.overridden) flags += "override ";
        if (a.overcommitted) flags += "OVERCOMMITTED";
        out += std::format("{:<24} {:>14} {:>8}  {:>10} {}\n", to_string(a.component), a.bytes,
                           hardware::to_string(a.tier), gib(a.bytes), flags);
    }
    out += std::format("{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}\n", "tier", "total", "available", "budget", "reserve",
                       "planned");
    for (const auto& t : tiers)
        out += std::format("{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}\n", hardware::to_string(t.tier), gib(t.total),
                           gib(t.available), gib(t.budget), gib(t.reserve), gib(t.planned));
    std::uint64_t pool = 0;
    std::uint64_t pool_budget = 0;
    for (const auto& t : tiers) {
        if (t.tier == MemoryTier::Gtt || t.tier == MemoryTier::Host) pool += t.planned;
        if (t.tier == MemoryTier::Host) pool_budget = t.budget;
    }
    out += std::format("HOST pool (HOST + GTT pages, one OS RAM pool): planned {} of budget {}\n", gib(pool),
                       gib(pool_budget));
    for (const auto& o : overflows)
        out += std::format("OVER {:<36} demand {} > budget {} (excess {})\n", constraint_label(o.constraint),
                           gib(o.demand), gib(o.budget), gib(o.excess));
    for (const auto& n : notes) out += "note: " + n + "\n";
    out += (ok ? "OK: " : "REFUSED: ") + message + "\n";
    return out;
}

// ---- budget search --------------------------------------------------------------------------

namespace {
bool fits(const ModelShape& shape, const PlanRequest& r, const hardware::MemoryTiers& tiers) {
    try {
        // A size "fits" only with everything the request asked for: never let the
        // checkpoint shrink-to-fit trade the requested checkpoints for context/slots.
        PlanRequest strict = r;
        strict.prefix_checkpoints.shrink_to_fit = false;
        return plan_memory(shape, strict, tiers).ok;
    } catch (const Error& e) {
        if (e.code() == ErrorCode::Config && std::string_view(e.what()).find("overflow") != std::string_view::npos)
            return false;  // too large to even represent: does not fit
        throw;
    }
}

template <typename T, typename Set>
T search_max(T lo_bound, T hi_bound, Set&& set_and_fit) {
    if (hi_bound < lo_bound || !set_and_fit(lo_bound)) return 0;
    if (set_and_fit(hi_bound)) return hi_bound;
    T lo = lo_bound;  // fits
    T hi = hi_bound;  // does not fit
    while (hi - lo > 1) {
        const T mid = lo + (hi - lo) / 2;
        if (set_and_fit(mid)) lo = mid;
        else hi = mid;
    }
    return lo;
}
}  // namespace

std::uint64_t max_context_for_budget(const ModelShape& shape, const PlanRequest& request,
                                     const hardware::MemoryTiers& tiers, std::optional<std::uint64_t> upper) {
    const std::uint64_t hi = upper.value_or(shape.max_trained_context);
    HALO_CHECK(hi >= 1, ErrorCode::Config, "max_context_for_budget: upper bound must be >= 1");
    PlanRequest r = request;
    return search_max<std::uint64_t>(1, hi, [&](std::uint64_t ctx) {
        r.max_context = ctx;
        return fits(shape, r, tiers);
    });
}

std::uint32_t max_sequences_for_budget(const ModelShape& shape, const PlanRequest& request,
                                       const hardware::MemoryTiers& tiers, std::uint32_t upper) {
    HALO_CHECK(upper >= 1, ErrorCode::Config, "max_sequences_for_budget: upper bound must be >= 1");
    PlanRequest r = request;
    return search_max<std::uint32_t>(1, upper, [&](std::uint32_t n) {
        r.max_sequences = n;
        return fits(shape, r, tiers);
    });
}

// ---- JSON -----------------------------------------------------------------------------------

void to_json(nlohmann::json& j, const MemoryPlan& p) {
    auto allocs = nlohmann::json::array();
    for (const auto& a : p.allocations)
        allocs.push_back({{"component", std::string(to_string(a.component))},
                          {"bytes", a.bytes},
                          {"tier", std::string(hardware::to_string(a.tier))},
                          {"overridden", a.overridden},
                          {"overcommitted", a.overcommitted}});
    auto tiers = nlohmann::json::array();
    for (const auto& t : p.tiers)
        tiers.push_back({{"tier", std::string(hardware::to_string(t.tier))},
                         {"total", t.total},
                         {"available", t.available},
                         {"budget", t.budget},
                         {"reserve", t.reserve},
                         {"planned", t.planned}});
    auto overflows = nlohmann::json::array();
    for (const auto& o : p.overflows)
        overflows.push_back({{"constraint", o.constraint}, {"demand", o.demand}, {"budget", o.budget}, {"excess", o.excess}});
    const auto& s = p.sizes;
    j = {{"status", p.ok ? "ok" : "refused"},
         {"message", p.message},
         {"safety_factor", p.safety_factor},
         {"planned_total", p.planned_total},
         {"sizes",
          {{"kv_bytes_per_token", s.kv_bytes_per_token},
           {"kv_cells", s.kv_cells},
           {"kv", s.kv},
           {"mtp_kv", s.mtp_kv},
           {"gdn_recurrent", s.gdn_recurrent},
           {"gdn_conv", s.gdn_conv},
           {"gdn_state_per_copy", s.gdn_state_per_copy},
           {"gdn_rollback_copies", s.gdn_rollback_copies},
           {"gdn_rollback", s.gdn_rollback},
           {"prefix_checkpoints_per_slot", s.prefix_checkpoints_per_slot},
           {"prefix_checkpoints_requested_per_slot", p.prefix_checkpoints_requested},
           {"prefix_checkpoints", s.prefix_checkpoints},
           {"logits", s.logits},
           {"activations", s.activations},
           {"workspace", s.workspace},
           {"weights_total", s.weights.total()}}},
         {"allocations", allocs},
         {"tiers", tiers},
         {"overflows", overflows},
         {"notes", p.notes}};
}

}  // namespace halo::memory
