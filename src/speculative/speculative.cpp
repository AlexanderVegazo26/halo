// MTP speculative decoding: batched draft / verify / commit, profit gate (D-005, D-012, FR-009).

#include "halo/speculative/speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "halo/core/error.h"
#include "halo/core/log.h"

namespace halo::speculative {

using models::LogitsMode;
using models::MtpStep;
using models::SeqStep;

namespace {
/// HALO_MTP_DEVICE_HIDDEN=1 (default off): chain the MTP hidden between draft depths on the device
/// (MtpStep::hidden_device / want_hidden_device) instead of a download + upload per depth. The trunk
/// hidden still round-trips through the host (it persists in the sequence / checkpoints).
bool device_hidden_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HALO_MTP_DEVICE_HIDDEN");
        return e != nullptr && *e != '\0' && std::string_view(e) != "0";
    }();
    return on;
}

/// HALO_MTP_CHAIN=1 (default off): record every draft depth of a tick into ONE command stream (one submit /
/// fence, Qwen35::mtp_draft_chain) instead of one mtp_forward per depth: the winning token of depth d feeds
/// depth d + 1's embedding lookup from device memory and the MTP hidden stays on the device (so
/// HALO_MTP_DEVICE_HIDDEN is moot in this mode). Needs a backend with Limits::lm_head_ids (CPU, Vulkan);
/// otherwise the per-depth path runs. The trunk verify is still its own forward().
bool chain_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HALO_MTP_CHAIN");
        return e != nullptr && *e != '\0' && std::string_view(e) != "0";
    }();
    return on;
}
}  // namespace

// ---------------------------------------------------------------------------------------
// ProfitGate
// ---------------------------------------------------------------------------------------

void ProfitGate::record_spec(std::size_t tokens, std::uint64_t bytes) noexcept {
    ++win_steps_;
    win_tokens_ += tokens;
    win_bytes_ += bytes;
    if (win_steps_ < std::max<std::size_t>(1, cfg_.window)) return;
    last_speedup_ = win_bytes_ == 0 ? 0.0
                                    : static_cast<double>(win_tokens_) * static_cast<double>(plain_bytes_) /
                                          static_cast<double>(win_bytes_);
    win_steps_ = 0;
    win_tokens_ = 0;
    win_bytes_ = 0;
    if (cfg_.mode == GateMode::Auto && enabled_ && last_speedup_ < cfg_.min_speedup) {
        enabled_ = false;
        plain_since_off_ = 0;
        ++disables_;
    }
}

void ProfitGate::record_plain() noexcept {
    if (cfg_.mode != GateMode::Auto || enabled_) return;
    if (++plain_since_off_ >= cfg_.probe_interval) {
        enabled_ = true;
        win_steps_ = 0;
        win_tokens_ = 0;
        win_bytes_ = 0;
        ++probes_;
    }
}

// ---------------------------------------------------------------------------------------
// Speculator
// ---------------------------------------------------------------------------------------

namespace {

bool is_decode(const StepRequest& r) { return r.prefill.empty(); }

/// Whether this sequence keeps MTP state (model has MTP and the sequence has an MTP KV).
bool mtp_on(const models::Qwen35& m, const state::SequenceState& s) { return m.has_mtp() && s.mtp_kv.has_value(); }

void check_mtp_invariant(const models::Qwen35& m, const state::SequenceState& s, std::size_t i) {
    if (!mtp_on(m, s)) return;
    const std::size_t L = s.length();
    const std::size_t E = m.n_embd();
    const std::size_t have = s.mtp_kv->length() + s.mtp_queue_size();
    HALO_CHECK(L == 0 ? have == 0 && s.last_hidden.empty() : have + 1 == L && s.last_hidden.size() == E, ErrorCode::Api,
               "speculative request {}: MTP state out of sync (trunk {}, MTP KV {}, queued {}, hidden {} floats)", i, L,
               s.mtp_kv->length(), s.mtp_queue_size(), s.last_hidden.size());
    HALO_CHECK(s.mtp_queue_hidden.size() == s.mtp_queue_size() * E, ErrorCode::Api,
               "speculative request {}: MTP queue has {} hidden floats for {} tokens", i, s.mtp_queue_hidden.size(),
               s.mtp_queue_size());
}

void push_pair(state::SequenceState& s, std::int32_t tok, const float* h, std::size_t E) {
    s.mtp_queue_tokens.push_back(tok);
    s.mtp_queue_hidden.insert(s.mtp_queue_hidden.end(), h, h + E);
}

/// Index of the largest logit other than `best` (ties to the lowest index, like the head's argmax); NaN entries
/// are never picked. -1 if `logits` is empty, `best` is invalid or nothing qualifies.
std::int32_t runner_up(std::span<const float> logits, std::int32_t best) {
    if (logits.empty() || best < 0) return -1;
    std::int32_t arg = -1;
    float top = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (static_cast<std::int32_t>(i) == best) continue;
        const float v = logits[i];
        if (arg < 0 ? !std::isnan(v) : v > top) {
            arg = static_cast<std::int32_t>(i);
            top = v;
        }
    }
    return arg;
}

}  // namespace

Speculator::Speculator(const models::Qwen35& model, SpecConfig cfg)
    : model_(&model), cfg_(cfg), cost_(CostModel::from(model)), gate_(cfg.gate, cost_.plain_step()) {}

bool tree_env_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HALO_MTP_TREE");
        return e != nullptr && *e != '\0' && std::string_view(e) != "0";
    }();
    return on;
}

bool Speculator::tree_ok(const StepRequest& r, std::size_t k) const {
    if (!cfg_.tree || k == 0 || !is_decode(r) || !r.greedy || !r.forced_drafts.empty() || r.max_emit < 2) return false;
    if (chain_enabled()) return false;  // HALO_MTP_CHAIN drafts through mtp_draft_chain, which yields no runner-up
    if (k + 2 > cpu::kMaxTreeRows || !r.seq->gdn.fits_tree_leaf(k + 1)) return false;
    if (r.seq->kv.pool().layout().type != backend::KvType::F32) return false;  // attention_tree.comp is fp32-only
    const backend::Kind kind = model_->backend().kind();
    return kind == backend::Kind::Cpu || kind == backend::Kind::Vulkan;  // the HIP adapters reject tree attention
}

std::size_t Speculator::effective_k(const StepRequest& r) const {
    if (!is_decode(r) || !r.greedy) {
        HALO_CHECK(r.forced_drafts.empty(), ErrorCode::Api, "forced drafts need a greedy decode request");
        return 0;
    }
    const std::size_t slots = r.seq->gdn.max_slots();
    const std::size_t cap = slots == 0 ? 0 : slots - 1;
    if (!r.forced_drafts.empty()) {
        HALO_CHECK(r.forced_drafts.size() <= cap, ErrorCode::Api, "{} forced drafts, the GDN state has {} slots",
                   r.forced_drafts.size(), slots);
        return r.forced_drafts.size();
    }
    if (!mtp_on(*model_, *r.seq) || r.seq->length() == 0 || !gate_.allow()) return 0;
    return std::min({r.max_draft, cfg_.max_draft, cap});
}

void Speculator::draft(std::span<const StepRequest> reqs, Tick& tick) {
    const std::size_t n = reqs.size();
    const std::size_t E = model_->n_embd();
    tick.out.assign(n, {});
    tick.seqs.assign(n, {});
    tick.phase = Tick::Phase::Empty;
    metrics_.last_cost = {};
    metrics_.last_weight_passes = 0;
    metrics_.last_trunk_passes = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const StepRequest& r = reqs[i];
        HALO_CHECK(r.seq != nullptr, ErrorCode::Api, "speculative request {}: no sequence", i);
        HALO_CHECK(r.max_emit >= 1, ErrorCode::Api, "speculative request {}: max_emit must be >= 1", i);
        check_mtp_invariant(*model_, *r.seq, i);
        Tick::Seq& s = tick.seqs[i];
        s.k = effective_k(r);
        s.tree = tree_ok(r, s.k);  // provisional: confirmed below once the runner-up draft exists
        s.mtp_len0 = r.seq->mtp_kv ? r.seq->mtp_kv->length() : 0;
    }

    // ---- depth 1: catch-up queue + (x, h_{L-1}) + first draft, one batched call --------------
    std::vector<std::size_t> idx;
    std::vector<std::vector<std::int32_t>> toks;
    std::vector<std::vector<float>> hid;
    std::vector<std::size_t> last_row;
    for (std::size_t i = 0; i < n; ++i) {
        const StepRequest& r = reqs[i];
        if (tick.seqs[i].k == 0 || !mtp_on(*model_, *r.seq) || r.seq->length() == 0) continue;
        idx.push_back(i);
        std::vector<std::int32_t> t(r.seq->mtp_queue_tokens);
        t.push_back(r.token);
        std::vector<float> h(r.seq->mtp_queue_hidden);
        h.insert(h.end(), r.seq->last_hidden.begin(), r.seq->last_hidden.end());
        last_row.push_back(t.size() - 1);
        toks.push_back(std::move(t));
        hid.push_back(std::move(h));
    }
    const bool dev_hidden = device_hidden_enabled();
    const bool chained = chain_enabled() && model_->backend().limits().lm_head_ids;
    try {
        if (!idx.empty()) {
            std::vector<MtpStep> steps(idx.size());
            for (std::size_t j = 0; j < idx.size(); ++j) {
                const StepRequest& r = reqs[idx[j]];
                const bool forced = !r.forced_drafts.empty();
                MtpStep& m = steps[j];
                m.tokens = toks[j];
                m.hidden = hid[j];
                m.kv = &*r.seq->mtp_kv;
                m.first_position = static_cast<std::int32_t>(tick.seqs[idx[j]].mtp_len0 + 1);
                m.logit_rows = std::span(&last_row[j], forced ? 0 : 1);
                // Tree sequences need the whole depth-1 logit row for the runner-up d1' (host scan below).
                m.logits = forced ? LogitsMode::None : tick.seqs[idx[j]].tree ? LogitsMode::Full : LogitsMode::Argmax;
                const bool keep = !forced && tick.seqs[idx[j]].k > 1;  // a deeper draft will need this hidden
                m.want_hidden = keep && !dev_hidden;
                m.want_hidden_device = keep && dev_hidden;
            }
            models::StepResult res;
            if (chained) {
                // Every depth of every sequence in one submit (HALO_MTP_CHAIN): forced-draft sequences only
                // flush their rows (k = 0); the others get all s.k drafts back, in depth order.
                std::vector<std::size_t> ks(idx.size());
                for (std::size_t j = 0; j < idx.size(); ++j) {
                    ks[j] = reqs[idx[j]].forced_drafts.empty() ? tick.seqs[idx[j]].k : 0;
                }
                model_->mtp_draft_chain(steps, ks, res);
            } else {
                model_->mtp_forward(steps, res);
            }
            metrics_.last_cost += res.cost;
            metrics_.last_weight_passes += res.cost.weight_passes;
            metrics_.weight_passes += res.cost.weight_passes;
            for (std::size_t j = 0; j < idx.size(); ++j) {
                Tick::Seq& s = tick.seqs[idx[j]];
                s.mtp_flushed = true;
                if (!reqs[idx[j]].forced_drafts.empty()) continue;
                if (chained) {
                    for (const cpu::TopKEntry& a : res.seqs[j].argmax) tick.out[idx[j]].drafts.push_back(a.index);
                    continue;
                }
                tick.out[idx[j]].drafts.push_back(res.seqs[j].argmax.at(0).index);
                if (s.tree) s.alt = runner_up(res.seqs[j].logits, res.seqs[j].argmax.at(0).index);
                if (s.k > 1) {
                    if (dev_hidden) {
                        s.mtp_hidden_dev = std::move(res.seqs[j].hidden_device);
                    } else {
                        s.mtp_hidden.assign(res.seqs[j].hidden.end() - static_cast<std::ptrdiff_t>(E), res.seqs[j].hidden.end());
                    }
                }
            }
        }
        // ---- depth i >= 2: the MTP's own hidden (D-005 amendment) --------------------------
        std::size_t max_k = 0;
        for (const auto& s : tick.seqs) max_k = std::max(max_k, s.k);
        if (chained) max_k = 0;  // mtp_draft_chain already produced every depth
        const std::size_t row0 = 0;
        for (std::size_t depth = 2; depth <= max_k; ++depth) {
            std::vector<std::size_t> at;
            std::vector<MtpStep> steps;
            for (std::size_t i = 0; i < n; ++i) {
                const Tick::Seq& s = tick.seqs[i];
                if (!s.mtp_flushed || !reqs[i].forced_drafts.empty() || s.k < depth) continue;
                at.push_back(i);
                MtpStep m;
                m.tokens = std::span(&tick.out[i].drafts.back(), 1);
                const bool from_device = s.mtp_hidden_dev != nullptr;
                if (from_device) {
                    m.hidden_device = s.mtp_hidden_dev;
                } else {
                    m.hidden = s.mtp_hidden;
                }
                m.kv = &*reqs[i].seq->mtp_kv;
                m.first_position = static_cast<std::int32_t>(reqs[i].seq->length() + depth - 1);
                HALO_CHECK(static_cast<std::size_t>(m.first_position) == m.kv->length() + 1, ErrorCode::Kernel,
                           "draft depth {}: MTP position {} but MTP KV length {}", depth, m.first_position, m.kv->length());
                m.logit_rows = std::span(&row0, 1);
                m.logits = LogitsMode::Argmax;
                m.want_hidden = depth < s.k && !dev_hidden;
                m.want_hidden_device = depth < s.k && dev_hidden;
                steps.push_back(m);
            }
            if (steps.empty()) break;
            models::StepResult res;
            model_->mtp_forward(steps, res);
            metrics_.last_cost += res.cost;
            metrics_.last_weight_passes += res.cost.weight_passes;
    metrics_.weight_passes += res.cost.weight_passes;
            for (std::size_t j = 0; j < at.size(); ++j) {
                tick.out[at[j]].drafts.push_back(res.seqs[j].argmax.at(0).index);
                if (depth < tick.seqs[at[j]].k) {
                    if (dev_hidden) {
                        tick.seqs[at[j]].mtp_hidden_dev = std::move(res.seqs[j].hidden_device);
                    } else {
                        tick.seqs[at[j]].mtp_hidden = std::move(res.seqs[j].hidden);
                    }
                }
            }
        }
        for (Tick::Seq& s : tick.seqs) s.mtp_hidden_dev.reset();  // device buffers never outlive the draft phase
    } catch (...) {
        for (Tick::Seq& s : tick.seqs) s.mtp_hidden_dev.reset();
        abort_draft(reqs, tick);
        throw;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const StepRequest& r = reqs[i];
        if (!r.forced_drafts.empty()) tick.out[i].drafts.assign(r.forced_drafts.begin(), r.forced_drafts.end());
        // An MTP sequence without an MTP depth-1 call (length 0) cannot draft; forced drafts keep k.
        if (r.forced_drafts.empty()) tick.seqs[i].k = tick.out[i].drafts.size();
        // A tree needs a real chain (k >= 1), a valid d1 and a runner-up token; otherwise: a plain chain.
        Tick::Seq& s = tick.seqs[i];
        s.tree = s.tree && s.k >= 1 && s.alt >= 0 && tick.out[i].drafts.front() >= 0 && s.alt != tick.out[i].drafts.front() &&
                 tree_ok(r, s.k);
        if (s.tree) {
            s.parents = cpu::chain_plus_root_leaf_parents(s.k + 2);
        } else {
            s.alt = -1;
        }
    }
    tick.phase = Tick::Phase::Drafted;
}

void Speculator::abort_draft(std::span<const StepRequest> reqs, Tick& tick) noexcept {
    for (std::size_t i = 0; i < reqs.size() && i < tick.seqs.size(); ++i) {
        if (reqs[i].seq != nullptr && reqs[i].seq->mtp_kv) reqs[i].seq->mtp_kv->truncate(tick.seqs[i].mtp_len0);
        tick.seqs[i].mtp_flushed = false;
    }
    tick.phase = Tick::Phase::Empty;
}

void Speculator::verify(std::span<const StepRequest> reqs, Tick& tick) {
    HALO_CHECK(tick.phase == Tick::Phase::Drafted && tick.seqs.size() == reqs.size(), ErrorCode::Api,
               "Speculator::verify: call draft() first with the same requests");
    const std::size_t n = reqs.size();
    std::vector<SeqStep> steps(n);
    std::vector<std::vector<std::size_t>> rows(n);
    for (std::size_t i = 0; i < n; ++i) {
        const StepRequest& r = reqs[i];
        Tick::Seq& s = tick.seqs[i];
        s.len0 = r.seq->length();
        if (is_decode(r)) {
            s.fed.assign(1, r.token);
            s.fed.insert(s.fed.end(), tick.out[i].drafts.begin(), tick.out[i].drafts.end());
            if (s.tree) s.fed.push_back(s.alt);  // the leaf row: d1' as a second child of x
        } else {
            s.fed.assign(r.prefill.begin(), r.prefill.end());
        }
        s.alt_win = false;
        SeqStep& st = steps[i];
        st.tokens = s.fed;
        st.kv = &r.seq->kv;
        st.gdn = &r.seq->gdn;
        if (s.tree) st.tree_parents = s.parents;
        st.want_hidden = mtp_on(*model_, *r.seq);
        if (is_decode(r)) {
            rows[i].resize(r.greedy ? s.fed.size() : 1);
            for (std::size_t j = 0; j < rows[i].size(); ++j) rows[i][j] = j;
            st.logits = r.greedy ? LogitsMode::Argmax : LogitsMode::Full;
            st.n_state_slots = s.k > 0 ? s.k + 1 : 0;
            st.gdn_path = models::GdnPath::Recurrent;
        } else {
            if (r.want_output) rows[i].push_back(s.fed.size() - 1);
            st.logits = !r.want_output ? LogitsMode::None : r.greedy ? LogitsMode::Argmax : LogitsMode::Full;
            st.gdn_path = models::GdnPath::Auto;
        }
        st.logit_rows = rows[i];
    }
    models::StepResult res;
    try {
        model_->forward(steps, res);
    } catch (...) {
        abort_draft(reqs, tick);
        throw;
    }
    metrics_.last_cost += res.cost;
    metrics_.last_weight_passes += res.cost.weight_passes;
    metrics_.weight_passes += res.cost.weight_passes;
    metrics_.last_trunk_passes += res.cost.weight_passes;
    metrics_.trunk_passes += res.cost.weight_passes;
    for (std::size_t i = 0; i < n; ++i) {
        const StepRequest& r = reqs[i];
        Tick::Seq& s = tick.seqs[i];
        StepOutput& o = tick.out[i];
        models::SeqOutput& so = res.seqs[i];
        s.hidden = std::move(so.hidden);
        if (!r.greedy) {
            o.logits = std::move(so.logits);
            o.n_keep = is_decode(r) ? 1 : s.fed.size();
            continue;
        }
        for (const auto& a : so.argmax) o.targets.push_back(a.index);
        // H1: the backend now reports a NaN logit as a poisoned row (index == -1) instead of
        // throwing from inside the forward, so pre-tick KV/GDN state stays intact (M1) and the
        // engine can fail just this sequence instead of the whole tick. A poisoned target
        // deliberately is not rejected here: this function runs once for every sequence in the
        // batch, so throwing here would still fail all of them together. The engine's own
        // per-sequence output-phase isolation (CpuEngine::tick(), review R-1) is where a
        // poisoned token id must be caught -- see the check there before it is ever fed,
        // sampled around, or detokenized.
        if (!is_decode(r)) {
            o.tokens = o.targets;
            o.n_keep = s.fed.size();
            continue;
        }
        std::int32_t leaf_target = -1;
        if (s.tree) {
            // Rows were [x, d1..dk, d1']: the last target belongs to the leaf; the rest is the chain, exactly
            // as without a tree.
            leaf_target = o.targets.back();
            o.targets.pop_back();
        }
        std::size_t acc = 0;
        while (acc < o.drafts.size() && o.drafts[acc] == o.targets[acc]) ++acc;
        o.accepted = acc;
        const auto is_stop = [&](std::int32_t t) {
            return std::find(r.stop_tokens.begin(), r.stop_tokens.end(), t) != r.stop_tokens.end();
        };
        if (s.tree && acc == 0 && s.alt == o.targets[0] && r.max_emit >= 2 && !is_stop(o.targets[0])) {
            // d1 was wrong but the runner-up d1' is the trunk's token after x: keep [x, d1'] and emit
            // [argmax(row 0), argmax(leaf row)] (two tokens instead of one). The leaf's rows are rewritten
            // as the committed 2-row step: fed = [x, d1'], hidden = rows {0, leaf}.
            s.alt_win = true;
            o.accepted = 1;
            o.n_keep = 2;
            o.tokens = {o.targets[0], leaf_target};
            o.targets = o.tokens;
            const std::size_t E = model_->n_embd();
            if (!s.hidden.empty()) {
                const std::size_t leaf_row = s.fed.size() - 1;
                std::vector<float> h2(2 * E);
                std::copy_n(s.hidden.begin(), E, h2.begin());
                std::copy_n(s.hidden.begin() + static_cast<std::ptrdiff_t>(leaf_row * E), E, h2.begin() + static_cast<std::ptrdiff_t>(E));
                s.hidden = std::move(h2);
            }
            s.fed = {s.fed[0], s.alt};
            continue;
        }
        std::size_t keep = std::min(acc + 1, r.max_emit);
        for (std::size_t j = 0; j < keep; ++j) {
            if (std::find(r.stop_tokens.begin(), r.stop_tokens.end(), o.targets[j]) != r.stop_tokens.end()) {
                keep = j + 1;
                break;
            }
        }
        o.n_keep = keep;
        o.tokens.assign(o.targets.begin(), o.targets.begin() + static_cast<std::ptrdiff_t>(keep));
    }
    tick.phase = Tick::Phase::Verified;
}

void Speculator::commit(std::span<const StepRequest> reqs, Tick& tick, std::span<const std::size_t> n_keep) {
    HALO_CHECK(tick.phase == Tick::Phase::Verified && tick.seqs.size() == reqs.size(), ErrorCode::Api,
               "Speculator::commit: call verify() first with the same requests");
    HALO_CHECK(n_keep.empty() || n_keep.size() == reqs.size(), ErrorCode::Api, "commit: {} n_keep values for {} requests",
               n_keep.size(), reqs.size());
    for (std::size_t i = 0; i < n_keep.size(); ++i) {
        if (!is_decode(reqs[i])) continue;
        HALO_CHECK(n_keep[i] >= 1 && n_keep[i] <= tick.out[i].n_keep, ErrorCode::Api,
                   "commit request {}: n_keep {} outside [1, {}]", i, n_keep[i], tick.out[i].n_keep);
    }
    const std::size_t E = model_->n_embd();
    // --- no Error from here on for valid ticks (GDN commit preconditions hold by construction)
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        const StepRequest& r = reqs[i];
        state::SequenceState& seq = *r.seq;
        Tick::Seq& s = tick.seqs[i];
        StepOutput& o = tick.out[i];
        std::size_t nk = o.n_keep;
        bool leaf = false;  // the tree's runner-up branch [x, d1'] was kept
        if (is_decode(r)) {
            if (!n_keep.empty()) nk = n_keep[i];
            o.n_keep = nk;
            if (o.tokens.size() > nk) o.tokens.resize(nk);
            leaf = s.alt_win && nk == 2;
            if (leaf) {
                // The leaf's KV row sits at L+k+1 (row order [x, d1..dk, d1']); it belongs at L+1. If the move
                // fails, fall back to keeping only x: a valid state (chain slot k of the GDN ring is x's).
                try {
                    model_->kv_move_row(seq.kv, s.len0 + s.k + 1, s.len0 + 1);
                } catch (const std::exception& e) {
                    HALO_WARN("speculative", "tree commit: KV row move failed ({}); keeping only the pending token", e.what());
                    leaf = false;
                    nk = 1;
                    o.n_keep = 1;
                    o.accepted = 0;
                    o.tokens.resize(1);
                }
            }
            seq.kv.truncate(s.len0 + nk);
            if (leaf) {
                seq.gdn.commit_tree_leaf(s.k + 1);
            } else if (s.k > 0) {
                seq.gdn.commit_rows_kept(s.k + 1, nk);
            } else {
                seq.gdn.drop_slots();
            }
        } else {
            seq.gdn.drop_slots();
        }
        if (mtp_on(*model_, seq)) {
            if (s.mtp_flushed) {
                // Keep the teacher-forced rows (queue + (x, h_{L-1})) = positions 1..L; drop
                // the draft-chain rows, which were computed from MTP hiddens.
                seq.mtp_kv->truncate(s.len0);
                seq.mtp_queue_tokens.clear();
                seq.mtp_queue_hidden.clear();
            } else if (s.len0 > 0) {
                push_pair(seq, s.fed[0], seq.last_hidden.data(), E);
            }
            for (std::size_t j = 1; j < nk; ++j) push_pair(seq, s.fed[j], &s.hidden[(j - 1) * E], E);
            seq.last_hidden.assign(&s.hidden[(nk - 1) * E], &s.hidden[nk * E]);
        }
        // ---- metrics + gate (decode only) -----------------------------------------------
        if (!is_decode(r)) continue;
        ++metrics_.decode_steps;
        if (s.tree) ++metrics_.tree_steps;
        if (leaf) ++metrics_.tree_leaf_kept;
        metrics_.emitted += o.tokens.size();
        metrics_.predicted_bytes += cost_.spec_step(s.k);
        if (s.k > 0) {
            ++metrics_.spec_steps;
            metrics_.drafted += s.k;
            metrics_.accepted += o.accepted;
            if (metrics_.accepted_at.size() < s.k) metrics_.accepted_at.resize(s.k, 0);
            for (std::size_t j = 0; j < o.accepted; ++j) ++metrics_.accepted_at[j];
            // L6: record what was actually kept (nk, post max_emit/stop-token cap), not
            // accepted+1 -- the uncapped count overstated speedup for workloads that mostly
            // hit their token cap, biasing GateMode::Auto toward keeping speculation on.
            if (r.forced_drafts.empty()) gate_.record_spec(nk, cost_.spec_step(s.k));
        } else if (r.greedy && r.max_draft > 0 && model_->has_mtp()) {
            gate_.record_plain();
        }
    }
    tick.phase = Tick::Phase::Committed;
}

void Speculator::flush_mtp(std::span<state::SequenceState* const> seqs) {
    if (!model_->has_mtp()) return;
    std::vector<MtpStep> steps;
    std::vector<state::SequenceState*> at;
    for (state::SequenceState* s : seqs) {
        if (s == nullptr || !s->mtp_kv || s->mtp_queue_tokens.empty()) continue;
        MtpStep m;
        m.tokens = s->mtp_queue_tokens;
        m.hidden = s->mtp_queue_hidden;
        m.kv = &*s->mtp_kv;
        m.first_position = static_cast<std::int32_t>(s->mtp_kv->length() + 1);
        m.logits = LogitsMode::None;
        steps.push_back(m);
        at.push_back(s);
    }
    if (steps.empty()) return;
    models::StepResult res;
    model_->mtp_forward(steps, res);
    metrics_.last_cost += res.cost;
    metrics_.last_weight_passes += res.cost.weight_passes;
    metrics_.weight_passes += res.cost.weight_passes;
    for (state::SequenceState* s : at) {
        metrics_.mtp_flush_rows += s->mtp_queue_tokens.size();
        s->mtp_queue_tokens.clear();
        s->mtp_queue_hidden.clear();
    }
}

void Speculator::step(std::span<const StepRequest> reqs, Tick& tick) {
    if (reqs.empty()) return;
    draft(reqs, tick);
    verify(reqs, tick);
    commit(reqs, tick);
    ++metrics_.ticks;
    std::vector<state::SequenceState*> full;
    for (const StepRequest& r : reqs) {
        if (r.seq->mtp_queue_size() >= std::max<std::size_t>(1, cfg_.mtp_flush_rows)) full.push_back(r.seq);
    }
    try {
        flush_mtp(full);
    } catch (const std::exception& e) {
        // The tick's results and every sequence's trunk state are valid. Keeping the queue
        // would let it grow without bound while nothing drafts (review N-1), so these
        // sequences stop using MTP: their greedy output is unaffected (drafts never change
        // emitted tokens), only speculation is lost for them.
        HALO_WARN("speculative", "MTP catch-up flush failed ({}); {} sequence(s) continue without MTP", e.what(),
                  full.size());
        for (state::SequenceState* s : full) {
            s->mtp_kv.reset();
            s->mtp_queue_tokens.clear();
            s->mtp_queue_hidden.clear();
            ++metrics_.mtp_dropped;
        }
    }
}

}  // namespace halo::speculative
