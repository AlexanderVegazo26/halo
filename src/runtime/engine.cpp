// CPU reference Engine: one worker thread, one batched tick per step, per-request token
// queues drained on the caller's thread, prefix cache (KV sharing + GDN checkpoints),
// MTP speculative decoding with a k = 0 fallback on MTP-KV exhaustion.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "halo/autotune/lookup.h"
#include "halo/backends/cpu/thread_pool.h"
#include "halo/core/error.h"
#include "halo/core/log.h"
#include "halo/hardware/hardware.h"
#include "halo/memory/planner.h"
#include "halo/model/model.h"
#include "halo/models/qwen35.h"
#include "halo/runtime/cpu_engine.h"
#include "halo/runtime/profile.h"
#include "halo/sampling/sampler.h"
#include "halo/speculative/speculative.h"
#include "halo/state/checkpoint.h"
#include "halo/state/sequence.h"
#include "halo/template/chat_template.h"
#include "halo/tokenizer/tokenizer.h"

namespace halo::runtime {

namespace {

using Clock = std::chrono::steady_clock;
using Toks = std::vector<std::int32_t>;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// ---- memory plan inputs --------------------------------------------------------------

std::uint64_t nb(const model::WeightRef& w) { return w.present() ? w.n_bytes() : 0; }

memory::ModelShape model_shape(const model::NormalizedModel& m, const models::Qwen35& q) {
    memory::ModelShape s;
    const auto& hp = m.hparams();
    auto& w = s.weights;
    for (const auto& l : m.layers()) {
        w.norms += nb(l.attn_norm) + nb(l.post_attention_norm);
        if (l.kind == model::LayerKind::FullAttention) {
            w.attention += nb(l.attn.q) + nb(l.attn.k) + nb(l.attn.v) + nb(l.attn.output);
            w.norms += nb(l.attn.q_norm) + nb(l.attn.k_norm);
        } else {
            const auto& g = l.gdn;
            w.gdn += nb(g.qkv) + nb(g.gate) + nb(g.conv1d) + nb(g.dt_bias) + nb(g.a) + nb(g.beta) + nb(g.alpha) +
                     nb(g.norm) + nb(g.out);
        }
        w.ffn += nb(l.ffn_gate) + nb(l.ffn_up) + nb(l.ffn_down);
    }
    w.norms += nb(m.output_norm());
    w.embeddings = nb(m.token_embd());
    w.lm_head = m.lm_head_tied() ? 0 : nb(m.output());
    w.mtp = q.mtp_block_bytes();
    s.n_attn_layers = hp.n_attn_layers;
    s.n_head = hp.n_head;
    s.n_head_kv = hp.n_head_kv;
    s.key_dim = hp.key_length;
    s.value_dim = hp.value_length;
    s.n_gdn_layers = hp.n_gdn_layers;
    s.n_v_heads = hp.gdn_n_v_heads;
    s.d_k = hp.gdn_head_k_dim;
    s.d_v = hp.gdn_head_v_dim;
    s.conv_kernel = hp.ssm_conv_kernel;
    s.conv_channels = hp.gdn_conv_channels;
    s.hidden = hp.n_embd;
    s.ffn_intermediate = hp.n_ff;
    s.vocab = static_cast<std::uint32_t>(hp.n_vocab);
    s.max_trained_context = hp.context_length;
    s.mtp_attn_layers = q.has_mtp() ? 1 : 0;
    return s;
}

// ---- requests ------------------------------------------------------------------------

struct Request {
    std::uint64_t id = 0;
    GenerateRequest req;
    std::optional<sampling::Sampler> sampler;  // non-greedy / penalties / structured
    bool fast_greedy = true;                    // greedy with no penalty or grammar: MTP path
    std::size_t max_new = 0;                    // max_tokens capped by the context
    Clock::time_point t_submit;

    // Delivery queue (worker -> caller thread).
    std::mutex mu;
    std::condition_variable cv;
    std::deque<TokenEvent> events;
    bool done = false;
    GenerateResult result;
    std::atomic<bool> cancel{false};
};

/// A sequence the worker is running.
struct Active {
    std::shared_ptr<Request> r;
    std::size_t slot = 0;
    Toks fed;                     ///< tokens whose rows are in the KV (== SequenceState length)
    std::size_t prompt_done = 0;  ///< prompt tokens fed (restored ones included)
    std::int32_t pending = -1;    ///< emitted, not yet fed
    Toks generated;               ///< emitted tokens
    Toks history;                 ///< prompt + generated (sampler penalties)
    std::optional<tokenizer::StreamDecoder> dec;
    std::string text;             ///< decoded output (stop strings)
    std::vector<std::size_t> ckpt_at;  ///< prompt positions that get a GDN checkpoint
    std::size_t next_ckpt = 0;
    std::optional<Clock::time_point> t_first;
    Clock::time_point t_last;
    std::size_t drafted = 0, accepted = 0;
    Toks forced;                  ///< draft_hook output for the current tick
    bool finished = false;
    FinishReason reason = FinishReason::Stop;
};

/// A retired sequence kept for prefix reuse.
struct CacheEntry {
    std::uint64_t id = 0;
    Toks tokens;  ///< == kv rows
    std::unique_ptr<kv_cache::SequenceKv> kv;
    std::unique_ptr<kv_cache::SequenceKv> mtp_kv;  ///< rows for positions 1..tokens.size()-1, or null
};

std::size_t lcp(std::span<const std::int32_t> a, std::span<const std::int32_t> b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

class CpuEngine final : public Engine {
public:
    CpuEngine(const EngineConfig& cfg, const CpuEngineOptions& opts);
    ~CpuEngine() override;
    CpuEngine(const CpuEngine&) = delete;
    CpuEngine& operator=(const CpuEngine&) = delete;

    const ModelInfo& model() const override { return info_; }
    const tokenizer::Tokenizer& tokenizer() const override { return *tok_; }
    const chat::ChatTemplate& chat_template() const override { return *tmpl_; }
    GenerateResult generate(const GenerateRequest& req, const TokenCallback& cb) override;
    EngineStats stats() const override;

private:
    void run() noexcept;
    void admit(const std::shared_ptr<Request>& r);
    void tick();
    bool evict_one_cache_entry();
    void ensure_capacity(std::size_t trunk_rows, std::size_t mtp_rows);
    /// Emits one token for `a`; returns true if the sequence finished.
    bool emit(Active& a, std::int32_t tok, std::vector<TokenEvent>& events);
    void retire(Active& a, FinishReason why, const std::string& error = {});
    void deliver(Request& r, std::vector<TokenEvent>& events, bool done);
    void checkpoint(Active& a);
    void apply_profile();

    EngineConfig cfg_;
    CpuEngineOptions opts_;
    std::unique_ptr<model::NormalizedModel> nm_;
    std::unique_ptr<cpu::ThreadPool> pool_;
    std::unique_ptr<models::Qwen35> model_;
    std::unique_ptr<tokenizer::Tokenizer> tok_;
    std::unique_ptr<chat::ChatTemplate> tmpl_;
    std::shared_ptr<const sampling::TokenVocab> vocab_;
    std::mutex vocab_mu_;
    ModelInfo info_;
    memory::MemoryPlan plan_;
    std::optional<std::int32_t> eos_;
    std::size_t max_draft_ = 0;
    std::size_t threads_ = 1;
    std::size_t gdn_chunk_ = 64;
    std::size_t prefill_chunk_ = 256;
    std::string tuning_ = "off";

    std::unique_ptr<kv_cache::KvPool> kv_pool_;
    std::unique_ptr<kv_cache::KvPool> mtp_pool_;
    std::vector<std::unique_ptr<state::SequenceState>> slots_;
    std::vector<std::size_t> free_slots_;
    std::unique_ptr<speculative::Speculator> spec_;
    std::unique_ptr<state::CheckpointStore> ckpts_;
    std::list<CacheEntry> cache_;  // front = most recently used
    std::size_t cache_cap_ = 0;
    std::size_t ckpt_spacing_ = 0;

    // shared with callers
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    bool stop_ = false;
    std::uint64_t next_id_ = 1;
    EngineStats stats_;

    // worker only
    std::list<Active> active_;
    std::thread worker_;
};

// ---------------------------------------------------------------------------------------

CpuEngine::CpuEngine(const EngineConfig& cfg, const CpuEngineOptions& opts) : cfg_(cfg), opts_(opts) {
    HALO_CHECK(cfg.backend == "cpu" || cfg.backend == "auto", ErrorCode::Unsupported,
               "backend '{}' is not available in this build (cpu reference only)", cfg.backend);
    HALO_CHECK(cfg.max_sequences >= 1 && cfg.max_context >= 2, ErrorCode::Config,
               "max_sequences {} / max_context {} out of range", cfg.max_sequences, cfg.max_context);
    HALO_CHECK(cfg.mtp_max_draft >= 0 && cfg.mtp_max_draft <= 8, ErrorCode::Config, "mtp_max_draft {} outside [0, 8]",
               cfg.mtp_max_draft);
    HALO_CHECK(opts.prefill_chunk >= 1 && opts.kv_block_tokens >= 1, ErrorCode::Config, "bad prefill chunk / KV block size");
    HALO_CHECK(!cfg.profile_db || !cfg.platform_power_mode.empty(), ErrorCode::Config,
               "profile_db needs platform_power_mode (part of the profile key, TRD §57)");
    nm_ = std::make_unique<model::NormalizedModel>(model::NormalizedModel::load(cfg.model_path));
    if (cfg.mtp_path) nm_->attach_mtp(*cfg.mtp_path);
    apply_profile();
    pool_ = std::make_unique<cpu::ThreadPool>(threads_);
    models::Qwen35Options mo;
    mo.gdn_chunk = gdn_chunk_;
    model_ = std::make_unique<models::Qwen35>(*nm_, pool_.get(), mo);
    const model::TokenizerMetadata meta = nm_->tokenizer();
    tok_ = std::make_unique<tokenizer::Tokenizer>(tokenizer::Tokenizer::from_spec(models::vocab_spec(meta)));
    HALO_CHECK(meta.chat_template.has_value() && !meta.chat_template->empty(), ErrorCode::Model,
               "GGUF has no tokenizer.chat_template");
    const auto piece = [&](std::optional<std::int32_t> id) { return id ? tok_->token_to_piece(*id) : std::string(); };
    tmpl_ = std::make_unique<chat::ChatTemplate>(std::string(*meta.chat_template), piece(tok_->bos()), piece(tok_->eos()));
    eos_ = opts.eos_token ? opts.eos_token : tok_->eos();

    const auto& hp = nm_->hparams();
    const std::string* gname = nm_->gguf().get_string("general.name");
    info_.id = gname != nullptr && !gname->empty() ? *gname : std::filesystem::path(cfg.model_path).stem().string();
    info_.architecture = "qwen35";
    info_.context_length = hp.context_length > 0 ? std::min<std::size_t>(cfg.max_context, hp.context_length) : cfg.max_context;
    info_.vocab_size = model_->n_vocab();
    info_.has_mtp = model_->has_mtp();
    const bool mtp = cfg.mtp_enabled && model_->has_mtp() && cfg.mtp_max_draft > 0;
    max_draft_ = mtp ? static_cast<std::size_t>(cfg.mtp_max_draft) : 0;

    // ---- memory plan (D-002: tier discovery, never assumption) --------------------------
    hardware::DiscoveryOptions dopts;
    dopts.root = opts.hardware_root;
    const hardware::HardwareInfo hw = hardware::discover(dopts);
    memory::PlanRequest pr;
    pr.max_context = info_.context_length;
    pr.max_sequences = static_cast<std::uint32_t>(cfg.max_sequences);
    pr.kv_dtype = memory::DtypeSize::f32();  // the CPU reference stores fp32 KV
    pr.mtp_enabled = mtp;
    pr.mtp_draft_depth = static_cast<std::uint32_t>(max_draft_);
    pr.prefix_checkpoints.enabled = cfg.prefix_cache;
    if (opts.checkpoint_spacing > 0) pr.prefix_checkpoints.spacing_tokens = static_cast<std::uint32_t>(opts.checkpoint_spacing);
    pr.max_memory = cfg.max_memory_bytes;
    plan_ = memory::plan_memory_or_throw(model_shape(*nm_, *model_), pr, hw.tiers);
    for (const auto& n : plan_.notes) HALO_INFO("runtime", "memory plan: {}", n);
    ckpt_spacing_ = pr.prefix_checkpoints.spacing_tokens;

    // ---- pools, slots ------------------------------------------------------------------
    // Prefill chunks stay a multiple of the GDN chunk (chunk boundaries line up).
    prefill_chunk_ = std::max(gdn_chunk_, opts.prefill_chunk / gdn_chunk_ * gdn_chunk_);
    const std::size_t bt = opts.kv_block_tokens;
    const std::size_t per_seq = ceil_div(info_.context_length + max_draft_ + 1, bt) + 1;  // +1: one COW copy
    cache_cap_ = cfg.prefix_cache ? (opts.prefix_cache_entries > 0 ? opts.prefix_cache_entries : cfg.max_sequences) : 0;
    const std::size_t blocks = opts.kv_blocks.value_or(per_seq * (cfg.max_sequences + cache_cap_));
    kv_pool_ = std::make_unique<kv_cache::KvPool>(model_->kv_layout(bt), blocks);
    if (mtp) {
        mtp_pool_ = std::make_unique<kv_cache::KvPool>(model_->mtp_kv_layout(bt),
                                                       opts.mtp_kv_blocks.value_or(per_seq * (cfg.max_sequences + cache_cap_)));
    }
    for (std::size_t i = 0; i < cfg.max_sequences; ++i) {
        slots_.push_back(std::make_unique<state::SequenceState>(*kv_pool_, mtp_pool_.get(), model_->gdn_shape(), max_draft_ + 1));
        free_slots_.push_back(cfg.max_sequences - 1 - i);
    }
    speculative::SpecConfig sc;
    sc.max_draft = max_draft_;
    sc.gate.mode = mtp ? opts.gate_mode : speculative::GateMode::Off;
    sc.gate.window = opts.gate_window;
    sc.gate.probe_interval = opts.gate_probe_interval;
    spec_ = std::make_unique<speculative::Speculator>(*model_, sc);

    // D-013 budget. The planner puts checkpoints in the GPU pool only; the CPU reference has
    // none on a host-only machine, so it budgets the requested count on the host.
    // A Checkpoint also stores its token prefix and the carried hidden (state::Checkpoint::
    // bytes), which the planner's per-copy state size does not include.
    const std::uint64_t extra = (static_cast<std::uint64_t>(hp.n_embd) + info_.context_length) * 4;
    std::uint64_t ck = plan_.sizes.prefix_checkpoints;
    if (ck > 0 && plan_.sizes.gdn_state_per_copy > 0) ck += ck / plan_.sizes.gdn_state_per_copy * extra;
    if (cfg.prefix_cache && ck == 0) {
        // Requested spacing checkpoints + prompt end + retirement, per sequence.
        const std::uint64_t ck_per_seq = std::max<std::uint64_t>(plan_.prefix_checkpoints_requested, 1) + 2;
        ck = ck_per_seq * cfg.max_sequences * (plan_.sizes.gdn_state_per_copy + extra);
        HALO_INFO("runtime", "no GPU tier for prefix checkpoints: CPU reference budgets {} bytes on the host", ck);
    }
    if (opts.checkpoint_budget_bytes) ck = *opts.checkpoint_budget_bytes;
    ckpts_ = std::make_unique<state::CheckpointStore>(cfg.prefix_cache ? ck : 0);
    if (cfg.max_memory_bytes) {
        // The planner sized KV for max_sequences; the pools also hold the prefix cache's
        // share, and the host checkpoint budget is outside the plan: check the real total.
        const std::uint64_t pools = static_cast<std::uint64_t>(kv_pool_->total_blocks()) * kv_pool_->layout().block_bytes() +
                                    (mtp_pool_ ? static_cast<std::uint64_t>(mtp_pool_->total_blocks()) * mtp_pool_->layout().block_bytes() : 0);
        const std::uint64_t planned_other = plan_.planned_total - plan_.sizes.kv - plan_.sizes.mtp_kv - plan_.sizes.prefix_checkpoints;
        const std::uint64_t real = planned_other + pools + ckpts_->budget_bytes();
        HALO_CHECK(real <= *cfg.max_memory_bytes, ErrorCode::Memory,
                   "allocation of {} bytes (KV pools {} incl. prefix cache, checkpoints {}) exceeds max_memory {}", real, pools,
                   ckpts_->budget_bytes(), *cfg.max_memory_bytes);
    }

    worker_ = std::thread([this] { run(); });
}

CpuEngine::~CpuEngine() {
    {
        const std::lock_guard lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

EngineStats CpuEngine::stats() const {
    const std::lock_guard lk(mu_);
    EngineStats s = stats_;
    s.queued_requests = pending_.size();
    // What is actually in effect (not what was requested): the live pool and model.
    s.threads = static_cast<std::uint32_t>(pool_->size());
    s.gdn_chunk = static_cast<std::uint32_t>(model_->gdn_chunk());
    s.tuning = tuning_;
    return s;
}

// TRD §64 / §56: read the profile once at creation (read-only snapshot; no tuning here).
void CpuEngine::apply_profile() {
    threads_ = cfg_.threads > 0 ? static_cast<std::size_t>(cfg_.threads) : cpu::ThreadPool::default_threads();
    gdn_chunk_ = 64;
    if (!cfg_.profile_db) {
        tuning_ = "off";
        return;
    }
    const autotune::ProfileKey key = engine_profile_key(cfg_, opts_.hardware_root);  // Error(Config) w/o power mode
    const std::filesystem::path db(*cfg_.profile_db);
    const autotune::ProfileLookup lookup =
        std::filesystem::exists(db) ? autotune::ProfileLookup::open(db) : autotune::ProfileLookup::empty();
    const auto& hp = nm_->hparams();
    const auto describe = [](const autotune::LookupResult& r) {
        std::string s(autotune::to_string(r.source));
        if (r.candidate) s += "(" + r.candidate->to_string() + ")";
        return s;
    };
    const auto measured = [](const autotune::LookupResult& r) {
        return r.candidate && (r.source == autotune::SelectionSource::Exact || r.source == autotune::SelectionSource::Compatible);
    };
    const autotune::LookupResult mm = autotune::select_kernel(lookup, key, matmul_op_key(hp), "cpu", matmul_candidates(), nullptr, {});
    const autotune::LookupResult gd = autotune::select_kernel(lookup, key, gdn_op_key(hp), "cpu", gdn_candidates(), nullptr, {});
    std::string note;
    if (measured(mm)) {
        if (cfg_.threads > 0) {
            note = " [explicit threads override the profile]";
        } else {
            threads_ = static_cast<std::size_t>(mm.candidate->get("threads"));
        }
    }
    if (measured(gd)) gdn_chunk_ = static_cast<std::size_t>(gd.candidate->get("chunk"));
    tuning_ = std::format("MATMUL={}; GATED_DELTANET={}{}{}", describe(mm), describe(gd), note,
                          std::filesystem::exists(db) ? "" : " [no profile database]");
    for (const auto& r : {mm, gd}) {
        for (const auto& why : r.rejections) HALO_WARN("runtime", "profile rejected: {}", why);
    }
    HALO_INFO("runtime", "kernel profile ({} entries): {}; threads {}, GDN chunk {}", lookup.size(), tuning_, threads_,
              gdn_chunk_);
}

// ---- caller side ---------------------------------------------------------------------

GenerateResult CpuEngine::generate(const GenerateRequest& req, const TokenCallback& cb) {
    HALO_CHECK(!req.prompt.empty(), ErrorCode::Api, "generate: empty prompt");
    HALO_CHECK(req.max_tokens >= 1, ErrorCode::Api, "generate: max_tokens must be >= 1");
    for (const std::int32_t t : req.prompt) {
        HALO_CHECK(t >= 0 && static_cast<std::size_t>(t) < info_.vocab_size, ErrorCode::Api,
                   "generate: prompt token {} outside the vocabulary", t);
    }
    HALO_CHECK(req.prompt.size() < info_.context_length, ErrorCode::Api, "generate: prompt of {} tokens does not fit context {}",
               req.prompt.size(), info_.context_length);
    for (const auto& s : req.stop_token_seqs) HALO_CHECK(!s.empty(), ErrorCode::Api, "generate: empty stop token sequence");
    auto r = std::make_shared<Request>();
    r->req = req;
    r->t_submit = Clock::now();
    r->max_new = std::min(req.max_tokens, info_.context_length - req.prompt.size());
    const SamplingParams& sp = req.sampling;
    sampling::validate(sp);  // Error(Api)
    r->fast_greedy = sp.greedy() && !sampling::penalties_active(sp) && !sampling::is_structured(sp);
    if (!r->fast_greedy) {
        sampling::SamplerConfig sc;
        sc.vocab_size = info_.vocab_size;
        if (eos_) sc.eos_ids = {*eos_};
        if (sampling::is_structured(sp)) {
            const std::lock_guard lk(vocab_mu_);
            if (!vocab_) vocab_ = sampling::TokenVocab::build(*tok_);
            sc.vocab = vocab_;
        }
        r->sampler.emplace(sampling::Sampler::create(sp, tok_.get(), std::move(sc)));
    }
    {
        const std::lock_guard lk(mu_);
        HALO_CHECK(!stop_, ErrorCode::Api, "generate: engine is shutting down");
        r->id = next_id_++;
        pending_.push_back(r);
    }
    cv_.notify_all();

    // Drain this request's queue on the caller's thread (engine.h threading guarantee).
    std::vector<std::int32_t> delivered;
    bool cancelled = false;
    std::exception_ptr cb_error;
    for (;;) {
        std::unique_lock lk(r->mu);
        r->cv.wait(lk, [&] { return !r->events.empty() || r->done; });
        if (r->events.empty()) break;  // done and drained
        TokenEvent ev = std::move(r->events.front());
        r->events.pop_front();
        lk.unlock();
        if (cancelled) continue;
        delivered.push_back(ev.token);
        bool keep = true;
        try {
            keep = !cb || cb(ev);
        } catch (...) {
            cb_error = std::current_exception();
            keep = false;
        }
        if (!keep) {
            cancelled = true;
            r->cancel.store(true, std::memory_order_relaxed);
            cv_.notify_all();
        }
    }
    GenerateResult res;
    {
        const std::lock_guard lk(r->mu);
        res = r->result;
    }
    if (cancelled) {
        res.tokens = std::move(delivered);
        res.finish = FinishReason::Cancelled;
    }
    if (cb_error) std::rethrow_exception(cb_error);
    return res;
}

// ---- worker --------------------------------------------------------------------------

void CpuEngine::deliver(Request& r, std::vector<TokenEvent>& events, bool done) {
    {
        const std::lock_guard lk(r.mu);
        for (auto& e : events) r.events.push_back(std::move(e));
        if (done) r.done = true;
    }
    events.clear();
    r.cv.notify_all();
}

void CpuEngine::run() noexcept {
    for (;;) {
        std::vector<std::shared_ptr<Request>> admit_now;
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !pending_.empty() || !active_.empty(); });
            if (stop_) break;
            while (!pending_.empty() && admit_now.size() < free_slots_.size()) {
                admit_now.push_back(pending_.front());
                pending_.pop_front();
            }
        }
        for (const auto& r : admit_now) {
            try {
                admit(r);
            } catch (const std::exception& e) {
                std::vector<TokenEvent> none;
                {
                    const std::lock_guard lk(r->mu);
                    r->result.finish = FinishReason::Error;
                    r->result.error = e.what();
                }
                deliver(*r, none, true);
            }
        }
        for (auto it = active_.begin(); it != active_.end();) {
            if (it->r->cancel.load(std::memory_order_relaxed)) {
                retire(*it, FinishReason::Cancelled);
                it = active_.erase(it);
            } else {
                ++it;
            }
        }
        if (active_.empty()) continue;
        try {
            tick();
        } catch (const std::exception& e) {
            // Not recoverable for this tick's sequences (e.g. Error(Kernel) after state was
            // modified): fail them, reset their state, keep serving.
            HALO_ERROR("runtime", "tick failed: {}", e.what());
            for (Active& a : active_) {
                slots_[a.slot]->reset();
                a.fed.clear();
                retire(a, FinishReason::Error, e.what());
            }
            active_.clear();
        }
        for (auto it = active_.begin(); it != active_.end();) {
            if (it->finished) {
                retire(*it, it->reason);
                it = active_.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Shutdown: fail everything still waiting or running.
    std::deque<std::shared_ptr<Request>> rest;
    {
        const std::lock_guard lk(mu_);
        rest.swap(pending_);
    }
    for (Active& a : active_) {
        a.fed.clear();
        retire(a, FinishReason::Error, "engine shut down");
    }
    active_.clear();
    for (const auto& r : rest) {
        std::vector<TokenEvent> none;
        {
            const std::lock_guard lk(r->mu);
            r->result.finish = FinishReason::Error;
            r->result.error = "engine shut down";
        }
        deliver(*r, none, true);
    }
}

void CpuEngine::admit(const std::shared_ptr<Request>& r) {
    Active a;
    a.r = r;
    a.slot = free_slots_.back();
    state::SequenceState& seq = *slots_[a.slot];
    seq.reset();
    if (!seq.mtp_kv && mtp_pool_) seq.mtp_kv.emplace(*mtp_pool_);  // re-attach after a fallback
    const Toks& prompt = r->req.prompt;
    a.history = prompt;
    a.dec.emplace(*tok_);
    // ---- prefix cache (D-013) -------------------------------------------------------------
    std::size_t reused = 0;
    if (cfg_.prefix_cache && !cache_.empty()) {
        std::size_t best_lcp = 0;
        for (const CacheEntry& e : cache_) best_lcp = std::max(best_lcp, lcp(prompt, e.tokens));
        const std::size_t limit = std::min(best_lcp, prompt.size() - 1);  // >= 1 row recomputed for logits
        const state::Checkpoint* ck = nullptr;
        if (limit > 0) {
            ck = ckpts_->find(prompt, limit, [&](const state::Checkpoint& c) {
                return std::any_of(cache_.begin(), cache_.end(), [&](const CacheEntry& e) { return e.id == c.owner; });
            });
        }
        if (ck != nullptr) {
            const std::size_t P = ck->tokens.size();
            auto owner = std::find_if(cache_.begin(), cache_.end(), [&](const CacheEntry& e) { return e.id == ck->owner; });
            seq.kv.share_prefix(*owner->kv, P);
            if (seq.mtp_kv) {
                if (owner->mtp_kv && owner->mtp_kv->length() >= P - 1) {
                    seq.mtp_kv->share_prefix(*owner->mtp_kv, P - 1);
                } else {
                    seq.mtp_kv.reset();  // no MTP rows to reuse: this sequence decodes without MTP
                }
            }
            seq.gdn.restore(ck->gdn);
            seq.last_hidden = ck->last_hidden;
            cache_.splice(cache_.begin(), cache_, owner);
            a.fed.assign(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(P));
            a.prompt_done = P;
            reused = P;
        }
    }
    {
        const std::lock_guard lk(mu_);
        ++stats_.active_sequences;
        if (cfg_.prefix_cache) (reused > 0 ? stats_.prefix_cache_hits : stats_.prefix_cache_misses)++;
        stats_.prefix_cache_reused_tokens += reused;
    }
    {
        const std::lock_guard lk(r->mu);
        r->result.prompt_tokens = prompt.size();
        r->result.cached_prompt_tokens = reused;
    }
    // Checkpoint positions for the rest of the prompt (D-013): caller hints (message
    // boundaries), every `spacing` tokens, and N - tail; ascending, >= min spacing apart.
    if (cfg_.prefix_cache && ckpts_->budget_bytes() > 0) {
        const std::size_t N = prompt.size();
        std::vector<std::size_t> cand;
        for (const std::size_t h : r->req.checkpoint_hints) {
            if (h < N) cand.push_back(h);
        }
        for (std::size_t p = ckpt_spacing_; p < N; p += ckpt_spacing_) cand.push_back(p);
        std::sort(cand.begin(), cand.end());
        std::size_t last = a.prompt_done;
        for (const std::size_t p : cand) {
            if (p > a.prompt_done && p >= last + opts_.checkpoint_min_spacing) {
                a.ckpt_at.push_back(p);
                last = p;
            }
        }
        // N - tail is always taken (it serves an exact re-submit); it is exempt from the
        // spacing rule so a message-boundary hint just before it is not dropped.
        if (N > opts_.checkpoint_tail && N - opts_.checkpoint_tail > a.prompt_done) {
            const std::size_t tail = N - opts_.checkpoint_tail;
            if (a.ckpt_at.empty() || a.ckpt_at.back() < tail) a.ckpt_at.push_back(tail);
        }
    }
    free_slots_.pop_back();
    active_.push_back(std::move(a));
}

bool CpuEngine::evict_one_cache_entry() {
    if (cache_.empty()) return false;
    ckpts_->erase_owner(cache_.back().id);
    cache_.pop_back();
    return true;
}

void CpuEngine::ensure_capacity(std::size_t trunk_rows, std::size_t mtp_rows) {
    const std::size_t bt = opts_.kv_block_tokens;
    const std::size_t seqs = active_.size();
    const auto need = [&](std::size_t rows) { return ceil_div(rows, bt) + 2 * seqs; };  // + partial + COW per sequence
    while (kv_pool_->free_blocks() < need(trunk_rows) && evict_one_cache_entry()) {
    }
    if (mtp_pool_) {
        while (mtp_pool_->free_blocks() < need(mtp_rows) && evict_one_cache_entry()) {
        }
    }
}

void CpuEngine::checkpoint(Active& a) {
    state::SequenceState& seq = *slots_[a.slot];
    state::Checkpoint c;
    c.tokens = a.fed;
    c.gdn = seq.gdn.snapshot();
    c.last_hidden = seq.last_hidden;
    c.owner = a.r->id;
    (void)ckpts_->insert(std::move(c));  // false: larger than the whole budget (always-recompute)
}

bool CpuEngine::emit(Active& a, std::int32_t tok, std::vector<TokenEvent>& events) {
    const auto now = Clock::now();
    if (!a.t_first) a.t_first = now;
    a.t_last = now;
    a.generated.push_back(tok);
    a.history.push_back(tok);
    TokenEvent ev;
    ev.token = tok;
    ev.is_eos = eos_ && tok == *eos_;
    if (!ev.is_eos) {
        ev.piece = a.dec->push(tok);
        a.text += ev.piece;
    }
    bool done = false;
    if (ev.is_eos) {
        done = true;
        a.reason = FinishReason::Stop;
    }
    for (const auto& s : a.r->req.stop_token_seqs) {
        if (!done && a.generated.size() >= s.size() &&
            std::equal(s.begin(), s.end(), a.generated.end() - static_cast<std::ptrdiff_t>(s.size()))) {
            done = true;
            a.reason = FinishReason::Stop;
        }
    }
    for (const auto& s : a.r->req.stop_strings) {
        if (!done && !s.empty() && !ev.piece.empty()) {
            // Only matches that end inside this token's piece are new.
            const std::size_t from = a.text.size() > s.size() + ev.piece.size() ? a.text.size() - s.size() - ev.piece.size() : 0;
            if (a.text.find(s, from) != std::string::npos) {
                done = true;
                a.reason = FinishReason::Stop;
            }
        }
    }
    if (!done && a.generated.size() >= a.r->max_new) {
        done = true;
        a.reason = FinishReason::Length;
    }
    if (done && !ev.is_eos) ev.piece += a.dec->flush();
    events.push_back(std::move(ev));
    {
        const std::lock_guard lk(mu_);
        ++stats_.tokens_generated;
    }
    a.finished = done;
    return done;
}

void CpuEngine::tick() {
    const std::size_t n = active_.size();
    std::vector<Active*> who;
    std::vector<speculative::StepRequest> reqs(n);
    const std::span<const std::int32_t> eos_span = eos_ ? std::span<const std::int32_t>(&*eos_, 1) : std::span<const std::int32_t>();
    std::size_t trunk_rows = 0, mtp_rows = 0, prefill_rows = 0;
    for (Active& a : active_) who.push_back(&a);
    for (std::size_t i = 0; i < n; ++i) {
        Active& a = *who[i];
        state::SequenceState& seq = *slots_[a.slot];
        speculative::StepRequest& q = reqs[i];
        q.seq = &seq;
        q.greedy = a.r->fast_greedy;
        const Toks& prompt = a.r->req.prompt;
        if (a.prompt_done < prompt.size()) {
            std::size_t end = std::min(prompt.size(), a.prompt_done + prefill_chunk_);
            if (a.next_ckpt < a.ckpt_at.size()) end = std::min(end, a.ckpt_at[a.next_ckpt]);
            q.prefill = std::span(prompt).subspan(a.prompt_done, end - a.prompt_done);
            q.want_output = end == prompt.size();
            trunk_rows += q.prefill.size();
            prefill_rows += q.prefill.size();
            mtp_rows += q.prefill.size() + seq.mtp_queue_size();
        } else {
            q.token = a.pending;
            q.max_draft = a.r->fast_greedy ? max_draft_ : 0;
            a.forced.clear();
            if (opts_.draft_hook && a.r->fast_greedy && max_draft_ > 0) {
                a.forced = opts_.draft_hook(a.r->req.prompt, a.generated);
                if (a.forced.size() > max_draft_) a.forced.resize(max_draft_);
                q.forced_drafts = a.forced;
            }
            q.max_emit = a.r->max_new - a.generated.size();
            q.stop_tokens = eos_span;
            trunk_rows += 1 + q.max_draft;
            mtp_rows += 1 + q.max_draft + seq.mtp_queue_size();
        }
    }
    ensure_capacity(trunk_rows, mtp_rows);

    speculative::Tick t;
    bool fallback = false;
    const std::uint64_t trunk0 = spec_->metrics().trunk_passes, passes0 = spec_->metrics().weight_passes;
    for (int attempt = 0;; ++attempt) {
        try {
            spec_->step(reqs, t);
            break;
        } catch (const Error& e) {
            if (e.code() != ErrorCode::Memory || attempt > 8) throw;
            // Pre-tick state is intact (Memory errors are atomic). Free cached blocks first,
            // then fall back to plain decoding without MTP (FR-009 / M3 constraint).
            if (evict_one_cache_entry()) continue;
            bool had_mtp = false;
            for (auto& q : reqs) {
                if (q.seq->mtp_kv && (q.max_draft > 0 || q.seq->mtp_queue_size() > 0 || !q.prefill.empty())) {
                    q.seq->mtp_kv.reset();
                    q.seq->mtp_queue_tokens.clear();
                    q.seq->mtp_queue_hidden.clear();
                    q.max_draft = 0;
                    had_mtp = true;
                }
            }
            if (!had_mtp) throw;
            fallback = true;
            HALO_WARN("runtime", "MTP KV exhausted ({}); continuing those sequences without MTP", e.what());
        }
    }

    // ---- outputs -------------------------------------------------------------------------
    TickInfo info;
    info.sequences = n;
    info.prefill_rows = prefill_rows;
    info.mtp_fallback = fallback;
    const auto& mt = spec_->metrics();
    // Cumulative deltas: a tick that ran several speculator steps would show up here.
    info.weight_passes = static_cast<std::uint32_t>(mt.weight_passes - passes0);
    info.trunk_passes = static_cast<std::uint32_t>(mt.trunk_passes - trunk0);
    info.measured_weight_bytes = mt.last_cost.weight_bytes;
    for (std::size_t i = 0; i < n; ++i) {
        Active& a = *who[i];
        const speculative::StepOutput& o = t.out[i];
        const speculative::StepRequest& q = reqs[i];
        std::vector<TokenEvent> events;
        if (!q.prefill.empty()) {
            a.fed.insert(a.fed.end(), q.prefill.begin(), q.prefill.end());
            a.prompt_done += q.prefill.size();
            if (a.next_ckpt < a.ckpt_at.size() && a.prompt_done == a.ckpt_at[a.next_ckpt]) {
                checkpoint(a);
                ++a.next_ckpt;
            }
            if (q.want_output) {
                const std::int32_t tok = a.r->fast_greedy ? o.tokens.at(0) : a.r->sampler->sample(o.logits, a.history);
                if (a.r->sampler) a.r->sampler->accept(tok);
                a.pending = tok;
                (void)emit(a, tok, events);
            }
        } else {
            info.decode_rows += 1 + o.drafts.size();
            info.max_draft = std::max(info.max_draft, o.drafts.size());
            info.drafted += o.drafts.size();
            info.accepted += o.accepted;
            a.drafted += o.drafts.size();
            a.accepted += o.accepted;
            if (a.r->fast_greedy) {
                // Rows fed: pending + accepted drafts == pending + tokens[0 .. n_keep-1).
                a.fed.push_back(a.pending);
                for (std::size_t j = 0; j + 1 < o.tokens.size(); ++j) a.fed.push_back(o.tokens[j]);
                a.pending = o.tokens.back();
                for (const std::int32_t tok : o.tokens) {
                    if (emit(a, tok, events)) break;
                }
            } else {
                a.fed.push_back(a.pending);
                const std::int32_t tok = a.r->sampler->sample(o.logits, a.history);
                a.r->sampler->accept(tok);
                a.pending = tok;
                (void)emit(a, tok, events);
            }
        }
        if (!events.empty()) deliver(*a.r, events, false);
        // The token list must describe exactly the KV rows (prefix-cache entries share them).
        HALO_CHECK(a.fed.size() == slots_[a.slot]->length(), ErrorCode::Kernel,
                   "runtime: sequence bookkeeping out of sync ({} tokens recorded, {} KV rows)", a.fed.size(),
                   slots_[a.slot]->length());
    }
    info.predicted_bytes = spec_->cost_model().spec_step(info.max_draft);
    {
        const std::lock_guard lk(mu_);
        ++stats_.ticks;
        stats_.speculative_attempts += info.drafted;
        stats_.speculative_accepts += info.accepted;
        if (fallback) ++stats_.mtp_fallbacks;
        stats_.last_tick_weight_passes = info.weight_passes;
        stats_.last_tick_predicted_bytes = info.predicted_bytes;
    }
    if (opts_.on_tick) opts_.on_tick(info);
}

void CpuEngine::retire(Active& a, FinishReason why, const std::string& error) {
    state::SequenceState& seq = *slots_[a.slot];
    // Keep the sequence for prefix reuse (not after an error: its state may be unspecified).
    if (cfg_.prefix_cache && why != FinishReason::Error && !a.fed.empty() && cache_cap_ > 0) {
        try {
            if (seq.mtp_kv && seq.mtp_queue_size() > 0) {
                state::SequenceState* one[] = {&seq};
                spec_->flush_mtp(one);
            }
        } catch (const Error& e) {
            HALO_WARN("runtime", "prefix cache: MTP catch-up for a retired sequence failed: {}", e.what());
            seq.mtp_kv.reset();
        }
        state::Checkpoint c;
        c.tokens = a.fed;
        c.gdn = seq.gdn.snapshot();
        c.last_hidden = seq.last_hidden;
        c.owner = a.r->id;
        CacheEntry e;
        e.id = a.r->id;
        e.tokens = a.fed;
        e.kv = std::make_unique<kv_cache::SequenceKv>(std::move(seq.kv));
        if (seq.mtp_kv && seq.mtp_queue_size() == 0) e.mtp_kv = std::make_unique<kv_cache::SequenceKv>(std::move(*seq.mtp_kv));
        cache_.push_front(std::move(e));
        (void)ckpts_->insert(std::move(c));
        while (cache_.size() > cache_cap_) evict_one_cache_entry();
    } else {
        ckpts_->erase_owner(a.r->id);
    }
    seq.reset();
    free_slots_.push_back(a.slot);

    {
        // Before the done event: a caller that returns from generate() never sees its own
        // sequence still counted as active.
        const std::lock_guard lk(mu_);
        --stats_.active_sequences;
    }
    Request& r = *a.r;
    std::vector<TokenEvent> none;
    {
        const std::lock_guard lk(r.mu);
        GenerateResult& res = r.result;
        res.tokens = a.generated;
        res.finish = why;
        res.error = error;
        res.draft_tokens = a.drafted;
        res.accepted_draft_tokens = a.accepted;
        if (a.t_first) {
            res.ttft_ms = ms_between(r.t_submit, *a.t_first);
            const double dt = ms_between(*a.t_first, a.t_last) / 1000.0;
            res.decode_tps = a.generated.size() > 1 && dt > 0 ? static_cast<double>(a.generated.size() - 1) / dt : 0.0;
        }
    }
    deliver(r, none, true);
}

}  // namespace

std::unique_ptr<Engine> create_cpu_engine(const EngineConfig& cfg, const CpuEngineOptions& opts) {
    return std::make_unique<CpuEngine>(cfg, opts);
}

std::unique_ptr<Engine> create_engine(const EngineConfig& cfg) { return create_cpu_engine(cfg, {}); }

}  // namespace halo::runtime
