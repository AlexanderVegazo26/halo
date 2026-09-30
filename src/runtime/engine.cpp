// Reference Engine: one worker thread, one batched tick per step, per-request token
// queues drained on the caller's thread, prefix cache (KV sharing + GDN checkpoints),
// MTP speculative decoding with a k = 0 fallback on MTP-KV exhaustion.
// BI-6: the forward runs on the backend named by EngineConfig::backend (cpu | vulkan |
// hip; "auto" = cpu until a measured GPU default policy exists, WS-BI-4). GDN/conv state
// (the §5.3 ring) and the KV pools are State-arena buffers of that backend (WS-BI-2):
// device-resident on GPU backends, zero-copy host memory on the CPU backend.

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <list>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "halo/autotune/lookup.h"
#include "halo/backend/backend.h"
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

#if defined(HALO_RUNTIME_VULKAN)
#include "backend/vulkan_adapter.h"
#include "halo/backends/vulkan/context.h"
#endif
#if defined(HALO_RUNTIME_HIP)
#include "halo/backend/hip_backend.h"
#endif

namespace halo::runtime {

namespace {

using Clock = std::chrono::steady_clock;
using Toks = std::vector<std::int32_t>;

/// KV pool element format from the environment (opt-in; default fp32, so behaviour and goldens
/// are unchanged): HALO_KV_FP16=1 stores fp16, HALO_KV_TYPE=q8 stores 8-bit values with a
/// per-32-group scale (HALO_KV_TYPE also accepts f32 / f16). Vulkan backend only -- the caller
/// rejects it elsewhere. An unknown value or a contradictory pair is Error(Config), never a
/// silent fallback.
backend::KvType kv_type_from_env() {
    const char* fp16 = std::getenv("HALO_KV_FP16");
    const char* type = std::getenv("HALO_KV_TYPE");
    const std::string_view f = fp16 ? fp16 : "";
    const std::string_view t = type ? type : "";
    HALO_CHECK(f.empty() || f == "0" || f == "1", ErrorCode::Config, "HALO_KV_FP16 must be 0 or 1, got '{}'", f);
    backend::KvType out = backend::KvType::F32;
    if (t.empty() || t == "f32" || t == "fp32") {
        out = backend::KvType::F32;
    } else if (t == "f16" || t == "fp16") {
        out = backend::KvType::F16;
    } else if (t == "q8") {
        out = backend::KvType::Q8;
    } else {
        throw_error(ErrorCode::Config, "HALO_KV_TYPE must be f32, f16 or q8, got '{}'", t);
    }
    if (f == "1") {
        HALO_CHECK(t.empty() || out == backend::KvType::F16, ErrorCode::Config,
                   "HALO_KV_FP16=1 contradicts HALO_KV_TYPE={}", t);
        out = backend::KvType::F16;
    }
    return out;
}

/// Prompt-lookup drafts (no model cost): find the most recent earlier occurrence of the last
/// `min_match`..`max_match` tokens of (prompt ++ generated) and propose the up-to-`k` tokens
/// that followed it. The trunk verifies every proposed token, so greedy output is unchanged;
/// a miss only costs the extra verify rows. Empty = no match (MTP drafts instead).
Toks ngram_draft(std::span<const std::int32_t> prompt, std::span<const std::int32_t> generated, std::size_t k,
                 std::size_t min_match = 3, std::size_t max_match = 8) {
    const std::size_t n = prompt.size() + generated.size();
    if (k == 0 || n < min_match + 1) return {};
    const auto at = [&](std::size_t i) { return i < prompt.size() ? prompt[i] : generated[i - prompt.size()]; };
    for (std::size_t m = std::min(max_match, n - 1); m >= min_match; --m) {
        // Scan back for the latest start p < n - m whose m tokens equal the last m tokens.
        for (std::size_t p = n - m; p-- > 0;) {
            std::size_t j = 0;
            while (j < m && at(p + j) == at(n - m + j)) ++j;
            if (j < m) continue;
            Toks out;
            for (std::size_t i = p + m; i < n && out.size() < k; ++i) out.push_back(at(i));
            if (!out.empty()) return out;
        }
    }
    return {};
}

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// ---- backend selection (BI-6) ----------------------------------------------------------

#if defined(HALO_RUNTIME_VULKAN) || defined(HALO_RUNTIME_HIP)
/// Constructs the GPU backend named by EngineConfig::backend. "auto" never selects one:
/// there is no measured default policy yet (WS-BI-4), so a GPU backend is explicit opt-in.
/// Throws Error(Device) when the device/driver is missing.
std::unique_ptr<backend::Backend> make_gpu_backend(const std::string& name) {
#if defined(HALO_RUNTIME_VULKAN)
    if (name == "vulkan") return backend::make_vulkan_backend(vulkan::Context::create({}));
#endif
#if defined(HALO_RUNTIME_HIP)
    if (name == "hip") {
        // Device mode. Construction succeeds (weights are read-only imports, copied into
        // VRAM). With WS-BI-2 the forward no longer needs writable host imports (GDN ring +
        // KV pools are backend buffers), but the HIP device path is not covered by tests yet.
        backend::HipBackendOptions o;
        o.mode = backend::HipMode::Device;
        return backend::make_hip_backend(o);
    }
#endif
    throw_error(ErrorCode::Unsupported, "backend '{}' is not available in this build", name);
}
#endif

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
    bool deadline_hit = false;  ///< worker-written before done (read under mu after done)
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
    std::size_t delivered = 0;    ///< generated tokens handed to the request's queue
    bool finished = false;
    FinishReason reason = FinishReason::Stop;
    std::string error;            ///< set with reason == Error by a per-sequence failure (R-1)
};

/// A retired sequence kept for prefix reuse.
struct CacheEntry {
    std::uint64_t id = 0;
    Toks tokens;  ///< == kv rows
    std::unique_ptr<kv_cache::SequenceKv> kv;
    std::unique_ptr<kv_cache::SequenceKv> mtp_kv;  ///< rows for positions 1..tokens.size()-1, or null
    /// L4: false when ckpts_->insert() failed (e.g. an oversized checkpoint) for this entry.
    /// admit() can only ever match a live checkpoint (ckpts_->find), so such an entry holds KV
    /// pool blocks hostage for nothing; evict_one_cache_entry() reclaims these first.
    bool has_checkpoint = true;
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
    /// L5 index upkeep: call exactly once when `it` is inserted into / about to be removed
    /// from cache_.
    void index_cache_entry(std::list<CacheEntry>::iterator it);
    void unindex_cache_entry(std::list<CacheEntry>::iterator it);
    /// Emits one token for `a`; returns true if the sequence finished.
    bool emit(Active& a, std::int32_t tok, std::vector<TokenEvent>& events);
    /// Never throws (R-2): the cache part is best-effort, the done delivery always happens.
    void retire(Active& a, FinishReason why, std::string error = {}) noexcept;
    void cache_retired(Active& a);
    /// Ends a request that never got (or no longer has) a sequence. Never throws.
    static void finish_unscheduled(Request& r, FinishReason why, std::string error) noexcept;
    /// R-3: the request's stop token / deadline, if either fired (Cancelled wins).
    [[nodiscard]] static std::optional<FinishReason> expired(const Request& r) noexcept;
    void deliver(Request& r, std::vector<TokenEvent>& events, bool done);
    void checkpoint(Active& a);
    void apply_profile();

    EngineConfig cfg_;
    CpuEngineOptions opts_;
    std::unique_ptr<model::NormalizedModel> nm_;
    std::unique_ptr<cpu::ThreadPool> pool_;       // CPU backend only
    std::unique_ptr<backend::Backend> backend_;   // GPU backends only (BI-6); before model_
    std::unique_ptr<models::Qwen35> model_;
    std::unique_ptr<tokenizer::Tokenizer> tok_;
    std::unique_ptr<chat::ChatTemplate> tmpl_;
    std::shared_ptr<const sampling::TokenVocab> vocab_;
    std::mutex vocab_mu_;
    ModelInfo info_;
    memory::MemoryPlan plan_;
    std::optional<std::int32_t> eos_;
    std::size_t max_draft_ = 0;
    /// Prompt-lookup drafts ahead of MTP (opt-in: HALO_NGRAM_DRAFT=1). Greedy requests only.
    bool ngram_draft_ = [] {
        const char* e = std::getenv("HALO_NGRAM_DRAFT");
        return e != nullptr && e[0] == '1';
    }();
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
    // L5: admit()'s best-LCP scan only needs entries sharing the query's first token -- any
    // entry that differs there has lcp() == 0, which can never beat the scan's own starting
    // best_lcp of 0, so bucketing by first token is exact, not an approximation. Iterators into
    // a std::list stay valid across push_front/erase-elsewhere/splice, so this index only needs
    // updating where cache_ itself gains or loses an entry.
    std::unordered_map<std::int32_t, std::vector<std::list<CacheEntry>::iterator>> by_first_tok_;
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
    const bool cpu_path = cfg.backend == "cpu" || cfg.backend == "auto";
    HALO_CHECK(cpu_path || cfg.backend == "vulkan" || cfg.backend == "hip", ErrorCode::Unsupported,
               "unknown backend '{}' (cpu | vulkan | hip | auto)", cfg.backend);
    if (!cpu_path && !opts.backend_factory) {
#if !defined(HALO_RUNTIME_VULKAN)
        HALO_CHECK(cfg.backend != "vulkan", ErrorCode::Unsupported, "backend 'vulkan' is not available in this build");
#endif
#if !defined(HALO_RUNTIME_HIP)
        HALO_CHECK(cfg.backend != "hip", ErrorCode::Unsupported, "backend 'hip' is not available in this build");
#endif
    }
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
    if (cpu_path) pool_ = std::make_unique<cpu::ThreadPool>(threads_);
    // Built before the model (M2): the model needs the tokenizer's real vocab size to clamp
    // the fused greedy argmax away from GGUF LM-head padding rows, and the tokenizer itself
    // only depends on GGUF metadata, not on the model.
    const model::TokenizerMetadata meta = nm_->tokenizer();
    tok_ = std::make_unique<tokenizer::Tokenizer>(tokenizer::Tokenizer::from_spec(models::vocab_spec(meta)));
    HALO_CHECK(meta.chat_template.has_value() && !meta.chat_template->empty(), ErrorCode::Model,
               "GGUF has no tokenizer.chat_template");
    models::Qwen35Options mo;
    mo.gdn_chunk = gdn_chunk_;
    mo.valid_vocab = tok_->vocab_size();
    if (cpu_path) {
        model_ = std::make_unique<models::Qwen35>(*nm_, pool_.get(), mo);
    } else {
        if (opts_.backend_factory) {
            backend_ = opts_.backend_factory();
        } else {
#if defined(HALO_RUNTIME_VULKAN) || defined(HALO_RUNTIME_HIP)
            backend_ = make_gpu_backend(cfg.backend);
#else
            throw_error(ErrorCode::Unsupported, "backend '{}' is not available in this build", cfg.backend);
#endif
        }
        HALO_INFO("runtime", "backend '{}' selected; GDN + KV state is device-resident (WS-BI-2)", cfg.backend);
        model_ = std::make_unique<models::Qwen35>(*nm_, *backend_, mo);
    }
    const auto piece = [&](std::optional<std::int32_t> id) { return id ? tok_->token_to_piece(*id) : std::string(); };
    tmpl_ = std::make_unique<chat::ChatTemplate>(std::string(*meta.chat_template), piece(tok_->bos()), piece(tok_->eos()));
    eos_ = opts.eos_token ? opts.eos_token : tok_->eos();

    const auto& hp = nm_->hparams();
    const std::string* gname = nm_->gguf().get_string("general.name");
    info_.id = gname != nullptr && !gname->empty() ? *gname : std::filesystem::path(cfg.model_path).stem().string();
    info_.architecture = "qwen35";
    // An explicit --ctx above the model's trained context is refused, not silently reduced (the
    // /v1/models context_length would otherwise disagree with what the user asked for).
    HALO_CHECK(!cfg.max_context_explicit || hp.context_length == 0 || cfg.max_context <= hp.context_length, ErrorCode::Config,
               "--ctx {} exceeds the model's trained context length {}; lower --ctx (or leave it unset for the default)",
               cfg.max_context, hp.context_length);
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
    // Activations / workspace scale with the rows of one forward, i.e. the prefill chunk (and
    // NOT with the context: the CPU and Vulkan attention are one-pass / online-softmax with no
    // per-context scratch, hence the planner's default flash_attention = true -- no
    // ubatch x n_head x ctx score buffer). Plan for the chunk the engine will actually run.
    pr.ubatch = static_cast<std::uint32_t>(std::max(gdn_chunk_, opts.prefill_chunk / gdn_chunk_ * gdn_chunk_));
    pr.batch = std::max(pr.batch, pr.ubatch);
    // KV element format: fp32 unless HALO_KV_FP16=1 / HALO_KV_TYPE=q8 (Vulkan only).
    const backend::KvType kv_type = kv_type_from_env();
    if (kv_type != backend::KvType::F32) {
        const backend::Backend& kvbe = backend_ ? *backend_ : model_->backend();
        HALO_CHECK(kvbe.kind() == backend::Kind::Vulkan, ErrorCode::Config,
                   "KV cache type {} (HALO_KV_FP16 / HALO_KV_TYPE) is implemented only on the Vulkan backend; this run uses "
                   "the {} backend (unset the variable or select --backend vulkan)",
                   backend::to_string(kv_type), backend::to_string(kvbe.kind()));
        HALO_INFO("runtime", "KV cache stored as {} (opt-in)", backend::to_string(kv_type));
    }
    pr.kv_dtype = kv_type == backend::KvType::F16 ? memory::DtypeSize::f16()
                  : kv_type == backend::KvType::Q8 ? memory::DtypeSize{"q8_kv", 32, 36}  // [fp32 scale | 32 x int8]
                                                   : memory::DtypeSize::f32();  // the CPU reference stores fp32 KV
    pr.mtp_enabled = mtp;
    pr.mtp_draft_depth = static_cast<std::uint32_t>(max_draft_);
    pr.prefix_checkpoints.enabled = cfg.prefix_cache;
    if (opts.checkpoint_spacing > 0) pr.prefix_checkpoints.spacing_tokens = static_cast<std::uint32_t>(opts.checkpoint_spacing);
    pr.max_memory = cfg.max_memory_bytes;
    try {
        plan_ = memory::plan_memory_or_throw(model_shape(*nm_, *model_), pr, hw.tiers);
    } catch (const Error& e) {
        // An explicit --ctx (with --parallel sequences of it) the memory tiers cannot hold is a
        // configuration error naming the request, not a bare planner refusal. A user max_memory
        // cap keeps its own Memory error.
        if (!cfg.max_context_explicit || cfg.max_memory_bytes || e.code() != ErrorCode::Memory) throw;
        throw_error(ErrorCode::Config,
                    "--ctx {} x --parallel {} cannot be honoured, lower --ctx or --parallel (or store KV as fp16/q8, "
                    "HALO_KV_FP16=1 / HALO_KV_TYPE=q8): {}",
                    info_.context_length, cfg.max_sequences, e.what());
    }
    for (const auto& n : plan_.notes) HALO_INFO("runtime", "memory plan: {}", n);
    ckpt_spacing_ = pr.prefix_checkpoints.spacing_tokens;

    // ---- pools, slots ------------------------------------------------------------------
    // Prefill chunks stay a multiple of the GDN chunk (chunk boundaries line up).
    prefill_chunk_ = std::max(gdn_chunk_, opts.prefill_chunk / gdn_chunk_ * gdn_chunk_);
    const std::size_t bt = opts.kv_block_tokens;
    kv_cache::KvLayout trunk_layout = model_->kv_layout(bt);
    trunk_layout.type = kv_type;
    kv_cache::KvLayout mtp_layout = model_->mtp_kv_layout(bt);
    mtp_layout.type = kv_type;
    const auto blocks_for = [&](std::size_t ctx) { return ceil_div(ctx + max_draft_ + 1, bt) + 1; };  // +1: one COW copy
    std::size_t per_seq = blocks_for(info_.context_length);
    cache_cap_ = cfg.prefix_cache ? (opts.prefix_cache_entries > 0 ? opts.prefix_cache_entries : cfg.max_sequences) : 0;
    // ---- KV pool sizing (large contexts) -------------------------------------------------
    // A backend caps ONE buffer (RADV: min(maxMemoryAllocationSize, maxStorageBufferRange) = 4 GiB)
    // and the attention / kv-write kernels bind one pool buffer. The pool is therefore split
    // per layer (kv_cache::Placement::PerLayer) when its contiguous image would not fit, which
    // lifts the limit by n_layers (16x for the 27B model: 128k tokens x 2 sequences = 32 GiB in fp32
    // is 2 GiB per layer; x 4 sequences is 4 GiB + the COW/draft slack per layer, which does NOT fit
    // under a 4 GiB cap in fp32 -- refused below -- but does in fp16/q8). The pool is sized for every
    // sequence at the full context; the prefix-cache
    // share (cache_cap_ extra sequences' worth of blocks, opportunistic: cached prefixes are
    // evicted under pressure) is what shrinks first, so the cache stays enabled. If the
    // sequences themselves do not fit (per-buffer cap, or the pools would exceed the VRAM the plan
    // leaves), an explicit --ctx is refused (Error(Config)) and a defaulted one is halved with a
    // warning. (backend_ is set only on the GPU paths; the CPU path's internal backend is the model's.)
    const backend::Backend& active_backend = backend_ ? *backend_ : model_->backend();
    const std::uint64_t buf_cap = opts.kv_buffer_cap_bytes.value_or(active_backend.limits().max_pool_buffer_bytes);
    const std::uint64_t trunk_blk = trunk_layout.block_bytes();
    const std::uint64_t mtp_blk = mtp ? mtp_layout.block_bytes() : 0;
    // Free-VRAM guard: the plan counts max_sequences x context of KV, not the cache share nor the
    // per-sequence COW/draft slack. The sequences' blocks may use the plan's KV budget plus all
    // FREE VRAM the plan leaves (hard limit: beyond it the allocation would fail); the prefix-cache
    // share only the plan's safety-reduced budget, so the reserve for the activation / workspace
    // estimate (formula, not measured) is not eaten by cache blocks.
    std::uint64_t vram_hard = 0, vram_soft = 0;  // bytes for both pools; 0 = no guard (KV not planned into VRAM)
    if (plan_.tier_of(memory::Component::KvCache) == hardware::MemoryTier::Vram) {
        for (const auto& t : plan_.tiers) {
            if (t.tier != hardware::MemoryTier::Vram) continue;
            vram_hard = plan_.sizes.kv + plan_.sizes.mtp_kv + (t.available > t.planned ? t.available - t.planned : 0);
            vram_soft = plan_.sizes.kv + plan_.sizes.mtp_kv + (t.budget > t.planned ? t.budget - t.planned : 0);
        }
    }
    std::size_t default_blocks = per_seq * (cfg.max_sequences + cache_cap_);
    if (!opts.kv_blocks) {
        std::size_t cap_blocks = kv_cache::KvPool::max_blocks(trunk_layout, kv_cache::Placement::PerLayer, buf_cap);
        if (mtp) cap_blocks = std::min(cap_blocks, kv_cache::KvPool::max_blocks(mtp_layout, kv_cache::Placement::PerLayer, buf_cap));
        bool ctx_reduced = false;
        for (;;) {
            std::size_t limit = cap_blocks;  // hard: the sequences' blocks
            std::size_t soft = cap_blocks;   // the sequences' blocks + the cache share
            std::string bound = buf_cap != 0 ? std::format("per-buffer cap {} bytes, split per layer", buf_cap) : "block id range";
            if (vram_hard != 0) {
                const std::size_t hard_blocks = static_cast<std::size_t>(vram_hard / (trunk_blk + mtp_blk));
                if (hard_blocks < limit) {
                    limit = hard_blocks;
                    bound = std::format("{} bytes of free VRAM for KV pools", vram_hard);
                }
                soft = std::min(soft, static_cast<std::size_t>(vram_soft / (trunk_blk + mtp_blk)));
            }
            const std::size_t need = per_seq * cfg.max_sequences;
            if (need <= limit) {
                default_blocks = std::min(per_seq * (cfg.max_sequences + cache_cap_), std::max(std::min(soft, limit), need));
                break;
            }
            HALO_CHECK(!cfg.max_context_explicit && info_.context_length > 1024, ErrorCode::Config,
                       "context {} x {} sequence(s) needs {} KV blocks ({} bytes per block) but only {} fit ({}): lower "
                       "--ctx or --parallel, or store KV as fp16/q8 (HALO_KV_FP16=1 / HALO_KV_TYPE=q8, Vulkan)",
                       info_.context_length, cfg.max_sequences, need, trunk_blk + mtp_blk, limit, bound);
            info_.context_length = std::max<std::size_t>(1024, info_.context_length / 2);
            per_seq = blocks_for(info_.context_length);
            ctx_reduced = true;
        }
        if (ctx_reduced) {
            HALO_WARN("runtime",
                      "context reduced to {} tokens (default --ctx): {} sequence(s) at the requested context do not fit the KV "
                      "pool limits; pass --ctx explicitly to make this an error",
                      info_.context_length, cfg.max_sequences);
        }
        if (default_blocks < per_seq * (cfg.max_sequences + cache_cap_)) {
            HALO_INFO("runtime", "prefix-cache share of the KV pool reduced to {} of {} blocks (pool limit); the cache stays on",
                      default_blocks - per_seq * cfg.max_sequences, per_seq * cache_cap_);
        }
    }
    const std::size_t blocks = opts.kv_blocks.value_or(default_blocks);
    const std::size_t mtp_blocks = opts.mtp_kv_blocks.value_or(default_blocks);
    const auto placement_for = [&](const kv_cache::KvLayout& l, std::size_t n, const char* what) {
        if (buf_cap == 0 || kv_cache::KvPool::segment_bytes_for(l, kv_cache::Placement::Single, n) <= buf_cap) {
            return kv_cache::Placement::Single;
        }
        HALO_CHECK(kv_cache::KvPool::segment_bytes_for(l, kv_cache::Placement::PerLayer, n) <= buf_cap, ErrorCode::Config,
                   "{} KV pool of {} blocks needs {}-byte buffers even when split per layer; the backend's per-buffer cap is {} "
                   "bytes: lower --ctx or --parallel, or store KV as fp16/q8",
                   what, n, kv_cache::KvPool::segment_bytes_for(l, kv_cache::Placement::PerLayer, n), buf_cap);
        return kv_cache::Placement::PerLayer;
    };
    kv_pool_ = std::make_unique<kv_cache::KvPool>(trunk_layout, blocks, placement_for(trunk_layout, blocks, "trunk"));
    // ADR-001 §5.2 (WS-BI-2 stage 2): the pool image is a State-arena buffer of the model's
    // backend — device-resident on GPU backends, zero-copy host memory on the CPU backend.
    kv_pool_->attach(model_->backend());
    if (mtp) {
        mtp_pool_ = std::make_unique<kv_cache::KvPool>(mtp_layout, mtp_blocks, placement_for(mtp_layout, mtp_blocks, "MTP"));
        mtp_pool_->attach(model_->backend());
    }
    {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        const std::uint64_t pool_bytes = kv_pool_->total_bytes() + (mtp_pool_ ? mtp_pool_->total_bytes() : 0);
        info_.kv_pool_bytes = pool_bytes;
        info_.kv_segments = kv_pool_->segment_count();
        HALO_INFO("runtime",
                  "KV pool: trunk {} blocks x {} B = {:.2f} GiB in {} {} segment(s) of {:.2f} GiB{}; context {} x {} sequence(s) "
                  "+ {} prefix-cache slot(s); {} B per token; per-buffer cap {}",
                  kv_pool_->total_blocks(), trunk_blk, static_cast<double>(kv_pool_->total_bytes()) / kGiB,
                  kv_pool_->segment_count(), kv_cache::to_string(kv_pool_->placement()),
                  static_cast<double>(kv_pool_->segment_bytes()) / kGiB,
                  mtp_pool_ ? std::format("; MTP {} blocks = {:.2f} GiB in {} segment(s)", mtp_pool_->total_blocks(),
                                          static_cast<double>(mtp_pool_->total_bytes()) / kGiB, mtp_pool_->segment_count())
                            : std::string(),
                  info_.context_length, cfg.max_sequences, cache_cap_, trunk_blk / bt,
                  buf_cap != 0 ? std::format("{} B", buf_cap) : std::string("none"));
    }
    free_slots_.reserve(cfg.max_sequences);
    for (std::size_t i = 0; i < cfg.max_sequences; ++i) {
        slots_.push_back(std::make_unique<state::SequenceState>(*kv_pool_, mtp_pool_.get(), model_->gdn_shape(), max_draft_ + 1));
        // ADR-001 §5.2/§5.3: the GDN ring slab is a State-arena buffer of the model's backend
        // (device-resident on GPU backends; zero-copy host memory on the CPU backend).
        slots_.back()->gdn.attach(model_->backend());
        free_slots_.push_back(cfg.max_sequences - 1 - i);
    }
    speculative::SpecConfig sc;
    sc.max_draft = max_draft_;
    sc.gate.mode = mtp ? opts.gate_mode : speculative::GateMode::Off;
    sc.gate.window = opts.gate_window;
    sc.gate.probe_interval = opts.gate_probe_interval;
    spec_ = std::make_unique<speculative::Speculator>(*model_, sc);

    // D-013 budget. The planner counts only the requested spacing checkpoints per slot and
    // its per-copy state size excludes the Checkpoint's token prefix and carried hidden
    // (state::Checkpoint::bytes). The engine actually takes, per sequence: the spacing
    // checkpoints + a prompt-end (tail) checkpoint + a retirement checkpoint. Budget what
    // the engine stores, whichever tier the plan found (M10: the GPU-tier and host-only
    // paths must agree, or cache behaviour changes with the hardware).
    const std::uint64_t extra = (static_cast<std::uint64_t>(hp.n_embd) + info_.context_length) * 4;
    std::uint64_t ck = 0;
    if (cfg.prefix_cache) {
        const bool planned = plan_.sizes.prefix_checkpoints > 0;
        const std::uint64_t per_slot = planned ? plan_.sizes.prefix_checkpoints_per_slot
                                               : std::max<std::uint64_t>(plan_.prefix_checkpoints_requested, 1);
        ck = (per_slot + 2) * cfg.max_sequences * (plan_.sizes.gdn_state_per_copy + extra);
        if (!planned)
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
    s.threads = static_cast<std::uint32_t>(pool_ != nullptr ? pool_->size() : 0);
    s.gdn_chunk = static_cast<std::uint32_t>(model_->gdn_chunk());
    s.tuning = tuning_;
    return s;
}

// TRD §64 / §56: read the profile once at creation (read-only snapshot; no tuning here).
void CpuEngine::apply_profile() {
    threads_ = cfg_.threads > 0 ? static_cast<std::size_t>(cfg_.threads) : cpu::ThreadPool::default_threads();
    gdn_chunk_ = 64;
    if (cfg_.backend != "cpu" && cfg_.backend != "auto") {
        // GPU backends: kernel-variant selection is WS-BI-4's KernelPlan; the profile DB's
        // cpu tunables (threads, GDN chunk) do not apply.
        tuning_ = "n/a (gpu backend)";
        if (cfg_.profile_db)
            HALO_WARN("runtime", "profile_db is ignored on the '{}' backend (cpu tunables only)", cfg_.backend);
        return;
    }
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
        // Notify while holding mu_: the destructor cannot pass its own lock of mu_ (and so
        // cannot destroy cv_) before this returns, and after this block generate() never
        // touches the Engine again, so a concurrent destruction is safe.
        cv_.notify_all();
    }

    // Drain this request's queue on the caller's thread (engine.h threading guarantee).
    std::vector<std::int32_t> delivered;
    bool cancelled = false;
    std::optional<FinishReason> stopped;  // R-3: stop token / deadline seen by this thread
    std::exception_ptr cb_error;
    for (;;) {
        std::unique_lock lk(r->mu);
        r->cv.wait(lk, [&] { return !r->events.empty() || r->done; });
        if (r->events.empty()) break;  // done and drained
        TokenEvent ev = std::move(r->events.front());
        r->events.pop_front();
        lk.unlock();
        if (cancelled) continue;
        // R-3: tokens the worker produced ahead of a slow callback are not delivered once
        // the stop token or deadline fired.
        if (req.cancel.stop_requested()) {
            stopped = FinishReason::Cancelled;
        } else if (req.deadline && Clock::now() >= *req.deadline) {
            stopped = FinishReason::Length;
        }
        if (stopped) {
            cancelled = true;
            r->cancel.store(true, std::memory_order_relaxed);
            continue;
        }
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
            // The worker polls the flag at every tick (it is never idle while this request is
            // queued or active), so no notify on the engine is needed: after enqueueing,
            // generate() touches only the Request, never the Engine.
            r->cancel.store(true, std::memory_order_relaxed);
        }
    }
    GenerateResult res;
    {
        const std::lock_guard lk(r->mu);
        res = r->result;
    }
    if (cancelled) {
        res.tokens = std::move(delivered);
        res.finish = stopped.value_or(FinishReason::Cancelled);
        if (stopped == FinishReason::Length) res.deadline_expired = true;
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

std::optional<FinishReason> CpuEngine::expired(const Request& r) noexcept {
    if (r.cancel.load(std::memory_order_relaxed) || r.req.cancel.stop_requested()) return FinishReason::Cancelled;
    if (r.req.deadline && Clock::now() >= *r.req.deadline) return FinishReason::Length;
    return std::nullopt;
}

void CpuEngine::finish_unscheduled(Request& r, FinishReason why, std::string error) noexcept {
    try {
        const std::lock_guard lk(r.mu);
        r.result.finish = why;
        r.result.error = std::move(error);
        r.result.tokens.clear();
        r.result.prompt_tokens = r.req.prompt.size();
        r.result.deadline_expired = r.deadline_hit;
        r.done = true;
    } catch (...) {
        r.done = true;
    }
    r.cv.notify_all();
}

void CpuEngine::run() noexcept {
    const auto log_error = [](const char* what, const char* msg) noexcept {
        try {
            HALO_ERROR("runtime", "{}: {}", what, msg);
        } catch (...) {  // logging must not take the worker down
        }
    };
    const auto fail_all = [&](const char* msg) noexcept {
        for (Active& a : active_) {
            slots_[a.slot]->reset();
            a.fed.clear();
            retire(a, FinishReason::Error, msg);
        }
        active_.clear();
    };
    for (;;) {
        std::vector<std::shared_ptr<Request>> admit_now;
        std::vector<std::pair<std::shared_ptr<Request>, FinishReason>> dropped;
        try {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !pending_.empty() || !active_.empty(); });
            if (stop_) break;
            // R-3: queued requests whose stop token / deadline fired never get a slot.
            for (auto it = pending_.begin(); it != pending_.end();) {
                if (auto why = expired(**it)) {
                    dropped.emplace_back(*it, *why);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            while (!pending_.empty() && admit_now.size() < free_slots_.size()) {
                admit_now.push_back(pending_.front());
                pending_.pop_front();
            }
        } catch (const std::exception& e) {
            log_error("scheduler", e.what());
            continue;
        }
        for (auto& [r, why] : dropped) {
            r->deadline_hit = why == FinishReason::Length;
            finish_unscheduled(*r, why, {});
        }
        for (const auto& r : admit_now) {
            if (auto why = expired(*r)) {
                r->deadline_hit = *why == FinishReason::Length;
                finish_unscheduled(*r, *why, {});
                continue;
            }
            try {
                admit(r);
            } catch (const std::exception& e) {
                finish_unscheduled(*r, FinishReason::Error, e.what());
            } catch (...) {
                finish_unscheduled(*r, FinishReason::Error, "admission failed");
            }
        }
        // Cancel / deadline for admitted sequences (prefilling or decoding), polled every tick.
        for (auto it = active_.begin(); it != active_.end();) {
            if (auto why = expired(*it->r)) {
                it->r->deadline_hit = *why == FinishReason::Length;
                retire(*it, *why);
                it = active_.erase(it);
            } else {
                ++it;
            }
        }
        if (active_.empty()) continue;
        try {
            tick();
        } catch (const std::exception& e) {
            // Tick-wide failure before or inside the batched step (review R-1: only these fail
            // every sequence, e.g. Error(Kernel) after the forward modified state). Per-sequence
            // failures in the output phase are isolated inside tick().
            log_error("tick failed", e.what());
            fail_all(e.what());
        } catch (...) {
            log_error("tick failed", "unknown exception");
            fail_all("internal error");
        }
        for (auto it = active_.begin(); it != active_.end();) {
            if (it->finished) {
                retire(*it, it->reason, std::move(it->error));
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
    for (const auto& r : rest) finish_unscheduled(*r, FinishReason::Error, "engine shut down");
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
    if (cfg_.prefix_cache && !cache_.empty() && !prompt.empty()) {
        // L5: any cache entry whose first token differs from prompt[0] has lcp() == 0, which
        // can never beat this scan's own starting best_lcp of 0 -- so restricting the scan to
        // the bucket for prompt[0] is exact (not an approximation of full-cache LCP).
        std::size_t best_lcp = 0;
        const auto bucket_it = by_first_tok_.find(prompt[0]);
        if (bucket_it != by_first_tok_.end()) {
            for (const auto& it : bucket_it->second) best_lcp = std::max(best_lcp, lcp(prompt, it->tokens));
        }
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
    const bool hit = reused > 0;
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
    active_.push_back(std::move(a));
    free_slots_.pop_back();  // after the push: a failed admission never loses a slot
    {
        const std::lock_guard lk(mu_);
        ++stats_.active_sequences;
        if (cfg_.prefix_cache) (hit ? stats_.prefix_cache_hits : stats_.prefix_cache_misses)++;
        stats_.prefix_cache_reused_tokens += reused;
    }
}

void CpuEngine::index_cache_entry(std::list<CacheEntry>::iterator it) {
    if (!it->tokens.empty()) by_first_tok_[it->tokens[0]].push_back(it);
}

void CpuEngine::unindex_cache_entry(std::list<CacheEntry>::iterator it) {
    if (it->tokens.empty()) return;
    auto bucket_it = by_first_tok_.find(it->tokens[0]);
    if (bucket_it == by_first_tok_.end()) return;
    auto& bucket = bucket_it->second;
    bucket.erase(std::remove(bucket.begin(), bucket.end(), it), bucket.end());
    if (bucket.empty()) by_first_tok_.erase(bucket_it);
}

bool CpuEngine::evict_one_cache_entry() {
    if (cache_.empty()) return false;
    // L4: an entry with no live checkpoint can never be matched by admit() (which requires
    // ckpts_->find to succeed), so it is pure dead weight -- reclaim it before any entry a
    // future request could actually reuse. Search is bounded by cache_cap_ (a config value,
    // not attacker/request controlled).
    auto victim = std::find_if(cache_.begin(), cache_.end(), [](const CacheEntry& e) { return !e.has_checkpoint; });
    if (victim == cache_.end()) victim = std::prev(cache_.end());  // else plain LRU: the oldest entry
    ckpts_->erase_owner(victim->id);
    unindex_cache_entry(victim);
    cache_.erase(victim);
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
    // Best-effort (D-013: always-recompute is the fallback; review R-1): a failure here
    // costs a future prefix-cache hit, never the request.
    try {
        if (opts_.faults.checkpoint) opts_.faults.checkpoint();
        state::SequenceState& seq = *slots_[a.slot];
        state::Checkpoint c;
        c.tokens = a.fed;
        c.gdn = seq.gdn.snapshot();
        c.last_hidden = seq.last_hidden;
        c.owner = a.r->id;
        (void)ckpts_->insert(std::move(c));  // false: larger than the whole budget
    } catch (const std::exception& e) {
        HALO_WARN("runtime", "prefix checkpoint at {} tokens skipped: {}", a.fed.size(), e.what());
    }
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
            } else if (ngram_draft_ && a.r->fast_greedy && max_draft_ > 0) {
                a.forced = ngram_draft(a.r->req.prompt, a.generated, max_draft_);
            }
            if (!a.forced.empty()) {
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
    for (;;) {
        try {
            if (opts_.faults.before_step) opts_.faults.before_step(reqs);
            spec_->step(reqs, t);
            break;
        } catch (const Error& e) {
            if (e.code() != ErrorCode::Memory) throw;
            // Pre-tick state is intact (Memory errors are atomic). Free cached blocks first
            // (not counted as attempts: bounded by the cache size, review N-2), then fall back
            // once to plain decoding without MTP (FR-009 / M3 constraint).
            if (evict_one_cache_entry()) continue;
            if (fallback) throw;
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
        // Review R-1: the step committed every sequence consistently, so a failure from here
        // on (sampler, grammar, bookkeeping, allocation) ends only this sequence.
        try {
        if (opts_.faults.output) opts_.faults.output(a.r->req.prompt);
        if (!q.prefill.empty()) {
            a.fed.insert(a.fed.end(), q.prefill.begin(), q.prefill.end());
            a.prompt_done += q.prefill.size();
            if (a.next_ckpt < a.ckpt_at.size() && a.prompt_done == a.ckpt_at[a.next_ckpt]) {
                checkpoint(a);
                ++a.next_ckpt;
            }
            if (q.want_output) {
                // H1: a poisoned argmax row (NaN logit, backend::decode_argmax) surfaces as
                // token id -1. Caught here, inside this sequence's own try, before it is ever
                // fed forward or detokenized -- the catch below resets just this slot, so
                // every other sequence in the tick is unaffected.
                if (a.r->fast_greedy) {
                    HALO_CHECK(o.tokens.at(0) >= 0, ErrorCode::Kernel, "trunk forward: NaN logit (poisoned argmax row)");
                }
                const std::int32_t tok = a.r->fast_greedy ? o.tokens.at(0) : a.r->sampler->sample(o.logits, a.history);
                if (a.r->sampler) a.r->sampler->accept(tok);
                a.pending = tok;
                (void)emit(a, tok, events);
            }
        } else {
            info.decode_rows += 1 + o.drafts.size();
            info.max_draft = std::max(info.max_draft, o.drafts.size());
            info.drafted += o.drafts.size();
            info.drafts.insert(info.drafts.end(), o.drafts.begin(), o.drafts.end());
            info.accepted += o.accepted;
            a.drafted += o.drafts.size();
            a.accepted += o.accepted;
            if (a.r->fast_greedy) {
                // H1: same check as the prefill branch above, before any of this sequence's
                // poisoned tokens are pushed into a.fed (which would otherwise resurface as an
                // invalid prompt token on a future, differently-batched tick).
                HALO_CHECK(std::none_of(o.tokens.begin(), o.tokens.end(), [](std::int32_t t) { return t < 0; }),
                           ErrorCode::Kernel, "trunk forward: NaN logit (poisoned argmax row)");
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
        // The token list must describe exactly the KV rows (prefix-cache entries share them).
        HALO_CHECK(a.fed.size() == slots_[a.slot]->length(), ErrorCode::Kernel,
                   "runtime: sequence bookkeeping out of sync ({} tokens recorded, {} KV rows)", a.fed.size(),
                   slots_[a.slot]->length());
        const std::size_t n_events = events.size();
        if (!events.empty()) deliver(*a.r, events, false);
        a.delivered += n_events;
        } catch (const std::exception& e) {
            HALO_ERROR("runtime", "sequence failed in the output phase: {}", e.what());
            a.generated.resize(a.delivered);  // result = exactly what the caller received
            slots_[a.slot]->reset();          // not cached: its bookkeeping may be off
            a.fed.clear();
            a.finished = true;
            a.reason = FinishReason::Error;
            a.error = e.what();
        }
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

void CpuEngine::cache_retired(Active& a) {
    state::SequenceState& seq = *slots_[a.slot];
    if (opts_.faults.retire_cache) opts_.faults.retire_cache();
    try {
        if (seq.mtp_kv && seq.mtp_queue_size() > 0) {
            state::SequenceState* one[] = {&seq};
            spec_->flush_mtp(one);
        }
    } catch (const std::exception& e) {  // bad_alloc included (review R-2)
        HALO_WARN("runtime", "prefix cache: MTP catch-up for a retired sequence failed: {}", e.what());
        seq.mtp_kv.reset();
    }
    // Everything that can throw happens before the KV blocks move into the cache.
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
    e.has_checkpoint = ckpts_->insert(std::move(c));  // L4: false leaves a KV-only entry, never matched
    cache_.push_front(std::move(e));
    index_cache_entry(cache_.begin());
    while (cache_.size() > cache_cap_) evict_one_cache_entry();
}

void CpuEngine::retire(Active& a, FinishReason why, std::string error) noexcept {
    state::SequenceState& seq = *slots_[a.slot];
    // Keep the sequence for prefix reuse (not after an error: its state may be unspecified).
    bool cached = false;
    if (cfg_.prefix_cache && why != FinishReason::Error && !a.fed.empty() && cache_cap_ > 0) {
        try {
            cache_retired(a);
            cached = true;
        } catch (const std::exception& e) {
            try {
                HALO_WARN("runtime", "prefix cache: retired sequence not cached: {}", e.what());
            } catch (...) {
            }
        } catch (...) {
        }
    }
    if (!cached) ckpts_->erase_owner(a.r->id);
    seq.reset();
    free_slots_.push_back(a.slot);  // capacity reserved at construction: no allocation
    try {
        // Before the done event: a caller that returns from generate() never sees its own
        // sequence still counted as active.
        const std::lock_guard lk(mu_);
        --stats_.active_sequences;
    } catch (...) {
    }
    Request& r = *a.r;
    try {
        const std::lock_guard lk(r.mu);
        GenerateResult& res = r.result;
        res.tokens = std::move(a.generated);
        res.finish = why;
        res.error = std::move(error);
        res.draft_tokens = a.drafted;
        res.accepted_draft_tokens = a.accepted;
        res.deadline_expired = r.deadline_hit;
        if (a.t_first) {
            res.ttft_ms = ms_between(r.t_submit, *a.t_first);
            const double dt = ms_between(*a.t_first, a.t_last) / 1000.0;
            res.decode_tps = res.tokens.size() > 1 && dt > 0 ? static_cast<double>(res.tokens.size() - 1) / dt : 0.0;
        }
        r.done = true;
    } catch (...) {
        r.done = true;  // the caller must be released whatever happens
    }
    r.cv.notify_all();
}

}  // namespace

std::unique_ptr<Engine> create_cpu_engine(const EngineConfig& cfg, const CpuEngineOptions& opts) {
    return std::make_unique<CpuEngine>(cfg, opts);
}

std::unique_ptr<Engine> create_engine(const EngineConfig& cfg) { return create_cpu_engine(cfg, {}); }

}  // namespace halo::runtime
