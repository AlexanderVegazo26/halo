#include "halo/model/model.h"

#include <charconv>
#include <cmath>
#include <format>
#include <initializer_list>
#include <set>
#include <string>

#include "halo/core/error.h"
#include "halo/core/log.h"

namespace halo::model {
namespace {

constexpr std::string_view kArch = "qwen35";
constexpr std::uint32_t kMaxDim = 1u << 24;    // sanity cap for any single hparam dimension
constexpr std::uint32_t kMaxBlocks = 4096;

std::string qkey(std::string_view k) { return std::format("{}.{}", kArch, k); }

// ---- architecture / vision gate -----------------------------------------------------------

bool is_vision_tensor(std::string_view n) noexcept {
    return n.starts_with("v.") || n.starts_with("mm.") || n.starts_with("vision") || n.starts_with("clip.") ||
           n.starts_with("mmproj");
}

void check_architecture(const GgufFile& f) {
    const std::string* arch = f.get_string("general.architecture");
    if (arch == nullptr) {
        throw_error(ErrorCode::Model, "{}: missing general.architecture", f.source());
    }
    if (*arch == "clip" || arch->find("vl") != std::string::npos || arch->find("vision") != std::string::npos) {
        throw_error(ErrorCode::Unsupported,
                    "{}: architecture '{}' is a vision/multimodal file; vision input is not supported in HALO v1",
                    f.source(), *arch);
    }
    if (*arch != kArch) {
        throw_error(ErrorCode::Unsupported, "{}: architecture '{}' is not supported (HALO v1 supports '{}' only)",
                    f.source(), *arch, kArch);
    }
    for (const auto& [k, v] : f.kvs()) {
        if (k.starts_with("clip.")) {
            throw_error(ErrorCode::Unsupported, "{}: vision encoder metadata ('{}') is not supported in HALO v1",
                        f.source(), k);
        }
    }
    for (const auto& t : f.tensors()) {
        if (is_vision_tensor(t.name)) {
            throw_error(ErrorCode::Unsupported, "{}: vision tower tensor '{}' found; vision input is not supported in HALO v1",
                        f.source(), t.name);
        }
    }
}

// ---- hyperparameters ----------------------------------------------------------------------

class KeyReader {
public:
    KeyReader(const GgufFile& f, std::vector<std::string>& warnings) : f_(f), warnings_(warnings) {}

    std::uint32_t req_u32(std::string_view k) const {
        const auto v = f_.get_u32(qkey(k));
        if (!v) throw_error(ErrorCode::Model, "{}: missing required key {}", f_.source(), qkey(k));
        return *v;
    }
    std::uint32_t opt_u32(std::string_view k, std::uint32_t def, bool warn) const {
        const auto v = f_.get_u32(qkey(k));
        if (!v && warn) warnings_.push_back(std::format("{} absent; using default {}", qkey(k), def));
        return v.value_or(def);
    }
    float req_f32(std::string_view k) const {
        const auto v = f_.get_float(qkey(k));
        if (!v) throw_error(ErrorCode::Model, "{}: missing required key {}", f_.source(), qkey(k));
        return static_cast<float>(*v);
    }
    float opt_f32(std::string_view k, float def) const {
        const auto v = f_.get_float(qkey(k));
        if (!v) warnings_.push_back(std::format("{} absent; using default {}", qkey(k), def));
        return v ? static_cast<float>(*v) : def;
    }
    const GgufFile& file() const { return f_; }

private:
    const GgufFile& f_;
    std::vector<std::string>& warnings_;
};

void check_dim(const GgufFile& f, std::string_view what, std::uint64_t v, std::uint64_t lo = 1,
               std::uint64_t hi = kMaxDim) {
    if (v < lo || v > hi) {
        throw_error(ErrorCode::Model, "{}: {} = {} out of range [{}, {}]", f.source(), what, v, lo, hi);
    }
}

std::uint32_t mul32(const GgufFile& f, std::string_view what, std::uint64_t a, std::uint64_t b) {
    const std::uint64_t p = a * b;  // both <= 2^24, cannot overflow uint64
    if (p > kMaxDim * 16ull) {
        throw_error(ErrorCode::Model, "{}: derived {} = {} x {} is implausibly large", f.source(), what, a, b);
    }
    return static_cast<std::uint32_t>(p);
}

// Integer-or-bool array element as int64.
std::int64_t array_int(const GgufArray& a, std::size_t i) {
    switch (a.elem_type) {
        case GgufType::Bool: return a.bools[i];
        case GgufType::U8:
        case GgufType::U16:
        case GgufType::U32:
        case GgufType::U64: return static_cast<std::int64_t>(a.uints[i] > 0x7FFFFFFFu ? 0x7FFFFFFFu : a.uints[i]);
        case GgufType::I8:
        case GgufType::I16:
        case GgufType::I32:
        case GgufType::I64: return a.ints[i];
        default: return 0;
    }
}

bool is_int_or_bool(GgufType t) noexcept {
    return t == GgufType::Bool || t == GgufType::U8 || t == GgufType::U16 || t == GgufType::U32 ||
           t == GgufType::U64 || t == GgufType::I8 || t == GgufType::I16 || t == GgufType::I32 ||
           t == GgufType::I64;
}

Qwen35HParams read_hparams(const GgufFile& f, std::vector<std::string>& warnings) {
    KeyReader r(f, warnings);
    Qwen35HParams h;
    h.context_length = r.req_u32("context_length");
    h.n_embd = r.req_u32("embedding_length");
    h.block_count = r.req_u32("block_count");
    h.n_ff = r.req_u32("feed_forward_length");
    h.n_head = r.req_u32("attention.head_count");
    check_dim(f, "embedding_length", h.n_embd);
    check_dim(f, "block_count", h.block_count, 1, kMaxBlocks);
    check_dim(f, "feed_forward_length", h.n_ff);
    check_dim(f, "attention.head_count", h.n_head, 1, 1u << 16);
    check_dim(f, "context_length", h.context_length, 1, 1u << 31);
    h.n_head_kv = r.opt_u32("attention.head_count_kv", h.n_head, true);
    check_dim(f, "attention.head_count_kv", h.n_head_kv, 1, h.n_head);
    if (h.n_head % h.n_head_kv != 0) {
        throw_error(ErrorCode::Model, "{}: head_count {} is not a multiple of head_count_kv {}", f.source(), h.n_head,
                    h.n_head_kv);
    }
    h.key_length = r.opt_u32("attention.key_length", h.n_embd / h.n_head, true);
    h.value_length = r.opt_u32("attention.value_length", h.n_embd / h.n_head, true);
    check_dim(f, "attention.key_length", h.key_length, 1, 1u << 16);
    check_dim(f, "attention.value_length", h.value_length, 1, 1u << 16);
    h.rms_eps = r.req_f32("attention.layer_norm_rms_epsilon");
    if (!(std::isfinite(h.rms_eps) && h.rms_eps > 0.0f && h.rms_eps < 1.0f)) {
        throw_error(ErrorCode::Model, "{}: invalid layer_norm_rms_epsilon {}", f.source(), h.rms_eps);
    }
    h.rope_dim = r.opt_u32("rope.dimension_count", h.key_length, true);
    if (h.rope_dim > h.key_length || h.rope_dim % 2 != 0) {
        throw_error(ErrorCode::Model, "{}: rope.dimension_count {} must be even and <= key_length {}", f.source(),
                    h.rope_dim, h.key_length);
    }
    h.rope_freq_base = r.opt_f32("rope.freq_base", 10000.0f);
    if (!(std::isfinite(h.rope_freq_base) && h.rope_freq_base > 0.0f)) {
        throw_error(ErrorCode::Model, "{}: invalid rope.freq_base {}", f.source(), h.rope_freq_base);
    }
    // rope.dimension_sections: required by llama.cpp (get_key_or_arr, n = 4); array of 4 or scalar.
    {
        const std::string k = qkey("rope.dimension_sections");
        const GgufValue* v = f.find(k);
        if (v == nullptr) throw_error(ErrorCode::Model, "{}: missing required key {}", f.source(), k);
        if (v->type == GgufType::Array) {
            if (!is_int_or_bool(v->arr.elem_type) || v->arr.elem_type == GgufType::Bool || v->arr.size() != 4) {
                throw_error(ErrorCode::Model, "{}: {} must be an integer array of length 4", f.source(), k);
            }
            for (std::size_t i = 0; i < 4; ++i) {
                const std::int64_t s = array_int(v->arr, i);
                if (s < 0 || s > 1 << 16) throw_error(ErrorCode::Model, "{}: {}[{}] = {} out of range", f.source(), k, i, s);
                h.rope_sections[i] = static_cast<std::int32_t>(s);
            }
        } else {
            const auto s = f.get_u32(k);
            if (!s || *s > 1u << 16) throw_error(ErrorCode::Model, "{}: invalid {}", f.source(), k);
            h.rope_sections.fill(static_cast<std::int32_t>(*s));
        }
    }
    h.ssm_conv_kernel = r.req_u32("ssm.conv_kernel");
    h.ssm_inner_size = r.req_u32("ssm.inner_size");
    h.ssm_state_size = r.req_u32("ssm.state_size");
    h.ssm_time_step_rank = r.req_u32("ssm.time_step_rank");
    h.ssm_group_count = r.req_u32("ssm.group_count");
    check_dim(f, "ssm.conv_kernel", h.ssm_conv_kernel, 1, 64);
    check_dim(f, "ssm.inner_size", h.ssm_inner_size);
    check_dim(f, "ssm.state_size", h.ssm_state_size, 1, 1u << 16);
    check_dim(f, "ssm.time_step_rank", h.ssm_time_step_rank, 1, 1u << 16);
    check_dim(f, "ssm.group_count", h.ssm_group_count, 1, 1u << 16);
    if (h.ssm_time_step_rank % h.ssm_group_count != 0) {
        throw_error(ErrorCode::Model, "{}: ssm.time_step_rank {} (value heads) is not a multiple of ssm.group_count {} "
                    "(key heads)", f.source(), h.ssm_time_step_rank, h.ssm_group_count);
    }
    h.full_attention_interval = r.opt_u32("full_attention_interval", 4, false);
    check_dim(f, "full_attention_interval", h.full_attention_interval, 1, kMaxBlocks);
    h.nextn_predict_layers = r.opt_u32("nextn_predict_layers", 0, false);
    if (h.nextn_predict_layers >= h.block_count) {
        throw_error(ErrorCode::Model, "{}: nextn_predict_layers {} leaves no trunk layers (block_count {})",
                    f.source(), h.nextn_predict_layers, h.block_count);
    }
    h.n_layer = h.block_count - h.nextn_predict_layers;

    // Layer types: attention.recurrent_layers if present (length must be block_count, as in
    // llama.cpp get_key_or_arr), else derived exactly like llama.cpp qwen35 load_arch_hparams:
    //   recurrent(i) = i < n_layer && (i + 1) % full_attention_interval != 0
    h.layer_kind.assign(h.block_count, LayerKind::FullAttention);
    const std::string rl_key = qkey("attention.recurrent_layers");
    if (const GgufValue* v = f.find(rl_key)) {
        if (v->type != GgufType::Array || !is_int_or_bool(v->arr.elem_type)) {
            throw_error(ErrorCode::Model, "{}: {} must be a bool/int array", f.source(), rl_key);
        }
        if (v->arr.size() != h.block_count) {
            throw_error(ErrorCode::Model, "{}: {} has length {}, expected block_count {}", f.source(), rl_key,
                        v->arr.size(), h.block_count);
        }
        h.recurrent_layers_from_file = true;
        for (std::uint32_t i = 0; i < h.block_count; ++i) {
            const bool recr = array_int(v->arr, i) != 0;
            if (recr && i >= h.n_layer) {
                throw_error(ErrorCode::Model, "{}: {} marks MTP block {} recurrent; MTP blocks are full attention",
                            f.source(), rl_key, i);
            }
            h.layer_kind[i] = recr ? LayerKind::GatedDeltaNet : LayerKind::FullAttention;
            const bool derived = (i < h.n_layer) && ((i + 1) % h.full_attention_interval != 0);
            if (derived != recr && i < h.n_layer) {
                warnings.push_back(std::format("layer {} type from recurrent_layers differs from full_attention_interval {}",
                                               i, h.full_attention_interval));
            }
        }
    } else {
        for (std::uint32_t i = 0; i < h.block_count; ++i) {
            const bool recr = (i < h.n_layer) && ((i + 1) % h.full_attention_interval != 0);
            h.layer_kind[i] = recr ? LayerKind::GatedDeltaNet : LayerKind::FullAttention;
        }
    }
    for (std::uint32_t i = 0; i < h.n_layer; ++i) {
        (h.layer_kind[i] == LayerKind::GatedDeltaNet ? h.n_gdn_layers : h.n_attn_layers) += 1;
    }

    h.gdn_n_k_heads = h.ssm_group_count;
    h.gdn_n_v_heads = h.ssm_time_step_rank;
    h.gdn_head_k_dim = h.ssm_state_size;
    h.gdn_head_v_dim = h.ssm_state_size;
    h.gdn_v_per_k = h.gdn_n_v_heads / h.gdn_n_k_heads;
    h.gdn_key_dim = mul32(f, "gdn key dim", h.gdn_n_k_heads, h.gdn_head_k_dim);
    h.gdn_value_dim = mul32(f, "gdn value dim", h.gdn_n_v_heads, h.gdn_head_v_dim);
    if (h.gdn_value_dim != h.ssm_inner_size) {
        throw_error(ErrorCode::Model, "{}: ssm.inner_size {} != time_step_rank {} x state_size {}", f.source(),
                    h.ssm_inner_size, h.gdn_n_v_heads, h.gdn_head_v_dim);
    }
    h.gdn_conv_channels = 2 * h.gdn_key_dim + h.gdn_value_dim;
    h.attn_q_dim = mul32(f, "attention q dim", h.n_head, h.key_length);
    h.attn_kv_dim = mul32(f, "attention k dim", h.n_head_kv, h.key_length);
    h.attn_v_dim = mul32(f, "attention v dim", h.n_head_kv, h.value_length);
    h.gqa_ratio = h.n_head / h.n_head_kv;

    if (const GgufTensorInfo* te = f.find_tensor("token_embd.weight")) {
        h.n_vocab = te->ne[1];
    }
    return h;
}

// ---- tensor expectations ------------------------------------------------------------------

enum class Policy { Matrix, Vector };  // Vector: small fp tensors (norms, biases, conv, ssm_a)

class TensorBinder {
public:
    TensorBinder(const GgufFile& f, std::set<std::string>& consumed) : f_(f), consumed_(consumed) {}

    WeightRef bind(const std::string& name, std::initializer_list<std::int64_t> shape, Policy p, bool required) {
        const GgufTensorInfo* t = f_.find_tensor(name);
        if (t == nullptr) {
            if (required) throw_error(ErrorCode::Model, "{}: missing tensor '{}'", f_.source(), name);
            return {};
        }
        consumed_.insert(name);
        std::array<std::int64_t, 4> want{1, 1, 1, 1};
        std::size_t i = 0;
        for (std::int64_t d : shape) want[i++] = d;
        if (t->ne != want) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' has shape {} but hparams require {}", f_.source(), name,
                        shape_str(t->ne), shape_str(want));
        }
        if (p == Policy::Vector) {
            if (t->type != DType::F32 && t->type != DType::F16 && t->type != DType::BF16) {
                throw_error(ErrorCode::Model, "{}: tensor '{}' has dtype {}; expected F32/F16/BF16 for this tensor",
                            f_.source(), name, traits(t->type).name);
            }
        } else if (!dequant_supported(t->type)) {
            throw_error(ErrorCode::Unsupported, "{}: tensor '{}' has dtype {}, which HALO cannot dequantize",
                        f_.source(), name, traits(t->type).name);
        }
        return {&f_, t};
    }

private:
    static std::string shape_str(const std::array<std::int64_t, 4>& ne) {
        std::string s = "[";
        std::size_t n = 4;
        while (n > 1 && ne[n - 1] == 1) --n;
        for (std::size_t i = 0; i < n; ++i) s += std::format("{}{}", i ? ", " : "", ne[i]);
        return s + "]";
    }

    const GgufFile& f_;
    std::set<std::string>& consumed_;
};

std::string blk(std::uint32_t il, std::string_view suffix) { return std::format("blk.{}.{}", il, suffix); }

void bind_ffn_and_norms(TensorBinder& b, const Qwen35HParams& h, LayerWeights& L) {
    const std::int64_t E = h.n_embd;
    L.attn_norm = b.bind(blk(L.index, "attn_norm.weight"), {E}, Policy::Vector, true);
    L.post_attention_norm = b.bind(blk(L.index, "post_attention_norm.weight"), {E}, Policy::Vector, true);
    L.ffn_gate = b.bind(blk(L.index, "ffn_gate.weight"), {E, h.n_ff}, Policy::Matrix, true);
    L.ffn_up = b.bind(blk(L.index, "ffn_up.weight"), {E, h.n_ff}, Policy::Matrix, true);
    L.ffn_down = b.bind(blk(L.index, "ffn_down.weight"), {h.n_ff, E}, Policy::Matrix, true);
}

void bind_attention(TensorBinder& b, const Qwen35HParams& h, LayerWeights& L) {
    const std::int64_t E = h.n_embd;
    const std::uint32_t il = L.index;
    L.attn.q = b.bind(blk(il, "attn_q.weight"), {E, 2 * static_cast<std::int64_t>(h.attn_q_dim)}, Policy::Matrix, true);
    L.attn.k = b.bind(blk(il, "attn_k.weight"), {E, h.attn_kv_dim}, Policy::Matrix, true);
    L.attn.v = b.bind(blk(il, "attn_v.weight"), {E, h.attn_v_dim}, Policy::Matrix, true);
    L.attn.output = b.bind(blk(il, "attn_output.weight"),
                           {static_cast<std::int64_t>(h.n_head) * h.value_length, E}, Policy::Matrix, true);
    L.attn.q_norm = b.bind(blk(il, "attn_q_norm.weight"), {h.key_length}, Policy::Vector, true);
    L.attn.k_norm = b.bind(blk(il, "attn_k_norm.weight"), {h.key_length}, Policy::Vector, true);
}

void bind_gdn(TensorBinder& b, const Qwen35HParams& h, LayerWeights& L) {
    const std::int64_t E = h.n_embd;
    const std::uint32_t il = L.index;
    L.gdn.qkv = b.bind(blk(il, "attn_qkv.weight"), {E, h.gdn_conv_channels}, Policy::Matrix, true);
    L.gdn.gate = b.bind(blk(il, "attn_gate.weight"), {E, h.gdn_value_dim}, Policy::Matrix, true);
    L.gdn.conv1d = b.bind(blk(il, "ssm_conv1d.weight"), {h.ssm_conv_kernel, h.gdn_conv_channels}, Policy::Vector, true);
    L.gdn.dt_bias = b.bind(blk(il, "ssm_dt.bias"), {h.gdn_n_v_heads}, Policy::Vector, true);
    L.gdn.a = b.bind(blk(il, "ssm_a"), {h.gdn_n_v_heads}, Policy::Vector, true);
    L.gdn.beta = b.bind(blk(il, "ssm_beta.weight"), {E, h.gdn_n_v_heads}, Policy::Matrix, true);
    L.gdn.alpha = b.bind(blk(il, "ssm_alpha.weight"), {E, h.gdn_n_v_heads}, Policy::Matrix, true);
    L.gdn.norm = b.bind(blk(il, "ssm_norm.weight"), {h.gdn_head_v_dim}, Policy::Vector, true);
    L.gdn.out = b.bind(blk(il, "ssm_out.weight"), {h.gdn_value_dim, E}, Policy::Matrix, true);
}

struct MtpOwn {  // optional in-block tensors of an MTP block
    WeightRef embed_tokens, shared_head_head, shared_head_norm;
};

MtpWeights bind_mtp_block(TensorBinder& b, const Qwen35HParams& h, std::int64_t n_vocab, MtpOwn& own) {
    MtpWeights m;
    m.block.index = h.n_layer;
    m.block.kind = LayerKind::FullAttention;
    bind_ffn_and_norms(b, h, m.block);
    bind_attention(b, h, m.block);
    const std::int64_t E = h.n_embd;
    const std::uint32_t il = h.n_layer;
    m.eh_proj = b.bind(blk(il, "nextn.eh_proj.weight"), {2 * E, E}, Policy::Matrix, true);
    m.enorm = b.bind(blk(il, "nextn.enorm.weight"), {E}, Policy::Vector, true);
    m.hnorm = b.bind(blk(il, "nextn.hnorm.weight"), {E}, Policy::Vector, true);
    own.embed_tokens = b.bind(blk(il, "nextn.embed_tokens.weight"), {E, n_vocab}, Policy::Matrix, false);
    own.shared_head_head = b.bind(blk(il, "nextn.shared_head_head.weight"), {E, n_vocab}, Policy::Matrix, false);
    own.shared_head_norm = b.bind(blk(il, "nextn.shared_head_norm.weight"), {E}, Policy::Vector, false);
    return m;
}

// Parses "blk.<N>." prefixes; returns nullopt for non-block tensors.
std::optional<std::uint64_t> block_index(std::string_view name) {
    if (!name.starts_with("blk.")) return std::nullopt;
    name.remove_prefix(4);
    std::uint64_t v = 0;
    const auto [p, ec] = std::from_chars(name.data(), name.data() + name.size(), v);
    if (ec != std::errc{} || p == name.data() || p == name.data() + name.size() || *p != '.') return std::nullopt;
    return v;
}

void check_leftovers(const GgufFile& f, const Qwen35HParams& h, const std::set<std::string>& consumed,
                     std::vector<std::string>& warnings) {
    std::size_t ignored = 0;
    for (const auto& t : f.tensors()) {
        if (consumed.contains(t.name)) continue;
        if (const auto il = block_index(t.name); il && *il >= h.block_count) {
            throw_error(ErrorCode::Model, "{}: tensor '{}' refers to block {} but block_count is {}", f.source(),
                        t.name, *il, h.block_count);
        }
        if (ignored++ < 8) warnings.push_back(std::format("{}: unrecognized tensor '{}' ignored", f.source(), t.name));
    }
    if (ignored > 8) warnings.push_back(std::format("{}: {} more unrecognized tensors ignored", f.source(), ignored - 8));
}

void check_tokenizer(const GgufFile& f, std::vector<std::string>& warnings, std::int64_t n_vocab) {
    const GgufArray* tokens = f.get_array("tokenizer.ggml.tokens");
    if (tokens != nullptr && tokens->elem_type != GgufType::String) {
        throw_error(ErrorCode::Model, "{}: tokenizer.ggml.tokens must be a string array", f.source());
    }
    if (const GgufArray* merges = f.get_array("tokenizer.ggml.merges");
        merges != nullptr && merges->elem_type != GgufType::String) {
        throw_error(ErrorCode::Model, "{}: tokenizer.ggml.merges must be a string array", f.source());
    }
    if (const GgufArray* tt = f.get_array("tokenizer.ggml.token_type"); tt != nullptr) {
        if (tt->elem_type != GgufType::I32 && tt->elem_type != GgufType::I64 && tt->elem_type != GgufType::I8 &&
            tt->elem_type != GgufType::I16) {
            throw_error(ErrorCode::Model, "{}: tokenizer.ggml.token_type must be a signed integer array", f.source());
        }
        if (tokens != nullptr && tt->size() != tokens->size()) {
            throw_error(ErrorCode::Model, "{}: tokenizer.ggml.token_type has {} entries for {} tokens", f.source(),
                        tt->size(), tokens->size());
        }
    }
    (void)f.get_string("tokenizer.ggml.model");
    (void)f.get_string("tokenizer.ggml.pre");
    (void)f.get_string("tokenizer.chat_template");
    (void)f.get_bool("tokenizer.ggml.add_bos_token");
    (void)f.get_bool("tokenizer.ggml.add_eos_token");
    for (const char* k : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                          "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id"}) {
        const auto id = f.get_u32(k);
        if (id && tokens != nullptr && *id >= tokens->size()) {
            throw_error(ErrorCode::Model, "{}: {} = {} is outside the {}-token vocabulary", f.source(), k, *id,
                        tokens->size());
        }
    }
    if (tokens != nullptr && n_vocab > 0 && static_cast<std::int64_t>(tokens->size()) != n_vocab) {
        warnings.push_back(std::format("tokenizer has {} tokens but token_embd has {} rows", tokens->size(), n_vocab));
    }
}

void require_same(const GgufFile& mtp, std::string_view what, std::uint64_t trunk, std::uint64_t other) {
    if (trunk != other) {
        throw_error(ErrorCode::Model, "{}: MTP file {} = {} does not match the trunk model ({})", mtp.source(), what,
                    other, trunk);
    }
}

}  // namespace

// ---- small enum helpers ---------------------------------------------------------------------

std::string_view to_string(LayerKind k) noexcept {
    return k == LayerKind::FullAttention ? "attention" : "gated_delta_net";
}

std::string_view to_string(MtpSource s) noexcept {
    switch (s) {
        case MtpSource::None: return "none";
        case MtpSource::Embedded: return "embedded";
        case MtpSource::SeparateFile: return "separate_file";
    }
    return "unknown";
}

std::string_view to_string(MtpTensorOrigin o) noexcept {
    switch (o) {
        case MtpTensorOrigin::NextnBlock: return "nextn_block";
        case MtpTensorOrigin::Trunk: return "trunk";
        case MtpTensorOrigin::MtpFile: return "mtp_file";
    }
    return "unknown";
}

const GgufTensorInfo& WeightRef::info() const {
    HALO_CHECK(info_ != nullptr, ErrorCode::Api, "access to an absent optional tensor");
    return *info_;
}

std::int64_t WeightRef::ne(int i) const {
    HALO_CHECK(i >= 0 && i < 4, ErrorCode::Api, "dimension index {} out of range [0, 4)", i);
    return info().ne[static_cast<std::size_t>(i)];
}

bool Qwen35HParams::is_attention(std::uint32_t il) const {
    HALO_CHECK(il < layer_kind.size(), ErrorCode::Api, "block {} out of range (block_count {})", il,
               layer_kind.size());
    return layer_kind[il] == LayerKind::FullAttention;
}

const GgufFile& WeightRef::file() const {
    HALO_CHECK(file_ != nullptr, ErrorCode::Api, "access to an absent optional tensor");
    return *file_;
}

// ---- NormalizedModel ----------------------------------------------------------------------

NormalizedModel NormalizedModel::load(const std::filesystem::path& path, GgufMode mode) {
    return from_gguf(GgufFile::open(path, mode));
}

NormalizedModel NormalizedModel::from_gguf(GgufFile file) {
    NormalizedModel m;
    m.file_ = std::make_unique<GgufFile>(std::move(file));
    const GgufFile& f = *m.file_;
    check_architecture(f);
    m.hp_ = read_hparams(f, m.warnings_);
    const Qwen35HParams& h = m.hp_;

    if (h.nextn_predict_layers > 0 && f.find_tensor("blk.0.attn_norm.weight") == nullptr &&
        f.find_tensor(blk(h.n_layer, "nextn.eh_proj.weight")) != nullptr) {
        throw_error(ErrorCode::Model,
                    "{}: this is an MTP-only GGUF (only blk.{}.* present); load the trunk model and attach this file "
                    "with attach_mtp()", f.source(), h.n_layer);
    }

    std::set<std::string> consumed;
    TensorBinder b(f, consumed);
    const GgufTensorInfo* te = f.find_tensor("token_embd.weight");
    if (te == nullptr) throw_error(ErrorCode::Model, "{}: missing tensor 'token_embd.weight'", f.source());
    if (te->ne[1] < 1 || te->ne[1] > (1 << 24)) {
        throw_error(ErrorCode::Model, "{}: token_embd.weight vocabulary size {} out of range", f.source(), te->ne[1]);
    }
    const std::int64_t V = h.n_vocab;
    const std::int64_t E = h.n_embd;
    m.token_embd_ = b.bind("token_embd.weight", {E, V}, Policy::Matrix, true);
    m.output_norm_ = b.bind("output_norm.weight", {E}, Policy::Vector, true);
    m.output_ = b.bind("output.weight", {E, V}, Policy::Matrix, false);
    if (!m.output_.present()) {
        m.output_ = m.token_embd_;
        m.lm_head_tied_ = true;
        m.warnings_.push_back("output.weight absent; LM head tied to token_embd.weight");
    }

    m.layers_.resize(h.n_layer);
    for (std::uint32_t il = 0; il < h.n_layer; ++il) {
        LayerWeights& L = m.layers_[il];
        L.index = il;
        L.kind = h.layer_kind[il];
        bind_ffn_and_norms(b, h, L);
        if (L.kind == LayerKind::FullAttention) {
            bind_attention(b, h, L);
        } else {
            bind_gdn(b, h, L);
        }
    }

    if (h.nextn_predict_layers > 0) {
        if (h.nextn_predict_layers != 1) {
            throw_error(ErrorCode::Unsupported, "{}: nextn_predict_layers = {}; only one MTP block is supported",
                        f.source(), h.nextn_predict_layers);
        }
        MtpOwn own;
        MtpWeights mtp = bind_mtp_block(b, h, V, own);
        mtp.embedding = own.embed_tokens.present() ? own.embed_tokens : m.token_embd_;
        mtp.embedding_origin = own.embed_tokens.present() ? MtpTensorOrigin::NextnBlock : MtpTensorOrigin::Trunk;
        mtp.lm_head = own.shared_head_head.present() ? own.shared_head_head : m.output_;
        mtp.lm_head_origin = own.shared_head_head.present() ? MtpTensorOrigin::NextnBlock : MtpTensorOrigin::Trunk;
        mtp.head_norm = own.shared_head_norm.present() ? own.shared_head_norm : m.output_norm_;
        mtp.head_norm_origin = own.shared_head_norm.present() ? MtpTensorOrigin::NextnBlock : MtpTensorOrigin::Trunk;
        m.mtp_ = std::move(mtp);
        m.mtp_source_ = MtpSource::Embedded;
    }

    check_leftovers(f, h, consumed, m.warnings_);
    check_tokenizer(f, m.warnings_, V);
    HALO_DEBUG("model", "loaded {}: {} trunk layers ({} GDN, {} attention), MTP {}", f.source(), h.n_layer,
               h.n_gdn_layers, h.n_attn_layers, to_string(m.mtp_source_));
    return m;
}

void NormalizedModel::attach_mtp(const std::filesystem::path& path, GgufMode mode, bool use_mtp_file_head) {
    attach_mtp(GgufFile::open(path, mode), use_mtp_file_head);
}

void NormalizedModel::attach_mtp(GgufFile file, bool use_mtp_file_head) {
    if (mtp_source_ != MtpSource::None) {
        throw_error(ErrorCode::Config, "{}: cannot attach MTP file {}: model already has an {} MTP block",
                    file_->source(), file.source(), to_string(mtp_source_));
    }
    auto owned = std::make_unique<GgufFile>(std::move(file));
    const GgufFile& f = *owned;
    check_architecture(f);
    std::vector<std::string> warnings;
    const Qwen35HParams mh = read_hparams(f, warnings);
    const Qwen35HParams& h = hp_;

    if (mh.nextn_predict_layers != 1) {
        throw_error(ErrorCode::Model, "{}: MTP file must have nextn_predict_layers = 1 (has {})", f.source(),
                    mh.nextn_predict_layers);
    }
    require_same(f, "trunk layer count (block_count - nextn_predict_layers)", h.n_layer, mh.n_layer);
    require_same(f, "embedding_length", h.n_embd, mh.n_embd);
    require_same(f, "feed_forward_length", h.n_ff, mh.n_ff);
    require_same(f, "attention.head_count", h.n_head, mh.n_head);
    require_same(f, "attention.head_count_kv", h.n_head_kv, mh.n_head_kv);
    require_same(f, "attention.key_length", h.key_length, mh.key_length);
    require_same(f, "attention.value_length", h.value_length, mh.value_length);
    require_same(f, "rope.dimension_count", h.rope_dim, mh.rope_dim);
    if (h.rope_freq_base != mh.rope_freq_base || h.rms_eps != mh.rms_eps || h.rope_sections != mh.rope_sections) {
        throw_error(ErrorCode::Model, "{}: MTP file rope.freq_base / rms epsilon / rope sections differ from the trunk",
                    f.source());
    }
    if (h.context_length != mh.context_length) {
        warnings.push_back(std::format("MTP file context_length {} differs from trunk {}", mh.context_length,
                                       h.context_length));
    }
    for (const auto& t : f.tensors()) {
        if (const auto il = block_index(t.name); il && *il < h.n_layer) {
            throw_error(ErrorCode::Model, "{}: contains trunk tensor '{}'; not an MTP-only file", f.source(), t.name);
        }
    }

    std::set<std::string> consumed;
    TensorBinder b(f, consumed);
    const std::int64_t V = hp_.n_vocab;
    const std::int64_t E = hp_.n_embd;
    const WeightRef own_embd = b.bind("token_embd.weight", {E, V}, Policy::Matrix, false);
    const WeightRef own_out = b.bind("output.weight", {E, V}, Policy::Matrix, false);
    const WeightRef own_norm = b.bind("output_norm.weight", {E}, Policy::Vector, false);
    MtpOwn in_block;
    MtpWeights mtp = bind_mtp_block(b, h, V, in_block);

    auto pick = [&](const WeightRef& nextn, const WeightRef& file_copy, const WeightRef& trunk, WeightRef& out,
                    MtpTensorOrigin& origin) {
        if (nextn.present()) {
            out = nextn;
            origin = MtpTensorOrigin::NextnBlock;
        } else if (use_mtp_file_head && file_copy.present()) {
            out = file_copy;
            origin = MtpTensorOrigin::MtpFile;
        } else {
            out = trunk;
            origin = MtpTensorOrigin::Trunk;
        }
    };
    pick(in_block.embed_tokens, own_embd, token_embd_, mtp.embedding, mtp.embedding_origin);
    pick(in_block.shared_head_head, own_out, output_, mtp.lm_head, mtp.lm_head_origin);
    pick(in_block.shared_head_norm, own_norm, output_norm_, mtp.head_norm, mtp.head_norm_origin);

    check_leftovers(f, mh, consumed, warnings);

    // Commit (strong guarantee: nothing above modified *this).
    mtp_file_ = std::move(owned);
    mtp_ = std::move(mtp);
    mtp_source_ = MtpSource::SeparateFile;
    for (auto& w : warnings) warnings_.push_back(std::move(w));
    HALO_DEBUG("model", "attached MTP {} (embedding: {}, lm_head: {})", mtp_file_->source(),
               to_string(mtp_->embedding_origin), to_string(mtp_->lm_head_origin));
}

const LayerWeights& NormalizedModel::layer(std::uint32_t il) const {
    HALO_CHECK(il < layers_.size(), ErrorCode::Api, "layer {} out of range (n_layer {})", il, layers_.size());
    return layers_[il];
}

TokenizerMetadata NormalizedModel::tokenizer() const {
    const GgufFile& f = *file_;
    TokenizerMetadata t;
    if (const std::string* s = f.get_string("tokenizer.ggml.model")) t.model = *s;
    if (const std::string* s = f.get_string("tokenizer.ggml.pre")) t.pre = *s;
    if (const GgufArray* a = f.get_array("tokenizer.ggml.tokens")) t.tokens = a->strings;
    if (const GgufArray* a = f.get_array("tokenizer.ggml.merges")) t.merges = a->strings;
    if (const GgufArray* a = f.get_array("tokenizer.ggml.token_type")) t.token_types = a->ints;
    t.bos_id = f.get_u32("tokenizer.ggml.bos_token_id");
    t.eos_id = f.get_u32("tokenizer.ggml.eos_token_id");
    t.pad_id = f.get_u32("tokenizer.ggml.padding_token_id");
    t.unk_id = f.get_u32("tokenizer.ggml.unknown_token_id");
    t.add_bos = f.get_bool("tokenizer.ggml.add_bos_token");
    t.add_eos = f.get_bool("tokenizer.ggml.add_eos_token");
    if (const std::string* s = f.get_string("tokenizer.chat_template")) t.chat_template = *s;
    return t;
}

}  // namespace halo::model
