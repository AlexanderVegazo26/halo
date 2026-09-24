#pragma once
// NormalizedModel: a validated, typed view of a Qwen3.8 ("qwen35" GGUF architecture) model
// (DECISIONS.md D-004..D-008). Loading checks every expected tensor's presence, dtype and
// shape against the hyperparameters; any mismatch throws MODEL_ERROR naming the tensor.
// Vision tensors or a vision/other-architecture file throw UNSUPPORTED_ERROR.
//
// Tensor shapes use ggml order: ne[0] is the input (contiguous) dimension, so a projection
// y = W x with x in R^in, y in R^out is stored as ne = {in, out}.
//
// Lifetime: every WeightRef/span returned points into GGUF files owned by the model; they
// stay valid for the model's lifetime, including after the model is moved.
// Thread-safety: const methods are safe to call concurrently; attach_mtp() is not.

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "halo/model/gguf.h"

namespace halo::model {

struct InspectReport;  // halo/model/inspect.h

enum class LayerKind : std::uint8_t { GatedDeltaNet, FullAttention };
[[nodiscard]] std::string_view to_string(LayerKind k) noexcept;

/// Hyperparameters from the `qwen35.*` keys plus derived quantities.
struct Qwen35HParams {
    // ---- from GGUF keys ------------------------------------------------------------------
    std::uint32_t context_length = 0;           // qwen35.context_length
    std::uint32_t n_embd = 0;                   // qwen35.embedding_length
    std::uint32_t block_count = 0;              // qwen35.block_count (trunk + MTP blocks)
    std::uint32_t n_ff = 0;                     // qwen35.feed_forward_length
    std::uint32_t n_head = 0;                   // qwen35.attention.head_count
    std::uint32_t n_head_kv = 0;                // qwen35.attention.head_count_kv (default n_head)
    std::uint32_t key_length = 0;               // qwen35.attention.key_length (default n_embd/n_head)
    std::uint32_t value_length = 0;             // qwen35.attention.value_length (default n_embd/n_head)
    float rms_eps = 0.0f;                       // qwen35.attention.layer_norm_rms_epsilon
    std::uint32_t rope_dim = 0;                 // qwen35.rope.dimension_count (default key_length)
    float rope_freq_base = 0.0f;                // qwen35.rope.freq_base (default 10000)
    std::array<std::int32_t, 4> rope_sections{};  // qwen35.rope.dimension_sections (M-RoPE)
    std::uint32_t ssm_conv_kernel = 0;          // qwen35.ssm.conv_kernel
    std::uint32_t ssm_state_size = 0;           // qwen35.ssm.state_size   (GDN head dim, k and v)
    std::uint32_t ssm_group_count = 0;          // qwen35.ssm.group_count  (GDN key heads)
    std::uint32_t ssm_time_step_rank = 0;       // qwen35.ssm.time_step_rank (GDN value heads)
    std::uint32_t ssm_inner_size = 0;           // qwen35.ssm.inner_size   (GDN value dim)
    std::uint32_t full_attention_interval = 4;  // qwen35.full_attention_interval (default 4)
    std::uint32_t nextn_predict_layers = 0;     // qwen35.nextn_predict_layers (default 0)
    bool recurrent_layers_from_file = false;    // qwen35.attention.recurrent_layers present

    // ---- derived --------------------------------------------------------------------------
    std::uint32_t n_layer = 0;               // trunk layers = block_count - nextn_predict_layers
    std::vector<LayerKind> layer_kind;       // size block_count; MTP blocks are FullAttention
    std::uint32_t n_gdn_layers = 0;          // trunk only
    std::uint32_t n_attn_layers = 0;         // trunk only
    std::int64_t n_vocab = 0;                // token_embd ne[1]
    std::uint32_t gdn_n_k_heads = 0;         // = ssm_group_count
    std::uint32_t gdn_n_v_heads = 0;         // = ssm_time_step_rank
    std::uint32_t gdn_head_k_dim = 0;        // = ssm_state_size
    std::uint32_t gdn_head_v_dim = 0;        // = ssm_state_size (llama.cpp qwen35: head_v_dim = d_state)
    std::uint32_t gdn_v_per_k = 0;           // n_v_heads / n_k_heads (GGUF tiled order: v head j uses k head j % n_k)
    std::uint32_t gdn_key_dim = 0;           // n_k_heads * head_k_dim
    std::uint32_t gdn_value_dim = 0;         // n_v_heads * head_v_dim (== ssm_inner_size, validated)
    std::uint32_t gdn_conv_channels = 0;     // 2 * key_dim + value_dim
    std::uint32_t attn_q_dim = 0;            // n_head * key_length (attn_q emits 2x this: Q + gate)
    std::uint32_t attn_kv_dim = 0;           // n_head_kv * key_length
    std::uint32_t attn_v_dim = 0;            // n_head_kv * value_length
    std::uint32_t gqa_ratio = 0;             // n_head / n_head_kv

    /// True if block `il` (trunk or MTP) is full attention. Error(Api) if il >= block_count.
    [[nodiscard]] bool is_attention(std::uint32_t il) const;
};

/// Non-owning reference to one tensor of a GGUF file owned by the model. May be empty
/// (optional tensors); check `present()` before use.
class WeightRef {
public:
    WeightRef() = default;
    WeightRef(const GgufFile* file, const GgufTensorInfo* info) : file_(file), info_(info) {}

    [[nodiscard]] bool present() const noexcept { return info_ != nullptr; }
    [[nodiscard]] const GgufTensorInfo& info() const;  // Error(Api) if empty
    [[nodiscard]] const GgufFile& file() const;         // Error(Api) if empty
    [[nodiscard]] const std::string& name() const { return info().name; }
    [[nodiscard]] DType type() const { return info().type; }
    [[nodiscard]] std::int64_t ne(int i) const;  // Error(Api) unless 0 <= i < 4
    [[nodiscard]] std::uint64_t n_bytes() const { return info().n_bytes; }
    /// Tensor bytes (Error(Model) if the file was opened header-only).
    [[nodiscard]] std::span<const std::byte> data() const { return file().tensor_data(info()); }

private:
    const GgufFile* file_ = nullptr;
    const GgufTensorInfo* info_ = nullptr;
};

/// Full-attention mixer (D-004). `q` is {n_embd, 2 * n_head * key_length}: per head the
/// first key_length outputs are Q, the next key_length the output gate.
struct AttentionWeights {
    WeightRef q, k, v, output, q_norm, k_norm;
};

/// Gated DeltaNet mixer (D-004), GGUF tiled V-head order.
struct GdnWeights {
    WeightRef qkv;      // attn_qkv  {n_embd, conv_channels}
    WeightRef gate;     // attn_gate {n_embd, value_dim}
    WeightRef conv1d;   // ssm_conv1d {conv_kernel, conv_channels}
    WeightRef dt_bias;  // ssm_dt.bias {n_v_heads}
    WeightRef a;        // ssm_a {n_v_heads} (= -exp(A_log))
    WeightRef beta;     // ssm_beta  {n_embd, n_v_heads}
    WeightRef alpha;    // ssm_alpha {n_embd, n_v_heads}
    WeightRef norm;     // ssm_norm  {head_v_dim}
    WeightRef out;      // ssm_out   {value_dim, n_embd}
};

struct LayerWeights {
    std::uint32_t index = 0;
    LayerKind kind = LayerKind::GatedDeltaNet;
    WeightRef attn_norm, post_attention_norm;
    AttentionWeights attn;  // populated iff kind == FullAttention
    GdnWeights gdn;         // populated iff kind == GatedDeltaNet
    WeightRef ffn_gate, ffn_up, ffn_down;
};

enum class MtpSource : std::uint8_t { None, Embedded, SeparateFile };
[[nodiscard]] std::string_view to_string(MtpSource s) noexcept;

/// Where a resolved MTP input/output tensor comes from.
enum class MtpTensorOrigin : std::uint8_t {
    NextnBlock,  // blk.<n>.nextn.embed_tokens / shared_head_head / shared_head_norm
    Trunk,       // the trunk model's token_embd / output / output_norm (D-006 default)
    MtpFile,     // the separate MTP file's own token_embd / output / output_norm copies
};
[[nodiscard]] std::string_view to_string(MtpTensorOrigin o) noexcept;

/// MTP / NextN drafter block (D-005).
struct MtpWeights {
    LayerWeights block;  // index = n_layer, kind FullAttention
    WeightRef eh_proj;   // {2 * n_embd, n_embd}; input = concat(enorm(embed), hnorm(h))
    WeightRef enorm, hnorm;
    WeightRef embedding;         // resolved token embedding used by the MTP block
    WeightRef head_norm;         // resolved final norm (nextn.shared_head_norm or output_norm)
    WeightRef lm_head;           // resolved LM head (nextn.shared_head_head or output)
    MtpTensorOrigin embedding_origin = MtpTensorOrigin::Trunk;
    MtpTensorOrigin head_norm_origin = MtpTensorOrigin::Trunk;
    MtpTensorOrigin lm_head_origin = MtpTensorOrigin::Trunk;
};

/// Tokenizer metadata passthrough (the tokenizer itself lives in the tokenizer module).
/// Views are empty / nullopt when the key is absent.
struct TokenizerMetadata {
    std::string_view model;  // tokenizer.ggml.model
    std::string_view pre;    // tokenizer.ggml.pre
    std::span<const std::string> tokens;
    std::span<const std::string> merges;
    std::span<const std::int64_t> token_types;
    std::optional<std::uint32_t> bos_id, eos_id, pad_id, unk_id;
    std::optional<bool> add_bos, add_eos;
    std::optional<std::string_view> chat_template;  // tokenizer.chat_template
};

class NormalizedModel {
public:
    /// Opens and validates a qwen35 GGUF. Embedded MTP (nextn_predict_layers > 0 and
    /// blk.<n_layer>.* present) is picked up automatically.
    static NormalizedModel load(const std::filesystem::path& path, GgufMode mode = GgufMode::Full);
    static NormalizedModel from_gguf(GgufFile file);

    /// Attaches a separate MTP GGUF (D-006): block_count = n_layer + 1, only blk.<n_layer>.*
    /// plus optional own token_embd/output/output_norm. Hyperparameters must match the
    /// trunk. By default the trunk's token_embd/output/output_norm are used; pass
    /// use_mtp_file_head = true to use the MTP file's copies instead. Throws Error(Config)
    /// if an MTP block is already present, Error(Model) on any mismatch.
    void attach_mtp(const std::filesystem::path& path, GgufMode mode = GgufMode::Full, bool use_mtp_file_head = false);
    void attach_mtp(GgufFile file, bool use_mtp_file_head = false);

    [[nodiscard]] const Qwen35HParams& hparams() const noexcept { return hp_; }
    [[nodiscard]] const GgufFile& gguf() const noexcept { return *file_; }
    [[nodiscard]] const GgufFile* mtp_gguf() const noexcept { return mtp_file_.get(); }

    [[nodiscard]] std::span<const LayerWeights> layers() const noexcept { return layers_; }  // trunk only
    [[nodiscard]] const LayerWeights& layer(std::uint32_t il) const;  // Error(Api) if il >= n_layer
    [[nodiscard]] const WeightRef& token_embd() const noexcept { return token_embd_; }
    [[nodiscard]] const WeightRef& output_norm() const noexcept { return output_norm_; }
    /// LM head: `output` if present, else token_embd (tied; recorded as a warning).
    [[nodiscard]] const WeightRef& output() const noexcept { return output_; }
    [[nodiscard]] bool lm_head_tied() const noexcept { return lm_head_tied_; }

    [[nodiscard]] MtpSource mtp_source() const noexcept { return mtp_source_; }
    [[nodiscard]] const MtpWeights* mtp() const noexcept { return mtp_ ? &*mtp_ : nullptr; }

    [[nodiscard]] TokenizerMetadata tokenizer() const;
    [[nodiscard]] const std::vector<std::string>& warnings() const noexcept { return warnings_; }

    [[nodiscard]] InspectReport inspect() const;  // defined in inspect.cpp

private:
    NormalizedModel() = default;

    std::unique_ptr<GgufFile> file_;
    std::unique_ptr<GgufFile> mtp_file_;
    Qwen35HParams hp_;
    std::vector<LayerWeights> layers_;
    WeightRef token_embd_, output_, output_norm_;
    bool lm_head_tied_ = false;
    MtpSource mtp_source_ = MtpSource::None;
    std::optional<MtpWeights> mtp_;
    std::vector<std::string> warnings_;
};

}  // namespace halo::model
