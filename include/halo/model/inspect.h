#pragma once
// `halo inspect` report (FR-002 / FR-003): a static description of a loaded model and its
// memory needs, computed from metadata only (works in header-only mode).
//
// Formulas (DECISIONS.md D-003):
//   kv_bytes_per_token        = n_attn_layers x n_head_kv x (key_length + value_length) x 2 (f16)
//   mtp_kv_bytes_per_token    = (MTP block present) n_head_kv x (key_length + value_length) x 2
//   gdn_recurrent_state_bytes = n_gdn_layers x n_v_heads x head_k_dim x head_v_dim x 4 (fp32)
//   gdn_conv_state_bytes      = n_gdn_layers x (conv_kernel - 1) x conv_channels x 4 (fp32)
//   gdn_state_bytes_per_sequence = recurrent + conv
//
// Tensor classes (trunk file): embeddings = token_embd; lm_head = output; norms = every
// *norm.weight (attn/post-attention/output/q/k/ssm norms); attention = attn_q/k/v/output of
// attention layers; gdn = attn_qkv/attn_gate/ssm_* (except ssm_norm); ffn = ffn_*;
// mtp = every tensor of block >= n_layer; other = anything else. A separate MTP file is
// reported in its own section (mtp_file_*).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace halo::model {

struct DTypeStat {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
};

struct InspectReport {
    std::string source;
    std::string mode;  // "full" | "header_only"
    std::uint32_t gguf_version = 0;
    std::uint64_t file_size = 0;
    std::uint64_t data_offset = 0;
    std::string architecture;
    std::string name;  // general.name

    // hybrid breakdown
    std::uint32_t block_count = 0;
    std::uint32_t n_layer = 0;
    std::uint32_t n_gdn_layers = 0;
    std::uint32_t n_attn_layers = 0;
    std::uint32_t full_attention_interval = 0;
    std::string layer_types_source;  // "recurrent_layers" | "full_attention_interval"
    std::vector<std::uint32_t> attention_layers;  // trunk indices

    // main hparams
    std::uint32_t n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, key_length = 0, value_length = 0;
    std::uint32_t rope_dim = 0;
    double rope_freq_base = 0.0;
    std::uint32_t context_length = 0;
    std::int64_t n_vocab = 0;
    std::uint32_t gdn_n_k_heads = 0, gdn_n_v_heads = 0, gdn_head_k_dim = 0, gdn_head_v_dim = 0;
    std::uint32_t gdn_conv_kernel = 0, gdn_conv_channels = 0;

    // weights
    std::map<std::string, DTypeStat> dtypes;       // trunk file, by dtype name
    std::map<std::string, DTypeStat> classes;      // trunk file, by tensor class
    std::uint64_t n_tensors = 0;
    std::uint64_t total_weight_bytes = 0;          // trunk file
    std::string lm_head_dtype;
    bool lm_head_tied = false;

    // state
    std::uint64_t kv_bytes_per_token = 0;
    std::uint64_t mtp_kv_bytes_per_token = 0;
    std::uint64_t gdn_recurrent_state_bytes = 0;
    std::uint64_t gdn_conv_state_bytes = 0;
    std::uint64_t gdn_state_bytes_per_sequence = 0;

    // MTP
    bool mtp_present = false;
    std::string mtp_source;  // none | embedded | separate_file
    std::string mtp_file;
    std::string mtp_embedding_origin, mtp_lm_head_origin, mtp_head_norm_origin;
    std::uint64_t mtp_block_bytes = 0;  // blk.<n_layer>.* tensors
    std::map<std::string, DTypeStat> mtp_file_dtypes;
    std::uint64_t mtp_file_weight_bytes = 0;

    // vision
    bool vision_present = false;          // always false for a loaded model (rejected otherwise)
    bool multimodal_source_model = false; // general.tags mentions image/video (tower not in file)

    // tokenizer summary
    std::string tokenizer_model, tokenizer_pre;
    std::uint64_t n_tokens = 0, n_merges = 0;
    std::int64_t bos_id = -1, eos_id = -1, pad_id = -1;  // -1 = absent
    std::string add_bos;  // "true" | "false" | "absent"
    std::uint64_t chat_template_chars = 0;

    std::vector<std::string> warnings;

    [[nodiscard]] nlohmann::json to_json() const;
};

}  // namespace halo::model
