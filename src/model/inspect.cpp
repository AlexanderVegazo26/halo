#include "halo/model/inspect.h"

#include <charconv>

#include "halo/model/model.h"

namespace halo::model {
namespace {

std::optional<std::uint64_t> block_of(std::string_view name) {
    if (!name.starts_with("blk.")) return std::nullopt;
    name.remove_prefix(4);
    std::uint64_t v = 0;
    const auto [p, ec] = std::from_chars(name.data(), name.data() + name.size(), v);
    if (ec != std::errc{} || p == name.data()) return std::nullopt;
    return v;
}

std::string classify(std::string_view name, std::uint32_t n_layer) {
    if (const auto il = block_of(name); il && *il >= n_layer) return "mtp";
    if (name == "token_embd.weight") return "embeddings";
    if (name == "output.weight") return "lm_head";
    if (name.ends_with("norm.weight")) return "norms";
    if (!name.starts_with("blk.")) return "other";
    const auto dot = name.find('.', 4);
    const std::string_view rest = name.substr(dot + 1);
    if (rest.starts_with("ffn_")) return "ffn";
    if (rest.starts_with("attn_qkv") || rest.starts_with("attn_gate") || rest.starts_with("ssm_")) return "gdn";
    if (rest.starts_with("attn_q.") || rest.starts_with("attn_k.") || rest.starts_with("attn_v.") ||
        rest.starts_with("attn_output")) {
        return "attention";
    }
    return "other";
}

nlohmann::json stats_json(const std::map<std::string, DTypeStat>& m) {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [k, v] : m) j[k] = {{"count", v.count}, {"bytes", v.bytes}};
    return j;
}

}  // namespace

InspectReport NormalizedModel::inspect() const {
    const GgufFile& f = *file_;
    const Qwen35HParams& h = hp_;
    InspectReport r;
    r.source = f.source();
    r.mode = f.has_tensor_data() ? "full" : "header_only";
    r.gguf_version = f.version();
    r.file_size = f.file_size();
    r.data_offset = f.data_offset();
    r.architecture = *f.get_string("general.architecture");
    if (const std::string* n = f.get_string("general.name")) r.name = *n;

    r.block_count = h.block_count;
    r.n_layer = h.n_layer;
    r.n_gdn_layers = h.n_gdn_layers;
    r.n_attn_layers = h.n_attn_layers;
    r.full_attention_interval = h.full_attention_interval;
    r.layer_types_source = h.recurrent_layers_from_file ? "recurrent_layers" : "full_attention_interval";
    for (std::uint32_t i = 0; i < h.n_layer; ++i) {
        if (h.layer_kind[i] == LayerKind::FullAttention) r.attention_layers.push_back(i);
    }
    r.n_embd = h.n_embd;
    r.n_ff = h.n_ff;
    r.n_head = h.n_head;
    r.n_head_kv = h.n_head_kv;
    r.key_length = h.key_length;
    r.value_length = h.value_length;
    r.rope_dim = h.rope_dim;
    r.rope_freq_base = static_cast<double>(h.rope_freq_base);
    r.context_length = h.context_length;
    r.n_vocab = h.n_vocab;
    r.gdn_n_k_heads = h.gdn_n_k_heads;
    r.gdn_n_v_heads = h.gdn_n_v_heads;
    r.gdn_head_k_dim = h.gdn_head_k_dim;
    r.gdn_head_v_dim = h.gdn_head_v_dim;
    r.gdn_conv_kernel = h.ssm_conv_kernel;
    r.gdn_conv_channels = h.gdn_conv_channels;

    for (const auto& t : f.tensors()) {
        const std::string dt(traits(t.type).name);
        r.dtypes[dt].count += 1;
        r.dtypes[dt].bytes += t.n_bytes;
        const std::string cls = classify(t.name, h.n_layer);
        r.classes[cls].count += 1;
        r.classes[cls].bytes += t.n_bytes;
        r.total_weight_bytes += t.n_bytes;
        ++r.n_tensors;
    }
    r.lm_head_dtype = std::string(traits(output_.type()).name);
    r.lm_head_tied = lm_head_tied_;

    const std::uint64_t kv_per_layer = std::uint64_t{h.n_head_kv} * (h.key_length + h.value_length) * 2;
    r.kv_bytes_per_token = std::uint64_t{h.n_attn_layers} * kv_per_layer;
    r.mtp_kv_bytes_per_token = mtp_ ? kv_per_layer : 0;
    r.gdn_recurrent_state_bytes =
        std::uint64_t{h.n_gdn_layers} * h.gdn_n_v_heads * h.gdn_head_k_dim * h.gdn_head_v_dim * 4;
    r.gdn_conv_state_bytes = std::uint64_t{h.n_gdn_layers} * (h.ssm_conv_kernel - 1) * h.gdn_conv_channels * 4;
    r.gdn_state_bytes_per_sequence = r.gdn_recurrent_state_bytes + r.gdn_conv_state_bytes;

    r.mtp_present = mtp_.has_value();
    r.mtp_source = std::string(to_string(mtp_source_));
    if (mtp_) {
        r.mtp_embedding_origin = std::string(to_string(mtp_->embedding_origin));
        r.mtp_lm_head_origin = std::string(to_string(mtp_->lm_head_origin));
        r.mtp_head_norm_origin = std::string(to_string(mtp_->head_norm_origin));
        const GgufFile& mf = mtp_file_ ? *mtp_file_ : f;
        for (const auto& t : mf.tensors()) {
            if (const auto il = block_of(t.name); il && *il == h.n_layer) r.mtp_block_bytes += t.n_bytes;
        }
    }
    if (mtp_file_) {
        r.mtp_file = mtp_file_->source();
        for (const auto& t : mtp_file_->tensors()) {
            const std::string dt(traits(t.type).name);
            r.mtp_file_dtypes[dt].count += 1;
            r.mtp_file_dtypes[dt].bytes += t.n_bytes;
            r.mtp_file_weight_bytes += t.n_bytes;
        }
    }

    r.vision_present = false;
    if (const GgufArray* tags = f.get_array("general.tags"); tags != nullptr) {
        for (const auto& s : tags->strings) {
            if (s.find("image") != std::string::npos || s.find("video") != std::string::npos) {
                r.multimodal_source_model = true;
            }
        }
    }

    const TokenizerMetadata tok = tokenizer();
    r.tokenizer_model = std::string(tok.model);
    r.tokenizer_pre = std::string(tok.pre);
    r.n_tokens = tok.tokens.size();
    r.n_merges = tok.merges.size();
    r.bos_id = tok.bos_id ? std::int64_t{*tok.bos_id} : -1;
    r.eos_id = tok.eos_id ? std::int64_t{*tok.eos_id} : -1;
    r.pad_id = tok.pad_id ? std::int64_t{*tok.pad_id} : -1;
    r.add_bos = tok.add_bos ? (*tok.add_bos ? "true" : "false") : "absent";
    r.chat_template_chars = tok.chat_template ? tok.chat_template->size() : 0;

    r.warnings = warnings_;
    if (r.multimodal_source_model) {
        r.warnings.push_back("source model is multimodal (general.tags); this file has no vision tower, text-only use");
    }
    return r;
}

nlohmann::json InspectReport::to_json() const {
    nlohmann::json j;
    j["source"] = source;
    j["mode"] = mode;
    j["gguf_version"] = gguf_version;
    j["file_size"] = file_size;
    j["data_offset"] = data_offset;
    j["architecture"] = architecture;
    j["name"] = name;
    j["hybrid"] = {{"block_count", block_count},
                   {"n_layer", n_layer},
                   {"gdn_layers", n_gdn_layers},
                   {"attention_layers", n_attn_layers},
                   {"full_attention_interval", full_attention_interval},
                   {"layer_types_source", layer_types_source},
                   {"attention_layer_indices", attention_layers}};
    j["hparams"] = {{"n_embd", n_embd},
                    {"n_ff", n_ff},
                    {"n_head", n_head},
                    {"n_head_kv", n_head_kv},
                    {"key_length", key_length},
                    {"value_length", value_length},
                    {"rope_dim", rope_dim},
                    {"rope_freq_base", rope_freq_base},
                    {"context_length", context_length},
                    {"n_vocab", n_vocab},
                    {"gdn_n_k_heads", gdn_n_k_heads},
                    {"gdn_n_v_heads", gdn_n_v_heads},
                    {"gdn_head_k_dim", gdn_head_k_dim},
                    {"gdn_head_v_dim", gdn_head_v_dim},
                    {"gdn_conv_kernel", gdn_conv_kernel},
                    {"gdn_conv_channels", gdn_conv_channels}};
    j["weights"] = {{"n_tensors", n_tensors},
                    {"total_bytes", total_weight_bytes},
                    {"by_dtype", stats_json(dtypes)},
                    {"by_class", stats_json(classes)},
                    {"lm_head_dtype", lm_head_dtype},
                    {"lm_head_tied", lm_head_tied}};
    j["state"] = {{"kv_bytes_per_token_f16", kv_bytes_per_token},
                  {"mtp_kv_bytes_per_token_f16", mtp_kv_bytes_per_token},
                  {"gdn_recurrent_state_bytes_fp32", gdn_recurrent_state_bytes},
                  {"gdn_conv_state_bytes_fp32", gdn_conv_state_bytes},
                  {"gdn_state_bytes_per_sequence", gdn_state_bytes_per_sequence}};
    j["mtp"] = {{"present", mtp_present},
                {"source", mtp_source},
                {"file", mtp_file},
                {"embedding_origin", mtp_embedding_origin},
                {"lm_head_origin", mtp_lm_head_origin},
                {"head_norm_origin", mtp_head_norm_origin},
                {"block_bytes", mtp_block_bytes},
                {"file_by_dtype", stats_json(mtp_file_dtypes)},
                {"file_total_bytes", mtp_file_weight_bytes}};
    j["vision"] = {{"present", vision_present}, {"multimodal_source_model", multimodal_source_model}};
    j["tokenizer"] = {{"model", tokenizer_model},
                      {"pre", tokenizer_pre},
                      {"n_tokens", n_tokens},
                      {"n_merges", n_merges},
                      {"bos_id", bos_id},
                      {"eos_id", eos_id},
                      {"pad_id", pad_id},
                      {"add_bos", add_bos},
                      {"chat_template_chars", chat_template_chars}};
    j["warnings"] = warnings;
    return j;
}

}  // namespace halo::model
