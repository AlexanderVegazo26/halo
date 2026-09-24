#pragma once
// In-memory GGUF v3 writer for tests: build small, fully controlled files (valid or
// deliberately broken) without touching the filesystem.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "halo/model/gguf.h"
#include "halo/tensor/dtype.h"
#include "halo/tensor/quant.h"

namespace halo::test {

using model::GgufType;

class Bytes {
public:
    std::vector<std::byte> b;
    template <typename T>
    Bytes& put(T v) {
        const auto* p = reinterpret_cast<const std::byte*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
        return *this;
    }
    Bytes& str(std::string_view s) {
        put<std::uint64_t>(s.size());
        const auto* p = reinterpret_cast<const std::byte*>(s.data());
        b.insert(b.end(), p, p + s.size());
        return *this;
    }
    Bytes& append(const Bytes& o) {
        b.insert(b.end(), o.b.begin(), o.b.end());
        return *this;
    }
};

struct KvSpec {
    std::string key;
    Bytes value;  // encoded: u32 type + payload
};

struct TensorSpec {
    std::string name;
    std::vector<std::int64_t> ne;
    std::uint32_t type = 0;                    // ggml type id
    std::optional<std::uint64_t> offset;       // default: packed + aligned
    std::optional<std::uint32_t> n_dims;       // default: ne.size()
    float fill = 0.0f;                         // F32 tensors: constant value
};

// ---- value encoders --------------------------------------------------------------------
inline Bytes v_u32(std::uint32_t v) { Bytes b; b.put<std::uint32_t>(4).put(v); return b; }
inline Bytes v_i32(std::int32_t v) { Bytes b; b.put<std::uint32_t>(5).put(v); return b; }
inline Bytes v_f32(float v) { Bytes b; b.put<std::uint32_t>(6).put(v); return b; }
inline Bytes v_bool(bool v) { Bytes b; b.put<std::uint32_t>(7).put<std::uint8_t>(v ? 1 : 0); return b; }
inline Bytes v_str(std::string_view s) { Bytes b; b.put<std::uint32_t>(8).str(s); return b; }
inline Bytes v_u64(std::uint64_t v) { Bytes b; b.put<std::uint32_t>(10).put(v); return b; }
inline Bytes v_arr_i32(const std::vector<std::int32_t>& a) {
    Bytes b; b.put<std::uint32_t>(9).put<std::uint32_t>(5).put<std::uint64_t>(a.size());
    for (auto v : a) b.put(v);
    return b;
}
inline Bytes v_arr_bool(const std::vector<bool>& a) {
    Bytes b; b.put<std::uint32_t>(9).put<std::uint32_t>(7).put<std::uint64_t>(a.size());
    for (bool v : a) b.put<std::uint8_t>(v ? 1 : 0);
    return b;
}
inline Bytes v_arr_str(const std::vector<std::string>& a) {
    Bytes b; b.put<std::uint32_t>(9).put<std::uint32_t>(8).put<std::uint64_t>(a.size());
    for (const auto& s : a) b.str(s);
    return b;
}

struct GgufSpec {
    std::uint32_t version = 3;
    std::uint64_t alignment = 32;  // used for default offsets; write general.alignment yourself
    std::vector<KvSpec> kvs;
    std::vector<TensorSpec> tensors;

    GgufSpec& kv(std::string key, Bytes v) {
        auto it = std::find_if(kvs.begin(), kvs.end(), [&](const KvSpec& k) { return k.key == key; });
        if (it != kvs.end()) it->value = std::move(v);
        else kvs.push_back({std::move(key), std::move(v)});
        return *this;
    }
    GgufSpec& drop_kv(std::string_view key) {
        std::erase_if(kvs, [&](const KvSpec& k) { return k.key == key; });
        return *this;
    }
    GgufSpec& tensor(std::string name, std::vector<std::int64_t> ne, DType t = DType::F32, float fill = 0.0f) {
        tensors.push_back({std::move(name), std::move(ne), static_cast<std::uint32_t>(t), {}, {}, fill});
        return *this;
    }
    TensorSpec* find(std::string_view name) {
        for (auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
    GgufSpec& drop_tensor(std::string_view name) {
        std::erase_if(tensors, [&](const TensorSpec& t) { return t.name == name; });
        return *this;
    }

    // Byte size of a well-formed spec; 0 for specs the parser must reject anyway (empty or
    // negative dims, unknown type, size overflow) so broken specs still serialize safely.
    static std::uint64_t nbytes(const TensorSpec& t) {
        const auto dt = dtype_from_id(t.type);
        if (!dt || t.ne.empty()) return 0;
        const auto& tr = traits(*dt);
        constexpr std::uint64_t kCap = 1ULL << 40;
        std::uint64_t n = 1;
        for (auto d : t.ne) {
            if (d < 0) return 0;
            const auto ud = static_cast<std::uint64_t>(d);
            if (ud != 0 && n > kCap / ud) return 0;
            n *= ud;
        }
        return n / tr.block_elems * tr.block_bytes;
    }

    /// Serializes. with_data=false produces a header-only image (no padding, no data).
    /// Tensors with an explicit offset do not grow the data section.
    std::vector<std::byte> build(bool with_data = true) const {
        Bytes out;
        out.put<std::uint32_t>(0x46554747u).put(version).put<std::uint64_t>(tensors.size()).put<std::uint64_t>(kvs.size());
        for (const auto& k : kvs) out.str(k.key).append(k.value);
        std::vector<std::uint64_t> offs;
        std::uint64_t cursor = 0;
        for (const auto& t : tensors) {
            const std::uint64_t off = t.offset.value_or(cursor);
            offs.push_back(off);
            if (!t.offset) cursor = (off + nbytes(t) + alignment - 1) / alignment * alignment;
            out.str(t.name).put<std::uint32_t>(t.n_dims.value_or(static_cast<std::uint32_t>(t.ne.size())));
            for (auto d : t.ne) out.put<std::uint64_t>(static_cast<std::uint64_t>(d));
            out.put(t.type).put(off);
        }
        if (!with_data) return out.b;
        while (out.b.size() % alignment) out.put<std::uint8_t>(0);
        const std::size_t data0 = out.b.size();
        out.b.resize(data0 + cursor);
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            const auto& t = tensors[i];
            if (t.type == static_cast<std::uint32_t>(DType::F32) && t.fill != 0.0f && offs[i] + nbytes(t) <= cursor) {
                const std::uint64_t n = nbytes(t) / 4;
                for (std::uint64_t k = 0; k < n; ++k) {
                    std::memcpy(out.b.data() + data0 + offs[i] + k * 4, &t.fill, 4);
                }
            }
        }
        return out.b;
    }
};

/// Small but structurally complete qwen35 model: 4 trunk layers (0-2 GDN, 3 attention),
/// optional embedded MTP block 4. All tensors F32 (zeros; norms 1.0).
struct TinySpecOptions {
    bool mtp = true;
    bool recurrent_layers_key = true;
};

inline GgufSpec tiny_qwen35(TinySpecOptions o = {}) {
    constexpr std::int64_t E = 64, FF = 128, V = 64, NH = 2, NKV = 1, HD = 32, DS = 16, NK = 1, NV = 2, K = 4;
    const std::uint32_t n_layer = 4;
    const std::uint32_t blocks = n_layer + (o.mtp ? 1 : 0);
    GgufSpec s;
    s.kv("general.architecture", v_str("qwen35"))
        .kv("general.name", v_str("tiny-synthetic"))
        .kv("qwen35.block_count", v_u32(blocks))
        .kv("qwen35.context_length", v_u32(4096))
        .kv("qwen35.embedding_length", v_u32(E))
        .kv("qwen35.feed_forward_length", v_u32(FF))
        .kv("qwen35.attention.head_count", v_u32(NH))
        .kv("qwen35.attention.head_count_kv", v_u32(NKV))
        .kv("qwen35.rope.dimension_sections", v_arr_i32({2, 2, 4, 0}))
        .kv("qwen35.rope.freq_base", v_f32(1e7f))
        .kv("qwen35.attention.layer_norm_rms_epsilon", v_f32(1e-6f))
        .kv("qwen35.attention.key_length", v_u32(HD))
        .kv("qwen35.attention.value_length", v_u32(HD))
        .kv("qwen35.ssm.conv_kernel", v_u32(K))
        .kv("qwen35.ssm.state_size", v_u32(DS))
        .kv("qwen35.ssm.group_count", v_u32(NK))
        .kv("qwen35.ssm.time_step_rank", v_u32(NV))
        .kv("qwen35.ssm.inner_size", v_u32(NV * DS))
        .kv("qwen35.full_attention_interval", v_u32(4))
        .kv("qwen35.rope.dimension_count", v_u32(8));
    if (o.mtp) s.kv("qwen35.nextn_predict_layers", v_u32(1));
    if (o.recurrent_layers_key) {
        std::vector<bool> rl{true, true, true, false};
        if (o.mtp) rl.push_back(false);
        s.kv("qwen35.attention.recurrent_layers", v_arr_bool(rl));
    }
    std::vector<std::string> toks;
    for (int i = 0; i < V; ++i) toks.push_back("t" + std::to_string(i));
    s.kv("tokenizer.ggml.model", v_str("gpt2"))
        .kv("tokenizer.ggml.pre", v_str("qwen35"))
        .kv("tokenizer.ggml.tokens", v_arr_str(toks))
        .kv("tokenizer.ggml.token_type", v_arr_i32(std::vector<std::int32_t>(V, 1)))
        .kv("tokenizer.ggml.merges", v_arr_str({"t 1", "t 2"}))
        .kv("tokenizer.ggml.bos_token_id", v_u32(1))
        .kv("tokenizer.ggml.eos_token_id", v_u32(2))
        .kv("tokenizer.ggml.add_bos_token", v_bool(false))
        .kv("tokenizer.chat_template", v_str("{{ messages }}"));

    s.tensor("token_embd.weight", {E, V}).tensor("output_norm.weight", {E}, DType::F32, 1.0f).tensor("output.weight", {E, V});
    const std::int64_t KD = NK * DS, VD = NV * DS, CC = 2 * KD + VD;
    auto common = [&](std::uint32_t il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        s.tensor(p + "attn_norm.weight", {E}, DType::F32, 1.0f)
            .tensor(p + "post_attention_norm.weight", {E}, DType::F32, 1.0f)
            .tensor(p + "ffn_gate.weight", {E, FF})
            .tensor(p + "ffn_up.weight", {E, FF})
            .tensor(p + "ffn_down.weight", {FF, E});
    };
    auto attn = [&](std::uint32_t il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        s.tensor(p + "attn_q.weight", {E, 2 * NH * HD})
            .tensor(p + "attn_k.weight", {E, NKV * HD})
            .tensor(p + "attn_v.weight", {E, NKV * HD})
            .tensor(p + "attn_output.weight", {NH * HD, E})
            .tensor(p + "attn_q_norm.weight", {HD}, DType::F32, 1.0f)
            .tensor(p + "attn_k_norm.weight", {HD}, DType::F32, 1.0f);
    };
    for (std::uint32_t il = 0; il < n_layer; ++il) {
        common(il);
        const std::string p = "blk." + std::to_string(il) + ".";
        if (il == 3) {
            attn(il);
        } else {
            s.tensor(p + "attn_qkv.weight", {E, CC})
                .tensor(p + "attn_gate.weight", {E, VD})
                .tensor(p + "ssm_conv1d.weight", {K, CC})
                .tensor(p + "ssm_dt.bias", {NV})
                .tensor(p + "ssm_a", {NV}, DType::F32, -1.0f)
                .tensor(p + "ssm_beta.weight", {E, NV})
                .tensor(p + "ssm_alpha.weight", {E, NV})
                .tensor(p + "ssm_norm.weight", {DS}, DType::F32, 1.0f)
                .tensor(p + "ssm_out.weight", {VD, E});
        }
    }
    if (o.mtp) {
        common(4);
        attn(4);
        s.tensor("blk.4.nextn.eh_proj.weight", {2 * E, E})
            .tensor("blk.4.nextn.enorm.weight", {E}, DType::F32, 1.0f)
            .tensor("blk.4.nextn.hnorm.weight", {E}, DType::F32, 1.0f)
            .tensor("blk.4.nextn.shared_head_norm.weight", {E}, DType::F32, 1.0f);
    }
    return s;
}

/// Separate-file MTP pack matching tiny_qwen35(mtp=false): block_count 5, nextn 1, only
/// blk.4.* plus its own token_embd/output/output_norm.
inline GgufSpec tiny_qwen35_mtp_file() {
    GgufSpec full = tiny_qwen35({.mtp = true, .recurrent_layers_key = true});
    GgufSpec s;
    s.kvs = full.kvs;
    for (const auto& t : full.tensors) {
        if (t.name.starts_with("blk.4.") || !t.name.starts_with("blk.")) s.tensors.push_back(t);
    }
    return s;
}

}  // namespace halo::test
