// Loads every tiny qwen35 GGUF (python/tools/make_tiny_model.py) in full mmap mode and
// checks hparams, dtypes, dequantization and — for tiny-f32 — tensor values against the HF
// safetensors checkpoint the GGUFs were converted from.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>

#include "../tensor/ref_data.h"
#include "halo/model/inspect.h"
#include "halo/model/mapped_file.h"
#include "halo/model/model.h"
#include "halo/tensor/quant.h"

using halo::DType;
using halo::model::GgufMode;
using halo::model::LayerKind;
using halo::model::MappedFile;
using halo::model::MtpSource;
using halo::model::MtpTensorOrigin;
using halo::model::NormalizedModel;
using halo::model::WeightRef;
using halo::test::ref_dir;
namespace fs = std::filesystem;

namespace {

fs::path tiny_dir() { return ref_dir() / "tiny"; }

std::vector<fs::path> tiny_files() {
    std::vector<fs::path> v;
    if (!fs::is_directory(tiny_dir())) return v;
    for (const auto& e : fs::directory_iterator(tiny_dir())) {
        if (e.path().extension() == ".gguf") v.push_back(e.path());
    }
    std::sort(v.begin(), v.end());
    return v;
}

std::vector<float> dequant_rows(const WeightRef& w, std::int64_t first, std::int64_t count) {
    const std::int64_t ne0 = w.ne(0);
    const std::size_t rb = halo::tensor::row_bytes(w.type(), ne0);
    const auto data = w.data();
    std::vector<float> out(static_cast<std::size_t>(count * ne0));
    halo::tensor::dequantize_row(w.type(), data.subspan(static_cast<std::size_t>(first) * rb, static_cast<std::size_t>(count) * rb),
                                 out);
    return out;
}

// Minimal safetensors reader (8-byte LE header length, JSON header, raw data).
class SafeTensors {
public:
    explicit SafeTensors(const fs::path& p) : map_(MappedFile::open(p)) {
        const auto b = map_.bytes();
        std::uint64_t n = 0;
        if (b.size() < 8) throw std::runtime_error("safetensors too small");
        std::memcpy(&n, b.data(), 8);
        if (n > b.size() - 8) throw std::runtime_error("bad safetensors header length");
        header_ = nlohmann::json::parse(std::string_view(reinterpret_cast<const char*>(b.data() + 8), n));
        base_ = 8 + n;
    }
    std::vector<float> f32(const std::string& name) const {
        const auto& t = header_.at(name);
        if (t.at("dtype") != "F32") throw std::runtime_error(name + " is not F32");
        const auto begin = t.at("data_offsets")[0].get<std::uint64_t>();
        const auto end = t.at("data_offsets")[1].get<std::uint64_t>();
        const auto b = map_.bytes();
        if (end < begin || base_ + end > b.size() || (end - begin) % 4) throw std::runtime_error("bad offsets " + name);
        std::vector<float> v((end - begin) / 4);
        std::memcpy(v.data(), b.data() + base_ + begin, end - begin);
        return v;
    }

private:
    MappedFile map_;
    nlohmann::json header_;
    std::uint64_t base_ = 0;
};

std::vector<float> all_f32(const WeightRef& w) { return dequant_rows(w, 0, w.info().rows()); }

}  // namespace

TEST(ModelTiny, HparamsMatchGoldenConfigInEveryFile) {
    const auto manifest = halo::test::read_json(tiny_dir() / "golden" / "manifest.json");
    const auto files = tiny_files();
    if (!manifest || files.empty()) GTEST_SKIP() << "tiny model not found under " << tiny_dir();
    const auto& c = (*manifest)["config"];
    ASSERT_EQ(files.size(), 8u) << "expected the 8 tiny GGUFs of make_tiny_model.py";
    for (const auto& p : files) {
        SCOPED_TRACE(p.filename().string());
        const NormalizedModel m = NormalizedModel::load(p, GgufMode::Full);
        const auto& h = m.hparams();
        EXPECT_EQ(h.n_embd, c["hidden_size"].get<std::uint32_t>());
        EXPECT_EQ(h.n_ff, c["intermediate_size"].get<std::uint32_t>());
        EXPECT_EQ(h.n_layer, c["num_hidden_layers"].get<std::uint32_t>());
        EXPECT_EQ(h.nextn_predict_layers, c["mtp_num_hidden_layers"].get<std::uint32_t>());
        EXPECT_EQ(h.block_count, h.n_layer + h.nextn_predict_layers);
        EXPECT_EQ(h.n_head, c["num_attention_heads"].get<std::uint32_t>());
        EXPECT_EQ(h.n_head_kv, c["num_key_value_heads"].get<std::uint32_t>());
        EXPECT_EQ(h.key_length, c["head_dim"].get<std::uint32_t>());
        EXPECT_EQ(h.value_length, c["head_dim"].get<std::uint32_t>());
        EXPECT_EQ(h.gdn_n_k_heads, c["linear_num_key_heads"].get<std::uint32_t>());
        EXPECT_EQ(h.gdn_n_v_heads, c["linear_num_value_heads"].get<std::uint32_t>());
        EXPECT_EQ(h.gdn_head_k_dim, c["linear_key_head_dim"].get<std::uint32_t>());
        EXPECT_EQ(h.gdn_head_v_dim, c["linear_value_head_dim"].get<std::uint32_t>());
        EXPECT_EQ(h.gdn_v_per_k, 3u);
        EXPECT_EQ(h.ssm_conv_kernel, c["linear_conv_kernel_dim"].get<std::uint32_t>());
        EXPECT_EQ(h.full_attention_interval, c["full_attention_interval"].get<std::uint32_t>());
        EXPECT_EQ(h.n_vocab, c["vocab_size"].get<std::int64_t>());
        EXPECT_EQ(h.context_length, c["max_position_embeddings"].get<std::uint32_t>());
        EXPECT_EQ(h.rms_eps, c["rms_norm_eps"].get<float>());
        const auto& rp = c["rope_parameters"];
        EXPECT_EQ(h.rope_freq_base, rp["rope_theta"].get<float>());
        EXPECT_EQ(h.rope_dim, static_cast<std::uint32_t>(c["head_dim"].get<double>() * rp["partial_rotary_factor"].get<double>()));
        for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(h.rope_sections[i], rp["mrope_section"][i].get<int>());
        EXPECT_EQ(h.rope_sections[3], 0);
        EXPECT_EQ(h.n_attn_layers, 2u);
        EXPECT_EQ(h.n_gdn_layers, 6u);
        EXPECT_EQ(h.layer_kind[3], LayerKind::FullAttention);
        EXPECT_EQ(h.layer_kind[7], LayerKind::FullAttention);
        EXPECT_EQ(h.layer_kind[8], LayerKind::FullAttention);  // MTP block
        EXPECT_EQ(m.mtp_source(), MtpSource::Embedded);
        EXPECT_EQ(m.mtp()->embedding_origin, MtpTensorOrigin::Trunk);
        EXPECT_EQ(m.mtp()->lm_head_origin, MtpTensorOrigin::Trunk);
        EXPECT_EQ(m.mtp()->head_norm_origin, MtpTensorOrigin::NextnBlock);
        const auto tok = m.tokenizer();
        EXPECT_EQ(tok.bos_id, c["bos_token_id"].get<std::uint32_t>());
        EXPECT_EQ(tok.eos_id, c["eos_token_id"].get<std::uint32_t>());
        EXPECT_EQ(tok.tokens.size(), 248320u);
        EXPECT_FALSE(m.lm_head_tied());
    }
}

// Every tensor's first and last row dequantizes; quantized embeddings are close to F32.
TEST(ModelTiny, AllDtypesDequantizeAndTrackF32) {
    const auto files = tiny_files();
    if (files.empty()) GTEST_SKIP() << "tiny model not found under " << tiny_dir();
    const NormalizedModel ref = NormalizedModel::load(tiny_dir() / "tiny-f32.gguf");
    const std::vector<float> ref_embd = dequant_rows(ref.token_embd(), 1000, 64);
    const std::vector<float> ref_head = dequant_rows(ref.output(), 2000, 64);
    std::map<std::string, int> seen;
    for (const auto& p : files) {
        SCOPED_TRACE(p.filename().string());
        const NormalizedModel m = NormalizedModel::load(p);
        std::map<std::string, int> counts;
        for (const auto& t : m.gguf().tensors()) {
            const std::string dt(halo::traits(t.type).name);
            ++counts[dt];
            ++seen[dt];
            ASSERT_TRUE(halo::dequant_supported(t.type)) << t.name << " " << dt;
            const WeightRef w(&m.gguf(), &t);
            for (std::int64_t r : {std::int64_t{0}, t.rows() - 1}) {
                const auto v = dequant_rows(w, r, 1);
                for (float x : v) ASSERT_TRUE(std::isfinite(x)) << t.name << " row " << r;
            }
        }
        std::string summary;
        for (const auto& [k, n] : counts) summary += k + ":" + std::to_string(n) + " ";
        std::printf("[ info ] %s %s\n", p.filename().c_str(), summary.c_str());

        // Relative RMS error of quantized embeddings / LM head vs F32. A wrong data offset or
        // block layout gives uncorrelated values (rel. error >= ~1); the worst legitimate case
        // here is 3-bit K-quant noise. Bound 0.25 separates the two with margin.
        auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                num += d * d;
                den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            return std::sqrt(num / den);
        };
        const double e1 = rel(dequant_rows(m.token_embd(), 1000, 64), ref_embd);
        const double e2 = rel(dequant_rows(m.output(), 2000, 64), ref_head);
        std::printf("[ info ]   token_embd %s rel-rms %.4f | output %s rel-rms %.4f\n",
                    std::string(halo::traits(m.token_embd().type()).name).c_str(), e1,
                    std::string(halo::traits(m.output().type()).name).c_str(), e2);
        EXPECT_LT(e1, 0.25);
        EXPECT_LT(e2, 0.25);
    }
    for (const char* t : {"F32", "Q8_0", "Q4_0", "Q5_0", "Q5_1", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL"}) {
        EXPECT_TRUE(seen.contains(t)) << t << " not present in any tiny file";
    }
}

// tiny-f32.gguf vs the HF checkpoint it was converted from (llama.cpp convert_hf_to_gguf).
TEST(ModelTiny, F32TensorsMatchHfCheckpoint) {
    const fs::path st = tiny_dir() / "hf" / "model.safetensors";
    if (!fs::exists(st) || !fs::exists(tiny_dir() / "tiny-f32.gguf")) GTEST_SKIP() << "tiny HF checkpoint not found";
    const SafeTensors hf(st);
    const NormalizedModel m = NormalizedModel::load(tiny_dir() / "tiny-f32.gguf");
    const auto& h = m.hparams();

    auto same = [&](const WeightRef& g, const std::string& hf_name) {
        const auto a = all_f32(g);
        const auto b = hf.f32(hf_name);
        ASSERT_EQ(a.size(), b.size()) << g.name();
        EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * 4), 0) << g.name() << " != " << hf_name;
    };
    // D-004: GGUF stores zero-centered HF norm weights as w + 1 (computed in float32).
    auto plus_one = [&](const WeightRef& g, const std::string& hf_name) {
        const auto a = all_f32(g);
        const auto b = hf.f32(hf_name);
        ASSERT_EQ(a.size(), b.size()) << g.name();
        std::size_t bad = 0;
        for (std::size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i] + 1.0f;
        EXPECT_EQ(bad, 0u) << g.name() << " != " << hf_name << " + 1";
    };

    same(m.token_embd(), "model.embed_tokens.weight");
    same(m.output(), "lm_head.weight");
    plus_one(m.output_norm(), "model.norm.weight");
    for (std::uint32_t il : {0u, 3u}) {
        const auto& L = m.layer(il);
        const std::string p = "model.layers." + std::to_string(il) + ".";
        plus_one(L.attn_norm, p + "input_layernorm.weight");
        plus_one(L.post_attention_norm, p + "post_attention_layernorm.weight");
        same(L.ffn_gate, p + "mlp.gate_proj.weight");
        same(L.ffn_up, p + "mlp.up_proj.weight");
        same(L.ffn_down, p + "mlp.down_proj.weight");
    }
    {
        const auto& A = m.layer(3).attn;
        plus_one(A.q_norm, "model.layers.3.self_attn.q_norm.weight");
        plus_one(A.k_norm, "model.layers.3.self_attn.k_norm.weight");
        same(A.q, "model.layers.3.self_attn.q_proj.weight");
        same(A.k, "model.layers.3.self_attn.k_proj.weight");
        same(A.output, "model.layers.3.self_attn.o_proj.weight");
    }
    {
        // ssm_norm (RMSNormGated) is not zero-centered: stored as-is (D-004).
        const auto& G = m.layer(0).gdn;
        same(G.norm, "model.layers.0.linear_attn.norm.weight");
        // D-004 item 5: GGUF value head j (tiled) = HF head (j % n_k) * (n_v / n_k) + j / n_k (grouped).
        const auto dt = all_f32(G.dt_bias);
        const auto a = all_f32(G.a);
        const auto hf_dt = hf.f32("model.layers.0.linear_attn.dt_bias");
        const auto hf_alog = hf.f32("model.layers.0.linear_attn.A_log");
        const std::uint32_t nk = h.gdn_n_k_heads, r = h.gdn_v_per_k;
        ASSERT_EQ(dt.size(), h.gdn_n_v_heads);
        for (std::uint32_t j = 0; j < h.gdn_n_v_heads; ++j) {
            const std::size_t hf_j = (j % nk) * r + j / nk;
            EXPECT_EQ(dt[j], hf_dt[hf_j]) << "dt head " << j;
            // ssm_a = -exp(A_log). libm expf vs torch exp may differ by 1 ulp: rel 3e-7.
            const float want = -std::exp(hf_alog[hf_j]);
            EXPECT_NEAR(a[j], want, std::fabs(want) * 3e-7f) << "ssm_a head " << j;
        }
    }
    {
        const auto* mtp = m.mtp();
        ASSERT_NE(mtp, nullptr);
        plus_one(mtp->enorm, "mtp.pre_fc_norm_embedding.weight");
        plus_one(mtp->hnorm, "mtp.pre_fc_norm_hidden.weight");
        plus_one(mtp->head_norm, "mtp.norm.weight");
        plus_one(mtp->block.attn_norm, "mtp.layers.0.input_layernorm.weight");
        same(mtp->eh_proj, "mtp.fc.weight");
        same(mtp->block.ffn_gate, "mtp.layers.0.mlp.gate_proj.weight");
    }
}

// Table-order packing (see RealHeaders.TensorsArePackedInTableOrder) plus exact file size:
// validates block geometry for every dtype llama-quantize wrote (incl. Q5_0, Q5_1, Q4_0).
TEST(ModelTiny, TensorsArePackedAndFillTheFile) {
    const auto files = tiny_files();
    if (files.empty()) GTEST_SKIP() << "tiny model not found under " << tiny_dir();
    for (const auto& p : files) {
        SCOPED_TRACE(p.filename().string());
        const auto f = halo::model::GgufFile::open(p, GgufMode::Full);
        std::uint64_t expect = 0;
        for (const auto& t : f.tensors()) {
            ASSERT_EQ(t.offset, expect) << t.name << " " << halo::traits(t.type).name;
            expect = (t.offset + t.n_bytes + f.alignment() - 1) / f.alignment() * f.alignment();
        }
        EXPECT_EQ(f.data_offset() + expect, f.file_size());
    }
}

TEST(ModelTiny, InspectReport) {
    if (!fs::exists(tiny_dir() / "tiny-q4_k_m.gguf")) GTEST_SKIP() << "tiny model not found";
    const NormalizedModel m = NormalizedModel::load(tiny_dir() / "tiny-q4_k_m.gguf");
    const auto r = m.inspect();
    EXPECT_EQ(r.mode, "full");
    EXPECT_EQ(r.n_gdn_layers, 6u);
    EXPECT_EQ(r.n_attn_layers, 2u);
    EXPECT_EQ(r.kv_bytes_per_token, 2u * 2u * (64u + 64u) * 2u);
    EXPECT_EQ(r.gdn_recurrent_state_bytes, 6u * 6u * 32u * 32u * 4u);
    EXPECT_EQ(r.gdn_conv_state_bytes, 6u * 3u * 320u * 4u);
    EXPECT_EQ(r.lm_head_dtype, "Q6_K");
    EXPECT_TRUE(r.mtp_present);
    std::uint64_t sum = 0;
    for (const auto& [k, v] : r.classes) sum += v.bytes;
    EXPECT_EQ(sum, r.total_weight_bytes);
    EXPECT_LE(r.data_offset + r.total_weight_bytes, r.file_size);
}
