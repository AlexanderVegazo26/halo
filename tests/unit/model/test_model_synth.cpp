// NormalizedModel validation on synthetic qwen35 files: positive structure checks and one
// negative test per validation rule (missing tensor, wrong shape/dtype, vision, arch, MTP).

#include <gtest/gtest.h>

#include "gguf_builder.h"
#include "halo/core/error.h"
#include "halo/model/inspect.h"
#include "halo/model/model.h"

using halo::DType;
using halo::ErrorCode;
using halo::model::GgufFile;
using halo::model::GgufMode;
using halo::model::LayerKind;
using halo::model::MtpSource;
using halo::model::MtpTensorOrigin;
using halo::model::NormalizedModel;
using namespace halo::test;

namespace {

NormalizedModel load(const GgufSpec& s, GgufMode mode = GgufMode::Full) {
    return NormalizedModel::from_gguf(GgufFile::parse(s.build(mode == GgufMode::Full), mode, "synth.gguf"));
}

template <typename F>
ErrorCode err(F&& f, std::string_view needle) {
    try {
        f();
    } catch (const halo::Error& e) {
        EXPECT_NE(std::string_view(e.what()).find(needle), std::string_view::npos) << e.what();
        return e.code();
    }
    ADD_FAILURE() << "expected halo::Error containing '" << needle << "'";
    return ErrorCode::Cancelled;
}

ErrorCode load_err(const GgufSpec& s, std::string_view needle) {
    return err([&] { (void)load(s); }, needle);
}

}  // namespace

TEST(ModelSynth, LoadsAndDerivesHparams) {
    const NormalizedModel m = load(tiny_qwen35());
    const auto& h = m.hparams();
    EXPECT_EQ(h.block_count, 5u);
    EXPECT_EQ(h.n_layer, 4u);
    EXPECT_EQ(h.n_gdn_layers, 3u);
    EXPECT_EQ(h.n_attn_layers, 1u);
    EXPECT_EQ(h.n_vocab, 64);
    EXPECT_EQ(h.gdn_v_per_k, 2u);
    EXPECT_EQ(h.gdn_conv_channels, 2u * 16u + 32u);
    EXPECT_EQ(h.attn_q_dim, 64u);
    EXPECT_TRUE(h.recurrent_layers_from_file);
    ASSERT_EQ(m.layers().size(), 4u);
    EXPECT_EQ(m.layer(3).kind, LayerKind::FullAttention);
    EXPECT_EQ(m.layer(0).kind, LayerKind::GatedDeltaNet);
    EXPECT_EQ(m.layer(0).gdn.qkv.name(), "blk.0.attn_qkv.weight");
    EXPECT_EQ(m.layer(3).attn.q.ne(1), 128);
    EXPECT_FALSE(m.layer(3).gdn.qkv.present());
    EXPECT_EQ(err([&] { (void)m.layer(4); }, "out of range"), ErrorCode::Api);
    EXPECT_EQ(err([&] { (void)m.layer(3).gdn.qkv.name(); }, "absent"), ErrorCode::Api);
    EXPECT_EQ(m.mtp_source(), MtpSource::Embedded);
    ASSERT_NE(m.mtp(), nullptr);
    EXPECT_EQ(m.mtp()->block.index, 4u);
    EXPECT_EQ(m.mtp()->embedding_origin, MtpTensorOrigin::Trunk);
    EXPECT_EQ(m.mtp()->head_norm_origin, MtpTensorOrigin::NextnBlock);
    EXPECT_EQ(m.mtp()->lm_head.name(), "output.weight");
    // Data is reachable in full mode; values are what the builder wrote.
    const auto norm = m.layer(0).attn_norm.data();
    float v = 0.0f;
    std::memcpy(&v, norm.data(), 4);
    EXPECT_EQ(v, 1.0f);
    const auto tok = m.tokenizer();
    EXPECT_EQ(tok.tokens.size(), 64u);
    EXPECT_EQ(tok.bos_id, 1u);
    EXPECT_EQ(tok.add_bos, false);
    EXPECT_FALSE(tok.pad_id.has_value());
    EXPECT_EQ(tok.chat_template, "{{ messages }}");
    EXPECT_TRUE(m.warnings().empty()) << m.warnings().front();
}

TEST(ModelSynth, DerivesLayerTypesFromIntervalWhenArrayAbsent) {
    const NormalizedModel m = load(tiny_qwen35({.mtp = true, .recurrent_layers_key = false}));
    EXPECT_FALSE(m.hparams().recurrent_layers_from_file);
    EXPECT_EQ(m.hparams().layer_kind,
              (std::vector<LayerKind>{LayerKind::GatedDeltaNet, LayerKind::GatedDeltaNet, LayerKind::GatedDeltaNet,
                                      LayerKind::FullAttention, LayerKind::FullAttention}));
}

TEST(ModelSynth, HeaderOnlyLoadsWithoutData) {
    const NormalizedModel m = load(tiny_qwen35(), GgufMode::HeaderOnly);
    EXPECT_EQ(m.hparams().n_layer, 4u);
    EXPECT_EQ(err([&] { (void)m.token_embd().data(); }, "header-only"), ErrorCode::Model);
}

TEST(ModelSynth, MissingTensorIsNamed) {
    for (const char* name : {"blk.1.ssm_conv1d.weight", "blk.3.attn_k_norm.weight", "output_norm.weight",
                             "blk.2.ffn_down.weight", "blk.4.nextn.eh_proj.weight", "token_embd.weight"}) {
        GgufSpec s = tiny_qwen35();
        s.drop_tensor(name);
        EXPECT_EQ(load_err(s, std::string("missing tensor '") + name + "'"), ErrorCode::Model) << name;
    }
}

TEST(ModelSynth, WrongShapeIsNamed) {
    struct Case { const char* name; std::vector<std::int64_t> ne; };
    for (const Case& c : {Case{"blk.0.attn_qkv.weight", {64, 63}}, Case{"blk.3.attn_q.weight", {64, 64}},
                          Case{"blk.0.ssm_conv1d.weight", {3, 64}}, Case{"blk.2.ssm_a", {3}},
                          Case{"output.weight", {64, 63}}, Case{"blk.4.nextn.eh_proj.weight", {64, 64}},
                          Case{"blk.0.ffn_up.weight", {128, 64}}}) {
        GgufSpec s = tiny_qwen35();
        s.find(c.name)->ne = c.ne;
        EXPECT_EQ(load_err(s, std::string("tensor '") + c.name + "' has shape"), ErrorCode::Model) << c.name;
    }
}

TEST(ModelSynth, WrongDtypeRejected) {
    {
        GgufSpec s = tiny_qwen35();
        s.find("blk.0.attn_norm.weight")->type = static_cast<std::uint32_t>(DType::Q8_0);
        EXPECT_EQ(load_err(s, "blk.0.attn_norm.weight' has dtype Q8_0"), ErrorCode::Model);
    }
    {
        // Correct shape, type HALO cannot dequantize: E=64 is not a multiple of 256, so use
        // a 32-block type without dequant support (Q8_1).
        GgufSpec s = tiny_qwen35();
        s.find("blk.1.ffn_up.weight")->type = static_cast<std::uint32_t>(DType::Q8_1);
        EXPECT_EQ(load_err(s, "cannot dequantize"), ErrorCode::Unsupported);
    }
}

TEST(ModelSynth, VisionAndForeignArchitecturesAreUnsupported) {
    {
        GgufSpec s = tiny_qwen35();
        s.tensor("v.blk.0.attn_q.weight", {64, 64});
        EXPECT_EQ(load_err(s, "vision tower tensor 'v.blk.0.attn_q.weight'"), ErrorCode::Unsupported);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("general.architecture", v_str("clip"));
        EXPECT_EQ(load_err(s, "vision"), ErrorCode::Unsupported);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("clip.has_vision_encoder", v_bool(true));
        EXPECT_EQ(load_err(s, "vision encoder metadata"), ErrorCode::Unsupported);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("general.architecture", v_str("llama"));
        EXPECT_EQ(load_err(s, "architecture 'llama' is not supported"), ErrorCode::Unsupported);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.drop_kv("general.architecture");
        EXPECT_EQ(load_err(s, "missing general.architecture"), ErrorCode::Model);
    }
}

TEST(ModelSynth, InvalidHparamsRejected) {
    {
        GgufSpec s = tiny_qwen35();
        s.drop_kv("qwen35.ssm.state_size");
        EXPECT_EQ(load_err(s, "missing required key qwen35.ssm.state_size"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.attention.recurrent_layers", v_arr_bool({true, true, true, false}));  // len 4 != 5
        EXPECT_EQ(load_err(s, "has length 4, expected block_count 5"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.attention.recurrent_layers", v_arr_bool({true, true, true, false, true}));  // MTP recurrent
        EXPECT_EQ(load_err(s, "MTP block 4 recurrent"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.ssm.inner_size", v_u32(48));
        EXPECT_EQ(load_err(s, "inner_size"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.attention.head_count_kv", v_u32(3));  // > head_count 2
        EXPECT_EQ(load_err(s, "head_count_kv"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.embedding_length", v_str("64"));
        EXPECT_EQ(load_err(s, "expected an integer"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.block_count", v_u32(0));
        EXPECT_EQ(load_err(s, "block_count"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("qwen35.nextn_predict_layers", v_u32(2));
        s.kv("qwen35.attention.recurrent_layers", v_arr_bool({true, true, true, false, false}));
        // n_layer becomes 3 with two MTP blocks; like llama.cpp, only one is supported.
        EXPECT_EQ(load_err(s, "only one MTP block is supported"), ErrorCode::Unsupported);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.tensor("blk.9.attn_norm.weight", {64});
        EXPECT_EQ(load_err(s, "refers to block 9 but block_count is 5"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("tokenizer.ggml.eos_token_id", v_u32(64));
        EXPECT_EQ(load_err(s, "outside the 64-token vocabulary"), ErrorCode::Model);
    }
    {
        GgufSpec s = tiny_qwen35();
        s.kv("tokenizer.ggml.token_type", v_arr_i32({1, 1}));
        EXPECT_EQ(load_err(s, "token_type has 2 entries for 64 tokens"), ErrorCode::Model);
    }
}

TEST(ModelSynth, UnknownTensorWarnsAndTiedHeadFallsBack) {
    GgufSpec s = tiny_qwen35();
    s.tensor("rope_freqs.weight", {16});
    s.drop_tensor("output.weight");
    const NormalizedModel m = load(s);
    EXPECT_TRUE(m.lm_head_tied());
    EXPECT_EQ(m.output().name(), "token_embd.weight");
    ASSERT_EQ(m.warnings().size(), 2u);
    EXPECT_NE(m.warnings()[0].find("tied"), std::string::npos);
    EXPECT_NE(m.warnings()[1].find("rope_freqs.weight"), std::string::npos);
}

TEST(ModelSynth, SeparateMtpFileAttaches) {
    NormalizedModel m = load(tiny_qwen35({.mtp = false, .recurrent_layers_key = true}));
    EXPECT_EQ(m.mtp_source(), MtpSource::None);
    EXPECT_EQ(m.mtp(), nullptr);
    const GgufSpec mtp = tiny_qwen35_mtp_file();
    // As a trunk, an MTP-only file is a clean error.
    EXPECT_EQ(load_err(mtp, "MTP-only GGUF"), ErrorCode::Model);

    m.attach_mtp(GgufFile::parse(mtp.build(), GgufMode::Full, "mtp.gguf"));
    EXPECT_EQ(m.mtp_source(), MtpSource::SeparateFile);
    ASSERT_NE(m.mtp(), nullptr);
    EXPECT_EQ(m.mtp()->embedding_origin, MtpTensorOrigin::Trunk);
    EXPECT_EQ(&m.mtp()->embedding.file(), &m.gguf());
    EXPECT_EQ(m.mtp()->lm_head_origin, MtpTensorOrigin::Trunk);
    EXPECT_EQ(&m.mtp()->block.attn.q.file(), m.mtp_gguf());
    const auto report = m.inspect();
    EXPECT_EQ(report.mtp_source, "separate_file");
    EXPECT_EQ(report.mtp_file, "mtp.gguf");
    EXPECT_GT(report.mtp_file_weight_bytes, report.mtp_block_bytes);

    EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(mtp.build(), GgufMode::Full, "again")); }, "already has"),
              ErrorCode::Config);

    NormalizedModel own = load(tiny_qwen35({.mtp = false, .recurrent_layers_key = true}));
    own.attach_mtp(GgufFile::parse(mtp.build(), GgufMode::Full, "mtp.gguf"), /*use_mtp_file_head=*/true);
    EXPECT_EQ(own.mtp()->embedding_origin, MtpTensorOrigin::MtpFile);
    EXPECT_EQ(own.mtp()->lm_head_origin, MtpTensorOrigin::MtpFile);
    EXPECT_EQ(&own.mtp()->embedding.file(), own.mtp_gguf());
}

TEST(ModelSynth, MtpAttachRejectsMismatches) {
    auto trunk = [] { return load(tiny_qwen35({.mtp = false, .recurrent_layers_key = true})); };
    {
        NormalizedModel m = load(tiny_qwen35());  // embedded MTP already
        EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(tiny_qwen35_mtp_file().build(), GgufMode::Full)); },
                      "already has an embedded"),
                  ErrorCode::Config);
    }
    {
        NormalizedModel m = trunk();
        GgufSpec s = tiny_qwen35_mtp_file();
        s.kv("qwen35.feed_forward_length", v_u32(256));
        EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(s.build(), GgufMode::Full)); }, "feed_forward_length"),
                  ErrorCode::Model);
        EXPECT_EQ(m.mtp_source(), MtpSource::None);  // strong guarantee
    }
    {
        NormalizedModel m = trunk();
        GgufSpec s = tiny_qwen35_mtp_file();
        s.kv("qwen35.rope.freq_base", v_f32(1e6f));
        EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(s.build(), GgufMode::Full)); }, "rope.freq_base"),
                  ErrorCode::Model);
    }
    {
        NormalizedModel m = trunk();
        GgufSpec s = tiny_qwen35_mtp_file();
        s.kv("qwen35.block_count", v_u32(6));
        s.kv("qwen35.attention.recurrent_layers", v_arr_bool({true, true, true, false, true, false}));
        EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(s.build(), GgufMode::Full)); }, "trunk layer count"),
                  ErrorCode::Model);
    }
    {
        NormalizedModel m = trunk();
        GgufSpec s = tiny_qwen35_mtp_file();
        s.tensor("blk.0.attn_norm.weight", {64});
        EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(s.build(), GgufMode::Full)); }, "not an MTP-only file"),
                  ErrorCode::Model);
    }
    {
        NormalizedModel m = trunk();
        GgufSpec s = tiny_qwen35_mtp_file();
        s.drop_tensor("blk.4.nextn.hnorm.weight");
        EXPECT_EQ(err([&] { m.attach_mtp(GgufFile::parse(s.build(), GgufMode::Full)); },
                      "missing tensor 'blk.4.nextn.hnorm.weight'"),
                  ErrorCode::Model);
    }
}

TEST(ModelSynth, InspectFormulas) {
    const NormalizedModel m = load(tiny_qwen35());
    const auto r = m.inspect();
    // 1 attention layer x 1 kv head x (32 + 32) x 2 B
    EXPECT_EQ(r.kv_bytes_per_token, 128u);
    EXPECT_EQ(r.mtp_kv_bytes_per_token, 128u);
    // 3 GDN layers x 2 v-heads x 16 x 16 x 4 B ; conv 3 x (4-1) x 64 x 4 B
    EXPECT_EQ(r.gdn_recurrent_state_bytes, 3u * 2u * 16u * 16u * 4u);
    EXPECT_EQ(r.gdn_conv_state_bytes, 3u * 3u * 64u * 4u);
    std::uint64_t sum = 0;
    for (const auto& [k, v] : r.classes) sum += v.bytes;
    EXPECT_EQ(sum, r.total_weight_bytes);
    EXPECT_EQ(r.classes.at("embeddings").count, 1u);
    EXPECT_EQ(r.classes.at("lm_head").count, 1u);
    EXPECT_EQ(r.classes.at("gdn").count, 3u * 8u);  // qkv, gate, conv1d, dt, a, beta, alpha, out
    EXPECT_EQ(r.classes.at("attention").count, 4u);
    EXPECT_EQ(r.classes.at("ffn").count, 4u * 3u);
    EXPECT_EQ(r.classes.at("mtp").count, 15u);  // 5 common + 6 attention + 4 nextn
    EXPECT_FALSE(r.classes.contains("other"));
    const auto j = r.to_json();
    EXPECT_EQ(j["state"]["kv_bytes_per_token_f16"], 128);
    EXPECT_EQ(j["hybrid"]["attention_layer_indices"], nlohmann::json::array({3}));
    EXPECT_EQ(j["tokenizer"]["add_bos"], "false");
    EXPECT_EQ(j["mtp"]["source"], "embedded");
}
