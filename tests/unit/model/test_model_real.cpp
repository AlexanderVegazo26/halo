// The three real Qwen3.8-27B GGUF headers (python/tools/fetch_gguf_headers.py), loaded in
// header-only mode. The parser is diffed against the independent Python reader's summaries
// (*.summary.json), and DECISIONS.md facts (D-003, D-006, D-007, D-008) are asserted.

#include <gtest/gtest.h>

#include <filesystem>
#include <map>

#include "../tensor/ref_data.h"
#include "halo/core/error.h"
#include "halo/model/gguf.h"
#include "halo/model/inspect.h"
#include "halo/model/model.h"

using halo::DType;
using halo::ErrorCode;
using halo::model::GgufFile;
using halo::model::GgufMode;
using halo::model::GgufType;
using halo::model::MtpSource;
using halo::model::MtpTensorOrigin;
using halo::model::NormalizedModel;
using halo::test::ref_dir;
namespace fs = std::filesystem;

namespace {

constexpr const char* kFiles[] = {"ggml-org-q4km", "ggml-org-mtp-q4_0", "unsloth-ud-q4kxl"};

fs::path header(const char* k) { return ref_dir() / (std::string(k) + ".header.gguf"); }

bool have_headers() {
    for (const char* k : kFiles) {
        if (!fs::exists(header(k))) return false;
    }
    return true;
}

#define REQUIRE_HEADERS() \
    if (!have_headers()) GTEST_SKIP() << "real GGUF headers not found under " << HALO_REF_DIR << " (fetch_gguf_headers.py)"

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

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace

// Differential: every KV and tensor info our parser reads equals what the Python reader read.
TEST(RealHeaders, ParserMatchesPythonSummaries) {
    REQUIRE_HEADERS();
    for (const char* k : kFiles) {
        SCOPED_TRACE(k);
        const auto summary = halo::test::read_json(ref_dir() / (std::string(k) + ".summary.json"));
        ASSERT_TRUE(summary.has_value());
        const GgufFile f = GgufFile::open(header(k), GgufMode::HeaderOnly);
        EXPECT_EQ(f.version(), (*summary)["version"].get<std::uint32_t>());
        EXPECT_EQ(f.header_size(), (*summary)["header_len"].get<std::uint64_t>());
        EXPECT_EQ(f.file_size(), f.header_size());  // the header file is exactly the header
        EXPECT_EQ(f.alignment(), 32u);
        EXPECT_EQ(f.data_offset(), align_up(f.header_size(), 32));
        // Header-only files end at or before the aligned data start (unsloth: 13 mod 32, MTP: 3 mod 32).
        EXPECT_GE(f.data_offset(), f.file_size());
        if (std::string(k) != "ggml-org-q4km") EXPECT_GT(f.data_offset(), f.file_size());

        // KVs: scalars compared by value; arrays by length (and values when short).
        std::size_t checked = 0;
        for (const auto& [key, val] : (*summary)["kv"].items()) {
            SCOPED_TRACE(key);
            const auto* v = f.find(key);
            ASSERT_NE(v, nullptr);
            if (val.is_boolean()) {
                EXPECT_EQ(f.get_bool(key), val.get<bool>());
            } else if (val.is_number_integer()) {
                EXPECT_EQ(f.get_int(key), val.get<std::int64_t>());
            } else if (val.is_number_float()) {
                EXPECT_EQ(f.get_float(key), val.get<double>());
            } else if (val.is_array()) {
                ASSERT_EQ(v->type, GgufType::Array);
                ASSERT_EQ(v->arr.size(), val.size());
                for (std::size_t i = 0; i < val.size(); ++i) {
                    if (val[i].is_boolean()) EXPECT_EQ(v->arr.bools[i] != 0, val[i].get<bool>());
                    else if (val[i].is_number_integer()) EXPECT_EQ(v->arr.ints.empty() ? static_cast<std::int64_t>(v->arr.uints[i]) : v->arr.ints[i], val[i].get<std::int64_t>());
                    else if (val[i].is_string()) EXPECT_EQ(v->arr.strings[i], val[i].get<std::string>());
                }
            } else {
                const auto s = val.get<std::string>();
                if (s.starts_with("<array len=")) {
                    ASSERT_EQ(v->type, GgufType::Array);
                    EXPECT_EQ(v->arr.size(), std::stoull(s.substr(11)));
                } else if (v->type == GgufType::String) {
                    EXPECT_EQ(*f.get_string(key), s);  // Python decoded with errors="replace"; all valid UTF-8 here
                } else {
                    ADD_FAILURE() << "unexpected summary value for " << key;
                }
            }
            ++checked;
        }
        EXPECT_EQ(checked, f.kvs().size());

        const auto& ts = (*summary)["tensors"];
        ASSERT_EQ(f.tensors().size(), ts.size());
        std::map<std::string, int> types;
        for (std::size_t i = 0; i < ts.size(); ++i) {
            const auto& t = f.tensors()[i];
            EXPECT_EQ(t.name, ts[i]["name"].get<std::string>());
            const auto& dims = ts[i]["dims"];
            ASSERT_EQ(t.n_dims, dims.size()) << t.name;
            for (std::size_t d = 0; d < dims.size(); ++d) EXPECT_EQ(t.ne[d], dims[d].get<std::int64_t>()) << t.name;
            EXPECT_EQ(halo::traits(t.type).name, ts[i]["type"].get<std::string>()) << t.name;
            EXPECT_EQ(t.offset, ts[i]["offset"].get<std::uint64_t>()) << t.name;
            ++types[std::string(halo::traits(t.type).name)];
        }
        if (std::string(k) == "unsloth-ud-q4kxl") {  // D-007
            EXPECT_EQ(types, (std::map<std::string, int>{{"F32", 360}, {"Q5_K", 191}, {"Q8_0", 110}, {"IQ4_XS", 70},
                                                         {"Q4_K", 69}, {"Q6_K", 56}, {"IQ4_NL", 6}, {"Q3_K", 3},
                                                         {"IQ3_S", 1}}));
        } else if (std::string(k) == "ggml-org-q4km") {
            EXPECT_EQ(types, (std::map<std::string, int>{{"F32", 353}, {"Q8_0", 288}, {"Q4_K", 193}, {"Q6_K", 17}}));
        } else {
            EXPECT_EQ(types, (std::map<std::string, int>{{"Q4_0", 10}, {"F32", 8}}));
        }
    }
}

// data_offset + max(offset + n_bytes) must equal the published file size (up to the final
// alignment pad). The remote size was measured by python/tools/fetch_quant_blocks.py.
TEST(RealHeaders, DataLayoutMatchesRemoteFileSize) {
    REQUIRE_HEADERS();
    const auto m = halo::test::read_json(ref_dir() / "quant_blocks" / "manifest.json");
    if (!m || !(*m)["files"].contains("unsloth-ud-q4kxl")) {
        GTEST_SKIP() << "quant_blocks/manifest.json with remote file sizes missing (run fetch_quant_blocks.py)";
    }
    for (const char* k : kFiles) {
        SCOPED_TRACE(k);
        const auto& info = (*m)["files"][k];
        const GgufFile f = GgufFile::open(header(k), GgufMode::HeaderOnly);
        std::uint64_t end = 0;
        for (const auto& t : f.tensors()) end = std::max(end, t.offset + t.n_bytes);
        const auto size = info["file_size"].get<std::uint64_t>();
        EXPECT_EQ(f.data_offset(), info["data_start"].get<std::uint64_t>());
        EXPECT_LE(f.data_offset() + end, size);
        EXPECT_GE(f.data_offset() + align_up(end, f.alignment()), size);
    }
}

// ggml's writer packs tensors in table order, each padded to the alignment (ggml's gguf
// reader requires offset[i] == align_up(offset[i-1] + nbytes[i-1])). Because every n_bytes
// comes from HALO's block geometry, this chain checks row_bytes/traits for every dtype in
// the file — the file-size identity above only constrains the last tensor.
TEST(RealHeaders, TensorsArePackedInTableOrder) {
    REQUIRE_HEADERS();
    for (const char* k : kFiles) {
        SCOPED_TRACE(k);
        const GgufFile f = GgufFile::open(header(k), GgufMode::HeaderOnly);
        std::uint64_t expect = 0;
        for (const auto& t : f.tensors()) {
            ASSERT_EQ(t.offset, expect) << t.name << " (" << halo::traits(t.type).name << " geometry?)";
            expect = align_up(t.offset + t.n_bytes, f.alignment());
        }
    }
}

TEST(RealHeaders, FullModeRejectsHeaderOnlyFile) {
    REQUIRE_HEADERS();
    for (const char* k : kFiles) {
        EXPECT_EQ(err([&] { (void)GgufFile::open(header(k), GgufMode::Full); }, "past end of file"), ErrorCode::Model)
            << k;
    }
}

TEST(RealHeaders, UnslothEmbeddedMtpAndDecisionNumbers) {
    REQUIRE_HEADERS();
    const NormalizedModel m = NormalizedModel::load(header("unsloth-ud-q4kxl"), GgufMode::HeaderOnly);
    const auto& h = m.hparams();
    EXPECT_EQ(h.block_count, 65u);
    EXPECT_EQ(h.n_layer, 64u);
    EXPECT_EQ(h.n_gdn_layers, 48u);
    EXPECT_EQ(h.n_attn_layers, 16u);
    EXPECT_FALSE(h.recurrent_layers_from_file);  // unsloth has no recurrent_layers key: derived
    for (std::uint32_t i = 0; i < 64; ++i) EXPECT_EQ(h.is_attention(i), (i + 1) % 4 == 0) << i;
    EXPECT_TRUE(h.is_attention(64));  // MTP block
    EXPECT_EQ(h.n_embd, 5120u);
    EXPECT_EQ(h.n_ff, 17408u);
    EXPECT_EQ(h.gdn_n_k_heads, 16u);
    EXPECT_EQ(h.gdn_n_v_heads, 48u);
    EXPECT_EQ(h.gdn_conv_channels, 10240u);
    EXPECT_EQ(h.attn_q_dim, 24u * 256u);
    EXPECT_EQ(h.rope_dim, 64u);
    EXPECT_EQ(h.n_vocab, 248320);

    EXPECT_EQ(m.mtp_source(), MtpSource::Embedded);
    ASSERT_NE(m.mtp(), nullptr);
    EXPECT_EQ(m.mtp()->block.index, 64u);
    EXPECT_EQ(m.mtp()->eh_proj.type(), DType::Q6_K);
    EXPECT_EQ(m.mtp()->embedding_origin, MtpTensorOrigin::Trunk);
    EXPECT_EQ(m.mtp()->lm_head_origin, MtpTensorOrigin::Trunk);
    EXPECT_EQ(m.mtp()->head_norm_origin, MtpTensorOrigin::NextnBlock);

    EXPECT_EQ(m.output().type(), DType::Q6_K);             // D-007: LM head Q6_K
    EXPECT_EQ(m.output().n_bytes(), 1042944000u);          // ~1.04 GB read per token
    EXPECT_EQ(m.token_embd().type(), DType::Q4_K);
    EXPECT_EQ(m.layer(14).ffn_down.type(), DType::IQ3_S);

    const auto r = m.inspect();
    EXPECT_EQ(r.kv_bytes_per_token, 64u * 1024u);                     // 16 x 4 x 512 x 2 B
    EXPECT_EQ(r.mtp_kv_bytes_per_token, 4u * 1024u);
    EXPECT_EQ(r.gdn_recurrent_state_bytes, 144u * 1024u * 1024u);    // D-003
    EXPECT_EQ(r.gdn_conv_state_bytes, 48u * 3u * 10240u * 4u);        // D-003 ~5.6 MiB
    EXPECT_EQ(r.mode, "header_only");
    EXPECT_EQ(r.mtp_source, "embedded");
    EXPECT_FALSE(r.vision_present);
    std::uint64_t sum = 0;
    for (const auto& [k, v] : r.classes) sum += v.bytes;
    EXPECT_EQ(sum, r.total_weight_bytes);
    EXPECT_GT(r.classes.at("mtp").bytes, 0u);
    EXPECT_EQ(r.classes.at("lm_head").bytes, 1042944000u);
    EXPECT_FALSE(r.classes.contains("other"));

    const auto t = m.tokenizer();  // D-008
    EXPECT_EQ(t.model, "gpt2");
    EXPECT_EQ(t.pre, "qwen35");
    EXPECT_EQ(t.tokens.size(), 248320u);
    EXPECT_EQ(t.merges.size(), 247587u);
    EXPECT_EQ(t.token_types.size(), 248320u);
    EXPECT_EQ(t.bos_id, 248044u);
    EXPECT_EQ(t.eos_id, 248046u);
    EXPECT_EQ(t.pad_id, 248055u);
    EXPECT_FALSE(t.add_bos.has_value());  // unsloth file carries no add_bos_token key
    ASSERT_TRUE(t.chat_template.has_value());
    EXPECT_EQ(t.chat_template->size(), 9993u);
    EXPECT_EQ(err([&] { (void)m.token_embd().data(); }, "header-only"), ErrorCode::Model);
}

TEST(RealHeaders, GgmlOrgTrunkAndSeparateMtpFile) {
    REQUIRE_HEADERS();
    // The MTP-only pack on its own is rejected with a precise message.
    EXPECT_EQ(err([] { (void)NormalizedModel::load(header("ggml-org-mtp-q4_0"), GgufMode::HeaderOnly); },
                  "MTP-only GGUF"),
              ErrorCode::Model);

    NormalizedModel m = NormalizedModel::load(header("ggml-org-q4km"), GgufMode::HeaderOnly);
    const auto& h = m.hparams();
    EXPECT_EQ(h.block_count, 64u);
    EXPECT_EQ(h.n_layer, 64u);
    EXPECT_TRUE(h.recurrent_layers_from_file);
    EXPECT_EQ(h.n_gdn_layers, 48u);
    EXPECT_EQ(h.n_attn_layers, 16u);
    EXPECT_EQ(m.mtp_source(), MtpSource::None);
    EXPECT_EQ(m.output().type(), DType::Q6_K);
    const auto t = m.tokenizer();
    EXPECT_EQ(t.pad_id, 248044u);
    EXPECT_EQ(t.add_bos, false);
    EXPECT_EQ(t.chat_template->size(), 8952u);
    const auto r0 = m.inspect();
    EXPECT_TRUE(r0.multimodal_source_model);  // general.tags image-text-to-text, no vision tensors
    EXPECT_FALSE(r0.vision_present);
    EXPECT_EQ(r0.kv_bytes_per_token, 64u * 1024u);
    EXPECT_EQ(r0.gdn_recurrent_state_bytes, 144u * 1024u * 1024u);

    m.attach_mtp(header("ggml-org-mtp-q4_0"), GgufMode::HeaderOnly);  // D-006
    EXPECT_EQ(m.mtp_source(), MtpSource::SeparateFile);
    ASSERT_NE(m.mtp(), nullptr);
    EXPECT_EQ(m.mtp()->block.index, 64u);
    EXPECT_EQ(m.mtp()->block.attn.q.type(), DType::Q4_0);
    EXPECT_EQ(m.mtp()->embedding_origin, MtpTensorOrigin::Trunk);  // default: trunk copies
    EXPECT_EQ(m.mtp()->embedding.type(), DType::Q4_K);
    EXPECT_EQ(m.mtp()->lm_head_origin, MtpTensorOrigin::Trunk);
    EXPECT_EQ(m.mtp()->lm_head.type(), DType::Q6_K);
    EXPECT_EQ(m.mtp()->head_norm_origin, MtpTensorOrigin::NextnBlock);
    const auto r = m.inspect();
    EXPECT_EQ(r.mtp_source, "separate_file");
    EXPECT_EQ(r.mtp_embedding_origin, "trunk");
    EXPECT_EQ(r.mtp_file_dtypes.at("Q4_0").count, 10u);
    EXPECT_EQ(r.mtp_kv_bytes_per_token, 4u * 1024u);
    const auto j = r.to_json();
    EXPECT_EQ(j["mtp"]["lm_head_origin"], "trunk");
    EXPECT_EQ(j["weights"]["lm_head_dtype"], "Q6_K");

    NormalizedModel own = NormalizedModel::load(header("ggml-org-q4km"), GgufMode::HeaderOnly);
    own.attach_mtp(header("ggml-org-mtp-q4_0"), GgufMode::HeaderOnly, /*use_mtp_file_head=*/true);
    EXPECT_EQ(own.mtp()->embedding_origin, MtpTensorOrigin::MtpFile);
    EXPECT_EQ(own.mtp()->embedding.type(), DType::Q4_0);
    EXPECT_EQ(own.mtp()->lm_head.type(), DType::Q4_0);

    NormalizedModel u = NormalizedModel::load(header("unsloth-ud-q4kxl"), GgufMode::HeaderOnly);
    EXPECT_EQ(err([&] { u.attach_mtp(header("ggml-org-mtp-q4_0"), GgufMode::HeaderOnly); }, "already has an embedded"),
              ErrorCode::Config);
}
