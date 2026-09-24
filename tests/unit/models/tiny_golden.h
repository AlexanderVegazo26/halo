#pragma once
// Shared helpers for tests on the tiny qwen35 model and its golden outputs
// (python/tools/make_tiny_model.py; HALO_REF_DIR/tiny). Used by tests/unit/{models,
// speculative,runtime}.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/backends/cpu/thread_pool.h"
#include "halo/kv_cache/paged_kv.h"
#include "halo/model/model.h"
#include "halo/models/qwen35.h"
#include "halo/state/gdn_state.h"

namespace halo::test {

inline std::filesystem::path tiny_dir() { return std::filesystem::path(HALO_REF_DIR) / "tiny"; }

inline std::vector<char> read_file_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

/// golden/manifest.json + raw little-endian arrays.
class Golden {
public:
    static std::optional<Golden> load() {
        const auto dir = tiny_dir() / "golden";
        std::ifstream f(dir / "manifest.json");
        if (!f) return std::nullopt;
        Golden g;
        g.dir_ = dir;
        g.manifest_ = nlohmann::json::parse(f);
        return g;
    }
    [[nodiscard]] bool has(const std::string& name) const { return manifest_["tensors"].contains(name); }
    [[nodiscard]] std::vector<std::int64_t> shape(const std::string& name) const {
        return manifest_["tensors"].at(name).at("shape").get<std::vector<std::int64_t>>();
    }
    [[nodiscard]] std::vector<float> f32(const std::string& name) const { return read<float>(name, "float32"); }
    [[nodiscard]] std::vector<std::int32_t> i32(const std::string& name) const { return read<std::int32_t>(name, "int32"); }
    [[nodiscard]] const nlohmann::json& manifest() const { return manifest_; }

private:
    template <class T>
    std::vector<T> read(const std::string& name, const char* dtype) const {
        const auto& e = manifest_["tensors"].at(name);
        if (e.at("dtype") != dtype) throw std::runtime_error(name + " has dtype " + e.at("dtype").get<std::string>());
        std::size_t n = 1;
        for (const auto& d : e.at("shape")) n *= d.get<std::size_t>();
        const auto bytes = read_file_bytes(dir_ / e.at("file").get<std::string>());
        if (bytes.size() != n * sizeof(T)) throw std::runtime_error(name + ": size mismatch");
        std::vector<T> v(n);
        std::memcpy(v.data(), bytes.data(), bytes.size());
        return v;
    }
    std::filesystem::path dir_;
    nlohmann::json manifest_;
};

/// A sequence's KV + GDN state for tests.
struct TestSeq {
    kv_cache::SequenceKv kv;
    kv_cache::SequenceKv mtp_kv;
    state::GdnState gdn;
    TestSeq(kv_cache::KvPool& pool, kv_cache::KvPool& mtp_pool, const state::GdnShape& shape, std::size_t slots)
        : kv(pool), mtp_kv(mtp_pool), gdn(shape, slots) {}
};

/// A loaded tiny model + pools sized for a few test sequences.
struct TinyModel {
    std::unique_ptr<model::NormalizedModel> nm;
    std::unique_ptr<cpu::ThreadPool> pool;
    std::unique_ptr<models::Qwen35> model;
    std::unique_ptr<kv_cache::KvPool> kv_pool;
    std::unique_ptr<kv_cache::KvPool> mtp_pool;

    static std::unique_ptr<TinyModel> load(const std::string& file, std::size_t kv_blocks = 512) {
        const auto p = tiny_dir() / file;
        if (!std::filesystem::exists(p)) return nullptr;
        auto t = std::make_unique<TinyModel>();
        t->nm = std::make_unique<model::NormalizedModel>(model::NormalizedModel::load(p));
        t->pool = std::make_unique<cpu::ThreadPool>(cpu::ThreadPool::default_threads());
        t->model = std::make_unique<models::Qwen35>(*t->nm, t->pool.get());
        t->kv_pool = std::make_unique<kv_cache::KvPool>(t->model->kv_layout(), kv_blocks);
        t->mtp_pool = std::make_unique<kv_cache::KvPool>(t->model->mtp_kv_layout(), kv_blocks);
        return t;
    }
    [[nodiscard]] std::unique_ptr<TestSeq> seq(std::size_t slots = 4) const {
        return std::make_unique<TestSeq>(*kv_pool, *mtp_pool, model->gdn_shape(), slots);
    }
};

/// max |a - b| and max |ref| over one row.
struct RowErr {
    double max_abs_err = 0;
    double max_abs_ref = 0;
    double rms_err = 0;
    double rms_ref = 0;
};

inline RowErr row_err(const float* a, const float* ref, std::size_t n) {
    RowErr e;
    double se = 0, sr = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(ref[i]));
        e.max_abs_err = std::max(e.max_abs_err, d);
        e.max_abs_ref = std::max(e.max_abs_ref, std::abs(static_cast<double>(ref[i])));
        se += d * d;
        sr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    e.rms_err = std::sqrt(se / static_cast<double>(n));
    e.rms_ref = std::sqrt(sr / static_cast<double>(n));
    return e;
}

}  // namespace halo::test
