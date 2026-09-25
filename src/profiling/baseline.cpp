#include "halo/profiling/baseline.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <thread>

#include "artifact_builder.h"
#include "halo/core/error.h"
#include "halo/core/log.h"
#include "halo/profiling/http_client.h"
#include "halo/profiling/sha256.h"

namespace halo::profiling {

namespace {

constexpr std::size_t kMaxTests = 4096;
constexpr std::size_t kMaxSamples = 100000;

// ---- strict JSON field access (tool output is untrusted) ------------------------------------

const nlohmann::json& field(const nlohmann::json& j, const char* key, const char* where) {
    HALO_CHECK(j.is_object(), ErrorCode::Api, "{}: not a JSON object", where);
    const auto it = j.find(key);
    HALO_CHECK(it != j.end(), ErrorCode::Api, "{}: missing '{}'", where, key);
    return *it;
}

std::string str(const nlohmann::json& j, const char* key, const char* where) {
    const auto& v = field(j, key, where);
    HALO_CHECK(v.is_string(), ErrorCode::Api, "{}: '{}' is not a string", where, key);
    return v.get<std::string>();
}

std::uint64_t uint_field(const nlohmann::json& j, const char* key, const char* where, std::uint64_t max) {
    const auto& v = field(j, key, where);
    std::optional<std::uint64_t> out;
    if (v.is_number_unsigned()) out = v.get<std::uint64_t>();
    else if (v.is_number_integer() && v.get<std::int64_t>() >= 0) out = static_cast<std::uint64_t>(v.get<std::int64_t>());
    HALO_CHECK(out && *out <= max, ErrorCode::Api, "{}: '{}' is not an integer in [0, {}]", where, key, max);
    return *out;
}

double num(const nlohmann::json& j, const char* key, const char* where) {
    const auto& v = field(j, key, where);
    HALO_CHECK(v.is_number(), ErrorCode::Api, "{}: '{}' is not a number", where, key);
    const double d = v.get<double>();
    HALO_CHECK(std::isfinite(d) && d >= 0, ErrorCode::Api, "{}: '{}' = {} is not a finite non-negative number", where,
               key, d);
    return d;
}

std::optional<double> opt_num(const nlohmann::json& j, const char* key, const char* where) {
    if (!j.is_object() || !j.contains(key) || j.at(key).is_null()) return std::nullopt;
    return num(j, key, where);
}

BenchmarkRecord base_record(const BaselineParseContext& ctx, std::string mode) {
    BenchmarkRecord r;
    r.suite = "baseline";
    r.engine = ctx.engine;
    r.backend = ctx.backend;
    r.model = ctx.model_name;
    r.pack = ctx.pack;
    r.model_hash = ctx.model_hash;
    r.host_label = ctx.host_label;
    r.mode = std::move(mode);
    r.phase = Phase::Steady;
    return r;
}

std::string join_u(const std::vector<unsigned>& v) {
    std::string s;
    for (const unsigned x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s;
}

void check_flash_attn(const std::string& fa) {
    HALO_CHECK(fa == "on" || fa == "off" || fa == "auto", ErrorCode::Config, "flash_attn '{}' is not on/off/auto", fa);
}

void check_common(const BaselineCommon& c, const std::filesystem::path& binary, const std::filesystem::path* model) {
    HALO_CHECK(!c.host_label.empty(), ErrorCode::Config, "baseline: common.host_label is required (D-001)");
    HALO_CHECK(binary.is_absolute(), ErrorCode::Config, "baseline: binary '{}' must be an absolute path",
               binary.string());
    if (model != nullptr) {
        HALO_CHECK(!model->empty(), ErrorCode::Config, "baseline: model path is empty");
        // A FIFO / device would block the hash forever, and a directory is not a model (S-36).
        std::error_code ec;
        HALO_CHECK(std::filesystem::is_regular_file(*model, ec), ErrorCode::Config,
                   "baseline: model '{}' is not a regular file", model->string());
    }
}

/// extra_args must not redirect the run away from what HALO hashes and binds (S-36):
/// another model, or (llama-server) another host/port.
void check_extra_args(const std::vector<std::string>& extra, std::initializer_list<std::string_view> forbidden,
                      const char* tool) {
    for (const auto& a : extra) {
        for (const std::string_view f : forbidden) {
            const bool eq_form = a.size() > f.size() && a.starts_with(f) && a[f.size()] == '=';
            HALO_CHECK(a != f && !eq_form, ErrorCode::Config,
                       "{}: extra_args may not contain '{}' (HALO sets the model/host/port itself)", tool, f);
        }
    }
}

std::string tail(const std::string& s, std::size_t n = 600) { return s.size() <= n ? s : s.substr(s.size() - n); }

/// Driver string from the before snapshot (best effort; unknown parts say so).
std::string driver_label(std::string_view backend, const std::optional<HardwareState>& hw) {
    if (!hw) return "unknown (hardware state not captured)";
    if (backend == "hip") return "rocm " + hw->rocm_version.value_or("unknown");
    if (backend == "vulkan") {
        return std::format("vulkan {} ({}{})", hw->vulkan_api_version.value_or("unknown"),
                           hw->mesa_version ? "mesa " + *hw->mesa_version : std::string("mesa unknown"),
                           hw->driver_version ? ", " + *hw->driver_version : std::string());
    }
    return "cpu";
}

Invocation make_invocation(const std::filesystem::path& binary, const BuildInfo& bi,
                           const std::vector<std::string>& argv, const ProcessOptions& po) {
    Invocation inv;
    inv.binary = binary.string();
    inv.version = bi.version;
    inv.commit = bi.commit;
    inv.argv = argv;
    inv.environment = recorded_environment(po);
    return inv;
}

std::string model_hash_of(const BaselineCommon& c, const std::filesystem::path& model) {
    if (c.model_hash) return *c.model_hash;
    return sha256_file(model);
}

/// Reads `binary --version` (llama.cpp prints it on stderr).
std::optional<BuildInfo> read_version(const std::filesystem::path& binary, const ProcessOptions& po,
                                      ArtifactBuilder& b) {
    ProcessOptions vo = po;
    vo.timeout = std::chrono::seconds(60);
    const ProcessResult r = run_process({binary.string(), "--version"}, vo);
    try {
        return parse_llama_version(r.err + "\n" + r.out);
    } catch (const Error& e) {
        b.failure(std::format("{} --version: {} (exit {}, stderr: {})", binary.string(), e.what(), r.exit_code,
                              tail(r.err)));
        return std::nullopt;
    }
}

/// Stamps identity fields the builder does not own and adds all records.
void add_all(ArtifactBuilder& b, BaselineParse& p, const Invocation& inv) {
    for (auto& n : p.notes) b.note(n);
    for (auto& f : p.failures) b.failure(f);
    for (auto& r : p.records) {
        r.invocation = inv;
        b.add(std::move(r));
    }
}

bool is_sha256_hex(std::string_view s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

/// Driver label from the hardware state, and PACK_ID (single-file pack: no separate MTP
/// GGUF) when the model hash is a real SHA-256 (not a caller placeholder).
void finish_driver(SuiteArtifact& a, std::string_view backend) {
    for (auto& r : a.records) {
        if (r.driver.empty()) r.driver = driver_label(backend, a.hardware_before);
        if (r.pack_hash.empty() && is_sha256_hex(r.model_hash)) r.pack_hash = make_pack_id(r.model_hash, std::nullopt);
        // The file is hashed, then re-opened by the tool: say so (S-36, TOCTOU).
        if (!r.model_hash.empty()) r.extra["model_hash_taken"] = "before the run (file re-opened by the tool)";
    }
}

}  // namespace

// ---- common -----------------------------------------------------------------------------------

BuildInfo parse_llama_version(std::string_view text) {
    const auto pos = text.find("version: ");
    HALO_CHECK(pos != std::string_view::npos, ErrorCode::Api, "no 'version:' line in tool output");
    const auto eol = std::min(text.find('\n', pos), text.size());
    std::string_view line = text.substr(pos + 9, eol - pos - 9);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
    HALO_CHECK(!line.empty() && line.size() <= 256, ErrorCode::Api, "malformed version line");
    BuildInfo bi;
    bi.version = std::string(line);
    const auto get = [&](std::string_view tag) -> std::string {
        const auto p = line.find(tag);
        if (p == std::string_view::npos) return {};
        const auto s = p + tag.size();
        const auto e = line.find_first_of(",)", s);
        return std::string(line.substr(s, e == std::string_view::npos ? line.size() - s : e - s));
    };
    bi.build = get("build ");
    bi.commit = get("commit ");
    HALO_CHECK(!bi.commit.empty(), ErrorCode::Api, "version line '{}' has no commit", bi.version);
    return bi;
}

std::string_view ggml_backend_name(std::string_view backend) {
    if (backend == "cpu") return "CPU";
    if (backend == "vulkan") return "Vulkan";
    if (backend == "hip") return "ROCm";
    throw_error(ErrorCode::Config, "backend '{}' is not cpu/vulkan/hip", backend);
}

// ---- llama-bench ------------------------------------------------------------------------------

std::vector<std::string> llama_bench_argv(const LlamaBenchConfig& c) {
    check_flash_attn(c.flash_attn);
    (void)ggml_backend_name(c.backend);
    HALO_CHECK(c.repetitions >= 1 && c.repetitions <= 1000, ErrorCode::Config, "llama-bench: repetitions {}",
               c.repetitions);
    HALO_CHECK(!c.n_prompt.empty() && !c.n_gen.empty() && !c.n_depth.empty(), ErrorCode::Config,
               "llama-bench: n_prompt / n_gen / n_depth lists must be non-empty");
    // llama-bench splits -m on ',': one hashed path must be one benchmarked model (S-36).
    HALO_CHECK(c.model.string().find(',') == std::string::npos, ErrorCode::Config,
               "llama-bench: model path '{}' contains ','", c.model.string());
    check_extra_args(c.extra_args, {"-m", "--model", "-hf", "--hf-repo", "-mu", "--model-url"}, "llama-bench");
    std::vector<std::string> a{c.binary.string(), "-m", c.model.string(), "-p", join_u(c.n_prompt), "-n",
                               join_u(c.n_gen),   "-d", join_u(c.n_depth), "-r", std::to_string(c.repetitions),
                               "-ngl", std::to_string(c.n_gpu_layers), "-fa", c.flash_attn, "-o", "json"};
    if (c.threads) {
        a.emplace_back("-t");
        a.push_back(std::to_string(*c.threads));
    }
    if (c.no_warmup) a.emplace_back("--no-warmup");
    a.insert(a.end(), c.extra_args.begin(), c.extra_args.end());
    return a;
}

BaselineParse parse_llama_bench_json(const nlohmann::json& j, const BaselineParseContext& ctx) {
    BaselineParse out;
    HALO_CHECK(j.is_array(), ErrorCode::Api, "llama-bench output is not a JSON array");
    HALO_CHECK(j.size() <= kMaxTests, ErrorCode::Api, "llama-bench output has {} tests (> {})", j.size(), kMaxTests);
    const std::string want = std::string(ggml_backend_name(ctx.backend));
    for (std::size_t ti = 0; ti < j.size(); ++ti) {
        const auto& t = j[ti];
        const std::string where = std::format("llama-bench test[{}]", ti);
        const char* w = where.c_str();
        const std::string commit = str(t, "build_commit", w);
        const std::string backends = str(t, "backends", w);
        const std::string model_type = str(t, "model_type", w);
        const auto n_prompt = uint_field(t, "n_prompt", w, 1U << 24);
        const auto n_gen = uint_field(t, "n_gen", w, 1U << 24);
        const auto n_depth = uint_field(t, "n_depth", w, 1U << 24);
        const double avg_ts = num(t, "avg_ts", w);
        const auto& samples = field(t, "samples_ts", w);
        HALO_CHECK(samples.is_array() && !samples.empty() && samples.size() <= kMaxSamples, ErrorCode::Api,
                   "{}: samples_ts must be a non-empty array of <= {}", where, kMaxSamples);
        const auto& samples_ns = field(t, "samples_ns", w);
        HALO_CHECK(samples_ns.is_array() && samples_ns.size() == samples.size(), ErrorCode::Api,
                   "{}: samples_ns length differs from samples_ts", where);
        HALO_CHECK(n_prompt > 0 || n_gen > 0, ErrorCode::Api, "{}: neither n_prompt nor n_gen is set", where);

        // Backend: the registry names joined by ',' ("CPU" when only the CPU backend).
        bool has = false;
        std::size_t p0 = 0;
        while (p0 <= backends.size()) {
            const auto e = std::min(backends.find(',', p0), backends.size());
            if (backends.substr(p0, e - p0) == want) has = true;
            p0 = e + 1;
        }
        if (!has) {
            out.failures.push_back(std::format("{}: binary reports backends '{}', config expects {} ('{}')", where,
                                               backends, ctx.backend, want));
        }
        if (ctx.expected_commit && commit != *ctx.expected_commit) {
            out.failures.push_back(
                std::format("{}: build_commit '{}' differs from --version commit '{}'", where, commit, *ctx.expected_commit));
        }
        std::string mode;
        if (n_prompt > 0 && n_gen == 0) mode = std::format("pp{}", n_prompt);
        else if (n_gen > 0 && n_prompt == 0) mode = std::format("tg{}", n_gen);
        else mode = std::format("pp{}+tg{}", n_prompt, n_gen);
        if (n_depth > 0) mode += std::format("@d{}", n_depth);

        double sum = 0;
        std::vector<double> ts;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            HALO_CHECK(samples[i].is_number(), ErrorCode::Api, "{}: samples_ts[{}] is not a number", where, i);
            const double v = samples[i].get<double>();
            HALO_CHECK(std::isfinite(v) && v > 0, ErrorCode::Api, "{}: samples_ts[{}] = {} is not positive", where, i, v);
            HALO_CHECK(samples_ns[i].is_number() && samples_ns[i].get<double>() > 0, ErrorCode::Api,
                       "{}: samples_ns[{}] is not positive", where, i);
            ts.push_back(v);
            sum += v;
        }
        const double mean = sum / static_cast<double>(ts.size());
        if (std::fabs(mean - avg_ts) > 0.01 * avg_ts) {
            out.notes.push_back(std::format("{}: mean(samples_ts) {:.3f} differs from avg_ts {:.3f} by > 1 %", where,
                                            mean, avg_ts));
        }
        // Quantization: last token of model_type ("qwen35 ?B Q8_0" -> "Q8_0"); a heuristic.
        const auto sp = model_type.find_last_of(' ');
        const std::string quant = sp == std::string::npos ? model_type : model_type.substr(sp + 1);
        for (std::size_t i = 0; i < ts.size(); ++i) {
            BenchmarkRecord r = base_record(ctx, mode);
            r.repetition = static_cast<std::uint32_t>(i);
            r.quantization = quant;
            r.context = n_depth + n_prompt + n_gen;
            if (n_prompt > 0 && n_gen == 0) r.prompt_tps = ts[i];
            else if (n_gen > 0 && n_prompt == 0) r.decode_tps = ts[i];
            else r.extra["combined_tps"] = ts[i];
            r.extra["sample_ns"] = samples_ns[i].get<double>();
            r.extra["n_prompt"] = n_prompt;
            r.extra["n_gen"] = n_gen;
            r.extra["n_depth"] = n_depth;
            r.extra["avg_ts"] = avg_ts;
            for (const char* k : {"n_threads", "n_gpu_layers", "n_batch", "n_ubatch", "flash_attn"}) {
                if (t.contains(k) && t.at(k).is_number()) r.extra[k] = t.at(k);
            }
            for (const char* k : {"type_k", "type_v", "cpu_info", "gpu_info", "model_type", "test_time", "devices"}) {
                if (t.contains(k) && t.at(k).is_string()) r.extra[std::string("llama_") + k] = t.at(k);
            }
            r.extra["backends"] = backends;
            r.extra["build_commit"] = commit;
            r.extra["quantization_source"] = "model_type (heuristic: last token)";
            out.records.push_back(std::move(r));
        }
    }
    return out;
}

SuiteArtifact run_llama_bench(const LlamaBenchConfig& c) {
    check_common(c.common, c.binary, &c.model);
    const auto argv = llama_bench_argv(c);  // validates the config
    RecordIdentity id;
    id.host_label = c.common.host_label;
    ArtifactBuilder b("baseline", c.common.environment, id, c.common.stability);
    const std::string hash = model_hash_of(c.common, c.model);
    const auto bi = read_version(c.binary, c.common.process, b);
    if (bi) {
        const ProcessResult r = run_process(argv, c.common.process);
        if (!r.ok()) {
            b.failure(std::format("llama-bench: {} (exit {}, signal {}, wall {:.1f} s); stderr: {}",
                                  r.timed_out ? "timed out" : "failed", r.exit_code, r.term_signal, r.wall_s,
                                  tail(r.err)));
        } else if (r.out_truncated) {
            b.failure("llama-bench: stdout exceeded max_output_bytes");
        } else {
            BaselineParseContext ctx{"llama-bench", c.backend, c.common.model_name, c.common.pack, hash,
                                     c.common.host_label, bi->commit};
            try {
                const auto j = nlohmann::json::parse(r.out);
                BaselineParse p = parse_llama_bench_json(j, ctx);
                add_all(b, p, make_invocation(c.binary, *bi, argv, c.common.process));
            } catch (const std::exception& e) {
                b.failure(std::format("llama-bench: unparsable output: {}", e.what()));
            }
        }
    }
    SuiteArtifact a = b.finish();
    finish_driver(a, c.backend);
    return a;
}

// ---- llama-server -----------------------------------------------------------------------------

std::vector<std::string> llama_server_argv(const LlamaServerConfig& c) {
    check_flash_attn(c.flash_attn);
    (void)ggml_backend_name(c.backend);
    HALO_CHECK(c.port != 0, ErrorCode::Config, "llama-server: port 0 (choose a free port)");
    HALO_CHECK(c.parallel >= 1 && c.spec_draft_n_max >= 1, ErrorCode::Config, "llama-server: invalid parallel/draft");
    check_extra_args(c.extra_args,
                     {"-m", "--model", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url", "--host", "--port"},
                     "llama-server");
    std::vector<std::string> a{c.binary.string(), "-m", c.model.string(), "-ngl", std::to_string(c.n_gpu_layers),
                               "--ctx-size", std::to_string(c.ctx_size), "--parallel", std::to_string(c.parallel),
                               "--flash-attn", c.flash_attn};
    if (c.mtp) {
        a.insert(a.end(), {"--spec-type", "draft-mtp", "--spec-draft-n-max", std::to_string(c.spec_draft_n_max)});
    }
    a.insert(a.end(), {"--host", "127.0.0.1", "--port", std::to_string(c.port), "--no-webui"});
    a.insert(a.end(), c.extra_args.begin(), c.extra_args.end());
    return a;
}

BaselineParse parse_llama_server_completion(const nlohmann::json& j, const BaselineParseContext& ctx, bool mtp) {
    BaselineParse out;
    const char* w = "llama-server /completion";
    const auto& t = field(j, "timings", w);
    const char* tw = "llama-server timings";
    const auto prompt_n = uint_field(t, "prompt_n", tw, 1ULL << 32);
    const auto predicted_n = uint_field(t, "predicted_n", tw, 1ULL << 32);
    const double prompt_ms = num(t, "prompt_ms", tw);
    const double predicted_ms = num(t, "predicted_ms", tw);
    const double pps = num(t, "prompt_per_second", tw);
    const double dps = num(t, "predicted_per_second", tw);
    const auto cache_n = t.contains("cache_n") ? uint_field(t, "cache_n", tw, 1ULL << 32) : 0;
    BenchmarkRecord r = base_record(ctx, mtp ? "mtp_completion" : "completion");
    r.context = prompt_n + predicted_n;
    if (prompt_n > 0) r.prompt_tps = pps;
    if (predicted_n > 0) (mtp ? r.decode_effective_tps : r.decode_tps) = dps;
    const bool has_draft = t.contains("draft_n");
    if (has_draft) {
        const auto dn = uint_field(t, "draft_n", tw, 1ULL << 32);
        const auto da = uint_field(t, "draft_n_accepted", tw, 1ULL << 32);
        HALO_CHECK(da <= dn, ErrorCode::Api, "{}: draft_n_accepted {} > draft_n {}", tw, da, dn);
        r.extra["draft_n"] = dn;
        r.extra["draft_n_accepted"] = da;
        if (dn > 0) {
            r.mtp_acceptance = static_cast<double>(da) / static_cast<double>(dn);
        } else {
            out.notes.push_back("llama-server: draft_n = 0 (no drafts proposed); mtp_acceptance left null");
        }
    }
    if (mtp && !has_draft) {
        out.failures.push_back("llama-server: MTP requested but the response has no draft counters "
                               "(speculation not active)");
    }
    if (!mtp && has_draft && t.at("draft_n").get<std::uint64_t>() > 0) {
        out.failures.push_back("llama-server: MTP not requested but the server proposed drafts");
    }
    r.extra["prompt_n"] = prompt_n;
    r.extra["predicted_n"] = predicted_n;
    r.extra["prompt_ms"] = prompt_ms;
    r.extra["predicted_ms"] = predicted_ms;
    r.extra["cache_n"] = cache_n;
    if (cache_n > 0) out.notes.push_back(std::format("llama-server: {} prompt tokens came from cache", cache_n));
    if (j.contains("stop_type") && j.at("stop_type").is_string()) r.extra["stop_type"] = j.at("stop_type");
    if (j.contains("tokens_cached") && j.at("tokens_cached").is_number()) r.extra["tokens_cached"] = j.at("tokens_cached");
    out.records.push_back(std::move(r));
    return out;
}

SuiteArtifact run_llama_server(const LlamaServerConfig& c) {
    check_common(c.common, c.binary, &c.model);
    const auto argv = llama_server_argv(c);
    HALO_CHECK(c.repetitions >= 1, ErrorCode::Config, "llama-server: repetitions must be >= 1");
    RecordIdentity id;
    id.host_label = c.common.host_label;
    ArtifactBuilder b("baseline", c.common.environment, id, c.common.stability);
    const std::string hash = model_hash_of(c.common, c.model);
    const auto bi = read_version(c.binary, c.common.process, b);
    // S-29: whatever already listens on the port would otherwise be taken for our server.
    if (bi && !loopback_port_free(c.port)) {
        b.failure(std::format("llama-server: port {} is already in use on 127.0.0.1; refusing to start "
                              "(a stale or foreign server would answer instead of the spawned one)",
                              c.port));
    } else if (bi) {
        ChildProcess server(argv, c.common.process, c.log_path);  // torn down on every path
        const auto deadline = std::chrono::steady_clock::now() + c.startup_timeout;
        bool ready = false;
        std::string last;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!server.running()) {
                last = std::format("server exited with {}", server.exit_code().value_or(-1));
                break;
            }
            try {
                HttpRequest h;
                h.port = c.port;
                h.path = "/health";
                h.timeout = std::chrono::seconds(2);
                const HttpResponse resp = http_request(h);
                if (resp.status == 200) {
                    ready = true;
                    break;
                }
                last = std::format("/health returned {}", resp.status);  // 503 while loading
            } catch (const Error& e) {
                last = e.what();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::optional<bool> owned;
        if (ready) owned = port_listener_is(server.pid(), c.port);
        if (!ready) {
            b.failure(std::format("llama-server did not become ready within {} ms: {}", c.startup_timeout.count(), last));
        } else if (owned != true) {
            b.failure(std::format("llama-server: the listener on port {} is {} the spawned server (pid {}); "
                                  "results would not be its own",
                                  c.port, owned ? "not" : "not verifiably", server.pid()));
        } else {
            const Invocation inv = make_invocation(c.binary, *bi, argv, c.common.process);
            BaselineParseContext ctx{"llama-server", c.backend, c.common.model_name, c.common.pack, hash,
                                     c.common.host_label, bi->commit};
            for (unsigned i = 0; i < c.warmup + c.repetitions; ++i) {
                HttpRequest req;
                req.method = "POST";
                req.port = c.port;
                req.path = "/completion";
                req.timeout = c.request_timeout;
                req.body = nlohmann::json{{"prompt", c.prompt},
                                          {"n_predict", c.n_predict},
                                          {"temperature", 0.0},
                                          {"cache_prompt", false},
                                          {"seed", c.seed},
                                          {"stream", false}}
                               .dump();
                if (!server.running()) {
                    b.failure(std::format("llama-server exited during the run (before request {})", i));
                    break;
                }
                try {
                    const HttpResponse resp = http_request(req);
                    HALO_CHECK(resp.status == 200, ErrorCode::Api, "/completion returned {}: {}", resp.status,
                               tail(resp.body, 300));
                    BaselineParse p = parse_llama_server_completion(nlohmann::json::parse(resp.body), ctx, c.mtp);
                    for (auto& r : p.records) {
                        r.phase = i < c.warmup ? Phase::Cold : Phase::Steady;
                        r.repetition = i < c.warmup ? i : i - c.warmup;
                    }
                    add_all(b, p, inv);
                } catch (const std::exception& e) {
                    b.failure(std::format("llama-server request {}: {}", i, e.what()));
                    break;
                }
            }
        }
        server.terminate();
    }
    SuiteArtifact a = b.finish();
    finish_driver(a, c.backend);
    return a;
}

// ---- Ollama -----------------------------------------------------------------------------------

BaselineParse parse_ollama_generate(const nlohmann::json& j, const BaselineParseContext& ctx) {
    BaselineParse out;
    const char* w = "ollama /api/generate";
    const auto& done = field(j, "done", w);
    HALO_CHECK(done.is_boolean() && done.get<bool>(), ErrorCode::Api, "{}: response is not done", w);
    const auto pe_count = uint_field(j, "prompt_eval_count", w, 1ULL << 32);
    const auto pe_ns = uint_field(j, "prompt_eval_duration", w, 1ULL << 62);  // nanoseconds
    const auto e_count = uint_field(j, "eval_count", w, 1ULL << 32);
    const auto e_ns = uint_field(j, "eval_duration", w, 1ULL << 62);          // nanoseconds
    BenchmarkRecord r = base_record(ctx, "generate");
    r.context = pe_count + e_count;
    if (pe_count > 0 && pe_ns > 0) r.prompt_tps = static_cast<double>(pe_count) * 1e9 / static_cast<double>(pe_ns);
    if (e_count > 0 && e_ns > 0) r.decode_tps = static_cast<double>(e_count) * 1e9 / static_cast<double>(e_ns);
    if (const auto ld = opt_num(j, "load_duration", w)) r.extra["load_duration_s"] = *ld * 1e-9;
    if (const auto td = opt_num(j, "total_duration", w)) r.extra["total_duration_s"] = *td * 1e-9;
    r.extra["prompt_eval_count"] = pe_count;
    r.extra["eval_count"] = e_count;
    r.extra["duration_unit"] = "ns";
    if (j.contains("model") && j.at("model").is_string()) r.extra["ollama_model"] = j.at("model");
    if (j.contains("done_reason") && j.at("done_reason").is_string()) r.extra["done_reason"] = j.at("done_reason");
    out.records.push_back(std::move(r));
    return out;
}

SuiteArtifact run_ollama(const OllamaConfig& c) {
    HALO_CHECK(!c.common.host_label.empty(), ErrorCode::Config, "ollama: common.host_label is required (D-001)");
    HALO_CHECK(c.repetitions >= 1, ErrorCode::Config, "ollama: repetitions must be >= 1");
    RecordIdentity id;
    id.host_label = c.common.host_label;
    ArtifactBuilder b("baseline", c.common.environment, id, c.common.stability);
    const auto get = [&](const std::string& path) {
        HttpRequest h;
        h.host = c.host;
        h.port = c.port;
        h.path = path;
        h.timeout = std::chrono::seconds(30);
        const HttpResponse resp = http_request(h);
        HALO_CHECK(resp.status == 200, ErrorCode::Api, "ollama {} returned {}", path, resp.status);
        return nlohmann::json::parse(resp.body);
    };
    try {
        const auto ver = get("/api/version");
        BuildInfo bi;
        bi.version = str(ver, "version", "ollama /api/version");
        std::string hash = c.common.model_hash.value_or("");
        if (hash.empty()) {
            const auto tags = get("/api/tags");
            const auto& models = field(tags, "models", "ollama /api/tags");
            HALO_CHECK(models.is_array() && models.size() <= 10000, ErrorCode::Api, "ollama /api/tags: bad models");
            for (const auto& m : models) {
                if (m.is_object() && m.value("name", "") == c.model && m.contains("digest") && m.at("digest").is_string()) {
                    hash = "ollama-digest:" + m.at("digest").get<std::string>();
                }
            }
            HALO_CHECK(!hash.empty(), ErrorCode::Api, "ollama: model '{}' not listed by /api/tags", c.model);
        }
        std::vector<std::string> cmd{"POST", std::format("http://{}:{}/api/generate", c.host, c.port), c.model};
        Invocation inv;
        inv.binary = std::format("ollama (HTTP {}:{})", c.host, c.port);
        inv.version = bi.version;
        inv.argv = cmd;
        BaselineParseContext ctx{"ollama", "ollama", c.common.model_name, c.common.pack, hash, c.common.host_label, {}};
        for (unsigned i = 0; i < c.warmup + c.repetitions; ++i) {
            HttpRequest req;
            req.method = "POST";
            req.host = c.host;
            req.port = c.port;
            req.path = "/api/generate";
            req.timeout = c.request_timeout;
            req.body = nlohmann::json{{"model", c.model},
                                      {"prompt", c.prompt},
                                      {"stream", false},
                                      {"options", {{"temperature", 0.0}, {"seed", c.seed}, {"num_predict", c.num_predict}}}}
                           .dump();
            const HttpResponse resp = http_request(req);
            HALO_CHECK(resp.status == 200, ErrorCode::Api, "ollama /api/generate returned {}: {}", resp.status,
                       tail(resp.body, 300));
            BaselineParse p = parse_ollama_generate(nlohmann::json::parse(resp.body), ctx);
            for (auto& r : p.records) {
                r.phase = i < c.warmup ? Phase::Cold : Phase::Steady;
                r.repetition = i < c.warmup ? i : i - c.warmup;
            }
            add_all(b, p, inv);
        }
    } catch (const std::exception& e) {
        b.failure(std::format("ollama: {}", e.what()));
    }
    return b.finish();
}

// ---- aggregates -------------------------------------------------------------------------------

std::vector<BenchmarkRecord> aggregate_records(const SuiteArtifact& a) {
    std::vector<BenchmarkRecord> out;
    for (const auto& s : a.summaries) {
        if (s.phase != Phase::Steady) continue;
        const BenchmarkRecord* first = nullptr;
        for (const auto& r : a.records) {
            if (r.phase == Phase::Steady && r.suite == s.suite && r.mode == s.mode && r.engine == s.engine &&
                r.context == s.context && r.concurrency == s.concurrency) {
                first = &r;
                break;
            }
        }
        if (first == nullptr) continue;
        BenchmarkRecord agg = *first;
        agg.repetition = 0;
        nlohmann::json metrics = nlohmann::json::object();
        for (const auto& [name, m] : s.metrics) {
            metrics[name] = {{"stats", m.stats}, {"stability", to_string(m.stability)}};
            const double med = m.stats.median;
            if (name == "ttft_ms") agg.ttft_ms = med;
            else if (name == "ttft_cached_ms") agg.ttft_cached_ms = med;
            else if (name == "prompt_tps") agg.prompt_tps = med;
            else if (name == "decode_tps") agg.decode_tps = med;
            else if (name == "decode_effective_tps") agg.decode_effective_tps = med;
            else if (name == "mtp_acceptance") agg.mtp_acceptance = med;
            else if (name == "prefix_cache_hit_rate") agg.prefix_cache_hit_rate = med;
            else if (name == "load_time_s") agg.load_time_s = med;
            else if (name == "efficiency") agg.efficiency = med;
        }
        agg.extra["aggregate"] = {{"method", "median"},
                                  {"n", s.measured_count},
                                  {"conformant", s.conformant},
                                  {"artifact_valid", a.valid},
                                  {"metrics", metrics}};
        out.push_back(std::move(agg));
    }
    return out;
}

}  // namespace halo::profiling
