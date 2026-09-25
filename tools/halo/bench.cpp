// `halo bench {micro|model|system}`: wires the halo_profiling suites (docs/benchmarks.md).
//
//   micro   CPU reference kernels at Qwen3.8-27B shapes (halo_backend_cpu); needs no model.
//   model   the PRD §18 / PR-004 matrix over runtime::Engine (needs the runtime).
//   system  N concurrent agents replaying a frozen workload (needs the runtime).
//
// The artifact (schema halo.bench.artifact/1) goes to --out ('-' = stdout). D-001: numbers
// from a development host exercise the harness only; --host-label says where they came from.

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>

#include <nlohmann/json.hpp>

#include "args.h"
#include "commands.h"
#include "config.h"
#include "halo/core/error.h"

#if HALO_CLI_HAVE_PROFILING
#include "halo/profiling/suite.h"
#include "halo/profiling/workload.h"
#include "halo/template/chat_template.h"
#include "halo/tokenizer/tokenizer.h"
#endif
#if HALO_CLI_HAVE_CPU
#include <random>

#include "halo/backends/cpu/ops.h"
#include "halo/backends/cpu/thread_pool.h"
#endif

namespace halo::cli {

using nlohmann::json;

#if !HALO_CLI_HAVE_PROFILING

int cmd_bench(const std::vector<std::string>&, Context& ctx) {
    *ctx.err << "halo: benchmarks are not part of this build (halo_profiling not built)\n";
    return kExitUsage;
}

#else

namespace {

const std::vector<OptionSpec>& common_opts() {
    static const std::vector<OptionSpec> o = {
        {"host-label", true, 0, false, false, "where this runs, e.g. dev-host or evo-x2 (required, D-001)"},
        {"power-mode", true, 0, false, false, "EVO-X2 BIOS/EC performance mode (not visible in sysfs)"},
        {"out", true, 'o', false, false, "artifact JSON path ('-' = stdout)"},
        {"thermal-threshold-c", true, 0, false, false, "max |T_after - T_before| (default 5)"},
        {"allow-nonconformant", false, 0, false, false, "allow fewer runs than the methodology minimum (smoke runs)"},
        {"warmup", true, 0, false, false, "warm-up runs"},
        {"repetitions", true, 0, false, false, "measured repetitions (model/system; default 3)"},
        {"root", true, 0, false, true, "filesystem root for the hardware snapshots (tests)"},
    };
    return o;
}

std::vector<OptionSpec> join(std::vector<OptionSpec> a, const std::vector<OptionSpec>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

profiling::EnvironmentOptions environment(const ParsedArgs& a, const Context& ctx) {
    profiling::EnvironmentOptions e;
    e.discovery = discovery_options(a.value("root").value_or("/"), ctx.env);
    e.platform_power_mode = a.value("power-mode").value_or("");
    if (auto t = a.number("thermal-threshold-c", 0.0, 100.0)) e.thermal_drift_threshold_c = *t;
    return e;
}

profiling::RecordIdentity identity(const ParsedArgs& a, const std::optional<std::string>& model,
                                   const std::string& backend) {
    profiling::RecordIdentity id;
    const auto label = a.value("host-label");
    if (!label) throw UsageError("--host-label is required (D-001: every record says where it was measured)");
    id.host_label = *label;
    if (model) {
        id.model = std::filesystem::path(*model).filename().string();
        id.pack = *model;
    }
    id.backend = backend;
    return id;
}

std::vector<std::uint64_t> u64_list(const std::string& s, const std::string& what, std::uint64_t lo, std::uint64_t hi) {
    std::vector<std::uint64_t> out;
    for (const auto& item : split_list(s, what)) out.push_back(parse_u64(item, what, lo, hi));
    return out;
}

#if HALO_CLI_HAVE_CPU
/// CPU reference kernels at Qwen3.8-27B shapes (D-003/D-004). Buffers are owned by the
/// closures. Dev-host timings only (D-001).
std::vector<profiling::MicroBenchmark> cpu_micro_benchmarks(std::shared_ptr<cpu::ThreadPool> pool) {
    struct Buffers {
        std::vector<float> a, b, c, d, e, state, out;
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto fill = [&](std::size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng);
        return v;
    };
    std::vector<profiling::MicroBenchmark> out;
    {
        auto b = std::make_shared<Buffers>();
        b->a = fill(5120);
        b->b = fill(5120);
        b->out.assign(5120, 0.0f);
        out.push_back({"RMS_NORM/T=1,D=5120", [b, pool] {
                           cpu::rms_norm(cpu::ConstRows(std::span<const float>(b->a), 1, 5120), b->b, 1e-6f,
                                         cpu::Rows(std::span<float>(b->out), 1, 5120), pool.get());
                       },
                       3ULL * 5120 * 4});
    }
    {
        auto b = std::make_shared<Buffers>();
        b->a = fill(17408);
        b->b = fill(17408);
        b->out.assign(17408, 0.0f);
        out.push_back({"SWIGLU/T=1,D=17408", [b, pool] {
                           cpu::swiglu(cpu::ConstRows(std::span<const float>(b->a), 1, 17408),
                                       cpu::ConstRows(std::span<const float>(b->b), 1, 17408),
                                       cpu::Rows(std::span<float>(b->out), 1, 17408), pool.get());
                       },
                       3ULL * 17408 * 4});
    }
    {
        auto b = std::make_shared<Buffers>();
        b->a = fill(248320);
        b->out.assign(248320, 0.0f);
        out.push_back({"SOFTMAX/T=1,V=248320", [b, pool] {
                           cpu::softmax_rows(cpu::ConstRows(std::span<const float>(b->a), 1, 248320),
                                             cpu::Rows(std::span<float>(b->out), 1, 248320), pool.get());
                       },
                       2ULL * 248320 * 4});
    }
    {
        const cpu::GdnDims dims{16, 48, 128, 128, cpu::GdnHeadMapping::Tiled};
        auto b = std::make_shared<Buffers>();
        b->a = fill(16 * 128);  // q
        b->b = fill(16 * 128);  // k
        b->c = fill(48 * 128);  // v
        b->d.assign(48, -0.1f);  // g
        b->e.assign(48, 0.5f);   // beta
        b->state.assign(48 * 128 * 128, 0.0f);
        b->out.assign(48 * 128, 0.0f);
        const std::uint64_t state_bytes = 48ULL * 128 * 128 * 4;
        out.push_back({"GATED_DELTANET/decode,T=1,n_v=48,n_k=16,d=128", [b, pool, dims] {
                           const cpu::GdnInputs in{cpu::ConstRows(std::span<const float>(b->a), 1, 16 * 128),
                                                   cpu::ConstRows(std::span<const float>(b->b), 1, 16 * 128),
                                                   cpu::ConstRows(std::span<const float>(b->c), 1, 48 * 128),
                                                   cpu::ConstRows(std::span<const float>(b->d), 1, 48),
                                                   cpu::ConstRows(std::span<const float>(b->e), 1, 48)};
                           cpu::gated_delta_rule_recurrent(dims, in, b->state,
                                                           cpu::Rows(std::span<float>(b->out), 1, 48 * 128),
                                                           cpu::GdnQkParams{}, pool.get());
                       },
                       2 * state_bytes});
    }
    return out;
}
#endif

/// Encoder for the system suite from the engine's tokenizer + chat template: the
/// conversation is rendered with the model's template; synthetic segments are spliced in as
/// deterministic token runs from the Normal-token range (profiling::synthetic_tokens).
profiling::WorkloadEncoder engine_encoder(const runtime::Engine& engine) {
    const tokenizer::Tokenizer* tok = &engine.tokenizer();
    const chat::ChatTemplate* tmpl = &engine.chat_template();
    std::int32_t normal_range = static_cast<std::int32_t>(tok->vocab_size());
    for (std::size_t i = 0; i < tok->vocab_size(); ++i) {
        const auto t = tok->token_type(static_cast<std::int32_t>(i));
        if (t == tokenizer::TokenType::Control || t == tokenizer::TokenType::UserDefined) {
            normal_range = static_cast<std::int32_t>(i);
            break;
        }
    }
    return [tok, tmpl, normal_range](const profiling::Workload& w, const profiling::AgentScript& a, std::size_t turn) {
        static constexpr std::string_view kMark = "HALOSYNTHSEG";
        nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
        std::vector<const profiling::WorkloadMessage*> synth;
        const auto add = [&](const profiling::WorkloadMessage& m) {
            std::string content;
            if (m.text) {
                content = *m.text;
            } else {
                content = std::string(kMark) + std::to_string(synth.size()) + "Z";
                synth.push_back(&m);
            }
            msgs.push_back({{"role", m.role}, {"content", content}});
        };
        for (const auto& m : a.preamble) add(m);
        for (std::size_t t = 0; t <= turn && t < a.turns.size(); ++t) add(a.turns[t]);
        chat::RenderOptions opt;
        const std::string text = tmpl->apply(msgs, w.tools.empty() ? nlohmann::ordered_json(nullptr)
                                                                  : nlohmann::ordered_json::parse(w.tools.dump()),
                                             opt);
        // The workload files are HALO's own frozen inputs, so their text is tokenized with
        // specials parsed (as the template intends); synthetic markers become token runs.
        std::vector<std::int32_t> ids;
        std::size_t pos = 0;
        while (true) {
            const auto m = text.find(kMark, pos);
            const std::string_view part(text.data() + pos, (m == std::string::npos ? text.size() : m) - pos);
            const auto enc = tok->encode(part, true);
            ids.insert(ids.end(), enc.begin(), enc.end());
            if (m == std::string::npos) break;
            const auto z = text.find('Z', m);
            const std::size_t idx = std::stoul(text.substr(m + kMark.size(), z - m - kMark.size()));
            const auto* seg = synth.at(idx);
            const auto st = profiling::synthetic_tokens(seg->seed, *seg->synthetic_tokens, normal_range);
            ids.insert(ids.end(), st.begin(), st.end());
            pos = z + 1;
        }
        return ids;
    };
}

int finish(const profiling::SuiteArtifact& art, const ParsedArgs& a, Context& ctx) {
    json j = art;
    const std::string out = a.value("out").value_or("-");
    if (out == "-") {
        *ctx.out << j.dump(2) << "\n";
    } else {
        std::ofstream f(out, std::ios::binary);
        HALO_CHECK(f.good(), ErrorCode::Io, "cannot write {}", out);
        f << j.dump(2) << "\n";
        HALO_CHECK(f.good(), ErrorCode::Io, "writing {} failed", out);
    }
    bool failed = false;
    for (const auto& n : art.notes) failed = failed || n.starts_with("FAILED");
    *ctx.err << std::format("[halo bench] suite {}: {} records, {} summaries, conformant {}, valid {}{}\n", art.suite,
                            art.records.size(), art.summaries.size(), art.conformant, art.valid,
                            out == "-" ? "" : ", written to " + out);
    for (const auto& n : art.notes) *ctx.err << "  note: " << n << "\n";
    if (!art.valid) {
        *ctx.err << "  this artifact is not valid for comparison (thermal drift, unpinned power mode, or below the "
                    "methodology minimum)\n";
    }
    return failed ? kExitFailure : kExitOk;
}

void stamp_invocation(profiling::SuiteArtifact& art, const std::vector<std::string>& args, const Context& ctx) {
    profiling::Invocation inv;
    inv.binary = ctx.argv0.empty() ? "halo" : ctx.argv0.front();
    inv.version = "0.2.0";
    inv.argv.push_back(inv.binary);
    inv.argv.push_back("bench");
    inv.argv.insert(inv.argv.end(), args.begin(), args.end());
    for (const auto& [k, v] : ctx.env) {
        if (k.starts_with("HALO_") && k != "HALO_API_KEY") inv.environment[k] = v;
    }
    for (auto& r : art.records) r.invocation = inv;
}

int bench_micro(const std::vector<std::string>& args, Context& ctx) {
    const auto opts = join(common_opts(), {{"iterations", true, 0, false, false, "measured iterations (default 20)"},
                                           {"filter", true, 0, false, false, "regex over benchmark names"},
                                           {"threads", true, 0, false, false, "CPU threads (default: all)"},
                                           {"list", false, 0, false, false, "list the benchmarks and exit"}});
    const std::string usage = "usage: halo bench micro --host-label L [options]\n" + options_help(opts);
    const ParsedArgs a = parse_args(args, opts, usage);
    if (!a.positionals.empty()) throw UsageError("bench micro takes no positional arguments\n" + usage);
#if HALO_CLI_HAVE_CPU
    const auto threads = a.u64("threads", 1, 1024).value_or(cpu::ThreadPool::default_threads());
    auto pool = std::make_shared<cpu::ThreadPool>(threads == 0 ? 1 : threads);
    auto benches = cpu_micro_benchmarks(pool);
#else
    std::vector<profiling::MicroBenchmark> benches;
#endif
    if (auto f = a.value("filter")) {
        std::regex re;
        try {
            re = std::regex(*f);
        } catch (const std::regex_error& e) {
            throw UsageError("--filter is not a valid regex: " + std::string(e.what()));
        }
        std::erase_if(benches, [&](const profiling::MicroBenchmark& b) { return !std::regex_search(b.name, re); });
    }
    if (a.flag("list")) {
        for (const auto& b : benches) *ctx.out << b.name << "\n";
        return kExitOk;
    }
    if (benches.empty()) {
        *ctx.err << "halo: no micro benchmarks selected (this build has " +
                        std::string(HALO_CLI_HAVE_CPU ? "the CPU reference kernels; check --filter" : "no kernels") +
                        ")\n";
        return kExitUsage;
    }
    profiling::SuiteRequest req;
    req.kind = profiling::SuiteKind::Micro;
    profiling::MicroSuiteConfig mc;
    mc.identity = identity(a, std::nullopt, "cpu");
    mc.benchmarks = std::move(benches);
    if (auto v = a.u64("warmup", 0, 100000)) mc.warmup = static_cast<unsigned>(*v);
    if (auto v = a.u64("iterations", 1, 1000000)) mc.iterations = static_cast<unsigned>(*v);
    mc.allow_nonconformant = a.flag("allow-nonconformant");
    mc.environment = environment(a, ctx);
    req.micro = std::move(mc);
    auto art = profiling::run_suite(req);
    stamp_invocation(art, args, ctx);
    return finish(art, a, ctx);
}

std::set<profiling::ModelMode> parse_modes(const std::string& s) {
    using M = profiling::ModelMode;
    static const std::map<std::string, M> names = {
        {"load_time", M::LoadTime},     {"prompt", M::PromptProcessing},         {"decode", M::Decode},
        {"mtp_decode", M::MtpDecode},   {"concurrency", M::Concurrency},         {"prefix_replay", M::PrefixReplay},
        {"cancellation_storm", M::CancellationStorm}};
    std::set<M> out;
    for (const auto& n : split_list(s, "--modes")) {
        const auto it = names.find(n);
        if (it == names.end()) throw UsageError("unknown mode '" + n + "' in --modes");
        out.insert(it->second);
    }
    return out;
}

/// Resolves runtime config (positional model allowed) and creates the engine.
struct EngineSetup {
    runtime::EngineConfig config;
    std::unique_ptr<runtime::Engine> engine;
};

std::optional<EngineSetup> make_engine(ParsedArgs& a, Context& ctx) {
    if (a.positionals.size() > 1) throw UsageError("expected at most one model file");
    if (a.positionals.size() == 1) {
        if (a.has("model")) throw UsageError("give the model as an argument or --model, not both");
        a.values_["model"] = {a.positionals[0]};
    }
    const ResolvedConfig cfg = resolve_config(a, ctx.env, runtime_keys());
    EngineSetup s{engine_config(cfg), nullptr};
    if (!a.has("host-label")) throw UsageError("--host-label is required (D-001: every record says where it was measured)");
    if (!ctx.engine_factory) {
        *ctx.err << "halo: " << kRuntimeNotBuilt << "\n";
        return std::nullopt;
    }
    s.engine = ctx.engine_factory(s.config);
    HALO_CHECK(s.engine != nullptr, ErrorCode::Config, "the engine factory returned no engine");
    return s;
}

int bench_model(const std::vector<std::string>& args, Context& ctx) {
    auto opts = join(common_opts(), config_options(runtime_keys()));
    opts = join(opts, {{"modes", true, 0, false, false, "comma list (default: all)"},
                       {"contexts", true, 0, false, false, "comma list (default 4096,32768,131072,262144)"},
                       {"concurrency", true, 0, false, false, "comma list (default 1,2,4,8)"},
                       {"prompt-tokens", true, 0, false, false, "prefill prompt length (default 512)"},
                       {"decode-tokens", true, 0, false, false, "decoded tokens (default 128)"},
                       {"bandwidth-gbps", true, 0, false, false, "measured memory bandwidth for efficiency"},
                       {"w-trunk-gb", true, 0, false, false, "trunk bytes per step (GB, incl. LM head)"},
                       {"w-mtp-gb", true, 0, false, false, "MTP block bytes (GB)"},
                       {"w-head-gb", true, 0, false, false, "LM head bytes (GB)"},
                       {"draft", true, 0, false, false, "MTP draft length n for efficiency"}});
    ParsedArgs a =
        parse_args(args, opts, "usage: halo bench model <model.gguf> --host-label L [options]\n" + options_help(opts));
    profiling::ModelSuiteConfig mc;
    if (auto v = a.value("modes")) mc.modes = parse_modes(*v);
    if (auto v = a.value("contexts")) mc.contexts = u64_list(*v, "--contexts", 1, 1u << 22);
    if (auto v = a.value("concurrency")) {
        mc.concurrency.clear();
        for (const auto x : u64_list(*v, "--concurrency", 1, 256)) mc.concurrency.push_back(static_cast<std::uint32_t>(x));
    }
    if (auto v = a.u64("prompt-tokens", 1, 1u << 22)) mc.prompt_tokens = *v;
    if (auto v = a.u64("decode-tokens", 1, 1u << 20)) mc.decode_tokens = *v;
    if (auto v = a.u64("warmup", 0, 1000)) mc.warmup = static_cast<unsigned>(*v);
    if (auto v = a.u64("repetitions", 1, 1000)) mc.repetitions = static_cast<unsigned>(*v);
    if (auto bw = a.number("bandwidth-gbps", 0.001, 100000.0)) {
        profiling::EfficiencyInputs e;
        e.measured_bandwidth_gbps = *bw;
        const auto trunk = a.number("w-trunk-gb", 0.0, 10000.0);
        if (!trunk) throw UsageError("--bandwidth-gbps needs --w-trunk-gb");
        e.byte_model.w_trunk_gb = *trunk;
        e.byte_model.w_mtp_gb = a.number("w-mtp-gb", 0.0, 10000.0).value_or(0.0);
        e.byte_model.w_head_gb = a.number("w-head-gb", 0.0, 10000.0).value_or(0.0);
        e.byte_model.draft_tokens = static_cast<std::uint32_t>(a.u64("draft", 0, 16).value_or(0));
        mc.efficiency = e;
    }
    mc.allow_nonconformant = a.flag("allow-nonconformant");
    auto setup = make_engine(a, ctx);
    if (!setup) return kExitUsage;
    mc.identity = identity(a, setup->config.model_path, setup->config.backend);
    mc.environment = environment(a, ctx);
    const auto factory = ctx.engine_factory;
    const auto ec = setup->config;
    mc.engine_factory = [factory, ec] { return factory(ec); };
    profiling::SuiteRequest req;
    req.kind = profiling::SuiteKind::Model;
    req.model = std::move(mc);
    req.engine = setup->engine.get();
    auto art = profiling::run_suite(req);
    stamp_invocation(art, args, ctx);
    return finish(art, a, ctx);
}

int bench_system(const std::vector<std::string>& args, Context& ctx) {
    auto opts = join(common_opts(), config_options(runtime_keys()));
    opts = join(opts, {{"workload", true, 'w', false, false, "workload JSON (bench/workloads/*.json)"},
                       {"agents", true, 0, false, false, "concurrent agents (default 4)"}});
    ParsedArgs a = parse_args(
        args, opts, "usage: halo bench system <model.gguf> --workload F --host-label L [options]\n" + options_help(opts));
    const auto wl = a.value("workload");
    if (!wl) throw UsageError("--workload is required");
    profiling::SystemSuiteConfig sc;
    sc.workload = profiling::load_workload(*wl);
    if (auto v = a.u64("agents", 1, profiling::kMaxAgents)) sc.agents = static_cast<std::uint32_t>(*v);
    if (auto v = a.u64("warmup", 0, 1000)) sc.warmup = static_cast<unsigned>(*v);
    if (auto v = a.u64("repetitions", 1, 1000)) sc.repetitions = static_cast<unsigned>(*v);
    sc.allow_nonconformant = a.flag("allow-nonconformant");
    auto setup = make_engine(a, ctx);
    if (!setup) return kExitUsage;
    sc.identity = identity(a, setup->config.model_path, setup->config.backend);
    sc.environment = environment(a, ctx);
    sc.encoder = engine_encoder(*setup->engine);
    profiling::SuiteRequest req;
    req.kind = profiling::SuiteKind::System;
    req.system = std::move(sc);
    req.engine = setup->engine.get();
    auto art = profiling::run_suite(req);
    stamp_invocation(art, args, ctx);
    return finish(art, a, ctx);
}

}  // namespace

int cmd_bench(const std::vector<std::string>& args, Context& ctx) {
    const std::string usage =
        "usage: halo bench micro|model|system [options]   (see docs/benchmarks.md; --help per suite)\n";
    if (args.empty()) throw UsageError(usage);
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (args[0] == "micro") return bench_micro(rest, ctx);
    if (args[0] == "model") return bench_model(rest, ctx);
    if (args[0] == "system") return bench_system(rest, ctx);
    throw UsageError("unknown bench suite '" + args[0] + "'\n" + usage);
}

#endif

}  // namespace halo::cli
