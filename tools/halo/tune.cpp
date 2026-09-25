// `halo tune`: offline autotuning into the profile database (TRD §58, §64; docs/benchmarks.md
// "Intended halo tune flags"), and `halo tune --list` / `--show`, which opens the database
// read-only and reports, per operator, what the runtime lookup (TRD §56) would select.
//
// Flow (TRD §64): hardware state -> host tier bandwidth (recorded) -> model/pack hashes ->
// profile key -> TunableOps for the model's operator shapes -> tune() -> TuneReport JSON.
// v0.2 tunes the CPU reference TunableOps only (halo_autotune_cpu); GPU TunableOps belong
// to the backends and are not wired yet.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

#include <nlohmann/json.hpp>

#include "args.h"
#include "commands.h"
#include "halo/core/error.h"
#include "hashing.h"

#if HALO_CLI_HAVE_AUTOTUNE
#include "halo/autotune/cost_model.h"
#include "halo/autotune/db.h"
#include "halo/autotune/lookup.h"
#include "halo/autotune/profile_key.h"
#include "halo/autotune/tuner.h"
#include "halo/hardware/bandwidth.h"
#include "halo/model/inspect.h"
#include "halo/model/model.h"
#include "halo/profiling/hw_state.h"
#endif
#if HALO_CLI_HAVE_AUTOTUNE_CPU
#include "halo/autotune/cpu_ops.h"
#endif

namespace halo::cli {

#if !HALO_CLI_HAVE_AUTOTUNE

int cmd_tune(const std::vector<std::string>&, Context& ctx) {
    *ctx.err << "halo tune: the autotuner is not part of this build (halo_autotune not built)\n";
    return kExitUsage;
}

#else

namespace {

using nlohmann::json;

const std::vector<OptionSpec>& tune_opts() {
    static const std::vector<OptionSpec> o = {
        {"model", true, 0, false, false, "trunk GGUF (hashed into the profile key; shapes from its header)"},
        {"mtp", true, 0, false, false, "separate MTP GGUF (part of PACK_ID)"},
        {"host-label", true, 0, false, false, "where this runs, e.g. dev-host or evo-x2 (required, D-001)"},
        {"power-mode", true, 0, false, false, "platform (BIOS/EC) performance mode (required; part of the key)"},
        {"isa-target", true, 0, false, false, "ISA label of the key (default: CPU ISA of this process)"},
        {"db", true, 0, false, false, "profile DB (default $HALO_PROFILE_DB or ~/.cache/halo/profiles.db)"},
        {"backend", true, 0, false, false, "cpu (v0.2 tunes the CPU reference kernels only)"},
        {"strategy", true, 0, false, false, "exhaustive | heuristic | grid | random (default: see below)"},
        {"gflops-per-thread", true, 0, false, false, "cost-model calibration; enables --strategy heuristic"},
        {"seed", true, 0, false, false, "random strategy seed"},
        {"random-budget", true, 0, false, false, "random strategy: candidates measured"},
        {"grid-stride", true, 0, false, false, "grid strategy stride"},
        {"heuristic-keep", true, 0, false, false, "heuristic strategy: candidates kept after the cost model"},
        {"warmup", true, 0, false, false, "warm-up runs per candidate (default 5)"},
        {"iterations", true, 0, false, false, "measured runs per candidate (default 20)"},
        {"max-cv", true, 0, false, false, "reject candidates with cv above this (default 0.05)"},
        {"ops", true, 0, false, false, "comma list: MATMUL,GATED_DELTANET (default both)"},
        {"threads", true, 0, false, false, "comma list of thread counts (default 1,2,4,.. up to the core count)"},
        {"tokens", true, 0, false, false, "matmul rows T (default 1: decode)"},
        {"gdn-tokens", true, 0, false, false, "chunked GDN prefill tokens (default 512)"},
        {"chunks", true, 0, false, false, "comma list of GDN chunk sizes (default 16,32,64,128)"},
        {"no-bandwidth", false, 0, false, false, "skip the host bandwidth measurement"},
        {"report", true, 'o', false, false, "TuneReport JSON path ('-' = stdout)"},
        {"list", false, 0, false, false, "read-only: show what the runtime lookup selects per operator"},
        {"show", false, 0, false, false, "same as --list"},
        {"allow-nonconformant", false, 0, false, false, "allow fewer runs than the TRD §50 minimum (smoke runs)"},
        {"root", true, 0, false, true, "filesystem root for hardware discovery (tests)"},
    };
    return o;
}

std::filesystem::path default_db(const Context& ctx) {
    const auto get = [&](const char* k) -> std::string {
        const auto it = ctx.env.find(k);
        return it == ctx.env.end() ? std::string{} : it->second;
    };
    if (auto p = get("HALO_PROFILE_DB"); !p.empty()) return p;
    if (auto x = get("XDG_CACHE_HOME"); !x.empty()) return std::filesystem::path(x) / "halo/profiles.db";
    const auto home = get("HOME");
    HALO_CHECK(!home.empty(), ErrorCode::Config, "no --db given and neither HALO_PROFILE_DB, XDG_CACHE_HOME nor HOME is set");
    return std::filesystem::path(home) / ".cache/halo/profiles.db";
}

std::string cpu_isa_label() {
    const auto s = hardware::runtime_simd_flags();
#if defined(__x86_64__)
    if (s && s->avx512f) return "x86-64-avx512";
    if (s && s->avx2) return "x86-64-avx2";
    return "x86-64";
#else
    (void)s;
    return "cpu";
#endif
}

std::vector<unsigned> default_threads() {
    const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
    std::vector<unsigned> t;
    for (unsigned n = 1; n <= hw; n *= 2) t.push_back(n);
    if (t.back() != hw) t.push_back(hw);
    return t;
}

std::vector<unsigned> unsigned_list(const std::string& s, const std::string& what, std::uint64_t lo, std::uint64_t hi) {
    std::vector<unsigned> out;
    for (const auto& item : split_list(s, what)) out.push_back(static_cast<unsigned>(parse_u64(item, what, lo, hi)));
    return out;
}

struct Setup {
    std::filesystem::path db;
    profiling::HardwareState hw;
    autotune::ProfileKey key;
    std::vector<std::unique_ptr<autotune::TunableOp>> ops;
    std::vector<std::string> notes;
};

Setup make_setup(const ParsedArgs& a, Context& ctx) {
    Setup s;
    const auto model = a.value("model");
    if (!model) throw UsageError("--model is required (its hash is part of the profile key)");
    if (!a.has("host-label")) throw UsageError("--host-label is required (D-001: every measurement says where it ran)");
    const auto power = a.value("power-mode");
    if (!power) throw UsageError("--power-mode is required (the platform performance mode is part of the profile key)");
    const std::string backend = a.value("backend").value_or("cpu");
    if (backend != "cpu") {
        throw UsageError("--backend " + backend + ": v0.2 tunes the CPU reference kernels only (GPU TunableOps are not wired)");
    }
    s.db = a.value("db") ? std::filesystem::path(*a.value("db")) : default_db(ctx);

    // Hardware state and profile key (TRD §57).
    s.hw = profiling::capture_hardware_state(discovery_options(a.value("root").value_or("/"), ctx.env), *power);
    const std::string trunk_sha = sha256_file_hex(*model);
    std::optional<std::string> mtp_sha;
    if (auto m = a.value("mtp")) mtp_sha = sha256_file_hex(*m);
    s.key = autotune::make_profile_key(s.hw, trunk_sha, pack_id(trunk_sha, mtp_sha),
                                       a.value("isa-target").value_or(cpu_isa_label()));

    // Operator shapes from the model header.
    auto nm = model::NormalizedModel::load(*model, model::GgufMode::HeaderOnly);
    const model::InspectReport r = nm.inspect();
    std::vector<std::string> ops = {"MATMUL", "GATED_DELTANET"};
    if (auto v = a.value("ops")) ops = split_list(*v, "--ops");
    const auto threads = a.value("threads") ? unsigned_list(*a.value("threads"), "--threads", 1, 1024) : default_threads();
#if HALO_CLI_HAVE_AUTOTUNE_CPU
    for (const auto& op : ops) {
        if (op == "MATMUL") {
            // The FFN up/gate projection: y[T, n_ff] = x[T, n_embd] W^T (dense f32 weights).
            const auto t = a.u64("tokens", 1, 1u << 16).value_or(1);
            s.ops.push_back(std::make_unique<autotune::CpuMatmulTunable>(t, r.n_embd, r.n_ff, threads));
        } else if (op == "GATED_DELTANET") {
            const auto t = a.u64("gdn-tokens", 1, 1u << 16).value_or(512);
            std::vector<unsigned> chunks = a.value("chunks") ? unsigned_list(*a.value("chunks"), "--chunks", 1, 1024)
                                                             : std::vector<unsigned>{16, 32, 64, 128};
            std::erase_if(chunks, [&](unsigned c) { return c > t; });
            if (chunks.empty()) throw UsageError("--chunks: no chunk size <= --gdn-tokens");
            const cpu::GdnDims dims{r.gdn_n_k_heads, r.gdn_n_v_heads, r.gdn_head_k_dim, r.gdn_head_v_dim,
                                    cpu::GdnHeadMapping::Tiled};
            s.ops.push_back(std::make_unique<autotune::CpuGdnChunkedTunable>(dims, t, chunks, threads));
        } else {
            throw UsageError("--ops: unknown operator '" + op + "' (v0.2: MATMUL, GATED_DELTANET)");
        }
    }
#else
    (void)threads;
    (void)r;
    s.notes.push_back("the CPU TunableOps (halo_autotune_cpu) are not part of this build: nothing to tune");
#endif
    return s;
}

void print_key(std::ostream& out, const autotune::ProfileKey& k) {
    out << std::format("profile key: gpu {} {}, power mode {}, isa {}, model {}..., pack {}...\n", k.gpu_device,
                       k.gpu_arch.empty() ? "(cpu-only)" : k.gpu_arch, k.power_mode, k.isa_target,
                       k.model_hash.substr(0, 12), k.pack_id.substr(0, 12));
}

int tune_list(const ParsedArgs& a, Context& ctx) {
    Setup s = make_setup(a, ctx);
    std::ostream& out = *ctx.out;
    const auto lookup = autotune::ProfileLookup::open(s.db);  // read-only
    out << std::format("profile DB: {} ({} winning configurations, opened read-only)\n", s.db.string(), lookup.size());
    print_key(out, s.key);
    json j = {{"db", s.db.string()}, {"key", s.key}, {"winners", lookup.size()}, {"ops", json::array()}};
    for (const auto& op : s.ops) {
        const auto res = autotune::select_kernel(lookup, s.key, op->key(), op->backend(), op->candidates(), nullptr, {});
        const std::string step = std::string(autotune::to_string(res.source));
        out << std::format("  {:<16} {:<44} {:<5} step {:<10} {}", op->key().family, op->key().shape, op->backend(), step,
                           res.candidate ? res.candidate->to_string() : "-");
        if (res.candidate && res.source != autotune::SelectionSource::Heuristic) {
            out << std::format("  (median {:.0f} ns, {}, {})", res.value, res.strategy, res.created_at);
        }
        out << "\n";
        for (const auto& d : res.differing) out << "      compatible match; differs in " << d << "\n";
        for (const auto& why : res.rejections) out << "      rejected: " << why << "\n";
        if (res.source == autotune::SelectionSource::None) out << "      no profile: run `halo tune` for this key\n";
        j["ops"].push_back({{"family", op->key().family},
                            {"shape", op->key().shape},
                            {"backend", op->backend()},
                            {"step", step},
                            {"candidate", res.candidate ? json(res.candidate->to_string()) : json(nullptr)},
                            {"median_ns", res.candidate ? json(res.value) : json(nullptr)},
                            {"differing", res.differing},
                            {"rejections", res.rejections}});
    }
    for (const auto& n : s.notes) out << "note: " << n << "\n";
    if (auto rep = a.value("report")) {
        if (*rep == "-") {
            out << j.dump(2) << "\n";
        } else {
            std::ofstream(*rep) << j.dump(2) << "\n";
        }
    }
    return kExitOk;
}

int tune_run(const ParsedArgs& a, Context& ctx) {
    autotune::TuneOptions o;
    const auto gflops = a.number("gflops-per-thread", 1e-6, 1e6);
    const std::string strat = a.value("strategy").value_or(gflops ? "heuristic" : "exhaustive");
    if (strat == "exhaustive") {
        o.strategy = autotune::Strategy::Exhaustive;
    } else if (strat == "heuristic") {
        o.strategy = autotune::Strategy::Heuristic;
        if (!gflops) {
            throw UsageError("--strategy heuristic needs a cost-model calibration: pass --gflops-per-thread "
                             "(measured with a HALO kernel) or use --strategy exhaustive");
        }
        if (a.flag("no-bandwidth")) throw UsageError("--strategy heuristic needs the bandwidth measurement (drop --no-bandwidth)");
    } else if (strat == "grid") {
        o.strategy = autotune::Strategy::Grid;
    } else if (strat == "random") {
        o.strategy = autotune::Strategy::Random;
    } else {
        throw UsageError("--strategy must be exhaustive, heuristic, grid or random (bayesian is not implemented)");
    }
    if (auto v = a.u64("seed", 0, std::numeric_limits<std::uint64_t>::max())) o.seed = *v;
    if (auto v = a.u64("random-budget", 1, 1u << 20)) o.random_budget = *v;
    if (auto v = a.u64("grid-stride", 1, 1u << 20)) o.grid_stride = *v;
    if (auto v = a.u64("heuristic-keep", 1, 1u << 20)) o.heuristic_keep = *v;
    if (auto v = a.u64("warmup", 0, 1u << 20)) o.warmup = static_cast<unsigned>(*v);
    if (auto v = a.u64("iterations", 1, 1u << 20)) o.iterations = static_cast<unsigned>(*v);
    if (auto v = a.number("max-cv", 0.0, 1e12)) o.stability.max_cv = *v;
    o.allow_nonconformant = a.flag("allow-nonconformant");

    Setup s = make_setup(a, ctx);
    std::ostream& out = *ctx.out;
    std::ostream& err = *ctx.err;
    if (s.ops.empty()) {
        for (const auto& n : s.notes) err << "halo tune: " << n << "\n";
        return kExitUsage;
    }
    std::filesystem::create_directories(s.db.parent_path().empty() ? "." : s.db.parent_path());
    autotune::ProfileDb db = autotune::ProfileDb::open(s.db);
    print_key(out, s.key);
    std::vector<hardware::TierBandwidth> bws;
    if (!a.flag("no-bandwidth")) {
        hardware::BandwidthOptions bo;
        bo.label = *a.value("host-label");
        bo.buffer_bytes = 64ULL << 20;
        bo.allow_nonconformant = o.allow_nonconformant;
        if (o.allow_nonconformant) {
            bo.warmup = std::min(bo.warmup, o.warmup);
            bo.iterations = std::min(bo.iterations, o.iterations);
        }
        bws.push_back(hardware::measure_host_bandwidth(bo));
        db.record_tier_bandwidth(s.key, bws.back());
        out << std::format("host bandwidth: read median {:.1f} GB/s ({} threads), recorded\n", bws.back().read_gbps.median,
                           bws.back().threads);
    }
    if (gflops) {
        o.cost_model = autotune::CostModel::from_measurements(bws, *gflops);
    }
    if (!a.has("strategy") && !gflops) {
        out << "strategy: exhaustive (no cost-model calibration available; pass --gflops-per-thread to use the "
               "heuristic strategy)\n";
    } else {
        out << "strategy: " << autotune::to_string(o.strategy) << "\n";
    }
    std::vector<autotune::TunableOp*> ptrs;
    for (auto& op : s.ops) ptrs.push_back(op.get());
    const autotune::TuneReport rep = autotune::tune(db, s.key, ptrs, o);
    for (const auto& r : rep.ops) {
        std::size_t measured = 0, rejected = 0;
        for (const auto& c : r.candidates) {
            measured += c.measured ? 1 : 0;
            rejected += c.rejected.empty() ? 0 : 1;
        }
        out << std::format("  {:<16} {:<44} {} candidates measured, {} rejected -> {}{}\n", r.key.family, r.key.shape,
                           measured, rejected,
                           r.winner ? std::format("{} (median {:.0f} ns)", r.winner->to_string(), r.winner_median_ns)
                                    : std::string("no winner"),
                           r.persisted ? ", persisted" : (r.note.empty() ? "" : " (" + r.note + ")"));
    }
    out << "profile DB: " << s.db.string() << "\n";
    json j = rep;
    j["host_label"] = *a.value("host-label");
    j["db"] = s.db.string();
    if (auto p = a.value("report")) {
        if (*p == "-") {
            out << j.dump(2) << "\n";
        } else {
            std::ofstream f(*p);
            HALO_CHECK(f.good(), ErrorCode::Io, "cannot write {}", *p);
            f << j.dump(2) << "\n";
        }
    }
    bool any_winner = false;
    for (const auto& r : rep.ops) any_winner = any_winner || r.winner.has_value();
    return any_winner ? kExitOk : kExitFailure;
}

}  // namespace

int cmd_tune(const std::vector<std::string>& args, Context& ctx) {
    const std::string usage =
        "usage: halo tune --model M --host-label L --power-mode P [options]\n"
        "       halo tune --list --model M --host-label L --power-mode P [--db F]\n"
        "Without --strategy the tuner measures every candidate (exhaustive) unless --gflops-per-thread\n"
        "supplies a cost-model calibration, in which case it uses the heuristic prefilter.\n" +
        options_help(tune_opts());
    const ParsedArgs a = parse_args(args, tune_opts(), usage);
    if (!a.positionals.empty()) throw UsageError("tune takes no positional arguments (use --model)\n" + usage);
    if (a.flag("list") || a.flag("show")) return tune_list(a, ctx);
    return tune_run(a, ctx);
}

#endif

}  // namespace halo::cli
