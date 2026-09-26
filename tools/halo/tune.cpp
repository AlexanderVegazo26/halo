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
#include <set>
#include <thread>

#include <nlohmann/json.hpp>

#include "args.h"
#include "commands.h"
#include "halo/core/error.h"
#include "fsutil.h"

#if HALO_CLI_HAVE_AUTOTUNE && HALO_CLI_HAVE_RUNTIME
#include "halo/autotune/cost_model.h"
#include "halo/autotune/db.h"
#include "halo/autotune/lookup.h"
#include "halo/autotune/profile_key.h"
#include "halo/autotune/tuner.h"
#include "halo/hardware/bandwidth.h"
#include "halo/model/inspect.h"
#include "halo/model/model.h"
#include "halo/profiling/hw_state.h"
#include "halo/runtime/profile.h"
#endif
#if HALO_CLI_HAVE_AUTOTUNE_CPU
#include "halo/autotune/cpu_ops.h"
#endif

namespace halo::cli {

#if !HALO_CLI_HAVE_AUTOTUNE || !HALO_CLI_HAVE_RUNTIME

int cmd_tune(const std::vector<std::string>&, Context& ctx) {
    *ctx.err << "halo tune: the autotuner is not part of this build (needs halo_autotune and halo_runtime, whose "
                "profile.h defines the keys the engine looks up)\n";
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
        {"isa-target", true, 0, false, false, "ISA label of the key (default: CPU ISA of this process; serve/run must use the same)"},
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
        {"gdn-tokens", true, 0, false, false, "must be 512: the engine looks the GDN tunable up at T=512"},
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
    autotune::ProfileKey key;
    std::vector<autotune::OpKey> op_keys;                        ///< the engine's keys
    std::vector<std::vector<autotune::Candidate>> op_candidates;  ///< the engine's candidate lists
    bool with_tunables = true;  ///< --list needs keys only (no measurement buffers)
    std::vector<std::unique_ptr<autotune::TunableOp>> ops;
    std::vector<std::string> notes;
};

Setup make_setup(const ParsedArgs& a, Context& ctx, bool with_tunables) {
    Setup s;
    s.with_tunables = with_tunables;
    const auto model = a.value("model");
    if (!model) throw UsageError("--model is required (its hash is part of the profile key)");
    if (!a.has("host-label")) throw UsageError("--host-label is required (D-001: every measurement says where it ran)");
    const auto power = a.value("power-mode");
    if (!power) throw UsageError("--power-mode is required (the platform performance mode is part of the profile key)");
    const std::string backend = a.value("backend").value_or("cpu");
    if (backend != "cpu") {
        throw UsageError("--backend " + backend + ": v0.2 tunes the CPU reference kernels only (GPU TunableOps are not wired)");
    }
    if (auto t = a.u64("gdn-tokens", 1, 1u << 16); t && *t != runtime::kGdnTuneTokens) {
        throw UsageError(std::format("--gdn-tokens {}: the engine looks the GDN tunable up at T={}, so a winner for any "
                                     "other length would never be applied",
                                     *t, runtime::kGdnTuneTokens));
    }
    s.db = a.value("db") ? std::filesystem::path(*a.value("db")) : default_profile_db(ctx.env);
    HALO_CHECK(!s.db.empty(), ErrorCode::Config,
               "no --db given and neither HALO_PROFILE_DB, XDG_CACHE_HOME nor HOME is set");

    // The profile key and the operator keys come from the runtime (include/halo/runtime/
    // profile.h), the same functions the Engine uses to look winners up: a key built any
    // other way could differ in one field and the winner would silently never apply.
    runtime::EngineConfig ec;
    ec.model_path = *model;
    ec.mtp_path = a.value("mtp");
    ec.platform_power_mode = *power;
    ec.isa_target = a.value("isa-target");
    // S-21: the key hashes the file and the shapes come from its header; refuse to mix two
    // different files if it is replaced between the two opens.
    const FileIdentity before = file_identity(*model);
    s.key = runtime::engine_profile_key(ec, a.value("root").value_or("/"));
    auto nm = model::NormalizedModel::load(*model, model::GgufMode::HeaderOnly);
    HALO_CHECK(file_identity(*model) == before, ErrorCode::Io,
               "{} changed while it was being hashed; run halo tune again", *model);
    const model::Qwen35HParams& hp = nm.hparams();

    // Candidate values: the defaults are a small subset of what the engine can run; an
    // explicit list must stay inside it (select_kernel rejects anything else at lookup).
    const auto engine_values = [](const std::vector<autotune::Candidate>& cs, const char* name) {
        std::set<std::int64_t> v;
        for (const auto& c : cs) v.insert(c.get(name));
        return v;
    };
    const std::set<std::int64_t> engine_threads = engine_values(runtime::matmul_candidates(), "threads");
    const std::set<std::int64_t> engine_chunks = engine_values(runtime::gdn_candidates(), "chunk");
    const auto checked = [](std::vector<unsigned> v, const std::set<std::int64_t>& allowed, const char* flag) {
        for (const unsigned x : v) {
            if (!allowed.contains(x)) {
                throw UsageError(std::format("--{} {}: not a value the engine can run (so a winner could never apply)",
                                             flag, x));
            }
        }
        return v;
    };
    std::vector<unsigned> threads = a.value("threads") ? unsigned_list(*a.value("threads"), "--threads", 1, 1024)
                                                       : default_threads();
    threads = checked(threads, engine_threads, "threads");
    std::vector<unsigned> chunks = a.value("chunks") ? unsigned_list(*a.value("chunks"), "--chunks", 1, 1024)
                                                     : std::vector<unsigned>{16, 32, 64, 128};
    chunks = checked(chunks, engine_chunks, "chunks");
    std::vector<std::string> ops = {"MATMUL", "GATED_DELTANET"};
    if (auto v = a.value("ops")) ops = split_list(*v, "--ops");
    for (const auto& op : ops) {
        if (op == "MATMUL") {
            s.op_keys.push_back(runtime::matmul_op_key(hp));
            s.op_candidates.push_back(runtime::matmul_candidates());
        } else if (op == "GATED_DELTANET") {
            s.op_keys.push_back(runtime::gdn_op_key(hp));
            s.op_candidates.push_back(runtime::gdn_candidates());
        } else {
            throw UsageError("--ops: unknown operator '" + op + "' (v0.2: MATMUL, GATED_DELTANET)");
        }
    }
    if (!s.with_tunables) return s;
#if HALO_CLI_HAVE_AUTOTUNE_CPU
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (ops[i] == "MATMUL") {
            // The FFN up/gate projection the engine keys on: y[1, n_ff] = x[1, n_embd] W^T.
            s.ops.push_back(std::make_unique<autotune::CpuMatmulTunable>(1, hp.n_embd, hp.n_ff, threads));
        } else {
            const cpu::GdnDims dims{hp.gdn_n_k_heads, hp.gdn_n_v_heads, hp.gdn_head_k_dim, hp.gdn_head_v_dim,
                                    cpu::GdnHeadMapping::Tiled};
            s.ops.push_back(
                std::make_unique<autotune::CpuGdnChunkedTunable>(dims, runtime::kGdnTuneTokens, chunks, threads));
        }
        // The tunable's key must be the engine's key, byte for byte.
        HALO_CHECK(s.ops.back()->key() == s.op_keys[i], ErrorCode::Config,
                   "internal: tunable key {} {} differs from the engine's {} {}", s.ops.back()->key().family,
                   s.ops.back()->key().shape, s.op_keys[i].family, s.op_keys[i].shape);
    }
#else
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
    Setup s = make_setup(a, ctx, false);
    std::ostream& out = *ctx.out;
    refuse_symlink(s.db, "profile database");  // S-21
    const auto lookup = autotune::ProfileLookup::open(s.db);  // read-only
    out << std::format("profile DB: {} ({} winning configurations, opened read-only)\n", s.db.string(), lookup.size());
    print_key(out, s.key);
    json j = {{"db", s.db.string()}, {"key", s.key}, {"winners", lookup.size()}, {"ops", json::array()}};
    for (std::size_t i = 0; i < s.op_keys.size(); ++i) {
        const autotune::OpKey& ok = s.op_keys[i];
        // Exactly the engine's lookup: its key, its op key, its candidate list.
        const auto res = autotune::select_kernel(lookup, s.key, ok, "cpu", s.op_candidates[i], nullptr, {});
        const std::string step = std::string(autotune::to_string(res.source));
        out << std::format("  {:<16} {:<44} {:<5} step {:<10} {}", ok.family, ok.shape, std::string("cpu"), step,
                           res.candidate ? res.candidate->to_string() : "-");
        if (res.candidate && res.source != autotune::SelectionSource::Heuristic) {
            out << std::format("  (median {:.0f} ns, {}, {})", res.value, res.strategy, res.created_at);
        }
        out << "\n";
        for (const auto& d : res.differing) out << "      compatible match; differs in " << d << "\n";
        for (const auto& why : res.rejections) out << "      rejected: " << why << "\n";
        if (res.source == autotune::SelectionSource::None) out << "      no profile: run `halo tune` for this key\n";
        j["ops"].push_back({{"family", ok.family},
                            {"shape", ok.shape},
                            {"backend", std::string("cpu")},
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
            write_file_no_follow(*rep, j.dump(2) + "\n");  // S-21: failures are errors
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

    Setup s = make_setup(a, ctx, true);
    std::ostream& out = *ctx.out;
    std::ostream& err = *ctx.err;
    if (s.ops.empty()) {
        for (const auto& n : s.notes) err << "halo tune: " << n << "\n";
        return kExitUsage;
    }
    std::filesystem::create_directories(s.db.parent_path().empty() ? "." : s.db.parent_path());
    refuse_symlink(s.db, "profile database");  // S-21
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
            write_file_no_follow(*p, j.dump(2) + "\n");  // S-21
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
