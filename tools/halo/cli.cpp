#include "cli.h"

#include <exception>
#include <iostream>

#include "args.h"
#include "commands.h"
#include "halo/core/error.h"

#if HALO_CLI_HAVE_RUNTIME
#include "halo/runtime/engine.h"
#endif

namespace halo::cli {

bool have_runtime() noexcept { return HALO_CLI_HAVE_RUNTIME != 0; }
bool have_api() noexcept { return HALO_CLI_HAVE_API != 0; }
bool have_profiling() noexcept { return HALO_CLI_HAVE_PROFILING != 0; }

EngineFactory default_engine_factory() {
#if HALO_CLI_HAVE_RUNTIME
    return [](const runtime::EngineConfig& c) { return runtime::create_engine(c); };
#else
    return nullptr;
#endif
}

namespace {

constexpr const char* kUsage =
    "usage: halo <command> [options]\n"
    "\n"
    "commands:\n"
    "  inspect <model.gguf>   compatibility report (FR-003) and memory estimate\n"
    "  devices                hardware discovery; --verify runs the deployment checklist\n"
    "  serve                  OpenAI / Anthropic HTTP API server (docs/api.md)\n"
    "  run <model.gguf> -p P  one generation streamed to stdout\n"
    "  tokenize <model.gguf>  tokenize text with the model's tokenizer\n"
    "  template <model.gguf>  render a chat with the model's chat template\n"
    "  bench micro|model|system   benchmark suites (docs/benchmarks.md)\n"
    "  tune                   autotuning (not implemented in v0.2)\n"
    "  version                print the version and the components in this build\n"
    "\n"
    "Configuration: command line > HALO_* environment > JSON config file (--config / HALO_CONFIG).\n"
    "Run 'halo <command> --help' for a command's options.\n";

bool wants_help(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--") return false;
        if (a == "--help" || a == "-h") return true;
    }
    return false;
}

int dispatch(const std::string& cmd, const std::vector<std::string>& rest, Context& ctx) {
    if (cmd == "inspect") return cmd_inspect(rest, ctx);
    if (cmd == "devices") return cmd_devices(rest, ctx);
    if (cmd == "serve") return cmd_serve(rest, ctx);
    if (cmd == "run") return cmd_run(rest, ctx);
    if (cmd == "tokenize") return cmd_tokenize(rest, ctx);
    if (cmd == "template") return cmd_template(rest, ctx);
    if (cmd == "bench" || cmd == "benchmark") return cmd_bench(rest, ctx);
    if (cmd == "tune") {
        *ctx.err << "halo tune: not implemented in v0.2\n";
        return kExitUsage;
    }
    if (cmd == "version" || cmd == "--version") {
        *ctx.out << "halo 0.2.0\n"
                 << "components: api " << (have_api() ? "yes" : "no") << ", runtime " << (have_runtime() ? "yes" : "no")
                 << ", profiling " << (have_profiling() ? "yes" : "no") << "\n";
        return kExitOk;
    }
    throw UsageError("unknown command '" + cmd + "'\n" + kUsage);
}

}  // namespace

int run_cli(const std::vector<std::string>& args, Context& ctx) {
    std::ostream& err = ctx.err != nullptr ? *ctx.err : std::cerr;
    if (ctx.out == nullptr) ctx.out = &std::cout;
    if (ctx.err == nullptr) ctx.err = &std::cerr;
    if (args.empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h") {
        (args.empty() ? err : *ctx.out) << kUsage;
        return args.empty() ? kExitUsage : kExitOk;
    }
    const std::string& cmd = args[0];
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    try {
        if (wants_help(rest)) {
            // Commands put their option list in the usage error for an unknown option; ask
            // for it by replacing --help with an option no command defines.
            std::vector<std::string> probe;
            for (const auto& r : rest) probe.push_back(r == "--help" || r == "-h" ? "--halo-show-usage" : r);
            try {
                return dispatch(cmd, probe, ctx);
            } catch (const UsageError& e) {
                std::string msg = e.what();
                if (const auto p = msg.find("usage:"); p != std::string::npos) msg = msg.substr(p);
                *ctx.out << msg << (msg.ends_with("\n") ? "" : "\n");
                return kExitOk;
            }
        }
        return dispatch(cmd, rest, ctx);
    } catch (const UsageError& e) {
        err << "halo " << cmd << ": " << e.what() << (std::string_view(e.what()).ends_with("\n") ? "" : "\n");
        return kExitUsage;
    } catch (const halo::Error& e) {
        err << "halo " << cmd << ": " << e.what() << "\n";
        return kExitFailure;
    } catch (const std::exception& e) {
        err << "halo " << cmd << ": internal error: " << e.what() << "\n";
        return kExitFailure;
    }
}

}  // namespace halo::cli
