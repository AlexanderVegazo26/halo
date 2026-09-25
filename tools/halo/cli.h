#pragma once
// The `halo` command-line tool (PRD §14, TRD §26): one binary, subcommands
//   inspect   FR-003 compatibility report + memory estimate (header-only files work)
//   devices   hardware discovery + deployment checklist (--verify)
//   serve     HTTP API server (needs the runtime)
//   run       one generation streamed to stdout (needs the runtime)
//   tokenize  tokenize text with a model's tokenizer
//   template  render a chat with a model's chat template
//   bench     benchmark suites (halo_profiling); micro runs without a model
//   tune      not implemented in v0.2 (exit 2)
//
// The command logic lives in a library (halo_cli) so tests drive it in-process: `run_cli`
// takes the arguments, the output streams, the environment and an engine factory instead
// of touching globals. main.cpp supplies the real ones.
//
// Exit codes: 0 success; 1 runtime failure (typed error printed); 2 usage error, a feature
// that is not implemented in v0.2, or a component that is not part of this build (e.g.
// "runtime not built"); `devices --verify` returns 1 when a check fails.

#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "halo/runtime/engine.h"

namespace halo::api { class ApiServer; }

namespace halo::cli {

inline constexpr int kExitOk = 0;
inline constexpr int kExitFailure = 1;
inline constexpr int kExitUsage = 2;

using EngineFactory = std::function<std::unique_ptr<runtime::Engine>(const runtime::EngineConfig&)>;

struct Context {
    std::ostream* out = nullptr;
    std::ostream* err = nullptr;
    std::istream* in = nullptr;                  ///< stdin (tokenize/template read "-")
    std::map<std::string, std::string> env;      ///< process environment (HALO_* consulted)
    /// Creates the engine for serve/run/bench model|system. Null = the runtime is not part
    /// of this build (those commands then exit 2 with "runtime not built").
    EngineFactory engine_factory;
    /// serve: called on the serving thread's behalf once the server accepts connections;
    /// receives a function that stops the server (main wires SIGINT/SIGTERM to it; tests
    /// call it). serve returns after the stop.
    std::function<void(api::ApiServer& server, std::function<void()> stop)> on_serving;
    std::vector<std::string> argv0;  ///< binary path (for bench invocation records)
};

/// Runs `halo <args...>` (args excludes the program name). Never throws.
[[nodiscard]] int run_cli(const std::vector<std::string>& args, Context& ctx);

/// The default engine factory: runtime::create_engine when halo_runtime is linked, else null.
[[nodiscard]] EngineFactory default_engine_factory();

/// True when this build links halo_runtime / halo_api / halo_profiling / halo_backend_cpu.
[[nodiscard]] bool have_runtime() noexcept;
[[nodiscard]] bool have_api() noexcept;
[[nodiscard]] bool have_profiling() noexcept;

}  // namespace halo::cli
