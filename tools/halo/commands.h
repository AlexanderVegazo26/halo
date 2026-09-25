#pragma once
// Subcommand entry points (internal to halo_cli). Each returns an exit code and throws
// UsageError / halo::Error, which run_cli turns into messages.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "cli.h"
#include "halo/hardware/hardware.h"

namespace halo::cli {

int cmd_inspect(const std::vector<std::string>& args, Context& ctx);
int cmd_devices(const std::vector<std::string>& args, Context& ctx);
int cmd_tokenize(const std::vector<std::string>& args, Context& ctx);
int cmd_template(const std::vector<std::string>& args, Context& ctx);
int cmd_run(const std::vector<std::string>& args, Context& ctx);
int cmd_serve(const std::vector<std::string>& args, Context& ctx);
int cmd_bench(const std::vector<std::string>& args, Context& ctx);

/// "12.3 GiB" style (binary units).
[[nodiscard]] std::string human_bytes(std::uint64_t b);

/// Discovery for the hidden --root option: the real root ("/") also consults the process
/// environment (Vulkan ICD variables); a fixture root uses none.
[[nodiscard]] hardware::DiscoveryOptions discovery_options(const std::string& root,
                                                           const std::map<std::string, std::string>& env);

/// Message printed (exit 2) when a command needs the runtime and this build has none.
inline constexpr const char* kRuntimeNotBuilt =
    "runtime not built: this halo binary does not include halo_runtime (the inference engine), "
    "so it cannot load a model for generation";

}  // namespace halo::cli
