#pragma once
// TRD §64 "the runtime consumes the profile on next launch": how the CPU Engine builds its
// profile key (TRD §57) and the operator keys of the CPU tunables it applies, so that a
// tuner writing the profile database and the engine reading it use the SAME strings.
//
// This is a cross-workstream contract. `halo tune` (tools/halo, WS-I) and any other
// producer must build keys with these functions (or with byte-identical logic): a key that
// differs in any must-match field, or an op key string that differs at all, means a tuned
// winner is silently never applied. The op keys are exactly those of
// autotune::CpuMatmulTunable / CpuGdnChunkedTunable (tested for equality).
//
//   MODEL_HASH  SHA-256 hex of the trunk GGUF file
//   PACK_ID     SHA-256 hex of "halo.pack/1\ntrunk=<trunk sha>\nmtp=<mtp sha or 'none'>\n"
//   ISA_TARGET  "x86-64-avx512" | "x86-64-avx2" | "x86-64" | "cpu" (CPU of this process)
//   hardware    profiling::capture_hardware_state(root, platform_power_mode)
//
// Hashing reads the whole file (a ~17 GB GGUF takes several seconds); results are cached
// per process by (canonical path, size, mtime). A persistent cache is future work.
//
// Tunables applied by the CPU Engine:
//   MATMUL          "T=1,K=<n_embd>,N=<n_ff>,w=f32"   candidate {threads}      -> ThreadPool size
//   GATED_DELTANET  "T=512,Hk=..,Hv=..,dk=..,dv=..,impl=chunked"  {chunk, threads} -> GDN chunk
// An explicit EngineConfig::threads overrides the profile's thread count.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "halo/autotune/profile_key.h"
#include "halo/autotune/types.h"
#include "halo/model/model.h"
#include "halo/runtime/engine.h"

namespace halo::runtime {

/// FIPS 180-4 SHA-256 as lowercase hex.
[[nodiscard]] std::string sha256_hex(std::string_view data);
/// Throws Error(Io) when the file cannot be read.
[[nodiscard]] std::string sha256_file_hex(const std::filesystem::path& path);
/// SHA-256 hex of "halo.pack/1\ntrunk=<trunk>\nmtp=<mtp or 'none'>\n".
[[nodiscard]] std::string pack_id(std::string_view trunk_sha256, const std::optional<std::string>& mtp_sha256);
/// ISA label of the CPU running this process.
[[nodiscard]] std::string cpu_isa_label();

/// The profile key the CPU Engine looks up for `cfg` (hashes the model files; cached).
/// Throws Error(Config) without cfg.platform_power_mode (make_profile_key refuses unknown
/// power modes, TRD §49 / §57).
[[nodiscard]] autotune::ProfileKey engine_profile_key(const EngineConfig& cfg, const std::string& hardware_root = "/");

inline constexpr std::size_t kGdnTuneTokens = 512;  ///< prefill length the GDN tunable is keyed on
[[nodiscard]] autotune::OpKey matmul_op_key(const model::Qwen35HParams& hp);
[[nodiscard]] autotune::OpKey gdn_op_key(const model::Qwen35HParams& hp);
/// Candidates the engine can run (a stored winner outside them is rejected by select_kernel).
[[nodiscard]] std::vector<autotune::Candidate> matmul_candidates();
[[nodiscard]] std::vector<autotune::Candidate> gdn_candidates();

}  // namespace halo::runtime
