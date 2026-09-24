#pragma once
// Runtime-side profile lookup (TRD §56 steps 1-3, §64 "the runtime consumes the profile on
// next launch").
//
// "Autotuning never runs per-request" holds by construction here:
//   * ProfileLookup opens the database READ-ONLY and copies every winning configuration
//     into memory at open; after that no SQL runs.
//   * Nothing in this header accepts a TunableOp, a Clock or a writable ProfileDb, so a
//     caller of the lookup cannot measure or persist. Measuring (step 4) and persisting
//     (step 5) exist only in autotune::tune() (tuner.h), an explicit offline call.
//   * Lookups are const and the snapshot is immutable: thread-safe, no locking.
//
// Every answer says which §56 step produced it; there is no silent default.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "halo/autotune/cost_model.h"
#include "halo/autotune/profile_key.h"
#include "halo/autotune/types.h"

namespace halo::autotune {

enum class SelectionSource : std::uint8_t {
    Exact,       ///< step 1: exact profile match
    Compatible,  ///< step 2: version-tolerant match (flagged: see `differing`)
    Heuristic,   ///< step 3: cost model only, nothing measured
    None,        ///< no profile and no usable cost model: caller must run `halo tune`
};
[[nodiscard]] std::string_view to_string(SelectionSource s) noexcept;

struct LookupResult {
    SelectionSource source = SelectionSource::None;
    std::optional<Candidate> candidate;
    std::vector<std::string> differing;   ///< Compatible: tolerant key fields that differ
    std::vector<std::string> rejections;  ///< profiles for this op that were rejected, and why
    double value = 0;       ///< Exact/Compatible: stored median ns; Heuristic: predicted total ns
    std::string strategy;   ///< strategy of the stored winner; "cost_model" for Heuristic
    std::string created_at;
};

class ProfileLookup {
public:
    /// Read-only open + snapshot. Throws ProfileDbError (db.h) on a corrupt, foreign or
    /// missing database.
    [[nodiscard]] static ProfileLookup open(const std::filesystem::path& path);
    /// A lookup with no profiles (no database yet): every find() returns None.
    [[nodiscard]] static ProfileLookup empty();

    /// §56 steps 1-2. Among exact matches the most recent wins; among compatible ones the
    /// fewest differing fields, then the most recent (created_at, then row id).
    [[nodiscard]] LookupResult find(const ProfileKey& key, const OpKey& op, std::string_view backend) const;

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    struct Entry {
        std::int64_t id = 0;
        ProfileKey key;
        OpKey op;
        std::string backend;
        std::string candidate;  ///< canonical text
        double value = 0;
        std::string strategy;
        std::string created_at;
    };

private:
    std::vector<Entry> entries_;
};

/// §56 steps 1-3 for one operator: the stored winner (exact, then compatible) if it is one
/// of `candidates` (a stored winner the current build can no longer run, e.g. more threads
/// than now available, is skipped with a rejection note); otherwise the cost model's best
/// predicted candidate (ties: smallest canonical string); otherwise None.
/// `cost` may be empty and `model` null (then step 3 is skipped).
[[nodiscard]] LookupResult select_kernel(const ProfileLookup& lookup, const ProfileKey& key, const OpKey& op,
                                         std::string_view backend, const std::vector<Candidate>& candidates,
                                         const CostModel* model,
                                         const std::function<std::optional<CostInputs>(const Candidate&)>& cost);

}  // namespace halo::autotune
