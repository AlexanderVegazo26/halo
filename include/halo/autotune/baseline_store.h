#pragma once
// Stores baseline artifacts (TRD §51, profiling/baseline.h) in the profile database's
// benchmark_run table, the "continuously reproducible baseline database all PR-004 claims
// are judged against".
//
// What is stored: profiling::aggregate_records(artifact) — one record per configuration
// (median of the measured samples, the per-metric statistics under extra.aggregate) with
// kind "baseline". Query with ProfileDb::latest_run("baseline", backend, pack, context,
// mode); see baseline.h for the mode / context contract.
//
// An artifact that is not `valid` (thermal drift, unpinned or changed power mode,
// nonconformant sample counts, or any tool failure) is refused with Error(Config) unless
// allow_invalid is set, so an unusable run never becomes the "latest baseline".

#include <cstddef>

#include "halo/autotune/db.h"
#include "halo/profiling/suite.h"

namespace halo::autotune {

/// Returns the number of records stored.
std::size_t store_baseline(ProfileDb& db, const profiling::SuiteArtifact& artifact, bool allow_invalid = false);

}  // namespace halo::autotune
