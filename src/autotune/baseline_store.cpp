#include "halo/autotune/baseline_store.h"

#include <string>

#include "halo/core/error.h"
#include "halo/profiling/baseline.h"

namespace halo::autotune {

std::size_t store_baseline(ProfileDb& db, const profiling::SuiteArtifact& a, bool allow_invalid) {
    HALO_CHECK(a.suite == "baseline", ErrorCode::Config, "store_baseline: artifact suite is '{}', not baseline",
               a.suite);
    if (!a.valid && !allow_invalid) {
        std::string why;
        if (!a.thermal.valid) why += " thermal: " + a.thermal.reason + ";";
        if (!a.power_mode_pinned.comparable) why += " power mode: " + a.power_mode_pinned.reason + ";";
        if (!a.conformant) why += " not conformant (sample counts or a FAILED note);";
        throw_error(ErrorCode::Config, "store_baseline: artifact is not valid:{} pass allow_invalid to store anyway",
                    why);
    }
    const auto recs = profiling::aggregate_records(a);
    for (const auto& r : recs) (void)db.add_benchmark_record(r, "baseline");
    return recs.size();
}

}  // namespace halo::autotune
