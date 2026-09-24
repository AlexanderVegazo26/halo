#include "halo/autotune/lookup.h"

#include <algorithm>
#include <format>
#include <tuple>

#include "halo/autotune/db.h"
#include "sqlite.h"

namespace halo::autotune {

std::string_view to_string(SelectionSource s) noexcept {
    switch (s) {
        case SelectionSource::Exact: return "exact";
        case SelectionSource::Compatible: return "compatible";
        case SelectionSource::Heuristic: return "heuristic";
        case SelectionSource::None: return "none";
    }
    return "?";
}

ProfileLookup ProfileLookup::open(const std::filesystem::path& path) {
    OpenOptions o;
    o.read_only = true;
    // Validates the file (typed errors) exactly like the writer does.
    ProfileDb db = ProfileDb::open(path, o);
    (void)db.schema_version();
    // A second read-only connection for the snapshot (ProfileDb keeps its SQL private).
    sql::Connection c(path.string(), /*read_only=*/true, o.busy_timeout_ms);
    sql::Statement s(c,
                     "SELECT w.id, h.halo_version, m.model_hash, m.pack_id, h.gpu_device, h.gpu_arch, "
                     "h.driver_version, h.rocm_version, h.vulkan_version, h.kernel_version, h.os, h.power_mode, "
                     "h.isa_target, o.family, o.shape, w.backend, k.params, w.value, w.strategy, w.created_at "
                     "FROM winning_configuration w "
                     "JOIN hardware_profile h ON h.id = w.hardware_profile_id "
                     "JOIN model_profile m ON m.id = w.model_profile_id "
                     "JOIN operator_profile o ON o.id = w.operator_profile_id "
                     "JOIN kernel_candidate k ON k.id = w.kernel_candidate_id "
                     "ORDER BY w.id");
    ProfileLookup lk;
    while (s.step()) {
        Entry e;
        e.id = s.int64(0);
        ProfileKey& k = e.key;
        k.halo_version = s.text(1);
        k.model_hash = s.text(2);
        k.pack_id = s.text(3);
        k.gpu_device = s.text(4);
        k.gpu_arch = s.text(5);
        k.driver_version = s.text(6);
        k.rocm_version = s.text(7);
        k.vulkan_version = s.text(8);
        k.kernel_version = s.text(9);
        k.os = s.text(10);
        k.power_mode = s.text(11);
        k.isa_target = s.text(12);
        e.op = {s.text(13), s.text(14)};
        e.backend = s.text(15);
        e.candidate = s.text(16);
        (void)Candidate::parse(e.candidate);  // reject a malformed row at open, not at use
        e.value = s.real(17);
        e.strategy = s.text(18);
        e.created_at = s.text(19);
        lk.entries_.push_back(std::move(e));
    }
    return lk;
}

ProfileLookup ProfileLookup::empty() { return ProfileLookup{}; }

LookupResult ProfileLookup::find(const ProfileKey& key, const OpKey& op, std::string_view backend) const {
    LookupResult r;
    const Entry* best_exact = nullptr;
    const Entry* best_compat = nullptr;
    std::size_t best_compat_diff = 0;
    std::vector<std::string> best_compat_fields;
    // Later (created_at, id) is more recent.
    const auto newer = [](const Entry* a, const Entry* b) {
        return std::tie(a->created_at, a->id) > std::tie(b->created_at, b->id);
    };
    for (const Entry& e : entries_) {
        if (e.op != op || e.backend != backend) continue;
        const KeyMatch m = match_keys(e.key, key);
        switch (m.kind) {
            case MatchKind::Exact:
                if (best_exact == nullptr || newer(&e, best_exact)) best_exact = &e;
                break;
            case MatchKind::Compatible:
                if (best_compat == nullptr || m.differing.size() < best_compat_diff ||
                    (m.differing.size() == best_compat_diff && newer(&e, best_compat))) {
                    best_compat = &e;
                    best_compat_diff = m.differing.size();
                    best_compat_fields = m.differing;
                }
                break;
            case MatchKind::Mismatch: {
                std::string fields;
                for (const auto& f : m.mismatched) fields += (fields.empty() ? "" : ",") + f;
                r.rejections.push_back(std::format("profile #{} rejected: {} differ", e.id, fields));
                break;
            }
        }
    }
    const Entry* pick = best_exact != nullptr ? best_exact : best_compat;
    if (pick == nullptr) return r;
    r.source = best_exact != nullptr ? SelectionSource::Exact : SelectionSource::Compatible;
    if (best_exact == nullptr) r.differing = best_compat_fields;
    r.candidate = Candidate::parse(pick->candidate);
    r.value = pick->value;
    r.strategy = pick->strategy;
    r.created_at = pick->created_at;
    return r;
}

LookupResult select_kernel(const ProfileLookup& lookup, const ProfileKey& key, const OpKey& op,
                           std::string_view backend, const std::vector<Candidate>& candidates, const CostModel* model,
                           const std::function<std::optional<CostInputs>(const Candidate&)>& cost) {
    LookupResult r = lookup.find(key, op, backend);
    if (r.source != SelectionSource::None) {
        if (std::find(candidates.begin(), candidates.end(), *r.candidate) != candidates.end()) return r;
        r.rejections.push_back(std::format("stored {} winner {} is not a current candidate", to_string(r.source),
                                           r.candidate->to_string()));
        r.source = SelectionSource::None;
        r.candidate.reset();
        r.differing.clear();
    }
    if (model == nullptr || !cost) return r;
    std::optional<std::pair<double, std::string>> best;
    for (const auto& c : candidates) {
        const auto in = cost(c);
        if (!in) continue;
        const double t = model->estimate(*in).total_ns;
        const std::string text = c.to_string();
        if (!best || t < best->first || (t == best->first && text < best->second)) {
            best = {t, text};
            r.candidate = c;
        }
    }
    if (best) {
        r.source = SelectionSource::Heuristic;
        r.value = best->first;
        r.strategy = "cost_model";
    }
    return r;
}

}  // namespace halo::autotune
