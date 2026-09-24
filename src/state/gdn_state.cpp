// Per-sequence GDN recurrent + conv state with rollback slots (D-012) and snapshots (D-013).

#include "halo/state/gdn_state.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

#include "halo/core/error.h"

namespace halo::state {

namespace {

std::size_t mul_checked(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw_error(ErrorCode::Config, "GDN state size overflow");
    }
    return a * b;
}

}  // namespace

std::size_t GdnShape::recurrent_floats() const {
    HALO_CHECK(n_v_heads > 0 && d_k > 0 && d_v > 0, ErrorCode::Config, "GdnShape: zero head dimension");
    return mul_checked(mul_checked(n_v_heads, d_k), d_v);
}

std::size_t GdnShape::conv_floats() const {
    HALO_CHECK(conv_kernel >= 1 && conv_channels > 0, ErrorCode::Config, "GdnShape: bad conv kernel {} / channels {}",
               conv_kernel, conv_channels);
    return mul_checked(conv_kernel - 1, conv_channels);
}

std::size_t GdnShape::total_floats() const { return mul_checked(n_layers, layer_floats()); }

GdnState::GdnState(GdnShape shape, std::size_t max_slots) : shape_(shape), max_slots_(max_slots) {
    const std::size_t total = shape_.total_floats();
    try {
        live_.assign(total, 0.0f);
        slots_.assign(mul_checked(total, max_slots), 0.0f);
    } catch (const std::bad_alloc&) {
        throw_error(ErrorCode::Memory, "GdnState: cannot allocate {} bytes ({} slots)",
                    total * sizeof(float) * (1 + max_slots), max_slots);
    }
}

void GdnState::check_layer(std::size_t layer) const {
    HALO_CHECK(layer < shape_.n_layers, ErrorCode::Kernel, "GdnState: layer {} of {}", layer, shape_.n_layers);
}

std::span<float> GdnState::recurrent(std::size_t layer) {
    check_layer(layer);
    return {live(layer), shape_.recurrent_floats()};
}

cpu::Rows GdnState::conv(std::size_t layer) {
    check_layer(layer);
    return {live(layer) + shape_.recurrent_floats(), shape_.conv_kernel - 1, shape_.conv_channels, shape_.conv_channels};
}

std::span<float> GdnState::recurrent_slots(std::size_t layer, std::size_t n) {
    check_layer(layer);
    HALO_CHECK(n <= max_slots_, ErrorCode::Kernel, "GdnState: {} slots requested, {} available", n, max_slots_);
    if (n == 0) return {};
    return {slots_.data() + layer * shape_.layer_floats() * max_slots_, n * shape_.recurrent_floats()};
}

std::span<float> GdnState::conv_slots(std::size_t layer, std::size_t n) {
    check_layer(layer);
    HALO_CHECK(n <= max_slots_, ErrorCode::Kernel, "GdnState: {} slots requested, {} available", n, max_slots_);
    if (n == 0 || shape_.conv_floats() == 0) return {};
    return {slots_.data() + layer * shape_.layer_floats() * max_slots_ + max_slots_ * shape_.recurrent_floats(),
            n * shape_.conv_floats()};
}

void GdnState::mark_slots_written(std::size_t rows, std::size_t n_slots) noexcept {
    slot_rows_ = rows;
    slots_valid_ = std::min(rows, n_slots);
}

void GdnState::commit_rows_kept(std::size_t rows, std::size_t kept) {
    HALO_CHECK(rows == slot_rows_ && kept >= 1 && kept <= rows, ErrorCode::Api,
               "GdnState::commit_rows_kept({}, {}): last slot-writing call had {} rows", rows, kept, slot_rows_);
    const std::size_t slot = rows - kept;
    HALO_CHECK(slot < slots_valid_ || slot == 0, ErrorCode::Api,
               "GdnState::commit_rows_kept: slot {} not written ({} valid)", slot, slots_valid_);
    if (slot > 0) {
        const std::size_t rf = shape_.recurrent_floats();
        const std::size_t cf = shape_.conv_floats();
        for (std::size_t l = 0; l < shape_.n_layers; ++l) {
            const float* base = slots_.data() + l * shape_.layer_floats() * max_slots_;
            std::memcpy(live(l), base + slot * rf, rf * sizeof(float));
            if (cf > 0) std::memcpy(live(l) + rf, base + max_slots_ * rf + slot * cf, cf * sizeof(float));
        }
    }
    drop_slots();
}

std::uint64_t GdnState::commit_bytes(std::size_t rows, std::size_t kept) const noexcept {
    if (kept >= rows) return 0;
    return 2ULL * shape_.n_layers * shape_.layer_floats() * sizeof(float);
}

void GdnState::reset() noexcept {
    std::fill(live_.begin(), live_.end(), 0.0f);
    drop_slots();
}

GdnSnapshot GdnState::snapshot() const {
    GdnSnapshot s;
    try {
        s.data = live_;
    } catch (const std::bad_alloc&) {
        throw_error(ErrorCode::Memory, "GdnState: cannot allocate a {}-byte snapshot", live_.size() * sizeof(float));
    }
    return s;
}

void GdnState::restore(const GdnSnapshot& s) {
    HALO_CHECK(s.data.size() == live_.size(), ErrorCode::Api, "GdnState::restore: snapshot of {} floats, state has {}",
               s.data.size(), live_.size());
    std::copy(s.data.begin(), s.data.end(), live_.begin());
    drop_slots();
}

void GdnState::copy_from(const GdnState& other) {
    HALO_CHECK(other.live_.size() == live_.size(), ErrorCode::Api, "GdnState::copy_from: shape mismatch");
    std::copy(other.live_.begin(), other.live_.end(), live_.begin());
    drop_slots();
}

}  // namespace halo::state
