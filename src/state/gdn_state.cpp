// Per-sequence GDN recurrent + conv state ring (D-012 / ADR-001 §5.3) and snapshots (D-013).

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

GdnState::GdnState(GdnShape shape, std::size_t max_slots) : shape_(shape), p_(std::max<std::size_t>(2, max_slots + 1)) {
    const std::size_t total = mul_checked(shape_.total_floats(), p_);
    try {
        host_.assign(total, 0.0f);
    } catch (const std::bad_alloc&) {
        throw_error(ErrorCode::Memory, "GdnState: cannot allocate {} bytes ({} states per layer)", total * sizeof(float), p_);
    }
}

void GdnState::check_layer(std::size_t layer) const {
    HALO_CHECK(layer < shape_.n_layers, ErrorCode::Kernel, "GdnState: layer {} of {}", layer, shape_.n_layers);
}

void GdnState::check_coherent(const char* what) const {
    HALO_CHECK(coherent_, ErrorCode::Api,
               "GdnState: {} needs the canonical host slab; this state is device-resident", what);
}

void GdnState::check_current(const char* what) const {
    HALO_CHECK(coherent_ || pulled_, ErrorCode::Api,
               "GdnState: {} on a device-resident state: the host slab is a stale mirror (call pull() first)", what);
}

// ---- backend attachment -----------------------------------------------------------------

void GdnState::attach(backend::Backend& be) {
    if (be_ == &be) return;
    HALO_CHECK(be_ == nullptr, ErrorCode::Api, "GdnState: already attached to another backend");
    const bool zero_copy = be.kind() == backend::Kind::Cpu || be.kind() == backend::Kind::HipEmulation;
    const std::uint64_t bytes = host_.size() * sizeof(float);
    if (zero_copy) {
        dev_ = be.import_host(std::as_writable_bytes(std::span(host_)));
        HALO_CHECK(dev_->host_ptr() == static_cast<void*>(host_.data()), ErrorCode::Backend,
                   "GdnState: a zero-copy backend returned an import that does not wrap the slab");
    } else {
        // Device-resident (ADR-001 §5.2 State arena). cpu/vulkan allocate() zero-fill; a host
        // slab that already carries content (a restore before attach) is uploaded.
        dev_ = be.allocate(bytes, backend::Tier::Vram);
        be_ = &be;
        bool nonzero = false;
        for (const float f : host_) nonzero = nonzero || f != 0.0f;
        if (nonzero) upload_live();
    }
    be_ = &be;
    coherent_ = zero_copy;
}

backend::TensorRef GdnState::recurrent_ref(std::size_t layer) const {
    check_layer(layer);
    HALO_CHECK(dev_ != nullptr, ErrorCode::Api, "GdnState: not attached to a backend");
    const std::uint64_t rf = shape_.recurrent_floats();
    return {dev_.get(), rec_at(layer, 0) * sizeof(float), rf * p_ * sizeof(float), 0};
}

backend::TensorRef GdnState::conv_ref(std::size_t layer) const {
    check_layer(layer);
    HALO_CHECK(dev_ != nullptr, ErrorCode::Api, "GdnState: not attached to a backend");
    const std::uint64_t cf = shape_.conv_floats();
    return {dev_.get(), conv_at(layer, 0) * sizeof(float), cf * p_ * sizeof(float), 0};
}

void GdnState::pull() {
    if (coherent_) return;
    HALO_CHECK(dev_ != nullptr, ErrorCode::Api, "GdnState: not attached to a backend");
    const auto s = be_->create_stream();
    be_->download(*s, backend::TensorRef::of(*dev_), std::as_writable_bytes(std::span(host_)));
    s->submit();
    s->wait();
    pulled_ = true;
}

// ---- host views --------------------------------------------------------------------------

std::span<float> GdnState::recurrent(std::size_t layer) {
    check_layer(layer);
    check_current("recurrent()");
    return {live_rec(layer), shape_.recurrent_floats()};
}

cpu::Rows GdnState::conv(std::size_t layer) {
    check_layer(layer);
    check_current("conv()");
    return {live_conv(layer), shape_.conv_kernel - 1, shape_.conv_channels, shape_.conv_channels};
}

cpu::StateRingView GdnState::recurrent_ring(std::size_t layer, std::size_t n_slots) {
    check_layer(layer);
    check_coherent("recurrent_ring()");
    const std::size_t rf = shape_.recurrent_floats();
    return {std::span<float>(host_.data() + rec_at(layer, 0), rf * p_), p_, live_, n_slots};
}

cpu::StateRingView GdnState::conv_ring(std::size_t layer, std::size_t n_slots) {
    check_layer(layer);
    check_coherent("conv_ring()");
    const std::size_t cf = shape_.conv_floats();
    return {std::span<float>(host_.data() + conv_at(layer, 0), cf * p_), p_, live_, n_slots};
}

std::span<const float> GdnState::slab() const {
    check_current("slab()");
    return host_;
}

// ---- the step protocol (ADR-001 §5.3) ------------------------------------------------------

void GdnState::begin_step(std::size_t rows, std::size_t n_slots) {
    HALO_CHECK(!pending_, ErrorCode::Api,
               "GdnState: a forward started with an uncommitted verify ({} rows, {} slots pending)", pend_rows_,
               pend_slots_);
    HALO_CHECK(rows >= 1, ErrorCode::Api, "GdnState: a step has no rows");
    HALO_CHECK(n_slots <= max_slots(), ErrorCode::Api, "GdnState: {} slots requested, the ring holds {}", n_slots,
               max_slots());
    base_ = live_;
    pulled_ = coherent_;  // device writes follow: the mirror is stale from here on
    if (needs_zero_) {
        // reset() on a device-resident state defers the zeroing to here (reset is noexcept).
        const std::size_t rf = shape_.recurrent_floats(), cf = shape_.conv_floats();
        std::vector<float> zeros(std::max(rf, cf), 0.0f);
        const auto s = be_->create_stream();
        for (std::size_t l = 0; l < shape_.n_layers; ++l) {
            be_->upload(*s, rec_slot_ref(l, live_), std::as_bytes(std::span(zeros.data(), rf)));
            if (cf > 0) be_->upload(*s, conv_slot_ref(l, live_), std::as_bytes(std::span(zeros.data(), cf)));
        }
        s->submit();
        s->wait();
        needs_zero_ = false;
    }
}

void GdnState::mark_slots_written(std::size_t rows, std::size_t n_slots) noexcept {
    // The clean-status commit (ADR-001 §5.3): keep all rows, live = (base + 1) mod P.
    live_ = (base_ + 1) % static_cast<std::uint32_t>(p_);
    pend_rows_ = rows;
    pend_slots_ = std::min(rows, n_slots);
    pending_ = n_slots > 0;  // a verify stays referenced until commit_rows_kept / drop_slots
}

void GdnState::commit_rows_kept(std::size_t rows, std::size_t kept) {
    HALO_CHECK(pending_, ErrorCode::Api, "GdnState::commit_rows_kept({}, {}): no pending verify", rows, kept);
    HALO_CHECK(rows == pend_rows_ && kept >= 1 && kept <= rows, ErrorCode::Api,
               "GdnState::commit_rows_kept({}, {}): last slot-writing call had {} rows", rows, kept, pend_rows_);
    const std::size_t slot = rows - kept;
    HALO_CHECK(slot < pend_slots_, ErrorCode::Api, "GdnState::commit_rows_kept: slot {} not written ({} valid)", slot,
               pend_slots_);
    // live = (base + 1 + (rows - kept)) mod P, from the recorded base: one integer, no copy.
    live_ = static_cast<std::uint32_t>((base_ + 1 + slot) % p_);
    drop_slots();
}

void GdnState::drop_slots() noexcept {
    pend_rows_ = 0;
    pend_slots_ = 0;
    pending_ = false;
}

std::uint64_t GdnState::commit_bytes(std::size_t rows, std::size_t kept) const noexcept {
    (void)rows;
    (void)kept;
    return 0;  // the ring commit moves one integer, not a state
}

// ---- snapshots / reset --------------------------------------------------------------------

void GdnState::zero_live() {
    const std::size_t rf = shape_.recurrent_floats(), cf = shape_.conv_floats();
    for (std::size_t l = 0; l < shape_.n_layers; ++l) {
        std::fill_n(live_rec(l), rf, 0.0f);
        std::fill_n(live_conv(l), cf, 0.0f);
    }
}

void GdnState::gather_live(float* dst) const {
    const std::size_t rf = shape_.recurrent_floats(), cf = shape_.conv_floats();
    for (std::size_t l = 0; l < shape_.n_layers; ++l) {
        std::memcpy(dst + l * (rf + cf), host_.data() + rec_at(l, live_), rf * sizeof(float));
        std::memcpy(dst + l * (rf + cf) + rf, host_.data() + conv_at(l, live_), cf * sizeof(float));
    }
}

void GdnState::scatter_live(const float* src) {
    const std::size_t rf = shape_.recurrent_floats(), cf = shape_.conv_floats();
    for (std::size_t l = 0; l < shape_.n_layers; ++l) {
        std::memcpy(host_.data() + rec_at(l, live_), src + l * (rf + cf), rf * sizeof(float));
        std::memcpy(host_.data() + conv_at(l, live_), src + l * (rf + cf) + rf, cf * sizeof(float));
    }
}

void GdnState::upload_live() {
    const std::size_t rf = shape_.recurrent_floats(), cf = shape_.conv_floats();
    const auto s = be_->create_stream();
    for (std::size_t l = 0; l < shape_.n_layers; ++l) {
        be_->upload(*s, rec_slot_ref(l, live_), std::as_bytes(std::span(live_rec(l), rf)));
        if (cf > 0) be_->upload(*s, conv_slot_ref(l, live_), std::as_bytes(std::span(live_conv(l), cf)));
    }
    s->submit();
    s->wait();
}

void GdnState::reset() noexcept {
    live_ = 0;
    base_ = 0;
    drop_slots();
    if (coherent_) {
        zero_live();
    } else {
        zero_live();        // keep the mirror's live slot in step with the deferred zeroing
        needs_zero_ = true;  // applied at the next begin_step (reset stays noexcept)
        pulled_ = false;     // the device's live slot is stale until then
    }
}

GdnSnapshot GdnState::snapshot() const {
    GdnSnapshot s;
    try {
        s.data.resize(shape_.total_floats());
    } catch (const std::bad_alloc&) {
        throw_error(ErrorCode::Memory, "GdnState: cannot allocate a {}-byte snapshot", shape_.total_bytes());
    }
    if (coherent_) {
        gather_live(s.data.data());
        return s;
    }
    // Device-resident: download slab[live] of every layer.
    const std::size_t rf = shape_.recurrent_floats(), cf = shape_.conv_floats();
    const auto st = be_->create_stream();
    for (std::size_t l = 0; l < shape_.n_layers; ++l) {
        be_->download(*st, {dev_.get(), rec_at(l, live_) * sizeof(float), rf * sizeof(float), 0},
                      std::as_writable_bytes(std::span(s.data.data() + l * (rf + cf), rf)));
        if (cf > 0) {
            be_->download(*st, {dev_.get(), conv_at(l, live_) * sizeof(float), cf * sizeof(float), 0},
                          std::as_writable_bytes(std::span(s.data.data() + l * (rf + cf) + rf, cf)));
        }
    }
    st->submit();
    st->wait();
    return s;
}

void GdnState::restore(const GdnSnapshot& s) {
    HALO_CHECK(s.data.size() == shape_.total_floats(), ErrorCode::Api,
               "GdnState::restore: snapshot of {} floats, state has {}", s.data.size(), shape_.total_floats());
    scatter_live(s.data.data());
    if (!coherent_) {
        upload_live();
        pulled_ = false;  // only the live slots of the mirror are current
    }
    needs_zero_ = false;
    drop_slots();
}

void GdnState::copy_from(const GdnState& other) {
    HALO_CHECK(other.shape_.total_floats() == shape_.total_floats(), ErrorCode::Api, "GdnState::copy_from: shape mismatch");
    restore(other.snapshot());
}

}  // namespace halo::state
