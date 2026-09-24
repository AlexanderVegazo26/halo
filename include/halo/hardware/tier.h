#pragma once
// Memory tiers (TRD §13.1). Header-only so that modules which only need the vocabulary
// (e.g. halo_memory) do not have to link halo_hardware.

#include <cstdint>
#include <string_view>

namespace halo::hardware {

/// Where bytes live. On the Strix Halo APU all tiers are the same physical LPDDR5x; they
/// differ in who manages the pages and in effective bandwidth.
enum class MemoryTier : std::uint8_t {
    Vram,    ///< BIOS UMA carveout reported by amdgpu as VRAM (`mem_info_vram_total`).
    Gtt,     ///< GPU-mapped system RAM (`mem_info_gtt_total`); pages come from the OS pool.
    Pinned,  ///< Pinned host staging allocations (part of the OS pool).
    Host,    ///< Ordinary pageable host memory (`/proc/meminfo`).
};

/// Which processor issues the memory traffic (TRD §13.1: CPU-issued bandwidth differs).
enum class Processor : std::uint8_t { Cpu, Gpu };

[[nodiscard]] constexpr std::string_view to_string(MemoryTier t) noexcept {
    switch (t) {
        case MemoryTier::Vram: return "VRAM";
        case MemoryTier::Gtt: return "GTT";
        case MemoryTier::Pinned: return "PINNED";
        case MemoryTier::Host: return "HOST";
    }
    return "?";
}

[[nodiscard]] constexpr std::string_view to_string(Processor p) noexcept {
    switch (p) {
        case Processor::Cpu: return "cpu";
        case Processor::Gpu: return "gpu";
    }
    return "?";
}

}  // namespace halo::hardware
