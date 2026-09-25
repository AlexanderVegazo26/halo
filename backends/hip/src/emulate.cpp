// Host emulation of the HIP kernels: the same kernel bodies as the device, run by HostExec.
// Compiled with -ffp-contract=off like the CPU reference and the device code.

#include "host_exec.h"
#include "launch.h"

namespace halo::hip::detail {

using namespace kern;

void emulate_gdn_recurrent(const GdnRecParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GdnRecRegs, GdnRecShared>(l, reverse, [&](auto& ex, GdnRecShared& sh, unsigned bx, unsigned by) {
        gdn_recurrent_body(ex, sh, p, bx, by);
    });
}

void emulate_gdn_check_g(const GdnCheckParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GdnNoRegs, GdnNoShared>(l, reverse, [&](auto& ex, GdnNoShared& sh, unsigned bx, unsigned by) {
        gdn_check_g_body(ex, sh, p, bx, by);
    });
}

void emulate_gdn_chunk_intra(const GdnChunkParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GdnNoRegs, GdnIntraShared>(l, reverse, [&](auto& ex, GdnIntraShared& sh, unsigned bx, unsigned by) {
        gdn_chunk_intra_body(ex, sh, p, bx, by);
    });
}

void emulate_gdn_chunk_state(const GdnChunkParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GdnNoRegs, GdnStateShared>(l, reverse, [&](auto& ex, GdnStateShared& sh, unsigned bx, unsigned by) {
        gdn_chunk_state_body(ex, sh, p, bx, by);
    });
}

void emulate_conv1d_silu(const ConvParams& p, const Launch& l, bool reverse) {
    emu::run_grid<ConvRegs, ConvShared>(l, reverse, [&](auto& ex, ConvShared& sh, unsigned bx, unsigned by) {
        conv1d_silu_body(ex, sh, p, bx, by);
    });
}

void emulate_norm(const NormParams& p, const Launch& l, bool reverse) {
    emu::run_grid<NormRegs, NormShared>(l, reverse, [&](auto& ex, NormShared& sh, unsigned bx, unsigned by) {
        norm_body(ex, sh, p, bx, by);
    });
}

void emulate_gemv_generic(const GemvParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GemvNoRegs, GemvGenericShared>(l, reverse, [&](auto& ex, GemvGenericShared& sh, unsigned bx, unsigned by) {
        gemv_generic_body(ex, sh, p, bx, by);
    });
}

void emulate_gemv_wave(const GemvParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GemvNoRegs, GemvWaveShared>(l, reverse, [&](auto& ex, GemvWaveShared& sh, unsigned bx, unsigned by) {
        gemv_wave_body(ex, sh, p, bx, by);
    });
}

}  // namespace halo::hip::detail

namespace halo::hip::detail {

using namespace kern;

void emulate_argmax_partial(const ArgmaxParams& p, const Launch& l, bool reverse) {
    emu::run_grid<ArgNoRegs, ArgShared>(l, reverse, [&](auto& ex, ArgShared& sh, unsigned bx, unsigned by) {
        argmax_partial_body(ex, sh, p, bx, by);
    });
}

void emulate_argmax_reduce(const ArgmaxParams& p, const Launch& l, bool reverse) {
    emu::run_grid<ArgNoRegs, ArgShared>(l, reverse, [&](auto& ex, ArgShared& sh, unsigned bx, unsigned by) {
        argmax_reduce_body(ex, sh, p, bx, by);
    });
}

void emulate_topk(const TopkParams& p, const Launch& l, bool reverse) {
    emu::run_grid<ArgNoRegs, TopkShared>(l, reverse, [&](auto& ex, TopkShared& sh, unsigned bx, unsigned by) {
        topk_body(ex, sh, p, bx, by);
    });
}

void emulate_rope(const RopeParams& p, const Launch& l, bool reverse) {
    emu::run_grid<ArgNoRegs, RopeShared>(l, reverse, [&](auto& ex, RopeShared& sh, unsigned bx, unsigned by) {
        rope_body(ex, sh, p, bx, by);
    });
}

void emulate_eltwise(const EwParams& p, const Launch& l, bool reverse) {
    emu::run_grid<ArgNoRegs, EwShared>(l, reverse, [&](auto& ex, EwShared& sh, unsigned bx, unsigned by) {
        ew_body(ex, sh, p, bx, by);
    });
}

}  // namespace halo::hip::detail

namespace halo::hip::detail {

using namespace kern;

void emulate_attn_check(const AttnParams& p, const Launch& l, bool reverse) {
    emu::run_grid<AttnNoRegs, AttnNoRegs>(l, reverse, [&](auto& ex, AttnNoRegs& sh, unsigned bx, unsigned by) {
        attn_check_body(ex, sh, p, bx, by);
    });
}

void emulate_attn_exact(const AttnParams& p, const Launch& l, bool reverse) {
    emu::run_grid<AttnNoRegs, AttnExactShared>(l, reverse, [&](auto& ex, AttnExactShared& sh, unsigned bx, unsigned by) {
        attn_exact_body(ex, sh, p, bx, by);
    });
}

void emulate_attn_online(const AttnParams& p, const Launch& l, bool reverse) {
    emu::run_grid<AttnOnlineRegs, AttnOnlineShared>(l, reverse,
                                                    [&](auto& ex, AttnOnlineShared& sh, unsigned bx, unsigned by) {
                                                        attn_online_body(ex, sh, p, bx, by);
                                                    });
}

}  // namespace halo::hip::detail

namespace halo::hip::detail {

using namespace kern;

void emulate_kv_write(const KvWriteParams& p, const Launch& l, bool reverse) {
    emu::run_grid<DecNoRegs, DecNoRegs>(l, reverse, [&](auto& ex, DecNoRegs& sh, unsigned bx, unsigned by) {
        kv_write_body(ex, sh, p, bx, by);
    });
}

void emulate_get_rows(const GetRowsParams& p, const Launch& l, bool reverse) {
    emu::run_grid<DecNoRegs, DecNoRegs>(l, reverse, [&](auto& ex, DecNoRegs& sh, unsigned bx, unsigned by) {
        get_rows_body(ex, sh, p, bx, by);
    });
}

void emulate_add_norm(const AddNormParams& p, const Launch& l, bool reverse) {
    emu::run_grid<NormRegs, NormShared>(l, reverse, [&](auto& ex, NormShared& sh, unsigned bx, unsigned by) {
        add_norm_body(ex, sh, p, bx, by);
    });
}

}  // namespace halo::hip::detail

namespace halo::hip::detail {

using namespace kern;

void emulate_gdn_gate(const GdnGateParams& p, const Launch& l, bool reverse) {
    emu::run_grid<DecNoRegs, DecNoRegs>(l, reverse, [&](auto& ex, DecNoRegs& sh, unsigned bx, unsigned by) {
        gdn_gate_body(ex, sh, p, bx, by);
    });
}

void emulate_gemm(const GemvParams& p, const Launch& l, bool reverse) {
    emu::run_grid<GemmRegs, GemmShared>(l, reverse, [&](auto& ex, GemmShared& sh, unsigned bx, unsigned by) {
        gemm_body(ex, sh, p, bx, by);
    });
}

}  // namespace halo::hip::detail
