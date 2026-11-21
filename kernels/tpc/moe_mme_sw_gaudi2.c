// gaudium_moe_mswiglu: SwiGLU of the MME prefill path's gate+up GEMM output (docs/notes/flash-next-prefill.md).
//   in0 G   f32 {2 EF, C, B}   N order of gaudium_moe_mdq13: per 256-block t, [0, 128) gate, [128, 256) up; position m = 2 l + h of
//                              each half holds channel 128 t + 64 h + l
//   in1 src i32 {B}            -1 = unused entry (skipped)
//   out0 M  bf16 {EF, C, B}    h = silu(g) * u (w4::ref::swiglu, f32, no FMA), written as one bf16 "all" conversion of the f32 vectors
//                              m in [0, 64) (even lanes) and [64, 128) (odd lanes): position 128 t + 2 p + s holds m = 64 s + p, i.e. channel
//                              128 t + 64 (m & 1) + (m >> 1). gaudium_moe_mdq2 writes the down matrix's K rows in this order (kpos).
// Index space {B}: one point = one entry, all C rows.
#include "moe_w4_common.h"
#define LDI(tn, c0) s_i32_ld_g(gen_addr(((int5){(c0), 0, 0, 0, 0}), tn))
void main(tensor G, tensor src, tensor M)
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    const int NT = get_dim_size(M, 0) / 128, C = get_dim_size(M, 1);
    for (int b = off[0]; b < off[0] + sz[0]; ++b) {
        if (LDI(src, b) < 0) continue;
        for (int c = 0; c < C; ++c)
            for (int t = 0; t < NT; ++t) {
                int5 gc = {256 * t, c, b, 0, 0};
                const float64 g0 = v_f32_ld_tnsr_b(gc, G); gc[0] += 64;
                const float64 g1 = v_f32_ld_tnsr_b(gc, G); gc[0] += 64;
                const float64 u0 = v_f32_ld_tnsr_b(gc, G); gc[0] += 64;
                const float64 u1 = v_f32_ld_tnsr_b(gc, G);
                float128 h; h.v1 = SWIGLU(g0, u0); h.v2 = SWIGLU(g1, u1);
                int5 mc = {128 * t, c, b, 0, 0};
                v_bf16_st_tnsr(mc, M, v_convert_f32_to_bf16_all_b(h, SW_RHNE));
            }
    }
}
