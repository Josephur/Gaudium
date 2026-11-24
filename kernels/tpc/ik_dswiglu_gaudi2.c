// gaudium_ik_dswiglu: the plain SwiGLU of a block-drafter layer: a = bf16( silu(g) u ) in f32 with one rounding.
//  in0 gu bf16 {2 F, T}: channels [0, F) the gate projection, [F, 2 F) the up projection   out0 a bf16 {F, T}
//  index space {F / 128, T}
#include "ik_common.h"

void main(tensor gu, tensor a)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int F = get_dim_size(a, 0);
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t)
        for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
            int5 gc = {128 * b, t, 0, 0, 0}, uc = {F + 128 * b, t, 0, 0, 0};
            const float128 g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, gu));
            const float128 u = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(uc, gu));
            float128 r; r.v1 = g.v1 * v_sigmoid_f32(g.v1) * u.v1; r.v2 = g.v2 * v_sigmoid_f32(g.v2) * u.v2;
            v_bf16_st_tnsr(gc, a, v_convert_f32_to_bf16_all_b(r));
        }
}
