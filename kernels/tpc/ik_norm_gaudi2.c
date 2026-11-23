// gaudium_ik_norm: the weighted RMS norm of an Inkling 2 layer on an f32 row:
//   y = bf16( x rsqrt(mean(x^2) + eps) w )
// in f32 with one rounding. One index-space member is one row and its sum of squares is accumulated in one fixed order, so a row has
// the same values in a pass of one row, of several rows and in a prefill tile.
//  in0 x f32 {H, T, NQ} split layout (slice j)   in1 w bf16 {H}   out0 y bf16 {H, T}.   index space {T}.   scalars: j, eps (f32 bits)
#include "ik_common.h"
#define LD2(x_, b_, t_, j_) ({ float128 _x; int5 _c = {128 * (b_), (t_), (j_), 0, 0}; _x.v1 = v_f32_ld_tnsr_b(_c, x_); _c[0] += 64; _x.v2 = v_f32_ld_tnsr_b(_c, x_); _x; })

void main(tensor x, tensor w, tensor y, int j, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int H = get_dim_size(y, 0), NB = H / 128;
    const float eps = *(float*)&eps_bits, invh = 1.0f / (float)H;
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        float64 a = 0.f;
        for (int b = 0; b < NB; ++b) { const float128 v = LD2(x, b, t, j); a = a + v.v1 * v.v1 + v.v2 * v.v2; }
        const float64 r = v_rsqrt_f32(REDUCE64(a, P8, P4, P2, P1) * invh + eps);
        for (int b = 0; b < NB; ++b) {
            const float128 v = LD2(x, b, t, j);
            int5 wc = {128 * b, 0, 0, 0, 0};
            const float128 g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
            float128 o; o.v1 = v.v1 * r * g.v1; o.v2 = v.v2 * r * g.v2;
            int5 oc = {128 * b, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o));
        }
    }
}
