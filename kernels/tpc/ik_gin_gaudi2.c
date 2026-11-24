// gaudium_ik_gin: the input of an Inkling 2 message gate: the layer's state h and the message m of the previous pass over the
// looped layers, each normalised without a weight, side by side:
//   out = bf16( [ h rsqrt(mean(h^2) + eps) | m rsqrt(mean(m^2) + eps) ] )
//  in0 h f32 {H, T, NQH} split layout (slice jh)   in1 m f32 {H, T, NQM} split layout (slice jm)
//  out0 out bf16 {2 H, T}: channels [0, H) from h, [H, 2 H) from m, the checkpoint's order
//  index space {T}.   scalars: jh, jm, eps (f32 bits)
#include "ik_common.h"
#define LD2(x_, b_, t_, j_) ({ float128 _x; int5 _c = {128 * (b_), (t_), (j_), 0, 0}; _x.v1 = v_f32_ld_tnsr_b(_c, x_); _c[0] += 64; _x.v2 = v_f32_ld_tnsr_b(_c, x_); _x; })

void main(tensor h, tensor m, tensor out, int jh, int jm, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int H = get_dim_size(h, 0), NB = H / 128;
    const float eps = *(float*)&eps_bits, invh = 1.0f / (float)H;
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        float64 ah = 0.f, am = 0.f;
        for (int b = 0; b < NB; ++b) {
            const float128 x = LD2(h, b, t, jh), y = LD2(m, b, t, jm);
            ah = ah + x.v1 * x.v1 + x.v2 * x.v2; am = am + y.v1 * y.v1 + y.v2 * y.v2;
        }
        const float64 rh = v_rsqrt_f32(REDUCE64(ah, P8, P4, P2, P1) * invh + eps);
        const float64 rm = v_rsqrt_f32(REDUCE64(am, P8, P4, P2, P1) * invh + eps);
        for (int b = 0; b < NB; ++b) {
            const float128 x = LD2(h, b, t, jh), y = LD2(m, b, t, jm);
            float128 o; o.v1 = x.v1 * rh; o.v2 = x.v2 * rh;
            int5 oc = {128 * b, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(o));
            o.v1 = y.v1 * rm; o.v2 = y.v2 * rm; oc[0] = H + 128 * b;
            v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(o));
        }
    }
}
