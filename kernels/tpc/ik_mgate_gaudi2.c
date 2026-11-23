// gaudium_ik_mgate: an Inkling 2 message gate. With the two gate rows f (forget) and g (gain) of the low-rank projection,
//   out = (1 - tanh(f)) h + tanh(g) m
//  in0 h f32 {H, T, NQH} split layout (slice jh)   in1 m f32 {H, T, NQM} split layout (slice jm)
//  in2 fg {2 H, T}: channels [0, H) = f, [H, 2 H) = g.
//      default build: bf16, the checkpoint's channel order.   -DFG_F32: f32, split layout (the loader stores the projection's
//      rows in that order)
//  out0 out f32 {H, T} split layout.   index space {H / 128, T}.   scalars: jh, jm
#include "ik_common.h"
#define LD2(x_, c0_, t_, j_) ({ float128 _x; int5 _c = {(c0_), (t_), (j_), 0, 0}; _x.v1 = v_f32_ld_tnsr_b(_c, x_); _c[0] += 64; _x.v2 = v_f32_ld_tnsr_b(_c, x_); _x; })
#ifdef FG_F32
#define LDFG(c0_, t_) LD2(fg, c0_, t_, 0)
#else
#define LDFG(c0_, t_) ({ int5 _c = {(c0_), (t_), 0, 0, 0}; v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_c, fg)); })
#endif

void main(tensor h, tensor m, tensor fg, tensor out, int jh, int jm)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int H = get_dim_size(out, 0);
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        const float128 x = LD2(h, 128 * b, t, jh), y = LD2(m, 128 * b, t, jm);
        const float128 f = LDFG(128 * b, t), g = LDFG(H + 128 * b, t);
        const float64 f1 = TANH(f.v1), f2 = TANH(f.v2), g1 = TANH(g.v1), g2 = TANH(g.v2);
        int5 oc = {128 * b, t, 0, 0, 0};
        v_f32_st_tnsr(oc, out, (1.0f - f1) * x.v1 + g1 * y.v1); oc[0] += 64;
        v_f32_st_tnsr(oc, out, (1.0f - f2) * x.v2 + g2 * y.v2);
    }
}
