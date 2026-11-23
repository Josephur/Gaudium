// gaudium_ik_conv: the short causal convolution after the attention and after the experts of an Inkling 2 layer, with its residual:
//   y[t] = bf16( ( sum over lag < KW of w[lag] x[pos - lag]  +  x[pos] ) / sqrt(KW + 1) )   rows before position 0 are zero
// The rows come from a ring by position (gaudium_ik_save wrote the rows of this pass there; earlier passes wrote the ones before).
//  in0 ring bf16 {H, RC}   in1 cw f32 {H, KW} split layout, column = lag (0 = the row itself)
//  default build (after the attention):
//      in2 xin bf16 {H, T} the layer's input, in3 ctl i32 {8}: [2] boundary (1: the layer opens a block), in4 cursor i32 {1}
//      out0 out f32 {H, T} split layout = y  (+ xin when the layer does not open a block)
//  -DMLP build (after the experts):
//      in2 res f32 {H, T} split layout, in3 cursor i32 {1}
//      out0 out bf16 {H, T} = bf16(res + y)
//  index space {H / 128, T}.   scalars: coff, isd (f32 bits of 1 / sqrt(KW + 1))
#include "ik_common.h"
#ifdef MLP
void main(tensor ring, tensor cw, tensor res, tensor cursor, tensor out, int coff, int isd_bits)
#else
void main(tensor ring, tensor cw, tensor xin, tensor ctl, tensor cursor, tensor out, int coff, int isd_bits)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int M1 = get_dim_size(ring, 1) - 1, KW = get_dim_size(cw, 1);
    const float isd = *(float*)&isd_bits;
#ifndef MLP
    zc[0] = 2;
    const int boundary = s_i32_ld_g(gen_addr(zc, ctl));
#endif
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        const int pos = cur + t;
        float64 a1 = 0.f, a2 = 0.f, x1 = 0.f, x2 = 0.f;
        for (int lag = 0; lag < KW; ++lag) {
            const int pp = pos - lag;
            if (pp < 0) break;
            int5 rc = {128 * b, pp & M1, 0, 0, 0}, wc = {128 * b, lag, 0, 0, 0};
            const float128 xv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, ring));
            const float64 w1 = v_f32_ld_tnsr_b(wc, cw); wc[0] += 64;
            const float64 w2 = v_f32_ld_tnsr_b(wc, cw);
            a1 = a1 + xv.v1 * w1; a2 = a2 + xv.v2 * w2;
            if (lag == 0) { x1 = xv.v1; x2 = xv.v2; }
        }
        // the convolution's result is a bf16 value in the reference (its module returns the layer's dtype)
        const float128 yr = RBF((a1 + x1) * isd, (a2 + x2) * isd);
        float64 y1 = yr.v1, y2 = yr.v2;
        int5 c = {128 * b, t, 0, 0, 0};
#ifdef MLP
        const float64 r1 = v_f32_ld_tnsr_b(c, res); c[0] += 64;
        const float64 r2 = v_f32_ld_tnsr_b(c, res); c[0] -= 64;
        float128 o; o.v1 = r1 + y1; o.v2 = r2 + y2;
        v_bf16_st_tnsr(c, out, v_convert_f32_to_bf16_all_b(o));
#else
        if (!boundary) { const float128 xi = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, xin)); y1 = y1 + xi.v1; y2 = y2 + xi.v2; }
        v_f32_st_tnsr(c, out, y1); c[0] += 64; v_f32_st_tnsr(c, out, y2);
#endif
    }
}
