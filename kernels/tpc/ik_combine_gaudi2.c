// gaudium_ik_combine: the back end of an Inkling 2 attention layer: merge the chunk partials, add the attention sink, normalise,
// and gate the result with the sigmoid of the layer's gate projection:
//   M = max(sink, max_ch m_ch);   S = exp(sink - M) + sum_ch exp(m_ch - M) s_ch;   a = sum_ch exp(m_ch - M) ao_part_ch / S
//   core = bf16( a sigmoid(gate) )
// The sink is one more logit of the row's softmax whose value is zero: it only takes probability mass.
//  in0 ao_part f32 {128, R, B}   unnormalised partial outputs, B = G nch chunk-major, R = GQ T, channels in the value rows' order
//              (the loader stores the value projection's rows so that this f32 block is in the split layout)
//  in1 stats   f32 {64, 2, R, B}
//  in2 qx      bf16 {QW, T}: channels [goff, goff + 128 NH) the gate projection, the checkpoint's order
//  in3 sink    bf16 {NH, T}
//  out0 core   bf16 {128 NH, T}, the checkpoint's order
//  index space {NH}.   scalars: T, nch, GQ, goff
#include "ik_common.h"

void main(tensor ao_part, tensor stats, tensor qx, tensor sink, tensor core, int T, int nch, int GQ, int goff)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int G = get_dim_size(ao_part, 2) / nch;
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        const int g = h / GQ, hl = h - g * GQ;
        for (int t = 0; t < T; ++t) {
            const int row = hl * T + t;
            int5 kc = {h, t, 0, 0, 0};
            const float sk = s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(kc, sink)));
            const float64 skv = sk;
            float64 M = skv;
            for (int ch = 0; ch < nch; ++ch) { int5 sc = {0, 0, row, ch * G + g, 0}; M = v_f32_max_b(M, v_f32_ld_tnsr_b(sc, stats)); }
            float64 S = EXPD(skv - M), o0 = 0.f, o1 = 0.f;
            for (int ch = 0; ch < nch; ++ch) {
                const int b = ch * G + g;
                int5 sc = {0, 0, row, b, 0};
                const float64 m = v_f32_ld_tnsr_b(sc, stats); sc[1] = 1;
                const float64 s = v_f32_ld_tnsr_b(sc, stats);
                const float64 w = EXPD(m - M);
                S = S + w * s;
                int5 ac = {0, row, b, 0, 0};
                o0 = o0 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 64;
                o1 = o1 + w * v_f32_ld_tnsr_b(ac, ao_part);
            }
            const float64 inv = RECIP(S);
            int5 gc = {goff + 128 * h, t, 0, 0, 0};
            const float128 gg = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, qx));
            float128 o; o.v1 = o0 * inv * v_sigmoid_f32(gg.v1); o.v2 = o1 * inv * v_sigmoid_f32(gg.v2);
            int5 oc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, core, v_convert_f32_to_bf16_all_b(o));
        }
    }
}
