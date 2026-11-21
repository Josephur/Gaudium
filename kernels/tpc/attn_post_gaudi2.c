#include "attn_shape.h"   // ATT_NKV, ATT_GQ, ATT_NH, ATT_NPT, ATT_GATE: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_attn_post: output gate of Qwen3.5 attention and re-layout to the bf16 row the o_proj GEMM reads.
//   core[h] = attn_out[h] * sigmoid(gate[h]);  gate[h] = qg channels 6144 + 256h ..
//  in0 ao bf16 {256, 6T, 4} (P.V output; row hl*T + t of group g = head 6g + hl)   in1 qg bf16 {12288, T}
//  out0 core bf16 {6144, T}.   Both inputs are read through the deinterleaving conversion, the output through the
//  interleaving one, so the bf16 row comes out in natural channel order.   index space: 24 heads.  scalars: T
void main(tensor ao, tensor qg, tensor core, int T)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            int5 ac = {0, (h % ATT_GQ) * T + t, h / ATT_GQ, 0, 0};
            float128 o1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ac, ao)); ac[0] = 128;
            float128 o2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ac, ao));
            int5 gc = {ATT_GATE + 256 * h, t, 0, 0, 0};
            float128 g1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, qg)); gc[0] = ATT_GATE + 256 * h + 128;
            float128 g2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, qg));
            float128 r1, r2;
            r1.v1 = o1.v1 * v_sigmoid_f32(g1.v1); r1.v2 = o1.v2 * v_sigmoid_f32(g1.v2);
            r2.v1 = o2.v1 * v_sigmoid_f32(g2.v1); r2.v2 = o2.v2 * v_sigmoid_f32(g2.v2);
            int5 cc = {256 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(r1)); cc[0] = 256 * h + 128;
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(r2));
        }
    }
}
