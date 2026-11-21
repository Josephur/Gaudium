#include "attn_shape.h"   // ATT_NKV, ATT_GQ, ATT_NH, ATT_NPT, ATT_GATE: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_attn_post_f32: output gate of Qwen3.5 attention from the f32 output of the chunked core (gaudium_attn_combine)
//  and re-layout to the bf16 row the o_proj GEMM reads.   core[h] = ao[h] * sigmoid(gate[h]);  gate[h] = qg channels 6144 + 256h ..
//  in0 ao f32 {256, 6T, 4} (row hl*T + t of group g = head 6g + hl), channels in LANE ORDER inside each 128-block
//      (lane(c) = c odd ? 64 + c/2 : c/2): the loader stores the rows of W_v in that order, so the V cache, the PV partials and
//      the combined output all carry it, and a plain f32 load of lanes 0..63 / 64..127 matches the deinterleaving bf16 load of the gate.
//  in1 qg bf16 {12288, T}   out0 core bf16 {6144, T}, natural channel order (interleaving conversion).   index space: 24 heads.  scalars: T
//  -DATT_DH=128 (plain Qwen3, GUID gaudium_attn128_post): ao f32 {128, GQ T, NKV} (lane order), no output gate (qg is not read; the
//  input stays so the node keeps one signature), core bf16 {128 NH, T} = the interleaving conversion of the f32 row.
void main(tensor ao, tensor qg, tensor core, int T)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
#if ATT_DH == 128
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            int5 ac = {0, (h % ATT_GQ) * T + t, h / ATT_GQ, 0, 0};
            float128 o1; o1.v1 = v_f32_ld_tnsr_b(ac, ao); ac[0] = 64; o1.v2 = v_f32_ld_tnsr_b(ac, ao);
            int5 cc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(o1));
        }
    }
    return;
#endif
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            int5 ac = {0, (h % ATT_GQ) * T + t, h / ATT_GQ, 0, 0};
            float128 o1, o2;
            o1.v1 = v_f32_ld_tnsr_b(ac, ao); ac[0] = 64;  o1.v2 = v_f32_ld_tnsr_b(ac, ao);
            ac[0] = 128; o2.v1 = v_f32_ld_tnsr_b(ac, ao); ac[0] = 192; o2.v2 = v_f32_ld_tnsr_b(ac, ao);
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
