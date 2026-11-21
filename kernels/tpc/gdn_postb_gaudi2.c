// -DGATE_SIGMOID build (qwen4_exp / Qwen3.8-Flash-Next): the output gate is sigmoid(z) instead of silu(z) = z * sigmoid(z)
#ifdef GATE_SIGMOID
#define GATE(z) v_sigmoid_f32(z)
#else
#define GATE(z) ((z) * v_sigmoid_f32(z))
#endif
#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_postb: phase B of the prefill gated RMSNorm (sigmoid table only): core = (o * r) * norm_w * silu(z).
// Same arithmetic as gdn_post2_gaudi2.c phase B: bit-identical.
//  in0 o f32 {128, 2C, 48, NCH}  in1 qkvz bf16 {16384, T}  in2 norm_w f32 {128}  in3 r f32 {64, T, 48} (gaudium_gdn_posta)  out0 core bf16 {6144, T}
//  scalars: T, C, RG.  Index space {48 heads, ceil(T/RG)}.
void main(tensor o, tensor qkvz, tensor norm_w, tensor r, tensor core, int T, int CH, int RG)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 wc = {0, 0, 0, 0, 0};
    float64 wa = v_f32_ld_tnsr_b(wc, norm_w); wc[0] = 64; float64 wb = v_f32_ld_tnsr_b(wc, norm_w);
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
      for (int rg = idx[1]; rg < idx[1] + isz[1]; ++rg) {
        const int t0 = rg * RG, t1 = (t0 + RG < T) ? t0 + RG : T;
#pragma unroll 4
        for (int t = t0; t < t1; ++t) {
            int5 oc = {0, t % CH, h, t / CH, 0};
            float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
            int5 rc = {0, t, h, 0, 0};
            float64 rr = v_f32_ld_tnsr_b(rc, r);
            int5 zc = {GDN_QKV + 128 * h, t, 0, 0, 0};
            float128 z = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(zc, qkvz));
            float128 res;
            res.v1 = (oa * rr) * wa * GATE(z.v1);
            res.v2 = (ob * rr) * wb * GATE(z.v2);
            int5 cc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(res));
        }
      }
    }
}
