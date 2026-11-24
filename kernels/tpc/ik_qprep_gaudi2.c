// gaudium_ik_qprep: the query side of an Inkling 2 attention layer for T rows.
//   query heads (NH = NKV GQ, 128 channels):  q = bf16( clip( q w_q , thr_q ) ) [ bf16( . tau(pos) ) in the global layers ]
//   relative states (NH heads of 16 values):   r = bf16( clip( r w_r , thr_r ) ) [ bf16( . tau(pos) ) ]
//   clip(x, thr) = x min(1, thr / (|x| + 1e-6)) over the head's values;  tau = 1 + alpha log(max((pos + 1) / floor, 1)), a table
// There is no rotary position code in this family: position enters through the relative states, tau and the short convolutions.
//  in0 qx  bf16 {QW, T}: channels [0, 128 NH) the query projection, [roff, roff + 16 NH) the relative-state projection
//  in1 qsc f32 {128, NH} split layout    in2 rsc f32 {128, NH / 8} split layout (a block of 128 = 8 heads of 16)
//  in3 tau f32 {Lpos}                    in4 cursor i32 {1}
//  out0 qb bf16 {128, GQ T, NKV nch}: row hl T + t of batch ch NKV + g holds query head GQ g + hl (written once per chunk ch)
//  out1 rs bf16 {128, (NH / 8) T}: row grp T + t holds the relative states of heads 8 grp .. 8 grp + 7 at channels 16 h' + c
//  A head's 16 relative values sit at 8 even and 8 odd positions of the block: after the converting load they are an aligned group of
//  8 lanes in v1 and the same group in v2, so the head's sum of squares is REDUCE8 of v1^2 + v2^2.
//  index space {NH + NH / 8}: the query heads, then the groups of relative states.
//  scalars: T, coff, nch, NKV, GQ, roff, logscale (1: multiply by tau), thr_q (f32 bits), thr_r (f32 bits)
#include "ik_common.h"

void main(tensor qx, tensor qsc, tensor rsc, tensor tau, tensor cursor, tensor qb, tensor rs,
          int T, int coff, int nch, int NKV, int GQ, int roff, int logscale, int thrq_bits, int thrr_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int NH = NKV * GQ;
    const float thrq = *(float*)&thrq_bits, thrr = *(float*)&thrr_bits;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int is_q = (i < NH);
        const int h = is_q ? i : i - NH;
        int5 wc = {0, h, 0, 0, 0};
        float64 w1, w2;
        if (is_q) { w1 = v_f32_ld_tnsr_b(wc, qsc); wc[0] = 64; w2 = v_f32_ld_tnsr_b(wc, qsc); }
        else      { w1 = v_f32_ld_tnsr_b(wc, rsc); wc[0] = 64; w2 = v_f32_ld_tnsr_b(wc, rsc); }
        for (int t = 0; t < T; ++t) {
            int5 xc = {is_q ? 128 * h : roff + 128 * h, t, 0, 0, 0};
            const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qx));
            const float64 a1 = x.v1 * w1, a2 = x.v2 * w2;
            const float64 sq = a1 * a1 + a2 * a2;
            float64 f;
            if (is_q) f = CLIPF(REDUCE64(sq, P8, P4, P2, P1), thrq);
            else      f = CLIPF(REDUCE8(sq, P4, P2, P1), thrr);
            float128 o = RBF(a1 * f, a2 * f);
            if (logscale) {
                int5 tc = {cur + t, 0, 0, 0, 0};
                const float64 tv = s_f32_ld_g(gen_addr(tc, tau));
                o.v1 = o.v1 * tv; o.v2 = o.v2 * tv;
            }
            const bfloat128 ob = v_convert_f32_to_bf16_all_b(o);
            if (is_q) {
                for (int ch = 0; ch < nch; ++ch) {
                    int5 oc = {0, (h % GQ) * T + t, ch * NKV + (h / GQ), 0, 0};
                    v_bf16_st_tnsr(oc, qb, ob);
                }
            } else {
                int5 oc = {0, h * T + t, 0, 0, 0};
                v_bf16_st_tnsr(oc, rs, ob);
            }
        }
    }
}
