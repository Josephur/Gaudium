// gaudium_ik_mstep: one step of the block drafter's recurrent ("Markov") head:
//   raw = bf16( ctxp[:, col] + swp + rb )                     [gate | candidate | latent], RK channels each, f32 sum, one rounding
//   g = bf16(sigmoid(raw_g));  c = bf16(tanh(raw_c));  s' = bf16( g s + (1 - g) c );  lat = bf16( tanh(raw_l) )
//  in0 ctxp f32 {3 RK, TR}: the part of the projection from the block row and the Markov context (column col)
//  in1 swp f32 {3 RK, 1}: the part from [state | the previous token's row]   in2 rb f32 {3 RK}: the bias
//      all three in the split layout (the loader orders the projection's output channels so)
//  in3 s bf16 {RK, 1}: the previous state, the checkpoint's order
//  out0 s' bf16 {RK, 1}   out1 lat bf16 {RK, 1}.   index space {RK / 128}.   scalar: col
#include "ik_common.h"
#define LD2F(t_, ch_, c1_) ({ float128 _x; int5 _c = {(ch_), (c1_), 0, 0, 0}; _x.v1 = v_f32_ld_tnsr_b(_c, t_); _c[0] += 64; _x.v2 = v_f32_ld_tnsr_b(_c, t_); _x; })
#define RAW(ch_) ({ const float128 _a = LD2F(ctxp, (ch_), col), _b = LD2F(swp, (ch_), 0), _r = LD2F(rb, (ch_), 0); \
    RBF(_a.v1 + _b.v1 + _r.v1, _a.v2 + _b.v2 + _r.v2); })

void main(tensor ctxp, tensor swp, tensor rb, tensor s, tensor s_new, tensor lat, int col)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int RK = get_dim_size(s, 0);
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
        const float128 rg = RAW(128 * b), rc = RAW(RK + 128 * b), rl = RAW(2 * RK + 128 * b);
        const float128 g = RBF(v_sigmoid_f32(rg.v1), v_sigmoid_f32(rg.v2));
        const float128 c = RBF(TANH(rc.v1), TANH(rc.v2));
        int5 sc = {128 * b, 0, 0, 0, 0};
        const float128 sp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, s));
        float128 sn; sn.v1 = g.v1 * sp.v1 + (1.0f - g.v1) * c.v1; sn.v2 = g.v2 * sp.v2 + (1.0f - g.v2) * c.v2;
        v_bf16_st_tnsr(sc, s_new, v_convert_f32_to_bf16_all_b(sn));
        float128 l; l.v1 = TANH(rl.v1); l.v2 = TANH(rl.v2);
        v_bf16_st_tnsr(sc, lat, v_convert_f32_to_bf16_all_b(l));
    }
}
