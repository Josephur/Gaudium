// gaudium_ik_mix: the block attention residual of an Inkling 2 layer. A row's input is a softmax mix, by learned queries, of the
// earlier block states and of the row's partial block state:
//   n_i   = x_i rsqrt(mean(x_i^2) + eps)                      each source normalised without a weight, in f32
//   a_ij  = softmax over the sources i of (q_j . n_i) / sqrt(H)
//   out_j = sum_i a_ij n_i
// Sources: the first ns of the four state inputs, then the partial state. ns and the bypass flag come from a small device tensor,
// so one recipe serves every layer of a kind (a layer is bound to its own control words at launch).
//  in0..3 s0..s3 bf16 {H, T}   the block states (inputs beyond ns are not read; bind any tensor of the shape)
//  in4 part   bf16 {H, T}  (-DPF32: f32 {H, T}, split layout)   the partial state
//  in5 query  f32  {H, NQ}  split layout
//  in6 ctl    i32  {8}: [0] ns (0..4), [1] bypass (1: out_0 = the partial state as it is; the first layer has no mix)
//  out0 out   f32  {H, T, NQ}  split layout
//  index space {T}: one member = one row.   scalars: eps (f32 bits), scale (f32 bits of H^-1/2)
#include "ik_common.h"
#define NEG (-1.0e30f)
#ifdef PF32
#define LDPART(b_, t_) ({ float128 _x; int5 _c = {128 * (b_), (t_), 0, 0, 0}; _x.v1 = v_f32_ld_tnsr_b(_c, part); _c[0] += 64; _x.v2 = v_f32_ld_tnsr_b(_c, part); _x; })
#else
#define LDPART(b_, t_) ({ int5 _c = {128 * (b_), (t_), 0, 0, 0}; v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_c, part)); })
#endif
#define LDST(s_, b_, t_) ({ int5 _c = {128 * (b_), (t_), 0, 0, 0}; v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_c, s_)); })
// accumulate source i's sum of squares and its dot products with the NQ queries
#define ACC1(i_, x_) { const float128 _x = (x_); SS[i_] = SS[i_] + _x.v1 * _x.v1 + _x.v2 * _x.v2; \
    for (int j = 0; j < NQ; ++j) D[4 * (i_) + j] = D[4 * (i_) + j] + _x.v1 * Q1[j] + _x.v2 * Q2[j]; }
#define ACC2(i_, x_) { const float128 _x = (x_); \
    for (int j = 0; j < NQ; ++j) { O1[j] = O1[j] + _x.v1 * D[4 * (i_) + j]; O2[j] = O2[j] + _x.v2 * D[4 * (i_) + j]; } }

void main(tensor s0, tensor s1, tensor s2, tensor s3, tensor part, tensor query, tensor ctl, tensor out, int eps_bits, int scale_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int H = get_dim_size(out, 0), NB = H / 128, NQ = get_dim_size(out, 2);
    const float eps = *(float*)&eps_bits, scale = *(float*)&scale_bits, invh = 1.0f / (float)H;
    int5 zc = {0, 0, 0, 0, 0};
    const int ns = s_i32_ld_g(gen_addr(zc, ctl));
    zc[0] = 1;
    const int bypass = s_i32_ld_g(gen_addr(zc, ctl));
    float64 SS[5], D[20], Q1[4], Q2[4], O1[4], O2[4], R[5], SC[5];   // 51 vectors of local memory (the unit has 64)
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        if (bypass) {
            for (int b = 0; b < NB; ++b) {
                const float128 x = LDPART(b, t);
                int5 oc = {128 * b, t, 0, 0, 0};
                v_f32_st_tnsr(oc, out, x.v1); oc[0] += 64; v_f32_st_tnsr(oc, out, x.v2);
            }
            continue;
        }
        for (int i = 0; i < 5; ++i) SS[i] = 0.f;
        for (int i = 0; i < 20; ++i) D[i] = 0.f;
        for (int b = 0; b < NB; ++b) {
            for (int j = 0; j < NQ; ++j) {
                int5 qc = {128 * b, j, 0, 0, 0};
                Q1[j] = v_f32_ld_tnsr_b(qc, query); qc[0] += 64; Q2[j] = v_f32_ld_tnsr_b(qc, query);
            }
            if (ns > 0) ACC1(0, LDST(s0, b, t))
            if (ns > 1) ACC1(1, LDST(s1, b, t))
            if (ns > 2) ACC1(2, LDST(s2, b, t))
            if (ns > 3) ACC1(3, LDST(s3, b, t))
            ACC1(4, LDPART(b, t))
        }
        // a source's 1 / rms; sources 0..ns-1 and the partial state (index 4) take part
        for (int i = 0; i < 5; ++i) R[i] = (i < ns || i == 4) ? v_rsqrt_f32(REDUCE64(SS[i], P8, P4, P2, P1) * invh + eps) : (float64)0.f;
        for (int j = 0; j < NQ; ++j) {
            float64 m = NEG;
            for (int i = 0; i < 5; ++i) {
                if (i < ns || i == 4) { SC[i] = REDUCE64(D[4 * i + j], P8, P4, P2, P1) * R[i] * scale; m = v_f32_max_b(m, SC[i]); }
            }
            float64 S = 0.f;
            for (int i = 0; i < 5; ++i) {
                if (i < ns || i == 4) { SC[i] = EXPD(SC[i] - m); S = S + SC[i]; }
            }
            const float64 inv = RECIP(S);
            // the dot products of query j have been used: their places take the sources' coefficients
            for (int i = 0; i < 5; ++i) D[4 * i + j] = (i < ns || i == 4) ? SC[i] * inv * R[i] : (float64)0.f;
        }
        for (int b = 0; b < NB; ++b) {
            for (int j = 0; j < NQ; ++j) { O1[j] = 0.f; O2[j] = 0.f; }
            if (ns > 0) ACC2(0, LDST(s0, b, t))
            if (ns > 1) ACC2(1, LDST(s1, b, t))
            if (ns > 2) ACC2(2, LDST(s2, b, t))
            if (ns > 3) ACC2(3, LDST(s3, b, t))
            ACC2(4, LDPART(b, t))
            for (int j = 0; j < NQ; ++j) {
                int5 oc = {128 * b, t, j, 0, 0};
                v_f32_st_tnsr(oc, out, O1[j]); oc[0] += 64; v_f32_st_tnsr(oc, out, O2[j]);
            }
        }
    }
}
