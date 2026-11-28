// gaudium_ik_route_r: routed Inkling 2 top-K selection and weights for T rows (ascending expert id order).
//   score_e = sigmoid(logit_e);   selection: the K largest of score_e + bias_e, tie -> lower expert id
//   w_k = score_k / (sum of the K scores + 1e-20) * route_scale
//  in0  logits bf16 {E, T}   router GEMM output (E = 128 * NB experts, NB <= 4)
//  in1  rb     f32  {E}      split layout: expert selection bias
//  out0 ids    i32  {K, T}   K selected expert ids per row in ascending order
//  out1 w      f32  {K, T}   corresponding router weights (same numerics as gaudium_ik_route)
//  index space {T}.   scalars: K (1 .. 12), route_scale (f32 bits)
#include "ik_common.h"
#define MOVDG_I(a, s0, s1, s2, s3, inc) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MIN_I(a, P8, P4, P2, P1) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_min_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_min_b(_a, _r); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
#define NEG (-3.0e38f)
#define BIG 0x7fffffff
#define MAXV 8
#define MAXK 12

void main(tensor logits, tensor rb, tensor ids, tensor w, int K, int rscale_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int64 lane = (int64)read_lane_id_4b_b();
    const float rscale = *(float*)&rscale_bits;
    const int NB = get_dim_size(logits, 0) / 128;
    float64 X[MAXV], SC[MAXV]; int64 ID[MAXV]; float64 WV[MAXK]; int64 WI[MAXK];
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        for (int b = 0; b < NB; ++b) {
            int5 xc = {128 * b, t, 0, 0, 0}, bc = {128 * b, 0, 0, 0, 0};
            const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, logits));
            SC[2 * b] = v_sigmoid_f32(x.v1); SC[2 * b + 1] = v_sigmoid_f32(x.v2);
            X[2 * b] = SC[2 * b] + v_f32_ld_tnsr_b(bc, rb); bc[0] += 64;
            X[2 * b + 1] = SC[2 * b + 1] + v_f32_ld_tnsr_b(bc, rb);
            ID[2 * b] = lane * 2 + 128 * b; ID[2 * b + 1] = ID[2 * b] + 1;
        }
        float64 S = 0.f;
        for (int k = 0; k < K; ++k) {
            float64 mx = X[0];
            for (int v = 1; v < 2 * NB; ++v) mx = v_f32_max_b(mx, X[v]);
            const float64 m = REDUCE_MAX64(mx, P8, P4, P2, P1);
            int64 cand = v_i32_sel_eq_f32_b(X[0], m, ID[0], BIG);
            for (int v = 1; v < 2 * NB; ++v) cand = v_i32_min_b(cand, v_i32_sel_eq_f32_b(X[v], m, ID[v], BIG));
            const int64 wi = REDUCE_MIN_I(cand, P8, P4, P2, P1);
            float64 sv = v_f32_sel_eq_i32_b(ID[0], wi, SC[0], -1.0f);
            for (int v = 1; v < 2 * NB; ++v) sv = v_f32_max_b(sv, v_f32_sel_eq_i32_b(ID[v], wi, SC[v], -1.0f));
            sv = REDUCE_MAX64(sv, P8, P4, P2, P1);
            for (int v = 0; v < 2 * NB; ++v) X[v] = v_f32_sel_eq_i32_b(ID[v], wi, NEG, X[v]);
            S = S + sv; WV[k] = sv; WI[k] = wi;
        }
        const float64 f = RECIP(S + 1.0e-20f) * rscale;
        int64 ov_id = 0; float64 ov_w = 0.f;
        for (int j = 0; j < K; ++j) {
            int64 cur = WI[0];
            for (int k = 1; k < K; ++k) cur = v_i32_min_b(cur, WI[k]);
            float64 ws = 0.f;
            for (int k = 0; k < K; ++k) ws = v_f32_sel_eq_i32_b(WI[k], cur, WV[k] * f, ws);
            for (int k = 0; k < K; ++k) WI[k] = v_i32_sel_eq_i32_b(WI[k], cur, BIG, WI[k]);
            ov_id = v_i32_sel_eq_i32_b(lane, j, cur, ov_id);
            ov_w = v_f32_sel_eq_i32_b(lane, j, ws, ov_w);
        }
        int5 oc = {0, t, 0, 0, 0};
        v_i32_st_tnsr(oc, ids, ov_id);
        v_f32_st_tnsr(oc, w, ov_w);
    }
}
