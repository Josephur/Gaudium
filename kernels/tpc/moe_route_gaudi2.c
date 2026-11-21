// gaudium_moe_route: the MoE router's top-K selection and weights for T rows (issue #8). One index-space point = one row.
//
//  in0  logits bf16 {E, T}   the router GEMM output in the model dtype (E = 128 * NB experts, NB <= 4)
//  out0 ids    i32  {K, T}   the K selected experts of each row in ASCENDING expert id (the reference accumulates the expert
//                            outputs of a row in that order)
//  out1 w      f32  {K, T}   their weights (values exactly representable in bf16), aligned with ids
//  out2 wd     bf16 {E, T}   the same weights as a dense table: wd[e, t] = weight of expert e for row t, 0 when not selected
//  out3 wd2    f32  {E/2 + 64, 2, T}   the dense table in scalar-addressable form for gaudium_moe_act: w(e, t) = wd2[e >> 1, e & 1, t].
//                            Slot [E/2, 0, t] is the weight of the SHARED expert's block: sigmoid(sgl[t]) when sgl is given, else 0.
//  out4 wb     bf16 {K, T}   the weights of out1 as bf16 (the SDK `moe` node's router_weights input)
//  in1  sgl    bf16 {1, T}   optional (the SG build): the shared-expert gate logit w_sg . h
//  scalar: K (1 .. 12)
//
// Semantics. The reference (HF Qwen3_5MoeTopKRouter / Qwen4Exp router) is: p = softmax_f32(logits) over all E, top-K of p, renormalise
// by the sum of the K, cast to the model dtype. Here:
//   selection  the K largest logits, a tie goes to the lower expert id. Softmax is monotonic, so this is the top-K of p, decided on the
//              bf16 logits themselves (no exp rounding can reorder two experts).
//   weights    p_k = exp(l_k - l_max) in f32 for the K selected (k = 0 is the largest), S = p_0 + p_1 + ... + p_{K-1} summed in that
//              order, w_k = bf16(p_k * (1 / S)), round to nearest even. The softmax normaliser over all E cancels in the
//              renormalisation and is not computed. A weight can differ from the reference's by one bf16 ulp (different exp and
//              division rounding in f32); the selection cannot, except between exactly equal logits.
// The weights are computed HERE for every consumer: device experts, host (CPU) experts and the combine kernel all read this table,
// so where an expert runs never changes its weight.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_F(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define MOVDG_I(a, s0, s1, s2, s3, inc) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
// max / min over all 64 lanes, broadcast to every lane (xor-butterfly shuffles inside each 16-lane dual group)
#define REDUCE_MAX_F(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_F(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_F(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MIN_I(a, P8, P4, P2, P1) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_min_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_min_b(_a, _r); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define NEG (-3.0e38f)
#define BIG 0x7fffffff
#define MAXV 8    // f32 vectors of one row's logits: E <= 512
#define MAXK 12

#ifdef SG
void main(tensor logits, tensor sgl, tensor ids, tensor w, tensor wd, tensor wd2, tensor wb, int K)
#else
void main(tensor logits, tensor ids, tensor w, tensor wd, tensor wd2, tensor wb, int K)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    const int NB = get_dim_size(logits, 0) / 128;       // bf16 vectors per row; each converts to two f32 vectors (even / odd channels)
    float64 X[MAXV]; int64 ID[MAXV]; float64 WV[MAXK]; int64 WI[MAXK];
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        for (int b = 0; b < NB; ++b) {
            int5 xc = {128 * b, t, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, logits));
            X[2 * b] = x.v1; X[2 * b + 1] = x.v2;
            ID[2 * b] = lane * 2 + 128 * b; ID[2 * b + 1] = ID[2 * b] + 1;        // expert id of each lane
        }
        // the K largest, in descending order; ties to the lower id
        float64 M0 = 0.f, S = 0.f;
        for (int k = 0; k < K; ++k) {
            float64 mx = X[0];
            for (int v = 1; v < 2 * NB; ++v) mx = v_f32_max_b(mx, X[v]);
            const float64 m = REDUCE_MAX_F(mx, P8, P4, P2, P1);
            int64 cand = v_i32_sel_eq_f32_b(X[0], m, ID[0], BIG);
            for (int v = 1; v < 2 * NB; ++v) cand = v_i32_min_b(cand, v_i32_sel_eq_f32_b(X[v], m, ID[v], BIG));
            const int64 wi = REDUCE_MIN_I(cand, P8, P4, P2, P1);
            for (int v = 0; v < 2 * NB; ++v) X[v] = v_f32_sel_eq_i32_b(ID[v], wi, NEG, X[v]);
            if (k == 0) M0 = m;
            const float64 p = v_exp_f32(m - M0);
            S = S + p; WV[k] = p; WI[k] = wi;
        }
        const float64 invS = v_reciprocal_f32(S);
        for (int k = 0; k < K; ++k) {     // round to bf16 once; every consumer reads this value
            float128 pr; pr.v1 = WV[k] * invS; pr.v2 = pr.v1;
            WV[k] = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pr)).v1;
        }
        // dense table
        for (int b = 0; b < NB; ++b) {
            float128 d; d.v1 = 0.f; d.v2 = 0.f;
            for (int k = 0; k < K; ++k) { d.v1 = v_f32_sel_eq_i32_b(ID[2 * b], WI[k], WV[k], d.v1); d.v2 = v_f32_sel_eq_i32_b(ID[2 * b + 1], WI[k], WV[k], d.v2); }
            int5 oc = {128 * b, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, wd, v_convert_f32_to_bf16_all_b(d));
            int5 o2 = {64 * b, 0, t, 0, 0};                    // lane j of d.v1 is expert 128 b + 2 j: slot [64 b + j, 0]; d.v2: the odd experts, [.., 1]
            v_f32_st_tnsr(o2, wd2, d.v1); o2[1] = 1; v_f32_st_tnsr(o2, wd2, d.v2);
        }
        {   // the shared expert's block weight in slot [E/2, 0] (the 64 slots from E/2 are padding, so the vector store stays in bounds)
            int5 o2 = {64 * NB, 0, t, 0, 0}; float64 sg = 0.f;
#ifdef SG
            int5 sc = {0, t, 0, 0, 0};
            float128 sx = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, sgl));   // lane 0 of v1 = sgl[0, t]; the other lanes are padding
            sg = v_sigmoid_f32(sx.v1);
#endif
            v_f32_st_tnsr(o2, wd2, sg); o2[1] = 1; const float64 z0 = 0.f; v_f32_st_tnsr(o2, wd2, z0);
        }
        // the K (id, weight) pairs in ascending id: K selection passes over the K broadcast vectors
        for (int j = 0; j < K; ++j) {
            int64 cur = WI[0];
            for (int k = 1; k < K; ++k) cur = v_i32_min_b(cur, WI[k]);
            float64 ws = 0.f;
            for (int k = 0; k < K; ++k) ws = v_f32_sel_eq_i32_b(WI[k], cur, WV[k], ws);
            for (int k = 0; k < K; ++k) WI[k] = v_i32_sel_eq_i32_b(WI[k], cur, BIG, WI[k]);
            int5 oc = {j, t, 0, 0, 0};
            v_i32_st_tnsr(oc, ids, cur); v_f32_st_tnsr(oc, w, ws);     // a full-vector store clipped at the row's end; j ascending, so entry j is final
            { float128 wp; wp.v1 = ws; wp.v2 = ws; v_bf16_st_tnsr(oc, wb, v_convert_f32_to_bf16_all_b(wp)); }
        }
    }
}
