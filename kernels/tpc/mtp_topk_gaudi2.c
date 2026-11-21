// gaudium_mtp_topk1 / gaudium_mtp_topk2 (port of feature/dflash2 df_topk with K = KTOP, default 4): top-KTOP (value, index) per row of the drafter's bf16 logits {V, T}, V = 128 * VB,
//  plus the drafter's softmax PROBABILITY of each candidate: stage 1 also writes per-lane sum exp(x - lane max) over its
//  chunk (ps {64, CH, T}; the lane max is the head of its top list), stage 2 combines them into the row's normaliser and writes
//  probs {KTOP, T} = exp(v - M) / S (exact softmax of the bf16 logits up to f32 rounding). Used by the device tree planner.
//  Stage 1 (index space CH*T, point p -> chunk p % CH, row p / CH): each lane keeps a sorted top-16 of its stride over the
//  chunk's vectors (ties: lower index first), written as pv f32 {64, 16, CH, T}, pi i32 {64, 16, CH, T}.
//  Stage 2 (index space T): per row, merge the CH*16 candidate vectors into per-lane top-16 lists, then extract the 16 best
//  across lanes (max value, lowest index on ties) -> ids i32 {16, T}, vals f32 {16, T}.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_F(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define MOVDG_I(a, s0, s1, s2, s3, inc) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX_F(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_F(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_F(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MIN_I(a, P8, P4, P2, P1) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_min_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_min_b(_a, _r); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_SUM_F(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_F(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_F(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#ifndef KTOP
#define KTOP 4
#endif
#define K KTOP
#define NEG (-3.0e38f)

// insert (x, xi) into the per-lane descending lists L/I (strictly greater displaces: ties keep the earlier entry)
#define INSERT(x, xi) do { float64 _x = (x); int64 _xi = (xi); \
    for (int _i = 0; _i < K; ++_i) { \
        float64 _nv = v_f32_sel_grt_f32_b(_x, L[_i], _x, L[_i]); int64 _ni = v_i32_sel_grt_f32_b(_x, L[_i], _xi, I[_i]); \
        float64 _ov = v_f32_sel_grt_f32_b(_x, L[_i], L[_i], _x); int64 _oi = v_i32_sel_grt_f32_b(_x, L[_i], I[_i], _xi); \
        L[_i] = _nv; I[_i] = _ni; _x = _ov; _xi = _oi; } } while (0)

#ifdef STAGE2
void main(tensor pv, tensor pi, tensor ps, tensor ids, tensor vals, tensor probs, int CH)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    float64 L[K]; int64 I[K];
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        for (int i = 0; i < K; ++i) { L[i] = NEG; I[i] = -1; }
        float64 MX = NEG;
        for (int c = 0; c < CH; ++c) {
            int5 c0 = {0, 0, c, t, 0}; MX = v_f32_max_b(MX, v_f32_ld_tnsr_b(c0, pv));
            for (int i = 0; i < K; ++i) {
                int5 cc = {0, i, c, t, 0};
                float64 x = v_f32_ld_tnsr_b(cc, pv); int64 xi = v_i32_ld_tnsr_b(cc, pi);
                INSERT(x, xi);
            }
        }
        const float64 M = REDUCE_MAX_F(MX, P8, P4, P2, P1);   // row max (all lanes)
        float64 SS = 0.f;
        for (int c = 0; c < CH; ++c) { int5 c0 = {0, 0, c, t, 0}; int5 s0 = {0, c, t, 0, 0};
            float64 lm = v_f32_ld_tnsr_b(c0, pv), ls = v_f32_ld_tnsr_b(s0, ps);
            SS = SS + v_f32_sel_grt_f32_b(lm, NEG, ls * v_exp_f32(lm - M), 0.f); }
        const float64 S = REDUCE_SUM_F(SS, P8, P4, P2, P1);
        const float64 invS = v_reciprocal_f32(S);
        for (int k = 0; k < K; ++k) {
            float64 m = REDUCE_MAX_F(L[0], P8, P4, P2, P1);
            int64 cand = v_i32_sel_eq_f32_b(L[0], m, I[0], 0x7fffffff);
            int64 w = REDUCE_MIN_I(cand, P8, P4, P2, P1);
            int5 oc = {k, t, 0, 0, 0};
            v_i32_st_tnsr(oc, ids, w); v_f32_st_tnsr(oc, vals, m); v_f32_st_tnsr(oc, probs, v_exp_f32(m - M) * invS);
            int64 I0 = I[0];
            for (int i = 0; i < K - 1; ++i) { L[i] = v_f32_sel_eq_i32_b(I0, w, L[i + 1], L[i]); I[i] = v_i32_sel_eq_i32_b(I0, w, I[i + 1], I[i]); }
            L[K - 1] = v_f32_sel_eq_i32_b(I0, w, NEG, L[K - 1]); I[K - 1] = v_i32_sel_eq_i32_b(I0, w, -1, I[K - 1]);
        }
    }
}
#else
void main(tensor logits, tensor pv, tensor pi, tensor ps, int CH, int VB)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    float64 L[K]; int64 I[K];
    const int nv = (VB + CH - 1) / CH;
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        const int c = p % CH, t = p / CH;
        int v1 = (c + 1) * nv; if (v1 > VB) v1 = VB;
        for (int i = 0; i < K; ++i) { L[i] = NEG; I[i] = -1; }
        for (int vi = c * nv; vi < v1; ++vi) {
            int5 xc = {128 * vi, t, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, logits));
            int64 base = lane * 2 + 128 * vi;
            INSERT(x.v1, base);
            INSERT(x.v2, base + 1);
        }
        for (int i = 0; i < K; ++i) { int5 oc = {0, i, c, t, 0}; v_f32_st_tnsr(oc, pv, L[i]); v_i32_st_tnsr(oc, pi, I[i]); }
        // per-lane sum of exp(x - lane max) over the chunk (second sweep over the chunk's vectors; empty lanes keep 0)
        float64 sacc = 0.f; const float64 lmax = L[0];
        for (int vi = c * nv; vi < v1; ++vi) {
            int5 xc = {128 * vi, t, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, logits));
            sacc = sacc + v_exp_f32(x.v1 - lmax) + v_exp_f32(x.v2 - lmax);
        }
        { int5 sc = {0, c, t, 0, 0}; v_f32_st_tnsr(sc, ps, sacc); }
    }
}
#endif
