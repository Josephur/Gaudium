// gaudium_cert_mark: candidate test of the certified screening readout, per vocabulary lane j (notes/engine-mtp-certified.md).
//  in0 S    f32  {V, 2M}      fp8_gemm_f32 screen: row m = hi part, row M+m = 16 x lo part, so a_jm = S[j,m] + S[j,M+m]/16
//  in1 nv   f32  {V}          n_j = max(||W_j||, ||W8_j||) rounded up        in2 ev f32 {V}  e_j = ||W_j - W8_j|| rounded up
//  in3 coef f32  {64, 8, M}   from gaudium_cert_hq (c1, c2 broadcast in every lane)
//  in4 La   f32  {2M, M}      logits of the 2M anchor tokens (argmax of each screen row) for every row m, bf16 GEMM, f32 output
//  in5 nA   f32  {1, 2M}      n_j of the anchor tokens (gathered from nv)
//  out0 f   f32  {V}          +1 if lane j is a candidate for some row m, else max_m (U_jm - tau_m) < 0
//  out1 vf  f32  {64, NV}     per 64-lane vector v (NV = V/64): max over its lanes of f, broadcast in all lanes
//  L_m = max_k (La[k,m] - 2 G ||h_m|| nA[k] (1 + 2^-7))  <= the reference accumulator S_A of that anchor (any fp32 order);
//  tau_m = pred_bf16(RD_bf16(L_m))  (RD = round toward -inf to bf16): every reference logit l_j <= pred(R) < R <= l_A.
//  These per-row thresholds are computed in scalar code (exact control of the rounding direction).  U_jm = a_jm + (c1_m n_j + c2_m e_j)(1 + 2^-9) + 2^-20 |a_jm|
//  j is excluded iff U_jm < tau_m for every m. NaN anywhere -> candidate (the comparison "< 0" fails).
//  scalars: M, NM (index-space size).  index space: NM members; member g takes vectors v = g, g + NM, ... (interleaved: spreads clustered candidates)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define MAXM 18

void main(tensor S, tensor nv, tensor ev, tensor coef, tensor La, tensor nA, tensor f, tensor vf, int M, int NM)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    const int V = get_dim_size(S, 0), NV = V / 64, K2 = 2 * M;
    float64 tau[MAXM], c1[MAXM], c2[MAXM];
    const float G2 = 2.0f * 6.103515625e-4f * 1.0078125f;
    for (int m = 0; m < M; ++m) {
        int5 hc = {0, 7, m, 0, 0}; const float nh = s_f32_ld_g(gen_addr(hc, coef));   // ||h_m|| (rounded up by cert_hq)
        float L = -3.0e38f;
        for (int k = 0; k < K2; ++k) {
            int5 lc = {k, m, 0, 0, 0}, nc = {0, k, 0, 0, 0};
            float x = s_f32_ld_g(gen_addr(lc, La)), na = s_f32_ld_g(gen_addr(nc, nA));
            float lo = x - G2 * nh * na;                     // G2 carries a 2^-7 margin over 2G (covers the rsqrt-based ||h|| and these f32 ops)
            L = lo > L ? lo : L;
        }
        int u = __builtin_astype(L, int);
        int r = u & (int)0xFFFF0000;                           // toward zero
        if (L < 0.0f && (u & 0xFFFF) != 0) r += 0x10000;        // negative: one more magnitude ulp = toward -inf
        float R = __builtin_astype(r, float);
        int pu = R > 0.0f ? r - 0x10000 : (R < 0.0f ? r + 0x10000 : (int)0x80010000);   // pred_bf16
        tau[m] = __builtin_astype(pu, float);                  // scalar broadcast into every lane
        int5 cc = {0, 0, m, 0, 0}; c1[m] = v_f32_ld_tnsr_b(cc, coef); cc[1] = 1; c2[m] = v_f32_ld_tnsr_b(cc, coef);
    }
    for (int g = idx[0]; g < idx[0] + isz[0]; ++g) for (int v = g; v < NV; v += NM) {
        int5 jc = {64 * v, 0, 0, 0, 0};
        float64 n = v_f32_ld_tnsr_b(jc, nv), e = v_f32_ld_tnsr_b(jc, ev);
        float64 best = -3.0e38f;
        for (int m = 0; m < M; ++m) {
            int5 sc = {64 * v, m, 0, 0, 0}; float64 hi = v_f32_ld_tnsr_b(sc, S); sc[1] = M + m; float64 lo = v_f32_ld_tnsr_b(sc, S);
            float64 a = hi + lo * 0.0625f;
            float64 U = a + (c1[m] * n + c2[m] * e) * 1.001953125f + v_f32_abs_b(a) * 9.5367431640625e-07f;
            float64 d = U - tau[m];
            d = v_f32_sel_less_f32_b(d, 0.0f, d, 1.0f);       // candidate (or NaN) -> +1
            best = v_f32_max_b(best, d);
        }
        v_f32_st_tnsr(jc, f, best);
        int5 vc = {0, v, 0, 0, 0};
        v_f32_st_tnsr(vc, vf, REDUCE_MAX64(best, P8, P4, P2, P1));
    }
}
