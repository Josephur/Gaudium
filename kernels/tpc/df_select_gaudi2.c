// gaudium_df_score / gaudium_df_walk: the DFlash2 candidate selector.
//  score (index space S = T-1 steps): scores[c, p, s] = unary[c, s+1] + bf16( sum_r bf16(pred[prev(p,s)][r] * h[s+1][r]) * succ[cand[c, s+1]][r] )
//    prev(p, 0) = anchor = ids_blk[0]; prev(p, s) = cand[p, s].   (the reference rounds the product and the dot to bf16; unary is f32)
//  in0 cand i32 {16, T}  in1 unary f32 {16, T}  in2 hproj bf16 {256, T}  in3 pred bf16 {256, V}  in4 succ bf16 {256, V}  in5 ids_blk i32 {T}
//  out0 scores f32 {16, 16, S}
//  walk (1 TPC): greedy lattice walk, ties -> lowest candidate index. drafts[s] = cand[c*, s+1]; alts[s] = the runner-up of the same
//  row (the rank-2 candidate given the chain prefix; a tree leaf); alts2[s] = rank 3.
//  in0 scores f32 {16, 16, S}  in1 cand i32 {16, T}  -> out0 drafts i32 {S}  out1 alts i32 {S}  out2 alts2 i32 {S}
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_F(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define MOVDG_I(a, s0, s1, s2, s3, inc) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_F(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_F(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_F(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
#define REDUCE_MAX_F(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_F(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_F(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MIN_I(a, P8, P4, P2, P1) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_min_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_min_b(_a, _r); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MAX_I(a, P8, P4, P2, P1) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_max_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_max_b(_a, _r); \
    _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
static inline float64 round_bf16(float64 x) { float128 t; t.v1 = x; t.v2 = x; return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(t)).v1; }
#define NEG (-3.0e38f)

#ifdef WALK
void main(tensor scores, tensor cand, tensor drafts, tensor alts, tensor alts2)
{
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    const int S = get_dim_size(scores, 2);
    int64 prev = 0;
    for (int s = 0; s < S; ++s) {
        float64 R = NEG;
        for (int p = 0; p < 16; ++p) { int5 c = {0, p, s, 0, 0}; float64 row = v_f32_ld_tnsr_b(c, scores); R = v_f32_sel_eq_i32_b(prev, p, row, R); }
        R = v_f32_sel_less_i32_b(lane, 16, R, NEG);
        int5 cc = {0, s + 1, 0, 0, 0};
        int64 cv = v_i32_ld_tnsr_b(cc, cand);
        int64 w = 0;
        for (int rank = 0; rank < 3; ++rank) {
            float64 m = REDUCE_MAX_F(R, P8, P4, P2, P1);
            int64 ix = v_i32_sel_eq_f32_b(R, m, lane, 0x7fffffff);
            w = REDUCE_MIN_I(ix, P8, P4, P2, P1);
            int64 tok = REDUCE_MAX_I(v_i32_sel_eq_i32_b(lane, w, cv, -1), P8, P4, P2, P1);
            int5 oc = {s, 0, 0, 0, 0};
            if (rank == 0) { v_i32_st_tnsr(oc, drafts, tok); prev = w; }
            else if (rank == 1) v_i32_st_tnsr(oc, alts, tok);
            else v_i32_st_tnsr(oc, alts2, tok);
            R = v_f32_sel_eq_i32_b(lane, w, NEG, R);
        }
    }
}
#else
void main(tensor cand, tensor unary, tensor hproj, tensor pred, tensor succ, tensor ids_blk, tensor scores)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0};
    const int anchor = s_i32_ld_g(gen_addr(zc, ids_blk));
    for (int s = idx[0]; s < idx[0] + isz[0]; ++s) {
        const int row = s + 1;
        int5 hc = {0, row, 0, 0, 0};
        float128 ha = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(hc, hproj)); hc[0] = 128;
        float128 hb = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(hc, hproj));
        int5 uc = {0, row, 0, 0, 0};
        float64 un = v_f32_ld_tnsr_b(uc, unary);
        for (int p = 0; p < 16; ++p) {
            int prev = anchor;
            if (s > 0) { int5 pc = {p, s, 0, 0, 0}; prev = s_i32_ld_g(gen_addr(pc, cand)); }
            int5 rc = {0, prev, 0, 0, 0};
            float128 pa = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, pred)); rc[0] = 128;
            float128 pb = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, pred));
            float64 w1 = round_bf16(pa.v1 * ha.v1), w2 = round_bf16(pa.v2 * ha.v2), w3 = round_bf16(pb.v1 * hb.v1), w4 = round_bf16(pb.v2 * hb.v2);
            float64 acc = 0.f;
            for (int c = 0; c < 16; ++c) {
                int5 cc = {c, row, 0, 0, 0};
                const int cid = s_i32_ld_g(gen_addr(cc, cand));
                int5 sc = {0, cid, 0, 0, 0};
                float128 sa = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, succ)); sc[0] = 128;
                float128 sb = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, succ));
                float64 e = w1 * sa.v1; e = v_f32_mac_b(w2, sa.v2, e); e = v_f32_mac_b(w3, sb.v1, e); e = v_f32_mac_b(w4, sb.v2, e);
                e = round_bf16(REDUCE_F(e, P8, P4, P2, P1));
                acc = v_f32_sel_eq_i32_b(lane, c, e, acc);
            }
            int5 oc = {0, p, s, 0, 0};
            v_f32_st_tnsr(oc, scores, un + acc);
        }
    }
}
#endif
