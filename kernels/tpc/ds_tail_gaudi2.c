// gaudium_ds_base / gaudium_ds_score: DSpine's serial last-write refinement (code_bridge_second_pass = serial_table), greedy form.
// For proposal row t (block rows 1..T-1, s = t - 1), candidate k (top-16 of the row's logits) and sender lane j (the candidate the
// previous row chose; the anchor for t = 1):
//   score[k, j, s] = base[k, t] + rms_t * sum_r sigmoid(q_t[r] + TSV[sid_j][r] + b[r]) * TSV[sid_j][R + r] * TG[cand_{k,t}][r]
//   base[k, t] = sum_h raw_t[h] w_norm[h] lm_head[cand_{k,t}][h]    (f32: the last write's pre-write state through the head)
// = the argmax-equivalent logits of the row refined with the EXACT code of the sender token (docs/notes/dspine-spec.md section 2).
// The greedy walk over scores is gaudium_df_walk (same {16, 16, S} layout as DFlash2's selector).
//  BASE   in0 raw bf16 {H, T}, in1 w_norm f32 {H} (lane order), in2 cand i32 {16, T}, in3 W_lm bf16 {H, V}  -> out0 base f32 {16, T}
//         index space S = T - 1 (point s -> row s + 1); scalar H
//  SCORE  in0 cand i32 {16, T}, in1 base f32 {16, T}, in2 packed bf16 {3R, T} (q = channels 0..R-1), in3 rms f32 {64, T},
//         in4 gate bias f32 {R} (lane order), in5 TSV bf16 {2R, V}, in6 TG bf16 {R, V}, in7 ids_blk i32 {T}  -> out0 scores f32 {16, 16, S}; R <= 512
//         index space 16 S (point p -> s = p / 16, sender lane j = p % 16); scalar R
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define K 16
#define NEG (-3.0e38f)

#if defined(BASE)
void main(tensor raw, tensor nw, tensor cand, tensor lm, tensor base, int H)
{
    const int5 idx = get_index_space_offset(); const int5 isz = get_index_space_size();
    const uchar256 l16 = (read_lane_id_1b_b() >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    for (int s = idx[0]; s < idx[0] + isz[0]; ++s) {
        const int t = s + 1;
        float64 out = NEG;
        for (int k = 0; k < K; ++k) {
            int5 cc = {k, t, 0, 0, 0}; const int cid = s_i32_ld_g(gen_addr(cc, cand));
            float64 e = 0.f;
            for (int h = 0; h < H; h += 128) {
                int5 rcd = {h, t, 0, 0, 0}, lc = {h, cid, 0, 0, 0}, wc = {h, 0, 0, 0, 0};
                float128 r = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rcd, raw)), w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(lc, lm));
                float64 n1 = v_f32_ld_tnsr_b(wc, nw); wc[0] = h + 64; float64 n2 = v_f32_ld_tnsr_b(wc, nw);
                e = v_f32_mac_b(r.v1 * n1, w.v1, e); e = v_f32_mac_b(r.v2 * n2, w.v2, e);
            }
            out = v_f32_sel_eq_i32_b(lane, k, REDUCE64(e, P8, P4, P2, P1), out);
        }
        int5 oc = {0, t, 0, 0, 0}; v_f32_st_tnsr(oc, base, out);
    }
}
#elif defined(SCORE)
void main(tensor cand, tensor base, tensor pk, tensor rms, tensor gb, tensor tsv, tensor tg, tensor ids, tensor scores, int R)
{
    const int5 idx = get_index_space_offset(); const int5 isz = get_index_space_size();
    const uchar256 l16 = (read_lane_id_1b_b() >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0}; const int anchor = s_i32_ld_g(gen_addr(zc, ids));
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        const int s = p / K, j = p % K, t = s + 1;
        int sid = anchor;
        if (s > 0) { int5 pc = {j, t - 1, 0, 0, 0}; sid = s_i32_ld_g(gen_addr(pc, cand)); }
        // m = sigmoid(q_t + TSV_g[sid] + b) * TSV_v[sid]: R = 512 -> 4 blocks of 128 (deinterleaved v1 / v2)
        float64 m0a = 0.f, m0b = 0.f, m1a = 0.f, m1b = 0.f, m2a = 0.f, m2b = 0.f, m3a = 0.f, m3b = 0.f;   // R <= 512: blocks past R stay 0
#define MBLK(i, ma, mb) if (128 * (i) < R) { const int r0 = 128 * (i); int5 qc = {r0, t, 0, 0, 0}, gc = {r0, sid, 0, 0, 0}, vc = {R + r0, sid, 0, 0, 0}, bc = {r0, 0, 0, 0, 0}; \
            float128 q = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc, pk)), g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, tsv)), v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vc, tsv)); \
            float64 b1 = v_f32_ld_tnsr_b(bc, gb); bc[0] = r0 + 64; float64 b2 = v_f32_ld_tnsr_b(bc, gb); \
            ma = v_sigmoid_f32(q.v1 + g.v1 + b1) * v.v1; mb = v_sigmoid_f32(q.v2 + g.v2 + b2) * v.v2; }
        MBLK(0, m0a, m0b) MBLK(1, m1a, m1b) MBLK(2, m2a, m2b) MBLK(3, m3a, m3b)
        float64 acc = 0.f;
        for (int k = 0; k < K; ++k) {
            int5 cc = {k, t, 0, 0, 0}; const int cid = s_i32_ld_g(gen_addr(cc, cand));
            int5 tc = {0, cid, 0, 0, 0};
            float128 g0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(tc, tg));
            float64 e = m0a * g0.v1; e = v_f32_mac_b(m0b, g0.v2, e);
            if (R > 128) { tc[0] = 128; float128 g1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(tc, tg)); e = v_f32_mac_b(m1a, g1.v1, e); e = v_f32_mac_b(m1b, g1.v2, e); }
            if (R > 256) { tc[0] = 256; float128 g2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(tc, tg)); e = v_f32_mac_b(m2a, g2.v1, e); e = v_f32_mac_b(m2b, g2.v2, e); }
            if (R > 384) { tc[0] = 384; float128 g3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(tc, tg)); e = v_f32_mac_b(m3a, g3.v1, e); e = v_f32_mac_b(m3b, g3.v2, e); }
            acc = v_f32_sel_eq_i32_b(lane, k, REDUCE64(e, P8, P4, P2, P1), acc);
        }
        int5 bc2 = {0, t, 0, 0, 0}; float64 bv = v_f32_ld_tnsr_b(bc2, base), rv = v_f32_ld_tnsr_b(bc2, rms);
        float64 sc = v_f32_mac_b(rv, acc, bv);
        sc = v_f32_sel_less_i32_b(lane, K, sc, NEG);
        int5 oc = {0, j, s, 0, 0}; v_f32_st_tnsr(oc, scores, sc);
    }
}
#endif
