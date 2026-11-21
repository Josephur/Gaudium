// gaudium_df_softmax: DFlash2 block attention softmax over two key sets: the context cache (positions [0, cur), sliding window
// W: p >= cur + t - (W-1), with the tree fix: slot fix[1] excluded, slot fix[0] included) and the block itself (keys 0..T-1,
// bidirectional; -DCAUSAL (gaudium_ds_softmax, the DSpine drafter): keys 0..t, causal inside the block). Row r of group g: t = r % T is the query row at position cur + t.
//  in0 sc f32 {Lmax, R, 8}  in1 sb f32 {128, R, 8}  in2 cur int32 {1}  in3 fix int32 {2}
//  out0 Pc bf16 {Lmax, R, 8}  out1 Pb bf16 {128, R, 8} (interleaved pairs, matched by the V row permutation)
//  index space: R*8.  scalars: T, Lmax, W
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
// context mask: valid iff (lo <= p < cur && p != fdst) || p == fsrc
static inline float64 cmask(float64 x, int64 posv, int lo, int cur, int fsrc, int fdst, float64 fill)
{
    float64 r = v_f32_sel_less_i32_b(posv, cur, x, fill);
    r = v_f32_sel_less_i32_b(posv, lo, fill, r);
    r = v_f32_sel_eq_i32_b(posv, fdst, fill, r);
    r = v_f32_sel_eq_i32_b(posv, fsrc, x, r);
    return r;
}

void main(tensor sc, tensor sb, tensor cursor, tensor fix, tensor Pc, tensor Pb, int T, int Lmax, int W)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    const int fsrc = s_i32_ld_g(gen_addr(zc, fix)); zc[0] = 1; const int fdst = s_i32_ld_g(gen_addr(zc, fix));
    const int R = get_dim_size(sc, 1);
    const float NEG = -1e30f;
    // cursor-aware scan: context keys live in [0, cur) plus the fix slot; everything beyond jend is masked (zero P)
    int jend = cur; if (fsrc + 1 > jend) jend = fsrc + 1; jend = (jend + 127) & ~127; if (jend > Lmax) jend = Lmax;
    // window-aware scan: row t's keys start at cur + t - (W-1), so no row has a valid key below cur - (W-1)
    // (the fix source slot aside). The sweeps start at jlo (a multiple of 128: the lanes of the max / sum accumulators see the
    // same positions as before, minus masked ones that contributed NEG / +0: bit-identical P) and P is zero below it. The scan
    // was [0, cur): at a 250K context 30x the window, the drafter softmax was most of a DSpine step (docs/notes/engine-decode-cursor.md).
    int jlo = cur - (W - 1); if (jlo < 0) jlo = 0; if (fsrc >= 0 && fsrc < jlo) jlo = fsrc; jlo &= ~127;
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        const int row = p % R, g = p / R, t = row % T;
        int lo = cur + t - (W - 1); if (lo < 0) lo = 0;
        int5 c = {0, row, g, 0, 0};
        float64 xb = v_f32_ld_tnsr_b(c, sb);
#ifdef CAUSAL
        const int nb = t + 1;   // DSpine: block row t sees block keys 0..t
#else
        const int nb = T;
#endif
        xb = v_f32_sel_less_i32_b(lane, nb, xb, NEG);
        float64 m = xb;
        for (int j = jlo; j < jend; j += 64) { c[0] = j; m = v_f32_max_b(m, cmask(v_f32_ld_tnsr_b(c, sc), lane + j, lo, cur, fsrc, fdst, NEG)); }
        m = REDUCE_MAX64(m, P8, P4, P2, P1);
        float64 eb = v_f32_sel_less_i32_b(lane, nb, v_exp_f32(xb - m), 0.f);
        float64 s = eb;
        for (int j = jlo; j < jend; j += 64) { c[0] = j; s = s + cmask(v_exp_f32(v_f32_ld_tnsr_b(c, sc) - m), lane + j, lo, cur, fsrc, fdst, 0.f); }
        s = REDUCE64(s, P8, P4, P2, P1);
        float64 inv = v_reciprocal_f32(s);
        { float128 pr; pr.v1 = eb * inv; pr.v2 = 0.f; int5 oc = {0, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pb, v_convert_f32_to_bf16_all_b(pr)); }
        for (int j = jend; j < Lmax; j += 128) { float128 z; z.v1 = 0.f; z.v2 = 0.f; int5 oc = {j, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(z)); }
        for (int j = 0; j < jlo; j += 128) { float128 z; z.v1 = 0.f; z.v2 = 0.f; int5 oc = {j, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(z)); }
        for (int j = jlo; j < jend; j += 128) {
            float128 pr;
            c[0] = j;      float64 x0 = v_f32_ld_tnsr_b(c, sc);
            c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, sc);
            pr.v1 = cmask(v_exp_f32(x0 - m) * inv, lane + j, lo, cur, fsrc, fdst, 0.f);
            pr.v2 = cmask(v_exp_f32(x1 - m) * inv, lane + (j + 64), lo, cur, fsrc, fdst, 0.f);
            int5 oc = {j, row, g, 0, 0};
            v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(pr));
        }
    }
}
