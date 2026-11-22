// gaudium_ds_softr: block attention softmax of a block drafter over two key sets, the layer's context RING and the block itself
// (causal inside the block), sweeping only the ring slots a row can see.
//
// The context keys live in a ring of RG slots (RG a multiple of 128; slot = position mod RG, written by gaudium_ds_ctxr).
// Row r of group g: t = r % T is the query row at position cur + t. It sees
//   the context positions p with max(cur + t - (W - 1), flo) <= p < cur: the nv = min(W - 1 - t, cur - flo) newest slots;
//   the block keys 0 .. t.
// With hi1 = ((cur - 1) mod RG) + 1, the slot after the newest context row, those context slots are
//   A = [hi1 - nv, hi1)                           while the window does not wrap (nv <= hi1), else
//   A = [0, hi1) and B = [RG - (nv - hi1), RG).
// The glue checks RG >= W + 256, and the engine sizes a ring to hold a window plus the rows of a context pass, so a row that
// a pass wrote past the cursor is never inside a window and the 128-aligned sweeps of A and B never meet.
// Only the aligned hull of row 0's slots (the widest window of the block) is swept, so the time follows min(cur, W), not RG.
// P = exp(x - max) / sum over both key sets inside the row's own bounds, 0 elsewhere (the P x V GEMM reads the whole ring).
//  in0 sc f32 {RG, R, 8}  in1 sb f32 {128, R, 8}  in2 cur int32 {1}
//  in3 lim int32 {2}: lim[0] = flo, the first context position whose ring row is valid (< 0: every position below the cursor)
//  out0 Pc bf16 {RG, R, 8}  out1 Pb bf16 {128, R, 8} (interleaved pairs, matched by the V row permutation)
//  index space: R*8.  scalars: T, RG, W
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
// slot mask: x where lo <= slot < up, fill elsewhere
#define SMASK(x, sv, lo, up, fill) v_f32_sel_less_i32_b((sv), (lo), (fill), v_f32_sel_less_i32_b((sv), (up), (x), (fill)))

void main(tensor sc, tensor sb, tensor cursor, tensor lim, tensor Pc, tensor Pb, int T, int RG, int W)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    int flo = s_i32_ld_g(gen_addr(zc, lim)); if (flo < 0) flo = 0;
    const int R = get_dim_size(sc, 1);
    const float NEG = -1e30f;
    int span = cur - flo; if (span < 0) span = 0;   // context rows a window may reach back over
    const int hi1 = (cur + RG - 1) % RG + 1;        // in [1, RG]; cur = 0 gives RG with span = 0: nothing is read
    // the sweeps: the aligned hull of the widest window (row 0)
    int nvm = W - 1; if (nvm > span) nvm = span; if (nvm < 0) nvm = 0;
    int a0 = hi1 - nvm; if (a0 < 0) a0 = 0; a0 &= ~127;
    int a1 = (hi1 + 127) & ~127; if (nvm == 0) a1 = a0;
    int ovm = nvm - hi1; if (ovm < 0) ovm = 0;
    int b0 = (RG - ovm) & ~127; if (b0 < a1) b0 = a1;
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        const int row = p % R, g = p / R, t = row % T;
        int nv = W - 1 - t; if (nv > span) nv = span; if (nv < 0) nv = 0;
        int loA = hi1 - nv; if (loA < 0) loA = 0;   // A = [loA, hi1)
        int ov = nv - hi1; if (ov < 0) ov = 0;
        const int loB = RG - ov;                    // B = [loB, RG)
        int5 c = {0, row, g, 0, 0};
        float64 xb = v_f32_ld_tnsr_b(c, sb);
        const int nb = t + 1;   // block row t sees block keys 0..t
        xb = v_f32_sel_less_i32_b(lane, nb, xb, NEG);
        float64 m = xb;
        for (int j = a0; j < a1; j += 64) { c[0] = j; m = v_f32_max_b(m, SMASK(v_f32_ld_tnsr_b(c, sc), lane + j, loA, hi1, NEG)); }
        for (int j = b0; j < RG; j += 64) { c[0] = j; m = v_f32_max_b(m, SMASK(v_f32_ld_tnsr_b(c, sc), lane + j, loB, RG, NEG)); }
        m = REDUCE_MAX64(m, P8, P4, P2, P1);
        float64 eb = v_f32_sel_less_i32_b(lane, nb, v_exp_f32(xb - m), 0.f);
        float64 s = eb;
        for (int j = a0; j < a1; j += 64) { c[0] = j; s = s + SMASK(v_exp_f32(v_f32_ld_tnsr_b(c, sc) - m), lane + j, loA, hi1, 0.f); }
        for (int j = b0; j < RG; j += 64) { c[0] = j; s = s + SMASK(v_exp_f32(v_f32_ld_tnsr_b(c, sc) - m), lane + j, loB, RG, 0.f); }
        s = REDUCE64(s, P8, P4, P2, P1);
        float64 inv = v_reciprocal_f32(s);
        { float128 pr; pr.v1 = eb * inv; pr.v2 = 0.f; int5 oc = {0, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pb, v_convert_f32_to_bf16_all_b(pr)); }
        // zero P outside the sweeps
        for (int j = 0; j < a0; j += 128) { float128 z; z.v1 = 0.f; z.v2 = 0.f; int5 oc = {j, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(z)); }
        for (int j = a1; j < b0; j += 128) { float128 z; z.v1 = 0.f; z.v2 = 0.f; int5 oc = {j, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(z)); }
        for (int j = a0; j < a1; j += 128) {
            float128 pr;
            c[0] = j;      float64 x0 = v_f32_ld_tnsr_b(c, sc);
            c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, sc);
            pr.v1 = SMASK(v_exp_f32(x0 - m) * inv, lane + j, loA, hi1, 0.f);
            pr.v2 = SMASK(v_exp_f32(x1 - m) * inv, lane + (j + 64), loA, hi1, 0.f);
            int5 oc = {j, row, g, 0, 0};
            v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(pr));
        }
        for (int j = b0; j < RG; j += 128) {
            float128 pr;
            c[0] = j;      float64 x0 = v_f32_ld_tnsr_b(c, sc);
            c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, sc);
            pr.v1 = SMASK(v_exp_f32(x0 - m) * inv, lane + j, loB, RG, 0.f);
            pr.v2 = SMASK(v_exp_f32(x1 - m) * inv, lane + (j + 64), loB, RG, 0.f);
            int5 oc = {j, row, g, 0, 0};
            v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(pr));
        }
    }
}
