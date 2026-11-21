// gaudium_ga_dfsoftr: the block attention softmax of a DFlash drafter's WINDOWED layer (docs/notes/engine-gemma4.md, section 19): one
// softmax over two key sets, the layer's context RING and the block itself. The ring build of gaudium_ds_softmax (df_softmax_gaudi2.c,
// -DCAUSAL), as a source of its own so that the kernels of the other families stay the same binaries.
// The context keys of a windowed layer live in a ring of RG slots (a power of two; slot = position & (RG - 1), written by
// gaudium_ga_dfctxr). Row r of group g: t = r % T is the query row at position cur + t. It sees
//   the context positions p with cur + t - (W - 1) <= p < cur: the nv = min(W - 1 - t, cur) slots whose ring distance from the last
//   context position, (cur - 1 - slot) & (RG - 1), is below nv;
//   the block keys 0 .. t (causal inside the block).
// RG >= W + (the rows a context pass writes) - 1, so a stale row beyond the cursor is never inside a window. The whole ring is swept
// (it is about two windows). P = exp(x - max) / sum over both sets, 0 elsewhere.
//  in0 sc f32 {RG, R, 8}  in1 sb f32 {128, R, 8}  in2 cur int32 {1}  in3 fix int32 {2} (not read: a chain has no slot move)
//  out0 Pc bf16 {RG, R, 8}  out1 Pb bf16 {128, R, 8} (interleaved pairs, matched by the V row permutation)
//  index space: R*8.  scalars: T, Lmax (= RG), W
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
    const int R = get_dim_size(sc, 1);
    const float NEG = -1e30f;
#define RSEL(x, jj, fill) v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + (jj)), M1, 0, 0, 1, 0), nv, (x), (fill))
    const int M1 = Lmax - 1;
    const int64 pv = (int64)(cur - 1);
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        const int row = p % R, g = p / R, t = row % T;
        int nv = W - 1 - t; if (nv > cur) nv = cur; if (nv < 0) nv = 0;
        int5 c = {0, row, g, 0, 0};
        float64 xb = v_f32_ld_tnsr_b(c, sb);
        const int nb = t + 1;
        xb = v_f32_sel_less_i32_b(lane, nb, xb, NEG);
        float64 m = xb;
        for (int j = 0; j < Lmax; j += 64) { c[0] = j; m = v_f32_max_b(m, RSEL(v_f32_ld_tnsr_b(c, sc), j, NEG)); }
        m = REDUCE_MAX64(m, P8, P4, P2, P1);
        float64 eb = v_f32_sel_less_i32_b(lane, nb, v_exp_f32(xb - m), 0.f);
        float64 s = eb;
        for (int j = 0; j < Lmax; j += 64) { c[0] = j; s = s + RSEL(v_exp_f32(v_f32_ld_tnsr_b(c, sc) - m), j, 0.f); }
        s = REDUCE64(s, P8, P4, P2, P1);
        float64 inv = v_reciprocal_f32(s);
        { float128 pr; pr.v1 = eb * inv; pr.v2 = 0.f; int5 oc = {0, row, g, 0, 0}; v_bf16_st_tnsr(oc, Pb, v_convert_f32_to_bf16_all_b(pr)); }
        for (int j = 0; j < Lmax; j += 128) {
            float128 pr;
            c[0] = j;      float64 x0 = v_f32_ld_tnsr_b(c, sc);
            c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, sc);
            pr.v1 = RSEL(v_exp_f32(x0 - m) * inv, j, 0.f);
            pr.v2 = RSEL(v_exp_f32(x1 - m) * inv, j + 64, 0.f);
            int5 oc = {j, row, g, 0, 0};
            v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(pr));
        }
    }
}
