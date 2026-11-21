// gaudium_ga_softmax_ring: masked softmax of a Gemma 4 SLIDING layer over its K / V ring (docs/notes/engine-gemma4.md).
// The layer keeps the keys of the last positions in a ring of RING slots (slot = position & (RING - 1), gaudium_ga_prep ring == 1).
// A row at position p sees the positions (p - W, p]: its own and the W - 1 before it (transformers sliding_window_overlay).
//  in0 scores f32 {RING, R, G}   G kv groups, R = GQ T rows per group, row = hl T + t
//  in1 cursor int32 {1}
//  out0 P bf16 {RING, R, G}: exp(x - m) at the valid slots, 0 elsewhere; interleaved order inside each 128-block exactly like
//       gaudium_attn_softmax_chunk (bf16 lane m <- column m even ? m / 2 : 64 + (m - 1) / 2), matched by the V ring rows (VPERM)
//  out1 stats f32 {64, 2, R, G}: [lane][0] = m (the row's max, 64 equal lanes), [lane][1] = s (sum of the valid exp)
//  valid(slot, row):  ((p - slot) & (RING - 1)) < nv,   nv = min(W, p + 1)
//      mode 0: p = cursor + coff + t        a pass that wrote its own rows into the ring (verify, prefill tile): row t at its position.
//                                           Slots of the pass's later rows (p + 1 .. p + T - 1) are at distance RING - j >= W and masked,
//                                           and so is every stale row of a rejected draft beyond the cursor, as long as RING >= W + T - 1.
//      mode 1: p = cursor + coff - 1        a pass that only READS the ring (a layer or a drafter that shares this layer's keys): every
//                                           row sees the positions below the cursor.
//  The normalisation (1 / s) is not applied here: the PV GEMM gives unnormalised outputs and gaudium_ga_combine scales them (nch = 1).
//  Two sweeps per row (max; exp + sum + store) with the row's scores staged in VLM. exp(d), d <= 0: the Cody-Waite + degree-6 polynomial
//  of gaudium_attn_softmax_chunk (EXPD: max 1 ulp on [-87, 0]; the vendor v_exp_f32 is wrong at 42 inputs, research/exp_bug).
//  index space 1-D: point pp = g R + row.   scalars: T, coff, W, mode.   RING <= 4096 and a multiple of 256.
//  -DEXT (gaudium_ga_softmax_ringx): one more input after the cursor, ext int32 {T}: row t also sees the ext[t] positions AFTER its own
//      (the rest of its image block: inside a block of image or video soft tokens a sliding layer attends in both directions,
//      transformers blockwise_overlay). The row's window then ends at e = p + ext[t]: valid = ((e - slot) & (RING - 1)) < nv,
//      nv = min(W + ext[t], e + 1), i.e. the positions (p - W, e]. The later rows of the pass must already be in the ring (the same
//      pass wrote them, or a keys-only pass of the following tile did), and RING >= W + (the rows from p to the last one written).
//      mode is not read (row t at its own position, as mode 0).
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
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
// lanes of the 64 slots starting at slot jj whose ring distance from p is below nv keep x, the others get `fill`
#define RSEL(x, jj, fill) v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + (jj)), M1, 0, 0, 1, 0), nv, (x), (fill))

__local__ float64 XV[64];   // one row's masked scores (RING <= 4096)

#ifdef EXT
void main(tensor scores, tensor cursor, tensor ext, tensor P, tensor stats, int T, int coff, int W, int mode)
#else
void main(tensor scores, tensor cursor, tensor P, tensor stats, int T, int coff, int W, int mode)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int RING = get_dim_size(scores, 0), RR = get_dim_size(scores, 1), M1 = RING - 1;
    const float NEG = -1e30f;
    for (int pp = idx[0]; pp < idx[0] + isz[0]; ++pp) {
        const int g = pp / RR, row = pp - g * RR;
        const int t = row % T;
#ifdef EXT
        int5 xc = {t, 0, 0, 0, 0};
        const int xr = s_i32_ld_g(gen_addr(xc, ext));
        const int p = cur + t + xr;                          // the last position the row sees (the end of its block)
        const int nv = (p + 1 < W + xr) ? p + 1 : W + xr;
#else
        const int p = mode ? cur - 1 : cur + t;
        const int nv = (p + 1 < W) ? p + 1 : W;
#endif
        int5 sc = {0, 0, row, g, 0};
        if (nv <= 0) {   // nothing below the cursor yet (mode 1 at position 0)
            float128 zz; zz.v1 = 0.f; zz.v2 = 0.f;
            for (int j = 0; j < RING; j += 128) { int5 oc = {j, row, g, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(zz)); }
            float64 mneg = NEG, zero = 0.f;
            v_f32_st_tnsr(sc, stats, mneg); sc[1] = 1; v_f32_st_tnsr(sc, stats, zero);
            continue;
        }
        const int64 pv = (int64)p;
        int5 c = {0, row, g, 0, 0};
        float64 m0 = NEG, m1 = NEG, m2 = NEG, m3 = NEG;
        for (int j = 0; j < RING; j += 256) {
            c[0] = j;       float64 y0 = v_f32_ld_tnsr_b(c, scores);
            c[0] = j + 64;  float64 y1 = v_f32_ld_tnsr_b(c, scores);
            c[0] = j + 128; float64 y2 = v_f32_ld_tnsr_b(c, scores);
            c[0] = j + 192; float64 y3 = v_f32_ld_tnsr_b(c, scores);
            y0 = RSEL(y0, j, NEG); y1 = RSEL(y1, j + 64, NEG); y2 = RSEL(y2, j + 128, NEG); y3 = RSEL(y3, j + 192, NEG);
            const int q = j >> 6;
            XV[q] = y0; XV[q + 1] = y1; XV[q + 2] = y2; XV[q + 3] = y3;
            m0 = v_f32_max_b(m0, y0); m1 = v_f32_max_b(m1, y1); m2 = v_f32_max_b(m2, y2); m3 = v_f32_max_b(m3, y3);
        }
        const float64 m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
        float64 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (int j = 0; j < RING; j += 256) {
            const int q = j >> 6;
            float64 e0 = EXPD(XV[q] - m), e1 = EXPD(XV[q + 1] - m), e2 = EXPD(XV[q + 2] - m), e3 = EXPD(XV[q + 3] - m);
            e0 = RSEL(e0, j, 0.f); e1 = RSEL(e1, j + 64, 0.f); e2 = RSEL(e2, j + 128, 0.f); e3 = RSEL(e3, j + 192, 0.f);
            s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3;
            float128 pa; pa.v1 = e0; pa.v2 = e1;
            float128 pb; pb.v1 = e2; pb.v2 = e3;
            int5 oc = {j, row, g, 0, 0};
            v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa)); oc[0] = j + 128;
            v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
        }
        const float64 s = REDUCE64((s0 + s1) + (s2 + s3), P8, P4, P2, P1);
        v_f32_st_tnsr(sc, stats, m); sc[1] = 1; v_f32_st_tnsr(sc, stats, s);
    }
}
