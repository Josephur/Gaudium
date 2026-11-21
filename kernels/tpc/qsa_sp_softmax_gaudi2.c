// gaudium_qsa_sp_softmax: gathered sparse attention core of QSA (docs/notes/flash-next-qsa-sparse.md), step 3 of 4: the softmax over
// each row's gathered slots (gaudium_qsa_gather) with per-row statistics, in the form of gaudium_attn_softmax_chunk with one chunk.
//  in0 scores f32 {G, GQ, BT}: q2 . kg^T per batch bt = g*T + t (batch GEMM), slot order of the gather
//  in1 cursor int32 {1}
//  out0 P bf16 {G, GQ, BT}: exp(x - m) at valid slots, 0 elsewhere, in the interleaved order of the chunk softmax (bf16 lane m <- slot
//       m even ? m/2 : 64 + (m-1)/2 of each 128-block), matched by the gather's vg rows
//  out1 stats f32 {64, 2, GQ, BT}: [.][0] = m (row max), [.][1] = s (sum of the valid exp), 64 equal lanes each
//  Valid slots of row t (p = cursor + coff + t, nb = (p + 1) / 4, nsel = min(nb, nblk), ntail = p + 1 - 4 nb):
//      [0, 4 nsel) and [4 nblk, 4 nblk + ntail). The exp is the FASTEXP formula of the chunk softmax (EXPD, max 1 ulp on [-87, 0]).
//  scalars: T, coff, nblk.   Index space {GQ * BT}: one (head, batch) row per point.
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
// valid(slot) for the 64 lanes at slot j0
#define VSEL(x_, j0_, fill_) v_f32_sel_less_i32_b(lane + (j0_), va, (x_), v_f32_sel_geq_i32_b(lane + (j0_), tb, v_f32_sel_less_i32_b(lane + (j0_), te, (x_), (fill_)), (fill_)))

void main(tensor scores, tensor cursor, tensor P, tensor stats, int T, int coff, int nblk)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int G = get_dim_size(scores, 0), GQ = get_dim_size(scores, 1);
    const float NEG = -1e30f;
    const int tb = 4 * nblk;                       // first tail slot
    for (int pp = idx[0]; pp < idx[0] + isz[0]; ++pp) {
        const int bt = pp / GQ, h = pp - bt * GQ, t = bt % T;
        const int p = cur + t, nb = (p + 1) >> 2, nsel = nb < nblk ? nb : nblk;
        const int va = 4 * nsel, te = tb + (p + 1 - 4 * nb);
        const int jA = (va + 127) & ~127;          // 128-blocks [0, jA) hold the attended blocks, [tb, G) the tail; the rest is zero
        int5 c = {0, h, bt, 0, 0};
        float64 m0 = NEG, m1 = NEG;
#ifdef SP_BASE   // the first form of this kernel (A/B builds): one 128-block per iteration, every lane through the validity selects
        for (int j = 0; j < G; j += 128) {
            if (j >= jA && j < tb) continue;
            c[0] = j; float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores);
            m0 = v_f32_max_b(m0, VSEL(x0, j, NEG)); m1 = v_f32_max_b(m1, VSEL(x1, j + 64, NEG));
        }
        const float64 m = REDUCE_MAX64(v_f32_max_b(m0, m1), P8, P4, P2, P1);
        float64 s0 = 0.f, s1 = 0.f;
        for (int j = 0; j < G; j += 128) {
            int5 oc = {j, h, bt, 0, 0};
            float128 pa;
            if (j >= jA && j < tb) { pa.v1 = 0.f; pa.v2 = 0.f; }
            else {
                c[0] = j; float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores);
                float64 e0 = VSEL(EXPD(x0 - m), j, 0.f), e1 = VSEL(EXPD(x1 - m), j + 64, 0.f);
                s0 = s0 + e0; s1 = s1 + e1; pa.v1 = e0; pa.v2 = e1;
            }
            v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
        }
#else
        // The row's slots as three kinds of 128-blocks: [0, jF) fully valid (every lane below va: the validity select is the identity,
        // so it is left out), [jF, jA) the one partly valid block of a row with fewer than nblk blocks, [jA, tb) empty (zero-stored
        // without loads) and [tb, G) the tail block. The fully valid blocks run two per iteration with no branch in the body (the
        // first form took one block per iteration with a branch and three selects per vector and sweep: latency-bound at ~80 cycles
        // per vector on the card, level-1 profile 2026-11-16). Same operations on the same values in the same order per
        // accumulator (m0 / s0: the even 64-lane halves, m1 / s1: the odd ones), so P and the stats are bit-identical to that form
        // (-DSP_BASE; tests/sim/sim_spsoftp.cpp).
        const int jF = va & ~127;
#define SP_M1(j_)  { c[0] = (j_); float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = (j_) + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores); \
                     m0 = v_f32_max_b(m0, x0); m1 = v_f32_max_b(m1, x1); }
#define SP_M1V(j_) { c[0] = (j_); float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = (j_) + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores); \
                     m0 = v_f32_max_b(m0, VSEL(x0, (j_), NEG)); m1 = v_f32_max_b(m1, VSEL(x1, (j_) + 64, NEG)); }
        { int j = 0;
          for (; j + 256 <= jF; j += 256) {
              c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
              c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
              m0 = v_f32_max_b(m0, x0); m1 = v_f32_max_b(m1, x1); m0 = v_f32_max_b(m0, x2); m1 = v_f32_max_b(m1, x3);
          }
          for (; j < jF; j += 128) SP_M1(j)
          for (; j < jA; j += 128) SP_M1V(j)
          for (j = tb; j < G; j += 128) SP_M1V(j) }
        const float64 m = REDUCE_MAX64(v_f32_max_b(m0, m1), P8, P4, P2, P1);
        float64 s0 = 0.f, s1 = 0.f;
#define SP_E1(j_)  { c[0] = (j_); float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = (j_) + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores); \
                     float64 e0 = EXPD(x0 - m), e1 = EXPD(x1 - m); s0 = s0 + e0; s1 = s1 + e1; \
                     float128 pa; pa.v1 = e0; pa.v2 = e1; int5 oc = {(j_), h, bt, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa)); }
#define SP_E1V(j_) { c[0] = (j_); float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = (j_) + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores); \
                     float64 e0 = VSEL(EXPD(x0 - m), (j_), 0.f), e1 = VSEL(EXPD(x1 - m), (j_) + 64, 0.f); s0 = s0 + e0; s1 = s1 + e1; \
                     float128 pa; pa.v1 = e0; pa.v2 = e1; int5 oc = {(j_), h, bt, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa)); }
        // a row that holds all nblk blocks (jF == tb: every row past the dense limit): its last two fully valid blocks and the tail
        // block run as ONE iteration (six exp chains side by side) instead of a 4-vector and a 2-vector one, each as long as a chain
        const int tfold = (jF == tb && jF >= 256 && tb + 128 <= G) ? 1 : 0, jL = tfold ? jF - 256 : jF;
        { int j = 0;
          for (; j + 256 <= jL; j += 256) {
              c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
              c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
              float64 e0 = EXPD(x0 - m), e1 = EXPD(x1 - m), e2 = EXPD(x2 - m), e3 = EXPD(x3 - m);
              s0 = s0 + e0; s1 = s1 + e1; s0 = s0 + e2; s1 = s1 + e3;
              float128 pa; pa.v1 = e0; pa.v2 = e1; float128 pb; pb.v1 = e2; pb.v2 = e3;
              int5 oc = {j, h, bt, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
              oc[0] = j + 128; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
          }
          if (tfold) {   // j == jF - 256: blocks j, j + 128 (fully valid) and the tail block tb, in the order of the plain loops
              c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
              c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores); c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
              c[0] = tb;      float64 x4 = v_f32_ld_tnsr_b(c, scores); c[0] = tb + 64; float64 x5 = v_f32_ld_tnsr_b(c, scores);
              float64 e0 = EXPD(x0 - m), e1 = EXPD(x1 - m), e2 = EXPD(x2 - m), e3 = EXPD(x3 - m);
              float64 e4 = VSEL(EXPD(x4 - m), tb, 0.f), e5 = VSEL(EXPD(x5 - m), tb + 64, 0.f);
              s0 = s0 + e0; s1 = s1 + e1; s0 = s0 + e2; s1 = s1 + e3; s0 = s0 + e4; s1 = s1 + e5;
              float128 pa; pa.v1 = e0; pa.v2 = e1; float128 pb; pb.v1 = e2; pb.v2 = e3; float128 pc; pc.v1 = e4; pc.v2 = e5;
              int5 oc = {j, h, bt, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
              oc[0] = j + 128; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
              oc[0] = tb; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pc));
              j = jF;
          }
          for (; j < jF; j += 128) SP_E1(j)
          for (; j < jA; j += 128) SP_E1V(j)
          { float128 zz; zz.v1 = 0.f; zz.v2 = 0.f; for (; j < tb; j += 128) { int5 oc = {j, h, bt, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(zz)); } }
          for (j = tb + (tfold ? 128 : 0); j < G; j += 128) SP_E1V(j) }
#endif
        const float64 s = REDUCE64(s0 + s1, P8, P4, P2, P1);
        int5 sc = {0, 0, h, bt, 0};
        v_f32_st_tnsr(sc, stats, m); sc[1] = 1; v_f32_st_tnsr(sc, stats, s);
    }
}
