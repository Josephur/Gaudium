// gaudium_attn_softmax16: the HALF-WIDTH chunk softmax of the prefill attention core (GAUDIUM_PREFILL_S16, docs/notes/prefill-s16.md).
// The chunk softmax is bound by the VPU (one operation per cycle); an f32 score fills a 32-bit lane, so a vector holds 64 scores.
// Here the QK GEMM writes its scores in bf16 and the whole softmax runs on 128-lane vectors: half the vectors, half the HBM bytes.
//  in0 scores bf16 {C, R, B}   in units of 1/512 OCTAVE: the QK GEMM runs on gaudium_attn_qs16's copy of the queries, q x 512 log2 e
//                              (a bf16 multiply by that constant inside this kernel would round it: a 0.18 % colder softmax).
//                              B = kv groups x chunks, chunk-major (b = ch * G + g) as in gaudium_attn_softmax_chunk; C a multiple of
//                              512, C <= 4096
//  in1 cursor int32 {1}        row t of the tile sees key positions < cursor + coff + t + 1 (mode 0 of the chunk softmax)
//  out0 P bf16 {C, R, B}       exp(x - m), in NATURAL key order (lane = GEMM column): the PV GEMM reads the natural-order copy of the
//                              V chunks (gaudium_attn_vnat), not the V cache with its interleaved rows
//  out1 stats f32 {64, 2, R, B} [lane][0] = m in natural units (row-chunk max x ln 2 / 512), [lane][1] = s (sum of the row's P as
//                              stored): the format of gaudium_attn_softmax_chunk, so gaudium_attn_combine takes both kinds of chunk
// The exp of a bf16 difference, 8 VPU operations per 128 scores:
//   d = x - m (bf16; exact near the maximum, where the two exponents agree)
//   t = rint(d) as int16, SATURATING: d below -64 octaves gives t = -32768, i.e. P = 2^-64 (the clamp costs no operation).
//       z = t >> 9 (floor, the octave), f = t & 511
//   q = 2^(f / 512) in [1, 2) by a quadratic in f (bf16 arithmetic: f above 255 is rounded to even; max error 0.51 %, rms 0.23 %
//       against 0.17 % for the correctly rounded value), P = q with the exponent z (FORM_FP_NUMBER)
// The sum is a MAC with an f32 accumulator (ACC_F32) of P x 1: exactly the sum of the stored bf16 values.
// A batch every row sees whole (a history chunk) runs software-pipelined, TWO ROWS AT A TIME: the loads, the VLM staging and the
// max of rows r + 2, r + 3 ride in the exp sweep of rows r, r + 1 (eight exp chains per iteration; sixteen ran slower), and the four reductions between
// two sweeps (two sums, two maxima: latency chains of 12 operations each) share one basic block instead of two. Any other batch (the
// tile's diagonal chunk, a tile that straddles a chunk boundary) and an odd last row take the row-at-a-time path: two sweeps over
// the 512-blocks that hold a valid key, masked lanes excluded from the maximum and stored as 0, the blocks beyond zero-stored; a row
// that sees nothing of the chunk writes m = -1e30, s = 0, P = 0.
// -DPF8 build (GAUDIUM_PREFILL_S16=2): P is stored as FP8 (E4M3), 128 x exp(x - m): one byte per key. One 256-byte vector per 256
// keys, two bf16 vectors packed by CONVERT: byte 2i <- key i, byte 2i + 1 <- key 128 + i of the block (gaudium_attn_vnat order 1
// gives the V chunks the same row order). The factor 128 (the largest power of two the card's E4M3 holds: its maximum is 240) is
// one int16 add on the exponent. The PV GEMM reads V / 2 as FP8 (V's calibrated maxima reach 122.5), so its partials are 64 x the
// true ones and the sum is written as 64 x: the combine's division cancels the factor. A stored P has 4 significant bits (half an
// ulp is 3 to 6 %); P below 2^-16 of the row maximum rounds to a subnormal or 0. s is the sum of the bf16 values before the FP8
// rounding. Measured: no faster than the bf16 P (the softmax, not P's bytes, bounds the core); kept as the record.
// index space 2-D: dim0 = R, dim1 = B.   scalars: T, nch, coff, pbase (key position of the node's first chunk)
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
// q(f) = 2^(f / 512) ~ 1 + f (Q1 + f Q2): bf16-representable coefficients found by a scan of the neighbours of the end-point quadratic
// (q stays below 2 with and without a fused MAC; tests/sim/sim_softmax16.cpp checks the card's arithmetic)
#define S16_Q1 0.00130462646484375f
#define S16_Q2 1.2367963790893555e-06f
#define S16_LN2U 0.0013538030870311432f   // ln 2 / 512: the stats' m in natural units
#ifdef PF8
#define S16_ZB(t_) v_i16_add_b((t_), (short128)3584)
#define S16_ST(j_, r_, ea_, eb_, ec_, ed_) do { int5 _oc = {(j_), (r_), b, 0, 0}; bfloat256 _pa; _pa.v1 = (ea_); _pa.v2 = (eb_); v_f8_st_tnsr(_oc, P, v_convert_bf16_to_f8_all_b(_pa, SW_RHNE | SW_CLIP_FP)); \
    _oc[0] = (j_) + 256; _pa.v1 = (ec_); _pa.v2 = (ed_); v_f8_st_tnsr(_oc, P, v_convert_bf16_to_f8_all_b(_pa, SW_RHNE | SW_CLIP_FP)); } while (0)
#define S16_ZERO(j0_) for (int _j = (j0_); _j < C; _j += 256) { int5 _oc = {_j, row, b, 0, 0}; char256 _z8 = 0; v_i8_st_tnsr(_oc, P, _z8); }
#else
#define S16_ZB(t_) (t_)
#define S16_ST(j_, r_, ea_, eb_, ec_, ed_) do { int5 _oc = {(j_), (r_), b, 0, 0}; v_bf16_st_tnsr(_oc, P, (ea_)); _oc[0] = (j_) + 128; v_bf16_st_tnsr(_oc, P, (eb_)); \
    _oc[0] = (j_) + 256; v_bf16_st_tnsr(_oc, P, (ec_)); _oc[0] = (j_) + 384; v_bf16_st_tnsr(_oc, P, (ed_)); } while (0)
#define S16_ZERO(j0_) for (int _j = (j0_); _j < C; _j += 128) { int5 _oc = {_j, row, b, 0, 0}; v_bf16_st_tnsr(_oc, P, zz); }
#endif
#ifdef S16_CLAMP   // A/B build: an explicit clamp instead of the saturating convert
#define S16_D(x_, m_) v_bf16_max_b(v_bf16_sub_b((x_), (m_)), (bfloat128)-32256.0f)
#else
#define S16_D(x_, m_) v_bf16_sub_b((x_), (m_))
#endif
#define EXP16(x_, m_) ({ short128 _t = v_convert_bf16_to_i16_b(S16_D((x_), (m_)), SW_RHNE); \
    bfloat128 _f = v_convert_i16_to_bf16_b(v_i16_and_b(_t, (short128)511), SW_RHNE); \
    bfloat128 _q = v_bf16_mac_b(_f, (bfloat128)S16_Q2, (bfloat128)S16_Q1); _q = v_bf16_mac_b(_q, _f, (bfloat128)1.0f); \
    short128 _z = v_i16_ash_b(S16_ZB(_t), (char256)-9); \
    v_bf16_form_fp_num_ie_b((char256)_z, _q, _q, SW_ADD_BIAS); })
__local__ bfloat128 XP[128];   // four rows of C <= 4096 bf16 scores (32 KB): the pair in its exp sweep and the pair being loaded

void main(tensor scores, tensor cursor, tensor P, tensor stats, int T, int nch, int coff, int pbase)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const short128 lane = (short128)read_lane_id_2b_b();
    const int C = get_dim_size(scores, 0);
    const int G = get_dim_size(scores, 2) / nch;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
#ifdef PF8
    const bfloat128 one = 0.5f;   // s = 64 x the sum: P is 128 x, the PV GEMM's V is V / 2
#else
    const bfloat128 one = 1.0f;
#endif
    const short128 negb = (short128)(bfloat128)-3.0e38f, zero16 = 0;
    const int row0 = idx[0], row1 = idx[0] + isz[0];
    // row maximum: bf16 lanes -> f32 lanes (exact) -> one value in every lane (mf: the stats; TO16(mf): the subtraction)
#define ROWMAX2(na_, nb_) ({ float128 _w = v_convert_bf16_to_f32_all_b(v_bf16_max_b((na_), (nb_))); REDUCE_MAX64(v_f32_max_b(_w.v1, _w.v2), P8, P4, P2, P1); })
#define TO16(mf_) ({ float128 _p; _p.v1 = (mf_); _p.v2 = (mf_); v_convert_f32_to_bf16_all_b(_p); })
#define SUM2(a_, b_) ({ float64 _s = ((a_).v1 + (a_).v2) + ((b_).v1 + (b_).v2); REDUCE64(_s, P8, P4, P2, P1); })
#define S16_STATS(r_, mf_, s_) do { int5 _sc = {0, 0, (r_), b, 0}; v_f32_st_tnsr(_sc, stats, (mf_) * S16_LN2U); _sc[1] = 1; v_f32_st_tnsr(_sc, stats, (s_)); } while (0)
    // one 512-column block: the exp sweep of the row staged at XP[xo_ ..] (reference mb_, sums a_ / b_, output row r_) ...
#define S16_E4(xo_, mb_, a_, b_, r_, j_) { const int q_ = (xo_) + ((j_) >> 7); \
        bfloat128 e0 = EXP16(XP[q_], (mb_)), e1 = EXP16(XP[q_ + 1], (mb_)), e2 = EXP16(XP[q_ + 2], (mb_)), e3 = EXP16(XP[q_ + 3], (mb_)); \
        a_ = v_bf16_mac_acc32_b(e0, one, a_); b_ = v_bf16_mac_acc32_b(e1, one, b_); a_ = v_bf16_mac_acc32_b(e2, one, a_); b_ = v_bf16_mac_acc32_b(e3, one, b_); \
        S16_ST((j_), (r_), e0, e1, e2, e3); }
    // ... and the loads, the staging at XP[xo_ ..] and the max (na_ / nb_) of the row at coordinates cn_
#define S16_L4(xo_, na_, nb_, cn_, j_) { const int q_ = (xo_) + ((j_) >> 7); \
        cn_[0] = (j_);       bfloat128 y0 = v_bf16_ld_tnsr_b(cn_, scores); \
        cn_[0] = (j_) + 128; bfloat128 y1 = v_bf16_ld_tnsr_b(cn_, scores); \
        cn_[0] = (j_) + 256; bfloat128 y2 = v_bf16_ld_tnsr_b(cn_, scores); \
        cn_[0] = (j_) + 384; bfloat128 y3 = v_bf16_ld_tnsr_b(cn_, scores); \
        XP[q_] = y0; XP[q_ + 1] = y1; XP[q_ + 2] = y2; XP[q_ + 3] = y3; \
        na_ = v_bf16_max_b(na_, y0); nb_ = v_bf16_max_b(nb_, y1); na_ = v_bf16_max_b(na_, y2); nb_ = v_bf16_max_b(nb_, y3); }
    for (int b = idx[1]; b < idx[1] + isz[1]; ++b) {
        if (isz[0] <= 0) continue;
        const int base = pbase + (b / G) * C;
        int row = row0;
        if (cur + 1 - base >= C) {   // every row sees the whole chunk: pairs of rows, software-pipelined
            int xb = 0;   // XP offset (0 or 64) of the pair in its exp sweep: rows at xb and xb + 32
            float64 mfA = 0.f, mfB = 0.f;
            if (row + 1 < row1) {   // sweep 1 of the first pair
                int5 cA = {0, row, b, 0, 0}, cB = {0, row + 1, b, 0, 0};
                bfloat128 nA0 = -3.0e38f, nA1 = -3.0e38f, nB0 = -3.0e38f, nB1 = -3.0e38f;
                for (int j = 0; j < C; j += 512) { S16_L4(0, nA0, nA1, cA, j) S16_L4(32, nB0, nB1, cB, j) }
                mfA = ROWMAX2(nA0, nA1); mfB = ROWMAX2(nB0, nB1);
            }
            while (row + 1 < row1) {
                const bfloat128 mbA = TO16(mfA), mbB = TO16(mfB);
                float128 aA, bA, aB, bB; aA.v1 = 0.f; aA.v2 = 0.f; bA = aA; aB = aA; bB = aA;
                if (row + 3 < row1) {   // the next pair's sweep 1 inside this pair's sweep 2
                    const int nx = xb ^ 64;
                    int5 cA = {0, row + 2, b, 0, 0}, cB = {0, row + 3, b, 0, 0};
                    bfloat128 nA0 = -3.0e38f, nA1 = -3.0e38f, nB0 = -3.0e38f, nB1 = -3.0e38f;
                    for (int j = 0; j < C; j += 512) { S16_L4(nx, nA0, nA1, cA, j) S16_L4(nx + 32, nB0, nB1, cB, j) S16_E4(xb, mbA, aA, bA, row, j) S16_E4(xb + 32, mbB, aB, bB, row + 1, j) }
                    // four reductions in one block: independent latency chains that the scheduler interleaves
                    const float64 sA = SUM2(aA, bA), sB = SUM2(aB, bB);
                    const float64 mnA = ROWMAX2(nA0, nA1), mnB = ROWMAX2(nB0, nB1);
                    S16_STATS(row, mfA, sA); S16_STATS(row + 1, mfB, sB);
                    mfA = mnA; mfB = mnB; xb = nx;
                } else {
                    for (int j = 0; j < C; j += 512) { S16_E4(xb, mbA, aA, bA, row, j) S16_E4(xb + 32, mbB, aB, bB, row + 1, j) }
                    const float64 sA = SUM2(aA, bA), sB = SUM2(aB, bB);
                    S16_STATS(row, mfA, sA); S16_STATS(row + 1, mfB, sB);
                }
                row += 2;
            }
        }
        // row-at-a-time path: masked batches, and the odd last row of a whole batch
#define MSK(y_, jj_) ((bfloat128)v_i16_sel_less_i16_b(lane + (short)(jj_), nv, (short128)(y_), negb))
#define MSK0(e_, jj_) ((bfloat128)v_i16_sel_less_i16_b(lane + (short)(jj_), nv, (short128)(e_), zero16))
        for (; row < row1; ++row) {
            const int nvalid = cur + (row % T) + 1 - base;
            const bfloat128 zz = 0.f;
            if (nvalid <= 0) {
                S16_ZERO(0)
                int5 sc = {0, 0, row, b, 0}; float64 mneg = -1e30f, sz = 0.f; v_f32_st_tnsr(sc, stats, mneg); sc[1] = 1; v_f32_st_tnsr(sc, stats, sz);
                continue;
            }
            const short nv = (short)(nvalid < C ? nvalid : C);
            const int jlim = nvalid >= C ? C : ((nvalid + 511) & ~511);
            int5 c = {0, row, b, 0, 0};
            bfloat128 n0 = -3.0e38f, n1 = -3.0e38f;
            for (int j = 0; j < jlim; j += 512) {
                c[0] = j;       bfloat128 y0 = v_bf16_ld_tnsr_b(c, scores);
                c[0] = j + 128; bfloat128 y1 = v_bf16_ld_tnsr_b(c, scores);
                c[0] = j + 256; bfloat128 y2 = v_bf16_ld_tnsr_b(c, scores);
                c[0] = j + 384; bfloat128 y3 = v_bf16_ld_tnsr_b(c, scores);
                if (j + 512 > nvalid) { y0 = MSK(y0, j); y1 = MSK(y1, j + 128); y2 = MSK(y2, j + 256); y3 = MSK(y3, j + 384); }
                XP[j >> 7] = y0; XP[(j >> 7) + 1] = y1; XP[(j >> 7) + 2] = y2; XP[(j >> 7) + 3] = y3;
                n0 = v_bf16_max_b(n0, y0); n1 = v_bf16_max_b(n1, y1); n0 = v_bf16_max_b(n0, y2); n1 = v_bf16_max_b(n1, y3);
            }
            const float64 mf = ROWMAX2(n0, n1); const bfloat128 mb = TO16(mf);
            float128 a0, a1; a0.v1 = 0.f; a0.v2 = 0.f; a1 = a0;
            for (int j = 0; j < jlim; j += 512) { const int q_ = j >> 7;
                bfloat128 e0 = EXP16(XP[q_], mb), e1 = EXP16(XP[q_ + 1], mb), e2 = EXP16(XP[q_ + 2], mb), e3 = EXP16(XP[q_ + 3], mb);
                if (j + 512 > nvalid) { e0 = MSK0(e0, j); e1 = MSK0(e1, j + 128); e2 = MSK0(e2, j + 256); e3 = MSK0(e3, j + 384); }
                a0 = v_bf16_mac_acc32_b(e0, one, a0); a1 = v_bf16_mac_acc32_b(e1, one, a1); a0 = v_bf16_mac_acc32_b(e2, one, a0); a1 = v_bf16_mac_acc32_b(e3, one, a1);
                S16_ST(j, row, e0, e1, e2, e3); }
            S16_ZERO(jlim)
            const float64 s = SUM2(a0, a1);
            S16_STATS(row, mf, s);
        }
    }
}
