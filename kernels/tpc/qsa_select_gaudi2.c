// gaudium_qsa_select: exact top-K block selection of each query row -> the softmax's mask words (docs/notes/engine-qsa.md).
//  in0 iss f32 {NB, T}: block scores of gaudium_qsa_score   in1 cursor int32 {1}   in2 cursor_big int32 {1} (mode 1; mode 0: unread)
//  out0 qsel int32 {64, NW, T}: word (lane l, w, t) bit k = 1 iff position 2048 w + 64 k + l is in row t's tail or in a selected block
//  Row t: p = cursor + coff + t. mode 0: nb = (p + 1) / 4 blocks, K = nblk; dense rows (nb <= K) are skipped (the softmax does not read
//  them). mode 1 (FRESH): nb = cursor_big / 4, K = nblk - ((p + 1) / 4 - nb); dense rows get all-ones words, K <= 0 selects no block.
//  Order: |s| descending, then block index ascending (the lower block wins a tie). The sort key (|s| bits, nb - 1 - b) is unique, so the
//  K-th largest key is a sharp threshold. It is found by radix bisection in three levels, branch-free (the TPC has no vector -> scalar
//  path, so the search state lives in vectors with every lane equal):
//    L1: k1 = (|s| >> 15) - 32768 as int16 in VLM (one iss pass), 16 steps of count(k1 >= mid) -> bucket k1 = kap1, cab1 = count above it
//    L2: VLM rewritten with k2 = |s| & 0x7FFF for the bucket's members (others -1), 15 steps with K - cab1 -> tau = exact f32 bits
//    L3: VLM rewritten with k3 = (nb - 1 - b) - 32767 for blocks with |s| == tau (others -32768), ceil(log2 nb) steps -> kap3
//  selected(b) = |s| > tau  or  (|s| == tau and nb - 1 - b - 32767 >= kap3);  b >= nb (the tail, or beyond the row) is set.
//  VLM holds the keys of the first NVV * 128 blocks (38912 = context 155648); blocks beyond are re-keyed from iss in every count.
//  The count loop runs 4 VLM vectors per iteration into 4 int16 accumulators (VPU-bound: 2 ops per 128 keys per bisection step).
//  Counts are summed in int16 lanes (<= NVV per lane) and reduced in f32 (exact below 2^24).
//  scalars: coff, mode, nblk.   Index space {T}: one row per point (rows spread over the TPCs).
//  -DBITMAP build (gathered sparse core, docs/notes/flash-next-qsa-sparse.md): + out1 qbm int32 {64, NBW, T}, the row's SELECTED BLOCKS
//  (tail and blocks >= nb excluded) as a block bitmap a scalar loop can scan in ascending block order: for the 2048-block group s,
//  lane i (0..31) of vector {., s, t} holds blocks 2048 s + 64 i + k (bit k, k < 32) and lane 32 + i blocks 2048 s + 64 i + 32 + k.
//  Written for the computed rows only (mode 0 rows with nb > K), vectors s < ceil(nwr * 512 / 2048); qsel is unchanged.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define MOVDG_ALLI(a, s0, s1, s2, s3) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (a))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define NVV 304
// The 4 count chains are deliberately NOT isomorphic (two count up, two count down, two compare with geq, two with less): written
// alike, the compiler re-rolled them into one loop with one latency-bound accumulator chain (~8 cycles per VLM vector instead of 2).
__local__ short128 KV[NVV];

#define UBITS(s_) ({ float64 _s = (s_); int64 _u = *(int64*)&_s; _u & 0x7FFFFFFF; })
// level keys of 64 blocks bb with |s| bits u (int64 lanes)
#define K1(u_, bb_) v_i32_sel_less_i32_b((bb_), nb, (((u_) >> 15) - 32768), -32768)
#define K2(u_, bb_) v_i32_sel_less_i32_b((bb_), nb, v_i32_sel_eq_i32_b((((u_) >> 15) - 32768), kap1, ((u_) & 0x7FFF), -1), -1)
#define K3(u_, bb_) v_i32_sel_less_i32_b((bb_), nb, v_i32_sel_eq_i32_b((u_), tau, ((nb - 32768) - (bb_)), -32768), -32768)
// (re)write the VLM keys of level KEYF for blocks [0, nvv * 128)
#define FILL(KEYF) for (int v = 0; v < nvv; v += 2) { \
        int5 _c = {128 * v, t, 0, 0, 0}; int64 _u0 = UBITS(v_f32_ld_tnsr_b(_c, iss)); _c[0] += 64; int64 _u1 = UBITS(v_f32_ld_tnsr_b(_c, iss)); \
        _c[0] += 64; int64 _u2 = UBITS(v_f32_ld_tnsr_b(_c, iss)); _c[0] += 64; int64 _u3 = UBITS(v_f32_ld_tnsr_b(_c, iss)); \
        int64 _b0 = lane + 128 * v; int128 _kk, _kl; _kk.v1 = KEYF(_u0, _b0); _kk.v2 = KEYF(_u1, _b0 + 64); _kl.v1 = KEYF(_u2, _b0 + 128); _kl.v2 = KEYF(_u3, _b0 + 192); \
        KV[v] = v_convert_i32_to_i16_all_b(_kk, 0); KV[v + 1] = v_convert_i32_to_i16_all_b(_kl, 0); }
// count of keys >= thr (thr: int64, every lane equal), VLM part + blocks [nbv, nb) re-keyed from iss; result f32, every lane equal
#define COUNT(thr_, KEYF) ({ int64 _thr = (thr_); int128 _tt; _tt.v1 = _thr; _tt.v2 = _thr; short128 _th = v_convert_i32_to_i16_all_b(_tt, 0); \
        short128 _a0 = 0, _a1 = 0, _a2 = 0, _a3 = 0; \
        for (int v = 0; v < nvv; v += 4) { \
            short128 _k0 = KV[v], _k1 = KV[v + 1], _k2 = KV[v + 2], _k3 = KV[v + 3]; \
            _a0 = v_i16_add_b(_a0, v_i16_sel_geq_i16_b(_k0, _th, one16, zero16), 0, _a0, 1, 0); \
            _a1 = v_i16_sub_b(_a1, v_i16_sel_geq_i16_b(_k1, _th, one16, zero16), 0, _a1, 1, 0); \
            _a2 = v_i16_add_b(_a2, v_i16_sel_less_i16_b(_k2, _th, zero16, one16), 0, _a2, 1, 0); \
            _a3 = v_i16_sub_b(_a3, v_i16_sel_less_i16_b(_k3, _th, zero16, one16), 0, _a3, 1, 0); } \
        float128 _f = v_convert_i16_to_f32_all_b(v_i16_sub_b(v_i16_add_b(_a0, _a2, 0, _a0, 1, 0), v_i16_add_b(_a1, _a3, 0, _a1, 1, 0), 0, _a0, 1, 0), 0); float64 _cf = _f.v1 + _f.v2; \
        if (nb > nbv) { int64 _a32 = 0; for (int b0 = nbv; b0 < nb; b0 += 64) { int5 _c = {b0, t, 0, 0, 0}; int64 _u = UBITS(v_f32_ld_tnsr_b(_c, iss)); \
            _a32 = _a32 + v_i32_sel_geq_i32_b(KEYF(_u, lane + b0), _thr, 1, 0); } _cf = _cf + v_convert_i32_to_f32_b(_a32, 0); } \
        REDUCE64(_cf, P8, P4, P2, P1); })
// NIT bisection steps on [lo, hi] for the largest key value kap with count(key >= kap) >= kreq; cab = count(key > kap)
#define BISECT(KEYF, LO_, HI_, NIT_, kreq_, kap_, cab_) { int64 _lo = (LO_), _hi = (HI_); float64 _cab = 0.f; const float64 _kr = (kreq_); \
        for (int it = 0; it < (NIT_); ++it) { int64 _mid = _lo + ((_hi - _lo + 1) >> 1); float64 _cnt = COUNT(_mid, KEYF); \
            _lo = v_i32_sel_geq_f32_b(_cnt, _kr, _mid, _lo); _hi = v_i32_sel_geq_f32_b(_cnt, _kr, _hi, _mid - 1); _cab = v_f32_sel_geq_f32_b(_cnt, _kr, _cab, _cnt); } \
        kap_ = _lo; cab_ = _cab; }

#ifdef BITMAP
void main(tensor iss, tensor cursor, tensor cursor_big, tensor qsel, tensor qbm, int coff, int mode, int nblk)
#else
void main(tensor iss, tensor cursor, tensor cursor_big, tensor qsel, int coff, int mode, int nblk)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    // word assembly: block lane m = 16k + q contributes bit 4i + k (k = m / 16) of word q; lane l of the stored word takes word l / 4,
    // i.e. element 4d + e / 4 of its dual group (d = l / 16, e = l % 16) once every dual group holds the OR over k
    const uchar256 e8 = (lid8 >> 2) & 15, d8 = (lid8 >> 6) & 3, q8 = (d8 << 2) + (e8 >> 2);
    const uchar256 WPAT = ((q8 & 7) | ((q8 >> 3) << 5)) | 0x80;
    const int64 lsh = lane >> 4;
    const short128 one16 = 1, zero16 = 0;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int cb = s_i32_ld_g(gen_addr(zc, cursor_big));
    const int NW = get_dim_size(qsel, 1);
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        const int p = cur + t;
        const int nb = mode == 0 ? (p + 1) >> 2 : cb >> 2;
        const int K = mode == 0 ? nblk : nblk - (((p + 1) >> 2) - nb);
        const int hlim = mode == 0 ? p + 1 : cb;              // positions the softmax may read for this row
        if (mode == 0 && nb <= K) continue;                    // dense: the softmax skips the mask
        int64 tau, kap3;
        if (nb <= K) { tau = -1; kap3 = 0; }                   // mode 1 dense: every block
        else if (K <= 0) { tau = 0x7FFFFFFF; kap3 = 0x7FFFFFFF; }   // mode 1, no budget left for the history: no block
        else {
            // VLM vectors in use, a multiple of 4 (the count loop's unroll; lanes past nb hold the sentinel)
            const int nvv = ((((nb + 127) >> 7) + 3) & ~3) < NVV ? ((((nb + 127) >> 7) + 3) & ~3) : NVV, nbv = nvv << 7;
            int64 kap1, kap2; float64 cab;
            const float64 kf = (float)K;
            FILL(K1)
            BISECT(K1, -32768, 32767, 16, kf, kap1, cab)
            const float64 kf2 = kf - cab;
            FILL(K2)
            BISECT(K2, 0, 32767, 15, kf2, kap2, cab)
            const float64 kf3 = kf2 - cab;
            tau = ((kap1 + 32768) << 15) | kap2;
            int nit3 = 0; while ((1 << nit3) < nb) ++nit3;
            FILL(K3)
            BISECT(K3, -32767, nb - 32768, nit3, kf3, kap3, cab)
        }
        const int nwr = ((hlim + 2047) >> 11) < NW ? ((hlim + 2047) >> 11) : NW;
#ifdef BITMAP
        int64 bmv = 0;
#endif
        for (int w = 0; w < nwr; ++w) {
            int64 acc = 0;
            for (int i = 0; i < 8; ++i) {
                const int b0 = 512 * w + 64 * i;
                int5 c = {b0, t, 0, 0, 0}; int64 u = UBITS(v_f32_ld_tnsr_b(c, iss));
                int64 bb = lane + b0;
                int64 sv = v_i32_sel_grt_i32_b(u, tau, 1, 0);
                sv = v_i32_sel_eq_i32_b(u, tau, v_i32_sel_geq_i32_b((nb - 32768) - bb, kap3, 1, 0), sv);
#ifdef BITMAP
                if (mode == 0) {   // block bitmap word of 32 blocks: OR over the lanes of each half (dual groups 0+1, 2+3)
                    int64 x = v_i32_shl_b(v_i32_sel_less_i32_b(bb, nb, sv, 0), lane & 31);
                    x = x | MOVDG_ALLI(x, 1, 0, 3, 2);
                    x = x | v_i32_shuffle_b(x, P8, 0, x); x = x | v_i32_shuffle_b(x, P4, 0, x);
                    x = x | v_i32_shuffle_b(x, P2, 0, x); x = x | v_i32_shuffle_b(x, P1, 0, x);
                    const int gi = 8 * w + i;
                    bmv = v_i32_sel_eq_i32_b(lane & 31, gi & 31, x, bmv);
                    if ((gi & 31) == 31 || (w == nwr - 1 && i == 7)) { int5 bc = {0, gi >> 5, t, 0, 0}; v_i32_st_tnsr(bc, qbm, bmv); bmv = 0; }
                }
#endif
                sv = v_i32_sel_less_i32_b(bb, nb, sv, 1);
                acc = acc | v_i32_shl_b(sv, lsh + 4 * i);
            }
            acc = acc | MOVDG_ALLI(acc, 1, 2, 3, 0);
            acc = acc | MOVDG_ALLI(acc, 2, 3, 0, 1);
            int64 wv = v_i32_shuffle_b(acc, WPAT, 0, acc);
            int5 oc = {0, w, t, 0, 0}; v_i32_st_tnsr(oc, qsel, wv);
        }
    }
}
