// gaudium_attn_softmax_chunk: chunked two-pass masked softmax with per-chunk statistics (the flash-decoding form of
// gaudium_attn_softmax; with nch = 1 and C = Lmax it is the plain two-pass "online" softmax).
//  in0 scores f32 {C, R, B}   B = G kv groups x nch chunks, CHUNK-MAJOR b = ch*G + g (G = B / nch, 4 in the model);
//                             R = 6T rows per group, row = hl*T + t
//  in1 cursor int32 {1}
//  out0 P bf16 {C, R, B}:  exp(x - m) over valid key positions, 0 elsewhere; interleaved order inside each 128-block exactly
//                          like gaudium_attn_softmax (bf16 lane m <- GEMM column m even ? m/2 : 64 + (m-1)/2), matched by the V cache rows
//  out1 stats f32 {64, 2, R, B}: [lane][0] = m (row-chunk max, all 64 lanes equal), [lane][1] = s (sum of the valid exp)
//  chunk ch covers absolute positions [ch*C, ch*C + C). Valid positions of row t:
//      mode 0:  pos < cursor + coff + t + 1   (a pass that attends to its own rows: verify, prefill tile, host-driven draft)
//      mode 1:  pos < cursor + coff           (engine MTP passes: the big cache holds the history below the step cursor only;
//                                              the rows of the current step come from the fresh window, gaudium_attn_softmax_fresh)
//  A fully masked chunk writes m = -1e30, s = 0, P = 0; gaudium_attn_combine then gives it weight exp(-1e30 - M) = 0.
//  EXIT PORT (opt/attn-chunk-exit): a PARTIALLY valid chunk also skips its masked tail: both sweeps stop at
//  jlim = roundup(nvalid, 256) and P blocks [jlim, C) are zero-stored without loading scores. Bit-identical to
//  the base kernel: masked lanes contributed exactly 0 to m/s/P before (sel -> NEG / 0), and +0.0 accumulation
//  is exact, so every P byte and both stats are unchanged. Only the boundary chunk's tail sweeps are saved.
//  The normalisation (1/s) is NOT applied here: the PV GEMM produces unnormalised partials that the combine kernel scales.
//  Two sweeps per row-chunk (max; exp + sum + store) instead of three, and 4 independent accumulators per sweep so that
//  4 x 256 B loads are in flight per TPC (g2-opt-manual 7.6: the loads-in-flight lever doubled a streaming kernel's rate).
//  index space 2-D: dim0 = R (rows), dim1 = B.   scalars: T, nch, C, coff, mode   (C multiple of 256)
//  -DTREE build (DFlash2 draft-tree verify, integration branch; the P14 original is attn_softmax_tree of feature/dflash2): + in2 anc
//  int32 {T}: the T rows are the nodes of a draft tree at KV slots cursor + j; row t sees the history (pos < cursor) and the slots
//  cursor + j whose bit j is set in anc[t] (t itself and its ancestors). mode is ignored (always this mask). On a pure chain
//  (anc[t] = bits 0..t) the valid set equals mode 0's and every operation is the same, so P and the stats are bit-identical.
//  -DFASTEXP build (prefill recipes only; decode keeps the plain build): the same two sweeps, with
//    (a) no mask in fully valid chunks (nvalid >= C: every column valid, the sel is the identity there), and
//    (b) exp(d), d = x - m <= 0, as Cody-Waite + a degree-6 polynomial: d clamped at -87, z = rint(d log2 e),
//        r = d - z ln2 (two-part ln2), e^r by Horner (relative error 1.05e-7 on |r| <= ln2/2), 2^z added to the exponent
//        bits: 13 VPU ops instead of v_exp_f32's 29 (tpc-special.h: "exp_f32 VPU ops = exp_fast_f32 VPU ops + 3 = 26+3"),
//        max error 1.5 ulp (f32) on d in [-87, 0] (host emulation), the vendor function's class. P, m, s keep their meaning.
//  -DQSA builds (Qwen3.8-Flash-Next sparse attention, docs/notes/engine-qsa.md): + in2 qsel int32 {64, NW, T} after the cursor (before
//  kv8 in the KV8 builds) and one more scalar qfull (last). Valid(pos, t) = the causal rule above AND bit ((pos % 2048) / 64) of
//  qsel[pos % 64, pos / 2048, t] (the row's selected blocks + its tail, written by gaudium_qsa_select). Mode 0 rows with
//  cursor + coff + t + 1 <= qfull (qfull != 0) are dense: they skip the mask and run exactly the non-QSA code (bit-identical);
//  mode 1 rows always read it. Not combined with -DTREE. Chunk bases must be multiples of 2048 (C % 2048 == 0, or one chunk).
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
// Decode exp (research/exp_bug): the vendor v_exp_f32 is off by a factor of ~2 at exactly 42 inputs in [-104, 0]
// (one per odd k in -121..-1, x log2 e just below k; x = -0.693147182 gives 1.0; exhaustive card scan == libtpcsim). DEXP is the
// FASTEXP formula of the prefill build (EXPD below: max 1 ulp on [-87, 0], none of those errors, 13 VPU ops instead of 29);
// -DVENDOR_EXP restores v_exp_f32 (A/B builds).
// e^d for d <= 0 (see the header): 13 VPU ops after the clamp
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#ifdef VENDOR_EXP
#define DEXP(x) v_exp_f32(x)
#else
#define DEXP(x) EXPD(x)
#endif
// -DUSE_EXP4 (prefill FASTEXP build only, OPTIONAL numerics change): a bf16-grade exp for the P / s computation: one
// f32 ln2 constant in the Cody-Waite step (|z| <= 126: r off by at most 2.4e-7) and a degree-4 minimax polynomial on |r| <= ln2/2
// (relative error 2.6e-6 against exp, i.e. ~40 f32 ulp but 1/750 of a bf16 ulp): 11 VPU ops instead of EXPD's 14. P is stored in bf16, so
// the stored values round the same way as EXPD's in all but ~0.1 % of the elements; s (f32) moves by ~1e-6 relative. NOT bit-identical:
// token forks at bf16 near ties, to be traced like the FASTEXP forks were (docs/notes/engine-prefill.md 3).
#define EXP4(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f);     float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0);     float64 _r = v_f32_mac_b(_z, -0.6931471824645996f, _d, 0, 1, 0);     float64 _p = v_f32_mac_b(_r, 0.041458321229633825f, (float64)0.16790863408841236f, 0, 1, 0);     _p = v_f32_mac_b(_p, _r, (float64)0.5000435895965287f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.9999634315325933f, 0, 1, 0);     _p = v_f32_mac_b(_p, _r, (float64)0.9999992619064664f, 0, 1, 0);     int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0);     int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#if defined(USE_EXP4) && defined(FASTEXP)   // (the flag is USE_EXP4: `defined(EXP4)` would be true for every build once the macro above exists)
#undef DEXP
#define DEXP(x) EXP4(x)
#endif
// -DUSE_EXP5 (prefill FASTEXP build only, OPTIONAL numerics change, GAUDIUM_PREFILL_FASTEXP=3): the bf16-grade exp with fewer VPU
// operations: 9 after the subtraction instead of EXP4's 11 and EXPD's 14.
//   u = d log2 e + 1.5 * 2^23 in ONE mac: at that magnitude an f32 holds integers only, so the add rounds to z + 1.5 * 2^23 (no
//       nearbyint), and the low mantissa bits of u ARE z: (bits(u) << 23) = z << 23 (the 2^22 of the mantissa and the exponent
//       field shift out), so the f32 -> i32 convert goes too. z in [-126, 0] (d clamped at -87).
//   r = d - z ln2 (one f32 ln2, as EXP4), e^r by a degree-3 minimax polynomial on |r| <= ln2 / 2: relative error 7.5e-5 against
//       exp, 1/27 to 1/53 of a bf16 half ulp, so about 2 % of the stored P values move by one bf16 ulp; s (f32) moves by < 1e-4.
// NOT bit-identical to either other build: traced like them before it may become a default.
#define EXP5(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _u = v_f32_mac_b(_d, 1.44269504088896341f, (float64)12582912.0f, 0, 1, 0); \
    float64 _z = _u - 12582912.0f; \
    float64 _r = v_f32_mac_b(_z, -0.6931471824645996f, _d, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.16566742956638336f, (float64)0.5049682259559631f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.00016450881958f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.9999279379844666f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(*(int64*)&_u, 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
// -DUSE_EXP6: EXP5's rounding (no nearbyint, no convert) with EXP4's degree-4 polynomial: EXP4's accuracy (2.6e-6 relative) for one operation less.
#define EXP6(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _u = v_f32_mac_b(_d, 1.44269504088896341f, (float64)12582912.0f, 0, 1, 0); \
    float64 _z = _u - 12582912.0f; \
    float64 _r = v_f32_mac_b(_z, -0.6931471824645996f, _d, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.041458321229633825f, (float64)0.16790863408841236f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.5000435895965287f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.9999634315325933f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.9999992619064664f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(*(int64*)&_u, 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#if defined(USE_EXP6) && defined(FASTEXP)
#undef DEXP
#define DEXP(x) EXP6(x)
#endif
#if defined(USE_EXP5) && defined(FASTEXP)
#undef DEXP
#define DEXP(x) EXP5(x)
#endif
// -DKV8 builds: the scores are the RAW fp8 GEMM sums (plain batch_gemm on fp8 q and K: no scale inputs, so the graph
// compiler adds no scaling pass over the scores; research/probes/p24_fp8_kv.cpp) and this kernel applies the layer's pow2 score scale
// sqk = kv8[16] inside the exp: d = (x - m) * sqk (bit-identical to scaling the scores first: a pow2 commutes with the f32 rounding).
// P is FP8 (fmt 0 = fp8_143, 1 = fp8_152; scalar after the others) for the fp8 PV GEMM: one 256-byte vector per 256 columns, byte 4l + k
// <- column 64k + l of the block (the V cache rows follow: attn_prep VPERM8), scaled by the kv group's P multiplier kv8[12 + g]; the PV
// sums are raw too, so the stats carry the PV scale instead: m in true units (m_raw * sqk), s = sum(e) * kv8[17] (= 1 / spv, a pow2),
// which the combine's division cancels exactly. kv8 f32 {32}: one more input after the others. Rounded half to even, saturating.
#ifdef KV8
#define SQK(d) ((d) * sqk8)
#define PST(oc_, x0, x1, x2, x3) do { float256 _f; _f.v1 = (x0) * pm8; _f.v2 = (x1) * pm8; _f.v3 = (x2) * pm8; _f.v4 = (x3) * pm8; \
    if (fmt) v_h8_st_tnsr((oc_), P, v_convert_f32_to_h8_all_b(_f, SW_RHNE | SW_CLIP_FP)); \
    else     v_f8_st_tnsr((oc_), P, v_convert_f32_to_f8_all_b(_f, SW_RHNE | SW_CLIP_FP)); } while (0)
#define PZERO(row_, b_, j0_) for (int _j = (j0_); _j < C; _j += 256) { int5 _oc = {_j, (row_), (b_), 0, 0}; char256 _z = 0; v_i8_st_tnsr(_oc, P, _z); }
#else
#define SQK(d) (d)
#endif
#if defined(QSA) && defined(TREE)
#error "QSA and TREE are not combined"
#endif
#ifdef QSA
// the QSA bit of in-chunk column jj (= j + 64k, j a multiple of 256 inside the current 2048-group) from the group's word vector qw
#define QBIT(jj) v_i32_and_b(qw, 1 << (((jj) >> 6) & 31))
#define QSEL(x, jj, fill) v_f32_sel_eq_i32_b(QBIT(jj), 0, (fill), (x))
#define QWORD(jg_) ({ int5 _qc = {0, (base + (jg_)) >> 11, t, 0, 0}; v_i32_ld_tnsr_b(_qc, qsel); })
#endif
#ifdef TREE
// x where in-chunk column (lane + j) is valid for this row, else fill: history below hv, tree slot hv + jj iff bit jj of am
static inline float64 tree_sel(float64 x, int64 col, int j, int hv, int N, int am, float64 fill)
{
    float64 r = v_f32_sel_less_i32_b(col, hv, x, fill);
    if (j + 64 > hv && j < hv + N) {
        // bit (col - hv) of am for every lane at once (was: one select per set ancestor bit, up to N per vector and sweep; level-1
        // traces 2026-11-10: 6 -> 41 us/layer from 16 chain rows to 15 + 7 leaves). Same lanes selected, so bit-identical.
        const int64 amv = am;
        const int64 sh = v_i32_min_b(v_i32_max_b(col - hv, 0), 31);
        int64 bit = v_i32_and_b(v_i32_shr_b(amv, sh), 1);
        bit = v_i32_sel_less_i32_b(col, hv, 0, bit);            // history lanes: already decided by the bound
        bit = v_i32_sel_less_i32_b(col, hv + N, bit, 0);        // lanes past the tree slots
        r = v_f32_sel_eq_i32_b(bit, 1, x, r);
    }
    return r;
}
#define MASKED(x, jj, fill) tree_sel((x), lane + (jj), (jj), hv, T, am, (fill))
// split sweeps: history blocks use HSEL (the history bound alone, no per-vector overlap test); tree blocks keep MASKED
#define HSEL(x, jj, fill) v_f32_sel_less_i32_b(lane + (jj), hv, (x), (fill))
#else
#define MASKED(x, jj, fill) v_f32_sel_less_i32_b(lane + (jj), nvalid, (x), (fill))
#endif

#ifdef KV8
#define STATS_ST(sc_, m_, s_) do { v_f32_st_tnsr((sc_), stats, (m_) * sqk8); (sc_)[1] = 1; v_f32_st_tnsr((sc_), stats, (s_) * sm8); } while (0)
#else
#define STATS_ST(sc_, m_, s_) do { v_f32_st_tnsr((sc_), stats, (m_)); (sc_)[1] = 1; v_f32_st_tnsr((sc_), stats, (s_)); } while (0)
#endif
#if !defined(FASTEXP) || defined(STAGE)
__local__ float64 XV[64];   // VLM staging of one row-chunk's scores (C <= 4096; the glue refuses larger chunks for this build)
#endif
#if defined(PIPE) && defined(FASTEXP) && !defined(QSA) && !defined(KV8)
// -DPIPE (prefill FASTEXP build): SOFTWARE-PIPELINED SWEEPS over the rows of a fully valid batch (a history chunk: every row of the
// tile sees all C columns). Sweep 1 of a row-chunk is load + max: almost no VPU work, so on the card it runs at the latency of its HBM
// loads, while sweep 2 (the exp) is VPU-bound and leaves the load and store pipes idle. Here the loads, the VLM staging and the max of
// row r + 1 ride inside sweep 2 of row r (one more VPU op per vector), so only the first row of a TPC's range pays for a sweep of
// its own. Per row-chunk the operations, their order and their accumulators are those of the unpipelined path: P and the stats are
// bit-identical (tests/sim/sim_softmaxp.cpp). Two VLM buffers of C <= 4096 f32 (32 KB): a larger chunk takes the unpipelined path.
#define PIPE_ON 1
__local__ float64 XP[128];
#endif
#if defined(QSA) && defined(KV8) && defined(FASTEXP)
void main(tensor scores, tensor cursor, tensor qsel, tensor kv8, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int pbase, int fmt, int qfull)
#elif defined(QSA) && defined(KV8)
void main(tensor scores, tensor cursor, tensor qsel, tensor kv8, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int fmt, int qfull)
#elif defined(QSA) && defined(FASTEXP)
void main(tensor scores, tensor cursor, tensor qsel, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int pbase, int qfull)
#elif defined(QSA)
void main(tensor scores, tensor cursor, tensor qsel, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int qfull)
#elif defined(KV8) && defined(TREE)
void main(tensor scores, tensor cursor, tensor anc, tensor kv8, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int fmt)
#elif defined(KV8) && defined(FASTEXP)
void main(tensor scores, tensor cursor, tensor kv8, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int pbase, int fmt)
#elif defined(KV8)
void main(tensor scores, tensor cursor, tensor kv8, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int fmt)
#elif defined(TREE)
void main(tensor scores, tensor cursor, tensor anc, tensor P, tensor stats, int T, int nch, int C, int coff, int mode)
#elif defined(FASTEXP)
// pbase: absolute key POSITION of this node's first chunk (feature/prefill chunk groups: the attention core split into groups of chunks so
// the MME's QK / PV of one group run while the TPC does another group's softmax; 0 for an ungrouped node). feature/prefill-tpc: a
// position rather than a chunk index, so a node over the tile's DIAGONAL chunk may carry only its valid key columns (C = the extent,
// a multiple of 256) while the chunk's base stays the full chunk's: base = pbase + (b / G) * C.
void main(tensor scores, tensor cursor, tensor P, tensor stats, int T, int nch, int C, int coff, int mode, int pbase)
#else
void main(tensor scores, tensor cursor, tensor P, tensor stats, int T, int nch, int C, int coff, int mode)
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
    const int G = get_dim_size(scores, 2) / nch;
    const float NEG = -1e30f;
#ifdef KV8
    int5 k16 = {16, 0, 0, 0, 0}; const float sqk8 = s_f32_ld_g(gen_addr(k16, kv8)); k16[0] = 17; const float sm8 = s_f32_ld_g(gen_addr(k16, kv8));
    int b8 = -1; float pm8 = 0.f;
#endif

#ifdef FASTEXP
    for (int b = idx[1]; b < idx[1] + isz[1]; ++b) {
        const int base = pbase + (b / G) * C;
#ifdef PIPE_ON
        #ifdef PIPE8
#define PIPE_COK ((C & 511) == 0)
#else
#define PIPE_COK 1
#endif
        if ((mode == 0 ? cur + 1 : cur) - base >= C && isz[0] > 0 && C <= 4096 && PIPE_COK) {   // row t = 0 sees the whole chunk, so every row does
            const int row0 = idx[0], row1 = idx[0] + isz[0];
            int5 c = {0, row0, b, 0, 0};
            float64 n0 = NEG, n1 = NEG, n2 = NEG, n3 = NEG;
            for (int j = 0; j < C; j += 256) {   // sweep 1 of the range's first row
                c[0] = j;       float64 y0 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 64;  float64 y1 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 128; float64 y2 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 192; float64 y3 = v_f32_ld_tnsr_b(c, scores);
                XP[j >> 6] = y0; XP[(j >> 6) + 1] = y1; XP[(j >> 6) + 2] = y2; XP[(j >> 6) + 3] = y3;
                n0 = v_f32_max_b(n0, y0); n1 = v_f32_max_b(n1, y1); n2 = v_f32_max_b(n2, y2); n3 = v_f32_max_b(n3, y3);
            }
            float64 m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(n0, n1), v_f32_max_b(n2, n3)), P8, P4, P2, P1);
            int xb = 0;   // VLM buffer (0 or 64) holding the current row's scores
            // one 256-column block of sweep 2 (row `row`, scores in VLM at xb): exp, sums, P
#define PIPE_S2(j_) { const int q_ = (j_) >> 6; \
                        float64 x0 = XP[xb + q_], x1 = XP[xb + q_ + 1], x2 = XP[xb + q_ + 2], x3 = XP[xb + q_ + 3]; \
                        float64 e0 = DEXP(SQK(x0 - m)), e1 = DEXP(SQK(x1 - m)), e2 = DEXP(SQK(x2 - m)), e3 = DEXP(SQK(x3 - m)); \
                        s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3; \
                        float128 pa; pa.v1 = e0; pa.v2 = e1; float128 pb; pb.v1 = e2; pb.v2 = e3; \
                        int5 oc = {(j_), row, b, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa)); \
                        oc[0] = (j_) + 128; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb)); }
            // the same block of sweep 1 of the next row (coordinates cn): loads, VLM staging at nb, max
#define PIPE_S1(j_) { const int q_ = (j_) >> 6; \
                        cn[0] = (j_);       float64 y0 = v_f32_ld_tnsr_b(cn, scores); \
                        cn[0] = (j_) + 64;  float64 y1 = v_f32_ld_tnsr_b(cn, scores); \
                        cn[0] = (j_) + 128; float64 y2 = v_f32_ld_tnsr_b(cn, scores); \
                        cn[0] = (j_) + 192; float64 y3 = v_f32_ld_tnsr_b(cn, scores); \
                        XP[nb + q_] = y0; XP[nb + q_ + 1] = y1; XP[nb + q_ + 2] = y2; XP[nb + q_ + 3] = y3; \
                        n0 = v_f32_max_b(n0, y0); n1 = v_f32_max_b(n1, y1); n2 = v_f32_max_b(n2, y2); n3 = v_f32_max_b(n3, y3); }
            for (int row = row0; row < row1; ++row) {
                int5 sc = {0, 0, row, b, 0};
                float64 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                if (row + 1 < row1) {   // sweep 2 of this row with sweep 1 of the next one inside it
                    const int nb = xb ^ 64;
                    int5 cn = {0, row + 1, b, 0, 0};
                    n0 = NEG; n1 = NEG; n2 = NEG; n3 = NEG;
#ifdef PIPE8   // eight exp chains per iteration (two 256-column blocks): for an exp whose chain is too short for four to fill the VPU
                    for (int j = 0; j < C; j += 512) { PIPE_S1(j) PIPE_S1(j + 256) PIPE_S2(j) PIPE_S2(j + 256) }
#else
                    for (int j = 0; j < C; j += 256) { PIPE_S1(j) PIPE_S2(j) }
#endif
                    // both reductions in one block: two independent latency chains that the scheduler interleaves
                    float64 s = REDUCE64((s0 + s1) + (s2 + s3), P8, P4, P2, P1);
                    float64 mn = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(n0, n1), v_f32_max_b(n2, n3)), P8, P4, P2, P1);
                    STATS_ST(sc, m, s);
                    m = mn; xb = nb;
                } else {                // the range's last row: sweep 2 alone
#ifdef PIPE8
                    for (int j = 0; j < C; j += 512) { PIPE_S2(j) PIPE_S2(j + 256) }
#else
                    for (int j = 0; j < C; j += 256) { PIPE_S2(j) }
#endif
                    float64 s = REDUCE64((s0 + s1) + (s2 + s3), P8, P4, P2, P1);
                    STATS_ST(sc, m, s);
                }
            }
            continue;
        }
#endif
        for (int row = idx[0]; row < idx[0] + isz[0]; ++row) {
#else
    // FLAT index space: point pp = b * R + row, so the 24 TPCs get equal row counts (2-D {R, B} split by batches left
    // 36 batches of a 32K bucket on 24 TPCs: 13.5 .. 50.6 us per instance); every row-chunk is computed exactly as before
    const int RR = get_dim_size(scores, 1);
    for (int pp = idx[0]; pp < idx[0] + isz[0]; ++pp) {
        const int b = pp / RR, row = pp % RR;
        const int base = (b / G) * C;
        {
#endif
            const int t = row % T;
#ifdef QSA
            const int qon = (mode != 0) || (qfull == 0) || (cur + t + 1 > qfull);   // dense rows (mode 0, nvis <= qfull) run the plain code
#endif
#ifdef KV8
            if (b != b8) { b8 = b; int5 kc = {12 + b % G, 0, 0, 0, 0}; pm8 = s_f32_ld_g(gen_addr(kc, kv8)); }   // per batch, not per row
#endif
#ifdef TREE
            int5 ac = {t, 0, 0, 0, 0};
            const int am = s_i32_ld_g(gen_addr(ac, anc));
            const int hv = cur - base;                                      // history columns of this chunk: [0, hv)
            const int nvalid = cur + T - base;                              // upper bound of the valid columns (tree slots end at cur + T)
#else
            const int nvalid = (mode == 0 ? cur + t + 1 : cur) - base;      // valid positions inside this chunk; <= 0 means all masked
#endif
            int5 c = {0, row, b, 0, 0};
            int5 sc = {0, 0, row, b, 0};
            if (nvalid <= 0) {
#ifdef KV8
                PZERO(row, b, 0)
#else
                float128 zz; zz.v1 = 0.f; zz.v2 = 0.f;
                for (int j = 0; j < C; j += 128) { int5 oc = {j, row, b, 0, 0}; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(zz)); }
#endif
                float64 mneg = NEG, zero = 0.f;
                v_f32_st_tnsr(sc, stats, mneg); sc[1] = 1; v_f32_st_tnsr(sc, stats, zero);
                continue;
            }
#ifdef FASTEXP
            float64 m0 = NEG, m1 = NEG, m2 = NEG, m3 = NEG;
            float64 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            // (a) EXIT PORT in the masked path too: a partially valid chunk (the tile's diagonal chunk) sweeps only
            // the 256-blocks that hold a valid column and zero-stores the rest without loading scores (bit-identical: masked lanes gave
            // NEG to the max and exactly 0 to the sums before). At 2K every chunk of every tile is such a chunk (on average a quarter
            // valid). (b) -DSTAGE: sweep 1 keeps the row-chunk's scores in VLM (16 KB at C = 4096) and sweep 2 reads them back instead
            // of HBM, as the decode build does: same values, same order (bit-identical), one HBM read of the scores instead of two.
#ifdef UNR
#define STR2(x) #x
#define STR1(x) STR2(x)
#define UNROLL_SWEEP _Pragma(STR1(unroll UNR))
#else
#define UNROLL_SWEEP
#endif
#ifdef STAGE
#define XLD(c_, j_, x0_, x1_, x2_, x3_) x0_ = XV[(j_) >> 6]; x1_ = XV[((j_) >> 6) + 1]; x2_ = XV[((j_) >> 6) + 2]; x3_ = XV[((j_) >> 6) + 3]
#define XST(j_, x0_, x1_, x2_, x3_) XV[(j_) >> 6] = x0_; XV[((j_) >> 6) + 1] = x1_; XV[((j_) >> 6) + 2] = x2_; XV[((j_) >> 6) + 3] = x3_
#else
#define XLD(c_, j_, x0_, x1_, x2_, x3_) c_[0] = (j_); x0_ = v_f32_ld_tnsr_b(c_, scores); c_[0] = (j_) + 64; x1_ = v_f32_ld_tnsr_b(c_, scores); \
            c_[0] = (j_) + 128; x2_ = v_f32_ld_tnsr_b(c_, scores); c_[0] = (j_) + 192; x3_ = v_f32_ld_tnsr_b(c_, scores)
#define XST(j_, x0_, x1_, x2_, x3_)
#endif
            const int jlim = nvalid >= C ? C : ((nvalid + 255) & ~255);
#ifdef QSA
            if (nvalid >= C && !qon) {   // fully valid chunk: no mask (QSA rows take the masked path below)
#else
            if (nvalid >= C) {   // fully valid chunk: no mask
#endif
                UNROLL_SWEEP
                for (int j = 0; j < C; j += 256) {
                    c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores);
                    c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
                    c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores);
                    c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
                    XST(j, x0, x1, x2, x3);
                    m0 = v_f32_max_b(m0, x0); m1 = v_f32_max_b(m1, x1); m2 = v_f32_max_b(m2, x2); m3 = v_f32_max_b(m3, x3);
                }
                float64 m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
                UNROLL_SWEEP
                for (int j = 0; j < C; j += 256) {
                    float64 x0, x1, x2, x3; XLD(c, j, x0, x1, x2, x3);
                    float64 e0 = DEXP(SQK(x0 - m)), e1 = DEXP(SQK(x1 - m)), e2 = DEXP(SQK(x2 - m)), e3 = DEXP(SQK(x3 - m));
                    s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3;
#ifdef KV8
                    int5 oc = {j, row, b, 0, 0}; PST(oc, e0, e1, e2, e3);
#else
                    float128 pa; pa.v1 = e0; pa.v2 = e1;
                    float128 pb; pb.v1 = e2; pb.v2 = e3;
                    int5 oc = {j, row, b, 0, 0};
                    v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
                    oc[0] = j + 128;
                    v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
#endif
                }
                float64 s = REDUCE64((s0 + s1) + (s2 + s3), P8, P4, P2, P1);
                STATS_ST(sc, m, s);
                continue;
            }
#ifdef QSA
            float64 m;
            if (qon) {   // the masked sweeps with the QSA bit: same blocks, order and accumulators as the plain masked path
                for (int jg = 0; jg < jlim; jg += 2048) {
                    const int64 qw = QWORD(jg); const int je = jg + 2048 < jlim ? jg + 2048 : jlim;
                    for (int j = jg; j < je; j += 256) {
                        c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores);
                        c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
                        c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores);
                        c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
                        XST(j, x0, x1, x2, x3);
                        m0 = v_f32_max_b(m0, QSEL(v_f32_sel_less_i32_b(lane + j, nvalid, x0, NEG), j, NEG));
                        m1 = v_f32_max_b(m1, QSEL(v_f32_sel_less_i32_b(lane + (j + 64), nvalid, x1, NEG), j + 64, NEG));
                        m2 = v_f32_max_b(m2, QSEL(v_f32_sel_less_i32_b(lane + (j + 128), nvalid, x2, NEG), j + 128, NEG));
                        m3 = v_f32_max_b(m3, QSEL(v_f32_sel_less_i32_b(lane + (j + 192), nvalid, x3, NEG), j + 192, NEG));
                    }
                }
                m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
                for (int jg = 0; jg < jlim; jg += 2048) {
                    const int64 qw = QWORD(jg); const int je = jg + 2048 < jlim ? jg + 2048 : jlim;
                    for (int j = jg; j < je; j += 256) {
                        float64 x0, x1, x2, x3; XLD(c, j, x0, x1, x2, x3);
                        float64 e0 = QSEL(v_f32_sel_less_i32_b(lane + j,         nvalid, DEXP(SQK(x0 - m)), 0.f), j, 0.f);
                        float64 e1 = QSEL(v_f32_sel_less_i32_b(lane + (j + 64),  nvalid, DEXP(SQK(x1 - m)), 0.f), j + 64, 0.f);
                        float64 e2 = QSEL(v_f32_sel_less_i32_b(lane + (j + 128), nvalid, DEXP(SQK(x2 - m)), 0.f), j + 128, 0.f);
                        float64 e3 = QSEL(v_f32_sel_less_i32_b(lane + (j + 192), nvalid, DEXP(SQK(x3 - m)), 0.f), j + 192, 0.f);
                        s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3;
#ifdef KV8
                        int5 oc = {j, row, b, 0, 0}; PST(oc, e0, e1, e2, e3);
#else
                        float128 pa; pa.v1 = e0; pa.v2 = e1;
                        float128 pb; pb.v1 = e2; pb.v2 = e3;
                        int5 oc = {j, row, b, 0, 0};
                        v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
                        oc[0] = j + 128;
                        v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
#endif
                    }
                }
            } else {
            UNROLL_SWEEP
            for (int j = 0; j < jlim; j += 256) {
                c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
                XST(j, x0, x1, x2, x3);
                m0 = v_f32_max_b(m0, v_f32_sel_less_i32_b(lane + j, nvalid, x0, NEG));
                m1 = v_f32_max_b(m1, v_f32_sel_less_i32_b(lane + (j + 64), nvalid, x1, NEG));
                m2 = v_f32_max_b(m2, v_f32_sel_less_i32_b(lane + (j + 128), nvalid, x2, NEG));
                m3 = v_f32_max_b(m3, v_f32_sel_less_i32_b(lane + (j + 192), nvalid, x3, NEG));
            }
            m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
            UNROLL_SWEEP
            for (int j = 0; j < jlim; j += 256) {
                float64 x0, x1, x2, x3; XLD(c, j, x0, x1, x2, x3);
                float64 e0 = v_f32_sel_less_i32_b(lane + j,         nvalid, DEXP(SQK(x0 - m)), 0.f);
                float64 e1 = v_f32_sel_less_i32_b(lane + (j + 64),  nvalid, DEXP(SQK(x1 - m)), 0.f);
                float64 e2 = v_f32_sel_less_i32_b(lane + (j + 128), nvalid, DEXP(SQK(x2 - m)), 0.f);
                float64 e3 = v_f32_sel_less_i32_b(lane + (j + 192), nvalid, DEXP(SQK(x3 - m)), 0.f);
                s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3;
#ifdef KV8
                int5 oc = {j, row, b, 0, 0}; PST(oc, e0, e1, e2, e3);
#else
                float128 pa; pa.v1 = e0; pa.v2 = e1;
                float128 pb; pb.v1 = e2; pb.v2 = e3;
                int5 oc = {j, row, b, 0, 0};
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
                oc[0] = j + 128;
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
#endif
            }
            }
#else
            UNROLL_SWEEP
            for (int j = 0; j < jlim; j += 256) {
                c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
                XST(j, x0, x1, x2, x3);
                m0 = v_f32_max_b(m0, v_f32_sel_less_i32_b(lane + j, nvalid, x0, NEG));
                m1 = v_f32_max_b(m1, v_f32_sel_less_i32_b(lane + (j + 64), nvalid, x1, NEG));
                m2 = v_f32_max_b(m2, v_f32_sel_less_i32_b(lane + (j + 128), nvalid, x2, NEG));
                m3 = v_f32_max_b(m3, v_f32_sel_less_i32_b(lane + (j + 192), nvalid, x3, NEG));
            }
            float64 m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
            UNROLL_SWEEP
            for (int j = 0; j < jlim; j += 256) {
                float64 x0, x1, x2, x3; XLD(c, j, x0, x1, x2, x3);
                float64 e0 = v_f32_sel_less_i32_b(lane + j,         nvalid, DEXP(SQK(x0 - m)), 0.f);
                float64 e1 = v_f32_sel_less_i32_b(lane + (j + 64),  nvalid, DEXP(SQK(x1 - m)), 0.f);
                float64 e2 = v_f32_sel_less_i32_b(lane + (j + 128), nvalid, DEXP(SQK(x2 - m)), 0.f);
                float64 e3 = v_f32_sel_less_i32_b(lane + (j + 192), nvalid, DEXP(SQK(x3 - m)), 0.f);
                s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3;
#ifdef KV8
                int5 oc = {j, row, b, 0, 0}; PST(oc, e0, e1, e2, e3);
#else
                float128 pa; pa.v1 = e0; pa.v2 = e1;
                float128 pb; pb.v1 = e2; pb.v2 = e3;
                int5 oc = {j, row, b, 0, 0};
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
                oc[0] = j + 128;
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
#endif
            }
#endif
            {   // the masked tail: zero P without loading scores
#ifdef KV8
                PZERO(row, b, jlim)
#else
                float128 zz; zz.v1 = 0.f; zz.v2 = 0.f;
                for (int j = jlim; j < C; j += 128) {
                    int5 oc = {j, row, b, 0, 0};
                    v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(zz));
                }
#endif
            }
#else
            // in-chunk exit: sweeps stop at the last 256-block that holds a valid column
            const int jlim = nvalid >= C ? C : ((nvalid + 255) & ~255);
            // sweep 1: max over valid columns, 4 accumulators
            float64 m0 = NEG, m1 = NEG, m2 = NEG, m3 = NEG;
            // VLM STAGING: sweep 1 keeps the row-chunk's scores in VLM (C <= 4096 f32 = 16 KB), sweep 2 reads them back from
            // VLM instead of HBM: the same values in the same order (bit-identical P and stats), half the global loads of this latency-bound
            // kernel (decode at 32K: 80 us per verify layer at ~330 GB/s, docs/notes/engine-speed.md)
#define SWEEP1(J0, J1, SEL) for (int j = (J0); j < (J1); j += 256) { \
                c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores); \
                c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores); \
                c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores); \
                c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores); \
                XV[j >> 6] = x0; XV[(j >> 6) + 1] = x1; XV[(j >> 6) + 2] = x2; XV[(j >> 6) + 3] = x3; \
                m0 = v_f32_max_b(m0, SEL(x0, j, NEG)); \
                m1 = v_f32_max_b(m1, SEL(x1, (j + 64), NEG)); \
                m2 = v_f32_max_b(m2, SEL(x2, (j + 128), NEG)); \
                m3 = v_f32_max_b(m3, SEL(x3, (j + 192), NEG)); }
#ifdef TREE
            // history blocks [0, jt0): the history bound only; blocks [jt0, jlim) overlap the tree slots: the ancestor mask
            // (split as in feature/mtp-tree's kernel: no per-vector branch in the history sweep; same blocks, same order, same
            // accumulators as one masked sweep, so m, s and P are bit-identical to it)
            int jt0 = hv <= 0 ? 0 : (hv / 256) * 256; if (jt0 > jlim) jt0 = jlim;
            SWEEP1(0, jt0, HSEL)
            SWEEP1(jt0, jlim, MASKED)
#elif defined(QSA)
            // QSA rows: the same sweep per 2048-position group with the group's mask word (same blocks, order, accumulators)
#define QMASKED(x, jj, fill) QSEL(MASKED((x), (jj), (fill)), (jj), (fill))
            if (qon) { for (int jg = 0; jg < jlim; jg += 2048) { const int64 qw = QWORD(jg); const int je = jg + 2048 < jlim ? jg + 2048 : jlim; SWEEP1(jg, je, QMASKED) } }
            else { SWEEP1(0, jlim, MASKED) }
#else
            SWEEP1(0, jlim, MASKED)
#endif
            float64 m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
            // sweep 2: exp, sum, unnormalised bf16 store
            float64 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#ifdef KV8
#define PSTORE4(j_, e0, e1, e2, e3) { int5 oc = {(j_), row, b, 0, 0}; PST(oc, e0, e1, e2, e3); }
#else
#define PSTORE4(j_, e0, e1, e2, e3) { float128 pa; pa.v1 = e0; pa.v2 = e1; float128 pb; pb.v1 = e2; pb.v2 = e3; int5 oc = {(j_), row, b, 0, 0}; \
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa)); oc[0] = (j_) + 128; v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb)); }
#endif
#define SWEEP2(J0, J1, SEL) for (int j = (J0); j < (J1); j += 256) { \
                float64 x0 = XV[j >> 6], x1 = XV[(j >> 6) + 1], x2 = XV[(j >> 6) + 2], x3 = XV[(j >> 6) + 3]; \
                float64 e0 = SEL(DEXP(SQK(x0 - m)), j, 0.f); \
                float64 e1 = SEL(DEXP(SQK(x1 - m)), (j + 64), 0.f); \
                float64 e2 = SEL(DEXP(SQK(x2 - m)), (j + 128), 0.f); \
                float64 e3 = SEL(DEXP(SQK(x3 - m)), (j + 192), 0.f); \
                s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3; \
                PSTORE4(j, e0, e1, e2, e3) }
#ifdef TREE
            SWEEP2(0, jt0, HSEL)
            SWEEP2(jt0, jlim, MASKED)
#elif defined(QSA)
            if (qon) { for (int jg = 0; jg < jlim; jg += 2048) { const int64 qw = QWORD(jg); const int je = jg + 2048 < jlim ? jg + 2048 : jlim; SWEEP2(jg, je, QMASKED) } }
            else { SWEEP2(0, jlim, MASKED) }
#else
            SWEEP2(0, jlim, MASKED)
#endif
            // zero-store the masked tail without loading scores (bit-identical: those P bytes were 0 anyway)
            {
#ifdef KV8
                PZERO(row, b, jlim)
#else
                float128 zz; zz.v1 = 0.f; zz.v2 = 0.f;
                for (int j = jlim; j < C; j += 128) {
                    int5 oc = {j, row, b, 0, 0};
                    v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(zz));
                }
#endif
            }
#endif
            float64 s = REDUCE64((s0 + s1) + (s2 + s3), P8, P4, P2, P1);
            STATS_ST(sc, m, s);
        }
    }
}
