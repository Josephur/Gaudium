#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gdn_prep_dec_gaudi2.c: the DECODE / VERIFY build of gaudium_gdn_prep (integration). This is gdn_prep_gaudi2.c as of e10f332 (before
// the feature/prefill merge): no chunked-layout (bit5) path and no unrolled row loop. Those cost ~1 ms per K=4 decode step (33.24 vs 32.27
// ms, tokens identical; ref/integration/decode_ab.sh). The glue picks this build (gdn_prepd / gdn_prepsd / gdn_prept) for row-major
// nodes of <= 32 rows; prefill tiles use gdn_prep_gaudi2.c. Keep the two in step for anything but the prefill layout.
// gaudium_gdn_prep: depthwise causal conv (kernel 4) + SiLU over the mixed q|k|v projections, with conv state in/out,
// then per-head l2-norm of q and k (q also scaled by 1/sqrt(128)), GQA repeat 16 -> 48 heads, and re-layout into the
// f32 {128, 48, T} tensors the recurrence kernel reads. One index-space point = one 128-channel block (80 blocks:
// 16 q heads, 16 k heads, 48 v heads). z (channels 10240..16383 of qkvz) is left for gdn_post.
//
//  in0 qkvz   bf16 {16384, T}    in1 conv_w f32 {10240, 4} (tap 3 = current token)   in2 conv_state_in f32 {10240, 3}
//  in3 conv_b f32 {10240}        out0 q f32 {128,48,T}  out1 k f32 {128,48,T}  out2 v f32 {128,48,T}  out3 conv_state_out f32 {10240,3}
//  out4 cs_ckpt f32 {10240, 3, T}: the conv state after each row (for rollback to an accepted prefix)
//  scalar: T, flags (bit0: add conv bias, bit1: write cs_ckpt, bit4: skip the cs_out store, bit7: k and q into out0 as kq {128, 2T, HV, 1})
//  Built twice: plain, and -DSLOT with cs_in {10240, 3, NS} + int32 `slot` input (conv state read from slot slot[0]).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))

// sum of all 64 lanes, result broadcast to every lane. PAT8/4/2/1: xor-butterfly shuffles inside each 16-lane dual group.
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })

static inline uchar256 xor_pat(uchar256 l16, int s) {   // dest lane i <- source lane i ^ s within its dual group
    uchar256 src = l16 ^ (unsigned char)s;
    return ((src & 7) | ((src >> 3) << 5)) | 0x80;
}
static inline float64 silu(float64 x) { return x * v_sigmoid_f32(x); }

// one row: conv over the taps (A3_/B3_ = x[-3], A2_/B2_ = x[-2], A1_/B1_ = x[-1], XF_ = the row's own input), SiLU, l2-norm, outputs, and the
// row's conv-state checkpoint (x[-2], x[-1], x[row])
#define PREP_ROW(T_, A3_, B3_, A2_, B2_, A1_, B1_, XF_) do { \
            int5 xc = {c0, (T_), 0, 0, 0}; \
            XF_ = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz)); \
            float64 ya = v_f32_mac_b(w3a, XF_.v1, ba_), yb = v_f32_mac_b(w3b, XF_.v2, bb_); \
            ya = v_f32_mac_b(w2a, A1_, ya); yb = v_f32_mac_b(w2b, B1_, yb); \
            ya = v_f32_mac_b(w1a, A2_, ya); yb = v_f32_mac_b(w1b, B2_, yb); \
            ya = v_f32_mac_b(w0a, A3_, ya); yb = v_f32_mac_b(w0b, B3_, yb); \
            ya = silu(ya); yb = silu(yb); \
            if (kind == 2) { \
                int5 oc = {0, head, (T_), 0, 0}; \
                v_f32_st_tnsr(oc, v, ya); oc[0] = 64; v_f32_st_tnsr(oc, v, yb); \
            } else { /* q / k: the l2-norm (rsqrt table) waits for PREP_NORM, after every row's SiLU (sigmoid table) */ \
                YV[2 * ((T_) & 15)] = ya; YV[2 * ((T_) & 15) + 1] = yb; \
            } \
            if (flags & 2) { \
                int5 kc = {c0, 0, (T_), 0, 0}; \
                kc[1] = 0; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, A2_); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, B2_); \
                kc[1] = 1; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, A1_); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, B1_); \
                kc[1] = 2; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, XF_.v1); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, XF_.v2); \
            } \
        } while (0)
// ONE special-function table at a time (2026-11-11, level-1 traces): sigmoid (SiLU) and rsqrt (l2-norm) are LUT functions and a TPC holds
// one table; alternating them per row cost ~0.3 ms per 16-row step in this kernel plus ~10 us of idle time after the node. PREP_ROW keeps
// the q / k rows' SiLU outputs in local memory; PREP_NORM does the l2-norm, the GQA copies and the stores afterwards. Same operations per
// element, bit-identical.
__local__ float64 YV[32];   // 16 rows: a kernel that uses special functions has 16 KB of vector memory for static data and spills together
#define PREP_NORM(T_) do { \
            float64 ya = YV[2 * ((T_) & 15)], yb = YV[2 * ((T_) & 15) + 1]; \
            float64 ss = REDUCE64(ya * ya + yb * yb, P8, P4, P2, P1); \
            float64 r = v_rsqrt_f32(ss + eps); \
            if (kind == 0) r = r * qscale; \
            ya = ya * r; yb = yb * r; \
            if (flags & 128) {   /* feature/verify-mme: q is kq {128, 2T, HV, 1}, row T_ = k, row T + T_ = q (the MME operand layout); k is not written */ \
                int5 oc = {0, (kind == 0) ? T + (T_) : (T_), GDN_REP * head, 0, 0}; \
                for (int rep = 0; rep < GDN_REP; ++rep) { oc[2] = GDN_REP * head + rep; oc[0] = 0; v_f32_st_tnsr(oc, q, ya); oc[0] = 64; v_f32_st_tnsr(oc, q, yb); } \
            } else { \
                int5 oc = {0, GDN_REP * head, (T_), 0, 0}; \
                for (int rep = 0; rep < GDN_REP; ++rep) { \
                    oc[1] = GDN_REP * head + rep; \
                    if (kind == 0) { oc[0] = 0; v_f32_st_tnsr(oc, q, ya); oc[0] = 64; v_f32_st_tnsr(oc, q, yb); } \
                    else           { oc[0] = 0; v_f32_st_tnsr(oc, k, ya); oc[0] = 64; v_f32_st_tnsr(oc, k, yb); } \
                } \
            } \
        } while (0)
#ifdef TREE
// -DTREE (implies SLOT): + parent int32 {T} (row 0 = the root). Rows 0..NCH-1 form the chain prefix (parent = the row before) and use
// the conv ring; any other row (a tree leaf) is computed from its ancestors' input rows and does not touch the ring; its checkpoint is
// (x[parent's parent], x[parent], x[row]). Nothing is read back from global memory written by this instance (vector load after
// vector store is not coherent on the TPC; an earlier reload from cs_ckpt[.., p] gave timing-dependent leaf rows).
__local__ int PAR[32];
void main(tensor qkvz, tensor conv_w, tensor cs_in, tensor conv_b, tensor slot, tensor parent, tensor q, tensor k, tensor v, tensor cs_out, tensor cs_ckpt, int T, int flags)
#elif defined(SLOT)
void main(tensor qkvz, tensor conv_w, tensor cs_in, tensor conv_b, tensor slot, tensor q, tensor k, tensor v, tensor cs_out, tensor cs_ckpt, int T, int flags)
#else
void main(tensor qkvz, tensor conv_w, tensor cs_in, tensor conv_b, tensor q, tensor k, tensor v, tensor cs_out, tensor cs_ckpt, int T, int flags)
#endif
{
    const int5 idx = get_index_space_offset();
#if defined(SLOT) || defined(TREE)
    const int5 zc = {0, 0, 0, 0, 0};
    const int slot_in = s_i32_ld_g(gen_addr(zc, slot));
    int Te = T;                                     // gated rows (prompt-lookup step): slot {2} = [slot, valid rows]; rows >= Te are skipped
    if (get_dim_size(slot, 0) >= 2) { int5 c1 = {1, 0, 0, 0, 0}; const int nv = s_i32_ld_g(gen_addr(c1, slot)); if (nv < Te) Te = nv; }
#else
    const int slot_in = 0;
    const int Te = T;
#endif
    const int no_out = flags & 16;
    const int5 isz = get_index_space_size();
    const int has_bias = flags & 1;
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, qscale = 0.08838834764831845f;   // 1/sqrt(128)

#ifdef TREE
    // the chain prefix, from a local copy of `parent` (once per instance; scalar global loads are slow)
    int NCH = 1;
    for (int t = 0; t < T; ++t) { int5 pc = {t, 0, 0, 0, 0}; PAR[t] = s_i32_ld_g(gen_addr(pc, parent)); }
    while (NCH < T && PAR[NCH] == NCH - 1) ++NCH;
#endif
    // BALANCED ORDER: index point jj -> block j interleaves the 32 heavy q/k blocks (l2-norm + 3 GQA copies) with the 48 v
    // blocks, 2 + 3 per 5 points, so a TPC's contiguous range never holds only q/k blocks (8 us vs 2.8 us per instance at T = 5)
    for (int jj = idx[0]; jj < idx[0] + isz[0]; ++jj) {
        const int jg = jj / (2 + GDN_REP), jr = jj - (2 + GDN_REP) * jg, j = jr < 2 ? 2 * jg + jr : 2 * GDN_HK + GDN_REP * jg + (jr - 2);   // per key head: its q and k block, then its GDN_REP v blocks
        const int c0 = 128 * j;
        int5 wc = {c0, 0, 0, 0, 0};
        float64 w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b, ba_ = 0.f, bb_ = 0.f;
        wc[1] = 0; wc[0] = c0; w0a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w0b = v_f32_ld_tnsr_b(wc, conv_w);
        wc[1] = 1; wc[0] = c0; w1a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w1b = v_f32_ld_tnsr_b(wc, conv_w);
        wc[1] = 2; wc[0] = c0; w2a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w2b = v_f32_ld_tnsr_b(wc, conv_w);
        wc[1] = 3; wc[0] = c0; w3a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w3b = v_f32_ld_tnsr_b(wc, conv_w);
        if (has_bias) { int5 bc = {c0, 0, 0, 0, 0}; ba_ = v_f32_ld_tnsr_b(bc, conv_b); bc[0] = c0 + 64; bb_ = v_f32_ld_tnsr_b(bc, conv_b); }
        // conv ring: x[-3], x[-2], x[-1] from the state
        int5 sc = {c0, 0, slot_in, 0, 0};
        float64 x3a, x3b, x2a, x2b, x1a, x1b;
        sc[1] = 0; sc[0] = c0; x3a = v_f32_ld_tnsr_b(sc, cs_in); sc[0] = c0 + 64; x3b = v_f32_ld_tnsr_b(sc, cs_in);
        sc[1] = 1; sc[0] = c0; x2a = v_f32_ld_tnsr_b(sc, cs_in); sc[0] = c0 + 64; x2b = v_f32_ld_tnsr_b(sc, cs_in);
        sc[1] = 2; sc[0] = c0; x1a = v_f32_ld_tnsr_b(sc, cs_in); sc[0] = c0 + 64; x1b = v_f32_ld_tnsr_b(sc, cs_in);

        const int kind = (j < GDN_HK) ? 0 : (j < 2 * GDN_HK) ? 1 : 2;      // q, k, v block
        const int head = (kind == 0) ? j : (kind == 1) ? j - GDN_HK : j - 2 * GDN_HK;

#ifdef TREE
        // chain prefix (rows 0..NCH-1, each the child of the row before): the plain build's ring loop. Every other row (a leaf, or any row
        // past the prefix) takes its three taps from its own ancestors' INPUT rows (qkvz is read-only here, so no load-after-store), or
        // from cs_in before the block: no ring, no order dependence, independent iterations. The per-row schedule through ORD[] kept this
        // build's row loop at 1.6x the plain one (level-1 traces 2026-11-10: 16 chain rows 1.04 vs 0.65 ms/step).
        for (int t0 = 0; t0 < NCH; t0 += 16) { const int t1 = t0 + 16 < NCH ? t0 + 16 : NCH;   // 16 rows of SiLU, then their l2-norms
#pragma nounroll
            for (int t = t0; t < t1; ++t) {
                float128 xf; PREP_ROW(t, x3a, x3b, x2a, x2b, x1a, x1b, xf);
                x3a = x2a; x3b = x2b; x2a = x1a; x2b = x1b; x1a = xf.v1; x1b = xf.v2;
            }
#pragma nounroll
            for (int t = t0; t < t1 && kind != 2; ++t) PREP_NORM(t);
        }
        for (int u0 = NCH; u0 < T; u0 += 16) { const int u1 = u0 + 16 < T ? u0 + 16 : T;
#pragma nounroll
        for (int u = u0; u < u1; ++u) {
            const int r1 = PAR[u], r2 = r1 > 0 ? PAR[r1] : r1 - 1, r3 = r2 > 0 ? PAR[r2] : r2 - 1;   // ancestors; -1, -2, -3 = cs_in rows 2, 1, 0
            float64 t1a, t1b, t2a, t2b, t3a, t3b;
#define TAP(r_, A_, B_) if ((r_) >= 0) { int5 tc_ = {c0, (r_), 0, 0, 0}; float128 tf_ = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(tc_, qkvz)); A_ = tf_.v1; B_ = tf_.v2; } \
            else { int5 tc_ = {c0, 3 + (r_), slot_in, 0, 0}; A_ = v_f32_ld_tnsr_b(tc_, cs_in); tc_[0] = c0 + 64; B_ = v_f32_ld_tnsr_b(tc_, cs_in); }
            TAP(r1, t1a, t1b) TAP(r2, t2a, t2b) TAP(r3, t3a, t3b)
            float128 xf; PREP_ROW(u, t3a, t3b, t2a, t2b, t1a, t1b, xf);
        }
#pragma nounroll
        for (int u = u0; u < u1 && kind != 2; ++u) PREP_NORM(u);
        }
#else
        for (int t0 = 0; t0 < Te; t0 += 16) { const int t1 = t0 + 16 < Te ? t0 + 16 : Te;   // 16 rows of SiLU, then their l2-norms
#pragma nounroll
            for (int t = t0; t < t1; ++t) {
                float128 xf; PREP_ROW(t, x3a, x3b, x2a, x2b, x1a, x1b, xf);
                x3a = x2a; x3b = x2b; x2a = x1a; x2b = x1b; x1a = xf.v1; x1b = xf.v2;
            }
#pragma nounroll
            for (int t = t0; t < t1 && kind != 2; ++t) PREP_NORM(t);
        }
#endif
        if (no_out) continue;
        sc[2] = 0;
        sc[1] = 0; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x3a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x3b);
        sc[1] = 1; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x2a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x2b);
        sc[1] = 2; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x1a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x1b);
    }
}
