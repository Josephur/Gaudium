#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_prep: depthwise causal conv (kernel 4) + SiLU over the mixed q|k|v projections, with conv state in/out,
// then per-head l2-norm of q and k (q also scaled by 1/sqrt(128)), GQA repeat 16 -> 48 heads, and re-layout into the
// f32 {128, 48, T} tensors the recurrence kernel reads. One index-space point = one 128-channel block (80 blocks:
// 16 q heads, 16 k heads, 48 v heads). z (channels 10240..16383 of qkvz) is left for gdn_post.
//
//  in0 qkvz   bf16 {16384, T}    in1 conv_w f32 {10240, 4} (tap 3 = current token)   in2 conv_state_in f32 {10240, 3}
//  in3 conv_b f32 {10240}        out0 q f32 {128,48,T}  out1 k f32 {128,48,T}  out2 v f32 {128,48,T}  out3 conv_state_out f32 {10240,3}
//  out4 cs_ckpt f32 {10240, 3, T}: the conv state after each row (for rollback to an accepted prefix)
//  scalar: T, flags (bit0: add conv bias, bit1: write cs_ckpt, bit4: skip the cs_out store,
//          bit5: chunked layout for the MME prefill path: q/k/v are {128, C, 48, NCH} with C = flags >> 16, row t at
//          {., t % C, head, t / C}; rows T..NCH*C-1 are written as zeros)
//  Built twice: plain, and -DSLOT with cs_in {10240, 3, NS} + int32 `slot` input (conv state read from slot slot[0]).
// Head counts are compile-time parameters (issue #7): GDN_HK key heads (q and k blocks), GDN_REP value heads per key head (the GQA
// repeat), so HV = GDN_HK * GDN_REP and the index space is 2 * GDN_HK + HV blocks. The defaults are Qwen3.8-27B's 16 / 3 (16 -> 48);
// kernels.mk builds one ELF per shape (suffix _k<HK>r<REP>), and the default build is byte-identical to the pre-#7 kernel.
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

#ifdef TREE
// -DTREE (implies SLOT): + parent int32 {T}: a row whose parent is not row t-1 (a tree leaf, parent = chain row p) is computed
// right after row p from the ring registers as they stand (x[p-2], x[p-1], x[p]), does not advance the ring, and writes its own
// checkpoint (x[p-1], x[p], x[leaf]). Nothing is read back from global memory written by this instance (vector load after
// vector store is not coherent on the TPC; the earlier reload from cs_ckpt[.., p] gave timing-dependent leaf rows).
__local__ int ORD[32];
__local__ int WBK[32];
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
    const int chunked = flags & 32, CH = chunked ? (flags >> 16) : 1;
    const int Tp = chunked ? get_dim_size(v, 1) * get_dim_size(v, 3) : T;
    const int5 isz = get_index_space_size();
    const int has_bias = flags & 1;
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, qscale = 0.08838834764831845f;   // 1/sqrt(128)

    for (int j = idx[0]; j < idx[0] + isz[0]; ++j) {
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
        int NSCH = 0;   // schedule: every chain row, each followed by its leaves (WBK = 0: the ring is not advanced)
        for (int t = 0; t < T; ++t) {
            int5 pc = {t, 0, 0, 0, 0};
            if (t > 0 && s_i32_ld_g(gen_addr(pc, parent)) != t - 1) continue;
            ORD[NSCH] = t; WBK[NSCH] = 1; ++NSCH;
            for (int u = t + 1; u < T; ++u) {
                int5 uc = {u, 0, 0, 0, 0}; const int pu = s_i32_ld_g(gen_addr(uc, parent));
                if (pu == t && pu != u - 1) { ORD[NSCH] = u; WBK[NSCH] = 0; ++NSCH; }
            }
        }
        for (int si = 0; si < NSCH; ++si) {
            const int t = ORD[si], leaf = !WBK[si];
#else
#pragma unroll 4
        for (int t = 0; t < Te; ++t) {
            int leaf = 0;
#endif
            int5 xc = {c0, t, 0, 0, 0};
            float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz));
            float64 ya = v_f32_mac_b(w3a, xf.v1, ba_), yb = v_f32_mac_b(w3b, xf.v2, bb_);
            ya = v_f32_mac_b(w2a, x1a, ya); yb = v_f32_mac_b(w2b, x1b, yb);
            ya = v_f32_mac_b(w1a, x2a, ya); yb = v_f32_mac_b(w1b, x2b, yb);
            ya = v_f32_mac_b(w0a, x3a, ya); yb = v_f32_mac_b(w0b, x3b, yb);
            ya = silu(ya); yb = silu(yb);
            const int tr = chunked ? t % CH : t, tc = chunked ? t / CH : 0;
            if (kind == 2) {
                int5 oc = {0, chunked ? tr : head, chunked ? head : t, tc, 0};
                v_f32_st_tnsr(oc, v, ya); oc[0] = 64; v_f32_st_tnsr(oc, v, yb);
            } else {
                float64 ss = REDUCE64(ya * ya + yb * yb, P8, P4, P2, P1);
                float64 r = v_rsqrt_f32(ss + eps);
                if (kind == 0) r = r * qscale;
                ya = ya * r; yb = yb * r;
                int5 oc = {0, GDN_REP * head, t, tc, 0};
                for (int rep = 0; rep < GDN_REP; ++rep) {
                    oc[1] = chunked ? tr : GDN_REP * head + rep; oc[2] = chunked ? GDN_REP * head + rep : t;
                    if (kind == 0) { oc[0] = 0; v_f32_st_tnsr(oc, q, ya); oc[0] = 64; v_f32_st_tnsr(oc, q, yb); }
                    else           { oc[0] = 0; v_f32_st_tnsr(oc, k, ya); oc[0] = 64; v_f32_st_tnsr(oc, k, yb); }
                }
            }
#ifdef TREE
            if (leaf) {   // checkpoint (x[p-1], x[p], x[leaf]); the ring stays at the parent
                if (flags & 2) {
                    int5 kc = {c0, 0, t, 0, 0};
                    kc[1] = 0; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, x2a); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, x2b);
                    kc[1] = 1; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, x1a); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, x1b);
                    kc[1] = 2; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, xf.v1); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, xf.v2);
                }
                continue;
            }
#endif
            x3a = x2a; x3b = x2b; x2a = x1a; x2b = x1b; x1a = xf.v1; x1b = xf.v2;
            (void)leaf;
            if (flags & 2) {   // per-row conv-state checkpoint
                int5 kc = {c0, 0, t, 0, 0};
                kc[1] = 0; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, x3a); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, x3b);
                kc[1] = 1; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, x2a); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, x2b);
                kc[1] = 2; kc[0] = c0; v_f32_st_tnsr(kc, cs_ckpt, x1a); kc[0] = c0 + 64; v_f32_st_tnsr(kc, cs_ckpt, x1b);
            }
        }
        for (int t = T; t < Tp; ++t) {        // chunked layout: zero the pad rows of the last chunk
            const float64 z = 0.f; const int tr = t % CH, tc = t / CH;
            if (kind == 2) { int5 oc = {0, tr, head, tc, 0}; v_f32_st_tnsr(oc, v, z); oc[0] = 64; v_f32_st_tnsr(oc, v, z); }
            else for (int rep = 0; rep < GDN_REP; ++rep) { int5 oc = {0, tr, GDN_REP * head + rep, tc, 0};
                if (kind == 0) { v_f32_st_tnsr(oc, q, z); oc[0] = 64; v_f32_st_tnsr(oc, q, z); } else { v_f32_st_tnsr(oc, k, z); oc[0] = 64; v_f32_st_tnsr(oc, k, z); } }
        }
        if (no_out) continue;
        sc[2] = 0;
        sc[1] = 0; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x3a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x3b);
        sc[1] = 1; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x2a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x2b);
        sc[1] = 2; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x1a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x1b);
    }
}
