#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_prep, row-parallel prefill build (selected by flags bit6 in the MME prefill recipes, chunked
// layout only). Same arithmetic per element as gdn_prep_gaudi2.c (results bit-identical), different schedule:
//  - index space {80 channel blocks, NRG row groups of RG rows} instead of {80}: a group's conv history is rows t0-3..t0-1 of
//    qkvz (t0 > 0) or the conv state (t0 = 0), exactly the values the serial ring holds there;
//  - rows of a group in batches of GB = 16: phase A conv + SiLU of the batch (sigmoid = tanh LUT), phase B the l2norm of the
//    q/k rows (rsqrt LUT). The Gen2 LUT cache cannot hold the f32 tanh and rsqrt tables together (tpc_llvm
//    TPCLutCacheCounter: "exceeds LUT cache capacity ... LUT misses"; tpc-clang warns on gdn_prep_gaudi2.c): the serial kernel
//    switched tables twice per row, this one twice per 16 rows. y of a batch waits in VLM (8 KB).
//  in/out as gdn_prep_gaudi2.c (flags bit5 required: q/k/v {128, C, 48, NCH}, C = flags >> 16; bit1 per-row checkpoints not
//  supported). scalars: T, flags, RG. Built plain and -DSLOT (cs_in {10240, 3, NS} + slot input).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
static inline float64 silu(float64 x) { return x * v_sigmoid_f32(x); }
#ifndef GB
#define GB 16   // rows per LUT phase (feature/prefill-tpc: -DGB=32/64 variants trade VLM for fewer table switches)
#endif
__local__ float64 YBUF[2 * GB];

#ifdef SLOT
void main(tensor qkvz, tensor conv_w, tensor cs_in, tensor conv_b, tensor slot, tensor q, tensor k, tensor v, tensor cs_out, tensor cs_ckpt, int T, int flags, int RG)
#else
void main(tensor qkvz, tensor conv_w, tensor cs_in, tensor conv_b, tensor q, tensor k, tensor v, tensor cs_out, tensor cs_ckpt, int T, int flags, int RG)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
#ifdef SLOT
    const int5 zc = {0, 0, 0, 0, 0};
    const int slot_in = s_i32_ld_g(gen_addr(zc, slot));
#else
    const int slot_in = 0;
#endif
    const int no_out = flags & 16, CH = flags >> 16;
    const int Tp = get_dim_size(v, 1) * get_dim_size(v, 3);
    const int has_bias = flags & 1;
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, qscale = 0.08838834764831845f;   // 1/sqrt(128)
    const int NRG = (T + RG - 1) / RG;

    for (int j = idx[0]; j < idx[0] + isz[0]; ++j) {
      const int c0 = 128 * j;
      const int kind = (j < GDN_HK) ? 0 : (j < 2 * GDN_HK) ? 1 : 2;      // q, k, v block
      const int head = (kind == 0) ? j : (kind == 1) ? j - GDN_HK : j - 2 * GDN_HK;
      int5 wc = {c0, 0, 0, 0, 0};
      float64 w0a, w0b, w1a, w1b, w2a, w2b, w3a, w3b, ba_ = 0.f, bb_ = 0.f;
      wc[1] = 0; wc[0] = c0; w0a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w0b = v_f32_ld_tnsr_b(wc, conv_w);
      wc[1] = 1; wc[0] = c0; w1a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w1b = v_f32_ld_tnsr_b(wc, conv_w);
      wc[1] = 2; wc[0] = c0; w2a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w2b = v_f32_ld_tnsr_b(wc, conv_w);
      wc[1] = 3; wc[0] = c0; w3a = v_f32_ld_tnsr_b(wc, conv_w); wc[0] = c0 + 64; w3b = v_f32_ld_tnsr_b(wc, conv_w);
      if (has_bias) { int5 bc = {c0, 0, 0, 0, 0}; ba_ = v_f32_ld_tnsr_b(bc, conv_b); bc[0] = c0 + 64; bb_ = v_f32_ld_tnsr_b(bc, conv_b); }
      for (int rg = idx[1]; rg < idx[1] + isz[1]; ++rg) {
        const int t0 = rg * RG, t1 = (t0 + RG < T) ? t0 + RG : T;
        float64 x3a, x3b, x2a, x2b, x1a, x1b;
        if (t0 == 0) {   // history from the conv state
            int5 sc = {c0, 0, slot_in, 0, 0};
            sc[1] = 0; sc[0] = c0; x3a = v_f32_ld_tnsr_b(sc, cs_in); sc[0] = c0 + 64; x3b = v_f32_ld_tnsr_b(sc, cs_in);
            sc[1] = 1; sc[0] = c0; x2a = v_f32_ld_tnsr_b(sc, cs_in); sc[0] = c0 + 64; x2b = v_f32_ld_tnsr_b(sc, cs_in);
            sc[1] = 2; sc[0] = c0; x1a = v_f32_ld_tnsr_b(sc, cs_in); sc[0] = c0 + 64; x1b = v_f32_ld_tnsr_b(sc, cs_in);
        } else {         // history = the previous three input rows (what the serial ring holds after row t0-1)
            int5 xc = {c0, t0 - 3, 0, 0, 0};
            float128 h3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz)); xc[1] = t0 - 2;
            float128 h2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz)); xc[1] = t0 - 1;
            float128 h1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz));
            x3a = h3.v1; x3b = h3.v2; x2a = h2.v1; x2b = h2.v2; x1a = h1.v1; x1b = h1.v2;
        }
        for (int tb = t0; tb < t1; tb += GB) {
            const int n = (tb + GB < t1) ? GB : t1 - tb;
            if (kind == 2) {
#pragma unroll 4
                for (int i = 0; i < n; ++i) {
                    const int t = tb + i;
                    int5 xc = {c0, t, 0, 0, 0};
                    float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz));
                    float64 ya = v_f32_mac_b(w3a, xf.v1, ba_), yb = v_f32_mac_b(w3b, xf.v2, bb_);
                    ya = v_f32_mac_b(w2a, x1a, ya); yb = v_f32_mac_b(w2b, x1b, yb);
                    ya = v_f32_mac_b(w1a, x2a, ya); yb = v_f32_mac_b(w1b, x2b, yb);
                    ya = v_f32_mac_b(w0a, x3a, ya); yb = v_f32_mac_b(w0b, x3b, yb);
                    ya = silu(ya); yb = silu(yb);
                    int5 oc = {0, t % CH, head, t / CH, 0};
                    v_f32_st_tnsr(oc, v, ya); oc[0] = 64; v_f32_st_tnsr(oc, v, yb);
                    x3a = x2a; x3b = x2b; x2a = x1a; x2b = x1b; x1a = xf.v1; x1b = xf.v2;
                }
            } else {
#pragma unroll 4
                for (int i = 0; i < n; ++i) {   // phase A: conv + SiLU (tanh table)
                    const int t = tb + i;
                    int5 xc = {c0, t, 0, 0, 0};
                    float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz));
                    float64 ya = v_f32_mac_b(w3a, xf.v1, ba_), yb = v_f32_mac_b(w3b, xf.v2, bb_);
                    ya = v_f32_mac_b(w2a, x1a, ya); yb = v_f32_mac_b(w2b, x1b, yb);
                    ya = v_f32_mac_b(w1a, x2a, ya); yb = v_f32_mac_b(w1b, x2b, yb);
                    ya = v_f32_mac_b(w0a, x3a, ya); yb = v_f32_mac_b(w0b, x3b, yb);
                    YBUF[2 * i] = silu(ya); YBUF[2 * i + 1] = silu(yb);
                    x3a = x2a; x3b = x2b; x2a = x1a; x2b = x1b; x1a = xf.v1; x1b = xf.v2;
                }
#pragma unroll 4
                for (int i = 0; i < n; ++i) {   // phase B: l2norm (rsqrt table), GQA repeat x3
                    const int t = tb + i;
                    float64 ya = YBUF[2 * i], yb = YBUF[2 * i + 1];
                    float64 ss = REDUCE64(ya * ya + yb * yb, P8, P4, P2, P1);
                    float64 r = v_rsqrt_f32(ss + eps);
                    if (kind == 0) r = r * qscale;
                    ya = ya * r; yb = yb * r;
                    int5 oc = {0, t % CH, GDN_REP * head, t / CH, 0};
                    for (int rep = 0; rep < GDN_REP; ++rep) {
                        oc[2] = GDN_REP * head + rep;
                        if (kind == 0) { oc[0] = 0; v_f32_st_tnsr(oc, q, ya); oc[0] = 64; v_f32_st_tnsr(oc, q, yb); }
                        else           { oc[0] = 0; v_f32_st_tnsr(oc, k, ya); oc[0] = 64; v_f32_st_tnsr(oc, k, yb); }
                    }
                }
            }
        }
        if (rg != NRG - 1) continue;
        for (int t = T; t < Tp; ++t) {        // zero the pad rows of the last chunk
            const float64 z = 0.f; const int tr = t % CH, tc = t / CH;
            if (kind == 2) { int5 oc = {0, tr, head, tc, 0}; v_f32_st_tnsr(oc, v, z); oc[0] = 64; v_f32_st_tnsr(oc, v, z); }
            else for (int rep = 0; rep < GDN_REP; ++rep) { int5 oc = {0, tr, GDN_REP * head + rep, tc, 0};
                if (kind == 0) { v_f32_st_tnsr(oc, q, z); oc[0] = 64; v_f32_st_tnsr(oc, q, z); } else { v_f32_st_tnsr(oc, k, z); oc[0] = 64; v_f32_st_tnsr(oc, k, z); } }
        }
        if (no_out) continue;
        int5 sc = {c0, 0, 0, 0, 0};
        sc[1] = 0; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x3a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x3b);
        sc[1] = 1; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x2a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x2b);
        sc[1] = 2; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x1a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x1b);
      }
    }
}
