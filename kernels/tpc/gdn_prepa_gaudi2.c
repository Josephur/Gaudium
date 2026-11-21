#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_prepa: phase A of the row-parallel prefill conv prep as its OWN node, so that one kernel uses one
// special-function table. gdn_prep2 alternates the f32 tanh table (SiLU's sigmoid, phase A) and the rsqrt table (l2norm, phase B) every
// 16 rows; the Gen2 LUT cache holds one of them (tpc-clang: "exceeds LUT cache capacity ... LUT misses"), and on the card the kernel
// takes 2x the simulator's cycles. Here: causal conv + SiLU for all 80 channel blocks (sigmoid table only); the 48 v blocks are stored
// in the chunked layout as before, the 32 q/k blocks go to y f32 {128, T, 32} for gaudium_gdn_prepb (l2norm + GQA repeat, rsqrt table
// only). Same arithmetic per element as gdn_prep2_gaudi2.c (y passes through HBM in f32 exactly as it passed through VLM): bit-identical.
//  in0 qkvz bf16 {16384, T}  in1 conv_w f32 {10240, 4}  in2 cs_in f32 {10240, 3, NS} (slot slot_in)  in3 conv_b f32 {10240}  in4 slot i32 {1}
//  out0 y f32 {128, T, 32}  out1 v f32 {128, C, 48, NCH}  out2 cs_out f32 {10240, 3}  out3 cs_ckpt f32 {10240, 3, 1} (unused here)
//  scalars: T, flags (bit0 bias, bit4 skip cs_out, C = flags >> 16), RG.  Index space {80 blocks, ceil(T/RG) row groups}.
static inline float64 silu(float64 x) { return x * v_sigmoid_f32(x); }

void main(tensor qkvz, tensor conv_w, tensor cs_in, tensor conv_b, tensor slot, tensor y, tensor v, tensor cs_out, tensor cs_ckpt, int T, int flags, int RG)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int5 zc = {0, 0, 0, 0, 0};
    const int slot_in = s_i32_ld_g(gen_addr(zc, slot));
    const int no_out = flags & 16, CH = flags >> 16;
    const int Tp = get_dim_size(v, 1) * get_dim_size(v, 3);
    const int has_bias = flags & 1;
    const int NRG = (T + RG - 1) / RG;

    for (int j = idx[0]; j < idx[0] + isz[0]; ++j) {
      const int c0 = 128 * j;
      const int kind = (j < GDN_HK) ? 0 : (j < 2 * GDN_HK) ? 1 : 2;      // q, k, v block
      const int head = (kind == 2) ? j - 2 * GDN_HK : j;              // v head, or the y block (q 0..15, k 16..31)
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
        } else {         // history = the previous three input rows
            int5 xc = {c0, t0 - 3, 0, 0, 0};
            float128 h3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz)); xc[1] = t0 - 2;
            float128 h2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz)); xc[1] = t0 - 1;
            float128 h1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz));
            x3a = h3.v1; x3b = h3.v2; x2a = h2.v1; x2b = h2.v2; x1a = h1.v1; x1b = h1.v2;
        }
        if (kind == 2) {
#pragma unroll 4
            for (int t = t0; t < t1; ++t) {
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
            for (int t = t0; t < t1; ++t) {
                int5 xc = {c0, t, 0, 0, 0};
                float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qkvz));
                float64 ya = v_f32_mac_b(w3a, xf.v1, ba_), yb = v_f32_mac_b(w3b, xf.v2, bb_);
                ya = v_f32_mac_b(w2a, x1a, ya); yb = v_f32_mac_b(w2b, x1b, yb);
                ya = v_f32_mac_b(w1a, x2a, ya); yb = v_f32_mac_b(w1b, x2b, yb);
                ya = v_f32_mac_b(w0a, x3a, ya); yb = v_f32_mac_b(w0b, x3b, yb);
                int5 oc = {0, t, head, 0, 0};
                v_f32_st_tnsr(oc, y, silu(ya)); oc[0] = 64; v_f32_st_tnsr(oc, y, silu(yb));
                x3a = x2a; x3b = x2b; x2a = x1a; x2b = x1b; x1a = xf.v1; x1b = xf.v2;
            }
        }
        if (rg != NRG - 1) continue;
        if (kind == 2) for (int t = T; t < Tp; ++t) {        // zero the pad rows of the last chunk (v; q/k pads are gdn_prepb's)
            const float64 z = 0.f; int5 oc = {0, t % CH, head, t / CH, 0}; v_f32_st_tnsr(oc, v, z); oc[0] = 64; v_f32_st_tnsr(oc, v, z); }
        if (no_out) continue;
        int5 sc = {c0, 0, 0, 0, 0};
        sc[1] = 0; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x3a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x3b);
        sc[1] = 1; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x2a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x2b);
        sc[1] = 2; sc[0] = c0; v_f32_st_tnsr(sc, cs_out, x1a); sc[0] = c0 + 64; v_f32_st_tnsr(sc, cs_out, x1b);
      }
    }
}
