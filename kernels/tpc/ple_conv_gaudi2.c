// gaudium_ple_conv: the n-gram embedding's dilated causal depthwise conv and the residual update (qwen4_exp;
// HF Qwen4ExpTextPLELayer._short_conv: Conv1d(groups = channels, kernel 4, dilation 3), state of (4 - 1) * 3 = 9 positions).
//   in0 cn    bf16 {RW, T}       norm_conv(gated_value): the conv input           (RW = S * HID channels)
//   in1 gv    bf16 {RW, T}       gated_value
//   in2 R     bf16 {RW, T}       the residual before the PLE layer
//   in3 ps_in f32  {RW, 9, NS}   conv history (the 9 previous cn rows, oldest first) of checkpoint slot slot[0]; DEINTERLEAVED lanes:
//                                f32 index 128 b + j holds channel 128 b + 2 j (j < 64) or 128 b + 2 (j - 64) + 1 (j >= 64), the order
//                                of the bf16 -> f32 "all" conversion, so a 64-lane load pairs with v1 / v2 of the bf16 vectors
//   in4 w     f32  {RW, 4}       the conv taps per channel (same deinterleaved order): out[t] = sum_k w[k] x[t - (3 - k) * 3]
//   in5 slot  i32  {1}
//   out0 Ro   bf16 {RW, T}       R + bf16(gv + bf16(silu(bf16(conv))))
//   out1 ps_ck f32 {RW, 9, T|1}  flags bit0: the history after each row t (row t's checkpoint, chosen by the next step's slot)
//   out2 ps_out f32 {RW, 9}      the history after the last row (prefill tiles chain through it)
//   scalar: T, flags
//   index space {RW / 128}: one point = 128 channels, all rows in order (the conv is sequential in t). Zero history = sequence start.
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })

void main(tensor cn, tensor gv, tensor R, tensor ps_in, tensor w, tensor slot, tensor Ro, tensor ps_ck, tensor ps_out, int T, int flags)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int sl = s_i32_ld_g(gen_addr(zc, slot));
    const bool ck = (flags & 1) != 0;
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
        const int c0 = b * 128;
        // history: h0 oldest .. h8 newest, a = even channels (v1), b = odd (v2); named registers (an array of vectors shifted in a loop
        // was not shifted by tpc-clang 2.7.33: sim_q4x, 2026-11-13)
#define HLD(j, A, B) { int5 hc = {c0, j, sl, 0, 0}; A = v_f32_ld_tnsr_b(hc, ps_in); hc[0] = c0 + 64; B = v_f32_ld_tnsr_b(hc, ps_in); }
#define HST(tn, j, A, B, c2) { int5 sc = {c0, j, c2, 0, 0}; v_f32_st_tnsr(sc, tn, A); sc[0] = c0 + 64; v_f32_st_tnsr(sc, tn, B); }
#define HALL(M, tn, c2) M(tn, 0, a0, b0, c2) M(tn, 1, a1, b1, c2) M(tn, 2, a2, b2, c2) M(tn, 3, a3, b3, c2) M(tn, 4, a4, b4, c2) \
                        M(tn, 5, a5, b5, c2) M(tn, 6, a6, b6, c2) M(tn, 7, a7, b7, c2) M(tn, 8, a8, b8, c2)
        float64 a0, a1, a2, a3, a4, a5, a6, a7, a8, b0, b1, b2, b3, b4, b5, b6, b7, b8;
        HLD(0, a0, b0) HLD(1, a1, b1) HLD(2, a2, b2) HLD(3, a3, b3) HLD(4, a4, b4) HLD(5, a5, b5) HLD(6, a6, b6) HLD(7, a7, b7) HLD(8, a8, b8)
        int5 wc = {c0, 0, 0, 0, 0};
        wc[1] = 0; const float64 wa0 = v_f32_ld_tnsr_b(wc, w); wc[0] = c0 + 64; const float64 wb0 = v_f32_ld_tnsr_b(wc, w);
        wc[1] = 1; wc[0] = c0; const float64 wa1 = v_f32_ld_tnsr_b(wc, w); wc[0] = c0 + 64; const float64 wb1 = v_f32_ld_tnsr_b(wc, w);
        wc[1] = 2; wc[0] = c0; const float64 wa2 = v_f32_ld_tnsr_b(wc, w); wc[0] = c0 + 64; const float64 wb2 = v_f32_ld_tnsr_b(wc, w);
        wc[1] = 3; wc[0] = c0; const float64 wa3 = v_f32_ld_tnsr_b(wc, w); wc[0] = c0 + 64; const float64 wb3 = v_f32_ld_tnsr_b(wc, w);
        for (int t = 0; t < T; ++t) {
            int5 xc = {c0, t, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, cn));
            float128 g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, gv));
            float128 r = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, R));
            // taps: t-9 = h0, t-6 = h3, t-3 = h6, t = x
            float64 ca = wa0 * a0; ca = ca + wa1 * a3; ca = ca + wa2 * a6; ca = ca + wa3 * x.v1;
            float64 cb = wb0 * b0; cb = cb + wb1 * b3; cb = cb + wb2 * b6; cb = cb + wb3 * x.v2;
            float128 cv = RBF(ca, cb);
            float128 sv = RBF(cv.v1 * v_sigmoid_f32(cv.v1), cv.v2 * v_sigmoid_f32(cv.v2));   // silu, bf16
            float128 ov = RBF(g.v1 + sv.v1, g.v2 + sv.v2);
            float128 o; o.v1 = r.v1 + ov.v1; o.v2 = r.v2 + ov.v2;
            v_bf16_st_tnsr(xc, Ro, v_convert_f32_to_bf16_all_b(o));
            a0 = a1; a1 = a2; a2 = a3; a3 = a4; a4 = a5; a5 = a6; a6 = a7; a7 = a8; a8 = x.v1;
            b0 = b1; b1 = b2; b2 = b3; b3 = b4; b4 = b5; b5 = b6; b6 = b7; b7 = b8; b8 = x.v2;
            if (ck) { HALL(HST, ps_ck, t) }
        }
        HALL(HST, ps_out, 0)
    }
}
