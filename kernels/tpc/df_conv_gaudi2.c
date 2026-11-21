// gaudium_df_conv: DFlash2 two-tap dynamic grouped convolution over the T rows of one draft block (block = the T rows).
//   y[t, c] = (base[0][c] + delta[t, 0, g]) * x[t, c] + (base[1][c] + delta[t, 1, g]) * x[t-1, c] * [t >= 1],   g = c / 16
//  in0 x    bf16 {H, T}             in1 coef f32 {H/4, T}: feature f = side*(H/8) + tap*toff + g, toff = H/16 (kernel_projection output;
//                                   H = 5120: 1280 features, side 640, toff 320; H = 2560 (Qwen3.8-Flash-Next drafter): 640, 320, 160)
//  in2 base f32 {5120, 2}: base_kernel[side][tap] in the deinterleaved lane order per 128-channel block
//  out0 y   bf16 {H, T}.   scalars: T, foff (= side * H/8), toff (= H/16).   index space: H/128 channel blocks.
//  Lane order: the bf16->f32 "all" conversion puts even channels in v1 and odd channels in v2; lane l of either vector is
//  channel c0 + 2l (+1), group (c0 + 2l) / 16 = 8b + l/8: the 8 group deltas of a block are expanded to 8 lanes each.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REP_GROUP(a, gs) MOVDG_ALL((a), gs, gs, gs, gs, (a))

void main(tensor x, tensor coef, tensor base, tensor y, int T, int foff, int toff)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 lane = lid8 >> 2;                      // f32 lane 0..63
    const uchar256 l16 = lane & 15, grp = lane >> 4;      // lane in dual group, dual group 0..3
    // expansion pattern: lane (grp, l16) <- source lane 2*grp + l16/8 of its own dual group (after REP_GROUP(., 0))
    const uchar256 pat = (((grp << 1) + (l16 >> 3)) & 7) | 0x80;

    for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
        const int c0 = 128 * b;
        int5 bc = {c0, 0, 0, 0, 0};
        float64 b0a = v_f32_ld_tnsr_b(bc, base); bc[0] = c0 + 64; float64 b0b = v_f32_ld_tnsr_b(bc, base);
        bc[1] = 1; bc[0] = c0; float64 b1a = v_f32_ld_tnsr_b(bc, base); bc[0] = c0 + 64; float64 b1b = v_f32_ld_tnsr_b(bc, base);
        float64 pa = 0.f, pb = 0.f;   // previous row
        for (int t = 0; t < T; ++t) {
            int5 xc = {c0, t, 0, 0, 0};
            float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, x));
            int5 cc = {foff + 8 * b, t, 0, 0, 0};
            float64 d0 = v_f32_ld_tnsr_b(cc, coef); cc[0] = foff + toff + 8 * b; float64 d1 = v_f32_ld_tnsr_b(cc, coef);
            d0 = v_f32_shuffle_b(REP_GROUP(d0, 0), pat, 0, d0);
            d1 = v_f32_shuffle_b(REP_GROUP(d1, 0), pat, 0, d1);
            float128 o;
            o.v1 = (b0a + d0) * xf.v1; o.v2 = (b0b + d0) * xf.v2;
            if (t >= 1) { o.v1 = v_f32_mac_b(b1a + d1, pa, o.v1); o.v2 = v_f32_mac_b(b1b + d1, pb, o.v2); }
            v_bf16_st_tnsr(xc, y, v_convert_f32_to_bf16_all_b(o));
            pa = xf.v1; pb = xf.v2;
        }
    }
}
