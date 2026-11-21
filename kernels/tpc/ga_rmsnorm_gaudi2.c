// gaudium_ga_rmsnorm: Gemma4RMSNorm with a weight over the C channels of one row, C any multiple of 128:
//   y = bf16( x (mean_c x^2 + eps)^-1/2 w )
// in f32 with one rounding: the reference's `_norm(x.float()) * weight.float()`, cast back once.
// One index-space member is one row, and a row's sum of squares is accumulated in one fixed order. So a row has the same values in
// a pass of one row, in a pass of several rows and in a prefill tile of any size. The vendor rms_norm_fwd_bf16 does not have that
// property: on the card its 1 / rms of one input row differed by one or two f32 steps between a pass of one row and a pass of five
// (docs/notes/engine-gemma4.md, section 16), a few outputs then round to the next bf16 value, and behind an FP8 activation cast
// that is a whole FP8 step: the K = 0 stream and the drafter's streams parted.
//  in0 x bf16 {C, P}   in1 w bf16 {C} (the checkpoint's order)   out0 y bf16 {C, P}.   index space {P}.
//  scalar: eps (f32 bits)
//  The weight is converted like the row (v_convert_bf16_to_f32_all_b puts the even channels of a block of 128 in v1 and the odd ones
//  in v2), so its lanes meet the row's by construction.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor x, tensor w, tensor y, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int C = get_dim_size(x, 0);
    const float eps = *(float*)&eps_bits, invc = 1.0f / (float)C;
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        int5 c = {0, p, 0, 0, 0};
        // the sum of squares: two accumulators (the even and the odd channels of each block), block after block from channel 0
        // (four blocks' loads in flight per iteration: the kernel is bound by the load latency. The order of the additions is the
        // plain loop's, block after block, so the unrolled and the tail iterations give one result.)
        float64 a1 = 0.f, a2 = 0.f; int k = 0;
        for (; k + 512 <= C; k += 512) {
            c[0] = k; const bfloat128 x0 = v_bf16_ld_tnsr_b(c, x); c[0] = k + 128; const bfloat128 x1 = v_bf16_ld_tnsr_b(c, x);
            c[0] = k + 256; const bfloat128 x2 = v_bf16_ld_tnsr_b(c, x); c[0] = k + 384; const bfloat128 x3 = v_bf16_ld_tnsr_b(c, x);
            const float128 f0 = v_convert_bf16_to_f32_all_b(x0), f1 = v_convert_bf16_to_f32_all_b(x1), f2 = v_convert_bf16_to_f32_all_b(x2), f3 = v_convert_bf16_to_f32_all_b(x3);
            a1 = a1 + f0.v1 * f0.v1; a2 = a2 + f0.v2 * f0.v2; a1 = a1 + f1.v1 * f1.v1; a2 = a2 + f1.v2 * f1.v2;
            a1 = a1 + f2.v1 * f2.v1; a2 = a2 + f2.v2 * f2.v2; a1 = a1 + f3.v1 * f3.v1; a2 = a2 + f3.v2 * f3.v2; }
        for (; k < C; k += 128) { c[0] = k; const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x)); a1 = a1 + a.v1 * a.v1; a2 = a2 + a.v2 * a.v2; }
        const float64 s = v_rsqrt_f32(REDUCE64(a1 + a2, P8, P4, P2, P1) * invc + eps);
        int5 wc = {0, 0, 0, 0, 0};
        for (k = 0; k + 256 <= C; k += 256) {
            c[0] = k; wc[0] = k; const bfloat128 x0 = v_bf16_ld_tnsr_b(c, x), w0 = v_bf16_ld_tnsr_b(wc, w);
            c[0] = k + 128; wc[0] = k + 128; const bfloat128 x1 = v_bf16_ld_tnsr_b(c, x), w1 = v_bf16_ld_tnsr_b(wc, w);
            const float128 f0 = v_convert_bf16_to_f32_all_b(x0), g0 = v_convert_bf16_to_f32_all_b(w0), f1 = v_convert_bf16_to_f32_all_b(x1), g1 = v_convert_bf16_to_f32_all_b(w1);
            float128 o0, o1; o0.v1 = f0.v1 * s * g0.v1; o0.v2 = f0.v2 * s * g0.v2; o1.v1 = f1.v1 * s * g1.v1; o1.v2 = f1.v2 * s * g1.v2;
            c[0] = k; v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(o0)); c[0] = k + 128; v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(o1)); }
        for (; k < C; k += 128) { c[0] = k; wc[0] = k; const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x)), g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
            float128 oo; oo.v1 = a.v1 * s * g.v1; oo.v2 = a.v2 * s * g.v2;
            v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(oo)); }
    }
}
