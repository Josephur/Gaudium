// gaudium_ga_lnorm: LayerNorm over the C channels of one row (a weight and a bias), C any multiple of 128: the three normalisations
// of the Gemma 4 unified image embedder (docs/notes/engine-gemma4.md, section 15: a row is a merged patch of 6912 pixel values, or
// its 3840-wide embedding).
//   m = mean_c x,  s = 1 / sqrt(mean_c (x - m)^2 + eps):   y = bf16( (x s - m s) g + b )
// in f32 in torch layer_norm's operation order, one rounding.
//  in0 x bf16 {C, P}   in1 g f32 {C}, in2 b f32 {C}: the weight and the bias in the lane order of converted bf16 blocks (within each
//      128 channels, f32 lane i < 64 is channel 2 i, lane 64 + i is channel 2 i + 1)   out0 y bf16 {C, P}.   index space {P}.
//  scalar: eps (f32 bits)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor x, tensor g, tensor b, tensor y, int eps_bits)
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
        float64 acc = 0.f;
        for (int k = 0; k < C; k += 128) { c[0] = k; const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x)); acc = acc + a.v1 + a.v2; }
        const float64 mean = REDUCE64(acc, P8, P4, P2, P1) * invc;
        float64 vacc = 0.f;
        for (int k = 0; k < C; k += 128) { c[0] = k; const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
            const float64 d1 = a.v1 - mean, d2 = a.v2 - mean; vacc = vacc + d1 * d1 + d2 * d2; }
        const float64 s = v_rsqrt_f32(REDUCE64(vacc, P8, P4, P2, P1) * invc + eps);
        const float64 bias = mean * s * -1.0f;
        for (int k = 0; k < C; k += 128) { c[0] = k; const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
            int5 w = {k, 0, 0, 0, 0};
            const float64 g1 = v_f32_ld_tnsr_b(w, g), b1 = v_f32_ld_tnsr_b(w, b); w[0] = k + 64;
            const float64 g2 = v_f32_ld_tnsr_b(w, g), b2 = v_f32_ld_tnsr_b(w, b);
            float128 oo; oo.v1 = (a.v1 * s + bias) * g1 + b1; oo.v2 = (a.v2 * s + bias) * g2 + b2;
            v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(oo)); }
    }
}
