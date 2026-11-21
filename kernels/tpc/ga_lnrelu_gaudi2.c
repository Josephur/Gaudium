// gaudium_ga_lnrelu: LayerNorm over the channels of one position (a weight, no bias), then ReLU: the normalisation of the two
// subsampling stages of the Gemma 4 audio encoder (docs/notes/engine-gemma4.md, section 14).
//   m = mean_c x,  s = 1 / sqrt(mean_c (x - m)^2 + eps):   y = max(0, bf16( (x s - m s) g ))
// in f32 in torch layer_norm's operation order, one rounding. A position whose channels are all zero stays zero (the padding row).
//  in0 x bf16 {C, P}   in1 g f32 {128}: the weight in the lane order of a converted bf16 block (f32 lane i < 64 is channel 2 i, lane
//      64 + i is channel 2 i + 1), zero beyond C   out0 y bf16 {C, P}.   C <= 128 and even.   index space {P}.   scalar: eps (f32 bits)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor x, tensor g, tensor y, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    const int C = get_dim_size(x, 0), HC = C / 2;   // the channels of a converted block: lanes 0 .. HC - 1 of each half
    const float eps = *(float*)&eps_bits, invc = 1.0f / (float)C;
    int5 z = {0, 0, 0, 0, 0};
    const float64 g1 = v_f32_ld_tnsr_b(z, g); z[0] = 64;
    const float64 g2 = v_f32_ld_tnsr_b(z, g);
    const float64 zero = 0.f;
    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        int5 c = {0, p, 0, 0, 0};
        const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
        const float64 x1 = v_f32_sel_less_i32_b(lane, HC, a.v1, zero), x2 = v_f32_sel_less_i32_b(lane, HC, a.v2, zero);
        const float64 mean = REDUCE64(x1 + x2, P8, P4, P2, P1) * invc;
        const float64 d1 = v_f32_sel_less_i32_b(lane, HC, x1 - mean, zero), d2 = v_f32_sel_less_i32_b(lane, HC, x2 - mean, zero);
        const float64 var = REDUCE64(d1 * d1 + d2 * d2, P8, P4, P2, P1) * invc;
        const float64 s = v_rsqrt_f32(var + eps);
        const float64 bias = mean * s * -1.0f;
        float128 oo;
        oo.v1 = v_f32_max_b((x1 * s + bias) * g1, zero); oo.v2 = v_f32_max_b((x2 * s + bias) * g2, zero);
        v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(oo));
    }
}
