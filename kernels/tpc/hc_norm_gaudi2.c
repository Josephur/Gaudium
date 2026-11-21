// gaudium_hc_norm: RMSNorm per hyper-connection stream (qwen4_exp / Qwen3.8-Flash-Next; HF Qwen4ExpTextRMSNorm with group_size = HID).
//   in0 x bf16 {S*HID, T}   in1 w bf16 {HID, S} (zero-centred: gamma = 1 + w, one gamma per stream)   out0 y bf16 {S*HID, T}
//   scalar: HID (multiple of 256)
//   y[s*HID + c, t] = bf16( x * rsqrt(mean_c(x^2) + 1e-6) * (1 + w[c, s]) ), f32 inside, one rounding (HF: norm(x.float()) * (1 + w.float()))
//   index space {S, T}: one point = one stream of one row. Used for the HC mixers (hc_attn / hc_ffn / the final mixer, the MTP mixers),
//   the MTP hnorm and the three PLE norms. x is read twice from HBM (an input, never a just-stored output: add_norm_gaudi2.c v2 lesson).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor x, tensor w, tensor y, int HID)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int NB = HID / 256;
    const float eps = 1e-6f, inv_n = 1.f / (float)HID;
    const float64 one = 1.f;
    for (int s = idx[0]; s < idx[0] + isz[0]; ++s)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        const int base = s * HID;
        float64 sq0 = 0.f, sq1 = 0.f, sq2 = 0.f, sq3 = 0.f;
        for (int b = 0; b < NB; ++b) {
            int5 xc = {base + b * 256, t, 0, 0, 0};
            bfloat128 xb0 = v_bf16_ld_tnsr_b(xc, x); xc[0] += 128; bfloat128 xb1 = v_bf16_ld_tnsr_b(xc, x);
            float128 f0 = v_convert_bf16_to_f32_all_b(xb0), f1 = v_convert_bf16_to_f32_all_b(xb1);
            sq0 = sq0 + f0.v1 * f0.v1; sq1 = sq1 + f0.v2 * f0.v2; sq2 = sq2 + f1.v1 * f1.v1; sq3 = sq3 + f1.v2 * f1.v2;
        }
        const float64 ss = REDUCE64((sq0 + sq1) + (sq2 + sq3), P8, P4, P2, P1);
        const float64 inv = v_rsqrt_f32(ss * inv_n + eps);
        for (int b = 0; b < NB; ++b) {
            int5 xc = {base + b * 256, t, 0, 0, 0};
            bfloat128 xb0 = v_bf16_ld_tnsr_b(xc, x); xc[0] += 128; bfloat128 xb1 = v_bf16_ld_tnsr_b(xc, x);
            int5 wc = {b * 256, s, 0, 0, 0};
            bfloat128 wb0 = v_bf16_ld_tnsr_b(wc, w); wc[0] += 128; bfloat128 wb1 = v_bf16_ld_tnsr_b(wc, w);
            float128 f0 = v_convert_bf16_to_f32_all_b(xb0), f1 = v_convert_bf16_to_f32_all_b(xb1);
            float128 g0 = v_convert_bf16_to_f32_all_b(wb0), g1 = v_convert_bf16_to_f32_all_b(wb1);
            float128 o0; o0.v1 = (f0.v1 * inv) * (g0.v1 + one); o0.v2 = (f0.v2 * inv) * (g0.v2 + one);
            float128 o1; o1.v1 = (f1.v1 * inv) * (g1.v1 + one); o1.v2 = (f1.v2 * inv) * (g1.v2 + one);
            int5 oc = {base + b * 256, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o1));
        }
    }
}
