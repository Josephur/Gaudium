// gaudium_cert_hq: activation split of the certified screening readout (notes/engine-mtp-certified.md).
//  in0  h    bf16 {HID, M}      the lm_head input rows (post-norm hidden)
//  out0 hq   bf16 {HID, 2M}     row m = hi_m * cs, row M+m = lo_m * 16 cs; every value lies on the e4m3 grid (|v| <= 240,
//                               no subnormals), so the vendor cast to fp8_143 with scale 1 is exact and the MME sees exactly
//                               hi = Q(h cs)/cs and lo = Q(16 (h cs - Q(h cs)))/(16 cs), Q = e4m3 nearest-even, subnormals -> 0
//  out1 coef f32  {64, 8, M}    every lane of vector (., k, m) holds coefficient k of row m (no vector->scalar moves needed):
//                               k=0 c1 = (||r|| + G (||h|| + ||hi|| + ||lo||)) (1 + 2^-8)  multiplies n_j = max(||W_j||, ||W8_j||)
//                               k=1 c2 = ||hq|| (1 + 2^-8)                                 multiplies e_j = ||W_j - W8_j||
//                               k=2..6: ||h||^2, ||r||^2, ||hq||^2, ||hi||^2, ||lo||^2 (diagnostics), r = h - hq, hq = hi + lo,
//                               G = 5120 * 2^-23 (fp32 accumulation of 5120 products, either rounding mode).
//  scalars: cs_bits (float bits of the pow2 activation scale), M.   index space: M points, one row each.
// The coefficients are upper bounds: the f32 sums of squares carry <= ~1e-5 relative error and the rsqrt-based roots far less
// than the 2^-8 margin. A non-finite h makes c1/c2 non-finite; gaudium_cert_mark treats that as "every lane is a candidate".
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
// e4m3 nearest-even on |x| <= 240 (clamped), results below 2^-6 flushed to 0: keep sign, exponent and 3 mantissa bits
#define QE4M3(x) ({ float64 _c = v_f32_min_b(v_f32_max_b((x), -240.0f), 240.0f); int64 _u = __builtin_astype(_c, int64); \
    int64 _l = v_i32_and_b(v_i32_shr_b(_u, 20), 1); int64 _r = v_i32_and_b(_u + 0x7FFFF + _l, (int)0xFFF00000); \
    float64 _y = __builtin_astype(_r, float64); v_f32_sel_less_f32_b(v_f32_abs_b(_y), 0.015625f, 0.0f, _y); })
#define SQRT_UP(s) ({ float64 _s = (s) + 1e-30f; _s * v_rsqrt_f32(_s); })

void main(tensor h, tensor hq, tensor coef, int cs_bits, int M)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float cs = __builtin_astype(cs_bits, float), ics = 1.0f / cs;   // cs is a power of two: ics exact
    const int HIDN = get_dim_size(h, 0);
    const float G = 6.103515625e-4f;   // 5120 * 2^-23
    for (int m = idx[0]; m < idx[0] + isz[0]; ++m) {
        float64 sh = 0.f, sr = 0.f, sq = 0.f, shi = 0.f, slo = 0.f;
        for (int c = 0; c < HIDN; c += 128) {
            int5 ci = {c, m, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ci, h));
            float128 ohi, olo;
            {   float64 v = x.v1, xs = v * cs, hi = QE4M3(xs), res = xs - hi, lo = QE4M3(res * 16.0f), rr = res - lo * 0.0625f, qq = hi + lo * 0.0625f;
                sh = sh + v * v; sr = sr + rr * rr; sq = sq + qq * qq; shi = shi + hi * hi; slo = slo + lo * lo; ohi.v1 = hi; olo.v1 = lo; }
            {   float64 v = x.v2, xs = v * cs, hi = QE4M3(xs), res = xs - hi, lo = QE4M3(res * 16.0f), rr = res - lo * 0.0625f, qq = hi + lo * 0.0625f;
                sh = sh + v * v; sr = sr + rr * rr; sq = sq + qq * qq; shi = shi + hi * hi; slo = slo + lo * lo; ohi.v2 = hi; olo.v2 = lo; }
            v_bf16_st_tnsr(ci, hq, v_convert_f32_to_bf16_all_b(ohi));
            int5 co = {c, M + m, 0, 0, 0};
            v_bf16_st_tnsr(co, hq, v_convert_f32_to_bf16_all_b(olo));
        }
        sh = REDUCE64(sh, P8, P4, P2, P1); sr = REDUCE64(sr, P8, P4, P2, P1); sq = REDUCE64(sq, P8, P4, P2, P1);
        shi = REDUCE64(shi, P8, P4, P2, P1); slo = REDUCE64(slo, P8, P4, P2, P1);
        // back to true units: r, hq, hi carry a factor cs; lo carries 16 cs
        const float i2 = ics * ics;
        float64 nh = SQRT_UP(sh), nr = SQRT_UP(sr * i2), nq = SQRT_UP(sq * i2), nhi = SQRT_UP(shi * i2), nlo = SQRT_UP(slo * (i2 * 0.00390625f));
        float64 c1 = (nr + (nh + nhi + nlo) * G) * 1.00390625f, c2 = nq * 1.00390625f;
        int5 o = {0, 0, m, 0, 0};
        v_f32_st_tnsr(o, coef, c1); o[1] = 1; v_f32_st_tnsr(o, coef, c2);
        o[1] = 2; v_f32_st_tnsr(o, coef, sh); o[1] = 3; v_f32_st_tnsr(o, coef, sr * i2); o[1] = 4; v_f32_st_tnsr(o, coef, sq * i2);
        o[1] = 5; v_f32_st_tnsr(o, coef, shi * i2); o[1] = 6; v_f32_st_tnsr(o, coef, slo * (i2 * 0.00390625f)); o[1] = 7; v_f32_st_tnsr(o, coef, nh);
    }
}
