// gaudium_fused_add_norm (v2): residual-add + full-row RMSNorm in ONE node.
//   sum[t][c] = bf16_rne(x[t][c] + res[t][c])                 (the running residual, == add_fwd_bf16)
//   y[t][c]   = bf16_rne(sum_r[t][c] * rsqrt(mean(sum_r^2) + eps) * (1 + w[c]))
// where sum_r is the bf16-ROUNDED sum re-upcast to f32 — exactly what HF and the vendor
// rms_norm_fwd_bf16(ParamsRmsNormV3{eps 1e-6, addOneToWeight=true}) normalise on {HID, T}.
//
// v1 lessons applied (review 7.2): reduction over the FULL HID row (not 256-channel
// groups); 1 + w in f32 (zero-centred weights); normalise the bf16-rounded sum (token
// identity with the engine); natural layouts throughout.
//
//   in0 x   bf16 {HID, T}    in1 res bf16 {HID, T}    in2 w bf16 {HID} (HF order)
//   out0 sum bf16 {HID, T}   out1 y bf16 {HID, T}     scalar: T
//   HID must be a multiple of 256 (5120 = 20 blocks). One index-space point per row.
//   The y pass consumes the rounded sums from VLM (__local__ bf16 pairs, 10 KB — fits
//   the SF-limited 16 KB static VLM). v2 lesson: reloading the just-stored `sum` from
//   HBM reads STALE ZEROS on card (TPC write-queue latency; tpcsim makes stores
//   immediately visible and hid this — card probe p16: sum bit-exact, y all-zero).
//   Weight alignment: convert_bf16_to_f32_all_b deinterleaves (v1 = even channels,
//   v2 = odd), the same idiom the sums use, so w lanes align by construction.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
// -DPLAIN_W build (plain Qwen3 norms, Qwen3RMSNorm: weight * bf16(x * rsqrt(mean(x^2) + eps))): the normalised row is rounded to bf16
// first, then multiplied by w (bf16) and rounded again, the reference's two roundings; the default build is the zero-centred (1 + w) form.
static inline float64 round_bf16(float64 x) { float128 t; t.v1 = x; t.v2 = x; return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(t)).v1; }
#ifdef PLAIN_W
#define WN(s_, g_) (round_bf16((s_) * inv) * (g_))
#else
#define WN(s_, g_) ((s_) * inv * ((g_) + one))
#endif

// rounded sums, VLM-resident (12 KB static; rows run sequentially per TPC instance).
// NB <= 24 => HID <= 6144. Fits the SF-limited 16 KB static VLM (rsqrt is a special function).
__local__ bfloat128 sv0[24], sv1[24];

void main(tensor x, tensor res, tensor w, tensor sum, tensor y, int T)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int NB = get_dim_size(x, 0) / 256;          // 256-channel blocks per row (5120 -> 20)
    const float eps = 1e-6f, inv_n = 1.f / ((float)NB * 256.f);

    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        // pass 1: sums (bf16 RNE, stored) + per-lane partial sum of squares of the rounded sums.
        // 2x unrolled: 8 loads in flight (the load-latency lever, g2-opt-manual 7.6; the kernel is
        // load-latency-bound at ~1.5k cycles/row vs ~600 cycles of VPU work).
        float64 sq0 = 0.f, sq1 = 0.f, sq2 = 0.f, sq3 = 0.f;
        int b = 0;
        for (; b + 2 <= NB; b += 2) {
            int5 xc = {b * 256, t, 0, 0, 0};
            bfloat128 xb00 = v_bf16_ld_tnsr_b(xc, x);  xc[0] += 128; bfloat128 xb01 = v_bf16_ld_tnsr_b(xc, x);
            xc[0] += 128; bfloat128 xb10 = v_bf16_ld_tnsr_b(xc, x);  xc[0] += 128; bfloat128 xb11 = v_bf16_ld_tnsr_b(xc, x);
            int5 rc = {b * 256, t, 0, 0, 0};
            bfloat128 rb00 = v_bf16_ld_tnsr_b(rc, res); rc[0] += 128; bfloat128 rb01 = v_bf16_ld_tnsr_b(rc, res);
            rc[0] += 128; bfloat128 rb10 = v_bf16_ld_tnsr_b(rc, res); rc[0] += 128; bfloat128 rb11 = v_bf16_ld_tnsr_b(rc, res);
            float128 f00 = v_convert_bf16_to_f32_all_b(xb00), f01 = v_convert_bf16_to_f32_all_b(xb01);
            float128 f10 = v_convert_bf16_to_f32_all_b(xb10), f11 = v_convert_bf16_to_f32_all_b(xb11);
            float128 r00 = v_convert_bf16_to_f32_all_b(rb00), r01 = v_convert_bf16_to_f32_all_b(rb01);
            float128 r10 = v_convert_bf16_to_f32_all_b(rb10), r11 = v_convert_bf16_to_f32_all_b(rb11);
            float64 a00 = f00.v1 + r00.v1, a01 = f00.v2 + r00.v2, a02 = f01.v1 + r01.v1, a03 = f01.v2 + r01.v2;
            float64 a10 = f10.v1 + r10.v1, a11 = f10.v2 + r10.v2, a12 = f11.v1 + r11.v1, a13 = f11.v2 + r11.v2;
            float128 p00; p00.v1 = a00; p00.v2 = a01;
            bfloat128 pb00 = v_convert_f32_to_bf16_all_b(p00);
            float128 q00 = v_convert_bf16_to_f32_all_b(pb00);
            float128 p01; p01.v1 = a02; p01.v2 = a03;
            bfloat128 pb01 = v_convert_f32_to_bf16_all_b(p01);
            float128 q01 = v_convert_bf16_to_f32_all_b(pb01);
            float128 p10; p10.v1 = a10; p10.v2 = a11;
            bfloat128 pb10 = v_convert_f32_to_bf16_all_b(p10);
            float128 q10 = v_convert_bf16_to_f32_all_b(pb10);
            float128 p11; p11.v1 = a12; p11.v2 = a13;
            bfloat128 pb11 = v_convert_f32_to_bf16_all_b(p11);
            float128 q11 = v_convert_bf16_to_f32_all_b(pb11);
            sv0[b] = pb00; sv1[b] = pb01; sv0[b + 1] = pb10; sv1[b + 1] = pb11;
            int5 sc = {b * 256, t, 0, 0, 0};
            v_bf16_st_tnsr(sc, sum, pb00); sc[0] += 128; v_bf16_st_tnsr(sc, sum, pb01);
            sc[0] += 128; v_bf16_st_tnsr(sc, sum, pb10); sc[0] += 128; v_bf16_st_tnsr(sc, sum, pb11);
            sq0 = sq0 + q00.v1 * q00.v1; sq1 = sq1 + q00.v2 * q00.v2;
            sq2 = sq2 + q01.v1 * q01.v1; sq3 = sq3 + q01.v2 * q01.v2;
            sq0 = sq0 + q10.v1 * q10.v1; sq1 = sq1 + q10.v2 * q10.v2;
            sq2 = sq2 + q11.v1 * q11.v1; sq3 = sq3 + q11.v2 * q11.v2;
        }
        for (; b < NB; ++b) {
            int5 xc = {b * 256, t, 0, 0, 0};
            bfloat128 xb0 = v_bf16_ld_tnsr_b(xc, x);  xc[0] += 128; bfloat128 xb1 = v_bf16_ld_tnsr_b(xc, x);
            int5 rc = {b * 256, t, 0, 0, 0};
            bfloat128 rb0 = v_bf16_ld_tnsr_b(rc, res); rc[0] += 128; bfloat128 rb1 = v_bf16_ld_tnsr_b(rc, res);
            float128 f0 = v_convert_bf16_to_f32_all_b(xb0), f1 = v_convert_bf16_to_f32_all_b(xb1);
            float128 r0 = v_convert_bf16_to_f32_all_b(rb0), r1 = v_convert_bf16_to_f32_all_b(rb1);
            float64 a0 = f0.v1 + r0.v1, a1 = f0.v2 + r0.v2, a2 = f1.v1 + r1.v1, a3 = f1.v2 + r1.v2;
            float128 p0; p0.v1 = a0; p0.v2 = a1;
            bfloat128 pb0 = v_convert_f32_to_bf16_all_b(p0);
            float128 q0 = v_convert_bf16_to_f32_all_b(pb0);
            float128 p1; p1.v1 = a2; p1.v2 = a3;
            bfloat128 pb1 = v_convert_f32_to_bf16_all_b(p1);
            float128 q1 = v_convert_bf16_to_f32_all_b(pb1);
            sv0[b] = pb0; sv1[b] = pb1;
            int5 sc = {b * 256, t, 0, 0, 0};
            v_bf16_st_tnsr(sc, sum, pb0); sc[0] += 128; v_bf16_st_tnsr(sc, sum, pb1);
            sq0 = sq0 + q0.v1 * q0.v1; sq1 = sq1 + q0.v2 * q0.v2;
            sq2 = sq2 + q1.v1 * q1.v1; sq3 = sq3 + q1.v2 * q1.v2;
        }
        float64 ss = REDUCE64((sq0 + sq1) + (sq2 + sq3), P8, P4, P2, P1);   // full 5120-row sum, broadcast
        float64 inv = v_rsqrt_f32(ss * inv_n + eps);

        // pass 2: the rounded sums from VLM (never a same-kernel HBM reload), apply inv * (1 + w)
        float64 one = 1.f; (void)one;
        int b2 = 0;
        for (; b2 + 2 <= NB; b2 += 2) {
            int5 wc = {b2 * 256, 0, 0, 0, 0};
            bfloat128 wb00 = v_bf16_ld_tnsr_b(wc, w);   wc[0] += 128; bfloat128 wb01 = v_bf16_ld_tnsr_b(wc, w);
            wc[0] += 128; bfloat128 wb10 = v_bf16_ld_tnsr_b(wc, w);   wc[0] += 128; bfloat128 wb11 = v_bf16_ld_tnsr_b(wc, w);
            bfloat128 sb00 = sv0[b2], sb01 = sv1[b2], sb10 = sv0[b2 + 1], sb11 = sv1[b2 + 1];
            float128 s00 = v_convert_bf16_to_f32_all_b(sb00), s01 = v_convert_bf16_to_f32_all_b(sb01);
            float128 s10 = v_convert_bf16_to_f32_all_b(sb10), s11 = v_convert_bf16_to_f32_all_b(sb11);
            float128 g00 = v_convert_bf16_to_f32_all_b(wb00), g01 = v_convert_bf16_to_f32_all_b(wb01);
            float128 g10 = v_convert_bf16_to_f32_all_b(wb10), g11 = v_convert_bf16_to_f32_all_b(wb11);
            float128 o00; o00.v1 = WN(s00.v1, g00.v1); o00.v2 = WN(s00.v2, g00.v2);
            float128 o01; o01.v1 = WN(s01.v1, g01.v1); o01.v2 = WN(s01.v2, g01.v2);
            float128 o10; o10.v1 = WN(s10.v1, g10.v1); o10.v2 = WN(s10.v2, g10.v2);
            float128 o11; o11.v1 = WN(s11.v1, g11.v1); o11.v2 = WN(s11.v2, g11.v2);
            int5 oc = {b2 * 256, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o00)); oc[0] += 128;
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o01)); oc[0] += 128;
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o10)); oc[0] += 128;
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o11));
        }
        for (; b2 < NB; ++b2) {
            bfloat128 sb0 = sv0[b2], sb1 = sv1[b2];
            int5 wc = {b2 * 256, 0, 0, 0, 0};
            bfloat128 wb0 = v_bf16_ld_tnsr_b(wc, w);    wc[0] += 128; bfloat128 wb1 = v_bf16_ld_tnsr_b(wc, w);
            float128 s0 = v_convert_bf16_to_f32_all_b(sb0), s1 = v_convert_bf16_to_f32_all_b(sb1);
            float128 g0 = v_convert_bf16_to_f32_all_b(wb0), g1 = v_convert_bf16_to_f32_all_b(wb1);
            float128 o0; o0.v1 = WN(s0.v1, g0.v1); o0.v2 = WN(s0.v2, g0.v2);
            float128 o1; o1.v1 = WN(s1.v1, g1.v1); o1.v2 = WN(s1.v2, g1.v2);
            int5 oc = {b2 * 256, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
            v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o1));
        }
    }
}
