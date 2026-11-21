// -DGATE_SIGMOID build (qwen4_exp / Qwen3.8-Flash-Next): the output gate is sigmoid(z) instead of silu(z) = z * sigmoid(z)
#ifdef GATE_SIGMOID
#define GATE(z) v_sigmoid_f32(z)
#else
#define GATE(z) ((z) * v_sigmoid_f32(z))
#endif
#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_post, row-parallel prefill build (flags bit6, chunked X input only). Same arithmetic per element
// as gdn_post_gaudi2.c (bit-identical), index space {48 heads, NRG row groups of RG rows}, rows in batches of GB = 16:
// phase A r_t = rsqrt(mean(o^2) + eps) (rsqrt table) into VLM, phase B silu(z) (tanh table) and the products. The serial
// kernel switched LUT tables twice per row (tpc-clang LUT-capacity warning on gdn_post_gaudi2.c).
//  in0 X f32 {128, 2C, 48, NCH} (row t at {., t % C, h, t / C})  in1 qkvz bf16 {16384, T}  in2 norm_w f32 {128}  out0 core bf16 {6144, T}
//  scalars: T, flags (C = flags >> 16), RG
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#ifndef GB
#define GB 16   // rows per LUT phase (feature/prefill-tpc: -DGB=32/64 variants trade VLM for fewer table switches)
#endif
__local__ float64 RBUF[GB];

void main(tensor o, tensor qkvz, tensor norm_w, tensor core, int T, int flags, int RG)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, inv_n = 1.0f / 128.0f;
    int5 wc = {0, 0, 0, 0, 0};
    float64 wa = v_f32_ld_tnsr_b(wc, norm_w); wc[0] = 64; float64 wb = v_f32_ld_tnsr_b(wc, norm_w);
    const int CH = flags >> 16;
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
      for (int rg = idx[1]; rg < idx[1] + isz[1]; ++rg) {
        const int t0 = rg * RG, t1 = (t0 + RG < T) ? t0 + RG : T;
        for (int tb = t0; tb < t1; tb += GB) {
            const int n = (tb + GB < t1) ? GB : t1 - tb;
#pragma unroll 4
            for (int i = 0; i < n; ++i) {   // phase A: rsqrt table
                const int t = tb + i;
                int5 oc = {0, t % CH, h, t / CH, 0};
                float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
                float64 ss = REDUCE64(oa * oa + ob * ob, P8, P4, P2, P1);
                RBUF[i] = v_rsqrt_f32(ss * inv_n + eps);
            }
#pragma unroll 4
            for (int i = 0; i < n; ++i) {   // phase B: tanh table
                const int t = tb + i;
                int5 oc = {0, t % CH, h, t / CH, 0};
                float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
                float64 r = RBUF[i];
                int5 zc = {GDN_QKV + 128 * h, t, 0, 0, 0};
                float128 z = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(zc, qkvz));
                float128 res;
                res.v1 = (oa * r) * wa * GATE(z.v1);
                res.v2 = (ob * r) * wb * GATE(z.v2);
                int5 cc = {128 * h, t, 0, 0, 0};
                v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(res));
            }
        }
      }
    }
}
