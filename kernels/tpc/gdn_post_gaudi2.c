// -DGATE_SIGMOID build (qwen4_exp / Qwen3.8-Flash-Next): the output gate is sigmoid(z) instead of silu(z) = z * sigmoid(z)
#ifdef GATE_SIGMOID
#define GATE(z) v_sigmoid_f32(z)
#else
#define GATE(z) ((z) * v_sigmoid_f32(z))
#endif
#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_post: gated RMSNorm per value head and re-layout to the bf16 {6144, T} row the out_proj GEMM reads.
//   core[h] = rmsnorm(o[h]) * w * silu(z[h]),  z[h] = channels 10240 + 128h .. of qkvz
//  in0 o f32 {128, 48, T}   in1 qkvz bf16 {16384, T}   in2 norm_w f32 {128}   out0 core bf16 {6144, T}   scalar: T, flags
//  flags bit5: o is the chunked X {128, 2C, 48, NCH} of the MME prefill path (C = flags >> 16): row t at {., t % C, h, t / C}
//  one index-space point = one head (48); loops over T.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

#ifdef DECODE
__local__ float64 RV[32];   // a kernel that uses special functions has 16 KB of vector memory for static data and spills together
#endif
void main(tensor o, tensor qkvz, tensor norm_w, tensor core, int T, int flags)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, inv_n = 1.0f / 128.0f;
    int5 wc = {0, 0, 0, 0, 0};
    float64 wa = v_f32_ld_tnsr_b(wc, norm_w); wc[0] = 64; float64 wb = v_f32_ld_tnsr_b(wc, norm_w);

#ifdef DECODE   // decode / verify build (row-major o, T <= 32; integration): the chunked-X code and the unroll compiled out
    const int chunked = 0, CH = 1; (void)CH; (void)flags;
#else
    const int chunked = flags & 32, CH = chunked ? (flags >> 16) : 1;
#endif
#ifdef DECODE
    // ONE special-function table at a time (2026-11-11, level-1 traces): rsqrt and sigmoid are LUT functions and a TPC holds one table;
    // alternating them per row cost this kernel 0.81 ms per 16-row step (0.13 with one table) plus ~8 us of idle time after the node.
    // Per group of heads: every row's rsqrt first (kept in local memory), then every row's gate. Same operations per element, bit-identical.
    const int HG = 32 / (T > 0 ? T : 1) > 0 ? 32 / (T > 0 ? T : 1) : 1;
    for (int hb = idx[0]; hb < idx[0] + isz[0]; hb += HG) {
        const int he = hb + HG < idx[0] + isz[0] ? hb + HG : idx[0] + isz[0];
        for (int h = hb; h < he; ++h)
#pragma nounroll
        for (int t = 0; t < T; ++t) {
            int5 oc = {0, h, t, 0, 0};
            float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
            float64 ss = REDUCE64(oa * oa + ob * ob, P8, P4, P2, P1);
            RV[(h - hb) * T + t] = v_rsqrt_f32(ss * inv_n + eps);
        }
        for (int h = hb; h < he; ++h)
#pragma nounroll
        for (int t = 0; t < T; ++t) {
            int5 oc = {0, h, t, 0, 0};
            float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
            float64 r = RV[(h - hb) * T + t];
            int5 zc = {GDN_QKV + 128 * h, t, 0, 0, 0};
            float128 z = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(zc, qkvz));
            float128 res;
            res.v1 = (oa * r) * wa * GATE(z.v1);
            res.v2 = (ob * r) * wb * GATE(z.v2);
            int5 cc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(res));
        }
    }
#else
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
#ifndef DECODE
#pragma unroll 4
#endif
        for (int t = 0; t < T; ++t) {
            int5 oc = {0, chunked ? t % CH : h, chunked ? h : t, chunked ? t / CH : 0, 0};
            float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
            float64 ss = REDUCE64(oa * oa + ob * ob, P8, P4, P2, P1);
            float64 r = v_rsqrt_f32(ss * inv_n + eps);
            int5 zc = {GDN_QKV + 128 * h, t, 0, 0, 0};
            float128 z = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(zc, qkvz));
            float128 res;
            res.v1 = (oa * r) * wa * GATE(z.v1);
            res.v2 = (ob * r) * wb * GATE(z.v2);
            int5 cc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(res));
        }
    }
#endif
}
