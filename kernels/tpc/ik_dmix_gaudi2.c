// gaudium_ik_dmix: the block drafter's context inputs. Per row of the main model's four taps (the inputs of three layer applications
// and the final norm output), four mixes with softmax weights, each normalised with its own weight:
//   c_l = bf16( m rsqrt(mean(m^2) + eps) w_l ),   m = sum_j alpha[4 l + j] x_j        (f32 inside, one rounding)
// Mixes 0, 1, 2 feed the drafter layers' context keys and values, mix 3 the Markov head. One index-space member is one row.
//  in0 .. in3 x_j bf16 {H, T}   in4 w bf16 {H, 4} (column l: the norm weight of mix l)   in5 alpha f32 {16}
//  out0 .. out3 c_l bf16 {H, T}.   index space {T}.   scalar: eps (f32 bits)
#include "ik_common.h"
#define LDX(x_, b_, t_) ({ int5 _c = {128 * (b_), (t_), 0, 0, 0}; v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_c, x_)); })
#define ALPHA(i_) ({ int5 _c = {(i_), 0, 0, 0, 0}; s_f32_ld_g(gen_addr(_c, alpha)); })
#define MIX(l_) ({ float128 _m; \
    _m.v1 = ya.v1 * a##l_##0 + yb.v1 * a##l_##1 + yc.v1 * a##l_##2 + yd.v1 * a##l_##3; \
    _m.v2 = ya.v2 * a##l_##0 + yb.v2 * a##l_##1 + yc.v2 * a##l_##2 + yd.v2 * a##l_##3; _m; })
#define OUTL(l_, cl_) { const float128 _m = MIX(l_); const float128 _w = LDW(b, l_); \
    float128 _o; _o.v1 = _m.v1 * r##l_ * _w.v1; _o.v2 = _m.v2 * r##l_ * _w.v2; \
    int5 _oc = {128 * b, t, 0, 0, 0}; v_bf16_st_tnsr(_oc, cl_, v_convert_f32_to_bf16_all_b(_o)); }
#define LDW(b_, l_) ({ int5 _c = {128 * (b_), (l_), 0, 0, 0}; v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_c, w)); })

void main(tensor x0, tensor x1, tensor x2, tensor x3, tensor w, tensor alpha, tensor c0, tensor c1, tensor c2, tensor c3, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int H = get_dim_size(c0, 0), NB = H / 128;
    const float eps = *(float*)&eps_bits, invh = 1.0f / (float)H;
    const float a00 = ALPHA(0), a01 = ALPHA(1), a02 = ALPHA(2), a03 = ALPHA(3);
    const float a10 = ALPHA(4), a11 = ALPHA(5), a12 = ALPHA(6), a13 = ALPHA(7);
    const float a20 = ALPHA(8), a21 = ALPHA(9), a22 = ALPHA(10), a23 = ALPHA(11);
    const float a30 = ALPHA(12), a31 = ALPHA(13), a32 = ALPHA(14), a33 = ALPHA(15);
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        float64 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (int b = 0; b < NB; ++b) {
            const float128 ya = LDX(x0, b, t);
            const float128 yb = LDX(x1, b, t);
            const float128 yc = LDX(x2, b, t);
            const float128 yd = LDX(x3, b, t);
            const float128 m0 = MIX(0);
            const float128 m1 = MIX(1);
            const float128 m2 = MIX(2);
            const float128 m3 = MIX(3);
            s0 = s0 + m0.v1 * m0.v1 + m0.v2 * m0.v2; s1 = s1 + m1.v1 * m1.v1 + m1.v2 * m1.v2;
            s2 = s2 + m2.v1 * m2.v1 + m2.v2 * m2.v2; s3 = s3 + m3.v1 * m3.v1 + m3.v2 * m3.v2;
        }
        const float64 r0 = v_rsqrt_f32(REDUCE64(s0, P8, P4, P2, P1) * invh + eps);
        const float64 r1 = v_rsqrt_f32(REDUCE64(s1, P8, P4, P2, P1) * invh + eps);
        const float64 r2 = v_rsqrt_f32(REDUCE64(s2, P8, P4, P2, P1) * invh + eps);
        const float64 r3 = v_rsqrt_f32(REDUCE64(s3, P8, P4, P2, P1) * invh + eps);
        for (int b = 0; b < NB; ++b) {
            const float128 ya = LDX(x0, b, t);
            const float128 yb = LDX(x1, b, t);
            const float128 yc = LDX(x2, b, t);
            const float128 yd = LDX(x3, b, t);
            OUTL(0, c0) OUTL(1, c1) OUTL(2, c2) OUTL(3, c3)
        }
    }
}
