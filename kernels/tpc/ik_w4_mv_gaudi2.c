// gaudium_ik_w4_gu / _dn / _w8_gu / _w8_dn: Inkling 2 routed expert MV kernels (block-16 NVFP4 or block-32 W8).
// Builds: -DGU (soft-capped SwiGLU + Q8 output), default (down), -DW8 (block-32 int8 codes). Must build with -ffp-contract=off.
#include "moe_w4_common.h"
__local__ uchar256 PAT[64];

#define IK_TANH(d_) ({ float64 _d = (d_); float64 _a = v_f32_abs_b(_d); float64 _t = EXPM(v_f32_max_b(_a * -2.0f, -87.0f)); \
    float64 _big = (1.0f - _t) * RECIP(_t + 1.0f); float64 _d2 = _a * _a; \
    float64 _t1 = _d2 * 0.13333333333f - 0.33333333333f; float64 _small = _a * (_t1 * _d2 + 1.0f); \
    float64 _m = v_f32_sel_less_f32_b(_a, 0.015625f, _small, _big); \
    v_f32_sel_less_f32_b(_d, 0.0f, _m * -1.0f, _m); })
#define IK_SWIGLU(g_, u_, icg_, icu_, cgcu_) ({ float64 _g = (g_), _u = (u_); \
    float64 _nx = v_f32_min_b(v_f32_max_b(-_g, -87.0f), 80.0f); \
    float64 _sig = RECIP(1.0f + EXPM(_nx)); \
    IK_TANH(_g * (icg_)) * _sig * IK_TANH(_u * (icu_)) * (cgcu_); })

#ifdef GU
#define XCO(g_, k_, t_) ((int5){0, (g_), (t_), 0, 0})
#define OCO(g_, k_, t_) ((int5){0, (g_), (k_), (t_), 0})
#ifdef W8
#define Q4_SPAN 16
#define XVEC(b_, pp_) (b_)
#define XIDX(b_, pp_) (4 * (pp_))
#define XIDX2(b_, pp_) (4 * (pp_) + 1)
#else
#define Q4_SPAN 8
#define XLOADB(b_) ((b_) >> 1)
#define XIDX(b_, pp_) (32 * ((b_) & 1) + 4 * (pp_))
#define XIDX2(b_, pp_) (32 * ((b_) & 1) + 4 * (pp_) + 1)
#endif
#else
#define XCO(g_, k_, t_) ((int5){0, (g_), (k_), (t_), 0})
#ifdef W8
#define Q4_SPAN 16
#define XVEC(b_, pp_) (2 * (b_) + ((pp_) >> 3))
#define XIDX(b_, pp_) (8 * ((pp_) & 7))
#define XIDX2(b_, pp_) (8 * ((pp_) & 7) + 4)
#else
#define Q4_SPAN 8
#define XLOADB(b_) (b_)
#define XIDX(b_, pp_) (8 * (pp_))
#define XIDX2(b_, pp_) (8 * (pp_) + 4)
#endif
#endif
#define LDI(tn, c0) s_i32_ld_g(gen_addr(((int5){(c0), 0, 0, 0, 0}), tn))

#ifdef W8
#define LUT_ARG
#define LUT_PAR
#else
#define LUT_ARG , LUT
#define LUT_PAR , char256 LUT
#endif
#ifdef GU
#define OUT_PAR tensor hq, tensor hs
#define OUT_ARG hq, hs
#define ACT_PAR , float icg, float icu, float cgcu
#define ACT_ARG , icg, icu, cgcu
#else
#define OUT_PAR tensor y
#define OUT_ARG y
#define ACT_PAR
#define ACT_ARG
#endif

#ifndef W4_NACC1
#define W4_NACC1 2
#endif
#ifndef W4_Q4_UNROLL
#define W4_Q4_UNROLL 2
#endif
#ifndef W4_Q4_UNROLL_N
#define W4_Q4_UNROLL_N 1
#endif
#define W4_CAT2(a, b) a##b
#define W4_CAT(a, b) W4_CAT2(a, b)
#define N 1
#include "ik_w4_mv_body.h"
#undef N
#define N 2
#include "ik_w4_mv_body.h"
#undef N
#define N 3
#include "ik_w4_mv_body.h"
#undef N
#define N 4
#include "ik_w4_mv_body.h"
#undef N

#if defined(W8) && defined(GU)
void main(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, OUT_PAR, int NT, int M, int KT, int cg_bits, int cu_bits)
#elif defined(W8)
void main(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, OUT_PAR, int NT, int M, int KT)
#elif defined(GU)
void main(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, tensor lut, OUT_PAR, int NT, int M, int KT, int cg_bits, int cu_bits)
#else
void main(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, tensor lut, OUT_PAR, int NT, int M, int KT)
#endif
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    for (int e = 0; e < 64; ++e) PAT[e] = (unsigned char)(e | 0x80);
#ifndef W8
    const char256 LUT = v_i8_ld_tnsr_b(((int5){0, 0, 0, 0, 0}), lut);
#endif
#ifdef GU
    const float cg = *(float*)&cg_bits, cu = *(float*)&cu_bits;
    const float icg = 1.0f / cg, icu = 1.0f / cu, cgcu = cg * cu;
#endif
    const int NBLK = get_dim_size(scales, 1);
    const int U = LDI(plan, 0);
    const int m0 = off[0], m1 = off[0] + sz[0];
    int tt = m0 % NT, q = m0 / NT;
    for (int m = m0; m < m1; ++m) {
        for (int j = q; j < U; j += M) {
            const int slot = LDI(plan, 1 + 4 * j), n = LDI(plan, 2 + 4 * j), start = LDI(plan, 3 + 4 * j);
            for (int r0 = 0; r0 < n; r0 += 4) {
                const int nr = n - r0 < 4 ? n - r0 : 4, rb = 1 + 4 * KT + start + r0;
                const int last = rb + nr - 1;
                const int i1 = rb + 1 < last ? rb + 1 : last, i2 = rb + 2 < last ? rb + 2 : last, i3 = rb + 3 < last ? rb + 3 : last;
                const int c0 = LDI(plan, rb), c1 = LDI(plan, i1), c2 = LDI(plan, i2), c3 = LDI(plan, i3);
                const int t0 = c0 >> 4, t1 = c1 >> 4, t2 = c2 >> 4, t3 = c3 >> 4, k0 = c0 & 15, k1 = c1 & 15, k2 = c2 & 15, k3 = c3 & 15;
                if (nr == 1) body1(xq, xs, codes, scales, OUT_ARG LUT_ARG ACT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
                else if (nr == 2) body2(xq, xs, codes, scales, OUT_ARG LUT_ARG ACT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
                else if (nr == 3) body3(xq, xs, codes, scales, OUT_ARG LUT_ARG ACT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
                else body4(xq, xs, codes, scales, OUT_ARG LUT_ARG ACT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
            }
        }
        if (++tt == NT) { tt = 0; ++q; }
    }
}
