// gaudium_moe_gu / gaudium_moe_dn: routed-expert matrix-vector products on W4L (4-bit codes + 16-level table) or W8 (int8) slot
// pools, experts selected on the device (docs/notes/moe-w4l.md 1.2, 2, 3). Builds: -DGU (gate+up, SwiGLU, Q8 of h), default (down),
// -DW8 (int8 codes, no table).
//
// Index space {NT * M}: point m works on tile tt = m % NT (256 output lanes) of the plan entries j = m / NT, + M, + 2M, ... < U.
// An entry is one device expert with its rows (gaudium_moe_plan); its rows are processed in groups of up to 4, each group reads the
// expert's tile once. Per column pair: W4L 2 unpacking loads (128 B -> 256 codes each), OR 0x80 + one u8 shuffle per column into the
// 16-level table; per row 2 broadcast shuffles of the quantised input and one v_i8_mac_x2. Per block of 32 columns and row (the
// numerics contract, w4::ref::matvec): A = A + float(int32 dot) * (w_scale * x_scale), separately rounded, blocks in ascending order.
//
// GU: in xq i8 {256, H/32, T}, xs f32 {64, H/32, T}, plan, codes u8 {256, H/2, NT, S}, scales f16 {256, H/32, NT, S}, lut i8 {256}
//     out hq i8 {256, EF/16, K, T}, hs f32 {64, EF/32, K, T}         (NT = EF/128; tile tt = gate and up channels 128 tt .. +127)
//     x operand: one vector per block (32 columns), column 32b + c' at index 4 (c' >> 1) + (c' & 1) of every dual group;
//     hq: one vector per 16 channels, channel 16v + c' at index 4 c' of every dual group. Every broadcast is a UNIFORM u8 shuffle
//     (non-uniform u8 / f32 shuffles that cross 16-byte halves do not follow the documented model: docs/notes/moe-w4l.md 2.4).
// DN: in hq, hs, plan, codes u8 {256, EF/2, NT, S} (W8: i8 {256, EF, NT, S}), scales f16 {256, EF/32, NT, S}, lut (W4L only)
//     out y f32 {H, K, T}, channels deinterleaved per 128 (position 64 h + p of chunk c = channel 128 c + 2p + h)   (NT = H/256)
// scalars: NT, M, KT (K * T: where the plan's row list starts)
#include "moe_w4_common.h"
__local__ uchar256 PAT[64];   // broadcast pattern of group column e: (e | 0x80) in every byte

#ifdef GU
#define XCO(g_, k_, t_) ((int5){0, (g_), (t_), 0, 0})
#define OCO(g_, k_, t_) ((int5){0, (g_), (k_), (t_), 0})
#define XVEC(b_, pp_) (b_)                                   // x vector of column pair pp of block b
#define XIDX(pp_) (4 * (pp_))                                // index of its first column (the second: + 1)
#define XIDX2(pp_) (4 * (pp_) + 1)
#else
#define XCO(g_, k_, t_) ((int5){0, (g_), (k_), (t_), 0})
#define XVEC(b_, pp_) (2 * (b_) + ((pp_) >> 3))
#define XIDX(pp_) (8 * ((pp_) & 7))
#define XIDX2(pp_) (8 * ((pp_) & 7) + 4)
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
#else
#define OUT_PAR tensor y
#define OUT_ARG y
#endif

#ifndef W4_NO_WIDE
#define W4_NO_WIDE 1    // the wide row-group path (bodyW) is slower on the card today (moe-w4l.md 6.3): opt-in with -DW4_NO_WIDE=0
#endif
#ifndef W4_NACC1
#define W4_NACC1 2      // independent accumulators for a one-row group (2: -9 % cycles against 4 in libtpcsim, 1: +6 %; 2026-11-14)
#endif
#ifndef W4_Q4_UNROLL
#define W4_Q4_UNROLL 2   // column-pair groups of 4 per loop trip. libtpcsim, T = 1, with W4_NACC1 2: 2 = gu -10.5 % / dn -8.7 % cycles against 1;
                         // 4 (the whole block) = +64 % (register pressure). Bit-exact either way (2026-11-14)
#endif
#ifndef W4_Q4_UNROLL_N
#define W4_Q4_UNROLL_N 1   // row-group sizes that get the two-group trip (larger bodies: tpc-clang does not finish)
#endif
#define W4_CAT2(a, b) a##b
#define W4_CAT(a, b) W4_CAT2(a, b)
#define N 1
#include "moe_w4_mv_body.h"
#undef N
#define N 2
#include "moe_w4_mv_body.h"
#undef N
#define N 3
#include "moe_w4_mv_body.h"
#undef N
#define N 4
#include "moe_w4_mv_body.h"
#undef N

// ---------------------------------------------------------------- wide row groups (prefill): an entry with more than 4 rows
// The tile's weights of one block (32 columns, 16 pairs = 32 decoded vectors, 8 KB) are decoded ONCE into VLM and then used by up to
// W4_GMAX rows (two rows at a time, one int32 accumulator chain each; the rows' f32 accumulators live in VLM). Every row performs the
// same operations in the same order as in body1..4 (exact int32 block dots, then cvt, mul, mul, add per block): bit-identical.
#define W4_GMAX 32
__local__ char256 WB[32];
__local__ float64 VA[4 * (W4_GMAX + 1)];   // [4 i + j]; row W4_GMAX is the dummy partner of an odd last row
__local__ int RCK[W4_GMAX];
__local__ int RCT[W4_GMAX];

static inline void finish_row(OUT_PAR, int tt, int kk, int tr, float64 A0, float64 A1, float64 A2, float64 A3)
{
#if defined(GU)
    const uchar256 P8 = xor_pat16(8), P4 = xor_pat16(4), P2 = xor_pat16(2), P1 = xor_pat16(1);
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const float64 h = hh ? SWIGLU(A1, A3) : SWIGLU(A0, A2);
        float64 a = DGMAX(v_f32_abs_b(h), P8, P4, P2, P1);
        a = v_f32_max_b(a, MOVDG_F(a, 1, 0, 3, 2));
        float64 q, d;
        Q8(h, a, q, d);
        const int blk = 4 * tt + 2 * hh;
        v_f32_st_tnsr(OCO(blk, kk, tr), hs, MOVDG_F(d, 0, 0, 0, 0));
        v_f32_st_tnsr(OCO(blk + 1, kk, tr), hs, MOVDG_F(d, 2, 2, 2, 2));
        float256 cq; cq.v1 = q; cq.v2 = q; cq.v3 = q; cq.v4 = q;
        const char256 C = v_convert_f32_to_i8_all_b(cq, SW_RHNE);
        const int v0 = 8 * tt + 4 * hh;
        v_i8_st_tnsr(OCO(v0, kk, tr), hq, MOVDG_B(C, 0, 0, 0, 0)); v_i8_st_tnsr(OCO(v0 + 1, kk, tr), hq, MOVDG_B(C, 1, 1, 1, 1));
        v_i8_st_tnsr(OCO(v0 + 2, kk, tr), hq, MOVDG_B(C, 2, 2, 2, 2)); v_i8_st_tnsr(OCO(v0 + 3, kk, tr), hq, MOVDG_B(C, 3, 3, 3, 3));
    }
#else
    int5 oc = {256 * tt, kk, tr, 0, 0};
    v_f32_st_tnsr(oc, y, A0); oc[0] += 64; v_f32_st_tnsr(oc, y, A1); oc[0] += 64;
    v_f32_st_tnsr(oc, y, A2); oc[0] += 64; v_f32_st_tnsr(oc, y, A3);
#endif
}

static inline void bodyW(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, OUT_PAR LUT_PAR, int NBLK, int slot, int tt, int rb, int nr)
{
    for (int i = 0; i < nr; ++i) { const int c = LDI(plan, rb + i); RCK[i] = c & 15; RCT[i] = c >> 4; }
    for (int i = 0; i < 4 * (W4_GMAX + 1); ++i) VA[i] = 0.0f;
    for (int b = 0; b < NBLK; ++b) {
        // decode the block once
        #pragma unroll 1
        for (int q4 = 0; q4 < 4; ++q4) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int pp = 4 * q4 + q, cp = 16 * b + pp;
#ifdef W8
                int5 cc = {0, 2 * cp, tt, slot, 0};
                WB[2 * pp] = v_i8_ld_tnsr_b(cc, codes); cc[1] += 1;
                WB[2 * pp + 1] = v_i8_ld_tnsr_b(cc, codes);
#else
                int5 cc = {0, cp, tt, slot, 0};
                const uchar256 u0 = v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8); cc[0] = 128;
                const uchar256 u1 = v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8);
                const uchar256 i0 = u0 | 0x80, i1 = u1 | 0x80;
                WB[2 * pp] = v_i8_shuffle_b(LUT, i0, 0, *(const char256*)&i0); WB[2 * pp + 1] = v_i8_shuffle_b(LUT, i1, 0, *(const char256*)&i1);
#endif
            }
        }
        int5 sc = {0, b, tt, slot, 0};
        const ushort128 s0 = v_u16_ld_tnsr_b(sc, scales); sc[0] = 128;
        const ushort128 s1 = v_u16_ld_tnsr_b(sc, scales);
        const float128 w01 = v_convert_f16_to_f32_all_b(*(half128*)&s0), w23 = v_convert_f16_to_f32_all_b(*(half128*)&s1);
        #pragma unroll 1
        for (int i = 0; i < nr; i += 2) {
            const int ib = i + 1 < nr ? i + 1 : i, vb = i + 1 < nr ? i + 1 : W4_GMAX;
            const int ka = RCK[i], ta = RCT[i], kb = RCK[ib], tb = RCT[ib];
            const char256 xga = v_i8_ld_tnsr_b(XCO(XVEC(b, 0), ka, ta), xq), xha = v_i8_ld_tnsr_b(XCO(XVEC(b, 8), ka, ta), xq);
            const char256 xgb = v_i8_ld_tnsr_b(XCO(XVEC(b, 0), kb, tb), xq), xhb = v_i8_ld_tnsr_b(XCO(XVEC(b, 8), kb, tb), xq);
            int256 aa, ab; aa.v1 = 0; aa.v2 = 0; aa.v3 = 0; aa.v4 = 0; ab.v1 = 0; ab.v2 = 0; ab.v3 = 0; ab.v4 = 0;
            #pragma unroll 1
            for (int q4 = 0; q4 < 4; ++q4) {
                const char256 xa = q4 < 2 ? xga : xha, xb = q4 < 2 ? xgb : xhb;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int pp = 4 * q4 + q;
                    const char256 l0 = WB[2 * pp], l1 = WB[2 * pp + 1];
                    const uchar256 pa0 = PAT[XIDX(pp)], pa1 = PAT[XIDX2(pp)], pb0 = PAT[XIDX(pp)], pb1 = PAT[XIDX2(pp)];
                    const char256 ba0 = v_i8_shuffle_b(xa, pa0, 0, *(const char256*)&pa0), ba1 = v_i8_shuffle_b(xa, pa1, 0, *(const char256*)&pa1);
                    const char256 bb0 = v_i8_shuffle_b(xb, pb0, 0, *(const char256*)&pb0), bb1 = v_i8_shuffle_b(xb, pb1, 0, *(const char256*)&pb1);
                    aa = v_i8_mac_b(l0, ba0, aa); ab = v_i8_mac_b(l0, bb0, ab);
                    aa = v_i8_mac_b(l1, ba1, aa); ab = v_i8_mac_b(l1, bb1, ab);
                }
            }
            const float64 dxa = v_f32_ld_tnsr_b(XCO(b, ka, ta), xs), dxb = v_f32_ld_tnsr_b(XCO(b, kb, tb), xs);
            float64 s, p;
            s = w01.v1 * dxa; p = v_convert_i32_to_f32_b(aa.v1, SW_RHNE) * s; VA[4 * i + 0] = VA[4 * i + 0] + p;
            s = w01.v2 * dxa; p = v_convert_i32_to_f32_b(aa.v2, SW_RHNE) * s; VA[4 * i + 1] = VA[4 * i + 1] + p;
            s = w23.v1 * dxa; p = v_convert_i32_to_f32_b(aa.v3, SW_RHNE) * s; VA[4 * i + 2] = VA[4 * i + 2] + p;
            s = w23.v2 * dxa; p = v_convert_i32_to_f32_b(aa.v4, SW_RHNE) * s; VA[4 * i + 3] = VA[4 * i + 3] + p;
            s = w01.v1 * dxb; p = v_convert_i32_to_f32_b(ab.v1, SW_RHNE) * s; VA[4 * vb + 0] = VA[4 * vb + 0] + p;
            s = w01.v2 * dxb; p = v_convert_i32_to_f32_b(ab.v2, SW_RHNE) * s; VA[4 * vb + 1] = VA[4 * vb + 1] + p;
            s = w23.v1 * dxb; p = v_convert_i32_to_f32_b(ab.v3, SW_RHNE) * s; VA[4 * vb + 2] = VA[4 * vb + 2] + p;
            s = w23.v2 * dxb; p = v_convert_i32_to_f32_b(ab.v4, SW_RHNE) * s; VA[4 * vb + 3] = VA[4 * vb + 3] + p;
        }
    }
    #pragma unroll 1
    for (int i = 0; i < nr; ++i) finish_row(OUT_ARG, tt, RCK[i], RCT[i], VA[4 * i], VA[4 * i + 1], VA[4 * i + 2], VA[4 * i + 3]);
}

#ifdef W8
void main(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, OUT_PAR, int NT, int M, int KT)
#else
void main(tensor xq, tensor xs, tensor plan, tensor codes, tensor scales, tensor lut, OUT_PAR, int NT, int M, int KT)
#endif
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    for (int e = 0; e < 64; ++e) PAT[e] = (unsigned char)(e | 0x80);
#ifdef GU
#endif
#ifndef W8
    const char256 LUT = v_i8_ld_tnsr_b(((int5){0, 0, 0, 0, 0}), lut);
#endif
    const int NBLK = get_dim_size(scales, 1);
    const int U = LDI(plan, 0);
    const int m0 = off[0], m1 = off[0] + sz[0];
    int tt = m0 % NT, q = m0 / NT;
    for (int m = m0; m < m1; ++m) {
        for (int j = q; j < U; j += M) {
            const int slot = LDI(plan, 1 + 4 * j), n = LDI(plan, 2 + 4 * j), start = LDI(plan, 3 + 4 * j);
            if (n > 4 && !W4_NO_WIDE) {   // prefill: all of the expert's rows (in groups of W4_GMAX) per decode of its weights
                for (int r0 = 0; r0 < n; r0 += W4_GMAX)
                    bodyW(xq, xs, plan, codes, scales, OUT_ARG LUT_ARG, NBLK, slot, tt, 1 + 4 * KT + start + r0, n - r0 < W4_GMAX ? n - r0 : W4_GMAX);
                continue;
            }
            for (int r0 = 0; r0 < n; r0 += 4) {
                const int nr = n - r0 < 4 ? n - r0 : 4, rb = 1 + 4 * KT + start + r0;
                const int last = rb + nr - 1;   // rows past nr re-read the last row (never stored): no predicated loads
                const int i1 = rb + 1 < last ? rb + 1 : last, i2 = rb + 2 < last ? rb + 2 : last, i3 = rb + 3 < last ? rb + 3 : last;
                const int c0 = LDI(plan, rb), c1 = LDI(plan, i1), c2 = LDI(plan, i2), c3 = LDI(plan, i3);
                const int t0 = c0 >> 4, t1 = c1 >> 4, t2 = c2 >> 4, t3 = c3 >> 4, k0 = c0 & 15, k1 = c1 & 15, k2 = c2 & 15, k3 = c3 & 15;
                if (nr == 1) body1(xq, xs, codes, scales, OUT_ARG LUT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
                else if (nr == 2) body2(xq, xs, codes, scales, OUT_ARG LUT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
                else if (nr == 3) body3(xq, xs, codes, scales, OUT_ARG LUT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
                else body4(xq, xs, codes, scales, OUT_ARG LUT_ARG, NBLK, slot, tt, t0, t1, t2, t3, k0, k1, k2, k3);
            }
        }
        if (++tt == NT) { tt = 0; ++q; }
    }
}
