// moe_w4_mv_body.h: the row-group body of moe_w4_mv_gaudi2.c, instantiated for N = 1..4 rows (included once per N; tpc-clang
// 2.7.33 crashes on a template here, signal 11).
#undef W4_NACC
#define W4_NACC (N == 1 ? W4_NACC1 : (N == 2 ? 2 : 1))
static inline void W4_CAT(body, N)(tensor xq, tensor xs, tensor codes, tensor scales, OUT_PAR LUT_PAR, int NBLK, int slot, int tt,
                        int t0, int t1, int t2, int t3, int k0, int k1, int k2, int k3)
{
    const int Tr[4] = {t0, t1, t2, t3}, Kr[4] = {k0, k1, k2, k3};
    float64 A[N][4];
    #pragma unroll
    for (int i = 0; i < N; ++i) { A[i][0] = 0.0f; A[i][1] = 0.0f; A[i][2] = 0.0f; A[i][3] = 0.0f; }
    char256 xg[N], xh[N];
    for (int b = 0; b < NBLK; ++b) {
        #pragma unroll
        for (int i = 0; i < N; ++i) { xg[i] = v_i8_ld_tnsr_b(XCO(XVEC(b, 0), Kr[i], Tr[i]), xq); xh[i] = v_i8_ld_tnsr_b(XCO(XVEC(b, 8), Kr[i], Tr[i]), xq); }
        int5 sc = {0, b, tt, slot, 0};
        const ushort128 s0 = v_u16_ld_tnsr_b(sc, scales); sc[0] = 128;
        const ushort128 s1 = v_u16_ld_tnsr_b(sc, scales);
        const float128 w01 = v_convert_f16_to_f32_all_b(*(half128*)&s0), w23 = v_convert_f16_to_f32_all_b(*(half128*)&s1);
        // the block's 16 column pairs fully unrolled: loads first, NACC independent int32 accumulators per row (MAC latency), summed
        // exactly at the block end
        int256 acc[N][W4_NACC];
        #pragma unroll
        for (int i = 0; i < N; ++i)
            #pragma unroll
            for (int a = 0; a < W4_NACC; ++a) { acc[i][a].v1 = 0; acc[i][a].v2 = 0; acc[i][a].v3 = 0; acc[i][a].v4 = 0; }
#if W4_Q4_UNROLL == 4 && N == 1
        #pragma unroll
#elif W4_Q4_UNROLL == 2 && N <= W4_Q4_UNROLL_N
        #pragma unroll 2
#else
        #pragma unroll 1   // fully unrolled, tpc-clang does not finish compiling this body (stopped after 10 min, 2026-11-14)
#endif
        for (int q4 = 0; q4 < 4; ++q4) {
            char256 xs4[N];
            #pragma unroll
            for (int i = 0; i < N; ++i) xs4[i] = q4 < 2 ? xg[i] : xh[i];
            char256 lv[8];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int cp = 16 * b + 4 * q4 + q;
#ifdef W8
                int5 cc = {0, 2 * cp, tt, slot, 0};
                lv[2 * q] = v_i8_ld_tnsr_b(cc, codes); cc[1] += 1;
                lv[2 * q + 1] = v_i8_ld_tnsr_b(cc, codes);
#else
                int5 cc = {0, cp, tt, slot, 0};
                const uchar256 u0 = v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8); cc[0] = 128;
                const uchar256 u1 = v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8);
                // income = the (dead) index register: every lane is enabled, and a tied income other than a dead value costs a mov
                const uchar256 i0 = u0 | 0x80, i1 = u1 | 0x80;
                lv[2 * q] = v_i8_shuffle_b(LUT, i0, 0, *(const char256*)&i0); lv[2 * q + 1] = v_i8_shuffle_b(LUT, i1, 0, *(const char256*)&i1);
#endif
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int pp = 4 * q4 + q;
                #pragma unroll
                for (int i = 0; i < N; ++i) {
                    const char256 xv = xs4[i];
                    const uchar256 p0 = PAT[XIDX(pp)], p1 = PAT[XIDX2(pp)];   // reloaded per row (LD slot): the dead pattern is the income
                    const char256 b0 = v_i8_shuffle_b(xv, p0, 0, *(const char256*)&p0), b1 = v_i8_shuffle_b(xv, p1, 0, *(const char256*)&p1);
                    // not v_i8_mac_x2_b: wrong sums in this loop (libtpcsim, moe-w4l.md 2.4)
                    acc[i][q % W4_NACC] = v_i8_mac_b(lv[2 * q], b0, acc[i][q % W4_NACC]);
                    acc[i][q % W4_NACC] = v_i8_mac_b(lv[2 * q + 1], b1, acc[i][q % W4_NACC]);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < N; ++i)
            #pragma unroll
            for (int a = 1; a < W4_NACC; ++a) { acc[i][0].v1 += acc[i][a].v1; acc[i][0].v2 += acc[i][a].v2; acc[i][0].v3 += acc[i][a].v3; acc[i][0].v4 += acc[i][a].v4; }
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const float64 dx = v_f32_ld_tnsr_b(XCO(b, Kr[i], Tr[i]), xs);
            float64 s, p;
            s = w01.v1 * dx; p = v_convert_i32_to_f32_b(acc[i][0].v1, SW_RHNE) * s; A[i][0] = A[i][0] + p;
            s = w01.v2 * dx; p = v_convert_i32_to_f32_b(acc[i][0].v2, SW_RHNE) * s; A[i][1] = A[i][1] + p;
            s = w23.v1 * dx; p = v_convert_i32_to_f32_b(acc[i][0].v3, SW_RHNE) * s; A[i][2] = A[i][2] + p;
            s = w23.v2 * dx; p = v_convert_i32_to_f32_b(acc[i][0].v4, SW_RHNE) * s; A[i][3] = A[i][3] + p;
        }
    }
#if defined(GU) && defined(GU_RAW)   // debug build (tests/sim): the raw gate/up accumulators into hs {64, 4 NT, K, T}
    #pragma unroll
    for (int i = 0; i < N; ++i) { v_f32_st_tnsr(OCO(4 * tt, Kr[i], Tr[i]), hs, A[i][0]); v_f32_st_tnsr(OCO(4 * tt + 1, Kr[i], Tr[i]), hs, A[i][1]);
        v_f32_st_tnsr(OCO(4 * tt + 2, Kr[i], Tr[i]), hs, A[i][2]); v_f32_st_tnsr(OCO(4 * tt + 3, Kr[i], Tr[i]), hs, A[i][3]); }
#elif defined(GU)
    const uchar256 P8 = xor_pat16(8), P4 = xor_pat16(4), P2 = xor_pat16(2), P1 = xor_pat16(1);
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float64 h = SWIGLU(A[i][hh], A[i][2 + hh]);                    // channels 128 tt + 64 hh + lane
            float64 a = DGMAX(v_f32_abs_b(h), P8, P4, P2, P1);
            a = v_f32_max_b(a, MOVDG_F(a, 1, 0, 3, 2));                          // lanes 0..31 / 32..63 = the two blocks of 32 channels
            float64 q, d;
            Q8(h, a, q, d);
            const int blk = 4 * tt + 2 * hh;
            v_f32_st_tnsr(OCO(blk, Kr[i], Tr[i]), hs, MOVDG_F(d, 0, 0, 0, 0));
            v_f32_st_tnsr(OCO(blk + 1, Kr[i], Tr[i]), hs, MOVDG_F(d, 2, 2, 2, 2));
            float256 cq; cq.v1 = q; cq.v2 = q; cq.v3 = q; cq.v4 = q;
            const char256 C = v_convert_f32_to_i8_all_b(cq, SW_RHNE);       // byte 4 l + j = q[l]: dual group s holds channels 16 s .. 16 s + 15
            const int v0 = 8 * tt + 4 * hh;
            v_i8_st_tnsr(OCO(v0, Kr[i], Tr[i]), hq, MOVDG_B(C, 0, 0, 0, 0)); v_i8_st_tnsr(OCO(v0 + 1, Kr[i], Tr[i]), hq, MOVDG_B(C, 1, 1, 1, 1));
            v_i8_st_tnsr(OCO(v0 + 2, Kr[i], Tr[i]), hq, MOVDG_B(C, 2, 2, 2, 2)); v_i8_st_tnsr(OCO(v0 + 3, Kr[i], Tr[i]), hq, MOVDG_B(C, 3, 3, 3, 3));
        }
    }
#else
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int5 oc = {256 * tt, Kr[i], Tr[i], 0, 0};
        v_f32_st_tnsr(oc, y, A[i][0]); oc[0] += 64; v_f32_st_tnsr(oc, y, A[i][1]); oc[0] += 64;
        v_f32_st_tnsr(oc, y, A[i][2]); oc[0] += 64; v_f32_st_tnsr(oc, y, A[i][3]);
    }
#endif
}

