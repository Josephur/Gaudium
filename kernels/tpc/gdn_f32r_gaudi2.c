// gaudium_gdn_f32r: the gated DeltaNet recurrence of a speculative verify step WITHOUT per-row state checkpoints.
//
// The row arithmetic is gaudium_gdn_f32's, operation for operation (same blob layout of the 128x128 fp32 state in VLM, same two
// passes per row, same MAC order): a row's output and the state it leaves are bit-identical to that kernel's. The per-row gate
// scalars are taken as broadcast vector loads, as in gdn_f32v_gaudi2.c (a scalar load from global memory costs about 0.2 us).
//
// What differs is how the state crosses a step. gaudium_gdn_f32 -DSLOT stores the state after every row (3.1 MB per row and layer at
// 48 heads) and the next step loads the accepted row's checkpoint. Here a step stores, per row, only what the row's update was
// built from: its k (already a tensor), its decay g (already a tensor) and its 8 delta vectors d (2 KB per head and row). The next
// step loads the state the previous step STARTED from and applies the accepted rows' updates to it again,
//     S[n] = g * S[n] + k (x) d        r = mac(kb, d, r * g), the row kernel's own expression on the row kernel's own operands,
// which gives the accepted row's state to the bit; it stores that state once (S_out: the state this step starts from, and the one
// the next step's replay starts from) and then runs its own rows in VLM. Per step and layer: one state load and one state store,
// whatever the number of verify rows, against one load and T stores.
//
// d is kept as the 8 full vectors the update used, not as 128 values: the group reduction RED4 leaves each dual group its own
// summation order, so the 4 groups of a d vector are not copies of one another in the last bit.
//
//  in0 S_in {64, 256, HV}     the state the previous step started from (first step of a request: the prefill's state)
//  in1 q, in2 k, in3 v {128, HV, T}   in4 eg, in5 beta {64, T}                         this step's rows
//  in6 pk {128, HV, TP}   in7 pd {64, 8, HV, TP}   in8 peg {64, TP}                    the previous step's rows (k, d, g)
//  in9 slot {1}   in10 cursor {1}   in11 cur0 {1}     rows to replay = slot[0] + 1 (the previous step's accepted row + 1), or 0
//                                                      when cursor[0] == cur0[0]: the request's first step has no previous step
//  out0 o {128, HV, T}   out1 S_out {64, 256, HV}   out2 dr {64, 8, HV, T}             this step's d vectors (the next step's pd)
//  flags: bit3 skip the rows' compute, bit5 skip the replay (both timing only).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)

__local__ float64 S[256];

// tpc-clang segfaults when mov_dual_group_all's source-group arguments reach it through an inline function parameter; the group
// selectors must be literals at the call site, hence macros (as in gdn_f32_gaudi2.c).
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define RED4(a) ({ float64 _a = (a); float64 _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a + _r; })
#define REP_GROUP(a, gs) MOVDG_ALL((a), gs, gs, gs, gs, (a))
#define BC0(x) ({ float64 _x = (x); _x = v_f32_shuffle_b(_x, P0, 0, _x); MOVDG_ALL(_x, 0, 0, 0, 0, _x); })   // lane 0 -> every lane
static inline uchar256 pat_of(int p) {
    unsigned char b = (unsigned char)((p & 7) | ((p >> 3) << 5) | 0x80);
    uchar256 r = b;
    return r;
}

void main(tensor S_in, tensor q, tensor k, tensor v, tensor eg, tensor beta,
          tensor pk, tensor pd, tensor peg, tensor slot, tensor cursor, tensor cur0,
          tensor o, tensor S_out, tensor dr, int T, int flags)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int h0 = idx[0], h1 = idx[0] + isz[0];
    const unsigned char pb0 = 0x80; const uchar256 P0 = pb0;   // shuffle pattern: element 0 of each dual group into its every lane
    const int no_compute = flags & 8, no_replay = flags & 32;
    const int5 zc = {0, 0, 0, 0, 0};
    const int a_in = s_i32_ld_g(gen_addr(zc, slot));
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    const int first = s_i32_ld_g(gen_addr(zc, cur0));
    const int NR = (cur == first || no_replay) ? 0 : a_in + 1;

    for (int h = h0; h < h1; ++h) {
        int5 sc = {0, 0, h, 0, 0};
        for (int n = 0; n < 256; n += 8) {
            sc[1] = n;     S[n]     = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 1; S[n + 1] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 2; S[n + 2] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 3; S[n + 3] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 4; S[n + 4] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 5; S[n + 5] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 6; S[n + 6] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 7; S[n + 7] = v_f32_ld_tnsr_b(sc, S_in);
        }

        // ---- replay: the previous step's accepted rows, in row order ----
        for (int t = 0; t < NR; ++t) {
            int5 hc = {h, t, 0, 0, 0};
            const float64 g = BC0(v_f32_ld_tnsr_b(hc, peg));   // lane 0 of a vector load, broadcast (no scalar global load)
            int5 vc = {0, h, t, 0, 0};
            float64 k0v = v_f32_ld_tnsr_b(vc, pk);
            vc[0] = 64;
            float64 k1v = v_f32_ld_tnsr_b(vc, pk);
            int5 dc = {0, 0, h, t, 0};
            float64 d0 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 1;
            float64 d1 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 2;
            float64 d2 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 3;
            float64 d3 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 4;
            float64 d4 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 5;
            float64 d5 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 6;
            float64 d6 = v_f32_ld_tnsr_b(dc, pd); dc[1] = 7;
            float64 d7 = v_f32_ld_tnsr_b(dc, pd);
            for (int p = 0; p < 16; ++p) {
                uchar256 pat = pat_of(p);
                float64 kb0 = v_f32_shuffle_b(k0v, pat, 0, k0v), kb1 = v_f32_shuffle_b(k1v, pat, 0, k1v);
                int n = p * 16;
                float64 r0 = S[n], r1 = S[n + 1], r2 = S[n + 2], r3 = S[n + 3], r4 = S[n + 4], r5 = S[n + 5], r6 = S[n + 6], r7 = S[n + 7];
                r0 = v_f32_mac_b(kb0, d0, r0 * g); r1 = v_f32_mac_b(kb1, d0, r1 * g);
                r2 = v_f32_mac_b(kb0, d1, r2 * g); r3 = v_f32_mac_b(kb1, d1, r3 * g);
                r4 = v_f32_mac_b(kb0, d2, r4 * g); r5 = v_f32_mac_b(kb1, d2, r5 * g);
                r6 = v_f32_mac_b(kb0, d3, r6 * g); r7 = v_f32_mac_b(kb1, d3, r7 * g);
                S[n] = r0; S[n + 1] = r1; S[n + 2] = r2; S[n + 3] = r3; S[n + 4] = r4; S[n + 5] = r5; S[n + 6] = r6; S[n + 7] = r7;
                float64 u0 = S[n + 8], u1 = S[n + 9], u2 = S[n + 10], u3 = S[n + 11], u4 = S[n + 12], u5 = S[n + 13], u6 = S[n + 14], u7 = S[n + 15];
                u0 = v_f32_mac_b(kb0, d4, u0 * g); u1 = v_f32_mac_b(kb1, d4, u1 * g);
                u2 = v_f32_mac_b(kb0, d5, u2 * g); u3 = v_f32_mac_b(kb1, d5, u3 * g);
                u4 = v_f32_mac_b(kb0, d6, u4 * g); u5 = v_f32_mac_b(kb1, d6, u5 * g);
                u6 = v_f32_mac_b(kb0, d7, u6 * g); u7 = v_f32_mac_b(kb1, d7, u7 * g);
                S[n + 8] = u0; S[n + 9] = u1; S[n + 10] = u2; S[n + 11] = u3; S[n + 12] = u4; S[n + 13] = u5; S[n + 14] = u6; S[n + 15] = u7;
            }
        }

        // ---- commit: the state this step starts from (nothing below reads S_out) ----
        for (int n = 0; n < 256; n += 8) {
            sc[1] = n;     v_f32_st_tnsr(sc, S_out, S[n]);
            sc[1] = n + 1; v_f32_st_tnsr(sc, S_out, S[n + 1]);
            sc[1] = n + 2; v_f32_st_tnsr(sc, S_out, S[n + 2]);
            sc[1] = n + 3; v_f32_st_tnsr(sc, S_out, S[n + 3]);
            sc[1] = n + 4; v_f32_st_tnsr(sc, S_out, S[n + 4]);
            sc[1] = n + 5; v_f32_st_tnsr(sc, S_out, S[n + 5]);
            sc[1] = n + 6; v_f32_st_tnsr(sc, S_out, S[n + 6]);
            sc[1] = n + 7; v_f32_st_tnsr(sc, S_out, S[n + 7]);
        }

        // ---- this step's rows (gaudium_gdn_f32's row, with the d vectors stored in place of the state) ----
        for (int t = 0; t < T && !no_compute; ++t) {
            int5 hc = {h, t, 0, 0, 0};
            const float64 g = BC0(v_f32_ld_tnsr_b(hc, eg));
            const float64 b = BC0(v_f32_ld_tnsr_b(hc, beta));
            int5 vc = {0, h, t, 0, 0};
            float64 v0 = v_f32_ld_tnsr_b(vc, v);  float64 k0v = v_f32_ld_tnsr_b(vc, k);  float64 q0v = v_f32_ld_tnsr_b(vc, q);
            vc[0] = 64;
            float64 v1 = v_f32_ld_tnsr_b(vc, v);  float64 k1v = v_f32_ld_tnsr_b(vc, k);  float64 q1v = v_f32_ld_tnsr_b(vc, q);

            // ---- pass 1: acc[vb] = sum_c k[c] S[c][v]  (per dual group partials) ----
            float64 a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
            for (int p = 0; p < 16; ++p) {
                uchar256 pat = pat_of(p);
                float64 kb0 = v_f32_shuffle_b(k0v, pat, 0, k0v);
                float64 kb1 = v_f32_shuffle_b(k1v, pat, 0, k1v);
                int n = p * 16;
                a0 = v_f32_mac_b(S[n],      kb0, a0); a0 = v_f32_mac_b(S[n + 1],  kb1, a0);
                a1 = v_f32_mac_b(S[n + 2],  kb0, a1); a1 = v_f32_mac_b(S[n + 3],  kb1, a1);
                a2 = v_f32_mac_b(S[n + 4],  kb0, a2); a2 = v_f32_mac_b(S[n + 5],  kb1, a2);
                a3 = v_f32_mac_b(S[n + 6],  kb0, a3); a3 = v_f32_mac_b(S[n + 7],  kb1, a3);
                a4 = v_f32_mac_b(S[n + 8],  kb0, a4); a4 = v_f32_mac_b(S[n + 9],  kb1, a4);
                a5 = v_f32_mac_b(S[n + 10], kb0, a5); a5 = v_f32_mac_b(S[n + 11], kb1, a5);
                a6 = v_f32_mac_b(S[n + 12], kb0, a6); a6 = v_f32_mac_b(S[n + 13], kb1, a6);
                a7 = v_f32_mac_b(S[n + 14], kb0, a7); a7 = v_f32_mac_b(S[n + 15], kb1, a7);
            }
            // ---- d[vb] = (v - g*a) * beta, replicated across the 4 dual groups ----
            float64 d0 = (REP_GROUP(v0, 0) - RED4(a0) * g) * b;
            float64 d1 = (REP_GROUP(v0, 1) - RED4(a1) * g) * b;
            float64 d2 = (REP_GROUP(v0, 2) - RED4(a2) * g) * b;
            float64 d3 = (REP_GROUP(v0, 3) - RED4(a3) * g) * b;
            float64 d4 = (REP_GROUP(v1, 0) - RED4(a4) * g) * b;
            float64 d5 = (REP_GROUP(v1, 1) - RED4(a5) * g) * b;
            float64 d6 = (REP_GROUP(v1, 2) - RED4(a6) * g) * b;
            float64 d7 = (REP_GROUP(v1, 3) - RED4(a7) * g) * b;

            // ---- pass 2: S = g*S + k (x) d ; oacc[vb] += S * qb ----
            float64 o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f, o4 = 0.f, o5 = 0.f, o6 = 0.f, o7 = 0.f;
            for (int p = 0; p < 16; ++p) {
                uchar256 pat = pat_of(p);
                float64 kb0 = v_f32_shuffle_b(k0v, pat, 0, k0v), kb1 = v_f32_shuffle_b(k1v, pat, 0, k1v);
                float64 qb0 = v_f32_shuffle_b(q0v, pat, 0, q0v), qb1 = v_f32_shuffle_b(q1v, pat, 0, q1v);
                int n = p * 16;
                float64 r0 = S[n], r1 = S[n + 1], r2 = S[n + 2], r3 = S[n + 3], r4 = S[n + 4], r5 = S[n + 5], r6 = S[n + 6], r7 = S[n + 7];
                r0 = v_f32_mac_b(kb0, d0, r0 * g); r1 = v_f32_mac_b(kb1, d0, r1 * g);
                r2 = v_f32_mac_b(kb0, d1, r2 * g); r3 = v_f32_mac_b(kb1, d1, r3 * g);
                r4 = v_f32_mac_b(kb0, d2, r4 * g); r5 = v_f32_mac_b(kb1, d2, r5 * g);
                r6 = v_f32_mac_b(kb0, d3, r6 * g); r7 = v_f32_mac_b(kb1, d3, r7 * g);
                o0 = v_f32_mac_b(r0, qb0, o0); o0 = v_f32_mac_b(r1, qb1, o0);
                o1 = v_f32_mac_b(r2, qb0, o1); o1 = v_f32_mac_b(r3, qb1, o1);
                o2 = v_f32_mac_b(r4, qb0, o2); o2 = v_f32_mac_b(r5, qb1, o2);
                o3 = v_f32_mac_b(r6, qb0, o3); o3 = v_f32_mac_b(r7, qb1, o3);
                S[n] = r0; S[n + 1] = r1; S[n + 2] = r2; S[n + 3] = r3; S[n + 4] = r4; S[n + 5] = r5; S[n + 6] = r6; S[n + 7] = r7;
                float64 u0 = S[n + 8], u1 = S[n + 9], u2 = S[n + 10], u3 = S[n + 11], u4 = S[n + 12], u5 = S[n + 13], u6 = S[n + 14], u7 = S[n + 15];
                u0 = v_f32_mac_b(kb0, d4, u0 * g); u1 = v_f32_mac_b(kb1, d4, u1 * g);
                u2 = v_f32_mac_b(kb0, d5, u2 * g); u3 = v_f32_mac_b(kb1, d5, u3 * g);
                u4 = v_f32_mac_b(kb0, d6, u4 * g); u5 = v_f32_mac_b(kb1, d6, u5 * g);
                u6 = v_f32_mac_b(kb0, d7, u6 * g); u7 = v_f32_mac_b(kb1, d7, u7 * g);
                o4 = v_f32_mac_b(u0, qb0, o4); o4 = v_f32_mac_b(u1, qb1, o4);
                o5 = v_f32_mac_b(u2, qb0, o5); o5 = v_f32_mac_b(u3, qb1, o5);
                o6 = v_f32_mac_b(u4, qb0, o6); o6 = v_f32_mac_b(u5, qb1, o6);
                o7 = v_f32_mac_b(u6, qb0, o7); o7 = v_f32_mac_b(u7, qb1, o7);
                S[n + 8] = u0; S[n + 9] = u1; S[n + 10] = u2; S[n + 11] = u3; S[n + 12] = u4; S[n + 13] = u5; S[n + 14] = u6; S[n + 15] = u7;
            }
            // ---- the row's update, for the next step's replay ----
            {
                int5 dc = {0, 0, h, t, 0};
                v_f32_st_tnsr(dc, dr, d0); dc[1] = 1;
                v_f32_st_tnsr(dc, dr, d1); dc[1] = 2;
                v_f32_st_tnsr(dc, dr, d2); dc[1] = 3;
                v_f32_st_tnsr(dc, dr, d3); dc[1] = 4;
                v_f32_st_tnsr(dc, dr, d4); dc[1] = 5;
                v_f32_st_tnsr(dc, dr, d5); dc[1] = 6;
                v_f32_st_tnsr(dc, dr, d6); dc[1] = 7;
                v_f32_st_tnsr(dc, dr, d7);
            }
            // ---- o: reduce each oacc over the 4 dual groups, assemble 2 contiguous vectors ----
            float64 r0 = RED4(o0), r1 = RED4(o1), r2 = RED4(o2), r3 = RED4(o3), r4 = RED4(o4), r5 = RED4(o5), r6 = RED4(o6), r7 = RED4(o7);
            float64 out0 = v_f32_mov_dual_group_all_b(r1, 0xFFFFFFFF, 0, 1, 0, 0, MkWrA(0b00, 0b11, 0b00, 0b00), r0);
            out0 = v_f32_mov_dual_group_all_b(r2, 0xFFFFFFFF, 0, 0, 2, 0, MkWrA(0b00, 0b00, 0b11, 0b00), out0);
            out0 = v_f32_mov_dual_group_all_b(r3, 0xFFFFFFFF, 0, 0, 0, 3, MkWrA(0b00, 0b00, 0b00, 0b11), out0);
            float64 out1 = v_f32_mov_dual_group_all_b(r5, 0xFFFFFFFF, 0, 1, 0, 0, MkWrA(0b00, 0b11, 0b00, 0b00), r4);
            out1 = v_f32_mov_dual_group_all_b(r6, 0xFFFFFFFF, 0, 0, 2, 0, MkWrA(0b00, 0b00, 0b11, 0b00), out1);
            out1 = v_f32_mov_dual_group_all_b(r7, 0xFFFFFFFF, 0, 0, 0, 3, MkWrA(0b00, 0b00, 0b00, 0b11), out1);
            int5 oc = {0, h, t, 0, 0};
            v_f32_st_tnsr(oc, o, out0);
            oc[0] = 64;
            v_f32_st_tnsr(oc, o, out1);
        }
    }
}
