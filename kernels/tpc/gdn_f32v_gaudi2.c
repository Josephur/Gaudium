// gaudium_gdn_f32v: gaudium_gdn_f32 -DSLOT (the gated DeltaNet recurrence of the decode / verify rows, state checkpoint per row)
// with the two per-row gate scalars taken as VECTOR loads.
//
// gaudium_gdn_f32 reads the decay g and the write strength beta of every (head, row) with a scalar load from global memory. On the
// card such a load costs about 0.2 us (the figure gdn_vrec_gaudi2.c was built around), and a verify step makes two of them per head
// and row: 64 per TPC and layer at 16 rows. Here each is lane 0 of a vector load, broadcast with one shuffle and one dual-group move
// (the BC0 of gdn_vrec_gaudi2.c), and the products that took the scalar take the broadcast vector.
//
// The arithmetic is gaudium_gdn_f32's, operation for operation: a vector times a broadcast scalar is the same IEEE product per lane
// as the vector times the scalar, every MAC keeps its operands and its place in its chain. Outputs and checkpoints are bit-identical.
//
// Blob layout, passes and flags: see gdn_f32_gaudi2.c. Inputs / outputs: those of its -DSLOT build
//   (S_in {64, 256, HV, NS}, q, k, v {128, HV, T}, eg, beta {64, T}, slot {1|2} -> o {128, HV, T}, S_out {64, 256, HV}, S_ckpt {64, 256, HV, T}).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)

__local__ float64 S[256];

// NOTE: tpc-clang 2.7.33 segfaults (exit 139) when mov_dual_group_all's source-group arguments reach it through an
// inline function parameter; the group selectors must be literals at the call site, hence macros.
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define RED4(a) ({ float64 _a = (a); float64 _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a + _r; })
#define REP_GROUP(a, gs) MOVDG_ALL((a), gs, gs, gs, gs, (a))
#define BC0(x) ({ float64 _x = (x); _x = v_f32_shuffle_b(_x, P0, 0, _x); MOVDG_ALL(_x, 0, 0, 0, 0, _x); })   // lane 0 -> every lane
static inline uchar256 pat_of(int p) {
    unsigned char b = (unsigned char)((p & 7) | ((p >> 3) << 5) | 0x80);
    uchar256 r = b;
    return r;
}

void main(tensor S_in, tensor q, tensor k, tensor v, tensor eg, tensor beta, tensor slot,
          tensor o, tensor S_out, tensor S_ckpt, int T, int flags)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int h0 = idx[0], h1 = idx[0] + isz[0];
    const unsigned char pb0 = 0x80; const uchar256 P0 = pb0;   // shuffle pattern: element 0 of each dual group into its every lane
    const int write_ckpt = flags & 1, no_state_io = flags & 4, no_compute = flags & 8, no_out = flags & 16;
    const bool ck = write_ckpt != 0;
    const int5 zc = {0, 0, 0, 0, 0};
    const int slot_in = s_i32_ld_g(gen_addr(zc, slot));
    int Te = T;                                     // gated rows (prompt-lookup step): slot {2} = [slot, valid rows]; rows >= Te are skipped
    if (get_dim_size(slot, 0) >= 2) { int5 c1 = {1, 0, 0, 0, 0}; const int nv = s_i32_ld_g(gen_addr(c1, slot)); if (nv < Te) Te = nv; }

    for (int h = h0; h < h1; ++h) {
        int5 sc = {0, 0, h, slot_in, 0};
        for (int n = 0; n < 256 && !no_state_io; n += 8) {
            sc[1] = n;     S[n]     = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 1; S[n + 1] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 2; S[n + 2] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 3; S[n + 3] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 4; S[n + 4] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 5; S[n + 5] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 6; S[n + 6] = v_f32_ld_tnsr_b(sc, S_in);
            sc[1] = n + 7; S[n + 7] = v_f32_ld_tnsr_b(sc, S_in);
        }

        for (int t = 0; t < Te && !no_compute; ++t) {
            int5 hc = {h, t, 0, 0, 0};
            // g and beta of (head, row): lane 0 of a vector load each, broadcast (no scalar global load)
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
            // rows are processed in groups of 8: load 8 VLM rows, compute, store 8. No VLM load follows a VLM store
            // inside a group, so the compiler cannot serialise the rows on a (false) memory dependency.
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
            // ---- per-token checkpoint: a separate tight store loop (stores inside the compute loop cost 14 us/layer) ----
            if (ck) {
                int5 cc = {0, 0, h, t, 0};
                for (int n = 0; n < 256; n += 8) {
                    cc[1] = n;     v_f32_st_tnsr(cc, S_ckpt, S[n]);
                    cc[1] = n + 1; v_f32_st_tnsr(cc, S_ckpt, S[n + 1]);
                    cc[1] = n + 2; v_f32_st_tnsr(cc, S_ckpt, S[n + 2]);
                    cc[1] = n + 3; v_f32_st_tnsr(cc, S_ckpt, S[n + 3]);
                    cc[1] = n + 4; v_f32_st_tnsr(cc, S_ckpt, S[n + 4]);
                    cc[1] = n + 5; v_f32_st_tnsr(cc, S_ckpt, S[n + 5]);
                    cc[1] = n + 6; v_f32_st_tnsr(cc, S_ckpt, S[n + 6]);
                    cc[1] = n + 7; v_f32_st_tnsr(cc, S_ckpt, S[n + 7]);
                }
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

        sc[3] = 0;
        for (int n = 0; n < 256 && !no_state_io && !no_out; n += 8) {
            sc[1] = n;     v_f32_st_tnsr(sc, S_out, S[n]);
            sc[1] = n + 1; v_f32_st_tnsr(sc, S_out, S[n + 1]);
            sc[1] = n + 2; v_f32_st_tnsr(sc, S_out, S[n + 2]);
            sc[1] = n + 3; v_f32_st_tnsr(sc, S_out, S[n + 3]);
            sc[1] = n + 4; v_f32_st_tnsr(sc, S_out, S[n + 4]);
            sc[1] = n + 5; v_f32_st_tnsr(sc, S_out, S[n + 5]);
            sc[1] = n + 6; v_f32_st_tnsr(sc, S_out, S[n + 6]);
            sc[1] = n + 7; v_f32_st_tnsr(sc, S_out, S[n + 7]);
        }
    }
}
