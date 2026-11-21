// gaudium_gdn_solve (block form): the diagonal 16x16 blocks of (I + A) are inverted on the TPC by forward substitution;
// the MME finishes (I + A)^-1 = (I + D0 L)^-1 D0 with the exact 3-term Neumann series (D0 L is block-strictly-lower with
// NB = C/16 blocks, so (D0 L)^NB = 0), see gdn_chunk.hpp. One index-space point per (value head, chunk), C <= 64.
//     D0  = blockdiag((I + A_bb)^-1)           D = D0 diag(beta)        Mu = (I + D0 L)^-1 D,  L = A - blockdiag(A)
//  in0 An {C, C, HV, NCH} f32 (-A, strictly lower)    in1 gb2 {C, 2, HV, NCH} (row 0 Gamma[u], row 1 beta[u])
//  out0 D0n {C, C, HV, NCH} = -D0      out1 Dn = -D       out2 Dg = D diag(Gamma)      out3 Lp = +L      out4 Id = identity
//  MME (per layer, batched): Y = D0n . Lp ; N1 = Id + D0n . Lp ; N2 = Id + Y . N1 ; N3 = Id + Y . N2 ; Mun = N3 . Dn ; Mw = N3 . Dg
//  scalars: C, HV.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REP_GROUP(a, gs) MOVDG_ALL((a), gs, gs, gs, gs, (a))
#define REP_ANY(dst, a, gs) do { switch (gs) { case 0: dst = REP_GROUP(a, 0); break; case 1: dst = REP_GROUP(a, 1); break; case 2: dst = REP_GROUP(a, 2); break; default: dst = REP_GROUP(a, 3); } } while (0)

__local__ float64 Z[64];       // rows of D0 (block-diagonal: row t only has lanes of its own block)

static inline uchar256 pat_of(int e) { unsigned char b = (unsigned char)((e & 7) | ((e >> 3) << 5) | 0x80); uchar256 r = b; return r; }

void main(tensor An, tensor gb2, tensor D0n, tensor Dn, tensor Dg, tensor Lp, tensor Id, int C, int HV)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 lg = lane >> 4;
    const float64 zero = 0.f;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int h = i % HV, j = i / HV;
        for (int n = 0; n < 64; n += 4) { Z[n] = zero; Z[n + 1] = zero; Z[n + 2] = zero; Z[n + 3] = zero; }
        int5 gc = {0, 0, h, j, 0};
        const float64 Gam = v_f32_ld_tnsr_b(gc, gb2);
        gc[1] = 1;
        const float64 B = v_f32_ld_tnsr_b(gc, gb2);
        for (int tg = 0; tg < (C + 15) / 16; ++tg) {
            float64 Br; REP_ANY(Br, B, tg);
#pragma unroll 2
            for (int te = 0; te < 16; ++te) {
                const int t = 16 * tg + te;
                int5 cc = {0, t, h, j, 0};
                const float64 At = v_f32_ld_tnsr_b(cc, An);
                const float64 bt = v_f32_shuffle_b(Br, pat_of(te), 0, Br);
                const float64 inblk = v_f32_sel_eq_i32_b(lg, tg, 1.f, zero);          // lanes u of row t's own block
                const float64 one_t = v_f32_sel_eq_i32_b(lane, t, 1.f, zero);
                v_f32_st_tnsr(cc, Lp, -At * (1.f - inblk));                           // +A outside the diagonal blocks
                v_f32_st_tnsr(cc, Id, one_t);
                // Z[t] = e_t + sum_{u<t, same block} An[t][u] Z[u]  (An is zero at u >= t and Z rows of this block above t are set)
                float64 rep; REP_ANY(rep, At, tg);
                float64 a0 = one_t, a1 = zero, a2 = zero, a3 = zero; const int ub = 16 * tg;
#define STEP(e, acc_) acc_ = v_f32_mac_b(v_f32_shuffle_b(rep, pat_of(e), 0, rep), Z[ub + (e)], acc_)
                STEP(0, a0); STEP(1, a1); STEP(2, a2); STEP(3, a3); STEP(4, a0); STEP(5, a1); STEP(6, a2); STEP(7, a3);
                STEP(8, a0); STEP(9, a1); STEP(10, a2); STEP(11, a3); STEP(12, a0); STEP(13, a1); STEP(14, a2); STEP(15, a3);
#undef STEP
                const float64 z = ((a0 + a1) + (a2 + a3)) * inblk;
                Z[t] = z;
                v_f32_st_tnsr(cc, D0n, -z);
                const float64 d = z * B;                                              // D[t][u] = D0[t][u] beta[u]
                v_f32_st_tnsr(cc, Dn, -d);
                v_f32_st_tnsr(cc, Dg, d * Gam);
            }
        }
    }
}
