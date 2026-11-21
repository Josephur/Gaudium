// gaudium_gdn_vrec: the gated delta rule of the VERIFY rows without touching the state.
//
// gaudium_gdn_f32 sweeps the 64 KB state of every head once per row (retrieve, update, read out) and stores it per row for the rollback.
// A verify step does not need any of those states: every row's retrieval and output are linear in the state the step STARTED from, S0,
// plus rank-1 terms of the rows before it on its root path:
//     S_t   = E_t S0 + sum_{i in path(t)} e(t,i) k_i (x) d_i        E_t = prod of g over path(t),  e(t,i) = prod of g over the path after i
//     g_t a_t = E_t P_t + sum_{i in anc(t)} e(t,i) (k_i . k_t) d_i  P_t = S0^T k_t
//     d_t   = (v_t - g_t a_t) beta_t
//     o_t   = E_t R_t + sum_{i in anc(t)} e(t,i) (k_i . q_t) d_i + (k_t . q_t) d_t      R_t = S0^T q_t
// P, R ([k ; q] . S0) and the dot products ([k ; q] . [k ; q]^T) are two batched fp32 GEMMs on the MME, which cost the same for 2 rows
// or 32. What is left here is one scalar-times-vector MAC chain per (row, ancestor): no state traffic, no horizontal sums (a first
// version computed the dot products here with lane reductions: 16.6k cycles per head group at T = 16 on libtpcsim, 45 us per layer on
// the card, as slow as the kernel it replaced). The step's state update is applied once, for the accepted row only, by the commit nodes
// (gaudium_gdn_vcommit + one GEMM per layer): S' = E_a S0 + sum_{i in path(a)} e(a,i) k_i (x) d_i.
// Rows are tree nodes (row 0 = the root), parents before children.
//
//  in0 PR   f32 {128, 2T, HV, 1}: row t = P_t, row T + t = R_t
//  in1 G    f32 {2T, 2T, HV, 1}:  G[r][s] = kq_r . kq_s, kq = [k_0..k_{T-1} ; q_0..q_{T-1}]  (element {s, r, h})
//  in2 v    f32 {128, HV, T}      in3 eg f32 {64, T, 1} (decay g per head)      in4 beta f32 {64, T}
//  out0 o   f32 {128, HV, T} (what gaudium_gdn_post reads)       out1 dl f32 {128, T, HV, 1}: row t = d_t (the commit GEMM's operand)
//  scalars: T, NC, L: rows 0..NC-1 are the chain (parent = the row before), then L leaves on chain rows 0..L-1, then the rest on chain
//  rows 0.. again (TreeShape, model_dflash2.hpp): the shape is static per recipe, so no parent table is read.
//  One index-space point = one value head. NO scalar loads from global memory: on the card each costs ~0.2 us (a version that read
//  the dot products and gates as scalars ran 112 us per layer at T = 16; libtpcsim does not show it). Every scalar is taken as lane 0
//  of a vector load and broadcast with one shuffle + one dual-group move.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define BC0(x) ({ float64 _x = (x); _x = v_f32_shuffle_b(_x, P0, 0, _x); MOVDG_ALL(_x, 0, 0, 0, 0, _x); })   // lane 0 -> every lane
#define PARENT(t) ((t) < NC ? (t) - 1 : ((t) - NC < L ? (t) - NC : (t) - NC - L))

__local__ float64 DV[64];     // d rows of this head: DV[2t + half]
__local__ float64 GT[32];     // g of row t (broadcast)
__local__ float64 ET[32];     // E of row t (broadcast)

void main(tensor PR, tensor G, tensor v, tensor eg, tensor beta, tensor o, tensor dl, int T, int NC, int L)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const unsigned char pb = 0x80; const uchar256 P0 = pb;   // shuffle pattern: element 0 of the dual group's first half into every lane of the group

    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            const int p = PARENT(t);
            int5 gc = {h, t, 0, 0, 0};
            const float64 g = BC0(v_f32_ld_tnsr_b(gc, eg)), b = BC0(v_f32_ld_tnsr_b(gc, beta));
            float64 E = g; if (p >= 0) E = ET[p] * g;
            GT[t] = g; ET[t] = E;
            int5 pc = {0, t, h, 0, 0};
            float64 a0 = v_f32_ld_tnsr_b(pc, PR) * E; pc[0] = 64; float64 a1 = v_f32_ld_tnsr_b(pc, PR) * E;
            pc[1] = T + t; pc[0] = 0;
            float64 o0 = v_f32_ld_tnsr_b(pc, PR) * E; pc[0] = 64; float64 o1 = v_f32_ld_tnsr_b(pc, PR) * E;
            float64 e = g;
            for (int i = p; i >= 0; i = PARENT(i)) {   // the rows before this one on its root path
                int5 kc = {i, t, h, 0, 0};
                const float64 ck = BC0(v_f32_ld_tnsr_b(kc, G)) * e;      // k_t . k_i
                kc[1] = T + t;
                const float64 cq = BC0(v_f32_ld_tnsr_b(kc, G)) * e;      // q_t . k_i
                const float64 d0 = DV[2 * i], d1 = DV[2 * i + 1];
                a0 = v_f32_mac_b(d0, ck, a0); a1 = v_f32_mac_b(d1, ck, a1);
                o0 = v_f32_mac_b(d0, cq, o0); o1 = v_f32_mac_b(d1, cq, o1);
                e = e * GT[i];
            }
            int5 qc = {t, T + t, h, 0, 0};
            const float64 qs = BC0(v_f32_ld_tnsr_b(qc, G));              // q_t . k_t
            int5 vc = {0, h, t, 0, 0};
            const float64 v0 = v_f32_ld_tnsr_b(vc, v); vc[0] = 64; const float64 v1 = v_f32_ld_tnsr_b(vc, v);
            const float64 d0 = (v0 - a0) * b, d1 = (v1 - a1) * b;
            DV[2 * t] = d0; DV[2 * t + 1] = d1;
            o0 = v_f32_mac_b(d0, qs, o0); o1 = v_f32_mac_b(d1, qs, o1);
            v_f32_st_tnsr(vc, o, o1); vc[0] = 0; v_f32_st_tnsr(vc, o, o0);
            int5 dc = {0, t, h, 0, 0};
            v_f32_st_tnsr(dc, dl, d0); dc[0] = 64; v_f32_st_tnsr(dc, dl, d1);
        }
    }
}
