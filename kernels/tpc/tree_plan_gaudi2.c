// gaudium_tree_plan: plan this step's draft tree on the device from the drafter's own probabilities.
//  A static recipe verifies N rows; WHICH nodes fill them is decided here, per step, with no host round trip. Candidates are the
//  spine nodes j = 1..Kd (value R_j = prod_{i<=j} p0_i, the drafter's probability that the spine survives to depth j) and the leaves
//  (j, r), r = 1..KT-1 (value R_{j-1} p_r,j: the probability that the spine reaches j-1 and the rank-r candidate is the target's token
//  at depth j - disjoint from the spine continuing). Greedy: the N-1 eligible candidates of highest value (spine j needs spine j-1,
//  a leaf at depth j needs spine j-1; ties: spine first, lower depth, lower rank). Output = a canonical caterpillar tree (spine
//  rows 1..D', then the leaves by (depth, rank)) that the tree verify kernels support: leaves hang off spine rows, have no children.
//  in0 cprob f32 {KT, CM}: cprob[r][j] = drafter probability of the rank-r candidate at depth j (gaudium_tree_cand of the previous step)
//  out0 parent, out1 depth, out2 rank, out3 anc int32 {N}      scalar Kd (depths with candidates)     one TPC, scalar
__local__ float VAL[64];
__local__ int VJ[64];
__local__ int VR[64];
__local__ int SEL[64];
__local__ float RR[17];
void main(tensor cprob, tensor parent, tensor depth, tensor rank, tensor anc, int Kd)
{
    int5 c = {0, 0, 0, 0, 0};
    const int KT = get_dim_size(cprob, 0), N = get_dim_size(parent, 0);
    int nc = 0; RR[0] = 1.0f;
    for (int j = 1; j <= Kd; ++j) {
        int5 pc = {0, j, 0, 0, 0}; const float p0 = s_f32_ld_g(gen_addr(pc, cprob));
        VAL[nc] = RR[j - 1] * p0; VJ[nc] = j; VR[nc] = 0; SEL[nc] = 0; ++nc;
        for (int r = 1; r < KT; ++r) { pc[0] = r; const float pr = s_f32_ld_g(gen_addr(pc, cprob)); VAL[nc] = RR[j - 1] * pr; VJ[nc] = j; VR[nc] = r; SEL[nc] = 0; ++nc; }
        RR[j] = RR[j - 1] * p0;
    }
    int spine = 0, picked = 0;
    for (int it = 0; it < N - 1; ++it) {
        int best = -1; float bv = -1.0f;
        for (int k = 0; k < nc; ++k) {
            if (SEL[k]) continue;
            const int j = VJ[k], r = VR[k];
            const bool ok = (r == 0) ? (j == spine + 1) : (j <= spine + 1);
            if (!ok) continue;
            if (VAL[k] > bv) { bv = VAL[k]; best = k; }   // candidates are visited spine-first, depth-ascending, rank-ascending: '>' keeps that order on ties
        }
        if (best < 0) break;
        SEL[best] = 1; ++picked; if (VR[best] == 0) spine = VJ[best];
    }
    // emit: root, spine rows 1..spine, then the selected leaves by (depth, rank) (= candidate order)
    int row = 0;
    c[0] = 0; s_i32_st_g(gen_addr(c, parent), -1); s_i32_st_g(gen_addr(c, depth), 0); s_i32_st_g(gen_addr(c, rank), 0); s_i32_st_g(gen_addr(c, anc), 1);
    for (int j = 1; j <= spine; ++j) { c[0] = j; s_i32_st_g(gen_addr(c, parent), j - 1); s_i32_st_g(gen_addr(c, depth), j); s_i32_st_g(gen_addr(c, rank), 0);
        s_i32_st_g(gen_addr(c, anc), (int)((1u << (j + 1)) - 1u)); }
    row = spine + 1;
    for (int k = 0; k < nc; ++k) {
        if (!SEL[k] || VR[k] == 0) continue;
        const int j = VJ[k];
        c[0] = row; s_i32_st_g(gen_addr(c, parent), j - 1); s_i32_st_g(gen_addr(c, depth), j); s_i32_st_g(gen_addr(c, rank), VR[k]);
        s_i32_st_g(gen_addr(c, anc), (int)(((1u << j) - 1u) | (1u << row))); ++row;
    }
    // fewer eligible candidates than rows (Kd too small for N): pad with rank-0 duplicates of the root's children that can never match
    // (parent = root, rank = KT-1 placeholder at depth 1 is still a valid leaf; the host checks N - 1 <= KT * Kd so this never runs)
    for (; row < N; ++row) { c[0] = row; s_i32_st_g(gen_addr(c, parent), 0); s_i32_st_g(gen_addr(c, depth), 1); s_i32_st_g(gen_addr(c, rank), KT - 1); s_i32_st_g(gen_addr(c, anc), (int)(1u | (1u << row))); }
}
