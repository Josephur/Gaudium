// gaudium_pld_scan: prompt-lookup n-gram search on the device, part 1 of 2 (index space NM members, scalar code).
// The sequence S the step ends with is virtual (no tensor is written and read back in one kernel):
//   S[i] = hist_in[i]            i <= cur                (confirmed history; hist_in[cur] = this step's ids[0])
//        = ids[i - cur]          cur < i <= cur + a      (the drafts this step accepted)
//        = chain[i - cur - a - 1] cur + a + 1 <= i <= E  (next step's root tokens[a] and MTP drafts d1..dK), E = cur + a + 1 + K
//  in0 hist_in int32 {Lmax}   in1 cursor int32 {1}   in2 a int32 {1}   in3 ids int32 {Tv}   in4 chain int32 {K+1}
//  out0 part  int32 {4, NM}: part[n-1][m] = largest p in member m's slice of [n-1, E-1] with S[p-n+1..p] == S[E-n+1..E]
//             (n = 1, 2, 3, 4; -1 if none). The continuation of a match is S[p+1..]: gaudium_pld_tail picks it.
#define LDH(i) s_i32_ld_g(gen_addr(((int5){(i), 0, 0, 0, 0}), hist_in))
#define SEQ(i) ((i) <= cur ? LDH(i) : ((i) <= cur + a ? s_i32_ld_g(gen_addr(((int5){(i) - cur, 0, 0, 0, 0}), ids)) \
                                                       : s_i32_ld_g(gen_addr(((int5){(i) - cur - a - 1, 0, 0, 0, 0}), chain))))
void main(tensor hist_in, tensor cursor, tensor a_in, tensor ids, tensor chain, tensor part)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 z = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(z, cursor));
    const int a = s_i32_ld_g(gen_addr(z, a_in));
    const int K = get_dim_size(chain, 0) - 1;
    const int NM = get_dim_size(part, 1);
    const int E = cur + a + 1 + K;
    const int k1 = SEQ(E), k2 = SEQ(E - 1), k3 = SEQ(E - 2), k4 = SEQ(E - 3);
    const int span = E - 1;                       // candidate p in [1, E-1]
    const int chunk = (span + NM - 1) / NM;
    for (int m = idx[0]; m < idx[0] + isz[0]; ++m) {
        int lo = 1 + m * chunk, hi = 1 + (m + 1) * chunk;   // [lo, hi)
        if (hi > E) hi = E;
        int b1 = -1, b2 = -1, b3 = -1, b4 = -1;
        for (int p = hi - 1; p >= lo; --p) {
            if (SEQ(p) != k1) continue;
            if (b1 < 0) b1 = p;
            if (SEQ(p - 1) != k2) continue;
            if (b2 < 0) b2 = p;
            if (p < 2 || SEQ(p - 2) != k3) continue;
            if (b3 < 0) b3 = p;
            if (p < 3 || SEQ(p - 3) != k4) continue;
            b4 = p; break;                        // descending: the first 4-gram match also set b1..b3 (same or later p)
        }
        int5 o = {0, m, 0, 0, 0};
        o[0] = 0; s_i32_st_g(gen_addr(o, part), b1);
        o[0] = 1; s_i32_st_g(gen_addr(o, part), b2);
        o[0] = 2; s_i32_st_g(gen_addr(o, part), b3);
        o[0] = 3; s_i32_st_g(gen_addr(o, part), b4);
    }
}
