// gaudium_accept_t: speculative acceptance on the device for a draft TREE (chain + single-node leaves), one TPC, scalar.
//  Rows: 0 = root (last emitted token), 1..S = the chain (row j has parent j-1, depth j), rows > S = leaves (parent[t] = chain
//  row depth[t]-1, depth[t] = parent depth + 1). The accepted path is the longest chain prefix; when the chain breaks at depth
//  d+1, a leaf of depth d+1 whose id equals tokens[chain row d] extends the path by one (first such leaf wins).
//  in0 ids {N}  in1 tokens {1, N} (verify argmax per row)  in2 cursor {1}  in3 parent {N}  in4 depth {N}
//  out0 a {1}: row of the last accepted node (checkpoint slot to resume from, bonus = tokens[a])
//  out1 cursor_out {1} = cursor + accepted depth + 1   out2 emitted {N}: accepted draft ids in path order, tokens[a], then -1 (entry N - 1 = -2 when a leaf was accepted)
//  out3 fix {2} = (cursor + leaf row, cursor + leaf depth) when a leaf was accepted, else (-1, -1): the next step's attention
//       prep and context kernels move that KV row to its chain slot before writing new rows.
void main(tensor ids, tensor tokens, tensor cursor, tensor parent, tensor depth, tensor a_out, tensor cursor_out, tensor emitted, tensor fix)
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int N = get_dim_size(ids, 0);
    const int cur = s_i32_ld_g(gen_addr(c, cursor));
    int S = 0;   // chain length = number of rows t >= 1 with parent[t] == t - 1 in sequence
    for (int t = 1; t < N; ++t) { c[0] = t; if (s_i32_ld_g(gen_addr(c, parent)) == t - 1) S = t; else break; }
    int d = 0, last = 0, ok = 1;
    for (int j = 1; j <= S; ++j) {
        ct[1] = j - 1; c[0] = j;
        ok = ok & (s_i32_ld_g(gen_addr(ct, tokens)) == s_i32_ld_g(gen_addr(c, ids)));
        if (ok) { d = j; last = j; }
    }
    int fsrc = -1, fdst = -1;
    if (d < S) {   // chain broke at depth d + 1: try the leaves of that depth
        ct[1] = d; const int want = s_i32_ld_g(gen_addr(ct, tokens));
        for (int t = S + 1; t < N; ++t) {
            c[0] = t;
            if (s_i32_ld_g(gen_addr(c, depth)) == d + 1 && s_i32_ld_g(gen_addr(c, ids)) == want) {
                last = t; fsrc = cur + t; fdst = cur + d + 1; d = d + 1; break;
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        int v = -1;
        if (i < d) { c[0] = (i + 1 <= S && i + 1 < d) ? i + 1 : ((i + 1 == d) ? last : i + 1); v = s_i32_ld_g(gen_addr(c, ids)); }
        else if (i == d) { ct[1] = last; v = s_i32_ld_g(gen_addr(ct, tokens)); }
        c[0] = i; s_i32_st_g(gen_addr(c, emitted), v);
    }
    if (fsrc >= 0) { c[0] = N - 1; s_i32_st_g(gen_addr(c, emitted), -2); }   // leaf accept marker (entry N - 1 is past the terminator: d + 1 <= S + 1 < N); the mixed policy's host statistics
    c[0] = 0;
    s_i32_st_g(gen_addr(c, a_out), last);
    s_i32_st_g(gen_addr(c, cursor_out), cur + d + 1);
    s_i32_st_g(gen_addr(c, fix), fsrc); c[0] = 1; s_i32_st_g(gen_addr(c, fix), fdst);
}
