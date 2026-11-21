// gaudium_df_next: next verify ids {N} = [tokens[0, a], drafts[0..S-1], alts[0..L-1]] with L = N - 1 - S (tree leaves, rank 2;
// for N - 1 - S > S the rest come from alts2: rank-3 leaves). One TPC. scalar Sc: the chain length when it is shorter than the drafted
// block (feature/dspine: a DSpine tree verifies Sc <= 15 chain drafts + leaves); 0 = the whole block (DFlash2).
void main(tensor tokens, tensor a_in, tensor drafts, tensor alts, tensor alts2, tensor next_ids, int Sc)
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int N = get_dim_size(next_ids, 0), S = Sc > 0 ? Sc : get_dim_size(drafts, 0);
    ct[1] = s_i32_ld_g(gen_addr(c, a_in));
    const int t0 = s_i32_ld_g(gen_addr(ct, tokens));
    c[0] = 0; s_i32_st_g(gen_addr(c, next_ids), t0);
    for (int j = 0; j < S && 1 + j < N; ++j) { c[0] = j; const int v = s_i32_ld_g(gen_addr(c, drafts)); c[0] = 1 + j; s_i32_st_g(gen_addr(c, next_ids), v); }
    for (int j = 0; 1 + S + j < N; ++j) {
        c[0] = j % S; const int v = (j < S) ? s_i32_ld_g(gen_addr(c, alts)) : s_i32_ld_g(gen_addr(c, alts2));
        c[0] = 1 + S + j; s_i32_st_g(gen_addr(c, next_ids), v);
    }
}
