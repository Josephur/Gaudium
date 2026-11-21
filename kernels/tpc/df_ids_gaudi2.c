// gaudium_df_ids: the DFlash2 block input ids: out {T} = [tokens[0, a], mask, ..., mask]. scalar: mask id. One TPC.
void main(tensor tokens, tensor a_in, tensor out, int mask)
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int T = get_dim_size(out, 0);
    ct[1] = s_i32_ld_g(gen_addr(c, a_in));
    const int bonus = s_i32_ld_g(gen_addr(ct, tokens));
    for (int j = 0; j < T; ++j) { c[0] = j; s_i32_st_g(gen_addr(c, out), j == 0 ? bonus : mask); }
}
