// gaudium_sel: out[j] = tok[0, idx[0] + j] for j < M (M = size of out). Picks row `a` of an argmax output {1, N} as a
// {1} ids tensor for the next MTP pass (idx = a), or flattens {1, N} -> {N} (idx = zero, M = N). Scalar, one TPC.
void main(tensor tok, tensor idx, tensor out)
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int M = get_dim_size(out, 0);
    const int base = s_i32_ld_g(gen_addr(c, idx));
    for (int j = 0; j < M; ++j) {
        ct[1] = base + j;
        const int v = s_i32_ld_g(gen_addr(ct, tok));
        c[0] = j; s_i32_st_g(gen_addr(c, out), v);
    }
}
