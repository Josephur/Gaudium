// gaudium_tree_ids: the verify ids of a static tree shape from the candidate table the previous step wrote.
//  in0 cand int32 {KT, CM}: cand[r][j] = rank-r candidate at depth j (cand[0][0] = the bonus token = root; rank 0 = the spine draft)
//  in1 depth int32 {N}, in2 rank int32 {N}: the node tables of the shape
//  out0 ids int32 {N}: ids[t] = cand[rank[t]][depth[t]]       one TPC, scalar
void main(tensor cand, tensor depth, tensor rank, tensor ids)
{
    int5 c = {0, 0, 0, 0, 0};
    const int N = get_dim_size(ids, 0);
    for (int t = 0; t < N; ++t) {
        c[0] = t;
        const int j = s_i32_ld_g(gen_addr(c, depth)), r = s_i32_ld_g(gen_addr(c, rank));
        int5 cc = {r, j, 0, 0, 0};
        const int v = s_i32_ld_g(gen_addr(cc, cand));
        s_i32_st_g(gen_addr(c, ids), v);
    }
}
