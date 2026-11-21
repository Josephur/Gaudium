// gaudium_next_ids: assemble the next verify step's ids on the device.
//  in0 tokens     int32 {1, K+1}  verify argmaxes      in1 a int32 {1}
//  in2 tokens_mtp int32 {1, K+1}  MTP catch-up argmaxes (draft 1 for the next step = tokens_mtp[a])
//  in3 drafts_rec int32 {1, K-1}  recursive drafts d2..dK (any {1, >=1} tensor when K == 1)
//  out0 next_ids  int32 {K+1} = [tokens[a], tokens_mtp[a], drafts_rec[0..K-2]]
void main(tensor tokens, tensor a_in, tensor tokens_mtp, tensor drafts_rec, tensor next_ids)
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int K = get_dim_size(next_ids, 0) - 1;
    const int a = s_i32_ld_g(gen_addr(c, a_in));
    ct[1] = a;
    const int t0 = s_i32_ld_g(gen_addr(ct, tokens));
    const int d1 = s_i32_ld_g(gen_addr(ct, tokens_mtp));
    c[0] = 0; s_i32_st_g(gen_addr(c, next_ids), t0);
    if (K >= 1) { c[0] = 1; s_i32_st_g(gen_addr(c, next_ids), d1); }
    for (int j = 0; j + 2 <= K; ++j) {
        ct[1] = j;
        const int v = s_i32_ld_g(gen_addr(ct, drafts_rec));
        c[0] = j + 2; s_i32_st_g(gen_addr(c, next_ids), v);
    }
}
