// gaudium_accept_e: speculative-decode acceptance decided on the device, engine version (one TPC, scalar code).
//  in0 ids     int32 {K+1}     the verify recipe's input ids: ids[0] = last emitted token, ids[1..K] = the drafts
//  in1 tokens  int32 {1, K+1}  verify argmax per row
//  in2 cursor  int32 {1}       this step's cursor (position of ids[0])
//  out0 a          int32 {1}    accepted draft count a in [0, K]: longest prefix with tokens[i] == ids[i+1]
//  out1 cursor_out int32 {1}    cursor + a + 1 (the next step's cursor, and the base position of the recursive drafts)
//  out2 emitted    int32 {K+1}  ids[1..a], tokens[a], then -1 (what the host reads from the token ring)
//  out3 ncommit    int32 {1}    a + 1: rows of this step's catch-up window the NEXT step commits to the MTP layer's big KV cache
//                               (gaudium_kv_commit; the prefill phase writes 0 for the first step)
#ifdef NV
// -DNV (gated prompt-lookup step): in3 a_in int32 {2} = [previous a, valid rows nv]; rows >= nv were not computed, so at most
// nv-1 drafts can be accepted (the bonus token comes from row a < nv)
void main(tensor ids, tensor tokens, tensor cursor, tensor a_in, tensor a_out, tensor cursor_out, tensor emitted, tensor ncommit)
#else
void main(tensor ids, tensor tokens, tensor cursor, tensor a_out, tensor cursor_out, tensor emitted, tensor ncommit)
#endif
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int K = get_dim_size(ids, 0) - 1;
#ifdef NV
    int5 c1 = {1, 0, 0, 0, 0}; const int nv = s_i32_ld_g(gen_addr(c1, a_in)); const int Ka = nv - 1 < K ? nv - 1 : K;
#else
    const int Ka = K;
#endif
    const int cur = s_i32_ld_g(gen_addr(c, cursor));
    int a = 0, ok = 1;
    for (int i = 0; i < Ka; ++i) {
        ct[1] = i; c[0] = i + 1;
        const int t = s_i32_ld_g(gen_addr(ct, tokens));
        const int d = s_i32_ld_g(gen_addr(c, ids));
        ok = ok & (t == d);
        a += ok;
    }
    for (int i = 0; i <= K; ++i) {
        int v = -1;
        if (i < a) { c[0] = i + 1; v = s_i32_ld_g(gen_addr(c, ids)); }
        else if (i == a) { ct[1] = a; v = s_i32_ld_g(gen_addr(ct, tokens)); }
        c[0] = i; s_i32_st_g(gen_addr(c, emitted), v);
    }
    c[0] = 0;
    s_i32_st_g(gen_addr(c, a_out), a);
    s_i32_st_g(gen_addr(c, cursor_out), cur + a + 1);
    s_i32_st_g(gen_addr(c, ncommit), a + 1);
}
