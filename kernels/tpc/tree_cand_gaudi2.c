// gaudium_tree_cand: the candidate table for the NEXT step's tree, from this step's MTP passes (qwen35 drafting).
//  in0 tokens {1, N}   verify argmax per row          in1 a {1} row of the last accepted node     in2 dep {1} accepted depth d
//  in3 c_tok {1, Tc}   MTP catch-up argmax per row    in4 c_top {KT, Tc} catch-up top-KT ids (descending, ties to the lower id)
//  in5 d_tok {1, M}    recursive draft argmaxes d2..d(Kd)   in6 d_top {KT, M} their top-KT ids   (M = max(1, Kd - 1))
//  out0 cand {KT, CM}: column 0 = (tokens[a], 0, ...); column 1 = spine c_tok[d], then the c_top[.][d] entries != spine;
//      column j = 2..Kd: spine d_tok[j-2], then d_top[.][j-2] != spine; columns > Kd: 0 (a placeholder id: any draft is lossless).
//  The spine (rank 0) is the argmax node's output, bit-identical to the chain engine's drafts.   scalar Kd.   one TPC
//  in7 c_prob {KT, Tc}, in8 d_prob {KT, M}: the drafter's softmax probabilities of the top-KT ids -> out1 cprob {KT, CM} f32, aligned
//  with cand (the spine's probability, then its alternatives'; 0 where cand is a placeholder): the input of gaudium_tree_plan.
void main(tensor tokens, tensor a_in, tensor dep_in, tensor c_tok, tensor c_top, tensor d_tok, tensor d_top, tensor c_prob, tensor d_prob, tensor cand, tensor cprob, int Kd)
{
    int5 c = {0, 0, 0, 0, 0};
    const int KT = get_dim_size(cand, 0), CM = get_dim_size(cand, 1);
    const int a = s_i32_ld_g(gen_addr(c, a_in)), d = s_i32_ld_g(gen_addr(c, dep_in));
    for (int j = 0; j < CM; ++j) {
        int spine = 0;
        if (j == 0) { int5 ct = {0, a, 0, 0, 0}; spine = s_i32_ld_g(gen_addr(ct, tokens)); }
        else if (j == 1) { int5 ct = {0, d, 0, 0, 0}; spine = s_i32_ld_g(gen_addr(ct, c_tok)); }
        else if (j <= Kd) { int5 ct = {0, j - 2, 0, 0, 0}; spine = s_i32_ld_g(gen_addr(ct, d_tok)); }
        int5 oc = {0, j, 0, 0, 0};
        s_i32_st_g(gen_addr(oc, cand), spine);
        float psp = (j == 0) ? 1.0f : 0.0f;
        int r = 1;
        if (j >= 1 && j <= Kd) {
            float p0 = 0.0f;
            for (int q = 0; q < KT; ++q) {
                int v; float pv;
                if (j == 1) { int5 tc = {q, d, 0, 0, 0}; v = s_i32_ld_g(gen_addr(tc, c_top)); pv = s_f32_ld_g(gen_addr(tc, c_prob)); }
                else { int5 tc = {q, j - 2, 0, 0, 0}; v = s_i32_ld_g(gen_addr(tc, d_top)); pv = s_f32_ld_g(gen_addr(tc, d_prob)); }
                if (q == 0) p0 = pv;
                if (v == spine) psp = pv;
                else if (r < KT) { oc[0] = r; s_i32_st_g(gen_addr(oc, cand), v); s_f32_st_g(gen_addr(oc, cprob), pv); ++r; }
            }
            if (psp == 0.0f) psp = p0;   // spine not among the top-KT ids (an argmax / top-k tie-rule mismatch): its probability is the top one
        }
        oc[0] = 0; s_f32_st_g(gen_addr(oc, cprob), psp);
        for (; r < KT; ++r) { oc[0] = r; s_i32_st_g(gen_addr(oc, cand), 0); s_f32_st_g(gen_addr(oc, cprob), 0.0f); }
    }
}
