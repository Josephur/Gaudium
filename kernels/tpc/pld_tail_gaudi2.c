// gaudium_pld_tail: prompt-lookup rows, part 2 of 2 (one member). Picks the longest n-gram (4, then 3, 2, 1) whose most
// recent earlier occurrence ends at p (max over gaudium_pld_scan's members), appends L continuation tokens S[p+1..p+L]
// (periodic beyond E: a repeat continues itself) to the MTP chain as the next step's extra verify rows, and advances the
// device token history. S, cur, a, E as in pld_scan_gaudi2.c. Any tail content is lossless: the verify accepts a row only
// when it equals the target's own argmax; no match -> the rows repeat dK (cheap, harmless).
// Vision: the history holds -1 at image-pad slots (written by the host from the prompt), so no n-gram can match
// inside an image span; a continuation that reaches a slot < 0 is cut there (that row and every later one become dK): an image pad
// (or the sentinel) is never proposed and never gathered as an embedding id.
//  in0 part int32 {4, NM}   in1 hist_in {Lmax}   in2 cursor {1}   in3 a {1}   in4 ids {Tv}   in5 chain {K+1}
//  out0 hist_out int32 {Lmax}: hist_in[0..cur], then S[cur+1..cur+a+1] (= accepted drafts and tokens[a] at cursor')
//  out1 next_ids int32 {K+1+L} = chain ++ tail
#define LDH(i) s_i32_ld_g(gen_addr(((int5){(i), 0, 0, 0, 0}), hist_in))
#define SEQ(i) ((i) <= cur ? LDH(i) : ((i) <= cur + a ? s_i32_ld_g(gen_addr(((int5){(i) - cur, 0, 0, 0, 0}), ids)) \
                                                       : s_i32_ld_g(gen_addr(((int5){(i) - cur - a - 1, 0, 0, 0, 0}), chain))))
#ifdef GATE
//  -DGATE: out2 a_next int32 {2} = [a, valid rows of the next step]: K+1+L when a match of n >= 2 exists, else K+1 (the next
//  step's DeltaNet kernels skip the invalid rows and its accept kernel never accepts past them)
void main(tensor part, tensor hist_in, tensor cursor, tensor a_in, tensor ids, tensor chain, tensor hist_out, tensor next_ids, tensor a_next)
#else
void main(tensor part, tensor hist_in, tensor cursor, tensor a_in, tensor ids, tensor chain, tensor hist_out, tensor next_ids)
#endif
{
    int5 z = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(z, cursor));
    const int a = s_i32_ld_g(gen_addr(z, a_in));
    const int K = get_dim_size(chain, 0) - 1;
    const int W = get_dim_size(next_ids, 0), L = W - K - 1;
    const int NM = get_dim_size(part, 1);
    const int E = cur + a + 1 + K;
    int p = -1, nbest = 0;
    for (int n = 3; n >= 0 && p < 0; --n) {
        for (int m = 0; m < NM; ++m) { int5 o = {n, m, 0, 0, 0}; const int v = s_i32_ld_g(gen_addr(o, part)); if (v > p) p = v; }
        if (p >= 0) nbest = n + 1;
    }
#ifdef GATE
    { int5 o = {0, 0, 0, 0, 0}; s_i32_st_g(gen_addr(o, a_next), a); o[0] = 1; s_i32_st_g(gen_addr(o, a_next), nbest >= 2 ? W : K + 1); }
#endif
    int5 c = {0, 0, 0, 0, 0};
    for (int j = 0; j <= K; ++j) { c[0] = j; const int v = s_i32_ld_g(gen_addr(c, chain)); s_i32_st_g(gen_addr(c, next_ids), v); }
    const int dK = SEQ(E);
    int cut = 0;
    for (int i = 0; i < L; ++i) {
        int v = dK;
        if (p >= 0) { int q = p + 1 + i; while (q > E) q -= E - p; v = SEQ(q); }
        cut = cut | (v < 0);
        if (cut) v = dK;
        c[0] = K + 1 + i; s_i32_st_g(gen_addr(c, next_ids), v);
    }
    // history: vector copy of the full 64-element blocks below cur+1, scalar copy of the rest, then the new tokens
    const int B = ((cur + 1) / 64) * 64;
    for (int i = 0; i < B; i += 64) { int5 vc = {i, 0, 0, 0, 0}; int64 v = v_i32_ld_tnsr_b(vc, hist_in); v_i32_st_tnsr(vc, hist_out, v); }
    for (int i = B; i <= cur; ++i) { c[0] = i; s_i32_st_g(gen_addr(c, hist_out), LDH(i)); }
    for (int i = cur + 1; i <= cur + a + 1; ++i) { c[0] = i; s_i32_st_g(gen_addr(c, hist_out), SEQ(i)); }
}
