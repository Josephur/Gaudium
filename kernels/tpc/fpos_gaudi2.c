// gaudium_fpos: slot -> absolute position tables of the fresh window for the MTP passes of one engine step (one TPC, scalar).
//  Window layout (NF slots, see gaudium_attn_prep -DFRESH): slots 0..K = the catch-up rows (positions cursor + j),
//  slot K+1 = a permanent zero row (position cursor + K + 1: what the versioned-copy engine attended to when all K drafts
//  were accepted, an unwritten row of its cache), slots K+2 .. K+Kd = recursive drafts 2..Kd at positions cursor + a + i
//  (draft i, slot K+i, position cursor_out + i - 1).
//  in0 cursor int32 {1}   in1 a int32 {1} (accepted drafts of this step, from gaudium_accept_e)
//  out0 fpos_c int32 {NF}  catch-up view: slots 0..K live, everything else DEAD (the catch-up's own causal mask comes from
//                          the query position: slot j is valid for row t iff j <= t)
//  out1 fpos_d int32 {NF}  draft view: catch-up slots 0..min(a+1, K+1) live (slot a+1 is the row at the new cursor; for a = K it is
//                          the zero row), draft slots K+2..K+Kd live at cursor + a + 1 + (slot - K - 1); everything else DEAD
//  scalars: K, Kd, NF.   DEAD = 0x3fffffff (never < any query position).
void main(tensor cursor, tensor a_in, tensor fpos_c, tensor fpos_d, int K, int Kd, int NF)
{
    int5 c = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(c, cursor));
    const int a = s_i32_ld_g(gen_addr(c, a_in));
    const int DEAD = 0x3fffffff;
    const int live_c = (a + 1 < K + 1) ? a + 1 : K + 1;
    for (int j = 0; j < NF; ++j) {
        int pc = DEAD, pd = DEAD;
        if (j <= K) pc = cur + j;
        if (j <= live_c) pd = cur + j;
        else if (j >= K + 2 && j <= K + Kd) pd = cur + a + 1 + (j - K - 1);
        c[0] = j;
        s_i32_st_g(gen_addr(c, fpos_c), pc);
        s_i32_st_g(gen_addr(c, fpos_d), pd);
    }
}
